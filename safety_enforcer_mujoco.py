# --------------------------------------------------------
# MuJoCo-trained ISAACS Safety Enforcer for Go2 Hardware
#
# Loads the new ISAACS safety policy trained in MuJoCo
# (safe_adaptation_dev) and provides value shielding (LRSF)
# and RCBF safety filtering for hardware deployment.
#
# Key differences from old PyBullet-based safety_enforcer.py:
#   - 48D obs for ctrl (36D state + 12D prev_ctrl_action)
#   - 60D obs for dstb (48D + 12D ctrl_action)
#   - Dstb sees ctrl action
#   - Ctrl network: [512, 512, 512] Sin activation
#   - Dstb network: [256, 256, 256] Sin activation
#   - Critic network: [256, 256, 256] Sin activation
#   - Joint order: PyBullet (FL, BL, FR, BR)
# --------------------------------------------------------

import os
import numpy as np
import torch
import yaml

from ISAACS_mujoco.model import GaussianPolicy, TwinnedQNetwork

# PyBullet joint order used by ISAACS training
PYBULLET_ORDER = ["FL", "BL", "FR", "BR"]


class MujocoSafetyEnforcer:
    """Safety enforcer using ISAACS MuJoCo-trained safety policy.

    Provides:
    - LRSF (Least-Restrictive Safety Filter): switch to safety ctrl when V < epsilon
    - RCBF (Robust Q-CBF): find closest feasible action to task action

    Observation layout (48D, augmented with prev_action):
        [0:3]   body linear velocity (body frame)
        [3:5]   roll, pitch
        [5:8]   body angular velocity (body frame)
        [8:20]  joint positions (12, PyBullet order: FL, BL, FR, BR)
        [20:32] joint velocities (12, PyBullet order)
        [32:36] foot contact flags (4)
        [36:48] previous ctrl action (12)
    """

    def __init__(self,
                 config_path: str = "train_result/nature/go2_mujoco_isaacs_v22_long/config.yaml",
                 model_dir: str = None,
                 ctrl_step: int = None,
                 dstb_step: int = None,
                 epsilon: float = 0.0,
                 device: str = 'cpu'):
        """Initialize the MuJoCo safety enforcer.

        Args:
            config_path: Path to ISAACS config YAML.
            model_dir: Path to model directory containing ctrl/, dstb/, central/ subdirs.
                       If None, derives from config's out_folder.
            ctrl_step: Checkpoint step for ctrl/critic. If None, auto-detect highest.
            dstb_step: Checkpoint step for dstb. If None, auto-detect highest.
            epsilon: LRSF threshold. V < epsilon triggers shielding.
            device: torch device.
        """
        self.epsilon = epsilon
        self.device = torch.device(device)

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        arch = config['arch']

        # Determine model directory
        if model_dir is None:
            model_dir = os.path.join(config['solver']['out_folder'], "model")

        # ── Build networks ──
        print("Building ISAACS MuJoCo safety networks...")

        # Ctrl actor: 48D obs -> 12D action
        ctrl_arch = arch['actor_0']
        ctrl_action_range = np.array(ctrl_arch['action_range'], dtype=np.float32)
        self.ctrl_net = GaussianPolicy(
            obsrv_dim=ctrl_arch['obsrv_dim'],
            mlp_dim=ctrl_arch['mlp_dim'],
            action_dim=ctrl_arch['action_dim'],
            action_range=ctrl_action_range,
            activation_type=ctrl_arch['activation'],
            device=device,
            verbose=True
        )

        # Dstb actor: 60D obs (48 + 12 ctrl) -> 6D action
        dstb_arch = arch['actor_1']
        dstb_action_range = np.array(dstb_arch['action_range'], dtype=np.float32)
        self.dstb_net = GaussianPolicy(
            obsrv_dim=dstb_arch['obsrv_dim'],
            mlp_dim=dstb_arch['mlp_dim'],
            action_dim=dstb_arch['action_dim'],
            action_range=dstb_action_range,
            activation_type=dstb_arch['activation'],
            device=device,
            verbose=True
        )

        # Critic: 48D obs + 18D action (12 ctrl + 6 dstb) -> scalar
        critic_arch = arch['critic_0']
        self.critic_net = TwinnedQNetwork(
            obsrv_dim=critic_arch['obsrv_dim'],
            mlp_dim=critic_arch['mlp_dim'],
            action_dim=critic_arch['action_dim'],
            activation_type=critic_arch['activation'],
            device=device,
            verbose=True
        )

        # Check if dstb sees ctrl action
        obsrv_list = config['agent'].get('obsrv_list', {})
        dstb_obsrv = obsrv_list.get('dstb', None)
        self.dstb_sees_ctrl = dstb_obsrv is not None and 'ctrl' in dstb_obsrv

        # ── Load weights ──
        if os.path.exists(model_dir):
            if ctrl_step is None:
                ctrl_step = self._find_latest_step(os.path.join(model_dir, "ctrl"), "ctrl")
            if dstb_step is None:
                dstb_step = self._find_latest_step(os.path.join(model_dir, "dstb"), "dstb")

            print(f"Loading ISAACS models from {model_dir}")
            print(f"  ctrl step: {ctrl_step}, dstb step: {dstb_step}")

            self._load_model(self.ctrl_net, os.path.join(model_dir, "ctrl", f"ctrl-{ctrl_step}.pth"))
            self._load_model(self.dstb_net, os.path.join(model_dir, "dstb", f"dstb-{dstb_step}.pth"))
            self._load_model(self.critic_net, os.path.join(model_dir, "central", f"central-{ctrl_step}.pth"))
            print("-> Done loading models")
        else:
            print(f"WARNING: Model directory {model_dir} not found!")
            print("  Safety enforcer initialized but will use random weights.")
            print("  Copy trained weights before using in hardware deployment.")

        self.ctrl_net.eval()
        self.dstb_net.eval()
        self.critic_net.eval()

        # State tracking
        self.prev_ctrl_action = np.zeros(12, dtype=np.float32)
        self.is_shielded = None
        self.prev_q = None
        self.prev_v_hat = None

    def _find_latest_step(self, dir_path, prefix):
        """Find the latest checkpoint step in a directory."""
        if not os.path.exists(dir_path):
            return 0
        steps = []
        for f in os.listdir(dir_path):
            if f.startswith(prefix) and f.endswith('.pth'):
                try:
                    step = int(f.replace(f"{prefix}-", "").replace(".pth", ""))
                    steps.append(step)
                except ValueError:
                    continue
        return max(steps) if steps else 0

    def _load_model(self, net, path):
        """Load model weights from checkpoint."""
        if os.path.exists(path):
            net.load_state_dict(torch.load(path, map_location=self.device))
            print(f"  Loaded: {path}")
        else:
            print(f"  WARNING: Checkpoint not found: {path}")

    def reset(self):
        """Reset internal state. Call at episode start."""
        self.prev_ctrl_action = np.zeros(12, dtype=np.float32)
        self.is_shielded = None
        self.prev_q = None
        self.prev_v_hat = None

    def build_isaacs_obs(self, wrapper):
        """Build 48D ISAACS observation from robot state.

        Args:
            wrapper: Wrapper instance providing robot state.

        Returns:
            np.ndarray of shape (48,) in PyBullet joint order.
        """
        state = wrapper.state
        obs = np.zeros(48, dtype=np.float32)

        # [0:3] body linear velocity
        obs[0:3] = state[0:3]

        # [3:5] roll, pitch
        obs[3:5] = state[3:5]

        # [5:8] body angular velocity
        obs[5:8] = state[5:8]

        # [8:20] joint positions in PyBullet order (FL, BL, FR, BR)
        joint_pos_hw = state[8:20]
        joint_pos_pb = wrapper.map(joint_pos_hw, wrapper.order, PYBULLET_ORDER)
        obs[8:20] = joint_pos_pb

        # [20:32] joint velocities in PyBullet order
        joint_vel_hw = state[20:32]
        joint_vel_pb = wrapper.map(joint_vel_hw, wrapper.order, PYBULLET_ORDER)
        obs[20:32] = joint_vel_pb

        # [32:36] foot contact flags
        obs[32:36] = state[32:36]

        # [36:48] previous ctrl action
        obs[36:48] = self.prev_ctrl_action

        return obs

    def _estimate_V(self, s_tensor):
        """Estimate V(x) = Q(x, pi_ctrl(x), pi_dstb(x, pi_ctrl(x))).

        Returns (V_hat, u_safe, d_safe).
        """
        with torch.no_grad():
            u_safe = self.ctrl_net(s_tensor)
            if self.dstb_sees_ctrl:
                s_dstb = torch.cat([s_tensor, u_safe], dim=-1)
            else:
                s_dstb = s_tensor
            d_safe = self.dstb_net(s_dstb)
            combined = torch.cat([u_safe, d_safe], dim=-1)
            q1, q2 = self.critic_net(s_tensor, combined)
            V_hat = torch.max(q1, q2).item()

        return V_hat, u_safe, d_safe

    def _eval_robust_q(self, s_tensor, u_ctrl):
        """Evaluate Q(x, u, pi_dstb(x, u)) for a given ctrl action."""
        with torch.no_grad():
            if self.dstb_sees_ctrl:
                s_dstb = torch.cat([s_tensor, u_ctrl], dim=-1)
            else:
                s_dstb = s_tensor
            d = self.dstb_net(s_dstb)
            combined = torch.cat([u_ctrl, d], dim=-1)
            q1, q2 = self.critic_net(s_tensor, combined)
            q_val = torch.max(q1, q2).item()
        return q_val, d

    def get_action(self, wrapper, ctrl_action: np.ndarray) -> np.ndarray:
        """Apply LRSF safety filter.

        Args:
            wrapper: Wrapper instance.
            ctrl_action: Proposed control action (12D, PyBullet order).
                         This is an incremental action, NOT absolute target.

        Returns:
            Filtered action (12D, PyBullet order).
        """
        obs = self.build_isaacs_obs(wrapper)
        s_tensor = torch.FloatTensor(obs).to(self.device)

        V_hat, u_safe, d_safe = self._estimate_V(s_tensor)
        self.prev_v_hat = V_hat

        if V_hat < self.epsilon:
            # Switch to safety controller
            action = u_safe.detach().cpu().numpy().flatten()
            self.is_shielded = True
        else:
            action = ctrl_action
            self.is_shielded = False

        self.prev_q = V_hat
        self.prev_ctrl_action = action.copy()
        return action

    def get_q(self, wrapper, ctrl_action: np.ndarray):
        """Evaluate Q value for a given action without filtering."""
        obs = self.build_isaacs_obs(wrapper)
        s_tensor = torch.FloatTensor(obs).to(self.device)
        u_ctrl = torch.FloatTensor(ctrl_action).to(self.device)
        q_val, _ = self._eval_robust_q(s_tensor, u_ctrl)
        self.prev_q = q_val
        return q_val

    def target_margin(self, wrapper):
        """Compute safety margin based on roll/pitch.

        Returns dict with margin values. Negative = inside target set.
        """
        state = wrapper.state
        return {
            "roll": 0.2 - abs(state[3]),
            "pitch": 0.2 - abs(state[4])
        }

    def rcbf_projected_gradient(self, wrapper, u_task_pb: np.ndarray,
                                 kappa: float = 0.995, tol: float = 1e-3,
                                 max_iters: int = 20, lr: float = 0.05):
        """RCBF safety filter via projected gradient ascent.

        Solves: u(x) = argmin ||u_task - u||^2
                s.t.  Q(x, u, pi_dstb(x,u)) >= kappa * V_hat(x)

        Args:
            wrapper: Wrapper instance.
            u_task_pb: Task policy action (12D, PyBullet order).
            kappa: Barrier preservation factor.
            tol: Convergence tolerance.
            max_iters: Maximum gradient iterations.
            lr: Gradient step size.

        Returns:
            (u_filtered, q_val, n_iters, alpha)
        """
        obs = self.build_isaacs_obs(wrapper)
        s_tensor = torch.FloatTensor(obs).to(self.device)

        V_hat, u_safe, _ = self._estimate_V(s_tensor)
        self.prev_v_hat = V_hat
        threshold = kappa * V_hat

        u_task = torch.FloatTensor(u_task_pb).to(self.device)

        # Check if task action already satisfies constraint
        Q_task, _ = self._eval_robust_q(s_tensor, u_task)
        if Q_task >= threshold:
            self.is_shielded = False
            self.prev_q = Q_task
            self.prev_ctrl_action = u_task_pb.copy()
            return u_task_pb, Q_task, 0, 0.0

        # If safe fallback can't meet threshold, return it as best-effort
        if V_hat < threshold:
            action = u_safe.detach().cpu().numpy().flatten()
            self.is_shielded = True
            self.prev_q = V_hat
            self.prev_ctrl_action = action.copy()
            return action, V_hat, 0, 1.0

        a_min = self.ctrl_net.a_min
        a_max = self.ctrl_net.a_max

        u = u_task.clone().detach()
        best_u = u_safe.clone().detach()
        best_q = V_hat
        best_alpha = 1.0

        for i in range(1, max_iters + 1):
            u_opt = u.detach().requires_grad_(True)

            # Forward pass with gradient
            if self.dstb_sees_ctrl:
                s_dstb = torch.cat([s_tensor, u_opt], dim=-1)
            else:
                s_dstb = s_tensor
            d = self.dstb_net(s_dstb)
            combined = torch.cat([u_opt, d], dim=-1)
            q1, q2 = self.critic_net(s_tensor, combined)
            Q = torch.min(q1, q2)
            q_val = Q.item()

            if q_val >= threshold:
                # Feasible - backtrack toward u_task
                u_lo = u_task.detach()
                u_hi = u_opt.detach()
                for _ in range(10):
                    u_mid = 0.5 * (u_lo + u_hi)
                    q_mid, _ = self._eval_robust_q(s_tensor, u_mid)
                    if q_mid >= threshold:
                        u_hi = u_mid
                        best_u = u_mid
                        best_q = q_mid
                    else:
                        u_lo = u_mid
                best_alpha = self._compute_alpha(u_task, u_safe, best_u)
                action = best_u.detach().cpu().numpy().flatten()
                self.is_shielded = best_alpha > 1e-4
                self.prev_q = best_q
                self.prev_ctrl_action = action.copy()
                return action, best_q, i, best_alpha

            if q_val > best_q:
                best_u = u_opt.detach()
                best_q = q_val
                best_alpha = self._compute_alpha(u_task, u_safe, best_u)

            # Gradient ascent on Q
            grad = torch.autograd.grad(Q, u_opt)[0]
            grad_norm = torch.norm(grad)
            if grad_norm < 1e-8:
                break
            u = u_opt.detach() + lr * grad / grad_norm
            u = torch.clamp(u, a_min, a_max)

        best_alpha = self._compute_alpha(u_task, u_safe, best_u)
        action = best_u.detach().cpu().numpy().flatten()
        self.is_shielded = best_alpha > 1e-4
        self.prev_q = best_q
        self.prev_ctrl_action = action.copy()
        return action, best_q, max_iters, best_alpha

    @staticmethod
    def _compute_alpha(u_task, u_safe, u_filtered):
        """Compute blend coefficient: 0 = pure task, 1 = pure safe."""
        diff = u_safe - u_task
        norm_sq = torch.dot(diff, diff).item()
        if norm_sq < 1e-12:
            return 0.0
        alpha = torch.dot(u_filtered - u_task, diff).item() / norm_sq
        return float(np.clip(alpha, 0.0, 1.0))
