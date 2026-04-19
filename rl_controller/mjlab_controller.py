# --------------------------------------------------------
# MjLab Walking Policy Controller for Go2 Hardware Deployment
#
# Loads a walking policy trained in unitree_rl_mjlab (MuJoCo)
# and deploys it to the real Go2 robot.
#
# Key differences from the Isaac Gym controller (rl_controller.py):
#   - 47D observations (no linear vel, has phase signal)
#   - EmpiricalNormalization from training checkpoint
#   - Different default DOF positions
#   - MuJoCo joint order: FL, FR, RL, RR
#   - Network: [512, 256, 128] with ELU activation
#   - Action scale: 0.25 with default offset
# --------------------------------------------------------

import os
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

# ── Joint-order definitions ──────────────────────────────────────────
# Hardware Go2 output order (wrapper.order): FR, FL, BR, BL
# MuJoCo order (used by MjLab training):    FL, FR, RL, RR
# In the wrapper's naming: BL=RL, BR=RR, so MuJoCo = FL, FR, BL, BR
MUJOCO_ORDER = ["FL", "FR", "BL", "BR"]

# MjLab default joint positions in MuJoCo order (FL, FR, RL, RR):
MJLAB_DEFAULT_DOF_POS = np.array([
    -0.1, 0.9, -1.8,   # FL: hip, thigh, calf
     0.1, 0.9, -1.8,   # FR
    -0.1, 0.9, -1.8,   # RL (BL)
     0.1, 0.9, -1.8,   # RR (BR)
], dtype=np.float32)

MJLAB_ACTION_SCALE = 0.25


class EmpiricalNormalization(nn.Module):
    """Observation normalization using saved running statistics from training."""

    def __init__(self, obs_dim: int):
        super().__init__()
        self.register_buffer('_mean', torch.zeros(1, obs_dim))
        self.register_buffer('_var', torch.ones(1, obs_dim))
        self.register_buffer('_std', torch.ones(1, obs_dim))
        self.register_buffer('count', torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + 1e-8)


class MjLabActorNet(nn.Module):
    """Standalone actor network matching MjLab's MLPModel architecture.

    Architecture:
        EmpiricalNormalization(47)
        Linear(47, 512) -> ELU
        Linear(512, 256) -> ELU
        Linear(256, 128) -> ELU
        Linear(128, 12)  (action mean)
    """

    def __init__(self, obs_dim: int = 47, action_dim: int = 12,
                 hidden_dims: tuple = (512, 256, 128)):
        super().__init__()
        self.obs_normalizer = EmpiricalNormalization(obs_dim)

        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.mlp = nn.Sequential(*layers)

        self.register_buffer('std_param', torch.ones(action_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic forward pass: normalized obs -> action mean."""
        x = self.obs_normalizer(obs)
        return self.mlp(x)


def load_mjlab_actor(checkpoint_path: str, device: str = 'cpu',
                     obs_dim: int = 47, action_dim: int = 12,
                     hidden_dims: tuple = (512, 256, 128)) -> MjLabActorNet:
    """Load MjLab actor weights from checkpoint into standalone network."""
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # Older torch (<1.13) doesn't support weights_only kwarg
        ckpt = torch.load(checkpoint_path, map_location=device)
    actor_sd = ckpt['actor_state_dict']

    net = MjLabActorNet(obs_dim, action_dim, hidden_dims).to(device)

    # Map checkpoint keys to our network
    state_dict = {}
    for key in ['obs_normalizer._mean', 'obs_normalizer._var',
                'obs_normalizer._std', 'obs_normalizer.count']:
        state_dict[key] = actor_sd[key]
    for key in actor_sd:
        if key.startswith('mlp.'):
            state_dict[key] = actor_sd[key]
    if 'distribution.std_param' in actor_sd:
        state_dict['std_param'] = actor_sd['distribution.std_param']

    net.load_state_dict(state_dict)
    net.eval()
    return net


class MjLabRLController:
    """MjLab-trained walking policy controller for Go2 hardware.

    Observation (47D, MuJoCo joint order):
        [0:3]   base angular velocity (body frame)
        [3:6]   projected gravity (body frame)
        [6:9]   command velocity [vx, vy, wz]
        [9:11]  gait phase [sin, cos] (period=0.6s)
        [11:23] joint position relative to default (12 joints)
        [23:35] joint velocity (12 joints)
        [35:47] last action (network output, clipped [-1,1])

    Action (12D): position targets in MuJoCo order
        target = default_pos + action * 0.25
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu',
                 phase_period: float = 0.6):
        self.device = device
        self.phase_period = phase_period

        print(f"Loading MjLab walking policy from: {checkpoint_path}")
        self.net = load_mjlab_actor(checkpoint_path, device)
        print("-> Done")

        self._last_action = np.zeros(12, dtype=np.float32)
        self._start_time = None
        # Diagnostic cache (populated by get_action() each step)
        self._last_obs = np.zeros(47, dtype=np.float32)
        self._last_raw_action = np.zeros(12, dtype=np.float32)
        self._last_target = MJLAB_DEFAULT_DOF_POS.copy()

    def reset(self):
        """Reset internal state. Call when starting a new walking session."""
        self._last_action = np.zeros(12, dtype=np.float32)
        self._start_time = time.time()
        self._last_obs = np.zeros(47, dtype=np.float32)
        self._last_raw_action = np.zeros(12, dtype=np.float32)
        self._last_target = MJLAB_DEFAULT_DOF_POS.copy()

    def build_observation(self, wrapper, command=(0, 0, 0)):
        """Build 47D MjLab observation from real robot state.

        Args:
            wrapper: Wrapper instance providing robot state and IMU data.
            command: (vx, vy, wz) velocity command.

        Returns:
            torch.Tensor of shape (47,)
        """
        state = wrapper.state
        obs = np.zeros(47, dtype=np.float32)

        # 1. Base angular velocity (body frame) [0:3]
        #    wrapper.state[5:8] = gyroscope [wx, wy, wz]
        obs[0:3] = state[5:8]

        # 2. Projected gravity (body frame) [3:6]
        #    Use quaternion from IMU for accurate gravity projection
        quat = wrapper.msgs[0].imu_state.quaternion  # [w, x, y, z]
        quat_xyzw = (quat[1], quat[2], quat[3], quat[0])
        rotmat = Rotation.from_quat(quat_xyzw).as_matrix()
        projected_gravity = np.linalg.inv(rotmat) @ np.array([0, 0, -1])
        obs[3:6] = projected_gravity

        # 3. Command velocity [6:9]
        obs[6:9] = command[:3]

        # 4. Gait phase [9:11] (sin/cos, period=0.6s)
        if self._start_time is None:
            self._start_time = time.time()
        t = time.time() - self._start_time
        phase = (t % self.phase_period) / self.phase_period
        cmd_norm = np.linalg.norm(command)
        if cmd_norm < 0.1:
            obs[9:11] = 0.0  # stand still -> zero phase
        else:
            obs[9] = np.sin(phase * 2.0 * np.pi)
            obs[10] = np.cos(phase * 2.0 * np.pi)

        # 5. Joint positions relative to default [11:23]
        #    Remap from hardware order (FR, FL, BR, BL) to MuJoCo (FL, FR, BL, BR)
        joint_pos_hw = state[8:20]
        joint_pos_mj = wrapper.map(joint_pos_hw, wrapper.order, MUJOCO_ORDER)
        obs[11:23] = np.array(joint_pos_mj) - MJLAB_DEFAULT_DOF_POS

        # 6. Joint velocities [23:35]
        joint_vel_hw = state[20:32]
        joint_vel_mj = wrapper.map(joint_vel_hw, wrapper.order, MUJOCO_ORDER)
        obs[23:35] = joint_vel_mj

        # 7. Last action (network output space) [35:47]
        obs[35:47] = self._last_action

        return torch.FloatTensor(obs)

    def get_action(self, wrapper, command=(0, 0, 0)):
        """Run policy and return absolute joint position targets in MuJoCo order.

        Args:
            wrapper: Wrapper instance.
            command: (vx, vy, wz) velocity command.

        Returns:
            np.ndarray of shape (12,) - joint targets in MuJoCo order (FL,FR,BL,BR)
        """
        obs = self.build_observation(wrapper, command)

        with torch.no_grad():
            obs_t = obs.unsqueeze(0).to(self.device)
            action_t = self.net(obs_t)
            raw_action = action_t.squeeze(0).cpu().numpy()

        # Clip and store for next step's observation
        action = np.clip(raw_action, -1.0, 1.0)
        self._last_action = action.copy()

        # Convert to absolute joint targets
        target = MJLAB_DEFAULT_DOF_POS + action * MJLAB_ACTION_SCALE

        # Cache for diagnostics
        self._last_obs = obs.cpu().numpy().copy()
        self._last_raw_action = raw_action.copy()
        self._last_target = target.copy()
        return target

    def test(self):
        """Quick sanity check."""
        try:
            dummy_obs = torch.zeros(1, 47)
            out = self.net(dummy_obs)
            assert out.shape == (1, 12)
            print("MjLab controller test passed!")
            return 1
        except Exception as e:
            print(f"MjLab controller test failed: {e}")
            return 0
