from .algorithms import PPO
from .modules import ActorCritic
import torch
import numpy as np
from .helpers import *
import numpy as np
np.set_printoptions(precision=2)

class Go2RLController():
    """
    The RL trained (unitree_rl_gym) controller for Go2
    The observation is 48D:
        (3) vx, vy, vz,
        (3) wx, wy, wz,
        (3) gx, gy, gz, # projected gravity
        (3) commands (lin_vel_x, lin_vel_y, ang_vel_yaw),
        (12) joint_pos offset, # this might be offset from a stable stance
        (12) joint_vel,
        (12) previous actions

    Code definition in legged_robot env showing how the obs components are assembled
        ```python
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        obs_scales = env_cfg.normalization.obs_scales
        commands_scale = torch.tensor([obs_scales.lin_vel, obs_scales.lin_vel, obs_scales.ang_vel], device=device, requires_grad=False,)
        observation = torch.cat((
            base_lin_vel * obs_scales.lin_vel,
            base_ang_vel  * obs_scales.ang_vel,
            projected_gravity,
            commands[:, :3] * commands_scale,
            (dof_pos - default_dof_pos) * obs_scales.dof_pos,
            dof_vel * obs_scales.dof_vel,
            actions),dim=-1)
        clip_obs = env_cfg.normalization.clip_observations
        observation = torch.clip(observation, -clip_obs, clip_obs)
        ```

    Action is position target, 12D
        ```python
        clip_actions = env_cfg.normalization.clip_actions
        actions = torch.clip(actions, -clip_actions, clip_actions).to(device)
        ```

    Extra information able the config files used:
        name: go2
        task_class: LeggedRobot
        env_cfg: GO2RoughCfg() 
        train_cfg: GO2RoughCfgPPO()
        policy_class_name: 'ActorCritic'
        algorithm_class_name: 'PPO'
    """

    def __init__(self, device='cpu'):
        self.device = device
        log_root = "/home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/GO2/go2_control_wrapper/train_result/controller_go2_5"

        train_cfg = GO2RoughCfgPPO()
        env_cfg = GO2RoughCfg()

        self.num_obs = env_cfg.env.num_observations
        self.num_actions = env_cfg.env.num_actions
        self.obs_scales = env_cfg.normalization.obs_scales
        self.clip_obs = env_cfg.normalization.clip_observations
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=device, requires_grad=False,)
        self.clip_actions = env_cfg.normalization.clip_actions
        # self.clip_actions = 10.0 # safety reasons
        self.default_dof_pos = torch.tensor([0.1,  0.8, -1.5, -0.1,  0.8, -1.5,  0.1,  1.0, -1.5, -0.1,  1.0, -1.5], device=device)
        # self.default_dof_pos = torch.tensor([0.1, 0.75, -1.8, -0.1, 0.75, -1.8, 0.1, 0.75, -1.8, -0.1, 0.75, -1.8], device=device)
        # self.default_dof_pos = torch.tensor([0.1, 0.9, -1.9, -0.1, 0.9, -1.9, 0.1, 0.9, -1.9, -0.1, 0.9, -1.9], device=device)
        self.action_scale = env_cfg.control.action_scale

        train_cfg_dict = class_to_dict(train_cfg)
        cfg=train_cfg_dict["runner"]
        alg_cfg = train_cfg_dict["algorithm"]
        policy_cfg = train_cfg_dict["policy"]

        actor_critic_class = eval(cfg["policy_class_name"])
        actor_critic: ActorCritic = actor_critic_class(env_cfg.env.num_observations, env_cfg.env.num_observations, env_cfg.env.num_actions, **policy_cfg).to(self.device)
        alg_class = eval(cfg["algorithm_class_name"])
        alg: PPO = alg_class(actor_critic, device=self.device, **alg_cfg)

        self.model_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
        print(f"Loading model from: {self.model_path}")

        loaded_dict = torch.load(self.model_path, map_location=torch.device(self.device))
        alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        
        self.policy = alg.actor_critic.act_inference
        self.action = torch.tensor(np.zeros(self.num_actions).astype(np.float32))

    def test(self):
        try:
            assert len(self.policy(torch.Tensor(np.zeros(self.num_obs))))==self.num_actions
            print("All good!")
            return 1
        except:
            print("something is wrong")
            return 0

    def get_action(self, state, **kwargs):
        """
        NEED TO REMAP EVERYTHING JOINT RELATED TO NEW ORDER
        PYBULLET USES           "FL", "BL", "FR", "BR"
        ISAAC TRAINING USES     "FL", "FR", "BL", "BR"

        Passed in state follows:
            obs = (
                tuple(robot_body_linear_vel) + 
                tuple(robot_body_angular_vel) +
                tuple(projected_gravity) + 
                tuple(command) +
                tuple(joint_pos) +
                tuple(joint_vel)
            )
        """
        isaac_gym_order = ["FL", "FR", "BL", "BR"]
        pybullet_order = ["FL", "BL", "FR", "BR"]

        observation = torch.cat((
            state[:3] * self.obs_scales.lin_vel,
            state[3:6] * self.obs_scales.ang_vel,
            state[6:9],
            state[9:12] * self.commands_scale,
            (torch.tensor(self.map(state[12:24], pybullet_order, isaac_gym_order)) - self.default_dof_pos) * self.obs_scales.dof_pos,
            torch.tensor(self.map(state[24:36], pybullet_order, isaac_gym_order)) * self.obs_scales.dof_vel,
            self.action
            ), dim=-1
        )

        observation = torch.clip(observation, -self.clip_obs, self.clip_obs)
        
        # action output is [FL, FR, BL, BR]
        action = self.policy(observation)
        self.action = torch.clip(action, -self.clip_actions, self.clip_actions).to(self.device)
        
        # we need to return [FL, BL, FR, BR]
        # target angle = actionScale * action + defaultAngle
        target_angle = self.action_scale * self.action + self.default_dof_pos
        return np.array(self.map(target_angle.detach().numpy(), isaac_gym_order, pybullet_order))

    def map(self, pos, current_order, new_order):
        mapped_pos = [0.0] * 12
        # the input_order is different from the self.order, remap before writing to the robot
        for i, o in enumerate(current_order):
            index = new_order.index(o)
            for j in range(3):
                mapped_pos[index*3 + j] = pos[i*3 + j]
        return mapped_pos

