from typing import Optional
import numpy as np
import os
import torch
from RARL.sac_adv import SAC_adv
from utils.utils import load_config
from RARL.sac_mini import SAC_mini


class SafetyEnforcer:

    def __init__(self,
                 epsilon: float = 0.0,
                 imaginary_horizon: int = 100,
                 shield_type: Optional[str] = "value",
                 parent_dir: Optional[str] = "") -> None:
        """_summary_

        Args:
            epsilon (float, optional): The epsilon value to be used for value shielding, determining the conservativeness of safety enforcer. Defaults to 0.0.
            imaginary_horizon (int, optional): The horizon to be used for rollout-based shielding. Defaults to 100.
            shield_type (Optional[str], optional): The shielding type to be used, choose from ["value", "rollout"]. Defaults to "value".
        """
        #! TODO: Apply rollout-based shielding with the simulator
        if shield_type != "value":
            raise NotImplementedError

        self.epsilon = epsilon
        self.imaginary_horizon = imaginary_horizon

        # training_dir = "train_result/test_go2/test_isaacs_centerSampling"

        # training_dir = "train_result/test_go2/test_isaacs_centerSampling_withContact"
        # load_dict = {"ctrl": 7_400_000, "dstb": 7_500_000}

        # training_dir = "train_result/test_go2/test_isaacs_postCoRL_arbitraryGx"
        # load_dict = {"ctrl": 7_200_000, "dstb": 8_000_001}

        # SMART
        # alternate
        # training_dir = "train_result/smart/go2_isaacs"
        # load_dict = {"ctrl": 2_100_000, "dstb": 2_100_000}

        # tgda
        training_dir = "train_result/smart/go2_tgda"
        load_dict = {"ctrl": 1_600_000, "dstb": 1_600_000}

        model_path = os.path.join(parent_dir, training_dir, "model")
        model_config_path = os.path.join(parent_dir, training_dir,
                                         "config.yaml")

        config_file = os.path.join(parent_dir, model_config_path)

        if not os.path.exists(config_file):
            raise ValueError(
                "Cannot find config file for the model, terminated")

        config = load_config(config_file)
        config_arch = config['arch']
        config_update = config['update']

        self.policy = SAC_adv(config_update, config_arch)
        self.policy.build_network(verbose=True)
        print("Loading frozen weights of model at {} with load_dict {}".format(
            model_path, load_dict))

        self.policy.restore_refactor(None, model_path, load_dict=load_dict)
        print("-> Done")

        self.critic = self.policy.adv_critic
        self.dstb = self.policy.dstb
        self.ctrl = self.policy.ctrl

        self.is_shielded = None
        self.prev_q = None

    def get_action(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        # state = np.concatenate((state[3:8], state[9:]), axis=0)
        assert len(state) == 36
        s_dstb = np.copy(state)
        # s_dstb = np.concatenate((state, action), axis=0)
        dstb = self.dstb(s_dstb)

        critic_q = max(
            self.critic(torch.FloatTensor(state), torch.FloatTensor(action),
                        torch.FloatTensor(dstb))).detach().numpy()

        # positive is good
        if critic_q < self.epsilon:
            action = self.ctrl(state)
            self.is_shielded = True
        else:
            self.is_shielded = False

        self.prev_q = critic_q.reshape(-1)[0]

        return action

    def get_q(self, state: np.ndarray, action: np.ndarray):
        if state is not None and action is not None:
            assert len(state) == 36
            # state = np.concatenate((state[3:8], state[9:]), axis=0)
            s_dstb = np.copy(state)
            # s_dstb = np.concatenate((state, action), axis=0)
            dstb = self.dstb(s_dstb)

            critic_q = max(
                self.critic(torch.FloatTensor(state),
                            torch.FloatTensor(action),
                            torch.FloatTensor(dstb))).detach().numpy()

            self.prev_q = critic_q.reshape(-1)[0]

        return self.prev_q

    def target_margin(self, state):
        """ (36) and 33D state, 32D state omits z
            (x, y), z,
            x_dot, y_dot, z_dot,
            roll, pitch, (yaw)
            w_x, w_y, w_z,
            joint_pos x 12,
            joint_vel x 12
        """
        # this is not the correct target margin, missing corner pos and toe pos, replacing corner pos with height, assuming that toes always touch ground
        # l(x) < 0 --> x \in T
        # state = np.concatenate((state[3:8], state[9:]), axis=0)
        assert len(state) == 36
        return {"roll": 0.2 - abs(state[3]), "pitch": 0.2 - abs(state[4])}

    def get_safety_action(self, state, target=True, threshold=0.0):
        assert len(state) == 36

        stable_stance = np.array([
            -0.5, 0.7, -2.0, 0.5, 0.7, -2.0, -0.5, 0.7, -2.0, -0.5, 0.7, -2.0
        ])

        if not target:
            return self.ctrl(state)
        else:
            # switch between fallback and target stable stance, depending on the current state
            margin = self.target_margin(state)
            lx = min(margin.values())
            current_joint_pos = state[8:20]

            if lx > threshold:  # account for sensor noise
                # in target set, just output stable stance
                #! TODO: enforce stable stance instead of just outputting zero changes to the current stance
                return np.clip(stable_stance - current_joint_pos,
                               -np.ones(12) * 0.5,
                               np.ones(12) * 0.5)
            else:
                return self.ctrl(state)

    def get_shielding_status(self):
        return self.is_shielded
