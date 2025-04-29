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


        training_dir = "train_result/success/"
        load_dict = {"ctrl": 300_000, "dstb": 1_000_000}


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
        assert len(state) == 78
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
            assert len(state) == 78
            s_dstb = np.copy(state)
            # s_dstb = np.concatenate((state, action), axis=0)
            dstb = self.dstb(s_dstb)

            critic_q = max(
                self.critic(torch.FloatTensor(state),
                            torch.FloatTensor(action),
                            torch.FloatTensor(dstb))).detach().numpy()

            self.prev_q = critic_q.reshape(-1)[0]

        return self.prev_q


    def get_shielding_status(self):
        return self.is_shielded
