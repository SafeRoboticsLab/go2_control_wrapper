# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

from abc import ABC, abstractmethod
from typing import Optional
import os
import copy
import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.optim import lr_scheduler

from .model import GaussianPolicy, TwinnedQNetwork
from .scheduler import StepLRMargin
from .utils import soft_update, save_model, remove_model, restore_model


class BaseSAC(ABC):

    def __init__(self, CONFIG, CONFIG_ARCH):
        """
    __init__: initialization.

    Args:
        CONFIG (object): update-related hyper-parameter configuration.
        CONFIG_ARCH (object): NN architecture configuration.
    """
        self.CONFIG = CONFIG
        self.CONFIG_ARCH = CONFIG_ARCH
        self.EVAL: bool = CONFIG.EVAL
        self.mode: str = CONFIG.MODE
        self.actor_type: str = CONFIG.ACTOR_TYPE

        # == ENV PARAM ==
        self.action_range = np.array(CONFIG_ARCH.ACTION_RANGE, dtype=float)
        self.action_dim: int = CONFIG_ARCH.ACTION_DIM
        self.obs_dim: int = CONFIG_ARCH.OBS_DIM

        # NN: device, action indicators
        self.device: str = CONFIG.DEVICE
        self.critic_has_act_ind: bool = CONFIG_ARCH.CRITIC_HAS_ACT_IND
        if self.critic_has_act_ind:
            assert hasattr(CONFIG_ARCH, "ACT_IND"), "Needs action indicator!"
        if hasattr(CONFIG_ARCH, "ACT_IND"):
            self.act_ind = torch.FloatTensor(CONFIG_ARCH.ACT_IND).to(
                self.device)
            self.act_ind_dim = self.act_ind.shape[0]

        # == PARAM FOR TRAINING ==
        if not self.EVAL:
            if CONFIG.OPT_TYPE == "AdamW":
                self.opt_cls = AdamW
            elif CONFIG.OPT_TYPE == "AdamW":
                self.opt_cls = Adam
            else:
                raise ValueError("Not supported optimizer type!")
            self.terminal_type: str = CONFIG.TERMINAL_TYPE  # Only for RARL.

            # NN
            self.batch_size: int = CONFIG.BATCH_SIZE
            self.update_period: int = CONFIG.UPDATE_PERIOD

            # Learning Rate
            self.LR_A_SCHEDULE: bool = CONFIG.LR_A_SCHEDULE
            self.LR_C_SCHEDULE: bool = CONFIG.LR_C_SCHEDULE
            if self.LR_A_SCHEDULE:
                self.LR_A_PERIOD: int = CONFIG.LR_A_PERIOD
                self.LR_A_DECAY: float = CONFIG.LR_A_DECAY
                self.LR_A_END: float = CONFIG.LR_A_END
            if self.LR_C_SCHEDULE:
                self.LR_C_PERIOD: int = CONFIG.LR_C_PERIOD
                self.LR_C_DECAY: float = CONFIG.LR_C_DECAY
                self.LR_C_END: float = CONFIG.LR_C_END
            self.LR_C: float = CONFIG.LR_C
            self.LR_A: float = CONFIG.LR_A

            # Discount factor
            self.GAMMA_SCHEDULE: bool = CONFIG.GAMMA_SCHEDULE
            if self.GAMMA_SCHEDULE:
                self.gamma_scheduler = StepLRMargin(init_value=CONFIG.GAMMA,
                                                    period=CONFIG.GAMMA_PERIOD,
                                                    decay=CONFIG.GAMMA_DECAY,
                                                    end_value=CONFIG.GAMMA_END,
                                                    goal_value=1.)
                self.GAMMA: float = self.gamma_scheduler.get_variable()
            else:
                self.GAMMA: float = CONFIG.GAMMA

            # Target Network Update
            self.TAU: float = CONFIG.TAU

            # alpha-related hyper-parameters
            self.init_alpha = torch.log(torch.FloatTensor([CONFIG.ALPHA
                                                           ])).to(self.device)
            self.LEARN_ALPHA: bool = CONFIG.LEARN_ALPHA
            self.log_alpha = self.init_alpha.detach().clone()
            self.target_entropy = torch.tensor(-self.action_dim).to(
                self.device)
            if self.LEARN_ALPHA:
                self.log_alpha.requires_grad = True
                self.LR_Al: float = CONFIG.LR_Al
                self.LR_Al_SCHEDULE: bool = CONFIG.LR_Al_SCHEDULE
                if self.LR_Al_SCHEDULE:
                    self.LR_Al_PERIOD: int = CONFIG.LR_Al_PERIOD
                    self.LR_Al_DECAY: float = CONFIG.LR_Al_DECAY
                    self.LR_Al_END: float = CONFIG.LR_Al_END

    # region: property
    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    @abstractmethod
    def has_latent(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def latent_dist(self):
        raise NotImplementedError

    # endregion

    # region: build models and optimizers
    def build_network(self, verbose=True, actor_path=None, critic_path=None):
        if self.critic_has_act_ind:
            critic_action_dim = self.action_dim + self.act_ind_dim
        else:
            critic_action_dim = self.action_dim

        self.critic = TwinnedQNetwork(
            obs_dim=self.obs_dim,
            mlp_dim=self.CONFIG_ARCH.DIM_LIST['critic'],
            action_dim=critic_action_dim,
            append_dim=self.CONFIG_ARCH.APPEND_DIM,
            latent_dim=self.CONFIG_ARCH.LATENT_DIM,
            activation_type=self.CONFIG_ARCH.ACTIVATION['critic'],
            device=self.device,
            verbose=verbose)
        self.actor = GaussianPolicy(
            obs_dim=self.obs_dim,
            mlp_dim=self.CONFIG_ARCH.DIM_LIST['actor'],
            action_dim=self.action_dim,
            action_range=self.action_range,
            append_dim=self.CONFIG_ARCH.APPEND_DIM,
            latent_dim=self.CONFIG_ARCH.LATENT_DIM,
            activation_type=self.CONFIG_ARCH.ACTIVATION['actor'],
            device=self.device,
            verbose=verbose)

        # Load model if specified
        if critic_path is not None:
            self.critic.load_state_dict(
                torch.load(critic_path, map_location=self.device))
            print("--> Load critic wights from {}".format(critic_path))

        if actor_path is not None:
            self.actor.load_state_dict(
                torch.load(actor_path, map_location=self.device))
            print("--> Load actor wights from {}".format(actor_path))

        # Copy for critic targer
        self.critic_target = copy.deepcopy(self.critic)

    def build_optimizer(self):
        print("Build basic optimizers.")

        self.critic_optimizer = self.opt_cls(self.critic.parameters(),
                                             lr=self.LR_C,
                                             weight_decay=0.01)
        self.actor_optimizer = self.opt_cls(self.actor.parameters(),
                                            lr=self.LR_A,
                                            weight_decay=0.01)

        if self.LR_C_SCHEDULE:
            self.critic_scheduler = lr_scheduler.StepLR(
                self.critic_optimizer,
                step_size=self.LR_C_PERIOD,
                gamma=self.LR_C_DECAY)
        if self.LR_A_SCHEDULE:
            self.actor_scheduler = lr_scheduler.StepLR(
                self.actor_optimizer,
                step_size=self.LR_A_PERIOD,
                gamma=self.LR_A_DECAY)

        if self.LEARN_ALPHA:
            self.log_alpha_optimizer = self.opt_cls([self.log_alpha],
                                                    lr=self.LR_Al,
                                                    weight_decay=0.01)
            if self.LR_Al_SCHEDULE:
                self.log_alpha_scheduler = lr_scheduler.StepLR(
                    self.log_alpha_optimizer,
                    step_size=self.LR_Al_PERIOD,
                    gamma=self.LR_Al_DECAY)

    # endregion

    # region: update functions
    def reset_alpha(self):
        self.log_alpha = self.init_alpha.detach().clone()
        if self.LEARN_ALPHA:
            self.log_alpha.requires_grad = True
            self.log_alpha_optimizer = self.opt_cls([self.log_alpha],
                                                    lr=self.LR_Al,
                                                    weight_decay=0.01)
            if self.LR_Al_SCHEDULE:
                self.log_alpha_scheduler = lr_scheduler.StepLR(
                    self.log_alpha_optimizer,
                    step_size=self.LR_Al_PERIOD,
                    gamma=self.LR_Al_DECAY)

    def update_alpha_hyper_param(self):
        if self.LR_Al_SCHEDULE:
            lr = self.log_alpha_optimizer.state_dict()['param_groups'][0]['lr']
            if lr <= self.LR_Al_END:
                for param_group in self.log_alpha_optimizer.param_groups:
                    param_group['lr'] = self.LR_Al_END
            else:
                self.log_alpha_scheduler.step()

    def update_critic_hyper_param(self):
        if self.LR_C_SCHEDULE:
            lr = self.critic_optimizer.state_dict()['param_groups'][0]['lr']
            if lr <= self.LR_C_END:
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = self.LR_C_END
            else:
                self.critic_scheduler.step()
        if self.GAMMA_SCHEDULE:
            old_gamma = self.gamma_scheduler.get_variable()
            self.gamma_scheduler.step()
            self.GAMMA = self.gamma_scheduler.get_variable()
            if self.GAMMA != old_gamma:
                self.reset_alpha()

    def update_actor_hyper_param(self):
        if self.LR_A_SCHEDULE:
            lr = self.actor_optimizer.state_dict()['param_groups'][0]['lr']
            if lr <= self.LR_A_END:
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = self.LR_A_END
            else:
                self.actor_scheduler.step()

    def update_hyper_param(self):
        self.update_critic_hyper_param()
        self.update_actor_hyper_param()
        if self.LEARN_ALPHA:
            self.update_alpha_hyper_param()

    def update_target_networks(self):
        soft_update(self.critic_target, self.critic, self.TAU)

    @abstractmethod
    def update_actor(self, batch):
        raise NotImplementedError

    @abstractmethod
    def update_critic(self, batch):
        raise NotImplementedError

    @abstractmethod
    def update(self, batch, timer, update_period=2):
        raise NotImplementedError

    # endregion

    # region: utils
    def save(self,
             step: int,
             model_folder: str,
             max_model: Optional[int] = None):
        path_c = os.path.join(model_folder, 'critic')
        path_a = os.path.join(model_folder, 'actor')
        save_model(self.critic, step, path_c, 'critic', max_model)
        save_model(self.actor, step, path_a, 'actor', max_model)

    def restore(self,
                step: Optional[int] = None,
                model_folder: Optional[str] = None,
                actor_path: Optional[str] = None,
                verbose: bool = False):
        if actor_path is None:
            actor_path = os.path.join(model_folder, 'actor',
                                      'actor-{}.pth'.format(step))
        restore_model(self.critic,
                      self.device,
                      step=step,
                      model_folder=os.path.join(model_folder, 'critic'),
                      types='critic',
                      verbose=verbose)
        restore_model(self.critic_target,
                      self.device,
                      step=step,
                      model_folder=os.path.join(model_folder, 'critic'),
                      types='critic',
                      verbose=False)
        restore_model(self.actor,
                      self.device,
                      model_path=actor_path,
                      verbose=verbose)

    def restore_refactor(self,
                         step: Optional[int] = None,
                         model_folder: Optional[str] = None,
                         actor_path: Optional[str] = None,
                         verbose: bool = False,
                         types="ctrl"):
        if actor_path is None:
            actor_path = os.path.join(model_folder, types,
                                      '{}-{}.pth'.format(types, step))
        restore_model(self.critic,
                      self.device,
                      step=step,
                      model_folder=os.path.join(model_folder, 'central'),
                      types='central',
                      verbose=verbose)
        restore_model(self.critic_target,
                      self.device,
                      step=step,
                      model_folder=os.path.join(model_folder, 'central'),
                      types='central',
                      verbose=False)
        restore_model(self.actor,
                      self.device,
                      model_path=actor_path,
                      verbose=verbose)

    def remove(self, step: int, model_folder: str):
        remove_model(os.path.join(model_folder, 'critic',
                                  f'critic-{step}.pth'))
        remove_model(os.path.join(model_folder, 'actor', f'actor-{step}.pth'))

    @abstractmethod
    def value(self, obs, append):
        raise NotImplementedError

    # endregion
