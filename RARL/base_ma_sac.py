# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Duy

from abc import ABC, abstractmethod
from typing import List, Optional, Dict
import os
import copy
import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR

from .model import GaussianPolicy, TwinnedQNetwork
from .scheduler import StepLRMargin
from .utils import soft_update, save_model


class BaseMASAC(ABC):
  """
  Implements the multi-agent SAC base class following the centralized training
  and decentralized execution framework. Assumes all actors take in the same
  state and there is a central critic.
  """
  actors: List[GaussianPolicy]
  actor_optimizers: List[StepLR]

  def __init__(self, CONFIG, CONFIG_ARCH):
    """
    __init__: initialization.

    Args:
        CONFIG (object): update-rekated hyper-parameter configuration.
        CONFIG_ARCH (object): NN architecture configuration.
    """
    self.CONFIG = CONFIG
    self.CONFIG_ARCH = CONFIG_ARCH
    self.EVAL = CONFIG.EVAL
    self.mode = CONFIG.MODE

    # == ENV PARAM ==
    self.action_range = [
        np.array(x, dtype=np.float32) for x in CONFIG_ARCH.ACTION_RANGE
    ]
    self.action_dim = np.array(CONFIG_ARCH.ACTION_DIM)
    self.num_agents = len(self.action_dim)
    self.action_dim_all = np.sum(self.action_dim)
    assert len(self.action_range) == self.num_agents, \
        "the number of agents is not consistent!"
    self.obs_dim: Dict[str, int] = CONFIG_ARCH.OBS_DIM

    # NN: device, action indicators
    self.device = CONFIG.DEVICE
    self.actor_type = CONFIG.ACTOR_TYPE  # a list of "min" or "max"
    self.critic_has_act_ind = CONFIG_ARCH.CRITIC_HAS_ACT_IND
    if self.critic_has_act_ind:
      assert hasattr(CONFIG_ARCH, "ACT_IND"), "Needs action indicator!"
      self.act_ind = torch.FloatTensor(CONFIG_ARCH.ACT_IND).to(self.device)
      self.act_ind_dim = self.act_ind.shape[0]

    # == PARAM FOR TRAINING ==
    # Assumes each agent has the same training hyper-parameters except alpha
    # for now.
    if not self.EVAL:
      if CONFIG.OPT_TYPE == "AdamW":
        self.opt_cls = AdamW
      elif CONFIG.OPT_TYPE == "Adam":
        self.opt_cls = Adam
      else:
        raise ValueError("Not supported optimizer type!")
      self.terminal_type = CONFIG.TERMINAL_TYPE

      # NN
      self.batch_size = CONFIG.BATCH_SIZE
      self.update_period = np.array(CONFIG.UPDATE_PERIOD)  # np.ndarray

      # Learning Rate
      self.LR_A_SCHEDULE = CONFIG.LR_A_SCHEDULE
      self.LR_C_SCHEDULE = CONFIG.LR_C_SCHEDULE
      if self.LR_A_SCHEDULE:
        self.LR_A_PERIOD = CONFIG.LR_A_PERIOD
        self.LR_A_DECAY = CONFIG.LR_A_DECAY
        self.LR_A_END = CONFIG.LR_A_END
      if self.LR_C_SCHEDULE:
        self.LR_C_PERIOD = CONFIG.LR_C_PERIOD
        self.LR_C_DECAY = CONFIG.LR_C_DECAY
        self.LR_C_END = CONFIG.LR_C_END
      self.LR_C = CONFIG.LR_C
      self.LR_A = CONFIG.LR_A

      # Discount factor
      self.GAMMA_SCHEDULE = CONFIG.GAMMA_SCHEDULE
      if self.GAMMA_SCHEDULE:
        self.gamma_scheduler = StepLRMargin(
            init_value=CONFIG.GAMMA, period=CONFIG.GAMMA_PERIOD,
            decay=CONFIG.GAMMA_DECAY, end_value=CONFIG.GAMMA_END, goal_value=1.
        )
        self.GAMMA = self.gamma_scheduler.get_variable()
      else:
        self.GAMMA = CONFIG.GAMMA

      # Target Network Update
      self.TAU = CONFIG.TAU

      # alpha-related hyper-parameters
      if isinstance(CONFIG.ALPHA, list):
        self.init_alpha = torch.log(torch.FloatTensor(CONFIG.ALPHA)
                                   ).to(self.device)
      else:
        self.init_alpha = torch.log(
            torch.full(size=(self.num_agents,), fill_value=CONFIG.ALPHA)
        ).to(self.device)
      self.LEARN_ALPHA = CONFIG.LEARN_ALPHA
      self.log_alpha = [
          log_alp.detach().clone() for log_alp in self.init_alpha
      ]
      self.target_entropy = -torch.tensor(self.action_dim).to(self.device)
      if self.LEARN_ALPHA:
        for log_alp in self.log_alpha:
          log_alp.requires_grad = True
        self.LR_Al = CONFIG.LR_Al
        self.LR_Al_SCHEDULE = CONFIG.LR_Al_SCHEDULE
        if self.LR_Al_SCHEDULE:
          self.LR_Al_PERIOD = CONFIG.LR_Al_PERIOD
          self.LR_Al_DECAY = CONFIG.LR_Al_DECAY
          self.LR_Al_END = CONFIG.LR_Al_END

  # region: property
  @property
  def alpha(self):
    return [log_alp.exp() for log_alp in self.log_alpha]

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
  def build_actor(
      self, obs_dim, mlp_dim, action_dim, action_range, latent_dim,
      activation_type, verbose=True
  ):
    actor = GaussianPolicy(
        obs_dim=obs_dim, mlp_dim=mlp_dim, action_dim=action_dim,
        action_range=action_range, append_dim=self.CONFIG_ARCH.APPEND_DIM,
        latent_dim=latent_dim, activation_type=activation_type,
        device=self.device, verbose=verbose
    )

    return actor

  def build_network(
      self, verbose: bool = True,
      actor_paths: Optional[List[Optional[str]]] = None,
      critic_path: Optional[str] = None
  ):
    if self.critic_has_act_ind:
      critic_action_dim = self.action_dim_all + self.act_ind_dim
    else:
      critic_action_dim = self.action_dim_all

    self.critic = TwinnedQNetwork(
        obs_dim=self.obs_dim['critic'],
        mlp_dim=self.CONFIG_ARCH.DIM_LIST['critic'],
        action_dim=critic_action_dim, append_dim=self.CONFIG_ARCH.APPEND_DIM,
        latent_dim=self.CONFIG_ARCH.LATENT_DIM,
        activation_type=self.CONFIG_ARCH.ACTIVATION['critic'],
        device=self.device, verbose=verbose
    )

    # Load model if specified
    if critic_path is not None:
      self.critic.load_state_dict(
          torch.load(critic_path, map_location=self.device)
      )
      print("--> Load central critic weights from {}".format(critic_path))

    # Copy for critic targer
    self.critic_target = copy.deepcopy(self.critic)

    if verbose:
      print("\nThe actor shares the same encoder with the critic.")
    self.actors = []
    for i in range(self.num_agents):
      actor = self.build_actor(
          obs_dim=self.obs_dim['actor_' + str(i)],
          mlp_dim=self.CONFIG_ARCH.DIM_LIST['actor_' + str(i)],
          action_dim=self.action_dim[i],
          action_range=self.action_range[i],
          #! below two args are assumed the same for now
          latent_dim=self.CONFIG_ARCH.LATENT_DIM,
          activation_type=self.CONFIG_ARCH.ACTIVATION['actor'],
          verbose=verbose
      )
      if actor_paths[i] is not None:
        actor_path = actor_paths[i]
        actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        print(f"--> Load actor_{i} wights from {actor_path}")
      self.actors.append(actor)

  def build_optimizer(self, verbose: bool = False):
    if verbose:
      print("Build basic optimizers.")
    # central critic
    self.critic_optimizer = self.opt_cls(
        self.critic.parameters(), lr=self.LR_C, weight_decay=0.01
    )
    if self.LR_C_SCHEDULE:
      self.critic_scheduler = StepLR(
          self.critic_optimizer, step_size=self.LR_C_PERIOD,
          gamma=self.LR_C_DECAY
      )

    # actors
    self.actor_optimizers = []
    if self.LR_A_SCHEDULE:
      self.actor_schedulers = []
    for i in range(self.num_agents):
      actor_optimizer = self.opt_cls(
          self.actors[i].parameters(), lr=self.LR_A, weight_decay=0.01
      )
      self.actor_optimizers.append(actor_optimizer)

      if self.LR_A_SCHEDULE:
        self.actor_schedulers.append(
            StepLR(
                actor_optimizer, step_size=self.LR_A_PERIOD,
                gamma=self.LR_A_DECAY
            )
        )

    # entropy temperature parameters
    if self.LEARN_ALPHA:
      self.log_alpha_optimizer = []
      if self.LR_Al_SCHEDULE:
        self.log_alpha_scheduler = []
      for i in range(self.num_agents):
        self.log_alpha_optimizer.append(
            self.opt_cls([self.log_alpha[i]], lr=self.LR_Al[i], weight_decay=0.01)
        )
        if self.LR_Al_SCHEDULE:
          self.log_alpha_scheduler.append(
              StepLR(
                  self.log_alpha_optimizer[i], step_size=self.LR_Al_PERIOD,
                  gamma=self.LR_Al_DECAY
              )
          )

  # endregion

  # region: update functions
  def reset_alpha(self):
    self.log_alpha = [log_alp.detach().clone() for log_alp in self.init_alpha]
    if self.LEARN_ALPHA:
      for log_alp in self.log_alpha:
        log_alp.requires_grad = True
      self.log_alpha_optimizer = []
      if self.LR_Al_SCHEDULE:
        self.log_alpha_scheduler = []
      for i in range(self.num_agents):
        self.log_alpha_optimizer.append(
            self.opt_cls([self.log_alpha[i]], lr=self.LR_Al[i], weight_decay=0.01)
        )
        if self.LR_Al_SCHEDULE:
          self.log_alpha_scheduler.append(
              StepLR(
                  self.log_alpha_optimizer[i], step_size=self.LR_Al_PERIOD,
                  gamma=self.LR_Al_DECAY
              )
          )

  def update_alpha_hyper_param(self):
    if self.LR_Al_SCHEDULE:
      for i in range(self.num_agents):
        log_alpha_optimizer = self.log_alpha_optimizer[i]
        lr = log_alpha_optimizer.state_dict()['param_groups'][0]['lr']
        if lr <= self.LR_Al_END:
          for param_group in log_alpha_optimizer.param_groups:
            param_group['lr'] = self.LR_Al_END
        else:
          self.log_alpha_scheduler[i].step()

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
      for i in range(self.num_agents):
        actor_optimizer = self.actor_optimizers[i]
        lr = actor_optimizer.state_dict()['param_groups'][0]['lr']
        if lr <= self.LR_A_END:
          for param_group in actor_optimizer.param_groups:
            param_group['lr'] = self.LR_A_END
        else:
          self.actor_schedulers[i].step()

  def update_hyper_param(self):
    self.update_critic_hyper_param()
    self.update_actor_hyper_param()
    if self.LEARN_ALPHA:
      self.update_alpha_hyper_param()

  def update_target_networks(self):
    soft_update(self.critic_target, self.critic, self.TAU)

  @abstractmethod
  def update_actor(self, batch: List[torch.Tensor]):
    raise NotImplementedError

  @abstractmethod
  def update_critic(self, batch: List[torch.Tensor]):
    raise NotImplementedError

  @abstractmethod
  def update(
      self, batch: List[torch.Tensor], timer: int,
      update_period: Optional[int] = None
  ):
    raise NotImplementedError

  # endregion

  # region: utils
  def save(
      self, step: int, model_folder: str, max_model: Optional[int] = None
  ):
    path_c = os.path.join(model_folder, 'critic')
    save_model(self.critic, step, path_c, 'critic', max_model)
    for i in range(self.num_agents):
      path_a = os.path.join(model_folder, 'actor_' + str(i))
      save_model(self.actors[i], step, path_a, 'actor', max_model)

  def restore(
      self, step: Optional[int] = None, model_folder: Optional[str] = None,
      critic_path: Optional[str] = None, actor_path: Optional[List[str]] = None, **kwargs
  ):
    if critic_path is None:
      path_c = os.path.join(
          model_folder, 'critic', 'critic-{}.pth'.format(step)
      )
    else:
      path_c = critic_path
    self.critic.load_state_dict(torch.load(path_c, map_location=self.device))
    self.critic.to(self.device)
    self.critic_target.load_state_dict(
        torch.load(path_c, map_location=self.device)
    )
    self.critic_target.to(self.device)
    for i in range(self.num_agents):
      if actor_path is not None:
        path_a = actor_path[i]
      else:
        path_a = os.path.join(
            model_folder, 'actor_' + str(i), 'actor-{}.pth'.format(step)
        )
      self.actors[i].load_state_dict(
          torch.load(path_a, map_location=self.device)
      )
      self.actors[i].to(self.device)

  def remove(self, step: int, model_folder: str):
    path_c = os.path.join(model_folder, 'critic', 'critic-{}.pth'.format(step))
    print("Remove", path_c)
    if os.path.exists(path_c):
      os.remove(path_c)

    for i in range(self.num_agents):
      path_a = os.path.join(
          model_folder, 'actor_' + str(i), 'actor-{}.pth'.format(step)
      )
      print("Remove", path_a)
      if os.path.exists(path_a):
        os.remove(path_a)

  @abstractmethod
  def value(self, obs, append):
    raise NotImplementedError

  # endregion
