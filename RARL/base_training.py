# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

from abc import ABC, abstractmethod
from typing import Optional, Union, List
from collections import namedtuple
from queue import PriorityQueue
import torch
import os
import numpy as np

from .replay_memory import ReplayMemory
from .base_sac import BaseSAC
from .base_ma_sac import BaseMASAC

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'done', 'info'])
TransitionLatent = namedtuple(
    'TransitionLatent', ['z', 's', 'a', 'r', 's_', 'done', 'info']
)


class BaseTraining(ABC):
  module_all: List[Union[BaseSAC, BaseMASAC]]
  module_folder_all: List[str]
  cnt_step: int
  policy: Optional[Union[BaseSAC, BaseMASAC]]
  performance: Optional[Union[BaseSAC, BaseMASAC]]
  backup: Optional[Union[BaseSAC, BaseMASAC]]

  def __init__(self, CONFIG, CONFIG_ENV, CONFIG_UPDATE):
    super(BaseTraining, self).__init__()

    self.device = CONFIG_UPDATE.DEVICE
    self.n_envs = CONFIG.NUM_ENVS
    self.CONFIG = CONFIG
    # self.NUM_VISUALIZE_TASK = CONFIG.NUM_VISUALIZE_TASK

    #! We assume all modules use the same parameters.
    self.batch_size = CONFIG_UPDATE.BATCH_SIZE
    self.MAX_MODEL = CONFIG_UPDATE.MAX_MODEL

    # memory
    self.build_memory(CONFIG.MEMORY_CAPACITY, CONFIG_ENV.SEED)
    self.rng = np.random.default_rng(seed=CONFIG_ENV.SEED)
    self.transition_cls = Transition

    # saving models
    self.save_top_k = self.CONFIG.SAVE_TOP_K
    self.pq_top_k = PriorityQueue()

    self.use_wandb = CONFIG.USE_WANDB

    # dummy attr.
    self.policy = None
    self.performance = None
    self.backup = None

  @property
  @abstractmethod
  def has_backup(self):
    raise NotImplementedError

  def build_memory(self, capacity, seed):
    self.memory = ReplayMemory(capacity, seed)

  def sample_batch(self, batch_size=None, recent_size=0):
    if batch_size is None:
      batch_size = self.batch_size
    if recent_size > 0:  # use recent
      transitions = self.memory.sample_recent(batch_size, recent_size)
    else:
      transitions = self.memory.sample(batch_size)
    batch = self.transition_cls(*zip(*transitions))
    return batch

  def store_transition(self, *args):
    self.memory.update(self.transition_cls(*args))

  def unpack_batch(
      self, batch, get_append=False, get_latent=False, get_perf_action=False
  ):
    non_final_mask = torch.tensor(
        tuple(map(lambda s: not s, batch.done)), dtype=torch.bool
    ).view(-1).to(self.device)
    non_final_state_nxt = torch.cat([
        s for done, s in zip(batch.done, batch.s_) if not done
    ]).to(self.device)
    state = torch.cat(batch.s).to(self.device)

    reward = torch.FloatTensor(batch.r).to(self.device)

    g_x = torch.FloatTensor([info['g_x'] for info in batch.info]
                           ).to(self.device).view(-1)
    l_x = torch.FloatTensor([info['l_x'] for info in batch.info]
                           ).to(self.device).view(-1)

    if get_perf_action:  # recovery RL separates a_shield and a_perf.
      if batch.info[0]['a_perf'].dim() == 1:
        action = torch.FloatTensor([info['a_perf'] for info in batch.info])
      else:
        action = torch.cat([info['a_perf'] for info in batch.info])
      action = action.to(self.device)
    else:
      action = torch.cat(batch.a).to(self.device)

    action = action.float()
    state = state.float()
    non_final_state_nxt = non_final_state_nxt.float()

    latent = None
    if get_latent:
      latent = torch.cat(batch.z).to(self.device)

    append = None
    non_final_append_nxt = None
    if get_append:
      append = torch.cat([info['append'] for info in batch.info]
                        ).to(self.device)
      non_final_append_nxt = torch.cat([
          info['append_nxt'] for info in batch.info
      ]).to(self.device)[non_final_mask]

    binary_cost = torch.FloatTensor([
        info['binary_cost'] for info in batch.info
    ])
    binary_cost = binary_cost.to(self.device).view(-1)

    return (
        non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x,
        latent, append, non_final_append_nxt, binary_cost
    )

  def save(self, metric=None, force_save=False):
    assert metric is not None or force_save, (
        "should provide metric of force save"
    )
    save_current = False
    if force_save:
      save_current = True
    elif self.pq_top_k.qsize() < self.save_top_k:
      self.pq_top_k.put((metric, self.cnt_step))
      save_current = True
    elif metric > self.pq_top_k.queue[0][0]:  # overwrite
      # Remove old one
      _, step_remove = self.pq_top_k.get()
      for module, module_folder in zip(
          self.module_all, self.module_folder_all
      ):
        module.remove(int(step_remove), module_folder)
      self.pq_top_k.put((metric, self.cnt_step))
      save_current = True

    if save_current:
      print('Saving current model...')
      for module, module_folder in zip(
          self.module_all, self.module_folder_all
      ):
        module.save(self.cnt_step, module_folder, self.MAX_MODEL)
      print(self.pq_top_k.queue)

  def restore(
      self, step, model_folder: str, agent_type: Optional[str] = None,
      actor_path: Optional[Union[List[str], str]] = None, **kwargs
  ):
    """Restore the weights of the neural network.

    Args:
        step (int): #updates trained.
        model_folder (str): the path to the models, under this folder there
            should be a folder named "agent_type". There are critic/ and agent/
            folders under model_folder/agent_type.
        agent_type (str, optional): performance, backup, or single agent
            (None). Defaults to None.
        actor_path (str, optional): the path to the actor model. Defaults to
            None.
    """
    if agent_type is None:
      agent_type = "agent"
      model_folder = os.path.join(model_folder)
    else:
      model_folder = os.path.join(model_folder, agent_type)

    if agent_type == 'agent':
      # self.policy.restore(step, model_folder, actor_path)
      self.policy.restore(step, model_folder, **kwargs)
    elif agent_type == 'backup':
      # self.backup.restore(step, model_folder, actor_path)
      self.backup.restore(step, model_folder, **kwargs)
    elif agent_type == 'performance':
      # self.performance.restore(step, model_folder, actor_path)
      self.performance.restore(step, model_folder, **kwargs)
    else:
      raise ValueError("Agent type ({}) is not supported".format(agent_type))
    
    if "load_dict" in kwargs.keys():
      print(
        '  <= Restore {} with load_dict {} updates from {}.'.format(
            agent_type, kwargs["load_dict"], model_folder
        )
      )
    else:
      print(
          '  <= Restore {} with {} updates from {}.'.format(
              agent_type, step, model_folder
          )
      )

  @abstractmethod
  def learn(self):
    raise NotImplementedError
