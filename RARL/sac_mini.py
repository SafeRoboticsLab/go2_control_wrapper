# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

from threading import active_count
from typing import Optional, List, Union
import numpy as np
import torch
from torch.nn.functional import mse_loss

from .base_sac import BaseSAC
from .utils import get_bellman_update


class SAC_mini(BaseSAC):

  def __init__(self, CONFIG, CONFIG_ARCH):
    super().__init__(CONFIG, CONFIG_ARCH)

  # region: property
  @property
  def has_latent(self):
    return False

  @property
  def latent_dist(self):
    return None

  # region: build models and optimizers
  def build_network(
      self, build_optimizer=True, verbose=True, actor_path=None,
      critic_path=None
  ):
    super().build_network(
        verbose, actor_path=actor_path, critic_path=critic_path
    )

    # Set up optimizer
    if build_optimizer and not self.EVAL:
      super().build_optimizer()
    else:
      for _, param in self.actor.named_parameters():
        param.requires_grad = False
      for _, param in self.critic.named_parameters():
        param.requires_grad = False
      self.actor.eval()
      self.critic.eval()

  # region: main update functions
  def update_critic(self, batch):
    (
        non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x,
        _, append, non_final_append_nxt, binary_cost
    ) = batch
    self.critic.train()
    self.critic_target.eval()
    self.actor.eval()

    # Gets Q(s, a).
    q1, q2 = self.critic(state, action, append=append)

    # Computes actor action_nxt and feed to critic_target
    with torch.no_grad():
      action_nxt, next_log_prob = self.actor.sample(
          non_final_state_nxt, append=non_final_append_nxt
      )
      if self.critic_has_act_ind:
        act_ind_rep = self.act_ind.repeat(action_nxt.shape[0], 1)
        action_nxt = torch.cat((action_nxt, act_ind_rep), dim=-1)
      next_q1, next_q2 = self.critic_target(
          non_final_state_nxt, action_nxt, append=non_final_append_nxt
      )

    # Gets Bellman update.
    y = get_bellman_update(
        self.mode, self.batch_size, self.device, next_q1, next_q2,
        non_final_mask, reward, g_x, l_x, binary_cost, self.GAMMA,
        terminal_type=self.terminal_type
    )

    if self.mode == 'performance':
      y[non_final_mask] -= self.GAMMA * self.alpha * next_log_prob.view(-1)

    # Regresses MSE loss for both Q1 and Q2.
    loss_q1 = mse_loss(input=q1.view(-1), target=y)
    loss_q2 = mse_loss(input=q2.view(-1), target=y)
    loss_q = loss_q1 + loss_q2

    # Backpropagates.
    self.critic_optimizer.zero_grad()
    loss_q.backward()
    self.critic_optimizer.step()

    self.actor.train()

    return loss_q.item()

  def update_actor(self, batch):
    state = batch[2]
    append = batch[8]

    self.critic.eval()
    self.actor.train()

    action_sample, log_prob = self.actor.sample(state, append=append)
    if self.critic_has_act_ind:
      act_ind_rep = self.act_ind.repeat(action_sample.shape[0], 1)
      action_sample = torch.cat((action_sample, act_ind_rep), dim=-1)

    q_pi_1, q_pi_2 = self.critic(state, action_sample, append=append)

    if self.actor_type == "min":
      q_pi = torch.max(q_pi_1, q_pi_2)
    elif self.actor_type == "max":
      q_pi = torch.min(q_pi_1, q_pi_2)

    # cost: min_theta E[ Q + alpha * (log pi + H)]
    # loss_pi = Q + alpha * log pi
    # reward: max_theta E[ Q - alpha * (log pi + H)]
    # loss_pi = -Q + alpha * log pi
    loss_entropy = self.alpha * log_prob.view(-1).mean()
    if self.actor_type == "min":
      loss_q_eval = q_pi.mean()
    elif self.actor_type == "max":
      loss_q_eval = -q_pi.mean()
    loss_pi = loss_q_eval + loss_entropy

    self.actor_optimizer.zero_grad()
    loss_pi.backward()
    self.actor_optimizer.step()

    # Automatic temperature tuning
    loss_alpha = (self.alpha *
                  (-log_prob.detach() - self.target_entropy)).mean()
    if self.LEARN_ALPHA:
      self.log_alpha_optimizer.zero_grad()
      loss_alpha.backward()
      self.log_alpha_optimizer.step()
      self.log_alpha.data = torch.min(
          self.log_alpha.data, self.init_alpha.data
      )
    return loss_q_eval.item(), loss_entropy.item(), loss_alpha.item()

  def update(
      self, batch: List[torch.Tensor], timer: int,
      update_period: Optional[int] = None
  ):
    if update_period is None:
      update_period = self.update_period
    self.critic.train()
    self.actor.train()

    loss_q = self.update_critic(batch)
    loss_pi, loss_entropy, loss_alpha = 0, 0, 0
    if timer % update_period == 0:
      loss_pi, loss_entropy, loss_alpha = self.update_actor(batch)
      self.update_target_networks()

    self.critic.eval()
    self.actor.eval()

    return loss_q, loss_pi, loss_entropy, loss_alpha

  # region: utils
  def value(
      self,
      state: np.ndarray,
      append: Optional[Union[np.ndarray, torch.Tensor]] = None,
  ) -> np.ndarray:
    action = self.actor.forward(state, append=append)
    action = torch.FloatTensor(action).to(self.device)
    if self.critic_has_act_ind:
      act_ind_rep = self.act_ind.repeat(action.shape[0], 1)
      action = torch.cat((action, act_ind_rep), dim=-1)

    q_pi_1, q_pi_2 = self.critic(state, action, append=append)
    value = (q_pi_1+q_pi_2) / 2
    return value
