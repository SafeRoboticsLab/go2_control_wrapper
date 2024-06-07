# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu (kaichieh@princeton.edu)
#          Duy P. Nguyen (duyn@princeton.edu)

import os
import copy
from typing import Optional, List, Union, Tuple
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.nn.functional import mse_loss

from .base_ma_sac import BaseMASAC
from .model import TwinnedQNetwork
from .utils import (soft_update, get_bellman_update, save_model, remove_model,
                    restore_model)


class SAC_adv(BaseMASAC):

    def __init__(self, CONFIG, CONFIG_ARCH):
        super().__init__(CONFIG, CONFIG_ARCH)
        self.action_dim_ctrl = self.action_dim[0]
        self.action_dim_dstb = self.action_dim[1]
        self.dstb_use_ctrl: bool = getattr(CONFIG, "DSTB_USE_CTRL", True)
        # assert self.actor_type[0] == "min", "Ctrl needs to minimize the cost!"
        # assert self.actor_type[1] == "max", "Dstb needs to maximize the cost!"

    # region: property
    @property
    def has_latent(self):
        return False

    @property
    def latent_dist(self):
        return None

    # endregion

    # region: build models and optimizers
    def build_network(self,
                      build_optimizer: bool = True,
                      verbose: bool = True,
                      ctrl_path: Optional[str] = None,
                      dstb_path: Optional[str] = None,
                      adv_critic_path: Optional[str] = None,
                      mean_critic_path: Optional[str] = None):

        # Builds central critic, ctrl actor, and dstb actor.
        actor_paths = [ctrl_path, dstb_path]
        super().build_network(verbose=verbose,
                              actor_paths=actor_paths,
                              critic_path=adv_critic_path)
        self.adv_critic = self.critic  # alias
        self.ctrl = self.actors[0]  # alias
        self.dstb = self.actors[1]  # alias

        # Builds an auxiliary critic (if no dstb for deployment). Assumes the same
        # architecture and activation functions as ones in the central critic.
        if self.critic_has_act_ind:
            mean_critic_action_dim = self.action_dim_ctrl + self.act_ind_dim
        else:
            mean_critic_action_dim = self.action_dim_ctrl
        self.mean_critic = TwinnedQNetwork(
            obs_dim=self.obs_dim['critic'],
            mlp_dim=self.CONFIG_ARCH.DIM_LIST['critic'],
            action_dim=mean_critic_action_dim,
            append_dim=self.CONFIG_ARCH.APPEND_DIM,
            latent_dim=self.CONFIG_ARCH.LATENT_DIM,
            activation_type=self.CONFIG_ARCH.ACTIVATION['critic'],
            device=self.device,
            verbose=verbose)

        # Load model if specified
        if mean_critic_path is not None:
            self.mean_critic.load_state_dict(
                torch.load(mean_critic_path, map_location=self.device))
            print("--> Load mean critic weights from {}".format(
                mean_critic_path))

        # Copy for critic targer
        self.adv_critic_target = copy.deepcopy(self.adv_critic)
        self.mean_critic_target = copy.deepcopy(self.mean_critic)

        # Set up optimizer
        if build_optimizer and not self.EVAL:
            self.build_optimizer()

    def build_optimizer(self, verbose: bool = False):
        super().build_optimizer(verbose)
        self.adv_critic_optimizer = self.critic_optimizer  # alias
        self.mean_critic_optimizer = self.opt_cls(
            self.mean_critic.parameters(), lr=self.LR_C, weight_decay=0.01)

        self.ctrl_optimizer = self.actor_optimizers[0]  # alias
        self.dstb_optimizer = self.actor_optimizers[1]  # alias

        if self.LR_C_SCHEDULE:
            self.adv_critic_scheduler = self.critic_scheduler  # alias
            self.mean_critic_scheduler = lr_scheduler.StepLR(
                self.mean_critic_optimizer,
                step_size=self.LR_C_PERIOD,
                gamma=self.LR_C_DECAY)
        if self.LR_A_SCHEDULE:
            self.ctrl_scheduler = self.actor_schedulers[0]  # alias
            self.dstb_scheduler = self.actor_schedulers[1]  # alias

    # endregion

    # region: main update functions
    def _update_mean_critic(self, batch: List[torch.Tensor]):
        # Gets transition information from the batch.
        (non_final_mask, non_final_state_nxt, state, ctrl, _, reward, g_x, l_x,
         _, append, non_final_append_nxt, binary_cost) = batch
        self.mean_critic.train()
        self.mean_critic_target.eval()
        self.ctrl.eval()

        # Gets Q(s, a).
        q1, q2 = self.mean_critic(state, ctrl, append=append)

        # Computes actor action_nxt and feed to critic_target
        with torch.no_grad():
            action_nxt, _ = self.ctrl.sample(non_final_state_nxt,
                                             append=non_final_append_nxt)
            # Appends action indicator if required.
            if self.critic_has_act_ind:
                act_ind_rep = self.act_ind.repeat(action_nxt.shape[0], 1)
                action_nxt = torch.cat((action_nxt, act_ind_rep), dim=-1)
            next_q1, next_q2 = self.mean_critic_target(
                non_final_state_nxt, action_nxt, append=non_final_append_nxt)

        #? Bellman update includes entropy from ctrl?
        # Gets Bellman update.
        y = get_bellman_update(self.mode,
                               self.batch_size,
                               self.device,
                               next_q1,
                               next_q2,
                               non_final_mask,
                               reward,
                               g_x,
                               l_x,
                               binary_cost,
                               self.GAMMA,
                               terminal_type=self.terminal_type)

        # Regresses MSE loss for both Q1 and Q2.
        loss_q1 = mse_loss(input=q1.view(-1), target=y)
        loss_q2 = mse_loss(input=q2.view(-1), target=y)
        loss_q = loss_q1 + loss_q2

        # Backpropagates.
        self.mean_critic_optimizer.zero_grad()
        loss_q.backward()
        self.mean_critic_optimizer.step()

        self.ctrl.train()

        return loss_q.item()

    def _update_adv_critic(self, batch: List[torch.Tensor]):
        # Gets transition information from the batch.
        (non_final_mask, non_final_state_nxt, state, ctrl, dstb, reward, g_x,
         l_x, _, append, non_final_append_nxt, binary_cost) = batch
        self.adv_critic.train()
        self.adv_critic_target.eval()
        self.ctrl.eval()
        self.dstb.eval()

        # Gets Q(s, a)
        action = torch.cat((ctrl, dstb), dim=-1)
        q1, q2 = self.adv_critic(state, action, append=append)

        # Computes actor action_nxt and feed to critic_target
        with torch.no_grad():
            ctrl_nxt, _ = self.ctrl.sample(non_final_state_nxt,
                                           append=non_final_append_nxt)
            if self.dstb_use_ctrl:
                # Appends ctrl action after state for dstb.
                non_final_state_nxt_dstb = torch.cat(
                    (non_final_state_nxt, ctrl_nxt), dim=-1)
            else:
                non_final_state_nxt_dstb = non_final_state_nxt
            dstb_nxt, _ = self.dstb.sample(non_final_state_nxt_dstb,
                                           append=non_final_append_nxt)
            action_nxt = torch.cat((ctrl_nxt, dstb_nxt), dim=-1)
            # Appends action indicator if required.
            if self.critic_has_act_ind:
                act_ind_rep = self.act_ind.repeat(action_nxt.shape[0], 1)
                action_nxt = torch.cat((action_nxt, act_ind_rep), dim=-1)
            next_q1, next_q2 = self.adv_critic_target(
                non_final_state_nxt, action_nxt, append=non_final_append_nxt)

        #! Bellman update includes entropy from ctrl and dstb?
        # Gets Bellman update.
        y = get_bellman_update(self.mode,
                               self.batch_size,
                               self.device,
                               next_q1,
                               next_q2,
                               non_final_mask,
                               reward,
                               g_x,
                               l_x,
                               binary_cost,
                               self.GAMMA,
                               terminal_type=self.terminal_type)

        # Regresses MSE loss for both Q1 and Q2.
        loss_q1 = mse_loss(input=q1.view(-1), target=y)
        loss_q2 = mse_loss(input=q2.view(-1), target=y)
        loss_q = loss_q1 + loss_q2

        # Backpropagates.
        self.adv_critic_optimizer.zero_grad()
        loss_q.backward()
        self.adv_critic_optimizer.step()

        self.ctrl.train()
        self.dstb.train()

        return loss_q.item()

    def update_critic(self, batch: List[torch.Tensor],
                      update_mean_critic: bool):
        loss_q_adv = self._update_adv_critic(batch)

        loss_q_mean = 0.
        if update_mean_critic:
            loss_q_mean = self._update_mean_critic(batch)
        return loss_q_adv, loss_q_mean

    def update_ctrl(self, batch: List[torch.Tensor]):
        state = batch[2]
        append = batch[8]
        dstb = batch[4]

        self.critic.eval()
        self.dstb.eval()
        self.ctrl.train()

        # Gets actions.
        action_ctrl, log_prob_ctrl = self.ctrl.sample(state, append=append)
        # with torch.no_grad():
        #   # Appends ctrl action after state for dstb.
        #   state_dstb = torch.cat((state, action_ctrl), dim=-1)
        #   dstb, _ = self.dstb.sample(state_dstb, append=append)
        action_sample = torch.cat((action_ctrl, dstb), dim=-1)
        if self.critic_has_act_ind:
            act_ind_rep = self.act_ind.repeat(action_sample.shape[0], 1)
            action_sample = torch.cat((action_sample, act_ind_rep), dim=-1)

        # Gets target values.
        q_pi_1, q_pi_2 = self.critic(state, action_sample, append=append)

        if self.actor_type[0] == "min":
            q_pi = torch.max(q_pi_1, q_pi_2)
        elif self.actor_type[0] == "max":
            q_pi = torch.min(q_pi_1, q_pi_2)

        loss_ent_ctrl = self.alpha[0] * log_prob_ctrl.view(-1).mean()
        if self.actor_type[0] == "min":
            # cost: min_u max_d E[Q + alpha_u * log pi_u - alpha_d * log pi_d]
            loss_q = q_pi.mean()
        elif self.actor_type[0] == "max":
            # reward: max_u min_d E[Q - alpha_u * log pi_u + alpha_d * log pi_d]
            loss_q = -q_pi.mean()
        loss_pi_ctrl = loss_q + loss_ent_ctrl

        # Backpropagates.
        self.ctrl_optimizer.zero_grad()
        loss_pi_ctrl.backward()
        self.ctrl_optimizer.step()

        # Tunes entropy temperature automatically.
        log_prob = log_prob_ctrl.detach()
        loss_alpha = (self.alpha[0] *
                      (-log_prob - self.target_entropy[0])).mean()
        if self.LEARN_ALPHA:
            self.log_alpha_optimizer[0].zero_grad()
            loss_alpha.backward()
            self.log_alpha_optimizer[0].step()

            self.log_alpha[0].data = torch.min(self.log_alpha[0].data,
                                               self.init_alpha[0].data)

        self.critic.train()
        self.dstb.train()
        return loss_q.item(), loss_ent_ctrl.item(), loss_alpha.item()

    def update_dstb(self, batch: List[torch.Tensor]):
        state = batch[2]
        append = batch[8]
        ctrl = batch[3]

        self.critic.eval()
        self.ctrl.eval()
        self.dstb.train()

        # Gets actions.
        # with torch.no_grad():
        #   ctrl, _ = self.ctrl.sample(state, append=append)
        if self.dstb_use_ctrl:
            # Appends ctrl action after state for dstb.
            state_dstb = torch.cat((state, ctrl), dim=-1)
        else:
            state_dstb = state
        action_dstb, log_prob_dstb = self.dstb.sample(state_dstb,
                                                      append=append)
        action_sample = torch.cat((ctrl, action_dstb), dim=-1)
        if self.critic_has_act_ind:
            act_ind_rep = self.act_ind.repeat(action_sample.shape[0], 1)
            action_sample = torch.cat((action_sample, act_ind_rep), dim=-1)

        # Gets target values.
        q_pi_1, q_pi_2 = self.critic(state, action_sample, append=append)

        if self.actor_type[1] == "min":
            q_pi = torch.max(q_pi_1, q_pi_2)
        elif self.actor_type[1] == "max":
            q_pi = torch.min(q_pi_1, q_pi_2)

        loss_ent_dstb = self.alpha[1] * log_prob_dstb.view(-1).mean()
        if self.actor_type[1] == "min":
            # reward: max_u min_d E[Q - alpha_u * log pi_u + alpha_d * log pi_d]
            loss_q = q_pi.mean()
        elif self.actor_type[1] == "max":
            # cost: min_u max_d E[Q + alpha_u * log pi_u - alpha_d * log pi_d]
            loss_q = -q_pi.mean()
        loss_pi_dstb = loss_q + loss_ent_dstb

        # Backpropagates.
        self.dstb_optimizer.zero_grad()
        loss_pi_dstb.backward()
        self.dstb_optimizer.step()

        # Tunes entropy temperature automatically.
        log_prob = log_prob_dstb.detach()
        loss_alpha = (self.alpha[1] *
                      (-log_prob - self.target_entropy[1])).mean()
        if self.LEARN_ALPHA:
            self.log_alpha_optimizer[1].zero_grad()
            loss_alpha.backward()
            self.log_alpha_optimizer[1].step()
            self.log_alpha[1].data = torch.min(self.log_alpha[1].data,
                                               self.init_alpha[1].data)

        self.critic.train()
        self.ctrl.train()
        return loss_q.item(), loss_ent_dstb.item(), loss_alpha.item()

    def update_actor(self, batch: List[torch.Tensor]):  # required by base_sac.
        pass

    def update_target_networks(self, update_mean_critic: bool):
        soft_update(self.adv_critic_target, self.adv_critic, self.TAU)
        if update_mean_critic:
            soft_update(self.mean_critic_target, self.mean_critic, self.TAU)

    def update(
        self,
        batch: List[torch.Tensor],
        timer: int,
        update_ctrl: bool,
        update_dstb: bool,
        update_period: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float, float, float]:
        if update_period is None:
            update_period = self.update_period

        loss_q_adv, loss_q_mean = self.update_critic(batch,
                                                     update_mean_critic=True)

        # Updates each actor according to its own update period.
        loss_pi, loss_entropy, loss_alpha = 0., 0., 0.
        assert not (update_ctrl and update_dstb), "cannot update both actors."
        if update_ctrl:
            policy_delay = update_period[0]
            update_fn = self.update_ctrl
        elif update_dstb:
            policy_delay = update_period[1]
            update_fn = self.update_dstb
        else:
            policy_delay = update_period[1]  #! defaults to dstb.
            update_fn = None

        if timer % policy_delay == 0:  # Updates dstb/ctrl and target networks.
            if update_fn is not None:
                loss_pi, loss_entropy, loss_alpha = update_fn(batch)
            self.update_target_networks(update_mean_critic=True)

        return loss_q_adv, loss_q_mean, loss_pi, loss_entropy, loss_alpha

    # endregion

    # region: utils
    def save(self,
             step: int,
             model_folder: str,
             max_model: Optional[int] = None,
             save_dstb: bool = True,
             save_ctrl: bool = True,
             save_critic: bool = True):  #* overrides
        if save_critic:
            save_model(self.mean_critic, step,
                       os.path.join(model_folder, 'mean_critic'), 'critic',
                       max_model)
            save_model(self.adv_critic, step,
                       os.path.join(model_folder, 'adv_critic'), 'critic',
                       max_model)

        if save_ctrl:
            save_model(self.ctrl, step, os.path.join(model_folder, 'ctrl'),
                       'actor', max_model)

        if save_dstb:
            save_model(self.dstb, step, os.path.join(model_folder, 'dstb'),
                       'actor', max_model)

    def remove(self,
               step: int,
               model_folder: str,
               rm_dstb: bool = True,
               rm_ctrl: bool = True,
               rm_critic: bool = True):
        if rm_critic:
            remove_model(
                os.path.join(model_folder, 'mean_critic',
                             f'critic-{step}.pth'))
            remove_model(
                os.path.join(model_folder, 'adv_critic', f'critic-{step}.pth'))

        if rm_ctrl:
            remove_model(
                os.path.join(model_folder, 'ctrl', f'actor-{step}.pth'))

        if rm_dstb:
            remove_model(
                os.path.join(model_folder, 'dstb', f'actor-{step}.pth'))

    def restore(self,
                step: int,
                model_folder: str,
                load_dict: Optional[dict] = None):
        if load_dict is not None:
            ## critic follows control
            ctrl_step = load_dict["ctrl"]
            dstb_step = load_dict["dstb"]
        else:
            ctrl_step = step
            dstb_step = step

        restore_model(self.mean_critic, self.device, ctrl_step,
                      os.path.join(model_folder, 'mean_critic'), 'critic')
        restore_model(self.adv_critic, self.device, ctrl_step,
                      os.path.join(model_folder, 'adv_critic'), 'critic')
        restore_model(self.mean_critic_target, self.device, ctrl_step,
                      os.path.join(model_folder, 'mean_critic'), 'critic')
        restore_model(self.adv_critic_target, self.device, ctrl_step,
                      os.path.join(model_folder, 'adv_critic'), 'critic')
        restore_model(self.ctrl, self.device, ctrl_step,
                      os.path.join(model_folder, 'ctrl'), 'actor')
        restore_model(self.dstb, self.device, dstb_step,
                      os.path.join(model_folder, 'dstb'), 'actor')

    def restore_refactor(self,
                         step: int,
                         model_folder: str,
                         load_dict: Optional[dict] = None):
        if load_dict is not None:
            ## critic follows control
            ctrl_step = load_dict["ctrl"]
            dstb_step = load_dict["dstb"]
        else:
            ctrl_step = step
            dstb_step = step

        restore_model(self.adv_critic, self.device, ctrl_step,
                      os.path.join(model_folder, 'central'), 'central')
        restore_model(self.adv_critic_target, self.device, ctrl_step,
                      os.path.join(model_folder, 'central'), 'central')
        restore_model(self.ctrl, self.device, ctrl_step,
                      os.path.join(model_folder, 'ctrl'), 'ctrl')
        restore_model(self.dstb, self.device, dstb_step,
                      os.path.join(model_folder, 'dstb'), 'dstb')

    def value(self,
              state: np.ndarray,
              append: Optional[Union[np.ndarray, torch.Tensor]] = None,
              use_adv: bool = True):
        action_ctrl = self.ctrl.forward(state, append=append)

        if use_adv:
            if self.dstb_use_ctrl:
                state_dstb = np.concatenate((state, action_ctrl), axis=-1)
            else:
                state_dstb = state
            action_dstb = self.dstb.forward(state_dstb, append=append)
            action = np.concatenate((action_ctrl, action_dstb), axis=-1)
        else:
            action = action_ctrl
        action = torch.FloatTensor(action).to(self.device)

        if self.critic_has_act_ind:
            act_ind_rep = self.act_ind.repeat(action.shape[0], 1)
            action = torch.cat((action, act_ind_rep), dim=-1)

        if use_adv:
            q_pi_1, q_pi_2 = self.adv_critic(state, action, append=append)
        else:
            q_pi_1, q_pi_2 = self.mean_critic(state, action, append=append)
        value = (q_pi_1 + q_pi_2) / 2
        return value

    def get_adversary(self,
                      obs: np.ndarray,
                      ctrl: np.ndarray,
                      append: Optional[Union[np.ndarray, torch.Tensor]] = None,
                      **kwargs) -> np.ndarray:
        if self.dstb_use_ctrl:
            obs_dstb = np.concatenate((obs, ctrl), axis=-1)
        else:
            obs_dstb = obs
        dstb = self.dstb.forward(obs_dstb, append=append)
        assert isinstance(dstb, np.ndarray)
        return dstb

    # endregion
