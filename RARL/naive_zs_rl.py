# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu (kaichieh@princeton.edu)
#          Duy P. Nguyen (duyn@princeton.edu)

import copy
from multiprocessing.sharedctypes import Value
from typing import Optional, Callable, Dict, Tuple, Union, List, Any
import os
import copy
import time
import math
import numpy as np
from scipy.fft import dst
import torch
import wandb
from functools import partial

from .sac_adv import SAC_adv
from .base_zs_training import BaseZeroSumTraining
from .utils import restore_model
from simulators.base_zs_env import BaseZeroSumEnv
from simulators.vec_env.vec_env import VecEnvBase
from simulators.agent import Agent


# region: Getting adversarial disturbance.
# Multiprocessing requires functions on the top instead in self.
def dummy_dstb(obs, ctrl, append=None, dim: int = 0, **kwargs):
  return np.zeros(dim)


def random_dstb(obs, ctrl, **kwargs):
  rng: np.random.Generator = kwargs.get("rng")
  dstb_range: np.ndarray = kwargs.get("dstb_range")
  return rng.uniform(low=dstb_range[:, 0], high=dstb_range[:, 1])


def get_adversary(
    obs: np.ndarray, ctrl: np.ndarray, append: Optional[np.ndarray] = None,
    dstb_policy=None, use_ctrl: bool = True, **kwargs
) -> np.ndarray:
  if use_ctrl:
    obs_dstb = np.concatenate((obs, ctrl), axis=-1)
  else:
    obs_dstb = obs
  dstb = dstb_policy(obs_dstb, append=append)
  assert isinstance(dstb, np.ndarray)
  return dstb


# endregion

import pybullet as p
import pandas as pd

class NaiveZeroSumRL(BaseZeroSumTraining):
  policy: SAC_adv

  def __init__(
      self, CONFIG, CONFIG_UPDATE, CONFIG_ARCH, CONFIG_ENV, verbose=True
  ):
    super().__init__(CONFIG, CONFIG_ENV, CONFIG_UPDATE)

    print("= Constructing policy agent")
    self.policy = SAC_adv(CONFIG_UPDATE, CONFIG_ARCH)
    CONFIG_UPDATE_DUP = copy.deepcopy(CONFIG_UPDATE)
    CONFIG_UPDATE_DUP.EVAL = True
    self.policy_sample = SAC_adv(CONFIG_UPDATE_DUP, CONFIG_ARCH)
    ctrl_path = getattr(CONFIG, "CTRL_PATH", None)
    dstb_path = getattr(CONFIG, "DSTB_PATH", None)
    self.policy.build_network(
        ctrl_path=ctrl_path, dstb_path=dstb_path, verbose=verbose
    )
    self.policy_sample.build_network(
        ctrl_path=ctrl_path, dstb_path=dstb_path, verbose=False
    )
    self.save_top_k = np.array(self.CONFIG.SAVE_TOP_K)  # 1st: ctrl, 2nd: dstb
    self.dstb_res_dict: Dict[int, Tuple] = {
        -1: (0, 0.)
    }  # key: step, value: (#gameplays, metric_avg)

    # alias
    self.module_all = [self.policy]
    self.performance = self.policy

    # This algorithm can also train a single actor by fixing the other.
    self.fix_ctrl: bool = getattr(self.CONFIG, "FIX_CTRL", False)
    self.fix_dstb: bool = getattr(self.CONFIG, "FIX_DSTB", False)
    self.use_random_dstb: bool = getattr(self.CONFIG, "RANDOM_DSTB", False)
    if self.use_random_dstb:
      self.fix_dstb = True

    # if self.fix_ctrl:
    #   for p in self.policy.ctrl.parameters():
    #     p.requires_grad = False
    # if self.fix_dstb:
    #   for p in self.policy.dstb.parameters():
    #     p.requires_grad = False

  @property
  def has_backup(self):
    return False

  def dummy_dstb_sample(self, obs, append=None, latent=None, **kwargs):
    return torch.zeros(self.policy.action_dim_dstb).to(self.device), 0.

  def random_dstb_sample(self, obs, **kwargs):
    dstb_range = self.policy.action_range[1]
    dstb = self.rng.uniform(low=dstb_range[:, 0], high=dstb_range[:, 1])
    return torch.FloatTensor(dstb).to(self.device), 0.

  def learn(
      self, env: BaseZeroSumEnv, current_step: Optional[int] = None,
      reset_kwargs: Optional[Dict] = None,
      action_kwargs: Optional[Dict] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None,
      visualize_callback: Optional[Callable] = None,
      **kwargs
  ):
    if reset_kwargs is None:
      reset_kwargs = {}

    controller=None
    if "controller" in kwargs.keys():
      controller = kwargs["controller"]

    # hyper-parameters
    max_steps: int = self.CONFIG.MAX_STEPS
    opt_freq: int = self.CONFIG.OPTIMIZE_FREQ
    num_update_per_opt_list = np.array(self.CONFIG.UPDATE_PER_OPT)
    check_opt_freq: int = self.CONFIG.CHECK_OPT_FREQ
    ctrl_opt_freq: int = self.CONFIG.CTRL_OPT_FREQ
    min_step_b4_opt: int = self.CONFIG.MIN_STEPS_B4_OPT
    min_steps_b4_exploit: int = self.CONFIG.MIN_STEPS_B4_EXPLOIT
    out_folder: str = self.CONFIG.OUT_FOLDER
    rollout_end_criterion: str = self.CONFIG.ROLLOUT_END_CRITERION
    warmup_critic_steps: int = getattr(self.CONFIG, "WARMUP_CRITIC_STEPS", 0)

    # main training
    start_learning = time.time()
    train_record = []
    train_progress = []
    violation_record = []
    episode_record = []
    cnt_opt = 0
    cnt_opt_period = 0
    cnt_dstb_opt = 0
    cnt_safety_violation = 0
    cnt_num_episode = 0
    first_update = True

    # saving model
    model_folder = os.path.join(out_folder, 'model')
    dstb_folder = os.path.join(model_folder, 'dstb')
    os.makedirs(dstb_folder, exist_ok=True)
    figure_folder = os.path.join(out_folder, 'figure')
    os.makedirs(figure_folder, exist_ok=True)
    self.module_folder_all = [model_folder]

    if current_step is None:
      self.cnt_step = 0
    else:
      self.cnt_step = current_step
      print("starting from {:d} steps".format(self.cnt_step))

    # TODO: Add variable to toggle multiprocessing or not
    single_env = True
    if not single_env:
      venv = VecEnvBase([copy.deepcopy(env) for _ in range(self.n_envs)],
                        device=self.device)
      # Venv can run on a different device.
      agent_list = venv.get_attr(attr_name='agent')
      for agent in agent_list:
        agent: Agent
        if agent.policy is not None:
          agent.policy.to(torch.device(self.CONFIG.VENV_DEVICE))
        if agent.safety_policy is not None:
          agent.safety_policy.to(torch.device(self.CONFIG.VENV_DEVICE))

      obs_all = venv.reset()
    else:
      venv = env
      obs = venv.reset(cast_torch=True, **reset_kwargs)
      obs_all = torch.FloatTensor(np.stack([obs])).to(self.device)
    
    dstb_sample_fn_all = [
        self.get_dstb_sample_fn(
            dstb_folder, dstb_sample_type=self.CONFIG.INIT_DSTB_SAMPLE_TYPE,
            verbose=True
        ) for _ in range(self.n_envs)
    ]
    
    while self.cnt_step <= max_steps:
      # Selects action.
      with torch.no_grad():
        if self.cnt_step < min_steps_b4_exploit:
          ctrl_all = np.empty(shape=(self.n_envs, env.action_dim_ctrl))
          dstb_all = np.empty(shape=(self.n_envs, env.action_dim_dstb))
          action_all = []
          for i in range(self.n_envs):
            action_zs = env.action_space.sample()
            action_all.append(action_zs)
            ctrl_all[i, :] = action_zs['ctrl']
            dstb_all[i, :] = action_zs['dstb']
          ctrl_all = torch.FloatTensor(ctrl_all).to(self.device)
          dstb_all = torch.FloatTensor(dstb_all).to(self.device)
        else:
          ctrl_all, _ = self.policy.ctrl.sample(
              obs_all, append=None, latent=None
          )
          dstb_all = torch.empty(size=(self.n_envs, env.action_dim_dstb)
                                ).to(self.device)
          action_all = []
          for i, dstb_sample_fn in enumerate(dstb_sample_fn_all):
            if self.policy.dstb_use_ctrl:
              obs_dstb = torch.cat((obs_all[i], ctrl_all[i]), dim=-1)
            else:
              obs_dstb = obs_all[i]
            dstb: torch.Tensor = dstb_sample_fn(
                obs_dstb, append=None, latent=None
            )[0]
            dstb_all[i] = dstb
            action_all.append({
                'ctrl': ctrl_all[i].cpu().numpy(),
                'dstb': dstb.cpu().numpy()
            })

      # Interacts with the env.
      if not single_env:
        obs_nxt_all, r_all, done_all, info_all = venv.step(action_all)
      else:
        obs_nxt, r, done, info = venv.step(action_all[0])
        obs_nxt_all = torch.FloatTensor(np.stack([obs_nxt])).to(self.device)
        r_all = torch.FloatTensor(np.stack([r])).unsqueeze(dim=1).float()
        done_all = np.stack([done])
        info_all = [info]

      for env_idx, (done, info) in enumerate(zip(done_all, info_all)):
        # Stores the transition in memory.
        self.store_transition(
            obs_all[[env_idx]], ctrl_all[[env_idx]], dstb_all[[env_idx]],
            r_all[env_idx], obs_nxt_all[[env_idx]], done, info
        )

        if done:
          if not single_env:
            obs = venv.reset_one(index=env_idx)
          else:
            obs = venv.reset(cast_torch=True, **reset_kwargs)
          obs_nxt_all[env_idx] = obs
          g_x = info['g_x']
          if g_x > 0:
            cnt_safety_violation += 1
          cnt_num_episode += 1
          dstb_sample_fn_all[env_idx] = self.get_dstb_sample_fn(dstb_folder)
      violation_record.append(cnt_safety_violation)
      episode_record.append(cnt_num_episode)
      obs_all = obs_nxt_all

      # Optimizes NNs and checks performance.
      if (self.cnt_step >= min_step_b4_opt and cnt_opt_period >= opt_freq):
        print(f"Updates at sample step {self.cnt_step}")
        cnt_opt_period = 0

        # Updates critic and/or actor.
        if self.fix_ctrl:
          update_ctrl = False
          update_dstb = True
        elif self.fix_dstb:
          update_ctrl = True
          update_dstb = False
        else:
          if self.cnt_step > warmup_critic_steps:
            update_dstb = True
            if cnt_dstb_opt == (ctrl_opt_freq - 1):
              update_ctrl = True
              cnt_dstb_opt = 0
            else:
              update_ctrl = False
              cnt_dstb_opt += 1
          else:
            update_ctrl = False
            update_dstb = False
        loss = self.update(num_update_per_opt_list, update_ctrl, update_dstb)
        train_record.append(loss)
        cnt_opt += 1  # Counts number of optimization.
        if self.CONFIG.USE_WANDB:
          # self.policy.critic_optimizer.state_dict()['param _groups'][0]['lr']
          log_dict = {
              "cnt_safety_violation": cnt_safety_violation,
              "cnt_num_episode": cnt_num_episode,
              "gamma": self.policy.GAMMA,
              "loss/q_adv": loss[0],
              "loss/q_mean": loss[1]
          }
          if update_dstb:
            log_dict["loss/dstb"] = loss[2]
            log_dict["loss/ent_dstb"] = loss[3]
            log_dict["loss/alpha_dstb"] = loss[4]
            log_dict["alpha_dstb"] = self.policy.alpha[1]
          if update_ctrl:
            log_dict["loss/ctrl"] = loss[5]
            log_dict["loss/ent_ctrl"] = loss[6]
            log_dict["loss/alpha_ctrl"] = loss[7]
            log_dict["alpha_ctrl"] = self.policy.alpha[0]
          wandb.log(log_dict, step=self.cnt_step, commit=False)

        # Checks after fixed number of gradient updates.
        if cnt_opt % check_opt_freq == 0 or first_update:
          print(f"Checks at sample step {self.cnt_step}:")
          cnt_opt = 0
          first_update = False
          # Updates the agent in the environment with the newest policy.
          env.agent.policy.update_policy(self.policy.ctrl)
          if not single_env:
            agent_list = venv.get_attr(attr_name='agent')
          else:
            agent_list = [env.agent]

          for agent in agent_list:
            agent: Agent
            agent.policy.update_policy(self.policy.ctrl)

          # Has gameplays with all stored dstb checkpoints.
          reset_kwargs_list = []  # Same initial states.
          for _ in range(self.CONFIG.NUM_EVAL_TRAJ):
            env.reset()
            reset_kwargs_list.append({"state": np.copy(env.state)})

          if self.fix_ctrl or self.fix_dstb:
            save_dict = self.save_best_response(
                venv, force_save=False, reset_kwargs_list=reset_kwargs_list,
                action_kwargs_list=action_kwargs,
                rollout_step_callback=rollout_step_callback,
                rollout_episode_callback=rollout_episode_callback
            )
          else:
            save_dict = self.save(
                venv, dstb_folder=os.path.join(model_folder, 'dstb'),
                force_save=False, reset_kwargs_list=reset_kwargs_list,
                action_kwargs_list=action_kwargs,
                rollout_step_callback=rollout_step_callback,
                rollout_episode_callback=rollout_episode_callback,
                check_nom=self.CONFIG.CHECK_NOM
            )


          # Logs.
          print('  => Safety violations: {:d}'.format(cnt_safety_violation))
          if rollout_end_criterion == "reach-avoid":
            success_rate = save_dict['metric']          
            safe_rate = save_dict['safe_rate']
            save_dict.pop('metric')
            save_dict["success_rate"] = success_rate
            train_progress.append(np.array([success_rate, safe_rate]))
            print('  => Success rate: {:.2f}'.format(success_rate))
            if self.CONFIG.CHECK_NOM:
              success_rate_nom = save_dict['metric_nom']
              save_dict.pop('metric_nom')
              save_dict["success_rate_nominal"] = success_rate_nom
              print('  => Success rate (nom): {:.2f}'.format(success_rate_nom))
            print('  => Safe rate: {:.2f}'.format(safe_rate))
          elif rollout_end_criterion == "failure":
            safe_rate = save_dict['metric']
            save_dict.pop('metric')
            #! safety policy here is a nominal policy that we want to test.
            save_dict["safe_rate"] = safe_rate
            train_progress.append(np.array([safe_rate]))
            print('  => Safe rate: {:.2f}'.format(safe_rate))
            if self.CONFIG.CHECK_NOM:
              safe_rate_nom = save_dict['metric_nom']
              save_dict.pop('metric_nom')
              save_dict["safe_rate_nominal"] = safe_rate_nom
              print('  => Safe rate (nom): {:.2f}'.format(safe_rate_nom))
          else:
            raise ValueError(f"Invalid end criterion {rollout_end_criterion}!")

          if self.CONFIG.USE_WANDB:
            wandb.log(save_dict, step=self.cnt_step, commit=True)
          torch.save({
              'train_record': train_record,
              'train_progress': train_progress,
              'violation_record': violation_record,
          }, os.path.join(out_folder, 'train_details'))

          # Visualizes.
          if visualize_callback is not None:
            visualize_callback(
                env, self.policy,
                os.path.join(figure_folder, str(self.cnt_step))
            )

          # Resets anyway.
          if not single_env:
            obs_all = venv.reset()
          else:
            obs = venv.reset(cast_torch=True, **reset_kwargs)
            obs_all = torch.FloatTensor(np.stack([obs])).to(self.device)

          dstb_sample_fn_all = [
              self.get_dstb_sample_fn(dstb_folder, verbose=True)
              for _ in range(self.n_envs)
          ]

      # Updates counter.
      self.cnt_step += self.n_envs
      cnt_opt_period += self.n_envs

      # Updates gamma, lr, etc.
      for _ in range(self.n_envs):
        self.policy.update_hyper_param()

    if self.fix_ctrl or self.fix_dstb:
      self.save_best_response(venv, force_save=True)
    else:
      self.save(venv, os.path.join(model_folder, 'dstb'), force_save=True)
    end_learning = time.time()
    time_learning = end_learning - start_learning
    print('\nLearning: {:.1f}'.format(time_learning))

    train_record = np.array(train_record)
    train_progress = np.array(train_progress)
    violation_record = np.array(violation_record)
    episode_record = np.array(episode_record)
    return (
        train_record, train_progress, violation_record, episode_record,
        self.pq_top_k
    )

  def update(
      self, num_update_per_opt_list: np.ndarray, update_ctrl: bool,
      update_dstb: bool
  ) -> np.ndarray:
    """Updates neural networks.

    Args:
        num_update_per_opt_list (np.ndarray): the number of NN updates per
            optimization. The first entry is for dstb, while the second one is
            for controller.
        update_ctrl (bool): updates controller in this optimization if True.

    Returns:
        np.ndarray: losses.
    """
    # q_adv, q_mean, pi_dstb, ent_dstb, alp_dstb, pi_ctrl, ent_ctrl, alp_ctrl
    loss = np.zeros(8)

    if update_dstb:
      n_update_dstb_critic = num_update_per_opt_list[1]
      n_update_dstb_actor = (
          num_update_per_opt_list[1] / self.policy.update_period[1]
      )
      for timer in range(n_update_dstb_critic):
        batch = self.unpack_batch(self.sample_batch())
        loss_tp = self.policy.update(
            batch, timer, update_ctrl=False, update_dstb=True
        )
        for i in [0, 2, 3, 4]:
          loss[i] += loss_tp[i]
      loss[2:5] /= n_update_dstb_actor

    if update_ctrl:
      n_update_ctrl_critic = num_update_per_opt_list[0]
      n_update_ctrl_actor = (
          num_update_per_opt_list[0] / self.policy.update_period[0]
      )
      for timer in range(n_update_ctrl_critic):
        batch = self.unpack_batch(self.sample_batch())
        loss_tp = self.policy.update(
            batch, timer, update_ctrl=True, update_dstb=False
        )
        for i, l in zip([0, 1, 5, 6, 7], loss_tp):
          loss[i] += l
      loss[1] /= n_update_ctrl_critic
      loss[5:] /= n_update_ctrl_actor

    if (not update_dstb) and (not update_ctrl):  # updates critic only.
      n_update_critic = num_update_per_opt_list[1]  #! defaults to dstb.
      for timer in range(n_update_critic):
        batch = self.unpack_batch(self.sample_batch())
        loss_tp = self.policy.update(
            batch, timer, update_ctrl=False, update_dstb=False
        )
        loss[0] += loss_tp[0]

    if update_dstb:
      if update_ctrl:
        loss[0] /= (n_update_ctrl_critic + n_update_dstb_critic)
      else:
        loss[0] /= n_update_dstb_critic
    else:
      if update_ctrl:
        loss[0] /= n_update_ctrl_critic
      else:
        loss[0] /= n_update_critic
    return loss

  def restore(self, step: int, model_folder: str, **kwargs):
    super().restore(step, model_folder, **kwargs)

  def save(
      self, venv: VecEnvBase, dstb_folder: str, force_save: bool = False,
      reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None,
      check_nom: bool = False
  ) -> Dict:
    """
    This saving utils is called if both actors are trained. Within this
    function, we play games between all dstb checkpoints and the current ctrl.
    """
    save_ctrl = False
    save_dstb = False
    if force_save:
      save_ctrl = True
      save_dstb = True
      info = None
    else:
      num_eval_traj = self.CONFIG.NUM_EVAL_TRAJ
      eval_timeout = self.CONFIG.EVAL_TIMEOUT
      rollout_end_criterion = self.CONFIG.ROLLOUT_END_CRITERION
      history_weight = self.CONFIG.HISTORY_WEIGHT

      steps = list(self.dstb_res_dict.keys())
      steps.append(self.cnt_step)

      metrics_weighted = []
      metrics_cur = []
      info_list = []
      for step in steps:
        adv_fn_list = []
        for _ in range(self.n_envs):
          if step == self.cnt_step:
            dstb_policy = copy.deepcopy(self.policy.dstb)
            dstb_policy.to(torch.device("cpu"))
            adv_fn_list.append(
                partial(
                    get_adversary, dstb_policy=dstb_policy,
                    use_ctrl=self.policy.dstb_use_ctrl
                )
            )
          elif step == -1:
            adv_fn_list.append(
                partial(dummy_dstb, dim=self.policy.action_dim_dstb)
            )
          else:
            restore_model(
                self.policy_sample.dstb, self.policy_sample.device, step=step,
                model_folder=dstb_folder, types='actor'
            )
            dstb_policy = copy.deepcopy(self.policy_sample.dstb)
            dstb_policy.to(torch.device("cpu"))
            adv_fn_list.append(
                partial(
                    get_adversary, dstb_policy=dstb_policy,
                    use_ctrl=self.policy.dstb_use_ctrl
                )
            )
        
        #! TODO: BAD CODING, NEED TO FIGURE OUT HOW TO IMPLEMENT simulate_trajectories_zs FOR SPIRIT ENV
        single_env = True
        if not single_env: # DIRTY WORKAROUND FOR SPIRIT ENV
          _, results, length = venv.simulate_trajectories_zs(
              num_trajectories=num_eval_traj, T_rollout=eval_timeout,
              end_criterion=rollout_end_criterion, adversary=adv_fn_list,
              reset_kwargs_list=reset_kwargs_list,
              action_kwargs_list=action_kwargs_list,
              rollout_step_callback=rollout_step_callback,
              rollout_episode_callback=rollout_episode_callback
          )
        else:
          _, results, length = venv.simulate_trajectories(
              num_trajectories=num_eval_traj, T_rollout=eval_timeout,
              end_criterion=rollout_end_criterion, adversary=adv_fn_list[0],
              reset_kwargs_list=reset_kwargs_list,
              action_kwargs_list=action_kwargs_list,
              rollout_step_callback=rollout_step_callback,
              rollout_episode_callback=rollout_episode_callback
          )
        if check_nom:
          _, results_nom, _ = venv.simulate_trajectories_zs(
              num_trajectories=num_eval_traj, T_rollout=eval_timeout,
              end_criterion=rollout_end_criterion, adversary=adv_fn_list,
              reset_kwargs_list=reset_kwargs_list,
              action_kwargs_list={'policy_type': 'safety'}
          )
          if rollout_end_criterion == "reach-avoid":
            metric_nom = np.sum(results_nom == 1) / num_eval_traj
          else:
            metric_nom = np.sum(results_nom != -1) / num_eval_traj

        del adv_fn_list
        safe_rate = np.sum(results != -1) / num_eval_traj
        if rollout_end_criterion == "reach-avoid":
          success_rate = np.sum(results == 1) / num_eval_traj
          metric = success_rate
          _info = dict(
              safe_rate=safe_rate, ep_length=np.mean(length), metric=metric
          )
        else:
          metric = safe_rate
          _info = dict(ep_length=np.mean(length), metric=metric)
        if check_nom:
          _info['metric_nom'] = metric_nom

        if step in self.dstb_res_dict:
          _n_gameplays, _metric_avg = self.dstb_res_dict[step]
          n_gameplays = _n_gameplays + 1
          metric_avg = (history_weight*_n_gameplays*_metric_avg
                        + metric) / (history_weight*_n_gameplays + 1)
          self.dstb_res_dict[step] = (n_gameplays, metric_avg)
          metrics_weighted.append(metric_avg)
        else:
          metrics_weighted.append(metric)
        info_list.append(_info)
        metrics_cur.append(metric)

      indices_weighted = np.argsort(np.array(metrics_weighted))
      # Gets the step that is not -1 and has the highest metric.
      for i in range(len(indices_weighted) - 1, -1, -1):
        step_rm_dstb = steps[indices_weighted[i]]
        if step_rm_dstb != -1:
          break
      info = info_list[indices_weighted[0]]
      metric_lowest = np.min(np.array(metrics_cur))
      print("  => Gameplays results:")
      print("     ", end='')
      for k, v in zip(steps, metrics_weighted):
        print(k, end=': ')
        print(v, end=" | ")
      print()

      if len(self.dstb_res_dict) < self.save_top_k[1] + 1:
        save_dstb = True
        self.dstb_res_dict[self.cnt_step] = (1, metrics_weighted[-1])
      elif step_rm_dstb != self.cnt_step:
        if step_rm_dstb == -1:  # cannot remove dummy.
          step_rm_dstb = steps[indices_weighted[-2]]
        save_dstb = True
        self.dstb_res_dict[self.cnt_step] = (1, metrics_weighted[-1])
        for module, module_folder in zip(
            self.module_all, self.module_folder_all
        ):
          module.remove(
              int(step_rm_dstb), module_folder, rm_dstb=True, rm_ctrl=False,
              rm_critic=False
          )
        self.dstb_res_dict.pop(step_rm_dstb)

      if self.pq_top_k.qsize() < self.save_top_k[0]:
        self.pq_top_k.put((metric_lowest, self.cnt_step))
        save_ctrl = True
      elif metric_lowest > self.pq_top_k.queue[0][0]:
        self.pq_top_k.put((metric_lowest, self.cnt_step))
        save_ctrl = True
        _, step_rm_ctrl = self.pq_top_k.get()
        for module, module_folder in zip(
            self.module_all, self.module_folder_all
        ):
          module.remove(
              int(step_rm_ctrl), module_folder, rm_dstb=False, rm_ctrl=True,
              rm_critic=True
          )

    if save_dstb or save_ctrl:
      for module, module_folder in zip(
          self.module_all, self.module_folder_all
      ):
        module.save(
            self.cnt_step, module_folder, self.MAX_MODEL, save_dstb=save_dstb,
            save_ctrl=save_ctrl, save_critic=save_ctrl
        )
      if save_ctrl:
        print("  => priority queue:", self.pq_top_k.queue)
    return info

  def save_best_response(
      self, venv: VecEnvBase, force_save: bool = False,
      reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None
  ) -> Dict:
    """
    This saving utils is called if one of the actors is fixed. We only keep a
    single heap.
    """
    if force_save:
      save_ctrl = not self.fix_ctrl
      save_dstb = not self.fix_dstb
      info = None
    else:
      num_eval_traj = self.CONFIG.NUM_EVAL_TRAJ
      eval_timeout = self.CONFIG.EVAL_TIMEOUT
      rollout_end_criterion = self.CONFIG.ROLLOUT_END_CRITERION
      save_ctrl = False
      save_dstb = False
      if self.fix_ctrl:
        dstb_policy = copy.deepcopy(self.policy.dstb)
      elif self.fix_dstb:
        dstb_policy = copy.deepcopy(self.policy_sample.dstb)
      dstb_policy.to(torch.device("cpu"))
      if self.use_random_dstb:
        adv_fn_list = [
            partial(
                random_dstb, rng=copy.deepcopy(self.rng),
                dstb_range=self.policy.action_range[1]
            ) for _ in range(self.n_envs)
        ]
      else:
        adv_fn_list = [
            partial(
                get_adversary, dstb_policy=copy.deepcopy(dstb_policy),
                use_ctrl=self.policy.dstb_use_ctrl
            ) for _ in range(self.n_envs)
        ]
      _, results, length = venv.simulate_trajectories_zs(
          num_trajectories=num_eval_traj, T_rollout=eval_timeout,
          end_criterion=rollout_end_criterion, adversary=adv_fn_list,
          reset_kwargs_list=reset_kwargs_list,
          action_kwargs_list=action_kwargs_list,
          rollout_step_callback=rollout_step_callback,
          rollout_episode_callback=rollout_episode_callback
      )

      safe_rate = np.sum(results != -1) / num_eval_traj
      if rollout_end_criterion == "reach-avoid":
        success_rate = np.sum(results == 1) / num_eval_traj
        metric = success_rate
        info = dict(
            safe_rate=safe_rate, ep_length=np.mean(length), metric=metric
        )
      else:
        metric = safe_rate
        info = dict(ep_length=np.mean(length), metric=metric)

      if self.fix_ctrl:
        # Dstb wants to maximize (1-metric).
        if self.pq_top_k.qsize() < self.save_top_k[0]:
          self.pq_top_k.put((1 - metric, self.cnt_step))
          save_dstb = True
        elif 1 - metric > self.pq_top_k.queue[0][0]:
          self.pq_top_k.put((1 - metric, self.cnt_step))
          save_dstb = True
          _, step_rm_dstb = self.pq_top_k.get()
          for module, module_folder in zip(
              self.module_all, self.module_folder_all
          ):
            module.remove(
                int(step_rm_dstb), module_folder, rm_dstb=True, rm_ctrl=False,
                rm_critic=True
            )
      elif self.fix_dstb:
        # Ctrl wants to maximize metric.
        if self.pq_top_k.qsize() < self.save_top_k[0]:
          self.pq_top_k.put((metric, self.cnt_step))
          save_ctrl = True
        elif metric > self.pq_top_k.queue[0][0]:
          self.pq_top_k.put((metric, self.cnt_step))
          save_ctrl = True
          _, step_rm_ctrl = self.pq_top_k.get()
          for module, module_folder in zip(
              self.module_all, self.module_folder_all
          ):
            module.remove(
                int(step_rm_ctrl), module_folder, rm_dstb=False, rm_ctrl=True,
                rm_critic=True
            )

    if self.fix_dstb:
      assert not save_dstb
    if self.fix_ctrl:
      assert not save_ctrl

    if save_ctrl or save_dstb:
      for module, module_folder in zip(
          self.module_all, self.module_folder_all
      ):
        module.save(
            self.cnt_step, module_folder, self.MAX_MODEL, save_dstb=save_dstb,
            save_ctrl=save_ctrl, save_critic=True
        )
      print("  => priority queue:", self.pq_top_k.queue)
    return info

  def get_dstb_sample_fn(
      self, dstb_folder: str, verbose: bool = False,
      dstb_sample_type: Optional[str] = None
  ) -> Callable[[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
                Tuple[torch.Tensor, Any]]:
    """
    Gets a disturbance sampler. This can be: (1) sampling from the checkpoints,
    (2) picking the one has the highest failure rate in the last checking
    round, (3) using the current one, (4) no disturbance, or (5) sampling
    randomly from the disturbance set.

    Args:
        dstb_folder (str): path to dstb checkpoints.
        verbose (bool, optional): Defaults to False.
        dstb_sample_type (Optional[str], optional): Defaults to None.

    Returns:
        Callable: a function to sample disturbances from.
    """
    if self.fix_ctrl or self.fix_dstb:
      step_to_restore = self.cnt_step  # always uses the current dstb.
    else:
      if dstb_sample_type is None:
        dstb_sample_type = self.CONFIG.DSTB_SAMPLE_TYPE
      dstb_sample_cur_weight = self.CONFIG.DSTB_SAMPLE_CUR_WEIGHT

      if dstb_sample_type == "recent":
        step_to_restore = self.cnt_step
      else:
        if dstb_sample_type == "strongest":
          lowest_metric = float("inf")
          for k, v in self.dstb_res_dict.items():
            if v[1] < lowest_metric:
              step_to_restore = k
        elif dstb_sample_type == "softmax":
          dstb_sample_sm_weight = getattr(
              self.CONFIG, "DSTB_SAMPLE_SM_WEIGHT", 5.
          )
          steps = []
          metrics = []
          for k, v in self.dstb_res_dict.items():
            steps.append(k)
            metrics.append(1 - v[1])
          steps.append(self.cnt_step)
          metrics = np.array(metrics)
          e_x: np.ndarray = np.exp(
              dstb_sample_sm_weight * (metrics - np.max(metrics))
          )
          probs = (e_x / e_x.sum()) * (1-dstb_sample_cur_weight)
          probs = np.append(probs, dstb_sample_cur_weight)
          step_to_restore = self.rng.choice(steps, p=probs)
        else:
          raise ValueError(
              f"DSTB_SAMPLE_TYPE ({dstb_sample_type}) is not supported!"
          )

    if step_to_restore == -1:
      dstb_sample_fn = self.dummy_dstb_sample
      if verbose:
        print("  => Uses dummy disturbance sampler.")
    elif step_to_restore == self.cnt_step:
      if self.use_random_dstb:
        dstb_sample_fn = self.random_dstb_sample
        if verbose:
          print("  => Uses domain randomization.")
      else:
        dstb_sample_fn = self.policy.dstb.sample
        if verbose:
          print("  => Uses the current disturbance sampler.")
    else:
      restore_model(
          self.policy_sample.dstb, self.policy_sample.device,
          step=step_to_restore, model_folder=dstb_folder, types='actor'
      )
      dstb_sample_fn = self.policy_sample.dstb.sample
      if verbose:
        print(f"  => Uses disturbance sampler from {step_to_restore}.")

    return dstb_sample_fn

  def evaluate(self, env, **kwargs):
    video = None
    step = None
    iteration = 0

    get_traj_data = False
    single_run = True
    state_array = []
    action_array = []
    
    if "video" in kwargs.keys():
      video = kwargs["video"]
    if "step" in kwargs.keys():
      step = kwargs["step"]
    if "get_traj_data" in kwargs.keys():
      get_traj_data = kwargs["get_traj_data"]
    if "single_run" in kwargs.keys():
      single_run = kwargs["single_run"]
    
    if video:
      video_output_file = "safety_action_only_{}".format(iteration) + ".avi"
    else:
      video_output_file = None

    s = env.reset(cast_torch = True, video_output_file=video_output_file, **kwargs).to(self.device)
    
    if get_traj_data:
      state_array.append(s)
      action_array.append(None)

    counter = 0
    while True:
      ctrl = self.policy.ctrl(s.float().to(self.device))
      s_dstb = torch.cat((s, ctrl), dim=-1)
      dstb = self.policy.dstb(s_dstb.float().to(self.device))

      critic_q = max(
          self.policy.adv_critic(s.float().to(self.device), ctrl.float().to(self.device), dstb.float().to(self.device))
      )

      # print("\r{:.3f}".format(float(critic_q.cpu().detach().numpy())), end="")

      a = {
        'ctrl': ctrl.cpu().detach().numpy(), 
        'dstb': dstb.cpu().detach().numpy()
      }
      s_, r, done, info = env.step(a, cast_torch=True)
      s = s_.to(self.device)

      if get_traj_data:
        state_array.append(s)
        action_array.append(a)

      counter += 1

      if step is not None:
        if counter > step:
          if get_traj_data:
            return done, info, env, state_array, action_array
          else:
            return done, info, env
      
      if done:
        if p.getKeyboardEvents().get(49):
          continue
        else:
          if single_run:
            if get_traj_data:
              return done, info, env, state_array, action_array
            else:
              return done, info, env

          counter = 0
          iteration += 1
            
          if video:
            video_output_file = "safety_action_only_{}".format(iteration) + ".avi"
          else:
            video_output_file = None

          s = env.reset(cast_torch=True, video_output_file=video_output_file, **kwargs)
          
          if get_traj_data:
            state_array = [s]
            action_array = [None]
  
  def shielding_with_IK(self, env, controller, epsilon=0.25, override=True, **kwargs):
    rollout = False
    env_imaginary = None
    rollout_step = None
    eval_horizon = 1000
    imaginary_solver = None
    
    if "rollout" in kwargs.keys():
      rollout = kwargs["rollout"]
    if "env_imaginary" in kwargs.keys():
      env_imaginary = kwargs["env_imaginary"]
      # ROLLOUT #2: trace env_imaginary wrt env
      """
      rollout_state = env_imaginary.reset(cast_torch = True, **kwargs)
      """
    if "rollout_step" in kwargs.keys():
      rollout_step = kwargs["rollout_step"]
    if "eval_horizon" in kwargs.keys():
      eval_horizon = kwargs["eval_horizon"]

    if "imaginary_solver" in kwargs.keys():
      imaginary_solver = kwargs["imaginary_solver"]

    verbose = False
    shield_count = 0
    max_q = -np.inf
    # state from SACRA
    state = env.reset(cast_torch = True, **kwargs).to(self.device)

    # ROLLOUT #3: reset env_imaginary and replay all ctrl-dstb sequences so far
    ctrl_array = []
    dstb_array = []

    for i in range(eval_horizon):
      # ROLLOUT #3: reset env_imaginary and replay all ctrl-dstb sequences so far
      if env_imaginary is not None:
        rollout_state = env_imaginary.reset(cast_torch = True, **kwargs).to(self.device)
        for j in range(i):
          rollout_action = {
            'ctrl': ctrl_array[j].cpu().detach().numpy(), 
            'dstb': dstb_array[j].cpu().detach().numpy()
          }
          rollout_state, r, done, info = env_imaginary.step(rollout_action, cast_torch=True)
          rollout_state = rollout_state.to(self.device)

      new_joint_pos = controller.get_action()
      ctrl = torch.FloatTensor(new_joint_pos - np.array(env.agent.dyn.robot.get_joint_position())).to(self.device)

      s_dstb = torch.cat((state, ctrl), dim=-1)
      # dstb, _ = self.policy.dstb.sample(s_dstb, append=None, latent=None)
      dstb = self.policy.dstb(s_dstb.float().to(self.device))
      # dstb = self.disturbance_inference(state.cpu().detach().numpy())

      if not rollout:
        # check if action and state are safe
        # critic_q = max(
        #     self.policy.mean_critic(state.float().to(self.device), action.float().to(self.device))
        # )
        
        # TODO: check to see if this is correct
        if imaginary_solver is not None:
          critic_q = max(
              imaginary_solver.policy.adv_critic(state.float().to(self.device), ctrl.float().to(self.device), dstb.float().to(self.device))
          )
        else:
          critic_q = max(
              self.policy.adv_critic(state.float().to(self.device), ctrl.float().to(self.device), dstb.float().to(self.device))
          )
        max_q = max(critic_q.cpu().detach().numpy(), max_q)

        if critic_q > epsilon:
          # NOT GOOD, USE SHIELDING
          if override:
            shield_count+=1
            # action, _ = self.actor.sample(torch.from_numpy(state_sacra).float().to(self.device))
            if imaginary_solver is not None:
              ctrl = imaginary_solver.policy.ctrl(state.float().to(self.device))
            else:
              ctrl = self.policy.ctrl(state.float().to(self.device))
            s_dstb = torch.cat((state, ctrl), dim=-1)
            dstb = self.policy.dstb(s_dstb.float().to(self.device))
            
            if env.agent.dyn.gui:
              if env.agent.dyn.shielding_status_debug_text_id is not None:
                p.removeUserDebugItem(env.agent.dyn.shielding_status_debug_text_id)
              env.agent.dyn.shielding_status_debug_text_id = p.addUserDebugText("SHIELDED", (0.0, 0.0, 0.5), (100, 0, 0), 1.5, parentObjectUniqueId = env.agent.dyn.robot.id, physicsClientId = env.agent.dyn.client)
          
          if verbose:
            print(
                "\rStep: {}\tQ: {:.3f}\tSHIELDED!           ".format(str(i).zfill(3), float(critic_q.cpu().detach().numpy())
                ), end=""
            )
        else:
          # GOOD, CONTINUE WITH THE ACTION CHOICE FROM PERFORMANCE
          if override and env.agent.dyn.gui:
            if env.agent.dyn.shielding_status_debug_text_id is not None:
                p.removeUserDebugItem(env.agent.dyn.shielding_status_debug_text_id)
            env.agent.dyn.shielding_status_debug_text_id = p.addUserDebugText("GOOD", (0.0, 0.0, 0.5), (0, 100, 0), 1.5, parentObjectUniqueId = env.agent.dyn.robot.id, physicsClientId = env.agent.dyn.client)

          if verbose:
            print(
                "\rStep: {}\tQ: {:.3f}\t                    ".format(str(i).zfill(3), float(critic_q.cpu().detach().numpy())
                ), end=""
            )
            
        action = {
          'ctrl': ctrl.cpu().detach().numpy(), 
          'dstb': dstb.cpu().detach().numpy()
        }
        state, r, done, info = env.step(action, cast_torch=True)
        state = state.to(self.device)
      else:
        """
        Rollout shielding

        For each of the step of the main env, replicate the state of current env to env_imaginary
        In env_imaginary, assuming we accept the next performance action
        Apply the performance action in env_imaginary, and proceed to take safety action in the next n steps
        Check g_x and l_x. If g_x > 0 --> rollout failed. If g_x <= 0 and l_x <= 0 --> rollout successful
        Go back to main env. If rollout failed, apply safety action. If not, apply performance action
        """
        max_q = -np.inf
        # reset env_imaginary with env information
        # current_state = env.agent.dyn.state

        # ROLLOUT #1: manual copy state of env to env_imaginary
        """
        current_pos, current_ang = p.getBasePositionAndOrientation(env.agent.dyn.robot.id, physicsClientId = env.agent.dyn.client)
        rollout_kwargs = kwargs.copy()
        rollout_kwargs["initial_height"] = current_pos[2]
        rollout_kwargs["initial_rotation"] = current_ang
        rollout_state = env_imaginary.reset(cast_torch = True, ox = current_pos[0], oy = current_pos[1], is_rollout_shielding_reset = True, **rollout_kwargs)
        
        for joint in range(p.getNumJoints(env.agent.dyn.robot.id, physicsClientId = env.agent.dyn.client)):
            current_joint_position, current_joint_velocity, _, _ = p.getJointState(env.agent.dyn.robot.id, joint, physicsClientId = env.agent.dyn.client)
            p.resetJointState(env_imaginary.agent.dyn.robot.id, joint, current_joint_position, targetVelocity = current_joint_velocity, physicsClientId = env_imaginary.agent.dyn.client)
          
        current_linear_velocity, current_angular_velocity = p.getBaseVelocity(env.agent.dyn.robot.id, physicsClientId = env.agent.dyn.client)
        p.resetBaseVelocity(env_imaginary.agent.dyn.robot.id, linearVelocity = current_linear_velocity, angularVelocity = current_angular_velocity, physicsClientId = env_imaginary.agent.dyn.client)
        """

        # ROLLOUT #2: trace env_imaginary wrt env
        """
        # save state of env_imaginary before each rollout run, the rollback to the saved state
        rollback_state_id = p.saveState(physicsClientId = env_imaginary.agent.dyn.client)
        """

        if imaginary_solver is not None:
          rollout_ctrl = copy.copy(ctrl)
          rollout_s_dstb = torch.cat((rollout_state, rollout_ctrl), dim=-1)
          rollout_dstb = imaginary_solver.policy.dstb(rollout_s_dstb.float().to(self.device))
          rollout_action = {
            'ctrl': rollout_ctrl.cpu().detach().numpy(), 
            'dstb': rollout_dstb.cpu().detach().numpy()
          }
        else:
          # apply the performance action
          rollout_action = {
            'ctrl': ctrl.cpu().detach().numpy(), 
            'dstb': dstb.cpu().detach().numpy()
          }
        rollout_state, r, done, info = env_imaginary.step(rollout_action, cast_torch=True)
        rollout_state = rollout_state.to(self.device)

        if info["g_x"] <= 0:
          rollout_failed = False
        else:
          rollout_failed = True

        if not rollout_failed:
          for j in range(rollout_step):
            if imaginary_solver is not None:
              rollout_ctrl = imaginary_solver.policy.ctrl(rollout_state.float().to(self.device))
              rollout_s_dstb = torch.cat((rollout_state, rollout_ctrl), dim=-1)
              rollout_dstb = imaginary_solver.policy.dstb(rollout_s_dstb.float().to(self.device))
            else:
              rollout_ctrl = self.policy.ctrl(rollout_state.float().to(self.device))
              rollout_s_dstb = torch.cat((rollout_state, rollout_ctrl), dim=-1)
              rollout_dstb = self.policy.dstb(rollout_s_dstb.float().to(self.device))

            rollout_action = {
              'ctrl': rollout_ctrl.cpu().detach().numpy(), 
              'dstb': rollout_dstb.cpu().detach().numpy()
            }
            rollout_state, r, done, info = env_imaginary.step(rollout_action, cast_torch=True)
            rollout_state = rollout_state.to(self.device)
        
            # TODO: apply end criterion logic
            if info["g_x"] > 0:
              rollout_failed = True
              break
            # elif info["g_x"] <= 0 and info["l_x"] <= 0:

        if rollout_failed:
          if override:
            shield_count+=1
            if imaginary_solver is not None:
              ctrl = imaginary_solver.policy.ctrl(state.float().to(self.device))
            else:
              ctrl = self.policy.ctrl(state.float().to(self.device))
            s_dstb = torch.cat((state, ctrl), dim=-1)
            dstb = self.policy.dstb(s_dstb.float().to(self.device))
            
            if env.agent.dyn.gui:
              if env.agent.dyn.shielding_status_debug_text_id is not None:
                p.removeUserDebugItem(env.agent.dyn.shielding_status_debug_text_id)
              env.agent.dyn.shielding_status_debug_text_id = p.addUserDebugText("SHIELDED", (0.0, 0.0, 0.5), (100, 0, 0), 1.5, parentObjectUniqueId = env.agent.dyn.robot.id, physicsClientId = env.agent.dyn.client)
          
          if verbose:
            print(
                "\rStep: {}\tQ: rollout\tSHIELDED!           ".format(str(i).zfill(3)
                ), end=""
            )
        else:
          if override and env.agent.dyn.gui:
            if env.agent.dyn.shielding_status_debug_text_id is not None:
                p.removeUserDebugItem(env.agent.dyn.shielding_status_debug_text_id)
            env.agent.dyn.shielding_status_debug_text_id = p.addUserDebugText("GOOD", (0.0, 0.0, 0.5), (0, 100, 0), 1.5, parentObjectUniqueId = env.agent.dyn.robot.id, physicsClientId = env.agent.dyn.client)
          
          if verbose:
            print(
                "\rStep: {}\tQ: rollout\t                    ".format(str(i).zfill(3)
                ), end=""
            )

        # ROLLOUT #2: trace env_imaginary wrt env (cont)
        # rollback
        """
        p.restoreState(rollback_state_id, physicsClientId = env_imaginary.agent.dyn.client)
        p.removeState(rollback_state_id, physicsClientId = env_imaginary.agent.dyn.client)
        
        for joint in range(p.getNumJoints(env.agent.dyn.robot.id, physicsClientId = env.agent.dyn.client)):
            current_joint_position, current_joint_velocity, _, _ = p.getJointState(env.agent.dyn.robot.id, joint, physicsClientId = env.agent.dyn.client)
            p.resetJointState(env_imaginary.agent.dyn.robot.id, joint, current_joint_position, targetVelocity = current_joint_velocity, physicsClientId = env_imaginary.agent.dyn.client)
        
        current_linear_velocity, current_angular_velocity = p.getBaseVelocity(env.agent.dyn.robot.id, physicsClientId = env.agent.dyn.client)
        p.resetBaseVelocity(env_imaginary.agent.dyn.robot.id, linearVelocity = current_linear_velocity, angularVelocity = current_angular_velocity, physicsClientId = env_imaginary.agent.dyn.client)
        """

        action = {
          'ctrl': ctrl.cpu().detach().numpy(), 
          'dstb': dstb.cpu().detach().numpy()
        }
        state, r, done, info = env.step(action, cast_torch=True)
        state = state.to(self.device)
        
        # ROLLOUT #3: reset env_imaginary and replay all ctrl-dstb sequences so far
        ctrl_array.append(ctrl)
        dstb_array.append(dstb)

        # ROLLOUT #2: trace env_imaginary wrt env (cont)
        # apply the same action
        """
        rollout_state, r, done, info = env_imaginary.step(action, cast_torch=True)
        """

      # Check if roll pitch are not too high
      error = False
      euler = p.getEulerFromQuaternion(env.agent.dyn.robot.linc_get_pos()[1])
      if ((abs(euler[1]) >= math.pi / 2) or (abs(euler[0]) >= math.pi / 2)):
        error = True

      # if done or error:
      if error:  # only care if the robot flips, do not care about the safety margin of safety policy
        return 0, np.linalg.norm(state[:2].cpu()), shield_count, max_q, i
    return 1, np.linalg.norm(state[:2].cpu()), shield_count, max_q, i

  def check_shielding(self, env, controller, epsilon=0.25, **kwargs):
    """
    Run 2 instances: one with shielding and one without shielding
    """
    result = {}
    distance = {}
    shield_count = {}
    max_q = {}
    steps = {}

    env.reset(cast_torch = True)
    terrain_data = env.agent.dyn.terrain_data
    initial_height = env.agent.dyn.initial_height
    initial_rotation = env.agent.dyn.initial_rotation
    initial_joint_value = env.agent.dyn.initial_joint_value

    for instance in ["no_shield", "shield"]:
      new_controller = copy.deepcopy(controller)
      if instance == "shield":
        result[instance], distance[instance], shield_count[instance], max_q[instance], steps[instance] = self.shielding_with_IK(env, new_controller, epsilon=epsilon, override=True, 
          terrain_data=terrain_data, initial_height=initial_height, initial_rotation=initial_rotation, initial_joint_value=initial_joint_value)
      else:
        result[instance], distance[instance], shield_count[instance], max_q[instance], steps[instance] = self.shielding_with_IK(env, new_controller, epsilon=epsilon, override=False, 
          terrain_data=terrain_data, initial_height=initial_height, initial_rotation=initial_rotation, initial_joint_value=initial_joint_value)
    return result, distance, shield_count, max_q, steps

  def check_shielding_rollout(self, env, env_imaginary, controller, epsilon=0.25, **kwargs):
    """
    Run 3 instances: rollout shielding, value shielding and no shielding
    """
    iteration = None
    video = None
    rollout_step = None
    batch_conditions = None
    batch_index = None
    eval_horizon = 1000

    imaginary_solver = None

    if "iteration" in kwargs.keys():
      iteration = kwargs["iteration"]
    if "video" in kwargs.keys():
      video = kwargs["video"]
    if "rollout_step" in kwargs.keys():
      rollout_step = kwargs["rollout_step"]
    if "batch_conditions" in kwargs.keys():
      batch_conditions = kwargs["batch_conditions"]
    if "batch_index" in kwargs.keys():
      batch_index = kwargs["batch_index"]
    if "eval_horizon" in kwargs.keys():
      eval_horizon = kwargs["eval_horizon"]
    if "imaginary_solver" in kwargs.keys():
      imaginary_solver = kwargs["imaginary_solver"]

    result = {}
    distance = {}
    shield_count = {}
    max_q = {}
    steps = {}

    if batch_conditions is not None:
      assert type(batch_conditions) is pd.core.frame.DataFrame, "Error: batch condition must be DataFrame type"
      if batch_index is not None:
        initial_conditions = batch_conditions.iloc[batch_index]
        print("\tHas batch_index flag, use batch index {}".format(batch_index))
      else:
        if iteration is not None:
          initial_conditions = batch_conditions.iloc[iteration]
          print("\tHas iteration flag, use iteration {}".format(iteration))
        else:
          print("Warning: No iteration is defined, will run the first condition in batch conditions")
          initial_conditions = batch_conditions.iloc[0]
      terrain_data = initial_conditions.terrain_data
      initial_height = initial_conditions.initial_height
      initial_rotation = initial_conditions.initial_rotation
      initial_joint_value = initial_conditions.initial_joint_value
    else:
      env.reset(cast_torch = True)
      terrain_data = env.agent.dyn.terrain_data
      initial_height = env.agent.dyn.initial_height
      initial_rotation = env.agent.dyn.initial_rotation
      initial_joint_value = env.agent.dyn.initial_joint_value

    for instance in ["no_shield", "shield", "rollout"]:
      if video:
        video_output_file = instance + "_{}".format(iteration) + ".avi"
      else:
        video_output_file = None

      new_controller = copy.deepcopy(controller)
      if instance == "rollout":
        result[instance], distance[instance], shield_count[instance], max_q[instance], steps[instance] = self.shielding_with_IK(
          env, new_controller, epsilon=epsilon, override=True, 
          terrain_data=terrain_data, initial_height=initial_height, initial_rotation=initial_rotation, initial_joint_value=initial_joint_value, 
          rollout=True, env_imaginary=env_imaginary, rollout_step=rollout_step, eval_horizon=eval_horizon, video_output_file=video_output_file, 
          imaginary_solver=imaginary_solver)
      elif instance == "shield":
        result[instance], distance[instance], shield_count[instance], max_q[instance], steps[instance] = self.shielding_with_IK(env, new_controller, epsilon=epsilon, override=True, 
          terrain_data=terrain_data, initial_height=initial_height, initial_rotation=initial_rotation, initial_joint_value=initial_joint_value,  
          eval_horizon=eval_horizon, video_output_file=video_output_file, imaginary_solver=imaginary_solver)
      elif instance == "no_shield":
        result[instance], distance[instance], shield_count[instance], max_q[instance], steps[instance] = self.shielding_with_IK(env, new_controller, epsilon=epsilon, override=False, 
          terrain_data=terrain_data, initial_height=initial_height, initial_rotation=initial_rotation, initial_joint_value=initial_joint_value,
          eval_horizon=eval_horizon, video_output_file=video_output_file, imaginary_solver=imaginary_solver)
    return result, distance, shield_count, max_q, steps