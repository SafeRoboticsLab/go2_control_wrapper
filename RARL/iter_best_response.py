# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu (kaichieh@princeton.edu)
#          Duy P. Nguyen (duyn@princeton.edu)

from multiprocessing.sharedctypes import Value
from typing import Optional, Callable, Dict, Tuple, Union, List, Any
import os
import copy
import time
import numpy as np
import torch
import wandb
from functools import partial

from .sac_adv import SAC_adv
from .base_zs_training import BaseZeroSumTraining
from .utils import restore_model
from simulators.base_zs_env import BaseZeroSumEnv
from simulators.vec_env.vec_env import VecEnvBase
from simulators.agent import Agent


# Multiprocessing requires functions on the top instead in self.
def dummy_dstb(obs, ctrl, append=None, dim: int = 0, **kwargs):
  return np.zeros(dim)


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


class IterBestResponse(BaseZeroSumTraining):
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
    self.save_top_k = np.array(self.CONFIG.SAVE_TOP_K)  # 0: ctrl, 1: dstb
    self.dstb_res_dict: Dict[int, Tuple] = {
        -1: (0, 0.)
    }  # key: step, value: (#gameplays, metric_avg)

    # alias
    self.module_all = [self.policy]
    self.performance = self.policy

  @property
  def has_backup(self):
    return False

  def dummy_dstb_sample(self, obs, append=None, latent=None, **kwargs):
    return torch.zeros(self.policy.action_dim_dstb).to(self.device), 0.

  def learn(
      self, env: BaseZeroSumEnv, current_step: Optional[int] = None,
      reset_kwargs: Optional[Dict] = None,
      action_kwargs: Optional[Dict] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None,
      visualize_callback: Optional[Callable] = None
  ):
    if reset_kwargs is None:
      reset_kwargs = {}

    # hyper-parameters
    max_steps: int = self.CONFIG.MAX_STEPS
    opt_freq: int = self.CONFIG.OPTIMIZE_FREQ
    num_update_per_opt_list = np.array(self.CONFIG.UPDATE_PER_OPT)
    check_opt_freq: int = self.CONFIG.CHECK_OPT_FREQ
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
      obs_nxt_all, r_all, done_all, info_all = venv.step(action_all)
      for env_idx, (done, info) in enumerate(zip(done_all, info_all)):
        # Stores the transition in memory.
        self.store_transition(
            obs_all[[env_idx]], ctrl_all[[env_idx]], dstb_all[[env_idx]],
            r_all[env_idx], obs_nxt_all[[env_idx]], done, info
        )

        if done:
          obs = venv.reset_one(index=env_idx)
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
        cnt_opt_period = 0

        # Updates critic and/or actor.
        if self.cnt_step > warmup_critic_steps:
          update_ctrl = self.in_ctrl_update_cycle()
          update_dstb = not update_ctrl
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
          agent_list = venv.get_attr(attr_name='agent')
          for agent in agent_list:
            agent: Agent
            agent.policy.update_policy(self.policy.ctrl)

          # Has gameplays with all stored dstb checkpoints.
          reset_kwargs_list = []  # Same initial states.
          for _ in range(self.CONFIG.NUM_EVAL_TRAJ):
            env.reset()
            reset_kwargs_list.append({"state": np.copy(env.state)})

          save_dict = self.save(
              venv, dstb_folder=dstb_folder, force_save=False,
              reset_kwargs_list=reset_kwargs_list,
              action_kwargs_list=action_kwargs,
              rollout_step_callback=rollout_step_callback,
              rollout_episode_callback=rollout_episode_callback
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
          obs_all = venv.reset()
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

    self.save(venv, force_save=True)
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

  def in_ctrl_update_cycle(self) -> str:
    ctrl_opt_freq: int = self.CONFIG.CTRL_OPT_FREQ
    min_step_b4_opt: int = self.CONFIG.MIN_STEPS_B4_OPT
    steps_per_cycle: int = self.CONFIG.STEPS_PER_CYCLE
    cycle_idx = int(
        np.floor(max(self.cnt_step - min_step_b4_opt, 0) / steps_per_cycle)
    )
    if (cycle_idx+1) % ctrl_opt_freq == 0:
      return True
    else:
      return False

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
      print(f"Updates DSTB at sample step {self.cnt_step}")
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
      print(f"Updates CTRL at sample step {self.cnt_step}")
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
      print(f"Updates CRITIC only at sample step {self.cnt_step}")
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

  def restore(self, step: int, model_folder: str):
    super().restore(step, model_folder, "performance")

  def save(
      self, venv: VecEnvBase, dstb_folder: Optional[str] = None,
      force_save: bool = False,
      reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None
  ) -> Dict:
    save_ctrl = False
    save_dstb = False
    if force_save:
      save_ctrl = True
      save_dstb = True
      info = None
    else:
      assert dstb_folder is not None
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

        _, results, length = venv.simulate_trajectories_zs(
            num_trajectories=num_eval_traj, T_rollout=eval_timeout,
            end_criterion=rollout_end_criterion, adversary=adv_fn_list,
            reset_kwargs_list=reset_kwargs_list,
            action_kwargs_list=action_kwargs_list,
            rollout_step_callback=rollout_step_callback,
            rollout_episode_callback=rollout_episode_callback
        )

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

  def get_dstb_sample_fn(
      self, dstb_folder: str, verbose: bool = False,
      dstb_sample_type: Optional[str] = None
  ) -> Callable[[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
                Tuple[torch.Tensor, Any]]:

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
