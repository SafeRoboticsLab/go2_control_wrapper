# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu (kaichieh@princeton.edu)
#          Duy P. Nguyen (duyn@princeton.edu)

from typing import Optional, Callable, Dict, Tuple, Union, List
import os
import copy
import time
import numpy as np
import torch
import wandb
from functools import partial

from .naive_rl import NaiveRL
from simulators.base_zs_env import BaseZeroSumEnv
from simulators.vec_env.vec_env import VecEnvBase
from simulators.agent import Agent

# TODO: makes this a best response training for both ctrl and dstb!


def get_adversary(
    obs: np.ndarray, ctrl: np.ndarray, dstb_policy=None, **kwargs
) -> np.ndarray:
  obs_dstb_all = np.concatenate((obs, ctrl), axis=-1)
  dstb = dstb_policy(obs_dstb_all, append=None)
  assert isinstance(dstb, np.ndarray)
  return dstb


def get_adversary_dummy(
    obs: np.ndarray, ctrl: np.ndarray, **kwargs
) -> np.ndarray:
  return np.zeros(5)


class NaiveDstb(NaiveRL):

  def __init__(
      self, CONFIG, CONFIG_UPDATE, CONFIG_ARCH, CONFIG_ENV, verbose=True
  ):
    super().__init__(CONFIG, CONFIG_UPDATE, CONFIG_ARCH, CONFIG_ENV, verbose)
    if self.policy.mode == "performance":
      assert self.policy.actor_type == 'min'
    else:
      assert self.policy.actor_type == 'max'

  def sample_action(
      self, obs_all: torch.Tensor, venv: VecEnvBase, min_step_b4_exploit: int,
      init_control: List[Optional[np.ndarray]]
  ) -> Tuple[Dict, torch.Tensor, np.ndarray]:
    action_dim_ctrl = venv.get_attr("action_dim_ctrl", indices=[0])[0]

    # Gets controls first.
    ctrl_all = np.empty(shape=(self.n_envs, action_dim_ctrl))
    state_all = venv.get_attr("state")
    agent_all = venv.get_attr("agent")
    method_args_list = []
    for i in range(venv.n_envs):
      method_args_list.append([
          obs_all[i].detach().cpu().numpy(), init_control[i], state_all[i]
      ])

    results = venv.env_method_arg("get_agent_action", method_args_list)
    for i in range(venv.n_envs):
      agent = agent_all[i]
      action = results[i][0]
      solver_info = results[i][1]
      if agent.policy.policy_type == "iLQR":
        init_control[i][:, :-1] = solver_info['controls'][:, 1:]
        init_control[i][:, -1] = 0.
      ctrl_all[i, :] = action

    # agent_all = venv.get_attr("agent")
    # state_all = venv.get_attr("state")
    # for i, (agent, state) in enumerate(zip(agent_all, state_all)):
    #   agent: Agent
    #   state: np.ndarray
    #   ctrl, solver_info = agent.get_action(
    #       obs=obs_all[i].detach().cpu().numpy(), controls=init_control[i],
    #       state=state.copy()
    #   )
    #   if agent.policy.policy_type == "iLQR":
    #     init_control[i][:, :-1] = solver_info['controls'][:, 1:]
    #     init_control[i][:, -1] = 0.
    #   ctrl_all[i, :] = ctrl

    # Gets disturbances.
    obs_dstb_all = torch.cat(
        (obs_all, torch.FloatTensor(ctrl_all).to(obs_all.device)), dim=-1
    )
    if self.cnt_step < min_step_b4_exploit:
      action_all = []
      action_space_all = venv.get_attr("action_space")
      for i, action_space in enumerate(action_space_all):
        _action_zs = action_space.sample()
        action_all.append({'ctrl': ctrl_all[i], 'dstb': _action_zs['dstb']})
    else:
      dstb_all, _ = self.policy.actor.sample(
          obs_dstb_all.to(self.device), append=None, latent=None
      )
      for i in range(self.n_envs):
        action_all.append({
            'ctrl': ctrl_all[i],
            'dstb': dstb_all[i].cpu().numpy()
        })
    return action_all, obs_dstb_all, init_control

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
    num_update_per_opt: int = self.CONFIG.UPDATE_PER_OPT
    check_opt_freq: int = self.CONFIG.CHECK_OPT_FREQ
    min_step_b4_opt: int = self.CONFIG.MIN_STEPS_B4_OPT
    min_step_b4_exploit: int = self.CONFIG.MIN_STEPS_B4_EXPLOIT

    out_folder: str = self.CONFIG.OUT_FOLDER
    num_eval_traj: int = self.CONFIG.NUM_EVAL_TRAJ
    eval_timeout: int = self.CONFIG.EVAL_TIMEOUT
    rollout_end_criterion: str = self.CONFIG.ROLLOUT_END_CRITERION

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
    os.makedirs(model_folder, exist_ok=True)
    figure_folder = os.path.join(out_folder, 'figure')
    os.makedirs(figure_folder, exist_ok=True)
    self.module_folder_all = [model_folder]

    if current_step is None:
      self.cnt_step = 0
    else:
      self.cnt_step = current_step
      print("starting from {:d} steps".format(self.cnt_step))

    reset_kwargs_list = []  # Same initial states.
    # Collects initial states from which the priori policy can maintain safe
    # without adversarial disturbances. --> Testing false safe/success rate.
    print("Collects initial states for eval ...", end=' ')
    while len(reset_kwargs_list) < num_eval_traj:
      env.reset()
      state = env.state.copy()
      if self.CONFIG.TEST_FALSE_NEG:
        _, result, _ = env.simulate_one_trajectory(
            T_rollout=eval_timeout, end_criterion='failure',
            adversary=get_adversary_dummy, reset_kwargs={"state": state}
        )
      else:
        result = 0  # adds to the list anyway.
      if result != -1:
        reset_kwargs_list.append({"state": state})
    print("DONE!!")

    venv = VecEnvBase([copy.deepcopy(env) for _ in range(self.n_envs)],
                      device=self.device)
    # Venv can run on different devices.
    agent_list = venv.get_attr(attr_name='agent')
    for agent in agent_list:
      agent: Agent
      if agent.policy is not None:
        agent.policy.to(torch.device(self.CONFIG.VENV_DEVICE))
      if agent.safety_policy is not None:
        agent.safety_policy.to(torch.device(self.CONFIG.VENV_DEVICE))

    obs_all = venv.reset()
    obs_dstb_prev_all = [None for _ in range(self.n_envs)]
    action_prev_all = [None for _ in range(self.n_envs)]
    r_prev_all = [None for _ in range(self.n_envs)]
    done_prev_all = [None for _ in range(self.n_envs)]
    info_prev_all = [None for _ in range(self.n_envs)]
    init_control = [None for _ in range(self.n_envs)]
    while self.cnt_step <= max_steps:
      # Selects action.
      with torch.no_grad():
        action_all, obs_dstb_all, init_control = self.sample_action(
            obs_all, venv, min_step_b4_exploit, init_control
        )

      # Interacts with the env.
      obs_nxt_all, r_all, done_all, info_all = venv.step(action_all)
      for env_idx, (done, info) in enumerate(zip(done_all, info_all)):
        if obs_dstb_prev_all[env_idx] is not None:
          # Stores the transition in memory.
          self.store_transition(
              obs_dstb_prev_all[env_idx].unsqueeze(0),
              action_prev_all[env_idx], r_prev_all[env_idx],
              obs_dstb_all[[env_idx]], done_prev_all[env_idx],
              info_prev_all[env_idx]
          )
        if done:
          obs = venv.reset_one(index=env_idx)
          obs_nxt_all[env_idx] = obs
          g_x = info['g_x']
          if g_x > 0:
            cnt_safety_violation += 1
          cnt_num_episode += 1

          # Stores this transition with zero controls.
          obs_dstb_next = torch.cat((
              obs_dstb_all[[env_idx]],
              torch.zeros(1, env.action_dim_ctrl).to(self.device)
          ), dim=-1)
          action = (
              torch.FloatTensor(action_all[env_idx]['dstb']
                               ).unsqueeze(0).to(self.device)
          )
          self.store_transition(
              obs_dstb_all[[env_idx]], action, r_all[env_idx], obs_dstb_next,
              done, info
          )
          obs_dstb_prev_all[env_idx] = None
          action_prev_all[env_idx] = None
          r_prev_all[env_idx] = None
          done_prev_all[env_idx] = None
          info_prev_all[env_idx] = None
        else:
          # Updates the temporary placeholder.
          obs_dstb_prev_all[env_idx] = obs_dstb_all[env_idx]
          action_prev_all[env_idx] = (
              torch.FloatTensor(action_all[env_idx]['dstb']
                               ).unsqueeze(0).to(self.device)
          )
          r_prev_all[env_idx] = r_all[env_idx]
          done_prev_all[env_idx] = done_all[env_idx]
          info_prev_all[env_idx] = info_all[env_idx]

      violation_record.append(cnt_safety_violation)
      episode_record.append(cnt_num_episode)
      obs_all = obs_nxt_all

      # Optimizes NNs and checks performance.
      if (self.cnt_step >= min_step_b4_opt and cnt_opt_period >= opt_freq):
        print(f"Updates at sample step {self.cnt_step}")
        cnt_opt_period = 0
        loss = self.update(num_update_per_opt)
        train_record.append(loss)
        cnt_opt += 1  # Counts number of optimization.
        if self.CONFIG.USE_WANDB:
          log_dict = {
              "loss/critic": loss[0],
              "loss/policy": loss[1],
              "loss/entropy": loss[2],
              "loss/alpha": loss[3],
              "cnt_safety_violation": cnt_safety_violation,
              "cnt_num_episode": cnt_num_episode,
              "alpha": self.policy.alpha,
              "gamma": self.policy.GAMMA,
          }
          wandb.log(log_dict, step=self.cnt_step, commit=False)

        # Checks after fixed number of gradient updates.
        if cnt_opt % check_opt_freq == 0 or first_update:
          print(f"Checks at sample step {self.cnt_step}:")
          cnt_opt = 0
          first_update = False

          save_dict = self.save(
              venv, force_save=False, reset_kwargs_list=reset_kwargs_list,
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
          elif rollout_end_criterion == "failure":
            safe_rate = save_dict['metric']
            save_dict.pop('metric')
            save_dict["safe_rate"] = safe_rate
            train_progress.append(np.array([safe_rate]))
            print('  => Safe rate: {:.2f}'.format(safe_rate))
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
                os.path.join(figure_folder, f"{self.cnt_step}.png")
            )

          # Resets anyway.
          obs_all = venv.reset()
          obs_dstb_prev_all = [None for _ in range(self.n_envs)]
          action_prev_all = [None for _ in range(self.n_envs)]
          r_prev_all = [None for _ in range(self.n_envs)]
          done_prev_all = [None for _ in range(self.n_envs)]
          info_prev_all = [None for _ in range(self.n_envs)]
          init_control = [None for _ in range(self.n_envs)]

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

  def save(
      self, venv: VecEnvBase, force_save: bool = False,
      reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None
  ) -> Dict:
    if force_save:
      info = {}
      metric = 0.
    else:
      num_eval_traj = self.CONFIG.NUM_EVAL_TRAJ
      eval_timeout = self.CONFIG.EVAL_TIMEOUT
      rollout_end_criterion = self.CONFIG.ROLLOUT_END_CRITERION
      adv_fn_list = [
          partial(get_adversary, dstb_policy=copy.deepcopy(self.policy.actor))
          for _ in range(self.n_envs)
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

    if self.policy.actor_type == 'max':
      # Maximizes cost -> minimizes safe/success rate.
      super().save(metric=1 - metric, force_save=force_save)
    else:
      super().save(metric=metric, force_save=force_save)
    return info
