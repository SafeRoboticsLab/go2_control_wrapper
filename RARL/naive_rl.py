# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu (kaichieh@princeton.edu)
#          Duy P. Nguyen (duyn@princeton.edu)
#          Allen Z. Ren (allen.ren@princeton.edu)

import copy
from typing import Optional, Callable, Dict, Union, List
import warnings
import os
import copy
import time
import numpy as np
import torch
import os
import time

import pybullet as p
import wandb
from gym import spaces

from .sac_mini import SAC_mini
from .base_training import BaseTraining
from simulators.base_single_env import BaseSingleEnv
from simulators.vec_env.vec_env import VecEnvBase
from simulators.agent import Agent
import math
import pandas as pd

class NaiveRL(BaseTraining):

  def __init__(
      self, CONFIG, CONFIG_UPDATE, CONFIG_ARCH, CONFIG_ENV, verbose=True
  ):
    super().__init__(CONFIG, CONFIG_ENV, CONFIG_UPDATE)

    print("= Constructing policy agent")
    self.policy = SAC_mini(CONFIG_UPDATE, CONFIG_ARCH)
    self.policy.build_network(verbose=verbose)

    w_action_space = np.array(CONFIG.WARMUP_ACTION_RANGE, dtype=np.float32)
    self.warmup_action_space = spaces.Box(
        low=w_action_space[:, 0], high=w_action_space[:, 1]
    )

    # alias
    self.module_all = [self.policy]

  @property
  def has_backup(self):
    return False

  def learn(
      self, env: BaseSingleEnv, current_step: Optional[int] = None,
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
    num_update_per_opt: int = self.CONFIG.UPDATE_PER_OPT
    check_opt_freq: int = self.CONFIG.CHECK_OPT_FREQ
    min_step_b4_opt: int = self.CONFIG.MIN_STEPS_B4_OPT
    out_folder: str = self.CONFIG.OUT_FOLDER
    rollout_end_criterion: str = self.CONFIG.ROLLOUT_END_CRITERION
    save_metric: str = self.CONFIG.SAVE_METRIC
    num_eval_traj: int = self.CONFIG.NUM_EVAL_TRAJ
    eval_timeout: int = self.CONFIG.EVAL_TIMEOUT

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

    while self.cnt_step <= max_steps:
      # Selects action.
      with torch.no_grad():
        if self.cnt_step < min_step_b4_opt:
          action_all = np.empty(shape=(self.n_envs, env.action_dim_ctrl))
          for i in range(self.n_envs):
            action_all[i, :] = self.warmup_action_space.sample()
          action_all = torch.FloatTensor(action_all).to(self.device)
        else:
          action_all, _ = self.policy.actor.sample(
              obs_all.float().to(self.device), append=None, latent=None
          )

      # Interacts with the env.
      if not single_env:
        obs_nxt_all, r_all, done_all, info_all = venv.step(
            action_all.cpu().numpy()
        )
      else:
        obs_nxt, r, done, info = venv.step(action_all[0].cpu().numpy())
        obs_nxt_all = torch.FloatTensor(np.stack([obs_nxt])).to(self.device)
        r_all = torch.FloatTensor(np.stack([r])).unsqueeze(dim=1).float()
        done_all = np.stack([done])
        info_all = [info]

      for env_idx, (done, info) in enumerate(zip(done_all, info_all)):
        # Store the transition in memory
        self.store_transition(
            obs_all[[env_idx]], action_all[[env_idx]], r_all[env_idx],
            obs_nxt_all[[env_idx]], done, info
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

          # Updates the agent in the environment with the newest policy.
          env.agent.policy.update_policy(self.policy.actor)
          
          if not single_env:
            agent_list = venv.get_attr(attr_name='agent')
          else:
            agent_list = [env.agent]
          
          for agent in agent_list:
            agent: Agent
            agent.policy.update_policy(self.policy.actor)
          reset_kwargs_list = []  # Same initial states.
          for _ in range(self.CONFIG.NUM_EVAL_TRAJ):
            env.reset()
            reset_kwargs_list.append({"state": np.copy(env.state)})
          save_dict = self.save(
              venv, force_save=False, reset_kwargs_list=reset_kwargs_list,
              action_kwargs_list=action_kwargs,
              rollout_step_callback=rollout_step_callback,
              rollout_episode_callback=rollout_episode_callback
          )

          if controller is not None:
            _, results, length = env.simulate_trajectories(
                num_trajectories=num_eval_traj, T_rollout=eval_timeout,
                end_criterion=env.end_criterion, reset_kwargs_list=reset_kwargs,
                action_kwargs_list=action_kwargs,
                rollout_step_callback=rollout_step_callback,
                rollout_episode_callback=rollout_episode_callback,
                controller=controller
            )

            perf_only_success_rate = np.sum(results == 1) / num_eval_traj          
            perf_only_safe_rate = np.sum(results != -1) / num_eval_traj
          
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
          
          if controller is not None:
            print('  - Perf only safe rate: {:.2f}'.format(perf_only_safe_rate))
            print('  - Perf only success rate: {:.2f}'.format(perf_only_success_rate))

          if self.CONFIG.USE_WANDB:
            wandb_data = {
              "safe_rate": safe_rate,
              "ep_length": np.mean(length),
            }
            
            if controller is not None:
              wandb_data["perf_safe_rate"] = perf_only_safe_rate

            if env.end_criterion == "reach-avoid":
              if controller is not None:
                wandb_data["perf_success_rate"] = perf_only_success_rate
              wandb_data["success_rate"] = success_rate
            
            wandb.log(wandb_data, step=self.cnt_step, commit=True)

          # if self.CONFIG.USE_WANDB:
          #   wandb.log(save_dict, step=self.cnt_step, commit=True)

          torch.save({
              'train_record': train_record,
              'train_progress': train_progress,
              'violation_record': violation_record,
          }, os.path.join(out_folder, 'train_details'))

          # Visualizes.
          if visualize_callback is not None:
            visualize_callback(
                env, self.policy.value,
                os.path.join(figure_folder, f"{self.cnt_step}.png")
            )

          # Resets anyway.
          if not single_env:
            obs_all = venv.reset()
          else:
            obs = venv.reset(cast_torch=True, **reset_kwargs)
            obs_all = torch.FloatTensor(np.stack([obs])).to(self.device)

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

  #! cleanup
  def evaluate(self, env):
    s = env.reset(cast_torch = True)
    # for i in range(1000):
    while True:
      # a, _ = self.actor.sample(torch.from_numpy(s).float().to(self.device))
      a = self.policy.actor(s.float().to(self.device))
      # a = torch.from_numpy(np.array([0] * 18)).float().to(self.device)

      # run for a_{t+1} and s_{t}
      critic_q = max(
          self.policy.critic(s.float().to(self.device), a.float().to(self.device))
      )

      # print(env.robot.get_observation()[2])

      # if larger than threshold, then not save nor live
      print("\r{:.3f}".format(float(critic_q.detach().numpy())), end="")

      # if critic_q > epsilon:
      #     print("\rNOT GOOD      ", end = "")
      # else:
      #     print("\rGOOD          ", end = "")

      a = a.detach().numpy()
      # print("State: {}, action: {}".format(s.detach().numpy()[:5], a))
      s_, r, done, info = env.step(a, cast_torch=True)
      s = s_
      # time.sleep(0.02)
      if done:
        if p.getKeyboardEvents().get(49):
          continue
        else:
          env.reset()

  def shielding_with_IK(self, env, controller, epsilon=0.25, override=True, **kwargs):
    rollout = False
    env_imaginary = None
    rollout_step = None
    if "rollout" in kwargs.keys():
      rollout = kwargs["rollout"]
    if "env_imaginary" in kwargs.keys():
      env_imaginary = kwargs["env_imaginary"]
    if "rollout_step" in kwargs.keys():
      rollout_step = kwargs["rollout_step"]

    verbose = False
    shield_count = 0
    max_q = -np.inf
    # state from SACRA
    state = env.reset(cast_torch = True, **kwargs)

    for i in range(3000):
      new_joint_pos = controller.get_action()
      action = torch.FloatTensor(new_joint_pos - np.array(env.agent.dyn.robot.get_joint_position()))

      if not rollout:
        # check if action and state are safe
        critic_q = max(
          self.policy.critic(
              state.float().to(self.device),
              action.float().to(self.device)
          )
        )
        max_q = max(critic_q.detach().numpy(), max_q)

        if critic_q > epsilon:
          # NOT GOOD, USE SHIELDING
          if override:
            shield_count+=1
            # action, _ = self.actor.sample(torch.from_numpy(state_sacra).float().to(self.device))
            action = self.policy.actor(state.float().to(self.device))
            action = action.detach().numpy()
            
            if env.agent.dyn.gui:
              p.addUserDebugText("SHIELDED", (0.0, 0.0, 0.5), (100, 0, 0), 1.5, 0.2, parentObjectUniqueId = env.agent.dyn.robot.id, physicsClientId = env.agent.dyn.client)
          
          if verbose:
            print(
                "\rStep: {}\tQ: {:.3f}\tSHIELDED!           ".format(str(i).zfill(3), float(critic_q.detach().numpy())
                ), end=""
            )
        else:
          # GOOD, CONTINUE WITH THE ACTION CHOICE FROM PERFORMANCE
          if override and env.agent.dyn.gui:
            p.addUserDebugText("GOOD", (0.0, 0.0, 0.5), (0, 100, 0), 1.5, 0.2, parentObjectUniqueId = env.agent.dyn.robot.id, physicsClientId = env.agent.dyn.client)
          if verbose:
            print(
                "\rStep: {}\tQ: {:.3f}\t                    ".format(str(i).zfill(3), float(critic_q.detach().numpy())
                ), end=""
            )
        state_, r, done, info = env.step(action, cast_torch=True)
        state = state_
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
        current_pos, current_ang = p.getBasePositionAndOrientation(env.agent.dyn.robot.id, physicsClientId = env.agent.dyn.client)
        rollout_kwargs = kwargs.copy()
        rollout_kwargs["initial_height"] = current_pos[2]
        rollout_kwargs["initial_rotation"] = current_ang
        rollout_state = env_imaginary.reset(cast_torch = True, ox = current_pos[0], oy = current_pos[1], **rollout_kwargs)
        
        for joint in range(p.getNumJoints(env.agent.dyn.robot.id, physicsClientId = env.agent.dyn.client)):
            current_joint_position, current_joint_velocity, _, _ = p.getJointState(env.agent.dyn.robot.id, joint, physicsClientId = env.agent.dyn.client)
            p.resetJointState(env_imaginary.agent.dyn.robot.id, joint, current_joint_position, targetVelocity = current_joint_velocity, physicsClientId = env_imaginary.agent.dyn.client)
          
        current_linear_velocity, current_angular_velocity = p.getBaseVelocity(env.agent.dyn.robot.id, physicsClientId = env.agent.dyn.client)
        p.resetBaseVelocity(env_imaginary.agent.dyn.robot.id, linearVelocity = current_linear_velocity, angularVelocity = current_angular_velocity, physicsClientId = env_imaginary.agent.dyn.client)

        # apply the performance action
        rollout_action = action.detach().numpy()
        rollout_state_, r, done, info = env_imaginary.step(rollout_action, cast_torch=True)
        rollout_state = rollout_state_

        rollout_failed = True
        for j in range(rollout_step):
          rollout_action = self.policy.actor(rollout_state.float().to(self.device))
          rollout_action = rollout_action.detach().numpy()
          
          rollout_state_, r, done, info = env_imaginary.step(rollout_action, cast_torch=True)
          rollout_state = rollout_state_
        
          if info["g_x"] > 0:
            rollout_failed = True
            break
          elif info["g_x"] <= 0 and info["l_x"] <= 0:
            rollout_failed = False
            break

        if rollout_failed:
          if override:
            shield_count+=1
            action = self.policy.actor(state.float().to(self.device))
            
            if env.agent.dyn.gui:
              p.addUserDebugText("SHIELDED", (0.0, 0.0, 0.5), (100, 0, 0), 1.5, 0.2, parentObjectUniqueId = env.agent.dyn.robot.id, physicsClientId = env.agent.dyn.client)
          
          if verbose:
            print(
                "\rStep: {}\tQ: rollout\tSHIELDED!           ".format(str(i).zfill(3)
                ), end=""
            )
        else:
          if override and env.agent.dyn.gui:
            p.addUserDebugText("GOOD", (0.0, 0.0, 0.5), (0, 100, 0), 1.5, 0.2, parentObjectUniqueId = env.agent.dyn.robot.id, physicsClientId = env.agent.dyn.client)
          
          if verbose:
            print(
                "\rStep: {}\tQ: rollout\t                    ".format(str(i).zfill(3)
                ), end=""
            )

        action = action.detach().numpy()
        state_, r, done, info = env.step(action, cast_torch=True)
        state = state_

      # Check if roll pitch are not too high
      error = False
      euler = p.getEulerFromQuaternion(env.agent.dyn.robot.linc_get_pos()[1])
      if ((abs(euler[1]) >= math.pi / 2) or (abs(euler[0]) >= math.pi / 2)):
        error = True

      # if done or sum(env.robot.linc_get_ground_contacts()) == 0:
      # if done or error:
      if error:  # only care if the robot flips, do not care about the safety margin of safety policy
        return 0, np.linalg.norm(state[:2]), shield_count, max_q, i
    return 1, np.linalg.norm(state[:2]), shield_count, max_q, i
  
  def check_shielding(self, env, controller, epsilon=0.25):
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

    for instance in ["shield", "no_shield"]:
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

    if "iteration" in kwargs.keys():
      iteration = kwargs["iteration"]
    if "video" in kwargs.keys():
      video = kwargs["video"]
    if "rollout_step" in kwargs.keys():
      rollout_step = kwargs["rollout_step"]
    if batch_conditions in kwargs.keys():
      batch_conditions = kwargs["batch_conditions"]

    result = {}
    distance = {}
    shield_count = {}
    max_q = {}
    steps = {}

    if batch_conditions is not None:
      assert type(batch_conditions) is pd.core.frame.DataFrame, "Error: batch condition must be DataFrame type"
      if iteration is not None:
         initial_conditions = batch_conditions.iloc[iteration]
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
          rollout=True, env_imaginary=env_imaginary, rollout_step=rollout_step, video_output_file=video_output_file)
      elif instance == "shield":
        result[instance], distance[instance], shield_count[instance], max_q[instance], steps[instance] = self.shielding_with_IK(env, new_controller, epsilon=epsilon, override=True, 
          terrain_data=terrain_data, initial_height=initial_height, initial_rotation=initial_rotation, initial_joint_value=initial_joint_value, video_output_file=video_output_file)
      elif instance == "no_shield":
        result[instance], distance[instance], shield_count[instance], max_q[instance], steps[instance] = self.shielding_with_IK(env, new_controller, epsilon=epsilon, override=False, 
          terrain_data=terrain_data, initial_height=initial_height, initial_rotation=initial_rotation, initial_joint_value=initial_joint_value, video_output_file=video_output_file)
    return result, distance, shield_count, max_q, steps

  # region: utils
  def restore(self, step, logs_path, **kwargs):
    # super().restore(step, logs_path, "policy")
    super().restore(step, logs_path, **kwargs)
  
  # def restore(self, step, logs_path):
  #   super().restore(step, logs_path, "performance")

  # endregion
  def update(self, num_update_per_opt: int) -> np.ndarray:
    """Updates neural networks.

    Args:
        num_update_per_opt_list (np.ndarray): the number of NN updates per
            optimization.

    Returns:
        np.ndarray: losses.
    """
    loss = np.zeros(4)
    total_updates = 0
    for timer in range(num_update_per_opt):
      sample = True
      cnt = 0
      while sample:
        batch_pre = self.sample_batch()
        sample = np.all(np.asarray(batch_pre.done))
        cnt += 1
        if cnt >= 10:
          break
      if sample:
        warnings.warn("Cannot get a valid batch!!", UserWarning)
        continue
      else:
        total_updates += 1

      batch = self.unpack_batch(batch_pre)
      # loss_q, loss_pi, loss_entropy, loss_alpha
      loss_tp = self.policy.update(batch, timer)
      for i, l in enumerate(loss_tp):
        loss[i] += l
    loss[0] /= total_updates
    loss[1:] /= (total_updates / self.policy.update_period)
    return loss

  def restore(self, step: int, model_folder: str):
    super().restore(step, model_folder)

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
      _, results, length = venv.simulate_trajectories(
          num_trajectories=num_eval_traj, T_rollout=eval_timeout,
          end_criterion=rollout_end_criterion,
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
