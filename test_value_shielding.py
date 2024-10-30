import time
import numpy as np
import math
import sys
import os
from inverse_kinematics.inverse_kinematics_controller import InverseKinematicsController
from rl_controller.rl_controller import Go2RLController
from wrapper import Wrapper
from safety_enforcer import SafetyEnforcer
import torch
from scipy.spatial.transform import Rotation

def transition(cur, new):
  traj = np.linspace(np.array(cur), np.array(new), 200)
  cur_time = time.time()
  count = 0
  while count < 200:
    wrapper.update(traj[count])
    if time.time() - cur_time > 0.005:
      count += 1
      cur_time = time.time()

def get_state(state, command=[0, 0, 0]):
  """
  state = 0:3   robot_body_linear_vel
          3:5   roll, pitch
          5:8   robot_body_angular_vel
          8:20  joint_pos
          20:32 joint_vel
          32:36 foot contact

  Controller requirement
    (3) vx, vy, vz,
    (3) wx, wy, wz,
    (3) gx, gy, gz, # projected gravity
    (3) commands (lin_vel_x, lin_vel_y, ang_vel_yaw),
    (12) joint_pos offset, # this might be offset from a stance
    (12) joint_vel,
    (12) actions
  """
  quat = wrapper.msgs[0].imu_state.quaternion # w, x, y, z
  ang = tuple(quat[1:]) + tuple([quat[0]])
  rotmat = Rotation.from_quat(ang).as_matrix()
  projected_gravity = (np.linalg.inv(rotmat) @ np.array([0, 0, -1]).T)
  obs = (
    tuple(state[0:3]) + 
    tuple(state[5:8]) +
    tuple(projected_gravity) + 
    tuple(command) +
    tuple(wrapper.map(state[8:20], wrapper.order, sim_order)) +
    tuple(wrapper.map(state[20:32], wrapper.order, sim_order))
  )
  return torch.Tensor(obs)

# controller = InverseKinematicsController(
#   # Xdist=0.37, Ydist=0.29, height=0.30, coxa=0.02, femur=0.25, tibia=0.27, L=2.0, angle=0, T=0.4, dt=0.02
#   Xdist=0.39, Ydist=0.28, height=0.30, coxa=0.02, femur=0.21, tibia=0.25, L=2.0, angle=0, T=0.5, dt=0.02
# )

# controller = InverseKinematicsController(
#     Xdist=0.387, Ydist=0.284, height=0.25, coxa=0.03, femur=0.2, tibia=0.2, L=2.0, angle=0, T=0.25, dt=0.02
# )

controller = Go2RLController()
command = [0., 0., 0.0]

sim_order = ["FL", "BL", "FR", "BR"]
wrapper = Wrapper()
stable_stance_switch = True

# SMART - alt
# safetyEnforcer = SafetyEnforcer(epsilon=-1.35)
# SMART - tgda
# safetyEnforcer = SafetyEnforcer(epsilon=-0.9)

# safetyEnforcer = SafetyEnforcer(epsilon=-0.9)
# shield only
# safetyEnforcer = SafetyEnforcer(epsilon=np.inf)
# perf only
# safetyEnforcer = SafetyEnforcer(epsilon=-np.inf)

# NEW CORL DEMO MODELS
# safetyEnforcer = SafetyEnforcer(epsilon=np.inf) # shield only
safetyEnforcer = SafetyEnforcer(epsilon=-0.05) # value shielding
# safetyEnforcer = SafetyEnforcer(epsilon=-np.inf) # task policy only

stand = [0, 0.75, -1.8, 0, 0.75, -1.8, 0, 0.75, -1.8, 0, 0.75, -1.8] # following real order
sit = [-0.1, 1.5, -2.5, 0.1, 1.5, -2.5, -0.4, 1.5, -2.5, 0.4, 1.5, -2.5] # following real order
stable_stance = np.array([0.5, 0.7, -1.5, 0.5, 0.7, -1.2, -0.5, 0.7, -1.5, -0.5, 0.7, -1.2]) # sim order

# put the robot in sitting stance
wrapper.update(sit)

# stand up
transition(sit, stand)

# action will ALWAYS in sim_order
action = wrapper.map(stand, wrapper.order, sim_order)
# action = wrapper.map(stable_stance, wrapper.order, sim_order)

command_time = time.time()
loop_time = time.time()
dt = 0.005
mode = 0

try:
  while True:
    if time.time() - loop_time > dt:
      if time.time() - command_time > 3.0:
        if mode == 0:
          command = [0.4, 0.0, -0.15] # forward
        elif mode == 1:
          command = [-0.4, 0.0, -0.15] # backward
        elif mode == 2:
          command = [0.3, 0, -0.5] # rotate
        elif mode == 3:
          command = [0.0, 0.4, -0.15] # rotate
        elif mode == 4:
          command = [0.0, -0.4, -0.15] # rotate
        mode = (mode + 1)%5
        command_time = time.time()

      # transition from real state to sim state
      joint_pos_sim = wrapper.map(wrapper.state[8:20], wrapper.order, sim_order)
      joint_vel_sim = wrapper.map(wrapper.state[20:32], wrapper.order, sim_order)
      state = wrapper.state[:8] + joint_pos_sim + joint_vel_sim + wrapper.state[32:] # this is now sim state

      # inverse kinematics
      # ctrl = np.array(controller.get_action(joint_order=sim_order, offset=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) - joint_pos_sim
      
      # rl controller
      sim_state = get_state(wrapper.state, command=command)
      ctrl = controller.get_action(sim_state) - joint_pos_sim # reference pos
      
      action = np.clip(safetyEnforcer.get_action(np.array(state), ctrl), -np.ones(12)*0.5, np.ones(12)*0.5) # joint increment in sim
      # action = safetyEnforcer.get_action(np.array(state), ctrl) # joint increment in sim

      if safetyEnforcer.is_shielded:
        # center sampling
        # action = action + np.array([0.2, 0.6, -1.5, 0.2, 0.6, -1.5, -0.2, 0.6, -1.5, -0.2, 0.6, -1.5]) # sim order

        # increment
        action = action + joint_pos_sim

        # switch between fallback and target stable stance, depending on the current state
        if stable_stance_switch:
          # stable_stance = np.array([0.5, 0.7, -1.5, 0.5, 0.7, -1.5, -0.5, 0.7, -1.5, -0.5, 0.7, -1.5]) # sim order
          margin = safetyEnforcer.target_margin(state)
          lx = min(margin.values())

          # if lx > -0.1:
          if lx > 0.01:
              action = stable_stance
      else:
        action = ctrl + joint_pos_sim
      
      loop_time = time.time()
    
    wrapper.update(action, input_order=sim_order) # mark that action is sim_order
    # wrapper.update(stable_stance, input_order=sim_order)
    # print(safetyEnforcer.prev_q)

    if abs(wrapper.state[3]) > np.pi*0.4 or abs(wrapper.state[4]) > np.pi*0.4:
      break

except KeyboardInterrupt:
  transition(wrapper.map(action, sim_order, wrapper.order), stand)
  transition(stand, sit)

except Exception as e:
  print(e)
  transition(wrapper.map(action, sim_order, wrapper.order), stand)
  transition(stand, sit)

print("lock in SIT mode, keyboard interrupt to stop")
while True:
  wrapper.update(sit)
