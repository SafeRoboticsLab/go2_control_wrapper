import time
import numpy as np
import math
import sys
import os
from rl_controller.rl_controller import Go2RLController
from wrapper import Wrapper
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

wrapper = Wrapper()

stand = [0, 0.75, -1.8, 0, 0.75, -1.8, 0, 0.75, -1.8, 0, 0.75, -1.8]
sit = [-0.1, 1.5, -2.5, 0.1, 1.5, -2.5, -0.4, 1.5, -2.5, 0.4, 1.5, -2.5]

# put the robot in sitting stance
wrapper.update(sit)

# stand up
transition(sit, stand)
action = stand

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

controller = Go2RLController()
sim_order = ["FL", "BL", "FR", "BR"]
command = [0., 0., 0.0]

command_time = time.time()

try:
  while True:
    if time.time() - command_time > 2.0:
      # command[0] = min(command[0] + 0.01, 0.5)
      # command = [0.5, 0., 0.0]
      command = [0.5, 0, 1.0]
      # command_time = time.time()
    elif time.time() - command_time > 4.0:
      # command[0] = max(command[0] - 0.01, 0.0)
      command = [0, 0, 0]

    state = get_state(wrapper.state, command=command)
    action = controller.get_action(state) # reference pos
    # constraint joint incremental
    # action = np.clip(np.array(wrapper.map(action, sim_order, wrapper.order)) - np.array(wrapper.state[8:20]), -np.ones(12) * 0.5, np.ones(12) * 0.5) + np.array(wrapper.state[8:20])
    action = wrapper.map(action, sim_order, wrapper.order)
    wrapper.update(action)

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

