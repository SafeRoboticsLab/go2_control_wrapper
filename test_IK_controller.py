import time
import numpy as np
import math
import sys
import os
from inverse_kinematics.inverse_kinematics_controller import InverseKinematicsController
from wrapper import Wrapper

def transition(cur, new):
  traj = np.linspace(np.array(cur), np.array(new), 200)
  cur_time = time.time()
  count = 0
  while count < 200:
    wrapper.update(traj[count])
    if time.time() - cur_time > 0.005:
      count += 1
      cur_time = time.time()

# high speed, no blocking
forward_controller = InverseKinematicsController(
  Xdist=0.37, Ydist=0.29, height=0.30, coxa=0.02, femur=0.23, tibia=0.25, L=3.0, angle=0, T=0.6, dt=0.005
)
backward_controller = InverseKinematicsController(
  Xdist=0.37, Ydist=0.29, height=0.25, coxa=0.02, femur=0.23, tibia=0.25, L=1.5, T=0.8, dt=0.005
)

# forward_controller = InverseKinematicsController(
#   Xdist=0.37, Ydist=0.29, height=0.30, coxa=0.02, femur=0.23, tibia=0.25, L=3.0, angle=0, T=1.0, dt=0.01
# )
# backward_controller = InverseKinematicsController(
#   Xdist=0.37, Ydist=0.29, height=0.25, coxa=0.02, femur=0.23, tibia=0.25, L=1.5, T=1.0, dt=0.01
# )

wrapper = Wrapper()

stand = [0, 0.75, -1.8, 0, 0.75, -1.8, 0, 0.75, -1.8, 0, 0.75, -1.8]
sit = [-0.1, 1.5, -2.5, 0.1, 1.5, -2.5, -0.4, 1.5, -2.5, 0.4, 1.5, -2.5]

# put the robot in sitting stance
wrapper.update(sit)

# stand up
transition(sit, stand)
action = stand

cur_time = time.time()
start_time = time.time()
mode = 0

"""
mode:
0: wait
1: forward
2: wait
3: backward
"""

duration = [2.0, 5.0, 2.0, 5.0]

try:
  while True:
    wrapper.update(action)
    if time.time() - cur_time > 0.005:
      cur_time = time.time()
      if mode == 0:
        nxt_action = forward_controller.get_action(joint_order=wrapper.order, offset=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        transition(action, nxt_action)
        action = nxt_action[:]
      elif mode == 1:
        action = forward_controller.get_action(joint_order=wrapper.order, offset=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      elif mode == 2:
        nxt_action = backward_controller.get_action(joint_order=wrapper.order, offset=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        transition(action, nxt_action)
        action = nxt_action[:]
      elif mode == 3:
        action = backward_controller.get_action(joint_order=wrapper.order, offset=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      
    if time.time() - start_time > duration[mode]:
      # mode = (mode + 1)%4
      mode = min(1, mode+1)
      start_time = time.time()

except KeyboardInterrupt:
  transition(action, stand)
  transition(stand, sit)

except Exception as e:
  print(e)
  transition(action, stand)
  transition(stand, sit)
