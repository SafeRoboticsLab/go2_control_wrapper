import time
import numpy as np
import math
import sys
import os
from inverse_kinematics.inverse_kinematics_controller import InverseKinematicsController
from wrapper import Wrapper
from safety_enforcer import SafetyEnforcer

def transition(cur, new):
  traj = np.linspace(np.array(cur), np.array(new), 200)
  cur_time = time.time()
  count = 0
  while count < 200:
    wrapper.update(traj[count])
    if time.time() - cur_time > 0.005:
      count += 1
      cur_time = time.time()

controller = InverseKinematicsController(
  Xdist=0.37, Ydist=0.29, height=0.30, coxa=0.02, femur=0.23, tibia=0.25, L=2.0, angle=0, T=0.6, dt=0.02
)

sim_order = ["FL", "BL", "FR", "BR"]
wrapper = Wrapper()
# safetyEnforcer = SafetyEnforcer(epsilon=-0.9)
# shield only
# safetyEnforcer = SafetyEnforcer(epsilon=np.inf)
# perf only
safetyEnforcer = SafetyEnforcer(epsilon=-np.inf)


stand = [0, 0.75, -1.8, 0, 0.75, -1.8, 0, 0.75, -1.8, 0, 0.75, -1.8] # following real order
sit = [-0.1, 1.5, -2.5, 0.1, 1.5, -2.5, -0.4, 1.5, -2.5, 0.4, 1.5, -2.5] # following real order

# put the robot in sitting stance
wrapper.update(sit)

# stand up
transition(sit, stand)

# action will ALWAYS in sim_order
action = wrapper.map(stand, wrapper.order, sim_order)

try:
  while True:
    ctrl = np.array(controller.get_action(joint_order=sim_order, offset=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) - np.array(wrapper.map(wrapper.state[8:20], wrapper.order, sim_order))
    action = safetyEnforcer.get_action(np.array(wrapper.state), ctrl) # joint increment
    if safetyEnforcer.is_shielded:
      # center sampling
      action = action + np.array([0.2, 0.6, -1.5, 0.2, 0.6, -1.5, -0.2, 0.6, -1.5, -0.2, 0.6, -1.5]) # sim order

      # switch between fallback and target stable stance, depending on the current state
      stable_stance = np.array([0.5, 0.7, -1.5, 0.5, 0.7, -1.5, -0.5, 0.7, -1.5, -0.5, 0.7, -1.5]) # sim order
      margin = safetyEnforcer.target_margin(wrapper.state)
      lx = min(margin.values())
      joint_pos = wrapper.state[8:20]

      if lx > -0.1:
          action = stable_stance
    else:
      action = ctrl + np.array(wrapper.map(wrapper.state[8:20], wrapper.order, sim_order))
    # action = ctrl + np.array(wrapper.map(wrapper.state[8:20], wrapper.order, sim_order))
    wrapper.update(action, input_order=sim_order) # mark that action is sim_order

    # print(safetyEnforcer.prev_q)

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
