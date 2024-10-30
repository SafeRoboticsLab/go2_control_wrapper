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


wrapper = Wrapper()

stand = [0, 0.75, -1.8, 0, 0.75, -1.8, 0, 0.75, -1.8, 0, 0.75, -1.8]
sit = [-0.1, 1.5, -2.5, 0.1, 1.5, -2.5, -0.4, 1.5, -2.5, 0.4, 1.5, -2.5]

stable_stance = [
    0.5, 0.7, -1.5, 0.5, 0.7, -1.2, -0.5, 0.7, -1.5, -0.5, 0.7, -1.2
]

# stable_stance = [
#     0.2, 0.7, -1.5, 0.5, 0.5, -1.5, -0.2, 0.7, -1.5, -0.5, 0.5, -1.5
# ]

# put the robot in sitting stance
wrapper.update(sit)

# stand up
transition(sit, stand)
action = stable_stance

sim_order = ["FL", "BL", "FR", "BR"]

try:
    while True:
        wrapper.update(stable_stance, input_order=sim_order)
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
