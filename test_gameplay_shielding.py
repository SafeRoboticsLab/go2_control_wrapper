import socket
import select
import struct
import time
import numpy as np
import math
import sys
import os
from inverse_kinematics.inverse_kinematics_controller import InverseKinematicsController
from wrapper import Wrapper
from safety_enforcer import SafetyEnforcer
import json


def transition(cur, new):
    traj = np.linspace(np.array(cur), np.array(new), 200)
    cur_time = time.time()
    count = 0
    while count < 200:
        wrapper.update(traj[count])
        if time.time() - cur_time > 0.005:
            count += 1
            cur_time = time.time()


controller = InverseKinematicsController(Xdist=0.387,
                                         Ydist=0.284,
                                         height=0.25,
                                         coxa=0.03,
                                         femur=0.2,
                                         tibia=0.2,
                                         L=2.0,
                                         angle=0,
                                         T=0.4,
                                         dt=0.02)

g_x = np.inf
l_x = np.inf
requested = False
L_horizon = 1
step = 0

sim_order = ["FL", "BL", "FR", "BR"]
wrapper = Wrapper()
safetyEnforcer = SafetyEnforcer(epsilon=0.0)

stand = [0, 0.75, -1.8, 0, 0.75, -1.8, 0, 0.75, -1.8, 0, 0.75, -1.8]  # following real order
sit = [-0.1, 1.5, -2.5, 0.1, 1.5, -2.5, -0.4, 1.5, -2.5, 0.4, 1.5, -2.5]  # following real order

# action will ALWAYS in sim_order
action = wrapper.map(stand, wrapper.order, sim_order)

# put the robot in sitting stance
wrapper.update(sit)

# stand up
transition(sit, stand)

# SETUP COMMAND RECEIVING SERVER
HOST = "192.168.0.248"
PORT = 65432

isConnected = False

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    isConnected = True
except OSError:
    try:
        s.sendall(b'Test')
        isConnected = True
    except OSError:
        print("Something is wrong")
        isConnected = False
    pass

s.setblocking(0)

try:
    while isConnected:
        #! WARNING: the joint pos and vel in wrapper.state is of real Go2, NOT sim
        state = wrapper.state[:8] + wrapper.map(wrapper.state[8:20], wrapper.order, sim_order) + wrapper.map(wrapper.state[20:32], wrapper.order, sim_order) + wrapper.state[32:]
        
        if g_x < 0 or l_x < 0:
            safetyEnforcer.is_shielded = True
            # switch between fallback and target stable stance, depending on the current state
            stable_stance = np.array([0.5, 0.7, -1.5, 0.5, 0.7, -1.2, -0.5, 0.7, -1.5, -0.5, 0.7, -1.2])  # sim order
            margin = safetyEnforcer.target_margin(state)
            lx = min(margin.values())

            if lx > 0.0: # -0.1
                action = stable_stance
            else:
                # center_sampling
                # action = safetyEnforcer.ctrl(np.array(state)) + np.array([0.2, 0.6, -1.5, 0.2, 0.6, -1.5, -0.2, 0.6, -1.5, -0.2, 0.6, -1.5])  # sim order

                # increment
                action = safetyEnforcer.ctrl(np.array(state)) + np.array(state[8:20])
        else:
            action = np.array(controller.get_action(joint_order=sim_order, offset=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

        if not requested:
            ctrl = action - np.array(state[8:20])
            data = [*state, *ctrl]
            s.sendall(struct.pack("!48f", *data))
            requested = True
        else:
            if step > L_horizon:
                if select.select([s], [], [], 0.01)[0]:
                    data = s.recv(1024)
                    data = json.loads(data.decode("utf-8"))
                    g_x = data["g_x"]
                    l_x = data["l_x"]
                    if g_x < 0 or l_x < 0:
                        stable_stance = np.array([0.5, 0.7, -1.5, 0.5, 0.7, -1.2, -0.5, 0.7, -1.5, -0.5, 0.7, -1.2])  # sim order
                        margin = safetyEnforcer.target_margin(state)
                        lx = min(margin.values())

                        if lx > 0: # -0.1
                            action = stable_stance
                        else:
                            # center_sampling
                            # action = safetyEnforcer.ctrl(np.array(state)) + np.array([0.2, 0.6, -1.5, 0.2, 0.6, -1.5, -0.2, 0.6, -1.5, -0.2, 0.6, -1.5])  # sim order

                            # increment
                            action = safetyEnforcer.ctrl(np.array(state)) + np.array(state[8:20])
                    requested = False
                    step = 0
            step += 1
        wrapper.update(action, input_order=sim_order)
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
