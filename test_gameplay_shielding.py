import socket
import select
import struct
import time
import numpy as np
import math
import sys
import os
from inverse_kinematics.inverse_kinematics_controller import InverseKinematicsController
from rl_controller.rl_controller import Go2RLController
from wrapper import Wrapper
from safety_enforcer import SafetyEnforcer
import json
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

# controller = InverseKinematicsController(Xdist=0.387,
#                                          Ydist=0.284,
#                                          height=0.25,
#                                          coxa=0.03,
#                                          femur=0.2,
#                                          tibia=0.2,
#                                          L=2.0,
#                                          angle=0,
#                                          T=0.4,
#                                          dt=0.02)
controller = Go2RLController()
command = [0.4, 0.0, -0.15]

g_x = np.inf
l_x = np.inf
requested = False
L_horizon = 10
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
cur_time = time.time()
dt = 0.005

for i in range(100):
    joint_pos_sim = wrapper.map(wrapper.state[8:20], wrapper.order, sim_order)
    joint_vel_sim = wrapper.map(wrapper.state[20:32], wrapper.order, sim_order)
    state = wrapper.state[:8] + joint_pos_sim + joint_vel_sim + wrapper.state[32:] # this is now sim state
    sim_state = get_state(wrapper.state, command=[0, 0, 0])
    wrapper.update(controller.get_action(sim_state), input_order=sim_order) # mark that action is sim_order

try:
    while isConnected:
        if time.time() - cur_time > dt:
            #! WARNING: the joint pos and vel in wrapper.state is of real Go2, NOT sim
            joint_pos_sim = wrapper.map(wrapper.state[8:20], wrapper.order, sim_order)
            joint_vel_sim = wrapper.map(wrapper.state[20:32], wrapper.order, sim_order)
            state = wrapper.state[:8] + joint_pos_sim + joint_vel_sim + wrapper.state[32:] # this is now sim state
            
            if g_x < 0 or l_x < 0:
                safetyEnforcer.is_shielded = True
                # switch between fallback and target stable stance, depending on the current state
                stable_stance = np.array([0.5, 0.7, -1.5, 0.5, 0.7, -1.2, -0.5, 0.7, -1.5, -0.5, 0.7, -1.2])  # sim order
                margin = safetyEnforcer.target_margin(state)
                lx = min(margin.values())

                if lx > 0.01: # -0.1
                    action = stable_stance
                else:
                    # center_sampling
                    # action = safetyEnforcer.ctrl(np.array(state)) + np.array([0.2, 0.6, -1.5, 0.2, 0.6, -1.5, -0.2, 0.6, -1.5, -0.2, 0.6, -1.5])  # sim order

                    # increment
                    action = safetyEnforcer.ctrl(np.array(state)) + np.array(state[8:20])
            else:
                # IK controller
                # action = np.array(controller.get_action(joint_order=sim_order, offset=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

                # rl controller
                sim_state = get_state(wrapper.state, command=command)
                action = controller.get_action(sim_state)

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

                            # if lx > -0.1: # -0.1
                            if lx > 0.01:
                                action = stable_stance
                            else:
                                # center_sampling
                                # action = safetyEnforcer.ctrl(np.array(state)) + np.array([0.2, 0.6, -1.5, 0.2, 0.6, -1.5, -0.2, 0.6, -1.5, -0.2, 0.6, -1.5])  # sim order

                                # increment
                                action = safetyEnforcer.ctrl(np.array(state)) + np.array(state[8:20])
                        requested = False
                        step = 0
                step += 1
            cur_time = time.time()
        wrapper.update(action, input_order=sim_order)

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
