import time
import numpy as np
import torch
import threading
from scipy.spatial.transform import Rotation
from inverse_kinematics.inverse_kinematics_controller import InverseKinematicsController
from rl_controller.rl_controller import Go2RLController
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

def get_state(state, command=[0, 0, 0]):
    quat = wrapper.msgs[0].imu_state.quaternion  # w, x, y, z
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
command = [0., 0., 0.]

sim_order = ["FL", "BL", "FR", "BR"]
wrapper = Wrapper()
stable_stance_switch = True
# safetyEnforcer = SafetyEnforcer(epsilon=np.inf) # shield only
safetyEnforcer = SafetyEnforcer(epsilon=-0.045) # value shielding

# safetyEnforcer = SafetyEnforcer(epsilon=-0.05) # value shielding

# safetyEnforcer = SafetyEnforcer(epsilon=-np.inf) # task policy only

stand = [0.1, 0.9, -1.9, -0.1, 0.9, -1.9, 0.1, 0.9, -1.9, -0.1, 0.9, -1.9]
sit = [-0.1, 1.5, -2.5, 0.1, 1.5, -2.5, -0.4, 1.5, -2.5, 0.4, 1.5, -2.5]
stable_stance = np.array([0.5, 0.7, -1.5, 0.5, 0.7, -1.2, -0.5, 0.7, -1.5, -0.5, 0.7, -1.2]) # sim order

wrapper.update(sit)
transition(sit, stand)

dt = 0.005
decimation_time = time.time()

# Define movement commands
key_commands = {
    '8': [0.2, 0.0, 0.0],  # Forward
    '2': [-0.3, 0.0, 0.0],  # Backward
    '4': [0.0, 0.3, 0.0],  # Left
    '6': [0.0, -0.3, 0.0],  # Right
    '7': [0.3, 0.2, -0.2],  # Forward + Left Turn
    '9': [0.3, -0.2, 0.2],  # Forward + Right Turn
    '1': [-0.3, 0.2, -0.2],  # Backward + Left Turn
    '3': [-0.3, -0.2, 0.2],  # Backward + Right Turn
    '5': [0.0, 0.0, 0.0],  # Stop
}

# Function to handle user input via SSH
def listen_input():
    global command
    while True:
        user_input = input().strip()
        if user_input in key_commands:
            command = key_commands[user_input]
        else:
            print("Invalid input. Use numpad-like keys.")

# Start input listener in a separate thread
input_thread = threading.Thread(target=listen_input, daemon=True)
input_thread.start()

try:
    while True:
        if time.time() - decimation_time > dt:
            # Transition from real state to sim state
            joint_pos_sim = wrapper.map(wrapper.state[8:20], wrapper.order, sim_order)
            joint_vel_sim = wrapper.map(wrapper.state[20:32], wrapper.order, sim_order)
            state = wrapper.state[:8] + joint_pos_sim + joint_vel_sim + wrapper.state[32:]  # Sim state
            
            sim_state = get_state(wrapper.state, command=command)
            ctrl = controller.get_action(sim_state) - joint_pos_sim  # Reference pos
            action = np.clip(safetyEnforcer.get_action(np.array(state), ctrl), -np.ones(12)*0.5, np.ones(12)*0.5)
            
            if safetyEnforcer.is_shielded:
                action = action + joint_pos_sim
                
                # switch between fallback and target stable stance, depending on the current state
                if stable_stance_switch:
                  margin = safetyEnforcer.target_margin(state)
                  lx = min(margin.values())

                #   if lx > -0.1:
                  if lx > -0.05:
                      action = stable_stance
            else:
                action = ctrl + joint_pos_sim
            
            action = np.array(action)
            action[[0, 3, 6, 9]] = np.clip(action[[0, 3, 6, 9]], -0.7, 0.7)
            action[[1, 4, 7, 10]] = np.clip(action[[1, 4, 7, 10]], -1.5, 1.5)
            action[[2, 5, 8, 11]] = np.clip(action[[2, 5, 8, 11]], -2.7, -0.85)
            
            wrapper.update(action, input_order=sim_order)
            decimation_time = time.time()

except KeyboardInterrupt:
    transition(wrapper.map(action, sim_order, wrapper.order), stand)
    transition(stand, sit)
    print("Robot locked in SIT mode. Keyboard interrupt to stop.")
    while True:
        wrapper.update(sit)
