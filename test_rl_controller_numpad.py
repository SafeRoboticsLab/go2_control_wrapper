import time
import numpy as np
import torch
import threading
from scipy.spatial.transform import Rotation
from rl_controller.rl_controller import Go2RLController
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

stand = [0.1, 0.9, -1.9, -0.1, 0.9, -1.9, 0.1, 0.9, -1.9, -0.1, 0.9, -1.9]
sit = [-0.1, 1.5, -2.5, 0.1, 1.5, -2.5, -0.4, 1.5, -2.5, 0.4, 1.5, -2.5]

wrapper.update(sit)
transition(sit, stand)

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
sim_order = ["FL", "BL", "FR", "BR"]
command = [0., 0., 0.]

# dt = 0.005
dt = 0.001
decimation_time = time.time()

# Define movement commands
key_commands = {
    '8': [0.2, 0.0, 0.0],  # Forward
    '2': [-0.3, 0.0, 0.0],  # Backward
    '4': [0.0, 0.3, 0.0],  # Left
    '6': [0.0, -0.3, 0.0],  # Right
    '7': [0.3, 0.3, -0.2],  # Forward + Left Turn
    '9': [0.3, -0.3, 0.2],  # Forward + Right Turn
    '1': [-0.3, 0.3, -0.2],  # Backward + Left Turn
    '3': [-0.3, -0.3, 0.2],  # Backward + Right Turn
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
        if time.time() - decimation_time > dt * 4:
            state = get_state(wrapper.state, command=command)
            action = controller.get_action(state)
            action = wrapper.map(action, sim_order, wrapper.order)
            wrapper.update(action)
            decimation_time = time.time()
except KeyboardInterrupt:
    transition(action, stand)
    transition(stand, sit)
    print("Robot locked in SIT mode. Keyboard interrupt to stop.")
    while True:
        wrapper.update(sit)