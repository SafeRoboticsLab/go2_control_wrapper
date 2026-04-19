# --------------------------------------------------------
# Test MjLab Walking Policy on Go2 Hardware (Numpad Control)
#
# Deploys the MuJoCo-trained walking policy from unitree_rl_mjlab
# to the real Go2 robot with numpad velocity commands.
#
# Usage:
#   python test_mjlab_walk_numpad.py [--checkpoint PATH] [--kp KP] [--kd KD]
#
# Default checkpoint: ckpts/mjlab_walk/model_1000.pt
# --------------------------------------------------------

import time
import argparse
import numpy as np
import threading

from rl_controller.mjlab_controller import MjLabRLController, MUJOCO_ORDER
from wrapper import Wrapper

# ── Joint ordering ──
# MuJoCo order used by the policy: FL, FR, BL(=RL), BR(=RR)
# This matches the wrapper.map convention where BL=RL, BR=RR
sim_order = MUJOCO_ORDER  # ["FL", "FR", "BL", "BR"]


def transition(wrapper, cur, new, input_order=None):
    """Smooth transition between two joint configurations."""
    traj = np.linspace(np.array(cur), np.array(new), 200)
    cur_time = time.time()
    count = 0
    while count < 200:
        wrapper.update(traj[count], input_order=input_order)
        if time.time() - cur_time > 0.005:
            count += 1
            cur_time = time.time()


def main():
    parser = argparse.ArgumentParser(description="MjLab walk policy deployment")
    parser.add_argument("--checkpoint", type=str,
                        default="ckpts/mjlab_walk/model_1000.pt",
                        help="Path to MjLab walking policy checkpoint")
    parser.add_argument("--kp", type=str, default="100,100,200",
                        help="Per-joint-type kp: hip,thigh,calf (default: 100,100,200)")
    parser.add_argument("--kd", type=str, default="1,1,2",
                        help="Per-joint-type kd: hip,thigh,calf (default: 1,1,2)")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Control loop period in seconds (default: 0.02, i.e. 50Hz)")
    args = parser.parse_args()

    # ── Initialize controller and wrapper ──
    controller = MjLabRLController(checkpoint_path=args.checkpoint)
    controller.test()

    wrapper = Wrapper()
    kp_per_joint = [float(x) for x in args.kp.split(",")]  # [hip, thigh, calf]
    kd_per_joint = [float(x) for x in args.kd.split(",")]
    wrapper.kp = kp_per_joint * 4  # tiled for 4 legs
    wrapper.kd = kd_per_joint * 4

    # ── Preset poses (in hardware order: FR, FL, BR, BL) ──
    # MjLab default standing pose in MuJoCo order: FL, FR, BL, BR
    stand_mj = [-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8]
    sit = [-0.1, 1.5, -2.5, 0.1, 1.5, -2.5, -0.4, 1.5, -2.5, 0.4, 1.5, -2.5]

    # Convert stand to hardware order for transition
    stand_hw = wrapper.map(stand_mj, sim_order, wrapper.order)

    command = [0., 0., 0.]
    dt = args.dt

    # ── Numpad velocity commands ──
    key_commands = {
        '8': [0.5, 0.0, 0.0],    # Forward
        '2': [-0.3, 0.0, 0.0],   # Backward
        '4': [0.0, 0.3, 0.0],    # Left
        '6': [0.0, -0.3, 0.0],   # Right
        '7': [0.5, 0.2, -0.3],   # Forward + Left Turn
        '9': [0.5, -0.2, 0.3],   # Forward + Right Turn
        '1': [-0.3, 0.2, -0.3],  # Backward + Left Turn
        '3': [-0.3, -0.2, 0.3],  # Backward + Right Turn
        '5': [0.0, 0.0, 0.0],    # Stop
    }

    def listen_input():
        nonlocal command
        while True:
            user_input = input().strip()
            if user_input in key_commands:
                command = key_commands[user_input]
                print(f"  cmd: vx={command[0]:.1f} vy={command[1]:.1f} wz={command[2]:.1f}")
            elif user_input == 'q':
                break
            else:
                print("Use numpad keys (1-9) or 'q' to quit")

    # ── Startup: sit -> stand ──
    print("Starting up: sit -> stand")
    wrapper.update(sit)
    time.sleep(0.5)
    transition(wrapper, sit, stand_hw)
    time.sleep(0.5)

    # ── Reset controller and start ──
    controller.reset()
    decimation_time = time.time()

    # Start input listener
    input_thread = threading.Thread(target=listen_input, daemon=True)
    input_thread.start()

    print("\n=== MjLab Walking Policy Active ===")
    print("Use numpad keys to control:")
    print("  8=Forward  2=Backward  4=Left  6=Right")
    print("  7=FwdLeft  9=FwdRight  1=BwdLeft  3=BwdRight")
    print("  5=Stop")
    print("  Ctrl+C to stop\n")

    try:
        last_action_hw = stand_hw
        while True:
            if time.time() - decimation_time > dt:
                # Get walking policy action (returns MuJoCo order targets)
                target_mj = controller.get_action(wrapper, command=command)

                # Clip joint limits
                target_mj = np.array(target_mj)
                target_mj[[0, 3, 6, 9]] = np.clip(target_mj[[0, 3, 6, 9]], -0.7, 0.7)      # hip
                target_mj[[1, 4, 7, 10]] = np.clip(target_mj[[1, 4, 7, 10]], -1.5, 1.5)    # thigh
                target_mj[[2, 5, 8, 11]] = np.clip(target_mj[[2, 5, 8, 11]], -2.7, -0.85)  # calf

                # Send to robot (wrapper.update remaps from sim_order to hardware order)
                wrapper.update(target_mj, input_order=sim_order)
                last_action_hw = wrapper.map(target_mj, sim_order, wrapper.order)

                decimation_time = time.time()

    except KeyboardInterrupt:
        print("\nShutting down: stand -> sit")
        stand_hw_current = wrapper.map(last_action_hw, wrapper.order, wrapper.order)
        transition(wrapper, stand_hw_current, stand_hw)
        transition(wrapper, stand_hw, sit)
        print("Robot locked in SIT mode. Ctrl+C again to fully stop.")
        while True:
            wrapper.update(sit)


if __name__ == "__main__":
    main()
