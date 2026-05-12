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
import csv
import datetime
import os
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
    parser.add_argument("--kp", type=str, default="80,80,180",
                        help="Per-joint-type kp: hip,thigh,calf (80-pct of training sim stiffness; "
                             "empirically best for this robot)")
    parser.add_argument("--kd", type=str, default="1,1,2",
                        help="Per-joint-type kd: hip,thigh,calf (matches training sim damping)")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Control loop period in seconds (default: 0.02, i.e. 50Hz)")
    parser.add_argument("--log", type=str, default=None,
                        help="Optional CSV log path. If 'auto', uses logs/walk_<timestamp>.csv")
    parser.add_argument("--print_every", type=int, default=25,
                        help="Print diagnostic summary every N control steps (25 @ 50Hz = 0.5s)")
    parser.add_argument("--hold_cmd_zero_steps", type=int, default=0,
                        help="Force cmd=[0,0,0] for the first N control steps to capture standstill behavior")
    args = parser.parse_args()

    # ── Set up CSV log ──
    log_file = None
    log_writer = None
    if args.log:
        log_path = args.log
        if log_path == "auto":
            os.makedirs("logs", exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            kp_tag = args.kp.replace(",", "-")
            kd_tag = args.kd.replace(",", "-")
            log_path = f"logs/walk_{ts}_kp{kp_tag}_kd{kd_tag}.csv"
        log_file = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        header = ["step", "time"]
        header += ["cmd_vx", "cmd_vy", "cmd_wz"]
        header += ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
        header += ["proj_g_x", "proj_g_y", "proj_g_z"]
        header += ["phase_s", "phase_c"]
        header += [f"jpos_rel_{i}" for i in range(12)]  # MJ order
        header += [f"jvel_mj_{i}" for i in range(12)]
        header += [f"last_act_{i}" for i in range(12)]
        header += [f"raw_act_{i}" for i in range(12)]    # policy output, pre-clip
        header += [f"target_mj_{i}" for i in range(12)]  # after clip + default offset
        header += [f"jpos_mj_{i}" for i in range(12)]    # actual joint pos (MJ order)
        header += ["roll", "pitch"]
        log_writer.writerow(header)
        print(f"Logging to {log_path}")

    # ── Initialize controller and wrapper ──
    controller = MjLabRLController(checkpoint_path=args.checkpoint)
    controller.test()

    wrapper = Wrapper()
    kp_per = [float(x) for x in args.kp.split(",")]
    kd_per = [float(x) for x in args.kd.split(",")]
    wrapper.kp = kp_per * 4  # [hip,thigh,calf] × 4 legs
    wrapper.kd = kd_per * 4
    print(f"PD gains: kp={wrapper.kp[:3]}×4  kd={wrapper.kd[:3]}×4")

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
        '8': [1.0, 0.0, 0.0],    # Forward
        '2': [-1.0, 0.0, 0.0],   # Backward
        '4': [0.0, 0.5, 0.0],    # Left
        '6': [0.0, -0.5, 0.0],   # Right
        '7': [1.0, 0.4, -0.5],   # Forward + Left Turn
        '9': [1.0, -0.4, 0.5],   # Forward + Right Turn
        '1': [-1.0, 0.4, -0.5],  # Backward + Left Turn
        '3': [-1.0, -0.4, 0.5],  # Backward + Right Turn
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

    # Start input listener
    input_thread = threading.Thread(target=listen_input, daemon=True)
    input_thread.start()

    print("\n=== MjLab Walking Policy Active ===")
    print("Use numpad keys to control:")
    print("  8=Forward  2=Backward  4=Left  6=Right")
    print("  7=FwdLeft  9=FwdRight  1=BwdLeft  3=BwdRight")
    print("  5=Stop")
    print("  Ctrl+C to stop\n")

    step = 0
    loop_start_time = time.time()
    next_control_time = time.time()
    try:
        last_action_hw = stand_hw
        while True:
            now = time.time()
            if now >= next_control_time:
                # Schedule next step BEFORE work so cadence is locked to wall clock.
                next_control_time += dt
                # If we fell behind by more than one period, resync (avoid runaway catch-up).
                if now - next_control_time > dt:
                    next_control_time = now + dt
                # Optionally force cmd=0 at start for diagnostic capture
                active_cmd = (
                    [0.0, 0.0, 0.0] if step < args.hold_cmd_zero_steps else command
                )

                # Get walking policy action (returns MuJoCo order targets)
                target_mj_raw = controller.get_action(wrapper, command=active_cmd)

                # Clip joint limits
                target_mj = np.array(target_mj_raw).copy()
                target_mj[[0, 3, 6, 9]] = np.clip(target_mj[[0, 3, 6, 9]], -0.7, 0.7)      # hip
                target_mj[[1, 4, 7, 10]] = np.clip(target_mj[[1, 4, 7, 10]], -1.5, 1.5)    # thigh
                target_mj[[2, 5, 8, 11]] = np.clip(target_mj[[2, 5, 8, 11]], -2.7, -0.85)  # calf

                # Send to robot (wrapper.update remaps from sim_order to hardware order)
                wrapper.update(target_mj, input_order=sim_order)
                last_action_hw = wrapper.map(target_mj, sim_order, wrapper.order)

                # ── Diagnostics ──
                obs = controller._last_obs          # 47D
                raw_act = controller._last_raw_action  # 12D pre-clip
                jpos_hw = wrapper.state[8:20]
                jpos_mj = np.array(wrapper.map(jpos_hw, wrapper.order, sim_order))
                roll, pitch = wrapper.state[3], wrapper.state[4]

                if log_writer is not None:
                    row = [step, time.time() - loop_start_time]
                    row += list(active_cmd)
                    row += list(obs[0:3])     # ang_vel
                    row += list(obs[3:6])     # proj_g
                    row += list(obs[9:11])    # phase
                    row += list(obs[11:23])   # jpos_rel
                    row += list(obs[23:35])   # jvel
                    row += list(obs[35:47])   # last_action (pre-step)
                    row += list(raw_act)      # raw_action (this step)
                    row += list(target_mj)    # clipped target in MJ order
                    row += list(jpos_mj)      # actual jpos (MJ order)
                    row += [roll, pitch]
                    log_writer.writerow(row)

                if step % args.print_every == 0:
                    raw_max = np.max(np.abs(raw_act))
                    raw_clip_count = int(np.sum(np.abs(raw_act) > 1.0))
                    jvel_max = np.max(np.abs(obs[23:35]))
                    ang_vel_max = np.max(np.abs(obs[0:3]))
                    proj_g_xy = np.linalg.norm(obs[3:5])  # 0 when level
                    print(
                        f"[{step:5d}] cmd=({active_cmd[0]:+.2f},{active_cmd[1]:+.2f},{active_cmd[2]:+.2f}) "
                        f"|raw_act|_max={raw_max:.3f} clipped={raw_clip_count}/12 "
                        f"|ang_vel|_max={ang_vel_max:.2f} |jvel|_max={jvel_max:.2f} "
                        f"|proj_g_xy|={proj_g_xy:.3f} roll={roll:+.2f} pitch={pitch:+.2f}"
                    )

                step += 1

    except KeyboardInterrupt:
        print("\nShutting down: stand -> sit")
        if log_file is not None:
            log_file.flush()
            log_file.close()
            print(f"Log closed.")
        stand_hw_current = wrapper.map(last_action_hw, wrapper.order, wrapper.order)
        transition(wrapper, stand_hw_current, stand_hw)
        transition(wrapper, stand_hw, sit)
        print("Robot locked in SIT mode. Ctrl+C again to fully stop.")
        while True:
            wrapper.update(sit)


if __name__ == "__main__":
    main()
