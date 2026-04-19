# --------------------------------------------------------
# MjLab Walking Policy + ISAACS RCBF Safety Filter
#
# Uses the Robust Control Barrier Function (RCBF) filter
# instead of hard LRSF switching. The RCBF finds the
# closest action to the task action that still satisfies:
#   Q(x, u, pi_dstb(x,u)) >= kappa * V_hat(x)
#
# This provides smoother safety interventions compared to
# the binary LRSF filter.
#
# Usage:
#   python test_mjlab_rcbf_numpad.py \
#       --walk_ckpt ckpts/mjlab_walk/model_1000.pt \
#       --safety_config train_result/nature/go2_mujoco_isaacs_v22_long/config.yaml \
#       --kappa 0.995
# --------------------------------------------------------

import time
import argparse
import numpy as np
import threading

from rl_controller.mjlab_controller import (
    MjLabRLController, MUJOCO_ORDER, MJLAB_DEFAULT_DOF_POS
)
from wrapper import Wrapper
from safety_enforcer_mujoco import MujocoSafetyEnforcer, PYBULLET_ORDER

mj_order = MUJOCO_ORDER
pb_order = PYBULLET_ORDER


def transition(wrapper, cur, new, input_order=None):
    traj = np.linspace(np.array(cur), np.array(new), 200)
    cur_time = time.time()
    count = 0
    while count < 200:
        wrapper.update(traj[count], input_order=input_order)
        if time.time() - cur_time > 0.005:
            count += 1
            cur_time = time.time()


def main():
    parser = argparse.ArgumentParser(
        description="MjLab walk + ISAACS RCBF safety filter")
    parser.add_argument("--walk_ckpt", type=str,
                        default="ckpts/mjlab_walk/model_1000.pt")
    parser.add_argument("--safety_config", type=str,
                        default="train_result/nature/go2_mujoco_isaacs_v22_long/config.yaml")
    parser.add_argument("--safety_model_dir", type=str, default=None)
    parser.add_argument("--ctrl_step", type=int, default=None)
    parser.add_argument("--dstb_step", type=int, default=None)
    parser.add_argument("--kappa", type=float, default=0.995,
                        help="Barrier preservation factor (0.98-0.999)")
    parser.add_argument("--cbf_tol", type=float, default=1e-3,
                        help="RCBF convergence tolerance")
    parser.add_argument("--cbf_max_iters", type=int, default=20,
                        help="Max RCBF gradient iterations")
    parser.add_argument("--rcbf_lr", type=float, default=0.05,
                        help="RCBF gradient step size")
    parser.add_argument("--kp", type=str, default="80,80,180",
                        help="Per-joint-type kp: hip,thigh,calf (matched to walking policy training)")
    parser.add_argument("--kd", type=str, default="1,1,2",
                        help="Per-joint-type kd: hip,thigh,calf (matched to walking policy training)")
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()

    # ── Initialize ──
    controller = MjLabRLController(checkpoint_path=args.walk_ckpt)
    controller.test()

    safetyEnforcer = MujocoSafetyEnforcer(
        config_path=args.safety_config,
        model_dir=args.safety_model_dir,
        ctrl_step=args.ctrl_step,
        dstb_step=args.dstb_step,
        epsilon=0.0,  # not used for RCBF
        device='cpu'
    )

    wrapper = Wrapper()
    kp_per = [float(x) for x in args.kp.split(",")]
    kd_per = [float(x) for x in args.kd.split(",")]
    wrapper.kp = kp_per * 4
    wrapper.kd = kd_per * 4
    print(f"PD gains: kp={wrapper.kp[:3]}x4  kd={wrapper.kd[:3]}x4")

    stand_mj = list(MJLAB_DEFAULT_DOF_POS)
    sit = [-0.1, 1.5, -2.5, 0.1, 1.5, -2.5, -0.4, 1.5, -2.5, 0.4, 1.5, -2.5]
    stand_hw = wrapper.map(stand_mj, mj_order, wrapper.order)

    # Training applies RCBF output as increment to a persistent target with
    # 0.3 EMA smoothing; see go2_dynamics_mujoco.py:1379-1385 and
    # eval_safety_filter.py:1092-1098 (RCBF does NOT restore saved_targets,
    # so the increment accumulates on top of walking policy's target).
    ACTION_SMOOTHING = 0.3

    command = [0., 0., 0.]
    dt = args.dt

    key_commands = {
        '8': [0.5, 0.0, 0.0],
        '2': [-0.3, 0.0, 0.0],
        '4': [0.0, 0.3, 0.0],
        '6': [0.0, -0.3, 0.0],
        '7': [0.5, 0.2, -0.3],
        '9': [0.5, -0.2, 0.3],
        '1': [-0.3, 0.2, -0.3],
        '3': [-0.3, -0.2, 0.3],
        '5': [0.0, 0.0, 0.0],
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

    # ── Startup ──
    wrapper.update(sit)
    time.sleep(0.5)
    transition(wrapper, sit, stand_hw)
    time.sleep(0.5)

    controller.reset()
    safetyEnforcer.reset()

    input_thread = threading.Thread(target=listen_input, daemon=True)
    input_thread.start()

    print(f"\n=== MjLab Walk + RCBF Filter (kappa={args.kappa}) ===")
    print("Use numpad keys to control. Ctrl+C to stop.\n")

    next_control_time = time.time()
    try:
        last_action_hw = stand_hw
        while True:
            now = time.time()
            if now >= next_control_time:
                next_control_time += dt
                if now - next_control_time > dt:
                    next_control_time = now + dt
                # 1. Walking policy action (absolute target in MJ order)
                target_mj = controller.get_action(wrapper, command=command)
                walk_target_pb = np.array(wrapper.map(target_mj, mj_order, pb_order),
                                          dtype=np.float32)

                # 2. RCBF filter.
                # In training's increment mode the task action passed to the
                # filter is zero — the walking policy has already set the
                # absolute target, and RCBF's output is a SEPARATE increment
                # layered on top of that target. See eval_safety_filter:1030.
                walk_ctrl_pb = np.zeros(12, dtype=np.float32)
                filtered_pb, q_val, n_iters, alpha = safetyEnforcer.rcbf_projected_gradient(
                    wrapper, walk_ctrl_pb,
                    kappa=args.kappa,
                    tol=args.cbf_tol,
                    max_iters=args.cbf_max_iters,
                    lr=args.rcbf_lr
                )

                # Apply the filtered correction with 0.3 EMA smoothing on top
                # of the walking policy's target (matches integrate_forward
                # in training when RCBF is active).
                inc_pb = np.clip(np.asarray(filtered_pb), -0.5, 0.5)
                action_pb = walk_target_pb + ACTION_SMOOTHING * inc_pb
                # Hardware joint limits from go2_dynamics_mujoco.py:73-75
                action_pb[[0, 3, 6, 9]]  = np.clip(action_pb[[0, 3, 6, 9]],  -0.8, 0.8)
                action_pb[[1, 4, 7, 10]] = np.clip(action_pb[[1, 4, 7, 10]], -1.2, 1.0)
                action_pb[[2, 5, 8, 11]] = np.clip(action_pb[[2, 5, 8, 11]], -2.5, -0.85)
                action_mj = np.array(wrapper.map(action_pb, pb_order, mj_order))

                # 3. Clip joint limits
                action_mj[[0, 3, 6, 9]] = np.clip(action_mj[[0, 3, 6, 9]], -0.7, 0.7)
                action_mj[[1, 4, 7, 10]] = np.clip(action_mj[[1, 4, 7, 10]], -1.5, 1.5)
                action_mj[[2, 5, 8, 11]] = np.clip(action_mj[[2, 5, 8, 11]], -2.7, -0.85)

                # 4. Send to robot
                wrapper.update(action_mj, input_order=mj_order)
                last_action_hw = wrapper.map(action_mj, mj_order, wrapper.order)

                status = "RCBF " if alpha > 1e-4 else "WALK "
                v_str = f"V={safetyEnforcer.prev_v_hat:.3f}" if safetyEnforcer.prev_v_hat is not None else "V=?"
                print(f"\r  [{status}] {v_str}  Q={q_val:.3f}  a={alpha:.3f}  it={n_iters}", end="")

    except KeyboardInterrupt:
        print("\nShutting down...")
        transition(wrapper, last_action_hw, stand_hw)
        transition(wrapper, stand_hw, sit)
        print("Robot locked in SIT mode.")
        while True:
            wrapper.update(sit)


if __name__ == "__main__":
    main()
