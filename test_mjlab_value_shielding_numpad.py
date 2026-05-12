# --------------------------------------------------------
# MjLab Walking Policy + MuJoCo ISAACS Value Shielding (LRSF)
#
# Combines the MuJoCo-trained walking policy with the
# MuJoCo-trained ISAACS safety policy for value-based
# safety filtering on the real Go2 robot.
#
# This is the MuJoCo equivalent of test_value_shielding_numpad.py
# which uses the old Isaac Gym walking policy + PyBullet safety policy.
#
# Usage:
#   python test_mjlab_value_shielding_numpad.py \
#       --walk_ckpt ckpts/mjlab_walk/model_1000.pt \
#       --epsilon -0.05
#
# Safety-only diagnostic (no walking policy; safety ctrl commands every step):
#   python test_mjlab_value_shielding_numpad.py --safety_only
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

# Joint orders
mj_order = MUJOCO_ORDER   # ["FL", "FR", "BL", "BR"] - walking policy
pb_order = PYBULLET_ORDER  # ["FL", "BL", "FR", "BR"] - safety policy


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
    parser = argparse.ArgumentParser(
        description="MjLab walk + ISAACS MuJoCo value shielding")
    parser.add_argument("--walk_ckpt", type=str,
                        default="ckpts/mjlab_walk/model_1000.pt",
                        help="Path to MjLab walking policy checkpoint")
    parser.add_argument("--safety_config", type=str,
                        default="train_result/nature/go2_mujoco_isaacs_v22_long/config.yaml",
                        help="Path to ISAACS config YAML (canonical copy lives with the weights)")
    parser.add_argument("--safety_model_dir", type=str, default=None,
                        help="Path to ISAACS model dir (auto from config if omitted)")
    parser.add_argument("--ctrl_step", type=int, default=None,
                        help="ISAACS ctrl/critic checkpoint step")
    parser.add_argument("--dstb_step", type=int, default=None,
                        help="ISAACS dstb checkpoint step")
    parser.add_argument("--epsilon", type=float, default=-0.05,
                        help="LRSF threshold (V < epsilon triggers shielding)")
    parser.add_argument("--kp", type=str, default="80,80,180",
                        help="Per-joint-type kp: hip,thigh,calf (matched to walking policy training)")
    parser.add_argument("--kd", type=str, default="1,1,2",
                        help="Per-joint-type kd: hip,thigh,calf (matched to walking policy training)")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Control loop period (sec)")
    parser.add_argument("--no_stable_stance", action="store_true",
                        help="Disable the stable-stance fallback when lx > lx_enter_stance. "
                             "With this flag, the learned safety controller commands the robot "
                             "on every shielded step — useful for evaluating the safety policy "
                             "itself, not the hand-designed fallback pose.")
    parser.add_argument("--stance_blend", type=float, default=0.02,
                        help="Per-step EMA rate for annealing safety_target toward stable_stance. "
                             "0.02 at 50Hz reaches ~95%% of target in ~3s. "
                             "Larger = faster but more jerky. Set 1.0 to replicate the old "
                             "instantaneous swap (may flip the robot).")
    parser.add_argument("--lx_enter_stance", type=float, default=-0.02,
                        help="Enter stable-stance mode when lx > this. Tighter (closer to 0) "
                             "means robot must be more upright before fallback engages. "
                             "Stance mode is latching: once entered, we stay in it until "
                             "shielding itself disengages (walking resumes).")
    parser.add_argument("--safety_only", action="store_true",
                        help="Skip walking policy entirely; safety ctrl commands every step. "
                             "Use for first-time validation of the safety policy on hardware.")
    args = parser.parse_args()

    # ── Initialize walking policy (skipped in safety-only mode) ──
    controller = None
    if not args.safety_only:
        controller = MjLabRLController(checkpoint_path=args.walk_ckpt)
        controller.test()

    # ── Initialize safety enforcer ──
    # In safety-only mode, force V_hat < epsilon always by setting epsilon huge.
    effective_epsilon = float("inf") if args.safety_only else args.epsilon
    safetyEnforcer = MujocoSafetyEnforcer(
        config_path=args.safety_config,
        model_dir=args.safety_model_dir,
        ctrl_step=args.ctrl_step,
        dstb_step=args.dstb_step,
        epsilon=effective_epsilon,
        device='cpu'
    )

    # ── Initialize wrapper ──
    wrapper = Wrapper()
    kp_per = [float(x) for x in args.kp.split(",")]
    kd_per = [float(x) for x in args.kd.split(",")]
    wrapper.kp = kp_per * 4
    wrapper.kd = kd_per * 4
    print(f"PD gains: kp={wrapper.kp[:3]}x4  kd={wrapper.kd[:3]}x4")

    # ── Preset poses ──
    # Stand in MuJoCo order
    stand_mj = list(MJLAB_DEFAULT_DOF_POS)
    sit = [-0.1, 1.5, -2.5, 0.1, 1.5, -2.5, -0.4, 1.5, -2.5, 0.4, 1.5, -2.5]
    # Stable stance for safety fallback (PyBullet order: FL, BL, FR, BR)
    stable_stance_pb = np.array([
        0.5, 0.7, -1.5,   # FL
        0.5, 0.7, -1.2,   # BL (RL)
        -0.5, 0.7, -1.5,  # FR
        -0.5, 0.7, -1.2   # BR (RR)
    ])

    stand_hw = wrapper.map(stand_mj, mj_order, wrapper.order)

    # Training uses a persistent target in MuJoCo order with EMA-smoothed
    # incremental updates (see go2_dynamics_mujoco.integrate_forward:1385
    # and eval_safety_filter line 1099-1102). We mirror that exactly in PB
    # order here. The home PB-order stance matches the training DEFAULT_STANCE.
    # ACTION_SMOOTHING = 0.3  # from config agent.action_smoothing
    ACTION_SMOOTHING = 0.5
    # MJLAB_DEFAULT_DOF_POS is MJ order [FL,FR,BL,BR]; PB order is [FL,BL,FR,BR]
    home_pb = np.array(wrapper.map(list(MJLAB_DEFAULT_DOF_POS), mj_order, pb_order),
                       dtype=np.float32)
    # Persistent target that the safety policy's increments accumulate into.
    # Re-initialized at the start of every shielded episode (see reset below).
    safety_target_pb = home_pb.copy()
    # Hysteretic stable-stance mode flag: once we enter stance mode we stay in
    # it until lx drops below lx_exit_stance, preventing flicker near the
    # boundary and the associated repeated target jumps.
    in_stance_mode = False

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

    # ── Startup ──
    print("Starting up: sit -> stand")
    wrapper.update(sit)
    time.sleep(0.5)
    transition(wrapper, sit, stand_hw)
    time.sleep(0.5)

    if controller is not None:
        controller.reset()
    safetyEnforcer.reset()

    input_thread = threading.Thread(target=listen_input, daemon=True)
    input_thread.start()

    if args.safety_only:
        print(f"\n=== SAFETY-ONLY (no walking policy; safety ctrl every step) ===")
    else:
        print(f"\n=== MjLab Walk + ISAACS Value Shielding (epsilon={args.epsilon}) ===")
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
                # ── 1. Get walking policy action (skipped in safety-only mode) ──
                joint_pos_hw = wrapper.state[8:20]
                joint_pos_pb = np.array(wrapper.map(joint_pos_hw, wrapper.order, pb_order))
                if args.safety_only:
                    # Placeholder incremental action; ignored because eps=inf forces shielding.
                    target_mj = None
                    walk_ctrl_pb = np.zeros(12, dtype=np.float32)
                else:
                    target_mj = controller.get_action(wrapper, command=command)
                    # Walk action in MuJoCo order -> PB order, then incremental
                    target_pb = np.array(wrapper.map(target_mj, mj_order, pb_order))
                    walk_ctrl_pb = np.clip(target_pb - joint_pos_pb, -0.5, 0.5)

                # ── 2. Safety filter (LRSF) ──
                filtered_action = safetyEnforcer.get_action(wrapper, walk_ctrl_pb)

                if safetyEnforcer.is_shielded:
                    # Training convention: safety action is an INCREMENT added
                    # to a persistent target (not to measured joint_pos), with
                    # 0.3 EMA smoothing. See eval_safety_filter.py:1099-1102
                    # and go2_dynamics_mujoco.py:1379-1385.
                    inc_pb = np.clip(np.asarray(filtered_action), -0.5, 0.5)
                    safety_target_pb = safety_target_pb + ACTION_SMOOTHING * inc_pb
                    # Hardware joint limits used in training (go2_dynamics_mujoco.py:73-75)
                    safety_target_pb[[0, 3, 6, 9]]  = np.clip(safety_target_pb[[0, 3, 6, 9]],  -0.8, 0.8)    # abduction
                    safety_target_pb[[1, 4, 7, 10]] = np.clip(safety_target_pb[[1, 4, 7, 10]], -1.2, 1.0)    # hip
                    safety_target_pb[[2, 5, 8, 11]] = np.clip(safety_target_pb[[2, 5, 8, 11]], -2.5, -0.85)  # knee
                    action_pb = safety_target_pb

                    # Optional hand-designed fallback when the robot is
                    # upright enough (lx > lx_enter_stance). Stance mode is
                    # latching: once entered we stay until shielding itself
                    # disengages (walk branch below clears in_stance_mode).
                    # Anneal toward stable_stance_pb with an EMA blend so the
                    # hand-off doesn't spike PD torques.
                    if not args.no_stable_stance:
                        if not in_stance_mode:
                            margin = safetyEnforcer.target_margin(wrapper)
                            lx = min(margin.values())
                            if lx > args.lx_enter_stance:
                                in_stance_mode = True
                        if in_stance_mode:
                            beta = args.stance_blend
                            safety_target_pb = (
                                (1.0 - beta) * safety_target_pb + beta * stable_stance_pb
                            )
                            action_pb = safety_target_pb
                            status = "STANCE"
                        else:
                            status = "SHIELD"
                    else:
                        status = "SHIELD"

                    action_mj = np.array(wrapper.map(action_pb, pb_order, mj_order))
                else:
                    # Walking policy drives the robot; keep safety_target_pb
                    # in sync with the walking target so that when safety
                    # later engages, its first increment references the
                    # current intended pose (mirrors eval script's saved_targets).
                    safety_target_pb = np.array(wrapper.map(target_mj, mj_order, pb_order),
                                                dtype=np.float32)
                    # Also clear stance mode so the next shielding episode
                    # re-evaluates lx fresh instead of inheriting stale state.
                    in_stance_mode = False
                    action_mj = target_mj
                    status = "WALK  "

                # ── 3. Clip joint limits ──
                action_mj = np.array(action_mj)
                action_mj[[0, 3, 6, 9]] = np.clip(action_mj[[0, 3, 6, 9]], -0.7, 0.7)
                action_mj[[1, 4, 7, 10]] = np.clip(action_mj[[1, 4, 7, 10]], -1.5, 1.5)
                action_mj[[2, 5, 8, 11]] = np.clip(action_mj[[2, 5, 8, 11]], -2.7, -0.85)

                # ── 4. Send to robot ──
                wrapper.update(action_mj, input_order=mj_order)
                last_action_hw = wrapper.map(action_mj, mj_order, wrapper.order)

                # Print status
                q_str = f"V={safetyEnforcer.prev_q:.3f}" if safetyEnforcer.prev_q is not None else "V=?"
                print(f"\r  [{status}] {q_str}  eps={args.epsilon}", end="")

    except KeyboardInterrupt:
        print("\nShutting down: stand -> sit")
        transition(wrapper, last_action_hw, stand_hw)
        transition(wrapper, stand_hw, sit)
        print("Robot locked in SIT mode.")
        while True:
            wrapper.update(sit)


if __name__ == "__main__":
    main()
