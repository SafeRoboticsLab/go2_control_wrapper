# --------------------------------------------------------
# Analyze a walk log CSV produced by test_mjlab_walk_numpad.py --log
#
# Usage:
#   python analyze_walk_log.py logs/walk_YYYYMMDD_HHMMSS_kpXX_kdY.csv
#
# Prints summary statistics and highlights any obvious sim2real mismatches
# (huge raw actions, saturation, unexpected projected-gravity magnitudes,
# joint-velocity noise floor, etc).
# --------------------------------------------------------

import sys
import numpy as np
import pandas as pd


def main():
    if len(sys.argv) != 2:
        print("usage: python analyze_walk_log.py <log.csv>")
        sys.exit(1)
    path = sys.argv[1]
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} steps from {path}")
    print(f"Duration: {df['time'].iloc[-1]:.2f}s")

    # Mask of "should be standing still" steps: cmd magnitude < 0.1
    cmd_norm = np.linalg.norm(df[["cmd_vx", "cmd_vy", "cmd_wz"]].values, axis=1)
    stand_mask = cmd_norm < 0.1
    walk_mask = ~stand_mask
    print(f"  standstill steps: {stand_mask.sum()}  walking steps: {walk_mask.sum()}")

    # ── Projected gravity sanity ──
    pg = df[["proj_g_x", "proj_g_y", "proj_g_z"]].values
    pg_norm = np.linalg.norm(pg, axis=1)
    pg_xy = np.linalg.norm(pg[:, :2], axis=1)
    print("\n[projected gravity]  expect |g|≈1, |g_xy|≈0 when level")
    print(f"  |g|    mean={pg_norm.mean():.3f}  std={pg_norm.std():.3f}  "
          f"min={pg_norm.min():.3f}  max={pg_norm.max():.3f}")
    print(f"  |g_xy| mean={pg_xy.mean():.3f}  max={pg_xy.max():.3f}")
    print(f"  g_z    mean={pg[:, 2].mean():.3f}  (should be near -1)")

    # ── Angular velocity ──
    av = df[["ang_vel_x", "ang_vel_y", "ang_vel_z"]].values
    av_max = np.max(np.abs(av), axis=1)
    print("\n[base angular velocity, body frame]")
    print(f"  |w|_max  mean={av_max.mean():.3f}  p95={np.percentile(av_max, 95):.3f}  "
          f"peak={av_max.max():.3f}")

    # ── Joint velocities ──
    jvel_cols = [f"jvel_mj_{i}" for i in range(12)]
    jvel = df[jvel_cols].values
    jvel_abs_mean = np.mean(np.abs(jvel))
    jvel_p99 = np.percentile(np.abs(jvel), 99)
    print("\n[joint velocity, MJ order]")
    print(f"  mean|q̇|={jvel_abs_mean:.3f}  p99={jvel_p99:.3f}  peak={np.max(np.abs(jvel)):.3f}")
    # Per-joint standstill noise floor (what the encoder differentiator produces
    # when the robot is commanded to stand still)
    if stand_mask.sum() > 0:
        jv_stand = jvel[stand_mask]
        per_joint_std = jv_stand.std(axis=0)
        print(f"  standstill per-joint std (rad/s):")
        names = [f"{leg}_{kind}" for leg in ["FL", "FR", "BL", "BR"] for kind in ["hip", "thi", "cal"]]
        for n, s in zip(names, per_joint_std):
            print(f"    {n:8s}  {s:.3f}")

    # ── Raw action stats ──
    raw_cols = [f"raw_act_{i}" for i in range(12)]
    raw = df[raw_cols].values
    raw_max_per_step = np.max(np.abs(raw), axis=1)
    clipped = np.sum(np.abs(raw) > 1.0, axis=1)
    print("\n[policy raw action, pre-clip — expect small at standstill]")
    print(f"  |a|_max  mean={raw_max_per_step.mean():.3f}  p95={np.percentile(raw_max_per_step, 95):.3f}  "
          f"peak={raw_max_per_step.max():.3f}")
    print(f"  frac-steps with ≥1 joint saturated (|a|>1): "
          f"{(clipped > 0).mean() * 100:.1f}%")
    if stand_mask.sum() > 0:
        raw_stand = np.max(np.abs(raw[stand_mask]), axis=1)
        print(f"  standstill |a|_max  mean={raw_stand.mean():.3f}  "
              f"p95={np.percentile(raw_stand, 95):.3f}  peak={raw_stand.max():.3f}")
        print(f"  → if standstill |a| is >0.2 consistently, the policy thinks the "
              f"robot is not at rest — likely an obs mismatch.")

        # Per-joint standstill raw action — sign + magnitude reveal obs bugs.
        # All hips pegged one sign → hip-sign flipped; one leg asymmetric → ordering bug.
        raw_s = raw[stand_mask]
        names = [f"{leg}_{kind}" for leg in ["FL", "FR", "BL", "BR"]
                 for kind in ["hip", "thi", "cal"]]
        print("  per-joint standstill raw action (MJ order):")
        print(f"    {'joint':8s}  {'mean':>7s}  {'std':>6s}  {'sat_pos%':>8s}  {'sat_neg%':>8s}")
        for i, n in enumerate(names):
            col = raw_s[:, i]
            sat_pos = (col > 1.0).mean() * 100
            sat_neg = (col < -1.0).mean() * 100
            print(f"    {n:8s}  {col.mean():+7.3f}  {col.std():6.3f}  "
                  f"{sat_pos:8.1f}  {sat_neg:8.1f}")

    # ── Target vs actual joint position tracking error ──
    tgt_cols = [f"target_mj_{i}" for i in range(12)]
    jp_cols = [f"jpos_mj_{i}" for i in range(12)]
    tgt = df[tgt_cols].values
    jp = df[jp_cols].values
    track_err = tgt - jp
    track_err_abs = np.abs(track_err)
    print("\n[target - actual joint pos (MJ order)]")
    print(f"  mean|err|={track_err_abs.mean():.3f}  p95={np.percentile(track_err_abs, 95):.3f}  "
          f"peak={track_err_abs.max():.3f}")
    # Per-joint-type tracking error
    hip_err = track_err_abs[:, [0, 3, 6, 9]].mean()
    thi_err = track_err_abs[:, [1, 4, 7, 10]].mean()
    cal_err = track_err_abs[:, [2, 5, 8, 11]].mean()
    print(f"  per-joint-type mean |err|:  hip={hip_err:.3f}  thigh={thi_err:.3f}  calf={cal_err:.3f}")
    print(f"  → large steady-state err → kp too low OR motors fighting policy output")

    # ── joint_pos_rel (obs[11:23]): joints relative to MjLab default ──
    jpr_cols = [f"jpos_rel_{i}" for i in range(12)]
    jpr = df[jpr_cols].values
    print("\n[joint pos relative to MjLab default (obs[11:23])]")
    print(f"  mean|q-q_def|={np.mean(np.abs(jpr)):.3f}  peak={np.max(np.abs(jpr)):.3f}")

    # ── Time between control steps ──
    dt_actual = np.diff(df["time"].values)
    print("\n[control loop timing]")
    print(f"  dt: mean={dt_actual.mean()*1000:.1f}ms  p99={np.percentile(dt_actual, 99)*1000:.1f}ms  "
          f"peak={dt_actual.max()*1000:.1f}ms")

    # ── Quick diagnosis ──
    print("\n=== Heuristic diagnosis ===")
    issues = []
    if pg_norm.mean() < 0.9 or pg_norm.mean() > 1.1:
        issues.append(f"projected gravity magnitude off ({pg_norm.mean():.3f}) — obs/quaternion bug")
    if pg[:, 2].mean() > -0.8:
        issues.append(f"proj_g_z = {pg[:, 2].mean():.3f} (should be ≈ -1) — gravity sign flipped?")
    if stand_mask.sum() > 0:
        raw_stand = np.max(np.abs(raw[stand_mask]), axis=1)
        if raw_stand.mean() > 0.3:
            issues.append(f"standstill raw action too large (mean |a|_max={raw_stand.mean():.2f}) — obs mismatch")
    if (clipped > 0).mean() > 0.1:
        issues.append(f"policy saturates >10% of steps — obs likely very wrong or unstable fall")
    if track_err_abs.mean() > 0.1:
        issues.append(f"large tracking error ({track_err_abs.mean():.2f} rad mean) — kp too low")
    if av_max.mean() > 2.0:
        issues.append(f"high base ang_vel — robot shaking")

    if not issues:
        print("  No obvious red flags.")
    else:
        for i in issues:
            print(f"  - {i}")


if __name__ == "__main__":
    main()
