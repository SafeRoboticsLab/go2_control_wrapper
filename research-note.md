# Research Notes: MjLab Walking Policy + ISAACS MuJoCo Safety Deployment

## Overview

This document covers deploying the MuJoCo-trained walking policy (from `unitree_rl_mjlab`) and the MuJoCo-trained ISAACS safety policy (from `safe_adaptation_dev`) to the real Go2 robot.

---

## 1. Joint Ordering Conventions

There are **four** joint ordering conventions used across the codebase:

| Convention | Order | Usage |
|---|---|---|
| **Hardware** (wrapper.order) | FR, FL, BR, BL | Physical Go2 motor output |
| **MuJoCo** (mjlab) | FL, FR, RL(=BL), RR(=BR) | Walking policy training |
| **PyBullet** (ISAACS) | FL, RL(=BL), FR, RR(=BR) | Safety policy training |
| **Isaac Gym** (old controller) | FL, FR, BL, BR | Old walking policy (same as MuJoCo) |

**Note:** In the wrapper's naming, BL = Back Left = Rear Left (RL), BR = Back Right = Rear Right (RR).

### Permutation: PyBullet <-> MuJoCo
```
_PB2MJ = _MJ2PB = [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]  (self-inverse)
```

### Permutation: Hardware -> MuJoCo
```
FR(hw[0]) -> FR(mj[1])
FL(hw[1]) -> FL(mj[0])
BR(hw[2]) -> RR(mj[3])
BL(hw[3]) -> RL(mj[2])
```

---

## 2. Walking Policy (MjLab)

**Source:** `unitree_rl_mjlab`, checkpoint `ckpts/mjlab_walk/model_1000.pt`

### Observation Space (47D, MuJoCo joint order)

| Index | Dimension | Content |
|---|---|---|
| 0:3 | 3 | Base angular velocity (body frame, gyroscope) |
| 3:6 | 3 | Projected gravity (body frame, from quaternion) |
| 6:9 | 3 | Command velocity [vx, vy, wz] |
| 9:11 | 2 | Gait phase [sin(2π·t/0.6), cos(2π·t/0.6)] |
| 11:23 | 12 | Joint positions relative to default |
| 23:35 | 12 | Joint velocities |
| 35:47 | 12 | Last action (network output, clipped [-1,1]) |

**Key:** No linear velocity in actor obs (critic-only). No height scan on real robot.

### Network Architecture
```
EmpiricalNormalization(47)  # running mean/std from training
Linear(47, 512) -> ELU
Linear(512, 256) -> ELU
Linear(256, 128) -> ELU
Linear(128, 12)  -> action mean
```

### Action Space (12D)
- Network output ∈ [-1, 1] (clipped)
- Target position = default_pos + action × 0.25
- Default positions (MuJoCo order):
  ```
  FL: [-0.1, 0.9, -1.8]  FR: [0.1, 0.9, -1.8]
  RL: [-0.1, 0.9, -1.8]  RR: [0.1, 0.9, -1.8]
  ```

### Gait Phase
- Period: 0.6 seconds
- Zeroed when command norm < 0.1 (standing still)
- Phase = (time % period) / period

### Training PD Gains (MuJoCo sim)
- Hip: kp=100, kd=1
- Thigh: kp=100, kd=1
- Calf: kp=200, kd=2

### Hardware PD Gains (empirically tuned for THIS robot + MjLab policy)
- **Per-joint-type, matched to training sim:** `kp=[80,80,180]*4`, `kd=[1,1,2]*4`.
- Tuned on 2026-04-20 by sweeping from uniform `50/3` upward while
  watching the standstill policy-saturation rate and walking behavior:
  - `50/3` (uniform): calves sag 0.4 rad at startup → `joint_pos_rel`
    outside training distribution → 100% action saturation → robot collapses.
  - `50,50,100 / 3,3,5`: 98% saturation, shakes, doesn't walk.
  - `60,60,150 / 3,3,6`: 49% saturation, walks slowly.
  - `100,100,200 / 1,1,2` (exact training sim): walks well but mild shake.
  - `80,80,180 / 1,1,2`: clean walk, no shake — **current default**.
- Training sim stiffness (inside MuJoCo actuator model) is `kp=100,100,200
  / kd=1,1,2`. Although that's a sim-side parameter, matching it on the
  real motor keeps the closed-loop dynamics close enough that the policy's
  observations stay in-distribution.
- `unitree_rl_mjlab/deploy.yaml` ships `kp=20,20,40 / kd=1,1,2`, which is
  far too soft for this particular Go2 unit.

---

## 3. Old Safety Policy (PyBullet-based, currently deployed)

**Source:** `train_result/test_go2/go2_corldemo_tgda_richURDF_1/`
**Config:** Same directory, `config.yaml`

### Observation (36D, PyBullet joint order)
| Index | Dim | Content |
|---|---|---|
| 0:3 | 3 | Body linear velocity |
| 3:5 | 2 | Roll, pitch |
| 5:8 | 3 | Body angular velocity |
| 8:20 | 12 | Joint positions (PyBullet order) |
| 20:32 | 12 | Joint velocities |
| 32:36 | 4 | Foot contact flags |

### Networks
- Ctrl: [256, 256, 256] Sin, obs=36, action=12
- Dstb: [256, 256, 256] Sin, obs=36, action=6
- Critic: [128, 128, 128] Sin, obs=36, action=18
- **Dstb does NOT see ctrl action**

### Model Loading
Uses `RARL/sac_adv.py` → `restore_refactor()` with paths:
- `ctrl/ctrl-STEP.pth`
- `dstb/dstb-STEP.pth`
- `central/central-STEP.pth`

---

## 4. New Safety Policy (MuJoCo ISAACS)

**Source:** `safe_adaptation_dev`, config bundled at `train_result/nature/go2_mujoco_isaacs_v22_long/config.yaml`
**Train output:** `train_result/nature/go2_mujoco_isaacs_v22_long/` — step 50,600,000 is the deployed checkpoint.

### Key Differences from Old Policy

| Aspect | Old (PyBullet) | New (MuJoCo) |
|---|---|---|
| Ctrl obs dim | 36 | **48** (36 + 12 prev_action) |
| Dstb obs dim | 36 | **60** (48 + 12 ctrl_action) |
| Dstb sees ctrl | No | **Yes** |
| Ctrl network | [256,256,256] | **[512,512,512]** |
| Dstb network | [256,256,256] | [256,256,256] |
| Critic network | [128,128,128] | **[256,256,256]** |
| Activation | Sin | Sin |
| Gamma | 0.9 (fixed) | 0.95 → 0.999 (scheduled) |
| Target margin | roll, pitch only | corner_height, elbow, body_ang, vel, foot_width |

### Observation (48D = 36 + 12, PyBullet joint order)
Same as old 36D, plus:
| 36:48 | 12 | Previous ctrl action |

### Dstb Observation (60D = 48 + 12)
| 0:48 | 48 | Same as ctrl obs |
| 48:60 | 12 | Current ctrl action |

### Networks
```
Ctrl:   [48] -> 512 -> 512 -> 512 -> 12  (Sin, tanh output, scaled [-0.5, 0.5])
Dstb:   [60] -> 256 -> 256 -> 256 -> 6   (Sin, tanh output, scaled per-dim)
Critic: [48+18=66] -> 256 -> 256 -> 256 -> 1 (Sin, twin Q)
```

### Model Files (expected)
- `model/ctrl/ctrl-STEP.pth`
- `model/dstb/dstb-STEP.pth`
- `model/central/central-STEP.pth`

---

## 5. Safety Filter Types

### LRSF (Least-Restrictive Safety Filter)
```
V̂(x) = Q(x, π_ctrl(x), π_dstb(x, π_ctrl(x)))
if V̂(x) < ε:
    u = π_ctrl(x)     # full safety fallback
else:
    u = u_task          # walking policy
```
- Binary switch, conservative
- ε = -0.05 typical

### RCBF (Robust Control Barrier Function)
```
Constraint: Q(x, u, π_dstb(x,u)) >= κ · V̂(x)
Solve: u* = argmin ||u_task - u||²  s.t. constraint
```
- Projected gradient ascent in 12D action space
- Smooth blend (α ∈ [0,1])
- κ = 0.995 typical for dt=0.02
- More computationally expensive per step

---

## 6. Deployment Files Created

| File | Purpose |
|---|---|
| `rl_controller/mjlab_controller.py` | MjLab walking policy loader + inference |
| `ISAACS_mujoco/neural_network.py` | MLP, Sin activation (ported from safe_adaptation_dev) |
| `ISAACS_mujoco/model.py` | GaussianPolicy, TwinnedQNetwork (ported) |
| `safety_enforcer_mujoco.py` | New MuJoCo ISAACS safety enforcer |
| `test_mjlab_walk_numpad.py` | Walk-only deployment test |
| `test_mjlab_value_shielding_numpad.py` | LRSF value shielding demo |
| `test_mjlab_rcbf_numpad.py` | RCBF safety filter demo |
| `ckpts/mjlab_walk/model_1000.pt` | Walking policy checkpoint (copied) |
| `train_result/nature/go2_mujoco_isaacs_v22_long/config.yaml` | ISAACS training config (bundled with weights) |
| `train_result/nature/go2_mujoco_isaacs_v22_long/model/` | ctrl/dstb/central checkpoints at step 50,600,000 |

---

## 7. Deployment Steps

### Step 1: Walk-only test
```bash
python test_mjlab_walk_numpad.py --checkpoint ckpts/mjlab_walk/model_1000.pt
```
- Test that the robot walks with the MjLab policy
- Default gains: `--kp 80,80,180 --kd 1,1,2` (hip,thigh,calf). Validated
  on hardware 2026-04-20: robot walks cleanly, 0% standstill saturation.
- Try `--dt 0.02` (50Hz, matching training) or `--dt 0.005` (200Hz)

### Step 2: Transfer safety model weights
Copy from remote PC:
```bash
scp -r remote:train_result/nature/go2_mujoco_isaacs_v22_long/ train_result/nature/go2_mujoco_isaacs_v22_long/
```
Or copy the model dir specifically:
```bash
scp -r remote:.../go2_mujoco_isaacs_v22_long/model/ train_result/nature/go2_mujoco_isaacs_v22_long/model/
scp remote:.../go2_mujoco_isaacs_v22_long/config.yaml train_result/nature/go2_mujoco_isaacs_v22_long/
```

### Step 3: Test safety policy alone
- Set `epsilon=np.inf` to always use safety controller
- Verify it maintains stable stance

### Step 4: LRSF value shielding
```bash
python test_mjlab_value_shielding_numpad.py --epsilon -0.05
```

### Step 5: RCBF filter
```bash
python test_mjlab_rcbf_numpad.py --kappa 0.995
```

---

## 8. Observation Construction Details

### For Walking Policy (from wrapper.state)

```python
# wrapper.state layout (36D, hardware order FR,FL,BR,BL):
# [0:3]   linear velocity (accelerometer * dt, rough estimate)
# [3:5]   roll, pitch
# [5:8]   angular velocity (gyroscope)
# [8:20]  joint positions (12, hardware order)
# [20:32] joint velocities (12, hardware order)
# [32:36] foot contact (4, already remapped)

# Build 47D MjLab obs:
obs[0:3]   = state[5:8]                    # angular velocity
obs[3:6]   = project_gravity(quaternion)    # from IMU quaternion
obs[6:9]   = command                        # user input
obs[9:11]  = gait_phase(t, period=0.6)      # sin/cos clock
obs[11:23] = remap(joint_pos, HW->MJ) - default_pos
obs[23:35] = remap(joint_vel, HW->MJ)
obs[35:47] = last_network_output            # clipped [-1,1]
```

### For Safety Policy (from wrapper.state)

```python
# Build 48D ISAACS obs:
obs[0:3]   = state[0:3]                    # linear velocity
obs[3:5]   = state[3:5]                    # roll, pitch
obs[5:8]   = state[5:8]                    # angular velocity
obs[8:20]  = remap(joint_pos, HW->PB)      # PyBullet order
obs[20:32] = remap(joint_vel, HW->PB)
obs[32:36] = state[32:36]                  # foot contacts
obs[36:48] = prev_ctrl_action               # from last step
```

---

## 9. Known Sim-to-Real Gaps

1. **Linear velocity:** Estimated from accelerometer × dt (noisy, drifts). The walking policy doesn't use it (good). The safety policy does use it in obs[0:3].

2. **PD gains:** Training sim uses per-joint-type gains (hip/thigh: 100/1, calf: 200/2). Hardware now runs 80/80/180 with kd=1/1/2 — ~80% of sim stiffness with training-matched damping. Earlier uniform 50/3 caused 0.4 rad calf sag at startup → out-of-distribution `joint_pos_rel` → saturated policy actions. Gap closed on 2026-04-20.

3. **Control frequency:** Training uses 50Hz (dt=0.02, frame_skip=10 with 0.002s sim step). Hardware can run faster if needed.

4. **Foot contacts:** Training uses MuJoCo contact detection. Hardware uses foot force sensors with threshold > 10N.

5. **No height scan:** The walking policy was trained with 160D height scan for critic, but actor only uses 47D without it. No issue for deployment.

---

## 10. Tuning Notes

- **Walk speed:** Adjust numpad commands. MjLab policy trained with vx ∈ [-1, 2], vy ∈ [-1, 1], wz ∈ [-1, 1].
- **Safety threshold (ε):** More negative = less conservative. Start with -0.05, adjust based on behavior.
- **RCBF kappa:** Higher = more conservative. 0.995 is a good start for dt=0.02.
- **PD gains:** If robot is too stiff, reduce kp. If too wobbly, increase kd.

---

*Last updated: 2026-04-19*
