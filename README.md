# MIRTE Stacked-Cylinder Balance Transport (Phase 1)

# RL-using-Mirte

Double Inverted Pendulum Mobile Transport benchmark: a holonomic MIRTE-Master-like
base must carry two un-linked stacked cylinders (held loosely between rigid gripper
blades) from a start pose, around randomized pillars, through a randomized doorway
(0.75–1.2 m), to a target zone — without tipping the stack.

## Setup

```bash
source ~/venvs/mirte_rl/bin/activate
```

The venv contains: `mujoco`, `gymnasium`, `stable-baselines3[extra]`,
`torch` (cu128 build — supports the RTX 5060 / Blackwell `sm_120`).

## Files

| File | Purpose |
|---|---|
| `mirte_env.py` | `MirteStackedBalanceEnv` — MJCF is regenerated with domain randomization on every `reset()` |
| `train_ppo.py` | Phase-1 PPO training (8 parallel envs, VecNormalize, checkpoints, TensorBoard) |
| `watch.py` | Roll out a trained policy in the interactive MuJoCo viewer |

## Run

```bash
cd ~/mirte_balance_rl
python mirte_env.py                    # smoke test (random actions)
python train_ppo.py --steps 100000     # quick verification run
python train_ppo.py                    # full 3M-step training
tensorboard --logdir runs/ppo_mirte    # monitor ep_rew_mean / ep_len_mean
python watch.py runs/ppo_mirte/ppo_final.zip runs/ppo_mirte/vecnormalize.pkl
```

## Design notes

- **Observation (19-dim).** The spec listed 18 components but asked for 19; the
  19th is normalized distance-to-goal. Target offset and velocities are expressed
  in the *body frame*, which makes the policy invariant to absolute heading.
- **Cylinder tilt** is `arccos(xmat[8])` (the `R[2,2]` element), i.e. tilt of the
  body z-axis from global gravity Z. Tilt *rates* are finite-differenced at the
  50 Hz control rate.
- **Holonomic base** is modeled as slide-x / slide-y / yaw joints with velocity
  actuators (the standard planar approximation of a mecanum base — roller
  dynamics are not simulated). Actions are body-frame `[Vx, Vy, Wz]` and rotated
  to world frame before actuation. Finite actuator gain (`kv`) gives realistic
  acceleration ramps, which is what makes the balancing problem non-trivial.
- **Reward:** `5·progress − 1·φ₁² − 3·φ₂² − 0.05·‖Δa‖² − 0.01` per step,
  −15 on drop/collision (terminal), +30 on reaching the target slowly (terminal).
  The top cylinder (φ₂) is weighted 3× the bottom one.
- **VecNormalize stats** are saved next to the model and must be loaded at
  inference — the policy was trained on normalized observations.
- **Training device is CPU on purpose:** MuJoCo steps on CPU and a 256×256 MLP
  is too small to amortize GPU transfer overhead; SB3 itself recommends CPU for
  `MlpPolicy`. The GPU matters in Phase 2 (VLA fine-tuning / image encoding).
- A front-facing `realsense` camera (58° FoV) is already mounted on the base in
  the MJCF for Phase 2 data collection (`mujoco.Renderer` + the trained expert).

## Phase 2 (later)

1. Load the Phase-1 expert (`ppo_final.zip` + `vecnormalize.pkl`).
2. Roll it out with `mujoco.Renderer(model, 240, 424)` rendering the `realsense`
   camera each control step; log `(image, task prompt, expert action)` triples.
3. Package as an HF dataset; fine-tune the VLA on Kaggle/Colab T4.

## v3: Gimbal + terrain env (`mirte_gimbal_env.py`)

Real MIRTE on a free-floating base (4 caster spheres), driving over scattered
bumps and a low beveled staircase, with the arm holding the stack on a 2-axis
powered gimbal. Obs 25-dim, action 5-dim `[Vx, Vy, Wz, trim_a, trim_b]`.

Stabilization architecture (each piece was validated by ablation):
- **Gimbal leveling at 500 Hz** — keeps the tray level as the chassis pitches/rolls.
  Gimbal OFF drops the stack in <0.4 s (raw gripper mount is ~4° tilted).
- **Base cart-pole reflex** — an inverted pendulum can only be caught by moving
  its *support*: `a = 14·lean + 4·lean_rate` integrated into a leaky velocity
  offset. Tray-tilt "catching" does NOT work (high-friction payload tips *with*
  the tray) and neither does accel-feedforward tray tilt (waiter trick) — both
  are positive feedback through friction.
- **Policy** (25-obs/5-act PPO) learns navigation + speed adaptation + gimbal trim.

Scripted straight-line crossing succeeds 5/5 seeds at terrain-appropriate speed;
`python train_gimbal.py` trains (~276 fps → 3M steps ≈ 3 h).
