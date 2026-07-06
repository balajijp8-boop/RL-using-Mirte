# MIRTE Gimbal Balance RL — Windows GPU Setup

**TL;DR:** Clone, create a Python 3.10 venv, install dependencies with GPU support, run training.

## Prerequisites

- **Python 3.10** (3.11+ may have compatibility issues with older packages)
- **CUDA 12.8** (RTX 5060 / Blackwell support)
- **Git** (for cloning)
- **7+ GB disk** (for venv + snapshots + final model)
- **GPU with CUDA support** (RTX 5060, RTX 4090, etc.)

## Step 1: Clone the Repository

```bash
git clone https://github.com/balajijp8-boop/RL-using-Mirte.git
cd RL-using-Mirte
```

## Step 2: Create a Python 3.10 Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

On PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## Step 3: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel

# Core RL stack
pip install gymnasium stable-baselines3 numpy

# Physics engine with GPU acceleration
pip install "torch>=2.0" "cuda-python" mujoco

# Video rendering (LinkedIn montage)
pip install Pillow imageio imageio-ffmpeg

# Optional: TensorBoard for training curves
pip install tensorboard
```

### GPU-Specific Notes

- **PyTorch + CUDA 12.8:** By default, `pip install torch` fetches cu121. To force cu128 (Blackwell):
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  ```

- **MuJoCo:** The pip version includes CPU-only bindings by default. For GPU physics:
  ```bash
  pip uninstall mujoco -y
  pip install mujoco --prefer-binary --no-build-isolation
  ```
  Then verify:
  ```bash
  python -c "import mujoco; print('MuJoCo OK')"
  ```

## Step 4: Verify Installation

```bash
python -c "
import torch
import mujoco
import gymnasium
import stable_baselines3
print('✓ torch:', torch.__version__, '(GPU)' if torch.cuda.is_available() else '(CPU only)')
print('✓ mujoco:', mujoco.__version__)
print('✓ gymnasium:', gymnasium.__version__)
print('✓ stable_baselines3: OK')
"
```

## Step 5: Train the Policy

### Full Training (3M steps, ~3–4 hours on RTX 5060)

```bash
cd mirte_balance_rl
python train_gimbal.py --steps 3000000 --n-envs 8 --out runs/ppo_gimbal
```

**Flags:**
- `--steps 3000000`: Total training steps (default). Reduce to `100000` for a quick test.
- `--n-envs 8`: Number of parallel environments. Tune for your GPU VRAM:
  - RTX 5060 (12 GB): `--n-envs 8` (safe, ~275 fps)
  - RTX 4090 (24 GB): `--n-envs 16` (~500 fps)
  - Less VRAM: reduce to `4` or `6`
- `--out runs/ppo_gimbal`: Output directory for checkpoints

### Monitor Training

In another terminal:
```bash
cd mirte_balance_rl
tensorboard --logdir runs/ppo_gimbal
```
Then open http://localhost:6006 in your browser.

### Snapshots & Video

Training auto-saves policy snapshots at: 10k, 25k, 50k, 100k, 200k, 350k, 500k, 750k, 1M, 1.5M, 2M, 2.5M, 3M steps.

After training finishes, render the LinkedIn-style learning progression video:
```bash
set MUJOCO_GL=egl
python record_progress.py --out mirte_rl_progress.mp4
```

The video shows:
- Random policy: **topples the cylinders in <1 second** (high entropy)
- Mid-training: **wobbles but recovers** (lower entropy)
- Trained policy: **smoothly drives to goal, tray stays level** (low entropy)

Video file: `mirte_rl_progress.mp4` (960×540, ~12 sec, ready to post on LinkedIn)

## Environment Details

### Observation Space (25-dim)

```
[ 0- 2]  Target position (body frame) + heading error
[ 3- 5]  Body velocity: Vx, Vy, yaw rate
[ 6-13]  8 LiDAR ranges (0°, 45°, 90°, ..., 315°)
[14-17]  Cylinder tilts & rates: phi1, dphi1, phi2, dphi2
[18]     Distance to goal
[19-20]  Base roll, pitch (from caster contact)
[21-22]  Gimbal joint angles: gimbal_a, gimbal_b
[23-24]  Tray tilt from vertical + tilt rate
```

### Action Space (5-dim)

```
[0]  Forward velocity (-0.8 to +0.8 m/s)
[1]  Lateral velocity (-0.8 to +0.8 m/s)
[2]  Yaw rate (-1.2 to +1.2 rad/s)
[3]  Gimbal_a trim (-0.15 to +0.15 rad, anticipatory)
[4]  Gimbal_b trim (-0.15 to +0.15 rad, anticipatory)
```

### Task

Navigate from start (x = -3.3 m) to goal (x = +3.2 m) through:
1. Randomized doorway (0.75–1.2 m wide)
2. Randomized pillars (obstacle avoidance)
3. **Rough terrain:** scattered bumps + low staircase (10–16 mm rise)

**Constraint:** Keep the two stacked cylinders balanced on the tray. Tipping either cylinder = **episode failure**.

**Reward Shaping:**

```python
reward = 5.0 * progress                 # +5 per meter closer to goal
       - 1.0 * (phi1 ** 2)              # penalty: bottom cylinder tilt
       - 3.0 * (phi2 ** 2)              # penalty: top cylinder tilt (3x weight)
       - 1.0 * (tray_tilt ** 2)         # penalty: gimbal not level
       - 0.05 * jerk                    # smooth actions
       - 0.01                           # time penalty
       
if success:    reward += 30.0
if failure:    reward -= 15.0
```

## Troubleshooting

### "MuJoCo: could not load library"
- Install `libglfw3`: `pip install glfw` (Windows: auto-bundled)
- Set environment variable: `set MUJOCO_GL=egl` (if X11/Wayland issues)

### GPU not detected during training
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
If False, re-install torch with correct CUDA version.

### Out of Memory (OOM)
- Reduce `--n-envs` (fewer parallel environments = less memory)
- Reduce batch size in code: `batch_size=256` (line 38 of train_gimbal.py)

### Rendering error when recording video
- Ensure ffmpeg is installed and in PATH: `ffmpeg -version`
- On Windows: Download ffmpeg from https://ffmpeg.org/download.html, add to PATH

## Next Steps

### Phase 2: Vision Language Model Fine-Tuning (Optional)

After training completes, collect an image dataset:

```bash
python render_dataset.py --policy runs/ppo_gimbal/ppo_final.zip --num-episodes 100 --out dataset/
```

Then fine-tune a VLA (e.g., OpenVLA, TinyVLA) on a Colab GPU.

### Real Robot Deployment (Advanced)

The trained policy operates on:
- **Goal position** (2-dim): provided by planning layer
- **LiDAR** (8-dim): onboard
- **IMU** (roll, pitch): onboard
- **Proprioception** (gimbal angles, cylinder pose est.): from arm + base sensors

Deploy via ROS2 bridge with the real MIRTE Master.

## References

- **MuJoCo:** https://mujoco.readthedocs.io/
- **Gymnasium:** https://gymnasium.farama.org/
- **Stable-Baselines3 PPO:** https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- **MIRTE Master platform:** https://www.tu-delft.nl/
