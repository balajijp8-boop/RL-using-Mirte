"""Render static close-ups of the payload grip for 3 configs, for visual approval
BEFORE any training: (A) current baseline, (B) tray-jaws clamp, (C) finger clamp.
Both clamp modes also use the shorter 2cm tray_drop."""
import os
import numpy as np
import mujoco
from PIL import Image
from mirte_gimbal_env import MirteGimbalBalanceEnv

OUT = os.path.join(os.path.dirname(__file__), "grips")
os.makedirs(OUT, exist_ok=True)
W, H = 1000, 760

CONFIGS = [
    ("A_baseline_6cm_open", dict()),                                   # current
    ("B_trayjaws_2cm",      dict(clamp_mode="tray_jaws", tray_drop=0.02)),
    ("C_fingers_2cm",       dict(clamp_mode="fingers",   tray_drop=0.02)),
]

for name, kw in CONFIGS:
    env = MirteGimbalBalanceEnv(randomize_on_reset=True, **kw)
    env.reset(seed=1)
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(50):                       # settle the stack into the grip
        env.step(a)
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False   # hide lidar rays
    r = mujoco.Renderer(env.model, height=H, width=W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    c1 = np.array(env.data.body("cyl1").xpos)
    c2 = np.array(env.data.body("cyl2").xpos)
    # look at the grip region (lower cylinder / tray), not the stack midpoint
    cam.lookat[:] = [c1[0], c1[1], c1[2] - 0.04]
    cam.distance, cam.elevation, cam.azimuth = 0.5, -6, 118
    r.update_scene(env.data, cam, scene_option=vopt)
    Image.fromarray(r.render()).save(os.path.join(OUT, f"{name}.png"))
    print(f"{name:22s} cyl1z={c1[2]:.3f} cyl2z={c2[2]:.3f} gap={c2[2]-c1[2]:.3f} -> saved")
    r.close()
    env.close()
print("done ->", OUT)
