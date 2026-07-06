"""
Offline render: save PNG frames from the environment to disk.
"""

from mirte_env import MirteStackedBalanceEnv
import numpy as np
import mujoco
from pathlib import Path

outdir = Path("frames")
outdir.mkdir(exist_ok=True)

env = MirteStackedBalanceEnv()
obs, _ = env.reset(seed=42)

renderer = mujoco.Renderer(env.model, height=480, width=640)

frame_idx = 0
for episode in range(3):
    obs, _ = env.reset()
    print(f"Episode {episode}")

    for step in range(150):
        action = env.action_space.sample() * 0.3
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 10 == 0:  # save every 10 steps
            renderer.update_scene(env.data)
            pixels = renderer.render()
            img_path = outdir / f"frame_{frame_idx:04d}.png"

            from PIL import Image
            Image.fromarray(pixels).save(img_path)
            print(f"  saved {img_path} | dist={info['dist']:.2f}, phi2={info['phi2']:.3f}")
            frame_idx += 1

        if terminated or truncated:
            print(f"  -> {info.get('failure', info.get('success', 'timeout'))}")
            break

env.close()
print(f"\nSaved {frame_idx} frames to {outdir}/")
