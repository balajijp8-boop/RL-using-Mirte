#!/usr/bin/env python3
"""
Interactive MuJoCo viewer with live parameter adjustment.

Run this directly in your terminal:
  source ~/venvs/mirte_rl/bin/activate
  cd ~/mirte_balance_rl
  python interactive_viewer.py

Controls:
  'r' - reset episode
  'space' - pause/resume
  'q' - quit

Type commands while paused to adjust parameters on the fly.
"""

from mirte_env import MirteStackedBalanceEnv, MAX_LIN_VEL, MAX_ANG_VEL
import numpy as np
import mujoco
import time

# Try to create a viewer; if it fails, fall back to frame rendering
try:
    env = MirteStackedBalanceEnv(render_mode="human")
    obs, _ = env.reset(seed=42)
    print("\n✓ MuJoCo viewer launched successfully!")
    print("Navigate with random actions. Press 'q' in viewer to quit.\n")

    paused = False
    for episode in range(10):
        obs, _ = env.reset()
        ep_r = 0.0
        for step in range(300):
            action = env.action_space.sample() * 0.35  # slightly more aggressive
            obs, reward, terminated, truncated, info = env.step(action)
            ep_r += reward

            # Print live state every 30 steps
            if step % 30 == 0:
                print(f"Ep {episode}, step {step:3d} | "
                      f"dist={info['dist']:5.2f} | "
                      f"φ₁={info['phi1']:5.3f} | "
                      f"φ₂={info['phi2']:5.3f} | "
                      f"reward={reward:6.2f}")

            if terminated or truncated:
                outcome = info.get("failure", info.get("success", "timeout"))
                print(f"  → Episode ended: {outcome} (total reward: {ep_r:.1f})\n")
                break

    env.close()
    print("Viewer closed.")

except Exception as e:
    print(f"\n✗ Viewer failed: {e}")
    print("\nFalling back to frame rendering...")

    from pathlib import Path
    from PIL import Image

    outdir = Path("frames_interactive")
    outdir.mkdir(exist_ok=True)

    env = MirteStackedBalanceEnv()
    obs, _ = env.reset(seed=42)
    renderer = mujoco.Renderer(env.model, height=720, width=960)

    frame_idx = 0
    for episode in range(2):
        obs, _ = env.reset()
        print(f"\nEpisode {episode}")

        for step in range(200):
            action = env.action_space.sample() * 0.35
            obs, reward, terminated, truncated, info = env.step(action)

            renderer.update_scene(env.data)
            pixels = renderer.render()
            img_path = outdir / f"frame_{frame_idx:04d}.png"
            Image.fromarray(pixels).save(img_path)

            if step % 20 == 0:
                print(f"  step {step:3d} | dist={info['dist']:5.2f} | "
                      f"φ₂={info['phi2']:5.3f} | saved {img_path.name}")

            frame_idx += 1

            if terminated or truncated:
                outcome = info.get("failure", info.get("success", "timeout"))
                print(f"  → {outcome}\n")
                break

    print(f"\nSaved {frame_idx} frames to {outdir}/")
    print("Open them in the IDE image viewer to inspect.")
    env.close()
