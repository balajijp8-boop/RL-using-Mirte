#!/usr/bin/env python3
"""
Stream viewer: save high-res frames in real-time to disk.
View the latest frame in your IDE image viewer to see live updates.

Run this in your terminal:
  source ~/venvs/mirte_rl/bin/activate
  cd ~/mirte_balance_rl
  python stream_viewer.py

Then open latest_frame.png in the IDE and hit F5 to refresh.
"""

from mirte_env import MirteStackedBalanceEnv
import numpy as np
import mujoco
from PIL import Image
from pathlib import Path
import time

outdir = Path("stream_frames")
outdir.mkdir(exist_ok=True)

env = MirteStackedBalanceEnv()
obs, _ = env.reset()  # must init model first
renderer = mujoco.Renderer(env.model, height=480, width=640)

try:
    print("\n🎬 Frame streamer started. Open 'latest_frame.png' in IDE and refresh to see live.")
    print("   Each frame saves automatically every 0.2s (real time)\n")

    frame_count = 0
    last_save = 0

    for episode in range(20):
        if episode > 0:
            obs, _ = env.reset()
        print(f"Episode {episode}", end="", flush=True)

        ep_r = 0.0
        for step in range(300):
            # Gentle random actions
            action = env.action_space.sample() * 0.35
            obs, reward, terminated, truncated, info = env.step(action)
            ep_r += reward

            # Save frame every ~10 physics steps (0.02 s wall time)
            now = time.time()
            if now - last_save > 0.02:
                renderer.update_scene(env.data)
                pixels = renderer.render()

                # Save to tmp file first, then rename (atomic)
                tmp_path = outdir / "_tmp.png"
                Image.fromarray(pixels).save(tmp_path)
                latest_path = outdir / "latest_frame.png"
                tmp_path.replace(latest_path)

                if frame_count % 50 == 0:
                    print(".", end="", flush=True)

                frame_count += 1
                last_save = now

            # Print stats every 30 steps
            if step % 30 == 0:
                print(f"\n  step {step:3d} | dist={info['dist']:5.2f} | "
                      f"φ₁={info['phi1']:5.3f} | φ₂={info['phi2']:5.3f}", end="", flush=True)

            if terminated or truncated:
                outcome = info.get("failure", info.get("success", "timeout"))
                print(f"\n  → {outcome} (reward: {ep_r:7.1f})")
                break

    env.close()
    print(f"\n✓ Done. Saved {frame_count} frames to {outdir}/")

except KeyboardInterrupt:
    print("\n\nInterrupted.")
    env.close()
