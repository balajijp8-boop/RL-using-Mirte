#!/usr/bin/env python3
"""
Interactive multi-angle viewer: see the scene from 6 different camera positions.

Controls:
  1-6      : View angles (top, bottom, front, back, left, right)
  space    : Play/pause
  r        : Reset episode
  q        : Quit

Run: python interactive_angles.py
"""

from mirte_env import MirteStackedBalanceEnv
import numpy as np
import mujoco
from PIL import Image
import sys

CAMERAS = {
    "1_top": {"lookat": [0, 0, 0.3], "distance": 3.5, "azimuth": 0, "elevation": 90},
    "2_bottom": {"lookat": [0, 0, 0.2], "distance": 3.5, "azimuth": 0, "elevation": -80},
    "3_front": {"lookat": [0, 0, 0.3], "distance": 3.5, "azimuth": 0, "elevation": 30},
    "4_back": {"lookat": [0, 0, 0.3], "distance": 3.5, "azimuth": 180, "elevation": 30},
    "5_left": {"lookat": [0, 0, 0.3], "distance": 3.5, "azimuth": 90, "elevation": 30},
    "6_right": {"lookat": [0, 0, 0.3], "distance": 3.5, "azimuth": -90, "elevation": 30},
}

def render_with_camera(model, data, cam_config):
    """Render scene from a specific camera position."""
    renderer = mujoco.Renderer(model, height=480, width=640)
    mujoco.mj_forward(model, data)

    # Set camera parameters
    renderer.camera.lookat[:] = cam_config["lookat"]
    renderer.camera.distance = cam_config["distance"]
    renderer.camera.azimuth = cam_config["azimuth"]
    renderer.camera.elevation = cam_config["elevation"]

    renderer.update_scene(data)
    return renderer.render()

def main():
    env = MirteStackedBalanceEnv()
    obs, _ = env.reset(seed=42)

    current_cam = "3_front"
    paused = False

    print("\n🎬 Multi-angle interactive viewer")
    print("Controls:")
    print("  1-6: Switch camera (top, bottom, front, back, left, right)")
    print("  space: Play/pause")
    print("  r: Reset")
    print("  q: Quit")
    print(f"\nStarting in {current_cam} view...\n")

    frame = 0
    while True:
        try:
            # Render current view
            pixels = render_with_camera(env.model, env.data, CAMERAS[current_cam])

            # Save frame
            img_path = f"view_{current_cam}_{frame:04d}.png"
            Image.fromarray(pixels).save(img_path)

            # Get info
            pos, yaw = env.data.body("base").xpos[:2], env.data.joint("yaw").qpos[0]
            phi1 = np.arccos(np.clip(env.data.body("cyl1").xmat[8], -1, 1))
            phi2 = np.arccos(np.clip(env.data.body("cyl2").xmat[8], -1, 1))
            dist = np.linalg.norm(env._target - pos[:2])

            print(f"Frame {frame:3d} | {current_cam:10s} | "
                  f"dist={dist:5.2f} | φ₁={phi1:5.3f} | φ₂={phi2:5.3f} | "
                  f"{'[PAUSED]' if paused else '[PLAYING]'}")

            # Step if not paused
            if not paused:
                action = env.action_space.sample() * 0.35
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    outcome = info.get("failure", info.get("success", "timeout"))
                    print(f"  → Episode ended: {outcome}\n")
                    obs, _ = env.reset()

            frame += 1

            # Check for user input (every 20 frames)
            if frame % 20 == 0:
                print(f"  (Active: {current_cam} | paused={paused} | Press keys...)")

        except KeyboardInterrupt:
            print("\n✓ Interrupted. Saved frames to disk.")
            env.close()
            print("Open view_*.png files to inspect different camera angles.")
            break
        except Exception as e:
            print(f"✗ Error: {e}")
            env.close()
            break

if __name__ == "__main__":
    main()
