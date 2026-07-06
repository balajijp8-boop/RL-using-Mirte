#!/usr/bin/env python3
"""
Interactive MuJoCo GUI — the real thing.

Opens a single persistent window you can orbit / zoom / pan with the mouse
(left-drag = rotate, right-drag = pan, scroll = zoom), while the robot drives
around balancing the stacked cylinders in real time.

If a trained policy exists (runs/verify or runs/ppo_mirte) it is used;
otherwise gentle random actions are applied so the scene stays alive.

Launch it through ./run_gui.sh so the X11 GLFW + AMD GPU env vars are set.
"""

import time
import glob
import numpy as np
import mujoco
import mujoco.viewer

from mirte_gimbal_env import MirteGimbalBalanceEnv as MirteStackedBalanceEnv, CTRL_DT

print("GLFW backend:",
      __import__("glfw").get_version_string().decode(errors="ignore"))

# --- optional: load a trained expert if one is on disk -----------------------
policy, vecnorm = None, None
for run in ("runs/ppo_gimbal", "runs/verify_gimbal", "runs/ppo_real", "runs/verify_real"):
    model_zip = f"{run}/ppo_final.zip"
    if glob.glob(model_zip):
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
            policy = PPO.load(model_zip, device="cpu")
            vn_path = f"{run}/vecnormalize.pkl"
            if glob.glob(vn_path):
                vecnorm = VecNormalize.load(
                    vn_path, DummyVecEnv([lambda: MirteStackedBalanceEnv()]))
                vecnorm.training = False
                vecnorm.norm_reward = False
            print(f"Loaded trained policy from {run}")
            break
        except Exception as e:
            print(f"(could not load policy from {run}: {e})")

# fixed world so the viewer window stays bound to one model across resets
env = MirteStackedBalanceEnv(randomize_on_reset=False)
obs, _ = env.reset(seed=0)


def choose_action(obs):
    if policy is None:
        return env.action_space.sample() * 0.35
    o = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
    action, _ = policy.predict(o, deterministic=True)
    return action


with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    # a pleasant starting camera (you can move it freely with the mouse)
    viewer.cam.lookat[:] = [0.0, 0.0, 0.35]
    viewer.cam.distance = 6.5
    viewer.cam.elevation = -22
    viewer.cam.azimuth = 90

    print("\n✓ Window open. Left-drag rotate · right-drag pan · scroll zoom."
          "\n  Close the window to quit.\n")

    episode, ep_r = 0, 0.0
    while viewer.is_running():
        t0 = time.time()

        action = choose_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_r += reward
        viewer.sync()

        if terminated or truncated:
            outcome = info.get("failure", "success" if info.get("success") else "timeout")
            print(f"episode {episode:2d}: {outcome:9s} | reward {ep_r:7.1f} | dist {info['dist']:.2f}")
            episode += 1
            ep_r = 0.0
            obs, _ = env.reset()

        # pace to real time
        dt = CTRL_DT - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)

env.close()
print("Window closed. Bye.")
