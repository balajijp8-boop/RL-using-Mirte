#!/usr/bin/env python3
"""
Run a trained MIRTE gimbal policy interactively in MuJoCo viewer.

Usage:
    python inference.py --model runs/ppo_gimbal/ppo_final.zip --seed 42 --render
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from mirte_gimbal_env import MirteGimbalBalanceEnv
import mujoco
import mujoco.viewer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/ppo_gimbal/ppo_final.zip")
    parser.add_argument("--vecnorm", default="runs/ppo_gimbal/vecnormalize.pkl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    # Load trained policy
    print(f"Loading policy from {args.model}...")
    model = PPO.load(args.model, device="cpu")

    # Load observation normalization stats
    vn = None
    try:
        vn = VecNormalize.load(args.vecnorm,
                               DummyVecEnv([lambda: MirteGimbalBalanceEnv()]))
        vn.training = False
        vn.norm_reward = False
        print(f"Loaded normalization from {args.vecnorm}")
    except:
        print("WARNING: Could not load vecnormalize stats; running without normalization")

    # Run episodes
    np.random.seed(args.seed)
    total_reward, total_steps = 0.0, 0

    for episode in range(args.episodes):
        env = MirteGimbalBalanceEnv(randomize_on_reset=True, gimbal_enabled=True)
        obs, _ = env.reset(seed=args.seed + episode)

        ep_reward, ep_steps, done = 0.0, 0, False

        if args.render:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                viewer.cam.lookat[:] = [0.0, 0.0, 0.35]
                viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 6.5, -22, 90

                while not done and ep_steps < args.max_steps:
                    o_norm = vn.normalize_obs(obs) if vn else obs
                    action, _ = model.predict(o_norm, deterministic=args.deterministic)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    ep_reward += reward
                    ep_steps += 1

                    viewer.sync()
                    if done:
                        break
        else:
            # Headless rollout
            while not done and ep_steps < args.max_steps:
                o_norm = vn.normalize_obs(obs) if vn else obs
                action, _ = model.predict(o_norm, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_steps += 1

        env.close()
        total_reward += ep_reward
        total_steps += ep_steps

        outcome = info.get("failure") or ("success" if info.get("success") else "timeout")
        print(f"Episode {episode+1}: {outcome:10s} | "
              f"reward {ep_reward:7.1f} | steps {ep_steps:4d} | dist {info['dist']:5.2f}")

    avg_reward = total_reward / args.episodes
    avg_steps = total_steps / args.episodes
    print(f"\nAverage: reward {avg_reward:.1f}, steps {avg_steps:.0f}")


if __name__ == "__main__":
    main()
