"""
Watch a trained policy in the interactive MuJoCo viewer.

Usage:
    python watch.py runs/ppo_mirte/ppo_final.zip runs/ppo_mirte/vecnormalize.pkl
"""

import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mirte_env import MirteStackedBalanceEnv, CTRL_DT


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo_mirte/ppo_final.zip"
    vecnorm_path = sys.argv[2] if len(sys.argv) > 2 else "runs/ppo_mirte/vecnormalize.pkl"

    env = DummyVecEnv([lambda: MirteStackedBalanceEnv(render_mode="human")])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, device="cpu")

    obs = env.reset()
    ep_reward, episodes = 0.0, 0
    while episodes < 10:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        ep_reward += float(reward[0])
        time.sleep(CTRL_DT)
        if done[0]:
            info = infos[0]
            outcome = ("SUCCESS" if info.get("success")
                       else info.get("failure", "timeout"))
            print(f"episode {episodes}: {outcome}, reward {ep_reward:.1f}")
            ep_reward = 0.0
            episodes += 1


if __name__ == "__main__":
    main()
