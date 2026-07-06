"""
Phase 1: train the vector-state "Expert Driver" with PPO.

Usage:
    python train_ppo.py                    # full run (3M steps)
    python train_ppo.py --steps 100000     # short verification run

An MLP policy on a 19-dim observation trains fastest on CPU (SB3 itself
recommends device="cpu" for MlpPolicy); the GPU is only needed in Phase 2.
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from mirte_real_env import MirteRealBalanceEnv as MirteStackedBalanceEnv


def make_env():
    return MirteStackedBalanceEnv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--out", type=str, default="runs/ppo_real")
    args = parser.parse_args()

    env = SubprocVecEnv([make_env for _ in range(args.n_envs)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=args.out,
        device="cpu",
        verbose=1,
    )

    ckpt = CheckpointCallback(save_freq=max(100_000 // args.n_envs, 1),
                              save_path=args.out, name_prefix="ppo")
    model.learn(total_timesteps=args.steps, callback=ckpt, progress_bar=True)

    model.save(f"{args.out}/ppo_final")
    env.save(f"{args.out}/vecnormalize.pkl")   # needed at inference time!
    print(f"saved model and normalization stats to {args.out}/")


if __name__ == "__main__":
    main()
