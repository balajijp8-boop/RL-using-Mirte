"""
Phase 1: train the gimbal-stabilizing "Expert Driver" with PPO.

This variant snapshots the policy AND its VecNormalize stats at a schedule of
timesteps chosen for a learning-progression video (dense early, where the
behaviour changes fastest). record_progress.py replays those snapshots into a
"watch RL learn" montage.

Usage:
    python train_gimbal.py                    # full run (3M steps)
    python train_gimbal.py --steps 100000     # short verification run
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from mirte_gimbal_env import MirteGimbalBalanceEnv as MirteStackedBalanceEnv


# timesteps at which to snapshot (model + normalization) for the montage.
# dense early because that is where the policy goes from flailing -> competent.
SNAPSHOT_STEPS = [10_000, 25_000, 50_000, 100_000, 200_000, 350_000,
                  500_000, 750_000, 1_000_000, 1_500_000, 2_000_000,
                  2_500_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000,
                  8_000_000, 10_000_000, 12_000_000, 14_000_000, 16_000_000]


class ProgressSnapshot(BaseCallback):
    """Save (model, vecnormalize) each time training crosses a target step."""

    def __init__(self, out_dir, targets):
        super().__init__()
        self.out_dir = out_dir
        self.targets = sorted(targets)
        self._idx = 0

    def _save(self, step):
        tag = f"{step // 1000:05d}k"
        self.model.save(os.path.join(self.out_dir, f"snap_{tag}.zip"))
        vn = self.model.get_vec_normalize_env()
        if vn is not None:
            vn.save(os.path.join(self.out_dir, f"snap_{tag}_vecnorm.pkl"))
        print(f"[snapshot] saved snap_{tag} at {step} steps")

    def _on_step(self):
        while self._idx < len(self.targets) and \
                self.num_timesteps >= self.targets[self._idx]:
            self._save(self.targets[self._idx])
            self._idx += 1
        return True


def make_env():
    # start_curriculum: TRAINING ONLY -- 65% of episodes spawn at a random
    # collision-free pose along the course (reverse curriculum). Eval and
    # video scripts construct the env without it, so success still means
    # "crossed the full course from the standard start".
    return MirteStackedBalanceEnv(start_curriculum=True)


def linear_lr(initial):
    """Linear decay to 0. SB3 calls this with progress_remaining in [1, 0].

    v3 post-mortem: constant 3e-4 with 10 epochs/rollout produced
    clip_fraction 0.52-0.63 and approx_kl 0.09-0.16 (healthy: <0.3, <0.05) --
    updates so aggressive that policy std collapsed 0.99 -> 0.117 and
    exploration died before navigation was learned."""
    return lambda progress_remaining: initial * progress_remaining


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16_000_000)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--out", type=str, default="runs/ppo_gimbal_v4")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    env = SubprocVecEnv([make_env for _ in range(args.n_envs)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_lr(2.5e-4),   # was constant 3e-4 (see linear_lr)
        n_steps=2048,
        batch_size=512,
        n_epochs=5,                        # was 10: halve update aggressiveness
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.006,                    # was 0.001: sustain exploration
        target_kl=0.04,                    # early-stop epochs on KL blowup
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=args.out,
        device="cuda",
        verbose=1,
    )

    # snapshot the untrained (random) policy first -> the "before" clip
    model.save(os.path.join(args.out, "snap_00000k.zip"))
    model.get_vec_normalize_env().save(
        os.path.join(args.out, "snap_00000k_vecnorm.pkl"))
    print("[snapshot] saved snap_00000k (random init)")

    targets = [s for s in SNAPSHOT_STEPS if s <= args.steps]
    model.learn(total_timesteps=args.steps,
                callback=ProgressSnapshot(args.out, targets),
                progress_bar=True)

    model.save(f"{args.out}/ppo_final")
    env.save(f"{args.out}/vecnormalize.pkl")   # needed at inference time!
    print(f"saved model and normalization stats to {args.out}/")


if __name__ == "__main__":
    main()
