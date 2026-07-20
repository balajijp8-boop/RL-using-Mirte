"""Resume PPO training from a saved snapshot after an interrupted run.

Loads the model + VecNormalize stats from a snapshot, rebuilds a FRESH vec env
(new subprocess workers, not tied to any crashed ones), and continues learn()
toward the original total_timesteps target. SB3 restores num_timesteps from the
checkpoint, so total_timesteps is the ORIGINAL run's target, not "steps left".

Usage:
    python resume_gimbal.py --snap runs/ppo_gimbal_v3/snap_08000k --total-steps 16000000 --n-envs 12
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from mirte_gimbal_env import MirteGimbalBalanceEnv as MirteStackedBalanceEnv

SNAPSHOT_STEPS = [10_000, 25_000, 50_000, 100_000, 200_000, 350_000,
                  500_000, 750_000, 1_000_000, 1_500_000, 2_000_000,
                  2_500_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000,
                  8_000_000, 10_000_000, 12_000_000, 14_000_000, 16_000_000]


class ProgressSnapshot(BaseCallback):
    def __init__(self, out_dir, targets, already_done):
        super().__init__()
        self.out_dir = out_dir
        self.targets = sorted(t for t in targets if t > already_done)
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
    return MirteStackedBalanceEnv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--total-steps", type=int, default=16_000_000)
    ap.add_argument("--n-envs", type=int, default=12)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    out_dir = args.out or os.path.dirname(args.snap)

    env = SubprocVecEnv([make_env for _ in range(args.n_envs)])
    env = VecMonitor(env)
    env = VecNormalize.load(f"{args.snap}_vecnorm.pkl", env)
    env.training = True
    env.norm_reward = True

    model = PPO.load(f"{args.snap}.zip", env=env, device="cuda",
                     tensorboard_log=out_dir)
    print(f"resumed from {args.snap}.zip at {model.num_timesteps:,} steps "
          f"-> target {args.total_steps:,} ({args.n_envs} envs)")

    model.learn(total_timesteps=args.total_steps,
               callback=ProgressSnapshot(out_dir, SNAPSHOT_STEPS, model.num_timesteps),
               reset_num_timesteps=False, progress_bar=True)

    model.save(f"{out_dir}/ppo_final")
    env.save(f"{out_dir}/vecnormalize.pkl")
    print(f"saved model and normalization stats to {out_dir}/")


if __name__ == "__main__":
    main()
