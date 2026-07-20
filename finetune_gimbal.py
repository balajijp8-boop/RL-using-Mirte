"""Fine-tune a trained gimbal policy under upgraded physics (tray remount +
speed cap), continuing from an existing checkpoint instead of from scratch.

Differences vs resume_gimbal.py (which continues an interrupted run):
  - fresh fine-tune LR schedule (linear decay from --lr), NOT the dregs of the
    original run's schedule (which has decayed to ~0 by the end of a full run)
  - reset_num_timesteps=True: snapshots/tensorboard number from 0 for this run
  - env kwargs for the new physics (tray mount, curriculum stays on)

Usage:
    python finetune_gimbal.py --from-model runs/ppo_gimbal_v4/ppo_final \
        --tray-mount-y -0.04 --tray-drop 0.06 --steps 8000000
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from mirte_gimbal_env import MirteGimbalBalanceEnv

SNAPSHOT_STEPS = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000,
                  1_500_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000,
                  6_000_000, 7_000_000, 8_000_000]


class ProgressSnapshot(BaseCallback):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-model", required=True,
                    help="checkpoint to fine-tune from (no .zip)")
    ap.add_argument("--vecnorm", default=None,
                    help="vecnormalize stats (default: <from-model dir>/vecnormalize.pkl)")
    ap.add_argument("--steps", type=int, default=8_000_000)
    ap.add_argument("--n-envs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=None,
                    help="override discount (e.g. 0.995 for long-horizon)")
    # defaults MUST match the approved robot geometry (env defaults)
    ap.add_argument("--tray-mount-y", type=float, default=0.03)
    ap.add_argument("--tray-drop", type=float, default=0.06)
    ap.add_argument("--out", type=str, default="runs/ppo_gimbal_v5_ft")
    ap.add_argument("--no-curriculum", action="store_true",
                    help="train 100%% full-course standard starts (polish mode: "
                    "matches the eval distribution; curriculum's value-propagation "
                    "job is done once the policy already completes the course)")
    args = ap.parse_args()

    vecnorm = args.vecnorm or os.path.join(
        os.path.dirname(args.from_model), "vecnormalize.pkl")
    os.makedirs(args.out, exist_ok=True)

    def make_env():
        return MirteGimbalBalanceEnv(start_curriculum=not args.no_curriculum,
                                     tray_mount_y=args.tray_mount_y,
                                     tray_drop=args.tray_drop)

    env = SubprocVecEnv([make_env for _ in range(args.n_envs)])
    env = VecMonitor(env)
    env = VecNormalize.load(vecnorm, env)
    env.training = True                 # stats keep adapting to new dynamics
    env.norm_reward = True

    lr0 = args.lr
    overrides = {  # fresh fine-tune schedules, not the old decayed ones
        "learning_rate": lambda progress_remaining: lr0 * progress_remaining,
        "clip_range": lambda _: 0.2,
    }
    if args.gamma is not None:
        overrides["gamma"] = args.gamma
    model = PPO.load(
        f"{args.from_model}.zip", env=env, device="cuda",
        tensorboard_log=args.out,
        custom_objects=overrides,
    )
    print(f"fine-tuning from {args.from_model} "
          f"(tray_mount_y={args.tray_mount_y}, tray_drop={args.tray_drop}, "
          f"lr {lr0} linear, {args.n_envs} envs, {args.steps:,} steps)")

    model.save(os.path.join(args.out, "snap_00000k"))
    model.get_vec_normalize_env().save(
        os.path.join(args.out, "snap_00000k_vecnorm.pkl"))

    targets = [s for s in SNAPSHOT_STEPS if s <= args.steps]
    model.learn(total_timesteps=args.steps,
                callback=ProgressSnapshot(args.out, targets),
                reset_num_timesteps=True, progress_bar=True)

    model.save(f"{args.out}/ppo_final")
    env.save(f"{args.out}/vecnormalize.pkl")
    print(f"saved model and normalization stats to {args.out}/")


if __name__ == "__main__":
    main()
