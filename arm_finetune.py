"""Warm-start a 7-dim policy-controlled-arm run from a 5-dim checkpoint.

Both plain fine-tuning and the speed cap saturated at ~73%; the residual failures
are payload drops. This gives the POLICY 2 extra action dims driving the shoulder
lift/pan servos (arm_action=True env). The 5-dim parent weights are transplanted
so the new policy STARTS at the parent's rate (arm outputs init ~0 -> arm holds
pose -> identical behavior) and can only improve as it learns to catch.

Safety: saves snap_00000k BEFORE training so its rate can be eval'd (must ~= parent,
proving the transplant preserved the skill) before committing hours.

Usage:
    python arm_finetune.py --from-model runs/ppo_gimbal_v22/snap_07000k \
        --vecnorm runs/ppo_gimbal_v22/snap_07000k_vecnorm.pkl \
        --lr 5e-5 --out runs/ppo_gimbal_v23arm
"""
import argparse, os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from mirte_gimbal_env import MirteGimbalBalanceEnv
from finetune_gimbal import ProgressSnapshot, SNAPSHOT_STEPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-model", required=True)
    ap.add_argument("--vecnorm", required=True)
    ap.add_argument("--steps", type=int, default=8_000_000)
    ap.add_argument("--n-envs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="transplant + save snap_00000k, then exit (no training)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def make_env():
        return MirteGimbalBalanceEnv(start_curriculum=True, arm_action=True)
    if args.dry_run:
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([make_env])
    else:
        env = SubprocVecEnv([make_env for _ in range(args.n_envs)])
    env = VecMonitor(env)
    env = VecNormalize.load(args.vecnorm, env)     # obs dim unchanged (61) -> stats valid
    env.action_space = env.venv.action_space       # pickle carried parent's 5-dim space; force 7
    env.training = True
    env.norm_reward = True
    assert env.action_space.shape[0] == 7, env.action_space

    old = PPO.load(f"{args.from_model}.zip", device="cuda")
    lr0 = args.lr
    new = PPO(
        "MlpPolicy", env, device="cuda", tensorboard_log=args.out,
        learning_rate=lambda p: lr0 * p, gamma=args.gamma,
        n_steps=old.n_steps, batch_size=old.batch_size, n_epochs=old.n_epochs,
        gae_lambda=old.gae_lambda, clip_range=0.2, ent_coef=old.ent_coef,
        vf_coef=old.vf_coef, max_grad_norm=old.max_grad_norm,
        policy_kwargs=old.policy_kwargs,
    )

    # ---- transplant: copy every shared weight; expand the action head 5 -> 7 ----
    osd, nsd = old.policy.state_dict(), new.policy.state_dict()
    new_state, copied, expanded, skipped = {}, [], [], []
    for k, nv in nsd.items():
        if k in osd and osd[k].shape == nv.shape:
            new_state[k] = osd[k].clone(); copied.append(k)
        elif k in osd:                              # action_net.* or log_std (dim0 5->7)
            ov = osd[k]; n = nv.clone(); m = ov.shape[0]
            n[:m] = ov
            if "action_net" in k:
                n[m:] = n[m:] * 0.01                 # tiny -> arm output ~0 at start
            else:                                    # log_std: match parent exploration
                n[m:] = ov.mean()
            new_state[k] = n; expanded.append(f"{k} {tuple(ov.shape)}->{tuple(nv.shape)}")
        else:
            new_state[k] = nv; skipped.append(k)
    new.policy.load_state_dict(new_state)
    print(f"transplant: {len(copied)} copied, expanded {expanded}, skipped {skipped}")

    new.save(os.path.join(args.out, "snap_00000k"))
    new.get_vec_normalize_env().save(os.path.join(args.out, "snap_00000k_vecnorm.pkl"))
    print(f"saved snap_00000k (pre-train). EVAL IT (--arm-action) before trusting the run.")
    if args.dry_run:
        print("dry-run: transplant done, skipping training."); return

    targets = [s for s in SNAPSHOT_STEPS if s <= args.steps]
    new.learn(total_timesteps=args.steps,
              callback=ProgressSnapshot(args.out, targets),
              reset_num_timesteps=True, progress_bar=True)
    new.save(f"{args.out}/ppo_final")
    env.save(f"{args.out}/vecnormalize.pkl")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
