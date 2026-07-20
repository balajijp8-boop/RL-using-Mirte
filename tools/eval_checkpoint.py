"""Headless eval of a training checkpoint (snap_XXXXk or ppo_final).

Protocol: 30 randomized episodes, deterministic policy, REAL success/drop/
collision/stall counts. Never trust the SB3 training curve alone (v2 looked
fine on curves and scored 0/20). Use MEDIAN / trimmed mean for decisions --
single outlier episodes can swing the mean by +-10.

Usage:
    python tools/eval_checkpoint.py --snap runs/ppo_gimbal_v11/snap_01000k --episodes 30
"""
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mirte_gimbal_env import MirteGimbalBalanceEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    # 2500 matches env truncation: full course at safe cruise needs 1400+ steps
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--arm-catch", choices=["on", "off"], default=None,
                    help="override arm-translation catch (default: env default)")
    ap.add_argument("--wall-hh", type=float, default=None,
                    help="override box wall half-height (default: env default)")
    ap.add_argument("--blade-density", type=float, default=None,
                    help="override box-wall density (original 1000)")
    ap.add_argument("--arm-action", action="store_true",
                    help="7-dim policy-controlled arm env (must match the checkpoint)")
    ap.add_argument("--seed-base", type=int, default=1000,
                    help="episode seeds start here (default 1000). Use a "
                    "different base for a genuinely independent confirmation "
                    "sample -- re-running with the default base always "
                    "replays the identical scenarios (fixed seeding).")
    args = ap.parse_args()

    envkw = {}
    if args.arm_catch is not None:
        envkw["arm_catch"] = (args.arm_catch == "on")
    if args.wall_hh is not None:
        envkw["wall_hh"] = args.wall_hh
    if args.blade_density is not None:
        envkw["blade_density"] = args.blade_density
    if args.arm_action:
        envkw["arm_action"] = True

    model = PPO.load(f"{args.snap}.zip", device="cpu")
    vn_path = f"{args.snap}_vecnorm.pkl"
    if not os.path.exists(vn_path):
        vn_path = os.path.join(os.path.dirname(args.snap), "vecnormalize.pkl")
    vn = VecNormalize.load(vn_path, DummyVecEnv([lambda: MirteGimbalBalanceEnv(**envkw)]))
    vn.training = False

    n_success = n_timeout = 0
    fail_causes = {}
    rewards, steps, dists = [], [], []

    for ep in range(args.episodes):
        env = MirteGimbalBalanceEnv(randomize_on_reset=True, **envkw)  # curriculum OFF: full course
        obs, _ = env.reset(seed=args.seed_base + ep)
        ep_r, ep_s, done, info = 0.0, 0, False, {}
        while not done and ep_s < args.max_steps:
            o = vn.normalize_obs(obs)
            action, _ = model.predict(o, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            ep_r += r
            ep_s += 1
        env.close()

        if info.get("success"):
            outcome, n_success = "SUCCESS", n_success + 1
        elif info.get("failure"):
            c = info["failure"]
            outcome = f"FAIL:{c}"
            fail_causes[c] = fail_causes.get(c, 0) + 1
        else:
            outcome, n_timeout = "timeout", n_timeout + 1
        rewards.append(ep_r); steps.append(ep_s)
        dists.append(info.get("dist", float("nan")))
        print(f"ep {ep+1:2d}: {outcome:16s} reward {ep_r:8.1f}  steps {ep_s:4d}  "
              f"dist {info.get('dist', float('nan')):5.2f}")

    n = args.episodes
    r = np.array(rewards)
    k = max(1, n // 10)
    trimmed = np.sort(r)[k:-k]
    print("-" * 60)
    print(f"checkpoint      : {args.snap}")
    print(f"episodes        : {n}")
    print(f"success         : {n_success}/{n}  ({100*n_success/n:.0f}%)")
    print(f"failure causes  : {fail_causes}")
    print(f"timeout         : {n_timeout}/{n}")
    print(f"avg reward      : {r.mean():.1f}   median {np.median(r):.1f}   "
          f"trimmed(10%) {trimmed.mean():.1f}")
    print(f"avg ep length   : {np.mean(steps):.0f} steps")
    print(f"avg final dist  : {np.mean(dists):.2f} m  (0 = goal, ~6.5 = no progress)")


if __name__ == "__main__":
    main()
