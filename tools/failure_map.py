"""WHERE does the policy die? Failure autopsy over N episodes.

Logs position, cause, speed at failure + course section. Use this (not
aggregate stats) to decide WHAT to fix next.

Usage:
    python tools/failure_map.py --snap runs/ppo_gimbal_v11/snap_04000k --episodes 30
"""
import argparse
import numpy as np
from collections import Counter, defaultdict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mirte_gimbal_env import MirteGimbalBalanceEnv


def section(x, stair_band):
    if x < -2.6: return "start_area"
    if x < -0.55: return "pillar_field"
    if x < 0.45: return "doorway"
    if stair_band[0] <= x <= stair_band[1]: return "stairs"
    if x < stair_band[0]: return "pre_stairs"
    return "post_stairs"


ap = argparse.ArgumentParser()
ap.add_argument("--snap", required=True)
ap.add_argument("--episodes", type=int, default=30)
args = ap.parse_args()

model = PPO.load(f"{args.snap}.zip", device="cpu")
vn_path = f"{args.snap}_vecnorm.pkl"
if not os.path.exists(vn_path):
    vn_path = os.path.join(os.path.dirname(args.snap), "vecnormalize.pkl")
vn = VecNormalize.load(vn_path, DummyVecEnv([lambda: MirteGimbalBalanceEnv()]))
vn.training = False

sec_cause = defaultdict(Counter)
speeds_at_drop, phis = [], []
for ep in range(args.episodes):
    env = MirteGimbalBalanceEnv(randomize_on_reset=True)
    obs, _ = env.reset(seed=3000 + ep)
    done, info, s = False, {}, 0
    while not done and s < 2500:
        a, _ = model.predict(vn.normalize_obs(obs), deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        s += 1
    x, y = env.data.body("base_link").xpos[:2]
    vel = env._base_vel()
    speed = float(np.hypot(vel[3], vel[4]))
    cause = info.get("failure", "SUCCESS" if info.get("success") else "timeout")
    sec = section(float(x), env._stair_band)
    sec_cause[sec][cause] += 1
    if cause == "dropped":
        speeds_at_drop.append(speed)
        phis.append(np.degrees(info.get("phi2", 0)))
    print(f"ep {ep+1:2d}: {cause:10s} at x={x:+5.2f} ({sec:12s}) speed {speed:.2f}")
    env.close()

print("\n===== failure map =====")
for sec in ("start_area", "pillar_field", "doorway", "pre_stairs", "stairs", "post_stairs"):
    if sec_cause[sec]:
        print(f"{sec:14s}: {dict(sec_cause[sec])}")
if speeds_at_drop:
    print(f"\ndrops: avg speed {np.mean(speeds_at_drop):.2f} m/s, avg phi2 {np.mean(phis):.0f} deg")
