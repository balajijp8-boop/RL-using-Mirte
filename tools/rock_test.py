"""Isolation rock-test for the payload mount (no policy, no learning).

Shakes the base in place with an aggressive full-speed velocity-reversal +
yaw-whip profile (the worst the slew limiter allows) on FLAT ground, and
counts how many control steps the box+2-cyl stack survives before a
"dropped"/"flipped" failure. This isolates mount stiffness from driving skill
and terrain, exactly like the wrist-arm rock-test in HANDOFF.md (300 vs 11).

Use it to compare tray_drop (pendulum arm length) candidates BEFORE any GPU
fine-tune. Higher survival = a payload that a decent policy can actually keep up.

Usage:
    python tools/rock_test.py --drops 0.06 0.03 0.02 0.015 --trials 8 --horizon 2000
"""
import argparse
import numpy as np
import mujoco

import mirte_gimbal_env as M
from mirte_gimbal_env import MirteGimbalBalanceEnv


def rock_once(env, horizon, period, seed, waiter=False):
    """Drive an in-place shake; return (survived_steps, cause)."""
    env.reset(seed=seed)
    env.waiter_ff = waiter               # tilt tray into accel (coffee-cup trick)
    # symmetric square-wave on vx (net drift ~0) + alternating yaw whip.
    # trims (action[3:4]) held at 0 => NO policy help, pure mount + feedback.
    v_cmd = np.zeros(2)
    wz_cmd = 0.0
    for t in range(horizon):
        phase = (t // period) % 2
        vx_tgt = M.MAX_LIN_VEL if phase == 0 else -M.MAX_LIN_VEL
        wz_tgt = M.MAX_ANG_VEL if ((t // period) % 4) < 2 else -M.MAX_ANG_VEL
        # replicate step()'s slew limiter (so we never exceed what the robot
        # can physically command; the mount is what's under test, not the cap)
        dv = np.array([vx_tgt, 0.0]) - v_cmd
        dv_max = M.ACC_CMD_MAX * M.CTRL_DT
        n = np.linalg.norm(dv)
        if n > dv_max:
            dv *= dv_max / n
        v_cmd = v_cmd + dv
        dwz_max = M.YAW_ACC_CMD_MAX * M.CTRL_DT
        wz_cmd += float(np.clip(wz_tgt - wz_cmd, -dwz_max, dwz_max))
        # acc estimate for the waiter feedforward, as step() would compute it
        vxy = env._base_vel()[3:5]
        env._acc_f = np.clip(0.7 * env._acc_f + 0.3 * (vxy - env._prev_vxy) / M.CTRL_DT,
                             -M.ACC_CLIP, M.ACC_CLIP)
        env._prev_vxy = vxy.copy()
        for _ in range(M.FRAME_SKIP):
            env._apply_drive(v_cmd[0], v_cmd[1], wz_cmd)
            env._apply_gimbal(0.0, 0.0)          # feedback leveling only
            mujoco.mj_step(env.model, env.data)
        fail = env._failure()
        if fail in ("dropped", "flipped"):
            return t + 1, fail
    return horizon, "survived"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drops", type=float, nargs="+",
                    default=[0.06, 0.03, 0.02, 0.015])
    ap.add_argument("--walls", type=float, nargs="+", default=None,
                    help="sweep wall half-heights instead of drops (tray_drop fixed)")
    ap.add_argument("--fixed-drop", type=float, default=0.03,
                    help="tray_drop used when sweeping --walls")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=2000)
    ap.add_argument("--period", type=int, default=25,
                    help="control steps per half shake cycle (25 = 0.5s reversals)")
    ap.add_argument("--waiter", action="store_true",
                    help="enable waiter feedforward (tilt tray into acceleration)")
    ap.add_argument("--arm", choices=["on", "off", "both"], default="on",
                    help="arm-translation catch: on, off, or A/B both")
    args = ap.parse_args()

    print(f"rock-test: {args.trials} trials x {args.horizon} steps, "
          f"reversal every {args.period} steps ({args.period*M.CTRL_DT:.2f}s), "
          f"policy DISABLED (mount+feedback only)\n")
    sweep_walls = args.walls is not None
    label = "wall_hh" if sweep_walls else "tray_drop"
    values = args.walls if sweep_walls else args.drops
    arm_modes = {"on": [True], "off": [False], "both": [False, True]}[args.arm]
    print(f"{label:>10} | {'arm':>4} | {'survived%':>9} | {'mean steps':>10} | "
          f"{'median':>7} | {'min':>5}")
    print("-" * 62)
    for arm in arm_modes:
        for v in values:
            # standard start (flat area), no curriculum; only the mount changes
            kw = dict(start_curriculum=False, arm_catch=arm)
            if sweep_walls:
                env = MirteGimbalBalanceEnv(tray_drop=args.fixed_drop, wall_hh=v, **kw)
            else:
                env = MirteGimbalBalanceEnv(tray_drop=v, **kw)
            steps, survived = [], 0
            for k in range(args.trials):
                n, cause = rock_once(env, args.horizon, args.period, seed=7000 + k,
                                     waiter=args.waiter)
                steps.append(n)
                survived += (cause == "survived")
            env.close()
            steps = np.array(steps)
            print(f"{v:>10.3f} | {('on' if arm else 'off'):>4} | "
                  f"{100*survived/args.trials:>8.0f}% | "
                  f"{steps.mean():>10.0f} | {np.median(steps):>7.0f} | {steps.min():>5d}")
    print(f"\n(higher = more robust; shipped: tray_drop 0.06, wall_hh 0.03. "
          f"cylinder half-len 0.09, stack ~0.36 tall)")


if __name__ == "__main__":
    main()
