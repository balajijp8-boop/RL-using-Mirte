#!/usr/bin/env python3
"""
Build the "watch RL learn" progression video for the MIRTE gimbal task.

Replays every training snapshot (snap_XXXXXk.zip + matching _vecnorm.pkl) on the
SAME fixed terrain/seed. Because we sample the policy stochastically, the arc is
authentic: an untrained net has huge action entropy so it lurches and topples the
tower in under a second; as PPO trains, entropy collapses and the same net drives
smoothly toward the goal. Each clip is overlaid with the training-step counter, a
training-progress bar, a distance-to-goal bar and a live outcome/tilt readout.

Memory-frugal by design (the machine has 14 GB RAM and no swap):
  * frames are STREAMED straight to ffmpeg, never accumulated in a Python list;
  * normalization stats are unpickled directly (no MuJoCo env built per snapshot);
  * a single env + renderer is reused across every snapshot.

Usage:
    MUJOCO_GL=egl python record_progress.py
    MUJOCO_GL=egl python record_progress.py --run runs/ppo_gimbal --out mirte_rl_progress.mp4
"""

import argparse
import glob
import os
import pickle
import re
import subprocess

import numpy as np
import torch
import mujoco
import matplotlib
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

from stable_baselines3 import PPO
from mirte_gimbal_env import MirteGimbalBalanceEnv

W, H = 960, 540                     # 16:9, crisp on LinkedIn
_FONTDIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
FONT = os.path.join(_FONTDIR, "DejaVuSans-Bold.ttf")     # portable: bundled with matplotlib
FONT_R = os.path.join(_FONTDIR, "DejaVuSans.ttf")
ROLL_SEED = 3                       # fixed scene for a fair before/after
NOISE_SEED = 0                      # fixed sampling noise -> reproducible clips
TOTAL_STEPS = 16_000_000

STATUS = {
    "running":   ("balancing…",       (150, 210, 255)),
    "DROPPED":   ("STACK DROPPED",    (255, 110, 110)),
    "CRASHED":   ("CRASHED",          (255, 110, 110)),
    "TIMEOUT":   ("ran out of time",  (240, 200, 120)),
    "DELIVERED": ("DELIVERED",        (110, 230, 140)),
}


def _font(path, size):
    return ImageFont.truetype(path, size)


def load_norm(vn_path):
    """Unpickle a saved VecNormalize and return a pure obs-normalizing fn.

    VecNormalize.__getstate__ drops the venv, so the pickle carries only the
    running-mean stats we need — no MuJoCo env has to be constructed."""
    if not vn_path or not os.path.exists(vn_path):
        return lambda o: o
    with open(vn_path, "rb") as f:
        vn = pickle.load(f)
    mean, var = vn.obs_rms.mean, vn.obs_rms.var
    eps, clip = vn.epsilon, vn.clip_obs
    return lambda o: np.clip((o - mean) / np.sqrt(var + eps), -clip, clip)


def rollout(model, normfn, step, env, renderer, cam, vopt, fonts, max_steps, sink):
    """Self-driven rollout on the fixed scene; frames streamed to `sink`."""
    torch.manual_seed(NOISE_SEED)
    np.random.seed(NOISE_SEED)
    obs, _ = env.reset(seed=ROLL_SEED)

    outcome, max_phi2, dist0 = "running", 0.0, None
    smooth = np.array(env.data.body("base_link").xpos)
    # cosmetic wheel spin: the wheels are massless/collision-off visual meshes
    # that nothing drives, so they'd sit frozen while the base glides. Roll their
    # joints by distance travelled so the render looks like the real MIRTE.
    wadr = [env.model.jnt_qposadr[env.model.joint(j).id] for j in
            ("front_left_wheel_joint", "rear_left_wheel_joint",
             "front_right_wheel_joint", "rear_right_wheel_joint")]
    _prevxy = np.array(env.data.body("base_link").xpos[:2])
    _wheel_ang = 0.0
    for t in range(max_steps):
        a, _ = model.predict(normfn(obs), deterministic=False)
        obs, rew, term, trunc, info = env.step(a)
        max_phi2 = max(max_phi2, info["phi2"])
        if dist0 is None:
            dist0 = info["dist"]
        bx = np.array(env.data.body("base_link").xpos)
        smooth = 0.85 * smooth + 0.15 * bx
        cam.lookat[:] = [smooth[0], smooth[1], 0.25]
        _, _, _yaw = env._base_rpy()
        _wheel_ang += float(np.dot(bx[:2] - _prevxy, [np.cos(_yaw), np.sin(_yaw)])) / 0.05
        _prevxy = bx[:2].copy()
        for _adr in wadr:
            env.data.qpos[_adr] = _wheel_ang
        mujoco.mj_kinematics(env.model, env.data)
        renderer.update_scene(env.data, cam, scene_option=vopt)
        raw = renderer.render()

        if term or trunc:
            if info.get("failure") == "dropped":
                outcome = "DROPPED"
            elif info.get("failure") == "collision":
                outcome = "CRASHED"
            elif info.get("success"):
                outcome = "DELIVERED"
            else:
                outcome = "TIMEOUT"
        sink(_overlay(raw, step, info, dist0, outcome, fonts))
        if term or trunc:
            flash = outcome in ("DROPPED", "CRASHED")
            win = outcome == "DELIVERED"
            frozen = _overlay(raw, step, info, dist0, outcome, fonts,
                              flash=flash, win=win)
            for _ in range(55):                       # ~1.1 s freeze on result
                sink(frozen)
            break
    else:
        outcome = "TIMEOUT"
    return outcome, np.degrees(max_phi2)


def _overlay(rgb, step, info, dist0, outcome, fonts, flash=False, win=False):
    f_title, f_big, f_lbl, f_small = fonts
    img = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")

    # top band + title
    d.rectangle([0, 0, W, 92], fill=(12, 14, 20, 210))
    d.text((26, 16), "RL · MIRTE Gimbal Transport", font=f_title,
           fill=(240, 244, 250))
    d.text((26, 56), "Carry the stacked cylinders to the goal without tipping",
           font=f_small, fill=(150, 200, 255))

    # training-step badge + training-progress bar (top right)
    step_txt = "RANDOM POLICY" if step == 0 else f"{step:,} steps"
    tw = d.textlength(step_txt, font=f_lbl)
    d.rectangle([W - tw - 42, 14, W - 18, 44], fill=(30, 34, 44, 230))
    d.text((W - tw - 30, 17), step_txt, font=f_lbl, fill=(255, 214, 120))
    frac = min(step / TOTAL_STEPS, 1.0)
    d.rectangle([W - 258, 56, W - 26, 68], fill=(60, 65, 78, 255))
    d.rectangle([W - 258, 56, W - 258 + int(232 * frac), 68],
                fill=(120, 200, 255, 255))

    # bottom status band
    d.rectangle([0, H - 78, W, H], fill=(12, 14, 20, 210))
    label, col = STATUS[outcome]
    d.text((26, H - 66), label, font=f_big, fill=col)
    phi2, tray = np.degrees(info["phi2"]), np.degrees(info["tray_tilt"])
    d.text((26, H - 24), f"top-cylinder tilt {phi2:4.1f}°     tray {tray:4.1f}°",
           font=f_small, fill=(200, 208, 220))

    if dist0 and dist0 > 1e-6:
        g = float(np.clip(1.0 - info["dist"] / dist0, 0, 1))
        d.text((W - 246, H - 66), "distance to goal", font=f_small,
               fill=(180, 188, 200))
        d.rectangle([W - 246, H - 40, W - 26, H - 26], fill=(60, 65, 78, 255))
        d.rectangle([W - 246, H - 40, W - 246 + int(220 * g), H - 26],
                    fill=(110, 230, 140, 255))

    if flash:
        d.rectangle([0, 0, W, H], fill=(200, 40, 40, 55))
    if win:
        d.rectangle([0, 0, W, H], fill=(60, 200, 90, 45))
    return np.asarray(img)


def title_card(text, sub, seconds, fps, sink):
    img = Image.new("RGB", (W, H), (10, 12, 18))
    d = ImageDraw.Draw(img)
    f, fs = _font(FONT, 52), _font(FONT_R, 26)
    tw = d.textlength(text, font=f)
    d.text(((W - tw) / 2, H / 2 - 60), text, font=f, fill=(240, 244, 250))
    sw = d.textlength(sub, font=fs)
    d.text(((W - sw) / 2, H / 2 + 16), sub, font=fs, fill=(150, 200, 255))
    frame = np.asarray(img)
    for _ in range(int(seconds * fps)):
        sink(frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/ppo_gimbal_v3")
    ap.add_argument("--out", default="mirte_rl_progress.mp4")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--total-steps", type=int, default=0,
                    help="run budget for the progress bar (0 = module default)")
    args = ap.parse_args()
    if args.total_steps:
        globals()["TOTAL_STEPS"] = args.total_steps

    snaps = sorted(glob.glob(os.path.join(args.run, "snap_*[0-9]k.zip")),
                   key=lambda p: int(re.search(r"snap_(\d+)k", p).group(1)))
    if not snaps:
        raise SystemExit(f"no snapshots in {args.run} yet")
    tags = [re.search(r"snap_(\d+)k", s).group(1) + "k" for s in snaps]
    print(f"found {len(snaps)} snapshots: {tags}")

    # ---- ffmpeg pipe: frames are streamed, never held in RAM ----
    proc = subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def sink(frame):
        proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    # ---- one env + one renderer, reused for every snapshot ----
    env = MirteGimbalBalanceEnv(randomize_on_reset=False, gimbal_enabled=True)
    env.reset(seed=ROLL_SEED)
    renderer = mujoco.Renderer(env.model, height=H, width=W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.distance, cam.elevation, cam.azimuth = 2.0, -16, 138
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    fonts = (_font(FONT, 30), _font(FONT, 46), _font(FONT_R, 22), _font(FONT_R, 19))

    title_card("Teaching a robot to balance",
               "Watch it learn, one policy at a time", 2.2, args.fps, sink)
    for zp in snaps:
        step = int(re.search(r"snap_(\d+)k", zp).group(1)) * 1000
        model = PPO.load(zp, device="cpu")
        normfn = load_norm(zp.replace(".zip", "_vecnorm.pkl"))
        outcome, mphi = rollout(model, normfn, step, env, renderer, cam, vopt,
                                fonts, args.max_steps, sink)
        print(f"  snap {step:>9,}: {outcome:9s} (peak tilt {mphi:.0f}°)")
        del model
    title_card("From flailing to fluent",
               "PPO · MuJoCo · real MIRTE Master", 2.4, args.fps, sink)

    proc.stdin.close()
    proc.wait()
    print(f"\nwrote {args.out}  ({W}x{H}@{args.fps})")


if __name__ == "__main__":
    main()
