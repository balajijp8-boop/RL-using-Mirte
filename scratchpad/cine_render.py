"""Render cinematic source clips for the 'v1 -> delivery' film.
Runs ONE winning episode of the best policy, stores the full trajectory, then
re-renders that identical run from several camera angles (so cuts between angles
are true match-cuts). Also renders short 'flailing' clips (high-entropy actions =
what an untrained net does) for the origin-story opening. Wheels rolled by
distance (the project's cosmetic fix). Frames streamed to ffmpeg per clip."""
import os, subprocess, sys
import numpy as np
import mujoco
import imageio_ffmpeg
from stable_baselines3 import PPO
from mirte_gimbal_env import MirteGimbalBalanceEnv
import record_progress as RP

W, H, FPS = 3840, 2160, 50   # 4K
OUT = os.path.join(os.path.dirname(__file__), "cine")
os.makedirs(OUT, exist_ok=True)
SNAP = "best_policy/mirte_best"
WHEELS = ("front_left_wheel_joint", "rear_left_wheel_joint",
          "front_right_wheel_joint", "rear_right_wheel_joint")

model_ppo = PPO.load(SNAP + ".zip", device="cpu")
normfn = RP.load_norm(SNAP + "_vecnorm.pkl")

# camera presets: (name, distance, elevation, azimuth, z_lookat)
CAMS = [
    ("chase",  1.7, -7,  138, 0.22),   # low hero chase from behind
    ("side",   2.3, -9,   90, 0.28),   # tracking side profile
    ("high",   3.0, -34, 150, 0.30),   # elevated wide
    ("front",  1.5, -8,  210, 0.26),   # facing, close on the stack
]


def run_episode(seed, deterministic, max_steps, random_act=False):
    """Run one episode; return (env, list of qpos copies, base_xy list, outcome).
    random_act=True uses uniform random actions = an untrained high-entropy net."""
    env = MirteGimbalBalanceEnv(randomize_on_reset=True)
    obs, _ = env.reset(seed=seed)
    traj, basexy = [], []
    wadr = [env.model.jnt_qposadr[env.model.joint(j).id] for j in WHEELS]
    prev = np.array(env.data.body("base_link").xpos[:2]); wang = 0.0
    outcome = "running"
    for t in range(max_steps):
        if random_act:
            a = env.action_space.sample()
        else:
            a, _ = model_ppo.predict(normfn(obs), deterministic=deterministic)
        obs, _, term, trunc, info = env.step(a)
        bx = np.array(env.data.body("base_link").xpos)
        _, _, yaw = env._base_rpy()
        wang += float(np.dot(bx[:2] - prev, [np.cos(yaw), np.sin(yaw)])) / 0.05
        prev = bx[:2].copy()
        q = env.data.qpos.copy()
        for adr in wadr:
            q[adr] = wang
        traj.append(q); basexy.append(bx[:2].copy())
        if term or trunc:
            outcome = ("DELIVERED" if info.get("success") else
                       "DROPPED" if info.get("failure") == "dropped" else
                       "CRASHED" if info.get("failure") == "collision" else "TIMEOUT")
            break
    return env, traj, basexy, outcome


def render_clip(env, traj, basexy, cam_preset, path, tail_hold=0):
    name, dist, elev, azim, zl = cam_preset
    env.model.vis.global_.offwidth = W      # allow 4K offscreen buffer
    env.model.vis.global_.offheight = H
    r = mujoco.Renderer(env.model, height=H, width=W)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.distance, cam.elevation, cam.azimuth = dist, elev, azim
    vopt = mujoco.MjvOption(); vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    smooth = np.array([*basexy[0], zl])
    proc = subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = traj + [traj[-1]] * tail_hold
    for i, q in enumerate(frames):
        env.data.qpos[:] = q
        mujoco.mj_forward(env.model, env.data)
        j = min(i, len(basexy) - 1)
        smooth = 0.88 * smooth + 0.12 * np.array([*basexy[j], zl])
        cam.lookat[:] = smooth
        r.update_scene(env.data, cam, scene_option=vopt)
        proc.stdin.write(np.ascontiguousarray(r.render(), np.uint8).tobytes())
    proc.stdin.close(); proc.wait(); r.close()
    print(f"  wrote {os.path.basename(path)} ({len(frames)} frames)", flush=True)


# 1) find a clean winning delivery seed (deterministic)
win = None
for s in range(40):
    env, traj, basexy, outcome = run_episode(s, True, 2600)
    print(f"seed {s}: {outcome} ({len(traj)} steps)", flush=True)
    if outcome == "DELIVERED" and len(traj) > 450:
        win = (s, env, traj, basexy); break
if win is None:
    sys.exit("no clean delivery found")
s, env, traj, basexy = win
print(f"DELIVERY seed {s}, {len(traj)} steps -> rendering {len(CAMS)} angles", flush=True)
for cam in CAMS:
    render_clip(env, traj, basexy, cam, os.path.join(OUT, f"deliver_{cam[0]}.mp4"), tail_hold=40)

# 2) flailing origin clips: high-entropy (stochastic) actions topple fast
flail_cams = [CAMS[3], CAMS[1]]   # front, side
n = 0
for s in range(100, 140):
    env2, traj2, basexy2, outcome2 = run_episode(s, False, 130, random_act=True)  # untrained flail
    if outcome2 in ("DROPPED", "CRASHED") and 25 < len(traj2) < 130:
        render_clip(env2, traj2, basexy2, flail_cams[n % 2],
                    os.path.join(OUT, f"flail_{n}.mp4"))
        n += 1
        if n >= 2:
            break
print("done ->", OUT, flush=True)
