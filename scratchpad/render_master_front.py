"""Render ONE extra WIDE mastery angle: a front-quarter steep aerial (distinct
composition from the rear drone, both wide + occlusion-free) so the mastery run
can match-cut between two WIDE angles — never close (close sway reads as luck)."""
import os, subprocess
import numpy as np, mujoco, imageio_ffmpeg
from stable_baselines3 import PPO
from mirte_gimbal_env import MirteGimbalBalanceEnv
import record_progress as RP

W, H, FPS = 3840, 2160, 50
OUT = os.path.join(os.path.dirname(__file__), "cine2")
SNAP = "best_policy/mirte_best"
WHEELS = ("front_left_wheel_joint", "rear_left_wheel_joint",
          "front_right_wheel_joint", "rear_right_wheel_joint")
m = PPO.load(SNAP + ".zip", device="cpu"); nf = RP.load_norm(SNAP + "_vecnorm.pkl")


def run(seed):
    env = MirteGimbalBalanceEnv(randomize_on_reset=True); obs, _ = env.reset(seed=seed)
    traj, pos, yaws = [], [], []
    wadr = [env.model.jnt_qposadr[env.model.joint(j).id] for j in WHEELS]
    prev = np.array(env.data.body("base_link").xpos[:2]); wang = 0.0; oc = "X"; info = {}
    for t in range(2600):
        a = m.predict(nf(obs), deterministic=True)[0]
        obs, _, term, trunc, info = env.step(a)
        bx = np.array(env.data.body("base_link").xpos); _, _, yaw = env._base_rpy()
        wang += float(np.dot(bx[:2] - prev, [np.cos(yaw), np.sin(yaw)])) / 0.05
        prev = bx[:2].copy()
        q = env.data.qpos.copy()
        for adr in wadr:
            q[adr] = wang
        traj.append(q); pos.append(bx.copy()); yaws.append(yaw)
        if term or trunc:
            oc = "DELIVERED" if info.get("success") else "X"; break
    return env, traj, pos, yaws, oc


env = traj = pos = yaws = None
for s in range(0, 40):
    env, traj, pos, yaws, oc = run(s)
    if oc == "DELIVERED" and len(traj) >= 450:
        print(f"mastery seed {s} len {len(traj)}", flush=True); break
else:
    raise SystemExit("no mastery episode found")

env.model.vis.global_.offwidth = W; env.model.vis.global_.offheight = H
r = mujoco.Renderer(env.model, height=H, width=W)
cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(env.model, cam)
vopt = mujoco.MjvOption(); vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
idx = list(range(len(traj))) + [len(traj) - 1] * 45
sm = None
p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo",
    "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
    "-pix_fmt", "yuv420p", "-crf", "17", "-movflags", "+faststart", f"{OUT}/master_dronef.mp4"],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for k, i in enumerate(idx):
    env.data.qpos[:] = traj[i]; mujoco.mj_forward(env.model, env.data)
    look = pos[i] + [0, 0, 0.12]
    sm = np.array(look) if sm is None else 0.85 * sm + 0.15 * np.array(look)
    cam.lookat[:], cam.distance, cam.elevation = sm, 3.7, -48
    cam.azimuth = np.degrees(yaws[i]) + 25          # front-quarter steep aerial (wide)
    r.update_scene(env.data, cam, scene_option=vopt)
    p.stdin.write(np.ascontiguousarray(r.render(), np.uint8).tobytes())
p.stdin.close(); p.wait(); r.close()
print("wrote master_dronef.mp4", flush=True)
