"""Cinematic source render v2 — cameras do the work, no post FX.
Robot-relative cameras that ALWAYS track MIRTE (no wall-only shots):
 drone (high follow), chase (behind), fpp (over the gripper, looking forward),
 orbit (sweeps around), side (heading-relative). Plus a learning arc: find real
 episodes for flail -> pillar-drop -> crash -> stair-drop -> mastery, and a clean
 hero frame of MIRTE for the title card. Frames streamed to ffmpeg."""
import os, subprocess, sys
import numpy as np
import mujoco
import imageio_ffmpeg
from stable_baselines3 import PPO
from mirte_gimbal_env import MirteGimbalBalanceEnv
import record_progress as RP

W, H, FPS = 3840, 2160, 50
OUT = os.path.join(os.path.dirname(__file__), "cine2"); os.makedirs(OUT, exist_ok=True)
SNAP = "best_policy/mirte_best"
WHEELS = ("front_left_wheel_joint", "rear_left_wheel_joint",
          "front_right_wheel_joint", "rear_right_wheel_joint")
model_ppo = PPO.load(SNAP + ".zip", device="cpu")
normfn = RP.load_norm(SNAP + "_vecnorm.pkl")


def run_episode(seed, mode, max_steps):
    """mode: 'det' | 'stoch' | 'rand'. Returns env, traj(qpos), pos(3), yaw, outcome, failx."""
    env = MirteGimbalBalanceEnv(randomize_on_reset=True)
    obs, _ = env.reset(seed=seed)
    traj, pos, yaws = [], [], []
    wadr = [env.model.jnt_qposadr[env.model.joint(j).id] for j in WHEELS]
    prev = np.array(env.data.body("base_link").xpos[:2]); wang = 0.0
    outcome, info = "running", {}
    for t in range(max_steps):
        if mode == "rand":
            a = env.action_space.sample()
        else:
            a, _ = model_ppo.predict(normfn(obs), deterministic=(mode == "det"))
        obs, _, term, trunc, info = env.step(a)
        bx = np.array(env.data.body("base_link").xpos)
        _, _, yaw = env._base_rpy()
        wang += float(np.dot(bx[:2] - prev, [np.cos(yaw), np.sin(yaw)])) / 0.05
        prev = bx[:2].copy()
        q = env.data.qpos.copy()
        for adr in wadr:
            q[adr] = wang
        traj.append(q); pos.append(bx.copy()); yaws.append(yaw)
        if term or trunc:
            outcome = ("DELIVERED" if info.get("success") else
                       "DROPPED" if info.get("failure") == "dropped" else
                       "CRASHED" if info.get("failure") == "collision" else "TIMEOUT")
            break
    failx = float(env.data.body("base_link").xpos[0])
    return env, traj, pos, yaws, outcome, failx


# ---- robot-relative cameras: (lookat3, distance, elevation, azimuth_deg) ----
def cam_drone(i, n, p, yaw):  return (p + [0, 0, 0.12], 3.7, -52, np.degrees(yaw) + 205)
def cam_chase(i, n, p, yaw):  return (p + [0, 0, 0.22], 1.55, -9, np.degrees(yaw) + 180)
def cam_fpp(i, n, p, yaw):
    return (p + [0, 0, 0.30], 1.12, -5, np.degrees(yaw) + 180)  # tight low behind-the-gripper
def cam_orbit(i, n, p, yaw): return (p + [0, 0, 0.24], 2.25, -13, 120 + 150 * (i / max(1, n - 1)))
def cam_side(i, n, p, yaw):  return (p + [0, 0, 0.24], 1.9, -8, np.degrees(yaw) + 90)
CAMS = {"drone": cam_drone, "chase": cam_chase, "fpp": cam_fpp,
        "orbit": cam_orbit, "side": cam_side}


def render_clip(env, traj, pos, yaws, camname, path, rng=None, hold=0):
    if os.path.exists(path):
        print(f"  skip {os.path.basename(path)} (exists)", flush=True); return
    camfn = CAMS[camname]
    env.model.vis.global_.offwidth = W; env.model.vis.global_.offheight = H
    r = mujoco.Renderer(env.model, height=H, width=W)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(env.model, cam)
    vopt = mujoco.MjvOption(); vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    idx = list(range(len(traj))) if rng is None else list(range(*rng))
    idx = idx + [idx[-1]] * hold
    n = len(idx)
    sm = None
    proc = subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "17", "-movflags", "+faststart", path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for k, i in enumerate(idx):
        env.data.qpos[:] = traj[i]; mujoco.mj_forward(env.model, env.data)
        look, dist, elev, az = camfn(k, n, pos[i], yaws[i])
        sm = np.array(look) if sm is None else 0.85 * sm + 0.15 * np.array(look)
        cam.lookat[:], cam.distance, cam.elevation, cam.azimuth = sm, dist, elev, az
        r.update_scene(env.data, cam, scene_option=vopt)
        proc.stdin.write(np.ascontiguousarray(r.render(), np.uint8).tobytes())
    proc.stdin.close(); proc.wait(); r.close()
    print(f"  wrote {os.path.basename(path)} ({n} frames, {camname})", flush=True)


def hero_frame(env, traj, pos, yaws, i, path):
    """single clean beauty PNG of MIRTE for the title card."""
    from PIL import Image
    env.model.vis.global_.offwidth = W; env.model.vis.global_.offheight = H
    r = mujoco.Renderer(env.model, height=H, width=W)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(env.model, cam)
    vopt = mujoco.MjvOption(); vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    env.data.qpos[:] = traj[i]; mujoco.mj_forward(env.model, env.data)
    cam.lookat[:] = pos[i] + [0, 0, 0.18]
    cam.distance, cam.elevation, cam.azimuth = 1.7, -6, np.degrees(yaws[i]) + 210
    r.update_scene(env.data, cam, scene_option=vopt)
    Image.fromarray(r.render()).save(path); r.close()
    print(f"  wrote {os.path.basename(path)} (hero)", flush=True)


# ============ find the learning-arc episodes ============
def find(mode, want, xrange=None, seeds=range(0, 90), minlen=12, maxlen=2600):
    for s in seeds:
        env, traj, pos, yaws, oc, fx = run_episode(s, mode, maxlen)
        if oc == want and len(traj) >= minlen and (xrange is None or xrange[0] <= fx <= xrange[1]):
            print(f"[{want}] seed {s}: len {len(traj)} failx {fx:.2f}", flush=True)
            return s, env, traj, pos, yaws
    return None


print("=== finding episodes ===", flush=True)
mastery = find("det", "DELIVERED", seeds=range(0, 40), minlen=450)
flail = find("rand", "DROPPED", seeds=range(100, 200), minlen=10, maxlen=120)
pillar = find("det", "DROPPED", xrange=(-3.2, -0.55), seeds=range(0, 120))
crash = find("det", "CRASHED", seeds=range(0, 120))
stair = find("det", "DROPPED", xrange=(0.6, 1.8), seeds=range(0, 160))
if mastery is None:
    sys.exit("no mastery episode")

# ============ render ============
print("=== rendering ===", flush=True)
s, env, traj, pos, yaws = mastery
hero_frame(env, traj, pos, yaws, min(140, len(traj) - 1), f"{OUT}/hero.png")
for cam in ("drone", "chase", "fpp", "orbit", "side"):
    render_clip(env, traj, pos, yaws, cam, f"{OUT}/master_{cam}.mp4", hold=45)

for tag, ep, cams in [("flail", flail, ("chase", "drone")),
                      ("pillar", pillar, ("fpp", "chase")),
                      ("crash", crash, ("drone", "chase")),
                      ("stair", stair, ("chase", "fpp"))]:
    if ep is None:
        print(f"  (no {tag} episode found, skipping)", flush=True); continue
    _, e2, t2, p2, y2 = ep
    for cam in cams:
        render_clip(e2, t2, p2, y2, cam, f"{OUT}/{tag}_{cam}.mp4", hold=18)
print("done ->", OUT, flush=True)
