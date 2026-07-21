"""Render the 6 failure clips for the learning arc — each shot CLOSE so the fall
is clearly readable (drop / wall bump / pillar hit / stair topple), and each label
is only what is TRUTHFULLY on screen:
  fail0_start     start flail (random policy)     "no idea how to balance"
  failX_pillarhit CRASHED in the pillar field     "it hits a pillar"
  fail1_pillars   DROPPED in the pillar field     "it drops the load"
  fail2_crash     CRASHED at the dividing wall    "it hits a wall"
  fail3_doorway   DROPPED near the doorway        "the stack tips over"
  fail4_stairs    fall GENUINELY ON the steps     "it topples on the stairs"
Physics continues ~55 frames past detection so the cylinders really tumble.
Pillar-field clips sit a touch higher/back so the 1 m pillars don't occlude;
open-area clips (wall/doorway/stairs) are low + close for drama."""
import os, glob, subprocess
import numpy as np
import mujoco
import imageio_ffmpeg
from stable_baselines3 import PPO
from mirte_gimbal_env import MirteGimbalBalanceEnv
import record_progress as RP

W, H, FPS = 3840, 2160, 50
OUT = os.path.join(os.path.dirname(__file__), "cine2")
SNAP = "best_policy/mirte_best"
WHEELS = ("front_left_wheel_joint", "rear_left_wheel_joint",
          "front_right_wheel_joint", "rear_right_wheel_joint")
model_ppo = PPO.load(SNAP + ".zip", device="cpu")
normfn = RP.load_norm(SNAP + "_vecnorm.pkl")

# (dist, elev, az_offset, lookat_lift) per category. Pillar clips: higher+back to
# clear the 1 m pillars. Open clips: low + close so the topple fills the frame.
CAM = {
    "start":     (2.0, -10,  25, 0.20),
    "pillarhit": (2.5, -24, 205, 0.16),
    "pillars":   (2.5, -24, 205, 0.16),
    "crash":     (2.0,  -9, 205, 0.20),
    "doorway":   (2.0, -10, 205, 0.20),
    "stairs":    (2.2,  -9, 205, 0.18),
}
NAME = {  # category -> output filename (matches the film's fails list)
    "start": "fail0_start", "pillarhit": "failX_pillarhit", "pillars": "fail1_pillars",
    "crash": "fail2_crash", "doorway": "fail3_doorway", "stairs": "fail4_stairs",
}


def run_episode(seed, mode, max_steps=2600, tail=55):
    env = MirteGimbalBalanceEnv(randomize_on_reset=True)
    obs, _ = env.reset(seed=seed)
    traj, pos, yaws = [], [], []
    wadr = [env.model.jnt_qposadr[env.model.joint(j).id] for j in WHEELS]
    prev = np.array(env.data.body("base_link").xpos[:2]); wang = 0.0
    outcome, info, failx, faili = "running", {}, 0.0, 0

    def snap():
        bx = np.array(env.data.body("base_link").xpos)
        nonlocal wang, prev
        _, _, yaw = env._base_rpy()
        wang += float(np.dot(bx[:2] - prev, [np.cos(yaw), np.sin(yaw)])) / 0.05
        prev = bx[:2].copy()
        q = env.data.qpos.copy()
        for adr in wadr:
            q[adr] = wang
        traj.append(q); pos.append(bx.copy()); yaws.append(yaw)

    for t in range(max_steps):
        a = env.action_space.sample() if mode == "rand" else \
            model_ppo.predict(normfn(obs), deterministic=(mode == "det"))[0]
        obs, _, term, trunc, info = env.step(a)
        snap()
        if term or trunc:
            outcome = ("DELIVERED" if info.get("success") else
                       "DROPPED" if info.get("failure") == "dropped" else
                       "CRASHED" if info.get("failure") == "collision" else "TIMEOUT")
            failx, faili = float(env.data.body("base_link").xpos[0]), len(traj) - 1
            break
    if outcome in ("DROPPED", "CRASHED"):
        env.data.xfrc_applied[:] = 0
        env.data.ctrl[:] = 0
        for _ in range(tail):
            mujoco.mj_step(env.model, env.data)
            snap()
    return env, traj, pos, yaws, outcome, failx, faili


def render(env, traj, pos, yaws, cat, path, start):
    dist, elev, azoff, lift = CAM[cat]
    env.model.vis.global_.offwidth = W; env.model.vis.global_.offheight = H
    r = mujoco.Renderer(env.model, height=H, width=W)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(env.model, cam)
    vopt = mujoco.MjvOption(); vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    idx = list(range(max(0, start), len(traj)))
    sm = None
    proc = subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "17", "-movflags", "+faststart", path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for k, i in enumerate(idx):
        env.data.qpos[:] = traj[i]; mujoco.mj_forward(env.model, env.data)
        look = pos[i] + [0, 0, lift]
        sm = np.array(look) if sm is None else 0.86 * sm + 0.14 * np.array(look)
        cam.lookat[:], cam.distance, cam.elevation = sm, dist, elev
        cam.azimuth = np.degrees(yaws[i]) + azoff
        r.update_scene(env.data, cam, scene_option=vopt)
        proc.stdin.write(np.ascontiguousarray(r.render(), np.uint8).tobytes())
    proc.stdin.close(); proc.wait(); r.close()
    print(f"  wrote {os.path.basename(path)} ({len(idx)} frames)", flush=True)


def on_stairs(env, x):
    """True only if the fall is on the actual step footprint, past the first riser
    (so the robot is genuinely up on the stairs, not at the approach)."""
    x0 = env._stair_band[0] + 0.25          # first riser
    xc = env._stair_band[1] - 0.25          # top of the down-step
    return (x0 + 0.30) <= x <= (xc - 0.05)


def categorize(env, oc, x):
    if oc == "CRASHED" and -2.6 <= x <= -0.72:  return "pillarhit"
    if oc == "DROPPED" and -2.6 <= x <= -0.72:  return "pillars"
    if oc in ("DROPPED", "CRASHED") and on_stairs(env, x): return "stairs"
    if oc == "CRASHED" and -0.55 <= x <= 0.20:  return "crash"
    if oc == "DROPPED" and -0.55 <= x <= 0.60:  return "doorway"
    return None


# wipe old clips so they actually re-render with the new close cameras
for cat, nm in NAME.items():
    p = f"{OUT}/{nm}.mp4"
    if os.path.exists(p): os.remove(p)

got = {}

# start flail: random policy drops the stack immediately near the back wall
print("=== start flail (random policy) ===", flush=True)
for s in range(200, 320):
    env, traj, pos, yaws, oc, fx, fi = run_episode(s, "rand")
    if oc in ("DROPPED", "CRASHED") and fx < -2.9:
        got["start"] = (env, traj, pos, yaws, fi); print(f"[start] seed {s} {oc} x={fx:.2f}", flush=True); break

# policy failures for the other five categories
need = ["pillarhit", "pillars", "crash", "doorway", "stairs"]
print("=== policy failures ===", flush=True)
for s in range(0, 900):
    if all(k in got for k in need):
        break
    env, traj, pos, yaws, oc, fx, fi = run_episode(s, "det")
    if oc not in ("DROPPED", "CRASHED"):
        continue
    cat = categorize(env, oc, fx)
    if cat is None or cat in got:
        continue
    got[cat] = (env, traj, pos, yaws, fi)
    print(f"[{cat}] seed {s} {oc} x={fx:.2f} band={env._stair_band}", flush=True)

print("=== rendering (close) ===", flush=True)
for cat in ["start", "pillarhit", "pillars", "crash", "doorway", "stairs"]:
    if cat not in got:
        print("  MISSING", cat, flush=True); continue
    env, traj, pos, yaws, fi = got[cat]
    render(env, traj, pos, yaws, cat, f"{OUT}/{NAME[cat]}.mp4", max(0, fi - 35))
print("HAVE:", sorted(got.keys()), flush=True)
print("done", flush=True)
