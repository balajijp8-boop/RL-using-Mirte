"""Re-render ONLY fail0_start and fail2_crash. Their low-angle 'behind' cameras
put the lens beyond a 1 m wall so the wall filled the frame (robot never visible).
Fix: elevated-but-close framing (elev ~-25, like the pillar clips that worked) that
reliably frames the robot even hard against a wall, + a shorter tumble tail so the
clip ends on the fallen stack, not empty floor."""
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


def episode(seed, mode, tail):
    env = MirteGimbalBalanceEnv(randomize_on_reset=True)
    obs, _ = env.reset(seed=seed)
    if mode == "rand":
        env.action_space.seed(seed)               # reproducible flail
    traj, pos, yaws = [], [], []
    wadr = [env.model.jnt_qposadr[env.model.joint(j).id] for j in WHEELS]
    prev = np.array(env.data.body("base_link").xpos[:2]); wang = 0.0
    oc, info, fx, fi = "running", {}, 0.0, 0

    def snap():
        nonlocal wang, prev
        bx = np.array(env.data.body("base_link").xpos); _, _, yaw = env._base_rpy()
        wang += float(np.dot(bx[:2] - prev, [np.cos(yaw), np.sin(yaw)])) / 0.05
        prev = bx[:2].copy()
        q = env.data.qpos.copy()
        for adr in wadr:
            q[adr] = wang
        traj.append(q); pos.append(bx.copy()); yaws.append(yaw)

    for t in range(2600):
        a = env.action_space.sample() if mode == "rand" else \
            m.predict(nf(obs), deterministic=True)[0]
        obs, _, term, trunc, info = env.step(a); snap()
        if term or trunc:
            oc = ("DELIVERED" if info.get("success") else
                  "DROPPED" if info.get("failure") == "dropped" else
                  "CRASHED" if info.get("failure") == "collision" else "TIMEOUT")
            fx, fi = float(env.data.body("base_link").xpos[0]), len(traj) - 1
            break
    if oc in ("DROPPED", "CRASHED"):
        env.data.xfrc_applied[:] = 0; env.data.ctrl[:] = 0
        for _ in range(tail):
            mujoco.mj_step(env.model, env.data); snap()
    return env, traj, pos, yaws, oc, fx, fi


def render(env, traj, pos, yaws, path, start, dist, elev, const_az, lift):
    """const_az: fixed world azimuth (deg). Robots are near-stationary at failure,
    so a constant azimuth keeps framing stable through the flail/tumble."""
    env.model.vis.global_.offwidth = W; env.model.vis.global_.offheight = H
    r = mujoco.Renderer(env.model, height=H, width=W)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(env.model, cam)
    vopt = mujoco.MjvOption(); vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    idx = list(range(max(0, start), len(traj))); sm = None
    proc = subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "17", "-movflags", "+faststart", path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for k, i in enumerate(idx):
        env.data.qpos[:] = traj[i]; mujoco.mj_forward(env.model, env.data)
        look = pos[i] + [0, 0, lift]
        sm = np.array(look) if sm is None else 0.86 * sm + 0.14 * np.array(look)
        cam.lookat[:], cam.distance, cam.elevation, cam.azimuth = sm, dist, elev, const_az
        r.update_scene(env.data, cam, scene_option=vopt)
        proc.stdin.write(np.ascontiguousarray(r.render(), np.uint8).tobytes())
    proc.stdin.close(); proc.wait(); r.close()
    print("  wrote", os.path.basename(path), len(idx), "frames", flush=True)


# fail2_crash: seed 19 det. az+270 (side) shows the robot + dividing wall + topple.
env, traj, pos, yaws, oc, fx, fi = episode(19, "det", 42)
print(f"[crash] seed 19 {oc} x={fx:.2f}", flush=True)
render(env, traj, pos, yaws, f"{OUT}/fail2_crash.mp4", max(0, fi - 35),
       2.2, -17, np.degrees(yaws[fi]) + 270, 0.16)

# fail0_start: rand flail in the back corner. az+225 frames the robot + toppling stack.
for s in range(200, 340):
    env, traj, pos, yaws, oc, fx, fi = episode(s, "rand", 42)
    if oc in ("DROPPED", "CRASHED") and fx < -2.9:
        print(f"[start] seed {s} {oc} x={fx:.2f}", flush=True)
        render(env, traj, pos, yaws, f"{OUT}/fail0_start.mp4", max(0, fi - 35),
               2.3, -18, np.degrees(yaws[fi]) + 225, 0.16)
        break
print("done", flush=True)
