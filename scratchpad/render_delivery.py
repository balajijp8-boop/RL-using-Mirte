"""Render a clean full-course DELIVERY video of the honest best policy
(v23/snap_01000k, 48/60). Reuses record_progress's overlay/wheel-spin/ffmpeg
pipeline, but runs the policy DETERMINISTICALLY and searches for a seed where it
delivers, so the clip shows the policy's real competent behavior end to end."""
import subprocess
import numpy as np
import mujoco
import imageio_ffmpeg
from stable_baselines3 import PPO
from mirte_gimbal_env import MirteGimbalBalanceEnv
import record_progress as RP

W, H = RP.W, RP.H
SNAP = "runs/ppo_gimbal_v23/snap_01000k"
OUT = "mirte_delivery_v23_best.mp4"
MAX_STEPS, FPS, STEP_LABEL = 2600, 50, 8_000_000
RP.TOTAL_STEPS = STEP_LABEL

model = PPO.load(SNAP + ".zip", device="cpu")
normfn = RP.load_norm(SNAP + "_vecnorm.pkl")
env = MirteGimbalBalanceEnv(randomize_on_reset=True, gimbal_enabled=True)
fonts = (RP._font(RP.FONT, 30), RP._font(RP.FONT, 46),
         RP._font(RP.FONT_R, 22), RP._font(RP.FONT_R, 19))
WHEELS = ("front_left_wheel_joint", "rear_left_wheel_joint",
          "front_right_wheel_joint", "rear_right_wheel_joint")


def episode(seed, sink=None):
    obs, _ = env.reset(seed=seed)
    render = sink is not None
    if render:
        r = mujoco.Renderer(env.model, height=H, width=W)
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(env.model, cam)
        cam.distance, cam.elevation, cam.azimuth = 2.0, -16, 138
        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        smooth = np.array(env.data.body("base_link").xpos)
        wadr = [env.model.jnt_qposadr[env.model.joint(j).id] for j in WHEELS]
        prevxy = np.array(env.data.body("base_link").xpos[:2]); wang = 0.0
    outcome, dist0 = "running", None
    for _ in range(MAX_STEPS):
        a, _ = model.predict(normfn(obs), deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        if dist0 is None:
            dist0 = info["dist"]
        if render:
            bx = np.array(env.data.body("base_link").xpos)
            smooth = 0.85 * smooth + 0.15 * bx
            cam.lookat[:] = [smooth[0], smooth[1], 0.25]
            _, _, yaw = env._base_rpy()
            wang += float(np.dot(bx[:2] - prevxy, [np.cos(yaw), np.sin(yaw)])) / 0.05
            prevxy = bx[:2].copy()
            for adr in wadr:
                env.data.qpos[adr] = wang
            mujoco.mj_kinematics(env.model, env.data)
            r.update_scene(env.data, cam, scene_option=vopt)
            raw = r.render()
        if term or trunc:
            outcome = ("DELIVERED" if info.get("success") else
                       "DROPPED" if info.get("failure") == "dropped" else
                       "CRASHED" if info.get("failure") == "collision" else "TIMEOUT")
        if render:
            sink(RP._overlay(raw, STEP_LABEL, info, dist0, outcome, fonts))
            if term or trunc:
                frozen = RP._overlay(raw, STEP_LABEL, info, dist0, outcome, fonts,
                                     win=(outcome == "DELIVERED"),
                                     flash=(outcome in ("DROPPED", "CRASHED")))
                for _ in range(60):
                    sink(frozen)
        if term or trunc:
            break
    if render:
        r.close()
    return outcome


good = None
for s in range(25):
    o = episode(s)
    print(f"  seed {s:2d}: {o}")
    if o == "DELIVERED":
        good = s
        break
if good is None:
    raise SystemExit("no delivering seed in 0..24")
print(f"rendering delivery on seed {good} -> {OUT}")

proc = subprocess.Popen(
    [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
     "-movflags", "+faststart", OUT],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sink(f):
    proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())


RP.title_card("MIRTE Gimbal Transport",
              "Best policy - carry the stacked cylinders to the goal", 2.0, FPS, sink)
final = episode(good, sink=sink)
proc.stdin.close(); proc.wait()
print(f"wrote {OUT}  (outcome {final})")
