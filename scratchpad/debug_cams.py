"""Render the failure MOMENT of fail0 (seed 200 rand) and fail2 (seed 19 det)
from a sweep of azimuths, so we can pick angles where the robot + wall bump are
actually visible (the low 'behind' shots put the lens outside the room / behind
the dividing wall)."""
import os, subprocess
import numpy as np, mujoco, imageio_ffmpeg
from PIL import Image
from stable_baselines3 import PPO
from mirte_gimbal_env import MirteGimbalBalanceEnv
import record_progress as RP

W, H = 1280, 720
OUT = os.path.join(os.path.dirname(__file__), "cine2", "v4")
SNAP = "best_policy/mirte_best"
WHEELS = ("front_left_wheel_joint", "rear_left_wheel_joint",
          "front_right_wheel_joint", "rear_right_wheel_joint")
m = PPO.load(SNAP + ".zip", device="cpu"); nf = RP.load_norm(SNAP + "_vecnorm.pkl")


def episode(seed, mode, extra=25):
    env = MirteGimbalBalanceEnv(randomize_on_reset=True)
    obs, _ = env.reset(seed=seed)
    if mode == "rand":
        env.action_space.seed(seed)
    wadr = [env.model.jnt_qposadr[env.model.joint(j).id] for j in WHEELS]
    prev = np.array(env.data.body("base_link").xpos[:2]); wang = 0.0; fi = 0
    traj, pos, yaws = [], [], []

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
        a = env.action_space.sample() if mode == "rand" else m.predict(nf(obs), deterministic=True)[0]
        obs, _, term, trunc, info = env.step(a); snap()
        if term or trunc:
            fi = len(traj) - 1; break
    env.data.xfrc_applied[:] = 0; env.data.ctrl[:] = 0
    for _ in range(extra):
        mujoco.mj_step(env.model, env.data); snap()
    fr = min(fi + 15, len(traj) - 1)          # mid-tumble frame
    return env, traj[fr], pos[fr], yaws[fr]


def sweep(env, q, p, yaw, tag, dist=2.6, elev=-20):
    env.model.vis.global_.offwidth = W; env.model.vis.global_.offheight = H
    r = mujoco.Renderer(env.model, height=H, width=W)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(env.model, cam)
    vopt = mujoco.MjvOption(); vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    env.data.qpos[:] = q; mujoco.mj_forward(env.model, env.data)
    azoffs = [0, 45, 90, 135, 180, 225, 270, 315]
    tiles = []
    for azo in azoffs:
        cam.lookat[:] = p + [0, 0, 0.16]; cam.distance = dist; cam.elevation = elev
        cam.azimuth = np.degrees(yaw) + azo
        r.update_scene(env.data, cam, scene_option=vopt)
        tiles.append((azo, Image.fromarray(np.ascontiguousarray(r.render(), np.uint8))))
    r.close()
    from PIL import ImageDraw
    cw, ch = W // 2, H // 2
    sheet = Image.new("RGB", (cw * 4, ch * 2 + 60), (12, 14, 16)); d = ImageDraw.Draw(sheet)
    for i, (azo, im) in enumerate(tiles):
        im = im.resize((cw, ch)); x = (i % 4) * cw; y = (i // 4) * (ch + 30)
        sheet.paste(im, (x, y + 30)); d.text((x + 8, y + 8), f"az+{azo}", fill=(255, 220, 120))
    sheet.save(f"{OUT}/sweep_{tag}.png"); print("wrote", f"{OUT}/sweep_{tag}.png", flush=True)


env, q, p, yaw = episode(19, "det"); print("crash pos", np.round(p, 2), flush=True)
sweep(env, q, p, yaw, "crash")
env, q, p, yaw = episode(200, "rand"); print("start pos", np.round(p, 2), flush=True)
sweep(env, q, p, yaw, "start")
print("done", flush=True)
