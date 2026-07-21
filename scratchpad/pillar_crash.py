"""Search for a genuine collision IN the pillar field (robot bumping a pillar),
distinct from the doorway-wall crash. Self-contained. Renders with tumble tail."""
import os, subprocess
import numpy as np, mujoco, imageio_ffmpeg
from stable_baselines3 import PPO
from mirte_gimbal_env import MirteGimbalBalanceEnv
import record_progress as RP

W, H, FPS = 3840, 2160, 50
OUT = os.path.join(os.path.dirname(__file__), "cine2")
WHEELS = ("front_left_wheel_joint","rear_left_wheel_joint","front_right_wheel_joint","rear_right_wheel_joint")
m = PPO.load("best_policy/mirte_best.zip", device="cpu"); nf = RP.load_norm("best_policy/mirte_best_vecnorm.pkl")

def run(seed, tail=55):
    env = MirteGimbalBalanceEnv(randomize_on_reset=True); obs,_ = env.reset(seed=seed)
    traj,pos,yaws=[],[],[]; wadr=[env.model.jnt_qposadr[env.model.joint(j).id] for j in WHEELS]
    prev=np.array(env.data.body("base_link").xpos[:2]); wang=0.0; oc="running"; info={}; fi=0
    def snap():
        nonlocal wang,prev
        bx=np.array(env.data.body("base_link").xpos); _,_,yaw=env._base_rpy()
        wang+=float(np.dot(bx[:2]-prev,[np.cos(yaw),np.sin(yaw)]))/0.05; prev=bx[:2].copy()
        q=env.data.qpos.copy()
        for a in wadr: q[a]=wang
        traj.append(q); pos.append(bx.copy()); yaws.append(yaw)
    for t in range(2600):
        a=m.predict(nf(obs),deterministic=True)[0]; obs,_,term,trunc,info=env.step(a); snap()
        if term or trunc:
            oc="DELIVERED" if info.get("success") else "DROPPED" if info.get("failure")=="dropped" else "CRASHED" if info.get("failure")=="collision" else "TIMEOUT"
            fx=float(env.data.body("base_link").xpos[0]); fi=len(traj)-1; break
    if oc in ("DROPPED","CRASHED"):
        env.data.xfrc_applied[:]=0; env.data.ctrl[:]=0
        for _ in range(tail): mujoco.mj_step(env.model,env.data); snap()
    return env,traj,pos,yaws,oc,fx,fi

def render(env,traj,pos,yaws,path,start):
    env.model.vis.global_.offwidth=W; env.model.vis.global_.offheight=H
    r=mujoco.Renderer(env.model,height=H,width=W); cam=mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(env.model,cam)
    vopt=mujoco.MjvOption(); vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER]=False
    idx=list(range(max(0,start),len(traj))); sm=None
    p=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-","-c:v","libx264","-pix_fmt","yuv420p","-crf","17",path],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    for k,i in enumerate(idx):
        env.data.qpos[:]=traj[i]; mujoco.mj_forward(env.model,env.data)
        look=pos[i]+[0,0,0.12]; sm=np.array(look) if sm is None else 0.86*sm+0.14*np.array(look)
        cam.lookat[:],cam.distance,cam.elevation,cam.azimuth = sm,3.9,-52,np.degrees(yaws[i])+205  # drone (no pillar occlusion)
        r.update_scene(env.data,cam,scene_option=vopt); p.stdin.write(np.ascontiguousarray(r.render(),np.uint8).tobytes())
    p.stdin.close(); p.wait(); r.close(); print("  wrote",os.path.basename(path),len(idx),"frames",flush=True)

print("searching pillar-field collisions (x in -2.6..-0.72)...",flush=True)
hit=None
for s in range(0,400):
    env,traj,pos,yaws,oc,fx,fi=run(s)
    if oc=="CRASHED" and -2.6<=fx<=-0.72:
        print(f"[pillar-crash] seed {s} x={fx:.2f}",flush=True); hit=(env,traj,pos,yaws,fi); break
if hit:
    render(hit[0],hit[1],hit[2],hit[3],f"{OUT}/failX_pillarhit.mp4",max(0,hit[4]-35))
else:
    print("NO pillar-field collision in 0..399 — policy weaves pillars; its crashes are at the doorway/wall.",flush=True)
print("done",flush=True)
