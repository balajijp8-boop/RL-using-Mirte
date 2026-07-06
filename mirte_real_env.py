"""
MirteRealBalanceEnv
===================

Same task/MDP as MirteStackedBalanceEnv, but the robot is the REAL MIRTE Master
(URDF-converted meshes) instead of a box. The base is driven by abstract planar
joints (slide_x / slide_y / yaw); the mecanum wheels are visual only; the real
4-DOF arm is locked rigid in a horizontal "tray" pose, and the two stacked
cylinders sit in a bladed cradle on the base at the gripper location.

Observation (19), action (3) and reward are identical to the box env, so the
existing train_ppo.py / watch.py pipeline works unchanged.
"""

from __future__ import annotations
import os
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import lxml.etree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "mirte_model")
ROBOT_XML = os.path.join(MODEL_DIR, "mirte_master.xml")

# ---- world / task constants (match the box env) -----------------------------
ROOM_HALF_X, ROOM_HALF_Y = 4.0, 2.5
WALL_HH, WALL_HT = 0.5, 0.05
DOOR_W_MIN, DOOR_W_MAX = 0.75, 1.20
N_PILLARS_MIN, N_PILLARS_MAX = 2, 4
PILLAR_R = 0.12
START_X, TARGET_X, TARGET_RADIUS = -3.3, 3.2, 0.35

MAX_LIN_VEL, MAX_ANG_VEL = 0.8, 1.2
LIDAR_CUTOFF, POS_SCALE = 5.0, 8.0
PHYSICS_DT, FRAME_SKIP = 0.002, 10
CTRL_DT = PHYSICS_DT * FRAME_SKIP
TILT_LIMIT = 0.5

W_PROGRESS, W_TILT_BOT, W_TILT_TOP = 5.0, 1.0, 3.0
W_JERK, TIME_PENALTY = 0.05, 0.01
P_COLLISION, P_FALL, R_SUCCESS = 15.0, 15.0, 30.0

LIDAR_ANGLES = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])

# arm tray pose + tray geometry (validated in build_tray_scene.py)
ARM_POSE = {"shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.0,
            "elbow_joint": -0.5, "wrist_joint": 0.0, "gripper_joint": 0.2}
GX, GY, GZ = 0.387, 0.012, 0.326 - 0.055     # tray center in base_link frame
CYL_R, CYL_HH = 0.03, 0.09
ARM_BODIES = ("shoulder_pan", "shoulder_lift", "elbow", "wrist", "gripper",
              "_gripper_link_r", "gripper_finger_r", "_gripper_link_l",
              "gripper_finger_l", "_Gripper_r")
WHEELS = ("front_left_wheel", "rear_left_wheel", "front_right_wheel", "rear_right_wheel")


class MirteRealBalanceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": int(1 / CTRL_DT)}

    def __init__(self, render_mode=None, max_episode_steps=1500, randomize_on_reset=True):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.randomize_on_reset = randomize_on_reset

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(19,), dtype=np.float32)

        self.model = self.data = self._viewer = None
        self._init_qpos = self._init_qvel = None
        self._target = np.zeros(2)
        self._prev_dist = 0.0
        self._prev_action = np.zeros(3)
        self._prev_phi = np.zeros(2)
        self._steps = 0
        self._start_y = 0.0

        # cache the robot asset block + base_link subtree, pre-modified into a
        # tray robot (planar joints, visual wheels, locked no-collision arm,
        # tray+blades, lidar sites)
        self._asset_xml, self._robot_body = self._prepare_robot()

    # ------------------------------------------------------------------ robot
    def _prepare_robot(self):
        tree = ET.parse(ROBOT_XML)
        root = tree.getroot()
        asset = root.find("asset")
        base = root.find("worldbody").find(".//body[@name='base_link']")

        # planar joints
        for i, (nm, ax) in enumerate([("slide_x", "1 0 0"), ("slide_y", "0 1 0")]):
            base.insert(i, ET.Element("joint", name=nm, type="slide", axis=ax, damping="2"))
        base.insert(2, ET.Element("joint", name="yaw", type="hinge", axis="0 0 1", damping="1"))
        base.set("pos", "0 0 0.055")

        # wheels visual only
        for wn in WHEELS:
            for g in base.find(f".//body[@name='{wn}']").findall("geom"):
                g.set("contype", "0"); g.set("conaffinity", "0")

        # arm: collision-free + rigidly sprung to the tray pose
        for an in ARM_BODIES:
            for g in base.find(f".//body[@name='{an}']").findall("geom"):
                g.set("contype", "0"); g.set("conaffinity", "0")
        for jn, q in ARM_POSE.items():
            jel = base.find(f".//joint[@name='{jn}']")
            jel.set("stiffness", "1500"); jel.set("springref", f"{q}"); jel.set("damping", "30")
            jel.attrib.pop("actuatorfrcrange", None)

        # tray plate + two blades (proven cradle) on base_link
        ET.SubElement(base, "geom", name="tray", type="box", size="0.055 0.055 0.004",
                      pos=f"{GX} {GY} {GZ}", rgba="0.2 0.2 0.25 1", friction="1.4 0.02 0.002")
        for nm, dy in (("blade_l", +0.048), ("blade_r", -0.048)):
            ET.SubElement(base, "geom", name=nm, type="box", size="0.06 0.006 0.03",
                          pos=f"{GX} {GY+dy} {GZ+0.03}", rgba="0.25 0.25 0.3 1",
                          friction="1.4 0.02 0.002")

        # 8 lidar sites on frame_link (ray = site +z)
        frame = base.find(".//body[@name='frame_link']")
        for i, a in enumerate(LIDAR_ANGLES):
            c, s = np.cos(a), np.sin(a)
            ET.SubElement(frame, "site", name=f"rf{i}",
                          pos=f"{0.20*c:.4f} {0.20*s:.4f} 0.05",
                          zaxis=f"{c:.5f} {s:.5f} 0", size="0.005")

        asset_xml = ET.tostring(asset).decode()
        robot_xml = ET.tostring(base).decode()
        return asset_xml, robot_xml

    # ------------------------------------------------------------------ scene
    def _build_xml(self, rng):
        door_w = rng.uniform(DOOR_W_MIN, DOOR_W_MAX)
        door_cy = rng.uniform(-1.4, 1.4)
        gap_lo = max(door_cy - door_w / 2, -ROOM_HALF_Y + 0.1)
        gap_hi = min(door_cy + door_w / 2, ROOM_HALF_Y - 0.1)
        self._start_y = rng.uniform(-1.8, 1.8)
        target_y = rng.uniform(-1.8, 1.8)
        self._target = np.array([TARGET_X, target_y])

        pillars, tries = [], 0
        n_pillars = rng.integers(N_PILLARS_MIN, N_PILLARS_MAX + 1)
        while len(pillars) < n_pillars and tries < 200:
            tries += 1
            p = np.array([rng.uniform(-2.6, -0.7), rng.uniform(-2.1, 2.1)])
            if np.linalg.norm(p - [START_X, self._start_y]) < 1.0:
                continue
            if any(np.linalg.norm(p - q) < 0.8 for q in pillars):
                continue
            pillars.append(p)

        def wall(n, cx, cy, hx, hy):
            return (f'<geom name="{n}" type="box" pos="{cx} {cy} {WALL_HH}" '
                    f'size="{hx} {hy} {WALL_HH}" rgba="0.55 0.55 0.6 1"/>')
        walls = [wall("wall_n", 0, ROOM_HALF_Y, ROOM_HALF_X, WALL_HT),
                 wall("wall_s", 0, -ROOM_HALF_Y, ROOM_HALF_X, WALL_HT),
                 wall("wall_e", ROOM_HALF_X, 0, WALL_HT, ROOM_HALF_Y),
                 wall("wall_w", -ROOM_HALF_X, 0, WALL_HT, ROOM_HALF_Y)]
        if gap_lo + ROOM_HALF_Y > 0.02:
            hy = (gap_lo + ROOM_HALF_Y) / 2
            walls.append(wall("wall_div_s", 0, -ROOM_HALF_Y + hy, WALL_HT, hy))
        if ROOM_HALF_Y - gap_hi > 0.02:
            hy = (ROOM_HALF_Y - gap_hi) / 2
            walls.append(wall("wall_div_n", 0, ROOM_HALF_Y - hy, WALL_HT, hy))
        pillar_geoms = "".join(
            f'<geom name="pillar{i}" type="cylinder" pos="{p[0]} {p[1]} 0.5" '
            f'size="{PILLAR_R} 0.5" rgba="0.7 0.45 0.2 1"/>' for i, p in enumerate(pillars))

        rf_sensors = "".join(f'<rangefinder name="rfs{i}" site="rf{i}" cutoff="{LIDAR_CUTOFF}"/>'
                             for i in range(8))

        # cylinders on the tray (base starts at START_X, start_y); world pos =
        # base + tray offset. seat with tiny domain-randomized offset.
        c1x = START_X + GX + rng.uniform(-0.004, 0.004)
        c1y = self._start_y + GY + rng.uniform(-0.004, 0.004)
        c1z = 0.055 + GZ + 0.004 + CYL_HH + 0.002
        c2z = c1z + 2 * CYL_HH + 0.002

        return f"""
<mujoco model="mirte_real_balance">
  <compiler angle="radian" meshdir="{MODEL_DIR}/"/>
  <option timestep="{PHYSICS_DT}" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  {self._asset_xml}
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="{ROOM_HALF_X+1} {ROOM_HALF_Y+1} 0.1"
          friction="1.2 0.01 0.001" rgba="0.85 0.85 0.85 1"/>
    {"".join(walls)}
    {pillar_geoms}
    <site name="target" pos="{TARGET_X} {target_y} 0.02" type="cylinder"
          size="{TARGET_RADIUS} 0.02" rgba="0.1 0.8 0.1 0.5"/>
    {self._robot_body}
    <body name="cyl1" pos="{c1x:.4f} {c1y:.4f} {c1z:.4f}">
      <freejoint/>
      <geom name="cyl1_g" type="cylinder" size="{CYL_R} {CYL_HH}" density="400"
            rgba="0.9 0.6 0.1 1"/>
    </body>
    <body name="cyl2" pos="{c1x:.4f} {c1y:.4f} {c2z:.4f}">
      <freejoint/>
      <geom name="cyl2_g" type="cylinder" size="{CYL_R} {CYL_HH}" density="400"
            rgba="0.9 0.2 0.1 1"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="act_x" joint="slide_x" kv="120" ctrlrange="-{MAX_LIN_VEL} {MAX_LIN_VEL}" forcerange="-80 80"/>
    <velocity name="act_y" joint="slide_y" kv="120" ctrlrange="-{MAX_LIN_VEL} {MAX_LIN_VEL}" forcerange="-80 80"/>
    <velocity name="act_w" joint="yaw" kv="12" ctrlrange="-{MAX_ANG_VEL} {MAX_ANG_VEL}" forcerange="-20 20"/>
  </actuator>
  <sensor>{rf_sensors}</sensor>
</mujoco>"""

    # ------------------------------------------------------------------ helpers
    def _tilt(self, body):
        return float(np.arccos(np.clip(self.data.body(body).xmat[8], -1.0, 1.0)))

    def _base_pose(self):
        pos = self.data.body("base_link").xpos[:2].copy()
        yaw = float(self.data.joint("yaw").qpos[0])
        return pos, yaw

    def _get_obs(self):
        pos, yaw = self._base_pose()
        c, s = np.cos(yaw), np.sin(yaw)
        dw = self._target - pos
        x_rel = c * dw[0] + s * dw[1]
        y_rel = -s * dw[0] + c * dw[1]
        dist = float(np.linalg.norm(dw))
        th = np.arctan2(dw[1], dw[0]) - yaw
        th = np.arctan2(np.sin(th), np.cos(th))
        vxw = float(self.data.joint("slide_x").qvel[0])
        vyw = float(self.data.joint("slide_y").qvel[0])
        wz = float(self.data.joint("yaw").qvel[0])
        vx, vy = c * vxw + s * vyw, -s * vxw + c * vyw
        lidar = self.data.sensordata[:8].copy()
        lidar[lidar < 0] = LIDAR_CUTOFF
        lidar = np.clip(lidar / LIDAR_CUTOFF, 0, 1)
        phi1, phi2 = self._tilt("cyl1"), self._tilt("cyl2")
        d1 = np.clip((phi1 - self._prev_phi[0]) / CTRL_DT, -10, 10)
        d2 = np.clip((phi2 - self._prev_phi[1]) / CTRL_DT, -10, 10)
        self._prev_phi[:] = (phi1, phi2)
        return np.array([x_rel/POS_SCALE, y_rel/POS_SCALE, th/np.pi,
                         vx/MAX_LIN_VEL, vy/MAX_LIN_VEL, wz/MAX_ANG_VEL, *lidar,
                         phi1, d1, phi2, d2, dist/POS_SCALE], dtype=np.float32)

    def _base_collision(self):
        base_geoms = set()
        for n in ("tray", "blade_l", "blade_r"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            if gid >= 0:
                base_geoms.add(gid)
        # frame_link box collisions too
        for i in range(self.model.ngeom):
            bid = self.model.geom_bodyid[i]
            bn = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bn == "frame_link" and self.model.geom_contype[i]:
                base_geoms.add(i)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in base_geoms) == (g2 in base_geoms):
                continue
            other = g2 if g1 in base_geoms else g1
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            if nm.startswith(("wall", "pillar")):
                return True
        return False

    def _dropped(self):
        if self._tilt("cyl1") > TILT_LIMIT or self._tilt("cyl2") > TILT_LIMIT:
            return True
        z1, z2 = self.data.body("cyl1").xpos[2], self.data.body("cyl2").xpos[2]
        tray_z = 0.055 + GZ
        if z1 < tray_z - 0.02:
            return True
        if z2 < z1 + 2 * CYL_HH - 0.06:
            return True
        return False

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.model is None or self.randomize_on_reset:
            xml = self._build_xml(self.np_random)
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data = mujoco.MjData(self.model)
            self.data.joint("slide_x").qpos[0] = START_X
            self.data.joint("slide_y").qpos[0] = self._start_y
            self.data.joint("yaw").qpos[0] = self.np_random.uniform(-0.4, 0.4)
            mujoco.mj_forward(self.model, self.data)
            # seat the cylinders on the tray's ACTUAL world position (accounts for
            # the randomized base yaw) with a small domain-randomized offset
            tid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tray")
            tx, ty, tz = self.data.geom_xpos[tid]
            for i, body in enumerate(("cyl1", "cyl2")):
                ja = self.model.jnt_qposadr[self.model.body(body).jntadr[0]]
                ox, oy = self.np_random.uniform(-0.004, 0.004, 2)
                self.data.qpos[ja:ja+3] = [tx + ox, ty + oy,
                                           tz + 0.004 + CYL_HH + 0.002 + i * (2 * CYL_HH + 0.002)]
                self.data.qpos[ja+3:ja+7] = [1, 0, 0, 0]
            mujoco.mj_forward(self.model, self.data)
            for _ in range(60):
                mujoco.mj_step(self.model, self.data)
            self._init_qpos = self.data.qpos.copy()
            self._init_qvel = self.data.qvel.copy()
        else:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self._init_qpos
            self.data.qvel[:] = self._init_qvel
            mujoco.mj_forward(self.model, self.data)

        pos, _ = self._base_pose()
        self._prev_dist = float(np.linalg.norm(self._target - pos))
        self._prev_action = np.zeros(3)
        self._prev_phi[:] = (self._tilt("cyl1"), self._tilt("cyl2"))
        self._steps = 0
        if self.render_mode == "human":
            self._relaunch_viewer()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        _, yaw = self._base_pose()
        c, s = np.cos(yaw), np.sin(yaw)
        vx, vy = action[0] * MAX_LIN_VEL, action[1] * MAX_LIN_VEL
        self.data.ctrl[0] = c * vx - s * vy
        self.data.ctrl[1] = s * vx + c * vy
        self.data.ctrl[2] = action[2] * MAX_ANG_VEL
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        obs = self._get_obs()
        pos, _ = self._base_pose()
        dist = float(np.linalg.norm(self._target - pos))
        phi1, phi2 = self._prev_phi
        progress = self._prev_dist - dist
        self._prev_dist = dist
        jerk = float(np.sum((action - self._prev_action) ** 2))
        self._prev_action = action.copy()
        reward = (W_PROGRESS*progress - W_TILT_BOT*phi1**2 - W_TILT_TOP*phi2**2
                  - W_JERK*jerk - TIME_PENALTY)
        terminated = False
        info = {"dist": dist, "phi1": phi1, "phi2": phi2}
        if self._dropped():
            reward -= P_FALL; terminated = True; info["failure"] = "dropped"
        elif self._base_collision():
            reward -= P_COLLISION; terminated = True; info["failure"] = "collision"
        else:
            spd = np.hypot(self.data.joint("slide_x").qvel[0], self.data.joint("slide_y").qvel[0])
            if dist < TARGET_RADIUS and spd < 0.15:
                reward += R_SUCCESS; terminated = True; info["success"] = True
        truncated = self._steps >= self.max_episode_steps
        if self.render_mode == "human" and self._viewer is not None:
            self._viewer.sync()
        return obs, float(reward), terminated, truncated, info

    def _relaunch_viewer(self):
        import mujoco.viewer
        if self._viewer is not None:
            self._viewer.close()
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def close(self):
        if self._viewer is not None:
            self._viewer.close(); self._viewer = None


if __name__ == "__main__":
    env = MirteRealBalanceEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (19,), obs.shape
    total = 0.0
    for t in range(400):
        obs, r, term, trunc, info = env.step(env.action_space.sample() * 0.25)
        total += r
        if term or trunc:
            print(f"episode ended step {t}: {info}")
            obs, _ = env.reset()
    print(f"real-MIRTE env smoke test OK, obs {obs.shape}, reward {total:.1f}")
