"""
MirteStackedBalanceEnv
======================

Custom Gymnasium environment for the Double Inverted Pendulum Mobile
Transport Problem on a simplified MIRTE Master mobile manipulator.

Task
----
A holonomic (mecanum) base carries two un-linked cylinders stacked
vertically between two rigid gripper blades that act as a transport tray.
The robot must drive from a random start pose, around randomized obstacle
pillars, through a randomized-width doorway (0.75 m - 1.2 m), to a target
zone -- without tipping either cylinder.

MDP
---
Action  (3,)  continuous in [-1, 1]:  [Vx, Vy, Wz]  (body frame)
Obs     (19,) flat vector:
    0  x_rel        target x in body frame            / POS_SCALE
    1  y_rel        target y in body frame            / POS_SCALE
    2  theta_rel    heading error to target           / pi
    3  vx           body-frame linear velocity x      / MAX_LIN_VEL
    4  vy           body-frame linear velocity y      / MAX_LIN_VEL
    5  wz           yaw rate                          / MAX_ANG_VEL
    6-13  lidar     8 ray-cast ranges                 / LIDAR_CUTOFF
    14 phi1         bottom cylinder tilt from gravity Z (rad)
    15 dphi1        finite-difference tilt rate (rad/s, clipped)
    16 phi2         top cylinder tilt (rad)
    17 dphi2        top cylinder tilt rate
    18 dist         distance to target                / POS_SCALE
    (the original spec lists 18 components but requests dim 19; the
     distance-to-goal scalar is added as the 19th)

Reward
------
R = w1 * progress
  - w2 * phi1^2  - w3 * phi2^2          (w3 > w2: top cylinder is chaotic)
  - w4 * ||a_t - a_{t-1}||^2            (action jerk)
  - time_penalty
  - P_collision / P_fall (terminal)
  + success bonus (terminal)

The world (doorway width & position, pillar layout, start/target y,
cylinder seating offsets) is re-randomized by rebuilding the MJCF model
on every reset().
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# -----------------------------------------------------------------------------
# World / robot constants
# -----------------------------------------------------------------------------
ROOM_HALF_X = 4.0          # room spans x in [-4, 4]
ROOM_HALF_Y = 2.5          # y in [-2.5, 2.5]
WALL_HH = 0.5              # wall half-height (walls are 1 m tall)
WALL_HT = 0.05             # wall half-thickness

DOOR_W_MIN, DOOR_W_MAX = 0.75, 1.20
N_PILLARS_MIN, N_PILLARS_MAX = 2, 4
PILLAR_R = 0.12

START_X = -3.3
TARGET_X = 3.2
TARGET_RADIUS = 0.35

TRAY_Z = 0.35              # top surface of the tray plate
CYL_R = 0.03
CYL_HH = 0.09              # cylinder half-height (18 cm tall)
# quasi-static tipping threshold: a > g*r/h = 9.81*0.03/0.09 ~ 3.3 m/s^2;
# the base can produce ~6 m/s^2, so aggressive driving WILL drop the stack

MAX_LIN_VEL = 0.8          # m/s   action scaling
MAX_ANG_VEL = 1.2          # rad/s
LIDAR_CUTOFF = 5.0
POS_SCALE = 8.0

PHYSICS_DT = 0.002
FRAME_SKIP = 10            # -> 50 Hz control
CTRL_DT = PHYSICS_DT * FRAME_SKIP

TILT_LIMIT = 0.5           # rad; either cylinder beyond this = dropped

# reward weights
W_PROGRESS = 5.0
W_TILT_BOT = 1.0
W_TILT_TOP = 3.0
W_JERK = 0.05
TIME_PENALTY = 0.01
P_COLLISION = 15.0
P_FALL = 15.0
R_SUCCESS = 30.0

LIDAR_ANGLES = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])


class MirteStackedBalanceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": int(1 / CTRL_DT)}

    def __init__(self, render_mode: str | None = None, max_episode_steps: int = 1500,
                 randomize_on_reset: bool = True):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        # When False, the MJCF world is built once and reused across resets
        # (only the state is re-zeroed). Required for a persistent interactive
        # viewer, whose window is bound to a single MjModel instance.
        self.randomize_on_reset = randomize_on_reset

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(19,), dtype=np.float32)

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._viewer = None
        self._init_qpos = None
        self._init_qvel = None

        self._target = np.zeros(2)
        self._prev_dist = 0.0
        self._prev_action = np.zeros(3)
        self._prev_phi = np.zeros(2)
        self._steps = 0

    # ------------------------------------------------------------------ MJCF
    def _build_xml(self, rng: np.random.Generator) -> str:
        door_w = rng.uniform(DOOR_W_MIN, DOOR_W_MAX)
        door_cy = rng.uniform(-1.4, 1.4)
        gap_lo = max(door_cy - door_w / 2, -ROOM_HALF_Y + 0.1)
        gap_hi = min(door_cy + door_w / 2, ROOM_HALF_Y - 0.1)

        start_y = rng.uniform(-1.8, 1.8)
        target_y = rng.uniform(-1.8, 1.8)
        self._target = np.array([TARGET_X, target_y])

        # rejection-sample pillar positions in the first room
        pillars, tries = [], 0
        n_pillars = rng.integers(N_PILLARS_MIN, N_PILLARS_MAX + 1)
        while len(pillars) < n_pillars and tries < 200:
            tries += 1
            p = np.array([rng.uniform(-2.6, -0.7), rng.uniform(-2.1, 2.1)])
            if np.linalg.norm(p - [START_X, start_y]) < 1.0:
                continue
            if any(np.linalg.norm(p - q) < 0.8 for q in pillars):
                continue
            pillars.append(p)

        def wall(name, cx, cy, hx, hy):
            return (f'<geom name="{name}" type="box" pos="{cx} {cy} {WALL_HH}" '
                    f'size="{hx} {hy} {WALL_HH}" rgba="0.55 0.55 0.6 1"/>')

        walls = [
            wall("wall_n", 0, ROOM_HALF_Y, ROOM_HALF_X, WALL_HT),
            wall("wall_s", 0, -ROOM_HALF_Y, ROOM_HALF_X, WALL_HT),
            wall("wall_e", ROOM_HALF_X, 0, WALL_HT, ROOM_HALF_Y),
            wall("wall_w", -ROOM_HALF_X, 0, WALL_HT, ROOM_HALF_Y),
        ]
        # dividing wall with doorway gap [gap_lo, gap_hi]
        if gap_lo - (-ROOM_HALF_Y) > 0.02:
            hy = (gap_lo + ROOM_HALF_Y) / 2
            walls.append(wall("wall_div_s", 0, -ROOM_HALF_Y + hy, WALL_HT, hy))
        if ROOM_HALF_Y - gap_hi > 0.02:
            hy = (ROOM_HALF_Y - gap_hi) / 2
            walls.append(wall("wall_div_n", 0, ROOM_HALF_Y - hy, WALL_HT, hy))

        pillar_geoms = "".join(
            f'<geom name="pillar{i}" type="cylinder" pos="{p[0]} {p[1]} 0.5" '
            f'size="{PILLAR_R} 0.5" rgba="0.7 0.45 0.2 1"/>'
            for i, p in enumerate(pillars))

        # 8 horizontal rangefinder sites around the base, +Z of each site = ray dir
        rf_sites, rf_sensors = [], []
        for i, a in enumerate(LIDAR_ANGLES):
            c, s = np.cos(a), np.sin(a)
            rf_sites.append(
                f'<site name="rf{i}" pos="{0.22 * c:.4f} {0.22 * s:.4f} 0.25" '
                f'zaxis="{c:.6f} {s:.6f} 0" size="0.005"/>')
            rf_sensors.append(f'<rangefinder name="rfs{i}" site="rf{i}" cutoff="{LIDAR_CUTOFF}"/>')

        # cylinder seating randomization (domain randomization of the stack)
        c1x = START_X + rng.uniform(-0.005, 0.005)
        c1y = start_y + rng.uniform(-0.005, 0.005)
        c2x = c1x + rng.uniform(-0.008, 0.008)
        c2y = c1y + rng.uniform(-0.008, 0.008)
        c1z = TRAY_Z + 0.005 + CYL_HH
        c2z = c1z + 2 * CYL_HH + 0.002

        return f"""
<mujoco model="mirte_stacked_balance">
  <option timestep="{PHYSICS_DT}" integrator="implicitfast"/>
  <default>
    <geom friction="1.0 0.02 0.002" density="600"/>
  </default>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="{ROOM_HALF_X + 1} {ROOM_HALF_Y + 1} 0.1"
          rgba="0.85 0.85 0.85 1"/>
    {"".join(walls)}
    {pillar_geoms}
    <site name="target" pos="{TARGET_X} {target_y} 0.02" type="cylinder"
          size="{TARGET_RADIUS} 0.02" rgba="0.1 0.8 0.1 0.5"/>

    <body name="base" pos="{START_X} {start_y} 0">
      <joint name="slide_x" type="slide" axis="1 0 0" damping="1"/>
      <joint name="slide_y" type="slide" axis="0 1 0" damping="1"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0.5"/>
      <geom name="base_col" type="box" pos="0 0 0.12" size="0.20 0.17 0.10"
            density="250" rgba="0.2 0.3 0.8 1"/>
      <!-- locked-rigid gripper: mast + tray plate + two parallel blades -->
      <geom name="mast" type="box" pos="-0.10 0 0.28" size="0.02 0.02 0.06"
            rgba="0.3 0.3 0.3 1"/>
      <geom name="tray" type="box" pos="0 0 {TRAY_Z - 0.005}" size="0.10 0.10 0.005"
            friction="1.2 0.02 0.002" rgba="0.3 0.3 0.3 1"/>
      <geom name="blade_l" type="box" pos="0 0.040 {TRAY_Z + 0.03}"
            size="0.10 0.006 0.03" rgba="0.6 0.1 0.1 1"/>
      <geom name="blade_r" type="box" pos="0 -0.040 {TRAY_Z + 0.03}"
            size="0.10 0.006 0.03" rgba="0.6 0.1 0.1 1"/>
      <camera name="realsense" pos="0.21 0 0.30" xyaxes="0 -1 0 0 0 1" fovy="58"/>
      {"".join(rf_sites)}
    </body>

    <body name="cyl1" pos="{c1x:.4f} {c1y:.4f} {c1z:.4f}">
      <freejoint/>
      <geom name="cyl1_g" type="cylinder" size="{CYL_R} {CYL_HH}"
            density="400" rgba="0.9 0.6 0.1 1"/>
    </body>
    <body name="cyl2" pos="{c2x:.4f} {c2y:.4f} {c2z:.4f}">
      <freejoint/>
      <geom name="cyl2_g" type="cylinder" size="{CYL_R} {CYL_HH}"
            density="400" rgba="0.9 0.2 0.1 1"/>
    </body>
  </worldbody>

  <actuator>
    <velocity name="act_x" joint="slide_x" kv="80" forcerange="-40 40"
              ctrlrange="-{MAX_LIN_VEL} {MAX_LIN_VEL}"/>
    <velocity name="act_y" joint="slide_y" kv="80" forcerange="-40 40"
              ctrlrange="-{MAX_LIN_VEL} {MAX_LIN_VEL}"/>
    <velocity name="act_w" joint="yaw" kv="8" forcerange="-20 20"
              ctrlrange="-{MAX_ANG_VEL} {MAX_ANG_VEL}"/>
  </actuator>

  <sensor>
    {"".join(rf_sensors)}
  </sensor>
</mujoco>"""

    # ------------------------------------------------------------------ helpers
    def _tilt(self, body_name: str) -> float:
        """Tilt from global +Z: arccos of R[2,2] (xmat index 8)."""
        r22 = self.data.body(body_name).xmat[8]
        return float(np.arccos(np.clip(r22, -1.0, 1.0)))

    def _base_pose(self):
        pos = self.data.body("base").xpos[:2].copy()
        yaw = float(self.data.joint("yaw").qpos[0])
        return pos, yaw

    def _get_obs(self) -> np.ndarray:
        pos, yaw = self._base_pose()
        c, s = np.cos(yaw), np.sin(yaw)

        d_world = self._target - pos
        # rotate into body frame
        x_rel = c * d_world[0] + s * d_world[1]
        y_rel = -s * d_world[0] + c * d_world[1]
        dist = float(np.linalg.norm(d_world))
        theta_rel = np.arctan2(d_world[1], d_world[0]) - yaw
        theta_rel = np.arctan2(np.sin(theta_rel), np.cos(theta_rel))

        vxw = float(self.data.joint("slide_x").qvel[0])
        vyw = float(self.data.joint("slide_y").qvel[0])
        wz = float(self.data.joint("yaw").qvel[0])
        vx = c * vxw + s * vyw
        vy = -s * vxw + c * vyw

        lidar = self.data.sensordata[:8].copy()
        lidar[lidar < 0] = LIDAR_CUTOFF          # -1 means "no hit"
        lidar = np.clip(lidar / LIDAR_CUTOFF, 0.0, 1.0)

        phi1, phi2 = self._tilt("cyl1"), self._tilt("cyl2")
        dphi1 = np.clip((phi1 - self._prev_phi[0]) / CTRL_DT, -10, 10)
        dphi2 = np.clip((phi2 - self._prev_phi[1]) / CTRL_DT, -10, 10)
        self._prev_phi[:] = (phi1, phi2)

        return np.array(
            [x_rel / POS_SCALE, y_rel / POS_SCALE, theta_rel / np.pi,
             vx / MAX_LIN_VEL, vy / MAX_LIN_VEL, wz / MAX_ANG_VEL,
             *lidar,
             phi1, dphi1, phi2, dphi2,
             dist / POS_SCALE],
            dtype=np.float32)

    def _base_collision(self) -> bool:
        base_geoms = {self.model.geom(n).id
                      for n in ("base_col", "tray", "blade_l", "blade_r", "mast")}
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            if (g1 in base_geoms) == (g2 in base_geoms):
                continue
            other = g2 if g1 in base_geoms else g1
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            if name.startswith(("wall", "pillar")):
                return True
        return False

    def _cylinders_dropped(self) -> bool:
        phi1, phi2 = self._tilt("cyl1"), self._tilt("cyl2")
        if phi1 > TILT_LIMIT or phi2 > TILT_LIMIT:
            return True
        z1 = self.data.body("cyl1").xpos[2]
        z2 = self.data.body("cyl2").xpos[2]
        if z1 < TRAY_Z - 0.05:                     # bottom cylinder left the tray
            return True
        if z2 < z1 + 2 * CYL_HH - 0.06:            # top cylinder slid off the stack
            return True
        return False

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.model is None or self.randomize_on_reset:
            xml = self._build_xml(self.np_random)
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data = mujoco.MjData(self.model)

            # random initial heading (cylinders sit at the yaw axis, so the
            # stack position is unaffected by the initial rotation)
            self.data.joint("yaw").qpos[0] = self.np_random.uniform(-0.4, 0.4)
            mujoco.mj_forward(self.model, self.data)

            # let the stack settle onto the tray
            for _ in range(50):
                mujoco.mj_step(self.model, self.data)

            self._init_qpos = self.data.qpos.copy()
            self._init_qvel = self.data.qvel.copy()
        else:
            # reuse the existing world (keeps a persistent viewer valid);
            # restore the settled initial state instead of rebuilding
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
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        _, yaw = self._base_pose()
        c, s = np.cos(yaw), np.sin(yaw)
        vx_cmd = action[0] * MAX_LIN_VEL
        vy_cmd = action[1] * MAX_LIN_VEL
        # body-frame command -> world-frame slide actuators
        self.data.ctrl[0] = c * vx_cmd - s * vy_cmd
        self.data.ctrl[1] = s * vx_cmd + c * vy_cmd
        self.data.ctrl[2] = action[2] * MAX_ANG_VEL

        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        obs = self._get_obs()
        pos, _ = self._base_pose()
        dist = float(np.linalg.norm(self._target - pos))
        phi1, phi2 = self._prev_phi          # updated inside _get_obs

        progress = self._prev_dist - dist
        self._prev_dist = dist
        jerk = float(np.sum((action - self._prev_action) ** 2))
        self._prev_action = action.copy()

        reward = (W_PROGRESS * progress
                  - W_TILT_BOT * phi1 ** 2
                  - W_TILT_TOP * phi2 ** 2
                  - W_JERK * jerk
                  - TIME_PENALTY)

        terminated = False
        info = {"dist": dist, "phi1": phi1, "phi2": phi2}

        if self._cylinders_dropped():
            reward -= P_FALL
            terminated = True
            info["failure"] = "dropped"
        elif self._base_collision():
            reward -= P_COLLISION
            terminated = True
            info["failure"] = "collision"
        else:
            speed = np.hypot(self.data.joint("slide_x").qvel[0],
                             self.data.joint("slide_y").qvel[0])
            if dist < TARGET_RADIUS and speed < 0.15:
                reward += R_SUCCESS
                terminated = True
                info["success"] = True

        truncated = self._steps >= self.max_episode_steps

        if self.render_mode == "human" and self._viewer is not None:
            self._viewer.sync()

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ render
    def _relaunch_viewer(self):
        import mujoco.viewer
        if self._viewer is not None:
            self._viewer.close()
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


if __name__ == "__main__":
    # quick random-action smoke test
    env = MirteStackedBalanceEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (19,), obs.shape
    total = 0.0
    for t in range(300):
        obs, r, term, trunc, info = env.step(env.action_space.sample() * 0.3)
        total += r
        if term or trunc:
            print(f"episode ended at step {t}: {info}")
            obs, _ = env.reset()
    print(f"smoke test OK, obs shape {obs.shape}, cumulative reward {total:.2f}")
