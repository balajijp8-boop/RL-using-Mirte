"""
MirteGimbalBalanceEnv
=====================

v3 of the task: the REAL MIRTE Master drives over rough terrain (scattered
low bumps + a low staircase crossing the second room) while the arm carries
the two stacked cylinders on a 2-axis POWERED GIMBAL tray that actively
keeps the stack level — like a camera gimbal.

Physics changes vs mirte_real_env:
  - base_link has a FREEJOINT and rests on 4 free-rolling caster spheres at
    the wheel positions -> the chassis genuinely pitches/rolls/climbs.
  - the base is driven by a velocity-tracking force controller (same
    body-frame [Vx, Vy, Wz] command interface as before).
  - the tray hangs from the gripper via two powered hinge joints
    (gimbal_a, gimbal_b). Every control step a feedback leveling law
    measures the tray normal and counter-rotates the gimbal; the policy
    adds learned trim on top (anticipatory "waiter" tilting).

MDP:
  Action (5): [Vx, Vy, Wz, trim_a, trim_b]  in [-1, 1]
  Obs   (25): [ 0- 2] target x/y (body frame), heading error
              [ 3- 5] body-frame vx, vy, yaw rate
              [ 6-13] 8 lidar ranges
              [14-17] phi1, dphi1, phi2, dphi2 (cylinder tilts)
              [18]    distance to target
              [19-20] base roll, pitch
              [21-22] gimbal_a, gimbal_b joint angles
              [23-24] tray tilt from vertical, tray tilt rate
  Reward: 5*progress - 1*phi1^2 - 3*phi2^2 - 0.5*tray_tilt^2
          - 0.05*||da||^2 - 0.01  (+30 success, -15 drop/collision/flip)
"""

from __future__ import annotations
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import lxml.etree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "mirte_model")
ROBOT_XML = os.path.join(MODEL_DIR, "mirte_master.xml")

# ---- world -------------------------------------------------------------------
ROOM_HALF_X, ROOM_HALF_Y = 4.0, 2.5
WALL_HH, WALL_HT = 0.5, 0.05
DOOR_W_MIN, DOOR_W_MAX = 0.75, 1.20
N_PILLARS_MIN, N_PILLARS_MAX = 2, 3
PILLAR_R = 0.12
START_X, TARGET_X, TARGET_RADIUS = -3.3, 3.2, 0.35

# terrain
STAIR_RISE_MIN, STAIR_RISE_MAX = 0.010, 0.016
N_BUMPS = 12

# control
MAX_LIN_VEL, MAX_ANG_VEL = 0.8, 1.2
KP_V, F_MAX = 30.0, 90.0           # cap -> a_max ~6.4 m/s^2 (tip threshold ~3.3)
KP_W, TZ_MAX = 4.0, 6.0
ACC_CLIP = 4.0                     # m/s^2, feedforward tilt authority
TRIM_SCALE = 0.15                  # rad, policy gimbal trim authority
LEVEL_GAIN = 1.0                   # scripted leveling feedback gain
# tray-tilt catch is ineffective for high-friction flat-based stacks (the stack
# tips WITH the tray); the effective catch is the BASE reflex below. Tray gains
# default 0 but stay tunable for experiments.
K_CATCH = 0.0
K_LEAND = 0.0
CATCH_CLIP = 0.25                  # max catch tilt component (rad-equivalent)
CATCH_DEADBAND = 0.02              # rad; ignore micro-lean so rest stays quiet
# cart-pole reflex as an ACCELERATION law (a ~ g*lean neutralizes gravity's
# tipping torque; rate term damps the swing), integrated into a leaky
# reflex-velocity state so the catch releases smoothly instead of limit-cycling
KA_REFLEX, KD_REFLEX = 14.0, 4.0
REFLEX_LEAK = 0.996                # per physics substep (0.96 per control step)
REFLEX_VMAX = 0.8
W_LEAN1, W_LEAN2 = 0.35, 0.65      # cylinder lean blend for the reflex signal
RATE_ALPHA = 0.4                   # lean-rate EMA (per substep)
REFLEX_DEADBAND = 0.01             # rad

LIDAR_CUTOFF, POS_SCALE = 5.0, 8.0
PHYSICS_DT, FRAME_SKIP = 0.002, 10
CTRL_DT = PHYSICS_DT * FRAME_SKIP
TILT_LIMIT = 0.5
FLIP_LIMIT = 0.6

W_PROGRESS, W_TILT_BOT, W_TILT_TOP, W_TRAY = 5.0, 1.0, 3.0, 0.5
W_JERK, TIME_PENALTY = 0.05, 0.01
P_FAIL, R_SUCCESS = 15.0, 30.0

LIDAR_ANGLES = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])

ARM_POSE = {"shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.0,
            "elbow_joint": -0.5, "wrist_joint": 0.0, "gripper_joint": 0.2}
ARM_BODIES = ("shoulder_pan", "shoulder_lift", "elbow", "wrist", "gripper",
              "_gripper_link_r", "gripper_finger_r", "_gripper_link_l",
              "gripper_finger_l", "_Gripper_r")
WHEELS = ("front_left_wheel", "rear_left_wheel", "front_right_wheel", "rear_right_wheel")
# caster spheres in frame_link coords (measured from the wheel collision geoms)
CASTERS = [(+0.10, -0.085), (+0.10, +0.085), (-0.10, -0.085), (-0.10, +0.085)]
CASTER_Z, CASTER_R = -0.0455, 0.05

CYL_R, CYL_HH = 0.03, 0.09


class MirteGimbalBalanceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": int(1 / CTRL_DT)}

    def __init__(self, render_mode=None, max_episode_steps=1500,
                 randomize_on_reset=True, gimbal_enabled=True):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.randomize_on_reset = randomize_on_reset
        self.gimbal_enabled = gimbal_enabled   # False => gimbal servos hold 0 (ablation)
        self.k_catch = K_CATCH                 # tunable catch gains
        self.k_leand = K_LEAND
        # stabilization architecture (found empirically, see README):
        # - gimbal: pure leveling. The waiter accel-feedforward is OFF by
        #   default: with a high-friction payload it drags the stack.
        # - base_reflex: ON. An inverted pendulum can only be caught by
        #   translating its support -> drive the base under the lean.
        self.base_reflex = True
        self.waiter_ff = False

        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float32)

        self.model = self.data = self._viewer = None
        self._target = np.zeros(2)
        self._prev_dist = 0.0
        self._prev_action = np.zeros(5)
        self._prev_phi = np.zeros(2)
        self._prev_tray_tilt = 0.0
        self._prev_vxy = np.zeros(2)
        self._acc_f = np.zeros(2)          # filtered base accel for feedforward
        self._prev_lean = np.zeros(2)      # stack lean tracker for catch feedback
        self._lean_rate_f = np.zeros(2)
        self._v_reflex = np.zeros(2)       # leaky reflex velocity state
        self._steps = 0
        self._start_y = 0.0

        self._asset_xml, self._robot_body = self._prepare_robot()

    # ------------------------------------------------------------------ robot
    def _prepare_robot(self):
        tree = ET.parse(ROBOT_XML)
        root = tree.getroot()
        asset = root.find("asset")
        base = root.find("worldbody").find(".//body[@name='base_link']")

        # free-floating base (needs its own inertial: original base_link is a dummy)
        base.insert(0, ET.Element("freejoint", name="base_free"))
        base.insert(1, ET.fromstring(
            '<inertial pos="0 0 0" mass="0.3" diaginertia="0.002 0.002 0.002"/>'))

        frame = base.find(".//body[@name='frame_link']")

        # 4 free-rolling caster spheres at the wheel contact points
        for i, (cx, cy) in enumerate(CASTERS):
            b = ET.SubElement(frame, "body", name=f"caster{i}",
                              pos=f"{cx} {cy} {CASTER_Z}")
            ET.SubElement(b, "joint", name=f"caster{i}_j", type="ball", damping="0.001")
            ET.SubElement(b, "geom", name=f"caster{i}_g", type="sphere",
                          size=f"{CASTER_R}", mass="0.15",
                          friction="1.0 0.005 0.0001", solref="0.015 0.7",
                          rgba="0.1 0.1 0.1 0.35")

        # real wheels stay visual-only
        for wn in WHEELS:
            for g in base.find(f".//body[@name='{wn}']").findall("geom"):
                g.set("contype", "0"); g.set("conaffinity", "0")

        # arm: collision-free (the known gotcha) + rigidly sprung at the tray pose
        for an in ARM_BODIES:
            for g in base.find(f".//body[@name='{an}']").findall("geom"):
                g.set("contype", "0"); g.set("conaffinity", "0")
        for jn, q in ARM_POSE.items():
            jel = base.find(f".//joint[@name='{jn}']")
            jel.set("stiffness", "5000"); jel.set("springref", f"{q}")
            jel.set("damping", "80")
            jel.attrib.pop("actuatorfrcrange", None)

        # 2-axis powered gimbal + tray + blades hanging from the gripper.
        # At the tray pose the gripper's local +y is (near) world-up, so both
        # gimbal axes (local z and local x) are horizontal.
        grip = base.find(".//body[@name='gripper']")
        gimbal = ET.fromstring(f"""
        <body name="gimbal_outer" pos="0 0.03 0">
          <joint name="gimbal_a" type="hinge" axis="0 0 1" range="-0.6 0.6" damping="0.3"/>
          <inertial pos="0 0 0" mass="0.02" diaginertia="1e-5 1e-5 1e-5"/>
          <body name="tray_body" pos="0 0 0">
            <joint name="gimbal_b" type="hinge" axis="1 0 0" range="-0.6 0.6" damping="0.3"/>
            <geom name="tray" type="box" size="0.055 0.004 0.055" pos="0 0 0"
                  friction="1.4 0.02 0.002" rgba="0.2 0.2 0.25 1"/>
            <geom name="blade_l" type="box" size="0.06 0.03 0.006" pos="0 0.03 0.048"
                  friction="1.4 0.02 0.002" rgba="0.25 0.25 0.3 1"/>
            <geom name="blade_r" type="box" size="0.06 0.03 0.006" pos="0 0.03 -0.048"
                  friction="1.4 0.02 0.002" rgba="0.25 0.25 0.3 1"/>
            <geom name="blade_f" type="box" size="0.006 0.03 0.055" pos="0.048 0.03 0"
                  friction="1.4 0.02 0.002" rgba="0.25 0.25 0.3 1"/>
            <geom name="blade_b" type="box" size="0.006 0.03 0.055" pos="-0.048 0.03 0"
                  friction="1.4 0.02 0.002" rgba="0.25 0.25 0.3 1"/>
          </body>
        </body>""")
        grip.append(gimbal)

        # lidar sites on frame_link
        for i, a in enumerate(LIDAR_ANGLES):
            c, s = np.cos(a), np.sin(a)
            ET.SubElement(frame, "site", name=f"rf{i}",
                          pos=f"{0.20*c:.4f} {0.20*s:.4f} 0.05",
                          zaxis=f"{c:.5f} {s:.5f} 0", size="0.005")

        return ET.tostring(asset).decode(), ET.tostring(base).decode()

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

        # ---- low staircase crossing room 2 (must be traversed) ----
        # stepped terraces with short ramped (beveled) edges: a rigid caster
        # cannot climb a sharp vertical riser without an impulsive jolt
        h = rng.uniform(STAIR_RISE_MIN, STAIR_RISE_MAX)
        x0 = rng.uniform(0.9, 1.3)
        segs = [(0.35, h), (0.60, 2 * h), (0.35, h)]     # up, plateau, down
        stair_geoms, xc = [], x0
        levels, bounds = [0.0], [x0]
        for i, (ln, hh) in enumerate(segs):
            stair_geoms.append(
                f'<geom name="stair{i}" type="box" '
                f'pos="{xc + ln/2:.3f} 0 {hh/2:.4f}" '
                f'size="{ln/2:.3f} {ROOM_HALF_Y} {hh/2:.4f}" '
                f'friction="1.2 0.01 0.001" solref="0.02 0.6" '
                f'rgba="0.6 0.6 0.65 1"/>')
            levels.append(hh); xc += ln; bounds.append(xc)
        levels.append(0.0)
        RUN = 0.06                                        # ramp run per edge
        for i, xb in enumerate(bounds):
            lo, hi = sorted((levels[i], levels[i + 1]))
            dh = hi - lo
            if dh < 1e-4:
                continue
            asc = levels[i + 1] > levels[i]               # rising in +x?
            cx = xb - RUN / 2 if asc else xb + RUN / 2
            ang = -np.arctan2(dh, RUN) if asc else np.arctan2(dh, RUN)
            L = 0.5 * np.hypot(RUN, dh) + 0.012
            stair_geoms.append(
                f'<geom name="ramp{i}" type="box" '
                f'pos="{cx:.4f} 0 {(lo + hi) / 2 - 0.002:.4f}" '
                f'quat="{np.cos(ang/2):.5f} 0 {np.sin(ang/2):.5f} 0" '
                f'size="{L:.4f} {ROOM_HALF_Y} 0.004" '
                f'friction="1.2 0.01 0.001" solref="0.02 0.6" '
                f'rgba="0.55 0.55 0.6 1"/>')

        # ---- scattered low bumps (surface irregularities) ----
        bump_geoms = []
        for i in range(N_BUMPS):
            for _ in range(50):
                bx = rng.uniform(-2.8, 0.8)
                by = rng.uniform(-2.2, 2.2)
                if np.hypot(bx - START_X, by - self._start_y) < 0.7:
                    continue
                if any(np.hypot(bx - p[0], by - p[1]) < 0.35 for p in pillars):
                    continue
                break
            hx, hy_ = rng.uniform(0.05, 0.12, 2)
            hz = rng.uniform(0.006, 0.012)
            yawq = rng.uniform(0, np.pi)
            # half-buried ellipsoids: rounded irregularities, kind to rigid wheels
            bump_geoms.append(
                f'<geom name="bump{i}" type="ellipsoid" pos="{bx:.3f} {by:.3f} 0" '
                f'size="{hx:.3f} {hy_:.3f} {hz:.4f}" '
                f'quat="{np.cos(yawq/2):.4f} 0 0 {np.sin(yawq/2):.4f}" '
                f'friction="1.2 0.01 0.001" rgba="0.7 0.68 0.6 1"/>')

        rf_sensors = "".join(f'<rangefinder name="rfs{i}" site="rf{i}" cutoff="{LIDAR_CUTOFF}"/>'
                             for i in range(8))

        return f"""
<mujoco model="mirte_gimbal_balance">
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
    {"".join(stair_geoms)}
    {"".join(bump_geoms)}
    <site name="target" pos="{TARGET_X} {target_y} 0.02" type="cylinder"
          size="{TARGET_RADIUS} 0.02" rgba="0.1 0.8 0.1 0.5"/>
    {self._robot_body}
    <body name="cyl1" pos="{START_X + 0.39:.3f} {self._start_y:.3f} 0.45">
      <freejoint/>
      <geom name="cyl1_g" type="cylinder" size="{CYL_R} {CYL_HH}" density="400"
            rgba="0.9 0.6 0.1 1"/>
    </body>
    <body name="cyl2" pos="{START_X + 0.39:.3f} {self._start_y:.3f} 0.65">
      <freejoint/>
      <geom name="cyl2_g" type="cylinder" size="{CYL_R} {CYL_HH}" density="400"
            rgba="0.9 0.2 0.1 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="srv_gimbal_a" joint="gimbal_a" kp="25" kv="0.8" ctrlrange="-0.6 0.6" forcerange="-8 8"/>
    <position name="srv_gimbal_b" joint="gimbal_b" kp="25" kv="0.8" ctrlrange="-0.6 0.6" forcerange="-8 8"/>
  </actuator>
  <sensor>{rf_sensors}</sensor>
</mujoco>"""

    # ------------------------------------------------------------------ helpers
    def _bid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _tilt(self, body):
        return float(np.arccos(np.clip(self.data.body(body).xmat[8], -1.0, 1.0)))

    def _base_rpy(self):
        q = self.data.joint("base_free").qpos[3:7]
        w, x, y, z = q
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return roll, pitch, yaw

    def _base_vel(self):
        """world-frame [wx wy wz vx vy vz] of base_link"""
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data,
                                 mujoco.mjtObj.mjOBJ_BODY, self._bid("base_link"), res, 0)
        return res

    def _tray_state(self):
        """tray up-vector (world), tilt angle from vertical"""
        R = self.data.body("tray_body").xmat.reshape(3, 3)
        n = R[:, 1]                      # tray local +y is 'up'
        tilt = float(np.arccos(np.clip(n[2], -1.0, 1.0)))
        return n, tilt

    def _apply_gimbal(self, trim_a=0.0, trim_b=0.0, catch=True):
        """gimbal law with three terms:
        1. leveling toward apparent gravity (waiter trick: n ~ (ax, ay, g))
        2. CATCH feedback: tilt toward the stack's lean (+ lean-rate damping)
           to push the cylinder base back under its CoM, cart-pole style
        3. policy trim (learned anticipation) on top."""
        if not self.gimbal_enabled:
            self.data.ctrl[0] = 0.0
            self.data.ctrl[1] = 0.0
            return
        n, _ = self._tray_state()
        tilt_xy = self._acc_f / 9.81 if self.waiter_ff else np.zeros(2)
        if catch:
            a1 = self.data.body("cyl1").xmat.reshape(3, 3)[:, 2]
            a2 = self.data.body("cyl2").xmat.reshape(3, 3)[:, 2]
            lean = W_LEAN1 * a1[:2] + W_LEAN2 * a2[:2]
            rate = (lean - self._prev_lean) / PHYSICS_DT
            self._prev_lean = lean.copy()
            self._lean_rate_f = (1 - RATE_ALPHA) * self._lean_rate_f + RATE_ALPHA * rate
            mag = np.linalg.norm(lean)
            if mag > CATCH_DEADBAND:
                lean_eff = lean * (mag - CATCH_DEADBAND) / mag
                tilt_xy = tilt_xy + np.clip(
                    self.k_catch * lean_eff + self.k_leand * self._lean_rate_f,
                    -CATCH_CLIP, CATCH_CLIP)
        n_des = np.array([tilt_xy[0], tilt_xy[1], 1.0])
        n_des /= np.linalg.norm(n_des)
        r = np.cross(n, n_des)             # small-angle correction rotation vector
        for k, jn in enumerate(("gimbal_a", "gimbal_b")):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            axis = self.data.xaxis[jid]
            q = self.data.joint(jn).qpos[0]
            tgt = q + LEVEL_GAIN * float(np.dot(r, axis)) + (trim_a if k == 0 else trim_b)
            self.data.ctrl[k] = np.clip(tgt, -0.6, 0.6)

    def _apply_drive(self, vx_c, vy_c, wz_c):
        """velocity-tracking force controller on the free base, plus a
        cart-pole reflex: steer the base under a leaning stack to catch it"""
        _, _, yaw = self._base_rpy()
        c, s = np.cos(yaw), np.sin(yaw)
        v_des = np.array([c * vx_c - s * vy_c, s * vx_c + c * vy_c])
        if self.gimbal_enabled and self.base_reflex:
            lean, lr = self._prev_lean, self._lean_rate_f
            mag = np.linalg.norm(lean)
            lean_eff = lean * max(mag - REFLEX_DEADBAND, 0.0) / (mag + 1e-9)
            a_cmd = KA_REFLEX * lean_eff + KD_REFLEX * lr
            self._v_reflex = (self._v_reflex + a_cmd * PHYSICS_DT) * REFLEX_LEAK
            vn = np.linalg.norm(self._v_reflex)
            if vn > REFLEX_VMAX:
                self._v_reflex *= REFLEX_VMAX / vn
            v_des = v_des + self._v_reflex
        vel = self._base_vel()
        v_cur, wz_cur = vel[3:5], vel[2]
        mtot = 14.0
        F = mtot * KP_V * (v_des - v_cur)
        Fn = np.linalg.norm(F)
        if Fn > F_MAX:
            F *= F_MAX / Fn
        tz = np.clip(KP_W * mtot * 0.05 * (wz_c - wz_cur), -TZ_MAX, TZ_MAX)
        bid = self._bid("base_link")
        self.data.xfrc_applied[bid, :] = 0
        self.data.xfrc_applied[bid, 0:2] = F
        self.data.xfrc_applied[bid, 5] = tz

    def _get_obs(self):
        roll, pitch, yaw = self._base_rpy()
        pos = self.data.body("base_link").xpos[:2]
        c, s = np.cos(yaw), np.sin(yaw)
        dw = self._target - pos
        x_rel = c * dw[0] + s * dw[1]
        y_rel = -s * dw[0] + c * dw[1]
        dist = float(np.linalg.norm(dw))
        th = np.arctan2(dw[1], dw[0]) - yaw
        th = np.arctan2(np.sin(th), np.cos(th))
        vel = self._base_vel()
        vx = c * vel[3] + s * vel[4]
        vy = -s * vel[3] + c * vel[4]
        wz = vel[2]
        lidar = self.data.sensordata[:8].copy()
        lidar[lidar < 0] = LIDAR_CUTOFF
        lidar = np.clip(lidar / LIDAR_CUTOFF, 0, 1)
        phi1, phi2 = self._tilt("cyl1"), self._tilt("cyl2")
        d1 = np.clip((phi1 - self._prev_phi[0]) / CTRL_DT, -10, 10)
        d2 = np.clip((phi2 - self._prev_phi[1]) / CTRL_DT, -10, 10)
        self._prev_phi[:] = (phi1, phi2)
        _, tray_tilt = self._tray_state()
        dtray = np.clip((tray_tilt - self._prev_tray_tilt) / CTRL_DT, -10, 10)
        self._prev_tray_tilt = tray_tilt
        ga = float(self.data.joint("gimbal_a").qpos[0])
        gb = float(self.data.joint("gimbal_b").qpos[0])
        return np.array([x_rel/POS_SCALE, y_rel/POS_SCALE, th/np.pi,
                         vx/MAX_LIN_VEL, vy/MAX_LIN_VEL, wz/MAX_ANG_VEL, *lidar,
                         phi1, d1, phi2, d2, dist/POS_SCALE,
                         roll, pitch, ga, gb, tray_tilt, dtray], dtype=np.float32)

    def _base_collision(self):
        hit_geoms = set()
        for i in range(self.model.ngeom):
            bid = self.model.geom_bodyid[i]
            bn = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bn in ("frame_link", "tray_body") and self.model.geom_contype[i]:
                hit_geoms.add(i)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            if (g1 in hit_geoms) == (g2 in hit_geoms):
                continue
            other = g2 if g1 in hit_geoms else g1
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            if nm.startswith(("wall", "pillar")):
                return True
        return False

    def _failure(self):
        roll, pitch, _ = self._base_rpy()
        if abs(roll) > FLIP_LIMIT or abs(pitch) > FLIP_LIMIT:
            return "flipped"
        if self._tilt("cyl1") > TILT_LIMIT or self._tilt("cyl2") > TILT_LIMIT:
            return "dropped"
        tray_z = self.data.body("tray_body").xpos[2]
        z1, z2 = self.data.body("cyl1").xpos[2], self.data.body("cyl2").xpos[2]
        if z1 < tray_z + 0.02:
            return "dropped"
        if z2 < z1 + 2 * CYL_HH - 0.06:
            return "dropped"
        if self._base_collision():
            return "collision"
        return None

    # ------------------------------------------------------------------ gym API
    def _seat_cylinders(self):
        tray = self.data.body("tray_body").xpos
        for i, body in enumerate(("cyl1", "cyl2")):
            ja = self.model.jnt_qposadr[self.model.body(body).jntadr[0]]
            va = self.model.jnt_dofadr[self.model.body(body).jntadr[0]]
            ox, oy = self.np_random.uniform(-0.004, 0.004, 2)
            self.data.qpos[ja:ja+3] = [tray[0] + ox, tray[1] + oy,
                                       tray[2] + 0.004 + CYL_HH + 0.003
                                       + i * (2 * CYL_HH + 0.002)]
            self.data.qpos[ja+3:ja+7] = [1, 0, 0, 0]
            self.data.qvel[va:va+6] = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.model is None or self.randomize_on_reset:
            xml = self._build_xml(self.np_random)
            self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # base pose
        ja = self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")]
        psi = self.np_random.uniform(-0.3, 0.3)
        self.data.qpos[ja:ja+3] = [START_X, self._start_y, 0.02]
        self.data.qpos[ja+3:ja+7] = [np.cos(psi/2), 0, 0, np.sin(psi/2)]
        # arm pose
        for jn, q in ARM_POSE.items():
            self.data.joint(jn).qpos[0] = q
        mujoco.mj_forward(self.model, self.data)

        # settle robot on its casters; no catch feedback yet (cylinders are
        # not seated, their pose would inject nonsense into the gimbal)
        for k in range(150):
            self._apply_gimbal(catch=False)
            mujoco.mj_step(self.model, self.data)
        # seat the cylinders on the (now level) tray, settle again
        self._seat_cylinders()
        mujoco.mj_forward(self.model, self.data)
        self._prev_lean = np.zeros(2)
        self._lean_rate_f = np.zeros(2)
        self._v_reflex = np.zeros(2)
        for k in range(80):
            self._apply_gimbal()
            mujoco.mj_step(self.model, self.data)

        pos = self.data.body("base_link").xpos[:2]
        self._prev_dist = float(np.linalg.norm(self._target - pos))
        self._prev_action = np.zeros(5)
        self._prev_phi[:] = (self._tilt("cyl1"), self._tilt("cyl2"))
        _, self._prev_tray_tilt = self._tray_state()
        self._prev_vxy = self._base_vel()[3:5].copy()
        self._acc_f = np.zeros(2)
        self._steps = 0
        if self.render_mode == "human":
            self._relaunch_viewer()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        # filtered base acceleration estimate (for the waiter feedforward)
        vxy = self._base_vel()[3:5]
        acc_raw = (vxy - self._prev_vxy) / CTRL_DT
        self._prev_vxy = vxy.copy()
        self._acc_f = np.clip(0.7 * self._acc_f + 0.3 * acc_raw, -ACC_CLIP, ACC_CLIP)
        # drive + gimbal stabilization run at full physics rate (like real
        # kHz motor/IMU servo loops); the policy command is held constant
        # across the control period
        for _ in range(FRAME_SKIP):
            self._apply_drive(action[0] * MAX_LIN_VEL, action[1] * MAX_LIN_VEL,
                              action[2] * MAX_ANG_VEL)
            self._apply_gimbal(action[3] * TRIM_SCALE, action[4] * TRIM_SCALE)
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        obs = self._get_obs()
        pos = self.data.body("base_link").xpos[:2]
        dist = float(np.linalg.norm(self._target - pos))
        phi1, phi2 = self._prev_phi
        tray_tilt = self._prev_tray_tilt
        progress = self._prev_dist - dist
        self._prev_dist = dist
        jerk = float(np.sum((action - self._prev_action) ** 2))
        self._prev_action = action.copy()

        reward = (W_PROGRESS * progress - W_TILT_BOT * phi1**2
                  - W_TILT_TOP * phi2**2 - W_TRAY * tray_tilt**2
                  - W_JERK * jerk - TIME_PENALTY)
        terminated = False
        info = {"dist": dist, "phi1": phi1, "phi2": phi2, "tray_tilt": tray_tilt}
        fail = self._failure()
        if fail:
            reward -= P_FAIL
            terminated = True
            info["failure"] = fail
        else:
            vel = self._base_vel()
            if dist < TARGET_RADIUS and np.hypot(vel[3], vel[4]) < 0.15:
                reward += R_SUCCESS
                terminated = True
                info["success"] = True
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
    env = MirteGimbalBalanceEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (25,), obs.shape
    total = 0.0
    for t in range(300):
        obs, r, term, trunc, info = env.step(env.action_space.sample() * 0.25)
        total += r
        if term or trunc:
            print(f"episode ended step {t}: {info}")
            obs, _ = env.reset()
    print(f"gimbal env smoke test OK, obs {obs.shape}, reward {total:.1f}")
