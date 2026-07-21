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

v4 sensor/obs upgrade (motivated by the v2 eval: 70% collisions, 30% drops):
  - LiDAR densified 8 -> 16 rays (22.5 deg): a 0.12 m pillar beyond ~1 m used
    to fit BETWEEN 45-deg rays; doorposts likewise.
  - 3 forward "ultrasonic" rangefinders (like the real MIRTE's HC-SR04s)
    pitched down so they graze the floor ~0.4 / 0.7 / 1.0 m ahead: terrain
    (stairs, bumps) is 6-16 mm tall and invisible to the horizontal lidar,
    so the policy could never slow down BEFORE a jolt.
  - stack lean DIRECTION (body frame) + lean rate added to obs: previously
    the policy saw only tilt magnitudes, so its directional gimbal trim had
    no directional error signal to act on.
  - proximity reward term: smooth quadratic penalty inside D_SAFE clearance
    gives a collision gradient (before, the first warning was the terminal
    -20 at contact).

v5 (post-mortem of the 14M v3/v4 run -- balance solved, pillar navigation not):
  - LiDAR 16 -> 32 rays: probe showed the policy DOES react to lidar, but a
    0.24 m pillar was invisible between 22.5-deg rays beyond 0.63 m -> it
    reacted too late. 11.25-deg spacing holds detection to ~1.2 m.
  - reflex velocity (v_reflex) added to obs: the scripted cart-pole catch
    shoves the base up to +-1.2 m/s; previously invisible to the policy.
  - START-STATE CURRICULUM (Florensa et al. 2017, reverse curriculum): the
    v3 policy died at x ~ -1.4 and never SAW the doorway/stairs/goal in 14M
    steps. With start_curriculum=True (training only), 65% of episodes spawn
    at a random collision-free pose along the course so goal value propagates
    backward. Eval/video keep the fixed full-course start.
  - W_PROX doubled 0.15 -> 0.30 (stronger clearance gradient).
  - COMMAND SLEW LIMIT: raw actions are targets; the executed command ramps
    at <= ACC_CMD_MAX (2 m/s^2, under the ~3.3 m/s^2 tip threshold), so no
    single action can slam the base hard enough to throw the stack. The
    balance reflex is NOT filtered (the catch must stay fast).

MDP:
  Action (5): [Vx, Vy, Wz, trim_a, trim_b]  in [-1, 1]  (velocity TARGETS)
  Obs   (61): [ 0- 2] target x/y (body frame), heading error
              [ 3- 5] body-frame vx, vy, yaw rate
              [ 6-37] 32 lidar ranges (11.25 deg spacing, ray 0 = forward)
              [38-40] 3 ultrasonic ground-profile ranges (down-pitched, fwd)
              [41-44] phi1, dphi1, phi2, dphi2 (cylinder tilts)
              [45]    distance to target
              [46-47] base roll, pitch
              [48-49] gimbal_a, gimbal_b joint angles
              [50-51] tray tilt from vertical, tray tilt rate
              [52-53] stack lean direction (body frame x, y)
              [54-55] stack lean rate (body frame x, y)
              [56-57] reflex velocity (body frame x, y)
              [58-60] slew-limited command state (v_cmd x/y, wz_cmd)
  Reward: W_PROGRESS*progress - W_TILT_BOT*phi1^2 - W_TILT_TOP*phi2^2
          - W_TRAY*tray_tilt^2 - W_JERK*||da||^2 - W_PROX*prox_shortfall^2
          (+R_SUCCESS success, -P_FAIL drop/collision/flip)
"""

from __future__ import annotations
import gc
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
# doorway widened 0.75-1.20 -> 1.00-1.50 (v14 failure map: 30% of deaths at the
# doorway -- high-speed drops + collisions squeezing the tight gap). More
# clearance = the policy needn't thread it so precisely.
DOOR_W_MIN, DOOR_W_MAX = 1.00, 1.50
N_PILLARS_MIN, N_PILLARS_MAX = 2, 3
PILLAR_R = 0.12
START_X, TARGET_X, TARGET_RADIUS = -3.3, 3.2, 0.35

# terrain
# stair risers lowered 0.010-0.016 -> 0.006-0.010 (v14 failure map: 40% of
# deaths on the stairs -- stalls from fear of the jolt + some drops). Shorter
# risers = smaller climbing jolt = less payload shake, so less to freeze about.
STAIR_RISE_MIN, STAIR_RISE_MAX = 0.006, 0.010
# N_BUMPS 12 -> 6 (terrain_diag on the 73% policy): 4+ corridor bumps drop
# success to 69% vs 82% at 2-3 bumps -- 12 scattered lumps is a minefield, not a
# realistic delivery floor. THE strongest failure correlate; halving it targets
# the residual drops directly (core course -- stairs/door/payload -- unchanged).
N_BUMPS = 6

# control
# MAX_LIN_VEL 0.8 -> 0.5 (v10): every failure map across v4-v8 shows drops at
# 0.9-1.3 m/s, and no reward penalty ever durably slowed the policy down
# (progress pays too well). Following the project's own proven pattern --
# hard physical limits worked (slew, speed cap, stall-terminal), soft reward
# nudges thrashed (stair rushing, doorway freezing) -- dangerous cruise speed
# is now physically impossible to command. The balance reflex keeps its full
# 1.2 m/s catch authority; the course fits in ~800-1200 of 2500 steps at 0.4.
MAX_LIN_VEL, MAX_ANG_VEL = 0.5, 1.2
KP_V, F_MAX = 30.0, 90.0           # cap -> a_max ~6.4 m/s^2 (tip threshold ~3.3)
KP_W, TZ_MAX = 4.0, 6.0
# COMMAND SLEW LIMITING: a raw policy command could swing -0.8 -> +0.8 m/s in
# one 20 ms step, driving base accel to the 6.4 m/s^2 controller cap -- ~2x the
# ~3.3 m/s^2 tipping threshold. The stack then falls unless the reflex catches
# it. Slew-limiting the POLICY command (not the reflex -- the catch must stay
# fast) makes tip-inducing accelerations physically impossible to command.
ACC_CMD_MAX = 2.0                  # m/s^2 max commanded accel (lin, body frame)
YAW_ACC_CMD_MAX = 6.0              # rad/s^2 max commanded yaw accel
ACC_CLIP = 4.0                     # m/s^2, feedforward tilt authority
TRIM_SCALE = 0.30                  # rad, policy gimbal trim authority (doubled for learning)
LEVEL_GAIN = 1.0                   # scripted leveling feedback gain
# ACTIVE ARM LEVELING -- true camera-gimbal law (v2 of this feature).
# A camera gimbal does NOT do incremental error cleanup: it measures the
# BASE's absolute attitude (IMU) and counter-rotates its motors 1:1 and
# immediately, so the payload never moves in the world frame. Applied here:
# the wrist servo target is the ABSOLUTE negative of the chassis attitude
# projected on the wrist axis -- chassis pitches 7 deg => wrist commands
# -7 deg within the same control period. The tray gimbal polishes residuals.
K_WRIST_FF = 1.0                   # 1.0 = exact 1:1 counter-rotation
WRIST_LOOKAHEAD = 0.04             # s, gyro-rate anticipation (gimbal-inspired:
                                   # act on where the tilt is GOING, not just is)
WRIST_RANGE = 0.40                 # rad, clamp around the ARM_POSE wrist angle
# ARM CATCH via shoulder_pan: TESTED AND REJECTED (2026-07-09). The pan sweep
# has real authority (0.43 m/rad lateral) but a position-servo'd ~1 kg arm
# accelerates the box too abruptly: support acceleration IS inverted-pendulum
# forcing, so every tested config (K=0.5/1.0/2.0, with/without base reflex)
# DROPPED shoves that the smooth, high-inertia base reflex catches. Visible
# arm-catch motion is anti-correlated with stack survival on this payload.
# Keep catching in the base; the arm's proven job is attitude-hold (wrist).
K_CATCH = 0.0
K_LEAND = 0.0
CATCH_CLIP = 0.25
CATCH_DEADBAND = 0.02
KA_REFLEX, KD_REFLEX = 25.0, 8.0   # stronger base cart-pole catch (was 14, 4)
REFLEX_LEAK = 0.98                 # per physics substep (more aggressive catch release)
REFLEX_VMAX = 1.2                  # allow larger catch velocity (was 0.8)
# failure-map finding (16M autopsy): avg speed AT drop was 0.74 m/s and several
# drops happened at 1.1-1.8 m/s -- the reflex stacks its catch velocity on top
# of the command, and the resulting runaway speed causes the next, bigger jolt.
# Cap the COMBINED desired speed: full catch authority at cruise, no runaway.
# HISTORY: zero-shot lowering failed on v15@6M (policy trained at 1.3 relies on
# fast catches; capping it for free just moved the drop speed). BUT the v20@7M
# autopsy (50 eps, 70% policy) showed 13/20 residual drops at 1.0-1.3 m/s across
# ALL sections — the cap IS the drop regime. v22 TRAINS under 1.1 so the policy
# can adapt (learn to never need >1.1), which zero-shot couldn't test. Watch: if
# stalls/timeouts reappear or evals sag hard past 2M, revert to 1.3.
V_TOTAL_MAX = 1.1                  # m/s cap on ||command + reflex|| (v22 experiment)
W_LEAN1, W_LEAN2 = 0.35, 0.65      # cylinder lean blend for the reflex signal
RATE_ALPHA = 0.4                   # lean-rate EMA (per substep)
REFLEX_DEADBAND = 0.01             # rad

# ARM-TRANSLATION catch: swing the shoulder to slide the whole gripper (and the
# box it holds) horizontally back under the leaning stack's CoM -- the cart-pole
# "move the cart under the pole", but via the arm, which is far faster and more
# local than steering the 14 kg base. Reuses the same lean signal the base
# reflex uses. Joints are unlocked from their 5000-stiffness freeze to a light
# spring + position servo (like the wrist). Signs are auto-probed from the arm
# Jacobian so "move toward the lean" is always correct regardless of geometry.
# ARM_CATCH default OFF: as a FIXED reflex it tested net-negative for a
# smoothly-driving policy. rock_test: +16% survival under violent shaking but
# -15% under gentle (policy-like) motion; zero-shot v10 collapsed FASTER with it
# on (65 vs 135 steps). Kept fully wired (arm_catch=True re-enables) because all
# tests were UN-ADAPTED -- a fine-tune WITH it on is the only conclusive test.
# The proper realization of "use the arm to balance" is to give the POLICY the
# arm as an action (5->7 dim), not a hard-coded reflex -- see HANDOFF NEXT ACTION.
ARM_CATCH = False
# gains tuned on rock_test.py (worst-case shake, policy off): position term only
# (a lean-RATE term at any nonzero gain destabilized -- the arm over-reacts and
# forces the stack). k=1.2, kd=0 was the survival sweet spot; higher k hurt.
K_ARM = 1.2                        # lean (rad tilt) -> shoulder deflection (rad)
K_ARM_D = 0.0                      # lean-rate term: 0 (any nonzero forced oscillation)
ARM_CATCH_CLIP = 0.35              # rad, max shoulder deflection from ARM_POSE
ARM_CATCH_JOINTS = ("shoulder_lift_joint", "shoulder_pan_joint")

LIDAR_CUTOFF, POS_SCALE = 5.0, 8.0
PHYSICS_DT, FRAME_SKIP = 0.002, 10
CTRL_DT = PHYSICS_DT * FRAME_SKIP
TILT_LIMIT = 0.5
FLIP_LIMIT = 0.6

# LEARNABLE reward: tilt penalties reduced 10x, progress rewarded
W_PROGRESS = 10.0                  # was 5.0; reaching goal is the goal
W_TILT_BOT = 0.1                   # was 1.0; allow some wobble while learning
W_TILT_TOP = 0.3                   # was 3.0; top is important but not prohibitive
# v7 reward restructure (v5-ft autopsy: policy SURVIVES 1000+ steps but hovers
# ~3.8 m from the goal -- "survive without finishing" had become the optimum):
W_TRAY = 0.02                      # was 0.1: the hanging tray is SUPPOSED to
                                   # swing (pendulum waiter-tilt); stop fining it
W_JERK = 0.02
TIME_PENALTY = 0.006               # was 0.01 -> 0.006: v6 autopsy showed the
                                   # full 0.01 rush-pressure taught the policy
                                   # to blow through stairs at 0.9-1.2 m/s
                                   # (was <0.7 before); still >0 so hovering
                                   # forever remains a loser, just less urgent
# explicit terrain-speed shaping: stairs need caution, but nothing in the
# reward said so directly (only the emergent tilt penalty, which the time
# penalty out-competed). Penalize speed while astride the stair footprint.
W_STAIR_SPEED = 0.5
SAFE_STAIR_SPEED = 0.45            # m/s; matches pre-restructure survival speeds
# pillar-zone speed shaping (OPT-IN via env pillar_speed_shape=True; default OFF
# = trained env unchanged). Mirror of the stair shaping, aimed at the diagnosed
# front-half residual: weaving pillars at ~0.9 m/s whips the tall stack. Lateral
# accel ~ v^2 * yaw_rate, so cutting v is more effective than the yaw-cap lever
# (which washed). Quadratic penalty for speed over SAFE while in the pillar band.
W_PILLAR_SPEED = 0.5
SAFE_PILLAR_SPEED = 0.6
PILLAR_BAND = (-2.6, -0.55)        # x-range of the pillar field (failure_map section)
# stall = terminal failure. v8@4M eval: 5/30 episodes spent the FULL 2500
# steps frozen ~0.5 m before the doorway (rewards -42..-179). Rational under
# the old MDP: attempting the door risks -20, stalling only drips the time
# penalty. Making no-progress terminal (-10) makes freezing strictly worse
# than trying. Threshold is lenient: 5 cm per 8 s >> any legitimate pause.
STALL_WINDOW = 400                 # control steps (8 s) without progress
STALL_MIN_PROGRESS = 0.05          # m of best-distance improvement required
# P_STALL: 10 (v14) -> 13 for v16. v15 6M autopsy: 14 stall-deaths still cluster
# at the stair APPROACH (x~0.65-0.87) -- the policy is still too timid to climb.
# A bump un-freezes it. This backfired at v14 (P_STALL 16 -> rushing drops) ONLY
# because speed was uncapped; now paired with V_TOTAL_MAX 1.0, pushing can't run
# away into a drop, so 13 pushes through the freeze safely. (v16; v15 used 10.)
P_STALL = 13.0
P_FAIL, R_SUCCESS = 20.0, 80.0     # success sweetened 50 -> 80 (pairs with
                                   # gamma 0.995 train-side so the goal is
                                   # visible from mid-course: 80*0.995^500 ~ 6.6
                                   # vs 50*0.99^500 ~ 0.33 before)

# 32 rays @ 11.25 deg. Geometry: a pillar (dia 0.24 m) fits BETWEEN adjacent
# rays when d*sin(spacing) > 0.24. At 22.5 deg that meant invisible beyond
# 0.63 m (~1.5 s warning at cruise) -- the policy provably reacted to lidar but
# reacted too late. At 11.25 deg detection holds to ~1.2 m (~3 s warning).
LIDAR_ANGLES = np.deg2rad(np.arange(0.0, 360.0, 11.25))
N_LIDAR = len(LIDAR_ANGLES)
# forward "ultrasonic" ground-profile sensors: pitched down so the ray meets the
# floor ~0.4/0.7/1.0 m ahead (site sits ~0.145 m above the floor when settled).
# A 12 mm stair rise shortens the steepest ray by ~32 mm and the shallowest by
# ~86 mm (dh/sin(pitch)) -> clearly visible well before the casters arrive.
ULTRA_PITCH = np.deg2rad([22.0, 12.0, 8.0])
N_ULTRA = len(ULTRA_PITCH)
ULTRA_CUTOFF = 1.2
# proximity shaping: penalize raw lidar clearance below D_SAFE (sites sit on a
# 0.20 m ring, so raw 0.30 = 0.50 m from robot center). Door transit at the
# narrowest door (0.75 m) reads ~0.175 raw on the side rays -> small, passable
# penalty; wall-hugging or head-on approach is expensive.
W_PROX = 0.30
D_SAFE = 0.30

# start-state curriculum (reverse-curriculum style, Florensa et al. 2017):
# with prob (1 - P_STANDARD_START) spawn anywhere along the course instead of
# always at x=-3.3. The v3 policy died at x ~ -1.4 in 14M steps and therefore
# NEVER experienced the doorway/stairs/goal; the value function had no reason
# to pay the detour cost around pillars. Distributed starts let goal value
# propagate backward through the whole course.
P_STANDARD_START = 0.35
SPAWN_X_RANGE = (-3.2, 3.0)
SPAWN_Y_RANGE = (-1.9, 1.9)
SPAWN_CLEAR_PILLAR = 0.50          # m, min distance to pillar centers
SPAWN_CLEAR_BUMP = 0.28            # m, min distance to bump centers
SPAWN_CLEAR_WALL_X = 0.40          # m, exclusion half-band around divider wall
SPAWN_MIN_GOAL_DIST = 0.80         # m, don't spawn already inside success zone

ARM_POSE = {"shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.0,
            "elbow_joint": -0.5, "wrist_joint": 0.0, "gripper_joint": 0.25}
# finger four-bar linkage joints: free in the URDF conversion (never sprung).
# Spring-locked OPEN so the fingers genuinely straddle the box the arm carries.
# Sweep result: (r=-0.8, l=+0.8) = 10.0 cm separation with fingers still
# forward -- the physical maximum with usable reach. The box is slimmed to
# 9.4 cm (TRAY_HALF below) so it actually fits the open grip.
GRIP_LINK_POSE = {"_gripper_link_joint_r": -0.8, "_gripper_link_joint_l": 0.8}
TRAY_HALF = 0.047                  # was 0.055: box 9.4 cm wide fits 10 cm grip
BLADE_POS = 0.041                  # blade walls; inner clearance 7.0 cm for the
                                   # 6 cm cylinders (seat jitter is +-0.4 cm)
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

    # max_episode_steps 1500 -> 2500: at the policy's safe cruise (~0.3 m/s)
    # the full course + detours needs ~1400+ steps; 1500 made full-course
    # completion barely feasible temporally, so training rarely experienced it.
    def __init__(self, render_mode=None, max_episode_steps=2500,
                 randomize_on_reset=True, gimbal_enabled=True,
                 start_curriculum=False, tray_mount_y=0.03, tray_drop=0.06,
                 tray_mount_x=0.045, wall_hh=0.05, arm_catch=ARM_CATCH,
                 blade_density=1000, arm_action=False, clamp_mode=None,
                 yaw_cap=None, pillar_speed_shape=False):
        super().__init__()
        # arm_action: give the POLICY 2 extra action dims that drive the shoulder
        # lift/pan servos (the residual-drop lever after plain-ladder + speed-cap
        # both saturated at ~73%). Reuses the srv_arm_lift/pan actuators from the
        # arm_catch machinery, but the POLICY commands them (not the reflex law).
        self.arm_action = arm_action
        # clamp_mode (OPT-IN, default None = trained env unchanged): payload-
        # retention experiments to attack drops. "fingers" closes the gripper
        # fingers onto the tray-box; "tray_jaws" adds jaws on the gimbal tray
        # that grip the LOWER cylinder. Paired with a shorter tray_drop (pass
        # tray_drop=0.02) to cut pendulum swing. Built for visual approval
        # BEFORE any training; None reproduces the exact trained geometry.
        self.clamp_mode = clamp_mode
        # yaw_cap: opt-in cap (rad/s) on the applied yaw command; None = unchanged.
        # Targets front-half pillar/doorway weaving drops (see _apply usage).
        self.yaw_cap = yaw_cap
        # pillar_speed_shape: opt-in train-time reward penalty for over-speed in
        # the pillar field (default False = unchanged). Eval behavior unaffected
        # (reward isn't used to score eval); only the training gradient changes.
        self.pillar_speed_shape = pillar_speed_shape
        self._grip_pose = ({"_gripper_link_joint_r": -0.62,
                            "_gripper_link_joint_l": 0.62}
                           if clamp_mode == "fingers" else dict(GRIP_LINK_POSE))
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.randomize_on_reset = randomize_on_reset
        self.gimbal_enabled = gimbal_enabled   # False => gimbal servos hold 0 (ablation)
        # TRAINING ONLY: distributed start states (reverse curriculum). Keep
        # False for eval/video so success still means "full course from start".
        self.start_curriculum = start_curriculum
        # tray mounting geometry (gripper local frame; +y ~ world up at pose):
        #   tray_mount_y: gimbal pivot offset from gripper origin. +0.03 = the
        #     original "on top of the gripper"; negative = between the fingers.
        #   tray_drop: how far the tray surface hangs BELOW its own hinges.
        #     0 = tray CoM at the pivot (original). >0 = pendulum-style: the
        #     payload weight acts below the pivot, so the tray passively lags
        #     into accelerations (mechanical waiter tilt), servo still on top.
        # DEFAULTS (+0.03, 0.06): pivot at the original mount, tray hanging
        # 6 cm below it -> the BOX (tray+blades) sits between the gripper
        # fingers (fingers at z~0.266, box spans ~0.236-0.30). Ablation: drops
        # 9/16 -> 6/16 vs the rigid on-top original, peak tilt ~-30%. The
        # deeper (-0.04, 0.06) variant scored one episode better but put the
        # fingers THROUGH the payload cylinder - physically invalid.
        self.tray_mount_y = tray_mount_y
        self.tray_drop = tray_drop
        self.tray_mount_x = tray_mount_x   # forward: box nests between fingers
        # wall_hh: HALF-height of the 4 box walls (blades). rock_test.py shows
        # survival is FLAT (~140 steps, still drops) up to wall_hh 0.09, then
        # rises sharply: 0.11->min 1620, 0.12->100%. Fully caging the stack
        # (0.12+) trivializes the balance task -- rejected by user as cheating.
        # Default 0.05 (10 cm walls, the user's cap): firmly seats the LOWER
        # cylinder but leaves the UPPER cylinder to be balanced by control, so
        # completing the course stays a real "don't jostle the stack" task.
        # Walls are low-density so the leveling servo still swings the tray.
        self.wall_hh = wall_hh
        # arm_catch: unlock the shoulder and use it to translate the gripper
        # under the leaning stack (see ARM_CATCH constants). _arm_ids/_arm_sign
        # are resolved lazily on first control step (need the built model).
        # blade_density: box-wall material density. ORIGINAL was 1000 (default
        # MuJoCo). A mistaken 200 this session lightened the walls ~0.27->0.05 kg
        # (walls are ~40% of payload mass) and REGRESSED the tuned dynamics --
        # v10 collapsed 400->91 steps. Keep 1000 so wall HEIGHT is the only knob.
        self.blade_density = blade_density
        self.arm_catch = arm_catch
        self._arm_ids = None
        self.k_arm = K_ARM             # tunable arm-catch gains (swept offline)
        self.k_arm_d = K_ARM_D
        self.arm_sign = 1.0            # +1 move toward lean, -1 move away
        self.k_catch = K_CATCH                 # tunable catch gains
        self.k_leand = K_LEAND
        # stabilization architecture (found empirically, see README):
        # - gimbal: pure leveling. The waiter accel-feedforward is OFF by
        #   default: with a high-friction payload it drags the stack.
        # - base_reflex: ON. An inverted pendulum can only be caught by
        #   translating its support -> drive the base under the lean.
        self.base_reflex = True
        self.waiter_ff = False

        n_act = 7 if arm_action else 5     # +2 = shoulder lift/pan targets
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        # 17 base dims (target 3, vel 3, phi 4, dist 1, r/p 2, gimbal 2, tray 2)
        # + lidar + ultrasonic + lean dir/rate 4 + reflex velocity 2
        # + slew-limited command state 3 (v_cmd x/y, wz_cmd)
        # (v_reflex: the scripted cart-pole catch adds up to +-1.2 m/s to the
        # base velocity; the policy must SEE that shove to avoid being pushed
        # into obstacles by its own balance reflex.)
        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape=(17 + N_LIDAR + N_ULTRA + 4 + 2 + 3,), dtype=np.float32)

        self.model = self.data = self._viewer = None
        self._target = np.zeros(2)
        self._prev_dist = 0.0
        self._prev_action = np.zeros(self.action_space.shape[0])
        self._prev_phi = np.zeros(2)
        self._prev_tray_tilt = 0.0
        self._prev_vxy = np.zeros(2)
        self._acc_f = np.zeros(2)          # filtered base accel for feedforward
        self._prev_lean = np.zeros(2)      # stack lean tracker for catch feedback
        self._lean_rate_f = np.zeros(2)
        self._v_reflex = np.zeros(2)       # leaky reflex velocity state
        self._prev_lean_obs = np.zeros(2)  # lean tracker at CTRL_DT for the obs
        self._min_lidar = LIDAR_CUTOFF     # raw min lidar range, for prox reward
        self._v_cmd = np.zeros(2)          # slew-limited commanded velocity (body)
        self._wz_cmd = 0.0                 # slew-limited commanded yaw rate
        self._steps = 0
        self._start_y = 0.0
        self._resets_since_gc = 0

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
        active = {"wrist_joint"}
        if self.arm_catch or self.arm_action:
            active |= set(ARM_CATCH_JOINTS)   # shoulder joints the catch/policy drives
        for jn, q in {**ARM_POSE, **self._grip_pose}.items():
            jel = base.find(f".//joint[@name='{jn}']")
            if jn in active:
                # ACTIVE servo axis: light spring only -- its servo drives it;
                # a 5000 spring would fight the control law.
                jel.set("stiffness", "5"); jel.set("springref", f"{q}")
                jel.set("damping", "3")
            else:
                jel.set("stiffness", "5000"); jel.set("springref", f"{q}")
                jel.set("damping", "80")
            jel.attrib.pop("actuatorfrcrange", None)

        # 2-axis powered gimbal + tray + blades hanging from the gripper.
        # At the tray pose the gripper's local +y is (near) world-up, so both
        # gimbal axes (local z and local x) are horizontal.
        grip = base.find(".//body[@name='gripper']")
        mx = self.tray_mount_x            # forward offset: box sits IN the grip
        my = self.tray_mount_y            # pivot offset from gripper origin
        ty = -self.tray_drop              # tray surface offset below its hinges
        wh = self.wall_hh                 # wall half-height (cages the stack)
        by = wh - self.tray_drop          # walls rise from the tray surface up
        th, bp = TRAY_HALF, BLADE_POS
        # opt-in clamp jaws mounted ON the tray, gripping the LOWER cylinder
        # (cyl1). They move with the gimbal so they don't fight the leveling.
        jaws = ""
        if self.clamp_mode == "tray_jaws":
            jy = ty + 0.004 + CYL_HH           # lower-cylinder center, tray frame
            jx = CYL_R + 0.006                 # inner face just kisses the cylinder
            jaws = (
                f'<geom name="jaw_f" type="box" size="0.005 {CYL_HH} {CYL_R}" '
                f'pos="{jx} {jy} 0" friction="1.6 0.05 0.005" rgba="0.9 0.45 0.1 1"/>'
                f'<geom name="jaw_b" type="box" size="0.005 {CYL_HH} {CYL_R}" '
                f'pos="-{jx} {jy} 0" friction="1.6 0.05 0.005" rgba="0.9 0.45 0.1 1"/>')
        gimbal = ET.fromstring(f"""
        <body name="gimbal_outer" pos="{mx} {my} 0">
          <joint name="gimbal_a" type="hinge" axis="0 0 1" range="-0.6 0.6" damping="0.3"/>
          <inertial pos="0 0 0" mass="0.02" diaginertia="1e-5 1e-5 1e-5"/>
          <body name="tray_body" pos="0 0 0">
            <joint name="gimbal_b" type="hinge" axis="1 0 0" range="-0.6 0.6" damping="0.3"/>
            <geom name="tray" type="box" size="{th} 0.004 {th}" pos="0 {ty} 0"
                  friction="1.4 0.02 0.002" rgba="0.2 0.2 0.25 1"/>
            <geom name="blade_l" type="box" size="{th} {wh} 0.006" pos="0 {by} {bp}"
                  density="{self.blade_density}" friction="1.4 0.02 0.002" rgba="0.25 0.25 0.3 1"/>
            <geom name="blade_r" type="box" size="{th} {wh} 0.006" pos="0 {by} -{bp}"
                  density="{self.blade_density}" friction="1.4 0.02 0.002" rgba="0.25 0.25 0.3 1"/>
            <geom name="blade_f" type="box" size="0.006 {wh} {th}" pos="{bp} {by} 0"
                  density="{self.blade_density}" friction="1.4 0.02 0.002" rgba="0.25 0.25 0.3 1"/>
            <geom name="blade_b" type="box" size="0.006 {wh} {th}" pos="-{bp} {by} 0"
                  density="{self.blade_density}" friction="1.4 0.02 0.002" rgba="0.25 0.25 0.3 1"/>
            {jaws}
          </body>
        </body>""")
        grip.append(gimbal)

        # lidar sites on frame_link
        for i, a in enumerate(LIDAR_ANGLES):
            c, s = np.cos(a), np.sin(a)
            ET.SubElement(frame, "site", name=f"rf{i}",
                          pos=f"{0.20*c:.4f} {0.20*s:.4f} 0.05",
                          zaxis=f"{c:.5f} {s:.5f} 0", size="0.005")

        # ultrasonic ground-profile sites: front of frame, pitched down-forward
        for i, p in enumerate(ULTRA_PITCH):
            cp, sp = np.cos(p), np.sin(p)
            ET.SubElement(frame, "site", name=f"us{i}",
                          pos="0.20 0 0.05",
                          zaxis=f"{cp:.5f} 0 {-sp:.5f}", size="0.005")

        return ET.tostring(asset).decode(), ET.tostring(base).decode()

    # ------------------------------------------------------------------ scene
    def _build_xml(self, rng):
        door_w = rng.uniform(DOOR_W_MIN, DOOR_W_MAX)
        self._door_w = door_w                 # exposed for failure diagnostics
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
        self._pillars = [p.copy() for p in pillars]   # for curriculum spawn rejection

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
        # stepped terraces with short ramped (beveled) edges. NOTE: a gentle
        # long-ramp variant (RUN 0.20, longer segs) was trialed 2026-07-10; the
        # user chose to keep the ORIGINAL short-edge stairs (lower risers only,
        # STAIR_RISE above) once v15 proved completions transfer to them.
        h = rng.uniform(STAIR_RISE_MIN, STAIR_RISE_MAX)
        self._stair_h = h                     # exposed for failure diagnostics
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
        self._stair_band = (x0 - 0.25, xc + 0.25)     # spawn exclusion zone
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
        self._bumps = []
        for i in range(N_BUMPS):
            for _ in range(50):
                bx = rng.uniform(-2.8, 0.8)
                by = rng.uniform(-2.2, 2.2)
                if np.hypot(bx - START_X, by - self._start_y) < 0.7:
                    continue
                if any(np.hypot(bx - p[0], by - p[1]) < 0.35 for p in pillars):
                    continue
                break
            self._bumps.append((bx, by))
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
                             for i in range(N_LIDAR))
        rf_sensors += "".join(f'<rangefinder name="uss{i}" site="us{i}" cutoff="{ULTRA_CUTOFF}"/>'
                              for i in range(N_ULTRA))

        arm_actuators = ""
        if self.arm_catch or self.arm_action:
            _lift0 = ARM_POSE["shoulder_lift_joint"]
            _pan0 = ARM_POSE["shoulder_pan_joint"]
            # arm_action: stiff servo (kp 400) so holding pose ~= the old rigid
            # shoulder -> neutral behaves like the 5-dim parent (no wobble tax);
            # still tracks fast enough to catch. arm_catch reflex keeps kp 90.
            _kp, _kv, _frc = (250, 18, 30) if self.arm_action else (90, 7, 25)
            arm_actuators = (
                f'<position name="srv_arm_lift" joint="shoulder_lift_joint" '
                f'kp="{_kp}" kv="{_kv}" ctrlrange="{_lift0-ARM_CATCH_CLIP-0.05} '
                f'{_lift0+ARM_CATCH_CLIP+0.05}" forcerange="-{_frc} {_frc}"/>'
                f'<position name="srv_arm_pan" joint="shoulder_pan_joint" '
                f'kp="{_kp}" kv="{_kv}" ctrlrange="{_pan0-ARM_CATCH_CLIP-0.05} '
                f'{_pan0+ARM_CATCH_CLIP+0.05}" forcerange="-{_frc} {_frc}"/>')

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
    <position name="srv_wrist" joint="wrist_joint" kp="120" kv="8"
              ctrlrange="-1.6 1.6" forcerange="-30 30"/>
    {arm_actuators}
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
            self.data.ctrl[2] = ARM_POSE["wrist_joint"]   # hold pose, no leveling
            self._hold_arm()
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
        # ARM leveling, camera-gimbal style: ABSOLUTE 1:1 counter-rotation of
        # the chassis attitude (not incremental error cleanup). The wrist
        # target mirrors the base tilt on its axis every physics substep.
        u_c = self.data.body("base_link").xmat.reshape(3, 3)[:, 2]
        r_c = np.cross(u_c, np.array([0.0, 0.0, 1.0]))   # rotvec that levels base
        wj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_joint")
        waxis = self.data.xaxis[wj]
        w0 = ARM_POSE["wrist_joint"]
        omega = self._base_vel()[0:3]                    # world angular velocity
        rate = -float(omega[0] * waxis[0] + omega[1] * waxis[1])  # d(tilt)/dt on axis
        wtgt = (w0 + K_WRIST_FF * float(np.dot(r_c, waxis))
                + WRIST_LOOKAHEAD * rate)
        self.data.ctrl[2] = np.clip(wtgt, w0 - WRIST_RANGE, w0 + WRIST_RANGE)
        self._apply_arm_catch()

    def _arm_setup(self):
        """resolve actuator/dof ids for the arm-catch servos (lazy: needs model)."""
        if self._arm_ids is not None or not (self.arm_catch or self.arm_action):
            return self._arm_ids
        self._arm_ids = {
            "lift_act": self.model.actuator("srv_arm_lift").id,
            "pan_act": self.model.actuator("srv_arm_pan").id,
            "lift_dof": int(self.model.joint("shoulder_lift_joint").dofadr[0]),
            "pan_dof": int(self.model.joint("shoulder_pan_joint").dofadr[0]),
            "tray_bid": self._bid("tray_body"),
        }
        return self._arm_ids

    def _hold_arm(self):
        """park the arm-catch servos at the neutral pose (stabilization off)."""
        ids = self._arm_setup()
        if ids is None:
            return
        self.data.ctrl[ids["lift_act"]] = ARM_POSE["shoulder_lift_joint"]
        self.data.ctrl[ids["pan_act"]] = ARM_POSE["shoulder_pan_joint"]

    def _apply_arm_catch(self):
        """Swing the shoulder so the gripper (hence the box + stack support)
        translates toward the stack's lean -- cart-pole 'move the cart under the
        pole', via the arm. The arm Jacobian gives the world direction each
        shoulder joint moves the gripper; we drive each joint by how well that
        direction aligns with the lean (+ lean-rate damping), so the sign is
        always correct regardless of arm pose or base yaw."""
        if not self.arm_catch:        # policy drives the arm in arm_action mode
            return
        ids = self._arm_setup()
        if ids is None:
            return
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, None, ids["tray_bid"])
        lean, lr = self._prev_lean, self._lean_rate_f
        for act, dof, q0 in (
                (ids["lift_act"], ids["lift_dof"], ARM_POSE["shoulder_lift_joint"]),
                (ids["pan_act"], ids["pan_dof"], ARM_POSE["shoulder_pan_joint"])):
            g = jacp[:2, dof]                       # gripper (x,y) motion per qvel
            gn = float(np.linalg.norm(g))
            if gn < 1e-6:
                self.data.ctrl[act] = q0
                continue
            ghat = g / gn
            drive = self.arm_sign * (self.k_arm * float(np.dot(lean, ghat))
                                     + self.k_arm_d * float(np.dot(lr, ghat)))
            self.data.ctrl[act] = q0 + np.clip(drive, -ARM_CATCH_CLIP, ARM_CATCH_CLIP)

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
            # cap the combined speed: prevents the catch from runaway-
            # accelerating the base into ever-bigger terrain jolts
            vt = np.linalg.norm(v_des)
            if vt > V_TOTAL_MAX:
                v_des *= V_TOTAL_MAX / vt
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
        lidar = self.data.sensordata[:N_LIDAR].copy()
        lidar[lidar < 0] = LIDAR_CUTOFF
        self._min_lidar = float(lidar.min())          # raw, for the prox reward
        lidar = np.clip(lidar / LIDAR_CUTOFF, 0, 1)
        ultra = self.data.sensordata[N_LIDAR:N_LIDAR + N_ULTRA].copy()
        ultra[ultra < 0] = ULTRA_CUTOFF
        ultra = np.clip(ultra / ULTRA_CUTOFF, 0, 1)
        phi1, phi2 = self._tilt("cyl1"), self._tilt("cyl2")
        d1 = np.clip((phi1 - self._prev_phi[0]) / CTRL_DT, -10, 10)
        d2 = np.clip((phi2 - self._prev_phi[1]) / CTRL_DT, -10, 10)
        self._prev_phi[:] = (phi1, phi2)
        _, tray_tilt = self._tray_state()
        dtray = np.clip((tray_tilt - self._prev_tray_tilt) / CTRL_DT, -10, 10)
        self._prev_tray_tilt = tray_tilt
        ga = float(self.data.joint("gimbal_a").qpos[0])
        gb = float(self.data.joint("gimbal_b").qpos[0])
        # stack lean DIRECTION in the body frame (same signal the reflex uses),
        # plus its rate: the directional error the gimbal trim can act on.
        a1 = self.data.body("cyl1").xmat.reshape(3, 3)[:, 2]
        a2 = self.data.body("cyl2").xmat.reshape(3, 3)[:, 2]
        lean_w = W_LEAN1 * a1[:2] + W_LEAN2 * a2[:2]
        lean_b = np.array([c * lean_w[0] + s * lean_w[1],
                           -s * lean_w[0] + c * lean_w[1]])
        lrate_b = np.clip((lean_b - self._prev_lean_obs) / CTRL_DT, -5, 5)
        self._prev_lean_obs = lean_b.copy()
        # reflex velocity (body frame): the balance catch shoves the base by up
        # to REFLEX_VMAX; expose it so the policy can compensate near obstacles
        vr_b = np.array([c * self._v_reflex[0] + s * self._v_reflex[1],
                         -s * self._v_reflex[0] + c * self._v_reflex[1]]) / REFLEX_VMAX
        return np.array([x_rel/POS_SCALE, y_rel/POS_SCALE, th/np.pi,
                         vx/MAX_LIN_VEL, vy/MAX_LIN_VEL, wz/MAX_ANG_VEL,
                         *lidar, *ultra,
                         phi1, d1, phi2, d2, dist/POS_SCALE,
                         roll, pitch, ga, gb, tray_tilt, dtray,
                         *lean_b, *lrate_b, *vr_b,
                         self._v_cmd[0]/MAX_LIN_VEL, self._v_cmd[1]/MAX_LIN_VEL,
                         self._wz_cmd/MAX_ANG_VEL], dtype=np.float32)

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
        tray_z = self.data.geom("tray").xpos[2]
        z1, z2 = self.data.body("cyl1").xpos[2], self.data.body("cyl2").xpos[2]
        if z1 < tray_z + 0.02:
            return "dropped"
        if z2 < z1 + 2 * CYL_HH - 0.06:
            return "dropped"
        if self._base_collision():
            return "collision"
        return None

    def _sample_spawn(self):
        """Curriculum spawn: a collision-free pose anywhere along the course.

        Rejection-samples against the stored scene layout (pillars, divider
        wall band, stair band, bumps, walls, goal zone). Falls back to the
        standard start if no valid pose is found.

        Spawn mix (v13 stall autopsy): the back-half failure is now STALLS at
        the stairs, so drill the CLIMB. 30% pre-stairs run-up (start just before
        the stair band and immediately practice approaching+climbing), 25%
        goal-side (x>0.45 back half), 20% doorway approach (x in [-1.3,-0.5]),
        25% anywhere. Can't spawn ON the slope (rejected below)."""
        u = self.np_random.uniform()
        lo, hi = self._stair_band
        for _ in range(100):
            if u < 0.30:                                          # pre-stairs run-up
                sx = self.np_random.uniform(
                    max(SPAWN_CLEAR_WALL_X + 0.05, lo - 0.7), lo - 0.08)
            elif u < 0.55:
                sx = self.np_random.uniform(0.45, SPAWN_X_RANGE[1])
            elif u < 0.75:
                sx = self.np_random.uniform(-1.3, -0.5)
            else:
                sx = self.np_random.uniform(*SPAWN_X_RANGE)
            sy = self.np_random.uniform(*SPAWN_Y_RANGE)
            if abs(sx) < SPAWN_CLEAR_WALL_X:                       # divider wall
                continue
            if lo < sx < hi:                                       # stairs
                continue
            if np.hypot(sx - self._target[0], sy - self._target[1]) < SPAWN_MIN_GOAL_DIST:
                continue
            if any(np.hypot(sx - p[0], sy - p[1]) < SPAWN_CLEAR_PILLAR
                   for p in self._pillars):
                continue
            if any(np.hypot(sx - bx, sy - by) < SPAWN_CLEAR_BUMP
                   for bx, by in self._bumps):
                continue
            psi = self.np_random.uniform(-0.4, 0.4)
            return sx, sy, psi
        return START_X, self._start_y, self.np_random.uniform(-0.3, 0.3)

    # ------------------------------------------------------------------ gym API
    def _seat_cylinders(self):
        # geom center, not body origin: with tray_drop > 0 the tray surface
        # hangs below the tray_body frame, and cylinders must seat on the geom
        tray = self.data.geom("tray").xpos
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
            # drop the old native model/data explicitly before building the new
            # one -- this env rebuilds MJCF every episode, and over a long run
            # (16 parallel envs x thousands of resets) relying on GC to catch up
            # with unreferenced MjModel/MjData eventually starved the allocator
            # ("Could not allocate memory" mid-run).
            self.model = None
            self.data = None
            self._resets_since_gc += 1
            if self._resets_since_gc >= 200:
                gc.collect()
                self._resets_since_gc = 0
            self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._arm_ids = None            # re-resolve arm-catch ids for this model

        # base pose: standard start, or (curriculum) anywhere along the course
        ja = self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")]
        if (self.start_curriculum and self.randomize_on_reset
                and self.np_random.uniform() > P_STANDARD_START):
            sx, sy, psi = self._sample_spawn()
        else:
            sx, sy = START_X, self._start_y
            psi = self.np_random.uniform(-0.3, 0.3)
        self.data.qpos[ja:ja+3] = [sx, sy, 0.02]
        self.data.qpos[ja+3:ja+7] = [np.cos(psi/2), 0, 0, np.sin(psi/2)]
        # arm pose (incl. spring-locked open finger linkage)
        for jn, q in {**ARM_POSE, **self._grip_pose}.items():
            self.data.joint(jn).qpos[0] = q
        mujoco.mj_forward(self.model, self.data)

        # lean trackers must exist before the settle loop (the arm-catch law
        # reads them every substep)
        self._prev_lean = np.zeros(2)
        self._lean_rate_f = np.zeros(2)
        # hold the policy-driven arm servos at pose during settle (else their
        # ctrl defaults to 0 -> range-clamps the shoulder to -0.6, seating the
        # cylinders against a mis-positioned box)
        if self.arm_action:
            self._hold_arm()

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
        self._best_dist = self._prev_dist
        self._steps_since_best = 0
        self._prev_action = np.zeros(self.action_space.shape[0])
        self._prev_phi[:] = (self._tilt("cyl1"), self._tilt("cyl2"))
        _, self._prev_tray_tilt = self._tray_state()
        self._prev_vxy = self._base_vel()[3:5].copy()
        self._acc_f = np.zeros(2)
        self._prev_lean_obs = np.zeros(2)
        self._min_lidar = LIDAR_CUTOFF
        self._v_cmd = np.zeros(2)
        self._wz_cmd = 0.0
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
        # SLEW LIMIT the base command: the raw action is a *target*; the
        # actual command ramps toward it at <= ACC_CMD_MAX so a single policy
        # step can never demand a tip-inducing acceleration. The balance
        # reflex inside _apply_drive stays UNfiltered (the catch must be fast).
        v_tgt = np.array([action[0], action[1]]) * MAX_LIN_VEL
        dv = v_tgt - self._v_cmd
        dv_max = ACC_CMD_MAX * CTRL_DT
        dvn = np.linalg.norm(dv)
        if dvn > dv_max:
            dv *= dv_max / dvn
        self._v_cmd = self._v_cmd + dv
        dwz_max = YAW_ACC_CMD_MAX * CTRL_DT
        self._wz_cmd += float(np.clip(action[2] * MAX_ANG_VEL - self._wz_cmd,
                                      -dwz_max, dwz_max))
        # OPT-IN yaw-rate cap (default None = unchanged). Diagnosis (v31 failure
        # map): residual drops are front-half pillar/doorway weaving at ~0.9 m/s,
        # stack tilting 30deg -- hard turns whip the tall stack sideways. Capping
        # the APPLIED yaw (not MAX_ANG_VEL, which normalizes the obs) forces
        # gentler S-curves; the policy trains under it and adapts. Untested lever
        # per the handoff ("yaw whip untested as a drop cause").
        if self.yaw_cap is not None:
            self._wz_cmd = float(np.clip(self._wz_cmd, -self.yaw_cap, self.yaw_cap))
        # policy-driven arm: map the 2 extra action dims to shoulder servo
        # targets (held across the control period, like a real position servo).
        # _apply_gimbal won't touch these (its _apply_arm_catch is gated on
        # arm_catch, which is False in arm_action mode).
        if self.arm_action:
            ids = self._arm_setup()
            if ids is not None:
                self.data.ctrl[ids["lift_act"]] = (
                    ARM_POSE["shoulder_lift_joint"] + float(action[5]) * ARM_CATCH_CLIP)
                self.data.ctrl[ids["pan_act"]] = (
                    ARM_POSE["shoulder_pan_joint"] + float(action[6]) * ARM_CATCH_CLIP)
        # drive + gimbal stabilization run at full physics rate (like real
        # kHz motor/IMU servo loops); the policy command is held constant
        # across the control period
        for _ in range(FRAME_SKIP):
            self._apply_drive(self._v_cmd[0], self._v_cmd[1], self._wz_cmd)
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

        # smooth collision gradient: quadratic shortfall below safe clearance
        # (obs was just computed, so self._min_lidar is current-step)
        prox = max(0.0, (D_SAFE - self._min_lidar) / D_SAFE)

        # terrain-speed shaping: quadratic penalty for exceeding safe cruise
        # while astride the stair footprint (fixes the 12/30-drop stair
        # rushing the v6 autopsy found -- see W_STAIR_SPEED comment above)
        speed = float(np.hypot(*self._base_vel()[3:5]))
        on_stairs = self._stair_band[0] <= pos[0] <= self._stair_band[1]
        stair_overspeed = max(0.0, speed - SAFE_STAIR_SPEED) if on_stairs else 0.0
        # opt-in pillar-zone speed shaping (same quadratic form as stairs)
        in_pillars = self.pillar_speed_shape and PILLAR_BAND[0] <= pos[0] < PILLAR_BAND[1]
        pillar_overspeed = max(0.0, speed - SAFE_PILLAR_SPEED) if in_pillars else 0.0

        reward = (W_PROGRESS * progress - W_TILT_BOT * phi1**2
                  - W_TILT_TOP * phi2**2 - W_TRAY * tray_tilt**2
                  - W_JERK * jerk - W_PROX * prox**2
                  - W_STAIR_SPEED * stair_overspeed**2
                  - W_PILLAR_SPEED * pillar_overspeed**2 - TIME_PENALTY)
        terminated = False
        info = {"dist": dist, "phi1": phi1, "phi2": phi2, "tray_tilt": tray_tilt}
        # stall detection: terminal failure if best distance hasn't improved
        if dist < self._best_dist - STALL_MIN_PROGRESS:
            self._best_dist = dist
            self._steps_since_best = 0
        else:
            self._steps_since_best += 1
        fail = self._failure()
        if fail is None and self._steps_since_best >= STALL_WINDOW:
            fail = "stalled"
        if fail:
            reward -= P_STALL if fail == "stalled" else P_FAIL
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
    assert obs.shape == env.observation_space.shape, obs.shape
    total = 0.0
    for t in range(300):
        obs, r, term, trunc, info = env.step(env.action_space.sample() * 0.25)
        total += r
        if term or trunc:
            print(f"episode ended step {t}: {info}")
            obs, _ = env.reset()
    print(f"gimbal env smoke test OK, obs {obs.shape}, reward {total:.1f}")
