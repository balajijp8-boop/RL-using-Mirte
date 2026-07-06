#!/usr/bin/env python3
"""
Build the real-MIRTE task scene with an ABSTRACT holonomic base.

  - base_link driven by slide_x / slide_y / yaw planar joints (velocity actuators)
  - real arm frozen in a horizontal "tray" pose (baked into a keyframe, held stiff)
  - a thin tray plate rigidly attached at the gripper so cylinders have a flat
    surface to stack on
  - two free stacked cylinders
  - floor + light

Usage:
  python build_tray_scene.py --lift -0.55 --elbow 1.15 --wrist 0.0 --render
Iterate the arm angles until the tray reads horizontal in the side view.
"""
import argparse
import numpy as np
import lxml.etree as ET

SRC = "/home/balaji/mirte_balance_rl/mirte_model/mirte_master.xml"
DST = "/home/balaji/mirte_balance_rl/mirte_model/scene_tray.xml"

MAX_LIN, MAX_ANG = 0.8, 1.2


def build(pan, lift, elbow, wrist, tray_pos, tray_quat):
    tree = ET.parse(SRC)
    root = tree.getroot()
    root.find("compiler").set("meshdir", "./")

    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="1280", offheight="960")

    wb = root.find("worldbody")
    ET.SubElement(wb, "light", pos="0 0 3", dir="0 0 -1", diffuse="0.9 0.9 0.9")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="5 5 0.1",
                  rgba="0.8 0.8 0.82 1")

    # planar base joints on base_link (world-frame slides + yaw)
    base = wb.find(".//body[@name='base_link']")
    for i, (nm, ax) in enumerate([("slide_x", "1 0 0"), ("slide_y", "0 1 0")]):
        j = ET.Element("joint", name=nm, type="slide", axis=ax, damping="2")
        base.insert(i, j)
    base.insert(2, ET.Element("joint", name="yaw", type="hinge", axis="0 0 1", damping="1"))
    # base needs to rest at the right height: lift base_link so wheels ~touch floor
    base.set("pos", "0 0 0.055")

    # make wheels visual-only (no contact) since base is abstract
    for wn in ("front_left_wheel", "rear_left_wheel", "front_right_wheel", "rear_right_wheel"):
        wbody = wb.find(f".//body[@name='{wn}']")
        for g in wbody.findall("geom"):
            g.set("contype", "0"); g.set("conaffinity", "0")

    # the posed arm is locked decoration only -> make every arm link collision-free
    # so it never kicks the cylinders (the tray/blades on base_link do the physics)
    for an in ("shoulder_pan", "shoulder_lift", "elbow", "wrist", "gripper",
               "_gripper_link_r", "gripper_finger_r", "_gripper_link_l",
               "gripper_finger_l", "_Gripper_r"):
        abody = wb.find(f".//body[@name='{an}']")
        for g in abody.findall("geom"):
            g.set("contype", "0"); g.set("conaffinity", "0")

    # LOCK the arm rigidly at the tray pose with stiff passive springs (not limited
    # by the real ±2 N·m actuators). This makes the gripper a rigid transport tray.
    arm_pose = {"shoulder_pan_joint": pan, "shoulder_lift_joint": lift,
                "elbow_joint": elbow, "wrist_joint": wrist, "gripper_joint": 0.2}
    for jn, q in arm_pose.items():
        jel = wb.find(f".//joint[@name='{jn}']")
        jel.set("stiffness", "1500")
        jel.set("springref", f"{q}")
        jel.set("damping", "30")
        if jel.get("actuatorfrcrange"):
            del jel.attrib["actuatorfrcrange"]   # let the passive spring hold firm

    # frame_link keeps its box collisions but shouldn't fall through: give base a
    # support geom is unnecessary (planar joints hold z fixed). Freeze z via no z joint.

    # tray plate attached to the gripper body. The gripper's local y-axis is
    # vertical at the tray pose, so the plate is thin in local y (normal up) and
    # offset along -y to sit as a shelf just under the payload.
    # Tray + cradle rim are attached to base_link (perfectly level, yaw-only) at
    # the posed gripper's location. The locked arm visually cradles it; physics
    # stays level so the stack is stable at rest and only topples under motion.
    # base_link sits at z=0.055; gripper world ~ (0.387, 0.012, 0.326).
    GX, GY, GZ = 0.387, 0.012, 0.326 - 0.055
    ET.SubElement(base, "geom", name="tray", type="box", size="0.055 0.055 0.004",
                  pos=f"{GX} {GY} {GZ}", rgba="0.2 0.2 0.25 1",
                  friction="1.4 0.02 0.002")
    # two blades matching the proven box-env cradle: 6 cm tall, 9.6 cm apart,
    # snug around the 6 cm bottom cylinder (the design that balanced for 12 s)
    for nm, dy in (("blade_l", +0.048), ("blade_r", -0.048)):
        ET.SubElement(base, "geom", name=nm, type="box", size="0.06 0.006 0.03",
                      pos=f"{GX} {GY+dy} {GZ+0.03}", rgba="0.25 0.25 0.3 1",
                      friction="1.4 0.02 0.002")

    # two stacked cylinders rest on top of the tray (gripper world ~0.387,0.012,0.326)
    cx, cy, cz = 0.387, 0.012, 0.425
    for i, dz in enumerate((0.0, 0.185)):
        cyl = ET.SubElement(wb, "body", name=f"cyl{i+1}", pos=f"{cx} {cy} {cz+dz}")
        ET.SubElement(cyl, "freejoint")
        ET.SubElement(cyl, "geom", name=f"cyl{i+1}_g", type="cylinder",
                      size="0.03 0.09", density="400",
                      rgba=("0.9 0.6 0.1 1" if i == 0 else "0.9 0.2 0.1 1"))

    # actuators: base velocity + arm position hold
    act = ET.SubElement(root, "actuator")
    ET.SubElement(act, "velocity", name="act_x", joint="slide_x", kv="120",
                  ctrlrange=f"-{MAX_LIN} {MAX_LIN}", forcerange="-80 80")
    ET.SubElement(act, "velocity", name="act_y", joint="slide_y", kv="120",
                  ctrlrange=f"-{MAX_LIN} {MAX_LIN}", forcerange="-80 80")
    ET.SubElement(act, "velocity", name="act_w", joint="yaw", kv="12",
                  ctrlrange=f"-{MAX_ANG} {MAX_ANG}", forcerange="-20 20")
    for j, q in [("shoulder_pan", pan), ("shoulder_lift", lift),
                 ("elbow", elbow), ("wrist", wrist), ("gripper", 0.2)]:
        ET.SubElement(act, "position", name="hold_" + j, joint=j + "_joint",
                      kp="30", ctrlrange="-3.14 3.14")

    # keyframe holding the arm pose
    key = ET.SubElement(root, "keyframe")
    # qpos layout: slide_x, slide_y, yaw, [wheels x4], pan, lift, elbow, wrist,
    # gripper, gripper_link_r, gripper_link_l, _Gripper_r, then 2 free cyls (7 each)
    ET.SubElement(key, "key", name="tray")

    tree.write(DST, pretty_print=True)
    return DST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pan", type=float, default=0.0)
    ap.add_argument("--lift", type=float, default=-0.55)
    ap.add_argument("--elbow", type=float, default=1.15)
    ap.add_argument("--wrist", type=float, default=0.0)
    ap.add_argument("--tx", type=float, default=0.0)
    ap.add_argument("--ty", type=float, default=0.0)
    ap.add_argument("--tz", type=float, default=-0.03)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()

    path = build(a.pan, a.lift, a.elbow, a.wrist,
                 (a.tx, a.ty, a.tz), (1, 0, 0, 0))
    import mujoco
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)

    def jadr(n): return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def aid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    pose = {"shoulder_pan_joint": a.pan, "shoulder_lift_joint": a.lift,
            "elbow_joint": a.elbow, "wrist_joint": a.wrist, "gripper_joint": 0.2}
    for j, q in pose.items():
        d.qpos[jadr(j)] = q
    for j, q in [("shoulder_pan", a.pan), ("shoulder_lift", a.lift),
                 ("elbow", a.elbow), ("wrist", a.wrist), ("gripper", 0.2)]:
        d.ctrl[aid("hold_" + j)] = q
    mujoco.mj_forward(m, d)
    print(f"compiled: {m.nbody} bodies, {m.nu} act")
    # report gripper/tray world orientation
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    print("gripper world pos:", np.round(d.xpos[gid], 3))

    if a.render:
        from PIL import Image
        for _ in range(300):
            mujoco.mj_step(m, d)
        r = mujoco.Renderer(m, height=480, width=640)
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
        cam.distance = 1.0; cam.lookat[:] = [0.15, 0, 0.25]
        tiles = []
        for az, el in [(0, -10), (90, -20)]:
            cam.azimuth = az; cam.elevation = el
            r.update_scene(d, cam); tiles.append(r.render())
        Image.fromarray(np.hstack(tiles)).save("/home/balaji/mirte_balance_rl/tray_pose.png")
        print("saved tray_pose.png (side | front)")


if __name__ == "__main__":
    main()
