#!/usr/bin/env python3
"""
Build a real-mecanum MIRTE scene from the native MJCF.

Takes mirte_model/mirte_master.xml (the URDF-converted robot) and:
  - gives base_link a <freejoint> so it drives on the floor via wheel contact
  - replaces each wheel's plain cylinder collision with N angled rollers
    (passive hinges at +/-45 deg) -> genuine holonomic mecanum slip
  - adds velocity actuators on the 4 wheel drive joints
  - locks the arm in a horizontal "tray" pose (position actuators holding it)
  - adds a floor, light, and a skybox-free ground

Writes mirte_model/scene_mecanum.xml. Run test_strafe.py to verify motion.
"""
import numpy as np
import lxml.etree as ET

SRC = "/home/balaji/mirte_balance_rl/mirte_model/mirte_master.xml"
DST = "/home/balaji/mirte_balance_rl/mirte_model/scene_mecanum.xml"

WHEEL_R = 0.05          # rim radius from URDF
ROLLER_R = 0.011        # roller (barrel) radius
N_ROLLERS = 14          # rollers per wheel (more -> smoother, less sink)
HALF_LEN = 0.017        # roller half length

# roller spin axis is +45 or -45 deg relative to the wheel axle, depending on
# the wheel's roller orientation. On a mecanum drive the two diagonal pairs are
# mirror images. From the URDF: FL & RR use "left" mesh, RL & FR use "right".
WHEEL_ROLLER_SIGN = {
    "front_left_wheel": +1,
    "rear_right_wheel": +1,
    "rear_left_wheel": -1,
    "front_right_wheel": -1,
}


def roller_bodies(sign):
    """XML string of N roller child-bodies around a wheel hub.

    Wheel local frame: spin axis = local z (from joint axis 0 0 -1),
    rim lies in local x-y plane. A roller at rim angle theta sits at
    p = (R-ROLLER_R)*(cos, sin, 0); its free-spin axis is the rim tangent
    rotated 45deg toward the axle:  a = cos45*tangent + sign*sin45*z.
    """
    out = []
    Rc = WHEEL_R - ROLLER_R
    c45 = np.cos(np.pi / 4)
    s45 = np.sin(np.pi / 4)
    for i in range(N_ROLLERS):
        th = 2 * np.pi * i / N_ROLLERS
        px, py = Rc * np.cos(th), Rc * np.sin(th)
        tangent = np.array([-np.sin(th), np.cos(th), 0.0])
        axis = c45 * tangent + sign * s45 * np.array([0, 0, 1.0])
        axis /= np.linalg.norm(axis)
        # capsule endpoints along the roller axis
        e1 = np.array([px, py, 0]) - HALF_LEN * axis
        e2 = np.array([px, py, 0]) + HALF_LEN * axis
        out.append(
            f'''<body name="{{wheel}}_roller{i}" pos="{px:.5f} {py:.5f} 0">
              <joint name="{{wheel}}_roller{i}_j" type="hinge" axis="{axis[0]:.5f} {axis[1]:.5f} {axis[2]:.5f}" damping="0.0002"/>
              <geom name="{{wheel}}_roller{i}_g" type="capsule" size="{ROLLER_R}"
                    fromto="{e1[0]:.5f} {e1[1]:.5f} {e1[2]:.5f} {e2[0]:.5f} {e2[1]:.5f} {e2[2]:.5f}"
                    friction="1.2 0.01 0.001" rgba="0.15 0.15 0.15 1" mass="0.03"/>
            </body>''')
    return "\n".join(out)


def main():
    tree = ET.parse(SRC)
    root = tree.getroot()

    # ---- compiler / options -------------------------------------------------
    comp = root.find("compiler")
    comp.set("meshdir", "./")
    comp.set("angle", "radian")

    opt = ET.SubElement(root, "option")
    opt.set("timestep", "0.002")
    opt.set("integrator", "implicitfast")

    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global").set("offwidth", "1280")
    vis[-1].set("offheight", "960")

    # ---- floor + light ------------------------------------------------------
    wb = root.find("worldbody")
    light = ET.SubElement(wb, "light")
    light.set("pos", "0 0 3"); light.set("dir", "0 0 -1"); light.set("diffuse", "0.9 0.9 0.9")
    floor = ET.SubElement(wb, "geom")
    floor.set("name", "floor"); floor.set("type", "plane"); floor.set("size", "5 5 0.1")
    floor.set("friction", "1.2 0.01 0.001"); floor.set("rgba", "0.8 0.8 0.82 1")

    # ---- base_link: add freejoint ------------------------------------------
    base = wb.find(".//body[@name='base_link']")
    fj = ET.Element("freejoint"); fj.set("name", "base_free")
    base.insert(0, fj)

    # ---- wheels: strip plain cylinder collision, add rollers ---------------
    for wheel_name, sign in WHEEL_ROLLER_SIGN.items():
        wbody = wb.find(f".//body[@name='{wheel_name}']")
        # remove the plain cylinder collision geom (the one without group=1)
        for g in list(wbody.findall("geom")):
            if g.get("type") == "cylinder" and g.get("group") != "1":
                wbody.remove(g)
        # make the visual wheel mesh non-colliding (already contype=0) - keep it
        rollers_xml = roller_bodies(sign).replace("{wheel}", wheel_name)
        frag = ET.fromstring(f"<root>{rollers_xml}</root>")
        for child in frag:
            wbody.append(child)

    # ---- actuators: 4 wheels (velocity) + arm hold (position) --------------
    act = ET.SubElement(root, "actuator")
    for w in ("front_left_wheel_joint", "rear_left_wheel_joint",
              "front_right_wheel_joint", "rear_right_wheel_joint"):
        m = ET.SubElement(act, "velocity")
        m.set("name", "drive_" + w.replace("_wheel_joint", ""))
        m.set("joint", w); m.set("kv", "0.5"); m.set("ctrlrange", "-30 30")
        m.set("forcerange", "-3 3")

    # lock arm joints in a tray pose with stiff position servos
    # tray pose: pan=0, lift raises arm, elbow bends so wrist is horizontal
    TRAY_POSE = {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -1.2,
        "elbow_joint": 0.0,
        "wrist_joint": 1.2,
        "gripper_joint": 0.15,
    }
    for j, q in TRAY_POSE.items():
        m = ET.SubElement(act, "position")
        m.set("name", "hold_" + j.replace("_joint", ""))
        m.set("joint", j); m.set("kp", "20"); m.set("ctrlrange", "-3.14 3.14")

    # keyframe with base lifted so wheels settle, arm at tray pose
    key = ET.SubElement(root, "keyframe")
    k = ET.SubElement(key, "key"); k.set("name", "start")

    tree.write(DST, pretty_print=True)
    print("wrote", DST)

    # quick compile check
    import mujoco
    m = mujoco.MjModel.from_xml_path(DST)
    print(f"COMPILES: {m.nbody} bodies, {m.njnt} joints, {m.nu} actuators, {m.ngeom} geoms")


if __name__ == "__main__":
    main()
