"""Film v3 — cameras only, no post-FX, NO MUSIC, NO dashes in titles.
Arc: title(MIRTE hero) -> problem -> approach -> 6 real failures (each shows the
actual fall) -> mastery montage (wide/drone through the stairs so sway != luck)
-> DELIVERED -> stats. Labels are only what's truthfully on screen."""
import os, shutil, subprocess
import imageio_ffmpeg
import matplotlib

FF = imageio_ffmpeg.get_ffmpeg_exe()
C = os.path.join(os.path.dirname(__file__), "cine2")
TMP = os.path.join(C, "seg3"); os.makedirs(TMP, exist_ok=True)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mirte_rl_film_4k.mp4"))
W, H, FPS = 3840, 2160, 50
S = H / 720.0

fdir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
shutil.copy(os.path.join(fdir, "DejaVuSans-Bold.ttf"), os.path.join(C, "bold.ttf"))
shutil.copy(os.path.join(fdir, "DejaVuSans.ttf"), os.path.join(C, "reg.ttf"))
B, R = "bold.ttf", "reg.ttf"


def run(a): subprocess.run([FF, "-y", *a], check=True, cwd=C,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def esc(t): return t.replace("'", "’").replace(":", r"\:")


def dtext(text, size, off, font, color, shadow=False):
    y = f"h/2{'+' if off >= 0 else '-'}{abs(off)*S:.0f}"
    sh = ":shadowcolor=black@0.85:shadowx=4:shadowy=4" if shadow else ""
    return (f"drawtext=fontfile={font}:text='{esc(text)}':expansion=none:"
            f"fontcolor={color}:fontsize={size*S:.0f}:x=(w-text_w)/2:y={y}{sh}")


def card(path, dur, lines, fade=0.5, bg="0x0a0c10"):
    vf = ",".join(dtext(*ln) for ln in lines)
    vf += f",fade=in:0:{int(fade*FPS)},fade=out:{int((dur-fade)*FPS)}:{int(fade*FPS)}"
    run(["-f", "lavfi", "-i", f"color=c={bg}:s={W}x{H}:d={dur}:r={FPS}",
         "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", path])


def titlecard(path, img, dur, lines, fade=0.6):
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
          + ",".join(dtext(*ln) for ln in lines)
          + f",fade=in:0:{int(fade*FPS)},fade=out:{int((dur-fade)*FPS)}:{int(fade*FPS)}")
    run(["-loop", "1", "-t", str(dur), "-i", img, "-vf", vf, "-r", str(FPS),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", path])


def clip(path, src, ss, to, label=None, lcolor="white"):
    dur = to - ss
    if label:
        vf = (f"fps={FPS},drawtext=fontfile={B}:text='{esc(label)}':expansion=none:"
              f"fontcolor={lcolor}:fontsize={34*S:.0f}:x=(w-text_w)/2:y=h-{115*S:.0f}:"
              f"shadowcolor=black@0.85:shadowx=4:shadowy=4:"
              f"alpha='min(1,min((t-0.1)/0.35,({dur}-t)/0.35))'")
    else:
        vf = f"fps={FPS}"
    run(["-ss", str(ss), "-to", str(to), "-i", src, "-vf", vf, "-an", "-r", str(FPS),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", path])


def dur_of(path):
    r = subprocess.run([FF, "-i", path], capture_output=True, text=True, cwd=C)
    for ln in r.stderr.splitlines():
        if "Duration" in ln:
            h, m, s = ln.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0


segs = []
def add(p): segs.append(p)

# ---- ACT 1 ----
titlecard(f"{TMP}/t.mp4", f"{C}/hero.png", 3.6, [
    ("M I R T E", 78, 150, B, "white", True),
    ("a robot learning to keep its balance", 30, 215, R, "0xd8e6ff", True)])
add(f"{TMP}/t.mp4")
card(f"{TMP}/c2.mp4", 4.0, [
    ("THE PROBLEM", 30, -120, B, "0xf0842e"),
    ("Carry a tall stacked payload across a", 38, -40, R, "white"),
    ("field of pillars, a doorway, and stairs", 38, 12, R, "white"),
    ("without ever dropping it.", 38, 64, R, "0xbfc4cc")])
add(f"{TMP}/c2.mp4")
card(f"{TMP}/c3.mp4", 4.0, [
    ("THE APPROACH", 30, -120, B, "0xf0842e"),
    ("No rules. No instructions.", 38, -40, R, "white"),
    ("Only trial, error, and a reward.", 38, 12, R, "white"),
    ("Reinforcement learning.", 40, 66, B, "0x8ab4ff")])
add(f"{TMP}/c3.mp4")

# ---- ACT 2: it learns by failing (6 real failures, each shows the fall) ----
card(f"{TMP}/lc.mp4", 2.4, [("IT LEARNS BY FAILING", 44, -24, B, "white")])
add(f"{TMP}/lc.mp4")
fails = [
    ("fail0_start",     "no idea how to balance"),
    ("failX_pillarhit", "it hits a pillar"),
    ("fail1_pillars",   "it drops the load"),
    ("fail2_crash",     "it hits a wall"),
    ("fail3_doorway",   "the stack tips over"),
    ("fail4_stairs",    "it topples on the stairs"),
]
for i, (name, lab) in enumerate(fails):
    src = f"{C}/{name}.mp4"
    if not os.path.exists(src):
        print("  MISSING", name); continue
    d = dur_of(src)
    cp = f"{TMP}/f{i}.mp4"; clip(cp, src, 0.05, d - 0.05, lab); add(cp)

# ---- ACT 3: mastery — WIDE ONLY. Close shots on the final run make the stack's
# micro-sway read as luck, so match-cut between two wide, occlusion-free aerials of
# the SAME run (front-quarter 'dronef' + rear top-down 'drone'). Stack stands tall. ----
card(f"{TMP}/mc.mp4", 2.4, [("MILLIONS OF ATTEMPTS LATER", 40, -24, B, "0x79e08c")])
add(f"{TMP}/mc.mp4")
M = lambda a: f"{C}/master_{a}.mp4"
mcuts = [("dronef",0.5,3.2),   # pillars      (front-quarter, stack standing tall)
         ("drone", 3.2,5.4),   # doorway      (rear top-down)
         ("dronef",5.4,7.6),   # stairs climb (front-quarter, WIDE)
         ("drone", 7.6,9.3),   # stairs crest (rear top-down, WIDE)
         ("dronef",9.3,10.6)]  # goal run-in  (front-quarter)
for i,(a,s,e) in enumerate(mcuts):
    if os.path.exists(M(a)):
        cp=f"{TMP}/m{i}.mp4"; clip(cp, M(a), s, e); add(cp)
# goal: wide front-quarter, DELIVERED stamp
run(["-ss","10.6","-to","12.2","-i",M("dronef"),"-vf",
     f"fps={FPS},drawtext=fontfile={B}:text='DELIVERED':expansion=none:fontcolor=0x79e08c:"
     f"fontsize={56*S:.0f}:x=(w-text_w)/2:y=h-{150*S:.0f}:shadowcolor=black@0.85:"
     f"shadowx=4:shadowy=4:alpha='min(1,t/0.4)'","-an","-r",str(FPS),
     "-c:v","libx264","-pix_fmt","yuv420p",f"{TMP}/goal.mp4"])
add(f"{TMP}/goal.mp4")

# ---- ACT 4: generalization proof — same policy, 4 fresh random courses, all
# delivered, straight top-down so the different layouts are obvious (no memorizing) ----
card(f"{TMP}/gc.mp4", 2.8, [
    ("IT DIDN'T MEMORIZE THE COURSE", 42, -30, B, "white"),
    ("every run is a brand new random layout", 30, 34, R, "0x79e08c")])
add(f"{TMP}/gc.mp4")
G = f"{C}/generalize_2x2.mp4"
if os.path.exists(G):
    cp = f"{TMP}/gen.mp4"
    run(["-i", G, "-vf",
         f"setpts=0.62*PTS,fps={FPS},"
         f"drawtext=fontfile={B}:text='4 random courses      one policy      all delivered':"
         f"expansion=none:fontcolor=white:fontsize={30*S:.0f}:x=(w-text_w)/2:y=h-{64*S:.0f}:"
         f"shadowcolor=black@0.85:shadowx=4:shadowy=4",
         "-an", "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", cp])
    add(cp)
else:
    print("  (generalize_2x2.mp4 missing — skipping proof section)")

# ---- end card (no dashes) ----
card(f"{TMP}/ec.mp4", 4.8, [
    ("0%   to   90%", 84, -95, B, "0x79e08c"),
    ("delivery success, learned from scratch", 32, 5, R, "white"),
    ("PPO  ·  MuJoCo  ·  trained in simulation", 26, 52, R, "0x8a8a97"),
    ("linkedin.com/in/balajijp     github.com/balajijp8-boop", 26, 128, R, "0x8ab4ff")])
add(f"{TMP}/ec.mp4")

# ---- concat (SILENT, no music) ----
with open(f"{TMP}/list.txt", "w") as f:
    for p in segs:
        f.write(f"file '{os.path.abspath(p)}'\n")
run(["-f", "concat", "-safe", "0", "-i", f"{TMP}/list.txt",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
     "-movflags", "+faststart", OUT])
print("WROTE", OUT)
r = subprocess.run([FF, "-i", OUT], capture_output=True, text=True)
for ln in r.stderr.splitlines():
    if "Duration" in ln or "Video:" in ln:
        print(ln.strip())
