"""Assemble the cinematic 'v1 -> delivery' film from rendered angle clips.
Structure: Problem -> Approach -> it fails -> it masters -> reaches the goal.
Cinematic grade + 2.35 scope letterbox + title cards + angle match-cuts +
slow-mo delivery beat. Resolution-aware (fonts/offsets scale from a 720 design)."""
import os, shutil, subprocess
import imageio_ffmpeg
import matplotlib

FF = imageio_ffmpeg.get_ffmpeg_exe()
CINE = os.path.join(os.path.dirname(__file__), "cine")
TMP = os.path.join(CINE, "seg"); os.makedirs(TMP, exist_ok=True)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mirte_rl_film_4k.mp4"))
W, H, FPS = 3840, 2160, 50
S = H / 720.0                      # design scale (all sizes/offsets tuned at 720)

fdir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
shutil.copy(os.path.join(fdir, "DejaVuSans-Bold.ttf"), os.path.join(CINE, "bold.ttf"))
shutil.copy(os.path.join(fdir, "DejaVuSans.ttf"), os.path.join(CINE, "reg.ttf"))

GRADE = ("eq=contrast=1.15:brightness=-0.02:saturation=1.2,"
         f"unsharp=5:5:0.5,vignette=PI/4.5,"
         f"crop=in_w:in_w/2.35,pad={W}:{H}:0:({H}-oh)/2:black")


def run(args):
    subprocess.run([FF, "-y", *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def esc(t):
    return t.replace("'", "’").replace(":", r"\:")


def card(path, dur, lines, fade=0.5):
    """lines = list of (text, size720, yoff720_from_center, font, color)."""
    dt = []
    for text, size, off, font, color in lines:
        y = f"h/2{'+' if off >= 0 else '-'}{abs(off)*S:.0f}"
        dt.append(f"drawtext=fontfile={font}.ttf:text='{esc(text)}':expansion=none:"
                  f"fontcolor={color}:fontsize={size*S:.0f}:x=(w-text_w)/2:y={y}:"
                  f"alpha='min(1,min(t/{fade},({dur}-t)/{fade}))'")
    vf = ",".join(dt) + f",fade=in:0:{int(fade*FPS)},fade=out:{int((dur-fade)*FPS)}:{int(fade*FPS)}"
    run(["-f", "lavfi", "-i", f"color=c=0x0a0c10:s={W}x{H}:d={dur}:r={FPS}",
         "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", path])


def seg(path, src, ss, to, speed=1.0, extra=""):
    vf = GRADE
    if speed != 1.0:
        vf = f"setpts={1/speed}*PTS," + vf
    if extra:
        vf += "," + extra
    run(["-ss", str(ss), "-to", str(to), "-i", src, "-vf", vf, "-an",
         "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", path])


D = lambda a: os.path.join(CINE, f"deliver_{a}.mp4")
segments = []
def add(p): segments.append(p)

# ---- ACT 1: problem + approach ----
card(f"{TMP}/c1.mp4", 3.0, [
    ("M I R T E", 74, -90, "bold", "white"),
    ("teaching a robot to keep its balance", 30, 20, "reg", "0x8ab4ff")])
add(f"{TMP}/c1.mp4")
card(f"{TMP}/c2.mp4", 4.2, [
    ("THE PROBLEM", 30, -120, "bold", "0xf0842e"),
    ("Carry a tall stacked payload across a", 38, -40, "reg", "white"),
    ("field of pillars, a doorway, and stairs", 38, 12, "reg", "white"),
    ("without ever dropping it.", 38, 64, "reg", "0xbfc4cc")])
add(f"{TMP}/c2.mp4")
card(f"{TMP}/c3.mp4", 4.2, [
    ("THE APPROACH", 30, -120, "bold", "0xf0842e"),
    ("No hand-written rules. No instructions.", 38, -40, "reg", "white"),
    ("Only trial, error, and a reward signal.", 38, 12, "reg", "white"),
    ("Reinforcement learning.", 40, 66, "bold", "0x8ab4ff")])
add(f"{TMP}/c3.mp4")

# ---- ACT 2: it fails ----
card(f"{TMP}/c4.mp4", 2.4, [("IT BEGINS BY FAILING", 44, -24, "bold", "white")])
add(f"{TMP}/c4.mp4")
seg(f"{TMP}/f0.mp4", f"{CINE}/flail_0.mp4", 0, 0.74, speed=0.6); add(f"{TMP}/f0.mp4")
seg(f"{TMP}/f1.mp4", f"{CINE}/flail_1.mp4", 0, 0.66, speed=0.6); add(f"{TMP}/f1.mp4")

# ---- ACT 3: mastery + goal (match-cut montage of ONE run) ----
card(f"{TMP}/c5.mp4", 2.4, [("MILLIONS OF ATTEMPTS LATER", 40, -24, "bold", "0x79e08c")])
add(f"{TMP}/c5.mp4")
cuts = [("chase",0.2,2.3),("high",2.3,4.5),("front",4.5,6.3),
        ("chase",6.3,8.3),("high",8.3,10.2),("front",10.2,11.3)]
for i,(a,s,e) in enumerate(cuts):
    seg(f"{TMP}/m{i}.mp4", D(a), s, e); add(f"{TMP}/m{i}.mp4")
seg(f"{TMP}/goal.mp4", D("high"), 11.3, 12.16, speed=0.4,
    extra=f"drawtext=fontfile=bold.ttf:text='DELIVERED':expansion=none:fontcolor=0x79e08c:"
          f"fontsize={54*S:.0f}:x=(w-text_w)/2:y=h-{150*S:.0f}:alpha='min(1,t/0.4)'")
add(f"{TMP}/goal.mp4")

# ---- end card ----
card(f"{TMP}/c6.mp4", 4.8, [
    ("0%  ->  90%", 88, -95, "bold", "0x79e08c"),
    ("delivery success, learned from scratch", 32, 5, "reg", "white"),
    ("PPO  ·  MuJoCo  ·  trained in simulation", 26, 52, "reg", "0x8a8a97"),
    ("linkedin.com/in/balajijp   ·   github.com/balajijp8-boop", 26, 128, "reg", "0x8ab4ff")])
add(f"{TMP}/c6.mp4")

# ---- concat ----
with open(f"{TMP}/list.txt", "w") as f:
    for p in segments:
        f.write(f"file '{os.path.abspath(p)}'\n")
run(["-f", "concat", "-safe", "0", "-i", f"{TMP}/list.txt",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
     "-movflags", "+faststart", OUT])
print("WROTE", OUT)
r = subprocess.run([FF, "-i", OUT], capture_output=True, text=True)
for ln in r.stderr.splitlines():
    if "Duration" in ln or ("Stream" in ln and "Video" in ln):
        print(ln.strip())
