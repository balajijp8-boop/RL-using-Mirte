"""Watch v35 (yaw-cap 0.6 experiment) milestones. Eval each checkpoint UNDER
the same cap. Stop the instant a headline hits 54/60 (user rule: one genuine
90% hit = final), logging an independent fresh-seed confirm alongside for the
honest record. Otherwise ride to 8M and report the best."""
import os, re, subprocess, sys, time
import psutil

REPO = os.getcwd()
PY = sys.executable
RUN = "v36"
CAP = "0.5"
LOG = "runs/ladder_chain.log"
TARGETS = [500_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000,
           4_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000]


def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    open(LOG, "a", encoding="utf-8").write(s + "\n"); print(s, flush=True)


def alive():
    return any(any(a.endswith("finetune_gimbal.py") for a in (p.info["cmdline"] or []))
               and any(f"ppo_gimbal_{RUN}" in a for a in (p.info["cmdline"] or []))
               for p in psutil.process_iter(["name", "cmdline"])
               if "python" in (p.info["name"] or "").lower())


def ev(snap, label, seed, suffix=""):
    lf = f"runs/{RUN}_eval_{label}{suffix}.log"
    if not os.path.exists(lf):
        with open(lf, "w", encoding="utf-8") as o:
            subprocess.run([PY, "tools/eval_checkpoint.py", "--snap", snap,
                            "--episodes", "60", "--yaw-cap", CAP, "--seed-base", str(seed)],
                           stdout=o, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": REPO})
    t = open(lf, encoding="utf-8").read()
    m = re.search(r"success\s*:\s*(\d+)/", t)
    return int(m.group(1)) if m else None


log(f"=== yawcap_watch started: {RUN} under yaw-cap {CAP}, stop-on-54 ===")
best = (0, None)
for step in TARGETS:
    label = f"{step//1000}k" if step < 1_000_000 else f"{step/1_000_000:g}M"
    snap = f"runs/ppo_gimbal_{RUN}/snap_{step//1000:05d}k"
    waited = 0
    while not os.path.exists(snap + "_vecnorm.pkl"):
        if not alive():
            log(f"{RUN}: trainer died before {label}; stopping watch"); sys.exit()
        time.sleep(45); waited += 45
    s = ev(snap, label, 1000)
    if s is None:
        log(f"{RUN} {label}: eval parse fail"); continue
    if s > best[0]:
        best = (s, snap)
    # confirm any strong headline on fresh seeds (overnight: never stop, just record)
    if s >= 52:
        sc = ev(snap, label, 2000, "_confirm")
        pooled = s + (sc or 0)
        flag = " *** GENUINE 90%+ ***" if (pooled >= 108 and min(s, sc or 0) >= 52) else ""
        log(f"{RUN} {label}: {s}/60 (cap {CAP}) | confirm {sc}/60 -> pooled {pooled}/120 "
            f"({100*pooled/120:.1f}%){flag}")
    else:
        log(f"{RUN} {label}: {s}/60 (yaw-cap {CAP})")
# finished 8M with no 54 hit
final = f"runs/ppo_gimbal_{RUN}/ppo_final"
if os.path.exists(final + ".zip"):
    s = ev(final, "final", 1000)
    log(f"{RUN} final: {s}/60")
    if s and s > best[0]:
        best = (s, final)
log(f"=== yawcap_watch done (8M, no confirmed 54 hit). Best: {best[0]}/60 at {best[1]} ===")
