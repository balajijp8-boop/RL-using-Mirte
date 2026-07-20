"""Overnight autopilot for v29: wait for each milestone snapshot, run a 60-ep
eval, log a one-line summary, repeat through 8M/ppo_final. One long-running
process so no re-arming is needed between milestones. Writes incrementally to
runs/v29_overnight_arc.log so progress is visible even before it finishes."""
import os, re, subprocess, sys, time
import psutil

RUN = "runs/ppo_gimbal_v29"
ARC_LOG = "runs/v29_overnight_arc.log"
TARGETS = [2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000]
PY = sys.executable
POLL_S = 45
MAX_WAIT_S = 3600  # 1h/milestone ceiling; flags a stall without hanging forever


def log(line):
    ts = time.strftime("%H:%M:%S")
    with open(ARC_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {line}\n")
    print(f"[{ts}] {line}", flush=True)


def training_alive():
    for p in psutil.process_iter(["cmdline"]):
        c = " ".join(p.info["cmdline"] or [])
        if "finetune_gimbal.py" in c and "spawn_main" not in c:
            return True
    return False


def wait_for(path, label):
    waited = 0
    while not os.path.exists(path):
        if not training_alive():
            log(f"TRAINING DIED before {label} (missing {path})")
            return False
        if waited >= MAX_WAIT_S:
            log(f"STALL WARNING: {label} not seen after {MAX_WAIT_S}s, still alive, continuing to wait")
            waited = 0
        time.sleep(POLL_S)
        waited += POLL_S
    return True


def run_eval(snap_prefix, label):
    logfile = f"runs/v29_eval_{label}.log"
    with open(logfile, "w", encoding="utf-8") as out:
        subprocess.run(
            [PY, "tools/eval_checkpoint.py", "--snap", snap_prefix, "--episodes", "60"],
            stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": os.getcwd()})
    text = open(logfile, encoding="utf-8").read()
    succ = re.search(r"success\s*:\s*(\d+)/(\d+)", text)
    med = re.search(r"median\s*(-?[\d.]+)", text)
    fail = re.search(r"failure causes\s*:\s*(\{.*\})", text)
    if succ and med:
        log(f"{label}: {succ.group(1)}/{succ.group(2)} success, median {med.group(1)}, "
            f"{fail.group(1) if fail else '?'}")
    else:
        log(f"{label}: EVAL FAILED TO PARSE (see {logfile})")


log("overnight watch started")
for step in TARGETS:
    label = f"{step//1000}k" if step < 1_000_000 else f"{step//1_000_000}M"
    snap = f"{RUN}/snap_{step//1000:05d}k"
    if not wait_for(snap + "_vecnorm.pkl", label):
        log("overnight watch exiting: training not alive")
        break
    run_eval(snap, label)
else:
    # full 8M done -> also eval ppo_final (post-training-complete save)
    if wait_for(f"{RUN}/ppo_final.zip", "ppo_final"):
        run_eval(f"{RUN}/ppo_final", "8M_final")
    log("overnight watch complete: v29 finished 8M")
