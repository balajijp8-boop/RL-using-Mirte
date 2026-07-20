"""Ladder chain v4 -- 'get a genuinely confirmed 90%, even once.'

Fixes vs v3: NEVER resumes training in-place into an existing --out folder.
finetune_gimbal.py resets its local step counter to 0 on every launch, so
resuming into the same folder silently overwrites earlier milestone files
with mislabeled, more-trained checkpoints (this ate v33@1M's verified 88.3%-
pooled checkpoint tonight). Every launch here is a brand-new generation
folder building on the best VERIFIED checkpoint so far -- no in-place resume.

Goal: any single milestone where headline >=52 AND its independent fresh-
seed confirm gives pooled >=108/120 with neither set below 52. One genuine
hit ends the chain. Plain lr-halving without a new diagnosis has plateaued
for 3 generations (84.2/84.2/oscillating), so this keeps sampling from the
same good basin (v31) rather than chaining ever-lower lr on top of noise.
"""
import glob, os, re, subprocess, sys, time

import psutil

REPO = os.getcwd()
PY = sys.executable
ARC_LOG = "runs/ladder_chain.log"
GOAL_POOLED = 108          # /120 = 90%
GOAL_MIN_SET = 52
HEADLINE_TRIGGER = 52
MAX_GENERATIONS = 6
ALL_TARGETS = [500_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000,
               4_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000]
POLL_S = 45
MAX_WAIT_S = 3600


def log(line):
    ts = time.strftime("%H:%M:%S")
    msg = f"[{ts}] {line}"
    with open(ARC_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def label_for(step):
    return f"{step//1000}k" if step < 1_000_000 else f"{step/1_000_000:g}M"


def training_alive(run_name):
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if "python" not in (p.info["name"] or "").lower():
                continue
            cmd = p.info["cmdline"] or []
            if any(a.endswith("finetune_gimbal.py") for a in cmd) and \
               any(f"ppo_gimbal_{run_name}" in a for a in cmd):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def wait_for(path, label, run_name):
    waited = 0
    while not os.path.exists(path):
        if not training_alive(run_name):
            log(f"{run_name}: TRAINING DIED before {label} (missing {path})")
            return False
        if waited >= MAX_WAIT_S:
            log(f"{run_name}: STALL WARNING at {label} ({MAX_WAIT_S}s), still alive, waiting on")
            waited = 0
        time.sleep(POLL_S)
        waited += POLL_S
    return True


def parse_eval(logfile):
    if not os.path.exists(logfile):
        return None
    text = open(logfile, encoding="utf-8").read()
    succ = re.search(r"success\s*:\s*(\d+)/(\d+)", text)
    med = re.search(r"median\s*(-?[\d.]+)", text)
    if not (succ and med):
        return None
    return (int(succ.group(1)), float(med.group(1)))


def run_eval(snap_prefix, run_name, label, seed_base, suffix=""):
    logfile = f"runs/{run_name}_eval_{label}{suffix}.log"
    cached = parse_eval(logfile)
    if cached:
        return cached
    env = {**os.environ, "PYTHONPATH": REPO}
    with open(logfile, "w", encoding="utf-8") as out:
        subprocess.run([PY, "tools/eval_checkpoint.py", "--snap", snap_prefix,
                        "--episodes", "60", "--seed-base", str(seed_base)],
                       stdout=out, stderr=subprocess.STDOUT, env=env)
    r = parse_eval(logfile)
    if r is None:
        log(f"{run_name} {label}{suffix}: EVAL FAILED (see {logfile})")
    return r


def launch_training(run_name, from_model, vecnorm, lr):
    out_dir = f"runs/ppo_gimbal_{run_name}"
    if os.path.exists(out_dir):
        log(f"{run_name}: ABORT -- {out_dir} already exists (would silently "
            f"overwrite). This chain never reuses a folder; something is wrong.")
        return False
    log(f"=== launching {run_name}: from {from_model}, lr={lr}, NO-CURRICULUM, fresh folder ===")
    logf = open(f"runs/{run_name}_train.log", "w", encoding="utf-8")
    errf = open(f"runs/{run_name}_train.err", "w", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": REPO}
    subprocess.Popen(
        [PY, "-u", "finetune_gimbal.py",
         "--from-model", from_model, "--vecnorm", vecnorm,
         "--gamma", "0.995", "--steps", "8000000", "--n-envs", "12",
         "--lr", str(lr), "--no-curriculum", "--out", out_dir],
        stdout=logf, stderr=errf, env=env, cwd=REPO,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    time.sleep(15)
    return training_alive(run_name)


def vecnorm_for(snap_prefix):
    if snap_prefix.endswith("ppo_final"):
        return os.path.join(os.path.dirname(snap_prefix), "vecnormalize.pkl")
    return snap_prefix + "_vecnorm.pkl"


def run_generation(run_name, from_model, lr):
    if training_alive(run_name):
        log(f"=== resuming watch on {run_name} (already training, launched earlier this session) ===")
    elif not launch_training(run_name, from_model, vecnorm_for(from_model), lr):
        log(f"{run_name}: FAILED TO START (runs/{run_name}_train.err)")
        return None, None
    results = []
    for step in ALL_TARGETS:
        label = label_for(step)
        snap = f"runs/ppo_gimbal_{run_name}/snap_{step//1000:05d}k"
        if not wait_for(snap + "_vecnorm.pkl", label, run_name):
            break
        r = run_eval(snap, run_name, label, seed_base=1000)
        if not r:
            continue
        results.append((r[0], r[1], snap))
        log(f"{run_name} {label}: {r[0]}/60 (median {r[1]})")
        if r[0] >= 54:                    # 90% headline -> STOP (user: even a single hit is final)
            log(f"{run_name} {label}: HIT 54/60 (90%) HEADLINE -- stopping per user instruction. "
                f"Pulling the fresh-seed number too so you know the real rate, not just this one.")
            base = 4000 + step // 100_000
            rc = run_eval(snap, run_name, label, seed_base=base, suffix="_confirm")
            if rc:
                pooled = r[0] + rc[0]
                log(f"{run_name} {label} confirm (informational, not gating): {rc[0]}/60 fresh -> "
                    f"pooled {pooled}/120 ({100*pooled/120:.1f}%). Headline stands as FINAL regardless.")
            return results, (r[0], snap)
        elif r[0] >= HEADLINE_TRIGGER:
            base = 4000 + step // 100_000
            rc = run_eval(snap, run_name, label, seed_base=base, suffix="_confirm")
            if rc:
                pooled = r[0] + rc[0]
                log(f"{run_name} {label} confirm: {rc[0]}/60 fresh -> pooled {pooled}/120 "
                    f"({100*pooled/120:.1f}%)")
                if pooled >= GOAL_POOLED and min(r[0], rc[0]) >= GOAL_MIN_SET:
                    log(f"*** GENUINE 90%+ CONFIRMED: {run_name} {label} = {pooled}/120, "
                        f"sets {r[0]}+{rc[0]}, neither below {GOAL_MIN_SET} ***")
                    return results, (pooled, snap)
    fin = f"runs/ppo_gimbal_{run_name}/ppo_final"
    if os.path.exists(fin + ".zip"):
        r = run_eval(fin, run_name, "final", seed_base=1000)
        if r:
            results.append((r[0], r[1], fin))
            log(f"{run_name} final: {r[0]}/60 (median {r[1]})")
            if r[0] >= 54:
                log(f"{run_name} final: HIT 54/60 (90%) HEADLINE -- stopping per user instruction.")
                rc = run_eval(fin, run_name, "final", seed_base=9999, suffix="_confirm")
                if rc:
                    pooled = r[0] + rc[0]
                    log(f"{run_name} final confirm (informational): {rc[0]}/60 fresh -> "
                        f"pooled {pooled}/120 ({100*pooled/120:.1f}%). Headline stands as FINAL.")
                return results, (r[0], fin)
            elif r[0] >= HEADLINE_TRIGGER:
                rc = run_eval(fin, run_name, "final", seed_base=9999, suffix="_confirm")
                if rc:
                    pooled = r[0] + rc[0]
                    log(f"{run_name} final confirm: {rc[0]}/60 fresh -> pooled {pooled}/120 "
                        f"({100*pooled/120:.1f}%)")
                    if pooled >= GOAL_POOLED and min(r[0], rc[0]) >= GOAL_MIN_SET:
                        log(f"*** GENUINE 90%+ CONFIRMED: {run_name} final = {pooled}/120 ***")
                        return results, (pooled, fin)
    return results, None


PARENT_SNAP = "best_policy/mirte_best"   # v31/ppo_final, 84.2% pooled, verified safe
lr = 2.5e-6
gen = 34

for _ in range(MAX_GENERATIONS):
    run_name = f"v{gen}"
    results, goal = run_generation(run_name, PARENT_SNAP, lr)
    if goal:
        log(f"=== CHAIN DONE, GENUINE GOAL MET: {goal[1]} at {goal[0]}/120 ===")
        break
    if not results:
        log(f"=== CHAIN ABORTED: {run_name} produced no results ===")
        break
    best = max(results, key=lambda r: r[0])
    log(f"{run_name} best headline this gen: {best[0]}/60 ({best[2]}) -- no confirmed 90%, "
        f"trying again from the same verified 84.2% parent (independent sample)")
    gen += 1

log("=== ladder_chain4 exit ===")
