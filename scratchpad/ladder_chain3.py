"""Ladder chain v3 -- the 'consistent 90%' push, diagnosis-driven.

Evidence baked in (2026-07-13 morning):
  - Fresh-seed tournament: ALL top checkpoints (v29@8M, v30@{3M,8M,final}) are
    statistically identical at ~82%. Parent choice doesn't matter; training
    distribution does.
  - failure_map v30@8M (60 fresh eps): 10/12 failures in the FRONT half
    (pillar_field 6, doorway 3, start 1); stairs nearly solved (2). The
    curriculum (65% scattered spawns, 40% goal-side) starves exactly that
    front-half full-course experience -> v31 trains with --no-curriculum
    (100% standard starts), lr kept at 5e-6 for distribution-shift headroom.

SUCCESS CRITERION ("consistent 90"): for one checkpoint, pooled >= 108/120
over the fixed set (seed 1000) + a never-used fresh set, AND neither set
below 52. A single lucky 60 can no longer declare victory.

Chain: if a generation ends without success, take its best checkpoint,
fresh-confirm it, and if pooled beats the parent's 82% -> next generation
(lr halved, still no-curriculum), else honest plateau report. Max 3 more
generations after v31.
"""
import glob, os, re, subprocess, sys, time

import psutil

REPO = os.getcwd()
PY = sys.executable
ARC_LOG = "runs/ladder_chain.log"
GOAL_POOLED = 108          # /120 = 90%
GOAL_MIN_SET = 52          # neither 60-ep set below this
HEADLINE_TRIGGER = 52      # headline >= this -> spend a confirm eval
PARENT_POOLED_PCT = 82.2   # v30 pooled truth; a generation must beat this
MAX_GENERATIONS = 2        # v31 + v32, then stop regardless (user decision)
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
    log(f"=== launching {run_name}: from {from_model}, lr={lr}, NO-CURRICULUM ===")
    logf = open(f"runs/{run_name}_train.log", "w", encoding="utf-8")
    errf = open(f"runs/{run_name}_train.err", "w", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": REPO}
    subprocess.Popen(
        [PY, "-u", "finetune_gimbal.py",
         "--from-model", from_model, "--vecnorm", vecnorm,
         "--gamma", "0.995", "--steps", "8000000", "--n-envs", "12",
         "--lr", str(lr), "--no-curriculum", "--out", f"runs/ppo_gimbal_{run_name}"],
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
        log(f"=== resuming watch on {run_name} (already training) ===")
    elif not launch_training(run_name, from_model, vecnorm_for(from_model), lr):
        log(f"{run_name}: FAILED TO START (runs/{run_name}_train.err)")
        return None, None
    results = []                      # (succ, median, snap)
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
        if r[0] >= HEADLINE_TRIGGER:
            base = 4000 + step // 100_000          # unique fresh set per milestone
            rc = run_eval(snap, run_name, label, seed_base=base, suffix="_confirm")
            if rc:
                pooled = r[0] + rc[0]
                log(f"{run_name} {label} confirm: {rc[0]}/60 fresh -> pooled {pooled}/120 "
                    f"({100*pooled/120:.1f}%)")
                if pooled >= GOAL_POOLED and min(r[0], rc[0]) >= GOAL_MIN_SET:
                    log(f"*** CONSISTENT 90 ACHIEVED: {run_name} {label} = {pooled}/120, "
                        f"sets {r[0]}+{rc[0]} ***")
                    return results, (pooled, snap)
    # generation over without confirmed goal -> eval final too
    fin = f"runs/ppo_gimbal_{run_name}/ppo_final"
    if os.path.exists(fin + ".zip"):
        r = run_eval(fin, run_name, "final", seed_base=1000)
        if r:
            results.append((r[0], r[1], fin))
            log(f"{run_name} final: {r[0]}/60 (median {r[1]})")
    return results, None


chain_from = "runs/ppo_gimbal_v32/snap_02000k"   # resumed after a manual pause; v32 killed mid-run at 2M
lr = 2.5e-6                                       # matches v32's original lr (not halved further)
prior_pooled_pct = 84.2                           # v32's last confirmed pooled truth (500k, still the bar to beat)
gen = 33

for _ in range(MAX_GENERATIONS):
    run_name = f"v{gen}"
    results, goal = run_generation(run_name, chain_from, lr)
    if goal:
        log(f"=== CHAIN DONE, GOAL MET: {goal[1]} at {goal[0]}/120 ===")
        break
    if not results:
        log(f"=== CHAIN ABORTED: {run_name} produced no results ===")
        break
    best = max(results, key=lambda r: r[0])
    # true-rate check on this generation's best (fresh set, unique base)
    rc = run_eval(best[2], run_name, "best", seed_base=6000 + gen * 100, suffix="_truth")
    pooled_pct = 100 * (best[0] + (rc[0] if rc else 0)) / 120 if rc else 100 * best[0] / 60
    log(f"{run_name} BEST {best[2]}: headline {best[0]}/60, fresh {rc[0] if rc else '?'}"
        f"/60 -> pooled {pooled_pct:.1f}% (parent bar {prior_pooled_pct:.1f}%)")
    if pooled_pct <= prior_pooled_pct:
        log(f"=== PLATEAU: {run_name} ({pooled_pct:.1f}%) did not beat parent "
            f"({prior_pooled_pct:.1f}%). Honest best stands. Chain stopping. ===")
        break
    chain_from = best[2]
    prior_pooled_pct = pooled_pct
    lr = lr / 2
    gen += 1
    log(f"chain continues -> v{gen} from {chain_from} at lr {lr}")

log("=== ladder_chain3 exit ===")
