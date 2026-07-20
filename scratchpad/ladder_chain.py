"""Autonomous ladder toward 90% (54/60), corrected version.

Fix vs v1: a single 60-ep eval hitting the goal is NOT enough to declare
victory (fixed seeding means eval_checkpoint.py is deterministic -- a
checkpoint's headline number can be a lucky draw on that particular 60-scenario
set, as v30@3M's 54/60 headline vs 49/60 on fresh seeds proved). Now: hitting
the goal triggers an independent confirmation eval on a DIFFERENT seed base
(--seed-base 2000); only stop if BOTH clear the bar. Also resumable: skips
milestones a prior process already evaluated (existing non-empty log file),
so it can pick up mid-generation (v30 was left running by the old process).
"""
import glob, os, re, subprocess, sys, time

import psutil

REPO = os.getcwd()
PY = sys.executable
ARC_LOG = "runs/ladder_chain.log"
V29_ARC = "runs/v29_overnight_arc.log"
GOAL_SUCCESS = 54          # /60 = 90%
MAX_GENERATIONS = 5
ALL_TARGETS = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000,
               1_500_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000,
               6_000_000, 7_000_000, 8_000_000]
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
            name = (p.info["name"] or "").lower()
            if "python" not in name:
                continue
            cmd = p.info["cmdline"] or []
            joined = " ".join(cmd)
            if "finetune_gimbal.py" in joined and f"ppo_gimbal_{run_name}" in joined:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def wait_for(path, label, run_name, max_wait=MAX_WAIT_S):
    waited = 0
    while not os.path.exists(path):
        if not training_alive(run_name):
            log(f"{run_name}: TRAINING DIED before {label} (missing {path})")
            return False
        if waited >= max_wait:
            log(f"{run_name}: STALL WARNING, {label} not seen after {max_wait}s, still alive, continuing")
            waited = 0
        time.sleep(POLL_S)
        waited += POLL_S
    return True


def parse_eval_log(logfile):
    if not os.path.exists(logfile):
        return None
    text = open(logfile, encoding="utf-8").read()
    succ = re.search(r"success\s*:\s*(\d+)/(\d+)", text)
    med = re.search(r"median\s*(-?[\d.]+)", text)
    trimmed = re.search(r"trimmed\(10%\)\s*(-?[\d.]+)", text)
    if not (succ and med):
        return None
    s = int(succ.group(1))
    t = float(trimmed.group(1)) if trimmed else float(med.group(1))
    return (s, t, float(med.group(1)))


def run_eval(snap_prefix, label, run_name, seed_base=1000, suffix=""):
    logfile = f"runs/{run_name}_eval_{label}{suffix}.log"
    parsed = parse_eval_log(logfile)
    if parsed:
        return (*parsed, snap_prefix)              # already done by a prior process -- skip
    env = {**os.environ, "PYTHONPATH": REPO}
    with open(logfile, "w", encoding="utf-8") as out:
        subprocess.run([PY, "tools/eval_checkpoint.py", "--snap", snap_prefix,
                        "--episodes", "60", "--seed-base", str(seed_base)],
                       stdout=out, stderr=subprocess.STDOUT, env=env)
    parsed = parse_eval_log(logfile)
    if not parsed:
        log(f"{run_name} {label}{suffix}: EVAL FAILED TO PARSE (see {logfile})")
        return None
    s, t, m = parsed
    fail = re.search(r"failure causes\s*:\s*(\{.*\})", open(logfile, encoding="utf-8").read())
    log(f"{run_name} {label}{suffix}: {s}/60 success, median {m}, trimmed {t}, "
        f"{fail.group(1) if fail else '?'}")
    return (s, t, m, snap_prefix)


def best_of(run_name):
    best = None
    for lp in glob.glob(f"runs/{run_name}_eval_*.log"):
        if lp.endswith("_confirm.log"):
            continue                                 # confirmation runs aren't generation milestones
        p = parse_eval_log(lp)
        if p is None:
            continue
        chk = re.search(r"checkpoint\s*:\s*(\S+)", open(lp, encoding="utf-8").read())
        if not chk:
            continue
        cand = (*p, chk.group(1))
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    return best


def run_generation(run_name, from_snap=None, from_vecnorm=None, lr=None, resume=False):
    """resume=True: the trainer for run_name is already running (launched by a
    prior process); just catch up on evaluations. Otherwise launch it fresh."""
    if not resume:
        log(f"=== launching {run_name}: from {from_snap}  lr={lr} ===")
        logf = open(f"runs/{run_name}_train.log", "w", encoding="utf-8")
        errf = open(f"runs/{run_name}_train.err", "w", encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": REPO}
        subprocess.Popen(
            [PY, "-u", "finetune_gimbal.py",
             "--from-model", from_snap, "--vecnorm", from_vecnorm,
             "--gamma", "0.995", "--steps", "8000000", "--n-envs", "12",
             "--lr", str(lr), "--out", f"runs/ppo_gimbal_{run_name}"],
            stdout=logf, stderr=errf, env=env, cwd=REPO,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            if os.name == "nt" else 0)
        time.sleep(10)
        if not training_alive(run_name):
            log(f"{run_name}: FAILED TO START (check runs/{run_name}_train.err)")
            return None
    else:
        log(f"=== resuming watch on {run_name} (already running) ===")

    results = []
    for step in ALL_TARGETS:
        label = label_for(step)
        snap = f"runs/ppo_gimbal_{run_name}/snap_{step//1000:05d}k"
        if not wait_for(snap + "_vecnorm.pkl", label, run_name):
            return results or None
        r = run_eval(snap, label, run_name)
        if r:
            results.append(r)
            if r[0] >= GOAL_SUCCESS:
                log(f"{run_name} {label} headline {r[0]}/60 (>=90%) -- running independent "
                    f"confirmation on fresh seeds before declaring anything")
                rc = run_eval(snap, label, run_name, seed_base=2000, suffix="_confirm")
                if rc and rc[0] >= GOAL_SUCCESS:
                    log(f"*** GOAL REACHED AND CONFIRMED: {run_name} {label} = "
                        f"{r[0]}/60 headline, {rc[0]}/60 independent confirm ***")
                    return results
                elif rc:
                    log(f"{run_name} {label}: headline {r[0]}/60 did NOT hold on confirmation "
                        f"({rc[0]}/60 fresh eps) -- not declaring victory, continuing to train")
                else:
                    log(f"{run_name} {label}: confirmation eval failed to parse, continuing")
    if wait_for(f"runs/ppo_gimbal_{run_name}/ppo_final.zip", "final", run_name):
        r = run_eval(f"runs/ppo_gimbal_{run_name}/ppo_final", "final", run_name)
        if r:
            results.append(r)
    log(f"{run_name}: generation complete (8M)")
    return results


# ---- resume v30 (already running, launched by the earlier buggy process) ----
log("ladder_chain v2 (confirmation-gated): resuming v30")
v30_results = run_generation("v30", resume=True)
prior_best = best_of("v30") or best_of("v29")
prior_run = "v30" if best_of("v30") else "v29"
log(f"{prior_run} BEST (post-fix): {prior_best[0]}/60 success, trimmed {prior_best[1]}, snap {prior_best[3]}")

lr = 2.5e-6                                   # halved again from v30's 5e-6
gen_num = 31

if prior_best[0] >= GOAL_SUCCESS:
    log(f"*** GOAL ALREADY CONFIRMED AT {prior_run} -- no further generations needed ***")
else:
    for _ in range(MAX_GENERATIONS):
        run_name = f"v{gen_num}"
        from_snap = prior_best[3][:-4] if prior_best[3].endswith(".zip") else prior_best[3]
        if from_snap.endswith("ppo_final"):
            from_vecnorm = from_snap.rsplit("/", 1)[0] + "/vecnormalize.pkl"
        else:
            from_vecnorm = from_snap + "_vecnorm.pkl"

        results = run_generation(run_name, from_snap, from_vecnorm, lr)
        if not results:
            log(f"{run_name}: no results (died early) -- stopping chain, keeping {prior_run} best")
            break
        gen_best = max(results, key=lambda r: (r[0], r[1]))
        log(f"{run_name} BEST: {gen_best[0]}/60 success, trimmed {gen_best[1]}, snap {gen_best[3]}")

        if gen_best[0] >= GOAL_SUCCESS:
            # run_generation only returns early on a CONFIRMED goal hit, so if
            # gen_best itself is the goal-hit checkpoint it's already confirmed.
            prior_best, prior_run = gen_best, run_name
            break
        if (gen_best[0], gen_best[1]) <= (prior_best[0], prior_best[1]):
            log(f"{run_name} did NOT beat {prior_run} ({gen_best[0]} vs {prior_best[0]}) "
                f"-- PLATEAU, stopping chain. Honest best remains {prior_run}: {prior_best[0]}/60")
            break

        prior_best, prior_run = gen_best, run_name
        lr = lr / 2
        gen_num += 1

log(f"=== LADDER CHAIN v2 DONE. Final best: {prior_run} = {prior_best[0]}/60 "
    f"({100*prior_best[0]/60:.0f}%), snap {prior_best[3]} ===")
