"""Local dashboard for all training runs: http://localhost:8500

One card per runs/ppo_gimbal_* dir showing a live progress bar (parsed from the
run's train log: total_timesteps / fps -> ETA), status (RUNNING / DONE /
STOPPED), and the 30-ep eval history (completions + median per checkpoint)
scraped from runs/*.log summaries. Stdlib only; near-zero overhead next to a
live training run (reads log tails on each refresh).

Usage:
    python tools/run_dashboard.py            # serve on :8500
    python tools/run_dashboard.py --port 9000
"""
import argparse
import glob
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(HERE, "runs")
TOTAL_DEFAULT = 8_000_000


def tail_bytes(path, n=60_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def scan_evals():
    """Parse every runs/*.log eval summary -> {run: [(ckpt_steps, succ, n, median)]}"""
    out = {}
    for lp in glob.glob(os.path.join(RUNS, "*.log")):
        txt = tail_bytes(lp, 2000)
        m = re.search(r"checkpoint\s*:\s*runs[/\\](ppo_gimbal_\w+)[/\\]snap_(\d+)k", txt)
        s = re.search(r"success\s*:\s*(\d+)/(\d+)", txt)
        md = re.search(r"median\s*(-?[\d.]+)", txt)
        if m and s and md:
            out.setdefault(m.group(1), {})[int(m.group(2)) * 1000] = (
                int(s.group(1)), int(s.group(2)), float(md.group(1)))
    return {k: sorted((ck, *v) for ck, v in d.items()) for k, d in out.items()}


def active_run_names():
    """Run dir names with a live training process (matched via its --out arg).

    Log-silence freshness alone is unreliable: SB3 emits nothing during a
    rollout, so a long-rollout run looks 'STOPPED' for minutes while healthy.
    A live process owning the run's --out dir is ground truth that it's running.
    """
    active = set()
    try:
        import psutil
    except ImportError:
        return active
    for p in psutil.process_iter(["cmdline"]):
        try:
            cmd = p.info.get("cmdline") or []
        except Exception:
            continue
        if "--out" in cmd:
            i = cmd.index("--out")
            if i + 1 < len(cmd):
                active.add(os.path.basename(cmd[i + 1].rstrip("/\\")))
    return active


def scan_runs():
    evals = scan_evals()
    active = active_run_names()
    cards = []
    for d in sorted(glob.glob(os.path.join(RUNS, "ppo_gimbal_*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        ver = name.replace("ppo_gimbal_", "")
        log = os.path.join(RUNS, f"{ver}_train.log")
        steps = fps = 0
        log_age = None
        done = os.path.exists(os.path.join(d, "ppo_final.zip"))
        if os.path.exists(log):
            txt = tail_bytes(log)
            ts = re.findall(r"total_timesteps\s*\|\s*(\d+)", txt)
            fp = re.findall(r"fps\s*\|\s*(\d+)", txt)
            if ts:
                steps = int(ts[-1])
            if fp:
                fps = int(fp[-1])
            log_age = time.time() - os.path.getmtime(log)
            if "saved model and normalization" in txt:
                done = True
        snap_paths = glob.glob(os.path.join(d, "snap_*k.zip"))
        # (steps, mtime) per snapshot, sorted by step -> used for the log-less
        # step count AND to extrapolate a live estimate between sparse snapshots.
        snap_info = sorted(
            (int(m.group(1)) * 1000, os.path.getmtime(p))
            for p in snap_paths if (m := re.search(r"snap_(\d+)k", p)))
        if steps == 0 and snap_info:  # fall back to newest snapshot tag
            steps = snap_info[-1][0]
        # freshness = most recent of (train log, newest snapshot). SB3 prints its
        # log block only ~once/min AFTER the first rollout, so during startup the
        # log looks stale while snapshots are already being written -> use both.
        snap_age = (time.time() - max(os.path.getmtime(p) for p in snap_paths)
                    if snap_paths else None)
        ages = [a for a in (log_age, snap_age) if a is not None]
        fresh = min(ages) if ages else None
        if done:
            status, steps = "DONE", max(steps, TOTAL_DEFAULT)
        elif name in active:  # live process owning this run's --out dir
            status = "RUNNING"
        elif fresh is not None and fresh < 180:
            status = "RUNNING"
        else:
            status = "STOPPED"
        # Live estimate: verbose=0 runs only reveal progress at sparse snapshot
        # milestones, so the count sits still for many minutes. Extrapolate from
        # the last two snapshots' rate so the bar moves each refresh; it snaps
        # back to truth when the next real snapshot lands. Marked '~' in the UI.
        disp_steps, est = steps, False
        if status == "RUNNING" and len(snap_info) >= 2:
            (s0, t0), (s1, t1) = snap_info[-2], snap_info[-1]
            if t1 > t0 and s1 > s0:
                rate = (s1 - s0) / (t1 - t0)          # steps/sec, last interval
                guess = int(s1 + rate * (time.time() - t1))
                disp_steps = min(max(steps, guess), TOTAL_DEFAULT)
                est = disp_steps > steps
        eta = ""
        if status == "RUNNING" and fps:
            rem = (TOTAL_DEFAULT - steps) / fps
            eta = f"{int(rem // 3600)}:{int(rem % 3600 // 60):02d} left @ {fps} it/s"
        cards.append({
            "name": ver, "steps": disp_steps, "total": TOTAL_DEFAULT,
            "pct": min(100.0, 100.0 * disp_steps / TOTAL_DEFAULT),
            "status": status, "eta": eta, "est": est,
            "evals": [{"ck": f"{ck/1e6:g}M", "succ": f"{s}/{n}", "median": md}
                      for ck, s, n, md in evals.get(name, [])],
        })
    order = {"RUNNING": 0, "STOPPED": 1, "DONE": 1}
    cards.sort(key=lambda c: (order[c["status"]],
                              -int(re.sub(r"\D", "", c["name"]) or 0)))
    return cards


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>MIRTE runs</title>
<style>
 body{background:#101014;color:#e8e8ee;font:14px/1.5 'Segoe UI',sans-serif;
      max-width:880px;margin:24px auto;padding:0 16px}
 h1{font-size:19px} .muted{color:#8a8a97}
 .card{background:#1a1a22;border:1px solid #2a2a35;border-radius:10px;
       padding:14px 16px;margin:12px 0}
 .row{display:flex;justify-content:space-between;align-items:baseline}
 .name{font-weight:700;font-size:16px}
 .st-RUNNING{color:#7ec8ff}.st-DONE{color:#79e08c}.st-STOPPED{color:#8a8a97}
 .barwrap{background:#26262f;border-radius:6px;height:14px;margin:8px 0;overflow:hidden}
 .bar{height:100%;background:linear-gradient(90deg,#3f8cff,#7ec8ff);
      border-radius:6px;transition:width .5s}
 .bar-DONE{background:linear-gradient(90deg,#2e9e4f,#79e08c)}
 table{border-collapse:collapse;margin-top:6px;font-size:13px}
 td,th{padding:1px 10px 1px 0;text-align:left;color:#bfbfca}
 th{color:#8a8a97;font-weight:600}
 .succ-hit{color:#79e08c;font-weight:700}
</style></head><body>
<h1>MIRTE gimbal - training runs <span class="muted" id="ts"></span></h1>
<div id="cards"></div>
<script>
async function tick(){
  const r = await fetch('/data'); const runs = await r.json();
  document.getElementById('ts').textContent = ' - ' + new Date().toLocaleTimeString();
  document.getElementById('cards').innerHTML = runs.map(c => `
   <div class="card">
    <div class="row"><span class="name">${c.name}</span>
      <span class="st-${c.status}">${c.status}${c.eta ? ' - ' + c.eta : ''}</span></div>
    <div class="barwrap"><div class="bar ${c.status==='DONE'?'bar-DONE':''}"
         style="width:${c.pct.toFixed(1)}%"></div></div>
    <div class="row"><span class="muted">
      ${c.est?'~':''}${(c.steps/1e6).toFixed(2)}M / ${(c.total/1e6).toFixed(0)}M steps${c.est?' (est)':''}</span>
      <span class="muted">${c.pct.toFixed(1)}%</span></div>
    ${c.evals.length ? `<table><tr><th>ckpt</th>${c.evals.map(e=>`<td>${e.ck}</td>`).join('')}</tr>
      <tr><th>success</th>${c.evals.map(e=>`<td class="${e.succ[0]!=='0'?'succ-hit':''}">${e.succ}</td>`).join('')}</tr>
      <tr><th>median</th>${c.evals.map(e=>`<td>${e.median.toFixed(1)}</td>`).join('')}</tr></table>` : ''}
   </div>`).join('');
}
tick(); setInterval(tick, 5000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/data":
            body = json.dumps(scan_runs()).encode()
            ctype = "application/json"
        else:
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # keep console quiet
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8500)
    args = ap.parse_args()
    print(f"dashboard: http://localhost:{args.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
