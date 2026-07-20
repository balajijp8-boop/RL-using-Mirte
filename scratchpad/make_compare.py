"""Assemble the 3 grip renders into a self-contained comparison HTML (base64
data URIs) for the Artifact viewer."""
import base64, os
HERE = os.path.dirname(__file__)
G = os.path.join(HERE, "grips")

def uri(p):
    with open(os.path.join(G, p), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

imgA, imgB, imgC = uri("A_baseline_6cm_open.png"), uri("B_trayjaws_2cm.png"), uri("C_fingers_2cm.png")

HTML = f"""<style>
:root {{
  --bg:#f3f4f6; --surface:#ffffff; --border:#d9dde3; --ink:#191d23;
  --muted:#5b6470; --accent:#c65c12; --accent-soft:#f4e6da; --mono:#3a4048;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#111419; --surface:#1a1f27; --border:#2a313c; --ink:#e7eaef;
           --muted:#9aa4b2; --accent:#f0842e; --accent-soft:#2a2117; --mono:#c3ccd8; }}
}}
:root[data-theme="light"] {{
  --bg:#f3f4f6; --surface:#ffffff; --border:#d9dde3; --ink:#191d23;
  --muted:#5b6470; --accent:#c65c12; --accent-soft:#f4e6da; --mono:#3a4048;
}}
:root[data-theme="dark"] {{
  --bg:#111419; --surface:#1a1f27; --border:#2a313c; --ink:#e7eaef;
  --muted:#9aa4b2; --accent:#f0842e; --accent-soft:#2a2117; --mono:#c3ccd8;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; }}
.wrap {{ font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  color:var(--ink); background:var(--bg); min-height:100vh;
  padding:40px 20px; max-width:1160px; margin:0 auto; }}
.eyebrow {{ font:600 12px/1 ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  letter-spacing:.14em; text-transform:uppercase; color:var(--accent); }}
h1 {{ font-size:30px; font-weight:700; margin:10px 0 6px; letter-spacing:-.01em;
  text-wrap:balance; }}
.sub {{ color:var(--muted); margin:0 0 30px; max-width:60ch; }}
.sub b {{ color:var(--ink); font-weight:600; }}
.grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:18px; }}
@media (max-width:820px) {{ .grid {{ grid-template-columns:1fr; }} }}
.card {{ background:var(--surface); border:1px solid var(--border);
  border-radius:14px; overflow:hidden; display:flex; flex-direction:column; }}
.card.rec {{ border-color:var(--accent); box-shadow:0 0 0 1px var(--accent); }}
.imgwrap {{ background:#0c0e12; aspect-ratio:1000/760; }}
.imgwrap img {{ width:100%; height:100%; object-fit:cover; display:block; }}
.body {{ padding:16px 18px 20px; display:flex; flex-direction:column; gap:8px; flex:1; }}
.head {{ display:flex; align-items:center; gap:10px; }}
.chip {{ flex:none; width:26px; height:26px; border-radius:7px; display:grid;
  place-items:center; font:700 14px/1 ui-monospace,monospace;
  background:var(--accent-soft); color:var(--accent); }}
.card.rec .chip {{ background:var(--accent); color:#fff; }}
h2 {{ font-size:18px; font-weight:650; margin:0; }}
.badge {{ margin-left:auto; font:600 11px/1 ui-monospace,monospace; letter-spacing:.08em;
  text-transform:uppercase; color:var(--accent); border:1px solid var(--accent);
  border-radius:999px; padding:4px 9px; }}
.spec {{ font:500 13px/1.4 ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  color:var(--mono); margin:0; }}
.cap {{ color:var(--muted); font-size:14.5px; margin:0; }}
.cap b {{ color:var(--ink); font-weight:600; }}
.take {{ margin-top:26px; background:var(--surface); border:1px solid var(--border);
  border-left:3px solid var(--accent); border-radius:12px; padding:18px 22px; }}
.take p {{ margin:0; color:var(--muted); }}
.take b {{ color:var(--ink); }}
</style>
<div class="wrap">
  <div class="eyebrow">MIRTE · payload retention</div>
  <h1>Grip options to stop the drop</h1>
  <p class="sub">Static renders for your approval. <b>Nothing is training on these</b> — the arm run (v28arm) continues to its 4M gate untouched; this is staged for the next run.</p>
  <div class="grid">
    <article class="card">
      <div class="imgwrap"><img src="{imgA}" alt="Baseline grip: stack hangs 6cm below the pivot in the open-finger tray"></div>
      <div class="body">
        <div class="head"><span class="chip">A</span><h2>Baseline</h2></div>
        <p class="spec">6&nbsp;cm drop &middot; open fingers</p>
        <p class="cap">Current setup. The stack hangs 6&nbsp;cm below the pivot; fingers straddle the tray but don't squeeze it.</p>
      </div>
    </article>
    <article class="card rec">
      <div class="imgwrap"><img src="{imgB}" alt="Tray-jaws clamp: orange jaws grip the lower cylinder, stack seated higher"></div>
      <div class="body">
        <div class="head"><span class="chip">B</span><h2>Tray-jaws clamp</h2><span class="badge">Recommended</span></div>
        <p class="spec">2&nbsp;cm drop &middot; jaws grip lower cylinder</p>
        <p class="cap"><b>Orange jaws grip the gold lower cylinder</b>, and the stack seats ~4&nbsp;cm higher. A real clamp on the exact part that slides out.</p>
      </div>
    </article>
    <article class="card">
      <div class="imgwrap"><img src="{imgC}" alt="Finger clamp: nearly identical to baseline, stack seated higher"></div>
      <div class="body">
        <div class="head"><span class="chip">C</span><h2>Finger clamp</h2></div>
        <p class="spec">2&nbsp;cm drop &middot; fingers closed</p>
        <p class="cap">Almost identical to A — the fingers already touch the box, so closing barely moves. Effectively <b>shorter drop only</b>.</p>
      </div>
    </article>
  </div>
  <div class="take">
    <p><b>The real choice:</b> “tray-jaws + shorter drop” (B) vs “shorter drop alone” (C). B grips the cylinder that actually topples out; C's fingers can't clamp harder without clipping the payload. Either way I shake-test it (policy off) before we trust it for training.</p>
  </div>
</div>"""

out = os.path.join(HERE, "grips_compare.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", out, f"({len(HTML)//1024} KB)")
