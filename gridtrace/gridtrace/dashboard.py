"""
Local dashboard: Flask + one page that polls the SQLite store.

    python -m gridtrace.dashboard      ->  http://127.0.0.1:8000

Reads only. The inference service is the writer; SQLite is in WAL mode so the two
never block each other.
"""
from __future__ import annotations

import argparse
import json
import sys

from flask import Flask, Response, jsonify

from gridtrace import config
from gridtrace.storage import Store

app = Flask(__name__)
store: Store | None = None

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>GridTrace AI</title>
<style>
:root{--bg:#0f151c;--card:#18212c;--line:#26323f;--ink:#e6edf4;--dim:#8fa3b5;
      --ok:#4fbf89;--sus:#e0b04a;--high:#ef8a4c;--crit:#f2685c;--accent:#7fb2ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:14px/1.5 "Segoe UI",system-ui,sans-serif}
header{display:flex;align-items:baseline;gap:14px;padding:12px 18px;
       background:var(--card);border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0;letter-spacing:.02em}
.sub{color:var(--dim);font-size:12px}
.pill{margin-left:auto;font:12px ui-monospace,monospace;color:var(--dim)}
main{padding:14px 18px;display:grid;gap:14px;grid-template-columns:1fr 380px}
@media(max-width:1080px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);
         margin:0;padding:10px 13px;border-bottom:1px solid var(--line);font-weight:600}
.card .in{padding:12px 13px}
table{width:100%;border-collapse:collapse;font:12.5px ui-monospace,monospace}
th{color:var(--dim);text-align:right;font-weight:600;padding:6px 8px;
   border-bottom:1px solid var(--line);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
th.l,td.l{text-align:left}
td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}
tr:last-child td{border-bottom:none}
.band{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;font-weight:700;
      font-family:"Segoe UI",sans-serif;letter-spacing:.03em}
.NORMAL{background:#12291f;color:var(--ok)}
.SUSPICIOUS{background:#2b2413;color:var(--sus)}
.HIGH.RISK,.HIGHRISK{background:#2f2013;color:var(--high)}
.CRITICAL{background:#2f1a17;color:var(--crit)}
.meter{height:5px;background:#22303d;border-radius:3px;overflow:hidden;min-width:60px}
.meter i{display:block;height:100%;border-radius:3px}
.loc{padding:12px 13px;border-bottom:1px solid var(--line)}
.loc .big{font:700 30px ui-monospace,monospace}
.loc .sum{color:var(--dim);font-size:12.5px;margin-top:6px}
ul.reasons{margin:8px 0 0;padding-left:17px;color:var(--dim);font-size:12.5px}
ul.reasons li{margin:2px 0}
.alert{padding:9px 13px;border-bottom:1px solid var(--line);font-size:12.5px}
.alert:last-child{border-bottom:none}
.alert b{font-family:ui-monospace,monospace}
.alert .t{color:var(--dim);font-size:11px}
canvas{width:100%;height:90px;display:block}
.foot{padding:10px 18px 24px;color:var(--dim);font-size:12px;max-width:80ch}
.kpis{display:flex;gap:18px;flex-wrap:wrap}
.kpi{display:flex;flex-direction:column}
.kpi b{font:600 17px ui-monospace,monospace}
.kpi span{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
</style></head><body>
<header><h1>GridTrace AI</h1>
<span class="sub">hybrid Isolation Forest + autoencoder &middot; feeder F01</span>
<span class="pill" id="pill">connecting…</span></header>
<main>
 <div style="display:flex;flex-direction:column;gap:14px">
  <div class="card"><h2>Live consumer readings &amp; hybrid scoring</h2>
   <div style="overflow-x:auto"><table id="tbl"><thead><tr>
   <th class="l">Consumer</th><th class="l">Zone</th><th>V</th><th>I</th><th>P (W)</th><th>PF</th>
   <th>E (kWh)</th><th>IF</th><th>AE</th><th>Hybrid</th><th>Risk</th><th class="l">Status</th>
   </tr></thead><tbody></tbody></table></div></div>
  <div class="card"><h2>Feeder risk &amp; imbalance trend</h2>
   <div class="in"><canvas id="chart"></canvas></div></div>
 </div>
 <div style="display:flex;flex-direction:column;gap:14px">
  <div class="card"><h2>Localization</h2><div id="loc"></div></div>
  <div class="card"><h2>Recent alerts</h2><div id="alerts"></div></div>
  <div class="card"><h2>Session counts</h2><div class="in kpis" id="kpis"></div></div>
 </div>
</main>
<p class="foot">Risk bands: 0&ndash;30 NORMAL &middot; 31&ndash;60 SUSPICIOUS &middot; 61&ndash;80 HIGH RISK &middot;
81&ndash;100 CRITICAL. A high score means <em>investigation required</em> &mdash; it is not proof of theft.
Feeder imbalance is supporting evidence only; when no meter explains the gap the system localizes to the
feeder and says the loss is unattributed rather than blaming a consumer.</p>
<script>
const bandClass = b => (b||"NORMAL").replace(/\\s+/g,"");
const bandColor = b => ({NORMAL:"#4fbf89",SUSPICIOUS:"#e0b04a","HIGH RISK":"#ef8a4c",CRITICAL:"#f2685c"}[b]||"#4fbf89");
let series=[];
async function tick(){
  let d; try{ d = await (await fetch("/api/live")).json(); }
  catch(e){ document.getElementById("pill").textContent="offline"; return; }
  document.getElementById("pill").textContent =
    d.readings.length ? `tick ${d.readings[0].tick} · ${d.counts.total} readings scored` : "waiting for telemetry…";

  document.querySelector("#tbl tbody").innerHTML = d.readings.map(r=>`<tr>
    <td class="l"><b>${r.consumer_id}</b></td><td class="l">${r.zone_id}</td>
    <td>${(r.voltage||0).toFixed(1)}</td><td>${(r.current||0).toFixed(2)}</td>
    <td>${(r.power_w||0).toFixed(0)}</td><td>${(r.power_factor||0).toFixed(2)}</td>
    <td>${(r.energy_kwh||0).toFixed(2)}</td>
    <td>${(r.iforest_score||0).toFixed(2)}</td><td>${(r.autoencoder_score||0).toFixed(2)}</td>
    <td>${(r.hybrid_score||0).toFixed(2)}</td>
    <td><div class="meter"><i style="width:${r.risk_score||0}%;background:${bandColor(r.risk_band)}"></i></div>
        ${(r.risk_score||0).toFixed(0)}</td>
    <td class="l"><span class="band ${bandClass(r.risk_band)}">${r.risk_band||""}</span></td></tr>`).join("");

  const L=d.localization;
  document.getElementById("loc").innerHTML = L ? `<div class="loc">
    <div class="big" style="color:${bandColor(L.risk_band)}">${(L.risk_score||0).toFixed(0)}<span style="font-size:15px;color:var(--dim)">/100</span></div>
    <div><span class="band ${bandClass(L.risk_band)}">${L.risk_band}</span>
      <span style="font-family:ui-monospace,monospace;margin-left:8px">${L.level}</span></div>
    <div class="sum">Feeder ${L.feeder_id||"—"} · Zone ${L.zone_id||"—"} · Consumer ${L.consumer_id||"—"}</div>
    <div class="sum">${L.summary||""}</div>
    ${(L.reasons&&L.reasons.length)?`<ul class="reasons">${L.reasons.map(x=>`<li>${x}</li>`).join("")}</ul>`:""}
    </div>` : `<div class="loc sum">waiting…</div>`;

  document.getElementById("alerts").innerHTML = d.alerts.length ? d.alerts.map(a=>`<div class="alert">
    <span class="band ${bandClass(a.risk_band)}">${a.risk_band}</span>
    <b> ${a.consumer_id||a.level}</b> — ${(a.risk_score||0).toFixed(0)}/100
    <div class="t">tick ${a.tick} · ${a.summary||""}</div></div>`).join("")
    : `<div class="alert t">no alerts yet</div>`;

  document.getElementById("kpis").innerHTML = Object.entries(d.counts.bands)
    .map(([k,v])=>`<div class="kpi"><b style="color:${bandColor(k)}">${v}</b><span>${k}</span></div>`).join("");

  series = d.series || [];
  draw();
}
function draw(){
  const c=document.getElementById("chart"), x=c.getContext("2d");
  const dpr=devicePixelRatio||1, w=c.clientWidth, h=90;
  if(c.width!==w*dpr||c.height!==h*dpr){c.width=w*dpr;c.height=h*dpr;}
  x.setTransform(dpr,0,0,dpr,0,0); x.clearRect(0,0,w,h);
  if(series.length<2) return;
  const PL=30,PR=6,PT=8,PB=14,iw=w-PL-PR,ih=h-PT-PB;
  const Y=v=>PT+ih-(Math.max(0,Math.min(100,v))/100)*ih, X=i=>PL+(i/(series.length-1))*iw;
  x.strokeStyle="#26323f";x.lineWidth=1;x.font='10px ui-monospace,monospace';x.fillStyle="#8fa3b5";
  [0,50,100].forEach(v=>{x.beginPath();x.moveTo(PL,Math.round(Y(v))+.5);x.lineTo(w-PR,Math.round(Y(v))+.5);x.stroke();
    x.textAlign="right";x.textBaseline="middle";x.fillText(v,PL-5,Y(v));});
  x.strokeStyle="#ef8a4c";x.lineWidth=1;x.setLineDash([3,3]);
  x.beginPath();x.moveTo(PL,Y(61));x.lineTo(w-PR,Y(61));x.stroke();x.setLineDash([]);
  x.strokeStyle="#7fb2ff";x.lineWidth=2;x.lineJoin="round";x.beginPath();
  series.forEach((p,i)=> i?x.lineTo(X(i),Y(p.risk)):x.moveTo(X(i),Y(p.risk)));x.stroke();
  x.fillStyle="#8fa3b5";x.textAlign="left";x.textBaseline="alphabetic";
  x.fillText("max consumer risk per frame (dashed = HIGH RISK threshold)",PL,h-3);
}
addEventListener("resize",draw); tick(); setInterval(tick,1000);
</script></body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/api/live")
def live():
    from gridtrace import infer_state
    return jsonify({
        "readings": store.latest_per_consumer(),
        "alerts": store.recent_alerts(12),
        "counts": store.counts(),
        "series": store.risk_series(150),
        "localization": infer_state.read_localization(),
    })


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "db": store.path})


def main(argv=None):
    global store
    ap = argparse.ArgumentParser(description="GridTrace AI local dashboard.")
    ap.add_argument("--host", default=config.DASH_HOST)
    ap.add_argument("--port", type=int, default=config.DASH_PORT)
    a = ap.parse_args(argv)
    store = Store()
    print(f"dashboard -> http://{a.host}:{a.port}")
    app.run(host=a.host, port=a.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
