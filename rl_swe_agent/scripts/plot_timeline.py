"""The timeline of a rollout run, as one self-contained HTML page (like Ray's dashboard).

    uv run python scripts/plot_timeline.py outputs/<date>/<time>   # writes timeline.html there

Charts on one time axis:
- the episodes, one row per worker thread of the pool (or per episode), coloured by phase:
  sandbox start, each model call split into its wait for the batch, its prompt reading
  (prefill) and its token generation (decode), the command in the sandbox, the verification;
- concurrency: episodes running, and sequences in each batched decoding step;
- throughput: tokens generated per second, all sequences together;
- memory: MLX's buffers, this process, the whole system.
Hover a bar or a curve for the values.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from rl_swe_agent.rollout.runner import read_episodes

PAGE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rollout timeline</title>
<style>
:root {
  color-scheme: light;
  --surface: #ffffff; --text-primary: #191919; --text-secondary: #666666;
  --grid: #e6e6e6; --axis: #b3b3b3;
  --series-1: #257de0; --series-2: #f47230; --series-3: #00b894; --series-4: #f2a600;
  --prefill: #ee7fac; --queue: #b3b3b3;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface: #191919; --text-primary: #efefef; --text-secondary: #9c9c9c;
    --grid: #2e2e2e; --axis: #555555;
    --series-1: #338ced; --series-2: #e76624; --series-3: #00a888; --series-4: #d28e05;
    --prefill: #e24d8b; --queue: #555555;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #191919; --text-primary: #efefef; --text-secondary: #9c9c9c;
  --grid: #2e2e2e; --axis: #555555;
  --series-1: #338ced; --series-2: #e76624; --series-3: #00a888; --series-4: #d28e05;
  --prefill: #e24d8b; --queue: #555555;
}
body { margin: 0; background: var(--surface); color: var(--text-primary);
  font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1400px; margin: 0 auto; padding: 16px; }
h1 { font-size: 18px; margin: 0 0 4px; }
h2 { font-size: 14px; margin: 20px 0 4px; }
.sub, .note { color: var(--text-secondary); }
.controls { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin: 12px 0; }
.legend { display: flex; flex-wrap: wrap; gap: 12px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.line-swatch { width: 14px; height: 2px; display: inline-block; }
button { font: inherit; color: var(--text-primary); background: transparent;
  border: 1px solid var(--axis); border-radius: 6px; padding: 3px 10px; cursor: pointer; }
button[aria-pressed="true"] { border-color: var(--text-primary); font-weight: 600; }
svg { display: block; width: 100%; }
svg text { fill: var(--text-secondary); font-size: 11px; }
#tip { position: fixed; pointer-events: none; background: var(--surface);
  color: var(--text-primary); border: 1px solid var(--axis); border-radius: 6px;
  padding: 6px 8px; font-size: 12px; display: none; white-space: pre; z-index: 2;
  box-shadow: 0 2px 8px rgba(0,0,0,.15); }
</style>
</head>
<body>
<main>
<h1>Rollout timeline</h1>
<div class="sub" id="summary"></div>
<div class="controls">
  <div>Lignes :
    <button id="by-worker" aria-pressed="true">par thread</button>
    <button id="by-episode" aria-pressed="false">par épisode</button>
  </div>
  <div>Zoom : glisser sur un graphique · <button id="unzoom">vue complète</button></div>
  <div class="legend" id="phase-legend"></div>
</div>
<svg id="gantt"></svg>
<h2>Concurrence</h2>
<div class="legend" id="concurrency-legend"></div>
<svg id="concurrency"></svg>
<h2>Débit (tokens générés par seconde, toutes séquences)</h2>
<svg id="throughput"></svg>
<h2>Mémoire (Go)</h2>
<div class="legend" id="memory-legend"></div>
<svg id="memory"></svg>
<p class="note">Un appel au modèle : attente (la demande attend que le thread de génération
l'ajoute au batch), lecture du prompt (prefill, entre deux pas de génération),
génération des tokens (chaque pas partagé avec toutes les séquences du batch). Une ligne « thread »
montre les épisodes qu'un même thread du pool a exécutés l'un après l'autre ; tous les threads
sont dans un seul processus Python, celui qui détient le modèle.</p>
</main>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const PHASES = [
  ["decode", "génération des tokens", "--series-1"],
  ["prefill", "lecture du prompt", "--prefill"],
  ["queue", "attente du batch", "--queue"],
  ["command", "commande (sandbox)", "--series-2"],
  ["sandbox", "démarrage du sandbox", "--series-3"],
  ["verify", "vérification", "--series-4"],
  ["generation", "appel au modèle", "--series-1"],
];
const color = Object.fromEntries(PHASES.map(([k, , v]) => [k, `var(${v})`]));
const label = Object.fromEntries(PHASES.map(([k, l]) => [k, l]));
const t0 = Math.min(...DATA.episodes.map(e => e.start));
const t1 = Math.max(...DATA.episodes.map(e => e.end), ...DATA.samples.map(s => s.time));
const LEFT = 120, RIGHT = 16, ROW = 16, GAP = 2;
const tip = document.getElementById("tip");
const NS = "http://www.w3.org/2000/svg";
let mode = "worker";
let v0 = t0, v1 = t1;  // the time range shown (zoom); timeline.html#zoom=120,140 opens on it
const hash = location.hash.match(/zoom=([\\d.]+),([\\d.]+)/);
if (hash) [v0, v1] = [t0 + Number(hash[1]), t0 + Number(hash[2])];

function el(name, attrs, parent) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  parent.appendChild(node);
  return node;
}
function showTip(event, text) {
  tip.textContent = text;
  tip.style.display = "block";
  const x = Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  tip.style.left = x + "px";
  tip.style.top = (event.clientY + 14) + "px";
}
function hideTip() { tip.style.display = "none"; }
function scaleX(width) {
  return t => LEFT + (t - v0) / (v1 - v0) * (width - LEFT - RIGHT);
}
function timeAt(px, width) {
  return v0 + (px - LEFT) / (width - LEFT - RIGHT) * (v1 - v0);
}
function timeAxis(svg, x, width, top, bottom) {
  const span = v1 - v0;
  const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  const step = steps.find(s => span / s <= 10) || 1200;
  const digits = step < 1 ? 1 : 0;
  for (let s = Math.ceil((v0 - t0) / step) * step; s <= v1 - t0; s += step) {
    el("line", {x1: x(t0 + s), x2: x(t0 + s), y1: top, y2: bottom, stroke: "var(--grid)"}, svg);
    el("text", {x: x(t0 + s), y: bottom + 14, "text-anchor": "middle"}, svg)
      .textContent = s.toFixed(digits) + " s";
  }
}
// the data of a chart, clipped to its plot area (zoom)
function plotArea(svg, width, height) {
  const id = svg.id + "-clip";
  const clip = el("clipPath", {id}, el("defs", {}, svg));
  el("rect", {x: LEFT, y: 0, width: width - LEFT - RIGHT, height}, clip);
  return el("g", {"clip-path": `url(#${id})`}, svg);
}
// drag across a chart to zoom on that time range
function zoomable(svg, width, height) {
  let from = null;
  const band = el("rect", {y: 0, height, fill: "var(--axis)", opacity: 0.25,
    visibility: "hidden", "pointer-events": "none"}, svg);
  const px = ev => ev.clientX - svg.getBoundingClientRect().left;
  svg.addEventListener("mousedown", ev => { from = px(ev); ev.preventDefault(); });
  svg.addEventListener("mousemove", ev => {
    if (from === null) return;
    const to = px(ev);
    band.setAttribute("x", Math.min(from, to));
    band.setAttribute("width", Math.abs(to - from));
    band.setAttribute("visibility", "visible");
  });
  svg.addEventListener("mouseup", ev => {
    const to = px(ev);
    if (from !== null && Math.abs(to - from) > 5) {
      const [a, b] = [timeAt(Math.min(from, to), width), timeAt(Math.max(from, to), width)];
      [v0, v1] = [Math.max(a, t0), Math.min(b, t1)];
      draw();
    }
    from = null;
    band.setAttribute("visibility", "hidden");
  });
}

function legend(id, items) {
  const box = document.getElementById(id);
  box.innerHTML = "";
  for (const [name, css, line] of items) {
    const span = document.createElement("span");
    span.innerHTML = `<i class="${line ? "line-swatch" : "swatch"}" style="background:${css}"></i>`;
    span.append(name);
    box.appendChild(span);
  }
}

function drawGantt() {
  const svg = document.getElementById("gantt");
  svg.innerHTML = "";
  const width = svg.clientWidth;
  const x = scaleX(width);
  const rows = mode === "worker"
    ? [...new Set(DATA.episodes.map(e => e.worker))].sort((a, b) =>
        a.localeCompare(b, undefined, {numeric: true}))
    : DATA.episodes.map(e => e.key);
  const height = rows.length * (ROW + GAP) + 24;
  svg.setAttribute("height", height);
  timeAxis(svg, x, width, 0, height - 24);
  const area = plotArea(svg, width, height - 24);
  rows.forEach((row, i) => {
    const name = mode === "worker" ? row.replace("episode_", "thread ") : row;
    el("text", {x: LEFT - 8, y: i * (ROW + GAP) + ROW - 4, "text-anchor": "end"}, svg)
      .textContent = name;
  });
  for (const e of DATA.episodes) {
    const i = rows.indexOf(mode === "worker" ? e.worker : e.key);
    const y = i * (ROW + GAP);
    const split = e.phases.some(p => p[0] === "decode");  // else one bar per model call
    for (const [phase, s, end] of e.phases) {
      if (split && phase === "generation") continue;
      // a prefill takes ~0.2 s: often less than a pixel, drawn 3 px wide (exact time on hover)
      const w = Math.max(x(end) - x(s), phase === "prefill" ? 3 : 1);
      const bar = el("rect", {x: x(s), y, width: w, height: ROW, rx: Math.min(4, w / 2),
        fill: color[phase]}, area);
      const text = `${e.key}  (${e.worker})\\n${label[phase]} : ${(end - s).toFixed(1)} s` +
        `\\nde ${(s - t0).toFixed(1)} s à ${(end - t0).toFixed(1)} s` +
        `\\nfin : ${e.exit}, tests ${e.passed}/10, récompense ${e.reward}`;
      bar.addEventListener("mousemove", ev => showTip(ev, text));
      bar.addEventListener("mouseleave", hideTip);
    }
    // each episode outlined: on a worker's row, where one ends and the next starts
    el("rect", {x: x(e.start) + 1, y: y + 1, width: Math.max(x(e.end) - x(e.start) - 2, 1),
      height: ROW - 2, rx: 4, fill: "none", stroke: "var(--text-primary)",
      "stroke-width": 1.5, "pointer-events": "none"}, area);
  }
  // prefills last: on top of the decode bars around them
  for (const bar of [...area.querySelectorAll("rect")].filter(r =>
    r.getAttribute("fill") === color.prefill)) area.appendChild(bar);
  zoomable(svg, width, height - 24);
}

function stepCounts(intervals) {
  const events = intervals.flatMap(([s, e]) => [[s, 1], [e, -1]]).sort((a, b) => a[0] - b[0]);
  let n = 0;
  const points = [[t0, 0]];
  for (const [t, d] of events) { points.push([t, n]); n += d; points.push([t, n]); }
  points.push([t1, n]);
  return points;
}
function valueAt(points, t) {
  let v = points[0][1];
  for (const [pt, pv] of points) { if (pt > t) break; v = pv; }
  return v;
}

function drawLines(id, series, unit, digits) {
  const svg = document.getElementById(id);
  svg.innerHTML = "";
  const width = svg.clientWidth, height = 160, bottom = height - 24;
  svg.setAttribute("height", height);
  const x = scaleX(width);
  const top = Math.max(...series.flatMap(s => s.points.map(p => p[1])), 1);
  const y = v => bottom - v / top * (bottom - 8);
  timeAxis(svg, x, width, 0, bottom);
  for (const v of [0, top / 2, top]) {
    el("line", {x1: LEFT, x2: width - RIGHT, y1: y(v), y2: y(v), stroke: "var(--grid)"}, svg);
    el("text", {x: LEFT - 8, y: y(v) + 4, "text-anchor": "end"}, svg)
      .textContent = v.toFixed(digits);
  }
  const area = plotArea(svg, width, bottom);
  for (const s of series) {
    const d = s.points.map(([t, v], i) => `${i ? "L" : "M"}${x(t)},${y(v)}`).join("");
    el("path", {d, fill: "none", stroke: s.color, "stroke-width": 2}, area);
  }
  const cross = el("line", {y1: 0, y2: bottom, stroke: "var(--axis)", visibility: "hidden"}, svg);
  const hit = el("rect", {x: LEFT, y: 0, width: width - LEFT - RIGHT, height: bottom,
    fill: "transparent"}, svg);
  hit.addEventListener("mousemove", ev => {
    const box = svg.getBoundingClientRect();
    const px = ev.clientX - box.left;
    const t = timeAt(px, width);
    cross.setAttribute("x1", px); cross.setAttribute("x2", px);
    cross.setAttribute("visibility", "visible");
    const lines = series.map(s => `${s.name} : ${valueAt(s.points, t).toFixed(digits)} ${unit}`);
    showTip(ev, `${(t - t0).toFixed(1)} s\\n` + lines.join("\\n"));
  });
  zoomable(svg, width, bottom);
  hit.addEventListener("mouseleave", () => {
    cross.setAttribute("visibility", "hidden");
    hideTip();
  });
}

function draw() {
  drawGantt();
  const running = stepCounts(DATA.episodes.map(e => [e.start, e.end]));
  const batch = [[t0, 0], ...DATA.batch.map(([t, n]) => [t, n]), [t1, 0]];
  const concurrency = [
    {name: "épisodes en cours", color: "var(--text-primary)", points: running},
    {name: "séquences dans le batch (par pas de génération)", color: "var(--series-1)",
     points: batch},
  ];
  legend("concurrency-legend", concurrency.map(s => [s.name, s.color, true]));
  drawLines("concurrency", concurrency, "", 0);
  const bins = new Map();
  for (const [t, n] of DATA.batch) {
    const b = Math.floor(t - t0);
    bins.set(b, (bins.get(b) || 0) + n);
  }
  const seconds = Math.ceil(t1 - t0);
  const rate = Array.from({length: seconds + 1}, (_, b) => [t0 + b, bins.get(b) || 0]);
  drawLines("throughput", [{name: "débit", color: "var(--series-1)", points: rate}],
    "tokens/s", 0);
  const memory = [
    ["MLX (poids, caches, activations)", "--series-1", "mlx_gb"],
    ["processus Python", "--series-2", "process_gb"],
    ["système entier", "--series-3", "system_used_gb"],
  ].map(([name, css, key]) => ({name, color: `var(${css})`,
    points: DATA.samples.map(s => [s.time, s[key]])}));
  legend("memory-legend", memory.map(s => [s.name, s.color, true]));
  if (DATA.samples.length) drawLines("memory", memory, "Go", 1);
}

const used = new Set(DATA.episodes.flatMap(e => {
  const split = e.phases.some(p => p[0] === "decode");
  return e.phases.map(p => p[0]).filter(k => !(split && k === "generation"));
}));
legend("phase-legend", PHASES.filter(([k]) => used.has(k))
  .map(([, name, css]) => [name, `var(${css})`, false]));
for (const [id, m] of [["by-worker", "worker"], ["by-episode", "episode"]]) {
  document.getElementById(id).addEventListener("click", () => {
    mode = m;
    document.getElementById("by-worker").setAttribute("aria-pressed", m === "worker");
    document.getElementById("by-episode").setAttribute("aria-pressed", m === "episode");
    drawGantt();
  });
}
const workers = new Set(DATA.episodes.map(e => e.worker)).size;
const peak = DATA.samples.length ? Math.max(...DATA.samples.map(s => s.mlx_gb)).toFixed(1) : "?";
const tokens = DATA.batch.reduce((sum, [, n]) => sum + n, 0);
document.getElementById("summary").textContent =
  `${DATA.episodes.length} épisodes, ${workers} threads, ${(t1 - t0).toFixed(0)} s au total, ` +
  `${tokens} tokens générés (${(tokens / (t1 - t0)).toFixed(0)} tokens/s en moyenne), ` +
  `pic mémoire MLX ${peak} Go`;
document.getElementById("unzoom").addEventListener("click", () => {
  [v0, v1] = [t0, t1];
  draw();
});
window.addEventListener("resize", draw);
draw();
</script>
</body>
</html>
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="run directory (scripts/rollout.py)")
    args = parser.parse_args()
    episodes = list(read_episodes(args.run / "episodes.jsonl"))
    tasks = list(dict.fromkeys(e.task_id for e in episodes))
    rollout: dict[str, int] = {}
    records = []
    for e in episodes:
        rollout[e.task_id] = rollout.get(e.task_id, -1) + 1
        records.append(
            {
                "key": f"tâche {tasks.index(e.task_id)} · rollout {rollout[e.task_id]}",
                "worker": e.worker or "main",
                "start": e.started,
                "end": e.started + e.seconds,
                "phases": e.timeline,
                "exit": e.exit_status,
                "passed": e.verdict.passed,
                "reward": e.reward,
            }
        )
    data = json.dumps(
        {
            "episodes": records,
            "samples": read_jsonl(args.run / "resources.jsonl"),
            "batch": [[s["time"], s["sequences"]] for s in read_jsonl(args.run / "batch.jsonl")],
        }
    )
    out = args.run / "timeline.html"
    out.write_text(PAGE.replace("__DATA__", data))
    print(out)


if __name__ == "__main__":
    main()
