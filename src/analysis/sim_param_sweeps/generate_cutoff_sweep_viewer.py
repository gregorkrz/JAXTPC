#!/usr/bin/env python
"""Generate a self-contained interactive HTML viewer for the Adam_20260601_cutoff_sweep results.

Sweeps: ADC cutoffs (5, 10, 20, 50) × Fourier cutoffs (0, 10, 100) × rotate-noise (off/5)
Each grid cell shows mean ± std of relative error at final step across seeds 100-104.

Usage (on S3DF):
    /sdf/home/g/gregork/envs/base_env/bin/python \
        src/analysis/sim_param_sweeps/generate_cutoff_sweep_viewer.py \
        --results-dir $RESULTS_DIR \
        --output plots/cutoff_sweep_viewer.html
"""

import argparse
import json
import math
import pickle
import sys
from pathlib import Path


ADC_CUTOFFS      = [5, 10, 20, 50]
FFT_CUTOFFS      = [0, 10, 100]
ROTATE_NOISE_VALS = [None, 5]   # None = disabled
SEEDS            = [100, 101, 102, 103, 104]
PARAM_LABEL      = "trans_and_long"
LOG_INTERVAL     = 50

PARAM_PRETTY = {
    'diffusion_trans_cm2_us': 'D⊥ (cm²/μs)',
    'diffusion_long_cm2_us':  'D∥ (cm²/μs)',
}
PARAM_SHORT = {
    'diffusion_trans_cm2_us': 'D⊥',
    'diffusion_long_cm2_us':  'D∥',
}

SEED_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def profile_tag(adc, fft, rot):
    adc_tag = f"adc{int(adc)}"
    fft_tag = f"ft{int(fft)}"
    rot_tag = "rotoff" if rot is None else f"rot{rot}"
    return f"Adam_20260601_cutoff_sweep_{PARAM_LABEL}_{adc_tag}_{fft_tag}_{rot_tag}"


def load_seed_result(results_dir: Path, adc, fft, rot, seed):
    tag     = profile_tag(adc, fft, rot)
    base    = results_dir / "opt" / tag / "noise"
    pattern = f"*/result_{seed}.pkl"
    matches = list(base.glob(pattern))
    if not matches:
        return None
    path = matches[0]
    with open(path, "rb") as f:
        return pickle.load(f)


def rel_err_trajectory(param_traj, p_n_gts):
    """Return rel_err[step][param] = |exp(p_n - p_n_gt) - 1|."""
    result = []
    for pns in param_traj:
        row = []
        for i, (pn, pn_gt) in enumerate(zip(pns, p_n_gts)):
            row.append(abs(math.exp(pn - pn_gt) - 1.0))
        result.append(row)
    return result


def phys_trajectory(param_traj, scales):
    """Return phys[step][param] = scale * exp(p_n)."""
    result = []
    for pns in param_traj:
        row = []
        for pn, sc in zip(pns, scales):
            row.append(sc * math.exp(pn))
        result.append(row)
    return result


def collect_data(results_dir: Path):
    cells = {}
    any_loaded = False
    for rot in ROTATE_NOISE_VALS:
        for adc in ADC_CUTOFFS:
            for fft in FFT_CUTOFFS:
                key = f"{'null' if rot is None else rot}|{adc}|{fft}"
                seed_data = []
                for seed in SEEDS:
                    res = load_seed_result(results_dir, adc, fft, rot, seed)
                    if res is None:
                        print(f"  MISSING: {profile_tag(adc, fft, rot)} seed {seed}", file=sys.stderr)
                        seed_data.append(None)
                        continue
                    any_loaded = True
                    trials = res.get("trials", [])
                    p_n_gts = res["p_n_gts"]
                    scales  = res["scales"]

                    if not trials:
                        # Job still running — use live_checkpoint for a single-point snapshot
                        ckpt = res.get("live_checkpoint")
                        if ckpt is None:
                            # pkl exists but job preempted before any checkpoint — show as pending
                            print(f"  PENDING (no ckpt): {profile_tag(adc, fft, rot)} seed {seed}", file=sys.stderr)
                            seed_data.append({
                                "seed":          seed,
                                "steps_run":     0,
                                "stopped_early": False,
                                "in_progress":   True,
                                "step_idxs":     [],
                                "phys_traj":     [],
                                "rel_err_traj":  [],
                                "p_n_gts":       list(p_n_gts),
                                "scales":        list(scales),
                                "param_gts":     list(res["param_gts"]),
                                "param_names":   list(res["param_names"]),
                            })
                            continue
                        step   = ckpt["step"]
                        p_cur  = ckpt["p"][:len(p_n_gts)]   # strip any MLP tail
                        p_traj = [p_cur]
                        step_idxs = [step]
                        print(f"  IN PROGRESS @ step {step}: {profile_tag(adc, fft, rot)} seed {seed}", file=sys.stderr)
                        seed_data.append({
                            "seed":          seed,
                            "steps_run":     step,
                            "stopped_early": False,
                            "in_progress":   True,
                            "step_idxs":     step_idxs,
                            "phys_traj":     phys_trajectory(p_traj, scales),
                            "rel_err_traj":  rel_err_trajectory(p_traj, p_n_gts),
                            "p_n_gts":       list(p_n_gts),
                            "scales":        list(scales),
                            "param_gts":     list(res["param_gts"]),
                            "param_names":   list(res["param_names"]),
                        })
                        continue

                    trial      = trials[0]
                    p_traj     = trial["param_trajectory"]
                    steps_run  = trial["steps_run"]
                    stopped_early = trial.get("stopped_early", False)
                    # wandb-patched trials store explicit step indices; pkl trials record
                    # every gradient step so the index IS the step number.
                    if "step_indices" in trial:
                        step_idxs = trial["step_indices"]
                    else:
                        step_idxs = list(range(len(p_traj)))
                    seed_data.append({
                        "seed":          seed,
                        "steps_run":     steps_run,
                        "stopped_early": stopped_early,
                        "in_progress":   False,
                        "step_idxs":     step_idxs,
                        "phys_traj":     phys_trajectory(p_traj, scales),
                        "rel_err_traj":  rel_err_trajectory(p_traj, p_n_gts),
                        "p_n_gts":       list(p_n_gts),
                        "scales":        list(scales),
                        "param_gts":     list(res["param_gts"]),
                        "param_names":   list(res["param_names"]),
                    })
                cells[key] = {
                    "adc": adc,
                    "fft": fft,
                    "rot": rot,
                    "seeds": seed_data,
                }
    if not any_loaded:
        print("WARNING: no pkl files found — check --results-dir path", file=sys.stderr)
    return cells


def build_payload(cells):
    """Flatten cells into a JSON-serialisable dict, trimming numpy arrays."""
    payload = {
        "adc_cutoffs":       ADC_CUTOFFS,
        "fft_cutoffs":       FFT_CUTOFFS,
        "rotate_noise_vals": ROTATE_NOISE_VALS,
        "seeds":             SEEDS,
        "cells":             {},
    }
    param_names = None
    param_gts   = None
    scales      = None

    for key, cell in cells.items():
        seeds_out = []
        for sd in cell["seeds"]:
            if sd is None:
                seeds_out.append(None)
                continue
            if param_names is None:
                param_names = sd["param_names"]
                param_gts   = sd["param_gts"]
                scales      = sd["scales"]
            seeds_out.append({
                "seed":          sd["seed"],
                "steps_run":     sd["steps_run"],
                "stopped_early": sd["stopped_early"],
                "in_progress":   sd.get("in_progress", False),
                "step_idxs":     sd["step_idxs"],
                "phys_traj":     sd["phys_traj"],
                "rel_err_traj":  sd["rel_err_traj"],
            })
        payload["cells"][key] = {
            "adc":   cell["adc"],
            "fft":   cell["fft"],
            "rot":   cell["rot"],
            "seeds": seeds_out,
        }

    payload["param_names"] = param_names or ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
    payload["param_gts"]   = param_gts   or [1e-5, 1e-5]
    payload["scales"]      = scales      or [1e-5, 1e-5]
    return payload


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cutoff Sweep — Diffusion Calibration</title>
<style>
*, *::before, *::after { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; padding: 16px; background: #f4f5f7; color: #222; }
h1 { margin: 0 0 12px; font-size: 1.2rem; }
.controls { display: flex; gap: 16px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }
.controls label { font-weight: 600; }
select { font-size: 0.95rem; padding: 4px 8px; border-radius: 4px; border: 1px solid #bbb; }

#grid-wrap { background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.12); display: inline-block; }
table.grid { border-collapse: collapse; }
table.grid th { background: #e8eaed; padding: 8px 14px; font-size: 0.85rem; text-align: center; }
table.grid td { border: 2px solid #fff; padding: 0; cursor: pointer; width: 120px; height: 64px; text-align: center; vertical-align: middle; transition: outline .1s; }
table.grid td:hover { outline: 3px solid #333; }
table.grid td.selected { outline: 3px solid #0057b7; }
.cell-inner { pointer-events: none; font-size: 0.82rem; line-height: 1.35; padding: 4px; }
.cell-inner .mean { font-weight: 700; font-size: 0.95rem; }
.cell-inner .std  { font-size: 0.77rem; color: #444; }
.cell-inner .miss { font-size: 0.78rem; color: #888; font-style: italic; }

#detail { margin-top: 20px; display: none; }
.detail-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.detail-header h2 { margin: 0; font-size: 1rem; }
.close-btn { cursor: pointer; font-size: 1.3rem; line-height: 1; color: #555; background: none; border: none; padding: 0 4px; }
.charts-row { display: flex; gap: 12px; flex-wrap: wrap; }
.chart-box { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.12); padding: 12px 14px; }
.chart-box h3 { margin: 0 0 8px; font-size: 0.88rem; color: #555; }
.plotly-chart { border-radius: 8px; overflow: hidden; }
</style>
</head>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<body>
<h1>Cutoff Sweep — Diffusion Calibration (Adam_20260601)</h1>

<div class="controls">
  <div><label>Noise cycles (--rotate-noise-seeds):&nbsp;</label>
    <select id="rot-sel">
      <option value="null">Disabled</option>
      <option value="5">5</option>
    </select>
  </div>
  <div><label>Grid metric (@ step 4000):&nbsp;</label>
    <select id="param-sel"></select>
  </div>
  <div id="colorscale-legend" style="display:flex;align-items:center;gap:8px;font-size:.82rem;">
    <span>Low error</span>
    <canvas id="cs-canvas" width="160" height="18" style="border-radius:3px;"></canvas>
    <span>High error</span>
  </div>
</div>

<div id="grid-wrap">
  <table class="grid" id="grid-table"></table>
</div>

<div id="detail">
  <div class="detail-header">
    <h2 id="detail-title"></h2>
    <button class="close-btn" onclick="closeDetail()">✕</button>
  </div>
  <div class="charts-row" id="charts-row"></div>
</div>

<script>
const DATA = __DATA_JSON__;

const PARAM_PRETTY = {
  'diffusion_trans_cm2_us': 'D⊥ (cm²/μs)',
  'diffusion_long_cm2_us':  'D∥ (cm²/μs)',
};
const PARAM_SHORT = {
  'diffusion_trans_cm2_us': 'D⊥',
  'diffusion_long_cm2_us':  'D∥',
};
const SEED_COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd'];
const SUMMARY_STEP = 4000;  // step used for grid summary stats

let selectedCell = null;

// ── Colour helpers ────────────────────────────────────────────────────────────

function errToHue(relErr, maxErr) {
  const t = Math.min(relErr / maxErr, 1.0);
  const hue = 120 * (1 - t);   // 120 = green, 0 = red
  return `hsl(${hue.toFixed(1)}, 80%, 52%)`;
}

function drawColorScale() {
  const c = document.getElementById('cs-canvas');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  for (let x = 0; x < W; x++) {
    const t = x / (W - 1);
    const hue = 120 * (1 - t);
    ctx.fillStyle = `hsl(${hue.toFixed(1)}, 80%, 52%)`;
    ctx.fillRect(x, 0, 1, H);
  }
}

// ── Grid rendering ────────────────────────────────────────────────────────────

function cellKey(rot, adc, fft) {
  return `${rot}|${adc}|${fft}`;
}

function getCellSummary(rot, adc, fft, paramIdx) {
  const key  = cellKey(rot, adc, fft);
  const cell = DATA.cells[key];
  if (!cell) return { mean: null, std: null, n: 0, n_prog: 0 };
  const errs = [];
  let n_prog = 0;
  for (const sd of cell.seeds) {
    if (!sd) continue;
    const traj = sd.rel_err_traj;
    if (!traj || traj.length === 0) continue;
    if (sd.in_progress) { n_prog++; continue; }  // exclude in-progress from summary stats
    // Use the entry at SUMMARY_STEP, or the last available if run stopped before it
    const idx = Math.min(
      sd.step_idxs.findIndex(s => s > SUMMARY_STEP) - 1,
      traj.length - 1
    );
    errs.push(traj[idx < 0 ? traj.length - 1 : idx][paramIdx]);
  }
  if (errs.length === 0) return { mean: null, std: null, n: 0, n_prog };
  const mean = errs.reduce((a,b)=>a+b,0) / errs.length;
  const variance = errs.reduce((s,v)=>s+(v-mean)**2, 0) / errs.length;
  return { mean, std: Math.sqrt(variance), n: errs.length, n_prog };
}

let GLOBAL_MAX_ERR_PER_PARAM = [];  // computed once at init; shared across rot, separate per param

function computeGlobalMaxErr() {
  const nParams = DATA.param_names.length;
  GLOBAL_MAX_ERR_PER_PARAM = Array(nParams).fill(0);
  for (const rot of DATA.rotate_noise_vals)
    for (const adc of DATA.adc_cutoffs)
      for (const fft of DATA.fft_cutoffs)
        for (let pi = 0; pi < nParams; pi++) {
          const s = getCellSummary(rot, adc, fft, pi);
          if (s.mean !== null)
            GLOBAL_MAX_ERR_PER_PARAM[pi] = Math.max(GLOBAL_MAX_ERR_PER_PARAM[pi], s.mean);
        }
  GLOBAL_MAX_ERR_PER_PARAM = GLOBAL_MAX_ERR_PER_PARAM.map(v => v || 0.5);
}

function fmtPct(v) {
  if (v === null) return '—';
  if (v < 0.01)  return (v * 100).toFixed(2) + '%';
  return (v * 100).toFixed(1) + '%';
}

function getSelectedRot() {
  const v = document.getElementById('rot-sel').value;
  return v === 'null' ? null : Number(v);
}

function getSelectedParamIdx() {
  return Number(document.getElementById('param-sel').value);
}

function buildGrid() {
  const table  = document.getElementById('grid-table');
  const rot    = getSelectedRot();
  const pIdx   = getSelectedParamIdx();
  const maxErr = GLOBAL_MAX_ERR_PER_PARAM[pIdx] || 0.5;

  table.innerHTML = '';

  // Header row
  const thead = table.createTHead();
  const hrow  = thead.insertRow();
  hrow.insertCell().innerHTML = '<th>|ADC| \\ FFT cut</th>';
  for (const fft of DATA.fft_cutoffs) {
    const th = document.createElement('th');
    th.textContent = fft === 0 ? 'No FFT cut' : `FFT ${fft} ADC²`;
    hrow.appendChild(th);
  }

  const tbody = table.createTBody();
  for (const adc of DATA.adc_cutoffs) {
    const row = tbody.insertRow();
    const lbl = document.createElement('th');
    lbl.textContent = `|ADC| ≥ ${adc}`;
    row.appendChild(lbl);

    for (const fft of DATA.fft_cutoffs) {
      const td  = row.insertCell();
      const key = cellKey(rot, adc, fft);
      td.dataset.key = key;
      if (selectedCell === key) td.classList.add('selected');

      const s = getCellSummary(rot, adc, fft, pIdx);
      const inner = document.createElement('div');
      inner.className = 'cell-inner';

      if (s.mean !== null) {
        td.style.background = errToHue(s.mean, maxErr);
        const progTag = s.n_prog > 0 ? ` <span style="opacity:.6">(+${s.n_prog}▶)</span>` : '';
        inner.innerHTML = `<div class="mean">${fmtPct(s.mean)}</div>`
                        + `<div class="std">± ${fmtPct(s.std)}</div>`
                        + `<div class="miss">${s.n}/${DATA.seeds.length} done${progTag}</div>`;
      } else if (s.n_prog > 0) {
        td.style.background = '#c8d8f0';
        inner.innerHTML = `<div class="miss">${s.n_prog} pending/running ▶</div>`;
      } else {
        td.style.background = '#ddd';
        inner.innerHTML = `<div class="miss">no data</div>`;
      }

      td.appendChild(inner);
      td.onclick = () => onCellClick(key, adc, fft, rot);
    }
  }
}

// ── Detail panel ──────────────────────────────────────────────────────────────

function onCellClick(key, adc, fft, rot) {
  selectedCell = key;
  buildGrid();  // re-render to update selection highlight

  const cell = DATA.cells[key];
  const rotLabel = rot === null ? 'disabled' : `${rot} steps`;
  document.getElementById('detail-title').textContent =
    `ADC cutoff ${adc}  ·  FFT cutoff ${fft === 0 ? 'none' : fft + ' ADC²'}  ·  noise rotation ${rotLabel}`;

  buildDetailCharts(cell);
  document.getElementById('detail').style.display = 'block';
  document.getElementById('detail').scrollIntoView({behavior:'smooth', block:'nearest'});
}

function closeDetail() {
  document.getElementById('detail').style.display = 'none';
  selectedCell = null;
  buildGrid();
}

function seedLabel(sd) {
  return `seed ${sd.seed}`;
}

const PLOTLY_LAYOUT_BASE = {
  margin: { l:60, r:20, t:36, b:50 },
  height: 240,
  width:  500,
  paper_bgcolor: '#fff',
  plot_bgcolor:  '#f9f9f9',
  xaxis: { title: 'Adam step', range: [0, SUMMARY_STEP], gridcolor: '#e8e8e8' },
  showlegend: true,
  legend: { font: { size: 10 }, orientation: 'v' },
  hovermode: 'x unified',
};
const PLOTLY_CONFIG = { displayModeBar: false, responsive: false };

function buildDetailCharts(cell) {
  const row = document.getElementById('charts-row');
  row.innerHTML = '';

  const pNames = DATA.param_names;
  const pGts   = DATA.param_gts;

  // Clamp trajectories to SUMMARY_STEP
  function clamp(xs, ys) {
    const cutIdx = xs.findIndex(x => x > SUMMARY_STEP);
    if (cutIdx === -1) return { xs, ys };
    return { xs: xs.slice(0, cutIdx), ys: ys.slice(0, cutIdx) };
  }

  // ── 1. Param value vs step (one chart per param) ──────────────────────────
  for (let pi = 0; pi < pNames.length; pi++) {
    const gt  = pGts[pi];
    const lbl = PARAM_PRETTY[pNames[pi]] || pNames[pi];

    const traces = [];
    for (let si = 0; si < cell.seeds.length; si++) {
      const sd = cell.seeds[si];
      if (!sd || sd.phys_traj.length === 0) continue;
      const { xs, ys } = clamp(sd.step_idxs, sd.phys_traj.map(r => r[pi]));
      traces.push({
        x: xs, y: ys,
        name: seedLabel(sd),
        mode: 'lines',
        line: { color: SEED_COLORS[si % SEED_COLORS.length], width: 1.8 },
      });
    }
    // GT as horizontal dashed line via shape
    const layout = Object.assign({}, PLOTLY_LAYOUT_BASE, {
      title: { text: lbl + ' vs step', font: { size: 13 } },
      yaxis: { title: lbl, tickformat: '.2e', gridcolor: '#e8e8e8' },
      shapes: [{ type:'line', x0:0, x1:SUMMARY_STEP, y0:gt, y1:gt,
                 line:{ color:'#888', width:1.5, dash:'dash' } }],
      annotations: [{ x:SUMMARY_STEP, y:gt, xanchor:'right', yanchor:'bottom',
                      text:`GT: ${fmt_si(gt)}`, showarrow:false,
                      font:{ size:10, color:'#888' } }],
    });

    const div = document.createElement('div');
    div.className = 'plotly-chart';
    row.appendChild(div);
    Plotly.newPlot(div, traces, layout, PLOTLY_CONFIG);
  }

  // ── 2. Relative error vs step (one chart per param) ───────────────────────
  for (let pi = 0; pi < pNames.length; pi++) {
    const lbl = PARAM_SHORT[pNames[pi]] || pNames[pi];

    const traces = [];
    for (let si = 0; si < cell.seeds.length; si++) {
      const sd = cell.seeds[si];
      if (!sd || sd.rel_err_traj.length === 0) continue;
      const { xs, ys } = clamp(sd.step_idxs, sd.rel_err_traj.map(r => r[pi]));
      traces.push({
        x: xs, y: ys,
        name: seedLabel(sd),
        mode: 'lines',
        line: { color: SEED_COLORS[si % SEED_COLORS.length], width: 1.8 },
        hovertemplate: `step %{x}<br>err: %{y:.2%}<extra>${seedLabel(sd)}</extra>`,
      });
    }
    const layout = Object.assign({}, PLOTLY_LAYOUT_BASE, {
      title: { text: `Rel. error ${lbl} vs step`, font: { size: 13 } },
      yaxis: { title: 'Relative error', tickformat: '.1%', gridcolor: '#e8e8e8' },
    });

    const div = document.createElement('div');
    div.className = 'plotly-chart';
    row.appendChild(div);
    Plotly.newPlot(div, traces, layout, PLOTLY_CONFIG);
  }
}

function fmt_si(v) {
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs >= 1e-3 && abs < 1e4) {
    const d = abs < 0.1 ? 4 : abs < 1 ? 3 : abs < 100 ? 2 : 1;
    return v.toFixed(d);
  }
  return v.toExponential(2);
}


// ── Init ──────────────────────────────────────────────────────────────────────

function init() {
  // Populate param dropdown
  const pSel = document.getElementById('param-sel');
  for (let i = 0; i < DATA.param_names.length; i++) {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = PARAM_PRETTY[DATA.param_names[i]] || DATA.param_names[i];
    pSel.appendChild(opt);
  }
  pSel.onchange = buildGrid;
  document.getElementById('rot-sel').onchange = () => {
    closeDetail();
    buildGrid();
  };
  computeGlobalMaxErr();
  drawColorScale();
  buildGrid();
}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""


def generate_html(payload: dict) -> str:
    data_json = json.dumps(payload, separators=(',', ':'))
    return HTML_TEMPLATE.replace('__DATA_JSON__', data_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', required=True,
                        help='Root results directory (e.g. $RESULTS_DIR)')
    parser.add_argument('--output', required=True,
                        help='Output HTML path')
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    if not results_dir.is_dir():
        print(f"ERROR: results-dir does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print("Loading pkl files...", file=sys.stderr)
    cells   = collect_data(results_dir)
    payload = build_payload(cells)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = generate_html(payload)
    out_path.write_text(html, encoding='utf-8')
    print(f"Written: {out_path}  ({len(html) // 1024} KB)", file=sys.stderr)


if __name__ == '__main__':
    main()
