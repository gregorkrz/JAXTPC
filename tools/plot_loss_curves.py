#!/usr/bin/env python3
"""
Generate an interactive HTML loss-curve viewer from result PKL files.

Each run gets a coloured trace on three stacked plots (total loss, Sobolev
loss, E-field curl/rotor loss). A run-checkbox sidebar lets you overlay any
combination of runs.

Loss sources
------------
* Total loss  — read from PKL ``trial['loss_trajectory']`` (always present
  for completed runs), or from W&B key ``loss``.
* Sobolev loss, Rotor (unweighted curl) loss — fetched from W&B keys
  ``loss/sobolev`` and ``loss/rotor_unweighted``.  These are logged for all
  E-field optimisation runs (even rot=0 runs, where sobolev==total and
  rotor is the unpenalised curl).  Falls back to showing only total loss
  when W&B credentials are unavailable.

Usage
-----
  python tools/plot_loss_curves.py --results-dir $RESULTS_DIR/opt/...
  python tools/plot_loss_curves.py /path/to/result_*.pkl --output out.html
  python tools/plot_loss_curves.py dir1 dir2 --output out.html
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT))


# ── Sweep parameter extraction (shared with plot_efield_eval.py) ──────────────

_PARAM_PATTERNS = [
    ('sce_map',  r'/sce_maps_([^/]+)/'),
    ('arch',     r'[/_]arch([\dx]+)[/_]'),
    ('n_tracks', r'[/_](\d+)tracks[/_]'),
    ('trk_seed', r'[/_]trk(\d+)[/_]'),
    ('nn',       r'[/_]nn(\d+)[/_]'),
    ('dropout',  r'[/_]do([\d.]+)[/_]'),
    ('curl_w',   r'[/_]rot(-?[\d.]+)[/_]'),
    ('N',        r'_N(\d+)_'),
    ('noise',    r'[/_]noise([\d.]+)[/_]'),
    ('steps',    r'_s(\d{4,})_'),
    ('lr',       r'_lr([\d.e+-]+)_'),
    ('seed',     r'/result_(\w+?)\.pkl$'),
]

_PARAM_LABELS = {
    'sce_map':  'SCE map',
    'arch':     'Architecture',
    'n_tracks': 'N tracks',
    'trk_seed': 'Track seed',
    'nn':       'NN seed',
    'dropout':  'Dropout',
    'curl_w':   'Curl weight',
    'N':        'N',
    'noise':    'Noise',
    'steps':    'Steps',
    'lr':       'LR',
    'seed':     'Seed',
}


def _parse_path_params(pkl_path):
    s = str(pkl_path)
    params = {}
    for key, pat in _PARAM_PATTERNS:
        m = re.search(pat, s)
        if m:
            params[key] = m.group(1)
    return params


def _run_label(pkl_path, params=None):
    """Short human-readable label from sweep params (or path fallback)."""
    if params:
        # Build label from most discriminating sweep params
        parts = []
        for key in ('sce_map', 'arch', 'n_tracks', 'trk_seed', 'nn', 'dropout', 'curl_w', 'seed'):
            v = params.get(key)
            if v is not None:
                parts.append(f"{key}={v}")
        if parts:
            return '  '.join(parts)
    # Fallback: use sweep-specific directory component (skip the long run-config dir)
    p = Path(pkl_path)
    parts_path = p.parts
    for i in range(len(parts_path) - 1, -1, -1):
        seg = parts_path[i]
        if '__' not in seg and any(k in seg for k in ('nn', 'rot', 'do', 'trk')):
            return f"{seg}/{p.stem}"
    parent = p.parent.name
    short_parts = parent.split('__')
    short = short_parts[-1] if len(short_parts) > 1 else parent
    return f"{short}/{p.stem}"


# ── W&B history fetch ─────────────────────────────────────────────────────────

def _fetch_wandb_losses(run_id, project='jaxtpc-optimization', samples=500):
    """Fetch loss histories from W&B.  Returns dict or None on failure."""
    try:
        import wandb
        api = wandb.Api(timeout=30)
        run = api.run(f"{project}/{run_id}")
        keys = ['_step', 'loss', 'loss/sobolev', 'loss/rotor_unweighted']
        rows = run.history(keys=keys, samples=samples, pandas=False)
        if not rows:
            return None
        def _safe(v):
            if v is None:
                return None
            try:
                f = float(v)
                return None if not np.isfinite(f) else round(f, 7)
            except (TypeError, ValueError):
                return None
        def _col(k):
            vals = [_safe(row.get(k)) for row in rows]
            return vals if any(v is not None for v in vals) else None
        return {
            'steps':   [int(row.get('_step', i)) for i, row in enumerate(rows)],
            'total':   _col('loss'),
            'sobolev': _col('loss/sobolev'),
            'rotor':   _col('loss/rotor_unweighted'),
        }
    except Exception as exc:
        print(f"  [wandb] {run_id}: {exc}")
        return None


# ── PKL loss extraction ───────────────────────────────────────────────────────

def _pkl_losses(pkl_path):
    """Extract loss trajectory from PKL trials (completed) or live_checkpoint."""
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)

    trials = result.get('trials', [])
    if trials:
        # Concatenate all trial loss trajectories end-to-end
        combined = []
        for t in trials:
            combined.extend(t.get('loss_trajectory', []))
        if combined:
            return {'steps': list(range(len(combined))),
                    'total': [round(float(v), 7) for v in combined],
                    'sobolev': None, 'rotor': None}
    return None


# ── Per-PKL record builder ────────────────────────────────────────────────────

def load_losses(pkl_path, wandb_project='jaxtpc-optimization',
                use_wandb=True, wandb_samples=500):
    pkl_path = Path(pkl_path)
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)

    params = _parse_path_params(pkl_path)
    label  = _run_label(pkl_path, params)
    run_id = result.get('wandb_run_id')

    losses = None
    if use_wandb and run_id:
        print(f"  W&B fetch  {run_id}  ({label})")
        losses = _fetch_wandb_losses(run_id, project=wandb_project,
                                     samples=wandb_samples)

    if losses is None:
        print(f"  PKL fallback  ({label})")
        losses = _pkl_losses(pkl_path)

    if losses is None:
        print(f"  [skip] no loss data in {pkl_path.name}")
        return None

    return {
        'label':      label,
        'params':     params,
        'wandb_id':   run_id,
        'steps':      losses['steps'],
        'total':      losses['total'],
        'sobolev':    losses['sobolev'],
        'rotor':      losses['rotor'],
    }


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Loss curves</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; background: #f0f0f0;
       color: #222; display: flex; height: 100vh; overflow: hidden; }
.sidebar {
  width: 280px; min-width: 220px; max-width: 340px; overflow-y: auto;
  background: #fff; padding: 12px; box-shadow: 2px 0 6px rgba(0,0,0,.1);
  display: flex; flex-direction: column; gap: 10px;
}
.sidebar h2 { margin: 0 0 6px; font-size: 14px; }
.cg { display: flex; flex-direction: column; gap: 3px; }
.cg label { font-size: 11px; color: #666; font-weight: 600; }
.cg select { padding: 4px 6px; border-radius: 5px; border: 1px solid #ccc;
             background: #fafafa; font-size: 12px; }
.run-list { flex: 1 1 auto; overflow-y: auto; min-height: 80px; }
.run-item { display: flex; align-items: flex-start; gap: 6px; padding: 3px 0;
            font-size: 11px; line-height: 1.3; border-bottom: 1px solid #f0f0f0; }
.run-item input { margin-top: 2px; flex-shrink: 0; }
.run-color { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; margin-top: 3px; }
.run-label { color: #333; word-break: break-all; flex: 1 1 auto; }
.run-link { font-size:10px; color:#5588cc; text-decoration:none; flex-shrink:0;
            padding:1px 4px; border-radius:3px; border:1px solid #cce;
            background:#f0f7ff; line-height:1.6; }
.run-link:hover { background:#ddeeff; }
.btn-row { display: flex; gap: 6px; flex-wrap: wrap; }
button { padding: 4px 10px; font-size: 11px; border-radius: 4px; border: 1px solid #bbb;
         background: #f5f5f5; cursor: pointer; }
button:hover { background: #e8e8e8; }
button.active { background: #dde8ff; border-color: #99b; }
.main { flex: 1 1 auto; overflow-y: auto; padding: 12px; display: flex;
        flex-direction: column; gap: 10px; }
.section-title { font-weight: 700; font-size: 12px; color: #555; margin: 0; }
.chart-panel { background: #fff; border-radius: 8px; padding: 8px 10px;
               box-shadow: 0 1px 3px rgba(0,0,0,.1); }
#no-data { color: #888; font-size: 13px; text-align: center; padding: 40px; }
</style>
</head>
<body>
<div class="sidebar">
  <h2>Loss curves</h2>
  <div id="filter-dropdowns"></div>
  <div style="font-size:11px;font-weight:600;color:#666;margin-top:4px">Runs</div>
  <div class="btn-row">
    <button onclick="selectAll()">All</button>
    <button onclick="selectNone()">None</button>
    <button id="log-btn" onclick="toggleLog()">Log scale</button>
  </div>
  <div class="run-list" id="run-list"></div>
</div>
<div class="main">
  <div id="no-data" style="display:none">No runs selected.</div>
  <div class="chart-panel">
    <div class="section-title">Total loss</div>
    <div id="plot-total"></div>
  </div>
  <div class="chart-panel">
    <div class="section-title">Sobolev loss</div>
    <div id="plot-sobolev"></div>
  </div>
  <div class="chart-panel">
    <div class="section-title">E-field curl (rotor) loss — unweighted</div>
    <div id="plot-rotor"></div>
  </div>
</div>
<script>
const ALL_DATA     = /*ALL_DATA*/null/*END_ALL_DATA*/;
const PARAM_KEYS   = /*PARAM_KEYS*/null/*END_PARAM_KEYS*/;
const PARAM_LABELS = /*PARAM_LABELS*/null/*END_PARAM_LABELS*/;
const COMPANION_URL = /*COMPANION_URL*/null/*END_COMPANION_URL*/;

// ── colour palette ────────────────────────────────────────────────────────────
const PALETTE = [
  '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
  '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
  '#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5',
  '#c49c94','#f7b6d2','#c7c7c7','#dbdb8d','#9edae5',
];
function color(i) { return PALETTE[i % PALETTE.length]; }

// ── state ─────────────────────────────────────────────────────────────────────
let logScale  = false;
let selected  = new Set();  // indices into ALL_DATA

// ── URL state ─────────────────────────────────────────────────────────────────
function _pushState() {
  const p = new URLSearchParams();
  if (logScale) p.set('log', '1');
  p.set('sel', [...selected].join(','));
  for (const k of PARAM_KEYS) {
    const el = document.getElementById('f_' + k);
    if (el && el.value) p.set('f_' + k, el.value);
  }
  history.replaceState(null, '', '?' + p.toString());
}
function _loadState() {
  const p = new URLSearchParams(location.search);
  logScale = p.get('log') === '1';
  if (logScale) document.getElementById('log-btn').classList.add('active');
  const selStr = p.get('sel');
  if (selStr) selStr.split(',').map(Number).forEach(i => selected.add(i));
  return p;
}

// ── companion cross-link ──────────────────────────────────────────────────────
function _companionUrl(r) {
  if (!COMPANION_URL) return null;
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(r.params || {})) {
    if (v != null) p.set('f_' + k, v);
  }
  return COMPANION_URL + '?' + p.toString();
}

// ── filter dropdowns ──────────────────────────────────────────────────────────
function _buildFilterDropdowns(urlP) {
  const container = document.getElementById('filter-dropdowns');
  container.innerHTML = '';
  for (const key of PARAM_KEYS) {
    const vals = [...new Set(ALL_DATA.map(r => r.params[key]).filter(v => v != null))].sort();
    if (vals.length < 2) continue;
    const saved = urlP ? urlP.get('f_' + key) : null;
    const div = document.createElement('div');
    div.className = 'cg';
    div.innerHTML = `<label>${PARAM_LABELS[key] || key}</label>
      <select id="f_${key}" onchange="filterRuns()">
        <option value="">— all —</option>
        ${vals.map(v => `<option value="${v}"${v===saved?' selected':''}>${v}</option>`).join('')}
      </select>`;
    container.appendChild(div);
  }
}

function _filteredIndices() {
  return ALL_DATA.map((r, i) => {
    for (const key of PARAM_KEYS) {
      const el = document.getElementById('f_' + key);
      if (el && el.value && r.params[key] !== el.value) return null;
    }
    return i;
  }).filter(i => i !== null);
}

function filterRuns() {
  const visible = new Set(_filteredIndices());
  // keep selected only within visible set
  for (const i of [...selected]) {
    if (!visible.has(i)) selected.delete(i);
  }
  _buildRunList(visible);
  _render();
  _pushState();
}

// ── run list ──────────────────────────────────────────────────────────────────
function _buildRunList(visible) {
  const container = document.getElementById('run-list');
  container.innerHTML = '';
  for (const i of visible) {
    const r = ALL_DATA[i];
    const checked = selected.has(i) ? 'checked' : '';
    const linkUrl = _companionUrl(r);
    const linkHtml = linkUrl
      ? `<a class="run-link" href="${linkUrl}" target="_blank" title="Open in E-field viewer">E-field ↗</a>`
      : '';
    const div = document.createElement('div');
    div.className = 'run-item';
    div.innerHTML = `
      <input type="checkbox" id="cb_${i}" ${checked} onchange="toggleRun(${i})">
      <div class="run-color" style="background:${color(i)}"></div>
      <label for="cb_${i}" class="run-label">${r.label}</label>
      ${linkHtml}`;
    container.appendChild(div);
  }
}

function toggleRun(i) {
  if (selected.has(i)) selected.delete(i);
  else selected.add(i);
  _render();
  _pushState();
}

function selectAll() {
  _filteredIndices().forEach(i => selected.add(i));
  _buildRunList(new Set(_filteredIndices()));
  _render();
  _pushState();
}
function selectNone() {
  selected.clear();
  _buildRunList(new Set(_filteredIndices()));
  _render();
  _pushState();
}
function toggleLog() {
  logScale = !logScale;
  document.getElementById('log-btn').classList.toggle('active', logScale);
  _render();
  _pushState();
}

// ── plotting ──────────────────────────────────────────────────────────────────
const LAYOUT_BASE = {
  margin: {l:60, r:20, t:10, b:50},
  height: 260,
  paper_bgcolor: '#fff',
  plot_bgcolor:  '#fff',
  xaxis: {title: {text: 'Step', font:{size:11}}, tickfont:{size:10}, gridcolor:'#eee'},
  yaxis: {tickfont:{size:10}, gridcolor:'#eee'},
  legend: {font:{size:10}, orientation:'h', yanchor:'bottom', y:1.02, xanchor:'left', x:0},
  hovermode: 'x unified',
};
const PLOTLY_CFG = {displayModeBar: true, responsive: true,
                   modeBarButtonsToRemove: ['toImage','sendDataToCloud','lasso2d','select2d']};

function _traces(key) {
  const traces = [];
  for (const i of selected) {
    const r = ALL_DATA[i];
    const y = r[key];
    if (!y) continue;
    traces.push({
      type: 'scatter', mode: 'lines',
      x: r.steps, y: y,
      name: r.label,
      line: {color: color(i), width: 1.5},
      hovertemplate: `%{y:.4e}<extra>${r.label}</extra>`,
    });
  }
  return traces;
}

function _layout(yLabel) {
  return {
    ...LAYOUT_BASE,
    yaxis: {
      ...LAYOUT_BASE.yaxis,
      title: {text: yLabel, font:{size:11}},
      type: logScale ? 'log' : 'linear',
    },
  };
}

function _render() {
  const noData = selected.size === 0;
  document.getElementById('no-data').style.display = noData ? '' : 'none';

  const plots = [
    {id: 'plot-total',   key: 'total',   ylabel: 'Total loss'},
    {id: 'plot-sobolev', key: 'sobolev', ylabel: 'Sobolev loss'},
    {id: 'plot-rotor',   key: 'rotor',   ylabel: 'Curl loss (unweighted)'},
  ];
  for (const {id, key, ylabel} of plots) {
    const traces = _traces(key);
    Plotly.react(id, traces, _layout(ylabel), PLOTLY_CFG);
  }
}

// ── init ──────────────────────────────────────────────────────────────────────
(function init() {
  if (!ALL_DATA || ALL_DATA.length === 0) {
    document.getElementById('no-data').style.display = '';
    document.getElementById('no-data').textContent = 'No data loaded.';
    return;
  }
  const urlP = _loadState();
  _buildFilterDropdowns(urlP);

  const visible = new Set(_filteredIndices());
  // if no selection saved, select first run
  if (selected.size === 0 && visible.size > 0) {
    selected.add([...visible][0]);
  }
  _buildRunList(visible);
  _render();
})();
</script>
</body>
</html>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('result_pkls', nargs='*',
                        help='result_*.pkl paths or directories to scan')
    parser.add_argument('--results-dir', default=None,
                        help='Scan this directory recursively for result_*.pkl files')
    parser.add_argument('--output', '-o', default=None,
                        help='Output HTML path (default: <first-dir>/loss_curves.html)')
    parser.add_argument('--wandb-project', default='jaxtpc-optimization')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Skip W&B fetch, use PKL loss_trajectory only')
    parser.add_argument('--wandb-samples', type=int, default=500,
                        help='Max history points to fetch per run from W&B')
    args = parser.parse_args()

    # ── collect PKLs ─────────────────────────────────────────────────────────
    pkls = []
    first_dir = None
    for p in args.result_pkls:
        if os.path.isdir(p):
            if first_dir is None:
                first_dir = p
            pkls += sorted(glob.glob(os.path.join(p, '**', 'result_*.pkl'), recursive=True))
        else:
            pkls.append(p)
    if args.results_dir:
        if first_dir is None:
            first_dir = args.results_dir
        pkls += sorted(glob.glob(
            os.path.join(args.results_dir, '**', 'result_*.pkl'), recursive=True))
    pkls = [p for p in pkls if not p.endswith('_efield_eval.pkl')]
    pkls = list(dict.fromkeys(pkls))  # deduplicate, preserve order

    if not pkls:
        print('No PKLs found.')
        return

    # ── determine output path ─────────────────────────────────────────────────
    if args.output:
        out = Path(args.output)
    elif first_dir:
        out = Path(first_dir) / 'loss_curves.html'
    else:
        out = Path(pkls[0]).parent / 'loss_curves.html'
    out.parent.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    runs = []
    for pkl in pkls:
        print(f'Loading {pkl}')
        try:
            rec = load_losses(
                pkl,
                wandb_project=args.wandb_project,
                use_wandb=not args.no_wandb,
                wandb_samples=args.wandb_samples,
            )
            if rec is not None:
                runs.append(rec)
        except Exception as exc:
            import traceback
            print(f'  ERROR: {exc}')
            traceback.print_exc()

    if not runs:
        print('No runs with loss data found.')
        return

    # ── collect param keys that have ≥2 distinct values ──────────────────────
    param_keys = [k for k, _ in _PARAM_PATTERNS
                  if len({r['params'].get(k) for r in runs} - {None}) >= 2]

    # ── write HTML ────────────────────────────────────────────────────────────
    companion_url = 'efield_eval.html'

    html = _HTML
    html = html.replace('/*ALL_DATA*/null/*END_ALL_DATA*/',
                        json.dumps(runs, separators=(',', ':')))
    html = html.replace('/*PARAM_KEYS*/null/*END_PARAM_KEYS*/',
                        json.dumps(param_keys))
    html = html.replace('/*PARAM_LABELS*/null/*END_PARAM_LABELS*/',
                        json.dumps(_PARAM_LABELS))
    html = html.replace('/*COMPANION_URL*/null/*END_COMPANION_URL*/',
                        json.dumps(companion_url))
    out.write_text(html, encoding='utf-8')
    print(f'\nWrote {out}  ({out.stat().st_size // 1024} KB,  {len(runs)} runs)')


if __name__ == '__main__':
    main()
