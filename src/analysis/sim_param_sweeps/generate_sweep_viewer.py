#!/usr/bin/env python
"""
Generate a self-contained interactive HTML viewer for sweep pkl files
produced by run_params.py.

Usage:
    python src/analysis/sim_param_sweeps/generate_sweep_viewer.py \
        --input-dir results/diffusion_sweep \
        --output viewer.html
"""

import argparse
import base64
import json
import pickle
import zlib
from pathlib import Path

import numpy as np

PARAM_PRETTY = {
    'diffusion_trans_cm2_us': 'D⊥ (cm²/μs)',
    'diffusion_long_cm2_us':  'D∥ (cm²/μs)',
    'velocity_cm_us':         'v (cm/μs)',
    'lifetime_us':            'τ (μs)',
    'recomb_alpha':           'α',
    'recomb_beta_90':         'β₉₀',
}

PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_chunks(input_dir: Path):
    paths = sorted(input_dir.glob('sweep_*.pkl'))
    if not paths:
        raise FileNotFoundError(f'No sweep_*.pkl files found in {input_dir}')
    chunks = []
    for p in paths:
        with open(p, 'rb') as f:
            chunks.append(pickle.load(f))
    chunks.sort(key=lambda c: c['chunk_idx'])
    return chunks


def build_data(chunks):
    meta         = chunks[0]['meta']
    param_names  = meta['param_names']
    param_grids  = meta['param_grids']
    plane_names  = meta['plane_names']
    deposit_names = list(chunks[0]['deposits'].keys())
    deposit_types = {n: chunks[0]['deposits'][n]['type'] for n in deposit_names}
    refs          = {n: chunks[0]['deposits'][n]['reference'] for n in deposit_names}

    if len(param_names) != 2:
        raise ValueError(f'Viewer requires exactly 2 sweep parameters, got {len(param_names)}')

    n_p0 = len(param_grids[param_names[0]])
    n_p1 = len(param_grids[param_names[1]])
    n_combos = n_p0 * n_p1

    # Assemble ordered sweep entries: deposit -> list[None|sweep_entry] indexed by global combo_idx
    ordered = {n: [None] * n_combos for n in deposit_names}
    for chunk in chunks:
        for local_i, global_i in enumerate(chunk['combo_indices']):
            for dep_name in deposit_names:
                ordered[dep_name][global_i] = chunk['deposits'][dep_name]['sweep'][local_i]

    # Pack traces per deposit × plane
    blobs = {}
    has_wire_trace = False

    for dep_name in deposit_names:
        blobs[dep_name] = {}
        for plane in plane_names:
            t_c, t_n, w_c, w_n = [], [], [], []
            for sw in ordered[dep_name]:
                c = sw['clean'][plane]
                n = sw['noisy'][plane]
                t_c.append(c.get('trace', np.array([], dtype=np.float32)))
                t_n.append(n.get('trace', np.array([], dtype=np.float32)))
                if 'wire_trace' in c:
                    has_wire_trace = True
                    w_c.append(c['wire_trace'])
                    w_n.append(n['wire_trace'])

            trace_len = len(t_c[0]) if t_c and len(t_c[0]) > 0 else 0
            wire_len  = len(w_c[0]) if w_c and len(w_c[0]) > 0 else 0
            blob = {'trace_len': trace_len, 'wire_len': wire_len, 't': None, 'w': None}

            if trace_len > 0:
                arr = np.stack([np.stack(t_c), np.stack(t_n)], axis=1).astype(np.float32)
                blob['t'] = base64.b64encode(zlib.compress(arr.tobytes(), 6)).decode()

            if wire_len > 0:
                arr = np.stack([np.stack(w_c), np.stack(w_n)], axis=1).astype(np.float32)
                blob['w'] = base64.b64encode(zlib.compress(arr.tobytes(), 6)).decode()

            blobs[dep_name][plane] = blob

    if has_wire_trace:
        print('wire_trace: YES')
    else:
        print('wire_trace: NO (re-run sweep after updating run_params.py to get V(wire))')

    return {
        'param_names':    param_names,
        'param_labels':   [PARAM_PRETTY.get(n, n) for n in param_names],
        'param_grids':    param_grids,
        'n_p0':           n_p0,
        'n_p1':           n_p1,
        'plane_names':    plane_names,
        'deposit_names':  deposit_names,
        'deposit_types':  deposit_types,
        'n_combos':       n_combos,
        'refs':           refs,
        'blobs':          blobs,
        'has_wire_trace': has_wire_trace,
        'palette':        PALETTE,
    }


# ── HTML template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js" charset="utf-8"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f0f2f5;color:#222}
h2{font-size:1.3em;font-weight:600;margin-bottom:12px}
h3{font-size:1em;font-weight:600;margin-bottom:10px}
.wrap{max-width:1500px;margin:0 auto;padding:14px}
.card{background:#fff;border-radius:8px;padding:14px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,.12)}
.ctrl-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.row{display:flex;align-items:center;gap:10px;margin:7px 0}
.lbl{width:160px;font-size:13px;color:#555;flex-shrink:0}
.slider{flex:1;accent-color:#2196F3}
.val{width:110px;font-family:monospace;font-size:12px;text-align:right;color:#333}
select{padding:5px 8px;border:1px solid #ccc;border-radius:4px;font-size:13px;flex:1}
button{padding:7px 14px;border:none;border-radius:4px;cursor:pointer;font-size:13px;font-weight:500}
.btn-add{background:#2196F3;color:#fff}
.btn-add:hover{background:#1976D2}
.btn-upd{background:#43A047;color:#fff}
.btn-upd:hover{background:#2E7D32}
.btn-cxl{background:#9e9e9e;color:#fff}
.btn-cxl:hover{background:#757575}
.btn-del{background:#e53935;color:#fff;padding:3px 8px;font-size:12px}
.btn-load{background:#FB8C00;color:#fff;padding:3px 8px;font-size:12px}
.plane-btn{padding:5px 12px;border:2px solid #ccc;border-radius:4px;cursor:pointer;background:#fff;font-size:13px}
.plane-btn.active{border-color:#2196F3;background:#E3F2FD;color:#1565C0;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#f5f5f5;padding:7px 8px;text-align:left;font-weight:600;border-bottom:2px solid #e0e0e0}
td{padding:5px 8px;border-bottom:1px solid #eee;vertical-align:middle}
tr.sel td{background:#E3F2FD}
tr:hover td{background:#fafafa}
tr.sel:hover td{background:#BBDEFB}
.swatch{width:14px;height:14px;border-radius:3px;display:inline-block;vertical-align:middle}
.plots-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.plot-wrap{height:420px}
.no-data{display:flex;align-items:center;justify-content:center;height:100%;color:#999;font-style:italic;text-align:center;font-size:13px;padding:20px}
#loading{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(255,255,255,.85);display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:16px;gap:10px;z-index:1000}
.spinner{width:36px;height:36px;border:4px solid #e0e0e0;border-top-color:#2196F3;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div id="loading"><div class="spinner"></div>Loading sweep data…</div>
<div class="wrap">
  <h2>__TITLE__</h2>

  <!-- Controls -->
  <div class="card">
    <h3>Controls</h3>
    <div class="ctrl-grid">
      <div>
        <div class="row">
          <span class="lbl">Deposit:</span>
          <select id="dep-sel"></select>
        </div>
        <div class="row">
          <span class="lbl">Plane:</span>
          <div id="plane-btns"></div>
        </div>
        <div class="row" style="margin-top:10px">
          <label style="font-size:13px;cursor:pointer">
            <input type="checkbox" id="noise-cb"> Show noise
          </label>
        </div>
      </div>
      <div>
        <div class="row">
          <span class="lbl" id="lbl-p0"></span>
          <input type="range" class="slider" id="sl-p0" min="0" step="1">
          <span class="val" id="val-p0"></span>
        </div>
        <div class="row">
          <span class="lbl" id="lbl-p1"></span>
          <input type="range" class="slider" id="sl-p1" min="0" step="1">
          <span class="val" id="val-p1"></span>
        </div>
      </div>
    </div>
    <div style="margin-top:12px;display:flex;gap:8px;align-items:center">
      <button class="btn-add" id="btn-add">Add Trace</button>
      <button class="btn-upd" id="btn-upd" style="display:none">Update Selected</button>
      <button class="btn-cxl" id="btn-cxl" style="display:none">Cancel</button>
      <span id="sel-hint" style="font-size:12px;color:#888;display:none">— editing entry <b id="sel-num"></b></span>
    </div>
  </div>

  <!-- Trace table -->
  <div class="card" id="tbl-card" style="display:none">
    <h3>Added Traces <span id="tbl-count" style="font-weight:400;color:#888"></span></h3>
    <table>
      <thead><tr>
        <th>#</th><th></th><th>Deposit</th><th>Plane</th>
        <th id="th-p0"></th><th id="th-p1"></th>
        <th>Noise</th><th>Actions</th>
      </tr></thead>
      <tbody id="tbl-body"></tbody>
    </table>
  </div>

  <!-- Plots -->
  <div class="plots-grid">
    <div class="card" style="padding:8px">
      <div id="plot-vt" class="plot-wrap"></div>
    </div>
    <div class="card" style="padding:8px">
      <div id="vw-outer" class="plot-wrap">
        <div id="vw-nodata" class="no-data" style="display:none">
          Wire-trace data not in these pkl files.<br>
          Re-run the sweep after updating run_params.py.
        </div>
        <div id="plot-vw" style="height:100%"></div>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;

/* ── blob cache & decode ── */
const _cache = {};
function _decode(dep, plane) {
  const k = dep + '/' + plane;
  if (_cache[k]) return _cache[k];
  const b = DATA.blobs[dep][plane];
  const r = {tLen: b.trace_len, wLen: b.wire_len, t: null, w: null};
  function inflate(b64) {
    const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    return new Float32Array(pako.inflate(bin).buffer);
  }
  if (b.t) r.t = inflate(b.t);
  if (b.w) r.w = inflate(b.w);
  _cache[k] = r;
  return r;
}

function getTrace(dep, plane, p0i, p1i, noisy) {
  const c = _decode(dep, plane);
  const ci = p0i * DATA.n_p1 + p1i;
  const ni = noisy ? 1 : 0;
  const ref = DATA.refs[dep][plane];
  const out = {ref};
  if (c.t) {
    const s = (ci * 2 + ni) * c.tLen;
    const ys = Array.from(c.t.slice(s, s + c.tLen));
    const t0 = ref.t_lo - ref.time;
    out.t = {x: ys.map((_, i) => i + t0), y: ys};
  }
  if (c.w) {
    const wLo = ref.w_lo != null ? ref.w_lo : 0;
    const s = (ci * 2 + ni) * c.wLen;
    const ys = Array.from(c.w.slice(s, s + c.wLen));
    const w0 = wLo - ref.wire;
    out.w = {x: ys.map((_, i) => i + w0), y: ys};
  }
  return out;
}

/* ── state ── */
let _entries = [];   // {id,dep,plane,p0i,p1i,noisy,color,label}
let _selId   = null;
let _nextId  = 0;
let _pIdx    = 0;
let _activePlane = DATA.plane_names[0];

function _nextColor() { return DATA.palette[_pIdx++ % DATA.palette.length]; }

function _fmtVal(n, i) {
  return DATA.param_grids[DATA.param_names[n]][i].toExponential(2);
}

function _makeLabel(ctrl) {
  const p0 = DATA.param_labels[0].replace(/ \(.+\)/, '');
  const p1 = DATA.param_labels[1].replace(/ \(.+\)/, '');
  return ctrl.dep + ' | ' + ctrl.plane +
    ' | ' + p0 + '=' + _fmtVal(0, ctrl.p0i) +
    ' ' + p1 + '=' + _fmtVal(1, ctrl.p1i) +
    (ctrl.noisy ? ' [noisy]' : '');
}

/* ── controls ── */
function _getCtrl() {
  return {
    dep:   document.getElementById('dep-sel').value,
    plane: _activePlane,
    p0i:   +document.getElementById('sl-p0').value,
    p1i:   +document.getElementById('sl-p1').value,
    noisy: document.getElementById('noise-cb').checked,
  };
}

function _setCtrl(ctrl) {
  document.getElementById('dep-sel').value       = ctrl.dep;
  document.getElementById('sl-p0').value         = ctrl.p0i;
  document.getElementById('sl-p1').value         = ctrl.p1i;
  document.getElementById('noise-cb').checked    = ctrl.noisy;
  _activePlane = ctrl.plane;
  _updatePlaneBtns();
  _updateValDisplays();
}

function _updateValDisplays() {
  document.getElementById('val-p0').textContent = _fmtVal(0, +document.getElementById('sl-p0').value);
  document.getElementById('val-p1').textContent = _fmtVal(1, +document.getElementById('sl-p1').value);
}

function _updatePlaneBtns() {
  DATA.plane_names.forEach(p => {
    const b = document.getElementById('pb-' + p);
    if (b) b.className = 'plane-btn' + (p === _activePlane ? ' active' : '');
  });
}

/* ── plot layouts ── */
const _lyVt = {
  xaxis: {title: 'Δt (bins from peak)', zeroline: true, zerolinecolor: '#ddd'},
  yaxis: {title: 'Signal (ADC)'},
  margin: {t:10, b:50, l:60, r:20},
  legend: {orientation: 'v', font: {size: 11}},
  title: {text: 'V(t) — main wire', font: {size: 14}, x: 0.05},
};
const _lyVw = {
  xaxis: {title: 'Δwire (channels from peak)', zeroline: true, zerolinecolor: '#ddd'},
  yaxis: {title: 'Signal (ADC)'},
  margin: {t:10, b:50, l:60, r:20},
  legend: {orientation: 'v', font: {size: 11}},
  title: {text: 'V(wire) — main time', font: {size: 14}, x: 0.05},
};
const _cfg = {responsive: true};

/* ── draw ── */
function _draw() {
  const vtT = [], vwT = [];

  function pushEntry(ctrl, color, label, dash, width) {
    const tr = getTrace(ctrl.dep, ctrl.plane, ctrl.p0i, ctrl.p1i, ctrl.noisy);
    if (tr.t) vtT.push({x: tr.t.x, y: tr.t.y, name: label, type: 'scatter', mode: 'lines',
      line: {color, dash, width}, showlegend: true});
    if (tr.w) vwT.push({x: tr.w.x, y: tr.w.y, name: label, type: 'scatter', mode: 'lines',
      line: {color, dash, width}, showlegend: true});
  }

  _entries.forEach(e => pushEntry(e, e.color, e.label, 'solid', 2));

  // preview
  const ctrl = _getCtrl();
  pushEntry(ctrl, '#bbb', 'Preview', 'dot', 1.5);

  Plotly.react('plot-vt', vtT, _lyVt, _cfg);
  if (DATA.has_wire_trace) Plotly.react('plot-vw', vwT, _lyVw, _cfg);
}

/* ── table ── */
function _renderTable() {
  const tbody = document.getElementById('tbl-body');
  tbody.innerHTML = '';
  const card = document.getElementById('tbl-card');
  card.style.display = _entries.length ? '' : 'none';
  document.getElementById('tbl-count').textContent = '(' + _entries.length + ')';

  _entries.forEach((e, idx) => {
    const tr = document.createElement('tr');
    if (e.id === _selId) tr.className = 'sel';
    tr.innerHTML =
      '<td>' + (idx + 1) + '</td>' +
      '<td><span class="swatch" style="background:' + e.color + '"></span></td>' +
      '<td title="' + e.dep + '" style="max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + e.dep + '</td>' +
      '<td>' + e.plane + '</td>' +
      '<td style="font-family:monospace">' + _fmtVal(0, e.p0i) + '</td>' +
      '<td style="font-family:monospace">' + _fmtVal(1, e.p1i) + '</td>' +
      '<td>' + (e.noisy ? 'Yes' : 'No') + '</td>' +
      '<td style="white-space:nowrap">' +
        '<button class="btn-load" onclick="_loadEntry(' + e.id + ')">Load</button> ' +
        '<button class="btn-del"  onclick="_delEntry(' + e.id + ')">Delete</button>' +
      '</td>';
    tr.addEventListener('click', ev => {
      if (ev.target.tagName === 'BUTTON') return;
      _selectEntry(e.id);
    });
    tbody.appendChild(tr);
  });
}

function _selectEntry(id) {
  _selId = id;
  const e = _entries.find(x => x.id === id);
  if (e) _setCtrl(e);
  document.getElementById('btn-upd').style.display = '';
  document.getElementById('btn-cxl').style.display = '';
  document.getElementById('sel-hint').style.display = '';
  document.getElementById('sel-num').textContent = _entries.findIndex(x => x.id === id) + 1;
  _renderTable();
  _draw();
}

function _loadEntry(id) { _selectEntry(id); }

function _delEntry(id) {
  _entries = _entries.filter(e => e.id !== id);
  if (_selId === id) _cancelEdit();
  _renderTable();
  _draw();
}

/* ── actions ── */
function _addTrace() {
  const ctrl = _getCtrl();
  _entries.push({id: _nextId++, ...ctrl, color: _nextColor(), label: _makeLabel(ctrl)});
  _renderTable();
  _draw();
}

function _updateSel() {
  if (_selId === null) return;
  const ctrl = _getCtrl();
  const e = _entries.find(x => x.id === _selId);
  if (!e) return;
  Object.assign(e, ctrl);
  e.label = _makeLabel(ctrl);
  _renderTable();
  _draw();
}

function _cancelEdit() {
  _selId = null;
  document.getElementById('btn-upd').style.display = 'none';
  document.getElementById('btn-cxl').style.display = 'none';
  document.getElementById('sel-hint').style.display = 'none';
  _renderTable();
}

/* ── init ── */
window.addEventListener('load', () => {
  // Deposit select
  const dep = document.getElementById('dep-sel');
  DATA.deposit_names.forEach(n => {
    const o = document.createElement('option');
    o.value = n;
    o.textContent = '[' + DATA.deposit_types[n] + '] ' + n;
    dep.appendChild(o);
  });

  // Plane buttons
  const pb = document.getElementById('plane-btns');
  DATA.plane_names.forEach(p => {
    const b = document.createElement('button');
    b.id = 'pb-' + p;
    b.textContent = p;
    b.className = 'plane-btn' + (p === _activePlane ? ' active' : '');
    b.onclick = () => { _activePlane = p; _updatePlaneBtns(); _draw(); };
    pb.appendChild(b);
    pb.appendChild(document.createTextNode(' '));
  });

  // Sliders
  const sl0 = document.getElementById('sl-p0');
  sl0.max = DATA.n_p0 - 1;
  sl0.value = Math.round((DATA.n_p0 - 1) / 2);
  document.getElementById('lbl-p0').textContent = DATA.param_labels[0] + ':';

  const sl1 = document.getElementById('sl-p1');
  sl1.max = DATA.n_p1 - 1;
  sl1.value = Math.round((DATA.n_p1 - 1) / 2);
  document.getElementById('lbl-p1').textContent = DATA.param_labels[1] + ':';

  document.getElementById('th-p0').textContent = DATA.param_labels[0];
  document.getElementById('th-p1').textContent = DATA.param_labels[1];

  _updateValDisplays();

  // Events
  dep.addEventListener('change',  _draw);
  sl0.addEventListener('input', () => { _updateValDisplays(); _draw(); });
  sl1.addEventListener('input', () => { _updateValDisplays(); _draw(); });
  document.getElementById('noise-cb').addEventListener('change', _draw);
  document.getElementById('btn-add').addEventListener('click', _addTrace);
  document.getElementById('btn-upd').addEventListener('click', _updateSel);
  document.getElementById('btn-cxl').addEventListener('click', _cancelEdit);

  // Init plots
  Plotly.newPlot('plot-vt', [], _lyVt, _cfg);
  if (DATA.has_wire_trace) {
    Plotly.newPlot('plot-vw', [], _lyVw, _cfg);
  } else {
    document.getElementById('vw-nodata').style.display = 'flex';
    document.getElementById('plot-vw').style.display   = 'none';
  }

  document.getElementById('loading').style.display = 'none';
  _draw();
});
</script>
</body>
</html>
"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir', required=True,
                   help='Directory containing sweep_*.pkl chunk files')
    p.add_argument('--output', default=None,
                   help='Output HTML path (default: <input-dir>/viewer.html)')
    return p.parse_args()


def main():
    args      = parse_args()
    input_dir = Path(args.input_dir)
    out_path  = Path(args.output) if args.output else input_dir / 'viewer.html'

    print(f'Loading chunks from {input_dir} …')
    chunks = load_chunks(input_dir)
    print(f'  {len(chunks)} chunk files, {sum(len(c["combos"]) for c in chunks)} combos total')

    print('Building data …')
    data = build_data(chunks)
    print(f'  {len(data["deposit_names"])} deposits, {len(data["plane_names"])} planes')
    print(f'  params: {data["param_names"]}')

    print('Serialising …')
    data_json = json.dumps(data, separators=(',', ':'))
    title     = 'Sweep Viewer — ' + ' vs '.join(
        PARAM_PRETTY.get(n, n) for n in data['param_names']
    )
    html = (HTML_TEMPLATE
            .replace('__DATA_JSON__', data_json)
            .replace('__TITLE__', title))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding='utf-8')
    size_mb = out_path.stat().st_size / 1e6
    print(f'\nWrote {out_path}  ({size_mb:.1f} MB)')
    print('Open in browser:  file://' + str(out_path.resolve()))


if __name__ == '__main__':
    main()
