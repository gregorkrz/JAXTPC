#!/usr/bin/env python
"""
Generate a self-contained interactive HTML viewer for pkl files produced by
src/analysis/1d_gradients.py with the --store-arrays flag.

The viewer shows:
  • Event display — wire × time heatmap for the selected track/plane/sweep point.
    Click the heatmap to set the wire/time reference for the trace plots.
  • Trace controls — pick track, plane, sweep point, noise; add to the list.
  • Trace plots — V(t) at the selected wire and V(wire) at the selected time,
    overlaying all added traces.

Usage
-----
    # Single pkl:
    python src/analysis/sim_param_sweeps/generate_gradient_viewer.py \\
        --pkl results/1d_gradients/diffusion_debug_20260515_3tracks/sobolev_...trans....pkl

    # Scan a directory and generate one viewer per pkl:
    python src/analysis/sim_param_sweeps/generate_gradient_viewer.py \\
        --dir results/1d_gradients/diffusion_debug_20260515_3tracks
"""

import argparse
import base64
import json
import os
import pickle
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── constants ──────────────────────────────────────────────────────────────────

# Signal-adaptive bounding box defaults
BBOX_THRESHOLD = 0.02   # fraction of peak above which a wire/time is "active"
BBOX_PAD_WIRE  = 20     # extra wires added on each side of the signal bbox
BBOX_PAD_TIME  = 30     # extra time bins added on each side
MAX_WIRE       = 300    # hard cap on stored window width (wires)
MAX_TIME       = 500    # hard cap on stored window height (time bins)

PARAM_PRETTY = {
    'diffusion_trans_cm2_us': 'D⊥ (cm²/μs)',
    'diffusion_long_cm2_us':  'D∥ (cm²/μs)',
    'velocity_cm_us':         'v (cm/μs)',
    'lifetime_us':            'τ (μs)',
    'recomb_alpha':           'α',
    'recomb_beta_90':         'β₉₀',
    'recomb_R':               'R',
}

PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
]


# ── data building ──────────────────────────────────────────────────────────────

def _pick_main(sig, axis):
    scores = np.abs(sig).mean(axis=1 if axis == 'wire' else 0)
    return int(np.argmax(scores))


def _signal_bbox(gt_p):
    """Return (wl, wh, tl, th) bounding box of signal, padded and capped.

    For clean arrays uses the 2%-of-peak threshold to find the signal extent.
    For noisy arrays the threshold activates most wires (noise RMS ≈ threshold),
    so when >50% of wires fire we fall back to centering on the array peak,
    which is still at the track location when signal >> noise.
    """
    nw, nt = gt_p.shape
    peak = float(np.abs(gt_p).max())

    if peak == 0.0:
        wl = max(0, nw // 2 - MAX_WIRE // 2)
        tl = max(0, nt // 2 - MAX_TIME // 2)
        return wl, min(nw, wl + MAX_WIRE), tl, min(nt, tl + MAX_TIME)

    mask = np.abs(gt_p) > BBOX_THRESHOLD * peak
    wire_idx = np.where(mask.any(axis=1))[0]
    time_idx = np.where(mask.any(axis=0))[0]

    # Noise inflation check: if more than half the wires are "active" the
    # threshold-based centroid is unreliable — centre on the peak instead.
    if len(wire_idx) > nw * 0.5 or len(time_idx) > nt * 0.5:
        pw, pt = np.unravel_index(np.abs(gt_p).argmax(), gt_p.shape)
        wl = max(0, pw - MAX_WIRE // 2)
        wh = min(nw, wl + MAX_WIRE)
        wl = max(0, wh - MAX_WIRE)
        tl = max(0, pt - MAX_TIME // 2)
        th = min(nt, tl + MAX_TIME)
        tl = max(0, th - MAX_TIME)
        return wl, wh, tl, th

    wl = max(0,  wire_idx[0]  - BBOX_PAD_WIRE)
    wh = min(nw, wire_idx[-1] + BBOX_PAD_WIRE + 1)
    tl = max(0,  time_idx[0]  - BBOX_PAD_TIME)
    th = min(nt, time_idx[-1] + BBOX_PAD_TIME + 1)

    if wh - wl > MAX_WIRE:
        cw = int((wire_idx[0] + wire_idx[-1]) // 2)
        wl = max(0, cw - MAX_WIRE // 2)
        wh = min(nw, wl + MAX_WIRE)
        wl = max(0, wh - MAX_WIRE)

    if th - tl > MAX_TIME:
        ct = int((time_idx[0] + time_idx[-1]) // 2)
        tl = max(0, ct - MAX_TIME // 2)
        th = min(nt, tl + MAX_TIME)
        tl = max(0, th - MAX_TIME)

    return wl, wh, tl, th


def _b64z(arr):
    return base64.b64encode(zlib.compress(np.asarray(arr, dtype=np.float32).tobytes(), 6)).decode()


def build_data(pkl, bbox_override=None):
    """Build viewer data dict from a pkl.

    bbox_override : dict {(track, plane): (wl, wh, tl, th)}, optional
        When provided, the given bbox is used instead of computing one from the
        GT array.  Used for noisy pkls so they share the clean GT bbox (noise
        inflates the threshold-based bbox to the full detector, hiding the track).
    """
    plane_names = pkl.get('plane_names', [])
    # Old pkls stored bare names without volume index (e.g. ['U','V','Y','U','V','Y']).
    # Detect duplicates and add a volume suffix so button IDs stay unique.
    _name_counts: dict = {}
    for n in plane_names:
        _name_counts[n] = _name_counts.get(n, 0) + 1
    if any(c > 1 for c in _name_counts.values()):
        _vol: dict = {}
        fixed = []
        for n in plane_names:
            _vol[n] = _vol.get(n, 0) + 1
            fixed.append(f'{n}{_vol[n]}')
        plane_names = fixed
    track_specs = pkl['track_specs']
    track_names = [ts['name'] for ts in track_specs]

    gt_arrays_all  = pkl.get('per_track_gt_arrays')
    sim_arrays_all = pkl.get('per_track_sim_arrays')

    if gt_arrays_all is None or sim_arrays_all is None:
        raise ValueError(
            'pkl missing per_track_gt_arrays / per_track_sim_arrays.\n'
            'Re-run 1d_gradients.py with --store-arrays.'
        )

    n_sweep = len(pkl['param_values'])
    noise_scale = float(pkl.get('noise_scale', 0.0))

    arrays = {}
    refs   = {}
    bboxes = {}
    for track in track_names:
        arrays[track] = {}
        refs[track]   = {}
        gt_all  = gt_arrays_all[track]   # list of n_planes arrays [n_wire, n_time]
        sim_all = sim_arrays_all[track]  # list[n_sweep] of list[n_planes] arrays

        for pi, plane in enumerate(plane_names):
            gt_p = np.array(gt_all[pi], dtype=np.float32)  # (n_wire, n_time)

            wire_ref = _pick_main(gt_p, 'wire')
            time_ref = _pick_main(gt_p, 'time')

            if bbox_override and (track, plane) in bbox_override:
                wl, wh, tl, th = bbox_override[(track, plane)]
            else:
                wl, wh, tl, th = _signal_bbox(gt_p)
            bboxes[(track, plane)] = (wl, wh, tl, th)
            nw, nt = wh - wl, th - tl

            # Stack: index 0 = GT, indices 1..n_sweep = sweep pts
            stacked = [gt_p[wl:wh, tl:th]]
            for sw_pt in sim_all:
                arr = np.array(sw_pt[pi], dtype=np.float32)
                stacked.append(arr[wl:wh, tl:th])
            combined = np.stack(stacked, axis=0)  # (n_pts, n_wire, n_time)

            print(f'    {track} {plane}: bbox [{wl}:{wh}, {tl}:{th}]  '
                  f'({nw}w × {nt}t)  '
                  f'{combined.nbytes / 1e6:.1f} MB raw')

            arrays[track][plane] = {
                'wire_lo':  wl,
                'time_lo':  tl,
                'wire_ref': wire_ref,
                'time_ref': time_ref,
                'n_wire':   nw,
                'n_time':   nt,
                'data':     _b64z(combined),
            }
            refs[track][plane] = {'wire': wire_ref, 'time': time_ref}

    # Sweep point labels: index 0 = GT, indices 1..n_sweep = sweep points
    factors = pkl['factors']
    param_values = pkl['param_values']
    pt_labels = ['GT (×1.00)'] + [
        f'×{f:.2f} = {v:.4g}' for f, v in zip(factors, param_values)
    ]
    gt_pt_idx = next(
        (i + 1 for i, f in enumerate(factors) if abs(f - 1.0) < 1e-9), 0
    )

    param_label = PARAM_PRETTY.get(pkl['param_name'], pkl['param_name'])
    noise_str   = f', noise={noise_scale:.2g}' if noise_scale > 0 else ', no noise'

    return {
        'param_name':    pkl['param_name'],
        'param_label':   param_label,
        'param_gt':      float(pkl['param_gt']),
        'param_values':  param_values,
        'factors':       factors,
        'pt_labels':     pt_labels,
        'gt_pt_idx':     gt_pt_idx,
        'n_sweep':       n_sweep,
        'noise_scale':   noise_scale,
        'plane_names':   plane_names,
        'track_names':   track_names,
        'arrays':        arrays,
        'refs':          refs,
        'bboxes':        bboxes,
        'loss_total':    pkl['loss_values'],
        'grad_total':    pkl['grad_values'],
        'loss_per_track': pkl['per_track_loss_values'],
        'grad_per_track': pkl['per_track_grad_values'],
        'palette':       PALETTE,
        'ds_label':      param_label + noise_str,
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
h3{font-size:1.0em;font-weight:600;margin-bottom:10px}
.wrap{max-width:1500px;margin:0 auto;padding:14px}
.card{background:#fff;border-radius:8px;padding:14px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,.12)}
.row{display:flex;align-items:center;gap:10px;margin:7px 0;flex-wrap:wrap}
.lbl{width:90px;font-size:13px;color:#555;flex-shrink:0}
.slider{flex:1;min-width:120px;accent-color:#2196F3}
.val{min-width:160px;font-family:monospace;font-size:12px;color:#333}
select{padding:5px 8px;border:1px solid #ccc;border-radius:4px;font-size:13px}
input[type=number]{width:80px;padding:4px 6px;border:1px solid #ccc;border-radius:4px;font-size:13px;font-family:monospace}
button{padding:7px 14px;border:none;border-radius:4px;cursor:pointer;font-size:13px;font-weight:500}
.btn-add{background:#2196F3;color:#fff}
.btn-add:hover{background:#1976D2}
.btn-upd{background:#43A047;color:#fff}
.btn-upd:hover{background:#2E7D32}
.btn-cxl{background:#9e9e9e;color:#fff}
.btn-cxl:hover{background:#757575}
.btn-del{background:#e53935;color:#fff;padding:3px 8px;font-size:12px}
.btn-load{background:#FB8C00;color:#fff;padding:3px 8px;font-size:12px}
.plane-btn{padding:4px 10px;border:2px solid #ccc;border-radius:4px;cursor:pointer;background:#fff;font-size:13px}
.plane-btn.active{border-color:#2196F3;background:#E3F2FD;color:#1565C0;font-weight:600}
.mode-btn{padding:4px 10px;border:2px solid #ccc;border-radius:4px;cursor:pointer;background:#fff;font-size:12px}
.mode-btn.active{border-color:#7B1FA2;background:#F3E5F5;color:#4A148C;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#f5f5f5;padding:7px 8px;text-align:left;font-weight:600;border-bottom:2px solid #e0e0e0}
td{padding:5px 8px;border-bottom:1px solid #eee;vertical-align:middle}
tr.sel td{background:#E3F2FD}
tr:hover td{background:#fafafa}
tr.sel:hover td{background:#BBDEFB}
.swatch{width:14px;height:14px;border-radius:3px;display:inline-block;vertical-align:middle}
.plots-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.plot-wrap{height:400px}
#loading{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(255,255,255,.88);display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:16px;gap:10px;z-index:1000}
.spinner{width:36px;height:36px;border:4px solid #e0e0e0;border-top-color:#2196F3;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
#evd-plot{height:360px;cursor:crosshair}
.ref-row{display:flex;align-items:center;gap:12px;margin-top:10px;padding-top:10px;border-top:1px solid #eee}
.ref-row label{font-size:13px;color:#555}
.hint{font-size:11px;color:#aaa;margin-left:4px}
</style>
</head>
<body>
<div id="loading"><div class="spinner"></div>Loading sweep data…</div>
<div class="wrap">
  <h2>__TITLE__</h2>

  <!-- ── Event Display ─────────────────────────────────────────────── -->
  <div class="card">
    <h3>Event Display</h3>

    <div class="row">
      <span class="lbl">Track:</span>
      <select id="evd-track"></select>

      <span class="lbl" style="margin-left:12px">Plane:</span>
      <div id="evd-plane-btns" style="display:flex;gap:4px"></div>
    </div>

    <div class="row">
      <span class="lbl">Show:</span>
      <div id="evd-mode-btns" style="display:flex;gap:4px">
        <button class="mode-btn active" data-mode="sim" id="mode-sim">Sim</button>
        <button class="mode-btn" data-mode="gt"  id="mode-gt" >GT</button>
        <button class="mode-btn" data-mode="diff" id="mode-diff">Sim − GT</button>
      </div>
      <label style="margin-left:16px;font-size:13px;cursor:pointer" id="noise-label">
        <input type="checkbox" id="noise-cb"> Noise
      </label>
    </div>

    <div id="evd-param-rows"></div>

    <div id="evd-plot"></div>

    <div class="ref-row">
      <label>Wire ref: <input type="number" id="wire-ref" min="0" value="0"></label>
      <label>Time ref: <input type="number" id="time-ref" min="0" value="0"></label>
      <span class="hint">← or click the heatmap</span>
    </div>
    <div class="plots-grid" style="margin-top:10px">
      <div id="evd-vt" style="height:200px"></div>
      <div id="evd-vw" style="height:200px"></div>
    </div>
  </div>

  <!-- ── Add Trace ─────────────────────────────────────────────────── -->
  <div class="card">
    <h3>Add Trace</h3>
    <div class="row">
      <span class="lbl">Track:</span>
      <select id="tr-track"></select>
    </div>
    <div class="row">
      <span class="lbl">Param:</span>
      <div id="tr-param-btns" style="display:flex;gap:4px"></div>
    </div>
    <div class="row" id="tr-noise-row">
      <label style="font-size:13px;cursor:pointer">
        <input type="checkbox" id="tr-noise-cb"> Noise
      </label>
    </div>
    <div class="row" style="margin-top:8px">
      <button class="btn-add" id="btn-add">Add Trace</button>
      <button class="btn-upd" id="btn-upd" style="display:none">Update Selected</button>
      <button class="btn-cxl" id="btn-cxl" style="display:none">Cancel</button>
      <span id="sel-hint" style="font-size:12px;color:#888;display:none">
        — editing <b id="sel-num"></b>
      </span>
    </div>
  </div>

  <!-- ── Added Traces ───────────────────────────────────────────────── -->
  <div class="card" id="tbl-card" style="display:none">
    <h3>Added Traces <span id="tbl-count" style="font-weight:400;color:#888"></span></h3>
    <table>
      <thead><tr>
        <th>#</th><th></th><th>Track</th>
        <th>Param</th><th>Sweep pt</th><th>Actions</th>
      </tr></thead>
      <tbody id="tbl-body"></tbody>
    </table>
  </div>

  <!-- ── ADC Distribution ──────────────────────────────────────────── -->
  <div class="card" id="hist-card" style="display:none">
    <h3 style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
      ADC Distribution
      <span style="font-weight:400;font-size:13px;color:#555">Plane:</span>
      <div id="hist-plane-btns" style="display:flex;gap:4px"></div>
    </h3>
    <div id="hist-plot" style="height:320px"></div>
  </div>

  <!-- ── 1-D Trace Plots (one row per wireplane) ───────────────────── -->
  <div id="plots-container"></div>

</div><!-- /.wrap -->

<script>
const PARAMS = __PARAMS_JSON__;

/* ── colour palette ── */
const _PALETTE = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
  '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
  '#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5'];

/* ── inflate / data access ── */
const _inflClean = {}, _inflNoisy = {};

function _getBuf(pi, track, plane, useNoisy) {
  const cache = useNoisy ? _inflNoisy : _inflClean;
  const k = pi + '/' + track + '/' + plane;
  if (cache[k]) return cache[k];
  const p = PARAMS[pi];
  const arrays = useNoisy ? p.arrays_noisy : p.arrays_clean;
  if (!arrays) return _getBuf(pi, track, plane, false);
  const a = arrays[track][plane];
  const bin = Uint8Array.from(atob(a.data), c => c.charCodeAt(0));
  cache[k] = new Float32Array(pako.inflate(bin).buffer);
  return cache[k];
}

function _dims(pi, track, plane) {
  return (PARAMS[pi].arrays_clean || PARAMS[pi].arrays_noisy)[track][plane];
}

function _ptBuf(pi, track, plane, pt, noisy) {
  return _getBuf(pi, track, plane, pt === 0 && noisy);
}

function _getZ(pi, track, plane, pt, noisy=false) {
  const a = _dims(pi, track, plane);
  const buf = _ptBuf(pi, track, plane, pt, noisy);
  const off = pt * a.n_wire * a.n_time;
  const z = [];
  for (let w = 0; w < a.n_wire; w++) {
    const row = [];
    for (let t = 0; t < a.n_time; t++) row.push(buf[off + w * a.n_time + t]);
    z.push(row);
  }
  return z;
}

function _getZDiff(pi, track, plane, pt, noisy=false) {
  const a    = _dims(pi, track, plane);
  const bufS = _getBuf(pi, track, plane, false);
  const bufG = _ptBuf(pi, track, plane, 0, noisy);
  const offS = pt * a.n_wire * a.n_time, offG = 0;
  const z = [];
  for (let w = 0; w < a.n_wire; w++) {
    const row = [];
    for (let t = 0; t < a.n_time; t++)
      row.push(bufS[offS + w * a.n_time + t] - bufG[offG + w * a.n_time + t]);
    z.push(row);
  }
  return z;
}

function _getTimeTrace(pi, track, plane, pt, wireAbs, noisy=false) {
  const a = _dims(pi, track, plane);
  const w = wireAbs - a.wire_lo;
  if (w < 0 || w >= a.n_wire) return null;
  const buf = _ptBuf(pi, track, plane, pt, noisy);
  const off = pt * a.n_wire * a.n_time + w * a.n_time;
  return {x: Array.from({length:a.n_time}, (_,i) => a.time_lo+i),
          y: Array.from(buf.slice(off, off+a.n_time))};
}

function _getWireTrace(pi, track, plane, pt, timeAbs, noisy=false) {
  const a = _dims(pi, track, plane);
  const t = timeAbs - a.time_lo;
  if (t < 0 || t >= a.n_time) return null;
  const buf = _ptBuf(pi, track, plane, pt, noisy);
  const off = pt * a.n_wire * a.n_time;
  const y = [];
  for (let w = 0; w < a.n_wire; w++) y.push(buf[off + w * a.n_time + t]);
  return {x: Array.from({length:a.n_wire}, (_,i) => a.wire_lo+i), y};
}

/* ── state ── */
let _evdTrack    = PARAMS[0].track_names[0];
let _evdPlane    = PARAMS[0].plane_names[0];
let _evdMode     = 'sim';
let _evdParamIdx = 0;
let _sweepPt     = PARAMS.map(() => 1);   // current sweep pt per param (0=GT)
let _noiseOn          = false;
const _refs           = {};   // per 'track/plane' → {wire, time}
let _trTrack          = PARAMS[0].track_names[0];
let _trParamIdx       = 0;
let _entries          = [];
let _selId            = null;
let _nextId           = 0;
let _pIdx             = 0;
let _evdInited        = false;
let _evdPreviewInited = false;

function _nextColor() { return _PALETTE[_pIdx++ % _PALETTE.length]; }
function _ptLabel(pi, pt) { return PARAMS[pi].pt_labels[pt]; }
function _makeLabel(e) {
  return e.track + ' | ' + PARAMS[e.pi].param_label + ' | ' + _ptLabel(e.pi, e.pt) +
    (e.noisy ? ' [noisy]' : '');
}

/* ── event display ── */
function _drawEvd() {
  const pi = _evdParamIdx;
  const a  = _dims(pi, _evdTrack, _evdPlane);
  const xArr = Array.from({length:a.n_time}, (_,i) => a.time_lo+i);
  const yArr = Array.from({length:a.n_wire}, (_,i) => a.wire_lo+i);
  const pt = Math.max(1, _sweepPt[pi]);

  let z, titleSuffix;
  if (_evdMode === 'gt') {
    z = _getZ(pi, _evdTrack, _evdPlane, 0, _noiseOn);
    titleSuffix = _noiseOn ? 'GT (noisy)' : 'GT';
  } else if (_evdMode === 'sim') {
    z = _getZ(pi, _evdTrack, _evdPlane, pt, false);
    titleSuffix = _ptLabel(pi, pt);
  } else {
    z = _getZDiff(pi, _evdTrack, _evdPlane, pt, _noiseOn);
    titleSuffix = 'Sim−GT @ ' + _ptLabel(pi, pt);
  }

  let absmax = 0;
  z.forEach(row => row.forEach(v => { if (Math.abs(v) > absmax) absmax = Math.abs(v); }));
  if (absmax === 0) absmax = 1;

  const trace = {
    type:'heatmap', x:xArr, y:yArr, z,
    colorscale:'RdBu', reversescale:true,
    zmin:-absmax, zmax:absmax,
    colorbar:{title:'ADC', thickness:14, len:0.9},
    hoverongaps:false,
  };

  const {wire: wRef, time: tRef} = _evdRef();
  const layout = {
    title:{text: _evdTrack+' — '+_evdPlane+' — '+PARAMS[pi].param_label+' — '+titleSuffix,
           font:{size:13}, x:0.04},
    xaxis:{title:'Time bin', fixedrange:false},
    yaxis:{title:'Wire index', fixedrange:false},
    margin:{t:32, b:50, l:60, r:80},
    shapes:[
      {type:'line', x0:tRef, x1:tRef, y0:yArr[0], y1:yArr[yArr.length-1],
       line:{color:'#00C853', width:1.5, dash:'dot'}},
      {type:'line', x0:xArr[0], x1:xArr[xArr.length-1], y0:wRef, y1:wRef,
       line:{color:'#00C853', width:1.5, dash:'dot'}},
    ],
    annotations:[{
      x:tRef, y:wRef, text:'('+wRef+','+tRef+')', showarrow:false,
      font:{size:10, color:'#00C853'},
      xanchor: tRef>(xArr[0]+xArr[xArr.length-1])/2 ? 'right':'left',
      yanchor: wRef>(yArr[0]+yArr[yArr.length-1])/2 ? 'top':'bottom',
    }],
  };

  const cfg = {responsive:true};
  if (!_evdInited) {
    Plotly.newPlot('evd-plot', [trace], layout, cfg);
    document.getElementById('evd-plot').on('plotly_click', d => {
      if (!d.points.length) return;
      const p = d.points[0];
      const rr = _evdRef(); rr.wire = Math.round(p.y); rr.time = Math.round(p.x);
      _syncRefInputs(); _drawEvd(); _drawTraces();
    });
    _evdInited = true;
  } else {
    Plotly.react('evd-plot', [trace], layout, cfg);
  }
  _drawEvdPreview();
}

function _drawEvdPreview() {
  const pi = _evdParamIdx;
  const pt = Math.max(1, _sweepPt[pi]);
  const {wire: wireAbs, time: timeAbs} = _evdRef();

  const sim_t = _getTimeTrace(pi, _evdTrack, _evdPlane, pt,  wireAbs, false);
  const gt_t  = _getTimeTrace(pi, _evdTrack, _evdPlane, 0,   wireAbs, _noiseOn);
  const sim_w = _getWireTrace(pi, _evdTrack, _evdPlane, pt,  timeAbs, false);
  const gt_w  = _getWireTrace(pi, _evdTrack, _evdPlane, 0,   timeAbs, _noiseOn);

  const vtT = [], vwT = [];
  const _evdGtLabel = (_noiseOn && PARAMS[pi].has_noise) ? 'GT (noisy)' : 'GT';
  if (gt_t)  vtT.push({x:gt_t.x,  y:gt_t.y,  name:_evdGtLabel, type:'scatter', mode:'lines', line:{color:'#aaa', width:1.5, dash:'dot'}});
  if (sim_t) vtT.push({x:sim_t.x, y:sim_t.y, name:'Sim '+_ptLabel(pi,pt), type:'scatter', mode:'lines', line:{color:'#1f77b4', width:2}});
  if (gt_w)  vwT.push({x:gt_w.x,  y:gt_w.y,  name:_evdGtLabel, type:'scatter', mode:'lines', line:{color:'#aaa', width:1.5, dash:'dot'}});
  if (sim_w) vwT.push({x:sim_w.x, y:sim_w.y, name:'Sim '+_ptLabel(pi,pt), type:'scatter', mode:'lines', line:{color:'#1f77b4', width:2}});

  const lyBase = {margin:{t:30,b:40,l:55,r:10}, legend:{orientation:'h',font:{size:10},y:1.12},
                  yaxis:{title:'ADC'}};
  const lyVt = Object.assign({}, lyBase, {xaxis:{title:'Time bin'},
    title:{text:_evdPlane+' — V(t) at wire '+wireAbs, font:{size:12}, x:0.04}});
  const lyVw = Object.assign({}, lyBase, {xaxis:{title:'Wire index'},
    title:{text:_evdPlane+' — V(wire) at time '+timeAbs, font:{size:12}, x:0.04}});

  if (!_evdPreviewInited) {
    Plotly.newPlot('evd-vt', vtT, lyVt, _cfg);
    Plotly.newPlot('evd-vw', vwT, lyVw, _cfg);
    _evdPreviewInited = true;
  } else {
    Plotly.react('evd-vt', vtT, lyVt, _cfg);
    Plotly.react('evd-vw', vwT, lyVw, _cfg);
  }
}

function _evdRef() { return _refs[_evdTrack + '/' + _evdPlane]; }
function _syncRefInputs() {
  const r = _evdRef();
  document.getElementById('wire-ref').value = r.wire;
  document.getElementById('time-ref').value = r.time;
}

/* ── trace plots (one row per wireplane) ── */
const _baseLy = {
  margin:{t:30,b:50,l:60,r:20},
  legend:{orientation:'v',font:{size:10}},
};
const _cfg = {responsive:true};

function _ensurePlaneRow(plane) {
  const vtId = 'plot-vt-' + plane;
  if (document.getElementById(vtId)) return;
  const vwId = 'plot-vw-' + plane;
  const container = document.getElementById('plots-container');

  const row = document.createElement('div');
  row.className = 'plots-grid'; row.style.marginBottom = '12px';

  [vtId, vwId].forEach(id => {
    const card = document.createElement('div');
    card.className = 'card'; card.style.padding = '8px';
    const div = document.createElement('div');
    div.id = id; div.className = 'plot-wrap';
    card.appendChild(div); row.appendChild(card);
  });
  container.appendChild(row);
}

function _drawTraces() {
  const planes = PARAMS[0].plane_names;

  planes.forEach(plane => {
    _ensurePlaneRow(plane);
    const vtId = 'plot-vt-' + plane;
    const vwId = 'plot-vw-' + plane;
    const vtT = [], vwT = [];

    // GT — clean or noisy depending on noise checkbox
    {
      const pi = _trParamIdx;
      const prRef = _refs[_trTrack + '/' + plane] || {wire:0, time:0};
      const gt_t = _getTimeTrace(pi, _trTrack, plane, 0, prRef.wire, _noiseOn);
      const gt_w = _getWireTrace(pi, _trTrack, plane, 0, prRef.time, _noiseOn);
      const gtLabel = (_noiseOn && PARAMS[pi].has_noise) ? 'GT (noisy)' : 'GT';
      const gtLine = {color:'#444', width:1.5, dash:'dash'};
      if (gt_t) vtT.push({x:gt_t.x, y:gt_t.y, name:gtLabel, type:'scatter', mode:'lines', line:gtLine});
      if (gt_w) vwT.push({x:gt_w.x, y:gt_w.y, name:gtLabel, type:'scatter', mode:'lines', line:gtLine});
    }

    // Preview (dashed grey) — shown in every plane using that plane's ref for _trTrack
    {
      const pi = _trParamIdx, pt = _sweepPt[pi];
      const prRef = _refs[_trTrack + '/' + plane] || {wire:0, time:0};
      const prev_t = _getTimeTrace(pi, _trTrack, plane, pt, prRef.wire);
      const prev_w = _getWireTrace(pi, _trTrack, plane, pt, prRef.time);
      if (prev_t) vtT.push({x:prev_t.x, y:prev_t.y, name:'Preview', type:'scatter', mode:'lines',
        line:{color:'#bbb', dash:'dot', width:1.5}});
      if (prev_w) vwT.push({x:prev_w.x, y:prev_w.y, name:'Preview', type:'scatter', mode:'lines',
        line:{color:'#bbb', dash:'dot', width:1.5}});
    }

    // All added entries — each extracted at this plane's ref for the entry's track
    _entries.forEach(e => {
      const er = _refs[e.track + '/' + plane] || {wire:0, time:0};
      const tr_t = _getTimeTrace(e.pi, e.track, plane, e.pt, er.wire, e.noisy || false);
      const tr_w = _getWireTrace(e.pi, e.track, plane, e.pt, er.time, e.noisy || false);
      if (tr_t) vtT.push({x:tr_t.x, y:tr_t.y, name:e.label, type:'scatter', mode:'lines',
        line:{color:e.color, width:2}});
      if (tr_w) vwT.push({x:tr_w.x, y:tr_w.y, name:e.label, type:'scatter', mode:'lines',
        line:{color:e.color, width:2}});
    });

    // Use EVD-track ref for this plane in the title
    const r = _refs[_evdTrack + '/' + plane] || {wire:'?', time:'?'};
    const lyVt = Object.assign({}, _baseLy,
      {xaxis:{title:'Time bin'}, yaxis:{title:'Signal (ADC)'},
       title:{text:plane+' — V(t) at wire '+r.wire, font:{size:13}, x:0.04}});
    const lyVw = Object.assign({}, _baseLy,
      {xaxis:{title:'Wire index'}, yaxis:{title:'Signal (ADC)'},
       title:{text:plane+' — V(wire) at time '+r.time, font:{size:13}, x:0.04}});

    const vtEl = document.getElementById(vtId);
    const vwEl = document.getElementById(vwId);
    if (vtEl._hasPlot) { Plotly.react(vtId, vtT, lyVt, _cfg); }
    else { Plotly.newPlot(vtId, vtT, lyVt, _cfg); vtEl._hasPlot = true; }
    if (vwEl._hasPlot) { Plotly.react(vwId, vwT, lyVw, _cfg); }
    else { Plotly.newPlot(vwId, vwT, lyVw, _cfg); vwEl._hasPlot = true; }
  });
}

/* ── ADC histogram ── */
let _histPlane  = null;
let _histInited = false;

function _getFlatADC(e, plane) {
  const a = _dims(e.pi, e.track, plane);
  const buf = _ptBuf(e.pi, e.track, plane, e.pt, e.noisy || false);
  const off = e.pt * a.n_wire * a.n_time;
  const flat = new Array(a.n_wire * a.n_time);
  for (let i = 0; i < flat.length; i++) flat[i] = buf[off + i];
  return flat;
}

function _drawHist() {
  const card = document.getElementById('hist-card');
  if (!_entries.length) { card.style.display = 'none'; return; }
  card.style.display = '';
  const plane = _histPlane || PARAMS[0].plane_names[0];

  const perEntry = _entries.map(e => _getFlatADC(e, plane));

  let gMin = Infinity, gMax = -Infinity;
  perEntry.forEach(vals => {
    for (let i = 0; i < vals.length; i++) {
      if (vals[i] < gMin) gMin = vals[i];
      if (vals[i] > gMax) gMax = vals[i];
    }
  });
  if (!isFinite(gMin)) return;
  const binSize = Math.max((gMax - gMin) / 200, 1e-9);

  const traces = _entries.map((e, i) => ({
    x: perEntry[i],
    name: e.label,
    type: 'histogram',
    opacity: 0.55,
    marker: {color: e.color, line: {width: 0}},
    xbins: {start: gMin, end: gMax + binSize, size: binSize},
    autobinx: false,
  }));

  const layout = {
    barmode: 'overlay',
    xaxis: {title: 'Signal (ADC)'},
    yaxis: {title: 'Count', type: 'log'},
    margin: {t:10, b:50, l:60, r:20},
    legend: {font: {size: 11}},
  };

  if (!_histInited) {
    Plotly.newPlot('hist-plot', traces, layout, _cfg);
    _histInited = true;
  } else {
    Plotly.react('hist-plot', traces, layout, _cfg);
  }
}

/* ── table ── */
function _renderTable() {
  const tbody = document.getElementById('tbl-body');
  tbody.innerHTML = '';
  document.getElementById('tbl-card').style.display = _entries.length ? '' : 'none';
  document.getElementById('tbl-count').textContent  = '(' + _entries.length + ')';
  _entries.forEach((e, idx) => {
    const tr = document.createElement('tr');
    if (e.id === _selId) tr.className = 'sel';
    tr.innerHTML =
      '<td>'+(idx+1)+'</td>'+
      '<td><span class="swatch" style="background:'+e.color+'"></span></td>'+
      '<td style="max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="'+e.track+'">'+e.track+'</td>'+
      '<td>'+PARAMS[e.pi].param_label+'</td>'+
      '<td style="font-family:monospace;font-size:12px">'+_ptLabel(e.pi,e.pt)+'</td>'+
      '<td style="white-space:nowrap">'+
        '<button class="btn-load" onclick="_loadEntry('+e.id+')">Load</button> '+
        '<button class="btn-del"  onclick="_delEntry('+e.id+')">Delete</button>'+
      '</td>';
    tr.addEventListener('click', ev => { if (ev.target.tagName==='BUTTON') return; _selectEntry(e.id); });
    tbody.appendChild(tr);
  });
  _drawHist();
}

/* ── entry management ── */
function _getTraceCtrl() {
  return {
    track: _trTrack,
    pi:    _trParamIdx,
    pt:    _sweepPt[_trParamIdx],
    noisy: document.getElementById('tr-noise-cb').checked,
  };
}

function _addTrace() {
  const ctrl = _getTraceCtrl();
  const e = {id:_nextId++, ...ctrl, color:_nextColor()};
  e.label = _makeLabel(e);
  _entries.push(e); _renderTable(); _drawTraces();
}

function _updateSel() {
  if (_selId === null) return;
  const e = _entries.find(x => x.id === _selId);
  if (!e) return;
  Object.assign(e, _getTraceCtrl());
  e.label = _makeLabel(e); _renderTable(); _drawTraces();
}

function _cancelEdit() {
  _selId = null;
  document.getElementById('btn-upd').style.display  = 'none';
  document.getElementById('btn-cxl').style.display  = 'none';
  document.getElementById('sel-hint').style.display = 'none';
  _renderTable();
}

function _selectEntry(id) {
  _selId = id;
  const e = _entries.find(x => x.id === id);
  if (e) {
    _trTrack = e.track; _trParamIdx = e.pi;
    _sweepPt[e.pi] = e.pt;
    document.getElementById('tr-track').value = e.track;
    document.getElementById('tr-noise-cb').checked = e.noisy || false;
    _updateTrParamBtn(e.pi);
    const sl = document.getElementById('evd-sl-'+e.pi);
    if (sl) { sl.value = String(e.pt); _updateSlLabel(e.pi); }
  }
  document.getElementById('btn-upd').style.display  = '';
  document.getElementById('btn-cxl').style.display  = '';
  document.getElementById('sel-hint').style.display = '';
  document.getElementById('sel-num').textContent = _entries.findIndex(x=>x.id===id)+1;
  _renderTable(); _drawTraces();
}

function _loadEntry(id) { _selectEntry(id); }
function _delEntry(id) {
  _entries = _entries.filter(e => e.id !== id);
  if (_selId === id) _cancelEdit();
  _renderTable(); _drawTraces();
}

/* ── plane / param button helpers ── */
function _updateEvdPlane(p) {
  _evdPlane = p;
  PARAMS[0].plane_names.forEach(n => {
    const b = document.getElementById('evd-pb-'+n);
    if (b) b.className = 'plane-btn'+(n===p?' active':'');
  });
}

function _updateEvdParamBtn(pi) {
  _evdParamIdx = pi;
  PARAMS.forEach((_,i) => {
    const b = document.getElementById('evd-param-view-'+i);
    if (b) b.className = 'plane-btn'+(i===pi?' active':'');
  });
}

function _updateTrParamBtn(pi) {
  _trParamIdx = pi;
  PARAMS.forEach((_,i) => {
    const b = document.getElementById('tr-pb-param-'+i);
    if (b) b.className = 'plane-btn'+(i===pi?' active':'');
  });
}

function _updateSlLabel(pi) {
  const sl = document.getElementById('evd-sl-'+pi);
  if (!sl) return;
  const pt = +sl.value;
  _sweepPt[pi] = pt;
  document.getElementById('evd-sl-val-'+pi).textContent = _ptLabel(pi, pt);
}

/* ── init ── */
function _initUI() {
  const p0 = PARAMS[0];
  _evdTrack = p0.track_names[0]; _evdPlane = p0.plane_names[0];
  _evdParamIdx = 0; _trParamIdx = 0;
  _trTrack = p0.track_names[0];
  _sweepPt = PARAMS.map(() => 1);

  // Initialise per-(track,plane) refs from the stored signal-peak defaults
  p0.track_names.forEach(track => {
    p0.plane_names.forEach(plane => {
      const r = p0.refs[track] && p0.refs[track][plane];
      _refs[track + '/' + plane] = {wire: r ? r.wire : 0, time: r ? r.time : 0};
    });
  });

  // EVD track dropdown
  const evdTrackSel = document.getElementById('evd-track');
  evdTrackSel.innerHTML = '';
  p0.track_names.forEach(n => {
    const o = document.createElement('option'); o.value=n; o.textContent=n;
    evdTrackSel.appendChild(o);
  });
  evdTrackSel.onchange = () => {
    _evdTrack = evdTrackSel.value;
    _syncRefInputs(); _drawEvd();
  };

  // EVD plane buttons
  const evdPb = document.getElementById('evd-plane-btns');
  evdPb.innerHTML = '';
  p0.plane_names.forEach(p => {
    const b = document.createElement('button');
    b.id='evd-pb-'+p; b.textContent=p;
    b.className='plane-btn'+(p===_evdPlane?' active':'');
    b.onclick = () => {
      _updateEvdPlane(p);
      _syncRefInputs(); _drawEvd(); _drawTraces();
    };
    evdPb.appendChild(b);
  });

  // Per-param slider rows + View button
  const paramRows = document.getElementById('evd-param-rows');
  paramRows.innerHTML = '';
  PARAMS.forEach((param, pi) => {
    const row = document.createElement('div');
    row.className = 'row';

    const lbl = document.createElement('span');
    lbl.className='lbl'; lbl.style.width='130px';
    lbl.textContent = param.param_label+':';
    row.appendChild(lbl);

    const sl = document.createElement('input');
    sl.type='range'; sl.className='slider'; sl.id='evd-sl-'+pi;
    sl.min='0'; sl.max=String(param.n_sweep); sl.step='1'; sl.value='1';
    row.appendChild(sl);

    const valSpan = document.createElement('span');
    valSpan.className='val'; valSpan.id='evd-sl-val-'+pi;
    row.appendChild(valSpan);

    const viewBtn = document.createElement('button');
    viewBtn.id='evd-param-view-'+pi; viewBtn.textContent='View';
    viewBtn.className='plane-btn'+(pi===0?' active':'');
    viewBtn.style.marginLeft='8px';
    row.appendChild(viewBtn);

    paramRows.appendChild(row);

    _updateSlLabel(pi);
    sl.oninput = () => {
      _updateSlLabel(pi);
      if (_evdParamIdx === pi) _drawEvd();
      _drawTraces();
    };
    viewBtn.onclick = () => {
      _updateEvdParamBtn(pi);
      _syncRefInputs(); _drawEvd();
    };
  });

  // Trace: track dropdown
  const trTrackSel = document.getElementById('tr-track');
  trTrackSel.innerHTML = '';
  p0.track_names.forEach(n => {
    const o = document.createElement('option'); o.value=n; o.textContent=n;
    trTrackSel.appendChild(o);
  });
  trTrackSel.onchange = () => { _trTrack = trTrackSel.value; _drawTraces(); };

  // Trace: param buttons
  const trParamBtns = document.getElementById('tr-param-btns');
  trParamBtns.innerHTML = '';
  PARAMS.forEach((param, pi) => {
    const b = document.createElement('button');
    b.id='tr-pb-param-'+pi; b.textContent=param.param_label;
    b.className='plane-btn'+(pi===0?' active':'');
    b.onclick = () => { _updateTrParamBtn(pi); _drawTraces(); };
    trParamBtns.appendChild(b);
  });

  // Histogram plane buttons
  _histPlane = p0.plane_names[0];
  const histPb = document.getElementById('hist-plane-btns');
  histPb.innerHTML = '';
  p0.plane_names.forEach(p => {
    const b = document.createElement('button');
    b.id = 'hist-pb-' + p; b.textContent = p;
    b.className = 'plane-btn' + (p === _histPlane ? ' active' : '');
    b.onclick = () => {
      _histPlane = p;
      p0.plane_names.forEach(n => {
        const bb = document.getElementById('hist-pb-' + n);
        if (bb) bb.className = 'plane-btn' + (n === p ? ' active' : '');
      });
      _drawHist();
    };
    histPb.appendChild(b);
  });

  _syncRefInputs(); _renderTable(); _drawEvd(); _drawTraces();
}

window.addEventListener('load', () => {
  // Mode buttons
  document.querySelectorAll('.mode-btn').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn').forEach(x => x.classList.remove('active'));
      b.classList.add('active'); _evdMode = b.dataset.mode; _drawEvd();
    });
  });

  // Noise checkboxes — hide if no param has paired noisy+clean data
  const anyNoise = PARAMS.some(p => p.has_noise);
  document.getElementById('noise-label').style.display   = anyNoise ? '' : 'none';
  document.getElementById('tr-noise-row').style.display  = anyNoise ? '' : 'none';
  document.getElementById('noise-cb').addEventListener('change', e => {
    _noiseOn = e.target.checked; _drawEvd(); _drawTraces();
  });
  document.getElementById('tr-noise-cb').addEventListener('change', _drawTraces);

  // Wire/time ref inputs
  document.getElementById('wire-ref').addEventListener('change', () => {
    _evdRef().wire = +document.getElementById('wire-ref').value; _drawEvd(); _drawTraces();
  });
  document.getElementById('time-ref').addEventListener('change', () => {
    _evdRef().time = +document.getElementById('time-ref').value; _drawEvd(); _drawTraces();
  });

  // Action buttons
  document.getElementById('btn-add').addEventListener('click', _addTrace);
  document.getElementById('btn-upd').addEventListener('click', _updateSel);
  document.getElementById('btn-cxl').addEventListener('click', _cancelEdit);

  document.getElementById('loading').style.display = 'none';
  _initUI();
});
</script>
</body>
</html>
"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--pkl', help='Single 1d_gradients pkl file')
    g.add_argument('--dir', help='Directory — generate one viewer per *.pkl found')
    p.add_argument('--output', default=None,
                   help='Output HTML path override (overrides the auto plots/ routing)')
    p.add_argument('--plots-dir', default=os.environ.get('PLOTS_DIR', 'plots'),
                   help='Root plots directory (default: $PLOTS_DIR env var, or "plots")')
    p.add_argument('--results-base', default=os.environ.get('RESULTS_DIR', 'results'),
                   help='Results root used to compute the plots/ sub-path '
                        '(default: $RESULTS_DIR env var, or "results")')
    p.add_argument('--bbox-threshold', type=float, default=BBOX_THRESHOLD,
                   help=f'Signal fraction threshold for bounding-box detection '
                        f'(default: {BBOX_THRESHOLD})')
    p.add_argument('--pad-wire', type=int, default=BBOX_PAD_WIRE,
                   help=f'Extra wires added outside signal bounding box (default: {BBOX_PAD_WIRE})')
    p.add_argument('--pad-time', type=int, default=BBOX_PAD_TIME,
                   help=f'Extra time bins added outside signal bounding box (default: {BBOX_PAD_TIME})')
    p.add_argument('--max-wire', type=int, default=MAX_WIRE,
                   help=f'Hard cap on stored wire window (default: {MAX_WIRE})')
    p.add_argument('--max-time', type=int, default=MAX_TIME,
                   help=f'Hard cap on stored time window (default: {MAX_TIME})')
    return p.parse_args()


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def _load_pkl(pkl_path: Path):
    """Load one pkl and return its data dict, or None if arrays are missing."""
    print(f'  Loading {pkl_path.name} …')
    with open(pkl_path, 'rb') as f:
        pkl = pickle.load(f)
    if 'per_track_gt_arrays' not in pkl:
        print(f'  SKIP {pkl_path.name}: no array data (re-run with --store-arrays)')
        return None
    print(f'  Building data …')
    data = build_data(pkl)
    data['_pkl_path'] = pkl_path
    print(f'  Tracks: {data["track_names"]},  Planes: {data["plane_names"]},  '
          f'N sweep={data["n_sweep"]},  noise={data["noise_scale"]}')
    return data


def _group_by_param(all_data: list) -> list:
    """Merge noisy/clean variants of the same param into one entry each.

    When both variants exist, the noisy arrays are rebuilt using the clean GT
    bbox.  Noise inflates the threshold-based bbox to the full detector (noise
    RMS ≈ 2% threshold), so the default crop centres on the detector midpoint
    and misses the track.  Using the clean bbox keeps both views aligned.
    """
    by_param = defaultdict(dict)
    for d in all_data:
        key = 'noisy' if d['noise_scale'] > 0 else 'clean'
        by_param[d['param_name']][key] = d

    params_list = []
    for param_name in sorted(by_param.keys()):
        variants = by_param[param_name]
        clean = variants.get('clean')
        noisy = variants.get('noisy')
        base = clean if clean else noisy

        # Rebuild noisy arrays with the clean bbox so the track stays visible.
        if clean and noisy and '_pkl_path' in noisy:
            print(f'  Rebuilding noisy arrays for {param_name} using clean bbox …')
            with open(noisy['_pkl_path'], 'rb') as f:
                noisy_pkl = pickle.load(f)
            noisy_rebuilt = build_data(noisy_pkl, bbox_override=clean['bboxes'])
            noisy_arrays = noisy_rebuilt['arrays']
        else:
            noisy_arrays = noisy['arrays'] if noisy else None

        entry = {
            'param_name':   param_name,
            'param_label':  base['param_label'],
            'param_gt':     float(base['param_gt']),
            'param_values': base['param_values'],
            'factors':      base['factors'],
            'pt_labels':    base['pt_labels'],
            'gt_pt_idx':    base['gt_pt_idx'],
            'n_sweep':      base['n_sweep'],
            'plane_names':  base['plane_names'],
            'track_names':  base['track_names'],
            'refs':         base['refs'],
            'has_noise':    (clean is not None and noisy is not None),
            'noise_scale':  float(noisy['noise_scale']) if noisy else 0.0,
            'arrays_clean': clean['arrays'] if clean else base['arrays'],
            'arrays_noisy': noisy_arrays,
        }
        params_list.append(entry)
    return params_list


def _write_html(params_list: list, out_path: Path, title: str):
    print(f'  Serialising {len(params_list)} param(s) …')
    params_json = json.dumps(params_list, cls=_NpEncoder, separators=(',', ':'))
    html = (HTML_TEMPLATE
            .replace('__PARAMS_JSON__', params_json)
            .replace('__TITLE__', title))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding='utf-8')
    size_mb = out_path.stat().st_size / 1e6
    print(f'  → {out_path}  ({size_mb:.1f} MB)')
    print(f'  Open: file://{out_path.resolve()}')


def main():
    args = parse_args()

    global BBOX_THRESHOLD, BBOX_PAD_WIRE, BBOX_PAD_TIME, MAX_WIRE, MAX_TIME
    BBOX_THRESHOLD = args.bbox_threshold
    BBOX_PAD_WIRE  = args.pad_wire
    BBOX_PAD_TIME  = args.pad_time
    MAX_WIRE       = args.max_wire
    MAX_TIME       = args.max_time

    def _auto_out(src_dir: Path, filename: str) -> Path:
        """Route output to plots/<rel-path>/<filename>, mirroring the results tree."""
        try:
            rel = src_dir.resolve().relative_to(Path(args.results_base).resolve())
            return Path(args.plots_dir) / rel / filename
        except ValueError:
            return src_dir / filename

    if args.pkl:
        pkl_path = Path(args.pkl)
        out_path = Path(args.output) if args.output else _auto_out(pkl_path.parent, pkl_path.stem + '.html')
        data = _load_pkl(pkl_path)
        if data:
            params_list = _group_by_param([data])
            title = f'1D Gradient Sweep — {params_list[0]["param_label"]}'
            _write_html(params_list, out_path, title)
    else:
        dir_path = Path(args.dir)
        pkls = sorted(dir_path.glob('*.pkl'))
        if not pkls:
            print(f'No *.pkl files found in {dir_path}')
            return
        print(f'Found {len(pkls)} pkl file(s) in {dir_path}')
        all_data = [d for p in pkls if (d := _load_pkl(p)) is not None]
        if not all_data:
            print('No datasets with array data found.')
            return
        params_list = _group_by_param(all_data)
        print(f'  Grouped into {len(params_list)} param(s): {[p["param_name"] for p in params_list]}')
        out_path = Path(args.output) if args.output else _auto_out(dir_path, 'viewer.html')
        title = f'1D Gradient Sweep — {dir_path.name}'
        _write_html(params_list, out_path, title)


if __name__ == '__main__':
    main()
