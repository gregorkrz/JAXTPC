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


def _inflate_b64z(b64_str: str) -> np.ndarray:
    """Decompress a b64z-encoded flat float32 array."""
    return np.frombuffer(zlib.decompress(base64.b64decode(b64_str)), dtype=np.float32).copy()


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

    pixel_loss_all  = pkl.get('per_track_pixel_loss')
    pixel_grad_all  = pkl.get('per_track_pixel_grad')
    fourier_C_all       = pkl.get('per_track_fourier_C')
    fourier_pwr_all     = pkl.get('per_track_fourier_power')
    fourier_sim_fft_all = pkl.get('per_track_fourier_sim_fft')
    fourier_W_all       = pkl.get('per_track_fourier_W')
    fourier_gt_fft_all  = pkl.get('per_track_fourier_gt_fft')
    fourier_res         = pkl.get('fourier_map_resolution', 128)
    has_fourier         = (fourier_C_all is not None and fourier_pwr_all is not None
                           and fourier_W_all is not None)

    n_sweep = len(pkl['param_values'])
    noise_scale = float(pkl.get('noise_scale', 0.0))

    arrays = {}
    refs   = {}
    bboxes = {}
    dims   = {}
    pixel_arrays   = {}
    fourier_arrays = {}
    for track in track_names:
        arrays[track] = {}
        refs[track]   = {}
        dims[track]   = {}
        pixel_arrays[track] = {}
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
            dims[track][plane] = {'wire_lo': wl, 'time_lo': tl, 'n_wire': nw, 'n_time': nt}

            if pixel_loss_all and pixel_grad_all:
                loss_sweeps = pixel_loss_all[track]
                grad_sweeps = pixel_grad_all[track]
                stacked_loss = np.stack([
                    np.array(loss_sweeps[sw][pi], dtype=np.float32)[wl:wh, tl:th]
                    for sw in range(n_sweep)
                ], axis=0)
                stacked_grad = np.stack([
                    np.array(grad_sweeps[sw][pi], dtype=np.float32)[wl:wh, tl:th]
                    for sw in range(n_sweep)
                ], axis=0)
                pixel_arrays[track][plane] = {
                    'wire_lo': wl, 'time_lo': tl, 'n_wire': nw, 'n_time': nt,
                    'data_loss': _b64z(stacked_loss),
                    'data_grad': _b64z(stacked_grad),
                }

            if has_fourier and track in fourier_C_all:
                C_sweeps       = fourier_C_all[track]
                pwr_sweeps     = fourier_pwr_all[track]
                W_planes       = fourier_W_all[track]
                sim_fft_sweeps = (fourier_sim_fft_all or {}).get(track)
                gt_fft_planes  = (fourier_gt_fft_all or {}).get(track)
                if pi < len(C_sweeps[0]):
                    stacked_C   = np.stack([np.array(C_sweeps[sw][pi],   dtype=np.float32) for sw in range(n_sweep)], axis=0)
                    stacked_pwr = np.stack([np.array(pwr_sweeps[sw][pi], dtype=np.float32) for sw in range(n_sweep)], axis=0)
                    W_arr = np.array(W_planes[pi], dtype=np.float32)
                    entry = {
                        'n_freq': fourier_res,
                        'data_C':     _b64z(stacked_C),
                        'data_power': _b64z(stacked_pwr),
                        'data_W':     _b64z(W_arr),
                    }
                    if sim_fft_sweeps is not None:
                        stacked_sim = np.stack([np.array(sim_fft_sweeps[sw][pi], dtype=np.float32) for sw in range(n_sweep)], axis=0)
                        entry['data_sim_fft'] = _b64z(stacked_sim)
                    if gt_fft_planes is not None:
                        entry['data_gt_fft'] = _b64z(np.array(gt_fft_planes[pi], dtype=np.float32))
                    fourier_arrays.setdefault(track, {})[plane] = entry

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

    print(f'  pixel maps: {"yes" if pixel_loss_all else "no"}  '
          f'fourier maps: {"yes" if has_fourier else "no"}')

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
        'dims':          dims,
        'bboxes':        bboxes,
        'loss_total':    pkl['loss_values'],
        'grad_total':    pkl['grad_values'],
        'loss_per_track': pkl['per_track_loss_values'],
        'grad_per_track': pkl['per_track_grad_values'],
        'palette':       PALETTE,
        'ds_label':      param_label + noise_str,
        'adc_cutoff':     float(pkl.get('adc_cutoff', 0.0)),
        'sobolev_s':      float(pkl.get('sobolev_s', 2.0)),
        'fourier_cutoff': float(pkl.get('fourier_cutoff', 0.0)),
        'pixel_arrays':   pixel_arrays if (pixel_loss_all and pixel_grad_all) else None,
        'fourier_arrays': fourier_arrays if has_fourier else None,
    }


# ── keys that hold large binary blobs (excluded from PARAMS meta JSON) ────────
_DATA_KEYS = ('arrays_clean', 'arrays_noisy', 'pixel_arrays_clean', 'pixel_arrays_noisy',
              'fourier_arrays_clean', 'fourier_arrays_noisy')

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
#evd-plot{height:360px}
#evd-pixel-loss{height:360px}
#evd-pixel-grad{height:360px}
#evd-fourier-C{height:360px}
#evd-fourier-power{height:360px}
#evd-fourier-W{height:360px}
#evd-fourier-sim{height:360px}
#evd-fourier-gt{height:360px}
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

    <div class="row">
      <span class="lbl">Mask |GT|&lt;:</span>
      <input type="number" id="pixel-mask-thresh" min="0" step="0.1" value="0">
      <span class="hint">ADC &nbsp;(0 = show all pixels; masked pixels are black)</span>
    </div>

    <div class="row" style="margin-top:6px">
      <div style="display:flex;gap:4px">
        <button class="mode-btn active" id="evd-tab-signal-btn" onclick="_switchEvdTab('signal')">Signal</button>
        <button class="mode-btn"        id="evd-tab-fourier-btn" onclick="_switchEvdTab('fourier')">Fourier</button>
      </div>
    </div>

    <div id="evd-tab-signal">
      <div class="row" style="margin-top:6px">
        <span class="lbl">Fourier cut:</span>
        <input type="number" id="signal-fourier-cutoff" min="0" step="0.001" value="0">
        <span class="hint">|FFT|²/N &nbsp;(0 = show all; filters the displayed signal in-place)</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;align-items:start;margin-top:6px">
        <div id="evd-plot" style="cursor:crosshair"></div>
        <div id="evd-pixel-loss"></div>
        <div id="evd-pixel-grad"></div>
      </div>
    </div>

    <div id="evd-tab-fourier" style="display:none">
      <div id="fourier-cutoff-row" style="display:none;margin-top:6px;margin-bottom:2px">
        <span style="font-size:12px;color:#555">ADC cutoff:</span>
        <span id="fourier-cutoff-btns" style="display:inline-flex;gap:4px;margin-left:6px"></span>
      </div>
      <div id="fourier-sobolev-row" style="display:none;margin-top:4px;margin-bottom:2px">
        <span style="font-size:12px;color:#555">Sobolev s:</span>
        <span id="fourier-sobolev-btns" style="display:inline-flex;gap:4px;margin-left:6px"></span>
      </div>
      <div id="fourier-fcutoff-row" style="display:none;margin-top:4px;margin-bottom:2px">
        <span style="font-size:12px;color:#555">Fourier cutoff:</span>
        <span id="fourier-fcutoff-btns" style="display:inline-flex;gap:4px;margin-left:6px"></span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;align-items:start;margin-top:6px">
        <div id="evd-fourier-C"></div>
        <div id="evd-fourier-power"></div>
        <div id="evd-fourier-W"></div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;align-items:start;margin-top:10px">
        <div id="evd-fourier-sim"></div>
        <div id="evd-fourier-gt"></div>
      </div>
      <div id="fourier-slice-card" style="display:none;margin-top:10px;border-top:1px solid #eee;padding-top:10px">
        <div style="font-size:12px;color:#555;margin-bottom:6px">
          Fourier slice — f_wire=<span id="fslice-fw" style="font-family:monospace">—</span>,
          f_time=<span id="fslice-ft" style="font-family:monospace">—</span>
          <span style="color:#aaa;font-size:11px;margin-left:8px">(click any Fourier map to update)</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;align-items:start">
          <div id="fourier-slice-wire" style="height:260px"></div>
          <div id="fourier-slice-time" style="height:260px"></div>
        </div>
      </div>
    </div>

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
const PARAMS = __PARAMS_META_JSON__;
const _DATA_DIR = '__DATA_DIR_NAME__';

/* ── on-demand value loading ─────────────────────────────────────────────────
   _loadValue(pi, vi) injects a <script> tag the first time it's called for a
   given (pi, vi) pair.  The script calls VALUE_DATA_READY(pi, vi, data) when
   it executes, which resolves _whenValueLoaded[pi][vi].promise and triggers a
   redraw if the UI is ready.  Nothing is fetched until you ask for it.
── */
const _valueLoaded = PARAMS.map(p => new Array(p.n_sweep + 1).fill(false));
const _valueRequested = PARAMS.map(p => new Array(p.n_sweep + 1).fill(false));
const _whenValueLoaded = PARAMS.map(p =>
  Array.from({length: p.n_sweep + 1}, () => {
    const o = {}; o.promise = new Promise(r => { o._r = r; }); return o;
  })
);

function _loadValue(pi, vi) {
  if (_valueRequested[pi][vi]) return;
  _valueRequested[pi][vi] = true;
  const s = document.createElement('script');
  s.src = _DATA_DIR + '/param_' + pi + '_val_' + vi + '.js';
  document.head.appendChild(s);
}

/* Buffer storage: [pi][vi]['track/plane'] = Float32Array (single frame, n_wire×n_time) */
const _cleanBufs = PARAMS.map(p => Array.from({length: p.n_sweep + 1}, () => ({})));
const _noisyBufs = PARAMS.map(p => Array.from({length: p.n_sweep + 1}, () => ({})));
/* Pixel bufs: [pi][vi]['track/plane'] = {loss: Float32Array, grad: Float32Array} */
const _pixelCleanBufs = PARAMS.map(p => Array.from({length: p.n_sweep + 1}, () => ({})));
const _pixelNoisyBufs = PARAMS.map(p => Array.from({length: p.n_sweep + 1}, () => ({})));
/* Fourier bufs: [pi][vi]['track/plane'] = {C: Float32Array, power: Float32Array, W: Float32Array} */
const _fourierCleanBufs = PARAMS.map(p => Array.from({length: p.n_sweep + 1}, () => ({})));
const _fourierNoisyBufs = PARAMS.map(p => Array.from({length: p.n_sweep + 1}, () => ({})));

function _inflateB64(b64) {
  const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  return new Float32Array(pako.inflate(bin).buffer);
}

function _storeValueData(pi, vi, data) {
  if (data.arrays_clean) {
    for (const track of Object.keys(data.arrays_clean)) {
      for (const plane of Object.keys(data.arrays_clean[track])) {
        _cleanBufs[pi][vi][track + '/' + plane] = _inflateB64(data.arrays_clean[track][plane].data);
      }
    }
  }
  if (data.arrays_noisy) {
    for (const track of Object.keys(data.arrays_noisy)) {
      for (const plane of Object.keys(data.arrays_noisy[track])) {
        _noisyBufs[pi][vi][track + '/' + plane] = _inflateB64(data.arrays_noisy[track][plane].data);
      }
    }
  }
  if (data.pixel_clean) {
    for (const track of Object.keys(data.pixel_clean)) {
      for (const plane of Object.keys(data.pixel_clean[track])) {
        const pa = data.pixel_clean[track][plane];
        _pixelCleanBufs[pi][vi][track + '/' + plane] = {
          loss: _inflateB64(pa.data_loss), grad: _inflateB64(pa.data_grad),
        };
      }
    }
  }
  if (data.pixel_noisy) {
    for (const track of Object.keys(data.pixel_noisy)) {
      for (const plane of Object.keys(data.pixel_noisy[track])) {
        const pa = data.pixel_noisy[track][plane];
        _pixelNoisyBufs[pi][vi][track + '/' + plane] = {
          loss: _inflateB64(pa.data_loss), grad: _inflateB64(pa.data_grad),
        };
      }
    }
  }
  if (data.fourier_clean) {
    for (const track of Object.keys(data.fourier_clean)) {
      for (const plane of Object.keys(data.fourier_clean[track])) {
        const fa = data.fourier_clean[track][plane];
        _fourierCleanBufs[pi][vi][track + '/' + plane] = {
          C: _inflateB64(fa.data_C), power: _inflateB64(fa.data_power), W: _inflateB64(fa.data_W),
          sim_fft: fa.data_sim_fft ? _inflateB64(fa.data_sim_fft) : null,
          gt_fft:  fa.data_gt_fft  ? _inflateB64(fa.data_gt_fft)  : null,
        };
      }
    }
  }
  if (data.fourier_noisy) {
    for (const track of Object.keys(data.fourier_noisy)) {
      for (const plane of Object.keys(data.fourier_noisy[track])) {
        const fa = data.fourier_noisy[track][plane];
        _fourierNoisyBufs[pi][vi][track + '/' + plane] = {
          C: _inflateB64(fa.data_C), power: _inflateB64(fa.data_power), W: _inflateB64(fa.data_W),
          sim_fft: fa.data_sim_fft ? _inflateB64(fa.data_sim_fft) : null,
          gt_fft:  fa.data_gt_fft  ? _inflateB64(fa.data_gt_fft)  : null,
        };
      }
    }
  }
}

function VALUE_DATA_READY(pi, vi, data) {
  _storeValueData(pi, vi, data);
  _valueLoaded[pi][vi] = true;
  _whenValueLoaded[pi][vi]._r(vi);
  if (_uiReady) {
    const needsEvd = (_evdParamIdx === pi) && (vi === 0 || vi === _sweepPt[pi]);
    const needsFourier = (_evdTab === 'fourier') && (_fourierParamIdx === pi)
                       && (vi === 0 || vi === _sweepPt[pi]);
    const needsTrace = (vi === 0) || _entries.some(e => e.pi === pi && e.pt === vi)
                     || (_trParamIdx === pi && vi === _sweepPt[pi]);
    if (needsEvd || needsTrace) { _drawEvd(); _drawTraces(); }
    else if (needsFourier) { _drawFourierMaps(); }
  }
}

/* ── Fourier picker selection state ── */
let _fourierSelAdc      = 0;
let _fourierSelSobolevS = 2.0;
let _fourierSelFcutoff  = 0.0;

/* ── Radix-2 FFT (in-place Cooley-Tukey) for client-side signal filtering ── */
function _nextPow2(n) { let p = 1; while (p < n) p <<= 1; return p; }
function _fft1d(re, im) {
  const N = re.length;
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= N; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wR = Math.cos(ang), wI = Math.sin(ang);
    for (let i = 0; i < N; i += len) {
      let cR = 1, cI = 0;
      for (let j = 0; j < len >> 1; j++) {
        const uR = re[i+j], uI = im[i+j];
        const vR = re[i+j+(len>>1)] * cR - im[i+j+(len>>1)] * cI;
        const vI = re[i+j+(len>>1)] * cI + im[i+j+(len>>1)] * cR;
        re[i+j] = uR + vR; im[i+j] = uI + vI;
        re[i+j+(len>>1)] = uR - vR; im[i+j+(len>>1)] = uI - vI;
        const nr = cR * wR - cI * wI; cI = cR * wI + cI * wR; cR = nr;
      }
    }
  }
}
function _ifft1d(re, im) {
  for (let i = 0; i < im.length; i++) im[i] = -im[i];
  _fft1d(re, im);
  const N = re.length;
  for (let i = 0; i < N; i++) { re[i] /= N; im[i] = -im[i] / N; }
}
function _applyFourierCutoff2d(z2d, cutoff) {
  const nw = z2d.length, nt = z2d[0].length;
  const Pw = _nextPow2(nw), Pt = _nextPow2(nt);
  const N = Pw * Pt;
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let w = 0; w < nw; w++)
    for (let t = 0; t < nt; t++)
      re[w * Pt + t] = z2d[w][t];
  // Row FFTs
  for (let w = 0; w < Pw; w++) {
    const rR = re.subarray(w*Pt, (w+1)*Pt), rI = im.subarray(w*Pt, (w+1)*Pt);
    _fft1d(rR, rI);
  }
  // Column FFTs
  const cR = new Float64Array(Pw), cI = new Float64Array(Pw);
  for (let t = 0; t < Pt; t++) {
    for (let w = 0; w < Pw; w++) { cR[w] = re[w*Pt+t]; cI[w] = im[w*Pt+t]; }
    _fft1d(cR, cI);
    for (let w = 0; w < Pw; w++) { re[w*Pt+t] = cR[w]; im[w*Pt+t] = cI[w]; }
  }
  // Threshold mask
  for (let i = 0; i < N; i++) {
    if ((re[i]*re[i] + im[i]*im[i]) / N < cutoff) { re[i] = 0; im[i] = 0; }
  }
  // Column IFFTs
  for (let t = 0; t < Pt; t++) {
    for (let w = 0; w < Pw; w++) { cR[w] = re[w*Pt+t]; cI[w] = im[w*Pt+t]; }
    _ifft1d(cR, cI);
    for (let w = 0; w < Pw; w++) { re[w*Pt+t] = cR[w]; im[w*Pt+t] = cI[w]; }
  }
  // Row IFFTs
  for (let w = 0; w < Pw; w++) {
    const rR = re.subarray(w*Pt, (w+1)*Pt), rI = im.subarray(w*Pt, (w+1)*Pt);
    _ifft1d(rR, rI);
  }
  const out = [];
  for (let w = 0; w < nw; w++) {
    const row = [];
    for (let t = 0; t < nt; t++) row.push(re[w*Pt+t]);
    out.push(row);
  }
  return out;
}

/* ── colour palette ── */
const _PALETTE = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
  '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
  '#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5'];

/* ── data access ─────────────────────────────────────────────────────────────
   All functions take an explicit pt (0=GT, 1..n_sweep=sweep).
   Dims come from PARAMS meta so they're always available before any value loads.
── */
function _dims(pi, track, plane) {
  return (PARAMS[pi].dims[track] && PARAMS[pi].dims[track][plane]) || null;
}

function _getBuf(pi, pt, track, plane, useNoisy) {
  if (!_valueLoaded[pi][pt]) return null;
  const k = track + '/' + plane;
  if (useNoisy) return _noisyBufs[pi][pt][k] || _cleanBufs[pi][pt][k] || null;
  return _cleanBufs[pi][pt][k] || null;
}

function _getZ(pi, track, plane, pt, noisy=false) {
  const a = _dims(pi, track, plane);
  if (!a) return [];
  const buf = _getBuf(pi, pt, track, plane, pt === 0 && noisy);
  if (!buf) return [];
  const z = [];
  for (let w = 0; w < a.n_wire; w++) {
    const row = [];
    for (let t = 0; t < a.n_time; t++) row.push(buf[w * a.n_time + t]);
    z.push(row);
  }
  return z;
}

function _getZDiff(pi, track, plane, pt, noisy=false) {
  const a = _dims(pi, track, plane);
  if (!a) return [];
  const bufS = _getBuf(pi, pt, track, plane, false);
  const bufG = _getBuf(pi, 0, track, plane, noisy);
  if (!bufS || !bufG) return [];
  const z = [];
  for (let w = 0; w < a.n_wire; w++) {
    const row = [];
    for (let t = 0; t < a.n_time; t++)
      row.push(bufS[w * a.n_time + t] - bufG[w * a.n_time + t]);
    z.push(row);
  }
  return z;
}

function _getTimeTrace(pi, track, plane, pt, wireAbs, noisy=false) {
  const a = _dims(pi, track, plane);
  if (!a) return null;
  const w = wireAbs - a.wire_lo;
  if (w < 0 || w >= a.n_wire) return null;
  const buf = _getBuf(pi, pt, track, plane, pt === 0 && noisy);
  if (!buf) return null;
  const off = w * a.n_time;
  return {x: Array.from({length:a.n_time}, (_,i) => a.time_lo+i),
          y: Array.from(buf.slice(off, off+a.n_time))};
}

function _getWireTrace(pi, track, plane, pt, timeAbs, noisy=false) {
  const a = _dims(pi, track, plane);
  if (!a) return null;
  const t = timeAbs - a.time_lo;
  if (t < 0 || t >= a.n_time) return null;
  const buf = _getBuf(pi, pt, track, plane, pt === 0 && noisy);
  if (!buf) return null;
  const y = [];
  for (let w = 0; w < a.n_wire; w++) y.push(buf[w * a.n_time + t]);
  return {x: Array.from({length:a.n_wire}, (_,i) => a.wire_lo+i), y};
}

function _getPixelZ(pi, track, plane, pt, type, useNoisy) {
  if (pt === 0 || !_valueLoaded[pi][pt]) return null;
  const k = track + '/' + plane;
  const noisyEntry = _pixelNoisyBufs[pi][pt][k];
  const cleanEntry = _pixelCleanBufs[pi][pt][k];
  const entry = (useNoisy && noisyEntry) ? noisyEntry : cleanEntry;
  if (!entry) return null;
  const buf = entry[type];
  if (!buf) return null;
  const a = _dims(pi, track, plane);
  const nw = a.n_wire, nt = a.n_time;
  const thresh = parseFloat(document.getElementById('pixel-mask-thresh').value) || 0;
  let gtBuf = null;
  if (thresh > 0) gtBuf = _getBuf(pi, 0, track, plane, useNoisy);
  const z = [];
  for (let w = 0; w < nw; w++) {
    const row = [];
    for (let t = 0; t < nt; t++) {
      const val = buf[w * nt + t];
      if (thresh > 0 && gtBuf) {
        row.push(Math.abs(gtBuf[w * nt + t]) < thresh ? null : val);
      } else {
        row.push(val);
      }
    }
    z.push(row);
  }
  return z;
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
let _uiReady          = false;

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
  if (!a) return;
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

  if (!z.length) return;  // value not yet loaded — skip until VALUE_DATA_READY fires

  const _sfCutoff = parseFloat(document.getElementById('signal-fourier-cutoff')?.value || '0') || 0;
  if (_sfCutoff > 0) z = _applyFourierCutoff2d(z, _sfCutoff);

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
  _drawPixelMaps();
  _drawFourierMaps();
  _saveState();
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

let _pixelLossInited = false, _pixelGradInited = false;

function _drawPixelMaps() {
  const pi = _evdParamIdx;
  const pt = Math.max(1, _sweepPt[pi]);
  const hasPx = PARAMS[pi].has_pixel_clean || PARAMS[pi].has_pixel_noisy;
  const lossEl = document.getElementById('evd-pixel-loss');
  const gradEl = document.getElementById('evd-pixel-grad');
  if (!hasPx) {
    const msg = '<div style="color:#aaa;padding:140px 10px;text-align:center;font-size:12px">No pixel data<br>(re-run with --store-per-pixel-loss-and-grad)</div>';
    lossEl.innerHTML = msg; gradEl.innerHTML = msg;
    _pixelLossInited = false; _pixelGradInited = false;
    return;
  }
  const a = _dims(pi, _evdTrack, _evdPlane);
  const xArr = Array.from({length:a.n_time}, (_,i) => a.time_lo+i);
  const yArr = Array.from({length:a.n_wire}, (_,i) => a.wire_lo+i);
  const zLoss = _getPixelZ(pi, _evdTrack, _evdPlane, pt, 'loss', _noiseOn) || [];
  const zGrad = _getPixelZ(pi, _evdTrack, _evdPlane, pt, 'grad', _noiseOn) || [];
  let gradMax = 1e-9;
  zGrad.forEach(row => row.forEach(v => { if (v !== null && Math.abs(v) > gradMax) gradMax = Math.abs(v); }));
  const ptLbl = _ptLabel(pi, pt);
  const noiseTag = _noiseOn ? ' [noisy]' : '';
  const cfg = {responsive:true};
  const traceLoss = {type:'heatmap', x:xArr, y:yArr, z:zLoss,
    colorscale:'Reds', zmin:0, connectgaps:false,
    colorbar:{title:'Loss', thickness:14, len:0.9}};
  const lyLoss = {
    title:{text:_evdTrack+' — '+_evdPlane+' — Pixel Loss'+noiseTag+' @ '+ptLbl, font:{size:12}, x:0.04},
    xaxis:{title:'Time bin'}, yaxis:{title:'Wire index'},
    margin:{t:32,b:50,l:60,r:80}, paper_bgcolor:'#fff', plot_bgcolor:'#111'};
  const traceGrad = {type:'heatmap', x:xArr, y:yArr, z:zGrad,
    colorscale:'RdBu', reversescale:true, zmin:-gradMax, zmax:gradMax, connectgaps:false,
    colorbar:{title:'∂L/∂sim', thickness:14, len:0.9}};
  const lyGrad = {
    title:{text:_evdTrack+' — '+_evdPlane+' — ∂L/∂sim'+noiseTag+' @ '+ptLbl, font:{size:12}, x:0.04},
    xaxis:{title:'Time bin'}, yaxis:{title:'Wire index'},
    margin:{t:32,b:50,l:60,r:80}, paper_bgcolor:'#fff', plot_bgcolor:'#111'};
  if (!_pixelLossInited) { Plotly.newPlot('evd-pixel-loss',[traceLoss],lyLoss,cfg); _pixelLossInited=true; }
  else { Plotly.react('evd-pixel-loss',[traceLoss],lyLoss,cfg); }
  if (!_pixelGradInited) { Plotly.newPlot('evd-pixel-grad',[traceGrad],lyGrad,cfg); _pixelGradInited=true; }
  else { Plotly.react('evd-pixel-grad',[traceGrad],lyGrad,cfg); }
}

let _fourierCInited = false, _fourierPwrInited = false, _fourierWInited = false;
let _fourierSimInited = false, _fourierGtInited = false;
let _fourierClickF = {wire: 0.0, time: 0.0};
let _fourierSliceWireInited = false, _fourierSliceTimeInited = false;
let _evdTab = 'signal';
let _fourierParamIdx = 0;

function _resetFourierInited() {
  _fourierCInited = false; _fourierPwrInited = false; _fourierWInited = false;
  _fourierSimInited = false; _fourierGtInited = false;
  _fourierSliceWireInited = false; _fourierSliceTimeInited = false;
}

function _findMatchingParam(baseName, adc, sobolevS, fCutoff) {
  for (let i = 0; i < PARAMS.length; i++) {
    const p = PARAMS[i];
    if ((p.param_name_base || p.param_name) === baseName &&
        Math.abs((p.adc_cutoff || 0) - adc) < 1e-9 &&
        Math.abs((p.sobolev_s || 2.0) - sobolevS) < 1e-6 &&
        Math.abs((p.fourier_cutoff || 0) - fCutoff) < 1e-9) return i;
  }
  return -1;
}

function _updateFourierCutoffBtns() {
  const baseName = PARAMS[_evdParamIdx].param_name_base || PARAMS[_evdParamIdx].param_name;
  const allEntries = PARAMS.map((p, i) => ({
    pi: i, adc: p.adc_cutoff || 0, s: p.sobolev_s || 2.0, fc: p.fourier_cutoff || 0,
    base: p.param_name_base || p.param_name,
  })).filter(e => e.base === baseName);

  const prevIdx = _fourierParamIdx;

  // Resolve _fourierParamIdx from the current selection state
  let newIdx = _findMatchingParam(baseName, _fourierSelAdc, _fourierSelSobolevS, _fourierSelFcutoff);
  if (newIdx < 0) {
    // Fall back to first entry matching baseName, then update selection state
    newIdx = allEntries.length ? allEntries[0].pi : _evdParamIdx;
    _fourierSelAdc      = PARAMS[newIdx].adc_cutoff || 0;
    _fourierSelSobolevS = PARAMS[newIdx].sobolev_s  || 2.0;
    _fourierSelFcutoff  = PARAMS[newIdx].fourier_cutoff || 0;
  }
  _fourierParamIdx = newIdx;

  function _makePicker(rowId, btnsId, uniqueVals, getVal, label, setVal) {
    const row  = document.getElementById(rowId);
    const btns = document.getElementById(btnsId);
    if (!row || !btns) return;
    btns.innerHTML = '';
    if (uniqueVals.length <= 1) { row.style.display = 'none'; return; }
    row.style.display = '';
    uniqueVals.forEach(v => {
      const b = document.createElement('button');
      b.textContent = label(v);
      b.className = 'mode-btn' + (Math.abs(getVal() - v) < 1e-9 ? ' active' : '');
      b.onclick = () => {
        setVal(v);
        const ni = _findMatchingParam(baseName, _fourierSelAdc, _fourierSelSobolevS, _fourierSelFcutoff);
        if (ni >= 0 && ni !== _fourierParamIdx) {
          _fourierParamIdx = ni;
          _resetFourierInited();
          _loadValue(_fourierParamIdx, 0);
          _loadValue(_fourierParamIdx, _sweepPt[_fourierParamIdx]);
        }
        _updateFourierCutoffBtns();
        _saveState();
        _drawFourierMaps();
      };
      btns.appendChild(b);
    });
  }

  const uniqAdc = [...new Set(allEntries.map(e => e.adc))].sort((a,b) => a-b);
  const uniqS   = [...new Set(allEntries.map(e => e.s))].sort((a,b) => a-b);
  const uniqFc  = [...new Set(allEntries.map(e => e.fc))].sort((a,b) => a-b);

  _makePicker('fourier-cutoff-row', 'fourier-cutoff-btns', uniqAdc,
    () => _fourierSelAdc,
    v => v === 0 ? 'No cutoff' : String(v) + ' ADC',
    v => { _fourierSelAdc = v; });
  _makePicker('fourier-sobolev-row', 'fourier-sobolev-btns', uniqS,
    () => _fourierSelSobolevS,
    v => 's=' + v,
    v => { _fourierSelSobolevS = v; });
  _makePicker('fourier-fcutoff-row', 'fourier-fcutoff-btns', uniqFc,
    () => _fourierSelFcutoff,
    v => v === 0 ? 'No cutoff' : 'fc=' + v,
    v => { _fourierSelFcutoff = v; });

  if (prevIdx !== _fourierParamIdx) _resetFourierInited();
}

function _switchEvdTab(tab) {
  _evdTab = tab;
  document.getElementById('evd-tab-signal').style.display   = tab === 'signal'  ? '' : 'none';
  document.getElementById('evd-tab-fourier').style.display  = tab === 'fourier' ? '' : 'none';
  document.getElementById('evd-tab-signal-btn').className  = 'mode-btn' + (tab === 'signal'  ? ' active' : '');
  document.getElementById('evd-tab-fourier-btn').className = 'mode-btn' + (tab === 'fourier' ? ' active' : '');
  _saveState();
  if (tab === 'fourier') {
    _updateFourierCutoffBtns();
    _loadValue(_fourierParamIdx, 0);
    _loadValue(_fourierParamIdx, _sweepPt[_fourierParamIdx]);
    _drawFourierMaps();
    ['evd-fourier-C','evd-fourier-power','evd-fourier-W','evd-fourier-sim','evd-fourier-gt',
     'fourier-slice-wire','fourier-slice-time'].forEach(id => {
      const el = document.getElementById(id);
      if (el && el._fullLayout) Plotly.Plots.resize(el);
    });
  } else {
    ['evd-plot','evd-pixel-loss','evd-pixel-grad'].forEach(id => {
      const el = document.getElementById(id);
      if (el && el._fullLayout) Plotly.Plots.resize(el);
    });
  }
}

function _onFourierClick(d) {
  if (!d.points.length) return;
  _fourierClickF.time = parseFloat(d.points[0].x);
  _fourierClickF.wire = parseFloat(d.points[0].y);
  _drawFourierMaps();
}

function _drawFourierSlices() {
  const pi = _fourierParamIdx;
  const pt = Math.max(1, _sweepPt[pi]);
  const card = document.getElementById('fourier-slice-card');
  const entry = _getFourierEntry(pi, pt, _evdTrack, _evdPlane, _noiseOn);
  if (!entry) { card.style.display = 'none'; return; }

  let gtEntry = null;
  for (let v = 1; v <= PARAMS[pi].n_sweep; v++) {
    const e = _getFourierEntry(pi, v, _evdTrack, _evdPlane, _noiseOn);
    if (e && e.gt_fft) { gtEntry = e; break; }
  }

  card.style.display = '';
  const N = Math.round(Math.sqrt(entry.C.length));
  const fVals = Array.from({length:N}, (_,i) => i / N - 0.5);
  const fw = _fourierClickF.wire, ft = _fourierClickF.time;
  const rowIdx = Math.max(0, Math.min(N-1, Math.round((fw + 0.5) * N)));
  const colIdx = Math.max(0, Math.min(N-1, Math.round((ft + 0.5) * N)));

  document.getElementById('fslice-fw').textContent = fw.toFixed(3);
  document.getElementById('fslice-ft').textContent = ft.toFixed(3);

  const TINY = 1e-30;
  function rowSlice(buf) {
    const y = [];
    for (let c = 0; c < N; c++) y.push(Math.log10(buf[rowIdx * N + c] + TINY));
    return y;
  }
  function colSlice(buf) {
    const y = [];
    for (let r = 0; r < N; r++) y.push(Math.log10(buf[r * N + colIdx] + TINY));
    return y;
  }

  const noiseTag = _noiseOn ? ' [noisy]' : '';
  const datasets = [
    {name:'C(f)',    buf:entry.C,     color:'#1f77b4'},
    {name:'|D̂|²/N', buf:entry.power, color:'#ff7f0e'},
    {name:'W(f)',    buf:entry.W,     color:'#2ca02c'},
  ];
  if (entry.sim_fft) datasets.push({name:'|Ŝ|²/N', buf:entry.sim_fft, color:'#d62728'});
  if (gtEntry && gtEntry.gt_fft) datasets.push({name:'|GT̂|²/N', buf:gtEntry.gt_fft, color:'#9467bd'});

  const hTraces = datasets.map(d => ({
    x:fVals, y:rowSlice(d.buf), name:d.name,
    type:'scatter', mode:'lines', line:{color:d.color, width:1.5},
  }));
  const vTraces = datasets.map(d => ({
    x:fVals, y:colSlice(d.buf), name:d.name,
    type:'scatter', mode:'lines', line:{color:d.color, width:1.5},
  }));

  const cfg = {responsive:true};
  const marg = {t:36,b:50,l:60,r:20};
  const lyH = {
    title:{text:'Slice at f_wire='+fw.toFixed(3)+noiseTag, font:{size:12}, x:0.04},
    xaxis:{title:'f_time (cycles/px)'}, yaxis:{title:'log₁₀'},
    legend:{font:{size:10}}, margin:marg,
  };
  const lyV = {
    title:{text:'Slice at f_time='+ft.toFixed(3)+noiseTag, font:{size:12}, x:0.04},
    xaxis:{title:'f_wire (cycles/px)'}, yaxis:{title:'log₁₀'},
    legend:{font:{size:10}}, margin:marg,
  };

  if (!_fourierSliceWireInited) { Plotly.newPlot('fourier-slice-wire', hTraces, lyH, cfg); _fourierSliceWireInited=true; }
  else { Plotly.react('fourier-slice-wire', hTraces, lyH, cfg); }
  if (!_fourierSliceTimeInited) { Plotly.newPlot('fourier-slice-time', vTraces, lyV, cfg); _fourierSliceTimeInited=true; }
  else { Plotly.react('fourier-slice-time', vTraces, lyV, cfg); }
}

function _getFourierEntry(pi, pt, track, plane, useNoisy) {
  if (pt === 0 || !_valueLoaded[pi][pt]) return null;
  const k = track + '/' + plane;
  const noisy = _fourierNoisyBufs[pi][pt][k];
  const clean = _fourierCleanBufs[pi][pt][k];
  return (useNoisy && noisy) ? noisy : (clean || null);
}

function _fourierZ(buf, N) {
  /* buf is Float32Array of length N*N; returns 2D row-major array with log10 scaling */
  const TINY = 1e-30;
  const z = [];
  for (let r = 0; r < N; r++) {
    const row = [];
    for (let c = 0; c < N; c++) row.push(Math.log10(buf[r * N + c] + TINY));
    z.push(row);
  }
  return z;
}

function _drawFourierMaps() {
  const pi = _fourierParamIdx;
  const pt = Math.max(1, _sweepPt[pi]);
  const hasFq = PARAMS[pi].has_fourier_clean || PARAMS[pi].has_fourier_noisy;
  const cEl   = document.getElementById('evd-fourier-C');
  const pEl   = document.getElementById('evd-fourier-power');
  const wEl   = document.getElementById('evd-fourier-W');
  const sEl   = document.getElementById('evd-fourier-sim');
  const gEl   = document.getElementById('evd-fourier-gt');
  if (!hasFq) {
    const msg = '<div style="color:#aaa;padding:140px 10px;text-align:center;font-size:12px">No Fourier data<br>(re-run with --store-per-pixel-loss-and-grad)</div>';
    cEl.innerHTML = msg; pEl.innerHTML = msg; wEl.innerHTML = msg;
    sEl.innerHTML = msg; gEl.innerHTML = msg;
    _fourierCInited = false; _fourierPwrInited = false; _fourierWInited = false;
    _fourierSimInited = false; _fourierGtInited = false;
    return;
  }
  const entry = _getFourierEntry(pi, pt, _evdTrack, _evdPlane, _noiseOn);
  if (!entry) return;
  const N = Math.round(Math.sqrt(entry.C.length));
  const fArr = Array.from({length:N}, (_,i) => (i / N - 0.5).toFixed(3));
  const f0 = parseFloat(fArr[0]), f1 = parseFloat(fArr[fArr.length-1]);
  const _fShapes = [
    {type:'line', x0:_fourierClickF.time, x1:_fourierClickF.time, y0:f0, y1:f1,
     line:{color:'#00E5FF', width:1, dash:'dot'}},
    {type:'line', x0:f0, x1:f1, y0:_fourierClickF.wire, y1:_fourierClickF.wire,
     line:{color:'#00E5FF', width:1, dash:'dot'}},
  ];
  const ptLbl = _ptLabel(pi, pt);
  const noiseTag = _noiseOn ? ' [noisy]' : '';
  const cfg = {responsive:true};
  const marg = {t:32,b:50,l:60,r:80};
  const axOpts = {title:'Frequency (cycles/px)'};

  const zC   = _fourierZ(entry.C,     N);
  const zPwr = _fourierZ(entry.power, N);
  const zW   = _fourierZ(entry.W,     N);

  const trC = {type:'heatmap', x:fArr, y:fArr, z:zC,
    colorscale:'Viridis', connectgaps:false,
    colorbar:{title:'log₁₀ C(f)', thickness:14, len:0.9}};
  const lyC = {
    title:{text:_evdTrack+' — '+_evdPlane+' — Fourier Loss C(f)'+noiseTag+' @ '+ptLbl, font:{size:12}, x:0.04},
    xaxis:axOpts, yaxis:axOpts, margin:marg, paper_bgcolor:'#fff', plot_bgcolor:'#111', shapes:_fShapes};

  const trPwr = {type:'heatmap', x:fArr, y:fArr, z:zPwr,
    colorscale:'Viridis', connectgaps:false,
    colorbar:{title:'log₁₀ |D̂|²/N', thickness:14, len:0.9}};
  const lyPwr = {
    title:{text:_evdTrack+' — '+_evdPlane+' — Power |D̂|²/N'+noiseTag+' @ '+ptLbl, font:{size:12}, x:0.04},
    xaxis:axOpts, yaxis:axOpts, margin:marg, paper_bgcolor:'#fff', plot_bgcolor:'#111', shapes:_fShapes};

  const trW = {type:'heatmap', x:fArr, y:fArr, z:zW,
    colorscale:'Viridis', connectgaps:false,
    colorbar:{title:'log₁₀ W(f)', thickness:14, len:0.9}};
  const lyW = {
    title:{text:_evdPlane+' — Sobolev Weight W(f)', font:{size:12}, x:0.04},
    xaxis:axOpts, yaxis:axOpts, margin:marg, paper_bgcolor:'#fff', plot_bgcolor:'#111', shapes:_fShapes};

  if (!_fourierCInited) {
    Plotly.newPlot('evd-fourier-C', [trC], lyC, cfg);
    document.getElementById('evd-fourier-C').on('plotly_click', _onFourierClick);
    _fourierCInited=true;
  } else { Plotly.react('evd-fourier-C', [trC], lyC, cfg); }
  if (!_fourierPwrInited) {
    Plotly.newPlot('evd-fourier-power', [trPwr], lyPwr, cfg);
    document.getElementById('evd-fourier-power').on('plotly_click', _onFourierClick);
    _fourierPwrInited=true;
  } else { Plotly.react('evd-fourier-power', [trPwr], lyPwr, cfg); }
  if (!_fourierWInited) {
    Plotly.newPlot('evd-fourier-W', [trW], lyW, cfg);
    document.getElementById('evd-fourier-W').on('plotly_click', _onFourierClick);
    _fourierWInited=true;
  } else { Plotly.react('evd-fourier-W', [trW], lyW, cfg); }

  if (entry.sim_fft) {
    const zSim = _fourierZ(entry.sim_fft, N);
    const trSim = {type:'heatmap', x:fArr, y:fArr, z:zSim,
      colorscale:'Inferno', connectgaps:false,
      colorbar:{title:'log₁₀ |Ŝ|²/N', thickness:14, len:0.9}};
    const lySim = {
      title:{text:_evdTrack+' — '+_evdPlane+' — Sim |FFT|²/N'+noiseTag+' @ '+ptLbl, font:{size:12}, x:0.04},
      xaxis:axOpts, yaxis:axOpts, margin:marg, paper_bgcolor:'#fff', plot_bgcolor:'#111', shapes:_fShapes};
    if (!_fourierSimInited) {
      Plotly.newPlot('evd-fourier-sim', [trSim], lySim, cfg);
      document.getElementById('evd-fourier-sim').on('plotly_click', _onFourierClick);
      _fourierSimInited=true;
    } else { Plotly.react('evd-fourier-sim', [trSim], lySim, cfg); }
  }

  /* GT FFT is fixed — grab from the first loaded sweep point for this param */
  let gtEntry = null;
  for (let v = 1; v <= PARAMS[pi].n_sweep; v++) {
    gtEntry = _getFourierEntry(pi, v, _evdTrack, _evdPlane, _noiseOn);
    if (gtEntry && gtEntry.gt_fft) break;
  }
  if (gtEntry && gtEntry.gt_fft) {
    const zGt = _fourierZ(gtEntry.gt_fft, N);
    const trGt = {type:'heatmap', x:fArr, y:fArr, z:zGt,
      colorscale:'Inferno', connectgaps:false,
      colorbar:{title:'log₁₀ |GT̂|²/N', thickness:14, len:0.9}};
    const lyGt = {
      title:{text:_evdTrack+' — '+_evdPlane+' — GT |FFT|²/N', font:{size:12}, x:0.04},
      xaxis:axOpts, yaxis:axOpts, margin:marg, paper_bgcolor:'#fff', plot_bgcolor:'#111', shapes:_fShapes};
    if (!_fourierGtInited) {
      Plotly.newPlot('evd-fourier-gt', [trGt], lyGt, cfg);
      document.getElementById('evd-fourier-gt').on('plotly_click', _onFourierClick);
      _fourierGtInited=true;
    } else { Plotly.react('evd-fourier-gt', [trGt], lyGt, cfg); }
  }

  _drawFourierSlices();
}

function _evdRef() { return _refs[_evdTrack + '/' + _evdPlane]; }
function _syncRefInputs() {
  const r = _evdRef();
  document.getElementById('wire-ref').value = r.wire;
  document.getElementById('time-ref').value = r.time;
}

function _saveState() {
  if (!_uiReady || !history.replaceState) return;
  const sp = new URLSearchParams();
  sp.set('t',   _evdTrack);
  sp.set('pl',  _evdPlane);
  sp.set('m',   _evdMode);
  sp.set('pi',  String(_evdParamIdx));
  sp.set('sp',  _sweepPt.join(','));
  sp.set('n',   _noiseOn ? '1' : '0');
  sp.set('tab', _evdTab);
  sp.set('fpi', String(_fourierParamIdx));
  sp.set('fadc', String(_fourierSelAdc));
  sp.set('fs',   String(_fourierSelSobolevS));
  sp.set('ffc',  String(_fourierSelFcutoff));
  const maskEl = document.getElementById('pixel-mask-thresh');
  if (maskEl && parseFloat(maskEl.value) > 0) sp.set('mask', maskEl.value);
  const sfcEl = document.getElementById('signal-fourier-cutoff');
  if (sfcEl && parseFloat(sfcEl.value) > 0) sp.set('sfc', sfcEl.value);
  history.replaceState(null, '', window.location.pathname + '?' + sp.toString());
}

function _restoreFromURL() {
  const sp = new URLSearchParams(window.location.search);
  if (!sp.has('t') && !sp.has('sp')) return;
  const allTracks = PARAMS[0].track_names;
  const allPlanes = PARAMS[0].plane_names;
  const t = sp.get('t');
  if (t && allTracks.includes(t)) {
    _evdTrack = t;
    const sel = document.getElementById('evd-track');
    if (sel) sel.value = t;
  }
  const pl = sp.get('pl');
  if (pl && allPlanes.includes(pl)) _updateEvdPlane(pl);
  const m = sp.get('m');
  if (m && ['sim','gt','diff'].includes(m)) {
    _evdMode = m;
    document.querySelectorAll('.mode-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.mode === m));
  }
  const pi = parseInt(sp.get('pi'));
  if (!isNaN(pi) && pi >= 0 && pi < PARAMS.length) _updateEvdParamBtn(pi);
  const spStr = sp.get('sp');
  if (spStr) spStr.split(',').forEach((v, i) => {
    const pt = parseInt(v);
    if (i < _sweepPt.length && !isNaN(pt) && pt >= 0 && pt <= PARAMS[i].n_sweep) {
      _sweepPt[i] = pt;
      const sl = document.getElementById('evd-sl-' + i);
      if (sl) { sl.value = String(pt); _updateSlLabel(i); }
    }
  });
  const n = sp.get('n');
  if (n !== null) {
    _noiseOn = n === '1';
    const cb = document.getElementById('noise-cb');
    if (cb) cb.checked = _noiseOn;
  }
  const mask = sp.get('mask');
  if (mask !== null) {
    const el = document.getElementById('pixel-mask-thresh');
    if (el) el.value = mask;
  }
  const tab = sp.get('tab');
  if (tab === 'fourier') {
    // Defer tab switch until after _initUI finishes so the Fourier cutoff buttons exist.
    requestAnimationFrame(() => _switchEvdTab('fourier'));
  }
  const fpi = parseInt(sp.get('fpi'));
  if (!isNaN(fpi) && fpi >= 0 && fpi < PARAMS.length) {
    _fourierParamIdx = fpi;
    _resetFourierInited();
  }
  const fadc = parseFloat(sp.get('fadc'));
  if (!isNaN(fadc)) _fourierSelAdc = fadc;
  const fs = parseFloat(sp.get('fs'));
  if (!isNaN(fs)) _fourierSelSobolevS = fs;
  const ffc = parseFloat(sp.get('ffc'));
  if (!isNaN(ffc)) _fourierSelFcutoff = ffc;
  const sfc = sp.get('sfc');
  if (sfc !== null) {
    const el = document.getElementById('signal-fourier-cutoff');
    if (el) el.value = sfc;
  }
  _syncRefInputs();
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
  if (!a) return [];
  const buf = _getBuf(e.pi, e.pt, e.track, plane, e.noisy || false);
  if (!buf) return [];
  return Array.from(buf);
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
  // Keep Fourier tab in sync with signal param when signal tab is active.
  if (_evdTab === 'signal') { _fourierParamIdx = pi; _resetFourierInited(); }
  _loadValue(pi, 0);
  _loadValue(pi, _sweepPt[pi]);
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
  _evdParamIdx = 0; _trParamIdx = 0; _fourierParamIdx = 0;
  _trTrack = p0.track_names[0];
  _sweepPt = PARAMS.map(() => 1);
  _fourierSelAdc      = p0.adc_cutoff || 0;
  _fourierSelSobolevS = p0.sobolev_s  || 2.0;
  _fourierSelFcutoff  = p0.fourier_cutoff || 0;

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

  // Per-param slider rows + View button (signal tab: cutoff=0 only)
  const paramRows = document.getElementById('evd-param-rows');
  paramRows.innerHTML = '';
  PARAMS.forEach((param, pi) => {
    if ((param.adc_cutoff || 0) !== 0) return; // non-zero cutoffs live in Fourier tab only
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
      const vi = _sweepPt[pi];
      _loadValue(pi, 0);
      _loadValue(pi, vi);
      if (_evdParamIdx === pi) _drawEvd();
      else if (_evdTab === 'fourier' && _fourierParamIdx === pi) _drawFourierMaps();
      _drawTraces();
    };
    viewBtn.onclick = () => {
      _loadValue(pi, 0);
      _loadValue(pi, _sweepPt[pi]);
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

  // Trace: param buttons (cutoff=0 only)
  const trParamBtns = document.getElementById('tr-param-btns');
  trParamBtns.innerHTML = '';
  PARAMS.forEach((param, pi) => {
    if ((param.adc_cutoff || 0) !== 0) return;
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

  _restoreFromURL();
  _syncRefInputs();
  _uiReady = true;
  _renderTable(); _drawEvd(); _drawTraces();
}

window.addEventListener('load', async () => {
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
  document.getElementById('pixel-mask-thresh').addEventListener('input', () => { _drawPixelMaps(); _saveState(); });
  document.getElementById('signal-fourier-cutoff').addEventListener('input', () => { _drawEvd(); _saveState(); });

  // Action buttons
  document.getElementById('btn-add').addEventListener('click', _addTrace);
  document.getElementById('btn-upd').addEventListener('click', _updateSel);
  document.getElementById('btn-cxl').addEventListener('click', _cancelEdit);

  // Load GT + first sweep point of first param, then show UI
  _loadValue(0, 0);
  _loadValue(0, 1);
  await Promise.all([
    _whenValueLoaded[0][0].promise,
    _whenValueLoaded[0][1].promise,
  ]);
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
        raw_pkl = pickle.load(f)
    if 'per_track_gt_arrays' not in raw_pkl:
        print(f'  SKIP {pkl_path.name}: no array data (re-run with --store-arrays)')
        return None
    print(f'  Building data …')
    data = build_data(raw_pkl)
    data['_raw_pkl'] = raw_pkl
    print(f'  Tracks: {data["track_names"]},  Planes: {data["plane_names"]},  '
          f'N sweep={data["n_sweep"]},  noise={data["noise_scale"]}')
    return data


def _merge_key(pkl: dict) -> tuple:
    """Grouping key: pkls with the same key belong to one sweep series."""
    return (
        pkl.get('param_name'),
        round(float(pkl.get('noise_scale', 0.0)), 6),
        int(pkl.get('noise_seed', 42)),
        round(float(pkl.get('adc_cutoff', 0.0)), 6),
        round(float(pkl.get('sobolev_s', 2.0)), 6),
        round(float(pkl.get('fourier_cutoff', 0.0)), 6),
    )


def _merge_pkl_group(pkls: list) -> dict:
    """Merge multiple single-sweep-point pkls (from per-factor jobs) into one.

    Pkls are sorted by factor value.  Sweep-indexed lists are concatenated;
    non-sweep fields (GT arrays, metadata) come from the pkl whose factor is
    closest to 1.0 (GT point), which is always present.
    """
    pkls_sorted = sorted(pkls, key=lambda p: (p['factors'] or [0])[0])
    base = min(pkls_sorted,
               key=lambda p: abs((p['factors'] or [0])[0] - 1.0))

    merged = dict(base)

    sweep_scalar_keys = ['factors', 'param_values', 'p_n_values',
                         'loss_values', 'grad_values', 'grad_times_s']
    for k in sweep_scalar_keys:
        merged[k] = []
        for p in pkls_sorted:
            merged[k].extend(p.get(k, []))

    track_names = [ts['name'] for ts in base['track_specs']]
    plane_names = base.get('plane_names', [])

    merged['per_track_loss_values'] = {t: [] for t in track_names}
    merged['per_track_grad_values'] = {t: [] for t in track_names}
    for p in pkls_sorted:
        for t in track_names:
            merged['per_track_loss_values'][t].extend(p['per_track_loss_values'][t])
            merged['per_track_grad_values'][t].extend(p['per_track_grad_values'][t])

    if 'per_track_sim_arrays' in base:
        merged['per_track_sim_arrays'] = {t: [] for t in track_names}
        for p in pkls_sorted:
            if 'per_track_sim_arrays' in p:
                for t in track_names:
                    merged['per_track_sim_arrays'][t].extend(p['per_track_sim_arrays'][t])

    if 'per_track_pixel_loss' in base:
        merged['per_track_pixel_loss'] = {t: [] for t in track_names}
        merged['per_track_pixel_grad'] = {t: [] for t in track_names}
        for p in pkls_sorted:
            if 'per_track_pixel_loss' in p:
                for t in track_names:
                    merged['per_track_pixel_loss'][t].extend(p['per_track_pixel_loss'][t])
                    merged['per_track_pixel_grad'][t].extend(p['per_track_pixel_grad'][t])

    if 'per_plane_loss_values' in base:
        merged['per_plane_loss_values'] = {t: {pn: [] for pn in plane_names}
                                            for t in track_names}
        for p in pkls_sorted:
            if 'per_plane_loss_values' in p:
                for t in track_names:
                    for pn in plane_names:
                        merged['per_plane_loss_values'][t][pn].extend(
                            p['per_plane_loss_values'][t][pn])

    for fourier_key in ('per_track_fourier_C', 'per_track_fourier_power',
                        'per_track_fourier_sim_fft'):
        if fourier_key in base:
            merged[fourier_key] = {t: [] for t in track_names}
            for p in pkls_sorted:
                if fourier_key in p:
                    for t in track_names:
                        merged[fourier_key][t].extend(p[fourier_key][t])

    return merged


def _scan_dir_groups(dir_path: Path) -> dict:
    """Scan pkls for grouping metadata only; return {merge_key: [Path, ...]}.

    Loads each pkl fully (pickle has no partial-read API) but drops array data
    immediately so only one pkl's heap is live at a time.
    """
    pkls_paths = sorted(dir_path.glob('*.pkl'))
    if not pkls_paths:
        print(f'No *.pkl files found in {dir_path}')
        return {}
    print(f'Found {len(pkls_paths)} pkl file(s) in {dir_path}')
    groups: dict = defaultdict(list)
    for pkl_path in pkls_paths:
        print(f'  Scanning {pkl_path.name} …')
        try:
            with open(pkl_path, 'rb') as f:
                raw = pickle.load(f)
        except Exception as e:
            print(f'  SKIP {pkl_path.name}: failed to load ({e}) — delete and re-run the job')
            continue
        if 'per_track_gt_arrays' not in raw:
            print(f'  SKIP {pkl_path.name}: no array data (re-run with --store-arrays)')
            continue
        groups[_merge_key(raw)].append(pkl_path)
        # raw goes out of scope; large arrays freed
    return dict(groups)


def _load_and_merge_paths(paths: list) -> dict:
    """Load (and merge if needed) a list of pkl paths into one raw pkl dict."""
    raws = []
    for pkl_path in paths:
        print(f'  Loading {pkl_path.name} …')
        with open(pkl_path, 'rb') as f:
            raws.append(pickle.load(f))
    if len(raws) == 1:
        return raws[0]
    print(f'  Merging {len(raws)} factor pkls …')
    return _merge_pkl_group(raws)


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
        adc_cutoff    = round(float(d.get('adc_cutoff', 0.0)), 6)
        sobolev_s     = round(float(d.get('sobolev_s', 2.0)), 6)
        fourier_cutoff = round(float(d.get('fourier_cutoff', 0.0)), 6)
        by_param[(d['param_name'], adc_cutoff, sobolev_s, fourier_cutoff)][key] = d

    params_list = []
    for (param_name, adc_cutoff, sobolev_s, fourier_cutoff) in sorted(by_param.keys()):
        variants = by_param[(param_name, adc_cutoff, sobolev_s, fourier_cutoff)]
        clean = variants.get('clean')
        noisy = variants.get('noisy')
        base = clean if clean else noisy

        # Rebuild noisy arrays with the clean bbox so the track stays visible.
        if clean and noisy and '_raw_pkl' in noisy:
            print(f'  Rebuilding noisy arrays for {param_name} cutoff={adc_cutoff:g} s={sobolev_s:g} fc={fourier_cutoff:g} using clean bbox …')
            noisy_rebuilt = build_data(noisy['_raw_pkl'], bbox_override=clean['bboxes'])
            noisy_arrays = noisy_rebuilt['arrays']
            noisy_pixel  = noisy_rebuilt['pixel_arrays']
        elif clean and noisy and '_pkl_path' in noisy:  # legacy fallback
            print(f'  Rebuilding noisy arrays for {param_name} cutoff={adc_cutoff:g} s={sobolev_s:g} fc={fourier_cutoff:g} using clean bbox …')
            with open(noisy['_pkl_path'], 'rb') as f:
                noisy_pkl = pickle.load(f)
            noisy_rebuilt = build_data(noisy_pkl, bbox_override=clean['bboxes'])
            noisy_arrays = noisy_rebuilt['arrays']
            noisy_pixel  = noisy_rebuilt['pixel_arrays']
        else:
            noisy_arrays = noisy['arrays'] if noisy else None
            noisy_pixel  = noisy['pixel_arrays'] if noisy else None

        cutoff_label = ''
        if adc_cutoff > 0:
            cutoff_label += f' [cut {adc_cutoff:g}]'
        if sobolev_s != 2.0:
            cutoff_label += f' [s={sobolev_s:g}]'
        if fourier_cutoff > 0:
            cutoff_label += f' [fc={fourier_cutoff:g}]'
        entry = {
            'param_name':      param_name,
            'param_name_base': param_name,
            'adc_cutoff':      adc_cutoff,
            'sobolev_s':       sobolev_s,
            'fourier_cutoff':  fourier_cutoff,
            'param_label':  base['param_label'] + cutoff_label,
            'param_gt':     float(base['param_gt']),
            'param_values': base['param_values'],
            'factors':      base['factors'],
            'pt_labels':    base['pt_labels'],
            'gt_pt_idx':    base['gt_pt_idx'],
            'n_sweep':      base['n_sweep'],
            'plane_names':  base['plane_names'],
            'track_names':  base['track_names'],
            'refs':         base['refs'],
            'dims':         base['dims'],
            'has_noise':    (clean is not None and noisy is not None),
            'noise_scale':  float(noisy['noise_scale']) if noisy else 0.0,
            # Large binary blobs — kept in params_list but excluded from PARAMS meta JSON
            'arrays_clean':         clean['arrays'] if clean else base['arrays'],
            'arrays_noisy':         noisy_arrays,
            'pixel_arrays_clean':   clean['pixel_arrays'] if clean else None,
            'pixel_arrays_noisy':   noisy_pixel,
            'fourier_arrays_clean': clean['fourier_arrays'] if clean else None,
            'fourier_arrays_noisy': noisy_rebuilt['fourier_arrays'] if (clean and noisy and '_raw_pkl' in noisy) else (noisy['fourier_arrays'] if noisy else None),
        }
        params_list.append(entry)
    return params_list


def _write_param_js(pi: int, p: dict, data_dir: Path, data_dir_name: str) -> dict:
    """Write per-value JS files for one param entry; return param_meta.

    Frees large binary blobs from ``p`` after writing so the caller can hold only
    one param's worth of array data in memory at a time.
    """
    n_sweep = p['n_sweep']

    param_meta = {k: v for k, v in p.items()
                  if k not in _DATA_KEYS and not k.startswith('_')}
    param_meta['has_pixel_clean']   = p.get('pixel_arrays_clean') is not None
    param_meta['has_pixel_noisy']   = p.get('pixel_arrays_noisy') is not None
    param_meta['has_fourier_clean'] = p.get('fourier_arrays_clean') is not None
    param_meta['has_fourier_noisy'] = p.get('fourier_arrays_noisy') is not None

    arrays_clean  = p.get('arrays_clean')
    arrays_noisy  = p.get('arrays_noisy')
    pixel_clean   = p.get('pixel_arrays_clean')
    pixel_noisy   = p.get('pixel_arrays_noisy')
    fourier_clean = p.get('fourier_arrays_clean')
    fourier_noisy = p.get('fourier_arrays_noisy')

    # Write one JS file per sweep value (vi=0 = GT, vi=1..n_sweep = sim pts).
    for vi in range(n_sweep + 1):
        val_data: dict = {}

        if arrays_clean:
            val_data['arrays_clean'] = {}
            for track, tdata in arrays_clean.items():
                for plane, pdata in tdata.items():
                    nw, nt = pdata['n_wire'], pdata['n_time']
                    frames = _inflate_b64z(pdata['data']).reshape(-1, nw, nt)
                    if vi < len(frames):
                        val_data['arrays_clean'].setdefault(track, {})[plane] = {
                            'data': _b64z(frames[vi]),
                        }

        if arrays_noisy:
            val_data['arrays_noisy'] = {}
            for track, tdata in arrays_noisy.items():
                for plane, pdata in tdata.items():
                    nw, nt = pdata['n_wire'], pdata['n_time']
                    frames = _inflate_b64z(pdata['data']).reshape(-1, nw, nt)
                    if vi < len(frames):
                        val_data['arrays_noisy'].setdefault(track, {})[plane] = {
                            'data': _b64z(frames[vi]),
                        }

        # Pixel arrays only exist for sweep points (vi >= 1); sw_idx is 0-based
        if vi >= 1:
            sw_idx = vi - 1

            if pixel_clean:
                val_data['pixel_clean'] = {}
                for track, tdata in pixel_clean.items():
                    for plane, pdata in tdata.items():
                        nw, nt = pdata['n_wire'], pdata['n_time']
                        lframes = _inflate_b64z(pdata['data_loss']).reshape(-1, nw, nt)
                        gframes = _inflate_b64z(pdata['data_grad']).reshape(-1, nw, nt)
                        if sw_idx < len(lframes):
                            val_data['pixel_clean'].setdefault(track, {})[plane] = {
                                'data_loss': _b64z(lframes[sw_idx]),
                                'data_grad': _b64z(gframes[sw_idx]),
                            }

            if pixel_noisy:
                val_data['pixel_noisy'] = {}
                for track, tdata in pixel_noisy.items():
                    for plane, pdata in tdata.items():
                        nw, nt = pdata['n_wire'], pdata['n_time']
                        lframes = _inflate_b64z(pdata['data_loss']).reshape(-1, nw, nt)
                        gframes = _inflate_b64z(pdata['data_grad']).reshape(-1, nw, nt)
                        if sw_idx < len(lframes):
                            val_data['pixel_noisy'].setdefault(track, {})[plane] = {
                                'data_loss': _b64z(lframes[sw_idx]),
                                'data_grad': _b64z(gframes[sw_idx]),
                            }

            if fourier_clean:
                val_data['fourier_clean'] = {}
                for track, tdata in fourier_clean.items():
                    for plane, pdata in tdata.items():
                        N = pdata['n_freq']
                        cframes = _inflate_b64z(pdata['data_C']).reshape(-1, N, N)
                        if sw_idx < len(cframes):
                            pframes = _inflate_b64z(pdata['data_power']).reshape(-1, N, N) if pdata.get('data_power') else None
                            W_arr   = _inflate_b64z(pdata['data_W']).reshape(N, N)          if pdata.get('data_W')     else None
                            sframes = _inflate_b64z(pdata['data_sim_fft']).reshape(-1, N, N) if pdata.get('data_sim_fft') else None
                            gt_arr  = _inflate_b64z(pdata['data_gt_fft']).reshape(N, N)      if pdata.get('data_gt_fft')  else None
                            val_data['fourier_clean'].setdefault(track, {})[plane] = {
                                'data_C':       _b64z(cframes[sw_idx]),
                                'data_power':   _b64z(pframes[sw_idx]) if pframes is not None else '',
                                'data_W':       _b64z(W_arr)           if W_arr   is not None else '',
                                'data_sim_fft': _b64z(sframes[sw_idx]) if sframes is not None else '',
                                'data_gt_fft':  _b64z(gt_arr)          if gt_arr  is not None else '',
                            }

            if fourier_noisy:
                val_data['fourier_noisy'] = {}
                for track, tdata in fourier_noisy.items():
                    for plane, pdata in tdata.items():
                        N = pdata['n_freq']
                        cframes = _inflate_b64z(pdata['data_C']).reshape(-1, N, N)
                        if sw_idx < len(cframes):
                            pframes = _inflate_b64z(pdata['data_power']).reshape(-1, N, N) if pdata.get('data_power') else None
                            W_arr   = _inflate_b64z(pdata['data_W']).reshape(N, N)          if pdata.get('data_W')     else None
                            sframes = _inflate_b64z(pdata['data_sim_fft']).reshape(-1, N, N) if pdata.get('data_sim_fft') else None
                            gt_arr  = _inflate_b64z(pdata['data_gt_fft']).reshape(N, N)      if pdata.get('data_gt_fft')  else None
                            val_data['fourier_noisy'].setdefault(track, {})[plane] = {
                                'data_C':       _b64z(cframes[sw_idx]),
                                'data_power':   _b64z(pframes[sw_idx]) if pframes is not None else '',
                                'data_W':       _b64z(W_arr)           if W_arr   is not None else '',
                                'data_sim_fft': _b64z(sframes[sw_idx]) if sframes is not None else '',
                                'data_gt_fft':  _b64z(gt_arr)          if gt_arr  is not None else '',
                            }

        data_js = (f'VALUE_DATA_READY({pi},{vi},'
                   + json.dumps(val_data, cls=_NpEncoder, separators=(',', ':'))
                   + ');')
        val_path = data_dir / f'param_{pi}_val_{vi}.js'
        val_path.write_text(data_js, encoding='utf-8')
        print(f'  → {val_path.name}  ({val_path.stat().st_size / 1e6:.2f} MB)')
    # Free large binary blobs — written to JS, no longer needed.
    for key in list(_DATA_KEYS):
        p.pop(key, None)

    return param_meta


def _write_html(params_list: list, out_path: Path, title: str):
    """Write viewer HTML with per-(param, value) data files; values load on demand."""
    print(f'  Serialising {len(params_list)} param(s) …')

    data_dir_name = out_path.stem + '_data'
    data_dir = out_path.parent / data_dir_name
    data_dir.mkdir(parents=True, exist_ok=True)

    all_params_meta = []
    for pi, p in enumerate(params_list):
        param_meta = _write_param_js(pi, p, data_dir, data_dir_name)
        all_params_meta.append(param_meta)

    _write_html_shell(all_params_meta, data_dir_name, out_path, title)


def _write_html_shell(params_meta: list, data_dir_name: str, out_path: Path, title: str):
    """Write the HTML file; JS value files are fetched on demand via _loadValue()."""
    params_meta_json = json.dumps(params_meta, cls=_NpEncoder, separators=(',', ':'))
    html = (HTML_TEMPLATE
            .replace('__DATA_DIR_NAME__', data_dir_name)
            .replace('__PARAMS_META_JSON__', params_meta_json)
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
        # Pass 1: scan all pkls for grouping metadata (arrays freed after each file).
        groups = _scan_dir_groups(dir_path)
        if not groups:
            print('No datasets with array data found.')
            return

        # Organise groups by (param_name, adc_cutoff, sobolev_s, fourier_cutoff)
        by_param: dict = {}
        for key, paths in sorted(groups.items()):
            param_name, noise_scale, _seed, adc_cutoff, sobolev_s, fourier_cutoff = (
                key[0], key[1], key[2], key[3], key[4], key[5])
            variant = 'noisy' if noise_scale > 0 else 'clean'
            by_param.setdefault((param_name, adc_cutoff, sobolev_s, fourier_cutoff), {})[variant] = (key, paths)

        param_keys = sorted(by_param.keys())
        print(f'  Grouped into {len(param_keys)} param(s): {param_keys}')

        out_path = Path(args.output) if args.output else _auto_out(dir_path, 'viewer.html')
        title = f'1D Gradient Sweep — {dir_path.name}'

        # Pass 2: process one param at a time — write JS immediately, then free arrays.
        data_dir_name = out_path.stem + '_data'
        data_dir = out_path.parent / data_dir_name
        data_dir.mkdir(parents=True, exist_ok=True)

        all_params_meta = []
        pi_global = 0

        for (param_name, adc_cutoff, sobolev_s, fourier_cutoff) in param_keys:
            variants = by_param[(param_name, adc_cutoff, sobolev_s, fourier_cutoff)]

            clean_data = None
            if 'clean' in variants:
                _, paths = variants['clean']
                raw = _load_and_merge_paths(paths)
                print(f'  Building data for {param_name} cutoff={adc_cutoff:g} s={sobolev_s:g} fc={fourier_cutoff:g} clean …')
                clean_data = build_data(raw)
                del raw

            noisy_data = None
            if 'noisy' in variants:
                _, paths = variants['noisy']
                raw = _load_and_merge_paths(paths)
                print(f'  Building data for {param_name} cutoff={adc_cutoff:g} s={sobolev_s:g} fc={fourier_cutoff:g} noisy …')
                if clean_data:
                    print(f'  Using clean bbox for noisy …')
                noisy_data = build_data(raw, bbox_override=clean_data['bboxes'] if clean_data else None)
                del raw

            # _group_by_param pairs clean+noisy; since no _raw_pkl is set, noisy arrays
            # are already bbox-corrected and used as-is.
            param_entry_list = _group_by_param(
                [d for d in [clean_data, noisy_data] if d is not None])
            del clean_data, noisy_data

            for p in param_entry_list:
                param_meta = _write_param_js(pi_global, p, data_dir, data_dir_name)
                all_params_meta.append(param_meta)
                pi_global += 1
                # p's large arrays are now freed by _write_param_js

        _write_html_shell(all_params_meta, data_dir_name, out_path, title)


if __name__ == '__main__':
    main()
