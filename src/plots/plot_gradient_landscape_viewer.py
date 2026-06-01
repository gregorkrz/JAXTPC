#!/usr/bin/env python
"""
Generate a self-contained interactive HTML viewer for 1d_gradients pkl files.

Loads all pkl files from a directory (produced by 1d_gradients.py with or
without --store-per-plane-loss), groups them by
(param_name, noise_scale, noise_seed, adc_cutoff), then writes a single HTML
file with Plotly.js that lets you:

  • Filter by parameter, noise on/off, noise seed, ADC cutoff
  • Select which tracks to include (losses summed across selected tracks)
  • Select which wire planes to include (summed; only when per-plane loss
    is present, i.e. --store-per-plane-loss was used)
  • Add the current selection as a named curve
  • Overlay any number of curves on one canvas
  • Toggle between Loss / |Gradient| / Signed gradient
  • Toggle Factor (p/GT) or Param value on the x-axis
  • Toggle log / linear y-scale

Works with both single-track-per-pkl files (older sweeps) and
multi-track-per-pkl files (15trk sweep with +). In the single-track case,
runs with the same (param, noise, seed, cutoff) are merged automatically.

Usage
-----
    python src/plots/plot_gradient_landscape_viewer.py

    python src/plots/plot_gradient_landscape_viewer.py \\
        --results-dir results/1d_gradients/sobolev_cutoff_15trk_all_planes \\
        --output plots/sobolev_cutoff_15trk_all_planes/landscape_viewer.html
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import glob
import json
import os
import pickle
from collections import defaultdict

import numpy as np

# ── constants ──────────────────────────────────────────────────────────────────

PARAM_LABELS = {
    'diffusion_trans_cm2_us': 'D⊥ (cm²/μs)',
    'diffusion_long_cm2_us':  'D∥ (cm²/μs)',
    'velocity_cm_us':         'v (cm/μs)',
    'lifetime_us':            'τ (μs)',
    'recomb_alpha':           'α',
    'recomb_beta_90':         'β₉₀',
    'recomb_R':               'R',
}

_RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')
_PLOTS_DIR   = os.environ.get('PLOTS_DIR',   'plots')

DEFAULT_RESULTS = os.path.join(
    _RESULTS_DIR, '1d_gradients', 'sobolev_cutoff_15trk_all_planes'
)


# ── data loading ──────────────────────────────────────────────────────────────

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def _round(vals, ndigits=6):
    return [round(float(v), ndigits) for v in vals]


def load_and_group(results_dir: str) -> dict:
    """Load all pkls, merge by (param, noise, seed, cutoff), return data dict.

    Handles both multi-factor pkls (one file per param/noise combo) and
    per-factor pkls produced by --save-per-factor (one file per sweep point).
    In the per-factor case the factor axis is reconstructed by concatenating
    across pkls, sorted by factor value.
    """
    paths = sorted(glob.glob(os.path.join(results_dir, '*.pkl')))
    if not paths:
        raise FileNotFoundError(f'No .pkl files in {results_dir!r}')

    # key → list of raw pkl dicts (accumulated, merged below)
    raw_groups: dict = defaultdict(list)

    for path in paths:
        fname = os.path.basename(path)
        with open(path, 'rb') as f:
            r = pickle.load(f)

        param_name     = r.get('param_name', 'unknown')
        noise_scale    = float(r.get('noise_scale', 0.0))
        noise_seed     = int(r.get('noise_seed', 42))
        adc_cutoff     = float(r.get('adc_cutoff', 0.0))
        fourier_cutoff = float(r.get('fourier_cutoff', 0.0))
        key            = (param_name, noise_scale, noise_seed, adc_cutoff, fourier_cutoff)
        raw_groups[key].append((fname, r))

        track_specs = r.get('track_specs', [])
        per_pl      = r.get('per_plane_loss_values', None)
        print(f'  {fname}: param={param_name}  noise={noise_scale:.3g}'
              f'  seed={noise_seed}  cutoff={adc_cutoff:.3g}  fcutoff={fourier_cutoff:.3g}'
              f'  tracks={len(track_specs)}  factors={len(r.get("factors", []))}  per_plane={per_pl is not None}')

    print(f'Loaded {len(paths)} file(s) → {len(raw_groups)} group(s)')

    groups: dict = {}
    for key, fname_raws in raw_groups.items():
        param_name, noise_scale, noise_seed, adc_cutoff, fourier_cutoff = key

        # If the group contains per-factor pkls (filename contains '_fac'), drop any
        # summary pkls (no '_fac') — they duplicate factor entries and misalign
        # per_plane_loss indices when both exist in the same directory.
        has_per_factor = any('_fac' in fn for fn, _ in fname_raws)
        if has_per_factor:
            dropped = [fn for fn, _ in fname_raws if '_fac' not in fn]
            if dropped:
                for fn in dropped:
                    print(f'  (skip summary pkl, per-factor pkls present) {fn}')
            fname_raws = [(fn, r) for fn, r in fname_raws if '_fac' in fn]
        raws = [r for _, r in fname_raws]

        # Sort per-factor pkls by their first (and usually only) factor value.
        raws.sort(key=lambda r: r.get('factors', [0.0])[0])

        # Collect all track names in encounter order.
        seen: set = set()
        all_track_names: list = []
        for r in raws:
            for ts in r.get('track_specs', []):
                t = ts['name']
                if t not in seen:
                    seen.add(t)
                    all_track_names.append(t)

        # Concatenate factor axis across pkls; merge track data per factor slice.
        all_factors:       list = []
        all_param_values:  list = []
        tl_parts: dict = defaultdict(list)
        tg_parts: dict = defaultdict(list)
        pl_parts: dict = defaultdict(lambda: defaultdict(list))
        plane_names: list = []
        has_plane = False

        for r in raws:
            all_factors.extend(r.get('factors', []))
            all_param_values.extend(r.get('param_values', []))
            if not plane_names and r.get('plane_names'):
                plane_names = r['plane_names']
            r_tracks   = {ts['name'] for ts in r.get('track_specs', [])}
            per_tl     = r.get('per_track_loss_values', {})
            per_tg     = r.get('per_track_grad_values', {})
            per_pl     = r.get('per_plane_loss_values', None)
            for tname in all_track_names:
                if tname not in r_tracks:
                    continue
                if tname in per_tl:
                    tl_parts[tname].extend(list(per_tl[tname]))
                if tname in per_tg:
                    tg_parts[tname].extend(list(per_tg[tname]))
                if per_pl and tname in per_pl:
                    has_plane = True
                    for pname, vals in per_pl[tname].items():
                        pl_parts[tname][pname].extend(list(vals))

        # Sort all arrays by factor value so plots draw left-to-right.
        # Only safe when every track has data for every factor (the --save-per-factor
        # case); skip if any track is missing from some pkls (different track sets).
        n = len(all_factors)
        _lengths_ok = (
            all(len(v) == n for v in tl_parts.values()) and
            all(len(v) == n for v in tg_parts.values()) and
            all(len(v) == n for td in pl_parts.values() for v in td.values())
        )
        if _lengths_ok and n > 1:
            ord_ = sorted(range(n), key=lambda i: all_factors[i])
            all_factors      = [all_factors[i]      for i in ord_]
            all_param_values = [all_param_values[i] for i in ord_]
            for tname in list(tl_parts.keys()):
                tl_parts[tname] = [tl_parts[tname][i] for i in ord_]
            for tname in list(tg_parts.keys()):
                tg_parts[tname] = [tg_parts[tname][i] for i in ord_]
            for tname in list(pl_parts.keys()):
                for pname in list(pl_parts[tname].keys()):
                    pl_parts[tname][pname] = [pl_parts[tname][pname][i] for i in ord_]

        first = raws[0]
        groups[key] = {
            'param_name':     param_name,
            'param_label':    PARAM_LABELS.get(param_name, param_name),
            'param_gt':       float(first.get('param_gt', 0.0)),
            'factors':        _round(all_factors),
            'param_values':   _round(all_param_values),
            'noise_scale':    noise_scale,
            'noise_seed':     noise_seed,
            'adc_cutoff':     adc_cutoff,
            'fourier_cutoff': fourier_cutoff,
            'loss_name':      first.get('loss_name', ''),
            'track_names':    all_track_names,
            'per_track_loss': {t: _round(v) for t, v in tl_parts.items()},
            'per_track_grad': {t: _round(v) for t, v in tg_parts.items()},
            'per_plane_loss': ({t: {p: _round(v) for p, v in pd.items()}
                                for t, pd in pl_parts.items()} if has_plane else None),
            'plane_names':    plane_names,
        }

    runs = list(groups.values())

    runs.sort(key=lambda r: (r['param_name'], r['noise_scale'], r['noise_seed'], r['adc_cutoff']))

    # Aggregate metadata for the JS side
    all_tracks     = sorted({t for r in runs for t in r['track_names']})
    all_planes     = sorted({p
                             for r in runs if r['per_plane_loss']
                             for tdata in r['per_plane_loss'].values()
                             for p in tdata})
    params         = sorted({r['param_name'] for r in runs})
    noise_options  = sorted({r['noise_scale'] for r in runs})
    seed_options   = sorted({r['noise_seed']  for r in runs if r['noise_scale'] > 0})
    cutoff_options         = sorted({r['adc_cutoff']     for r in runs})
    fourier_cutoff_options = sorted({r['fourier_cutoff'] for r in runs})
    has_per_plane          = bool(all_planes)

    return {
        'runs':                   runs,
        'params':                 params,
        'all_tracks':             all_tracks,
        'all_planes':             all_planes,
        'has_per_plane':          has_per_plane,
        'noise_options':          noise_options,
        'seed_options':           seed_options,
        'cutoff_options':         cutoff_options,
        'fourier_cutoff_options': fourier_cutoff_options,
    }


# ── HTML template ──────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Gradient Landscape Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js" charset="utf-8"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f0f2f5;color:#222;font-size:13px}
#app{display:flex;height:100vh;overflow:hidden}
/* ── left panel ── */
#left{width:310px;min-width:270px;background:#fff;border-right:1px solid #ddd;overflow-y:auto;display:flex;flex-direction:column;flex-shrink:0}
.sec{padding:10px 12px;border-bottom:1px solid #eee}
.sec h3{font-size:11px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px}
select{width:100%;padding:5px 7px;border:1px solid #ccc;border-radius:4px;font-size:13px;background:#fff;margin-bottom:5px}
.tab-bar{display:flex;gap:3px;margin-bottom:6px}
.tab{flex:1;padding:5px 4px;border:1px solid #ccc;border-radius:4px;background:#f5f5f5;cursor:pointer;text-align:center;font-size:12px;font-weight:500;white-space:nowrap}
.tab.active{background:#1565C0;color:#fff;border-color:#0d47a1}
.tab:disabled{opacity:.4;cursor:default}

label.ck{display:flex;align-items:center;gap:6px;padding:2px 0;cursor:pointer;user-select:none}
label.ck:hover{color:#1565C0}
label.ck input{cursor:pointer;accent-color:#1565C0}
button{padding:6px 14px;border:none;border-radius:4px;cursor:pointer;font-size:13px;font-weight:500}
.btn-add{background:#1565C0;color:#fff;width:100%;padding:9px;margin-top:2px;font-size:14px}
.btn-add:hover{background:#0d47a1}
.btn-sm{padding:3px 8px;font-size:11px;border:1px solid #ccc}
.btn-del{background:#e53935;color:#fff;border:none}
.btn-del:hover{background:#c62828}
.btn-clear{background:#9e9e9e;color:#fff;border:none;padding:4px 10px;font-size:12px}
.quick-btn{background:#e8f5e9;color:#2e7d32;border:1px solid #a5d6a7}
.quick-btn:hover{background:#c8e6c9}
.quick-btn.none{background:#fce4ec;color:#c62828;border:1px solid #f48fb1}
.quick-btn.none:hover{background:#f8bbd0}
.quick-row{display:flex;gap:4px;margin-bottom:5px}
.scrollable{max-height:190px;overflow-y:auto;padding-right:2px}
#series-list{flex:1;overflow-y:auto;min-height:60px}
.s-item{display:flex;align-items:center;gap:7px;padding:5px 12px;border-bottom:1px solid #f5f5f5;font-size:12px}
.s-item:hover{background:#f5f5f5}
.swatch{width:13px;height:13px;border-radius:3px;flex-shrink:0}
.s-lbl{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;cursor:default}
.warn{color:#e65100;font-size:11px;margin-top:3px;padding:3px 6px;background:#fff3e0;border-radius:3px;display:none}
#seed-row{display:none}
/* ── right panel ── */
#right{flex:1;display:flex;flex-direction:column;overflow:hidden}
#right-tab-bar{display:flex;gap:0;background:#fff;border-bottom:1px solid #ddd;flex-shrink:0}
.rtab{padding:8px 18px;border:none;border-bottom:3px solid transparent;background:none;cursor:pointer;font-size:13px;font-weight:500;color:#666;border-radius:0}
.rtab:hover{color:#1565C0;background:#f5f8ff}
.rtab.active{color:#1565C0;border-bottom-color:#1565C0;background:#fff}
#plot-bar{display:flex;align-items:center;gap:10px;padding:7px 12px;background:#fafafa;border-bottom:1px solid #ddd;flex-wrap:wrap;flex-shrink:0}
#plot-bar .bar-lbl{font-size:12px;color:#666;white-space:nowrap}
#plot-bar .tab-bar{margin:0}
#plots-row{flex:1;display:flex;min-height:0;gap:0}
.plot-col{flex:1;display:flex;flex-direction:column;min-width:0;border-right:1px solid #e0e0e0}
.plot-col:last-child{border-right:none}
.col-header{padding:5px 12px;font-size:11px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:.5px;background:#fafafa;border-bottom:1px solid #eee;flex-shrink:0}
.plot-div{flex:1;min-height:0}
/* ── min-factor table ── */
#table-view{background:#fff}
.min-tbl{border-collapse:collapse;font-size:12px;white-space:nowrap}
.min-tbl th{padding:7px 12px;background:#f5f5f5;border:1px solid #ddd;font-weight:600;text-align:center;position:sticky;top:0;z-index:1}
.min-tbl th.trk-hdr{text-align:left}
.min-tbl td{padding:6px 12px;border:1px solid #ddd;text-align:center;font-family:monospace;font-size:12px}
.min-tbl td.trk-lbl{text-align:left;font-family:inherit;font-weight:500;background:#fafafa}
/* ── min-factor vs cutoff tab ── */
#minfactor-view{background:#fff;display:flex;flex-direction:column}
/* ── seed ensemble tab ── */
#seeds-view{flex:1;flex-direction:column;min-height:0;overflow:hidden;background:#fff}
.mf-bar{display:flex;align-items:center;gap:10px;padding:7px 12px;background:#fafafa;border-bottom:1px solid #ddd;flex-wrap:wrap;flex-shrink:0}
.mf-section{padding:10px 12px 8px}
.mf-section-hdr{font-size:11px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}
.mf-row{display:flex;gap:8px}
.mf-plot{flex:1;min-width:0;height:400px}
</style>
</head>
<body>
<div id="app">

<!-- ══ Left panel ══════════════════════════════════════════════════════════ -->
<div id="left">

  <div class="sec" style="padding:8px 12px">
    <button id="btn-copy" onclick="copyLink()" style="width:100%;padding:6px;background:#e8f5e9;color:#2e7d32;border:1px solid #a5d6a7;border-radius:4px;cursor:pointer;font-size:12px;font-weight:500">Copy sharable link</button>
  </div>

  <div class="sec">
    <h3>Filters</h3>

    <div style="font-size:11px;color:#888;margin-bottom:2px">Parameter</div>
    <select id="sel-param"></select>

    <div style="font-size:11px;color:#888;margin-bottom:2px">Noise</div>
    <select id="sel-noise-seed"></select>

    <div style="font-size:11px;color:#888;margin-bottom:2px">ADC cutoff</div>
    <select id="sel-cutoff"></select>

    <div style="font-size:11px;color:#888;margin-bottom:2px;margin-top:6px">Fourier cutoff</div>
    <select id="sel-fourier-cutoff"></select>

    <div id="n-points-info" style="font-size:11px;color:#555;margin-top:2px;padding:2px 0"></div>
    <div id="no-run-warn" class="warn">⚠ No data matches this filter combination.</div>
  </div>

  <div class="sec">
    <h3>Tracks</h3>
    <div class="quick-row">
      <button class="btn-sm quick-btn"      onclick="selAll('track')">All</button>
      <button class="btn-sm quick-btn none" onclick="selNone('track')">None</button>
      <button class="btn-sm quick-btn"      onclick="selTrackGroup('FirstQuarter')">1st¼</button>
      <button class="btn-sm quick-btn"      onclick="selTrackGroup('LastQuarter')">Last¼</button>
      <button class="btn-sm quick-btn"      onclick="selTrackGroup('original')">Orig</button>
      <button class="btn-sm quick-btn"      onclick="selTrackGroup('nice')">Nice</button>
    </div>
    <div id="track-checks" class="scrollable"></div>
  </div>

  <div class="sec" id="plane-sec" style="display:none">
    <h3>Wire planes <span style="font-weight:400;color:#aaa">(loss only)</span></h3>
    <div class="quick-row">
      <button class="btn-sm quick-btn"      onclick="selAll('plane')">All</button>
      <button class="btn-sm quick-btn none" onclick="selNone('plane')">None</button>
    </div>
    <div id="plane-checks" class="scrollable"></div>
  </div>

  <div class="sec">
    <button class="btn-add" onclick="addSeries()">＋ Add to plot</button>
  </div>

  <div class="sec" style="display:flex;justify-content:space-between;align-items:center;padding-bottom:6px">
    <h3 style="margin:0">Series on canvas</h3>
    <button class="btn-clear" onclick="clearAll()">Clear all</button>
  </div>
  <div id="series-list">
    <p style="color:#bbb;font-size:12px;padding:8px 12px" id="s-empty">No series yet.</p>
  </div>

  <div class="sec" id="resources-sec">
    <h3>Resources</h3>
    <div style="display:flex;flex-direction:column;gap:5px;font-size:12px">
      <a href="https://d3jk0djzcq11zh.cloudfront.net/landscape_interactive_20260508.html"
         target="_blank" style="color:#1565C0">2D loss landscape viewer</a>
      <a href="viewer.html" target="_blank" style="color:#1565C0">Signal viewer (interactive)</a>
      <a id="res-signals" href="#" target="_blank" style="color:#1565C0">Signal wireplanes PDF</a>
      <a id="res-tracks"  href="#" target="_blank" style="color:#1565C0">Track event display</a>
    </div>
  </div>

</div><!-- /#left -->

<!-- ══ Right panel ══════════════════════════════════════════════════════════ -->
<div id="right">
  <div id="right-tab-bar">
    <button class="rtab active" id="rtab-plots"     onclick="setRightTab('plots')">Plots</button>
    <button class="rtab"        id="rtab-table"     onclick="setRightTab('table')">Min. factor table</button>
    <button class="rtab"        id="rtab-minfactor" onclick="setRightTab('minfactor')">Min-factor vs Cutoff</button>
    <button class="rtab"        id="rtab-seeds"     onclick="setRightTab('seeds')">Seed ensemble</button>
  </div>
  <div id="plot-bar">
    <span class="bar-lbl">Quantity:</span>
    <div class="tab-bar">
      <button class="tab active" id="qty-loss"    onclick="setQty('loss')">Loss</button>
      <button class="tab"        id="qty-absgrad"  onclick="setQty('absgrad')">|Gradient|</button>
      <button class="tab"        id="qty-grad"     onclick="setQty('grad')">Signed grad</button>
    </div>
    <span class="bar-lbl" style="margin-left:6px">X axis:</span>
    <div class="tab-bar">
      <button class="tab active" id="xax-factor" onclick="setXax('factor')">Factor (p/GT)</button>
      <button class="tab"        id="xax-param"  onclick="setXax('param')">Param value</button>
    </div>
    <span class="bar-lbl" style="margin-left:6px">Y scale:</span>
    <div class="tab-bar">
      <button class="tab active" id="ys-log" onclick="setYscale('log')">Log</button>
      <button class="tab"        id="ys-lin" onclick="setYscale('lin')">Linear</button>
    </div>
    <span class="bar-lbl" style="margin-left:6px">Seed bands:</span>
    <div class="tab-bar">
      <button class="tab"        id="band-none"  onclick="setBandMode('none')">Off</button>
      <button class="tab active" id="band-range" onclick="setBandMode('range')">Min–Max</button>
    </div>
  </div>
  <div id="plots-row">
    <div class="plot-col">
      <div class="col-header">Live preview</div>
      <div id="live-plot" class="plot-div"></div>
    </div>
    <div class="plot-col">
      <div class="col-header">Canvas</div>
      <div id="main-plot" class="plot-div"></div>
    </div>
  </div>
  <div id="table-view" style="display:none;flex:1;overflow:auto;padding:16px"></div>

  <!-- Min-factor vs Cutoff tab -->
  <div id="minfactor-view" style="display:none;flex:1;overflow-y:auto">
    <!-- top bar -->
    <div class="mf-bar">
      <span class="bar-lbl">Noise/Seed:</span>
      <select id="mf-ns-sel" style="width:auto;min-width:140px;margin:0"></select>
      <span class="bar-lbl" style="margin-left:8px">Plane group (groups plot):</span>
      <div class="tab-bar" id="mf-plane-tabs" style="margin:0">
        <button class="tab active" data-pg="all" onclick="setMfPlaneGroup('all')">All</button>
        <button class="tab" data-pg="U"   onclick="setMfPlaneGroup('U')">U</button>
        <button class="tab" data-pg="V"   onclick="setMfPlaneGroup('V')">V</button>
        <button class="tab" data-pg="Y"   onclick="setMfPlaneGroup('Y')">Y</button>
      </div>
    </div>
    <!-- sub-tab bar -->
    <div style="display:flex;background:#fff;border-bottom:1px solid #ddd;padding:0 4px;flex-shrink:0">
      <button id="mfst-adc" class="rtab active" onclick="setMfSubTab('adc')">vs ADC cutoff</button>
      <button id="mfst-fc"  class="rtab"        onclick="setMfSubTab('fc')">vs Fourier cutoff</button>
    </div>
    <!-- axis picker bar -->
    <div class="mf-bar">
      <span class="bar-lbl" id="mf-axis-lbl">Fourier cutoff:</span>
      <select id="mf-fc-sel"  style="width:auto;min-width:120px;margin:0"></select>
      <select id="mf-adc-sel" style="width:auto;min-width:140px;margin:0;display:none"></select>
    </div>
    <!-- fixed groups section -->
    <div class="mf-section">
      <div class="mf-section-hdr" id="mf-fixed-hdr">Track groups — min-factor vs ADC cutoff</div>
      <div id="mf-fixed-row"    class="mf-row"></div>
      <div id="mf-fixed-fc-row" class="mf-row" style="display:none"></div>
    </div>
    <hr style="margin:0 12px">
    <!-- custom section -->
    <div style="display:flex;gap:12px;padding:10px 12px">
      <div style="width:210px;flex-shrink:0">
        <div class="mf-section-hdr">Custom selection</div>
        <div style="font-size:11px;color:#888;margin-bottom:3px">Tracks</div>
        <div class="quick-row">
          <button class="btn-sm quick-btn" onclick="mfSelAll()">All</button>
          <button class="btn-sm quick-btn none" onclick="mfSelNone()">None</button>
          <button class="btn-sm quick-btn" onclick="mfSelGroup('original')">Orig</button>
          <button class="btn-sm quick-btn" onclick="mfSelGroup('nice')">Nice</button>
        </div>
        <div id="mf-track-checks" class="scrollable" style="max-height:320px"></div>
      </div>
      <div style="flex:1;min-width:0">
        <div class="mf-section-hdr" id="mf-custom-hdr">Custom tracks — min-factor vs ADC cutoff</div>
        <div id="mf-custom-row"    class="mf-row"></div>
        <div id="mf-custom-fc-row" class="mf-row" style="display:none"></div>
      </div>
    </div>
  </div>

  <div id="seeds-view" style="display:none">
    <div class="col-header" style="flex-shrink:0">Seed ensemble <span style="font-weight:400;color:#aaa;text-transform:none;font-size:11px">— all seeds, individual points + mean ± 1σ band</span><span id="seeds-min-label" style="font-weight:400;color:#333;text-transform:none;font-size:12px;margin-left:14px"></span></div>
    <div id="seeds-plot" style="flex:1;min-height:0"></div>
    <div id="seeds-stats" style="flex-shrink:0;padding:6px 12px;font-size:12px;font-family:monospace;color:#333;border-top:1px solid #e0e0e0;background:#fafafa;white-space:pre-wrap"></div>
  </div>

</div>

</div><!-- /#app -->

<script>
/* ─────────────────────────────── embedded data ─────────────────────────────── */
const DATA = __DATA_JSON__;

/* ─────────────────────────────── constants ─────────────────────────────── */
const PALETTE = [
  '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
  '#8c564b','#e377c2','#17becf','#bcbd22','#7f7f7f',
  '#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5',
  '#e8a0bf','#b5ead7','#ffdac1','#c7ceea','#ff9aa2',
];

/* ─────────────────────────────── state ─────────────────────────────── */
let qty      = 'loss';
let xax      = 'factor';
let yscale   = 'log';
let bandMode = 'range';

let fParam   = DATA.params[0];
let fNoise   = DATA.noise_options[0] || 0.0;
let fSeed    = DATA.seed_options[0]  || 42;
let fCutoff  = DATA.cutoff_options[0] || 0.0;
let fFourier = DATA.fourier_cutoff_options[0] || 0.0;

let series   = [];
let colorIdx = 0;
let _noiseSeedOptions = [];
let liveInited = false;
let canvasInited = false;
let seedsInited = false;
let rightTab = 'plots';

function nextColor() { return PALETTE[colorIdx++ % PALETTE.length]; }

/* ─────────────────────────────── helpers ─────────────────────────────── */
function sid(raw) { return String(raw).replace(/[^a-zA-Z0-9]/g, '_'); }

function getRun(param, noise, seed, cutoff, fourier) {
  return DATA.runs.find(r =>
    r.param_name === param &&
    Math.abs(r.noise_scale - noise) < 1e-9 &&
    (noise < 1e-9 || r.noise_seed === seed) &&
    Math.abs(r.adc_cutoff - cutoff) < 1e-9 &&
    Math.abs(r.fourier_cutoff - fourier) < 1e-9
  ) || null;
}

function getSelectedTracks() {
  return DATA.all_tracks.filter(t => {
    const el = document.getElementById('ck-t-' + sid(t));
    return el && el.checked;
  });
}

function getSelectedPlanes() {
  return DATA.all_planes.filter(p => {
    const el = document.getElementById('ck-p-' + sid(p));
    return el && el.checked;
  });
}

/* Compute loss / grad over selected tracks and planes for one run.
   Loss uses geomean_log1p over selected planes (matching the actual loss):
     expm1( mean_p( log1p(L_p) ) )   summed across selected tracks.
   When no per-plane data is available, falls back to per_track_loss. */
function computeVals(run, tracks, planes) {
  const n    = run.factors.length;
  const loss = new Array(n).fill(0);
  const grad = new Array(n).fill(0);

  for (const t of tracks) {
    const tLoss = run.per_track_loss[t];
    const tGrad = run.per_track_grad[t];
    if (!tLoss) continue;

    if (planes.length > 0 && run.per_plane_loss && run.per_plane_loss[t]) {
      // geomean_log1p: expm1( mean_p( log1p(L_p) ) )
      const plData = planes.map(p => run.per_plane_loss[t][p]).filter(Boolean);
      const k = plData.length;
      if (k > 0) {
        for (let i = 0; i < n; i++) {
          let logSum = 0;
          for (const pl of plData) logSum += Math.log1p(pl[i]);
          loss[i] += Math.expm1(logSum / k);
        }
      }
    } else {
      for (let i = 0; i < n; i++) loss[i] += tLoss[i];
    }
    if (tGrad) for (let i = 0; i < n; i++) grad[i] += tGrad[i];
  }
  return { factors: run.factors, param_values: run.param_values,
           loss, absgrad: grad.map(Math.abs), grad };
}

function buildLabel(run, tracks, planes) {
  const noise  = run.noise_scale > 0
               ? `σ=${run.noise_scale} s${run.noise_seed}` : 'clean';
  const cut    = run.fourier_cutoff > 0
               ? `cut=${run.adc_cutoff} fc=${run.fourier_cutoff}`
               : `cut=${run.adc_cutoff}`;
  const nT     = tracks.length;
  const tStr   = nT === DATA.all_tracks.length ? 'all trk'
               : nT === 1 ? tracks[0] : `${nT} trk`;
  const pStr   = planes.length === 0 ? ''
               : planes.length === DATA.all_planes.length ? ' [all planes]'
               : ` [${planes.join('+')}]`;
  return `${run.param_label} | ${noise} | ${cut} | ${tStr}${pStr}`;
}

/* ─────────────────────────────── filter UI ─────────────────────────────── */
function buildFilters() {
  /* param */
  const sp = document.getElementById('sel-param');
  sp.innerHTML = '';
  DATA.params.forEach(p => {
    const o = document.createElement('option');
    o.value = p;
    o.textContent = DATA.runs.find(r => r.param_name === p)?.param_label || p;
    sp.appendChild(o);
  });
  sp.value = fParam;
  sp.onchange = () => { fParam = sp.value; onFilt(); };

  /* combined noise + seed dropdown */
  const ns = document.getElementById('sel-noise-seed');
  ns.innerHTML = '';
  _noiseSeedOptions = [];
  if (DATA.noise_options.includes(0.0)) {
    _noiseSeedOptions.push({ noise: 0.0, seed: null, label: 'No noise' });
  }
  DATA.noise_options.filter(n => n > 0).forEach(n => {
    DATA.seed_options.forEach(s => {
      _noiseSeedOptions.push({ noise: n, seed: s, label: `Noise seed ${s}` });
    });
  });
  _noiseSeedOptions.forEach((opt, i) => {
    const o = document.createElement('option'); o.value = i; o.textContent = opt.label; ns.appendChild(o);
  });
  const _initIdx = _noiseSeedOptions.findIndex(o =>
    Math.abs(o.noise - fNoise) < 1e-9 && (o.seed === null || o.seed === fSeed));
  ns.value = _initIdx >= 0 ? _initIdx : 0;
  ns.onchange = () => {
    const opt = _noiseSeedOptions[parseInt(ns.value)];
    fNoise = opt.noise;
    if (opt.seed !== null) fSeed = opt.seed;
    onFilt();
  };

  /* cutoff */
  const sc = document.getElementById('sel-cutoff');
  sc.innerHTML = '';
  DATA.cutoff_options.forEach(c => {
    const o = document.createElement('option'); o.value = c;
    o.textContent = c === 0 ? 'cutoff = 0 (none)' : `cutoff = ${c} ADC`;
    sc.appendChild(o);
  });
  sc.value = fCutoff;
  sc.onchange = () => { fCutoff = parseFloat(sc.value); onFilt(); };

  /* fourier cutoff */
  const sfc = document.getElementById('sel-fourier-cutoff');
  sfc.innerHTML = '';
  DATA.fourier_cutoff_options.forEach(c => {
    const o = document.createElement('option'); o.value = c;
    o.textContent = c === 0 ? 'fc = 0 (none)' : `fc = ${c}`;
    sfc.appendChild(o);
  });
  sfc.value = fFourier;
  sfc.onchange = () => { fFourier = parseFloat(sfc.value); onFilt(); };
}

function buildCheckboxes() {
  /* tracks */
  const td = document.getElementById('track-checks');
  td.innerHTML = '';
  DATA.all_tracks.forEach(t => {
    const lbl = document.createElement('label'); lbl.className = 'ck';
    const cb  = document.createElement('input'); cb.type = 'checkbox'; cb.id = 'ck-t-' + sid(t); cb.checked = true;
    cb.onchange = () => { renderLivePlot(); if (rightTab === 'seeds') renderSeedsPlot(); saveState(); };
    lbl.appendChild(cb); lbl.appendChild(document.createTextNode(' ' + t));
    td.appendChild(lbl);
  });

  /* planes */
  if (!DATA.has_per_plane || DATA.all_planes.length === 0) return;
  document.getElementById('plane-sec').style.display = '';
  const pd = document.getElementById('plane-checks');
  pd.innerHTML = '';
  DATA.all_planes.forEach(p => {
    const lbl = document.createElement('label'); lbl.className = 'ck';
    const cb  = document.createElement('input'); cb.type = 'checkbox'; cb.id = 'ck-p-' + sid(p); cb.checked = true;
    cb.onchange = () => { renderLivePlot(); if (rightTab === 'seeds') renderSeedsPlot(); saveState(); };
    lbl.appendChild(cb); lbl.appendChild(document.createTextNode(' ' + p));
    pd.appendChild(lbl);
  });
}

function selAll(kind) {
  const items = kind === 'track' ? DATA.all_tracks : DATA.all_planes;
  const prefix = kind === 'track' ? 'ck-t-' : 'ck-p-';
  items.forEach(x => { const el = document.getElementById(prefix + sid(x)); if (el) el.checked = true; });
  renderLivePlot(); if (rightTab === 'seeds') renderSeedsPlot(); saveState();
}
function selNone(kind) {
  const items = kind === 'track' ? DATA.all_tracks : DATA.all_planes;
  const prefix = kind === 'track' ? 'ck-t-' : 'ck-p-';
  items.forEach(x => { const el = document.getElementById(prefix + sid(x)); if (el) el.checked = false; });
  renderLivePlot(); if (rightTab === 'seeds') renderSeedsPlot(); saveState();
}
const _UGLY_TRACKS = ['Muon4_100MeV','Muon5_100MeV','Muon10_100MeV','Muon12_100MeV','Muon13_100MeV'];
function _isNice(t) { return !_UGLY_TRACKS.some(u => t.includes(u)); }

function selTrackGroup(group) {
  DATA.all_tracks.forEach(t => {
    const el = document.getElementById('ck-t-' + sid(t));
    if (!el) return;
    if (group === 'FirstQuarter')     el.checked = t.includes('FirstQuarter');
    else if (group === 'LastQuarter') el.checked = t.includes('LastQuarter');
    else if (group === 'nice')        el.checked = _isNice(t);
    else /* original */               el.checked = !t.includes('Quarter');
  });
  renderLivePlot(); if (rightTab === 'seeds') renderSeedsPlot(); saveState();
}
function updateResourceLinks() {
  const base = 'https://d3jk0djzcq11zh.cloudfront.net/20260605/event_displays_15_tracks';
  const sig = document.getElementById('res-signals');
  const trk = document.getElementById('res-tracks');
  if (sig) { sig.href = base + '/wireplanes_15x6_gt_signals.pdf'; }
  if (trk) { trk.href = base + '/index.html'; }
}

function onFilt() {
  const run = getRun(fParam, fNoise, fSeed, fCutoff, fFourier);
  document.getElementById('no-run-warn').style.display = run ? 'none' : '';
  const nPts = document.getElementById('n-points-info');
  if (nPts) nPts.textContent = run ? `${run.factors.length} sweep points` : '';
  if      (rightTab === 'plots') renderLivePlot();
  else if (rightTab === 'seeds') renderSeedsPlot();
  else if (rightTab === 'table') renderMinTable();
  saveState();
}

/* ─────────────────────────────── right tab ─────────────────────────────── */
function setRightTab(tab) {
  rightTab = tab;
  const isPlots = tab === 'plots';
  const isMf    = tab === 'minfactor';
  const isSeeds = tab === 'seeds';
  document.getElementById('plots-row').style.display       = isPlots ? '' : 'none';
  document.getElementById('plot-bar').style.display        = (isPlots || isSeeds) ? '' : 'none';
  document.getElementById('table-view').style.display      = tab === 'table' ? '' : 'none';
  document.getElementById('minfactor-view').style.display  = isMf ? '' : 'none';
  document.getElementById('seeds-view').style.display      = isSeeds ? 'flex' : 'none';
  ['plots','table','minfactor','seeds'].forEach(k =>
    document.getElementById('rtab-'+k).classList.toggle('active', k===tab));
  if (isPlots)       { renderLivePlot(); renderCanvas(); }
  else if (tab === 'table') renderMinTable();
  else if (isMf)     { if (!mfInited) initMfTab(); else { _mfRenderFixed(); renderMfCustom(); } }
  else if (isSeeds)  renderSeedsPlot();
  saveState();
}

/* ─────────────────────────────── min-factor table ─────────────────────────────── */
function _argmin(arr) {
  let mi = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] < arr[mi]) mi = i;
  return mi;
}

function renderMinTable() {
  const el  = document.getElementById('table-view');
  const run = getRun(fParam, fNoise, fSeed, fCutoff, fFourier);

  if (!run) {
    el.innerHTML = '<p style="padding:20px;color:#e65100">⚠ No data for current filter selection.</p>';
    return;
  }

  const tracks  = run.track_names;
  const factors = run.factors;
  const hasPlanes = run.per_plane_loss !== null;
  const planeCols = hasPlanes ? run.plane_names : [];

  /* Collect min-factor per (track, col). col = plane name or 'All'. */
  const cells = {};   // cells[track][col] = factor value
  const allDevs = [];

  tracks.forEach(t => {
    cells[t] = {};

    /* Per-plane columns */
    if (hasPlanes && run.per_plane_loss[t]) {
      planeCols.forEach(p => {
        const pl = run.per_plane_loss[t][p];
        if (pl) {
          cells[t][p] = factors[_argmin(pl)];
          allDevs.push(Math.abs(cells[t][p] - 1.0));
        }
      });
    }

    /* "All" column — use per_track_loss (real geomean_log1p over all planes) */
    const tl = run.per_track_loss[t];
    if (tl) {
      cells[t]['All'] = factors[_argmin(tl)];
      allDevs.push(Math.abs(cells[t]['All'] - 1.0));
    }
  });

  const maxDev = allDevs.length ? Math.max(...allDevs, 1e-6) : 1;

  function bgColor(factor) {
    if (factor === undefined) return '#f5f5f5';
    const t = Math.min(Math.abs(factor - 1.0) / maxDev, 1.0);
    /* white → orange → red */
    const r = 255;
    const g = Math.round(255 * (1 - t * 0.85));
    const b = Math.round(255 * (1 - t));
    return `rgb(${r},${g},${b})`;
  }

  const cols = [...planeCols, 'All'];
  const noiseLabel = run.noise_scale > 0 ? `Noise seed ${run.noise_seed}` : 'No noise';
  const subtitle = `${run.param_label}  ·  ${noiseLabel}  ·  cutoff=${run.adc_cutoff}  ·  factor at minimum loss`;

  let html = `<p style="font-size:12px;color:#888;margin-bottom:10px">${subtitle}</p>`;
  html += '<table class="min-tbl"><thead><tr>';
  html += '<th class="trk-hdr">Track</th>';
  cols.forEach(c => { html += `<th>${c}</th>`; });
  html += '</tr></thead><tbody>';

  tracks.forEach(t => {
    html += `<tr><td class="trk-lbl">${t}</td>`;
    cols.forEach(c => {
      const val = cells[t][c];
      const bg  = bgColor(val);
      const txt = val !== undefined ? val.toFixed(3) : '—';
      const fg  = val !== undefined && Math.abs(val - 1.0) / maxDev > 0.6 ? '#fff' : '#222';
      html += `<td style="background:${bg};color:${fg}">${txt}</td>`;
    });
    html += '</tr>';
  });

  html += '</tbody></table>';
  el.innerHTML = html;
}

/* ─────────────────────────────── series ─────────────────────────────── */
function addSeries() {
  const run = getRun(fParam, fNoise, fSeed, fCutoff, fFourier);
  if (!run) { alert('No data matches the current filter combination.'); return; }

  const tracks = getSelectedTracks();
  if (tracks.length === 0) { alert('Select at least one track.'); return; }

  const planes = getSelectedPlanes();
  const vals   = computeVals(run, tracks, planes);
  const label  = buildLabel(run, tracks, planes);
  const color  = nextColor();

  series.push({ id: colorIdx - 1, label, color, vals,
                spec: { param: fParam, noise: fNoise, seed: fSeed, cutoff: fCutoff, fourier: fFourier, tracks, planes } });
  renderSeriesList();
  renderCanvas();
  saveState();
}

function removeSeries(id) {
  series = series.filter(s => s.id !== id);
  renderSeriesList();
  renderCanvas();
  saveState();
}

function clearAll() { series = []; colorIdx = 0; renderSeriesList(); renderCanvas(); saveState(); }

function renderSeriesList() {
  const el    = document.getElementById('series-list');
  const empty = document.getElementById('s-empty');
  empty.style.display = series.length ? 'none' : '';
  el.querySelectorAll('.s-item').forEach(x => x.remove());

  series.forEach(s => {
    const div = document.createElement('div'); div.className = 's-item';
    div.innerHTML =
      `<div class="swatch" style="background:${s.color}"></div>` +
      `<span class="s-lbl" title="${s.label}">${s.label}</span>` +
      `<button class="btn-sm btn-del" onclick="removeSeries(${s.id})">✕</button>`;
    el.appendChild(div);
  });
}

/* ─────────────────────────────── plot helpers ─────────────────────────────── */
function getXY(s) {
  const x = xax === 'factor' ? s.vals.factors : s.vals.param_values;
  const y = qty === 'loss' ? s.vals.loss : qty === 'absgrad' ? s.vals.absgrad : s.vals.grad;
  return { x, y };
}

function _baseLayout(yTitle, xTitle) {
  const shapes = [];
  if (xax === 'factor') {
    shapes.push({ type:'line', x0:1, x1:1, y0:0, y1:1, xref:'x', yref:'paper',
                  line:{ color:'#555', width:1.5, dash:'dot' } });
  }
  if (qty === 'grad') {
    shapes.push({ type:'line', x0:0, x1:1, y0:0, y1:0, xref:'paper', yref:'y',
                  line:{ color:'#aaa', width:1, dash:'dash' } });
  }
  return {
    xaxis: { title: xTitle, gridcolor:'#eee', zeroline:false },
    yaxis: { title: yTitle, type: (yscale==='log' && qty!=='grad') ? 'log' : 'linear',
             gridcolor:'#eee', zeroline: qty==='grad' },
    shapes,
    legend: { font:{ size:11 }, bgcolor:'rgba(255,255,255,.85)', bordercolor:'#ddd', borderwidth:1 },
    margin: { t:16, b:52, l:72, r:20 },
    paper_bgcolor:'#fff', plot_bgcolor:'#fcfcfc',
    hovermode:'x unified',
  };
}

/* ── live preview (current filter + track + plane selection) ── */
function renderLivePlot() {
  const xTitle = xax === 'factor' ? 'Factor  (param / GT)' : 'Parameter value';
  const yTitle = { loss: 'Loss', absgrad: '|∂L/∂p|', grad: '∂L/∂p' }[qty];
  const layout = _baseLayout(yTitle, xTitle);
  const cfg    = { responsive: true };

  const run    = getRun(fParam, fNoise, fSeed, fCutoff, fFourier);
  const tracks = getSelectedTracks();
  let traces   = [];

  if (run && tracks.length > 0) {
    const planes = getSelectedPlanes();
    const vals   = computeVals(run, tracks, planes);
    const { x, y } = { x: xax==='factor' ? vals.factors : vals.param_values,
                        y: qty==='loss' ? vals.loss : qty==='absgrad' ? vals.absgrad : vals.grad };
    traces = [{ x, y, name: buildLabel(run, tracks, planes),
                type:'scatter', mode:'lines+markers',
                line:{ color:'#1565C0', width:2.5 }, marker:{ color:'#1565C0', size:5 } }];
  }

  if (!liveInited) {
    Plotly.newPlot('live-plot', traces, layout, cfg);
    liveInited = true;
  } else {
    Plotly.react('live-plot', traces, layout, cfg);
  }
}

/* ── canvas (accumulated series) ── */
function _bandGroups() {
  if (bandMode === 'none') return null;
  const byKey = {};
  series.forEach(s => {
    const gk = s.label.replace(/\bs\d+\b/, 'SEED');
    if (!byKey[gk]) byKey[gk] = [];
    byKey[gk].push(s);
  });
  return byKey;
}

function renderCanvas() {
  const xTitle = xax === 'factor' ? 'Factor  (param / GT)' : 'Parameter value';
  const yTitle = { loss: 'Loss', absgrad: '|∂L/∂p|', grad: '∂L/∂p' }[qty];
  const layout = _baseLayout(yTitle, xTitle);
  const cfg    = { responsive: true };
  const traces = [];
  const byKey  = _bandGroups();

  series.forEach(s => {
    const { x, y } = getXY(s);
    traces.push({ x, y, name: s.label, type:'scatter', mode:'lines+markers',
                  line:{ color:s.color, width:2 }, marker:{ color:s.color, size:4 } });
  });

  if (byKey) {
    Object.values(byKey).forEach(grp => {
      if (grp.length < 2) return;
      const refXY = getXY(grp[0]);
      const n = refXY.x.length;
      const yMin = new Array(n).fill(Infinity);
      const yMax = new Array(n).fill(-Infinity);
      grp.forEach(s => {
        const { y } = getXY(s);
        for (let i = 0; i < n; i++) {
          if (y[i] < yMin[i]) yMin[i] = y[i];
          if (y[i] > yMax[i]) yMax[i] = y[i];
        }
      });
      const xFwd = refXY.x, xRev = [...refXY.x].reverse();
      traces.push({ x:[...xFwd,...xRev], y:[...yMax,...yMin.slice().reverse()],
                    fill:'toself', fillcolor:grp[0].color+'28',
                    line:{ color:'transparent' }, showlegend:false,
                    hoverinfo:'skip', type:'scatter' });
    });
  }

  if (!canvasInited) {
    Plotly.newPlot('main-plot', traces, layout, cfg);
    canvasInited = true;
  } else {
    Plotly.react('main-plot', traces, layout, cfg);
  }
}

/* ─────────────────────────────── toolbar toggles ─────────────────────────────── */
function _renderBoth() { renderLivePlot(); renderCanvas(); if (rightTab === 'seeds') renderSeedsPlot(); }

function setQty(q) {
  qty = q;
  ['loss','absgrad','grad'].forEach(k =>
    document.getElementById('qty-'+k).classList.toggle('active', k===q));
  const logBtn = document.getElementById('ys-log');
  if (q === 'grad' && yscale === 'log') { yscale='lin'; _syncYscale(); }
  logBtn.disabled = (q === 'grad');
  _renderBoth();
  saveState();
}

function setXax(x) {
  xax = x;
  ['factor','param'].forEach(k =>
    document.getElementById('xax-'+k).classList.toggle('active', k===x));
  _renderBoth();
  saveState();
}

function setYscale(s) {
  yscale = s;
  _syncYscale();
  _renderBoth();
  saveState();
}
function _syncYscale() {
  ['log','lin'].forEach(k =>
    document.getElementById('ys-'+k).classList.toggle('active', k===yscale));
}

function setBandMode(m) {
  bandMode = m;
  ['none','range'].forEach(k =>
    document.getElementById('band-'+k).classList.toggle('active', k===m));
  renderCanvas();
  saveState();
}

/* ─────────────────────────────── URL state persistence ─────────────────────────────── */
function _encState(obj) {
  try { return btoa(unescape(encodeURIComponent(JSON.stringify(obj)))); } catch(e) { return ''; }
}
function _decState(s) {
  try { return JSON.parse(decodeURIComponent(escape(atob(s)))); } catch(e) { return null; }
}

function saveState() {
  const mfct = mfInited ? DATA.all_tracks.filter(t => {
    const e = document.getElementById('mf-ck-' + sid(t)); return e && e.checked;
  }) : null;
  const state = {
    v:1, qty, xax, ys:yscale, bm:bandMode, rt:rightTab,
    fp:fParam, fn:fNoise, fs:fSeed, fc:fCutoff, ff:fFourier,
    trk:getSelectedTracks(),
    pln:getSelectedPlanes(),
    ser:series.map(s=>({
      p:s.spec.param, n:s.spec.noise, sd:s.spec.seed,
      c:s.spec.cutoff, ff:s.spec.fourier, t:s.spec.tracks, pl:s.spec.planes, clr:s.color
    })),
    mfn:mfNoise, mfsd:mfSeed, mfas:mfAllSeeds, mfpg:mfPlaneGroup, mfst:mfSubTab,
    mffc:mfFourierCutoff, mfac:mfAdcCutoff,
    mfct
  };
  const enc = _encState(state);
  if (enc) window.location.hash = enc;
}

function loadState() {
  const raw = window.location.hash.slice(1);
  if (!raw) return false;
  const st = _decState(raw);
  if (!st || st.v !== 1) return false;

  if (st.qty) qty = st.qty;
  if (st.xax) xax = st.xax;
  if (st.ys)  yscale = st.ys;
  if (st.bm)  bandMode = st.bm;
  if (st.rt)  rightTab = st.rt;
  if (st.fp)  fParam = st.fp;
  if (st.fn !== undefined) fNoise  = st.fn;
  if (st.fs !== undefined) fSeed   = st.fs;
  if (st.fc !== undefined) fCutoff  = st.fc;
  if (st.ff !== undefined) fFourier = st.ff;

  // Sync filter selects
  const sp = document.getElementById('sel-param');
  if (sp) sp.value = fParam;
  const ns = document.getElementById('sel-noise-seed');
  if (ns) {
    const idx = _noiseSeedOptions.findIndex(o =>
      Math.abs(o.noise - fNoise) < 1e-9 && (o.seed === null || o.seed === fSeed));
    if (idx >= 0) ns.value = idx;
  }
  const sc = document.getElementById('sel-cutoff');
  if (sc) sc.value = fCutoff;
  const sfc = document.getElementById('sel-fourier-cutoff');
  if (sfc) sfc.value = fFourier;

  // Sync track checkboxes
  if (st.trk) {
    DATA.all_tracks.forEach(t => {
      const el = document.getElementById('ck-t-' + sid(t));
      if (el) el.checked = st.trk.includes(t);
    });
  }
  // Sync plane checkboxes
  if (st.pln) {
    DATA.all_planes.forEach(p => {
      const el = document.getElementById('ck-p-' + sid(p));
      if (el) el.checked = st.pln.includes(p);
    });
  }

  // Sync toolbar button states
  ['loss','absgrad','grad'].forEach(k => document.getElementById('qty-'+k)?.classList.toggle('active', k===qty));
  ['factor','param'].forEach(k => document.getElementById('xax-'+k)?.classList.toggle('active', k===xax));
  ['log','lin'].forEach(k => document.getElementById('ys-'+k)?.classList.toggle('active', k===yscale));
  ['none','range'].forEach(k => document.getElementById('band-'+k)?.classList.toggle('active', k===bandMode));
  ['plots','table','minfactor','seeds'].forEach(k => document.getElementById('rtab-'+k)?.classList.toggle('active', k===rightTab));
  const isPlots = rightTab === 'plots';
  const isSeeds = rightTab === 'seeds';
  document.getElementById('plots-row').style.display       = isPlots ? '' : 'none';
  document.getElementById('plot-bar').style.display        = (isPlots || isSeeds) ? '' : 'none';
  document.getElementById('table-view').style.display      = rightTab === 'table' ? '' : 'none';
  document.getElementById('minfactor-view').style.display  = rightTab === 'minfactor' ? '' : 'none';
  document.getElementById('seeds-view').style.display      = isSeeds ? 'flex' : 'none';
  const logBtn = document.getElementById('ys-log');
  if (logBtn) logBtn.disabled = (qty === 'grad');

  // Restore series
  if (st.ser && st.ser.length > 0) {
    series = []; colorIdx = 0;
    st.ser.forEach(ss => {
      const run = getRun(ss.p, ss.n, ss.sd, ss.c, ss.ff || 0.0);
      if (!run) return;
      const tracks = ss.t || [];
      const planes = ss.pl || [];
      const vals   = computeVals(run, tracks, planes);
      const label  = buildLabel(run, tracks, planes);
      series.push({ id: colorIdx, label, color: ss.clr, vals,
                    spec: { param: ss.p, noise: ss.n, seed: ss.sd,
                            cutoff: ss.c, fourier: ss.ff || 0.0, tracks, planes } });
      colorIdx++;
    });
  }

  // Restore minfactor tab state (applied lazily when the tab is initialised)
  if (st.mfn  !== undefined) mfNoise        = st.mfn;
  if (st.mfsd !== undefined) mfSeed         = st.mfsd;
  if (st.mfas !== undefined) mfAllSeeds     = st.mfas;
  if (st.mfpg)               mfPlaneGroup   = st.mfpg;
  if (st.mfst)               mfSubTab       = st.mfst;
  if (st.mffc !== undefined) mfFourierCutoff = st.mffc;
  if (st.mfac !== undefined) mfAdcCutoff    = st.mfac;
  if (st.mfct)               _savedMfCustomTracks = st.mfct;

  return true;
}

function copyLink() {
  const url = window.location.href;
  if (navigator.clipboard) {
    navigator.clipboard.writeText(url).then(() => {
      const btn = document.getElementById('btn-copy');
      const orig = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = orig; }, 1500);
    }).catch(() => prompt('Copy this link:', url));
  } else {
    prompt('Copy this link:', url);
  }
}

/* ─────────────────────────────── Seed ensemble tab ─────────────────────────────── */
function renderSeedsPlot() {
  const el = document.getElementById('seeds-plot');
  if (!el) return;

  const paramLabel = DATA.runs.find(r => r.param_name === fParam)?.param_label || fParam;
  const xTitle = xax === 'factor' ? `Factor  (${paramLabel} / GT)` : paramLabel;
  const yTitle = { loss: 'Loss', absgrad: '|∂L/∂p|', grad: '∂L/∂p' }[qty];
  const layout = _baseLayout(yTitle, xTitle);
  const cfg    = { responsive: true };

  // All runs for current (param, noise_scale, cutoff, fourier_cutoff) regardless of seed
  const matchRuns = DATA.runs.filter(r =>
    r.param_name === fParam &&
    Math.abs(r.noise_scale    - fNoise)   < 1e-9 &&
    Math.abs(r.adc_cutoff     - fCutoff)  < 1e-9 &&
    Math.abs(r.fourier_cutoff - fFourier) < 1e-9
  ).sort((a, b) => a.noise_seed - b.noise_seed);
  const tracks = getSelectedTracks();
  const planes = getSelectedPlanes();
  const traces = [];

  const _flush = () => {
    if (!seedsInited) { Plotly.newPlot('seeds-plot', traces, layout, cfg); seedsInited = true; }
    else               { Plotly.react('seeds-plot', traces, layout, cfg); }
  };

  if (matchRuns.length === 0 || tracks.length === 0) {
    const lbl = document.getElementById('seeds-min-label');
    if (lbl) lbl.textContent = paramLabel;
    _flush(); return;
  }

  const allVals = matchRuns.map(r => ({
    seed: r.noise_seed, v: computeVals(r, tracks, planes)
  }));

  const refX   = allVals[0].v[xax === 'factor' ? 'factors' : 'param_values'];
  const n      = refX.length;
  const getY   = v => qty === 'loss' ? v.loss : qty === 'absgrad' ? v.absgrad : v.grad;
  const yArrs  = allVals.map(d => getY(d.v));

  const _argmin = arr => arr.indexOf(Math.min(...arr));
  const _fmtFactor = (x, arr) => {
    const f = refX[_argmin(arr)];
    return typeof f === 'number' ? f.toFixed(4) : '—';
  };
  const _showStats = (lines) => {
    const el = document.getElementById('seeds-stats');
    if (el) el.textContent = lines.join('\n');
  };

  if (allVals.length < 2) {
    // Single seed or clean — just a line
    const label = fNoise === 0 ? 'clean' : `seed ${allVals[0].seed}`;
    traces.push({ x: refX, y: yArrs[0], mode: 'lines+markers', type: 'scatter',
                  name: label, line: { color: '#1565C0', width: 2.5 },
                  marker: { color: '#1565C0', size: 5 } });
    _flush();
    _showStats([`${paramLabel} / GT at min loss:  ${_fmtFactor(null, yArrs[0])}`]);
    return;
  }

  // Mean and std — skip missing/NaN values from incomplete runs
  const yMean = new Array(n).fill(null);
  const yStd  = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const vals = yArrs.map(y => (i < y.length && y[i] != null && isFinite(y[i])) ? y[i] : null)
                      .filter(v => v !== null);
    if (!vals.length) continue;
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    yMean[i] = mean;
    yStd[i]  = vals.length > 1
      ? Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length) : 0;
  }
  const yHi = yMean.map((m, i) => m !== null ? m + yStd[i] : null);
  const yLo = yMean.map((m, i) => m !== null ? m - yStd[i] : null);

  // ± 1σ band: closed polygon built only from valid (non-null) points
  const validIdx = [];
  for (let i = 0; i < n; i++) if (yMean[i] !== null) validIdx.push(i);
  if (validIdx.length > 1) {
    const bx  = validIdx.map(i => refX[i]);
    const bHi = validIdx.map(i => yHi[i]);
    const bLo = validIdx.map(i => yLo[i]);
    traces.push({
      x: [...bx, ...[...bx].reverse()],
      y: [...bHi, ...[...bLo].reverse()],
      fill: 'toself', fillcolor: 'rgba(21,101,192,0.12)',
      line: { color: 'transparent' },
      name: 'mean ± 1σ', showlegend: true,
      hoverinfo: 'skip', type: 'scatter',
    });
  }

  // Individual seed dots (one trace per seed, distinct colors)
  allVals.forEach((d, i) => {
    const hex = PALETTE[i % PALETTE.length];
    const [r, g, b] = hex.match(/[\da-f]{2}/gi).slice(0, 3).map(h => parseInt(h, 16));
    const yi = yArrs[i].map(v => (v != null && isFinite(v)) ? v : null);
    traces.push({
      x: refX, y: yi,
      mode: 'markers', type: 'scatter',
      name: `seed ${d.seed}`,
      marker: { size: 6, color: `rgba(${r},${g},${b},0.55)` },
      showlegend: true,
    });
  });

  // Mean line (on top, no gap bridging)
  traces.push({
    x: refX, y: yMean,
    mode: 'lines', type: 'scatter',
    name: 'mean',
    line: { color: '#1565C0', width: 2.5 },
    showlegend: true,
    connectgaps: false,
  });

  _flush();

  // Stats panel: param/GT at min loss for mean and each seed
  const statsLines = [];
  const yMeanFinite = yMean.filter(v => v !== null);
  statsLines.push(`${paramLabel} / GT at min loss:`);
  const minLbl = document.getElementById('seeds-min-label');
  if (yMeanFinite.length) {
    const mi = _argmin(yMean.map(v => v ?? Infinity));
    const minFactor = refX[mi];
    const minStd    = yStd[mi];
    statsLines.push(`  mean  →  ${minFactor.toFixed(4)}  (σ = ${minStd.toFixed(4)})`);
    if (minLbl) minLbl.textContent = `${paramLabel}  |  min at ${minFactor.toFixed(4)} ± ${minStd.toFixed(4)}`;
  } else {
    if (minLbl) minLbl.textContent = '';
  }
  allVals.forEach((d, i) => {
    const yi = yArrs[i];
    if (yi.some(v => v != null && isFinite(v)))
      statsLines.push(`  seed ${d.seed}  →  ${refX[_argmin(yi.map(v => v ?? Infinity))].toFixed(4)}`);
  });
  _showStats(statsLines);
}

/* ─────────────────────────────── Min-factor vs Cutoff tab ─────────────────────────────── */
let mfNoise = 1.0, mfSeed = 42, mfPlaneGroup = 'all', mfAllSeeds = true, mfInited = false;
let mfSubTab = 'adc';
let mfFourierCutoff = DATA.fourier_cutoff_options[0] || 0.0;
let mfAdcCutoff     = DATA.cutoff_options[0]         || 0.0;
let _savedMfCustomPlane = null, _savedMfCustomTracks = null;
const _mfDivs = new Set();

// Six predefined track groups
const MF_GROUPS = [
  { label:'FirstQuarter', color:'#1f77b4', fn: t =>  t.includes('_FirstQuarter') },
  { label:'LastQuarter',  color:'#ff7f0e', fn: t =>  t.includes('_LastQuarter')  },
  { label:'All 15',       color:'#2ca02c', fn: t => !t.includes('Quarter') },
  { label:'1000 MeV',     color:'#d62728', fn: t => !t.includes('Quarter') && (t.includes('_1000MeV') || t === 'diagonal') },
  { label:'500 MeV',      color:'#9467bd', fn: t => !t.includes('Quarter') && t.includes('_500MeV') },
  { label:'100 MeV',      color:'#8c564b', fn: t => !t.includes('Quarter') && t.includes('_100MeV') },
];

// null → use per_track_loss (all planes); otherwise filter plane names by prefix
function mfGetPlanes(group) {
  if (!group || group === 'all') return null;
  return DATA.all_planes.filter(p => p.toUpperCase().startsWith(group.toUpperCase()));
}

// Sum losses across tracks (geomean_log1p over planes, matching computeVals), return factor at argmin
function mfMinFactor(run, tracks, planes) {
  if (!tracks.length) return null;
  const n = run.factors.length;
  const loss = new Float64Array(n);
  let ok = false;
  for (const t of tracks) {
    if (planes === null) {
      const tl = run.per_track_loss[t];
      if (!tl) continue;
      for (let i = 0; i < n; i++) loss[i] += tl[i];
      ok = true;
    } else {
      const ppl = run.per_plane_loss;
      if (!ppl || !ppl[t]) continue;
      const pds = planes.map(p => ppl[t][p]).filter(Boolean);
      if (!pds.length) continue;
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (const pd of pds) s += Math.log1p(pd[i]);
        loss[i] += Math.expm1(s / pds.length);
      }
      ok = true;
    }
  }
  if (!ok) return null;
  let mi = 0;
  for (let i = 1; i < n; i++) if (loss[i] < loss[mi]) mi = i;
  return run.factors[mi];
}

function mfRunsByCutoff(param, noise, seed) {
  return DATA.runs.filter(r =>
    r.param_name === param &&
    Math.abs(r.noise_scale    - noise)           < 1e-9 &&
    Math.abs(r.fourier_cutoff - mfFourierCutoff) < 1e-9 &&
    (noise < 1e-9 || r.noise_seed === seed)
  ).sort((a,b) => a.adc_cutoff - b.adc_cutoff);
}
function mfAllSeedsByCutoff(param, noise) {
  const noisy = DATA.runs.filter(r =>
    r.param_name === param &&
    Math.abs(r.noise_scale    - noise)           < 1e-9 &&
    Math.abs(r.fourier_cutoff - mfFourierCutoff) < 1e-9 &&
    noise > 1e-9
  );
  const map = {};
  noisy.forEach(r => { (map[r.adc_cutoff] = map[r.adc_cutoff] || []).push(r); });
  return Object.keys(map).map(Number).sort((a,b)=>a-b).map(c=>({cutoff:c, runs:map[c]}));
}
function mfRunsByFCutoff(param, noise, seed) {
  return DATA.runs.filter(r =>
    r.param_name === param &&
    Math.abs(r.noise_scale - noise)       < 1e-9 &&
    Math.abs(r.adc_cutoff  - mfAdcCutoff) < 1e-9 &&
    (noise < 1e-9 || r.noise_seed === seed)
  ).sort((a,b) => a.fourier_cutoff - b.fourier_cutoff);
}
function mfAllSeedsByFCutoff(param, noise) {
  const noisy = DATA.runs.filter(r =>
    r.param_name === param &&
    Math.abs(r.noise_scale - noise)       < 1e-9 &&
    Math.abs(r.adc_cutoff  - mfAdcCutoff) < 1e-9 &&
    noise > 1e-9
  );
  const map = {};
  noisy.forEach(r => { (map[r.fourier_cutoff] = map[r.fourier_cutoff] || []).push(r); });
  return Object.keys(map).map(Number).sort((a,b)=>a-b).map(c=>({cutoff:c, runs:map[c]}));
}

function _mfDo(divId, traces, layout) {
  const cfg = { responsive:true };
  if (_mfDivs.has(divId)) Plotly.react(divId, traces, layout, cfg);
  else { Plotly.newPlot(divId, traces, layout, cfg); _mfDivs.add(divId); }
}

function _mfLayout(title) {
  return {
    title: { text:title, font:{size:13} },
    xaxis: { title:'ADC cutoff', gridcolor:'#eee', zeroline:false },
    yaxis: { title:'Factor at min loss (param/GT)', gridcolor:'#eee', zeroline:false },
    shapes: [{ type:'line', x0:0, x1:1, y0:1, y1:1, xref:'paper', yref:'y',
               line:{ color:'#555', width:1.5, dash:'dot' } }],
    legend: { font:{size:11} },
    margin: { t:40, b:52, l:76, r:16 },
    paper_bgcolor:'#fff', plot_bgcolor:'#fcfcfc',
    hovermode:'x unified',
  };
}
function _mfFCLayout(title) {
  return {
    title: { text:title, font:{size:13} },
    xaxis: { title:'Fourier cutoff', gridcolor:'#eee', zeroline:false },
    yaxis: { title:'Factor at min loss (param/GT)', gridcolor:'#eee', zeroline:false },
    shapes: [{ type:'line', x0:0, x1:1, y0:1, y1:1, xref:'paper', yref:'y',
               line:{ color:'#555', width:1.5, dash:'dot' } }],
    legend: { font:{size:11} },
    margin: { t:40, b:52, l:76, r:16 },
    paper_bgcolor:'#fff', plot_bgcolor:'#fcfcfc',
    hovermode:'x unified',
  };
}

// Create one div per parameter inside rowId if not already done
function _mfEnsureRow(rowId, prefix) {
  const row = document.getElementById(rowId);
  if (!row || row.children.length) return;
  DATA.params.forEach(p => {
    const d = document.createElement('div');
    d.id = prefix + sid(p);
    d.className = 'mf-plot';
    row.appendChild(d);
  });
}

function _mfMeanErrTrace(groups, tracks, planes, label, color) {
  const xs = [], ys = [], errs = [];
  groups.forEach(({cutoff, runs}) => {
    const vals = runs.map(r => mfMinFactor(r, tracks, planes)).filter(v => v !== null);
    if (!vals.length) return;
    const mean = vals.reduce((a,b)=>a+b,0) / vals.length;
    const std  = vals.length > 1
      ? Math.sqrt(vals.reduce((a,b)=>a+(b-mean)**2,0) / vals.length) : 0;
    xs.push(cutoff); ys.push(mean); errs.push(std);
  });
  return { x:xs, y:ys, name:label, type:'scatter', mode:'lines+markers', connectgaps:true,
    line:{color, width:2}, marker:{color, size:6},
    error_y:{ type:'data', array:errs, visible:true, color, thickness:1.5, width:4 } };
}

function _mfRenderFixed() {
  _mfEnsureRow('mf-fixed-row', 'mf-fp-');
  const planes = mfGetPlanes(mfPlaneGroup);
  DATA.params.forEach(p => {
    let traces;
    if (mfAllSeeds) {
      const groups = mfAllSeedsByCutoff(p, mfNoise);
      traces = MF_GROUPS.map(g =>
        _mfMeanErrTrace(groups, DATA.all_tracks.filter(g.fn), planes, g.label, g.color));
    } else {
      const runs = mfRunsByCutoff(p, mfNoise, mfSeed);
      const xs   = runs.map(r => r.adc_cutoff);
      traces = MF_GROUPS.map(g => ({
        x: xs,
        y: runs.map(r => mfMinFactor(r, DATA.all_tracks.filter(g.fn), planes)),
        name: g.label, type:'scatter', mode:'lines+markers', connectgaps:true,
        line:{ color:g.color, width:2 }, marker:{ color:g.color, size:6 },
      }));
    }
    const lbl = DATA.runs.find(r => r.param_name === p)?.param_label || p;
    _mfDo('mf-fp-' + sid(p), traces, _mfLayout(lbl));
  });
}

function renderMfCustom() {
  _mfEnsureRow('mf-custom-row', 'mf-cp-');
  const tracks = DATA.all_tracks.filter(t => {
    const e = document.getElementById('mf-ck-' + sid(t)); return e && e.checked;
  });
  const planes = mfGetPlanes(mfPlaneGroup);
  const color  = '#1565C0';
  DATA.params.forEach(p => {
    let traces;
    if (mfAllSeeds) {
      const groups = mfAllSeedsByCutoff(p, mfNoise);
      traces = tracks.length
        ? [_mfMeanErrTrace(groups, tracks, planes, `${tracks.length} tracks [${mfPlaneGroup}]`, color)]
        : [];
    } else {
      const runs = mfRunsByCutoff(p, mfNoise, mfSeed);
      const xs   = runs.map(r => r.adc_cutoff);
      const ys   = runs.map(r => mfMinFactor(r, tracks, planes));
      traces = tracks.length ? [{ x:xs, y:ys,
        name:`${tracks.length} tracks [${mfPlaneGroup}]`, type:'scatter', mode:'lines+markers', connectgaps:true,
        line:{color, width:2.5}, marker:{color, size:7} }] : [];
    }
    const lbl = DATA.runs.find(r => r.param_name === p)?.param_label || p;
    _mfDo('mf-cp-' + sid(p), traces, _mfLayout(lbl));
  });
}

function _mfRenderFixedFourier() {
  _mfEnsureRow('mf-fixed-fc-row', 'mf-ff-');
  const planes = mfGetPlanes(mfPlaneGroup);
  DATA.params.forEach(p => {
    let traces;
    if (mfAllSeeds) {
      const groups = mfAllSeedsByFCutoff(p, mfNoise);
      traces = MF_GROUPS.map(g =>
        _mfMeanErrTrace(groups, DATA.all_tracks.filter(g.fn), planes, g.label, g.color));
    } else {
      const runs = mfRunsByFCutoff(p, mfNoise, mfSeed);
      const xs   = runs.map(r => r.fourier_cutoff);
      traces = MF_GROUPS.map(g => ({
        x: xs,
        y: runs.map(r => mfMinFactor(r, DATA.all_tracks.filter(g.fn), planes)),
        name: g.label, type:'scatter', mode:'lines+markers', connectgaps:true,
        line:{ color:g.color, width:2 }, marker:{ color:g.color, size:6 },
      }));
    }
    const lbl = DATA.runs.find(r => r.param_name === p)?.param_label || p;
    _mfDo('mf-ff-' + sid(p), traces, _mfFCLayout(lbl));
  });
}

function renderMfCustomFourier() {
  _mfEnsureRow('mf-custom-fc-row', 'mf-cf-');
  const tracks = DATA.all_tracks.filter(t => {
    const e = document.getElementById('mf-ck-' + sid(t)); return e && e.checked;
  });
  const planes = mfGetPlanes(mfPlaneGroup);
  const color  = '#1565C0';
  DATA.params.forEach(p => {
    let traces;
    if (mfAllSeeds) {
      const groups = mfAllSeedsByFCutoff(p, mfNoise);
      traces = tracks.length
        ? [_mfMeanErrTrace(groups, tracks, planes, `${tracks.length} tracks [${mfPlaneGroup}]`, color)]
        : [];
    } else {
      const runs = mfRunsByFCutoff(p, mfNoise, mfSeed);
      const xs   = runs.map(r => r.fourier_cutoff);
      const ys   = runs.map(r => mfMinFactor(r, tracks, planes));
      traces = tracks.length ? [{ x:xs, y:ys,
        name:`${tracks.length} tracks [${mfPlaneGroup}]`, type:'scatter', mode:'lines+markers', connectgaps:true,
        line:{color, width:2.5}, marker:{color, size:7} }] : [];
    }
    const lbl = DATA.runs.find(r => r.param_name === p)?.param_label || p;
    _mfDo('mf-cf-' + sid(p), traces, _mfFCLayout(lbl));
  });
}

function setMfSubTab(tab) {
  mfSubTab = tab;
  const isAdc = tab === 'adc';
  document.getElementById('mfst-adc').classList.toggle('active', isAdc);
  document.getElementById('mfst-fc').classList.toggle('active', !isAdc);
  document.getElementById('mf-fixed-row').style.display    = isAdc ? '' : 'none';
  document.getElementById('mf-fixed-fc-row').style.display = isAdc ? 'none' : '';
  document.getElementById('mf-custom-row').style.display    = isAdc ? '' : 'none';
  document.getElementById('mf-custom-fc-row').style.display = isAdc ? 'none' : '';
  document.getElementById('mf-fc-sel').style.display  = isAdc ? '' : 'none';
  document.getElementById('mf-adc-sel').style.display = isAdc ? 'none' : '';
  document.getElementById('mf-axis-lbl').textContent  = isAdc ? 'Fourier cutoff:' : 'ADC cutoff:';
  document.getElementById('mf-fixed-hdr').textContent  = isAdc
    ? 'Track groups — min-factor vs ADC cutoff'
    : 'Track groups — min-factor vs Fourier cutoff';
  document.getElementById('mf-custom-hdr').textContent = isAdc
    ? 'Custom tracks — min-factor vs ADC cutoff'
    : 'Custom tracks — min-factor vs Fourier cutoff';
  if (isAdc) { _mfRenderFixed(); renderMfCustom(); }
  else        { _mfRenderFixedFourier(); renderMfCustomFourier(); }
  saveState();
}

function setMfPlaneGroup(g) {
  mfPlaneGroup = g;
  document.querySelectorAll('#mf-plane-tabs .tab').forEach(b =>
    b.classList.toggle('active', b.dataset.pg === g));
  if (mfSubTab === 'adc') { _mfRenderFixed(); renderMfCustom(); }
  else { _mfRenderFixedFourier(); renderMfCustomFourier(); }
}

function buildMfNoiseSel() {
  const sel = document.getElementById('mf-ns-sel');
  if (!sel) return;
  sel.innerHTML = '';
  // Build mf-local options: shared options + "All seeds" if >1 seeds exist
  const mfOpts = [..._noiseSeedOptions];
  const noisyBase = mfOpts.find(o => o.noise > 0);
  if (noisyBase && DATA.seed_options.length > 1)
    mfOpts.push({ noise: noisyBase.noise, seed: '__all__', label: 'All seeds (mean ± σ)' });
  mfOpts.forEach((opt,i) => {
    const o = document.createElement('option'); o.value = i; o.textContent = opt.label;
    sel.appendChild(o);
  });
  let idx;
  if (mfAllSeeds) {
    idx = mfOpts.findIndex(o => o.seed === '__all__');
  } else {
    idx = mfOpts.findIndex(o =>
      Math.abs(o.noise - mfNoise) < 1e-9 && o.seed !== '__all__' &&
      (o.seed === null || Math.abs(o.seed - mfSeed) < 1e-9));
    if (idx < 0) idx = mfOpts.findIndex(o => o.noise > 0 && o.seed !== '__all__');
  }
  if (idx < 0) idx = 0;
  sel.value = idx;
  const initOpt = mfOpts[idx];
  if (initOpt) {
    mfNoise = initOpt.noise; mfAllSeeds = initOpt.seed === '__all__';
    if (!mfAllSeeds && initOpt.seed !== null) mfSeed = initOpt.seed;
  }
  sel.onchange = () => {
    const opt = mfOpts[+sel.value];
    mfNoise = opt.noise; mfAllSeeds = opt.seed === '__all__';
    if (!mfAllSeeds && opt.seed !== null) mfSeed = opt.seed;
    _mfRenderFixed(); renderMfCustom(); _mfRenderFixedFourier(); renderMfCustomFourier();
  };
}

function buildMfCheckboxes() {
  const el = document.getElementById('mf-track-checks');
  if (!el) return;
  el.innerHTML = '';
  DATA.all_tracks.forEach(t => {
    const lbl = document.createElement('label'); lbl.className = 'ck';
    const cb  = document.createElement('input'); cb.type = 'checkbox';
    cb.id = 'mf-ck-' + sid(t); cb.checked = true;
    cb.onchange = () => { renderMfCustom(); renderMfCustomFourier(); };
    lbl.appendChild(cb); lbl.appendChild(document.createTextNode(' ' + t));
    el.appendChild(lbl);
  });
  if (_savedMfCustomTracks !== null) {
    DATA.all_tracks.forEach(t => {
      const e = document.getElementById('mf-ck-' + sid(t));
      if (e) e.checked = _savedMfCustomTracks.includes(t);
    });
    _savedMfCustomTracks = null;
  }
}

function mfSelAll()  { DATA.all_tracks.forEach(t=>{const e=document.getElementById('mf-ck-'+sid(t));if(e)e.checked=true; }); renderMfCustom(); renderMfCustomFourier(); }
function mfSelNone() { DATA.all_tracks.forEach(t=>{const e=document.getElementById('mf-ck-'+sid(t));if(e)e.checked=false;}); renderMfCustom(); renderMfCustomFourier(); }
function mfSelGroup(group) {
  DATA.all_tracks.forEach(t => {
    const e = document.getElementById('mf-ck-' + sid(t));
    if (!e) return;
    if (group === 'FirstQuarter')     e.checked = t.includes('FirstQuarter');
    else if (group === 'LastQuarter') e.checked = t.includes('LastQuarter');
    else if (group === 'nice')        e.checked = _isNice(t);
    else                              e.checked = !t.includes('Quarter');
  });
  renderMfCustom(); renderMfCustomFourier();
}

function buildMfFCutoffSel() {
  const sel = document.getElementById('mf-fc-sel');
  if (!sel) return;
  sel.innerHTML = '';
  DATA.fourier_cutoff_options.forEach(fc => {
    const o = document.createElement('option'); o.value = fc;
    o.textContent = fc === 0 ? 'fc = 0 (none)' : `fc = ${fc}`;
    sel.appendChild(o);
  });
  sel.value = mfFourierCutoff;
  sel.onchange = () => { mfFourierCutoff = parseFloat(sel.value); _mfRenderFixed(); renderMfCustom(); saveState(); };
}

function buildMfAdcCutoffSel() {
  const sel = document.getElementById('mf-adc-sel');
  if (!sel) return;
  sel.innerHTML = '';
  DATA.cutoff_options.forEach(c => {
    const o = document.createElement('option'); o.value = c;
    o.textContent = c === 0 ? 'cutoff = 0 (none)' : `cutoff = ${c} ADC`;
    sel.appendChild(o);
  });
  sel.value = mfAdcCutoff;
  sel.onchange = () => { mfAdcCutoff = parseFloat(sel.value); _mfRenderFixedFourier(); renderMfCustomFourier(); saveState(); };
}

function initMfTab() {
  buildMfNoiseSel();
  buildMfFCutoffSel();
  buildMfAdcCutoffSel();
  buildMfCheckboxes();
  // Sync plane group button active states from restored state
  document.querySelectorAll('#mf-plane-tabs .tab').forEach(b =>
    b.classList.toggle('active', b.dataset.pg === mfPlaneGroup));
  // Sync sub-tab state
  setMfSubTab(mfSubTab);
  mfInited = true;
}

/* ─────────────────────────────── init ─────────────────────────────── */
window.addEventListener('load', () => {
  buildFilters();
  buildCheckboxes();
  loadState();   // applies URL hash state if present, no-op otherwise
  onFilt();      // render live plot + update warning
  renderSeriesList();
  renderCanvas();
  updateResourceLinks();
  if (rightTab === 'minfactor' && !mfInited) initMfTab();
  else if (rightTab === 'seeds') renderSeedsPlot();
});
</script>
</body>
</html>
"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', default=DEFAULT_RESULTS,
                   help='Directory containing *.pkl files (default: %(default)s)')
    p.add_argument('--output', default=None,
                   help='Output HTML path (default: $PLOTS_DIR/<dirname>/landscape_viewer.html)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.output is None:
        dirname = os.path.basename(args.results_dir.rstrip('/\\'))
        output  = os.path.join(_PLOTS_DIR, dirname, 'landscape_viewer.html')
    else:
        output = args.output

    print(f'Results dir : {args.results_dir}')
    print(f'Output      : {output}')

    data = load_and_group(args.results_dir)
    print(f'Params      : {data["params"]}')
    print(f'Tracks      : {len(data["all_tracks"])}  {data["all_tracks"][:4]}...')
    print(f'Planes      : {data["all_planes"]}')
    print(f'Noise opts  : {data["noise_options"]}')
    print(f'Seeds       : {data["seed_options"]}')
    print(f'Cutoffs     : {data["cutoff_options"]}')
    print(f'FFT cutoffs : {data["fourier_cutoff_options"]}')
    print(f'Runs total  : {len(data["runs"])}')

    data_json = json.dumps(data, cls=_NpEncoder, separators=(',', ':'))
    html = _HTML.replace('__DATA_JSON__', data_json)

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        f.write(html)

    size_mb = os.path.getsize(output) / 1e6
    print(f'Written     : {output}  ({size_mb:.1f} MB)')
    print(f'Open        : file://{os.path.abspath(output)}')


if __name__ == '__main__':
    main()
