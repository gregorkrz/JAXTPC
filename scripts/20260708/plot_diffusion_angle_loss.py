#!/usr/bin/env python3
"""
Build a self-contained interactive HTML dashboard for the output of
scripts/20260708/compute_diffusion_angle_loss.py — per-plane MSE / Sobolev
loss and per-frequency loss maps between a noisy truth-diffusion track and
the same track re-simulated at a wrong (--diffusion-factor) diffusion
constant, across a set of (theta, alpha) track direction angles.

Accepts one or more compute_diffusion_angle_loss.py output pickles (e.g. if
different angle pairs were computed in separate invocations/Slurm jobs) and
merges every (theta, alpha) entry found across them into one dashboard.

Two views:
  1. "Per-plane loss vs angle" — one box plot per (volume, plane), x-axis =
     angle pair, y-axis = the selected metric (MSE or Sobolev), one box per
     angle pair showing the full spread across noise realizations (not just
     mean+std) so outlier noise draws are visible. Small enough to embed
     directly in the HTML.
  2. "Per-frequency loss map" — a single heatmap of the Sobolev per-frequency
     loss contribution C(f) (mean or std across noise realizations, or
     log10(mean)), selectable by angle pair + plane. These maps are the
     heaviest data (out_h x out_w x 2 x n_planes per angle pair), so they are
     lazy-loaded per angle pair the same way tools/plot_wireplane_eval.py
     lazy-loads per-run data (separate `window.KEY = {...}` .js files,
     fetched only when that angle pair is first selected).

Usage
-----
  python scripts/20260708/plot_diffusion_angle_loss.py \\
      results/20260708_diffusion_angle_loss/diffusion_angle_loss.pkl

  python scripts/20260708/plot_diffusion_angle_loss.py \\
      --results-dir results/20260708_diffusion_angle_loss \\
      --output plots/20260708_diffusion_angle_loss/dashboard.html
"""
import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

_VOL_NAME = {0: 'East', 1: 'West'}
_META_FIELDS = (
    'labels', 'diffusion_factor', 'truth_diffusion_trans_cm2_us',
    'diff_diffusion_trans_cm2_us', 'noise_scale', 'momentum_mev',
    'start_x_mm', 'step_size_mm', 'sobolev_max_pad', 'sobolev_s', 'fourier_out_size',
)


def _cutoff_key(c):
    return f'{c:g}'


def _round_sig(arr, sig=6):
    """Round to `sig` significant figures (not decimal places) — the weighted C(f) and
    the raw power spectrum differ in magnitude by many orders (the Sobolev weight blows
    up near f=0), so a fixed decimal-place round zeros out the smaller-magnitude one."""
    arr = np.asarray(arr, dtype=np.float64)
    out = np.zeros_like(arr)
    nz = arr != 0
    if np.any(nz):
        exponent = np.floor(np.log10(np.abs(arr[nz])))
        factor = 10.0 ** (sig - 1 - exponent)
        out[nz] = np.round(arr[nz] * factor) / factor
    return out


def _rounded_or_none(x):
    return _round_sig(x).tolist() if x is not None else None


def _write_js(path, var_name, obj):
    path.write_text(f'window.{var_name}=' + json.dumps(obj, separators=(',', ':')) + ';',
                     encoding='utf-8')


def load_and_merge(pkl_paths):
    """Merge every (theta, alpha) entry across all input pickles.

    Later files win on key collisions (a warning is printed); metadata
    (labels, diffusion factor, noise settings, ...) is taken from the first
    file encountered and assumed consistent across the rest.

    Normalizes per_plane[p] to always be {adc_cutoff: {metrics...}} — pickles
    from before --adc-cutoffs was added stored the metrics dict directly (an
    implicit single cutoff of 0.0), so those get wrapped as {0.0: {...}}.
    """
    merged = {}
    meta = None
    for p in pkl_paths:
        with open(p, 'rb') as f:
            d = pickle.load(f)
        if meta is None:
            meta = {k: d.get(k) for k in _META_FIELDS}
            meta['adc_cutoffs'] = d.get('adc_cutoffs') or [0.0]
            meta['sources'] = []
        meta['sources'].append(dict(path=str(p), command=d.get('command')))
        for key, r in d['results'].items():
            key = (float(key[0]), float(key[1]))
            if key in merged:
                print(f'  [warn] duplicate angle pair theta={key[0]},alpha={key[1]} in {p} '
                      f'(overwriting entry from {merged[key]["_source"]})')
            r = dict(r)
            r['_source'] = str(p)
            r['per_plane'] = {
                plane_idx: (pp if 'mse_per_realization' not in pp else {0.0: pp})
                for plane_idx, pp in r['per_plane'].items()
            }
            merged[key] = r
    return merged, meta


def build_summary(pairs, merged, n_planes, adc_cutoffs):
    """{'mse': {plane_idx: {cutoff_key: [[values pair0], [values pair1], ...]}},
        'sobolev': {...}, 'mse_no_noise': {plane_idx: {cutoff_key: [scalar per pair]}}, 'sobolev_no_noise': {...}}

    *_no_noise entries are None for pickles from before the no-noise reference was added
    (compute_diffusion_angle_loss.py); the dashboard skips the reference marker in that case.
    A pair/plane/cutoff combination missing entirely (cutoff not computed for that pickle)
    falls back to the first available cutoff for that entry.
    """
    summary = {k: {p: {} for p in range(n_planes)}
               for k in ('mse', 'sobolev', 'mse_no_noise', 'sobolev_no_noise')}
    for p in range(n_planes):
        for c in adc_cutoffs:
            ck = _cutoff_key(c)
            for k in summary:
                summary[k][p][ck] = []
            for pair in pairs:
                per_plane = merged[pair]['per_plane'][p]
                pp = per_plane.get(c, next(iter(per_plane.values())))
                summary['mse'][p][ck].append(np.asarray(pp['mse_per_realization']).tolist())
                summary['sobolev'][p][ck].append(np.asarray(pp['sobolev_per_realization']).tolist())
                summary['mse_no_noise'][p][ck].append(pp.get('mse_no_noise'))
                summary['sobolev_no_noise'][p][ck].append(pp.get('sobolev_no_noise'))
    return summary


def write_freq_data(pairs, merged, n_planes, adc_cutoffs, data_dir):
    """One lazy-loaded .js file per angle pair:
    {plane_idx: {cutoff_key: {mean:[[..]], std:[[..]], no_noise:[[..]]|null, all:[[[..]], ...]|null}}}
    `all` holds every individual noise realization's C(f) map (n_noise of them), so the
    dashboard can page through single realizations instead of only the aggregate mean/std —
    useful when the mean looks fine but the per-realization variance/argmin location doesn't.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    pair_files = []
    for i, pair in enumerate(pairs):
        per_plane = merged[pair]['per_plane']
        obj = {}
        for p in range(n_planes):
            by_cutoff = per_plane[p]
            obj[str(p)] = {}
            for c in adc_cutoffs:
                pp = by_cutoff.get(c, next(iter(by_cutoff.values())))
                obj[str(p)][_cutoff_key(c)] = dict(
                    # Weighted per-frequency Sobolev loss contribution C(f) = power * weight.
                    mean=_rounded_or_none(pp.get('freq_map_mean')),
                    std=_rounded_or_none(pp.get('freq_map_std')),
                    no_noise=_rounded_or_none(pp.get('freq_map_no_noise')),
                    all=_rounded_or_none(pp.get('freq_maps_all')),
                    # Raw (unweighted) per-frequency power spectrum of the difference — None
                    # for pickles from before this was added.
                    power_mean=_rounded_or_none(pp.get('power_map_mean')),
                    power_std=_rounded_or_none(pp.get('power_map_std')),
                    power_no_noise=_rounded_or_none(pp.get('power_map_no_noise')),
                    power_all=_rounded_or_none(pp.get('power_maps_all')),
                )
        var_name = f'__DA_FREQ_{i}__'
        path = data_dir / f'pair_{i}.js'
        _write_js(path, var_name, obj)
        pair_files.append(dict(file=path.name, key=var_name))
    return pair_files


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>D_transverse loss vs track angle</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; padding: 12px;
       background: #f0f0f0; color: #222; }
h2 { margin: 0 0 4px; font-size: 17px; }
h3 { margin: 0 0 8px; font-size: 14px; color: #333; }
.meta { font-size: 12px; color: #666; margin: 0 0 12px; line-height: 1.5; }
.meta code { background: #e8e8ec; padding: 1px 4px; border-radius: 3px; }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-end;
            background: #fff; padding: 10px 14px; border-radius: 8px;
            margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.cg { display: flex; flex-direction: column; gap: 3px; }
.cg label { font-size: 11px; color: #666; font-weight: 600; }
.cg select { padding: 5px 8px; border-radius: 5px; border: 1px solid #ccc;
             background: #fafafa; font-size: 13px; }
.cg.chk { flex-direction: row; align-items: center; gap: 6px; }
#loading-ind { display:none; padding: 4px 12px; background: #fff3cd;
               border: 1px solid #ffc107; border-radius: 5px;
               font-size: 12px; align-self: center; }
.panel { background: #fff; border-radius: 8px; padding: 10px 12px;
         box-shadow: 0 1px 3px rgba(0,0,0,.1); margin-bottom: 14px; }
.hint { font-size: 11px; color: #888; margin: 0 0 8px; }
.plane-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.plane-grid > div { min-height: 260px; }
table.agg-table { border-collapse: collapse; font-size: 13px; width: 100%; max-width: 640px; }
table.agg-table th, table.agg-table td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }
table.agg-table th { background: #f5f5f7; font-size: 11px; color: #555; }
table.agg-table td:first-child, table.agg-table th:first-child { text-align: left; }
table.agg-table td.na { color: #bbb; }
</style>
</head>
<body>
<h2>D_transverse loss vs track angle</h2>
<p class="meta" id="meta-line"></p>

<div class="panel">
  <h3>1. Per-plane loss vs angle (spread over noise realizations)</h3>
  <p class="hint">One box per angle pair per plane, built from all
  <code>--n-noise</code> noise realizations (not just mean/std) so skew or
  outlier noise draws are visible. Compare boxes across angle pairs within a
  plane to see whether a given angle makes the D_transverse loss behave
  differently from its neighbors. The crimson diamond is the no-noise
  reference (diff-sim vs the clean, un-noised truth-D_T signal) — it should
  sit close to the noisy box for every angle; if it doesn't, the noise itself
  (not the diffusion mismatch) is driving that box. ADC cutoff zeros both
  signals wherever the noisy target is below that threshold (run_optimization.py's
  masking convention) before computing the loss — a plane is mostly
  below-threshold background, so cutoff=0 can wash out real differences that
  only live in the track region; try cutoff=50 if everything looks flat.</p>
  <div class="controls">
    <div class="cg">
      <label>Metric</label>
      <select id="metric-sel" onchange="renderPlaneGrid()">
        <option value="mse">MSE</option>
        <option value="sobolev">Sobolev (per-frequency total)</option>
      </select>
    </div>
    <div class="cg">
      <label>ADC cutoff</label>
      <select id="cutoff-sel" onchange="renderPlaneGrid()"></select>
    </div>
    <div class="cg chk">
      <input type="checkbox" id="logy-chk" onchange="renderPlaneGrid()">
      <label for="logy-chk" style="font-weight:400">Log Y axis</label>
    </div>
  </div>
  <div class="plane-grid" id="plane-grid"></div>
</div>

<div class="panel">
  <h3>2. Per-frequency loss map</h3>
  <p class="hint">Sobolev per-frequency loss contribution C(f) — sums (over all
  frequencies) to the Sobolev scalar loss shown in panel 1. Frequency axes are
  normalized (cycles/sample), zero-centered; both are always [-0.5, 0.5]
  regardless of plane shape. "No noise" compares diff-sim directly to the
  clean truth-D_T signal (no realizations involved) — the systematic
  frequency signature of the diffusion mismatch with noise entirely removed.
  Pick an individual "Realization #" to see a single noise draw's map instead
  of the aggregate — useful when the mean looks unbiased but you suspect the
  per-draw variance (or where the loss minimum sits) is what's actually
  driving the odd behavior, since Mean/Std can hide that. Median/Max/Min/IQR
  are computed per frequency bin across all realizations too (elementwise,
  not the realization whose scalar total is largest/smallest) — Max
  highlights worst-case bins, Min the floor, Median/IQR are more
  outlier-robust than Mean/Std when a few noisy draws skew them (with only
  ~20 realizations, Max/Min are themselves noisy single-draw extremes). CV
  (Std/Mean) highlights bins that are unstable <em>relative to their own
  signal level</em>, rather than just wherever Std happens to be largest in
  absolute terms — cells where the mean is exactly zero show as blank
  (undefined ratio). Correlation shows, per frequency bin, how strongly that
  bin's value tracks the realization's total Sobolev scalar loss across the
  same noise draws (Pearson r, diverging red/blue scale, blank where a bin
  or the scalar loss has zero variance) — bins near +/-1 are the ones
  actually driving whether a given noise draw's total loss is high or low;
  bins near 0 vary "for their own reasons" unrelated to the overall loss.
  "Weighting" switches
  between the actual Sobolev loss contribution C(f) = power &times; weight (what
  panel 1's Sobolev metric is built from) and the raw, unweighted power
  spectrum of the difference — the Sobolev weight 1/(f²+eps)² blows up near
  f=0 by design (it's most sensitive to smooth, large-scale differences), so
  the weighted view is almost always concentrated in a handful of pixels near
  the center with everything else near the numeric floor; the raw power
  spectrum has no such blowup and can reveal structure elsewhere in frequency
  space that the weighted view hides entirely. "Zoom to signal" (on by
  default) crops the view to the region actually carrying 98% of the total,
  instead of showing a mostly-empty canvas; uncheck it to see the full
  [-0.5, 0.5] extent.</p>
  <div class="controls">
    <div class="cg">
      <label>Angle pair</label>
      <select id="freq-pair-sel" onchange="onFreqPairChange()"></select>
    </div>
    <div class="cg">
      <label>Plane</label>
      <select id="freq-plane-sel" onchange="renderFreqMap()"></select>
    </div>
    <div class="cg">
      <label>ADC cutoff</label>
      <select id="freq-cutoff-sel" onchange="renderFreqMap()"></select>
    </div>
    <div class="cg">
      <label>Weighting</label>
      <select id="freq-weight-sel" onchange="renderFreqMap()">
        <option value="weighted">Weighted loss C(f) = power &times; Sobolev weight</option>
        <option value="raw">Raw power spectrum (unweighted)</option>
      </select>
    </div>
    <div class="cg">
      <label>Map</label>
      <select id="freq-map-sel" onchange="renderFreqMap()">
        <option value="mean">Mean (across realizations)</option>
        <option value="std">Std (across realizations)</option>
        <option value="median">Median (across realizations)</option>
        <option value="max">Max (across realizations)</option>
        <option value="min">Min (across realizations)</option>
        <option value="iqr">IQR = 75th - 25th pct (across realizations)</option>
        <option value="cv">CV = Std / Mean (across realizations)</option>
        <option value="corr">Correlation with Sobolev scalar loss (across realizations)</option>
        <option value="no_noise">No-noise</option>
        <optgroup id="real-optgroup" label="Individual realizations"></optgroup>
      </select>
    </div>
    <div class="cg chk">
      <input type="checkbox" id="freq-log-chk" onchange="renderFreqMap()">
      <label for="freq-log-chk" style="font-weight:400">Log10 scale</label>
    </div>
    <div class="cg chk">
      <input type="checkbox" id="freq-zoom-chk" checked onchange="renderFreqMap()">
      <label for="freq-zoom-chk" style="font-weight:400">Zoom to signal</label>
    </div>
    <div id="loading-ind">Loading…</div>
  </div>
  <p class="hint" id="freq-zoom-hint" style="display:none"></p>
  <div id="freq-div"></div>
</div>

<div class="panel">
  <h3>3. Aggregate |metric| by frequency radius, per track</h3>
  <p class="hint">Always uses the raw (unweighted) power spectrum. For the selected
  (ADC cutoff, metric, plane), one row per angle pair: mean absolute value of that
  metric's per-frequency-bin map, over (1) every bin, (2) only bins with
  sqrt(f_time&sup2;+f_wire&sup2;) &le; 0.05 (near DC), (3) only bins outside that
  radius. NaN bins (e.g. CV/correlation where the denominator is exactly 0) are
  excluded from all three averages. Loads every angle pair's frequency data on first
  use (not just the one selected in panel 2), so the first render may take a moment.</p>
  <div class="controls">
    <div class="cg">
      <label>ADC cutoff</label>
      <select id="agg-cutoff-sel" onchange="renderAggTable()"></select>
    </div>
    <div class="cg">
      <label>Metric</label>
      <select id="agg-metric-sel" onchange="renderAggTable()">
        <option value="mean">Mean (across realizations)</option>
        <option value="std">Std (across realizations)</option>
        <option value="median">Median (across realizations)</option>
        <option value="max">Max (across realizations)</option>
        <option value="min">Min (across realizations)</option>
        <option value="iqr">IQR = 75th - 25th pct</option>
        <option value="cv">CV = Std / Mean</option>
        <option value="corr">Correlation with Sobolev scalar loss</option>
      </select>
    </div>
    <div class="cg">
      <label>Plane</label>
      <select id="agg-plane-sel" onchange="renderAggTable()"></select>
    </div>
    <div id="agg-loading-ind" style="display:none; padding: 4px 12px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 5px; font-size: 12px;">Loading all angle pairs…</div>
  </div>
  <div id="agg-table-div"></div>
</div>

<script>
const SUMMARY = /*SUMMARY*/null/*END_SUMMARY*/;
const PAIRS = /*PAIRS*/null/*END_PAIRS*/;           // [{label, theta, alpha}, ...]
const PLANES = /*PLANES*/null/*END_PLANES*/;         // [plane label str, ...]
const FREQ_FILES = /*FREQ_FILES*/null/*END_FREQ_FILES*/;  // [{file, key}, ...] aligned with PAIRS
const FREQ_AXIS = /*FREQ_AXIS*/null/*END_FREQ_AXIS*/; // {y: [...], x: [...]}
const ADC_CUTOFFS = /*ADC_CUTOFFS*/null/*END_ADC_CUTOFFS*/;  // [cutoff key str, ...], e.g. ["0","50"]
const META = /*META*/null/*END_META*/;

document.getElementById('meta-line').innerHTML =
  `diffusion factor: <code>${META.diffusion_factor}</code> ` +
  `(truth D_T=<code>${META.truth_diffusion_trans_cm2_us.toExponential(3)}</code>, ` +
  `sim D_T=<code>${META.diff_diffusion_trans_cm2_us.toExponential(3)}</code>) &middot; ` +
  `noise_scale=<code>${META.noise_scale}</code> &middot; ` +
  `momentum=<code>${META.momentum_mev}</code> MeV &middot; ` +
  `step_size=<code>${META.step_size_mm}</code> mm &middot; ` +
  `sobolev s=<code>${META.sobolev_s}</code>, max_pad=<code>${META.sobolev_max_pad}</code> &middot; ` +
  `ADC cutoffs=<code>${ADC_CUTOFFS.join(', ')}</code> &middot; ` +
  `${PAIRS.length} angle pair(s) from ${META.sources.length} source file(s)`;

function populateCutoffSelects() {
  ['cutoff-sel', 'freq-cutoff-sel'].forEach(id => {
    const sel = document.getElementById(id);
    ADC_CUTOFFS.forEach(c => {
      const o = document.createElement('option');
      o.value = c; o.text = `${c} ADC`;
      sel.appendChild(o);
    });
  });
}

// ── Panel 1: per-plane box-plot grid ────────────────────────────────────────

function renderPlaneGrid() {
  const metric = document.getElementById('metric-sel').value;
  const cutoff = document.getElementById('cutoff-sel').value;
  const logy = document.getElementById('logy-chk').checked;
  const grid = document.getElementById('plane-grid');
  grid.innerHTML = '';
  PLANES.forEach((planeLabel, p) => {
    const div = document.createElement('div');
    div.id = `plane-chart-${p}`;
    grid.appendChild(div);

    const values = SUMMARY[metric][p][cutoff];  // [[realization values for pair0], [pair1], ...]
    const noNoise = SUMMARY[metric + '_no_noise'][p][cutoff];  // [scalar or null, per pair]
    const traces = PAIRS.map((pair, i) => ({
      type: 'box', y: values[i], name: pair.label,
      boxpoints: 'all', jitter: 0.4, pointpos: 0, marker: {size: 3},
      line: {width: 1}, showlegend: false,
    }));
    if (noNoise.some(v => v !== null)) {
      traces.push({
        type: 'scatter', mode: 'markers', name: 'no noise',
        x: PAIRS.map(pair => pair.label), y: noNoise,
        marker: {symbol: 'diamond', size: 10, color: 'crimson', line: {width: 1, color: 'black'}},
        showlegend: true,
      });
    }
    const layout = {
      title: {text: planeLabel, font: {size: 12}},
      margin: {l: 55, r: 10, t: 30, b: 70},
      showlegend: true,
      legend: {font: {size: 9}, x: 1, xanchor: 'right', y: 1.15, orientation: 'h'},
      yaxis: {title: {text: metric === 'mse' ? 'Normalized MSE' : 'Sobolev loss', font: {size: 10}},
               type: logy ? 'log' : 'linear'},
      xaxis: {tickfont: {size: 9}},
    };
    Plotly.newPlot(div.id, traces, layout, {displayModeBar: false, responsive: true});
  });
}

// ── Panel 2: per-frequency heatmap (lazy-loaded per angle pair) ────────────

const freqCache = {};

function loadScript(src, key) {
  return new Promise((resolve) => {
    if (window[key] !== undefined) { resolve(window[key]); return; }
    const el = document.createElement('script');
    el.src = src;
    el.onload  = () => resolve(window[key]);
    el.onerror = () => resolve(null);
    document.head.appendChild(el);
  });
}

async function loadFreqPair(pi) {
  if (freqCache[pi]) return freqCache[pi];
  const meta = FREQ_FILES[pi];
  const data = await loadScript(meta.file, meta.key);
  freqCache[pi] = data;
  return data;
}

function populateSelects() {
  const pairSel = document.getElementById('freq-pair-sel');
  PAIRS.forEach((pair, i) => {
    const o = document.createElement('option');
    o.value = i; o.text = pair.label;
    pairSel.appendChild(o);
  });
  const planeSel = document.getElementById('freq-plane-sel');
  PLANES.forEach((lbl, i) => {
    const o = document.createElement('option');
    o.value = i; o.text = lbl;
    planeSel.appendChild(o);
  });
}

function populateRealizationOptions(data) {
  const grp = document.getElementById('real-optgroup');
  const prevValue = document.getElementById('freq-map-sel').value;
  grp.innerHTML = '';
  if (!data) return;
  const firstPlane = Object.keys(data)[0];
  const cutoff = document.getElementById('freq-cutoff-sel').value;
  const entry = data[firstPlane] && data[firstPlane][cutoff];
  const all = entry && (entry.all || entry.power_all);
  if (!all) return;
  all.forEach((_, i) => {
    const o = document.createElement('option');
    o.value = `r${i}`; o.text = `Realization ${i}`;
    grp.appendChild(o);
  });
  // Keep the same realization index selected across angle-pair/plane switches, if it still exists.
  if (prevValue.startsWith('r') && +prevValue.slice(1) < all.length) {
    document.getElementById('freq-map-sel').value = prevValue;
  }
}

async function onFreqPairChange() {
  document.getElementById('loading-ind').style.display = 'block';
  const data = await loadFreqPair(+document.getElementById('freq-pair-sel').value);
  document.getElementById('loading-ind').style.display = 'none';
  populateRealizationOptions(data);
  renderFreqMap();
}

// Bounding box (row/col index ranges) around the cells holding `energyFrac` of the
// total sum, padded — Sobolev C(f) is normally concentrated in a handful of pixels
// near f=0 (the weight 1/(f^2+eps)^2 blows up there by design), so most of the
// [-0.5, 0.5] canvas is at the numeric floor and not worth displaying by default.
function energyBBox(z, energyFrac, padFrac, padMin) {
  const H = z.length, W = z[0].length;
  const cells = [];
  let total = 0;
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const v = z[r][c];
      cells.push([v, r, c]);
      total += v;
    }
  }
  if (total <= 0) return null;
  cells.sort((a, b) => b[0] - a[0]);
  let acc = 0, r0 = H, r1 = -1, c0 = W, c1 = -1;
  for (const [v, r, c] of cells) {
    acc += v;
    if (r < r0) r0 = r; if (r > r1) r1 = r;
    if (c < c0) c0 = c; if (c > c1) c1 = c;
    if (acc >= energyFrac * total) break;
  }
  const rpad = Math.max(padMin, Math.round((r1 - r0 + 1) * padFrac));
  const cpad = Math.max(padMin, Math.round((c1 - c0 + 1) * padFrac));
  return {
    r0: Math.max(0, r0 - rpad), r1: Math.min(H - 1, r1 + rpad),
    c0: Math.max(0, c0 - cpad), c1: Math.min(W - 1, c1 + cpad),
  };
}

function cropGrid(z, x, y, bbox) {
  const zCrop = z.slice(bbox.r0, bbox.r1 + 1).map(row => row.slice(bbox.c0, bbox.c1 + 1));
  return {z: zCrop, x: x.slice(bbox.c0, bbox.c1 + 1), y: y.slice(bbox.r0, bbox.r1 + 1)};
}

// Elementwise (per frequency bin) reduction across the n_noise realization maps in
// `all` — e.g. the max-map's cell (r,c) is max over realizations of all[i][r][c], NOT
// the single realization with the largest scalar total.
function reduceRealizations(all, fn) {
  const n = all.length, H = all[0].length, W = all[0][0].length;
  const out = new Array(H);
  for (let r = 0; r < H; r++) {
    const row = new Array(W);
    for (let c = 0; c < W; c++) {
      const vals = new Array(n);
      for (let i = 0; i < n; i++) vals[i] = all[i][r][c];
      row[c] = fn(vals);
    }
    out[r] = row;
  }
  return out;
}
function _median(vals) {
  const s = [...vals].sort((a, b) => a - b);
  const m = s.length;
  return m % 2 ? s[(m - 1) / 2] : (s[m / 2 - 1] + s[m / 2]) / 2;
}
function _percentile(sortedVals, p) {
  const n = sortedVals.length;
  if (n === 1) return sortedVals[0];
  const idx = (p / 100) * (n - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return sortedVals[lo];
  const frac = idx - lo;
  return sortedVals[lo] * (1 - frac) + sortedVals[hi] * frac;
}
function _iqr(vals) {
  const s = [...vals].sort((a, b) => a - b);
  return _percentile(s, 75) - _percentile(s, 25);
}
const _REDUCERS = {max: vals => Math.max(...vals), min: vals => Math.min(...vals), median: _median, iqr: _iqr};

// CV = Std/Mean per bin — needs only the already-precomputed mean/std maps, not `all`.
// NaN (renders as a blank cell in Plotly) wherever mean is exactly 0 (undefined ratio;
// implies std is 0 there too, since a zero mean across non-negative realizations means
// every realization was exactly 0).
function computeCV(meanMap, stdMap) {
  return stdMap.map((row, r) => row.map((v, c) => {
    const m = meanMap[r][c];
    return m === 0 ? NaN : v / m;
  }));
}

function pearsonCorr(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let cov = 0, vx = 0, vy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    cov += dx * dy; vx += dx * dx; vy += dy * dy;
  }
  return (vx === 0 || vy === 0) ? NaN : cov / Math.sqrt(vx * vy);
}

// Per-bin Pearson correlation between that bin's value (across the n_noise
// realizations) and the realization's scalar Sobolev loss for the same (pair, plane,
// cutoff) — both are indexed by the same realization order, since compute_diffusion_angle_loss.py
// fills mse/sobolev_per_realization and freq_maps_all/power_maps_all inside the same
// per-realization loop iteration.
function reduceCorrelation(all, scalars) {
  const n = all.length, H = all[0].length, W = all[0][0].length;
  const out = new Array(H);
  for (let r = 0; r < H; r++) {
    const row = new Array(W);
    for (let c = 0; c < W; c++) {
      const vals = new Array(n);
      for (let i = 0; i < n; i++) vals[i] = all[i][r][c];
      row[c] = pearsonCorr(vals, scalars);
    }
    out[r] = row;
  }
  return out;
}

async function renderFreqMap() {
  const pi = +document.getElementById('freq-pair-sel').value;
  const plane = document.getElementById('freq-plane-sel').value;
  const cutoff = document.getElementById('freq-cutoff-sel').value;
  const weightMode = document.getElementById('freq-weight-sel').value;  // 'weighted' or 'raw'
  const mapType = document.getElementById('freq-map-sel').value;
  const logScale = document.getElementById('freq-log-chk').checked;
  const zoomToSignal = document.getElementById('freq-zoom-chk').checked;
  const data = await loadFreqPair(pi);
  if (!data) {
    document.getElementById('freq-div').innerHTML = '<p style="color:#a55">Failed to load data.</p>';
    return;
  }
  const entry = data[plane][cutoff];
  const pfx = weightMode === 'raw' ? 'power_' : '';
  const isRealization = mapType.startsWith('r');
  const realizationIdx = isRealization ? +mapType.slice(1) : null;
  const needsAll = isRealization || mapType === 'corr' || mapType in _REDUCERS;
  if (mapType === 'no_noise' && entry[pfx + 'no_noise'] === null) {
    document.getElementById('freq-div').innerHTML =
      `<p style="color:#a55">This pickle predates the ${weightMode === 'raw' ? 'raw power spectrum' : 'no-noise reference'} — ` +
      'recompute with the current compute_diffusion_angle_loss.py to get it.</p>';
    return;
  }
  if (needsAll && (!entry[pfx + 'all'] || (isRealization && !entry[pfx + 'all'][realizationIdx]))) {
    document.getElementById('freq-div').innerHTML =
      `<p style="color:#a55">This pickle predates ${weightMode === 'raw' ? 'raw power spectrum' : 'per-realization'} maps — ` +
      'recompute with the current compute_diffusion_angle_loss.py to get them.</p>';
    return;
  }
  let z;
  const isDiverging = mapType === 'corr';
  if (mapType === 'std') z = entry[pfx + 'std'];
  else if (mapType === 'no_noise') z = entry[pfx + 'no_noise'];
  else if (isRealization) z = entry[pfx + 'all'][realizationIdx];
  else if (mapType === 'cv') z = computeCV(entry[pfx + 'mean'], entry[pfx + 'std']);
  else if (mapType === 'corr') {
    const scalars = SUMMARY.sobolev[plane][cutoff][pi];
    z = reduceCorrelation(entry[pfx + 'all'], scalars);
  }
  else if (mapType in _REDUCERS) z = reduceRealizations(entry[pfx + 'all'], _REDUCERS[mapType]);
  else z = entry[pfx + 'mean'];

  let x = FREQ_AXIS.x, y = FREQ_AXIS.y;
  const zoomHint = document.getElementById('freq-zoom-hint');
  // Energy-mass cropping assumes a non-negative, summable quantity — doesn't apply to a
  // signed ratio (CV) or correlation (-1..1), so always show the full extent for those.
  if (zoomToSignal && !isDiverging && mapType !== 'cv') {
    const bbox = energyBBox(z, 0.98, 0.2, 3);
    if (bbox) {
      const cropped = cropGrid(z, x, y, bbox);
      z = cropped.z; x = cropped.x; y = cropped.y;
      zoomHint.style.display = 'block';
      zoomHint.textContent = `Zoomed to the ${z[0].length}x${z.length}-cell region holding 98% of the ` +
        `total (of ${FREQ_AXIS.x.length}x${FREQ_AXIS.y.length} native) — uncheck "Zoom to signal" for the full extent.`;
    } else {
      zoomHint.style.display = 'none';
    }
  } else {
    zoomHint.style.display = 'none';
  }
  // log10 doesn't make sense for a signed correlation (-1..1); CV and everything else
  // are non-negative (aside from NaN, which Math.log10 just passes through as NaN).
  if (logScale && !isDiverging) z = z.map(row => row.map(v => Math.log10(Math.max(v, 1e-30))));

  const mapLabel = isRealization ? `realization ${realizationIdx}` : {
    mean: 'mean', std: 'std', median: 'median', max: 'max', min: 'min',
    iqr: 'IQR', cv: 'CV', corr: 'corr w/ Sobolev loss', no_noise: 'no noise',
  }[mapType];
  const zLabel = mapType === 'corr' ? 'r' : mapType === 'cv' ? 'CV' : (weightMode === 'raw' ? 'power' : 'C(f)');
  const trace = {
    type: 'heatmap', z, x, y,
    colorscale: isDiverging ? 'RdBu' : 'Viridis',
    zmin: isDiverging ? -1 : undefined, zmax: isDiverging ? 1 : undefined,
    hovertemplate: `f_time:%{x:.3f}<br>f_wire:%{y:.3f}<br>${zLabel}:%{z:.4g}<extra></extra>`,
  };
  const weightLabel = weightMode === 'raw' ? 'raw power' : 'weighted C(f)';
  const layout = {
    height: 480,
    margin: {l: 55, r: 20, t: 30, b: 45},
    title: {text: `${PAIRS[pi].label} — ${PLANES[plane]} — ${weightLabel} — ${mapLabel} — cutoff=${cutoff} ADC${logScale ? ' (log10)' : ''}`, font: {size: 13}},
    xaxis: {title: {text: 'freq (time axis, cycles/sample)'}},
    yaxis: {title: {text: 'freq (wire axis, cycles/sample)'}},
  };
  Plotly.newPlot('freq-div', [trace], layout, {displayModeBar: true, responsive: true});
}

// ── Panel 3: aggregate |metric| by frequency radius, per track ─────────────────

const AGG_RADIUS = 0.05;

function populateAggSelects() {
  const cutoffSel = document.getElementById('agg-cutoff-sel');
  ADC_CUTOFFS.forEach(c => {
    const o = document.createElement('option');
    o.value = c; o.text = `${c} ADC`;
    cutoffSel.appendChild(o);
  });
  const planeSel = document.getElementById('agg-plane-sel');
  PLANES.forEach((lbl, i) => {
    const o = document.createElement('option');
    o.value = i; o.text = lbl;
    planeSel.appendChild(o);
  });
}

// Always raw (unweighted) power — same reducers as panel 2's Weighting=raw path, minus
// the individual-realization/crop/log options panel 2 needs for visualization.
function computeMetricMapRaw(entry, mapType, scalars) {
  if (!entry) return null;
  if (mapType === 'mean') return entry.power_mean;
  if (mapType === 'std') return entry.power_std;
  if (mapType === 'cv') return entry.power_mean ? computeCV(entry.power_mean, entry.power_std) : null;
  if (mapType === 'corr') return entry.power_all ? reduceCorrelation(entry.power_all, scalars) : null;
  if (mapType in _REDUCERS) return entry.power_all ? reduceRealizations(entry.power_all, _REDUCERS[mapType]) : null;
  return null;
}

// Mean absolute value of `z` over (1) every bin, (2) bins with sqrt(fx^2+fy^2) <= radius,
// (3) bins outside that radius. NaN bins (undefined CV/corr) are skipped everywhere.
function radiusAverages(z, radius) {
  const H = z.length, W = z[0].length;
  let sumAll = 0, cntAll = 0, sumIn = 0, cntIn = 0, sumOut = 0, cntOut = 0;
  for (let r = 0; r < H; r++) {
    const fy = FREQ_AXIS.y[r];
    for (let c = 0; c < W; c++) {
      const v = z[r][c];
      if (!Number.isFinite(v)) continue;
      const av = Math.abs(v);
      sumAll += av; cntAll++;
      const fx = FREQ_AXIS.x[c];
      if (Math.sqrt(fx * fx + fy * fy) <= radius) { sumIn += av; cntIn++; } else { sumOut += av; cntOut++; }
    }
  }
  return {
    all: cntAll ? sumAll / cntAll : NaN,
    inner: cntIn ? sumIn / cntIn : NaN,
    outer: cntOut ? sumOut / cntOut : NaN,
  };
}

function _fmtAgg(v) {
  return Number.isFinite(v) ? v.toExponential(3) : '<span class="na">n/a</span>';
}

async function renderAggTable() {
  const cutoff = document.getElementById('agg-cutoff-sel').value;
  const metric = document.getElementById('agg-metric-sel').value;
  const plane = document.getElementById('agg-plane-sel').value;
  document.getElementById('agg-loading-ind').style.display = 'block';
  const allData = await Promise.all(PAIRS.map((_, pi) => loadFreqPair(pi)));
  document.getElementById('agg-loading-ind').style.display = 'none';

  const rows = PAIRS.map((pair, pi) => {
    const data = allData[pi];
    const entry = data ? data[plane][cutoff] : null;
    const scalars = SUMMARY.sobolev[plane][cutoff][pi];
    const z = computeMetricMapRaw(entry, metric, scalars);
    const agg = z ? radiusAverages(z, AGG_RADIUS) : {all: NaN, inner: NaN, outer: NaN};
    return {label: pair.label, ...agg};
  });

  const rowsHtml = rows.map(r => `<tr><td>${r.label}</td>` +
    `<td>${_fmtAgg(r.all)}</td><td>${_fmtAgg(r.inner)}</td><td>${_fmtAgg(r.outer)}</td></tr>`).join('');
  document.getElementById('agg-table-div').innerHTML = `
    <table class="agg-table">
      <thead><tr><th>Track (angle pair)</th><th>Mean |metric| — all bins</th>
      <th>Mean |metric| — radius &le; ${AGG_RADIUS}</th>
      <th>Mean |metric| — radius &gt; ${AGG_RADIUS}</th></tr></thead>
      <tbody>${rowsHtml}</tbody>
    </table>`;
}

populateSelects();
populateCutoffSelects();
populateAggSelects();
renderPlaneGrid();
document.getElementById('freq-pair-sel').value = 0;
onFreqPairChange();
renderAggTable();
</script>
</body>
</html>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pkls', nargs='*',
                         help='compute_diffusion_angle_loss.py output pickle(s)')
    parser.add_argument('--results-dir', nargs='+', default=[], metavar='DIR',
                         help='Scan recursively for *diffusion_angle_loss*.pkl (repeatable)')
    parser.add_argument('--output', default='plots/20260708_diffusion_angle_loss/dashboard.html',
                         help='Output HTML path.')
    args = parser.parse_args()

    pkls = list(args.pkls)
    for d in args.results_dir:
        pkls += sorted(glob.glob(os.path.join(d, '**', '*diffusion_angle_loss*.pkl'), recursive=True))
    pkls = sorted(set(pkls))
    if not pkls:
        sys.exit('No input pickles found (pass paths or --results-dir).')

    print(f'Loading {len(pkls)} pickle(s)...')
    merged, meta = load_and_merge(pkls)
    if not merged:
        sys.exit('No (theta, alpha) results found across the given pickle(s).')

    pairs = sorted(merged.keys())
    labels = meta['labels']  # [(vol_idx, plane_idx, plane_name), ...]
    plane_labels = [f'{_VOL_NAME.get(v, f"vol{v}")}-{name}' for v, _, name in labels]
    n_planes = len(plane_labels)

    adc_cutoffs = meta['adc_cutoffs']
    print(f'{len(pairs)} angle pair(s), {n_planes} plane(s), ADC cutoffs {adc_cutoffs}')
    summary = build_summary(pairs, merged, n_planes, adc_cutoffs)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    data_dir = out.parent / f'{out.stem}_data'
    freq_files_meta = write_freq_data(pairs, merged, n_planes, adc_cutoffs, data_dir)
    freq_files = [dict(file=f'{out.stem}_data/{fm["file"]}', key=fm['key']) for fm in freq_files_meta]

    out_h, out_w = meta['fourier_out_size']
    freq_axis = dict(y=np.linspace(-0.5, 0.5, out_h).round(4).tolist(),
                      x=np.linspace(-0.5, 0.5, out_w).round(4).tolist())

    pairs_json = [dict(label=f'θ={t:g}°, α={a:g}°', theta=t, alpha=a) for t, a in pairs]

    html = (_HTML
            .replace('/*SUMMARY*/null/*END_SUMMARY*/', json.dumps(summary, separators=(',', ':')))
            .replace('/*PAIRS*/null/*END_PAIRS*/', json.dumps(pairs_json, separators=(',', ':')))
            .replace('/*PLANES*/null/*END_PLANES*/', json.dumps(plane_labels, separators=(',', ':')))
            .replace('/*FREQ_FILES*/null/*END_FREQ_FILES*/', json.dumps(freq_files, separators=(',', ':')))
            .replace('/*FREQ_AXIS*/null/*END_FREQ_AXIS*/', json.dumps(freq_axis, separators=(',', ':')))
            .replace('/*ADC_CUTOFFS*/null/*END_ADC_CUTOFFS*/',
                     json.dumps([_cutoff_key(c) for c in adc_cutoffs], separators=(',', ':')))
            .replace('/*META*/null/*END_META*/', json.dumps(
                {k: meta[k] for k in _META_FIELDS if k != 'labels'} | {'sources': meta['sources']},
                separators=(',', ':'))))
    out.write_text(html, encoding='utf-8')
    print(f'Wrote {out}')
    print(f'Freq-map data under {data_dir}/')


if __name__ == '__main__':
    main()
