#!/usr/bin/env python3
"""Generate interactive HTML viewer for diffusion loss landscape studies.

Reads pkl files produced by 1d_gradients.py from:
  $RESULTS_DIR/1d_gradients/diffusion_startx_study/
  $RESULTS_DIR/1d_gradients/diffusion_angle_study/

Three tabs:
  1. Loss landscape (start-x study)  — seeds 0-4, one landscape per dropdown selection
  2. Loss landscape (angle study)    — same
  3. Bias vs drift distance          — argmin factor over all 100 seeds, per track and combined

Usage:
  python tools/plot_diffusion_loss_study.py
  python tools/plot_diffusion_loss_study.py --output $PLOTS_DIR/diffusion_loss_study.html
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
PLOTS_DIR   = os.environ.get("PLOTS_DIR",   "plots")

PARAMS = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
PARAM_LABELS = {
    "diffusion_trans_cm2_us": "D_trans (transverse)",
    "diffusion_long_cm2_us":  "D_long (longitudinal)",
}
ADC_CUTOFFS = [0, 5, 10, 20, 50]
SEEDS       = list(range(5))    # first 5 seeds → landscape view
ALL_SEEDS   = list(range(100))  # all 100 seeds  → bias computation

_STARTX_TRACKS_DEF = [
    ("Muon5_100MeV",  [2000, 1900, 1800, 1750, 1700, 1600, 1500, 1000, 500, 0]),
    ("Muon12_100MeV", [2000, 1900, 1800, 1750, 1700, 1600, 1500, 1000, 500, 0]),
    ("Muon4_100MeV",  [-2000, -1900, -1800, -1750, -1700, -1600, -1500, -1000, -500, 0]),
    ("Muon10_100MeV", [-2000, -1900, -1800, -1750, -1700, -1600, -1500, -1000, -500, 0]),
]
_ANGLE_THETAS = sorted(set(range(-90, 91, 20)) | {25, 15, 5, -5, -15, -25})

# Pivot-angle study: 400 MeV muon, midpoint fixed at (1000, 0, 0) mm (west volume).
# CSDA range of 400 MeV muon in LAr ≈ 1700.6 mm → half-length 850.3 mm.
# start = (1000 + 850.3·cos θ,  −850.3·sin θ,  0)  mm
_ANGLE_PIVOT_X_MM       = 1000.0
_ANGLE_PIVOT_HALF_LEN_MM = 850.3

BASE_TRACKS = [bt for bt, _ in _STARTX_TRACKS_DEF]
BASE_TRACK_LABELS = {
    "Muon5_100MeV":  "Muon 5 (west)",
    "Muon12_100MeV": "Muon 12 (west)",
    "Muon4_100MeV":  "Muon 4 (east)",
    "Muon10_100MeV": "Muon 10 (east)",
}


def _pkl_path(study_subdir, param, track_name, seed, adc_cutoff):
    range_tag  = "_range0p2"
    noise_tag  = f"_noise1_seed{seed}"
    cutoff_tag = f"_cutoff{adc_cutoff:.3g}".replace(".", "p") if adc_cutoff > 0.0 else ""
    fname = (f"sobolev_loss_geomean_log1p_N100{range_tag}_{param}"
             f"_{track_name}{noise_tag}{cutoff_tag}_perplane.pkl")
    return os.path.join(RESULTS_DIR, "1d_gradients", study_subdir, fname)


def load_landscape_data(study_subdir, track_names):
    """Flat dict 'param|track|adc|seed' → {factors, loss}. Seeds 0-4 only."""
    flat = {}
    n_found = n_miss = 0
    for param in PARAMS:
        for track in track_names:
            for adc in ADC_CUTOFFS:
                for seed in SEEDS:
                    path = _pkl_path(study_subdir, param, track, seed, adc)
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            d = pickle.load(f)
                        flat[f"{param}|{track}|{adc}|{seed}"] = {
                            "factors": [float(x) for x in d["factors"]],
                            "loss":    [float(x) for x in d["loss_values"]],
                        }
                        n_found += 1
                    else:
                        n_miss += 1
    print(f"  {n_found} found, {n_miss} missing")
    return flat


def _compute_drift_bias(verbose=True):
    """Read individual startx pkl files and aggregate argmin-factor per seed.

    Returns the bias dict (same structure as the cache file).
    """
    flat = {}
    n_found = n_miss = 0
    startx_map = {bt: xs for bt, xs in _STARTX_TRACKS_DEF}
    n_total = len(PARAMS) * len(startx_map) * len(ADC_CUTOFFS) * 5
    done = 0

    for param in PARAMS:
        for base_track, startxs in startx_map.items():
            for adc in ADC_CUTOFFS:
                if verbose:
                    print(f"  {param}  |  {base_track}  |  adc={adc}")
                for startx in startxs:
                    done += 1
                    track_name = f"{base_track}_startx_{startx}_stepsize_1mm"
                    drift_dist = abs(startx)
                    seed_factors = []  # list of (seed_idx, factor)
                    for seed in ALL_SEEDS:
                        path = _pkl_path("diffusion_startx_study", param, track_name, seed, adc)
                        if os.path.exists(path):
                            with open(path, "rb") as f:
                                d = pickle.load(f)
                            factors = np.array(d["factors"])
                            losses  = np.array(d["loss_values"])
                            seed_factors.append((seed, float(factors[np.argmin(losses)])))
                            n_found += 1
                        else:
                            n_miss += 1
                    if verbose:
                        print(f"    x={startx:+5d}  [{done:3d}/{n_total}]  "
                              f"{len(seed_factors)}/{len(ALL_SEEDS)} seeds found")
                    if seed_factors:
                        arr = np.array([v for _, v in seed_factors])
                        flat[f"{param}|{base_track}|{adc}|{drift_dist}"] = {
                            "mean":   float(arr.mean()),
                            "std":    float(arr.std()),
                            "n":      len(seed_factors),
                            "vals10": [[s, round(v, 6)] for s, v in seed_factors[:10]],
                        }

    print(f"  Drift bias done: {n_found} pkls read, {n_miss} missing")
    return {
        "baseTracks":      BASE_TRACKS,
        "baseTrackLabels": BASE_TRACK_LABELS,
        "driftDists": {
            bt: sorted([abs(x) for x in xs])
            for bt, xs in startx_map.items()
        },
        "data": flat,
    }


def load_bias_data(cache_path=None, recompute=False):
    """Load drift bias from cache pkl if available, otherwise compute and save."""
    if cache_path and not recompute and os.path.exists(cache_path):
        print(f"  Loading drift bias cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    result = _compute_drift_bias(verbose=True)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved drift bias cache → {cache_path}")
    return result


def _compute_angle_bias(verbose=True):
    """Read individual angle pkl files and aggregate argmin-factor per seed."""
    flat = {}
    n_found = n_miss = 0
    n_total = len(PARAMS) * len(ADC_CUTOFFS) * len(_ANGLE_THETAS)
    done = 0

    for param in PARAMS:
        for adc in ADC_CUTOFFS:
            if verbose:
                print(f"  {param}  |  adc={adc}")
            for theta in _ANGLE_THETAS:
                done += 1
                track_name = f"Muon_400MeV_theta_{theta}_stepsize_1mm"
                seed_factors = []  # list of (seed_idx, factor)
                for seed in ALL_SEEDS:
                    path = _pkl_path("diffusion_angle_study", param, track_name, seed, adc)
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            d = pickle.load(f)
                        factors = np.array(d["factors"])
                        losses  = np.array(d["loss_values"])
                        seed_factors.append((seed, float(factors[np.argmin(losses)])))
                        n_found += 1
                    else:
                        n_miss += 1
                if verbose:
                    print(f"    theta={theta:+4d}°  [{done:3d}/{n_total}]  "
                          f"{len(seed_factors)}/{len(ALL_SEEDS)} seeds found")
                if seed_factors:
                    arr = np.array([v for _, v in seed_factors])
                    flat[f"{param}|{adc}|{theta}"] = {
                        "mean":   float(arr.mean()),
                        "std":    float(arr.std()),
                        "n":      len(seed_factors),
                        "vals10": [[s, round(v, 6)] for s, v in seed_factors[:10]],
                    }

    print(f"  Angle bias done: {n_found} pkls read, {n_miss} missing")
    return {
        "thetas": _ANGLE_THETAS,
        "data":   flat,
    }


def load_angle_bias_data(cache_path=None, recompute=False):
    """Load start-angle bias from cache pkl if available, otherwise compute and save."""
    if cache_path and not recompute and os.path.exists(cache_path):
        print(f"  Loading angle bias cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    result = _compute_angle_bias(verbose=True)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved angle bias cache → {cache_path}")
    return result


def _compute_angle_pivot_bias(verbose=True):
    """Like _compute_angle_bias but track midpoint is fixed at (_ANGLE_PIVOT_X_MM, 0, 0)."""
    flat = {}
    n_found = n_miss = 0
    n_total = len(PARAMS) * len(ADC_CUTOFFS) * len(_ANGLE_THETAS)
    done = 0

    for param in PARAMS:
        for adc in ADC_CUTOFFS:
            if verbose:
                print(f"  {param}  |  adc={adc}")
            for theta in _ANGLE_THETAS:
                done += 1
                track_name = f"Muon_400MeV_theta_{theta}_pivot_x1000_stepsize_1mm"
                seed_factors = []
                for seed in ALL_SEEDS:
                    path = _pkl_path("diffusion_angle_pivot_study", param, track_name, seed, adc)
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            d = pickle.load(f)
                        factors = np.array(d["factors"])
                        losses  = np.array(d["loss_values"])
                        seed_factors.append((seed, float(factors[np.argmin(losses)])))
                        n_found += 1
                    else:
                        n_miss += 1
                if verbose:
                    print(f"    theta={theta:+4d}°  [{done:3d}/{n_total}]  "
                          f"{len(seed_factors)}/{len(ALL_SEEDS)} seeds found")
                if seed_factors:
                    arr = np.array([v for _, v in seed_factors])
                    flat[f"{param}|{adc}|{theta}"] = {
                        "mean":   float(arr.mean()),
                        "std":    float(arr.std()),
                        "n":      len(seed_factors),
                        "vals10": [[s, round(v, 6)] for s, v in seed_factors[:10]],
                    }

    print(f"  Angle-pivot bias done: {n_found} pkls read, {n_miss} missing")
    return {
        "thetas": _ANGLE_THETAS,
        "data":   flat,
    }


def load_angle_pivot_bias_data(cache_path=None, recompute=False):
    """Load pivot-angle bias from cache pkl if available, otherwise compute and save."""
    if cache_path and not recompute and os.path.exists(cache_path):
        print(f"  Loading angle-pivot bias cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    result = _compute_angle_pivot_bias(verbose=True)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved angle-pivot bias cache → {cache_path}")
    return result


def build_data(drift_bias_cache=None, angle_bias_cache=None, angle_pivot_bias_cache=None, recompute_bias=False):
    startx_tracks = [
        f"{base}_startx_{x}_stepsize_1mm"
        for base, xs in _STARTX_TRACKS_DEF
        for x in xs
    ]
    startx_track_labels = {
        f"{base}_startx_{x}_stepsize_1mm": f"{base}  x={x:+d} mm"
        for base, xs in _STARTX_TRACKS_DEF
        for x in xs
    }

    angle_tracks = [
        f"Muon_400MeV_theta_{theta}_stepsize_1mm"
        for theta in _ANGLE_THETAS
    ]
    angle_track_labels = {
        f"Muon_400MeV_theta_{theta}_stepsize_1mm": f"θ = {theta:+d}°"
        for theta in _ANGLE_THETAS
    }

    print("Loading diffusion_startx_study (landscape) …")
    startx_data = load_landscape_data("diffusion_startx_study", startx_tracks)
    print("Loading diffusion_angle_study (landscape) …")
    angle_data  = load_landscape_data("diffusion_angle_study",  angle_tracks)
    print("Drift bias (all 100 seeds) …")
    bias_data             = load_bias_data(cache_path=drift_bias_cache, recompute=recompute_bias)
    print("Angle bias (all 100 seeds) …")
    angle_bias_data       = load_angle_bias_data(cache_path=angle_bias_cache, recompute=recompute_bias)
    print("Angle-pivot bias (all 100 seeds) …")
    angle_pivot_bias_data = load_angle_pivot_bias_data(cache_path=angle_pivot_bias_cache, recompute=recompute_bias)

    return {
        "startx": {
            "tracks":       startx_tracks,
            "trackLabels":  startx_track_labels,
            "data":         startx_data,
        },
        "angle": {
            "tracks":       angle_tracks,
            "trackLabels":  angle_track_labels,
            "data":         angle_data,
        },
        "bias":             bias_data,
        "angle_bias":       angle_bias_data,
        "angle_pivot_bias": angle_pivot_bias_data,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Diffusion Loss Landscape Studies</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #f0f2f5; color: #1a1a1a; }
header { background: #1a1a2e; color: #eee; padding: 14px 24px; display: flex; align-items: baseline; gap: 1rem; }
header h1 { font-size: 1.1rem; font-weight: 600; letter-spacing: 0.04em; margin: 0; }
.tabs { display: flex; background: #16213e; padding: 0 24px; gap: 0; }
.tab-btn { padding: 10px 22px; border: none; cursor: pointer; background: transparent;
           color: #8899aa; font-size: 0.88rem; font-weight: 500;
           border-bottom: 3px solid transparent; transition: color .15s; }
.tab-btn.active { color: #fff; border-bottom-color: #e94560; }
.tab-btn:hover:not(.active) { color: #ccd; }
.tab-pane { display: none; padding: 20px 24px 24px; }
.tab-pane.active { display: block; }
.controls { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 16px;
            align-items: flex-end; }
.ctrl-group { display: flex; flex-direction: column; gap: 4px; }
.ctrl-group label { font-size: 0.7rem; color: #667; text-transform: uppercase;
                    letter-spacing: 0.06em; font-weight: 600; }
select { padding: 7px 10px; border: 1px solid #d0d5dd; border-radius: 6px;
         background: #fff; font-size: 0.87rem; min-width: 190px;
         box-shadow: 0 1px 2px rgba(0,0,0,.06); }
select:focus { outline: none; border-color: #e94560;
               box-shadow: 0 0 0 3px rgba(233,69,96,.12); }
.seg-group { display: flex; flex-direction: column; gap: 4px; }
.seg-btns { display: flex; gap: 0; }
.seg-btns button { padding: 7px 14px; border: 1px solid #d0d5dd; background: #fff;
                   font-size: 0.87rem; cursor: pointer; color: #444; transition: all .12s; }
.seg-btns button:first-child { border-radius: 6px 0 0 6px; }
.seg-btns button:last-child  { border-radius: 0 6px 6px 0; }
.seg-btns button:not(:first-child) { border-left: none; }
.seg-btns button.active { background: #e94560; color: #fff; border-color: #e94560; }
.seg-btns button:hover:not(.active) { background: #f8f0f2; }
.plot-wrap { background: #fff; border-radius: 8px;
             box-shadow: 0 1px 4px rgba(0,0,0,.09); overflow: hidden; }
.no-data { padding: 60px; text-align: center; color: #aab; font-size: 0.9rem;
           font-style: italic; }
</style>
</head>
<body>

<header>
  <h1>Diffusion Loss Landscape Studies</h1>
  <a href="diffusion_study_tracks/index.html" style="font-size:0.85rem;color:#aac4ff;margin-left:1rem;text-decoration:none;vertical-align:middle" target="_blank">&#128065; 3D track viewer &rarr;</a>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('startx',this)">Start-X Landscape</button>
  <button class="tab-btn"        onclick="switchTab('angle',this)">Angle Landscape</button>
  <button class="tab-btn"        onclick="switchTab('bias',this)">Bias vs |start_x|</button>
  <button class="tab-btn"        onclick="switchTab('anglebias',this)">Bias vs Angle</button>
  <button class="tab-btn"        onclick="switchTab('anglepivot',this)">Bias vs Angle (pivot x=1000)</button>
</div>

<!-- ── Tab 1: start-x landscape ─────────────────────────────────────────── -->
<div id="pane-startx" class="tab-pane active">
  <div class="controls">
    <div class="ctrl-group">
      <label>Parameter</label>
      <select id="sx-param" onchange="updateLandscape('startx')"></select>
    </div>
    <div class="ctrl-group">
      <label>ADC cutoff</label>
      <select id="sx-adc" onchange="updateLandscape('startx')"></select>
    </div>
    <div class="ctrl-group">
      <label>Track / start-x</label>
      <select id="sx-track" onchange="updateLandscape('startx')"></select>
    </div>
  </div>
  <div class="plot-wrap">
    <div id="plot-startx" style="height:520px"></div>
  </div>
</div>

<!-- ── Tab 2: angle landscape ────────────────────────────────────────────── -->
<div id="pane-angle" class="tab-pane">
  <div class="controls">
    <div class="ctrl-group">
      <label>Parameter</label>
      <select id="an-param" onchange="updateLandscape('angle')"></select>
    </div>
    <div class="ctrl-group">
      <label>ADC cutoff</label>
      <select id="an-adc" onchange="updateLandscape('angle')"></select>
    </div>
    <div class="ctrl-group">
      <label>Track angle</label>
      <select id="an-track" onchange="updateLandscape('angle')"></select>
    </div>
  </div>
  <div class="plot-wrap">
    <div id="plot-angle" style="height:520px"></div>
  </div>
</div>

<!-- ── Tab 3: bias vs |start_x| (distance from cathode) ─────────────────── -->
<div id="pane-bias" class="tab-pane">
  <div class="controls">
    <div class="ctrl-group">
      <label>Parameter</label>
      <select id="bi-param" onchange="updateBias()"></select>
    </div>
    <div class="ctrl-group">
      <label>ADC cutoff</label>
      <select id="bi-adc" onchange="updateBias()"></select>
    </div>
    <div class="seg-group">
      <label>View</label>
      <div class="seg-btns">
        <button id="bi-view-combined" class="active" onclick="setBiasView('combined')">All tracks combined</button>
        <button id="bi-view-grid"                    onclick="setBiasView('grid')">2×2 per-track grid</button>
      </div>
    </div>
  </div>
  <div class="plot-wrap">
    <div id="plot-bias" style="height:560px"></div>
  </div>
</div>

<!-- ── Tab 4: bias vs angle ───────────────────────────────────────────────── -->
<div id="pane-anglebias" class="tab-pane">
  <div class="controls">
    <div class="ctrl-group">
      <label>Parameter</label>
      <select id="ab-param" onchange="updateAngleBias()"></select>
    </div>
    <div class="ctrl-group">
      <label>ADC cutoff</label>
      <select id="ab-adc" onchange="updateAngleBias()"></select>
    </div>
  </div>
  <div class="plot-wrap">
    <div id="plot-anglebias" style="height:520px"></div>
  </div>
  <p style="margin-top:10px; font-size:0.8rem; color:#888; padding: 0 4px;">
    All tracks: 400 MeV muon starting at (1900, 0, 0) mm (west volume, near anode).
    Direction rotated by θ in the XY plane: dx = −cos θ, dy = sin θ, dz = 0.
    θ = 0° (pure drift) not included in sweep. Mean ± 1σ over 100 noise seeds.
  </p>
</div>

<!-- ── Tab 5: bias vs angle (pivot) ──────────────────────────────────────── -->
<div id="pane-anglepivot" class="tab-pane">
  <div class="controls">
    <div class="ctrl-group">
      <label>Parameter</label>
      <select id="ap-param" onchange="updateAnglePivotBias()"></select>
    </div>
    <div class="ctrl-group">
      <label>ADC cutoff</label>
      <select id="ap-adc" onchange="updateAnglePivotBias()"></select>
    </div>
  </div>
  <div class="plot-wrap">
    <div id="plot-anglepivot" style="height:520px"></div>
  </div>
  <p style="margin-top:10px; font-size:0.8rem; color:#888; padding: 0 4px;">
    All tracks: 400 MeV muon, midpoint fixed at (1000, 0, 0) mm (west volume, x = 1000 mm from cathode).
    Start: (1000 + 850.3·cos θ, −850.3·sin θ, 0) mm.  Direction: dx = −cos θ, dy = sin θ, dz = 0.
    Mean ± 1σ over 100 noise seeds.
  </p>
</div>

<script>
const DATA        = __DATA_JSON__;
const PARAMS      = __PARAMS_JSON__;
const PARAM_LABELS = __PARAM_LABELS_JSON__;
const ADC_CUTOFFS = __ADC_CUTOFFS_JSON__;
const SEEDS       = __SEEDS_JSON__;

// ── colours ──────────────────────────────────────────────────────────────
const SEED_COLORS = [
  'rgba(66,133,244,0.50)',
  'rgba(52,168,83,0.50)',
  'rgba(234,67,53,0.50)',
  'rgba(251,188,5,0.60)',
  'rgba(137,78,202,0.50)',
];
const TRACK_COLORS = {
  'Muon5_100MeV':  '#1f77b4',
  'Muon12_100MeV': '#ff7f0e',
  'Muon4_100MeV':  '#2ca02c',
  'Muon10_100MeV': '#d62728',
};

let biasView = 'combined';

// ── helpers ───────────────────────────────────────────────────────────────
function switchTab(study, btn) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('pane-' + study).classList.add('active');
  btn.classList.add('active');
  if (study === 'startx' || study === 'angle') updateLandscape(study);
  else if (study === 'bias') updateBias();
  else if (study === 'anglebias') updateAngleBias();
  else if (study === 'anglepivot') updateAnglePivotBias();
}

function selVal(id) { return document.getElementById(id).value; }

function populateSelect(id, values, labelMap) {
  const el = document.getElementById(id);
  el.innerHTML = '';
  values.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = labelMap ? (labelMap[v] != null ? labelMap[v] : v) : v;
    el.appendChild(opt);
  });
}

function noData(plotId) {
  document.getElementById(plotId).innerHTML =
    '<div class="no-data">No data available for this selection.</div>';
}

// ── Tab 1 & 2: landscape ──────────────────────────────────────────────────
function updateLandscape(study) {
  const pfx   = study === 'startx' ? 'sx' : 'an';
  const param = selVal(pfx + '-param');
  const adc   = selVal(pfx + '-adc');
  const track = selVal(pfx + '-track');
  const plotId = 'plot-' + study;

  const traces = [];
  const allLoss = [];
  let factors = null;

  SEEDS.forEach((seed, i) => {
    const key   = param + '|' + track + '|' + adc + '|' + seed;
    const entry = DATA[study].data[key];
    if (!entry) return;
    if (!factors) factors = entry.factors;
    allLoss.push(entry.loss);
    traces.push({
      x: entry.factors, y: entry.loss,
      mode: 'lines',
      name: 'seed ' + seed,
      line: { color: SEED_COLORS[i], width: 1.3 },
      hovertemplate: 'factor=%{x:.3f}  loss=%{y:.5g}<extra>seed ' + seed + '</extra>',
      legendgroup: 'seeds',
    });
  });

  if (allLoss.length === 0) { noData(plotId); return; }

  const nPts = factors.length;
  const mean  = new Float64Array(nPts);
  const mean2 = new Float64Array(nPts);
  for (const lv of allLoss) {
    lv.forEach((v, j) => { mean[j] += v; mean2[j] += v * v; });
  }
  const n = allLoss.length;
  for (let j = 0; j < nPts; j++) {
    mean[j]  /= n;
    mean2[j]  = Math.sqrt(Math.max(0, mean2[j] / n - mean[j] * mean[j]));
  }

  const upper = Array.from(mean).map((v, j) => v + mean2[j]);
  const lower = Array.from(mean).map((v, j) => v - mean2[j]);

  traces.push({
    x: [...factors, ...[...factors].reverse()],
    y: [...upper,   ...[...lower].reverse()],
    fill: 'toself', fillcolor: 'rgba(100,100,255,0.10)',
    line: { color: 'transparent' },
    hoverinfo: 'skip', showlegend: false, mode: 'lines', legendgroup: 'stat',
  });
  traces.push({
    x: factors, y: Array.from(mean),
    mode: 'lines', name: 'mean',
    line: { color: 'rgba(20,20,20,0.9)', width: 2.5 },
    hovertemplate: 'factor=%{x:.3f}  loss=%{y:.5g}<extra>mean</extra>',
    legendgroup: 'stat',
  });
  traces.push({
    x: factors, y: upper, mode: 'lines', name: 'mean+1σ',
    line: { color: 'rgba(80,80,200,0.5)', width: 1, dash: 'dot' },
    hoverinfo: 'skip', showlegend: true, legendgroup: 'stat',
  });
  traces.push({
    x: factors, y: lower, mode: 'lines', name: 'mean−1σ',
    line: { color: 'rgba(80,80,200,0.5)', width: 1, dash: 'dot' },
    hoverinfo: 'skip', showlegend: true, legendgroup: 'stat',
  });

  const trackLabel = DATA[study].trackLabels[track] || track;
  const paramLabel = PARAM_LABELS[param] || param;
  const adcLabel   = adc === '0' ? 'no ADC cut' : 'ADC ≥ ' + adc;

  Plotly.react(plotId, traces, {
    title: { text: paramLabel + '  —  ' + adcLabel + '  —  ' + trackLabel, font: {size:13}, x: 0.5 },
    xaxis: { title: {text:'param / GT', standoff:8}, tickformat:'.2f', gridcolor:'#eee', zeroline:false,
             range: [Math.min(...factors)-0.01, Math.max(...factors)+0.01] },
    yaxis: { title: {text:'loss', standoff:8}, gridcolor:'#eee', zeroline:false, rangemode:'tozero' },
    margin: {t:50, b:55, l:65, r:20},
    paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    legend: { x:1.01, xanchor:'left', y:0.99, font:{size:11},
              bgcolor:'rgba(255,255,255,0.8)', bordercolor:'#ddd', borderwidth:1 },
    shapes: [{ type:'line', x0:1, x1:1, y0:0, y1:1, yref:'paper',
               line:{color:'#bbb', width:1.2, dash:'dot'} }],
    annotations: [{ x:1, y:1, yref:'paper', text:'GT', showarrow:false,
                    xanchor:'left', xshift:5, font:{size:11, color:'#888'} }],
  }, { responsive:true, displayModeBar:true });
}

// ── Tab 3: bias ───────────────────────────────────────────────────────────
function setBiasView(view) {
  biasView = view;
  document.getElementById('bi-view-combined').classList.toggle('active', view === 'combined');
  document.getElementById('bi-view-grid').classList.toggle('active', view === 'grid');
  updateBias();
}

function _seedLines(vals10) {
  if (!vals10 || vals10.length === 0) return '(no data)';
  return vals10.map(([s, v]) => 'seed=' + s + ': ' + v.toFixed(4)).join('<br>');
}

function _biasTraces(param, adc, baseTrack, driftDists, opts) {
  const means = [], stds = [], ns = [], perSeed = [];
  driftDists.forEach(d => {
    const e = DATA.bias.data[param + '|' + baseTrack + '|' + adc + '|' + d];
    means.push(e ? e.mean : null);
    stds.push(e ? e.std : null);
    ns.push(e ? e.n : 0);
    perSeed.push(e ? _seedLines(e.vals10) : '(no data)');
  });

  const col   = opts.color;
  const axMap = (opts.xaxis ? { xaxis: opts.xaxis } : {});
  const ayMap = (opts.yaxis ? { yaxis: opts.yaxis } : {});

  const traceAll = {
    x: driftDists, y: means,
    customdata: stds.map((s, i) => [s, ns[i], perSeed[i]]),
    error_y: { type:'data', array: stds, visible:true, color: col, thickness:1.8, width:5 },
    mode: 'lines+markers',
    name: opts.name,
    legendgroup: opts.name,
    line:   { color: col, width: 2 },
    marker: { color: col, size: 6, symbol: 'circle' },
    hovertemplate: '|start_x|=%{x} mm<br>'
      + 'mean=%{y:.4f}  std=%{customdata[0]:.4f}  n=%{customdata[1]}<br>'
      + '<b>first seeds:</b><br>%{customdata[2]}'
      + '<extra>' + opts.name + '</extra>',
    ...axMap, ...ayMap,
  };

  return [traceAll];
}

function updateBias() {
  const param   = selVal('bi-param');
  const adc     = selVal('bi-adc');
  const plotId  = 'plot-bias';
  const biasD   = DATA.bias;
  const baseTracks = biasD.baseTracks;

  // Check if any data exists
  const hasAny = baseTracks.some(bt =>
    biasD.driftDists[bt].some(d => biasD.data[param + '|' + bt + '|' + adc + '|' + d])
  );
  if (!hasAny) { noData(plotId); return; }

  const paramLabel = PARAM_LABELS[param] || param;
  const adcLabel   = adc === '0' ? 'no ADC cut' : 'ADC ≥ ' + adc;
  const refShape   = { type:'line', y0:1, y1:1, x0:0, x1:1, xref:'paper',
                       line:{color:'#888', width:1.2, dash:'dot'} };

  if (biasView === 'combined') {
    // ── single plot, 4 lines ──────────────────────────────────────────────
    const traces = baseTracks.flatMap(bt =>
      _biasTraces(param, adc, bt, biasD.driftDists[bt], {
        color: TRACK_COLORS[bt],
        name:  biasD.baseTrackLabels[bt],
      })
    );

    Plotly.react(plotId, traces, {
      title: { text: paramLabel + '  —  ' + adcLabel + '  —  All tracks  (x=0: cathode → long drift; x=2000: near anode → short drift)', font:{size:11}, x:0.5 },
      xaxis: { title:{text:'|start_x| — distance from cathode (mm)', standoff:8}, gridcolor:'#eee', zeroline:false },
      yaxis: { title:{text:'Recovered factor (argmin loss)', standoff:8}, gridcolor:'#eee',
               zeroline:false },
      margin: {t:50, b:55, l:75, r:20},
      paper_bgcolor:'#fff', plot_bgcolor:'#fff',
      shapes: [refShape],
      annotations: [{ x:0.5, y:1, yref:'y', text:'GT (factor=1)', showarrow:false,
                      xref:'paper', yanchor:'bottom', font:{size:10, color:'#888'} }],
      legend: { x:1.01, xanchor:'left', y:0.99, font:{size:11},
                bgcolor:'rgba(255,255,255,0.8)', bordercolor:'#ddd', borderwidth:1 },
    }, { responsive:true, displayModeBar:true });

  } else {
    // ── 2×2 subplot grid ─────────────────────────────────────────────────
    // Layout: col1=[0,0.43] col2=[0.57,1]  row1=[0.55,1]  row2=[0,0.45]
    const xDoms = [[0.00, 0.43], [0.57, 1.00]];
    const yDoms = [[0.55, 1.00], [0.00, 0.45]];
    const axSuffix = ['', '2', '3', '4'];

    const traces = [];
    const annotations = [];
    const shapes = [];
    const layout = {
      margin: {t:50, b:55, l:65, r:20},
      paper_bgcolor:'#fff', plot_bgcolor:'#fff',
      title: { text: paramLabel + '  —  ' + adcLabel + '  —  Per track  (x=0: cathode; x=2000: near anode)', font:{size:11}, x:0.5 },
    };

    baseTracks.forEach((bt, i) => {
      const col = i % 2, row = Math.floor(i / 2);
      const ax  = axSuffix[i];
      const xKey = 'xaxis' + ax, yKey = 'yaxis' + ax;
      const xRef = 'x' + ax,    yRef = 'y' + ax;

      layout[xKey] = { domain: xDoms[col], anchor: yRef,
                       title: {text:'|start_x| — distance from cathode (mm)', standoff:6},
                       gridcolor:'#eee', zeroline:false };
      layout[yKey] = { domain: yDoms[row], anchor: xRef,
                       title: col === 0 ? {text:'Recovered factor', standoff:6} : undefined,
                       gridcolor:'#eee', zeroline:false };

      _biasTraces(param, adc, bt, biasD.driftDists[bt], {
        color: TRACK_COLORS[bt],
        name:  biasD.baseTrackLabels[bt],
        xaxis: xRef, yaxis: yRef,
      }).forEach(tr => { tr.showlegend = false; traces.push(tr); });

      // subplot title via annotation
      const titleX = (xDoms[col][0] + xDoms[col][1]) / 2;
      const titleY = yDoms[row][1] + 0.01;
      annotations.push({
        text: biasD.baseTrackLabels[bt], showarrow:false,
        xref:'paper', yref:'paper', x: titleX, y: titleY,
        xanchor:'center', yanchor:'bottom',
        font: { size:12, color: TRACK_COLORS[bt], weight:600 },
      });

      // GT reference line per subplot
      shapes.push({
        type:'line', y0:1, y1:1, x0:0, x1:1,
        xref:'paper', yref: yRef,
        line: {color:'#aaa', width:1, dash:'dot'},
      });
    });

    layout.annotations = annotations;
    layout.shapes      = shapes;
    Plotly.react(plotId, traces, layout, { responsive:true, displayModeBar:true });
  }
}

// ── Tab 4: bias vs angle ──────────────────────────────────────────────────
function updateAngleBias() {
  const param  = selVal('ab-param');
  const adc    = selVal('ab-adc');
  const plotId = 'plot-anglebias';
  const ab     = DATA.angle_bias;

  const thetas = ab.thetas;
  const means = [], stds = [], ns = [], perSeed = [];
  thetas.forEach(theta => {
    const entry = ab.data[param + '|' + adc + '|' + theta];
    means.push(entry ? entry.mean : null);
    stds.push( entry ? entry.std  : null);
    ns.push(   entry ? entry.n    : 0);
    perSeed.push(entry ? _seedLines(entry.vals10) : '(no data)');
  });

  if (means.every(v => v === null)) { noData(plotId); return; }

  const paramLabel = PARAM_LABELS[param] || param;
  const adcLabel   = adc === '0' ? 'no ADC cut' : 'ADC ≥ ' + adc;
  const col = '#1f77b4';

  const traces = [
    // shaded band for 100-seed line
    {
      x: [...thetas, ...[...thetas].reverse()],
      y: [...means.map((v,j) => v != null ? v + stds[j] : null),
          ...[...means].reverse().map((v,j2) => {
            const j = means.length - 1 - j2;
            return v != null ? v - stds[j] : null;
          })],
      fill: 'toself', fillcolor: 'rgba(31,119,180,0.10)',
      line: { color: 'transparent' },
      hoverinfo: 'skip', showlegend: false, mode: 'lines',
    },
    // 100-seed line
    {
      x: thetas, y: means,
      customdata: stds.map((s, i) => [s, ns[i], perSeed[i]]),
      error_y: { type:'data', array: stds, visible:true,
                 color: col, thickness:1.8, width:6 },
      mode: 'lines+markers',
      name: '100 seeds',
      legendgroup: 'all',
      line:   { color: col, width: 2.2 },
      marker: { color: col, size: 7, symbol: 'circle' },
      hovertemplate: 'θ=%{x}°<br>'
        + 'mean=%{y:.4f}  std=%{customdata[0]:.4f}  n=%{customdata[1]}<br>'
        + '<b>first seeds:</b><br>%{customdata[2]}'
        + '<extra>100 seeds</extra>',
    },
  ];

  Plotly.react(plotId, traces, {
    title: { text: paramLabel + '  —  ' + adcLabel + '  —  Bias vs track angle',
             font:{size:13}, x:0.5 },
    xaxis: {
      title: { text: 'Track angle θ (degrees)', standoff:8 },
      tickvals: thetas, ticktext: thetas.map(t => t + '°'),
      gridcolor:'#eee', zeroline:false,
    },
    yaxis: {
      title: { text: 'Recovered factor (argmin loss)', standoff:8 },
      gridcolor:'#eee', zeroline:false,
    },
    margin: {t:50, b:60, l:75, r:20},
    paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    showlegend: false,
    shapes: [
      { type:'line', y0:1, y1:1, x0:0, x1:1, xref:'paper',
        line:{color:'#888', width:1.2, dash:'dot'} },
      { type:'line', x0:0, x1:0, y0:0, y1:1, yref:'paper',
        line:{color:'#ddd', width:1, dash:'dot'} },
    ],
    annotations: [
      { x:0.5, y:1, yref:'y', text:'GT (factor=1)', showarrow:false,
        xref:'paper', yanchor:'bottom', font:{size:10, color:'#888'} },
      { x:0, y:1, yref:'paper', text:'θ=0° (pure drift)', showarrow:false,
        xanchor:'left', xshift:4, yshift:-14, font:{size:9, color:'#aaa'} },
    ],
  }, { responsive:true, displayModeBar:true });
}

function updateAnglePivotBias() {
  const param  = selVal('ap-param');
  const adc    = selVal('ap-adc');
  const plotId = 'plot-anglepivot';
  const ab     = DATA.angle_pivot_bias;

  const thetas = ab.thetas;
  const means = [], stds = [], ns = [], perSeed = [];
  thetas.forEach(theta => {
    const entry = ab.data[param + '|' + adc + '|' + theta];
    means.push(entry ? entry.mean : null);
    stds.push( entry ? entry.std  : null);
    ns.push(   entry ? entry.n    : 0);
    perSeed.push(entry ? _seedLines(entry.vals10) : '(no data)');
  });

  if (means.every(v => v === null)) { noData(plotId); return; }

  const paramLabel = PARAM_LABELS[param] || param;
  const adcLabel   = adc === '0' ? 'no ADC cut' : 'ADC ≥ ' + adc;
  const col = '#1f77b4';

  const traces = [
    {
      x: [...thetas, ...[...thetas].reverse()],
      y: [...means.map((v,j) => v != null ? v + stds[j] : null),
          ...[...means].reverse().map((v,j2) => {
            const j = means.length - 1 - j2;
            return v != null ? v - stds[j] : null;
          })],
      fill: 'toself', fillcolor: 'rgba(31,119,180,0.10)',
      line: { color: 'transparent' },
      hoverinfo: 'skip', showlegend: false, mode: 'lines',
    },
    {
      x: thetas, y: means,
      customdata: stds.map((s, i) => [s, ns[i], perSeed[i]]),
      error_y: { type:'data', array: stds, visible:true,
                 color: col, thickness:1.8, width:6 },
      mode: 'lines+markers',
      name: '100 seeds',
      line:   { color: col, width: 2.2 },
      marker: { color: col, size: 7, symbol: 'circle' },
      hovertemplate: 'θ=%{x}°<br>'
        + 'mean=%{y:.4f}  std=%{customdata[0]:.4f}  n=%{customdata[1]}<br>'
        + '<b>first seeds:</b><br>%{customdata[2]}'
        + '<extra>100 seeds</extra>',
    },
  ];

  Plotly.react(plotId, traces, {
    title: { text: paramLabel + '  —  ' + adcLabel + '  —  Bias vs angle (pivot x=1000 mm)',
             font:{size:13}, x:0.5 },
    xaxis: {
      title: { text: 'Track angle θ (degrees)', standoff:8 },
      tickvals: thetas, ticktext: thetas.map(t => t + '°'),
      gridcolor:'#eee', zeroline:false,
    },
    yaxis: {
      title: { text: 'Recovered factor (argmin loss)', standoff:8 },
      gridcolor:'#eee', zeroline:false,
    },
    margin: {t:50, b:60, l:75, r:20},
    paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    showlegend: false,
    shapes: [
      { type:'line', y0:1, y1:1, x0:0, x1:1, xref:'paper',
        line:{color:'#888', width:1.2, dash:'dot'} },
      { type:'line', x0:0, x1:0, y0:0, y1:1, yref:'paper',
        line:{color:'#ddd', width:1, dash:'dot'} },
    ],
    annotations: [
      { x:0.5, y:1, yref:'y', text:'GT (factor=1)', showarrow:false,
        xref:'paper', yanchor:'bottom', font:{size:10, color:'#888'} },
      { x:0, y:1, yref:'paper', text:'θ=0° (pure drift)', showarrow:false,
        xanchor:'left', xshift:4, yshift:-14, font:{size:9, color:'#aaa'} },
    ],
  }, { responsive:true, displayModeBar:true });
}

// ── init ──────────────────────────────────────────────────────────────────
function init() {
  const adcOptions = ADC_CUTOFFS.map(String);
  const adcLabels  = Object.fromEntries(
    ADC_CUTOFFS.map(a => [String(a), a === 0 ? '0 (no cut)' : String(a)])
  );

  ['startx', 'angle'].forEach(study => {
    const pfx  = study === 'startx' ? 'sx' : 'an';
    const info = DATA[study];
    populateSelect(pfx + '-param',  PARAMS,        PARAM_LABELS);
    populateSelect(pfx + '-adc',    adcOptions,    adcLabels);
    populateSelect(pfx + '-track',  info.tracks,   info.trackLabels);
  });

  populateSelect('bi-param', PARAMS,       PARAM_LABELS);
  populateSelect('bi-adc',   adcOptions,   adcLabels);
  populateSelect('ab-param', PARAMS,       PARAM_LABELS);
  populateSelect('ab-adc',   adcOptions,   adcLabels);
  populateSelect('ap-param', PARAMS,       PARAM_LABELS);
  populateSelect('ap-adc',   adcOptions,   adcLabels);

  updateLandscape('startx');
}

init();
</script>
</body>
</html>
"""


def emit_html(studies, output_path):
    replacements = {
        '__DATA_JSON__':         json.dumps(studies, separators=(',', ':')),
        '__PARAMS_JSON__':       json.dumps(PARAMS),
        '__PARAM_LABELS_JSON__': json.dumps(PARAM_LABELS),
        '__ADC_CUTOFFS_JSON__':  json.dumps(ADC_CUTOFFS),
        '__SEEDS_JSON__':        json.dumps(SEEDS),
    }
    html = _HTML
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding='utf-8')
    sz = Path(output_path).stat().st_size / 1024
    print(f"Written → {output_path}  ({sz:.0f} KB)")


def _default_cache(name):
    return os.path.join(RESULTS_DIR, "1d_gradients", f"bias_{name}_cache.pkl")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', default=os.path.join(PLOTS_DIR, 'diffusion_loss_study.html'),
                   help='Output HTML path (default: $PLOTS_DIR/diffusion_loss_study.html)')
    p.add_argument('--drift-bias-cache', default=_default_cache("drift"), metavar='PATH',
                   help='Cache pkl for drift bias data (default: $RESULTS_DIR/1d_gradients/bias_drift_cache.pkl)')
    p.add_argument('--angle-bias-cache', default=_default_cache("angle"), metavar='PATH',
                   help='Cache pkl for angle bias data (default: $RESULTS_DIR/1d_gradients/bias_angle_cache.pkl)')
    p.add_argument('--angle-pivot-bias-cache', default=_default_cache("angle_pivot"), metavar='PATH',
                   help='Cache pkl for angle-pivot bias data (default: $RESULTS_DIR/1d_gradients/bias_angle_pivot_cache.pkl)')
    p.add_argument('--recompute-bias', action='store_true',
                   help='Ignore existing bias cache and recompute from individual pkl files')
    return p.parse_args()


def main():
    args = parse_args()
    studies = build_data(
        drift_bias_cache=args.drift_bias_cache,
        angle_bias_cache=args.angle_bias_cache,
        angle_pivot_bias_cache=args.angle_pivot_bias_cache,
        recompute_bias=args.recompute_bias,
    )
    emit_html(studies, args.output)


if __name__ == '__main__':
    main()
