#!/usr/bin/env python3
"""Generate 3-D energy-deposit HTML viewer + wire-plane PDF for the diffusion study tracks.

Simulates and visualises:
  • Muon4_100MeV at |start_x| = 0, 1000, 1900 mm (east volume, fixed direction)
  • 400 MeV muon at θ = 10°, 30°, 90° in XY plane, starting at (1900, 0, 0) mm

Outputs:
  $PLOTS_DIR/diffusion_study_tracks/
      index.html                    — tabbed page: 3D viewer + ADC retention plots
      edep_3d_<name>.html           — one Plotly 3-D HTML per track
      wireplanes_6x6_gt_signals.pdf — N-track × 6-plane signal grid (shared colour scale)

Usage:
  python tools/plot_diffusion_study_tracks.py
  python tools/plot_diffusion_study_tracks.py --output-dir plots/diffusion_study_tracks
"""
import argparse
import html as _html
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from src.plots.plot_mixed_tracks_edep_wireplanes import (
    build_simulator,
    track_and_forward,
    write_edep_3d_html,
    _safe_stem,
    _compute_edep_hist_data,
    _compute_drift_dist_stats,
)
from tools.noise import generate_noise

_PLOTS_DIR = os.environ.get('PLOTS_DIR', 'plots')

# ADC cutoffs for pixel-retention plots
ADC_CUTOFFS_PIXEL = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

# ---------------------------------------------------------------------------
# Track definitions
# ---------------------------------------------------------------------------

_MUON4_DIR    = (-0.694627880, 0.476880059, 0.538588450)
_MUON4_ENERGY = 100   # MeV
_ANGLE_START  = (1900.0, 0.0, 0.0)
_ANGLE_ENERGY = 400   # MeV

# CSDA range of a 400 MeV muon in LAr ≈ 1700.6 mm → half-length 850.3 mm.
# Start is offset from the center (1000, 0, 0) by L_half along the reverse direction,
# so the track midpoint sits at x=1000 regardless of angle.
# (Matches _ANGLE_PIVOT_HALF_LEN_MM in submit_jobs_loss_studies.py.)
_PIVOT1000_CENTER  = (1000.0, 0.0, 0.0)
_PIVOT_HALF_LEN_MM = 850.3


def _angle_dir(theta_deg, alpha_deg=0):
    t, a = math.radians(theta_deg), math.radians(alpha_deg)
    return (round(-math.cos(t) * math.cos(a), 9),
            round( math.sin(t) * math.cos(a), 9),
            round( math.sin(a), 9))


def _pivot_start(theta_deg, alpha_deg=0, center=_PIVOT1000_CENTER, half_len=_PIVOT_HALF_LEN_MM):
    """Start position so the track centre is at `center`."""
    t, a = math.radians(theta_deg), math.radians(alpha_deg)
    return (
        round(center[0] + half_len * math.cos(t) * math.cos(a), 1),
        round(center[1] - half_len * math.sin(t) * math.cos(a), 1),
        round(center[2] - half_len * math.sin(a), 1),
    )


TRACKS = [
    # name, direction, energy_mev, start_position_mm
    ("Muon4_100MeV_startx-1900",  _MUON4_DIR,       _MUON4_ENERGY, (-1900.0, 0.0, 0.0)),
    ("Muon4_100MeV_startx-1000",  _MUON4_DIR,       _MUON4_ENERGY, (-1000.0, 0.0, 0.0)),
    ("Muon4_100MeV_startx0",      _MUON4_DIR,       _MUON4_ENERGY, (   0.0, 0.0, 0.0)),
    ("Muon_400MeV_theta10",  _angle_dir(10),   _ANGLE_ENERGY, _ANGLE_START),
    ("Muon_400MeV_theta30",  _angle_dir(30),   _ANGLE_ENERGY, _ANGLE_START),
    ("Muon_400MeV_theta90",  _angle_dir(90),   _ANGLE_ENERGY, _ANGLE_START),
    ("Muon_400MeV_pivot1000_theta10", _angle_dir(10), _ANGLE_ENERGY, _pivot_start(10)),
    ("Muon_400MeV_pivot1000_theta30", _angle_dir(30), _ANGLE_ENERGY, _pivot_start(30)),
    ("Muon_400MeV_pivot1000_theta90", _angle_dir(90), _ANGLE_ENERGY, _pivot_start(90)),
    ("Muon_400MeV_pivot1000_theta0_alpha30",  _angle_dir( 0, 30), _ANGLE_ENERGY, _pivot_start( 0, 30)),
    ("Muon_400MeV_pivot1000_theta20_alpha30", _angle_dir(20, 30), _ANGLE_ENERGY, _pivot_start(20, 30)),
]

_LABELS = {
    "Muon4_100MeV_startx-1900": "Muon4 100 MeV  |start_x|=1900 mm (near anode)",
    "Muon4_100MeV_startx-1000": "Muon4 100 MeV  |start_x|=1000 mm",
    "Muon4_100MeV_startx0":     "Muon4 100 MeV  |start_x|=0 mm (cathode, zero deposits expected)",
    "Muon_400MeV_theta10":      "400 MeV  θ=10° (nearly along drift)",
    "Muon_400MeV_theta30":      "400 MeV  θ=30°",
    "Muon_400MeV_theta90":      "400 MeV  θ=90° (transverse)",
    "Muon_400MeV_pivot1000_theta10": "400 MeV  pivot x=1000 mm  θ=10° (nearly along drift)",
    "Muon_400MeV_pivot1000_theta30": "400 MeV  pivot x=1000 mm  θ=30°",
    "Muon_400MeV_pivot1000_theta90": "400 MeV  pivot x=1000 mm  θ=90° (transverse)",
    "Muon_400MeV_pivot1000_theta0_alpha30":  "400 MeV  pivot x=1000 mm  θ=0°  α=30°",
    "Muon_400MeV_pivot1000_theta20_alpha30": "400 MeV  pivot x=1000 mm  θ=20°  α=30°",
}

# Short labels used in the ADC retention legend
_SHORT_LABELS = {
    "Muon4_100MeV_startx-1900": "Muon4 |x|=1900 mm",
    "Muon4_100MeV_startx-1000": "Muon4 |x|=1000 mm",
    "Muon4_100MeV_startx0":     "Muon4 |x|=0 mm",
    "Muon_400MeV_theta10":      "400 MeV θ=10°",
    "Muon_400MeV_theta30":      "400 MeV θ=30°",
    "Muon_400MeV_theta90":      "400 MeV θ=90°",
    "Muon_400MeV_pivot1000_theta10": "400 MeV piv1000 θ=10°",
    "Muon_400MeV_pivot1000_theta30": "400 MeV piv1000 θ=30°",
    "Muon_400MeV_pivot1000_theta90": "400 MeV piv1000 θ=90°",
    "Muon_400MeV_pivot1000_theta0_alpha30":  "400 MeV piv1000 θ=0° α=30°",
    "Muon_400MeV_pivot1000_theta20_alpha30": "400 MeV piv1000 θ=20° α=30°",
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir',
                   default=os.path.join(_PLOTS_DIR, 'diffusion_study_tracks'),
                   help='Output directory (default: $PLOTS_DIR/diffusion_study_tracks)')
    p.add_argument('--signal-percentile', type=float, default=99.0,
                   help='Percentile of |signal| used for wire-plane colour scale (default: 99)')
    p.add_argument('--no-response', action='store_true',
                   help='Also generate a wire-plane PDF with the wire field response disabled '
                        '(diffusion only, no induction/collection response)')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Building simulator …")
    simulator = build_simulator()
    cfg = simulator.config
    time_step_us = float(cfg.time_step_us)
    n_p = cfg.volumes[0].n_planes
    col_labels = [f'{cfg.plane_names[v][p]}{v + 1}'
                  for v in range(cfg.n_volumes) for p in range(n_p)]

    specs      = []
    tracks_raw = []
    all_planes = []
    stats_list = []

    for name, direction, energy_mev, start_mm in TRACKS:
        spec = {'name': name, 'direction': direction, 'momentum_mev': float(energy_mev)}
        print(f"  Simulating {name} …")
        track, planes = track_and_forward(simulator, spec, start_position_mm=list(start_mm))
        de = np.asarray(track['de'])
        st = {
            'n_deposits': int(len(de)),
            'mean_de':    float(de.mean()) if len(de) else 0.0,
            'total_de':   float(de.sum())  if len(de) else 0.0,
        }
        specs.append(spec)
        tracks_raw.append(track)
        all_planes.append(planes)
        stats_list.append(st)
        print(f"    → {st['n_deposits']} deposits, total dE = {st['total_de']:.4g} MeV")

    # ── shared dE colour scale ───────────────────────────────────────────────
    all_de = np.concatenate([np.asarray(t['de']) for t in tracks_raw if len(t['de']) > 0])
    if len(all_de) > 0:
        de_min   = float(np.percentile(all_de, 1))
        de_max   = float(np.percentile(all_de, 99))
        de_range = max(de_max - de_min, 1e-9)
    else:
        de_min = de_max = de_range = 1.0

    print("Writing per-track HTML files …")
    for spec, track, st in zip(specs, tracks_raw, stats_list):
        stem      = _safe_stem(spec['name'])
        html_path = os.path.join(args.output_dir, f'edep_3d_{stem}.html')
        write_edep_3d_html(track, spec, html_path,
                           de_min, de_max, de_range, cfg.volumes, stats=st)
        print(f"  Saved {html_path}")

    # ── Noise for ADC retention ──────────────────────────────────────────────
    print("Generating noise realization …")
    noise_dict = generate_noise(cfg, key=jax.random.PRNGKey(0))
    jax.block_until_ready(list(noise_dict.values()))

    def _pcts(plane_arr):
        abs_p = np.abs(np.asarray(plane_arr, dtype=np.float32))
        total = abs_p.size
        return [round(float(100.0 * np.sum(abs_p >= c) / total), 3)
                for c in ADC_CUTOFFS_PIXEL]

    def _noisy_planes(clean_planes):
        n_p = cfg.volumes[0].n_planes
        result = []
        for v in range(cfg.n_volumes):
            for p in range(n_p):
                clean = clean_planes[v * n_p + p]
                noise = np.asarray(noise_dict[(v, p)])
                if noise.shape[0] < clean.shape[0]:
                    noise = np.pad(noise, ((0, clean.shape[0] - noise.shape[0]), (0, 0)))
                result.append(clean + noise[:clean.shape[0], :clean.shape[1]])
        return result

    # ── ADC pixel-retention data ─────────────────────────────────────────────
    adc_pixel_data = []
    for (name, *_rest), planes in zip(TRACKS, all_planes):
        noisy = _noisy_planes(planes)
        adc_pixel_data.append({
            'name':         _SHORT_LABELS.get(name, name),
            'planes':       [_pcts(pl) for pl in planes],
            'planes_noisy': [_pcts(pl) for pl in noisy],
        })

    # Noise-only: one realization, track-independent
    n_p = cfg.volumes[0].n_planes
    noise_only_pcts = [
        _pcts(np.asarray(noise_dict[(v, p)]))
        for v in range(cfg.n_volumes) for p in range(n_p)
    ]

    # Noise histogram: all planes combined
    noise_all = np.concatenate([
        np.asarray(noise_dict[(v, p)]).ravel()
        for v in range(cfg.n_volumes) for p in range(n_p)
    ])
    noise_counts, noise_edges = np.histogram(noise_all, bins=200)
    noise_hist_data = {
        'edges':  [round(float(x), 4) for x in noise_edges],
        'counts': noise_counts.tolist(),
        'mean':   round(float(noise_all.mean()), 4),
        'std':    round(float(noise_all.std()),  4),
    }

    # ── wire-plane PDF ───────────────────────────────────────────────────────
    abs_stack = np.concatenate([np.abs(p).ravel() for planes in all_planes for p in planes])
    vmax_sig  = max(float(np.nanpercentile(abs_stack, args.signal_percentile)), 1e-9)
    norm_2d   = mcolors.Normalize(vmin=-vmax_sig, vmax=vmax_sig)

    n_tracks = len(specs)
    pdf_name = f'wireplanes_{n_tracks}x6_gt_signals.pdf'
    pdf_path = os.path.join(args.output_dir, pdf_name)
    print(f"Writing wire-plane PDF ({n_tracks} tracks × 6 planes) …")
    fig, axes = plt.subplots(n_tracks, 6, figsize=(18, max(8.0, 3.0 * n_tracks)),
                              constrained_layout=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(
        f'GT wire-plane signals  (shared scale ±{vmax_sig:.3g} e⁻, p{args.signal_percentile:g} |all|)\n'
        f'dE colour range [{de_min:.5g}, {de_max:.5g}] MeV',
        fontsize=11,
    )
    for i, (spec, planes) in enumerate(zip(specs, all_planes)):
        row_label = _LABELS.get(spec['name'], spec['name'])
        for j, plane in enumerate(planes):
            ax = axes[i, j]
            n_wires, n_time = plane.shape
            ax.imshow(plane, aspect='auto', origin='lower', norm=norm_2d, cmap='RdBu_r',
                      extent=[0, n_time * time_step_us, 0, n_wires])
            if i == 0:
                ax.set_title(col_labels[j], fontsize=9)
            ax.set_ylabel((row_label + '\nwire') if j == 0 else 'wire', fontsize=7,
                          rotation=90, va='center')
            ax.set_xlabel('t (μs)', fontsize=7)
            ax.tick_params(axis='both', labelsize=6)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm_2d)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.35, aspect=60, pad=0.02, label='signal (e⁻)')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {pdf_path}")

    # ── Optional no-response wire-plane PDF ──────────────────────────────────
    noresponse_pdf_name = None
    if args.no_response:
        print("Building no-response simulator (delta kernels, diffusion only) …")
        sim_nr = build_simulator(include_wire_response=False)
        all_planes_nr = []
        for name, direction, energy_mev, start_mm in TRACKS:
            spec = {'name': name, 'direction': direction, 'momentum_mev': float(energy_mev)}
            print(f"  Simulating {name} (no response) …")
            _track_nr, planes_nr = track_and_forward(sim_nr, spec, start_position_mm=list(start_mm))
            all_planes_nr.append(planes_nr)

        abs_stack_nr = np.concatenate([np.abs(p).ravel() for planes in all_planes_nr for p in planes])
        vmax_nr = max(float(np.nanpercentile(abs_stack_nr, args.signal_percentile)), 1e-9)
        norm_nr = mcolors.Normalize(vmin=-vmax_nr, vmax=vmax_nr)

        noresponse_pdf_name = f'wireplanes_{n_tracks}x6_noresponse_signals.pdf'
        noresponse_pdf_path = os.path.join(args.output_dir, noresponse_pdf_name)
        print(f"Writing no-response wire-plane PDF ({n_tracks} tracks × 6 planes) …")
        fig_nr, axes_nr = plt.subplots(n_tracks, 6, figsize=(18, max(8.0, 3.0 * n_tracks)),
                                       constrained_layout=True)
        axes_nr = np.asarray(axes_nr)
        if axes_nr.ndim == 1:
            axes_nr = axes_nr.reshape(1, -1)
        fig_nr.suptitle(
            f'Wire-plane signals — NO wire response (diffusion only)  '
            f'(shared scale ±{vmax_nr:.3g} e⁻, p{args.signal_percentile:g} |all|)',
            fontsize=11,
        )
        for i, (spec, planes) in enumerate(zip(specs, all_planes_nr)):
            row_label = _LABELS.get(spec['name'], spec['name'])
            for j, plane in enumerate(planes):
                ax = axes_nr[i, j]
                n_wires, n_time = plane.shape
                ax.imshow(plane, aspect='auto', origin='lower', norm=norm_nr, cmap='RdBu_r',
                          extent=[0, n_time * time_step_us, 0, n_wires])
                if i == 0:
                    ax.set_title(col_labels[j], fontsize=9)
                ax.set_ylabel((row_label + '\nwire') if j == 0 else 'wire', fontsize=7,
                              rotation=90, va='center')
                ax.set_xlabel('t (μs)', fontsize=7)
                ax.tick_params(axis='both', labelsize=6)
        sm_nr = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm_nr)
        sm_nr.set_array([])
        fig_nr.colorbar(sm_nr, ax=axes_nr.ravel().tolist(), shrink=0.35, aspect=60,
                        pad=0.02, label='signal (e⁻)')
        fig_nr.savefig(noresponse_pdf_path, bbox_inches='tight')
        plt.close(fig_nr)
        print(f"  Saved {noresponse_pdf_path}")

    # ── index.html ───────────────────────────────────────────────────────────
    # If --no-response wasn't requested this run, keep any existing noresponse PDF link.
    if noresponse_pdf_name is None:
        for fname in sorted(os.listdir(args.output_dir)):
            if 'noresponse' in fname and fname.endswith('.pdf'):
                noresponse_pdf_name = fname
                break

    print("Computing histogram data …")
    dist_stats    = _compute_drift_dist_stats(tracks_raw)
    edep_hist_data = _compute_edep_hist_data(tracks_raw)

    print("Writing index.html …")
    labelled_specs = [dict(s, name=_LABELS.get(s['name'], s['name'])) for s in specs]
    _write_index(specs, labelled_specs, stats_list, args.output_dir,
                 pdf_name=pdf_name,
                 noresponse_pdf_name=noresponse_pdf_name,
                 adc_data=adc_pixel_data,
                 col_labels=col_labels,
                 adc_cutoffs=ADC_CUTOFFS_PIXEL,
                 noise_only_pcts=noise_only_pcts,
                 noise_hist=noise_hist_data,
                 dist_stats=dist_stats,
                 edep_hist_data=edep_hist_data)
    print(f"  Saved {os.path.join(args.output_dir, 'index.html')}")
    print("Done.")


# ---------------------------------------------------------------------------
# index.html template (uses __PLACEHOLDER__ substitution, not f-strings,
# to avoid escaping all the JS curly braces)
# ---------------------------------------------------------------------------

_INDEX_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Diffusion study tracks</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:system-ui,sans-serif;background:#f0f2f5;color:#1a1a1a}
    .page-hdr{background:#1a1a2e;color:#eee;padding:12px 20px}
    .page-hdr h1{font-size:1.05rem;font-weight:600;margin-bottom:3px}
    .page-hdr p{font-size:0.78rem;color:#aac4ff;margin:0}
    .tabs{display:flex;background:#16213e;padding:0 20px;gap:0}
    .tab-btn{padding:9px 20px;border:none;cursor:pointer;background:transparent;
             color:#8899aa;font-size:0.85rem;font-weight:500;
             border-bottom:3px solid transparent;transition:color .15s}
    .tab-btn.active{color:#fff;border-bottom-color:#e94560}
    .tab-btn:hover:not(.active){color:#ccd}
    .tab-pane{display:none;padding:14px 20px 20px}
    .tab-pane.active{display:block}
    .toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:.8rem;margin-bottom:.4rem}
    select{padding:6px 10px;border:1px solid #d0d5dd;border-radius:5px;
           background:#fff;font-size:.87rem;min-width:140px}
    .stats-bar{font-size:.82rem;color:#444;font-family:monospace;
               background:#f5f5f5;padding:.25rem .5rem;border-radius:4px;display:inline-block}
    iframe{width:100%;height:calc(100vh - 13.5rem);border:1px solid #ccc;border-radius:4px;margin-top:.4rem}
    .seg-btns{display:flex;gap:0}
    .seg-btns button{padding:6px 15px;border:1px solid #d0d5dd;background:#fff;
                     font-size:.85rem;cursor:pointer;color:#444;transition:all .12s}
    .seg-btns button:first-child{border-radius:5px 0 0 5px}
    .seg-btns button:last-child{border-radius:0 5px 5px 0}
    .seg-btns button:not(:first-child){border-left:none}
    .seg-btns button.active{background:#e94560;color:#fff;border-color:#e94560}
    .plot-wrap{background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.09);overflow:hidden}
    .hist-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:.75rem}
    .hist-card{background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:.5rem .75rem}
    .hist-card h3{font-size:.82rem;margin-bottom:.25rem;color:#333}
    svg{display:block;overflow:visible}
    .hbar{fill:#4682B4;opacity:.72}.hbar:hover{opacity:1}
    .ebar{fill:#3a9e6a;opacity:.75}.ebar:hover{opacity:1}
    .muon-item{display:block;font-size:.76rem;padding:2px 0;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  </style>
</head>
<body>
<div class="page-hdr">
  <h1>Diffusion study tracks — energy deposits &amp; wire-plane signals</h1>
  <p>Muon4: 100 MeV, east volume, dir (−0.695, 0.477, 0.539) &nbsp;|&nbsp;
     Angle: 400 MeV, start (1900, 0, 0) mm, XY-plane rotation &nbsp;|&nbsp;
     Pivot x=1000: 400 MeV, start (1000, 0, 0) mm, XY-plane rotation</p>
</div>
<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('view3d',this)">3D View</button>
  <button class="tab-btn"        onclick="switchTab('adc',this)">ADC Retention</button>
  <button class="tab-btn"        onclick="switchTab('hist',this)">Histograms</button>
</div>

<!-- ── Tab 1: 3D View ──────────────────────────────────────────────────────── -->
<div id="pane-view3d" class="tab-pane active">
  <div class="toolbar">
    <div>
      <label for="track-select"><strong>Track</strong></label>&nbsp;
      <select id="track-select" aria-label="Choose track">
__OPTIONS__
      </select>
    </div>
    __PDF_LINK__
    __NORESPONSE_PDF_LINK__
  </div>
  __STATS_PANEL__
  <iframe id="track-frame" src="__FIRST__" title="3D energy deposits"></iframe>
</div>

<!-- ── Tab 3: Histograms ────────────────────────────────────────────────────── -->
<div id="pane-hist" class="tab-pane">
  <div style="display:flex;gap:1rem;height:calc(100vh - 9rem)">
    <div style="width:210px;min-width:140px;padding-right:.75rem;border-right:1px solid #ddd;overflow-y:auto;flex-shrink:0">
      <div style="font-size:.8rem;font-weight:600;margin-bottom:.35rem">Tracks</div>
      <div class="btn-row" style="margin-bottom:.35rem">
        <button onclick="histPreset(true)" style="padding:3px 9px;border:1px solid #d0d5dd;background:#fff;font-size:.78rem;cursor:pointer;border-radius:4px">All</button>
        <button onclick="histPreset(false)" style="padding:3px 9px;border:1px solid #d0d5dd;background:#fff;font-size:.78rem;cursor:pointer;border-radius:4px">None</button>
      </div>
      <div id="hist-cbs"></div>
    </div>
    <div style="flex:1;overflow-y:auto">
      <div class="hist-grid" id="hist-grid-main"></div>
      <h2 style="font-size:.88rem;font-weight:600;margin:1rem 0 .45rem;color:#444">Energy deposit coordinates (mm)</h2>
      <div class="hist-grid" id="edep-hist-main"></div>
    </div>
  </div>
</div>

<!-- ── Tab 2: ADC Retention ────────────────────────────────────────────────── -->
<div id="pane-adc" class="tab-pane">
  <div style="display:flex;align-items:center;gap:1rem;margin-bottom:12px;flex-wrap:wrap">
    <span style="font-size:.85rem;color:#555;font-weight:600">View:</span>
    <div class="seg-btns">
      <button class="adc-btn active" data-v="per-plane" onclick="setAdcView('per-plane')">Per wireplane</button>
      <button class="adc-btn"        data-v="per-group" onclick="setAdcView('per-group')">Per group (U / V / Y)</button>
      <button class="adc-btn"        data-v="all"       onclick="setAdcView('all')">All planes</button>
      <button class="adc-btn"        data-v="noise-hist" onclick="setAdcView('noise-hist')">Noise histogram</button>
    </div>
    <div class="seg-btns" style="margin-left:.5rem">
      <button id="log-btn" onclick="toggleLog()" style="padding:6px 12px;border:1px solid #d0d5dd;background:#fff;font-size:.85rem;cursor:pointer;border-radius:5px;color:#444">Log Y</button>
    </div>
  </div>
  <div class="plot-wrap"><div id="plot-adc"></div></div>
  <p style="margin-top:8px;font-size:.78rem;color:#888">
    % of pixels with |signal| &ge; ADC cutoff.
    Wire planes: __COL_LABELS_STR__. ADC cutoffs: __ADC_CUTOFFS_STR__ e&minus;.
  </p>
</div>

<script>
// ── 3D-view controls ────────────────────────────────────────────────────────
__STATS_JS__
const sel3d = document.getElementById('track-select');
const frame  = document.getElementById('track-frame');
__STATS_SCRIPT__

function switchTab(id, btn) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('pane-' + id).classList.add('active');
  btn.classList.add('active');
  if (id === 'adc') {
    const active = document.querySelector('.adc-btn.active');
    plotAdcRetention(active ? active.dataset.v : 'per-plane');
  }
  if (id === 'hist') { renderHist(); }
}

// ── ADC retention plots ─────────────────────────────────────────────────────
// ADC_DATA[i] = {name, planes (signal), planes_noisy (signal+noise)}
// NOISE_DATA  = {planes: [[pct,...] per plane]} — noise only, track-independent
const ADC_DATA    = __ADC_DATA_JSON__;
const NOISE_DATA  = {planes: __NOISE_DATA_JSON__};
const NOISE_HIST  = __NOISE_HIST_JSON__;
const COL_LABELS  = __COL_LABELS_JSON__;
const ADC_CUTS    = __ADC_CUTS_JSON__;
const TRACK_COLS  = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#aec7e8'];
const PLN_GROUPS  = [{label:'U',idx:[0,3]},{label:'V',idx:[1,4]},{label:'Y',idx:[2,5]}];

function avgPlanesKey(track, indices, key) {
  return ADC_CUTS.map(function(_,ci) {
    return indices.reduce(function(s,pi){return s+track[key][pi][ci];},0)/indices.length;
  });
}
function avgPcts(track, indices) { return avgPlanesKey(track, indices, 'planes'); }
function avgNoisyPcts(track, indices) { return avgPlanesKey(track, indices, 'planes_noisy'); }
function avgNoisePcts(indices) {
  return ADC_CUTS.map(function(_,ci) {
    return indices.reduce(function(s,pi){return s+NOISE_DATA.planes[pi][ci];},0)/indices.length;
  });
}

function _sigTrace(x, y, name, ti, an, showLeg) {
  return {
    x:x, y:y, name:name, legendgroup:'tr'+ti, showlegend:showLeg,
    mode:'lines+markers', marker:{size:4},
    line:{color:TRACK_COLS[ti%TRACK_COLS.length], width:1.8, dash:'solid'},
    xaxis:'x'+an, yaxis:'y'+an,
    hovertemplate:'ADC≥%{x}: %{y:.4g}%<extra>'+name+'</extra>',
  };
}
function _noisyTrace(x, y, name, ti, an, showLeg) {
  return {
    x:x, y:y, name:name+' +n', legendgroup:'tr'+ti, showlegend:showLeg,
    mode:'lines', marker:{size:3},
    line:{color:TRACK_COLS[ti%TRACK_COLS.length], width:1.2, dash:'dash'},
    xaxis:'x'+an, yaxis:'y'+an,
    hovertemplate:'ADC≥%{x}: %{y:.4g}%<extra>'+name+' +noise</extra>',
  };
}
function _noiseTrace(x, y, an, showLeg) {
  return {
    x:x, y:y, name:'noise only', legendgroup:'noise', showlegend:showLeg,
    mode:'lines',
    line:{color:'#999', width:1.2, dash:'dot'},
    xaxis:'x'+an, yaxis:'y'+an,
    hovertemplate:'ADC≥%{x}: %{y:.4g}%<extra>noise only</extra>',
  };
}
function ax(i)  { return i===0 ? ''   : String(i+1); }
function xk(i)  { return i===0 ? 'xaxis'  : 'xaxis'+String(i+1); }
function yk(i)  { return i===0 ? 'yaxis'  : 'yaxis'+String(i+1); }
function xref(i){ return i===0 ? 'x domain' : 'x'+String(i+1)+' domain'; }
function yref(i){ return i===0 ? 'y domain' : 'y'+String(i+1)+' domain'; }

var logY = false;
var adcView = 'per-plane';

function toggleLog() {
  logY = !logY;
  var btn = document.getElementById('log-btn');
  btn.style.background = logY ? '#e94560' : '#fff';
  btn.style.color = logY ? '#fff' : '#444';
  btn.style.borderColor = logY ? '#e94560' : '#d0d5dd';
  plotAdcRetention(adcView);
}

function setAdcView(v) {
  adcView = v;
  document.querySelectorAll('.adc-btn').forEach(function(b){
    b.classList.toggle('active', b.dataset.v===v);
  });
  document.getElementById('log-btn').style.display = '';
  plotAdcRetention(v);
}

function _yAxis(extraProps) {
  var base = {gridcolor:'#eee',zeroline:false};
  if (logY) {
    base.type = 'log';
  } else {
    base.type = 'linear';
    base.range = [0, 100];
  }
  return Object.assign(base, extraProps||{});
}

function plotAdcRetention(view) {
  var traces = [];
  var layout = {
    paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    margin:{t:50,b:50,l:58,r:200},
    showlegend:true,
    legend:{x:1.01,y:1,xanchor:'left',font:{size:10},tracegroupgap:2},
  };

  if (view === 'per-plane') {
    for (var pi=0; pi<6; pi++) {
      var an = ax(pi);
      ADC_DATA.forEach(function(track,ti) {
        traces.push(_sigTrace(ADC_CUTS, track.planes[pi],       track.name, ti, an, pi===0));
        traces.push(_noisyTrace(ADC_CUTS, track.planes_noisy[pi], track.name, ti, an, pi===0));
      });
      traces.push(_noiseTrace(ADC_CUTS, NOISE_DATA.planes[pi], an, pi===0));
    }
    layout.grid = {rows:2,columns:3,pattern:'independent',ygap:0.36,xgap:0.08};
    layout.height = 600;
    COL_LABELS.forEach(function(label,pi) {
      layout[xk(pi)] = {
        title:{text:pi>=3?'ADC (e⁻)':'',font:{size:10}},
        gridcolor:'#eee',zeroline:false,tickfont:{size:9},
      };
      layout[yk(pi)] = _yAxis({
        title:{text:pi%3===0?'% pixels':'',font:{size:10}},
        tickfont:{size:9},
      });
    });
    layout.annotations = COL_LABELS.map(function(label,pi) {
      return {
        text:'<b>'+label+'</b>',
        xref:xref(pi), yref:yref(pi), x:0.5, y:1.12,
        xanchor:'center', yanchor:'bottom', showarrow:false, font:{size:11},
      };
    });

  } else if (view === 'per-group') {
    PLN_GROUPS.forEach(function(grp,gi) {
      var an = ax(gi);
      ADC_DATA.forEach(function(track,ti) {
        traces.push(_sigTrace(ADC_CUTS, avgPcts(track,grp.idx),       track.name, ti, an, gi===0));
        traces.push(_noisyTrace(ADC_CUTS, avgNoisyPcts(track,grp.idx), track.name, ti, an, gi===0));
      });
      traces.push(_noiseTrace(ADC_CUTS, avgNoisePcts(grp.idx), an, gi===0));
    });
    layout.grid = {rows:1,columns:3,pattern:'independent',xgap:0.08};
    layout.height = 380;
    PLN_GROUPS.forEach(function(grp,gi) {
      layout[xk(gi)] = {title:{text:'ADC (e⁻)'},gridcolor:'#eee',zeroline:false};
      layout[yk(gi)] = _yAxis({title:{text:gi===0?'% pixels':''}});
    });
    layout.annotations = PLN_GROUPS.map(function(grp,gi) {
      return {
        text:'<b>'+grp.label+' planes (avg vol 0+1)</b>',
        xref:xref(gi), yref:yref(gi), x:0.5, y:1.12,
        xanchor:'center', yanchor:'bottom', showarrow:false, font:{size:12},
      };
    });

  } else if (view === 'noise-hist') {
    var edges  = NOISE_HIST.edges;
    var counts = NOISE_HIST.counts;
    var mu     = NOISE_HIST.mean;
    var sigma  = NOISE_HIST.std;
    var centers = [];
    for (var i=0; i<counts.length; i++) centers.push((edges[i]+edges[i+1])/2);
    var maxCount = Math.max.apply(null, counts);
    traces.push({
      x: centers, y: counts,
      type: 'bar', name: 'noise ADC',
      marker: {color: '#999', opacity: 0.75},
      hovertemplate: 'ADC=%{x:.2f}<br>count=%{y}<extra></extra>',
    });
    layout.height = 420;
    layout.xaxis = {title:{text:'ADC value (e⁻)'},gridcolor:'#eee',zeroline:true,zerolinecolor:'#ccc'};
    layout.yaxis = {title:{text:'count'},gridcolor:'#eee',zeroline:false,type:logY?'log':'linear'};
    layout.title = {
      text: 'Noise ADC distribution (all planes, seed=0)'
            + '<br><span style="font-size:12px">μ = ' + mu.toFixed(4)
            + '   σ = ' + sigma.toFixed(4) + ' e⁻</span>',
      font:{size:13}, x:0.5,
    };
    layout.bargap = 0;
    layout.shapes = [
      {type:'line',x0:mu,x1:mu,y0:0,y1:maxCount,line:{color:'#e94560',width:2,dash:'solid'}},
      {type:'line',x0:mu-sigma,x1:mu-sigma,y0:0,y1:maxCount,line:{color:'#ff7f0e',width:1.5,dash:'dash'}},
      {type:'line',x0:mu+sigma,x1:mu+sigma,y0:0,y1:maxCount,line:{color:'#ff7f0e',width:1.5,dash:'dash'}},
    ];
    layout.annotations = [
      {x:mu,         y:maxCount, text:'μ='+mu.toFixed(2),      showarrow:false, yanchor:'bottom', font:{color:'#e94560',size:11}},
      {x:mu-sigma,   y:maxCount, text:'μ−σ='+String((mu-sigma).toFixed(2)), showarrow:false, yanchor:'bottom', xanchor:'right', font:{color:'#ff7f0e',size:11}},
      {x:mu+sigma,   y:maxCount, text:'μ+σ='+String((mu+sigma).toFixed(2)), showarrow:false, yanchor:'bottom', xanchor:'left',  font:{color:'#ff7f0e',size:11}},
    ];

  } else {  // all — single subplot, an='' gives 'x'/'y' defaults
    ADC_DATA.forEach(function(track,ti) {
      var allAvg = ADC_CUTS.map(function(_,ci) {
        return track.planes.reduce(function(s,pl){return s+pl[ci];},0)/track.planes.length;
      });
      var allAvgN = ADC_CUTS.map(function(_,ci) {
        return track.planes_noisy.reduce(function(s,pl){return s+pl[ci];},0)/track.planes_noisy.length;
      });
      traces.push(_sigTrace(  ADC_CUTS, allAvg,  track.name, ti, '', true));
      traces.push(_noisyTrace(ADC_CUTS, allAvgN, track.name, ti, '', true));
    });
    traces.push(_noiseTrace(ADC_CUTS, avgNoisePcts([0,1,2,3,4,5]), '', true));
    layout.height = 420;
    layout.xaxis = {title:{text:'ADC cutoff (e⁻)'},gridcolor:'#eee',zeroline:false};
    layout.yaxis = _yAxis({title:{text:'% pixels with |signal| ≥ cutoff'}});
    layout.title = {text:'ADC retention — all wire planes (mean over 6)',font:{size:13},x:0.5};
  }

  Plotly.react('plot-adc', traces, layout, {responsive:true, displayModeBar:false});
}

// ── Histograms tab ────────────────────────────────────────────────────────────
const HIST_TRACKS = __HIST_TRACKS_JSON__;
const HIST_EDEP   = __HIST_EDEP_JSON__;
const HIST_QTYS = [
  {key:'E',         label:'T (MeV)'},
  {key:'theta',     label:'θ (°)'},
  {key:'phi',       label:'φ (°)'},
  {key:'mean_east', label:'Mean dist east wireplane (mm)'},
  {key:'mean_west', label:'Mean dist west wireplane (mm)'},
];
const H_BINS=8, H_W=290, H_H=170, H_PL=36, H_PR=10, H_PT=8, H_PB=38;
const HSVGNS='http://www.w3.org/2000/svg';

(function(){
  var cbDiv=document.getElementById('hist-cbs');
  HIST_TRACKS.forEach(function(tr,i){
    var lbl=document.createElement('label'); lbl.className='muon-item';
    var cb=document.createElement('input'); cb.type='checkbox'; cb.className='hist-cb'; cb.value=i; cb.checked=true;
    cb.addEventListener('change',renderHist);
    lbl.appendChild(cb); lbl.appendChild(document.createTextNode(' '+tr.name));
    cbDiv.appendChild(lbl);
  });
})();

function histPreset(all){
  document.querySelectorAll('.hist-cb').forEach(function(c){c.checked=all;});
  renderHist();
}

function hse(tag,attrs){
  var e=document.createElementNS(HSVGNS,tag);
  for(var k in attrs){if(Object.prototype.hasOwnProperty.call(attrs,k))e.setAttribute(k,attrs[k]);}
  return e;
}
function hst(tag,attrs,text){var e=hse(tag,attrs);e.textContent=String(text);return e;}

function hFmt(v){if(v===null||!isFinite(v))return'-';return Math.abs(v)>=100?v.toFixed(1):v.toFixed(2);}

function hComputeHist(vals){
  var finite=[];
  for(var i=0;i<vals.length;i++){if(vals[i]!==null&&isFinite(vals[i]))finite.push(vals[i]);}
  if(finite.length===0)return null;
  var lo=finite[0],hi=finite[0];
  for(var j=1;j<finite.length;j++){if(finite[j]<lo)lo=finite[j];if(finite[j]>hi)hi=finite[j];}
  if(lo===hi){lo-=0.5;hi+=0.5;}
  var w=(hi-lo)/H_BINS,counts=[],bnames=[];
  for(var bi=0;bi<H_BINS;bi++){counts.push(0);bnames.push([]);}
  for(var k=0;k<vals.length;k++){
    if(vals[k]===null||!isFinite(vals[k]))continue;
    var b=Math.floor((vals[k]-lo)/w);if(b>=H_BINS)b=H_BINS-1;
    counts[b]++;bnames[b].push(k);
  }
  return {lo:lo,hi:hi,w:w,counts:counts,bnames:bnames};
}

function hBuildSVG(hist,selNames){
  var pw=H_W-H_PL-H_PR,ph=H_H-H_PT-H_PB;
  var svg=hse('svg',{width:H_W,height:H_H});
  if(!hist){svg.appendChild(hst('text',{x:H_W/2,y:H_H/2,'text-anchor':'middle','font-size':'12',fill:'#999'},'no data'));return svg;}
  var maxCnt=1;
  for(var mi=0;mi<hist.counts.length;mi++){if(hist.counts[mi]>maxCnt)maxCnt=hist.counts[mi];}
  var bw=pw/H_BINS;
  for(var b=0;b<H_BINS;b++){
    var x=H_PL+b*bw,cnt=hist.counts[b],bh=(cnt/maxCnt)*ph;
    if(cnt>0&&bh<1)bh=1;
    var y=H_PT+ph-bh;
    var r=hse('rect',{'class':'hbar',x:(x+1).toFixed(1),y:y.toFixed(1),width:(bw-2).toFixed(1),height:bh.toFixed(1)});
    if(hist.bnames[b].length>0){
      var tip=document.createElementNS(HSVGNS,'title');
      var lines=[cnt+' track(s):'];
      for(var ti=0;ti<hist.bnames[b].length;ti++)lines.push(selNames[hist.bnames[b][ti]]);
      tip.textContent=lines.join('\\n');r.appendChild(tip);
    }
    svg.appendChild(r);
    if(cnt>0)svg.appendChild(hst('text',{x:(x+bw/2).toFixed(1),y:(y-2).toFixed(1),'text-anchor':'middle','font-size':'10',fill:'#333'},cnt));
  }
  svg.appendChild(hse('line',{x1:H_PL,y1:H_PT+ph,x2:H_PL+pw,y2:H_PT+ph,stroke:'#888','stroke-width':'1'}));
  for(var t=0;t<=4;t++){
    var frac=t/4,xp=H_PL+frac*pw,xval=hist.lo+frac*(hist.hi-hist.lo);
    svg.appendChild(hse('line',{x1:xp.toFixed(1),y1:(H_PT+ph).toFixed(1),x2:xp.toFixed(1),y2:(H_PT+ph+4).toFixed(1),stroke:'#888','stroke-width':'1'}));
    svg.appendChild(hst('text',{x:xp.toFixed(1),y:(H_PT+ph+14).toFixed(1),'text-anchor':'middle','font-size':'9',fill:'#555'},hFmt(xval)));
  }
  svg.appendChild(hse('line',{x1:H_PL,y1:H_PT,x2:H_PL,y2:H_PT+ph,stroke:'#888','stroke-width':'1'}));
  for(var t2=0;t2<=4;t2++){
    var frac2=t2/4,yp=H_PT+ph-frac2*ph,ycnt=Math.round(frac2*maxCnt);
    svg.appendChild(hse('line',{x1:(H_PL-3).toFixed(1),y1:yp.toFixed(1),x2:H_PL,y2:yp.toFixed(1),stroke:'#888','stroke-width':'1'}));
    svg.appendChild(hst('text',{x:(H_PL-5).toFixed(1),y:(yp+3).toFixed(1),'text-anchor':'end','font-size':'9',fill:'#555'},ycnt));
  }
  return svg;
}

function hBuildPreBinnedSVG(counts,edges){
  var n=counts.length,pw=H_W-H_PL-H_PR,ph=H_H-H_PT-H_PB;
  var svg=hse('svg',{width:H_W,height:H_H});
  var total=0;for(var i=0;i<n;i++)total+=counts[i];
  if(total===0){svg.appendChild(hst('text',{x:H_W/2,y:H_H/2,'text-anchor':'middle','font-size':'12',fill:'#999'},'no data'));return svg;}
  var maxCnt=1;for(var mi=0;mi<n;mi++){if(counts[mi]>maxCnt)maxCnt=counts[mi];}
  var bw=pw/n;
  for(var b=0;b<n;b++){
    var x=H_PL+b*bw,cnt=counts[b],bh=(cnt/maxCnt)*ph;
    if(cnt>0&&bh<1)bh=1;
    svg.appendChild(hse('rect',{'class':'ebar',x:(x+0.5).toFixed(1),y:(H_PT+ph-bh).toFixed(1),width:Math.max(1,bw-1).toFixed(1),height:bh.toFixed(1)}));
  }
  svg.appendChild(hse('line',{x1:H_PL,y1:H_PT+ph,x2:H_PL+pw,y2:H_PT+ph,stroke:'#888','stroke-width':'1'}));
  var lo=edges[0],hi=edges[edges.length-1];
  for(var t=0;t<=4;t++){
    var frac=t/4,xp=H_PL+frac*pw,xval=lo+frac*(hi-lo);
    svg.appendChild(hse('line',{x1:xp.toFixed(1),y1:(H_PT+ph).toFixed(1),x2:xp.toFixed(1),y2:(H_PT+ph+4).toFixed(1),stroke:'#888','stroke-width':'1'}));
    svg.appendChild(hst('text',{x:xp.toFixed(1),y:(H_PT+ph+14).toFixed(1),'text-anchor':'middle','font-size':'9',fill:'#555'},hFmt(xval)));
  }
  svg.appendChild(hse('line',{x1:H_PL,y1:H_PT,x2:H_PL,y2:H_PT+ph,stroke:'#888','stroke-width':'1'}));
  for(var t2=0;t2<=4;t2++){
    var frac2=t2/4,yp=H_PT+ph-frac2*ph,ycnt=Math.round(frac2*maxCnt);
    svg.appendChild(hse('line',{x1:(H_PL-3).toFixed(1),y1:yp.toFixed(1),x2:H_PL,y2:yp.toFixed(1),stroke:'#888','stroke-width':'1'}));
    svg.appendChild(hst('text',{x:(H_PL-5).toFixed(1),y:(yp+3).toFixed(1),'text-anchor':'end','font-size':'9',fill:'#555'},ycnt));
  }
  return svg;
}

function hMeanStd(vals){
  var fin=[];
  for(var i=0;i<vals.length;i++){if(vals[i]!==null&&isFinite(vals[i]))fin.push(vals[i]);}
  if(fin.length===0)return null;
  var m=0; for(var i=0;i<fin.length;i++)m+=fin[i]; m/=fin.length;
  var v=0; for(var i=0;i<fin.length;i++)v+=(fin[i]-m)*(fin[i]-m); v/=fin.length;
  return {mean:m,std:Math.sqrt(v)};
}
function hMeanStdBinned(counts,edges){
  var n=counts.length,tot=0,ws=0,ws2=0;
  for(var b=0;b<n;b++){var c=(edges[b]+edges[b+1])/2;tot+=counts[b];ws+=counts[b]*c;ws2+=counts[b]*c*c;}
  if(tot===0)return null;
  var m=ws/tot; return {mean:m,std:Math.sqrt(Math.max(0,ws2/tot-m*m))};
}
function hTitle(label,st){
  if(!st)return label;
  return label+'  μ='+hFmt(st.mean)+'  σ='+hFmt(st.std);
}

function renderHist(){
  var cbs=document.querySelectorAll('.hist-cb:checked');
  var selTracks=[],selNames=[],selIdx=[];
  for(var ci=0;ci<cbs.length;ci++){
    var idx=+cbs[ci].value;
    selIdx.push(idx);selTracks.push(HIST_TRACKS[idx]);selNames.push(HIST_TRACKS[idx].name);
  }
  var grid=document.getElementById('hist-grid-main');
  while(grid.firstChild)grid.removeChild(grid.firstChild);
  HIST_QTYS.forEach(function(q){
    var vals=selTracks.map(function(t){return t[q.key];});
    var card=document.createElement('div');card.className='hist-card';
    var h3=document.createElement('h3');h3.textContent=hTitle(q.label,hMeanStd(vals));card.appendChild(h3);
    card.appendChild(hBuildSVG(hComputeHist(vals),selNames));
    grid.appendChild(card);
  });
  if(!HIST_EDEP)return;
  var n=HIST_EDEP.x_edges.length-1,sx=[],sy=[],sz=[];
  for(var b=0;b<n;b++){sx.push(0);sy.push(0);sz.push(0);}
  selIdx.forEach(function(i){
    var d=HIST_EDEP.tracks[i];if(!d)return;
    for(var b2=0;b2<n;b2++){sx[b2]+=d.hx[b2];sy[b2]+=d.hy[b2];sz[b2]+=d.hz[b2];}
  });
  var eg=document.getElementById('edep-hist-main');
  while(eg.firstChild)eg.removeChild(eg.firstChild);
  [{label:'Edep x (mm)',counts:sx,edges:HIST_EDEP.x_edges},
   {label:'Edep y (mm)',counts:sy,edges:HIST_EDEP.y_edges},
   {label:'Edep z (mm)',counts:sz,edges:HIST_EDEP.z_edges}].forEach(function(ax){
    var card=document.createElement('div');card.className='hist-card';
    var h3=document.createElement('h3');h3.textContent=hTitle(ax.label,hMeanStdBinned(ax.counts,ax.edges));card.appendChild(h3);
    card.appendChild(hBuildPreBinnedSVG(ax.counts,ax.edges));
    eg.appendChild(card);
  });
}
</script>
</body>
</html>
"""


def _write_index(specs, labelled_specs, stats_list, output_dir,
                 pdf_name=None, noresponse_pdf_name=None,
                 adc_data=None, col_labels=None, adc_cutoffs=None,
                 noise_only_pcts=None, noise_hist=None,
                 dist_stats=None, edep_hist_data=None):
    has_stats = len(stats_list) == len(specs)

    option_lines  = []
    stats_js_rows = []
    for i, (spec, lspec) in enumerate(zip(specs, labelled_specs)):
        basename = f'edep_3d_{_safe_stem(spec["name"])}.html'
        label    = lspec['name']
        sel      = ' selected' if i == 0 else ''
        option_lines.append(
            f'        <option value="{_html.escape(basename, quote=True)}"{sel}>'
            f'{_html.escape(label)}</option>'
        )
        if has_stats:
            st = stats_list[i]
            stats_js_rows.append(
                f'  {{n:{st["n_deposits"]},mean:{st["mean_de"]:.6g},total:{st["total_de"]:.6g}}}'
            )

    first   = f'edep_3d_{_safe_stem(specs[0]["name"])}.html'
    options = '\n'.join(option_lines)

    if has_stats:
        stats_js     = 'const STATS=[\n' + ',\n'.join(stats_js_rows) + '\n];'
        stats_panel  = ('<div class="stats-bar" id="stats-bar"></div>')
        stats_script = (
            'function updateStats(idx){\n'
            '  var s=STATS[idx];\n'
            '  document.getElementById(\'stats-bar\').textContent=\n'
            '    \'N deposits: \'+s.n.toLocaleString()+\n'
            '    \'   mean dE: \'+s.mean.toPrecision(4)+\' MeV\'+\n'
            '    \'   total dE: \'+s.total.toPrecision(4)+\' MeV\';\n'
            '}\n'
            'updateStats(sel3d.selectedIndex);\n'
            'sel3d.addEventListener(\'change\',function(){'
            'frame.src=sel3d.value;updateStats(sel3d.selectedIndex);});'
        )
    else:
        stats_js     = ''
        stats_panel  = ''
        stats_script = "sel3d.addEventListener('change',function(){frame.src=sel3d.value;});"

    pdf_link = (
        f'<a href="{_html.escape(pdf_name, quote=True)}" target="_blank" '
        f'style="font-size:.85rem;text-decoration:none">&#128196; Wire-plane signals PDF</a>'
    ) if pdf_name else ''
    noresponse_pdf_link = (
        f'<a href="{_html.escape(noresponse_pdf_name, quote=True)}" target="_blank" '
        f'style="font-size:.85rem;text-decoration:none;margin-left:.6rem">'
        f'&#128196; No-response PDF (diffusion only)</a>'
    ) if noresponse_pdf_name else ''

    # histogram tab data
    hist_track_data = []
    for i, spec in enumerate(specs):
        d = np.array(spec['direction'], dtype=float)
        norm = float(np.linalg.norm(d))
        if norm > 0:
            d = d / norm
        theta_deg = float(np.degrees(np.arccos(float(np.clip(d[2], -1.0, 1.0)))))
        phi_deg   = float(np.degrees(np.arctan2(float(d[1]), float(d[0]))))
        ds = (dist_stats[i]
              if dist_stats and i < len(dist_stats) and dist_stats[i] is not None
              else None)
        hist_track_data.append({
            'name':      spec['name'],
            'E':         float(spec['momentum_mev']),
            'theta':     theta_deg,
            'phi':       phi_deg,
            'mean_east': float(ds['east']['mean']) if ds else None,
            'mean_west': float(ds['west']['mean']) if ds else None,
        })

    page = _INDEX_TMPL
    page = page.replace('__OPTIONS__',        options)
    page = page.replace('__FIRST__',          _html.escape(first))
    page = page.replace('__STATS_JS__',       stats_js)
    page = page.replace('__STATS_PANEL__',    stats_panel)
    page = page.replace('__STATS_SCRIPT__',   stats_script)
    page = page.replace('__PDF_LINK__',       pdf_link)
    page = page.replace('__NORESPONSE_PDF_LINK__', noresponse_pdf_link)
    page = page.replace('__ADC_DATA_JSON__',   json.dumps(adc_data or [], separators=(',', ':')))
    page = page.replace('__NOISE_DATA_JSON__', json.dumps(noise_only_pcts or [], separators=(',', ':')))
    page = page.replace('__NOISE_HIST_JSON__', json.dumps(noise_hist or {}, separators=(',', ':')))
    page = page.replace('__COL_LABELS_JSON__', json.dumps(col_labels or []))
    page = page.replace('__ADC_CUTS_JSON__',   json.dumps(adc_cutoffs or []))
    page = page.replace('__COL_LABELS_STR__', ', '.join(col_labels or []))
    page = page.replace('__ADC_CUTOFFS_STR__', ', '.join(str(c) for c in (adc_cutoffs or [])))
    page = page.replace('__HIST_TRACKS_JSON__', json.dumps(hist_track_data, separators=(',', ':')))
    page = page.replace('__HIST_EDEP_JSON__',
                        json.dumps(edep_hist_data, separators=(',', ':')) if edep_hist_data else 'null')

    with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(page)


if __name__ == '__main__':
    main()
