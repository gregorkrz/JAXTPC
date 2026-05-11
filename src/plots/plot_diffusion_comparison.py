#!/usr/bin/env python
"""
Compare wire-plane signals for two sets of diffusion parameters (GT1 vs GT2).

Each output PDF contains one page per track with three rows of panels:
  row 1 — GT1 wire-plane signals
  row 2 — GT2 wire-plane signals
  row 3 — difference  GT2 − GT1

All volumes are shown side by side (U1 V1 Y1 | U2 V2 Y2).
GT1 and GT2 share a symmetric colour scale; the difference row has its own.

Usage
-----
    # Compare nominal transverse diffusion vs 5× larger
    python src/plots/plot_diffusion_comparison.py \\
        --gt2-diffusion-trans 6e-5 \\
        --gt2-label "5× trans diffusion" \\
        --output-dir plots/diffusion_comparison/trans5x

    # Compare nominal longitudinal diffusion vs 10× larger
    python src/plots/plot_diffusion_comparison.py \\
        --gt2-diffusion-long 7.2e-5 \\
        --gt2-label "10× long diffusion" \\
        --tracks diagonal

    # Both diffusion values different, custom labels
    python src/plots/plot_diffusion_comparison.py \\
        --gt1-diffusion-trans 6e-6 --gt1-diffusion-long 3.6e-6 \\
        --gt2-diffusion-trans 2.4e-5 --gt2-diffusion-long 1.44e-5 \\
        --gt1-label "0.5× diffusion" --gt2-label "2× diffusion"
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

import argparse
import time

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import build_deposit_data
from tools.particle_generator import generate_muon_track
from tools.random_boundary_tracks import filter_track_inside_volumes

_PLOTS_DIR = os.environ.get('PLOTS_DIR', 'plots')

CONFIG_PATH        = 'config/cubic_wireplane_config.yaml'
N_SEGMENTS         = 50_000
MAX_ACTIVE_BUCKETS = 1000
GT_LIFETIME_US     = 10_000.0
GT_VELOCITY_CM_US  = 0.160

TRACK_PRESETS = {
    'diagonal':        ((1.0,  1.0,  1.0),  1000.0),
    'diagonal_100MeV': ((1.0,  1.0,  1.0),   100.0),
    'X':               ((1.0,  0.0,  0.0),  1000.0),
    'Y':               ((0.0,  1.0,  0.0),  1000.0),
    'Z':               ((0.0,  0.0,  1.0),  1000.0),
    'U':               ((0.0,  0.866, 0.5), 1000.0),
    'V':               ((0.0, -0.866, 0.5), 1000.0),
    'track2':          ((0.5,  1.05, 0.2),   200.0),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tracks', default='diagonal',
                   help='"+"-separated preset names or name:dx,dy,dz:mom_mev. '
                        f'Presets: {", ".join(TRACK_PRESETS)}')

    # GT1 diffusion — defaults to simulator nominal
    p.add_argument('--gt1-diffusion-trans', type=float, default=None,
                   help='GT1 transverse diffusion in cm²/μs (default: simulator nominal ~1.2e-5)')
    p.add_argument('--gt1-diffusion-long', type=float, default=None,
                   help='GT1 longitudinal diffusion in cm²/μs (default: simulator nominal ~7.2e-6)')
    p.add_argument('--gt1-label', default='GT1', help='Label for GT1 (default: GT1)')

    # GT2 diffusion — must differ from GT1 in at least one value
    p.add_argument('--gt2-diffusion-trans', type=float, default=None,
                   help='GT2 transverse diffusion in cm²/μs (default: same as GT1)')
    p.add_argument('--gt2-diffusion-long', type=float, default=None,
                   help='GT2 longitudinal diffusion in cm²/μs (default: same as GT1)')
    p.add_argument('--gt2-label', default='GT2', help='Label for GT2 (default: GT2)')

    p.add_argument('--output-dir',
                   default=os.path.join(_PLOTS_DIR, 'diffusion_comparison'),
                   help='Output directory (default: plots/diffusion_comparison)')
    return p.parse_args()


def parse_tracks(tracks_str):
    specs = []
    for item in tracks_str.split('+'):
        item = item.strip()
        if ':' in item:
            parts = item.split(':')
            name      = parts[0]
            direction = tuple(float(x) for x in parts[1].split(','))
            momentum_mev = float(parts[2])
        elif item in TRACK_PRESETS:
            direction, momentum_mev = TRACK_PRESETS[item]
            name = item
        else:
            raise ValueError(f'Unknown track {item!r}. Presets: {list(TRACK_PRESETS)}')
        specs.append({'name': name, 'direction': direction, 'momentum_mev': momentum_mev})
    return specs


def build_simulator():
    detector_config = generate_detector(CONFIG_PATH)
    return DetectorSimulator(
        detector_config,
        differentiable=True,
        n_segments=N_SEGMENTS,
        use_bucketed=True,
        max_active_buckets=MAX_ACTIVE_BUCKETS,
        include_noise=False,
        include_electronics=False,
        include_track_hits=False,
        include_digitize=False,
        track_config=None,
    )


def make_params(simulator, diffusion_trans=None, diffusion_long=None):
    """Return sim params with velocity/lifetime at GT and optional diffusion override."""
    params = simulator.default_sim_params._replace(
        lifetime_us    = jnp.array(GT_LIFETIME_US),
        velocity_cm_us = jnp.array(GT_VELOCITY_CM_US),
    )
    if diffusion_trans is not None:
        params = params._replace(diffusion_trans_cm2_us=jnp.array(diffusion_trans))
    if diffusion_long is not None:
        params = params._replace(diffusion_long_cm2_us=jnp.array(diffusion_long))
    return params


def get_arrays(simulator, track_spec, sim_params):
    track = generate_muon_track(
        start_position_mm=(0.0, 0.0, 0.0),
        direction=track_spec['direction'],
        kinetic_energy_mev=track_spec['momentum_mev'],
        step_size_mm=0.1,
        track_id=1,
    )
    track    = filter_track_inside_volumes(track, simulator.config.volumes)
    deposits = build_deposit_data(
        track['position'], track['de'], track['dx'], simulator.config,
        theta=track['theta'], phi=track['phi'],
        track_ids=track['track_id'],
    )
    arrays = simulator.forward(sim_params, deposits)
    jax.block_until_ready(arrays)
    return [np.asarray(a) for a in arrays]


def make_comparison_figure(gt1_arrays, gt2_arrays, col_labels, title, time_step_us,
                           gt1_label, gt2_label):
    """
    3-row × n_planes figure:
      row 0 — GT1
      row 1 — GT2
      row 2 — GT2 − GT1
    """
    n_planes = len(gt1_arrays)
    fig, axes = plt.subplots(3, n_planes, figsize=(3.5 * n_planes, 10),
                             constrained_layout=True)
    if n_planes == 1:
        axes = axes.reshape(3, 1)

    fig.suptitle(title, fontsize=10, y=1.01)

    # Shared symmetric colour scale for GT1 / GT2
    vmax_gt = max(
        np.nanpercentile(np.abs(np.concatenate(
            [a.ravel() for a in gt1_arrays + gt2_arrays])), 99),
        1e-9,
    )
    norm_gt = mcolors.Normalize(vmin=-vmax_gt, vmax=vmax_gt)

    # Difference colour scale
    diffs = [gt2_arrays[i] - gt1_arrays[i] for i in range(n_planes)]
    vmax_diff = max(
        np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in diffs])), 99),
        1e-9,
    )
    norm_diff = mcolors.Normalize(vmin=-vmax_diff, vmax=vmax_diff)

    rows = [
        (gt1_label,                  gt1_arrays, norm_gt,   'RdBu_r', 'signal (e⁻)'),
        (gt2_label,                  gt2_arrays, norm_gt,   'RdBu_r', 'signal (e⁻)'),
        (f'Δ  ({gt2_label}−{gt1_label})', diffs, norm_diff, 'RdBu_r', 'Δ signal (e⁻)'),
    ]

    for row_idx, (row_label, arrays, norm, cmap, cbar_label) in enumerate(rows):
        for col_idx, (arr, col_label) in enumerate(zip(arrays, col_labels)):
            ax = axes[row_idx, col_idx]
            n_wires, n_time = arr.shape
            t_max_us = n_time * time_step_us
            im = ax.imshow(
                arr, aspect='auto', origin='lower', norm=norm, cmap=cmap,
                extent=[0, t_max_us, 0, n_wires],
            )
            fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)
            ax.set_xlabel('time (μs)', fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f'{row_label}\nwire index', fontsize=8)
            else:
                ax.set_ylabel('wire index', fontsize=8)
            if row_idx == 0:
                ax.set_title(col_label, fontsize=9, fontweight='bold')

    return fig


def _signal_mid(arr, axis):
    """Mid index (along `axis`) of the range of rows/cols that carry signal."""
    profile = np.abs(arr).sum(axis=1 - axis)
    threshold = profile.max() * 0.01
    indices = np.where(profile > threshold)[0]
    if len(indices) == 0:
        return arr.shape[axis] // 2
    return int(indices[len(indices) // 2])


def make_profiles_figure(gt1_arrays, gt2_arrays, col_labels, title, time_step_us,
                         gt1_label, gt2_label):
    """
    2-row × n_planes figure of 1-D profiles:
      row 0 — signal vs wire index at mid-signal time step
      row 1 — signal vs time at mid-signal wire
    Each panel shows GT1, GT2, and GT2−GT1.
    """
    n_planes = len(gt1_arrays)
    fig, axes = plt.subplots(2, n_planes, figsize=(3.5 * n_planes, 6),
                             constrained_layout=True)
    if n_planes == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(title, fontsize=9, y=1.01)

    c1, c2, cd = '#1f77b4', '#d62728', '#2ca02c'  # blue, red, green

    for col_idx, (a1, a2, col_label) in enumerate(zip(gt1_arrays, gt2_arrays, col_labels)):
        diff = a2 - a1
        n_wires, n_time = a1.shape
        t_axis = np.arange(n_time) * time_step_us

        # ── row 0: signal vs wire at mid time ────────────────────────────────
        # Use GT1 to find the mid-signal time (representative of where signal lives)
        t_mid = _signal_mid(a1, axis=1)   # axis=1 → profile over time → mid time
        ax = axes[0, col_idx]
        wire_idx = np.arange(n_wires)
        ax.plot(wire_idx, a1[:, t_mid], color=c1, lw=1.2, label=gt1_label)
        ax.plot(wire_idx, a2[:, t_mid], color=c2, lw=1.2, ls='--', label=gt2_label)
        ax.plot(wire_idx, diff[:, t_mid], color=cd, lw=1.0, ls=':', label=f'Δ')
        ax.axhline(0, color='k', lw=0.4, ls='-')
        ax.set_xlabel('wire index', fontsize=8)
        ax.set_ylabel('signal (e⁻)', fontsize=8)
        ax.set_title(f'{col_label}  [t={t_axis[t_mid]:.2f} μs]', fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=7)
        if col_idx == 0:
            ax.legend(fontsize=7, loc='upper right')

        # ── row 1: signal vs time at mid wire ────────────────────────────────
        w_mid = _signal_mid(a1, axis=0)   # axis=0 → profile over wires → mid wire
        ax = axes[1, col_idx]
        ax.plot(t_axis, a1[w_mid, :], color=c1, lw=1.2, label=gt1_label)
        ax.plot(t_axis, a2[w_mid, :], color=c2, lw=1.2, ls='--', label=gt2_label)
        ax.plot(t_axis, diff[w_mid, :], color=cd, lw=1.0, ls=':', label=f'Δ')
        ax.axhline(0, color='k', lw=0.4, ls='-')
        ax.set_xlabel('time (μs)', fontsize=8)
        ax.set_ylabel('signal (e⁻)', fontsize=8)
        ax.set_title(f'{col_label}  [wire {w_mid}]', fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=7)
        if col_idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    return fig


def main():
    args = parse_args()
    track_specs = parse_tracks(args.tracks)

    print(f'JAX devices: {jax.devices()}')
    print('Building simulator...')
    simulator = build_simulator()

    cfg          = simulator.config
    n_volumes    = cfg.n_volumes
    n_planes     = cfg.volumes[0].n_planes
    time_step_us = float(cfg.time_step_us)

    # Build flat column labels across all volumes: U1 V1 Y1 U2 V2 Y2
    col_labels = [
        f'{cfg.plane_names[v][p]}{v + 1}'
        for v in range(n_volumes)
        for p in range(n_planes)
    ]

    print('Warming up JIT...')
    t0 = time.time()
    simulator.warm_up()
    print(f'Done ({time.time() - t0:.1f} s)')

    # Build sim params
    params1 = make_params(simulator, args.gt1_diffusion_trans, args.gt1_diffusion_long)
    params2 = make_params(simulator, args.gt2_diffusion_trans, args.gt2_diffusion_long)

    # Resolve actual values for labels / filenames
    dt1 = float(params1.diffusion_trans_cm2_us)
    dl1 = float(params1.diffusion_long_cm2_us)
    dt2 = float(params2.diffusion_trans_cm2_us)
    dl2 = float(params2.diffusion_long_cm2_us)

    print(f'\n{args.gt1_label}:  trans={dt1:.3g}  long={dl1:.3g} cm²/μs')
    print(f'{args.gt2_label}:  trans={dt2:.3g}  long={dl2:.3g} cm²/μs')

    os.makedirs(args.output_dir, exist_ok=True)

    for ts in track_specs:
        name = ts['name']
        print(f'\nTrack: {name}  dir={ts["direction"]}  T={ts["momentum_mev"]} MeV')

        arrays1 = get_arrays(simulator, ts, params1)
        arrays2 = get_arrays(simulator, ts, params2)

        # Flatten to one list per plane across all volumes
        planes1 = [arrays1[v * n_planes + p] for v in range(n_volumes) for p in range(n_planes)]
        planes2 = [arrays2[v * n_planes + p] for v in range(n_volumes) for p in range(n_planes)]

        title = (
            f'Diffusion comparison  —  {name}  dir={ts["direction"]}  T={ts["momentum_mev"]:.0f} MeV\n'
            f'{args.gt1_label}: trans={dt1:.3g}  long={dl1:.3g} cm²/μs    '
            f'{args.gt2_label}: trans={dt2:.3g}  long={dl2:.3g} cm²/μs'
        )

        out_path = os.path.join(args.output_dir, f'diffusion_comparison_{name}.pdf')
        with PdfPages(out_path) as pdf:
            fig = make_comparison_figure(
                planes1, planes2, col_labels, title,
                time_step_us, args.gt1_label, args.gt2_label,
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        print(f'  Saved: {out_path}')

        profiles_path = os.path.join(args.output_dir, f'diffusion_profiles_{name}.pdf')
        with PdfPages(profiles_path) as pdf:
            fig = make_profiles_figure(
                planes1, planes2, col_labels, title,
                time_step_us, args.gt1_label, args.gt2_label,
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        print(f'  Saved: {profiles_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
