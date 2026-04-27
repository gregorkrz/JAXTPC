#!/usr/bin/env python
"""
Produce per-track-combination PDFs from 2-D recomb optimisation results
and combined loss landscape pkl files.

Output layout per track-combination folder:
    <output-dir>/<tracks-key>/No_Noise.pdf      — opt trajectories, no noise
    <output-dir>/<tracks-key>/Noise.pdf         — opt trajectories, noise=1
    <output-dir>/<tracks-key>/Loss_Landscape.pdf — landscape heatmaps (page 1: no noise, page 2: noisy)

Usage
-----
    python src/plots/plot_2d_landscape_by_tracks.py
    python src/plots/plot_2d_landscape_by_tracks.py \\
        --opt-results-dir $RESULTS_DIR/opt/2D_recomb \\
        --landscape-dir   $RESULTS_DIR/2d_landscape/combined_landscapes \\
        --output-dir      $PLOTS_DIR/opt/2D_recomb_sorted_noise
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import glob
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

_RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')
_PLOTS_DIR   = os.environ.get('PLOTS_DIR',   'plots')

PARAM_LABELS = {
    'recomb_alpha':   'recombination α',
    'recomb_beta_90': 'recombination β₉₀',
}

LOSS_LABELS = {
    'sobolev_loss':               'Sobolev',
    'sobolev_loss_geomean_log1p': 'Sobolev geomean log1p',
    'mse_loss':                   'MSE',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--opt-results-dir',
                   default=os.path.join(_RESULTS_DIR, 'opt', '2D_recomb'),
                   help='Root dir containing per-run subdirs with result_*.pkl files')
    p.add_argument('--landscape-dir', nargs='+',
                   default=[os.path.join(_RESULTS_DIR, '2d_landscape', 'combined_landscapes'),
                            os.path.join(_RESULTS_DIR, '2d_landscape')],
                   help='One or more dirs containing landscape pkl files (combined or single-track)')
    p.add_argument('--output-dir',
                   default=os.path.join(_PLOTS_DIR, 'opt', '2D_recomb_sorted_noise'),
                   help='Where to write track-combo subdirs with PDFs')
    return p.parse_args()


# ── Opt trajectory plots (No_Noise.pdf / Noise.pdf) ───────────────────────────

def _trial_colors(n):
    cmap = cm.tab10 if n <= 10 else cm.tab20
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _opt_tracks_key(result):
    return '+'.join(t['name'] for t in result['tracks'])


def make_opt_figure(result):
    param1, param2 = result['param_names']
    p_n_gt1, p_n_gt2 = result['p_n_gts']
    trials      = result['trials']
    N           = result['N']
    optimizer   = result['optimizer']
    lr          = result['lr']
    lr_schedule = result.get('lr_schedule', 'constant')
    max_steps   = result['max_steps']
    loss_name   = result['loss_name']
    noise_scale = result.get('noise_scale', 0.0)
    _tracks     = result.get('tracks') or [{}]

    loss_label = LOSS_LABELS.get(loss_name, loss_name)
    p1_label   = PARAM_LABELS.get(param1, param1)
    p2_label   = PARAM_LABELS.get(param2, param2)
    colors     = _trial_colors(N)

    tracks_lines = '  |  '.join(
        f'{t.get("name","?")}  dir={t.get("direction","?")}  T={t.get("momentum_mev","?")} MeV'
        for t in _tracks
    )
    noise_tag = f'  noise={noise_scale}' if noise_scale else ''
    title = (
        f'2-D optimisation  |  {param1} + {param2}  |  '
        f'{optimizer}  lr={lr}  sched={lr_schedule}  |  loss: {loss_label}  |  N={N}  steps={max_steps}{noise_tag}\n'
        f'tracks:  {tracks_lines}'
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_phase, ax_loss, ax_p1, ax_p2 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    fig.suptitle(title, fontsize=9, y=1.01)

    for i, trial in enumerate(trials):
        color = colors[i]
        traj  = np.array(trial['param_trajectory'])
        loss  = np.array(trial['loss_trajectory'])
        steps = np.arange(len(loss))

        ax_phase.plot(traj[:, 0], traj[:, 1], color=color, lw=0.9, alpha=0.7)
        ax_phase.plot(traj[0, 0], traj[0, 1], 'o', color=color, ms=5, zorder=3)
        if len(traj) >= 2:
            ax_phase.annotate('', xy=(traj[-1, 0], traj[-1, 1]),
                              xytext=(traj[-2, 0], traj[-2, 1]),
                              arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
        ax_loss.plot(steps, loss, color=color, lw=1.0, alpha=0.8)
        ax_p1.plot(steps, traj[:, 0], color=color, lw=1.0, alpha=0.8)
        ax_p2.plot(steps, traj[:, 1], color=color, lw=1.0, alpha=0.8)

    ax_phase.plot(p_n_gt1, p_n_gt2, '*', color='black', ms=14, zorder=5,
                  label=f'GT  ({p_n_gt1:.3g}, {p_n_gt2:.3g})')
    ax_phase.set_xlabel(f'{param1}  (p_n)', fontsize=9)
    ax_phase.set_ylabel(f'{param2}  (p_n)', fontsize=9)
    ax_phase.set_title('phase portrait  (p_n space)', fontsize=9)
    ax_phase.legend(fontsize=8)
    ax_phase.grid(True, alpha=0.25)

    ax_p1.axhline(p_n_gt1, color='black', ls='--', lw=1.0, alpha=0.6,
                  label=f'GT  p_n={p_n_gt1:.4g}')
    ax_p2.axhline(p_n_gt2, color='black', ls='--', lw=1.0, alpha=0.6,
                  label=f'GT  p_n={p_n_gt2:.4g}')

    ax_loss.set_yscale('log')
    ax_loss.set_xlabel('step')
    ax_loss.set_ylabel(f'{loss_label}  (log scale)')
    ax_loss.set_title('loss vs step')
    ax_loss.grid(True, which='both', alpha=0.25)

    ax_p1.set_xlabel('step')
    ax_p1.set_ylabel('p_n')
    ax_p1.set_title(f'{param1}  —  {p1_label}')
    ax_p1.legend(fontsize=8)
    ax_p1.grid(True, alpha=0.25)

    ax_p2.set_xlabel('step')
    ax_p2.set_ylabel('p_n')
    ax_p2.set_title(f'{param2}  —  {p2_label}')
    ax_p2.legend(fontsize=8)
    ax_p2.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


# ── Loss landscape figure ──────────────────────────────────────────────────────

def make_landscape_figure(landscape):
    loss_name   = landscape['loss_name']
    track_name  = landscape['track_name']
    alpha_vals  = np.array(landscape['alpha_vals'])
    beta90_vals = np.array(landscape['beta90_vals'])
    grid        = np.array(landscape['grid'])
    gt_alpha    = landscape['gt_alpha']
    gt_beta90   = landscape['gt_beta90']
    direction   = landscape['direction']
    grid_size   = landscape['grid_size']
    range_frac  = landscape['range_frac']
    noise_scale = landscape.get('noise_scale', 0.0)

    loss_label  = LOSS_LABELS.get(loss_name, loss_name)
    noise_str   = f'  noise={noise_scale}' if noise_scale else '  no noise'

    # log-scale norm
    vmax    = np.nanpercentile(grid, 98)
    pos_min = grid[grid > 0].min() if np.any(grid > 0) else 1e-10
    norm    = mcolors.LogNorm(vmin=max(pos_min, vmax * 1e-6), vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.pcolormesh(beta90_vals, alpha_vals, grid,
                       norm=norm, cmap='viridis', shading='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'{loss_label}  (log scale)', fontsize=9)

    try:
        ax.contour(beta90_vals, alpha_vals, grid, levels=12,
                   colors='white', linewidths=0.5, alpha=0.4, norm=norm)
    except Exception:
        pass

    if 'grad_alpha' in landscape and 'grad_beta90' in landscape:
        ga = np.array(landscape['grad_alpha'])
        gb = np.array(landscape['grad_beta90'])
        span_a = alpha_vals[-1]  - alpha_vals[0]  or 1.0
        span_b = beta90_vals[-1] - beta90_vals[0] or 1.0
        u = -gb / span_b
        v = -ga / span_a
        ax.streamplot(beta90_vals, alpha_vals, u, v,
                      color='white', linewidth=0.8, arrowsize=1.2,
                      density=1.2, minlength=0.05, zorder=3)

    ax.plot(gt_beta90, gt_alpha, '*', color='red', ms=14, zorder=5,
            label=f'GT  (α={gt_alpha:.4g}, β₉₀={gt_beta90:.4g})')

    ax.set_xlabel('recomb β₉₀', fontsize=10)
    ax.set_ylabel('recomb α', fontsize=10)
    if isinstance(direction, list):
        dir_str = ' | '.join(str(d) for d in direction)
    else:
        dir_str = str(direction)
    ax.set_title(
        f'Loss landscape  |  {loss_label}  |  tracks: {track_name}{noise_str}\n'
        f'dir={dir_str}  |  {grid_size}×{grid_size} grid  ±{range_frac*100:.0f}% around GT  [log scale]',
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = os.path.normpath(args.output_dir)

    # ── Load opt results ───────────────────────────────────────────────────────
    opt_groups = defaultdict(list)
    opt_pkls = sorted(glob.glob(os.path.join(args.opt_results_dir, '*', 'result_*.pkl')))
    for path in opt_pkls:
        r = pickle.load(open(path, 'rb'))
        key = (_opt_tracks_key(r), r.get('noise_scale', 0.0) > 0)
        opt_groups[key].append(r)
    print(f'Opt results: {len(opt_pkls)} pkl(s) in {args.opt_results_dir!r}')

    # ── Load landscape pkls ────────────────────────────────────────────────────
    landscape_groups = defaultdict(list)
    seen_paths = set()
    total_land = 0
    for land_dir in args.landscape_dir:
        for path in sorted(glob.glob(os.path.join(land_dir, '*.pkl'))):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            ld = pickle.load(open(path, 'rb'))
            key = (ld['track_name'], ld.get('noise_scale', 0.0) > 0)
            landscape_groups[key].append(ld)
            total_land += 1
    print(f'Landscape pkls: {total_land} pkl(s) across {len(args.landscape_dir)} dir(s)')

    # ── All track combos seen in either source ─────────────────────────────────
    all_tracks_keys = {k for k, _ in list(opt_groups) + list(landscape_groups)}

    for tracks_key in sorted(all_tracks_keys):
        folder = os.path.join(output_dir, tracks_key)
        os.makedirs(folder, exist_ok=True)

        # No_Noise.pdf
        no_noise_results = sorted(opt_groups.get((tracks_key, False), []),
                                  key=lambda r: (r.get('lr_schedule', 'constant') == 'cosine'))
        if no_noise_results:
            out_path = os.path.join(folder, 'No_Noise.pdf')
            with PdfPages(out_path) as pdf:
                for r in no_noise_results:
                    fig = make_opt_figure(r)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            print(f'  {tracks_key:45s}  no noise  ({len(no_noise_results)}p)  →  No_Noise.pdf')

        # Noise.pdf
        noise_results = sorted(opt_groups.get((tracks_key, True), []),
                               key=lambda r: (r.get('lr_schedule', 'constant') == 'cosine'))
        if noise_results:
            out_path = os.path.join(folder, 'Noise.pdf')
            with PdfPages(out_path) as pdf:
                for r in noise_results:
                    fig = make_opt_figure(r)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            print(f'  {tracks_key:45s}  noisy     ({len(noise_results)}p)  →  Noise.pdf')

        # Loss_Landscape.pdf — page 1: no noise, page 2: noisy
        land_no_noise = landscape_groups.get((tracks_key, False), [])
        land_noisy    = landscape_groups.get((tracks_key, True),  [])
        if land_no_noise or land_noisy:
            out_path = os.path.join(folder, 'Loss_Landscape.pdf')
            pages = land_no_noise + land_noisy   # no-noise first
            with PdfPages(out_path) as pdf:
                for ld in pages:
                    fig = make_landscape_figure(ld)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            tag = f'{len(land_no_noise)} no-noise + {len(land_noisy)} noisy'
            print(f'  {tracks_key:45s}  landscape ({tag})  →  Loss_Landscape.pdf')

    print('\nDone.')


if __name__ == '__main__':
    main()
