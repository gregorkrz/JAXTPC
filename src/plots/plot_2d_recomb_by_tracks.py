#!/usr/bin/env python
"""
Produce per-track-combination PDFs from 2-D recomb optimisation results.

Output layout:
    <output-dir>/<tracks-key>/No_Noise.pdf
    <output-dir>/<tracks-key>/Noise.pdf

Each PDF is multi-page when multiple runs share the same track combination and
noise level (e.g. constant vs cosine lr schedule variants).

Usage
-----
    python src/plots/plot_2d_recomb_by_tracks.py
    python src/plots/plot_2d_recomb_by_tracks.py --results-dir results/opt/2D_recomb --output-dir plots/2D_recomb
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
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

_RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')
_PLOTS_DIR   = os.environ.get('PLOTS_DIR',   'plots')

PARAM_LABELS = {
    'velocity_cm_us':         'drift velocity  (cm/μs)',
    'lifetime_us':            'electron lifetime  (μs)',
    'diffusion_trans_cm2_us': 'transverse diffusion  (cm²/μs)',
    'diffusion_long_cm2_us':  'longitudinal diffusion  (cm²/μs)',
    'recomb_alpha':           'recombination α',
    'recomb_beta':            'recombination β',
    'recomb_beta_90':         'recombination β₉₀',
    'recomb_R':               'recombination R',
}

LOSS_LABELS = {
    'sobolev_loss':               'Sobolev',
    'sobolev_loss_geomean_log1p': 'Sobolev geomean log1p',
    'mse_loss':                   'MSE',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', default=os.path.join(_RESULTS_DIR, 'opt', '2D_recomb'),
                   help='Root directory containing per-run subdirs with result_*.pkl files')
    p.add_argument('--output-dir', default=None,
                   help='Where to write track-combo subdirs with PDFs '
                        '(default: <results-dir>/../plots/2D_recomb)')
    return p.parse_args()


def _trial_colors(n):
    cmap = cm.tab10 if n <= 10 else cm.tab20
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _tracks_key(result):
    return '+'.join(t['name'] for t in result['tracks'])


def make_figure(result):
    param1, param2 = result['param_names']
    p_n_gt1, p_n_gt2 = result['p_n_gts']
    trials     = result['trials']
    N          = result['N']
    optimizer  = result['optimizer']
    lr         = result['lr']
    lr_schedule = result.get('lr_schedule', 'constant')
    max_steps  = result['max_steps']
    loss_name  = result['loss_name']
    noise_scale = result.get('noise_scale', 0.0)
    _tracks    = result.get('tracks') or [{}]

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
    ax_phase, ax_loss, ax_p1, ax_p2 = (axes[0, 0], axes[0, 1],
                                        axes[1, 0], axes[1, 1])
    fig.suptitle(title, fontsize=9, y=1.01)

    for i, trial in enumerate(trials):
        color = colors[i]
        traj  = np.array(trial['param_trajectory'])
        loss  = np.array(trial['loss_trajectory'])
        steps = np.arange(len(loss))

        ax_phase.plot(traj[:, 0], traj[:, 1], color=color, lw=0.9, alpha=0.7)
        ax_phase.plot(traj[0, 0],  traj[0, 1],  'o', color=color, ms=5, zorder=3)
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


def main():
    args = parse_args()
    results_dir = args.results_dir
    output_dir  = args.output_dir or os.path.join(
        os.path.dirname(results_dir), '..', 'plots', '2D_recomb'
    )
    output_dir = os.path.normpath(output_dir)

    pkl_paths = sorted(glob.glob(os.path.join(results_dir, '*', 'result_*.pkl')))
    if not pkl_paths:
        raise FileNotFoundError(f'No result_*.pkl files found under {results_dir!r}')

    print(f'Found {len(pkl_paths)} pkl file(s) in {results_dir!r}')

    # group by (tracks_key, is_noisy)
    groups = defaultdict(list)
    for path in pkl_paths:
        r = pickle.load(open(path, 'rb'))
        key = (_tracks_key(r), r.get('noise_scale', 0.0) > 0)
        groups[key].append(r)

    for (tracks_key, is_noisy), results in sorted(groups.items()):
        fname      = 'Noise.pdf' if is_noisy else 'No_Noise.pdf'
        folder     = os.path.join(output_dir, tracks_key)
        out_path   = os.path.join(folder, fname)
        os.makedirs(folder, exist_ok=True)

        # sort pages: constant lr first, then cosine
        results_sorted = sorted(results, key=lambda r: (r.get('lr_schedule','constant') == 'cosine'))

        with PdfPages(out_path) as pdf:
            for r in results_sorted:
                fig = make_figure(r)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        noise_label = 'noisy' if is_noisy else 'no noise'
        print(f'  {tracks_key:40s}  {noise_label:8s}  ({len(results)} page(s))  →  {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
