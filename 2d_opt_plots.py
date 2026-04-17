#!/usr/bin/env python
"""
Plot optimization trajectories produced by 2d_opt.py.

For each (pair × loss) pickle one PDF is created with a 2×2 layout:
  top-left  — 2-D phase portrait: trajectories in (p_n₁, p_n₂) space
  top-right — loss vs step  (log scale, one line per trial)
  bot-left  — param1 p_n vs step
  bot-right — param2 p_n vs step

Each of the N random trials is drawn in a distinct colour.

Usage
-----
    python 2d_opt_plots.py
    python 2d_opt_plots.py --N 5 --optimizer adam
    python 2d_opt_plots.py --results-dir results/2d_opt --output-dir plots/2d_opt
"""

import argparse
import glob
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


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
    p.add_argument('--N', type=int, default=None,
                   help='N used when running 2d_opt.py (default: all found)')
    p.add_argument('--optimizer', default=None,
                   help='Comma-separated optimizer(s) to plot; default: all found')
    p.add_argument('--track-name', default='diagonal',
                   help='Comma-separated track name(s) to plot (default: diagonal)')
    p.add_argument('--results-dir', default='results/2d_opt',
                   help='Directory containing pkl files (default: results/2d_opt)')
    p.add_argument('--output-dir', default=None,
                   help='Where to save PDFs (default: same as --results-dir)')
    return p.parse_args()


def load_results(results_dir, N, track_names, optimizers):
    pkl_paths = sorted(glob.glob(os.path.join(results_dir, '*.pkl')))
    if not pkl_paths:
        raise FileNotFoundError(f'No .pkl files found in {results_dir!r}')

    results = []
    for path in pkl_paths:
        with open(path, 'rb') as f:
            r = pickle.load(f)
        if N is not None and r.get('N') != N:
            continue
        if r.get('track_name') not in track_names:
            continue
        if optimizers is not None and r.get('optimizer') not in optimizers:
            continue
        results.append(r)

    print(f'Loaded {len(results)} result(s) from {results_dir!r}')
    return results


def _trial_colors(n):
    cmap = cm.tab10 if n <= 10 else cm.tab20
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def make_figure(result, output_dir):
    param1, param2 = result['param_names']
    p_n_gt1, p_n_gt2 = result['p_n_gts']
    trials     = result['trials']
    N          = result['N']
    optimizer  = result['optimizer']
    lr         = result['lr']
    max_steps  = result['max_steps']
    loss_name  = result['loss_name']
    track_name = result['track_name']
    direction  = result.get('direction', '?')
    mom_mev    = result.get('momentum_mev', '?')

    loss_label = LOSS_LABELS.get(loss_name, loss_name)
    p1_label   = PARAM_LABELS.get(param1, param1)
    p2_label   = PARAM_LABELS.get(param2, param2)

    colors = _trial_colors(N)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_phase, ax_loss, ax_p1, ax_p2 = (axes[0, 0], axes[0, 1],
                                        axes[1, 0], axes[1, 1])

    fig.suptitle(
        f'2-D optimisation  |  {param1} + {param2}  |  '
        f'{optimizer}  lr={lr}  |  '
        f'loss: {loss_label}  |  track: {track_name}  '
        f'dir={direction}  T={mom_mev} MeV  |  N={N}  steps={max_steps}',
        fontsize=9, y=1.01,
    )

    for i, trial in enumerate(trials):
        color = colors[i]
        traj  = np.array(trial['param_trajectory'])  # (steps+1, 2)
        loss  = np.array(trial['loss_trajectory'])
        steps = np.arange(len(loss))

        # ── phase portrait ───────────────────────────────────────────────────
        ax_phase.plot(traj[:, 0], traj[:, 1], color=color, lw=0.9, alpha=0.7)
        ax_phase.plot(traj[0, 0],  traj[0, 1],  'o', color=color, ms=5, zorder=3)
        ax_phase.annotate('', xy=(traj[-1, 0], traj[-1, 1]),
                          xytext=(traj[-2, 0], traj[-2, 1]),
                          arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

        # ── loss vs step ─────────────────────────────────────────────────────
        ax_loss.plot(steps, loss, color=color, lw=1.0, alpha=0.8)

        # ── param vs step ────────────────────────────────────────────────────
        ax_p1.plot(steps, traj[:, 0], color=color, lw=1.0, alpha=0.8)
        ax_p2.plot(steps, traj[:, 1], color=color, lw=1.0, alpha=0.8)

    # GT marker on phase portrait
    ax_phase.plot(p_n_gt1, p_n_gt2, '*', color='black', ms=14, zorder=5,
                  label=f'GT  ({p_n_gt1:.3g}, {p_n_gt2:.3g})')
    ax_phase.set_xlabel(f'{param1}  (p_n)', fontsize=9)
    ax_phase.set_ylabel(f'{param2}  (p_n)', fontsize=9)
    ax_phase.set_title('phase portrait  (p_n space)', fontsize=9)
    ax_phase.legend(fontsize=8)
    ax_phase.grid(True, alpha=0.25)

    # GT reference lines on param plots
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

    # colour bar for trial index
    sm = cm.ScalarMappable(
        cmap=cm.tab10 if N <= 10 else cm.tab20,
        norm=mcolors.Normalize(vmin=0, vmax=max(N - 1, 1)),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.02)
    cbar.set_label('trial index', fontsize=9)
    cbar.set_ticks(np.arange(N))

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    pair_tag = f'{param1}+{param2}'
    lr_schedule = result.get('lr_schedule', 'constant')
    sched_tag   = '_cosine' if lr_schedule == 'cosine' else ''
    fname    = (f'2d_opt_N{N}_{optimizer}_lr{lr}{sched_tag}_{loss_name}_{pair_tag}_{track_name}.pdf')
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def main():
    args = parse_args()

    track_names = [t.strip() for t in args.track_name.split(',')]
    optimizers  = ([o.strip() for o in args.optimizer.split(',')]
                   if args.optimizer else None)
    output_dir  = args.output_dir or args.results_dir

    print(f'N           : {args.N or "all"}')
    print(f'Optimizers  : {optimizers or "all"}')
    print(f'Track names : {track_names}')
    print(f'Results dir : {args.results_dir}')
    print(f'Output dir  : {output_dir}')

    results = load_results(args.results_dir, args.N, track_names, optimizers)
    if not results:
        print('No matching results found.')
        return

    for r in results:
        p1, p2 = r['param_names']
        print(f'\nPlotting  pair={p1}+{p2}  loss={r["loss_name"]}  N={r["N"]}')
        make_figure(r, output_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
