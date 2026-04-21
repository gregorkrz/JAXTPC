#!/usr/bin/env python
"""
Plot optimization trajectories produced by run_optimization.py.

For each result pkl one PDF is created with:
  rows  — one per optimised parameter
  col 0 — loss vs step
  col 1 — normalised parameter value vs step  (GT = p_n_gt)

A second PDF shows the 2-D endpoint scatter (only for 2-parameter results).

A summary PDF compares convergence across all loaded track configurations.

Usage
-----
    python src/plots/opt_plots.py
    python src/plots/opt_plots.py --results-dir results/opt --output-dir plots/opt
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import glob
import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


PARAM_LABELS = {
    'recomb_alpha':           'recombination α',
    'recomb_beta_90':         'recombination β₉₀',
    'recomb_beta':            'recombination β',
    'recomb_R':               'recombination R',
    'velocity_cm_us':         'drift velocity  (cm/μs)',
    'lifetime_us':            'electron lifetime  (μs)',
    'diffusion_trans_cm2_us': 'transverse diffusion  (cm²/μs)',
    'diffusion_long_cm2_us':  'longitudinal diffusion  (cm²/μs)',
}

LOSS_LABELS = {
    'sobolev_loss':                 'Sobolev',
    'sobolev_loss_geomean_log1p':   'Sobolev geomean log1p',
    'mse_loss':                     'MSE',
    'l1_loss':                      'L1',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', default='results/opt',
                   help='Directory (or tree) containing result_*.pkl files')
    p.add_argument('--output-dir', default='plots/opt',
                   help='Where to save PDFs')
    return p.parse_args()


def load_results(results_dir):
    pattern = os.path.join(results_dir, '**', 'result_*.pkl')
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        pattern2 = os.path.join(results_dir, 'result_*.pkl')
        paths = sorted(glob.glob(pattern2))
    results = []
    for path in paths:
        with open(path, 'rb') as f:
            r = pickle.load(f)
        r['_path'] = path
        results.append(r)
    print(f'Loaded {len(results)} result(s) from {results_dir!r}')
    return results


def _track_label(tracks):
    parts = []
    for t in tracks:
        name = t['name']
        mev  = int(t['momentum_mev'])
        parts.append(f'{name}({mev} MeV)')
    return ' + '.join(parts)


def _pad(traj, full_len):
    arr = np.array(traj)
    if len(arr) < full_len:
        arr = np.concatenate([arr, np.full(full_len - len(arr), arr[-1])])
    return arr


def _starting_colors(n):
    cmap = cm.plasma
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def make_trajectory_figure(r, output_dir):
    """One PDF per result: rows=params, col0=loss, col1=param trajectory."""
    param_names = r['param_names']
    p_n_gts     = r['p_n_gts']
    trials      = r['trials']
    max_steps   = r['max_steps']
    full_len    = max_steps + 1
    N           = len(trials)

    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 2,
                             figsize=(13, 4.5 * n_params),
                             squeeze=False)

    track_lbl  = _track_label(r['tracks'])
    loss_label = LOSS_LABELS.get(r['loss_name'], r['loss_name'])
    noise_str  = f'  noise_scale={r["noise_scale"]}' if r.get('noise_scale', 0) > 0 else ''
    fig.suptitle(
        f'Optimization  |  {r["optimizer"]}  lr={r["lr"]}  steps={max_steps}  N={N}\n'
        f'tracks: {track_lbl}  |  loss: {loss_label}{noise_str}',
        fontsize=10, y=1.01,
    )

    colors = _starting_colors(N)

    for i_trial, (trial, color) in enumerate(zip(trials, colors)):
        loss_traj  = _pad(trial['loss_trajectory'],  full_len)
        param_traj = np.array([_pad([step[pi] for step in trial['param_trajectory']], full_len)
                                for pi in range(n_params)])

        steps = np.arange(len(loss_traj))
        stopped = trial.get('stopped_early', False)
        lw = 1.5
        alpha = 0.85
        label = f'trial {i_trial}' + (' ✓' if stopped else '')

        for row in range(n_params):
            axes[row, 0].plot(steps, loss_traj, color=color, lw=lw, alpha=alpha,
                              label=label if row == 0 else None)
            axes[row, 1].plot(steps, param_traj[row], color=color, lw=lw, alpha=alpha)

    for row, (pname, p_n_gt) in enumerate(zip(param_names, p_n_gts)):
        ax_l = axes[row, 0]
        ax_p = axes[row, 1]
        plabel = PARAM_LABELS.get(pname, pname)

        ax_l.set_yscale('log')
        ax_l.set_xlabel('step')
        ax_l.set_ylabel(f'{loss_label}  (log)')
        ax_l.set_title(f'loss vs step')
        ax_l.grid(True, which='both', alpha=0.25)
        if row == 0:
            ax_l.legend(fontsize=7, ncol=2)

        ax_p.axhline(p_n_gt, color='black', ls='--', lw=1.2,
                     label=f'GT = {p_n_gt:.4g}')
        ax_p.set_xlabel('step')
        ax_p.set_ylabel('p_n')
        ax_p.set_title(f'{plabel}  —  value vs step')
        ax_p.legend(fontsize=8)
        ax_p.grid(True, alpha=0.25)

    fig.tight_layout()

    seed = r.get('seed', 'noseed')
    track_tag = '+'.join(t['name'] for t in r['tracks'])
    fname = f'opt_traj__{track_tag}__{r["loss_name"]}__seed{seed}.pdf'
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def make_2d_figure(r, output_dir):
    """2×2 phase portrait for 2-param results (reuses 2d_opt_plots layout)."""
    if len(r['param_names']) != 2:
        return

    param1, param2   = r['param_names']
    p_n_gt1, p_n_gt2 = r['p_n_gts']
    trials = r['trials']
    N      = len(trials)

    loss_label = LOSS_LABELS.get(r['loss_name'], r['loss_name'])
    p1_label   = PARAM_LABELS.get(param1, param1)
    p2_label   = PARAM_LABELS.get(param2, param2)
    track_lbl  = _track_label(r['tracks'])

    colors = _starting_colors(N)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_phase, ax_loss, ax_p1, ax_p2 = (axes[0, 0], axes[0, 1],
                                        axes[1, 0], axes[1, 1])

    fig.suptitle(
        f'2-D optimisation  |  {param1} + {param2}  |  '
        f'{r["optimizer"]}  lr={r["lr"]}  |  loss: {loss_label}  |  '
        f'tracks: {track_lbl}  |  N={N}  steps={r["max_steps"]}',
        fontsize=9, y=1.01,
    )

    for i, (trial, color) in enumerate(zip(trials, colors)):
        traj  = np.array(trial['param_trajectory'])  # (steps+1, 2)
        loss  = np.array(trial['loss_trajectory'])
        steps = np.arange(len(loss))

        ax_phase.plot(traj[:, 0], traj[:, 1], color=color, lw=0.9, alpha=0.7)
        ax_phase.plot(traj[0, 0], traj[0, 1], 'o', color=color, ms=5, zorder=3)
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

    ax_p1.set_xlabel('step'); ax_p1.set_ylabel('p_n')
    ax_p1.set_title(f'{param1}  —  {p1_label}')
    ax_p1.legend(fontsize=8); ax_p1.grid(True, alpha=0.25)

    ax_p2.set_xlabel('step'); ax_p2.set_ylabel('p_n')
    ax_p2.set_title(f'{param2}  —  {p2_label}')
    ax_p2.legend(fontsize=8); ax_p2.grid(True, alpha=0.25)

    fig.tight_layout()

    seed      = r.get('seed', 'noseed')
    track_tag = '+'.join(t['name'] for t in r['tracks'])
    fname     = f'opt_2d__{track_tag}__{r["loss_name"]}__seed{seed}.pdf'
    out_path  = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def make_summary_figure(results, output_dir):
    """Summary: final relative error per parameter, one column per track config."""
    if not results:
        return

    param_names = results[0]['param_names']
    n_params    = len(param_names)
    n_configs   = len(results)

    fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 5), squeeze=False)

    loss_label = LOSS_LABELS.get(results[0]['loss_name'], results[0]['loss_name'])
    fig.suptitle(
        f'Final relative error  |  {results[0]["optimizer"]}  lr={results[0]["lr"]}\n'
        f'loss: {loss_label}',
        fontsize=11,
    )

    config_labels = [_track_label(r['tracks']) for r in results]
    x = np.arange(n_configs)

    for pi, pname in enumerate(param_names):
        ax = axes[0, pi]
        plabel = PARAM_LABELS.get(pname, pname)

        for xi, r in enumerate(results):
            p_n_gt = r['p_n_gts'][pi]
            finals = [np.array(t['param_trajectory'][-1])[pi] for t in r['trials']]
            rel_errs = [abs(f - p_n_gt) / (abs(p_n_gt) + 1e-30) for f in finals]
            jitter = np.random.default_rng(xi).uniform(-0.15, 0.15, len(rel_errs))
            ax.scatter(np.full(len(rel_errs), xi) + jitter, rel_errs,
                       s=60, alpha=0.7, zorder=3)
            ax.scatter([xi], [np.median(rel_errs)], marker='D', s=100,
                       color='black', zorder=4)

        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('|final p_n − GT| / |GT|  (log)')
        ax.set_title(plabel)
        ax.grid(True, which='both', axis='y', alpha=0.25)
        ax.axhline(0.01, color='green', ls=':', lw=1, label='1% error')
        ax.legend(fontsize=8)

    fig.tight_layout()
    fname = f'opt_summary__{results[0]["loss_name"]}.pdf'
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def make_loss_curves_comparison(results, output_dir):
    """Overlay loss curves from all track configs on one plot."""
    if not results:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    max_steps = max(r['max_steps'] for r in results)
    full_len  = max_steps + 1

    colors = cm.tab10(np.linspace(0, 1, len(results)))
    for r, color in zip(results, colors):
        label = _track_label(r['tracks'])
        all_loss = []
        for trial in r['trials']:
            all_loss.append(_pad(trial['loss_trajectory'], full_len))
        arr = np.array(all_loss)
        mean_l = arr.mean(0)
        std_l  = arr.std(0)
        steps  = np.arange(full_len)
        ax.plot(steps, mean_l, color=color, lw=2, label=label)
        ax.fill_between(steps,
                        np.maximum(mean_l - std_l, 1e-30),
                        mean_l + std_l,
                        color=color, alpha=0.15)

    loss_label = LOSS_LABELS.get(results[0]['loss_name'], results[0]['loss_name'])
    ax.set_yscale('log')
    ax.set_xlabel('step')
    ax.set_ylabel(f'{loss_label}  (log)')
    ax.set_title(f'Loss curves — all track configs  |  {results[0]["optimizer"]}  lr={results[0]["lr"]}')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()

    fname = f'opt_loss_curves__{results[0]["loss_name"]}.pdf'
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = load_results(args.results_dir)
    if not results:
        print('No results found.')
        return

    for r in results:
        make_trajectory_figure(r, args.output_dir)
        make_2d_figure(r, args.output_dir)

    # Group by param_names + loss + optimizer for summary / loss-curve plots
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = (tuple(r['param_names']), r['loss_name'], r['optimizer'])
        groups[key].append(r)

    for (pnames, loss_name, opt), group in groups.items():
        if len(group) > 1:
            make_summary_figure(group, args.output_dir)
            make_loss_curves_comparison(group, args.output_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
