#!/usr/bin/env python
"""
Plot loss and parameter errors from a single run_optimization.py result pkl.

Saves two PDFs next to the pkl:
  loss_vs_step.pdf   — loss on log scale vs optimisation step
  param_errors.pdf   — relative error |p_n − GT| / |GT| vs step, one subplot per param

Usage
-----
    python src/plots/plot_result.py results/my_run/result_0.pkl
    python src/plots/plot_result.py              # uses the most recently modified pkl in results/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import glob
import pickle

import matplotlib.pyplot as plt
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
    'sobolev_loss':               'Sobolev',
    'sobolev_loss_geomean_log1p': 'Sobolev geomean log1p',
    'mse_loss':                   'MSE',
    'l1_loss':                    'L1',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pkl', nargs='?', default=None,
                   help='Path to result_*.pkl (default: most recently modified pkl in results/)')
    return p.parse_args()


def find_latest_pkl():
    pkls = glob.glob('results/**/*.pkl', recursive=True) + glob.glob('results/*.pkl')
    if not pkls:
        raise FileNotFoundError('No result_*.pkl found under results/')
    return max(pkls, key=os.path.getmtime)


def _pad(traj, full_len):
    arr = np.array(traj)
    if len(arr) < full_len:
        arr = np.concatenate([arr, np.full(full_len - len(arr), arr[-1])])
    return arr


def plot_loss(r, out_path):
    trials     = r['trials']
    max_steps  = r['max_steps']
    full_len   = max_steps + 1
    loss_label = LOSS_LABELS.get(r['loss_name'], r['loss_name'])
    track_lbl  = ' + '.join(
        f'{t["name"]}({int(t["momentum_mev"])} MeV)' for t in r['tracks'])

    fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')

    cmap = plt.cm.plasma
    for i, trial in enumerate(trials):
        color = cmap(i / max(len(trials) - 1, 1))
        loss  = _pad(trial['loss_trajectory'], full_len)
        steps = np.arange(len(loss))
        label = f'trial {i}' + (' [early]' if trial.get('stopped_early') else '')
        ax.plot(steps, loss, color=color, lw=1.5, alpha=0.85, label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Step (log scale)', fontsize=12)
    ax.set_ylabel(f'{loss_label}  (log scale)', fontsize=12)
    ax.set_title(
        f'Loss vs step  |  {r["optimizer"]}  lr={r["lr"]}  N={len(trials)}\n'
        f'tracks: {track_lbl}',
        fontsize=11,
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_param_errors(r, out_path):
    trials      = r['trials']
    param_names = r['param_names']
    p_n_gts     = r['p_n_gts']
    scales      = r['scales']
    max_steps   = r['max_steps']
    full_len    = max_steps + 1
    n_params    = len(param_names)
    loss_label  = LOSS_LABELS.get(r['loss_name'], r['loss_name'])
    track_lbl   = ' + '.join(
        f'{t["name"]}({int(t["momentum_mev"])} MeV)' for t in r['tracks'])

    ncols = min(n_params, 3)
    nrows = (n_params + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4.5 * nrows),
                             facecolor='white', squeeze=False)
    # hide unused axes
    for idx in range(n_params, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(
        f'Parameter relative error vs step  |  {r["optimizer"]}  lr={r["lr"]}  N={len(trials)}\n'
        f'tracks: {track_lbl}  |  loss: {loss_label}',
        fontsize=11, y=1.01,
    )

    cmap = plt.cm.plasma
    steps = np.arange(full_len)

    for pi, (pname, p_n_gt) in enumerate(zip(param_names, p_n_gts)):
        ax    = axes[pi // ncols, pi % ncols]
        plabel = PARAM_LABELS.get(pname, pname)
        gt_phys = p_n_gt * scales[pi]

        for i, trial in enumerate(trials):
            color = cmap(i / max(len(trials) - 1, 1))
            traj  = np.array([_pad([step[pi] for step in trial['param_trajectory']], full_len)])
            traj  = traj[0]
            rel_err = np.abs(traj - p_n_gt) / (abs(p_n_gt) + 1e-30)
            label = f'trial {i}' + (' [early]' if trial.get('stopped_early') else '')
            ax.plot(steps[:len(rel_err)], rel_err,
                    color=color, lw=1.5, alpha=0.85, label=label)

        ax.axhline(0.01, color='green', ls=':', lw=1.2, label='1% error')
        ax.axhline(0.05, color='orange', ls=':', lw=1.0, label='5% error')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Step (log scale)', fontsize=10)
        ax.set_ylabel('|p_n − GT| / |GT|', fontsize=10)
        ax.set_title(f'{plabel}\nGT = {gt_phys:.4g}', fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, which='both', alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    args = parse_args()
    pkl_path = args.pkl or find_latest_pkl()
    print(f'Loading: {pkl_path}')

    with open(pkl_path, 'rb') as f:
        r = pickle.load(f)

    out_dir = os.path.dirname(os.path.abspath(pkl_path))
    plot_loss(r, os.path.join(out_dir, 'loss_vs_step.pdf'))
    plot_param_errors(r, os.path.join(out_dir, 'param_errors.pdf'))
    print('Done.')


if __name__ == '__main__':
    main()
