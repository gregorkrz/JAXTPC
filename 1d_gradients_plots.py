#!/usr/bin/env python
"""
Plot 1-D gradient and loss sweeps produced by 1d_gradients.py.

For each track direction one figure is created with:
  rows  — one per optimisation parameter (e.g. velocity_cm_us, lifetime_us)
  col 0 — signed gradient vs parameter value  (one curve per loss)
  col 1 — loss vs parameter value, log scale  (one curve per loss)

Usage
-----
    python 1d_gradients_plots.py
    python 1d_gradients_plots.py --N 5
    python 1d_gradients_plots.py --track-name diagonal,along_x
    python 1d_gradients_plots.py --results-dir results/1d_gradients --output-dir plots
"""

import argparse
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


PARAM_LABELS = {
    'velocity_cm_us': 'drift velocity  (cm/μs)',
    'lifetime_us':    'electron lifetime  (μs)',
}

LOSS_STYLES = {
    'sobolev_loss': dict(
        color='steelblue', ls='-', label='sobolev_loss',
    ),
    'sobolev_loss_geomean_log1p': dict(
        color='darkorange', ls='--', label='geomean_log1p',
    ),
}

_FALLBACK_COLORS = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--N', type=int, default=2,
                   help='N used when running 1d_gradients.py (default: 2)')
    p.add_argument('--results-dir', default='results/1d_gradients',
                   help='Directory containing pkl files '
                        '(default: results/1d_gradients)')
    p.add_argument('--track-name', default='diagonal',
                   help='Comma-separated track name(s) to plot '
                        '(default: diagonal)')
    p.add_argument('--output-dir', default=None,
                   help='Where to save PDFs (default: same as --results-dir)')
    return p.parse_args()


def load_results(results_dir, N, track_names):
    """Return  data[track_name][param_name][loss_name] = result dict."""
    data = {}
    pkl_paths = sorted(glob.glob(os.path.join(results_dir, '*.pkl')))
    if not pkl_paths:
        raise FileNotFoundError(f'No .pkl files found in {results_dir!r}')

    loaded = 0
    for path in pkl_paths:
        with open(path, 'rb') as f:
            result = pickle.load(f)
        if result.get('N') != N:
            continue
        if result.get('track_name') not in track_names:
            continue
        tn = result['track_name']
        pn = result['param_name']
        ln = result['loss_name']
        data.setdefault(tn, {}).setdefault(pn, {})[ln] = result
        loaded += 1

    print(f'Loaded {loaded} result(s) from {results_dir!r}')
    return data


def _loss_style(loss_name, fallback_idx=0):
    if loss_name in LOSS_STYLES:
        return LOSS_STYLES[loss_name]
    color = _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]
    return dict(color=color, ls='-.', label=loss_name)


def make_figure(track_name, param_data, output_dir, N):
    """Create and save the grid figure for one track direction."""
    params = sorted(param_data.keys())
    n_rows = len(params)
    if n_rows == 0:
        print(f'  No data for track {track_name!r}, skipping.')
        return

    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(17, 4.5 * n_rows),
        squeeze=False,
    )

    # Retrieve track metadata from any result
    first_result = next(
        iter(next(iter(param_data.values())).values())
    )
    direction    = first_result.get('direction', '?')
    momentum_mev = first_result.get('momentum_mev', '?')

    fig.suptitle(
        f'1-D sweep  |  track: {track_name}  '
        f'direction={direction}  T={momentum_mev} MeV  '
        f'N={N}',
        fontsize=12, y=1.01,
    )

    for row, param_name in enumerate(params):
        loss_data   = param_data[param_name]
        ax_grad     = axes[row, 0]
        ax_grad_abs = axes[row, 1]
        ax_loss     = axes[row, 2]

        param_gt = None
        for fi, (loss_name, result) in enumerate(sorted(loss_data.items())):
            xvals  = np.array(result['param_values'])
            grads  = np.array(result['grad_values'])
            losses = np.array(result['loss_values'])
            param_gt = result['param_gt']

            style = _loss_style(loss_name, fi)

            # Col 0: signed gradient (line + uniform marker)
            ax_grad.plot(xvals, grads,
                         color=style['color'], ls=style['ls'],
                         marker='o', markersize=5, lw=1.8,
                         label=style['label'])

            # Col 1: log|gradient|, marker encodes sign (▲ pos, ▼ neg)
            abs_grads = np.abs(grads)
            pos_mask  = grads >= 0
            neg_mask  = ~pos_mask
            # connecting line through all points
            ax_grad_abs.plot(xvals, abs_grads,
                             color=style['color'], ls=style['ls'],
                             lw=1.8, zorder=1)
            # positive-gradient points
            if pos_mask.any():
                ax_grad_abs.scatter(xvals[pos_mask], abs_grads[pos_mask],
                                    marker='^', s=55, color=style['color'],
                                    zorder=2,
                                    label=f'{style["label"]} (+)')
            # negative-gradient points
            if neg_mask.any():
                ax_grad_abs.scatter(xvals[neg_mask], abs_grads[neg_mask],
                                    marker='v', s=55, color=style['color'],
                                    zorder=2, facecolors='none',
                                    linewidths=1.5,
                                    label=f'{style["label"]} (−)')

            # Col 2: loss (log scale)
            ax_loss.plot(xvals, losses,
                         color=style['color'], ls=style['ls'],
                         marker='o', markersize=5, lw=1.8,
                         label=style['label'])

        if param_gt is not None:
            for ax in (ax_grad, ax_grad_abs, ax_loss):
                ax.axvline(param_gt, color='black', ls=':', lw=1.2,
                           alpha=0.7, label='GT')

        ax_grad.axhline(0, color='grey', ls='--', lw=0.8, alpha=0.6)

        xlabel = PARAM_LABELS.get(param_name, param_name)

        ax_grad.set_xlabel(xlabel)
        ax_grad.set_ylabel('∂L / ∂(param / GT)')
        ax_grad.set_title(f'{param_name}  —  gradient')
        ax_grad.legend(fontsize=8)
        ax_grad.grid(True, alpha=0.3)

        ax_grad_abs.set_xlabel(xlabel)
        ax_grad_abs.set_ylabel('|∂L / ∂(param / GT)|  (log scale)')
        ax_grad_abs.set_yscale('log')
        ax_grad_abs.set_title(f'{param_name}  —  |gradient|  (▲ +, ▽ −)')
        ax_grad_abs.legend(fontsize=8)
        ax_grad_abs.grid(True, which='both', alpha=0.3)

        ax_loss.set_xlabel(xlabel)
        ax_loss.set_ylabel('loss  (log scale)')
        ax_loss.set_yscale('log')
        ax_loss.set_title(f'{param_name}  —  loss')
        ax_loss.legend(fontsize=8)
        ax_loss.grid(True, which='both', alpha=0.3)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'1d_gradients_N{N}_{track_name}.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    args = parse_args()

    track_names = [t.strip() for t in args.track_name.split(',')]
    output_dir  = args.output_dir or args.results_dir

    print(f'N           : {args.N}')
    print(f'Results dir : {args.results_dir}')
    print(f'Track names : {track_names}')
    print(f'Output dir  : {output_dir}')

    data = load_results(args.results_dir, args.N, track_names)

    if not data:
        print(f'No matching results found (N={args.N}, tracks={track_names}).')
        return

    for track_name in track_names:
        if track_name not in data:
            print(f'Warning: no data found for track {track_name!r}')
            continue
        print(f'\nPlotting track: {track_name}')
        make_figure(track_name, data[track_name], output_dir, args.N)

    print('\nDone.')


if __name__ == '__main__':
    main()
