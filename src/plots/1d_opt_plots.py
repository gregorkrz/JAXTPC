#!/usr/bin/env python
"""
Plot optimization trajectories produced by 1d_opt.py.

For each (track, optimizer, loss) combination one PDF is created with:
  rows  — one per optimisation parameter
  col 0 — loss vs step
  col 1 — normalized parameter value vs step  (p_n, GT = 1.0)

Each starting point is drawn with:
  - M thin, semi-transparent lines (one per trial)
  - one thick line showing the mean across trials

Lines are coloured by starting factor using a diverging colormap
(blue = below GT, red = above GT, centred at GT = 1.0).

Usage
-----
    python 1d_opt_plots.py
    python 1d_opt_plots.py --N 3 --M 3 --optimizer adam
    python 1d_opt_plots.py --track-name diagonal,X --optimizer adam,sgd
    python 1d_opt_plots.py --results-dir results/1d_opt --output-dir plots/1d_opt
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import glob
import os
import pickle

_RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')
_PLOTS_DIR   = os.environ.get('PLOTS_DIR',   'plots')

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
    'sobolev_loss':                  'Sobolev',
    'sobolev_loss_geomean_log1p':    'Sobolev geomean log1p',
    'mse_loss':                      'MSE',
}

LOSS_COLORS = {
    'sobolev_loss':                 'steelblue',
    'sobolev_loss_geomean_log1p':   'darkorange',
    'mse_loss':                     '#2ca02c',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--N', type=int, default=3,
                   help='N used when running 1d_opt.py (default: 3)')
    p.add_argument('--M', type=int, default=3,
                   help='M used when running 1d_opt.py (default: 3)')
    p.add_argument('--optimizer', default=None,
                   help='Comma-separated optimizer(s) to plot; '
                        'default: all found in results dir')
    p.add_argument('--track-name', default='diagonal',
                   help='Comma-separated track name(s) to plot (default: diagonal)')
    p.add_argument('--results-dir', default=os.path.join(_RESULTS_DIR, '1d_opt'),
                   help='Directory containing pkl files (default: results/1d_opt)')
    p.add_argument('--recursive', action='store_true',
                   help='Also scan subdirectories of --results-dir (e.g. lr_*/) '
                        'for pkl files when building the convergence histogram')
    p.add_argument('--output-dir', default=None,
                   help='Where to save PDFs (default: same as --results-dir)')
    return p.parse_args()


def _iter_pkls(results_dir, recursive=False):
    """Yield all .pkl paths under results_dir, optionally recursing into subdirs."""
    yield from sorted(glob.glob(os.path.join(results_dir, '*.pkl')))
    if recursive:
        yield from sorted(glob.glob(os.path.join(results_dir, '**', '*.pkl'),
                                    recursive=True))


def load_results(results_dir, N, M, track_names, optimizers):
    """Return data[track][optimizer][loss][param] = result dict."""
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
        if result.get('M') != M:
            continue
        if result.get('track_name') not in track_names:
            continue
        if optimizers is not None and result.get('optimizer') not in optimizers:
            continue
        tn  = result['track_name']
        opt = result['optimizer']
        ln  = result['loss_name']
        pn  = result['param_name']
        data.setdefault(tn, {}).setdefault(opt, {}).setdefault(ln, {})[pn] = result
        loaded += 1

    print(f'Loaded {loaded} result(s) from {results_dir!r}')
    return data


def _starting_colormap(factors):
    """Return a list of RGBA colours, one per factor, diverging around GT=1."""
    cmap = cm.RdYlBu
    f_min, f_max = min(factors), max(factors)
    # centre the normalisation so that 1.0 maps exactly to the midpoint
    half = max(abs(f_min - 1.0), abs(f_max - 1.0)) + 1e-9
    norm = mcolors.Normalize(vmin=1.0 - half, vmax=1.0 + half)
    return [cmap(norm(f)) for f in factors]


def _pad_traj(traj, full_len):
    arr = np.array(traj)
    if len(arr) < full_len:
        arr = np.concatenate([arr, np.full(full_len - len(arr), arr[-1])])
    return arr


def _aggregate_convergence(result):
    """Return mean/std of normalised loss and relative param error over all starts × trials.

    normalised loss:   loss(t) / loss(0)           — loss-scale-independent
    relative p_n error: |p_n(t) - p_n_gt| / |p_n_gt|  — param-space convergence
    Both arrays have shape (max_steps+1,).
    """
    full_len = result['max_steps'] + 1
    p_n_gt   = result.get('p_n_gt', 1.0)

    norm_loss_list = []
    param_err_list = []

    for point_trials in result['trials']:
        for trial in point_trials:
            loss_arr  = _pad_traj(trial['loss_trajectory'],  full_len)
            param_arr = _pad_traj(trial['param_trajectory'], full_len)

            l0 = loss_arr[0]
            if l0 > 0:
                norm_loss_list.append(loss_arr / l0)

            param_err_list.append(np.abs(param_arr - p_n_gt) / (abs(p_n_gt) + 1e-30))

    nl  = np.array(norm_loss_list)
    pe  = np.array(param_err_list)
    return nl.mean(0), nl.std(0), pe.mean(0), pe.std(0)


def make_loss_comparison_figure(track_name, optimizer, loss_data, output_dir, N, M):
    """Create a PDF comparing convergence speed across losses.

    One row per parameter, two columns:
      col 0 — normalised loss  loss(t)/loss(0)  averaged over all starts × trials
      col 1 — relative p_n error  |p_n(t)-p_n_gt|/|p_n_gt|  averaged likewise

    Shaded band = ±1 std across starts × trials.
    """
    all_params = sorted({p for pd in loss_data.values() for p in pd})
    if not all_params:
        return

    n_rows = len(all_params)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 4.5 * n_rows), squeeze=False)

    first = next(iter(next(iter(loss_data.values())).values()))
    lr           = first.get('lr', '?')
    max_steps    = first.get('max_steps', '?')
    direction    = first.get('direction', '?')
    momentum_mev = first.get('momentum_mev', '?')

    fig.suptitle(
        f'1-D optimisation — loss comparison  |  {optimizer}  lr={lr}  |  '
        f'track: {track_name}  dir={direction}  T={momentum_mev} MeV  |  '
        f'N={N}  M={M}  steps={max_steps}',
        fontsize=10, y=1.01,
    )

    steps = np.arange(max_steps + 1)

    for row, param_name in enumerate(all_params):
        ax_nl = axes[row, 0]
        ax_pe = axes[row, 1]

        for loss_name, param_data in sorted(loss_data.items()):
            if param_name not in param_data:
                continue
            result = param_data[param_name]
            color  = LOSS_COLORS.get(loss_name, 'gray')
            label  = LOSS_LABELS.get(loss_name, loss_name)

            nl_mean, nl_std, pe_mean, pe_std = _aggregate_convergence(result)
            eps = 1e-12

            ax_nl.plot(steps, nl_mean, color=color, lw=2.0, label=label, zorder=2)
            ax_nl.fill_between(steps,
                               np.maximum(nl_mean - nl_std, eps),
                               nl_mean + nl_std,
                               color=color, alpha=0.15, zorder=1)

            ax_pe.plot(steps, pe_mean, color=color, lw=2.0, label=label, zorder=2)
            ax_pe.fill_between(steps,
                               np.maximum(pe_mean - pe_std, eps),
                               pe_mean + pe_std,
                               color=color, alpha=0.15, zorder=1)

        plabel = PARAM_LABELS.get(param_name, param_name)

        ax_nl.set_yscale('log')
        ax_nl.set_xlabel('step')
        ax_nl.set_ylabel('loss(t) / loss(0)  (log)')
        ax_nl.set_title(f'{param_name}  —  normalised loss')
        ax_nl.legend(fontsize=8)
        ax_nl.grid(True, which='both', alpha=0.25)

        ax_pe.set_yscale('log')
        ax_pe.set_xlabel('step')
        ax_pe.set_ylabel('|p_n − p_n_gt| / |p_n_gt|  (log)')
        ax_pe.set_title(f'{param_name}  —  {plabel}  relative error')
        ax_pe.legend(fontsize=8)
        ax_pe.grid(True, which='both', alpha=0.25)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fname = f'1d_opt_N{N}_M{M}_{optimizer}_loss_comparison_{track_name}.pdf'
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def make_figure(track_name, optimizer, loss_name, param_data, output_dir, N, M):
    """Create and save a PDF for one (track, optimizer, loss) combination."""
    params = sorted(param_data.keys())
    n_rows = len(params)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(13, 4.5 * n_rows),
        squeeze=False,
    )

    first = next(iter(param_data.values()))
    direction    = first.get('direction', '?')
    momentum_mev = first.get('momentum_mev', '?')
    lr           = first.get('lr', '?')
    max_steps    = first.get('max_steps', '?')
    loss_label   = LOSS_LABELS.get(loss_name, loss_name)

    fig.suptitle(
        f'1-D optimisation  |  {optimizer}  lr={lr}  |  '
        f'track: {track_name}  dir={direction}  T={momentum_mev} MeV  |  '
        f'loss: {loss_label}  |  N={N}  M={M}  steps={max_steps}',
        fontsize=10, y=1.01,
    )

    for row, param_name in enumerate(params):
        result  = param_data[param_name]
        ax_loss = axes[row, 0]
        ax_pval = axes[row, 1]

        factors    = result['starting_factors']    # length 2N+1
        trials     = result['trials']              # [2N+1][ M ]{ dict }
        colors     = _starting_colormap(factors)
        max_steps  = result['max_steps']

        for fi, (factor, point_trials, color) in enumerate(
                zip(factors, trials, colors)):
            label = f'p_n={factor:.4f}' if fi == 0 or fi == len(factors) - 1 else None

            loss_lists  = [t['loss_trajectory']  for t in point_trials]
            param_lists = [t['param_trajectory'] for t in point_trials]

            # thin lines — each has its own (possibly shorter) step axis
            for traj in loss_lists:
                ax_loss.plot(np.arange(len(traj)), traj, color=color,
                             lw=0.7, alpha=0.35, zorder=1)
            for traj in param_lists:
                ax_pval.plot(np.arange(len(traj)), traj, color=color,
                             lw=0.7, alpha=0.35, zorder=1)

            # mean across trials — pad shorter trajectories with their last value
            full_len   = max_steps + 1
            loss_mean  = np.mean([_pad_traj(t, full_len) for t in loss_lists],  axis=0)
            param_mean = np.mean([_pad_traj(t, full_len) for t in param_lists], axis=0)
            steps      = np.arange(full_len)

            ax_loss.plot(steps, loss_mean,  color=color, lw=2.0, alpha=0.9,
                         label=label, zorder=2)
            ax_pval.plot(steps, param_mean, color=color, lw=2.0, alpha=0.9,
                         label=label, zorder=2)

        # GT reference line (p_n_gt may differ from 1.0 when using TYPICAL_SCALES)
        p_n_gt = result.get('p_n_gt', 1.0)
        ax_pval.axhline(p_n_gt, color='black', ls='--', lw=1.0,
                        alpha=0.6, label=f'GT  (p_n = {p_n_gt:.4g})')

        # ── loss axis ──────────────────────────────────────────────────────
        ax_loss.set_yscale('log')
        ax_loss.set_xlabel('step')
        ax_loss.set_ylabel(f'{loss_label}  (log scale)')
        ax_loss.set_title(f'{param_name}  —  loss vs step')
        ax_loss.legend(fontsize=7, ncol=2)
        ax_loss.grid(True, which='both', alpha=0.25)

        # ── param value axis ───────────────────────────────────────────────
        plabel = PARAM_LABELS.get(param_name, param_name)
        ax_pval.set_xlabel('step')
        ax_pval.set_ylabel('p_n')
        ax_pval.set_title(f'{param_name}  —  {plabel} vs step')
        ax_pval.legend(fontsize=7, ncol=2)
        ax_pval.grid(True, alpha=0.25)

        # per-subplot colorbar
        factors = result['starting_factors']
        half    = max(abs(min(factors) - 1.0), abs(max(factors) - 1.0)) + 1e-9
        sm = cm.ScalarMappable(
            cmap=cm.RdYlBu,
            norm=mcolors.Normalize(vmin=1.0 - half, vmax=1.0 + half),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_pval, pad=0.02)
        cbar.set_label('starting factor', fontsize=8)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fname = f'1d_opt_N{N}_M{M}_{optimizer}_{loss_name}_{track_name}.pdf'
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def load_results_flat(results_dir, N, M, track_names, optimizers, recursive=False):
    """Return a flat list of result dicts (any loss, any LR) for use in histograms."""
    results = []
    seen    = set()
    for path in _iter_pkls(results_dir, recursive=recursive):
        with open(path, 'rb') as f:
            r = pickle.load(f)
        if r.get('N') != N or r.get('M') != M:
            continue
        if r.get('track_name') not in track_names:
            continue
        if optimizers is not None and r.get('optimizer') not in optimizers:
            continue
        if path not in seen:
            seen.add(path)
            results.append(r)
    return results


def make_convergence_histogram_figure(track_name, loss_name, flat_results,
                                      output_dir, N, M):
    """Histogram of steps-to-convergence grouped by optimizer and LR.

    Rows = params, cols = optimizers.
    Each subplot overlays one histogram per LR (coloured by LR).
    Trials that hit max_steps without early-stopping are included at
    max_steps and rendered with hatching so they stand out.
    Only skips if stopped_early metadata is missing (old pkl files).
    """
    # ── collect data ──────────────────────────────────────────────────────────
    groups = {}   # (param, optimizer, lr) -> list of (steps_run, stopped_early)
    for r in flat_results:
        if r.get('track_name') != track_name:
            continue
        if r.get('loss_name') != loss_name:
            continue
        key = (r['param_name'], r['optimizer'], float(r['lr']))
        entries = groups.setdefault(key, [])
        for point_trials in r['trials']:
            for trial in point_trials:
                if 'steps_run' not in trial:
                    continue
                entries.append((trial['steps_run'], trial.get('stopped_early', False),
                                 r['max_steps']))

    if not groups:
        print(f'  No convergence data for track={track_name} loss={loss_name}')
        return

    params     = sorted({k[0] for k in groups})
    optimizers = sorted({k[1] for k in groups})
    lrs        = sorted({k[2] for k in groups})

    # colour each LR from a sequential palette
    lr_cmap   = plt.cm.viridis
    lr_norm   = mcolors.LogNorm(vmin=min(lrs), vmax=max(lrs)) if len(lrs) > 1 \
                else mcolors.Normalize(vmin=0, vmax=1)
    lr_color  = {lr: lr_cmap(lr_norm(lr) if len(lrs) > 1 else 0.5) for lr in lrs}

    n_rows = len(params)
    n_cols = len(optimizers)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4.0 * n_rows),
                             squeeze=False)

    fig.suptitle(
        f'Steps to convergence  |  track: {track_name}  '
        f'loss: {LOSS_LABELS.get(loss_name, loss_name)}  |  N={N}  M={M}\n'
        f'(hatched = did not converge, hit max_steps)',
        fontsize=10, y=1.02,
    )

    for row, param in enumerate(params):
        for col, optimizer in enumerate(optimizers):
            ax = axes[row, col]
            max_s_seen = 100   # fallback

            for lr in lrs:
                key = (param, optimizer, lr)
                if key not in groups or not groups[key]:
                    continue
                entries  = groups[key]
                max_s    = entries[0][2]
                max_s_seen = max_s
                steps_conv   = [e[0] for e in entries if     e[1]]
                steps_noconv = [e[0] for e in entries if not e[1]]
                color = lr_color[lr]
                bins  = np.linspace(0, max_s, min(max_s // 5, 40) + 1)

                if steps_conv:
                    ax.hist(steps_conv, bins=bins, color=color,
                            alpha=0.55, label=f'lr={lr}',
                            edgecolor='white', linewidth=0.4)
                if steps_noconv:
                    ax.hist(steps_noconv, bins=bins, color=color,
                            alpha=0.55, hatch='//', edgecolor=color,
                            linewidth=0.4, label=f'lr={lr} (no conv)')

            ax.set_title(f'{param}  |  {optimizer}', fontsize=9)
            ax.set_xlabel('steps run')
            ax.set_ylabel('trial count')
            ax.set_xlim(0, max_s_seen)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, axis='y', alpha=0.25)

    # colour bar for LR
    if len(lrs) > 1:
        sm = plt.cm.ScalarMappable(cmap=lr_cmap, norm=lr_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.02)
        cbar.set_label('learning rate', fontsize=9)
        cbar.set_ticks(lrs)
        cbar.set_ticklabels([str(lr) for lr in lrs])

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fname    = f'1d_opt_N{N}_M{M}_convergence_hist_{loss_name}_{track_name}.pdf'
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

    print(f'N           : {args.N}')
    print(f'M           : {args.M}')
    print(f'Optimizers  : {optimizers or "all"}')
    print(f'Track names : {track_names}')
    print(f'Results dir : {args.results_dir}')
    print(f'Recursive   : {args.recursive}')
    print(f'Output dir  : {output_dir}')

    data = load_results(
        args.results_dir, args.N, args.M, track_names, optimizers
    )
    flat = load_results_flat(
        args.results_dir, args.N, args.M, track_names, optimizers,
        recursive=args.recursive,
    )
    print(f'Flat results for histogram: {len(flat)}')

    if not data and not flat:
        print('No matching results found.')
        return

    # ── per-loss trajectory plots ─────────────────────────────────────────────
    for track_name in track_names:
        if track_name not in data:
            print(f'Warning: no trajectory data for track {track_name!r}')
            continue
        for optimizer, loss_data in sorted(data[track_name].items()):
            for loss_name, param_data in sorted(loss_data.items()):
                print(f'\nPlotting  track={track_name}  '
                      f'optimizer={optimizer}  loss={loss_name}  '
                      f'({len(param_data)} params)')
                make_figure(
                    track_name, optimizer, loss_name,
                    param_data, output_dir, args.N, args.M,
                )

            if len(loss_data) > 1:
                print(f'\nLoss comparison  track={track_name}  '
                      f'optimizer={optimizer}  ({len(loss_data)} losses)')
                make_loss_comparison_figure(
                    track_name, optimizer, loss_data, output_dir, args.N, args.M,
                )

    # ── convergence histograms (one per track × loss, across all optimizers/LRs)
    if flat:
        loss_names_found = sorted({r['loss_name'] for r in flat})
        for track_name in track_names:
            for loss_name in loss_names_found:
                print(f'\nConvergence histogram  track={track_name}  loss={loss_name}')
                make_convergence_histogram_figure(
                    track_name, loss_name, flat, output_dir, args.N, args.M,
                )

    print('\nDone.')


if __name__ == '__main__':
    main()
