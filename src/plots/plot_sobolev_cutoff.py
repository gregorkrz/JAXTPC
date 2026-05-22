#!/usr/bin/env python
"""
Compare Sobolev loss landscapes across ADC cutoff values, clean and noisy GT.

Loads pkl files produced by 1d_gradients.py --adc-cutoff ... --noise-scale ...,
groups them by sweep parameter and track, and saves:

  sobolev_cutoff_<param>_<track>.pdf
      Per-track: 2-row grid (top = clean GT, bottom = noisy GT), 3 columns:
        • Loss vs. factor          (log y)
        • |Gradient| vs. factor   (log y)
        • Signed gradient vs. factor
      Colour encodes ADC cutoff (viridis).

  sobolev_cutoff_<param>_combined.pdf
      Same layout but loss/gradient summed across all tracks.

  sobolev_cutoff_grad_overlay.pdf
      One panel per sweep parameter, |gradient| curves for all cutoffs;
      solid = clean GT, dashed = noisy GT; one subplot row per track + combined.

Usage
-----
    python src/plots/plot_sobolev_cutoff.py
    python src/plots/plot_sobolev_cutoff.py \\
        --results-dir results/1d_gradients/sobolev_cutoff_diffusion_N20_range75pct \\
        --output-dir  plots/sobolev_cutoff_diffusion_N20_range75pct
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

_PLOTS_DIR   = os.environ.get('PLOTS_DIR',   'plots')
_RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')

DEFAULT_RESULTS = os.path.join(
    _RESULTS_DIR, '1d_gradients', 'sobolev_cutoff_diffusion_N20_range75pct'
)

PARAM_LABELS = {
    'diffusion_trans_cm2_us': 'D⊥ (cm²/μs)',
    'diffusion_long_cm2_us':  'D∥ (cm²/μs)',
    'velocity_cm_us':         'v (cm/μs)',
    'lifetime_us':            'τ (μs)',
    'recomb_alpha':           'α',
    'recomb_beta_90':         'β₉₀',
    'recomb_R':               'R',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', default=DEFAULT_RESULTS,
                   help='Directory with pkl files (default: %(default)s)')
    p.add_argument('--output-dir', default=None,
                   help='Where to save PDFs (default: plots/<results-dir-name>)')
    return p.parse_args()


def _track_name_from_run(r):
    specs = r.get('track_specs', [])
    if len(specs) == 1:
        return specs[0]['name']
    if len(specs) > 1:
        return f'{len(specs)}tracks'
    return 'unknown'


def load_pkls(results_dir):
    paths = sorted(glob.glob(os.path.join(results_dir, '*.pkl')))
    if not paths:
        raise FileNotFoundError(f'No .pkl files in {results_dir!r}')
    results = []
    for path in paths:
        with open(path, 'rb') as f:
            r = pickle.load(f)
        r['_track_name'] = _track_name_from_run(r)
        results.append(r)
        print(f'  {os.path.basename(path)}'
              f'  param={r.get("param_name","?")}  '
              f'track={r["_track_name"]}  '
              f'noise={r.get("noise_scale", 0.0):.3g}  '
              f'cutoff={r.get("adc_cutoff", 0.0):.3g}  '
              f'N={r.get("N","?")}')
    print(f'Loaded {len(results)} file(s)')
    return results


def _cutoff_colormap(all_runs):
    """Return dict {cutoff_value: rgba_color} with a shared viridis scale."""
    cutoffs = sorted(set(r.get('adc_cutoff', 0.0) for r in all_runs))
    cmap = cm.get_cmap('viridis')
    # Stop at 0.82 to avoid the bright yellow end of viridis.
    norms = np.linspace(0.0, 0.82, max(len(cutoffs), 1))
    return {c: cmap(norms[i]) for i, c in enumerate(cutoffs)}, cutoffs


def _plot_row(axes, runs, cutoff_color, label_noise):
    """Fill one row of axes (loss, |grad|, grad) with curves for a set of runs."""
    ax_loss, ax_grad_abs, ax_grad = axes
    for run in sorted(runs, key=lambda r: r.get('adc_cutoff', 0.0)):
        cutoff  = run.get('adc_cutoff', 0.0)
        color   = cutoff_color[cutoff]
        factors = np.array(run['factors'])
        losses  = np.array(run['loss_values'])
        grads   = np.array(run['grad_values'])
        label   = f'cutoff={cutoff:.4g}'

        kw = dict(color=color, lw=1.8, marker='o', ms=4, label=label)
        ax_loss.plot(factors, losses, **kw)
        ax_grad_abs.plot(factors, np.abs(grads), **kw)
        ax_grad.plot(factors, grads, **kw)

    for ax in axes:
        ax.axvline(1.0, color='black', ls=':', lw=1.2, alpha=0.7)
        ax.set_xlabel('Factor  (param / GT)')
        ax.grid(True, which='both', alpha=0.3)

    ax_loss.set_yscale('log')
    ax_loss.set_ylabel(f'Loss  [{label_noise}]')
    ax_grad_abs.set_yscale('log')
    ax_grad_abs.set_ylabel(f'|∂L/∂pₙ|  [{label_noise}]')
    ax_grad.axhline(0, color='grey', ls='--', lw=0.8, alpha=0.6)
    ax_grad.set_ylabel(f'∂L/∂pₙ  [{label_noise}]')
    ax_loss.legend(fontsize=7, ncol=2)


def _make_per_param_figure(param_name, track_name, runs, output_dir, title_suffix=''):
    """2-row (clean / noisy) × 3-column (loss, |grad|, grad) figure."""
    clean_runs = sorted([r for r in runs if r.get('noise_scale', 0.0) == 0.0],
                        key=lambda r: r.get('adc_cutoff', 0.0))
    noisy_runs = sorted([r for r in runs if r.get('noise_scale', 0.0) > 0.0],
                        key=lambda r: r.get('adc_cutoff', 0.0))

    has_noise  = len(noisy_runs) > 0
    n_rows     = 2 if has_noise else 1
    cutoff_color, cutoffs = _cutoff_colormap(runs)

    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), squeeze=False)

    _plot_row(axes[0], clean_runs, cutoff_color, 'clean GT')
    axes[0, 0].set_title('Loss vs. factor  [clean GT]')
    axes[0, 1].set_title('|Gradient| vs. factor  [clean GT]')
    axes[0, 2].set_title('Signed gradient vs. factor  [clean GT]')

    if has_noise:
        noise_scale = noisy_runs[0].get('noise_scale', 1.0)
        _plot_row(axes[1], noisy_runs, cutoff_color, f'noisy GT  σ={noise_scale:.3g}')
        axes[1, 0].set_title(f'Loss vs. factor  [noisy GT  σ={noise_scale:.3g}]')
        axes[1, 1].set_title(f'|Gradient| vs. factor  [noisy GT  σ={noise_scale:.3g}]')
        axes[1, 2].set_title(f'Signed gradient vs. factor  [noisy GT  σ={noise_scale:.3g}]')

    plabel     = PARAM_LABELS.get(param_name, param_name)
    N          = runs[0].get('N', '?')
    range_frac = runs[0].get('range_frac', '?')
    loss_name  = runs[0].get('loss_name', '?')
    track_label = f'  |  track: {track_name}' if track_name else ''
    fig.suptitle(
        f'{plabel}{track_label}{title_suffix}   |   {loss_name}   |   N={N}  ±{range_frac:.0%}\n'
        f'ADC cutoff comparison  ({len(cutoffs)} cutoffs: {cutoffs})',
        fontsize=12,
    )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    safe_track = track_name.replace('/', '_')
    out = os.path.join(output_dir, f'sobolev_cutoff_{param_name}_{safe_track}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out}')


def _aggregate_across_tracks(runs_per_track, param_name):
    """
    Sum loss_values and grad_values across all tracks for each (noise_scale, cutoff).
    Returns a list of synthetic run dicts with the combined totals.
    """
    # key: (noise_scale, cutoff) → list of runs (one per track)
    groups = defaultdict(list)
    for runs in runs_per_track.values():
        for r in runs:
            key = (r.get('noise_scale', 0.0), r.get('adc_cutoff', 0.0))
            groups[key].append(r)

    combined = []
    for (noise_scale, cutoff), group in sorted(groups.items()):
        factors_ref = np.array(group[0]['factors'])
        total_loss  = np.zeros_like(factors_ref)
        total_grad  = np.zeros_like(factors_ref)
        for r in group:
            total_loss += np.array(r['loss_values'])
            total_grad += np.array(r['grad_values'])
        synth = dict(group[0])  # copy metadata from first run
        synth['loss_values']  = total_loss.tolist()
        synth['grad_values']  = total_grad.tolist()
        synth['noise_scale']  = noise_scale
        synth['adc_cutoff']   = cutoff
        synth['param_name']   = param_name
        synth['_track_name']  = 'combined'
        combined.append(synth)
    return combined


def _make_overlay_figure(by_param_track, all_track_names, output_dir):
    """
    |Gradient| overlay figure.

    Rows: one per individual track + one 'combined' row.
    Columns: one per sweep parameter.
    solid = clean GT, dashed = noisy GT.
    """
    param_list  = sorted(by_param_track.keys())
    track_order = sorted(all_track_names)
    row_labels  = track_order + ['combined']
    n_rows      = len(row_labels)
    n_cols      = len(param_list)

    if not param_list:
        return

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(9 * n_cols, 4 * n_rows),
                             squeeze=False)

    for col, param_name in enumerate(param_list):
        runs_per_track = by_param_track[param_name]

        all_runs_flat  = [r for runs in runs_per_track.values() for r in runs]
        cutoff_color, _ = _cutoff_colormap(all_runs_flat)

        combined_runs = _aggregate_across_tracks(runs_per_track, param_name)

        for row, row_label in enumerate(row_labels):
            ax = axes[row, col]
            if row_label == 'combined':
                row_runs = combined_runs
            else:
                row_runs = runs_per_track.get(row_label, [])

            for run in sorted(row_runs, key=lambda r: (r.get('noise_scale', 0.0),
                                                        r.get('adc_cutoff', 0.0))):
                cutoff      = run.get('adc_cutoff', 0.0)
                noise_scale = run.get('noise_scale', 0.0)
                color       = cutoff_color[cutoff]
                ls          = '--' if noise_scale > 0.0 else '-'
                noise_tag   = f'  σ={noise_scale:.3g}' if noise_scale > 0.0 else ''
                factors     = np.array(run['factors'])
                grads       = np.abs(np.array(run['grad_values']))
                ax.plot(factors, grads, color=color, lw=1.8, ls=ls, marker='o', ms=3,
                        label=f'cutoff={cutoff:.4g}{noise_tag}')

            ax.axvline(1.0, color='black', ls=':', lw=1.2, alpha=0.7, label='GT')
            ax.set_yscale('log')
            ax.set_xlabel('Factor  (param / GT)')
            ax.set_ylabel('|∂L / ∂pₙ|  (log scale)')
            ax.grid(True, which='both', alpha=0.3)
            ax.legend(fontsize=6, ncol=2)

            if row == 0:
                ax.set_title(PARAM_LABELS.get(param_name, param_name), fontsize=11)
            ax.set_ylabel(f'{row_label}\n|∂L / ∂pₙ|', fontsize=8)

    fig.suptitle(
        '|Gradient| vs. factor  —  ADC cutoff comparison\n'
        'solid = clean GT   dashed = noisy GT',
        fontsize=13,
    )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, 'sobolev_cutoff_grad_overlay.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out}')


def _make_summary_noise_figure(by_param_track, all_track_names, output_dir):
    """
    Summary: noisy-GT loss landscape for every track + combined.

    Rows: one per individual track (sorted) + 'combined'.
    Columns: D⊥ (trans) left, D∥ (long) right (or whatever two params exist).
    Each cell: loss vs. factor curves, one per ADC cutoff (viridis, log y).
    """
    # Put trans before long for the column order; fall back to sorted for other params.
    def _param_sort_key(p):
        if 'trans' in p: return 0
        if 'long'  in p: return 1
        return 2
    param_list  = sorted(by_param_track.keys(), key=_param_sort_key)
    track_order = sorted(all_track_names)
    row_labels  = track_order + ['combined']
    n_rows      = len(row_labels)
    n_cols      = len(param_list)

    if not param_list:
        return

    # Build a shared cutoff→colour map across all runs.
    all_runs_flat = [r for pt in by_param_track.values()
                       for runs in pt.values() for r in runs]
    cutoff_color, cutoffs = _cutoff_colormap(all_runs_flat)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(8 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for col, param_name in enumerate(param_list):
        runs_per_track  = by_param_track[param_name]
        combined_runs   = _aggregate_across_tracks(runs_per_track, param_name)

        for row, row_label in enumerate(row_labels):
            ax = axes[row, col]
            src = (combined_runs if row_label == 'combined'
                   else runs_per_track.get(row_label, []))
            noisy_runs = sorted([r for r in src if r.get('noise_scale', 0.0) > 0.0],
                                key=lambda r: r.get('adc_cutoff', 0.0))

            for run in noisy_runs:
                cutoff  = run.get('adc_cutoff', 0.0)
                factors = np.array(run['factors'])
                losses  = np.array(run['loss_values'])
                ax.plot(factors, losses, color=cutoff_color[cutoff],
                        lw=1.6, marker='o', ms=3, label=f'cutoff={cutoff:.4g}')

            ax.axvline(1.0, color='black', ls=':', lw=1.2, alpha=0.7)
            ax.set_yscale('log')
            ax.set_xlabel('Factor  (param / GT)')
            ax.grid(True, which='both', alpha=0.3)
            ax.set_ylabel(f'{row_label}\nLoss', fontsize=8)

            if row == 0:
                ax.set_title(PARAM_LABELS.get(param_name, param_name), fontsize=11)
            ax.legend(fontsize=7, ncol=2)

    n_tracks = len(track_order)
    noise_scale = next((r.get('noise_scale', 1.0)
                        for pt in by_param_track.values()
                        for runs in pt.values()
                        for r in runs if r.get('noise_scale', 0.0) > 0.0), 1.0)
    fig.suptitle(
        f'Loss landscape — noisy GT (σ={noise_scale:.3g}) — ADC cutoff comparison\n'
        f'{n_tracks} tracks + combined   |   '
        f'{len(cutoffs)} cutoffs: {cutoffs}',
        fontsize=12,
    )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, 'sobolev_cutoff_summary_noise.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out}')


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(
        _PLOTS_DIR, os.path.basename(args.results_dir.rstrip('/'))
    )

    print(f'Results dir : {args.results_dir}')
    print(f'Output dir  : {output_dir}')
    print()

    results = load_pkls(args.results_dir)

    # Group: by_param_track[param_name][track_name] = [runs]
    by_param_track = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_param_track[r.get('param_name', 'unknown')][r['_track_name']].append(r)

    all_track_names = sorted({r['_track_name'] for r in results})
    print(f'\nFound tracks : {all_track_names}')
    print(f'Found params : {sorted(by_param_track.keys())}')
    print()

    for param_name, runs_per_track in sorted(by_param_track.items()):
        for track_name, runs in sorted(runs_per_track.items()):
            n_clean = sum(1 for r in runs if r.get('noise_scale', 0.0) == 0.0)
            n_noisy = len(runs) - n_clean
            print(f'Plotting {param_name}  track={track_name}'
                  f'  ({n_clean} clean + {n_noisy} noisy runs) …')
            _make_per_param_figure(param_name, track_name, runs, output_dir)

        # Combined figure (sum across all tracks)
        combined_runs = _aggregate_across_tracks(runs_per_track, param_name)
        n_tracks = len(runs_per_track)
        print(f'Plotting {param_name}  combined ({n_tracks} tracks) …')
        _make_per_param_figure(param_name, 'combined', combined_runs, output_dir,
                               title_suffix=f'  |  {n_tracks} tracks summed')

    print('Plotting gradient overlay …')
    _make_overlay_figure(by_param_track, all_track_names, output_dir)

    print('Plotting noise summary …')
    _make_summary_noise_figure(by_param_track, all_track_names, output_dir)

    print(f'\nDone.  Figures in {output_dir}')


if __name__ == '__main__':
    main()
