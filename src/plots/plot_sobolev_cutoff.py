#!/usr/bin/env python
"""
Compare Sobolev loss landscapes across ADC cutoff values, clean and noisy GT.

Loads pkl files produced by 1d_gradients.py --adc-cutoff ... --noise-scale ...,
groups them by sweep parameter and noise variant, and saves:

  sobolev_cutoff_<param>.pdf
      2-row grid (top = clean GT, bottom = noisy GT), 3 columns:
        • Loss vs. factor          (log y)
        • |Gradient| vs. factor   (log y)
        • Signed gradient vs. factor
      Colour encodes ADC cutoff (viridis).

  sobolev_cutoff_grad_overlay.pdf
      One panel per sweep parameter, |gradient| curves for all cutoffs;
      solid = clean GT, dashed = noisy GT.

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


def load_pkls(results_dir):
    paths = sorted(glob.glob(os.path.join(results_dir, '*.pkl')))
    if not paths:
        raise FileNotFoundError(f'No .pkl files in {results_dir!r}')
    results = []
    for path in paths:
        with open(path, 'rb') as f:
            r = pickle.load(f)
        results.append(r)
        print(f'  {os.path.basename(path)}'
              f'  param={r.get("param_name","?")}  '
              f'noise={r.get("noise_scale", 0.0):.3g}  '
              f'cutoff={r.get("adc_cutoff", 0.0):.3g}  '
              f'N={r.get("N","?")}')
    print(f'Loaded {len(results)} file(s)')
    return results


def _cutoff_colormap(all_runs):
    """Return dict {cutoff_value: rgba_color} with a shared viridis scale."""
    cutoffs = sorted(set(r.get('adc_cutoff', 0.0) for r in all_runs))
    cmap = cm.get_cmap('viridis', len(cutoffs))
    return {c: cmap(i) for i, c in enumerate(cutoffs)}, cutoffs


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


def _make_per_param_figure(param_name, runs, output_dir):
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
    fig.suptitle(
        f'{plabel}   |   {loss_name}   |   N={N}  ±{range_frac:.0%}\n'
        f'ADC cutoff comparison  ({len(cutoffs)} cutoffs: {cutoffs})',
        fontsize=12,
    )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f'sobolev_cutoff_{param_name}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out}')


def _make_overlay_figure(by_param, output_dir):
    """|Gradient| overlay: one panel per param, solid=clean, dashed=noisy."""
    param_list = sorted(by_param.keys())
    if not param_list:
        return

    fig, axes = plt.subplots(1, len(param_list), figsize=(9 * len(param_list), 5),
                             sharey=False, squeeze=False)

    for ax, param_name in zip(axes[0], param_list):
        runs = by_param[param_name]
        cutoff_color, _ = _cutoff_colormap(runs)

        for run in sorted(runs, key=lambda r: (r.get('noise_scale', 0.0),
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
        ax.set_title(PARAM_LABELS.get(param_name, param_name))
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

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


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(
        _PLOTS_DIR, os.path.basename(args.results_dir.rstrip('/'))
    )

    print(f'Results dir : {args.results_dir}')
    print(f'Output dir  : {output_dir}')
    print()

    results = load_pkls(args.results_dir)

    by_param = {}
    for r in results:
        by_param.setdefault(r.get('param_name', 'unknown'), []).append(r)

    print()
    for param_name, runs in sorted(by_param.items()):
        n_clean = sum(1 for r in runs if r.get('noise_scale', 0.0) == 0.0)
        n_noisy = len(runs) - n_clean
        print(f'Plotting {param_name}  ({n_clean} clean + {n_noisy} noisy runs) …')
        _make_per_param_figure(param_name, runs, output_dir)

    print('Plotting gradient overlay …')
    _make_overlay_figure(by_param, output_dir)

    print(f'\nDone.  Figures in {output_dir}')


if __name__ == '__main__':
    main()
