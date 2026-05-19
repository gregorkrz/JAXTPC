#!/usr/bin/env python
"""
Plot parameter trajectories for Adam_NoiseCutoffDiffusion_3k across ADC cutoffs.

Produces one PDF with one page per param-set (trans_only / long_only / both_diff).
Each page shows 2 panels per optimised diffusion parameter:
  - top:    physical value vs iteration  (GT marked as dashed grey line)
  - bottom: relative error vs iteration  (log y-scale)

Each ADC cutoff gets a distinct colour (viridis); all seeds are overlaid as
semi-transparent lines.

Usage
-----
    python src/plots/plot_cutoff_trajectories.py
    python src/plots/plot_cutoff_trajectories.py \\
        --results-dir results/opt \\
        --output     plots/opt/cutoff_trajectories.pdf
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from dotenv import load_dotenv
load_dotenv()

import argparse, glob, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

_RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')
_PLOTS_DIR   = os.environ.get('PLOTS_DIR',   'plots')

CUTOFFS = [5, 10, 15, 20]

PARAM_SETS = [
    ('trans_only', ['diffusion_trans_cm2_us']),
    ('long_only',  ['diffusion_long_cm2_us']),
    ('both_diff',  ['diffusion_trans_cm2_us', 'diffusion_long_cm2_us']),
]

PARAM_LABELS = {
    'diffusion_trans_cm2_us': 'D⊥  (cm²/μs)',
    'diffusion_long_cm2_us':  'D∥  (cm²/μs)',
}

GT_COLOR   = '#555555'
TRIAL_ALPHA = 0.45
TRIAL_LW    = 1.2


def _cutoff_colors(cutoffs):
    cmap = cm.get_cmap('viridis', len(cutoffs))
    return {c: cmap(i) for i, c in enumerate(cutoffs)}


def load_runs_for(results_dir, param_label, cutoff):
    tag  = f'Adam_NoiseCutoffDiffusion_3k_{param_label}_cutoff{int(cutoff)}'
    base = os.path.join(results_dir, tag, 'noise')
    paths = sorted(glob.glob(os.path.join(base, '**', 'result_*.pkl'), recursive=True))
    runs = []
    for path in paths:
        try:
            with open(path, 'rb') as f:
                r = pickle.load(f)
        except Exception as exc:
            print(f'  [warn] could not load {path}: {exc}')
            continue
        r['_path'] = path
        r['_cutoff'] = cutoff
        runs.append(r)
    if not paths:
        print(f'  [warn] no pkl files found in {base!r}')
    else:
        print(f'  {len(runs):2d} run(s)  {tag}')
    return runs


def _draw_param(ax_val, ax_err, runs, param_name, color, first_cutoff):
    """Draw value and relative-error trajectories for one cutoff onto the two axes."""
    gt_drawn = False
    for r in runs:
        if param_name not in r.get('param_names', []):
            continue
        pi      = r['param_names'].index(param_name)
        scale   = r['scales'][pi]
        gt_phys = r['param_gts'][pi]
        is_log  = np.isclose(np.exp(r['p_n_gts'][pi]) * scale, gt_phys, rtol=1e-3)

        for trial in r.get('trials', []):
            arr   = np.array([s[pi] for s in trial['param_trajectory']])
            phys  = np.exp(arr) * scale if is_log else arr * scale
            steps = np.arange(len(phys))
            err   = np.abs(phys - gt_phys) / max(abs(gt_phys), 1e-30)
            ax_val.plot(steps, phys, color=color, alpha=TRIAL_ALPHA, lw=TRIAL_LW)
            ax_err.plot(steps, err,  color=color, alpha=TRIAL_ALPHA, lw=TRIAL_LW)

        if not gt_drawn and first_cutoff:
            ax_val.axhline(gt_phys, color=GT_COLOR, ls='--', lw=1.5, zorder=6, label='GT')
            gt_drawn = True


def make_page(pdf, param_label, diff_params, all_runs_by_cutoff, cutoff_colors):
    """One PDF page: 2 rows × len(diff_params) columns per row-pair."""
    n_params = len(diff_params)
    fig, axes = plt.subplots(
        2, n_params,
        figsize=(7 * n_params, 8),
        squeeze=False,
    )

    for col, param_name in enumerate(diff_params):
        ax_val = axes[0, col]
        ax_err = axes[1, col]
        plabel = PARAM_LABELS.get(param_name, param_name)

        # legend proxy handles (one per cutoff)
        legend_handles = []

        for i, cutoff in enumerate(CUTOFFS):
            runs  = all_runs_by_cutoff.get(cutoff, [])
            color = cutoff_colors[cutoff]
            _draw_param(ax_val, ax_err, runs, param_name, color, first_cutoff=(i == 0))
            patch = plt.Line2D([0], [0], color=color, lw=2, label=f'cutoff = {int(cutoff)} ADC')
            legend_handles.append(patch)

        # GT line in legend
        gt_handle = plt.Line2D([0], [0], color=GT_COLOR, ls='--', lw=1.5, label='GT')
        legend_handles.append(gt_handle)

        ax_val.set_ylabel(plabel, fontsize=11)
        ax_val.set_xlabel('iteration', fontsize=9)
        ax_val.set_title(f'Parameter value — {plabel}', fontsize=10, fontweight='bold')
        ax_val.grid(True, alpha=0.25)
        ax_val.legend(handles=legend_handles, fontsize=8, loc='best', handlelength=1.5)

        ax_err.set_ylabel(f'relative error in {plabel}', fontsize=11)
        ax_err.set_xlabel('iteration', fontsize=9)
        ax_err.set_title(f'Relative error — {plabel}', fontsize=10, fontweight='bold')
        ax_err.set_yscale('log')
        ax_err.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        ax_err.grid(True, which='both', alpha=0.2)
        ax_err.legend(handles=legend_handles, fontsize=8, loc='best', handlelength=1.5)

    param_set_title = {
        'trans_only': 'transverse diffusion only',
        'long_only':  'longitudinal diffusion only',
        'both_diff':  'both diffusion constants',
    }.get(param_label, param_label)

    fig.suptitle(
        f'ADC-cutoff comparison — {param_set_title}\n'
        f'(noise scale = 1.0,  seeds 43/44/45,  each curve = one seed)',
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', default=os.path.join(_RESULTS_DIR, 'opt'),
                   help='Base opt results directory (default: $RESULTS_DIR/opt)')
    p.add_argument('--output', default=None,
                   help='Output PDF path (default: $PLOTS_DIR/opt/cutoff_trajectories.pdf)')
    return p.parse_args()


def main():
    args  = parse_args()
    output = args.output or os.path.join(_PLOTS_DIR, 'opt', 'cutoff_trajectories.pdf')
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

    cutoff_colors = _cutoff_colors(CUTOFFS)

    with PdfPages(output) as pdf:
        for param_label, diff_params in PARAM_SETS:
            print(f'\nLoading {param_label} ...')
            all_runs_by_cutoff = {}
            for cutoff in CUTOFFS:
                runs = load_runs_for(args.results_dir, param_label, cutoff)
                all_runs_by_cutoff[cutoff] = runs

            total = sum(len(v) for v in all_runs_by_cutoff.values())
            if total == 0:
                print(f'  → skipping (no data)')
                continue

            make_page(pdf, param_label, diff_params, all_runs_by_cutoff, cutoff_colors)
            print(f'  → page written for {param_label}')

    print(f'\nSaved → {output}')


if __name__ == '__main__':
    main()
