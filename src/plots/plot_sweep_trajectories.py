#!/usr/bin/env python
"""
Plot parameter trajectories from Adam_NoiseSeedSweep_3k sweeps.

Outputs (in --output-dir):
  sweep_trajectories_{noise,nonoise}.pdf             — GT1 vs GT2
  sweep_trajectories_noise_and_nonoise.pdf            — GT1 vs GT2, noise=light / no-noise=dark
  sweep_trajectories_{noise,nonoise}_GT_step_size.pdf — GT1 1mm vs 0.1mm GT step
  sweep_trajectories_cont_{noise,nonoise}.pdf         — stitched Adam→Newton, GT1 vs GT2
  sweep_trajectories_cont_noise_and_nonoise.pdf       — stitched Adam→Newton, noise+no-noise

Outputs (in --output-dir/sweep_trajectories_varying_step_sizes):
  sweep_trajectories_{noise,nonoise}_step_sizes.pdf       — 1mm/1mm vs 0.1mm GT/1mm sim vs 0.1mm/0.1mm
  sweep_trajectories_noise_and_nonoise_step_sizes.pdf     — same, noise=light / no-noise=dark
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from dotenv import load_dotenv
load_dotenv()

import argparse, glob, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')
_PLOTS_DIR   = os.environ.get('PLOTS_DIR',   'plots')

GT1_BASE              = os.path.join(_RESULTS_DIR, 'opt', 'Adam_NoiseSeedSweep_3k')
GT2_BASE              = os.path.join(_RESULTS_DIR, 'opt', 'Adam_NoiseSeedSweep_3k_GT2')
GT1_01MM_BASE         = os.path.join(_RESULTS_DIR, 'opt', 'Adam_NoiseSeedSweep_3k_0p1mm_step_GT')
GT1_01MM_SIM_BASE     = os.path.join(_RESULTS_DIR, 'opt', 'Adam_NoiseSeedSweep_3k_0p1mm_step_GT_and_sim')
GT1_NEWTON_BASE       = os.path.join(_RESULTS_DIR, 'opt', 'Adam_NoiseSeedSweep_3k_Newton_cont')
GT2_NEWTON_BASE       = os.path.join(_RESULTS_DIR, 'opt', 'Adam_NoiseSeedSweep_3k_GT2_Newton_cont')

GT1_DARK,    GT1_LIGHT    = '#2ca02c', '#98df8a'  # green
GT2_DARK,    GT2_LIGHT    = '#1f77b4', '#aec7e8'  # blue
STEP01_DARK, STEP01_LIGHT = '#ff7f0e', '#ffbb78'  # orange
BOTH01_DARK, BOTH01_LIGHT = '#9467bd', '#c5b0d5'  # purple
GT_LINE_GRAY = '#888888'
TRIAL_ALPHA, TRIAL_LW = 0.5, 1.0
NEWTON_LS = '--'  # linestyle for Newton continuation segment

PARAM_LABELS = {
    'velocity_cm_us':         'drift velocity\n(cm/μs)',
    'lifetime_us':            'electron lifetime\n(μs)',
    'diffusion_trans_cm2_us': 'transverse diffusion\n(cm²/μs)',
    'diffusion_long_cm2_us':  'longitudinal diffusion\n(cm²/μs)',
    'recomb_alpha':           'recombination α',
    'recomb_beta_90':         'recombination β₉₀',
    'recomb_R':               'recombination R',
}


def load_runs(base_dir):
    paths = sorted(glob.glob(os.path.join(base_dir, '**', 'result_*.pkl'), recursive=True))
    if not paths:
        print(f'  [warn] no pkl files found under {base_dir!r}')
        return []
    runs = []
    for path in paths:
        with open(path, 'rb') as f:
            r = pickle.load(f)
        r['_path'] = path
        runs.append(r)
    print(f'  loaded {len(runs)} run(s) from {base_dir!r}')
    return runs


def _filter(runs, with_noise):
    return [r for r in runs if (r.get('noise_scale', 0.0) > 0.0) == with_noise]


def _pairs(runs, dark, light):
    """Assign dark to no-noise runs, light to noise runs."""
    return [(r, dark if r.get('noise_scale', 0.0) == 0.0 else light) for r in runs]


def stitch_runs(adam_runs, newton_runs):
    """Concatenate Adam and Newton trajectories matched by (seed, noise_scale).

    Newton[0] is dropped since it duplicates Adam[-1].  The returned run dicts
    gain a '_adam_steps' key = len(adam param_trajectory), used to draw a
    vertical transition line in the plot.
    """
    nmap = {(r.get('seed'), r.get('noise_scale', 0.0)): r for r in newton_runs}
    stitched = []
    for ar in adam_runs:
        if not ar.get('trials'):
            continue
        key = (ar.get('seed'), ar.get('noise_scale', 0.0))
        nr = nmap.get(key)
        if nr is None or not nr.get('trials'):
            continue
        adam_n = len(ar['trials'][0]['param_trajectory'])
        merged_trials = [
            {'param_trajectory': at['param_trajectory'] + nt['param_trajectory'][1:]}
            for at, nt in zip(ar['trials'], nr['trials'])
        ]
        stitched.append({**ar, 'trials': merged_trials, '_adam_steps': adam_n})
    if not stitched:
        print(f'  [warn] no matching Adam+Newton pairs found')
    return stitched


def _plot_group(run_color_pairs, param_name, ax_val, ax_err, gt_line_color, color_labels=None):
    """Plot trajectories. First curve of each color is labeled via color_labels dict.

    Stitched runs (with '_adam_steps') are drawn solid up to the Adam/Newton
    boundary and dashed after; both segments share the same color.
    """
    gt_drawn, seen = False, set()
    for r, color in run_color_pairs:
        if param_name not in r['param_names']:
            continue
        pi      = r['param_names'].index(param_name)
        scale   = r['scales'][pi]
        gt_phys = r['param_gts'][pi]
        is_log  = np.isclose(np.exp(r['p_n_gts'][pi]) * scale, gt_phys, rtol=1e-3)
        adam_n  = r.get('_adam_steps')
        for trial in r['trials']:
            arr   = np.array([s[pi] for s in trial['param_trajectory']])
            phys  = np.exp(arr) * scale if is_log else arr * scale
            steps = np.arange(len(phys))
            err   = np.abs(phys - gt_phys) / np.abs(gt_phys)
            lbl   = (color_labels or {}).get(color) if color not in seen else '_nolegend_'
            kw    = dict(color=color, alpha=TRIAL_ALPHA, lw=TRIAL_LW)
            if adam_n is not None:
                ax_val.plot(steps[:adam_n], phys[:adam_n], **kw, label=lbl or '_nolegend_')
                ax_err.plot(steps[:adam_n], err[:adam_n],  **kw, label=lbl or '_nolegend_')
                ax_val.plot(steps[adam_n-1:], phys[adam_n-1:], **kw, ls=NEWTON_LS)
                ax_err.plot(steps[adam_n-1:], err[adam_n-1:],  **kw, ls=NEWTON_LS)
            else:
                ax_val.plot(steps, phys, **kw, label=lbl or '_nolegend_')
                ax_err.plot(steps, err,  **kw, label=lbl or '_nolegend_')
            seen.add(color)
        if not gt_drawn:
            ax_val.axhline(gt_phys, color=gt_line_color, ls='--', lw=1.4, zorder=6)
            gt_drawn = True


def make_figure(a_pairs, b_pairs, output_path, title_suffix='',
                color_labels=None, group_labels=('A', 'B', 'A + B'),
                adam_transition_step=None, c_pairs=None):
    """color_labels: dict mapping color hex → legend label shown on each subplot.

    When c_pairs is provided the figure has 8 columns: A, B, C, A+B+C.
    group_labels should then be a 4-tuple: (A label, B label, C label, combined label).
    """
    all_pairs = a_pairs + b_pairs + (c_pairs or [])
    ref = [r for r, _ in all_pairs]
    if not ref:
        print('No runs loaded; nothing to plot.')
        return
    param_names = ref[0]['param_names']
    a_color = a_pairs[0][1] if a_pairs else GT1_DARK
    b_color = b_pairs[0][1] if b_pairs else GT2_DARK

    n_groups = 4 if c_pairs is not None else 3
    n_cols   = n_groups * 2
    fig, axes = plt.subplots(len(param_names), n_cols,
                             figsize=(4.5 * n_cols, 3.5 * len(param_names)), squeeze=False)

    col_starts = list(range(0, n_cols, 2))
    for label, col in zip(group_labels, col_starts):
        for ci in [col, col + 1]:
            axes[0, ci].set_title(
                f'{"param value" if ci % 2 == 0 else "relative error"}\n[{label}]',
                fontsize=9, fontweight='bold')

    for row, param_name in enumerate(param_names):
        plabel = PARAM_LABELS.get(param_name, param_name)
        av, ae = axes[row, 0], axes[row, 1]
        bv, be = axes[row, 2], axes[row, 3]

        _plot_group(a_pairs, param_name, av, ae, GT_LINE_GRAY, color_labels)
        _plot_group(b_pairs, param_name, bv, be, GT_LINE_GRAY, color_labels)

        if c_pairs is not None:
            # 4-group layout: A | B | C | A+B+C
            c_color = c_pairs[0][1]
            cv, ce = axes[row, 4], axes[row, 5]
            dv, de = axes[row, 6], axes[row, 7]
            _plot_group(c_pairs, param_name, cv,  ce,  GT_LINE_GRAY, color_labels)
            _plot_group(a_pairs, param_name, dv,  de,  a_color,      color_labels)
            _plot_group(b_pairs, param_name, dv,  de,  b_color,      color_labels)
            _plot_group(c_pairs, param_name, dv,  de,  c_color,      color_labels)
            err_axes = [ae, be, ce, de]
            val_axes = [av, bv, cv, dv]
        else:
            # 3-group layout: A | B | A+B
            dv, de = axes[row, 4], axes[row, 5]
            _plot_group(a_pairs, param_name, dv, de, a_color, color_labels)
            _plot_group(b_pairs, param_name, dv, de, b_color, color_labels)
            err_axes = [ae, be, de]
            val_axes = [av, bv, dv]
        for ax in err_axes:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
            ax.grid(True, which='both', alpha=0.2)
            ax.set_ylabel(f'rel. error in {plabel}', fontsize=8)
        for ax in val_axes:
            ax.grid(True, alpha=0.2)
            ax.set_ylabel(plabel, fontsize=8)
        for ax in axes[row]:
            ax.set_xlabel('step', fontsize=8)
            ax.tick_params(labelsize=7)
            if adam_transition_step is not None:
                ax.axvline(adam_transition_step, color='gray', ls=':', lw=0.8, alpha=0.7, zorder=5)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7, handlelength=1.2, loc='best')

    fig.suptitle(
        f'Parameter trajectories  [{title_suffix}]\n(each curve = one seed run)',
        fontsize=11, y=1.01)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {output_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gt1-dir',    default=GT1_BASE)
    p.add_argument('--gt2-dir',    default=GT2_BASE)
    p.add_argument('--output-dir', default=os.path.join(_PLOTS_DIR, 'opt'))
    args = p.parse_args()

    gt1_runs           = load_runs(args.gt1_dir)
    gt2_runs           = load_runs(args.gt2_dir)
    gt1_01mm_runs      = load_runs(GT1_01MM_BASE)
    gt1_01mm_sim_runs  = load_runs(GT1_01MM_SIM_BASE)

    gt1_newton_runs = load_runs(GT1_NEWTON_BASE)
    gt2_newton_runs = load_runs(GT2_NEWTON_BASE)
    gt1_stitched    = stitch_runs(gt1_runs, gt1_newton_runs)
    gt2_stitched    = stitch_runs(gt2_runs, gt2_newton_runs)
    _adam_ts = gt1_stitched[0]['_adam_steps'] - 1 if gt1_stitched else (
               gt2_stitched[0]['_adam_steps'] - 1 if gt2_stitched else None)

    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda name: os.path.join(args.output_dir, name)

    # GT1 vs GT2, split by noise condition
    for with_noise, tag, label in [(True, 'noise', 'with noise'), (False, 'nonoise', 'no noise')]:
        g1, g2 = _filter(gt1_runs, with_noise), _filter(gt2_runs, with_noise)
        if not g1 and not g2:
            continue
        make_figure(
            [(r, GT1_DARK) for r in g1], [(r, GT2_DARK) for r in g2],
            out(f'sweep_trajectories_{tag}.pdf'), title_suffix=label,
            color_labels={GT1_DARK: 'GT1', GT2_DARK: 'GT2'},
            group_labels=('GT1', 'GT2', 'GT1 + GT2'),
        )

    # GT1 vs GT2, noise + no-noise combined (light = noise, dark = no-noise)
    make_figure(
        _pairs(gt1_runs, GT1_DARK, GT1_LIGHT),
        _pairs(gt2_runs, GT2_DARK, GT2_LIGHT),
        out('sweep_trajectories_noise_and_nonoise.pdf'),
        title_suffix='noise + no noise',
        color_labels={GT1_DARK:  'GT1',       GT1_LIGHT: 'GT1 noise',
                      GT2_DARK:  'GT2',        GT2_LIGHT: 'GT2 noise'},
        group_labels=('GT1', 'GT2', 'GT1 + GT2'),
    )

    # Stitched Adam→Newton: GT1 vs GT2, noise+no-noise combined
    make_figure(
        _pairs(gt1_stitched, GT1_DARK, GT1_LIGHT),
        _pairs(gt2_stitched, GT2_DARK, GT2_LIGHT),
        out('sweep_trajectories_cont_noise_and_nonoise.pdf'),
        title_suffix='Adam→Newton  [noise + no noise]',
        color_labels={GT1_DARK: 'GT1', GT1_LIGHT: 'GT1 noise',
                      GT2_DARK: 'GT2', GT2_LIGHT: 'GT2 noise'},
        group_labels=('GT1', 'GT2', 'GT1 + GT2'),
        adam_transition_step=_adam_ts,
    )

    # Stitched Adam→Newton: GT1 vs GT2, split by noise condition
    for with_noise, tag, label in [(True, 'noise', 'with noise'), (False, 'nonoise', 'no noise')]:
        g1, g2 = _filter(gt1_stitched, with_noise), _filter(gt2_stitched, with_noise)
        if not g1 and not g2:
            continue
        ts = g1[0]['_adam_steps'] - 1 if g1 else g2[0]['_adam_steps'] - 1
        make_figure(
            [(r, GT1_DARK) for r in g1], [(r, GT2_DARK) for r in g2],
            out(f'sweep_trajectories_cont_{tag}.pdf'),
            title_suffix=f'Adam→Newton  [{label}]',
            color_labels={GT1_DARK: 'GT1', GT2_DARK: 'GT2'},
            group_labels=('GT1', 'GT2', 'GT1 + GT2'),
            adam_transition_step=ts,
        )

    # GT step-size comparison: 1mm vs 0.1mm GT, split by noise
    for with_noise, tag, label in [(True, 'noise', 'with noise'), (False, 'nonoise', 'no noise')]:
        g1, g01 = _filter(gt1_runs, with_noise), _filter(gt1_01mm_runs, with_noise)
        if not g1 and not g01:
            continue
        c1  = GT1_LIGHT    if with_noise else GT1_DARK
        c01 = STEP01_LIGHT if with_noise else STEP01_DARK
        make_figure(
            [(r, c1) for r in g1], [(r, c01) for r in g01],
            out(f'sweep_trajectories_{tag}_GT_step_size.pdf'),
            title_suffix=f'GT step size  [{label}]',
            color_labels={c1: 'GT 1mm, sim 1mm', c01: 'GT 0.1mm, sim 1mm'},
            group_labels=('GT1  1mm step', 'GT1  0.1mm step', 'GT1  1mm + 0.1mm'),
        )

    # 3-way step-size comparison: 1mm/1mm vs 0.1mm GT/1mm sim vs 0.1mm/0.1mm
    step_out = lambda name: os.path.join(args.output_dir, 'sweep_trajectories_varying_step_sizes', name)
    os.makedirs(os.path.join(args.output_dir, 'sweep_trajectories_varying_step_sizes'), exist_ok=True)

    for with_noise, tag, label in [(True, 'noise', 'with noise'), (False, 'nonoise', 'no noise')]:
        g1   = _filter(gt1_runs,          with_noise)
        g01  = _filter(gt1_01mm_runs,     with_noise)
        g01s = _filter(gt1_01mm_sim_runs, with_noise)
        if not g1 and not g01 and not g01s:
            continue
        c1   = GT1_LIGHT    if with_noise else GT1_DARK
        c01  = STEP01_LIGHT if with_noise else STEP01_DARK
        c01s = BOTH01_LIGHT if with_noise else BOTH01_DARK
        make_figure(
            [(r, c1)   for r in g1],
            [(r, c01)  for r in g01],
            step_out(f'sweep_trajectories_{tag}_step_sizes.pdf'),
            title_suffix=f'step sizes  [{label}]',
            color_labels={c1: 'GT 1mm, sim 1mm', c01: 'GT 0.1mm, sim 1mm', c01s: 'GT 0.1mm, sim 0.1mm'},
            group_labels=('GT 1mm / sim 1mm', 'GT 0.1mm / sim 1mm', 'GT 0.1mm / sim 0.1mm', 'all combined'),
            c_pairs=[(r, c01s) for r in g01s],
        )

    # 3-way step-size comparison: noise + no-noise combined
    g1   = gt1_runs
    g01  = gt1_01mm_runs
    g01s = gt1_01mm_sim_runs
    if g1 or g01 or g01s:
        make_figure(
            _pairs(g1,   GT1_DARK,    GT1_LIGHT),
            _pairs(g01,  STEP01_DARK, STEP01_LIGHT),
            step_out('sweep_trajectories_noise_and_nonoise_step_sizes.pdf'),
            title_suffix='step sizes  [noise + no noise]',
            color_labels={GT1_DARK:    'GT 1mm, sim 1mm',
                          GT1_LIGHT:   'GT 1mm, sim 1mm (noise)',
                          STEP01_DARK: 'GT 0.1mm, sim 1mm',
                          STEP01_LIGHT:'GT 0.1mm, sim 1mm (noise)',
                          BOTH01_DARK: 'GT 0.1mm, sim 0.1mm',
                          BOTH01_LIGHT:'GT 0.1mm, sim 0.1mm (noise)'},
            group_labels=('GT 1mm / sim 1mm', 'GT 0.1mm / sim 1mm', 'GT 0.1mm / sim 0.1mm', 'all combined'),
            c_pairs=_pairs(g01s, BOTH01_DARK, BOTH01_LIGHT),
        )


if __name__ == '__main__':
    main()
