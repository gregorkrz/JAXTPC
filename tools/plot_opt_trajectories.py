#!/usr/bin/env python3
"""
Plot parameter trajectories from pkl result files for W&B runs with a given tag.

Loads pkls directly via the output_path stored in each W&B run config.
Groups runs by their ground-truth parameter vector; same GT → same color.

Outputs (in --output-dir):
  trajectories_{tag}.html        — interactive Plotly HTML with GT selector checkboxes,
                                    trajectories + final-step distributions, links to PDFs
  trajectories_{tag}.pdf         — param value trajectories, color by GT
  relative_errors_{tag}.pdf      — relative error trajectories, color by GT
  final_distributions_{tag}.pdf  — histograms of final-step values + rel. errors, color by GT

Usage:
  python tools/plot_opt_trajectories.py --tag Run_Opt_20260609
  python tools/plot_opt_trajectories.py --tag Run_Opt_20260609 \\
      --entity fcc_ml --project jaxtpc-optimization \\
      --output-dir plots/trajectories
"""

import argparse
import json
import math
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    _WANDB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


PARAM_LABELS = {
    'velocity_cm_us':         'drift velocity (cm/μs)',
    'lifetime_us':            'electron lifetime (μs)',
    'diffusion_trans_cm2_us': 'trans. diffusion (cm²/μs)',
    'diffusion_long_cm2_us':  'long. diffusion (cm²/μs)',
    'recomb_alpha':           'recombination α',
    'recomb_beta_90':         'recombination β₉₀',
    'recomb_R':               'recombination R',
}

QUALITATIVE_COLORS = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
    '#9467bd', '#8c564b', '#e377c2', '#17becf',
    '#bcbd22', '#7f7f7f',
]

TRIAL_ALPHA = 0.50
TRIAL_LW    = 0.85


# ── helpers ───────────────────────────────────────────────────────────────────

# config-variant tags used in submit_jobs_optimization.py, in priority order
_VARIANT_TAGS = ('adc50_D_only', 'phase2_diffusion', 'adc50')


def _variant(run_data):
    """Return the experiment-config variant tag (adc50 / adc50_D_only / phase2_diffusion)."""
    tags = set(run_data.get('_wandb_tags') or [])
    for v in _VARIANT_TAGS:
        if v in tags:
            return v
    return None


def _gt_key(run_data):
    """Hashable key for the GT + config variant of this run (color/group key)."""
    return (tuple(round(x, 8) for x in run_data['param_gts']), _variant(run_data))


def _gt_label(run_data):
    """Human-readable group label: GT tag (or fallback param values) + config variant."""
    tags = run_data.get('_wandb_tags') or []
    # well-known per-GT tag names used in submit_jobs_optimization.py
    known = {'nominal', 'gt80pct', 'gt50pct', 'gt120pct', 'gt150pct', 'gt70pct', 'gt90pct'}
    gt_label = None
    for tag in tags:
        if tag in known:
            gt_label = tag
            break
        # generic gt* pattern (short, no underscore)
        if tag.startswith('gt') and len(tag) <= 9 and '_' not in tag:
            gt_label = tag
            break
    if gt_label is None:
        # fallback: first three param values
        names = run_data.get('param_names', [])
        gts   = run_data.get('param_gts', [])
        parts = [f"{n.split('_')[0]}={v:.4g}" for n, v in zip(names, gts)]
        gt_label = ', '.join(parts[:3]) or 'GT'
    variant = _variant(run_data)
    return f'{gt_label} / {variant}' if variant else gt_label


def _phys_traj(trial, pi, scale, step=1):
    arr = np.array([s[pi] for s in trial['param_trajectory'][::step]])
    return np.exp(arr) * scale


def _rel_err_traj(trial, pi, scale, gt_phys, step=1):
    phys = _phys_traj(trial, pi, scale, step)
    return np.abs(phys - gt_phys) / np.abs(gt_phys)


def _step_axis(trial, step=1):
    if 'step_indices' in trial:
        return np.array(trial['step_indices'][::step])
    offset = trial.get('_x_offset', 0)
    return np.arange(0, len(trial['param_trajectory']), step) + offset


def _trial_max_step(trial):
    if 'step_indices' in trial:
        return trial['step_indices'][-1]
    return len(trial['param_trajectory']) - 1 + trial.get('_x_offset', 0)


def _final_phys(trial, pi, scale):
    return float(np.exp(trial['param_trajectory'][-1][pi]) * scale)


def _final_rel_err(trial, pi, scale, gt_phys):
    phys = _final_phys(trial, pi, scale)
    return abs(phys - gt_phys) / abs(gt_phys)


# ── W&B + pkl loading ─────────────────────────────────────────────────────────

def fetch_runs_from_wandb(tag, entity, project):
    """Return list of dicts with run_id, output_path (pkl file), wandb_tags."""
    if not _WANDB_AVAILABLE:
        raise RuntimeError('wandb not installed; pip install wandb')
    api = _wandb.Api()
    runs = api.runs(f'{entity}/{project}', filters={'tags': {'$in': [tag]}})
    result = []
    for run in runs:
        cfg = run.config
        output_path = cfg.get('output_path')
        if not output_path:
            print(f'  [warn] run {run.id} has no output_path in config, skipping')
            continue
        result.append({
            'run_id':     run.id,
            'output_path': output_path,
            'wandb_tags': list(run.tags),
            'run':        run,
        })
    print(f'  found {len(result)} runs with tag {tag!r}')
    return result


def _fetch_wandb_param_history(run, param_names):
    """Reconstruct a (param_trajectory, step_indices) pair from W&B's per-step
    'params/{name}_normalized' history (logged by _wandb_log_step every
    log_interval steps + step 0). Survives even if the run later crashed.

    Returns ([], []) if no usable history is found.
    """
    keys = [f'params/{name}_normalized' for name in param_names]
    rows = list(run.scan_history(keys=keys + ['_step']))
    rows = [r for r in rows if all(r.get(k) is not None for k in keys)]
    rows.sort(key=lambda r: r['_step'])
    traj  = [[r[k] for k in keys] for r in rows]
    steps = [int(r['_step']) for r in rows]
    return traj, steps


def load_pkls(run_infos):
    all_runs = []
    for ri in run_infos:
        pkl_path = ri['output_path']
        if not os.path.isfile(pkl_path):
            print(f'  [warn] not found: {pkl_path}')
            continue
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        if not d.get('trials'):
            param_names = d.get('param_names', [])
            ckpt = d.get('live_checkpoint')
            traj, steps = ([], [])
            if ri.get('run') is not None and param_names:
                traj, steps = _fetch_wandb_param_history(ri['run'], param_names)
            if traj:
                if ckpt and ckpt['step'] > steps[-1]:
                    traj  = traj + [ckpt['p'][:len(param_names)]]
                    steps = steps + [ckpt['step']]
                d = dict(d)
                d['trials'] = [{'param_trajectory': traj, 'step_indices': steps}]
                print(f'  [info] {pkl_path} (run {ri["run_id"]}): no completed trials, '
                      f'reconstructed {len(traj)}-point trajectory from W&B history '
                      f'(steps 0-{steps[-1]})')
            elif ckpt:
                n_params = len(param_names)
                d = dict(d)
                d['trials'] = [{
                    'param_trajectory': [ckpt['p'][:n_params]],
                    'step_indices': [ckpt['step']],
                }]
                print(f'  [info] {pkl_path} (run {ri["run_id"]}): no completed trials/history, '
                      f'using live checkpoint @ step {ckpt["step"]}')
            else:
                print(f'  [warn] no trials/checkpoint/history in {pkl_path} (run {ri["run_id"]}), skipping')
                continue
        d['_run_id']     = ri['run_id']
        d['_wandb_tags'] = ri['wandb_tags']
        all_runs.append(d)
    print(f'  loaded {len(all_runs)} pkl files')
    return all_runs


_NPZ_README = """\
trajectories_<tag>.npz -- full-resolution optimization trajectory data.

Load with:
    import numpy as np
    data = np.load('trajectories_<tag>.npz', allow_pickle=True)
    records = data['records']                   # array of dicts, one per (run, trial)
    param_names = data['param_names'].tolist()  # global param order used in the HTML/PDF plots

Each record is a dict with keys:
  gt_label       group label (ground truth + config variant), as shown in the HTML legend
  run_id         W&B run id
  trial_index    index of this trial within the run's pkl 'trials' list
  wandb_tags     list of W&B tags for this run
  param_names    param names for this run (order matches the columns below)
  param_gts      (n_params,) ground-truth physical values
  scales         (n_params,) scale factors; physical = exp(log_normalized) * scale
  p_n_gts        (n_params,) GT values in log-normalized units, or None
  step_indices   (n_steps,) optimizer step number for each row
  param_trajectory_log_normalized  (n_steps, n_params) raw optimizer state
  param_trajectory_physical        (n_steps, n_params) = exp(log_normalized) * scales
  grad_trajectory   (n_steps, n_params) gradients, if present
  loss_trajectory   (n_steps,) loss values, if present
  optimizer, lr, lr_schedule, max_steps, loss_name, tracks   run config
"""


def build_npz(grouped, gt_order, gt_labels, param_names, output_dir, tag):
    """Write trajectories_{tag}.npz with full-resolution per-run trajectory data."""
    records = []
    for gt_key in gt_order:
        gt_label = gt_labels[gt_key]
        for r in grouped[gt_key]:
            scales    = np.asarray(r['scales'], dtype=np.float64)
            param_gts = np.asarray(r['param_gts'], dtype=np.float64)
            p_n_gts   = r.get('p_n_gts')
            p_n_gts   = np.asarray(p_n_gts, dtype=np.float64) if p_n_gts is not None else None
            for ti, trial in enumerate(r['trials']):
                traj  = np.asarray(trial['param_trajectory'], dtype=np.float64)
                steps = np.asarray(_step_axis(trial, step=1), dtype=np.int64)
                phys  = np.exp(traj) * scales[np.newaxis, :]
                rec = dict(
                    gt_label=gt_label,
                    run_id=r.get('_run_id'),
                    trial_index=ti,
                    wandb_tags=list(r.get('_wandb_tags') or []),
                    param_names=list(r['param_names']),
                    param_gts=param_gts,
                    scales=scales,
                    p_n_gts=p_n_gts,
                    step_indices=steps,
                    param_trajectory_log_normalized=traj,
                    param_trajectory_physical=phys,
                    optimizer=r.get('optimizer'),
                    lr=r.get('lr'),
                    lr_schedule=r.get('lr_schedule'),
                    max_steps=r.get('max_steps'),
                    loss_name=r.get('loss_name'),
                    tracks=r.get('tracks'),
                )
                if 'grad_trajectory' in trial:
                    rec['grad_trajectory'] = np.asarray(trial['grad_trajectory'], dtype=np.float64)
                if 'loss_trajectory' in trial:
                    rec['loss_trajectory'] = np.asarray(trial['loss_trajectory'], dtype=np.float64)
                records.append(rec)

    out_path = os.path.join(output_dir, f'trajectories_{tag}.npz')
    np.savez_compressed(
        out_path,
        records=np.array(records, dtype=object),
        param_names=np.array(param_names),
        readme=np.array(_NPZ_README),
    )
    print(f'Saved → {out_path}')
    return out_path


def group_by_gt(all_runs):
    groups = defaultdict(list)
    for r in all_runs:
        groups[_gt_key(r)].append(r)
    return dict(groups)


# ── HTML ──────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Param Trajectories – {tag}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #f8f8f8; color: #222; }}
h2   {{ margin-bottom: 6px; }}
h3   {{ margin: 20px 0 4px 0; font-size: 14px; color: #444; }}
#gt-selector {{ margin: 10px 0 4px 0; padding: 10px 16px; background: #fff;
                border: 1px solid #ccc; border-radius: 6px; display: inline-block; }}
#gt-selector b   {{ font-size: 13px; }}
.gt-label        {{ display: inline-block; margin: 5px 16px 5px 0;
                    font-size: 13px; cursor: pointer; font-weight: 600; }}
.gt-label input  {{ margin-right: 4px; cursor: pointer; }}
.btns            {{ margin-top: 6px; }}
.btns button     {{ margin-right: 6px; padding: 3px 10px; cursor: pointer;
                    font-size: 12px; border: 1px solid #aaa; border-radius: 3px;
                    background: #efefef; }}
.btns button:hover {{ background: #ddd; }}
.pdf-links {{ margin: 6px 0 10px 0; font-size: 13px; }}
.pdf-links a {{ color: #1a6fc4; text-decoration: none; margin-right: 14px; }}
.pdf-links a:hover {{ text-decoration: underline; }}
.back-link {{ display: inline-block; margin-bottom: 8px; font-size: 13px;
              color: #1a6fc4; text-decoration: none; }}
.back-link:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<a class="back-link" href="index.html">&larr; Back to index</a>
<h2>Parameter trajectories – {tag}</h2>
<div id="gt-selector">
  <b>Ground truth:</b><br>
{checkboxes}
  <div class="btns">
    <button onclick="selectAll(true)">All</button>
    <button onclick="selectAll(false)">None</button>
  </div>
</div>
<p class="pdf-links">
  &#128196; <a href="trajectories_{tag}.pdf">Trajectories PDF</a>
  <a href="relative_errors_{tag}.pdf">Relative errors PDF</a>
  <a href="final_distributions_{tag}.pdf">Final distributions PDF</a>
  &#128190; <a href="trajectories_{tag}.npz">Full-resolution data (.npz)</a>
</p>
{traj_div}
<h3>Final-step distributions</h3>
{hist_div}
<script>
var _traceGTs      = {trace_gts_json};
var _traceGTs_hist = {trace_gts_hist_json};
function _applyFilter() {{
  var ok = {{}};
  document.querySelectorAll('.gt-cb').forEach(function(cb) {{ ok[cb.value] = cb.checked; }});
  var vis      = _traceGTs.map(function(g)      {{ return ok[g] ? true : false; }});
  var vis_hist = _traceGTs_hist.map(function(g) {{ return ok[g] ? true : false; }});
  Plotly.restyle('plt',      {{visible: vis}});
  Plotly.restyle('plt_hist', {{visible: vis_hist}});
}}
function selectAll(v) {{
  document.querySelectorAll('.gt-cb').forEach(function(cb) {{ cb.checked = v; }});
  _applyFilter();
}}
document.querySelectorAll('.gt-cb').forEach(function(cb) {{
  cb.addEventListener('change', _applyFilter);
}});
</script>
</body>
</html>
"""


def build_html(grouped, gt_order, gt_labels, gt_colors, param_names, tag, max_points=50):
    if not _PLOTLY_AVAILABLE:
        raise RuntimeError('plotly not installed; pip install plotly')

    # Integer GT index for JS comparison (avoids float serialization issues)
    gt_idx = {k: i for i, k in enumerate(gt_order)}

    n = len(param_names)
    subplot_titles = []
    for pn in param_names:
        lbl = PARAM_LABELS.get(pn, pn)
        subplot_titles += [lbl, f'{lbl} – rel. error']

    fig = make_subplots(
        rows=n, cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=max(0.02, 0.10 / max(n, 1)),
        column_widths=[0.58, 0.42],
    )

    trace_gts = []  # GT index string per trace (for JS filtering)

    for gt_key in gt_order:
        runs   = grouped[gt_key]
        color  = gt_colors[gt_key]
        idx_s  = str(gt_idx[gt_key])

        # Compute max step for GT reference lines (account for the step
        # numbering of single-point/crashed-run trials, in the same units)
        max_step = max(
            (_trial_max_step(t) for r in runs for t in r['trials']),
            default=0,
        )

        for r in runs:
            scales    = r['scales']
            param_gts = r['param_gts']
            for trial in r['trials']:
                for pi, pname in enumerate(param_names):
                    if pname not in r['param_names']:
                        continue
                    ri      = r['param_names'].index(pname)
                    scale   = scales[ri]
                    gt_phys = param_gts[ri]
                    n_steps = len(trial['param_trajectory'])
                    step    = max(1, math.ceil(n_steps / max_points))
                    xs      = _step_axis(trial, step).tolist()
                    phys    = _phys_traj(trial, ri, scale, step).tolist()
                    err     = _rel_err_traj(trial, ri, scale, gt_phys, step).tolist()
                    kw = dict(mode='lines+markers', line=dict(color=color, width=0.8),
                              marker=dict(size=3), opacity=0.55, showlegend=False)
                    fig.add_trace(go.Scatter(x=xs, y=phys, **kw), row=pi + 1, col=1)
                    trace_gts.append(idx_s)
                    fig.add_trace(go.Scatter(x=xs, y=err, **kw),  row=pi + 1, col=2)
                    trace_gts.append(idx_s)

        # GT reference dashed line for value panels
        for pi, pname in enumerate(param_names):
            r0 = next((r for r in runs if pname in r.get('param_names', [])), None)
            if r0 is None:
                continue
            ri0     = r0['param_names'].index(pname)
            gt_phys = r0['param_gts'][ri0]
            fig.add_trace(go.Scatter(
                x=[0, max_step], y=[gt_phys, gt_phys],
                mode='lines', line=dict(color=color, width=1.6, dash='dash'),
                opacity=0.85, showlegend=False,
            ), row=pi + 1, col=1)
            trace_gts.append(idx_s)

    for pi in range(n):
        fig.update_yaxes(type='log', row=pi + 1, col=2)

    fig.update_layout(
        height=340 * n,
        width=1100,
        title=dict(text=f'Parameter trajectories – {tag}', font_size=14),
        template='plotly_white',
        margin=dict(t=60, b=40, l=70, r=20),
    )

    plotly_div = fig.to_html(
        include_plotlyjs=False, full_html=False,
        div_id='plt', config={'responsive': True},
    )

    # ── final-step distribution histograms ──
    hist_titles = []
    for pn in param_names:
        lbl = PARAM_LABELS.get(pn, pn)
        hist_titles += [f'{lbl} – final value', f'{lbl} – final rel. error']

    fig_hist = make_subplots(
        rows=n, cols=2,
        subplot_titles=hist_titles,
        vertical_spacing=max(0.02, 0.10 / max(n, 1)),
        column_widths=[0.58, 0.42],
    )

    trace_gts_hist = []  # GT index string per trace (for JS filtering)

    # first pass: collect data per (GT, param) so bins can be shared across GTs
    hist_data = {}  # (gt_key, pi) -> (final_vals, log_errs, gt_phys_ref)
    for gt_key in gt_order:
        runs = grouped[gt_key]
        for pi, pname in enumerate(param_names):
            final_vals  = []
            final_errs  = []
            gt_phys_ref = None
            for r in runs:
                if pname not in r.get('param_names', []):
                    continue
                ri      = r['param_names'].index(pname)
                scale   = r['scales'][ri]
                gt_phys = r['param_gts'][ri]
                gt_phys_ref = gt_phys
                for trial in r['trials']:
                    final_vals.append(_final_phys(trial, ri, scale))
                    final_errs.append(_final_rel_err(trial, ri, scale, gt_phys))
            log_errs = np.log10(np.array([e for e in final_errs if e > 0])).tolist()
            hist_data[(gt_key, pi)] = (final_vals, log_errs, gt_phys_ref)

    def _common_xbins(all_vals, nbins=30):
        """Shared Plotly xbins covering all GTs' values for one subplot."""
        vals = [v for v in all_vals if np.isfinite(v)]
        if not vals:
            return None
        lo, hi = min(vals), max(vals)
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        size = (hi - lo) / nbins
        return dict(start=lo, end=hi + size, size=size)

    xbins_val = [
        _common_xbins([v for gt_key in gt_order
                       for v in hist_data[(gt_key, pi)][0]])
        for pi in range(n)
    ]
    xbins_err = [
        _common_xbins([v for gt_key in gt_order
                       for v in hist_data[(gt_key, pi)][1]])
        for pi in range(n)
    ]

    for gt_key in gt_order:
        color = gt_colors[gt_key]
        label = gt_labels[gt_key]
        idx_s = str(gt_idx[gt_key])

        for pi in range(n):
            final_vals, log_errs, gt_phys_ref = hist_data[(gt_key, pi)]

            fig_hist.add_trace(go.Histogram(
                x=final_vals, name=label, legendgroup=idx_s,
                marker_color=color, opacity=0.6, showlegend=(pi == 0),
                xbins=xbins_val[pi], autobinx=False,
            ), row=pi + 1, col=1)
            trace_gts_hist.append(idx_s)

            fig_hist.add_trace(go.Histogram(
                x=log_errs, name=label, legendgroup=idx_s,
                marker_color=color, opacity=0.6, showlegend=False,
                xbins=xbins_err[pi], autobinx=False,
            ), row=pi + 1, col=2)
            trace_gts_hist.append(idx_s)

            if gt_phys_ref is not None:
                fig_hist.add_vline(
                    x=gt_phys_ref, row=pi + 1, col=1,
                    line=dict(color=color, width=1.6, dash='dash'),
                )

    for pi in range(n):
        fig_hist.update_xaxes(title_text='log10(rel. error)', row=pi + 1, col=2)

    fig_hist.update_layout(
        height=300 * n,
        width=1100,
        barmode='overlay',
        title=dict(text=f'Final-step distributions – {tag}', font_size=14),
        template='plotly_white',
        margin=dict(t=60, b=40, l=70, r=20),
    )

    hist_div = fig_hist.to_html(
        include_plotlyjs=False, full_html=False,
        div_id='plt_hist', config={'responsive': True},
    )

    checkboxes = ''
    for gt_key in gt_order:
        label = gt_labels[gt_key]
        color = gt_colors[gt_key]
        idx_s = str(gt_idx[gt_key])
        n_runs = len(grouped[gt_key])
        checkboxes += (
            f'  <label class="gt-label" style="color:{color}">'
            f'<input type="checkbox" class="gt-cb" value="{idx_s}" checked>'
            f' {label} ({n_runs} run{"s" if n_runs != 1 else ""})</label>\n'
        )

    return _HTML_TEMPLATE.format(
        tag=tag,
        checkboxes=checkboxes,
        traj_div=plotly_div,
        hist_div=hist_div,
        trace_gts_json=json.dumps(trace_gts),
        trace_gts_hist_json=json.dumps(trace_gts_hist),
    )


# ── PDF ───────────────────────────────────────────────────────────────────────

def build_pdfs(grouped, gt_order, gt_labels, gt_colors, param_names, output_dir, tag):
    os.makedirs(output_dir, exist_ok=True)
    ncols = min(len(param_names), 4)
    nrows = math.ceil(len(param_names) / ncols)

    for mode, suffix in [('value', 'trajectories'), ('relerr', 'relative_errors')]:
        out_path = os.path.join(output_dir, f'{suffix}_{tag}.pdf')
        title_str = ('Parameter trajectories' if mode == 'value' else 'Relative errors')
        title_str += f'  [{tag}]'

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 4 * nrows), squeeze=False)

        for idx, pname in enumerate(param_names):
            row, col = divmod(idx, ncols)
            ax       = axes[row][col]
            plabel   = PARAM_LABELS.get(pname, pname)
            seen_gt  = set()

            for gt_key in gt_order:
                runs  = grouped[gt_key]
                color = gt_colors[gt_key]
                label = gt_labels[gt_key]

                # GT reference line (value panels only)
                if mode == 'value':
                    r0 = next((r for r in runs if pname in r.get('param_names', [])), None)
                    if r0 is not None:
                        ri0 = r0['param_names'].index(pname)
                        ax.axhline(r0['param_gts'][ri0], color=color,
                                   ls='--', lw=1.2, alpha=0.75)

                for r in runs:
                    if pname not in r.get('param_names', []):
                        continue
                    pi      = r['param_names'].index(pname)
                    scale   = r['scales'][pi]
                    gt_phys = r['param_gts'][pi]
                    for trial in r['trials']:
                        lbl   = label if gt_key not in seen_gt else '_nolegend_'
                        steps = _step_axis(trial)
                        y     = (_phys_traj(trial, pi, scale)
                                 if mode == 'value' else
                                 _rel_err_traj(trial, pi, scale, gt_phys))
                        ax.plot(steps, y, color=color, alpha=TRIAL_ALPHA,
                                lw=TRIAL_LW, marker='.', markersize=3, label=lbl)
                        seen_gt.add(gt_key)

            if mode == 'relerr':
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
                ax.set_ylabel('relative error', fontsize=9)
            else:
                ax.set_ylabel(plabel, fontsize=9)
            ax.set_title(plabel, fontsize=10)
            ax.set_xlabel('iteration', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.2)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8, handlelength=1.2)

        for idx in range(len(param_names), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle(title_str, fontsize=12, y=1.01)
        fig.tight_layout()
        with PdfPages(out_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved → {out_path}')


def build_hist_pdf(grouped, gt_order, gt_labels, gt_colors, param_names, output_dir, tag):
    """final_distributions_{tag}.pdf — histograms of final-step values + rel. errors."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'final_distributions_{tag}.pdf')
    n = len(param_names)

    fig, axes = plt.subplots(n, 2, figsize=(10, 3.2 * n), squeeze=False)

    for pi, pname in enumerate(param_names):
        plabel = PARAM_LABELS.get(pname, pname)
        ax_val, ax_err = axes[pi]

        # collect per-GT data first so all GTs share the same bin edges
        per_gt = {}  # gt_key -> (final_vals, log_errs, gt_phys_ref)
        for gt_key in gt_order:
            runs = grouped[gt_key]
            final_vals  = []
            final_errs  = []
            gt_phys_ref = None
            for r in runs:
                if pname not in r.get('param_names', []):
                    continue
                ri      = r['param_names'].index(pname)
                scale   = r['scales'][ri]
                gt_phys = r['param_gts'][ri]
                gt_phys_ref = gt_phys
                for trial in r['trials']:
                    final_vals.append(_final_phys(trial, ri, scale))
                    final_errs.append(_final_rel_err(trial, ri, scale, gt_phys))
            log_errs = np.log10(np.array([e for e in final_errs if e > 0]))
            per_gt[gt_key] = (final_vals, log_errs, gt_phys_ref)

        all_vals = [v for vals, _, _ in per_gt.values() for v in vals if np.isfinite(v)]
        all_errs = np.concatenate([errs for _, errs, _ in per_gt.values()]) \
            if per_gt else np.array([])
        val_bins = np.histogram_bin_edges(all_vals, bins=15) if all_vals else 15
        err_bins = np.histogram_bin_edges(all_errs, bins=15) if len(all_errs) else 15

        for gt_key in gt_order:
            color = gt_colors[gt_key]
            label = gt_labels[gt_key]
            final_vals, log_errs, gt_phys_ref = per_gt[gt_key]

            if final_vals:
                ax_val.hist(final_vals, bins=val_bins, color=color, alpha=0.5, label=label)
            if gt_phys_ref is not None:
                ax_val.axvline(gt_phys_ref, color=color, ls='--', lw=1.4, alpha=0.85)

            if len(log_errs):
                ax_err.hist(log_errs, bins=err_bins, color=color, alpha=0.5, label=label)

        ax_val.set_title(f'{plabel} – final value', fontsize=10)
        ax_val.set_xlabel(plabel, fontsize=8)
        ax_val.set_ylabel('count', fontsize=8)
        ax_val.tick_params(labelsize=8)
        ax_val.grid(True, alpha=0.2)
        if ax_val.get_legend_handles_labels()[0]:
            ax_val.legend(fontsize=8)

        ax_err.set_title(f'{plabel} – final rel. error', fontsize=10)
        ax_err.set_xlabel('log10(rel. error)', fontsize=8)
        ax_err.set_ylabel('count', fontsize=8)
        ax_err.tick_params(labelsize=8)
        ax_err.grid(True, alpha=0.2)

    fig.suptitle(f'Final-step distributions  [{tag}]', fontsize=12, y=1.01)
    fig.tight_layout()
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag',        required=True, help='W&B tag (e.g. Run_Opt_20260609)')
    p.add_argument('--entity',     default='fcc_ml')
    p.add_argument('--project',    default='jaxtpc-optimization')
    p.add_argument('--output-dir', default=None,
                   help='Output directory (default: $PLOTS_DIR/trajectories_{tag})')
    p.add_argument('--out-tag',    default=None,
                   help='Tag used for output filenames/titles (default: --tag). '
                        'Useful with --exclude-variant to write a filtered subset '
                        'under a different name.')
    p.add_argument('--exclude-variant', default='',
                   help='Comma-separated config-variant tags to drop from the plot '
                        '(e.g. phase2_diffusion); see _VARIANT_TAGS')
    p.add_argument('--exclude-gt-label', action='append', default=[],
                   help='Drop GT groups whose label starts with this string '
                        '(can be passed multiple times). Match against the label '
                        'printed in the "GT groups (...)" summary, e.g. '
                        '"velocity=0.16, lifetime=1e+04, diffusion=1.2e-05"')
    p.add_argument('--max-points', type=int, default=50,
                   help='Max points per trajectory in HTML, adaptively subsampled '
                        '(default 50; PDF uses all steps)')
    p.add_argument('--no-html',    action='store_true')
    p.add_argument('--no-pdf',     action='store_true')
    p.add_argument('--no-npz',     action='store_true',
                   help='Skip writing trajectories_{tag}.npz with full-resolution data')
    args = p.parse_args()

    out_tag = args.out_tag or args.tag
    output_dir = args.output_dir or os.path.join(
        os.environ.get('PLOTS_DIR', 'plots'), f'trajectories_{out_tag}'
    )

    print(f"Fetching runs from {args.entity}/{args.project} with tag '{args.tag}'...")
    run_infos = fetch_runs_from_wandb(args.tag, args.entity, args.project)
    if not run_infos:
        print('No runs found.')
        return

    all_runs = load_pkls(run_infos)
    if not all_runs:
        print('No pkl files loaded.')
        return

    if args.exclude_variant:
        excl = {t.strip() for t in args.exclude_variant.split(',') if t.strip()}
        before = len(all_runs)
        all_runs = [r for r in all_runs if _variant(r) not in excl]
        print(f'  excluded {before - len(all_runs)} run(s) with variant in {sorted(excl)}')
        if not all_runs:
            print('No runs left after exclusion.')
            return

    grouped  = group_by_gt(all_runs)
    gt_order = sorted(grouped.keys(), key=lambda k: _gt_label(grouped[k][0]))
    gt_labels = {k: _gt_label(grouped[k][0]) for k in gt_order}

    if args.exclude_gt_label:
        before = len(gt_order)
        gt_order = [k for k in gt_order
                    if not any(gt_labels[k].startswith(s) for s in args.exclude_gt_label)]
        removed = before - len(gt_order)
        if removed:
            print(f'  excluded {removed} GT group(s) matching {args.exclude_gt_label}')

    gt_colors = {k: QUALITATIVE_COLORS[i % len(QUALITATIVE_COLORS)]
                 for i, k in enumerate(gt_order)}

    print(f'GT groups ({len(gt_order)}):')
    for k in gt_order:
        print(f'  {gt_labels[k]!r}: {len(grouped[k])} run(s)')

    param_names = all_runs[0]['param_names']
    os.makedirs(output_dir, exist_ok=True)

    if not args.no_html:
        print('Building HTML...')
        html = build_html(grouped, gt_order, gt_labels, gt_colors,
                          param_names, out_tag, max_points=args.max_points)
        html_path = os.path.join(output_dir, f'trajectories_{out_tag}.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f'Saved → {html_path}')

    if not args.no_pdf:
        print('Building PDFs...')
        build_pdfs(grouped, gt_order, gt_labels, gt_colors,
                   param_names, output_dir, out_tag)
        build_hist_pdf(grouped, gt_order, gt_labels, gt_colors,
                       param_names, output_dir, out_tag)

    if not args.no_npz:
        print('Building npz...')
        build_npz(grouped, gt_order, gt_labels, param_names, output_dir, out_tag)

    print('Done.')


if __name__ == '__main__':
    main()
