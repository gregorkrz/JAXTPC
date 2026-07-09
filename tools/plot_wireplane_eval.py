#!/usr/bin/env python3
"""
Build a lazy-loading interactive HTML viewer for *_wireplanes.pkl files
(the output of tools/eval_efield_track_wireplanes.py): a dropdown to pick a
run+track, a dropdown to pick an MLP snapshot step, and a 3-row grid of
zoomable 2D plots — row 1 = learned (simulation), row 2 = GT (includes the
same detector noise draw training compared against), row 3 = simulation −
GT (computed client-side, own color scale) — one column per (volume, plane).

Each wireplane array is huge (num_wires x num_time, e.g. ~2000x2700). To keep
this "lazy-loading, no giant HTML" as requested while still rendering a proper
2D grid (not a scatter of points), each plot is a Plotly `heatmap` cropped to
a padded bounding box around the hits — full per-cell resolution, but only
over the small region that's actually populated (e.g. ~150x300 cells instead
of ~2000x2700), which is what keeps the per-step payload small. The bbox (and
the sim/GT color range) is computed from the LEARNED arrays only, never GT:
GT has real detector noise added everywhere (~100% non-zero), so using it for
cropping/ranging would blow both back up to full-plane size.

Design choice made without a reply to a clarifying question (documented here
so it's easy to revisit): one flat "run" dropdown entry per (source PKL,
track) pair, i.e. exactly matching the original 2-dropdown ask, rather than
splitting run/track into separate dropdowns.

All 3 subplots in a column share their axes (Plotly `matches`), so zooming or
panning any one of them keeps the others in sync.

Data layout (mirrors plot_efield_eval.py's lazy-loading convention: separate
.js files that do `window.KEY = {...}`, dynamically <script>-injected on
demand and cached, rather than embedding data in the HTML):
  <output>                              — the HTML shell + embedded RUN_META
                                           (labels, per-plane bbox/color-range,
                                           step list — all small; NOT signal data)
  <output_stem>_data/run_{i}/gt.js      — GT cropped grids, one per run (fetched once)
  <output_stem>_data/run_{i}/step_{k}.js — learned cropped grids for step k
                                           (fetched only when that step is viewed)
  <output_stem>_data/run_{i}/{sim,gt,diff}.gif
                                         — one downloadable animation per row (all 6
                                           planes side by side, 500ms/step, step number
                                           in the title); download links in the page
                                           update to the selected run's 3 GIFs.

Usage
-----
  python tools/plot_wireplane_eval.py --results-dir $RESULTS_DIR/opt/efield_calib/1k_tracks_sweep
  python tools/plot_wireplane_eval.py PKL [PKL ...] --output plots/wireplanes.html
"""
import argparse
import glob
import io
import json
import os
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT))

_VOL_NAME = {0: 'East', 1: 'West'}
_BBOX_PAD_FRAC = 0.08
_BBOX_PAD_MIN = 5
_VALUE_DECIMALS = 4
_GIF_MS_PER_STEP = 500


def _sanitize(s):
    return re.sub(r'[^0-9A-Za-z_]+', '_', str(s))


def _common_root(paths):
    try:
        return Path(os.path.commonpath([str(Path(p).resolve()) for p in paths]))
    except ValueError:
        return None


def _nonzero_points(arr):
    wire_idx, time_idx = np.nonzero(arr != 0)
    return wire_idx, time_idx, arr[wire_idx, time_idx]


def _padded_bbox(coord_lists, shape):
    """coord_lists: list of (wire_idx, time_idx) arrays (possibly empty). Returns
    [w0, w1, t0, t1] — a padded bounding box over their union, clipped to shape."""
    wires = [w for w, _ in coord_lists if w.size]
    times = [t for _, t in coord_lists if t.size]
    if not wires:
        return [0, shape[0] - 1, 0, shape[1] - 1]
    w = np.concatenate(wires)
    t = np.concatenate(times)
    w0, w1 = int(w.min()), int(w.max())
    t0, t1 = int(t.min()), int(t.max())
    wpad = max(_BBOX_PAD_MIN, int((w1 - w0) * _BBOX_PAD_FRAC))
    tpad = max(_BBOX_PAD_MIN, int((t1 - t0) * _BBOX_PAD_FRAC))
    return [
        max(0, w0 - wpad), min(shape[0] - 1, w1 + wpad),
        max(0, t0 - tpad), min(shape[1] - 1, t1 + tpad),
    ]


def _grid_json(arr, bbox):
    """Dense (small, cropped-to-bbox) 2D grid — z[wire-row][time-col] — for a Plotly heatmap."""
    w0, w1, t0, t1 = bbox
    sub = arr[w0:w1 + 1, t0:t1 + 1]
    return np.round(sub, _VALUE_DECIMALS).tolist()


def _write_js(path, var_name, obj):
    path.write_text(f'window.{var_name}=' + json.dumps(obj, separators=(',', ':')) + ';',
                     encoding='utf-8')


def _render_row_gif(path, label_strs, bbox, vmin, vmax, frames, step_labels, row_title):
    """Animated GIF: one frame per step, 6 side-by-side plane subplots per frame.

    frames : list (over steps) of dict{label_str: 2D np.ndarray} (already cropped to bbox).
    step_labels : list of str, aligned with frames, shown in the frame title.
    """
    n = len(label_strs)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.2))
    if n == 1:
        axes = [axes]
    ims = []
    for ax, lstr in zip(axes, label_strs):
        w0, w1, t0, t1 = bbox[lstr]
        im = ax.imshow(frames[0][lstr], cmap='RdBu', vmin=vmin[lstr], vmax=vmax[lstr],
                        origin='lower', aspect='auto', extent=[t0, t1, w0, w1])
        ax.set_title(lstr, fontsize=9)
        ax.set_xlabel('time tick', fontsize=7)
        ax.tick_params(labelsize=6)
        ims.append(im)
    axes[0].set_ylabel('wire', fontsize=7)
    suptitle = fig.suptitle('', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    pil_frames = []
    for frame, step_label in zip(frames, step_labels):
        for im, lstr in zip(ims, label_strs):
            im.set_data(frame[lstr])
        suptitle.set_text(f'{row_title} — step {step_label}')
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        pil_frames.append(Image.open(buf).convert('P', palette=Image.ADAPTIVE))

    plt.close(fig)
    pil_frames[0].save(
        path, save_all=True, append_images=pil_frames[1:],
        duration=_GIF_MS_PER_STEP, loop=0,
    )


# ── Per-PKL processing ────────────────────────────────────────────────────────

def process_wireplane_pkl(pkl_path, run_idx, out_stem, data_root, label_root):
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)

    labels = d['labels']  # [(vol_idx, plane_idx, plane_name), ...]
    label_strs = [f'{_VOL_NAME.get(v, f"vol{v}")}-{name}' for v, _, name in labels]
    gt = d['gt']
    steps_data = d['steps']

    run_dir = data_root / f'run_{run_idx}'
    run_dir.mkdir(parents=True, exist_ok=True)

    bbox, vrange, diff_vrange = {}, {}, {}
    gt_json = {}
    gt_crops, learned_crops_by_step, vmin_by_label, vmax_by_label = {}, [{} for _ in steps_data], {}, {}
    for li, lstr in enumerate(label_strs):
        arr_gt = np.asarray(gt[li])
        learned_arrs = [np.asarray(s['learned'][li]) for s in steps_data]

        # bbox from the LEARNED arrays only: GT gets real detector noise added
        # (apply_noise_to_gt, matching what training compared against) which fills
        # ~100% of the plane, so including it here would defeat cropping to the hit
        # region. The learned/diff-sim output is never noised, so it stays sparse
        # and reliably marks where the physics response actually lives.
        coord_lists = []
        for a in learned_arrs:
            wi, ti, _ = _nonzero_points(a)
            coord_lists.append((wi, ti))
        bbox[lstr] = _padded_bbox(coord_lists, arr_gt.shape)
        w0, w1, t0, t1 = bbox[lstr]

        # Color ranges computed only over the cropped region actually displayed,
        # so full-plane GT noise outside the crop doesn't skew the scale.
        gt_crop = arr_gt[w0:w1 + 1, t0:t1 + 1]
        gt_crops[lstr] = gt_crop
        vmin, vmax = float(gt_crop.min()), float(gt_crop.max())
        dmax = 0.0
        for step_idx, a in enumerate(learned_arrs):
            crop = a[w0:w1 + 1, t0:t1 + 1]
            learned_crops_by_step[step_idx][lstr] = crop
            vmin = min(vmin, float(crop.min()))
            vmax = max(vmax, float(crop.max()))
            dmax = max(dmax, float(np.abs(crop - gt_crop).max()))
        if vmin < 0.0 < vmax:
            vabs = max(abs(vmin), abs(vmax))
            vrange[lstr] = [-vabs, vabs]
        else:
            vrange[lstr] = [vmin, vmax]
        diff_vrange[lstr] = [-dmax, dmax] if dmax > 0 else [-1.0, 1.0]
        vmin_by_label[lstr], vmax_by_label[lstr] = vrange[lstr]

        gt_json[lstr] = _grid_json(arr_gt, bbox[lstr])

    gt_key = f'__WP_GT_{run_idx}__'
    _write_js(run_dir / 'gt.js', gt_key, gt_json)

    steps_meta = []
    for s in steps_data:
        step = s['step']
        step_key_frag = _sanitize(step)
        step_json = {lstr: _grid_json(np.asarray(arr), bbox[lstr])
                     for lstr, arr in zip(label_strs, s['learned'])}
        var_name = f'__WP_STEP_{run_idx}_{step_key_frag}__'
        _write_js(run_dir / f'step_{step_key_frag}.js', var_name, step_json)
        steps_meta.append({'step': step, 'label': s['label'], 'key': step_key_frag})

    # ── Downloadable per-row animations (500ms/step, step number in the title) ──
    step_num_labels = [str(s['step']) for s in steps_data]
    gt_frames = [gt_crops] * len(steps_data)  # GT itself doesn't change across steps
    diff_frames = [
        {lstr: learned_crops_by_step[si][lstr] - gt_crops[lstr] for lstr in label_strs}
        for si in range(len(steps_data))
    ]
    diff_vmin = {lstr: diff_vrange[lstr][0] for lstr in label_strs}
    diff_vmax = {lstr: diff_vrange[lstr][1] for lstr in label_strs}

    gif_paths = {}
    for key, frames, vmin_d, vmax_d, title in [
        ('sim',  learned_crops_by_step, vmin_by_label, vmax_by_label, 'Simulation (learned)'),
        ('gt',   gt_frames,             vmin_by_label, vmax_by_label, 'GT'),
        ('diff', diff_frames,           diff_vmin,     diff_vmax,     'Difference (Sim - GT)'),
    ]:
        gif_path = run_dir / f'{key}.gif'
        _render_row_gif(gif_path, label_strs, bbox, vmin_d, vmax_d, frames, step_num_labels, title)
        gif_paths[key] = f'{out_stem}_data/run_{run_idx}/{key}.gif'
        print(f'    wrote {gif_path.name} ({len(steps_data)} frames)')

    ts = d['track_spec']
    try:
        rel_dir = Path(pkl_path).parent.relative_to(label_root)
        rel_str = str(rel_dir) if str(rel_dir) != '.' else Path(pkl_path).parent.name
    except ValueError:
        rel_str = Path(pkl_path).parent.name
    label = f"{rel_str} — track{d['track_idx']} ({ts['name']})"

    return {
        'label':           label,
        'source_pkl':      str(pkl_path),
        'labels':          label_strs,
        'bbox':            bbox,
        'vrange':          vrange,
        'diff_vrange':     diff_vrange,
        'steps':           steps_meta,
        'gt_key':          gt_key,
        'gt_file':         f'{out_stem}_data/run_{run_idx}/gt.js',
        'step_dir':        f'{out_stem}_data/run_{run_idx}',
        'step_key_prefix': f'__WP_STEP_{run_idx}_',
        'gif':             gif_paths,  # {'sim': path, 'gt': path, 'diff': path}
    }


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Wireplane eval — GT vs learned</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; padding: 12px;
       background: #f0f0f0; color: #222; }
h2 { margin: 0 0 10px; font-size: 16px; }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-end;
            background: #fff; padding: 10px 14px; border-radius: 8px;
            margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.cg { display: flex; flex-direction: column; gap: 3px; }
.cg label { font-size: 11px; color: #666; font-weight: 600; }
.cg select { padding: 5px 8px; border-radius: 5px; border: 1px solid #ccc;
             background: #fafafa; font-size: 13px; }
#loading-ind { display:none; padding: 4px 12px; background: #fff3cd;
               border: 1px solid #ffc107; border-radius: 5px;
               font-size: 12px; align-self: center; }
.panel { background: #fff; border-radius: 8px; padding: 10px 12px;
         box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.hint { font-size: 11px; color: #888; margin: 0 0 8px; }
.downloads { display: flex; gap: 8px; align-items: center; }
.downloads a { font-size: 12px; padding: 5px 10px; border-radius: 5px;
               border: 1px solid #cce; background: #f0f7ff; color: #558;
               text-decoration: none; }
.downloads a:hover { background: #e0efff; }
</style>
</head>
<body>
<h2>Wireplane eval — learned (simulation) vs GT</h2>
<div class="controls">
  <div class="cg">
    <label>Run + track</label>
    <select id="run-sel" style="max-width:520px" onchange="onRunChange()"></select>
  </div>
  <div class="cg">
    <label>Step</label>
    <select id="step-sel" onchange="render()"></select>
  </div>
  <div id="loading-ind">Loading…</div>
  <div class="cg">
    <label>Download animation (500ms/step)</label>
    <div class="downloads">
      <a id="dl-sim"  download>Simulation ↓</a>
      <a id="dl-gt"   download>GT ↓</a>
      <a id="dl-diff" download>Difference ↓</a>
    </div>
  </div>
</div>
<div class="panel">
  <p class="hint">Each plot is a 2D grid cropped to a padded box around the hits (consistent across
  steps for a given plane). Box-zoom / double-click-to-reset syncs across all 3 subplots in a
  column. Row 1 = learned (simulation) at the selected step; row 2 = GT (includes the same
  detector noise draw training compared against, so it doesn't change across steps); row 3 =
  simulation − GT (its own color scale, since the residual is typically much smaller than the
  raw signal). The download links above give an animated GIF per row (all 6 planes, 500ms/step,
  step number in the title) for the currently selected run.</p>
  <div id="grid-div"></div>
</div>
<script>
const RUN_META = /*RUN_META*/null/*END_RUN_META*/;

const gtCache = {};
const stepCache = {};

function loadScript(src, key) {
  return new Promise((resolve) => {
    if (window[key] !== undefined) { resolve(window[key]); return; }
    const el = document.createElement('script');
    el.src = src;
    el.onload  = () => resolve(window[key]);
    el.onerror = () => resolve(null);
    document.head.appendChild(el);
  });
}

async function loadGt(ri) {
  if (gtCache[ri]) return gtCache[ri];
  const meta = RUN_META[ri];
  const data = await loadScript(meta.gt_file, meta.gt_key);
  gtCache[ri] = data;
  return data;
}

async function loadStep(ri, stepEntry) {
  const cacheKey = ri + ':' + stepEntry.key;
  if (stepCache[cacheKey]) return stepCache[cacheKey];
  const meta = RUN_META[ri];
  const src = `${meta.step_dir}/step_${stepEntry.key}.js`;
  const key = `${meta.step_key_prefix}${stepEntry.key}__`;
  const data = await loadScript(src, key);
  stepCache[cacheKey] = data;
  return data;
}

function populateRunSelect() {
  const sel = document.getElementById('run-sel');
  RUN_META.forEach((m, i) => {
    const o = document.createElement('option');
    o.value = i; o.text = m.label;
    sel.appendChild(o);
  });
}

function populateStepSelect(ri) {
  const meta = RUN_META[ri];
  const sel = document.getElementById('step-sel');
  sel.innerHTML = '';
  meta.steps.forEach((s, i) => {
    const o = document.createElement('option');
    o.value = i; o.text = `${s.step}`;
    sel.appendChild(o);
  });
  sel.value = meta.steps.length - 1;  // default: most-trained snapshot
}

function axisKey(prefix, idx) {
  return idx === 1 ? prefix : `${prefix}${idx}`;
}

async function render() {
  const ri = +document.getElementById('run-sel').value;
  const si = +document.getElementById('step-sel').value;
  const meta = RUN_META[ri];
  const stepEntry = meta.steps[si];
  if (!stepEntry) return;

  document.getElementById('loading-ind').style.display = 'block';
  const [gt, learned] = await Promise.all([loadGt(ri), loadStep(ri, stepEntry)]);
  document.getElementById('loading-ind').style.display = 'none';
  if (!gt || !learned) {
    document.getElementById('grid-div').innerHTML = '<p style="color:#a55">Failed to load data.</p>';
    return;
  }

  const n = meta.labels.length;
  const nRows = 3;
  const traces = [];
  const layout = {
    grid: {rows: nRows, columns: n, pattern: 'independent'},
    height: 900,
    margin: {l: 55, r: 20, t: 40, b: 45},
    paper_bgcolor: '#fff',
    plot_bgcolor: '#fff',
    showlegend: false,
    annotations: [
      {text: 'Simulation (learned)', xref: 'paper', yref: 'paper',
       x: -0.06, y: 0.85, xanchor: 'right', textangle: -90, showarrow: false,
       font: {size: 12, color: '#444'}},
      {text: 'GT', xref: 'paper', yref: 'paper',
       x: -0.06, y: 0.5, xanchor: 'right', textangle: -90, showarrow: false,
       font: {size: 12, color: '#444'}},
      {text: 'Difference (Sim − GT)', xref: 'paper', yref: 'paper',
       x: -0.06, y: 0.15, xanchor: 'right', textangle: -90, showarrow: false,
       font: {size: 12, color: '#444'}},
    ],
  };

  meta.labels.forEach((lstr, ci) => {
    const [w0, w1, t0, t1] = meta.bbox[lstr];
    const [vmin, vmax] = meta.vrange[lstr];
    const [dmin, dmax] = meta.diff_vrange[lstr];
    const topIdx = ci + 1;
    const midIdx = ci + 1 + n;
    const botIdx = ci + 1 + 2 * n;
    const xTop = axisKey('x', topIdx), yTop = axisKey('y', topIdx);
    const xMid = axisKey('x', midIdx), yMid = axisKey('y', midIdx);
    const xBot = axisKey('x', botIdx), yBot = axisKey('y', botIdx);

    const xs = []; for (let i = t0; i <= t1; i++) xs.push(i);
    const ys = []; for (let i = w0; i <= w1; i++) ys.push(i);

    const learnedZ = (learned && learned[lstr]) || [[0]];
    const gtZ = (gt && gt[lstr]) || [[0]];
    const diffZ = learnedZ.map((row, ri) => row.map((v, ci2) => v - (gtZ[ri] ? gtZ[ri][ci2] : 0)));

    function mkTrace(z, xaxis, yaxis, zmin, zmax, rowY) {
      return {
        type: 'heatmap', z, x: xs, y: ys,
        zmin, zmax, colorscale: 'RdBu',
        showscale: ci === n - 1,
        colorbar: ci === n - 1 ? {len: 0.28, thickness: 10, y: rowY} : undefined,
        xaxis, yaxis,
        hovertemplate: 'wire:%{y}<br>time:%{x}<br>val:%{z:.3f}<extra></extra>',
      };
    }

    traces.push(mkTrace(learnedZ, xTop, yTop, vmin, vmax, 0.85));
    traces.push(mkTrace(gtZ, xMid, yMid, vmin, vmax, 0.5));
    traces.push(mkTrace(diffZ, xBot, yBot, dmin, dmax, 0.15));

    // Mid (GT) and bottom (diff) axes `matches` the top (simulation) axes for this
    // column, so zooming/panning any of the 3 subplots keeps all of them in sync.
    layout[axisKey('xaxis', topIdx)] = {range: [t0, t1], showticklabels: false, title: {text: lstr, font: {size: 11}}};
    layout[axisKey('yaxis', topIdx)] = {range: [w0, w1]};
    layout[axisKey('xaxis', midIdx)] = {range: [t0, t1], matches: xTop, showticklabels: false};
    layout[axisKey('yaxis', midIdx)] = {range: [w0, w1], matches: yTop};
    layout[axisKey('xaxis', botIdx)] = {range: [t0, t1], matches: xTop, title: {text: 'time tick', font: {size: 10}}};
    layout[axisKey('yaxis', botIdx)] = {range: [w0, w1], matches: yTop};
  });

  Plotly.newPlot('grid-div', traces, layout, {displayModeBar: true, responsive: false});
}

function updateDownloadLinks(ri) {
  const gif = RUN_META[ri].gif || {};
  document.getElementById('dl-sim').href  = gif.sim  || '#';
  document.getElementById('dl-gt').href   = gif.gt   || '#';
  document.getElementById('dl-diff').href = gif.diff || '#';
}

async function onRunChange() {
  const ri = +document.getElementById('run-sel').value;
  populateStepSelect(ri);
  updateDownloadLinks(ri);
  await render();
}

populateRunSelect();
onRunChange();
</script>
</body>
</html>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('wireplane_pkls', nargs='*',
                        help='*_wireplanes.pkl files or directories to include')
    parser.add_argument('--results-dir', nargs='+', default=[], metavar='DIR',
                        help='Scan recursively for *_wireplanes.pkl (repeatable)')
    parser.add_argument('--output', default='wireplane_eval.html',
                        help='Output HTML path (default: wireplane_eval.html)')
    parser.add_argument('--max-runs', type=int, default=None, metavar='N',
                        help='Only include the first N *_wireplanes.pkl files found (sorted), '
                             'instead of every one under --results-dir/positional dirs. '
                             'Useful to cap how big the generated viewer is.')
    args = parser.parse_args()

    pkls = []
    for p in args.wireplane_pkls:
        if os.path.isdir(p):
            pkls += sorted(glob.glob(os.path.join(p, '**', '*_wireplanes.pkl'), recursive=True))
        else:
            pkls.append(p)
    for d in args.results_dir:
        pkls += sorted(glob.glob(os.path.join(d, '**', '*_wireplanes.pkl'), recursive=True))
    pkls = sorted(set(pkls))

    if not pkls:
        sys.exit('No *_wireplanes.pkl files found.')

    if args.max_runs is not None:
        print(f'Found {len(pkls)} run(s); limiting to the first {args.max_runs} (--max-runs).')
        pkls = pkls[:args.max_runs]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    data_root = out.parent / f'{out.stem}_data'
    data_root.mkdir(parents=True, exist_ok=True)
    label_root = _common_root(pkls) or Path(pkls[0]).parent

    run_meta = []
    for i, pkl in enumerate(pkls):
        print(f'Processing {pkl}')
        try:
            run_meta.append(process_wireplane_pkl(pkl, i, out.stem, data_root, label_root))
            print(f'  -> {run_meta[-1]["label"]}  ({len(run_meta[-1]["steps"])} step(s), '
                  f'{len(run_meta[-1]["labels"])} plane(s))')
        except Exception as exc:
            import traceback
            print(f'  ERROR: {exc}')
            traceback.print_exc()

    if not run_meta:
        sys.exit('Nothing to plot.')

    html = _HTML.replace('/*RUN_META*/null/*END_RUN_META*/',
                          json.dumps(run_meta, separators=(',', ':')))
    out.write_text(html, encoding='utf-8')
    print(f'\nWrote {out}')
    print(f'Data files under {data_root}/')


if __name__ == '__main__':
    main()
