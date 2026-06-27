#!/usr/bin/env python3
"""
Generate an interactive HTML visualization from *_efield_eval.pkl files.

Quantities are organized by type (E-field / Corrections / Potential) and
component (Ex/Ey/Ez/|E|, dx/dy/dz/|d|, phi).  For potential and efield
modes the corrections are derived from the learned E-field via Euler
integration, so all three types are available regardless of MLP mode.

Usage
-----
  python tools/plot_efield_eval.py --results-dir $RESULTS_DIR/opt/E_debug
  python tools/plot_efield_eval.py path/to/*_efield_eval.pkl --output out.html
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np


# ── Data helpers ──────────────────────────────────────────────────────────────

def _axes(grid_side):
    o, s, sh = grid_side['origin_cm'], grid_side['spacing_cm'], grid_side['shape']
    xs = (o[0] + np.arange(sh[0]) * s[0]).round(3).tolist()
    ys = (o[1] + np.arange(sh[1]) * s[1]).round(3).tolist()
    zs = (o[2] + np.arange(sh[2]) * s[2]).round(3).tolist()
    return xs, ys, zs


def _slices(arr3d, xs, ys, zs):
    """Three central-plane slices. z[row][col] for Plotly: x=cols, y=rows."""
    a = np.asarray(arr3d, dtype=np.float32)
    Nx, Ny, Nz = a.shape
    return {
        'xy': {'z': a[:, :, Nz // 2].T.round(4).tolist(),
               'x': xs, 'xlabel': 'x [cm]', 'y': ys, 'ylabel': 'y [cm]'},
        'xz': {'z': a[:, Ny // 2, :].T.round(4).tolist(),
               'x': xs, 'xlabel': 'x [cm]', 'y': zs, 'ylabel': 'z [cm]'},
        'yz': {'z': a[Nx // 2, :, :].T.round(4).tolist(),
               'x': ys, 'xlabel': 'y [cm]', 'y': zs, 'ylabel': 'z [cm]'},
    }


def _vminmax(gt_arr, learned_arr):
    vals = []
    for a in (gt_arr, learned_arr):
        if a is not None:
            vals.extend([float(np.nanmin(a)), float(np.nanmax(a))])
    if not vals:
        return None, None
    return min(vals), max(vals)


def _reldiff(cg, cl, eps_frac=1e-3):
    if cg is None:
        return None
    eps = eps_frac * float(np.abs(cg).max()) + 1e-30
    return (cg - cl) / (np.abs(cg) + eps)


def _qty(cl, cg, xs, ys, zs):
    cd = (cg - cl) if cg is not None else None
    cr = _reldiff(cg, cl)
    vmin_l, vmax_l = _vminmax(None, cl)
    vmin_g, vmax_g = _vminmax(cg, None)
    hist_reldiff = None
    if cr is not None:
        counts, edges = np.histogram(np.clip(cr.ravel(), -3.0, 3.0), bins=80)
        hist_reldiff = {'counts': counts.tolist(),
                        'edges':  edges[:-1].round(5).tolist(),
                        'width':  round(float(edges[1] - edges[0]), 6)}
    return {
        'has_gt':       cg is not None,
        'learned':      _slices(cl, xs, ys, zs),
        'gt':           _slices(cg, xs, ys, zs) if cg is not None else None,
        'diff':         _slices(cd, xs, ys, zs) if cd is not None else None,
        'reldiff':      _slices(cr, xs, ys, zs) if cr is not None else None,
        'hist_reldiff': hist_reldiff,
        'vmin_l': vmin_l, 'vmax_l': vmax_l,
        'vmin_g': vmin_g, 'vmax_g': vmax_g,
    }


def _corrections_3d_data(co_arr, xs, ys, zs, step=2):
    """Return subsampled grid + correction values for Plotly Scatter3d."""
    co = np.asarray(co_arr, dtype=np.float32)
    xs_s = np.array(xs)[::step]
    ys_s = np.array(ys)[::step]
    zs_s = np.array(zs)[::step]
    co_s = co[::step, ::step, ::step]
    XG, YG, ZG = np.meshgrid(xs_s, ys_s, zs_s, indexing='ij')
    return {
        'x':  XG.ravel().round(2).tolist(),
        'y':  YG.ravel().round(2).tolist(),
        'z':  ZG.ravel().round(2).tolist(),
        'dx': co_s[..., 0].ravel().round(3).tolist(),
        'dy': co_s[..., 1].ravel().round(3).tolist(),
        'dz': co_s[..., 2].ravel().round(3).tolist(),
    }


def _efield_3d_data(ef_arr, xs, ys, zs, step=5):
    """Return subsampled grid + E-field vectors for Plotly cone trace."""
    ef = np.asarray(ef_arr, dtype=np.float32)
    xs_s = np.array(xs)[::step]
    ys_s = np.array(ys)[::step]
    zs_s = np.array(zs)[::step]
    ef_s = ef[::step, ::step, ::step]
    XG, YG, ZG = np.meshgrid(xs_s, ys_s, zs_s, indexing='ij')
    return {
        'x': XG.ravel().round(2).tolist(),
        'y': YG.ravel().round(2).tolist(),
        'z': ZG.ravel().round(2).tolist(),
        'u': ef_s[..., 0].ravel().round(4).tolist(),
        'v': ef_s[..., 1].ravel().round(4).tolist(),
        'w': ef_s[..., 2].ravel().round(4).tolist(),
    }


def _build_vol3d(step_snap, gt_data, grid):
    """Return {side: {corr: {learned, gt}, efield: {learned, gt} or None}}."""
    out = {}
    for side in ('east', 'west'):
        lrn     = step_snap['learned'].get(side, {})
        gt_side = (gt_data or {}).get(side)
        if not lrn:
            out[side] = None
            continue
        xs, ys, zs = _axes(grid[side])
        side_data = {}

        if 'corrections_cm' in lrn:
            co_l = np.array(lrn['corrections_cm'])
            co_g = np.array(gt_side['corrections_cm']) if gt_side else None
            side_data['corr'] = {
                'learned': _corrections_3d_data(co_l, xs, ys, zs),
                'gt':      _corrections_3d_data(co_g, xs, ys, zs) if co_g is not None else None,
            }

        if 'efield_Vcm' in lrn:
            ef_l = np.array(lrn['efield_Vcm'])
            ef_g = np.array(gt_side['efield_Vcm']) \
                   if gt_side and 'efield_Vcm' in gt_side else None
            side_data['efield'] = {
                'learned': _efield_3d_data(ef_l, xs, ys, zs),
                'gt':      _efield_3d_data(ef_g, xs, ys, zs) if ef_g is not None else None,
            }

        out[side] = side_data if side_data else None
    return out


def _curl(efield, spacing_cm):
    """Numerical curl ∇×E from (Nx,Ny,Nz,3) grid. Returns (Nx,Ny,Nz,3) in V/cm²."""
    dx, dy, dz = float(spacing_cm[0]), float(spacing_cm[1]), float(spacing_cm[2])
    Ex, Ey, Ez = efield[..., 0], efield[..., 1], efield[..., 2]
    curl_x = np.gradient(Ez, dy, axis=1) - np.gradient(Ey, dz, axis=2)
    curl_y = np.gradient(Ex, dz, axis=2) - np.gradient(Ez, dx, axis=0)
    curl_z = np.gradient(Ey, dx, axis=0) - np.gradient(Ex, dy, axis=1)
    return np.stack([curl_x, curl_y, curl_z], axis=-1)


def _build_quantities(step_snap, gt_data, grid):
    """Return {side: {type: {component: qty_entry}}}.

    Types: 'E-field', 'Corrections', 'Potential', 'Curl |∇×E|'.
    Each component entry has: has_gt, learned, gt, diff, reldiff, vmin, vmax.
    """
    sides = {}
    for side in ('east', 'west'):
        lrn = step_snap['learned'].get(side, {})
        gt_side = (gt_data or {}).get(side)
        xs, ys, zs = _axes(grid[side])
        groups = {}

        # ── E-field ──────────────────────────────────────────────────────────
        if 'efield_Vcm' in lrn:
            ef_l = np.array(lrn['efield_Vcm'])
            ef_g = np.array(gt_side['efield_Vcm']) if gt_side else None
            grp = {}
            for ci, name in enumerate(['Ex', 'Ey', 'Ez']):
                cl = ef_l[..., ci]
                cg = ef_g[..., ci] if ef_g is not None else None
                grp[name] = _qty(cl, cg, xs, ys, zs)
            mag_l = np.linalg.norm(ef_l, axis=-1)
            mag_g = np.linalg.norm(ef_g, axis=-1) if ef_g is not None else None
            grp['|E|'] = _qty(mag_l, mag_g, xs, ys, zs)
            groups['E-field'] = grp

        # ── Drift corrections ─────────────────────────────────────────────────
        if 'corrections_cm' in lrn:
            co_l = np.array(lrn['corrections_cm'])
            co_g = np.array(gt_side['corrections_cm']) if gt_side else None
            grp = {}
            for ci, name in enumerate(['Δx', 'Δy', 'Δz']):
                cl = co_l[..., ci]
                cg = co_g[..., ci] if co_g is not None else None
                grp[name] = _qty(cl, cg, xs, ys, zs)
            mag_l = np.linalg.norm(co_l, axis=-1)
            mag_g = np.linalg.norm(co_g, axis=-1) if co_g is not None else None
            grp['|Δ|'] = _qty(mag_l, mag_g, xs, ys, zs)
            groups['Corrections'] = grp

        # ── Distortion potential (learned-only) ───────────────────────────────
        if 'potential_Vcm_cm' in lrn:
            pot = np.array(lrn['potential_Vcm_cm'])
            vmin_l, vmax_l = _vminmax(None, pot)
            groups['Potential'] = {
                'δφ': {
                    'has_gt': False,
                    'learned': _slices(pot, xs, ys, zs),
                    'gt': None, 'diff': None, 'reldiff': None,
                    'vmin_l': vmin_l, 'vmax_l': vmax_l,
                    'vmin_g': None,   'vmax_g': None,
                }
            }

        # ── Curl |∇×E| — zero for a physical (conservative) field ────────────
        if 'efield_Vcm' in lrn:
            sp = np.array(grid[side]['spacing_cm'], dtype=np.float64)
            curl_l = _curl(np.array(lrn['efield_Vcm'], dtype=np.float64), sp).astype(np.float32)
            curl_g = _curl(np.array(gt_side['efield_Vcm'], dtype=np.float64), sp).astype(np.float32) \
                     if gt_side and 'efield_Vcm' in gt_side else None
            grp = {}
            for ci, name in enumerate(['(∇×E)_x', '(∇×E)_y', '(∇×E)_z']):
                cl = curl_l[..., ci]
                cg = curl_g[..., ci] if curl_g is not None else None
                grp[name] = _qty(cl, cg, xs, ys, zs)
            mag_l = np.linalg.norm(curl_l, axis=-1)
            mag_g = np.linalg.norm(curl_g, axis=-1) if curl_g is not None else None
            grp['|∇×E|'] = _qty(mag_l, mag_g, xs, ys, zs)
            groups['Curl'] = grp

        sides[side] = groups
    return sides


def _run_label(source_pkl, meta):
    mode = meta.get('mode', '?')
    stem = Path(source_pkl).stem
    m = re.search(r'result_(\w+)$', stem)
    seed = m.group(1) if m else stem
    parent = Path(source_pkl).parent.name
    rotor = meta.get('penalize_rotor', 0.0) or 0.0
    rotor_str = f' | rotor={rotor:g}' if rotor else ' | rotor=off'
    return f"{mode} | seed {seed} | {parent[:40]}{rotor_str}"


def load_eval_pkl(path):
    with open(path, 'rb') as f:
        ev = pickle.load(f)
    meta    = ev['efield_meta']
    gt_data = ev.get('gt')
    grid    = ev['grid']
    steps_out = []
    for snap in ev['steps']:
        sides  = _build_quantities(snap, gt_data, grid)
        vol3d  = _build_vol3d(snap, gt_data, grid)
        if not any(sides.values()):
            continue
        steps_out.append({
            'label': f"step {snap['step']}",
            'step':  snap['step'],
            'sides': sides,
            'vol3d': vol3d,
        })
    if not steps_out:
        return None
    return {
        'label':      _run_label(ev['source_pkl'], meta),
        'mode':       meta.get('mode', '?'),
        'source_pkl': ev['source_pkl'],
        'steps':      steps_out,
    }


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>E-field MLP evaluation</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; padding: 12px;
       background: #f0f0f0; color: #222; }
h2  { margin: 0 0 10px; font-size: 16px; }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-end;
            background: #fff; padding: 10px 14px; border-radius: 8px;
            margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.cg { display: flex; flex-direction: column; gap: 3px; }
.cg label { font-size: 11px; color: #666; font-weight: 600; }
.cg select { padding: 5px 8px; border-radius: 5px; border: 1px solid #ccc;
             background: #fafafa; font-size: 13px; }
.plots    { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; }
.plots-3d { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 10px; }
.section-title { font-weight: 700; font-size: 13px; color: #444; margin: 12px 0 4px; }
.panel { background: #fff; border-radius: 8px; padding: 8px 10px;
         box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.panel-title { text-align: center; font-weight: 700; font-size: 13px;
               color: #444; margin-bottom: 2px; }
.panel-info  { text-align: center; font-size: 10px; color: #999;
               margin-bottom: 4px; min-height: 14px; }
.sub-controls { display: flex; gap: 10px; align-items: flex-end; margin: 6px 0 6px; }
</style>
</head>
<body>
<h2>E-field MLP evaluation</h2>
<div class="controls">
  <div class="cg">
    <label>Run</label>
    <select id="run-sel" style="max-width:380px" onchange="onRunChange()"></select>
  </div>
  <div class="cg">
    <label>Step</label>
    <select id="step-sel" onchange="render()"></select>
  </div>
  <div class="cg">
    <label>Side</label>
    <select id="side-sel" onchange="render()">
      <option value="east">East</option>
      <option value="west">West</option>
    </select>
  </div>
  <div class="cg">
    <label>Type</label>
    <select id="type-sel" onchange="onTypeChange()"></select>
  </div>
  <div class="cg">
    <label>Component</label>
    <select id="comp-sel" onchange="render()"></select>
  </div>
  <div class="cg">
    <label>Slice</label>
    <select id="slice-sel" onchange="render()">
      <option value="xy">XY (central z)</option>
      <option value="xz">XZ (central y)</option>
      <option value="yz">YZ (central x)</option>
    </select>
  </div>
  <div class="cg">
    <label>3D corr source</label>
    <select id="src3d-sel" onchange="renderCorr3d()">
      <option value="learned">Learned</option>
      <option value="gt">GT</option>
      <option value="diff">GT − Learned</option>
    </select>
  </div>
  <div class="cg">
    <label>E-field 3D source</label>
    <select id="src-ef3d-sel" onchange="renderEfield3d()">
      <option value="learned">Learned</option>
      <option value="gt">GT</option>
      <option value="diff">GT − Learned</option>
    </select>
  </div>
</div>
<div class="plots">
  <div class="panel"><div class="panel-title">Ground truth</div><div class="panel-info" id="gt-info"></div><div id="gt-div"></div></div>
  <div class="panel"><div class="panel-title">Learned (MLP)</div><div class="panel-info" id="lrn-info"></div><div id="lrn-div"></div></div>
  <div class="panel"><div class="panel-title">Difference (GT − learned)</div><div class="panel-info" id="diff-info"></div><div id="diff-div"></div></div>
  <div class="panel"><div class="panel-title">Relative diff ((GT − learned) / |GT|)</div><div class="panel-info" id="reldiff-info"></div><div id="reldiff-div"></div></div>
</div>
<div class="section-title">Relative diff distribution (full volume)</div>
<div class="panel"><div id="hist-div"></div></div>
<div class="section-title">3D drift corrections [cm]</div>
<div class="plots-3d">
  <div class="panel"><div class="panel-title">Δx</div><div id="3d-dx"></div></div>
  <div class="panel"><div class="panel-title">Δy</div><div id="3d-dy"></div></div>
  <div class="panel"><div class="panel-title">Δz</div><div id="3d-dz"></div></div>
</div>
<div class="section-title">3D E-field vectors [V/cm]</div>
<div class="panel"><div id="3d-ef"></div></div>
<script>
const DATA = /*DATA_PLACEHOLDER*/null/*END*/;

const LAYOUT_BASE = {
  margin: {l:60, r:20, t:10, b:55},
  height: 330,
  paper_bgcolor: '#fff',
  plot_bgcolor:  '#fff',
};
const PLOTLY_CFG = {displayModeBar: false, responsive: false};
const CFG3D = {displayModeBar: true, responsive: false,
               modeBarButtonsToRemove: ['toImage','sendDataToCloud']};

function sliceZRange(z2d) {
  const flat = z2d.flat();
  let lo = Infinity, hi = -Infinity;
  for (const v of flat) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!isFinite(lo)) return [null, null];
  if (lo === hi) { lo -= Math.abs(lo) * 0.01 + 1e-9; hi += Math.abs(hi) * 0.01 + 1e-9; }
  return [lo, hi];
}

function heatTrace(sl, zmin, zmax, cs) {
  if (zmin === null || zmax === null) [zmin, zmax] = sliceZRange(sl.z);
  return {
    type: 'heatmap', z: sl.z, x: sl.x, y: sl.y,
    colorscale: cs, zmin: zmin, zmax: zmax,
    showscale: true,
    colorbar: {thickness: 14, len: 0.85, tickfont: {size: 10}},
  };
}
function axLayout(sl) {
  return {
    ...LAYOUT_BASE,
    xaxis: {title: {text: sl.xlabel, font:{size:11}}, tickfont:{size:10}},
    yaxis: {title: {text: sl.ylabel, font:{size:11}}, tickfont:{size:10},
            scaleanchor: 'x', scaleratio: 1},
  };
}
function noData(id) {
  Plotly.newPlot(id, [], {
    ...LAYOUT_BASE,
    annotations: [{text: 'not available', xref:'paper', yref:'paper',
                   x:0.5, y:0.5, showarrow:false, font:{size:13, color:'#bbb'}}]
  }, PLOTLY_CFG);
}
function safeRange(vmin, vmax) {
  if (vmin === null || vmax === null || vmin === vmax) return [null, null];
  return [vmin, vmax];
}
function sliceInfo(infoId, sl) {
  const el = document.getElementById(infoId);
  if (!sl || !sl.z || sl.z.length === 0) { el.textContent = ''; return; }
  const h = sl.z.length, w = sl.z[0].length;
  const nz = sl.z.flat().filter(v => v !== 0 && v !== null).length;
  el.textContent = `${w}×${h} | ${nz.toLocaleString()} nonzero`;
}
function plotHeat(id, sl, zmin, zmax, cs, lyt) {
  Plotly.newPlot(id, [heatTrace(sl, zmin, zmax, cs)], lyt, PLOTLY_CFG);
}
function noData3d(id, h=420) {
  Plotly.newPlot(id, [], {
    ...LAYOUT_BASE, height: h,
    annotations: [{text: 'not available', xref:'paper', yref:'paper',
                   x:0.5, y:0.5, showarrow:false, font:{size:13, color:'#bbb'}}]
  }, PLOTLY_CFG);
}

const SCENE_AXES = {
  xaxis: {title: {text:'x [cm]', font:{size:10}}, tickfont:{size:9}},
  yaxis: {title: {text:'y [cm]', font:{size:10}}, tickfont:{size:9}},
  zaxis: {title: {text:'z [cm]', font:{size:10}}, tickfont:{size:9}},
  aspectmode: 'data',
};

// ── Corrections 3D ────────────────────────────────────────────────────────────
function renderCorr3d() {
  const ri   = +document.getElementById('run-sel').value;
  const si   = +document.getElementById('step-sel').value;
  const side = document.getElementById('side-sel').value;
  const src  = document.getElementById('src3d-sel').value;

  const step = DATA[ri].steps[si];
  const vd   = step.vol3d && step.vol3d[side];
  const cd   = vd && vd.corr;
  if (!cd || !cd.learned) { ['3d-dx','3d-dy','3d-dz'].forEach(id => noData3d(id)); return; }

  const lrn = cd.learned, gt = cd.gt;
  const COMPS = [['dx','Δx [cm]','3d-dx'], ['dy','Δy [cm]','3d-dy'], ['dz','Δz [cm]','3d-dz']];
  for (const [key, title, pid] of COMPS) {
    let vals;
    if (src === 'gt') {
      if (!gt) { noData3d(pid); continue; }
      vals = gt[key];
    } else if (src === 'diff') {
      if (!gt) { noData3d(pid); continue; }
      vals = lrn[key].map((v, i) => gt[key][i] - v);
    } else {
      vals = lrn[key];
    }
    const absMax = Math.max(...vals.map(Math.abs));
    const cRange = absMax > 1e-6 ? {cmin:-absMax, cmax:absMax} : {cmin:-1e-6, cmax:1e-6};
    Plotly.newPlot(pid, [{
      type: 'scatter3d', mode: 'markers',
      x: lrn.x, y: lrn.y, z: lrn.z,
      marker: {size:2.5, opacity:0.85, color:vals, colorscale:'RdBu', reversescale:true,
               ...cRange,
               colorbar:{thickness:12, len:0.7, tickfont:{size:9},
                         title:{text:title, font:{size:10}}}},
      hovertemplate:`x:%{x:.1f} y:%{y:.1f} z:%{z:.1f}<br>${title}:%{marker.color:.3f}<extra></extra>`,
    }], {margin:{l:0,r:0,t:4,b:0}, height:420, paper_bgcolor:'#fff',
         scene:SCENE_AXES}, CFG3D);
  }
}

// ── E-field 3D cones ─────────────────────────────────────────────────────────
function renderEfield3d() {
  const ri   = +document.getElementById('run-sel').value;
  const si   = +document.getElementById('step-sel').value;
  const side = document.getElementById('side-sel').value;
  const src  = document.getElementById('src-ef3d-sel').value;

  const step = DATA[ri].steps[si];
  const vd   = step.vol3d && step.vol3d[side];
  const efd  = vd && vd.efield;
  if (!efd || !efd.learned) { noData3d('3d-ef', 500); return; }

  const lrn = efd.learned, gt = efd.gt;
  let u, v, w;
  if (src === 'gt') {
    if (!gt) { noData3d('3d-ef', 500); return; }
    u = gt.u; v = gt.v; w = gt.w;
  } else if (src === 'diff') {
    if (!gt) { noData3d('3d-ef', 500); return; }
    u = lrn.u.map((val, i) => gt.u[i] - val);
    v = lrn.v.map((val, i) => gt.v[i] - val);
    w = lrn.w.map((val, i) => gt.w[i] - val);
  } else {
    u = lrn.u; v = lrn.v; w = lrn.w;
  }

  const maxMag = Math.max(...u.map((_, i) => Math.sqrt(u[i]**2 + v[i]**2 + w[i]**2))) || 1;
  Plotly.newPlot('3d-ef', [{
    type: 'cone',
    x: lrn.x, y: lrn.y, z: lrn.z,
    u, v, w,
    colorscale: 'Viridis',
    sizemode: 'scaled',
    sizeref: maxMag,
    showscale: true,
    colorbar: {thickness:12, len:0.7, tickfont:{size:9},
               title:{text:'|E| [V/cm]', font:{size:10}}},
    hovertemplate: 'x:%{x:.1f} y:%{y:.1f} z:%{z:.1f}<br>' +
                   'E:(%{u:.3f},%{v:.3f},%{w:.3f})<extra></extra>',
  }], {margin:{l:0,r:0,t:4,b:0}, height:520, paper_bgcolor:'#fff',
       scene:SCENE_AXES}, CFG3D);
}

// ── Reldiff histogram ─────────────────────────────────────────────────────────
function renderHist() {
  const ri   = +document.getElementById('run-sel').value;
  const si   = +document.getElementById('step-sel').value;
  const side = document.getElementById('side-sel').value;
  const type = document.getElementById('type-sel').value;
  const comp = document.getElementById('comp-sel').value;

  const step = DATA[ri].steps[si];
  const grp  = (step.sides[side] || {})[type];
  const qd   = grp ? grp[comp] : null;
  const h    = qd && qd.hist_reldiff;
  if (!h) { noData('hist-div'); return; }

  Plotly.newPlot('hist-div', [{
    type: 'bar',
    x: h.edges,
    y: h.counts,
    width: Array(h.counts.length).fill(h.width * 0.92),
    marker: {color: '#4477bb', line: {width: 0}},
  }], {
    margin: {l:60, r:20, t:6, b:50},
    height: 210,
    paper_bgcolor: '#fff',
    plot_bgcolor: '#fff',
    bargap: 0,
    xaxis: {title: {text:'(GT − learned) / (|GT| + ε)', font:{size:11}}, tickfont:{size:10},
            zeroline:true, zerolinecolor:'#e44', zerolinewidth:2},
    yaxis: {title: {text:'count', font:{size:11}}, tickfont:{size:10}},
  }, PLOTLY_CFG);
}

// ── URL query-string state ────────────────────────────────────────────────────
function pushState() {
  const g = id => document.getElementById(id).value;
  const p = new URLSearchParams({
    run:      g('run-sel'),
    step:     g('step-sel'),
    side:     g('side-sel'),
    type:     g('type-sel'),
    comp:     g('comp-sel'),
    slice:    g('slice-sel'),
    src3d:    g('src3d-sel'),
    src_ef3d: g('src-ef3d-sel'),
  });
  history.replaceState(null, '', '?' + p.toString());
}

function restoreState() {
  const p   = new URLSearchParams(location.search);
  const set = (id, val) => {
    if (!val) return;
    const el = document.getElementById(id);
    if ([...el.options].some(o => o.value === val)) el.value = val;
  };
  set('run-sel',  p.get('run'));
  onRunChange(true);           // populate step + type selects without rendering
  set('step-sel', p.get('step'));
  set('side-sel', p.get('side'));
  set('type-sel', p.get('type'));
  onTypeChange(true);          // populate comp select without rendering
  set('comp-sel',      p.get('comp'));
  set('slice-sel',     p.get('slice'));
  set('src3d-sel',     p.get('src3d'));
  set('src-ef3d-sel',  p.get('src_ef3d'));
  render();
}

// ── Main render ───────────────────────────────────────────────────────────────
function render() {
  const ri   = +document.getElementById('run-sel').value;
  const si   = +document.getElementById('step-sel').value;
  const side = document.getElementById('side-sel').value;
  const type = document.getElementById('type-sel').value;
  const comp = document.getElementById('comp-sel').value;
  const slk  = document.getElementById('slice-sel').value;

  const step = DATA[ri].steps[si];
  const grp  = (step.sides[side] || {})[type];
  const qd   = grp ? grp[comp] : null;
  if (!qd) {
    ['gt-div','lrn-div','diff-div','reldiff-div'].forEach(noData);
    noData('hist-div');
  } else {
    const {vmin_l, vmax_l, vmin_g, vmax_g} = qd;
    const csFor = (lo, hi) => (lo !== null && lo < 0 && hi > 0) ? 'RdBu' : 'Viridis';
    const [zlmin, zlmax] = safeRange(vmin_l, vmax_l);
    const [zgmin, zgmax] = safeRange(vmin_g, vmax_g);
    const slL = qd.learned[slk];
    const lyt = axLayout(slL);

    plotHeat('lrn-div', slL, zlmin, zlmax, csFor(vmin_l, vmax_l), lyt);
    sliceInfo('lrn-info', slL);

    if (qd.has_gt && qd.gt) {
      const slG = qd.gt[slk];
      plotHeat('gt-div', slG, zgmin, zgmax, csFor(vmin_g, vmax_g), lyt);
      sliceInfo('gt-info', slG);
    } else { noData('gt-div'); document.getElementById('gt-info').textContent = ''; }

    if (qd.has_gt && qd.diff) {
      const slD = qd.diff[slk];
      const dmax = Math.max(...slD.z.flat().map(v => Math.abs(v)));
      const [dmin2, dmax2] = safeRange(-dmax, dmax);
      plotHeat('diff-div', slD, dmin2, dmax2, 'RdBu', lyt);
      sliceInfo('diff-info', slD);
    } else { noData('diff-div'); document.getElementById('diff-info').textContent = ''; }

    if (qd.has_gt && qd.reldiff) {
      const slR = qd.reldiff[slk];
      const rmax = Math.max(...slR.z.flat().map(v => Math.abs(v)));
      const [rmin2, rmax2] = safeRange(-rmax, rmax);
      plotHeat('reldiff-div', slR, rmin2, rmax2, 'RdBu', lyt);
      sliceInfo('reldiff-info', slR);
    } else { noData('reldiff-div'); document.getElementById('reldiff-info').textContent = ''; }

    renderHist();
  }

  renderCorr3d();
  renderEfield3d();
  pushState();
}

function onTypeChange(noRender = false) {
  const ri   = +document.getElementById('run-sel').value;
  const si   = +document.getElementById('step-sel').value;
  const side = document.getElementById('side-sel').value;
  const type = document.getElementById('type-sel').value;
  const grp  = (DATA[ri].steps[si].sides[side] || {})[type] || {};
  const compSel = document.getElementById('comp-sel');
  const prev = compSel.value;
  compSel.innerHTML = '';
  Object.keys(grp).forEach(c => {
    const o = document.createElement('option');
    o.value = c; o.text = c; compSel.appendChild(o);
  });
  if ([...compSel.options].some(o => o.value === prev)) compSel.value = prev;
  if (!noRender) render();
}

function onRunChange(noRender = false) {
  const ri  = +document.getElementById('run-sel').value;
  const run = DATA[ri];

  const stepSel = document.getElementById('step-sel');
  stepSel.innerHTML = '';
  run.steps.forEach((s, i) => {
    const o = document.createElement('option');
    o.value = i; o.text = s.label; stepSel.appendChild(o);
  });
  stepSel.value = run.steps.length - 1;

  const types = Object.keys(run.steps[0].sides.east || {});
  const typeSel = document.getElementById('type-sel');
  typeSel.innerHTML = '';
  types.forEach(t => {
    const o = document.createElement('option');
    o.value = t; o.text = t; typeSel.appendChild(o);
  });

  if (!noRender) onTypeChange();
}

// Init
const runSel = document.getElementById('run-sel');
DATA.forEach((run, i) => {
  const o = document.createElement('option');
  o.value = i; o.text = run.label; runSel.appendChild(o);
});
restoreState();
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
    parser.add_argument('eval_pkls', nargs='*',
                        help='*_efield_eval.pkl files to include')
    parser.add_argument('--results-dir', nargs='+', default=[], metavar='DIR',
                        help='Scan recursively for *_efield_eval.pkl (repeatable)')
    parser.add_argument('--output', default='efield_eval.html',
                        help='Output HTML path (default: efield_eval.html)')
    args = parser.parse_args()

    pkls = list(args.eval_pkls)
    for rd in args.results_dir:
        pkls += sorted(glob.glob(
            os.path.join(rd, '**', '*_efield_eval.pkl'),
            recursive=True))
    pkls = sorted(set(pkls))

    if not pkls:
        sys.exit('No *_efield_eval.pkl files found.')

    runs = []
    for p in pkls:
        print(f'Loading {p}')
        try:
            run = load_eval_pkl(p)
            if run is None:
                print(f'  [skip] no usable data')
            else:
                runs.append(run)
                print(f'  -> {run["label"]}  ({len(run["steps"])} step(s))')
        except Exception as exc:
            import traceback
            print(f'  ERROR: {exc}')
            traceback.print_exc()

    if not runs:
        sys.exit('Nothing to plot.')

    json_data = json.dumps(runs, separators=(',', ':'))
    html = _HTML.replace('/*DATA_PLACEHOLDER*/null/*END*/', json_data)

    out = Path(args.output)
    out.write_text(html, encoding='utf-8')
    print(f'\nWrote {out}  ({out.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
