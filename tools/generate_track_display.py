#!/usr/bin/env python3
"""Generate a self-contained 3-D event-display for 20 randomly generated boundary tracks.

Outputs:
  plots/20260609_Track_Generator/
      index.html                     — tabbed viewer: 3D View + Histograms
      edep_3d_Track{i}_*.html        — one Plotly 3-D HTML per track
      wireplanes_20x6_gt_signals.pdf — wire-plane × time images (N rows × 6 columns)

Usage:
  python tools/generate_track_display.py
  python tools/generate_track_display.py --n 20 --seed 42 --output-dir plots/20260609_Track_Generator
  python tools/generate_track_display.py --min-x-mm 1500
"""
import argparse
import html as _html
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from tools.random_boundary_tracks import (
    generate_random_boundary_track,
    filter_track_inside_volumes,
)
from src.plots.plot_mixed_tracks_edep_wireplanes import (
    build_simulator,
    track_and_forward,
    write_edep_3d_html,
    _safe_stem,
)

_PLOTS_DIR = os.environ.get('PLOTS_DIR', 'plots')

_FACE_LABELS = ['x−', 'x+', 'y−', 'y+', 'z−', 'z+']
_SIGNAL_PERCENTILE = 99.0


def _face_label(start_mm, min_x_mm, y0, y1, z0, z1):
    """Infer which face the track started on from its start position."""
    sx, sy, sz = start_mm
    tol = 1e-3
    if abs(sx - (-min_x_mm)) < tol:
        return 'x−'
    if abs(sx - min_x_mm) < tol:
        return 'x+'
    if abs(sy - y0) < tol:
        return 'y−'
    if abs(sy - y1) < tol:
        return 'y+'
    if abs(sz - z0) < tol:
        return 'z−'
    return 'z+'


def _compute_edep_coord_hists(tracks_raw, n_bins=60):
    """Pre-bin x/y/z deposit coordinates for the histogram tab (summed across selected tracks)."""
    vol_half = 2160.0
    edges = np.linspace(-vol_half, vol_half, n_bins + 1).tolist()
    per_track = []
    for track in tracks_raw:
        pos = np.asarray(track['position'])
        if len(pos) == 0:
            per_track.append({'hx': [0]*n_bins, 'hy': [0]*n_bins, 'hz': [0]*n_bins})
            continue
        hx, _ = np.histogram(pos[:, 0], bins=edges)
        hy, _ = np.histogram(pos[:, 1], bins=edges)
        hz, _ = np.histogram(pos[:, 2], bins=edges)
        per_track.append({'hx': hx.tolist(), 'hy': hy.tolist(), 'hz': hz.tolist()})
    return {'edges': edges, 'tracks': per_track}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=20, help='Number of tracks')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--min-x-mm', type=float, default=1000.0,
                        help='Inner-box half-width in x (mm)')
    parser.add_argument('--output-dir', default=os.path.join(_PLOTS_DIR, '20260609_Track_Generator'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Building simulator (with wire response)...')
    simulator = build_simulator(include_wire_response=True)
    cfg = simulator.config
    volumes = cfg.volumes
    n_p = volumes[0].n_planes
    time_step_us = float(cfg.time_step_us)

    col_labels = []
    for v in range(cfg.n_volumes):
        for p in range(n_p):
            col_labels.append(f'{cfg.plane_names[v][p]}{v + 1}')

    print('Warming up JIT...')
    t0 = time.time()
    simulator.warm_up()
    print(f'  Done ({time.time() - t0:.1f} s)')

    east = volumes[0]
    y0 = east.ranges_cm[1][0] * 10.0
    y1 = east.ranges_cm[1][1] * 10.0
    z0 = east.ranges_cm[2][0] * 10.0
    z1 = east.ranges_cm[2][1] * 10.0

    rng_energy = np.random.default_rng(args.seed)

    print(f'\nSimulating {args.n} tracks ...')
    specs = []
    tracks_raw = []
    all_planes = []
    track_stats = []
    meta = []

    for i in range(args.n):
        direction, start_mm = generate_random_boundary_track(
            volumes, seed=args.seed + i, min_x_mm=args.min_x_mm
        )
        energy_mev = float(rng_energy.uniform(100.0, 1000.0))
        name = f'Track{i + 1}_{int(round(energy_mev))}MeV'

        spec = dict(
            name=name,
            direction=direction,
            momentum_mev=energy_mev,
            start_position_mm=start_mm,
        )
        specs.append(spec)

        t0 = time.time()
        track, planes = track_and_forward(simulator, spec, start_mm)
        tracks_raw.append(track)
        all_planes.append(planes)

        de = np.asarray(track['de'])
        n_dep = int(len(de))
        mean_de = float(de.mean()) if n_dep > 0 else 0.0
        total_de = float(de.sum()) if n_dep > 0 else 0.0
        track_stats.append(dict(n_deposits=n_dep, mean_de=mean_de, total_de=total_de))

        dx, dy, dz = direction
        theta_deg = float(math.degrees(math.acos(max(-1.0, min(1.0, dx)))))
        alpha_deg = float(math.degrees(math.atan2(dz, dy))) % 360.0
        face = _face_label(start_mm, args.min_x_mm, y0, y1, z0, z1)
        path_len = float(np.asarray(track['dx']).sum()) if n_dep > 0 else 0.0

        meta.append(dict(
            name=name,
            E=energy_mev,
            theta=theta_deg,
            alpha=alpha_deg,
            start_x=float(start_mm[0]),
            start_y=float(start_mm[1]),
            start_z=float(start_mm[2]),
            face=face,
            n_deposits=n_dep,
            path_len_mm=path_len,
        ))
        print(f'  [{i + 1:2d}/{args.n}] {name}: N_dep={n_dep:,}  face={face}  ({time.time()-t0:.1f}s)')

    # Shared dE colour scale
    all_de = np.concatenate([np.asarray(t['de']) for t in tracks_raw if len(t['de']) > 0])
    if len(all_de) > 0:
        de_min, de_max = float(all_de.min()), float(all_de.max())
        de_range = max(de_max - de_min, 1e-9)
    else:
        de_min, de_max, de_range = 0.0, 1.0, 1.0

    # Write per-track 3D HTML files
    html_files = []
    for spec, track, stats in zip(specs, tracks_raw, track_stats):
        stem = _safe_stem(spec['name'])
        fname = f'edep_3d_{stem}.html'
        write_edep_3d_html(track, spec, os.path.join(args.output_dir, fname),
                           de_min, de_max, de_range, volumes, stats=stats)
        html_files.append(fname)

    # Wire-plane PDF
    pdf_name = f'wireplanes_{args.n}x6_gt_signals.pdf'
    pdf_path = os.path.join(args.output_dir, pdf_name)
    _write_wireplanes_pdf(
        specs, tracks_raw, all_planes, col_labels,
        time_step_us, de_min, de_max, pdf_path,
    )
    print(f'Saved {pdf_path}')

    # Edep coordinate histograms for the Histograms tab
    edep_hist = _compute_edep_coord_hists(tracks_raw)

    # Build index.html
    _write_index_html(args.output_dir, specs, html_files, track_stats, meta,
                      args.min_x_mm, pdf_name, edep_hist)
    print(f'\nWrote {os.path.join(args.output_dir, "index.html")}')


def _write_wireplanes_pdf(specs, tracks_raw, all_planes, col_labels,
                          time_step_us, de_min, de_max, pdf_path):
    n_row = len(specs)
    abs_stack = np.concatenate(
        [np.abs(p).ravel() for planes in all_planes for p in planes if p.size > 0])
    vmax_sig = float(np.nanpercentile(abs_stack, _SIGNAL_PERCENTILE)) if abs_stack.size > 0 else 1.0
    vmax_sig = max(vmax_sig, 1e-9)
    norm_2d = mcolors.Normalize(vmin=-vmax_sig, vmax=vmax_sig)

    fig_h = max(8.0, 3.0 * n_row)
    fig, axes = plt.subplots(n_row, 6, figsize=(18, fig_h), constrained_layout=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(
        f'GT wire-plane signals (shared scale ±{vmax_sig:.3g} e⁻, p{_SIGNAL_PERCENTILE:g} |all|)\n'
        f'dE 3-D HTML colour range [{de_min:.5g}, {de_max:.5g}] MeV',
        fontsize=11,
    )
    im = None
    for i, spec in enumerate(specs):
        planes = all_planes[i]
        for j in range(6):
            ax = axes[i, j]
            data = planes[j]
            n_wires, n_time = data.shape
            t_max_us = n_time * time_step_us
            im = ax.imshow(
                data, aspect='auto', origin='lower',
                norm=norm_2d, cmap='RdBu_r',
                extent=[0, t_max_us, 0, n_wires],
            )
            if i == 0:
                ax.set_title(col_labels[j], fontsize=9)
            ax.set_ylabel(
                f"{spec['name']}\nwire" if j == 0 else 'wire',
                fontsize=7, rotation=90, va='center',
            )
            ax.set_xlabel('t (μs)', fontsize=7)
            ax.tick_params(axis='both', labelsize=6)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.35, aspect=60,
                     pad=0.02, label='signal (e⁻)')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)


def _write_index_html(output_dir, specs, html_files, track_stats, meta, min_x_mm,
                      pdf_name=None, edep_hist=None):
    options_html = '\n'.join(
        f'        <option value="{_html.escape(f)}"'
        f'{"" if i else " selected"}>'
        f'{_html.escape(s["name"])}  |  T={s["momentum_mev"]:.0f} MeV  '
        f'dir=({s["direction"][0]:.3f},{s["direction"][1]:.3f},{s["direction"][2]:.3f})'
        f'</option>'
        for i, (s, f) in enumerate(zip(specs, html_files))
    )

    stats_js = json.dumps([
        {'n': st['n_deposits'], 'mean': st['mean_de'], 'total': st['total_de']}
        for st in track_stats
    ])
    hist_tracks_js = json.dumps(meta)

    index_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Track Generator — 20260609</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:system-ui,sans-serif;background:#f0f2f5;color:#1a1a1a}}
    .page-hdr{{background:#1a1a2e;color:#eee;padding:12px 20px}}
    .page-hdr h1{{font-size:1.05rem;font-weight:600;margin-bottom:3px}}
    .page-hdr p{{font-size:0.78rem;color:#aac4ff;margin:0}}
    .tabs{{display:flex;background:#16213e;padding:0 20px;gap:0}}
    .tab-btn{{padding:9px 20px;border:none;cursor:pointer;background:transparent;
             color:#8899aa;font-size:0.85rem;font-weight:500;
             border-bottom:3px solid transparent;transition:color .15s}}
    .tab-btn.active{{color:#fff;border-bottom-color:#e94560}}
    .tab-btn:hover:not(.active){{color:#ccd}}
    .tab-pane{{display:none;padding:14px 20px 20px}}
    .tab-pane.active{{display:block}}
    .toolbar{{display:flex;flex-wrap:wrap;align-items:center;gap:.8rem;margin-bottom:.4rem}}
    select{{padding:6px 10px;border:1px solid #d0d5dd;border-radius:5px;
           background:#fff;font-size:.87rem;min-width:140px}}
    .stats-bar{{font-size:.82rem;color:#444;font-family:monospace;
               background:#f5f5f5;padding:.25rem .5rem;border-radius:4px;display:inline-block}}
    iframe{{width:100%;height:calc(100vh - 13.5rem);border:1px solid #ccc;border-radius:4px;margin-top:.4rem}}
    .hist-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:.75rem}}
    .hist-card{{background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:.5rem .75rem}}
    .hist-card h3{{font-size:.82rem;margin-bottom:.25rem;color:#333}}
    svg{{display:block;overflow:visible}}
    .hbar{{fill:#4682B4;opacity:.72}}.hbar:hover{{opacity:1}}
    .ebar{{fill:#3a9e6a;opacity:.75}}.ebar:hover{{opacity:1}}
    .catbar{{fill:#e94560;opacity:.78}}.catbar:hover{{opacity:1}}
    .muon-item{{display:block;font-size:.76rem;padding:2px 0;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
  </style>
</head>
<body>
<div class="page-hdr">
  <h1>Track Generator — {len(specs)} random boundary muons  |  inner-box half-x = {min_x_mm:.0f} mm</h1>
  <p>Start positions sampled uniformly on the 6 faces of the inner box (area-weighted).
     Direction: θ (from x/drift axis) and α (azimuthal) drawn uniformly.
     Energy: uniform [100, 1000] MeV.</p>
</div>
<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('view3d',this)">3D View</button>
  <button class="tab-btn"        onclick="switchTab('hist',this)">Histograms</button>
</div>

<!-- ── Tab 1: 3D View ──────────────────────────────────────────────────────── -->
<div id="pane-view3d" class="tab-pane active">
  <div class="toolbar">
    <div>
      <label for="track-select"><strong>Track</strong></label>&nbsp;
      <select id="track-select" aria-label="Choose track">
{options_html}
      </select>
    </div>
    {f'<a href="{_html.escape(pdf_name, quote=True)}" target="_blank" style="font-size:.85rem;text-decoration:none">&#128196; Wire-plane signals PDF</a>' if pdf_name else ''}
  </div>
  <div class="stats-bar" id="stats-bar"></div>
  <iframe id="track-frame" src="{html_files[0]}" title="3D energy deposits"></iframe>
</div>

<!-- ── Tab 2: Histograms ────────────────────────────────────────────────────── -->
<div id="pane-hist" class="tab-pane">
  <div style="display:flex;gap:1rem;height:calc(100vh - 9rem)">
    <div style="width:210px;min-width:140px;padding-right:.75rem;border-right:1px solid #ddd;overflow-y:auto;flex-shrink:0">
      <div style="font-size:.8rem;font-weight:600;margin-bottom:.35rem">Tracks</div>
      <div class="btn-row" style="margin-bottom:.35rem">
        <button onclick="histPreset(true)" style="padding:3px 9px;border:1px solid #d0d5dd;background:#fff;font-size:.78rem;cursor:pointer;border-radius:4px">All</button>
        <button onclick="histPreset(false)" style="padding:3px 9px;border:1px solid #d0d5dd;background:#fff;font-size:.78rem;cursor:pointer;border-radius:4px">None</button>
      </div>
      <div id="hist-cbs"></div>
    </div>
    <div style="flex:1;overflow-y:auto">
      <h2 style="font-size:.88rem;font-weight:600;margin:0 0 .45rem;color:#444">Track parameters</h2>
      <div class="hist-grid" id="hist-grid-main"></div>
      <h2 style="font-size:.88rem;font-weight:600;margin:1rem 0 .45rem;color:#444">Start-position coordinates (mm)</h2>
      <div class="hist-grid" id="hist-grid-pos"></div>
      <h2 style="font-size:.88rem;font-weight:600;margin:1rem 0 .45rem;color:#444">Start face &amp; deposit statistics</h2>
      <div class="hist-grid" id="hist-grid-extra"></div>
      <h2 style="font-size:.88rem;font-weight:600;margin:1rem 0 .45rem;color:#444">Energy deposit coordinates (mm)</h2>
      <div class="hist-grid" id="hist-grid-edep"></div>
    </div>
  </div>
</div>

<script>
// ── 3D-view controls ─────────────────────────────────────────────────────────
const STATS = {stats_js};
const sel3d = document.getElementById('track-select');
const frame  = document.getElementById('track-frame');
function updateStats(idx){{
  var s=STATS[idx];
  document.getElementById('stats-bar').textContent=
    'N deposits: '+s.n.toLocaleString()+
    '   mean dE: '+s.mean.toPrecision(4)+' MeV'+
    '   total dE: '+s.total.toPrecision(4)+' MeV';
}}
updateStats(sel3d.selectedIndex);
sel3d.addEventListener('change',function(){{frame.src=sel3d.value;updateStats(sel3d.selectedIndex);}});

function switchTab(id, btn) {{
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('pane-' + id).classList.add('active');
  btn.classList.add('active');
  if (id === 'hist') {{ renderHist(); }}
}}

// ── Histograms ───────────────────────────────────────────────────────────────
const HIST_TRACKS = {hist_tracks_js};
const HIST_EDEP   = {json.dumps(edep_hist) if edep_hist else 'null'};
const FACE_ORDER  = ['x−','x+','y−','y+','z−','z+'];
const H_BINS=8, H_W=290, H_H=170, H_PL=36, H_PR=10, H_PT=8, H_PB=38;
const HSVGNS='http://www.w3.org/2000/svg';

(function(){{
  var cbDiv=document.getElementById('hist-cbs');
  HIST_TRACKS.forEach(function(tr,i){{
    var lbl=document.createElement('label'); lbl.className='muon-item';
    var cb=document.createElement('input'); cb.type='checkbox'; cb.className='hist-cb'; cb.value=i; cb.checked=true;
    cb.addEventListener('change',renderHist);
    lbl.appendChild(cb); lbl.appendChild(document.createTextNode(' '+tr.name));
    cbDiv.appendChild(lbl);
  }});
}})();

function histPreset(all){{
  document.querySelectorAll('.hist-cb').forEach(function(c){{c.checked=all;}});
  renderHist();
}}

function hse(tag,attrs){{
  var e=document.createElementNS(HSVGNS,tag);
  for(var k in attrs){{if(Object.prototype.hasOwnProperty.call(attrs,k))e.setAttribute(k,attrs[k]);}}
  return e;
}}
function hst(tag,attrs,text){{var e=hse(tag,attrs);e.textContent=String(text);return e;}}
function hFmt(v){{if(v===null||!isFinite(v))return'-';return Math.abs(v)>=100?v.toFixed(1):v.toFixed(2);}}

function hComputeHist(vals){{
  var finite=[];
  for(var i=0;i<vals.length;i++){{if(vals[i]!==null&&isFinite(vals[i]))finite.push(vals[i]);}}
  if(finite.length===0)return null;
  var lo=finite[0],hi=finite[0];
  for(var j=1;j<finite.length;j++){{if(finite[j]<lo)lo=finite[j];if(finite[j]>hi)hi=finite[j];}}
  if(lo===hi){{lo-=0.5;hi+=0.5;}}
  var w=(hi-lo)/H_BINS,counts=[],bnames=[];
  for(var bi=0;bi<H_BINS;bi++){{counts.push(0);bnames.push([]);}}
  for(var k=0;k<vals.length;k++){{
    if(vals[k]===null||!isFinite(vals[k]))continue;
    var b=Math.floor((vals[k]-lo)/w);if(b>=H_BINS)b=H_BINS-1;
    counts[b]++;bnames[b].push(k);
  }}
  return {{lo:lo,hi:hi,w:w,counts:counts,bnames:bnames}};
}}

function hBuildSVG(hist,selNames){{
  var pw=H_W-H_PL-H_PR,ph=H_H-H_PT-H_PB;
  var svg=hse('svg',{{width:H_W,height:H_H}});
  if(!hist){{svg.appendChild(hst('text',{{x:H_W/2,y:H_H/2,'text-anchor':'middle','font-size':'12',fill:'#999'}},'no data'));return svg;}}
  var maxCnt=1;
  for(var mi=0;mi<hist.counts.length;mi++){{if(hist.counts[mi]>maxCnt)maxCnt=hist.counts[mi];}}
  var bw=pw/H_BINS;
  for(var b=0;b<H_BINS;b++){{
    var x=H_PL+b*bw,cnt=hist.counts[b],bh=(cnt/maxCnt)*ph;
    if(cnt>0&&bh<1)bh=1;
    var y=H_PT+ph-bh;
    var r=hse('rect',{{'class':'hbar',x:(x+1).toFixed(1),y:y.toFixed(1),width:(bw-2).toFixed(1),height:bh.toFixed(1)}});
    if(hist.bnames[b].length>0){{
      var tip=document.createElementNS(HSVGNS,'title');
      var lines=[cnt+' track(s):'];
      for(var ti=0;ti<hist.bnames[b].length;ti++)lines.push(selNames[hist.bnames[b][ti]]);
      tip.textContent=lines.join('\\n');r.appendChild(tip);
    }}
    svg.appendChild(r);
    if(cnt>0)svg.appendChild(hst('text',{{x:(x+bw/2).toFixed(1),y:(y-2).toFixed(1),'text-anchor':'middle','font-size':'10',fill:'#333'}},cnt));
  }}
  svg.appendChild(hse('line',{{x1:H_PL,y1:H_PT+ph,x2:H_PL+pw,y2:H_PT+ph,stroke:'#888','stroke-width':'1'}}));
  for(var t=0;t<=4;t++){{
    var frac=t/4,xp=H_PL+frac*pw,xval=hist.lo+frac*(hist.hi-hist.lo);
    svg.appendChild(hse('line',{{x1:xp.toFixed(1),y1:(H_PT+ph).toFixed(1),x2:xp.toFixed(1),y2:(H_PT+ph+4).toFixed(1),stroke:'#888','stroke-width':'1'}}));
    svg.appendChild(hst('text',{{x:xp.toFixed(1),y:(H_PT+ph+14).toFixed(1),'text-anchor':'middle','font-size':'9',fill:'#555'}},hFmt(xval)));
  }}
  svg.appendChild(hse('line',{{x1:H_PL,y1:H_PT,x2:H_PL,y2:H_PT+ph,stroke:'#888','stroke-width':'1'}}));
  for(var t2=0;t2<=4;t2++){{
    var frac2=t2/4,yp=H_PT+ph-frac2*ph,ycnt=Math.round(frac2*maxCnt);
    svg.appendChild(hse('line',{{x1:(H_PL-3).toFixed(1),y1:yp.toFixed(1),x2:H_PL,y2:yp.toFixed(1),stroke:'#888','stroke-width':'1'}}));
    svg.appendChild(hst('text',{{x:(H_PL-5).toFixed(1),y:(yp+3).toFixed(1),'text-anchor':'end','font-size':'9',fill:'#555'}},ycnt));
  }}
  return svg;
}}

function hBuildCategoricalSVG(counts, labels){{
  var n=labels.length,pw=H_W-H_PL-H_PR,ph=H_H-H_PT-H_PB;
  var svg=hse('svg',{{width:H_W,height:H_H}});
  var total=0;for(var i=0;i<n;i++)total+=counts[i];
  if(total===0){{svg.appendChild(hst('text',{{x:H_W/2,y:H_H/2,'text-anchor':'middle','font-size':'12',fill:'#999'}},'no data'));return svg;}}
  var maxCnt=1;for(var mi=0;mi<n;mi++){{if(counts[mi]>maxCnt)maxCnt=counts[mi];}}
  var bw=pw/n;
  for(var b=0;b<n;b++){{
    var x=H_PL+b*bw,cnt=counts[b],bh=(cnt/maxCnt)*ph;
    if(cnt>0&&bh<1)bh=1;
    var y=H_PT+ph-bh;
    svg.appendChild(hse('rect',{{'class':'catbar',x:(x+1).toFixed(1),y:y.toFixed(1),width:(bw-2).toFixed(1),height:bh.toFixed(1)}}));
    if(cnt>0)svg.appendChild(hst('text',{{x:(x+bw/2).toFixed(1),y:(y-2).toFixed(1),'text-anchor':'middle','font-size':'10',fill:'#333'}},cnt));
    svg.appendChild(hst('text',{{x:(x+bw/2).toFixed(1),y:(H_PT+ph+14).toFixed(1),'text-anchor':'middle','font-size':'9',fill:'#555'}},labels[b]));
  }}
  svg.appendChild(hse('line',{{x1:H_PL,y1:H_PT+ph,x2:H_PL+pw,y2:H_PT+ph,stroke:'#888','stroke-width':'1'}}));
  svg.appendChild(hse('line',{{x1:H_PL,y1:H_PT,x2:H_PL,y2:H_PT+ph,stroke:'#888','stroke-width':'1'}}));
  for(var t2=0;t2<=4;t2++){{
    var frac2=t2/4,yp=H_PT+ph-frac2*ph,ycnt=Math.round(frac2*maxCnt);
    svg.appendChild(hse('line',{{x1:(H_PL-3).toFixed(1),y1:yp.toFixed(1),x2:H_PL,y2:yp.toFixed(1),stroke:'#888','stroke-width':'1'}}));
    svg.appendChild(hst('text',{{x:(H_PL-5).toFixed(1),y:(yp+3).toFixed(1),'text-anchor':'end','font-size':'9',fill:'#555'}},ycnt));
  }}
  return svg;
}}

function hBuildPreBinnedSVG(counts,edges){{
  var n=counts.length,pw=H_W-H_PL-H_PR,ph=H_H-H_PT-H_PB;
  var svg=hse('svg',{{width:H_W,height:H_H}});
  var total=0;for(var i=0;i<n;i++)total+=counts[i];
  if(total===0){{svg.appendChild(hst('text',{{x:H_W/2,y:H_H/2,'text-anchor':'middle','font-size':'12',fill:'#999'}},'no data'));return svg;}}
  var maxCnt=1;for(var mi=0;mi<n;mi++){{if(counts[mi]>maxCnt)maxCnt=counts[mi];}}
  var bw=pw/n;
  for(var b=0;b<n;b++){{
    var x=H_PL+b*bw,cnt=counts[b],bh=(cnt/maxCnt)*ph;
    if(cnt>0&&bh<1)bh=1;
    svg.appendChild(hse('rect',{{'class':'ebar',x:(x+0.5).toFixed(1),y:(H_PT+ph-bh).toFixed(1),width:Math.max(1,bw-1).toFixed(1),height:bh.toFixed(1)}}));
  }}
  svg.appendChild(hse('line',{{x1:H_PL,y1:H_PT+ph,x2:H_PL+pw,y2:H_PT+ph,stroke:'#888','stroke-width':'1'}}));
  var lo=edges[0],hi=edges[edges.length-1];
  for(var t=0;t<=4;t++){{
    var frac=t/4,xp=H_PL+frac*pw,xval=lo+frac*(hi-lo);
    svg.appendChild(hse('line',{{x1:xp.toFixed(1),y1:(H_PT+ph).toFixed(1),x2:xp.toFixed(1),y2:(H_PT+ph+4).toFixed(1),stroke:'#888','stroke-width':'1'}}));
    svg.appendChild(hst('text',{{x:xp.toFixed(1),y:(H_PT+ph+14).toFixed(1),'text-anchor':'middle','font-size':'9',fill:'#555'}},hFmt(xval)));
  }}
  svg.appendChild(hse('line',{{x1:H_PL,y1:H_PT,x2:H_PL,y2:H_PT+ph,stroke:'#888','stroke-width':'1'}}));
  for(var t2=0;t2<=4;t2++){{
    var frac2=t2/4,yp=H_PT+ph-frac2*ph,ycnt=Math.round(frac2*maxCnt);
    svg.appendChild(hse('line',{{x1:(H_PL-3).toFixed(1),y1:yp.toFixed(1),x2:H_PL,y2:yp.toFixed(1),stroke:'#888','stroke-width':'1'}}));
    svg.appendChild(hst('text',{{x:(H_PL-5).toFixed(1),y:(yp+3).toFixed(1),'text-anchor':'end','font-size':'9',fill:'#555'}},ycnt));
  }}
  return svg;
}}

function hMeanStd(vals){{
  var fin=[];
  for(var i=0;i<vals.length;i++){{if(vals[i]!==null&&isFinite(vals[i]))fin.push(vals[i]);}}
  if(fin.length===0)return null;
  var m=0; for(var i=0;i<fin.length;i++)m+=fin[i]; m/=fin.length;
  var v=0; for(var i=0;i<fin.length;i++)v+=(fin[i]-m)*(fin[i]-m); v/=fin.length;
  return {{mean:m,std:Math.sqrt(v)}};
}}
function hMeanStdBinned(counts,edges){{
  var n=counts.length,tot=0,ws=0,ws2=0;
  for(var b=0;b<n;b++){{var c=(edges[b]+edges[b+1])/2;tot+=counts[b];ws+=counts[b]*c;ws2+=counts[b]*c*c;}}
  if(tot===0)return null;
  var m=ws/tot; return {{mean:m,std:Math.sqrt(Math.max(0,ws2/tot-m*m))}};
}}
function hTitle(label,st){{
  if(!st)return label;
  return label+'  μ='+hFmt(st.mean)+'  σ='+hFmt(st.std);
}}

function renderHist(){{
  var cbs=document.querySelectorAll('.hist-cb:checked');
  var selTracks=[],selNames=[],selIdx=[];
  for(var ci=0;ci<cbs.length;ci++){{
    var idx=+cbs[ci].value;
    selIdx.push(idx);selTracks.push(HIST_TRACKS[idx]);selNames.push(HIST_TRACKS[idx].name);
  }}

  // Track parameter histograms
  var grid=document.getElementById('hist-grid-main');
  while(grid.firstChild)grid.removeChild(grid.firstChild);
  [
    {{key:'E',     label:'T (MeV)'}},
    {{key:'theta', label:'θ from x-axis (°)'}},
    {{key:'alpha', label:'α azimuthal (°)'}},
  ].forEach(function(q){{
    var vals=selTracks.map(function(t){{return t[q.key];}});
    var card=document.createElement('div');card.className='hist-card';
    var h3=document.createElement('h3');h3.textContent=hTitle(q.label,hMeanStd(vals));card.appendChild(h3);
    card.appendChild(hBuildSVG(hComputeHist(vals),selNames));
    grid.appendChild(card);
  }});

  // Start-position coordinate histograms
  var pg=document.getElementById('hist-grid-pos');
  while(pg.firstChild)pg.removeChild(pg.firstChild);
  [
    {{key:'start_x', label:'Start x (mm)'}},
    {{key:'start_y', label:'Start y (mm)'}},
    {{key:'start_z', label:'Start z (mm)'}},
  ].forEach(function(q){{
    var vals=selTracks.map(function(t){{return t[q.key];}});
    var card=document.createElement('div');card.className='hist-card';
    var h3=document.createElement('h3');h3.textContent=hTitle(q.label,hMeanStd(vals));card.appendChild(h3);
    card.appendChild(hBuildSVG(hComputeHist(vals),selNames));
    pg.appendChild(card);
  }});

  // Face distribution + deposit stats
  var eg=document.getElementById('hist-grid-extra');
  while(eg.firstChild)eg.removeChild(eg.firstChild);

  var faceCounts=FACE_ORDER.map(function(f){{
    return selTracks.reduce(function(s,t){{return s+(t.face===f?1:0);}},0);
  }});
  (function(){{
    var card=document.createElement('div');card.className='hist-card';
    var h3=document.createElement('h3');h3.textContent='Start face';card.appendChild(h3);
    card.appendChild(hBuildCategoricalSVG(faceCounts,FACE_ORDER));
    eg.appendChild(card);
  }})();

  [
    {{key:'n_deposits',  label:'N deposits'}},
    {{key:'path_len_mm', label:'Path length (mm)'}},
  ].forEach(function(q){{
    var vals=selTracks.map(function(t){{return t[q.key];}});
    var card=document.createElement('div');card.className='hist-card';
    var h3=document.createElement('h3');h3.textContent=hTitle(q.label,hMeanStd(vals));card.appendChild(h3);
    card.appendChild(hBuildSVG(hComputeHist(vals),selNames));
    eg.appendChild(card);
  }});

  // Edep coordinate histograms (pre-binned, summed over selected tracks)
  var dg=document.getElementById('hist-grid-edep');
  while(dg.firstChild)dg.removeChild(dg.firstChild);
  if(HIST_EDEP){{
    var n=HIST_EDEP.edges.length-1;
    var sx=[],sy=[],sz=[];
    for(var b=0;b<n;b++){{sx.push(0);sy.push(0);sz.push(0);}}
    selIdx.forEach(function(i){{
      var d=HIST_EDEP.tracks[i];if(!d)return;
      for(var b2=0;b2<n;b2++){{sx[b2]+=d.hx[b2];sy[b2]+=d.hy[b2];sz[b2]+=d.hz[b2];}}
    }});
    [{{label:'Edep x (mm)',counts:sx}},
     {{label:'Edep y (mm)',counts:sy}},
     {{label:'Edep z (mm)',counts:sz}}].forEach(function(ax){{
      var card=document.createElement('div');card.className='hist-card';
      var ms=hMeanStdBinned(ax.counts,HIST_EDEP.edges);
      var h3=document.createElement('h3');h3.textContent=hTitle(ax.label,ms);card.appendChild(h3);
      card.appendChild(hBuildPreBinnedSVG(ax.counts,HIST_EDEP.edges));
      dg.appendChild(card);
    }});
  }}
}}
</script>
</body>
</html>'''

    with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(index_html)


if __name__ == '__main__':
    main()
