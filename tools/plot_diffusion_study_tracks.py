#!/usr/bin/env python3
"""Generate 3-D energy-deposit HTML viewer + wire-plane PDF for the diffusion study tracks.

Simulates and visualises:
  • Muon4_100MeV at |start_x| = 0, 1000, 2000 mm (east volume, fixed direction)
  • 400 MeV muon at θ = 10°, 30°, 90° in XY plane, starting at (1900, 0, 0) mm

Outputs:
  $PLOTS_DIR/diffusion_study_tracks/
      index.html                    — tabbed page: 3D viewer + ADC retention plots
      edep_3d_<name>.html           — one Plotly 3-D HTML per track
      wireplanes_6x6_gt_signals.pdf — N-track × 6-plane signal grid (shared colour scale)

Usage:
  python tools/plot_diffusion_study_tracks.py
  python tools/plot_diffusion_study_tracks.py --output-dir plots/diffusion_study_tracks
"""
import argparse
import html as _html
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from src.plots.plot_mixed_tracks_edep_wireplanes import (
    build_simulator,
    track_and_forward,
    write_edep_3d_html,
    _safe_stem,
)

_PLOTS_DIR = os.environ.get('PLOTS_DIR', 'plots')

# ADC cutoffs for pixel-retention plots
ADC_CUTOFFS_PIXEL = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

# ---------------------------------------------------------------------------
# Track definitions
# ---------------------------------------------------------------------------

_MUON4_DIR    = (-0.694627880, 0.476880059, 0.538588450)
_MUON4_ENERGY = 100   # MeV
_ANGLE_START  = (1900.0, 0.0, 0.0)
_ANGLE_ENERGY = 400   # MeV


def _angle_dir(theta_deg):
    r = math.radians(theta_deg)
    return (round(-math.cos(r), 9), round(math.sin(r), 9), 0.0)


TRACKS = [
    # name, direction, energy_mev, start_position_mm
    ("Muon4_100MeV_startx-2000",  _MUON4_DIR,       _MUON4_ENERGY, (-2000.0, 0.0, 0.0)),
    ("Muon4_100MeV_startx-1000",  _MUON4_DIR,       _MUON4_ENERGY, (-1000.0, 0.0, 0.0)),
    ("Muon4_100MeV_startx0",      _MUON4_DIR,       _MUON4_ENERGY, (   0.0, 0.0, 0.0)),
    ("Muon_400MeV_theta10",  _angle_dir(10),   _ANGLE_ENERGY, _ANGLE_START),
    ("Muon_400MeV_theta30",  _angle_dir(30),   _ANGLE_ENERGY, _ANGLE_START),
    ("Muon_400MeV_theta90",  _angle_dir(90),   _ANGLE_ENERGY, _ANGLE_START),
]

_LABELS = {
    "Muon4_100MeV_startx-2000": "Muon4 100 MeV  |start_x|=2000 mm (near anode)",
    "Muon4_100MeV_startx-1000": "Muon4 100 MeV  |start_x|=1000 mm",
    "Muon4_100MeV_startx0":     "Muon4 100 MeV  |start_x|=0 mm (cathode, zero deposits expected)",
    "Muon_400MeV_theta10":      "400 MeV  θ=10° (nearly along drift)",
    "Muon_400MeV_theta30":      "400 MeV  θ=30°",
    "Muon_400MeV_theta90":      "400 MeV  θ=90° (transverse)",
}

# Short labels used in the ADC retention legend
_SHORT_LABELS = {
    "Muon4_100MeV_startx-2000": "Muon4 |x|=2000 mm",
    "Muon4_100MeV_startx-1000": "Muon4 |x|=1000 mm",
    "Muon4_100MeV_startx0":     "Muon4 |x|=0 mm",
    "Muon_400MeV_theta10":      "400 MeV θ=10°",
    "Muon_400MeV_theta30":      "400 MeV θ=30°",
    "Muon_400MeV_theta90":      "400 MeV θ=90°",
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir',
                   default=os.path.join(_PLOTS_DIR, 'diffusion_study_tracks'),
                   help='Output directory (default: $PLOTS_DIR/diffusion_study_tracks)')
    p.add_argument('--signal-percentile', type=float, default=99.0,
                   help='Percentile of |signal| used for wire-plane colour scale (default: 99)')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Building simulator …")
    simulator = build_simulator()
    cfg = simulator.config
    time_step_us = float(cfg.time_step_us)
    n_p = cfg.volumes[0].n_planes
    col_labels = [f'{cfg.plane_names[v][p]}{v + 1}'
                  for v in range(cfg.n_volumes) for p in range(n_p)]

    specs      = []
    tracks_raw = []
    all_planes = []
    stats_list = []

    for name, direction, energy_mev, start_mm in TRACKS:
        spec = {'name': name, 'direction': direction, 'momentum_mev': float(energy_mev)}
        print(f"  Simulating {name} …")
        track, planes = track_and_forward(simulator, spec, start_position_mm=list(start_mm))
        de = np.asarray(track['de'])
        st = {
            'n_deposits': int(len(de)),
            'mean_de':    float(de.mean()) if len(de) else 0.0,
            'total_de':   float(de.sum())  if len(de) else 0.0,
        }
        specs.append(spec)
        tracks_raw.append(track)
        all_planes.append(planes)
        stats_list.append(st)
        print(f"    → {st['n_deposits']} deposits, total dE = {st['total_de']:.4g} MeV")

    # ── shared dE colour scale ───────────────────────────────────────────────
    all_de = np.concatenate([np.asarray(t['de']) for t in tracks_raw if len(t['de']) > 0])
    if len(all_de) > 0:
        de_min   = float(np.percentile(all_de, 1))
        de_max   = float(np.percentile(all_de, 99))
        de_range = max(de_max - de_min, 1e-9)
    else:
        de_min = de_max = de_range = 1.0

    print("Writing per-track HTML files …")
    for spec, track, st in zip(specs, tracks_raw, stats_list):
        stem      = _safe_stem(spec['name'])
        html_path = os.path.join(args.output_dir, f'edep_3d_{stem}.html')
        write_edep_3d_html(track, spec, html_path,
                           de_min, de_max, de_range, cfg.volumes, stats=st)
        print(f"  Saved {html_path}")

    # ── ADC pixel-retention data ─────────────────────────────────────────────
    adc_pixel_data = []
    for (name, *_rest), planes in zip(TRACKS, all_planes):
        per_plane_pcts = []
        for plane in planes:
            abs_plane = np.abs(np.asarray(plane, dtype=np.float32))
            total = abs_plane.size
            pcts  = [round(float(100.0 * np.sum(abs_plane >= c) / total), 3)
                     for c in ADC_CUTOFFS_PIXEL]
            per_plane_pcts.append(pcts)
        adc_pixel_data.append({
            'name':  _SHORT_LABELS.get(name, name),
            'planes': per_plane_pcts,
        })

    # ── wire-plane PDF ───────────────────────────────────────────────────────
    abs_stack = np.concatenate([np.abs(p).ravel() for planes in all_planes for p in planes])
    vmax_sig  = max(float(np.nanpercentile(abs_stack, args.signal_percentile)), 1e-9)
    norm_2d   = mcolors.Normalize(vmin=-vmax_sig, vmax=vmax_sig)

    n_tracks = len(specs)
    pdf_name = f'wireplanes_{n_tracks}x6_gt_signals.pdf'
    pdf_path = os.path.join(args.output_dir, pdf_name)
    print(f"Writing wire-plane PDF ({n_tracks} tracks × 6 planes) …")
    fig, axes = plt.subplots(n_tracks, 6, figsize=(18, max(8.0, 3.0 * n_tracks)),
                              constrained_layout=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(
        f'GT wire-plane signals  (shared scale ±{vmax_sig:.3g} e⁻, p{args.signal_percentile:g} |all|)\n'
        f'dE colour range [{de_min:.5g}, {de_max:.5g}] MeV',
        fontsize=11,
    )
    for i, (spec, planes) in enumerate(zip(specs, all_planes)):
        row_label = _LABELS.get(spec['name'], spec['name'])
        for j, plane in enumerate(planes):
            ax = axes[i, j]
            n_wires, n_time = plane.shape
            ax.imshow(plane, aspect='auto', origin='lower', norm=norm_2d, cmap='RdBu_r',
                      extent=[0, n_time * time_step_us, 0, n_wires])
            if i == 0:
                ax.set_title(col_labels[j], fontsize=9)
            ax.set_ylabel((row_label + '\nwire') if j == 0 else 'wire', fontsize=7,
                          rotation=90, va='center')
            ax.set_xlabel('t (μs)', fontsize=7)
            ax.tick_params(axis='both', labelsize=6)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm_2d)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.35, aspect=60, pad=0.02, label='signal (e⁻)')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {pdf_path}")

    # ── index.html ───────────────────────────────────────────────────────────
    print("Writing index.html …")
    labelled_specs = [dict(s, name=_LABELS.get(s['name'], s['name'])) for s in specs]
    _write_index(specs, labelled_specs, stats_list, args.output_dir,
                 pdf_name=pdf_name,
                 adc_data=adc_pixel_data,
                 col_labels=col_labels,
                 adc_cutoffs=ADC_CUTOFFS_PIXEL)
    print(f"  Saved {os.path.join(args.output_dir, 'index.html')}")
    print("Done.")


# ---------------------------------------------------------------------------
# index.html template (uses __PLACEHOLDER__ substitution, not f-strings,
# to avoid escaping all the JS curly braces)
# ---------------------------------------------------------------------------

_INDEX_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Diffusion study tracks</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:system-ui,sans-serif;background:#f0f2f5;color:#1a1a1a}
    .page-hdr{background:#1a1a2e;color:#eee;padding:12px 20px}
    .page-hdr h1{font-size:1.05rem;font-weight:600;margin-bottom:3px}
    .page-hdr p{font-size:0.78rem;color:#aac4ff;margin:0}
    .tabs{display:flex;background:#16213e;padding:0 20px;gap:0}
    .tab-btn{padding:9px 20px;border:none;cursor:pointer;background:transparent;
             color:#8899aa;font-size:0.85rem;font-weight:500;
             border-bottom:3px solid transparent;transition:color .15s}
    .tab-btn.active{color:#fff;border-bottom-color:#e94560}
    .tab-btn:hover:not(.active){color:#ccd}
    .tab-pane{display:none;padding:14px 20px 20px}
    .tab-pane.active{display:block}
    .toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:.8rem;margin-bottom:.4rem}
    select{padding:6px 10px;border:1px solid #d0d5dd;border-radius:5px;
           background:#fff;font-size:.87rem;min-width:140px}
    .stats-bar{font-size:.82rem;color:#444;font-family:monospace;
               background:#f5f5f5;padding:.25rem .5rem;border-radius:4px;display:inline-block}
    iframe{width:100%;height:calc(100vh - 13.5rem);border:1px solid #ccc;border-radius:4px;margin-top:.4rem}
    .seg-btns{display:flex;gap:0}
    .seg-btns button{padding:6px 15px;border:1px solid #d0d5dd;background:#fff;
                     font-size:.85rem;cursor:pointer;color:#444;transition:all .12s}
    .seg-btns button:first-child{border-radius:5px 0 0 5px}
    .seg-btns button:last-child{border-radius:0 5px 5px 0}
    .seg-btns button:not(:first-child){border-left:none}
    .seg-btns button.active{background:#e94560;color:#fff;border-color:#e94560}
    .plot-wrap{background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.09);overflow:hidden}
  </style>
</head>
<body>
<div class="page-hdr">
  <h1>Diffusion study tracks — energy deposits &amp; wire-plane signals</h1>
  <p>Muon4: 100 MeV, east volume, dir (−0.695, 0.477, 0.539) &nbsp;|&nbsp;
     Angle: 400 MeV, start (1900, 0, 0) mm, XY-plane rotation</p>
</div>
<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('view3d',this)">3D View</button>
  <button class="tab-btn"        onclick="switchTab('adc',this)">ADC Retention</button>
</div>

<!-- ── Tab 1: 3D View ──────────────────────────────────────────────────────── -->
<div id="pane-view3d" class="tab-pane active">
  <div class="toolbar">
    <div>
      <label for="track-select"><strong>Track</strong></label>&nbsp;
      <select id="track-select" aria-label="Choose track">
__OPTIONS__
      </select>
    </div>
    __PDF_LINK__
  </div>
  __STATS_PANEL__
  <iframe id="track-frame" src="__FIRST__" title="3D energy deposits"></iframe>
</div>

<!-- ── Tab 2: ADC Retention ────────────────────────────────────────────────── -->
<div id="pane-adc" class="tab-pane">
  <div style="display:flex;align-items:center;gap:1rem;margin-bottom:12px;flex-wrap:wrap">
    <span style="font-size:.85rem;color:#555;font-weight:600">View:</span>
    <div class="seg-btns">
      <button class="adc-btn active" data-v="per-plane" onclick="setAdcView('per-plane')">Per wireplane</button>
      <button class="adc-btn"        data-v="per-group" onclick="setAdcView('per-group')">Per group (U / V / Y)</button>
      <button class="adc-btn"        data-v="all"       onclick="setAdcView('all')">All planes</button>
    </div>
    <div class="seg-btns" style="margin-left:.5rem">
      <button id="log-btn" onclick="toggleLog()" style="padding:6px 12px;border:1px solid #d0d5dd;background:#fff;font-size:.85rem;cursor:pointer;border-radius:5px;color:#444">Log Y</button>
    </div>
  </div>
  <div class="plot-wrap"><div id="plot-adc"></div></div>
  <p style="margin-top:8px;font-size:.78rem;color:#888">
    % of pixels with |signal| &ge; ADC cutoff.
    Wire planes: __COL_LABELS_STR__. ADC cutoffs: __ADC_CUTOFFS_STR__ e&minus;.
  </p>
</div>

<script>
// ── 3D-view controls ────────────────────────────────────────────────────────
__STATS_JS__
const sel3d = document.getElementById('track-select');
const frame  = document.getElementById('track-frame');
__STATS_SCRIPT__

function switchTab(id, btn) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('pane-' + id).classList.add('active');
  btn.classList.add('active');
  if (id === 'adc') {
    const active = document.querySelector('.adc-btn.active');
    plotAdcRetention(active ? active.dataset.v : 'per-plane');
  }
}

// ── ADC retention plots ─────────────────────────────────────────────────────
const ADC_DATA    = __ADC_DATA_JSON__;
const COL_LABELS  = __COL_LABELS_JSON__;
const ADC_CUTS    = __ADC_CUTS_JSON__;
const TRACK_COLS  = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b'];
const PLN_GROUPS  = [{label:'U',idx:[0,3]},{label:'V',idx:[1,4]},{label:'Y',idx:[2,5]}];

function avgPcts(track, indices) {
  return ADC_CUTS.map(function(_,ci) {
    return indices.reduce(function(s,pi){return s+track.planes[pi][ci];},0)/indices.length;
  });
}
function ax(i)  { return i===0 ? ''   : String(i+1); }
function xk(i)  { return i===0 ? 'xaxis'  : 'xaxis'+String(i+1); }
function yk(i)  { return i===0 ? 'yaxis'  : 'yaxis'+String(i+1); }
function xref(i){ return i===0 ? 'x domain' : 'x'+String(i+1)+' domain'; }
function yref(i){ return i===0 ? 'y domain' : 'y'+String(i+1)+' domain'; }

function setAdcView(v) {
  document.querySelectorAll('.adc-btn').forEach(function(b){
    b.classList.toggle('active', b.dataset.v===v);
  });
  plotAdcRetention(v);
}

function plotAdcRetention(view) {
  var traces = [];
  var layout = {
    paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    margin:{t:50,b:50,l:58,r:200},
    showlegend:true,
    legend:{x:1.01,y:1,xanchor:'left',font:{size:10},tracegroupgap:2},
  };

  if (view === 'per-plane') {
    for (var pi=0; pi<6; pi++) {
      var an = ax(pi);
      ADC_DATA.forEach(function(track,ti) {
        traces.push({
          x:ADC_CUTS, y:track.planes[pi],
          name:track.name, legendgroup:String(ti), showlegend:pi===0,
          mode:'lines+markers', marker:{size:4},
          line:{color:TRACK_COLS[ti%TRACK_COLS.length],width:1.8},
          xaxis:'x'+an, yaxis:'y'+an,
          hovertemplate:'ADC≥%{x}: %{y:.1f}%<extra>'+track.name+'</extra>',
        });
      });
    }
    layout.grid = {rows:2,columns:3,pattern:'independent',ygap:0.36,xgap:0.08};
    layout.height = 600;
    COL_LABELS.forEach(function(label,pi) {
      layout[xk(pi)] = {
        title:{text:pi>=3?'ADC (e⁻)':'',font:{size:10}},
        gridcolor:'#eee',zeroline:false,tickfont:{size:9},
      };
      layout[yk(pi)] = {
        title:{text:pi%3===0?'% pixels':'',font:{size:10}},
        range:[0,100],gridcolor:'#eee',zeroline:false,tickfont:{size:9},
      };
    });
    layout.annotations = COL_LABELS.map(function(label,pi) {
      return {
        text:'<b>'+label+'</b>',
        xref:xref(pi), yref:yref(pi), x:0.5, y:1.12,
        xanchor:'center', yanchor:'bottom', showarrow:false, font:{size:11},
      };
    });

  } else if (view === 'per-group') {
    PLN_GROUPS.forEach(function(grp,gi) {
      var an = ax(gi);
      ADC_DATA.forEach(function(track,ti) {
        traces.push({
          x:ADC_CUTS, y:avgPcts(track,grp.idx),
          name:track.name, legendgroup:String(ti), showlegend:gi===0,
          mode:'lines+markers', marker:{size:5},
          line:{color:TRACK_COLS[ti%TRACK_COLS.length],width:1.8},
          xaxis:'x'+an, yaxis:'y'+an,
          hovertemplate:'ADC≥%{x}: %{y:.1f}%<extra>'+track.name+'</extra>',
        });
      });
    });
    layout.grid = {rows:1,columns:3,pattern:'independent',xgap:0.08};
    layout.height = 380;
    PLN_GROUPS.forEach(function(grp,gi) {
      layout[xk(gi)] = {title:{text:'ADC (e⁻)'},gridcolor:'#eee',zeroline:false};
      layout[yk(gi)] = {
        title:{text:gi===0?'% pixels':''},
        range:[0,100],gridcolor:'#eee',zeroline:false,
      };
    });
    layout.annotations = PLN_GROUPS.map(function(grp,gi) {
      return {
        text:'<b>'+grp.label+' planes (avg vol 0+1)</b>',
        xref:xref(gi), yref:yref(gi), x:0.5, y:1.12,
        xanchor:'center', yanchor:'bottom', showarrow:false, font:{size:12},
      };
    });

  } else {  // all
    ADC_DATA.forEach(function(track,ti) {
      var allAvg = ADC_CUTS.map(function(_,ci) {
        return track.planes.reduce(function(s,pl){return s+pl[ci];},0)/track.planes.length;
      });
      traces.push({
        x:ADC_CUTS, y:allAvg,
        name:track.name,
        mode:'lines+markers', marker:{size:6},
        line:{color:TRACK_COLS[ti%TRACK_COLS.length],width:2},
        hovertemplate:'ADC≥%{x}: %{y:.1f}%<extra>'+track.name+'</extra>',
      });
    });
    layout.height = 400;
    layout.xaxis = {title:{text:'ADC cutoff (e⁻)'},gridcolor:'#eee',zeroline:false};
    layout.yaxis = {
      title:{text:'% pixels with |signal| ≥ cutoff'},
      range:[0,100],gridcolor:'#eee',zeroline:false,
    };
    layout.title = {text:'ADC retention — all wire planes (mean over 6)',font:{size:13},x:0.5};
  }

  Plotly.react('plot-adc', traces, layout, {responsive:true, displayModeBar:false});
}
</script>
</body>
</html>
"""


def _write_index(specs, labelled_specs, stats_list, output_dir,
                 pdf_name=None, adc_data=None, col_labels=None, adc_cutoffs=None):
    has_stats = len(stats_list) == len(specs)

    option_lines  = []
    stats_js_rows = []
    for i, (spec, lspec) in enumerate(zip(specs, labelled_specs)):
        basename = f'edep_3d_{_safe_stem(spec["name"])}.html'
        label    = lspec['name']
        sel      = ' selected' if i == 0 else ''
        option_lines.append(
            f'        <option value="{_html.escape(basename, quote=True)}"{sel}>'
            f'{_html.escape(label)}</option>'
        )
        if has_stats:
            st = stats_list[i]
            stats_js_rows.append(
                f'  {{n:{st["n_deposits"]},mean:{st["mean_de"]:.6g},total:{st["total_de"]:.6g}}}'
            )

    first   = f'edep_3d_{_safe_stem(specs[0]["name"])}.html'
    options = '\n'.join(option_lines)

    if has_stats:
        stats_js     = 'const STATS=[\n' + ',\n'.join(stats_js_rows) + '\n];'
        stats_panel  = ('<div class="stats-bar" id="stats-bar"></div>')
        stats_script = (
            'function updateStats(idx){\n'
            '  var s=STATS[idx];\n'
            '  document.getElementById(\'stats-bar\').textContent=\n'
            '    \'N deposits: \'+s.n.toLocaleString()+\n'
            '    \'   mean dE: \'+s.mean.toPrecision(4)+\' MeV\'+\n'
            '    \'   total dE: \'+s.total.toPrecision(4)+\' MeV\';\n'
            '}\n'
            'updateStats(sel3d.selectedIndex);\n'
            'sel3d.addEventListener(\'change\',function(){'
            'frame.src=sel3d.value;updateStats(sel3d.selectedIndex);});'
        )
    else:
        stats_js     = ''
        stats_panel  = ''
        stats_script = "sel3d.addEventListener('change',function(){frame.src=sel3d.value;});"

    pdf_link = (
        f'<a href="{_html.escape(pdf_name, quote=True)}" target="_blank" '
        f'style="font-size:.85rem;text-decoration:none">&#128196; Wire-plane signals PDF</a>'
    ) if pdf_name else ''

    page = _INDEX_TMPL
    page = page.replace('__OPTIONS__',        options)
    page = page.replace('__FIRST__',          _html.escape(first))
    page = page.replace('__STATS_JS__',       stats_js)
    page = page.replace('__STATS_PANEL__',    stats_panel)
    page = page.replace('__STATS_SCRIPT__',   stats_script)
    page = page.replace('__PDF_LINK__',       pdf_link)
    page = page.replace('__ADC_DATA_JSON__',  json.dumps(adc_data or [], separators=(',', ':')))
    page = page.replace('__COL_LABELS_JSON__',json.dumps(col_labels or []))
    page = page.replace('__ADC_CUTS_JSON__',  json.dumps(adc_cutoffs or []))
    page = page.replace('__COL_LABELS_STR__', ', '.join(col_labels or []))
    page = page.replace('__ADC_CUTOFFS_STR__', ', '.join(str(c) for c in (adc_cutoffs or [])))

    with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(page)


if __name__ == '__main__':
    main()
