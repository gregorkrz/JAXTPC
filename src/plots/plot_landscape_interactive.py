#!/usr/bin/env python
"""
Build a self-contained interactive HTML viewer for 2D loss landscapes.

Scans pkl files produced by ``2d_loss_landscape.py``, embeds all data as JSON,
and writes a single HTML file powered by Plotly.js:
  - Pick parameter pair via X / Y dropdowns
  - Select one or more tracks (checkboxes) — grids are summed
  - Toggle noise (default: off)
  - Log-scale heatmap + gradient streamlines (RK4, computed in JS)
  - Ground-truth star marker

Handles both old-style pkls (recomb_alpha / recomb_beta_90 axes) and new-style
generic pair pkls.  For duplicate (track, pair, noise) entries across run dates,
the latest pkl path wins.

Usage
-----
  python src/plots/plot_landscape_interactive.py
  python src/plots/plot_landscape_interactive.py \\
      --landscape-dir /fs/ddn/.../results/landscape \\
      --output plots/landscape_interactive.html
"""
import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PARAM_LABELS = {
    'velocity_cm_us':         'Drift velocity (cm/μs)',
    'lifetime_us':            'Electron lifetime (μs)',
    'diffusion_trans_cm2_us': 'Transverse diffusion (cm²/μs)',
    'diffusion_long_cm2_us':  'Longitudinal diffusion (cm²/μs)',
    'recomb_alpha':           'Recombination α',
    'recomb_beta':            'Recombination β',
    'recomb_beta_90':         'Recombination β₉₀',
    'recomb_R':               'Recombination R',
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--landscape-dir',
        default='results/landscape',
        help='Root dir containing run-date subdirs',
    )
    p.add_argument(
        '--run-dates',
        nargs='+',
        default=None,
        metavar='DATE',
        help='Run-date subdir(s) to load (e.g. 20260508 20260508_noise). '
             'Default: scan entire --landscape-dir recursively.',
    )
    p.add_argument(
        '--output',
        default=str(_REPO_ROOT / 'plots' / 'landscape_interactive.html'),
        help='Output HTML path',
    )
    return p.parse_args()


# ── PKL loading ────────────────────────────────────────────────────────────────

def _load_one(path):
    """Load a pkl and return a normalized record dict, or None on failure."""
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
    except Exception as e:
        print('  warning: could not load %s: %s' % (path, e), file=sys.stderr)
        return None

    # Normalise old-style (alpha/beta90) and new-style (generic param_y/param_x) pkls
    param_y  = d.get('param_y',   'recomb_alpha')
    param_x  = d.get('param_x',   'recomb_beta_90')
    vals_y   = d.get('vals_y',    d.get('alpha_vals',  []))
    vals_x   = d.get('vals_x',    d.get('beta90_vals', []))
    gt_y     = d.get('gt_param_y', d.get('gt_alpha',   None))
    gt_x     = d.get('gt_param_x', d.get('gt_beta90',  None))
    grad_y   = d.get('grad_param_y', d.get('grad_alpha',   None))
    grad_x   = d.get('grad_param_x', d.get('grad_beta90',  None))
    grid     = d.get('grid', [])

    # Convert numpy arrays to plain lists
    import numpy as np
    def _to_list(v):
        return v.tolist() if isinstance(v, np.ndarray) else (list(v) if v is not None else None)

    return dict(
        path        = path,
        track_name  = str(d.get('track_name', 'unknown')),
        param_y     = param_y,
        param_x     = param_x,
        vals_y      = _to_list(vals_y),
        vals_x      = _to_list(vals_x),
        gt_y        = float(gt_y) if gt_y is not None else None,
        gt_x        = float(gt_x) if gt_x is not None else None,
        grid        = _to_list(grid),
        grad_y      = _to_list(grad_y),
        grad_x      = _to_list(grad_x),
        noise_scale = float(d.get('noise_scale', 0.0)),
        loss_name   = str(d.get('loss_name', '')),
        grid_size   = d.get('grid_size', '?'),
        range_frac  = float(d.get('range_frac', 0.0)),
    )


def load_pkls(landscape_dir, run_dates=None):
    if run_dates:
        paths = []
        for rd in run_dates:
            subdir = os.path.join(landscape_dir, rd)
            paths += glob.glob(os.path.join(subdir, '**', '*.pkl'), recursive=True)
        paths = sorted(paths)
        label = ', '.join(run_dates)
    else:
        paths = sorted(glob.glob(os.path.join(landscape_dir, '**', '*.pkl'), recursive=True))
        label = landscape_dir
    records = [r for r in (_load_one(p) for p in paths) if r is not None]
    print('Loaded %d/%d pkl files from %r' % (len(records), len(paths), label))
    return records


# ── Data assembly ──────────────────────────────────────────────────────────────

def _round(obj, digits=6):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, list):
        return [_round(x, digits) for x in obj]
    return obj


def build_js_data(records):
    """
    Returns (data, meta).

    data[pair_key][noise_key][track_name] = {
        vals_y, vals_x, gt_y, gt_x, grid,
        has_grad, grad_y?, grad_x?
    }
    pair_key  = "{param_y}__{param_x}"  (as stored in pkl)
    noise_key = "noise" | "no_noise"
    """
    data = {}
    meta = {}

    for r in records:
        pair_key  = '%s__%s' % (r['param_y'], r['param_x'])
        noise_key = 'noise' if r['noise_scale'] > 0 else 'no_noise'

        if pair_key not in data:
            data[pair_key] = {'no_noise': {}, 'noise': {}}
            meta[pair_key] = {
                'param_y':    r['param_y'],
                'param_x':    r['param_x'],
                'loss_name':  r['loss_name'],
                'grid_size':  r['grid_size'],
                'range_frac': r['range_frac'],
            }

        has_grad = r['grad_y'] is not None and r['grad_x'] is not None
        entry = {
            'vals_y':   r['vals_y'],
            'vals_x':   r['vals_x'],
            'gt_y':     r['gt_y'],
            'gt_x':     r['gt_x'],
            'grid':     _round(r['grid']),
            'has_grad': has_grad,
        }
        if has_grad:
            entry['grad_y'] = _round(r['grad_y'])
            entry['grad_x'] = _round(r['grad_x'])

        # Latest path wins for duplicate (track, pair, noise)
        data[pair_key][noise_key][r['track_name']] = entry

    return data, meta


# ── HTML emission ──────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>2D Loss Landscape Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;font-size:13px;background:#12121f;color:#ddd;display:flex;flex-direction:column;height:100vh;overflow:hidden}
header{padding:7px 14px;background:#1a1a30;border-bottom:1px solid #2a2a50;display:flex;align-items:center;gap:16px;flex-shrink:0}
header h1{font-size:14px;font-weight:600;color:#e05070;letter-spacing:.03em}
#hdr-status{font-size:11px;color:#888;margin-left:auto}
.main{display:flex;flex:1;overflow:hidden}
.sidebar{width:250px;min-width:200px;padding:10px;overflow-y:auto;background:#1a1a30;border-right:1px solid #2a2a50;display:flex;flex-direction:column;gap:12px;flex-shrink:0}
.sidebar h2{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#666;margin-bottom:3px}
.ctrl{display:flex;flex-direction:column;gap:5px}
select{background:#0d1224;color:#ddd;border:1px solid #3a3a60;border-radius:4px;padding:4px 7px;font-size:12px;width:100%}
select:focus{outline:none;border-color:#e05070}
.track-list{max-height:260px;overflow-y:auto;display:flex;flex-direction:column;gap:2px;background:#0d1224;border-radius:4px;padding:5px;border:1px solid #2a2a50}
.tk{display:flex;align-items:center;gap:5px;padding:2px 3px;border-radius:3px;cursor:pointer}
.tk:hover{background:#1f2a50}
.tk input{cursor:pointer;accent-color:#e05070;flex-shrink:0}
.tk label{cursor:pointer;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:190px}
.row{display:flex;align-items:center;gap:7px}
.row input{accent-color:#e05070;cursor:pointer}
.row label{cursor:pointer;font-size:12px}
.mini-btns{display:flex;gap:4px}
.btn{border:none;border-radius:4px;padding:4px 9px;cursor:pointer;font-size:11px}
.btn-accent{background:#e05070;color:#fff}.btn-accent:hover{background:#c03050}
.btn-sec{background:#0d1224;color:#bbb;border:1px solid #3a3a60}.btn-sec:hover{background:#1f2a50}
#status{font-size:11px;color:#777;min-height:16px;padding-top:2px}
#plot-area{flex:1;min-width:0}
</style>
</head>
<body>
<header>
  <h1>2D Loss Landscape Explorer</h1>
  <span id="hdr-status"></span>
</header>
<div class="main">
  <div class="sidebar">

    <div class="ctrl">
      <h2>Parameter Y axis</h2>
      <select id="sel-y"></select>
    </div>

    <div class="ctrl">
      <h2>Parameter X axis</h2>
      <select id="sel-x"></select>
    </div>

    <div class="ctrl">
      <h2>Options</h2>
      <div class="row"><input type="checkbox" id="chk-noise"><label for="chk-noise">Noise</label></div>
      <div class="row"><input type="checkbox" id="chk-grad" checked><label for="chk-grad">Gradient streamlines</label></div>
      <div class="row" style="gap:5px;margin-top:3px">
        <label for="sld-density" style="white-space:nowrap;font-size:12px">Arrow density</label>
        <input type="range" id="sld-density" min="2" max="20" step="1" value="8" style="flex:1;accent-color:#e05070">
        <span id="lbl-density" style="font-size:11px;color:#aaa;min-width:18px;text-align:right">8</span>
      </div>
    </div>

    <div class="ctrl">
      <h2>Color scale max (log₁₀ loss)</h2>
      <div style="display:flex;gap:5px;align-items:center">
        <input type="number" id="inp-zmax" step="0.05" style="width:72px;background:#0d1224;color:#ddd;border:1px solid #3a3a60;border-radius:4px;padding:3px 5px;font-size:12px">
        <input type="range" id="sld-zmax" step="0.05" style="flex:1;accent-color:#e05070">
      </div>
      <div style="font-size:10px;color:#555;margin-top:2px">values above → gray</div>
    </div>

    <div class="ctrl">
      <h2>Tracks</h2>
      <div class="mini-btns">
        <button class="btn btn-sec" onclick="selAll(true)">All</button>
        <button class="btn btn-sec" onclick="selAll(false)">None</button>
      </div>
      <div class="track-list" id="track-list"></div>
    </div>

    <div id="status"></div>
    <div id="debug-stats" style="font-size:10px;color:#666;line-height:1.7;margin-top:4px;font-family:monospace"></div>
    <button class="btn btn-accent" onclick="render()" style="margin-top:auto">Update plot</button>

  </div>
  <div id="plot-area"></div>
</div>

<script>
/* ── Embedded data ─────────────────────────────────────────────────────── */
const DATA   = {DATA};
const META   = {META};
const TRACKS = {TRACKS};
const PAIRS  = {PAIRS};
const PLABELS = {PLABELS};
const VIRIDIS_GRAY=[
  [0,'#440154'],[0.143,'#482878'],[0.286,'#3e4989'],
  [0.429,'#31688e'],[0.571,'#26828e'],[0.714,'#1f9e89'],
  [0.857,'#35b779'],[0.999,'#fde725'],[1,'#a0a0a0'],
];
let _zmaxLocked=false;

/* ── Helpers ───────────────────────────────────────────────────────────── */
function pl(p){{ return PLABELS[p] || p; }}

function addZero2d(a, b){{
  // element-wise add 2-d arrays a and b (same shape), mutates a
  for(let i=0;i<a.length;i++) for(let j=0;j<a[i].length;j++) a[i][j]+=b[i][j];
}}

function zeroGrid(ny,nx){{
  return Array.from({{length:ny}},()=>new Float64Array(nx));
}}

function toRegularArray(g){{
  // convert Float64Array rows → plain arrays for Plotly
  return Array.from(g, row=>Array.from(row));
}}

/* ── Gradient arrows (simple quiver on subsampled grid) ─────────────────── */
function computeArrows(valsX, valsY, gradX, gradY, nGrid){{
  const nx=valsX.length, ny=valsY.length;
  const rangeX=valsX[nx-1]-valsX[0], rangeY=valsY[ny-1]-valsY[0];
  const arrowFrac=0.45/nGrid/4;  // fraction of each axis range
  const stepX=Math.max(1,Math.round(nx/nGrid));
  const stepY=Math.max(1,Math.round(ny/nGrid));
  const lineX=[], lineY=[], tipX=[], tipY=[], tipAngle=[];
  for(let iy=Math.floor(stepY/2);iy<ny;iy+=stepY){{
    for(let ix=Math.floor(stepX/2);ix<nx;ix+=stepX){{
      const x=valsX[ix], y=valsY[iy];
      // Convert physical-space gradient to log-space: d_loss/d_log(p) = d_loss/d_p * p
      // Then normalise by log-span for equal visual arrow length
      const logSpanX=Math.log(valsX[nx-1]/valsX[0])||1e-30;
      const logSpanY=Math.log(valsY[ny-1]/valsY[0])||1e-30;
      const gx_n=gradX[iy][ix]*x/logSpanX, gy_n=gradY[iy][ix]*y/logSpanY;
      const mag_n=Math.sqrt(gx_n*gx_n+gy_n*gy_n)||1e-30;
      const dx=-(gx_n/mag_n)*arrowFrac*rangeX, dy=-(gy_n/mag_n)*arrowFrac*rangeY;
      lineX.push(x, x+dx, null);
      lineY.push(y, y+dy, null);
      tipX.push(x+dx); tipY.push(y+dy);
      // Plotly marker angle: 0=up, clockwise; atan2(dx,dy) gives angle from up
      tipAngle.push(Math.atan2(dx, dy)*180/Math.PI);
    }}
  }}
  return {{lineX, lineY, tipX, tipY, tipAngle}};
}}

/* ── Controls ──────────────────────────────────────────────────────────── */
function buildParamDropdowns(){{
  // collect all param_y and param_x values seen across all pair keys
  const allParams={{}};
  for(const pk of PAIRS){{
    const [py,px]=pk.split('__');
    allParams[py]=true; allParams[px]=true;
  }}
  const params=Object.keys(allParams).sort();

  ['sel-y','sel-x'].forEach((id,idx)=>{{
    const sel=document.getElementById(id);
    sel.innerHTML='';
    params.forEach(p=>{{
      const o=document.createElement('option');
      o.value=p; o.textContent=pl(p); sel.appendChild(o);
    }});
    if(PAIRS.length){{
      const [py,px]=PAIRS[0].split('__');
      sel.value=idx===0?py:px;
    }}
    sel.addEventListener('change',()=>{{_zmaxLocked=false;render();}});
  }});
}}

function buildTrackList(){{
  const div=document.getElementById('track-list');
  div.innerHTML='';
  TRACKS.forEach(t=>{{
    const row=document.createElement('div'); row.className='tk';
    const cb=document.createElement('input'); cb.type='checkbox'; cb.id='t_'+t; cb.checked=true;
    cb.addEventListener('change',render);
    const lbl=document.createElement('label'); lbl.htmlFor='t_'+t; lbl.textContent=t; lbl.title=t;
    row.appendChild(cb); row.appendChild(lbl); div.appendChild(row);
  }});
}}

function selAll(v){{ TRACKS.forEach(t=>{{const c=document.getElementById('t_'+t);if(c)c.checked=v;}}); render(); }}

function updateTrackAvailability(){{
  const paramY=document.getElementById('sel-y').value;
  const paramX=document.getElementById('sel-x').value;
  const useNoise=document.getElementById('chk-noise').checked;
  const noiseKey=useNoise?'noise':'no_noise';
  let pairKey=paramY+'__'+paramX;
  if(!DATA[pairKey]) pairKey=paramX+'__'+paramY;
  const noiseData=(DATA[pairKey]||{{}})[noiseKey]||{{}};
  TRACKS.forEach(t=>{{
    const el=document.getElementById('t_'+t);
    if(!el) return;
    const hasData=!!noiseData[t];
    el.parentElement.style.opacity=hasData?'1':'0.3';
    el.parentElement.title=hasData?'':'No data computed for this pair';
  }});
}}

/* ── Render ────────────────────────────────────────────────────────────── */
function setStatus(msg){{ document.getElementById('status').textContent=msg; }}

function render(){{
  updateTrackAvailability();
  const paramY=document.getElementById('sel-y').value;
  const paramX=document.getElementById('sel-x').value;
  const useNoise=document.getElementById('chk-noise').checked;
  const showGrad=document.getElementById('chk-grad').checked;

  if(paramY===paramX){{ setStatus('Select two different parameters.'); return; }}

  // Try canonical key, then flipped
  let pairKey=paramY+'__'+paramX;
  let flipped=false;
  if(!DATA[pairKey]){{
    pairKey=paramX+'__'+paramY;
    flipped=true;
    if(!DATA[pairKey]){{ setStatus('No data for this parameter pair.'); Plotly.purge('plot-area'); return; }}
  }}

  const noiseKey=useNoise?'noise':'no_noise';
  const noiseData=DATA[pairKey][noiseKey]||{{}};

  const checked=TRACKS.filter(t=>{{const cb=document.getElementById('t_'+t);return cb&&cb.checked;}});
  const selected=checked.filter(t=>noiseData[t]);

  if(!selected.length){{
    const msg=checked.length?'No data computed for this pair yet ('+checked.length+' tracks checked, 0 available).':'No tracks selected.';
    setStatus(msg);
    Plotly.purge('plot-area');
    return;
  }}

  // Reference axes from first track (aligned across tracks for same pair)
  const ref=noiseData[selected[0]];
  const valsX=flipped?ref.vals_y:ref.vals_x;
  const valsY=flipped?ref.vals_x:ref.vals_y;
  const gtX  =flipped?ref.gt_y :ref.gt_x;
  const gtY  =flipped?ref.gt_x :ref.gt_y;
  const nx=valsX.length, ny=valsY.length;

  // Sum grids (and gradients) across selected tracks
  const sumGrid=zeroGrid(ny,nx);
  let sumGX=null, sumGY=null, hasGrad=showGrad;

  for(const t of selected){{
    const td=noiseData[t];
    // grid is [ny][nx] for canonical orientation; transpose if flipped
    const srcGrid=flipped?transposeToFloat(td.grid,nx,ny):td.grid;
    addZero2d(sumGrid,srcGrid);

    if(showGrad&&td.has_grad){{
      if(!sumGX){{ sumGX=zeroGrid(ny,nx); sumGY=zeroGrid(ny,nx); }}
      const srcGX=flipped?transposeToFloat(td.grad_y,nx,ny):td.grad_x;
      const srcGY=flipped?transposeToFloat(td.grad_x,nx,ny):td.grad_y;
      addZero2d(sumGX,srcGX);
      addZero2d(sumGY,srcGY);
    }} else if(showGrad&&!td.has_grad){{
      hasGrad=false;
    }}
  }}

  // log10 transform for heatmap (null = non-positive → transparent)
  const logZ=Array.from({{length:ny}},(_,i)=>
    Array.from({{length:nx}},(_,j)=>{{
      const v=sumGrid[i][j]; return v>0?Math.log10(v):null;
    }})
  );

  // ── Color-scale clipping control ────────────────────────────────────────
  const flatZ=logZ.flat().filter(v=>v!=null);
  const zDataMin=flatZ.length?flatZ.reduce((a,b)=>Math.min(a,b),Infinity):0;
  const zDataMax=flatZ.length?flatZ.reduce((a,b)=>Math.max(a,b),-Infinity):1;
  const sld=document.getElementById('sld-zmax');
  const inp=document.getElementById('inp-zmax');
  sld.min=zDataMin.toFixed(5); sld.max=zDataMax.toFixed(5);
  if(!_zmaxLocked){{sld.value=zDataMax.toFixed(5);inp.value=zDataMax.toFixed(5);}}
  const threshold=Math.min(zDataMax,Math.max(zDataMin,parseFloat(inp.value)||zDataMax));
  const plotZmax=Math.max(threshold,zDataMin+1e-9);
  const zMean=flatZ.length?flatZ.reduce((a,b)=>a+b,0)/flatZ.length:0;
  const aboveThresh=flatZ.filter(v=>v>plotZmax).length;
  document.getElementById('debug-stats').innerHTML=
    'cells: '+nx*ny+' ('+flatZ.length+' valid, '+aboveThresh+' grayed)<br>'+
    'x: '+valsX[0].toPrecision(4)+' … '+valsX[nx-1].toPrecision(4)+'<br>'+
    'y: '+valsY[0].toPrecision(4)+' … '+valsY[ny-1].toPrecision(4)+'<br>'+
    'log₁₀ min: '+zDataMin.toFixed(3)+'<br>'+
    'log₁₀ max: '+zDataMax.toFixed(3)+'<br>'+
    'log₁₀ mean: '+zMean.toFixed(3);

  const traces=[];

  // Heatmap
  traces.push({{
    type:'heatmap', x:valsX, y:valsY, z:logZ,
    colorscale:VIRIDIS_GRAY, showscale:true,
    zmin:zDataMin, zmax:plotZmax,
    colorbar:{{title:{{text:'log₁₀(loss)',side:'right'}},thickness:14,len:.9,tickfont:{{color:'#bbb'}},titlefont:{{color:'#bbb'}}}},
    hovertemplate:pl(paramX)+': %{{x:.5g}}<br>'+pl(paramY)+': %{{y:.5g}}<br>log₁₀(loss): %{{z:.3f}}<extra></extra>',
  }});

  // Gradient arrows
  if(hasGrad&&sumGX){{
    const nGrid=parseInt(document.getElementById('sld-density').value);
    const gxArr=toRegularArray(sumGX), gyArr=toRegularArray(sumGY);
    const arr=computeArrows(valsX,valsY,gxArr,gyArr,nGrid);
    traces.push({{
      type:'scatter',mode:'lines',x:arr.lineX,y:arr.lineY,
      line:{{color:'rgba(255,255,255,0.55)',width:1.5}},
      showlegend:false,hoverinfo:'skip',
    }});
    traces.push({{
      type:'scatter',mode:'markers',x:arr.tipX,y:arr.tipY,
      marker:{{symbol:'arrow',size:7,angle:arr.tipAngle,color:'rgba(255,255,255,0.55)'}},
      showlegend:false,hoverinfo:'skip',
    }});
  }}

  // GT star
  if(gtX!=null&&gtY!=null) traces.push({{
    type:'scatter',mode:'markers',x:[gtX],y:[gtY],
    marker:{{symbol:'star',size:16,color:'#ff4466',line:{{width:1,color:'white'}}}},
    name:'Ground truth',
    hovertemplate:'GT: ('+Number(gtX).toPrecision(5)+', '+Number(gtY).toPrecision(5)+')<extra></extra>',
  }});

  const m=META[pairKey]||{{}};
  const subtitle=[
    selected.length+' track'+(selected.length>1?'s':'')+' summed',
    useNoise?'noise':'no noise',
    m.grid_size?m.grid_size+'\xd7'+m.grid_size+' grid':'',
    m.range_frac?'\xb1'+(m.range_frac*100).toFixed(0)+'% around GT':'',
  ].filter(Boolean).join('  •  ');

  const layout={{
    paper_bgcolor:'#12121f', plot_bgcolor:'#0a0a1a',
    font:{{color:'#ccc',size:12}},
    title:{{text:pl(paramY)+' vs '+pl(paramX)+'<br><sup>'+subtitle+'</sup>',font:{{size:14}},x:.5}},
    xaxis:{{title:pl(paramX),color:'#aaa',gridcolor:'#252545',linecolor:'#333',zerolinecolor:'#333'}},
    yaxis:{{title:pl(paramY),color:'#aaa',gridcolor:'#252545',linecolor:'#333',zerolinecolor:'#333'}},
    margin:{{t:75,b:55,l:75,r:20}},
    legend:{{x:1.02,bgcolor:'rgba(0,0,0,0)',font:{{color:'#ccc'}}}},
  }};

  Plotly.react('plot-area',traces,layout,{{responsive:true,displayModeBar:true,
    modeBarButtonsToRemove:['lasso2d','select2d']}});

  const missingNote=selected.length<checked.length?' ('+selected.length+'/'+checked.length+' have data)':'';
  setStatus(selected.length+' track'+(selected.length>1?'s':'')+missingNote+
    '  ·  '+nx+'\xd7'+ny+
    (hasGrad&&sumGX?'  ·  gradients':''));
  document.getElementById('hdr-status').textContent=pl(paramY)+' vs '+pl(paramX);
}}

function transposeToFloat(grid,newNy,newNx){{
  // grid is [oldNy][oldNx], returns Float64Array rows of shape [newNy][newNx]
  const out=zeroGrid(newNy,newNx);
  for(let i=0;i<newNy;i++) for(let j=0;j<newNx;j++) out[i][j]=grid[j][i];
  return out;
}}

/* ── Init ──────────────────────────────────────────────────────────────── */
buildParamDropdowns();
buildTrackList();
document.getElementById('chk-noise').addEventListener('change',()=>{{_zmaxLocked=false;render();}});
document.getElementById('chk-grad').addEventListener('change',render);
document.getElementById('sld-density').addEventListener('input',()=>{{
  document.getElementById('lbl-density').textContent=document.getElementById('sld-density').value;
  render();
}});
document.getElementById('sld-zmax').addEventListener('input',()=>{{
  document.getElementById('inp-zmax').value=parseFloat(document.getElementById('sld-zmax').value).toFixed(2);
  _zmaxLocked=true; render();
}});
document.getElementById('inp-zmax').addEventListener('change',()=>{{
  const v=parseFloat(document.getElementById('inp-zmax').value);
  if(!isNaN(v)) document.getElementById('sld-zmax').value=v;
  _zmaxLocked=true; render();
}});
render();
</script>
</body>
</html>
"""


def emit_html(data, meta, output_path):
    all_tracks = sorted({
        t
        for pair_dict in data.values()
        for noise_dict in pair_dict.values()
        for t in noise_dict
    })
    all_pairs = sorted(data.keys())

    sep = (',', ':')
    # Unescape {{ / }} in the JS template first, before substituting data
    # (so the replace doesn't corrupt }} that appear naturally in the JSON).
    template = _HTML_TEMPLATE.replace('{{', '{').replace('}}', '}')
    html = template \
        .replace('{DATA}',    json.dumps(data,         separators=sep)) \
        .replace('{META}',    json.dumps(meta,         separators=sep)) \
        .replace('{TRACKS}',  json.dumps(all_tracks,   separators=sep)) \
        .replace('{PAIRS}',   json.dumps(all_pairs,    separators=sep)) \
        .replace('{PLABELS}', json.dumps(PARAM_LABELS, separators=sep))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding='utf-8')
    size_kb = out.stat().st_size // 1024
    print('Wrote %s  (%d KB)' % (out, size_kb))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    records = load_pkls(args.landscape_dir, args.run_dates)
    if not records:
        print('No pkl files found. Nothing to do.')
        sys.exit(0)
    data, meta = build_js_data(records)
    emit_html(data, meta, args.output)


if __name__ == '__main__':
    main()
