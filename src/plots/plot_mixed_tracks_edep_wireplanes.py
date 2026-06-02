#!/usr/bin/env python
"""
3-D energy-deposit (dE) visualizations and 2-D wire-plane GT signals.

**Default:** 12 random muons ``Muon1_{T}MeV``, … plus **three** fixed 1000 MeV chords across
East+West: **Muon_diagCross_1000MeV** ((2000,2000,2000) mm toward (‑2000)³ mm),
**Muon_throughEw_skew02_1000MeV**, and **Muon_throughWe_skew03_1000MeV**
(oblique headings through both drift volumes).

Outputs
-------
  • One Plotly HTML per track: semi-transparent active-volume boxes from config
    plus 3-D scatter of segment positions coloured by dE (shared dE colour scale
    across all HTML files).
  • One PDF: N rows × 6 columns (default N=15) — rows are tracks, columns are
    U1, V1, Y1, U2, V2, Y2 (GT forward response, electrons; shared symmetric
    colour scale across all panels).
  • ``track_catalog.pdf`` — table of track name, T (MeV), and (dx, dy, dz).
  • ``index.html`` — dropdown (name, T, direction) to view each track's 3-D HTML in an iframe.

Usage
-----
    python src/plots/plot_mixed_tracks_edep_wireplanes.py
    python src/plots/plot_mixed_tracks_edep_wireplanes.py --output-dir plots/20260605/event_displays_12_tracks
    python src/plots/plot_mixed_tracks_edep_wireplanes.py --tracks 'diagonal:1,1,1:1000+xm:1,0.1,0.2:500'

Default ``--output-dir`` is ``$PLOTS_DIR/20260605/mixed_tracks_edep`` (normally ``plots/20260605/mixed_tracks_edep``).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

import argparse
import html
import json
import re
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go

from tools.geometry import generate_detector
from tools.loader import build_deposit_data
from tools.particle_generator import generate_muon_track
from tools.random_boundary_tracks import (
    N_DEFAULT_BOUNDARY_MUONS,
    filter_track_inside_volumes,
    generate_random_boundary_tracks,
)
from tools.simulation import DetectorSimulator

_PLOTS_DIR = os.path.join(os.environ.get('PLOTS_DIR', 'plots'))

CONFIG_PATH        = 'config/cubic_wireplane_config.yaml'
N_SEGMENTS         = 50_000
MAX_ACTIVE_BUCKETS = 1000
GT_LIFETIME_US     = 10_000.0
GT_VELOCITY_CM_US  = 0.160

# Bare names without :direction:momentum (optional ``--tracks`` presets)
_TRACK_NAME_ONLY = {
    'diagonal': ((1.0, 1.0, 1.0), 1000.0),
}


def parse_mixed_tracks(tracks_str: str):
    """Parse '+'-separated items: ``name:dx,dy,dz:T[:sx,sy,sz]`` or preset ``name`` only.

    The optional 4th colon-field is a comma-separated start position in mm.
    This matches the 4-field format used in ``_GRADIENT_15_TRACKS`` in submit_jobs.py.
    """
    specs = []
    for item in tracks_str.split('+'):
        item = item.strip()
        if not item:
            continue
        if ':' in item:
            parts = item.split(':')
            if len(parts) not in (3, 4):
                raise ValueError(
                    f'Expected name:dx,dy,dz:T or name:dx,dy,dz:T:sx,sy,sz, got {item!r}')
            name = parts[0].strip()
            direction = tuple(float(x) for x in parts[1].split(','))
            momentum_mev = float(parts[2])
            if len(direction) != 3:
                raise ValueError(f'Direction must have 3 components in {item!r}')
            spec = dict(name=name, direction=direction, momentum_mev=momentum_mev)
            if len(parts) == 4:
                start = tuple(float(x) for x in parts[3].split(','))
                if len(start) != 3:
                    raise ValueError(f'Start position must have 3 components in {item!r}')
                spec['start_position_mm'] = start
        else:
            if item not in _TRACK_NAME_ONLY:
                raise ValueError(
                    f'Unknown bare track name {item!r}. Known: {list(_TRACK_NAME_ONLY)}')
            direction, momentum_mev = _TRACK_NAME_ONLY[item]
            name = item
            spec = dict(name=name, direction=direction, momentum_mev=momentum_mev)
        specs.append(spec)
    if not specs:
        raise ValueError('No tracks parsed from --tracks')
    return specs


def _safe_stem(name: str) -> str:
    return re.sub(r'[^0-9A-Za-z._-]+', '_', name)


def outer_boundary_starts_mm(volumes):
    """Near East / West outer x faces but inset so positions satisfy loader ``[min,max)`` tests."""
    if len(volumes) < 2:
        raise ValueError('Expected at least 2 TPC volumes for East/West boundary starts')
    east, west = volumes[0], volumes[1]
    x_e_lo = east.ranges_cm[0][0] * 10.0
    x_w_hi = west.ranges_cm[0][1] * 10.0
    y_c = 0.5 * (east.ranges_cm[1][0] + east.ranges_cm[1][1]) * 10.0
    z_c = 0.5 * (east.ranges_cm[2][0] + east.ranges_cm[2][1]) * 10.0
    east_outer_mm = (x_e_lo, y_c, z_c)
    west_outer_mm = (x_w_hi, y_c, z_c)
    return east_outer_mm, west_outer_mm


def start_mm_for_track(spec, east_outer_mm, west_outer_mm):
    """Legacy ``--tracks``: ``_west`` suffix → West outer x face at volume yz center."""
    if spec['name'].endswith('_west'):
        return west_outer_mm
    return east_outer_mm


# Semi-transparent fills for active TPC boxes (cycle if n_volumes > len).
_VOLUME_MESH_COLORS = ('#4682B4', '#CD853F', '#6B8E23', '#9370DB')


def _detector_volume_mesh_traces(volumes, *, opacity=0.14):
    """Plotly Mesh3d cuboids from ``VolumeGeometry.ranges_cm`` (axis-aligned, mm)."""
    traces = []
    for idx, vol in enumerate(volumes):
        (x0, x1), (y0, y1), (z0, z1) = (
            (vol.ranges_cm[0][0] * 10.0, vol.ranges_cm[0][1] * 10.0),
            (vol.ranges_cm[1][0] * 10.0, vol.ranges_cm[1][1] * 10.0),
            (vol.ranges_cm[2][0] * 10.0, vol.ranges_cm[2][1] * 10.0),
        )
        # Vertices: bottom z0 then top z1; CCW faces seen from outside.
        x = (x0, x1, x1, x0, x0, x1, x1, x0)
        y = (y0, y0, y1, y1, y0, y0, y1, y1)
        z = (z0, z0, z0, z0, z1, z1, z1, z1)
        i = (0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1)
        j = (1, 2, 6, 7, 1, 5, 3, 7, 3, 7, 2, 6)
        k = (2, 3, 5, 6, 5, 4, 7, 6, 7, 4, 6, 5)
        color = _VOLUME_MESH_COLORS[idx % len(_VOLUME_MESH_COLORS)]
        traces.append(go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            color=color,
            opacity=opacity,
            name=f'Active volume {vol.volume_id}',
            showlegend=True,
            hoverinfo='skip',
            flatshading=True,
            lighting=dict(ambient=0.85, diffuse=0.35, specular=0.2),
        ))
    return traces


def build_simulator():
    detector_config = generate_detector(CONFIG_PATH)
    return DetectorSimulator(
        detector_config,
        differentiable=True,
        n_segments=N_SEGMENTS,
        use_bucketed=True,
        max_active_buckets=MAX_ACTIVE_BUCKETS,
        include_noise=False,
        include_electronics=False,
        include_track_hits=False,
        include_digitize=False,
        track_config=None,
    )


def track_and_forward(simulator, spec, start_position_mm):
    track = generate_muon_track(
        start_position_mm=start_position_mm,
        direction=spec['direction'],
        kinetic_energy_mev=spec['momentum_mev'],
        step_size_mm=0.1,
        track_id=1,
    )
    track = filter_track_inside_volumes(track, simulator.config.volumes)
    deposits = build_deposit_data(
        track['position'], track['de'], track['dx'], simulator.config,
        theta=track['theta'], phi=track['phi'],
        track_ids=track['track_id'],
    )
    gt_params = simulator.default_sim_params._replace(
        lifetime_us=jnp.array(GT_LIFETIME_US),
        velocity_cm_us=jnp.array(GT_VELOCITY_CM_US),
    )
    arrays = simulator.forward(gt_params, deposits)
    jax.block_until_ready(arrays)
    cfg = simulator.config
    n_p = cfg.volumes[0].n_planes
    planes = []
    for v in range(cfg.n_volumes):
        for p in range(n_p):
            planes.append(np.asarray(arrays[v * n_p + p], dtype=np.float64))
    return track, planes


def write_edep_3d_html(track, spec, path, de_min, de_max, de_range, volumes, stats=None):
    pos = np.asarray(track['position'])
    de = np.asarray(track['de'])
    fig = go.Figure()
    for tr in _detector_volume_mesh_traces(volumes):
        fig.add_trace(tr)
    if len(de) > 0:
        sizes = 3.0 + (de - de_min) / de_range * 15.0
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode='markers',
            name='dE segments',
            marker=dict(
                size=sizes,
                color=de,
                cmin=de_min,
                cmax=de_max,
                colorscale='Viridis',
                colorbar=dict(title='dE (MeV)'),
                opacity=1.0,
                # Default WebGL marker stroke reads as a bright halo when zoomed out.
                line=dict(width=0, color='rgba(0,0,0,0)'),
            ),
            text=[f'dE={v:.5f} MeV' for v in de],
            hovertemplate='%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>',
        ))
    if len(de) == 0:
        title = f"{spec['name']} — no deposits"
    else:
        stats_str = ''
        if stats is not None:
            stats_str = (f"  |  N={stats['n_deposits']:,}  "
                         f"mean dE={stats['mean_de']:.4g} MeV  "
                         f"total dE={stats['total_de']:.4g} MeV")
        title = (f"Energy deposits — {spec['name']}  dir={spec['direction']}  "
                 f"T={spec['momentum_mev']:.0f} MeV{stats_str}")
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x (mm)',
            yaxis_title='y (mm)',
            zaxis_title='z (mm)',
            aspectmode='data',
        ),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
        margin=dict(l=0, r=0, b=0, t=60),
    )
    fig.write_html(path)


def _track_kinematics_table(specs):
    """Closed <details> table: per-track kinetic energy, theta, phi."""
    if not specs:
        return ''
    rows = []
    for spec in specs:
        d = np.array(spec['direction'], dtype=float)
        norm = np.linalg.norm(d)
        if norm > 0:
            d = d / norm
        theta_deg = float(np.degrees(np.arccos(np.clip(d[2], -1.0, 1.0))))
        phi_deg = float(np.degrees(np.arctan2(d[1], d[0])))
        rows.append(
            f'<tr>'
            f'<td style="text-align:left;white-space:nowrap">{html.escape(spec["name"])}</td>'
            f'<td>{spec["momentum_mev"]:.6g}</td>'
            f'<td>{theta_deg:.2f}</td>'
            f'<td>{phi_deg:.2f}</td>'
            f'</tr>'
        )
    header = (
        '<th style="text-align:left">Track</th>'
        '<th>T (MeV)</th>'
        '<th>θ (°)</th>'
        '<th>φ (°)</th>'
    )
    return f'''<details style="margin:0.6rem 0">
  <summary style="cursor:pointer;font-weight:600;font-size:0.9rem">
    Track kinematics (θ from z-axis, φ = atan2(y,x))
  </summary>
  <table style="border-collapse:collapse;font-size:0.78rem;margin-top:0.4rem">
    <thead><tr style="background:#f0f0f0">{header}</tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</details>'''


def _pixel_count_table(specs, pixel_counts, plane_names=None):
    """Return a closed <details> with optional plane picker and pixel-count table.

    If ``plane_names`` and per-plane data ('by_plane_cutoff', 'plane_totals') are
    present in pixel_counts entries, renders an interactive JS plane picker.
    Otherwise falls back to a static combined table.
    """
    if not pixel_counts:
        return ''
    cutoffs = pixel_counts[0]['cutoffs']
    has_per_plane = (plane_names and 'by_plane_cutoff' in pixel_counts[0]
                     and 'plane_totals' in pixel_counts[0])

    if not has_per_plane:
        header = '<th>Track</th>' + ''.join(f'<th>≥{c}</th>' for c in cutoffs)
        rows = []
        for spec, pc in zip(specs, pixel_counts):
            total = pc['total']
            cells = ''.join(
                f'<td>{pc["by_cutoff"][c]:,}<br><small style="color:#888">'
                f'{100*pc["by_cutoff"][c]/total:.1f}%</small></td>'
                for c in cutoffs
            )
            rows.append(f'<tr><td style="text-align:left;white-space:nowrap">'
                        f'{html.escape(spec["name"])}</td>{cells}</tr>')
        return f'''<details style="margin:0.6rem 0">
  <summary style="cursor:pointer;font-weight:600;font-size:0.9rem">
    Pixels passing |ADC| ≥ cutoff (all 6 planes combined)
  </summary>
  <table style="border-collapse:collapse;font-size:0.78rem;margin-top:0.4rem">
    <thead><tr style="background:#f0f0f0">{header}</tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</details>'''

    n_planes = len(plane_names)
    js_plane_names = '[' + ', '.join(f'"{html.escape(n, quote=True)}"' for n in plane_names) + ']'
    js_cutoffs = '[' + ', '.join(str(c) for c in cutoffs) + ']'
    js_track_names = '[' + ', '.join(f'"{html.escape(spec["name"], quote=True)}"' for spec in specs) + ']'
    js_data_rows = []
    for pc in pixel_counts:
        pt = '[' + ', '.join(str(t) for t in pc['plane_totals']) + ']'
        per_plane = '[' + ', '.join(
            '[' + ', '.join(str(pc['by_plane_cutoff'][pi][c]) for c in cutoffs) + ']'
            for pi in range(n_planes)
        ) + ']'
        js_data_rows.append(f'  {{planeTotals: {pt}, byPlaneCutoff: {per_plane}}}')
    js_data = 'const PC_DATA = [\n' + ',\n'.join(js_data_rows) + '\n];'

    cb_html = ' '.join(
        f'<label style="margin-right:0.4rem"><input type="checkbox" class="pc-plane-cb"'
        f' value="{pi}" checked> {html.escape(pn)}</label>'
        for pi, pn in enumerate(plane_names)
    )

    return f'''<details style="margin:0.6rem 0">
  <summary style="cursor:pointer;font-weight:600;font-size:0.9rem">
    Pixels passing |ADC| ≥ cutoff
  </summary>
  <div style="margin:0.5rem 0 0.3rem;font-size:0.82rem">
    <strong>Planes:</strong> {cb_html}
    <button onclick="pcSelectAll(true)" style="margin-left:0.6rem;font-size:0.78rem">All</button>
    <button onclick="pcSelectAll(false)" style="font-size:0.78rem">None</button>
  </div>
  <div id="pc-table"></div>
  <script>
  (function() {{
    const PLANE_NAMES = {js_plane_names};
    const CUTOFFS = {js_cutoffs};
    const TRACK_NAMES = {js_track_names};
    {js_data}

    function render() {{
      const sel = Array.from(document.querySelectorAll('.pc-plane-cb:checked')).map(cb => +cb.value);
      const el = document.getElementById('pc-table');
      if (sel.length === 0) {{
        el.innerHTML = '<em style="color:#888;font-size:0.82rem">No planes selected.</em>';
        return;
      }}
      const label = sel.length === PLANE_NAMES.length
        ? 'all ' + PLANE_NAMES.length + ' planes'
        : sel.map(i => PLANE_NAMES[i]).join(', ');
      let hdr = '<th style="text-align:left">Track</th>';
      for (const c of CUTOFFS) hdr += '<th>≥' + c + '</th>';
      let tbody = '';
      for (let ti = 0; ti < TRACK_NAMES.length; ti++) {{
        const d = PC_DATA[ti];
        const total = sel.reduce((s, pi) => s + d.planeTotals[pi], 0);
        let cells = '';
        for (let ci = 0; ci < CUTOFFS.length; ci++) {{
          const cnt = sel.reduce((s, pi) => s + d.byPlaneCutoff[pi][ci], 0);
          const pct = total > 0 ? (100 * cnt / total).toFixed(1) : '0.0';
          cells += '<td>' + cnt.toLocaleString() + '<br><small style="color:#888">' + pct + '%</small></td>';
        }}
        tbody += '<tr><td style="text-align:left;white-space:nowrap">' + TRACK_NAMES[ti] + '</td>' + cells + '</tr>';
      }}
      el.innerHTML =
        '<div style="font-size:0.75rem;color:#666;margin-bottom:0.25rem">Showing: ' + label + '</div>' +
        '<table style="border-collapse:collapse;font-size:0.78rem"><thead><tr style="background:#f0f0f0">' +
        hdr + '</tr></thead><tbody>' + tbody + '</tbody></table>';
    }}

    document.querySelectorAll('.pc-plane-cb').forEach(cb => cb.addEventListener('change', render));
    window.pcSelectAll = function(v) {{
      document.querySelectorAll('.pc-plane-cb').forEach(cb => {{ cb.checked = v; }});
      render();
    }};
    render();
  }})();
  </script>
</details>'''


def _compute_drift_dist_stats(tracks_raw):
    """Per-track deposit-distance stats to wireplanes at x=+2000 mm (east) and x=−2000 mm (west).

    Returns a list (one entry per track) of dicts with keys 'east' and 'west', each containing
    min / max / mean / ewm (energy-weighted mean, weights = dE) / stdev in mm.
    Entry is None if the track has no deposits.
    """
    result = []
    for track in tracks_raw:
        pos = np.asarray(track['position'])
        de  = np.asarray(track['de'])
        if len(pos) == 0:
            result.append(None)
            continue
        px = pos[:, 0]
        de_sum = float(de.sum())
        plane_stats = {}
        for name, xp in (('east', 2000.0), ('west', -2000.0)):
            d    = np.abs(px - xp)
            mean = float(d.mean())
            ewm  = float((de * d).sum() / de_sum) if de_sum > 0 else float('nan')
            plane_stats[name] = dict(
                min=float(d.min()), max=float(d.max()),
                mean=mean,
                ewm=ewm,
                stdev=float(np.sqrt(np.mean((d - mean) ** 2))),
            )
        result.append(plane_stats)
    return result


def _drift_dist_table(specs, dist_stats):
    """Closed <details> table: per-track drift-distance stats to east and west wireplanes."""
    if not dist_stats or not any(d is not None for d in dist_stats):
        return ''

    def fmt(v):
        if v is None or (isinstance(v, float) and v != v):
            return '—'
        return f'{v:.1f}'

    col_keys = ('min', 'max', 'mean', 'ewm', 'stdev')
    col_heads = 'min max mean ewm stdev'.split()
    header = (
        '<tr style="background:#f0f0f0">'
        '<th rowspan="2" style="text-align:left;vertical-align:bottom">Track</th>'
        '<th colspan="5" style="text-align:center">East wireplane (x = +2000 mm)</th>'
        '<th colspan="5" style="text-align:center">West wireplane (x = −2000 mm)</th>'
        '</tr>'
        '<tr style="background:#f5f5f5">'
        + ''.join(f'<th>{h}</th>' for h in col_heads * 2)
        + '</tr>'
    )
    rows_html = []
    for spec, ds in zip(specs, dist_stats):
        if ds is None:
            cells = '<td colspan="10" style="text-align:center;color:#888">no deposits</td>'
        else:
            cells = ''
            for side in ('east', 'west'):
                cells += ''.join(f'<td>{fmt(ds[side][k])}</td>' for k in col_keys)
        rows_html.append(
            f'<tr><td style="text-align:left;white-space:nowrap">'
            f'{html.escape(spec["name"])}</td>{cells}</tr>'
        )

    return f'''<details style="margin:0.6rem 0">
  <summary style="cursor:pointer;font-weight:600;font-size:0.9rem">
    Drift distance to wireplanes (mm)
  </summary>
  <div style="font-size:0.72rem;color:#666;margin:0.3rem 0 0.25rem">
    All values in mm &nbsp;|&nbsp; ewm = energy-weighted mean (weights = dE per deposit)
  </div>
  <table style="border-collapse:collapse;font-size:0.78rem">
    <thead>{header}</thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</details>'''


def write_edep_index_html(specs, output_dir, stats=None, pdf_name=None, pixel_counts=None, plane_names=None, dist_stats=None):
    """Single-page picker: iframe loads ``edep_3d_<stem>.html`` for the selected track."""
    if not specs:
        return
    has_stats = stats is not None and len(stats) == len(specs)
    option_lines = []
    stats_js_rows = []
    for i, spec in enumerate(specs):
        basename = f'edep_3d_{_safe_stem(spec["name"])}.html'
        d = spec['direction']
        dir_str = f'({d[0]:.6g}, {d[1]:.6g}, {d[2]:.6g})'
        label = f'{spec["name"]} — T={spec["momentum_mev"]:.6g} MeV — dir={dir_str}'
        sel = ' selected' if i == 0 else ''
        option_lines.append(
            f'        <option value="{html.escape(basename, quote=True)}"{sel}>'
            f'{html.escape(label)}</option>'
        )
        if has_stats:
            st = stats[i]
            stats_js_rows.append(
                f'  {{n: {st["n_deposits"]}, mean: {st["mean_de"]:.6g}, total: {st["total_de"]:.6g}}}'
            )

    first = f'edep_3d_{_safe_stem(specs[0]["name"])}.html'
    options_block = '\n'.join(option_lines)

    if has_stats:
        stats_js = 'const STATS = [\n' + ',\n'.join(stats_js_rows) + '\n];'
        stats_panel = """
  <div id="stats-bar" style="margin:0.4rem 0;font-size:0.85rem;color:#444;font-family:monospace;background:#f5f5f5;padding:0.3rem 0.6rem;border-radius:4px;display:inline-block"></div>"""
        stats_script = """
    function updateStats(idx) {
      const s = STATS[idx];
      document.getElementById('stats-bar').textContent =
        'N deposits: ' + s.n.toLocaleString() +
        '   mean dE: ' + s.mean.toPrecision(4) + ' MeV' +
        '   total dE: ' + s.total.toPrecision(4) + ' MeV';
    }
    updateStats(sel.selectedIndex);
    sel.addEventListener('change', () => { frame.src = sel.value; updateStats(sel.selectedIndex); });"""
    else:
        stats_js = ''
        stats_panel = ''
        stats_script = "    sel.addEventListener('change', () => { frame.src = sel.value; });"

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Energy deposits — track picker</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1rem; }}
    h1 {{ font-size: 1.25rem; font-weight: 600; }}
    label {{ margin-right: 0.5rem; }}
    select {{ min-width: min(40rem, 100%); max-width: 100%; font-size: 0.9rem; }}
    table td, table th {{ border: 1px solid #ddd; padding: 3px 7px; text-align: right; }}
    iframe {{
      width: 100%;
      height: calc(100vh - 8rem);
      border: 1px solid #ccc;
      border-radius: 4px;
      margin-top: 0.5rem;
    }}
  </style>
</head>
<body>
  <h1>3D energy deposits</h1>
  <p style="display:flex;flex-wrap:wrap;gap:1rem;font-size:0.9rem">
    {f'<a href="{html.escape(pdf_name, quote=True)}" target="_blank">📄 Wireplanes PDF</a>' if pdf_name else ''}
    <a href="dedx_distributions.pdf" target="_blank">📄 dE/dx histograms</a>
    <a href="bragg_peak_dedx_vs_pathlen.pdf" target="_blank">📄 Bragg peak (PDF)</a>
    <a href="bragg_peak_tail150mm.pdf" target="_blank">📄 Bragg peak tail 150 mm</a>
    <a href="bragg_peak_tail50mm.pdf" target="_blank">📄 Bragg peak tail 50 mm</a>
    <a href="bragg_peak_dedx_vs_pathlen.html" target="_blank">🌐 Bragg peak (interactive)</a>
    <a href="histograms.html" target="_blank">🌐 Track histograms</a>
    <a href="coordinate_distributions.pdf" target="_blank">📄 Coordinate distributions</a>
    <a href="track_catalog.pdf" target="_blank">📄 Track catalog</a>
  </p>
  {_track_kinematics_table(specs)}
  {_pixel_count_table(specs, pixel_counts, plane_names=plane_names) if pixel_counts else ''}
  {_drift_dist_table(specs, dist_stats) if dist_stats else ''}
  <p>
    <label for="track-select">Track</label>
    <select id="track-select" aria-label="Choose track">
{options_block}
    </select>
  </p>{stats_panel}
  <iframe id="plot-frame" title="Plotly 3D energy deposits" src="{html.escape(first, quote=True)}"></iframe>
  <script>
    {stats_js}
    const sel = document.getElementById('track-select');
    const frame = document.getElementById('plot-frame');
{stats_script}
  </script>
</body>
</html>
"""
    path = os.path.join(output_dir, 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page)
    print(f'  Saved {path}')


_STEP_SIZES_MM  = [1.0, 0.5, 0.1]
_STEP_COLORS    = ['tab:blue', 'tab:orange', 'tab:green']
_STEP_LABELS    = ['1 mm', '0.5 mm', '0.1 mm']


def _build_step_size_tracks(specs, tracks_01mm, start_positions_mm, cfg):
    """Return dict[(track_idx, step_size_mm) -> track] for all 3 step sizes.

    The 0.1 mm entries reuse ``tracks_01mm`` (already filtered); the 1 mm and
    0.5 mm entries are generated fresh and filtered inside volumes.
    """
    result = {}
    for i, (spec, start_mm) in enumerate(zip(specs, start_positions_mm)):
        for ss in _STEP_SIZES_MM:
            if ss == 0.1:
                result[(i, ss)] = tracks_01mm[i]
            else:
                track = generate_muon_track(
                    start_position_mm=start_mm,
                    direction=spec['direction'],
                    kinetic_energy_mev=spec['momentum_mev'],
                    step_size_mm=ss,
                    track_id=1,
                )
                result[(i, ss)] = filter_track_inside_volumes(track, cfg.volumes)
    return result


def _shared_bins(arrays, n_bins, plo=0.5, phi=99.5):
    all_vals = np.concatenate([a for a in arrays if len(a) > 0])
    return np.linspace(float(np.percentile(all_vals, plo)),
                       float(np.percentile(all_vals, phi)), n_bins + 1)


def write_dedx_distributions_pdf(specs, step_tracks, output_dir):
    """dE/dx histograms per track for step sizes 1, 0.5, 0.1 mm.

    Layout: 2 rows × N_tracks columns.
      Row 0 — linear scale.
      Row 1 — log scale (same bins).
    The 3 step sizes are overlaid in each panel.
    ``step_tracks`` is the dict returned by ``_build_step_size_tracks``.
    """
    n_tracks = max(i for i, _ in step_tracks) + 1

    # Build dE/dx arrays from pre-generated tracks.
    all_dedx = {}
    for i in range(n_tracks):
        for ss in _STEP_SIZES_MM:
            track = step_tracks[(i, ss)]
            de    = np.asarray(track['de'])
            all_dedx[(i, ss)] = de / (ss / 10.0) if len(de) > 0 else np.array([])

    bin_edges = _shared_bins(list(all_dedx.values()), n_bins=80)

    col_w = max(2.5, min(3.5, 45.0 / n_tracks))
    fig, axes = plt.subplots(
        2, n_tracks,
        figsize=(col_w * n_tracks, 5.5),
        constrained_layout=True,
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        'dE/dx distributions per track — step sizes 0.1 / 0.5 / 1 mm (shared bins)',
        fontsize=11,
    )

    for i in range(n_tracks):
        name = specs[i]['name']
        for row, use_log in enumerate([False, True]):
            ax = axes[row, i]
            for ss, color, lbl in zip(_STEP_SIZES_MM, _STEP_COLORS, _STEP_LABELS):
                dedx = all_dedx[(i, ss)]
                if len(dedx) == 0:
                    continue
                ax.hist(dedx, bins=bin_edges, histtype='step', color=color,
                        label=lbl, density=True, linewidth=1.0)
            if use_log:
                ax.set_yscale('log')
            ax.set_xlabel('dE/dx (MeV/cm)', fontsize=7)
            if i == 0:
                ax.set_ylabel('density' + (' (log)' if use_log else ''), fontsize=8)
            ax.tick_params(axis='both', labelsize=5)
            if row == 0:
                ax.set_title(name, fontsize=6)
            if i == 0 and row == 0:
                ax.legend(fontsize=5)

    path = os.path.join(output_dir, 'dedx_distributions.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def write_bragg_peak_pdf(specs, step_tracks, output_dir):
    """dE/dx vs cumulative path length (Bragg peak) per track, step sizes overlaid.

    Layout: 2 × ceil(N/3) rows × 3 columns — top half linear, bottom half log y-scale.
    """
    n_tracks = max(i for i, _ in step_tracks) + 1
    n_cols = 3
    n_data_rows = (n_tracks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        2 * n_data_rows, n_cols,
        figsize=(5.0 * n_cols, 3.2 * 2 * n_data_rows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(2 * n_data_rows, n_cols)
    fig.suptitle('dE/dx vs path length (Bragg peak) — step sizes 0.1 / 0.5 / 1 mm\n'
                 'top: linear  |  bottom: log y', fontsize=11)

    for i in range(n_tracks):
        data_row, col = divmod(i, n_cols)
        name = specs[i]['name']
        for use_log in (False, True):
            ax = axes[2 * data_row + int(use_log), col]
            for ss, color, lbl in zip(_STEP_SIZES_MM, _STEP_COLORS, _STEP_LABELS):
                track = step_tracks[(i, ss)]
                de = np.asarray(track['de'])
                dx = np.asarray(track['dx'])
                if len(de) == 0:
                    continue
                dedx = de / (dx / 10.0)
                path_mm = np.cumsum(dx)
                ax.plot(path_mm, dedx, color=color, label=lbl, linewidth=0.8, alpha=0.85)
            if use_log:
                ax.set_yscale('log')
            ax.set_xlabel('path length (mm)', fontsize=7)
            ax.set_ylabel('dE/dx (MeV/cm)' + (' (log)' if use_log else ''), fontsize=7)
            ax.set_title(name, fontsize=8)
            ax.tick_params(axis='both', labelsize=6)
            if i == 0 and not use_log:
                ax.legend(fontsize=6)

    for i in range(n_tracks, n_data_rows * n_cols):
        data_row, col = divmod(i, n_cols)
        for use_log in (False, True):
            axes[2 * data_row + int(use_log), col].set_visible(False)

    path = os.path.join(output_dir, 'bragg_peak_dedx_vs_pathlen.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def write_bragg_peak_tail_pdf(specs, step_tracks, output_dir, tail_mm=150.0):
    """Same as write_bragg_peak_pdf but zoomed to the last ``tail_mm`` of each track."""
    n_tracks = max(i for i, _ in step_tracks) + 1
    n_cols = 3
    n_data_rows = (n_tracks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        2 * n_data_rows, n_cols,
        figsize=(5.0 * n_cols, 3.2 * 2 * n_data_rows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(2 * n_data_rows, n_cols)
    fig.suptitle(f'dE/dx vs path length — last {tail_mm:.0f} mm (Bragg peak tail)\n'
                 'top: linear  |  bottom: log y', fontsize=11)

    for i in range(n_tracks):
        data_row, col = divmod(i, n_cols)
        name = specs[i]['name']
        for use_log in (False, True):
            ax = axes[2 * data_row + int(use_log), col]
            for ss, color, lbl in zip(_STEP_SIZES_MM, _STEP_COLORS, _STEP_LABELS):
                track = step_tracks[(i, ss)]
                de = np.asarray(track['de'])
                dx = np.asarray(track['dx'])
                if len(de) == 0:
                    continue
                dedx    = de / (dx / 10.0)
                path_mm = np.cumsum(dx)
                mask    = path_mm >= path_mm[-1] - tail_mm
                ax.plot(path_mm[mask], dedx[mask],
                        color=color, label=lbl, linewidth=0.8, alpha=0.85,
                        marker='.', markersize=3)
            if use_log:
                ax.set_yscale('log')
            ax.set_xlabel('path length (mm)', fontsize=7)
            ax.set_ylabel('dE/dx (MeV/cm)' + (' (log)' if use_log else ''), fontsize=7)
            ax.set_title(name, fontsize=8)
            ax.tick_params(axis='both', labelsize=6)
            if i == 0 and not use_log:
                ax.legend(fontsize=6)

    for i in range(n_tracks, n_data_rows * n_cols):
        data_row, col = divmod(i, n_cols)
        for use_log in (False, True):
            axes[2 * data_row + int(use_log), col].set_visible(False)

    path = os.path.join(output_dir, f'bragg_peak_tail{tail_mm:.0f}mm.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def write_bragg_peak_html(specs, step_tracks, output_dir):
    """Interactive Plotly: dE/dx vs path length for all tracks; toggle per track in legend.

    One trace per (track, step_size). 0.5 mm and 1 mm traces start hidden (legendonly)
    to keep the initial view readable. Click a legend group to show/hide all step sizes
    for that track; click an individual entry to toggle just that step size.
    """
    n_tracks = max(i for i, _ in step_tracks) + 1

    _PLOTLY_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    ]
    _STEP_DASH = {1.0: 'solid', 0.5: 'dash', 0.1: 'dot'}

    fig = go.Figure()
    for i in range(n_tracks):
        name = specs[i]['name']
        color = _PLOTLY_COLORS[i % len(_PLOTLY_COLORS)]
        first_in_group = True
        for ss in _STEP_SIZES_MM:
            track = step_tracks[(i, ss)]
            de = np.asarray(track['de'])
            dx = np.asarray(track['dx'])
            if len(de) == 0:
                continue
            dedx = de / (dx / 10.0)
            path_mm = np.cumsum(dx)
            extra_kwargs = {}
            if first_in_group:
                extra_kwargs['legendgrouptitle_text'] = name
                first_in_group = False
            fig.add_trace(go.Scatter(
                x=path_mm,
                y=dedx,
                mode='lines',
                name=f'{ss} mm',
                legendgroup=name,
                line=dict(color=color, dash=_STEP_DASH[ss], width=1.5),
                visible=True if ss == 0.1 else 'legendonly',
                hovertemplate=(
                    f'<b>{name}</b> ({ss} mm)<br>'
                    'path=%{x:.1f} mm<br>dE/dx=%{y:.3f} MeV/cm<extra></extra>'
                ),
                **extra_kwargs,
            ))

    fig.update_layout(
        title='dE/dx vs path length (Bragg peak) — click legend to toggle tracks / step sizes',
        xaxis_title='path length (mm)',
        yaxis_title='dE/dx (MeV/cm)',
        legend=dict(groupclick='togglegroup', tracegroupgap=4),
        hovermode='closest',
        margin=dict(l=60, r=20, t=60, b=60),
    )
    path = os.path.join(output_dir, 'bragg_peak_dedx_vs_pathlen.html')
    fig.write_html(path)
    print(f'  Saved {path}')


def write_coordinate_distributions_pdf(specs, step_tracks, output_dir):
    """Histograms of track geometry and raw energy deposits.

    Layout: 2 rows × 7 columns (linear top, log bottom).
      Cols 0–2  avg x / avg y / avg z — one value per track (15 points),
                3 step sizes overlaid, counts (not density).
      Cols 3–6  x / y / z / dE — all segment deposits pooled over all tracks,
                3 step sizes overlaid, density-normalised.
    ``step_tracks`` is the dict returned by ``_build_step_size_tracks``.
    """
    n_tracks = max(i for i, _ in step_tracks) + 1

    # Per-track mean positions: shape (n_tracks,) per step size.
    avg_pos = {ss: {c: np.full(n_tracks, np.nan) for c in 'xyz'}
               for ss in _STEP_SIZES_MM}
    for ss in _STEP_SIZES_MM:
        for i in range(n_tracks):
            pos = np.asarray(step_tracks[(i, ss)]['position'])
            if len(pos) > 0:
                for ci, c in enumerate('xyz'):
                    avg_pos[ss][c][i] = float(np.mean(pos[:, ci]))

    # Pooled segment coords and dE over all tracks.
    pooled = {ss: {k: [] for k in ('x', 'y', 'z', 'de')} for ss in _STEP_SIZES_MM}
    for ss in _STEP_SIZES_MM:
        for i in range(n_tracks):
            track = step_tracks[(i, ss)]
            pos   = np.asarray(track['position'])
            de    = np.asarray(track['de'])
            if len(pos) > 0:
                pooled[ss]['x'].append(pos[:, 0])
                pooled[ss]['y'].append(pos[:, 1])
                pooled[ss]['z'].append(pos[:, 2])
                pooled[ss]['de'].append(de)
        for k in ('x', 'y', 'z', 'de'):
            arrs = pooled[ss][k]
            pooled[ss][k] = np.concatenate(arrs) if arrs else np.array([])

    # Shared bins per quantity.
    avg_bins = {
        c: _shared_bins([avg_pos[ss][c] for ss in _STEP_SIZES_MM], n_bins=8)
        for c in 'xyz'
    }
    pool_bins = {
        k: _shared_bins([pooled[ss][k] for ss in _STEP_SIZES_MM], n_bins=60)
        for k in ('x', 'y', 'z', 'de')
    }

    col_specs = [
        ('avg',    'x',  'avg x (mm)',  False),
        ('avg',    'y',  'avg y (mm)',  False),
        ('avg',    'z',  'avg z (mm)',  False),
        ('pooled', 'x',  'x (mm)',      True),
        ('pooled', 'y',  'y (mm)',      True),
        ('pooled', 'z',  'z (mm)',      True),
        ('pooled', 'de', 'dE (MeV)',    True),
    ]

    fig, axes = plt.subplots(2, 7, figsize=(26, 5.5), constrained_layout=True)
    axes = np.asarray(axes)

    fig.suptitle(
        'Track geometry & energy deposit distributions — step sizes 0.1 / 0.5 / 1 mm  |  '
        'Cols 0–2: per-track mean position (15 pts each)   Cols 3–6: all segments pooled over 15 tracks',
        fontsize=9,
    )

    for col_idx, (kind, key, xlabel, use_density) in enumerate(col_specs):
        if kind == 'avg':
            bins_arr = avg_bins[key]
            data_fn  = lambda ss, k=key: avg_pos[ss][k][np.isfinite(avg_pos[ss][k])]
        else:
            bins_arr = pool_bins[key]
            data_fn  = lambda ss, k=key: pooled[ss][k]

        for row, use_log in enumerate([False, True]):
            ax = axes[row, col_idx]
            for ss, color, lbl in zip(_STEP_SIZES_MM, _STEP_COLORS, _STEP_LABELS):
                vals = data_fn(ss)
                if len(vals) == 0:
                    continue
                ax.hist(vals, bins=bins_arr, histtype='step', color=color,
                        label=lbl, density=use_density, linewidth=1.0)
            if use_log:
                ax.set_yscale('log')
            ax.set_xlabel(xlabel, fontsize=8)
            if col_idx == 0:
                ylabel = ('count' if not use_density else 'density')
                ax.set_ylabel(ylabel + (' (log)' if use_log else ''), fontsize=8)
            elif col_idx == 3:
                ax.set_ylabel('density' + (' (log)' if use_log else ''), fontsize=8)
            ax.tick_params(axis='both', labelsize=6)
            if row == 0:
                ax.set_title(xlabel, fontsize=9)
            if col_idx == 0 and row == 0:
                ax.legend(fontsize=6)

    path = os.path.join(output_dir, 'coordinate_distributions.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def write_track_catalog_pdf(specs, path, stats=None):
    """Single-page PDF: index, name, kinetic energy, unit direction components, optional stats."""
    n = len(specs)
    fig_h = min(22.0, max(4.0, 0.55 * n + 1.5))
    has_stats = stats is not None and len(stats) == len(specs)
    fig, ax = plt.subplots(figsize=(18 if has_stats else 11, fig_h))
    ax.axis('off')
    headers = ['#', 'name', 'T (MeV)', '(dx, dy, dz)']
    if has_stats:
        headers += ['N deposits', 'mean dE (MeV)', 'total dE (MeV)', 'n_time', 'n_wires (U1,V1,Y1,U2,V2,Y2)']
    rows = []
    for i, s in enumerate(specs, start=1):
        d = s['direction']
        dir_str = f'({d[0]:.6g}, {d[1]:.6g}, {d[2]:.6g})'
        row = [str(i), s['name'], f'{s["momentum_mev"]:.6g}', dir_str]
        if has_stats:
            st = stats[i - 1]
            n_wires_str = ','.join(str(w) for w in st.get('n_wires', []))
            row += [f'{st["n_deposits"]:,}',
                    f'{st["mean_de"]:.4g}',
                    f'{st["total_de"]:.4g}',
                    str(st.get('n_time', '')),
                    n_wires_str]
        rows.append(row)
    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='upper center',
        cellLoc='left',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.05, 1.65)
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor('#e0e0e0')
        tbl[(0, j)].set_text_props(weight='bold')
    ax.set_title('Tracks — name, kinetic energy, direction' +
                 (' + deposit stats + readout shape' if has_stats else ''), fontsize=12, pad=6)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def write_histograms_html(specs, dist_stats, output_dir):
    """Self-contained interactive HTML: pick muons, see histograms of E / theta / phi / drift distances."""
    _UGLY_TRACKS = ['Muon4_100MeV', 'Muon5_100MeV', 'Muon10_100MeV',
                    'Muon12_100MeV', 'Muon13_100MeV']

    track_data = []
    for i, spec in enumerate(specs):
        d = np.array(spec['direction'], dtype=float)
        norm = np.linalg.norm(d)
        if norm > 0:
            d = d / norm
        theta_deg = float(np.degrees(np.arccos(np.clip(d[2], -1.0, 1.0))))
        phi_deg = float(np.degrees(np.arctan2(d[1], d[0])))
        ds = (dist_stats[i]
              if dist_stats and i < len(dist_stats) and dist_stats[i] is not None
              else None)
        track_data.append({
            'name': spec['name'],
            'E': float(spec['momentum_mev']),
            'theta': theta_deg,
            'phi': phi_deg,
            'mean_east': float(ds['east']['mean']) if ds else None,
            'mean_west': float(ds['west']['mean']) if ds else None,
        })

    js_data = 'const TRACKS = ' + json.dumps(track_data) + ';'
    ugly_js = json.dumps(_UGLY_TRACKS)

    checkbox_items_html = ''.join(
        f'<label class="muon-item">'
        f'<input type="checkbox" class="muon-cb" value="{i}" checked> '
        f'{html.escape(td["name"])}</label>\n'
        for i, td in enumerate(track_data)
    )

    _TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Track histograms</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { display: flex; height: 100vh; font-family: system-ui, sans-serif; overflow: hidden; }
    #sidebar {
      width: 240px; min-width: 140px; border-right: 1px solid #ddd;
      padding: 0.75rem; overflow-y: auto; flex-shrink: 0; background: #fafafa;
    }
    #sidebar h2 { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.4rem; }
    .btn-row { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 0.5rem; }
    .btn-row button { font-size: 0.72rem; padding: 2px 7px; cursor: pointer;
                      border: 1px solid #bbb; border-radius: 3px; background: #fff; }
    .btn-row button:hover { background: #e8e8e8; }
    .muon-item { display: block; font-size: 0.76rem; padding: 2px 0;
                 cursor: pointer; white-space: nowrap; overflow: hidden;
                 text-overflow: ellipsis; }
    #main { flex: 1; padding: 0.85rem; overflow-y: auto; }
    #main h1 { font-size: 1.05rem; margin-bottom: 0.75rem; font-weight: 600; }
    .hist-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
      gap: 0.85rem;
    }
    .hist-card {
      background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
      padding: 0.6rem 0.8rem;
    }
    .hist-card h3 { font-size: 0.85rem; margin-bottom: 0.3rem; color: #333; }
    svg { display: block; overflow: visible; }
    rect.bar { fill: #4682B4; opacity: 0.72; transition: opacity 0.1s; }
    rect.bar:hover { opacity: 1; }
  </style>
</head>
<body>
  <div id="sidebar">
    <h2>Muons</h2>
    <div class="btn-row">
      <button onclick="selectPreset('all')">All</button>
      <button onclick="selectPreset('none')">None</button>
      <button onclick="selectPreset('orig')">Orig</button>
      <button onclick="selectPreset('nice')">Nice</button>
    </div>
    __CHECKBOXES__
  </div>
  <div id="main">
    <h1>Track histograms</h1>
    <div class="hist-grid" id="hist-grid"></div>
  </div>
  <script>
  __JS_DATA__
  const UGLY_TRACKS = __UGLY_JS__;
  const QUANTITIES = [
    {key: 'E',         label: 'T (MeV)'},
    {key: 'theta',     label: 'θ (°)'},
    {key: 'phi',       label: 'φ (°)'},
    {key: 'mean_east', label: 'Mean dist east (mm)'},
    {key: 'mean_west', label: 'Mean dist west (mm)'},
  ];
  const N_BINS = 8;
  const SVG_W = 290, SVG_H = 170;
  const PAD = {top: 8, right: 10, bottom: 38, left: 36};

  function computeHist(vals) {
    const finite = vals.filter(v => v !== null && isFinite(v));
    if (finite.length === 0) return null;
    let lo = Math.min(...finite), hi = Math.max(...finite);
    if (lo === hi) { lo -= 0.5; hi += 0.5; }
    const w = (hi - lo) / N_BINS;
    const counts = Array(N_BINS).fill(0);
    const bnames = Array.from({length: N_BINS}, () => []);
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i];
      if (v === null || !isFinite(v)) continue;
      let b = Math.floor((v - lo) / w);
      if (b >= N_BINS) b = N_BINS - 1;
      counts[b]++;
      bnames[b].push(i);
    }
    return {lo, hi, w, counts, bnames};
  }

  function fmtVal(v) {
    if (v === null || !isFinite(v)) return '—';
    return Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2);
  }

  function buildSVG(hist, selNames) {
    const pw = SVG_W - PAD.left - PAD.right;
    const ph = SVG_H - PAD.top - PAD.bottom;
    const nodata = `<text x="${SVG_W / 2}" y="${SVG_H / 2}" text-anchor="middle" font-size="12" fill="#999">no data</text>`;
    if (!hist) return `<svg width="${SVG_W}" height="${SVG_H}">${nodata}</svg>`;
    const maxCnt = Math.max(...hist.counts, 1);
    let inner = '';
    const bw = pw / N_BINS;
    for (let b = 0; b < N_BINS; b++) {
      const x = PAD.left + b * bw;
      const cnt = hist.counts[b];
      const bh = Math.max((cnt / maxCnt) * ph, cnt > 0 ? 1 : 0);
      const y = PAD.top + ph - bh;
      const tip = hist.bnames[b].map(i => selNames[i]).join('\n');
      inner += `<rect class="bar" x="${(x+1).toFixed(1)}" y="${y.toFixed(1)}" width="${(bw-2).toFixed(1)}" height="${bh.toFixed(1)}"><title>${cnt} track(s):\n${tip}</title></rect>`;
      if (cnt > 0) {
        inner += `<text x="${(x+bw/2).toFixed(1)}" y="${(y-2).toFixed(1)}" text-anchor="middle" font-size="10" fill="#333">${cnt}</text>`;
      }
    }
    inner += `<line x1="${PAD.left}" y1="${PAD.top+ph}" x2="${PAD.left+pw}" y2="${PAD.top+ph}" stroke="#888" stroke-width="1"/>`;
    for (let t = 0; t <= 4; t++) {
      const frac = t / 4;
      const xp = PAD.left + frac * pw;
      const val = hist.lo + frac * (hist.hi - hist.lo);
      inner += `<line x1="${xp.toFixed(1)}" y1="${(PAD.top+ph).toFixed(1)}" x2="${xp.toFixed(1)}" y2="${(PAD.top+ph+4).toFixed(1)}" stroke="#888" stroke-width="1"/>`;
      inner += `<text x="${xp.toFixed(1)}" y="${(PAD.top+ph+14).toFixed(1)}" text-anchor="middle" font-size="9" fill="#555">${fmtVal(val)}</text>`;
    }
    inner += `<line x1="${PAD.left}" y1="${PAD.top}" x2="${PAD.left}" y2="${PAD.top+ph}" stroke="#888" stroke-width="1"/>`;
    for (let t = 0; t <= 4; t++) {
      const frac = t / 4;
      const yp = PAD.top + ph - frac * ph;
      const cnt = Math.round(frac * maxCnt);
      inner += `<line x1="${(PAD.left-3).toFixed(1)}" y1="${yp.toFixed(1)}" x2="${PAD.left}" y2="${yp.toFixed(1)}" stroke="#888" stroke-width="1"/>`;
      inner += `<text x="${(PAD.left-5).toFixed(1)}" y="${(yp+3).toFixed(1)}" text-anchor="end" font-size="9" fill="#555">${cnt}</text>`;
    }
    return `<svg width="${SVG_W}" height="${SVG_H}">${inner}</svg>`;
  }

  function render() {
    const sel = Array.from(document.querySelectorAll('.muon-cb:checked')).map(cb => +cb.value);
    const selTracks = sel.map(i => TRACKS[i]);
    const selNames = selTracks.map(t => t.name);
    const grid = document.getElementById('hist-grid');
    grid.innerHTML = '';
    for (const q of QUANTITIES) {
      const vals = selTracks.map(t => t[q.key]);
      const hist = computeHist(vals);
      const card = document.createElement('div');
      card.className = 'hist-card';
      card.innerHTML = `<h3>${q.label}</h3>` + buildSVG(hist, selNames);
      grid.appendChild(card);
    }
  }

  function selectPreset(preset) {
    document.querySelectorAll('.muon-cb').forEach(cb => {
      const name = TRACKS[+cb.value].name;
      if      (preset === 'all')  cb.checked = true;
      else if (preset === 'none') cb.checked = false;
      else if (preset === 'orig') cb.checked = !name.includes('Quarter');
      else if (preset === 'nice') cb.checked = !UGLY_TRACKS.some(u => name.includes(u));
    });
    render();
  }

  document.querySelectorAll('.muon-cb').forEach(cb => cb.addEventListener('change', render));
  render();
  </script>
</body>
</html>"""

    page = (_TMPL
            .replace('__JS_DATA__', js_data)
            .replace('__UGLY_JS__', ugly_js)
            .replace('__CHECKBOXES__', checkbox_items_html))
    path = os.path.join(output_dir, 'histograms.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page)
    print(f'  Saved {path}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', default=os.path.join(_PLOTS_DIR, '20260605', 'mixed_tracks_edep'),
                   help='Directory for HTML files and the combined PDF')
    p.add_argument('--tracks', default=None,
                   help="If set: '+'-separated name:dx,dy,dz:T specs. "
                        f'If omitted: {N_DEFAULT_BOUNDARY_MUONS} random boundary muons '
                        f'plus three fixed 1000 MeV chords through East+West (random part '
                        f'T ∈ {{100,500,1000}} MeV, --seed).')
    p.add_argument('--seed', type=int, default=42,
                   help='RNG seed for default random muons (default: 42)')
    p.add_argument('--signal-percentile', type=float, default=99.0,
                   help='Percentile of |signal| for symmetric 2-D colour limits (default: 99)')
    p.add_argument('--start-position-mm', type=float, nargs=3, default=None,
                   metavar=('X', 'Y', 'Z'),
                   help='If set, use this vertex (mm) for every track. '
                        'Default: per-track random face vertex (default mode) or East/West center (``--tracks``).')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'JAX devices : {jax.devices()}')
    print('Building simulator...')
    simulator = build_simulator()
    cfg = simulator.config
    n_p = cfg.volumes[0].n_planes
    if cfg.n_volumes * n_p != 6:
        raise RuntimeError(
            f'Expected 2 volumes × 3 wire planes = 6 readouts; got n_volumes={cfg.n_volumes}, '
            f'n_planes[0]={n_p}')
    time_step_us = float(cfg.time_step_us)

    col_labels = []
    for v in range(cfg.n_volumes):
        for p in range(n_p):
            col_labels.append(f'{cfg.plane_names[v][p]}{v + 1}')

    print('Warming up JIT...')
    t0 = time.time()
    simulator.warm_up()
    print(f'Done ({time.time() - t0:.1f} s)')

    if args.tracks is None:
        specs = generate_random_boundary_tracks(
            cfg.volumes, n=N_DEFAULT_BOUNDARY_MUONS, seed=args.seed)
        print(f'Default mode: {len(specs)} tracks ({N_DEFAULT_BOUNDARY_MUONS} random + '
              f'3 fixed East–West chords, seed={args.seed})')
    else:
        specs = parse_mixed_tracks(args.tracks)

    catalog_path = os.path.join(args.output_dir, 'track_catalog.pdf')

    east_mm, west_mm = outer_boundary_starts_mm(cfg.volumes)
    if args.start_position_mm is not None:
        print(f'Fixed start for all tracks (mm): {tuple(args.start_position_mm)}')
    elif args.tracks is None:
        print('Per-track starts: random (y,z) on East or West outer x face')
    else:
        print(f'Legacy ``--tracks`` face centers (mm): East outer {east_mm}, West outer {west_mm}')

    all_de = []
    all_planes = []
    tracks_raw = []
    track_stats = []
    start_positions_used = []
    for spec in specs:
        if args.start_position_mm is not None:
            start_mm = tuple(args.start_position_mm)
        elif spec.get('start_position_mm') is not None:
            start_mm = tuple(spec['start_position_mm'])
        else:
            start_mm = start_mm_for_track(spec, east_mm, west_mm)
        start_positions_used.append(start_mm)
        print(
            f"  Forward: {spec['name']}  start={start_mm}  "
            f"dir={spec['direction']}  T={spec['momentum_mev']:.0f} MeV"
        )
        t0 = time.time()
        track, planes = track_and_forward(simulator, spec, start_mm)
        print(f'    ({time.time() - t0:.1f} s)')
        tracks_raw.append(track)
        de = np.asarray(track['de'])
        n_dep = len(de)
        mean_de = float(np.mean(de)) if n_dep > 0 else 0.0
        total_de = float(np.sum(de)) if n_dep > 0 else 0.0
        n_time = planes[0].shape[1] if planes else 0
        n_wires_list = [p.shape[0] for p in planes]
        print(f'    N_deposits={n_dep:,}  mean_dE={mean_de:.4g} MeV  total_dE={total_de:.4g} MeV'
              f'  n_time={n_time}  n_wires={n_wires_list}')
        track_stats.append(dict(n_deposits=n_dep, mean_de=mean_de, total_de=total_de,
                                n_time=n_time, n_wires=n_wires_list))
        all_de.append(de)
        all_planes.append(planes)

    write_track_catalog_pdf(specs, catalog_path, stats=track_stats)

    print('Building step-size track variants (1 mm, 0.5 mm) for distribution plots...')
    step_tracks = _build_step_size_tracks(
        specs, tracks_raw, start_positions_used, cfg)
    write_dedx_distributions_pdf(specs, step_tracks, args.output_dir)
    write_bragg_peak_pdf(specs, step_tracks, args.output_dir)
    write_bragg_peak_tail_pdf(specs, step_tracks, args.output_dir, tail_mm=150.0)
    write_bragg_peak_tail_pdf(specs, step_tracks, args.output_dir, tail_mm=50.0)
    write_bragg_peak_html(specs, step_tracks, args.output_dir)
    write_coordinate_distributions_pdf(specs, step_tracks, args.output_dir)

    de_all = np.concatenate(all_de) if all_de else np.array([0.0])
    de_min, de_max = float(np.min(de_all)), float(np.max(de_all))
    de_range = de_all.max() - de_all.min() if len(de_all) and de_all.max() > de_all.min() else 1.0

    abs_stack = np.concatenate(
        [np.abs(p).ravel() for planes in all_planes for p in planes])
    vmax_sig = float(np.nanpercentile(abs_stack, args.signal_percentile))
    vmax_sig = max(vmax_sig, 1e-9)
    norm_2d = mcolors.Normalize(vmin=-vmax_sig, vmax=vmax_sig)

    n_row = len(specs)
    pdf_path = os.path.join(args.output_dir, f'wireplanes_{n_row}x6_gt_signals.pdf')
    fig_h = max(8.0, 3.0 * n_row)
    fig_pdf, axes = plt.subplots(n_row, 6, figsize=(18, fig_h), constrained_layout=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    fig_pdf.suptitle(
        f'GT wire-plane signals (shared scale ±{vmax_sig:.3g} e⁻, p{args.signal_percentile:g} |all|)\n'
        f'dE 3-D HTML colour range [{de_min:.5g}, {de_max:.5g}] MeV',
        fontsize=11,
    )

    for i, spec in enumerate(specs):
        stem = _safe_stem(spec['name'])
        html_path = os.path.join(args.output_dir, f'edep_3d_{stem}.html')
        write_edep_3d_html(
            tracks_raw[i], spec, html_path, de_min, de_max, de_range, cfg.volumes,
            stats=track_stats[i])
        print(f'  Saved {html_path}')

        planes = all_planes[i]
        for j in range(6):
            ax = axes[i, j]
            data = planes[j]
            n_wires, n_time = data.shape
            t_max_us = n_time * time_step_us
            im = ax.imshow(
                data,
                aspect='auto',
                origin='lower',
                norm=norm_2d,
                cmap='RdBu_r',
                extent=[0, t_max_us, 0, n_wires],
            )
            if i == 0:
                ax.set_title(col_labels[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(
                    f"{spec['name']}\nwire",
                    fontsize=7,
                    rotation=90,
                    va='center',
                )
            else:
                ax.set_ylabel('wire', fontsize=7)
            ax.set_xlabel('t (μs)', fontsize=7)
            ax.tick_params(axis='both', labelsize=6)

    _ADC_CUTOFFS = [0, 1, 2, 5, 10, 15, 20, 25, 30, 50]
    pixel_counts = []
    for planes in all_planes:
        plane_totals = [p.size for p in planes]
        by_plane_cutoff = [{c: int(np.sum(np.abs(p) >= c)) for c in _ADC_CUTOFFS} for p in planes]
        by_cut = {c: sum(bpc[c] for bpc in by_plane_cutoff) for c in _ADC_CUTOFFS}
        pixel_counts.append({
            'total': sum(plane_totals),
            'by_cutoff': by_cut,
            'cutoffs': _ADC_CUTOFFS,
            'plane_totals': plane_totals,
            'by_plane_cutoff': by_plane_cutoff,
        })

    dist_stats = _compute_drift_dist_stats(tracks_raw)
    write_histograms_html(specs, dist_stats, args.output_dir)
    write_edep_index_html(specs, args.output_dir, stats=track_stats,
                          pdf_name=os.path.basename(pdf_path),
                          pixel_counts=pixel_counts,
                          plane_names=col_labels,
                          dist_stats=dist_stats)

    cbar = fig_pdf.colorbar(
        im, ax=axes.ravel().tolist(), shrink=0.35, aspect=60, pad=0.02, label='signal (e⁻)')
    fig_pdf.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig_pdf)
    print(f'\nSaved {pdf_path}')
    print('Done.')


if __name__ == '__main__':
    main()
