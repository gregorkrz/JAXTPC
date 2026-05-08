#!/usr/bin/env python
"""
3-D energy-deposit (dE) visualizations and 2-D wire-plane GT signals.

**Default:** 12 random muons ``Muon1_{T}MeV``, … (``T`` ∈ {100, 500, 1000} MeV)
plus one fixed **Muon_diagCross_1000MeV** (1000 MeV: start (2000,2000,2000) mm,
direction toward (-2000,-2000,-2000) mm). Random tracks are reproducible via
``--seed``. Override with ``--tracks`` (``+``-separated ``name:dx,dy,dz:T`` specs)
or ``--start-position-mm`` for one vertex shared by all tracks.

Outputs
-------
  • One Plotly HTML per track: semi-transparent active-volume boxes from config
    plus 3-D scatter of segment positions coloured by dE (shared dE colour scale
    across all HTML files).
  • One PDF: N rows × 6 columns (default N=13) — rows are tracks, columns are
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
    """Parse '+'-separated items: ``name:dx,dy,dz:T`` or preset ``name`` only."""
    specs = []
    for item in tracks_str.split('+'):
        item = item.strip()
        if not item:
            continue
        if ':' in item:
            parts = item.split(':')
            if len(parts) != 3:
                raise ValueError(f'Expected name:dx,dy,dz:momentum_mev, got {item!r}')
            name = parts[0].strip()
            direction = tuple(float(x) for x in parts[1].split(','))
            momentum_mev = float(parts[2])
            if len(direction) != 3:
                raise ValueError(f'Direction must have 3 components in {item!r}')
        else:
            if item not in _TRACK_NAME_ONLY:
                raise ValueError(
                    f'Unknown bare track name {item!r}. Known: {list(_TRACK_NAME_ONLY)}')
            direction, momentum_mev = _TRACK_NAME_ONLY[item]
            name = item
        specs.append(dict(name=name, direction=direction, momentum_mev=momentum_mev))
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


def write_edep_3d_html(track, spec, path, de_min, de_max, de_range, volumes):
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
    title = (
        f"{spec['name']} — no deposits" if len(de) == 0 else
        f"Energy deposits — {spec['name']}  dir={spec['direction']}  "
        f"T={spec['momentum_mev']:.0f} MeV"
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x (mm)',
            yaxis_title='y (mm)',
            zaxis_title='z (mm)',
            aspectmode='data',
        ),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
        margin=dict(l=0, r=0, b=0, t=50),
    )
    fig.write_html(path)


def write_edep_index_html(specs, output_dir):
    """Single-page picker: iframe loads ``edep_3d_<stem>.html`` for the selected track."""
    if not specs:
        return
    option_lines = []
    for i, spec in enumerate(specs):
        basename = f'edep_3d_{_safe_stem(spec["name"])}.html'
        d = spec['direction']
        dir_str = f'({d[0]:.6g}, {d[1]:.6g}, {d[2]:.6g})'
        label = (
            f'{spec["name"]} — T={spec["momentum_mev"]:.6g} MeV — dir={dir_str}'
        )
        sel = ' selected' if i == 0 else ''
        option_lines.append(
            f'        <option value="{html.escape(basename, quote=True)}"{sel}>'
            f'{html.escape(label)}</option>'
        )
    first = f'edep_3d_{_safe_stem(specs[0]["name"])}.html'
    options_block = '\n'.join(option_lines)
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
    iframe {{
      width: 100%;
      height: calc(100vh - 6rem);
      border: 1px solid #ccc;
      border-radius: 4px;
      margin-top: 1rem;
    }}
  </style>
</head>
<body>
  <h1>3D energy deposits</h1>
  <p>
    <label for="track-select">Track</label>
    <select id="track-select" aria-label="Choose track">
{options_block}
    </select>
  </p>
  <iframe id="plot-frame" title="Plotly 3D energy deposits" src="{html.escape(first, quote=True)}"></iframe>
  <script>
    const sel = document.getElementById('track-select');
    const frame = document.getElementById('plot-frame');
    sel.addEventListener('change', () => {{ frame.src = sel.value; }});
  </script>
</body>
</html>
"""
    path = os.path.join(output_dir, 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page)
    print(f'  Saved {path}')


def write_track_catalog_pdf(specs, path):
    """Single-page PDF: index, name, kinetic energy, unit direction components."""
    n = len(specs)
    fig_h = min(22.0, max(4.0, 0.55 * n + 1.5))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis('off')
    headers = ['#', 'name', 'T (MeV)', '(dx, dy, dz)']
    rows = []
    for i, s in enumerate(specs, start=1):
        d = s['direction']
        dir_str = f'({d[0]:.6g}, {d[1]:.6g}, {d[2]:.6g})'
        rows.append([str(i), s['name'], f'{s["momentum_mev"]:.6g}', dir_str])
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
    ax.set_title('Tracks — name, kinetic energy, direction', fontsize=12, pad=6)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', default=os.path.join(_PLOTS_DIR, '20260605', 'mixed_tracks_edep'),
                   help='Directory for HTML files and the combined PDF')
    p.add_argument('--tracks', default=None,
                   help="If set: '+'-separated name:dx,dy,dz:T specs. "
                        f'If omitted: {N_DEFAULT_BOUNDARY_MUONS} random boundary muons '
                        f'plus one diagonal cross at 1000 MeV (100/500/1000 MeV random part, --seed).')
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
        print(f'Default mode: {len(specs)} tracks ({N_DEFAULT_BOUNDARY_MUONS} random + diagonal cross, seed={args.seed})')
    else:
        specs = parse_mixed_tracks(args.tracks)

    catalog_path = os.path.join(args.output_dir, 'track_catalog.pdf')
    write_track_catalog_pdf(specs, catalog_path)

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
    for spec in specs:
        if args.start_position_mm is not None:
            start_mm = tuple(args.start_position_mm)
        elif spec.get('start_position_mm') is not None:
            start_mm = tuple(spec['start_position_mm'])
        else:
            start_mm = start_mm_for_track(spec, east_mm, west_mm)
        print(
            f"  Forward: {spec['name']}  start={start_mm}  "
            f"dir={spec['direction']}  T={spec['momentum_mev']:.0f} MeV"
        )
        t0 = time.time()
        track, planes = track_and_forward(simulator, spec, start_mm)
        print(f'    ({time.time() - t0:.1f} s)')
        tracks_raw.append(track)
        de = np.asarray(track['de'])
        all_de.append(de)
        all_planes.append(planes)

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
            tracks_raw[i], spec, html_path, de_min, de_max, de_range, cfg.volumes)
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

    write_edep_index_html(specs, args.output_dir)

    cbar = fig_pdf.colorbar(
        im, ax=axes.ravel().tolist(), shrink=0.35, aspect=60, pad=0.02, label='signal (e⁻)')
    fig_pdf.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig_pdf)
    print(f'\nSaved {pdf_path}')
    print('Done.')


if __name__ == '__main__':
    main()
