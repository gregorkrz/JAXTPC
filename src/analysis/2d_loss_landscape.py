#!/usr/bin/env python
"""
Compute and plot the loss landscape over a 2-D grid of two SimParams scalars.

Default axes are (recomb_alpha, recomb_beta_90).  Use --param-y and --param-x to
sweep any distinct pair accepted by 2d_opt.py (e.g. velocity_cm_us + lifetime_us).

Grid limits: --range-frac around each GT value, or --range-y / --range-x MIN MAX,
or (legacy) --alpha-range / --beta-range when the corresponding default recomb
parameter is on that axis.

Optionally overlays optimization trajectories from 2d_opt.py pkl files
(recomb_alpha + recomb_beta_90 only).

Usage
-----
    python 2d_loss_landscape.py
    python 2d_loss_landscape.py --grid 20 --range-frac 0.2
    python 2d_loss_landscape.py --param-y velocity_cm_us --param-x lifetime_us
    python 2d_loss_landscape.py --loss sobolev_loss --overlay results/2d_opt
    python 2d_loss_landscape.py --load-pkl results/2d_landscape/landscape_sobolev_loss_diagonal_10x10.pkl
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

import argparse
import importlib.util
import os
import pickle
import time

# 2d_opt.py is not a valid module name; load by path for param helpers.
_2d_opt_path = os.path.join(os.path.dirname(__file__), '..', 'opt', '2d_opt.py')
_spec = importlib.util.spec_from_file_location('jaxtpc_2d_opt', _2d_opt_path)
_2d_opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_2d_opt)
_get_gt_val = _2d_opt._get_gt_val
_apply_param = _2d_opt._apply_param
VALID_PARAMS = _2d_opt.VALID_PARAMS

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tools.geometry import generate_detector
from tools.random_boundary_tracks import filter_track_inside_volumes
from tools.losses import make_sobolev_weight, sobolev_loss, sobolev_loss_geomean_log1p, mse_loss, l1_loss
from tools.noise import generate_noise
from tools.particle_generator import generate_muon_track
from tools.simulation import DetectorSimulator

# ── Constants (shared with 2d_opt.py) ─────────────────────────────────────────
GT_LIFETIME_US    = 10_000.0
GT_VELOCITY_CM_US = 0.160
SOBOLEV_MAX_PAD   = 128

CONFIG_PATH        = 'config/cubic_wireplane_config.yaml'
N_SEGMENTS         = 50_000
MAX_ACTIVE_BUCKETS = 1000
#DETECTOR_BOUNDS_MM = ((-300, 300), (-300, 300), (-300, 300))

_JAX_CACHE_DIR = os.environ.get('JAX_COMPILATION_CACHE_DIR',
                                os.path.expanduser('~/.cache/jax_compilation_cache'))
jax.config.update('jax_compilation_cache_dir', _JAX_CACHE_DIR)
_RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')
_PLOTS_DIR   = os.environ.get('PLOTS_DIR',   'plots')
jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)

VALID_LOSSES = ('sobolev_loss', 'sobolev_loss_geomean_log1p', 'mse_loss', 'l1_loss')

LOSS_LABELS = {
    'sobolev_loss':               'Sobolev',
    'sobolev_loss_geomean_log1p': 'Sobolev geomean log1p',
    'mse_loss':                   'MSE',
    'l1_loss':                    'L1',
}

PARAM_LABELS = {
    'velocity_cm_us':         'drift velocity  (cm/μs)',
    'lifetime_us':            'electron lifetime  (μs)',
    'diffusion_trans_cm2_us': 'transverse diffusion  (cm²/μs)',
    'diffusion_long_cm2_us':  'longitudinal diffusion  (cm²/μs)',
    'recomb_alpha':           'recombination α',
    'recomb_beta':            'recombination β',
    'recomb_beta_90':         'recombination β₉₀',
    'recomb_R':               'recombination R',
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--grid', type=int, default=10,
                   help='Grid resolution N: evaluates N×N points (default: 10)')
    p.add_argument('--range-frac', type=float, default=0.15,
                   help='Fractional range around GT on each axis (default: 0.15 → ±15%%)')
    p.add_argument('--param-y', default='recomb_alpha',
                   help='Y-axis (row) parameter name (default: recomb_alpha)')
    p.add_argument('--param-x', default='recomb_beta_90',
                   help='X-axis (column) parameter name (default: recomb_beta_90)')
    p.add_argument('--range-y', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                   help='Explicit Y-axis limits (overrides --range-frac for param-y)')
    p.add_argument('--range-x', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                   help='Explicit X-axis limits (overrides --range-frac for param-x)')
    p.add_argument('--alpha-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   help='Explicit alpha limits when --param-y is recomb_alpha (legacy)')
    p.add_argument('--beta-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   help='Explicit beta_90 limits when --param-x is recomb_beta_90 (legacy)')
    p.add_argument('--loss', default='sobolev_loss,sobolev_loss_geomean_log1p',
                   help='Comma-separated loss(es) to evaluate (default: both Sobolev)')
    p.add_argument('--track-name', default='diagonal',
                   help='Label for track direction (default: diagonal)')
    p.add_argument('--direction', default='1,1,1',
                   help='Muon direction as x,y,z (default: 1,1,1)')
    p.add_argument('--momentum', type=float, default=1000.0,
                   help='Muon kinetic energy in MeV (default: 1000.0)')
    p.add_argument('--results-dir', default=os.path.join(_RESULTS_DIR, '2d_landscape'),
                   help='Output directory for pkl and plots (default: results/2d_landscape)')
    p.add_argument('--overlay', default=None,
                   help='Path to a 2d_opt results dir; overlays matching trajectories on plot')
    p.add_argument('--load-pkl', default=None,
                   help='Skip computation and load a previously saved landscape pkl')
    p.add_argument('--gradients', action='store_true',
                   help='Also compute dL/dα and dL/dβ₉₀ at each grid point and store in pkl')
    p.add_argument('--noise-scale', type=float, default=0.0,
                   help='Noise amplitude as a multiple of the calibrated detector noise '
                        '(MicroBooNE model, converted to signal units via electrons_per_adc). '
                        '0.0 = no noise (default), 1.0 = realistic noise.')
    p.add_argument('--noise-seed', type=int, default=0,
                   help='Seed for the noise draw (default: 0)')
    p.add_argument('--start-position-mm', type=float, nargs=3, default=None,
                   metavar=('X', 'Y', 'Z'),
                   help='Muon vertex in mm (default: 0 0 0). Use with boundary tracks from launch_2d_landscape_pairs.')
    p.add_argument('--output-pkl', default=None,
                   help='Explicit path for the landscape pickle (parent dirs created); '
                        'overrides default filename in --results-dir')
    p.add_argument('--no-plots', action='store_true',
                   help='Skip figure exports (matplotlib + optional Plotly); write pkl only')
    return p.parse_args()


# ── Loss evaluation ────────────────────────────────────────────────────────────

def make_loss_fn(loss_name, simulator, deposits, gt_arrays, weights, gt_params,
                 param_y, param_x):
    def fwd(vy, vx):
        p = _apply_param(param_y, vy, gt_params)
        p = _apply_param(param_x, vx, p)
        return simulator.forward(p, deposits)

    if loss_name == 'sobolev_loss':
        def fn(vy, vx):
            return sobolev_loss(fwd(vy, vx), gt_arrays, weights)
    elif loss_name == 'sobolev_loss_geomean_log1p':
        def fn(vy, vx):
            return sobolev_loss_geomean_log1p(fwd(vy, vx), gt_arrays, weights)
    elif loss_name == 'mse_loss':
        def fn(vy, vx):
            return mse_loss(fwd(vy, vx), gt_arrays)
    elif loss_name == 'l1_loss':
        def fn(vy, vx):
            return l1_loss(fwd(vy, vx), gt_arrays)
    else:
        raise ValueError(f'Unknown loss {loss_name!r}')

    return jax.jit(fn)


def evaluate_grid(loss_fn, vals_y, vals_x, param_y, param_x, compute_gradients=False):
    """Evaluate loss_fn on all (param_y, param_x) grid points.

    Returns grid (N, M) always.  With compute_gradients=True also returns
    ∂L/∂param_y and ∂L/∂param_x as (N, M) arrays.
    """
    N, M = len(vals_y), len(vals_x)
    grid = np.zeros((N, M), dtype=np.float32)
    grad_y = np.zeros((N, M), dtype=np.float32) if compute_gradients else None
    grad_x = np.zeros((N, M), dtype=np.float32) if compute_gradients else None
    total = N * M

    if compute_gradients:
        val_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))

    for i, vy in enumerate(vals_y):
        for j, vx in enumerate(vals_x):
            vy_j = jnp.array(vy, dtype=jnp.float32)
            vx_j = jnp.array(vx, dtype=jnp.float32)
            if compute_gradients:
                (lv, (gy, gx)) = val_and_grad_fn(vy_j, vx_j)
                jax.block_until_ready((lv, gy, gx))
                grad_y[i, j] = float(gy)
                grad_x[i, j] = float(gx)
            else:
                lv = loss_fn(vy_j, vx_j)
                jax.block_until_ready(lv)
            grid[i, j] = float(lv)
            done = i * M + j + 1
            grad_str = (f'  d{param_y}={float(gy):.3e}  d{param_x}={float(gx):.3e}'
                        if compute_gradients else '')
            print(f'  [{done:4d}/{total}]  {param_y}={vy:.6g}  {param_x}={vx:.6g}'
                  f'  loss={float(lv):.4e}{grad_str}', flush=True)

    if compute_gradients:
        return grid, grad_y, grad_x
    return grid


# ── Plotting ───────────────────────────────────────────────────────────────────

def _landscape_axes(landscape):
    """Normalize axis metadata for old pkls (α/β only) and new generic pair pkls."""
    py = landscape.get('param_y', 'recomb_alpha')
    px = landscape.get('param_x', 'recomb_beta_90')
    vals_y = np.array(landscape.get('vals_y', landscape['alpha_vals']))
    vals_x = np.array(landscape.get('vals_x', landscape['beta90_vals']))
    gt_y = float(landscape.get('gt_param_y', landscape['gt_alpha']))
    gt_x = float(landscape.get('gt_param_x', landscape['gt_beta90']))
    grad_y = landscape.get('grad_param_y')
    grad_x = landscape.get('grad_param_x')
    if grad_y is None and 'grad_alpha' in landscape:
        grad_y = landscape['grad_alpha']
    if grad_x is None and 'grad_beta90' in landscape:
        grad_x = landscape['grad_beta90']
    # PKLs and in-memory dicts store these as nested lists; plotting needs ndarray ops.
    if grad_y is not None:
        grad_y = np.asarray(grad_y, dtype=np.float64)
    if grad_x is not None:
        grad_x = np.asarray(grad_x, dtype=np.float64)
    return py, px, vals_y, vals_x, gt_y, gt_x, grad_y, grad_x


def _pair_file_suffix(landscape):
    py = landscape.get('param_y', 'recomb_alpha')
    px = landscape.get('param_x', 'recomb_beta_90')
    if py == 'recomb_alpha' and px == 'recomb_beta_90':
        return ''
    return f'_{py}__{px}'


def _load_overlay_trajectories(overlay_dir, loss_name, track_name):
    """Load matching 2d_opt pkl trajectories for overlay."""
    import glob as _glob
    pattern = os.path.join(overlay_dir, f'{loss_name}_*recomb_alpha+recomb_beta_90*{track_name}.pkl')
    paths = sorted(_glob.glob(pattern))
    trajectories = []
    for path in paths:
        with open(path, 'rb') as f:
            r = pickle.load(f)
        scales = r['scales']  # [scale_alpha, scale_beta90]
        for trial in r['trials']:
            traj_pn = np.array(trial['param_trajectory'])  # (steps, 2) in p_n space
            # convert p_n → physical
            traj_phys = traj_pn * np.array(scales)
            trajectories.append(traj_phys)
    print(f'  Loaded {len(trajectories)} trajectories from {overlay_dir!r}')
    return trajectories


def plot_landscape_plotly(landscape, output_dir):
    import plotly.graph_objects as go

    loss_name   = landscape['loss_name']
    track_name  = landscape['track_name']
    param_y, param_x, vals_y, vals_x, gt_y, gt_x, grad_y, grad_x = _landscape_axes(landscape)
    lbl_y = PARAM_LABELS.get(param_y, param_y)
    lbl_x = PARAM_LABELS.get(param_x, param_x)
    grid        = np.array(landscape['grid'])
    direction   = landscape['direction']
    mom_mev     = landscape['momentum_mev']
    grid_size   = landscape['grid_size']
    range_frac  = landscape['range_frac']

    loss_label = LOSS_LABELS.get(loss_name, loss_name)

    vmin = np.nanpercentile(grid, 2)
    vmax = np.nanpercentile(grid, 98)

    # Build hover text
    def _hover(i, j):
        s = (f'{param_y}={vals_y[i]:.5g}<br>{param_x}={vals_x[j]:.5g}<br>loss={grid[i,j]:.4e}')
        if grad_y is not None:
            s += (f'<br>∂L/∂{param_y}={grad_y[i,j]:.3e}'
                  f'<br>∂L/∂{param_x}={grad_x[i,j]:.3e}')
        return s

    hover = np.array([[_hover(i, j) for j in range(len(vals_x))]
                      for i in range(len(vals_y))])

    heatmap = go.Heatmap(
        x=vals_x.tolist(),
        y=vals_y.tolist(),
        z=grid.tolist(),
        text=hover.tolist(),
        hovertemplate='%{text}<extra></extra>',
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(
            title=dict(text=loss_label, side='right'),
        ),
    )

    gt_marker = go.Scatter(
        x=[gt_x], y=[gt_y],
        mode='markers',
        marker=dict(symbol='star', size=14, color='red'),
        name=f'GT  ({param_y}={gt_y:.4g}, {param_x}={gt_x:.4g})',
        hovertemplate=(f'GT<br>{param_y}={gt_y:.6g}<br>{param_x}={gt_x:.6g}<extra></extra>'),
    )

    title = (f'Loss landscape | {loss_label} | track: {track_name} '
             f'dir={direction} T={mom_mev} MeV<br>'
             f'{grid_size}×{grid_size} grid ±{range_frac*100:.0f}% around GT')

    traces = [heatmap, gt_marker]

    if grad_y is not None:
        # Subsample to avoid clutter: aim for ~20×20 arrows max
        step = max(1, len(vals_y) // 20)
        da = grad_y[::step, ::step]
        db = grad_x[::step, ::step]
        aa = vals_y[::step]
        bb = vals_x[::step]
        # Normalise arrow lengths to a fixed fraction of the axis range
        mag = np.sqrt(da**2 + db**2) + 1e-30
        scale_a  = (vals_y[-1]  - vals_y[0])  * 0.03
        scale_b  = (vals_x[-1] - vals_x[0]) * 0.03
        ua = -da / mag * scale_a   # descent direction
        ub = -db / mag * scale_b
        # One Scatter trace per arrow (x=[tail, head, None], y=[tail, head, None])
        ax_list, ay_list = [], []
        for ii in range(len(aa)):
            for jj in range(len(bb)):
                ax_list += [bb[jj], bb[jj] + ub[ii, jj], None]
                ay_list += [aa[ii], aa[ii] + ua[ii, jj], None]
        traces.append(go.Scatter(
            x=ax_list, y=ay_list,
            mode='lines',
            line=dict(color='rgba(255,255,255,0.6)', width=1),
            hoverinfo='skip',
            showlegend=True,
            name='−∇L (descent)',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        xaxis=dict(title=lbl_x),
        yaxis=dict(title=lbl_y),
        legend=dict(x=0.01, y=0.99),
        width=750, height=600,
    )

    noise_scale = landscape.get('noise_scale', 0.0)
    noise_tag   = f'_noise{noise_scale:.3g}'.replace('.', 'p') if noise_scale > 0.0 else ''
    mom_tag     = f'_T{mom_mev:.0f}MeV'
    pair_slug   = _pair_file_suffix(landscape)
    os.makedirs(output_dir, exist_ok=True)
    fname = f'landscape_{loss_name}_{track_name}{mom_tag}{pair_slug}_{grid_size}x{grid_size}{noise_tag}.html'
    out_path = os.path.join(output_dir, fname)
    fig.write_html(out_path)
    print(f'  Saved: {out_path}')


def _plot_landscape_single(landscape, output_dir, overlay_dir, norm, log_scale):
    """Render one matplotlib landscape figure with the given norm."""
    loss_name   = landscape['loss_name']
    track_name  = landscape['track_name']
    param_y, param_x, vals_y, vals_x, gt_y, gt_x, grad_y, grad_x = _landscape_axes(landscape)
    lbl_y = PARAM_LABELS.get(param_y, param_y)
    lbl_x = PARAM_LABELS.get(param_x, param_x)
    grid        = np.array(landscape['grid'])
    direction   = landscape['direction']
    mom_mev     = landscape['momentum_mev']
    grid_size   = landscape['grid_size']
    range_frac  = landscape['range_frac']

    loss_label = LOSS_LABELS.get(loss_name, loss_name)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.pcolormesh(vals_x, vals_y, grid,
                       norm=norm, cmap='viridis', shading='auto')
    cbar = fig.colorbar(im, ax=ax)
    scale_tag = ' (log)' if log_scale else ''
    cbar.set_label(f'{loss_label}{scale_tag}', fontsize=9)

    try:
        ax.contour(vals_x, vals_y, grid, levels=12,
                   colors='white', linewidths=0.5, alpha=0.4,
                   norm=norm)
    except Exception:
        pass

    if grad_y is not None:
        ga = grad_y
        gb = grad_x
        span_a = vals_y[-1]  - vals_y[0]  or 1.0
        span_b = vals_x[-1] - vals_x[0] or 1.0
        u = -gb / span_b
        v = -ga / span_a
        ax.streamplot(vals_x, vals_y, u, v,
                      color='white', linewidth=0.8, arrowsize=1.2,
                      density=1.2, minlength=0.05, zorder=3)

    ax.plot(gt_x, gt_y, '*', color='red', ms=14, zorder=5,
            label=f'GT  ({param_y}={gt_y:.4g}, {param_x}={gt_x:.4g})')

    if overlay_dir is not None:
        if param_y == 'recomb_alpha' and param_x == 'recomb_beta_90':
            trajs = _load_overlay_trajectories(overlay_dir, loss_name, track_name)
            for k, traj in enumerate(trajs):
                label = 'opt trajectories' if k == 0 else None
                ax.plot(traj[:, 1], traj[:, 0], lw=0.8, alpha=0.5, color='cyan', label=label)
                ax.plot(traj[0, 1], traj[0, 0], 'o', color='cyan', ms=3, alpha=0.7)
                ax.annotate('', xy=(traj[-1, 1], traj[-1, 0]),
                            xytext=(traj[-2, 1], traj[-2, 0]),
                            arrowprops=dict(arrowstyle='->', color='cyan', lw=0.8))
        else:
            print('  (overlay skipped: trajectories only for recomb_alpha + recomb_beta_90)')

    ax.set_xlabel(lbl_x, fontsize=10)
    ax.set_ylabel(lbl_y, fontsize=10)
    title_scale = '  [log color scale]' if log_scale else ''
    ax.set_title(
        f'Loss landscape  |  {loss_label}  |  track: {track_name}  '
        f'dir={direction}  T={mom_mev} MeV{title_scale}\n'
        f'{grid_size}×{grid_size} grid  ±{range_frac*100:.0f}%% around GT',
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    noise_scale = landscape.get('noise_scale', 0.0)
    noise_tag   = f'_noise{noise_scale:.3g}'.replace('.', 'p') if noise_scale > 0.0 else ''
    log_suffix  = '_log' if log_scale else ''
    mom_tag     = f'_T{mom_mev:.0f}MeV'
    pair_slug   = _pair_file_suffix(landscape)
    os.makedirs(output_dir, exist_ok=True)
    fname = f'landscape_{loss_name}_{track_name}{mom_tag}{pair_slug}_{grid_size}x{grid_size}{noise_tag}{log_suffix}.pdf'
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_landscape(landscape, output_dir, overlay_dir=None):
    grid = np.array(landscape['grid'])

    vmin = np.nanpercentile(grid, 2)
    vmax = np.nanpercentile(grid, 98)

    linear_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    _plot_landscape_single(landscape, output_dir, overlay_dir, linear_norm, log_scale=False)

    pos_min = grid[grid > 0].min() if np.any(grid > 0) else 1e-10
    log_norm = mcolors.LogNorm(vmin=max(pos_min, vmax * 1e-6), vmax=vmax)
    _plot_landscape_single(landscape, output_dir, overlay_dir, log_norm, log_scale=True)

    plot_landscape_plotly(landscape, output_dir)


# ── Noise ─────────────────────────────────────────────────────────────────────

def apply_noise_to_gt(gt_arrays, simulator, noise_scale, noise_seed):
    """Add calibrated detector noise to GT arrays (MicroBooNE model).

    Converts ADC noise to signal units via electrons_per_adc.
    noise_scale=1.0 gives realistic detector noise amplitude.
    """
    cfg       = simulator.config
    noise_dict = generate_noise(cfg, key=jax.random.PRNGKey(noise_seed))
    n_readouts = cfg.volumes[0].n_planes if simulator._readout_type == 'wire' else 1
    noisy = []
    for v in range(cfg.n_volumes):
        for p in range(n_readouts):
            gt  = gt_arrays[v * n_readouts + p]
            noise = noise_dict[(v, p)] * noise_scale
            # forward pads planes to max_wires; pad noise rows to match
            if noise.shape[0] < gt.shape[0]:
                noise = jnp.pad(noise, ((0, gt.shape[0] - noise.shape[0]), (0, 0)))
            noisy.append(gt + noise)
    return tuple(noisy)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    direction = tuple(float(x) for x in args.direction.split(','))
    loss_names = [l.strip() for l in args.loss.split(',')]
    for name in loss_names:
        if name not in VALID_LOSSES:
            raise ValueError(f'Unknown loss {name!r}')
    if args.param_y not in VALID_PARAMS or args.param_x not in VALID_PARAMS:
        raise ValueError(
            f'Unknown param. Choose from {VALID_PARAMS!r}; got '
            f'{args.param_y!r}, {args.param_x!r}')
    if args.param_y == args.param_x:
        raise ValueError('--param-y and --param-x must differ')

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Load-only mode (skip computation) ─────────────────────────────────────
    if args.load_pkl:
        print(f'Loading landscape from {args.load_pkl!r}')
        with open(args.load_pkl, 'rb') as f:
            landscape = pickle.load(f)
        if not args.no_plots:
            plot_landscape(landscape, args.results_dir, overlay_dir=args.overlay)
        print('Done.')
        return

    # ── Build simulator ────────────────────────────────────────────────────────
    print(f'JAX devices : {jax.devices()}')
    print(f'Grid        : {args.grid}×{args.grid}  range ±{args.range_frac*100:.0f}%')
    print(f'Axes        : Y={args.param_y}  X={args.param_x}')
    print(f'Losses      : {loss_names}')
    print(f'Track       : {args.track_name}  direction={direction}  '
          f'start_mm={tuple(args.start_position_mm) if args.start_position_mm is not None else (0.0, 0.0, 0.0)}')

    print('\nBuilding differentiable simulator...')
    detector_config = generate_detector(CONFIG_PATH)
    simulator = DetectorSimulator(
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

    start_mm = (0.0, 0.0, 0.0)
    if args.start_position_mm is not None:
        start_mm = tuple(float(x) for x in args.start_position_mm)

    print(f'Generating muon track  direction={direction}  T={args.momentum} MeV  start_mm={start_mm}...')
    track = generate_muon_track(
        start_position_mm=start_mm,
        direction=direction,
        kinetic_energy_mev=args.momentum,
        step_size_mm=0.1,
        track_id=1,
        #detector_bounds_mm=DETECTOR_BOUNDS_MM,
    )
    track = filter_track_inside_volumes(track, simulator.config.volumes)
    deposits = build_deposit_data(
        track['position'], track['de'], track['dx'], simulator.config,
        theta=track['theta'], phi=track['phi'],
        track_ids=track['track_id'],
    )
    print(f'Generated {sum(v.n_actual for v in deposits.volumes):,} deposits')

    print('Warming up JIT...')
    t0 = time.time()
    simulator.warm_up()
    print(f'Done ({time.time() - t0:.1f} s)')

    gt_params = simulator.default_sim_params._replace(
        lifetime_us    = jnp.array(GT_LIFETIME_US),
        velocity_cm_us = jnp.array(GT_VELOCITY_CM_US),
    )

    recomb_model = simulator.recomb_model
    gt_y = _get_gt_val(args.param_y, gt_params, recomb_model)
    gt_x = _get_gt_val(args.param_x, gt_params, recomb_model)
    print(f'GT  {args.param_y}={gt_y:.6g}  {args.param_x}={gt_x:.6g}')

    print('Computing GT forward pass...')
    t0 = time.time()
    gt_arrays = simulator.forward(gt_params, deposits)
    jax.block_until_ready(gt_arrays)
    print(f'Done ({time.time() - t0:.1f} s)  —  {len(gt_arrays)} plane arrays')

    signal_rms = float(np.mean([float(jnp.std(a)) for a in gt_arrays]))
    if args.noise_scale > 0.0:
        noisy_gt = apply_noise_to_gt(gt_arrays, simulator, args.noise_scale, args.noise_seed)
        noise_rms = float(np.mean([float(jnp.std(n - c)) for n, c in zip(noisy_gt, gt_arrays)]))
        gt_arrays = noisy_gt
        print(f'  Noise applied  scale={args.noise_scale}  seed={args.noise_seed}')
        print(f'  Signal RMS : {signal_rms:.4g}  Noise RMS : {noise_rms:.4g}'
              f'  SNR ≈ {signal_rms / max(noise_rms, 1e-30):.2f}')
    else:
        print(f'  Signal RMS : {signal_rms:.4g}  (no noise; use --noise-scale to add noise)')

    weights = tuple(
        make_sobolev_weight(arr.shape[0], arr.shape[1], max_pad=SOBOLEV_MAX_PAD)
        for arr in gt_arrays
    )

    # ── Grid ──────────────────────────────────────────────────────────────────
    N = args.grid
    rf = args.range_frac
    if args.range_y is not None:
        y_lo, y_hi = args.range_y
    elif args.alpha_range is not None and args.param_y == 'recomb_alpha':
        y_lo, y_hi = args.alpha_range
    else:
        y_lo, y_hi = gt_y * (1 - rf), gt_y * (1 + rf)
    if args.range_x is not None:
        x_lo, x_hi = args.range_x
    elif args.beta_range is not None and args.param_x == 'recomb_beta_90':
        x_lo, x_hi = args.beta_range
    else:
        x_lo, x_hi = gt_x * (1 - rf), gt_x * (1 + rf)
    vals_y = np.linspace(y_lo, y_hi, N)
    vals_x = np.linspace(x_lo, x_hi, N)

    # approximate relative span (for plot subtitles)
    f = max(
        abs(y_hi - gt_y) / max(abs(gt_y), 1e-30),
        abs(x_hi - gt_x) / max(abs(gt_x), 1e-30),
    )

    # ── Loop over losses ───────────────────────────────────────────────────────
    for loss_name in loss_names:
        print(f'\n{"#" * 60}')
        print(f'  Loss: {loss_name}')
        print(f'{"#" * 60}')

        loss_fn = make_loss_fn(
            loss_name, simulator, deposits, gt_arrays, weights, gt_params,
            args.param_y, args.param_x)

        # Warm up
        print('  Compiling loss fn...')
        t0 = time.time()
        gy = jnp.array(gt_y, dtype=jnp.float32)
        gx = jnp.array(gt_x, dtype=jnp.float32)
        _ = loss_fn(gy, gx)
        jax.block_until_ready(_)
        _ = loss_fn(gy, gx)
        jax.block_until_ready(_)
        print(f'  Done ({time.time() - t0:.1f} s)')

        print(f'  Evaluating {N}×{N} grid ({N*N} points)'
              f'{"  + gradients" if args.gradients else ""}...')
        t0 = time.time()
        result = evaluate_grid(
            loss_fn, vals_y, vals_x, args.param_y, args.param_x,
            compute_gradients=args.gradients)
        print(f'  Grid done in {time.time() - t0:.1f} s')

        if args.gradients:
            grid, grad_y_grid, grad_x_grid = result
        else:
            grid = result

        noise_tag = f'_noise{args.noise_scale:.3g}'.replace('.', 'p') if args.noise_scale > 0.0 else ''
        landscape = dict(
            loss_name    = loss_name,
            track_name   = args.track_name,
            direction    = direction,
            momentum_mev = args.momentum,
            start_position_mm = list(start_mm),
            grid_size    = N,
            range_frac   = f,
            param_y      = args.param_y,
            param_x      = args.param_x,
            gt_param_y   = gt_y,
            gt_param_x   = gt_x,
            vals_y       = vals_y.tolist(),
            vals_x       = vals_x.tolist(),
            gt_alpha     = float(gt_y),
            gt_beta90    = float(gt_x),
            alpha_vals   = vals_y.tolist(),
            beta90_vals  = vals_x.tolist(),
            grid         = grid.tolist(),
            noise_scale  = args.noise_scale,
            noise_seed   = args.noise_seed,
        )
        if args.gradients:
            landscape['grad_param_y'] = grad_y_grid.tolist()
            landscape['grad_param_x'] = grad_x_grid.tolist()
            landscape['grad_alpha']   = grad_y_grid.tolist()
            landscape['grad_beta90']  = grad_x_grid.tolist()

        if args.output_pkl:
            pkl_path = os.path.abspath(args.output_pkl)
            os.makedirs(os.path.dirname(pkl_path) or '.', exist_ok=True)
        else:
            if args.param_y == 'recomb_alpha' and args.param_x == 'recomb_beta_90':
                pkl_name = f'landscape_{loss_name}_{args.track_name}_T{args.momentum:.0f}MeV_{N}x{N}{noise_tag}.pkl'
            else:
                pair_tag = f'{args.param_y}__{args.param_x}'
                pkl_name = (f'landscape_{loss_name}_{args.track_name}_T{args.momentum:.0f}MeV_'
                            f'{pair_tag}_{N}x{N}{noise_tag}.pkl')
            pkl_path = os.path.join(args.results_dir, pkl_name)
        with open(pkl_path, 'wb') as f_out:
            pickle.dump(landscape, f_out)
        print(f'  Saved pkl: {pkl_path}')

        if not args.no_plots:
            plot_landscape(landscape, args.results_dir, overlay_dir=args.overlay)

    print('\nDone.')


if __name__ == '__main__':
    main()
