#!/usr/bin/env python
"""
Compute and plot the loss landscape over a 2-D grid of
(recomb_alpha, recomb_beta_90) values.

Grid is defined relative to GT values:
    alpha    in [gt * (1 - range_frac), gt * (1 + range_frac)]
    beta_90  in [gt * (1 - range_frac), gt * (1 + range_frac)]

Optionally overlays optimization trajectories from 2d_opt.py pkl files.

Usage
-----
    python 2d_loss_landscape.py
    python 2d_loss_landscape.py --grid 20 --range-frac 0.2
    python 2d_loss_landscape.py --loss sobolev_loss --overlay results/2d_opt
    python 2d_loss_landscape.py --load-pkl results/2d_landscape/landscape_sobolev_loss_diagonal_10x10.pkl
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tools.geometry import generate_detector
from tools.loader import build_deposit_data
from tools.losses import make_sobolev_weight, sobolev_loss, sobolev_loss_geomean_log1p, mse_loss, l1_loss
from tools.noise import generate_noise
from tools.particle_generator import generate_muon_track
from tools.simulation import DetectorSimulator

# ── Constants (shared with 2d_opt.py) ─────────────────────────────────────────
GT_LIFETIME_US    = 10_000.0
GT_VELOCITY_CM_US = 0.160
SOBOLEV_MAX_PAD   = 128

CONFIG_PATH        = 'config/cubic_wireplane_config.yaml'
N_SEGMENTS         = 10_000
MAX_ACTIVE_BUCKETS = 1000
DETECTOR_BOUNDS_MM = ((-300, 300), (-300, 300), (-300, 300))

TYPICAL_SCALES = {
    'recomb_alpha':   1.0,
    'recomb_beta_90': 0.2,
}

_JAX_CACHE_DIR = os.path.expanduser('~/.cache/jax_compilation_cache')
jax.config.update('jax_compilation_cache_dir', _JAX_CACHE_DIR)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)

VALID_LOSSES = ('sobolev_loss', 'sobolev_loss_geomean_log1p', 'mse_loss', 'l1_loss')

LOSS_LABELS = {
    'sobolev_loss':               'Sobolev',
    'sobolev_loss_geomean_log1p': 'Sobolev geomean log1p',
    'mse_loss':                   'MSE',
    'l1_loss':                    'L1',
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--grid', type=int, default=10,
                   help='Grid resolution N: evaluates N×N points (default: 10)')
    p.add_argument('--range-frac', type=float, default=0.15,
                   help='Fractional range around GT on each axis (default: 0.15 → ±15%%)')
    p.add_argument('--alpha-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   help='Explicit alpha limits, e.g. --alpha-range 0.89 0.92 (overrides --range-frac for alpha)')
    p.add_argument('--beta-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   help='Explicit beta_90 limits, e.g. --beta-range 0.19 0.22 (overrides --range-frac for beta)')
    p.add_argument('--loss', default='sobolev_loss,sobolev_loss_geomean_log1p',
                   help='Comma-separated loss(es) to evaluate (default: both Sobolev)')
    p.add_argument('--track-name', default='diagonal',
                   help='Label for track direction (default: diagonal)')
    p.add_argument('--direction', default='1,1,1',
                   help='Muon direction as x,y,z (default: 1,1,1)')
    p.add_argument('--momentum', type=float, default=1000.0,
                   help='Muon kinetic energy in MeV (default: 1000.0)')
    p.add_argument('--results-dir', default='results/2d_landscape',
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
    return p.parse_args()


# ── Loss evaluation ────────────────────────────────────────────────────────────

def make_loss_fn(loss_name, simulator, deposits, gt_arrays, weights, gt_params):
    rp = gt_params.recomb_params

    def fwd(alpha, beta_90):
        new_rp = rp._replace(alpha=alpha, beta_90=beta_90)
        params  = gt_params._replace(recomb_params=new_rp)
        return simulator.forward(params, deposits)

    if loss_name == 'sobolev_loss':
        def fn(alpha, beta_90):
            return sobolev_loss(fwd(alpha, beta_90), gt_arrays, weights)
    elif loss_name == 'sobolev_loss_geomean_log1p':
        def fn(alpha, beta_90):
            return sobolev_loss_geomean_log1p(fwd(alpha, beta_90), gt_arrays, weights)
    elif loss_name == 'mse_loss':
        def fn(alpha, beta_90):
            return mse_loss(fwd(alpha, beta_90), gt_arrays)
    elif loss_name == 'l1_loss':
        def fn(alpha, beta_90):
            return l1_loss(fwd(alpha, beta_90), gt_arrays)
    else:
        raise ValueError(f'Unknown loss {loss_name!r}')

    return jax.jit(fn)


def evaluate_grid(loss_fn, alpha_vals, beta_90_vals, compute_gradients=False):
    """Evaluate loss_fn on all (alpha, beta_90) grid points.

    Returns grid (N, M) always.  With compute_gradients=True also returns
    grad_alpha (N, M) and grad_beta90 (N, M).
    """
    N, M = len(alpha_vals), len(beta_90_vals)
    grid       = np.zeros((N, M), dtype=np.float32)
    grad_alpha  = np.zeros((N, M), dtype=np.float32) if compute_gradients else None
    grad_beta90 = np.zeros((N, M), dtype=np.float32) if compute_gradients else None
    total = N * M

    if compute_gradients:
        val_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))

    for i, a in enumerate(alpha_vals):
        for j, b in enumerate(beta_90_vals):
            a_jax = jnp.array(a, dtype=jnp.float32)
            b_jax = jnp.array(b, dtype=jnp.float32)
            if compute_gradients:
                (lv, (ga, gb)) = val_and_grad_fn(a_jax, b_jax)
                jax.block_until_ready((lv, ga, gb))
                grad_alpha[i, j]  = float(ga)
                grad_beta90[i, j] = float(gb)
            else:
                lv = loss_fn(a_jax, b_jax)
                jax.block_until_ready(lv)
            grid[i, j] = float(lv)
            done = i * M + j + 1
            grad_str = (f'  dα={float(ga):.3e}  dβ={float(gb):.3e}'
                        if compute_gradients else '')
            print(f'  [{done:4d}/{total}]  alpha={a:.4f}  beta_90={b:.4f}'
                  f'  loss={float(lv):.4e}{grad_str}', flush=True)

    if compute_gradients:
        return grid, grad_alpha, grad_beta90
    return grid


# ── Plotting ───────────────────────────────────────────────────────────────────

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
    alpha_vals  = np.array(landscape['alpha_vals'])
    beta90_vals = np.array(landscape['beta90_vals'])
    grid        = np.array(landscape['grid'])
    gt_alpha    = landscape['gt_alpha']
    gt_beta90   = landscape['gt_beta90']
    direction   = landscape['direction']
    mom_mev     = landscape['momentum_mev']
    grid_size   = landscape['grid_size']
    range_frac  = landscape['range_frac']

    loss_label = LOSS_LABELS.get(loss_name, loss_name)

    vmin = np.nanpercentile(grid, 2)
    vmax = np.nanpercentile(grid, 98)

    grad_alpha  = np.array(landscape['grad_alpha'])  if 'grad_alpha'  in landscape else None
    grad_beta90 = np.array(landscape['grad_beta90']) if 'grad_beta90' in landscape else None

    # Build hover text
    def _hover(i, j):
        s = f'α={alpha_vals[i]:.5f}<br>β₉₀={beta90_vals[j]:.5f}<br>loss={grid[i,j]:.4e}'
        if grad_alpha is not None:
            s += (f'<br>∂L/∂α={grad_alpha[i,j]:.3e}'
                  f'<br>∂L/∂β₉₀={grad_beta90[i,j]:.3e}')
        return s

    hover = np.array([[_hover(i, j) for j in range(len(beta90_vals))]
                      for i in range(len(alpha_vals))])

    heatmap = go.Heatmap(
        x=beta90_vals.tolist(),
        y=alpha_vals.tolist(),
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
        x=[gt_beta90], y=[gt_alpha],
        mode='markers',
        marker=dict(symbol='star', size=14, color='red'),
        name=f'GT  (α={gt_alpha:.4g}, β₉₀={gt_beta90:.4g})',
        hovertemplate=f'GT<br>α={gt_alpha:.6g}<br>β₉₀={gt_beta90:.6g}<extra></extra>',
    )

    title = (f'Loss landscape | {loss_label} | track: {track_name} '
             f'dir={direction} T={mom_mev} MeV<br>'
             f'{grid_size}×{grid_size} grid ±{range_frac*100:.0f}% around GT')

    traces = [heatmap, gt_marker]

    if grad_alpha is not None:
        # Subsample to avoid clutter: aim for ~20×20 arrows max
        step = max(1, len(alpha_vals) // 20)
        da = grad_alpha[::step, ::step]
        db = grad_beta90[::step, ::step]
        aa = alpha_vals[::step]
        bb = beta90_vals[::step]
        # Normalise arrow lengths to a fixed fraction of the axis range
        mag = np.sqrt(da**2 + db**2) + 1e-30
        scale_a  = (alpha_vals[-1]  - alpha_vals[0])  * 0.03
        scale_b  = (beta90_vals[-1] - beta90_vals[0]) * 0.03
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
        xaxis=dict(title='recomb β₉₀'),
        yaxis=dict(title='recomb α'),
        legend=dict(x=0.01, y=0.99),
        width=750, height=600,
    )

    noise_scale = landscape.get('noise_scale', 0.0)
    noise_tag   = f'_noise{noise_scale:.3g}'.replace('.', 'p') if noise_scale > 0.0 else ''
    os.makedirs(output_dir, exist_ok=True)
    fname = f'landscape_{loss_name}_{track_name}_{grid_size}x{grid_size}{noise_tag}.html'
    out_path = os.path.join(output_dir, fname)
    fig.write_html(out_path)
    print(f'  Saved: {out_path}')


def plot_landscape(landscape, output_dir, overlay_dir=None):
    loss_name   = landscape['loss_name']
    track_name  = landscape['track_name']
    alpha_vals  = np.array(landscape['alpha_vals'])
    beta90_vals = np.array(landscape['beta90_vals'])
    grid        = np.array(landscape['grid'])        # shape (N_alpha, N_beta90)
    gt_alpha    = landscape['gt_alpha']
    gt_beta90   = landscape['gt_beta90']
    direction   = landscape['direction']
    mom_mev     = landscape['momentum_mev']
    grid_size   = landscape['grid_size']
    range_frac  = landscape['range_frac']

    loss_label = LOSS_LABELS.get(loss_name, loss_name)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Heatmap — rows = alpha (y-axis), cols = beta_90 (x-axis) → transpose
    # grid[i, j] = loss at (alpha_vals[i], beta90_vals[j])
    vmin = np.nanpercentile(grid, 2)
    vmax = np.nanpercentile(grid, 98)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.pcolormesh(beta90_vals, alpha_vals, grid,
                       norm=norm, cmap='viridis', shading='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'{loss_label}', fontsize=9)

    # Contour overlay
    try:
        ax.contour(beta90_vals, alpha_vals, grid, levels=12,
                   colors='white', linewidths=0.5, alpha=0.4,
                   norm=norm)
    except Exception:
        pass

    # Gradient field lines (descent direction) if available
    if 'grad_alpha' in landscape and 'grad_beta90' in landscape:
        ga = np.array(landscape['grad_alpha'])   # dL/d_alpha, shape (N_alpha, N_beta90)
        gb = np.array(landscape['grad_beta90'])  # dL/d_beta90
        # Normalise each component by its axis span so neither axis dominates visually
        span_a = alpha_vals[-1]  - alpha_vals[0]  or 1.0
        span_b = beta90_vals[-1] - beta90_vals[0] or 1.0
        u = -gb / span_b   # descent in beta90 direction (x-axis), normalised
        v = -ga / span_a   # descent in alpha direction   (y-axis), normalised
        # streamplot requires a uniform 1-D grid; ours already is
        ax.streamplot(beta90_vals, alpha_vals, u, v,
                      color='white', linewidth=0.8, arrowsize=1.2,
                      density=1.2, minlength=0.05, zorder=3)

    # GT marker
    ax.plot(gt_beta90, gt_alpha, '*', color='red', ms=14, zorder=5,
            label=f'GT  (α={gt_alpha:.4g}, β₉₀={gt_beta90:.4g})')

    # Overlay trajectories
    if overlay_dir is not None:
        trajs = _load_overlay_trajectories(overlay_dir, loss_name, track_name)
        for k, traj in enumerate(trajs):
            label = 'opt trajectories' if k == 0 else None
            ax.plot(traj[:, 1], traj[:, 0], lw=0.8, alpha=0.5, color='cyan', label=label)
            ax.plot(traj[0, 1], traj[0, 0], 'o', color='cyan', ms=3, alpha=0.7)
            ax.annotate('', xy=(traj[-1, 1], traj[-1, 0]),
                        xytext=(traj[-2, 1], traj[-2, 0]),
                        arrowprops=dict(arrowstyle='->', color='cyan', lw=0.8))

    ax.set_xlabel('recomb β₉₀', fontsize=10)
    ax.set_ylabel('recomb α', fontsize=10)
    ax.set_title(
        f'Loss landscape  |  {loss_label}  |  track: {track_name}  '
        f'dir={direction}  T={mom_mev} MeV\n'
        f'{grid_size}×{grid_size} grid  ±{range_frac*100:.0f}%% around GT',
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    noise_scale = landscape.get('noise_scale', 0.0)
    noise_tag   = f'_noise{noise_scale:.3g}'.replace('.', 'p') if noise_scale > 0.0 else ''
    os.makedirs(output_dir, exist_ok=True)
    fname = f'landscape_{loss_name}_{track_name}_{grid_size}x{grid_size}{noise_tag}.pdf'
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')

    plot_landscape_plotly(landscape, output_dir)


# ── Noise ─────────────────────────────────────────────────────────────────────

def apply_noise_to_gt(gt_arrays, simulator, noise_scale, noise_seed):
    """Add calibrated detector noise to GT arrays (MicroBooNE model).

    Converts ADC noise to signal units via electrons_per_adc.
    noise_scale=1.0 gives realistic detector noise amplitude.
    """
    cfg       = simulator.config
    noise_dict = generate_noise(cfg, key=jax.random.PRNGKey(noise_seed))
    e_per_adc  = cfg.electrons_per_adc
    n_readouts = cfg.volumes[0].n_planes if simulator._readout_type == 'wire' else 1
    noisy = []
    for v in range(cfg.n_volumes):
        for p in range(n_readouts):
            gt  = gt_arrays[v * n_readouts + p]
            noise = noise_dict[(v, p)] * e_per_adc * noise_scale
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

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Load-only mode (skip computation) ─────────────────────────────────────
    if args.load_pkl:
        print(f'Loading landscape from {args.load_pkl!r}')
        with open(args.load_pkl, 'rb') as f:
            landscape = pickle.load(f)
        plot_landscape(landscape, args.results_dir, overlay_dir=args.overlay)
        print('Done.')
        return

    # ── Build simulator ────────────────────────────────────────────────────────
    print(f'JAX devices : {jax.devices()}')
    print(f'Grid        : {args.grid}×{args.grid}  range ±{args.range_frac*100:.0f}%')
    print(f'Losses      : {loss_names}')
    print(f'Track       : {args.track_name}  direction={direction}')

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

    print(f'Generating muon track  direction={direction}  T={args.momentum} MeV...')
    track = generate_muon_track(
        start_position_mm=(0.0, 0.0, 0.0),
        direction=direction,
        kinetic_energy_mev=args.momentum,
        step_size_mm=0.1,
        track_id=1,
        detector_bounds_mm=DETECTOR_BOUNDS_MM,
    )
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

    gt_alpha  = float(gt_params.recomb_params.alpha)
    gt_beta90 = float(gt_params.recomb_params.beta_90)
    print(f'GT  alpha={gt_alpha:.6g}  beta_90={gt_beta90:.6g}')

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
    f = args.range_frac
    a_lo, a_hi = args.alpha_range if args.alpha_range else (gt_alpha  * (1 - f), gt_alpha  * (1 + f))
    b_lo, b_hi = args.beta_range  if args.beta_range  else (gt_beta90 * (1 - f), gt_beta90 * (1 + f))
    alpha_vals  = np.linspace(a_lo, a_hi, N)
    beta90_vals = np.linspace(b_lo, b_hi, N)

    # range_frac stored in pkl is approximate when explicit limits are used
    f = max(abs(a_hi - gt_alpha) / gt_alpha, abs(b_hi - gt_beta90) / gt_beta90)

    # ── Loop over losses ───────────────────────────────────────────────────────
    for loss_name in loss_names:
        print(f'\n{"#" * 60}')
        print(f'  Loss: {loss_name}')
        print(f'{"#" * 60}')

        loss_fn = make_loss_fn(loss_name, simulator, deposits, gt_arrays, weights, gt_params)

        # Warm up
        print('  Compiling loss fn...')
        t0 = time.time()
        _ = loss_fn(jnp.array(gt_alpha, dtype=jnp.float32),
                    jnp.array(gt_beta90, dtype=jnp.float32))
        jax.block_until_ready(_)
        _ = loss_fn(jnp.array(gt_alpha, dtype=jnp.float32),
                    jnp.array(gt_beta90, dtype=jnp.float32))
        jax.block_until_ready(_)
        print(f'  Done ({time.time() - t0:.1f} s)')

        print(f'  Evaluating {N}×{N} grid ({N*N} points)'
              f'{"  + gradients" if args.gradients else ""}...')
        t0 = time.time()
        result = evaluate_grid(loss_fn, alpha_vals, beta90_vals,
                               compute_gradients=args.gradients)
        print(f'  Grid done in {time.time() - t0:.1f} s')

        if args.gradients:
            grid, grad_alpha_grid, grad_beta90_grid = result
        else:
            grid = result

        noise_tag = f'_noise{args.noise_scale:.3g}'.replace('.', 'p') if args.noise_scale > 0.0 else ''
        landscape = dict(
            loss_name    = loss_name,
            track_name   = args.track_name,
            direction    = direction,
            momentum_mev = args.momentum,
            grid_size    = N,
            range_frac   = f,
            gt_alpha     = gt_alpha,
            gt_beta90    = gt_beta90,
            alpha_vals   = alpha_vals.tolist(),
            beta90_vals  = beta90_vals.tolist(),
            grid         = grid.tolist(),
            noise_scale  = args.noise_scale,
            noise_seed   = args.noise_seed,
        )
        if args.gradients:
            landscape['grad_alpha']  = grad_alpha_grid.tolist()
            landscape['grad_beta90'] = grad_beta90_grid.tolist()

        pkl_name = f'landscape_{loss_name}_{args.track_name}_{N}x{N}{noise_tag}.pkl'
        pkl_path = os.path.join(args.results_dir, pkl_name)
        with open(pkl_path, 'wb') as f_out:
            pickle.dump(landscape, f_out)
        print(f'  Saved pkl: {pkl_path}')

        plot_landscape(landscape, args.results_dir, overlay_dir=args.overlay)

    print('\nDone.')


if __name__ == '__main__':
    main()
