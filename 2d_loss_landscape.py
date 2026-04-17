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
from tools.losses import make_sobolev_weight, sobolev_loss, sobolev_loss_geomean_log1p
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

VALID_LOSSES = ('sobolev_loss', 'sobolev_loss_geomean_log1p', 'mse_loss')

LOSS_LABELS = {
    'sobolev_loss':               'Sobolev',
    'sobolev_loss_geomean_log1p': 'Sobolev geomean log1p',
    'mse_loss':                   'MSE',
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--grid', type=int, default=10,
                   help='Grid resolution N: evaluates N×N points (default: 10)')
    p.add_argument('--range-frac', type=float, default=0.15,
                   help='Fractional range around GT on each axis (default: 0.15 → ±15%%)')
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
            pred  = fwd(alpha, beta_90)
            total = jnp.zeros(())
            for pr, gt in zip(pred, gt_arrays):
                norm  = jnp.sum(jnp.abs(gt)) + 1e-12
                total = total + jnp.mean(((pr - gt) / norm) ** 2)
            return total
    else:
        raise ValueError(f'Unknown loss {loss_name!r}')

    return jax.jit(fn)


def evaluate_grid(loss_fn, alpha_vals, beta_90_vals):
    """Evaluate loss_fn on all (alpha, beta_90) grid points, return (N, M) array."""
    N, M = len(alpha_vals), len(beta_90_vals)
    grid = np.zeros((N, M), dtype=np.float32)
    total = N * M
    for i, a in enumerate(alpha_vals):
        for j, b in enumerate(beta_90_vals):
            lv = loss_fn(jnp.array(a, dtype=jnp.float32),
                         jnp.array(b, dtype=jnp.float32))
            jax.block_until_ready(lv)
            grid[i, j] = float(lv)
            done = i * M + j + 1
            print(f'  [{done:3d}/{total}]  alpha={a:.4f}  beta_90={b:.4f}  loss={float(lv):.4e}',
                  flush=True)
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
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)

    im = ax.pcolormesh(beta90_vals, alpha_vals, grid,
                       norm=norm, cmap='viridis', shading='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'{loss_label}  (log scale)', fontsize=9)

    # Contour overlay
    try:
        ax.contour(beta90_vals, alpha_vals, grid, levels=12,
                   colors='white', linewidths=0.5, alpha=0.4,
                   norm=norm)
    except Exception:
        pass

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

    os.makedirs(output_dir, exist_ok=True)
    fname = f'landscape_{loss_name}_{track_name}_{grid_size}x{grid_size}.pdf'
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


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

    weights = tuple(
        make_sobolev_weight(arr.shape[0], arr.shape[1], max_pad=SOBOLEV_MAX_PAD)
        for arr in gt_arrays
    )

    # ── Grid ──────────────────────────────────────────────────────────────────
    N = args.grid
    f = args.range_frac
    alpha_vals  = np.linspace(gt_alpha  * (1 - f), gt_alpha  * (1 + f), N)
    beta90_vals = np.linspace(gt_beta90 * (1 - f), gt_beta90 * (1 + f), N)

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

        print(f'  Evaluating {N}×{N} grid ({N*N} points)...')
        t0 = time.time()
        grid = evaluate_grid(loss_fn, alpha_vals, beta90_vals)
        print(f'  Grid done in {time.time() - t0:.1f} s')

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
        )

        pkl_name = f'landscape_{loss_name}_{args.track_name}_{N}x{N}.pkl'
        pkl_path = os.path.join(args.results_dir, pkl_name)
        with open(pkl_path, 'wb') as f_out:
            pickle.dump(landscape, f_out)
        print(f'  Saved pkl: {pkl_path}')

        plot_landscape(landscape, args.results_dir, overlay_dir=args.overlay)

    print('\nDone.')


if __name__ == '__main__':
    main()
