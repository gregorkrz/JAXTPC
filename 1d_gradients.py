#!/usr/bin/env python
"""
Sweep one simulation parameter over ±10 % of its ground-truth value and
record the loss and gradient at each point.

For each requested loss function the script evaluates 2N+1 evenly spaced
points (N to the left, ground truth, N to the right) and saves a pickle
containing the parameter values, loss values, and signed gradients.

Usage examples
--------------
    python 1d_gradients.py
    python 1d_gradients.py --param lifetime_us --N 5
    python 1d_gradients.py --loss sobolev_loss --N 4
    python 1d_gradients.py --direction 1,0,0 --track-name along_x --momentum 500

Output (one file per loss)
--------------------------
    results/1d_gradients/{loss_name}_N{N}_{param_name}_{track_name}.pkl

Each pickle contains a dict with keys:
    param_name, param_gt, param_values, factors,
    loss_values, grad_values, loss_name, N,
    track_name, direction, momentum_mev
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

from tools.geometry import generate_detector
from tools.loader import build_deposit_data
from tools.losses import (
    make_sobolev_weight,
    sobolev_loss,
    sobolev_loss_geomean_log1p,
)
from tools.particle_generator import generate_muon_track
from tools.simulation import DetectorSimulator

# ── Constants ──────────────────────────────────────────────────────────────────
GT_LIFETIME_US    = 10_000.0  # μs
GT_VELOCITY_CM_US = 0.160     # cm/μs
SOBOLEV_MAX_PAD   = 128
RANGE_FRAC        = 0.10      # ±10 %

CONFIG_PATH        = 'config/cubic_wireplane_config.yaml'
N_SEGMENTS         = 10_000
MAX_ACTIVE_BUCKETS = 1000

DETECTOR_BOUNDS_MM = ((-300, 300), (-300, 300), (-300, 300))

PARAM_GT = {
    'velocity_cm_us': GT_VELOCITY_CM_US,
    'lifetime_us':    GT_LIFETIME_US,
}

VALID_LOSSES = ('sobolev_loss', 'sobolev_loss_geomean_log1p')

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--param', default='velocity_cm_us',
                   choices=list(PARAM_GT),
                   help='Parameter to vary (default: velocity_cm_us)')
    p.add_argument('--loss',
                   default='sobolev_loss,sobolev_loss_geomean_log1p',
                   help='Comma-separated list of losses to compute '
                        '(default: both)')
    p.add_argument('--N', type=int, default=2,
                   help='Number of evaluation points on each side of GT '
                        '(default: 2, giving 2N+1 total)')
    p.add_argument('--results-dir', default='results/1d_gradients',
                   help='Output directory (default: results/1d_gradients)')
    p.add_argument('--track-name', default='diagonal',
                   help='Label identifying this track direction, used in the '
                        'output filename (default: diagonal)')
    p.add_argument('--direction', default='1,1,1',
                   help='Muon direction as comma-separated x,y,z '
                        '(default: 1,1,1)')
    p.add_argument('--momentum', type=float, default=1000.0,
                   help='Muon kinetic energy in MeV (default: 1000.0)')
    return p.parse_args()

# ── Loss builders ──────────────────────────────────────────────────────────────

def build_value_and_grad(loss_name, fwd_fn, gt_arrays, weights):
    """Return a JIT-compiled (loss, grad) function of the normalised parameter."""
    if loss_name == 'sobolev_loss':
        def fn(p_n):
            pred = fwd_fn(p_n)
            return sobolev_loss(pred, gt_arrays, weights)
    elif loss_name == 'sobolev_loss_geomean_log1p':
        def fn(p_n):
            pred = fwd_fn(p_n)
            return sobolev_loss_geomean_log1p(pred, gt_arrays, weights)
    else:
        raise ValueError(f'Unknown loss {loss_name!r}. Choose from: {VALID_LOSSES}')
    return jax.jit(jax.value_and_grad(fn))

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    loss_names = [l.strip() for l in args.loss.split(',')]
    for name in loss_names:
        if name not in VALID_LOSSES:
            raise ValueError(f'Unknown loss {name!r}. Choose from: {VALID_LOSSES}')

    direction = tuple(float(x) for x in args.direction.split(','))
    if len(direction) != 3:
        raise ValueError(f'--direction must have exactly 3 components, got {args.direction!r}')

    os.makedirs(args.results_dir, exist_ok=True)

    print(f'JAX devices : {jax.devices()}')
    print(f'Parameter   : {args.param}')
    print(f'Losses      : {loss_names}')
    print(f'N           : {args.N}  ({2 * args.N + 1} total points)')
    print(f'Track name  : {args.track_name}')
    print(f'Direction   : {direction}')
    print(f'Momentum    : {args.momentum} MeV')
    print(f'Results dir : {args.results_dir}')

    # ── Simulator ─────────────────────────────────────────────────────────────
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

    # ── Generate track and deposits ───────────────────────────────────────────
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
    n_total = sum(v.n_actual for v in deposits.volumes)
    print(f'Generated {n_total:,} deposits')

    print('Warming up JIT...')
    t0 = time.time()
    simulator.warm_up()
    print(f'Done ({time.time() - t0:.1f} s)')

    # ── Ground-truth params and forward function ───────────────────────────────
    base_params = simulator.default_sim_params
    gt_params = base_params._replace(
        lifetime_us    = jnp.array(GT_LIFETIME_US),
        velocity_cm_us = jnp.array(GT_VELOCITY_CM_US),
    )

    param_gt = PARAM_GT[args.param]

    if args.param == 'velocity_cm_us':
        def _make_params(p_n):
            return gt_params._replace(velocity_cm_us=p_n * GT_VELOCITY_CM_US)
    else:  # lifetime_us
        def _make_params(p_n):
            return gt_params._replace(lifetime_us=p_n * GT_LIFETIME_US)

    fwd_fn = jax.jit(lambda p_n: simulator.forward(_make_params(p_n), deposits))

    # ── GT arrays and Sobolev weights ─────────────────────────────────────────
    print('Computing GT forward pass...')
    t0 = time.time()
    gt_arrays = simulator.forward(gt_params, deposits)
    jax.block_until_ready(gt_arrays)
    print(f'Done ({time.time() - t0:.1f} s)  —  {len(gt_arrays)} plane arrays')

    weights = tuple(
        make_sobolev_weight(arr.shape[0], arr.shape[1], max_pad=SOBOLEV_MAX_PAD)
        for arr in gt_arrays
    )

    # ── Parameter grid ────────────────────────────────────────────────────────
    # N points strictly left of GT, GT, N points strictly right → 2N+1 total
    left_factors  = np.linspace(1.0 - RANGE_FRAC, 1.0, args.N + 1)[:-1]
    right_factors = np.linspace(1.0, 1.0 + RANGE_FRAC, args.N + 1)[1:]
    factors       = np.concatenate([left_factors, [1.0], right_factors])
    param_values  = factors * param_gt

    print(f'\nParameter grid ({len(factors)} points, GT factor = 1.0):')
    for f, v in zip(factors, param_values):
        marker = ' ← GT' if f == 1.0 else ''
        print(f'  factor={f:.6f}  {args.param}={v:.6g}{marker}')

    # ── Evaluate each loss ────────────────────────────────────────────────────
    for loss_name in loss_names:
        print(f'\n{"=" * 60}')
        print(f'Loss: {loss_name}')

        val_and_grad = build_value_and_grad(loss_name, fwd_fn, gt_arrays, weights)

        print('Compiling value_and_grad...')
        t0 = time.time()
        _ = val_and_grad(jnp.array(factors[0]))
        jax.block_until_ready(_)
        print(f'Done ({time.time() - t0:.1f} s)')

        loss_values = []
        grad_values = []
        grad_times_s = []

        for i, (factor, pval) in enumerate(zip(factors, param_values)):
            p_n = jnp.array(float(factor))
            t0 = time.time()
            lv, gv = val_and_grad(p_n)
            jax.block_until_ready((lv, gv))
            elapsed = time.time() - t0

            lv, gv = float(lv), float(gv)
            loss_values.append(lv)
            grad_values.append(gv)
            grad_times_s.append(elapsed)

            marker = ' ← GT' if factor == 1.0 else ''
            print(f'  [{i + 1:2d}/{len(factors)}] factor={factor:.6f}  '
                  f'{args.param}={pval:.6g}  '
                  f'loss={lv:.4e}  grad={gv:+.4e}  '
                  f'({elapsed * 1e3:.0f} ms){marker}')

        result = dict(
            param_name   = args.param,
            param_gt     = param_gt,
            param_values = list(param_values),
            factors      = list(factors),
            loss_values  = loss_values,
            grad_values  = grad_values,
            grad_times_s = grad_times_s,
            loss_name    = loss_name,
            N            = args.N,
            track_name   = args.track_name,
            direction    = direction,
            momentum_mev = args.momentum,
        )

        pkl_name = f'{loss_name}_N{args.N}_{args.param}_{args.track_name}.pkl'
        pkl_path = os.path.join(args.results_dir, pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump(result, f)
        print(f'Saved: {pkl_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
