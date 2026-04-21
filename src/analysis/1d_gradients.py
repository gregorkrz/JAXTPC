#!/usr/bin/env python
"""
Sweep one simulation parameter over ±RANGE_FRAC of its ground-truth value and
record the loss and gradient at each point.

For each requested loss function the script evaluates 2N+1 evenly spaced
points (N to the left, ground truth, N to the right) and saves a pickle
containing the parameter values, loss values, and signed gradients.

Supported parameters
--------------------
  velocity_cm_us        drift velocity
  lifetime_us           electron lifetime
  diffusion_trans_cm2_us transverse diffusion coefficient
  diffusion_long_cm2_us  longitudinal diffusion coefficient
  recomb_alpha          recombination α  (both models)
  recomb_beta           recombination β  (modified_box only)
  recomb_beta_90        recombination β₉₀ (emb only)
  recomb_R              recombination R anisotropy (emb only)

Usage examples
--------------
    python 1d_gradients.py
    python 1d_gradients.py --param lifetime_us --N 5
    python 1d_gradients.py --param recomb_alpha --N 4
    python 1d_gradients.py --direction 1,0,0 --track-name along_x --momentum 500
    python 1d_gradients.py --param recomb_beta_90 --fixed-param recomb_alpha --fixed-value 0.905

Output (one file per loss)
--------------------------
    results/1d_gradients/{loss_name}_N{N}_{param_name}_{track_name}.pkl
    results/1d_gradients/{loss_name}_N{N}_{param_name}_{track_name}_fixed_{fixed_param}{fixed_value}.pkl

Each pickle contains a dict with keys:
    param_name, param_gt, param_values, factors,
    loss_values, grad_values, grad_times_s, loss_name, N,
    track_name, direction, momentum_mev
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
GT_LIFETIME_US    = 10_000.0  # μs  — override config value
GT_VELOCITY_CM_US = 0.160     # cm/μs — override config value
SOBOLEV_MAX_PAD   = 128
RANGE_FRAC        = 0.05      # ±5 %

CONFIG_PATH        = 'config/cubic_wireplane_config.yaml'
N_SEGMENTS         = 10_000
MAX_ACTIVE_BUCKETS = 1000
DETECTOR_BOUNDS_MM = ((-300, 300), (-300, 300), (-300, 300))

# Persist compiled XLA programs across runs
_JAX_CACHE_DIR = os.path.expanduser('~/.cache/jax_compilation_cache')
jax.config.update('jax_compilation_cache_dir', _JAX_CACHE_DIR)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)

VALID_PARAMS = (
    'velocity_cm_us',
    'lifetime_us',
    'diffusion_trans_cm2_us',
    'diffusion_long_cm2_us',
    'recomb_alpha',
    'recomb_beta',        # modified_box model only
    'recomb_beta_90',     # emb model only
    'recomb_R',           # emb model only
)

VALID_LOSSES = ('sobolev_loss', 'sobolev_loss_geomean_log1p', 'mse_loss')

# Fixed normalisation scales — round numbers independent of GT.
# p_n = param / TYPICAL_SCALE  (GT sits at p_n ≈ 1, but not pinned to 1 exactly)
TYPICAL_SCALES = {
    'velocity_cm_us':         0.1,        # cm/μs
    'lifetime_us':            10_000.0,   # μs
    'diffusion_trans_cm2_us': 1e-5,       # cm²/μs
    'diffusion_long_cm2_us':  1e-5,       # cm²/μs
    'recomb_alpha':           1.0,        # dimensionless
    'recomb_beta':            0.2,        # (kV/cm)(g/cm²)/MeV
    'recomb_beta_90':         0.2,        # (kV/cm)(g/cm²)/MeV
    'recomb_R':               1.0,        # dimensionless
}

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--param', default='velocity_cm_us',
                   choices=VALID_PARAMS,
                   help='Parameter to vary (default: velocity_cm_us)')
    p.add_argument('--loss',
                   default='sobolev_loss,sobolev_loss_geomean_log1p',
                   help='Comma-separated list of losses (default: both)')
    p.add_argument('--N', type=int, default=2,
                   help='Points on each side of GT (default: 2, giving 2N+1 total)')
    p.add_argument('--results-dir', default='results/1d_gradients',
                   help='Output directory (default: results/1d_gradients)')
    p.add_argument('--track-name', default='diagonal',
                   help='Label for the track direction, used in the output '
                        'filename (default: diagonal)')
    p.add_argument('--direction', default='1,1,1',
                   help='Muon direction as x,y,z (default: 1,1,1)')
    p.add_argument('--momentum', type=float, default=1000.0,
                   help='Muon kinetic energy in MeV (default: 1000.0)')
    p.add_argument('--fixed-param', default=None, choices=VALID_PARAMS,
                   help='Fix this parameter to --fixed-value instead of GT '
                        'when evaluating the sweep (GT arrays still use true GT)')
    p.add_argument('--fixed-value', type=float, default=None,
                   help='Physical value to fix --fixed-param to')
    return p.parse_args()

# ── Parameter setter factory ───────────────────────────────────────────────────

def make_param_setter(param_name, gt_params, recomb_model):
    """Return (setter, gt_val, scale).

    setter(p_n) -> SimParams  where  param = p_n * scale
    scale = TYPICAL_SCALES[param_name]  (GT-independent round number)
    gt_val = actual physical GT value  →  p_n_gt = gt_val / scale
    """
    rp    = gt_params.recomb_params
    scale = TYPICAL_SCALES[param_name]

    if param_name == 'velocity_cm_us':
        gt_val = float(gt_params.velocity_cm_us)
        def setter(p_n):
            return gt_params._replace(velocity_cm_us=p_n * scale)

    elif param_name == 'lifetime_us':
        gt_val = float(gt_params.lifetime_us)
        def setter(p_n):
            return gt_params._replace(lifetime_us=p_n * scale)

    elif param_name == 'diffusion_trans_cm2_us':
        gt_val = float(gt_params.diffusion_trans_cm2_us)
        def setter(p_n):
            return gt_params._replace(diffusion_trans_cm2_us=p_n * scale)

    elif param_name == 'diffusion_long_cm2_us':
        gt_val = float(gt_params.diffusion_long_cm2_us)
        def setter(p_n):
            return gt_params._replace(diffusion_long_cm2_us=p_n * scale)

    elif param_name == 'recomb_alpha':
        gt_val = float(rp.alpha)
        def setter(p_n):
            return gt_params._replace(recomb_params=rp._replace(alpha=p_n * scale))

    elif param_name == 'recomb_beta':
        if recomb_model != 'modified_box':
            raise ValueError(
                f'recomb_beta is only valid for the modified_box model; '
                f'this simulator uses {recomb_model!r}')
        gt_val = float(rp.beta)
        def setter(p_n):
            return gt_params._replace(recomb_params=rp._replace(beta=p_n * scale))

    elif param_name == 'recomb_beta_90':
        if recomb_model != 'emb':
            raise ValueError(
                f'recomb_beta_90 is only valid for the emb model; '
                f'this simulator uses {recomb_model!r}')
        gt_val = float(rp.beta_90)
        def setter(p_n):
            return gt_params._replace(recomb_params=rp._replace(beta_90=p_n * scale))

    elif param_name == 'recomb_R':
        if recomb_model != 'emb':
            raise ValueError(
                f'recomb_R is only valid for the emb model; '
                f'this simulator uses {recomb_model!r}')
        gt_val = float(rp.R)
        def setter(p_n):
            return gt_params._replace(recomb_params=rp._replace(R=p_n * scale))

    else:
        raise ValueError(f'Unknown param {param_name!r}. Choose from: {VALID_PARAMS}')

    return setter, gt_val, scale

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
    elif loss_name == 'mse_loss':
        def fn(p_n):
            pred = fwd_fn(p_n)
            total = jnp.zeros(())
            for pr, gt in zip(pred, gt_arrays):
                norm = jnp.sum(jnp.abs(gt)) + 1e-12
                total = total + jnp.mean(((pr - gt) / norm) ** 2)
            return total
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
        raise ValueError(f'--direction must have 3 components, got {args.direction!r}')

    os.makedirs(args.results_dir, exist_ok=True)

    if (args.fixed_param is None) != (args.fixed_value is None):
        raise ValueError('--fixed-param and --fixed-value must be given together')

    print(f'JAX devices : {jax.devices()}')
    print(f'Parameter   : {args.param}')
    print(f'Losses      : {loss_names}')
    print(f'N           : {args.N}  ({2 * args.N + 1} total points)')
    print(f'Track name  : {args.track_name}')
    print(f'Direction   : {direction}')
    print(f'Momentum    : {args.momentum} MeV')
    if args.fixed_param:
        print(f'Fixed param : {args.fixed_param} = {args.fixed_value}')
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

    # ── Ground-truth params ────────────────────────────────────────────────────
    # Override velocity and lifetime with known good values; all other fields
    # (diffusion, recomb) come from the config via default_sim_params.
    gt_params = simulator.default_sim_params._replace(
        lifetime_us    = jnp.array(GT_LIFETIME_US),
        velocity_cm_us = jnp.array(GT_VELOCITY_CM_US),
    )

    _make_params, param_gt, scale = make_param_setter(
        args.param, gt_params, simulator.recomb_model
    )
    p_n_gt = param_gt / scale
    print(f'GT value    : {args.param} = {param_gt:.6g}  '
          f'(scale={scale:.6g},  p_n_gt={p_n_gt:.6g})')

    # Optionally fix a second parameter to a non-GT value for the sweep
    sweep_base_params = gt_params
    if args.fixed_param is not None:
        _fix_setter, _, _ = make_param_setter(args.fixed_param, gt_params, simulator.recomb_model)
        fix_scale = TYPICAL_SCALES[args.fixed_param]
        sweep_base_params = _fix_setter(args.fixed_value / fix_scale)
        print(f'Sweep base  : {args.fixed_param} overridden to {args.fixed_value}')

    _make_params, param_gt, scale = make_param_setter(
        args.param, sweep_base_params, simulator.recomb_model
    )
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
    # factors are relative to GT (0.95 … 1.05); p_n_values are in normalised space
    left_factors  = np.linspace(1.0 - RANGE_FRAC, 1.0, args.N + 1)[:-1]
    right_factors = np.linspace(1.0, 1.0 + RANGE_FRAC, args.N + 1)[1:]
    factors       = np.concatenate([left_factors, [1.0], right_factors])
    p_n_values    = p_n_gt * factors          # actual values fed to fwd_fn
    param_values  = p_n_values * scale        # physical values (= factors * param_gt)

    print(f'\nParameter grid ({len(factors)} points, GT factor = 1.0):')
    for f, pn, v in zip(factors, p_n_values, param_values):
        marker = ' ← GT' if f == 1.0 else ''
        print(f'  factor={f:.6f}  p_n={pn:.6f}  {args.param}={v:.6g}{marker}')

    # ── Evaluate each loss ────────────────────────────────────────────────────
    for loss_name in loss_names:
        print(f'\n{"=" * 60}')
        print(f'Loss: {loss_name}')

        val_and_grad = build_value_and_grad(loss_name, fwd_fn, gt_arrays, weights)

        print('Compiling value_and_grad...')
        t0 = time.time()
        _ = val_and_grad(jnp.array(p_n_values[0]))
        jax.block_until_ready(_)
        _ = val_and_grad(jnp.array(p_n_values[0]))   # flush GPU kernel cache
        jax.block_until_ready(_)
        print(f'Done ({time.time() - t0:.1f} s)')

        loss_values  = []
        grad_values  = []
        grad_times_s = []

        for i, (factor, p_n, pval) in enumerate(zip(factors, p_n_values, param_values)):
            p_n_arr = jnp.array(float(p_n))
            t0 = time.time()
            lv, gv = val_and_grad(p_n_arr)
            jax.block_until_ready((lv, gv))
            elapsed = time.time() - t0

            lv, gv = float(lv), float(gv)
            loss_values.append(lv)
            grad_values.append(gv)
            grad_times_s.append(elapsed)

            marker = ' ← GT' if factor == 1.0 else ''
            print(f'  [{i + 1:2d}/{len(factors)}] factor={factor:.6f}  '
                  f'p_n={p_n:.6f}  {args.param}={pval:.6g}  '
                  f'loss={lv:.4e}  grad={gv:+.4e}  '
                  f'({elapsed * 1e3:.0f} ms){marker}')

        result = dict(
            param_name   = args.param,
            param_gt     = param_gt,
            scale        = scale,
            p_n_gt       = p_n_gt,
            param_values = list(param_values),
            p_n_values   = list(p_n_values),
            factors      = list(factors),
            loss_values  = loss_values,
            grad_values  = grad_values,
            grad_times_s = grad_times_s,
            loss_name    = loss_name,
            N            = args.N,
            track_name   = args.track_name,
            direction    = direction,
            momentum_mev = args.momentum,
            fixed_param  = args.fixed_param,
            fixed_value  = args.fixed_value,
        )

        fixed_tag = f'_fixed_{args.fixed_param}{args.fixed_value}' if args.fixed_param else ''
        pkl_name = f'{loss_name}_N{args.N}_{args.param}_{args.track_name}{fixed_tag}.pkl'
        pkl_path = os.path.join(args.results_dir, pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump(result, f)
        print(f'Saved: {pkl_path}')

    fixed_tag = f'_fixed_{args.fixed_param}{args.fixed_value}' if args.fixed_param else ''
    fixed_args = (f' --fixed-param {args.fixed_param} --fixed-value {args.fixed_value}'
                  if args.fixed_param else '')
    print('\nCommand used:')
    print(f'  python 1d_gradients.py'
          f' --param {args.param}'
          f' --N {args.N}'
          f' --track-name {args.track_name}'
          f' --direction {",".join(str(d) for d in direction)}'
          f' --momentum {args.momentum}'
          f' --loss {",".join(loss_names)}'
          f'{fixed_args}'
          f' --results-dir {args.results_dir}')
    print('\nDone.')


if __name__ == '__main__':
    main()
