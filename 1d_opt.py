#!/usr/bin/env python
"""
One-parameter gradient-based optimization starting from ±RANGE_FRAC of GT.

For each of 2N+1 evenly-spaced starting points and M independent trials,
runs a gradient-based optimizer and records the full parameter + loss
trajectory at every step.

Supported optimizers
--------------------
  adam          Adam (no weight decay, β₁=0.9, β₂=0.999, ε=1e-8)
  sgd           Vanilla SGD
  momentum_sgd  SGD with momentum (momentum=0.9)

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
    python 1d_opt.py
    python 1d_opt.py --param lifetime_us --optimizer adam --lr 0.01 --max-steps 200
    python 1d_opt.py --param lifetime_us --N 5 --M 5 --optimizer momentum_sgd

Output (one file per loss)
--------------------------
    results/1d_opt/{loss_name}_N{N}_M{M}_{optimizer}_{param_name}_{track_name}.pkl

Each pickle contains a dict with keys:
    param_name, param_gt, optimizer, lr, max_steps, N, M,
    loss_name, track_name, direction, momentum_mev,
    starting_factors, starting_param_values,
    trials: list of length 2N+1, each a list of M dicts with keys:
        param_trajectory  (max_steps+1 values, step 0 = starting point)
        loss_trajectory   (max_steps+1 values)
        total_time_s
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
import optax

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
GT_LIFETIME_US    = 10_000.0
GT_VELOCITY_CM_US = 0.160
SOBOLEV_MAX_PAD   = 128
RANGE_FRAC        = 0.05

CONFIG_PATH        = 'config/cubic_wireplane_config.yaml'
N_SEGMENTS         = 10_000
MAX_ACTIVE_BUCKETS = 1000
DETECTOR_BOUNDS_MM = ((-300, 300), (-300, 300), (-300, 300))

_JAX_CACHE_DIR = os.path.expanduser('~/.cache/jax_compilation_cache')
jax.config.update('jax_compilation_cache_dir', _JAX_CACHE_DIR)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)

VALID_PARAMS = (
    'velocity_cm_us',
    'lifetime_us',
    'diffusion_trans_cm2_us',
    'diffusion_long_cm2_us',
    'recomb_alpha',
    'recomb_beta',
    'recomb_beta_90',
    'recomb_R',
)

VALID_LOSSES     = ('sobolev_loss', 'sobolev_loss_geomean_log1p', 'mse_loss')
VALID_OPTIMIZERS = ('adam', 'sgd', 'momentum_sgd')

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
    p.add_argument('--param', default='velocity_cm_us', choices=VALID_PARAMS,
                   help='Parameter to optimize (default: velocity_cm_us)')
    p.add_argument('--optimizer', default='adam', choices=VALID_OPTIMIZERS,
                   help='Optimizer: adam | sgd | momentum_sgd (default: adam)')
    p.add_argument('--lr', type=float, default=0.01,
                   help='Learning rate (default: 0.01)')
    p.add_argument('--max-steps', type=int, default=100,
                   help='Max optimization steps per trial (default: 100)')
    p.add_argument('--tol', type=float, default=1e-5,
                   help='Early-stop relative tolerance on p_n (default: 1e-5)')
    p.add_argument('--patience', type=int, default=20,
                   help='Steps over which relative change is checked (default: 20)')
    p.add_argument('--loss',
                   default='sobolev_loss,sobolev_loss_geomean_log1p',
                   help='Comma-separated list of losses (default: both Sobolev)')
    p.add_argument('--N', type=int, default=3,
                   help='Starting points on each side of GT (default: 3, '
                        'giving 2N+1 total)')
    p.add_argument('--M', type=int, default=3,
                   help='Trials per starting point (default: 3)')
    p.add_argument('--results-dir', default='results/1d_opt',
                   help='Output directory (default: results/1d_opt)')
    p.add_argument('--track-name', default='diagonal',
                   help='Label for the track direction (default: diagonal)')
    p.add_argument('--direction', default='1,1,1',
                   help='Muon direction as x,y,z (default: 1,1,1)')
    p.add_argument('--momentum', type=float, default=1000.0,
                   help='Muon kinetic energy in MeV (default: 1000.0)')
    return p.parse_args()

# ── Parameter setter (identical to 1d_gradients.py) ───────────────────────────

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

# ── Loss builder ───────────────────────────────────────────────────────────────

def build_loss_fn(loss_name, fwd_fn, gt_arrays, weights):
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

# ── Optimizer factory ──────────────────────────────────────────────────────────

def make_optax_optimizer(optimizer_name, lr):
    """Return an optax GradientTransformation for the chosen optimizer."""
    if optimizer_name == 'adam':
        return optax.adam(lr)
    elif optimizer_name == 'sgd':
        return optax.sgd(lr)
    elif optimizer_name == 'momentum_sgd':
        return optax.sgd(lr, momentum=0.9)
    else:
        raise ValueError(f'Unknown optimizer {optimizer_name!r}. '
                         f'Choose from: {VALID_OPTIMIZERS}')

# ── Single optimization trial ──────────────────────────────────────────────────

def run_trial(p0_pn, val_and_grad_fn, optimizer, max_steps, tol=1e-5, patience=20):
    """Run one optimization trial from starting p_n value p0_pn.

    Early stops when the relative change in p_n over the last `patience`
    steps is below `tol`:  |p_now - p_patience_ago| / |p_patience_ago| < tol

    Returns dict with param_trajectory, loss_trajectory, total_time_s,
    stopped_early (bool), and steps_run (int, excluding step 0).
    """
    p = jnp.array(float(p0_pn))
    opt_state = optimizer.init(p)

    param_traj = []
    loss_traj  = []

    t_start = time.time()

    # Step 0: record starting point
    lv, _ = val_and_grad_fn(p)
    jax.block_until_ready(lv)
    param_traj.append(float(p))
    loss_traj.append(float(lv))

    stopped_early = False
    for step in range(max_steps):
        lv, gv = val_and_grad_fn(p)
        jax.block_until_ready((lv, gv))
        updates, opt_state = optimizer.update(gv, opt_state)
        p = optax.apply_updates(p, updates)
        lv_new, _ = val_and_grad_fn(p)
        jax.block_until_ready(lv_new)
        param_traj.append(float(p))
        loss_traj.append(float(lv_new))

        if step >= patience:
            p_now  = param_traj[-1]
            p_prev = param_traj[-1 - patience]
            if abs(p_now - p_prev) / (abs(p_prev) + 1e-30) < tol:
                stopped_early = True
                break

    return dict(
        param_trajectory = param_traj,
        loss_trajectory  = loss_traj,
        total_time_s     = time.time() - t_start,
        stopped_early    = stopped_early,
        steps_run        = len(param_traj) - 1,
    )

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

    print(f'JAX devices : {jax.devices()}')
    print(f'Parameter   : {args.param}')
    print(f'Optimizer   : {args.optimizer}  lr={args.lr}')
    print(f'Max steps   : {args.max_steps}')
    print(f'Losses      : {loss_names}')
    print(f'N           : {args.N}  ({2 * args.N + 1} starting points)')
    print(f'M           : {args.M}  (trials per starting point)')
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

    # ── Ground-truth params ────────────────────────────────────────────────────
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

    # ── Starting point grid ────────────────────────────────────────────────────
    # factors are relative to GT (0.95 … 1.05); p_n_values are in normalised space
    left_factors  = np.linspace(1.0 - RANGE_FRAC, 1.0, args.N + 1)[:-1]
    right_factors = np.linspace(1.0, 1.0 + RANGE_FRAC, args.N + 1)[1:]
    factors       = np.concatenate([left_factors, [1.0], right_factors])
    p_n_values    = p_n_gt * factors          # actual values fed to fwd_fn
    param_values  = p_n_values * scale        # physical values (= factors * param_gt)

    print(f'\nStarting point grid ({len(factors)} points, GT factor = 1.0):')
    for f, pn, v in zip(factors, p_n_values, param_values):
        marker = ' ← GT' if f == 1.0 else ''
        print(f'  factor={f:.6f}  p_n={pn:.6f}  {args.param}={v:.6g}{marker}')

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = make_optax_optimizer(args.optimizer, args.lr)

    # ── Evaluate each loss ────────────────────────────────────────────────────
    for loss_name in loss_names:
        print(f'\n{"=" * 60}')
        print(f'Loss: {loss_name}')

        val_and_grad_fn = build_loss_fn(loss_name, fwd_fn, gt_arrays, weights)

        print('Compiling value_and_grad...')
        t0 = time.time()
        _ = val_and_grad_fn(jnp.array(p_n_values[0]))
        jax.block_until_ready(_)
        _ = val_and_grad_fn(jnp.array(p_n_values[0]))
        jax.block_until_ready(_)
        print(f'Done ({time.time() - t0:.1f} s)')

        all_trials = []  # list of 2N+1 entries, each a list of M trial dicts

        n_starting      = len(factors)
        n_total_trials  = n_starting * args.M
        trial_counter   = 0

        for i, (factor, p_n, pval) in enumerate(zip(factors, p_n_values, param_values)):
            marker = ' ← GT' if factor == 1.0 else ''
            print(f'\n  Starting point [{i + 1}/{n_starting}]  '
                  f'factor={factor:.6f}  p_n={p_n:.6f}  {args.param}={pval:.6g}{marker}')

            point_trials = []
            for m in range(args.M):
                trial_counter += 1
                print(f'    Trial {m + 1}/{args.M}  '
                      f'(overall {trial_counter}/{n_total_trials}) ...',
                      end='', flush=True)

                trial = run_trial(
                    p_n, val_and_grad_fn, optimizer,
                    args.max_steps, tol=args.tol, patience=args.patience,
                )
                point_trials.append(trial)

                start_loss = trial['loss_trajectory'][0]
                final_loss = trial['loss_trajectory'][-1]
                final_p    = trial['param_trajectory'][-1]
                early_tag  = f'  [early@{trial["steps_run"]}]' if trial['stopped_early'] else ''
                print(f'  loss {start_loss:.4e} → {final_loss:.4e}  '
                      f'p_n {p_n:.4f} → {final_p:.4f}  '
                      f'({trial["total_time_s"]:.1f} s){early_tag}')

            all_trials.append(point_trials)

        result = dict(
            param_name            = args.param,
            param_gt              = param_gt,
            scale                 = scale,
            p_n_gt                = p_n_gt,
            optimizer             = args.optimizer,
            lr                    = args.lr,
            max_steps             = args.max_steps,
            tol                   = args.tol,
            patience              = args.patience,
            N                     = args.N,
            M                     = args.M,
            loss_name             = loss_name,
            track_name            = args.track_name,
            direction             = direction,
            momentum_mev          = args.momentum,
            starting_factors      = list(factors),
            starting_p_n_values   = list(p_n_values),
            starting_param_values = list(param_values),
            trials                = all_trials,
        )

        pkl_name = (f'{loss_name}_N{args.N}_M{args.M}_{args.optimizer}_'
                    f'{args.param}_{args.track_name}.pkl')
        pkl_path = os.path.join(args.results_dir, pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump(result, f)
        print(f'\nSaved: {pkl_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
