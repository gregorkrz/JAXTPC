#!/usr/bin/env python
"""
Two-parameter gradient-based optimization starting from a (2N+1)×(2N+1) grid
of ±RANGE_FRAC starting points around the ground truth.

For each grid point and M independent trials, runs a gradient-based optimizer
and records the full parameter + loss trajectory at every step.

Default parameter pairs
-----------------------
  velocity_cm_us   + lifetime_us              drift physics (both affect charge collection)
  recomb_alpha     + recomb_beta_90           recombination shape (EMB model)
  diffusion_trans_cm2_us + diffusion_long_cm2_us  diffusion tensor

Custom pairs via --pairs, e.g.:
  --pairs velocity_cm_us+lifetime_us,recomb_alpha+recomb_beta_90

Supported optimizers
--------------------
  adam          Adam (no weight decay, β₁=0.9, β₂=0.999, ε=1e-8)
  sgd           Vanilla SGD
  momentum_sgd  SGD with momentum (momentum=0.9)

Output (one file per pair × loss)
----------------------------------
    results/2d_opt/{loss}_N{N}_M{M}_{optimizer}_{param1}+{param2}_{track}.pkl

Each pickle contains:
    param_names, param_gts, scales, p_n_gts,
    optimizer, lr, max_steps, tol, patience, N, M,
    loss_name, track_name, direction, momentum_mev,
    factor_grid,          # list of (f1, f2) relative-to-GT factor pairs
    starting_p_n_values,  # list of (pn1, pn2) absolute p_n starting values
    trials,               # list of (2N+1)^2 entries, each a list of M dicts:
        param_trajectory  (max_steps+1, 2) — or shorter if early stopped
        loss_trajectory   (max_steps+1,)
        total_time_s, stopped_early, steps_run
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

TYPICAL_SCALES = {
    'velocity_cm_us':         0.1,
    'lifetime_us':            10_000.0,
    'diffusion_trans_cm2_us': 1e-5,
    'diffusion_long_cm2_us':  1e-5,
    'recomb_alpha':           1.0,
    'recomb_beta':            0.2,
    'recomb_beta_90':         0.2,
    'recomb_R':               1.0,
}

DEFAULT_PAIRS = [
    ('velocity_cm_us',         'lifetime_us'),
    ('recomb_alpha',           'recomb_beta_90'),
    ('diffusion_trans_cm2_us', 'diffusion_long_cm2_us'),
]
DEFAULT_PAIRS_STR = ','.join(f'{p1}+{p2}' for p1, p2 in DEFAULT_PAIRS)

# Adam / momentum hyperparameters (fixed)
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS   = 1e-8
MOMENTUM   = 0.9

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pairs', default=DEFAULT_PAIRS_STR,
                   help='Comma-separated list of param1+param2 pairs to optimize '
                        f'(default: {DEFAULT_PAIRS_STR})')
    p.add_argument('--optimizer', default='adam', choices=VALID_OPTIMIZERS,
                   help='Optimizer (default: adam)')
    p.add_argument('--lr', type=float, default=0.01,
                   help='Learning rate (default: 0.01)')
    p.add_argument('--max-steps', type=int, default=100,
                   help='Max optimization steps per trial (default: 100)')
    p.add_argument('--tol', type=float, default=1e-5,
                   help='Early-stop relative tolerance on p_n norm (default: 1e-5)')
    p.add_argument('--patience', type=int, default=20,
                   help='Steps over which relative change is checked (default: 20)')
    p.add_argument('--loss',
                   default='sobolev_loss,sobolev_loss_geomean_log1p',
                   help='Comma-separated list of losses (default: both Sobolev)')
    p.add_argument('--N', type=int, default=2,
                   help='Grid points on each side per parameter '
                        '(default: 2, giving (2N+1)²=25 starting points)')
    p.add_argument('--M', type=int, default=3,
                   help='Trials per starting point (default: 3)')
    p.add_argument('--results-dir', default='results/2d_opt',
                   help='Output directory (default: results/2d_opt)')
    p.add_argument('--track-name', default='diagonal',
                   help='Label for the track direction (default: diagonal)')
    p.add_argument('--direction', default='1,1,1',
                   help='Muon direction as x,y,z (default: 1,1,1)')
    p.add_argument('--momentum', type=float, default=1000.0,
                   help='Muon kinetic energy in MeV (default: 1000.0)')
    return p.parse_args()


def parse_pairs(pairs_str):
    """Parse 'p1+p2,p3+p4' into [('p1','p2'), ('p3','p4')]."""
    pairs = []
    for item in pairs_str.split(','):
        item = item.strip()
        if '+' not in item:
            raise ValueError(f'Each pair must be param1+param2, got {item!r}')
        p1, p2 = item.split('+', 1)
        p1, p2 = p1.strip(), p2.strip()
        for name in (p1, p2):
            if name not in VALID_PARAMS:
                raise ValueError(f'Unknown param {name!r}. Choose from: {VALID_PARAMS}')
        if p1 == p2:
            raise ValueError(f'Both parameters in a pair must differ, got {p1!r} twice')
        pairs.append((p1, p2))
    return pairs

# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_gt_val(param_name, gt_params, recomb_model):
    """Return the physical GT value for a named parameter."""
    rp = gt_params.recomb_params
    if param_name == 'velocity_cm_us':         return float(gt_params.velocity_cm_us)
    if param_name == 'lifetime_us':            return float(gt_params.lifetime_us)
    if param_name == 'diffusion_trans_cm2_us': return float(gt_params.diffusion_trans_cm2_us)
    if param_name == 'diffusion_long_cm2_us':  return float(gt_params.diffusion_long_cm2_us)
    if param_name == 'recomb_alpha':           return float(rp.alpha)
    if param_name == 'recomb_beta':
        if recomb_model != 'modified_box':
            raise ValueError(f'recomb_beta requires modified_box model')
        return float(rp.beta)
    if param_name == 'recomb_beta_90':
        if recomb_model != 'emb':
            raise ValueError(f'recomb_beta_90 requires emb model')
        return float(rp.beta_90)
    if param_name == 'recomb_R':
        if recomb_model != 'emb':
            raise ValueError(f'recomb_R requires emb model')
        return float(rp.R)
    raise ValueError(f'Unknown param {param_name!r}')


def _apply_param(param_name, value, sim_params):
    """Return sim_params with param_name set to physical value."""
    rp = sim_params.recomb_params
    if param_name == 'velocity_cm_us':
        return sim_params._replace(velocity_cm_us=value)
    if param_name == 'lifetime_us':
        return sim_params._replace(lifetime_us=value)
    if param_name == 'diffusion_trans_cm2_us':
        return sim_params._replace(diffusion_trans_cm2_us=value)
    if param_name == 'diffusion_long_cm2_us':
        return sim_params._replace(diffusion_long_cm2_us=value)
    if param_name == 'recomb_alpha':
        return sim_params._replace(recomb_params=rp._replace(alpha=value))
    if param_name == 'recomb_beta':
        return sim_params._replace(recomb_params=rp._replace(beta=value))
    if param_name == 'recomb_beta_90':
        return sim_params._replace(recomb_params=rp._replace(beta_90=value))
    if param_name == 'recomb_R':
        return sim_params._replace(recomb_params=rp._replace(R=value))
    raise ValueError(f'Unknown param {param_name!r}')


def make_2param_setter(param1, param2, gt_params, recomb_model):
    """Return (setter, gt_vals, scales, p_n_gts).

    setter(p_n_vec) -> SimParams  where p_n_vec is shape (2,)
      param1 = p_n_vec[0] * scales[0]
      param2 = p_n_vec[1] * scales[1]
    """
    scale1  = TYPICAL_SCALES[param1]
    scale2  = TYPICAL_SCALES[param2]
    gt_val1 = _get_gt_val(param1, gt_params, recomb_model)
    gt_val2 = _get_gt_val(param2, gt_params, recomb_model)

    def setter(p_n_vec):
        params = _apply_param(param1, p_n_vec[0] * scale1, gt_params)
        params = _apply_param(param2, p_n_vec[1] * scale2, params)
        return params

    gt_vals  = [gt_val1, gt_val2]
    scales   = [scale1,  scale2]
    p_n_gts  = [gt_val1 / scale1, gt_val2 / scale2]
    return setter, gt_vals, scales, p_n_gts

# ── Loss builder ───────────────────────────────────────────────────────────────

def build_loss_fn(loss_name, fwd_fn, gt_arrays, weights):
    """Return JIT-compiled (loss, grad) function of p_n_vec (shape (2,))."""
    if loss_name == 'sobolev_loss':
        def fn(p_n_vec):
            return sobolev_loss(fwd_fn(p_n_vec), gt_arrays, weights)
    elif loss_name == 'sobolev_loss_geomean_log1p':
        def fn(p_n_vec):
            return sobolev_loss_geomean_log1p(fwd_fn(p_n_vec), gt_arrays, weights)
    elif loss_name == 'mse_loss':
        def fn(p_n_vec):
            pred  = fwd_fn(p_n_vec)
            total = jnp.zeros(())
            for pr, gt in zip(pred, gt_arrays):
                norm  = jnp.sum(jnp.abs(gt)) + 1e-12
                total = total + jnp.mean(((pr - gt) / norm) ** 2)
            return total
    else:
        raise ValueError(f'Unknown loss {loss_name!r}')
    return jax.jit(jax.value_and_grad(fn))

# ── Optimizer factory ──────────────────────────────────────────────────────────

def make_optax_optimizer(optimizer_name, lr):
    if optimizer_name == 'adam':         return optax.adam(lr)
    if optimizer_name == 'sgd':          return optax.sgd(lr)
    if optimizer_name == 'momentum_sgd': return optax.sgd(lr, momentum=MOMENTUM)
    raise ValueError(f'Unknown optimizer {optimizer_name!r}')

# ── Single optimization trial ──────────────────────────────────────────────────

def run_trial(p0_pn_vec, val_and_grad_fn, optimizer, max_steps, tol=1e-5, patience=20):
    """Run one 2-parameter trial from starting p_n vector p0_pn_vec (shape (2,)).

    Early stops when the relative change in ||p_n|| over `patience` steps < tol.
    Returns dict with param_trajectory (steps+1, 2), loss_trajectory, etc.
    """
    p = jnp.array(p0_pn_vec, dtype=jnp.float32)
    opt_state = optimizer.init(p)

    param_traj = []
    loss_traj  = []

    t_start = time.time()

    lv, _ = val_and_grad_fn(p)
    jax.block_until_ready(lv)
    param_traj.append(p.tolist())
    loss_traj.append(float(lv))

    stopped_early = False
    for step in range(max_steps):
        lv, gv = val_and_grad_fn(p)
        jax.block_until_ready((lv, gv))
        updates, opt_state = optimizer.update(gv, opt_state)
        p = optax.apply_updates(p, updates)
        lv_new, _ = val_and_grad_fn(p)
        jax.block_until_ready(lv_new)
        param_traj.append(p.tolist())
        loss_traj.append(float(lv_new))

        if step >= patience:
            p_now  = np.array(param_traj[-1])
            p_prev = np.array(param_traj[-1 - patience])
            rel_change = np.linalg.norm(p_now - p_prev) / (np.linalg.norm(p_prev) + 1e-30)
            if rel_change < tol:
                stopped_early = True
                break

    return dict(
        param_trajectory = param_traj,       # list of [pn1, pn2] per step
        loss_trajectory  = loss_traj,
        total_time_s     = time.time() - t_start,
        stopped_early    = stopped_early,
        steps_run        = len(param_traj) - 1,
    )

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    pairs      = parse_pairs(args.pairs)
    loss_names = [l.strip() for l in args.loss.split(',')]
    for name in loss_names:
        if name not in VALID_LOSSES:
            raise ValueError(f'Unknown loss {name!r}')

    direction = tuple(float(x) for x in args.direction.split(','))
    if len(direction) != 3:
        raise ValueError(f'--direction must have 3 components')

    os.makedirs(args.results_dir, exist_ok=True)

    print(f'JAX devices : {jax.devices()}')
    print(f'Pairs       : {[(p1, p2) for p1, p2 in pairs]}')
    print(f'Optimizer   : {args.optimizer}  lr={args.lr}')
    print(f'Max steps   : {args.max_steps}  tol={args.tol}  patience={args.patience}')
    print(f'Losses      : {loss_names}')
    print(f'N           : {args.N}  ({(2*args.N+1)**2} starting points per pair)')
    print(f'M           : {args.M}  (trials per starting point)')
    print(f'Track name  : {args.track_name}  direction={direction}')
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

    # ── Track & deposits ──────────────────────────────────────────────────────
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

    gt_params = simulator.default_sim_params._replace(
        lifetime_us    = jnp.array(GT_LIFETIME_US),
        velocity_cm_us = jnp.array(GT_VELOCITY_CM_US),
    )

    # ── GT arrays & Sobolev weights ───────────────────────────────────────────
    print('Computing GT forward pass...')
    t0 = time.time()
    gt_arrays = simulator.forward(gt_params, deposits)
    jax.block_until_ready(gt_arrays)
    print(f'Done ({time.time() - t0:.1f} s)  —  {len(gt_arrays)} plane arrays')

    weights = tuple(
        make_sobolev_weight(arr.shape[0], arr.shape[1], max_pad=SOBOLEV_MAX_PAD)
        for arr in gt_arrays
    )

    # ── Loop over pairs ────────────────────────────────────────────────────────
    for param1, param2 in pairs:
        print(f'\n{"#" * 65}')
        print(f'  Pair: {param1}  +  {param2}')
        print(f'{"#" * 65}')

        setter, gt_vals, scales, p_n_gts = make_2param_setter(
            param1, param2, gt_params, simulator.recomb_model
        )
        p_n_gt1, p_n_gt2 = p_n_gts

        print(f'  {param1}: GT={gt_vals[0]:.6g}  scale={scales[0]:.6g}  p_n_gt={p_n_gt1:.6g}')
        print(f'  {param2}: GT={gt_vals[1]:.6g}  scale={scales[1]:.6g}  p_n_gt={p_n_gt2:.6g}')

        fwd_fn = jax.jit(lambda p_n_vec: simulator.forward(setter(p_n_vec), deposits))

        # ── Starting grid ─────────────────────────────────────────────────────
        left   = np.linspace(1.0 - RANGE_FRAC, 1.0, args.N + 1)[:-1]
        right  = np.linspace(1.0, 1.0 + RANGE_FRAC, args.N + 1)[1:]
        rel_fs = np.concatenate([left, [1.0], right])   # 2N+1 relative factors

        f1_grid, f2_grid = np.meshgrid(rel_fs, rel_fs, indexing='ij')
        factor_grid = list(zip(f1_grid.ravel().tolist(), f2_grid.ravel().tolist()))
        p_n_starts  = [(p_n_gt1 * f1, p_n_gt2 * f2) for f1, f2 in factor_grid]

        n_starts     = len(factor_grid)   # (2N+1)^2
        n_total_jobs = n_starts * args.M

        print(f'  Grid: {2*args.N+1}×{2*args.N+1} = {n_starts} starting points')
        print(f'  Total trials per loss: {n_total_jobs}')

        optimizer = make_optax_optimizer(args.optimizer, args.lr)

        # ── Loop over losses ──────────────────────────────────────────────────
        for loss_name in loss_names:
            print(f'\n  {"=" * 58}')
            print(f'  Loss: {loss_name}')

            val_and_grad_fn = build_loss_fn(loss_name, fwd_fn, gt_arrays, weights)

            print('  Compiling value_and_grad...')
            t0 = time.time()
            _p0 = jnp.array([p_n_starts[0][0], p_n_starts[0][1]], dtype=jnp.float32)
            _ = val_and_grad_fn(_p0); jax.block_until_ready(_)
            _ = val_and_grad_fn(_p0); jax.block_until_ready(_)
            print(f'  Done ({time.time() - t0:.1f} s)')

            all_trials  = []
            job_counter = 0

            for gi, ((f1, f2), (pn1, pn2)) in enumerate(zip(factor_grid, p_n_starts)):
                gt_marker = ' ← GT' if (f1 == 1.0 and f2 == 1.0) else ''
                print(f'\n    Grid [{gi+1}/{n_starts}]  '
                      f'factors=({f1:.4f},{f2:.4f})  '
                      f'p_n=({pn1:.4f},{pn2:.4f}){gt_marker}')

                point_trials = []
                for m in range(args.M):
                    job_counter += 1
                    print(f'      Trial {m+1}/{args.M} '
                          f'(overall {job_counter}/{n_total_jobs}) ...',
                          end='', flush=True)

                    trial = run_trial(
                        [pn1, pn2], val_and_grad_fn, optimizer,
                        args.max_steps, tol=args.tol, patience=args.patience,
                    )
                    point_trials.append(trial)

                    final_pn  = trial['param_trajectory'][-1]
                    early_tag = f'  [early@{trial["steps_run"]}]' if trial['stopped_early'] else ''
                    print(f'  loss {trial["loss_trajectory"][0]:.3e} → '
                          f'{trial["loss_trajectory"][-1]:.3e}  '
                          f'p_n ({pn1:.3f},{pn2:.3f}) → ({final_pn[0]:.3f},{final_pn[1]:.3f})  '
                          f'({trial["total_time_s"]:.1f} s){early_tag}')

                all_trials.append(point_trials)

            pair_tag = f'{param1}+{param2}'
            result = dict(
                param_names           = [param1, param2],
                param_gts             = gt_vals,
                scales                = scales,
                p_n_gts               = p_n_gts,
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
                factor_grid           = factor_grid,
                starting_p_n_values   = p_n_starts,
                trials                = all_trials,
            )

            pkl_name = (f'{loss_name}_N{args.N}_M{args.M}_{args.optimizer}_'
                        f'{pair_tag}_{args.track_name}.pkl')
            pkl_path = os.path.join(args.results_dir, pkl_name)
            with open(pkl_path, 'wb') as f:
                pickle.dump(result, f)
            print(f'\n  Saved: {pkl_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
