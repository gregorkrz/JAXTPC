#!/usr/bin/env python
"""
Two-parameter gradient-based optimization with N random starting points
sampled uniformly from [0.95, 1.05] × [0.95, 1.05] relative to the GT.

For each random starting point, runs a gradient-based optimizer and records
the full parameter + loss trajectory at every step.

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

Multi-track optimization
------------------------
  Use --tracks to optimize over multiple tracks simultaneously.  The loss at
  each step is the sum of per-track losses.  Tracks are run sequentially in
  the forward pass (no vmap).

  Format: name:dx,dy,dz:momentum_mev  (comma-separated list of specs)
  Example:
    --tracks diagonal:1,1,1:1000,track2_100MeV:0.5,1.05,0.2:100

  --track-name / --direction / --momentum are still accepted for single-track
  runs and are ignored when --tracks is given.

Output (one file per pair × loss)
----------------------------------
    results/2d_opt/{loss}_N{N}_{optimizer}_{param1}+{param2}_{track_tag}.pkl

Each pickle contains:
    param_names, param_gts, scales, p_n_gts,
    optimizer, lr, max_steps, tol, patience, N,
    loss_name, track_name, tracks,
    factor_grid,          # list of (f1, f2) relative-to-GT factor pairs (random)
    starting_p_n_values,  # list of (pn1, pn2) absolute p_n starting values
    trials,               # list of N dicts:
        param_trajectory  (max_steps+1, 2) — or shorter if early stopped
        loss_trajectory   (max_steps+1,)
        total_time_s, stopped_early, steps_run
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

_JAX_CACHE_DIR = os.environ.get('JAX_COMPILATION_CACHE_DIR',
                                os.path.expanduser('~/.cache/jax_compilation_cache'))
jax.config.update('jax_compilation_cache_dir', _JAX_CACHE_DIR)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)
_RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')
_PLOTS_DIR   = os.environ.get('PLOTS_DIR',   'plots')

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
                   help='Peak learning rate (default: 0.01)')
    p.add_argument('--lr-schedule', default='constant',
                   choices=('constant', 'cosine'),
                   help='LR schedule: constant or cosine decay to 0 (default: constant)')
    p.add_argument('--max-steps', type=int, default=100,
                   help='Max optimization steps per trial (default: 100)')
    p.add_argument('--tol', type=float, default=1e-5,
                   help='Early-stop relative tolerance on p_n norm (default: 1e-5)')
    p.add_argument('--patience', type=int, default=20,
                   help='Steps over which relative change is checked (default: 20)')
    p.add_argument('--loss',
                   default='sobolev_loss,sobolev_loss_geomean_log1p',
                   help='Comma-separated list of losses (default: both Sobolev)')
    p.add_argument('--N', type=int, default=25,
                   help='Number of random trials (default: 25); each trial '
                        'draws both parameters uniformly from [0.95, 1.05] × GT')
    p.add_argument('--results-dir', default=os.path.join(_RESULTS_DIR, '2d_opt'),
                   help='Output directory (default: results/2d_opt)')
    p.add_argument('--track-name', default='diagonal',
                   help='Label for the track direction (default: diagonal)')
    p.add_argument('--direction', default='1,1,1',
                   help='Muon direction as x,y,z (default: 1,1,1)')
    p.add_argument('--momentum', type=float, default=1000.0,
                   help='Muon kinetic energy in MeV (default: 1000.0)')
    p.add_argument('--tracks', default=None,
                   help='Multi-track spec: name:dx,dy,dz:mom_mev comma-separated, '
                        'e.g. diagonal:1,1,1:1000,track2_100MeV:0.5,1.05,0.2:100. '
                        'Overrides --track-name/--direction/--momentum.')
    return p.parse_args()


def parse_tracks(tracks_str, fallback_name, fallback_dir, fallback_mom):
    """Parse --tracks spec or fall back to single-track args.

    Returns list of dicts with keys: name, direction (tuple), momentum_mev.
    """
    if tracks_str is None:
        return [dict(name=fallback_name, direction=fallback_dir,
                     momentum_mev=fallback_mom)]
    specs = []
    for item in tracks_str.split(','):
        item = item.strip()
        parts = item.split(':')
        if len(parts) != 3:
            raise ValueError(
                f'Each --tracks entry must be name:dx,dy,dz:momentum_mev, got {item!r}')
        name = parts[0].strip()
        try:
            direction = tuple(float(x) for x in parts[1].split(','))
        except ValueError:
            raise ValueError(f'Bad direction in --tracks entry {item!r}')
        if len(direction) != 3:
            raise ValueError(f'Direction must have 3 components in {item!r}')
        try:
            momentum_mev = float(parts[2])
        except ValueError:
            raise ValueError(f'Bad momentum in --tracks entry {item!r}')
        specs.append(dict(name=name, direction=direction, momentum_mev=momentum_mev))
    if not specs:
        raise ValueError('--tracks produced no entries')
    return specs


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

def make_optax_optimizer(optimizer_name, lr, lr_schedule, max_steps):
    if lr_schedule == 'cosine':
        schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=max_steps)
    else:
        schedule = lr

    if optimizer_name == 'adam':         return optax.adam(schedule)
    if optimizer_name == 'sgd':          return optax.sgd(schedule)
    if optimizer_name == 'momentum_sgd': return optax.sgd(schedule, momentum=MOMENTUM)
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

    fallback_dir = tuple(float(x) for x in args.direction.split(','))
    if len(fallback_dir) != 3:
        raise ValueError('--direction must have 3 components')
    track_specs = parse_tracks(args.tracks, args.track_name, fallback_dir, args.momentum)
    track_tag   = '+'.join(t['name'] for t in track_specs)

    os.makedirs(args.results_dir, exist_ok=True)

    print(f'JAX devices : {jax.devices()}')
    print(f'Pairs       : {[(p1, p2) for p1, p2 in pairs]}')
    print(f'Optimizer   : {args.optimizer}  lr={args.lr}')
    print(f'Max steps   : {args.max_steps}  tol={args.tol}  patience={args.patience}')
    print(f'Losses      : {loss_names}')
    print(f'N           : {args.N}  (random trials per pair, start range [0.95, 1.05])')
    print(f'Tracks      : {[t["name"] for t in track_specs]}')
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

    # ── Tracks & deposits ─────────────────────────────────────────────────────
    all_deposits = []
    for ts in track_specs:
        print(f'Generating muon track  name={ts["name"]}  '
              f'direction={ts["direction"]}  T={ts["momentum_mev"]} MeV...')
        track = generate_muon_track(
            start_position_mm=(0.0, 0.0, 0.0),
            direction=ts['direction'],
            kinetic_energy_mev=ts['momentum_mev'],
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
        print(f'  {n_total:,} deposits')
        all_deposits.append(deposits)

    print('Warming up JIT...')
    t0 = time.time()
    simulator.warm_up()
    print(f'Done ({time.time() - t0:.1f} s)')

    gt_params = simulator.default_sim_params._replace(
        lifetime_us    = jnp.array(GT_LIFETIME_US),
        velocity_cm_us = jnp.array(GT_VELOCITY_CM_US),
    )

    # ── GT arrays & Sobolev weights (concatenated across all tracks) ──────────
    print('Computing GT forward passes...')
    t0 = time.time()
    gt_arrays = []
    for deposits in all_deposits:
        arrs = simulator.forward(gt_params, deposits)
        jax.block_until_ready(arrs)
        gt_arrays.extend(arrs)
    gt_arrays = tuple(gt_arrays)
    print(f'Done ({time.time() - t0:.1f} s)  —  {len(gt_arrays)} plane arrays total')

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

        def _multi_fwd(p_n_vec, _deps=all_deposits, _setter=setter):
            arrays = []
            for dep in _deps:
                arrays.extend(simulator.forward(_setter(p_n_vec), dep))
            return tuple(arrays)
        fwd_fn = jax.jit(_multi_fwd)

        # ── Random starting points ────────────────────────────────────────────
        rng = np.random.default_rng()
        f1_vals = rng.uniform(1.0 - RANGE_FRAC, 1.0 + RANGE_FRAC, args.N)
        f2_vals = rng.uniform(1.0 - RANGE_FRAC, 1.0 + RANGE_FRAC, args.N)
        factor_grid = list(zip(f1_vals.tolist(), f2_vals.tolist()))
        p_n_starts  = [(p_n_gt1 * f1, p_n_gt2 * f2) for f1, f2 in factor_grid]

        print(f'  Random trials: {args.N}  (factors drawn from [0.95, 1.05])')

        optimizer = make_optax_optimizer(args.optimizer, args.lr, args.lr_schedule, args.max_steps)

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

            all_trials = []

            for gi, ((f1, f2), (pn1, pn2)) in enumerate(zip(factor_grid, p_n_starts)):
                print(f'\n    Trial [{gi+1}/{args.N}]  '
                      f'factors=({f1:.4f},{f2:.4f})  '
                      f'p_n=({pn1:.4f},{pn2:.4f})',
                      end='', flush=True)

                trial = run_trial(
                    [pn1, pn2], val_and_grad_fn, optimizer,
                    args.max_steps, tol=args.tol, patience=args.patience,
                )
                all_trials.append(trial)

                final_pn  = trial['param_trajectory'][-1]
                early_tag = f'  [early@{trial["steps_run"]}]' if trial['stopped_early'] else ''
                print(f'  loss {trial["loss_trajectory"][0]:.3e} → '
                      f'{trial["loss_trajectory"][-1]:.3e}  '
                      f'p_n ({pn1:.3f},{pn2:.3f}) → ({final_pn[0]:.3f},{final_pn[1]:.3f})  '
                      f'({trial["total_time_s"]:.1f} s){early_tag}')

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
                lr_schedule           = args.lr_schedule,
                loss_name             = loss_name,
                track_name            = track_tag,
                tracks                = track_specs,
                factor_grid           = factor_grid,
                starting_p_n_values   = p_n_starts,
                trials                = all_trials,
            )

            sched_tag = f'_cosine' if args.lr_schedule == 'cosine' else ''
            pkl_name = (f'{loss_name}_N{args.N}_{args.optimizer}_lr{args.lr}{sched_tag}_'
                        f'{pair_tag}_{track_tag}.pkl')
            pkl_path = os.path.join(args.results_dir, pkl_name)
            with open(pkl_path, 'wb') as f:
                pickle.dump(result, f)
            print(f'\n  Saved: {pkl_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
