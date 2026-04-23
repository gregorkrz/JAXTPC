#!/usr/bin/env python
"""
Gradient-based optimization over any number of parameters, N random starting points.

Each run targets one (params, tracks, loss, optimizer) configuration and saves all
trial data to a single pkl under a generated folder name.

Usage
-----
  python run_optimization.py \\
    --params recomb_alpha,recomb_beta_90 \\
    --range 0.98 1.02 \\
    --tracks diagonal,Z \\
    --loss sobolev_loss_geomean_log1p \\
    --optimizer adam --lr 0.01 \\
    --max-steps 200 --lr-schedule cosine \\
    --N 25

Named track presets
-------------------
  diagonal  (1,1,1)        1000 MeV
  X         (1,0,0)        1000 MeV
  Y         (0,1,0)        1000 MeV
  Z         (0,0,1)        1000 MeV
  U         (0,0.866,0.5)  1000 MeV
  V         (0,-0.866,0.5) 1000 MeV
  track2    (0.5,1.05,0.2)  200 MeV

  Custom tracks: name:dx,dy,dz:momentum_mev  (mixed with presets is fine)
  e.g. --tracks diagonal+mytrack:0.1,0.2,0.9:500

Output
------
  results/<folder>/result_0.pkl   (next available index)

  <folder> encodes: params, tracks, loss, optimizer, lr, schedule, max_steps, N, range.

Each pickle contains:
  param_names, param_gts, scales, p_n_gts,
  optimizer, lr, lr_schedule, max_steps, tol, patience, N,
  loss_name, tracks,
  range_lo, range_hi,
  factor_grid,           list of [f1, ...] per trial
  starting_p_n_values,   list of [pn1, ...] per trial
  trials,                list of N dicts:
      param_trajectory   list of length steps+1, each entry [pn1, ...]
      grad_trajectory    list of length steps+1, each entry [g1, ...]
      loss_trajectory    list of length steps+1
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

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

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
    mse_loss,
    l1_loss,
)
from tools.noise import generate_noise
from tools.particle_generator import generate_muon_track
from tools.simulation import DetectorSimulator

# ── Constants ──────────────────────────────────────────────────────────────────
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

VALID_LOSSES     = ('sobolev_loss', 'sobolev_loss_geomean_log1p', 'mse_loss', 'l1_loss')
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

# Named track presets: name → (direction_xyz_tuple, momentum_mev)
TRACK_PRESETS = {
    'diagonal': ((1.0,  1.0,  1.0),  1000.0),
    'X':        ((1.0,  0.0,  0.0),  1000.0),
    'Y':        ((0.0,  1.0,  0.0),  1000.0),
    'Z':        ((0.0,  0.0,  1.0),  1000.0),
    'U':        ((0.0,  0.866, 0.5), 1000.0),
    'V':        ((0.0, -0.866, 0.5), 1000.0),
    'track2':   ((0.5,  1.05, 0.2),   200.0),
}

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS   = 1e-8
MOMENTUM   = 0.9

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--params', required=True,
                   help='Comma-separated params to optimize, e.g. recomb_alpha,recomb_beta_90')
    p.add_argument('--range', type=float, nargs=2, metavar=('LO', 'HI'), default=(0.95, 1.05),
                   help='Relative factor range for random starting points (default: 0.95 1.05)')
    p.add_argument('--tracks', default='diagonal',
                   help='"+"-separated track presets or name:dx,dy,dz:mom_mev specs '
                        '("+" separates tracks, "," separates direction components). '
                        f'Default: diagonal.  Presets: {", ".join(TRACK_PRESETS)}')
    p.add_argument('--loss', default='sobolev_loss_geomean_log1p', choices=VALID_LOSSES,
                   help='Loss function (default: sobolev_loss_geomean_log1p)')
    p.add_argument('--optimizer', default='adam', choices=VALID_OPTIMIZERS,
                   help='Optimizer (default: adam)')
    p.add_argument('--lr', type=float, default=0.01,
                   help='Peak learning rate (default: 0.01)')
    p.add_argument('--lr-schedule', default='constant', choices=('constant', 'cosine'),
                   help='LR schedule: constant or cosine decay over --max-steps (default: constant)')
    p.add_argument('--max-steps', type=int, default=200,
                   help='Max gradient steps per trial (default: 200)')
    p.add_argument('--tol', type=float, default=1e-5,
                   help='Early-stop relative tolerance on p_n norm (default: 1e-5)')
    p.add_argument('--patience', type=int, default=20,
                   help='Steps over which relative change is checked (default: 20)')
    p.add_argument('--N', type=int, default=25,
                   help='Number of random trials (default: 25)')
    p.add_argument('--results-base', default=_RESULTS_DIR,
                   help='Base directory; output goes to <results-base>/<folder>/ (default: $RESULTS_DIR or results)')
    p.add_argument('--seed', type=int, default=None,
                   help='Master RNG seed (default: random). Seeds everything: '
                        'trial starting points and GT noise draw. '
                        'The resolved seed is printed and stored in the pkl.')
    p.add_argument('--noise-scale', type=float, default=0.0,
                   help='Noise amplitude as a multiple of the calibrated detector noise '
                        '(MicroBooNE model, converted to signal units via electrons_per_adc). '
                        '0.0 = no noise (default), 1.0 = realistic noise. '
                        'Signal and noise RMS are printed at startup for reference.')
    p.add_argument('--no-wandb', action='store_true',
                   help='Disable Weights & Biases logging (enabled by default).')
    p.add_argument('--wandb-project', default='jaxtpc-optimization',
                   help='W&B project name (default: jaxtpc-optimization).')
    p.add_argument('--log-interval', type=int, default=50,
                   help='Log to W&B every this many steps (default: 50).')
    return p.parse_args()


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_params(params_str):
    names = [n.strip() for n in params_str.split(',') if n.strip()]
    if not names:
        raise ValueError('--params is empty')
    for name in names:
        if name not in VALID_PARAMS:
            raise ValueError(f'Unknown param {name!r}. Choose from: {VALID_PARAMS}')
    if len(names) != len(set(names)):
        raise ValueError('Duplicate param names in --params')
    return names


def parse_tracks(tracks_str):
    """Parse '+'-separated preset names or name:dx,dy,dz:mom_mev specs.

    '+' separates tracks; ',' is used only inside direction components.
    Mixed input is supported, e.g. 'diagonal+mytrack:0.1,0.2,0.9:500'.
    Returns list of dicts: {name, direction (tuple), momentum_mev}.
    """
    specs = []
    for item in tracks_str.split('+'):
        item = item.strip()
        if not item:
            continue
        if ':' in item:
            # Full spec: name:dx,dy,dz:momentum_mev
            parts = item.split(':')
            if len(parts) != 3:
                raise ValueError(
                    f'Custom track must be name:dx,dy,dz:momentum_mev, got {item!r}')
            name = parts[0].strip()
            try:
                direction = tuple(float(x) for x in parts[1].split(','))
            except ValueError:
                raise ValueError(f'Bad direction in track spec {item!r}')
            if len(direction) != 3:
                raise ValueError(f'Direction must have 3 components in {item!r}')
            try:
                momentum_mev = float(parts[2])
            except ValueError:
                raise ValueError(f'Bad momentum in track spec {item!r}')
            specs.append(dict(name=name, direction=direction, momentum_mev=momentum_mev))
        else:
            # Preset name
            if item not in TRACK_PRESETS:
                raise ValueError(
                    f'Unknown track preset {item!r}. '
                    f'Known: {list(TRACK_PRESETS)}. '
                    f'Use name:dx,dy,dz:mom_mev for custom tracks.')
            direction, momentum_mev = TRACK_PRESETS[item]
            specs.append(dict(name=item, direction=direction, momentum_mev=momentum_mev))
    if not specs:
        raise ValueError('--tracks produced no entries')
    return specs


# ── Folder name ────────────────────────────────────────────────────────────────

def make_folder_name(param_names, track_specs, loss_name, optimizer, lr,
                     lr_schedule, max_steps, N, range_lo, range_hi, noise_scale=0.0):
    params_tag    = '+'.join(param_names)
    tracks_tag    = '+'.join(t['name'] for t in track_specs)
    sched_tag     = '_cosine' if lr_schedule == 'cosine' else ''
    range_tag     = f'r{range_lo:.3g}_{range_hi:.3g}'.replace('.', 'p')
    noise_tag     = f'_noise{noise_scale:.3g}'.replace('.', 'p') if noise_scale > 0.0 else ''
    return (f'{params_tag}__{tracks_tag}__{loss_name}__'
            f'{optimizer}_lr{lr}{sched_tag}_s{max_steps}_N{N}_{range_tag}{noise_tag}')


def next_result_path(folder, seed=None):
    """Return the output pkl path inside folder.

    If seed is given: results/<folder>/result_<seed>.pkl (fixed, deterministic).
    If seed is None:  results/<folder>/result_0.pkl, result_1.pkl, ... (first unused).
    """
    os.makedirs(folder, exist_ok=True)
    if seed is not None:
        return os.path.join(folder, f'result_{seed}.pkl')
    i = 0
    while True:
        path = os.path.join(folder, f'result_{i}.pkl')
        if not os.path.exists(path):
            return path
        i += 1


# ── Physics helpers (mirrors 2d_opt.py) ───────────────────────────────────────

def _get_gt_val(param_name, gt_params, recomb_model):
    rp = gt_params.recomb_params
    if param_name == 'velocity_cm_us':         return float(gt_params.velocity_cm_us)
    if param_name == 'lifetime_us':            return float(gt_params.lifetime_us)
    if param_name == 'diffusion_trans_cm2_us': return float(gt_params.diffusion_trans_cm2_us)
    if param_name == 'diffusion_long_cm2_us':  return float(gt_params.diffusion_long_cm2_us)
    if param_name == 'recomb_alpha':           return float(rp.alpha)
    if param_name == 'recomb_beta':
        if recomb_model != 'modified_box':
            raise ValueError('recomb_beta requires modified_box model')
        return float(rp.beta)
    if param_name == 'recomb_beta_90':
        if recomb_model != 'emb':
            raise ValueError('recomb_beta_90 requires emb model')
        return float(rp.beta_90)
    if param_name == 'recomb_R':
        if recomb_model != 'emb':
            raise ValueError('recomb_R requires emb model')
        return float(rp.R)
    raise ValueError(f'Unknown param {param_name!r}')


def _apply_param(param_name, value, sim_params):
    rp = sim_params.recomb_params
    if param_name == 'velocity_cm_us':         return sim_params._replace(velocity_cm_us=value)
    if param_name == 'lifetime_us':            return sim_params._replace(lifetime_us=value)
    if param_name == 'diffusion_trans_cm2_us': return sim_params._replace(diffusion_trans_cm2_us=value)
    if param_name == 'diffusion_long_cm2_us':  return sim_params._replace(diffusion_long_cm2_us=value)
    if param_name == 'recomb_alpha':           return sim_params._replace(recomb_params=rp._replace(alpha=value))
    if param_name == 'recomb_beta':            return sim_params._replace(recomb_params=rp._replace(beta=value))
    if param_name == 'recomb_beta_90':         return sim_params._replace(recomb_params=rp._replace(beta_90=value))
    if param_name == 'recomb_R':               return sim_params._replace(recomb_params=rp._replace(R=value))
    raise ValueError(f'Unknown param {param_name!r}')


def make_nparam_setter(param_names, gt_params, recomb_model):
    """Return (setter, gt_vals, scales, p_n_gts) for any number of params.

    setter(p_n_vec) -> SimParams  where p_n_vec[i] = physical_val[i] / scales[i]
    """
    scales  = [TYPICAL_SCALES[n] for n in param_names]
    gt_vals = [_get_gt_val(n, gt_params, recomb_model) for n in param_names]
    p_n_gts = [v / s for v, s in zip(gt_vals, scales)]

    def setter(p_n_vec):
        params = gt_params
        for i, name in enumerate(param_names):
            params = _apply_param(name, p_n_vec[i] * scales[i], params)
        return params

    return setter, gt_vals, scales, p_n_gts


# ── Loss builder ───────────────────────────────────────────────────────────────

def build_loss_fn(loss_name, fwd_fn, gt_arrays, weights):
    """Return JIT-compiled (loss, grad) function of p_n_vec."""
    if loss_name == 'sobolev_loss':
        def fn(p_n_vec): return sobolev_loss(fwd_fn(p_n_vec), gt_arrays, weights)
    elif loss_name == 'sobolev_loss_geomean_log1p':
        def fn(p_n_vec): return sobolev_loss_geomean_log1p(fwd_fn(p_n_vec), gt_arrays, weights)
    elif loss_name == 'mse_loss':
        def fn(p_n_vec): return mse_loss(fwd_fn(p_n_vec), gt_arrays)
    elif loss_name == 'l1_loss':
        def fn(p_n_vec): return l1_loss(fwd_fn(p_n_vec), gt_arrays)
    else:
        raise ValueError(f'Unknown loss {loss_name!r}')
    return jax.jit(jax.value_and_grad(fn))


# ── Optimizer factory ──────────────────────────────────────────────────────────

def make_optax_optimizer(optimizer_name, lr, lr_schedule, max_steps):
    if lr_schedule == 'cosine':
        schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=max_steps)
    else:
        schedule = lr
    if optimizer_name == 'adam':         return optax.adam(schedule, b1=ADAM_BETA1, b2=ADAM_BETA2, eps=ADAM_EPS)
    if optimizer_name == 'sgd':          return optax.sgd(schedule)
    if optimizer_name == 'momentum_sgd': return optax.sgd(schedule, momentum=MOMENTUM)
    raise ValueError(f'Unknown optimizer {optimizer_name!r}')


# ── Trial runner ───────────────────────────────────────────────────────────────

def run_trial(p0, val_and_grad_fn, optimizer, max_steps, tol=1e-5, patience=20,
              log_interval=50, param_names=None, scales=None, p_n_gts=None,
              use_wandb=False, trial_idx=0):
    """Run one optimization trial from starting p_n vector p0 (any length).

    Records param, loss, and gradient at every step.

    Returns dict with:
      param_trajectory   list[list]  length steps+1
      grad_trajectory    list[list]  length steps+1  (grad at each recorded position)
      loss_trajectory    list[float] length steps+1
      total_time_s, stopped_early, steps_run
    """
    p = jnp.array(p0, dtype=jnp.float32)
    opt_state = optimizer.init(p)

    param_traj = []
    loss_traj  = []
    grad_traj  = []

    t_start    = time.time()
    step_start = time.time()

    lv, gv = val_and_grad_fn(p)
    jax.block_until_ready((lv, gv))
    param_traj.append(p.tolist())
    loss_traj.append(float(lv))
    grad_traj.append(gv.tolist())

    if use_wandb and _WANDB_AVAILABLE:
        _wandb_log_step(0, float(lv), gv, p, param_names, scales, p_n_gts,
                        step_time_s=0.0, trial_idx=trial_idx)

    stopped_early = False
    for step in range(max_steps):
        step_start = time.time()
        lv, gv = val_and_grad_fn(p)
        jax.block_until_ready((lv, gv))
        updates, opt_state = optimizer.update(gv, opt_state)
        p = optax.apply_updates(p, updates)
        lv_new, gv_new = val_and_grad_fn(p)
        jax.block_until_ready((lv_new, gv_new))
        step_time = time.time() - step_start

        param_traj.append(p.tolist())
        loss_traj.append(float(lv_new))
        grad_traj.append(gv_new.tolist())

        if use_wandb and _WANDB_AVAILABLE and (step + 1) % log_interval == 0:
            _wandb_log_step(step + 1, float(lv_new), gv_new, p,
                            param_names, scales, p_n_gts,
                            step_time_s=step_time, trial_idx=trial_idx)

        if step >= patience:
            p_now  = np.array(param_traj[-1])
            p_prev = np.array(param_traj[-1 - patience])
            rel = np.linalg.norm(p_now - p_prev) / (np.linalg.norm(p_prev) + 1e-30)
            if rel < tol:
                stopped_early = True
                break

    return dict(
        param_trajectory = param_traj,
        grad_trajectory  = grad_traj,
        loss_trajectory  = loss_traj,
        total_time_s     = time.time() - t_start,
        stopped_early    = stopped_early,
        steps_run        = len(param_traj) - 1,
    )


def _wandb_log_step(step, loss, gv, p, param_names, scales, p_n_gts,
                    step_time_s, trial_idx):
    """Log one step to W&B."""
    p_np  = np.array(p)
    gv_np = np.array(gv)

    log = {
        'trial':          trial_idx,
        'loss':           loss,
        'grad_norm':      float(np.linalg.norm(gv_np)),
        'param_norm':     float(np.linalg.norm(p_np)),
        'step_time_s':    step_time_s,
    }

    if param_names is not None:
        for i, name in enumerate(param_names):
            log[f'params/{name}_normalized'] = float(p_np[i])
            if scales is not None:
                log[f'params/{name}_physical'] = float(p_np[i] * scales[i])
            if p_n_gts is not None:
                rel_err = abs(float(p_np[i]) - p_n_gts[i]) / (abs(p_n_gts[i]) + 1e-30)
                log[f'params/{name}_rel_err'] = rel_err
            if gv_np is not None:
                log[f'grads/{name}'] = float(gv_np[i])

    _wandb.log(log, step=step)


# ── Noise ─────────────────────────────────────────────────────────────────────

def apply_noise_to_gt(gt_arrays, simulator, noise_scale, noise_seed):
    """Add calibrated detector noise to GT arrays once before optimization.

    Uses generate_noise() (MicroBooNE model: series + white components) and
    converts from ADC to signal units via config.electrons_per_adc.
    noise_scale=1.0 gives the realistic detector noise amplitude.
    The same draw is used for all trials so they all optimise against the
    same fixed noisy target.
    """
    cfg = simulator.config
    noise_dict   = generate_noise(cfg, key=jax.random.PRNGKey(noise_seed))
    n_readouts = cfg.volumes[0].n_planes if simulator._readout_type == 'wire' else 1
    noisy = []
    for v in range(cfg.n_volumes):
        for p in range(n_readouts):
            gt = gt_arrays[v * n_readouts + p]
            noise = noise_dict[(v, p)] * noise_scale
            if noise.shape[0] < gt.shape[0]:
                noise = jnp.pad(noise, ((0, gt.shape[0] - noise.shape[0]), (0, 0)))
            noisy.append(gt + noise)
    return tuple(noisy)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    param_names = parse_params(args.params)
    track_specs = parse_tracks(args.tracks)
    range_lo, range_hi = getattr(args, 'range')

    # ── Seeding ───────────────────────────────────────────────────────────────
    # SeedSequence(None) draws entropy from the OS; spawn gives independent
    # child sequences so starting-point and noise draws never collide.
    ss = np.random.SeedSequence(args.seed)
    effective_seed   = int(ss.entropy) if args.seed is None else args.seed
    start_ss, noise_ss = ss.spawn(2)
    start_rng        = np.random.default_rng(start_ss)
    noise_seed       = int(noise_ss.generate_state(1)[0])

    folder_name = make_folder_name(
        param_names, track_specs, args.loss, args.optimizer,
        args.lr, args.lr_schedule, args.max_steps, args.N, range_lo, range_hi,
        noise_scale=args.noise_scale,
    )
    output_dir  = os.path.join(args.results_base, folder_name)
    output_path = next_result_path(output_dir, seed=args.seed)

    if args.seed is not None and os.path.exists(output_path):
        print(f'Skipping: {output_path} already exists.')
        return

    use_wandb = (not args.no_wandb) and _WANDB_AVAILABLE
    if not args.no_wandb and not _WANDB_AVAILABLE:
        print('Warning: wandb not installed — logging disabled. pip install wandb to enable.')

    print(f'JAX devices  : {jax.devices()}')
    print(f'Params       : {param_names}')
    print(f'Range        : [{range_lo}, {range_hi}] × GT')
    print(f'Tracks       : {[t["name"] for t in track_specs]}')
    print(f'Loss         : {args.loss}')
    print(f'Optimizer    : {args.optimizer}  lr={args.lr}  schedule={args.lr_schedule}')
    print(f'Max steps    : {args.max_steps}  tol={args.tol}  patience={args.patience}')
    print(f'N            : {args.N}')
    print(f'Seed         : {effective_seed}')
    print(f'Noise scale  : {args.noise_scale}')
    print(f'W&B          : {"enabled  project=" + args.wandb_project if use_wandb else "disabled"}')
    print(f'Log interval : {args.log_interval} steps')
    print(f'Output       : {output_path}')

    # ── W&B init ──────────────────────────────────────────────────────────────
    if use_wandb:
        _wandb.init(
            project=args.wandb_project,
            name=folder_name,
            config=dict(
                param_names   = param_names,
                tracks        = [t['name'] for t in track_specs],
                loss          = args.loss,
                optimizer     = args.optimizer,
                lr            = args.lr,
                lr_schedule   = args.lr_schedule,
                max_steps     = args.max_steps,
                tol           = args.tol,
                patience      = args.patience,
                N             = args.N,
                range_lo      = range_lo,
                range_hi      = range_hi,
                noise_scale   = args.noise_scale,
                seed          = effective_seed,
                log_interval  = args.log_interval,
            ),
        )

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
    )

    # ── Tracks & deposits ─────────────────────────────────────────────────────
    all_deposits = []
    for ts in track_specs:
        print(f'Generating track  name={ts["name"]}  '
              f'direction={ts["direction"]}  T={ts["momentum_mev"]} MeV...')
        track = generate_muon_track(
            start_position_mm=(0.0, 0.0, 0.0),
            direction=ts['direction'],
            kinetic_energy_mev=ts['momentum_mev'],
            step_size_mm=0.1,
            track_id=1,
            #detector_bounds_mm=DETECTOR_BOUNDS_MM,
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

    # ── GT arrays & Sobolev weights ───────────────────────────────────────────
    print('Computing GT forward passes...')
    t0 = time.time()
    gt_arrays = []
    for deposits in all_deposits:
        arrs = simulator.forward(gt_params, deposits)
        jax.block_until_ready(arrs)
        gt_arrays.extend(arrs)
    gt_arrays = tuple(gt_arrays)
    print(f'Done ({time.time() - t0:.1f} s)  —  {len(gt_arrays)} plane arrays total')

    # ── Optional noise on GT ──────────────────────────────────────────────────
    signal_rms = float(np.mean([float(jnp.std(a)) for a in gt_arrays]))
    if args.noise_scale > 0.0:
        noisy_gt_arrays = apply_noise_to_gt(gt_arrays, simulator, args.noise_scale, noise_seed)
        noise_rms = float(np.mean([float(jnp.std(n - c))
                                   for n, c in zip(noisy_gt_arrays, gt_arrays)]))
        gt_arrays = noisy_gt_arrays
        print(f'  Signal RMS : {signal_rms:.4g}  (mean over planes)')
        print(f'  Noise  RMS : {noise_rms:.4g}  (mean over planes,  SNR ≈ {signal_rms / max(noise_rms, 1e-30):.2f})')
    else:
        print(f'  Signal RMS : {signal_rms:.4g}  (mean over planes, use --noise-scale to add noise)')

    weights = tuple(
        make_sobolev_weight(arr.shape[0], arr.shape[1], max_pad=SOBOLEV_MAX_PAD)
        for arr in gt_arrays
    )

    # ── Setter & forward ──────────────────────────────────────────────────────
    setter, gt_vals, scales, p_n_gts = make_nparam_setter(
        param_names, gt_params, simulator.recomb_model)

    for name, gt_val, scale, p_n_gt in zip(param_names, gt_vals, scales, p_n_gts):
        print(f'  {name}: GT={gt_val:.6g}  scale={scale:.6g}  p_n_gt={p_n_gt:.6g}')

    def _multi_fwd(p_n_vec, _deps=all_deposits, _setter=setter):
        arrays = []
        for dep in _deps:
            arrays.extend(simulator.forward(_setter(p_n_vec), dep))
        return tuple(arrays)
    fwd_fn = jax.jit(_multi_fwd)

    # ── Random starting points ────────────────────────────────────────────────
    rng = start_rng
    n_params = len(param_names)
    factors = rng.uniform(range_lo, range_hi, size=(args.N, n_params))   # (N, n_params)
    factor_grid   = factors.tolist()
    p_n_starts    = (factors * np.array(p_n_gts)).tolist()

    # ── Compile ───────────────────────────────────────────────────────────────
    val_and_grad_fn = build_loss_fn(args.loss, fwd_fn, gt_arrays, weights)
    print('\nCompiling value_and_grad...')
    t0 = time.time()
    _p0 = jnp.array(p_n_starts[0], dtype=jnp.float32)
    _ = val_and_grad_fn(_p0); jax.block_until_ready(_)
    _ = val_and_grad_fn(_p0); jax.block_until_ready(_)
    print(f'Done ({time.time() - t0:.1f} s)')

    optimizer = make_optax_optimizer(args.optimizer, args.lr, args.lr_schedule, args.max_steps)

    # ── Trials ────────────────────────────────────────────────────────────────
    all_trials = []
    for gi, (factors_i, pn_start) in enumerate(zip(factor_grid, p_n_starts)):
        factors_str = ', '.join(f'{f:.4f}' for f in factors_i)
        pn_str      = ', '.join(f'{v:.4f}' for v in pn_start)
        print(f'\nTrial [{gi+1}/{args.N}]  factors=({factors_str})  p_n=({pn_str})',
              end='', flush=True)

        trial = run_trial(
            pn_start, val_and_grad_fn, optimizer,
            args.max_steps, tol=args.tol, patience=args.patience,
            log_interval=args.log_interval,
            param_names=param_names, scales=scales, p_n_gts=p_n_gts,
            use_wandb=use_wandb, trial_idx=gi,
        )
        all_trials.append(trial)

        final_pn  = trial['param_trajectory'][-1]
        early_tag = f'  [early@{trial["steps_run"]}]' if trial['stopped_early'] else ''
        final_str = ', '.join(f'{v:.3f}' for v in final_pn)
        print(f'  loss {trial["loss_trajectory"][0]:.3e} → '
              f'{trial["loss_trajectory"][-1]:.3e}  '
              f'p_n ({pn_str}) → ({final_str})  '
              f'({trial["total_time_s"]:.1f} s){early_tag}')

    # ── Save ──────────────────────────────────────────────────────────────────
    result = dict(
        param_names         = param_names,
        param_gts           = gt_vals,
        scales              = scales,
        p_n_gts             = p_n_gts,
        optimizer           = args.optimizer,
        lr                  = args.lr,
        lr_schedule         = args.lr_schedule,
        max_steps           = args.max_steps,
        tol                 = args.tol,
        patience            = args.patience,
        N                   = args.N,
        loss_name           = args.loss,
        tracks              = track_specs,
        range_lo            = range_lo,
        range_hi            = range_hi,
        seed                = effective_seed,
        noise_scale         = args.noise_scale,
        factor_grid         = factor_grid,
        starting_p_n_values = p_n_starts,
        trials              = all_trials,
    )

    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    print(f'\nSaved: {output_path}')

    if use_wandb:
        _wandb.finish()


if __name__ == '__main__':
    main()
