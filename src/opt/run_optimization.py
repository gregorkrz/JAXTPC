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
import gc
import sys, os, signal
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

from tools.config import pad_deposit_data
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
                                os.path.expanduser('/tmp/jax_cache'))
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

# "All params" = all non-beta params + at least one beta variant (model-specific).
_BETA_VARIANTS = frozenset({'recomb_beta', 'recomb_beta_90'})
_BASE_PARAMS   = frozenset(VALID_PARAMS) - _BETA_VARIANTS

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
    p.add_argument('--lr-schedule', default='cosine', choices=('constant', 'cosine'),
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
    p.add_argument('--warmup-steps', type=int, default=100,
                   help='Linear LR warmup from 0 to --lr over this many steps '
                        '(default: 100, set to 0 to disable).')
    p.add_argument('--clip-grad-norm', type=float, default=10.0,
                   help='Clip each gradient component independently to [-value, value] '
                        'before optimizer update (default: 10.0, set to 0 to disable).')
    p.add_argument('--lr-multipliers', default=None,
                   help='Per-parameter LR multipliers as comma-separated name:factor pairs, '
                        'e.g. "velocity_cm_us:0.01,lifetime_us:0.1". '
                        'Unlisted parameters keep multiplier 1.0.')
    p.add_argument('--batch-size', type=int, default=1,
                   help='Number of tracks processed together on GPU per grad call (default: 1). '
                        'Larger values use vmap to parallelize tracks; try 2–4.')
    p.add_argument('--step-size', type=float, default=0.1,
                   help='Muon track step size in mm (default: 0.1). '
                        'Larger values reduce deposit count and memory use.')
    p.add_argument('--max-num-deposits', type=int, default=50_000,
                   help='Static deposit buffer size passed to the differentiable simulator '
                        'as n_segments (default: 50000). Must be >= actual deposits per track.')
    p.add_argument('--num-buckets', type=int, default=1000,
                   help='Max active buckets for non-differentiable bucketed accumulation '
                        '(default: 1000). Increase if you see bucket overflow warnings.')
    p.add_argument('--schedule-steps', default=None,
                   help='Comma-separated step thresholds that divide optimization into phases '
                        '(e.g. "1000" → 2 phases; "1000,5000" → 3 phases).')
    p.add_argument('--schedule-step-sizes', default=None,
                   help='Comma-separated step sizes in mm, one per phase (e.g. "1.0,0.1").')
    p.add_argument('--schedule-deposits', default=None,
                   help='Comma-separated max-num-deposits, one per phase (e.g. "5000,50000").')
    p.add_argument('--schedule-batch-sizes', default=None,
                   help='Comma-separated batch sizes, one per phase (e.g. "5,1").')
    p.add_argument('--gt-step-size', type=float, default=0.1,
                   help='Step size in mm used to generate GT signals (default: 0.1). '
                        'Independent of the forward simulation schedule.')
    p.add_argument('--gt-max-deposits', type=int, default=50_000,
                   help='Static deposit buffer for the GT simulator (default: 50000). '
                        'Must be >= actual deposits per track at --gt-step-size.')
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
                     lr_schedule, max_steps, N, range_lo, range_hi,
                     noise_scale=0.0, step_size=0.1, max_num_deposits=50_000, n_phases=1):
    _is_all = (_BASE_PARAMS <= frozenset(param_names) and
               bool(frozenset(param_names) & _BETA_VARIANTS))
    params_tag = 'all_params' if _is_all else '+'.join(param_names)
    tracks_tag = (f'{len(track_specs)}tracks' if len(track_specs) >= 6
                  else '+'.join(t['name'] for t in track_specs))
    sched_tag     = '_cosine' if lr_schedule == 'cosine' else ''
    range_tag     = f'r{range_lo:.3g}_{range_hi:.3g}'.replace('.', 'p')
    noise_tag     = f'_noise{noise_scale:.3g}'.replace('.', 'p') if noise_scale > 0.0 else ''
    ss_tag        = f'_ss{step_size:.3g}'.replace('.', 'p') if step_size != 0.1 else ''
    dep_tag       = f'_dep{max_num_deposits // 1000}k' if max_num_deposits != 50_000 else ''
    phase_tag     = f'_sched{n_phases}' if n_phases > 1 else ''
    return (f'{params_tag}__{tracks_tag}__{loss_name}__'
            f'{optimizer}_lr{lr}{sched_tag}_s{max_steps}_N{N}_{range_tag}'
            f'{noise_tag}{ss_tag}{dep_tag}{phase_tag}')


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
    # q[i] = log(physical[i] / scale[i]);  physical[i] = exp(q[i]) * scale[i]
    p_n_gts = [float(np.log(v / s)) for v, s in zip(gt_vals, scales)]

    def setter(p_n_vec):
        params = gt_params
        for i, name in enumerate(param_names):
            params = _apply_param(name, jnp.exp(p_n_vec[i]) * scales[i], params)
        return params

    return setter, gt_vals, scales, p_n_gts


# ── Loss builder ───────────────────────────────────────────────────────────────

def build_loss_fn(loss_name, fwd_fn, gt_arrays, weights):
    """Return JIT-compiled (loss, grad) function for a single track."""
    planes = tuple(range(len(gt_arrays)))
    if loss_name == 'sobolev_loss':
        def fn(p_n_vec): return sobolev_loss(fwd_fn(p_n_vec), gt_arrays, weights, planes)
    elif loss_name == 'sobolev_loss_geomean_log1p':
        def fn(p_n_vec): return sobolev_loss_geomean_log1p(fwd_fn(p_n_vec), gt_arrays, weights, planes)
    elif loss_name == 'mse_loss':
        def fn(p_n_vec): return mse_loss(fwd_fn(p_n_vec), gt_arrays)
    elif loss_name == 'l1_loss':
        def fn(p_n_vec): return l1_loss(fwd_fn(p_n_vec), gt_arrays)
    else:
        raise ValueError(f'Unknown loss {loss_name!r}')
    return jax.jit(jax.value_and_grad(fn))


def build_phase_fns(loss_name, simulator, setter, batches):
    """Return one callable fn(p_n_vec) -> (loss, grad) per batch in a phase.

    All batches with the same size share a single compiled XLA kernel — only
    the deposit/gt/wt arrays differ between calls, so JAX compiles once per
    unique batch size rather than once per batch.

    batches: list of (batch_deposits, batch_gts, batch_wts)
    """
    cfg = simulator.config
    n_volumes = cfg.n_volumes
    n_planes_per_vol = cfg.volumes[0].n_planes
    n_planes = n_volumes * n_planes_per_vol
    planes = tuple(range(n_planes))
    _batched_diff = jax.vmap(simulator._forward_diff, in_axes=(None, 0))

    # Pre-pad and stack deposits per batch.
    # gts/wts are kept as-is (lists of tuples) — no duplicate allocation.
    processed = []
    for batch_deposits, batch_gts, batch_wts in batches:
        bs = len(batch_deposits)
        stacked_list = []
        for dep in batch_deposits:
            dep_padded = pad_deposit_data(dep, cfg.total_pad)
            s = jax.tree.map(lambda *xs: jnp.stack(xs), *dep_padded.volumes)
            stacked_list.append(s)
        batch_deps = jax.tree.map(lambda *xs: jnp.stack(xs), *stacked_list)
        processed.append((bs, batch_deps, batch_gts, batch_wts))

    # One compiled function per unique batch size (last batch may be smaller).
    # batch_deps/gts/wts are explicit arguments so all batches share one kernel,
    # and stop_gradient prevents JAX from storing deposit residuals in the backward pass.
    compiled_cache = {}

    def _get_compiled(bs):
        if bs in compiled_cache:
            return compiled_cache[bs]

        def fn(p_n_vec, batch_deps, batch_gts, batch_wts):
            batch_deps = jax.lax.stop_gradient(batch_deps)
            all_signals = _batched_diff(setter(p_n_vec), batch_deps)
            total = 0.0
            for b in range(bs):
                pred = tuple(
                    all_signals[b, v, pl]
                    for v in range(n_volumes)
                    for pl in range(n_planes_per_vol)
                )
                gt_b  = tuple(jax.lax.stop_gradient(batch_gts[b][p]) for p in range(n_planes))
                wts_b = tuple(jax.lax.stop_gradient(batch_wts[b][p]) for p in range(n_planes))
                if loss_name == 'sobolev_loss':
                    total = total + sobolev_loss(pred, gt_b, wts_b, planes)
                elif loss_name == 'sobolev_loss_geomean_log1p':
                    total = total + sobolev_loss_geomean_log1p(pred, gt_b, wts_b, planes)
                elif loss_name == 'mse_loss':
                    total = total + mse_loss(pred, gt_b)
                elif loss_name == 'l1_loss':
                    total = total + l1_loss(pred, gt_b)
                else:
                    raise ValueError(f'Unknown loss {loss_name!r}')
            return total

        compiled = jax.jit(jax.value_and_grad(fn, argnums=0))
        compiled_cache[bs] = compiled
        return compiled

    def _make_fn(compiled, deps, gts, wts):
        return lambda p: compiled(p, deps, gts, wts)

    return [_make_fn(_get_compiled(bs), d, g, w) for bs, d, g, w in processed]


# ── Optimizer factory ──────────────────────────────────────────────────────────

def parse_lr_multipliers(spec, param_names):
    """Parse 'name:factor,...' string into a per-parameter scale vector (length = len(param_names)).

    Unlisted parameters default to 1.0.
    """
    scales = [1.0] * len(param_names)
    if not spec:
        return scales
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        name, factor = item.split(':')
        name = name.strip()
        if name not in param_names:
            raise ValueError(f'--lr-multipliers: unknown param {name!r}. Known: {param_names}')
        scales[param_names.index(name)] = float(factor)
    return scales


def parse_schedule(args):
    """Return a list of phase dicts: {step_size, max_num_deposits, batch_size, until_step}.

    Single-phase (no --schedule-steps) returns a one-element list using the
    top-level --step-size / --max-num-deposits / --batch-size values.
    """
    if args.schedule_steps is None:
        return [dict(step_size=args.step_size,
                     max_num_deposits=args.max_num_deposits,
                     batch_size=args.batch_size,
                     until_step=args.max_steps)]

    thresholds = [int(x.strip()) for x in args.schedule_steps.split(',')]
    n_phases   = len(thresholds) + 1

    def _csv(s, typ, default):
        if s is None:
            return [typ(default)] * n_phases
        vals = [typ(x.strip()) for x in s.split(',')]
        if len(vals) != n_phases:
            raise ValueError(
                f'Expected {n_phases} comma-separated values (got {len(vals)}): {s!r}')
        return vals

    step_sizes  = _csv(args.schedule_step_sizes,  float, args.step_size)
    deposits    = _csv(args.schedule_deposits,     int,   args.max_num_deposits)
    batch_sizes = _csv(args.schedule_batch_sizes,  int,   args.batch_size)

    until_steps = thresholds + [args.max_steps]
    return [dict(step_size=ss, max_num_deposits=dep, batch_size=bs, until_step=us)
            for ss, dep, bs, us in zip(step_sizes, deposits, batch_sizes, until_steps)]


def _scale_by_vector(scales):
    """Optax transform that element-wise multiplies gradients by a fixed scale vector."""
    scales_arr = jnp.array(scales, dtype=jnp.float32)
    def init_fn(params):
        return ()
    def update_fn(updates, state, params=None):
        return updates * scales_arr, state
    return optax.GradientTransformation(init_fn, update_fn)


class _ManualPreconditionedOptimizer:
    """Fallback optimizer wrapper when optax.chain/clip are unavailable.

    Applies per-parameter LR multipliers and component-wise clipping before
    delegating to the wrapped optimizer.
    """
    def __init__(self, base_optimizer, lr_multipliers=None, clip_grad_norm=0.0):
        self._base = base_optimizer
        self._lr_multipliers = lr_multipliers
        self._clip_grad_norm = clip_grad_norm

    def init(self, params):
        return self._base.init(params)

    def update(self, grads, state):
        updates = grads
        if self._lr_multipliers is not None and any(s != 1.0 for s in self._lr_multipliers):
            updates = updates * jnp.array(self._lr_multipliers, dtype=jnp.float32)
        if self._clip_grad_norm > 0.0:
            updates = jnp.clip(updates, -self._clip_grad_norm, self._clip_grad_norm)
        return self._base.update(updates, state)


def make_optax_optimizer(optimizer_name, lr, lr_schedule, max_steps, clip_grad_norm=0.0,
                         warmup_steps=0, lr_multipliers=None):
    if warmup_steps > 0 and all(
        hasattr(optax, name) for name in ("linear_schedule", "constant_schedule", "join_schedules")
    ):
        warmup = optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps)
        if lr_schedule == 'cosine':
            post = optax.cosine_decay_schedule(init_value=lr, decay_steps=max_steps - warmup_steps)
        else:
            post = optax.constant_schedule(lr)
        schedule = optax.join_schedules([warmup, post], boundaries=[warmup_steps])
    elif warmup_steps > 0:
        # Compatibility fallback for lightweight optax stubs used in import tests.
        if lr_schedule == "cosine":
            post = optax.cosine_decay_schedule(init_value=lr, decay_steps=max(max_steps - warmup_steps, 1))
        else:
            post = lambda _step, _lr=lr: _lr

        def schedule(step):
            if step < warmup_steps:
                return lr * float(step) / float(max(warmup_steps, 1))
            return post(step - warmup_steps)
    elif lr_schedule == 'cosine':
        schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=max_steps)
    else:
        schedule = lr

    if optimizer_name == 'adam':
        try:
            base = optax.adam(schedule, b1=ADAM_BETA1, b2=ADAM_BETA2, eps=ADAM_EPS)
        except TypeError:
            base = optax.adam(schedule)
    elif optimizer_name == 'sgd':
        base = optax.sgd(schedule)
    elif optimizer_name == 'momentum_sgd':
        base = optax.sgd(schedule, momentum=MOMENTUM)
    else: raise ValueError(f'Unknown optimizer {optimizer_name!r}')

    use_multipliers = lr_multipliers is not None and any(s != 1.0 for s in lr_multipliers)
    use_clipping = clip_grad_norm > 0.0
    if hasattr(optax, "chain") and (not use_clipping or hasattr(optax, "clip")):
        transforms = []
        if use_multipliers:
            transforms.append(_scale_by_vector(lr_multipliers))
        if use_clipping:
            transforms.append(optax.clip(clip_grad_norm))
        transforms.append(base)
        tx = optax.chain(*transforms)
    elif use_multipliers or use_clipping:
        tx = _ManualPreconditionedOptimizer(
            base_optimizer=base,
            lr_multipliers=lr_multipliers,
            clip_grad_norm=clip_grad_norm,
        )
    else:
        tx = base
    # Wrap schedule in a callable so callers can query lr at any step
    schedule_fn = schedule if callable(schedule) else (lambda _s, _lr=schedule: _lr)
    return tx, schedule_fn


# ── Trial runner ───────────────────────────────────────────────────────────────

def _serialize_opt_state(opt_state):
    """Convert JAX arrays in optax state to numpy arrays for pickling."""
    tree_util = getattr(jax, "tree_util", None)
    if tree_util is not None and hasattr(tree_util, "tree_map"):
        return tree_util.tree_map(np.asarray, opt_state)

    def _to_numpy(value):
        if isinstance(value, dict):
            return {k: _to_numpy(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(_to_numpy(v) for v in value)
        if isinstance(value, list):
            return [_to_numpy(v) for v in value]
        try:
            return np.asarray(value)
        except Exception:
            return value

    return _to_numpy(opt_state)


def run_trial(p0, phase_schedule=None, optimizer=None, max_steps=0, tol=1e-5, patience=20,
              log_interval=50, param_names=None, scales=None, p_n_gts=None,
              use_wandb=False, trial_idx=0, schedule_fn=None,
              checkpoint_callback=None, lr_multipliers=None,
              initial_opt_state=None, start_step=0, val_and_grad_fn=None):
    """Run one optimization trial from starting p_n vector p0 (any length).

    phase_schedule: list of (until_step, build_fn) sorted by until_step.
    build_fn(p) -> list[batch_fn] compiles and warms up on first call (cached).
    Each step uses the batch_fns of the first phase whose until_step > step.

    Returns dict with:
      param_trajectory   list[list]  length steps+1
      grad_trajectory    list[list]  length steps+1
      loss_trajectory    list[float] length steps+1
      total_time_s, stopped_early, steps_run
    """
    if optimizer is None:
        raise ValueError("optimizer must be provided")

    if isinstance(optimizer, tuple):
        # Backward compatibility with older call sites/tests that pass
        # make_optax_optimizer() output directly.
        optimizer, schedule_from_tuple = optimizer
        if schedule_fn is None:
            schedule_fn = schedule_from_tuple

    p = jnp.array(p0, dtype=jnp.float32)
    if initial_opt_state is not None:
        tree_util = getattr(jax, "tree_util", None)
        if tree_util is not None and hasattr(tree_util, "tree_map"):
            opt_state = tree_util.tree_map(jnp.array, initial_opt_state)
        else:
            opt_state = initial_opt_state
    else:
        opt_state = optimizer.init(p)

    param_traj = []
    loss_traj  = []
    grad_traj  = []

    t_start = time.time()

    # Legacy/simple optimization path used by lightweight unit tests.
    if val_and_grad_fn is not None and phase_schedule is None:
        lv_init, gv_init = val_and_grad_fn(p)
        param_traj.append(p.tolist())
        loss_traj.append(float(lv_init))
        grad_traj.append(np.asarray(gv_init).tolist())

        stopped_early = False
        for step in range(start_step, max_steps):
            lv, gv = val_and_grad_fn(p)
            updates, opt_state = optimizer.update(gv, opt_state)
            p = optax.apply_updates(p, updates)
            lv_new, gv_new = val_and_grad_fn(p)

            param_traj.append(p.tolist())
            loss_traj.append(float(lv_new))
            grad_traj.append(np.asarray(gv_new).tolist())

            if len(param_traj) > patience:
                p_now = np.array(param_traj[-1])
                p_prev = np.array(param_traj[-1 - patience])
                rel = np.linalg.norm(p_now - p_prev) / (np.linalg.norm(p_prev) + 1e-30)
                if rel < tol:
                    stopped_early = True
                    break

        return dict(
            param_trajectory=param_traj,
            grad_trajectory=grad_traj,
            loss_trajectory=loss_traj,
            total_time_s=time.time() - t_start,
            stopped_early=stopped_early,
            steps_run=len(param_traj) - 1,
            final_opt_state=_serialize_opt_state(opt_state),
        )

    if phase_schedule is None:
        raise ValueError("phase_schedule must be provided when val_and_grad_fn is not set")

    multi_phase = len(phase_schedule) > 1

    # Local cache of built fns per phase index; build_fn itself is the persistent cache.
    _cur_ph_idx  = [-1]
    _cur_fns     = [None]

    def _get_fns(ph_idx, p):
        if ph_idx != _cur_ph_idx[0]:
            # Release the previous phase's fns before building the new one so
            # that the builder can free its store entry without a dangling ref here.
            _cur_fns[0]    = None
            _cur_ph_idx[0] = ph_idx
            _, build_fn = phase_schedule[ph_idx]
            _cur_fns[0] = build_fn(p)
        return _cur_fns[0]

    def _phase_at(step, p):
        for ph_idx, (until_step, _) in enumerate(phase_schedule):
            if step < until_step:
                return ph_idx, _get_fns(ph_idx, p)
        last = len(phase_schedule) - 1
        return last, _get_fns(last, p)

    # Step 0: aggregate phase-0 batches for a representative initial loss/grad.
    lv_init = 0.0
    gv_init = jnp.zeros_like(p)
    for fn in _get_fns(0, p):
        lv, gv = fn(p)
        jax.block_until_ready((lv, gv))
        lv_init += float(lv)
        gv_init  = gv_init + gv
    param_traj.append(p.tolist())
    loss_traj.append(lv_init)
    grad_traj.append(gv_init.tolist())

    if use_wandb and _WANDB_AVAILABLE:
        _wandb_log_step(start_step, lv_init, gv_init, p, param_names, scales, p_n_gts,
                        step_time_s=0.0, trial_idx=trial_idx, schedule_fn=schedule_fn,
                        lr_multipliers=lr_multipliers,
                        phase=0 if multi_phase else None)

    stopped_early = False
    fn = batch_fns = None  # initial state; cleared each iteration for phase-transition GC
    for step in range(start_step, max_steps):
        step_start = time.time()
        fn = None; batch_fns = None  # drop previous refs so phase transition can free old fns
        ph_idx, batch_fns = _phase_at(step, p)
        fn = batch_fns[step % len(batch_fns)]
        lv, gv = fn(p)
        jax.block_until_ready((lv, gv))
        updates, opt_state = optimizer.update(gv, opt_state)
        p = optax.apply_updates(p, updates)
        lv_new, gv_new = fn(p)
        jax.block_until_ready((lv_new, gv_new))
        step_time = time.time() - step_start

        param_traj.append(p.tolist())
        loss_traj.append(float(lv_new))
        grad_traj.append(gv_new.tolist())

        if (step + 1) % log_interval == 0:
            if use_wandb and _WANDB_AVAILABLE:
                _wandb_log_step(step + 1, float(lv_new), gv_new, p,
                                param_names, scales, p_n_gts,
                                step_time_s=step_time, trial_idx=trial_idx,
                                schedule_fn=schedule_fn,
                                lr_multipliers=lr_multipliers,
                                phase=ph_idx if multi_phase else None)
            if checkpoint_callback is not None:
                checkpoint_callback(step + 1, p, opt_state)

        if len(param_traj) > patience:
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
        final_opt_state  = _serialize_opt_state(opt_state),
    )


def _collect_gpu_metrics():
    """Collect GPU utilization and memory from nvidia-smi and JAX device memory stats."""
    metrics = {}
    devs = jax.local_devices()
    for i, dev in enumerate(devs):
        try:
            mem = dev.memory_stats()
            if mem:
                pfx = f'gpu{i}' if len(devs) > 1 else 'gpu'
                metrics[f'sys/{pfx}/jax_mem_gb']  = mem.get('bytes_in_use', 0) / 2**30
                metrics[f'sys/{pfx}/jax_peak_gb'] = mem.get('peak_bytes_in_use', 0) / 2**30
        except Exception:
            pass
    try:
        import subprocess
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=index,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode()
        for line in out.strip().splitlines():
            idx_s, util_s, used_s, total_s = [s.strip() for s in line.split(',')]
            pfx = f'sys/gpu{idx_s}'
            metrics[f'{pfx}/util_pct']     = float(util_s)
            metrics[f'{pfx}/mem_used_gb']  = float(used_s) / 1024.0
            metrics[f'{pfx}/mem_total_gb'] = float(total_s) / 1024.0
    except Exception:
        pass
    return metrics


def _wandb_log_step(step, loss, gv, p, param_names, scales, p_n_gts,
                    step_time_s, trial_idx, schedule_fn=None, lr_multipliers=None,
                    phase=None):
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
    if phase is not None:
        log['phase'] = phase
    if schedule_fn is not None:
        log['lr'] = float(schedule_fn(step))

    if param_names is not None:
        for i, name in enumerate(param_names):
            log[f'params/{name}_normalized'] = float(p_np[i])
            if scales is not None:
                log[f'params/{name}_physical'] = float(np.exp(p_np[i]) * scales[i])
            if p_n_gts is not None:
                # rel_err in physical space: |exp(q) - exp(q_gt)| / exp(q_gt) = |exp(q - q_gt) - 1|
                rel_err = abs(float(np.exp(p_np[i] - p_n_gts[i])) - 1.0)
                log[f'params/{name}_rel_err'] = rel_err
            if gv_np is not None:
                scale = lr_multipliers[i] if lr_multipliers is not None else 1.0
                log[f'grads/{name}'] = float(gv_np[i] * scale)

    log.update(_collect_gpu_metrics())
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
    schedule    = parse_schedule(args)

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
        step_size=schedule[-1]['step_size'],
        max_num_deposits=schedule[-1]['max_num_deposits'],
        n_phases=len(schedule),
    )
    output_dir  = os.path.join(args.results_base, folder_name)
    output_path = next_result_path(output_dir, seed=effective_seed)

    with open(os.path.join(output_dir, f'command_{effective_seed}.txt'), 'w') as _f:
        _f.write(' '.join(sys.argv))

    existing_result = None
    if args.seed is not None and os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            existing_result = pickle.load(f)
        n_done = len(existing_result.get('trials', []))
        if n_done >= args.N:
            print(f'Skipping: {output_path} already complete ({n_done}/{args.N} trials).')
            return
        print(f'Resuming: {output_path} ({n_done}/{args.N} trials done).')

    use_wandb = (not args.no_wandb) and _WANDB_AVAILABLE
    if not args.no_wandb and not _WANDB_AVAILABLE:
        print('Warning: wandb not installed — logging disabled. pip install wandb to enable.')

    print(f'JAX devices  : {jax.devices()}')
    print(f'Params       : {param_names}')
    print(f'Range        : [{range_lo}, {range_hi}] × GT')
    print(f'Tracks       : {[t["name"] for t in track_specs]}')
    print(f'Loss         : {args.loss}')
    clip_tag   = f'{args.clip_grad_norm}' if args.clip_grad_norm > 0.0 else 'disabled'
    warmup_tag = f'{args.warmup_steps}' if args.warmup_steps > 0 else 'disabled'
    print(f'Optimizer    : {args.optimizer}  lr={args.lr}  schedule={args.lr_schedule}  '
          f'warmup={warmup_tag}  clip_grad_norm={clip_tag}')
    print(f'Max steps    : {args.max_steps}  tol={args.tol}  patience={args.patience}')
    print(f'N            : {args.N}')
    print(f'Seed         : {effective_seed}')
    print(f'Noise scale  : {args.noise_scale}')
    print(f'Num buckets  : {args.num_buckets:,}')
    print(f'GT step size : {args.gt_step_size} mm  max deposits={args.gt_max_deposits:,}')
    if len(schedule) == 1:
        print(f'Fwd step size: {schedule[0]["step_size"]} mm')
        print(f'Fwd deposits : {schedule[0]["max_num_deposits"]:,}')
        print(f'Batch size   : {schedule[0]["batch_size"]}')
    else:
        print(f'Fwd schedule : {len(schedule)} phases  '
              + '  →  '.join(f'{ph["step_size"]}mm/{ph["max_num_deposits"]//1000}k/bs{ph["batch_size"]}@{ph["until_step"]}' for ph in schedule))
    print(f'W&B          : {"enabled  project=" + args.wandb_project if use_wandb else "disabled"}')
    print(f'Log interval : {args.log_interval} steps')
    print(f'Output       : {output_path}')

    # ── W&B init ──────────────────────────────────────────────────────────────
    _is_all_params = (_BASE_PARAMS <= frozenset(param_names) and
                      bool(frozenset(param_names) & _BETA_VARIANTS))
    wandb_name = ('all_params__' + folder_name.split('__', 1)[1]) if _is_all_params else folder_name

    if use_wandb:
        _wandb.init(
            project=args.wandb_project,
            name=wandb_name,
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
                warmup_steps  = args.warmup_steps,
                clip_grad_norm = args.clip_grad_norm,
                seed          = effective_seed,
                log_interval  = args.log_interval,
                output_path   = output_path,
                jax_compilation_cache_dir = jax.config.jax_compilation_cache_dir,
            ),
        )


    # ── Schedule ──────────────────────────────────────────────────────────────
    if len(schedule) > 1:
        print(f'\nSchedule: {len(schedule)} phases')
        for i, ph in enumerate(schedule):
            print(f'  Phase {i}: steps 0–{ph["until_step"]}  '
                  f'step_size={ph["step_size"]}mm  '
                  f'deposits={ph["max_num_deposits"]:,}  '
                  f'batch_size={ph["batch_size"]}')

    # ── Simulators (cached by n_segments) ─────────────────────────────────────
    detector_config = generate_detector(CONFIG_PATH)
    _sim_cache: dict = {}

    def _get_sim(n_seg):
        if n_seg not in _sim_cache:
            print(f'\nBuilding differentiable simulator (n_segments={n_seg:,})...')
            sim = DetectorSimulator(
                detector_config,
                differentiable=True,
                n_segments=n_seg,
                use_bucketed=True,
                max_active_buckets=args.num_buckets,
                include_noise=False,
                include_electronics=False,
                include_track_hits=False,
                include_digitize=False,
            )
            print('Warming up JIT...')
            t0 = time.time()
            sim.warm_up()
            print(f'Done ({time.time() - t0:.1f} s)')
            _sim_cache[n_seg] = sim
        return _sim_cache[n_seg]

    gt_sim = _get_sim(args.gt_max_deposits)
    gt_params = gt_sim.default_sim_params._replace(
        lifetime_us    = jnp.array(GT_LIFETIME_US),
        velocity_cm_us = jnp.array(GT_VELOCITY_CM_US),
    )

    # ── Setter ────────────────────────────────────────────────────────────────
    setter, gt_vals, scales, p_n_gts = make_nparam_setter(
        param_names, gt_params, gt_sim.recomb_model)

    for name, gt_val, scale, p_n_gt in zip(param_names, gt_vals, scales, p_n_gts):
        print(f'  {name}: GT={gt_val:.6g}  scale={scale:.6g}  log(GT/scale)={p_n_gt:.6g}')

    # ── GT signals (computed once at fixed fine resolution) ────────────────────
    print(f'\nComputing GT signals '
          f'(step_size={args.gt_step_size}mm  max_deposits={args.gt_max_deposits:,})...')
    t0 = time.time()
    gt_signals_per_track = []
    gt_weights_per_track = []
    sig_rms_acc = []
    noi_rms_acc = []

    for ts in track_specs:
        print(f'  track {ts["name"]}  dir={ts["direction"]}  T={ts["momentum_mev"]} MeV')
        track_gt = generate_muon_track(
            start_position_mm=(0.0, 0.0, 0.0),
            direction=ts['direction'],
            kinetic_energy_mev=ts['momentum_mev'],
            step_size_mm=args.gt_step_size,
            track_id=1,
        )
        deposits_gt = build_deposit_data(
            track_gt['position'], track_gt['de'], track_gt['dx'], gt_sim.config,
            theta=track_gt['theta'], phi=track_gt['phi'],
            track_ids=track_gt['track_id'],
        )
        n_total = sum(v.n_actual for v in deposits_gt.volumes)
        print(f'    {n_total:,} deposits')

        gt = tuple(gt_sim.forward(gt_params, deposits_gt))
        jax.block_until_ready(gt)

        if args.noise_scale > 0.0:
            gt_noisy = apply_noise_to_gt(gt, gt_sim, args.noise_scale, noise_seed)
            sig_rms_acc.append(float(np.mean([float(jnp.std(a)) for a in gt])))
            noi_rms_acc.append(float(np.mean([float(jnp.std(n - c))
                                               for n, c in zip(gt_noisy, gt)])))
            gt = gt_noisy
        else:
            sig_rms_acc.append(float(np.mean([float(jnp.std(a)) for a in gt])))

        wts = tuple(make_sobolev_weight(a.shape[0], a.shape[1], max_pad=SOBOLEV_MAX_PAD)
                    for a in gt)
        gt_signals_per_track.append(gt)
        gt_weights_per_track.append(wts)

    signal_rms = float(np.mean(sig_rms_acc))
    if args.noise_scale > 0.0:
        noise_rms = float(np.mean(noi_rms_acc))
        print(f'Done ({time.time() - t0:.1f} s) — '
              f'Signal RMS: {signal_rms:.4g}  Noise RMS: {noise_rms:.4g}  '
              f'SNR ≈ {signal_rms / max(noise_rms, 1e-30):.2f}')
    else:
        print(f'Done ({time.time() - t0:.1f} s) — Signal RMS: {signal_rms:.4g}')

    # ── Per-phase forward: build deposits upfront, compile lazily ─────────────
    # Deposit generation is cheap; compilation is deferred to first entry of each
    # phase so that only the active phase's arrays and XLA buffers are live at once.
    # Shared store for built phase fns; allows the previous phase to be freed
    # before the next one is compiled, keeping peak memory to one phase at a time.
    _phase_fns_store: dict = {}
    phase_schedule = []   # [(until_step, build_fn), ...]

    for ph_idx, phase in enumerate(schedule):
        sim_ph = _get_sim(phase['max_num_deposits'])
        prefix = f'Phase {ph_idx}' if len(schedule) > 1 else 'Building deposits'
        print(f'\n{prefix} (fwd step_size={phase["step_size"]}mm  '
              f'deposits={phase["max_num_deposits"]:,}  '
              f'batch_size={phase["batch_size"]})...')
        t0 = time.time()

        _batches = []
        _deps, _gts, _wts = [], [], []

        for ti, ts in enumerate(track_specs):
            track_ph = generate_muon_track(
                start_position_mm=(0.0, 0.0, 0.0),
                direction=ts['direction'],
                kinetic_energy_mev=ts['momentum_mev'],
                step_size_mm=phase['step_size'],
                track_id=1,
            )
            deposits_ph = build_deposit_data(
                track_ph['position'], track_ph['de'], track_ph['dx'], sim_ph.config,
                theta=track_ph['theta'], phi=track_ph['phi'],
                track_ids=track_ph['track_id'],
            )
            n_total = sum(v.n_actual for v in deposits_ph.volumes)
            print(f'  track {ts["name"]}  {n_total:,} fwd deposits')

            _deps.append(deposits_ph)
            _gts.append(gt_signals_per_track[ti])
            _wts.append(gt_weights_per_track[ti])

            if len(_deps) == phase['batch_size']:
                _batches.append((list(_deps), list(_gts), list(_wts)))
                _deps.clear(); _gts.clear(); _wts.clear()

        if _deps:
            _batches.append((list(_deps), list(_gts), list(_wts)))

        print(f'Done ({time.time() - t0:.1f} s) — {len(_batches)} batches, compiling on first use')

        def _make_build_fn(ph_idx, phase, sim_ph, batches):
            def _build(p):
                if ph_idx not in _phase_fns_store:
                    # Free previous phase's compiled fns and GPU buffers before
                    # allocating this phase's (potentially larger) data.
                    prev = ph_idx - 1
                    if prev in _phase_fns_store:
                        del _phase_fns_store[prev]
                        gc.collect()          # flush Python GC so JAX array __del__ runs
                        jax.clear_caches()    # release XLA compiled executables + device memory
                    cp = f'Phase {ph_idx}' if len(schedule) > 1 else 'Compiling loss fn'
                    print(f'\n{cp} (step_size={phase["step_size"]}mm  '
                          f'deposits={phase["max_num_deposits"]:,}  '
                          f'batch_size={phase["batch_size"]})  compiling...',
                          flush=True)
                    t0 = time.time()
                    fns = build_phase_fns(args.loss, sim_ph, setter, batches)
                    for fn in fns:
                        _ = fn(p); jax.block_until_ready(_)
                        _ = fn(p); jax.block_until_ready(_)
                    print(f'  done ({time.time() - t0:.1f} s) — {len(fns)} batches',
                          flush=True)
                    _phase_fns_store[ph_idx] = fns
                return _phase_fns_store[ph_idx]
            return _build

        phase_schedule.append((phase['until_step'],
                               _make_build_fn(ph_idx, phase, sim_ph, _batches)))

    # ── Random starting points ────────────────────────────────────────────────
    rng = start_rng
    n_params = len(param_names)
    factors = rng.uniform(range_lo, range_hi, size=(args.N, n_params))   # (N, n_params)
    factor_grid   = factors.tolist()
    p_n_starts    = (np.log(factors) + np.array(p_n_gts)).tolist()

    lr_multipliers = parse_lr_multipliers(args.lr_multipliers, param_names)
    if any(s != 1.0 for s in lr_multipliers):
        pairs = ', '.join(f'{n}×{s}' for n, s in zip(param_names, lr_multipliers) if s != 1.0)
        print(f'LR multipliers : {pairs}')

    optimizer, schedule_fn = make_optax_optimizer(args.optimizer, args.lr, args.lr_schedule,
                                                  args.max_steps,
                                                  clip_grad_norm=args.clip_grad_norm,
                                                  warmup_steps=args.warmup_steps,
                                                  lr_multipliers=lr_multipliers)

    # ── Trials ────────────────────────────────────────────────────────────────
    all_trials = []
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
        command             = ' '.join(sys.argv),
        trials              = all_trials,
    )

    # ── Intra-trial checkpoint ────────────────────────────────────────────────
    def _intra_trial_checkpoint(trial_idx, step, p, opt_state):
        result['live_checkpoint'] = dict(
            trial_idx = trial_idx,
            step      = step,
            p         = p.tolist(),
            opt_state = _serialize_opt_state(opt_state),
        )
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)

    # ── Preemption handler ────────────────────────────────────────────────────
    def _sigterm_handler(_sig, _frame):
        print('\nSIGTERM received — saving checkpoint before exit...', flush=True)
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)
        print(f'Saved: {output_path}', flush=True)
        if use_wandb and _WANDB_AVAILABLE:
            _wandb.finish()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Pre-populate completed trials from a prior (preempted) run.
    if existing_result:
        all_trials.extend(existing_result['trials'])
    live_ckpt = (existing_result or {}).get('live_checkpoint')

    cumulative_steps   = 0
    last_save_at_step  = 0

    for gi, (factors_i, pn_start) in enumerate(zip(factor_grid, p_n_starts)):
        if gi < len(all_trials):
            continue  # already completed in a previous run

        factors_str = ', '.join(f'{f:.4f}' for f in factors_i)
        pn_str      = ', '.join(f'{v:.4f}' for v in pn_start)
        print(f'\nTrial [{gi+1}/{args.N}]  factors=({factors_str})  p_n=({pn_str})',
              end='', flush=True)

        pn0            = pn_start
        init_opt_state = None
        start_step     = 0
        if live_ckpt and live_ckpt.get('trial_idx') == gi:
            pn0            = live_ckpt['p']
            init_opt_state = live_ckpt['opt_state']
            start_step     = live_ckpt['step']
            print(f'  [resuming from step {start_step}]', end='', flush=True)

        trial = run_trial(
            pn0, phase_schedule, optimizer,
            args.max_steps, tol=args.tol, patience=args.patience,
            log_interval=args.log_interval,
            param_names=param_names, scales=scales, p_n_gts=p_n_gts,
            use_wandb=use_wandb, trial_idx=gi, schedule_fn=schedule_fn,
            checkpoint_callback=lambda step, p, opt_state: _intra_trial_checkpoint(gi, step, p, opt_state),
            lr_multipliers=lr_multipliers,
            initial_opt_state=init_opt_state,
            start_step=start_step,
        )
        all_trials.append(trial)
        result.pop('live_checkpoint', None)

        final_pn  = trial['param_trajectory'][-1]
        early_tag = f'  [early@{trial["steps_run"]}]' if trial['stopped_early'] else ''
        final_str = ', '.join(f'{v:.3f}' for v in final_pn)
        print(f'  loss {trial["loss_trajectory"][0]:.3e} → '
              f'{trial["loss_trajectory"][-1]:.3e}  '
              f'p_n ({pn_str}) → ({final_str})  '
              f'({trial["total_time_s"]:.1f} s){early_tag}')

        cumulative_steps += trial['steps_run']
        if cumulative_steps - last_save_at_step >= 100:
            last_save_at_step = cumulative_steps
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)
            print(f'  [checkpoint @ {cumulative_steps} steps → {output_path}]')

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    print(f'\nSaved: {output_path}')

    if use_wandb:
        _wandb.finish()


if __name__ == '__main__':
    main()
