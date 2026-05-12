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
  tol_per_param, patience_per_param  optional; present when coordinate freezing is enabled
  loss_name, tracks,
  range_lo, range_hi,
  factor_grid,           list of [f1, ...] per trial
  starting_p_n_values,   list of [pn1, ...] per trial
  trials,                list of N dicts:
      param_trajectory   list of length steps+1, each entry [pn1, ...]
      grad_trajectory    list of length steps+1, each entry [g1, ...]
      loss_trajectory    list of length steps+1
      total_time_s, stopped_early, steps_run
      frozen_mask_final, tol_per_param, patience_per_param  optional; coordinate-freeze mode
  run_complete          bool; True after successful final save (informational; resume logic uses trials+N)
  live_checkpoint       optional mid-trial resume payload (must be absent when complete);
                        may include frozen_mask when using --tol-per-param
  wandb_run_id          optional; persisted so resumed jobs continue the same W&B run
  lr_multipliers        optional list[float]; resolved per-param grad scales (always stored)
  lr_mult_auto_meta     optional dict when --lr-multipliers auto (median_abs_grad, abs_grad,
                        burn_in_steps, burn_in_steps_used)
"""

import gc
import sys, os, signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

import argparse
import hashlib
import os
import pickle
import shlex
import tempfile
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

# Log per-track loss curves to W&B only below this track count (avoids huge metric cardinality).
WANDB_PER_TRACK_LOSS_MAX_TRACKS = 50
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
VALID_OPTIMIZERS = ('adam', 'sgd', 'momentum_sgd', 'newton')

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
    p.add_argument('--tol-per-param', type=float, default=None,
                   metavar='TOL',
                   help='With --patience-per-param: freeze when relative change from t-W to '
                        'now and every step-to-step change in the window are all < TOL '
                        '(default: disabled).')
    p.add_argument('--patience-per-param', type=int, default=None,
                   metavar='STEPS',
                   help='Window length W for --tol-per-param: compare to t-W and check each '
                        'of the W consecutive updates (default: disabled). '
                        'Set both flags together to enable per-parameter freezing.')
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
                   help='If > 0, rescale the full gradient vector so its L2 norm is at most '
                        'this value (global norm clip; default: 10.0). Set to 0 to disable.')
    p.add_argument('--lr-multipliers', default=None,
                   help='Per-parameter LR multipliers as comma-separated name:factor pairs, '
                        'e.g. "velocity_cm_us:0.01,lifetime_us:0.1". '
                        'Unlisted parameters keep multiplier 1.0. '
                        'Use "auto" to set each multiplier once from |dL/dp| (median-scaled, '
                        'clipped to [0.01, 10]); see --lr-mult-auto-burn-in-steps. '
                        'Values are stored in the result pickle for resume.')
    p.add_argument('--lr-mult-auto-burn-in-steps', type=int, default=100,
                   help='With --lr-multipliers auto: run this many optimizer steps first '
                        '(same LR/clip/warmup/schedule as trials, but no per-param grad scaling) '
                        'and set each multiplier from the mean |dL/dp_i| over those steps, so '
                        'multi-phase schedules are exercised by step index. '
                        '0 = use a single summed grad at trial start (step 0). Default: 100.')
    p.add_argument('--batch-size', type=int, default=1,
                   help='Number of tracks processed together on GPU per grad call (default: 1). '
                        'Larger values use vmap to parallelize tracks; try 2–4.')
    p.add_argument('--effective-batch-size', type=int, default=1,
                   help='Number of consecutive micro-batches to accumulate before one optimizer '
                        'update (default: 1). This increases effective batch size without '
                        'holding all tracks in memory at once.')
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
    p.add_argument('--wandb-tags', default=None,
                   help='Comma-separated W&B run tags (e.g. "sched_v2,fine_stage").')
    p.add_argument('--log-interval', type=int, default=50,
                   help='Log to W&B every this many steps (default: 50).')
    p.add_argument('--newton-damping', type=float, default=1e-3,
                   help='Damping for Newton optimizer (lambda in H + lambda*I). Default 1e-3.')
    p.add_argument('--adam-beta2', type=float, default=ADAM_BETA2,
                   help=f'Adam beta2 (second-moment decay). Default {ADAM_BETA2}.')
    p.add_argument('--init-from-wandb-run', default=None, metavar='RUN_ID',
                   help='Start trial 0 from param values of an existing W&B run '
                        '(fetches params/<name>_physical). '
                        'Remaining trials use random starts as usual.')
    p.add_argument('--init-from-wandb-step', type=int, default=-1, metavar='STEP',
                   help='Step to read from --init-from-wandb-run. '
                        '-1 (default) uses the run summary (last logged value). '
                        'A non-negative value fetches that exact logged step.')
    p.add_argument('--gt-param-multiplier', type=float, default=1.0,
                   help='Multiply all optimized GT parameter values by this factor before '
                        'generating the reference signal (default: 1.0, i.e. no shift). '
                        'Use 1.2 to shift the true parameters 20%% upward.')
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
    """Return sJIT-compiled (loss, grad) function for a single track."""
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


def build_phase_fns(loss_name, simulator, setter, batches, *, return_per_track_loss=False,
                    return_hessian=False):
    """Return one callable per batch in a phase.

    Default ``fn(p) -> (loss, grad)``. When ``return_per_track_loss=True``,
    ``fn(p) -> (loss, grad, per_track_losses)`` where ``per_track_losses`` has
    shape ``(batch_size,)`` and sums to ``loss``.
    When ``return_hessian=True``, ``fn(p) -> (loss, grad, hessian)`` where
    ``hessian`` has shape ``(n_params, n_params)``. Mutually exclusive with
    ``return_per_track_loss``.

    All batches with the same size share a single compiled XLA kernel — only
    the deposit/gt/wt arrays differ between calls, so JAX compiles once per
    unique batch size rather than once per batch.

    batches: list of (batch_deposits, batch_gts, batch_wts)
    """
    if return_hessian and return_per_track_loss:
        raise ValueError('return_hessian and return_per_track_loss are mutually exclusive')
    cfg = simulator.config
    n_volumes = cfg.n_volumes
    n_planes_per_vol = cfg.volumes[0].n_planes
    n_planes = n_volumes * n_planes_per_vol
    planes = tuple(range(n_planes))
    _batched_diff = jax.vmap(simulator._forward_diff, in_axes=(None, 0))

    # Pre-pad and stack deposits per batch as numpy arrays (host memory).
    # gts/wts are already numpy (moved to CPU after GT computation).
    # All three are explicit JIT arguments so JAX transfers them to device per call,
    # keeping only one batch's data on GPU at a time.
    processed = []
    for batch_deposits, batch_gts, batch_wts in batches:
        bs = len(batch_deposits)
        stacked_list = []
        for dep in batch_deposits:
            dep_padded = pad_deposit_data(dep, cfg.total_pad)
            s = jax.tree.map(lambda *xs: np.stack(xs), *dep_padded.volumes)
            stacked_list.append(s)
        batch_deps = jax.tree.map(lambda *xs: np.stack(xs), *stacked_list)
        processed.append((bs, batch_deps, batch_gts, batch_wts))

    # One compiled function per unique batch size (last batch may be smaller).
    # batch_deps/gts/wts are explicit arguments so all batches share one kernel,
    # and stop_gradient prevents JAX from storing deposit residuals in the backward pass.
    compiled_cache = {}

    def _get_compiled(bs):
        key = (bs, return_per_track_loss, return_hessian)
        if key in compiled_cache:
            return compiled_cache[key]

        if return_hessian:
            def fn_scalar(p_n_vec, batch_deps, batch_gts, batch_wts):
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

            _grad_fn = jax.grad(fn_scalar, argnums=0)

            def fn_newton(p_n_vec, batch_deps, batch_gts, batch_wts):
                val, grad = jax.value_and_grad(fn_scalar, argnums=0)(
                    p_n_vec, batch_deps, batch_gts, batch_wts)
                hess = jax.jacfwd(_grad_fn, argnums=0)(
                    p_n_vec, batch_deps, batch_gts, batch_wts)
                return val, grad, hess

            compiled = jax.jit(fn_newton)
        elif return_per_track_loss:
            def fn(p_n_vec, batch_deps, batch_gts, batch_wts):
                batch_deps = jax.lax.stop_gradient(batch_deps)
                all_signals = _batched_diff(setter(p_n_vec), batch_deps)
                loss_terms = []
                for b in range(bs):
                    pred = tuple(
                        all_signals[b, v, pl]
                        for v in range(n_volumes)
                        for pl in range(n_planes_per_vol)
                    )
                    gt_b  = tuple(jax.lax.stop_gradient(batch_gts[b][p]) for p in range(n_planes))
                    wts_b = tuple(jax.lax.stop_gradient(batch_wts[b][p]) for p in range(n_planes))
                    if loss_name == 'sobolev_loss':
                        lb = sobolev_loss(pred, gt_b, wts_b, planes)
                    elif loss_name == 'sobolev_loss_geomean_log1p':
                        lb = sobolev_loss_geomean_log1p(pred, gt_b, wts_b, planes)
                    elif loss_name == 'mse_loss':
                        lb = mse_loss(pred, gt_b)
                    elif loss_name == 'l1_loss':
                        lb = l1_loss(pred, gt_b)
                    else:
                        raise ValueError(f'Unknown loss {loss_name!r}')
                    loss_terms.append(lb)
                losses_arr = jnp.stack(loss_terms)
                return jnp.sum(losses_arr), losses_arr

            compiled = jax.jit(jax.value_and_grad(fn, argnums=0, has_aux=True))
        else:
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

        compiled_cache[key] = compiled
        return compiled

    def _make_fn(compiled, deps, gts, wts):
        if return_hessian:
            def call(p):
                val, grad, hess = compiled(p, deps, gts, wts)
                return val, grad, hess
            return call
        if return_per_track_loss:
            def call(p):
                (lv, pt), gv = compiled(p, deps, gts, wts)
                return lv, gv, pt
            return call
        return lambda p: compiled(p, deps, gts, wts)

    return [_make_fn(_get_compiled(bs), d, g, w) for bs, d, g, w in processed]


# ── Optimizer factory ──────────────────────────────────────────────────────────

def _unpack_batch_fn_ret(ret):
    if len(ret) == 3:
        return ret[0], ret[1], ret[2]
    lv, gv = ret
    return lv, gv, None


def _phase_index_at(step, phase_schedule):
    for ph_idx, (until_step, _) in enumerate(phase_schedule):
        if step < until_step:
            return ph_idx
    return len(phase_schedule) - 1


def sum_grad_batches_at_step(p0, phase_schedule, start_step):
    """Sum ∂L/∂p over all batches for the phase active at ``start_step`` (same convention as run_trial)."""
    p = jnp.asarray(p0, dtype=jnp.float32)
    ph_idx = _phase_index_at(start_step, phase_schedule)
    _, build_fn = phase_schedule[ph_idx]
    fns = build_fn(p)
    gv_acc = jnp.zeros_like(p)
    for fn in fns:
        lv, gv, _ = _unpack_batch_fn_ret(fn(p))
        jax.block_until_ready((lv, gv))
        gv_acc = gv_acc + gv
    return gv_acc


def burn_in_mean_abs_grad(p0, phase_schedule, optimizer, burn_in_steps, effective_batch_size=1):
    """Run ``burn_in_steps`` trial-like optimizer steps and return mean |∂L/∂p| per coordinate.

    Uses the same phase-vs-step indexing as ``run_trial`` (so schedule boundaries apply).
    ``optimizer`` should be built **without** per-param LR multipliers (uniform scaling).

    Returns:
        mean_abs: JAX vector, time-average of |grad| over the burn-in window
        steps_used: int, number of steps actually run (``burn_in_steps``)
    """
    n_steps = int(burn_in_steps)
    eff_bs = int(effective_batch_size)
    if n_steps <= 0:
        raise ValueError('burn_in_mean_abs_grad: burn_in_steps must be positive')
    if eff_bs < 1:
        raise ValueError('burn_in_mean_abs_grad: effective_batch_size must be >= 1')
    p = jnp.asarray(p0, dtype=jnp.float32)
    opt_state = optimizer.init(p)
    _cur_ph_idx = [-1]
    _cur_fns = [None]

    def _get_fns(ph_idx, p_):
        if ph_idx != _cur_ph_idx[0]:
            _cur_fns[0] = None
            _cur_ph_idx[0] = ph_idx
            _, build_fn = phase_schedule[ph_idx]
            _cur_fns[0] = build_fn(p_)
        return _cur_fns[0]

    def _phase_at(step, p_):
        ph_idx = _phase_index_at(step, phase_schedule)
        return ph_idx, _get_fns(ph_idx, p_)

    sum_abs = jnp.zeros_like(p)
    for step in range(n_steps):
        _, batch_fns = _phase_at(step, p)
        gv_acc = jnp.zeros_like(p)
        for micro in range(eff_bs):
            batch_idx = (step * eff_bs + micro) % len(batch_fns)
            fn = batch_fns[batch_idx]
            lv, gv, _ = _unpack_batch_fn_ret(fn(p))
            jax.block_until_ready((lv, gv))
            sum_abs = sum_abs + jnp.abs(gv)
            gv_acc = gv_acc + gv
        gv_eff = gv_acc / float(eff_bs)
        updates, opt_state = optimizer.update(gv_eff, opt_state)
        p = optax.apply_updates(p, updates)

    mean_abs = sum_abs / float(n_steps * eff_bs)
    return mean_abs, n_steps


def auto_lr_multipliers_from_grad(gv):
    """Per-parameter scales from per-coordinate sensitivity (non-negative).

    sens_i = |v_i| (typically v = ∂L/∂p or a time-mean thereof), then
    lr_mult_i = clip(median(sens) / (sens_i + 1e-8), 0.01, 10).

    Returns (multipliers list, median(sens), list of sens_i per param).
    """
    sens = jnp.abs(gv)
    med = jnp.median(sens)
    mult = jnp.clip(med / (sens + 1e-8), 0.01, 10.0)
    mult_list = [float(x) for x in mult]
    sens_list = [float(x) for x in sens]
    return mult_list, float(med), sens_list


def parse_lr_multipliers(spec, param_names):
    """Parse 'name:factor,...' string into a per-parameter scale vector (length = len(param_names)).

    Unlisted parameters default to 1.0.
    """
    scales = [1.0] * len(param_names)
    if not spec:
        return scales
    if spec.strip().lower() == 'auto':
        raise ValueError('parse_lr_multipliers: use resolve path for "auto" (caller handles this)')
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


def make_optax_optimizer(optimizer_name, lr, lr_schedule, max_steps, clip_grad_norm=0.0,
                         warmup_steps=0, lr_multipliers=None, adam_beta2=ADAM_BETA2):
    if warmup_steps > 0:
        warmup = optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps)
        if lr_schedule == 'cosine':
            post = optax.cosine_decay_schedule(init_value=lr, decay_steps=max_steps - warmup_steps)
        else:
            post = optax.constant_schedule(lr)
        schedule = optax.join_schedules([warmup, post], boundaries=[warmup_steps])
    elif lr_schedule == 'cosine':
        schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=max_steps)
    else:
        schedule = lr
    if optimizer_name == 'adam':         base = optax.adam(schedule, b1=ADAM_BETA1, b2=adam_beta2, eps=ADAM_EPS)
    elif optimizer_name == 'sgd':          base = optax.sgd(schedule)
    elif optimizer_name == 'momentum_sgd': base = optax.sgd(schedule, momentum=MOMENTUM)
    elif optimizer_name == 'newton':
        raise ValueError(
            'Newton optimizer bypasses optax entirely — do not call make_optax_optimizer for newton')
    else: raise ValueError(f'Unknown optimizer {optimizer_name!r}')
    transforms = []
    if lr_multipliers is not None and any(s != 1.0 for s in lr_multipliers):
        transforms.append(_scale_by_vector(lr_multipliers))
    if clip_grad_norm > 0.0:
        transforms.append(optax.clip_by_global_norm(clip_grad_norm))
    transforms.append(base)
    tx = optax.chain(*transforms)
    # Wrap schedule in a callable so callers can query lr at any step
    schedule_fn = schedule if callable(schedule) else (lambda _s, _lr=schedule: _lr)
    return tx, schedule_fn


# ── Trial runner ───────────────────────────────────────────────────────────────

def _serialize_opt_state(opt_state):
    """Convert JAX arrays in optax state to numpy arrays for pickling."""
    return jax.tree_util.tree_map(np.asarray, opt_state)


def _safe_pickle_dump(path, obj):
    """Write pickle atomically so interrupted writes do not leave truncated pkls."""
    path = os.path.abspath(path)
    out_dir = os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".result_", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _wandb_track_metric_suffix(track_name):
    """Fragment safe for use inside W&B metric keys."""
    return str(track_name).replace('/', '_').replace(' ', '_')


def _wandb_json_safe(value):
    """Convert values for wandb.init(config=...) / JSON-ish summaries."""
    if isinstance(value, tuple):
        return [_wandb_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_wandb_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _wandb_json_safe(v) for k, v in value.items()}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def wandb_config_dict(args, *, param_names, track_specs, schedule, effective_seed,
                      output_path, wandb_tag_list, argv_cmd):
    """Full CLI snapshot plus derived fields for W&B run config."""
    cfg = {k: _wandb_json_safe(v) for k, v in vars(args).items()}
    cfg.update(
        effective_seed=effective_seed,
        command=argv_cmd,
        param_names=param_names,
        track_names=[t['name'] for t in track_specs],
        schedule_phases=[_wandb_json_safe(ph) for ph in schedule],
        output_path=output_path,
        jax_compilation_cache_dir=jax.config.jax_compilation_cache_dir,
        wandb_tags=wandb_tag_list or None,
    )
    return cfg


def _wandb_sidecar_path(output_dir, seed):
    return os.path.join(output_dir, f'.wandb_run_id_{seed}')


def _read_stored_wandb_run_id(output_dir, seed, existing_result):
    if existing_result:
        rid = existing_result.get('wandb_run_id')
        if isinstance(rid, str) and rid.strip():
            return rid.strip()
    path = _wandb_sidecar_path(output_dir, seed)
    if os.path.isfile(path):
        try:
            with open(path, encoding='utf-8') as f:
                s = f.read().strip()
                return s or None
        except OSError:
            return None
    return None


def _stable_wandb_run_id(project, folder_name, seed):
    """Deterministic W&B run id when none was persisted (legacy checkpoints)."""
    digest = hashlib.sha256(f'{project}:{folder_name}:{seed}'.encode()).hexdigest()
    return digest[:12]


def _write_wandb_sidecar(output_dir, seed, run_id):
    if not run_id:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = _wandb_sidecar_path(output_dir, seed)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(run_id)


def optimization_run_complete(data):
    """Return True when all N trials are present (nothing left to optimize).

    ``run_complete`` is only written on clean shutdown and is **not** required here:
    treating ``run_complete=False`` after SIGTERM even though ``trials`` is full
    used to queue redundant Slurm jobs (duplicate W&B runs).
    """
    trials = data.get("trials")
    if trials is None:
        return False
    n_expected = data.get("N")
    if not isinstance(n_expected, int) or n_expected < 0:
        return False
    if len(trials) < n_expected:
        return False
    if data.get("live_checkpoint"):
        return False
    return True


def run_trial(p0, phase_schedule, optimizer, max_steps, tol=1e-5, patience=20,
              log_interval=50, param_names=None, scales=None, p_n_gts=None,
              use_wandb=False, trial_idx=0, schedule_fn=None,
              checkpoint_callback=None, lr_multipliers=None,
              initial_opt_state=None, start_step=0,
              wandb_track_batch_groups=None,
              tol_per_param=None, patience_per_param=None,
              initial_frozen_mask=None,
              effective_batch_size=1,
              newton_damping=None,
              clip_grad_norm=0.0):
    """Run one optimization trial from starting p_n vector p0 (any length).

    phase_schedule: list of (until_step, build_fn) sorted by until_step.
    build_fn(p) -> list[batch_fn] compiles and warms up on first call (cached).
    Each step uses the batch_fns of the first phase whose until_step > step.
    With start_step > 0 (resume), only that step's phase is compiled for the
    entry loss/grad row — earlier phases are not rebuilt first.

    wandb_track_batch_groups: optional ``[phase][batch] -> [(global_track_idx, name), ...]``
    built when logging per-track losses (few tracks + W&B enabled).

    tol_per_param / patience_per_param: when both set, each coordinate stops receiving
    updates when, over the last ``patience_per_param`` optimizer steps, (i) its relative
    change from the value ``patience_per_param`` steps ago is below ``tol_per_param``,
    and (ii) every consecutive step in that window has relative step change below
    ``tol_per_param``. Resumed trials restore ``initial_frozen_mask``.

    checkpoint_callback: ``callback(step, p, opt_state, frozen_mask=None)`` —
    ``frozen_mask`` is a numpy bool vector when per-parameter freezing is enabled.

    Returns dict with:
      param_trajectory   list[list]  length steps+1
      grad_trajectory    list[list]  length steps+1
      loss_trajectory    list[float] length steps+1
      total_time_s, stopped_early, steps_run
    """
    p = jnp.array(p0, dtype=jnp.float32)
    if newton_damping is not None:
        opt_state = None
    else:
        opt_state = (jax.tree_util.tree_map(jnp.array, initial_opt_state)
                     if initial_opt_state is not None else optimizer.init(p))
    eff_bs = int(effective_batch_size)
    if eff_bs < 1:
        raise ValueError('effective_batch_size must be >= 1')

    freeze_enabled = tol_per_param is not None and patience_per_param is not None
    n_dim = int(p.shape[0])
    frozen_np = np.zeros(n_dim, dtype=bool)
    if freeze_enabled and initial_frozen_mask is not None:
        fm = np.asarray(initial_frozen_mask, dtype=bool).reshape(-1)
        if fm.shape[0] != n_dim:
            raise ValueError(
                f'initial_frozen_mask length {fm.shape[0]} != n_params {n_dim}')
        frozen_np = fm.copy()

    param_traj = []
    loss_traj  = []
    grad_traj  = []

    _freeze_eps = 1e-30

    def _per_param_freeze_ok(traj, coord_i, W, tol):
        """Relative move t-W→t and every intermediate step are below ``tol`` for coord ``coord_i``."""
        L = len(traj) - 1
        a0 = float(traj[L - W][coord_i])
        a1 = float(traj[L][coord_i])
        if abs(a1 - a0) / (abs(a0) + _freeze_eps) >= tol:
            return False
        for k in range(L - W, L):
            p0 = float(traj[k][coord_i])
            p1 = float(traj[k + 1][coord_i])
            if abs(p1 - p0) / (abs(p0) + _freeze_eps) >= tol:
                return False
        return True

    t_start    = time.time()
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

    def _phase_index_at(step):
        """Which schedule phase is active at optimization loop index ``step``."""
        for ph_idx, (until_step, _) in enumerate(phase_schedule):
            if step < until_step:
                return ph_idx
        return len(phase_schedule) - 1

    def _phase_at(step, p):
        ph_idx = _phase_index_at(step)
        return ph_idx, _get_fns(ph_idx, p)

    # Representative loss/grad at loop entry: phase 0 on a fresh trial; on resume,
    # only the active phase is compiled (avoids loading phase 0 before a phase-1 resume).
    ph_idx_init = _phase_index_at(start_step)
    lv_init = 0.0
    gv_init = jnp.zeros_like(p)
    wandb_track_extra_init = {}
    if wandb_track_batch_groups is not None:
        ph_init_groups = wandb_track_batch_groups[ph_idx_init]
    else:
        ph_init_groups = None
    for batch_idx, fn in enumerate(_get_fns(ph_idx_init, p)):
        lv, gv, pt = _unpack_batch_fn_ret(fn(p))
        jax.block_until_ready((lv, gv))
        lv_init += float(lv)
        gv_init  = gv_init + gv
        if pt is not None and ph_init_groups is not None:
            groups = ph_init_groups[batch_idx]
            pt_np = np.asarray(pt)
            for loc_i, (gi, nm) in enumerate(groups):
                sk = _wandb_track_metric_suffix(nm)
                wandb_track_extra_init[f'loss/track/{gi}_{sk}'] = float(pt_np[loc_i])
    param_traj.append(p.tolist())
    loss_traj.append(lv_init)
    grad_traj.append(gv_init.tolist())

    # Skip initial W&B row when resuming — that step was already logged before checkpoint.
    if use_wandb and _WANDB_AVAILABLE and start_step == 0:
        extra0 = dict(wandb_track_extra_init or {})
        if freeze_enabled and param_names is not None:
            for i in range(n_dim):
                extra0[f'freeze/{param_names[i]}'] = (1.0 if frozen_np[i] else 0.0)
        _wandb_log_step(start_step, lv_init, gv_init, p, param_names, scales, p_n_gts,
                        step_time_s=0.0, trial_idx=trial_idx, schedule_fn=schedule_fn,
                        lr_multipliers=lr_multipliers,
                        phase=ph_idx_init if multi_phase else None,
                        extra_metrics=extra0 if extra0 else None)

    stopped_early = False
    fn = batch_fns = None  # initial state; cleared each iteration for phase-transition GC
    for step in range(start_step, max_steps):
        step_start = time.time()
        fn = None; batch_fns = None  # drop previous refs so phase transition can free old fns
        ph_idx, batch_fns = _phase_at(step, p)
        gv_acc = jnp.zeros_like(p)
        last_batch_idx = None
        newton_step_metrics = None
        if newton_damping is not None:
            # Newton update path: accumulate grad and hessian
            hv_acc = jnp.zeros((n_dim, n_dim), dtype=jnp.float32)
            for micro in range(eff_bs):
                batch_idx = (step * eff_bs + micro) % len(batch_fns)
                fn = batch_fns[batch_idx]
                lv, gv, hv = fn(p)
                #jax.block_until_ready((lv, gv, hv))
                gv_acc = gv_acc + gv
                hv_acc = hv_acc + hv
                last_batch_idx = batch_idx
            gv_avg = gv_acc / float(eff_bs)
            hv_avg = hv_acc / float(eff_bs)
            if freeze_enabled:
                active = jnp.asarray(~frozen_np, dtype=jnp.float32)
                gv_avg = gv_avg * active
            H_reg = hv_avg + newton_damping * jnp.eye(n_dim, dtype=jnp.float32)
            delta_p = jnp.linalg.solve(H_reg, -gv_avg)
            delta_norm_unclipped = float(jnp.linalg.norm(delta_p))
            if clip_grad_norm > 0:
                delta_p = jnp.where(
                    delta_norm_unclipped > clip_grad_norm,
                    delta_p * (clip_grad_norm / (delta_norm_unclipped + 1e-30)),
                    delta_p)
            lr = float(schedule_fn(step)) if schedule_fn is not None else 1.0
            p = p + lr * delta_p
            newton_step_metrics = {
                'newton_step_norm_unclipped': delta_norm_unclipped,
                'newton_step_norm': float(jnp.linalg.norm(delta_p)) * lr,
            }
            fn_eval = batch_fns[last_batch_idx]
            lv_new, gv_new, _ = fn_eval(p)
            pt_new = None
            jax.block_until_ready((lv_new, gv_new))
        else:
            # Adam / SGD path
            for micro in range(eff_bs):
                batch_idx = (step * eff_bs + micro) % len(batch_fns)
                fn = batch_fns[batch_idx]
                lv, gv, _ = _unpack_batch_fn_ret(fn(p))
                #jax.block_until_ready((lv, gv))
                gv_acc = gv_acc + gv
                last_batch_idx = batch_idx
            gv = gv_acc / float(eff_bs)
            if freeze_enabled:
                active = jnp.asarray(~frozen_np, dtype=jnp.float32)
                gv = gv * active
            updates, opt_state = optimizer.update(gv, opt_state)
            p = optax.apply_updates(p, updates)
            fn_eval = batch_fns[last_batch_idx]
            lv_new, gv_new, pt_new = _unpack_batch_fn_ret(fn_eval(p))
            jax.block_until_ready((lv_new, gv_new))
        step_time = time.time() - step_start

        param_traj.append(p.tolist())
        loss_traj.append(float(lv_new))
        grad_traj.append(gv_new.tolist())

        if use_wandb and _WANDB_AVAILABLE and wandb_track_batch_groups is not None and pt_new is not None:
            groups = wandb_track_batch_groups[ph_idx][last_batch_idx]
            pt_np = np.asarray(pt_new)
            thin = {'trial': trial_idx}
            for loc_i, (gi, nm) in enumerate(groups):
                sk = _wandb_track_metric_suffix(nm)
                thin[f'loss/track/{gi}_{sk}'] = float(pt_np[loc_i])
            _wandb.log(thin, step=step + 1)

        if freeze_enabled and len(param_traj) > patience_per_param:
            W = patience_per_param
            frozen_names = []
            for i in range(n_dim):
                if frozen_np[i]:
                    continue
                if _per_param_freeze_ok(param_traj, i, W, tol_per_param):
                    frozen_np[i] = True
                    if param_names is not None:
                        frozen_names.append(param_names[i])
            if frozen_names:
                print(f'\n    [freeze @{step + 1}] {", ".join(frozen_names)}', flush=True)

        if freeze_enabled and frozen_np.all():
            stopped_early = True
            break

        if len(param_traj) > patience:
            p_now  = np.array(param_traj[-1])
            p_prev = np.array(param_traj[-1 - patience])
            rel = np.linalg.norm(p_now - p_prev) / (np.linalg.norm(p_prev) + 1e-30)
            if rel < tol:
                stopped_early = True
                break

        if (step + 1) % log_interval == 0:
            if use_wandb and _WANDB_AVAILABLE:
                freeze_metrics = None
                if freeze_enabled and param_names is not None:
                    freeze_metrics = {
                        f'freeze/{param_names[i]}': (1.0 if frozen_np[i] else 0.0)
                        for i in range(n_dim)}
                _wandb_log_step(step + 1, float(lv_new), gv_new, p,
                                param_names, scales, p_n_gts,
                                step_time_s=step_time, trial_idx=trial_idx,
                                schedule_fn=schedule_fn,
                                lr_multipliers=lr_multipliers,
                                phase=ph_idx if multi_phase else None,
                                extra_metrics={**(freeze_metrics or {}),
                                               **(newton_step_metrics or {})})
            if checkpoint_callback is not None:
                checkpoint_callback(
                    step + 1, p, opt_state,
                    frozen_np if freeze_enabled else None)

    out = dict(
        param_trajectory = param_traj,
        grad_trajectory  = grad_traj,
        loss_trajectory  = loss_traj,
        total_time_s     = time.time() - t_start,
        stopped_early    = stopped_early,
        steps_run        = len(param_traj) - 1,
        final_opt_state  = (_serialize_opt_state(opt_state) if opt_state is not None else None),
    )
    if freeze_enabled:
        out['frozen_mask_final'] = frozen_np.tolist()
        out['tol_per_param'] = tol_per_param
        out['patience_per_param'] = patience_per_param
    return out


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
                    phase=None, extra_metrics=None):
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

    if extra_metrics:
        log.update(extra_metrics)

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

def fetch_init_params_from_wandb(run_id, param_names, scales, wandb_project, step=-1):
    """Fetch physical param values from a W&B run and return a p_n list.

    step=-1 (default): use the run summary (last logged value).
    step>=0: fetch that specific logged step via scan_history.
    Raises ValueError if any params/<name>_physical key is missing.
    """
    if not _WANDB_AVAILABLE:
        raise RuntimeError('wandb not installed; cannot use --init-from-wandb-run')
    api = _wandb.Api()
    run = api.run(f"{wandb_project}/{run_id}")
    keys = [f"params/{name}_physical" for name in param_names]

    if step < 0:
        source = run.summary
        row = {k: source.get(k) for k in keys}
        step_label = 'summary (latest)'
    else:
        rows = list(run.scan_history(keys=keys, min_step=step, max_step=step + 1))
        if not rows:
            raise ValueError(
                f"No history row found at step {step} in W&B run {run_id}. "
                f"Check that this step was actually logged."
            )
        row = rows[0]
        step_label = f'step {step}'

    p_n_init = []
    for name, scale, key in zip(param_names, scales, keys):
        val = row.get(key)
        if val is None:
            available = sorted(k for k in run.summary.keys() if k.startswith('params/'))
            raise ValueError(
                f"Key {key!r} not found at {step_label} in W&B run {run_id}.\n"
                f"Available params/* keys in summary: {available}"
            )
        p_n = float(np.log(float(val) / scale))
        p_n_init.append(p_n)
        print(f"  {name}: {float(val):.6g}  (p_n={p_n:.4f})")
    return p_n_init


def main():
    args = parse_args()
    is_newton = args.optimizer == 'newton'

    if args.effective_batch_size < 1:
        print('error: --effective-batch-size must be >= 1.', file=sys.stderr)
        raise SystemExit(2)
    if args.lr_mult_auto_burn_in_steps < 0:
        print('error: --lr-mult-auto-burn-in-steps must be >= 0.', file=sys.stderr)
        raise SystemExit(2)

    if (args.tol_per_param is None) ^ (args.patience_per_param is None):
        print('error: use both --tol-per-param and --patience-per-param, or neither.',
              file=sys.stderr)
        raise SystemExit(2)
    freeze_params_enabled = (
        args.tol_per_param is not None and args.patience_per_param is not None)
    if freeze_params_enabled and args.patience_per_param < 1:
        print('error: --patience-per-param must be >= 1.', file=sys.stderr)
        raise SystemExit(2)

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

    # Include interpreter — sys.argv[0] is only the script path (restores incorrectly omit ``python``).
    _argv_cmd = shlex.join([sys.executable] + sys.argv)
    with open(os.path.join(output_dir, f'command_{effective_seed}.txt'), 'w') as _f:
        _f.write(_argv_cmd)

    # SIGTERM as soon as paths exist — covers long GT/build phases before ``result`` exists.
    preemption_state = {'output_path': output_path, 'result': None, 'wandb_active': False}

    def _sigterm_handler(_sig, _frame):
        print('\nSIGTERM received — saving checkpoint before exit...', flush=True)
        path = preemption_state['output_path']
        r = preemption_state['result']
        try:
            if r is not None:
                r['run_complete'] = False
                _safe_pickle_dump(path, r)
                print(f'Saved: {path}', flush=True)
            else:
                print('No result checkpoint yet (still in setup phase).', flush=True)
            if preemption_state['wandb_active'] and _WANDB_AVAILABLE:
                try:
                    _wandb.finish()
                except Exception:
                    pass
        except Exception as exc:
            print(f'SIGTERM checkpoint failed: {exc}', flush=True)
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    existing_result = None
    if args.seed is not None and os.path.exists(output_path):
        try:
            with open(output_path, 'rb') as f:
                existing_result = pickle.load(f)
        except Exception as exc:
            print(f'Cannot read checkpoint {output_path}: {exc}\n'
                  f'Rename or delete this file to start fresh.')
            raise SystemExit(1) from exc
        if optimization_run_complete(existing_result):
            n_done = len(existing_result.get('trials', []))
            print(f'Skipping: {output_path} already complete ({n_done}/{args.N} trials).')
            return
        n_done = len(existing_result.get('trials', []))
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
    if freeze_params_enabled:
        print(f'Freeze coords: tol_per_param={args.tol_per_param}  '
              f'patience_per_param={args.patience_per_param}')
    print(f'N            : {args.N}')
    print(f'Seed         : {effective_seed}')
    print(f'Noise scale  : {args.noise_scale}')
    print(f'Num buckets  : {args.num_buckets:,}')
    print(f'Eff batch    : {args.effective_batch_size}')
    print(f'GT step size : {args.gt_step_size} mm  max deposits={args.gt_max_deposits:,}')
    if len(schedule) == 1:
        print(f'Fwd step size: {schedule[0]["step_size"]} mm')
        print(f'Fwd deposits : {schedule[0]["max_num_deposits"]:,}')
        print(f'Batch size   : {schedule[0]["batch_size"]}')
    else:
        print(f'Fwd schedule : {len(schedule)} phases  '
              + '  →  '.join(f'{ph["step_size"]}mm/{ph["max_num_deposits"]//1000}k/bs{ph["batch_size"]}@{ph["until_step"]}' for ph in schedule))
    _wb_tag_list = []
    if args.wandb_tags:
        _wb_tag_list = [t.strip() for t in args.wandb_tags.split(',') if t.strip()]
    _wb_summary = ("enabled  project=" + args.wandb_project +
                   ("  tags=" + ",".join(_wb_tag_list) if _wb_tag_list else ""))
    print(f'W&B          : {_wb_summary if use_wandb else "disabled"}')
    print(f'Log interval : {args.log_interval} steps')
    print(f'Output       : {output_path}')

    # ── W&B init ──────────────────────────────────────────────────────────────
    _is_all_params = (_BASE_PARAMS <= frozenset(param_names) and
                      bool(frozenset(param_names) & _BETA_VARIANTS))
    wandb_name = ('all_params__' + folder_name.split('__', 1)[1]) if _is_all_params else folder_name

    wb_run_id_for_result = None
    if use_wandb:
        resume_training = (
            existing_result is not None
            and not optimization_run_complete(existing_result)
        )
        stored_wb_id = _read_stored_wandb_run_id(output_dir, effective_seed, existing_result)

        _wandb_kw = dict(
            project=args.wandb_project,
            name=wandb_name,
            config=wandb_config_dict(
                args,
                param_names=param_names,
                track_specs=track_specs,
                schedule=schedule,
                effective_seed=effective_seed,
                output_path=output_path,
                wandb_tag_list=_wb_tag_list,
                argv_cmd=_argv_cmd,
            ),
        )
        if _wb_tag_list:
            _wandb_kw['tags'] = _wb_tag_list

        if resume_training:
            run_id = stored_wb_id or _stable_wandb_run_id(
                args.wandb_project, folder_name, effective_seed)
            _wandb_kw['id'] = run_id
            _wandb_kw['resume'] = 'allow'
            note = 'stored id' if stored_wb_id else 'synthetic id (legacy ckpt)'
            print(f'W&B resume   : {note} → {run_id}')

        _wandb.init(**_wandb_kw)
        wb_run_id_for_result = _wandb.run.id
        _write_wandb_sidecar(output_dir, effective_seed, wb_run_id_for_result)
        preemption_state['wandb_active'] = True


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

    if args.gt_param_multiplier != 1.0:
        for _pname in param_names:
            _val = _get_gt_val(_pname, gt_params, gt_sim.recomb_model) * args.gt_param_multiplier
            gt_params = _apply_param(_pname, _val, gt_params)
        print(f'GT params scaled by {args.gt_param_multiplier}x:')
        for _pname in param_names:
            print(f'  {_pname}: {_get_gt_val(_pname, gt_params, gt_sim.recomb_model):.6g}')

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
        # Move to CPU after computation to free GPU memory; JAX JIT transfers back per call.
        gt_signals_per_track.append(tuple(np.array(a) for a in gt))
        gt_weights_per_track.append(tuple(np.array(a) for a in wts))

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
    wandb_track_batch_groups = []   # [phase][batch] -> [(global_track_idx, name), ...]

    log_track_losses_wandb = (
        use_wandb and len(track_specs) < WANDB_PER_TRACK_LOSS_MAX_TRACKS
        and not is_newton)

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

        ti_cursor = 0
        phase_track_groups = []
        for batch_deps, _, _ in _batches:
            bs = len(batch_deps)
            phase_track_groups.append(
                [(ti_cursor + j, track_specs[ti_cursor + j]['name']) for j in range(bs)])
            ti_cursor += bs
        wandb_track_batch_groups.append(phase_track_groups)

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
                    fns = build_phase_fns(
                        args.loss, sim_ph, setter, batches,
                        return_per_track_loss=log_track_losses_wandb,
                        return_hessian=is_newton)
                    for fn in fns:
                        out = fn(p)
                        jax.block_until_ready(out)
                        out = fn(p)
                        jax.block_until_ready(out)
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

    if args.init_from_wandb_run:
        _step_label = (f'step {args.init_from_wandb_step}'
                       if args.init_from_wandb_step >= 0 else 'latest')
        print(f'\nFetching initial params from W&B run {args.init_from_wandb_run!r} '
              f'({_step_label})...')
        p_n_from_run = fetch_init_params_from_wandb(
            args.init_from_wandb_run, param_names, scales, args.wandb_project,
            step=args.init_from_wandb_step)
        p_n_starts[0] = p_n_from_run
        factor_grid[0] = [float(np.exp(pn - pngt))
                          for pn, pngt in zip(p_n_from_run, p_n_gts)]
        print(f'  → trial 0 initialised from W&B run '
              f'(remaining {args.N - 1} trial(s) use random starts)')

    all_trials = []
    if existing_result:
        all_trials.extend(existing_result['trials'])
    live_ckpt = (existing_result or {}).get('live_checkpoint')

    lr_spec = args.lr_multipliers
    want_auto = (not is_newton) and lr_spec is not None and lr_spec.strip().lower() == 'auto'
    lr_mult_auto_meta = None
    stored_mult = None
    if existing_result and isinstance(existing_result.get('lr_multipliers'), (list, tuple)):
        stored_mult = [float(x) for x in existing_result['lr_multipliers']]
        if len(stored_mult) != len(param_names):
            stored_mult = None

    if want_auto:
        if stored_mult is not None:
            lr_multipliers = stored_mult
            lr_mult_auto_meta = existing_result.get('lr_mult_auto_meta')
            print('LR multipliers : auto (restored from checkpoint)')
            if use_wandb and _WANDB_AVAILABLE:
                _wandb.config.update({
                    'lr_multipliers_resolved': {
                        n: float(m) for n, m in zip(param_names, lr_multipliers)},
                }, allow_val_change=True)
                if isinstance(lr_mult_auto_meta, dict):
                    wb_meta = {}
                    if lr_mult_auto_meta.get('median_abs_grad') is not None:
                        wb_meta['lr_mult_auto_median_abs_grad'] = lr_mult_auto_meta['median_abs_grad']
                    if lr_mult_auto_meta.get('burn_in_steps') is not None:
                        wb_meta['lr_mult_auto_burn_in_steps'] = lr_mult_auto_meta['burn_in_steps']
                    if lr_mult_auto_meta.get('burn_in_steps_used') is not None:
                        wb_meta['lr_mult_auto_burn_in_steps_used'] = lr_mult_auto_meta['burn_in_steps_used']
                    if wb_meta:
                        _wandb.config.update(wb_meta, allow_val_change=True)
        else:
            n_next = len(all_trials)
            pn_ref = p_n_starts[n_next]
            n_burn = int(args.lr_mult_auto_burn_in_steps)
            t_auto = time.time()
            if n_burn > 0:
                print(f'LR multipliers : auto — burn-in {n_burn} steps (mean |grad|)...',
                      flush=True)
                opt_burnin, _ = make_optax_optimizer(
                    args.optimizer, args.lr, args.lr_schedule,
                    args.max_steps,
                    clip_grad_norm=args.clip_grad_norm,
                    warmup_steps=args.warmup_steps,
                    lr_multipliers=None,
                )
                mean_abs, n_used = burn_in_mean_abs_grad(
                    pn_ref, phase_schedule, opt_burnin, n_burn,
                    effective_batch_size=args.effective_batch_size)
                jax.block_until_ready(mean_abs)
                sens_vec = mean_abs
            else:
                print('LR multipliers : auto — single grad at trial start (step 0)...',
                      flush=True)
                gv0 = sum_grad_batches_at_step(pn_ref, phase_schedule, start_step=0)
                jax.block_until_ready(gv0)
                sens_vec = gv0
                n_used = 0
            lr_multipliers, med_abs, sens_list = auto_lr_multipliers_from_grad(sens_vec)
            lr_mult_auto_meta = dict(
                median_abs_grad=med_abs,
                abs_grad={n: s for n, s in zip(param_names, sens_list)},
                burn_in_steps=n_burn,
                burn_in_steps_used=n_used,
            )
            print(f'  done ({time.time() - t_auto:.2f} s)  median(sens)={med_abs:.4e}')
            print('  resolved      : ' + ', '.join(
                f'{n}×{s:.4g}' for n, s in zip(param_names, lr_multipliers)))
            if use_wandb and _WANDB_AVAILABLE:
                _wandb.config.update({
                    'lr_multipliers_resolved': {
                        n: float(m) for n, m in zip(param_names, lr_multipliers)},
                    'lr_mult_auto_median_abs_grad': med_abs,
                    'lr_mult_auto_burn_in_steps': n_burn,
                    'lr_mult_auto_burn_in_steps_used': n_used,
                }, allow_val_change=True)
                wb_row = {f'lr_mult/{n}': float(m) for n, m in zip(param_names, lr_multipliers)}
                wb_row['lr_mult/auto_median_abs_grad'] = med_abs
                wb_row['lr_mult/auto_burn_in_steps'] = float(n_burn)
                wb_row['lr_mult/auto_burn_in_steps_used'] = float(n_used)
                for n, s in zip(param_names, sens_list):
                    wb_row[f'lr_mult/auto_abs_grad/{n}'] = float(s)
                _wandb.log(wb_row, step=0)
    elif not is_newton:
        lr_multipliers = parse_lr_multipliers(lr_spec, param_names)

    if not is_newton:
        if not want_auto and any(s != 1.0 for s in lr_multipliers):
            pairs = ', '.join(f'{n}×{s}' for n, s in zip(param_names, lr_multipliers) if s != 1.0)
            print(f'LR multipliers : {pairs}')

        optimizer, schedule_fn = make_optax_optimizer(args.optimizer, args.lr, args.lr_schedule,
                                                      args.max_steps,
                                                      clip_grad_norm=args.clip_grad_norm,
                                                      warmup_steps=args.warmup_steps,
                                                      lr_multipliers=lr_multipliers,
                                                      adam_beta2=args.adam_beta2)
    else:
        lr_multipliers = [1.0] * len(param_names)
        optimizer = None
        schedule_fn = lambda _s: args.lr

    # ── Trials ────────────────────────────────────────────────────────────────
    result = dict(
        param_names         = param_names,
        param_gts           = gt_vals,
        scales              = scales,
        p_n_gts             = p_n_gts,
        optimizer           = args.optimizer,
        lr                  = args.lr,
        lr_schedule         = args.lr_schedule,
        effective_batch_size= args.effective_batch_size,
        max_steps           = args.max_steps,
        tol                 = args.tol,
        patience            = args.patience,
        tol_per_param       = args.tol_per_param,
        patience_per_param  = args.patience_per_param,
        N                   = args.N,
        loss_name           = args.loss,
        tracks              = track_specs,
        range_lo            = range_lo,
        range_hi            = range_hi,
        seed                = effective_seed,
        noise_scale         = args.noise_scale,
        factor_grid         = factor_grid,
        starting_p_n_values = p_n_starts,
        command             = _argv_cmd,
        trials              = all_trials,
        run_complete        = False,
        wandb_run_id        = wb_run_id_for_result,
        lr_multipliers      = lr_multipliers,
    )
    if lr_mult_auto_meta is not None:
        result['lr_mult_auto_meta'] = lr_mult_auto_meta
    preemption_state['result'] = result

    # ── Intra-trial checkpoint ────────────────────────────────────────────────
    def _intra_trial_checkpoint(trial_idx, step, p, opt_state, frozen_mask=None):
        ckpt = dict(
            trial_idx = trial_idx,
            step      = step,
            p         = p.tolist(),
            opt_state = (_serialize_opt_state(opt_state) if opt_state is not None else None),
        )
        if frozen_mask is not None:
            ckpt['frozen_mask'] = np.asarray(frozen_mask, dtype=bool).tolist()
        result['live_checkpoint'] = ckpt
        _safe_pickle_dump(output_path, result)

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
        init_frozen_mask = None
        if live_ckpt and live_ckpt.get('trial_idx') == gi:
            pn0            = live_ckpt['p']
            init_opt_state = live_ckpt['opt_state']
            start_step     = live_ckpt['step']
            init_frozen_mask = live_ckpt.get('frozen_mask')
            print(f'  [resuming from step {start_step}]', end='', flush=True)

        trial = run_trial(
            pn0, phase_schedule, optimizer,
            args.max_steps, tol=args.tol, patience=args.patience,
            log_interval=args.log_interval,
            param_names=param_names, scales=scales, p_n_gts=p_n_gts,
            use_wandb=use_wandb, trial_idx=gi, schedule_fn=schedule_fn,
            checkpoint_callback=(
                lambda step, p, opt_state, fm=None: _intra_trial_checkpoint(
                    gi, step, p, opt_state, frozen_mask=fm)),
            lr_multipliers=lr_multipliers,
            initial_opt_state=init_opt_state,
            start_step=start_step,
            wandb_track_batch_groups=(wandb_track_batch_groups if log_track_losses_wandb else None),
            effective_batch_size=args.effective_batch_size,
            tol_per_param=(args.tol_per_param if freeze_params_enabled else None),
            patience_per_param=(
                args.patience_per_param if freeze_params_enabled else None),
            initial_frozen_mask=init_frozen_mask,
            newton_damping=(args.newton_damping if is_newton else None),
            clip_grad_norm=args.clip_grad_norm,
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
            _safe_pickle_dump(output_path, result)
            print(f'  [checkpoint @ {cumulative_steps} steps → {output_path}]')

    # ── Save ──────────────────────────────────────────────────────────────────
    result['run_complete'] = True
    _safe_pickle_dump(output_path, result)
    print(f'\nSaved: {output_path}')

    if use_wandb:
        _wandb.finish()


if __name__ == '__main__':
    main()
