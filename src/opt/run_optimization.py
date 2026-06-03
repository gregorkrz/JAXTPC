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
  range_intervals,     list of (lo, hi) factor pairs
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # repo root (tools.*)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))        # src/ (optlib.*)

from dotenv import load_dotenv
load_dotenv()

import argparse
import hashlib
import os
import pickle
import shlex
import tempfile
import time

# wandb (_wandb, _WANDB_AVAILABLE) imported from optlib.wandb_utils below

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tools.config import pad_deposit_data
from tools.geometry import generate_detector
from tools.loader import build_deposit_data
from tools.losses import (
    make_sobolev_weight,
    apply_fourier_power_mask,
    sobolev_loss,
    sobolev_loss_geomean_log1p,
    mse_loss,
    l1_loss,
)
from tools.noise import generate_noise
from tools.particle_generator import generate_muon_track
from tools.simulation import DetectorSimulator

# ── Constants ──────────────────────────────────────────────────────────────────

from optlib.constants import (  # noqa: E402  (constants live in optlib now)
    WANDB_PER_TRACK_LOSS_MAX_TRACKS, GT_LIFETIME_US, GT_VELOCITY_CM_US, SOBOLEV_MAX_PAD,
    CONFIG_PATH, N_SEGMENTS, MAX_ACTIVE_BUCKETS,
    EFIELD_PARAM, VALID_PARAMS, _BETA_VARIANTS, _BASE_PARAMS,
    VALID_LOSSES, VALID_OPTIMIZERS, TYPICAL_SCALES, TRACK_PRESETS,
    ADAM_BETA1, ADAM_BETA2, ADAM_EPS, MOMENTUM, PLANE_NAME_MAP,
)
from optlib.paths import (  # noqa: E402
    make_folder_name, next_result_path,
    _serialize_opt_state, _safe_pickle_dump, optimization_run_complete,
)
from optlib.params import (  # noqa: E402
    _get_gt_val, _apply_param, make_nparam_setter, build_efield_config,
)
from optlib.parsing import (  # noqa: E402
    parse_args, parse_params, parse_tracks, parse_planes,
    parse_lr_multipliers, parse_cutoff_per_param, parse_planes_per_param, parse_schedule,
)
from optlib.optim import (  # noqa: E402
    _unpack_batch_fn_ret, _phase_index_at, sum_grad_batches_at_step,
    burn_in_mean_abs_grad, auto_lr_multipliers_from_grad, make_optax_optimizer,
)
from optlib.wandb_utils import (  # noqa: E402
    _wandb, _WANDB_AVAILABLE,
    _wandb_track_metric_suffix, _wandb_json_safe, wandb_config_dict,
    _wandb_sidecar_path, _read_stored_wandb_run_id, _stable_wandb_run_id,
    _write_wandb_sidecar, _collect_gpu_metrics, _wandb_log_step,
    fetch_init_params_from_wandb,
)

# JAX compilation cache (side-effectful config stays here).
_JAX_CACHE_DIR = os.environ.get('JAX_COMPILATION_CACHE_DIR',
                                os.path.expanduser('/tmp/jax_cache'))
jax.config.update('jax_compilation_cache_dir', _JAX_CACHE_DIR)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)

# ── CLI + spec parsing: see optlib.parsing ───────────────────────────────────────

# ── Folder name + result path: see optlib.paths ─────────────────────────────────


# ── Physics / param helpers: see optlib.params ───────────────────────────────────


# ── Loss builder ───────────────────────────────────────────────────────────────

def _apply_adc_mask(pred_tuple, gt_tuple, cutoff):
    """Zero both pred and gt where |gt| < cutoff. No-op when cutoff <= 0."""
    if cutoff <= 0.0:
        return pred_tuple, gt_tuple
    masked_pred, masked_gt = [], []
    for p, g in zip(pred_tuple, gt_tuple):
        mask = jnp.abs(g) >= cutoff
        masked_pred.append(jnp.where(mask, p, 0.0))
        masked_gt.append(jnp.where(mask, g, 0.0))
    return tuple(masked_pred), tuple(masked_gt)


def build_loss_fn(loss_name, fwd_fn, gt_arrays, weights, adc_cutoff=0.0, active_planes=None):
    """Return sJIT-compiled (loss, grad) function for a single track."""
    planes = active_planes if active_planes is not None else tuple(range(len(gt_arrays)))
    if loss_name == 'sobolev_loss':
        def fn(p_n_vec):
            pred, gt = _apply_adc_mask(fwd_fn(p_n_vec), gt_arrays, adc_cutoff)
            return sobolev_loss(pred, gt, weights, planes)
    elif loss_name == 'sobolev_loss_geomean_log1p':
        def fn(p_n_vec):
            pred, gt = _apply_adc_mask(fwd_fn(p_n_vec), gt_arrays, adc_cutoff)
            return sobolev_loss_geomean_log1p(pred, gt, weights, planes)
    elif loss_name == 'mse_loss':
        def fn(p_n_vec):
            pred, gt = _apply_adc_mask(fwd_fn(p_n_vec), gt_arrays, adc_cutoff)
            return mse_loss(pred, gt)
    elif loss_name == 'l1_loss':
        def fn(p_n_vec):
            pred, gt = _apply_adc_mask(fwd_fn(p_n_vec), gt_arrays, adc_cutoff)
            return l1_loss(pred, gt)
    else:
        raise ValueError(f'Unknown loss {loss_name!r}')
    return jax.jit(jax.value_and_grad(fn))


def build_phase_fns(loss_name, simulator, setter, batches, *, return_per_track_loss=False,
                    return_hessian=False, adc_cutoff=0.0, active_planes=None):
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

    active_planes: tuple of global plane indices to include in the loss, or None for all.

    batches: list of (batch_deposits, batch_gts, batch_wts)
    """
    if return_hessian and return_per_track_loss:
        raise ValueError('return_hessian and return_per_track_loss are mutually exclusive')
    cfg = simulator.config
    n_volumes = cfg.n_volumes
    n_planes_per_vol = cfg.volumes[0].n_planes
    n_planes = n_volumes * n_planes_per_vol
    planes = active_planes if active_planes is not None else tuple(range(n_planes))
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
                    pred, gt_b = _apply_adc_mask(pred, gt_b, adc_cutoff)
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
                    pred, gt_b = _apply_adc_mask(pred, gt_b, adc_cutoff)
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
                    pred, gt_b = _apply_adc_mask(pred, gt_b, adc_cutoff)
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

# (optimizer/LR/burn-in helpers moved to optlib.optim)

# ── Trial runner ───────────────────────────────────────────────────────────────


# (W&B config/sidecar/run-id helpers moved to optlib.wandb_utils)


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
              clip_grad_norm=0.0,
              rotating_phase_schedules=None,
              n_record_coords=None,
              mlp_snapshot_interval=0):
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
    # Coordinates recorded in the param/grad trajectories. When optimizing an MLP
    # (Efield), only the leading scalar coords are recorded to keep pickles small;
    # the full vector is still saved via checkpoints and `final_p`.
    n_rec = int(n_record_coords) if n_record_coords is not None else n_dim
    n_rec = max(0, min(n_rec, n_dim))
    # Per-param features (freezing, wandb) only apply to named (scalar) coords.
    n_named = len(param_names) if param_names is not None else n_dim
    n_named = min(n_named, n_dim)
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
    mlp_traj   = []   # [(step, flat_p)] snapshots; populated when mlp_snapshot_interval > 0
    _mlp_snap  = int(mlp_snapshot_interval) if mlp_snapshot_interval and mlp_snapshot_interval > 0 else 0

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

    # Local cache of built fns per (phase, rotation) key; build_fn is the persistent cache.
    _cur_key  = [(-1, -1)]  # (ph_idx, rot_idx)
    _cur_fns  = [None]
    _n_rotations = len(rotating_phase_schedules) if rotating_phase_schedules is not None else 1

    def _get_fns(ph_idx, rot_idx, p):
        key = (ph_idx, rot_idx)
        if key != _cur_key[0]:
            # Release the previous fns before building the new ones so
            # that the builder can free its store entry without a dangling ref here.
            _cur_fns[0] = None
            _cur_key[0] = key
            sched = rotating_phase_schedules[rot_idx] if rotating_phase_schedules is not None else phase_schedule
            _, build_fn = sched[ph_idx]
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
        rot_idx = step % _n_rotations
        return ph_idx, _get_fns(ph_idx, rot_idx, p)

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
    for batch_idx, fn in enumerate(_get_fns(ph_idx_init, start_step % _n_rotations, p)):
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
    param_traj.append(p[:n_rec].tolist())
    loss_traj.append(lv_init)
    grad_traj.append(gv_init[:n_rec].tolist())
    if _mlp_snap:
        mlp_traj.append((start_step, np.asarray(p).tolist()))

    # Skip initial W&B row when resuming — that step was already logged before checkpoint.
    if use_wandb and _WANDB_AVAILABLE and start_step == 0:
        extra0 = dict(wandb_track_extra_init or {})
        if freeze_enabled and param_names is not None:
            for i in range(n_named):
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

        param_traj.append(p[:n_rec].tolist())
        loss_traj.append(float(lv_new))
        grad_traj.append(gv_new[:n_rec].tolist())
        if _mlp_snap and (step + 1) % _mlp_snap == 0:
            mlp_traj.append((step + 1, np.asarray(p).tolist()))

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
            for i in range(n_named):
                if frozen_np[i]:
                    continue
                if _per_param_freeze_ok(param_traj, i, W, tol_per_param):
                    frozen_np[i] = True
                    if param_names is not None:
                        frozen_names.append(param_names[i])
            if frozen_names:
                print(f'\n    [freeze @{step + 1}] {", ".join(frozen_names)}', flush=True)

        if freeze_enabled and n_named > 0 and frozen_np[:n_named].all():
            stopped_early = True
            break

        if n_rec > 0 and len(param_traj) > patience:
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
                        for i in range(n_named)}
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

    # Ensure the final step is always included in mlp_traj (avoid duplicate if already there).
    if _mlp_snap:
        final_step = len(param_traj) - 1 + start_step
        if not mlp_traj or mlp_traj[-1][0] != final_step:
            mlp_traj.append((final_step, np.asarray(p).tolist()))

    out = dict(
        param_trajectory = param_traj,
        grad_trajectory  = grad_traj,
        loss_trajectory  = loss_traj,
        total_time_s     = time.time() - t_start,
        stopped_early    = stopped_early,
        steps_run        = len(param_traj) - 1,
        final_opt_state  = (_serialize_opt_state(opt_state) if opt_state is not None else None),
        final_p          = np.asarray(p).tolist(),  # full vector incl. any MLP block
    )
    if _mlp_snap:
        out['mlp_trajectory'] = mlp_traj  # list of (step, flat_p) pairs
    if freeze_enabled:
        out['frozen_mask_final'] = frozen_np.tolist()
        out['tol_per_param'] = tol_per_param
        out['patience_per_param'] = patience_per_param
    return out


# (_collect_gpu_metrics, _wandb_log_step moved to optlib.wandb_utils)


# ── Noise ─────────────────────────────────────────────────────────────────────

def apply_noise_to_gt(gt_arrays, simulator, noise_scale, noise_key):
    """Add calibrated detector noise to GT arrays once before optimization.

    Uses generate_noise() (MicroBooNE model: series + white components) and
    converts from ADC to signal units via config.electrons_per_adc.
    noise_scale=1.0 gives the realistic detector noise amplitude.
    The same draw is used for all trials so they all optimise against the
    same fixed noisy target.
    noise_key : jax.Array  — PRNGKey (pass a per-track key so each track,
    living in a separate event, gets an independent noise realisation).
    """
    cfg = simulator.config
    noise_dict   = generate_noise(cfg, key=noise_key)
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

# (fetch_init_params_from_wandb moved to optlib.wandb_utils)


def main():
    args = parse_args(__doc__)
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

    _all_param_names = parse_params(args.params)
    # 'Efield' is a non-scalar MLP model handled separately from the scalar
    # param machinery; strip it out and keep param_names as scalars only.
    efield_present = EFIELD_PARAM in _all_param_names
    param_names = [n for n in _all_param_names if n != EFIELD_PARAM]
    if efield_present:
        if args.electric_dist_path is None:
            print('error: --electric-dist-path is required when "Efield" is in --params.',
                  file=sys.stderr)
            raise SystemExit(2)
        if not os.path.exists(args.electric_dist_path):
            print(f'error: --electric-dist-path not found: {args.electric_dist_path}',
                  file=sys.stderr)
            raise SystemExit(2)
        if args.optimizer == 'newton':
            print('error: "Efield" optimization is not supported with --optimizer newton '
                  '(dense Hessian over MLP weights is intractable).', file=sys.stderr)
            raise SystemExit(2)
        if args.init_from_wandb_run:
            print('error: --init-from-wandb-run is not supported together with "Efield".',
                  file=sys.stderr)
            raise SystemExit(2)

    # Per-parameter ADC cutoffs + planes: build a list of (cutoff, planes_tuple, [param_indices])
    # groups. Params sharing the same (cutoff, planes) share one compiled phase-fn set.
    _param_groups = None  # None → use single global cutoff + planes (standard path)
    # _param_groups entries are 5-tuples:
    #   (adc_cutoff, planes, freq_cutoff, fourier_cutoff, indices)
    _need_per_param = (
        args.sobolev_loss_cutoff_per_param or args.planes_per_param
        or args.freq_cutoff_per_param or args.fourier_cutoff_per_param
    )
    if _need_per_param:
        if is_newton:
            print('error: per-parameter cutoff/planes are not compatible with --optimizer newton.',
                  file=sys.stderr)
            raise SystemExit(2)
        _cutoff_per_param = parse_cutoff_per_param(
            args.sobolev_loss_cutoff_per_param or '', param_names, args.sobolev_loss_cutoff)
        _freq_cutoff_per_param_list = parse_cutoff_per_param(
            args.freq_cutoff_per_param or '', param_names, args.freq_cutoff)
        _fourier_cutoff_per_param_list = parse_cutoff_per_param(
            args.fourier_cutoff_per_param or '', param_names, args.fourier_cutoff)
        # planes_per_param parsed after active_planes is resolved (need n_planes from simulator).
        # Store the raw spec for later; resolution happens below after active_planes is set.
        _planes_per_param_spec = args.planes_per_param

    track_specs = parse_tracks(args.tracks)
    _range_vals = list(getattr(args, 'range'))
    if len(_range_vals) % 2 != 0:
        raise ValueError(f'--range requires an even number of values, got {len(_range_vals)}')
    range_intervals = [(float(_range_vals[i]), float(_range_vals[i + 1]))
                       for i in range(0, len(_range_vals), 2)]
    if any(lo >= hi for lo, hi in range_intervals):
        raise ValueError(f'Each --range pair must satisfy lo < hi, got {range_intervals}')
    _spm = [float(v) for v in args.start_position_mm.split(',')]
    track_start_mm = tuple(_spm)
    schedule     = parse_schedule(args)
    active_planes = parse_planes(args.planes) if args.planes else None

    # Resolve per-param groups now that active_planes is known.
    if _param_groups is None and _need_per_param:
        _planes_per_param = parse_planes_per_param(
            _planes_per_param_spec or '', param_names, default_planes=active_planes)
        # Group params by (adc_cutoff, planes_tuple, freq_cutoff, fourier_cutoff)
        group_key_to_indices: dict = {}
        for i, (c, pl, fc_mag, fc_pwr) in enumerate(zip(
                _cutoff_per_param, _planes_per_param,
                _freq_cutoff_per_param_list, _fourier_cutoff_per_param_list)):
            key = (c, pl, fc_mag, fc_pwr)
            group_key_to_indices.setdefault(key, []).append(i)
        # Sort groups for determinism
        _param_groups = [
            (c, pl, fc_mag, fc_pwr, indices)
            for (c, pl, fc_mag, fc_pwr), indices in sorted(
                group_key_to_indices.items(),
                key=lambda kv: (kv[0][0], str(kv[0][1]), str(kv[0][2]), str(kv[0][3]))
            )
        ]

    # ── Seeding ───────────────────────────────────────────────────────────────
    # SeedSequence(None) draws entropy from the OS; spawn gives independent
    # child sequences so starting-point and noise draws never collide.
    ss = np.random.SeedSequence(args.seed)
    effective_seed   = int(ss.entropy) if args.seed is None else args.seed
    start_ss, noise_ss = ss.spawn(2)
    start_rng        = np.random.default_rng(start_ss)
    noise_seed       = int(noise_ss.generate_state(1)[0])

    _folder_param_names = list(param_names) + (
        [f'{EFIELD_PARAM}-{args.efield_mode}'] if efield_present else [])
    folder_name = make_folder_name(
        _folder_param_names, track_specs, args.loss, args.optimizer,
        args.lr, args.lr_schedule, args.max_steps, args.N, range_intervals,
        noise_scale=args.noise_scale,
        step_size=schedule[-1]['step_size'],
        max_num_deposits=schedule[-1]['max_num_deposits'],
        n_phases=len(schedule),
        active_planes=active_planes,
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
    _range_str = ' ∪ '.join(f'[{lo}, {hi}]' for lo, hi in range_intervals)
    print(f'Range        : {_range_str} × GT')
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
    if _param_groups is not None and len(_param_groups) > 1:
        for c, pl, fc_mag, fc_pwr, idxs in _param_groups:
            names_str = ', '.join(param_names[i] for i in idxs)
            pl_str = str(pl) if pl is not None else 'all'
            fc_str = f'  freq_cutoff={fc_mag}' if fc_mag is not None else ''
            fp_str = f'  fourier_cutoff={fc_pwr}' if fc_pwr and fc_pwr > 0.0 else ''
            print(f'ADC cutoff   : {c}  planes={pl_str}{fc_str}{fp_str}  → {names_str}')
    elif args.sobolev_loss_cutoff > 0.0:
        print(f'ADC cutoff   : {args.sobolev_loss_cutoff}')
    if args.sobolev_exponent != 2.0:
        print(f'Sobolev s    : {args.sobolev_exponent}')
    if args.freq_cutoff is not None and _param_groups is None:
        print(f'Freq cutoff  : {args.freq_cutoff}')
    if args.fourier_cutoff > 0.0 and _param_groups is None:
        print(f'Fourier cut  : {args.fourier_cutoff} ADC²')
    if active_planes is not None and _param_groups is None:
        print(f'Active planes: {active_planes} (of 6 total)')
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

    # ── Simulators (cached by n_segments, role) ───────────────────────────────
    detector_config = generate_detector(CONFIG_PATH)
    _sim_cache: dict = {}
    # FieldConfig + zero-init weights for the MLP SCE model. Assigned after gt_sim
    # is built (needs geometry); read lazily by _get_sim when building diff sims.
    efield_cfg = None
    efield_zero_params = None

    def _get_sim(n_seg, role='diff'):
        # role 'gt' uses the static GT distortion map; role 'diff' uses the
        # differentiable MLP SCE model (when Efield is being optimized).
        key = (n_seg, role)
        if key not in _sim_cache:
            print(f'\nBuilding {role} simulator (n_segments={n_seg:,})...')
            kw = dict(
                differentiable=True,
                n_segments=n_seg,
                use_bucketed=True,
                max_active_buckets=args.num_buckets,
                include_noise=False,
                include_electronics=False,
                include_track_hits=False,
                include_digitize=False,
            )
            if efield_present and role == 'gt':
                kw['include_electric_dist'] = True
                kw['electric_dist_path'] = args.electric_dist_path
            elif efield_present and role == 'diff':
                kw['efield_model'] = efield_cfg
            sim = DetectorSimulator(detector_config, **kw)
            # Diff sim: inject zero MLP weights so warm-up compiles with a valid
            # sce_models pytree (real weights flow in later with same structure).
            if efield_present and role == 'diff':
                sim._default_sim_params = sim._default_sim_params._replace(
                    sce_models=efield_zero_params)
            print('Warming up JIT...')
            t0 = time.time()
            sim.warm_up()
            print(f'Done ({time.time() - t0:.1f} s)')
            _sim_cache[key] = sim
        return _sim_cache[key]

    gt_sim = _get_sim(args.gt_max_deposits, role='gt')
    gt_lifetime = args.gt_lifetime_us if args.gt_lifetime_us is not None else GT_LIFETIME_US
    gt_params = gt_sim.default_sim_params._replace(
        lifetime_us    = jnp.array(gt_lifetime),
        velocity_cm_us = jnp.array(GT_VELOCITY_CM_US),
    )
    if args.gt_lifetime_us is not None:
        print(f'GT lifetime overridden: {gt_lifetime:.0f} μs ({gt_lifetime / 1000:.1f} ms)')

    # Build the MLP field config + zero starting weights now that geometry is known.
    efield_n_mlp = 0
    efield_unravel = None
    if efield_present:
        import tools.nonlocal_efield as _nl
        efield_cfg = build_efield_config(
            gt_sim, gt_params, mode=args.efield_mode, hidden=args.efield_hidden)
        efield_zero_params = _nl.zero_params(efield_cfg)
        _flat0, efield_unravel = _nl.flatten_params(efield_zero_params)
        efield_n_mlp = int(_flat0.size)
        print(f'Efield MLP    : mode={args.efield_mode}  hidden={tuple(args.efield_hidden)}  '
              f'weights={efield_n_mlp:,}  lr_mult={args.efield_lr_mult}  '
              f'GT map={args.electric_dist_path}')

    if args.gt_param_multiplier != 1.0:
        for _pname in param_names:
            _val = _get_gt_val(_pname, gt_params, gt_sim.recomb_model) * args.gt_param_multiplier
            gt_params = _apply_param(_pname, _val, gt_params)
        print(f'GT params scaled by {args.gt_param_multiplier}x:')
        for _pname in param_names:
            print(f'  {_pname}: {_get_gt_val(_pname, gt_params, gt_sim.recomb_model):.6g}')

    # ── Setter ────────────────────────────────────────────────────────────────
    scalar_setter, gt_vals, scales, p_n_gts = make_nparam_setter(
        param_names, gt_params, gt_sim.recomb_model)

    for name, gt_val, scale, p_n_gt in zip(param_names, gt_vals, scales, p_n_gts):
        print(f'  {name}: GT={gt_val:.6g}  scale={scale:.6g}  log(GT/scale)={p_n_gt:.6g}')

    n_scalar = len(param_names)
    if efield_present:
        # The optimized vector is [scalar block | flattened MLP weights]. The
        # scalar block keeps the existing log-normalization; the MLP block is raw
        # (no log-norm) and is reshaped into SimParams.sce_models.
        _unravel = efield_unravel
        _n_scalar = n_scalar
        def setter(p_n_vec):
            base = scalar_setter(p_n_vec[:_n_scalar])
            mlp = _unravel(p_n_vec[_n_scalar:])
            return base._replace(sce_models=mlp)
    else:
        setter = scalar_setter

    # ── GT signals (computed once at fixed fine resolution) ────────────────────
    print(f'\nComputing GT signals '
          f'(step_size={args.gt_step_size}mm  max_deposits={args.gt_max_deposits:,})...')
    t0 = time.time()
    gt_signals_per_track = []
    gt_signals_clean_per_track = []  # clean (no noise) GT arrays, used for rotation
    gt_weights_per_track = []   # weights for the global/simple path
    sig_rms_acc = []
    noi_rms_acc = []
    # For per-param weight keys: dict {(fc_mag, fc_pwr) -> list[tuple[np.array,...]]}
    _unique_wt_keys = (
        list({(fc_mag, fc_pwr) for _, _, fc_mag, fc_pwr, _ in _param_groups})
        if _param_groups is not None and len(_param_groups) > 1
        else []
    )
    gt_weights_per_track_per_key: dict = {k: [] for k in _unique_wt_keys}

    noise_base_key = jax.random.PRNGKey(noise_seed)
    for track_idx, ts in enumerate(track_specs):
        print(f'  track {ts["name"]}  dir={ts["direction"]}  T={ts["momentum_mev"]} MeV')
        track_gt = generate_muon_track(
            start_position_mm=ts['start_position_mm'] if ts['start_position_mm'] is not None else track_start_mm,
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
        gt_signals_clean_per_track.append(tuple(np.array(a) for a in gt))

        if args.noise_scale > 0.0:
            track_noise_key = jax.random.fold_in(noise_base_key, track_idx)
            gt_noisy = apply_noise_to_gt(gt, gt_sim, args.noise_scale, track_noise_key)
            sig_rms_acc.append(float(np.mean([float(jnp.std(a)) for a in gt])))
            noi_rms_acc.append(float(np.mean([float(jnp.std(n - c))
                                               for n, c in zip(gt_noisy, gt)])))
            gt = gt_noisy
        else:
            sig_rms_acc.append(float(np.mean([float(jnp.std(a)) for a in gt])))

        wts = tuple(make_sobolev_weight(a.shape[0], a.shape[1], max_pad=SOBOLEV_MAX_PAD,
                                        s=args.sobolev_exponent, freq_cutoff=args.freq_cutoff)
                    for a in gt)
        if args.fourier_cutoff > 0.0:
            wts = tuple(apply_fourier_power_mask(w, a, SOBOLEV_MAX_PAD, args.fourier_cutoff)
                        for w, a in zip(wts, gt))
        # Move to CPU after computation to free GPU memory; JAX JIT transfers back per call.
        gt_signals_per_track.append(tuple(np.array(a) for a in gt))
        gt_weights_per_track.append(tuple(np.array(a) for a in wts))
        for (fc_mag, fc_pwr) in _unique_wt_keys:
            wts_k = tuple(make_sobolev_weight(a.shape[0], a.shape[1], max_pad=SOBOLEV_MAX_PAD,
                                              s=args.sobolev_exponent, freq_cutoff=fc_mag)
                          for a in gt)
            if fc_pwr and fc_pwr > 0.0:
                wts_k = tuple(apply_fourier_power_mask(w, a, SOBOLEV_MAX_PAD, fc_pwr)
                              for w, a in zip(wts_k, gt))
            gt_weights_per_track_per_key[(fc_mag, fc_pwr)].append(tuple(np.array(a) for a in wts_k))

    signal_rms = float(np.mean(sig_rms_acc))
    if args.noise_scale > 0.0:
        noise_rms = float(np.mean(noi_rms_acc))
        print(f'Done ({time.time() - t0:.1f} s) — '
              f'Signal RMS: {signal_rms:.4g}  Noise RMS: {noise_rms:.4g}  '
              f'SNR ≈ {signal_rms / max(noise_rms, 1e-30):.2f}')
    else:
        print(f'Done ({time.time() - t0:.1f} s) — Signal RMS: {signal_rms:.4g}')

    # ── Pre-generate rotating noisy GT arrays ────────────────────────────────
    # _rotate_n > 0: at step s use noise seed index s % _rotate_n.
    # Rotation 0 = the existing gt_signals_per_track (noise_base_key).
    # Rotations 1..N-1 use jax.random.fold_in(noise_base_key, r) as their base key.
    _rotate_n = args.rotate_noise_seeds if args.noise_scale > 0.0 and args.rotate_noise_seeds > 0 else -1
    gt_signals_rotating: list = []  # [rot_idx][track_idx] -> tuple of numpy arrays
    if _rotate_n > 0:
        gt_signals_rotating.append(gt_signals_per_track)  # rotation 0 = existing
        for r in range(1, _rotate_n):
            noise_key_r = jax.random.fold_in(noise_base_key, r)
            gt_r = []
            for ti, gt_clean in enumerate(gt_signals_clean_per_track):
                track_key_r = jax.random.fold_in(noise_key_r, ti)
                gt_noisy_r = apply_noise_to_gt(gt_clean, gt_sim, args.noise_scale, track_key_r)
                gt_r.append(tuple(np.array(a) for a in gt_noisy_r))
            gt_signals_rotating.append(gt_r)
        print(f'Rotating noise seeds: {_rotate_n} realisations pre-generated')

    # ── Per-phase forward: build deposits upfront, compile lazily ─────────────
    # Deposit generation is cheap; compilation is deferred to first entry of each
    # phase so that only the active phase's arrays and XLA buffers are live at once.
    # Shared store for built phase fns; allows the previous phase to be freed
    # before the next one is compiled, keeping peak memory to one phase at a time.
    _phase_fns_store: dict = {}
    phase_schedule = []   # [(until_step, build_fn), ...]
    # Per-rotation phase schedules: _rotating_phase_schedules[r] is a phase_schedule
    # that uses GT arrays from rotation r.  Each rotation has its own _fns_store so
    # compiled-fn cache entries don't collide.  Only populated when _rotate_n > 0.
    _rotating_phase_schedules: list = [[] for _ in range(max(_rotate_n, 1))]
    _rotating_fns_stores: list = [{} for _ in range(max(_rotate_n, 1))]
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
                start_position_mm=ts['start_position_mm'] if ts['start_position_mm'] is not None else track_start_mm,
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

        # Build per-key batches by substituting per-key weights into _batches.
        _batches_per_key: dict = {}
        for wt_key in _unique_wt_keys:
            wts_flat = gt_weights_per_track_per_key[wt_key]
            ti_cur = 0
            key_batches = []
            for batch_deps, batch_gts, _ in _batches:
                bs = len(batch_deps)
                key_batches.append((batch_deps, batch_gts,
                                    [wts_flat[ti_cur + j] for j in range(bs)]))
                ti_cur += bs
            _batches_per_key[wt_key] = key_batches

        ti_cursor = 0
        phase_track_groups = []
        for batch_deps, _, _ in _batches:
            bs = len(batch_deps)
            phase_track_groups.append(
                [(ti_cursor + j, track_specs[ti_cursor + j]['name']) for j in range(bs)])
            ti_cursor += bs
        wandb_track_batch_groups.append(phase_track_groups)

        print(f'Done ({time.time() - t0:.1f} s) — {len(_batches)} batches, compiling on first use')

        def _make_build_fn(ph_idx, phase, sim_ph, batches, batches_per_key,
                           _store=None, _do_gc=True):
            _fns_store = _store if _store is not None else _phase_fns_store
            def _build(p):
                if ph_idx not in _fns_store:
                    # Free previous phase's compiled fns and GPU buffers before
                    # allocating this phase's (potentially larger) data.
                    # Skip for rotating schedules (_do_gc=False) — each rotation keeps
                    # its own store; freeing one store must not clear the XLA cache
                    # shared by the other rotations.
                    prev = ph_idx - 1
                    if _do_gc and prev in _fns_store:
                        del _fns_store[prev]
                        gc.collect()          # flush Python GC so JAX array __del__ runs
                        jax.clear_caches()    # release XLA compiled executables + device memory
                    cp = f'Phase {ph_idx}' if len(schedule) > 1 else 'Compiling loss fn'
                    print(f'\n{cp} (step_size={phase["step_size"]}mm  '
                          f'deposits={phase["max_num_deposits"]:,}  '
                          f'batch_size={phase["batch_size"]})  compiling...',
                          flush=True)
                    t0 = time.time()

                    if _param_groups is not None and len(_param_groups) > 1:
                        # Per-parameter cutoff/planes/freq/fourier: build one fn set per group.
                        # Per-track loss and Hessian are not supported with per-param groups.
                        n_params_local = len(param_names)
                        all_group_fns = []
                        for cutoff_g, planes_g, fc_mag_g, fc_pwr_g, indices_g in _param_groups:
                            batches_g = batches_per_key.get((fc_mag_g, fc_pwr_g), batches)
                            fns_g = build_phase_fns(
                                args.loss, sim_ph, setter, batches_g,
                                return_per_track_loss=False,
                                return_hessian=False,
                                adc_cutoff=cutoff_g,
                                active_planes=planes_g)
                            for fn in fns_g:
                                out = fn(p); jax.block_until_ready(out)
                                out = fn(p); jax.block_until_ready(out)
                            all_group_fns.append(fns_g)

                        # Build boolean masks for gradient assembly: mask[k][i] is True
                        # when param i belongs to group k.
                        group_masks = []
                        for _, _, _, _, indices_g in _param_groups:
                            m = np.zeros(n_params_local, dtype=bool)
                            m[list(indices_g)] = True
                            group_masks.append(jnp.array(m))

                        n_batches = len(all_group_fns[0])
                        fns = []
                        for bi in range(n_batches):
                            def _make_wrapped(bi=bi, gfns=all_group_fns, masks=group_masks):
                                def wfn(p):
                                    losses = []
                                    grads = []
                                    for fns_k in gfns:
                                        lv, gv, _ = _unpack_batch_fn_ret(fns_k[bi](p))
                                        losses.append(lv)
                                        grads.append(gv)
                                    # Assemble gradient: for each group k, override the
                                    # gradient entries belonging to group k.
                                    assembled = grads[0]
                                    for k in range(1, len(grads)):
                                        assembled = jnp.where(masks[k], grads[k], assembled)
                                    return sum(losses) / len(losses), assembled
                                return wfn
                            fns.append(_make_wrapped())
                    else:
                        fns = build_phase_fns(
                            args.loss, sim_ph, setter, batches,
                            return_per_track_loss=log_track_losses_wandb,
                            return_hessian=is_newton,
                            adc_cutoff=args.sobolev_loss_cutoff,
                            active_planes=active_planes)
                        for fn in fns:
                            out = fn(p)
                            jax.block_until_ready(out)
                            out = fn(p)
                            jax.block_until_ready(out)

                    print(f'  done ({time.time() - t0:.1f} s) — {len(fns)} batches',
                          flush=True)
                    _fns_store[ph_idx] = fns
                return _fns_store[ph_idx]
            return _build

        phase_schedule.append((phase['until_step'],
                               _make_build_fn(ph_idx, phase, sim_ph, _batches, _batches_per_key)))

        # Build per-rotation batches and phase schedules.
        if _rotate_n > 0:
            n_tracks_total = len(track_specs)
            for r in range(_rotate_n):
                ti_cur = 0
                batches_r = []
                for batch_deps, _, batch_wts in _batches:
                    bs = len(batch_deps)
                    gts_r = [gt_signals_rotating[r][ti_cur + j] for j in range(bs)]
                    batches_r.append((batch_deps, gts_r, batch_wts))
                    ti_cur += bs
                batches_per_key_r: dict = {}
                for wt_key, key_batches in _batches_per_key.items():
                    ti_cur = 0
                    kbatches_r = []
                    for batch_deps, _, batch_wts in key_batches:
                        bs = len(batch_deps)
                        gts_r = [gt_signals_rotating[r][ti_cur + j] for j in range(bs)]
                        kbatches_r.append((batch_deps, gts_r, batch_wts))
                        ti_cur += bs
                    batches_per_key_r[wt_key] = kbatches_r
                _rotating_phase_schedules[r].append(
                    (phase['until_step'],
                     _make_build_fn(ph_idx, phase, sim_ph, batches_r, batches_per_key_r,
                                    _store=_rotating_fns_stores[r], _do_gc=False)))

    # ── Random starting points ────────────────────────────────────────────────
    rng = start_rng
    n_params = len(param_names)
    _widths  = np.array([hi - lo for lo, hi in range_intervals])
    _cum_w   = np.cumsum(_widths)
    _total_w = _cum_w[-1]
    _u       = rng.uniform(0.0, _total_w, size=(args.N, n_params))
    _k       = np.searchsorted(_cum_w, _u, side='right').clip(0, len(range_intervals) - 1)
    _los     = np.array([lo for lo, _ in range_intervals])
    _offsets = np.where(_k > 0, _cum_w[np.maximum(_k - 1, 0)], 0.0)
    factors  = _los[_k] + (_u - _offsets)                                # (N, n_scalar)
    factor_grid   = factors.tolist()
    if n_params > 0:
        _p_n_scalar = np.log(factors) + np.array(p_n_gts)               # (N, n_scalar)
    else:
        _p_n_scalar = np.zeros((args.N, 0))
    if efield_present:
        # MLP block: random hidden + zeroed output layer ⇒ the field starts at
        # nominal (no distortion) yet has nonzero gradient so it can learn.
        # Distinct seed per trial so trials explore different hidden features.
        import tools.nonlocal_efield as _nl
        mlp_blocks = []
        for _t in range(args.N):
            _k = jax.random.PRNGKey(int(effective_seed) + 7919 + _t)
            _ip = _nl.nominal_start_params(efield_cfg, _k)
            _flat, _ = _nl.flatten_params(_ip)
            mlp_blocks.append(np.asarray(_flat, dtype=np.float64))
        _p_n_scalar = np.hstack([_p_n_scalar, np.stack(mlp_blocks)])
    p_n_starts    = _p_n_scalar.tolist()

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
        if efield_present:
            # Extend per-param multipliers to per-coordinate: keep the scalar
            # block, set every MLP weight to the single Efield multiplier. The
            # optimizer's _scale_by_vector needs length = len(p) = n_coords.
            lr_multipliers = (list(lr_multipliers[:n_scalar])
                              + [args.efield_lr_mult] * efield_n_mlp)
        if not want_auto and any(s != 1.0 for s in lr_multipliers[:n_scalar]):
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
        active_planes       = active_planes,
        tracks              = track_specs,
        range_intervals     = range_intervals,
        seed                = effective_seed,
        noise_scale         = args.noise_scale,
        factor_grid         = factor_grid,
        # Store only the scalar block of starts / multipliers (MLP block is all
        # zeros / a constant) to keep the pickle small and resume-compatible.
        starting_p_n_values = ([row[:n_scalar] for row in p_n_starts]
                               if efield_present else p_n_starts),
        command             = _argv_cmd,
        trials              = all_trials,
        run_complete        = False,
        wandb_run_id        = wb_run_id_for_result,
        lr_multipliers      = (list(lr_multipliers[:n_scalar])
                               if efield_present else lr_multipliers),
    )
    if lr_mult_auto_meta is not None:
        result['lr_mult_auto_meta'] = lr_mult_auto_meta
    if efield_present:
        # Everything needed to reconstruct the learned field from a trial's
        # `final_p` (the trailing block after the scalar coords).
        result['efield'] = dict(
            present       = True,
            mode          = efield_cfg.mode,
            hidden        = list(efield_cfg.hidden),
            n_weights     = efield_n_mlp,
            n_scalar      = n_scalar,
            lr_mult       = args.efield_lr_mult,
            gt_map_path   = args.electric_dist_path,
            center_cm     = list(efield_cfg.center_cm),
            half_cm       = list(efield_cfg.half_cm),
            bg_field_Vcm  = list(efield_cfg.bg_field_Vcm),
            out_scale     = efield_cfg.out_scale,
        )
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
        pn_str      = ', '.join(f'{v:.4f}' for v in pn_start[:n_scalar])  # scalar coords only
        _ef_tag     = f'  +Efield[{efield_n_mlp}w@0]' if efield_present else ''
        print(f'\nTrial [{gi+1}/{args.N}]  factors=({factors_str})  p_n=({pn_str}){_ef_tag}',
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
            rotating_phase_schedules=(_rotating_phase_schedules if _rotate_n > 0 else None),
            n_record_coords=(n_scalar if efield_present else None),
            mlp_snapshot_interval=(args.mlp_snapshot_interval if efield_present else 0),
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
