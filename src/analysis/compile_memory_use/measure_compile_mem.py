#!/usr/bin/env python
"""
Measure peak GPU memory consumed during XLA compilation of the batched
value_and_grad loss function used in run_optimization.py.

Runs in a fresh process (one configuration per invocation), so the peak
memory reported is the high-water mark from simulator warm-up + the
compilation of the value_and_grad kernel for the requested batch size.

Inputs to the compiled function are dummy zero arrays of the correct static
shapes — values don't matter for compilation, only shapes do.

Usage
-----
    python src/analysis/compile_memory_use/measure_compile_mem.py \\
        --batch-size 2 \\
        --max-deposits 50000 \\
        --xla-effort 3 \\
        --output-json /tmp/result.json

XLA effort levels (xla_backend_optimization_level)
    0 = no optimisation
    1 = basic
    2 = moderate
    3 = maximum (default in JAX)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import argparse
import json
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from tools.geometry import generate_detector
from tools.losses import make_sobolev_weight, sobolev_loss_geomean_log1p
from tools.simulation import DetectorSimulator
from tools.config import VolumeDeposits, DepositData, pad_deposit_data

GT_LIFETIME_US    = 10_000.0
GT_VELOCITY_CM_US = 0.160
SOBOLEV_MAX_PAD   = 128
CONFIG_PATH       = 'config/cubic_wireplane_config.yaml'
MAX_ACTIVE_BUCKETS = 1000

_JAX_CACHE_DIR = os.environ.get('JAX_COMPILATION_CACHE_DIR',
                                os.path.expanduser('/tmp/jax_cache_measure'))
jax.config.update('jax_compilation_cache_dir', _JAX_CACHE_DIR)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 9999.0)


def _gpu_mem_stats():
    out = {}
    devs = jax.local_devices()
    for i, dev in enumerate(devs):
        try:
            m = dev.memory_stats()
            if m:
                pfx = f'gpu{i}' if len(devs) > 1 else 'gpu'
                out[f'{pfx}_jax_bytes_in_use'] = m.get('bytes_in_use', 0)
                out[f'{pfx}_jax_peak_bytes']   = m.get('peak_bytes_in_use', 0)
        except Exception:
            pass
    try:
        r = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode()
        for line in r.strip().splitlines():
            idx_s, used_s, total_s = [s.strip() for s in line.split(',')]
            out[f'gpu{idx_s}_nvml_used_mb']  = float(used_s)
            out[f'gpu{idx_s}_nvml_total_mb'] = float(total_s)
    except Exception:
        pass
    return out


def _dummy_deposits(cfg, total_pad, n_vol):
    """Build a DepositData of the right shape filled with zeros (dx=1 to avoid /0)."""
    def _vol(vol_idx):
        p = total_pad
        return VolumeDeposits(
            positions_mm         = np.zeros((p, 3), dtype=np.float32),
            de                   = np.zeros(p, dtype=np.float32),
            dx                   = np.ones(p, dtype=np.float32),
            theta                = np.zeros(p, dtype=np.float32),
            phi                  = np.zeros(p, dtype=np.float32),
            track_ids            = np.zeros(p, dtype=np.int32),
            group_ids            = np.zeros(p, dtype=np.int32),
            t0_us                = np.zeros(p, dtype=np.float32),
            interaction_ids      = np.full(p, -1, dtype=np.int16),
            ancestor_track_ids   = np.full(p, -1, dtype=np.int32),
            pdg                  = np.zeros(p, dtype=np.int32),
            charge               = np.zeros(p, dtype=np.float32),
            photons              = np.zeros(p, dtype=np.float32),
            qs_fractions         = np.zeros(p, dtype=np.float32),
            n_actual             = 1,
        )
    n_groups = getattr(cfg, 'max_groups', total_pad)
    return DepositData(
        volumes          = tuple(_vol(v) for v in range(n_vol)),
        group_to_track   = np.zeros(n_groups, dtype=np.int32),
        original_indices = np.zeros(total_pad, dtype=np.int32),
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--batch-size',   type=int, default=1)
    p.add_argument('--max-deposits', type=int, default=50_000)
    p.add_argument('--xla-effort',   type=int, default=3, choices=[0, 1, 2, 3])
    p.add_argument('--n-params',     type=int, default=2,
                   help='Length of p_n_vec (default: 2 = velocity + lifetime)')
    p.add_argument('--accumulate',   action='store_true',
                   help='Use scan over batch instead of vmap — compiles body once '
                        '(O(1) memory in batch size) at the cost of sequential execution')
    p.add_argument('--hessian',      action='store_true',
                   help='Compile jacfwd(grad) in addition to value_and_grad, '
                        'matching the Newton optimizer path (n_params extra forward passes)')
    p.add_argument('--output-json',  default=None)
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    print(f'[measure_compile_mem] batch_size={args.batch_size}  '
          f'max_deposits={args.max_deposits}  xla_effort={args.xla_effort}')
    print(f'  JAX devices: {jax.devices()}')

    # ── Build simulator ────────────────────────────────────────────────────────
    print('  Building simulator...')
    detector_config = generate_detector(CONFIG_PATH)
    simulator = DetectorSimulator(
        detector_config,
        differentiable=True,
        n_segments=args.max_deposits,
        use_bucketed=True,
        max_active_buckets=MAX_ACTIVE_BUCKETS,
        include_noise=False,
        include_electronics=False,
        include_track_hits=False,
        include_digitize=False,
        track_config=None,
    )
    cfg              = simulator.config
    n_volumes        = cfg.n_volumes
    n_planes_per_vol = cfg.volumes[0].n_planes
    n_planes         = n_volumes * n_planes_per_vol
    planes           = tuple(range(n_planes))

    print('  Warming up simulator JIT...')
    simulator.warm_up()

    gt_params = simulator.default_sim_params._replace(
        lifetime_us    = jnp.array(GT_LIFETIME_US),
        velocity_cm_us = jnp.array(GT_VELOCITY_CM_US),
    )

    # ── Infer output plane shapes from a single dummy forward pass ────────────
    # (needed to build gt/weight arrays of correct shape; no real deposits required)
    print('  Inferring output shapes via one dummy forward pass...')
    dummy_dep = _dummy_deposits(cfg, cfg.total_pad, n_volumes)
    ref_arrays = simulator.forward(gt_params, dummy_dep)
    jax.effects_barrier()
    plane_shapes = [np.asarray(a).shape for a in ref_arrays]
    print(f'  Plane shapes: {plane_shapes}')

    # ── Build dummy batch inputs (all zeros, correct shapes) ──────────────────
    # batch_gts / batch_wts: list[batch_size] of tuples of plane arrays
    dummy_gt_np  = tuple(np.zeros(sh, dtype=np.float32) for sh in plane_shapes)
    dummy_wts_np = tuple(
        np.asarray(make_sobolev_weight(sh[0], sh[1], max_pad=SOBOLEV_MAX_PAD))
        for sh in plane_shapes
    )
    all_gts = [dummy_gt_np] * args.batch_size
    all_wts = [dummy_wts_np] * args.batch_size

    # batch_deps: stack batch_size copies of a padded dummy deposit
    dummy_padded = pad_deposit_data(dummy_dep, cfg.total_pad)
    stacked_vol  = jax.tree.map(lambda *xs: np.stack(xs), *dummy_padded.volumes)
    batch_deps_np = jax.tree.map(
        lambda x: np.stack([x] * args.batch_size), stacked_vol)

    # ── Build p_n_vec setter ───────────────────────────────────────────────────
    PARAM_NAMES = ['velocity_cm_us', 'lifetime_us',
                   'diffusion_trans_cm2_us', 'diffusion_long_cm2_us',
                   'recomb_alpha', 'recomb_beta_90', 'recomb_R']
    SCALES      = [0.1, 10_000.0, 1e-5, 1e-5, 1.0, 0.2, 1.0]
    used_params = PARAM_NAMES[:args.n_params]
    used_scales = SCALES[:args.n_params]

    def apply_param(name, val, params):
        rp = params.recomb_params
        if name == 'velocity_cm_us':         return params._replace(velocity_cm_us=val)
        if name == 'lifetime_us':            return params._replace(lifetime_us=val)
        if name == 'diffusion_trans_cm2_us': return params._replace(diffusion_trans_cm2_us=val)
        if name == 'diffusion_long_cm2_us':  return params._replace(diffusion_long_cm2_us=val)
        if name == 'recomb_alpha':           return params._replace(recomb_params=rp._replace(alpha=val))
        if name == 'recomb_beta_90':         return params._replace(recomb_params=rp._replace(beta_90=val))
        if name == 'recomb_R':              return params._replace(recomb_params=rp._replace(R=val))
        raise ValueError(name)

    def setter(p_n_vec):
        params = gt_params
        for i, name in enumerate(used_params):
            params = apply_param(name, jnp.exp(p_n_vec[i]) * used_scales[i], params)
        return params

    bs = args.batch_size

    if args.accumulate:
        # scan over batch: body compiled once → O(1) graph in batch size
        print('  Mode: scan (gradient accumulation, sequential over batch)')

        def loss_fn(p_n_vec, batch_deps, batch_gts, batch_wts):
            params = setter(p_n_vec)

            def body(carry, x):
                deps_b, gt_b, wts_b = x
                deps_b = jax.lax.stop_gradient(deps_b)
                signals = simulator._forward_diff(params, deps_b)
                pred = tuple(signals[v, pl]
                             for v in range(n_volumes) for pl in range(n_planes_per_vol))
                gt_b  = tuple(jax.lax.stop_gradient(g) for g in gt_b)
                wts_b = tuple(jax.lax.stop_gradient(w) for w in wts_b)
                l = sobolev_loss_geomean_log1p(pred, gt_b, wts_b, planes)
                return carry + l, None

            # Stack gts/wts into arrays so scan can iterate over axis 0
            gt_stacked  = tuple(np.stack([batch_gts[b][p]  for b in range(bs)]) for p in range(n_planes))
            wts_stacked = tuple(np.stack([batch_wts[b][p]  for b in range(bs)]) for p in range(n_planes))
            total, _ = jax.lax.scan(body, jnp.zeros(()), (batch_deps, gt_stacked, wts_stacked))
            return total

    else:
        # vmap over batch: all elements in parallel → O(batch_size) graph
        print('  Mode: vmap (parallel over batch)')
        _batched_diff = jax.vmap(simulator._forward_diff, in_axes=(None, 0))

        def loss_fn(p_n_vec, batch_deps, batch_gts, batch_wts):
            batch_deps  = jax.lax.stop_gradient(batch_deps)
            all_signals = _batched_diff(setter(p_n_vec), batch_deps)
            total = jnp.zeros(())
            for b in range(bs):
                pred  = tuple(all_signals[b, v, pl]
                              for v in range(n_volumes) for pl in range(n_planes_per_vol))
                gt_b  = tuple(jax.lax.stop_gradient(batch_gts[b][p]) for p in range(n_planes))
                wts_b = tuple(jax.lax.stop_gradient(batch_wts[b][p]) for p in range(n_planes))
                total = total + sobolev_loss_geomean_log1p(pred, gt_b, wts_b, planes)
            return total

    if args.hessian:
        print(f'  Mode: {"scan" if args.accumulate else "vmap"} + hessian '
              f'(jacfwd(grad), {args.n_params} extra forward passes)')
        _grad_fn = jax.grad(loss_fn, argnums=0)

        def loss_grad_hess(p_n_vec, batch_deps, batch_gts, batch_wts):
            val, grad = jax.value_and_grad(loss_fn, argnums=0)(
                p_n_vec, batch_deps, batch_gts, batch_wts)
            hess = jax.jacfwd(_grad_fn, argnums=0)(
                p_n_vec, batch_deps, batch_gts, batch_wts)
            return val, grad, hess

        compiled_fn = jax.jit(
            loss_grad_hess,
            compiler_options={'xla_backend_optimization_level': args.xla_effort},
        )
    else:
        compiled_fn = jax.jit(
            jax.value_and_grad(loss_fn, argnums=0),
            compiler_options={'xla_backend_optimization_level': args.xla_effort},
        )

    # ── Trigger compilation, measure memory delta ──────────────────────────────
    jax.effects_barrier()
    mem_before = _gpu_mem_stats()
    t_before   = time.time()
    kernel_desc = 'value_and_grad + jacfwd(grad)' if args.hessian else 'value_and_grad'
    print(f'  Triggering JIT compilation of {kernel_desc}...')

    p_n_gt = jnp.zeros(args.n_params)
    oom = False
    try:
        ret = compiled_fn(p_n_gt, batch_deps_np, all_gts, all_wts)
        jax.effects_barrier()
        loss_val = ret[0]
        loss_str = f'{float(loss_val):.4g}'
    except Exception as e:
        oom = True
        loss_str = 'OOM'
        print(f'  OOM / error during compilation+first call: {e}')

    t_compile  = time.time() - t_before
    mem_after  = _gpu_mem_stats()
    print(f'  Done in {t_compile:.1f} s  (loss={loss_str})')

    peak_before_bytes = mem_before.get('gpu_jax_peak_bytes', 0)
    peak_after_bytes  = mem_after.get('gpu_jax_peak_bytes', 0)
    nvml_key = next((k for k in mem_after if k.endswith('_nvml_used_mb')), None)
    nvml_used_mb = mem_after.get(nvml_key) if nvml_key else None

    result = dict(
        batch_size          = args.batch_size,
        max_deposits        = args.max_deposits,
        xla_effort          = args.xla_effort,
        n_params            = args.n_params,
        hessian             = args.hessian,
        mode                = ('scan' if args.accumulate else 'vmap') + ('+hessian' if args.hessian else ''),
        oom                 = oom,
        compile_time_s      = round(t_compile, 2),
        jax_peak_total_gib  = round(peak_after_bytes / 2**30, 4),
        jax_peak_delta_gib  = round((peak_after_bytes - peak_before_bytes) / 2**30, 4),
        nvml_used_mb        = round(nvml_used_mb, 1) if nvml_used_mb is not None else None,
        total_wall_s        = round(time.time() - t_start, 1),
        mem_before          = mem_before,
        mem_after           = mem_after,
    )

    print('\n  Result:')
    for k, v in result.items():
        if k not in ('mem_before', 'mem_after'):
            print(f'    {k}: {v}')

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\n  Written to {args.output_json}')

    print(f'RESULT_JSON: {json.dumps(result)}')


if __name__ == '__main__':
    main()
