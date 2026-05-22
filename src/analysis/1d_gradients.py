#!/usr/bin/env python
"""
Sweep one simulation parameter over a range around its ground-truth value and
record the loss and gradient at each point, summed across all requested tracks.

For each run the script evaluates 2N+1 evenly spaced points (N to the left,
ground truth, N to the right) over ±RANGE_FRAC of the GT value, then saves a
pickle with parameter values, loss values, and signed gradients.

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

Track specification
-------------------
  --tracks 'name:dx,dy,dz:mom_mev+name2:dx2,dy2,dz2:mom_mev2'
  Multiple tracks are '+'-separated; loss is summed across them.

Loss names
----------
  default                     sobolev_loss_geomean_log1p  (the optimisation loss)
  sobolev_loss                plain Sobolev loss
  sobolev_loss_geomean_log1p  geometric-mean log1p Sobolev loss
  mse_loss                    normalised MSE

Usage examples
--------------
    python 1d_gradients.py
    python 1d_gradients.py --param lifetime_us --N 5
    python 1d_gradients.py --param recomb_alpha --N 4 --noise-scale 1.0
    python 1d_gradients.py --param diffusion_trans_cm2_us --N 10 --range-frac 0.5 \\
        --tracks 'trk1:1,1,1:1000+trk2:-1,1,1:500'
    python 1d_gradients.py --param recomb_beta_90 --fixed-param recomb_alpha \\
        --fixed-value 0.905

Output (one file per run)
-------------------------
    results/1d_gradients/{loss_name}_N{N}_range{RANGE_FRAC}_{param_name}_{tracks_tag}[_noise{scale}][_fixed_...].pkl

Each pickle contains a dict with keys:
    param_name, param_gt, scale, p_n_gt,
    param_values, p_n_values, factors,
    loss_values, grad_values, grad_times_s,
    per_track_loss_values, per_track_grad_values,
    loss_name, N, range_frac,
    track_specs, noise_scale, noise_seed,
    sobolev_max_pad,
    fixed_param, fixed_value

With --store-arrays:
    per_track_gt_arrays, per_track_sim_arrays

With --store-per-pixel-loss-and-grad:
    per_track_pixel_loss  -- (sim-gt)^2 arrays per sweep point, per track, per plane
    per_track_pixel_grad  -- ∂loss/∂sim[w,t] arrays; same shape as sim arrays
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

import argparse
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
    sobolev_loss_single,
)

# JIT-compiled per-plane loss; reuses one compiled kernel per distinct array shape.
_plane_loss_jit = jax.jit(sobolev_loss_single)
from tools.noise import generate_noise
from tools.particle_generator import generate_muon_track
from tools.simulation import DetectorSimulator

# ── Constants ──────────────────────────────────────────────────────────────────
GT_LIFETIME_US    = 10_000.0
GT_VELOCITY_CM_US = 0.160

CONFIG_PATH        = 'config/cubic_wireplane_config.yaml'
MAX_ACTIVE_BUCKETS = 1000
DETECTOR_BOUNDS_MM = ()

_JAX_CACHE_DIR = os.environ.get('JAX_COMPILATION_CACHE_DIR',
                                os.path.expanduser('~/.cache/jax_compilation_cache'))
jax.config.update('jax_compilation_cache_dir', _JAX_CACHE_DIR)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)
_RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')

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

# 'default' maps to the standard optimisation loss
LOSS_ALIAS = {'default': 'sobolev_loss_geomean_log1p'}
VALID_LOSSES = ('default', 'sobolev_loss', 'sobolev_loss_geomean_log1p', 'mse_loss')

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

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--param', default='velocity_cm_us', choices=VALID_PARAMS,
                   help='Parameter to vary (default: velocity_cm_us)')
    p.add_argument('--loss', default='default', choices=VALID_LOSSES,
                   help='Loss function (default: "default" = sobolev_loss_geomean_log1p)')
    p.add_argument('--N', type=int, default=2,
                   help='Points on each side of GT (default: 2 → 2N+1 total)')
    p.add_argument('--range-frac', type=float, default=0.05,
                   help='Half-range as fraction of GT value (default: 0.05 = ±5%%)')
    p.add_argument('--tracks', default='diagonal:1,1,1:1000',
                   help='"+"-separated track specs  name:dx,dy,dz:mom_mev '
                        '(default: diagonal:1,1,1:1000)')
    p.add_argument('--step-size', type=float, default=1.0,
                   help='Track step size in mm for deposit generation (default: 1.0)')
    p.add_argument('--max-deposits', type=int, default=5000,
                   help='Max deposits per track (default: 5000)')
    p.add_argument('--noise-scale', type=float, default=0.0,
                   help='Noise amplitude (0 = no noise, 1.0 = realistic; default: 0)')
    p.add_argument('--noise-seed', type=int, default=42,
                   help='RNG seed for noise (default: 42)')
    p.add_argument('--sobolev-max-pad', type=int, default=128,
                   help='Max padding for Sobolev weight construction (default: 128)')
    p.add_argument('--adc-cutoff', type=float, default=0.0,
                   help='Zero out pixels where |GT| < cutoff before loss (default: 0 = off)')
    p.add_argument('--adc-cutoffs', default=None,
                   help='Comma-separated list of ADC cutoffs to sweep in one invocation. '
                        'Each cutoff writes its own output file. '
                        'Overrides --adc-cutoff; cannot be combined with --output.')
    p.add_argument('--store-arrays', action='store_true',
                   help='Store full 2D signal arrays (all planes) at each sweep point per track')
    p.add_argument('--store-per-pixel-loss-and-grad', action='store_true',
                   help='Store per-pixel (sim-gt)^2 loss and ∂loss/∂sim[w,t] sensitivity maps '
                        'at each sweep point')
    p.add_argument('--store-per-plane-loss', action='store_true',
                   help='Store sobolev_loss_single for each wire plane at every sweep point')
    p.add_argument('--output', default=None,
                   help='Explicit output pkl path (overrides auto-generated name)')
    p.add_argument('--results-dir', default=os.path.join(_RESULTS_DIR, '1d_gradients'),
                   help='Output directory used when --output is not set '
                        '(default: results/1d_gradients)')
    p.add_argument('--fixed-param', default=None, choices=VALID_PARAMS,
                   help='Fix this parameter to --fixed-value instead of GT')
    p.add_argument('--fixed-value', type=float, default=None,
                   help='Physical value for --fixed-param')
    return p.parse_args()


# ── Track parsing ──────────────────────────────────────────────────────────────

def parse_tracks(tracks_str):
    """Parse '+'-separated track specs into list of dicts.

    Each spec: name:dx,dy,dz:mom_mev  or  name:dx,dy,dz:mom_mev:sx,sy,sz
               or  name:dx,dy,dz:mom_mev:sx,sy,sz:frac_start,frac_end
    The optional 4th field sets start_position_mm (default: 0,0,0).
    The optional 5th field sets deposit fraction [frac_start, frac_end] in [0,1]
    (default: 0.0,1.0 = full track). Requires the 4th field to be present.
    """
    specs = []
    for item in tracks_str.split('+'):
        item = item.strip()
        if not item:
            continue
        parts = item.split(':')
        if len(parts) not in (3, 4, 5):
            raise ValueError(
                f'Track must be name:dx,dy,dz:mom_mev[:sx,sy,sz[:frac_start,frac_end]], got {item!r}')
        name = parts[0].strip()
        direction = tuple(float(x) for x in parts[1].split(','))
        if len(direction) != 3:
            raise ValueError(f'Direction must have 3 components in {item!r}')
        momentum_mev = float(parts[2])
        if len(parts) >= 4:
            start = tuple(float(x) for x in parts[3].split(','))
            if len(start) != 3:
                raise ValueError(f'Start position must have 3 components in {item!r}')
        else:
            start = (0.0, 0.0, 0.0)
        if len(parts) == 5:
            frac = tuple(float(x) for x in parts[4].split(','))
            if len(frac) != 2 or not (0.0 <= frac[0] < frac[1] <= 1.0):
                raise ValueError(
                    f'Fraction must be frac_start,frac_end with 0<=start<end<=1, got {parts[4]!r}')
            frac_start, frac_end = frac
        else:
            frac_start, frac_end = 0.0, 1.0
        specs.append(dict(name=name, direction=direction, momentum_mev=momentum_mev,
                          start_position_mm=start, frac_start=frac_start, frac_end=frac_end))
    if not specs:
        raise ValueError('--tracks produced no entries')
    return specs


# ── Parameter setter factory ───────────────────────────────────────────────────

def make_param_setter(param_name, gt_params, recomb_model):
    """Return (setter, gt_val, scale).  setter(p_n) -> SimParams."""
    rp    = gt_params.recomb_params
    scale = TYPICAL_SCALES[param_name]

    if param_name == 'velocity_cm_us':
        gt_val = float(gt_params.velocity_cm_us)
        setter = lambda p_n: gt_params._replace(velocity_cm_us=p_n * scale)
    elif param_name == 'lifetime_us':
        gt_val = float(gt_params.lifetime_us)
        setter = lambda p_n: gt_params._replace(lifetime_us=p_n * scale)
    elif param_name == 'diffusion_trans_cm2_us':
        gt_val = float(gt_params.diffusion_trans_cm2_us)
        setter = lambda p_n: gt_params._replace(diffusion_trans_cm2_us=p_n * scale)
    elif param_name == 'diffusion_long_cm2_us':
        gt_val = float(gt_params.diffusion_long_cm2_us)
        setter = lambda p_n: gt_params._replace(diffusion_long_cm2_us=p_n * scale)
    elif param_name == 'recomb_alpha':
        gt_val = float(rp.alpha)
        setter = lambda p_n: gt_params._replace(recomb_params=rp._replace(alpha=p_n * scale))
    elif param_name == 'recomb_beta':
        if recomb_model != 'modified_box':
            raise ValueError(f'recomb_beta requires modified_box model, got {recomb_model!r}')
        gt_val = float(rp.beta)
        setter = lambda p_n: gt_params._replace(recomb_params=rp._replace(beta=p_n * scale))
    elif param_name == 'recomb_beta_90':
        if recomb_model != 'emb':
            raise ValueError(f'recomb_beta_90 requires emb model, got {recomb_model!r}')
        gt_val = float(rp.beta_90)
        setter = lambda p_n: gt_params._replace(recomb_params=rp._replace(beta_90=p_n * scale))
    elif param_name == 'recomb_R':
        if recomb_model != 'emb':
            raise ValueError(f'recomb_R requires emb model, got {recomb_model!r}')
        gt_val = float(rp.R)
        setter = lambda p_n: gt_params._replace(recomb_params=rp._replace(R=p_n * scale))
    else:
        raise ValueError(f'Unknown param {param_name!r}')

    return setter, gt_val, scale


# ── Noise helper ───────────────────────────────────────────────────────────────

def apply_noise(gt_arrays, simulator, noise_scale, noise_key):
    cfg        = simulator.config
    noise_dict = generate_noise(cfg, key=noise_key)
    n_readouts = cfg.volumes[0].n_planes if simulator._readout_type == 'wire' else 1
    noisy = []
    for v in range(cfg.n_volumes):
        for plane in range(n_readouts):
            gt    = gt_arrays[v * n_readouts + plane]
            noise = noise_dict[(v, plane)] * noise_scale
            if noise.shape[0] < gt.shape[0]:
                noise = jnp.pad(noise, ((0, gt.shape[0] - noise.shape[0]), (0, 0)))
            noisy.append(gt + noise)
    return tuple(noisy)


# ── Loss builder ───────────────────────────────────────────────────────────────

def build_value_and_grad(loss_name, simulator, make_params, n_planes):
    """Return a single JIT-compiled fn(p_n, deposits, gt_arrays, weights).

    Compiled once; different tracks just pass different (deposits, gt_arrays,
    weights) as dynamic JAX-array arguments — no retracing.
    planes is fixed by n_planes and captured as a static Python tuple.
    """
    planes = tuple(range(n_planes))

    def fwd(p_n, deposits):
        return simulator.forward(make_params(p_n), deposits)

    def _mask(arrays, masks):
        return tuple(jnp.where(m, a, 0.0) for a, m in zip(arrays, masks))

    if loss_name == 'sobolev_loss':
        def fn(p_n, deposits, gt_arrays, weights, masks):
            sim = _mask(fwd(p_n, deposits), masks)
            gt  = _mask(gt_arrays, masks)
            return sobolev_loss(sim, gt, weights, planes)
    elif loss_name == 'sobolev_loss_geomean_log1p':
        def fn(p_n, deposits, gt_arrays, weights, masks):
            sim = _mask(fwd(p_n, deposits), masks)
            gt  = _mask(gt_arrays, masks)
            return sobolev_loss_geomean_log1p(sim, gt, weights, planes)
    elif loss_name == 'mse_loss':
        def fn(p_n, deposits, gt_arrays, weights, masks):
            sim   = _mask(fwd(p_n, deposits), masks)
            gt    = _mask(gt_arrays, masks)
            total = jnp.zeros(())
            for pr, g in zip(sim, gt):
                norm  = jnp.sum(jnp.abs(g)) + 1e-12
                total = total + jnp.mean(((pr - g) / norm) ** 2)
            return total
    else:
        raise ValueError(f'Unknown loss {loss_name!r}')

    return jax.jit(jax.value_and_grad(fn))


def build_pixel_grad_fn(loss_name, n_planes):
    """Return a JIT-compiled fn(sim_arrays, gt_arrays, weights, masks) -> grad_arrays.

    Computes ∂loss/∂sim[w,t] for every pixel — how much the loss changes if that
    simulated pixel is perturbed.  For MSE this is 2*(sim-gt)/norm; for Sobolev it
    is the frequency-weighted residual back-transformed to pixel space.
    Returns a tuple of arrays with the same shapes as sim_arrays.
    """
    planes = tuple(range(n_planes))

    def _mask(arrays, masks):
        return tuple(jnp.where(m, a, 0.0) for a, m in zip(arrays, masks))

    if loss_name == 'sobolev_loss':
        def fn(sim_arrays, gt_arrays, weights, masks):
            return sobolev_loss(_mask(sim_arrays, masks), _mask(gt_arrays, masks), weights, planes)
    elif loss_name == 'sobolev_loss_geomean_log1p':
        def fn(sim_arrays, gt_arrays, weights, masks):
            return sobolev_loss_geomean_log1p(_mask(sim_arrays, masks), _mask(gt_arrays, masks), weights, planes)
    elif loss_name == 'mse_loss':
        def fn(sim_arrays, gt_arrays, weights, masks):
            sim = _mask(sim_arrays, masks)
            gt  = _mask(gt_arrays, masks)
            total = jnp.zeros(())
            for pr, g in zip(sim, gt):
                norm  = jnp.sum(jnp.abs(g)) + 1e-12
                total = total + jnp.mean(((pr - g) / norm) ** 2)
            return total
    else:
        raise ValueError(f'Unknown loss {loss_name!r}')

    return jax.jit(jax.grad(fn))  # grad w.r.t. first arg (sim_arrays tuple)


# ── Output path ────────────────────────────────────────────────────────────────

def auto_output_path(args, loss_name, track_specs):
    tracks_tag  = (f'{len(track_specs)}tracks' if len(track_specs) > 1
                   else track_specs[0]['name'])
    if args.noise_scale > 0.0:
        seed_suffix = f'_seed{args.noise_seed}' if args.noise_seed != 42 else ''
        noise_tag = f'_noise{args.noise_scale:.3g}'.replace('.', 'p') + seed_suffix
    else:
        noise_tag = ''
    cutoff_tag  = f'_cutoff{args.adc_cutoff:.3g}'.replace('.', 'p') if args.adc_cutoff > 0.0 else ''
    fixed_tag   = f'_fixed_{args.fixed_param}{args.fixed_value}' if args.fixed_param else ''
    range_tag   = f'_range{args.range_frac:.3g}'.replace('.', 'p')
    perplane_tag = '_perplane' if args.store_per_plane_loss else ''
    name = (f'{loss_name}_N{args.N}{range_tag}_{args.param}'
            f'_{tracks_tag}{noise_tag}{cutoff_tag}{perplane_tag}{fixed_tag}.pkl')
    return os.path.join(args.results_dir, name)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if (args.fixed_param is None) != (args.fixed_value is None):
        raise ValueError('--fixed-param and --fixed-value must be given together')

    loss_name   = LOSS_ALIAS.get(args.loss, args.loss)
    track_specs = parse_tracks(args.tracks)

    # ── Resolve cutoff list ────────────────────────────────────────────────────
    if args.adc_cutoffs is not None:
        if args.output is not None:
            raise ValueError('--output cannot be combined with --adc-cutoffs')
        cutoffs_list = [float(x.strip()) for x in args.adc_cutoffs.split(',')]
    else:
        cutoffs_list = [args.adc_cutoff]

    print(f'JAX devices   : {jax.devices()}')
    print(f'Parameter     : {args.param}')
    print(f'Loss          : {loss_name}')
    print(f'N             : {args.N}  ({2 * args.N + 1} total points)')
    print(f'Range frac    : ±{args.range_frac:.1%}')
    print(f'Tracks        : {[t["name"] for t in track_specs]}')
    print(f'Step size     : {args.step_size} mm  max deposits={args.max_deposits:,}')
    print(f'Noise scale   : {args.noise_scale}')
    print(f'Sobolev pad   : {args.sobolev_max_pad}')
    print(f'ADC cutoffs   : {cutoffs_list}')
    print(f'Per-plane loss: {args.store_per_plane_loss}')
    if args.fixed_param:
        print(f'Fixed param   : {args.fixed_param} = {args.fixed_value}')

    # ── Skip already-computed cutoffs ─────────────────────────────────────────
    if args.output is not None:
        # Single explicit output path; existence check done once.
        if os.path.exists(args.output):
            print(f'Output already exists, skipping: {args.output}')
            return
        pending_cutoffs = cutoffs_list
    else:
        pending_cutoffs = []
        for cutoff in cutoffs_list:
            args.adc_cutoff = cutoff
            p = auto_output_path(args, loss_name, track_specs)
            if os.path.exists(p):
                print(f'  exists, skip cutoff={cutoff}: {p}')
            else:
                pending_cutoffs.append(cutoff)
        if not pending_cutoffs:
            print('All cutoffs already computed.')
            return
        if len(pending_cutoffs) < len(cutoffs_list):
            print(f'  {len(pending_cutoffs)}/{len(cutoffs_list)} cutoffs pending.')
    cutoffs_list = pending_cutoffs

    # ── Simulator ─────────────────────────────────────────────────────────────
    print('\nBuilding differentiable simulator...')
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

    print('Warming up JIT...')
    t0 = time.time()
    simulator.warm_up()
    print(f'Done ({time.time() - t0:.1f} s)')

    # ── Ground-truth params ────────────────────────────────────────────────────
    gt_params = simulator.default_sim_params._replace(
        lifetime_us    = jnp.array(GT_LIFETIME_US),
        velocity_cm_us = jnp.array(GT_VELOCITY_CM_US),
    )

    sweep_base_params = gt_params
    if args.fixed_param is not None:
        _fix_setter, _, _ = make_param_setter(args.fixed_param, gt_params, simulator.recomb_model)
        fix_scale = TYPICAL_SCALES[args.fixed_param]
        sweep_base_params = _fix_setter(args.fixed_value / fix_scale)
        print(f'Sweep base    : {args.fixed_param} overridden to {args.fixed_value}')

    _make_params, param_gt, scale = make_param_setter(
        args.param, sweep_base_params, simulator.recomb_model)
    p_n_gt = param_gt / scale
    print(f'GT value      : {args.param} = {param_gt:.6g}  '
          f'(scale={scale:.6g},  p_n_gt={p_n_gt:.6g})')

    # ── Build GT arrays for every track (cutoff-independent) ──────────────────
    print('\nGenerating GT arrays for all tracks...')
    per_track_base = []   # (name, deposits, gt_arrays, weights)  — no masks

    noise_base_key = jax.random.PRNGKey(args.noise_seed)
    for track_idx, ts in enumerate(track_specs):
        frac_start = ts.get('frac_start', 0.0)
        frac_end   = ts.get('frac_end',   1.0)
        print(f'  track {ts["name"]}  dir={ts["direction"]}  T={ts["momentum_mev"]} MeV'
              + (f'  frac=[{frac_start:.2g},{frac_end:.2g}]' if (frac_start, frac_end) != (0.0, 1.0) else ''))
        track = generate_muon_track(
            start_position_mm=ts['start_position_mm'],
            direction=ts['direction'],
            kinetic_energy_mev=ts['momentum_mev'],
            step_size_mm=args.step_size,
            track_id=1,
            detector_bounds_mm=DETECTOR_BOUNDS_MM,
        )
        if frac_start > 0.0 or frac_end < 1.0:
            n = track['position'].shape[0]
            i_start = int(n * frac_start)
            i_end   = max(i_start + 1, int(n * frac_end))
            print(f'    slicing deposits [{i_start}:{i_end}] of {n} total')
            track = {k: v[i_start:i_end] for k, v in track.items()}
        deposits = build_deposit_data(
            track['position'], track['de'], track['dx'], simulator.config,
            theta=track['theta'], phi=track['phi'],
            track_ids=track['track_id'],
        )
        n_total = sum(v.n_actual for v in deposits.volumes)
        print(f'    {n_total:,} deposits')

        gt_arrays = tuple(simulator.forward(gt_params, deposits))
        jax.block_until_ready(gt_arrays)

        if args.noise_scale > 0.0:
            track_noise_key = jax.random.fold_in(noise_base_key, track_idx)
            gt_arrays = apply_noise(gt_arrays, simulator, args.noise_scale, track_noise_key)

        weights = tuple(
            make_sobolev_weight(a.shape[0], a.shape[1], max_pad=args.sobolev_max_pad)
            for a in gt_arrays
        )
        all_abs = jnp.concatenate([jnp.abs(a).ravel() for a in gt_arrays])
        print(f'    Signal ADC: max={float(all_abs.max()):.3g}  '
              f'mean={float(all_abs.mean()):.3g}')
        per_track_base.append((ts['name'], deposits, gt_arrays, weights))

    # ── Plane name list (U1, V1, Y1, U2, V2, Y2) ─────────────────────────────
    cfg = simulator.config
    plane_names_all = []
    for v in range(cfg.n_volumes):
        for p in range(cfg.volumes[v].n_planes):
            plane_names_all.append(f'{cfg.plane_names[v][p]}{v + 1}')

    # GT arrays collected once for store_arrays (cutoff-independent)
    if args.store_arrays:
        per_track_gt_arrays = {
            name: [np.array(a, dtype=np.float32) for a in gt_arrs]
            for name, _, gt_arrs, _ in per_track_base
        }

    # ── Parameter grid ────────────────────────────────────────────────────────
    left_factors  = np.linspace(1.0 - args.range_frac, 1.0, args.N + 1)[:-1]
    right_factors = np.linspace(1.0, 1.0 + args.range_frac, args.N + 1)[1:]
    factors       = np.concatenate([left_factors, [1.0], right_factors])
    p_n_values    = p_n_gt * factors
    param_values  = p_n_values * scale

    print(f'\nParameter grid ({len(factors)} points, ±{args.range_frac:.0%} around GT):')
    for f, pn, v in zip(factors, p_n_values, param_values):
        marker = ' ← GT' if f == 1.0 else ''
        print(f'  factor={f:.6f}  p_n={pn:.6f}  {args.param}={v:.6g}{marker}')

    # ── Compile once (mask values are dynamic — no retrace per cutoff) ────────
    n_planes = len(per_track_base[0][2])
    vag_fn   = build_value_and_grad(loss_name, simulator, _make_params, n_planes)

    print('\nCompiling value_and_grad (once for all tracks and cutoffs)...')
    t0 = time.time()
    _, _dep0, _gt0, _wt0 = per_track_base[0]
    _allmask0 = tuple(jnp.ones(a.shape, dtype=bool) for a in _gt0)
    _ = vag_fn(jnp.array(p_n_values[0]), _dep0, _gt0, _wt0, _allmask0)
    jax.block_until_ready(_)
    _ = vag_fn(jnp.array(p_n_values[0]), _dep0, _gt0, _wt0, _allmask0)
    jax.block_until_ready(_)
    print(f'Done ({time.time() - t0:.1f} s)')

    if args.store_per_pixel_loss_and_grad:
        pixel_grad_fn = build_pixel_grad_fn(loss_name, n_planes)
        print('\nCompiling pixel_grad_fn...')
        t0 = time.time()
        _fwd0 = simulator.forward(_make_params(jnp.array(p_n_values[0])), _dep0)
        _ = pixel_grad_fn(_fwd0, _gt0, _wt0, _allmask0)
        jax.block_until_ready(_)
        _ = pixel_grad_fn(_fwd0, _gt0, _wt0, _allmask0)
        jax.block_until_ready(_)
        print(f'Done ({time.time() - t0:.1f} s)')

    # ── Loop over cutoffs ─────────────────────────────────────────────────────
    for cutoff in cutoffs_list:
        print(f'\n{"─" * 60}')
        print(f'ADC cutoff = {cutoff}  ({len(factors)} param points × {len(per_track_base)} tracks)')

        # Output path for this cutoff
        args.adc_cutoff = cutoff
        output_path = args.output or auto_output_path(args, loss_name, track_specs)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # Build masks for this cutoff and attach to per_track_data
        per_track_data = []
        for name, deposits, gt_arrays, weights in per_track_base:
            masks = tuple(
                jnp.ones(a.shape, dtype=bool) if cutoff == 0.0
                else (jnp.abs(a) >= cutoff)
                for a in gt_arrays
            )
            if cutoff > 0.0:
                n_total_px  = sum(m.size for m in masks)
                n_masked_px = sum(int(jnp.sum(~m)) for m in masks)
                print(f'  {name}: {n_masked_px:,}/{n_total_px:,} pixels zeroed '
                      f'({100 * n_masked_px / n_total_px:.1f}%)')
            per_track_data.append((name, deposits, gt_arrays, weights, masks))

        # Per-cutoff accumulators
        if args.store_arrays:
            per_track_sim_arrays = {name: [] for name, *_ in per_track_data}
        if args.store_per_pixel_loss_and_grad:
            per_track_pixel_loss = {name: [] for name, *_ in per_track_data}
            per_track_pixel_grad = {name: [] for name, *_ in per_track_data}
        if args.store_per_plane_loss:
            per_plane_loss_values = {
                name: {pname: [] for pname in plane_names_all}
                for name, *_ in per_track_data
            }

        per_track_loss  = {name: [] for name, *_ in per_track_data}
        per_track_grad  = {name: [] for name, *_ in per_track_data}
        total_loss_vals = []
        total_grad_vals = []
        grad_times_s    = []

        for i, (factor, p_n, pval) in enumerate(zip(factors, p_n_values, param_values)):
            p_n_arr = jnp.array(float(p_n))
            t0      = time.time()

            total_loss = 0.0
            total_grad = 0.0
            for tname, dep, gt_arrays, weights, masks in per_track_data:
                lv, gv = vag_fn(p_n_arr, dep, gt_arrays, weights, masks)
                jax.block_until_ready((lv, gv))
                lv, gv = float(lv), float(gv)
                per_track_loss[tname].append(lv)
                per_track_grad[tname].append(gv)
                total_loss += lv
                total_grad += gv
                need_fwd = args.store_arrays or args.store_per_pixel_loss_and_grad or args.store_per_plane_loss
                if need_fwd:
                    fwd_out = simulator.forward(_make_params(p_n_arr), dep)
                    jax.block_until_ready(fwd_out)
                    if args.store_arrays:
                        per_track_sim_arrays[tname].append(
                            [np.array(a, dtype=np.float32) for a in fwd_out]
                        )
                    if args.store_per_pixel_loss_and_grad:
                        pg = pixel_grad_fn(fwd_out, gt_arrays, weights, masks)
                        jax.block_until_ready(pg)
                        per_track_pixel_grad[tname].append(
                            [np.array(a, dtype=np.float32) for a in pg]
                        )
                        per_track_pixel_loss[tname].append(
                            [np.array((np.array(s, dtype=np.float32) - np.array(g, dtype=np.float32))**2)
                             for s, g in zip(fwd_out, gt_arrays)]
                        )
                    if args.store_per_plane_loss:
                        for pi, pname in enumerate(plane_names_all):
                            sim_m = jnp.where(masks[pi], fwd_out[pi], 0.0)
                            gt_m  = jnp.where(masks[pi], gt_arrays[pi], 0.0)
                            per_plane_loss_values[tname][pname].append(
                                float(_plane_loss_jit(sim_m, gt_m, weights[pi]))
                            )

            elapsed = time.time() - t0
            total_loss_vals.append(total_loss)
            total_grad_vals.append(total_grad)
            grad_times_s.append(elapsed)

            marker = ' ← GT' if factor == 1.0 else ''
            print(f'  [{i + 1:2d}/{len(factors)}] factor={factor:.6f}  '
                  f'p_n={p_n:.6f}  {args.param}={pval:.6g}  '
                  f'loss={total_loss:.4e}  grad={total_grad:+.4e}  '
                  f'({elapsed * 1e3:.0f} ms){marker}')

        result = dict(
            param_name             = args.param,
            param_gt               = param_gt,
            scale                  = scale,
            p_n_gt                 = p_n_gt,
            param_values           = list(param_values),
            p_n_values             = list(p_n_values),
            factors                = list(factors),
            loss_values            = total_loss_vals,
            grad_values            = total_grad_vals,
            grad_times_s           = grad_times_s,
            per_track_loss_values  = per_track_loss,
            per_track_grad_values  = per_track_grad,
            loss_name              = loss_name,
            N                      = args.N,
            range_frac             = args.range_frac,
            track_specs            = track_specs,
            noise_scale            = args.noise_scale,
            noise_seed             = args.noise_seed,
            adc_cutoff             = cutoff,
            sobolev_max_pad        = args.sobolev_max_pad,
            fixed_param            = args.fixed_param,
            fixed_value            = args.fixed_value,
            plane_names            = plane_names_all,
        )
        if args.store_arrays:
            result['per_track_gt_arrays']  = per_track_gt_arrays
            result['per_track_sim_arrays'] = per_track_sim_arrays
        if args.store_per_pixel_loss_and_grad:
            result['per_track_pixel_loss'] = per_track_pixel_loss
            result['per_track_pixel_grad'] = per_track_pixel_grad
        if args.store_per_plane_loss:
            result['per_plane_loss_values'] = per_plane_loss_values

        with open(output_path, 'wb') as f:
            pickle.dump(result, f)
        print(f'Saved: {output_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
