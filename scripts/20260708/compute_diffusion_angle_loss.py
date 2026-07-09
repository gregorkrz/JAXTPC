#!/usr/bin/env python
"""
Per-plane and per-frequency loss between a noisy GT track (truth transverse
diffusion) and the same track re-simulated with a wrong diffusion constant
(--diffusion-factor x truth), across a set of (theta, alpha) track direction
angles.

Motivation: for small (theta, alpha) and for theta=alpha=50deg the D_transverse
loss looks "normal", but theta=50/alpha=45 looks worse/different. This script
produces the raw per-plane MSE and per-frequency (Sobolev-weighted) loss
numbers needed to compare those cases; a follow-up script will visualize them.

Track direction convention matches the (theta, alpha) diffusion-vs-angle study
in src/jobs/submit_jobs_loss_studies.py (submit_diffusion_angle_theta_alpha_study):
    theta = azimuthal angle in the XY plane from the -x axis
    alpha = "lift" angle from the XY plane toward +z
    dx = -cos(theta)*cos(alpha), dy = sin(theta)*cos(alpha), dz = sin(alpha)

For each (theta, alpha) pair:
  1. Build one muon track at that direction (fixed momentum, fixed start x).
  2. Simulate once at truth diffusion_trans_cm2_us -> clean signal.
  3. Add N independent noise realizations -> N noisy "target" signals
     (same apply_noise_to_gt idiom as run_optimization.py).
  4. Simulate once at --diffusion-factor * truth diffusion_trans_cm2_us
     -> the "wrong" simulated signal (no noise).
  5. Per plane, per noise realization, per --adc-cutoffs threshold: MSE and
     Sobolev per-frequency loss map between the wrong-diffusion signal and
     the noisy truth-diffusion target (both signals masked to zero wherever
     the noisy target is below that ADC threshold — run_optimization.py's
     ADC-mask convention). Also computed once against the clean (no noise)
     truth-diffusion signal, as a noise-free reference.

Output: one pickle with per-(theta,alpha) x per-plane arrays (raw, not
plotted) — see `results` structure built in `main()`.

Usage
-----
  .venv/bin/python scripts/20260708/compute_diffusion_angle_loss.py \\
      --angle-pairs 10,10 50,50 50,45 --n-noise 20

  .venv/bin/python scripts/20260708/compute_diffusion_angle_loss.py \\
      --angle-pairs 50,45 --n-noise 50 --diffusion-factor 0.95 \\
      --output results/20260708_diffusion_angle/theta50_alpha45.pkl
"""
import argparse
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.particle_generator import generate_muon_track
from tools.loader import build_deposit_data
from tools.noise import generate_noise
from tools.losses import make_sobolev_weight, sobolev_loss_single, sobolev_fourier_map_single
from optlib.constants import (
    CONFIG_PATH, GT_LIFETIME_US, GT_VELOCITY_CM_US, MAX_ACTIVE_BUCKETS, SOBOLEV_MAX_PAD,
)

_ANGLE_START_X_MM = 1900.0
_ANGLE_MOMENTUM_MEV = 400.0


def _apply_noise_to_gt(gt_arrays, simulator, noise_scale, noise_key):
    """Verbatim of run_optimization.py's apply_noise_to_gt (not imported — that
    module is a CLI driver, not meant to be imported as a library)."""
    cfg = simulator.config
    noise_dict = generate_noise(cfg, key=noise_key)
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


def _output_signal_labels(cfg):
    """[(vol_idx, plane_idx, plane_name), ...] matching simulator.forward()'s output order."""
    labels = []
    for v in range(cfg.n_volumes):
        plane_names = cfg.plane_names[v]
        for p in range(cfg.volumes[v].n_planes):
            labels.append((v, p, plane_names[p]))
    return labels


def _direction_from_theta_alpha(theta_deg, alpha_deg):
    theta_rad = math.radians(theta_deg)
    alpha_rad = math.radians(alpha_deg)
    dx = -math.cos(theta_rad) * math.cos(alpha_rad)
    dy = math.sin(theta_rad) * math.cos(alpha_rad)
    dz = math.sin(alpha_rad)
    return (dx, dy, dz)


def _parse_angle_pairs(raw_pairs):
    pairs = []
    for raw in raw_pairs:
        theta_str, alpha_str = raw.split(',')
        pairs.append((float(theta_str), float(alpha_str)))
    return pairs


def build_simulator(max_deposits, num_buckets):
    detector_config = generate_detector(CONFIG_PATH)
    sim = DetectorSimulator(
        detector_config, differentiable=True, n_segments=max_deposits,
        use_bucketed=True, max_active_buckets=num_buckets,
        include_noise=False, include_electronics=False,
        include_track_hits=False, include_digitize=False,
    )
    sim.warm_up()
    return sim


def build_deposits(sim, direction, momentum_mev, start_x_mm, step_size_mm, track_id=1):
    track = generate_muon_track(
        start_position_mm=(start_x_mm, 0.0, 0.0), direction=direction,
        kinetic_energy_mev=momentum_mev, step_size_mm=step_size_mm, track_id=track_id)
    deposits = build_deposit_data(
        track['position'], track['de'], track['dx'], sim.config,
        theta=track['theta'], phi=track['phi'], track_ids=track['track_id'])
    return deposits, len(track['position'])


def _apply_adc_mask_single(a, b, cutoff):
    """Zero both a (sim) and b (target) where |b| < cutoff. No-op when cutoff <= 0.
    Verbatim convention of run_optimization.py's _apply_adc_mask, single-plane form."""
    if cutoff <= 0.0:
        return a, b
    mask = jnp.abs(b) >= cutoff
    return jnp.where(mask, a, 0.0), jnp.where(mask, b, 0.0)


def _plane_metrics(a, b, spectral_weight, fourier_out_size, adc_cutoff=0.0):
    """MSE, scalar Sobolev loss, the per-frequency Sobolev loss map C(f) = power * weight,
    and the raw (unweighted) per-frequency power spectrum of the difference, between two
    plane signals `a` (sim) and `b` (target) — same normalization as tools/losses.py.
    `adc_cutoff` zeros both arrays wherever |b| < cutoff first (run_optimization.py's
    ADC-threshold masking): with most of a plane below-threshold "background", an
    unmasked comparison is dominated by that background and can mask real differences
    concentrated in the track region.

    C(f) and the raw power spectrum are worth keeping separately: the Sobolev weight
    1/(f^2+eps)^2 blows up near f=0 by design, so C(f) is almost always concentrated in
    a handful of low-frequency pixels; the raw power spectrum has no such blowup and can
    show structure elsewhere in frequency space that C(f) alone hides."""
    a, b = _apply_adc_mask_single(a, b, adc_cutoff)
    norm = jnp.sum(jnp.abs(b)) + 1e-12
    mse = float(jnp.mean(((a - b) / norm) ** 2))
    sob = float(sobolev_loss_single(a, b, spectral_weight))
    c_shift, pwr_shift = sobolev_fourier_map_single(a, b, spectral_weight, out_size=fourier_out_size)
    return mse, sob, np.asarray(c_shift), np.asarray(pwr_shift)


def eval_angle_pair(theta_deg, alpha_deg, sim, jitted_forward, gt_params, diff_params,
                     spectral_weights, momentum_mev, start_x_mm, step_size_mm,
                     n_noise, noise_scale, base_noise_key, pair_idx, fourier_out_size,
                     adc_cutoffs):
    direction = _direction_from_theta_alpha(theta_deg, alpha_deg)
    deposits, n_deposits = build_deposits(sim, direction, momentum_mev, start_x_mm, step_size_mm)

    clean_gt = tuple(jitted_forward(gt_params, deposits))
    diff_sig = tuple(jitted_forward(diff_params, deposits))

    n_planes = len(clean_gt)
    mse_per_plane = {c: [np.empty(n_noise, dtype=np.float64) for _ in range(n_planes)] for c in adc_cutoffs}
    sobolev_per_plane = {c: [np.empty(n_noise, dtype=np.float64) for _ in range(n_planes)] for c in adc_cutoffs}
    freq_maps_per_plane = {c: [[] for _ in range(n_planes)] for c in adc_cutoffs}
    power_maps_per_plane = {c: [[] for _ in range(n_planes)] for c in adc_cutoffs}

    pair_key = jax.random.fold_in(base_noise_key, pair_idx)
    for i in range(n_noise):
        noise_key = jax.random.fold_in(pair_key, i)
        noisy_gt = _apply_noise_to_gt(clean_gt, sim, noise_scale, noise_key)
        for p in range(n_planes):
            for c in adc_cutoffs:
                mse, sob, c_shift, pwr_shift = _plane_metrics(
                    diff_sig[p], noisy_gt[p], spectral_weights[p], fourier_out_size, adc_cutoff=c)
                mse_per_plane[c][p][i] = mse
                sobolev_per_plane[c][p][i] = sob
                freq_maps_per_plane[c][p].append(c_shift)
                power_maps_per_plane[c][p].append(pwr_shift)

    # No-noise reference: diff-sim vs the clean (un-noised) truth-D_T signal directly —
    # a sanity check independent of any noise realization, since diff_sig/clean_gt are
    # already computed above regardless of n_noise.
    no_noise = {c: [_plane_metrics(diff_sig[p], clean_gt[p], spectral_weights[p], fourier_out_size, adc_cutoff=c)
                     for p in range(n_planes)]
                for c in adc_cutoffs}

    per_plane = {}
    for p in range(n_planes):
        per_plane[p] = {}
        for c in adc_cutoffs:
            maps = np.stack(freq_maps_per_plane[c][p], axis=0)  # (n_noise, out_h, out_w)
            pwr_maps = np.stack(power_maps_per_plane[c][p], axis=0)
            no_noise_mse, no_noise_sob, no_noise_freq, no_noise_pwr = no_noise[c][p]
            per_plane[p][c] = dict(
                mse_per_realization=mse_per_plane[c][p],
                sobolev_per_realization=sobolev_per_plane[c][p],
                # Weighted per-frequency Sobolev loss contribution C(f) = power * weight.
                freq_map_mean=maps.mean(axis=0),
                freq_map_std=maps.std(axis=0),
                freq_maps_all=maps,  # (n_noise, out_h, out_w) — every individual realization's C(f)
                freq_map_no_noise=no_noise_freq,
                # Raw (unweighted) per-frequency power spectrum of the difference — no
                # 1/(f^2+eps)^2 blowup near f=0, so structure elsewhere is visible.
                power_map_mean=pwr_maps.mean(axis=0),
                power_map_std=pwr_maps.std(axis=0),
                power_maps_all=pwr_maps,
                power_map_no_noise=no_noise_pwr,
                mse_no_noise=no_noise_mse,
                sobolev_no_noise=no_noise_sob,
            )

    return dict(
        theta_deg=theta_deg, alpha_deg=alpha_deg, direction=direction,
        n_deposits=n_deposits,
        per_plane=per_plane,  # {plane_idx: {adc_cutoff: {...}}}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--angle-pairs', nargs='+', required=True, metavar='THETA,ALPHA',
                         help='One or more "theta_deg,alpha_deg" pairs, e.g. --angle-pairs 10,10 50,50 50,45')
    parser.add_argument('--n-noise', type=int, required=True,
                         help='Number of independent noise realizations per angle pair.')
    parser.add_argument('--diffusion-factor', type=float, default=0.95,
                         help='Wrong-simulation diffusion_trans_cm2_us as a fraction of truth (default 0.95).')
    parser.add_argument('--noise-scale', type=float, default=1.0,
                         help='Noise amplitude multiplier passed to apply_noise_to_gt (default 1.0).')
    parser.add_argument('--noise-seed', type=int, default=0,
                         help='Base PRNG seed for noise realizations (default 0).')
    parser.add_argument('--adc-cutoffs', type=float, nargs='+', default=[0.0, 50.0],
                         help='ADC thresholds to mask both signals with before computing loss '
                              '(run_optimization.py convention: zero both sim and target wherever '
                              '|target| < cutoff). Default 0 (no cutoff) and 50 — a plane is mostly '
                              'below-threshold background, so an uncut comparison can wash out real '
                              'differences concentrated in the track region; each cutoff is computed '
                              'and stored separately, selectable in the dashboard.')
    parser.add_argument('--momentum-mev', type=float, default=_ANGLE_MOMENTUM_MEV,
                         help=f'Muon kinetic energy in MeV (default {_ANGLE_MOMENTUM_MEV}, matches the '
                              'existing theta/alpha diffusion study).')
    parser.add_argument('--start-x-mm', type=float, default=_ANGLE_START_X_MM,
                         help=f'Track start x position in mm (default {_ANGLE_START_X_MM}).')
    parser.add_argument('--step-size-mm', type=float, default=1.0, help='Track propagation step size in mm.')
    parser.add_argument('--max-deposits', type=int, default=5000,
                         help='Simulator n_segments / deposit padding (default 5000; see CLAUDE.md '
                              'deposit-count-overflow note if momentum > ~1000 MeV).')
    parser.add_argument('--num-buckets', type=int, default=MAX_ACTIVE_BUCKETS)
    parser.add_argument('--sobolev-max-pad', type=int, default=SOBOLEV_MAX_PAD)
    parser.add_argument('--sobolev-s', type=float, default=2.0)
    parser.add_argument('--fourier-out-size', type=int, nargs=2, default=(128, 128), metavar=('H', 'W'),
                         help='Bilinear-resize per-frequency loss maps to this size before saving '
                              '(default 128x128; pass the native plane shape to disable resizing). '
                              'Every individual noise realization\'s map is saved (not just mean/std), '
                              'so output size scales with n_noise * H * W * n_planes * n_angle_pairs — '
                              'lower this if --n-noise is large and the pickle gets unwieldy.')
    parser.add_argument('--output', default=None,
                         help='Output pickle path (default: $RESULTS_DIR/20260708_diffusion_angle_loss/'
                              'diffusion_angle_loss.pkl)')
    args = parser.parse_args()

    angle_pairs = _parse_angle_pairs(args.angle_pairs)
    fourier_out_size = tuple(args.fourier_out_size)

    output_path = args.output
    if output_path is None:
        results_dir = os.environ.get('RESULTS_DIR', 'results')
        output_path = os.path.join(results_dir, '20260708_diffusion_angle_loss', 'diffusion_angle_loss.pkl')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    print(f'Building simulator (n_segments={args.max_deposits:,}, num_buckets={args.num_buckets:,})...')
    sim = build_simulator(args.max_deposits, args.num_buckets)
    jitted_forward = jax.jit(sim.forward)
    labels = _output_signal_labels(sim.config)

    gt_params = sim.default_sim_params._replace(
        lifetime_us=jnp.array(GT_LIFETIME_US), velocity_cm_us=jnp.array(GT_VELOCITY_CM_US))
    truth_diffusion_trans = float(gt_params.diffusion_trans_cm2_us)
    diff_params = gt_params._replace(
        diffusion_trans_cm2_us=jnp.array(args.diffusion_factor * truth_diffusion_trans))
    print(f'truth diffusion_trans_cm2_us={truth_diffusion_trans:.6g}, '
          f'wrong-sim diffusion_trans_cm2_us={float(diff_params.diffusion_trans_cm2_us):.6g} '
          f'(factor={args.diffusion_factor})')

    # spectral_weight depends only on (H, W); the plane shapes are fixed by the config,
    # so build a probe track once to get per-plane shapes and cache weights by shape.
    probe_direction = _direction_from_theta_alpha(*angle_pairs[0])
    probe_deposits, _ = build_deposits(sim, probe_direction, args.momentum_mev, args.start_x_mm, args.step_size_mm)
    probe_signals = tuple(jitted_forward(gt_params, probe_deposits))
    spectral_weights = tuple(
        make_sobolev_weight(a.shape[0], a.shape[1], max_pad=args.sobolev_max_pad, s=args.sobolev_s)
        for a in probe_signals)

    base_noise_key = jax.random.PRNGKey(args.noise_seed)

    results = {}
    t0 = time.time()
    for pair_idx, (theta_deg, alpha_deg) in enumerate(angle_pairs):
        t1 = time.time()
        print(f'[{pair_idx + 1}/{len(angle_pairs)}] theta={theta_deg}, alpha={alpha_deg} ...')
        results[(theta_deg, alpha_deg)] = eval_angle_pair(
            theta_deg, alpha_deg, sim, jitted_forward, gt_params, diff_params,
            spectral_weights, args.momentum_mev, args.start_x_mm, args.step_size_mm,
            args.n_noise, args.noise_scale, base_noise_key, pair_idx, fourier_out_size,
            args.adc_cutoffs)
        print(f'    done in {time.time() - t1:.1f}s')

    output = dict(
        command=' '.join(sys.argv),
        angle_pairs=angle_pairs,
        labels=labels,  # [(vol_idx, plane_idx, plane_name), ...], matches per_plane dict keys 0..n_planes-1
        truth_diffusion_trans_cm2_us=truth_diffusion_trans,
        diffusion_factor=args.diffusion_factor,
        diff_diffusion_trans_cm2_us=float(diff_params.diffusion_trans_cm2_us),
        noise_scale=args.noise_scale,
        n_noise=args.n_noise,
        noise_seed=args.noise_seed,
        adc_cutoffs=list(args.adc_cutoffs),
        momentum_mev=args.momentum_mev,
        start_x_mm=args.start_x_mm,
        step_size_mm=args.step_size_mm,
        sobolev_max_pad=args.sobolev_max_pad,
        sobolev_s=args.sobolev_s,
        fourier_out_size=fourier_out_size,
        # fftshift'd frequency axes are always linspace(-0.5, 0.5, out_size) regardless of
        # plane shape/padding — fftfreq is normalized (cycles/sample), independent of H_pad/W_pad.
        freq_axis_note='freq axis per dim is np.linspace(-0.5, 0.5, out_size[dim])',
        results=results,  # {(theta_deg, alpha_deg): {..., 'per_plane': {plane_idx: {adc_cutoff: {...}}}}}
    )
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    print(f'Saved {output_path}  (total {time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
