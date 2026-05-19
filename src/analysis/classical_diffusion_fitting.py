#!/usr/bin/env python
"""
Classical diffusion fitting on the canonical 15-track boundary ensemble.

The script:
  1) builds the 15-track ensemble (12 random boundary tracks + 3 fixed chords),
  2) samples deposits from each track,
  3) simulates each sampled deposit as an isolated single-step response,
  4) measures transverse and longitudinal signal widths vs drift time,
  5) fits sigma^2(t) = intercept + slope * t and reports
       D = slope / 2
     for both transverse and longitudinal diffusion.

Optional noisy-GT mode:
  - with --noise-scale > 0, widths are measured from a noisy GT image
    while the simulation response itself stays noise-free.

Outputs by default:
  - pkl files to:
      $RESULTS_DIR/classical_diffusion_fitting_20260518/
  - pdf files to:
      $PLOTS_DIR/classical_diffusion_fitting_20260518/

Plot-only mode:
  - pass --plots-only to regenerate PDFs from an existing fit summary pickle
    without rerunning simulation/fitting.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tools.geometry import generate_detector
from tools.loader import build_deposit_data
from tools.noise import generate_noise
from tools.particle_generator import generate_muon_track
from tools.random_boundary_tracks import (
    N_DEFAULT_BOUNDARY_MUONS,
    filter_track_inside_volumes,
    generate_random_boundary_tracks,
)
from tools.simulation import DetectorSimulator


_RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')
_PLOTS_DIR = os.environ.get('PLOTS_DIR', 'plots')

CONFIG_PATH = 'config/cubic_wireplane_config.yaml'


class _Vol:
    def __init__(self, ranges_cm):
        self.ranges_cm = ranges_cm


_VOLUMES = [
    _Vol([[-216.0, 0.0], [-216.0, 216.0], [-216.0, 216.0]]),
    _Vol([[0.0, 216.0], [-216.0, 216.0], [-216.0, 216.0]]),
]


@dataclass
class SegmentSample:
    track_name: str
    segment_idx: int
    position_mm: np.ndarray
    de_mev: float
    dx_mm: float
    theta_rad: float
    phi_rad: float


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        '--results-dir',
        default=os.path.join(_RESULTS_DIR, 'classical_diffusion_fitting_20260518'),
        help='Directory for pickle outputs '
             '(default: $RESULTS_DIR/classical_diffusion_fitting_20260518)',
    )
    p.add_argument(
        '--plots-dir',
        default=os.path.join(_PLOTS_DIR, 'classical_diffusion_fitting_20260518'),
        help='Directory for PDF outputs '
             '(default: $PLOTS_DIR/classical_diffusion_fitting_20260518)',
    )
    p.add_argument(
        '--n-boundary-tracks',
        type=int,
        default=N_DEFAULT_BOUNDARY_MUONS,
        help=f'Number of random boundary tracks before adding 3 fixed chords '
             f'(default: {N_DEFAULT_BOUNDARY_MUONS} -> total 15)',
    )
    p.add_argument('--track-seed', type=int, default=42,
                   help='RNG seed for boundary track generation (default: 42)')
    p.add_argument('--track-step-mm', type=float, default=1.0,
                   help='Track propagation step size in mm (default: 1.0)')
    p.add_argument('--max-segments-per-track', type=int, default=160,
                   help='Maximum sampled segments per track (default: 160)')
    p.add_argument('--sampling-mode', choices=('uniform', 'random'),
                   default='uniform',
                   help='Segment sampling strategy (default: uniform)')
    p.add_argument('--sampling-seed', type=int, default=123,
                   help='RNG seed for random segment sampling (default: 123)')
    p.add_argument('--plane', default='Y',
                   help='Wire plane used for fitting (default: Y)')
    p.add_argument('--n-segments', type=int, default=1,
                   help='Simulator n_segments in differentiable mode. '
                        'Use 1 for isolated single-step responses (default: 1)')
    p.add_argument('--plots-only', action='store_true',
                   help='Skip simulation/fitting and regenerate PDFs from a saved '
                        'fit summary pickle')
    p.add_argument('--summary-pkl', default=None,
                   help='Path to fit_summary.pkl for --plots-only mode '
                        '(default: <results-dir>/fit_summary.pkl)')
    p.add_argument('--noise-scale', type=float, default=0.0,
                   help='Noise amplitude multiplier for GT construction. '
                        '0.0 = no added noise (default). '
                        'When > 0, GT is noisy and simulation remains noise-free.')
    p.add_argument('--noise-seed', type=int, default=0,
                   help='Seed for GT noise draw used with --noise-scale (default: 0)')
    return p.parse_args()


def _sample_segment_indices(n_steps: int, max_segments: int, mode: str, rng: np.random.Generator):
    if n_steps <= 0:
        return np.zeros(0, dtype=np.int32)
    if n_steps <= max_segments:
        return np.arange(n_steps, dtype=np.int32)
    if mode == 'random':
        return np.sort(rng.choice(n_steps, size=max_segments, replace=False).astype(np.int32))
    # uniform
    return np.unique(np.linspace(0, n_steps - 1, num=max_segments, dtype=np.int32))


def _weighted_mean_variance(profile: np.ndarray):
    total = float(np.sum(profile))
    if not np.isfinite(total) or total <= 0.0:
        return np.nan, np.nan
    idx = np.arange(profile.shape[0], dtype=np.float64)
    mean = float(np.sum(idx * profile) / total)
    var = float(np.sum(((idx - mean) ** 2) * profile) / total)
    return mean, max(var, 0.0)


def _fit_sigma2_vs_time(drift_time_us: np.ndarray, sigma_cm: np.ndarray, quantity_name: str):
    x = np.asarray(drift_time_us, dtype=np.float64)
    y = np.asarray(sigma_cm, dtype=np.float64) ** 2

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3:
        return None

    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    y_pred = slope * x + intercept
    resid = y - y_pred

    dof = max(int(x.size - 2), 1)
    sse = float(np.sum(resid ** 2))
    mse = sse / dof
    ata_inv = np.linalg.inv(A.T @ A)
    cov = mse * ata_inv
    slope_err = float(np.sqrt(max(cov[0, 0], 0.0)))
    intercept_err = float(np.sqrt(max(cov[1, 1], 0.0)))

    y_mean = float(np.mean(y))
    sst = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - (sse / sst) if sst > 0 else np.nan

    return {
        'quantity': quantity_name,
        'n_points': int(x.size),
        'slope_cm2_per_us2': slope,
        'slope_err_cm2_per_us2': slope_err,
        'intercept_cm2': intercept,
        'intercept_err_cm2': intercept_err,
        'D_cm2_per_us': 0.5 * slope,
        'D_err_cm2_per_us': 0.5 * slope_err,
        'r2': float(r2),
        'sse': sse,
        'x_drift_time_us': x,
        'y_sigma2_cm2': y,
        'y_fit_sigma2_cm2': y_pred,
    }


def _select_plane_index(cfg, plane_name: str, volume_idx: int):
    plane_names = tuple(cfg.plane_names[volume_idx])
    target = plane_name.upper().strip()
    for pi, name in enumerate(plane_names):
        if str(name).upper() == target:
            return pi
    raise ValueError(f'Plane {plane_name!r} not found in volume {volume_idx} plane list {plane_names}')


def _make_single_deposit(track_sample: SegmentSample, sim_cfg):
    pos = np.asarray(track_sample.position_mm, dtype=np.float32).reshape(1, 3)
    de = np.asarray([track_sample.de_mev], dtype=np.float32)
    dx = np.asarray([track_sample.dx_mm], dtype=np.float32)
    theta = np.asarray([track_sample.theta_rad], dtype=np.float32)
    phi = np.asarray([track_sample.phi_rad], dtype=np.float32)
    tid = np.asarray([1], dtype=np.int32)
    return build_deposit_data(
        pos, de, dx, sim_cfg,
        theta=theta, phi=phi, track_ids=tid,
    )


def _active_volume_idx(deposits):
    for vi, vol in enumerate(deposits.volumes):
        if int(vol.n_actual) > 0:
            return vi
    return None


def _collect_samples(track_specs: list[dict[str, Any]], cfg, args):
    rng = np.random.default_rng(args.sampling_seed)
    all_samples: list[SegmentSample] = []
    track_manifest = []

    for spec in track_specs:
        track = generate_muon_track(
            start_position_mm=spec.get('start_position_mm', (0.0, 0.0, 0.0)),
            direction=tuple(spec['direction']),
            kinetic_energy_mev=float(spec['momentum_mev']),
            step_size_mm=args.track_step_mm,
            track_id=1,
        )
        track = filter_track_inside_volumes(track, cfg.volumes)
        n_steps = len(track['de'])
        if n_steps <= 0:
            track_manifest.append({
                'name': spec['name'],
                'momentum_mev': float(spec['momentum_mev']),
                'n_steps_total': 0,
                'n_steps_sampled': 0,
            })
            continue

        chosen = _sample_segment_indices(
            n_steps, args.max_segments_per_track, args.sampling_mode, rng)

        for idx in chosen:
            all_samples.append(SegmentSample(
                track_name=spec['name'],
                segment_idx=int(idx),
                position_mm=np.asarray(track['position'][idx], dtype=np.float32),
                de_mev=float(track['de'][idx]),
                dx_mm=float(track['dx'][idx]),
                theta_rad=float(track['theta'][idx]),
                phi_rad=float(track['phi'][idx]),
            ))

        track_manifest.append({
            'name': spec['name'],
            'momentum_mev': float(spec['momentum_mev']),
            'n_steps_total': int(n_steps),
            'n_steps_sampled': int(chosen.size),
        })

    return all_samples, track_manifest


def _match_noise_shape(noise: np.ndarray, target_shape: tuple[int, int]):
    out = np.asarray(noise, dtype=np.float64)
    tw, tt = target_shape
    nw, nt = out.shape
    if nw < tw:
        out = np.pad(out, ((0, tw - nw), (0, 0)))
    elif nw > tw:
        out = out[:tw, :]
    if nt < tt:
        out = np.pad(out, ((0, 0), (0, tt - nt)))
    elif nt > tt:
        out = out[:, :tt]
    return out


def _plot_global_fit(trans_fit, long_fit, out_pdf: Path,
                     comparison_trans_fit=None, comparison_long_fit=None,
                     gt_label: str = 'GT', sim_label: str = 'Noise-free simulation'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, fit, title in [
        (axes[0], trans_fit, 'Transverse fit'),
        (axes[1], long_fit, 'Longitudinal fit'),
    ]:
        x = fit['x_drift_time_us']
        y = fit['y_sigma2_cm2']
        y_fit = fit['y_fit_sigma2_cm2']
        order = np.argsort(x)
        ax.scatter(x, y, s=8, alpha=0.45, label=f'{gt_label} samples')
        ax.plot(x[order], y_fit[order], lw=2.0, color='tab:red', label=f'{gt_label} fit')
        if comparison_trans_fit is not None and comparison_long_fit is not None:
            cfit = comparison_trans_fit if title.startswith('Transverse') else comparison_long_fit
            cx = cfit['x_drift_time_us']
            cy = cfit['y_sigma2_cm2']
            cy_fit = cfit['y_fit_sigma2_cm2']
            corder = np.argsort(cx)
            ax.scatter(cx, cy, s=8, alpha=0.25, label=f'{sim_label} samples')
            ax.plot(cx[corder], cy_fit[corder], lw=1.6, color='tab:green', label=f'{sim_label} fit')
        ax.set_xlabel('drift time (us)')
        ax.set_ylabel('sigma^2 (cm^2)')
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc='best')

    if comparison_trans_fit is not None and comparison_long_fit is not None:
        fig.suptitle(
            'Classical diffusion fit: sigma^2(t) = intercept + slope * t\n'
            f"{gt_label}: D_T={trans_fit['D_cm2_per_us']:.4e}±{trans_fit['D_err_cm2_per_us']:.2e}, "
            f"D_L={long_fit['D_cm2_per_us']:.4e}±{long_fit['D_err_cm2_per_us']:.2e} cm^2/us\n"
            f"{sim_label}: D_T={comparison_trans_fit['D_cm2_per_us']:.4e}±{comparison_trans_fit['D_err_cm2_per_us']:.2e}, "
            f"D_L={comparison_long_fit['D_cm2_per_us']:.4e}±{comparison_long_fit['D_err_cm2_per_us']:.2e} cm^2/us"
        )
    else:
        fig.suptitle(
            'Classical diffusion fit: sigma^2(t) = intercept + slope * t\n'
            f"D_T={trans_fit['D_cm2_per_us']:.4e}±{trans_fit['D_err_cm2_per_us']:.2e} cm^2/us, "
            f"D_L={long_fit['D_cm2_per_us']:.4e}±{long_fit['D_err_cm2_per_us']:.2e} cm^2/us"
        )
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


def _plot_per_track(per_track_fits, out_pdf: Path,
                    per_track_fits_sim_noise_free=None):
    names = []
    dts = []
    dls = []
    dts_err = []
    dls_err = []
    dts_sim = []
    dls_sim = []
    dts_sim_err = []
    dls_sim_err = []
    for name, entry in sorted(per_track_fits.items()):
        tf = entry.get('transverse_fit')
        lf = entry.get('longitudinal_fit')
        if tf is None or lf is None:
            continue
        sim_entry = None
        if per_track_fits_sim_noise_free is not None:
            sim_entry = per_track_fits_sim_noise_free.get(name, {})
            stf = sim_entry.get('transverse_fit')
            slf = sim_entry.get('longitudinal_fit')
            if stf is None or slf is None:
                continue
        names.append(name)
        dts.append(tf['D_cm2_per_us'])
        dls.append(lf['D_cm2_per_us'])
        dts_err.append(tf['D_err_cm2_per_us'])
        dls_err.append(lf['D_err_cm2_per_us'])
        if sim_entry is not None:
            dts_sim.append(stf['D_cm2_per_us'])
            dls_sim.append(slf['D_cm2_per_us'])
            dts_sim_err.append(stf['D_err_cm2_per_us'])
            dls_sim_err.append(slf['D_err_cm2_per_us'])

    if not names:
        return

    x = np.arange(len(names), dtype=np.int32)
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(12, 0.6 * len(names)), 5), constrained_layout=True)
    ax.bar(x - width / 2, dts, width, yerr=dts_err, capsize=2, label='D_T')
    ax.bar(x + width / 2, dls, width, yerr=dls_err, capsize=2, label='D_L')
    if per_track_fits_sim_noise_free is not None and len(dts_sim) == len(names):
        ax.errorbar(x - width / 2, dts_sim, yerr=dts_sim_err, fmt='o', ms=3.5,
                    color='tab:blue', mfc='none', capsize=2, label='D_T sim (noise-free)')
        ax.errorbar(x + width / 2, dls_sim, yerr=dls_sim_err, fmt='o', ms=3.5,
                    color='tab:orange', mfc='none', capsize=2, label='D_L sim (noise-free)')
    ax.set_ylabel('D (cm^2/us)')
    if per_track_fits_sim_noise_free is not None:
        ax.set_title('Per-track diffusion estimates (noisy GT vs noise-free simulation)')
    else:
        ax.set_title('Per-track classical diffusion estimates')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=65, ha='right')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(loc='best')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f'JAX devices: {jax.devices()}')
    print(f'Results dir: {results_dir}')
    print(f'Plots dir  : {plots_dir}')

    if args.plots_only:
        summary_pkl = Path(args.summary_pkl) if args.summary_pkl else (results_dir / 'fit_summary.pkl')
        if not summary_pkl.exists():
            raise FileNotFoundError(
                f'--plots-only requested but summary pickle not found: {summary_pkl}')
        with open(summary_pkl, 'rb') as f:
            fit_summary = pickle.load(f)

        trans_fit = fit_summary.get('global_transverse_fit')
        long_fit = fit_summary.get('global_longitudinal_fit')
        per_track_fits = fit_summary.get('per_track_fits', {})
        sim_trans_fit = fit_summary.get('global_transverse_fit_sim_noise_free')
        sim_long_fit = fit_summary.get('global_longitudinal_fit_sim_noise_free')
        per_track_fits_sim = fit_summary.get('per_track_fits_sim_noise_free')
        if trans_fit is None or long_fit is None:
            raise ValueError(
                f'Summary pickle {summary_pkl} is missing global fit entries.')

        global_pdf = plots_dir / 'global_sigma2_vs_drift_fit.pdf'
        per_track_pdf = plots_dir / 'per_track_diffusion_estimates.pdf'
        _plot_global_fit(
            trans_fit,
            long_fit,
            global_pdf,
            comparison_trans_fit=sim_trans_fit,
            comparison_long_fit=sim_long_fit,
            gt_label='Noisy GT' if sim_trans_fit is not None else 'GT',
            sim_label='Noise-free simulation',
        )
        _plot_per_track(per_track_fits, per_track_pdf,
                        per_track_fits_sim_noise_free=per_track_fits_sim)
        print(f'Plots-only mode complete from: {summary_pkl}')
        print(f'Saved: {global_pdf}')
        print(f'Saved: {per_track_pdf}')
        return

    detector_config = generate_detector(CONFIG_PATH)
    sim = DetectorSimulator(
        detector_config,
        differentiable=True,
        n_segments=args.n_segments,
        include_noise=False,
        include_electronics=False,
        include_track_hits=False,
        include_digitize=False,
    )
    sim.warm_up()

    cfg = sim.config
    base_params = sim.default_sim_params
    velocity_cm_us = float(base_params.velocity_cm_us)
    time_step_us = float(cfg.time_step_us)
    noise_dict = None
    if args.noise_scale > 0.0:
        noise_dict = generate_noise(cfg, key=jax.random.PRNGKey(args.noise_seed))

    # Build canonical 15-track ensemble.
    track_specs = generate_random_boundary_tracks(
        _VOLUMES,
        n=args.n_boundary_tracks,
        seed=args.track_seed,
        include_diagonal_cross_muon=True,
    )
    print(f'Generated {len(track_specs)} track specs.')

    segment_samples, track_manifest = _collect_samples(track_specs, cfg, args)
    print(f'Total sampled deposits: {len(segment_samples)}')

    measurements = []
    n_planes = cfg.volumes[0].n_planes

    for i, sample in enumerate(segment_samples, start=1):
        deposits = _make_single_deposit(sample, cfg)
        vol_idx = _active_volume_idx(deposits)
        if vol_idx is None:
            continue
        plane_idx = _select_plane_index(cfg, args.plane, vol_idx)

        arrays = sim.forward(base_params, deposits)
        jax.block_until_ready(arrays)
        arr_sim = np.asarray(arrays[vol_idx * n_planes + plane_idx], dtype=np.float64)
        arr_gt = arr_sim
        if noise_dict is not None:
            noise_arr = _match_noise_shape(
                np.asarray(noise_dict[(vol_idx, plane_idx)], dtype=np.float64),
                arr_sim.shape,
            )
            arr_gt = arr_sim + args.noise_scale * noise_arr

        abs_arr_gt = np.abs(arr_gt)
        wire_prof_gt = np.sum(abs_arr_gt, axis=1)
        time_prof_gt = np.sum(abs_arr_gt, axis=0)
        _, var_wire_idx_gt = _weighted_mean_variance(wire_prof_gt)
        _, var_time_idx_gt = _weighted_mean_variance(time_prof_gt)

        abs_arr_sim = np.abs(arr_sim)
        wire_prof_sim = np.sum(abs_arr_sim, axis=1)
        time_prof_sim = np.sum(abs_arr_sim, axis=0)
        _, var_wire_idx_sim = _weighted_mean_variance(wire_prof_sim)
        _, var_time_idx_sim = _weighted_mean_variance(time_prof_sim)

        if (not np.isfinite(var_wire_idx_gt)
                or not np.isfinite(var_time_idx_gt)
                or not np.isfinite(var_wire_idx_sim)
                or not np.isfinite(var_time_idx_sim)):
            continue

        x_local_cm = float(np.asarray(deposits.volumes[vol_idx].positions_mm)[0, 0] / 10.0)
        plane_dist_cm = float(cfg.volumes[vol_idx].plane_distances_cm[plane_idx])
        drift_distance_cm = float(max(abs(x_local_cm - plane_dist_cm), 0.0))
        drift_time_us = float(drift_distance_cm / max(velocity_cm_us, 1e-12))

        wire_spacing_cm = float(cfg.volumes[vol_idx].wire_spacings_cm[plane_idx])
        sigma_wire_cm_gt = float(np.sqrt(var_wire_idx_gt) * wire_spacing_cm)
        sigma_time_us_gt = float(np.sqrt(var_time_idx_gt) * time_step_us)
        sigma_long_cm_gt = float(sigma_time_us_gt * velocity_cm_us)

        sigma_wire_cm_sim = float(np.sqrt(var_wire_idx_sim) * wire_spacing_cm)
        sigma_time_us_sim = float(np.sqrt(var_time_idx_sim) * time_step_us)
        sigma_long_cm_sim = float(sigma_time_us_sim * velocity_cm_us)

        measurements.append({
            'track_name': sample.track_name,
            'segment_idx': int(sample.segment_idx),
            'volume_idx': int(vol_idx),
            'plane_name': str(cfg.plane_names[vol_idx][plane_idx]),
            'drift_distance_cm': drift_distance_cm,
            'drift_time_us': drift_time_us,
            'sigma_wire_cm': sigma_wire_cm_gt,
            'sigma_long_cm': sigma_long_cm_gt,
            'sigma_time_us': sigma_time_us_gt,
            'wire_variance_idx2': float(var_wire_idx_gt),
            'time_variance_idx2': float(var_time_idx_gt),
            'sigma_wire_cm_gt': sigma_wire_cm_gt,
            'sigma_long_cm_gt': sigma_long_cm_gt,
            'sigma_time_us_gt': sigma_time_us_gt,
            'wire_variance_idx2_gt': float(var_wire_idx_gt),
            'time_variance_idx2_gt': float(var_time_idx_gt),
            'sigma_wire_cm_sim_noise_free': sigma_wire_cm_sim,
            'sigma_long_cm_sim_noise_free': sigma_long_cm_sim,
            'sigma_time_us_sim_noise_free': sigma_time_us_sim,
            'wire_variance_idx2_sim_noise_free': float(var_wire_idx_sim),
            'time_variance_idx2_sim_noise_free': float(var_time_idx_sim),
            'de_mev': float(sample.de_mev),
            'dx_mm': float(sample.dx_mm),
        })

        if i % 100 == 0:
            print(f'  processed {i}/{len(segment_samples)} sampled deposits')

    if len(measurements) < 10:
        raise RuntimeError(
            f'Only {len(measurements)} valid measurements were collected; '
            f'cannot perform robust fits.')

    drift_time = np.array([m['drift_time_us'] for m in measurements], dtype=np.float64)
    sigma_trans = np.array([m['sigma_wire_cm_gt'] for m in measurements], dtype=np.float64)
    sigma_long = np.array([m['sigma_long_cm_gt'] for m in measurements], dtype=np.float64)

    trans_fit = _fit_sigma2_vs_time(drift_time, sigma_trans, 'transverse')
    long_fit = _fit_sigma2_vs_time(drift_time, sigma_long, 'longitudinal')
    if trans_fit is None or long_fit is None:
        raise RuntimeError('Global fit failed (insufficient points after filtering).')

    # Per-track fits.
    per_track_fits = {}
    track_names = sorted(set(m['track_name'] for m in measurements))
    for tname in track_names:
        tm = [m for m in measurements if m['track_name'] == tname]
        tx = np.array([m['drift_time_us'] for m in tm], dtype=np.float64)
        ts = np.array([m['sigma_wire_cm_gt'] for m in tm], dtype=np.float64)
        ls = np.array([m['sigma_long_cm_gt'] for m in tm], dtype=np.float64)
        per_track_fits[tname] = {
            'n_points': int(len(tm)),
            'transverse_fit': _fit_sigma2_vs_time(tx, ts, 'transverse'),
            'longitudinal_fit': _fit_sigma2_vs_time(tx, ls, 'longitudinal'),
        }

    trans_fit_sim = None
    long_fit_sim = None
    per_track_fits_sim = None
    if args.noise_scale > 0.0:
        sigma_trans_sim = np.array(
            [m['sigma_wire_cm_sim_noise_free'] for m in measurements], dtype=np.float64)
        sigma_long_sim = np.array(
            [m['sigma_long_cm_sim_noise_free'] for m in measurements], dtype=np.float64)
        trans_fit_sim = _fit_sigma2_vs_time(drift_time, sigma_trans_sim, 'transverse')
        long_fit_sim = _fit_sigma2_vs_time(drift_time, sigma_long_sim, 'longitudinal')
        per_track_fits_sim = {}
        for tname in track_names:
            tm = [m for m in measurements if m['track_name'] == tname]
            tx = np.array([m['drift_time_us'] for m in tm], dtype=np.float64)
            ts = np.array([m['sigma_wire_cm_sim_noise_free'] for m in tm], dtype=np.float64)
            ls = np.array([m['sigma_long_cm_sim_noise_free'] for m in tm], dtype=np.float64)
            per_track_fits_sim[tname] = {
                'n_points': int(len(tm)),
                'transverse_fit': _fit_sigma2_vs_time(tx, ts, 'transverse'),
                'longitudinal_fit': _fit_sigma2_vs_time(tx, ls, 'longitudinal'),
            }

    fit_summary = {
        'config': {
            'track_seed': args.track_seed,
            'track_step_mm': args.track_step_mm,
            'n_boundary_tracks': args.n_boundary_tracks,
            'max_segments_per_track': args.max_segments_per_track,
            'sampling_mode': args.sampling_mode,
            'sampling_seed': args.sampling_seed,
            'plane': args.plane,
            'n_segments': args.n_segments,
            'noise_scale': args.noise_scale,
            'noise_seed': args.noise_seed,
            'time_step_us': time_step_us,
            'velocity_cm_us': velocity_cm_us,
            'results_dir': str(results_dir),
            'plots_dir': str(plots_dir),
        },
        'n_tracks': len(track_specs),
        'n_sampled_deposits': len(segment_samples),
        'n_valid_measurements': len(measurements),
        'global_transverse_fit': trans_fit,
        'global_longitudinal_fit': long_fit,
        'per_track_fits': per_track_fits,
        'global_transverse_fit_sim_noise_free': trans_fit_sim,
        'global_longitudinal_fit_sim_noise_free': long_fit_sim,
        'per_track_fits_sim_noise_free': per_track_fits_sim,
        'track_manifest': track_manifest,
    }

    # Save pkl outputs.
    measurements_pkl = results_dir / 'segment_measurements.pkl'
    summary_pkl = results_dir / 'fit_summary.pkl'
    manifest_pkl = results_dir / 'track_manifest.pkl'

    with open(measurements_pkl, 'wb') as f:
        pickle.dump({
            'measurements': measurements,
            'track_specs': track_specs,
        }, f)
    with open(summary_pkl, 'wb') as f:
        pickle.dump(fit_summary, f)
    with open(manifest_pkl, 'wb') as f:
        pickle.dump(track_manifest, f)

    # Save plots.
    global_pdf = plots_dir / 'global_sigma2_vs_drift_fit.pdf'
    per_track_pdf = plots_dir / 'per_track_diffusion_estimates.pdf'
    _plot_global_fit(
        trans_fit,
        long_fit,
        global_pdf,
        comparison_trans_fit=trans_fit_sim,
        comparison_long_fit=long_fit_sim,
        gt_label='Noisy GT' if args.noise_scale > 0.0 else 'GT',
        sim_label='Noise-free simulation',
    )
    _plot_per_track(per_track_fits, per_track_pdf,
                    per_track_fits_sim_noise_free=per_track_fits_sim)

    print('\n=== Classical diffusion fit summary ===')
    prefix = 'Noisy GT' if args.noise_scale > 0.0 else 'GT'
    print(f"{prefix} D_T = {trans_fit['D_cm2_per_us']:.6e} ± {trans_fit['D_err_cm2_per_us']:.2e} cm^2/us"
          f"  (R^2={trans_fit['r2']:.4f})")
    print(f"{prefix} D_L = {long_fit['D_cm2_per_us']:.6e} ± {long_fit['D_err_cm2_per_us']:.2e} cm^2/us"
          f"  (R^2={long_fit['r2']:.4f})")
    if trans_fit_sim is not None and long_fit_sim is not None:
        print(f"Noise-free simulation D_T = {trans_fit_sim['D_cm2_per_us']:.6e} ± {trans_fit_sim['D_err_cm2_per_us']:.2e} cm^2/us"
              f"  (R^2={trans_fit_sim['r2']:.4f})")
        print(f"Noise-free simulation D_L = {long_fit_sim['D_cm2_per_us']:.6e} ± {long_fit_sim['D_err_cm2_per_us']:.2e} cm^2/us"
              f"  (R^2={long_fit_sim['r2']:.4f})")
    print(f'Saved: {measurements_pkl}')
    print(f'Saved: {summary_pkl}')
    print(f'Saved: {manifest_pkl}')
    print(f'Saved: {global_pdf}')
    print(f'Saved: {per_track_pdf}')


if __name__ == '__main__':
    main()
