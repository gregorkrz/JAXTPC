#!/usr/bin/env python
"""
Estimate the noise floor of the loss function.

The noise floor is the loss between two independent noise realisations of the
same clean ground-truth signal.  It sets the irreducible lower bound that the
optimiser can never cross, even at perfect parameters.

Two quantities are reported for each track and in aggregate:

  floor_clean   loss(GT_clean, GT + noise)
                — how far the TRUE parameters are from the noisy target.
                  This is what the optimiser converges toward.

  floor_pair    loss(GT + noise_A, GT + noise_B)
                — loss between two independent noisy draws.
                  Approximately 2 × floor_clean for Gaussian noise.

Both are averaged over --n-seeds independent noise draws.

Usage
-----
    # Quick estimate (5 seeds, 15-track ensemble, sobolev loss)
    python src/analysis/noise_floor.py

    # More seeds, specific tracks, different loss
    python src/analysis/noise_floor.py --n-seeds 20 --loss mse_loss

    # Save results to CSV
    python src/analysis/noise_floor.py --output results/noise_floor.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import build_deposit_data
from tools.losses import (
    make_sobolev_weight,
    sobolev_loss_geomean_log1p,
    sobolev_loss,
    mse_loss,
    l1_loss,
)
from tools.noise import generate_noise
from tools.particle_generator import generate_muon_track
from tools.random_boundary_tracks import (
    generate_random_boundary_tracks,
    filter_track_inside_volumes,
    N_DEFAULT_BOUNDARY_MUONS,
)

CONFIG_PATH        = 'config/cubic_wireplane_config.yaml'
N_SEGMENTS         = 50_000
MAX_ACTIVE_BUCKETS = 1_000
GT_LIFETIME_US     = 10_000.0
GT_VELOCITY_CM_US  = 0.160
SOBOLEV_MAX_PAD    = 128

VALID_LOSSES = ('sobolev_loss_geomean_log1p', 'sobolev_loss', 'mse_loss', 'l1_loss')


class _Vol:
    def __init__(self, ranges_cm):
        self.ranges_cm = ranges_cm


_VOLUMES = [
    _Vol([[-216.0, 0.0],  [-216.0, 216.0], [-216.0, 216.0]]),
    _Vol([[0.0,  216.0],  [-216.0, 216.0], [-216.0, 216.0]]),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n-seeds', type=int, default=10,
                   help='Number of independent noise draws to average over (default: 10)')
    p.add_argument('--noise-scale', type=float, default=1.0,
                   help='Noise amplitude as multiple of calibrated detector noise '
                        '(default: 1.0 = realistic)')
    p.add_argument('--loss', default='sobolev_loss_geomean_log1p', choices=VALID_LOSSES,
                   help='Loss function (default: sobolev_loss_geomean_log1p)')
    p.add_argument('--step-size', type=float, default=1.0,
                   help='Muon track step size in mm (default: 1.0)')
    p.add_argument('--max-deposits', type=int, default=5_000,
                   help='Max deposits per track (default: 5000)')
    p.add_argument('--n-boundary-tracks', type=int, default=N_DEFAULT_BOUNDARY_MUONS,
                   help=f'Random boundary muons (default: {N_DEFAULT_BOUNDARY_MUONS})')
    p.add_argument('--track-seed', type=int, default=42,
                   help='RNG seed for boundary track generation (default: 42)')
    p.add_argument('--output', default=None, metavar='PATH',
                   help='Save per-track results to CSV (default: print only)')
    return p.parse_args()


def compute_loss(signals_a, signals_b, weights, loss_name):
    planes = tuple(range(len(signals_a)))
    if loss_name == 'sobolev_loss_geomean_log1p':
        return float(sobolev_loss_geomean_log1p(signals_a, signals_b, weights, planes))
    if loss_name == 'sobolev_loss':
        return float(sobolev_loss(signals_a, signals_b, weights, planes))
    if loss_name == 'mse_loss':
        return float(mse_loss(signals_a, signals_b))
    if loss_name == 'l1_loss':
        return float(l1_loss(signals_a, signals_b))
    raise ValueError(loss_name)


def add_noise(clean_arrays, noise_dict, noise_scale, n_vols, n_planes):
    noisy = []
    for v in range(n_vols):
        for p in range(n_planes):
            arr   = clean_arrays[v * n_planes + p]
            noise = np.asarray(noise_dict[(v, p)]) * noise_scale
            if noise.shape[0] < arr.shape[0]:
                noise = np.pad(noise, ((0, arr.shape[0] - noise.shape[0]), (0, 0)))
            noisy.append(arr + noise)
    return tuple(noisy)


def main():
    args = parse_args()

    print(f'JAX devices  : {jax.devices()}')
    print(f'Loss         : {args.loss}')
    print(f'Noise scale  : {args.noise_scale}')
    print(f'N seeds      : {args.n_seeds}')
    print(f'Step size    : {args.step_size} mm  max_deposits={args.max_deposits:,}')
    print()

    # ── Simulator ──────────────────────────────────────────────────────────────
    detector_config = generate_detector(CONFIG_PATH)
    sim = DetectorSimulator(
        detector_config,
        differentiable=True,
        n_segments=args.max_deposits,
        use_bucketed=True,
        max_active_buckets=MAX_ACTIVE_BUCKETS,
        include_noise=False,
        include_electronics=False,
        include_track_hits=False,
        include_digitize=False,
    )

    print('Warming up JIT...')
    t0 = time.time()
    sim.warm_up()
    print(f'Done ({time.time() - t0:.1f} s)\n')

    cfg      = sim.config
    n_vols   = cfg.n_volumes
    n_planes = cfg.volumes[0].n_planes

    gt_params = sim.default_sim_params._replace(
        lifetime_us    = jnp.array(GT_LIFETIME_US),
        velocity_cm_us = jnp.array(GT_VELOCITY_CM_US),
    )

    # ── Tracks ─────────────────────────────────────────────────────────────────
    tracks = generate_random_boundary_tracks(_VOLUMES, n=args.n_boundary_tracks,
                                             seed=args.track_seed)
    print(f'Tracks: {len(tracks)}')

    # ── Per-track GT signals + weights ─────────────────────────────────────────
    print('Computing GT signals...')
    track_signals = []   # list of tuples of np arrays (one per plane)
    track_weights = []
    track_names   = []

    for ts in tracks:
        d = ts['direction']
        direction = tuple(float(x) for x in d.split(',')) if isinstance(d, str) else tuple(d)
        smm = ts.get('start_position_mm', (0.0, 0.0, 0.0))
        track = generate_muon_track(
            start_position_mm=smm,
            direction=direction,
            kinetic_energy_mev=float(ts['momentum_mev']),
            step_size_mm=args.step_size,
            track_id=1,
        )
        track = filter_track_inside_volumes(track, cfg.volumes)
        deposits = build_deposit_data(
            track['position'], track['de'], track['dx'], cfg,
            theta=track['theta'], phi=track['phi'],
            track_ids=track['track_id'],
        )
        arrays = sim.forward(gt_params, deposits)
        jax.block_until_ready(arrays)
        clean = tuple(np.asarray(a) for a in arrays)

        weights = tuple(
            np.asarray(make_sobolev_weight(a.shape[0], a.shape[1], max_pad=SOBOLEV_MAX_PAD))
            for a in clean
        )

        signal_rms = float(np.mean([np.std(a) for a in clean]))

        track_signals.append(clean)
        track_weights.append(weights)
        track_names.append(ts['name'])
        print(f'  {ts["name"]:<45s}  deposits={sum(v.n_actual for v in deposits.volumes):,}'
              f'  signal_rms={signal_rms:.3g}')

    print()

    # ── Noise floor estimation ──────────────────────────────────────────────────
    # For each seed pair (A, B), compute:
    #   floor_clean[track] = loss(clean, clean + noise_A)
    #   floor_pair[track]  = loss(clean + noise_A, clean + noise_B)

    n_tracks = len(tracks)
    floors_clean = np.zeros((args.n_seeds, n_tracks))
    floors_pair  = np.zeros((args.n_seeds, n_tracks))

    print(f'Computing noise floors over {args.n_seeds} seed pairs...')
    for seed_idx in range(args.n_seeds):
        seed_A = seed_idx * 2
        seed_B = seed_idx * 2 + 1

        noise_A = generate_noise(cfg, key=jax.random.PRNGKey(seed_A))
        noise_B = generate_noise(cfg, key=jax.random.PRNGKey(seed_B))

        for ti, (clean, weights) in enumerate(zip(track_signals, track_weights)):
            noisy_A = add_noise(clean, noise_A, args.noise_scale, n_vols, n_planes)
            noisy_B = add_noise(clean, noise_B, args.noise_scale, n_vols, n_planes)

            floors_clean[seed_idx, ti] = compute_loss(clean, noisy_A, weights, args.loss)
            floors_pair[seed_idx, ti]  = compute_loss(noisy_A, noisy_B, weights, args.loss)

        print(f'  seed pair {seed_A}/{seed_B}: '
              f'floor_clean={floors_clean[seed_idx].mean():.4g}  '
              f'floor_pair={floors_pair[seed_idx].mean():.4g}')

    # ── Results ────────────────────────────────────────────────────────────────
    print()
    print('=' * 72)
    print(f'  Noise floor summary  (loss={args.loss}, noise_scale={args.noise_scale})')
    print('=' * 72)
    print(f'{"Track":<45s}  {"floor_clean":>12s}  {"floor_pair":>12s}')
    print('-' * 72)

    rows = []
    for ti, name in enumerate(track_names):
        fc_mean = floors_clean[:, ti].mean()
        fc_std  = floors_clean[:, ti].std()
        fp_mean = floors_pair[:, ti].mean()
        fp_std  = floors_pair[:, ti].std()
        print(f'  {name:<43s}  {fc_mean:8.4g} ±{fc_std:.2g}  {fp_mean:8.4g} ±{fp_std:.2g}')
        rows.append(dict(track=name,
                         floor_clean_mean=fc_mean, floor_clean_std=fc_std,
                         floor_pair_mean=fp_mean,  floor_pair_std=fp_std))

    print('-' * 72)
    fc_all = floors_clean.mean(axis=1)   # mean over tracks, per seed
    fp_all = floors_pair.mean(axis=1)
    print(f'  {"MEAN OVER ALL TRACKS":<43s}  '
          f'{fc_all.mean():8.4g} ±{fc_all.std():.2g}  '
          f'{fp_all.mean():8.4g} ±{fp_all.std():.2g}')
    print('=' * 72)
    print()
    print('floor_clean = loss(GT_clean, GT + noise)  '
          '— irreducible loss at true parameters')
    print('floor_pair  = loss(GT + noise_A, GT + noise_B)  '
          '— loss between two independent noisy draws')

    if args.output:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
