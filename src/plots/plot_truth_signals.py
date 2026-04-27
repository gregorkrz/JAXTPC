#!/usr/bin/env python
"""
Generate Truth_signals.pdf for one or more tracks.

Each PDF has two pages:
  page 1 — GT wire-plane signals (no noise)
  page 2 — GT signals with realistic detector noise (noise-scale 1.0)

Each page shows one subplot per wire plane (U / V / Y), with wire index
on the Y axis and time step on the X axis.

Usage
-----
    python src/plots/plot_truth_signals.py --tracks diagonal_100MeV:1,1,1:100
    python src/plots/plot_truth_signals.py --tracks diagonal_100MeV:1,1,1:100+x100:1,0,0:100+z100:0,0,1:100
    python src/plots/plot_truth_signals.py --tracks diagonal --output-dir plots/truth_signals
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

import argparse
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import build_deposit_data
from tools.noise import generate_noise
from tools.particle_generator import generate_muon_track

_PLOTS_DIR = os.environ.get('PLOTS_DIR', 'plots')

CONFIG_PATH        = 'config/cubic_wireplane_config.yaml'
N_SEGMENTS         = 50_000
MAX_ACTIVE_BUCKETS = 1000
GT_LIFETIME_US     = 10_000.0
GT_VELOCITY_CM_US  = 0.160

TRACK_PRESETS = {
    'diagonal':     ((1.0,  1.0,  1.0),  1000.0),
    'diagonal_100MeV': ((1.0, 1.0, 1.0), 100.0),
    'X':            ((1.0,  0.0,  0.0),  1000.0),
    'Y':            ((0.0,  1.0,  0.0),  1000.0),
    'Z':            ((0.0,  0.0,  1.0),  1000.0),
    'x100':         ((1.0,  0.0,  0.0),   100.0),
    'y100':         ((0.0,  1.0,  0.0),   100.0),
    'z100':         ((0.0,  0.0,  1.0),   100.0),
    'U':            ((0.0,  0.866, 0.5), 1000.0),
    'V':            ((0.0, -0.866, 0.5), 1000.0),
    'track2':       ((0.5,  1.05, 0.2),   200.0),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tracks', default='diagonal',
                   help='"+"-separated track presets or name:dx,dy,dz:mom_mev specs. '
                        f'Presets: {", ".join(TRACK_PRESETS)}')
    p.add_argument('--noise-scale', type=float, default=1.0,
                   help='Noise amplitude for page 2 (default: 1.0)')
    p.add_argument('--noise-seed', type=int, default=0,
                   help='RNG seed for noise draw (default: 0)')
    p.add_argument('--volume', type=int, default=0,
                   help='Volume index to plot (default: 0)')
    p.add_argument('--output-dir', default=os.path.join(_PLOTS_DIR, 'truth_signals'),
                   help='Output directory (default: plots/truth_signals)')
    return p.parse_args()


def parse_tracks(tracks_str):
    """Parse '+'-separated preset names or name:dx,dy,dz:mom_mev specs."""
    specs = []
    for item in tracks_str.split('+'):
        item = item.strip()
        if ':' in item:
            parts = item.split(':')
            name = parts[0]
            direction = tuple(float(x) for x in parts[1].split(','))
            momentum_mev = float(parts[2])
        elif item in TRACK_PRESETS:
            direction, momentum_mev = TRACK_PRESETS[item]
            name = item
        else:
            raise ValueError(f'Unknown track {item!r}. Known presets: {list(TRACK_PRESETS)}')
        specs.append({'name': name, 'direction': direction, 'momentum_mev': momentum_mev})
    return specs


def build_simulator():
    detector_config = generate_detector(CONFIG_PATH)
    return DetectorSimulator(
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


def get_gt_arrays(simulator, track_spec):
    direction    = track_spec['direction']
    momentum_mev = track_spec['momentum_mev']
    track = generate_muon_track(
        start_position_mm=(0.0, 0.0, 0.0),
        direction=direction,
        kinetic_energy_mev=momentum_mev,
        step_size_mm=0.1,
        track_id=1,
    )
    deposits = build_deposit_data(
        track['position'], track['de'], track['dx'], simulator.config,
        theta=track['theta'], phi=track['phi'],
        track_ids=track['track_id'],
    )
    gt_params = simulator.default_sim_params._replace(
        lifetime_us    = jnp.array(GT_LIFETIME_US),
        velocity_cm_us = jnp.array(GT_VELOCITY_CM_US),
    )
    arrays = simulator.forward(gt_params, deposits)
    jax.block_until_ready(arrays)
    return arrays


def apply_noise(gt_arrays, simulator, noise_scale, noise_seed):
    cfg        = simulator.config
    noise_dict = generate_noise(cfg, key=jax.random.PRNGKey(noise_seed))
    n_readouts = cfg.volumes[0].n_planes if simulator._readout_type == 'wire' else 1
    noisy = []
    for v in range(cfg.n_volumes):
        for p in range(n_readouts):
            gt    = gt_arrays[v * n_readouts + p]
            noise = noise_dict[(v, p)] * noise_scale
            if noise.shape[0] < gt.shape[0]:
                noise = jnp.pad(noise, ((0, gt.shape[0] - noise.shape[0]), (0, 0)))
            noisy.append(gt + noise)
    return tuple(noisy)


def make_signals_page(arrays, plane_labels, title, time_step_us):
    n_planes = len(arrays)
    fig, axes = plt.subplots(1, n_planes, figsize=(6 * n_planes, 5))
    if n_planes == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=10)

    for ax, arr, label in zip(axes, arrays, plane_labels):
        data = np.asarray(arr)
        # symmetric colour scale around 0
        vmax = np.nanpercentile(np.abs(data), 99)
        vmax = max(vmax, 1e-9)
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

        n_wires, n_time = data.shape
        t_max_us = n_time * time_step_us
        im = ax.imshow(
            data,
            aspect='auto',
            origin='lower',
            norm=norm,
            cmap='RdBu_r',
            extent=[0, t_max_us, 0, n_wires],
        )
        fig.colorbar(im, ax=ax, label='signal (e⁻)')
        ax.set_xlabel('time (μs)', fontsize=9)
        ax.set_ylabel('wire index', fontsize=9)
        ax.set_title(f'plane {label}', fontsize=9)

    fig.tight_layout()
    return fig


def main():
    args     = parse_args()
    track_specs = parse_tracks(args.tracks)

    print(f'JAX devices : {jax.devices()}')
    print(f'Building simulator...')
    simulator = build_simulator()

    cfg          = simulator.config
    n_readouts   = cfg.volumes[0].n_planes if simulator._readout_type == 'wire' else 1
    plane_labels = list(cfg.plane_names[0])  # e.g. ['U', 'V', 'Y']
    time_step_us = float(cfg.time_step_us)

    print('Warming up JIT...')
    t0 = time.time()
    simulator.warm_up()
    print(f'Done ({time.time() - t0:.1f} s)')

    for ts in track_specs:
        name = ts['name']
        print(f'\nTrack: {name}  dir={ts["direction"]}  T={ts["momentum_mev"]} MeV')

        gt_arrays = get_gt_arrays(simulator, ts)
        noisy_arrays = apply_noise(gt_arrays, simulator, args.noise_scale, args.noise_seed)

        # one plane tuple per readout plane
        vol = args.volume
        gt_planes    = [gt_arrays[vol * n_readouts + p]    for p in range(n_readouts)]
        noisy_planes = [noisy_arrays[vol * n_readouts + p] for p in range(n_readouts)]

        folder = os.path.join(args.output_dir, name)
        os.makedirs(folder, exist_ok=True)
        out_path = os.path.join(folder, 'Truth_signals.pdf')

        with PdfPages(out_path) as pdf:
            fig = make_signals_page(
                gt_planes, plane_labels,
                title=f'Truth signals  —  {name}  dir={ts["direction"]}  T={ts["momentum_mev"]} MeV  vol={args.volume}  (no noise)',
                time_step_us=time_step_us,
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            fig = make_signals_page(
                noisy_planes, plane_labels,
                title=f'Truth signals  —  {name}  dir={ts["direction"]}  T={ts["momentum_mev"]} MeV  vol={args.volume}  (noise scale={args.noise_scale})',
                time_step_us=time_step_us,
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        print(f'  Saved: {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
