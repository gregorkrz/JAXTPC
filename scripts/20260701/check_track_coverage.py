#!/usr/bin/env python
"""
Visualize the xyz spatial coverage of the near-cathode track ensemble used for
E-field calibration (tools.random_boundary_tracks.generate_random_nice_tracks).

Generates the same track specs run_optimization.py builds from
``--N-random-tracks``/``--tracks-random-seed`` (y/z-face entries with
|x| < 1000 mm, polar angle from x-axis in [30°, 150°], T ~ U[100, 1000] MeV),
propagates each one through liquid argon with the real PDG dE/dx table
(tools.particle_generator), and plots the resulting paths: a 3D view plus
xy/xz/yz projections, colored by kinetic energy, over the East/West TPC
volume outlines.

Usage
-----
  .venv/bin/python scripts/20260701/check_track_coverage.py
  .venv/bin/python scripts/20260701/check_track_coverage.py --n-random-tracks 12 --seed 7
  .venv/bin/python scripts/20260701/check_track_coverage.py --output plots/20260701/coverage.png --show
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from tools.particle_generator import generate_muon_track
from tools.random_boundary_tracks import generate_random_nice_tracks

# East x∈[-216,0] cm, West x∈[0,216] cm, y,z∈[-216,216] cm — matches
# config/cubic_wireplane_config.yaml (mirrors the _Vol stub in
# src/analysis/launch_2d_landscape_pairs.py).
class _Vol:
    def __init__(self, ranges_cm):
        self.ranges_cm = ranges_cm


_VOLUMES = [
    _Vol([[-216.0, 0.0], [-216.0, 216.0], [-216.0, 216.0]]),  # East
    _Vol([[0.0, 216.0], [-216.0, 216.0], [-216.0, 216.0]]),   # West
]
_DETECTOR_BOUNDS_MM = ((-2160.0, 2160.0), (-2160.0, 2160.0), (-2160.0, 2160.0))


def build_track_paths(specs, step_size_mm):
    """Propagate each spec through LAr; returns (paths, de_values).

    paths: list of (N,3) position arrays (mm); de_values: list of (N,)
    per-step energy deposits (MeV), aligned with paths.
    """
    paths, de_values = [], []
    for spec in specs:
        track = generate_muon_track(
            start_position_mm=spec['start_position_mm'],
            direction=spec['direction'],
            kinetic_energy_mev=spec['momentum_mev'],
            step_size_mm=step_size_mm,
            detector_bounds_mm=_DETECTOR_BOUNDS_MM,
        )
        paths.append(np.asarray(track['position'], dtype=np.float64))
        de_values.append(np.asarray(track['de'], dtype=np.float64))
    return paths, de_values


def _box_edges_mm(vol):
    (x0, x1), (y0, y1), (z0, z1) = (tuple(r) for r in vol.ranges_cm)
    x0, x1, y0, y1, z0, z1 = (v * 10.0 for v in (x0, x1, y0, y1, z0, z1))
    corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                edges.append((corners[i], corners[j]))
    return edges, (x0, x1, y0, y1, z0, z1)


def _draw_volumes_3d(ax):
    for vol in _VOLUMES:
        edges, _ = _box_edges_mm(vol)
        ax.add_collection3d(Line3DCollection(edges, colors='0.6', linewidths=0.6, linestyles='--'))


def _draw_volumes_2d(ax, i, j):
    for vol in _VOLUMES:
        _, (x0, x1, y0, y1, z0, z1) = _box_edges_mm(vol)
        lo = (x0, y0, z0)
        hi = (x1, y1, z1)
        rect_x = [lo[i], hi[i], hi[i], lo[i], lo[i]]
        rect_y = [lo[j], lo[j], hi[j], hi[j], lo[j]]
        ax.plot(rect_x, rect_y, color='0.6', lw=0.6, ls='--')


def plot_coverage(specs, paths, output_path, show):
    energies = np.array([s['momentum_mev'] for s in specs])
    cmap = plt.get_cmap('viridis')
    norm = matplotlib.colors.Normalize(vmin=energies.min(), vmax=energies.max())

    fig = plt.figure(figsize=(14, 10))

    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    for spec, pos in zip(specs, paths):
        if len(pos) == 0:
            continue
        ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=cmap(norm(spec['momentum_mev'])), lw=1.2)
    _draw_volumes_3d(ax3d)
    ax3d.set_xlabel('x [mm]')
    ax3d.set_ylabel('y [mm]')
    ax3d.set_zlabel('z [mm]')
    ax3d.set_title('3D coverage')

    projections = [(0, 1, 'x', 'y'), (0, 2, 'x', 'z'), (1, 2, 'y', 'z')]
    for k, (i, j, li, lj) in enumerate(projections):
        ax = fig.add_subplot(2, 2, k + 2)
        for spec, pos in zip(specs, paths):
            if len(pos) == 0:
                continue
            ax.plot(pos[:, i], pos[:, j], color=cmap(norm(spec['momentum_mev'])), lw=1.0)
        _draw_volumes_2d(ax, i, j)
        ax.set_xlabel(f'{li} [mm]')
        ax.set_ylabel(f'{lj} [mm]')
        ax.set_aspect('equal')
        ax.set_title(f'{li}-{lj} projection')

    fig.suptitle(f'Track ensemble spatial coverage (n={len(specs)})')
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cax, label='Kinetic energy [MeV]')

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f'Saved {output_path}')
    if show or not output_path:
        plt.show()


def plot_edep_histograms(paths, de_values, output_path, show):
    """Energy-weighted histograms of edep positions along x, y, z (mm)."""
    xyz = np.concatenate(paths, axis=0)
    de = np.concatenate(de_values, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    bounds = _DETECTOR_BOUNDS_MM
    for axis, (ax, label) in enumerate(zip(axes, 'xyz')):
        ax.hist(xyz[:, axis], bins=100, range=bounds[axis], weights=de,
                color='tab:blue', edgecolor='none')
        ax.set_xlabel(f'{label} [mm]')
        ax.set_ylabel('energy deposited [MeV]')
        ax.set_title(f'Edep distribution vs {label}')

    fig.suptitle(f'Energy-deposit spatial distribution (n_tracks={len(paths)})')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f'Saved {output_path}')
    if show or not output_path:
        plt.show()


def print_summary(specs, paths):
    print(f'{"name":<28}{"E [MeV]":>9}{"n_steps":>9}{"length [mm]":>13}  start_mm')
    all_pos = []
    for spec, pos in zip(specs, paths):
        length = float(np.linalg.norm(pos[-1] - pos[0])) if len(pos) > 1 else 0.0
        print(f'{spec["name"]:<28}{spec["momentum_mev"]:>9.0f}{len(pos):>9d}{length:>13.1f}  '
              f'{tuple(round(v, 1) for v in spec["start_position_mm"])}')
        if len(pos):
            all_pos.append(pos)
    if all_pos:
        stacked = np.concatenate(all_pos, axis=0)
        for axis, label in enumerate('xyz'):
            lo, hi = stacked[:, axis].min(), stacked[:, axis].max()
            print(f'  {label} coverage: [{lo:.0f}, {hi:.0f}] mm')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-random-tracks', type=int, default=12,
                         help='Number of random near-cathode tracks (generate_random_nice_tracks).')
    parser.add_argument('--seed', type=int, default=7, help='RNG seed (tracks-random-seed).')
    parser.add_argument('--step-size-mm', type=float, default=5.0,
                         help='Propagation step size in mm (coarser = faster, default 5mm).')
    parser.add_argument('--output', default=os.path.join(
        os.environ.get('PLOTS_DIR', 'plots'), '20260701_track_coverage', 'coverage.png'),
        help='Path to save the figure (PNG). Pass "" to skip saving.')
    parser.add_argument('--show', action='store_true', help='Also open an interactive window.')
    args = parser.parse_args()

    if not args.output and not args.show:
        args.show = True
    if not args.show:
        matplotlib.use('Agg')

    specs = generate_random_nice_tracks(_VOLUMES, n=args.n_random_tracks, seed=args.seed)
    print(f'Generated {len(specs)} near-cathode tracks, seed={args.seed}')

    paths, de_values = build_track_paths(specs, step_size_mm=args.step_size_mm)
    print_summary(specs, paths)
    plot_coverage(specs, paths, args.output or None, args.show)

    edep_output = None
    if args.output:
        edep_output = os.path.join(os.path.dirname(args.output), 'edep_histograms.png')
    plot_edep_histograms(paths, de_values, edep_output, args.show)


if __name__ == '__main__':
    main()
