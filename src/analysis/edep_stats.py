#!/usr/bin/env python
"""
Plot dE/dx distributions and 3-D energy-deposit visualisations for muon tracks.

Usage
-----
    python edep_stats.py
    python edep_stats.py --output-dir results/my_vis
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from tools.particle_generator import generate_muon_track

#DETECTOR_BOUNDS_MM = ((-300, 300), (-300, 300), (-300, 300))

TRACKS = [
    dict(name='diagonal',       direction=(1.0, 1.0, 1.0), momentum_mev=1000.0),
    dict(name='track2',         direction=(0.5, 1.05, 0.2), momentum_mev=1000.0),
    dict(name='Z',              direction=(0.0, 0.0, 1.0), momentum_mev=1000.0),
    dict(name='track2_200MeV',  direction=(0.5, 1.05, 0.2), momentum_mev=200.0),
    dict(name='track2_100MeV',  direction=(0.5, 1.05, 0.2), momentum_mev=100.0),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', default='results/muon_track_visualization',
                   help='Output directory (default: results/muon_track_visualization)')
    return p.parse_args()


def plot_edep_3d(t, track, output_path):
    pos = np.array(track['position'])   # (N, 3) mm
    de  = np.array(track['de'])         # (N,)   MeV

    de_min, de_max = de.min(), de.max()
    drange = de_max - de_min if de_max > de_min else 1.0
    sizes = 3.0 + (de - de_min) / drange * 15.0

    fig = go.Figure(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='markers',
        marker=dict(
            size=sizes,
            color=de,
            colorscale='Viridis',
            colorbar=dict(title='dE (MeV)'),
            opacity=0.8,
        ),
        text=[f'dE={v:.4f} MeV' for v in de],
        hoverinfo='text+x+y+z',
    ))
    fig.update_layout(
        title=f"Energy deposits — {t['name']}  dir={t['direction']}  T={t['momentum_mev']:.0f} MeV",
        scene=dict(
            xaxis_title='x (mm)',
            yaxis_title='y (mm)',
            zaxis_title='z (mm)',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(output_path)
    print(f'  Saved 3-D plot: {output_path}')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 10))

    for t in TRACKS:
        print(f"Generating {t['name']}  dir={t['direction']}  T={t['momentum_mev']} MeV ...")
        track = generate_muon_track(
            start_position_mm=(0.0, 0.0, 0.0),
            direction=t['direction'],
            kinetic_energy_mev=t['momentum_mev'],
            step_size_mm=0.1,
            track_id=1,
            #detector_bounds_mm=DETECTOR_BOUNDS_MM,
        )
        de = np.array(track['de'])   # MeV
        dx = np.array(track['dx'])   # mm
        dedx = de / (dx / 10.0)      # MeV/cm

        label = f"{t['name']}  ({t['direction']}, {t['momentum_mev']:.0f} MeV)  n={len(dedx):,}"
        is_100mev = t['momentum_mev'] == 100.0

        if not is_100mev:
            ax_top.hist(dedx, bins=80, histtype='step', linewidth=1.5, label=label, density=True)
        ax_bot.hist(dedx, bins=80, histtype='step', linewidth=1.5, label=label, density=True)

        print(f"  steps={len(dedx):,}  dE/dx: mean={dedx.mean():.3f}  median={np.median(dedx):.3f}"
              f"  min={dedx.min():.3f}  max={dedx.max():.3f}  MeV/cm")

        html_path = os.path.join(args.output_dir, f"edep_3d_{t['name']}.html")
        plot_edep_3d(t, track, html_path)

    ax_top.set_xlabel('dE/dx  (MeV/cm)', fontsize=11)
    ax_top.set_ylabel('density', fontsize=11)
    ax_top.set_title('dE/dx distributions — without 100 MeV track  (linear)', fontsize=11)
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)

    ax_bot.set_xlabel('dE/dx  (MeV/cm)', fontsize=11)
    ax_bot.set_ylabel('density', fontsize=11)
    ax_bot.set_title('dE/dx distributions — all tracks incl. 100 MeV  (log scale)', fontsize=11)
    ax_bot.set_yscale('log')
    ax_bot.legend(fontsize=8)
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout(pad=2.0)

    stats_path = os.path.join(args.output_dir, 'stats.pdf')
    fig.savefig(stats_path, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {stats_path}')


if __name__ == '__main__':
    main()
