#!/usr/bin/env python
"""
Generate muon track events and save them to an HDF5 file consumable by
``run_optimization.py --events-file``.

By default each output event holds a single track (one straight-line muon,
stepped via ``generate_muon_track``), matching how ``--tracks`` and
``--N-random-tracks`` work in run_optimization.py today. The file format
itself (``tools/event_io.py``) places no such restriction — it stores
per-step deposits with a ``track_id`` per step, so events with multiple
tracks (e.g. real GEANT4 output, or hand-built multi-track events) load
through the same path.

Usage
-----
  python generate_muon_tracks.py --tracks diagonal+Z --output events.h5
  python generate_muon_tracks.py --N-random-tracks 50 --tracks-random-seed 7 \\
      --output random_events.h5

Named track presets
-------------------
  diagonal  (1,1,1)        1000 MeV
  X         (1,0,0)        1000 MeV
  Y         (0,1,0)        1000 MeV
  Z         (0,0,1)        1000 MeV
  U         (0,0.866,0.5)  1000 MeV
  V         (0,-0.866,0.5) 1000 MeV
  track2    (0.5,1.05,0.2)  200 MeV

  Custom tracks: name:dx,dy,dz:momentum_mev[:x,y,z]  (mixed with presets is fine)
  e.g. --tracks diagonal+mytrack:0.1,0.2,0.9:500
"""

import argparse
import sys

from optlib.constants import CONFIG_PATH, TRACK_PRESETS
from optlib.parsing import parse_tracks
from tools.event_io import save_events_h5
from tools.geometry import generate_detector
from tools.particle_generator import generate_muon_track
from tools.random_boundary_tracks import generate_random_nice_tracks


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tracks', default='diagonal',
                   help='"+"-separated track presets or name:dx,dy,dz:mom_mev specs. '
                        f'Default: diagonal.  Presets: {", ".join(TRACK_PRESETS)}')
    p.add_argument('--N-random-tracks', type=int, default=0, metavar='N',
                   help='If > 0, ignore --tracks and generate N random near-cathode tracks '
                        'via generate_random_nice_tracks (same generator used by '
                        'run_optimization.py --N-random-tracks).')
    p.add_argument('--tracks-random-seed', type=int, default=7, metavar='SEED',
                   help='RNG seed for --N-random-tracks (default: 7).')
    p.add_argument('--start-position-mm', default='0,0,0',
                   help='Default start position "x,y,z" in mm for tracks that don\'t '
                        'specify their own (default: 0,0,0). Ignored for --N-random-tracks, '
                        'which always samples its own start positions.')
    p.add_argument('--step-size-mm', type=float, default=0.1,
                   help='Step size in mm for the generated tracks (default: 0.1).')
    p.add_argument('--min-energy-mev', type=float, default=10.0,
                   help='Stop stepping a track below this kinetic energy (default: 10.0).')
    p.add_argument('--output', required=True, metavar='PATH',
                   help='Output HDF5 path, e.g. events.h5.')
    return p.parse_args()


def main():
    args = parse_args()

    if args.N_random_tracks > 0:
        volumes = generate_detector(CONFIG_PATH).volumes
        track_specs = generate_random_nice_tracks(
            volumes, n=args.N_random_tracks, seed=args.tracks_random_seed)
    else:
        track_specs = parse_tracks(args.tracks)

    default_start_mm = tuple(float(v) for v in args.start_position_mm.split(','))

    events = []
    for ts in track_specs:
        start_mm = ts['start_position_mm'] if ts['start_position_mm'] is not None else default_start_mm
        track = generate_muon_track(
            start_position_mm=start_mm,
            direction=ts['direction'],
            kinetic_energy_mev=ts['momentum_mev'],
            step_size_mm=args.step_size_mm,
            track_id=1,
            min_energy_mev=args.min_energy_mev,
        )
        n_deposits = len(track['de'])
        if n_deposits == 0:
            print(f'warning: {ts["name"]} produced 0 deposits, skipping.', file=sys.stderr)
            continue
        events.append(dict(
            name=ts['name'],
            x=track['x'], y=track['y'], z=track['z'],
            theta=track['theta'], phi=track['phi'],
            de=track['de'], dx=track['dx'], track_id=track['track_id'],
        ))
        print(f'  {ts["name"]}  dir={ts["direction"]}  T={ts["momentum_mev"]} MeV  '
              f'{n_deposits:,} deposits')

    if not events:
        print('error: no events generated.', file=sys.stderr)
        raise SystemExit(1)

    save_events_h5(args.output, events)
    print(f'Saved {len(events)} event(s) to {args.output}')


if __name__ == '__main__':
    main()
