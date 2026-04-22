#!/usr/bin/env python
"""
Combine two or more 2-D loss landscape pkl files by summing their losses
(and gradients, if present) on a shared (alpha, beta_90) grid.

All input pkls must share the same loss_name, grid_size, alpha_vals, and
beta90_vals.  Gradients are summed only when every pkl contains them.

Usage
-----
    python combine_landscapes.py \\
        results/track1/landscape_sobolev_loss_diagonal_50x50.pkl \\
        results/track2/landscape_sobolev_loss_track2_50x50.pkl \\
        --output-dir results/2d_landscape_two_tracks

    # Mix any number of pkls; output name is auto-generated
    python combine_landscapes.py a.pkl b.pkl c.pkl --output-dir results/combined
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import os

_RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')
_PLOTS_DIR   = os.environ.get('PLOTS_DIR',   'plots')
import pickle

import importlib.util
import numpy as np

# Re-use plotting functions from the landscape script (name starts with digit,
# so importlib is needed instead of a regular import statement)
_spec = importlib.util.spec_from_file_location(
    'landscape_mod', os.path.join(os.path.dirname(__file__), '2d_loss_landscape.py')
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
plot_landscape = _mod.plot_landscape
LOSS_LABELS    = _mod.LOSS_LABELS


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pkls', nargs='+',
                   help='Two or more landscape pkl files to combine')
    p.add_argument('--output-dir', default=os.path.join(_RESULTS_DIR, '2d_landscape_combined'),
                   help='Directory for combined pkl and plots')
    p.add_argument('--output-name', default=None,
                   help='Base filename (without extension) for outputs; '
                        'auto-generated from loss name and track count if omitted')
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def grids_compatible(landscapes):
    """Return True if all landscapes share the same loss, grid, and param ranges."""
    ref = landscapes[0]
    for other in landscapes[1:]:
        if other['loss_name'] != ref['loss_name']:
            raise ValueError(
                f"Loss mismatch: {ref['loss_name']!r} vs {other['loss_name']!r}.\n"
                "All input pkls must use the same loss function."
            )
        if other['grid_size'] != ref['grid_size']:
            raise ValueError(
                f"Grid size mismatch: {ref['grid_size']} vs {other['grid_size']}."
            )
        if not np.allclose(other['alpha_vals'], ref['alpha_vals'], rtol=1e-5):
            raise ValueError("alpha_vals do not match across pkl files.")
        if not np.allclose(other['beta90_vals'], ref['beta90_vals'], rtol=1e-5):
            raise ValueError("beta90_vals do not match across pkl files.")
    return True


# ── Combination ────────────────────────────────────────────────────────────────

def combine(landscapes, output_name_hint=None):
    grids_compatible(landscapes)
    ref = landscapes[0]
    n   = len(landscapes)

    combined_grid = sum(np.array(ld['grid']) for ld in landscapes)

    has_grads = all('grad_alpha' in ld and 'grad_beta90' in ld for ld in landscapes)
    combined_ga = combined_gb = None
    if has_grads:
        combined_ga = sum(np.array(ld['grad_alpha'])  for ld in landscapes)
        combined_gb = sum(np.array(ld['grad_beta90']) for ld in landscapes)

    track_names = [ld['track_name'] for ld in landscapes]
    combined_track = '+'.join(track_names)

    directions = [ld['direction'] for ld in landscapes]

    landscape = dict(
        loss_name    = ref['loss_name'],
        track_name   = combined_track,
        direction    = directions,          # list of directions
        momentum_mev = ref['momentum_mev'],
        grid_size    = ref['grid_size'],
        range_frac   = ref['range_frac'],
        gt_alpha     = ref['gt_alpha'],
        gt_beta90    = ref['gt_beta90'],
        alpha_vals   = ref['alpha_vals'],
        beta90_vals  = ref['beta90_vals'],
        grid         = combined_grid.tolist(),
        n_tracks     = n,
        source_pkls  = [os.path.basename(ld.get('_source', '?')) for ld in landscapes],
    )
    if has_grads:
        landscape['grad_alpha']  = combined_ga.tolist()
        landscape['grad_beta90'] = combined_gb.tolist()

    N = ref['grid_size']
    loss_name = ref['loss_name']
    base = output_name_hint or f'landscape_{loss_name}_combined_{n}tracks_{N}x{N}'
    return landscape, base


# ── Patched plot title ─────────────────────────────────────────────────────────
# plot_landscape uses landscape['track_name'] and landscape['direction'] in the
# title.  We monkey-patch LOSS_LABELS so the combined label is clear, and the
# track_name already carries the merged names.

def _patch_direction_str(landscape):
    """Return a short string describing the combined directions."""
    dirs = landscape['direction']
    if isinstance(dirs, list):
        return ' | '.join(str(d) for d in dirs)
    return str(dirs)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if len(args.pkls) < 2:
        raise SystemExit('Need at least 2 pkl files to combine.')

    print(f'Loading {len(args.pkls)} pkl files...')
    landscapes = []
    for path in args.pkls:
        ld = load_pkl(path)
        ld['_source'] = path
        landscapes.append(ld)
        print(f'  {path}  loss={ld["loss_name"]}  track={ld["track_name"]}'
              f'  grid={ld["grid_size"]}x{ld["grid_size"]}'
              f'  grads={"yes" if "grad_alpha" in ld else "no"}')

    combined, base = combine(landscapes, output_name_hint=args.output_name)

    os.makedirs(args.output_dir, exist_ok=True)
    pkl_path = os.path.join(args.output_dir, base + '.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(combined, f)
    print(f'\nSaved combined pkl: {pkl_path}')

    # Fix direction string for plot titles (plot_landscape reads it as a plain value)
    combined['direction'] = _patch_direction_str(combined)

    print('Plotting...')
    plot_landscape(combined, args.output_dir)
    print('Done.')


if __name__ == '__main__':
    main()
