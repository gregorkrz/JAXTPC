"""Plot GT event-display heatmaps from 1d_gradients pkl files.

For each parameter, produces one PDF page per track showing all planes.
When both clean and noisy pkls exist for the same param, they are shown
side-by-side so the track signal can be compared directly.

Usage (run from repo root on S3DF):
    python src/analysis/plot_gt_edisplay.py \
        --dir $RESULTS_DIR/1d_gradients/diffusion_debug_20260515_3tracks \
        --output plots/edisplay_debug.pdf
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ── bbox helpers ───────────────────────────────────────────────────────────────

BBOX_PAD_WIRE = 20
BBOX_PAD_TIME = 30
MAX_WIRE      = 300
MAX_TIME      = 500


def _signal_bbox_clean(arr):
    """Bbox from clean (no-noise) array via 2%-of-peak threshold."""
    nw, nt = arr.shape
    peak = float(np.abs(arr).max())
    if peak == 0:
        wl = max(0, nw // 2 - MAX_WIRE // 2)
        tl = max(0, nt // 2 - MAX_TIME // 2)
        return wl, min(nw, wl + MAX_WIRE), tl, min(nt, tl + MAX_TIME)

    mask = np.abs(arr) > 0.02 * peak
    wi = np.where(mask.any(axis=1))[0]
    ti = np.where(mask.any(axis=0))[0]
    wl = max(0,  wi[0]  - BBOX_PAD_WIRE)
    wh = min(nw, wi[-1] + BBOX_PAD_WIRE + 1)
    tl = max(0,  ti[0]  - BBOX_PAD_TIME)
    th = min(nt, ti[-1] + BBOX_PAD_TIME + 1)

    if wh - wl > MAX_WIRE:
        cw = int((wi[0] + wi[-1]) // 2)
        wl = max(0, cw - MAX_WIRE // 2)
        wh = min(nw, wl + MAX_WIRE)
        wl = max(0, wh - MAX_WIRE)
    if th - tl > MAX_TIME:
        ct = int((ti[0] + ti[-1]) // 2)
        tl = max(0, ct - MAX_TIME // 2)
        th = min(nt, tl + MAX_TIME)
        tl = max(0, th - MAX_TIME)
    return wl, wh, tl, th


def _peak_bbox(arr):
    """Bbox centred on the array peak — robust when noise fills the detector."""
    nw, nt = arr.shape
    pw, pt = np.unravel_index(np.abs(arr).argmax(), arr.shape)
    wl = max(0, pw - MAX_WIRE // 2)
    wh = min(nw, wl + MAX_WIRE)
    wl = max(0, wh - MAX_WIRE)
    tl = max(0, pt - MAX_TIME // 2)
    th = min(nt, tl + MAX_TIME)
    tl = max(0, th - MAX_TIME)
    return wl, wh, tl, th


# ── pkl loading ────────────────────────────────────────────────────────────────

def _plane_names(pkl):
    names = list(pkl.get('plane_names', []))
    counts = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    if any(c > 1 for c in counts.values()):
        vol, fixed = {}, []
        for n in names:
            vol[n] = vol.get(n, 0) + 1
            fixed.append(f'{n}{vol[n]}')
        return fixed
    return names


def load_pkls(directory):
    pkls = []
    for p in sorted(Path(directory).glob('*.pkl')):
        with open(p, 'rb') as f:
            d = pickle.load(f)
        if 'per_track_gt_arrays' not in d:
            print(f'  skip {p.name}: no per_track_gt_arrays')
            continue
        d['_path'] = p
        d['_plane_names'] = _plane_names(d)
        print(f'  loaded {p.name}  noise={d.get("noise_scale", 0):.2g}')
        pkls.append(d)
    return pkls


def group_by_param(pkls):
    """Return dict param_name → {'clean': pkl|None, 'noisy': pkl|None}."""
    by_param = defaultdict(dict)
    for d in pkls:
        key = 'noisy' if d.get('noise_scale', 0) > 0 else 'clean'
        by_param[d['param_name']][key] = d
    return dict(by_param)


# ── plotting ───────────────────────────────────────────────────────────────────

def _imshow(ax, crop, title, subtitle=''):
    absmax = float(np.abs(crop).max())
    if absmax == 0:
        absmax = 1.0
    ax.imshow(crop.T, aspect='auto', origin='lower',
              cmap='RdBu_r', vmin=-absmax, vmax=absmax,
              interpolation='nearest')
    ax.set_title(f'{title}\n{subtitle}' if subtitle else title,
                 fontsize=7, pad=2)
    ax.set_xlabel('Wire', fontsize=6)
    ax.set_ylabel('Time', fontsize=6)
    ax.tick_params(labelsize=5)
    ax.text(0.98, 0.02, f'max={absmax:.1f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=5, color='k',
            bbox=dict(fc='white', alpha=0.6, pad=1))


def plot_param(pdf, param_name, clean_pkl, noisy_pkl):
    variants = [v for v in [('clean', clean_pkl), ('noisy', noisy_pkl)]
                if v[1] is not None]

    track_names = [ts['name'] for ts in variants[0][1]['track_specs']]

    for track in track_names:
        n_cols = len(variants[0][1]['_plane_names'])
        n_rows = len(variants)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(2.5 * n_cols, 2.8 * n_rows),
            squeeze=False)
        fig.suptitle(f'{param_name} — {track}', fontsize=10, y=1.01)


        for row, (label, pkl) in enumerate(variants):
            gt_all = pkl['per_track_gt_arrays'][track]
            plane_names = pkl['_plane_names']

            for col, plane in enumerate(plane_names):
                arr = np.array(gt_all[col], dtype=np.float32)
                signal_max = float(np.abs(np.array(
                    clean_pkl['per_track_gt_arrays'][track][col]
                    if clean_pkl else gt_all[col])).max())
                subtitle = f'sig={signal_max:.1f}'
                _imshow(axes[row, col], arr,
                        title=f'{label} — {plane}', subtitle=subtitle)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f'    saved page: {param_name} / {track}')


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True,
                    help='Directory containing *.pkl files')
    ap.add_argument('--output', default=None,
                    help='Output PDF path (default: <dir>/edisplay.pdf)')
    args = ap.parse_args()

    dir_path = Path(args.dir)
    out_path = Path(args.output) if args.output else dir_path / 'edisplay.pdf'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Loading pkls from {dir_path} …')
    pkls = load_pkls(dir_path)
    if not pkls:
        print('No usable pkl files found.')
        sys.exit(1)

    groups = group_by_param(pkls)
    print(f'Found {len(groups)} param(s): {list(groups.keys())}')

    with PdfPages(out_path) as pdf:
        for param_name, variants in sorted(groups.items()):
            clean_pkl = variants.get('clean')
            noisy_pkl = variants.get('noisy')
            print(f'  Plotting {param_name} …')
            plot_param(pdf, param_name, clean_pkl, noisy_pkl)

    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
