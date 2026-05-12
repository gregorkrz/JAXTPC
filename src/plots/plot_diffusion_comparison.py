#!/usr/bin/env python
"""
Compare wire-plane signals for two sets of diffusion parameters (GT1 vs GT2).

Each output PDF contains one page per track with three rows of panels:
  row 1 — GT1 wire-plane signals
  row 2 — GT2 wire-plane signals
  row 3 — difference  GT2 − GT1

All volumes are shown side by side (U1 V1 Y1 | U2 V2 Y2).
GT1 and GT2 share a symmetric colour scale; the difference row has its own.

Usage
-----
    # Compare nominal transverse diffusion vs 5× larger
    python src/plots/plot_diffusion_comparison.py \\
        --gt2-diffusion-trans 6e-5 \\
        --gt2-label "5× trans diffusion" \\
        --output-dir plots/diffusion_comparison/trans5x

    # Compare nominal longitudinal diffusion vs 10× larger
    python src/plots/plot_diffusion_comparison.py \\
        --gt2-diffusion-long 7.2e-5 \\
        --gt2-label "10× long diffusion" \\
        --tracks diagonal

    # Both diffusion values different, custom labels
    python src/plots/plot_diffusion_comparison.py \\
        --gt1-diffusion-trans 6e-6 --gt1-diffusion-long 3.6e-6 \\
        --gt2-diffusion-trans 2.4e-5 --gt2-diffusion-long 1.44e-5 \\
        --gt1-label "0.5× diffusion" --gt2-label "2× diffusion"
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

import argparse
import time

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import build_deposit_data
from tools.losses import make_sobolev_weight
from tools.noise import generate_noise
from tools.particle_generator import generate_muon_track
from tools.random_boundary_tracks import filter_track_inside_volumes

_PLOTS_DIR = os.environ.get('PLOTS_DIR', 'plots')

CONFIG_PATH        = 'config/cubic_wireplane_config.yaml'
N_SEGMENTS         = 50_000
MAX_ACTIVE_BUCKETS = 1000
GT_LIFETIME_US     = 10_000.0
GT_VELOCITY_CM_US  = 0.160
SOBOLEV_MAX_PAD    = 128  # default; overrideable via --sobolev-max-pad

TRACK_PRESETS = {
    'diagonal':        ((1.0,  1.0,  1.0),  1000.0),
    'diagonal_100MeV': ((1.0,  1.0,  1.0),   100.0),
    'X':               ((1.0,  0.0,  0.0),  1000.0),
    'Y':               ((0.0,  1.0,  0.0),  1000.0),
    'Z':               ((0.0,  0.0,  1.0),  1000.0),
    'U':               ((0.0,  0.866, 0.5), 1000.0),
    'V':               ((0.0, -0.866, 0.5), 1000.0),
    'track2':          ((0.5,  1.05, 0.2),   200.0),
}


VALID_PARAMS = (
    'velocity_cm_us', 'lifetime_us',
    'diffusion_trans_cm2_us', 'diffusion_long_cm2_us',
    'recomb_alpha', 'recomb_beta', 'recomb_beta_90', 'recomb_R',
)


def parse_param_overrides(s):
    """Parse 'name=val,name=val' into {name: float}. Empty/None → {}."""
    if not s:
        return {}
    result = {}
    for token in s.split(','):
        token = token.strip()
        if not token:
            continue
        if '=' not in token:
            raise argparse.ArgumentTypeError(
                f'Expected name=value, got {token!r}')
        name, val = token.split('=', 1)
        name = name.strip()
        if name not in VALID_PARAMS:
            raise argparse.ArgumentTypeError(
                f'Unknown param {name!r}. Valid: {VALID_PARAMS}')
        result[name] = float(val)
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tracks', default='diagonal',
                   help='"+"-separated preset names or name:dx,dy,dz:mom_mev. '
                        f'Presets: {", ".join(TRACK_PRESETS)}')

    p.add_argument('--gt1-params', default=None, metavar='name=val,...',
                   help='Comma-separated param overrides for GT1, e.g. '
                        '"diffusion_trans_cm2_us=2.4e-5,recomb_alpha=0.8". '
                        f'Valid names: {", ".join(VALID_PARAMS)}. '
                        'Unspecified params use simulator defaults.')
    p.add_argument('--gt1-label', default='GT1', help='Label for GT1 (default: GT1)')

    p.add_argument('--gt2-params', default=None, metavar='name=val,...',
                   help='Comma-separated param overrides for GT2 (same format as --gt1-params).')
    p.add_argument('--gt2-label', default='GT2', help='Label for GT2 (default: GT2)')

    p.add_argument('--output-dir',
                   default=os.path.join(_PLOTS_DIR, 'diffusion_comparison'),
                   help='Output directory (default: plots/diffusion_comparison)')
    p.add_argument('--noise-scale', type=float, default=0.0,
                   help='Noise amplitude (multiple of calibrated detector noise) '
                        'applied to GT1 when producing noisy per-pixel loss PDFs. '
                        '0 = no noisy PDFs (default). 1.0 = realistic noise.')
    p.add_argument('--noise-seed', type=int, default=0,
                   help='RNG seed for the noise draw (default: 0)')
    p.add_argument('--sobolev-max-pad', type=int, default=SOBOLEV_MAX_PAD,
                   metavar='N',
                   help=f'Padding size for Sobolev spectral weight (default: {SOBOLEV_MAX_PAD}). '
                        'Larger values capture lower spatial frequencies.')
    return p.parse_args()


def parse_tracks(tracks_str):
    specs = []
    for item in tracks_str.split('+'):
        item = item.strip()
        if ':' in item:
            parts = item.split(':')
            name      = parts[0]
            direction = tuple(float(x) for x in parts[1].split(','))
            momentum_mev = float(parts[2])
        elif item in TRACK_PRESETS:
            direction, momentum_mev = TRACK_PRESETS[item]
            name = item
        else:
            raise ValueError(f'Unknown track {item!r}. Presets: {list(TRACK_PRESETS)}')
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


def _apply_param(params, name, value):
    v = jnp.array(value)
    rp = params.recomb_params
    if name == 'velocity_cm_us':         return params._replace(velocity_cm_us=v)
    if name == 'lifetime_us':            return params._replace(lifetime_us=v)
    if name == 'diffusion_trans_cm2_us': return params._replace(diffusion_trans_cm2_us=v)
    if name == 'diffusion_long_cm2_us':  return params._replace(diffusion_long_cm2_us=v)
    if name == 'recomb_alpha':           return params._replace(recomb_params=rp._replace(alpha=v))
    if name == 'recomb_beta':            return params._replace(recomb_params=rp._replace(beta=v))
    if name == 'recomb_beta_90':         return params._replace(recomb_params=rp._replace(beta_90=v))
    if name == 'recomb_R':               return params._replace(recomb_params=rp._replace(R=v))
    raise ValueError(f'Unknown param {name!r}')


def make_params(simulator, overrides=None):
    """Return sim params at GT velocity/lifetime with any additional overrides applied."""
    params = simulator.default_sim_params._replace(
        lifetime_us    = jnp.array(GT_LIFETIME_US),
        velocity_cm_us = jnp.array(GT_VELOCITY_CM_US),
    )
    for name, value in (overrides or {}).items():
        params = _apply_param(params, name, value)
    return params


def get_arrays(simulator, track_spec, sim_params):
    track = generate_muon_track(
        start_position_mm=(0.0, 0.0, 0.0),
        direction=track_spec['direction'],
        kinetic_energy_mev=track_spec['momentum_mev'],
        step_size_mm=0.1,
        track_id=1,
    )
    track    = filter_track_inside_volumes(track, simulator.config.volumes)
    deposits = build_deposit_data(
        track['position'], track['de'], track['dx'], simulator.config,
        theta=track['theta'], phi=track['phi'],
        track_ids=track['track_id'],
    )
    arrays = simulator.forward(sim_params, deposits)
    jax.block_until_ready(arrays)
    return [np.asarray(a) for a in arrays]


def make_comparison_figure(gt1_arrays, gt2_arrays, col_labels, title, time_step_us,
                           gt1_label, gt2_label):
    """
    3-row × n_planes figure:
      row 0 — GT1
      row 1 — GT2
      row 2 — GT2 − GT1
    """
    n_planes = len(gt1_arrays)
    fig, axes = plt.subplots(3, n_planes, figsize=(3.5 * n_planes, 10),
                             constrained_layout=True)
    if n_planes == 1:
        axes = axes.reshape(3, 1)

    fig.suptitle(title, fontsize=10, y=1.01)

    # Shared symmetric colour scale for GT1 / GT2
    vmax_gt = max(
        np.nanpercentile(np.abs(np.concatenate(
            [a.ravel() for a in gt1_arrays + gt2_arrays])), 99),
        1e-9,
    )
    norm_gt = mcolors.Normalize(vmin=-vmax_gt, vmax=vmax_gt)

    # Difference colour scale
    diffs = [gt2_arrays[i] - gt1_arrays[i] for i in range(n_planes)]
    vmax_diff = max(
        np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in diffs])), 99),
        1e-9,
    )
    norm_diff = mcolors.Normalize(vmin=-vmax_diff, vmax=vmax_diff)

    rows = [
        (gt1_label,                  gt1_arrays, norm_gt,   'RdBu_r', 'signal (e⁻)'),
        (gt2_label,                  gt2_arrays, norm_gt,   'RdBu_r', 'signal (e⁻)'),
        (f'Δ  ({gt2_label}−{gt1_label})', diffs, norm_diff, 'RdBu_r', 'Δ signal (e⁻)'),
    ]

    for row_idx, (row_label, arrays, norm, cmap, cbar_label) in enumerate(rows):
        for col_idx, (arr, col_label) in enumerate(zip(arrays, col_labels)):
            ax = axes[row_idx, col_idx]
            n_wires, n_time = arr.shape
            t_max_us = n_time * time_step_us
            im = ax.imshow(
                arr, aspect='auto', origin='lower', norm=norm, cmap=cmap,
                extent=[0, t_max_us, 0, n_wires],
            )
            fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)
            ax.set_xlabel('time (μs)', fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f'{row_label}\nwire index', fontsize=8)
            else:
                ax.set_ylabel('wire index', fontsize=8)
            if row_idx == 0:
                ax.set_title(col_label, fontsize=9, fontweight='bold')

    return fig


def _signal_mid(arr, axis):
    """Mid index (along `axis`) of the range of rows/cols that carry signal."""
    profile = np.abs(arr).sum(axis=1 - axis)
    threshold = profile.max() * 0.01
    indices = np.where(profile > threshold)[0]
    if len(indices) == 0:
        return arr.shape[axis] // 2
    return int(indices[len(indices) // 2])


def _signal_range(arr, axis, margin=3):
    """(lo, hi) index range along `axis` where the signal is non-negligible.

    Includes `margin` extra indices on each side, clamped to array bounds.
    Works on the union of all curves passed (arr can be a list of arrays).
    """
    arrays = arr if isinstance(arr, (list, tuple)) else [arr]
    combined = np.abs(np.stack([a.sum(axis=1 - axis) for a in arrays])).max(axis=0)
    threshold = combined.max() * 0.01
    indices = np.where(combined > threshold)[0]
    if len(indices) == 0:
        n = arrays[0].shape[axis]
        return 0, n - 1
    lo = max(0, int(indices[0]) - margin)
    hi = min(arrays[0].shape[axis] - 1, int(indices[-1]) + margin)
    return lo, hi


def make_profiles_figure(gt1_arrays, gt2_arrays, col_labels, title, time_step_us,
                         gt1_label, gt2_label, gt1_noisy_arrays=None, noise_label=None):
    """
    2-row × n_planes figure of 1-D profiles:
      row 0 — signal vs wire index at mid-signal time step
      row 1 — signal vs time at mid-signal wire
    Each panel shows GT1, GT2, GT2−GT1, and optionally GT1+noise.
    """
    n_planes = len(gt1_arrays)
    fig, axes = plt.subplots(2, n_planes, figsize=(3.5 * n_planes, 6),
                             constrained_layout=True)
    if n_planes == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(title, fontsize=9, y=1.01)

    c1, c2, cd, cn = '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e'  # blue, red, green, orange
    noise_lbl = noise_label or f'{gt1_label}+noise'

    for col_idx, (a1, a2, col_label) in enumerate(zip(gt1_arrays, gt2_arrays, col_labels)):
        diff = a2 - a1
        n_wires, n_time = a1.shape
        t_axis = np.arange(n_time) * time_step_us
        an = gt1_noisy_arrays[col_idx] if gt1_noisy_arrays is not None else None

        range_arrays = [a1, a2, diff] + ([an] if an is not None else [])

        # ── row 0: signal vs wire at mid time ────────────────────────────────
        t_mid = _signal_mid(a1, axis=1)
        w_lo, w_hi = _signal_range(range_arrays, axis=0, margin=3)
        ax = axes[0, col_idx]
        wire_idx = np.arange(n_wires)
        ax.plot(wire_idx, a1[:, t_mid], color=c1, lw=1.2, label=gt1_label)
        ax.plot(wire_idx, a2[:, t_mid], color=c2, lw=1.2, ls='--', label=gt2_label)
        if an is not None:
            ax.plot(wire_idx, an[:, t_mid], color=cn, lw=0.8, ls='-.', label=noise_lbl)
        ax.plot(wire_idx, diff[:, t_mid], color=cd, lw=1.0, ls='-', label='Δ')
        ax.axhline(0, color='k', lw=0.4, ls='-')
        ax.set_xlim(w_lo, w_hi)
        ax.set_xlabel('wire index', fontsize=8)
        ax.set_ylabel('signal (e⁻)', fontsize=8)
        ax.set_title(f'{col_label}  [t={t_axis[t_mid]:.2f} μs]', fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='best')

        # ── row 1: signal vs time at mid wire ────────────────────────────────
        w_mid = _signal_mid(a1, axis=0)
        t_lo, t_hi = _signal_range(range_arrays, axis=1, margin=3)
        ax = axes[1, col_idx]
        ax.plot(t_axis, a1[w_mid, :], color=c1, lw=1.2, label=gt1_label)
        ax.plot(t_axis, a2[w_mid, :], color=c2, lw=1.2, ls='--', label=gt2_label)
        if an is not None:
            ax.plot(t_axis, an[w_mid, :], color=cn, lw=0.8, ls='-.', label=noise_lbl)
        ax.plot(t_axis, diff[w_mid, :], color=cd, lw=1.0, ls='-', label='Δ')
        ax.axhline(0, color='k', lw=0.4, ls='-')
        ax.set_xlim(t_axis[t_lo], t_axis[t_hi])
        ax.set_xlabel('time (μs)', fontsize=8)
        ax.set_ylabel('signal (e⁻)', fontsize=8)
        ax.set_title(f'{col_label}  [wire {w_mid}]', fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='best')

    return fig


def sobolev_loss_per_pixel(A, B, spectral_weight):
    """Per-pixel Sobolev loss map.  Sums to sobolev_loss_single(A, B, W)."""
    H, W = A.shape
    pad_h = (spectral_weight.shape[0] - H) // 2
    pad_w = (spectral_weight.shape[1] - W) // 2
    norm     = np.sum(np.abs(B)) + 1e-12
    diff     = (A - B) / norm
    diff_pad = np.pad(diff, ((pad_h, pad_h), (pad_w, pad_w)))
    diff_fft = np.fft.fft2(diff_pad)
    filtered = np.fft.ifft2(diff_fft * np.sqrt(spectral_weight)).real
    filtered = filtered[pad_h:pad_h + H, pad_w:pad_w + W]
    N = diff_pad.shape[0] * diff_pad.shape[1]
    return filtered ** 2 / N


def mse_loss_per_pixel(A, B):
    """Per-pixel MSE map.  Sums to mse_loss_single * N_pixels (= plane contribution)."""
    norm = np.sum(np.abs(B)) + 1e-12
    return ((A - B) / norm) ** 2


def l1_loss_per_pixel(A, B):
    """Per-pixel L1 map.  Sums to l1_loss_single * N_pixels (= plane contribution)."""
    norm = np.sum(np.abs(B)) + 1e-12
    return np.abs((A - B) / norm)


def _per_pixel_maps(pred, ref, max_pad=SOBOLEV_MAX_PAD):
    """Return (sobolev_map, mse_map, l1_map) for one plane pair."""
    sw = make_sobolev_weight(pred.shape[0], pred.shape[1], max_pad=max_pad)
    return (
        sobolev_loss_per_pixel(pred, ref, np.asarray(sw)),
        mse_loss_per_pixel(pred, ref),
        l1_loss_per_pixel(pred, ref),
    )


def make_per_pixel_figure(pred_arrays, ref_arrays, col_labels, title, time_step_us,
                          pred_label, ref_label, max_pad=SOBOLEV_MAX_PAD):
    """Per-pixel loss figure: 3 loss rows (Sobolev / MSE / L1) × n_planes columns.

    Shows loss(pred | ref) at each pixel with log-scale colorbars.
    Panel titles include the scalar sum of each map.
    """
    n_planes = len(pred_arrays)
    loss_names = ['Sobolev', 'MSE', 'L1']
    n_rows = len(loss_names)

    fig, axes = plt.subplots(n_rows, n_planes,
                             figsize=(3.5 * n_planes, 3.5 * n_rows),
                             constrained_layout=True)
    if n_planes == 1:
        axes = axes.reshape(n_rows, 1)

    fig.suptitle(
        f'Per-pixel losses  loss({pred_label} | {ref_label})  [max_pad={max_pad}]\n{title}',
        fontsize=9, y=1.01,
    )

    # Pre-compute all maps: list of (sob, mse, l1) per plane
    all_maps = [_per_pixel_maps(pred, ref, max_pad=max_pad)
                for pred, ref in zip(pred_arrays, ref_arrays)]

    for row_idx, loss_name in enumerate(loss_names):
        maps = [plane_maps[row_idx] for plane_maps in all_maps]
        all_vals = np.concatenate([m.ravel() for m in maps])
        pos_vals = all_vals[all_vals > 0]
        vmax = float(all_vals.max()) if len(pos_vals) else 1.0
        vmin = float(pos_vals.min()) if len(pos_vals) else vmax * 1e-6
        vmin = max(vmin, vmax * 1e-6)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

        for col_idx, (m, col_label) in enumerate(zip(maps, col_labels)):
            ax = axes[row_idx, col_idx]
            n_wires, n_time = m.shape
            t_max_us = n_time * time_step_us
            im = ax.imshow(
                np.clip(m, vmin, None),
                aspect='auto', origin='lower', norm=norm, cmap='hot_r',
                extent=[0, t_max_us, 0, n_wires],
            )
            ax.set_title(f'{col_label}  Σ={m.sum():.3g}', fontsize=8, fontweight='bold')
            ax.set_xlabel('time (μs)', fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f'{loss_name}\nwire index', fontsize=8)
            else:
                ax.set_ylabel('wire index', fontsize=8)
            fig.colorbar(im, ax=ax, label='loss / pixel', shrink=0.85)

    return fig


def apply_noise(arrays, simulator, noise_scale, noise_seed):
    """Add calibrated detector noise to a flat list of plane arrays."""
    cfg = simulator.config
    noise_dict = generate_noise(cfg, key=jax.random.PRNGKey(noise_seed))
    n_vols   = cfg.n_volumes
    n_planes = cfg.volumes[0].n_planes
    noisy = []
    for v in range(n_vols):
        for p in range(n_planes):
            arr   = arrays[v * n_planes + p]
            noise = np.asarray(noise_dict[(v, p)]) * noise_scale
            if noise.shape[0] < arr.shape[0]:
                noise = np.pad(noise, ((0, arr.shape[0] - noise.shape[0]), (0, 0)))
            noisy.append(arr + noise)
    return noisy


def main():
    args = parse_args()
    track_specs = parse_tracks(args.tracks)

    print(f'JAX devices: {jax.devices()}')
    print('Building simulator...')
    simulator = build_simulator()

    cfg          = simulator.config
    n_volumes    = cfg.n_volumes
    n_planes     = cfg.volumes[0].n_planes
    time_step_us = float(cfg.time_step_us)

    # Build flat column labels across all volumes: U1 V1 Y1 U2 V2 Y2
    col_labels = [
        f'{cfg.plane_names[v][p]}{v + 1}'
        for v in range(n_volumes)
        for p in range(n_planes)
    ]

    print('Warming up JIT...')
    t0 = time.time()
    simulator.warm_up()
    print(f'Done ({time.time() - t0:.1f} s)')

    # Parse and apply param overrides
    overrides1 = parse_param_overrides(args.gt1_params)
    overrides2 = parse_param_overrides(args.gt2_params)
    params1 = make_params(simulator, overrides1)
    params2 = make_params(simulator, overrides2)

    def _override_str(ov):
        return '  '.join(f'{k}={v:.3g}' for k, v in ov.items()) if ov else '(nominal)'

    print(f'\n{args.gt1_label}: {_override_str(overrides1)}')
    print(f'{args.gt2_label}: {_override_str(overrides2)}')

    os.makedirs(args.output_dir, exist_ok=True)

    for ts in track_specs:
        name = ts['name']
        print(f'\nTrack: {name}  dir={ts["direction"]}  T={ts["momentum_mev"]} MeV')

        arrays1 = get_arrays(simulator, ts, params1)
        arrays2 = get_arrays(simulator, ts, params2)

        # Flatten to one list per plane across all volumes
        planes1 = [arrays1[v * n_planes + p] for v in range(n_volumes) for p in range(n_planes)]
        planes2 = [arrays2[v * n_planes + p] for v in range(n_volumes) for p in range(n_planes)]

        title = (
            f'Parameter comparison  —  {name}  dir={ts["direction"]}  T={ts["momentum_mev"]:.0f} MeV\n'
            f'{args.gt1_label}: {_override_str(overrides1)}    '
            f'{args.gt2_label}: {_override_str(overrides2)}'
        )

        # Compute noisy GT1 once per track (used by both profiles and per-pixel figures)
        planes1_noisy = None
        noise_lbl = None
        if args.noise_scale > 0.0:
            planes1_noisy = apply_noise(planes1, simulator, args.noise_scale, args.noise_seed)
            noise_lbl = f'{args.gt1_label}+noise'

        out_path = os.path.join(args.output_dir, f'diffusion_comparison_{name}.pdf')
        with PdfPages(out_path) as pdf:
            fig = make_comparison_figure(
                planes1, planes2, col_labels, title,
                time_step_us, args.gt1_label, args.gt2_label,
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        print(f'  Saved: {out_path}')

        profiles_path = os.path.join(args.output_dir, f'diffusion_profiles_{name}.pdf')
        with PdfPages(profiles_path) as pdf:
            fig = make_profiles_figure(
                planes1, planes2, col_labels, title,
                time_step_us, args.gt1_label, args.gt2_label,
                gt1_noisy_arrays=planes1_noisy, noise_label=noise_lbl,
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        print(f'  Saved: {profiles_path}')

        # Per-pixel losses: clean reference (GT1)
        pxloss_path = os.path.join(args.output_dir, f'diffusion_per_pixel_{name}.pdf')
        with PdfPages(pxloss_path) as pdf:
            for pred_arrs, ref_arrs, pred_lbl, ref_lbl in [
                (planes2, planes1, args.gt2_label, args.gt1_label),
                (planes1, planes2, args.gt1_label, args.gt2_label),
            ]:
                fig = make_per_pixel_figure(
                    pred_arrs, ref_arrs, col_labels, title,
                    time_step_us, pred_lbl, ref_lbl,
                    max_pad=args.sobolev_max_pad,
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        print(f'  Saved: {pxloss_path}')

        # Per-pixel losses: noisy reference (GT1 + noise)
        if planes1_noisy is not None:
            pxloss_noise_path = os.path.join(
                args.output_dir, f'diffusion_per_pixel_{name}_noise.pdf')
            with PdfPages(pxloss_noise_path) as pdf:
                for pred_arrs, ref_arrs, pred_lbl, ref_lbl in [
                    (planes2, planes1_noisy, args.gt2_label, noise_lbl),
                    (planes1, planes1_noisy, args.gt1_label, noise_lbl),
                ]:
                    fig = make_per_pixel_figure(
                        pred_arrs, ref_arrs, col_labels, title,
                        time_step_us, pred_lbl, ref_lbl,
                        max_pad=args.sobolev_max_pad,
                    )
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            print(f'  Saved: {pxloss_noise_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
