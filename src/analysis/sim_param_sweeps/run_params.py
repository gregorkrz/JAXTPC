#!/usr/bin/env python
"""
Sweep forward-simulation parameters across a track ensemble and random point
deposits, recording main-wire + main-time signals with and without noise.

The full combo grid is split into N_CHUNKS chunks.  Each chunk is saved as a
separate pkl file as soon as it finishes.  Re-running the script skips any
chunk whose file already exists and passes a basic integrity check.

Deposit sources
---------------
  • Track ensemble  — 15 muon tracks (12 random boundary + 3 fixed chords)
    from generate_random_boundary_tracks, at TRACK_STEP_MM step size.
  • Point deposits  — N_POINT_DEPOSITS random single-step deposits distributed
    uniformly inside the East TPC, with random orientation angles.

For each deposit, main wire and main time indices are fixed from a nominal-
params reference run (computed once before chunking); every sweep entry stores
the signal at those same indices, so results are directly comparable.

Output files
------------
One pkl per chunk:  <output-dir>/sweep_<tag>_chunk<i>of<N>.pkl

Each chunk file has the same structure:
    chunk['meta']                           # run configuration
    chunk['chunk_idx'], chunk['n_chunks']
    chunk['combo_indices']                  # global indices into the full grid
    chunk['combos']                         # list of {params: {name: value}}
    chunk['deposits'][name]['type']         # 'track' | 'point'
    chunk['deposits'][name]['source_meta']
    chunk['deposits'][name]['reference']    # {plane: {wire, time, t_lo, t_hi}}
    chunk['deposits'][name]['sweep'][i]     # results for combos[i] in this chunk
    chunk['deposits'][name]['sweep'][i]['clean'][plane]['peak']
    chunk['deposits'][name]['sweep'][i]['clean'][plane]['trace']  # if STORE_TRACES
    chunk['deposits'][name]['sweep'][i]['noisy'][plane]['peak']

Usage
-----
    python src/analysis/sim_param_sweeps/run_params.py \\
        --output-dir results/diffusion_sweep
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import argparse
import itertools
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tools.geometry import generate_detector
from tools.loader import build_deposit_data
from tools.noise import generate_noise
from tools.particle_generator import generate_muon_track
from tools.random_boundary_tracks import (
    generate_random_boundary_tracks,
    filter_track_inside_volumes,
)
from tools.simulation import DetectorSimulator

# ── JAX compilation cache ─────────────────────────────────────────────────────
_JAX_CACHE_DIR = os.environ.get('JAX_COMPILATION_CACHE_DIR',
                                os.path.expanduser('~/.cache/jax_compilation_cache'))
jax.config.update('jax_compilation_cache_dir', _JAX_CACHE_DIR)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# SWEEP CONFIGURATION — edit this section to change what is swept
# ═════════════════════════════════════════════════════════════════════════════
#
# Each entry: (param_name, 1-D array of values).  All combinations are taken
# (cartesian product).  Supported param names:
#   velocity_cm_us, lifetime_us,
#   diffusion_trans_cm2_us, diffusion_long_cm2_us,
#   recomb_alpha, recomb_beta_90
#
# Nominal detector-config defaults:
#   diffusion_trans_cm2_us  ≈ 1.2e-5 cm²/μs
#   diffusion_long_cm2_us   ≈ 7.2e-6 cm²/μs

SWEEP_PARAMS = [
    ('diffusion_trans_cm2_us', np.linspace(0.5e-5, 5.0e-5, 10)),
    ('diffusion_long_cm2_us',  np.linspace(0.5e-6, 5.0e-5, 10)),
]

# ── Chunking ──────────────────────────────────────────────────────────────────
N_CHUNKS = 20   # combo grid is split into this many files; finished chunks are skipped on re-run

# ── Track ensemble ────────────────────────────────────────────────────────────
TRACK_N_BOUNDARY = 12      # random boundary-start muons; 3 fixed chords always added → 15 total
TRACK_SEED       = 42
TRACK_STEP_MM    = 1.0     # step size in mm for generate_muon_track

# ── Random point deposits ─────────────────────────────────────────────────────
N_POINT_DEPOSITS = 10      # number of random single-step deposits
POINT_SEED       = 0       # RNG seed for positions and angles
POINT_DE_MEV     = 0.3     # energy deposit [MeV]
POINT_DX_MM      = 3.0     # step length [mm]

# ── Noise ─────────────────────────────────────────────────────────────────────
NOISE_SEED  = 42
NOISE_SCALE = 1.0          # multiplier on the ADC-unit noise from generate_noise()

# ── Output ────────────────────────────────────────────────────────────────────
STORE_TRACES   = True      # store windowed 1-D trace at main wire (uses more disk)
TRACE_HALF_WIN = 150       # half-window in time bins around reference peak
WIRE_HALF_WIN  = 100       # half-window in wire channels around reference wire

# ── Simulator ─────────────────────────────────────────────────────────────────
CONFIG_PATH        = 'config/cubic_wireplane_config.yaml'
N_SEGMENTS         = 50_000   # must cover the longest track at TRACK_STEP_MM
MAX_ACTIVE_BUCKETS = 1000

# ═════════════════════════════════════════════════════════════════════════════


# ── Parameter application ─────────────────────────────────────────────────────

def _apply_param(name, value, params):
    """Return updated SimParams with one field replaced by value."""
    v = jnp.array(float(value))
    if name == 'diffusion_trans_cm2_us':
        return params._replace(diffusion_trans_cm2_us=v)
    if name == 'diffusion_long_cm2_us':
        return params._replace(diffusion_long_cm2_us=v)
    if name == 'velocity_cm_us':
        return params._replace(velocity_cm_us=v)
    if name == 'lifetime_us':
        return params._replace(lifetime_us=v)
    if name == 'recomb_alpha':
        return params._replace(recomb_params=params.recomb_params._replace(alpha=v))
    if name == 'recomb_beta_90':
        return params._replace(recomb_params=params.recomb_params._replace(beta_90=v))
    raise ValueError(f'Unknown param: {name!r}')


# ── Signal utilities ──────────────────────────────────────────────────────────

def pick_main(sig, axis='wire'):
    """Return (idx, score) for the wire or time bin with the largest mean |signal|.

    axis='wire' — average over time (axis=1), argmax over wires.
    axis='time' — average over wires (axis=0), argmax over time bins.
    """
    scores = np.abs(sig).mean(axis=1 if axis == 'wire' else 0)
    idx = int(np.argmax(scores))
    return idx, float(scores[idx])


def _plane_index(cfg):
    """Return (plane_names_flat, vol_indices_flat) lists."""
    names, vols = [], []
    for v in range(cfg.n_volumes):
        for name in cfg.plane_names[v]:
            names.append(name)
            vols.append(v)
    return names, vols


def _run(sim, params, deposits):
    sigs = sim.forward(params, deposits)
    jax.block_until_ready(sigs)
    return [np.array(s) for s in sigs]


def _add_noise(signals, noise_dict, cfg):
    """Return signals + noise_dict * NOISE_SCALE for every plane."""
    noisy = []
    i = 0
    for v in range(cfg.n_volumes):
        for p in range(cfg.volumes[v].n_planes):
            sig   = signals[i]
            noise = np.array(noise_dict[(v, p)]) * NOISE_SCALE
            if noise.shape[0] < sig.shape[0]:
                noise = np.pad(noise, ((0, sig.shape[0] - noise.shape[0]), (0, 0)))
            elif noise.shape[0] > sig.shape[0]:
                noise = noise[:sig.shape[0]]
            noisy.append(sig + noise)
            i += 1
    return noisy


def _reference_indices(signals, plane_names, vol_indices, vol_idx):
    """Compute per-plane {wire, time, t_lo, t_hi, w_lo, w_hi} from a reference signal run."""
    ref = {}
    for sig, name, vi in zip(signals, plane_names, vol_indices):
        if vi != vol_idx:
            continue
        wire, _     = pick_main(sig, axis='wire')
        time_bin, _ = pick_main(sig, axis='time')
        t_lo = max(0, time_bin - TRACE_HALF_WIN)
        t_hi = min(sig.shape[1], time_bin + TRACE_HALF_WIN + 1)
        w_lo = max(0, wire - WIRE_HALF_WIN)
        w_hi = min(sig.shape[0], wire + WIRE_HALF_WIN + 1)
        ref[name] = {'wire': wire, 'time': time_bin,
                     't_lo': t_lo, 't_hi': t_hi,
                     'w_lo': w_lo, 'w_hi': w_hi}
    return ref


def _extract(signals, plane_names, vol_indices, vol_idx, ref):
    """Extract peak, windowed time trace, and windowed wire trace at reference indices."""
    results = {}
    for sig, name, vi in zip(signals, plane_names, vol_indices):
        if vi != vol_idx:
            continue
        w, t = ref[name]['wire'], ref[name]['time']
        trace_full = sig[w, :]
        entry = {'peak': float(trace_full[t])}
        if STORE_TRACES:
            t_lo, t_hi = ref[name]['t_lo'], ref[name]['t_hi']
            entry['trace'] = trace_full[t_lo:t_hi].astype(np.float32)
            entry['t_lo']  = t_lo
            w_lo, w_hi = ref[name]['w_lo'], ref[name]['w_hi']
            entry['wire_trace'] = sig[:, t][w_lo:w_hi].astype(np.float32)
            entry['w_lo']       = w_lo
        results[name] = entry
    return results


# ── Deposit construction ──────────────────────────────────────────────────────

def generate_random_point_specs(seed, n, volumes):
    """Generate specs for n random single-step deposits in the East TPC (vol 0).

    Positions are sampled uniformly within the volume bounds; angles are sampled
    uniformly over the sphere.
    """
    rng = np.random.default_rng(seed)
    vol = volumes[0]
    lo = [c[0] * 10.0 for c in vol.ranges_cm]
    hi = [c[1] * 10.0 for c in vol.ranges_cm]
    specs = []
    for i in range(n):
        specs.append({
            'name':        f'point_{i}',
            'position_mm': [float(rng.uniform(lo[j], hi[j])) for j in range(3)],
            'theta':       float(rng.uniform(0.0, np.pi)),
            'phi':         float(rng.uniform(0.0, 2.0 * np.pi)),
            'de_mev':      POINT_DE_MEV,
            'dx_mm':       POINT_DX_MM,
        })
    return specs


def _build_track_deposit(spec, cfg):
    """Build DepositData from a track spec dict.  Returns (deposits, meta) or (None, None)."""
    track = generate_muon_track(
        start_position_mm=spec['start_position_mm'],
        direction=spec['direction'],
        kinetic_energy_mev=spec['momentum_mev'],
        step_size_mm=TRACK_STEP_MM,
        track_id=1,
    )
    track = filter_track_inside_volumes(track, cfg.volumes)
    n_steps = len(track['de'])
    if n_steps == 0:
        return None, None
    deposits = build_deposit_data(
        track['position'], track['de'], track['dx'], cfg,
        theta=track['theta'], phi=track['phi'],
        track_ids=track['track_id'],
    )
    meta = {
        'momentum_mev':      spec['momentum_mev'],
        'direction':         spec['direction'],
        'start_position_mm': spec['start_position_mm'],
        'step_size_mm':      TRACK_STEP_MM,
        'n_steps':           n_steps,
    }
    return deposits, meta


def _build_point_deposit(spec, cfg):
    """Build DepositData from a single-point spec dict."""
    pos   = np.array([spec['position_mm']], dtype=np.float32)
    de    = np.array([spec['de_mev']],      dtype=np.float32)
    dx    = np.array([spec['dx_mm']],       dtype=np.float32)
    theta = np.array([spec['theta']],       dtype=np.float32)
    phi   = np.array([spec['phi']],         dtype=np.float32)
    return build_deposit_data(pos, de, dx, cfg, theta=theta, phi=phi)


# ── Sweep ─────────────────────────────────────────────────────────────────────

def _sweep_combos(deposit_data, combos, param_names, base_params,
                  sim, plane_names, vol_indices, vol_idx, noise_dict, reference):
    """Run a list of param combos for one deposit. Returns a sweep list."""
    sweep = []
    for combo in combos:
        params = base_params
        for name, value in zip(param_names, combo):
            params = _apply_param(name, value, params)
        sigs       = _run(sim, params, deposit_data)
        noisy_sigs = _add_noise(sigs, noise_dict, sim.config)
        sweep.append({
            'clean': _extract(sigs,       plane_names, vol_indices, vol_idx, reference),
            'noisy': _extract(noisy_sigs, plane_names, vol_indices, vol_idx, reference),
        })
    return sweep


# ── Chunk I/O ─────────────────────────────────────────────────────────────────

def _chunk_path(out_dir, sweep_tag, chunk_idx, n_chunks):
    return out_dir / f'sweep_{sweep_tag}_chunk{chunk_idx:03d}of{n_chunks:03d}.pkl'


def _chunk_is_valid(path, expected_n_combos, expected_deposit_names):
    """Return True if chunk file exists, loads cleanly, and has the right shape."""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        assert set(data['deposits'].keys()) == set(expected_deposit_names)
        for dep in data['deposits'].values():
            assert len(dep['sweep']) == expected_n_combos
            if STORE_TRACES:
                first = dep['sweep'][0]
                first_plane = next(iter(first['clean']))
                assert 'wire_trace' in first['clean'][first_plane]
        return True
    except Exception:
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', default='results/diffusion_sweep',
                   help='Output directory for chunk pkl files (default: results/diffusion_sweep)')
    p.add_argument('--config', default=CONFIG_PATH,
                   help=f'Detector config YAML (default: {CONFIG_PATH})')
    p.add_argument('--vol-idx', type=int, default=0,
                   help='Detector volume to analyse (default: 0 = East TPC)')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    param_names = [n for n, _ in SWEEP_PARAMS]
    param_grids = [v for _, v in SWEEP_PARAMS]
    combos      = list(itertools.product(*param_grids))

    # Split combos into N_CHUNKS (ceiling division)
    chunk_size   = (len(combos) + N_CHUNKS - 1) // N_CHUNKS
    combo_chunks = [combos[i:i + chunk_size] for i in range(0, len(combos), chunk_size)]
    n_chunks     = len(combo_chunks)   # may be < N_CHUNKS if len(combos) < N_CHUNKS

    sweep_tag = '_vs_'.join(
        n.replace('diffusion_trans_cm2_us', 'diff_trans')
         .replace('diffusion_long_cm2_us',  'diff_long')
         .replace('_cm2_us', '').replace('_cm_us', '').replace('_us', '')
        for n in param_names
    )

    print('Sweep parameters:')
    for name, vals in SWEEP_PARAMS:
        print(f'  {name}: {len(vals)} values  [{vals[0]:.3g} … {vals[-1]:.3g}]')
    print(f'Total combinations : {len(combos)}  →  {n_chunks} chunks of ≤{chunk_size}')
    print(f'Track ensemble     : {TRACK_N_BOUNDARY} random + 3 fixed = {TRACK_N_BOUNDARY + 3} tracks')
    print(f'Point deposits     : {N_POINT_DEPOSITS}')
    print()

    # ── Simulator ────────────────────────────────────────────────────────────
    detector_config = generate_detector(args.config)
    sim = DetectorSimulator(
        detector_config,
        differentiable=True,
        n_segments=N_SEGMENTS,
        max_active_buckets=MAX_ACTIVE_BUCKETS,
        include_noise=False,
        include_electronics=False,
        include_track_hits=False,
        include_digitize=False,
    )
    print('Warming up JIT...')
    t0 = time.time()
    sim.warm_up()
    print(f'Done ({time.time()-t0:.1f} s)\n')

    cfg         = sim.config
    base_params = sim.default_sim_params
    plane_names, vol_indices = _plane_index(cfg)
    vol_plane_names = [n for n, vi in zip(plane_names, vol_indices) if vi == args.vol_idx]

    # ── Precompute noise (one draw, shared across all deposits and combos) ────
    noise_dict = generate_noise(cfg, key=jax.random.PRNGKey(NOISE_SEED))
    print(f'Noise precomputed  (seed={NOISE_SEED}, scale={NOISE_SCALE})\n')

    # ── Build deposit registry ────────────────────────────────────────────────
    # (name, deposit_data, type, source_meta)
    deposit_registry = []

    print(f'Building track deposits ({TRACK_N_BOUNDARY + 3} tracks, step={TRACK_STEP_MM} mm)...')
    for spec in generate_random_boundary_tracks(cfg.volumes, n=TRACK_N_BOUNDARY, seed=TRACK_SEED):
        dep, meta = _build_track_deposit(spec, cfg)
        if dep is None:
            print(f'  SKIP {spec["name"]}: 0 steps inside volume')
            continue
        deposit_registry.append((spec['name'], dep, 'track', meta))
        print(f'  {spec["name"]:45s} {meta["n_steps"]:5d} steps')

    print(f'\nBuilding point deposits ({N_POINT_DEPOSITS} deposits)...')
    for spec in generate_random_point_specs(POINT_SEED, N_POINT_DEPOSITS, cfg.volumes):
        dep = _build_point_deposit(spec, cfg)
        deposit_registry.append((spec['name'], dep, 'point', spec))
        pos = ', '.join(f'{x:.0f}' for x in spec['position_mm'])
        print(f'  {spec["name"]}  pos=({pos}) mm')

    deposit_names = [name for name, *_ in deposit_registry]
    print(f'\nTotal deposits: {len(deposit_registry)}\n')

    # ── Compute references once (nominal params, one pass per deposit) ────────
    print('Computing reference wire/time indices (nominal params)...')
    references = {}
    for name, dep, _, _ in deposit_registry:
        ref_sigs        = _run(sim, base_params, dep)
        references[name] = _reference_indices(ref_sigs, plane_names, vol_indices, args.vol_idx)
        ref_str = '  '.join(f'{p}:w{references[name][p]["wire"]}/t{references[name][p]["time"]}'
                            for p in vol_plane_names)
        print(f'  {name:45s} [{ref_str}]')
    print()

    # ── Shared metadata written into every chunk file ─────────────────────────
    meta = {
        'param_names':      param_names,
        'param_grids':      {n: v.tolist() for n, v in SWEEP_PARAMS},
        'vol_idx':          args.vol_idx,
        'plane_names':      vol_plane_names,
        'noise_seed':       NOISE_SEED,
        'noise_scale':      NOISE_SCALE,
        'track_step_mm':    TRACK_STEP_MM,
        'track_n_boundary': TRACK_N_BOUNDARY,
        'track_seed':       TRACK_SEED,
        'n_point_deposits': N_POINT_DEPOSITS,
        'point_seed':       POINT_SEED,
        'store_traces':     STORE_TRACES,
        'trace_half_win':   TRACE_HALF_WIN,
        'config_path':      args.config,
        'n_chunks':         n_chunks,
    }

    # ── Chunk loop ────────────────────────────────────────────────────────────
    t_total = time.time()
    n_done  = 0

    for chunk_idx, combo_chunk in enumerate(combo_chunks):
        path = _chunk_path(out_dir, sweep_tag, chunk_idx, n_chunks)

        if _chunk_is_valid(path, len(combo_chunk), deposit_names):
            print(f'[{chunk_idx+1:>{len(str(n_chunks))}}/{n_chunks}] '
                  f'SKIP  {path.name}  (already complete)')
            n_done += 1
            continue

        print(f'[{chunk_idx+1:>{len(str(n_chunks))}}/{n_chunks}] '
              f'chunk {chunk_idx}  ({len(combo_chunk)} combos × {len(deposit_registry)} deposits) ...',
              flush=True)
        t0 = time.time()

        chunk_deposits = {}
        for name, dep, dep_type, source_meta in deposit_registry:
            sweep = _sweep_combos(
                dep, combo_chunk, param_names, base_params,
                sim, plane_names, vol_indices, args.vol_idx, noise_dict, references[name],
            )
            chunk_deposits[name] = {
                'type':        dep_type,
                'source_meta': source_meta,
                'reference':   references[name],
                'sweep':       sweep,
            }

        global_start = chunk_idx * chunk_size
        payload = {
            'meta':          meta,
            'chunk_idx':     chunk_idx,
            'n_chunks':      n_chunks,
            'combo_indices': list(range(global_start, global_start + len(combo_chunk))),
            'combos':        [{'params': dict(zip(param_names, [float(v) for v in c]))}
                              for c in combo_chunk],
            'deposits':      chunk_deposits,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)

        elapsed = time.time() - t0
        n_done += 1
        print(f'  → saved {path.name}  ({elapsed:.0f} s, '
              f'{elapsed / (len(combo_chunk) * len(deposit_registry)) * 1000:.0f} ms/combo/deposit)')

    print(f'\nAll {n_chunks} chunks done in {(time.time()-t_total)/60:.1f} min.')
    print(f'Results in {out_dir}/')


if __name__ == '__main__':
    main()
