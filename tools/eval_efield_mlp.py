#!/usr/bin/env python3
"""
Run forward inference of the learned E-field MLP on a 3D grid.

Loads finished Efield optimization result PKLs, evaluates the learned MLP
at each available weight snapshot (currently only 'final' — intermediate
snapshots require saving param_trajectory for the MLP block, which is not
done by default since n_record_coords=0 for Efield-only runs), loads the GT
field from the NPZ, and writes a side-by-side PKL next to the source result.

Works with all three MLP modes:
  potential   — E = E_bg − ∇δφ  (saves total E-field + distortion potential)
  efield      — E is the direct MLP output (saves total E-field)
  correction  — MLP outputs drift Δ(x,y,z) in cm  (saves corrections)

All arrays in the output are in **world frame** for direct comparison with
the GT NPZ.

Usage
-----
  python tools/eval_efield_mlp.py RESULT_PKL [RESULT_PKL ...]
  python tools/eval_efield_mlp.py --results-dir $RESULTS_DIR/opt/E_debug

Output
------
  {result_pkl_stem}_efield_eval.pkl  placed next to each result PKL.
  Shape of per-side arrays: (Nx, Ny, Nz, 3) for E-field / corrections,
                             (Nx, Ny, Nz)    for scalar potential.
"""
import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT))


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _world_to_local_pos(pts_world, x_anode_cm, drift_dir, yz_center_cm=(0.0, 0.0)):
    """(N,3) world-frame positions → volume-local frame.

    Local frame: x_local = drift_dir * (x_anode_cm - x_world),
    yz centered on the volume.
    """
    x_loc = drift_dir * (x_anode_cm - pts_world[:, 0])
    y_loc = pts_world[:, 1] - yz_center_cm[0]
    z_loc = pts_world[:, 2] - yz_center_cm[1]
    return np.stack([x_loc, y_loc, z_loc], axis=-1)


def _local_to_world_vec(vecs_local, drift_dir):
    """(N,3) vector local → world frame: Ex_world = -drift_dir * Ex_local."""
    out = vecs_local.copy()
    out[:, 0] = -drift_dir * vecs_local[:, 0]
    return out


# ── MLP helpers ───────────────────────────────────────────────────────────────

def _build_cfg(meta):
    """Reconstruct FieldConfig from result['efield'] metadata dict."""
    from tools.nonlocal_efield import FieldConfig
    return FieldConfig(
        mode=meta['mode'],
        hidden=tuple(meta['hidden']),
        center_cm=tuple(meta['center_cm']),
        half_cm=tuple(meta['half_cm']),
        bg_field_Vcm=tuple(meta['bg_field_Vcm']),
        out_scale=meta['out_scale'],
    )


def _unflatten_mlp(flat_p, meta):
    """Extract and unravel MLP weights from the full flat parameter vector."""
    from tools.nonlocal_efield import zero_params, flatten_params
    cfg = _build_cfg(meta)
    n_scalar = meta['n_scalar']
    n_weights = meta['n_weights']
    import jax.numpy as jnp
    flat_mlp = jnp.array(flat_p[n_scalar: n_scalar + n_weights], dtype=jnp.float32)
    _, unravel = flatten_params(zero_params(cfg))
    return unravel(flat_mlp), cfg


def _eval_grid_local(params, cfg, pts_local):
    """
    Evaluate MLP on (N,3) local-frame positions.

    Returns dict with keys depending on mode (all in local frame):
      efield_Vcm       — (N,3) total E-field, present for potential / efield
      corrections_cm   — (N,3) drift corrections, present for correction
      potential_Vcm_cm — (N,)  distortion potential δφ, present for potential
    """
    import jax
    import jax.numpy as jnp
    import tools.nonlocal_efield as _nl

    pts_j = jnp.array(pts_local, dtype=jnp.float32)
    field_fn = _nl.make_field_fn(params, cfg)
    field_vals = np.array(field_fn(pts_j))  # (N, 3)

    out = {}
    if cfg.mode == 'potential':
        pot_fn = jax.jit(jax.vmap(lambda p: _nl.potential(params, p, cfg)))
        out['potential_Vcm_cm'] = np.array(pot_fn(pts_j))  # (N,) distortion δφ
        out['efield_Vcm'] = field_vals                      # total field = bg − ∇δφ
    elif cfg.mode == 'efield':
        out['efield_Vcm'] = field_vals                      # total field = bg + MLP
    else:  # correction
        out['corrections_cm'] = field_vals
    return out


def _process_side(params, cfg, origin_cm, spacing_cm, shape,
                  x_anode_cm, drift_dir, yz_center_cm=(0.0, 0.0)):
    """
    Evaluate MLP on a full 3-D side grid.  Returns world-frame arrays.

    Saves exactly what the MLP directly outputs — no derived quantities:
      potential  → efield_Vcm + potential_Vcm_cm
      efield     → efield_Vcm
      correction → corrections_cm

    Shapes: efield_Vcm / corrections_cm → (Nx,Ny,Nz,3),
            potential_Vcm_cm            → (Nx,Ny,Nz).
    """
    Nx, Ny, Nz = shape

    xs = origin_cm[0] + np.arange(Nx) * spacing_cm[0]
    ys = origin_cm[1] + np.arange(Ny) * spacing_cm[1]
    zs = origin_cm[2] + np.arange(Nz) * spacing_cm[2]
    XW, YW, ZW = np.meshgrid(xs, ys, zs, indexing='ij')
    pts_world = np.stack([XW.ravel(), YW.ravel(), ZW.ravel()], axis=-1)  # (N,3)

    pts_local = _world_to_local_pos(pts_world, x_anode_cm, drift_dir, yz_center_cm)
    local_out = _eval_grid_local(params, cfg, pts_local)

    world_out = {}
    if 'efield_Vcm' in local_out:
        ef_world = _local_to_world_vec(local_out['efield_Vcm'], drift_dir)
        world_out['efield_Vcm'] = ef_world.reshape(Nx, Ny, Nz, 3)
    if 'corrections_cm' in local_out:
        corr_world = _local_to_world_vec(local_out['corrections_cm'], drift_dir)
        world_out['corrections_cm'] = corr_world.reshape(Nx, Ny, Nz, 3)
    if 'potential_Vcm_cm' in local_out:
        world_out['potential_Vcm_cm'] = local_out['potential_Vcm_cm'].reshape(Nx, Ny, Nz)

    return world_out


# ── Per-PKL processing ────────────────────────────────────────────────────────

def process_pkl(pkl_path, overwrite=False, last_only=False):
    pkl_path = Path(pkl_path)
    out_path = pkl_path.parent / (pkl_path.stem + '_efield_eval.pkl')

    if out_path.exists() and not overwrite:
        print(f'  [skip] {out_path.name} already exists')
        return

    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)

    meta = result.get('efield')
    if not meta or not meta.get('present'):
        print(f'  [skip] {pkl_path.name}: no efield metadata')
        return

    trials = result.get('trials', [])
    if not trials:
        print(f'  [skip] {pkl_path.name}: no trials')
        return

    # ── Load GT NPZ ──────────────────────────────────────────────────────────
    gt_path = meta.get('gt_map_path')
    gt_data = None
    if gt_path and os.path.exists(gt_path):
        d = np.load(gt_path)
        gt_data = {
            'east': {
                'efield_Vcm':     d['east_efield'],
                'corrections_cm': d['east_corrections'],
            },
            'west': {
                'efield_Vcm':     d['west_efield'],
                'corrections_cm': d['west_corrections'],
            },
        }
        grid = {
            'east': {
                'origin_cm':  d['east_origin'],
                'spacing_cm': d['east_spacing'],
                'shape':      tuple(d['east_efield'].shape[:3]),
            },
            'west': {
                'origin_cm':  d['west_origin'],
                'spacing_cm': d['west_spacing'],
                'shape':      tuple(d['west_efield'].shape[:3]),
            },
        }
        print(f'  GT loaded from {gt_path}')
    else:
        print(f'  [warn] GT NPZ not found at {gt_path!r}, skipping GT; using FieldConfig for grid')
        cfg_tmp = _build_cfg(meta)
        hx, hy, hz = cfg_tmp.half_cm
        N = 21
        sp = np.array([hx * 2 / (N - 1), hy * 2 / (N - 1), hz * 2 / (N - 1)])
        grid = {
            'east': {'origin_cm': np.array([-hx * 2, -hy, -hz]),  'spacing_cm': sp, 'shape': (N, N, N)},
            'west': {'origin_cm': np.array([0.0,      -hy, -hz]),  'spacing_cm': sp, 'shape': (N, N, N)},
        }

    # ── Volume geometry (for world ↔ local transforms) ────────────────────────
    # East: anode at x_min (origin[0]), drift_dir = -1
    # West: anode at x_max (origin + (N-1)*spacing), drift_dir = +1
    vol_geom = {
        'east': {
            'x_anode_cm': float(grid['east']['origin_cm'][0]),
            'drift_dir': -1,
            'yz_center_cm': (0.0, 0.0),
        },
        'west': {
            'x_anode_cm': float(
                grid['west']['origin_cm'][0]
                + (grid['west']['shape'][0] - 1) * grid['west']['spacing_cm'][0]
            ),
            'drift_dir': +1,
            'yz_center_cm': (0.0, 0.0),
        },
    }

    # ── Evaluate MLP for each trial ──────────────────────────────────────────
    # mlp_trajectory (list of (step, flat_p) pairs) is present when the run
    # used --mlp-snapshot-interval > 0.  Falls back to final_p only otherwise.
    step_snapshots = []
    for trial_idx, trial in enumerate(trials):
        mlp_traj = trial.get('mlp_trajectory')  # list of (step, flat_p) or None
        final_p  = trial.get('final_p')

        if mlp_traj and not last_only:
            snapshots = mlp_traj  # [(step, flat_p), ...]
            snapshot_src = f'mlp_trajectory ({len(snapshots)} snapshots)'
        elif final_p is not None:
            snapshots = [(trial.get('steps_run', '?'), final_p)]
            snapshot_src = 'final_p only'
        else:
            print(f'  [warn] trial {trial_idx}: no weights found, skipping')
            continue

        steps_run = trial.get('steps_run', '?')
        print(f'  trial {trial_idx}  mode={meta["mode"]}  steps={steps_run}'
              f'  source={snapshot_src}')

        for step, flat_p in snapshots:
            params, cfg = _unflatten_mlp(flat_p, meta)
            label = f'trial{trial_idx}_step{step}'
            print(f'    evaluating step {step}...')

            learned = {}
            for side, geom in vol_geom.items():
                g = grid[side]
                learned[side] = _process_side(
                    params, cfg,
                    origin_cm=g['origin_cm'],
                    spacing_cm=g['spacing_cm'],
                    shape=g['shape'],
                    x_anode_cm=geom['x_anode_cm'],
                    drift_dir=geom['drift_dir'],
                    yz_center_cm=geom['yz_center_cm'],
                )

            step_snapshots.append({
                'label': label,
                'trial_idx': trial_idx,
                'step': step,
                'learned': learned,
            })

    if not step_snapshots:
        print(f'  [skip] {pkl_path.name}: nothing to save')
        return

    output = {
        'source_pkl': str(pkl_path),
        'efield_meta': meta,
        'grid': grid,
        'vol_geom': vol_geom,
        'gt': gt_data,
        'steps': step_snapshots,
    }

    with open(out_path, 'wb') as f:
        pickle.dump(output, f)
    print(f'  saved → {out_path}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('result_pkls', nargs='*',
                        help='Result PKL paths to process directly')
    parser.add_argument('--results-dir', default=None,
                        help='Scan this directory recursively for result_*.pkl files')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-run even if output PKL already exists')
    parser.add_argument('--last-only', action='store_true',
                        help='Only evaluate the final snapshot per trial, '
                             'ignoring mlp_trajectory even if present')
    args = parser.parse_args()

    pkls = list(args.result_pkls)
    if args.results_dir:
        pkls += sorted(glob.glob(
            os.path.join(args.results_dir, '**', 'result_*.pkl'), recursive=True))
    pkls = [p for p in pkls if not p.endswith('_efield_eval.pkl')]

    if not pkls:
        print('No PKLs found.')
        return

    for pkl in pkls:
        print(f'Processing {pkl}')
        try:
            process_pkl(pkl, overwrite=args.overwrite, last_only=args.last_only)
        except Exception as exc:
            import traceback
            print(f'  ERROR: {exc}')
            traceback.print_exc()


if __name__ == '__main__':
    main()
