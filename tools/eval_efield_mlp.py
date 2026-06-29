#!/usr/bin/env python3
"""
Run forward inference of the learned E-field model on a 3D grid.

Loads Efield optimization result PKLs (finished or in-progress via
``live_checkpoint``), evaluates the learned SIREN or legacy FieldConfig MLP
at each available weight snapshot, loads the GT field from the NPZ, and
writes a side-by-side PKL next to the source result.

For SIREN runs (``mode='siren'``, the default), the output contains:
  efield_Vcm     — (Nx,Ny,Nz,3) total E-field in V/cm
  corrections_cm — (Nx,Ny,Nz,3) drift distortions Δ(r) in cm

For legacy FieldConfig runs (potential/efield/correction modes) the keys
depend on the mode, same as before.

All arrays are in **world frame** for direct comparison with the GT NPZ.

Usage
-----
  python tools/eval_efield_mlp.py RESULT_PKL [RESULT_PKL ...]
  python tools/eval_efield_mlp.py --results-dir $RESULTS_DIR/opt/E_debug
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
    """(N,3) world-frame positions → volume-local frame."""
    x_loc = drift_dir * (x_anode_cm - pts_world[:, 0])
    y_loc = pts_world[:, 1] - yz_center_cm[0]
    z_loc = pts_world[:, 2] - yz_center_cm[1]
    return np.stack([x_loc, y_loc, z_loc], axis=-1)


def _local_to_world_vec(vecs_local, drift_dir):
    """(N,3) vector local → world frame: vx_world = -drift_dir * vx_local."""
    out = np.array(vecs_local, dtype=np.float32).copy()
    out[:, 0] = -drift_dir * vecs_local[:, 0]
    return out


# ── SIREN helpers ─────────────────────────────────────────────────────────────

def _build_siren_meta(meta):
    """Extract SIREN config from efield metadata dict, rebuilding v/E tables."""
    from tools.sce_siren import build_vinv_table
    v_table, E_table = build_vinv_table(T=89.0)
    hidden = meta.get('hidden', [32, 32, 32])
    return {
        'omega_0':       meta['omega_0'],
        'norm_offsets':  np.array(meta['norm_offsets'], dtype=np.float32),
        'norm_scales':   np.array(meta['norm_scales'],  dtype=np.float32),
        'E0':            float(meta['E0']),
        'v0':            float(meta['v0']),
        'v_table':       v_table,
        'E_table':       E_table,
        'hidden_features': int(hidden[0]),
        'hidden_layers':   int(len(hidden)),
    }


def _unflatten_siren(flat_p, meta):
    """Extract and unravel SIREN weights from the full flat parameter vector."""
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from tools.sce_siren import init_siren
    hidden = meta.get('hidden', [32, 32, 32])
    hf, hl = int(hidden[0]), int(len(hidden))
    omega_0 = float(meta['omega_0'])
    zero_siren = jax.tree.map(
        jnp.zeros_like,
        init_siren(jax.random.PRNGKey(0), hidden_features=hf, hidden_layers=hl, omega_0=omega_0),
    )
    _, unravel = ravel_pytree(zero_siren)
    n_scalar  = int(meta['n_scalar'])
    n_weights = int(meta['n_weights'])
    flat_mlp  = jnp.array(flat_p[n_scalar: n_scalar + n_weights], dtype=jnp.float32)
    return unravel(flat_mlp)


def _eval_grid_siren(params, sm, pts_local):
    """
    Evaluate SIREN on (N,3) local-frame positions.
    Returns {'delta_cm': (N,3), 'efield_Vcm': (N,3)} in local frame.
    """
    import jax.numpy as jnp
    from tools.sce_siren import recover_efield, siren_delta
    pts_j   = jnp.array(pts_local, dtype=jnp.float32)
    no      = jnp.array(sm['norm_offsets'])
    ns      = jnp.array(sm['norm_scales'])
    delta   = np.array(siren_delta(params, pts_j, no, ns, sm['omega_0']))
    efield  = np.array(recover_efield(
        params, pts_j, sm['E0'], sm['v0'],
        sm['v_table'], sm['E_table'], no, ns, sm['omega_0'],
    ))
    return {'delta_cm': delta, 'efield_Vcm': efield}


def _process_side_siren(params, sm, origin_cm, spacing_cm, shape,
                        x_anode_cm, drift_dir, yz_center_cm=(0.0, 0.0)):
    """Evaluate SIREN on a 3-D grid side. Returns world-frame arrays."""
    Nx, Ny, Nz = shape
    xs = origin_cm[0] + np.arange(Nx) * spacing_cm[0]
    ys = origin_cm[1] + np.arange(Ny) * spacing_cm[1]
    zs = origin_cm[2] + np.arange(Nz) * spacing_cm[2]
    XW, YW, ZW = np.meshgrid(xs, ys, zs, indexing='ij')
    pts_world = np.stack([XW.ravel(), YW.ravel(), ZW.ravel()], axis=-1)
    pts_local = _world_to_local_pos(pts_world, x_anode_cm, drift_dir, yz_center_cm)

    local_out = _eval_grid_siren(params, sm, pts_local)

    delta_world  = _local_to_world_vec(local_out['delta_cm'],   drift_dir)
    efield_world = _local_to_world_vec(local_out['efield_Vcm'], drift_dir)
    return {
        'efield_Vcm':     efield_world.reshape(Nx, Ny, Nz, 3),
        'corrections_cm': delta_world.reshape(Nx, Ny, Nz, 3),
    }


# ── Legacy FieldConfig helpers ────────────────────────────────────────────────

def _build_legacy_cfg(meta):
    from tools.nonlocal_efield import FieldConfig
    return FieldConfig(
        mode=meta['mode'],
        hidden=tuple(meta['hidden']),
        center_cm=tuple(meta['center_cm']),
        half_cm=tuple(meta['half_cm']),
        bg_field_Vcm=tuple(meta['bg_field_Vcm']),
        out_scale=meta['out_scale'],
    )


def _unflatten_legacy(flat_p, meta):
    from tools.nonlocal_efield import zero_params, flatten_params
    import jax.numpy as jnp
    cfg = _build_legacy_cfg(meta)
    n_scalar  = meta['n_scalar']
    n_weights = meta['n_weights']
    flat_mlp  = jnp.array(flat_p[n_scalar: n_scalar + n_weights], dtype=jnp.float32)
    _, unravel = flatten_params(zero_params(cfg))
    return unravel(flat_mlp), cfg


def _eval_grid_legacy(params, cfg, pts_local):
    import jax
    import jax.numpy as jnp
    import tools.nonlocal_efield as _nl
    pts_j  = jnp.array(pts_local, dtype=jnp.float32)
    field_vals = np.array(_nl.make_field_fn(params, cfg)(pts_j))
    out = {}
    if cfg.mode == 'potential':
        pot_fn = jax.jit(jax.vmap(lambda p: _nl.potential(params, p, cfg)))
        out['potential_Vcm_cm'] = np.array(pot_fn(pts_j))
        out['efield_Vcm'] = field_vals
    elif cfg.mode == 'efield':
        out['efield_Vcm'] = field_vals
    else:
        out['corrections_cm'] = field_vals
    return out


def _process_side_legacy(params, cfg, origin_cm, spacing_cm, shape,
                         x_anode_cm, drift_dir, yz_center_cm=(0.0, 0.0)):
    Nx, Ny, Nz = shape
    xs = origin_cm[0] + np.arange(Nx) * spacing_cm[0]
    ys = origin_cm[1] + np.arange(Ny) * spacing_cm[1]
    zs = origin_cm[2] + np.arange(Nz) * spacing_cm[2]
    XW, YW, ZW = np.meshgrid(xs, ys, zs, indexing='ij')
    pts_world = np.stack([XW.ravel(), YW.ravel(), ZW.ravel()], axis=-1)
    pts_local = _world_to_local_pos(pts_world, x_anode_cm, drift_dir, yz_center_cm)
    local_out = _eval_grid_legacy(params, cfg, pts_local)
    world_out = {}
    if 'efield_Vcm' in local_out:
        world_out['efield_Vcm'] = _local_to_world_vec(
            local_out['efield_Vcm'], drift_dir).reshape(Nx, Ny, Nz, 3)
    if 'corrections_cm' in local_out:
        world_out['corrections_cm'] = _local_to_world_vec(
            local_out['corrections_cm'], drift_dir).reshape(Nx, Ny, Nz, 3)
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

    mode = meta.get('mode', 'siren')
    is_siren = (mode == 'siren')

    # ── Build snapshots list from completed trials or live_checkpoint ─────────
    # Each entry: (label, step, flat_p)
    snapshot_triples = []

    trials = result.get('trials', [])
    if trials:
        for trial_idx, trial in enumerate(trials):
            mlp_traj = trial.get('mlp_trajectory')
            final_p  = trial.get('final_p')
            steps_run = trial.get('steps_run', '?')
            if mlp_traj and not last_only:
                for step, flat_p in mlp_traj:
                    snapshot_triples.append((f'trial{trial_idx}_step{step}', step, flat_p))
            elif final_p is not None:
                snapshot_triples.append((f'trial{trial_idx}_step{steps_run}', steps_run, final_p))
            else:
                print(f'  [warn] trial {trial_idx}: no weights, skipping')
    else:
        lc = result.get('live_checkpoint')
        if lc is not None and lc.get('p') is not None:
            step = lc.get('step', '?')
            print(f'  no completed trials; using live_checkpoint at step {step}')
            snapshot_triples.append((f'live_step{step}', step, lc['p']))
        else:
            print(f'  [skip] {pkl_path.name}: no trials and no live_checkpoint')
            return

    # ── Load GT NPZ ──────────────────────────────────────────────────────────
    gt_path = meta.get('gt_map_path')
    gt_data = None
    if gt_path and os.path.exists(gt_path):
        d = np.load(gt_path)
        gt_data = {
            'east': {'efield_Vcm': d['east_efield'],     'corrections_cm': d['east_corrections']},
            'west': {'efield_Vcm': d['west_efield'],     'corrections_cm': d['west_corrections']},
        }
        grid = {
            'east': {'origin_cm': d['east_origin'], 'spacing_cm': d['east_spacing'],
                     'shape': tuple(d['east_efield'].shape[:3])},
            'west': {'origin_cm': d['west_origin'], 'spacing_cm': d['west_spacing'],
                     'shape': tuple(d['west_efield'].shape[:3])},
        }
        print(f'  GT loaded from {gt_path}')
    else:
        print(f'  [warn] GT NPZ not found at {gt_path!r}; using a default 21³ grid')
        if is_siren:
            sm = _build_siren_meta(meta)
            hx, hy, hz = sm['norm_scales'].tolist()
        else:
            cfg_tmp = _build_legacy_cfg(meta)
            hx, hy, hz = cfg_tmp.half_cm
        N = 21
        sp = np.array([hx * 2 / (N - 1), hy * 2 / (N - 1), hz * 2 / (N - 1)])
        grid = {
            'east': {'origin_cm': np.array([-hx * 2, -hy, -hz]), 'spacing_cm': sp, 'shape': (N, N, N)},
            'west': {'origin_cm': np.array([0.0,      -hy, -hz]), 'spacing_cm': sp, 'shape': (N, N, N)},
        }

    # ── Volume geometry for world ↔ local transforms ──────────────────────────
    vol_geom = {
        'east': {
            'x_anode_cm':   float(grid['east']['origin_cm'][0]),
            'drift_dir':    -1,
            'yz_center_cm': (0.0, 0.0),
        },
        'west': {
            'x_anode_cm': float(
                grid['west']['origin_cm'][0]
                + (grid['west']['shape'][0] - 1) * grid['west']['spacing_cm'][0]
            ),
            'drift_dir':    +1,
            'yz_center_cm': (0.0, 0.0),
        },
    }

    # ── Build SIREN meta once ─────────────────────────────────────────────────
    if is_siren:
        sm = _build_siren_meta(meta)

    # ── Evaluate each snapshot ────────────────────────────────────────────────
    step_snapshots = []
    for label, step, flat_p in snapshot_triples:
        print(f'  evaluating {label}  mode={mode}')

        if is_siren:
            params = _unflatten_siren(flat_p, meta)
        else:
            params, cfg = _unflatten_legacy(flat_p, meta)

        learned = {}
        for side, geom in vol_geom.items():
            g = grid[side]
            if is_siren:
                learned[side] = _process_side_siren(
                    params, sm,
                    origin_cm=g['origin_cm'], spacing_cm=g['spacing_cm'], shape=g['shape'],
                    x_anode_cm=geom['x_anode_cm'], drift_dir=geom['drift_dir'],
                    yz_center_cm=geom['yz_center_cm'],
                )
            else:
                learned[side] = _process_side_legacy(
                    params, cfg,
                    origin_cm=g['origin_cm'], spacing_cm=g['spacing_cm'], shape=g['shape'],
                    x_anode_cm=geom['x_anode_cm'], drift_dir=geom['drift_dir'],
                    yz_center_cm=geom['yz_center_cm'],
                )

        step_snapshots.append({'label': label, 'step': step, 'learned': learned})

    if not step_snapshots:
        print(f'  [skip] {pkl_path.name}: nothing to save')
        return

    output = {
        'source_pkl':  str(pkl_path),
        'efield_meta': meta,
        'grid':        grid,
        'vol_geom':    vol_geom,
        'gt':          gt_data,
        'steps':       step_snapshots,
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

    pkls = []
    for p in args.result_pkls:
        if os.path.isdir(p):
            pkls += sorted(glob.glob(os.path.join(p, '**', 'result_*.pkl'), recursive=True))
        else:
            pkls.append(p)
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
