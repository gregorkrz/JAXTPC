#!/usr/bin/env python3
"""
Evaluate wireplane signals (GT vs learned SIREN E-field MLP) for selected tracks
of an E-field calibration run, at every saved MLP weight snapshot.

For each result PKL and each requested --track-idx, re-simulates that track
through:
  - the GT simulator (static electric-distortion NPZ), once, and
  - the differentiable simulator with the learned SIREN MLP, once per snapshot
    step in `mlp_trajectory` (or the single live_checkpoint/final_p weight set
    if no trajectory was saved),
and saves the raw per-(volume, plane) wireplane signal arrays — one PKL per
track, written next to the source result PKL. A later script will turn these
into an interactive HTML viewer (same two-step pattern as eval_efield_mlp.py /
plot_efield_eval.py); this script only produces the data.

When the run used --noise-scale > 0 (true for every submit_jobs_E_field_calibration.py
profile), the saved GT includes the exact same noise draw run_optimization.py's loss was
computed against (same seeding: SeedSequence(result['seed']) -> noise_seed -> fold_in(track_idx)),
not the clean signal — so "learned vs GT" here matches what was actually optimized.

This needs a real GPU (each snapshot re-runs the full detector-physics forward
pass) — use --submit to launch it as a Slurm job on a single A100 instead of
running locally.

Usage
-----
  python tools/eval_efield_track_wireplanes.py RESULT_PKL [RESULT_PKL ...]
  python tools/eval_efield_track_wireplanes.py --results-dir $RESULTS_DIR/opt/efield_calib/1k_tracks_sweep
  python tools/eval_efield_track_wireplanes.py --results-dir ... --track-idx 0,1,2
  python tools/eval_efield_track_wireplanes.py --results-dir ... --submit --time 02:00:00
"""
import argparse
import glob
import os
import pickle
import shlex
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT))

from tools.eval_efield_mlp import _unflatten_siren  # reuse SIREN weight reconstruction

REMOTE_CODE_DIR = '/sdf/home/g/gregork/jaxtpc'


# ── argv-command parsing (result['command'] is the source of truth for the
#    exact sim hyperparameters a run used — mirrors make_efield_calib_command's
#    fixed defaults as a fallback when a flag wasn't found) ────────────────────

def _parse_argv_flag(command, flag, default, cast=float):
    if not command:
        return default
    try:
        toks = shlex.split(command)
    except ValueError:
        return default
    if flag not in toks:
        return default
    i = toks.index(flag)
    if i + 1 >= len(toks):
        return default
    try:
        return cast(toks[i + 1])
    except ValueError:
        return default


# ── Snapshot + label helpers ───────────────────────────────────────────────────

def _snapshot_triples(result):
    """[(label, step, flat_p), ...]; mirrors eval_efield_mlp.process_pkl's fallback."""
    triples = []
    trials = result.get('trials', [])
    if trials:
        for trial_idx, trial in enumerate(trials):
            mlp_traj = trial.get('mlp_trajectory')
            final_p = trial.get('final_p')
            steps_run = trial.get('steps_run', '?')
            if mlp_traj:
                for step, flat_p in mlp_traj:
                    triples.append((f'trial{trial_idx}_step{step}', step, flat_p))
            elif final_p is not None:
                triples.append((f'trial{trial_idx}_step{steps_run}', steps_run, final_p))
    else:
        lc = result.get('live_checkpoint')
        if lc is not None and lc.get('p') is not None:
            step = lc.get('step', '?')
            triples.append((f'live_step{step}', step, lc['p']))
    return triples


def _output_signal_labels(cfg):
    """[(vol_idx, plane_idx, plane_name), ...] matching simulator.forward()'s output order."""
    labels = []
    for v in range(cfg.n_volumes):
        plane_names = cfg.plane_names[v]
        for p in range(cfg.volumes[v].n_planes):
            labels.append((v, p, plane_names[p]))
    return labels


# ── Noise (mirrors run_optimization.py's seeding + apply_noise_to_gt exactly, so the
#    "GT" shown here matches the actual noisy target the loss was computed against) ──

def _noise_seed_from_result_seed(seed):
    """Reproduces run_optimization.py's `noise_seed` derivation (run_optimization.py:1041-1045):
    ss = SeedSequence(args.seed); _, noise_ss = ss.spawn(2); noise_seed = noise_ss.generate_state(1)[0].
    Only exact when the run passed an explicit --seed (true for every submit_jobs_E_field_calibration.py
    profile), since then result['seed'] == args.seed."""
    ss = np.random.SeedSequence(seed)
    _, noise_ss = ss.spawn(2)
    return int(noise_ss.generate_state(1)[0])


def _apply_noise_to_gt(gt_arrays, simulator, noise_scale, noise_key):
    """Verbatim copy of run_optimization.py's apply_noise_to_gt (not imported — that module
    is a CLI driver, not meant to be imported as a library)."""
    import jax.numpy as jnp
    from tools.noise import generate_noise
    cfg = simulator.config
    noise_dict = generate_noise(cfg, key=noise_key)
    n_readouts = cfg.volumes[0].n_planes if simulator._readout_type == 'wire' else 1
    noisy = []
    for v in range(cfg.n_volumes):
        for p in range(n_readouts):
            gt = gt_arrays[v * n_readouts + p]
            noise = noise_dict[(v, p)] * noise_scale
            if noise.shape[0] < gt.shape[0]:
                noise = jnp.pad(noise, ((0, gt.shape[0] - noise.shape[0]), (0, 0)))
            noisy.append(gt + noise)
    return tuple(noisy)


# ── Simulator construction (mirrors run_optimization.py's _get_sim, gt/diff roles) ─

def build_simulators(meta, command):
    """Build (gt_sim, gt_params, diff_sim) matching the training run's configuration."""
    import jax.numpy as jnp
    from tools.geometry import generate_detector
    from tools.simulation import DetectorSimulator
    from tools.distortion import SirenDistortionConfig
    from tools.sce_siren import build_vinv_table
    from optlib.constants import CONFIG_PATH, GT_LIFETIME_US, GT_VELOCITY_CM_US

    gt_max_deposits  = int(_parse_argv_flag(command, '--gt-max-deposits', 5000))
    max_num_deposits = int(_parse_argv_flag(command, '--max-num-deposits', 5000))
    num_buckets      = int(_parse_argv_flag(command, '--num-buckets', 1000))

    detector_config = generate_detector(CONFIG_PATH)

    print(f'  Building GT simulator (n_segments={gt_max_deposits:,})...')
    gt_sim = DetectorSimulator(
        detector_config, differentiable=True, n_segments=gt_max_deposits,
        use_bucketed=True, max_active_buckets=num_buckets,
        include_noise=False, include_electronics=False,
        include_track_hits=False, include_digitize=False,
        include_electric_dist=True, electric_dist_path=meta['gt_map_path'],
    )
    gt_sim.warm_up()
    gt_params = gt_sim.default_sim_params._replace(
        lifetime_us=jnp.array(GT_LIFETIME_US), velocity_cm_us=jnp.array(GT_VELOCITY_CM_US))

    v_table, E_table = build_vinv_table(T=89.0)
    hidden = meta.get('hidden', [32, 32, 32])
    siren_cfg = SirenDistortionConfig(
        omega_0=float(meta['omega_0']),
        norm_offsets=jnp.array(meta['norm_offsets'], dtype=jnp.float32),
        norm_scales=jnp.array(meta['norm_scales'], dtype=jnp.float32),
        E0=float(meta['E0']), v0=float(meta['v0']),
        v_table=v_table, E_table=E_table,
        hidden_features=int(hidden[0]), hidden_layers=len(hidden),
    )
    print(f'  Building diff simulator (n_segments={max_num_deposits:,})...')
    diff_sim = DetectorSimulator(
        detector_config, differentiable=True, n_segments=max_num_deposits,
        use_bucketed=True, max_active_buckets=num_buckets,
        include_noise=False, include_electronics=False,
        include_track_hits=False, include_digitize=False,
        efield_model=siren_cfg, efield_per_volume=bool(meta.get('per_volume', False)),
    )
    # warm_up() compiles using self._default_sim_params — seed sce_models with a
    # correctly-shaped zero-init pytree first (mirrors run_optimization.py's _get_sim);
    # otherwise sce_models is None and the SIREN sce_factory crashes on {**None, ...}.
    from tools.sce_siren import init_siren
    import jax
    per_volume = bool(meta.get('per_volume', False))
    _single_zero = jax.tree.map(
        jnp.zeros_like,
        init_siren(jax.random.PRNGKey(0), hidden_features=siren_cfg.hidden_features,
                   hidden_layers=siren_cfg.hidden_layers, omega_0=siren_cfg.omega_0))
    efield_zero_params = (
        jax.tree.map(lambda a, b: jnp.stack([a, b], axis=0), _single_zero, _single_zero)
        if per_volume else _single_zero
    )
    diff_sim._default_sim_params = diff_sim._default_sim_params._replace(
        sce_models=efield_zero_params)
    diff_sim.warm_up()

    gt_step_size   = _parse_argv_flag(command, '--gt-step-size', 1.0)
    diff_step_size = _parse_argv_flag(command, '--step-size', 1.0)
    return gt_sim, gt_params, diff_sim, gt_step_size, diff_step_size


# ── Per-track evaluation ───────────────────────────────────────────────────────

def eval_track(ts, gt_sim, gt_params, jitted_gt_forward, gt_step_size,
                diff_sim, jitted_diff_forward, diff_step_size,
                snapshot_triples, meta, noise_scale=0.0, noise_key=None):
    from tools.particle_generator import generate_muon_track
    from tools.loader import build_deposit_data

    def _deposits(sim, step_size):
        track = generate_muon_track(
            start_position_mm=ts['start_position_mm'], direction=ts['direction'],
            kinetic_energy_mev=ts['momentum_mev'], step_size_mm=step_size, track_id=1)
        return build_deposit_data(
            track['position'], track['de'], track['dx'], sim.config,
            theta=track['theta'], phi=track['phi'], track_ids=track['track_id'])

    gt_deposits = _deposits(gt_sim, gt_step_size)
    gt_out = tuple(jitted_gt_forward(gt_params, gt_deposits))
    if noise_scale > 0.0:
        # Matches training's target exactly: run_optimization.py always compares the
        # learned forward pass against the NOISY GT, not the clean signal.
        gt_out = _apply_noise_to_gt(gt_out, gt_sim, noise_scale, noise_key)
    gt_out = tuple(np.array(a) for a in gt_out)

    diff_deposits = _deposits(diff_sim, diff_step_size)
    steps_out = []
    for label, step, flat_p in snapshot_triples:
        mlp_params = _unflatten_siren(np.asarray(flat_p), meta)
        diff_params = gt_params._replace(sce_models=mlp_params)
        learned_out = tuple(np.array(a) for a in jitted_diff_forward(diff_params, diff_deposits))
        steps_out.append({'label': label, 'step': step, 'learned': learned_out})

    return gt_out, steps_out


# ── Per-PKL processing ────────────────────────────────────────────────────────

def process_pkl(pkl_path, track_indices, overwrite=False, last_only=False):
    pkl_path = Path(pkl_path)
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)

    meta = result.get('efield')
    if not meta or not meta.get('present') or meta.get('mode', 'siren') != 'siren':
        print(f'  [skip] {pkl_path.name}: no SIREN efield metadata')
        return

    tracks = result.get('tracks') or []
    triples = _snapshot_triples(result)
    if last_only and triples:
        triples = [triples[-1]]
    if not triples:
        print(f'  [skip] {pkl_path.name}: no snapshots (no trials/live_checkpoint)')
        return

    todo = []  # [(track_idx, out_path), ...]
    for idx in track_indices:
        if idx >= len(tracks):
            print(f'  [warn] {pkl_path.name}: track-idx {idx} out of range '
                  f'(only {len(tracks)} tracks); skipping')
            continue
        out_path = pkl_path.parent / f'{pkl_path.stem}_track{idx}_wireplanes.pkl'
        if out_path.exists() and not overwrite and out_path.stat().st_mtime >= pkl_path.stat().st_mtime:
            print(f'  [skip] {out_path.name} already exists (up to date)')
            continue
        todo.append((idx, out_path))

    if not todo:
        return

    command = result.get('command')
    print(f'  {pkl_path.name}: {len(triples)} snapshot(s), {len(todo)} track(s) to evaluate')
    t0 = time.time()
    gt_sim, gt_params, diff_sim, gt_step_size, diff_step_size = build_simulators(meta, command)
    print(f'  simulators ready ({time.time() - t0:.1f}s)')

    import jax
    jitted_gt_forward = jax.jit(gt_sim.forward)
    jitted_diff_forward = jax.jit(diff_sim.forward)
    labels = _output_signal_labels(gt_sim.config)

    noise_scale = float(result.get('noise_scale', 0.0))
    noise_base_key = None
    if noise_scale > 0.0:
        noise_seed = _noise_seed_from_result_seed(result.get('seed'))
        noise_base_key = jax.random.PRNGKey(noise_seed)
        print(f'  noise_scale={noise_scale:g} (noise_seed={noise_seed}) — GT will include '
              f'the same noise draw training compared against')

    for idx, out_path in todo:
        ts = tracks[idx]
        print(f'  [{pkl_path.name}] track {idx} ({ts["name"]})...')
        t1 = time.time()
        noise_key = jax.random.fold_in(noise_base_key, idx) if noise_base_key is not None else None
        gt_out, steps_out = eval_track(
            ts, gt_sim, gt_params, jitted_gt_forward, gt_step_size,
            diff_sim, jitted_diff_forward, diff_step_size, triples, meta,
            noise_scale=noise_scale, noise_key=noise_key)
        output = dict(
            source_pkl=str(pkl_path),
            wandb_run_id=result.get('wandb_run_id'),
            track_idx=idx,
            track_spec=ts,
            efield_meta=meta,
            labels=labels,        # [(vol_idx, plane_idx, plane_name), ...], order matches gt/learned
            gt=gt_out,            # tuple of (num_wires, num_time) arrays, one per label
            steps=steps_out,      # [{'label', 'step', 'learned': tuple of arrays}, ...]
        )
        with open(out_path, 'wb') as f:
            pickle.dump(output, f)
        print(f'    saved → {out_path}  ({time.time() - t1:.1f}s, '
              f'{len(steps_out)} steps x {len(labels)} planes)')


# ── Self-submission (this script needs a real GPU) ────────────────────────────

def _submit_self(args):
    sys.path.insert(0, str(_ROOT / 'src' / 'jobs'))
    from job_submission_tools import s3df_submit

    passthrough = []
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a in ('--submit', '--print-commands'):
            continue
        if a in ('--time', '--mem-gb'):
            skip_next = True
            continue
        passthrough.append(a)

    cmd = (f'python {REMOTE_CODE_DIR}/tools/eval_efield_track_wireplanes.py '
           + ' '.join(shlex.quote(a) for a in passthrough))
    print(cmd)
    s3df_submit(
        cmd, time=args.time, gpus=1, mem_gb=args.mem_gb,
        submit=args.submit, print_sbatch_command=args.print_commands,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('result_pkls', nargs='*',
                        help='Result PKL paths (or directories) to process directly')
    parser.add_argument('--results-dir', default=None,
                        help='Scan this directory recursively for result_*.pkl files')
    parser.add_argument('--track-idx', default='0,1,2,3,4,5',
                        help='Comma-separated indices into result["tracks"] '
                             '(default: 0,1,2,3,4,5)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-run even if an output PKL already exists')
    parser.add_argument('--last-only', action='store_true',
                        help='Only evaluate the final snapshot (fast smoke test)')
    parser.add_argument('--max-runs', type=int, default=None, metavar='N',
                        help='Only process the first N result PKLs found (sorted), instead '
                             'of every run under --results-dir. Useful to cap how much data '
                             'a sweep-wide --results-dir generates.')
    parser.add_argument('--submit', action='store_true',
                        help='Submit as a Slurm job on 1 A100 GPU instead of running locally '
                             '(this script needs a real GPU).')
    parser.add_argument('--print-commands', action='store_true',
                        help='Print only the sbatch command (implies not running locally).')
    parser.add_argument('--time', default='02:00:00',
                        help='Slurm wall-clock time when --submit/--print-commands (default: 02:00:00).')
    parser.add_argument('--mem-gb', type=int, default=64,
                        help='Slurm memory in GB when --submit/--print-commands (default: 64).')
    args = parser.parse_args()

    if args.submit or args.print_commands:
        _submit_self(args)
        return

    track_indices = [int(s) for s in args.track_idx.split(',') if s.strip() != '']

    pkls = []
    for p in args.result_pkls:
        if os.path.isdir(p):
            pkls += sorted(glob.glob(os.path.join(p, '**', 'result_*.pkl'), recursive=True))
        else:
            pkls.append(p)
    if args.results_dir:
        pkls += sorted(glob.glob(
            os.path.join(args.results_dir, '**', 'result_*.pkl'), recursive=True))
    pkls = sorted(set(pkls))
    pkls = [p for p in pkls if not p.endswith('_efield_eval.pkl') and '_wireplanes' not in Path(p).stem]

    if not pkls:
        print('No PKLs found.')
        return

    if args.max_runs is not None:
        print(f'Found {len(pkls)} run(s); limiting to the first {args.max_runs} (--max-runs).')
        pkls = pkls[:args.max_runs]

    for pkl in pkls:
        print(f'Processing {pkl}')
        try:
            process_pkl(pkl, track_indices, overwrite=args.overwrite, last_only=args.last_only)
        except Exception as exc:
            import traceback
            print(f'  ERROR: {exc}')
            traceback.print_exc()


if __name__ == '__main__':
    main()
