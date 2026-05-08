#!/usr/bin/env python
"""
Launch 2D loss landscapes for all pairs of SimParams scalars × multiple tracks.

Partitions work across N Slurm array tasks (or local --job-id). Each task runs
`2d_loss_landscape.py` as a subprocess with --no-plots and a fixed --output-pkl.

Default output layout:
  {RESULTS_DIR}/landscape/{run_date}/{track_name}/{param_y}__{param_x}.pkl

Default tracks (no ``--tracks-yaml``): same ensemble as ``plot_mixed_tracks_edep_wireplanes``
— ``N`` random boundary muons (default 12) plus one fixed 1000 MeV diagonal cross
track (``tools/random_boundary_tracks.py``), with
``--track-seed`` / ``--n-boundary-tracks`` matching that script. Each subprocess gets
``--start-position-mm`` where applicable.

Examples
--------
  # Dry-run: print commands for job 0 of 4
  python src/analysis/launch_2d_landscape_pairs.py --n-jobs 4 --dry-run

  # Run locally (all tasks on one process)
  python src/analysis/launch_2d_landscape_pairs.py --n-jobs 1 --job-id 0 \\
      --grid 20 --range-frac 0.03 --loss sobolev_loss_geomean_log1p --gradients

  # Slurm array worker (#SBATCH --array=0-(N-1); launcher reads SLURM_ARRAY_TASK_ID)
  python src/analysis/launch_2d_landscape_pairs.py --n-jobs 8 --grid 20 \\
      --loss sobolev_loss_geomean_log1p --gradients

  # Emit a minimal sbatch stub
  python src/analysis/launch_2d_landscape_pairs.py --emit-sbatch --n-jobs 8 \\
      --sbatch-partition gpu --sbatch-time 04:00:00 --grid 20 \\
      --loss sobolev_loss_geomean_log1p --gradients
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

# ── Repo root & VALID_PARAMS (same source as 2d_loss_landscape / 2d_opt) ─────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
_2d_opt_path = _REPO_ROOT / 'src' / 'opt' / '2d_opt.py'
_spec = importlib.util.spec_from_file_location('jaxtpc_2d_opt', str(_2d_opt_path))
_2d_opt = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_2d_opt)
VALID_PARAMS: tuple[str, ...] = _2d_opt.VALID_PARAMS

from tools.geometry import generate_detector
from tools.random_boundary_tracks import (
    N_DEFAULT_BOUNDARY_MUONS,
    generate_random_boundary_tracks,
)
from tools.simulation import DetectorSimulator

_RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')

_CONFIG_PATH = 'config/cubic_wireplane_config.yaml'
_N_SEGMENTS = 50_000
_MAX_ACTIVE_BUCKETS = 1000


def _simulator_for_volume_geometry() -> DetectorSimulator:
    detector_config = generate_detector(_CONFIG_PATH)
    return DetectorSimulator(
        detector_config,
        differentiable=True,
        n_segments=_N_SEGMENTS,
        use_bucketed=True,
        max_active_buckets=_MAX_ACTIVE_BUCKETS,
        include_noise=False,
        include_electronics=False,
        include_track_hits=False,
        include_digitize=False,
        track_config=None,
    )


def _default_boundary_track_dicts(
    *,
    seed: int,
    n: int,
) -> list[dict[str, Any]]:
    """Same muon ensemble as ``plot_mixed_tracks_edep_wireplanes`` default (boundary starts)."""
    sim = _simulator_for_volume_geometry()
    raw = generate_random_boundary_tracks(sim.config.volumes, n=n, seed=seed)
    out: list[dict[str, Any]] = []
    for spec in raw:
        dx, dy, dz = spec['direction']
        out.append({
            'name': spec['name'],
            'direction': f'{dx},{dy},{dz}',
            'momentum_mev': float(spec['momentum_mev']),
            'start_position_mm': spec['start_position_mm'],
        })
    return out


# ── Legacy fixed triple (optional manual testing); normal CLI default uses
# ``_default_boundary_track_dicts`` (same RNG as ``plot_mixed_tracks_edep_wireplanes``).


def _venv_python() -> str:
    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        return str(Path(venv) / 'bin' / 'python')
    cand = _REPO_ROOT / '.venv' / 'bin' / 'python'
    if cand.is_file():
        return str(cand)
    return sys.executable


def _resolve_job_id(cli_job_id: int | None) -> int:
    if cli_job_id is not None:
        return cli_job_id
    s = os.environ.get('SLURM_ARRAY_TASK_ID')
    if s is not None:
        return int(s)
    return 0


def _load_tracks_yaml(path: Path) -> list[dict[str, Any]]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise SystemExit('Install PyYAML to use --tracks-yaml') from e
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not data or 'tracks' not in data:
        raise SystemExit(f'{path}: expected top-level "tracks" list')
    tracks_raw = data['tracks']
    out: list[dict[str, Any]] = []
    for i, t in enumerate(tracks_raw):
        if not isinstance(t, dict):
            raise SystemExit(f'{path}: tracks[{i}] must be a mapping')
        name = t.get('name')
        if not name:
            raise SystemExit(f'{path}: tracks[{i}] missing "name"')
        direction = t.get('direction')
        if isinstance(direction, (list, tuple)):
            direction = ','.join(str(x) for x in direction)
        if not direction:
            raise SystemExit(f'{path}: tracks[{i}] missing "direction"')
        mom = t.get('momentum_mev', t.get('momentum'))
        if mom is None:
            raise SystemExit(f'{path}: tracks[{i}] missing "momentum_mev"')
        rec: dict[str, Any] = {
            'name': str(name),
            'direction': str(direction),
            'momentum_mev': float(mom),
        }
        sp = t.get('start_position_mm')
        if sp is not None:
            if not isinstance(sp, (list, tuple)) or len(sp) != 3:
                raise SystemExit(
                    f'{path}: tracks[{i}] "start_position_mm" must be [x,y,z] mm')
            rec['start_position_mm'] = tuple(float(x) for x in sp)
        out.append(rec)
    return out


def _load_params_yaml(path: Path) -> list[str] | None:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise SystemExit('Install PyYAML to use --tracks-yaml') from e
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not data:
        return None
    params = data.get('params')
    if params is None:
        return None
    if not isinstance(params, list) or not all(isinstance(p, str) for p in params):
        raise SystemExit(f'{path}: "params" must be a list of strings')
    return list(params)


def _param_pairs(params: Sequence[str]) -> list[tuple[str, str]]:
    s = sorted(set(params))
    for p in s:
        if p not in VALID_PARAMS:
            raise SystemExit(f'Unknown param {p!r}; valid: {VALID_PARAMS!r}')
    return list(combinations(s, 2))


def _iter_tasks(
    tracks: Iterable[dict[str, Any]],
    pairs: Sequence[tuple[str, str]],
) -> list[tuple[dict[str, Any], str, str]]:
    return [(tr, py, px) for tr in tracks for py, px in pairs]


def _filter_tasks_for_job(
    tasks: Sequence[tuple[dict[str, Any], str, str]],
    job_id: int,
    n_jobs: int,
) -> list[tuple[dict[str, Any], str, str]]:
    return [t for k, t in enumerate(tasks) if k % n_jobs == job_id]


def _build_landscape_command(
    *,
    python_exe: str,
    param_y: str,
    param_x: str,
    track: dict[str, Any],
    results_dir_for_track: Path,
    output_pkl: Path,
    grid: int,
    range_frac: float,
    loss: str,
    gradients: bool,
    noise_scale: float,
    noise_seed: int,
) -> list[str]:
    cmd = [
        python_exe,
        str(_REPO_ROOT / 'src' / 'analysis' / '2d_loss_landscape.py'),
        '--param-y', param_y,
        '--param-x', param_x,
        '--track-name', str(track['name']),
        '--direction', str(track['direction']),
        '--momentum', str(track['momentum_mev']),
        '--grid', str(grid),
        '--range-frac', str(range_frac),
        '--loss', loss,
        '--results-dir', str(results_dir_for_track),
        '--output-pkl', str(output_pkl),
        '--noise-scale', str(noise_scale),
        '--noise-seed', str(noise_seed),
        '--no-plots',
    ]
    if gradients:
        cmd.append('--gradients')
    smm = track.get('start_position_mm')
    if smm is not None:
        cmd.extend([
            '--start-position-mm', str(smm[0]), str(smm[1]), str(smm[2]),
        ])
    return cmd


def _launcher_cli_tokens(args: argparse.Namespace) -> list[str]:
    """Flags for `launch_2d_landscape_pairs.py` (same worker, omit --job-id for Slurm)."""
    t = [
        '--n-jobs', str(args.n_jobs),
        '--grid', str(args.grid),
        '--range-frac', str(args.range_frac),
        '--loss', args.loss,
        '--noise-scale', str(args.noise_scale),
        '--noise-seed', str(args.noise_seed),
    ]
    if args.run_date is not None:
        t += ['--run-date', args.run_date]
    if args.tracks_yaml is not None:
        t += ['--tracks-yaml', str(args.tracks_yaml)]
    else:
        t += ['--track-seed', str(args.track_seed),
              '--n-boundary-tracks', str(args.n_boundary_tracks)]
    if args.params is not None:
        t.append('--params')
        t.extend(args.params)
    if args.gradients:
        t.append('--gradients')
    return t


def _emit_sbatch(args: argparse.Namespace) -> None:
    n = args.n_jobs
    if n < 1:
        raise SystemExit('--n-jobs must be >= 1')
    py = _venv_python()
    launch_script = _REPO_ROOT / 'src' / 'analysis' / 'launch_2d_landscape_pairs.py'
    inner = [py, str(launch_script)] + _launcher_cli_tokens(args)
    inner_line = shlex.join(inner)
    lines = [
        '#!/bin/bash',
        '#SBATCH --job-name=landscape_pairs',
        f'#SBATCH --array=0-{n - 1}',
    ]
    if args.sbatch_partition:
        lines.append(f'#SBATCH --partition={args.sbatch_partition}')
    if args.sbatch_time:
        lines.append(f'#SBATCH --time={args.sbatch_time}')
    if args.sbatch_account:
        lines.append(f'#SBATCH --account={args.sbatch_account}')
    if args.sbatch_gpus is not None:
        lines.append(f'#SBATCH --gpus={args.sbatch_gpus}')
    if args.sbatch_cpus is not None:
        lines.append(f'#SBATCH --cpus-per-task={args.sbatch_cpus}')
    if args.sbatch_mem:
        lines.append(f'#SBATCH --mem={args.sbatch_mem}')
    lines += [
        'set -euo pipefail',
        f'cd {shlex.quote(str(_REPO_ROOT))}',
        f'export RESULTS_DIR={shlex.quote(os.environ.get("RESULTS_DIR", "results"))}',
        'echo "[task ${SLURM_ARRAY_TASK_ID:-0}] starting"',
        inner_line,
    ]
    print('\n'.join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--n-jobs', type=int, default=1,
                   help='Number of Slurm array tasks / parallel workers (default: 1)')
    p.add_argument('--job-id', type=int, default=None,
                   help='This worker index in [0, n_jobs). Default: SLURM_ARRAY_TASK_ID or 0')
    p.add_argument('--run-date', default=None,
                   help='YYYYMMDD subdirectory under results/landscape (default: today UTC date)')
    p.add_argument('--tracks-yaml', type=Path, default=None,
                   help='YAML with "tracks" list (optional "params" override)')
    p.add_argument('--params', nargs='*', default=None,
                   help='Subset of param names (default: all VALID_PARAMS). Implies sorted pairs.')
    p.add_argument('--grid', type=int, default=10)
    p.add_argument('--range-frac', type=float, default=0.15)
    p.add_argument('--loss', default='sobolev_loss_geomean_log1p',
                   help='Single loss name (default: sobolev_loss_geomean_log1p)')
    p.add_argument('--gradients', action='store_true')
    p.add_argument('--noise-scale', type=float, default=0.0)
    p.add_argument('--noise-seed', type=int, default=0)
    p.add_argument(
        '--track-seed', type=int, default=42,
        help='RNG seed for default random boundary tracks (same as event-display script; '
             'ignored with --tracks-yaml)',
    )
    p.add_argument(
        '--n-boundary-tracks', type=int, default=N_DEFAULT_BOUNDARY_MUONS,
        metavar='N',
        help=(
            f'How many random x-face boundary tracks before the fixed diagonal cross '
            f'(default: {N_DEFAULT_BOUNDARY_MUONS}; total default tracks = this + 1)'
        ),
    )
    p.add_argument('--dry-run', action='store_true', help='Print commands; do not run')
    p.add_argument('--emit-sbatch', action='store_true',
                   help='Print a minimal sbatch script to stdout and exit')
    p.add_argument('--sbatch-partition', default=None)
    p.add_argument('--sbatch-time', default=None)
    p.add_argument('--sbatch-account', default=None)
    p.add_argument('--sbatch-gpus', default=None,
                   help='e.g. 1 or a full constraint string for #SBATCH --gpus=')
    p.add_argument('--sbatch-cpus', type=int, default=None)
    p.add_argument('--sbatch-mem', default=None, help='e.g. 32G for #SBATCH --mem=')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.emit_sbatch:
        _emit_sbatch(args)
        return

    os.chdir(_REPO_ROOT)
    job_id = _resolve_job_id(args.job_id)
    n_jobs = args.n_jobs
    if n_jobs < 1:
        raise SystemExit('--n-jobs must be >= 1')
    if job_id < 0 or job_id >= n_jobs:
        raise SystemExit(f'--job-id {job_id} out of range for --n-jobs {n_jobs}')

    run_date = args.run_date
    if run_date is None:
        run_date = date.today().strftime('%Y%m%d')

    if args.tracks_yaml is not None:
        tracks = _load_tracks_yaml(args.tracks_yaml)
        yaml_params = _load_params_yaml(args.tracks_yaml)
    else:
        tracks = _default_boundary_track_dicts(
            seed=args.track_seed, n=args.n_boundary_tracks)
        yaml_params = None

    if args.params is not None:
        param_list = list(args.params)
    elif yaml_params is not None:
        param_list = yaml_params
    else:
        param_list = list(VALID_PARAMS)

    pairs = _param_pairs(param_list)
    all_tasks = _iter_tasks(tracks, pairs)
    my_tasks = _filter_tasks_for_job(all_tasks, job_id, n_jobs)

    landscape_script = _REPO_ROOT / 'src' / 'analysis' / '2d_loss_landscape.py'
    if not landscape_script.is_file():
        raise SystemExit(f'Missing {landscape_script}')

    py = _venv_python()
    root_out = Path(_RESULTS_DIR) / 'landscape' / run_date

    print(f'Repo     : {_REPO_ROOT}')
    print(f'Python   : {py}')
    print(f'Out root : {root_out}')
    print(f'Job      : {job_id + 1}/{n_jobs}  ({len(my_tasks)} tasks in this worker)')
    print(f'Tracks   : {len(tracks)}   Pairs : {len(pairs)}   Total tasks : {len(all_tasks)}')
    if args.tracks_yaml is None:
        print(f'           (default: {args.n_boundary_tracks} random + 1 diagonal cross, seed={args.track_seed})')

    for track, param_y, param_x in my_tasks:
        track_dir = root_out / str(track['name'])
        track_dir.mkdir(parents=True, exist_ok=True)
        out_pkl = track_dir / f'{param_y}__{param_x}.pkl'
        cmd = _build_landscape_command(
            python_exe=py,
            param_y=param_y,
            param_x=param_x,
            track=track,
            results_dir_for_track=track_dir,
            output_pkl=out_pkl,
            grid=args.grid,
            range_frac=args.range_frac,
            loss=args.loss,
            gradients=args.gradients,
            noise_scale=args.noise_scale,
            noise_seed=args.noise_seed,
        )
        printable = shlex.join(cmd)
        if args.dry_run:
            print(printable)
            continue
        print(f'\n>>> {track["name"]}  {param_y}  {param_x}', flush=True)
        subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))

    print('\nDone.')


if __name__ == '__main__':
    main()
