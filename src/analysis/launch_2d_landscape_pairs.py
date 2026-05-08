#!/usr/bin/env python
"""
Submit one Slurm job per (track × param-pair) to compute 2D loss landscapes on S3DF.

Each sbatch job runs ``2d_loss_landscape.py`` directly inside Apptainer on a GPU node.
Edit the S3DF / Slurm constants near the top of this file to match your allocation.

Usage
-----
  # Dry run: print expected job count + one example sbatch script
  python src/analysis/launch_2d_landscape_pairs.py --dry-run

  # Dry run with all inner commands printed
  python src/analysis/launch_2d_landscape_pairs.py --dry-run --verbose

  # Submit (default: 15 tracks × 28 pairs = 420 jobs)
  python src/analysis/launch_2d_landscape_pairs.py

  # Fewer params
  python src/analysis/launch_2d_landscape_pairs.py --params velocity_cm_us lifetime_us recomb_alpha

  # Custom tracks from YAML
  python src/analysis/launch_2d_landscape_pairs.py --tracks-yaml my_tracks.yaml

  # Override Slurm settings per run
  python src/analysis/launch_2d_landscape_pairs.py --account neutrino --partition ampere
"""
import argparse
import os
import shlex
import subprocess
import sys
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from tools.random_boundary_tracks import (  # noqa: E402
    N_DEFAULT_BOUNDARY_MUONS,
    generate_random_boundary_tracks,
)

# ──────────────────────────────────────────────────────────────────────────────
# S3DF Slurm / Apptainer settings — edit to match your allocation
# ──────────────────────────────────────────────────────────────────────────────
ACCOUNT      = "mli:nu-ml-dev"
PARTITION    = "ampere"
QOS = "normal"  # type: Optional[str]  # e.g. "normal"; None omits the #SBATCH --qos line
TIME         = "00:30:00"
MEM          = "32G"
GPUS         = "1"
CPUS_PER_GPU = 1

REMOTE_DIR         = "/sdf/home/g/gregork/jaxtpc"
LOGS_DIR           = "/fs/ddn/sdf/group/atlas/d/gregork/logs"
APPTAINER_IMAGE    = "docker://gkrz/jaxtpc:v2"
BIND_MOUNTS        = [
    "/sdf/home/g/gregork/jaxtpc",
    "/fs/ddn/sdf/group/atlas/d/gregork/jaxtpc",
    "/sdf/scratch/atlas/gregork/jax_cache",
]
APPTAINER_CACHEDIR = "/sdf/scratch/atlas/gregork/apptainer_cache"
APPTAINER_TMPDIR   = "/sdf/scratch/atlas/gregork/apptainer_tmp"
JAX_CACHE_DIR      = "/sdf/scratch/atlas/gregork/jax_cache"

# ──────────────────────────────────────────────────────────────────────────────
# Physics parameters — must match src/opt/2d_opt.py VALID_PARAMS
# ──────────────────────────────────────────────────────────────────────────────
VALID_PARAMS = (  # type: Tuple[str, ...]
    'velocity_cm_us',
    'lifetime_us',
    'diffusion_trans_cm2_us',
    'diffusion_long_cm2_us',
    'recomb_alpha',
    'recomb_beta',
    'recomb_beta_90',
    'recomb_R',
)

# ──────────────────────────────────────────────────────────────────────────────
# Repo layout
# ──────────────────────────────────────────────────────────────────────────────
_LANDSCAPE_PY = 'src/analysis/2d_loss_landscape.py'  # relative to REMOTE_DIR


# ─── Track generation ──────────────────────────────────────────────────────────
# Lightweight volume stubs — provide only the ranges_cm attribute that
# generate_random_boundary_tracks needs.  Bounds from
# config/cubic_wireplane_config.yaml: East x∈[-216,0] cm, West x∈[0,216] cm,
# y,z ∈ [-216,216] cm for both.

class _Vol(object):
    def __init__(self, ranges_cm):
        self.ranges_cm = ranges_cm


_VOLUMES = [
    _Vol([[-216.0, 0.0],   [-216.0, 216.0], [-216.0, 216.0]]),  # East
    _Vol([[0.0,   216.0],  [-216.0, 216.0], [-216.0, 216.0]]),  # West
]


def generate_tracks(n=N_DEFAULT_BOUNDARY_MUONS, seed=42):
    # type: (int, int) -> List[Dict[str, Any]]
    """Default ensemble via tools/random_boundary_tracks.generate_random_boundary_tracks."""
    return generate_random_boundary_tracks(_VOLUMES, n=n, seed=seed)


# ─── YAML loader ───────────────────────────────────────────────────────────────

def _load_tracks_yaml(path):
    # type: (Path) -> Tuple[List[Dict[str, Any]], Optional[List[str]]]
    try:
        import yaml
    except ImportError as e:
        raise SystemExit('Install PyYAML to use --tracks-yaml') from e
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not data or 'tracks' not in data:
        raise SystemExit(f'{path}: expected top-level "tracks" list')
    tracks = []
    for i, t in enumerate(data['tracks']):
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
        rec = {  # type: Dict[str, Any]
            'name': str(name),
            'direction': str(direction),
            'momentum_mev': float(mom),
        }
        sp = t.get('start_position_mm')
        if sp is not None:
            if not isinstance(sp, (list, tuple)) or len(sp) != 3:
                raise SystemExit(f'{path}: tracks[{i}] "start_position_mm" must be [x,y,z]')
            rec['start_position_mm'] = tuple(float(x) for x in sp)
        tracks.append(rec)
    params_raw = data.get('params')
    params = list(params_raw) if isinstance(params_raw, list) else None  # type: Optional[List[str]]
    return tracks, params


# ─── Job building ──────────────────────────────────────────────────────────────

def _param_pairs(params):
    # type: (List[str]) -> List[Tuple[str, str]]
    s = sorted(set(params))
    for p in s:
        if p not in VALID_PARAMS:
            raise SystemExit(f'Unknown param {p!r}; valid: {VALID_PARAMS}')
    return list(combinations(s, 2))


def _out_pkl(track, param_y, param_x, run_date, results_dir):
    # type: (Dict[str, Any], str, str, str, str) -> Path
    return Path(results_dir) / 'landscape' / run_date / track['name'] / f'{param_y}__{param_x}.pkl'


def _inner_cmd(track, param_y, param_x, run_date, results_dir, args):
    # type: (Dict[str, Any], str, str, str, str, argparse.Namespace) -> str
    """Build the ``python 2d_loss_landscape.py ...`` command (no sbatch wrapper)."""
    pkl = _out_pkl(track, param_y, param_x, run_date, results_dir)
    out_dir = str(pkl.parent)
    out_pkl = str(pkl)
    d = track['direction']
    direction_str = f'{d[0]},{d[1]},{d[2]}' if isinstance(d, tuple) else str(d)
    parts = [
        'python', _LANDSCAPE_PY,
        '--param-y', param_y,
        '--param-x', param_x,
        '--track-name', track['name'],
        '--direction', direction_str,
        '--momentum', str(track['momentum_mev']),
        '--grid', str(args.grid),
        '--range-frac', str(args.range_frac),
        '--loss', args.loss,
        '--results-dir', out_dir,
        '--output-pkl', out_pkl,
        '--noise-scale', str(args.noise_scale),
        '--noise-seed', str(args.noise_seed),
        '--no-plots',
    ]
    if args.gradients:
        parts.append('--gradients')
    smm = track.get('start_position_mm')
    if smm is not None:
        parts += ['--start-position-mm', str(smm[0]), str(smm[1]), str(smm[2])]
    return ' '.join(shlex.quote(p) for p in parts)


def _build_sbatch_script(track, param_y, param_x, run_date, results_dir, args):
    # type: (Dict[str, Any], str, str, str, str, argparse.Namespace) -> str
    account   = args.account   or ACCOUNT
    partition = args.partition or PARTITION
    qos       = args.qos       or QOS
    time      = args.time      or TIME
    mem       = args.mem       or MEM
    gpus      = args.gpus      or GPUS
    cpus      = args.cpus_per_gpu if args.cpus_per_gpu is not None else CPUS_PER_GPU

    job_name = f'ls_{track["name"][:16]}_{param_y[:8]}_{param_x[:8]}'

    cmd = _inner_cmd(track, param_y, param_x, run_date, results_dir, args)
    inner_script = (
        f'set -euo pipefail && '
        f'export RESULTS_DIR={shlex.quote(results_dir)} && '
        f'{cmd}'
    )
    binds = ' '.join(f'--bind {shlex.quote(p)}' for p in BIND_MOUNTS)
    wrapped = (
        f'apptainer exec --nv {binds} {APPTAINER_IMAGE} '
        f'bash -c {shlex.quote(inner_script)}'
    )

    header = [
        '#!/bin/bash',
        f'#SBATCH --job-name={job_name}',
        f'#SBATCH --account={account}',
        f'#SBATCH --partition={partition}',
        f'#SBATCH --time={time}',
        f'#SBATCH --gpus={gpus}',
        f'#SBATCH --cpus-per-gpu={cpus}',
        f'#SBATCH --mem={mem}',
        f'#SBATCH --output={LOGS_DIR}/{job_name}_%j.out',
        f'#SBATCH --error={LOGS_DIR}/{job_name}_%j.err',
    ]
    if qos:
        header.append(f'#SBATCH --qos={qos}')

    body = [
        'set -euo pipefail',
        f'mkdir -p {shlex.quote(LOGS_DIR)}',
        f'mkdir -p {shlex.quote(JAX_CACHE_DIR)}',
        f'export APPTAINER_CACHEDIR={shlex.quote(APPTAINER_CACHEDIR)}',
        f'export APPTAINER_TMPDIR={shlex.quote(APPTAINER_TMPDIR)}',
        f'cd {shlex.quote(REMOTE_DIR)}',
        'source .env',
        f'export JAX_COMPILATION_CACHE_DIR={shlex.quote(JAX_CACHE_DIR)}',
        'export XLA_PYTHON_CLIENT_PREALLOCATE=false',
        'export TF_GPU_ALLOCATOR=cuda_malloc_async',
        'export WANDB_DISABLE_SERVICE=true',
        '',
        'nvidia-smi',
        '',
        wrapped,
    ]
    return '\n'.join(header + body)


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--tracks-yaml', type=Path, default=None,
                   help='YAML with "tracks" list (optional "params" key)')
    p.add_argument('--params', nargs='*', default=None,
                   help='Subset of VALID_PARAMS to pair (default: all 8 → 28 pairs)')
    p.add_argument('--n-boundary-tracks', type=int, default=N_DEFAULT_BOUNDARY_MUONS,
                   metavar='N',
                   help=f'Random x-face muons before fixed chords (default: {N_DEFAULT_BOUNDARY_MUONS})')
    p.add_argument('--track-seed', type=int, default=42,
                   help='RNG seed for random boundary tracks (default: 42)')
    p.add_argument('--run-date', default=None,
                   help='YYYYMMDD output subdir (default: today)')
    p.add_argument('--results-dir', default=None,
                   help='Output root (default: $RESULTS_DIR or "results")')
    p.add_argument('--grid', type=int, default=10)
    p.add_argument('--range-frac', type=float, default=0.15)
    p.add_argument('--loss', default='sobolev_loss_geomean_log1p')
    p.add_argument('--gradients', action='store_true')
    p.add_argument('--noise-scale', type=float, default=0.0)
    p.add_argument('--noise-seed', type=int, default=0)
    p.add_argument('--dry-run', action='store_true',
                   help='Print job count + one example script; do not submit')
    p.add_argument('--verbose', action='store_true',
                   help='With --dry-run: print all inner commands (one per line)')
    # Slurm overrides (defaults from module-level constants)
    p.add_argument('--account', default=None,
                   help=f'Slurm account (default: {ACCOUNT})')
    p.add_argument('--partition', default=None,
                   help=f'Slurm partition (default: {PARTITION})')
    p.add_argument('--qos', default=None,
                   help=f'Slurm QOS (default: {QOS!r})')
    p.add_argument('--time', default=None,
                   help=f'Wall time (default: {TIME})')
    p.add_argument('--mem', default=None,
                   help=f'Memory (default: {MEM})')
    p.add_argument('--gpus', default=None,
                   help=f'GPUs per task (default: {GPUS})')
    p.add_argument('--cpus-per-gpu', type=int, default=None,
                   help=f'CPUs per GPU (default: {CPUS_PER_GPU})')
    p.add_argument('--N-jobs', type=int, default=-1, metavar='N',
                   help='Max jobs to submit (-1 = all pending, default: -1)')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_date    = args.run_date or date.today().strftime('%Y%m%d')
    results_dir = args.results_dir or os.environ.get('RESULTS_DIR', 'results')

    yaml_params = None  # type: Optional[List[str]]
    if args.tracks_yaml is not None:
        tracks, yaml_params = _load_tracks_yaml(args.tracks_yaml)
    else:
        tracks = generate_tracks(n=args.n_boundary_tracks, seed=args.track_seed)

    if args.params is not None:
        param_list = list(args.params)
    elif yaml_params is not None:
        param_list = yaml_params
    else:
        param_list = list(VALID_PARAMS)

    pairs = _param_pairs(param_list)
    total = len(tracks) * len(pairs)

    print(f'Tracks    : {len(tracks)}')
    print(f'Pairs     : {len(pairs)}  ({len(param_list)} params)')
    print(f'Total jobs: {total}')
    print(f'Run date  : {run_date}')
    print(f'Results   : {results_dir}/landscape/{run_date}/')
    print(f'Account   : {args.account or ACCOUNT}   '
          f'Partition: {args.partition or PARTITION}   '
          f'QOS: {args.qos or QOS!r}')

    if args.dry_run:
        pending = sum(
            0 if _out_pkl(tr, py, px, run_date, results_dir).exists() else 1
            for tr in tracks for py, px in pairs
        )
        print(f'\n--- DRY RUN: {pending}/{total} jobs pending, {total - pending} already done ---')
        if args.verbose:
            print()
            for track in tracks:
                for param_y, param_x in pairs:
                    pkl = _out_pkl(track, param_y, param_x, run_date, results_dir)
                    if pkl.exists():
                        print(f'# SKIP (exists): {pkl}')
                    else:
                        print(_inner_cmd(track, param_y, param_x, run_date, results_dir, args))
        else:
            first_track, first_py, first_px = tracks[0], pairs[0][0], pairs[0][1]
            print('\nExample sbatch script (first job):')
            print('---')
            print(_build_sbatch_script(first_track, first_py, first_px,
                                       run_date, results_dir, args))
        return

    submitted = 0
    skipped = 0
    max_submit = args.N_jobs
    for track in tracks:
        for param_y, param_x in pairs:
            if max_submit >= 0 and submitted >= max_submit:
                break
            pkl = _out_pkl(track, param_y, param_x, run_date, results_dir)
            if pkl.exists():
                skipped += 1
                continue
            script = _build_sbatch_script(track, param_y, param_x,
                                          run_date, results_dir, args)
            result = subprocess.run(
                ['sbatch'],
                input=script,
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                print(f'sbatch error: {result.stderr.strip()}', file=sys.stderr)
                sys.exit(1)
            print(result.stdout.strip())
            submitted += 1
        if max_submit >= 0 and submitted >= max_submit:
            break

    print(f'\nSubmitted {submitted} jobs  ({skipped} already done, skipped).')


if __name__ == '__main__':
    main()
