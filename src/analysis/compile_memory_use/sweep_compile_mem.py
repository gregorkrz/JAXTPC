#!/usr/bin/env python
"""
Sweep over batch sizes, max_deposits, and XLA effort levels, running
measure_compile_mem.py for each combination sequentially (each as a
fresh subprocess so JAX peak-memory counters reset cleanly).

Usage
-----
    python src/analysis/sweep_compile_mem.py \\
        --batch-sizes 1,2,4 \\
        --max-deposits 10000,25000,50000 \\
        --xla-efforts 0,2,3 \\
        --output-csv results/compile_mem_sweep.csv

    # Minimal single run (for testing)
    python src/analysis/sweep_compile_mem.py \\
        --batch-sizes 1 --max-deposits 50000 --xla-efforts 3

    # On S3DF use:
    python src/analysis/sweep_compile_mem.py \\
        --python /sdf/home/g/gregork/envs/base_env/bin/python \\
        --batch-sizes 1,2,4 --max-deposits 10000,50000 --xla-efforts 0,3
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from itertools import product


_SCRIPT = os.path.join(os.path.dirname(__file__), 'measure_compile_mem.py')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--batch-sizes',   default='1',
                   help='Comma-separated batch sizes to sweep (default: 1)')
    p.add_argument('--max-deposits',  default='50000',
                   help='Comma-separated max-deposit values to sweep (default: 50000)')
    p.add_argument('--xla-efforts',   default='3',
                   help='Comma-separated XLA effort levels 0-3 to sweep (default: 3)')
    p.add_argument('--n-params',      type=int, default=2,
                   help='Number of optimisation params in p_n_vec (default: 2)')
    p.add_argument('--loss',          default='sobolev_loss_geomean_log1p')
    p.add_argument('--step-size-mm',  type=float, default=0.1,
                   help='Track step size in mm (default: 0.1)')
    p.add_argument('--momentum-mev',  type=float, default=1000.0)
    p.add_argument('--track-direction', default='1,1,1')
    p.add_argument('--output-csv',    default=None,
                   help='Write summary CSV to this path')
    p.add_argument('--python',        default=sys.executable,
                   help=f'Python interpreter to use (default: {sys.executable})')
    p.add_argument('--timeout',       type=int, default=1800,
                   help='Timeout per subprocess in seconds (default: 1800)')
    return p.parse_args()


def _csv_ints(s):
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def run_one(python, batch_size, max_deposits, xla_effort, args):
    """Run measure_compile_mem.py in a subprocess, return result dict."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
    tmp.close()

    cmd = [
        python, _SCRIPT,
        '--batch-size',      str(batch_size),
        '--max-deposits',    str(max_deposits),
        '--xla-effort',      str(xla_effort),
        '--n-params',        str(args.n_params),
        '--loss',            args.loss,
        '--step-size-mm',    str(args.step_size_mm),
        '--momentum-mev',    str(args.momentum_mev),
        '--track-direction', args.track_direction,
        '--output-json',     tmp.name,
    ]
    print(f'\n{"="*70}')
    print(f'  [{batch_size=}  {max_deposits=}  {xla_effort=}]')
    print(f'  cmd: {" ".join(cmd)}')
    print('='*70, flush=True)

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(f'  TIMEOUT after {args.timeout}s')
        return dict(batch_size=batch_size, max_deposits=max_deposits,
                    xla_effort=xla_effort, error='timeout',
                    wall_s=args.timeout)
    except Exception as e:
        print(f'  ERROR: {e}')
        return dict(batch_size=batch_size, max_deposits=max_deposits,
                    xla_effort=xla_effort, error=str(e),
                    wall_s=round(time.time() - t0, 1))
    finally:
        pass

    if proc.returncode != 0:
        print(f'  FAILED (returncode={proc.returncode})')
        return dict(batch_size=batch_size, max_deposits=max_deposits,
                    xla_effort=xla_effort,
                    error=f'returncode={proc.returncode}',
                    wall_s=round(time.time() - t0, 1))

    try:
        with open(tmp.name) as f:
            result = json.load(f)
        os.unlink(tmp.name)
        return result
    except Exception as e:
        print(f'  WARNING: could not read result JSON: {e}')
        return dict(batch_size=batch_size, max_deposits=max_deposits,
                    xla_effort=xla_effort, error='no_result_json',
                    wall_s=round(time.time() - t0, 1))


_CSV_FIELDS = [
    'batch_size', 'max_deposits', 'xla_effort', 'n_params',
    'oom', 'compile_time_s', 'jax_peak_total_gib', 'jax_peak_delta_gib',
    'nvml_used_mb', 'total_wall_s', 'error',
]


def print_table(results):
    header = (f"{'bs':>4}  {'max_dep':>8}  {'effort':>6}  "
              f"{'compile_s':>10}  {'peak_tot_GiB':>13}  "
              f"{'peak_delta_GiB':>14}  {'nvml_MB':>8}  {'error'}")
    print('\n' + '─'*len(header))
    print(header)
    print('─'*len(header))
    for r in results:
        err = r.get('error', '')
        print(
            f"{r.get('batch_size','?'):>4}  "
            f"{r.get('max_deposits','?'):>8}  "
            f"{r.get('xla_effort','?'):>6}  "
            f"{r.get('compile_time_s', float('nan')):>10.1f}  "
            f"{r.get('jax_peak_total_gib', float('nan')):>13.3f}  "
            f"{r.get('jax_peak_delta_gib', float('nan')):>14.3f}  "
            f"{r.get('nvml_used_mb') or float('nan'):>8.0f}  "
            f"{err}"
        )
    print('─'*len(header))


def main():
    args = parse_args()

    batch_sizes  = _csv_ints(args.batch_sizes)
    max_deposits = _csv_ints(args.max_deposits)
    xla_efforts  = _csv_ints(args.xla_efforts)
    combos       = list(product(batch_sizes, max_deposits, xla_efforts))

    print(f'Sweep: {len(batch_sizes)} batch sizes × '
          f'{len(max_deposits)} deposit counts × '
          f'{len(xla_efforts)} XLA effort levels = {len(combos)} runs')
    print(f'  batch_sizes  : {batch_sizes}')
    print(f'  max_deposits : {max_deposits}')
    print(f'  xla_efforts  : {xla_efforts}')
    print(f'  python       : {args.python}')

    results = []
    for i, (bs, md, xe) in enumerate(combos, 1):
        print(f'\n[{i}/{len(combos)}]', flush=True)
        r = run_one(args.python, bs, md, xe, args)
        results.append(r)
        print_table(results)

    # ── Write CSV ──────────────────────────────────────────────────────────────
    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction='ignore')
            w.writeheader()
            w.writerows(results)
        print(f'\nCSV written to {args.output_csv}')

    print('\nDone.')


if __name__ == '__main__':
    main()
