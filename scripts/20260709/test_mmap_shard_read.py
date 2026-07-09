"""
Benchmark: fetch a single track's GT signal out of a list of h5 shard files without
decoding whole shards into RAM (the current LazyGtCache._load_shard behavior).

Motivation: for the 1k_tracks_sweep GT cache (50 shards x ~2.1GB, ke500-1500), holding
every shard fully decoded resident (max_open_shards = len(shards)) costs ~105-115GB per
job. Each track lives in its own HDF5 group, and HDF5 chunk compression means a read of
one group's datasets only touches that group's compressed chunks on disk -- in principle
you never need to decode the other 19 tracks in a shard just to read 1. This script tests
two ways of doing that:

  A) plain  -- h5py.File(path, 'r') opened directly against the file on disk, then index
              a single track's datasets. No mmap, no full-shard decode.
  B) mmap   -- the file's bytes are mmap'd (mmap.mmap, read-only) and handed to h5py via
              driver='fileobj' (h5py >= 3.9). Reads lazily page in only the bytes HDF5
              actually touches. Kept-open handles across many shards should cost ~0 extra
              resident memory until something is actually read.

Usage (against the real cache on S3DF):
    .venv/bin/python scripts/20260709/test_mmap_shard_read.py \
        --shard-glob '/fs/ddn/sdf/group/atlas/d/gregork/jaxtpc/results/opt/efield_calib/1k_tracks_sweep/gt_cache/ke500-1500/shard*of50.h5'

Self-test against a small synthetic fixture (no real cache needed):
    .venv/bin/python scripts/20260709/test_mmap_shard_read.py --self-test
"""
import argparse
import glob as _glob
import mmap
import os
import resource
import sys
import time

import h5py
import numpy as np


def _rss_mb():
    """Current process resident set size, MB (Linux: ru_maxrss is KB; macOS: bytes)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1024.0 if sys.platform == 'linux' else ru / (1024.0 * 1024.0)


def list_track_names(path):
    with h5py.File(path, 'r') as f:
        return list(f.keys())


def read_track_plain(path, track_name):
    """Approach A: direct disk-backed h5py.File, read only one track's datasets."""
    t0_wall, t0_cpu = time.time(), time.process_time()
    with h5py.File(path, 'r') as f:
        grp = f[track_name]
        n_arrays = int(grp.attrs['n_arrays'])
        arrays = tuple(np.asarray(grp[f'sig_{i}']) for i in range(n_arrays))
    return arrays, time.time() - t0_wall, time.process_time() - t0_cpu


def open_mmap_h5(path):
    """Open path read-only, mmap its bytes, hand the mmap to h5py via driver='fileobj'.

    Returns (h5py.File, mmap.mmap, raw file object) -- caller must keep all three
    referenced (and eventually close them) or the mapping can be torn down early.
    """
    fh = open(path, 'rb')
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    f = h5py.File(mm, 'r', driver='fileobj')
    return f, mm, fh


def read_track_mmap(f, track_name):
    """Approach B: read one track's datasets out of an already-open mmap-backed h5py.File."""
    t0_wall, t0_cpu = time.time(), time.process_time()
    grp = f[track_name]
    n_arrays = int(grp.attrs['n_arrays'])
    arrays = tuple(np.asarray(grp[f'sig_{i}']) for i in range(n_arrays))
    return arrays, time.time() - t0_wall, time.process_time() - t0_cpu


def run_benchmark(shard_paths, track_shard_idx=0, track_name=None):
    print(f'{len(shard_paths)} shard(s)')
    rss0 = _rss_mb()
    print(f'RSS before anything: {rss0:.1f} MB')

    # --- Approach A: plain, single shard, single track ------------------------------
    target_path = shard_paths[track_shard_idx]
    if track_name is None:
        track_name = list_track_names(target_path)[0]
    print(f'\nTarget: shard={target_path}  track={track_name!r}')

    arrays_a, t_a_wall, t_a_cpu = read_track_plain(target_path, track_name)
    rss_a = _rss_mb()
    total_bytes_a = sum(a.nbytes for a in arrays_a)
    print(f'[A: plain]  read 1 track in {t_a_wall:.4f}s wall / {t_a_cpu:.4f}s cpu  '
          f'({total_bytes_a / 1e6:.2f} MB decoded)  RSS now={rss_a:.1f} MB')

    # --- Approach B: mmap every shard, keep all open, then read one track -----------
    t0 = time.time()
    handles = [open_mmap_h5(p) for p in shard_paths]
    t_open_all = time.time() - t0
    rss_open = _rss_mb()
    print(f'\n[B: mmap]   opened+mmap\'d all {len(shard_paths)} shards in {t_open_all:.4f}s  '
          f'RSS now={rss_open:.1f} MB (delta from before-anything: {rss_open - rss0:.1f} MB)')

    f_target = handles[track_shard_idx][0]
    arrays_b, t_b_wall, t_b_cpu = read_track_mmap(f_target, track_name)
    rss_b = _rss_mb()
    total_bytes_b = sum(a.nbytes for a in arrays_b)
    print(f'[B: mmap]   read 1 track in {t_b_wall:.4f}s wall / {t_b_cpu:.4f}s cpu  '
          f'({total_bytes_b / 1e6:.2f} MB decoded)  RSS now={rss_b:.1f} MB  '
          f'(delta from open-all: {rss_b - rss_open:.1f} MB)')

    for label, t_wall, t_cpu in (('A: plain', t_a_wall, t_a_cpu), ('B: mmap', t_b_wall, t_b_cpu)):
        cpu_frac = t_cpu / t_wall if t_wall > 0 else float('nan')
        bound = ('CPU/decompression-bound' if cpu_frac > 0.7 else
                 'I/O-bound' if cpu_frac < 0.3 else 'mixed')
        print(f'  [{label}] cpu/wall = {cpu_frac:.2f}  -> looks {bound}')

    for a, b in zip(arrays_a, arrays_b):
        if not np.array_equal(a, b):
            raise AssertionError('plain vs mmap read produced different data!')
    print('\nOK: plain and mmap reads agree.')

    for f, mm, fh in handles:
        f.close()
        mm.close()
        fh.close()

    print(f'\nSummary: plain single-shard-open read={t_a_wall * 1000:.1f}ms wall '
          f'({t_a_cpu * 1000:.1f}ms cpu)  '
          f'mmap read (after {len(shard_paths)} shards already open)='
          f'{t_b_wall * 1000:.1f}ms wall ({t_b_cpu * 1000:.1f}ms cpu)  '
          f'RSS after opening all {len(shard_paths)} shards via mmap stayed at '
          f'{rss_open - rss0:.1f} MB above baseline (vs decoding all of them fully, which '
          f'would cost ~shard_size_MB * n_shards).')


def _make_self_test_fixture(tmpdir, n_shards=5, tracks_per_shard=4, arr_shape=(8, 16)):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'optlib'))
    from gt_signals import save_gt_cache_h5  # noqa: E402

    paths = []
    rng = np.random.default_rng(0)
    for s in range(n_shards):
        track_specs = [
            dict(name=f'track_s{s}_t{i}', direction=[0, 0, 1], momentum_mev=500.0,
                 start_position_mm=[0, 0, 0])
            for i in range(tracks_per_shard)
        ]
        sigs = [(rng.standard_normal(arr_shape).astype(np.float32),
                 rng.standard_normal(arr_shape).astype(np.float32))
                for _ in range(tracks_per_shard)]
        path = os.path.join(tmpdir, f'shard{s}of{n_shards}.h5')
        save_gt_cache_h5(path, track_specs, sigs, noise_seed=0)
        paths.append(path)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shard-glob', default=None,
                    help='Glob pattern matching shard h5 files, e.g. '
                         "'.../ke500-1500/shard*of50.h5'.")
    p.add_argument('--track-shard-idx', type=int, default=0,
                    help='Index into the sorted shard list to pull the test track from.')
    p.add_argument('--track-name', default=None,
                    help='Track name to fetch (default: first group in the target shard).')
    p.add_argument('--self-test', action='store_true',
                    help='Build a small synthetic fixture locally and run against it '
                         '(no real GT cache needed).')
    args = p.parse_args()

    if args.self_test:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_self_test_fixture(tmpdir)
            run_benchmark(paths, track_shard_idx=args.track_shard_idx,
                          track_name=args.track_name)
        return

    if not args.shard_glob:
        p.error('--shard-glob is required unless --self-test is passed.')
    shard_paths = sorted(_glob.glob(args.shard_glob))
    if not shard_paths:
        p.error(f'no files matched --shard-glob {args.shard_glob!r}')
    run_benchmark(shard_paths, track_shard_idx=args.track_shard_idx,
                  track_name=args.track_name)


if __name__ == '__main__':
    main()
