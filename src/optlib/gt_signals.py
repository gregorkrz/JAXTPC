"""
Cache of per-track (noisy) GT detector-response signals, keyed by track name.

Lets ``run_optimization.py`` skip the expensive ``generate_muon_track`` ->
``build_deposit_data`` -> ``simulator.forward`` loop (one un-batched, real
detector-physics forward pass per track) when the same track ensemble has
already been computed by a prior run — e.g. sharded across several parallel
precompute jobs, or reused across otherwise-identical training runs that only
differ in a training hyperparameter (rotor weight, dropout, ...).

The cached signal already has detector noise applied (``apply_noise_to_gt``,
same track-indexed seeding run_optimization.py itself uses) — i.e. it's the
literal training target, not a clean signal needing noise re-applied at load
time. Because the noise draw depends on ``nn_seed`` (via
``SeedSequence(args.seed)`` -> ``noise_seed``), a cache is only valid for
consumers using the SAME ``noise_seed`` it was written with; ``noise_seed`` is
stored as a file attribute and validated on load (``load_gt_cache_lazy``).

``load_gt_cache_lazy`` returns a dict-like ``LazyGtCache`` that bulk-reads one
shard file at a time into memory on first access to any of its tracks, keeps at
most ``max_open_shards`` (default 2) shards' worth of decoded arrays cached
(LRU eviction), and serves further lookups for that shard's other tracks from
memory — instead of materializing every track into host RAM up front (at
N_random_tracks in the hundreds-to-thousands, eagerly holding every track's
full-res signal, ~100+ MB each, is what caused host-memory OOMs) or re-reading
from disk for every single track. This relies on training visiting batches in
``track_specs`` order (a fixed, contiguous cycle — see run_optimization.py's
modular batch indexing) so consecutive lookups stay within 1-2 shards at a time.
"""
import glob as _glob
import os
import time as _time
from collections import OrderedDict as _OrderedDict

import h5py
import numpy as np


def _track_group_name(name: str) -> str:
    return name


def _expand_paths(paths):
    if isinstance(paths, str):
        paths = [paths]
    expanded = []
    for p in paths:
        expanded.extend(sorted(_glob.glob(p)) if any(c in p for c in '*?[') else [p])
    return expanded


def save_gt_cache_h5(path, track_specs, gt_signals_per_track, *, noise_seed=None,
                      compress=True):
    """Write (noisy) GT signals for ``track_specs`` to an h5 file, keyed by track name.

    Parameters
    ----------
    path : str
    track_specs : list[dict]
        Each has keys name, direction, momentum_mev, start_position_mm.
    gt_signals_per_track : list[tuple[np.ndarray, ...]]
        Aligned with track_specs; each tuple is the per-(volume, plane) signal
        arrays (with noise already applied, if any) returned by ``simulator.forward``
        + ``apply_noise_to_gt``.
    noise_seed : int, optional
        The integer noise seed (see ``run_optimization.py``'s ``noise_seed`` /
        ``_noise_seed_from_result_seed``-style derivation) used to generate the
        per-track noise draws baked into this file. Stored as a file attribute and
        checked by ``load_gt_cache_lazy(..., expected_noise_seed=...)`` so a cache
        can't silently be reused with a mismatched noise realization.
    compress : bool, optional
        Default True (gzip level 4, matching prior behavior). The cached signal has
        real detector noise baked in everywhere, so it's nearly incompressible —
        measured ~16% size reduction on a real shard — while gzip decompression is a
        meaningful chunk of read time (measured ~46% of wall-clock on a cold read of a
        real 2.1GB shard, the rest being network I/O wait). Pass False to store raw
        float32 with no compression: ~19% larger on disk, but removes that CPU cost
        entirely and frees host CPU that's otherwise competing with the training loop.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with h5py.File(path, 'w') as f:
        if noise_seed is not None:
            f.attrs['noise_seed'] = int(noise_seed)
        for ts, sig in zip(track_specs, gt_signals_per_track):
            grp = f.create_group(_track_group_name(ts['name']))
            grp.attrs['direction'] = np.asarray(ts['direction'], dtype=np.float64)
            grp.attrs['momentum_mev'] = float(ts['momentum_mev'])
            grp.attrs['start_position_mm'] = np.asarray(
                ts['start_position_mm'], dtype=np.float64)
            grp.attrs['n_arrays'] = len(sig)
            comp_kw = dict(compression='gzip', compression_opts=4) if compress else {}
            for i, arr in enumerate(sig):
                grp.create_dataset(f'sig_{i}', data=np.asarray(arr, dtype=np.float32),
                                   **comp_kw)


def load_gt_cache_h5(paths):
    """Eagerly load one or more GT-cache h5 files (shards) into a {name: tuple} dict.

    Prefer ``load_gt_cache_lazy`` for large track counts — this materializes every
    track's arrays in host RAM at once, which is exactly the O(N_tracks) memory
    behavior that's unsafe for N_random_tracks in the hundreds-to-thousands.
    ``paths`` may be a single path, a list of paths, or a glob pattern; shards
    are merged. Raises if the same track name appears in more than one file
    (shards are expected to be disjoint).
    """
    paths = _expand_paths(paths)
    if not paths:
        raise FileNotFoundError('load_gt_cache_h5: no files matched the given path(s)')

    cache = {}
    for path in sorted(paths):
        with h5py.File(path, 'r') as f:
            for name in f.keys():
                if name in cache:
                    raise ValueError(f'load_gt_cache_h5: track {name!r} appears in '
                                      f'more than one shard (last seen in {path})')
                grp = f[name]
                n_arrays = int(grp.attrs['n_arrays'])
                cache[name] = tuple(np.asarray(grp[f'sig_{i}']) for i in range(n_arrays))
    return cache


class LazyGtCache:
    """Dict-like {name: tuple[np.ndarray, ...]} accessor backed by h5 shard files.

    Reads a whole shard in one bulk pass the first time any of its tracks is
    looked up, caches the decoded {name: tuple} dict for up to ``max_open_shards``
    shards (LRU eviction — least-recently-touched shard's dict is dropped, not
    a file handle; nothing stays open on disk), and serves further lookups for
    that shard's other tracks from memory instead of re-reading per track.

    This assumes (and run_optimization.py's usage guarantees) that batches are
    visited in track_specs order, cycling repeatedly — since each shard covers a
    contiguous range of track_specs and batches are built from consecutive
    tracks, at most ~2 shards are ever "hot" at once even though training
    revisits every batch thousands of times over the run. Building the index
    only lists each shard's group names (cheap — no dataset reads).
    """

    def __init__(self, paths, expected_noise_seed=None, max_open_shards=2):
        self._paths = _expand_paths(paths)
        if not self._paths:
            raise FileNotFoundError('LazyGtCache: no files matched the given path(s)')
        self._name_to_path = {}
        self._max_open_shards = max_open_shards
        self._shard_cache = _OrderedDict()  # path -> {name: tuple[np.ndarray, ...]}, LRU-ordered
        for path in sorted(self._paths):
            with h5py.File(path, 'r') as f:
                seed = f.attrs.get('noise_seed')
                if expected_noise_seed is not None:
                    if seed is None:
                        raise ValueError(
                            f'LazyGtCache: {path} has no stored noise_seed attribute '
                            f'(stale/old-format cache?) but expected_noise_seed='
                            f'{expected_noise_seed} was given. Regenerate this cache '
                            f'with the current precompute profile.')
                    if int(seed) != int(expected_noise_seed):
                        raise ValueError(
                            f'LazyGtCache: {path} was cached with noise_seed={int(seed)}, '
                            f'but this run expects noise_seed={expected_noise_seed} '
                            f'(derived from a different --seed/nn_seed). Regenerate the '
                            f'cache with the seed this run actually uses.')
                for name in f.keys():
                    if name in self._name_to_path:
                        raise ValueError(
                            f'LazyGtCache: track {name!r} appears in more than one '
                            f'shard (last seen in {path})')
                    self._name_to_path[name] = path

    def __contains__(self, name):
        return name in self._name_to_path

    def __len__(self):
        return len(self._name_to_path)

    @staticmethod
    def _load_shard(path):
        t0 = _time.time()
        shard = {}
        with h5py.File(path, 'r') as f:
            for name in f.keys():
                grp = f[name]
                n_arrays = int(grp.attrs['n_arrays'])
                shard[name] = tuple(np.asarray(grp[f'sig_{i}']) for i in range(n_arrays))
        print(f'[LazyGtCache] read shard {path} ({len(shard)} tracks) in '
              f'{_time.time() - t0:.2f}s', flush=True)
        return shard

    def __getitem__(self, name):
        path = self._name_to_path.get(name)
        if path is None:
            raise KeyError(name)
        if path in self._shard_cache:
            self._shard_cache.move_to_end(path)
        else:
            self._shard_cache[path] = self._load_shard(path)
            while len(self._shard_cache) > self._max_open_shards:
                self._shard_cache.popitem(last=False)
        return self._shard_cache[path][name]

    def close(self):
        self._shard_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def load_gt_cache_lazy(paths, expected_noise_seed=None, max_open_shards=2):
    return LazyGtCache(paths, expected_noise_seed=expected_noise_seed,
                        max_open_shards=max_open_shards)
