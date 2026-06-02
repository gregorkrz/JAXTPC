"""Output path / folder naming, pickle IO, and run-completion checks.

Extracted verbatim from ``src/opt/run_optimization.py`` (no logic changes).
"""

import os
import pickle
import tempfile

import jax
import numpy as np

from optlib.constants import _BASE_PARAMS, _BETA_VARIANTS


# ── Folder name ────────────────────────────────────────────────────────────────

def make_folder_name(param_names, track_specs, loss_name, optimizer, lr,
                     lr_schedule, max_steps, N, range_intervals,
                     noise_scale=0.0, step_size=0.1, max_num_deposits=50_000, n_phases=1,
                     active_planes=None):
    _is_all = (_BASE_PARAMS <= frozenset(param_names) and
               bool(frozenset(param_names) & _BETA_VARIANTS))
    params_tag = 'all_params' if _is_all else '+'.join(param_names)
    tracks_tag = (f'{len(track_specs)}tracks' if len(track_specs) >= 6
                  else '+'.join(t['name'] for t in track_specs))
    sched_tag     = '_cosine' if lr_schedule == 'cosine' else ''
    range_tag     = '_'.join(f'r{lo:.3g}_{hi:.3g}' for lo, hi in range_intervals).replace('.', 'p')
    noise_tag     = f'_noise{noise_scale:.3g}'.replace('.', 'p') if noise_scale > 0.0 else ''
    ss_tag        = f'_ss{step_size:.3g}'.replace('.', 'p') if step_size != 0.1 else ''
    dep_tag       = f'_dep{max_num_deposits // 1000}k' if max_num_deposits != 50_000 else ''
    phase_tag     = f'_sched{n_phases}' if n_phases > 1 else ''
    _all_planes   = tuple(range(6))
    planes_tag    = (f'_planes{"".join(str(p) for p in active_planes)}'
                     if active_planes is not None and tuple(active_planes) != _all_planes else '')
    return (f'{params_tag}__{tracks_tag}__{loss_name}__'
            f'{optimizer}_lr{lr}{sched_tag}_s{max_steps}_N{N}_{range_tag}'
            f'{noise_tag}{ss_tag}{dep_tag}{phase_tag}{planes_tag}')


def next_result_path(folder, seed=None):
    """Return the output pkl path inside folder.

    If seed is given: results/<folder>/result_<seed>.pkl (fixed, deterministic).
    If seed is None:  results/<folder>/result_0.pkl, result_1.pkl, ... (first unused).
    """
    os.makedirs(folder, exist_ok=True)
    if seed is not None:
        return os.path.join(folder, f'result_{seed}.pkl')
    i = 0
    while True:
        path = os.path.join(folder, f'result_{i}.pkl')
        if not os.path.exists(path):
            return path
        i += 1


# ── Pickle IO ──────────────────────────────────────────────────────────────────

def _serialize_opt_state(opt_state):
    """Convert JAX arrays in optax state to numpy arrays for pickling."""
    return jax.tree_util.tree_map(np.asarray, opt_state)


def _safe_pickle_dump(path, obj):
    """Write pickle atomically so interrupted writes do not leave truncated pkls."""
    path = os.path.abspath(path)
    out_dir = os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".result_", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── Run completion ───────────────────────────────────────────────────────────────

def optimization_run_complete(data, min_steps=2000):
    """Return True when all N trials are present (nothing left to optimize).

    ``run_complete`` is only written on clean shutdown and is **not** required here:
    treating ``run_complete=False`` after SIGTERM even though ``trials`` is full
    used to queue redundant Slurm jobs (duplicate W&B runs).

    Also returns True when a live_checkpoint exists with more than ``min_steps``
    steps, treating SIGTERM'd mid-trial runs as effectively complete.
    """
    trials = data.get("trials")
    if trials is None:
        return False
    n_expected = data.get("N")
    if not isinstance(n_expected, int) or n_expected < 0:
        return False
    live_ckpt = data.get("live_checkpoint")
    if live_ckpt and live_ckpt.get("step", 0) > min_steps:
        return True
    if len(trials) < n_expected:
        return False
    if live_ckpt:
        return False
    return True
