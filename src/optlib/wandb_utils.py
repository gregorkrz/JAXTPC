"""Weights & Biases logging helpers + GPU metrics.

Extracted verbatim from ``src/opt/run_optimization.py`` (no logic changes), except
``_wandb`` is explicitly set to ``None`` when wandb is unavailable so the name can
be imported by the driver (all uses remain guarded by ``_WANDB_AVAILABLE``).
"""

import hashlib
import os

import jax
import numpy as np

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    _WANDB_AVAILABLE = False


def _wandb_track_metric_suffix(track_name):
    """Fragment safe for use inside W&B metric keys."""
    return str(track_name).replace('/', '_').replace(' ', '_')


def _wandb_json_safe(value):
    """Convert values for wandb.init(config=...) / JSON-ish summaries."""
    if isinstance(value, tuple):
        return [_wandb_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_wandb_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _wandb_json_safe(v) for k, v in value.items()}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def wandb_config_dict(args, *, param_names, track_specs, schedule, effective_seed,
                      output_path, wandb_tag_list, argv_cmd):
    """Full CLI snapshot plus derived fields for W&B run config."""
    cfg = {k: _wandb_json_safe(v) for k, v in vars(args).items()}
    cfg.update(
        effective_seed=effective_seed,
        command=argv_cmd,
        param_names=param_names,
        track_names=[t['name'] for t in track_specs],
        track_specs_full=[{
            'name': t['name'],
            'direction': list(t['direction']),
            'momentum_mev': t['momentum_mev'],
            'start_position_mm': list(t['start_position_mm']) if t.get('start_position_mm') else None,
        } for t in track_specs],
        schedule_phases=[_wandb_json_safe(ph) for ph in schedule],
        output_path=output_path,
        jax_compilation_cache_dir=jax.config.jax_compilation_cache_dir,
        wandb_tags=wandb_tag_list or None,
    )
    return cfg


def _wandb_sidecar_path(output_dir, seed):
    return os.path.join(output_dir, f'.wandb_run_id_{seed}')


def _read_stored_wandb_run_id(output_dir, seed, existing_result):
    if existing_result:
        rid = existing_result.get('wandb_run_id')
        if isinstance(rid, str) and rid.strip():
            return rid.strip()
    path = _wandb_sidecar_path(output_dir, seed)
    if os.path.isfile(path):
        try:
            with open(path, encoding='utf-8') as f:
                s = f.read().strip()
                return s or None
        except OSError:
            return None
    return None


def _stable_wandb_run_id(project, folder_name, seed):
    """Deterministic W&B run id when none was persisted (legacy checkpoints)."""
    digest = hashlib.sha256(f'{project}:{folder_name}:{seed}'.encode()).hexdigest()
    return digest[:12]


def _write_wandb_sidecar(output_dir, seed, run_id):
    if not run_id:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = _wandb_sidecar_path(output_dir, seed)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(run_id)


def _collect_gpu_metrics():
    """Collect GPU utilization and memory from nvidia-smi and JAX device memory stats."""
    metrics = {}
    devs = jax.local_devices()
    for i, dev in enumerate(devs):
        try:
            mem = dev.memory_stats()
            if mem:
                pfx = f'gpu{i}' if len(devs) > 1 else 'gpu'
                metrics[f'sys/{pfx}/jax_mem_gb']  = mem.get('bytes_in_use', 0) / 2**30
                metrics[f'sys/{pfx}/jax_peak_gb'] = mem.get('peak_bytes_in_use', 0) / 2**30
        except Exception:
            pass
    try:
        import subprocess
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=index,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode()
        for line in out.strip().splitlines():
            idx_s, util_s, used_s, total_s = [s.strip() for s in line.split(',')]
            pfx = f'sys/gpu{idx_s}'
            metrics[f'{pfx}/util_pct']     = float(util_s)
            metrics[f'{pfx}/mem_used_gb']  = float(used_s) / 1024.0
            metrics[f'{pfx}/mem_total_gb'] = float(total_s) / 1024.0
    except Exception:
        pass
    return metrics


def _wandb_log_step(step, loss, gv, p, param_names, scales, p_n_gts,
                    step_time_s, trial_idx, schedule_fn=None, lr_multipliers=None,
                    phase=None, extra_metrics=None):
    """Log one step to W&B."""
    p_np  = np.array(p)
    gv_np = np.array(gv)

    log = {
        'trial':          trial_idx,
        'loss':           loss,
        'grad_norm':      float(np.linalg.norm(gv_np)),
        'param_norm':     float(np.linalg.norm(p_np)),
        'step_time_s':    step_time_s,
    }
    if phase is not None:
        log['phase'] = phase
    if schedule_fn is not None:
        log['lr'] = float(schedule_fn(step))

    if param_names is not None:
        for i, name in enumerate(param_names):
            log[f'params/{name}_normalized'] = float(p_np[i])
            if scales is not None:
                log[f'params/{name}_physical'] = float(np.exp(p_np[i]) * scales[i])
            if p_n_gts is not None:
                # rel_err in physical space: |exp(q) - exp(q_gt)| / exp(q_gt) = |exp(q - q_gt) - 1|
                rel_err = abs(float(np.exp(p_np[i] - p_n_gts[i])) - 1.0)
                log[f'params/{name}_rel_err'] = rel_err
            if gv_np is not None:
                scale = lr_multipliers[i] if lr_multipliers is not None else 1.0
                log[f'grads/{name}'] = float(gv_np[i] * scale)

    if extra_metrics:
        log.update(extra_metrics)

    log.update(_collect_gpu_metrics())
    _wandb.log(log, step=step)


def fetch_init_params_from_wandb(run_id, param_names, scales, wandb_project, step=-1):
    """Fetch physical param values from a W&B run and return a p_n list.

    step=-1 (default): use the run summary (last logged value).
    step>=0: fetch that specific logged step via scan_history.
    Raises ValueError if any params/<name>_physical key is missing.
    """
    if not _WANDB_AVAILABLE:
        raise RuntimeError('wandb not installed; cannot use --init-from-wandb-run')
    api = _wandb.Api()
    run = api.run(f"{wandb_project}/{run_id}")
    keys = [f"params/{name}_physical" for name in param_names]

    if step < 0:
        source = run.summary
        row = {k: source.get(k) for k in keys}
        step_label = 'summary (latest)'
    else:
        rows = list(run.scan_history(keys=keys, min_step=step, max_step=step + 1))
        if not rows:
            raise ValueError(
                f"No history row found at step {step} in W&B run {run_id}. "
                f"Check that this step was actually logged."
            )
        row = rows[0]
        step_label = f'step {step}'

    p_n_init = []
    for name, scale, key in zip(param_names, scales, keys):
        val = row.get(key)
        if val is None:
            available = sorted(k for k in run.summary.keys() if k.startswith('params/'))
            raise ValueError(
                f"Key {key!r} not found at {step_label} in W&B run {run_id}.\n"
                f"Available params/* keys in summary: {available}"
            )
        p_n = float(np.log(float(val) / scale))
        p_n_init.append(p_n)
        print(f"  {name}: {float(val):.6g}  (p_n={p_n:.4f})")
    return p_n_init
