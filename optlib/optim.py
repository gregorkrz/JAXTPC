"""Optimizer factory, LR-multiplier logic, burn-in, and phase/batch plumbing.

Extracted verbatim from ``src/opt/run_optimization.py`` (no logic changes).
``_unpack_batch_fn_ret`` / ``_phase_index_at`` live here (used by the grad helpers
below and re-imported by the driver's build_phase_fns / run_trial) to avoid a cycle.
"""

import jax
import jax.numpy as jnp
import optax

from optlib.constants import ADAM_BETA1, ADAM_BETA2, ADAM_EPS, MOMENTUM


def _unpack_batch_fn_ret(ret):
    if len(ret) == 3:
        return ret[0], ret[1], ret[2]
    lv, gv = ret
    return lv, gv, None


def _phase_index_at(step, phase_schedule):
    for ph_idx, (until_step, _) in enumerate(phase_schedule):
        if step < until_step:
            return ph_idx
    return len(phase_schedule) - 1


def sum_grad_batches_at_step(p0, phase_schedule, start_step):
    """Sum ∂L/∂p over all batches for the phase active at ``start_step`` (same convention as run_trial)."""
    p = jnp.asarray(p0, dtype=jnp.float32)
    ph_idx = _phase_index_at(start_step, phase_schedule)
    _, build_fn = phase_schedule[ph_idx]
    fns = build_fn(p)
    gv_acc = jnp.zeros_like(p)
    for fn in fns:
        lv, gv, _ = _unpack_batch_fn_ret(fn(p))
        jax.block_until_ready((lv, gv))
        gv_acc = gv_acc + gv
    return gv_acc


def burn_in_mean_abs_grad(p0, phase_schedule, optimizer, burn_in_steps, effective_batch_size=1):
    """Run ``burn_in_steps`` trial-like optimizer steps and return mean |∂L/∂p| per coordinate.

    Uses the same phase-vs-step indexing as ``run_trial`` (so schedule boundaries apply).
    ``optimizer`` should be built **without** per-param LR multipliers (uniform scaling).

    Returns:
        mean_abs: JAX vector, time-average of |grad| over the burn-in window
        steps_used: int, number of steps actually run (``burn_in_steps``)
    """
    n_steps = int(burn_in_steps)
    eff_bs = int(effective_batch_size)
    if n_steps <= 0:
        raise ValueError('burn_in_mean_abs_grad: burn_in_steps must be positive')
    if eff_bs < 1:
        raise ValueError('burn_in_mean_abs_grad: effective_batch_size must be >= 1')
    p = jnp.asarray(p0, dtype=jnp.float32)
    opt_state = optimizer.init(p)
    _cur_ph_idx = [-1]
    _cur_fns = [None]

    def _get_fns(ph_idx, p_):
        if ph_idx != _cur_ph_idx[0]:
            _cur_fns[0] = None
            _cur_ph_idx[0] = ph_idx
            _, build_fn = phase_schedule[ph_idx]
            _cur_fns[0] = build_fn(p_)
        return _cur_fns[0]

    def _phase_at(step, p_):
        ph_idx = _phase_index_at(step, phase_schedule)
        return ph_idx, _get_fns(ph_idx, p_)

    sum_abs = jnp.zeros_like(p)
    for step in range(n_steps):
        _, batch_fns = _phase_at(step, p)
        gv_acc = jnp.zeros_like(p)
        for micro in range(eff_bs):
            batch_idx = (step * eff_bs + micro) % len(batch_fns)
            fn = batch_fns[batch_idx]
            lv, gv, _ = _unpack_batch_fn_ret(fn(p))
            jax.block_until_ready((lv, gv))
            sum_abs = sum_abs + jnp.abs(gv)
            gv_acc = gv_acc + gv
        gv_eff = gv_acc / float(eff_bs)
        updates, opt_state = optimizer.update(gv_eff, opt_state)
        p = optax.apply_updates(p, updates)

    mean_abs = sum_abs / float(n_steps * eff_bs)
    return mean_abs, n_steps


def auto_lr_multipliers_from_grad(gv):
    """Per-parameter scales from per-coordinate sensitivity (non-negative).

    sens_i = |v_i| (typically v = ∂L/∂p or a time-mean thereof), then
    lr_mult_i = clip(median(sens) / (sens_i + 1e-8), 0.01, 10).

    Returns (multipliers list, median(sens), list of sens_i per param).
    """
    sens = jnp.abs(gv)
    med = jnp.median(sens)
    mult = jnp.clip(med / (sens + 1e-8), 0.01, 10.0)
    mult_list = [float(x) for x in mult]
    sens_list = [float(x) for x in sens]
    return mult_list, float(med), sens_list


def _scale_by_vector(scales):
    """Optax transform that element-wise multiplies gradients by a fixed scale vector."""
    scales_arr = jnp.array(scales, dtype=jnp.float32)
    def init_fn(params):
        return ()
    def update_fn(updates, state, params=None):
        return updates * scales_arr, state
    return optax.GradientTransformation(init_fn, update_fn)


def make_optax_optimizer(optimizer_name, lr, lr_schedule, max_steps, clip_grad_norm=0.0,
                         warmup_steps=0, lr_multipliers=None, adam_beta2=ADAM_BETA2):
    if warmup_steps > 0:
        warmup = optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps)
        if lr_schedule == 'cosine':
            post = optax.cosine_decay_schedule(init_value=lr, decay_steps=max_steps - warmup_steps)
        else:
            post = optax.constant_schedule(lr)
        schedule = optax.join_schedules([warmup, post], boundaries=[warmup_steps])
    elif lr_schedule == 'cosine':
        schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=max_steps)
    else:
        schedule = lr
    if optimizer_name == 'adam':         base = optax.adam(schedule, b1=ADAM_BETA1, b2=adam_beta2, eps=ADAM_EPS)
    elif optimizer_name == 'sgd':          base = optax.sgd(schedule)
    elif optimizer_name == 'momentum_sgd': base = optax.sgd(schedule, momentum=MOMENTUM)
    elif optimizer_name == 'newton':
        raise ValueError(
            'Newton optimizer bypasses optax entirely — do not call make_optax_optimizer for newton')
    else: raise ValueError(f'Unknown optimizer {optimizer_name!r}')
    transforms = []
    if lr_multipliers is not None and any(s != 1.0 for s in lr_multipliers):
        transforms.append(_scale_by_vector(lr_multipliers))
    if clip_grad_norm > 0.0:
        transforms.append(optax.clip_by_global_norm(clip_grad_norm))
    transforms.append(base)
    tx = optax.chain(*transforms)
    # Wrap schedule in a callable so callers can query lr at any step
    schedule_fn = schedule if callable(schedule) else (lambda _s, _lr=schedule: _lr)
    return tx, schedule_fn
