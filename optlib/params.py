"""Parameter helpers: GT value extraction and SimParams setters.

Extracted verbatim from ``src/opt/run_optimization.py`` (no logic changes).
"""

import jax.numpy as jnp
import numpy as np

from optlib.constants import TYPICAL_SCALES


# ── Physics helpers (mirrors 2d_opt.py) ───────────────────────────────────────

def _get_gt_val(param_name, gt_params, recomb_model):
    rp = gt_params.recomb_params
    if param_name == 'velocity_cm_us':         return float(gt_params.velocity_cm_us)
    if param_name == 'lifetime_us':            return float(gt_params.lifetime_us)
    if param_name == 'diffusion_trans_cm2_us': return float(gt_params.diffusion_trans_cm2_us)
    if param_name == 'diffusion_long_cm2_us':  return float(gt_params.diffusion_long_cm2_us)
    if param_name == 'recomb_alpha':           return float(rp.alpha)
    if param_name == 'recomb_beta':
        if recomb_model != 'modified_box':
            raise ValueError('recomb_beta requires modified_box model')
        return float(rp.beta)
    if param_name == 'recomb_beta_90':
        if recomb_model != 'emb':
            raise ValueError('recomb_beta_90 requires emb model')
        return float(rp.beta_90)
    if param_name == 'recomb_R':
        if recomb_model != 'emb':
            raise ValueError('recomb_R requires emb model')
        return float(rp.R)
    raise ValueError(f'Unknown param {param_name!r}')


def _apply_param(param_name, value, sim_params):
    rp = sim_params.recomb_params
    if param_name == 'velocity_cm_us':         return sim_params._replace(velocity_cm_us=value)
    if param_name == 'lifetime_us':            return sim_params._replace(lifetime_us=value)
    if param_name == 'diffusion_trans_cm2_us': return sim_params._replace(diffusion_trans_cm2_us=value)
    if param_name == 'diffusion_long_cm2_us':  return sim_params._replace(diffusion_long_cm2_us=value)
    if param_name == 'recomb_alpha':           return sim_params._replace(recomb_params=rp._replace(alpha=value))
    if param_name == 'recomb_beta':            return sim_params._replace(recomb_params=rp._replace(beta=value))
    if param_name == 'recomb_beta_90':         return sim_params._replace(recomb_params=rp._replace(beta_90=value))
    if param_name == 'recomb_R':               return sim_params._replace(recomb_params=rp._replace(R=value))
    raise ValueError(f'Unknown param {param_name!r}')


def make_nparam_setter(param_names, gt_params, recomb_model):
    """Return (setter, gt_vals, scales, p_n_gts) for any number of params.

    setter(p_n_vec) -> SimParams  where p_n_vec[i] = physical_val[i] / scales[i]
    """
    scales  = [TYPICAL_SCALES[n] for n in param_names]
    gt_vals = [_get_gt_val(n, gt_params, recomb_model) for n in param_names]
    # q[i] = log(physical[i] / scale[i]);  physical[i] = exp(q[i]) * scale[i]
    p_n_gts = [float(np.log(v / s)) for v, s in zip(gt_vals, scales)]

    def setter(p_n_vec):
        params = gt_params
        for i, name in enumerate(param_names):
            params = _apply_param(name, jnp.exp(p_n_vec[i]) * scales[i], params)
        return params

    return setter, gt_vals, scales, p_n_gts
