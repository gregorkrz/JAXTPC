"""Parameter helpers: GT value extraction, SimParams setters, Efield config.

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


def build_siren_config(sim, gt_params, hidden=(64, 64, 64), omega_0=5.0, T=89.0):
    """Build a ``SirenDistortionConfig`` from simulator geometry and GT params.

    The SIREN operates in volume-local coordinates (anode at x=0, x increasing
    toward cathode, yz centered).  Normalization maps the local volume to [-1,1]³
    so that the anode BC factor ``(x_norm+1)=0`` at x=0 is automatic.
    """
    from tools.sce_siren import drift_velocity_jax, build_vinv_table
    from tools.distortion import SirenDistortionConfig
    vol0 = sim._sim_config.volumes[0]
    E0 = float(gt_params.recomb_params.field_strength_Vcm)
    v0 = float(drift_velocity_jax(E0, T=T))
    # Local frame: x ∈ [0, max_drift_cm]; yz centered at 0
    half_x = float(vol0.max_drift_cm) / 2.0
    (_, _), (ylo, yhi), (zlo, zhi) = vol0.ranges_cm
    half_y = abs(yhi - ylo) / 2.0
    half_z = abs(zhi - zlo) / 2.0
    norm_offsets = jnp.array([half_x, 0.0, 0.0])
    norm_scales  = jnp.array([max(half_x, 1e-6), max(half_y, 1e-6), max(half_z, 1e-6)])
    v_table, E_table = build_vinv_table(T=T)
    return SirenDistortionConfig(
        omega_0=omega_0,
        norm_offsets=norm_offsets,
        norm_scales=norm_scales,
        E0=E0,
        v0=v0,
        v_table=v_table,
        E_table=E_table,
        hidden_features=int(hidden[0]) if hidden else 64,
        hidden_layers=len(hidden),
    )


def build_efield_config(sim, gt_params, mode='potential', hidden=(64, 64, 64)):
    """Build a local-frame ``FieldConfig`` for the MLP SCE model.

    The simulator works in volume-local coordinates (anode at x_local=0, drift
    along +x, yz centered).  In that frame the nominal field is +E0 along x for
    both volumes, so a single MLP serves both.  Weights are zero-initialised
    elsewhere, so ``out_scale`` only sets the natural output magnitude (combined
    with the LR multiplier it controls learning speed), not the starting point.
    """
    from tools.nonlocal_efield import FieldConfig
    vol0 = sim._sim_config.volumes[0]
    E0 = float(gt_params.recomb_params.field_strength_Vcm)
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = vol0.ranges_cm
    half_x = float(vol0.max_drift_cm) / 2.0          # local x ∈ [0, max_drift]
    half_y = abs(yhi - ylo) / 2.0
    half_z = abs(zhi - zlo) / 2.0
    center = (half_x, 0.0, 0.0)
    half = (max(half_x, 1e-6), max(half_y, 1e-6), max(half_z, 1e-6))
    mean_half = (half[0] + half[1] + half[2]) / 3.0
    if mode == 'potential':
        out_scale = E0 * 0.05 * mean_half            # φ ~ field × length
    elif mode == 'efield':
        out_scale = E0 * 0.05                         # distortion ~ few % of E0
    else:  # correction (cm)
        out_scale = 1.0
    return FieldConfig(
        mode=mode, center_cm=center, half_cm=half,
        bg_field_Vcm=(E0, 0.0, 0.0), out_scale=out_scale, hidden=tuple(hidden),
    )
