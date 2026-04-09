"""
Loss landscape utilities for JAXTPC calibration experiments.

Provides helpers to:
  - Convert simulation outputs (response signals or truth hits) to dense 2D arrays
  - Precompute spectral weights for loss functions
  - Sweep a single SimParams scalar and record the loss at each value

IMPORTANT — two forward paths, different parameter sensitivity
--------------------------------------------------------------
DetectorSimulator has two execution modes:

  process_event()  — production path.  Uses a *static* pre-baked DKernel
                     (computed once at construction from the default config).
                     sim_params.diffusion_trans/long have NO EFFECT on the
                     response signals here.  Sensitive to: velocity, lifetime,
                     recomb_params.

  forward()        — differentiable path (for jax.grad).  Recomputes DKernel
                     from sim_params.diffusion_trans/long every call.
                     Sensitive to ALL SimParams fields including diffusion.

sweep_parameter() uses:
  mode='response'    → simulator.forward()      (diffusion-sensitive)
  mode='truth_hits'  → simulator.process_event() + finalize_track_hits()
                       (NOT diffusion-sensitive — truth hits use a static
                       diffusion config baked into VolumeGeometry)

Typical usage::

    from tools.landscape import make_weights, sweep_parameter

    weights, keys = make_weights(cfg, weight_type='sobolev')
    losses = sweep_parameter(
        simulator, deposits, base_params,
        param_name='diffusion_trans_cm2_us',
        values=np.linspace(D_min, D_max, 15),
        mode='response',          # 'response' uses forward(); 'truth_hits' uses process_event()
        weights=weights,
        cfg=cfg,
        weight_type='sobolev',
    )
"""

import numpy as np
import jax.numpy as jnp

from tools.output import to_dense
from tools.losses import (
    make_spectral_weight, make_sobolev_weight,
    blur_mse_loss_single, sobolev_loss_single,
    DEFAULT_BLUR_SIGMAS,
)


# ---------------------------------------------------------------------------
# Signal → dense 2D array conversion
# ---------------------------------------------------------------------------

def truth_hits_to_arrays(track_hits, cfg):
    """Convert finalized track_hits to dense (num_wires, num_time) arrays.

    Parameters
    ----------
    track_hits : dict
        Output of simulator.finalize_track_hits().  Keyed by (vi, pi).
        NOTE: finalize_track_hits() pops 'group_to_track' — pass the
        already-finalized dict, not the raw one.
    cfg : SimConfig

    Returns
    -------
    dict : (vol_idx, plane_idx) -> jnp.ndarray, shape (num_wires, num_time)
    """
    arrays = {}
    num_time = cfg.num_time_steps
    for (vi, pi), data in track_hits.items():
        num_wires = cfg.volumes[vi].num_wires[pi]
        dense = np.zeros((num_wires, num_time), dtype=np.float32)
        nh = int(data['num_hits'])
        if nh > 0:
            hbt = np.array(data['hits_by_track'][:nh])
            wires  = hbt[:, 0].astype(int)
            times  = hbt[:, 1].astype(int)
            charges = hbt[:, 2].astype(np.float32)
            valid = (wires >= 0) & (wires < num_wires) & (times >= 0) & (times < num_time)
            np.add.at(dense, (wires[valid], times[valid]), charges[valid])
        arrays[(vi, pi)] = jnp.array(dense)
    return arrays


def response_to_arrays(response_signals, cfg):
    """Convert response_signals dict to dense (num_wires, num_time) arrays.

    Parameters
    ----------
    response_signals : dict
        Output of simulator.process_event(). Keyed by (vi, pi).
    cfg : SimConfig

    Returns
    -------
    dict : (vol_idx, plane_idx) -> jnp.ndarray, shape (num_wires, num_time)
    """
    dense_dict = to_dense(response_signals, cfg)
    return {k: jnp.array(v.astype(np.float32)) for k, v in dense_dict.items()}


def simulate_plane_arrays(simulator, deposits, key):
    """Run the simulation and return truth and detector hit arrays per wire plane.

    Parameters
    ----------
    simulator : DetectorSimulator
    deposits : DepositData
    key : jax.random.PRNGKey

    Returns
    -------
    truth : dict, plane_name -> jnp.ndarray shape (num_time, num_wires)
        Charge from finalized track hits (truth signal).
    detector : dict, plane_name -> jnp.ndarray shape (num_time, num_wires)
        Charge from detector response signals.
    """
    cfg = simulator.config
    response_signals, track_hits_raw, deposits = simulator.process_event(deposits, key=key)
    track_hits = simulator.finalize_track_hits(track_hits_raw)

    truth_arrays = truth_hits_to_arrays(track_hits, cfg)
    detector_arrays = response_to_arrays(response_signals, cfg)

    truth, detector = {}, {}
    for (vi, pi) in truth_arrays:
        name = cfg.plane_names[vi][pi]
        t = truth_arrays[(vi, pi)].T      # (num_time, num_wires)
        d = detector_arrays[(vi, pi)].T
        if name in truth:
            truth[name] = truth[name] + t
            detector[name] = detector[name] + d
        else:
            truth[name] = t
            detector[name] = d
    return truth, detector


def arrays_to_tuple(arrays_dict):
    """Sort the plane dict into a deterministic tuple for loss functions.

    Returns
    -------
    arrays_tuple : tuple of jnp.ndarray
    keys : list of (vol_idx, plane_idx) in the same order
    """
    keys = sorted(arrays_dict.keys())
    return tuple(arrays_dict[k] for k in keys), keys


# ---------------------------------------------------------------------------
# Spectral weight precomputation
# ---------------------------------------------------------------------------

def make_weights(cfg, weight_type='sobolev'):
    """Precompute spectral weights for all wire planes.

    Parameters
    ----------
    cfg : SimConfig
    weight_type : 'sobolev' | 'blur_mse'

    Returns
    -------
    weights : tuple of jnp.ndarray  (one per plane, same order as arrays_to_tuple)
    keys    : list of (vol_idx, plane_idx)
    """
    n_vols = cfg.n_volumes
    n_planes = cfg.volumes[0].n_planes
    keys = sorted((vi, pi) for vi in range(n_vols) for pi in range(n_planes))

    weights = []
    for (vi, pi) in keys:
        H = cfg.volumes[vi].num_wires[pi]
        W = cfg.num_time_steps
        if weight_type == 'sobolev':
            w = make_sobolev_weight(H, W)
        else:
            w = make_spectral_weight(H, W, DEFAULT_BLUR_SIGMAS)
        weights.append(w)
    return tuple(weights), keys


# ---------------------------------------------------------------------------
# Loss between two sets of plane arrays
# ---------------------------------------------------------------------------

def compute_loss(arrays_a, arrays_b, weights, weight_type='sobolev'):
    """Sum spectral loss over all planes.

    Parameters
    ----------
    arrays_a, arrays_b : tuples of jnp.ndarray (same order)
    weights : tuple of spectral weight arrays (from make_weights)
    weight_type : 'sobolev' | 'blur_mse'

    Returns
    -------
    float scalar loss
    """
    fn = sobolev_loss_single if weight_type == 'sobolev' else blur_mse_loss_single
    total = 0.0
    for a, b, w in zip(arrays_a, arrays_b, weights):
        total = total + float(fn(a, b, w))
    return total


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def sweep_parameter(simulator, deposits, base_params, param_name, values,
                    mode, weights, cfg, weight_type='sobolev', verbose=True):
    """Sweep one top-level SimParams scalar and record loss at each value.

    The ground truth is base_params itself.  For each value in `values`,
    a perturbed SimParams is built with base_params._replace(**{param_name: v})
    and the full forward simulation is run.

    Parameters
    ----------
    simulator : DetectorSimulator
    deposits : DepositData
    base_params : SimParams
        Ground-truth parameters (defines the target signal).
    param_name : str
        A top-level scalar field of SimParams, e.g. 'diffusion_trans_cm2_us'.
    values : array-like of float
        Parameter values to evaluate.
    mode : 'response' | 'truth_hits'
        Which signals to use for the loss computation.
    weights : tuple of jnp.ndarray
        Spectral weights from make_weights().
    cfg : SimConfig
    weight_type : 'sobolev' | 'blur_mse'
    verbose : bool

    Returns
    -------
    losses : np.ndarray, shape (len(values),)
    """
    values = np.asarray(values, dtype=np.float64)

    if mode == 'response':
        # forward() requires a simulator built with n_segments set.
        # _forward_diff is only compiled when n_segments is not None.
        if not hasattr(simulator, '_forward_diff'):
            raise RuntimeError(
                "mode='response' requires a simulator built with n_segments set "
                "(e.g. DetectorSimulator(..., n_segments=total_pad)).  "
                "The default simulator (n_segments=None) does not build the "
                "differentiable path and forward() will not work."
            )
        run_gt  = lambda p: simulator.forward(p, deposits)
        run_sim = lambda p: simulator.forward(p, deposits)
    elif mode == 'truth_hits':
        def run_gt(p):
            _, raw, _ = simulator.process_event(deposits, sim_params=p)
            hits = simulator.finalize_track_hits(raw)
            arr, _ = arrays_to_tuple(truth_hits_to_arrays(hits, cfg))
            return arr
        run_sim = run_gt
    else:
        raise ValueError(f"mode must be 'response' or 'truth_hits', got {mode!r}")

    # --- ground truth ---
    if verbose:
        print(f"Running ground truth ({param_name}={float(getattr(base_params, param_name)):.4g})...")
    gt_arrays = run_gt(base_params)

    # --- sweep ---
    losses = []
    for i, v in enumerate(values):
        if verbose:
            print(f"  [{i+1:2d}/{len(values)}] {param_name} = {v:.4g}", end="  ", flush=True)

        perturbed = base_params._replace(**{param_name: jnp.array(float(v))})
        arrays = run_sim(perturbed)

        loss = compute_loss(arrays, gt_arrays, weights, weight_type)
        losses.append(loss)
        if verbose:
            print(f"loss = {loss:.4e}")

    return np.array(losses)