"""
Differentiable (MLP-based) electric-field distortions for LArTPC simulation.

This is the *differentiable* counterpart to ``tools/efield_distortions.py``.
There, the distortion maps are produced once, offline, by a non-differentiable
analytic + SciPy-Euler pipeline.  Here, the distortion is represented by a small
MLP whose parameters can be fit by gradient descent, and whose output E-field is
differentiable both w.r.t. the query position (for use inside the JAX sim) and
w.r.t. the network parameters (for calibration).

Why "nonlocal": a grid map only knows the field at sampled pixels and interpolates
locally; an MLP is a single global function of position, so a deposit anywhere in
the volume sees a smooth field determined by *all* the training data, not just the
nearest grid cells.

Three parameterizations (``mode``)
----------------------------------
- ``"potential"`` (default, recommended) — the MLP outputs a scalar distortion
  potential ``δφ_θ(x)`` and the distortion field is ``E_dist = -∇δφ_θ``.  This is
  **conservative (curl-free) by construction**, which is the physically correct
  property of an electrostatic field.  The total field is ``E = E_bg + E_dist``,
  where ``E_bg`` is the (fixed) nominal uniform drift field.
- ``"efield"`` — the MLP outputs the 3-vector distortion directly,
  ``E = E_bg + MLP_θ(x)``.  More expressive, but not guaranteed conservative.
- ``"correction"`` — the MLP outputs the per-position drift correction
  ``Δ(x) = [Δx, Δy, Δz]`` directly (analogue of the SCE drift-correction map).

API parallels ``tools/efield_distortions.py``
---------------------------------------------
- ``make_field_fn(params, cfg) -> fn(positions_cm) -> (N, 3)`` is the drop-in
  analogue of ``create_single_interpolation_fn`` — JIT/vmap/grad-friendly.
- ``fit_to_map(...)`` fits an MLP to a target grid map (e.g. the ``.npz``
  produced by ``tools/efield_distortions.py``), demonstrating the differentiable
  determination of the distortions.

Positions are in cm throughout; fields in V/cm; corrections in cm.
"""

import os
from dataclasses import dataclass, field
from typing import Sequence, Tuple, List

import numpy as np
import jax
import jax.numpy as jnp


# =========================================================================
# MLP primitives (functional / pytree params)
# =========================================================================

Params = List[Tuple[jnp.ndarray, jnp.ndarray]]  # list of (W, b)


@dataclass
class FieldConfig:
    """Static configuration for an MLP field model (not a pytree)."""
    mode: str = "potential"              # "potential" | "efield" | "correction"
    center_cm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_cm: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    bg_field_Vcm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    out_scale: float = 1.0               # multiplies the raw MLP output
    hidden: Tuple[int, ...] = (64, 64, 64)

    @property
    def out_dim(self) -> int:
        return 1 if self.mode == "potential" else 3


def init_params(key, cfg: FieldConfig, in_dim: int = 3) -> Params:
    """Glorot-initialised weights for a tanh MLP with ``cfg.out_dim`` outputs."""
    sizes = [in_dim, *cfg.hidden, cfg.out_dim]
    params: Params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (n_in, n_out) in zip(keys, zip(sizes[:-1], sizes[1:])):
        limit = jnp.sqrt(6.0 / (n_in + n_out))           # Glorot uniform
        W = jax.random.uniform(k, (n_in, n_out), minval=-limit, maxval=limit)
        b = jnp.zeros((n_out,))
        params.append((W, b))
    return params


def _mlp(params: Params, h: jnp.ndarray) -> jnp.ndarray:
    """Forward pass for a single input vector ``h`` (shape (in_dim,))."""
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return h @ W + b                                     # (out_dim,)


# =========================================================================
# Field evaluation (single point); compose with vmap/grad/jit externally
# =========================================================================

def _normalize(pos_cm, cfg: FieldConfig):
    center = jnp.asarray(cfg.center_cm)
    half = jnp.asarray(cfg.half_cm)
    return (pos_cm - center) / half


def potential(params: Params, pos_cm, cfg: FieldConfig):
    """Scalar distortion potential δφ_θ(x) at a single point (mode='potential')."""
    h = _normalize(pos_cm, cfg)
    return cfg.out_scale * _mlp(params, h)[0]


def _distortion_point(params: Params, pos_cm, cfg: FieldConfig):
    """Distortion 3-vector at a single point (no background added)."""
    if cfg.mode == "potential":
        # Conservative field: E_dist = -∇δφ.  grad w.r.t. physical position;
        # the chain rule carries the 1/half normalization automatically.
        return -jax.grad(potential, argnums=1)(params, pos_cm, cfg)
    # "efield" / "correction": MLP outputs the 3-vector directly.
    return cfg.out_scale * _mlp(params, _normalize(pos_cm, cfg))


def field_point(params: Params, pos_cm, cfg: FieldConfig):
    """Total field/correction at a single point (background added for E-field modes)."""
    out = _distortion_point(params, pos_cm, cfg)
    if cfg.mode == "correction":
        return out                                       # corrections have no background
    return jnp.asarray(cfg.bg_field_Vcm) + out


def make_field_fn(params: Params, cfg: FieldConfig):
    """
    Build a JIT/vmap/grad-friendly batched evaluator.

    Returns
    -------
    fn : callable
        ``fn(positions_cm) -> (N, 3)``.  Drop-in analogue of
        ``efield_distortions.create_single_interpolation_fn``.  Returns the
        total E-field (V/cm) for ``mode in {"potential","efield"}`` or the
        drift correction (cm) for ``mode == "correction"``.
    """
    def fn(positions_cm):
        positions_cm = jnp.atleast_2d(jnp.asarray(positions_cm))
        return jax.vmap(lambda p: field_point(params, p, cfg))(positions_cm)
    return fn


def make_distortion_fn(params: Params, cfg: FieldConfig):
    """Like ``make_field_fn`` but returns only the distortion (background removed)."""
    def fn(positions_cm):
        positions_cm = jnp.atleast_2d(jnp.asarray(positions_cm))
        return jax.vmap(lambda p: _distortion_point(params, p, cfg))(positions_cm)
    return fn


# =========================================================================
# Simulator integration helpers
# =========================================================================
#
# These let an MLP field model live inside ``SimParams.sce_models`` and be
# carried by the calibration optimizer as a flat block of weights, while the
# static ``FieldConfig`` is captured outside the differentiable pytree.

def flatten_params(params: Params):
    """Flatten an MLP param pytree to a 1-D array + an unflatten closure.

    Thin wrapper over ``jax.flatten_util.ravel_pytree``.  Returns
    ``(flat, unravel)`` where ``unravel(flat) -> params``.
    """
    from jax.flatten_util import ravel_pytree
    flat, unravel = ravel_pytree(params)
    return flat, unravel


def unflatten_params(flat, unravel) -> Params:
    """Rebuild an MLP param pytree from a flat array using ``unravel``."""
    return unravel(flat)


def zero_params(cfg: FieldConfig, in_dim: int = 3) -> Params:
    """Same-shaped param pytree with all weights/biases zero.

    Output is identically zero (nominal field), but **all gradients vanish** too
    (every backward path multiplies by a zero weight/activation), so an all-zero
    MLP cannot be trained.  Use it only for warm-up / structure; for a trainable
    nominal start use :func:`nominal_start_params`.
    """
    template = init_params(jax.random.PRNGKey(0), cfg, in_dim=in_dim)
    return [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in template]


def nominal_start_params(cfg: FieldConfig, key, in_dim: int = 3) -> Params:
    """Trainable nominal start: random hidden layers, zeroed **output** layer.

    The output layer is zero, so the MLP output is identically zero at the start
    (``E = E_bg``, no distortion / zero correction) — but because the hidden
    layers carry random weights, the gradient w.r.t. the output-layer weights is
    nonzero, so optimization can move off the nominal field.  This avoids the
    dead-network trap of :func:`zero_params`.
    """
    params = init_params(key, cfg, in_dim=in_dim)
    W, b = params[-1]
    params[-1] = (jnp.zeros_like(W), jnp.zeros_like(b))
    return params


def sce_outputs(params: Params, positions_cm, cfg: FieldConfig, nominal_field_Vcm):
    """Compute ``(efield_correction, drift_corr_cm)`` for the simulator's SCE hook.

    Matches ``tools.config.SCEOutputs`` semantics:

    - ``potential`` / ``efield`` modes: the MLP drives the E-field.
      ``efield_correction = E_total / |E_nominal|`` (dimensionless, like the
      grid-map SCE path) and ``drift_corr_cm = 0``.
    - ``correction`` mode: the MLP drives the drift correction directly.
      ``efield_correction = [1, 0, 0]`` (nominal) and ``drift_corr_cm`` is the
      MLP output (cm).

    Parameters
    ----------
    params : Params
        MLP weights (typically ``sim_params.sce_models``).
    positions_cm : (N, 3)
        Query positions in the volume-local frame (as passed to the grid-map
        SCE functions from ``load_sce_per_volume``).
    cfg : FieldConfig
        Static field config (mode, normalization, background field).
    nominal_field_Vcm : float
        Nominal |E| used to normalize the E-field into a correction factor.

    Returns
    -------
    (efield_correction, drift_corr_cm) : each (N, 3) jnp.ndarray
    """
    positions_cm = jnp.atleast_2d(jnp.asarray(positions_cm))
    N = positions_cm.shape[0]
    if cfg.mode == "correction":
        drift = jax.vmap(lambda p: field_point(params, p, cfg))(positions_cm)
        efield_corr = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), (N, 3))
        return efield_corr, drift
    # potential / efield: MLP drives the E-field; correction = E_total / nominal.
    E_total = jax.vmap(lambda p: field_point(params, p, cfg))(positions_cm)
    efield_corr = E_total / nominal_field_Vcm
    drift = jnp.zeros((N, 3))
    return efield_corr, drift


# =========================================================================
# Fitting an MLP to a target grid map (the differentiable "determination")
# =========================================================================

def _grid_positions(origin_cm, spacing_cm, shape_xyz):
    """(N, 3) physical coordinates of every grid point, C-order over (x,y,z)."""
    nx, ny, nz = shape_xyz
    gx = origin_cm[0] + np.arange(nx) * spacing_cm[0]
    gy = origin_cm[1] + np.arange(ny) * spacing_cm[1]
    gz = origin_cm[2] + np.arange(nz) * spacing_cm[2]
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
    return np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=-1)


def cfg_from_grid(origin_cm, spacing_cm, shape_xyz, mode="potential",
                  bg_field_Vcm=(0.0, 0.0, 0.0), hidden=(64, 64, 64),
                  out_scale=None, target=None):
    """
    Build a ``FieldConfig`` whose normalization matches a grid map's extent.

    ``center_cm``/``half_cm`` are derived from the grid corners so inputs land in
    ~[-1, 1].  If ``out_scale`` is None it is auto-set to the RMS magnitude of the
    (background-subtracted) target, which keeps the initial output well-scaled.
    """
    pos = _grid_positions(origin_cm, spacing_cm, shape_xyz)
    lo, hi = pos.min(0), pos.max(0)
    center = (hi + lo) / 2.0
    half = np.maximum((hi - lo) / 2.0, 1e-6)

    if out_scale is None:
        if target is not None and mode != "correction":
            resid = np.asarray(target).reshape(-1, 3) - np.asarray(bg_field_Vcm)
            out_scale = float(np.sqrt(np.mean(resid ** 2)) + 1e-9)
        elif target is not None:
            out_scale = float(np.sqrt(np.mean(np.asarray(target) ** 2)) + 1e-9)
        else:
            out_scale = 1.0
        if mode == "potential":
            # φ has units of (field × length); rescale so -∇φ ~ target magnitude.
            out_scale *= float(np.mean(half))

    return FieldConfig(
        mode=mode, center_cm=tuple(center), half_cm=tuple(half),
        bg_field_Vcm=tuple(bg_field_Vcm), out_scale=out_scale, hidden=tuple(hidden),
    )


def fit_to_map(target, origin_cm, spacing_cm, *, mode="potential",
               bg_field_Vcm=(0.0, 0.0, 0.0), hidden=(64, 64, 64),
               epochs=2000, lr=1e-3, seed=0, batch_size=None, verbose=True):
    """
    Fit an MLP field model to a target grid map by gradient descent.

    Parameters
    ----------
    target : array (Nx, Ny, Nz, 3)
        Target field (V/cm) for ``mode in {"potential","efield"}`` or target
        drift correction (cm) for ``mode == "correction"``.
    origin_cm, spacing_cm : array (3,)
        Grid metadata (same convention as ``efield_distortions``).
    mode : str
        See module docstring.
    bg_field_Vcm : tuple
        Nominal uniform field subtracted before fitting the distortion
        (ignored for ``mode == "correction"``).
    epochs, lr, seed, batch_size : training hyperparameters.

    Returns
    -------
    params : Params
    cfg : FieldConfig
    history : list of float   (MSE per epoch)
    """
    import optax

    target = np.asarray(target, dtype=np.float64)
    shape_xyz = target.shape[:3]
    positions = jnp.asarray(_grid_positions(origin_cm, spacing_cm, shape_xyz))
    targets = jnp.asarray(target.reshape(-1, 3))

    cfg = cfg_from_grid(origin_cm, spacing_cm, shape_xyz, mode=mode,
                        bg_field_Vcm=bg_field_Vcm, hidden=hidden, target=target)

    key = jax.random.PRNGKey(seed)
    key, ik = jax.random.split(key)
    params = init_params(ik, cfg)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(params, pos, tgt):
        pred = jax.vmap(lambda p: field_point(params, p, cfg))(pos)
        return jnp.mean((pred - tgt) ** 2)

    @jax.jit
    def step(params, opt_state, pos, tgt):
        val, grads = jax.value_and_grad(loss_fn)(params, pos, tgt)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, val

    n = positions.shape[0]
    history: List[float] = []
    for epoch in range(epochs):
        if batch_size is None or batch_size >= n:
            params, opt_state, val = step(params, opt_state, positions, targets)
        else:
            key, sk = jax.random.split(key)
            idx = jax.random.choice(sk, n, (batch_size,), replace=False)
            params, opt_state, val = step(params, opt_state, positions[idx], targets[idx])
        history.append(float(val))
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            rms = float(jnp.sqrt(jnp.mean(targets ** 2))) + 1e-12
            print(f"  epoch {epoch:5d}  mse={val:.6e}  rel_rms={np.sqrt(val)/rms:.4%}")

    return params, cfg, history


# =========================================================================
# (De)serialisation
# =========================================================================

def save_model(path, params: Params, cfg: FieldConfig, side: str = ""):
    """Save MLP params + config to an .npz under a ``side`` prefix (e.g. 'east')."""
    flat = {}
    p = f"{side}_" if side else ""
    flat[f"{p}n_layers"] = np.array(len(params))
    for i, (W, b) in enumerate(params):
        flat[f"{p}W{i}"] = np.asarray(W)
        flat[f"{p}b{i}"] = np.asarray(b)
    flat[f"{p}mode"] = np.array(cfg.mode)
    flat[f"{p}center_cm"] = np.asarray(cfg.center_cm)
    flat[f"{p}half_cm"] = np.asarray(cfg.half_cm)
    flat[f"{p}bg_field_Vcm"] = np.asarray(cfg.bg_field_Vcm)
    flat[f"{p}out_scale"] = np.array(cfg.out_scale)
    flat[f"{p}hidden"] = np.asarray(cfg.hidden)
    return flat


def load_model(npz, side: str = "") -> Tuple[Params, FieldConfig]:
    """Inverse of ``save_model`` for one ``side`` prefix."""
    p = f"{side}_" if side else ""
    n = int(npz[f"{p}n_layers"])
    params = [(jnp.asarray(npz[f"{p}W{i}"]), jnp.asarray(npz[f"{p}b{i}"]))
              for i in range(n)]
    cfg = FieldConfig(
        mode=str(npz[f"{p}mode"]),
        center_cm=tuple(npz[f"{p}center_cm"].tolist()),
        half_cm=tuple(npz[f"{p}half_cm"].tolist()),
        bg_field_Vcm=tuple(npz[f"{p}bg_field_Vcm"].tolist()),
        out_scale=float(npz[f"{p}out_scale"]),
        hidden=tuple(int(x) for x in npz[f"{p}hidden"].tolist()),
    )
    return params, cfg


# =========================================================================
# CLI: fit MLPs to the toy SCE maps from tools/efield_distortions.py
# =========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit a differentiable MLP E-field distortion to a target SCE .npz "
                    "(produced by tools/efield_distortions.py).")
    parser.add_argument("--input", required=True,
                        help="Input SCE .npz (east_efield/west_efield/... keys).")
    parser.add_argument("--output", required=True, help="Output .npz with fitted MLP params.")
    parser.add_argument("--mode", default="potential",
                        choices=["potential", "efield", "correction"],
                        help="What the MLP learns. 'potential' => conservative E=-grad(phi).")
    parser.add_argument("--target", default="efield", choices=["efield", "correction"],
                        help="Which map to fit (efield modes fit *_efield; "
                             "correction mode always fits *_corrections).")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.input)
    fit_target = "correction" if args.mode == "correction" else args.target

    out_flat = {}
    for side, anode_sign in (("east", +1.0), ("west", -1.0)):
        # Nominal uniform drift field: +E0 x on east, -E0 x on west.  Inferred
        # from the target map's mean Ex so we don't need the config here.
        ef = data[f"{side}_efield"]
        e0 = float(np.mean(ef[..., 0]))                  # ~ +E0 (east) / -E0 (west)
        bg = (e0, 0.0, 0.0) if fit_target == "efield" else (0.0, 0.0, 0.0)

        if fit_target == "efield":
            tgt = data[f"{side}_efield"]
        else:
            tgt = data[f"{side}_corrections"]

        print(f"[nonlocal_efield] fitting {side} side  mode={args.mode}  "
              f"target={fit_target}  bg={bg}")
        params, cfg, _ = fit_to_map(
            tgt, data[f"{side}_origin"], data[f"{side}_spacing"],
            mode=args.mode, bg_field_Vcm=bg, hidden=tuple(args.hidden),
            epochs=args.epochs, lr=args.lr, seed=args.seed)

        out_flat.update(save_model(None, params, cfg, side=side))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, **out_flat)
    print(f"[nonlocal_efield] saved fitted MLP params → {args.output}")
