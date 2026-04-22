#!/usr/bin/env python
"""
Run the lifetime/velocity joint optimization and save results to a pickle.

Usage:
    python run_opt.py [--output-dir results] [--n-tries 10] [--seed 42]

The pickle is saved as <output-dir>/optimization_results.pkl.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

import argparse
import gc
import os

_RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')
_PLOTS_DIR   = os.environ.get('PLOTS_DIR',   'plots')
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp

from tools.simulation import DetectorSimulator
from tools.geometry import generate_detector
from tools.loader import load_event
from tools.losses import sobolev_loss_single, make_sobolev_weight

# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(description='Run lifetime/velocity optimization')
parser.add_argument('--output-dir', default=_RESULTS_DIR,
                    help='Directory to save results (default: $RESULTS_DIR or results)')
parser.add_argument('--output-file', default='optimization_results.pkl',
                    help='Filename for the pickle (default: optimization_results.pkl)')
parser.add_argument('--n-tries', type=int, default=10,
                    help='Number of random starting points (default: 10)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for starting points (default: 42)')
args = parser.parse_args()

OUTPUT_DIR  = args.output_dir
OUTPUT_FILE = args.output_file
N_TRIES     = args.n_tries
SEED        = args.seed

# =============================================================================
# CONFIGURATION
# =============================================================================

# Ground-truth parameter values
GT_LIFETIME_US    = 10_000.0   # μs
GT_VELOCITY_CM_US = 0.160      # cm/μs

# Starting-point noise
NOISE_FRAC = 0.20   # 20% Gaussian σ around GT

# Optimisation
N_STEPS    = 200     # steps per run

# Learning rates  (normalised parameter space: both params ≈ 1 at GT)
LR_SGD    = 0.01
LR_SGDM   = 0.01
MOM_SGDM  = 0.9
LR_ADAM   = 0.05
LR_ADAMW  = 0.05
WD_ADAMW  = 0.002
LR_RMSPROP = 0.02
RHO_RMSPROP = 0.99
LR_NADAM  = 0.05
LR_YOGI   = 0.05
LR_LION   = 0.005

# Gradient clipping (in normalised space)
MAX_GRAD_NORM = 1.0

# Parameter bounds (normalised: 1.0 = GT value)
PARAM_LO_N = 0.1
PARAM_HI_N = 4.0

# Sobolev loss padding per side (screening length L = max_pad/2 pixels).
SOBOLEV_MAX_PAD = 128

# Simulator
CONFIG_PATH        = 'config/cubic_wireplane_config.yaml'
DATA_PATH          = 'muon.h5'
EVENT_IDX          = 0
MAX_ACTIVE_BUCKETS = 1000
N_SEGMENTS         = 10_000

print(f'JAX version: {jax.__version__}')
print(f'JAX devices: {jax.devices()}')
print(f'Output dir : {OUTPUT_DIR}')
print(f'N_TRIES    : {N_TRIES}   SEED: {SEED}')

# =============================================================================
# BUILD DIFFERENTIABLE SIMULATOR
# =============================================================================

detector_config = generate_detector(CONFIG_PATH)
jax.clear_caches()
gc.collect()

simulator = DetectorSimulator(
    detector_config,
    differentiable=True,
    n_segments=N_SEGMENTS,
    use_bucketed=True,
    max_active_buckets=MAX_ACTIVE_BUCKETS,
    include_noise=False,
    include_electronics=False,
    include_track_hits=False,
    include_digitize=False,
    track_config=None,
)
cfg = simulator.config

deposits = load_event(DATA_PATH, cfg, event_idx=EVENT_IDX)
n_total  = sum(v.n_actual for v in deposits.volumes)
print(f'Loaded {n_total:,} deposits  (n_segments={N_SEGMENTS})')

# =============================================================================
# WARM UP  (compiles production JIT — ~60-90 s)
# =============================================================================

print('Warming up...')
t0 = time.time()
simulator.warm_up()
print(f'Done ({time.time()-t0:.1f}s)')

# =============================================================================
# GROUND-TRUTH SIGNAL + LOSS FUNCTIONS
# =============================================================================

base_params = simulator.default_sim_params

gt_params = base_params._replace(
    lifetime_us    = jnp.array(GT_LIFETIME_US),
    velocity_cm_us = jnp.array(GT_VELOCITY_CM_US),
)

print('Computing GT forward pass...')
t0 = time.time()
gt_arrays = simulator.forward(gt_params, deposits)
jax.block_until_ready(gt_arrays)
print(f'Done ({time.time()-t0:.1f}s)  —  {len(gt_arrays)} plane arrays')

weights = tuple(
    make_sobolev_weight(arr.shape[0], arr.shape[1], max_pad=SOBOLEV_MAX_PAD)
    for arr in gt_arrays
)
print(f'Sobolev weights: max_pad={SOBOLEV_MAX_PAD}, '
      f'FFT sizes: {[w.shape for w in weights]}')

def unpack(params_n):
    return params_n[0] * GT_LIFETIME_US, params_n[1] * GT_VELOCITY_CM_US

def _sobolev_plane(pr, gt, w):
    return sobolev_loss_single(pr, gt, w)

def _mse_plane(pr, gt):
    norm = jnp.sum(jnp.abs(gt)) + 1e-12
    return jnp.mean(((pr - gt) / norm) ** 2)

def _l1_plane(pr, gt):
    norm = jnp.sum(jnp.abs(gt)) + 1e-12
    return jnp.mean(jnp.abs((pr - gt) / norm))

def make_loss_fn(loss_type):
    def loss_fn(params_n):
        lt, vel = unpack(params_n)
        p = gt_params._replace(lifetime_us=lt, velocity_cm_us=vel)
        pred = simulator.forward(p, deposits)
        if loss_type == 'Sobolev':
            return jnp.sum(jnp.array([_sobolev_plane(pr, gt, w)
                                       for pr, gt, w in zip(pred, gt_arrays, weights)]))
        elif loss_type == 'MSE':
            return jnp.sum(jnp.array([_mse_plane(pr, gt)
                                       for pr, gt in zip(pred, gt_arrays)]))
        else:  # L1
            return jnp.sum(jnp.array([_l1_plane(pr, gt)
                                       for pr, gt in zip(pred, gt_arrays)]))
    return loss_fn

print('\nLoss at GT (should be ~0):')
for ln in ('Sobolev', 'MSE', 'L1'):
    v = make_loss_fn(ln)(jnp.array([1.0, 1.0]))
    print(f'  {ln:7s}: {float(v):.3e}')

# =============================================================================
# OPTIMISERS  (pure JAX, no optax)
# =============================================================================

_B1, _B2, _EPS = 0.9, 0.999, 1e-8

def sgd_init(p):   return {}
def sgd_step(p, g, state): return p - LR_SGD * g, state

# SGD with momentum  m = β*m + g,  p = p - lr*m
def sgdm_init(p):  return dict(m=jnp.zeros(2))
def sgdm_step(p, g, state):
    m = MOM_SGDM * state['m'] + g
    return p - LR_SGDM * m, dict(m=m)

def adam_init(p):  return dict(m=jnp.zeros(2), v=jnp.zeros(2), t=0)
def adam_step(p, g, state, lr=LR_ADAM):
    t = state['t'] + 1
    m = _B1 * state['m'] + (1 - _B1) * g
    v = _B2 * state['v'] + (1 - _B2) * g ** 2
    m_hat = m / (1 - _B1 ** t)
    v_hat = v / (1 - _B2 ** t)
    return p - lr * m_hat / (jnp.sqrt(v_hat) + _EPS), dict(m=m, v=v, t=t)

def adamw_init(p): return dict(m=jnp.zeros(2), v=jnp.zeros(2), t=0)
def adamw_step(p, g, state):
    t = state['t'] + 1
    m = _B1 * state['m'] + (1 - _B1) * g
    v = _B2 * state['v'] + (1 - _B2) * g ** 2
    m_hat = m / (1 - _B1 ** t)
    v_hat = v / (1 - _B2 ** t)
    new_p = p - LR_ADAMW * (m_hat / (jnp.sqrt(v_hat) + _EPS) + WD_ADAMW * p)
    return new_p, dict(m=m, v=v, t=t)

# RMSProp  v = ρ*v + (1-ρ)*g²,  p = p - lr*g / (√v + ε)
def rmsprop_init(p): return dict(v=jnp.zeros(2))
def rmsprop_step(p, g, state):
    v = RHO_RMSPROP * state['v'] + (1 - RHO_RMSPROP) * g ** 2
    return p - LR_RMSPROP * g / (jnp.sqrt(v) + _EPS), dict(v=v)

# Nadam  (Adam + Nesterov lookahead on the moment estimate)
def nadam_init(p): return dict(m=jnp.zeros(2), v=jnp.zeros(2), t=0)
def nadam_step(p, g, state):
    t = state['t'] + 1
    m = _B1 * state['m'] + (1 - _B1) * g
    v = _B2 * state['v'] + (1 - _B2) * g ** 2
    # Nesterov: use next-step moment estimate
    m_hat = _B1 * m / (1 - _B1 ** (t + 1)) + (1 - _B1) * g / (1 - _B1 ** t)
    v_hat = v / (1 - _B2 ** t)
    return p - LR_NADAM * m_hat / (jnp.sqrt(v_hat) + _EPS), dict(m=m, v=v, t=t)

# Yogi  — like Adam but uses sign(g²-v)*g² for the variance update,
#          preventing over-inflation of learning rate early in training.
def yogi_init(p): return dict(m=jnp.zeros(2), v=jnp.zeros(2), t=0)
def yogi_step(p, g, state):
    t = state['t'] + 1
    m = _B1 * state['m'] + (1 - _B1) * g
    g2 = g ** 2
    v = state['v'] + (1 - _B2) * jnp.sign(g2 - state['v']) * g2
    m_hat = m / (1 - _B1 ** t)
    v_hat = v / (1 - _B2 ** t)
    return p - LR_YOGI * m_hat / (jnp.sqrt(v_hat) + _EPS), dict(m=m, v=v, t=t)

# Lion  — EvoLved Sign Momentum: very memory-efficient, no second moment.
#          update = sign(β1*m + (1-β1)*g),  m = β1*m + (1-β1)*g
_LION_B1, _LION_B2 = 0.9, 0.99
def lion_init(p):  return dict(m=jnp.zeros(2))
def lion_step(p, g, state):
    update = jnp.sign(_LION_B1 * state['m'] + (1 - _LION_B1) * g)
    m = _LION_B2 * state['m'] + (1 - _LION_B2) * g
    return p - LR_LION * update, dict(m=m)

OPTIMISERS = {
    # 'SGD':     (sgd_init,     sgd_step),
    'SGD+Mom': (sgdm_init,    sgdm_step),
    'RMSProp': (rmsprop_init, rmsprop_step),
    'Adam':    (adam_init,    adam_step),
    'AdamW':   (adamw_init,   adamw_step),
    'Nadam':   (nadam_init,   nadam_step),
    'Yogi':    (yogi_init,    yogi_step),
    'Lion':    (lion_init,    lion_step),
}
print('Optimisers ready:', list(OPTIMISERS))

# =============================================================================
# JIT-COMPILE GRADIENT FUNCTIONS  (one per loss type)
# =============================================================================

LOSS_NAMES = ['Sobolev', 'MSE', 'L1']

LOSS_AND_GRADS = {}
for ln in LOSS_NAMES:
    lag = jax.jit(jax.value_and_grad(make_loss_fn(ln)))
    print(f'Compiling {ln}...')
    t0 = time.time()
    _lv, _gv = lag(jnp.array([1.0, 1.0]))
    jax.block_until_ready((_lv, _gv))
    print(f'  Done ({time.time()-t0:.1f}s)  loss={float(_lv):.3e}  grad={np.array(_gv)}')
    LOSS_AND_GRADS[ln] = lag

# =============================================================================
# GENERATE STARTING POINTS
# =============================================================================

rng = np.random.RandomState(SEED)
starts_n = []

for _ in range(N_TRIES):
    lt_n  = 1.0 + rng.randn() * NOISE_FRAC
    vel_n = 1.0 + rng.randn() * NOISE_FRAC
    starts_n.append((float(lt_n), float(vel_n)))

print(f'\n{N_TRIES} starting points (normalised, GT = 1.0):')
for i, (lt, vel) in enumerate(starts_n):
    print(f'  {i+1:2d}: lifetime={lt*GT_LIFETIME_US:.0f} μs  '
          f'velocity={vel*GT_VELOCITY_CM_US:.4f} cm/μs')

# =============================================================================
# RUN OPTIMISATION  (N_TRIES starts × optimisers × loss functions)
# all_results[loss_name][opt_name] = list of trajectories
# Each trajectory entry: (lifetime_us, velocity_cm_us, loss, grad_mag, grad_dir_lt, grad_dir_vel, step_time_s)
# Step 0 and final evaluation have NaN for grad and timing fields.
# =============================================================================

all_results = {}

for loss_name, lag in LOSS_AND_GRADS.items():
    print(f'\n{"="*60}')
    print(f'Loss: {loss_name}')
    all_results[loss_name] = {}

    for opt_name, (init_fn, step_fn) in OPTIMISERS.items():
        print(f'\n  --- {opt_name} ---')
        trajs = []

        for i, (lt0_n, vel0_n) in enumerate(starts_n):
            p = jnp.array([lt0_n, vel0_n])
            state = init_fn(p)

            lt0, vel0 = unpack(p)
            _nan = float('nan')
            traj = [(float(lt0), float(vel0), _nan, _nan, _nan, _nan, _nan)]

            for s in range(N_STEPS):
                t_step = time.time()
                loss_val, grads = lag(p)
                jax.block_until_ready((loss_val, grads))
                step_time = time.time() - t_step

                gnorm = float(jnp.linalg.norm(grads))
                grad_unit = np.array(grads) / gnorm if gnorm > 0 else np.zeros(2)

                safe_gnorm = max(gnorm, 1e-12)
                grads_clipped = jnp.where(gnorm > MAX_GRAD_NORM,
                                          grads * (MAX_GRAD_NORM / safe_gnorm),
                                          grads)
                p, state = step_fn(p, grads_clipped, state)
                p = jnp.clip(p, PARAM_LO_N, PARAM_HI_N)

                lt, vel = unpack(p)
                traj.append((float(lt), float(vel), float(loss_val),
                             gnorm, float(grad_unit[0]), float(grad_unit[1]), step_time))

            loss_final, _ = lag(p)
            lt, vel = unpack(p)
            traj.append((float(lt), float(vel), float(loss_final), _nan, _nan, _nan, _nan))

            trajs.append(traj)
            step_times = [entry[6] for entry in traj[1:-1]]
            print(f'    try {i+1:2d}: ({lt0:.0f} μs, {vel0:.4f}) → '
                  f'({float(lt):.0f} μs, {float(vel):.4f})  '
                  f'loss={float(loss_final):.3e}  '
                  f'step={np.mean(step_times)*1e3:.0f}ms avg')

        all_results[loss_name][opt_name] = trajs

# =============================================================================
# SAVE RESULTS
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

save_data = dict(
    all_results=all_results,
    starts_n=starts_n,
    GT_LIFETIME_US=GT_LIFETIME_US,
    GT_VELOCITY_CM_US=GT_VELOCITY_CM_US,
    N_STEPS=N_STEPS,
    N_TRIES=N_TRIES,
    NOISE_FRAC=NOISE_FRAC,
)
with open(out_path, 'wb') as f:
    pickle.dump(save_data, f)
print(f'\nSaved → {out_path}')
