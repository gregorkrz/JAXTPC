# JAXTPC

## Overview

JAXTPC is a GPU-accelerated Time Projection Chamber (TPC) simulation framework built with JAX. It models the full detector response chain in liquid argon TPCs: charge recombination, electron drift, diffusion-convolved wire/pixel response, electronics shaping, noise injection, and ADC digitization. Supports arbitrary multi-volume detector geometries (SBND, MicroBooNE, ICARUS, DUNE ND-LAr, DUNE FD1) with both wire and pixel readout.

## Repository Structure

```
JAXTPC/
├── tools/                        # Core simulation package
│   ├── simulation.py             # DetectorSimulator class (scan/vmap volume iteration)
│   ├── config.py                 # Data types (SimParams, SimConfig, VolumeGeometry, ...)
│   ├── physics.py                # Physics pipeline (volume + plane computations)
│   ├── geometry.py               # YAML config parser → per-volume geometry
│   ├── kernels.py                # Diffusion kernel generation (spatial conv) + interpolation
│   ├── drift.py                  # Electron drift physics
│   ├── wires.py                  # Wire/pixel geometry, deposit preparation, accumulation
│   ├── recombination.py          # Charge recombination (Modified Box + EMB models)
│   ├── electronics.py            # RC-RC electronics response via sparse FFT
│   ├── noise.py                  # Intrinsic noise generation (MicroBooNE model)
│   ├── track_hits.py             # Group-based track labeling + Q_s fractions
│   ├── efield_distortions.py     # Space charge effects (SCE maps, trilinear interpolation)
│   ├── loader.py                 # HDF5 I/O, volume splitting, local coord transform
│   ├── output.py                 # Format conversion (dense ↔ sparse ↔ bucketed)
│   ├── visualization.py          # Wire signal / track label / diffused charge plotting
│   ├── particle_generator.py     # Differentiable muon track generation
│   ├── losses.py                 # Sobolev / spectral loss functions
│   ├── nonlocal_efield.py        # Differentiable MLP space-charge-effect model
│   ├── random_boundary_tracks.py # Boundary-track event generator (angle/position selection)
│   ├── landscape.py              # Loss-landscape sweep helpers (notebook-facing)
│   └── responses/                # Pre-computed wire response kernels (U/V/Y NPZ)
├── optlib/                       # Calibration optimizer building blocks
│   ├── constants.py              # Valid params/losses/optimizers, typical param scales
│   ├── parsing.py                # run_optimization.py CLI argument definitions
│   ├── params.py                 # SimParams getters/setters, Efield config builder
│   ├── optim.py                  # Adam/SGD/momentum_sgd schedules, Newton-step plumbing
│   ├── paths.py                  # Result-folder naming, atomic pickle I/O
│   └── wandb_utils.py            # Weights & Biases logging (optional)
├── run_optimization.py           # Gradient-based detector-parameter calibration driver
├── scripts/                      # Loss-landscape sweep pipeline
│   ├── 2d_loss_landscape.py      # 2D grid sweep over two SimParams scalars
│   ├── 1d_gradients.py           # 1D gradient/cutoff sweeps
│   ├── launch_2d_landscape_pairs.py  # Slurm launcher: one job per (track, param-pair)
│   └── combine_landscapes.py     # Merge landscape .pkl outputs
├── production/                   # Batch production pipeline
│   ├── run_batch.py              # Batch simulation with threaded save workers
│   ├── save.py                   # HDF5 save functions (resp/seg/corr)
│   ├── load.py                   # HDF5 load/decode functions
│   ├── view_production.ipynb     # Visualize production output
│   ├── README.md                 # Pipeline docs, CLI flags, output schema
│   └── DATA_FORMAT.md            # Output file schema documentation
├── profiler/                     # Production parameter optimization
│   ├── setup_production.py       # Auto-tune total_pad, chunks, max_keys
│   ├── find_optimal_pad.py       # Scan data for max deposits per volume
│   ├── find_optimal_chunks.py    # Find optimal chunk sizes
│   ├── find_optimal_max_keys.py  # Probe track-hits capacity
│   └── ...                       # Per-parameter tuning scripts
├── tests/                        # Pytest test suite
│   ├── test_pipeline.py          # End-to-end integration tests
│   ├── test_pipeline_forward.py  # Differentiable path tests
│   ├── test_simulation.py        # Simulator unit tests
│   ├── test_electronics.py       # Electronics/noise tests
│   └── ...                       # Per-module tests
├── config/                       # Detector configurations
│   ├── cubic_wireplane_config.yaml   # Default: dual-TPC, SBND-scale
│   ├── sbnd_config.yaml              # SBND
│   ├── microboone_config.yaml        # MicroBooNE
│   ├── icarus_config.yaml            # ICARUS (4 volumes)
│   ├── dune_ndlar_config.yaml        # DUNE ND-LAr (70 volumes)
│   ├── dune_fd1_config.yaml          # DUNE Far Detector
│   ├── pixel_cube_config.yaml        # Pixel readout test config
│   ├── noise_spectrum.npz            # Empirical noise spectral shape
│   └── sce_jaxtpc.h5                 # Space charge effect correction maps
└── run_simulation.ipynb          # Interactive single-event simulation notebook
```

## Installation

### Dependencies

- JAX (with GPU support recommended)
- NumPy
- Matplotlib
- H5py
- PyYAML

```bash
pip install jax[cuda] numpy matplotlib h5py pyyaml
```

For GPU support, follow the [JAX installation guide](https://github.com/google/jax#installation).

**Note:** Use `python3` (not `python`).

## Quick Start

### Interactive notebook

```bash
jupyter notebook run_simulation.ipynb
```

### Python API

```python
from tools.simulation import DetectorSimulator
from tools.geometry import generate_detector
from tools.loader import load_event
from tools.config import create_track_hits_config
import jax

# Load configuration and create simulator
detector_config = generate_detector('config/cubic_wireplane_config.yaml')
simulator = DetectorSimulator(
    detector_config,
    use_bucketed=True,
    include_track_hits=True,
    include_digitize=True,
)

# Load event (deposits are automatically transformed to local coordinates)
deposits = load_event('data.h5', simulator.config, event_idx=0)

# Run simulation
response_signals, track_hits_raw, deposits = simulator.process_event(
    deposits, key=jax.random.PRNGKey(42))

# Convert to sparse format
sparse = simulator.to_sparse(response_signals, threshold_enc=1200)

# Finalize track labels
track_hits = simulator.finalize_track_hits(track_hits_raw)
```

### Production batch

```bash
python3 production/run_batch.py --data events.h5 --events 1000 --bucketed --workers 2
python3 production/run_batch.py --data events.h5 --config config/dune_ndlar_config.yaml \
    --total-pad 70000 --response-chunk 10000 --bucketed
```

See `production/README.md` for pipeline details, CLI flags, and output schema.

### Differentiable path

```python
simulator = DetectorSimulator(detector_config, differentiable=True, n_segments=1000)
signals = simulator.forward_segments(params, positions_mm, de, dx=5.0)
# Gradients flow through velocity, lifetime, diffusion, recombination
```

## Calibration & Optimization

Beyond running the simulator, JAXTPC includes a gradient-based calibration toolchain for
fitting detector physics parameters (drift velocity, lifetime, diffusion, recombination,
space-charge distortions) against a ground-truth signal.

### Event generator (`tools/random_boundary_tracks.py`)

`generate_random_boundary_tracks()` builds a small ensemble of muon tracks for calibration
studies: N random tracks entering through a detector x-face, with energies balanced evenly
across 100/500/1000 MeV, plus three fixed 1000 MeV chords that span both volumes (one body
diagonal, two skew chords). `generate_random_nice_tracks()` additionally filters by start
position (near-cathode) and minimum polar angle from the drift direction, to avoid
near-degenerate track geometries. `filter_track_inside_volumes()` drops any propagated steps
that fall outside the active volume.

### Differentiable space-charge-effect (SCE) calibration (`tools/nonlocal_efield.py`)

`tools/efield_distortions.py` generates non-differentiable analytic SCE maps (toy E-field +
drift-correction integration) used as a ground-truth distortion to fit against; it accepts
both HDF5 and `.npz` map formats and ships a CLI to produce them.

`tools/nonlocal_efield.py` is the differentiable counterpart: an MLP that maps detector
position to an E-field correction. `FieldConfig` selects between three parameterizations
(`potential`, `efield`, `correction`); `mode='potential'` is conservative by construction
(`E = E_bg - grad(phi)`, via `jax.grad`). `nominal_start_params` initializes the MLP with
random hidden-layer weights but a zeroed output layer, so it starts at the nominal field
(zero distortion) while still having a nonzero gradient — an all-zero MLP would have none.
`DetectorSimulator(..., efield_model=FieldConfig)` reads the MLP weights from
`SimParams.sce_models`, so `jax.grad` reaches them during optimization.

### Loss-landscape sweeps (`scripts/`)

`scripts/2d_loss_landscape.py` evaluates the loss over a 2D grid of two `SimParams` scalars
(e.g. `recomb_alpha` vs `recomb_beta_90`, or `velocity_cm_us` vs `lifetime_us`) and plots the
resulting surface, optionally overlaid with optimizer trajectories. `scripts/1d_gradients.py`
does the 1D equivalent with gradient/frequency-cutoff sweeps. `scripts/launch_2d_landscape_pairs.py`
submits one Slurm job per (track × parameter-pair) using the boundary-track ensemble above;
`scripts/combine_landscapes.py` merges the resulting `.pkl` outputs.

### `run_optimization.py` + `optlib/`

`run_optimization.py` runs gradient-based optimization over any number of detector parameters
from N random starting points, against one or more muon tracks:

```bash
python run_optimization.py \
  --params recomb_alpha,recomb_beta_90 \
  --range 0.98 1.02 \
  --tracks diagonal,Z \
  --loss sobolev_loss_geomean_log1p \
  --optimizer adam --lr 0.01 --max-steps 200 --lr-schedule cosine \
  --N 25
```

- **Optimizers**: `adam`, `sgd`, `momentum_sgd` (via `optax`, with warmup + cosine-decay
  schedule), or `newton` (damped Newton step using the per-batch Hessian). The optimizer is a
  whole-run choice via `--optimizer`; there is no built-in Adam-then-Newton phase switch.
  A separate, orthogonal `--phase2-params`/`--phase2-start-step` schedule can freeze a subset
  of parameters until partway through the run (gradient masking, not an optimizer switch).
- **Noisy-GT fitting**: `--noise-scale` adds calibrated MicroBooNE-style detector noise to the
  ground-truth signal once before optimization (`--rotate-noise-seeds` to vary it per restart),
  so the simulation can be fit against a *noisy* GT while staying noise-free itself — the
  realistic calibration scenario.
- **Efield/SCE optimization**: pass `Efield` in `--params` (combinable with scalar params) plus
  `--electric-dist-path <npz-or-h5>` to fit the differentiable MLP SCE model above against a
  static ground-truth distortion map. Not compatible with `--optimizer newton`.
- Implementation is split into `optlib/`: CLI parsing, `SimParams` getters/setters, the
  optimizer/schedule plumbing, result-folder naming + atomic pickle I/O, and optional W&B
  logging.

### Notes

- The diffusion sigma used to size the static wire-response convolution kernel was previously
  computed in wire-pitch units rather than bins (an off-by-`wire_spacing` factor), making the
  kernel roughly 10x too small; this is fixed in `tools/kernels.py`/`tools/simulation.py`.

## Features

- **GPU-accelerated**: Full JAX JIT compilation, `lax.scan` volume iteration
- **N-volume architecture**: Arbitrary number of detector volumes (2 to 70+ tested)
- **Wire and pixel readout**: Configurable per detector config
- **Local coordinates**: Deposits transformed to volume-local frame in loader; all volumes geometrically identical for physics
- **Scan/vmap iteration**: Volumes processed via `lax.scan` (default) or `jax.vmap`; one compiled body for any N
- **Electron drift**: Diffusion (spatial convolution kernel generation) and lifetime attenuation
- **Angle-dependent recombination**: Modified Box (ArgoNeuT) and EMB (ICARUS 2024) models
- **Electronics response**: RC-RC convolution via sparse FFT
- **Intrinsic noise**: Wire-length-dependent noise model (MicroBooNE)
- **ADC digitization**: Configurable bit depth, pedestal, gain
- **Space charge effects**: Per-volume SCE maps loaded in local frame
- **Track correspondence**: Group-based 3D-to-2D mapping with Q_s disaggregation fractions
- **Differentiable path**: `jax.remat` + scan for gradients through all physics parameters
- **Threaded production**: Overlapped GPU simulation with CPU save workers
- **Production profiler**: Auto-tune total_pad, chunk sizes, max_keys from data

## Architecture

### Local Coordinates

The loader transforms deposits to volume-local coordinates:
```
x_local = drift_direction * (x_anode - x_global)    # distance from anode, >= 0
y_local = y_global - y_center
z_local = z_global - z_center
```

In local frame, all volumes share reference geometry (anode at x=0, drift toward -x, yz centered). The physics uses fixed constants — no per-volume geometry indexing needed in the scan body. Seg files save global positions (inverse transform applied before writing).

### Volume Iteration

All volumes are processed by a single `lax.scan` (or `vmap`) body compiled once. The body handles recombination, drift, per-plane wire response, electronics, noise, digitization, and track labeling. Plane loops (typically 3 for wire) are unrolled at trace time inside the body.

### Configuration

- **`SimConfig`** (static, closure-captured) — Array dimensions, mode flags, volume geometry. Changing triggers recompilation.
- **`SimParams`** (dynamic, JIT argument) — Physics scalars (velocity, lifetime, diffusion, recombination). Changeable per-call.

## Input Data Format

HDF5 files with particle segments from simulation (e.g., Geant4):
- `position`: (N, 3) — x, y, z in mm
- `dE`: (N,) — energy deposits in MeV
- `dx`: (N,) — step length in mm
- `theta`: (N,) — polar angle
- `phi`: (N,) — azimuthal angle
- `track_id`: (N,) — particle track IDs

## Output

The simulator returns `(response_signals, track_hits_raw, deposits)`:

1. **response_signals**: `{(vol_idx, plane_idx): array}` — wire signals (dense, bucketed, or wire-sparse)
2. **track_hits_raw**: `{(vol_idx, plane_idx): tuple}` — raw group correspondence for track labeling
3. **deposits**: `DepositData` — input deposits with `charge`, `photons`, `qs_fractions` filled

## Detector Configurations

| Config | Volumes | Readout | Description |
|--------|---------|---------|-------------|
| `cubic_wireplane_config.yaml` | 2 | Wire (U/V/Y) | Default, SBND-scale |
| `sbnd_config.yaml` | 2 | Wire | SBND |
| `microboone_config.yaml` | 1 | Wire | MicroBooNE |
| `icarus_config.yaml` | 4 | Wire | ICARUS |
| `dune_ndlar_config.yaml` | 70 | Wire | DUNE ND-LAr (5x7 module grid) |
| `dune_fd1_config.yaml` | 2 | Wire | DUNE Far Detector |
| `pixel_cube_config.yaml` | 1 | Pixel | Pixel readout test |

## Simulation Parameters

| Parameter | Default | Description |
|---|---|---|
| `total_pad` | 200,000 | Padded array size per volume (sets JIT shape) |
| `response_chunk_size` | 50,000 | Deposits per fori_loop iteration |
| `iterate_mode` | `'scan'` | Volume iteration: `'scan'` or `'vmap'` |
| `use_bucketed` | False | Sparse bucket accumulation (required for pixel) |
| `max_active_buckets` | 1,000 | Max buckets per plane (bucketed mode) |
| `include_noise` | False | Enable intrinsic noise |
| `include_electronics` | False | Enable RC-RC electronics response |
| `include_digitize` | False | Enable ADC digitization |
| `include_track_hits` | True | Enable track correspondence |
| `include_electric_dist` | False | Enable space charge effects |
| `differentiable` | False | Enable differentiable path (with `n_segments`) |
