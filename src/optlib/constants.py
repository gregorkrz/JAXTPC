"""Pure constants shared across the optimization driver.

Extracted from ``src/opt/run_optimization.py`` (no logic, no side effects).
"""

import os

# Output directory defaults (env-overridable; no side effects).
_RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')
_PLOTS_DIR   = os.environ.get('PLOTS_DIR',   'plots')

# Log per-track loss curves to W&B only below this track count (avoids huge metric cardinality).
WANDB_PER_TRACK_LOSS_MAX_TRACKS = 50
GT_LIFETIME_US    = 10_000.0
GT_VELOCITY_CM_US = 0.160
SOBOLEV_MAX_PAD   = 128

CONFIG_PATH        = 'config/cubic_wireplane_config.yaml'
N_SEGMENTS         = 50_000
MAX_ACTIVE_BUCKETS = 1000

# Special non-scalar param: a differentiable MLP E-field distortion model whose
# weights live in SimParams.sce_models (see tools/nonlocal_efield.py). It is
# stripped from the scalar param list in main() and handled separately.
EFIELD_PARAM = 'Efield'

VALID_PARAMS = (
    'velocity_cm_us',
    'lifetime_us',
    'diffusion_trans_cm2_us',
    'diffusion_long_cm2_us',
    'recomb_alpha',
    'recomb_beta',
    'recomb_beta_90',
    'recomb_R',
    EFIELD_PARAM,
)

# "All params" = all non-beta scalar params + at least one beta variant (model-specific).
_BETA_VARIANTS = frozenset({'recomb_beta', 'recomb_beta_90'})
_BASE_PARAMS   = frozenset(VALID_PARAMS) - _BETA_VARIANTS - {EFIELD_PARAM}

VALID_LOSSES     = ('sobolev_loss', 'sobolev_loss_geomean_log1p', 'mse_loss', 'l1_loss')
VALID_OPTIMIZERS = ('adam', 'sgd', 'momentum_sgd', 'newton')

TYPICAL_SCALES = {
    'velocity_cm_us':         0.1,
    'lifetime_us':            10_000.0,
    'diffusion_trans_cm2_us': 1e-5,
    'diffusion_long_cm2_us':  1e-5,
    'recomb_alpha':           1.0,
    'recomb_beta':            0.2,
    'recomb_beta_90':         0.2,
    'recomb_R':               1.0,
}

# Named track presets: name → (direction_xyz_tuple, momentum_mev)
TRACK_PRESETS = {
    'diagonal': ((1.0,  1.0,  1.0),  1000.0),
    'X':        ((1.0,  0.0,  0.0),  1000.0),
    'Y':        ((0.0,  1.0,  0.0),  1000.0),
    'Z':        ((0.0,  0.0,  1.0),  1000.0),
    'U':        ((0.0,  0.866, 0.5), 1000.0),
    'V':        ((0.0, -0.866, 0.5), 1000.0),
    'track2':   ((0.5,  1.05, 0.2),   200.0),
}

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS   = 1e-8
MOMENTUM   = 0.9

# Plane name → global plane indices (0=U1, 1=V1, 2=Y1, 3=U2, 4=V2, 5=Y2)
PLANE_NAME_MAP = {
    'U1': (0,), 'V1': (1,), 'Y1': (2,),
    'U2': (3,), 'V2': (4,), 'Y2': (5,),
    'U':  (0, 3), 'V': (1, 4), 'Y': (2, 5),
}
