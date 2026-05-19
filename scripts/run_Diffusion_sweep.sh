#!/usr/bin/env bash
# run_Diffusion_sweep.sh
#
# Sweeps transverse and longitudinal diffusion coefficients across the 15-track
# ensemble and N random point deposits, saving main-wire traces + peak values
# (clean and noisy) to a pkl file.
#
# Edit SWEEP_PARAMS / TRACK_* / POINT_* / NOISE_* constants at the top of
# run_params.py to change what is swept and over what ranges.
#
# Edit SWEEP_PARAMS in src/analysis/sim_param_sweeps/run_params.py to change
# which parameters are swept and their ranges.
#
# Usage (local):
#   bash scripts/run_Diffusion_sweep.sh
#
# Usage (S3DF):
#   PYTHON=/sdf/home/g/gregork/envs/base_env/bin/python \
#     bash scripts/run_Diffusion_sweep.sh

set -euo pipefail

PYTHON=${PYTHON:-.venv/bin/python}
SCRIPT=src/analysis/sim_param_sweeps/run_params.py
OUTPUT_DIR=${OUTPUT_DIR:-results/diffusion_sweep}
CONFIG=${CONFIG:-config/cubic_wireplane_config.yaml}
VOL_IDX=${VOL_IDX:-0}

echo "=== Diffusion parameter sweep ==="
echo "Python     : $PYTHON"
echo "Script     : $SCRIPT"
echo "Output dir : $OUTPUT_DIR"
echo "Config     : $CONFIG"
echo "Volume idx : $VOL_IDX"
echo

$PYTHON "$SCRIPT" \
    --output-dir "$OUTPUT_DIR" \
    --config     "$CONFIG" \
    --vol-idx    "$VOL_IDX"

echo
echo "Done. Results in $OUTPUT_DIR/"
