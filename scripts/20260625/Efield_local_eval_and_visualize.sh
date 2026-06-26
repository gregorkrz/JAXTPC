#!/usr/bin/env bash
# Evaluate the learned SIREN E-field and generate an interactive HTML viewer.
#
# Step 1 (eval_efield_mlp.py):
#   Loads result PKLs from the local opt run, evaluates the learned SIREN on a
#   3D grid, compares against the GT NPZ, writes *_efield_eval.pkl files.
#
# Step 2 (plot_efield_eval.py):
#   Reads the eval PKLs and produces an interactive HTML with learned / GT /
#   difference slices for E-field, drift corrections, and potential.
#
# NOTE: eval_efield_mlp.py currently uses the old FieldConfig MLP interface and
# will need updating for the new SirenDistortionConfig before this works end-to-end.
set -euo pipefail

cd /home/gregor/JAXTPC
source .env

PY=.venv/bin/python
RESULTS_DIR_LOCAL="$RESULTS_DIR/opt/efield_siren_local"
OUT_HTML="$PLOTS_DIR/opt/efield_siren_local/efield_eval.html"

mkdir -p "$(dirname "$OUT_HTML")"

# Step 1: run inference on all result PKLs in the local opt output directory
$PY tools/eval_efield_mlp.py \
  --results-dir "$RESULTS_DIR_LOCAL" \
  --overwrite

# Step 2: generate HTML viewer from the eval PKLs
$PY tools/plot_efield_eval.py \
  --results-dir "$RESULTS_DIR_LOCAL" \
  --output "$OUT_HTML"

echo "Viewer: $OUT_HTML"
