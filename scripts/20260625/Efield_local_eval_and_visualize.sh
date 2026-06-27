#!/usr/bin/env bash
# Evaluate the learned SIREN E-field and generate a combined interactive HTML.
#
# Covers two runs (both appear in the Run dropdown of one HTML file):
#   1. efield_siren_local            — baseline (no rotor penalty)
#   2. efield_siren_local_penalize_rotor — with curl penalty
#
# Step 1 (eval_efield_mlp.py):
#   Loads result PKLs, evaluates the learned SIREN on a 3D grid, compares
#   against the GT NPZ, writes *_efield_eval.pkl files.
#
# Step 2 (plot_efield_eval.py):
#   Reads eval PKLs from both dirs and produces one HTML; the Run dropdown
#   labels include the rotor penalty weight for easy comparison.
set -euo pipefail

cd /home/gregor/JAXTPC
source .env

PY=.venv/bin/python

RESULTS_DIR_1="$RESULTS_DIR/opt/efield_siren_local"
RESULTS_DIR_2="$RESULTS_DIR/opt/efield_siren_local_penalize_rotor"
OUT_HTML="$PLOTS_DIR/opt/efield_siren_local/efield_eval.html"

mkdir -p "$(dirname "$OUT_HTML")"

# ── Step 1: evaluate both dirs ────────────────────────────────────────────────
$PY tools/eval_efield_mlp.py --results-dir "$RESULTS_DIR_1" --overwrite
$PY tools/eval_efield_mlp.py --results-dir "$RESULTS_DIR_2" --overwrite

# ── Step 2: combined HTML (both runs in one Run dropdown) ─────────────────────
$PY tools/plot_efield_eval.py \
  --results-dir "$RESULTS_DIR_1" "$RESULTS_DIR_2" \
  --output "$OUT_HTML"

echo "Viewer: $OUT_HTML"
