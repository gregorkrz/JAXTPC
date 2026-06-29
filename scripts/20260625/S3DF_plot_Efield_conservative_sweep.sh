#!/usr/bin/env bash
# Evaluate the learned E-field MLP and generate combined interactive HTMLs
# for the conservative_curl_sweep runs on S3DF.
#
# Step 1: eval_efield_mlp.py  — loads result PKLs, evaluates MLP on 3D grid,
#                               writes *_efield_eval.pkl files next to each PKL.
# Step 2: plot_efield_eval.py — lazy-loading HTML from all eval PKLs in sweep.
# Step 3: plot_loss_curves.py — loss curves HTML (fetches from W&B).
set -euo pipefail

PY=/sdf/home/g/gregork/envs/base_env/bin/python

SWEEP_DIR="$RESULTS_DIR/opt/efield_calib/conservative_curl_sweep/sce_maps_jaxtpc_conservative_41"
TRACKS100_DIR="$RESULTS_DIR/opt/efield_calib/tracks100_ebs20_rotor"
EFIELD_HTML="$PLOTS_DIR/opt/efield_calib/conservative_curl_sweep/efield_eval.html"
LOSS_HTML="$PLOTS_DIR/opt/efield_calib/conservative_curl_sweep/loss_curves.html"

mkdir -p "$(dirname "$EFIELD_HTML")"

# ── Step 1: evaluate all runs in both sweeps ─────────────────────────────────
$PY tools/eval_efield_mlp.py --results-dir "$SWEEP_DIR"
$PY tools/eval_efield_mlp.py --results-dir "$TRACKS100_DIR"

# ── Step 2: lazy-loading E-field viz HTML ─────────────────────────────────────
$PY tools/plot_efield_eval.py \
  --results-dir "$SWEEP_DIR" "$TRACKS100_DIR" \
  --output "$EFIELD_HTML"

# ── Step 3: loss curves HTML (fetches from W&B) ───────────────────────────────
$PY tools/plot_loss_curves.py \
  --results-dir "$SWEEP_DIR" "$TRACKS100_DIR" \
  --output "$LOSS_HTML"

echo "E-field viewer: $EFIELD_HTML"
echo "Loss curves:    $LOSS_HTML"
