#!/usr/bin/env bash
# Generate HTML viewers for the 20-track ensemble into cutoff_loss_landscape_20260526/.
#
#   viewer.html          — interactive signal viewer (from signal-array pkls)
#   landscape_viewer.html — loss landscape (from sobolev_cutoff_15trk_all_planes pkls)
#
# Run on sdfiana005 after 20tracks_signal_jobs_S3DF.sh jobs complete:
#   bash 20tracks_displays_S3DF.sh
set -euo pipefail

cd /sdf/home/g/gregork/jaxtpc
source .env

PY=/sdf/home/g/gregork/envs/base_env/bin/python
OUT=$PLOTS_DIR/cutoff_loss_landscape_20260526

mkdir -p "$OUT"

echo "=== Signal viewer (viewer.html) ==="
$PY src/analysis/sim_param_sweeps/generate_gradient_viewer.py \
    --dir "$RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260526" \
    --output "$OUT/viewer.html"

echo ""
echo "=== Loss landscape (landscape_viewer.html) ==="
$PY src/plots/plot_gradient_landscape_viewer.py \
    --results-dir "$RESULTS_DIR/1d_gradients/sobolev_cutoff_15trk_all_planes" \
    --output "$OUT/landscape_viewer.html"

echo ""
echo "Done. Output in $OUT"
