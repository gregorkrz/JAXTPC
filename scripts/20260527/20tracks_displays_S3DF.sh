#!/usr/bin/env bash
# Generate HTML viewers for the 20-track ensemble into cutoff_loss_landscape_20260526/.
#
#   viewer.html              — landscape viewer, s=2 default (cutoff_loss_landscape_20260526)
#   viewer_exp_1.html        — landscape viewer, s=1        (cutoff_loss_landscape_20260526_exp_1)
#   viewer_exp_0.html        — landscape viewer, s=0        (cutoff_loss_landscape_20260526_exp_0)
#   landscape_viewer.html    — loss landscape (from sobolev_cutoff_15trk_all_planes pkls)
#
# Run on sdfiana005 after 20tracks_signal_jobs_S3DF.sh jobs complete:
#   bash 20tracks_displays_S3DF.sh
set -euo pipefail

cd /sdf/home/g/gregork/jaxtpc
source .env

PY=/sdf/home/g/gregork/envs/base_env/bin/python
OUT=$PLOTS_DIR/cutoff_loss_landscape_20260526

mkdir -p "$OUT"

echo "=== Loss landscape (landscape_viewer.html) ==="
$PY src/plots/plot_gradient_landscape_viewer.py \
    --results-dir "$RESULTS_DIR/1d_gradients/sobolev_cutoff_20trk_all_planes" \
    --output "$OUT/landscape_viewer.html"

echo "=== Landscape viewer s=2 default (viewer.html) ==="
$PY src/plots/plot_gradient_landscape_viewer.py \
    --results-dir "$RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260526" \
    --output "$OUT/viewer.html"

echo "=== Landscape viewer s=1 (viewer_exp_1.html) ==="
$PY src/plots/plot_gradient_landscape_viewer.py \
    --results-dir "$RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260526_exp_1" \
    --output "$OUT/viewer_exp_1.html"

echo "=== Landscape viewer s=0 (viewer_exp_0.html) ==="
$PY src/plots/plot_gradient_landscape_viewer.py \
    --results-dir "$RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260526_exp_0" \
    --output "$OUT/
    viewer_exp_0.html"

echo ""

echo "Done. Output in $OUT"
