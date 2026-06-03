#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python \
    src/plots/plot_mixed_tracks_edep_wireplanes.py \
    --nice-tracks --n-nice 10 --seed 7 \
    --output-dir "$PLOTS_DIR/20260602/nice_tracks_edep"
