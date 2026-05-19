#!/usr/bin/env bash
# Plot Sobolev loss cutoff study — diffusion parameters
#
# Reads pkl files from the cutoff study run by Run_SobolevLossWithCutoff.sh
# and produces per-parameter PDFs plus a gradient overlay figure.
#
# Run from the JAXTPC repo root:
#   bash scripts/Plot_SobolevLossWithCutoff.sh

set -euo pipefail

PYTHON=.venv/bin/python
INDIR=results/1d_gradients/sobolev_cutoff_diffusion_N20_range75pct
OUTDIR=plots/sobolev_cutoff_diffusion_N20_range75pct

echo "=== Sobolev loss ADC-cutoff plots ==="
echo "  Input : ${INDIR}"
echo "  Output: ${OUTDIR}"
echo

$PYTHON src/plots/plot_sobolev_cutoff.py \
    --results-dir "$INDIR" \
    --output-dir  "$OUTDIR"

echo
echo "=== Done. PDFs written to ${OUTDIR} ==="
