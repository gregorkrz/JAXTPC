#!/usr/bin/env bash
# Diffusion_visualization.sh
#
# Generates wire-plane signal comparison PDFs (GT1 vs GT2 + difference) for
# several transverse and longitudinal diffusion scenarios.
#
# Nominal values from the detector config:
#   diffusion_trans = 1.2e-5 cm²/μs
#   diffusion_long  = 7.2e-6 cm²/μs
#
# Usage (local):
#   bash scripts/Diffusion_visualization.sh
#
# Usage (S3DF — just swap the python path):
#   PYTHON=/sdf/home/g/gregork/envs/base_env/bin/python \
#     bash scripts/Diffusion_visualization.sh

set -euo pipefail

PYTHON=${PYTHON:-.venv/bin/python}
SCRIPT=src/plots/plot_diffusion_comparison.py
OUTBASE=plots/diffusion_visualization
TRACKS=diagonal

echo "=== Diffusion visualization ==="
echo "Python  : $PYTHON"
echo "Output  : $OUTBASE"
echo "Track   : $TRACKS"
echo

# ── 1. Transverse diffusion: nominal vs 2× ────────────────────────────────────
echo "[1/10] Transverse: nominal (1.2e-5) vs 2× (2.4e-5)"
$PYTHON $SCRIPT \
    --tracks "$TRACKS" \
    --gt1-label "nominal (1.2e-5)" \
    --gt2-params "diffusion_trans_cm2_us=2.4e-5" \
    --gt2-label "2× trans (2.4e-5)" \
    --output-dir "$OUTBASE/trans_2x"

# ── 2. Transverse diffusion: nominal vs 5× ────────────────────────────────────
echo "[2/10] Transverse: nominal (1.2e-5) vs 5× (6.0e-5)"
$PYTHON $SCRIPT \
    --tracks "$TRACKS" \
    --gt1-label "nominal (1.2e-5)" \
    --gt2-params "diffusion_trans_cm2_us=6.0e-5" \
    --gt2-label "5× trans (6.0e-5)" \
    --output-dir "$OUTBASE/trans_5x"

# ── 3. Transverse diffusion: nominal vs 10× ───────────────────────────────────
echo "[3/10] Transverse: nominal (1.2e-5) vs 10× (1.2e-4)"
$PYTHON $SCRIPT \
    --tracks "$TRACKS" \
    --gt1-label "nominal (1.2e-5)" \
    --gt2-params "diffusion_trans_cm2_us=1.2e-4" \
    --gt2-label "10× trans (1.2e-4)" \
    --output-dir "$OUTBASE/trans_10x"

# ── 4. Longitudinal diffusion: nominal vs 2× ──────────────────────────────────
echo "[4/10] Longitudinal: nominal (7.2e-6) vs 2× (1.44e-5)"
$PYTHON $SCRIPT \
    --tracks "$TRACKS" \
    --gt1-label "nominal (7.2e-6)" \
    --gt2-params "diffusion_long_cm2_us=1.44e-5" \
    --gt2-label "2× long (1.44e-5)" \
    --output-dir "$OUTBASE/long_2x"

# ── 5. Longitudinal diffusion: nominal vs 5× ──────────────────────────────────
echo "[5/10] Longitudinal: nominal (7.2e-6) vs 5× (3.6e-5)"
$PYTHON $SCRIPT \
    --tracks "$TRACKS" \
    --gt1-label "nominal (7.2e-6)" \
    --gt2-params "diffusion_long_cm2_us=3.6e-5" \
    --gt2-label "5× long (3.6e-5)" \
    --output-dir "$OUTBASE/long_5x"

# ── 6. Trans vs long: same factor, different axis (diagnostic) ────────────────
# GT1 = 5× transverse only, GT2 = 5× longitudinal only — shows the visual
# difference between the two types of diffusion at equal amplification.
echo "[6/10] 5× trans vs 5× long (cross-comparison)"
$PYTHON $SCRIPT \
    --tracks "$TRACKS" \
    --gt1-params "diffusion_trans_cm2_us=6.0e-5" \
    --gt1-label "5× trans (6.0e-5 / 7.2e-6)" \
    --gt2-params "diffusion_long_cm2_us=3.6e-5" \
    --gt2-label "5× long (1.2e-5 / 3.6e-5)" \
    --output-dir "$OUTBASE/trans_vs_long_5x"

# ── 7. Transverse 2× with 2× padding (256 vs default 128) ────────────────────
echo "[7/10] Transverse: nominal vs 2× — with 2× Sobolev padding (256)"
$PYTHON $SCRIPT \
    --tracks "$TRACKS" \
    --gt1-label "nominal (1.2e-5)" \
    --gt2-params "diffusion_trans_cm2_us=2.4e-5" \
    --gt2-label "2× trans (2.4e-5)" \
    --sobolev-max-pad 256 \
    --output-dir "$OUTBASE/trans_2x_padding256"

# ── 8. Longitudinal 2× with 2× padding (256 vs default 128) ──────────────────
echo "[8/10] Longitudinal: nominal vs 2× — with 2× Sobolev padding (256)"
$PYTHON $SCRIPT \
    --tracks "$TRACKS" \
    --gt1-label "nominal (7.2e-6)" \
    --gt2-params "diffusion_long_cm2_us=1.44e-5" \
    --gt2-label "2× long (1.44e-5)" \
    --sobolev-max-pad 256 \
    --output-dir "$OUTBASE/long_2x_padding256"

# ── 9. Transverse 2× with 2× padding + noise ──────────────────────────────────
echo "[9/10] Transverse: nominal vs 2× — 2× padding (256) + noise"
$PYTHON $SCRIPT \
    --tracks "$TRACKS" \
    --gt1-label "nominal (1.2e-5)" \
    --gt2-params "diffusion_trans_cm2_us=2.4e-5" \
    --gt2-label "2× trans (2.4e-5)" \
    --sobolev-max-pad 256 \
    --noise-scale 1.0 \
    --output-dir "$OUTBASE/trans_2x_padding256_noise"

# ── 10. Longitudinal 2× with 2× padding + noise ───────────────────────────────
echo "[10/10] Longitudinal: nominal vs 2× — 2× padding (256) + noise"
$PYTHON $SCRIPT \
    --tracks "$TRACKS" \
    --gt1-label "nominal (7.2e-6)" \
    --gt2-params "diffusion_long_cm2_us=1.44e-5" \
    --gt2-label "2× long (1.44e-5)" \
    --sobolev-max-pad 256 \
    --noise-scale 1.0 \
    --output-dir "$OUTBASE/long_2x_padding256_noise"

echo
echo "All done. PDFs saved under $OUTBASE/"
