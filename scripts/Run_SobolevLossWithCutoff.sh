#!/usr/bin/env bash
# Sobolev loss cutoff study — diffusion parameters (local run)
#
# Sweeps both diffusion constants over ±75% of GT in 41 points (N=20),
# at ADC cutoffs of 0 (baseline), 1, 3, 5, 10, 20,
# for both clean and noisy GT (noise-scale 1.0).
# No arrays stored — only loss values and gradients are saved.
#
# Results: results/1d_gradients/sobolev_cutoff_diffusion_N20_range75pct/
#
# Run from the JAXTPC repo root:
#   bash scripts/Run_SobolevLossWithCutoff.sh

set -euo pipefail

PYTHON=.venv/bin/python
OUTDIR=results/1d_gradients/sobolev_cutoff_diffusion_N20_range75pct

# Tracks to sweep (name:dx,dy,dz:mom_mev).
# diagonal = body-diagonal chord used in landscape plots.
# Muon1/2/4 = from the 15-track boundary ensemble (seed=42, tools/random_boundary_tracks.py).
declare -A TRACKS=(
  [diagonal]="diagonal:-0.577350,-0.577350,-0.577350:1000"
  [Muon1_1000MeV]="Muon1_1000MeV:-0.747872530,0.661463000,0.056154945:1000"
  [Muon2_500MeV]="Muon2_500MeV:-0.641581737,0.275323919,-0.715939672:500"
  [Muon4_100MeV]="Muon4_100MeV:-0.694627880,0.476880059,0.538588450:100"
)

echo "=== Sobolev loss ADC-cutoff study — diffusion parameters ==="
echo "  params      : diffusion_trans_cm2_us  diffusion_long_cm2_us"
echo "  tracks      : ${!TRACKS[*]}"
echo "  noise scales: 0  1.0"
echo "  cutoffs     : 0  1  3  5  10  15  20  25  (ADC)"
echo "  N=20, range=±75%,  output: ${OUTDIR}"
echo

for TRACK_KEY in "${!TRACKS[@]}"; do
  TRACK_SPEC="${TRACKS[$TRACK_KEY]}"
  for PARAM in diffusion_trans_cm2_us diffusion_long_cm2_us; do
    for NOISE in 0.0 1.0; do
      for CUTOFF in 0.0 1.0 3.0 5.0 10.0 15.0 20.0 25.0; do
        echo "--- track=${TRACK_KEY}  param=${PARAM}  noise=${NOISE}  cutoff=${CUTOFF} ---"
        $PYTHON src/analysis/1d_gradients.py \
          --param        "$PARAM"       \
          --tracks       "$TRACK_SPEC"  \
          --N            20             \
          --range-frac   0.75           \
          --noise-scale  "$NOISE"       \
          --adc-cutoff   "$CUTOFF"      \
          --results-dir  "$OUTDIR"
        echo
      done
    done
  done
done

echo "=== All runs complete. Results in ${OUTDIR} ==="
