#!/usr/bin/env bash
# Sobolev loss cutoff study — diffusion parameters (local run)
#
# Sweeps both diffusion constants over ±75% of GT in 41 points (N=20),
# at ADC cutoffs of 0 (baseline), 1, 3, 5, 10, 15, 20, 25, 50,
# for both clean and noisy GT (noise-scale 1.0).
# No arrays stored — only loss values and gradients are saved.
#
# Original 4 tracks: full cutoff sweep (0–25).
# Remaining 11 tracks from the 15-track ensemble: ADC cutoff 25 only.
#
# Results: results/1d_gradients/sobolev_cutoff_diffusion_N20_range75pct/
#
# Run from the JAXTPC repo root:
#   bash scripts/Run_SobolevLossWithCutoff.sh

set -euo pipefail

OVERWRITE=0
CUTOFFS_CSV=""        # comma-separated; empty → default 0,1,3,5,10,15,20,25
STORE_PER_PLANE=0

for arg in "$@"; do
  case $arg in
    --overwrite)        OVERWRITE=1 ;;
    --cutoffs=*)        CUTOFFS_CSV="${arg#--cutoffs=}" ;;
    --store-per-plane)  STORE_PER_PLANE=1 ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# Cutoff list for the full-sweep tracks (space-separated for bash iteration).
if [[ -n "$CUTOFFS_CSV" ]]; then
  CUTOFF_LIST="${CUTOFFS_CSV//,/ }"
else
  CUTOFF_LIST="0.0 1.0 3.0 5.0 10.0 15.0 20.0 25.0 50.0"
fi

# Extra flags forwarded to 1d_gradients.py.
EXTRA_FLAGS=()
[[ $STORE_PER_PLANE -eq 1 ]] && EXTRA_FLAGS+=(--store-per-plane-loss)

PYTHON=.venv/bin/python
OUTDIR=results/1d_gradients/sobolev_cutoff_diffusion_N20_range75pct

if [[ $OVERWRITE -eq 1 && -d "$OUTDIR" ]]; then
  echo "=== --overwrite: removing existing results in ${OUTDIR} ==="
  rm -rf "$OUTDIR"
fi

# Original 4 tracks — full cutoff sweep (0, 1, 3, 5, 10, 15, 20, 25, 50  ADC).
# diagonal = body-diagonal chord; Muon1/2/4 from the 15-track ensemble (seed=42).
declare -A TRACKS=(
  [diagonal]="diagonal:-0.577350,-0.577350,-0.577350:1000"
  [Muon1_1000MeV]="Muon1_1000MeV:-0.747872530,0.661463000,0.056154945:1000"
  [Muon2_500MeV]="Muon2_500MeV:-0.641581737,0.275323919,-0.715939672:500"
  [Muon4_100MeV]="Muon4_100MeV:-0.694627880,0.476880059,0.538588450:100"
)

# Remaining 11 tracks from the 15-track boundary ensemble (seed=42,
# tools/random_boundary_tracks.py) — ADC cutoff 25 only.
# Format: name:dx,dy,dz:mom_mev:sx,sy,sz  (start position in mm).
declare -A TRACKS_CUTOFF25=(
  [Muon3_500MeV]="Muon3_500MeV:-0.483652826,0.868350593,-0.109759697:500:2160.0,568.790204207,1114.939037169"
  [Muon5_100MeV]="Muon5_100MeV:-0.448568523,-0.712616910,0.539410252:100:2160.0,1057.372513522,2019.642044116"
  [Muon6_1000MeV]="Muon6_1000MeV:-0.624672693,-0.613017831,0.483728401:1000:2160.0,-1341.483728756,-1598.739096951"
  [Muon7_500MeV]="Muon7_500MeV:-0.610394124,-0.747896572,0.260901765:500:2160.0,1437.169806970,865.145240650"
  [Muon8_1000MeV]="Muon8_1000MeV:0.773174642,0.198385012,0.602365637:1000:-2160.0,-486.093402590,-914.422591021"
  [Muon9_500MeV]="Muon9_500MeV:-0.931562076,-0.204366326,-0.300710000:500:2160.0,1239.513310809,712.155700478"
  [Muon10_100MeV]="Muon10_100MeV:0.754859526,-0.437194999,0.488924973:100:-2160.0,296.961966517,-1556.076968089"
  [Muon11_1000MeV]="Muon11_1000MeV:-0.482051010,0.584842086,0.652369955:1000:2160.0,1144.795064037,581.983142403"
  [Muon12_100MeV]="Muon12_100MeV:-0.553810025,-0.123483953,-0.823435589:100:2160.0,-2026.866954667,-273.380878516"
  [Muon_throughEw_skew02_1000MeV]="Muon_throughEw_skew02_1000MeV:0.934631179,-0.282614666,0.215855296:1000:-2100.0,750.0,-550.0"
  [Muon_throughWe_skew03_1000MeV]="Muon_throughWe_skew03_1000MeV:-0.938658230,0.268188066,-0.216785353:1000:2100.0,-620.0,480.0"
)

echo "=== Sobolev loss ADC-cutoff study — diffusion parameters ==="
echo "  params             : diffusion_trans_cm2_us  diffusion_long_cm2_us"
echo "  tracks (full sweep): ${!TRACKS[*]}"
echo "  tracks (cutoff=25) : ${!TRACKS_CUTOFF25[*]}"
echo "  noise scales       : 0  1.0"
echo "  cutoffs (full)     : ${CUTOFF_LIST}"
echo "  cutoffs (cutoff25) : 25 (ADC)"
echo "  N=20, range=±75%,  output: ${OUTDIR}"
echo "  extra flags        : ${EXTRA_FLAGS[*]:-none}"
echo

for TRACK_KEY in "${!TRACKS[@]}"; do
  TRACK_SPEC="${TRACKS[$TRACK_KEY]}"
  for PARAM in diffusion_trans_cm2_us diffusion_long_cm2_us; do
    for NOISE in 0.0 1.0; do
      for CUTOFF in $CUTOFF_LIST; do
        echo "--- track=${TRACK_KEY}  param=${PARAM}  noise=${NOISE}  cutoff=${CUTOFF} ---"
        $PYTHON src/analysis/1d_gradients.py \
          --param        "$PARAM"       \
          --tracks       "$TRACK_SPEC"  \
          --N            20             \
          --range-frac   0.75           \
          --noise-scale  "$NOISE"       \
          --adc-cutoff   "$CUTOFF"      \
          --results-dir  "$OUTDIR"      \
          "${EXTRA_FLAGS[@]}"
        echo
      done
    done
  done
done

for TRACK_KEY in "${!TRACKS_CUTOFF25[@]}"; do
  TRACK_SPEC="${TRACKS_CUTOFF25[$TRACK_KEY]}"
  for PARAM in diffusion_trans_cm2_us diffusion_long_cm2_us; do
    for NOISE in 0.0 1.0; do
      echo "--- track=${TRACK_KEY}  param=${PARAM}  noise=${NOISE}  cutoff=25.0 ---"
      $PYTHON src/analysis/1d_gradients.py \
        --param        "$PARAM"       \
        --tracks       "$TRACK_SPEC"  \
        --N            20             \
        --range-frac   0.75           \
        --noise-scale  "$NOISE"       \
        --adc-cutoff   25.0           \
        --results-dir  "$OUTDIR"      \
        "${EXTRA_FLAGS[@]}"
      echo
    done
  done
done

echo "=== All runs complete. Results in ${OUTDIR} ==="
