#!/usr/bin/env bash
# 1D loss-landscape viewer for the pivoting (theta, alpha) muon tracks,
# diagonal sweep only (theta == alpha), higher-angle continuation of the
# diagonal already covered in pivot_theta_alpha_landscape_local.sh
# (0,0 / 10,10 / 20,20).
#
# Pivot: 400 MeV muon, midpoint fixed at (1000, 0, 0) mm (CSDA half-length
# 850.3 mm), same convention as submit_diffusion_angle_pivot_theta_alpha_study
# in src/jobs/submit_jobs_loss_studies.py:
#   dx = -cos(theta)*cos(alpha), dy = sin(theta)*cos(alpha), dz = sin(alpha)
#   start = (1000 + L*cos(theta)*cos(alpha), -L*sin(theta)*cos(alpha), -L*sin(alpha))
#
# 6 tracks, diagonal sweep continuing past the local script's 0/10/20:
#   (theta, alpha) = (25,25), (30,30), (35,35), (40,40), (45,45), (50,50)
#
# Same params as the Slurm profile (submit_diffusion_angle_pivot_theta_alpha_study):
#   N=100, range-frac=0.2 -> 201-point landscape spanning 0.8-1.2x GT per param.
#   ADC cutoffs [0, 50] only (no multi-cutoff sweep here, to keep pkls small).
#   Noise: scale=1.0, seeds 0-49 (50 seeds) -> noisy results dir.
#          scale=0.0, seed 0 (single)       -> separate "_nonoise" results dir.
#   Only scalar losses are stored (no --store-per-plane-loss / per-pixel /
#   array dumps), so the output pkls stay small.
#
# Output: $RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_diag/
#         $RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_diag_nonoise/
#         viewer -> $PLOTS_DIR/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_diag/viewer.html
set -euo pipefail

cd /home/gregor/JAXTPC
source .env

PY=.venv/bin/python

TRACKS="Muon_400MeV_theta_25_alpha_25_pivot_x1000_stepsize_1mm:-0.821393805,0.383022222,0.422618262:400:1698.431,-325.684,-359.352"\
"+Muon_400MeV_theta_30_alpha_30_pivot_x1000_stepsize_1mm:-0.750000000,0.433012702,0.500000000:400:1637.725,-368.191,-425.15"\
"+Muon_400MeV_theta_35_alpha_35_pivot_x1000_stepsize_1mm:-0.671010072,0.469846310,0.573576436:400:1570.56,-399.51,-487.712"\
"+Muon_400MeV_theta_40_alpha_40_pivot_x1000_stepsize_1mm:-0.586824089,0.492403877,0.642787610:400:1498.977,-418.691,-546.562"\
"+Muon_400MeV_theta_45_alpha_45_pivot_x1000_stepsize_1mm:-0.500000000,0.500000000,0.707106781:400:1425.15,-425.15,-601.253"\
"+Muon_400MeV_theta_50_alpha_50_pivot_x1000_stepsize_1mm:-0.413175911,0.492403877,0.766044443:400:1351.323,-418.691,-651.368"

OUT_DIR="$RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_diag"
OUT_DIR_NN="$RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_diag_nonoise"

SEEDS=$(seq -s, 0 49)

COMMON="--tracks $TRACKS --N 100 --range-frac 0.2 --step-size 1.0 --max-deposits 5000 --sobolev-max-pad 128 --adc-cutoffs 0,50"

$PY src/analysis/1d_gradients.py --param diffusion_trans_cm2_us --noise-scale 1.0 --noise-seeds "$SEEDS" --results-dir "$OUT_DIR" $COMMON
$PY src/analysis/1d_gradients.py --param diffusion_long_cm2_us  --noise-scale 1.0 --noise-seeds "$SEEDS" --results-dir "$OUT_DIR" $COMMON

$PY src/analysis/1d_gradients.py --param diffusion_trans_cm2_us --noise-scale 0.0 --noise-seed 0 --results-dir "$OUT_DIR_NN" $COMMON
$PY src/analysis/1d_gradients.py --param diffusion_long_cm2_us  --noise-scale 0.0 --noise-seed 0 --results-dir "$OUT_DIR_NN" $COMMON

# Generate HTML viewer for all pkls in both results directories
$PY src/analysis/sim_param_sweeps/generate_gradient_viewer.py --dir "$OUT_DIR"
$PY src/analysis/sim_param_sweeps/generate_gradient_viewer.py --dir "$OUT_DIR_NN"
