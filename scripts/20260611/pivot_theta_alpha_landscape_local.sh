#!/usr/bin/env bash
# 1D loss-landscape viewer for the pivoting (theta, alpha) muon tracks.
#
# Pivot: 400 MeV muon, midpoint fixed at (1000, 0, 0) mm (CSDA half-length
# 850.3 mm), same convention as submit_diffusion_angle_pivot_theta_alpha_study
# in src/jobs/submit_jobs_loss_studies.py:
#   dx = -cos(theta)*cos(alpha), dy = sin(theta)*cos(alpha), dz = sin(alpha)
#   start = (1000 + L*cos(theta)*cos(alpha), -L*sin(theta)*cos(alpha), -L*sin(alpha))
#
# 9 unique tracks (the (theta=0, alpha=0) combo is shared between the
# requested sets, so it appears only once):
#   (theta, alpha) = (0,0), (10,10), (20,20)        -- diagonal sweep
#   (theta, alpha) = (10,0), (20,0), (30,0)         -- alpha=0, theta sweep (0,0 already above)
#   (theta, alpha) = (0,10), (0,20), (0,30)         -- theta=0, alpha sweep (0,0 already above)
#
# ADC cutoff 50, no Fourier cutoff (FT cutoff 0 = off, but Fourier loss/power
# maps are still stored and viewable per track via --store-per-pixel-loss-and-grad).
#
# Runs both clean (noise=0) and noisy (noise=1, seeds 42 and 43) variants;
# the viewer shows clean + each noisy seed as separate entries per param.
#
# Output: $RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta/
#         viewer -> $PLOTS_DIR/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta/viewer.html
set -euo pipefail

cd /home/gregor/JAXTPC
source .env

PY=.venv/bin/python

TRACKS="Muon_400MeV_theta_0_alpha_0_pivot_x1000_stepsize_1mm:-1.000000000,0.000000000,0.000000000:400:1850.3,-0.0,-0.0"\
"+Muon_400MeV_theta_0_alpha_10_pivot_x1000_stepsize_1mm:-0.984807753,0.000000000,0.173648178:400:1837.382,-0.0,-147.653"\
"+Muon_400MeV_theta_0_alpha_20_pivot_x1000_stepsize_1mm:-0.939692621,0.000000000,0.342020143:400:1799.021,-0.0,-290.82"\
"+Muon_400MeV_theta_0_alpha_30_pivot_x1000_stepsize_1mm:-0.866025404,0.000000000,0.500000000:400:1736.381,-0.0,-425.15"\
"+Muon_400MeV_theta_10_alpha_10_pivot_x1000_stepsize_1mm:-0.969846310,0.171010072,0.173648178:400:1824.66,-145.41,-147.653"\
"+Muon_400MeV_theta_20_alpha_20_pivot_x1000_stepsize_1mm:-0.883022222,0.321393805,0.342020143:400:1750.834,-273.281,-290.82"\
"+Muon_400MeV_theta_10_alpha_0_pivot_x1000_stepsize_1mm:-0.984807753,0.173648178,0.000000000:400:1837.382,-147.653,-0.0"\
"+Muon_400MeV_theta_20_alpha_0_pivot_x1000_stepsize_1mm:-0.939692621,0.342020143,0.000000000:400:1799.021,-290.82,-0.0"\
"+Muon_400MeV_theta_30_alpha_0_pivot_x1000_stepsize_1mm:-0.866025404,0.500000000,0.000000000:400:1736.381,-425.15,-0.0"

OUT_DIR="$RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta"

# Drop the old 6-track pkls: the track count is part of the output filename
# (e.g. "_6tracks_" -> "_9tracks_"), so the runs below recompute everything
# under new names. The stale 6-track files would otherwise share a merge key
# with the new 9-track ones in generate_gradient_viewer.py and break merging.
rm -fv "$OUT_DIR"/*_6tracks_*.pkl

COMMON="--tracks $TRACKS --factors 0.75,1,1.25 --step-size 1.0 --max-deposits 5000 --sobolev-max-pad 128 --results-dir $OUT_DIR --adc-cutoff 50 --store-per-plane-loss --store-per-pixel-loss-and-grad --store-arrays --save-per-factor"

$PY src/analysis/1d_gradients.py --param diffusion_trans_cm2_us --noise-scale 0.0 --noise-seed 42 $COMMON
$PY src/analysis/1d_gradients.py --param diffusion_long_cm2_us  --noise-scale 0.0 --noise-seed 42 $COMMON
$PY src/analysis/1d_gradients.py --param diffusion_trans_cm2_us --noise-scale 1.0 --noise-seeds 42,43 $COMMON
$PY src/analysis/1d_gradients.py --param diffusion_long_cm2_us  --noise-scale 1.0 --noise-seeds 42,43 $COMMON

# Generate HTML viewer for all pkls in the results directory
$PY src/analysis/sim_param_sweeps/generate_gradient_viewer.py --dir "$OUT_DIR"
