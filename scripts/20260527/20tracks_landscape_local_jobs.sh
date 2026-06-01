#!/usr/bin/env bash
# Run 1d-gradient signal-array jobs locally (3 factors: 0.75, 1.0, 1.25).
# Outputs land in $RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260526_small/
#
# After jobs finish, run 20tracks_displays_S3DF_small.sh to generate HTML viewers.
set -euo pipefail

cd /home/gregor/JAXTPC
source .env

PY=.venv/bin/python

TRACKS="Muon1_1000MeV:-0.747872530,0.661463000,0.056154945:1000:2100.0,1600.0,-200.0+Muon9_500MeV:-0.931562076,-0.204366326,-0.300710000:500:2160.0,1239.513310809,712.155700478+Muon15_100MeV:0.290155194,0.467674594,-0.834919420:100:-738.504,-605.471,2160.000"

#Muon15_100MeV:0.290155194,0.467674594,-0.834919420:100:-738.504,-605.471,2160.000
#Muon9_500MeV:-0.931562076,-0.204366326,-0.300710000:500:2160.0,1239.513310809,712.155700478
#Muon1_1000MeV:-0.747872530,0.661463000,0.056154945:1000

COMMON="--tracks $TRACKS --factors 0.75,1,1.25 --step-size 1.0 --max-deposits 5000 --sobolev-max-pad 128 --results-dir $RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260526_small --adc-cutoffs 0,3,5,30,50 --store-per-plane-loss --store-per-pixel-loss-and-grad --store-arrays --save-per-factor"

$PY src/analysis/1d_gradients.py --param diffusion_trans_cm2_us --noise-scale 0.0 --noise-seed 42 $COMMON
$PY src/analysis/1d_gradients.py --param diffusion_trans_cm2_us --noise-scale 1.0 --noise-seed 42 $COMMON
$PY src/analysis/1d_gradients.py --param diffusion_long_cm2_us  --noise-scale 0.0 --noise-seed 42 $COMMON
$PY src/analysis/1d_gradients.py --param diffusion_long_cm2_us  --noise-scale 1.0 --noise-seed 42 $COMMON

# Sobolev exponent sweep (s=0.25,0.5,0.75; adc-cutoffs=0 only to avoid combinatorial explosion)
COMMON_S="--tracks $TRACKS --factors 0.75,1,1.25 --step-size 1.0 --max-deposits 5000 --sobolev-max-pad 128 --results-dir $RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260526_small --adc-cutoffs 0 --store-per-plane-loss --store-per-pixel-loss-and-grad --store-arrays --save-per-factor"

#$PY src/analysis/1d_gradients.py --param diffusion_trans_cm2_us --noise-scale 0.0 --noise-seed 42 --sobolev-s 0.25,0.5,0.75 $COMMON_S
#$PY src/analysis/1d_gradients.py --param diffusion_trans_cm2_us --noise-scale 1.0 --noise-seed 42 --sobolev-s 0.25,0.5,0.75 $COMMON_S
#$PY src/analysis/1d_gradients.py --param diffusion_long_cm2_us  --noise-scale 0.0 --noise-seed 42 --sobolev-s 0.25,0.5,0.75 $COMMON_S
#$PY src/analysis/1d_gradients.py --param diffusion_long_cm2_us  --noise-scale 1.0 --noise-seed 42 --sobolev-s 0.25,0.5,0.75 $COMMON_S

# Fourier-space cutoff sweep (fc=1,10,100; adc-cutoffs=0 only)
$PY src/analysis/1d_gradients.py --param diffusion_trans_cm2_us --noise-scale 0.0 --noise-seed 42 --fourier-cutoffs 1,10,100 $COMMON_S
$PY src/analysis/1d_gradients.py --param diffusion_trans_cm2_us --noise-scale 1.0 --noise-seed 42 --fourier-cutoffs 1,10,100 $COMMON_S
$PY src/analysis/1d_gradients.py --param diffusion_long_cm2_us  --noise-scale 0.0 --noise-seed 42 --fourier-cutoffs 1,10,100 $COMMON_S
$PY src/analysis/1d_gradients.py --param diffusion_long_cm2_us  --noise-scale 1.0 --noise-seed 42 --fourier-cutoffs 1,10,100 $COMMON_S

# Generate HTML viewer for all pkls in the results directory
$PY src/analysis/sim_param_sweeps/generate_gradient_viewer.py --dir $RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260526_small
