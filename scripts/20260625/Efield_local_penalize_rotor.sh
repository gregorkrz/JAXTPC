#!/usr/bin/env bash
# Local E-field calibration using the SIREN distortion model (tools/distortion.py).
#
# GT simulator uses the pre-generated toy SCE map (41x41x41 grid per side).
# Diff simulator learns a SIREN Δ(r) field that reproduces the GT waveforms.
# E is derived from Δ via ∂Δ/∂x + Walkowiak inversion (tools/sce_siren.py).
#
# Output: $RESULTS_DIR/opt/efield_siren_local/
set -euo pipefail

cd /home/gregor/JAXTPC
source .env

PY=.venv/bin/python

$PY src/opt/run_optimization.py \
  --params Efield \
  --electric-dist-path results/efield_distortions/sce_maps_jaxtpc_41.npz \
  --N-random-tracks 15 \
  --tracks-random-seed 42 \
  --seed 0 \
  --N 1 \
  --optimizer adam \
  --lr 1e-3 \
  --lr-schedule cosine \
  --adam-beta2 0.9 \
  --warmup-steps 200 \
  --max-steps 5000 \
  --tol 1e-9 \
  --patience 5000 \
  --loss sobolev_loss_geomean_log1p \
  --sobolev-exponent 2.0 \
  --max-num-deposits 5000 \
  --num-buckets 1000 \
  --gt-max-deposits 5000 \
  --gt-step-size 1.0 \
  --step-size 1.0 \
  --batch-size 1 \
  --effective-batch-size 15 \
  --efield-hidden 32 32 32 \
  --efield-lr-mult 10.0 \
  --noise-scale 1.0 \
  --clip-grad-norm 1.0 \
  --log-interval 10 \
  --results-base "$RESULTS_DIR/opt/efield_siren_local_penalize_rotor" \
  --wandb-tags efield_siren,local,siren \
  --penalize-rotor 1.0

