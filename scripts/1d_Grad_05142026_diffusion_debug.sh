#!/usr/bin/env bash
# 1D gradient sweeps for diffusion coefficients — 2026-05-14
#
# Submits 4 Slurm jobs to S3DF (2 params × noise/nonoise).
# Each job: ±50% range, N=10 (21 pts), 15-track boundary ensemble, 64 GB.
#
# Run from the JAXTPC repo root on S3DF:
#   bash scripts/1d_Grad_05142026_diffusion_debug.sh

/sdf/home/g/gregork/envs/base_env/bin/python src/jobs/submit_jobs.py 1d_Grad_diffusion_debug --submit
