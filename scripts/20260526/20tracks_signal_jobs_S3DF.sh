#!/usr/bin/env bash
# Submit 1d-gradient signal-array jobs for all 20 tracks to Slurm.
# Outputs land in $RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260526/
#
# Run on S3DF login node (not sdfiana):
#   bash 20tracks_signal_jobs_S3DF.sh
#
# After jobs finish, run 20tracks_displays_S3DF.sh to generate HTML viewers.
set -euo pipefail

cd /sdf/home/g/gregork/jaxtpc
source .env

/sdf/home/g/gregork/envs/base_env/bin/python src/jobs/submit_jobs.py \
    gradient_signal_viewer_20trk \
    --submit
