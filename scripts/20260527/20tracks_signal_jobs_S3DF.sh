#!/bin/bash
# Signal-array 1d-gradient jobs for all 20 tracks, sweeping Sobolev exponents.
# Run from the repo root on S3DF (RESULTS_DIR must be set).

#/sdf/home/g/gregork/envs/base_env/bin/python src/jobs/submit_jobs.py gradient_signal_viewer_20trk_sobolev_exp_1 --submit --skip-complete
#/sdf/home/g/gregork/envs/base_env/bin/python src/jobs/submit_jobs.py gradient_signal_viewer_20trk_sobolev_exp_0 --submit --skip-complete
#/sdf/home/g/gregork/envs/base_env/bin/python src/jobs/submit_jobs.py gradient_signal_viewer_20trk --submit --skip-complete
/sdf/home/g/gregork/envs/base_env/bin/python src/jobs/submit_jobs.py gradient_signal_viewer_20trk_sobolev_exp_1_denser --submit --skip-complete
/sdf/home/g/gregork/envs/base_env/bin/python src/jobs/submit_jobs.py gradient_signal_viewer_20trk_sobolev_exp_0_denser --submit --skip-complete
/sdf/home/g/gregork/envs/base_env/bin/python src/jobs/submit_jobs.py gradient_signal_viewer_20trk_denser --submit --skip-complete
