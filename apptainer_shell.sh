#!/bin/bash
# Request an interactive GPU session on S3DF and drop into the Apptainer container shell.
# Usage: bash apptainer_shell.sh

srun \
    --account=neutrino \
    --partition=ampere \
    --time=04:00:00 \
    --gpus=1 \
    --cpus-per-gpu=1 \
    --mem=64G \
    --pty \
    bash -c '
        mkdir -p /sdf/scratch/atlas/gregork/jax_cache
        export APPTAINER_CACHEDIR=/sdf/scratch/atlas/gregork/apptainer_cache
        export APPTAINER_TMPDIR=/sdf/scratch/atlas/gregork/apptainer_tmp
        export JAX_COMPILATION_CACHE_DIR=/sdf/scratch/atlas/gregork/jax_cache
        cd /sdf/home/g/gregork/jaxtpc
        source .env
        apptainer shell --nv \
            --bind /sdf/home/g/gregork/jaxtpc \
            --bind /fs/ddn/sdf/group/atlas/d/gregork/jaxtpc \
            --bind /sdf/scratch/atlas/gregork/jax_cache \
            docker://gkrz/jaxtpc:v2
    '
