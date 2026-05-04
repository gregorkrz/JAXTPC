#!/bin/bash
# Open an interactive Apptainer shell directly (no srun/job request).
# Usage: bash apptainer_shell_CPU.sh

mkdir -p /sdf/scratch/atlas/gregork/jax_cache
export APPTAINER_CACHEDIR=/sdf/scratch/atlas/gregork/apptainer_cache
export APPTAINER_TMPDIR=/sdf/scratch/atlas/gregork/apptainer_tmp
export JAX_COMPILATION_CACHE_DIR=/sdf/scratch/atlas/gregork/jax_cache

cd /sdf/home/g/gregork/jaxtpc || exit 1
source .env

apptainer shell \
    --bind /sdf/home/g/gregork/jaxtpc \
    --bind /fs/ddn/sdf/group/atlas/d/gregork/jaxtpc \
    --bind /sdf/scratch/atlas/gregork/jax_cache \
    docker://gkrz/jaxtpc:v2
