#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=results/efield_distortions

TOY_NPZ=${OUT_DIR}/sce_maps_jaxtpc_toy_41.npz
python -m tools.efield_distortions \
    --detector jaxtpc \
    --model toy \
    --Nxo 41 --Nyo 41 --Nzo 41 \
    --output "${TOY_NPZ}"
python scripts/20260601/plot_efield_distortions.py "${TOY_NPZ}"

CONS_NPZ=${OUT_DIR}/sce_maps_jaxtpc_conservative_41.npz
python -m tools.efield_distortions \
    --detector jaxtpc \
    --model conservative \
    --Nxo 41 --Nyo 41 --Nzo 41 \
    --output "${CONS_NPZ}"
python scripts/20260601/plot_efield_distortions.py "${CONS_NPZ}"
