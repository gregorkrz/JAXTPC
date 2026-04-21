#!/usr/bin/env bash
# Optimize recomb_alpha + recomb_beta_90 over four track configurations.
#
# Track configs:
#   1. track2(1000 MeV) + diagonal(1000 MeV)
#   2. diagonal(1000 MeV) + Z(1000 MeV)
#   3. track2(100 MeV) only
#   4. track2(100 MeV) + track2(1000 MeV)
#
# Seeds start at 42 and increment by 1 per config (result_<seed>.pkl).
set -euo pipefail

PY=.venv/bin/python
SCRIPT=src/opt/run_optimization.py

# ── Shared settings ──────────────────────────────────────────────────────────
PARAMS="recomb_alpha,recomb_beta_90"
RANGE="0.95 1.05"
OPTIMIZER="adam"
LR="0.001"
LR_SCHEDULE="constant"
MAX_STEPS="200"
LOSS="sobolev_loss_geomean_log1p"
N="5"

# ── Track specs ──────────────────────────────────────────────────────────────
# Presets (diagonal, Z) are at 1000 MeV by default.
# Custom: name:dx,dy,dz:momentum_mev  ('+' separates tracks)
TRACK2_DIR="0.5,1.05,0.2"

CONFIGS=(
    "track2_1000:${TRACK2_DIR}:1000+diagonal"
    "diagonal+Z"
    "track2_100:${TRACK2_DIR}:100"
    "track2_100:${TRACK2_DIR}:100+track2_1000:${TRACK2_DIR}:1000"
)

LABELS=(
    "track2(1000 MeV) + diagonal(1000 MeV)"
    "diagonal(1000 MeV) + Z(1000 MeV)"
    "track2(100 MeV)"
    "track2(100 MeV) + track2(1000 MeV)"
)

# ── Run ──────────────────────────────────────────────────────────────────────
TOTAL=${#CONFIGS[@]}
START_SEED=42

for i in "${!CONFIGS[@]}"; do
    SEED=$((START_SEED + i))
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Job $((i+1))/${TOTAL}: ${LABELS[$i]}  (seed=${SEED})"
    echo "════════════════════════════════════════════════════════════"

    $PY "$SCRIPT" \
        --params       "$PARAMS" \
        --range        $RANGE \
        --tracks       "${CONFIGS[$i]}" \
        --loss         "$LOSS" \
        --optimizer    "$OPTIMIZER" \
        --lr           "$LR" \
        --lr-schedule  "$LR_SCHEDULE" \
        --max-steps    "$MAX_STEPS" \
        --N            "$N" \
        --seed         "$SEED" \
        --results-base "results/opt"
done

echo ""
echo "All ${TOTAL} jobs done."
