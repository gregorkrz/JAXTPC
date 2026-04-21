#!/usr/bin/env bash
# Regenerate plots (no computation).
#
# Usage:
#   ./plots.sh                          # redo everything
#   ./plots.sh --1d-opt                 # only 1d_opt
#   ./plots.sh --2d-opt                 # only 2d_opt
#   ./plots.sh --1d-gradients           # only 1d_gradients
#   ./plots.sh --1d-opt --2d-opt        # multiple groups
set -euo pipefail

PY=.venv/bin/python

DO_1D_GRADIENTS=0
DO_1D_OPT=0
DO_2D_OPT=0

if [ $# -eq 0 ]; then
    DO_1D_GRADIENTS=1
    DO_1D_OPT=1
    DO_2D_OPT=1
else
    for arg in "$@"; do
        case "$arg" in
            --1d-gradients) DO_1D_GRADIENTS=1 ;;
            --1d-opt)       DO_1D_OPT=1 ;;
            --2d-opt)       DO_2D_OPT=1 ;;
            *) echo "Unknown flag: $arg"; echo "Valid flags: --1d-gradients  --1d-opt  --2d-opt"; exit 1 ;;
        esac
    done
fi

# ── 1d_gradients ──────────────────────────────────────────────────────────────
if [ $DO_1D_GRADIENTS -eq 1 ]; then
    echo "=========================================="
    echo "  1d_gradients plots"
    echo "=========================================="
    for N in 3 5 10; do
        echo "--- N=$N ---"
        $PY 1d_gradients_plots.py \
            --N "$N" \
            --track-name diagonal,U,V,X,Y,Z \
            --results-dir results/1d_gradients \
            --output-dir results/1d_gradients
    done
fi

# ── 1d_opt ────────────────────────────────────────────────────────────────────
if [ $DO_1D_OPT -eq 1 ]; then
    echo ""
    echo "=========================================="
    echo "  1d_opt plots  (root dir, N=3 M=3)"
    echo "=========================================="
    $PY 1d_opt_plots.py \
        --N 3 --M 3 \
        --track-name diagonal \
        --results-dir results/1d_opt \
        --output-dir results/1d_opt

    echo ""
    echo "=========================================="
    echo "  1d_opt plots  (lr subdirs)"
    echo "=========================================="
    for LR in 0.1 0.01 0.001 0.0001; do
        DIR="results/1d_opt/lr_${LR}"
        if [ ! -d "$DIR" ]; then
            echo "Skipping $DIR (not found)"
            continue
        fi
        echo "--- lr=$LR ---"
        for N in 1 3 5; do
            for M in 1 3 5; do
                if ls "$DIR"/*_N${N}_M${M}_*.pkl 2>/dev/null | grep -q .; then
                    echo "  N=$N M=$M"
                    $PY 1d_opt_plots.py \
                        --N "$N" --M "$M" \
                        --track-name diagonal \
                        --results-dir "$DIR" \
                        --output-dir "$DIR"
                fi
            done
        done
    done
fi

# ── 2d_opt ────────────────────────────────────────────────────────────────────
if [ $DO_2D_OPT -eq 1 ]; then
    echo ""
    echo "=========================================="
    echo "  2d_opt plots"
    echo "=========================================="
    $PY 2d_opt_plots.py \
        --track-name diagonal \
        --results-dir results/2d_opt \
        --output-dir results/2d_opt
fi

echo ""
echo "All plots done."
