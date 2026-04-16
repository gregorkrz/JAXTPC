#!/usr/bin/env bash
# Run 1d_opt.py for every combination of:
#   - learning rate             (0.1, 0.01, 0.001)
#   - optimizer                 (adam, sgd, momentum_sgd)
#   - optimisable parameter     (velocity, lifetime, diffusion, recombination)
#   - muon track direction      (diagonal, X, Y, Z, U, V)
# with N=3 starting points on each side of GT, M=3 trials per starting point.
#
# Each LR gets its own sub-directory: results/1d_opt/lr_<LR>/
# Jobs whose output pkl files already exist are skipped automatically.
#
# U and V directions are parallel to the U/V wire planes:
#   Y wires run along z  →  U = +60° from z in y-z plane
#                           V = -60° from z in y-z plane
#
# Usage: bash run_1d_opt.sh

set -euo pipefail

N=3
M=3
MAX_STEPS=200
BASE_RESULTS_DIR="results/1d_opt"
LOSSES_DEFAULT="sobolev_loss,sobolev_loss_geomean_log1p"
LOSSES_RECOMB_R="sobolev_loss,sobolev_loss_geomean_log1p"

# ── sweep definitions ───────────────────────────────────────────────────────
LRS=("0.1" "0.01" "0.001")
OPTIMIZERS=("adam" "sgd" "momentum_sgd")

# ── direction definitions ───────────────────────────────────────────────────
declare -A DIRECTIONS
DIRECTIONS["diagonal"]="1,1,1"
DIRECTIONS["X"]="1,0,0"
DIRECTIONS["Y"]="0,1,0"
DIRECTIONS["Z"]="0,0,1"
DIRECTIONS["U"]="0,0.866,0.5"     # +60° from z in y-z plane
DIRECTIONS["V"]="0,-0.866,0.5"    # -60° from z in y-z plane

TRACK_ORDER=("diagonal" "X" "Y" "Z" "U" "V")

# ── parameter definitions ───────────────────────────────────────────────────
# recomb_beta → modified_box model only; use recomb_beta_90 / recomb_R for emb
PARAMS=(
    "velocity_cm_us"
    "lifetime_us"
    "diffusion_trans_cm2_us"
    "diffusion_long_cm2_us"
    "recomb_alpha"
    "recomb_beta_90"    # emb model (default); swap for recomb_beta if using modified_box
    "recomb_R"          # emb model (default)
)

# ── skip helper ─────────────────────────────────────────────────────────────
# Returns 0 (skip) if all expected pkl files already exist, 1 (run) otherwise.
# Args: results_dir optimizer param track
all_pkls_exist() {
    local results_dir=$1
    local optimizer=$2
    local param=$3
    local track=$4
    local losses="sobolev_loss sobolev_loss_geomean_log1p"
    for loss in $losses; do
        local pkl="${results_dir}/${loss}_N${N}_M${M}_${optimizer}_${param}_${track}.pkl"
        if [[ ! -f "$pkl" ]]; then
            return 1
        fi
    done
    return 0
}

# ── run ─────────────────────────────────────────────────────────────────────
TOTAL=$(( ${#LRS[@]} * ${#OPTIMIZERS[@]} * ${#PARAMS[@]} * ${#TRACK_ORDER[@]} ))
JOB=0
SKIPPED=0

for LR in "${LRS[@]}"; do
    RESULTS_DIR="${BASE_RESULTS_DIR}/lr_${LR}"
    for OPTIMIZER in "${OPTIMIZERS[@]}"; do
        for PARAM in "${PARAMS[@]}"; do
            for TRACK in "${TRACK_ORDER[@]}"; do
                JOB=$(( JOB + 1 ))
                DIR="${DIRECTIONS[$TRACK]}"

                if all_pkls_exist "$RESULTS_DIR" "$OPTIMIZER" "$PARAM" "$TRACK"; then
                    echo "  [${JOB}/${TOTAL}] SKIP  lr=${LR}  optimizer=${OPTIMIZER}  param=${PARAM}  track=${TRACK}  (pkl files exist)"
                    SKIPPED=$(( SKIPPED + 1 ))
                    continue
                fi

                echo ""
                echo "════════════════════════════════════════════════════════════"
                echo "  Job ${JOB}/${TOTAL}:  lr=${LR}  optimizer=${OPTIMIZER}  param=${PARAM}  track=${TRACK}  dir=(${DIR})"
                echo "════════════════════════════════════════════════════════════"
                LOSSES="${LOSSES_DEFAULT}"
                if [[ "${PARAM}" == "recomb_R" ]]; then
                    LOSSES="${LOSSES_RECOMB_R}"
                fi

                python 1d_opt.py \
                    --param        "${PARAM}" \
                    --optimizer    "${OPTIMIZER}" \
                    --lr           "${LR}" \
                    --max-steps    "${MAX_STEPS}" \
                    --N            "${N}" \
                    --M            "${M}" \
                    --track-name   "${TRACK}" \
                    --direction    "${DIR}" \
                    --results-dir  "${RESULTS_DIR}" \
                    --loss         "${LOSSES}"
            done
        done
    done
done

echo ""
echo "Done.  ${TOTAL} jobs total,  ${SKIPPED} skipped,  $(( TOTAL - SKIPPED )) run."
echo "Results in ${BASE_RESULTS_DIR}/lr_*/"
