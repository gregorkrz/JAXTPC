#!/usr/bin/env bash
# Run 1d_gradients.py for every combination of:
#   - optimisable parameter  (velocity, lifetime, diffusion, recombination)
#   - muon track direction   (diagonal, X, Y, Z, U, V)
# with N=5 evaluation points on each side of the ground truth.
#
# Jobs whose output pkl files already exist are skipped automatically.
#
# U and V directions are parallel to the U/V wire planes:
#   Y wires run along z  →  U = +60° from z in y-z plane
#                           V = -60° from z in y-z plane
#
# Usage: bash run_1d_gradients.sh

set -euo pipefail

N=5
RESULTS_DIR="results/1d_gradients"
LOSSES_DEFAULT="sobolev_loss,sobolev_loss_geomean_log1p"
# recomb_R also runs MSE to compare a simpler baseline loss
LOSSES_RECOMB_ALPHA="sobolev_loss,sobolev_loss_geomean_log1p,mse_loss"

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
# recomb_beta   → modified_box model only; use recomb_beta_90 / recomb_R for emb
PARAMS=(
    "velocity_cm_us"
    "lifetime_us"
    "diffusion_trans_cm2_us"
    "diffusion_long_cm2_us"
    "recomb_R"
    "recomb_beta_90"    # emb model (default); swap for recomb_beta if using modified_box
    "recomb_R"          # emb model (default)
)

# ── skip helper ─────────────────────────────────────────────────────────────
# Returns 0 (skip) if all expected pkl files already exist, 1 (run) otherwise.
all_pkls_exist() {
    local param=$1
    local track=$2
    local losses="sobolev_loss sobolev_loss_geomean_log1p"
    if [[ "$param" == "recomb_R" ]]; then
        losses="sobolev_loss sobolev_loss_geomean_log1p mse_loss"
    fi
    for loss in $losses; do
        local pkl="${RESULTS_DIR}/${loss}_N${N}_${param}_${track}.pkl"
        if [[ ! -f "$pkl" ]]; then
            return 1
        fi
    done
    return 0
}

# ── run ─────────────────────────────────────────────────────────────────────
TOTAL=$(( ${#PARAMS[@]} * ${#TRACK_ORDER[@]} ))
JOB=0
SKIPPED=0

for PARAM in "${PARAMS[@]}"; do
    for TRACK in "${TRACK_ORDER[@]}"; do
        JOB=$(( JOB + 1 ))
        DIR="${DIRECTIONS[$TRACK]}"

        if all_pkls_exist "$PARAM" "$TRACK"; then
            echo "  [${JOB}/${TOTAL}] SKIP  param=${PARAM}  track=${TRACK}  (pkl files exist)"
            SKIPPED=$(( SKIPPED + 1 ))
            continue
        fi

        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "  Job ${JOB}/${TOTAL}:  param=${PARAM}  track=${TRACK}  dir=(${DIR})"
        echo "════════════════════════════════════════════════════════════"
        LOSSES="${LOSSES_DEFAULT}"
        if [[ "${PARAM}" == "recomb_R" ]]; then
            LOSSES="${LOSSES_RECOMB_ALPHA}"
        fi

        python src/analysis/1d_gradients.py \
            --param        "${PARAM}" \
            --N            "${N}" \
            --track-name   "${TRACK}" \
            --direction    "${DIR}" \
            --results-dir  "${RESULTS_DIR}" \
            --loss         "${LOSSES}"
    done
done

echo ""
echo "Done.  ${TOTAL} jobs total,  ${SKIPPED} skipped,  $(( TOTAL - SKIPPED )) run."
echo "Results in ${RESULTS_DIR}/"
python src/plots/1d_gradients_plots.py --N 5 --track-name diagonal,X,Y,Z,U,V
