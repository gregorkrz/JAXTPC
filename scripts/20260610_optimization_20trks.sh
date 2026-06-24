# Run on S3DF: trajectory plots for the Run_Opt_20260609 20-track optimization jobs.
#
# Usage: bash scripts/20260610_optimization_20trks.sh [target] [--no-pdfs]
#   target one of: all_params | adc50 | trans_and_long | index | all (default)
#   --no-pdfs: only (re)generate the HTML, skip the PDF outputs

PY=/sdf/home/g/gregork/envs/base_env/bin/python
OUT="$PLOTS_DIR/trajectories_Run_Opt_20260609"
TARGET="all"
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --no-pdfs) EXTRA_ARGS+=(--no-pdf) ;;
    *)         TARGET="$arg" ;;
  esac
done

run_all_params() {
  # all_params (nominal GT; includes adc50 / adc50_D_only / phase2_diffusion sub-variants)
  $PY tools/plot_opt_trajectories.py \
    --tag all_params \
    --output-dir "$OUT" "${EXTRA_ARGS[@]}"
}

run_adc50() {
  # all_params, adc50 + adc50_D_only only (drops phase2_diffusion -> smaller/faster page)
  # also drops the stale GT group velocity=0.16, lifetime=1e+04, diffusion=1.2e-05 (40 runs)
  $PY tools/plot_opt_trajectories.py \
    --tag all_params \
    --exclude-variant phase2_diffusion \
    --exclude-gt-label "velocity=0.16, lifetime=1e+04, diffusion=1.2e-05" \
    --out-tag all_params_adc50 \
    --output-dir "$OUT" "${EXTRA_ARGS[@]}"
}

run_trans_and_long() {
  # "both D constants" optimization (trans_and_long; GTs: nominal + gt80pct)
  $PY tools/plot_opt_trajectories.py \
    --tag trans_and_long \
    --output-dir "$OUT" "${EXTRA_ARGS[@]}"
}

run_index() {
  # index.html linking the trajectory pages above
  $PY tools/generate_trajectories_index.py \
    --output-dir "$OUT"
}

case "$TARGET" in
  all_params)     run_all_params ;;
  adc50)          run_adc50 ;;
  trans_and_long) run_trans_and_long ;;
  index)          run_index ;;
  all)
    run_all_params
    run_adc50
    run_trans_and_long
    run_index
    ;;
  *)
    echo "Unknown target '$TARGET'. Expected: all_params | adc50 | trans_and_long | index | all" >&2
    exit 1
    ;;
esac
