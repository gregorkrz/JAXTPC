#!/usr/bin/env bash
# W&B trajectory plots (2026-05-06): compare no-param-freeze with vs without velocity.
#
# From repo root:
#   ./scripts/plots_06052026.sh
#
# Requires W&B login / WANDB_API_KEY and src/plots/run_results.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
PLOT="${ROOT}/src/plots/run_results.py"
TAGS="tracks12_mixed_cos30k_nosched_auto_clip1_noparamfreeze_no_vel,tracks12_mixed_cos30k_nosched_auto_clip1_noparamfreeze"
OUT_DIR="${PLOTS_DIR:-${ROOT}/plots}/wandb_run_results_06052026"

echo "Writing plots under: ${OUT_DIR}"
"${PY}" "${PLOT}" \
  --tags "${TAGS}" \
  --tag-label "12 trk, cos30k: no per-param freeze; auto LR; clip 1; no velocity param" \
  --tag-label "12 trk, cos30k: no per-param freeze; auto LR; clip 1; includes velocity" \
  --output-dir "${OUT_DIR}" \
  -v

echo "Done."
