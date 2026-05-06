#!/usr/bin/env bash
# W&B trajectory plots (2026-05-04 batch): grads / physical / normalized params per tag.
#
# From repo root:
#   ./scripts/plots_05042026.sh
#
# Requires W&B login / WANDB_API_KEY and src/plots/run_results.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
PLOT="${ROOT}/src/plots/run_results.py"
TAGS="fine_nosched_bs1_tol1e4_p2000,tracks12_mixed_cos30k_nosched,tracks12_mixed_cos30k_nosched_tol1e4_p2000"
OUT_DIR="${PLOTS_DIR:-${ROOT}/plots}/wandb_run_results_05042026"

echo "Writing plots under: ${OUT_DIR}"
"${PY}" "${PLOT}" \
  --tags "${TAGS}" \
  --tag-label "0.1mm, 12 tracks w nice directions, early freezing" \
  --tag-label "0.1mm, 12 tracks" \
  --tag-label "0.1mm, 12 tracks, early freezing" \
  --output-dir "${OUT_DIR}" \
  -v

echo "Done."
