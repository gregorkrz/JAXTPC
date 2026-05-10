#!/usr/bin/env bash
# W&B trajectory plots (2026-05-05): grads / physical / normalized params per tag.
#
# Tags below are the *unique* profile tags among the 20 most recently *created* runs in
# fcc_ml/jaxtpc-optimization (W&B API order -createdAt, snapshot 2026-05-05 UTC).
#
# From repo root:
#   ./scripts/plots_05052026.sh
#
# Requires W&B login / WANDB_API_KEY and src/plots/run_results.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
PLOT="${ROOT}/src/plots/run_results.py"
TAGS="tracks12_mixed_cos30k_nosched_auto_clip1_noparamfreeze,tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1_p500_ebs12,tracks12_mixed_cos30k_auto_clip1_p500_no_vel,tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1_p500,tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1,tracks24_mixed_cos30k_nosched_tol1e4_p2000,tracks12_mixed_cos30k_nosched_tol1e4_p2000"
OUT_DIR="${PLOTS_DIR:-${ROOT}/plots}/wandb_run_results_05052026"

echo "Writing plots under: ${OUT_DIR}"
"${PY}" "${PLOT}" \
  --tags "${TAGS}" \
  --tag-label "12 trk, cos30k: no per-param freeze; auto LR; grad clip 1" \
  --tag-label "12 trk, cos30k: per-param freeze p500; auto LR; clip 1; eff. batch 12" \
  --tag-label "12 trk, cos30k: freeze vel.; 6 params; auto LR; per-param p500" \
  --tag-label "12 trk, cos30k: per-param tol1e-4, patience 500; auto LR; clip 1" \
  --tag-label "12 trk, cos30k: per-param tol1e-4, patience 2000; auto LR; clip 1" \
  --tag-label "24 trk (12 + x-flip): per-param tol1e-4, patience 2000" \
  --tag-label "12 trk, cos30k: per-param tol1e-4, patience 2000 (no auto LR)" \
  --output-dir "${OUT_DIR}" \
  -v

echo "Done."
