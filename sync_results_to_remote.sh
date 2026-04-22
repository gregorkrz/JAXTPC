#!/usr/bin/env bash
# Sync plots/ and results/ to s3://gregor-research/JAXTPC/
set -euo pipefail

# Load .env if present
if [[ -f .env ]]; then
  set -o allexport
  source .env
  set +o allexport
fi

BUCKET="s3://gregor-research/JAXTPC"
RESULTS_DIR="${RESULTS_DIR:-results}"
PLOTS_DIR="${PLOTS_DIR:-plots}"

echo "Syncing $PLOTS_DIR/ ..."
aws s3 sync "$PLOTS_DIR/" "$BUCKET/plots/"

echo "Syncing $RESULTS_DIR/ ..."
aws s3 sync "$RESULTS_DIR/" "$BUCKET/results/"

echo "Done. Files available at:"
echo "  https://s3.console.aws.amazon.com/s3/buckets/gregor-research?prefix=JAXTPC/"
