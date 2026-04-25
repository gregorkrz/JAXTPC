#!/usr/bin/env bash
# Sync a results subfolder from S3 to the local results/ directory.
# Usage: bash from_remote_sync.sh results/opt/2d_recomb
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <path>   (e.g. results/opt/2d_recomb)" >&2
  exit 1
fi

if [[ -f .env ]]; then
  set -o allexport
  source .env
  set +o allexport
fi

BUCKET="s3://gregor-research/JAXTPC"
PATH_ARG="${1%/}"   # strip trailing slash

aws s3 sync "$BUCKET/$PATH_ARG/" "$PATH_ARG/"
