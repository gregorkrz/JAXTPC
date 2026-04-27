#!/usr/bin/env bash
# Sync a results subfolder from S3 to the local results/ directory.
# Usage: bash from_remote_sync.sh results/opt/2d_recomb
set -euo pipefail

PDF_ONLY=0
POSITIONAL=()
for arg in "$@"; do
  if [[ "$arg" == "--pdf" ]]; then
    PDF_ONLY=1
  else
    POSITIONAL+=("$arg")
  fi
done

if [[ ${#POSITIONAL[@]} -ne 1 ]]; then
  echo "Usage: $0 <path> [--pdf]   (e.g. results/opt/2d_recomb)" >&2
  exit 1
fi

if [[ -f .env ]]; then
  set -o allexport
  source .env
  set +o allexport
fi

BUCKET="s3://gregor-research/JAXTPC"
PATH_ARG="${POSITIONAL[0]%/}"   # strip trailing slash

if [[ $PDF_ONLY -eq 1 ]]; then
  aws s3 sync "$BUCKET/$PATH_ARG/" "$PATH_ARG/" --exclude "*" --include "*.pdf"
else
  aws s3 sync "$BUCKET/$PATH_ARG/" "$PATH_ARG/"
fi
