#!/usr/bin/env bash
# Sync plots/ and results/ to s3://gregor-research/JAXTPC/
set -euo pipefail

BUCKET="s3://gregor-research/JAXTPC"

echo "Syncing plots/ ..."
aws s3 sync plots/ "$BUCKET/plots/"

echo "Syncing results/ ..."
aws s3 sync results/ "$BUCKET/results/"

echo "Done. Files available at:"
echo "  https://s3.console.aws.amazon.com/s3/buckets/gregor-research?prefix=JAXTPC/"
