#!/bin/bash

# Default to pm if no argument provided
MODE="${1:-pm}"
REMOTE_PATH="pm:/global/homes/g/gregork/jaxtpc"

# Validate mode
if [[ "$MODE" != "pm" && "$MODE" != "slac" ]]; then
    echo "Error: Invalid mode '$MODE'. Use 'pm' or 'slac'."
    exit 1
fi

# Check if slac is requested
if [[ "$MODE" == "slac" ]]; then
    echo "Error: slac mode is not supported yet."
    exit 1
fi

# Sync to pm remote
if [[ "$MODE" == "pm" ]]; then
    echo "Syncing to $REMOTE_PATH ..."
    rsync -av --progress --stats \
        --exclude="__pycache__" \
        --exclude=".venv" \
        --exclude="plots" \
        --exclude=".env" \
        --exclude="results" \
        --exclude ".git" \
        . "$REMOTE_PATH"
    echo "Done."
fi
