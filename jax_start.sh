#!/usr/bin/env bash
# Ensure the jaxtpc container is running (detached). Idempotent.

IMAGE="docker.io/gkrz/jaxtpc:v1"
NAME="jaxtpc_v1"
WORKSPACE="$HOME"

if podman-hpc container exists "$NAME"; then
    if [ "$(podman-hpc inspect -f '{{.State.Running}}' $NAME)" != "true" ]; then
        echo "Starting existing container $NAME..."
        podman-hpc start "$NAME"
    else
        echo "Container $NAME already running."
    fi
else
    echo "Creating and starting container $NAME (detached)..."
    podman-hpc run -d --name "$NAME" --shm-size=16g --gpu \
        -v "${WORKSPACE}:/workspace" \
        -v /global/cfs/cdirs/m3246/gregork:/global/cfs/cdirs/m3246/gregork \
        -v /pscratch/sd/g/gregork:/pscratch/sd/g/gregork \
        "$IMAGE" sleep infinity
fi
