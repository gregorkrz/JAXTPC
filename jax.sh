#!/usr/bin/env bash

IMAGE="docker.io/gkrz/jaxtpc:v0"
NAME="jaxtpc"
WORKSPACE="$HOME"

if [ -n "$1" ]; then
    # Non-interactive: run a script file inside the container.
    # $1 is a host path (e.g. ~/.jaxtpc_tmp/gpu_0.sh).
    # Map $HOME/... → /workspace/... for the container filesystem.
    HOST_SCRIPT="$1"
    CONTAINER_SCRIPT="/workspace${HOST_SCRIPT#$HOME}"

    if podman-hpc container exists "$NAME"; then
        if [ "$(podman-hpc inspect -f '{{.State.Running}}' $NAME)" != "true" ]; then
            podman-hpc start "$NAME"
        fi
        podman-hpc exec "$NAME" /bin/bash "$CONTAINER_SCRIPT"
    else
        podman-hpc run --name "$NAME" --shm-size=16g --gpu \
            -v "${WORKSPACE}:/workspace" \
            -v /global/cfs/cdirs/m3246/gregork:/global/cfs/cdirs/m3246/gregork \
            -v /pscratch/sd/g/gregork:/pscratch/sd/g/gregork \
            "$IMAGE" /bin/bash "$CONTAINER_SCRIPT"
    fi
else
    # Interactive mode (unchanged).
    if podman-hpc container exists "$NAME"; then
        echo "🔁 Container $NAME already exists."
        if [ "$(podman-hpc inspect -f '{{.State.Running}}' $NAME)" != "true" ]; then
            echo "   Starting container..."
            podman-hpc start "$NAME"
        fi
        echo "🐚 Attaching shell..."
        podman-hpc exec -it "$NAME" /bin/bash
    else
        echo "🚀 Creating new container $NAME ..."
        podman-hpc run -it --name "$NAME" --shm-size=16g --gpu \
            -v "${WORKSPACE}:/workspace" \
            -v /global/cfs/cdirs/m3246/gregork:/global/cfs/cdirs/m3246/gregork \
            -v /pscratch/sd/g/gregork:/pscratch/sd/g/gregork \
            "$IMAGE" /bin/bash
    fi
fi
