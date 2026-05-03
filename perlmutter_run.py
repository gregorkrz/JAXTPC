#!/usr/bin/env python3
"""
Execute commands on Perlmutter GPUs in parallel inside an salloc session.

Workflow:
  1. Sync code to Perlmutter (sync_code.sh)
  2. Write per-GPU shell scripts to ~/.jaxtpc_tmp/ on Perlmutter (shared FS)
  3. Submit an salloc job that:
       - resolves the allocated compute node ($SLURM_NODELIST)
       - SSHes to that node in parallel, one connection per GPU
       - each connection runs: sh ~/jax.sh <gpu_script_path>  (no TTY needed)
       - waits for all GPUs to finish
  4. Clean up temp scripts
  5. Run sync_results_to_remote.sh on Perlmutter login node → S3

Plan file (YAML):
  time_limit: "01:00:00"    # salloc wall-clock limit (HH:MM:SS)
  command_timeout: 20        # per-command timeout in minutes (default 20)
  gpus:
    0:
      - python src/analysis/2d_loss_landscape.py ...
      - python src/analysis/2d_loss_landscape.py ...
    1:
      - python src/analysis/...
    # omit or leave empty to skip a GPU

Usage:
  python perlmutter_run.py plan.yaml
  python perlmutter_run.py plan.yaml --dry-run
  python perlmutter_run.py plan.yaml --no-sync-code --no-sync-results
"""
import argparse
import os
import subprocess
import time

import yaml

REMOTE        = "pm"
REMOTE_DIR    = "/global/homes/g/gregork/jaxtpc"   # host path (for rsync, SSH outside container)
CONTAINER_DIR = "/workspace/jaxtpc"                # same path inside container ($HOME → /workspace)
REMOTE_TMP    = "$HOME/.jaxtpc_tmp"                # shared FS, visible from compute nodes + container
SLURM_ACCOUNT = "m3246"


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(msg, tag="[main]"):
    print(f"{tag} {msg}", flush=True)


# ── Script builders ───────────────────────────────────────────────────────────

def _gpu_script(gpu_id, commands, command_timeout_min, container_dir):
    """Shell script that runs inside the container for one GPU.
    Output flows to stdout/stderr; the salloc script prefixes and tees to a log file.
    """
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {container_dir}",
        "source .env",
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "export TF_GPU_ALLOCATOR=cuda_malloc_async",
        f"export CUDA_VISIBLE_DEVICES={gpu_id}",
    ]
    for cmd in commands:
        lines.append(f"timeout {command_timeout_min}m {cmd}")
    return "\n".join(lines) + "\n"


def _post_gpu_script(commands, command_timeout_min, container_dir):
    """Shell script that runs CPU-only post-GPU jobs inside the container."""
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {container_dir}",
        "source .env",
    ]
    for cmd in commands:
        lines.append(f"timeout {command_timeout_min}m {cmd}")
    return "\n".join(lines) + "\n"


def _salloc_script(gpu_ids, pid, remote_tmp, has_post_gpu):
    """
    Bash script that salloc runs on the login node.
    Each GPU worker's output is prefixed with [GPU N] and tee'd to a log file
    so it streams live to the terminal AND is saved for later inspection.
    """
    lines = [
        "#!/bin/bash",
        'NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)',
        'echo "[salloc] allocated node: $NODE"',
        'echo "[salloc] ensuring container is running..."',
        'ssh "$NODE" "sh ~/jax_start.sh"',
        'echo "[salloc] container ready — launching GPU workers"',
    ]
    for gpu_id in gpu_ids:
        log = f"{remote_tmp}/gpu_{gpu_id}_{pid}.log"
        script = f"/workspace/.jaxtpc_tmp/gpu_{gpu_id}_{pid}.sh"
        lines.append(
            f'{{ ssh "$NODE" "podman-hpc exec jaxtpc /bin/bash {script}" 2>&1'
            f" | sed -u 's/^/[GPU {gpu_id}] /' | tee {log}; }} &"
        )
    lines.append("wait")
    lines.append('echo "[salloc] all GPU workers finished"')
    if has_post_gpu:
        post_log = f"{remote_tmp}/post_gpu_{pid}.log"
        post_script = f"/workspace/.jaxtpc_tmp/post_gpu_{pid}.sh"
        lines += [
            'echo "[post-GPU] running CPU-only jobs..."',
            f'ssh "$NODE" "podman-hpc exec jaxtpc /bin/bash {post_script}" 2>&1'
            f" | sed -u 's/^/[post-GPU] /' | tee {post_log}",
            'echo "[post-GPU] done"',
        ]
    lines += [
        'echo "[salloc] stopping container"',
        'ssh "$NODE" "podman-hpc stop jaxtpc"',
    ]
    return "\n".join(lines) + "\n"


# ── SSH ControlMaster — one persistent connection for the whole run ───────────

_control_socket = None


def open_control_connection():
    global _control_socket
    _control_socket = f"/tmp/ssh_cm_{REMOTE}_{os.getpid()}"
    _log(f"opening persistent SSH connection to {REMOTE} ...")
    subprocess.run(
        ["ssh", "-M", "-S", _control_socket, "-fN", REMOTE],
        check=True,
    )
    _log("connection established")


def close_control_connection():
    if _control_socket:
        subprocess.run(
            ["ssh", "-S", _control_socket, "-O", "exit", REMOTE],
            check=False, capture_output=True,
        )
        _log("SSH connection closed")


# ── Remote helpers ────────────────────────────────────────────────────────────

def _ssh(cmd, input=None, check=True, capture=False):
    """Run a command on the login node, reusing the ControlMaster socket."""
    ssh_args = ["ssh"]
    if _control_socket:
        ssh_args += ["-S", _control_socket]
    ssh_args += [REMOTE, cmd]
    kwargs = dict(text=True, check=check)
    if input is not None:
        kwargs["input"] = input
    if capture:
        kwargs["capture_output"] = True
    return subprocess.run(ssh_args, **kwargs)


def _write_remote(path, content):
    """Write content to a file on Perlmutter via SSH + cat."""
    _ssh(f"cat > {path}", input=content)


def _cleanup(pid):
    _ssh(
        f"rm -f {REMOTE_TMP}/gpu_*_{pid}.sh {REMOTE_TMP}/post_gpu_{pid}.sh "
        f"{REMOTE_TMP}/salloc_{pid}.sh {REMOTE_TMP}/*_{pid}.txt",
        check=False,
    )


# ── Sync ──────────────────────────────────────────────────────────────────────

def sync_code(dry_run):
    _log("syncing code → Perlmutter (sync_code.sh)")
    if dry_run:
        _log("dry-run: skipped")
        return
    subprocess.run(["bash", "sync_code.sh"], check=True)


def sync_results(dry_run):
    _log(f"syncing results → S3 (sync_results_to_remote.sh on {REMOTE})")
    cmd = f"cd {REMOTE_DIR} && bash sync_results_to_remote.sh"
    if dry_run:
        _log(f"dry-run: would ssh {REMOTE} '{cmd}'")
        return
    _ssh(cmd)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan", help="YAML plan file")
    p.add_argument("--dry-run",          action="store_true",
                   help="Print scripts without executing")
    p.add_argument("--no-sync-code",     action="store_true")
    p.add_argument("--no-sync-results",  action="store_true")
    args = p.parse_args()

    with open(args.plan) as f:
        raw = yaml.safe_load(f)

    time_limit       = raw.get("time_limit", "01:00:00")
    command_timeout  = int(raw.get("command_timeout", 20))
    gpu_section      = raw.get("gpus", {})
    post_gpu_cmds    = [str(c) for c in raw.get("post_gpu_jobs", []) or []]

    gpu_cmds = {}
    for k, v in gpu_section.items():
        gpu_id = int(str(k).replace("gpu_", ""))
        if v is not None and len(v) > 0:
            gpu_cmds[gpu_id] = list(v)

    if not gpu_cmds:
        print("No commands in plan — nothing to do.")
        return

    n_cmds = sum(len(v) for v in gpu_cmds.values())
    _log(f"plan: {n_cmds} command(s) across GPU(s) {sorted(gpu_cmds)}  "
         f"salloc time={time_limit}  command_timeout={command_timeout}m")
    if post_gpu_cmds:
        _log(f"post-GPU jobs: {len(post_gpu_cmds)} command(s) (CPU-only, run after all GPUs finish)")

    pid = os.getpid()

    # ── Step 1: sync code ─────────────────────────────────────────────────────
    if not args.no_sync_code:
        sync_code(args.dry_run)

    # ── Open persistent SSH connection ────────────────────────────────────────
    if not args.dry_run:
        open_control_connection()

    # ── Step 2: write scripts to Perlmutter shared FS ────────────────────────
    if args.dry_run:
        _log("dry-run: would create ~/.jaxtpc_tmp/ on Perlmutter")
    else:
        _ssh(f"mkdir -p {REMOTE_TMP}")

    for gpu_id, commands in sorted(gpu_cmds.items()):
        script = _gpu_script(gpu_id, commands, command_timeout, CONTAINER_DIR)
        path   = f"{REMOTE_TMP}/gpu_{gpu_id}_{pid}.sh"
        if args.dry_run:
            _log(f"dry-run: gpu_{gpu_id} script → {path}\n{script}")
        else:
            _write_remote(path, script)
            _ssh(f"chmod +x {path}")
            _log(f"wrote {path}")

    if post_gpu_cmds:
        post_script = _post_gpu_script(post_gpu_cmds, command_timeout, CONTAINER_DIR)
        post_path   = f"{REMOTE_TMP}/post_gpu_{pid}.sh"
        if args.dry_run:
            _log(f"dry-run: post_gpu script → {post_path}\n{post_script}")
        else:
            _write_remote(post_path, post_script)
            _ssh(f"chmod +x {post_path}")
            _log(f"wrote {post_path}")

    salloc_script = _salloc_script(sorted(gpu_cmds), pid, REMOTE_TMP, has_post_gpu=bool(post_gpu_cmds))
    salloc_path   = f"{REMOTE_TMP}/salloc_{pid}.sh"
    if args.dry_run:
        _log(f"dry-run: salloc script → {salloc_path}\n{salloc_script}")
    else:
        _write_remote(salloc_path, salloc_script)
        _ssh(f"chmod +x {salloc_path}")
        _log(f"wrote {salloc_path}")

    # ── Step 3: run salloc ────────────────────────────────────────────────────
    salloc_cmd = (
        f"salloc --nodes 1 --qos interactive --time {time_limit} "
        f"--constraint gpu --gpus 4 --account {SLURM_ACCOUNT} "
        f"bash {salloc_path}"
    )
    _log(f"submitting: {salloc_cmd}")

    if not args.dry_run:
        t0 = time.time()
        try:
            _ssh(salloc_cmd)
        finally:
            _log(f"salloc finished in {time.time() - t0:.0f}s — cleaning up temp files")
            _cleanup(pid)
            close_control_connection()

    # ── Step 4: sync results ──────────────────────────────────────────────────
    if not args.no_sync_results:
        sync_results(args.dry_run)

    _log("all done.")


if __name__ == "__main__":
    main()
