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

REMOTE       = "pm"
REMOTE_DIR   = "/global/homes/g/gregork/jaxtpc"
REMOTE_TMP   = "$HOME/.jaxtpc_tmp"           # on shared FS, visible from compute nodes
SLURM_ACCOUNT = "m3246"


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(msg, tag="[main]"):
    print(f"{tag} {msg}", flush=True)


# ── Script builders ───────────────────────────────────────────────────────────

def _gpu_script(gpu_id, commands, command_timeout_min, remote_dir):
    """Shell script that runs inside the container for one GPU."""
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {remote_dir}",
        "source .env",
        f"export CUDA_VISIBLE_DEVICES={gpu_id}",
    ]
    for cmd in commands:
        lines.append(f"timeout {command_timeout_min}m {cmd}")
    return "\n".join(lines) + "\n"


def _salloc_script(gpu_ids, pid, remote_tmp):
    """
    Bash script that salloc runs on the login node.
    Ensures the container is running (detached) on the compute node first,
    then fans out one ssh per GPU using exec (no race on container creation).
    """
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)',
        'echo "[salloc] allocated node: $NODE"',
        # Start container once, detached — idempotent, safe to call even if running
        'echo "[salloc] ensuring container is running..."',
        'ssh "$NODE" "sh ~/jax_start.sh"',
        'echo "[salloc] container ready — launching GPU workers"',
    ]
    for gpu_id in gpu_ids:
        lines.append(
            f'ssh "$NODE" "podman-hpc exec jaxtpc /bin/bash /workspace/.jaxtpc_tmp/gpu_{gpu_id}_{pid}.sh" &'
        )
    lines += [
        "wait",
        'echo "[salloc] all GPU workers finished — stopping container"',
        'ssh "$NODE" "podman-hpc stop jaxtpc"',
    ]
    return "\n".join(lines) + "\n"


# ── Remote helpers ────────────────────────────────────────────────────────────

def _ssh(cmd, input=None, check=True, capture=False):
    """Run a command on the login node via SSH."""
    full = ["ssh", REMOTE, cmd]
    kwargs = dict(text=True, check=check)
    if input is not None:
        kwargs["input"] = input
        kwargs["stdin"] = subprocess.PIPE
    if capture:
        kwargs["capture_output"] = True
    return subprocess.run(full, **kwargs)


def _write_remote(path, content):
    """Write content to a file on Perlmutter via SSH + cat."""
    _ssh(f"cat > {path}", input=content)


def _cleanup(pid):
    _ssh(f"rm -f {REMOTE_TMP}/gpu_*_{pid}.sh {REMOTE_TMP}/salloc_{pid}.sh",
         check=False)


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

    gpu_cmds = {}
    for k, v in gpu_section.items():
        gpu_id = int(str(k).replace("gpu_", ""))
        if v:
            gpu_cmds[gpu_id] = list(v)

    if not gpu_cmds:
        print("No commands in plan — nothing to do.")
        return

    n_cmds = sum(len(v) for v in gpu_cmds.values())
    _log(f"plan: {n_cmds} command(s) across GPU(s) {sorted(gpu_cmds)}  "
         f"salloc time={time_limit}  command_timeout={command_timeout}m")

    pid = os.getpid()

    # ── Step 1: sync code ─────────────────────────────────────────────────────
    if not args.no_sync_code:
        sync_code(args.dry_run)

    # ── Step 2: write scripts to Perlmutter shared FS ────────────────────────
    if args.dry_run:
        _log("dry-run: would create ~/.jaxtpc_tmp/ on Perlmutter")
    else:
        _ssh(f"mkdir -p {REMOTE_TMP}")

    for gpu_id, commands in sorted(gpu_cmds.items()):
        script = _gpu_script(gpu_id, commands, command_timeout, REMOTE_DIR)
        path   = f"{REMOTE_TMP}/gpu_{gpu_id}_{pid}.sh"
        if args.dry_run:
            _log(f"dry-run: gpu_{gpu_id} script → {path}\n{script}")
        else:
            _write_remote(path, script)
            _ssh(f"chmod +x {path}")
            _log(f"wrote {path}")

    salloc_script = _salloc_script(sorted(gpu_cmds), pid, REMOTE_TMP)
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

    # ── Step 4: sync results ──────────────────────────────────────────────────
    if not args.no_sync_results:
        sync_results(args.dry_run)

    _log("all done.")


if __name__ == "__main__":
    main()
