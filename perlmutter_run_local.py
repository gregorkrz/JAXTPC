#!/usr/bin/env python3
"""
Run a YAML plan on Perlmutter GPU nodes from a login-node tmux session.

Identical to perlmutter_run.py except:
  - No SSH to the login node (you are already here).
  - Scripts are written directly to the local shared filesystem.
  - salloc is invoked as a local subprocess.
  - No code/results sync (do those manually or with sync_code.sh / sync_results_to_remote.sh).

Usage:
  python perlmutter_run_local.py plan.yaml
  python perlmutter_run_local.py plan.yaml --dry-run
"""
import argparse
import glob
import os
import subprocess
import time

import yaml

CONTAINER_DIR = "/workspace/jaxtpc"
LOCAL_TMP     = os.path.expanduser("~/.jaxtpc_tmp")
SLURM_ACCOUNT = "m3246"


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(msg, tag="[main]"):
    print(f"{tag} {msg}", flush=True)


# ── Script builders (identical to perlmutter_run.py) ─────────────────────────

def _gpu_script(gpu_id, commands, command_timeout_min, container_dir):
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {container_dir}",
        "source .env",
        f"export CUDA_VISIBLE_DEVICES={gpu_id}",
    ]
    for cmd in commands:
        lines.append(f"timeout {command_timeout_min}m {cmd}")
    return "\n".join(lines) + "\n"


def _post_gpu_script(commands, command_timeout_min, container_dir):
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {container_dir}",
        "source .env",
    ]
    for cmd in commands:
        lines.append(f"timeout {command_timeout_min}m {cmd}")
    return "\n".join(lines) + "\n"


def _salloc_script(gpu_ids, pid, tmp_dir, has_post_gpu):
    lines = [
        "#!/bin/bash",
        'NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)',
        'echo "[salloc] allocated node: $NODE"',
        'echo "[salloc] ensuring container is running..."',
        'ssh "$NODE" "sh ~/jax_start.sh"',
        'echo "[salloc] container ready — launching GPU workers"',
    ]
    for gpu_id in gpu_ids:
        log    = f"{tmp_dir}/gpu_{gpu_id}_{pid}.log"
        script = f"/workspace/.jaxtpc_tmp/gpu_{gpu_id}_{pid}.sh"
        lines.append(
            f'{{ ssh "$NODE" "podman-hpc exec jaxtpc /bin/bash {script}" 2>&1'
            f" | sed -u 's/^/[GPU {gpu_id}] /' | tee {log}; }} &"
        )
    lines.append("wait")
    lines.append('echo "[salloc] all GPU workers finished"')
    if has_post_gpu:
        post_log    = f"{tmp_dir}/post_gpu_{pid}.log"
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


# ── Local helpers ─────────────────────────────────────────────────────────────

def _write(path, content):
    with open(path, "w") as f:
        f.write(content)


def _cleanup(pid):
    patterns = [
        f"gpu_*_{pid}.sh",
        f"post_gpu_{pid}.sh",
        f"salloc_{pid}.sh",
        f"*_{pid}.txt",
    ]
    for pat in patterns:
        for p in glob.glob(os.path.join(LOCAL_TMP, pat)):
            try:
                os.remove(p)
            except OSError:
                pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan", help="YAML plan file")
    p.add_argument("--dry-run", action="store_true",
                   help="Print scripts without executing")
    args = p.parse_args()

    with open(args.plan) as f:
        raw = yaml.safe_load(f)

    time_limit      = raw.get("time_limit", "01:00:00")
    command_timeout = int(raw.get("command_timeout", 20))
    gpu_section     = raw.get("gpus", {})
    post_gpu_cmds   = [str(c) for c in raw.get("post_gpu_jobs", []) or []]

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
        _log(f"post-GPU jobs: {len(post_gpu_cmds)} command(s)")

    pid = os.getpid()
    os.makedirs(LOCAL_TMP, exist_ok=True)

    # ── Write GPU scripts ─────────────────────────────────────────────────────
    for gpu_id, commands in sorted(gpu_cmds.items()):
        script = _gpu_script(gpu_id, commands, command_timeout, CONTAINER_DIR)
        path   = os.path.join(LOCAL_TMP, f"gpu_{gpu_id}_{pid}.sh")
        if args.dry_run:
            _log(f"dry-run: gpu_{gpu_id} script → {path}\n{script}")
        else:
            _write(path, script)
            os.chmod(path, 0o755)
            _log(f"wrote {path}")

    if post_gpu_cmds:
        post_script = _post_gpu_script(post_gpu_cmds, command_timeout, CONTAINER_DIR)
        post_path   = os.path.join(LOCAL_TMP, f"post_gpu_{pid}.sh")
        if args.dry_run:
            _log(f"dry-run: post_gpu script → {post_path}\n{post_script}")
        else:
            _write(post_path, post_script)
            os.chmod(post_path, 0o755)
            _log(f"wrote {post_path}")

    salloc_script = _salloc_script(sorted(gpu_cmds), pid, LOCAL_TMP,
                                   has_post_gpu=bool(post_gpu_cmds))
    salloc_path   = os.path.join(LOCAL_TMP, f"salloc_{pid}.sh")
    if args.dry_run:
        _log(f"dry-run: salloc script → {salloc_path}\n{salloc_script}")
    else:
        _write(salloc_path, salloc_script)
        os.chmod(salloc_path, 0o755)
        _log(f"wrote {salloc_path}")

    # ── Submit salloc ─────────────────────────────────────────────────────────
    salloc_cmd = [
        "salloc",
        "--nodes", "1",
        "--qos", "interactive",
        "--time", time_limit,
        "--constraint", "gpu",
        "--gpus", "4",
        "--account", SLURM_ACCOUNT,
        "bash", salloc_path,
    ]
    _log(f"submitting: {' '.join(salloc_cmd)}")

    if not args.dry_run:
        t0 = time.time()
        try:
            subprocess.run(salloc_cmd, check=True)
        finally:
            _log(f"salloc finished in {time.time() - t0:.0f}s — cleaning up")
            _cleanup(pid)

    _log("all done.")


if __name__ == "__main__":
    main()
