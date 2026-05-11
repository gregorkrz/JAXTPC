import os
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

PARTITION  = "ampere"
ACCOUNT    = "mli:nu-ml-dev"
QOS        = "normal"
REMOTE_DIR = "/sdf/home/g/gregork/jaxtpc"
JOBS_DIR   = Path(__file__).parents[2] / "jobs"

LOGS_DIR = "/fs/ddn/sdf/group/atlas/d/gregork/logs"

APPTAINER_CACHEDIR = "/sdf/scratch/atlas/gregork/apptainer_cache"
APPTAINER_TMPDIR   = "/sdf/scratch/atlas/gregork/apptainer_tmp"
JAX_CACHE_DIR      = "/sdf/scratch/atlas/gregork/jax_cache"

APPTAINER_IMAGE = "docker://gkrz/jaxtpc:v2"
BIND_MOUNTS = [
    "/sdf/home/g/gregork/jaxtpc",
    "/fs/ddn/sdf/group/atlas/d/gregork/jaxtpc",
    JAX_CACHE_DIR,
]


def _ensure_python_opt_command(command: str) -> str:
    """Legacy runs stored ``' '.join(sys.argv)``, which omits the interpreter — bash then
    tries to execute ``run_optimization.py`` directly (Permission denied). Prepend ``python``.
    """
    c = command.strip()
    if not c:
        return c
    try:
        parts = shlex.split(c)
    except ValueError:
        return c
    if not parts:
        return c
    base = os.path.basename(parts[0])
    if base.startswith("python"):
        return c
    if parts[0].endswith("run_optimization.py"):
        return shlex.join(["python"] + parts)
    return c


def _sbatch_job_name(command: str) -> str:
    """Unique Slurm/script basename: script stem + optional seed + microsecond timestamp."""
    parts = command.strip().split()
    stem = "job"
    for p in parts:
        if p.endswith(".py"):
            stem = Path(p).stem
            break
    seed_m = re.search(r"--seed\s+(\S+)", command)
    seed_part = f"_seed{seed_m.group(1)}" if seed_m else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{stem}{seed_part}_{ts}"


def s3df_submit(command: str, *, time: str = "02:00:00", gpus: int = 1,
                mem_gb: int = 32, cpus_per_gpu: int = 1, submit: bool = False,
                print_sbatch_command: bool = False,
                sbatch_commands_out: Optional[List[str]] = None) -> Path:
    JOBS_DIR.mkdir(exist_ok=True)

    command = _ensure_python_opt_command(command)
    name = _sbatch_job_name(command)
    stdout_log = f"{LOGS_DIR}/{name}_stdout.txt"
    stderr_log = f"{LOGS_DIR}/{name}_stderr.txt"
    path = JOBS_DIR / f"{name}.sh"

    binds = " ".join(f"--bind {m}" for m in BIND_MOUNTS)
    wrapped = f"apptainer exec --nv {binds} {APPTAINER_IMAGE} bash -c {shlex.quote(command)}"

    script = "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --account={ACCOUNT}",
        f"#SBATCH --partition={PARTITION}",
        f"#SBATCH --qos={QOS}",
        f"#SBATCH --time={time}",
        f"#SBATCH --gpus={gpus}",
        f"#SBATCH --cpus-per-gpu={cpus_per_gpu}",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH --output={stdout_log}",
        f"#SBATCH --error={stderr_log}",
        "",
        "set -euo pipefail",
        f"mkdir -p {LOGS_DIR}",
        f"mkdir -p {JAX_CACHE_DIR}",
        f"export APPTAINER_CACHEDIR={APPTAINER_CACHEDIR}",
        f"export APPTAINER_TMPDIR={APPTAINER_TMPDIR}",
        f"cd {REMOTE_DIR}",
        "source .env",
        f"export JAX_COMPILATION_CACHE_DIR={JAX_CACHE_DIR}",
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "export TF_GPU_ALLOCATOR=cuda_malloc_async",
        "export WANDB_DISABLE_SERVICE=true",
        "",
        "nvidia-smi",
        "",
        wrapped,
    ]) + "\n"

    path.write_text(script)

    if submit:
        result = subprocess.run(["sbatch", str(path)], universal_newlines=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.strip() or result.stderr.strip())
    elif print_sbatch_command:
        line = f"sbatch {path.resolve()}"
        if sbatch_commands_out is not None:
            sbatch_commands_out.append(line)
        else:
            print(line)
    else:
        print(f"wrote {path}")

    return path
