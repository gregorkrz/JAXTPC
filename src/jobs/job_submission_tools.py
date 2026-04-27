import shlex
import subprocess
from datetime import datetime
from pathlib import Path

PARTITION  = "ampere"
ACCOUNT    = "neutrino"
REMOTE_DIR = "/sdf/home/g/gregork/jaxtpc"
JOBS_DIR   = Path(__file__).parents[2] / "jobs"

LOGS_DIR = "/fs/ddn/sdf/group/atlas/d/gregork/logs"

APPTAINER_CACHEDIR = "/sdf/scratch/atlas/gregork/apptainer_cache"
APPTAINER_TMPDIR   = "/sdf/scratch/atlas/gregork/apptainer_tmp"

APPTAINER_IMAGE = "docker://gkrz/jaxtpc:v2"
BIND_MOUNTS = [
    "/sdf/home/g/gregork/jaxtpc",
    "/fs/ddn/sdf/group/atlas/d/gregork/jaxtpc",
]


def s3df_submit(command: str, *, time: str = "02:00:00", gpus: int = 1,
                mem_gb: int = 32, cpus_per_gpu: int = 1, submit: bool = False) -> Path:
    JOBS_DIR.mkdir(exist_ok=True)

    stem = Path(command.strip().split()[0]).stem
    name = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        f"#SBATCH --time={time}",
        f"#SBATCH --gpus={gpus}",
        f"#SBATCH --cpus-per-gpu={cpus_per_gpu}",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH --output={stdout_log}",
        f"#SBATCH --error={stderr_log}",
        "",
        "set -euo pipefail",
        f"mkdir -p {LOGS_DIR}",
        f"export APPTAINER_CACHEDIR={APPTAINER_CACHEDIR}",
        f"export APPTAINER_TMPDIR={APPTAINER_TMPDIR}",
        f"cd {REMOTE_DIR}",
        "source .env",
        "",
        "nvidia-smi",
        "",
        wrapped,
    ]) + "\n"

    path.write_text(script)
    print(f"wrote {path}")

    if submit:
        result = subprocess.run(["sbatch", str(path)], universal_newlines=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.strip() or result.stderr.strip())

    return path
