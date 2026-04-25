import subprocess
from datetime import datetime
from pathlib import Path

PARTITION  = "ampere"
ACCOUNT    = "kipac:kipac"
REMOTE_DIR = "/sdf/home/g/gregork/jaxtpc"
JOBS_DIR   = Path(__file__).parent / "jobs"


def s3df_submit(command: str, *, time: str = "02:00:00", gpus: int = 1,
                mem_gb: int = 32, cpus_per_gpu: int = 4, submit: bool = False) -> Path:
    JOBS_DIR.mkdir(exist_ok=True)

    stem = Path(command.strip().split()[0]).stem
    name = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log  = JOBS_DIR / f"{name}.log"
    path = JOBS_DIR / f"{name}.sh"

    script = "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --account={ACCOUNT}",
        f"#SBATCH --partition={PARTITION}",
        f"#SBATCH --time={time}",
        f"#SBATCH --gpus={gpus}",
        f"#SBATCH --cpus-per-gpu={cpus_per_gpu}",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH --output={log}",
        f"#SBATCH --error={log}",
        "",
        "set -euo pipefail",
        f"cd {REMOTE_DIR}",
        "source .env",
        "",
        command,
    ]) + "\n"

    path.write_text(script)
    print(f"wrote {path}")

    if submit:
        result = subprocess.run(["sbatch", str(path)], text=True, capture_output=True)
        print(result.stdout.strip() or result.stderr.strip())

    return path
