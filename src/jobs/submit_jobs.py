#!/usr/bin/env python3
"""
Helpers for building run_optimization.py commands and submitting them to S3DF.
"""
from job_submission_tools import s3df_submit

# All params compatible with the EMB recombination model
ALL_PARAMS = (
    "velocity_cm_us,"
    "lifetime_us,"
    "diffusion_trans_cm2_us,"
    "diffusion_long_cm2_us,"
    "recomb_alpha,"
    "recomb_beta_90,"
    "recomb_R"
)


def make_opt_command(
    params=ALL_PARAMS,
    tracks="diagonal_100MeV:1,1,1:100",
    loss="sobolev_loss_geomean_log1p",
    lr=0.001,
    lr_schedule="constant",
    max_steps=5000,
    tol=1e-6,
    patience=20,
    N=10,
    range_lo=0.9,
    range_hi=1.1,
    seed=None,
    noise_scale=0.0,
    results_base="$RESULTS_DIR/opt/all_params",
    grad_clip=10.0
):
    """Return a run_optimization.py command string with the given settings."""
    parts = [
        "python src/opt/run_optimization.py",
        f"--params {params}",
        f"--tracks {tracks}",
        f"--loss {loss}",
        f"--lr {lr}",
        f"--lr-schedule {lr_schedule}",
        f"--max-steps {max_steps}",
        f"--tol {tol}",
        f"--patience {patience}",
        f"--N {N}",
        f"--range {range_lo} {range_hi}",
        f"--results-base {results_base}",
        f"--clip-grad-norm {grad_clip}"
    ]
    if seed is not None:
        parts.append(f"--seed {seed}")
    if noise_scale > 0.0:
        parts.append(f"--noise-scale {noise_scale}")
    return " ".join(parts)


if __name__ == "__main__":
    for seed in [46, 47, 48, 49, 50]:
        command = make_opt_command(seed=seed, grad_clip=10.0, max_steps=10000)
        print("Command:")
        print(f"  {command}")
        print()
        s3df_submit(command, time="01:00:00", submit=True)
   