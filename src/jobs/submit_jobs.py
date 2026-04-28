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
    grad_clip=10.0,
    lr_multipliers=None,
    warmup_steps=100,
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
        f"--clip-grad-norm {grad_clip}",
        f"--warmup-steps {warmup_steps}",
    ]

    if seed is not None:
        parts.append(f"--seed {seed}")
    if noise_scale > 0.0:
        parts.append(f"--noise-scale {noise_scale}")
    if lr_multipliers is not None:
        parts.append(f"--lr-multipliers {lr_multipliers}")
    return " ".join(parts)


if __name__ == "__main__":
    # diagonal + X + Y + Z at 1000, 100, and 50 MeV in one run
    TRACKS_12 = (
        "diagonal+X+Y+Z"
        "+diagonal_100MeV:1,1,1:100+x100:1,0,0:100+y100:0,1,0:100+z100:0,0,1:100"
        "+diagonal_50MeV:1,1,1:50+x50:1,0,0:50+y50:0,1,0:50+z50:0,0,1:50"
    )

    PARAM_LIST = [p.strip() for p in ALL_PARAMS.split(",") if p.strip()]

    SHARED = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="constant",
        max_steps=40000,
        tol=1e-6,
        patience=100,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/all_params_bugfix_20260428_1",
        grad_clip=10.0,
        lr_multipliers="velocity_cm_us:0.005",
        warmup_steps=1000,
    )

    for n_params in range(1, len(PARAM_LIST) + 1):
        params = ",".join(PARAM_LIST[:n_params])
        for seed in [42, 43, 44]:
            command = make_opt_command(
                params=params,
                tracks=TRACKS_12,
                seed=seed,
                noise_scale=0.0,
                **SHARED,
            )
            s3df_submit(command, time="10:00:00", submit=True)
   
