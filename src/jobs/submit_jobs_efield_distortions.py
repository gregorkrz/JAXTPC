#!/usr/bin/env python3
"""
Helpers for E-field distortion optimization jobs on S3DF.

Usage
-----
  python src/jobs/submit_jobs_efield_distortions.py <profile>
  python src/jobs/submit_jobs_efield_distortions.py <profile> --print-commands

Available profiles
------------------
  E_debug           Efield MLP (all 3 modes), 16 nice+ext tracks, 1 seed, 2000 steps.
  correction_seeds  Efield MLP correction mode, 16 nice+ext tracks, 5 seeds, 2000 steps.
  eval_E_debug      MLP inference on finished E_debug PKLs → *_efield_eval.pkl.

Profile runs pass --wandb-tags <profile> to run_optimization.py (optional extras via
--wandb-extra-tags comma,separated).
"""
import argparse

from job_submission_tools import s3df_submit
from submit_jobs import (
    TRACKS_13_NICE_EXT,
    make_opt_command,
)

# Pre-generated SCE distortion map (written by tools/efield_distortions.py).
EFIELD_DIST_PATH = "/fs/ddn/sdf/group/atlas/d/gregork/jaxtpc/results/efield_distortions/sce_maps_jaxtpc_41.npz"


def make_efield_opt_command(
    *,
    electric_dist_path=EFIELD_DIST_PATH,
    efield_mode="potential",
    efield_hidden=None,
    efield_lr_mult=None,
    mlp_snapshot_interval=None,
    **kwargs,
):
    """Like make_opt_command but appends E-field specific flags."""
    cmd = make_opt_command(**kwargs)
    cmd += f" --electric-dist-path {electric_dist_path}"
    cmd += f" --efield-mode {efield_mode}"
    if efield_hidden is not None:
        cmd += f" --efield-hidden {' '.join(str(h) for h in efield_hidden)}"
    if efield_lr_mult is not None and efield_lr_mult != 1.0:
        cmd += f" --efield-lr-mult {efield_lr_mult}"
    if mlp_snapshot_interval is not None and mlp_snapshot_interval > 0:
        cmd += f" --mlp-snapshot-interval {mlp_snapshot_interval}"
    return cmd


def profile_E_debug(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """Efield MLP, 16 nice+ext tracks, 1 seed, 2000 steps — one job per mode.

    Adam settings match Adam_20260601_cutoff_sweep: lr=1e-3, cosine LR,
    beta2=0.9, no grad clip, warmup=1000, step_size=1 mm, dep=5k,
    sobolev_exp=2, noisy GT (noise_scale=1).
    Spawns 3 jobs: potential, efield, correction.
    """
    shared = dict(
        params="Efield",
        tracks=TRACKS_13_NICE_EXT,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=2000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        seed=0,
        noise_scale=1.0,
        grad_clip=1.0,
        warmup_steps=1000,
        batch_size=1,
        effective_batch_size=1,
        step_size=1.0,
        max_num_deposits=5000,
        num_buckets=1000,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        adam_beta2=0.9,
        log_interval=50,
        sobolev_exponent=2.0,
    )
    for mode in ("potential", "efield", "correction"):
        command = make_efield_opt_command(
            results_base=f"$RESULTS_DIR/opt/E_debug/{mode}/noise",
            efield_mode=mode,
            mlp_snapshot_interval=500,
            wandb_tags=(wandb_tags or []) + ["E_debug", "efield", mode, "13trks_nice", "noise", "fixbug2"],
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        s3df_submit(
            command,
            time="01:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
        )


def profile_correction_seeds(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """Efield MLP correction mode, 16 nice+ext tracks, 5 seeds, 2000 steps.

    Same Adam settings as E_debug. Spawns 5 jobs (seeds 0–4), all using
    correction mode.
    """
    shared = dict(
        params="Efield",
        tracks=TRACKS_13_NICE_EXT,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=2000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        noise_scale=1.0,
        grad_clip=1.0,
        warmup_steps=1000,
        batch_size=1,
        effective_batch_size=1,
        step_size=1.0,
        max_num_deposits=5000,
        num_buckets=1000,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        adam_beta2=0.9,
        log_interval=50,
        sobolev_exponent=2.0,
    )
    for seed in range(5):
        command = make_efield_opt_command(
            results_base=f"$RESULTS_DIR/opt/correction_seeds/seed{seed}",
            efield_mode="correction",
            mlp_snapshot_interval=500,
            seed=seed,
            wandb_tags=(wandb_tags or []) + ["correction_seeds", "efield", "correction", "13trks_nice", "noise", f"seed{seed}", "fixbug2"],
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        s3df_submit(
            command,
            time="01:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
        )


def profile_eval_E_debug(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """Single job: MLP inference on all finished E_debug result PKLs.

    Writes *_efield_eval.pkl next to each result PKL.  Covers all three modes
    (potential / efield / correction) found under $RESULTS_DIR/opt/E_debug/.
    """
    command = (
        "python tools/eval_efield_mlp.py"
        " --results-dir $RESULTS_DIR/opt/E_debug"
        " --overwrite"
    )
    if not print_sbatch_only:
        print(command)
    s3df_submit(
        command,
        time="00:30:00",
        submit=submit,
        mem_gb=32,
        print_sbatch_command=print_sbatch_only,
    )


PROFILES = {
    "E_debug": profile_E_debug,
    "correction_seeds": profile_correction_seeds,
    "eval_E_debug": profile_eval_E_debug,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "profile",
        nargs="?",
        choices=list(PROFILES),
        help="Submission profile to run",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit jobs (default is dry-run)",
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Write batch scripts and print sbatch lines instead of submitting.",
    )
    parser.add_argument(
        "--wandb-extra-tags",
        default=None,
        metavar="TAGS",
        help="Comma-separated extra W&B tags appended to the profile tags.",
    )
    args = parser.parse_args()

    if args.print_commands and args.submit:
        parser.error("Use either --print-commands or --submit, not both.")

    if not args.profile:
        parser.error("Provide a profile name.")

    wandb_tags = [args.profile]
    if args.wandb_extra_tags:
        wandb_tags.extend(t.strip() for t in args.wandb_extra_tags.split(",") if t.strip())

    PROFILES[args.profile](
        submit=not args.print_commands,
        print_sbatch_only=args.print_commands,
        wandb_tags=wandb_tags,
    )
