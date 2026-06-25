#!/usr/bin/env python3
"""
Optimization jobs using 20 randomly generated boundary tracks.

Usage
-----
  python src/jobs/submit_jobs_optimization.py <profile> [--submit] [--print-commands]
  python src/jobs/submit_jobs_optimization.py <profile> --print-commands  # sbatch lines only

Available profiles
------------------
  diffusion_20seeds_adc50  Trans + long diffusion, 20 seeds, ±30% range, ADC cutoff 50,
                            cosine LR 4k steps, noisy GT. Two GTs: nominal and 80%
                            (gt_param_multiplier=0.8). 3 independent chains, max 3 jobs
                            simultaneously. 40 jobs total.
  diffusion_20seeds_adc50_all_params  Same as diffusion_20seeds_adc50, but ``--params``
                            covers all 7 EMB parameters. Nominal GT only, 30k steps,
                            no early stopping. Single dependency chain of 20 jobs.
  diffusion_20seeds_adc50_all_params_cutoff_D_only  Same as
                            diffusion_20seeds_adc50_all_params, but the ADC cutoff of 50
                            is applied only to the two diffusion params (via
                            --sobolev-loss-cutoff-per-param); all other params use no
                            cutoff. Nominal GT only, 30k steps, no early stopping.
                            Single dependency chain of 20 jobs.
  diffusion_20seeds_adc50_all_params_phase2_diffusion  Same as
                            diffusion_20seeds_adc50_all_params, but uses
                            --phase2-params/--phase2-start-step: the first 5k steps
                            optimize all params except the two diffusion constants
                            (frozen), then steps 5k-30k optimize only the diffusion
                            constants (everything else frozen). Nominal GT only, 30k
                            steps, no early stopping. Single dependency chain of 20
                            jobs.
"""
import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
from job_submission_tools import s3df_submit
from submit_jobs import make_opt_command, PARAM_LIST


def _build_tracks_string(n_tracks=20, seed=42, min_x_mm=1000.0):
    """Return '+'-separated track spec string from generate_random_boundary_track."""
    from tools.config import create_sim_config
    from tools.geometry import generate_detector
    from tools.random_boundary_tracks import generate_random_boundary_track

    raw = generate_detector(str(_REPO_ROOT / "config/cubic_wireplane_config.yaml"))
    volumes = create_sim_config(raw).volumes

    rng_energy = np.random.default_rng(seed)
    parts = []
    for i in range(n_tracks):
        direction, start_mm = generate_random_boundary_track(
            volumes, seed=seed + i, min_x_mm=min_x_mm
        )
        energy_mev = float(rng_energy.uniform(100.0, 1000.0))
        name = f"Track{i + 1}_{int(round(energy_mev))}MeV"
        dx, dy, dz = direction
        sx, sy, sz = start_mm
        parts.append(
            f"{name}:{dx:.9f},{dy:.9f},{dz:.9f}:{energy_mev:.1f}:{sx:.3f},{sy:.3f},{sz:.3f}"
        )
    return "+".join(parts)


def profile_diffusion_20seeds_adc50(
    *,
    submit=False,
    print_sbatch_only=False,
    wandb_tags=None,
    n_tracks=20,
    track_seed=42,
    min_x_mm=1000.0,
):
    """Trans + long diffusion, 20 seeds, ±30%, ADC50, cosine LR, noisy GT.

    Two ground truths (nominal / gt_param_multiplier=0.8) chained sequentially
    within each of 3 independent chains → max 3 jobs simultaneously.
    """
    tracks = _build_tracks_string(n_tracks=n_tracks, seed=track_seed, min_x_mm=min_x_mm)

    shared = dict(
        tracks=tracks,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=4000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.7,
        range_hi=1.3,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=1,
        effective_batch_size=1,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        adam_beta2=0.9,
        log_interval=50,
        noise_scale=1.0,
        sobolev_exponent=2.0,
        sobolev_loss_cutoff=50.0,
        fourier_cutoff=100.0,
        rotate_noise_seeds=-1,
    )

    params = "diffusion_trans_cm2_us,diffusion_long_cm2_us"
    profile_name = "diffusion_20seeds_adc50_20trks"
    base_tags = (wandb_tags or []) + [
        "Run_Opt_20260609", "trans_and_long", "noise",
        "20trks_boundary", "adc50", "ft100", "rot-1",
        f"track_seed_{track_seed}",
    ]

    all_seeds = list(range(20))
    # Split into 3 chains: 7 + 7 + 6
    chain_seed_groups = [all_seeds[0:7], all_seeds[7:14], all_seeds[14:20]]

    gt_configs = [
        ("nominal", None),
        ("gt80pct", 0.8),
    ]

    # Each chain runs all GT-nominal seeds then all GT-80pct seeds sequentially
    for chain_idx, chain_seeds in enumerate(chain_seed_groups):
        prev_job = None
        for gt_label, gt_mult in gt_configs:
            results_base = f"$RESULTS_DIR/opt/{profile_name}/{gt_label}"
            for seed in chain_seeds:
                command = make_opt_command(
                    params=params,
                    seed=seed,
                    results_base=f"{results_base}/noise",
                    gt_param_multiplier=gt_mult,
                    wandb_tags=base_tags + [gt_label, f"chain{chain_idx}"],
                    **shared,
                )
                if not print_sbatch_only:
                    print(command)
                prev_job = s3df_submit(
                    command,
                    time="00:12:00",
                    submit=submit,
                    mem_gb=64,
                    print_sbatch_command=print_sbatch_only,
                    dependency=prev_job,
                )


def profile_diffusion_20seeds_adc50_all_params(
    *,
    submit=False,
    print_sbatch_only=False,
    wandb_tags=None,
    n_tracks=20,
    track_seed=42,
    min_x_mm=1000.0,
):
    """Same as diffusion_20seeds_adc50, but optimizes all 7 EMB params at once.

    Nominal GT only, single dependency chain over all 20 seeds, 30k steps,
    no early stopping (patience == max_steps).
    """
    tracks = _build_tracks_string(n_tracks=n_tracks, seed=track_seed, min_x_mm=min_x_mm)

    shared = dict(
        tracks=tracks,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=30000,
        N=1,
        range_lo=0.7,
        range_hi=1.3,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=1,
        effective_batch_size=1,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        adam_beta2=0.9,
        log_interval=50,
        noise_scale=1.0,
        sobolev_exponent=2.0,
        sobolev_loss_cutoff=50.0,
        fourier_cutoff=100.0,
        rotate_noise_seeds=-1,
    )

    params = ",".join(PARAM_LIST)
    profile_name = "diffusion_20seeds_adc50_all_params_20trks"
    base_tags = (wandb_tags or []) + [
        "Run_Opt_20260609", "all_params", "noise",
        "20trks_boundary", "adc50", "ft100", "rot-1",
        f"track_seed_{track_seed}",
    ]

    results_base = f"$RESULTS_DIR/opt/{profile_name}/nominal"
    prev_job = None
    for seed in range(20):
        command = make_opt_command(
            params=params,
            seed=seed,
            results_base=f"{results_base}/noise",
            gt_param_multiplier=None,
            wandb_tags=base_tags + ["nominal", "chain0"],
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        prev_job = s3df_submit(
            command,
            time="02:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            dependency=prev_job,
        )


def profile_diffusion_20seeds_adc50_all_params_cutoff_D_only(
    *,
    submit=False,
    print_sbatch_only=False,
    wandb_tags=None,
    n_tracks=20,
    track_seed=42,
    min_x_mm=1000.0,
):
    """Same as diffusion_20seeds_adc50_all_params, but the ADC cutoff of 50 is applied
    only to the two diffusion params (via --sobolev-loss-cutoff-per-param); all other
    params get no cutoff.

    Nominal GT only, single dependency chain over all 20 seeds, 30k steps,
    no early stopping (patience == max_steps).
    """
    tracks = _build_tracks_string(n_tracks=n_tracks, seed=track_seed, min_x_mm=min_x_mm)

    shared = dict(
        tracks=tracks,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=30000,
        N=1,
        range_lo=0.7,
        range_hi=1.3,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=1,
        effective_batch_size=1,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        adam_beta2=0.9,
        log_interval=50,
        noise_scale=1.0,
        sobolev_exponent=2.0,
        sobolev_loss_cutoff_per_param="diffusion_trans_cm2_us:50,diffusion_long_cm2_us:50",
        fourier_cutoff=100.0,
        rotate_noise_seeds=-1,
    )

    params = ",".join(PARAM_LIST)
    profile_name = "diffusion_20seeds_adc50_all_params_cutoff_D_only_20trks"
    base_tags = (wandb_tags or []) + [
        "Run_Opt_20260609", "all_params", "noise",
        "20trks_boundary", "adc50_D_only", "ft100", "rot-1",
        f"track_seed_{track_seed}",
    ]

    results_base = f"$RESULTS_DIR/opt/{profile_name}/nominal"
    prev_job = None
    for seed in range(20):
        command = make_opt_command(
            params=params,
            seed=seed,
            results_base=f"{results_base}/noise",
            gt_param_multiplier=None,
            wandb_tags=base_tags + ["nominal", "chain0"],
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        prev_job = s3df_submit(
            command,
            time="02:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            dependency=prev_job,
        )


def profile_diffusion_20seeds_adc50_all_params_phase2_diffusion(
    *,
    submit=False,
    print_sbatch_only=False,
    wandb_tags=None,
    n_tracks=20,
    track_seed=42,
    min_x_mm=1000.0,
):
    """Same as diffusion_20seeds_adc50_all_params, but with a two-phase param schedule:
    steps 0-5000 optimize all params except the diffusion constants (frozen), then
    steps 5000-30000 optimize only the diffusion constants (everything else frozen).

    Nominal GT only, single dependency chain over all 20 seeds, 30k steps,
    no early stopping (patience == max_steps).
    """
    tracks = _build_tracks_string(n_tracks=n_tracks, seed=track_seed, min_x_mm=min_x_mm)

    shared = dict(
        tracks=tracks,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=30000,
        N=1,
        range_lo=0.7,
        range_hi=1.3,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=1,
        effective_batch_size=1,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        adam_beta2=0.9,
        log_interval=50,
        noise_scale=1.0,
        sobolev_exponent=2.0,
        sobolev_loss_cutoff=50.0,
        fourier_cutoff=100.0,
        rotate_noise_seeds=-1,
        phase2_params="diffusion_trans_cm2_us,diffusion_long_cm2_us",
        phase2_start_step=5000,
    )

    params = ",".join(PARAM_LIST)
    profile_name = "diffusion_20seeds_adc50_all_params_phase2_diffusion_20trks"
    base_tags = (wandb_tags or []) + [
        "Run_Opt_20260609", "all_params", "phase2_diffusion", "noise",
        "20trks_boundary", "adc50", "ft100", "rot-1",
        f"track_seed_{track_seed}",
    ]

    results_base = f"$RESULTS_DIR/opt/{profile_name}/nominal"
    prev_job = None
    for seed in range(20):
        command = make_opt_command(
            params=params,
            seed=seed,
            results_base=f"{results_base}/noise",
            gt_param_multiplier=None,
            wandb_tags=base_tags + ["nominal", "chain0"],
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        prev_job = s3df_submit(
            command,
            time="02:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            dependency=prev_job,
        )


PROFILES = {
    "diffusion_20seeds_adc50": profile_diffusion_20seeds_adc50,
    "diffusion_20seeds_adc50_all_params": profile_diffusion_20seeds_adc50_all_params,
    "diffusion_20seeds_adc50_all_params_cutoff_D_only":
        profile_diffusion_20seeds_adc50_all_params_cutoff_D_only,
    "diffusion_20seeds_adc50_all_params_phase2_diffusion":
        profile_diffusion_20seeds_adc50_all_params_phase2_diffusion,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=list(PROFILES))
    parser.add_argument("--submit", action="store_true", help="actually submit to Slurm")
    parser.add_argument(
        "--print-commands", action="store_true",
        help="print run_optimization.py commands (default) or sbatch lines"
    )
    parser.add_argument("--n-tracks", type=int, default=20)
    parser.add_argument("--track-seed", type=int, default=42)
    parser.add_argument("--min-x-mm", type=float, default=1000.0)
    parser.add_argument("--wandb-extra-tags", default="")
    args = parser.parse_args()

    extra_tags = [t.strip() for t in args.wandb_extra_tags.split(",") if t.strip()]

    fn = PROFILES[args.profile]
    fn(
        submit=args.submit,
        print_sbatch_only=args.print_commands,
        wandb_tags=extra_tags or None,
        n_tracks=args.n_tracks,
        track_seed=args.track_seed,
        min_x_mm=args.min_x_mm,
    )


if __name__ == "__main__":
    main()
