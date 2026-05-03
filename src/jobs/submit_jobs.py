#!/usr/bin/env python3
"""
Helpers for building run_optimization.py commands and submitting them to S3DF.

Usage
-----
  python src/jobs/submit_jobs.py <profile>
  python src/jobs/submit_jobs.py <profile> --print-commands

  Print Slurm sbatch lines for incomplete/preempted jobs (dry-run; paste on login node).
  Progress / diagnostics print on stderr first, then a separator line; sbatch commands print on stdout last only,
  so interleaved tty output does not mix them:

    python src/jobs/submit_jobs.py --restart-preempted \\
        '$RESULTS_DIR/opt/sched2_longer_schedule_20260430' \\
        --time 04:00:00 --mem-gb 64

    python src/jobs/submit_jobs.py --restart-preempted \\
        '$RESULTS_DIR/opt/sched2_longer_schedule_20260430/no_sched_fine' \\
        --time 04:00:00 --mem-gb 64

  Actually submit those restarts:

    python src/jobs/submit_jobs.py --restart-preempted <results_dir> --submit ...

Available profiles
------------------
  3_part_schedule              Coarse-to-fine two-phase schedule (1.0→0.1 mm), seeds 44–47
  2_part_schedule                  1.0 mm phase (5k steps) then 0.1 mm (15k steps), constant LR, 20k steps
  2_part_schedule_cosine_30k       Same 2-phase fwd schedule; cosine LR over 30k steps (5k + 25k)
  fine_nosched_bs1                  Single-phase 0.1 mm run, seeds 44–47
  fine_nosched_bs1_tol1e4_p300      Same as fine_nosched_bs1; per-param freeze tol=1e-4, window=300
  longitudinal_diffusion_only       Same tracks/seeds as fine_nosched_bs1; optimize diffusion_long_cm2_us only
  longitudinal_transverse_diffusion Same setup; optimize diffusion_long_cm2_us and diffusion_trans_cm2_us together
  timing_study_diag50mev           Seven jobs: single 50 MeV diagonal track, deposit pads 5k–100k, 1000 steps (OOM sweep)
  timing_study_cont                Three jobs: same setup, deposit pads 45k / 50k / 55k only
  no_schedule_less_params         No phase schedule; sweep n_params=3..7 always including
                                  diffusion_long_cm2_us, then other physics (transverse diffusion last).
                                  Same tracks/seeds as fine_nosched_bs1 (20 Slurm jobs).

Profile runs pass --wandb-tags <profile> to run_optimization.py (optional extras via
--wandb-extra-tags comma,separated).
"""
import argparse
import glob
import os
import pickle
import sys

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

# diagonal + X + Y + Z at 1000, 100, and 50 MeV
TRACKS_12 = (
    "diagonal+X+Y+Z"
    "+diagonal_100MeV:1,1,1:100+x100:1,0,0:100+y100:0,1,0:100+z100:0,0,1:100"
    "+diagonal_50MeV:1,1,1:50+x50:1,0,0:50+y50:0,1,0:50+z50:0,0,1:50"
)

# Single track (same physics tag as diagonal_50MeV in TRACKS_12)
TRACK_DIAG_50MEV = "diagonal_50MeV:1,1,1:50"

PARAM_LIST = [p.strip() for p in ALL_PARAMS.split(",") if p.strip()]

# Joint fits with longitudinal diffusion + growing nuisance set (transverse diffusion last).
_LONG_DIFF_GROW_EXTRAS = (
    "velocity_cm_us,"
    "lifetime_us,"
    "recomb_alpha,"
    "recomb_beta_90,"
    "recomb_R,"
    "diffusion_trans_cm2_us"
)
_LONG_DIFF_GROW_EXTRAS_LIST = [
    s.strip() for s in _LONG_DIFF_GROW_EXTRAS.split(",") if s.strip()
]


def params_growing_with_long_diffusion(n_params: int) -> str:
    """Comma-separated ``--params`` for ``n_params`` in [3, 7].

    Always starts with ``diffusion_long_cm2_us``, then appends other parameters in
    fixed order; ``diffusion_trans_cm2_us`` is the last one added (full ``ALL_PARAMS``
    at ``n_params`` == 7).
    """
    n_max = 1 + len(_LONG_DIFF_GROW_EXTRAS_LIST)  # 7
    if n_params < 3 or n_params > n_max:
        raise ValueError(
            f"n_params must be in [3, {n_max}] (got {n_params})"
        )
    names = ["diffusion_long_cm2_us"] + _LONG_DIFF_GROW_EXTRAS_LIST[: n_params - 1]
    return ",".join(names)


def _optimization_pickle_incomplete(data) -> bool:
    """True when this result pickle needs another Slurm/job run.

    Mirrors ``optimization_run_complete`` in ``run_optimization.py`` (keep in sync).
    """
    trials = data.get("trials")
    if trials is None:
        return True
    n_expected = data.get("N")
    if not isinstance(n_expected, int) or n_expected < 0:
        return True
    if len(trials) < n_expected:
        return True
    if data.get("live_checkpoint"):
        return True
    return False


def _optimization_steps_progress_line(data) -> str:
    """Human-readable optimizer-step progress (mirrors run_optimization checkpoints).

    Each trial contributes ``steps_run``; an active ``live_checkpoint`` adds the
    current trial's completed steps at last intra-trial save (same ``step`` used
    as ``start_step`` on resume). Upper bound shown as ``N * max_steps`` when known.
    """
    trials = data.get("trials") or []
    n_expect = data.get("N")
    max_steps = data.get("max_steps")
    done = sum(int(t.get("steps_run", 0) or 0) for t in trials)
    ckpt = data.get("live_checkpoint") or {}
    ckpt_frag = ""
    if ckpt:
        trial_idx = ckpt.get("trial_idx")
        st = ckpt.get("step")
        try:
            partial = int(st) if st is not None else 0
        except (TypeError, ValueError):
            partial = 0
        done += partial
        ckpt_frag = f", checkpoint trial_idx={trial_idx} step={partial}"

    if isinstance(n_expect, int) and isinstance(max_steps, int) and n_expect > 0 and max_steps > 0:
        cap = n_expect * max_steps
        return (
            f"optimizer_steps≈{done}/{cap} max ({len(trials)}/{n_expect} trials)"
            f"{ckpt_frag}"
        )
    return f"optimizer_steps≈{done} ({len(trials)} trials logged){ckpt_frag}"


def _command_txt_for_result_pkl(pkl_path: str):
    stem = os.path.basename(pkl_path)
    if not (stem.startswith("result_") and stem.endswith(".pkl")):
        return None
    seed_part = stem[len("result_") : -len(".pkl")]
    cand = os.path.join(os.path.dirname(pkl_path), f"command_{seed_part}.txt")
    if os.path.isfile(cand):
        with open(cand) as f:
            return f.read().strip()
    return None


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
    step_size=None,
    max_num_deposits=None,
    num_buckets=None,
    batch_size=None,
    schedule_steps=None,
    schedule_step_sizes=None,
    schedule_deposits=None,
    schedule_batch_sizes=None,
    gt_step_size=None,
    gt_max_deposits=None,
    wandb_tags=None,
    tol_per_param=None,
    patience_per_param=None,
    log_interval=None,
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
    if batch_size is not None:
        parts.append(f"--batch-size {batch_size}")
    if step_size is not None:
        parts.append(f"--step-size {step_size}")
    if max_num_deposits is not None:
        parts.append(f"--max-num-deposits {max_num_deposits}")
    if num_buckets is not None:
        parts.append(f"--num-buckets {num_buckets}")
    if schedule_steps is not None:
        parts.append(f"--schedule-steps {schedule_steps}")
    if schedule_step_sizes is not None:
        parts.append(f"--schedule-step-sizes {schedule_step_sizes}")
    if schedule_deposits is not None:
        parts.append(f"--schedule-deposits {schedule_deposits}")
    if schedule_batch_sizes is not None:
        parts.append(f"--schedule-batch-sizes {schedule_batch_sizes}")
    if gt_step_size is not None:
        parts.append(f"--gt-step-size {gt_step_size}")
    if gt_max_deposits is not None:
        parts.append(f"--gt-max-deposits {gt_max_deposits}")
    if wandb_tags:
        tags_csv = ",".join(w.strip() for w in wandb_tags if str(w).strip())
        if tags_csv:
            parts.append(f"--wandb-tags {tags_csv}")
    if tol_per_param is not None:
        parts.append(f"--tol-per-param {tol_per_param}")
    if patience_per_param is not None:
        parts.append(f"--patience-per-param {patience_per_param}")
    if log_interval is not None:
        parts.append(f"--log-interval {int(log_interval)}")
    return " ".join(parts)


def resubmit_preempted(results_dir: str, *, time: str = "10:00:00",
                       gpus: int = 1, mem_gb: int = 32, submit: bool = False,
                       print_sbatch_command: bool = False):
    """Scan results_dir for incomplete pkl files and resubmit their jobs.

    Incomplete means: fewer than N trials, a non-empty ``live_checkpoint``, unreadable
    pkls (uses sibling ``command_<seed>.txt`` when corrupt), or duplicate commands are
    skipped within one invocation. Runs with N trials and no ``live_checkpoint`` are
    treated as complete even if ``run_complete`` was never flipped after SIGTERM.
    """
    verbose = not print_sbatch_command

    pkls = sorted(glob.glob(os.path.join(results_dir, "**", "result_*.pkl"), recursive=True))

    seen_restart_commands = set()
    resubmitted = 0
    sbatch_accum = [] if print_sbatch_command else None
    for pkl_path in pkls:
        command = None
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as exc:
            if verbose:
                print(f"  UNREADABLE {pkl_path}: {exc}")
            command = _command_txt_for_result_pkl(pkl_path)
            if not command:
                if verbose:
                    print(f"  SKIP {pkl_path}: corrupt pkl and no matching command_*.txt")
                continue
            if verbose:
                print(f"  Fallback: using sibling command file for corrupt pkl")
            print(f"{pkl_path}: optimizer_steps=? (pkl unreadable)", file=sys.stderr)
        else:
            if not _optimization_pickle_incomplete(data):
                continue
            n_done = len(data.get("trials", []))
            n_total = data.get("N", -1)
            command = data.get("command")
            print(f"{pkl_path}: {_optimization_steps_progress_line(data)}", file=sys.stderr)
            if verbose:
                print(f"  {'Submitting' if submit else 'Would submit'} "
                      f"{pkl_path}  ({n_done}/{n_total} trials)")
        if not command:
            if verbose:
                print(f"  SKIP {pkl_path}: no 'command' field (needs a run with newer code)")
            continue
        if command in seen_restart_commands:
            if verbose:
                print(f"  SKIP duplicate restart command (already queued): {pkl_path}")
            continue
        seen_restart_commands.add(command)
        s3df_submit(
            command,
            time=time,
            gpus=gpus,
            mem_gb=mem_gb,
            submit=submit,
            print_sbatch_command=print_sbatch_command,
            sbatch_commands_out=sbatch_accum,
        )
        resubmitted += 1

    # Also resubmit jobs that were preempted before writing any pkl.
    txts = sorted(glob.glob(os.path.join(results_dir, "**", "command_*.txt"), recursive=True))
    for txt_path in txts:
        stem = os.path.basename(txt_path)
        seed_str = stem[len('command_'):-len('.txt')]
        pkl_path = os.path.join(os.path.dirname(txt_path), f'result_{seed_str}.pkl')
        if os.path.exists(pkl_path):
            continue  # pkl exists — handled by the loop above
        with open(txt_path) as f:
            command = f.read().strip()
        if command in seen_restart_commands:
            if verbose:
                print(f"  SKIP duplicate restart command (already queued): {txt_path}")
            continue
        seen_restart_commands.add(command)
        print(f"{txt_path}: optimizer_steps≈0 (no checkpoint yet)", file=sys.stderr)
        if verbose:
            print(f"  {'Submitting' if submit else 'Would submit'} "
                  f"{txt_path}  (no pkl — preempted before first write)")
        s3df_submit(
            command,
            time=time,
            gpus=gpus,
            mem_gb=mem_gb,
            submit=submit,
            print_sbatch_command=print_sbatch_command,
            sbatch_commands_out=sbatch_accum,
        )
        resubmitted += 1

    if print_sbatch_command:
        if resubmitted == 0:
            print(f"\nNo incomplete/preempted jobs to resubmit under {results_dir}.",
                  file=sys.stderr)
        else:
            noun = "job" if resubmitted == 1 else "jobs"
            print(
                f"\n{resubmitted} {noun} dry-run — sbatch lines below on stdout "
                f"(copy-paste after the separator).\n",
                file=sys.stderr,
            )
            print("---------- sbatch commands ----------", file=sys.stderr)
            for line in sbatch_accum:
                print(line)
    elif verbose:
        if resubmitted == 0:
            print(f"\nNo incomplete/preempted jobs to resubmit under {results_dir}.")
        else:
            noun = "job" if resubmitted == 1 else "jobs"
            action = "submitted" if submit else "found (pass submit=True to actually submit)"
            print(f"\n{resubmitted} {noun} {action}.")


# ── Profiles ──────────────────────────────────────────────────────────────────

def profile_3_part_schedule(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """Coarse-to-fine two-phase schedule over all params, 12 tracks, seeds 44–47.

    Phase 0: step_size=1.0 mm, 5k deposits, batch_size=5  (steps 0–5000)
    Phase 1: step_size=0.1 mm, 50k deposits, batch_size=1  (steps 5000–40000)
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="constant",
        max_steps=40000,
        tol=1e-6,
        patience=100,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/sched2_longer_schedule_20260430",
        grad_clip=10.0,
        lr_multipliers="velocity_cm_us:0.005",
        warmup_steps=1000,
        num_buckets=1000,
        schedule_steps="5000",
        schedule_step_sizes="1.0,0.1",
        schedule_deposits="5000,50000",
        schedule_batch_sizes="5,1",
    )
    params = ",".join(PARAM_LIST)
    for seed in [44, 45, 46, 47]:
        command = make_opt_command(
            params=params,
            tracks=TRACKS_12,
            seed=seed,
            noise_scale=0.0,
            wandb_tags=wandb_tags,
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        s3df_submit(
            command,
            time="04:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
        )


def profile_fine_nosched_bs1(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    # no schedule, just do 0.1mm, 50k steps, save into subdir no_sched_fine
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="constant",
        max_steps=50000,
        tol=1e-6,
        patience=100,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/sched2_longer_schedule_20260430/no_sched_fine",
        grad_clip=10.0,
        lr_multipliers="velocity_cm_us:0.005",
        warmup_steps=1000,
        num_buckets=1000,
        step_size=0.1,
        max_num_deposits=50000,
        batch_size=1,
    )
    params = ",".join(PARAM_LIST)
    for seed in [44, 45, 46, 47]:
        command = make_opt_command(
            params=params,
            tracks=TRACKS_12,
            seed=seed,
            noise_scale=0.0,
            wandb_tags=wandb_tags,
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        s3df_submit(
            command,
            time="04:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
        )


def profile_fine_nosched_bs1_tol1e4_2k(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """Same as fine_nosched_bs1 plus coordinate freezing: --tol-per-param / --patience-per-param.

    Per-parameter freeze when movement vs t-``patience_per_param`` and each step in the
    window are relatively below ``tol_per_param`` (see ``run_optimization.run_trial``),
    not the global ``--tol`` / ``--patience`` vector criterion.
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="constant",
        max_steps=50000,
        tol=1e-6,
        patience=2000,
        tol_per_param=1e-4,
        patience_per_param=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/sched2_longer_schedule_20260430/no_sched_fine_tol1e4_p2000_PER_PARAM",
        grad_clip=10.0,
        lr_multipliers="velocity_cm_us:0.005",
        warmup_steps=1000,
        num_buckets=1000,
        step_size=0.1,
        max_num_deposits=50000,
        batch_size=1,
    )
    params = ",".join(PARAM_LIST)
    for seed in [44, 45, 46, 47]:
        command = make_opt_command(
            params=params,
            tracks=TRACKS_12,
            seed=seed,
            noise_scale=0.0,
            wandb_tags=wandb_tags,
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        s3df_submit(
            command,
            time="04:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
        )


def profile_no_schedule_less_params(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """Grow the fitted set around ``diffusion_long_cm2_us`` (3..7 params); transverse diffusion last.

    For each ``n_params`` in ``[3, 4, 5, 6, 7]`` and each seed in ``[44, 45, 46, 47]``, submit one
    job. ``run_optimization`` picks a distinct output folder per ``--params`` string; W&B gets an
    extra tag ``n_params_<k>`` to filter runs.
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="constant",
        max_steps=50000,
        tol=1e-6,
        patience=100,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/sched2_longer_schedule_20260430/no_schedule_less_params",
        grad_clip=10.0,
        lr_multipliers="velocity_cm_us:0.005",
        warmup_steps=1000,
        num_buckets=1000,
        step_size=0.1,
        max_num_deposits=50000,
        batch_size=1,
    )
    for n_params in range(3, 8):
        params = params_growing_with_long_diffusion(n_params)
        tags = list(wandb_tags) + [f"n_params_{n_params}"]
        for seed in [50]:
            command = make_opt_command(
                params=params,
                tracks=TRACKS_12,
                seed=seed,
                noise_scale=0.0,
                wandb_tags=tags,
                **shared,
            )
            if not print_sbatch_only:
                print(command)
            s3df_submit(
                command,
                time="04:00:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
            )


def profile_longitudinal_diffusion_only(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """Single-parameter fits: longitudinal diffusion only (12 tracks, seeds 44–47)."""
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="constant",
        max_steps=20000,
        tol=1e-6,
        patience=500,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/longitudinal_diffusion_only",
        grad_clip=10.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=0.1,
        max_num_deposits=50000,
        batch_size=1,
    )
    params = "diffusion_long_cm2_us"
    for seed in [44, 45, 46, 47]:
        command = make_opt_command(
            params=params,
            tracks=TRACKS_12,
            seed=seed,
            noise_scale=0.0,
            wandb_tags=wandb_tags,
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        s3df_submit(
            command,
            time="04:00:00",
            submit=submit,
            mem_gb=32,
            print_sbatch_command=print_sbatch_only,
        )


def profile_longitudinal_transverse_diffusion(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """Joint fit: longitudinal + transverse diffusion (12 tracks, seeds 44–47)."""
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="constant",
        max_steps=20000,
        tol=1e-6,
        patience=500,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/longitudinal_transverse_diffusion",
        grad_clip=10.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=0.1,
        max_num_deposits=50000,
        batch_size=1,
    )
    params = "diffusion_long_cm2_us,diffusion_trans_cm2_us"
    for seed in [44, 45, 46, 47]:
        command = make_opt_command(
            params=params,
            tracks=TRACKS_12,
            seed=seed,
            noise_scale=0.0,
            wandb_tags=wandb_tags,
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        s3df_submit(
            command,
            time="04:00:00",
            submit=submit,
            mem_gb=32,
            print_sbatch_command=print_sbatch_only,
        )


def profile_2_part_schedule(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """Two-phase coarse-to-fine: 1.0 mm for 5k steps, then 0.1 mm for 15k (20k total).

    Matches ``profile_3_part_schedule`` style but drops the middle 0.5 mm phase.
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="constant",
        max_steps=20000,
        tol=1e-6,
        patience=100,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/2_part_schedule_05012026",
        grad_clip=10.0,
        lr_multipliers="velocity_cm_us:0.005",
        warmup_steps=1000,
        num_buckets=1000,
        schedule_steps="5000",
        schedule_step_sizes="1.0,0.1",
        schedule_deposits="5000,50000",
        schedule_batch_sizes="5,1",
    )
    params = ",".join(PARAM_LIST)
    for seed in [44, 45, 46, 47]:
        command = make_opt_command(
            params=params,
            tracks=TRACKS_12,
            seed=seed,
            noise_scale=0.0,
            wandb_tags=wandb_tags,
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        s3df_submit(
            command,
            time="04:00:00",
            submit=submit,
            mem_gb=32,
            print_sbatch_command=print_sbatch_only,
        )


def profile_2_part_schedule_cosine_30k(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """Two-phase coarse-to-fine like ``profile_2_part_schedule``, cosine LR over 30k steps.

    Phase 0: step_size=1.0 mm (steps 0–5000), phase 1: 0.1 mm (steps 5000–30000).
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=100,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/2_part_schedule_cosine_30k",
        grad_clip=10.0,
        lr_multipliers="velocity_cm_us:0.005",
        warmup_steps=1000,
        num_buckets=1000,
        schedule_steps="5000",
        schedule_step_sizes="1.0,0.1",
        schedule_deposits="5000,50000",
        schedule_batch_sizes="5,1",
    )
    params = ",".join(PARAM_LIST)
    for seed in [44, 45, 46, 47]:
        command = make_opt_command(
            params=params,
            tracks=TRACKS_12,
            seed=seed,
            noise_scale=0.0,
            wandb_tags=wandb_tags,
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        s3df_submit(
            command,
            time="04:00:00",
            submit=submit,
            mem_gb=32,
            print_sbatch_command=print_sbatch_only,
        )


def profile_timing_study_diag50mev(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """Deposit-buffer sweep on one 50 MeV diagonal track; 1000 steps for timing / OOM boundary."""
    deposits_list = (5000, 20000, 40000, 50000, 60000, 80000, 100000)
    shared_base = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="constant",
        max_steps=1000,
        tol=1e-6,
        patience=5000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/timing_study_diag50mev",
        grad_clip=10.0,
        lr_multipliers="velocity_cm_us:0.005",
        warmup_steps=100,
        num_buckets=1000,
        step_size=0.1,
        batch_size=1,
        log_interval=1000,
    )
    params = ",".join(PARAM_LIST)
    tags_base = list(wandb_tags) if wandb_tags else []
    for dep in deposits_list:
        tags = tags_base + [f"dep_{dep}"]
        command = make_opt_command(
            params=params,
            tracks=TRACK_DIAG_50MEV,
            seed=47,
            noise_scale=0.0,
            max_num_deposits=dep,
            gt_max_deposits=dep,
            wandb_tags=tags,
            **shared_base,
        )
        if not print_sbatch_only:
            print(command)
        s3df_submit(
            command,
            time="04:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
        )


def profile_timing_study_cont(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """Fine deposit sweep around 50k pads on one 50 MeV diagonal track; 1000 steps."""
    deposits_list = (45000, 50000, 55000)
    shared_base = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="constant",
        max_steps=1000,
        tol=1e-6,
        patience=5000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/timing_study_cont",
        grad_clip=10.0,
        lr_multipliers="velocity_cm_us:0.005",
        warmup_steps=100,
        num_buckets=1000,
        step_size=0.1,
        batch_size=1,
        log_interval=1000,
    )
    params = ",".join(PARAM_LIST)
    tags_base = list(wandb_tags) if wandb_tags else []
    for dep in deposits_list:
        tags = tags_base + [f"dep_{dep}"]
        command = make_opt_command(
            params=params,
            tracks=TRACK_DIAG_50MEV,
            seed=47,
            noise_scale=0.0,
            max_num_deposits=dep,
            gt_max_deposits=dep,
            wandb_tags=tags,
            **shared_base,
        )
        if not print_sbatch_only:
            print(command)
        s3df_submit(
            command,
            time="04:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
        )


PROFILES = {
    "3_part_schedule": profile_3_part_schedule,
    "2_part_schedule": profile_2_part_schedule,
    "2_part_schedule_cosine_30k": profile_2_part_schedule_cosine_30k,
    "fine_nosched_bs1": profile_fine_nosched_bs1,
    "fine_nosched_bs1_tol1e4_p2000": profile_fine_nosched_bs1_tol1e4_2k,
    "no_schedule_less_params": profile_no_schedule_less_params,
    "longitudinal_diffusion_only": profile_longitudinal_diffusion_only,
    "longitudinal_transverse_diffusion": profile_longitudinal_transverse_diffusion,
    "timing_study_diag50mev": profile_timing_study_diag50mev,
    "timing_study_cont": profile_timing_study_cont,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profile", nargs="?", choices=list(PROFILES),
                        help="Submission profile to run")
    parser.add_argument(
        "--restart-preempted",
        metavar="RESULTS_DIR",
        help="Scan RESULTS_DIR and resubmit incomplete/preempted jobs",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit jobs (default is dry-run)",
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Profiles only: write batch scripts and print sbatch lines instead of submitting. "
        "Ignored with --restart-preempted (restart dry-run prints sbatch lines without this flag).",
    )
    parser.add_argument(
        "--time",
        default="10:00:00",
        help="Wall time for resubmitted jobs (default: 10:00:00)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="GPUs for resubmitted jobs (default: 1)",
    )
    parser.add_argument(
        "--mem-gb",
        type=int,
        default=32,
        help="Memory in GB for resubmitted jobs (default: 32)",
    )
    parser.add_argument(
        "--wandb-extra-tags",
        default=None,
        metavar="TAGS",
        help="Comma-separated extra W&B tags for profile runs (profile name is always included first).",
    )
    args = parser.parse_args()

    if args.profile and args.restart_preempted:
        parser.error("Choose either a profile or --restart-preempted, not both.")

    if args.print_commands and args.submit:
        parser.error("Use either --print-commands or --submit, not both.")

    if args.restart_preempted:
        resubmit_preempted(
            args.restart_preempted,
            time=args.time,
            gpus=args.gpus,
            mem_gb=args.mem_gb,
            submit=args.submit,
            print_sbatch_command=not args.submit,
        )
    elif args.profile:
        wandb_tags = [args.profile]
        if args.wandb_extra_tags:
            wandb_tags.extend(
                t.strip() for t in args.wandb_extra_tags.split(",") if t.strip()
            )
        PROFILES[args.profile](
            submit=not args.print_commands,
            print_sbatch_only=args.print_commands,
            wandb_tags=wandb_tags,
        )
    else:
        parser.error("Provide a profile or use --restart-preempted RESULTS_DIR.")
