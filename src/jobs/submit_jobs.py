#!/usr/bin/env python3
"""
Helpers for building run_optimization.py commands and submitting them to S3DF.
"""
import glob
import os
import pickle

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
    return " ".join(parts)


def resubmit_preempted(results_dir: str, *, time: str = "10:00:00",
                       gpus: int = 1, mem_gb: int = 32, submit: bool = False):
    """Scan results_dir for incomplete pkl files and resubmit their jobs.

    Each pkl stores the original command in a 'command' field (added in newer
    runs). Pkls without that field are skipped with a warning.
    """
    pkls = sorted(glob.glob(os.path.join(results_dir, "**", "result_*.pkl"), recursive=True))
    if not pkls:
        print(f"No pkl files found under {results_dir}")
        return

    resubmitted = 0
    for pkl_path in pkls:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        n_done  = len(data.get('trials', []))
        n_total = data.get('N', -1)
        if n_done >= n_total:
            continue
        command = data.get('command')
        if not command:
            print(f"  SKIP {pkl_path}: no 'command' field (needs a run with newer code)")
            continue
        print(f"  {'Submitting' if submit else 'Would submit'} "
              f"{pkl_path}  ({n_done}/{n_total} trials done)")
        s3df_submit(command, time=time, gpus=gpus, mem_gb=mem_gb, submit=submit)
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
        print(f"  {'Submitting' if submit else 'Would submit'} "
              f"{txt_path}  (no pkl — preempted before first write)")
        s3df_submit(command, time=time, gpus=gpus, mem_gb=mem_gb, submit=submit)
        resubmitted += 1

    noun   = "job" if resubmitted == 1 else "jobs"
    action = "submitted" if submit else "found (pass submit=True to actually submit)"
    print(f"\n{resubmitted} {noun} {action}.")


if __name__ == "__main__":
    # diagonal + X + Y + Z at 1000, 100, and 50 MeV in one run
    TRACKS_12 = (
        "diagonal+X+Y+Z"
        "+diagonal_100MeV:1,1,1:100+x100:1,0,0:100+y100:0,1,0:100+z100:0,0,1:100"
        "+diagonal_50MeV:1,1,1:50+x50:1,0,0:50+y50:0,1,0:50+z50:0,0,1:50"
    )

    PARAM_LIST = [p.strip() for p in ALL_PARAMS.split(",") if p.strip()]

    '''SHARED = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="constant",
        max_steps=40000,
        tol=1e-6,
        patience=100,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/all_params_bugfix_20260428_2",
        grad_clip=10.0,
        lr_multipliers="velocity_cm_us:0.005",
        warmup_steps=1000,
        step_size=1.0,
        max_num_deposits=5_000,
        num_buckets=1000,
        batch_size=5
    )'''

    '''for n_params in range(1, len(PARAM_LIST) + 1):
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
   '''
    '''cmd = make_opt_command(
        params=ALL_PARAMS,
        tracks=TRACKS_12,
        seed=42,
        noise_scale=0.0,
        **SHARED,
    )
    print(cmd)
    s3df_submit(cmd, time="05:00:00", submit=True)'''

    # ── Resolution schedule: coarse-to-fine over step size + segment count ────
    # Phase 0 → 1 → 2 → 3 : step_size 1.0 → 0.5 → 0.25 → 0.1 mm
    # Deposits scale proportionally: 5k → 10k → 20k → 50k
    # Batch size scales inversely to keep GPU memory roughly constant.

    SCHED_SHARED = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="constant",
        max_steps=40000,
        tol=1e-6,
        patience=100,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/sched2_longer_schedule",
        grad_clip=10.0,
        lr_multipliers="velocity_cm_us:0.005",
        warmup_steps=1000,
        num_buckets=1000,
        schedule_steps="2000",
        schedule_step_sizes="1.0,0.1",
        schedule_deposits="5000,50000",
        schedule_batch_sizes="5,1"
    )

    #for n_params in range(1, len(PARAM_LIST) + 1):
    params = ",".join(PARAM_LIST)
    for seed in [44, 45, 46, 47]:
        command = make_opt_command(
            params=params,
            tracks=TRACKS_12,
            seed=seed,
            noise_scale=0.0,
            **SCHED_SHARED,
        )
        print(command)
        s3df_submit(command, time="04:00:00", submit=True, mem_gb=64)
