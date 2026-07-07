#!/usr/bin/env python3
"""
Submit E-field SIREN calibration jobs to S3DF.

Each job runs run_optimization.py with --params Efield for 10k steps and saves
MLP snapshots at steps 0, 50, 100, 200, 400 and then every 500 steps.

Usage
-----
  python src/jobs/submit_jobs_E_field_calibration.py <profile>
  python src/jobs/submit_jobs_E_field_calibration.py <profile> --print-commands
  python src/jobs/submit_jobs_E_field_calibration.py <profile> --submit

Available profiles
------------------
  baseline                Single run: default arch/seed/no regularisation. Good sanity check.
  seed_sweep              4 track seeds × 4 NN seeds = 16 jobs, no regularisation.
  arch_sweep              3 architectures × 2 NN seeds = 6 jobs.
  rotor_sweep             6 rotor-penalty weights × 2 NN seeds = 12 jobs.
  dropout_sweep           5 dropout rates × 2 NN seeds = 10 jobs.
  full_sweep              arch × rotor × dropout × seeds — large grid, submit carefully.
  tracks100_ebs20_rotor   100 tracks, eff. batch 20, no dropout; rotor weight 0 and 1 (2 jobs).
  1k_tracks_gt_precompute Precompute GT signals for 1k_tracks_sweep's 2 track ensembles,
                          sharded (--n-gt-shards, default 50) and chained; run to
                          completion before 1k_tracks_sweep (2 x n_gt_shards jobs).
                          Pass --only-missing to re-submit just the shards whose h5
                          output doesn't exist yet (e.g. after some jobs timed out/OOMed).
  1k_tracks_sweep         1000 tracks, batch=2/eff.batch=16, rotor {0,1} x T-range
                          {100-1000,500-1500} MeV = 4 jobs. Use --time 08:00:00.
                          Loads GT cache from 1k_tracks_gt_precompute by default
                          (--no-gt-cache to recompute instead; --n-gt-shards must match).
  100_tracks_sweep        Same as 1k_tracks_sweep but only the first 100 of the 1000
                          tracks (4 jobs). Reuses the same GT cache — with the default
                          --n-gt-shards 50 this loads just the first 5 shards, no
                          separate precompute needed.

All profiles accept optional overrides via the CLI; see --help.

Arguments visible in every run label / wandb tag:
  efield file, tracks seed, NN seed, hidden dims, dropout rate, rotor weight.
"""
import argparse
import itertools
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT))

from job_submission_tools import s3df_submit

# ── Default S3DF paths ────────────────────────────────────────────────────────
REMOTE_RESULTS_DIR = '/fs/ddn/sdf/group/atlas/d/gregork/jaxtpc/results'
REMOTE_EFIELD_NPZ  = (
    '/fs/ddn/sdf/group/atlas/d/gregork/jaxtpc/results'
    '/efield_distortions/sce_maps_jaxtpc_41.npz'
)
REMOTE_CONSERVATIVE_EFIELD_NPZ = (
    '/sdf/home/g/gregork/jaxtpc/results'
    '/efield_distortions/sce_maps_jaxtpc_conservative_41.npz'
)
REMOTE_CODE_DIR = '/sdf/home/g/gregork/jaxtpc'

# ── Snapshot schedule ─────────────────────────────────────────────────────────
# Steps 0,50,100,200,400 explicitly + every 500 thereafter → 0,50,100,200,400,500,1000,…,10000
SNAPSHOT_STEPS    = '0,50,100,200,400'
SNAPSHOT_INTERVAL = 500
MAX_STEPS         = 10_000


def _hidden_tag(hidden):
    """'[32,32,32]' → '32x3'."""
    if len(set(hidden)) == 1:
        return f'{hidden[0]}x{len(hidden)}'
    return 'x'.join(str(h) for h in hidden)


def _max_deposits_for_ke(ke_hi):
    """Safe --gt-max-deposits/--max-num-deposits for a given upper kinetic-energy
    bound (MeV). generate_muon_track is called without detector_bounds_mm in
    run_optimization.py, so at --gt-step-size/--step-size 1.0mm the deposit count is
    purely energy-determined (dE/dx integration down to min_energy_mev=10) — an exact,
    direction/geometry-independent ceiling. Verified locally: KE=1000 -> 4409 steps
    (safe under the 5000 default, ~13% headroom); KE=1500 -> 6543 steps, which overflows
    5000 (RuntimeError: "Volume N has X deposits > total_pad") — 7500 restores similar
    headroom for the 500-1500 MeV range used by ke_range_mev=(500, 1500) jobs.
    """
    return 7500 if ke_hi > 1000.0 else 5000


def make_efield_calib_command(
    *,
    electric_dist_path=REMOTE_EFIELD_NPZ,
    tracks_random_seed=42,
    nn_seed=0,
    hidden=(32, 32, 32),
    dropout_rate=0.0,
    penalize_rotor=0.0,
    results_base,
    wandb_tags=None,
    # fixed hyper-params matching the local scripts
    lr=1e-3,
    lr_mult=10.0,
    max_steps=MAX_STEPS,
    n_random_tracks=15,
    effective_batch_size=15,
    batch_size=1,
    ke_range_mev=None,
    max_deposits=5000,
    track_shard=None,
    gt_cache_save=None,
    gt_cache_load=None,
    exit_after_gt_cache=False,
    no_wandb=False,
):
    """Return a run_optimization.py command string for E-field calibration."""
    hidden_str = ' '.join(str(h) for h in hidden)
    tags = list(wandb_tags or []) + [
        'efield_siren', 'calib',
        f'arch_{_hidden_tag(hidden)}',
        f'rotor{penalize_rotor:g}',
        f'drop{dropout_rate:g}',
        f'trkseed{tracks_random_seed}',
        f'nnseed{nn_seed}',
    ]
    if ke_range_mev is not None:
        tags.append(f'Trange{ke_range_mev[0]:g}-{ke_range_mev[1]:g}')
    tags_csv = ','.join(t.strip() for t in tags if t.strip())

    parts = [
        f'python {REMOTE_CODE_DIR}/src/opt/run_optimization.py',
        '--params Efield',
        f'--electric-dist-path {electric_dist_path}',
        f'--N-random-tracks {n_random_tracks}',
        f'--tracks-random-seed {tracks_random_seed}',
        f'--seed {nn_seed}',
        '--N 1',
        '--optimizer adam',
        f'--lr {lr}',
        '--lr-schedule cosine',
        '--adam-beta2 0.9',
        '--warmup-steps 200',
        f'--max-steps {max_steps}',
        '--tol 1e-9',
        f'--patience {max_steps}',
        '--loss sobolev_loss_geomean_log1p',
        '--sobolev-exponent 2.0',
        f'--max-num-deposits {max_deposits}',
        '--num-buckets 1000',
        f'--gt-max-deposits {max_deposits}',
        '--gt-step-size 1.0',
        '--step-size 1.0',
        f'--batch-size {batch_size}',
        f'--effective-batch-size {effective_batch_size}',
        f'--efield-hidden {hidden_str}',
        f'--efield-lr-mult {lr_mult}',
        '--noise-scale 1.0',
        '--clip-grad-norm 1.0',
        '--log-interval 50',
        f'--mlp-snapshot-steps {SNAPSHOT_STEPS}',
        f'--mlp-snapshot-interval {SNAPSHOT_INTERVAL}',
        f'--results-base {results_base}',
        f'--wandb-tags {tags_csv}',
    ]
    if penalize_rotor > 0.0:
        parts.append(f'--penalize-rotor {penalize_rotor:g}')
    if dropout_rate > 0.0:
        parts.append(f'--efield-dropout-rate {dropout_rate:g}')
    if ke_range_mev is not None:
        parts.append(f'--track-energy-range-mev {ke_range_mev[0]:g} {ke_range_mev[1]:g}')
    if track_shard is not None:
        parts.append(f'--track-shard {track_shard[0]} {track_shard[1]}')
    if gt_cache_save is not None:
        parts.append(f'--gt-cache-save {gt_cache_save}')
    if gt_cache_load is not None:
        parts.append(f'--gt-cache-load {" ".join(gt_cache_load)}')
    if exit_after_gt_cache:
        parts.append('--exit-after-gt-cache')
    if no_wandb:
        parts.append('--no-wandb')

    return ' \\\n  '.join(parts)


def _track_shard_ranges(n_total, n_shards):
    """Split [0, n_total) into n_shards contiguous, near-equal-size ranges."""
    base, rem = divmod(n_total, n_shards)
    ranges = []
    start = 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def _gt_cache_shard_path(results_base_prefix, ke_lo, ke_hi, shard_idx, n_shards):
    """Shared path convention: written by profile_1k_tracks_gt_precompute, read by
    profile_*_tracks_sweep's --gt-cache-load — both must use the same results_base_prefix."""
    return (f'{results_base_prefix}/1k_tracks_sweep/gt_cache/ke{ke_lo:g}-{ke_hi:g}'
            f'/shard{shard_idx}of{n_shards}.h5')


def _n_leading_shards_covering(n_active_tracks, n_total_tracks, n_gt_shards):
    """Number of leading shards (from _track_shard_ranges) whose union is exactly
    [0, n_active_tracks) — i.e. how many of the precompute's shard files a sweep using
    only the first n_active_tracks needs to load. Raises if n_active_tracks doesn't fall
    on a shard boundary for the given n_total_tracks/n_gt_shards."""
    ranges = _track_shard_ranges(n_total_tracks, n_gt_shards)
    for k, (_, end) in enumerate(ranges, start=1):
        if end == n_active_tracks:
            return k
    raise ValueError(
        f'{n_active_tracks} tracks does not land on a GT-cache shard boundary for '
        f'n_total_tracks={n_total_tracks}, n_gt_shards={n_gt_shards} '
        f'(boundaries: {[end for _, end in ranges]}). Pick n_active_tracks equal to '
        f'one of those boundaries, or adjust --n-gt-shards.')


def _submit_grid(
    *,
    efield_files,
    tracks_seeds,
    nn_seeds,
    hidden_choices,
    dropout_rates,
    rotor_weights,
    results_base_prefix,
    profile_tag,
    submit,
    print_sbatch_only,
    time='03:00:00',
    mem_gb=64,
):
    """Enumerate the full Cartesian product and submit one job per combination."""
    n_jobs = 0
    for ef, ts, ns, hidden, dr, rw in itertools.product(
        efield_files, tracks_seeds, nn_seeds,
        hidden_choices, dropout_rates, rotor_weights,
    ):
        arch_tag  = _hidden_tag(hidden)
        ef_stem   = Path(ef).stem
        tag       = (f'{profile_tag}/{ef_stem}/arch{arch_tag}'
                     f'/trk{ts}_nn{ns}_do{dr:g}_rot{rw:g}')
        results_base = f'{results_base_prefix}/{tag}'

        cmd = make_efield_calib_command(
            electric_dist_path=ef,
            tracks_random_seed=ts,
            nn_seed=ns,
            hidden=hidden,
            dropout_rate=dr,
            penalize_rotor=rw,
            results_base=results_base,
            wandb_tags=[profile_tag],
        )

        if not print_sbatch_only:
            print(cmd)
            print()

        s3df_submit(
            cmd,
            time=time,
            submit=submit,
            mem_gb=mem_gb,
            print_sbatch_command=print_sbatch_only,
        )
        n_jobs += 1

    if not print_sbatch_only:
        print(f'[submit_jobs_E_field_calibration] {n_jobs} job(s) for profile "{profile_tag}"')


# ── Profiles ──────────────────────────────────────────────────────────────────

def profile_baseline(*, submit, print_sbatch_only, results_base_prefix, **_):
    _submit_grid(
        efield_files    = [REMOTE_EFIELD_NPZ],
        tracks_seeds    = [42],
        nn_seeds        = [0],
        hidden_choices  = [(32, 32, 32)],
        dropout_rates   = [0.0],
        rotor_weights   = [0.0],
        results_base_prefix = results_base_prefix,
        profile_tag     = 'baseline',
        submit          = submit,
        print_sbatch_only = print_sbatch_only,
    )


def profile_seed_sweep(*, submit, print_sbatch_only, results_base_prefix, **_):
    _submit_grid(
        efield_files    = [REMOTE_EFIELD_NPZ],
        tracks_seeds    = [42, 43, 44, 45],
        nn_seeds        = [0, 1, 2, 3],
        hidden_choices  = [(32, 32, 32)],
        dropout_rates   = [0.0],
        rotor_weights   = [0.0],
        results_base_prefix = results_base_prefix,
        profile_tag     = 'seed_sweep',
        submit          = submit,
        print_sbatch_only = print_sbatch_only,
    )


def profile_arch_sweep(*, submit, print_sbatch_only, results_base_prefix, **_):
    _submit_grid(
        efield_files    = [REMOTE_EFIELD_NPZ],
        tracks_seeds    = [42],
        nn_seeds        = [0, 1],
        hidden_choices  = [(16, 16, 16), (32, 32, 32), (64, 64, 64)],
        dropout_rates   = [0.0],
        rotor_weights   = [0.0],
        results_base_prefix = results_base_prefix,
        profile_tag     = 'arch_sweep',
        submit          = submit,
        print_sbatch_only = print_sbatch_only,
    )


def profile_rotor_sweep(*, submit, print_sbatch_only, results_base_prefix, **_):
    _submit_grid(
        efield_files    = [REMOTE_EFIELD_NPZ],
        tracks_seeds    = [42],
        nn_seeds        = [0, 1],
        hidden_choices  = [(32, 32, 32)],
        dropout_rates   = [0.0],
        rotor_weights   = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0],
        results_base_prefix = results_base_prefix,
        profile_tag     = 'rotor_sweep',
        submit          = submit,
        print_sbatch_only = print_sbatch_only,
    )


def profile_dropout_sweep(*, submit, print_sbatch_only, results_base_prefix, **_):
    _submit_grid(
        efield_files    = [REMOTE_EFIELD_NPZ],
        tracks_seeds    = [42],
        nn_seeds        = [0, 1],
        hidden_choices  = [(32, 32, 32)],
        dropout_rates   = [0.0, 0.1, 0.2, 0.3, 0.5],
        rotor_weights   = [0.0],
        results_base_prefix = results_base_prefix,
        profile_tag     = 'dropout_sweep',
        submit          = submit,
        print_sbatch_only = print_sbatch_only,
    )


def profile_conservative_curl_sweep(*, submit, print_sbatch_only, results_base_prefix, time, mem_gb,
                                     chain=True, **_):
    """
    Conservative E-field sweep against the curl-free NPZ.

    3 configs × 2 rotor weights × 3 NN seeds = 18 jobs.
    Configs:
      - 32³  baseline (no dropout)
      - 64³  2× wider  (no dropout)
      - 32³  + dropout=0.1

    Each NN seed is submitted as an independent chain of 6 sequential jobs
    (rotor=0 and rotor=1 for each arch/dropout combo run one after another),
    unless --no-chain is passed, in which case all jobs are submitted independently.
    """
    EF = REMOTE_CONSERVATIVE_EFIELD_NPZ
    DROPOUT = 0.1
    CONFIGS = [
        dict(hidden=(32, 32, 32), dropout_rate=0.0,     penalize_rotor=0.0),
        dict(hidden=(32, 32, 32), dropout_rate=0.0,     penalize_rotor=1.0),
        dict(hidden=(64, 64, 64), dropout_rate=0.0,     penalize_rotor=0.0),
        dict(hidden=(64, 64, 64), dropout_rate=0.0,     penalize_rotor=1.0),
        dict(hidden=(32, 32, 32), dropout_rate=DROPOUT, penalize_rotor=0.0),
        dict(hidden=(32, 32, 32), dropout_rate=DROPOUT, penalize_rotor=1.0),
    ]
    profile_tag = 'conservative_curl_sweep'

    for nn_seed in [0, 1, 2]:
        prev_job_id = None
        for cfg in CONFIGS:
            hidden     = cfg['hidden']
            dr         = cfg['dropout_rate']
            rw         = cfg['penalize_rotor']
            arch_tag   = _hidden_tag(hidden)
            ef_stem    = Path(EF).stem
            tag = (f'{profile_tag}/{ef_stem}/arch{arch_tag}'
                   f'/trk42_nn{nn_seed}_do{dr:g}_rot{rw:g}')
            results_base = f'{results_base_prefix}/{tag}'

            cmd = make_efield_calib_command(
                electric_dist_path=EF,
                tracks_random_seed=42,
                nn_seed=nn_seed,
                hidden=hidden,
                dropout_rate=dr,
                penalize_rotor=rw,
                results_base=results_base,
                wandb_tags=[profile_tag],
            )

            if not print_sbatch_only:
                print(cmd)
                print()

            job_id = s3df_submit(
                cmd,
                time=time,
                submit=submit,
                mem_gb=mem_gb,
                print_sbatch_command=print_sbatch_only,
                dependency=prev_job_id if chain else None,
            )
            if chain:
                prev_job_id = job_id

    if not print_sbatch_only:
        print(f'[submit_jobs_E_field_calibration] 18 jobs for profile "{profile_tag}"')


def profile_tracks100_ebs20_rotor(*, submit, print_sbatch_only, results_base_prefix, time, mem_gb, **_):
    """100-track run, effective batch size 20, no dropout; rotor weight 0 and 1 (2 jobs).

    Each optimizer step accumulates gradients over 20 randomly-drawn tracks (batch_size=1
    per vmap call, 20 accumulation steps). No dropout. Two jobs submitted: one without the
    curl (rotor) penalty and one with weight=1.
    """
    profile_tag = 'tracks100_ebs20_rotor'
    for rw in [0.0, 1.0]:
        tag = f'{profile_tag}/arch32x3/trk42_nn0_do0_rot{rw:g}'
        results_base = f'{results_base_prefix}/{tag}'
        cmd = make_efield_calib_command(
            electric_dist_path=REMOTE_EFIELD_NPZ,
            tracks_random_seed=42,
            nn_seed=0,
            hidden=(32, 32, 32),
            dropout_rate=0.0,
            penalize_rotor=rw,
            results_base=results_base,
            wandb_tags=[profile_tag],
            n_random_tracks=100,
            effective_batch_size=20,
        )
        if not print_sbatch_only:
            print(cmd)
            print()
        s3df_submit(
            cmd,
            time=time,
            submit=submit,
            mem_gb=mem_gb,
            print_sbatch_command=print_sbatch_only,
        )

    if not print_sbatch_only:
        print(f'[submit_jobs_E_field_calibration] 2 jobs for profile "{profile_tag}"')


def profile_1k_tracks_gt_precompute(*, submit, print_sbatch_only, results_base_prefix, time, mem_gb,
                                     chain=True, n_gt_shards=50, only_missing=False, **_):
    """Precompute GT signals for 1k_tracks_sweep, sharded per kinetic-energy range.

    GT-signal generation (generate_muon_track -> build_deposit_data -> simulator.forward,
    once per track) doesn't depend on rotor weight — only on the track ensemble (seed,
    N, kinetic-energy range). 1k_tracks_sweep's 4 jobs cover only 2 distinct ensembles
    (rotor is a training-time regulariser), so precomputing each ensemble's GT signals
    once here, split across n_gt_shards jobs (default 50; 1000/50 = 20 tracks/shard),
    avoids redundant work: short (--exit-after-gt-cache, no training) jobs, instead of
    paying the expensive per-track GT loop again inside each of the 4 real training
    jobs. Each job's peak host memory scales with tracks-per-shard (the per-track GT
    signal arrays accumulate in memory for the whole shard before being flushed to h5),
    so if a shard still gets OOM-killed, raise --n-gt-shards further (or --mem-gb).

    By default all shards (both kinetic-energy ranges) run as a single sequential
    dependency chain, matching profile_conservative_curl_sweep's style; pass --no-chain
    to submit them all independently/in parallel instead.

    Pass --only-missing to skip any shard whose output h5 file already exists on disk
    (checked directly against results_base_prefix — only meaningful when actually
    running on S3DF, where that path is real) — use this to re-submit just the shards
    that failed/timed out on a prior attempt, instead of redoing all n_gt_shards*2.

    Run this profile to completion FIRST (same --results-base as 1k_tracks_sweep), then
    run 1k_tracks_sweep, which loads these h5 caches by default (see --no-gt-cache;
    it must be invoked with the same --n-gt-shards used here).
    """
    profile_tag = '1k_tracks_gt_precompute'
    n_jobs = 0
    n_skipped = 0
    prev_job_id = None
    for ke_lo, ke_hi in [(100.0, 1000.0), (500.0, 1500.0)]:
        for shard_idx, (start, end) in enumerate(_track_shard_ranges(1000, n_gt_shards)):
            cache_path = _gt_cache_shard_path(results_base_prefix, ke_lo, ke_hi, shard_idx, n_gt_shards)
            if only_missing and os.path.exists(cache_path):
                n_skipped += 1
                continue
            tag = f'{profile_tag}/ke{ke_lo:g}-{ke_hi:g}/shard{shard_idx}of{n_gt_shards}'
            results_base = f'{results_base_prefix}/{tag}'
            cmd = make_efield_calib_command(
                electric_dist_path=REMOTE_EFIELD_NPZ,
                tracks_random_seed=42,
                nn_seed=0,
                hidden=(32, 32, 32),
                dropout_rate=0.0,
                penalize_rotor=0.0,
                results_base=results_base,
                wandb_tags=[profile_tag],
                n_random_tracks=1000,
                ke_range_mev=(ke_lo, ke_hi),
                max_deposits=_max_deposits_for_ke(ke_hi),
                track_shard=(start, end),
                gt_cache_save=cache_path,
                exit_after_gt_cache=True,
                no_wandb=True,
            )
            if not print_sbatch_only:
                print(cmd)
                print()
            job_id = s3df_submit(
                cmd,
                time=time,
                submit=submit,
                mem_gb=mem_gb,
                print_sbatch_command=print_sbatch_only,
                dependency=prev_job_id if chain else None,
            )
            if chain:
                prev_job_id = job_id
            n_jobs += 1

    if not print_sbatch_only:
        chain_note = 'chained' if chain else 'independent'
        skip_note = f', {n_skipped} already-done shard(s) skipped (--only-missing)' if only_missing else ''
        print(f'[submit_jobs_E_field_calibration] {n_jobs} {chain_note} jobs for profile '
              f'"{profile_tag}"{skip_note}')


def _submit_efield_tracks_sweep(*, submit, print_sbatch_only, results_base_prefix, time, mem_gb,
                                 chain, use_gt_cache, n_gt_shards, profile_tag, n_active_tracks,
                                 n_total_tracks=1000, effective_batch_size=16, batch_size=2):
    """Shared body for profile_1k_tracks_sweep / profile_100_tracks_sweep.

    Runs the same N_random_tracks=n_total_tracks ensemble (tracks_random_seed=42) as the
    GT precompute step, then restricts to the first n_active_tracks via --track-shard
    when n_active_tracks < n_total_tracks — this reuses the SAME deterministic track
    generation as profile_1k_tracks_gt_precompute, so the leading n_active_tracks are
    byte-identical to (and their GT cache reusable from) the full n_total_tracks run.
    Sweeps rotor {0, 1} x kinetic-energy range {100-1000, 500-1500} MeV = 4 jobs.
    """
    n_shards_needed = (
        _n_leading_shards_covering(n_active_tracks, n_total_tracks, n_gt_shards)
        if n_active_tracks < n_total_tracks else n_gt_shards
    )
    prev_job_id = None
    for ke_lo, ke_hi in [(100.0, 1000.0), (500.0, 1500.0)]:
        gt_cache_load = (
            [_gt_cache_shard_path(results_base_prefix, ke_lo, ke_hi, i, n_gt_shards)
             for i in range(n_shards_needed)]
            if use_gt_cache else None
        )
        for rw in [0.0, 1.0]:
            tag = f'{profile_tag}/arch32x3/trk42_nn0_do0_rot{rw:g}_ke{ke_lo:g}-{ke_hi:g}'
            results_base = f'{results_base_prefix}/{tag}'
            cmd = make_efield_calib_command(
                electric_dist_path=REMOTE_EFIELD_NPZ,
                tracks_random_seed=42,
                nn_seed=0,
                hidden=(32, 32, 32),
                dropout_rate=0.0,
                penalize_rotor=rw,
                results_base=results_base,
                wandb_tags=[profile_tag],
                n_random_tracks=n_total_tracks,
                effective_batch_size=effective_batch_size,
                batch_size=batch_size,
                ke_range_mev=(ke_lo, ke_hi),
                max_deposits=_max_deposits_for_ke(ke_hi),
                gt_cache_load=gt_cache_load,
                track_shard=(0, n_active_tracks) if n_active_tracks < n_total_tracks else None,
            )
            if not print_sbatch_only:
                print(cmd)
                print()
            job_id = s3df_submit(
                cmd,
                time=time,
                submit=submit,
                mem_gb=mem_gb,
                print_sbatch_command=print_sbatch_only,
                dependency=prev_job_id if chain else None,
            )
            if chain:
                prev_job_id = job_id

    if not print_sbatch_only:
        chain_note = 'chained' if chain else 'independent'
        print(f'[submit_jobs_E_field_calibration] 4 {chain_note} jobs for profile "{profile_tag}"')


def profile_1k_tracks_sweep(*, submit, print_sbatch_only, results_base_prefix, time, mem_gb,
                             chain=True, use_gt_cache=True, n_gt_shards=50, **_):
    """1000-track run, batch_size=2, effective batch size 16, no dropout, arch 32x3.

    Only 2 tracks are held on GPU at once (--batch-size 2); gradients accumulate over
    16 microbatches (--effective-batch-size 16) before each optimizer step. Sweeps
    2 rotor weights (0, 1) x 2 kinetic-energy ranges (the default T~U[100,1000] MeV and
    a shifted T~U[500,1500] MeV) = 4 jobs, all with tracks_random_seed=42, nn_seed=0.

    Per wandb runs ctyg4znw/hra9nlws, peak GPU memory is governed by batch_size /
    effective_batch_size, not track count, so this is memory-safe; but wall-clock time
    scales with N_random_tracks x effective_batch_size and is estimated at ~4-5h for
    10k steps — pass --time 08:00:00 (or larger) when invoking this profile.

    By default the 4 jobs are submitted as a single sequential dependency chain (each
    job runs only after the previous one finishes), matching profile_conservative_curl_sweep's
    style; pass --no-chain to submit all 4 independently instead.

    By default each job loads its GT signals from the n_gt_shards-shard h5 cache written
    by profile_1k_tracks_gt_precompute (must have been run first with the same
    --results-base AND the same --n-gt-shards), skipping the expensive per-track GT loop
    entirely. Pass --no-gt-cache to recompute from scratch instead (e.g. if the
    precompute profile hasn't been run yet).
    """
    _submit_efield_tracks_sweep(
        submit=submit, print_sbatch_only=print_sbatch_only, results_base_prefix=results_base_prefix,
        time=time, mem_gb=mem_gb, chain=chain, use_gt_cache=use_gt_cache, n_gt_shards=n_gt_shards,
        profile_tag='1k_tracks_sweep', n_active_tracks=1000, n_total_tracks=1000,
    )


def profile_100_tracks_sweep(*, submit, print_sbatch_only, results_base_prefix, time, mem_gb,
                              chain=True, use_gt_cache=True, n_gt_shards=50, **_):
    """Same as profile_1k_tracks_sweep, but trains on only the first 100 of the 1000 tracks.

    Otherwise identical: batch_size=2, effective_batch_size=16, arch 32x3, no dropout,
    rotor {0, 1} x kinetic-energy range {100-1000, 500-1500} MeV = 4 jobs.

    Reuses the SAME GT cache as profile_1k_tracks_sweep — the first 100 tracks of the
    1000-track ensemble are byte-identical to a from-scratch 100-track ensemble (track
    generation is a sequential RNG draw per track, independent of the total N), so with
    the default --n-gt-shards 50 (20 tracks/shard) this loads only the first 5 shards
    per kinetic-energy range instead of recomputing anything — no separate precompute
    step needed, as long as profile_1k_tracks_gt_precompute has already been run.
    """
    _submit_efield_tracks_sweep(
        submit=submit, print_sbatch_only=print_sbatch_only, results_base_prefix=results_base_prefix,
        time=time, mem_gb=mem_gb, chain=chain, use_gt_cache=use_gt_cache, n_gt_shards=n_gt_shards,
        profile_tag='100_tracks_sweep', n_active_tracks=100, n_total_tracks=1000,
    )


def profile_full_sweep(*, submit, print_sbatch_only, results_base_prefix, **_):
    _submit_grid(
        efield_files    = [REMOTE_EFIELD_NPZ],
        tracks_seeds    = [42, 43],
        nn_seeds        = [0, 1],
        hidden_choices  = [(32, 32, 32), (64, 64, 64)],
        dropout_rates   = [0.0, 0.2],
        rotor_weights   = [0.0, 0.1, 1.0],
        results_base_prefix = results_base_prefix,
        profile_tag     = 'full_sweep',
        submit          = submit,
        print_sbatch_only = print_sbatch_only,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

PROFILES = {
    'baseline':                  profile_baseline,
    'seed_sweep':                profile_seed_sweep,
    'arch_sweep':                profile_arch_sweep,
    'rotor_sweep':               profile_rotor_sweep,
    'dropout_sweep':             profile_dropout_sweep,
    'full_sweep':                profile_full_sweep,
    'conservative_curl_sweep':   profile_conservative_curl_sweep,
    'tracks100_ebs20_rotor':     profile_tracks100_ebs20_rotor,
    '1k_tracks_gt_precompute':   profile_1k_tracks_gt_precompute,
    '1k_tracks_sweep':           profile_1k_tracks_sweep,
    '100_tracks_sweep':          profile_100_tracks_sweep,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('profile', choices=list(PROFILES),
                        help='Job profile to run.')
    parser.add_argument('--submit', action='store_true',
                        help='Actually submit via sbatch (default: dry-run, print commands).')
    parser.add_argument('--print-commands', action='store_true',
                        help='Print only the sbatch command lines (for paste on login node).')
    parser.add_argument('--results-base', default=f'{REMOTE_RESULTS_DIR}/opt/efield_calib',
                        metavar='DIR',
                        help='Root directory for all result PKLs '
                             f'(default: {REMOTE_RESULTS_DIR}/opt/efield_calib).')
    parser.add_argument('--time', default='03:00:00',
                        help='Slurm wall-clock time per job (default: 03:00:00).')
    parser.add_argument('--mem-gb', type=int, default=64,
                        help='Memory per job in GB (default: 64).')
    parser.add_argument('--no-chain', action='store_true',
                        help='Submit jobs independently instead of Slurm-dependency-chaining '
                             'them (only affects profiles that chain by default, e.g. '
                             '1k_tracks_sweep, conservative_curl_sweep).')
    parser.add_argument('--no-gt-cache', action='store_true',
                        help='Recompute GT signals from scratch instead of loading the '
                             '1k_tracks_gt_precompute h5 cache (only affects 1k_tracks_sweep).')
    parser.add_argument('--n-gt-shards', type=int, default=50,
                        help='Number of shards to split the 1000-track ensemble into for GT '
                             'precompute (only affects 1k_tracks_gt_precompute and '
                             '1k_tracks_sweep\'s cache lookup — must match between the two '
                             'invocations). Default: 50 (20 tracks/shard). Raise this if '
                             'precompute jobs get OOM-killed — per-shard host memory scales '
                             'with tracks-per-shard.')
    parser.add_argument('--only-missing', action='store_true',
                        help='1k_tracks_gt_precompute only: skip shards whose output h5 '
                             'already exists on disk, instead of resubmitting all of them — '
                             'use this to re-run just the shards that failed/timed out.')
    args = parser.parse_args()

    PROFILES[args.profile](
        submit              = args.submit,
        print_sbatch_only   = args.print_commands,
        results_base_prefix = args.results_base,
        time                = args.time,
        mem_gb              = args.mem_gb,
        chain               = not args.no_chain,
        use_gt_cache        = not args.no_gt_cache,
        n_gt_shards         = args.n_gt_shards,
        only_missing        = args.only_missing,
    )


if __name__ == '__main__':
    main()
