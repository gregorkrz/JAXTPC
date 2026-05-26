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
  fine_nosched_bs1_mixed_xyz        Same as fine_nosched_bs1; Xm/Ym/Zm tracks (all direction components nonzero)
  tracks50_mixed_cos30k_nosched     12 mixed-XYZ tracks + 38 random dirs, T~U(50,1000) MeV; single-phase 0.1 mm; warmup 1k; cosine LR / 30k steps
  tracks50_mixed_cos30k_2phase      Same 50 tracks; 1 mm / 5k steps then 0.1 mm; same LR settings as 2_part_schedule_cosine_30k
  tracks24_mixed_cos30k_nosched_tol1e4_p2000  Like tracks12_mixed…tol1e4_p2000 but 24 tracks (12 mixed-XYZ + x-flipped copy each)
  tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1  Like tracks12_mixed…tol1e4_p2000; --lr-multipliers auto (100-step burn-in); global grad clip 1.0
  tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1_p500  Same as …_auto_clip1; --patience-per-param 500 (not 2000)
  tracks12_mixed_cos30k_nosched_auto_clip1_noparamfreeze  Like …_auto_clip1_p500 but disables per-parameter freezing
  tracks12_mixed_cos30k_nosched_auto_clip1_noparamfreeze_no_vel  Same as …_noparamfreeze but ``velocity_cm_us`` held at nominal (not in ``--params``)
  tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1_p500_ebs12  Same as …_p500 with --effective-batch-size 12
  tracks12_mixed_cos30k_auto_clip1_p500_no_vel  Same as …_p500; all EMB params except velocity_cm_us
  fine_nosched_bs1_tol1e4_p300      Same as fine_nosched_bs1; per-param freeze tol=1e-4, window=300
  longitudinal_diffusion_only       Same tracks/seeds as fine_nosched_bs1; optimize diffusion_long_cm2_us only
  longitudinal_transverse_diffusion Same setup; optimize diffusion_long_cm2_us and diffusion_trans_cm2_us together
  timing_study_diag50mev           Seven jobs: single 50 MeV diagonal track, deposit pads 5k–100k, 1000 steps (OOM sweep)
  timing_study_cont                Three jobs: same setup, deposit pads 45k / 50k / 55k only
  15_Tracks_Adam_default_BS1       15-track boundary ensemble (seed=42), no LR multipliers, batch-size 1, single job
  15_Tracks_Adam_default_BS1_AutoMultipliers  Same; with --lr-multipliers auto
  15_Tracks_Adam_default_BS15      Same; accumulates gradients over all 15 tracks (--effective-batch-size 15)
  no_schedule_less_params         No phase schedule; sweep n_params=3..7 always including
                                  diffusion_long_cm2_us, then other physics (transverse diffusion last).
                                  Same tracks/seeds as fine_nosched_bs1 (20 Slurm jobs).
  Adam_NoiseCutoff25_DebugTracks_3k_CollectionOnly  Same as Adam_NoiseCutoff25_DebugTracks_3k but loss over Y1,Y2 (collection) planes only.
  Adam_NoiseCutoff25_DebugTracks_3k_CollectionOnly_3trk  Same but 3 tracks only (Muon4_100MeV removed).
  Adam_NoiseCutoff50_DebugTracks_3k_3trk_BothPlanes      Same 3-track setup; cutoff=50; submits collection-only AND all-planes variants (24 jobs).
  Adam_NoiseCutoff25_DebugTracks_3k_CollectionOnly_3trk_MoreCutoffs  Same; sweeps cutoffs 5,10,15,20,30 (60 jobs).
  Adam_NoiseCutoff25_DebugTracks_2phase  Phase-1: trans+long diffusion 4k steps; phase-2: all params 8k steps;
                                         plus trans-only 8k, long-only 8k, and growing param chain
                                         (trans→+recomb_alpha→+recomb_beta_90 4k each).
                                         ±10% spread, noise, debug 4-track, seeds 1000–1001.
  gradient_cutoff_sweep_15trk  1d-gradient landscape: 15 tracks, cutoffs 0–50, per-plane loss;
                                32 commands (8 noisy seeds) in n_jobs=2 Slurm jobs

Profile runs pass --wandb-tags <profile> to run_optimization.py (optional extras via
--wandb-extra-tags comma,separated).
"""
import argparse
import glob
import math
import os
import pickle
import random
import re
import sys
from pathlib import Path

from job_submission_tools import s3df_submit, s3df_submit_multi

# ---------------------------------------------------------------------------
# Resolve RESULTS_DIR: environment variable takes precedence; fall back to
# parsing the project-root .env file (same file sourced inside sbatch jobs).
# ---------------------------------------------------------------------------
def _read_results_dir_from_env_file() -> str | None:
    env_file = Path(__file__).parents[2] / ".env"
    try:
        text = env_file.read_text()
    except OSError:
        return None
    m = re.search(r'^\s*(?:export\s+)?RESULTS_DIR\s*=\s*(.+)$', text, re.MULTILINE)
    if not m:
        return None
    val = m.group(1).strip().strip('"').strip("'")
    if not os.path.isabs(val):
        val = str(Path(__file__).parents[2] / val)
    return val

_RESULTS_DIR: str = os.environ.get("RESULTS_DIR") or _read_results_dir_from_env_file() or "results"

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

# Same 12-track layout; axis-aligned tracks replaced by Xm/Ym/Zm so (dx,dy,dz) are all nonzero.
TRACKS_12_MIXED_XYZ = (
    "diagonal"
    "+xm1000:1,0.1,0.2:1000+ym1000:0.15,1,0.25:1000+zm1000:0.2,0.05,1:1000"
    "+diagonal_100MeV:1,1,1:100+xm100:1,0.1,0.2:100+ym100:0.15,1,0.25:100+zm100:0.2,0.05,1:100"
    "+diagonal_50MeV:1,1,1:50+xm50:1,0.1,0.2:50+ym50:0.15,1,0.25:50+zm50:0.2,0.05,1:50"
)

# ``TRACKS_12_MIXED_XYZ`` plus a second copy of each track with direction (dx,dy,dz) -> (-dx,dy,dz);
# same kinetic energy per pair. Names suffixed ``_rnx`` for the x-flipped copy.
TRACKS_24_MIXED_XYZ = (
    TRACKS_12_MIXED_XYZ
    + "+diagonal_rnx:-1,1,1:1000"
    + "+xm1000_rnx:-1,0.1,0.2:1000+ym1000_rnx:-0.15,1,0.25:1000+zm1000_rnx:-0.2,0.05,1:1000"
    + "+diagonal_100MeV_rnx:-1,1,1:100+xm100_rnx:-1,0.1,0.2:100+ym100_rnx:-0.15,1,0.25:100+zm100_rnx:-0.2,0.05,1:100"
    + "+diagonal_50MeV_rnx:-1,1,1:50+xm50_rnx:-1,0.1,0.2:50+ym50_rnx:-0.15,1,0.25:50+zm50_rnx:-0.2,0.05,1:50"
)

# Single track (same physics tag as diagonal_50MeV in TRACKS_12)
TRACK_DIAG_50MEV = "diagonal_50MeV:1,1,1:50"

# 15-track boundary ensemble from tools/random_boundary_tracks.generate_random_boundary_tracks
# (n=12 random x-face muons, seed=42, balanced across 100/500/1000 MeV) plus three fixed chords.
# Directions are frozen here; regenerate with: generate_random_boundary_tracks(_VOLUMES, n=12, seed=42)
TRACKS_15_BOUNDARY = (
    "Muon1_1000MeV:-0.747872530,0.661463000,0.056154945:1000"
    "+Muon2_500MeV:-0.641581737,0.275323919,-0.715939672:500"
    "+Muon3_500MeV:-0.483652826,0.868350593,-0.109759697:500"
    "+Muon4_100MeV:-0.694627880,0.476880059,0.538588450:100"
    "+Muon5_100MeV:-0.448568523,-0.712616910,0.539410252:100"
    "+Muon6_1000MeV:-0.624672693,-0.613017831,0.483728401:1000"
    "+Muon7_500MeV:-0.610394124,-0.747896572,0.260901765:500"
    "+Muon8_1000MeV:0.773174642,0.198385012,0.602365637:1000"
    "+Muon9_500MeV:-0.931562076,-0.204366326,-0.300710000:500"
    "+Muon10_100MeV:0.754859526,-0.437194999,0.488924973:100"
    "+Muon11_1000MeV:-0.482051010,0.584842086,0.652369955:1000"
    "+Muon12_100MeV:-0.553810025,-0.123483953,-0.823435589:100"
    "+Muon_diagCross_1000MeV:-0.577350269,-0.577350269,-0.577350269:1000"
    "+Muon_throughEw_skew02_1000MeV:0.934631179,-0.282614666,0.215855296:1000"
    "+Muon_throughWe_skew03_1000MeV:-0.938658230,0.268188066,-0.216785353:1000"
)

# 15-track ensemble for 1d_gradients.py landscape sweeps.
# Each entry: (track_name, tracks_arg_string) where tracks_arg_string is
# passed directly to --tracks.  The 4 "origin" tracks start at (0,0,0);
# the 11 boundary tracks include a start position as the 4th colon-field.
_GRADIENT_15_TRACKS = [
    # 4 origin tracks (match Run_SobolevLossWithCutoff.sh TRACKS array)
    ("diagonal",
     "diagonal:-0.577350,-0.577350,-0.577350:1000"),
    ("Muon1_1000MeV",
     "Muon1_1000MeV:-0.747872530,0.661463000,0.056154945:1000"),
    ("Muon2_500MeV",
     "Muon2_500MeV:-0.641581737,0.275323919,-0.715939672:500"),
    ("Muon4_100MeV",
     "Muon4_100MeV:-0.694627880,0.476880059,0.538588450:100"),
    # 11 boundary tracks with start positions (TRACKS_CUTOFF25)
    ("Muon3_500MeV",
     "Muon3_500MeV:-0.483652826,0.868350593,-0.109759697:500:2160.0,568.790204207,1114.939037169"),
    ("Muon5_100MeV",
     "Muon5_100MeV:-0.448568523,-0.712616910,0.539410252:100:2160.0,1057.372513522,2019.642044116"),
    ("Muon6_1000MeV",
     "Muon6_1000MeV:-0.624672693,-0.613017831,0.483728401:1000:2160.0,-1341.483728756,-1598.739096951"),
    ("Muon7_500MeV",
     "Muon7_500MeV:-0.610394124,-0.747896572,0.260901765:500:2160.0,1437.169806970,865.145240650"),
    ("Muon8_1000MeV",
     "Muon8_1000MeV:0.773174642,0.198385012,0.602365637:1000:-2160.0,-486.093402590,-914.422591021"),
    ("Muon9_500MeV",
     "Muon9_500MeV:-0.931562076,-0.204366326,-0.300710000:500:2160.0,1239.513310809,712.155700478"),
    ("Muon10_100MeV",
     "Muon10_100MeV:0.754859526,-0.437194999,0.488924973:100:-2160.0,296.961966517,-1556.076968089"),
    ("Muon11_1000MeV",
     "Muon11_1000MeV:-0.482051010,0.584842086,0.652369955:1000:2160.0,1144.795064037,581.983142403"),
    ("Muon12_100MeV",
     "Muon12_100MeV:-0.553810025,-0.123483953,-0.823435589:100:2160.0,-2026.866954667,-273.380878516"),
    ("Muon_throughEw_skew02_1000MeV",
     "Muon_throughEw_skew02_1000MeV:0.934631179,-0.282614666,0.215855296:1000:-2100.0,750.0,-550.0"),
    ("Muon_throughWe_skew03_1000MeV",
     "Muon_throughWe_skew03_1000MeV:-0.938658230,0.268188066,-0.216785353:1000:2100.0,-620.0,480.0"),
]


def _make_fraction_tracks(base_tracks, fractions=(('FirstQuarter', 0.0, 0.25), ('LastQuarter', 0.75, 1.0))):
    """Return fraction-slice variants of each track in base_tracks.

    Each base track gets one entry per fraction label. The 4th colon-field
    (start position) is set to '0,0,0' when the base spec has only 3 fields.
    """
    result = []
    for base_name, base_spec in base_tracks:
        parts = base_spec.split(':')
        dir_str   = parts[1]
        mom_str   = parts[2]
        start_str = parts[3] if len(parts) >= 4 else '0,0,0'
        for label, fs, fe in fractions:
            new_name = f'{base_name}_{label}'
            result.append((new_name, f'{new_name}:{dir_str}:{mom_str}:{start_str}:{fs},{fe}'))
    return result


# FirstQuarter (0–25%) and LastQuarter (75–100%) deposit-slice variants of each
# of the 15 boundary tracks.  These produce separate pkl files that the viewer
# merges with the full-track pkls by (param, noise, seed, cutoff) key.
_GRADIENT_FRACTION_TRACKS = _make_fraction_tracks(_GRADIENT_15_TRACKS)


def make_gradient_command(
    param,
    tracks,
    N=20,
    range_frac=0.75,
    noise_scale=0.0,
    noise_seed=42,
    adc_cutoff=0.0,
    adc_cutoffs=None,
    results_dir="$RESULTS_DIR/1d_gradients",
    step_size=1.0,
    max_deposits=5000,
    sobolev_max_pad=128,
    store_per_plane_loss=False,
    store_per_pixel_loss_and_grad=False,
    store_arrays=False,
):
    """Return a 1d_gradients.py command string.

    Pass adc_cutoffs as a list to use --adc-cutoffs (multi-cutoff sweep in one call).
    """
    parts = [
        "python src/analysis/1d_gradients.py",
        f"--param {param}",
        f"--tracks {tracks}",
        f"--N {N}",
        f"--range-frac {range_frac}",
        f"--noise-scale {noise_scale}",
        f"--noise-seed {noise_seed}",
        f"--step-size {step_size}",
        f"--max-deposits {max_deposits}",
        f"--sobolev-max-pad {sobolev_max_pad}",
        f"--results-dir {results_dir}",
    ]
    if adc_cutoffs is not None:
        parts.append(f"--adc-cutoffs {','.join(str(c) for c in adc_cutoffs)}")
    else:
        parts.append(f"--adc-cutoff {adc_cutoff}")
    if store_per_plane_loss:
        parts.append("--store-per-plane-loss")
    if store_per_pixel_loss_and_grad:
        parts.append("--store-per-pixel-loss-and-grad")
    if store_arrays:
        parts.append("--store-arrays")
    return " ".join(parts)


def _chunks(lst, n):
    """Split lst into n roughly equal consecutive chunks."""
    k, r = divmod(len(lst), n)
    i = 0
    for c in range(n):
        size = k + (1 if c < r else 0)
        yield lst[i: i + size]
        i += size


def _tracks_mixed_xyz_plus_random(
    *,
    n_random: int = 38,
    rng_seed: int = 170_452_067,
) -> str:
    """``TRACKS_12_MIXED_XYZ`` plus ``n_random`` extra tracks.

    Directions are uniform on S² (normalized i.i.d. Gaussians). Kinetic energy T is
    ``random.uniform(50, 1000)`` MeV per extra track. Reproducible for fixed
    ``rng_seed``.
    """
    rng = random.Random(rng_seed)
    extras = []
    for i in range(n_random):
        x, y, z = rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)
        nrm = math.sqrt(x * x + y * y + z * z)
        x, y, z = x / nrm, y / nrm, z / nrm
        T = rng.uniform(50.0, 1000.0)
        extras.append(f"rnd{i + 1:03d}:{x:.12g},{y:.12g},{z:.12g}:{T:.12g}")
    return TRACKS_12_MIXED_XYZ + "+" + "+".join(extras)


# 12-track mixed layout + 38 random (50 total). See ``WANDB_PER_TRACK_LOSS_MAX_TRACKS`` in run_optimization:
# per-track W&B loss is disabled when track count reaches 50.
TRACKS_50_MIXED_RANDOM = _tracks_mixed_xyz_plus_random()

PARAM_LIST = [p.strip() for p in ALL_PARAMS.split(",") if p.strip()]
PARAM_LIST_NO_DIFF = [p for p in PARAM_LIST
                      if p not in ("diffusion_trans_cm2_us", "diffusion_long_cm2_us")]
PARAM_LIST_NO_DIFF_LIFETIME = [p for p in PARAM_LIST
                                if p not in ("diffusion_trans_cm2_us", "diffusion_long_cm2_us",
                                             "lifetime_us")]

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


def _seed_is_complete(results_base: str, noise_tag: str, seed: int, verbose: bool = False) -> bool:
    """Return True if a complete result pkl exists for this (results_base, noise_tag, seed).

    Expands $RESULTS_DIR so this works both locally (if set) and on S3DF.
    Scans results_base/noise_tag/*/result_{seed}.pkl — the extra subdirectory
    level is the auto-generated folder_name from run_optimization.py.
    Returns False (i.e. "needs submission") if the directory doesn't exist.
    """
    resolved = results_base.replace("$RESULTS_DIR", _RESULTS_DIR)
    base = os.path.join(resolved, noise_tag)
    pattern = os.path.join(base, "**", f"result_{seed}.pkl")
    if verbose:
        print(f"  checking {pattern}  [RESULTS_DIR={_RESULTS_DIR}]")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        if verbose:
            if not os.path.isdir(base):
                print(f"    → directory does not exist: {base}")
            else:
                print(f"    → no result_{seed}.pkl found under {base}")
        return False
    for pkl_path in matches:
        if verbose:
            print(f"    found: {pkl_path}")
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            if verbose:
                print(f"    → unreadable ({exc})")
            continue
        trials = data.get("trials") or []
        n_expected = data.get("N", -1)
        live = bool(data.get("live_checkpoint"))
        incomplete = _optimization_pickle_incomplete(data)
        if verbose:
            if incomplete:
                print(f"    → incomplete: {len(trials)}/{n_expected} trials, live_checkpoint={live}")
            else:
                print(f"    → complete: {len(trials)}/{n_expected} trials ✓")
        if not incomplete:
            return True
    return False


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
    optimizer="adam",
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
    effective_batch_size=None,
    schedule_steps=None,
    schedule_step_sizes=None,
    schedule_deposits=None,
    schedule_batch_sizes=None,
    gt_step_size=None,
    gt_max_deposits=None,
    gt_param_multiplier=None,
    gt_lifetime_us=None,
    wandb_tags=None,
    tol_per_param=None,
    patience_per_param=None,
    log_interval=None,
    lr_mult_auto_burn_in_steps=None,
    newton_damping=None,
    adam_beta2=None,
    init_from_wandb_run=None,
    init_from_wandb_step=None,
    sobolev_loss_cutoff=None,
    start_position_mm=None,
    planes=None,
):
    """Return a run_optimization.py command string with the given settings."""
    parts = [
        "python src/opt/run_optimization.py",
        f"--params {params}",
        f"--tracks {tracks}",
        f"--loss {loss}",
        f"--optimizer {optimizer}",
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
    if lr_mult_auto_burn_in_steps is not None:
        parts.append(f"--lr-mult-auto-burn-in-steps {int(lr_mult_auto_burn_in_steps)}")
    if batch_size is not None:
        parts.append(f"--batch-size {batch_size}")
    if effective_batch_size is not None:
        parts.append(f"--effective-batch-size {int(effective_batch_size)}")
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
    if gt_param_multiplier is not None:
        parts.append(f"--gt-param-multiplier {gt_param_multiplier}")
    if gt_lifetime_us is not None:
        parts.append(f"--gt-lifetime-us {gt_lifetime_us}")
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
    if newton_damping is not None:
        parts.append(f"--newton-damping {newton_damping}")
    if adam_beta2 is not None:
        parts.append(f"--adam-beta2 {adam_beta2}")
    if init_from_wandb_run is not None:
        parts.append(f"--init-from-wandb-run {init_from_wandb_run}")
    if init_from_wandb_step is not None:
        parts.append(f"--init-from-wandb-step {init_from_wandb_step}")
    if sobolev_loss_cutoff is not None and sobolev_loss_cutoff > 0.0:
        parts.append(f"--sobolev-loss-cutoff {sobolev_loss_cutoff}")
    if start_position_mm is not None:
        parts.append(f"--start-position-mm {start_position_mm}")
    if planes is not None:
        parts.append(f"--planes {planes}")
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


def profile_fine_nosched_bs1_mixed_xyz(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """No phase schedule; 0.1 mm, 50k deposits, bs=1 — same hyperparams as fine_nosched_bs1.

    Track list matches the 12-track layout but replaces axis-aligned X/Y/Z with mixed
    directions Xm (1,0.1,0.2), Ym (0.15,1,0.25), Zm (0.2,0.05,1) at 1000 / 100 / 50 MeV.
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
        results_base="$RESULTS_DIR/opt/sched2_longer_schedule_20260430/no_sched_fine_mixed_xyz",
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
            tracks=TRACKS_12_MIXED_XYZ,
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


def profile_tracks50_mixed_cos30k_nosched(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """50 tracks (12 mixed XYZ + 38 random directions, T ~ U(50,1000) MeV).

    Single forward phase: 0.1 mm, 50k deposits, batch 1. Warmup 1k steps; cosine LR decay
    over 30k optimizer steps (no coarse/fine step-size schedule).
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
        results_base="$RESULTS_DIR/opt/tracks50_mixed_cos30k/nosched",
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
            tracks=TRACKS_50_MIXED_RANDOM,
            seed=seed,
            noise_scale=0.0,
            wandb_tags=wandb_tags,
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        s3df_submit(
            command,
            time="06:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
        )


def profile_tracks50_mixed_cos30k_2phase(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """Same 50 tracks as ``profile_tracks50_mixed_cos30k_nosched``.

    Two-phase forward schedule like ``profile_2_part_schedule_cosine_30k``: 1.0 mm for
    steps 0–5000 (5k-deposit pad, batch 5), then 0.1 mm to step 30k (50k pads, batch 1).
    Warmup 1k steps; cosine LR over 30k steps.
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
        results_base="$RESULTS_DIR/opt/tracks50_mixed_cos30k/2phase",
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
            tracks=TRACKS_50_MIXED_RANDOM,
            seed=seed,
            noise_scale=0.0,
            wandb_tags=wandb_tags,
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        s3df_submit(
            command,
            time="06:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
        )


def profile_tracks12_mixed_cos30k_nosched(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """12-track core of ``TRACKS_50_MIXED_RANDOM`` (``TRACKS_12_MIXED_XYZ``).

    Same optimizer/forward setup as ``profile_tracks50_mixed_cos30k_nosched`` but on the
    12 mixed-XYZ tracks only: single forward phase 0.1 mm, 50k deposits, batch 1; warmup
    1k steps; cosine LR decay over 30k optimizer steps (no coarse/fine schedule).
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
        results_base="$RESULTS_DIR/opt/tracks12_mixed_cos30k/nosched",
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
            tracks=TRACKS_12_MIXED_XYZ,
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


def profile_tracks12_mixed_cos30k_2phase(*, submit=True, print_sbatch_only=False, wandb_tags=None):
    """Same 12 tracks as ``profile_tracks12_mixed_cos30k_nosched``.

    Two-phase forward schedule like ``profile_tracks50_mixed_cos30k_2phase``: 1.0 mm for
    steps 0–5000 (5k-deposit pad, batch 5), then 0.1 mm to step 30k (50k pads, batch 1).
    Warmup 1k steps; cosine LR over 30k steps.
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
        results_base="$RESULTS_DIR/opt/tracks12_mixed_cos30k/2phase",
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
            tracks=TRACKS_12_MIXED_XYZ,
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


def profile_tracks12_mixed_cos30k_nosched_tol1e4_p2000(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """tracks12_mixed_cos30k_nosched with per-parameter freezing enabled.

    Uses ``--tol-per-param 1e-4`` and ``--patience-per-param 2000`` while keeping the same
    optimizer and forward settings as ``profile_tracks12_mixed_cos30k_nosched``.
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        tol_per_param=1e-4,
        patience_per_param=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/tracks12_mixed_cos30k/nosched_tol1e4_p2000_per_param",
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
            tracks=TRACKS_12_MIXED_XYZ,
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


def profile_tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Same as ``profile_tracks12_mixed_cos30k_nosched_tol1e4_p2000`` except per-param LR
    scales come from ``--lr-multipliers auto`` (default 100 burn-in steps in
    ``run_optimization``) and gradients use global norm clip ``--clip-grad-norm 1.0``.
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        tol_per_param=1e-4,
        patience_per_param=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/tracks12_mixed_cos30k/nosched_tol1e4_p2000_auto_clip1",
        grad_clip=1.0,
        lr_multipliers="auto",
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
            tracks=TRACKS_12_MIXED_XYZ,
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


def profile_tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1_p500(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Same as ``profile_tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1`` but
    ``--patience-per-param 500`` instead of 2000 (still ``--tol-per-param 1e-4``).
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        tol_per_param=1e-4,
        patience_per_param=500,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/tracks12_mixed_cos30k/nosched_tol1e4_p2000_auto_clip1_p500",
        grad_clip=1.0,
        lr_multipliers="auto",
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
            tracks=TRACKS_12_MIXED_XYZ,
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


def profile_tracks12_mixed_cos30k_auto_clip1_p500_no_vel(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Same as ``profile_tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1_p500`` but
    ``--params`` lists every EMB parameter except ``velocity_cm_us`` (held at nominal).
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        tol_per_param=1e-4,
        patience_per_param=500,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base=(
            "$RESULTS_DIR/opt/tracks12_mixed_cos30k/"
            "nosched_tol1e4_p2000_auto_clip1_p500_no_vel"
        ),
        grad_clip=1.0,
        lr_multipliers="auto",
        warmup_steps=1000,
        num_buckets=1000,
        step_size=0.1,
        max_num_deposits=50000,
        batch_size=1,
    )
    params = ",".join(p for p in PARAM_LIST if p != "velocity_cm_us")
    for seed in [44, 45, 46, 47]:
        command = make_opt_command(
            params=params,
            tracks=TRACKS_12_MIXED_XYZ,
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


def profile_tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1_p500_ebs12(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Same as ``profile_tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1_p500``
    with ``--effective-batch-size 12`` and global grad clip 10.0.
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        tol_per_param=1e-4,
        patience_per_param=500,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base=(
            "$RESULTS_DIR/opt/tracks12_mixed_cos30k/"
            "nosched_tol1e4_p2000_auto_clip1_p500_ebs12"
        ),
        grad_clip=10.0,
        lr_multipliers="auto",
        warmup_steps=1000,
        num_buckets=1000,
        step_size=0.1,
        max_num_deposits=50000,
        batch_size=1,
        effective_batch_size=12,
    )
    params = ",".join(PARAM_LIST)
    for seed in [44, 45, 46, 47]:
        command = make_opt_command(
            params=params,
            tracks=TRACKS_12_MIXED_XYZ,
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


def profile_tracks12_mixed_cos30k_nosched_auto_clip1_noparamfreeze(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Same core setup as ``..._auto_clip1_p500`` but without per-parameter freezing.

    I.e. no ``--tol-per-param`` / ``--patience-per-param``.
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base=(
            "$RESULTS_DIR/opt/tracks12_mixed_cos30k/"
            "nosched_auto_clip1_noparamfreeze"
        ),
        grad_clip=1.0,
        lr_multipliers="auto",
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
            tracks=TRACKS_12_MIXED_XYZ,
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


def profile_tracks12_mixed_cos30k_nosched_auto_clip1_noparamfreeze_no_vel(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Same as ``profile_tracks12_mixed_cos30k_nosched_auto_clip1_noparamfreeze`` but
    ``--params`` omits ``velocity_cm_us`` (held at nominal).
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base=(
            "$RESULTS_DIR/opt/tracks12_mixed_cos30k/"
            "nosched_auto_clip1_noparamfreeze_no_vel"
        ),
        grad_clip=1.0,
        lr_multipliers="auto",
        warmup_steps=1000,
        num_buckets=1000,
        step_size=0.1,
        max_num_deposits=50000,
        batch_size=1,
    )
    params = ",".join(p for p in PARAM_LIST if p != "velocity_cm_us")
    for seed in [44, 45, 46, 47]:
        command = make_opt_command(
            params=params,
            tracks=TRACKS_12_MIXED_XYZ,
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


def profile_tracks24_mixed_cos30k_nosched_tol1e4_p2000(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Same as ``profile_tracks12_mixed_cos30k_nosched_tol1e4_p2000`` but 24 mixed tracks.

    Uses ``TRACKS_24_MIXED_XYZ``: the 12 mixed-XYZ layout plus an x-reflected copy
    ``(-dx, dy, dz)`` at the same momentum for each explicit track (``*_rnx`` names).
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        tol_per_param=1e-4,
        patience_per_param=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/tracks24_mixed_cos30k/nosched_tol1e4_p2000_per_param",
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
            tracks=TRACKS_24_MIXED_XYZ,
            seed=seed,
            noise_scale=0.0,
            wandb_tags=wandb_tags,
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        s3df_submit(
            command,
            time="06:00:00",
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


def profile_15_Tracks_Adam_default_BS1(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """15-track boundary ensemble (seed=42), Adam, no LR multipliers, batch-size 1, seed 42.

    Same core setup as ``profile_tracks12_mixed_cos30k_nosched_auto_clip1_noparamfreeze``
    (cosine LR, 30k steps, warmup 1k, clip 1.0, patience 2000, no per-param freeze) but:
      - tracks: TRACKS_15_BOUNDARY (the same 15 tracks used in 2D landscape plots)
      - no ``--lr-multipliers`` (uniform per-param LR)
      - single job at seed=42
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/tracks15_boundary_cos30k/nosched_bs1",
        grad_clip=1.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=0.1,
        max_num_deposits=50000,
        batch_size=1,
    )
    params = ",".join(PARAM_LIST)
    command = make_opt_command(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        seed=42,
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


def profile_15_Tracks_Adam_default_BS1_AutoMultipliers(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Same as ``profile_15_Tracks_Adam_default_BS1`` but with ``--lr-multipliers auto``."""
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/tracks15_boundary_cos30k/nosched_automult_bs1",
        grad_clip=1.0,
        lr_multipliers="auto",
        warmup_steps=1000,
        num_buckets=1000,
        step_size=0.1,
        max_num_deposits=50000,
        batch_size=1,
    )
    params = ",".join(PARAM_LIST)
    command = make_opt_command(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        seed=42,
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


def profile_15_Tracks_Adam_default_BS15(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Same as ``profile_15_Tracks_Adam_default_BS1`` but accumulates gradients over all 15 tracks.

    Uses ``--effective-batch-size 15`` so each optimizer step sees the gradient summed
    across all 15 tracks before applying Adam; ``--batch-size 1`` keeps memory usage low.
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/tracks15_boundary_cos30k/nosched_ebs15",
        grad_clip=1.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=0.1,
        max_num_deposits=50000,
        batch_size=1,
        effective_batch_size=15,
    )
    params = ",".join(PARAM_LIST)
    command = make_opt_command(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        seed=42,
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


def profile_15_Tracks_Adam_default_BS15_coarse1mm(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Same as ``profile_15_Tracks_Adam_default_BS15`` but coarse forward pass: 1 mm step, 5k deposits.

    Uses ``--batch-size 5 --effective-batch-size 3``: 3 vmap batches of (5,5,5) tracks cover
    all 15 per optimizer step. batch_size=15 and 7 OOM during XLA compilation.
    GT also uses 1 mm / 5k deposits to match the forward pass.
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=0.0001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/tracks15_boundary_cos30k/nosched_coarse1mm_vmap5",
        grad_clip=1.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=5,
        effective_batch_size=3,
        gt_step_size=1.0,
        gt_max_deposits=5000,
    )
    params = ",".join(PARAM_LIST)
    command = make_opt_command(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        seed=42,
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


def profile_15_Tracks_Adam_default_BS15_coarse1mm_LR1e3(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Same as ``profile_15_Tracks_Adam_default_BS15_coarse1mm`` but lr=1e-3."""
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=1e-3,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/tracks15_boundary_cos30k/nosched_coarse1mm_vmap5_lr1e3",
        grad_clip=1.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=5,
        effective_batch_size=3,
        gt_step_size=1.0,
        gt_max_deposits=5000,
    )
    params = ",".join(PARAM_LIST)
    command = make_opt_command(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        seed=42,
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


def profile_15_Tracks_Adam_default_BS15_coarse1mm_LR1e3_NoVelocity(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Same as ``profile_15_Tracks_Adam_default_BS15_coarse1mm_LR1e3`` but omits ``velocity_cm_us`` from params."""
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=1e-3,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/tracks15_boundary_cos30k/nosched_coarse1mm_vmap5_lr1e3_no_vel",
        grad_clip=1.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=5,
        effective_batch_size=3,
        gt_step_size=1.0,
        gt_max_deposits=5000,
    )
    params = ",".join(p for p in PARAM_LIST if p != "velocity_cm_us")
    command = make_opt_command(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        seed=42,
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


def profile_15_Tracks_Newton_recombAlpha_velocity_BS1(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Newton optimizer on recomb_alpha + velocity_cm_us only, 15-track boundary ensemble.

    LR=1.0 (pure Newton step scale), damping=1e-3 (H + 1e-3*I), coarse forward pass
    (1 mm / 5k deposits), batch_size=1 / effective_batch_size=15 to accumulate
    gradient + Hessian over all 15 tracks per optimizer step.
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=1.0,
        lr_schedule="cosine",  # no-op for Newton
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/tracks15_boundary_cos30k/newton_recombAlpha_velocity_coarse1mm_ebs15",
        grad_clip=1.0,
        warmup_steps=1000,  # no-op for Newton
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=1,
        effective_batch_size=15,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        newton_damping=1e-3,
    )
    command = make_opt_command(
        params="recomb_alpha,velocity_cm_us",
        tracks=TRACKS_15_BOUNDARY,
        optimizer="newton",
        seed=42,
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


def profile_15_Tracks_Newton_diffRecomb_BS1(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Newton optimizer on diffusion + recomb block (5 params), 15-track boundary ensemble.

    Params: diffusion_trans, diffusion_long, recomb_alpha, recomb_beta_90, recomb_R.
    Fine forward pass (0.1 mm / 50k deposits), lr=1.0, damping=1e-3,
    batch_size=1 / effective_batch_size=15 (all 15 tracks per optimizer step).
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=1.0,
        lr_schedule="cosine",  # no-op for Newton
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/tracks15_boundary_cos30k/newton_diffRecomb_ebs15",
        grad_clip=1.0,
        warmup_steps=1000,  # no-op for Newton
        num_buckets=1000,
        step_size=0.1,
        max_num_deposits=50000,
        batch_size=1,
        effective_batch_size=15,
        newton_damping=1e-3,
    )
    params = "diffusion_trans_cm2_us,diffusion_long_cm2_us,recomb_alpha,recomb_beta_90,recomb_R"
    command = make_opt_command(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="newton",
        seed=42,
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


def profile_15_Tracks_Newton_allParams_BS1(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Newton optimizer on all 7 params, 15-track boundary ensemble.

    Coarse forward pass (1 mm / 5k deposits), lr=1.0, damping=1e-3,
    batch_size=1 / effective_batch_size=15 (all 15 tracks per optimizer step).
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=1.0,
        lr_schedule="cosine",  # no-op for Newton
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/tracks15_boundary_cos30k/newton_allParams_coarse1mm_ebs15",
        grad_clip=1.0,
        warmup_steps=1000,  # no-op for Newton
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=1,
        effective_batch_size=15,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        newton_damping=1e-3,
        log_interval=5,
    )
    params = ",".join(PARAM_LIST)
    command = make_opt_command(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="newton",
        seed=42,
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


def profile_15_Tracks_Newton_allParams_BS1_DampClipSweep(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Sweep over 8 combinations of newton_damping × clip_grad_norm, all with seed=42.

    damping in [0.001, 1.0] × clip in [1.0, 0.1, 0.01, 0.001] = 8 jobs.
    Each combination gets a unique results_base so output paths don't collide.
    All other settings identical to profile_15_Tracks_Newton_allParams_BS1.
    """
    params = ",".join(PARAM_LIST)
    lr = 1e-4
    for damping in [0.01]:
        for clip in [-1]:
            damping_str = f"{damping}".replace(".", "p")
            clip_str = f"{clip}".replace(".", "p")
            lr_str = f"{lr}".replace(".", "p")
            results_base = (
                f"$RESULTS_DIR/opt/tracks15_boundary_cos30k"
                f"/newton_allParams_sweep/d{damping_str}_c{clip_str}_lr{lr_str}"
            )
            command = make_opt_command(
                params=params,
                tracks=TRACKS_15_BOUNDARY,
                optimizer="newton",
                seed=42,
                noise_scale=0.0,
                loss="sobolev_loss_geomean_log1p",
                lr=lr,
                lr_schedule="cosine",
                max_steps=30000,
                tol=1e-6,
                patience=2000,
                N=1,
                range_lo=0.9,
                range_hi=1.1,
                results_base=results_base,
                grad_clip=clip,
                warmup_steps=1000,
                num_buckets=1000,
                step_size=1.0,
                max_num_deposits=5000,
                batch_size=1,
                effective_batch_size=15,
                gt_step_size=1.0,
                gt_max_deposits=5000,
                newton_damping=damping,
                log_interval=5,
                wandb_tags=(wandb_tags or []) + ["15_Tracks_Newton_allParams_BS1_DampClipSweep"],
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


def profile_15_Tracks_Newton_allParams_BS1_Damping1p0(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Same as ``profile_15_Tracks_Newton_allParams_BS1`` but with newton_damping=1.0.

    Higher damping regularises the Hessian inverse in flat directions (lifetime_us,
    diffusion params, recomb_alpha) and prevents the initial Newton step from blowing
    those parameters to unphysical values.
    """
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=1.0,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/tracks15_boundary_cos30k/newton_allParams_coarse1mm_ebs15_damping1p0",
        grad_clip=1.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=1,
        effective_batch_size=15,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        newton_damping=1.0,
        log_interval=5,
    )
    params = ",".join(PARAM_LIST)
    command = make_opt_command(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="newton",
        seed=42,
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


def profile_15_Tracks_Adam_default_BS15_LR4e4(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Same as ``profile_15_Tracks_Adam_default_BS15`` but lr=4e-4 (≈ sqrt(15) × 1e-4 scaling rule)."""
    shared = dict(
        loss="sobolev_loss_geomean_log1p",
        lr=4e-4,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base="$RESULTS_DIR/opt/tracks15_boundary_cos30k/nosched_ebs15_lr4e4",
        grad_clip=1.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=0.1,
        max_num_deposits=50000,
        batch_size=1,
        effective_batch_size=15,
    )
    params = ",".join(PARAM_LIST)
    command = make_opt_command(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        seed=42,
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


def profile_15Trk_Adam_Noise_20260511(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Sweep over 5 Adam beta2 values with clip disabled, seed=42.

    beta2 in [0.9, 0.99, 0.999, 0.9999, 0.99999].  All other settings match
    run 4t0kxso1 (lr=0.001, cosine, coarse 1mm/5k, eff_bs=5), except
    clip_grad_norm=0 (disabled).  Motivation: with clip=1.0 the velocity_cm_us
    gradient (34k) crushes diffusion_trans into Adam's eps regime; removing clip
    lets each param's second moment adapt independently, and varying beta2
    controls how quickly that adaptation tracks the current gradient scale.
    """
    params = ",".join(PARAM_LIST)
    beta2 = 0.9
    results_base = (
        f"$RESULTS_DIR/opt/Adam_Noise_20260511"
    )
    command = make_opt_command(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="adam",
        seed=42,
        noise_scale=1.0,
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base=results_base,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=5,
        effective_batch_size=3,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        adam_beta2=beta2,
        log_interval=50,
        wandb_tags=(wandb_tags or []) + ["Adam_Noise_20260511"],
    )
    if not print_sbatch_only:
        print(command)
    s3df_submit(
        command,
        time="05:00:00",
        submit=submit,
        mem_gb=64,
        print_sbatch_command=print_sbatch_only,
    )


def profile_15_Tracks_Adam_Beta2Sweep_NoClip(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Sweep over 5 Adam beta2 values with clip disabled, seed=42.

    beta2 in [0.9, 0.99, 0.999, 0.9999, 0.99999].  All other settings match
    run 4t0kxso1 (lr=0.001, cosine, coarse 1mm/5k, eff_bs=5), except
    clip_grad_norm=0 (disabled).  Motivation: with clip=1.0 the velocity_cm_us
    gradient (34k) crushes diffusion_trans into Adam's eps regime; removing clip
    lets each param's second moment adapt independently, and varying beta2
    controls how quickly that adaptation tracks the current gradient scale.
    """
    params = ",".join(PARAM_LIST)
    for beta2 in [0.9, 0.99, 0.999, 0.9999, 0.99999]:
        beta2_str = f"{beta2}".replace(".", "p")
        results_base = (
            f"$RESULTS_DIR/opt/tracks15_boundary_cos30k"
            f"/adam_beta2sweep_noclip/b2_{beta2_str}"
        )
        command = make_opt_command(
            params=params,
            tracks=TRACKS_15_BOUNDARY,
            optimizer="adam",
            seed=42,
            noise_scale=0.0,
            loss="sobolev_loss_geomean_log1p",
            lr=0.001,
            lr_schedule="cosine",
            max_steps=30000,
            tol=1e-6,
            patience=2000,
            N=1,
            range_lo=0.9,
            range_hi=1.1,
            results_base=results_base,
            grad_clip=0.0,
            warmup_steps=1000,
            num_buckets=1000,
            step_size=1.0,
            max_num_deposits=5000,
            batch_size=3,
            effective_batch_size=5,
            gt_step_size=1.0,
            gt_max_deposits=5000,
            adam_beta2=beta2,
            log_interval=50,
            wandb_tags=(wandb_tags or []) + ["15_Tracks_Adam_Beta2Sweep_NoClip"],
        )
        if not print_sbatch_only:
            print(command)
        s3df_submit(
            command,
            time="08:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
        )

def profile_15_Tracks_Newton_ContinueFrom_eizhqsj0_step1k(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Newton continuation from Adam run eizhqsj0.

    Picks up where the Adam run left off: initialises trial 0 from the last
    logged params/<name>_physical values of run eizhqsj0, then runs Newton
    with damping=0.01, lr=1.0, no gradient clipping.
    All other settings match profile_15_Tracks_Newton_allParams_BS1.
    """
    params = ",".join(PARAM_LIST)
    results_base = (
        "$RESULTS_DIR/opt/tracks15_boundary_cos30k"
        "/newton_allParams_continue_eizhqsj0"
    )
    command = make_opt_command(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="newton",
        seed=42,
        noise_scale=0.0,
        loss="sobolev_loss_geomean_log1p",
        lr=1.0,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base=results_base,
        grad_clip=-1,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=1,
        effective_batch_size=15,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        newton_damping=0.01,
        log_interval=5,
        init_from_wandb_run="eizhqsj0",
        init_from_wandb_step=1000,
        wandb_tags=(wandb_tags or []) + ["15_Tracks_Newton_ContinueFrom_eizhqsj0_Step1k"],
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
def profile_15_Tracks_Newton_ContinueFrom_eizhqsj0(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Newton continuation from Adam run eizhqsj0.

    Picks up where the Adam run left off: initialises trial 0 from the last
    logged params/<name>_physical values of run eizhqsj0, then runs Newton
    with damping=0.01, lr=1.0, no gradient clipping.
    All other settings match profile_15_Tracks_Newton_allParams_BS1.
    """
    params = ",".join(PARAM_LIST)
    results_base = (
        "$RESULTS_DIR/opt/tracks15_boundary_cos30k"
        "/newton_allParams_continue_eizhqsj0"
    )
    command = make_opt_command(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="newton",
        seed=42,
        noise_scale=0.0,
        loss="sobolev_loss_geomean_log1p",
        lr=1.0,
        lr_schedule="cosine",
        max_steps=30000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        results_base=results_base,
        grad_clip=-1,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=1,
        effective_batch_size=15,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        newton_damping=0.01,
        log_interval=5,
        init_from_wandb_run="eizhqsj0",
        wandb_tags=(wandb_tags or []) + ["15_Tracks_Newton_ContinueFrom_eizhqsj0"],
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


def profile_15Trk_Adam_NoiseSeedSweep_3k(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Seed sweep (43–47) × noise on/off, 3k steps, identical to Adam_Noise_20260511 otherwise.

    10 jobs total: 5 seeds × 2 noise conditions (noise_scale=1.0 and noise_scale=0.0).
    All other settings match profile_15Trk_Adam_Noise_20260511:
    lr=0.001, cosine, beta2=0.9, no clip, coarse 1mm/5k, bs=5, eff_bs=3.
    """
    params = ",".join(PARAM_LIST)
    shared = dict(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=5,
        effective_batch_size=3,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        adam_beta2=0.9,
        log_interval=50,
    )
    for noise_scale in [1.0, 0.0]:
        noise_tag = "noise" if noise_scale > 0.0 else "nonoise"
        prev_job = None
        for seed in [43, 44, 45, 46, 47]:
            results_base = (
                f"$RESULTS_DIR/opt/Adam_NoiseSeedSweep_3k/{noise_tag}"
            )
            command = make_opt_command(
                seed=seed,
                noise_scale=noise_scale,
                results_base=results_base,
                wandb_tags=(wandb_tags or []) + ["Adam_NoiseSeedSweep_3k", noise_tag],
                **shared,
            )
            if not print_sbatch_only:
                print(command)
            prev_job = s3df_submit(
                command,
                time="01:05:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                dependency=prev_job,
            )


def profile_15Trk_Adam_NoiseSeedSweep_3k_GT2(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    skip_complete=False,
    verbose=False,
):
    """Like Adam_NoiseSeedSweep_3k but GT parameters shifted 20% up (multiplier=1.2).

    10 jobs total: 5 seeds × 2 noise conditions. No dependency chain; all jobs run in parallel.
    """
    params = ",".join(PARAM_LIST)
    shared = dict(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=5,
        effective_batch_size=3,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        gt_param_multiplier=1.2,
        adam_beta2=0.9,
        log_interval=50,
    )
    results_base = "$RESULTS_DIR/opt/Adam_NoiseSeedSweep_3k_GT2"
    for noise_scale in [1.0, 0.0]:
        noise_tag = "noise" if noise_scale > 0.0 else "nonoise"
        for seed in [43, 44, 45, 46, 47]:
            if verbose or skip_complete:
                is_done = _seed_is_complete(results_base, noise_tag, seed, verbose=verbose)
                if is_done and skip_complete:
                    print(f"  SKIP complete: {noise_tag} seed={seed}")
                    continue
                elif not is_done and verbose:
                    print(f"  → will submit: {noise_tag} seed={seed}")

            command = make_opt_command(
                seed=seed,
                noise_scale=noise_scale,
                results_base=f"{results_base}/{noise_tag}",
                wandb_tags=(wandb_tags or []) + ["Adam_NoiseSeedSweep_3k_GT2", noise_tag],
                **shared,
            )
            if not print_sbatch_only:
                print(command)
            s3df_submit(
                command,
                time="01:05:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
            )


def profile_15Trk_Adam_NoiseSeedSweep_3k_GT3(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Like Adam_NoiseSeedSweep_3k (GT1) but GT electron lifetime = 6 ms (6000 μs).

    10 jobs total: 5 seeds × 2 noise conditions. Two dependency chains (one per
    noise condition), each serialising the 5 seeds so only one job runs at a time.
    """
    params = ",".join(PARAM_LIST)
    shared = dict(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=5,
        effective_batch_size=3,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        gt_lifetime_us=6000.0,
        adam_beta2=0.9,
        log_interval=50,
    )
    for noise_scale in [1.0, 0.0]:
        noise_tag = "noise" if noise_scale > 0.0 else "nonoise"
        prev_job = None
        for seed in [43, 44, 45, 46, 47]:
            results_base = (
                f"$RESULTS_DIR/opt/Adam_NoiseSeedSweep_3k_GT3/{noise_tag}"
            )
            command = make_opt_command(
                seed=seed,
                noise_scale=noise_scale,
                results_base=results_base,
                wandb_tags=(wandb_tags or []) + ["Adam_NoiseSeedSweep_3k_GT3", noise_tag],
                **shared,
            )
            if not print_sbatch_only:
                print(command)
            prev_job = s3df_submit(
                command,
                time="01:05:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                dependency=prev_job,
            )


def profile_15Trk_Adam_NoiseSeedSweep_3k_NoDiff(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Like Adam_NoiseSeedSweep_3k (GT1) but optimizes all params except diffusion.

    5 jobs total: 5 seeds × noise only. One dependency chain serialising the seeds.
    Parameters: velocity_cm_us, lifetime_us, recomb_alpha, recomb_beta_90, recomb_R.
    """
    params = ",".join(PARAM_LIST_NO_DIFF)
    shared = dict(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=5,
        effective_batch_size=3,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        adam_beta2=0.9,
        log_interval=50,
    )
    noise_tag = "noise"
    prev_job = None
    for seed in [43, 44, 45, 46, 47]:
        command = make_opt_command(
            seed=seed,
            noise_scale=1.0,
            results_base=f"$RESULTS_DIR/opt/Adam_NoiseSeedSweep_3k_NoDiff/{noise_tag}",
            wandb_tags=(wandb_tags or []) + ["Adam_NoiseSeedSweep_3k_NoDiff", noise_tag],
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        prev_job = s3df_submit(
            command,
            time="01:05:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            dependency=prev_job,
        )


def profile_15Trk_Adam_NoiseSeedSweep_3k_GT2_NoDiff(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Like Adam_NoiseSeedSweep_3k_GT2 but optimizes all params except diffusion.

    5 jobs total: 5 seeds × noise only. One dependency chain serialising the seeds.
    Parameters: velocity_cm_us, lifetime_us, recomb_alpha, recomb_beta_90, recomb_R.
    GT parameters shifted 20% up (gt_param_multiplier=1.2).
    """
    params = ",".join(PARAM_LIST_NO_DIFF)
    shared = dict(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=5,
        effective_batch_size=3,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        gt_param_multiplier=1.2,
        adam_beta2=0.9,
        log_interval=50,
    )
    noise_tag = "noise"
    prev_job = None
    for seed in [43, 44, 45, 46, 47]:
        command = make_opt_command(
            seed=seed,
            noise_scale=1.0,
            results_base=f"$RESULTS_DIR/opt/Adam_NoiseSeedSweep_3k_GT2_NoDiff/{noise_tag}",
            wandb_tags=(wandb_tags or []) + ["Adam_NoiseSeedSweep_3k_GT2_NoDiff", noise_tag],
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        prev_job = s3df_submit(
            command,
            time="01:05:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            dependency=prev_job,
        )


def profile_15Trk_Adam_NoiseSeedSweep_3k_NoDiffLifetime(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Like Adam_NoiseSeedSweep_3k_NoDiff (GT1) but also excludes lifetime_us.

    5 jobs total: 5 seeds × noise only. One dependency chain serialising the seeds.
    Parameters: velocity_cm_us, recomb_alpha, recomb_beta_90, recomb_R.
    """
    params = ",".join(PARAM_LIST_NO_DIFF_LIFETIME)
    shared = dict(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=5,
        effective_batch_size=3,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        adam_beta2=0.9,
        log_interval=50,
    )
    noise_tag = "noise"
    prev_job = None
    for seed in [43, 44, 45, 46, 47]:
        command = make_opt_command(
            seed=seed,
            noise_scale=1.0,
            results_base=f"$RESULTS_DIR/opt/Adam_NoiseSeedSweep_3k_NoDiffLifetime/{noise_tag}",
            wandb_tags=(wandb_tags or []) + ["Adam_NoiseSeedSweep_3k_NoDiffLifetime", noise_tag],
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        prev_job = s3df_submit(
            command,
            time="01:05:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            dependency=prev_job,
        )


def profile_15Trk_Adam_NoiseSeedSweep_3k_GT2_NoDiffLifetime(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Like Adam_NoiseSeedSweep_3k_GT2_NoDiff but also excludes lifetime_us.

    5 jobs total: 5 seeds × noise only. One dependency chain serialising the seeds.
    Parameters: velocity_cm_us, recomb_alpha, recomb_beta_90, recomb_R.
    GT parameters shifted 20% up (gt_param_multiplier=1.2).
    """
    params = ",".join(PARAM_LIST_NO_DIFF_LIFETIME)
    shared = dict(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=5,
        effective_batch_size=3,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        gt_param_multiplier=1.2,
        adam_beta2=0.9,
        log_interval=50,
    )
    noise_tag = "noise"
    prev_job = None
    for seed in [43, 44, 45, 46, 47]:
        command = make_opt_command(
            seed=seed,
            noise_scale=1.0,
            results_base=f"$RESULTS_DIR/opt/Adam_NoiseSeedSweep_3k_GT2_NoDiffLifetime/{noise_tag}",
            wandb_tags=(wandb_tags or []) + ["Adam_NoiseSeedSweep_3k_GT2_NoDiffLifetime", noise_tag],
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        prev_job = s3df_submit(
            command,
            time="01:05:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            dependency=prev_job,
        )


def profile_15Trk_Adam_NoiseSeedSweep_3k_GT3_NoDiff(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Like Adam_NoiseSeedSweep_3k_NoDiff but GT electron lifetime = 6 ms (6000 μs).

    5 jobs total: 5 seeds × noise only. One dependency chain serialising the seeds.
    Parameters: velocity_cm_us, lifetime_us, recomb_alpha, recomb_beta_90, recomb_R.
    """
    params = ",".join(PARAM_LIST_NO_DIFF)
    shared = dict(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=5,
        effective_batch_size=3,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        gt_lifetime_us=6000.0,
        adam_beta2=0.9,
        log_interval=50,
    )
    noise_tag = "noise"
    prev_job = None
    for seed in [43, 44, 45, 46, 47]:
        command = make_opt_command(
            seed=seed,
            noise_scale=1.0,
            results_base=f"$RESULTS_DIR/opt/Adam_NoiseSeedSweep_3k_GT3_NoDiff/{noise_tag}",
            wandb_tags=(wandb_tags or []) + ["Adam_NoiseSeedSweep_3k_GT3_NoDiff", noise_tag],
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        prev_job = s3df_submit(
            command,
            time="01:05:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            dependency=prev_job,
        )


def profile_15Trk_Adam_NoiseSeedSweep_3k_GT3_NoDiffLifetime(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
):
    """Like Adam_NoiseSeedSweep_3k_NoDiffLifetime but GT electron lifetime = 6 ms (6000 μs).

    5 jobs total: 5 seeds × noise only. One dependency chain serialising the seeds.
    Parameters: velocity_cm_us, recomb_alpha, recomb_beta_90, recomb_R.
    """
    params = ",".join(PARAM_LIST_NO_DIFF_LIFETIME)
    shared = dict(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=5,
        effective_batch_size=3,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        gt_lifetime_us=6000.0,
        adam_beta2=0.9,
        log_interval=50,
    )
    noise_tag = "noise"
    prev_job = None
    for seed in [43, 44, 45, 46, 47]:
        command = make_opt_command(
            seed=seed,
            noise_scale=1.0,
            results_base=f"$RESULTS_DIR/opt/Adam_NoiseSeedSweep_3k_GT3_NoDiffLifetime/{noise_tag}",
            wandb_tags=(wandb_tags or []) + ["Adam_NoiseSeedSweep_3k_GT3_NoDiffLifetime", noise_tag],
            **shared,
        )
        if not print_sbatch_only:
            print(command)
        prev_job = s3df_submit(
            command,
            time="01:05:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            dependency=prev_job,
        )


def profile_15Trk_Adam_NoiseSeedSweep_3k_0p1mm_step_GT(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    skip_complete=False,
    verbose=False,
):
    """Like Adam_NoiseSeedSweep_3k but GT uses 0.1mm step / 50k deposits; bs=1, eff_bs=15.

    4 jobs total: 2 seeds (43, 44) × 2 noise conditions. No dependency chain; all parallel.
    """
    params = ",".join(PARAM_LIST)
    shared = dict(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=1,
        effective_batch_size=15,
        gt_step_size=0.1,
        gt_max_deposits=50000,
        adam_beta2=0.9,
        log_interval=50,
    )
    results_base = "$RESULTS_DIR/opt/Adam_NoiseSeedSweep_3k_0p1mm_step_GT"
    for noise_scale in [1.0, 0.0]:
        noise_tag = "noise" if noise_scale > 0.0 else "nonoise"
        for seed in [43, 44]:
            if verbose or skip_complete:
                is_done = _seed_is_complete(results_base, noise_tag, seed, verbose=verbose)
                if is_done and skip_complete:
                    print(f"  SKIP complete: {noise_tag} seed={seed}")
                    continue
                elif not is_done and verbose:
                    print(f"  → will submit: {noise_tag} seed={seed}")

            command = make_opt_command(
                seed=seed,
                noise_scale=noise_scale,
                results_base=f"{results_base}/{noise_tag}",
                wandb_tags=(wandb_tags or []) + ["Adam_NoiseSeedSweep_3k_0p1mm_step_GT", noise_tag],
                **shared,
            )
            if not print_sbatch_only:
                print(command)
            s3df_submit(
                command,
                time="04:05:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
            )


def profile_15Trk_Adam_NoiseSeedSweep_3k_0p1mm_step_GT_and_sim(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    skip_complete=False,
    verbose=False,
):
    """Like Adam_NoiseSeedSweep_3k but both GT and sim use 0.1mm step / 50k deposits; bs=1, eff_bs=15.

    10 jobs total: 5 seeds (43–47) × 2 noise conditions. No dependency chain; all parallel.
    """
    params = ",".join(PARAM_LIST)
    shared = dict(
        params=params,
        tracks=TRACKS_15_BOUNDARY,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=0.1,
        max_num_deposits=50000,
        batch_size=1,
        effective_batch_size=15,
        gt_step_size=0.1,
        gt_max_deposits=50000,
        adam_beta2=0.9,
        log_interval=50,
    )
    results_base = "$RESULTS_DIR/opt/Adam_NoiseSeedSweep_3k_0p1mm_step_GT_and_sim"
    for noise_scale in [1.0, 0.0]:
        noise_tag = "noise" if noise_scale > 0.0 else "nonoise"
        for seed in [43, 44, 45, 46, 47]:
            if verbose or skip_complete:
                is_done = _seed_is_complete(results_base, noise_tag, seed, verbose=verbose)
                if is_done and skip_complete:
                    print(f"  SKIP complete: {noise_tag} seed={seed}")
                    continue
                elif not is_done and verbose:
                    print(f"  → will submit: {noise_tag} seed={seed}")

            command = make_opt_command(
                seed=seed,
                noise_scale=noise_scale,
                results_base=f"{results_base}/{noise_tag}",
                wandb_tags=(wandb_tags or []) + ["Adam_NoiseSeedSweep_3k_0p1mm_step_GT_and_sim", noise_tag],
                **shared,
            )
            if not print_sbatch_only:
                print(command)
            s3df_submit(
                command,
                time="04:05:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
            )


def _load_sweep_pkls(base_dir: str) -> list:
    """Return list of dicts with keys: path, wandb_run_id, seed, noise_scale.

    Scans base_dir/**/result_*.pkl recursively and reads each pickle.
    Skips files that are missing a wandb_run_id.
    """
    resolved = base_dir.replace("$RESULTS_DIR", _RESULTS_DIR)
    pattern  = os.path.join(resolved, "**", "result_*.pkl")
    entries  = []
    for pkl_path in sorted(glob.glob(pattern, recursive=True)):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            print(f"  [warn] could not read {pkl_path}: {exc}")
            continue
        run_id = data.get("wandb_run_id")
        if not run_id:
            print(f"  [warn] no wandb_run_id in {pkl_path}, skipping")
            continue
        entries.append(dict(
            path         = pkl_path,
            wandb_run_id = run_id,
            seed         = data.get("seed"),
            noise_scale  = data.get("noise_scale", 0.0),
        ))
    return entries


def profile_Adam_NoiseSeedSweep_3k_Cont_Newton(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    skip_complete=False,
    verbose=False,
):
    """Newton continuation from every Adam_NoiseSeedSweep_3k and GT2 run on disk.

    Scans results/opt/Adam_NoiseSeedSweep_3k and Adam_NoiseSeedSweep_3k_GT2 for
    completed pkl files, reads their wandb_run_id, and submits one Newton job per
    run that initialises from the last logged step (step 3000).

    Settings:
      optimizer   = newton, damping = 0.01, lr = 1.0
      max_steps   = 100, log_interval = 5
      batch_size  = 1, effective_batch_size = 15
      time limit  = 20 min
    """
    params = ",".join(PARAM_LIST)

    shared = dict(
        params              = params,
        tracks              = TRACKS_15_BOUNDARY,
        optimizer           = "newton",
        loss                = "sobolev_loss_geomean_log1p",
        lr                  = 1.0,
        lr_schedule         = "cosine",
        max_steps           = 100,
        tol                 = 1e-9,
        patience            = 200,
        N                   = 1,
        range_lo            = 0.9,
        range_hi            = 1.1,
        grad_clip           = -1,
        warmup_steps        = 0,
        num_buckets         = 1000,
        step_size           = 1.0,
        max_num_deposits    = 5000,
        batch_size          = 1,
        effective_batch_size= 15,
        gt_step_size        = 1.0,
        gt_max_deposits     = 5000,
        newton_damping      = 0.01,
        log_interval        = 5,
    )

    for gt_label, base_dir, gt_multiplier, wandb_tag in [
        ("GT1", "$RESULTS_DIR/opt/Adam_NoiseSeedSweep_3k",      None, "Adam_GT1_Cont_Newton_from3k"),
        ("GT2", "$RESULTS_DIR/opt/Adam_NoiseSeedSweep_3k_GT2",  1.2,  "Adam_GT2_Cont_Newton_from3k"),
    ]:
        runs = _load_sweep_pkls(base_dir)
        print(f"  {gt_label}: found {len(runs)} pkl(s) under {base_dir.replace('$RESULTS_DIR', _RESULTS_DIR)}")
        for run in runs:
            noise_tag    = "noise" if run["noise_scale"] > 0.0 else "nonoise"
            results_base = (
                f"$RESULTS_DIR/opt/Adam_NoiseSeedSweep_3k{'_GT2' if gt_multiplier else ''}"
                f"_Newton_cont/{noise_tag}"
            )

            if skip_complete:
                resolved = results_base.replace("$RESULTS_DIR", _RESULTS_DIR)
                pattern  = os.path.join(resolved, "**", f"result_{run['seed']}.pkl")
                existing = glob.glob(pattern, recursive=True)
                if existing:
                    if verbose:
                        print(f"  SKIP complete: {gt_label} {noise_tag} seed={run['seed']}")
                    continue
                elif verbose:
                    print(f"  → will submit: {gt_label} {noise_tag} seed={run['seed']} run={run['wandb_run_id']}")

            extra = {}
            if gt_multiplier is not None:
                extra["gt_param_multiplier"] = gt_multiplier

            command = make_opt_command(
                seed                = run["seed"],
                noise_scale         = run["noise_scale"],
                results_base        = results_base,
                init_from_wandb_run = run["wandb_run_id"],
                wandb_tags          = (wandb_tags or []) + [wandb_tag, noise_tag],
                **shared,
                **extra,
            )
            if not print_sbatch_only:
                print(command)
            s3df_submit(
                command,
                time="00:20:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
            )


def profile_1d_Grad_diffusion_debug(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    overwrite=False,
):
    """1D gradient sweep over ±50% for each diffusion coefficient, 7-track subset.

    4 jobs total: 2 params × 2 noise conditions.  N=2 → 5 points per sweep.
    Tracks: Muon1_1000MeV, Muon2_500MeV, all four 100 MeV tracks (Muon4/5/10/12),
    Muon_diagCross_1000MeV.
    Step 1 mm, max 5k deposits.
    Full 2D signal arrays for all 6 planes (U1,V1,Y1,U2,V2,Y2) stored per track.
    Results land in $RESULTS_DIR/1d_gradients/.
    Pass overwrite=True to delete existing pkl files before submitting.
    """
    python = "python"
    script = "src/analysis/1d_gradients.py"
    tracks = (
        "Muon1_1000MeV:-0.747872530,0.661463000,0.056154945:1000"
        "+Muon2_500MeV:-0.641581737,0.275323919,-0.715939672:500"
        "+Muon4_100MeV:-0.694627880,0.476880059,0.538588450:100"
        "+Muon5_100MeV:-0.448568523,-0.712616910,0.539410252:100:2160.0,1057.372513522,2019.642044116"
        "+Muon10_100MeV:0.754859526,-0.437194999,0.488924973:100:-2160.0,296.961966517,-1556.076968089"
        "+Muon12_100MeV:-0.553810025,-0.123483953,-0.823435589:100:2160.0,-2026.866954667,-273.380878516"
        "+Muon_diagCross_1000MeV:-0.577350269,-0.577350269,-0.577350269:1000:2000,2000,2000"
    )

    results_subdir = "diffusion_debug_20260519_7tracks"
    results_dir_arg = f"$RESULTS_DIR/1d_gradients/{results_subdir}"
    results_dir_resolved = results_dir_arg.replace("$RESULTS_DIR", _RESULTS_DIR)
    common = (
        f"{python} {script}"
        f" --N 2"
        f" --range-frac 0.5"
        f" --loss default"
        f" --tracks '{tracks}'"
        f" --step-size 1.0"
        f" --max-deposits 5000"
        f" --store-arrays"
        f" --store-per-pixel-loss-and-grad"
        f" --results-dir {results_dir_arg}"
    )

    prev_job = None
    for param in ("diffusion_trans_cm2_us", "diffusion_long_cm2_us"):
        for noise_scale, noise_seed in ((0.0, None), (1.0, 42)):
            noise_tag = f"_noise{noise_scale:.3g}".replace(".", "p") if noise_scale > 0.0 else ""
            expected_pkl = os.path.join(
                results_dir_resolved,
                f"sobolev_loss_geomean_log1p_N2_range0p5_{param}_7tracks{noise_tag}.pkl",
            )
            if overwrite and os.path.exists(expected_pkl):
                os.remove(expected_pkl)
                print(f"  Removed {expected_pkl}")
            if not overwrite and os.path.exists(expected_pkl):
                print(f"  Skipping {param} noise={noise_scale} [{results_subdir}]: output already exists: {expected_pkl}")
                continue
            noise_args = f" --noise-scale {noise_scale}"
            if noise_seed is not None:
                noise_args += f" --noise-seed {noise_seed}"
            command = f"{common} --param {param}{noise_args}"
            if not print_sbatch_only:
                print(command)
            prev_job = s3df_submit(
                command,
                time="00:15:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                dependency=prev_job,
            )


def profile_15Trk_Adam_NoiseCutoffDiffusion_3k(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    skip_complete=False,
    verbose=False,
):
    """Diffusion calibration with noisy GT and ADC cutoffs.

    Noisy GT (noise_scale=1.0): 3 seeds × 3 param sets × 4 cutoffs = 36 jobs.
    Clean GT (noise_scale=0.0): 3 seeds × 3 param sets × 2 cutoffs (5, 20) = 18 jobs.
    Param sets: trans-diff only, long-diff only, both diffs.
    Seeds are chained within each (param_set, cutoff, noise) group.
    """
    shared = dict(
        tracks=TRACKS_15_BOUNDARY,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=1.0,
        max_num_deposits=5000,
        batch_size=5,
        effective_batch_size=3,
        gt_step_size=1.0,
        gt_max_deposits=5000,
        adam_beta2=0.9,
        log_interval=50,
    )

    param_sets = [
        ("trans_only", "diffusion_trans_cm2_us"),
        ("long_only",  "diffusion_long_cm2_us"),
        ("both_diff",  "diffusion_trans_cm2_us,diffusion_long_cm2_us"),
    ]

    for cutoff in [5.0, 10.0, 15.0, 20.0]:
        cutoff_tag = f"cutoff{int(cutoff)}"
        for param_label, params in param_sets:
            profile_tag = f"Adam_NoiseCutoffDiffusion_3k_{param_label}_{cutoff_tag}"
            results_base = f"$RESULTS_DIR/opt/{profile_tag}"
            prev_job = None
            for seed in [43, 44, 45]:
                if verbose or skip_complete:
                    is_done = _seed_is_complete(results_base, "noise", seed, verbose=verbose)
                    if is_done and skip_complete:
                        print(f"  SKIP complete: {profile_tag} seed={seed}")
                        continue
                    elif not is_done and verbose:
                        print(f"  → will submit: {profile_tag} seed={seed}")

                command = make_opt_command(
                    params=params,
                    seed=seed,
                    noise_scale=1.0,
                    sobolev_loss_cutoff=cutoff,
                    results_base=f"{results_base}/noise",
                    wandb_tags=(wandb_tags or []) + [cutoff_tag, param_label, "noise"],
                    **shared,
                )
                if not print_sbatch_only:
                    print(command)
                prev_job = s3df_submit(
                    command,
                    time="01:05:00",
                    submit=submit,
                    mem_gb=64,
                    print_sbatch_command=print_sbatch_only,
                    dependency=prev_job,
                )

    # Clean GT (no noise) — cutoffs 5 and 20 only
    for cutoff in [5.0, 20.0]:
        cutoff_tag = f"cutoff{int(cutoff)}"
        for param_label, params in param_sets:
            profile_tag = f"Adam_NoiseCutoffDiffusion_3k_{param_label}_{cutoff_tag}"
            results_base = f"$RESULTS_DIR/opt/{profile_tag}"
            prev_job = None
            for seed in [43, 44, 45]:
                if verbose or skip_complete:
                    is_done = _seed_is_complete(results_base, "nonoise", seed, verbose=verbose)
                    if is_done and skip_complete:
                        print(f"  SKIP complete: {profile_tag}/nonoise seed={seed}")
                        continue
                    elif not is_done and verbose:
                        print(f"  → will submit: {profile_tag}/nonoise seed={seed}")

                command = make_opt_command(
                    params=params,
                    seed=seed,
                    noise_scale=0.0,
                    sobolev_loss_cutoff=cutoff,
                    results_base=f"{results_base}/nonoise",
                    wandb_tags=(wandb_tags or []) + [cutoff_tag, param_label, "nonoise"],
                    **shared,
                )
                if not print_sbatch_only:
                    print(command)
                prev_job = s3df_submit(
                    command,
                    time="01:05:00",
                    submit=submit,
                    mem_gb=64,
                    print_sbatch_command=print_sbatch_only,
                    dependency=prev_job,
                )


def profile_15Trk_Adam_NoiseCutoffDiffusion_3k_LargeSpread(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    skip_complete=False,
    verbose=False,
):
    """Transverse diffusion calibration with ±75% initial spread, diagonal track only.

    Single diagonal track from (2000,2000,2000) along (-1,-1,-1)/√3 at 1000 MeV.
    Noisy GT (noise_scale=1.0) only: 4 seeds × 2 cutoffs (20, 25) = 8 jobs.
    Only trans_only param set; seeds 100/101/102/103.
    Seeds are chained within each cutoff.
    """
    _DIAG_TRACK = "Muon_diagBody_1000MeV:-0.577350269,-0.577350269,-0.577350269:1000"

    shared = dict(
        tracks=_DIAG_TRACK,
        start_position_mm="2000,2000,2000",
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.25,
        range_hi=1.75,
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
    )

    param_label = "trans_only"
    params      = "diffusion_trans_cm2_us"

    for cutoff in [20.0, 25.0]:
        cutoff_tag   = f"cutoff{int(cutoff)}"
        profile_tag  = f"Adam_NoiseCutoffDiffusion_3k_LargeSpread_{param_label}_{cutoff_tag}"
        results_base = f"$RESULTS_DIR/opt/{profile_tag}"

        # Noisy GT
        prev_job = None
        for seed in [100, 101, 102, 103]:
            if verbose or skip_complete:
                is_done = _seed_is_complete(results_base, "noise", seed, verbose=verbose)
                if is_done and skip_complete:
                    print(f"  SKIP complete: {profile_tag} seed={seed}")
                    continue
                elif not is_done and verbose:
                    print(f"  → will submit: {profile_tag} seed={seed}")

            command = make_opt_command(
                params=params,
                seed=seed,
                noise_scale=1.0,
                sobolev_loss_cutoff=cutoff,
                results_base=f"{results_base}/noise",
                wandb_tags=(wandb_tags or []) + [cutoff_tag, param_label, "noise", "large_spread"],
                **shared,
            )
            if not print_sbatch_only:
                print(command)
            prev_job = s3df_submit(
                command,
                time="01:05:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                dependency=prev_job,
            )


def profile_Adam_NoiseCutoff25_DebugTracks_3k(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    skip_complete=False,
    verbose=False,
):
    """Diffusion calibration with ADC cutoff=25, ±75% spread, debug 4-track ensemble.

    Same 4 tracks used in the generate_gradient_viewer.py visualizations
    (diffusion_debug_20260519_4tracks): Muon1_1000MeV, Muon2_500MeV, Muon4_100MeV,
    Muon_diagCross_1000MeV (all starting at origin).
    Noisy GT (noise_scale=1.0), 3 param sets × 4 seeds = 12 jobs.
    Seeds chained within each param set.
    """
    _DEBUG_4TRACKS = (
        "Muon1_1000MeV:-0.747872530,0.661463000,0.056154945:1000"
        "+Muon2_500MeV:-0.641581737,0.275323919,-0.715939672:500"
        "+Muon4_100MeV:-0.694627880,0.476880059,0.538588450:100"
        "+Muon_diagCross_1000MeV:-0.577350269,-0.577350269,-0.577350269:1000"
    )

    shared = dict(
        tracks=_DEBUG_4TRACKS,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.25,
        range_hi=1.75,
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
        sobolev_loss_cutoff=25.0,
    )

    cutoff_tag = "cutoff25"

    param_sets = [
        ("trans_only",     "diffusion_trans_cm2_us"),
        ("long_only",      "diffusion_long_cm2_us"),
        ("trans_and_long", "diffusion_trans_cm2_us,diffusion_long_cm2_us"),
    ]

    for param_label, params in param_sets:
        profile_tag  = f"Adam_NoiseCutoff25_DebugTracks_{param_label}_indepnoise"
        results_base = f"$RESULTS_DIR/opt/{profile_tag}"

        prev_job = None
        for seed in [44, 45, 46, 47]:
            if verbose or skip_complete:
                is_done = _seed_is_complete(results_base, "noise", seed, verbose=verbose)
                if is_done and skip_complete:
                    print(f"  SKIP complete: {profile_tag} seed={seed}")
                    continue
                elif not is_done and verbose:
                    print(f"  → will submit: {profile_tag} seed={seed}")

            command = make_opt_command(
                params=params,
                seed=seed,
                noise_scale=1.0,
                results_base=f"{results_base}/noise",
                wandb_tags=(wandb_tags or []) + [cutoff_tag, param_label, "noise", "debug_4trk"],
                **shared,
            )
            if not print_sbatch_only:
                print(command)
            prev_job = s3df_submit(
                command,
                time="01:05:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                dependency=prev_job,
            )


def profile_Adam_NoiseCutoff25_DebugTracks_3k_CollectionOnly(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    skip_complete=False,
    verbose=False,
):
    """Same as Adam_NoiseCutoff25_DebugTracks_3k but loss computed over collection planes only (Y1, Y2).

    Identical to profile_Adam_NoiseCutoff25_DebugTracks_3k in every respect except
    ``--planes Y1,Y2`` restricts the Sobolev loss to the two collection planes.
    """
    _DEBUG_4TRACKS = (
        "Muon1_1000MeV:-0.747872530,0.661463000,0.056154945:1000"
        "+Muon2_500MeV:-0.641581737,0.275323919,-0.715939672:500"
        "+Muon4_100MeV:-0.694627880,0.476880059,0.538588450:100"
        "+Muon_diagCross_1000MeV:-0.577350269,-0.577350269,-0.577350269:1000"
    )

    shared = dict(
        tracks=_DEBUG_4TRACKS,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.25,
        range_hi=1.75,
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
        sobolev_loss_cutoff=25.0,
        planes="Y1,Y2",
    )

    cutoff_tag = "cutoff25"

    param_sets = [
        ("trans_only",     "diffusion_trans_cm2_us"),
        ("long_only",      "diffusion_long_cm2_us"),
        ("trans_and_long", "diffusion_trans_cm2_us,diffusion_long_cm2_us"),
    ]

    for param_label, params in param_sets:
        profile_tag  = f"Adam_NoiseCutoff25_DebugTracks_{param_label}_CollectionOnly"
        results_base = f"$RESULTS_DIR/opt/{profile_tag}"

        prev_job = None
        for seed in [44, 45, 46, 47]:
            if verbose or skip_complete:
                is_done = _seed_is_complete(results_base, "noise", seed, verbose=verbose)
                if is_done and skip_complete:
                    print(f"  SKIP complete: {profile_tag} seed={seed}")
                    continue
                elif not is_done and verbose:
                    print(f"  → will submit: {profile_tag} seed={seed}")

            command = make_opt_command(
                params=params,
                seed=seed,
                noise_scale=1.0,
                results_base=f"{results_base}/noise",
                wandb_tags=(wandb_tags or []) + [cutoff_tag, param_label, "noise", "debug_4trk", "collection_only"],
                **shared,
            )
            if not print_sbatch_only:
                print(command)
            prev_job = s3df_submit(
                command,
                time="01:05:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                dependency=prev_job,
            )


def profile_Adam_NoiseCutoff25_DebugTracks_3k_CollectionOnly_3trk_MoreCutoffs(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    skip_complete=False,
    verbose=False,
):
    """Same as Adam_NoiseCutoff25_DebugTracks_3k_CollectionOnly_3trk but sweeps cutoffs 5,10,15,20,30.

    3 tracks (no Muon4_100MeV), collection planes only (Y1,Y2), ±75% spread, noise_scale=1.0.
    5 cutoffs × 3 param sets × 4 seeds = 60 jobs.
    """
    _DEBUG_3TRACKS = (
        "Muon1_1000MeV:-0.747872530,0.661463000,0.056154945:1000"
        "+Muon2_500MeV:-0.641581737,0.275323919,-0.715939672:500"
        "+Muon_diagCross_1000MeV:-0.577350269,-0.577350269,-0.577350269:1000"
    )

    shared = dict(
        tracks=_DEBUG_3TRACKS,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.25,
        range_hi=1.75,
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
        planes="Y1,Y2",
    )

    param_sets = [
        ("trans_only",     "diffusion_trans_cm2_us"),
        ("long_only",      "diffusion_long_cm2_us"),
        ("trans_and_long", "diffusion_trans_cm2_us,diffusion_long_cm2_us"),
    ]

    for cutoff in [5.0, 10.0, 15.0, 20.0, 30.0]:
        cutoff_tag = f"cutoff{int(cutoff)}"
        prev_job = None
        for param_label, params in param_sets:
            profile_tag  = f"Adam_NoiseCutoff25_DebugTracks_{param_label}_CollectionOnly_3trk_{cutoff_tag}"
            results_base = f"$RESULTS_DIR/opt/{profile_tag}"

            for seed in [44, 45, 46, 47]:
                if verbose or skip_complete:
                    is_done = _seed_is_complete(results_base, "noise", seed, verbose=verbose)
                    if is_done and skip_complete:
                        print(f"  SKIP complete: {profile_tag} seed={seed}")
                        continue
                    elif not is_done and verbose:
                        print(f"  → will submit: {profile_tag} seed={seed}")

                command = make_opt_command(
                    params=params,
                    seed=seed,
                    noise_scale=1.0,
                    sobolev_loss_cutoff=cutoff,
                    results_base=f"{results_base}/noise",
                    wandb_tags=(wandb_tags or []) + [cutoff_tag, param_label, "noise", "debug_3trk", "collection_only"],
                    **shared,
                )
                if not print_sbatch_only:
                    print(command)
                prev_job = s3df_submit(
                    command,
                    time="01:05:00",
                    submit=submit,
                    mem_gb=64,
                    print_sbatch_command=print_sbatch_only,
                    dependency=prev_job,
                )


def profile_Adam_NoiseCutoff25_DebugTracks_3k_CollectionOnly_3trk(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    skip_complete=False,
    verbose=False,
):
    """Same as Adam_NoiseCutoff25_DebugTracks_3k_CollectionOnly but with Muon4_100MeV removed.

    3 tracks: Muon1_1000MeV, Muon2_500MeV, Muon_diagCross_1000MeV.
    Loss over collection planes only (Y1, Y2).
    """
    _DEBUG_3TRACKS = (
        "Muon1_1000MeV:-0.747872530,0.661463000,0.056154945:1000"
        "+Muon2_500MeV:-0.641581737,0.275323919,-0.715939672:500"
        "+Muon_diagCross_1000MeV:-0.577350269,-0.577350269,-0.577350269:1000"
    )

    shared = dict(
        tracks=_DEBUG_3TRACKS,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.25,
        range_hi=1.75,
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
        sobolev_loss_cutoff=25.0,
        planes="Y1,Y2",
    )

    cutoff_tag = "cutoff25"

    param_sets = [
        ("trans_only",     "diffusion_trans_cm2_us"),
        ("long_only",      "diffusion_long_cm2_us"),
        ("trans_and_long", "diffusion_trans_cm2_us,diffusion_long_cm2_us"),
    ]

    for param_label, params in param_sets:
        profile_tag  = f"Adam_NoiseCutoff25_DebugTracks_{param_label}_CollectionOnly_3trk"
        results_base = f"$RESULTS_DIR/opt/{profile_tag}"

        prev_job = None
        for seed in [44, 45, 46, 47]:
            if verbose or skip_complete:
                is_done = _seed_is_complete(results_base, "noise", seed, verbose=verbose)
                if is_done and skip_complete:
                    print(f"  SKIP complete: {profile_tag} seed={seed}")
                    continue
                elif not is_done and verbose:
                    print(f"  → will submit: {profile_tag} seed={seed}")

            command = make_opt_command(
                params=params,
                seed=seed,
                noise_scale=1.0,
                results_base=f"{results_base}/noise",
                wandb_tags=(wandb_tags or []) + [cutoff_tag, param_label, "noise", "debug_3trk", "collection_only"],
                **shared,
            )
            if not print_sbatch_only:
                print(command)
            prev_job = s3df_submit(
                command,
                time="01:05:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                dependency=prev_job,
            )


def profile_Adam_NoiseCutoff50_DebugTracks_3k_3trk_BothPlanes(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    skip_complete=False,
    verbose=False,
):
    """ADC cutoff=50, ±75% spread, debug 3-track ensemble, noisy GT.

    Same as Adam_NoiseCutoff25_DebugTracks_3k_CollectionOnly_3trk but:
      • sobolev_loss_cutoff = 50.0 (was 25)
      • Submits each param set twice: once collection-only (Y1,Y2) and once all planes.
    3 param sets × 2 plane configs × 4 seeds = 24 jobs.
    Seeds chained within each (param_set, plane_config) group.
    """
    _DEBUG_3TRACKS = (
        "Muon1_1000MeV:-0.747872530,0.661463000,0.056154945:1000"
        "+Muon2_500MeV:-0.641581737,0.275323919,-0.715939672:500"
        "+Muon_diagCross_1000MeV:-0.577350269,-0.577350269,-0.577350269:1000"
    )

    shared = dict(
        tracks=_DEBUG_3TRACKS,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.25,
        range_hi=1.75,
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
        sobolev_loss_cutoff=50.0,
    )

    cutoff_tag = "cutoff50"

    param_sets = [
        ("trans_only",     "diffusion_trans_cm2_us"),
        ("long_only",      "diffusion_long_cm2_us"),
        ("trans_and_long", "diffusion_trans_cm2_us,diffusion_long_cm2_us"),
    ]

    plane_configs = [
        ("collection_only", "Y1,Y2"),
        ("all_planes",      None),
    ]

    # Collect all pending commands in submission order, then assign to 2 lanes
    # so at most 2 jobs run simultaneously.
    pending = []
    for param_label, params in param_sets:
        for plane_tag, planes in plane_configs:
            profile_tag  = f"Adam_NoiseCutoff50_DebugTracks_{param_label}_{plane_tag}_3trk"
            results_base = f"$RESULTS_DIR/opt/{profile_tag}"
            for seed in [44, 45, 46, 47]:
                if verbose or skip_complete:
                    is_done = _seed_is_complete(results_base, "noise", seed, verbose=verbose)
                    if is_done and skip_complete:
                        print(f"  SKIP complete: {profile_tag} seed={seed}")
                        continue
                    elif not is_done and verbose:
                        print(f"  → will submit: {profile_tag} seed={seed}")
                command = make_opt_command(
                    params=params,
                    seed=seed,
                    noise_scale=1.0,
                    results_base=f"{results_base}/noise",
                    planes=planes,
                    wandb_tags=(wandb_tags or []) + [cutoff_tag, param_label, "noise", "debug_3trk", plane_tag],
                    **shared,
                )
                pending.append(command)

    # Submit across 2 lanes: job i goes to lane i%2, depends on the previous job in that lane
    prev = [None, None]
    for i, command in enumerate(pending):
        lane = i % 2
        if not print_sbatch_only:
            print(command)
        prev[lane] = s3df_submit(
            command,
            time="01:05:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            dependency=prev[lane],
        )


def profile_Adam_NoiseCutoff25_DebugTracks_3k_3trk(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    skip_complete=False,
    verbose=False,
):
    """Diffusion calibration with ADC cutoff=25, ±75% spread, debug 3-track ensemble.

    Same as Adam_NoiseCutoff25_DebugTracks_3k but with Muon4_100MeV removed:
    Muon1_1000MeV, Muon2_500MeV, Muon_diagCross_1000MeV.
    Noisy GT (noise_scale=1.0), 3 param sets × 4 seeds = 12 jobs.
    Seeds chained within each param set.
    """
    _DEBUG_3TRACKS = (
        "Muon1_1000MeV:-0.747872530,0.661463000,0.056154945:1000"
        "+Muon2_500MeV:-0.641581737,0.275323919,-0.715939672:500"
        "+Muon_diagCross_1000MeV:-0.577350269,-0.577350269,-0.577350269:1000"
    )

    shared = dict(
        tracks=_DEBUG_3TRACKS,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.25,
        range_hi=1.75,
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
        sobolev_loss_cutoff=25.0,
    )

    cutoff_tag = "cutoff25"

    param_sets = [
        ("trans_only",     "diffusion_trans_cm2_us"),
        ("long_only",      "diffusion_long_cm2_us"),
        ("trans_and_long", "diffusion_trans_cm2_us,diffusion_long_cm2_us"),
    ]

    for param_label, params in param_sets:
        profile_tag  = f"Adam_NoiseCutoff25_DebugTracks_3trk_{param_label}"
        results_base = f"$RESULTS_DIR/opt/{profile_tag}"

        prev_job = None
        for seed in [44, 45, 46, 47]:
            if verbose or skip_complete:
                is_done = _seed_is_complete(results_base, "noise", seed, verbose=verbose)
                if is_done and skip_complete:
                    print(f"  SKIP complete: {profile_tag} seed={seed}")
                    continue
                elif not is_done and verbose:
                    print(f"  → will submit: {profile_tag} seed={seed}")

            command = make_opt_command(
                params=params,
                seed=seed,
                noise_scale=1.0,
                results_base=f"{results_base}/noise",
                wandb_tags=(wandb_tags or []) + [cutoff_tag, param_label, "noise", "debug_3trk"],
                **shared,
            )
            if not print_sbatch_only:
                print(command)
            prev_job = s3df_submit(
                command,
                time="01:05:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                dependency=prev_job,
            )


def profile_Adam_NoiseCutoff25_DebugTracks_3k_TransLong_Chain(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    skip_complete=False,
    verbose=False,
):
    """Diffusion calibration with ADC cutoff=25, ±75% spread, debug 4-track ensemble.

    Trans-only and long-only, seeds 1001–1003 each. All 6 jobs chained into a
    single sequence so at most one GPU job runs at a time.
    """
    _DEBUG_4TRACKS = (
        "Muon1_1000MeV:-0.747872530,0.661463000,0.056154945:1000"
        "+Muon2_500MeV:-0.641581737,0.275323919,-0.715939672:500"
        "+Muon4_100MeV:-0.694627880,0.476880059,0.538588450:100"
        "+Muon_diagCross_1000MeV:-0.577350269,-0.577350269,-0.577350269:1000"
    )

    shared = dict(
        tracks=_DEBUG_4TRACKS,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.25,
        range_hi=1.75,
        grad_clip=0.0,
        warmup_steps=1000,
        num_buckets=1000,
        step_size=0.1,
        max_num_deposits=50000,
        batch_size=1,
        effective_batch_size=1,
        gt_step_size=0.1,
        gt_max_deposits=50000,
        adam_beta2=0.9,
        log_interval=50,
        sobolev_loss_cutoff=25.0,
    )

    cutoff_tag = "cutoff25"

    param_sets = [
        ("trans_only", "diffusion_trans_cm2_us"),
        ("long_only",  "diffusion_long_cm2_us"),
    ]

    prev_job = None  # single chain across all param sets
    for param_label, params in param_sets:
        profile_tag  = f"Adam_NoiseCutoff25_DebugTracks_3k_TransLong_{param_label}"
        results_base = f"$RESULTS_DIR/opt/{profile_tag}"

        for seed in [1001, 1002, 1003]:
            if verbose or skip_complete:
                is_done = _seed_is_complete(results_base, "noise", seed, verbose=verbose)
                if is_done and skip_complete:
                    print(f"  SKIP complete: {profile_tag} seed={seed}")
                    continue
                elif not is_done and verbose:
                    print(f"  → will submit: {profile_tag} seed={seed}")

            command = make_opt_command(
                params=params,
                seed=seed,
                noise_scale=1.0,
                results_base=f"{results_base}/noise",
                wandb_tags=(wandb_tags or []) + [cutoff_tag, param_label, "noise", "debug_4trk"],
                **shared,
            )
            if not print_sbatch_only:
                print(command)
            prev_job = s3df_submit(
                command,
                time="01:05:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                dependency=prev_job,
            )


def profile_Adam_NoiseCutoff25_4Trk_LargeSpread(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    skip_complete=False,
    verbose=False,
):
    """Diffusion calibration with ADC cutoff=25, ±75% spread, 4-track sobolev-cutoff ensemble.

    Tracks: diagonal + Muon1_1000MeV + Muon2_500MeV + Muon4_100MeV
    (same 4 as Run_SobolevLossWithCutoff.sh / sobolev_cutoff_diffusion_N20_range75pct).
    Noisy GT (noise_scale=1.0), both diffusion params, seeds 100–103.
    Seeds chained within each param set.
    """
    _4TRACKS = (
        "diagonal:-0.577350,-0.577350,-0.577350:1000"
        "+Muon1_1000MeV:-0.747872530,0.661463000,0.056154945:1000"
        "+Muon2_500MeV:-0.641581737,0.275323919,-0.715939672:500"
        "+Muon4_100MeV:-0.694627880,0.476880059,0.538588450:100"
    )

    shared = dict(
        tracks=_4TRACKS,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        max_steps=3000,
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.25,
        range_hi=1.75,
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
        sobolev_loss_cutoff=25.0,
    )

    cutoff_tag = "cutoff25"

    param_sets = [
        ("trans_only",      "diffusion_trans_cm2_us"),
        ("long_only",       "diffusion_long_cm2_us"),
        ("trans_and_long",  "diffusion_trans_cm2_us,diffusion_long_cm2_us"),
    ]

    for param_label, params in param_sets:
        profile_tag  = f"Adam_NoiseCutoff25_4Trk_LargeSpread_{param_label}"
        results_base = f"$RESULTS_DIR/opt/{profile_tag}"

        prev_job = None
        for seed in [100, 101, 102, 103]:
            if verbose or skip_complete:
                is_done = _seed_is_complete(results_base, "noise", seed, verbose=verbose)
                if is_done and skip_complete:
                    print(f"  SKIP complete: {profile_tag} seed={seed}")
                    continue
                elif not is_done and verbose:
                    print(f"  → will submit: {profile_tag} seed={seed}")

            command = make_opt_command(
                params=params,
                seed=seed,
                noise_scale=1.0,
                results_base=f"{results_base}/noise",
                wandb_tags=(wandb_tags or []) + [cutoff_tag, param_label, "noise", "4trk", "large_spread"],
                **shared,
            )
            if not print_sbatch_only:
                print(command)
            prev_job = s3df_submit(
                command,
                time="01:05:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                dependency=prev_job,
            )



def profile_Adam_NoiseCutoff25_DebugTracks_2phase(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    skip_complete=False,
    verbose=False,
):
    """Diffusion calibration on debug 4-track ensemble, cutoff=25, ±10% spread.

    Original 2-phase chain (4 jobs, sequential):
      Phase 1: trans+long diffusion only, 4000 steps.
      Phase 2: all parameters, 8000 steps.

    Added single-param chains (2 jobs each, independent):
      trans_only_8k: diffusion_trans_cm2_us only, 8000 steps.
      long_only_8k:  diffusion_long_cm2_us only, 8000 steps.

    Added growing-param chain (6 jobs, sequential) — see where opt breaks:
      trans_4k:                      trans diffusion, 4000 steps.
      trans_recomb_alpha_4k:         + recomb_alpha, 4000 steps.
      trans_recomb_alpha_beta90_4k:  + recomb_beta_90, 4000 steps.

    Same optimizer settings as Adam_NoiseCutoff25_DebugTracks_3k (lr=0.001, cosine,
    adam_beta2=0.9, step_size=1.0, sobolev_loss_cutoff=25, noise_scale=1.0).
    Seeds 1000 and 1001.
    """
    _DEBUG_4TRACKS = (
        "Muon1_1000MeV:-0.747872530,0.661463000,0.056154945:1000"
        "+Muon2_500MeV:-0.641581737,0.275323919,-0.715939672:500"
        "+Muon4_100MeV:-0.694627880,0.476880059,0.538588450:100"
        "+Muon_diagCross_1000MeV:-0.577350269,-0.577350269,-0.577350269:1000"
    )

    shared = dict(
        tracks=_DEBUG_4TRACKS,
        optimizer="adam",
        loss="sobolev_loss_geomean_log1p",
        lr=0.001,
        lr_schedule="cosine",
        tol=1e-6,
        patience=2000,
        N=1,
        range_lo=0.9,
        range_hi=1.1,
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
        sobolev_loss_cutoff=25.0,
    )

    cutoff_tag = "cutoff25"

    phases = [
        ("trans_and_long_4k", "diffusion_trans_cm2_us,diffusion_long_cm2_us", 4000, "01:30:00"),
        ("all_params_8k",     ",".join(PARAM_LIST),                           8000, "03:00:00"),
    ]

    prev_job = None
    for phase_label, params, max_steps, wall_time in phases:
        results_base = f"$RESULTS_DIR/opt/Adam_NoiseCutoff25_DebugTracks_2phase/{phase_label}"
        for seed in [1000, 1001]:
            if verbose or skip_complete:
                is_done = _seed_is_complete(results_base, "noise", seed, verbose=verbose)
                if is_done and skip_complete:
                    print(f"  SKIP complete: {phase_label} seed={seed}")
                    continue
                elif not is_done and verbose:
                    print(f"  → will submit: {phase_label} seed={seed}")

            command = make_opt_command(
                params=params,
                seed=seed,
                max_steps=max_steps,
                noise_scale=1.0,
                results_base=f"{results_base}/noise",
                wandb_tags=(wandb_tags or []) + [cutoff_tag, phase_label, "noise", "debug_4trk"],
                **shared,
            )
            if not print_sbatch_only:
                print(command)
            prev_job = s3df_submit(
                command,
                time=wall_time,
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                dependency=prev_job,
            )

    # ── trans-only and long-only single-phase jobs (independent chains) ───────
    single_param_phases = [
        ("trans_only_8k", "diffusion_trans_cm2_us", 8000, "02:30:00"),
        ("long_only_8k",  "diffusion_long_cm2_us",  8000, "02:30:00"),
    ]
    for phase_label, params, max_steps, wall_time in single_param_phases:
        prev_single = None
        results_base = f"$RESULTS_DIR/opt/Adam_NoiseCutoff25_DebugTracks_2phase/{phase_label}"
        for seed in [1000, 1001]:
            if verbose or skip_complete:
                is_done = _seed_is_complete(results_base, "noise", seed, verbose=verbose)
                if is_done and skip_complete:
                    print(f"  SKIP complete: {phase_label} seed={seed}")
                    continue
                elif not is_done and verbose:
                    print(f"  → will submit: {phase_label} seed={seed}")
            command = make_opt_command(
                params=params,
                seed=seed,
                max_steps=max_steps,
                noise_scale=1.0,
                results_base=f"{results_base}/noise",
                wandb_tags=(wandb_tags or []) + [cutoff_tag, phase_label, "noise", "debug_4trk"],
                **shared,
            )
            if not print_sbatch_only:
                print(command)
            prev_single = s3df_submit(
                command,
                time=wall_time,
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                dependency=prev_single,
            )

    # ── growing-param chain: see where optimization breaks ────────────────────
    growing_phases = [
        ("trans_4k",
         "diffusion_trans_cm2_us",
         4000, "01:30:00"),
        ("trans_recomb_alpha_4k",
         "diffusion_trans_cm2_us,recomb_alpha",
         4000, "01:30:00"),
        ("trans_recomb_alpha_beta90_4k",
         "diffusion_trans_cm2_us,recomb_alpha,recomb_beta_90",
         4000, "01:30:00"),
    ]
    prev_growing = None
    for phase_label, params, max_steps, wall_time in growing_phases:
        results_base = f"$RESULTS_DIR/opt/Adam_NoiseCutoff25_DebugTracks_2phase/{phase_label}"
        for seed in [1000, 1001]:
            if verbose or skip_complete:
                is_done = _seed_is_complete(results_base, "noise", seed, verbose=verbose)
                if is_done and skip_complete:
                    print(f"  SKIP complete: {phase_label} seed={seed}")
                    continue
                elif not is_done and verbose:
                    print(f"  → will submit: {phase_label} seed={seed}")
            command = make_opt_command(
                params=params,
                seed=seed,
                max_steps=max_steps,
                noise_scale=1.0,
                results_base=f"{results_base}/noise",
                wandb_tags=(wandb_tags or []) + [cutoff_tag, phase_label, "noise", "debug_4trk"],
                **shared,
            )
            if not print_sbatch_only:
                print(command)
            prev_growing = s3df_submit(
                command,
                time=wall_time,
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                dependency=prev_growing,
            )



def profile_gradient_cutoff_sweep_15trk(
    *,
    submit=True,
    print_sbatch_only=False,
    wandb_tags=None,
    n_jobs=2,
    time="20:00:00",
):
    """1d-gradient landscape sweep: 15 tracks + 30 fraction-slice tracks, 10 ADC cutoffs.

    Params       : diffusion_trans_cm2_us, diffusion_long_cm2_us
    Noise        : 0.0 (seed 42) and 1.0 (seeds 42, 0, 1, 2, 3, 4, 5, 6)
    Cutoffs      : 0, 1, 2, 5, 10, 15, 20, 25, 30, 50  (ADC units)
    N=20, ±75% range, step_size=1.0 mm, max_deposits=5000.
    Output       : $RESULTS_DIR/1d_gradients/sobolev_cutoff_15trk_all_planes/

    Structure: each 1d_gradients.py call covers all tracks (combined with '+')
    AND all 10 cutoffs (--adc-cutoffs), so it compiles once and sweeps everything.
    32 commands total: 2 sets (15 full tracks + 30 fraction tracks)
                       × 2 params × (1 no-noise + 8 noisy seeds).
    The fraction-track pkls are merged with the full-track pkls by the viewer
    via the (param, noise, seed, cutoff) key.
    With n_jobs=2 (default) each job handles 16 invocations.
    Increase n_jobs to distribute across more Slurm jobs.
    """
    results_dir      = "$RESULTS_DIR/1d_gradients/sobolev_cutoff_15trk_all_planes"
    cutoffs          = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0]
    params           = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
    noise_seeds      = [42, 0, 1, 2, 3, 4, 5, 6]  # 8 seeds for the noisy (noise_scale=1.0) runs

    all_tracks_arg      = "+".join(spec for _, spec in _GRADIENT_15_TRACKS)
    fraction_tracks_arg = "+".join(spec for _, spec in _GRADIENT_FRACTION_TRACKS)

    all_commands = []
    for tracks_arg in [all_tracks_arg, fraction_tracks_arg]:
        for param in params:
            # no-noise run
            all_commands.append(make_gradient_command(
                param=param,
                tracks=tracks_arg,
                N=20,
                range_frac=0.75,
                noise_scale=0.0,
                noise_seed=42,
                adc_cutoffs=cutoffs,
                results_dir=results_dir,
                store_per_plane_loss=True,
            ))
            # noisy runs — one per seed
            for seed in noise_seeds:
                all_commands.append(make_gradient_command(
                    param=param,
                    tracks=tracks_arg,
                    N=20,
                    range_frac=0.75,
                    noise_scale=1.0,
                    noise_seed=seed,
                    adc_cutoffs=cutoffs,
                    results_dir=results_dir,
                    store_per_plane_loss=True,
                ))

    chunks = list(_chunks(all_commands, n_jobs))
    n_total = len(all_commands)

    print(
        f"gradient_cutoff_sweep_15trk: {n_total} invocations "
        f"(15 full + 30 fraction tracks, 10 cutoffs each) → {len(chunks)} Slurm jobs"
    )

    for job_idx, chunk in enumerate(chunks):
        label = f"1d_grad_cutoff_{job_idx:03d}"
        if not print_sbatch_only:
            print(f"  {label}: {len(chunk)} invocations")
        s3df_submit_multi(
            chunk,
            job_label=label,
            time=time,
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
        )


# ── run_params.py helpers ─────────────────────────────────────────────────────

def make_run_params_command(
    output_dir: str = "$RESULTS_DIR/diffusion_sweep",
    config: str = "config/cubic_wireplane_config.yaml",
    store_per_pixel: bool = False,
    sobolev_max_pad: int = 128,
) -> str:
    parts = [
        "python src/analysis/sim_param_sweeps/run_params.py",
        f"--output-dir {output_dir}",
        f"--config {config}",
    ]
    if store_per_pixel:
        parts.append("--store-per-pixel-loss-and-grad")
        parts.append(f"--sobolev-max-pad {sobolev_max_pad}")
    return " ".join(parts)


def profile_run_params_diffusion_sweep(
    *,
    submit: bool = True,
    print_sbatch_only: bool = False,
    wandb_tags=None,
    output_dir: str = "$RESULTS_DIR/diffusion_sweep",
    time: str = "06:00:00",
    mem_gb: int = 64,
):
    """Forward-sweep run_params.py: diffusion_trans × diffusion_long grid (15 tracks + 10 points).

    Sweeps SWEEP_PARAMS defined in run_params.py (default 10×10 = 100 combos) across
    25 deposits in 20 chunks, with resumption. Signals and traces only — no pixel maps.
    """
    command = make_run_params_command(output_dir=output_dir)
    if not print_sbatch_only:
        print(command)
    s3df_submit(
        command,
        time=time,
        submit=submit,
        mem_gb=mem_gb,
        print_sbatch_command=print_sbatch_only,
    )


def profile_run_params_diffusion_sweep_pixel(
    *,
    submit: bool = True,
    print_sbatch_only: bool = False,
    wandb_tags=None,
    output_dir: str = "$RESULTS_DIR/diffusion_sweep_pixel",
    time: str = "10:00:00",
    mem_gb: int = 64,
):
    """Same as run_params_diffusion_sweep but also stores Sobolev per-pixel loss/grad maps.

    Adds --store-per-pixel-loss-and-grad, producing companion _grads.pkl files per chunk
    with windowed pixel_loss=(sim-GT)^2 and pixel_grad=d(Sobolev_loss)/d(sim[w,t]) maps.
    Longer wall time (10h) to account for the extra pixel_grad_fn calls per combo.
    """
    command = make_run_params_command(output_dir=output_dir, store_per_pixel=True)
    if not print_sbatch_only:
        print(command)
    s3df_submit(
        command,
        time=time,
        submit=submit,
        mem_gb=mem_gb,
        print_sbatch_command=print_sbatch_only,
    )


PROFILES = {
    "3_part_schedule": profile_3_part_schedule,
    "2_part_schedule": profile_2_part_schedule,
    "2_part_schedule_cosine_30k": profile_2_part_schedule_cosine_30k,
    "fine_nosched_bs1": profile_fine_nosched_bs1,
    "fine_nosched_bs1_mixed_xyz": profile_fine_nosched_bs1_mixed_xyz,
    "tracks50_mixed_cos30k_nosched": profile_tracks50_mixed_cos30k_nosched,
    "tracks50_mixed_cos30k_2phase": profile_tracks50_mixed_cos30k_2phase,
    "tracks12_mixed_cos30k_nosched": profile_tracks12_mixed_cos30k_nosched,
    "tracks12_mixed_cos30k_2phase": profile_tracks12_mixed_cos30k_2phase,
    "tracks12_mixed_cos30k_nosched_tol1e4_p2000": profile_tracks12_mixed_cos30k_nosched_tol1e4_p2000,
    "tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1": (
        profile_tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1
    ),
    "tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1_p500": (
        profile_tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1_p500
    ),
    "tracks12_mixed_cos30k_nosched_auto_clip1_noparamfreeze": (
        profile_tracks12_mixed_cos30k_nosched_auto_clip1_noparamfreeze
    ),
    "tracks12_mixed_cos30k_nosched_auto_clip1_noparamfreeze_no_vel": (
        profile_tracks12_mixed_cos30k_nosched_auto_clip1_noparamfreeze_no_vel
    ),
    "tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1_p500_ebs12": (
        profile_tracks12_mixed_cos30k_nosched_tol1e4_p2000_auto_clip1_p500_ebs12
    ),
    "tracks12_mixed_cos30k_auto_clip1_p500_no_vel": (
        profile_tracks12_mixed_cos30k_auto_clip1_p500_no_vel
    ),
    "tracks24_mixed_cos30k_nosched_tol1e4_p2000": profile_tracks24_mixed_cos30k_nosched_tol1e4_p2000,
    "fine_nosched_bs1_tol1e4_p2000": profile_fine_nosched_bs1_tol1e4_2k,
    "no_schedule_less_params": profile_no_schedule_less_params,
    "longitudinal_diffusion_only": profile_longitudinal_diffusion_only,
    "longitudinal_transverse_diffusion": profile_longitudinal_transverse_diffusion,
    "run_params_diffusion_sweep": profile_run_params_diffusion_sweep,
    "run_params_diffusion_sweep_pixel": profile_run_params_diffusion_sweep_pixel,
    "timing_study_diag50mev": profile_timing_study_diag50mev,
    "timing_study_cont": profile_timing_study_cont,
    "15_Tracks_Adam_default_BS1": profile_15_Tracks_Adam_default_BS1,
    "15_Tracks_Adam_default_BS1_AutoMultipliers": profile_15_Tracks_Adam_default_BS1_AutoMultipliers,
    "15_Tracks_Adam_default_BS15": profile_15_Tracks_Adam_default_BS15,
    "15_Tracks_Adam_default_BS15_coarse1mm": profile_15_Tracks_Adam_default_BS15_coarse1mm,
    "15_Tracks_Adam_default_BS15_coarse1mm_LR1e3": profile_15_Tracks_Adam_default_BS15_coarse1mm_LR1e3,
    "15_Tracks_Adam_default_BS15_coarse1mm_LR1e3_NoVelocity": profile_15_Tracks_Adam_default_BS15_coarse1mm_LR1e3_NoVelocity,
    "15_Tracks_Newton_recombAlpha_velocity_BS1": profile_15_Tracks_Newton_recombAlpha_velocity_BS1,
    "15_Tracks_Newton_diffRecomb_BS1": profile_15_Tracks_Newton_diffRecomb_BS1,
    "15_Tracks_Newton_allParams_BS1": profile_15_Tracks_Newton_allParams_BS1,
    "15_Tracks_Newton_allParams_BS1_Damping1p0": profile_15_Tracks_Newton_allParams_BS1_Damping1p0,
    "15_Tracks_Newton_allParams_BS1_DampClipSweep": profile_15_Tracks_Newton_allParams_BS1_DampClipSweep,
    "15_Tracks_Newton_ContinueFrom_eizhqsj0": profile_15_Tracks_Newton_ContinueFrom_eizhqsj0,
    "15_Tracks_Adam_Beta2Sweep_NoClip": profile_15_Tracks_Adam_Beta2Sweep_NoClip,
    "15_Tracks_Adam_default_BS15_LR4e4": profile_15_Tracks_Adam_default_BS15_LR4e4,
    "15_Tracks_Newton_ContinueFrom_eizhqsj0_step1k": profile_15_Tracks_Newton_ContinueFrom_eizhqsj0_step1k,
    "Adam_Noise_20260511": profile_15Trk_Adam_Noise_20260511,
    "Adam_NoiseSeedSweep_3k": profile_15Trk_Adam_NoiseSeedSweep_3k,
    "Adam_NoiseSeedSweep_3k_GT2": profile_15Trk_Adam_NoiseSeedSweep_3k_GT2,
    "Adam_NoiseSeedSweep_3k_GT3": profile_15Trk_Adam_NoiseSeedSweep_3k_GT3,
    "Adam_NoiseSeedSweep_3k_NoDiff": profile_15Trk_Adam_NoiseSeedSweep_3k_NoDiff,
    "Adam_NoiseSeedSweep_3k_GT2_NoDiff": profile_15Trk_Adam_NoiseSeedSweep_3k_GT2_NoDiff,
    "Adam_NoiseSeedSweep_3k_NoDiffLifetime": profile_15Trk_Adam_NoiseSeedSweep_3k_NoDiffLifetime,
    "Adam_NoiseCutoffDiffusion_3k": profile_15Trk_Adam_NoiseCutoffDiffusion_3k,
    "Adam_NoiseCutoffDiffusion_3k_LargeSpread": profile_15Trk_Adam_NoiseCutoffDiffusion_3k_LargeSpread,
    "Adam_NoiseCutoff25_4Trk_LargeSpread": profile_Adam_NoiseCutoff25_4Trk_LargeSpread,
    "Adam_NoiseCutoff25_DebugTracks_3k": profile_Adam_NoiseCutoff25_DebugTracks_3k,
    "Adam_NoiseCutoff25_DebugTracks_3k_CollectionOnly": profile_Adam_NoiseCutoff25_DebugTracks_3k_CollectionOnly,
    "Adam_NoiseCutoff25_DebugTracks_3k_CollectionOnly_3trk": profile_Adam_NoiseCutoff25_DebugTracks_3k_CollectionOnly_3trk,
    "Adam_NoiseCutoff50_DebugTracks_3k_3trk_BothPlanes":    profile_Adam_NoiseCutoff50_DebugTracks_3k_3trk_BothPlanes,
    "Adam_NoiseCutoff25_DebugTracks_3k_CollectionOnly_3trk_MoreCutoffs": profile_Adam_NoiseCutoff25_DebugTracks_3k_CollectionOnly_3trk_MoreCutoffs,
    "Adam_NoiseCutoff25_DebugTracks_3k_3trk": profile_Adam_NoiseCutoff25_DebugTracks_3k_3trk,
    "Adam_NoiseCutoff25_DebugTracks_3k_TransLong_Chain": profile_Adam_NoiseCutoff25_DebugTracks_3k_TransLong_Chain,
    "Adam_NoiseCutoff25_DebugTracks_2phase": profile_Adam_NoiseCutoff25_DebugTracks_2phase,
    "gradient_cutoff_sweep_15trk": profile_gradient_cutoff_sweep_15trk,
    "Adam_NoiseSeedSweep_3k_GT2_NoDiffLifetime": profile_15Trk_Adam_NoiseSeedSweep_3k_GT2_NoDiffLifetime,
    "Adam_NoiseSeedSweep_3k_GT3_NoDiff": profile_15Trk_Adam_NoiseSeedSweep_3k_GT3_NoDiff,
    "Adam_NoiseSeedSweep_3k_GT3_NoDiffLifetime": profile_15Trk_Adam_NoiseSeedSweep_3k_GT3_NoDiffLifetime,
    "Adam_NoiseSeedSweep_3k_0p1mm_step_GT": profile_15Trk_Adam_NoiseSeedSweep_3k_0p1mm_step_GT,
    "Adam_NoiseSeedSweep_3k_0p1mm_step_GT_and_sim": profile_15Trk_Adam_NoiseSeedSweep_3k_0p1mm_step_GT_and_sim,
    "Adam_NoiseSeedSweep_3k_Cont_Newton": profile_Adam_NoiseSeedSweep_3k_Cont_Newton,
    "1d_Grad_diffusion_debug": profile_1d_Grad_diffusion_debug,
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
        "--skip-complete",
        action="store_true",
        help="Skip (seed, noise) combinations that already have a complete result pkl. "
             "Checks $RESULTS_DIR on the current machine, so run this on S3DF where $RESULTS_DIR is set.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="For each (seed, noise) combination: print which pkl was checked and why it is "
             "complete or incomplete. Does not submit anything on its own; combine with "
             "--skip-complete to also suppress already-done jobs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output pkl files before submitting (profiles that support it).",
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
        profile_fn = PROFILES[args.profile]
        import inspect as _inspect
        extra = {}
        if "skip_complete" in _inspect.signature(profile_fn).parameters:
            extra["skip_complete"] = args.skip_complete
        if "verbose" in _inspect.signature(profile_fn).parameters:
            extra["verbose"] = args.verbose
        if "overwrite" in _inspect.signature(profile_fn).parameters:
            extra["overwrite"] = args.overwrite
        profile_fn(
            submit=not args.print_commands,
            print_sbatch_only=args.print_commands,
            wandb_tags=wandb_tags,
            **extra,
        )
    else:
        parser.error("Provide a profile or use --restart-preempted RESULTS_DIR.")
