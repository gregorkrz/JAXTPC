#!/usr/bin/env python3
"""
Loss landscape sweeps: diffusion coefficients at varying drift distances.

Usage
-----
  python src/jobs/submit_jobs_loss_studies.py <profile>
  python src/jobs/submit_jobs_loss_studies.py <profile> --submit
  python src/jobs/submit_jobs_loss_studies.py <profile> --print-sbatch-command

Available profiles
------------------
  diffusion_startx_study              D_trans / D_long loss vs. start-x position (Muons 4,5,10,12)
  diffusion_startx_study_no_wire_response
                                      Same as startx study but delta kernels (no wire response);
                                      all 4 tracks x 10 x-positions (same grid as startx study)
  diffusion_angle_study               D_trans / D_long loss vs. track angle (400 MeV, x=1900 mm)
  diffusion_angle_study_no_wire_response
                                      Same as angle study but delta kernels (no wire response);
                                      full angle grid (same as angle study)
  diffusion_angle_theta_alpha         D_trans / D_long loss vs. (theta, alpha) grid (400 MeV, x=1900 mm)
  diffusion_angle_theta_alpha_no_wire_response
                                      Same but delta kernels; full 5x5 (theta, alpha) grid
  diffusion_angle_theta_alpha_extended
                                      Same (theta, alpha) study, extended to angles > 20 deg;
                                      56 new combos (9x9 grid minus the existing 5x5),
                                      written into the same diffusion_angle_theta_alpha dirs
  diffusion_angle_theta_alpha_extended_no_wire_response
                                      Same but delta kernels
  diffusion_angle_pivot_study         D_trans / D_long loss vs. angle, pivot at x=1000 mm
  diffusion_angle_pivot_study_no_wire_response
                                      Same but delta kernels; full angle grid (same as pivot study)
  diffusion_angle_pivot_theta_alpha   D_trans / D_long loss vs. (theta, alpha) grid, pivot at x=1000 mm
  diffusion_angle_pivot_theta_alpha_no_wire_response
                                      Same but delta kernels; full 5x5 (theta, alpha) grid
  diffusion_angle_pivot_theta_alpha_extended
                                      Same pivot (theta, alpha) study, extended to angles > 20 deg
                                      (25,30,35,40,45,50); 96 new combos (11x11 grid minus the
                                      existing 5x5), written into the same
                                      diffusion_angle_pivot_theta_alpha dirs. ADC cutoffs limited
                                      to [0, 50] for this study; bundled into 10 sbatch jobs total
                                      (noisy + no-noise commands for ~10 combos per job)
  diffusion_angle_pivot_theta_alpha_extended_no_wire_response
                                      Same but delta kernels
"""

import math
import os
import sys
import time as _time

sys.path.insert(0, os.path.dirname(__file__))
from submit_jobs import make_gradient_command, s3df_submit_multi

_RESULTS_DIR: str = os.environ.get("RESULTS_DIR", "results")


def _expected_pkl_path(results_dir, param, track_name, noise_seed, adc_cutoff,
                       loss_name="sobolev_loss_geomean_log1p", N=100, range_frac=0.2,
                       noise_scale=1.0):
    """Reconstruct the output path that 1d_gradients.py auto_output_path() would produce."""
    range_tag   = f"_range{range_frac:.3g}".replace(".", "p")
    seed_suffix = f"_seed{noise_seed}" if noise_seed != 42 else ""
    noise_tag   = f"_noise{noise_scale:.3g}".replace(".", "p") + seed_suffix
    cutoff_tag  = f"_cutoff{adc_cutoff:.3g}".replace(".", "p") if adc_cutoff > 0.0 else ""
    fname = (f"{loss_name}_N{N}{range_tag}_{param}"
             f"_{track_name}{noise_tag}{cutoff_tag}_perplane.pkl")
    resolved = results_dir.replace("$RESULTS_DIR", _RESULTS_DIR)
    return os.path.join(resolved, fname)


def _check_completions(results_dir, invocations, adc_cutoffs):
    """Check completion for a list of (label, param, track_name, seed) invocations.

    Prints a summary and returns (n_complete, n_total, incomplete_labels).
    An invocation is complete when all per-adc-cutoff pkl files exist.

    Lists the results directory once and checks against a filename set to avoid
    per-file stat calls on network filesystems.
    """
    resolved = results_dir.replace("$RESULTS_DIR", _RESULTS_DIR)
    try:
        existing = set(os.listdir(resolved))
    except FileNotFoundError:
        existing = set()

    n_total = len(invocations)
    incomplete = []
    for label, param, track_name, seed in invocations:
        done = all(
            os.path.basename(_expected_pkl_path(results_dir, param, track_name, seed, ac)) in existing
            for ac in adc_cutoffs
        )
        if not done:
            incomplete.append(label)

    n_complete = n_total - len(incomplete)
    pct = 100.0 * n_complete / n_total if n_total else 0.0
    print(f"  {n_complete}/{n_total} invocations complete ({pct:.1f}%).")
    if incomplete:
        print(f"  {len(incomplete)} still need to run:")
        max_listed = 50
        for label in incomplete[:max_listed]:
            print(f"    {label}")
        if len(incomplete) > max_listed:
            print(f"    ... and {len(incomplete) - max_listed} more")
    return n_complete, n_total, incomplete


def _fmt_seconds(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m{s % 60:02d}s"


def _log_progress(step: int, total: int, label: str, t_start: float) -> None:
    elapsed = _time.time() - t_start
    if step > 1 and elapsed > 0:
        avg = elapsed / (step - 1)
        eta = avg * (total - step + 1)
        suffix = f"  (avg {avg:.1f}s/job, ETA ~{_fmt_seconds(eta)})"
    else:
        suffix = ""
    print(f"  [{step}/{total}] {label}{suffix}")


def _split_evenly(seq, n_chunks):
    """Split seq into n_chunks contiguous, near-equal-size pieces (sizes differ by <=1)."""
    n = len(seq)
    base, rem = divmod(n, n_chunks)
    chunks = []
    idx = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        if size:
            chunks.append(seq[idx:idx + size])
        idx += size
    return chunks


# ---------------------------------------------------------------------------
# Diffusion start-x study
# ---------------------------------------------------------------------------
# Four 100 MeV muon tracks (4, 5, 10, 12) with y set to 0 and z kept from the
# original boundary-track ensemble.  Each track is placed at 5 x positions
# spanning the detector depth (2000→0 mm for west-volume tracks, -2000→0 mm
# for east-volume tracks).
#
# Each entry: (base_name, direction_str, energy_mev, z_mm, x_positions_mm)
# ---------------------------------------------------------------------------
_DIFFUSION_STARTX_TRACKS = [
    # West-volume tracks (dx < 0, entering from x = +2160)
    ("Muon5_100MeV",  "-0.448568523,-0.712616910,0.539410252",  100,  2019.642044116,  [2000, 1900, 1800, 1750, 1700, 1600, 1500, 1000, 500, 0]),
    ("Muon12_100MeV", "-0.553810025,-0.123483953,-0.823435589", 100,  -273.380878516,  [2000, 1900, 1800, 1750, 1700, 1600, 1500, 1000, 500, 0]),
    # East-volume tracks (Muon4 dx < 0 from cathode; Muon10 dx > 0 from x = -2160)
    ("Muon4_100MeV",  "-0.694627880,0.476880059,0.538588450",   100,   0.0,           [-2000, -1900, -1800, -1750, -1700, -1600, -1500, -1000, -500, 0]),
    ("Muon10_100MeV", "0.754859526,-0.437194999,0.488924973",   100, -1556.076968089, [-2000, -1900, -1800, -1750, -1700, -1600, -1500, -1000, -500, 0]),
]


def submit_diffusion_startx_study(
    *,
    submit=True,
    print_sbatch_only=False,
    check_complete=False,
    no_noise_only=False,
    overwrite=False,
    chain=False,
    time="08:00:00",
):
    """1D loss landscape: diffusion coefficients at varying x start positions.

    Tracks    : Muon4, Muon5, Muon10, Muon12 (100 MeV) at 5 start-x positions each
    Params    : diffusion_trans_cm2_us, diffusion_long_cm2_us
    Range     : ±20% of GT (N=100, 201 sweep points covering [0.8, 1.2] × GT)
    ADC cuts  : [0, 5, 10, 20, 50] — separate pkl per cutoff (auto-named by 1d_gradients.py)
    FT cutoff : 0 (default)
    Noise     : scale=1.0, seeds 0–99 (100 seeds)  +  scale=0 single no-noise job
    step_size : 1 mm, max_deposits=5000

    Structure : 20 SLURM jobs (one per track × start-x) for the noisy runs.
                + 20 no-noise jobs (2 commands each: 2 params × 1 seed).
    Output    : $RESULTS_DIR/1d_gradients/diffusion_startx_study/
                $RESULTS_DIR/1d_gradients/diffusion_startx_study_nonoise/
    """
    results_dir    = "$RESULTS_DIR/1d_gradients/diffusion_startx_study"
    results_dir_nn = "$RESULTS_DIR/1d_gradients/diffusion_startx_study_nonoise"
    params      = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
    adc_cutoffs = [0, 5, 10, 20, 50]
    seeds       = list(range(100))  # 0–99

    if check_complete:
        invocations = [
            (f"{base_name}_startx_{start_x}_stepsize_1mm  param={param}  seed={seed}",
             param,
             f"{base_name}_startx_{start_x}_stepsize_1mm",
             seed)
            for base_name, _, _, _, x_positions in _DIFFUSION_STARTX_TRACKS
            for start_x in x_positions
            for param in params
            for seed in seeds
        ]
        print(f"diffusion_startx_study — checking {len(invocations)} invocations "
              f"in {results_dir.replace('$RESULTS_DIR', _RESULTS_DIR)}")
        _check_completions(results_dir, invocations, adc_cutoffs)
        return

    prev_id = None
    step = 0
    n_total =sum(len(xp) for _, _, _, _, xp in _DIFFUSION_STARTX_TRACKS)
    t_start = _time.time()
    for base_name, direction, energy, z, x_positions in _DIFFUSION_STARTX_TRACKS:
        for start_x in x_positions:
            track_name = f"{base_name}_startx_{start_x}_stepsize_1mm"
            track_spec = f"{track_name}:{direction}:{energy}:{start_x},0,{z}"
            if not print_sbatch_only:
                step += 1
                _log_progress(step, n_total, track_name, t_start)

            if not no_noise_only:
                commands = [
                    make_gradient_command(
                        param=param,
                        tracks=track_spec,
                        N=100,
                        range_frac=0.2,
                        noise_scale=1.0,
                        noise_seeds=seeds,
                        adc_cutoffs=adc_cutoffs,
                        results_dir=results_dir,
                        step_size=1.0,
                        max_deposits=5000,
                        store_per_plane_loss=True,
                        overwrite=overwrite,
                    )
                    for param in params
                ]
                label = f"loss_diff_startx_{track_name}"
                if not print_sbatch_only:
                    print(f"  {label}: {len(commands)} invocations")
                prev_id = s3df_submit_multi(
                    commands,
                    job_label=label,
                    time=time,
                    submit=submit,
                    mem_gb=64,
                    print_sbatch_command=print_sbatch_only,
                    log_progress=True,
                    dependency=prev_id if chain else None,
                )

            # No-noise job (deterministic: one run per param suffices)
            nn_commands = [
                make_gradient_command(
                    param=param,
                    tracks=track_spec,
                    N=100,
                    range_frac=0.2,
                    noise_scale=0.0,
                    noise_seed=0,
                    adc_cutoffs=adc_cutoffs,
                    results_dir=results_dir_nn,
                    step_size=1.0,
                    max_deposits=5000,
                    store_per_plane_loss=True,
                    overwrite=overwrite,
                )
                for param in params
            ]
            nn_label = f"loss_diff_startx_nonoise_{track_name}"
            if not print_sbatch_only:
                print(f"  {nn_label}: {len(nn_commands)} invocations (no noise)")
            prev_id = s3df_submit_multi(
                nn_commands,
                job_label=nn_label,
                time="01:00:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                log_progress=True,
                dependency=prev_id if chain else None,
            )


def submit_diffusion_startx_study_no_wire_response(
    *,
    submit=True,
    print_sbatch_only=False,
    check_complete=False,
    no_noise_only=False,
    overwrite=False,
    chain=False,
    time="08:00:00",
):
    """1D loss landscape: diffusion at varying x, delta kernels (no wire response).

    Tracks    : Muon4, Muon5, Muon10, Muon12 (100 MeV) at 10 x positions each (same grid as startx study)
    Params    : diffusion_trans_cm2_us, diffusion_long_cm2_us
    Range     : ±20% of GT (N=100)
    ADC cuts  : [0, 5, 10, 20, 50]
    Noise     : scale=1.0, seeds 0–49  +  scale=0 no-noise job
    Wire resp : disabled (--no-wire-response)

    Output    : $RESULTS_DIR/1d_gradients/diffusion_startx_study_no_wire_response/
                $RESULTS_DIR/1d_gradients/diffusion_startx_study_no_wire_response_nonoise/
    """
    results_dir    = "$RESULTS_DIR/1d_gradients/diffusion_startx_study_no_wire_response"
    results_dir_nn = "$RESULTS_DIR/1d_gradients/diffusion_startx_study_no_wire_response_nonoise"
    params      = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
    adc_cutoffs = [0, 5, 10, 20, 50]
    seeds       = list(range(50))

    if check_complete:
        invocations = [
            (f"{base_name}_startx_{start_x}_stepsize_1mm  param={param}  seed={seed}",
             param,
             f"{base_name}_startx_{start_x}_stepsize_1mm",
             seed)
            for base_name, _, _, _, x_positions in _DIFFUSION_STARTX_TRACKS
            for start_x in x_positions
            for param in params
            for seed in seeds
        ]
        print(f"diffusion_startx_study_no_wire_response — checking {len(invocations)} invocations "
              f"in {results_dir.replace('$RESULTS_DIR', _RESULTS_DIR)}")
        _check_completions(results_dir, invocations, adc_cutoffs)
        return

    prev_id = None
    step = 0
    n_total =sum(len(xp) for _, _, _, _, xp in _DIFFUSION_STARTX_TRACKS)
    t_start = _time.time()
    for base_name, direction, energy, z, x_positions in _DIFFUSION_STARTX_TRACKS:
        for start_x in x_positions:
            track_name = f"{base_name}_startx_{start_x}_stepsize_1mm"
            track_spec = f"{track_name}:{direction}:{energy}:{start_x},0,{z}"
            if not print_sbatch_only:
                step += 1
                _log_progress(step, n_total, track_name, t_start)

            if not no_noise_only:
                commands = [
                    make_gradient_command(
                        param=param,
                        tracks=track_spec,
                        N=100,
                        range_frac=0.2,
                        noise_scale=1.0,
                        noise_seeds=seeds,
                        adc_cutoffs=adc_cutoffs,
                        results_dir=results_dir,
                        step_size=1.0,
                        max_deposits=5000,
                        store_per_plane_loss=True,
                        no_wire_response=True,
                        overwrite=overwrite,
                    )
                    for param in params
                ]
                label = f"loss_diff_startx_nwr_{track_name}"
                if not print_sbatch_only:
                    print(f"  {label}: {len(commands)} invocations")
                prev_id = s3df_submit_multi(
                    commands,
                    job_label=label,
                    time=time,
                    submit=submit,
                    mem_gb=64,
                    print_sbatch_command=print_sbatch_only,
                    log_progress=True,
                    dependency=prev_id if chain else None,
                )

            nn_commands = [
                make_gradient_command(
                    param=param,
                    tracks=track_spec,
                    N=100,
                    range_frac=0.2,
                    noise_scale=0.0,
                    noise_seed=0,
                    adc_cutoffs=adc_cutoffs,
                    results_dir=results_dir_nn,
                    step_size=1.0,
                    max_deposits=5000,
                    store_per_plane_loss=True,
                    no_wire_response=True,
                    overwrite=overwrite,
                )
                for param in params
            ]
            nn_label = f"loss_diff_startx_nwr_nonoise_{track_name}"
            if not print_sbatch_only:
                print(f"  {nn_label}: {len(nn_commands)} invocations (no noise)")
            prev_id = s3df_submit_multi(
                nn_commands,
                job_label=nn_label,
                time="01:00:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                log_progress=True,
                dependency=prev_id if chain else None,
            )


# ---------------------------------------------------------------------------
# Diffusion angle study
# ---------------------------------------------------------------------------
# Single 400 MeV muon starting at (1900, 0, 0) mm in the west volume.
# The direction is rotated by theta degrees from the -x axis in the XY plane:
#   dx = -cos(theta),  dy = sin(theta),  dz = 0
# theta sweeps -90° to +90° in 10 equally-spaced steps (step = 20°).
# ---------------------------------------------------------------------------
_ANGLE_THETAS   = sorted(set(range(-90, 91, 20)) | {25, 15, 5, -5, -15, -25})  # [-90, -70, ..., 70, 90] + fine points near 0
_ANGLE_START_X  = 1900   # mm
_ANGLE_ENERGY   = 400    # MeV


def submit_diffusion_angle_study(
    *,
    submit=True,
    print_sbatch_only=False,
    check_complete=False,
    no_noise_only=False,
    overwrite=False,
    chain=False,
    time="08:00:00",
):
    """1D loss landscape: diffusion coefficients vs. track angle.

    Track     : 400 MeV muon at (1900, 0, 0) mm, rotated by theta in the XY plane
    Angles    : -90° to +90° in steps (every 20°) + fine points near 0
    Params    : diffusion_trans_cm2_us, diffusion_long_cm2_us
    Range     : ±20% of GT (N=100, 201 sweep points)
    ADC cuts  : [0, 5, 10, 20, 50]
    Noise     : scale=1.0, seeds 0–49 (50 seeds)  +  scale=0 single no-noise job
    step_size : 1 mm, max_deposits=5000

    Output    : $RESULTS_DIR/1d_gradients/diffusion_angle_study/
                $RESULTS_DIR/1d_gradients/diffusion_angle_study_nonoise/
    """
    results_dir    = "$RESULTS_DIR/1d_gradients/diffusion_angle_study"
    results_dir_nn = "$RESULTS_DIR/1d_gradients/diffusion_angle_study_nonoise"
    params      = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
    adc_cutoffs = [0, 5, 10, 20, 50]
    seeds       = list(range(50))  # 0–49

    if check_complete:
        invocations = [
            (f"Muon_400MeV_theta_{theta_deg}_stepsize_1mm  param={param}  seed={seed}",
             param,
             f"Muon_400MeV_theta_{theta_deg}_stepsize_1mm",
             seed)
            for theta_deg in _ANGLE_THETAS
            for param in params
            for seed in seeds
        ]
        print(f"diffusion_angle_study — checking {len(invocations)} invocations "
              f"in {results_dir.replace('$RESULTS_DIR', _RESULTS_DIR)}")
        _check_completions(results_dir, invocations, adc_cutoffs)
        return

    prev_id = None
    step = 0
    n_total =len(_ANGLE_THETAS)
    t_start = _time.time()
    for theta_deg in _ANGLE_THETAS:
        theta_rad = math.radians(theta_deg)
        dx = round(-math.cos(theta_rad), 9)
        dy = round( math.sin(theta_rad), 9)
        dz = 0.0
        direction = f"{dx:.9f},{dy:.9f},{dz:.1f}"

        track_name = f"Muon_400MeV_theta_{theta_deg}_stepsize_1mm"
        track_spec = (f"{track_name}:{direction}:{_ANGLE_ENERGY}"
                      f":{_ANGLE_START_X},0,0")
        if not print_sbatch_only:
            step += 1
            _log_progress(step, n_total, track_name, t_start)

        if not no_noise_only:
            commands = [
                make_gradient_command(
                    param=param,
                    tracks=track_spec,
                    N=100,
                    range_frac=0.2,
                    noise_scale=1.0,
                    noise_seeds=seeds,
                    adc_cutoffs=adc_cutoffs,
                    results_dir=results_dir,
                    step_size=1.0,
                    max_deposits=5000,
                    store_per_plane_loss=True,
                    overwrite=overwrite,
                )
                for param in params
            ]
            label = f"loss_diff_angle_{track_name}"
            if not print_sbatch_only:
                print(f"  {label}: {len(commands)} invocations")
            prev_id = s3df_submit_multi(
                commands,
                job_label=label,
                time=time,
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                log_progress=True,
                dependency=prev_id if chain else None,
            )

        # No-noise job
        nn_commands = [
            make_gradient_command(
                param=param,
                tracks=track_spec,
                N=100,
                range_frac=0.2,
                noise_scale=0.0,
                noise_seed=0,
                adc_cutoffs=adc_cutoffs,
                results_dir=results_dir_nn,
                step_size=1.0,
                max_deposits=5000,
                store_per_plane_loss=True,
                overwrite=overwrite,
            )
            for param in params
        ]
        nn_label = f"loss_diff_angle_nonoise_{track_name}"
        if not print_sbatch_only:
            print(f"  {nn_label}: {len(nn_commands)} invocations (no noise)")
        prev_id = s3df_submit_multi(
            nn_commands,
            job_label=nn_label,
            time="01:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            log_progress=True,
            dependency=prev_id if chain else None,
        )


def submit_diffusion_angle_study_no_wire_response(
    *,
    submit=True,
    print_sbatch_only=False,
    check_complete=False,
    no_noise_only=False,
    overwrite=False,
    chain=False,
    time="08:00:00",
):
    """1D loss landscape: diffusion vs. track angle, delta kernels (no wire response).

    Track     : 400 MeV muon at (1900, 0, 0) mm
    Angles    : same full angle grid as diffusion_angle_study (_ANGLE_THETAS)
    Params    : diffusion_trans_cm2_us, diffusion_long_cm2_us
    Range     : ±20% of GT (N=100)
    ADC cuts  : [0, 5, 10, 20, 50]
    Noise     : scale=1.0, seeds 0–49  +  scale=0 no-noise job
    Wire resp : disabled (--no-wire-response)

    Output    : $RESULTS_DIR/1d_gradients/diffusion_angle_study_no_wire_response/
                $RESULTS_DIR/1d_gradients/diffusion_angle_study_no_wire_response_nonoise/
    """
    results_dir    = "$RESULTS_DIR/1d_gradients/diffusion_angle_study_no_wire_response"
    results_dir_nn = "$RESULTS_DIR/1d_gradients/diffusion_angle_study_no_wire_response_nonoise"
    params      = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
    adc_cutoffs = [0, 5, 10, 20, 50]
    seeds       = list(range(50))

    if check_complete:
        invocations = [
            (f"Muon_400MeV_theta_{theta_deg}_stepsize_1mm  param={param}  seed={seed}",
             param,
             f"Muon_400MeV_theta_{theta_deg}_stepsize_1mm",
             seed)
            for theta_deg in _ANGLE_THETAS
            for param in params
            for seed in seeds
        ]
        print(f"diffusion_angle_study_no_wire_response — checking {len(invocations)} invocations "
              f"in {results_dir.replace('$RESULTS_DIR', _RESULTS_DIR)}")
        _check_completions(results_dir, invocations, adc_cutoffs)
        return

    prev_id = None
    step = 0
    n_total =len(_ANGLE_THETAS)
    t_start = _time.time()
    for theta_deg in _ANGLE_THETAS:
        theta_rad = math.radians(theta_deg)
        dx = round(-math.cos(theta_rad), 9)
        dy = round( math.sin(theta_rad), 9)
        dz = 0.0
        direction = f"{dx:.9f},{dy:.9f},{dz:.1f}"

        track_name = f"Muon_400MeV_theta_{theta_deg}_stepsize_1mm"
        track_spec = (f"{track_name}:{direction}:{_ANGLE_ENERGY}"
                      f":{_ANGLE_START_X},0,0")
        if not print_sbatch_only:
            step += 1
            _log_progress(step, n_total, track_name, t_start)

        if not no_noise_only:
            commands = [
                make_gradient_command(
                    param=param,
                    tracks=track_spec,
                    N=100,
                    range_frac=0.2,
                    noise_scale=1.0,
                    noise_seeds=seeds,
                    adc_cutoffs=adc_cutoffs,
                    results_dir=results_dir,
                    step_size=1.0,
                    max_deposits=5000,
                    store_per_plane_loss=True,
                    no_wire_response=True,
                    overwrite=overwrite,
                )
                for param in params
            ]
            label = f"loss_diff_angle_nwr_{track_name}"
            if not print_sbatch_only:
                print(f"  {label}: {len(commands)} invocations")
            prev_id = s3df_submit_multi(
                commands,
                job_label=label,
                time=time,
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                log_progress=True,
                dependency=prev_id if chain else None,
            )

        nn_commands = [
            make_gradient_command(
                param=param,
                tracks=track_spec,
                N=100,
                range_frac=0.2,
                noise_scale=0.0,
                noise_seed=0,
                adc_cutoffs=adc_cutoffs,
                results_dir=results_dir_nn,
                step_size=1.0,
                max_deposits=5000,
                store_per_plane_loss=True,
                no_wire_response=True,
                overwrite=overwrite,
            )
            for param in params
        ]
        nn_label = f"loss_diff_angle_nwr_nonoise_{track_name}"
        if not print_sbatch_only:
            print(f"  {nn_label}: {len(nn_commands)} invocations (no noise)")
        prev_id = s3df_submit_multi(
            nn_commands,
            job_label=nn_label,
            time="01:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            log_progress=True,
            dependency=prev_id if chain else None,
        )


def submit_diffusion_angle_pivot_study_no_wire_response(
    *,
    submit=True,
    print_sbatch_only=False,
    check_complete=False,
    no_noise_only=False,
    overwrite=False,
    chain=False,
    time="08:00:00",
):
    """1D loss landscape: diffusion vs. track angle, pivot at x=1000 mm, delta kernels (no wire response).

    Track     : 400 MeV muon, midpoint fixed at (1000, 0, 0) mm (west volume)
    Angles    : same full angle grid as diffusion_angle_pivot_study (_ANGLE_THETAS)
    Params    : diffusion_trans_cm2_us, diffusion_long_cm2_us
    Range     : ±20% of GT (N=100)
    ADC cuts  : [0, 5, 10, 20, 50]
    Noise     : scale=1.0, seeds 0–49  +  scale=0 no-noise job
    Wire resp : disabled (--no-wire-response)

    Output    : $RESULTS_DIR/1d_gradients/diffusion_angle_pivot_study_no_wire_response/
                $RESULTS_DIR/1d_gradients/diffusion_angle_pivot_study_no_wire_response_nonoise/
    """
    results_dir    = "$RESULTS_DIR/1d_gradients/diffusion_angle_pivot_study_no_wire_response"
    results_dir_nn = "$RESULTS_DIR/1d_gradients/diffusion_angle_pivot_study_no_wire_response_nonoise"
    params      = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
    adc_cutoffs = [0, 5, 10, 20, 50]
    seeds       = list(range(50))

    if check_complete:
        invocations = [
            (f"Muon_400MeV_theta_{theta_deg}_pivot_x1000_stepsize_1mm  param={param}  seed={seed}",
             param,
             f"Muon_400MeV_theta_{theta_deg}_pivot_x1000_stepsize_1mm",
             seed)
            for theta_deg in _ANGLE_THETAS
            for param in params
            for seed in seeds
        ]
        print(f"diffusion_angle_pivot_study_no_wire_response — checking {len(invocations)} invocations "
              f"in {results_dir.replace('$RESULTS_DIR', _RESULTS_DIR)}")
        _check_completions(results_dir, invocations, adc_cutoffs)
        return

    prev_id = None
    step = 0
    n_total =len(_ANGLE_THETAS)
    t_start = _time.time()
    for theta_deg in _ANGLE_THETAS:
        theta_rad = math.radians(theta_deg)
        dx = round(-math.cos(theta_rad), 9)
        dy = round( math.sin(theta_rad), 9)
        dz = 0.0
        direction = f"{dx:.9f},{dy:.9f},{dz:.1f}"

        start_x = round(_ANGLE_PIVOT_X_MM + _ANGLE_PIVOT_HALF_LEN_MM * math.cos(theta_rad), 3)
        start_y = round(-_ANGLE_PIVOT_HALF_LEN_MM * math.sin(theta_rad), 3)

        track_name = f"Muon_400MeV_theta_{theta_deg}_pivot_x1000_stepsize_1mm"
        track_spec = (f"{track_name}:{direction}:{_ANGLE_ENERGY}"
                      f":{start_x},{start_y},0")
        if not print_sbatch_only:
            step += 1
            _log_progress(step, n_total, track_name, t_start)

        if not no_noise_only:
            commands = [
                make_gradient_command(
                    param=param,
                    tracks=track_spec,
                    N=100,
                    range_frac=0.2,
                    noise_scale=1.0,
                    noise_seeds=seeds,
                    adc_cutoffs=adc_cutoffs,
                    results_dir=results_dir,
                    step_size=1.0,
                    max_deposits=5000,
                    store_per_plane_loss=True,
                    no_wire_response=True,
                    overwrite=overwrite,
                )
                for param in params
            ]
            label = f"loss_diff_pivot_nwr_{track_name}"
            if not print_sbatch_only:
                print(f"  {label}: {len(commands)} invocations")
            prev_id = s3df_submit_multi(
                commands,
                job_label=label,
                time=time,
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                log_progress=True,
                dependency=prev_id if chain else None,
            )

        nn_commands = [
            make_gradient_command(
                param=param,
                tracks=track_spec,
                N=100,
                range_frac=0.2,
                noise_scale=0.0,
                noise_seed=0,
                adc_cutoffs=adc_cutoffs,
                results_dir=results_dir_nn,
                step_size=1.0,
                max_deposits=5000,
                store_per_plane_loss=True,
                no_wire_response=True,
                overwrite=overwrite,
            )
            for param in params
        ]
        nn_label = f"loss_diff_pivot_nwr_nonoise_{track_name}"
        if not print_sbatch_only:
            print(f"  {nn_label}: {len(nn_commands)} invocations (no noise)")
        prev_id = s3df_submit_multi(
            nn_commands,
            job_label=nn_label,
            time="01:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            log_progress=True,
            dependency=prev_id if chain else None,
        )


# ---------------------------------------------------------------------------
# Diffusion (theta, alpha) grid study
# ---------------------------------------------------------------------------
# Single 400 MeV muon starting at (1900, 0, 0) mm in the west volume.
# theta = azimuthal angle in the XY plane from the -x axis (as in angle_study).
# alpha = "lift" angle from the XY plane toward +z (polar elevation).
#   dx = -cos(theta)*cos(alpha),  dy = sin(theta)*cos(alpha),  dz = sin(alpha)
# Both angles sweep 0°–20° in 5° steps → 5×5 = 25 grid points.
# ---------------------------------------------------------------------------
_TA_THETAS = list(range(0, 21, 5))   # [0, 5, 10, 15, 20]
_TA_ALPHAS = list(range(0, 21, 5))   # [0, 5, 10, 15, 20]

# Extended grid: adds 25°/30°/35°/40° to both axes (9x9 = 81 combos total).
# Only the 56 combos with theta>20° or alpha>20° are new (the 5x5=25 combos
# with both <=20° are already covered by the base theta_alpha study above).
# Written into the SAME results dirs as the base study so the plot's 9x9
# (theta, alpha) grid is fully populated.
_TA_THETAS_EXT = list(range(0, 41, 5))   # [0, 5, ..., 40]
_TA_ALPHAS_EXT = list(range(0, 41, 5))   # [0, 5, ..., 40]
_TA_EXTENDED_COMBOS = [
    (theta_deg, alpha_deg)
    for theta_deg in _TA_THETAS_EXT
    for alpha_deg in _TA_ALPHAS_EXT
    if theta_deg > 20 or alpha_deg > 20
]

# Pivot extended grid: adds 25/30/35/40/45/50 deg to both axes (11x11 = 121
# combos total). Only the 96 combos with theta>20 or alpha>20 are new (the
# 5x5=25 combos with both <=20 are already covered by
# diffusion_angle_pivot_theta_alpha). Written into the SAME results dirs as
# the base pivot study so the plot's 11x11 (theta, alpha) grid is fully
# populated.
_TA_THETAS_PIVOT_EXT = list(range(0, 51, 5))   # [0, 5, ..., 50]
_TA_ALPHAS_PIVOT_EXT = list(range(0, 51, 5))   # [0, 5, ..., 50]
_TA_PIVOT_EXTENDED_COMBOS = [
    (theta_deg, alpha_deg)
    for theta_deg in _TA_THETAS_PIVOT_EXT
    for alpha_deg in _TA_ALPHAS_PIVOT_EXT
    if theta_deg > 20 or alpha_deg > 20
]


def submit_diffusion_angle_theta_alpha_study(
    *,
    submit=True,
    print_sbatch_only=False,
    check_complete=False,
    no_noise_only=False,
    overwrite=False,
    chain=False,
    time="08:00:00",
):
    """1D loss landscape: diffusion coefficients vs. (theta, alpha) grid.

    Track     : 400 MeV muon at (1900, 0, 0) mm.
                theta — azimuthal angle in XY plane from -x axis (0°–20°, step 5°)
                alpha — lift angle from XY plane toward +z   (0°–20°, step 5°)
    Grid      : 5×5 = 25 (theta, alpha) combinations
    Params    : diffusion_trans_cm2_us, diffusion_long_cm2_us
    Range     : ±20% of GT (N=100, 201 sweep points)
    ADC cuts  : [0, 5, 10, 20, 50]
    Noise     : scale=1.0, seeds 0–49 (50 seeds)  +  scale=0 single no-noise job
    step_size : 1 mm, max_deposits=5000

    Output    : $RESULTS_DIR/1d_gradients/diffusion_angle_theta_alpha/
                $RESULTS_DIR/1d_gradients/diffusion_angle_theta_alpha_nonoise/
    """
    results_dir    = "$RESULTS_DIR/1d_gradients/diffusion_angle_theta_alpha"
    results_dir_nn = "$RESULTS_DIR/1d_gradients/diffusion_angle_theta_alpha_nonoise"
    params      = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
    adc_cutoffs = [0, 5, 10, 20, 50]
    seeds       = list(range(50))  # 0–49

    if check_complete:
        invocations = [
            (f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_stepsize_1mm  param={param}  seed={seed}",
             param,
             f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_stepsize_1mm",
             seed)
            for theta_deg in _TA_THETAS
            for alpha_deg in _TA_ALPHAS
            for param in params
            for seed in seeds
        ]
        print(f"diffusion_angle_theta_alpha — checking {len(invocations)} invocations "
              f"in {results_dir.replace('$RESULTS_DIR', _RESULTS_DIR)}")
        _check_completions(results_dir, invocations, adc_cutoffs)
        return

    prev_id = None
    step = 0
    n_total =len(_TA_THETAS) * len(_TA_ALPHAS)
    t_start = _time.time()
    for theta_deg in _TA_THETAS:
        for alpha_deg in _TA_ALPHAS:
            theta_rad = math.radians(theta_deg)
            alpha_rad = math.radians(alpha_deg)
            dx = round(-math.cos(theta_rad) * math.cos(alpha_rad), 9)
            dy = round( math.sin(theta_rad) * math.cos(alpha_rad), 9)
            dz = round( math.sin(alpha_rad), 9)
            direction = f"{dx:.9f},{dy:.9f},{dz:.9f}"

            track_name = f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_stepsize_1mm"
            track_spec = (f"{track_name}:{direction}:{_ANGLE_ENERGY}"
                          f":{_ANGLE_START_X},0,0")
            if not print_sbatch_only:
                step += 1
                _log_progress(step, n_total, track_name, t_start)

            if not no_noise_only:
                commands = [
                    make_gradient_command(
                        param=param,
                        tracks=track_spec,
                        N=100,
                        range_frac=0.2,
                        noise_scale=1.0,
                        noise_seeds=seeds,
                        adc_cutoffs=adc_cutoffs,
                        results_dir=results_dir,
                        step_size=1.0,
                        max_deposits=5000,
                        store_per_plane_loss=True,
                        overwrite=overwrite,
                    )
                    for param in params
                ]
                label = f"loss_diff_ta_{track_name}"
                if not print_sbatch_only:
                    print(f"  {label}: {len(commands)} invocations")
                prev_id = s3df_submit_multi(
                    commands,
                    job_label=label,
                    time=time,
                    submit=submit,
                    mem_gb=64,
                    print_sbatch_command=print_sbatch_only,
                    log_progress=True,
                    dependency=prev_id if chain else None,
                )

            # No-noise job
            nn_commands = [
                make_gradient_command(
                    param=param,
                    tracks=track_spec,
                    N=100,
                    range_frac=0.2,
                    noise_scale=0.0,
                    noise_seed=0,
                    adc_cutoffs=adc_cutoffs,
                    results_dir=results_dir_nn,
                    step_size=1.0,
                    max_deposits=5000,
                    store_per_plane_loss=True,
                    overwrite=overwrite,
                )
                for param in params
            ]
            nn_label = f"loss_diff_ta_nonoise_{track_name}"
            if not print_sbatch_only:
                print(f"  {nn_label}: {len(nn_commands)} invocations (no noise)")
            prev_id = s3df_submit_multi(
                nn_commands,
                job_label=nn_label,
                time="01:00:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                log_progress=True,
                dependency=prev_id if chain else None,
            )


def submit_diffusion_angle_theta_alpha_study_no_wire_response(
    *,
    submit=True,
    print_sbatch_only=False,
    check_complete=False,
    no_noise_only=False,
    overwrite=False,
    chain=False,
    time="08:00:00",
):
    """1D loss landscape: diffusion vs. (theta, alpha) grid, delta kernels (no wire response).

    Track     : 400 MeV muon at (1900, 0, 0) mm.
                theta — azimuthal angle in XY plane from -x axis (0°–20°, step 5°)
                alpha — lift angle from XY plane toward +z   (0°–20°, step 5°)
    Grid      : 5×5 = 25 (theta, alpha) combinations (same as full theta_alpha study)
    Params    : diffusion_trans_cm2_us, diffusion_long_cm2_us
    Range     : ±20% of GT (N=100)
    ADC cuts  : [0, 5, 10, 20, 50]
    Noise     : scale=1.0, seeds 0–49  +  scale=0 no-noise job
    Wire resp : disabled (--no-wire-response)

    Output    : $RESULTS_DIR/1d_gradients/diffusion_angle_theta_alpha_no_wire_response/
                $RESULTS_DIR/1d_gradients/diffusion_angle_theta_alpha_no_wire_response_nonoise/
    """
    results_dir    = "$RESULTS_DIR/1d_gradients/diffusion_angle_theta_alpha_no_wire_response"
    results_dir_nn = "$RESULTS_DIR/1d_gradients/diffusion_angle_theta_alpha_no_wire_response_nonoise"
    params      = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
    adc_cutoffs = [0, 5, 10, 20, 50]
    seeds       = list(range(50))

    if check_complete:
        invocations = [
            (f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_stepsize_1mm  param={param}  seed={seed}",
             param,
             f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_stepsize_1mm",
             seed)
            for theta_deg in _TA_THETAS
            for alpha_deg in _TA_ALPHAS
            for param in params
            for seed in seeds
        ]
        print(f"diffusion_angle_theta_alpha_no_wire_response — checking {len(invocations)} invocations "
              f"in {results_dir.replace('$RESULTS_DIR', _RESULTS_DIR)}")
        _check_completions(results_dir, invocations, adc_cutoffs)
        return

    prev_id = None
    step = 0
    n_total =len(_TA_THETAS) * len(_TA_ALPHAS)
    t_start = _time.time()
    for theta_deg in _TA_THETAS:
        for alpha_deg in _TA_ALPHAS:
            theta_rad = math.radians(theta_deg)
            alpha_rad = math.radians(alpha_deg)
            dx = round(-math.cos(theta_rad) * math.cos(alpha_rad), 9)
            dy = round( math.sin(theta_rad) * math.cos(alpha_rad), 9)
            dz = round( math.sin(alpha_rad), 9)
            direction = f"{dx:.9f},{dy:.9f},{dz:.9f}"

            track_name = f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_stepsize_1mm"
            track_spec = (f"{track_name}:{direction}:{_ANGLE_ENERGY}"
                          f":{_ANGLE_START_X},0,0")
            if not print_sbatch_only:
                step += 1
                _log_progress(step, n_total, track_name, t_start)

            if not no_noise_only:
                commands = [
                    make_gradient_command(
                        param=param,
                        tracks=track_spec,
                        N=100,
                        range_frac=0.2,
                        noise_scale=1.0,
                        noise_seeds=seeds,
                        adc_cutoffs=adc_cutoffs,
                        results_dir=results_dir,
                        step_size=1.0,
                        max_deposits=5000,
                        store_per_plane_loss=True,
                        no_wire_response=True,
                        overwrite=overwrite,
                    )
                    for param in params
                ]
                label = f"loss_diff_ta_nwr_{track_name}"
                if not print_sbatch_only:
                    print(f"  {label}: {len(commands)} invocations")
                prev_id = s3df_submit_multi(
                    commands,
                    job_label=label,
                    time=time,
                    submit=submit,
                    mem_gb=64,
                    print_sbatch_command=print_sbatch_only,
                    log_progress=True,
                    dependency=prev_id if chain else None,
                )

            nn_commands = [
                make_gradient_command(
                    param=param,
                    tracks=track_spec,
                    N=100,
                    range_frac=0.2,
                    noise_scale=0.0,
                    noise_seed=0,
                    adc_cutoffs=adc_cutoffs,
                    results_dir=results_dir_nn,
                    step_size=1.0,
                    max_deposits=5000,
                    store_per_plane_loss=True,
                    no_wire_response=True,
                    overwrite=overwrite,
                )
                for param in params
            ]
            nn_label = f"loss_diff_ta_nwr_nonoise_{track_name}"
            if not print_sbatch_only:
                print(f"  {nn_label}: {len(nn_commands)} invocations (no noise)")
            prev_id = s3df_submit_multi(
                nn_commands,
                job_label=nn_label,
                time="01:00:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                log_progress=True,
                dependency=prev_id if chain else None,
            )


def submit_diffusion_angle_theta_alpha_extended_study(
    *,
    submit=True,
    print_sbatch_only=False,
    check_complete=False,
    no_noise_only=False,
    overwrite=False,
    chain=False,
    time="08:00:00",
):
    """1D loss landscape: diffusion vs (theta, alpha), extended grid (angles > 20°).

    Track     : 400 MeV muon at (1900, 0, 0) mm (same as diffusion_angle_theta_alpha).
                theta, alpha each range 0°-40° in 5° steps (9x9 = 81 combos);
                only the 56 combos with theta>20° or alpha>20° are submitted
                here — the 5x5=25 combos with both <=20° are covered by
                diffusion_angle_theta_alpha.
    Params    : diffusion_trans_cm2_us, diffusion_long_cm2_us
    Range     : ±20% of GT (N=100, 201 sweep points)
    ADC cuts  : [0, 5, 10, 20, 50]
    Noise     : scale=1.0, seeds 0–49 (50 seeds)  +  scale=0 single no-noise job
    step_size : 1 mm, max_deposits=5000

    Output    : $RESULTS_DIR/1d_gradients/diffusion_angle_theta_alpha/  (same dir as base study)
                $RESULTS_DIR/1d_gradients/diffusion_angle_theta_alpha_nonoise/
    """
    results_dir    = "$RESULTS_DIR/1d_gradients/diffusion_angle_theta_alpha"
    results_dir_nn = "$RESULTS_DIR/1d_gradients/diffusion_angle_theta_alpha_nonoise"
    params      = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
    adc_cutoffs = [0, 5, 10, 20, 50]
    seeds       = list(range(50))  # 0–49

    if check_complete:
        invocations = [
            (f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_stepsize_1mm  param={param}  seed={seed}",
             param,
             f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_stepsize_1mm",
             seed)
            for theta_deg, alpha_deg in _TA_EXTENDED_COMBOS
            for param in params
            for seed in seeds
        ]
        print(f"diffusion_angle_theta_alpha_extended — checking {len(invocations)} invocations "
              f"in {results_dir.replace('$RESULTS_DIR', _RESULTS_DIR)}")
        _check_completions(results_dir, invocations, adc_cutoffs)
        return

    prev_id = None
    step = 0
    n_total = len(_TA_EXTENDED_COMBOS)
    t_start = _time.time()
    for theta_deg, alpha_deg in _TA_EXTENDED_COMBOS:
        theta_rad = math.radians(theta_deg)
        alpha_rad = math.radians(alpha_deg)
        dx = round(-math.cos(theta_rad) * math.cos(alpha_rad), 9)
        dy = round( math.sin(theta_rad) * math.cos(alpha_rad), 9)
        dz = round( math.sin(alpha_rad), 9)
        direction = f"{dx:.9f},{dy:.9f},{dz:.9f}"

        track_name = f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_stepsize_1mm"
        track_spec = (f"{track_name}:{direction}:{_ANGLE_ENERGY}"
                      f":{_ANGLE_START_X},0,0")
        if not print_sbatch_only:
            step += 1
            _log_progress(step, n_total, track_name, t_start)

        if not no_noise_only:
            commands = [
                make_gradient_command(
                    param=param,
                    tracks=track_spec,
                    N=100,
                    range_frac=0.2,
                    noise_scale=1.0,
                    noise_seeds=seeds,
                    adc_cutoffs=adc_cutoffs,
                    results_dir=results_dir,
                    step_size=1.0,
                    max_deposits=5000,
                    store_per_plane_loss=True,
                    overwrite=overwrite,
                )
                for param in params
            ]
            label = f"loss_diff_ta_ext_{track_name}"
            if not print_sbatch_only:
                print(f"  {label}: {len(commands)} invocations")
            prev_id = s3df_submit_multi(
                commands,
                job_label=label,
                time=time,
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                log_progress=True,
                dependency=prev_id if chain else None,
            )

        # No-noise job
        nn_commands = [
            make_gradient_command(
                param=param,
                tracks=track_spec,
                N=100,
                range_frac=0.2,
                noise_scale=0.0,
                noise_seed=0,
                adc_cutoffs=adc_cutoffs,
                results_dir=results_dir_nn,
                step_size=1.0,
                max_deposits=5000,
                store_per_plane_loss=True,
                overwrite=overwrite,
            )
            for param in params
        ]
        nn_label = f"loss_diff_ta_ext_nonoise_{track_name}"
        if not print_sbatch_only:
            print(f"  {nn_label}: {len(nn_commands)} invocations (no noise)")
        prev_id = s3df_submit_multi(
            nn_commands,
            job_label=nn_label,
            time="01:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            log_progress=True,
            dependency=prev_id if chain else None,
        )


def submit_diffusion_angle_theta_alpha_extended_study_no_wire_response(
    *,
    submit=True,
    print_sbatch_only=False,
    check_complete=False,
    no_noise_only=False,
    overwrite=False,
    chain=False,
    time="08:00:00",
):
    """1D loss landscape: diffusion vs (theta, alpha), extended grid (angles > 20°), delta kernels.

    Same grid and structure as submit_diffusion_angle_theta_alpha_extended_study,
    but with wire response disabled (--no-wire-response); writes into the same
    dirs as diffusion_angle_theta_alpha_no_wire_response.

    Output    : $RESULTS_DIR/1d_gradients/diffusion_angle_theta_alpha_no_wire_response/
                $RESULTS_DIR/1d_gradients/diffusion_angle_theta_alpha_no_wire_response_nonoise/
    """
    results_dir    = "$RESULTS_DIR/1d_gradients/diffusion_angle_theta_alpha_no_wire_response"
    results_dir_nn = "$RESULTS_DIR/1d_gradients/diffusion_angle_theta_alpha_no_wire_response_nonoise"
    params      = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
    adc_cutoffs = [0, 5, 10, 20, 50]
    seeds       = list(range(50))  # 0–49

    if check_complete:
        invocations = [
            (f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_stepsize_1mm  param={param}  seed={seed}",
             param,
             f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_stepsize_1mm",
             seed)
            for theta_deg, alpha_deg in _TA_EXTENDED_COMBOS
            for param in params
            for seed in seeds
        ]
        print(f"diffusion_angle_theta_alpha_extended_no_wire_response — checking {len(invocations)} invocations "
              f"in {results_dir.replace('$RESULTS_DIR', _RESULTS_DIR)}")
        _check_completions(results_dir, invocations, adc_cutoffs)
        return

    prev_id = None
    step = 0
    n_total = len(_TA_EXTENDED_COMBOS)
    t_start = _time.time()
    for theta_deg, alpha_deg in _TA_EXTENDED_COMBOS:
        theta_rad = math.radians(theta_deg)
        alpha_rad = math.radians(alpha_deg)
        dx = round(-math.cos(theta_rad) * math.cos(alpha_rad), 9)
        dy = round( math.sin(theta_rad) * math.cos(alpha_rad), 9)
        dz = round( math.sin(alpha_rad), 9)
        direction = f"{dx:.9f},{dy:.9f},{dz:.9f}"

        track_name = f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_stepsize_1mm"
        track_spec = (f"{track_name}:{direction}:{_ANGLE_ENERGY}"
                      f":{_ANGLE_START_X},0,0")
        if not print_sbatch_only:
            step += 1
            _log_progress(step, n_total, track_name, t_start)

        if not no_noise_only:
            commands = [
                make_gradient_command(
                    param=param,
                    tracks=track_spec,
                    N=100,
                    range_frac=0.2,
                    noise_scale=1.0,
                    noise_seeds=seeds,
                    adc_cutoffs=adc_cutoffs,
                    results_dir=results_dir,
                    step_size=1.0,
                    max_deposits=5000,
                    store_per_plane_loss=True,
                    no_wire_response=True,
                    overwrite=overwrite,
                )
                for param in params
            ]
            label = f"loss_diff_ta_ext_nwr_{track_name}"
            if not print_sbatch_only:
                print(f"  {label}: {len(commands)} invocations")
            prev_id = s3df_submit_multi(
                commands,
                job_label=label,
                time=time,
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                log_progress=True,
                dependency=prev_id if chain else None,
            )

        nn_commands = [
            make_gradient_command(
                param=param,
                tracks=track_spec,
                N=100,
                range_frac=0.2,
                noise_scale=0.0,
                noise_seed=0,
                adc_cutoffs=adc_cutoffs,
                results_dir=results_dir_nn,
                step_size=1.0,
                max_deposits=5000,
                store_per_plane_loss=True,
                no_wire_response=True,
                overwrite=overwrite,
            )
            for param in params
        ]
        nn_label = f"loss_diff_ta_ext_nwr_nonoise_{track_name}"
        if not print_sbatch_only:
            print(f"  {nn_label}: {len(nn_commands)} invocations (no noise)")
        prev_id = s3df_submit_multi(
            nn_commands,
            job_label=nn_label,
            time="01:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            log_progress=True,
            dependency=prev_id if chain else None,
        )


# ---------------------------------------------------------------------------
# Diffusion angle-pivot study
# ---------------------------------------------------------------------------
# Same as angle study but the track *midpoint* is fixed at (_PIVOT_X, 0, 0) mm.
# CSDA range of a 400 MeV muon in LAr ≈ 1700.6 mm → half-length 850.3 mm.
# start = (pivot_x + half_len*cos θ,  −half_len*sin θ,  0)  mm
# ---------------------------------------------------------------------------
_ANGLE_PIVOT_X_MM        = 1000.0
_ANGLE_PIVOT_HALF_LEN_MM = 850.3


def submit_diffusion_angle_pivot_study(
    *,
    submit=True,
    print_sbatch_only=False,
    check_complete=False,
    no_noise_only=False,
    overwrite=False,
    chain=False,
    time="08:00:00",
):
    """1D loss landscape: diffusion coefficients vs. track angle, pivot at x=1000 mm.

    Track     : 400 MeV muon, midpoint fixed at (1000, 0, 0) mm (west volume).
                Start: (1000 + 850.3·cos θ, −850.3·sin θ, 0) mm.
    Angles    : same as diffusion_angle_study
    Params    : diffusion_trans_cm2_us, diffusion_long_cm2_us
    Range     : ±20% of GT (N=100, 201 sweep points)
    ADC cuts  : [0, 5, 10, 20, 50]
    Noise     : scale=1.0, seeds 0–49 (50 seeds)  +  scale=0 single no-noise job
    step_size : 1 mm, max_deposits=5000

    Output    : $RESULTS_DIR/1d_gradients/diffusion_angle_pivot_study/
                $RESULTS_DIR/1d_gradients/diffusion_angle_pivot_study_nonoise/
    """
    results_dir    = "$RESULTS_DIR/1d_gradients/diffusion_angle_pivot_study"
    results_dir_nn = "$RESULTS_DIR/1d_gradients/diffusion_angle_pivot_study_nonoise"
    params      = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
    adc_cutoffs = [0, 5, 10, 20, 50]
    seeds       = list(range(50))  # 0–49

    if check_complete:
        invocations = [
            (f"Muon_400MeV_theta_{theta_deg}_pivot_x1000_stepsize_1mm  param={param}  seed={seed}",
             param,
             f"Muon_400MeV_theta_{theta_deg}_pivot_x1000_stepsize_1mm",
             seed)
            for theta_deg in _ANGLE_THETAS
            for param in params
            for seed in seeds
        ]
        print(f"diffusion_angle_pivot_study — checking {len(invocations)} invocations "
              f"in {results_dir.replace('$RESULTS_DIR', _RESULTS_DIR)}")
        _check_completions(results_dir, invocations, adc_cutoffs)
        return

    prev_id = None
    step = 0
    n_total =len(_ANGLE_THETAS)
    t_start = _time.time()
    for theta_deg in _ANGLE_THETAS:
        theta_rad = math.radians(theta_deg)
        dx = round(-math.cos(theta_rad), 9)
        dy = round( math.sin(theta_rad), 9)
        dz = 0.0
        direction = f"{dx:.9f},{dy:.9f},{dz:.1f}"

        start_x = round(_ANGLE_PIVOT_X_MM + _ANGLE_PIVOT_HALF_LEN_MM * math.cos(theta_rad), 3)
        start_y = round(-_ANGLE_PIVOT_HALF_LEN_MM * math.sin(theta_rad), 3)

        track_name = f"Muon_400MeV_theta_{theta_deg}_pivot_x1000_stepsize_1mm"
        track_spec = (f"{track_name}:{direction}:{_ANGLE_ENERGY}"
                      f":{start_x},{start_y},0")
        if not print_sbatch_only:
            step += 1
            _log_progress(step, n_total, track_name, t_start)

        if not no_noise_only:
            commands = [
                make_gradient_command(
                    param=param,
                    tracks=track_spec,
                    N=100,
                    range_frac=0.2,
                    noise_scale=1.0,
                    noise_seeds=seeds,
                    adc_cutoffs=adc_cutoffs,
                    results_dir=results_dir,
                    step_size=1.0,
                    max_deposits=5000,
                    store_per_plane_loss=True,
                    overwrite=overwrite,
                )
                for param in params
            ]
            label = f"loss_diff_anglepivot_{track_name}"
            if not print_sbatch_only:
                print(f"  {label}: {len(commands)} invocations")
            prev_id = s3df_submit_multi(
                commands,
                job_label=label,
                time=time,
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                log_progress=True,
                dependency=prev_id if chain else None,
            )

        # No-noise job
        nn_commands = [
            make_gradient_command(
                param=param,
                tracks=track_spec,
                N=100,
                range_frac=0.2,
                noise_scale=0.0,
                noise_seed=0,
                adc_cutoffs=adc_cutoffs,
                results_dir=results_dir_nn,
                step_size=1.0,
                max_deposits=5000,
                store_per_plane_loss=True,
                overwrite=overwrite,
            )
            for param in params
        ]
        nn_label = f"loss_diff_anglepivot_nonoise_{track_name}"
        if not print_sbatch_only:
            print(f"  {nn_label}: {len(nn_commands)} invocations (no noise)")
        prev_id = s3df_submit_multi(
            nn_commands,
            job_label=nn_label,
            time="01:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            log_progress=True,
            dependency=prev_id if chain else None,
        )


# ---------------------------------------------------------------------------
# Diffusion angle-pivot (theta, alpha) grid study
# ---------------------------------------------------------------------------
# Same as angle-pivot study but sweeps a 5×5 (theta, alpha) grid.
# Midpoint fixed at (_ANGLE_PIVOT_X_MM, 0, 0) mm.
# start = (pivot_x + L·cos θ·cos α,  −L·sin θ·cos α,  −L·sin α)  where L = half-length
# ---------------------------------------------------------------------------

def submit_diffusion_angle_pivot_theta_alpha_study(
    *,
    submit=True,
    print_sbatch_only=False,
    check_complete=False,
    no_noise_only=False,
    overwrite=False,
    chain=False,
    time="08:00:00",
):
    """1D loss landscape: diffusion coefficients vs. (theta, alpha) grid, pivot at x=1000 mm.

    Track     : 400 MeV muon, midpoint fixed at (1000, 0, 0) mm (west volume).
                theta — azimuthal angle in XY plane from -x axis (0°–20°, step 5°)
                alpha — lift angle from XY plane toward +z   (0°–20°, step 5°)
                Start: (1000 + L·cos θ·cos α,  −L·sin θ·cos α,  −L·sin α) mm,
                       where L = 850.3 mm (CSDA half-range of 400 MeV muon in LAr).
    Grid      : 5×5 = 25 (theta, alpha) combinations
    Params    : diffusion_trans_cm2_us, diffusion_long_cm2_us
    Range     : ±20% of GT (N=100, 201 sweep points)
    ADC cuts  : [0, 5, 10, 20, 50]
    Noise     : scale=1.0, seeds 0–49 (50 seeds)  +  scale=0 single no-noise job
    step_size : 1 mm, max_deposits=5000

    Output    : $RESULTS_DIR/1d_gradients/diffusion_angle_pivot_theta_alpha/
                $RESULTS_DIR/1d_gradients/diffusion_angle_pivot_theta_alpha_nonoise/
    """
    results_dir    = "$RESULTS_DIR/1d_gradients/diffusion_angle_pivot_theta_alpha"
    results_dir_nn = "$RESULTS_DIR/1d_gradients/diffusion_angle_pivot_theta_alpha_nonoise"
    params      = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
    adc_cutoffs = [0, 5, 10, 20, 50]
    seeds       = list(range(50))  # 0–49

    if check_complete:
        invocations = [
            (f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_pivot_x1000_stepsize_1mm  param={param}  seed={seed}",
             param,
             f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_pivot_x1000_stepsize_1mm",
             seed)
            for theta_deg in _TA_THETAS
            for alpha_deg in _TA_ALPHAS
            for param in params
            for seed in seeds
        ]
        print(f"diffusion_angle_pivot_theta_alpha — checking {len(invocations)} invocations "
              f"in {results_dir.replace('$RESULTS_DIR', _RESULTS_DIR)}")
        _check_completions(results_dir, invocations, adc_cutoffs)
        return

    prev_id = None
    step = 0
    n_total =len(_TA_THETAS) * len(_TA_ALPHAS)
    t_start = _time.time()
    for theta_deg in _TA_THETAS:
        for alpha_deg in _TA_ALPHAS:
            theta_rad = math.radians(theta_deg)
            alpha_rad = math.radians(alpha_deg)
            dx = round(-math.cos(theta_rad) * math.cos(alpha_rad), 9)
            dy = round( math.sin(theta_rad) * math.cos(alpha_rad), 9)
            dz = round( math.sin(alpha_rad), 9)
            direction = f"{dx:.9f},{dy:.9f},{dz:.9f}"

            start_x = round(_ANGLE_PIVOT_X_MM + _ANGLE_PIVOT_HALF_LEN_MM * math.cos(theta_rad) * math.cos(alpha_rad), 3)
            start_y = round(-_ANGLE_PIVOT_HALF_LEN_MM * math.sin(theta_rad) * math.cos(alpha_rad), 3)
            start_z = round(-_ANGLE_PIVOT_HALF_LEN_MM * math.sin(alpha_rad), 3)

            track_name = f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_pivot_x1000_stepsize_1mm"
            track_spec = (f"{track_name}:{direction}:{_ANGLE_ENERGY}"
                          f":{start_x},{start_y},{start_z}")
            if not print_sbatch_only:
                step += 1
                _log_progress(step, n_total, track_name, t_start)

            if not no_noise_only:
                commands = [
                    make_gradient_command(
                        param=param,
                        tracks=track_spec,
                        N=100,
                        range_frac=0.2,
                        noise_scale=1.0,
                        noise_seeds=seeds,
                        adc_cutoffs=adc_cutoffs,
                        results_dir=results_dir,
                        step_size=1.0,
                        max_deposits=5000,
                        store_per_plane_loss=True,
                        overwrite=overwrite,
                    )
                    for param in params
                ]
                label = f"loss_diff_pivot_ta_{track_name}"
                if not print_sbatch_only:
                    print(f"  {label}: {len(commands)} invocations")
                prev_id = s3df_submit_multi(
                    commands,
                    job_label=label,
                    time=time,
                    submit=submit,
                    mem_gb=64,
                    print_sbatch_command=print_sbatch_only,
                    log_progress=True,
                    dependency=prev_id if chain else None,
                )

            # No-noise job
            nn_commands = [
                make_gradient_command(
                    param=param,
                    tracks=track_spec,
                    N=100,
                    range_frac=0.2,
                    noise_scale=0.0,
                    noise_seed=0,
                    adc_cutoffs=adc_cutoffs,
                    results_dir=results_dir_nn,
                    step_size=1.0,
                    max_deposits=5000,
                    store_per_plane_loss=True,
                    overwrite=overwrite,
                )
                for param in params
            ]
            nn_label = f"loss_diff_pivot_ta_nonoise_{track_name}"
            if not print_sbatch_only:
                print(f"  {nn_label}: {len(nn_commands)} invocations (no noise)")
            prev_id = s3df_submit_multi(
                nn_commands,
                job_label=nn_label,
                time="01:00:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                log_progress=True,
                dependency=prev_id if chain else None,
            )


def submit_diffusion_angle_pivot_theta_alpha_study_no_wire_response(
    *,
    submit=True,
    print_sbatch_only=False,
    check_complete=False,
    no_noise_only=False,
    overwrite=False,
    chain=False,
    time="08:00:00",
):
    """1D loss landscape: diffusion vs. (theta, alpha) grid, pivot at x=1000 mm, delta kernels (no wire response).

    Track     : 400 MeV muon, midpoint fixed at (1000, 0, 0) mm (west volume).
                theta — azimuthal angle in XY plane from -x axis (0°–20°, step 5°)
                alpha — lift angle from XY plane toward +z   (0°–20°, step 5°)
    Grid      : 5×5 = 25 (theta, alpha) combinations (same as full theta_alpha study)
    Params    : diffusion_trans_cm2_us, diffusion_long_cm2_us
    Range     : ±20% of GT (N=100)
    ADC cuts  : [0, 5, 10, 20, 50]
    Noise     : scale=1.0, seeds 0–49  +  scale=0 no-noise job
    Wire resp : disabled (--no-wire-response)

    Output    : $RESULTS_DIR/1d_gradients/diffusion_angle_pivot_theta_alpha_no_wire_response/
                $RESULTS_DIR/1d_gradients/diffusion_angle_pivot_theta_alpha_no_wire_response_nonoise/
    """
    results_dir    = "$RESULTS_DIR/1d_gradients/diffusion_angle_pivot_theta_alpha_no_wire_response"
    results_dir_nn = "$RESULTS_DIR/1d_gradients/diffusion_angle_pivot_theta_alpha_no_wire_response_nonoise"
    params      = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
    adc_cutoffs = [0, 5, 10, 20, 50]
    seeds       = list(range(50))

    if check_complete:
        invocations = [
            (f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_pivot_x1000_stepsize_1mm  param={param}  seed={seed}",
             param,
             f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_pivot_x1000_stepsize_1mm",
             seed)
            for theta_deg in _TA_THETAS
            for alpha_deg in _TA_ALPHAS
            for param in params
            for seed in seeds
        ]
        print(f"diffusion_angle_pivot_theta_alpha_no_wire_response — checking {len(invocations)} invocations "
              f"in {results_dir.replace('$RESULTS_DIR', _RESULTS_DIR)}")
        _check_completions(results_dir, invocations, adc_cutoffs)
        return

    prev_id = None
    step = 0
    n_total =len(_TA_THETAS) * len(_TA_ALPHAS)
    t_start = _time.time()
    for theta_deg in _TA_THETAS:
        for alpha_deg in _TA_ALPHAS:
            theta_rad = math.radians(theta_deg)
            alpha_rad = math.radians(alpha_deg)
            dx = round(-math.cos(theta_rad) * math.cos(alpha_rad), 9)
            dy = round( math.sin(theta_rad) * math.cos(alpha_rad), 9)
            dz = round( math.sin(alpha_rad), 9)
            direction = f"{dx:.9f},{dy:.9f},{dz:.9f}"

            start_x = round(_ANGLE_PIVOT_X_MM + _ANGLE_PIVOT_HALF_LEN_MM * math.cos(theta_rad) * math.cos(alpha_rad), 3)
            start_y = round(-_ANGLE_PIVOT_HALF_LEN_MM * math.sin(theta_rad) * math.cos(alpha_rad), 3)
            start_z = round(-_ANGLE_PIVOT_HALF_LEN_MM * math.sin(alpha_rad), 3)

            track_name = f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_pivot_x1000_stepsize_1mm"
            track_spec = (f"{track_name}:{direction}:{_ANGLE_ENERGY}"
                          f":{start_x},{start_y},{start_z}")
            if not print_sbatch_only:
                step += 1
                _log_progress(step, n_total, track_name, t_start)

            if not no_noise_only:
                commands = [
                    make_gradient_command(
                        param=param,
                        tracks=track_spec,
                        N=100,
                        range_frac=0.2,
                        noise_scale=1.0,
                        noise_seeds=seeds,
                        adc_cutoffs=adc_cutoffs,
                        results_dir=results_dir,
                        step_size=1.0,
                        max_deposits=5000,
                        store_per_plane_loss=True,
                        no_wire_response=True,
                        overwrite=overwrite,
                    )
                    for param in params
                ]
                label = f"loss_diff_pivot_ta_nwr_{track_name}"
                if not print_sbatch_only:
                    print(f"  {label}: {len(commands)} invocations")
                prev_id = s3df_submit_multi(
                    commands,
                    job_label=label,
                    time=time,
                    submit=submit,
                    mem_gb=64,
                    print_sbatch_command=print_sbatch_only,
                    log_progress=True,
                    dependency=prev_id if chain else None,
                )

            nn_commands = [
                make_gradient_command(
                    param=param,
                    tracks=track_spec,
                    N=100,
                    range_frac=0.2,
                    noise_scale=0.0,
                    noise_seed=0,
                    adc_cutoffs=adc_cutoffs,
                    results_dir=results_dir_nn,
                    step_size=1.0,
                    max_deposits=5000,
                    store_per_plane_loss=True,
                    no_wire_response=True,
                    overwrite=overwrite,
                )
                for param in params
            ]
            nn_label = f"loss_diff_pivot_ta_nwr_nonoise_{track_name}"
            if not print_sbatch_only:
                print(f"  {nn_label}: {len(nn_commands)} invocations (no noise)")
            prev_id = s3df_submit_multi(
                nn_commands,
                job_label=nn_label,
                time="01:00:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                log_progress=True,
                dependency=prev_id if chain else None,
            )


def submit_diffusion_angle_pivot_theta_alpha_extended_study(
    *,
    submit=True,
    print_sbatch_only=False,
    check_complete=False,
    no_noise_only=False,
    overwrite=False,
    chain=False,
    n_chains=1,
    time="08:00:00",
):
    """1D loss landscape: diffusion vs (theta, alpha), pivot at x=1000 mm, extended grid (angles > 20 deg).

    Track     : 400 MeV muon, midpoint fixed at (1000, 0, 0) mm (same as
                diffusion_angle_pivot_theta_alpha). theta, alpha each range
                0-50 deg in 5 deg steps (11x11 = 121 combos); only the 96
                combos with theta>20 or alpha>20 are submitted here — the
                5x5=25 combos with both <=20 are covered by
                diffusion_angle_pivot_theta_alpha.
    Params    : diffusion_trans_cm2_us, diffusion_long_cm2_us
    Range     : ±20% of GT (N=100, 201 sweep points)
    ADC cuts  : [0, 50]  (reduced from the base study's [0,5,10,20,50] to keep job count down)
    Noise     : scale=1.0, seeds 0-49 (50 seeds)  +  scale=0 no-noise commands, same job
    step_size : 1 mm, max_deposits=5000
    Jobs      : the 96 combos are split into 10 sbatch jobs (~10 combos/job); each job
                bundles the noisy + no-noise commands for its combos sequentially,
                still within the single time= walltime budget (default 8h)

    Output    : $RESULTS_DIR/1d_gradients/diffusion_angle_pivot_theta_alpha/  (same dir as base study)
                $RESULTS_DIR/1d_gradients/diffusion_angle_pivot_theta_alpha_nonoise/
    """
    results_dir    = "$RESULTS_DIR/1d_gradients/diffusion_angle_pivot_theta_alpha"
    results_dir_nn = "$RESULTS_DIR/1d_gradients/diffusion_angle_pivot_theta_alpha_nonoise"
    params      = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
    adc_cutoffs = [0, 50]
    seeds       = list(range(50))  # 0-49
    n_jobs      = 10

    if check_complete:
        invocations = [
            (f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_pivot_x1000_stepsize_1mm  param={param}  seed={seed}",
             param,
             f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_pivot_x1000_stepsize_1mm",
             seed)
            for theta_deg, alpha_deg in _TA_PIVOT_EXTENDED_COMBOS
            for param in params
            for seed in seeds
        ]
        print(f"diffusion_angle_pivot_theta_alpha_extended — checking {len(invocations)} invocations "
              f"in {results_dir.replace('$RESULTS_DIR', _RESULTS_DIR)}")
        _check_completions(results_dir, invocations, adc_cutoffs)
        return

    chunks = _split_evenly(_TA_PIVOT_EXTENDED_COMBOS, n_jobs)
    prev_ids = [None] * n_chains
    t_start = _time.time()
    for chunk_idx, chunk in enumerate(chunks):
        if not print_sbatch_only:
            _log_progress(chunk_idx + 1, len(chunks), f"chunk {chunk_idx} ({len(chunk)} combos)", t_start)

        commands = []
        for theta_deg, alpha_deg in chunk:
            theta_rad = math.radians(theta_deg)
            alpha_rad = math.radians(alpha_deg)
            dx = round(-math.cos(theta_rad) * math.cos(alpha_rad), 9)
            dy = round( math.sin(theta_rad) * math.cos(alpha_rad), 9)
            dz = round( math.sin(alpha_rad), 9)
            direction = f"{dx:.9f},{dy:.9f},{dz:.9f}"

            start_x = round(_ANGLE_PIVOT_X_MM + _ANGLE_PIVOT_HALF_LEN_MM * math.cos(theta_rad) * math.cos(alpha_rad), 3)
            start_y = round(-_ANGLE_PIVOT_HALF_LEN_MM * math.sin(theta_rad) * math.cos(alpha_rad), 3)
            start_z = round(-_ANGLE_PIVOT_HALF_LEN_MM * math.sin(alpha_rad), 3)

            track_name = f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_pivot_x1000_stepsize_1mm"
            track_spec = (f"{track_name}:{direction}:{_ANGLE_ENERGY}"
                          f":{start_x},{start_y},{start_z}")

            if not no_noise_only:
                commands += [
                    make_gradient_command(
                        param=param,
                        tracks=track_spec,
                        N=100,
                        range_frac=0.2,
                        noise_scale=1.0,
                        noise_seeds=seeds,
                        adc_cutoffs=adc_cutoffs,
                        results_dir=results_dir,
                        step_size=1.0,
                        max_deposits=5000,
                        store_per_plane_loss=True,
                        overwrite=overwrite,
                    )
                    for param in params
                ]
            # No-noise commands, bundled into the same job
            commands += [
                make_gradient_command(
                    param=param,
                    tracks=track_spec,
                    N=100,
                    range_frac=0.2,
                    noise_scale=0.0,
                    noise_seed=0,
                    adc_cutoffs=adc_cutoffs,
                    results_dir=results_dir_nn,
                    step_size=1.0,
                    max_deposits=5000,
                    store_per_plane_loss=True,
                    overwrite=overwrite,
                )
                for param in params
            ]

        label = f"loss_diff_pivot_ta_ext_chunk{chunk_idx}"
        if not print_sbatch_only:
            print(f"  {label}: {len(commands)} invocations ({len(chunk)} combos)")
        chain_idx = chunk_idx % n_chains
        prev_ids[chain_idx] = s3df_submit_multi(
            commands,
            job_label=label,
            time=time,
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            log_progress=True,
            dependency=prev_ids[chain_idx] if chain else None,
        )


def submit_diffusion_angle_pivot_theta_alpha_extended_study_no_wire_response(
    *,
    submit=True,
    print_sbatch_only=False,
    check_complete=False,
    no_noise_only=False,
    overwrite=False,
    chain=False,
    n_chains=1,
    time="08:00:00",
):
    """1D loss landscape: diffusion vs (theta, alpha), pivot at x=1000 mm, extended grid (angles > 20 deg), delta kernels.

    Same grid and structure as submit_diffusion_angle_pivot_theta_alpha_extended_study,
    but with wire response disabled (--no-wire-response); writes into the same
    dirs as diffusion_angle_pivot_theta_alpha_no_wire_response.

    Output    : $RESULTS_DIR/1d_gradients/diffusion_angle_pivot_theta_alpha_no_wire_response/
                $RESULTS_DIR/1d_gradients/diffusion_angle_pivot_theta_alpha_no_wire_response_nonoise/
    """
    results_dir    = "$RESULTS_DIR/1d_gradients/diffusion_angle_pivot_theta_alpha_no_wire_response"
    results_dir_nn = "$RESULTS_DIR/1d_gradients/diffusion_angle_pivot_theta_alpha_no_wire_response_nonoise"
    params      = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
    adc_cutoffs = [0, 50]
    seeds       = list(range(50))
    n_jobs      = 10

    if check_complete:
        invocations = [
            (f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_pivot_x1000_stepsize_1mm  param={param}  seed={seed}",
             param,
             f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_pivot_x1000_stepsize_1mm",
             seed)
            for theta_deg, alpha_deg in _TA_PIVOT_EXTENDED_COMBOS
            for param in params
            for seed in seeds
        ]
        print(f"diffusion_angle_pivot_theta_alpha_extended_no_wire_response — checking {len(invocations)} invocations "
              f"in {results_dir.replace('$RESULTS_DIR', _RESULTS_DIR)}")
        _check_completions(results_dir, invocations, adc_cutoffs)
        return

    chunks = _split_evenly(_TA_PIVOT_EXTENDED_COMBOS, n_jobs)
    prev_ids = [None] * n_chains
    t_start = _time.time()
    for chunk_idx, chunk in enumerate(chunks):
        if not print_sbatch_only:
            _log_progress(chunk_idx + 1, len(chunks), f"chunk {chunk_idx} ({len(chunk)} combos)", t_start)

        commands = []
        for theta_deg, alpha_deg in chunk:
            theta_rad = math.radians(theta_deg)
            alpha_rad = math.radians(alpha_deg)
            dx = round(-math.cos(theta_rad) * math.cos(alpha_rad), 9)
            dy = round( math.sin(theta_rad) * math.cos(alpha_rad), 9)
            dz = round( math.sin(alpha_rad), 9)
            direction = f"{dx:.9f},{dy:.9f},{dz:.9f}"

            start_x = round(_ANGLE_PIVOT_X_MM + _ANGLE_PIVOT_HALF_LEN_MM * math.cos(theta_rad) * math.cos(alpha_rad), 3)
            start_y = round(-_ANGLE_PIVOT_HALF_LEN_MM * math.sin(theta_rad) * math.cos(alpha_rad), 3)
            start_z = round(-_ANGLE_PIVOT_HALF_LEN_MM * math.sin(alpha_rad), 3)

            track_name = f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}_pivot_x1000_stepsize_1mm"
            track_spec = (f"{track_name}:{direction}:{_ANGLE_ENERGY}"
                          f":{start_x},{start_y},{start_z}")

            if not no_noise_only:
                commands += [
                    make_gradient_command(
                        param=param,
                        tracks=track_spec,
                        N=100,
                        range_frac=0.2,
                        noise_scale=1.0,
                        noise_seeds=seeds,
                        adc_cutoffs=adc_cutoffs,
                        results_dir=results_dir,
                        step_size=1.0,
                        max_deposits=5000,
                        store_per_plane_loss=True,
                        no_wire_response=True,
                        overwrite=overwrite,
                    )
                    for param in params
                ]
            # No-noise commands, bundled into the same job
            commands += [
                make_gradient_command(
                    param=param,
                    tracks=track_spec,
                    N=100,
                    range_frac=0.2,
                    noise_scale=0.0,
                    noise_seed=0,
                    adc_cutoffs=adc_cutoffs,
                    results_dir=results_dir_nn,
                    step_size=1.0,
                    max_deposits=5000,
                    store_per_plane_loss=True,
                    no_wire_response=True,
                    overwrite=overwrite,
                )
                for param in params
            ]

        label = f"loss_diff_pivot_ta_ext_nwr_chunk{chunk_idx}"
        if not print_sbatch_only:
            print(f"  {label}: {len(commands)} invocations ({len(chunk)} combos)")
        chain_idx = chunk_idx % n_chains
        prev_ids[chain_idx] = s3df_submit_multi(
            commands,
            job_label=label,
            time=time,
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            log_progress=True,
            dependency=prev_ids[chain_idx] if chain else None,
        )


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------
_PROFILES = {
    "diffusion_startx_study":                               submit_diffusion_startx_study,
    "diffusion_startx_study_no_wire_response":              submit_diffusion_startx_study_no_wire_response,
    "diffusion_angle_study":                                submit_diffusion_angle_study,
    "diffusion_angle_study_no_wire_response":               submit_diffusion_angle_study_no_wire_response,
    "diffusion_angle_theta_alpha":                          submit_diffusion_angle_theta_alpha_study,
    "diffusion_angle_theta_alpha_no_wire_response":         submit_diffusion_angle_theta_alpha_study_no_wire_response,
    "diffusion_angle_theta_alpha_extended":                 submit_diffusion_angle_theta_alpha_extended_study,
    "diffusion_angle_theta_alpha_extended_no_wire_response": submit_diffusion_angle_theta_alpha_extended_study_no_wire_response,
    "diffusion_angle_pivot_study":                          submit_diffusion_angle_pivot_study,
    "diffusion_angle_pivot_study_no_wire_response":         submit_diffusion_angle_pivot_study_no_wire_response,
    "diffusion_angle_pivot_theta_alpha":                    submit_diffusion_angle_pivot_theta_alpha_study,
    "diffusion_angle_pivot_theta_alpha_no_wire_response":   submit_diffusion_angle_pivot_theta_alpha_study_no_wire_response,
    "diffusion_angle_pivot_theta_alpha_extended":                    submit_diffusion_angle_pivot_theta_alpha_extended_study,
    "diffusion_angle_pivot_theta_alpha_extended_no_wire_response":   submit_diffusion_angle_pivot_theta_alpha_extended_study_no_wire_response,
}

if __name__ == "__main__":
    import argparse
    import inspect

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("profile", choices=list(_PROFILES))
    p.add_argument("--submit", action="store_true",
                   help="Actually submit jobs to Slurm (default: dry-run)")
    p.add_argument("--print-sbatch-command", action="store_true",
                   help="Print sbatch script to stdout instead of writing to disk")
    p.add_argument("--check-complete", action="store_true",
                   help="Check how many 1d_gradients.py invocations completed and which still need to run")
    p.add_argument("--no-noise-only", action="store_true",
                   help="Only submit the no-noise jobs (noise_scale=0, seed=0); skip the noisy seed sweep")
    p.add_argument("--overwrite", action="store_true",
                   help="Delete and rerun even if output pkl files already exist")
    p.add_argument("--chain", action="store_true",
                   help="Chain SLURM jobs within the profile: each job depends on the previous "
                        "(requires --submit; no-op in dry-run mode)")
    p.add_argument("--n-chains", type=int, default=1,
                   help="When --chain is set, round-robin jobs across this many independent "
                        "chains so they run in parallel streams instead of one long sequence "
                        "(only supported by profiles that expose n_chains; ignored otherwise)")
    args = p.parse_args()

    kwargs = dict(
        submit=args.submit,
        print_sbatch_only=args.print_sbatch_command,
        check_complete=args.check_complete,
        no_noise_only=args.no_noise_only,
        overwrite=args.overwrite,
        chain=args.chain,
    )
    fn = _PROFILES[args.profile]
    if "n_chains" in inspect.signature(fn).parameters:
        kwargs["n_chains"] = args.n_chains
    fn(**kwargs)
