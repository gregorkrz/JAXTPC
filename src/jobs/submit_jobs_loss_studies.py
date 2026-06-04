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
  diffusion_startx_study   D_trans / D_long loss vs. start-x position (Muons 4,5,10,12)
  diffusion_angle_study    D_trans / D_long loss vs. track angle (400 MeV, x=1900 mm)
"""

import math
import os
import sys

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
    print(f"  {n_complete}/{n_total} invocations complete.")
    if incomplete:
        print(f"  {len(incomplete)} still need to run:")
        for label in incomplete:
            print(f"    {label}")
    return n_complete, n_total, incomplete


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

    for base_name, direction, energy, z, x_positions in _DIFFUSION_STARTX_TRACKS:
        for start_x in x_positions:
            track_name = f"{base_name}_startx_{start_x}_stepsize_1mm"
            track_spec = f"{track_name}:{direction}:{energy}:{start_x},0,{z}"

            if not no_noise_only:
                commands = [
                    make_gradient_command(
                        param=param,
                        tracks=track_spec,
                        N=100,
                        range_frac=0.2,
                        noise_scale=1.0,
                        noise_seed=seed,
                        adc_cutoffs=adc_cutoffs,
                        results_dir=results_dir,
                        step_size=1.0,
                        max_deposits=5000,
                        store_per_plane_loss=True,
                    )
                    for param in params
                    for seed in seeds
                ]
                label = f"loss_diff_startx_{track_name}"
                if not print_sbatch_only:
                    print(f"  {label}: {len(commands)} invocations")
                s3df_submit_multi(
                    commands,
                    job_label=label,
                    time=time,
                    submit=submit,
                    mem_gb=64,
                    print_sbatch_command=print_sbatch_only,
                    log_progress=True,
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
                )
                for param in params
            ]
            nn_label = f"loss_diff_startx_nonoise_{track_name}"
            if not print_sbatch_only:
                print(f"  {nn_label}: {len(nn_commands)} invocations (no noise)")
            s3df_submit_multi(
                nn_commands,
                job_label=nn_label,
                time="01:00:00",
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                log_progress=True,
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

    for theta_deg in _ANGLE_THETAS:
        theta_rad = math.radians(theta_deg)
        dx = round(-math.cos(theta_rad), 9)
        dy = round( math.sin(theta_rad), 9)
        dz = 0.0
        direction = f"{dx:.9f},{dy:.9f},{dz:.1f}"

        track_name = f"Muon_400MeV_theta_{theta_deg}_stepsize_1mm"
        track_spec = (f"{track_name}:{direction}:{_ANGLE_ENERGY}"
                      f":{_ANGLE_START_X},0,0")

        if not no_noise_only:
            commands = [
                make_gradient_command(
                    param=param,
                    tracks=track_spec,
                    N=100,
                    range_frac=0.2,
                    noise_scale=1.0,
                    noise_seed=seed,
                    adc_cutoffs=adc_cutoffs,
                    results_dir=results_dir,
                    step_size=1.0,
                    max_deposits=5000,
                    store_per_plane_loss=True,
                )
                for param in params
                for seed in seeds
            ]
            label = f"loss_diff_angle_{track_name}"
            if not print_sbatch_only:
                print(f"  {label}: {len(commands)} invocations")
            s3df_submit_multi(
                commands,
                job_label=label,
                time=time,
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                log_progress=True,
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
            )
            for param in params
        ]
        nn_label = f"loss_diff_angle_nonoise_{track_name}"
        if not print_sbatch_only:
            print(f"  {nn_label}: {len(nn_commands)} invocations (no noise)")
        s3df_submit_multi(
            nn_commands,
            job_label=nn_label,
            time="01:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            log_progress=True,
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

        if not no_noise_only:
            commands = [
                make_gradient_command(
                    param=param,
                    tracks=track_spec,
                    N=100,
                    range_frac=0.2,
                    noise_scale=1.0,
                    noise_seed=seed,
                    adc_cutoffs=adc_cutoffs,
                    results_dir=results_dir,
                    step_size=1.0,
                    max_deposits=5000,
                    store_per_plane_loss=True,
                )
                for param in params
                for seed in seeds
            ]
            label = f"loss_diff_anglepivot_{track_name}"
            if not print_sbatch_only:
                print(f"  {label}: {len(commands)} invocations")
            s3df_submit_multi(
                commands,
                job_label=label,
                time=time,
                submit=submit,
                mem_gb=64,
                print_sbatch_command=print_sbatch_only,
                log_progress=True,
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
            )
            for param in params
        ]
        nn_label = f"loss_diff_anglepivot_nonoise_{track_name}"
        if not print_sbatch_only:
            print(f"  {nn_label}: {len(nn_commands)} invocations (no noise)")
        s3df_submit_multi(
            nn_commands,
            job_label=nn_label,
            time="01:00:00",
            submit=submit,
            mem_gb=64,
            print_sbatch_command=print_sbatch_only,
            log_progress=True,
        )


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------
_PROFILES = {
    "diffusion_startx_study":        submit_diffusion_startx_study,
    "diffusion_angle_study":         submit_diffusion_angle_study,
    "diffusion_angle_pivot_study":   submit_diffusion_angle_pivot_study,
}

if __name__ == "__main__":
    import argparse

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
    args = p.parse_args()

    _PROFILES[args.profile](
        submit=args.submit,
        print_sbatch_only=args.print_sbatch_command,
        check_complete=args.check_complete,
        no_noise_only=args.no_noise_only,
    )
