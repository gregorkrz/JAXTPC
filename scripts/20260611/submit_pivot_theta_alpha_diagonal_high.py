#!/usr/bin/env python3
"""Submit pivot (theta, alpha) diffusion studies as Slurm jobs.

Two track groups are supported (select with --group):

  diag    — 6 diagonal tracks (theta==alpha): 25/30/35/40/45/50 deg.
             Continuation of the 0/10/20 diagonal in the local script.

  offdiag — 18 off-diagonal tracks: one angle high {40,45,50} deg,
             the other low {0,10,20} deg, all combinations.
             (High-theta × low-alpha) + (Low-theta × high-alpha).

Each group is submitted as two separate Slurm jobs (diffusion_trans and
diffusion_long), each job running sequentially on one GPU via s3df_submit_multi.

Noise seeds: 0–49 (50 seeds). ADC cutoffs: 0 and 50. Wire response ON.

Usage (default is dry-run: writes the job script under jobs/ but does not
call sbatch):
    /sdf/home/g/gregork/envs/base_env/bin/python scripts/20260611/submit_pivot_theta_alpha_diagonal_high.py [--group diag|offdiag|all]
Add --submit to actually submit to Slurm:
    ... --submit

After the job completes, generate the viewer locally with:
    .venv/bin/python src/analysis/sim_param_sweeps/generate_gradient_viewer.py \
        --dir results/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_diag
    .venv/bin/python src/analysis/sim_param_sweeps/generate_gradient_viewer.py \
        --dir results/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_offdiag
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "jobs"))
from submit_jobs import make_gradient_command, s3df_submit_multi

# ── Diagonal tracks: theta == alpha ──────────────────────────────────────────
TRACKS_DIAG = (
    "Muon_400MeV_theta_25_alpha_25_pivot_x1000_stepsize_1mm:-0.821393805,0.383022222,0.422618262:400:1698.431,-325.684,-359.352"
    "+Muon_400MeV_theta_30_alpha_30_pivot_x1000_stepsize_1mm:-0.750000000,0.433012702,0.500000000:400:1637.725,-368.191,-425.15"
    "+Muon_400MeV_theta_35_alpha_35_pivot_x1000_stepsize_1mm:-0.671010072,0.469846310,0.573576436:400:1570.56,-399.51,-487.712"
    "+Muon_400MeV_theta_40_alpha_40_pivot_x1000_stepsize_1mm:-0.586824089,0.492403877,0.642787610:400:1498.977,-418.691,-546.562"
    "+Muon_400MeV_theta_45_alpha_45_pivot_x1000_stepsize_1mm:-0.500000000,0.500000000,0.707106781:400:1425.15,-425.15,-601.253"
    "+Muon_400MeV_theta_50_alpha_50_pivot_x1000_stepsize_1mm:-0.413175911,0.492403877,0.766044443:400:1351.323,-418.691,-651.368"
)

# ── Off-diagonal tracks: one angle high {40,45,50}, other low {0,10,20} ──────
# Direction: dx=-cos(t)cos(a), dy=sin(t)cos(a), dz=sin(a)
# Start:     (1000 + L*cos(t)cos(a), -L*sin(t)cos(a), -L*sin(a)),  L=850.3 mm
TRACKS_OFFDIAG = (
    # High theta (40/45/50) × Low alpha (0/10/20)
    "Muon_400MeV_theta_40_alpha_0_pivot_x1000_stepsize_1mm:-0.766044443,0.642787610,0.000000000:400:1651.368,-546.562,0.000"
    "+Muon_400MeV_theta_40_alpha_10_pivot_x1000_stepsize_1mm:-0.754406507,0.633022222,0.173648178:400:1641.472,-538.259,-147.653"
    "+Muon_400MeV_theta_40_alpha_20_pivot_x1000_stepsize_1mm:-0.719846310,0.604022774,0.342020143:400:1612.085,-513.601,-290.820"
    "+Muon_400MeV_theta_45_alpha_0_pivot_x1000_stepsize_1mm:-0.707106781,0.707106781,0.000000000:400:1601.253,-601.253,0.000"
    "+Muon_400MeV_theta_45_alpha_10_pivot_x1000_stepsize_1mm:-0.696364240,0.696364240,0.173648178:400:1592.119,-592.119,-147.653"
    "+Muon_400MeV_theta_45_alpha_20_pivot_x1000_stepsize_1mm:-0.664463024,0.664463024,0.342020143:400:1564.993,-564.993,-290.820"
    "+Muon_400MeV_theta_50_alpha_0_pivot_x1000_stepsize_1mm:-0.642787610,0.766044443,0.000000000:400:1546.562,-651.368,0.000"
    "+Muon_400MeV_theta_50_alpha_10_pivot_x1000_stepsize_1mm:-0.633022222,0.754406507,0.173648178:400:1538.259,-641.472,-147.653"
    "+Muon_400MeV_theta_50_alpha_20_pivot_x1000_stepsize_1mm:-0.604022774,0.719846310,0.342020143:400:1513.601,-612.085,-290.820"
    # Low theta (0/10/20) × High alpha (40/45/50)
    "+Muon_400MeV_theta_0_alpha_40_pivot_x1000_stepsize_1mm:-0.766044443,0.000000000,0.642787610:400:1651.368,0.000,-546.562"
    "+Muon_400MeV_theta_0_alpha_45_pivot_x1000_stepsize_1mm:-0.707106781,0.000000000,0.707106781:400:1601.253,0.000,-601.253"
    "+Muon_400MeV_theta_0_alpha_50_pivot_x1000_stepsize_1mm:-0.642787610,0.000000000,0.766044443:400:1546.562,0.000,-651.368"
    "+Muon_400MeV_theta_10_alpha_40_pivot_x1000_stepsize_1mm:-0.754406507,0.133022222,0.642787610:400:1641.472,-113.109,-546.562"
    "+Muon_400MeV_theta_10_alpha_45_pivot_x1000_stepsize_1mm:-0.696364240,0.122787804,0.707106781:400:1592.119,-104.406,-601.253"
    "+Muon_400MeV_theta_10_alpha_50_pivot_x1000_stepsize_1mm:-0.633022222,0.111618897,0.766044443:400:1538.259,-94.910,-651.368"
    "+Muon_400MeV_theta_20_alpha_40_pivot_x1000_stepsize_1mm:-0.719846310,0.262002630,0.642787610:400:1612.085,-222.781,-546.562"
    "+Muon_400MeV_theta_20_alpha_45_pivot_x1000_stepsize_1mm:-0.664463024,0.241844763,0.707106781:400:1564.993,-205.641,-601.253"
    "+Muon_400MeV_theta_20_alpha_50_pivot_x1000_stepsize_1mm:-0.604022774,0.219846310,0.766044443:400:1513.601,-186.935,-651.368"
)

OUT_DIR_DIAG    = "$RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_diag"
OUT_DIR_OFFDIAG = "$RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_offdiag"

NOISY_SEEDS = list(range(50))

COMMON_BASE = dict(
    N=100,
    range_frac=0.2,
    step_size=1.0,
    max_deposits=5000,
    sobolev_max_pad=128,
    adc_cutoffs=[0, 50],
    store_per_plane_loss=True,
    store_per_pixel_loss_and_grad=False,
    store_arrays=False,
)

# (param, job_label, tracks, out_dir)
JOBS = [
    ("diffusion_trans_cm2_us", "pivot_ta_diag_high_trans",  TRACKS_DIAG,    OUT_DIR_DIAG),
    ("diffusion_long_cm2_us",  "pivot_ta_diag_high_long",   TRACKS_DIAG,    OUT_DIR_DIAG),
    ("diffusion_trans_cm2_us", "pivot_ta_offdiag_trans",    TRACKS_OFFDIAG, OUT_DIR_OFFDIAG),
    ("diffusion_long_cm2_us",  "pivot_ta_offdiag_long",     TRACKS_OFFDIAG, OUT_DIR_OFFDIAG),
]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--submit", action="store_true",
                   help="Actually submit the job to Slurm (default: dry-run, writes job script only)")
    p.add_argument("--print-sbatch-command", action="store_true",
                   help="Print the sbatch command to stdout instead of writing to disk")
    p.add_argument("--time", default="08:00:00")
    p.add_argument("--param", choices=["diffusion_trans_cm2_us", "diffusion_long_cm2_us"],
                   default=None, help="Submit only this param (default: both)")
    p.add_argument("--group", choices=["diag", "offdiag", "all"], default="all",
                   help="Track group to submit: diag (theta==alpha), offdiag (mixed angles), or all (default: all)")
    args = p.parse_args()

    jobs = [
        (param, label, tracks, out_dir)
        for param, label, tracks, out_dir in JOBS
        if (args.param is None or param == args.param)
        and (args.group == "all"
             or (args.group == "diag"    and "diag"    in label and "offdiag" not in label)
             or (args.group == "offdiag" and "offdiag" in label))
    ]
    for param, job_label, tracks, out_dir in jobs:
        cmd = make_gradient_command(param=param, noise_scale=1.0, noise_seeds=NOISY_SEEDS,
                                    tracks=tracks, results_dir=out_dir, **COMMON_BASE)
        print(f"submitting job: {job_label}")
        s3df_submit_multi(
            [cmd],
            job_label=job_label,
            time=args.time,
            submit=args.submit,
            print_sbatch_command=args.print_sbatch_command,
            mem_gb=64,
        )
