#!/usr/bin/env python3
"""Submit the higher-angle diagonal pivot (theta, alpha) study as a SINGLE Slurm job.

Bundles the same 4 invocations as pivot_theta_alpha_landscape_diagonal.sh
(diffusion_trans/long x clean/noisy) into one sbatch job that runs them
sequentially on one GPU via s3df_submit_multi, instead of running locally.

Tracks: 6 diagonal (theta, alpha) pivot tracks at 25/30/35/40/45/50 deg,
continuing the 0/10/20 diagonal already covered by the local script.

Usage (default is dry-run: writes the job script under jobs/ but does not
call sbatch):
    /sdf/home/g/gregork/envs/base_env/bin/python scripts/20260611/submit_pivot_theta_alpha_diagonal_high.py
Add --submit to actually submit to Slurm:
    ... --submit

After the job completes, generate the viewer locally with:
    .venv/bin/python src/analysis/sim_param_sweeps/generate_gradient_viewer.py \
        --dir results/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_diag
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "jobs"))
from submit_jobs import make_gradient_command, s3df_submit_multi

TRACKS = (
    "Muon_400MeV_theta_25_alpha_25_pivot_x1000_stepsize_1mm:-0.821393805,0.383022222,0.422618262:400:1698.431,-325.684,-359.352"
    "+Muon_400MeV_theta_30_alpha_30_pivot_x1000_stepsize_1mm:-0.750000000,0.433012702,0.500000000:400:1637.725,-368.191,-425.15"
    "+Muon_400MeV_theta_35_alpha_35_pivot_x1000_stepsize_1mm:-0.671010072,0.469846310,0.573576436:400:1570.56,-399.51,-487.712"
    "+Muon_400MeV_theta_40_alpha_40_pivot_x1000_stepsize_1mm:-0.586824089,0.492403877,0.642787610:400:1498.977,-418.691,-546.562"
    "+Muon_400MeV_theta_45_alpha_45_pivot_x1000_stepsize_1mm:-0.500000000,0.500000000,0.707106781:400:1425.15,-425.15,-601.253"
    "+Muon_400MeV_theta_50_alpha_50_pivot_x1000_stepsize_1mm:-0.413175911,0.492403877,0.766044443:400:1351.323,-418.691,-651.368"
)

OUT_DIR = "$RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_diag"

COMMON = dict(
    tracks=TRACKS,
    factors=[0.75, 1.0, 1.25],
    step_size=1.0,
    max_deposits=5000,
    sobolev_max_pad=128,
    results_dir=OUT_DIR,
    adc_cutoff=50,
    store_per_plane_loss=True,
    store_per_pixel_loss_and_grad=True,
    store_arrays=True,
    save_per_factor=True,
)


def build_commands():
    return [
        make_gradient_command(param="diffusion_trans_cm2_us", noise_scale=0.0, noise_seed=42, **COMMON),
        make_gradient_command(param="diffusion_long_cm2_us", noise_scale=0.0, noise_seed=42, **COMMON),
        make_gradient_command(param="diffusion_trans_cm2_us", noise_scale=1.0, noise_seeds=[42, 43], **COMMON),
        make_gradient_command(param="diffusion_long_cm2_us", noise_scale=1.0, noise_seeds=[42, 43], **COMMON),
    ]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--submit", action="store_true",
                   help="Actually submit the job to Slurm (default: dry-run, writes job script only)")
    p.add_argument("--print-sbatch-command", action="store_true",
                   help="Print the sbatch command to stdout instead of writing to disk")
    p.add_argument("--time", default="08:00:00")
    args = p.parse_args()

    commands = build_commands()
    print(f"bundling {len(commands)} invocations into a single Slurm job")
    s3df_submit_multi(
        commands,
        job_label="pivot_ta_diag_high",
        time=args.time,
        submit=args.submit,
        print_sbatch_command=args.print_sbatch_command,
        mem_gb=64,
    )
