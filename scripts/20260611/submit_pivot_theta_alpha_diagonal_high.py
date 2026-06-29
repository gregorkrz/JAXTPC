#!/usr/bin/env python3
"""Submit pivot (theta, alpha) diffusion studies as Slurm jobs.

Four track groups are supported (select with --group):

  diag            — 6 diagonal tracks (theta==alpha): 25/30/35/40/45/50 deg.
                    Continuation of the 0/10/20 diagonal in the local script.

  offdiag         — 18 off-diagonal tracks: one angle high {40,45,50} deg,
                    the other low {0,10,20} deg, all combinations.
                    (High-theta × low-alpha) + (Low-theta × high-alpha).
                    Wire response ON.

  offdiag_nowire  — Same 18 off-diagonal tracks as offdiag but with
                    --no-wire-response (pixel-only simulation).

  fullgrid        — All 81 (theta, alpha) pairs from
                    {0,10,20,25,30,35,40,45,50}^2.  Already-completed entries
                    are skipped automatically by the underlying script.
                    Use --num-chunks N to spread across N Slurm jobs
                    (each chunk = one job, tracks split evenly).

Each group is submitted as two separate Slurm jobs (diffusion_trans and
diffusion_long), each job running sequentially on one GPU via s3df_submit_multi.

Noise seeds: 0–49 (50 seeds). ADC cutoffs: 0 and 50.

Usage (default is dry-run: writes the job script under jobs/ but does not
call sbatch):
    /sdf/home/g/gregork/envs/base_env/bin/python scripts/20260611/submit_pivot_theta_alpha_diagonal_high.py [--group diag|offdiag|fullgrid|all]
Full-grid example (9 chunks of 9 tracks each, 4 h walltime per chunk):
    /sdf/home/g/gregork/envs/base_env/bin/python scripts/20260611/submit_pivot_theta_alpha_diagonal_high.py --group fullgrid --num-chunks 9 --time 04:00:00 --submit
Add --submit to actually submit to Slurm:
    ... --submit

After the job completes, generate the viewer locally with:
    .venv/bin/python src/analysis/sim_param_sweeps/generate_gradient_viewer.py \
        --dir results/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_diag
    .venv/bin/python src/analysis/sim_param_sweeps/generate_gradient_viewer.py \
        --dir results/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_offdiag
    .venv/bin/python src/analysis/sim_param_sweeps/generate_gradient_viewer.py \
        --dir results/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_fullgrid
"""
import argparse
import math
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

OUT_DIR_DIAG           = "$RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_diag"
OUT_DIR_OFFDIAG        = "$RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_offdiag"
OUT_DIR_OFFDIAG_NOWIRE = "$RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_offdiag_nowire"
OUT_DIR_FULLGRID       = "$RESULTS_DIR/1d_gradients/cutoff_loss_landscape_20260611_pivot_ta_fullgrid"

# Full grid: all (theta, alpha) pairs from this set
ANGLES_FULL = [0, 10, 20, 25, 30, 35, 40, 45, 50]
# L = track half-length (mm); start = pivot + L * direction_reversed
_L = 850.3


def _make_fullgrid_tracks():
    """Return list of track definition strings for all 81 (theta, alpha) pairs."""
    tracks = []
    for theta_deg in ANGLES_FULL:
        for alpha_deg in ANGLES_FULL:
            t = math.radians(theta_deg)
            a = math.radians(alpha_deg)
            dx = -math.cos(t) * math.cos(a)
            dy =  math.sin(t) * math.cos(a)
            dz =  math.sin(a)
            x  = 1000 + _L * math.cos(t) * math.cos(a)
            y  =       -_L * math.sin(t) * math.cos(a)
            z  =       -_L * math.sin(a)
            name = (f"Muon_400MeV_theta_{theta_deg}_alpha_{alpha_deg}"
                    f"_pivot_x1000_stepsize_1mm")
            tracks.append(f"{name}:{dx:.9f},{dy:.9f},{dz:.9f}:400:{x:.3f},{y:.3f},{z:.3f}")
    return tracks


def _split(lst, n):
    """Split lst into n roughly equal chunks."""
    k, r = divmod(len(lst), n)
    out, i = [], 0
    for chunk_idx in range(n):
        size = k + (1 if chunk_idx < r else 0)
        out.append(lst[i:i + size])
        i += size
    return out

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

# (param, job_label, tracks, out_dir, no_wire_response)
JOBS = [
    ("diffusion_trans_cm2_us", "pivot_ta_diag_high_trans",       TRACKS_DIAG,    OUT_DIR_DIAG,           False),
    ("diffusion_long_cm2_us",  "pivot_ta_diag_high_long",        TRACKS_DIAG,    OUT_DIR_DIAG,           False),
    ("diffusion_trans_cm2_us", "pivot_ta_offdiag_trans",         TRACKS_OFFDIAG, OUT_DIR_OFFDIAG,        False),
    ("diffusion_long_cm2_us",  "pivot_ta_offdiag_long",          TRACKS_OFFDIAG, OUT_DIR_OFFDIAG,        False),
    ("diffusion_trans_cm2_us", "pivot_ta_offdiag_nowire_trans",  TRACKS_OFFDIAG, OUT_DIR_OFFDIAG_NOWIRE, True),
    ("diffusion_long_cm2_us",  "pivot_ta_offdiag_nowire_long",   TRACKS_OFFDIAG, OUT_DIR_OFFDIAG_NOWIRE, True),
]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--submit", action="store_true",
                   help="Actually submit the job to Slurm (default: dry-run, writes job script only)")
    p.add_argument("--print-sbatch-command", action="store_true",
                   help="Print the sbatch command to stdout instead of writing to disk")
    p.add_argument("--time", default="08:00:00",
                   help="Walltime per Slurm job (default: 08:00:00)")
    p.add_argument("--num-chunks", type=int, default=1, metavar="N",
                   help="(fullgrid only) split 81 tracks into N Slurm jobs (default: 1)")
    p.add_argument("--param", choices=["diffusion_trans_cm2_us", "diffusion_long_cm2_us"],
                   default=None, help="Submit only this param (default: both)")
    p.add_argument("--group",
                   choices=["diag", "offdiag", "offdiag_nowire", "fullgrid", "all"],
                   default="all",
                   help="Track group to submit: diag, offdiag (wire ON), "
                        "offdiag_nowire (wire OFF), fullgrid (all 81 pairs), "
                        "or all (default: all)")
    args = p.parse_args()

    # ── Existing groups (diag / offdiag / offdiag_nowire) ────────────────────
    jobs = [
        (param, label, tracks, out_dir, nowire)
        for param, label, tracks, out_dir, nowire in JOBS
        if (args.param is None or param == args.param)
        and (args.group in ("all", "diag", "offdiag", "offdiag_nowire"))
        and (args.group == "all"
             or (args.group == "diag"           and "diag" in label and "offdiag" not in label)
             or (args.group == "offdiag"        and "offdiag" in label and "nowire" not in label)
             or (args.group == "offdiag_nowire" and "nowire" in label))
    ]
    for param, job_label, tracks, out_dir, nowire in jobs:
        cmd = make_gradient_command(param=param, noise_scale=1.0, noise_seeds=NOISY_SEEDS,
                                    tracks=tracks, results_dir=out_dir,
                                    no_wire_response=nowire, **COMMON_BASE)
        print(f"submitting job: {job_label}")
        s3df_submit_multi(
            [cmd],
            job_label=job_label,
            time=args.time,
            submit=args.submit,
            print_sbatch_command=args.print_sbatch_command,
            mem_gb=64,
        )

    # ── Full grid ─────────────────────────────────────────────────────────────
    if args.group in ("fullgrid", "all"):
        import glob as _glob

        def _track_names(bundle_str):
            return {s.split(":")[0] for s in bundle_str.split("+")}

        diag_names    = _track_names(TRACKS_DIAG)
        offdiag_names = _track_names(TRACKS_OFFDIAG)

        diag_dir    = os.path.expandvars(OUT_DIR_DIAG)
        offdiag_dir = os.path.expandvars(OUT_DIR_OFFDIAG)
        full_dir    = os.path.expandvars(OUT_DIR_FULLGRID)

        def _already_done(name):
            """Check if this track has results in the fullgrid, diag, or offdiag dir."""
            # Per-track pkl in the fullgrid dir (previous partial run)
            if os.path.isdir(full_dir) and _glob.glob(os.path.join(full_dir, f"*_{name}_*.pkl")):
                return f"per-track pkl in fullgrid dir"
            # Per-track or bundled pkl in the diag dir
            if name in diag_names and os.path.isdir(diag_dir):
                if _glob.glob(os.path.join(diag_dir, f"*_{name}_*.pkl")):
                    return "per-track pkl in diag dir"
                n = len(diag_names)
                if _glob.glob(os.path.join(diag_dir, f"*_{n}tracks_*.pkl")):
                    return f"covered by diag bundle ({n} tracks)"
            # Per-track or bundled pkl in the offdiag dir
            if name in offdiag_names and os.path.isdir(offdiag_dir):
                if _glob.glob(os.path.join(offdiag_dir, f"*_{name}_*.pkl")):
                    return "per-track pkl in offdiag dir"
                n = len(offdiag_names)
                if _glob.glob(os.path.join(offdiag_dir, f"*_{n}tracks_*.pkl")):
                    return f"covered by offdiag bundle ({n} tracks)"
            return None

        all_tracks = _make_fullgrid_tracks()
        new_tracks = []
        for track_str in all_tracks:
            name = track_str.split(":")[0]
            reason = _already_done(name)
            if reason:
                print(f"  [skip] {name}: {reason}")
            else:
                new_tracks.append(track_str)

        print(f"Fullgrid: {len(new_tracks)}/{len(all_tracks)} tracks to submit "
              f"({len(all_tracks) - len(new_tracks)} already done)")

        if new_tracks:
            chunks = _split(new_tracks, args.num_chunks)
            for param in ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]:
                if args.param is not None and param != args.param:
                    continue
                param_short = "trans" if "trans" in param else "long"
                for i, chunk in enumerate(chunks):
                    suffix = f"_chunk{i+1}of{args.num_chunks}" if args.num_chunks > 1 else ""
                    job_label = f"pivot_ta_fullgrid_{param_short}{suffix}"
                    # One command per track so pkls carry the track name and re-runs skip cleanly
                    cmds = [
                        make_gradient_command(
                            param=param, noise_scale=1.0, noise_seeds=NOISY_SEEDS,
                            tracks=track_spec, results_dir=OUT_DIR_FULLGRID,
                            no_wire_response=False, **COMMON_BASE,
                        )
                        for track_spec in chunk
                    ]
                    print(f"submitting job: {job_label} ({len(chunk)} tracks)")
                    s3df_submit_multi(
                        cmds,
                        job_label=job_label,
                        time=args.time,
                        submit=args.submit,
                        print_sbatch_command=args.print_sbatch_command,
                        mem_gb=64,
                    )
