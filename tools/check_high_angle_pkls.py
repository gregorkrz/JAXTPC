#!/usr/bin/env python3
"""Check which high-angle pivot theta×alpha pkl files exist on disk.

Checks two sources used by plot_diffusion_loss_study.py for the 'full' mode:
  1. Per-track pkls in diffusion_angle_pivot_theta_alpha/
  2. Diagonal bundled pkls in cutoff_loss_landscape_20260611_pivot_ta_diag/

Usage:
  python tools/check_high_angle_pkls.py
  RESULTS_DIR=/path/to/results python tools/check_high_angle_pkls.py
"""
import os
import pickle

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

PARAMS      = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
ADC_CUTOFFS = [0, 5, 10, 20, 50]
SEEDS       = list(range(5))

PIVOT_SUBDIR = "diffusion_angle_pivot_theta_alpha"
DIAG_SUBDIR  = "cutoff_loss_landscape_20260611_pivot_ta_diag"

HIGH_THETAS = (30, 35, 40, 45, 50)
ALL_ALPHAS  = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)


def _pkl_path(subdir, param, track_name, seed, adc):
    cutoff_tag = f"_cutoff{adc:.3g}".replace(".", "p") if adc > 0.0 else ""
    fname = (f"sobolev_loss_geomean_log1p_N100_range0p2_{param}"
             f"_{track_name}_noise1_seed{seed}{cutoff_tag}_perplane.pkl")
    return os.path.join(RESULTS_DIR, "1d_gradients", subdir, fname)


def _diag_pkl_path(param, seed, adc):
    cutoff_tag = f"_cutoff{adc:.3g}".replace(".", "p") if adc > 0.0 else ""
    fname = (f"sobolev_loss_geomean_log1p_N100_range0p2_{param}"
             f"_6tracks_noise1_seed{seed}{cutoff_tag}.pkl")
    return os.path.join(RESULTS_DIR, "1d_gradients", DIAG_SUBDIR, fname)


def check_per_track_pkls():
    print("=" * 60)
    print(f"Source 1: per-track pkls in {PIVOT_SUBDIR}/")
    print("=" * 60)
    for theta in HIGH_THETAS:
        for alpha in ALL_ALPHAS:
            track = f"Muon_400MeV_theta_{theta}_alpha_{alpha}_pivot_x1000_stepsize_1mm"
            found = missing = 0
            for param in PARAMS:
                for adc in ADC_CUTOFFS:
                    for seed in SEEDS:
                        p = _pkl_path(PIVOT_SUBDIR, param, track, seed, adc)
                        if os.path.exists(p):
                            found += 1
                        else:
                            missing += 1
            status = "OK" if found > 0 else "MISSING"
            print(f"  theta={theta:2d} alpha={alpha:2d}  [{status}]  {found} found, {missing} missing")


def check_diag_pkls():
    print()
    print("=" * 60)
    print(f"Source 2: diagonal bundled pkls in {DIAG_SUBDIR}/")
    print("=" * 60)
    for param in PARAMS[:1]:  # just check one param to find track list
        for adc in ADC_CUTOFFS[:1]:
            for seed in SEEDS[:1]:
                path = _diag_pkl_path(param, seed, adc)
                if not os.path.exists(path):
                    print(f"  {os.path.basename(path)}: NOT FOUND")
                    continue
                with open(path, "rb") as f:
                    d = pickle.load(f)
                tracks = sorted(d.get("per_track_loss_values", {}).keys())
                print(f"  {os.path.basename(path)}: {len(tracks)} tracks")
                for t in tracks:
                    print(f"    {t}")

    print()
    print("  Presence of diag pkls across params/seeds/adcs:")
    for param in PARAMS:
        for adc in ADC_CUTOFFS:
            found = sum(
                os.path.exists(_diag_pkl_path(param, seed, adc))
                for seed in SEEDS
            )
            status = "OK" if found == len(SEEDS) else f"{found}/{len(SEEDS)} seeds"
            print(f"  {param}  adc={adc}  [{status}]")


if __name__ == "__main__":
    check_per_track_pkls()
    check_diag_pkls()
