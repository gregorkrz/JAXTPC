#!/usr/bin/env python3
"""Check completion of diagonal pivot theta=alpha runs in the high-angle bundled pkl directory.

Usage:
    python tools/check_diag_completion.py
    RESULTS_DIR=/path/to/results python tools/check_diag_completion.py
"""
import os
import pickle
from collections import defaultdict
from pathlib import Path

RESULTS_DIR    = os.environ.get("RESULTS_DIR", "results")
DIAG_TA_SUBDIR = "cutoff_loss_landscape_20260611_pivot_ta_diag"
DIAG_DIR       = Path(RESULTS_DIR) / "1d_gradients" / DIAG_TA_SUBDIR

PARAMS      = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
PARAM_LABELS = {"diffusion_trans_cm2_us": "D_trans", "diffusion_long_cm2_us": "D_long"}
N_SEEDS     = 50
ADC_CUTOFFS = [0, 50]
DIAG_ANGLES = [25, 30, 35, 40, 45, 50]
TRACK_NAMES = [f"Muon_400MeV_theta_{t}_alpha_{t}_pivot_x1000_stepsize_1mm" for t in DIAG_ANGLES]


def cutoff_tag(adc):
    return f"_cutoff{adc:.3g}".replace(".", "p") if adc > 0 else ""


def count_seeds(param):
    """Return {adc: {track_name: seed_count}} for the given param."""
    counts = {adc: defaultdict(int) for adc in ADC_CUTOFFS}
    for adc in ADC_CUTOFFS:
        for seed in range(N_SEEDS):
            path = DIAG_DIR / (f"sobolev_loss_geomean_log1p_N100_range0p2_{param}"
                               f"_6tracks_noise1_seed{seed}{cutoff_tag(adc)}_perplane.pkl")
            if not path.exists():
                continue
            try:
                with open(path, "rb") as f:
                    d = pickle.load(f)
                per_track = d.get("per_track_loss_values", {})
                for track in TRACK_NAMES:
                    if track in per_track:
                        counts[adc][track] += 1
            except (EOFError, pickle.UnpicklingError, Exception) as e:
                print(f"  WARNING: seed={seed} cutoff={adc}: {e}")
    return counts


def print_table(param, counts):
    label = PARAM_LABELS[param]
    print(f"\n=== {label} ===")
    header = f"  {'theta=alpha':<14}" + "".join(f"  cutoff={adc:>2d}" for adc in ADC_CUTOFFS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for angle, track in zip(DIAG_ANGLES, TRACK_NAMES):
        row = f"  {angle:<14}"
        for adc in ADC_CUTOFFS:
            n = counts[adc][track]
            row += f"  {n:>4}/{N_SEEDS}"
        print(row)


def main():
    print(f"Directory: {DIAG_DIR}")
    if not DIAG_DIR.exists():
        print("  ERROR: directory does not exist")
        return

    for param in PARAMS:
        counts = count_seeds(param)
        print_table(param, counts)
    print()


if __name__ == "__main__":
    main()
