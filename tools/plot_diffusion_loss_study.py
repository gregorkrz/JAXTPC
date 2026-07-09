#!/usr/bin/env python3
"""Generate interactive HTML viewer for diffusion loss landscape studies.

Reads pkl files produced by 1d_gradients.py.  Use --mode to select which study
directories and track/angle grids to load.

Modes
-----
  full (default)
    diffusion_startx_study / diffusion_angle_study — full track & angle grids
  no_wire_response
    diffusion_startx_study_no_wire_response / diffusion_angle_study_no_wire_response
    + diffusion_angle_pivot_study_no_wire_response  (pivot angle, δ kernels)
    + diffusion_angle_theta_alpha_no_wire_response  (θ×α grid, δ kernels)
    + diffusion_angle_pivot_theta_alpha_no_wire_response  (pivot θ×α grid)
    Muon5 at x ∈ {1500, 1750, 1900} mm; θ ∈ {0, 5, 10, 20}°; 50 noise seeds

Usage:
  python tools/plot_diffusion_loss_study.py
  python tools/plot_diffusion_loss_study.py --mode no_wire_response --recompute-bias
  python tools/plot_diffusion_loss_study.py --output $PLOTS_DIR/diffusion_loss_study_no_wire_response.html
"""
import argparse
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
PLOTS_DIR   = os.environ.get("PLOTS_DIR",   "plots")

PARAMS = ["diffusion_trans_cm2_us", "diffusion_long_cm2_us"]
PARAM_LABELS = {
    "diffusion_trans_cm2_us": "D_trans (transverse)",
    "diffusion_long_cm2_us":  "D_long (longitudinal)",
}
ADC_CUTOFFS = [0, 5, 10, 20, 50]
SEEDS       = list(range(5))    # first 5 seeds → landscape view

_STARTX_TRACKS_DEF_FULL = (
    ("Muon5_100MeV",  [1900, 1800, 1750, 1700, 1600, 1500, 1000, 500, 0]),
    ("Muon12_100MeV", [1900, 1800, 1750, 1700, 1600, 1500, 1000, 500, 0]),
    ("Muon4_100MeV",  [-1900, -1800, -1750, -1700, -1600, -1500, -1000, -500, 0]),
    ("Muon10_100MeV", [-1900, -1800, -1750, -1700, -1600, -1500, -1000, -500, 0]),
)
_STARTX_TRACKS_DEF_NWR = (
    ("Muon5_100MeV", [1500, 1750, 1900]),
)
_ANGLE_THETAS_FULL = tuple(sorted(set(range(-90, 91, 20)) | {25, 15, 5, -5, -15, -25}))
_ANGLE_THETAS_NWR  = (0, 5, 10, 20)
_NWR_TA_THETAS     = (0, 5, 10, 15, 20, 25, 30, 35, 40)   # θ values for θ×α grid studies
_NWR_TA_ALPHAS     = (0, 5, 10, 15, 20, 25, 30, 35, 40)   # α values for θ×α grid studies
_PIVOT_TA_THETAS   = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)  # θ values for pivot θ×α grid (extended)
_PIVOT_TA_ALPHAS   = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)  # α values for pivot θ×α grid (extended)

# High-angle (25-50 deg) diagonal-only extension of the pivot θ×α grid, run via
# scripts/20260611/pivot_theta_alpha_landscape_diagonal.sh (local) /
# submit_pivot_theta_alpha_diagonal_high.py (Slurm). Lives outside the
# official theta_alpha_pivot_subdir as one bundled 6-track pkl per
# (param, noise seed, adc_cutoff) instead of one pkl per (theta, alpha).
# Seeds found here accumulate on top of (rather than only filling gaps in)
# the official per-(theta, alpha) pkls in _compute_theta_alpha_bias.
_DIAG_TA_SUBDIR     = "cutoff_loss_landscape_20260611_pivot_ta_diag"
_FULLGRID_TA_SUBDIR = "cutoff_loss_landscape_20260611_pivot_ta_fullgrid"

# Pivot-angle study: 400 MeV muon, midpoint fixed at (1000, 0, 0) mm (west volume).
# CSDA range of 400 MeV muon in LAr ≈ 1700.6 mm → half-length 850.3 mm.
# start = (1000 + 850.3·cos θ,  −850.3·sin θ,  0)  mm
_ANGLE_PIVOT_X_MM       = 1000.0
_ANGLE_PIVOT_HALF_LEN_MM = 850.3

BASE_TRACK_LABELS = {
    "Muon5_100MeV":  "Muon 5 (west)",
    "Muon12_100MeV": "Muon 12 (west)",
    "Muon4_100MeV":  "Muon 4 (east)",
    "Muon10_100MeV": "Muon 10 (east)",
}


@dataclass(frozen=True)
class StudyMode:
    name: str
    page_title: str
    startx_subdir: str
    angle_subdir: str
    angle_pivot_subdir: str
    startx_tracks_def: tuple
    angle_thetas: tuple
    all_seeds: int
    include_angle_pivot: bool
    output_basename: str
    bias_cache_suffix: str
    angle_bias_footnote: str
    # Optional NWR-only extras (default: not included)
    angle_pivot_nwr_subdir: str = ""
    theta_alpha_subdir: str = ""
    theta_alpha_pivot_subdir: str = ""
    ta_thetas: tuple = ()
    ta_alphas: tuple = ()
    ta_pivot_thetas: tuple = ()
    ta_pivot_alphas: tuple = ()
    include_pivot_nwr: bool = False
    include_theta_alpha: bool = False


STUDY_MODES = {
    "full": StudyMode(
        name="full",
        page_title="Diffusion Loss Landscape Studies",
        startx_subdir="diffusion_startx_study",
        angle_subdir="diffusion_angle_study",
        angle_pivot_subdir="diffusion_angle_pivot_study",
        startx_tracks_def=_STARTX_TRACKS_DEF_FULL,
        angle_thetas=_ANGLE_THETAS_FULL,
        all_seeds=100,
        include_angle_pivot=True,
        output_basename="diffusion_loss_study.html",
        bias_cache_suffix="",
        angle_bias_footnote=(
            "All tracks: 400 MeV muon starting at (1900, 0, 0) mm (west volume, near anode). "
            "Direction rotated by θ in the XY plane: dx = −cos θ, dy = sin θ, dz = 0. "
            "θ = 0° (pure drift) not included in sweep. Mean ± 1σ over 100 noise seeds."
        ),
        theta_alpha_subdir="diffusion_angle_theta_alpha",
        theta_alpha_pivot_subdir="diffusion_angle_pivot_theta_alpha",
        ta_thetas=_NWR_TA_THETAS,
        ta_alphas=_NWR_TA_ALPHAS,
        ta_pivot_thetas=_PIVOT_TA_THETAS,
        ta_pivot_alphas=_PIVOT_TA_ALPHAS,
        include_theta_alpha=True,
    ),
    "no_wire_response": StudyMode(
        name="no_wire_response",
        page_title="Diffusion Loss Studies (no wire response)",
        startx_subdir="diffusion_startx_study_no_wire_response",
        angle_subdir="diffusion_angle_study_no_wire_response",
        angle_pivot_subdir="diffusion_angle_pivot_study",
        startx_tracks_def=_STARTX_TRACKS_DEF_FULL,
        angle_thetas=_ANGLE_THETAS_FULL,
        all_seeds=50,
        include_angle_pivot=False,
        output_basename="diffusion_loss_study_no_wire_response.html",
        bias_cache_suffix="_no_wire_response",
        angle_bias_footnote=(
            "400 MeV muon at (1900, 0, 0) mm; delta kernels (no wire response). "
            "Mean ± 1σ over 50 noise seeds."
        ),
        angle_pivot_nwr_subdir="diffusion_angle_pivot_study_no_wire_response",
        theta_alpha_subdir="diffusion_angle_theta_alpha_no_wire_response",
        theta_alpha_pivot_subdir="diffusion_angle_pivot_theta_alpha_no_wire_response",
        ta_thetas=_NWR_TA_THETAS,
        ta_alphas=_NWR_TA_ALPHAS,
        ta_pivot_thetas=_PIVOT_TA_THETAS,
        ta_pivot_alphas=_PIVOT_TA_ALPHAS,
        include_pivot_nwr=True,
        include_theta_alpha=True,
    ),
}


def _base_tracks(mode):
    return [bt for bt, _ in mode.startx_tracks_def]


def _pkl_path(study_subdir, param, track_name, seed, adc_cutoff):
    range_tag  = "_range0p2"
    noise_tag  = f"_noise1_seed{seed}"
    cutoff_tag = f"_cutoff{adc_cutoff:.3g}".replace(".", "p") if adc_cutoff > 0.0 else ""
    fname = (f"sobolev_loss_geomean_log1p_N100{range_tag}_{param}"
             f"_{track_name}{noise_tag}{cutoff_tag}_perplane.pkl")
    return os.path.join(RESULTS_DIR, "1d_gradients", study_subdir, fname)


def _nonoise_pkl_path(study_subdir, param, track_name, adc_cutoff):
    """Path for no-noise pkl (noise_scale=0 → no noise_tag in filename)."""
    range_tag  = "_range0p2"
    cutoff_tag = f"_cutoff{adc_cutoff:.3g}".replace(".", "p") if adc_cutoff > 0.0 else ""
    fname = (f"sobolev_loss_geomean_log1p_N100{range_tag}_{param}"
             f"_{track_name}{cutoff_tag}_perplane.pkl")
    return os.path.join(RESULTS_DIR, "1d_gradients", study_subdir, fname)


def _load_diag_ta_pkl(param, seed, adc_cutoff):
    """Load the bundled 6-track diagonal high-angle pkl (cached per call site)."""
    cutoff_tag = f"_cutoff{adc_cutoff:.3g}".replace(".", "p") if adc_cutoff > 0.0 else ""
    fname = (f"sobolev_loss_geomean_log1p_N100_range0p2_{param}"
             f"_6tracks_noise1_seed{seed}{cutoff_tag}_perplane.pkl")
    path = os.path.join(RESULTS_DIR, "1d_gradients", _DIAG_TA_SUBDIR, fname)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        return None


def load_landscape_data(study_subdir, track_names):
    """Flat dict 'param|track|adc|seed' → {factors, loss}. Seeds 0-4 only."""
    flat = {}
    n_found = n_miss = 0
    for param in PARAMS:
        for track in track_names:
            for adc in ADC_CUTOFFS:
                for seed in SEEDS:
                    path = _pkl_path(study_subdir, param, track, seed, adc)
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            d = pickle.load(f)
                        entry = {
                            "factors": [float(x) for x in d["factors"]],
                            "loss":    [float(x) for x in d["loss_values"]],
                        }
                        ppv = d.get("per_plane_loss_values", {})
                        if ppv:
                            first_trk = next(iter(ppv))
                            entry["plane_loss"] = {
                                pname: [float(x) for x in pvals]
                                for pname, pvals in ppv[first_trk].items()
                            }
                        flat[f"{param}|{track}|{adc}|{seed}"] = entry
                        n_found += 1
                    else:
                        n_miss += 1
    print(f"  {n_found} found, {n_miss} missing")
    return flat


def load_nonoise_landscape_data(study_subdir, track_names):
    """Like load_landscape_data but for the single no-noise run (no seed dimension)."""
    flat = {}
    n_found = n_miss = 0
    for param in PARAMS:
        for track in track_names:
            for adc in ADC_CUTOFFS:
                path = _nonoise_pkl_path(study_subdir, param, track, adc)
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        d = pickle.load(f)
                    entry = {
                        "factors": [float(x) for x in d["factors"]],
                        "loss":    [float(x) for x in d["loss_values"]],
                    }
                    ppv = d.get("per_plane_loss_values", {})
                    if ppv:
                        first_trk = next(iter(ppv))
                        entry["plane_loss"] = {
                            pname: [float(x) for x in pvals]
                            for pname, pvals in ppv[first_trk].items()
                        }
                    flat[f"{param}|{track}|{adc}"] = entry
                    n_found += 1
                else:
                    n_miss += 1
    print(f"  (no-noise) {n_found} found, {n_miss} missing")
    return flat


def _load_nonoise_argmin(subdir, param, track_name, adc):
    """Return argmin factor from a no-noise pkl, or None if missing/corrupt."""
    path = _nonoise_pkl_path(subdir, param, track_name, adc)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
        factors = np.array(d["factors"])
        losses  = np.array(d["loss_values"])
        return float(factors[np.argmin(losses)])
    except (EOFError, pickle.UnpicklingError):
        return None


def _plane_group_argmins(d):
    """Return {grp: argmin_factor} for U/V/Y plane groups using geomean-log1p loss.

    Returns an empty dict when per_plane_loss_values is absent (old pkls).
    """
    ppv = d.get("per_plane_loss_values", {})
    if not ppv:
        return {}
    first_trk = next(iter(ppv))
    plane_losses = ppv[first_trk]
    factors = np.array(d["factors"])
    result = {}
    for grp in ("U", "V", "Y"):
        grp_planes = [k for k in plane_losses if k.startswith(grp)]
        if not grp_planes:
            continue
        combined = np.zeros(len(factors))
        for k in grp_planes:
            combined += np.log1p(np.array(plane_losses[k]))
        combined = np.expm1(combined / len(grp_planes))
        result[grp] = float(factors[np.argmin(combined)])
    return result


def _load_nonoise_entry(subdir, param, track_name, adc):
    """Return {"factor": f, "plane_factors": {...}} from a no-noise pkl, or None."""
    path = _nonoise_pkl_path(subdir, param, track_name, adc)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
        factors = np.array(d["factors"])
        losses  = np.array(d["loss_values"])
        entry = {"factor": float(factors[np.argmin(losses)])}
        pg = _plane_group_argmins(d)
        if pg:
            entry["plane_factors"] = pg
        return entry
    except (EOFError, pickle.UnpicklingError):
        return None


def compute_nonoise_drift_bias(nonoise_subdir, mode):
    """Argmin factor from no-noise startx pkls; keyed param|basetrack|adc|drift_dist."""
    flat = {}
    startx_map = {bt: xs for bt, xs in mode.startx_tracks_def}
    for param in PARAMS:
        for base_track, startxs in startx_map.items():
            for adc in ADC_CUTOFFS:
                for startx in startxs:
                    track_name = f"{base_track}_startx_{startx}_stepsize_1mm"
                    factor = _load_nonoise_argmin(nonoise_subdir, param, track_name, adc)
                    if factor is not None:
                        flat[f"{param}|{base_track}|{adc}|{abs(startx)}"] = {"factor": factor}
    return {"data": flat}


def compute_nonoise_angle_bias(nonoise_subdir, thetas):
    """Argmin factor from no-noise angle pkls; keyed param|adc|theta."""
    flat = {}
    for param in PARAMS:
        for adc in ADC_CUTOFFS:
            for theta in thetas:
                track_name = f"Muon_400MeV_theta_{theta}_stepsize_1mm"
                entry = _load_nonoise_entry(nonoise_subdir, param, track_name, adc)
                if entry is not None:
                    flat[f"{param}|{adc}|{theta}"] = entry
    return {"thetas": list(thetas), "data": flat}


def compute_nonoise_angle_pivot_bias(nonoise_subdir, thetas):
    """Argmin factor from no-noise pivot angle pkls; keyed param|adc|theta."""
    flat = {}
    for param in PARAMS:
        for adc in ADC_CUTOFFS:
            for theta in thetas:
                track_name = f"Muon_400MeV_theta_{theta}_pivot_x1000_stepsize_1mm"
                entry = _load_nonoise_entry(nonoise_subdir, param, track_name, adc)
                if entry is not None:
                    flat[f"{param}|{adc}|{theta}"] = entry
    return {"thetas": list(thetas), "data": flat}


def compute_nonoise_theta_alpha_bias(nonoise_subdir, thetas, alphas, pivot):
    """Argmin factor from no-noise theta×alpha pkls; keyed param|adc|theta|alpha."""
    flat = {}
    for param in PARAMS:
        for adc in ADC_CUTOFFS:
            for theta in thetas:
                for alpha in alphas:
                    if pivot:
                        track_name = (f"Muon_400MeV_theta_{theta}_alpha_{alpha}"
                                      f"_pivot_x1000_stepsize_1mm")
                    else:
                        track_name = f"Muon_400MeV_theta_{theta}_alpha_{alpha}_stepsize_1mm"
                    entry = _load_nonoise_entry(nonoise_subdir, param, track_name, adc)
                    if entry is not None:
                        flat[f"{param}|{adc}|{theta}|{alpha}"] = entry
    return {"thetas": list(thetas), "alphas": list(alphas), "data": flat}


def _compute_drift_bias(mode, verbose=True):
    """Read individual startx pkl files and aggregate argmin-factor per seed.

    Returns the bias dict (same structure as the cache file).
    """
    flat = {}
    n_found = n_miss = 0
    startx_map = {bt: xs for bt, xs in mode.startx_tracks_def}
    all_seeds = list(range(mode.all_seeds))
    n_positions = sum(len(xs) for xs in startx_map.values())
    n_total = len(PARAMS) * n_positions * len(ADC_CUTOFFS)
    done = 0

    for param in PARAMS:
        for base_track, startxs in startx_map.items():
            for adc in ADC_CUTOFFS:
                if verbose:
                    print(f"  {param}  |  {base_track}  |  adc={adc}")
                for startx in startxs:
                    done += 1
                    track_name = f"{base_track}_startx_{startx}_stepsize_1mm"
                    drift_dist = abs(startx)
                    seed_factors = []  # list of (seed_idx, factor)
                    for seed in all_seeds:
                        path = _pkl_path(mode.startx_subdir, param, track_name, seed, adc)
                        if os.path.exists(path):
                            try:
                                with open(path, "rb") as f:
                                    d = pickle.load(f)
                                factors = np.array(d["factors"])
                                losses  = np.array(d["loss_values"])
                                seed_factors.append((seed, float(factors[np.argmin(losses)])))
                                n_found += 1
                            except (EOFError, pickle.UnpicklingError):
                                n_miss += 1
                        else:
                            n_miss += 1
                    if verbose:
                        print(f"    x={startx:+5d}  [{done:3d}/{n_total}]  "
                              f"{len(seed_factors)}/{len(all_seeds)} seeds found")
                    if seed_factors:
                        arr = np.array([v for _, v in seed_factors])
                        flat[f"{param}|{base_track}|{adc}|{drift_dist}"] = {
                            "mean":   float(arr.mean()),
                            "std":    float(arr.std()),
                            "n":      len(seed_factors),
                            "vals": [[s, round(v, 6)] for s, v in seed_factors],
                        }

    print(f"  Drift bias done: {n_found} pkls read, {n_miss} missing")
    base_tracks = _base_tracks(mode)
    return {
        "baseTracks":      base_tracks,
        "baseTrackLabels": {bt: BASE_TRACK_LABELS[bt] for bt in base_tracks},
        "driftDists": {
            bt: sorted([abs(x) for x in xs])
            for bt, xs in startx_map.items()
        },
        "data": flat,
    }


def load_bias_data(mode, cache_path=None, recompute=False):
    """Load drift bias from cache pkl if available, otherwise compute and save."""
    if cache_path and not recompute and os.path.exists(cache_path):
        print(f"  Loading drift bias cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    result = _compute_drift_bias(mode, verbose=True)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved drift bias cache → {cache_path}")
    return result


def _compute_angle_bias(mode, verbose=True):
    """Read individual angle pkl files and aggregate argmin-factor per seed."""
    flat = {}
    n_found = n_miss = 0
    all_seeds = list(range(mode.all_seeds))
    n_total = len(PARAMS) * len(ADC_CUTOFFS) * len(mode.angle_thetas)
    done = 0

    for param in PARAMS:
        for adc in ADC_CUTOFFS:
            if verbose:
                print(f"  {param}  |  adc={adc}")
            for theta in mode.angle_thetas:
                done += 1
                track_name = f"Muon_400MeV_theta_{theta}_stepsize_1mm"
                seed_factors = []  # list of (seed_idx, factor)
                plane_seed_factors = {}
                for seed in all_seeds:
                    path = _pkl_path(mode.angle_subdir, param, track_name, seed, adc)
                    if os.path.exists(path):
                        try:
                            with open(path, "rb") as f:
                                d = pickle.load(f)
                            factors = np.array(d["factors"])
                            losses  = np.array(d["loss_values"])
                            seed_factors.append((seed, float(factors[np.argmin(losses)])))
                            for grp, gf in _plane_group_argmins(d).items():
                                plane_seed_factors.setdefault(grp, []).append((seed, gf))
                            n_found += 1
                        except (EOFError, pickle.UnpicklingError):
                            n_miss += 1
                    else:
                        n_miss += 1
                if verbose:
                    print(f"    theta={theta:+4d}°  [{done:3d}/{n_total}]  "
                          f"{len(seed_factors)}/{len(all_seeds)} seeds found")
                if seed_factors:
                    arr = np.array([v for _, v in seed_factors])
                    entry = {
                        "mean":   float(arr.mean()),
                        "std":    float(arr.std()),
                        "n":      len(seed_factors),
                        "vals": [[s, round(v, 6)] for s, v in seed_factors],
                    }
                    if plane_seed_factors:
                        entry["plane_vals"] = {
                            grp: [[s, round(v, 6)] for s, v in psf]
                            for grp, psf in plane_seed_factors.items()
                        }
                    flat[f"{param}|{adc}|{theta}"] = entry

    print(f"  Angle bias done: {n_found} pkls read, {n_miss} missing")
    return {
        "thetas": list(mode.angle_thetas),
        "data":   flat,
    }


def load_angle_bias_data(mode, cache_path=None, recompute=False):
    """Load start-angle bias from cache pkl if available, otherwise compute and save."""
    if cache_path and not recompute and os.path.exists(cache_path):
        print(f"  Loading angle bias cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    result = _compute_angle_bias(mode, verbose=True)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved angle bias cache → {cache_path}")
    return result


def _compute_angle_pivot_bias(mode, verbose=True):
    """Like _compute_angle_bias but track midpoint is fixed at (_ANGLE_PIVOT_X_MM, 0, 0)."""
    flat = {}
    n_found = n_miss = 0
    all_seeds = list(range(mode.all_seeds))
    n_total = len(PARAMS) * len(ADC_CUTOFFS) * len(mode.angle_thetas)
    done = 0

    for param in PARAMS:
        for adc in ADC_CUTOFFS:
            if verbose:
                print(f"  {param}  |  adc={adc}")
            for theta in mode.angle_thetas:
                done += 1
                track_name = f"Muon_400MeV_theta_{theta}_pivot_x1000_stepsize_1mm"
                seed_factors = []
                plane_seed_factors = {}
                for seed in all_seeds:
                    path = _pkl_path(mode.angle_pivot_subdir, param, track_name, seed, adc)
                    if os.path.exists(path):
                        try:
                            with open(path, "rb") as f:
                                d = pickle.load(f)
                            factors = np.array(d["factors"])
                            losses  = np.array(d["loss_values"])
                            seed_factors.append((seed, float(factors[np.argmin(losses)])))
                            for grp, gf in _plane_group_argmins(d).items():
                                plane_seed_factors.setdefault(grp, []).append((seed, gf))
                            n_found += 1
                        except (EOFError, pickle.UnpicklingError):
                            n_miss += 1
                    else:
                        n_miss += 1
                if verbose:
                    print(f"    theta={theta:+4d}°  [{done:3d}/{n_total}]  "
                          f"{len(seed_factors)}/{len(all_seeds)} seeds found")
                if seed_factors:
                    arr = np.array([v for _, v in seed_factors])
                    entry = {
                        "mean":   float(arr.mean()),
                        "std":    float(arr.std()),
                        "n":      len(seed_factors),
                        "vals": [[s, round(v, 6)] for s, v in seed_factors],
                    }
                    if plane_seed_factors:
                        entry["plane_vals"] = {
                            grp: [[s, round(v, 6)] for s, v in psf]
                            for grp, psf in plane_seed_factors.items()
                        }
                    flat[f"{param}|{adc}|{theta}"] = entry

    print(f"  Angle-pivot bias done: {n_found} pkls read, {n_miss} missing")
    return {
        "thetas": list(mode.angle_thetas),
        "data":   flat,
    }


def load_angle_pivot_bias_data(mode, cache_path=None, recompute=False):
    """Load pivot-angle bias from cache pkl if available, otherwise compute and save."""
    if cache_path and not recompute and os.path.exists(cache_path):
        print(f"  Loading angle-pivot bias cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    result = _compute_angle_pivot_bias(mode, verbose=True)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved angle-pivot bias cache → {cache_path}")
    return result


def _compute_angle_pivot_nwr_bias(mode, verbose=True):
    """Like _compute_angle_pivot_bias but reads from angle_pivot_nwr_subdir."""
    flat = {}
    n_found = n_miss = 0
    all_seeds = list(range(mode.all_seeds))
    n_total = len(PARAMS) * len(ADC_CUTOFFS) * len(mode.angle_thetas)
    done = 0

    for param in PARAMS:
        for adc in ADC_CUTOFFS:
            if verbose:
                print(f"  {param}  |  adc={adc}")
            for theta in mode.angle_thetas:
                done += 1
                track_name = f"Muon_400MeV_theta_{theta}_pivot_x1000_stepsize_1mm"
                seed_factors = []
                plane_seed_factors = {}
                for seed in all_seeds:
                    path = _pkl_path(mode.angle_pivot_nwr_subdir, param, track_name, seed, adc)
                    if os.path.exists(path):
                        try:
                            with open(path, "rb") as f:
                                d = pickle.load(f)
                            factors = np.array(d["factors"])
                            losses  = np.array(d["loss_values"])
                            seed_factors.append((seed, float(factors[np.argmin(losses)])))
                            for grp, gf in _plane_group_argmins(d).items():
                                plane_seed_factors.setdefault(grp, []).append((seed, gf))
                            n_found += 1
                        except (EOFError, pickle.UnpicklingError):
                            n_miss += 1
                    else:
                        n_miss += 1
                if verbose:
                    print(f"    theta={theta:+4d}°  [{done:3d}/{n_total}]  "
                          f"{len(seed_factors)}/{len(all_seeds)} seeds found")
                if seed_factors:
                    arr = np.array([v for _, v in seed_factors])
                    entry = {
                        "mean":   float(arr.mean()),
                        "std":    float(arr.std()),
                        "n":      len(seed_factors),
                        "vals": [[s, round(v, 6)] for s, v in seed_factors],
                    }
                    if plane_seed_factors:
                        entry["plane_vals"] = {
                            grp: [[s, round(v, 6)] for s, v in psf]
                            for grp, psf in plane_seed_factors.items()
                        }
                    flat[f"{param}|{adc}|{theta}"] = entry

    print(f"  Angle-pivot NWR bias done: {n_found} pkls read, {n_miss} missing")
    return {
        "thetas": list(mode.angle_thetas),
        "data":   flat,
    }


def load_angle_pivot_nwr_bias_data(mode, cache_path=None, recompute=False):
    if cache_path and not recompute and os.path.exists(cache_path):
        print(f"  Loading angle-pivot-NWR bias cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    result = _compute_angle_pivot_nwr_bias(mode, verbose=True)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved angle-pivot-NWR bias cache → {cache_path}")
    return result


def _compute_theta_alpha_bias(mode, subdir, pivot, verbose=True, diagonal_only=False,
                              skip_keys=None):
    """Compute argmin-factor bias over the θ×α grid.

    pivot=True  → track name includes _pivot_x1000
    pivot=False → track name is the plain theta/alpha variant
    diagonal_only=True → only process pairs where theta == alpha
    skip_keys → set of "param|adc|theta|alpha" strings to skip (already cached)
    """
    flat = {}
    n_found = n_miss = n_skipped = 0
    all_seeds = list(range(mode.all_seeds))
    thetas = mode.ta_pivot_thetas if pivot else mode.ta_thetas
    alphas = mode.ta_pivot_alphas if pivot else mode.ta_alphas
    pairs = [(t, a) for t in thetas for a in alphas if not diagonal_only or t == a]
    n_total = len(PARAMS) * len(ADC_CUTOFFS) * len(pairs)
    done = 0

    for param in PARAMS:
        for adc in ADC_CUTOFFS:
            if verbose:
                print(f"  {param}  |  adc={adc}")
            for theta, alpha in pairs:
                done += 1
                if skip_keys and f"{param}|{adc}|{theta}|{alpha}" in skip_keys:
                    n_skipped += 1
                    continue
                if pivot:
                    track_name = (f"Muon_400MeV_theta_{theta}_alpha_{alpha}"
                                  f"_pivot_x1000_stepsize_1mm")
                else:
                    track_name = f"Muon_400MeV_theta_{theta}_alpha_{alpha}_stepsize_1mm"
                seed_factors = []
                plane_seed_factors = {}
                for seed in all_seeds:
                    sources = []  # list of (seed_label, d, factors, losses)
                    path = _pkl_path(subdir, param, track_name, seed, adc)
                    if os.path.exists(path):
                        try:
                            with open(path, "rb") as f:
                                d = pickle.load(f)
                            sources.append((seed, d, np.array(d["factors"]), np.array(d["loss_values"])))
                        except (EOFError, pickle.UnpicklingError):
                            pass
                    # Extra sources accumulate on top of the primary per-track pkl —
                    # the same (theta, alpha) may have been run multiple times.
                    # Both sources below have wire response ON and must only feed
                    # the "full" (wire-response) bias, never NWR.
                    if pivot and mode.name == "full":
                        # Bundled 6-track diagonal pkl.
                        d = _load_diag_ta_pkl(param, seed, adc)
                        if d is not None and track_name in d.get("per_track_loss_values", {}):
                            sources.append((f"diag{seed}", d,
                                            np.array(d["factors"]),
                                            np.array(d["per_track_loss_values"][track_name])))
                        # Per-track pkls from the full 81-pair grid job.
                        fg_path = _pkl_path(_FULLGRID_TA_SUBDIR, param, track_name, seed, adc)
                        if os.path.exists(fg_path):
                            try:
                                with open(fg_path, "rb") as f:
                                    fg_d = pickle.load(f)
                                sources.append((f"fg{seed}", fg_d,
                                                np.array(fg_d["factors"]),
                                                np.array(fg_d["loss_values"])))
                            except (EOFError, pickle.UnpicklingError):
                                pass
                    if sources:
                        for seed_label, d, factors, losses in sources:
                            seed_factors.append((seed_label, float(factors[np.argmin(losses)])))
                            for grp, gf in _plane_group_argmins(d).items():
                                plane_seed_factors.setdefault(grp, []).append((seed_label, gf))
                        n_found += len(sources)
                    else:
                        n_miss += 1
                if verbose:
                    print(f"    θ={theta:+3d}° α={alpha:+3d}°  [{done:3d}/{n_total}]  "
                          f"{len(seed_factors)}/{len(all_seeds)} seeds found")
                if seed_factors:
                    arr = np.array([v for _, v in seed_factors])
                    entry = {
                        "mean":   float(arr.mean()),
                        "std":    float(arr.std()),
                        "n":      len(seed_factors),
                        "vals": [[s, round(v, 6)] for s, v in seed_factors],
                    }
                    if plane_seed_factors:
                        entry["plane_vals"] = {
                            grp: [[s, round(v, 6)] for s, v in psf]
                            for grp, psf in plane_seed_factors.items()
                        }
                    flat[f"{param}|{adc}|{theta}|{alpha}"] = entry

    tag = "pivot-" if pivot else ""
    print(f"  Theta-alpha {tag}bias done: {n_found} pkls read, {n_miss} missing"
          + (f", {n_skipped} skipped (cached)" if n_skipped else ""))
    return {
        "thetas": list(thetas),
        "alphas": list(alphas),
        "data":   flat,
    }


def load_theta_alpha_bias_data(mode, pivot, cache_path=None, recompute=False,
                               diagonal_only=False):
    subdir = mode.theta_alpha_pivot_subdir if pivot else mode.theta_alpha_subdir
    tag = "pivot-" if pivot else ""

    existing = None
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading theta-alpha {tag}bias cache: {cache_path}")
        with open(cache_path, "rb") as f:
            existing = pickle.load(f)

    if existing is not None and not recompute:
        # Skip entries already in the cache (diagonal_only still recomputes all diagonal pairs).
        skip = None if diagonal_only else set(existing["data"].keys())
        result = _compute_theta_alpha_bias(mode, subdir, pivot, verbose=True,
                                           diagonal_only=diagonal_only, skip_keys=skip)
        if result["data"]:
            existing["data"].update(result["data"])
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(existing, f)
                print(f"  Updated theta-alpha {tag}bias cache → {cache_path}")
        return existing

    # recompute=True or no cache yet: compute everything.
    result = _compute_theta_alpha_bias(mode, subdir, pivot, verbose=True,
                                       diagonal_only=diagonal_only)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved theta-alpha {tag}bias cache → {cache_path}")
    return result


def build_data(mode, drift_bias_cache=None, angle_bias_cache=None,
               angle_pivot_bias_cache=None, angle_pivot_nwr_bias_cache=None,
               theta_alpha_bias_cache=None, theta_alpha_pivot_bias_cache=None,
               recompute_bias=False, diagonal_only=False, theta_alpha_only=False):
    startx_tracks = [
        f"{base}_startx_{x}_stepsize_1mm"
        for base, xs in mode.startx_tracks_def
        for x in xs
    ]
    startx_track_labels = {
        f"{base}_startx_{x}_stepsize_1mm": f"{base}  x={x:+d} mm"
        for base, xs in mode.startx_tracks_def
        for x in xs
    }

    angle_tracks = [
        f"Muon_400MeV_theta_{theta}_stepsize_1mm"
        for theta in mode.angle_thetas
    ]
    angle_track_labels = {
        f"Muon_400MeV_theta_{theta}_stepsize_1mm": f"θ = {theta:+d}°"
        for theta in mode.angle_thetas
    }

    print(f"Loading {mode.startx_subdir} (landscape) …")
    startx_data = load_landscape_data(mode.startx_subdir, startx_tracks)
    print(f"Loading {mode.angle_subdir} (landscape) …")
    angle_data  = load_landscape_data(mode.angle_subdir, angle_tracks)

    # No-noise landscape overlays
    nn_startx_subdir = mode.startx_subdir + "_nonoise"
    nn_angle_subdir  = mode.angle_subdir  + "_nonoise"
    print(f"Loading {nn_startx_subdir} (no-noise landscape) …")
    nn_startx_data = load_nonoise_landscape_data(nn_startx_subdir, startx_tracks)
    print(f"Loading {nn_angle_subdir} (no-noise landscape) …")
    nn_angle_data  = load_nonoise_landscape_data(nn_angle_subdir, angle_tracks)

    print(f"Drift bias (all {mode.all_seeds} seeds) …")
    bias_data = load_bias_data(mode, cache_path=drift_bias_cache,
                               recompute=recompute_bias and not theta_alpha_only)
    print(f"Angle bias (all {mode.all_seeds} seeds) …")
    angle_bias_data = load_angle_bias_data(mode, cache_path=angle_bias_cache,
                                           recompute=recompute_bias and not theta_alpha_only)

    # No-noise bias reference
    print(f"No-noise drift bias …")
    nn_bias = compute_nonoise_drift_bias(nn_startx_subdir, mode)
    print(f"No-noise angle bias …")
    nn_angle_bias = compute_nonoise_angle_bias(nn_angle_subdir, mode.angle_thetas)

    result = {
        "startx": {
            "tracks":        startx_tracks,
            "trackLabels":   startx_track_labels,
            "data":          startx_data,
            "nonoise_data":  nn_startx_data,
        },
        "angle": {
            "tracks":        angle_tracks,
            "trackLabels":   angle_track_labels,
            "data":          angle_data,
            "nonoise_data":  nn_angle_data,
        },
        "bias":             bias_data,
        "angle_bias":       angle_bias_data,
        "nonoise_bias":     nn_bias,
        "nonoise_angle_bias": nn_angle_bias,
    }
    if mode.include_angle_pivot:
        print(f"Angle-pivot bias (all {mode.all_seeds} seeds) …")
        result["angle_pivot_bias"] = load_angle_pivot_bias_data(
            mode, cache_path=angle_pivot_bias_cache,
            recompute=recompute_bias and not theta_alpha_only)
        nn_pivot_subdir = mode.angle_pivot_subdir + "_nonoise"
        print(f"No-noise angle-pivot bias …")
        result["nonoise_angle_pivot_bias"] = compute_nonoise_angle_pivot_bias(
            nn_pivot_subdir, mode.angle_thetas)
    if mode.include_pivot_nwr:
        print(f"Angle-pivot NWR bias (all {mode.all_seeds} seeds) …")
        result["angle_pivot_nwr_bias"] = load_angle_pivot_nwr_bias_data(
            mode, cache_path=angle_pivot_nwr_bias_cache,
            recompute=recompute_bias and not theta_alpha_only)
        nn_pivot_nwr_subdir = mode.angle_pivot_nwr_subdir + "_nonoise"
        print(f"No-noise angle-pivot-NWR bias …")
        result["nonoise_angle_pivot_nwr_bias"] = compute_nonoise_angle_pivot_bias(
            nn_pivot_nwr_subdir, mode.angle_thetas)
    if mode.include_theta_alpha:
        print(f"Theta-alpha bias (all {mode.all_seeds} seeds) …")
        result["theta_alpha_bias"] = load_theta_alpha_bias_data(
            mode, pivot=False, cache_path=theta_alpha_bias_cache, recompute=recompute_bias,
            diagonal_only=diagonal_only)
        print(f"Theta-alpha pivot bias (all {mode.all_seeds} seeds) …")
        result["theta_alpha_pivot_bias"] = load_theta_alpha_bias_data(
            mode, pivot=True, cache_path=theta_alpha_pivot_bias_cache, recompute=recompute_bias,
            diagonal_only=diagonal_only)
        nn_ta_subdir       = mode.theta_alpha_subdir       + "_nonoise"
        nn_ta_pivot_subdir = mode.theta_alpha_pivot_subdir + "_nonoise"
        print(f"No-noise θ×α bias …")
        result["nonoise_theta_alpha_bias"] = compute_nonoise_theta_alpha_bias(
            nn_ta_subdir, mode.ta_thetas, mode.ta_alphas, pivot=False)
        result["nonoise_theta_alpha_pivot_bias"] = compute_nonoise_theta_alpha_bias(
            nn_ta_pivot_subdir, mode.ta_pivot_thetas, mode.ta_pivot_alphas, pivot=True)
    return result


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__PAGE_TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #f0f2f5; color: #1a1a1a; }
header { background: #1a1a2e; color: #eee; padding: 14px 24px; display: flex; align-items: baseline; gap: 1rem; }
header h1 { font-size: 1.1rem; font-weight: 600; letter-spacing: 0.04em; margin: 0; }
.tabs { display: flex; background: #16213e; padding: 0 24px; gap: 0; }
.tab-btn { padding: 10px 22px; border: none; cursor: pointer; background: transparent;
           color: #8899aa; font-size: 0.88rem; font-weight: 500;
           border-bottom: 3px solid transparent; transition: color .15s; }
.tab-btn.active { color: #fff; border-bottom-color: #e94560; }
.tab-btn:hover:not(.active) { color: #ccd; }
.tab-pane { display: none; padding: 20px 24px 24px; }
.tab-pane.active { display: block; }
.controls { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 16px;
            align-items: flex-end; }
.ctrl-group { display: flex; flex-direction: column; gap: 4px; }
.ctrl-group label { font-size: 0.7rem; color: #667; text-transform: uppercase;
                    letter-spacing: 0.06em; font-weight: 600; }
select { padding: 7px 10px; border: 1px solid #d0d5dd; border-radius: 6px;
         background: #fff; font-size: 0.87rem; min-width: 190px;
         box-shadow: 0 1px 2px rgba(0,0,0,.06); }
select:focus { outline: none; border-color: #e94560;
               box-shadow: 0 0 0 3px rgba(233,69,96,.12); }
.seg-group { display: flex; flex-direction: column; gap: 4px; }
.seg-btns { display: flex; gap: 0; }
.seg-btns button { padding: 7px 14px; border: 1px solid #d0d5dd; background: #fff;
                   font-size: 0.87rem; cursor: pointer; color: #444; transition: all .12s; }
.seg-btns button:first-child { border-radius: 6px 0 0 6px; }
.seg-btns button:last-child  { border-radius: 0 6px 6px 0; }
.seg-btns button:not(:first-child) { border-left: none; }
.seg-btns button.active { background: #e94560; color: #fff; border-color: #e94560; }
.seg-btns button:hover:not(.active) { background: #f8f0f2; }
.plot-wrap { background: #fff; border-radius: 8px;
             box-shadow: 0 1px 4px rgba(0,0,0,.09); overflow: hidden; }
.no-data { padding: 60px; text-align: center; color: #aab; font-size: 0.9rem;
           font-style: italic; }
.ds-btn { padding: 5px 12px; border: 1px solid #3a3a5a; background: transparent;
          color: #aac4ff; font-size: 0.82rem; cursor: pointer; border-radius: 4px;
          transition: all .12s; }
.ds-btn.active { background: #e94560; color: #fff; border-color: #e94560; }
.ds-btn:hover:not(.active) { background: rgba(255,255,255,0.1); }
</style>
</head>
<body>

<header>
  <h1>__PAGE_TITLE__</h1>
  <a href="diffusion_study_tracks/index.html" style="font-size:0.82rem;color:#fff;background:#2a4a7f;border:1px solid #4a7abf;border-radius:5px;padding:5px 12px;margin-left:1.2rem;text-decoration:none;vertical-align:middle;white-space:nowrap" target="_blank">&#128065; 3D track viewer &rarr;</a>
  __SIBLING_LINKS__
  __METRIC_PICKER__
  __DATASET_TOGGLE__
</header>

<div class="tabs">
  <button class="tab-btn active" data-mode="both" onclick="switchTab('startx',this)">Start-X Landscape</button>
  <button class="tab-btn"        data-mode="both" onclick="switchTab('angle',this)">Angle Landscape</button>
  <button class="tab-btn"        data-mode="both" onclick="switchTab('bias',this)">Bias vs |start_x|</button>
  <button class="tab-btn"        data-mode="both" onclick="switchTab('anglebias',this)">Bias vs Angle</button>
  __ANGLE_PIVOT_TAB_BTN__
  __PIVOT_NWR_TAB_BTN__
  __TA_TAB_BTN__
  __TA_PIVOT_TAB_BTN__
</div>

<!-- ── Tab 1: start-x landscape ─────────────────────────────────────────── -->
<div id="pane-startx" class="tab-pane active">
  <div class="controls">
    <div class="ctrl-group">
      <label>Parameter</label>
      <select id="sx-param" onchange="updateLandscape('startx')"></select>
    </div>
    <div class="ctrl-group">
      <label>ADC cutoff</label>
      <select id="sx-adc" onchange="updateLandscape('startx')"></select>
    </div>
    <div class="ctrl-group">
      <label>Track / start-x</label>
      <select id="sx-track" onchange="updateLandscape('startx')"></select>
    </div>
    <div class="ctrl-group">
      <label>Wireplane group</label>
      <select id="sx-plane-group" onchange="updateLandscape('startx')">
        <option value="all">All planes</option>
        <option value="U">U planes</option>
        <option value="V">V planes</option>
        <option value="Y">Y planes</option>
      </select>
    </div>
  </div>
  <div class="plot-wrap">
    <div id="plot-startx" style="height:520px"></div>
  </div>
</div>

<!-- ── Tab 2: angle landscape ────────────────────────────────────────────── -->
<div id="pane-angle" class="tab-pane">
  <div class="controls">
    <div class="ctrl-group">
      <label>Parameter</label>
      <select id="an-param" onchange="updateLandscape('angle')"></select>
    </div>
    <div class="ctrl-group">
      <label>ADC cutoff</label>
      <select id="an-adc" onchange="updateLandscape('angle')"></select>
    </div>
    <div class="ctrl-group">
      <label>Track angle</label>
      <select id="an-track" onchange="updateLandscape('angle')"></select>
    </div>
    <div class="ctrl-group">
      <label>Wireplane group</label>
      <select id="an-plane-group" onchange="updateLandscape('angle')">
        <option value="all">All planes</option>
        <option value="U">U planes</option>
        <option value="V">V planes</option>
        <option value="Y">Y planes</option>
      </select>
    </div>
  </div>
  <div class="plot-wrap">
    <div id="plot-angle" style="height:520px"></div>
  </div>
</div>

<!-- ── Tab 3: bias vs |start_x| (distance from cathode) ─────────────────── -->
<div id="pane-bias" class="tab-pane">
  <div class="controls">
    <div class="ctrl-group">
      <label>Parameter</label>
      <select id="bi-param" onchange="updateBias()"></select>
    </div>
    <div class="ctrl-group">
      <label>ADC cutoff</label>
      <select id="bi-adc" onchange="updateBias()"></select>
    </div>
    <div class="seg-group">
      <label>View</label>
      <div class="seg-btns">
        <button id="bi-view-combined" class="active" onclick="setBiasView('combined')">All tracks combined</button>
        <button id="bi-view-grid"                    onclick="setBiasView('grid')">2×2 per-track grid</button>
      </div>
    </div>
    <div class="seg-group">
      <label>Histogram x-axis</label>
      <div class="seg-btns">
        <button id="bi-xaxis-auto"  class="active" onclick="setSeedHistSameX(false)">Auto</button>
        <button id="bi-xaxis-fixed"               onclick="setSeedHistSameX(true)">Same range</button>
      </div>
    </div>
  </div>
  <div class="plot-wrap">
    <div id="plot-bias" style="height:560px"></div>
  </div>
</div>

<!-- ── Tab 4: bias vs angle ───────────────────────────────────────────────── -->
<div id="pane-anglebias" class="tab-pane">
  <div class="controls">
    <div class="ctrl-group">
      <label>Parameter</label>
      <select id="ab-param" onchange="updateAngleBias()"></select>
    </div>
    <div class="ctrl-group">
      <label>ADC cutoff</label>
      <select id="ab-adc" onchange="updateAngleBias()"></select>
    </div>
    <div class="ctrl-group">
      <label>Wireplane group</label>
      <select id="ab-plane-group" onchange="updateAngleBias()">
        <option value="all">All planes</option>
        <option value="U">U planes</option>
        <option value="V">V planes</option>
        <option value="Y">Y planes</option>
      </select>
    </div>
  </div>
  <div class="plot-wrap">
    <div id="plot-anglebias" style="height:520px"></div>
  </div>
  <div id="table-anglebias" style="margin-top:14px; overflow-x:auto;"></div>
  <p style="margin-top:10px; font-size:0.8rem; color:#888; padding: 0 4px;">
    __ANGLE_BIAS_FOOTNOTE__
  </p>
</div>

__ANGLE_PIVOT_PANE__
__PIVOT_NWR_PANE__
__TA_PANE__
__TA_PIVOT_PANE__

<div id="seed-hist-wrap" style="display:none;margin:10px 0;padding:8px 14px;background:#f5f7ff;border:1px solid #c8d0e8;border-radius:6px;">
  <span id="seed-hist-title" style="font-size:0.8rem;color:#446;font-weight:600;"></span>
  <div id="seed-hist-plot" style="height:150px;margin-top:4px;"></div>
</div>

<script>
__DATA_DECL__
const PARAMS      = __PARAMS_JSON__;
const PARAM_LABELS = __PARAM_LABELS_JSON__;
const ADC_CUTOFFS = __ADC_CUTOFFS_JSON__;
const SEEDS       = __SEEDS_JSON__;
__BIAS_SEED_LABEL_DECL__

// ── colours ──────────────────────────────────────────────────────────────
const SEED_COLORS = [
  'rgba(66,133,244,0.50)',
  'rgba(52,168,83,0.50)',
  'rgba(234,67,53,0.50)',
  'rgba(251,188,5,0.60)',
  'rgba(137,78,202,0.50)',
];
const TRACK_COLORS = {
  'Muon5_100MeV':  '#1f77b4',
  'Muon12_100MeV': '#ff7f0e',
  'Muon4_100MeV':  '#2ca02c',
  'Muon10_100MeV': '#d62728',
};
const NWR_SEED_COLORS = [
  'rgba(255,120,0,0.45)',
  'rgba(20,180,100,0.45)',
  'rgba(160,0,180,0.45)',
  'rgba(0,180,210,0.55)',
  'rgba(180,130,0,0.45)',
];
const NWR_MEAN_COLOR = 'rgba(200,80,0,0.92)';

let biasView = 'combined';
let taSameXAxis = {ta: false, tapivot: false};
let taXRange    = {ta: null, tapivot: null};
let taHistLast  = {ta: {vals: null, title: null}, tapivot: {vals: null, title: null}};
let seedHistSameX = false;
let seedHistXRange = null;
let seedHistLast = {vals: null, title: null};

// ── state persistence (hash) ──────────────────────────────────────────────
let _loadingState = false;
const _STATE_IDS = ['sx-param','sx-adc','sx-track','sx-plane-group','an-param','an-adc','an-track','an-plane-group',
                    'bi-param','bi-adc','ab-param','ab-adc','ab-plane-group','ap-param','ap-adc','ap-plane-group',
                    'ta-param','ta-adc','ta-plane-group','tap-param','tap-adc','tap-plane-group','global-metric'];
function saveState() {
  if (_loadingState) return;
  const p = new URLSearchParams();
  const ap = document.querySelector('.tab-pane.active');
  if (ap) p.set('tab', ap.id.replace('pane-', ''));
  if (typeof currentDataset !== 'undefined') p.set('ds', currentDataset);
  _STATE_IDS.forEach(id => { const el = document.getElementById(id); if (el) p.set(id, el.value); });
  p.set('bi-view', biasView);
  history.replaceState(null, '', '#' + p.toString());
}
function loadState() {
  const hash = location.hash.slice(1);
  if (!hash) { updateLandscape('startx'); return; }
  const p = new URLSearchParams(hash);
  _loadingState = true;
  if (typeof setDataset === 'function' && p.has('ds')) setDataset(p.get('ds'));
  _STATE_IDS.forEach(id => { const el = document.getElementById(id); if (el && p.has(id)) el.value = p.get(id); });
  if (p.has('bi-view')) {
    biasView = p.get('bi-view');
    document.getElementById('bi-view-combined').classList.toggle('active', biasView === 'combined');
    document.getElementById('bi-view-grid').classList.toggle('active', biasView === 'grid');
  }
  _loadingState = false;
  if (p.has('tab')) {
    const tabName = p.get('tab');
    let btn = null;
    document.querySelectorAll('.tab-btn').forEach(b => {
      if ((b.getAttribute('onclick')||'').includes("'" + tabName + "'")) btn = b;
    });
    if (btn && getComputedStyle(btn).display !== 'none') { switchTab(tabName, btn); return; }
  }
  updateLandscape('startx');
}

// ── helpers ───────────────────────────────────────────────────────────────
__SET_DATASET_FN__
function switchTab(study, btn) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('pane-' + study).classList.add('active');
  btn.classList.add('active');
  if (study === 'startx' || study === 'angle') updateLandscape(study);
  else if (study === 'bias') updateBias();
  else if (study === 'anglebias') updateAngleBias();
  else if (study === 'anglepivot') updateAnglePivotCombined();
  else if (study === 'ta') updateThetaAlphaBias('ta');
  else if (study === 'tapivot') updateThetaAlphaBias('tapivot');
  saveState();
}

function _renderActiveTab() {
  const ap = document.querySelector('.tab-pane.active');
  if (!ap) return;
  const s = ap.id.replace('pane-', '');
  if      (s === 'startx')     updateLandscape('startx');
  else if (s === 'angle')      updateLandscape('angle');
  else if (s === 'bias')       updateBias();
  else if (s === 'anglebias')  updateAngleBias();
  else if (s === 'anglepivot') updateAnglePivotCombined();
  else if (s === 'ta')         updateThetaAlphaBias('ta');
  else if (s === 'tapivot')    updateThetaAlphaBias('tapivot');
}

function onGlobalMetricChange() {
  saveState();
  _renderActiveTab();
}

function selVal(id) { return document.getElementById(id).value; }

function populateSelect(id, values, labelMap) {
  const el = document.getElementById(id);
  el.innerHTML = '';
  values.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = labelMap ? (labelMap[v] != null ? labelMap[v] : v) : v;
    el.appendChild(opt);
  });
}

function noData(plotId) {
  Plotly.purge(plotId);
  document.getElementById(plotId).innerHTML =
    '<div class="no-data">No data available for this selection.</div>';
}

// ── Tab 1 & 2: landscape ──────────────────────────────────────────────────
function _getGroupLoss(entry, grp) {
  if (grp === 'all' || !entry || !entry.plane_loss) return entry ? entry.loss : [];
  const planeKeys = Object.keys(entry.plane_loss).filter(k => k[0] === grp);
  if (planeKeys.length === 0) return entry.loss;
  const n = entry.loss.length;
  const result = new Array(n);
  for (let j = 0; j < n; j++) {
    let logSum = 0;
    for (const k of planeKeys) logSum += Math.log1p(entry.plane_loss[k][j]);
    result[j] = Math.expm1(logSum / planeKeys.length);
  }
  return result;
}
function updateLandscape(study) {
  const pfx        = study === 'startx' ? 'sx' : 'an';
  const param      = selVal(pfx + '-param');
  const adc        = selVal(pfx + '-adc');
  const track      = selVal(pfx + '-track');
  const planeGroup = selVal(pfx + '-plane-group') || 'all';
  const plotId     = 'plot-' + study;

  const traces = [];
  const allLoss = [];
  let factors = null;

  SEEDS.forEach((seed, i) => {
    const key   = param + '|' + track + '|' + adc + '|' + seed;
    const entry = DATA[study].data[key];
    if (!entry) return;
    if (!factors) factors = entry.factors;
    allLoss.push(_getGroupLoss(entry, planeGroup));
    traces.push({
      x: entry.factors, y: _getGroupLoss(entry, planeGroup),
      mode: 'lines',
      name: 'seed ' + seed,
      line: { color: SEED_COLORS[i], width: 1.3 },
      hovertemplate: 'factor=%{x:.3f}  loss=%{y:.5g}<extra>seed ' + seed + '</extra>',
      legendgroup: 'seeds',
    });
  });

  if (allLoss.length === 0) { noData(plotId); return; }

  const nPts = factors.length;
  const mean  = new Float64Array(nPts);
  const mean2 = new Float64Array(nPts);
  for (const lv of allLoss) {
    lv.forEach((v, j) => { mean[j] += v; mean2[j] += v * v; });
  }
  const n = allLoss.length;
  for (let j = 0; j < nPts; j++) {
    mean[j]  /= n;
    mean2[j]  = Math.sqrt(Math.max(0, mean2[j] / n - mean[j] * mean[j]));
  }

  const upper = Array.from(mean).map((v, j) => v + mean2[j]);
  const lower = Array.from(mean).map((v, j) => v - mean2[j]);

  traces.push({
    x: [...factors, ...[...factors].reverse()],
    y: [...upper,   ...[...lower].reverse()],
    fill: 'toself', fillcolor: 'rgba(100,100,255,0.10)',
    line: { color: 'transparent' },
    hoverinfo: 'skip', showlegend: false, mode: 'lines', legendgroup: 'stat',
  });
  traces.push({
    x: factors, y: Array.from(mean),
    mode: 'lines', name: 'mean',
    line: { color: 'rgba(20,20,20,0.9)', width: 2.5 },
    hovertemplate: 'factor=%{x:.3f}  loss=%{y:.5g}<extra>mean</extra>',
    legendgroup: 'stat',
  });
  traces.push({
    x: factors, y: upper, mode: 'lines', name: 'mean+1σ',
    line: { color: 'rgba(80,80,200,0.5)', width: 1, dash: 'dot' },
    hoverinfo: 'skip', showlegend: true, legendgroup: 'stat',
  });
  traces.push({
    x: factors, y: lower, mode: 'lines', name: 'mean−1σ',
    line: { color: 'rgba(80,80,200,0.5)', width: 1, dash: 'dot' },
    hoverinfo: 'skip', showlegend: true, legendgroup: 'stat',
  });

  // No-noise overlay
  if (DATA[study].nonoise_data) {
    const nnEntry = DATA[study].nonoise_data[param + '|' + track + '|' + adc];
    if (nnEntry) {
      traces.push({
        x: nnEntry.factors, y: _getGroupLoss(nnEntry, planeGroup),
        mode: 'lines',
        name: 'no noise',
        line: { color: 'rgba(200,80,20,0.85)', width: 2.5, dash: 'dash' },
        hovertemplate: 'factor=%{x:.3f}  loss=%{y:.5g}<extra>no noise</extra>',
        legendgroup: 'nonoise',
      });
    }
  }

  // "Both" mode: overlay NWR traces as additional lines
  if (typeof currentDataset !== 'undefined' && currentDataset === 'both'
      && typeof DATASETS !== 'undefined' && DATASETS.nwr) {
    const nwrStudy = DATASETS.nwr[study];
    if (nwrStudy && nwrStudy.data) {
      const nwrAllLoss = [];
      let nwrFactors = null;
      SEEDS.forEach((seed, i) => {
        const key   = param + '|' + track + '|' + adc + '|' + seed;
        const entry = nwrStudy.data[key];
        if (!entry) return;
        if (!nwrFactors) nwrFactors = entry.factors;
        nwrAllLoss.push(_getGroupLoss(entry, planeGroup));
        traces.push({
          x: entry.factors, y: _getGroupLoss(entry, planeGroup),
          mode: 'lines',
          name: 'NWR seed ' + seed,
          line: { color: NWR_SEED_COLORS[i % NWR_SEED_COLORS.length], width: 1.3 },
          hovertemplate: 'factor=%{x:.3f}  loss=%{y:.5g}<extra>NWR seed ' + seed + '</extra>',
          legendgroup: 'nwr-seeds',
        });
      });
      if (nwrFactors && nwrAllLoss.length > 0) {
        const nPtsN = nwrFactors.length;
        const nwrM  = new Float64Array(nPtsN);
        const nwrM2 = new Float64Array(nPtsN);
        for (const lv of nwrAllLoss) {
          lv.forEach((v, j) => { nwrM[j] += v; nwrM2[j] += v * v; });
        }
        const nN = nwrAllLoss.length;
        for (let j = 0; j < nPtsN; j++) {
          nwrM[j]  /= nN;
          nwrM2[j]  = Math.sqrt(Math.max(0, nwrM2[j] / nN - nwrM[j] * nwrM[j]));
        }
        const nwrUpper = Array.from(nwrM).map((v, j) => v + nwrM2[j]);
        const nwrLower = Array.from(nwrM).map((v, j) => v - nwrM2[j]);
        traces.push({
          x: [...nwrFactors, ...[...nwrFactors].reverse()],
          y: [...nwrUpper,   ...[...nwrLower].reverse()],
          fill: 'toself', fillcolor: 'rgba(200,80,0,0.07)',
          line: { color: 'transparent' },
          hoverinfo: 'skip', showlegend: false, mode: 'lines', legendgroup: 'nwr-stat',
        });
        traces.push({
          x: nwrFactors, y: Array.from(nwrM),
          mode: 'lines', name: 'NWR mean',
          line: { color: NWR_MEAN_COLOR, width: 2.5 },
          hovertemplate: 'factor=%{x:.3f}  loss=%{y:.5g}<extra>NWR mean</extra>',
          legendgroup: 'nwr-stat',
        });
        traces.push({
          x: nwrFactors, y: nwrUpper, mode: 'lines', name: 'NWR mean+1σ',
          line: { color: 'rgba(200,80,0,0.45)', width: 1, dash: 'dot' },
          hoverinfo: 'skip', showlegend: true, legendgroup: 'nwr-stat',
        });
        traces.push({
          x: nwrFactors, y: nwrLower, mode: 'lines', name: 'NWR mean−1σ',
          line: { color: 'rgba(200,80,0,0.45)', width: 1, dash: 'dot' },
          hoverinfo: 'skip', showlegend: true, legendgroup: 'nwr-stat',
        });
      }
      if (nwrStudy.nonoise_data) {
        const nnE = nwrStudy.nonoise_data[param + '|' + track + '|' + adc];
        if (nnE) {
          traces.push({
            x: nnE.factors, y: _getGroupLoss(nnE, planeGroup),
            mode: 'lines', name: 'NWR no noise',
            line: { color: 'rgba(20,160,60,0.85)', width: 2.5, dash: 'dash' },
            hovertemplate: 'factor=%{x:.3f}  loss=%{y:.5g}<extra>NWR no noise</extra>',
            legendgroup: 'nwr-nonoise',
          });
        }
      }
    }
  }

  const trackLabel = DATA[study].trackLabels[track] || track;
  const paramLabel = PARAM_LABELS[param] || param;
  const adcLabel   = adc === '0' ? 'no ADC cut' : 'ADC ≥ ' + adc;
  const grpLabel   = planeGroup === 'all' ? 'all planes' : planeGroup + ' planes';

  Plotly.react(plotId, traces, {
    title: { text: paramLabel + '  —  ' + adcLabel + '  —  ' + grpLabel + '  —  ' + trackLabel, font: {size:13}, x: 0.5 },
    xaxis: { title: {text:'param / GT', standoff:8}, tickformat:'.2f', gridcolor:'#eee', zeroline:false,
             range: [Math.min(...factors)-0.01, Math.max(...factors)+0.01] },
    yaxis: { title: {text:'loss', standoff:8}, gridcolor:'#eee', zeroline:false, rangemode:'tozero' },
    margin: {t:50, b:55, l:65, r:20},
    paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    legend: { x:1.01, xanchor:'left', y:0.99, font:{size:11},
              bgcolor:'rgba(255,255,255,0.8)', bordercolor:'#ddd', borderwidth:1 },
    shapes: [{ type:'line', x0:1, x1:1, y0:0, y1:1, yref:'paper',
               line:{color:'#bbb', width:1.2, dash:'dot'} }],
    annotations: [{ x:1, y:1, yref:'paper', text:'GT', showarrow:false,
                    xanchor:'left', xshift:5, font:{size:11, color:'#888'} }],
  }, { responsive:true, displayModeBar:true });
}

// ── Tab 3: bias ───────────────────────────────────────────────────────────
function setBiasView(view) {
  biasView = view;
  document.getElementById('bi-view-combined').classList.toggle('active', view === 'combined');
  document.getElementById('bi-view-grid').classList.toggle('active', view === 'grid');
  updateBias();
  saveState();
}

function _histUnicode(vals, bins) {
  bins = bins || 15;
  if (!vals || vals.length === 0) return '';
  const vs = vals.map(sv => sv[1]);
  const mn = Math.min(...vs), mx = Math.max(...vs);
  const range = mx - mn || 1e-9;
  const counts = new Array(bins).fill(0);
  vs.forEach(v => { let b = Math.floor((v - mn) / range * bins); counts[Math.min(b, bins-1)]++; });
  const maxC = Math.max(...counts);
  const BARS = ' ▁▂▃▄▅▆▇█';
  return mn.toFixed(3) + ' ' + counts.map(c => BARS[Math.round(c / maxC * 8)]).join('') + ' ' + mx.toFixed(3);
}

function _seedLines(vals) {
  if (!vals || vals.length === 0) return '(no data)';
  const hist = _histUnicode(vals);
  const lines = vals.slice(0, 10).map(([s, v]) => 'seed=' + s + ': ' + v.toFixed(4)).join('<br>');
  return (hist ? '<b>dist:</b> ' + hist + '<br>' : '') + lines;
}

// Convert a CSS color (#rrggbb or rgb(...)) to rgba with given alpha.
function _hexAlpha(col, a) {
  if (col.startsWith('rgba(')) return col.replace(/,\s*[\d.]+\s*\)$/, ',' + a + ')');
  if (col.startsWith('rgb('))  return col.replace('rgb(', 'rgba(').replace(')', ',' + a + ')');
  const r = parseInt(col.slice(1,3),16), g = parseInt(col.slice(3,5),16), b = parseInt(col.slice(5,7),16);
  return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
}

function setSeedHistSameX(val) {
  seedHistSameX = val;
  document.getElementById('bi-xaxis-auto').classList.toggle('active', !val);
  document.getElementById('bi-xaxis-fixed').classList.toggle('active', val);
  if (seedHistLast.vals) showSeedHist(seedHistLast.vals, seedHistLast.title);
}

// Show a small Plotly histogram below the active plot.
function showSeedHist(vals, title) {
  if (!vals || vals.length === 0) { hideSeedHist(); return; }
  seedHistLast = {vals, title};
  const vs = vals.map(sv => sv[1]);
  const xRange = (seedHistSameX && seedHistXRange) ? seedHistXRange : null;
  const _nBins = 25;
  const _binSpec = xRange
    ? { xbins: { start: xRange[0], end: xRange[1], size: (xRange[1] - xRange[0]) / _nBins }, autobinx: false }
    : { nbinsx: _nBins };
  document.getElementById('seed-hist-title').textContent = title || '';
  document.getElementById('seed-hist-wrap').style.display = '';
  Plotly.react('seed-hist-plot', [Object.assign({
    type: 'histogram', x: vs,
    marker: { color: 'rgba(66,133,244,0.60)', line: { color: '#446', width: 0.5 } },
  }, _binSpec)], {
    margin: {t:4, b:32, l:42, r:10},
    paper_bgcolor:'#f5f7ff', plot_bgcolor:'#f5f7ff',
    xaxis: Object.assign({ title:{text:'factor',standoff:3}, gridcolor:'#dde',
             zeroline:true, zerolinecolor:'#bbc', tickfont:{size:9} },
             xRange ? {range: xRange} : {}),
    yaxis: { title:{text:'count',standoff:3}, gridcolor:'#dde', tickfont:{size:9} },
    bargap: 0.04,
    shapes: [{ type:'line', x0:1, x1:1, y0:0, y1:1, yref:'paper',
               line:{color:'#c33', width:1.5, dash:'dot'} }],
  }, { responsive:true, displayModeBar:false });
}
function hideSeedHist() {
  document.getElementById('seed-hist-wrap').style.display = 'none';
}
// Bind plotly_hover/unhover on a bias plot to update the seed histogram.
// isHeatmap=true: customdata per cell is the raw vals array directly.
// isHeatmap=false: customdata[4] holds the raw vals array.
function _bindHoverHist(plotId, isHeatmap) {
  const el = document.getElementById(plotId);
  if (!el || el._histBound) return;
  el._histBound = true;
  el.on('plotly_hover', function(data) {
    const pt = data.points[0];
    const vals = isHeatmap ? pt.customdata : (pt.customdata && pt.customdata[4]);
    const title = isHeatmap
      ? ('θ=' + pt.y + '°  α=' + pt.x + '°  — seed factor distribution')
      : ((pt.data.name || '') + '  — seed factor distribution');
    showSeedHist(vals, title);
  });
  el.on('plotly_unhover', hideSeedHist);
}

function _biasTraces(param, adc, baseTrack, driftDists, opts, srcBias, srcNNBias, metric) {
  srcBias   = srcBias   || DATA.bias;
  srcNNBias = srcNNBias || DATA.nonoise_bias;
  metric    = metric || 'mean';
  const isMean = METRIC_KIND[metric] === 'mean';
  const dash = opts.dash || null;
  const means = [], stds = [], ns = [], perSeed = [], nnFactors = [], rawVals = [];
  driftDists.forEach(d => {
    const e = srcBias.data[param + '|' + baseTrack + '|' + adc + '|' + d];
    const st = _metricStats(e ? (e.vals || e.vals10) : null, metric, e ? e.mean : null, e ? e.std : null);
    means.push(st.value);
    stds.push(st.err);
    ns.push(e ? e.n : 0);
    perSeed.push(e ? _seedLines(e.vals || e.vals10) : '(no data)');
    rawVals.push(e ? (e.vals || e.vals10 || []) : []);
    const nn = srcNNBias && srcNNBias.data
               ? srcNNBias.data[param + '|' + baseTrack + '|' + adc + '|' + d] : null;
    nnFactors.push(nn ? nn.factor : null);
  });

  const sems = stds.map((s, i) => (s != null && ns[i] > 0) ? s / Math.sqrt(ns[i]) : null);

  const col   = opts.color;
  const axMap = (opts.xaxis ? { xaxis: opts.xaxis } : {});
  const ayMap = (opts.yaxis ? { yaxis: opts.yaxis } : {});
  const valueLabel = isMean ? 'mean' : (METRIC_LABELS[metric] || metric);

  const traceAll = {
    x: driftDists, y: means,
    customdata: stds.map((s, i) => [s, ns[i], perSeed[i], sems[i], rawVals[i]]),
    error_y: { type:'data', array: stds, visible: isMean, color: col, thickness:1.8, width:5 },
    mode: 'lines+markers',
    name: opts.name,
    legendgroup: opts.name,
    line:   Object.assign({ color: col, width: 2 }, dash ? {dash} : {}),
    marker: { color: col, size: 6, symbol: dash ? 'square' : 'circle' },
    hovertemplate: '|start_x|=%{x} mm<br>'
      + (isMean
          ? valueLabel + '=%{y:.4f}  std=%{customdata[0]:.4f}  SEM=%{customdata[3]:.5f}  n=%{customdata[1]}<br>'
          : valueLabel + '=%{y:.4f}  n=%{customdata[1]}<br>')
      + '<b>first seeds:</b><br>%{customdata[2]}'
      + '<extra>' + opts.name + '</extra>',
    ...axMap, ...ayMap,
  };

  if (!isMean) return [traceAll];

  // 99.7% CI band on the mean (±3×SEM)
  const ciTrace = {
    x: [...driftDists, ...[...driftDists].reverse()],
    y: [...means.map((v,j) => (v != null && sems[j] != null) ? v + 3*sems[j] : null),
        ...[...means].reverse().map((v,j2) => {
          const j = means.length - 1 - j2;
          return (v != null && sems[j] != null) ? v - 3*sems[j] : null;
        })],
    fill: 'toself', fillcolor: _hexAlpha(col, 0.22),
    line: { color: 'transparent' },
    hoverinfo: 'skip', showlegend: false, mode: 'lines',
    legendgroup: opts.name,
    ...axMap, ...ayMap,
  };

  const traceNN = {
    x: driftDists, y: nnFactors,
    mode: 'markers',
    name: (opts.name || '') + ' (no noise)',
    legendgroup: opts.name,
    showlegend: false,
    marker: { color: col, size: 9, symbol: 'square', line: { color: '#fff', width: 1.5 } },
    hovertemplate: '|start_x|=%{x} mm<br>no-noise=%{y:.4f}<extra>no noise</extra>',
    ...axMap, ...ayMap,
  };

  return [ciTrace, traceAll, traceNN];
}

function updateBias() {
  const param   = selVal('bi-param');
  const adc     = selVal('bi-adc');
  const metric  = selVal('global-metric') || 'mean';
  const isMean  = METRIC_KIND[metric] === 'mean';
  const yAxisTitle = isMean ? 'Recovered factor (argmin loss)' : (METRIC_LABELS[metric] || metric);
  const plotId  = 'plot-bias';
  const biasD   = DATA.bias;
  const baseTracks = biasD.baseTracks;

  // Check if any data exists
  const hasAny = baseTracks.some(bt =>
    biasD.driftDists[bt].some(d => biasD.data[param + '|' + bt + '|' + adc + '|' + d])
  );
  if (!hasAny) { noData(plotId); return; }

  const paramLabel = PARAM_LABELS[param] || param;
  const adcLabel   = adc === '0' ? 'no ADC cut' : 'ADC ≥ ' + adc;
  const refShape   = { type:'line', y0:1, y1:1, x0:0, x1:1, xref:'paper',
                       line:{color:'#888', width:1.2, dash:'dot'} };

  if (biasView === 'combined') {
    // ── single plot, 4 lines ──────────────────────────────────────────────
    const traces = baseTracks.flatMap(bt =>
      _biasTraces(param, adc, bt, biasD.driftDists[bt], {
        color: TRACK_COLORS[bt],
        name:  biasD.baseTrackLabels[bt],
      }, undefined, undefined, metric)
    );
    // "Both" mode: add NWR tracks as dashed lines
    if (typeof currentDataset !== 'undefined' && currentDataset === 'both'
        && typeof DATASETS !== 'undefined' && DATASETS.nwr && DATASETS.nwr.bias) {
      const nwrBias   = DATASETS.nwr.bias;
      const nwrNNBias = DATASETS.nwr.nonoise_bias;
      if (nwrBias.baseTracks) {
        nwrBias.baseTracks.forEach(bt => {
          _biasTraces(param, adc, bt, nwrBias.driftDists[bt], {
            color: TRACK_COLORS[bt] || '#999',
            name:  (nwrBias.baseTrackLabels[bt] || bt) + ' (NWR)',
            dash: 'dash',
          }, nwrBias, nwrNNBias, metric).forEach(tr => traces.push(tr));
        });
      }
    }

    Plotly.react(plotId, traces, {
      title: { text: paramLabel + '  —  ' + adcLabel + '  —  All tracks  (x=0: cathode → long drift; x=2000: near anode → short drift)', font:{size:11}, x:0.5 },
      xaxis: { title:{text:'|start_x| — distance from cathode (mm)', standoff:8}, gridcolor:'#eee', zeroline:false },
      yaxis: { title:{text: yAxisTitle, standoff:8}, gridcolor:'#eee',
               zeroline:false },
      margin: {t:50, b:55, l:75, r:20},
      paper_bgcolor:'#fff', plot_bgcolor:'#fff',
      shapes: isMean ? [refShape] : [],
      annotations: isMean ? [{ x:0.5, y:1, yref:'y', text:'GT (factor=1)', showarrow:false,
                      xref:'paper', yanchor:'bottom', font:{size:10, color:'#888'} }] : [],
      legend: { x:1.01, xanchor:'left', y:0.99, font:{size:11},
                bgcolor:'rgba(255,255,255,0.8)', bordercolor:'#ddd', borderwidth:1 },
    }, { responsive:true, displayModeBar:true });
    _bindHoverHist(plotId);

  } else {
    // ── 2×2 subplot grid ─────────────────────────────────────────────────
    // Layout: col1=[0,0.43] col2=[0.57,1]  row1=[0.55,1]  row2=[0,0.45]
    const xDoms = [[0.00, 0.43], [0.57, 1.00]];
    const yDoms = [[0.55, 1.00], [0.00, 0.45]];
    const axSuffix = ['', '2', '3', '4'];

    const traces = [];
    const annotations = [];
    const shapes = [];
    const layout = {
      margin: {t:50, b:55, l:65, r:20},
      paper_bgcolor:'#fff', plot_bgcolor:'#fff',
      title: { text: paramLabel + '  —  ' + adcLabel + '  —  Per track  (x=0: cathode; x=2000: near anode)', font:{size:11}, x:0.5 },
    };

    baseTracks.forEach((bt, i) => {
      const col = i % 2, row = Math.floor(i / 2);
      const ax  = axSuffix[i];
      const xKey = 'xaxis' + ax, yKey = 'yaxis' + ax;
      const xRef = 'x' + ax,    yRef = 'y' + ax;

      layout[xKey] = { domain: xDoms[col], anchor: yRef,
                       title: {text:'|start_x| — distance from cathode (mm)', standoff:6},
                       gridcolor:'#eee', zeroline:false };
      layout[yKey] = { domain: yDoms[row], anchor: xRef,
                       title: col === 0 ? {text: yAxisTitle, standoff:6} : undefined,
                       gridcolor:'#eee', zeroline:false };

      _biasTraces(param, adc, bt, biasD.driftDists[bt], {
        color: TRACK_COLORS[bt],
        name:  biasD.baseTrackLabels[bt],
        xaxis: xRef, yaxis: yRef,
      }, undefined, undefined, metric).forEach(tr => { tr.showlegend = false; traces.push(tr); });

      // subplot title via annotation
      const titleX = (xDoms[col][0] + xDoms[col][1]) / 2;
      const titleY = yDoms[row][1] + 0.01;
      annotations.push({
        text: biasD.baseTrackLabels[bt], showarrow:false,
        xref:'paper', yref:'paper', x: titleX, y: titleY,
        xanchor:'center', yanchor:'bottom',
        font: { size:12, color: TRACK_COLORS[bt], weight:600 },
      });

      // GT reference line per subplot
      if (isMean) {
        shapes.push({
          type:'line', y0:1, y1:1, x0:0, x1:1,
          xref:'paper', yref: yRef,
          line: {color:'#aaa', width:1, dash:'dot'},
        });
      }
    });

    // "Both" mode: add NWR data to matching subplots
    if (typeof currentDataset !== 'undefined' && currentDataset === 'both'
        && typeof DATASETS !== 'undefined' && DATASETS.nwr && DATASETS.nwr.bias) {
      const nwrBias   = DATASETS.nwr.bias;
      const nwrNNBias = DATASETS.nwr.nonoise_bias;
      if (nwrBias.baseTracks) {
        nwrBias.baseTracks.forEach(bt => {
          const idx = baseTracks.indexOf(bt);
          if (idx < 0) return;
          const ax2  = axSuffix[idx];
          const xRef2 = 'x' + ax2, yRef2 = 'y' + ax2;
          _biasTraces(param, adc, bt, nwrBias.driftDists[bt], {
            color: TRACK_COLORS[bt] || '#999',
            name:  (nwrBias.baseTrackLabels[bt] || bt) + ' (NWR)',
            dash: 'dash',
            xaxis: xRef2, yaxis: yRef2,
          }, nwrBias, nwrNNBias, metric).forEach(tr => { tr.showlegend = false; traces.push(tr); });
        });
      }
    }

    layout.annotations = annotations;
    layout.shapes      = shapes;
    Plotly.react(plotId, traces, layout, { responsive:true, displayModeBar:true });
  }

  // Compute global factor range across all cells for the "same x axis" toggle
  const _allBiasFactors = [];
  biasD.baseTracks.forEach(bt => {
    biasD.driftDists[bt].forEach(d => {
      const e = biasD.data[param + '|' + bt + '|' + adc + '|' + d];
      if (e && e.vals) e.vals.forEach(sv => _allBiasFactors.push(sv[1]));
    });
  });
  seedHistXRange = _allBiasFactors.length > 0
    ? [Math.min(..._allBiasFactors), Math.max(..._allBiasFactors)] : null;

  _bindHoverHist(plotId);
}

// ── Plane-group helpers for bias tabs ─────────────────────────────────────
// Returns {mean, std, n, vals, sem} for the selected plane group.
// Falls back to all-plane stats when grp='all' or plane_vals is absent (old caches).
function _biasGroupStats(entry, grp) {
  if (!entry) return {mean: null, std: null, n: 0, vals: [], sem: null};
  if (grp === 'all' || !entry.plane_vals || !entry.plane_vals[grp]) {
    const n   = entry.n   || 0;
    const std = entry.std != null ? entry.std : null;
    const sem = (std != null && n > 0) ? std / Math.sqrt(n) : null;
    return {mean: entry.mean, std, n, vals: entry.vals || [], sem};
  }
  const pv = entry.plane_vals[grp];
  if (!pv || !pv.length) return {mean: null, std: null, n: 0, vals: [], sem: null};
  const fs = pv.map(v => v[1]);
  const n  = fs.length;
  const mean = fs.reduce((a, b) => a + b, 0) / n;
  const std  = n > 1
    ? Math.sqrt(fs.map(v => (v - mean) * (v - mean)).reduce((a, b) => a + b, 0) / n)
    : 0;
  const sem = n > 0 ? std / Math.sqrt(n) : null;
  return {mean, std, n, vals: pv, sem};
}
// Returns the no-noise factor for the selected plane group (falls back to total).
function _biasNNFactor(nn, grp) {
  if (!nn) return null;
  if (grp === 'all' || !nn.plane_factors || nn.plane_factors[grp] == null) return nn.factor;
  return nn.plane_factors[grp];
}

// ── Tab 4: bias vs angle ──────────────────────────────────────────────────
function updateAngleBias() {
  const param  = selVal('ab-param');
  const adc    = selVal('ab-adc');
  const grp    = selVal('ab-plane-group') || 'all';
  const metric = selVal('global-metric') || 'mean';
  const isMean = METRIC_KIND[metric] === 'mean';
  const yAxisTitle = isMean ? 'Recovered factor (argmin loss)' : (METRIC_LABELS[metric] || metric);
  const plotId = 'plot-anglebias';
  const ab     = DATA.angle_bias;

  const thetas = ab.thetas;
  const means = [], stds = [], ns = [], perSeed = [], rawVals = [];
  thetas.forEach(theta => {
    const entry = ab.data[param + '|' + adc + '|' + theta];
    const s = _biasGroupStats(entry, grp);
    const st = _metricStats(s.vals, metric, s.mean, s.std);
    means.push(st.value);
    stds.push( st.err);
    ns.push(   s.n);
    perSeed.push(s.vals.length ? _seedLines(s.vals) : '(no data)');
    rawVals.push(s.vals);
  });

  if (means.every(v => v === null)) { noData(plotId); return; }

  const paramLabel = PARAM_LABELS[param] || param;
  const adcLabel   = adc === '0' ? 'no ADC cut' : 'ADC ≥ ' + adc;
  const grpLabel   = grp === 'all' ? 'all planes' : grp + ' planes';
  const col = '#1f77b4';
  const sems = stds.map((s, i) => (s != null && ns[i] > 0) ? s / Math.sqrt(ns[i]) : null);
  const valueLabel = isMean ? 'mean' : (METRIC_LABELS[metric] || metric);

  const traces = [];

  if (isMean) {
    // outer shaded band: ±1σ seed spread
    traces.push({
      x: [...thetas, ...[...thetas].reverse()],
      y: [...means.map((v,j) => v != null ? v + stds[j] : null),
          ...[...means].reverse().map((v,j2) => {
            const j = means.length - 1 - j2;
            return v != null ? v - stds[j] : null;
          })],
      fill: 'toself', fillcolor: 'rgba(31,119,180,0.08)',
      line: { color: 'transparent' },
      hoverinfo: 'skip', showlegend: true, mode: 'lines',
      name: '±1σ seed spread',
    });
    // inner shaded band: 99.7% CI on mean (±3×SEM)
    traces.push({
      x: [...thetas, ...[...thetas].reverse()],
      y: [...means.map((v,j) => (v != null && sems[j] != null) ? v + 3*sems[j] : null),
          ...[...means].reverse().map((v,j2) => {
            const j = means.length - 1 - j2;
            return (v != null && sems[j] != null) ? v - 3*sems[j] : null;
          })],
      fill: 'toself', fillcolor: 'rgba(31,119,180,0.30)',
      line: { color: 'transparent' },
      hoverinfo: 'skip', showlegend: true, mode: 'lines',
      name: '99.7% CI on mean',
    });
  }
  // 100-seed line
  traces.push({
    x: thetas, y: means,
    customdata: stds.map((s, i) => [s, ns[i], perSeed[i], sems[i], rawVals[i]]),
    error_y: { type:'data', array: stds, visible: isMean,
               color: col, thickness:1.8, width:6 },
    mode: 'lines+markers',
    name: BIAS_SEED_LABEL,
    legendgroup: 'all',
    line:   { color: col, width: 2.2 },
    marker: { color: col, size: 7, symbol: 'circle' },
    hovertemplate: 'θ=%{x}°<br>'
      + (isMean
          ? valueLabel + '=%{y:.4f}  std=%{customdata[0]:.4f}  SEM=%{customdata[3]:.5f}  n=%{customdata[1]}<br>'
          : valueLabel + '=%{y:.4f}  n=%{customdata[1]}<br>')
      + '<b>first seeds:</b><br>%{customdata[2]}'
      + '<extra>' + BIAS_SEED_LABEL + '</extra>',
  });

  // No-noise reference markers
  if (isMean && DATA.nonoise_angle_bias) {
    const nnThetas = [], nnVals = [];
    thetas.forEach(theta => {
      const nn = DATA.nonoise_angle_bias.data[param + '|' + adc + '|' + theta];
      const f  = _biasNNFactor(nn, grp);
      if (f != null) { nnThetas.push(theta); nnVals.push(f); }
    });
    if (nnVals.length > 0) {
      traces.push({
        x: nnThetas, y: nnVals,
        mode: 'markers',
        name: 'no noise',
        marker: { color: 'rgba(200,80,20,0.9)', size: 9, symbol: 'square',
                  line: { color: '#fff', width: 1.5 } },
        hovertemplate: 'θ=%{x}°<br>no-noise factor=%{y:.4f}<extra>no noise</extra>',
      });
    }
  }

  // "Both" mode: add NWR angle bias as dashed line
  if (typeof currentDataset !== 'undefined' && currentDataset === 'both'
      && typeof DATASETS !== 'undefined' && DATASETS.nwr && DATASETS.nwr.angle_bias) {
    const nwrAB = DATASETS.nwr.angle_bias;
    const nwrThetas = nwrAB.thetas;
    const nwrMeans = [], nwrStds = [], nwrNs = [], nwrPerSeed = [];
    nwrThetas.forEach(theta => {
      const entry = nwrAB.data[param + '|' + adc + '|' + theta];
      const ns2   = _biasGroupStats(entry, grp);
      const st2   = _metricStats(ns2.vals, metric, ns2.mean, ns2.std);
      nwrMeans.push(st2.value);
      nwrStds.push( st2.err);
      nwrNs.push(   ns2.n);
      nwrPerSeed.push(ns2.vals.length ? _seedLines(ns2.vals) : '(no data)');
    });
    if (!nwrMeans.every(v => v === null)) {
      const nwrCol = '#ff7f0e';
      if (isMean) {
        traces.push({
          x: [...nwrThetas, ...[...nwrThetas].reverse()],
          y: [...nwrMeans.map((v,j) => v != null ? v + nwrStds[j] : null),
              ...[...nwrMeans].reverse().map((v,j2) => {
                const j = nwrMeans.length - 1 - j2;
                return v != null ? v - nwrStds[j] : null;
              })],
          fill: 'toself', fillcolor: 'rgba(255,127,14,0.08)',
          line: { color: 'transparent' },
          hoverinfo: 'skip', showlegend: false, mode: 'lines',
        });
      }
      traces.push({
        x: nwrThetas, y: nwrMeans,
        customdata: nwrStds.map((s, i) => [s, nwrNs[i], nwrPerSeed[i]]),
        error_y: { type:'data', array: nwrStds, visible: isMean,
                   color: nwrCol, thickness:1.8, width:6 },
        mode: 'lines+markers',
        name: 'NWR (' + DATASET_LABELS.nwr + ')',
        legendgroup: 'nwr-all',
        line:   { color: nwrCol, width: 2.2, dash: 'dash' },
        marker: { color: nwrCol, size: 7, symbol: 'square' },
        hovertemplate: 'θ=%{x}°<br>'
          + (isMean
              ? 'NWR ' + valueLabel + '=%{y:.4f}  std=%{customdata[0]:.4f}  n=%{customdata[1]}<br>'
              : 'NWR ' + valueLabel + '=%{y:.4f}  n=%{customdata[1]}<br>')
          + '<extra>NWR</extra>',
      });
      if (isMean && DATASETS.nwr.nonoise_angle_bias) {
        const nwrNNThetas = [], nwrNNVals = [];
        nwrThetas.forEach(theta => {
          const nn = DATASETS.nwr.nonoise_angle_bias.data[param + '|' + adc + '|' + theta];
          const f  = _biasNNFactor(nn, grp);
          if (f != null) { nwrNNThetas.push(theta); nwrNNVals.push(f); }
        });
        if (nwrNNVals.length > 0) {
          traces.push({
            x: nwrNNThetas, y: nwrNNVals,
            mode: 'markers', name: 'NWR no noise',
            marker: { color: nwrCol, size: 9, symbol: 'diamond',
                      line: { color: '#fff', width: 1.5 } },
            hovertemplate: 'θ=%{x}°<br>NWR no-noise=%{y:.4f}<extra>NWR no noise</extra>',
          });
        }
      }
    }
  }

  Plotly.react(plotId, traces, {
    title: { text: paramLabel + '  —  ' + adcLabel + '  —  Bias vs track angle  [' + grpLabel + ']',
             font:{size:13}, x:0.5 },
    xaxis: {
      title: { text: 'Track angle θ (degrees)', standoff:8 },
      tickvals: thetas, ticktext: thetas.map(t => t + '°'),
      gridcolor:'#eee', zeroline:false,
    },
    yaxis: {
      title: { text: yAxisTitle, standoff:8 },
      gridcolor:'#eee', zeroline:false,
    },
    margin: {t:50, b:60, l:75, r:20},
    paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    showlegend: true,
    legend: { x:1.01, xanchor:'left', y:0.99, font:{size:11},
              bgcolor:'rgba(255,255,255,0.8)', bordercolor:'#ddd', borderwidth:1 },
    shapes: [
      ...(isMean ? [{ type:'line', y0:1, y1:1, x0:0, x1:1, xref:'paper',
        line:{color:'#888', width:1.2, dash:'dot'} }] : []),
      { type:'line', x0:0, x1:0, y0:0, y1:1, yref:'paper',
        line:{color:'#ddd', width:1, dash:'dot'} },
    ],
    annotations: [
      ...(isMean ? [{ x:0.5, y:1, yref:'y', text:'GT (factor=1)', showarrow:false,
        xref:'paper', yanchor:'bottom', font:{size:10, color:'#888'} }] : []),
      { x:0, y:1, yref:'paper', text:'θ=0° (pure drift)', showarrow:false,
        xanchor:'left', xshift:4, yshift:-14, font:{size:9, color:'#aaa'} },
    ],
  }, { responsive:true, displayModeBar:true });
  _bindHoverHist(plotId);

  // Table of values sorted by theta (x)
  const sorted = thetas.map((t,i) => ({t, mean: means[i], std: stds[i], n: ns[i], sem: sems[i]}))
    .filter(r => r.mean !== null)
    .sort((a,b) => a.t - b.t);
  const rows = sorted.map(r => {
    const nn  = DATA.nonoise_angle_bias && DATA.nonoise_angle_bias.data[param + '|' + adc + '|' + r.t];
    const nnF = _biasNNFactor(nn, grp);
    const nnCell = nnF != null ? nnF.toFixed(5) : '—';
    const stdStr = isMean ? r.std.toFixed(5) : '—';
    const semStr = isMean && r.sem != null ? r.sem.toFixed(5) : '—';
    const zStr   = isMean && r.sem != null ? ((r.mean - 1) / r.sem).toFixed(2) : '—';
    return `<tr><td>${r.t}°</td><td>${r.mean.toFixed(5)}</td><td>${stdStr}</td><td>${r.n}</td><td>${semStr}</td><td>${zStr}</td><td>${nnCell}</td></tr>`;
  }).join('');
  document.getElementById('table-anglebias').innerHTML =
    `<table style="border-collapse:collapse;font-size:0.82rem;width:auto;">
      <thead><tr style="background:#f0f2f5;">
        <th style="padding:4px 14px;border:1px solid #ddd;">θ</th>
        <th style="padding:4px 14px;border:1px solid #ddd;">${METRIC_LABELS[metric] || metric}</th>
        <th style="padding:4px 14px;border:1px solid #ddd;">std</th>
        <th style="padding:4px 14px;border:1px solid #ddd;">n seeds</th>
        <th style="padding:4px 14px;border:1px solid #ddd;">SEM</th>
        <th style="padding:4px 14px;border:1px solid #ddd;">z=(mean−1)/SEM</th>
        <th style="padding:4px 14px;border:1px solid #ddd;">no-noise factor</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ── Tab: bias vs angle (pivot x=1000, combined WR + NWR) ─────────────────
function updateAnglePivotCombined() {
  const param  = selVal('ap-param');
  const adc    = selVal('ap-adc');
  const grp    = selVal('ap-plane-group') || 'all';
  const metric = selVal('global-metric') || 'mean';
  const isMean = METRIC_KIND[metric] === 'mean';
  const yAxisTitle = isMean ? 'Recovered factor (argmin loss)' : (METRIC_LABELS[metric] || metric);
  const valueLabel = isMean ? 'mean' : (METRIC_LABELS[metric] || metric);
  const plotId = 'plot-anglepivot';
  const _wrSrc  = (typeof DATASETS !== 'undefined' && DATASETS.full) ? DATASETS.full : DATA;
  const _nwrSrc = (typeof DATASETS !== 'undefined' && DATASETS.nwr)  ? DATASETS.nwr  : DATA;
  const wrAB  = _wrSrc.angle_pivot_bias;
  const nwrAB = _nwrSrc.angle_pivot_nwr_bias;
  if (!wrAB && !nwrAB) { noData(plotId); return; }

  const grpLabel = grp === 'all' ? 'all planes' : grp + ' planes';
  const traces = [];

  if (wrAB) {
    const wrThetas = wrAB.thetas;
    const wrMeans = [], wrStds = [], wrNs = [], wrPerSeed = [], wrRawVals = [];
    wrThetas.forEach(theta => {
      const entry = wrAB.data[param + '|' + adc + '|' + theta];
      const s = _biasGroupStats(entry, grp);
      const st = _metricStats(s.vals, metric, s.mean, s.std);
      wrMeans.push(st.value);
      wrStds.push( st.err);
      wrNs.push(   s.n);
      wrPerSeed.push(s.vals.length ? _seedLines(s.vals) : '(no data)');
      wrRawVals.push(s.vals);
    });
    if (!wrMeans.every(v => v === null)) {
      const col = '#1f77b4';
      const wrSems = wrStds.map((s, i) => (s != null && wrNs[i] > 0) ? s / Math.sqrt(wrNs[i]) : null);
      if (isMean) {
        traces.push({
          x: [...wrThetas, ...[...wrThetas].reverse()],
          y: [...wrMeans.map((v,j) => v != null ? v + wrStds[j] : null),
              ...[...wrMeans].reverse().map((v,j2) => {
                const j = wrMeans.length - 1 - j2;
                return v != null ? v - wrStds[j] : null;
              })],
          fill: 'toself', fillcolor: 'rgba(31,119,180,0.08)',
          line: { color: 'transparent' },
          hoverinfo: 'skip', showlegend: false, mode: 'lines',
        });
      }
      traces.push({
        x: wrThetas, y: wrMeans,
        customdata: wrStds.map((s, i) => [s, wrNs[i], wrPerSeed[i], wrSems[i], wrRawVals[i]]),
        error_y: { type:'data', array: wrStds, visible: isMean, color: col, thickness:1.8, width:6 },
        mode: 'lines+markers',
        name: 'WR — ' + (typeof DATASET_LABELS !== 'undefined' ? DATASET_LABELS.full : BIAS_SEED_LABEL),
        line:   { color: col, width: 2.2 },
        marker: { color: col, size: 7, symbol: 'circle' },
        hovertemplate: 'θ=%{x}°<br>'
          + (isMean
              ? 'WR ' + valueLabel + '=%{y:.4f}  std=%{customdata[0]:.4f}  SEM=%{customdata[3]:.5f}  n=%{customdata[1]}<br>'
              : 'WR ' + valueLabel + '=%{y:.4f}  n=%{customdata[1]}<br>')
          + '<b>first seeds:</b><br>%{customdata[2]}'
          + '<extra>WR</extra>',
      });
      if (isMean && _wrSrc.nonoise_angle_pivot_bias) {
        const nnThetas = [], nnVals = [];
        wrThetas.forEach(theta => {
          const nn = _wrSrc.nonoise_angle_pivot_bias.data[param + '|' + adc + '|' + theta];
          const f  = _biasNNFactor(nn, grp);
          if (f != null) { nnThetas.push(theta); nnVals.push(f); }
        });
        if (nnVals.length > 0) {
          traces.push({
            x: nnThetas, y: nnVals,
            mode: 'markers', name: 'WR no noise',
            marker: { color: col, size: 9, symbol: 'square',
                      line: { color: '#fff', width: 1.5 } },
            hovertemplate: 'θ=%{x}°<br>WR no-noise=%{y:.4f}<extra>WR no noise</extra>',
          });
        }
      }
    }
  }

  if (nwrAB) {
    const nwrThetas = nwrAB.thetas;
    const nwrMeans = [], nwrStds = [], nwrNs = [], nwrPerSeed = [], nwrRawVals = [];
    nwrThetas.forEach(theta => {
      const entry = nwrAB.data[param + '|' + adc + '|' + theta];
      const s = _biasGroupStats(entry, grp);
      const st = _metricStats(s.vals, metric, s.mean, s.std);
      nwrMeans.push(st.value);
      nwrStds.push( st.err);
      nwrNs.push(   s.n);
      nwrPerSeed.push(s.vals.length ? _seedLines(s.vals) : '(no data)');
      nwrRawVals.push(s.vals);
    });
    if (!nwrMeans.every(v => v === null)) {
      const col = '#ff7f0e';
      const nwrSems = nwrStds.map((s, i) => (s != null && nwrNs[i] > 0) ? s / Math.sqrt(nwrNs[i]) : null);
      if (isMean) {
        traces.push({
          x: [...nwrThetas, ...[...nwrThetas].reverse()],
          y: [...nwrMeans.map((v,j) => v != null ? v + nwrStds[j] : null),
              ...[...nwrMeans].reverse().map((v,j2) => {
                const j = nwrMeans.length - 1 - j2;
                return v != null ? v - nwrStds[j] : null;
              })],
          fill: 'toself', fillcolor: 'rgba(255,127,14,0.08)',
          line: { color: 'transparent' },
          hoverinfo: 'skip', showlegend: false, mode: 'lines',
        });
      }
      traces.push({
        x: nwrThetas, y: nwrMeans,
        customdata: nwrStds.map((s, i) => [s, nwrNs[i], nwrPerSeed[i], nwrSems[i], nwrRawVals[i]]),
        error_y: { type:'data', array: nwrStds, visible: isMean, color: col, thickness:1.8, width:6 },
        mode: 'lines+markers',
        name: 'NWR — ' + (typeof DATASET_LABELS !== 'undefined' ? DATASET_LABELS.nwr : BIAS_SEED_LABEL),
        line:   { color: col, width: 2.2, dash: 'dash' },
        marker: { color: col, size: 7, symbol: 'square' },
        hovertemplate: 'θ=%{x}°<br>'
          + (isMean
              ? 'NWR ' + valueLabel + '=%{y:.4f}  std=%{customdata[0]:.4f}  SEM=%{customdata[3]:.5f}  n=%{customdata[1]}<br>'
              : 'NWR ' + valueLabel + '=%{y:.4f}  n=%{customdata[1]}<br>')
          + '<b>first seeds:</b><br>%{customdata[2]}'
          + '<extra>NWR</extra>',
      });
      if (isMean && _nwrSrc.nonoise_angle_pivot_nwr_bias) {
        const nnThetas = [], nnVals = [];
        nwrThetas.forEach(theta => {
          const nn = _nwrSrc.nonoise_angle_pivot_nwr_bias.data[param + '|' + adc + '|' + theta];
          const f  = _biasNNFactor(nn, grp);
          if (f != null) { nnThetas.push(theta); nnVals.push(f); }
        });
        if (nnVals.length > 0) {
          traces.push({
            x: nnThetas, y: nnVals,
            mode: 'markers', name: 'NWR no noise',
            marker: { color: col, size: 9, symbol: 'diamond',
                      line: { color: '#fff', width: 1.5 } },
            hovertemplate: 'θ=%{x}°<br>NWR no-noise=%{y:.4f}<extra>NWR no noise</extra>',
          });
        }
      }
    }
  }

  if (traces.length === 0) { noData(plotId); return; }

  const paramLabel = PARAM_LABELS[param] || param;
  const adcLabel   = adc === '0' ? 'no ADC cut' : 'ADC ≥ ' + adc;
  const allThetas  = [...new Set([...(wrAB ? wrAB.thetas : []), ...(nwrAB ? nwrAB.thetas : [])])].sort((a,b) => a-b);

  Plotly.react(plotId, traces, {
    title: { text: paramLabel + '  —  ' + adcLabel + '  —  Bias vs angle (pivot x=1000 mm)  [' + grpLabel + ']',
             font:{size:13}, x:0.5 },
    xaxis: {
      title: { text: 'Track angle θ (degrees)', standoff:8 },
      tickvals: allThetas, ticktext: allThetas.map(t => t + '°'),
      gridcolor:'#eee', zeroline:false,
    },
    yaxis: {
      title: { text: yAxisTitle, standoff:8 },
      gridcolor:'#eee', zeroline:false,
    },
    margin: {t:50, b:60, l:75, r:20},
    paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    showlegend: true,
    legend: { x:1.01, xanchor:'left', y:0.99, font:{size:11},
              bgcolor:'rgba(255,255,255,0.8)', bordercolor:'#ddd', borderwidth:1 },
    shapes: [
      ...(isMean ? [{ type:'line', y0:1, y1:1, x0:0, x1:1, xref:'paper',
        line:{color:'#888', width:1.2, dash:'dot'} }] : []),
      { type:'line', x0:0, x1:0, y0:0, y1:1, yref:'paper',
        line:{color:'#ddd', width:1, dash:'dot'} },
    ],
    annotations: [
      ...(isMean ? [{ x:0.5, y:1, yref:'y', text:'GT (factor=1)', showarrow:false,
        xref:'paper', yanchor:'bottom', font:{size:10, color:'#888'} }] : []),
      { x:0, y:1, yref:'paper', text:'θ=0° (pure drift)', showarrow:false,
        xanchor:'left', xshift:4, yshift:-14, font:{size:9, color:'#aaa'} },
    ],
  }, { responsive:true, displayModeBar:true });
  _bindHoverHist(plotId);

  const tableEl = document.getElementById('table-anglepivot');
  if (tableEl && nwrAB) {
    const nwrThetas = nwrAB.thetas;
    const sorted = nwrThetas.map((t) => {
      const entry = nwrAB.data[param + '|' + adc + '|' + t];
      const s = _biasGroupStats(entry, grp);
      const st = _metricStats(s.vals, metric, s.mean, s.std);
      const sem = (s.std != null && s.n > 0) ? s.std / Math.sqrt(s.n) : null;
      return { t, mean: st.value, std: s.std, n: s.n, sem };
    }).filter(r => r.mean !== null).sort((a,b) => a.t - b.t);
    const rows = sorted.map(r => {
      const nn  = _nwrSrc.nonoise_angle_pivot_nwr_bias && _nwrSrc.nonoise_angle_pivot_nwr_bias.data[param + '|' + adc + '|' + r.t];
      const nnF = _biasNNFactor(nn, grp);
      const nnCell = nnF != null ? nnF.toFixed(5) : '—';
      const stdStr = isMean ? r.std.toFixed(5) : '—';
      const semStr = isMean && r.sem != null ? r.sem.toFixed(5) : '—';
      const zStr   = isMean && r.sem != null ? ((r.mean - 1) / r.sem).toFixed(2) : '—';
      return `<tr><td>${r.t}°</td><td>${r.mean.toFixed(5)}</td><td>${stdStr}</td><td>${r.n}</td><td>${semStr}</td><td>${zStr}</td><td>${nnCell}</td></tr>`;
    }).join('');
    tableEl.innerHTML = `<table style="border-collapse:collapse;font-size:0.82rem;width:auto;">
      <thead><tr style="background:#f0f2f5;">
        <th style="padding:4px 14px;border:1px solid #ddd;">θ</th>
        <th style="padding:4px 14px;border:1px solid #ddd;">NWR ${METRIC_LABELS[metric] || metric}</th>
        <th style="padding:4px 14px;border:1px solid #ddd;">std</th>
        <th style="padding:4px 14px;border:1px solid #ddd;">n seeds</th>
        <th style="padding:4px 14px;border:1px solid #ddd;">SEM</th>
        <th style="padding:4px 14px;border:1px solid #ddd;">z=(mean−1)/SEM</th>
        <th style="padding:4px 14px;border:1px solid #ddd;">no-noise factor</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  } else if (tableEl) {
    tableEl.innerHTML = '';
  }
}

// ── Tabs: bias vs (θ, α) and (θ, α) pivot ────────────────────────────────
const _TA_COLORSCALE = [
  [0.0,  'rgb(103,0,31)'],   [0.25, 'rgb(214,96,77)'],
  [0.45, 'rgb(253,219,199)'],[0.5,  'rgb(247,247,247)'],
  [0.55, 'rgb(209,229,240)'],[0.75, 'rgb(67,147,195)'],
  [1.0,  'rgb(5,48,97)'],
];
// Sequential colorscales for "lower is better" / "higher is better" metrics
// (white = good in both cases).
const _SEQ_WHITE_BLUE = [[0, 'rgb(247,247,247)'], [1, 'rgb(5,48,97)']];
const _SEQ_BLUE_WHITE = [[0, 'rgb(5,48,97)'], [1, 'rgb(247,247,247)']];

// Global "recovered factor" metrics, selected from the header picker and
// applied consistently across the bias/anglebias/anglepivot/ta/tapivot tabs.
// kind: 'mean' → factor value centered on GT=1 (diverging colorscale; error
//                bars/bands shown for plain 'mean' only)
//       'dist' → avg distance |factor−1|, 0 = best (white→blue, no error bars)
//       'pct'  → % of seeds within a tolerance band, 100% = best (blue→white)
const METRIC_LABELS = {
  mean:       'Mean factor',
  mean_in10:  'Mean factor (within 0.9–1.1)',
  mean_in19:  'Mean factor (within 0.81–1.19)',
  mad1:       'Avg |factor − 1|',
  mad1_in10:  'Avg |factor − 1| (within 0.9–1.1)',
  mad1_in19:  'Avg |factor − 1| (within 0.81–1.19)',
  mad1_out10: 'Avg |factor − 1| (outside 0.9–1.1)',
  mad1_out19: 'Avg |factor − 1| (outside 0.81–1.19)',
  pct_in10:   '% within 0.9–1.1',
  pct_in19:   '% within 0.81–1.19',
};
const METRIC_KIND = {
  mean: 'mean', mean_in10: 'mean', mean_in19: 'mean',
  mad1: 'dist', mad1_in10: 'dist', mad1_in19: 'dist', mad1_out10: 'dist', mad1_out19: 'dist',
  pct_in10: 'pct', pct_in19: 'pct',
};

// Computes {value, err, n, nSub} for the selected metric from a cell/x-position's
// raw [seed, factor] pairs. 'mean' keeps the precomputed mean/std (used as the
// error bar/band); every other metric gets err=0, optionally restricted to the
// in-/out-of-band subset of `vals` for the chosen tolerance band.
function _metricStats(vals, metric, mean, std) {
  const n = vals ? vals.length : 0;
  if (n === 0) return { value: null, err: 0, n: 0, nSub: 0 };
  if (metric === 'mean') return { value: mean, err: std || 0, n, nSub: n };
  const fs = vals.map(v => v[1]);
  const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
  if (metric === 'mad1') return { value: avg(fs.map(f => Math.abs(f - 1))), err: 0, n, nSub: n };
  const halfWidth = metric.endsWith('19') ? 0.19 : 0.1;
  if (metric.startsWith('mean_in')) {
    const sub = fs.filter(f => Math.abs(f - 1) <= halfWidth);
    return { value: avg(sub), err: 0, n, nSub: sub.length };
  }
  if (metric.startsWith('mad1_in')) {
    const sub = fs.filter(f => Math.abs(f - 1) <= halfWidth).map(f => Math.abs(f - 1));
    return { value: avg(sub), err: 0, n, nSub: sub.length };
  }
  if (metric.startsWith('mad1_out')) {
    const sub = fs.filter(f => Math.abs(f - 1) > halfWidth).map(f => Math.abs(f - 1));
    return { value: avg(sub), err: 0, n, nSub: sub.length };
  }
  if (metric.startsWith('pct_in')) {
    const nSub = fs.filter(f => Math.abs(f - 1) <= halfWidth).length;
    return { value: 100 * nSub / n, err: 0, n, nSub };
  }
  return { value: mean, err: std || 0, n, nSub: n };
}

function _extractTaData(ab, param, adc, grp, metric) {
  if (!ab) return null;
  grp = grp || 'all';
  metric = metric || 'mean';
  const { thetas, alphas } = ab;
  const _get = (t, a) => ab.data[param + '|' + adc + '|' + t + '|' + a];
  const _gs  = (t, a) => _biasGroupStats(_get(t, a), grp);
  const cellStats = thetas.map(t => alphas.map(a => {
    const s = _gs(t, a);
    return _metricStats(s.vals, metric, s.mean, s.std);
  }));
  const z     = cellStats.map(row => row.map(c => c.value));
  const zN    = cellStats.map(row => row.map(c => c.n));
  const zSub  = cellStats.map(row => row.map(c => c.nSub));
  const zVals = thetas.map(t => alphas.map(a => _gs(t,a).vals));
  const zStd  = thetas.map(t => alphas.map(a => _gs(t,a).std));
  const zSEM  = thetas.map((_,i) => alphas.map((_,j) =>
    (zStd[i][j] != null && zN[i][j] > 0) ? zStd[i][j] / Math.sqrt(zN[i][j]) : null));
  const kind  = METRIC_KIND[metric] || 'mean';
  const hovertext = thetas.map((t, i) => alphas.map((a, j) => {
    const v = z[i][j], n = zN[i][j], nSub = zSub[i][j];
    if (v === null) return '(no data)';
    if (kind === 'pct') {
      return `θ=${t}°  α=${a}°<br>${v.toFixed(1)}% within band  n=${nSub}/${n}`;
    }
    if (kind === 'dist') {
      return `θ=${t}°  α=${a}°<br>${METRIC_LABELS[metric]}=${v.toFixed(4)}  n=${nSub}/${n}`;
    }
    if (metric !== 'mean') {
      return `θ=${t}°  α=${a}°<br>${METRIC_LABELS[metric]}=${v.toFixed(4)}  n=${nSub}/${n}`;
    }
    const s = zStd[i][j], sem = zSEM[i][j];
    const zs = sem != null ? ((v - 1) / sem).toFixed(2) : '—';
    return `θ=${t}°  α=${a}°<br>mean=${v.toFixed(4)}  std=${s != null ? s.toFixed(4) : '—'}  SEM=${sem != null ? sem.toFixed(5) : '—'}  n=${n}<br>z=(mean−1)/SEM=${zs}`;
  }));
  return { thetas, alphas, z, zStd, zN, zSEM, zSub, zVals, hovertext, metric };
}

// Heatmap colorscale + zmin/zmid/zmax + colorbar title for the selected metric.
// 'mean' kind is centered on GT = 1 (diverging, white = on target).
// 'dist' kind (0 = best) uses white→blue, sequential.
// 'pct' kind (100% = best) uses blue→white, sequential, fixed 0–100 range.
function _heatmapColorOpts(metric, allZ) {
  const kind = METRIC_KIND[metric] || 'mean';
  if (kind === 'mean') {
    const halfR = allZ.length ? Math.max(...allZ.map(v => Math.abs(v - 1)), 0.01) : 0.2;
    return { colorscale: _TA_COLORSCALE, zmid: 1.0, zmin: 1 - halfR, zmax: 1 + halfR,
             colorbarTitle: METRIC_LABELS[metric] };
  }
  if (kind === 'pct') {
    return { colorscale: _SEQ_BLUE_WHITE, zmin: 0, zmax: 100,
             colorbarTitle: METRIC_LABELS[metric] };
  }
  const zmax = allZ.length ? Math.max(...allZ, 1e-3) : 0.2;
  return { colorscale: _SEQ_WHITE_BLUE, zmin: 0, zmax, colorbarTitle: METRIC_LABELS[metric] };
}

function _buildTaAnnotations(d, xref, yref, fontSize) {
  const ann = [];
  const metric = d.metric || 'mean';
  const kind = METRIC_KIND[metric] || 'mean';
  d.thetas.forEach((theta, i) => {
    d.alphas.forEach((alpha, j) => {
      const v = d.z[i][j];
      if (v === null) return;
      const nSub = d.zSub[i][j], n = d.zN[i][j];
      let text;
      if (kind === 'pct') {
        text = v.toFixed(1) + '%';
      } else if (metric === 'mean') {
        const sem = d.zSEM[i][j];
        const zscore = sem != null ? (v - 1) / sem : null;
        text = v.toFixed(4) + (zscore != null
          ? '<br><span style="font-size:' + (fontSize - 2) + 'px;color:#555">z=' + zscore.toFixed(1) + '</span>'
          : '');
      } else if (metric === 'mad1') {
        text = v.toFixed(4);
      } else {
        text = v.toFixed(4) + '<br><span style="font-size:' + (fontSize - 2) + 'px;color:#555">n=' + nSub + '/' + n + '</span>';
      }
      ann.push({
        x: alpha, y: theta, xref, yref, text,
        showarrow: false, font: { size: fontSize, color: '#333' },
      });
    });
  });
  return ann;
}

function updateThetaAlphaBias(study) {
  const pfx       = study === 'ta' ? 'ta' : 'tap';
  const param     = selVal(pfx + '-param');
  const adc       = selVal(pfx + '-adc');
  const grp       = selVal(pfx + '-plane-group') || 'all';
  const metric    = selVal('global-metric') || 'mean';
  const plotId    = 'plot-' + study;
  const dataKey   = study === 'ta' ? 'theta_alpha_bias' : 'theta_alpha_pivot_bias';
  const nnDataKey = study === 'ta' ? 'nonoise_theta_alpha_bias' : 'nonoise_theta_alpha_pivot_bias';

  const paramLabel = PARAM_LABELS[param] || param;
  const adcLabel   = adc === '0' ? 'no ADC cut' : 'ADC ≥ ' + adc;
  const pivotLabel = study === 'tapivot' ? ' (pivot x=1000)' : '';
  const grpLabel   = grp === 'all' ? '' : '  [' + grp + ' planes]';
  const metricTitle = METRIC_LABELS[metric] || METRIC_LABELS.mean;

  const inBothMode = (typeof currentDataset !== 'undefined' && currentDataset === 'both'
                      && typeof DATASETS !== 'undefined');

  if (inBothMode) {
    // ── Two heatmaps side by side: WR left, NWR right ────────────────────
    const dFull = _extractTaData(DATASETS.full && DATASETS.full[dataKey], param, adc, grp, metric);
    const dNwr  = _extractTaData(DATASETS.nwr  && DATASETS.nwr[dataKey],  param, adc, grp, metric);

    if ((!dFull || dFull.z.flat().every(v => v === null))
     && (!dNwr  || dNwr.z.flat().every(v => v === null))) { noData(plotId); return; }

    const ref = dNwr || dFull;
    const { thetas, alphas } = ref;

    // Shared color range across both heatmaps
    const allZ = [];
    [dFull, dNwr].forEach(d => d && d.z.forEach(row => row.forEach(v => { if (v !== null) allZ.push(v); })));
    const { colorscale, zmid, zmin, zmax, colorbarTitle } = _heatmapColorOpts(metric, allZ);

    // Combined taXRange for histogram toggle
    const _allFactors = [];
    [dFull, dNwr].forEach(d => d && d.zVals.forEach(row => row.forEach(cell => cell.forEach(sv => _allFactors.push(sv[1])))));
    taXRange[study] = _allFactors.length > 0 ? [Math.min(..._allFactors), Math.max(..._allFactors)] : null;

    const traces = [], annotations = [];

    if (dFull && !dFull.z.flat().every(v => v === null)) {
      traces.push({
        type: 'heatmap', x: alphas, y: thetas, z: dFull.z,
        text: dFull.hovertext, customdata: dFull.zVals,
        hovertemplate: '%{text}<extra></extra>',
        colorscale, zmid, zmin, zmax,
        showscale: false, xaxis: 'x', yaxis: 'y',
      });
      annotations.push(..._buildTaAnnotations(dFull, 'x', 'y', 9));
    }
    if (dNwr && !dNwr.z.flat().every(v => v === null)) {
      traces.push({
        type: 'heatmap', x: alphas, y: thetas, z: dNwr.z,
        text: dNwr.hovertext, customdata: dNwr.zVals,
        hovertemplate: '%{text}<extra></extra>',
        colorscale, zmid, zmin, zmax,
        showscale: true,
        colorbar: { title: {text:colorbarTitle, side:'right'}, thickness:14, x: 1.02 },
        xaxis: 'x2', yaxis: 'y2',
      });
      annotations.push(..._buildTaAnnotations(dNwr, 'x2', 'y2', 9));
    }
    annotations.push(
      { text: 'Wire response ON',  xref:'paper', yref:'paper', x: 0.225, y: 1.04,
        xanchor:'center', yanchor:'bottom', showarrow: false, font:{size:11, color:'#333'} },
      { text: 'Wire response OFF', xref:'paper', yref:'paper', x: 0.775, y: 1.04,
        xanchor:'center', yanchor:'bottom', showarrow: false, font:{size:11, color:'#333'} }
    );

    Plotly.react(plotId, traces, {
      title: { text: paramLabel + '  —  ' + adcLabel + '  —  ' + metricTitle + pivotLabel + grpLabel,
               font:{size:13}, x:0.5 },
      xaxis:  { title:{text:'α (°)', standoff:6}, domain:[0, 0.45],
                tickvals:alphas, ticktext:alphas.map(a => a + '°'), gridcolor:'#eee', zeroline:false },
      yaxis:  { title:{text:'θ — in-plane angle (°)', standoff:6}, domain:[0, 1],
                tickvals:thetas, ticktext:thetas.map(t => t + '°'), gridcolor:'#eee', zeroline:false },
      xaxis2: { title:{text:'α (°)', standoff:6}, domain:[0.55, 1],
                tickvals:alphas, ticktext:alphas.map(a => a + '°'), gridcolor:'#eee', zeroline:false },
      yaxis2: { domain:[0, 1],
                tickvals:thetas, ticktext:thetas.map(t => t + '°'),
                gridcolor:'#eee', zeroline:false, showticklabels:false },
      margin: {t:70, b:65, l:80, r:90},
      paper_bgcolor:'#fff', plot_bgcolor:'#fff',
      annotations,
    }, { responsive:true, displayModeBar:true });
    _bindTaHoverHist(plotId, study);
    document.getElementById('table-' + study).innerHTML = '';
    return;
  }

  // ── Single-dataset mode ─────────────────────────────────────────────────
  const ab = DATA[dataKey];
  if (!ab) { noData(plotId); return; }

  const d = _extractTaData(ab, param, adc, grp, metric);
  const { thetas, alphas, z, zVals, hovertext } = d;

  // Compute global x range for histogram toggle
  const _allFactors = [];
  zVals.forEach(row => row.forEach(cell => cell.forEach(sv => _allFactors.push(sv[1]))));
  taXRange[study] = _allFactors.length > 0 ? [Math.min(..._allFactors), Math.max(..._allFactors)] : null;

  if (z.flat().every(v => v === null)) { noData(plotId); return; }

  const allZ = z.flat().filter(v => v !== null);
  const { colorscale, zmid, zmin, zmax, colorbarTitle } = _heatmapColorOpts(metric, allZ);

  Plotly.react(plotId, [{
    type: 'heatmap', x: alphas, y: thetas, z,
    text: hovertext, customdata: zVals,
    hovertemplate: '%{text}<extra></extra>',
    colorscale,
    ...(zmid !== undefined ? { zmid } : {}),
    ...(METRIC_KIND[metric] === 'mean' ? {} : { zmin, zmax }),
    showscale: true,
    colorbar: { title: {text:colorbarTitle, side:'right'}, thickness:16 },
  }], {
    title: { text: paramLabel + '  —  ' + adcLabel + '  —  ' + metricTitle + pivotLabel + grpLabel,
             font:{size:13}, x:0.5 },
    xaxis: { title:{text:'α — out-of-plane angle (degrees)', standoff:8},
             tickvals:alphas, ticktext:alphas.map(a => a + '°'), gridcolor:'#eee', zeroline:false },
    yaxis: { title:{text:'θ — in-plane angle (degrees)', standoff:8},
             tickvals:thetas, ticktext:thetas.map(t => t + '°'), gridcolor:'#eee', zeroline:false },
    margin: {t:50, b:65, l:80, r:80},
    paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    annotations: _buildTaAnnotations(d, 'x', 'y', 11),
  }, { responsive:true, displayModeBar:true });
  _bindTaHoverHist(plotId, study);

  // Summary table sorted by (theta, alpha)
  const rows = [];
  const nnD = DATA[nnDataKey] && DATA[nnDataKey].data ? DATA[nnDataKey].data : null;
  thetas.slice().sort((a,b) => a-b).forEach(theta => {
    alphas.slice().sort((a,b) => a-b).forEach(alpha => {
      const e  = ab.data[param + '|' + adc + '|' + theta + '|' + alpha];
      const s  = _biasGroupStats(e, grp);
      if (s.mean !== null) {
        const nn  = nnD ? nnD[param + '|' + adc + '|' + theta + '|' + alpha] : null;
        const nnF = _biasNNFactor(nn, grp);
        const nnCell = nnF != null ? nnF.toFixed(5) : '—';
        const sem    = s.n > 0 && s.std != null ? s.std / Math.sqrt(s.n) : null;
        const semStr = sem != null ? sem.toFixed(5) : '—';
        const zStr   = sem != null ? ((s.mean - 1) / sem).toFixed(2) : '—';
        const fs      = s.vals.map(v => v[1]);
        const mad1Str = (fs.reduce((a,b) => a + Math.abs(b - 1), 0) / fs.length).toFixed(5);
        const pctStr  = (100 * fs.filter(f => Math.abs(f - 1) > 0.1).length / fs.length).toFixed(1) + '%';
        rows.push(`<tr><td>${theta}°</td><td>${alpha}°</td>`
          + `<td>${s.mean.toFixed(5)}</td><td>${s.std != null ? s.std.toFixed(5) : '—'}</td><td>${s.n}</td><td>${semStr}</td><td>${zStr}</td>`
          + `<td>${mad1Str}</td><td>${pctStr}</td><td>${nnCell}</td></tr>`);
      }
    });
  });
  document.getElementById('table-' + study).innerHTML =
    `<table style="border-collapse:collapse;font-size:0.82rem;width:auto;">
      <thead><tr style="background:#f0f2f5;">
        <th style="padding:4px 12px;border:1px solid #ddd;">θ</th>
        <th style="padding:4px 12px;border:1px solid #ddd;">α</th>
        <th style="padding:4px 12px;border:1px solid #ddd;">mean factor</th>
        <th style="padding:4px 12px;border:1px solid #ddd;">std</th>
        <th style="padding:4px 12px;border:1px solid #ddd;">n seeds</th>
        <th style="padding:4px 12px;border:1px solid #ddd;">SEM</th>
        <th style="padding:4px 12px;border:1px solid #ddd;">z=(mean−1)/SEM</th>
        <th style="padding:4px 12px;border:1px solid #ddd;">avg |f−1|</th>
        <th style="padding:4px 12px;border:1px solid #ddd;">% |f−1|&gt;0.1</th>
        <th style="padding:4px 12px;border:1px solid #ddd;">no-noise factor</th>
      </tr></thead>
      <tbody>${rows.join('')}</tbody>
    </table>`;
}

// ── Theta-alpha per-pane seed histogram ───────────────────────────────────
function setTaSameX(study, val) {
  taSameXAxis[study] = val;
  const sfx = study === 'ta' ? 'ta' : 'tap';
  document.getElementById(sfx + '-xaxis-auto').classList.toggle('active', !val);
  document.getElementById(sfx + '-xaxis-fixed').classList.toggle('active', val);
  const last = taHistLast[study];
  if (last.vals) showTaHist(study, last.vals, last.title);
}

function showTaHist(study, vals, title) {
  if (!vals || vals.length === 0) return;
  taHistLast[study] = {vals, title};
  const vs = vals.map(sv => sv[1]);
  const n  = vs.length;
  const mn = vs.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(vs.reduce((a, b) => a + (b - mn) * (b - mn), 0) / n);
  document.getElementById('taHist-' + study + '-title').textContent = title || '';
  document.getElementById('taHist-' + study + '-stats').textContent =
    'n=' + n + '   mean=' + mn.toFixed(4) + '   std=' + sd.toFixed(4);
  document.getElementById('taHist-' + study).style.display = '';
  const xRange = (taSameXAxis[study] && taXRange[study]) ? taXRange[study] : null;
  const _nBins = 25;
  const _binSpec = xRange
    ? { xbins: { start: xRange[0], end: xRange[1], size: (xRange[1] - xRange[0]) / _nBins }, autobinx: false }
    : { nbinsx: Math.min(_nBins, n) };
  Plotly.react('taHist-' + study + '-plot', [Object.assign({
    type: 'histogram', x: vs,
    marker: { color: 'rgba(66,133,244,0.60)', line: { color: '#446', width: 0.5 } },
  }, _binSpec)], {
    margin: {t:4, b:32, l:42, r:10},
    paper_bgcolor:'#f5f7ff', plot_bgcolor:'#f5f7ff',
    xaxis: Object.assign({ title:{text:'recovered factor',standoff:3}, gridcolor:'#dde',
             zeroline:true, zerolinecolor:'#bbc', tickfont:{size:9} },
             xRange ? {range: xRange} : {}),
    yaxis: { title:{text:'seeds',standoff:3}, gridcolor:'#dde', tickfont:{size:9} },
    bargap: 0.04,
    shapes: [
      { type:'line', x0:1,  x1:1,  y0:0, y1:1, yref:'paper',
        line:{color:'#c33', width:1.5, dash:'dot'} },
      { type:'line', x0:mn, x1:mn, y0:0, y1:1, yref:'paper',
        line:{color:'#335', width:1.5} },
    ],
    annotations: [
      { x:mn, y:0.97, yref:'paper', text:'mean', showarrow:false,
        xanchor:'left', xshift:3, font:{size:9, color:'#335'} },
    ],
  }, { responsive:true, displayModeBar:false });
}
function hideTaHist(study) {
  const wrap = document.getElementById('taHist-' + study);
  wrap.style.display = 'none';
  wrap._pinned = false;
}
function _bindTaHoverHist(plotId, study) {
  const el = document.getElementById(plotId);
  if (!el || el._taHistBound) return;
  el._taHistBound = true;
  const histWrap = document.getElementById('taHist-' + study);
  el.on('plotly_hover', function(data) {
    if (histWrap._pinned) return;
    const pt = data.points[0];
    showTaHist(study, pt.customdata,
      'θ=' + pt.y + '°  α=' + pt.x + '°  — hover to explore, click to pin');
  });
  el.on('plotly_unhover', function() {
    if (!histWrap._pinned) hideTaHist(study);
  });
  el.on('plotly_click', function(data) {
    histWrap._pinned = true;
    const pt = data.points[0];
    showTaHist(study, pt.customdata,
      'θ=' + pt.y + '°  α=' + pt.x + '°  (pinned — click ✕ to close)');
  });
}

// ── init ──────────────────────────────────────────────────────────────────
function init() {
  const adcOptions = ADC_CUTOFFS.map(String);
  const adcLabels  = Object.fromEntries(
    ADC_CUTOFFS.map(a => [String(a), a === 0 ? '0 (no cut)' : String(a)])
  );

  ['startx', 'angle'].forEach(study => {
    const pfx  = study === 'startx' ? 'sx' : 'an';
    const info = DATA[study];
    populateSelect(pfx + '-param',  PARAMS,        PARAM_LABELS);
    populateSelect(pfx + '-adc',    adcOptions,    adcLabels);
    populateSelect(pfx + '-track',  info.tracks,   info.trackLabels);
  });

  populateSelect('bi-param', PARAMS,       PARAM_LABELS);
  populateSelect('bi-adc',   adcOptions,   adcLabels);
  populateSelect('ab-param', PARAMS,       PARAM_LABELS);
  populateSelect('ab-adc',   adcOptions,   adcLabels);
  populateSelect('global-metric', Object.keys(METRIC_LABELS), METRIC_LABELS);
  __INIT_ANGLE_PIVOT_JS__
  __INIT_PIVOT_NWR_JS__
  __INIT_TA_JS__

  document.querySelectorAll('select').forEach(el => el.addEventListener('change', saveState));
  __INIT_DATASET_SELECT__
  loadState();
}

init();
</script>
</body>
</html>
"""


_ANGLE_PIVOT_TAB_BTN = (
    '<button class="tab-btn" data-mode="both" onclick="switchTab(\'anglepivot\',this)">'
    'Bias vs Angle (pivot x=1000)</button>'
)
_ANGLE_PIVOT_PANE = """\
<!-- ── Tab: bias vs angle (pivot x=1000, combined WR + NWR) ─────────────── -->
<div id="pane-anglepivot" class="tab-pane">
  <div class="controls">
    <div class="ctrl-group">
      <label>Parameter</label>
      <select id="ap-param" onchange="updateAnglePivotCombined()"></select>
    </div>
    <div class="ctrl-group">
      <label>ADC cutoff</label>
      <select id="ap-adc" onchange="updateAnglePivotCombined()"></select>
    </div>
    <div class="ctrl-group">
      <label>Wireplane group</label>
      <select id="ap-plane-group" onchange="updateAnglePivotCombined()">
        <option value="all">All planes</option>
        <option value="U">U planes</option>
        <option value="V">V planes</option>
        <option value="Y">Y planes</option>
      </select>
    </div>
  </div>
  <div class="plot-wrap">
    <div id="plot-anglepivot" style="height:520px"></div>
  </div>
  <div id="table-anglepivot" style="margin-top:14px; overflow-x:auto;"></div>
  <p style="margin-top:10px; font-size:0.8rem; color:#888; padding: 0 4px;">
    All tracks: 400 MeV muon, midpoint fixed at (1000, 0, 0) mm (west volume, x = 1000 mm from cathode).
    Start: (1000 + 850.3·cos θ, −850.3·sin θ, 0) mm.  Direction: dx = −cos θ, dy = sin θ, dz = 0.
    WR (solid blue) = with wire response; NWR (dashed orange) = no wire response (delta kernels).
    Mean ± 1σ over __BIAS_SEED_COUNT__ noise seeds.
  </p>
</div>
"""
_INIT_ANGLE_PIVOT_JS = """\
  populateSelect('ap-param', PARAMS,       PARAM_LABELS);
  populateSelect('ap-adc',   adcOptions,   adcLabels);
"""
_PIVOT_NWR_TAB_BTN = ''
_PIVOT_NWR_PANE    = ''
_INIT_PIVOT_NWR_JS = ''

_TA_TAB_BTN = (
    '<button class="tab-btn" data-mode="nwr" onclick="switchTab(\'ta\',this)">'
    'Bias vs (&theta;,&alpha;)</button>'
)
_TA_PIVOT_TAB_BTN = (
    '<button class="tab-btn" data-mode="nwr" onclick="switchTab(\'tapivot\',this)">'
    'Bias vs (&theta;,&alpha;) pivot</button>'
)
_TA_PANE = """\
<!-- ── Tab: theta × alpha bias heatmap ──────────────────────────────────── -->
<div id="pane-ta" class="tab-pane">
  <div class="controls">
    <div class="ctrl-group">
      <label>Parameter</label>
      <select id="ta-param" onchange="updateThetaAlphaBias('ta')"></select>
    </div>
    <div class="ctrl-group">
      <label>ADC cutoff</label>
      <select id="ta-adc" onchange="updateThetaAlphaBias('ta')"></select>
    </div>
    <div class="ctrl-group">
      <label>Wireplane group</label>
      <select id="ta-plane-group" onchange="updateThetaAlphaBias('ta')">
        <option value="all">All planes</option>
        <option value="U">U planes</option>
        <option value="V">V planes</option>
        <option value="Y">Y planes</option>
      </select>
    </div>
    <div class="seg-group">
      <label>Histogram x-axis</label>
      <div class="seg-btns">
        <button id="ta-xaxis-auto" class="active" onclick="setTaSameX('ta', false)">Auto</button>
        <button id="ta-xaxis-fixed" onclick="setTaSameX('ta', true)">Same range</button>
      </div>
    </div>
  </div>
  <div class="plot-wrap">
    <div id="plot-ta" style="height:420px"></div>
  </div>
  <div id="taHist-ta" style="display:none;margin:8px 0 0;padding:8px 14px;background:#f5f7ff;border:1px solid #c8d0e8;border-radius:6px;">
    <div style="display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;">
      <span id="taHist-ta-title" style="font-size:0.82rem;color:#446;font-weight:600;"></span>
      <span id="taHist-ta-stats" style="font-size:0.82rem;color:#555;"></span>
      <button onclick="hideTaHist('ta')" style="margin-left:auto;font-size:0.75rem;background:none;border:none;cursor:pointer;color:#888;padding:2px 6px;">&#10005; close</button>
    </div>
    <div id="taHist-ta-plot" style="height:150px;margin-top:4px;"></div>
  </div>
  <div id="table-ta" style="margin-top:14px; overflow-x:auto;"></div>
  <p style="margin-top:10px; font-size:0.8rem; color:#888; padding: 0 4px;">
    400 MeV muon at (1900, 0, 0) mm; delta kernels (no wire response).
    &theta; = in-plane angle; &alpha; = out-of-plane angle.
    Mean &plusmn; 1&sigma; over __BIAS_SEED_COUNT__ noise seeds.
  </p>
</div>
"""
_TA_PIVOT_PANE = """\
<!-- ── Tab: theta × alpha pivot bias heatmap ────────────────────────────── -->
<div id="pane-tapivot" class="tab-pane">
  <div class="controls">
    <div class="ctrl-group">
      <label>Parameter</label>
      <select id="tap-param" onchange="updateThetaAlphaBias('tapivot')"></select>
    </div>
    <div class="ctrl-group">
      <label>ADC cutoff</label>
      <select id="tap-adc" onchange="updateThetaAlphaBias('tapivot')"></select>
    </div>
    <div class="ctrl-group">
      <label>Wireplane group</label>
      <select id="tap-plane-group" onchange="updateThetaAlphaBias('tapivot')">
        <option value="all">All planes</option>
        <option value="U">U planes</option>
        <option value="V">V planes</option>
        <option value="Y">Y planes</option>
      </select>
    </div>
    <div class="seg-group">
      <label>Histogram x-axis</label>
      <div class="seg-btns">
        <button id="tap-xaxis-auto" class="active" onclick="setTaSameX('tapivot', false)">Auto</button>
        <button id="tap-xaxis-fixed" onclick="setTaSameX('tapivot', true)">Same range</button>
      </div>
    </div>
  </div>
  <div class="plot-wrap">
    <div id="plot-tapivot" style="height:420px"></div>
  </div>
  <div id="taHist-tapivot" style="display:none;margin:8px 0 0;padding:8px 14px;background:#f5f7ff;border:1px solid #c8d0e8;border-radius:6px;">
    <div style="display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;">
      <span id="taHist-tapivot-title" style="font-size:0.82rem;color:#446;font-weight:600;"></span>
      <span id="taHist-tapivot-stats" style="font-size:0.82rem;color:#555;"></span>
      <button onclick="hideTaHist('tapivot')" style="margin-left:auto;font-size:0.75rem;background:none;border:none;cursor:pointer;color:#888;padding:2px 6px;">&#10005; close</button>
    </div>
    <div id="taHist-tapivot-plot" style="height:150px;margin-top:4px;"></div>
  </div>
  <div id="table-tapivot" style="margin-top:14px; overflow-x:auto;"></div>
  <p style="margin-top:10px; font-size:0.8rem; color:#888; padding: 0 4px;">
    400 MeV muon, midpoint fixed at (1000, 0, 0) mm; delta kernels (no wire response).
    &theta; = in-plane angle; &alpha; = out-of-plane angle.
    Mean &plusmn; 1&sigma; over __BIAS_SEED_COUNT__ noise seeds.
  </p>
</div>
"""
_INIT_TA_JS = """\
  populateSelect('ta-param',  PARAMS,     PARAM_LABELS);
  populateSelect('ta-adc',    adcOptions, adcLabels);
  populateSelect('tap-param', PARAMS,     PARAM_LABELS);
  populateSelect('tap-adc',   adcOptions, adcLabels);
"""


_SET_DATASET_FN = """\
let currentDataset = 'full';
function setDataset(name) {
  currentDataset = name;
  DATA = DATASETS[name === 'both' ? 'full' : name];
  BIAS_SEED_LABEL = DATASET_LABELS[name === 'both' ? 'full' : name];
  document.querySelectorAll('.ds-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.ds === name));
  let activeIsHidden = false;
  document.querySelectorAll('.tab-btn[data-mode]').forEach(btn => {
    const m = btn.dataset.mode;
    const vis = (name === 'both' || m === 'both' || m === name);
    btn.style.display = vis ? '' : 'none';
    if (!vis && btn.classList.contains('active')) activeIsHidden = true;
  });
  if (activeIsHidden) {
    const fb = document.querySelector('.tab-btn[data-mode="both"]');
    if (fb) { fb.click(); return; }
  }
  ['startx', 'angle'].forEach(function(study) {
    const pfx = study === 'startx' ? 'sx' : 'an';
    const info = DATA[study];
    if (info) populateSelect(pfx + '-track', info.tracks, info.trackLabels);
  });
  if (_loadingState) return;
  saveState();
  _renderActiveTab();
}
"""

def _sibling_link(href, label):
    return (f'<a href="{href}" style="font-size:0.82rem;color:#cce;border:1px solid #445;'
            f'border-radius:5px;padding:5px 10px;margin-left:6px;text-decoration:none;'
            f'vertical-align:middle;white-space:nowrap">{label}</a>')

_FNAME_FULL     = 'diffusion_loss_study.html'
_FNAME_NWR      = 'diffusion_loss_study_no_wire_response.html'
_FNAME_COMBINED = 'diffusion_loss_study_combined.html'

_SIBLING_LINKS = {
    'full':     (_sibling_link(_FNAME_NWR,      'No-wire-resp. →')
                 + _sibling_link(_FNAME_COMBINED, 'Combined →')),
    'no_wire_response': (_sibling_link(_FNAME_FULL,     'Wire-resp. →')
                         + _sibling_link(_FNAME_COMBINED, 'Combined →')),
    'combined': (_sibling_link(_FNAME_FULL, 'Wire-resp. →')
                 + _sibling_link(_FNAME_NWR,  'No-wire-resp. →')),
}

_DATASET_TOGGLE_HTML = (
    '<div style="display:flex;gap:6px;align-items:center;">'
    '<span style="font-size:0.8rem;color:#aac4;">Wire response:</span>'
    '<button class="ds-btn" data-ds="full" onclick="setDataset(\'full\')">On</button>'
    '<button class="ds-btn" data-ds="nwr" onclick="setDataset(\'nwr\')">Off</button>'
    '<button class="ds-btn active" data-ds="both" onclick="setDataset(\'both\')">Both</button>'
    '</div>'
)

_METRIC_PICKER_HTML = (
    '<div style="margin-left:auto;display:flex;gap:6px;align-items:center;">'
    '<span style="font-size:0.8rem;color:#aac4;">Metric:</span>'
    '<select id="global-metric" onchange="onGlobalMetricChange()" '
    'style="font-size:0.8rem;padding:3px 6px;border-radius:4px;border:1px solid #445;'
    'background:#1c2e4a;color:#fff;"></select>'
    '</div>'
)


def emit_html(studies, output_path, mode):
    bias_seed_label = f"{mode.all_seeds} seeds"
    seed_count = str(mode.all_seeds)

    def _pane(template):
        return template.replace("__BIAS_SEED_COUNT__", seed_count)

    data_decl = 'const DATA = ' + json.dumps(studies, separators=(',', ':')) + ';'
    bias_decl = 'const BIAS_SEED_LABEL = ' + json.dumps(bias_seed_label) + ';'

    replacements = {
        '__PAGE_TITLE__':           mode.page_title,
        '__DATA_DECL__':            data_decl,
        '__PARAMS_JSON__':          json.dumps(PARAMS),
        '__PARAM_LABELS_JSON__':    json.dumps(PARAM_LABELS),
        '__ADC_CUTOFFS_JSON__':     json.dumps(ADC_CUTOFFS),
        '__SEEDS_JSON__':           json.dumps(SEEDS),
        '__BIAS_SEED_LABEL_DECL__': bias_decl,
        '__SIBLING_LINKS__':         _SIBLING_LINKS.get(mode.name, ''),
        '__METRIC_PICKER__':        _METRIC_PICKER_HTML,
        '__DATASET_TOGGLE__':       '',
        '__SET_DATASET_FN__':       '',
        '__INIT_DATASET_SELECT__':  '',
        '__ANGLE_BIAS_FOOTNOTE__':  mode.angle_bias_footnote,
        '__ANGLE_PIVOT_TAB_BTN__':  _ANGLE_PIVOT_TAB_BTN if (mode.include_angle_pivot or mode.include_pivot_nwr) else '',
        '__ANGLE_PIVOT_PANE__':     _pane(_ANGLE_PIVOT_PANE) if (mode.include_angle_pivot or mode.include_pivot_nwr) else '',
        '__INIT_ANGLE_PIVOT_JS__':  _INIT_ANGLE_PIVOT_JS if (mode.include_angle_pivot or mode.include_pivot_nwr) else '',
        '__PIVOT_NWR_TAB_BTN__':    '',
        '__PIVOT_NWR_PANE__':       '',
        '__INIT_PIVOT_NWR_JS__':    '',
        '__TA_TAB_BTN__':           _TA_TAB_BTN if mode.include_theta_alpha else '',
        '__TA_PIVOT_TAB_BTN__':     _TA_PIVOT_TAB_BTN if mode.include_theta_alpha else '',
        '__TA_PANE__':              _pane(_TA_PANE) if mode.include_theta_alpha else '',
        '__TA_PIVOT_PANE__':        _pane(_TA_PIVOT_PANE) if mode.include_theta_alpha else '',
        '__INIT_TA_JS__':           _INIT_TA_JS if mode.include_theta_alpha else '',
    }
    html = _HTML
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding='utf-8')
    sz = Path(output_path).stat().st_size / 1024
    print(f"Written → {output_path}  ({sz:.0f} KB)")


def emit_combined_html(studies_full, studies_nwr, output_path):
    """Emit a combined HTML with a wire-response toggle."""
    mode_full = STUDY_MODES['full']
    mode_nwr  = STUDY_MODES['no_wire_response']
    sc_full   = str(mode_full.all_seeds)
    sc_nwr    = str(mode_nwr.all_seeds)

    data_decl = (
        'const DATASETS = {full: ' + json.dumps(studies_full, separators=(',', ':'))
        + ', nwr: '                 + json.dumps(studies_nwr,  separators=(',', ':')) + '};'
        '\nlet DATA = DATASETS[\'full\'];'
    )
    bias_decl = (
        'const DATASET_LABELS = {full: ' + json.dumps(f"{mode_full.all_seeds} seeds")
        + ', nwr: '                       + json.dumps(f"{mode_nwr.all_seeds} seeds") + '};'
        '\nlet BIAS_SEED_LABEL = DATASET_LABELS[\'full\'];'
    )

    def _pane(template, seed_count):
        return template.replace("__BIAS_SEED_COUNT__", seed_count)

    replacements = {
        '__PAGE_TITLE__':           'Diffusion Loss Studies (Combined)',
        '__DATA_DECL__':            data_decl,
        '__PARAMS_JSON__':          json.dumps(PARAMS),
        '__PARAM_LABELS_JSON__':    json.dumps(PARAM_LABELS),
        '__ADC_CUTOFFS_JSON__':     json.dumps(ADC_CUTOFFS),
        '__SEEDS_JSON__':           json.dumps(SEEDS),
        '__BIAS_SEED_LABEL_DECL__': bias_decl,
        '__ANGLE_BIAS_FOOTNOTE__':  (mode_full.angle_bias_footnote
                                     + ' / ' + mode_nwr.angle_bias_footnote),
        '__SIBLING_LINKS__':         _SIBLING_LINKS['combined'],
        '__METRIC_PICKER__':        _METRIC_PICKER_HTML,
        '__DATASET_TOGGLE__':       _DATASET_TOGGLE_HTML,
        '__SET_DATASET_FN__':       _SET_DATASET_FN,
        '__INIT_DATASET_SELECT__':  "_loadingState = true; setDataset('both'); _loadingState = false;",
        '__ANGLE_PIVOT_TAB_BTN__':  _ANGLE_PIVOT_TAB_BTN,
        '__ANGLE_PIVOT_PANE__':     _pane(_ANGLE_PIVOT_PANE, sc_nwr),
        '__INIT_ANGLE_PIVOT_JS__':  _INIT_ANGLE_PIVOT_JS,
        '__PIVOT_NWR_TAB_BTN__':    '',
        '__PIVOT_NWR_PANE__':       '',
        '__INIT_PIVOT_NWR_JS__':    '',
        '__TA_TAB_BTN__':           _TA_TAB_BTN,
        '__TA_PIVOT_TAB_BTN__':     _TA_PIVOT_TAB_BTN,
        '__TA_PANE__':              _pane(_TA_PANE, sc_nwr),
        '__TA_PIVOT_PANE__':        _pane(_TA_PIVOT_PANE, sc_nwr),
        '__INIT_TA_JS__':           _INIT_TA_JS,
    }
    html = _HTML
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding='utf-8')
    sz = Path(output_path).stat().st_size / 1024
    print(f"Written → {output_path}  ({sz:.0f} KB)")


def _default_cache(mode, name):
    suffix = mode.bias_cache_suffix
    return os.path.join(RESULTS_DIR, "1d_gradients", f"bias_{name}{suffix}_cache.pkl")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=list(STUDY_MODES) + ['combined'], default='full',
                   help='Study preset (default: full); "combined" loads both full+no_wire_response')
    p.add_argument('--output', default=None,
                   help='Output HTML path (default: $PLOTS_DIR/<mode output basename>)')
    p.add_argument('--drift-bias-cache', default=None, metavar='PATH',
                   help='Cache pkl for drift bias (default: mode-specific under $RESULTS_DIR/1d_gradients/)')
    p.add_argument('--angle-bias-cache', default=None, metavar='PATH',
                   help='Cache pkl for angle bias (default: mode-specific)')
    p.add_argument('--angle-pivot-bias-cache', default=None, metavar='PATH',
                   help='Cache pkl for angle-pivot bias (default: mode-specific; full mode only)')
    p.add_argument('--angle-pivot-nwr-bias-cache', default=None, metavar='PATH',
                   help='Cache pkl for angle-pivot-NWR bias (no_wire_response mode only)')
    p.add_argument('--theta-alpha-bias-cache', default=None, metavar='PATH',
                   help='Cache pkl for θ×α bias (no_wire_response mode only)')
    p.add_argument('--theta-alpha-pivot-bias-cache', default=None, metavar='PATH',
                   help='Cache pkl for θ×α pivot bias (no_wire_response mode only)')
    p.add_argument('--recompute-bias', action='store_true',
                   help='Ignore existing bias cache and recompute from individual pkl files')
    p.add_argument('--diagonal-only', action='store_true',
                   help='When recomputing θ×α bias, only process diagonal pairs (theta==alpha) '
                        'and merge into the existing cache; faster when only diagonal data changed')
    p.add_argument('--theta-alpha-only', action='store_true',
                   help='With --recompute-bias, only recompute the θ×α (pivot) bias caches; '
                        'all other bias caches (drift, angle, angle-pivot) are loaded from disk')
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == 'combined':
        output = args.output or os.path.join(PLOTS_DIR, 'diffusion_loss_study_combined.html')
        mode_full = STUDY_MODES['full']
        mode_nwr  = STUDY_MODES['no_wire_response']
        print("=== Building full-mode data ===")
        studies_full = build_data(
            mode_full,
            drift_bias_cache=_default_cache(mode_full, "drift"),
            angle_bias_cache=_default_cache(mode_full, "angle"),
            angle_pivot_bias_cache=_default_cache(mode_full, "angle_pivot"),
            angle_pivot_nwr_bias_cache=_default_cache(mode_full, "angle_pivot_nwr"),
            theta_alpha_bias_cache=_default_cache(mode_full, "theta_alpha"),
            theta_alpha_pivot_bias_cache=_default_cache(mode_full, "theta_alpha_pivot"),
            recompute_bias=args.recompute_bias,
            diagonal_only=args.diagonal_only,
            theta_alpha_only=args.theta_alpha_only,
        )
        print("=== Building no_wire_response-mode data ===")
        studies_nwr = build_data(
            mode_nwr,
            drift_bias_cache=_default_cache(mode_nwr, "drift"),
            angle_bias_cache=_default_cache(mode_nwr, "angle"),
            angle_pivot_bias_cache=_default_cache(mode_nwr, "angle_pivot"),
            angle_pivot_nwr_bias_cache=_default_cache(mode_nwr, "angle_pivot_nwr"),
            theta_alpha_bias_cache=_default_cache(mode_nwr, "theta_alpha"),
            theta_alpha_pivot_bias_cache=_default_cache(mode_nwr, "theta_alpha_pivot"),
            recompute_bias=args.recompute_bias,
            diagonal_only=args.diagonal_only,
            theta_alpha_only=args.theta_alpha_only,
        )
        emit_combined_html(studies_full, studies_nwr, output)
        return

    mode = STUDY_MODES[args.mode]
    output = args.output or os.path.join(PLOTS_DIR, mode.output_basename)
    studies = build_data(
        mode,
        drift_bias_cache=args.drift_bias_cache or _default_cache(mode, "drift"),
        angle_bias_cache=args.angle_bias_cache or _default_cache(mode, "angle"),
        angle_pivot_bias_cache=args.angle_pivot_bias_cache or _default_cache(mode, "angle_pivot"),
        angle_pivot_nwr_bias_cache=(args.angle_pivot_nwr_bias_cache
                                    or _default_cache(mode, "angle_pivot_nwr")),
        theta_alpha_bias_cache=(args.theta_alpha_bias_cache
                                or _default_cache(mode, "theta_alpha")),
        theta_alpha_pivot_bias_cache=(args.theta_alpha_pivot_bias_cache
                                      or _default_cache(mode, "theta_alpha_pivot")),
        recompute_bias=args.recompute_bias,
        diagonal_only=args.diagonal_only,
        theta_alpha_only=args.theta_alpha_only,
    )
    emit_html(studies, output, mode)


if __name__ == '__main__':
    main()
