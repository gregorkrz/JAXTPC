#!/usr/bin/env python
"""Patch incomplete pkl files for Adam_20260601_cutoff_sweep by fetching history from W&B.

For each pkl that has no completed trial (job killed mid-run), fetches the
param/loss/grad trajectory from W&B and injects a synthetic trial so the
viewer shows the full partial trajectory instead of a single checkpoint point.

The original pkl is backed up as result_<seed>.pkl.bak before writing.

Usage (on S3DF):
    /sdf/home/g/gregork/envs/base_env/bin/python \
        src/analysis/sim_param_sweeps/patch_pkls_from_wandb.py \
        --results-dir $RESULTS_DIR \
        [--dry-run]
"""

import argparse
import pickle
import shutil
import sys
from pathlib import Path

WANDB_ENTITY  = "fcc_ml"
WANDB_PROJECT = "jaxtpc-optimization"

ADC_CUTOFFS       = [5, 10, 20, 50]
FFT_CUTOFFS       = [0, 10, 100]
ROTATE_NOISE_VALS = [None, 5]
SEEDS             = [100, 101, 102, 103, 104]
PARAM_LABEL       = "trans_and_long"


def profile_tag(adc, fft, rot):
    adc_tag = f"adc{int(adc)}"
    fft_tag = f"ft{int(fft)}"
    rot_tag = "rotoff" if rot is None else f"rot{rot}"
    return f"Adam_20260601_cutoff_sweep_{PARAM_LABEL}_{adc_tag}_{fft_tag}_{rot_tag}"


def find_pkl(results_dir: Path, adc, fft, rot, seed):
    base    = results_dir / "opt" / profile_tag(adc, fft, rot) / "noise"
    matches = list(base.glob(f"*/result_{seed}.pkl"))
    return matches[0] if matches else None


def needs_patch(data: dict) -> bool:
    """True if the pkl has no completed trial."""
    trials = data.get("trials", [])
    return len(trials) == 0


def fetch_wandb_history(run_id: str, param_names: list):
    """Fetch full step-by-step history from W&B for the given run.

    Returns list of dicts sorted by _step, or None if run not found.
    """
    try:
        import wandb
        api  = wandb.Api(timeout=60)
        run  = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    except Exception as e:
        print(f"    W&B error fetching run {run_id}: {e}", file=sys.stderr)
        return None

    keys = ["_step", "loss"]
    for name in param_names:
        keys += [
            f"params/{name}_normalized",
            f"grads/{name}",
        ]

    try:
        rows = list(run.scan_history(keys=keys))
    except Exception as e:
        print(f"    W&B scan_history error for {run_id}: {e}", file=sys.stderr)
        return None

    rows = [r for r in rows if r.get("_step") is not None]
    rows.sort(key=lambda r: r["_step"])
    return rows if rows else None


def build_synthetic_trial(rows: list, param_names: list, p_n_gts: list) -> dict:
    """Convert W&B history rows into a trial dict matching run_optimization output."""
    import math

    param_traj = []
    grad_traj  = []
    loss_traj  = []

    for row in rows:
        pns = []
        gns = []
        for name in param_names:
            pn = row.get(f"params/{name}_normalized")
            gn = row.get(f"grads/{name}", 0.0)
            if pn is None:
                pn = 0.0  # shouldn't happen; fall back to GT
            pns.append(float(pn))
            gns.append(float(gn) if gn is not None else 0.0)
        param_traj.append(pns)
        grad_traj.append(gns)
        loss = row.get("loss")
        loss_traj.append(float(loss) if loss is not None else float("nan"))

    last_step   = int(rows[-1]["_step"]) if rows else 0
    step_indices = [int(r["_step"]) for r in rows]

    return dict(
        param_trajectory = param_traj,
        grad_trajectory  = grad_traj,
        loss_trajectory  = loss_traj,
        step_indices     = step_indices,   # actual wandb _step values (multiples of log_interval)
        steps_run        = len(param_traj) - 1,
        stopped_early    = True,           # killed before max_steps
        total_time_s     = 0.0,            # not recoverable
        from_wandb       = True,           # mark as reconstructed
        last_step        = last_step,
    )


def patch_pkl(path: Path, dry_run: bool) -> bool:
    """Load pkl, fetch W&B history, inject synthetic trial. Returns True on success."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    run_id = data.get("wandb_run_id")
    if not run_id:
        print(f"    no wandb_run_id in pkl — skipping", file=sys.stderr)
        return False

    param_names = data.get("param_names", [])
    p_n_gts     = data.get("p_n_gts", [])

    print(f"    fetching W&B history for run {run_id} ...", file=sys.stderr)
    rows = fetch_wandb_history(run_id, param_names)
    if not rows:
        print(f"    no history returned — skipping", file=sys.stderr)
        return False

    print(f"    got {len(rows)} steps (last={rows[-1]['_step']})", file=sys.stderr)
    trial = build_synthetic_trial(rows, param_names, p_n_gts)

    if dry_run:
        print(f"    [dry-run] would inject trial with {trial['steps_run']} logged intervals", file=sys.stderr)
        return True

    # Back up original
    bak = path.with_suffix(".pkl.bak")
    if not bak.exists():
        shutil.copy2(path, bak)

    data["trials"] = [trial]
    data.pop("live_checkpoint", None)   # replaced by the synthetic trial
    data["run_complete"] = False         # was killed; mark as incomplete

    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    print(f"    patched → {path}", file=sys.stderr)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()

    patched = skipped = missing = 0

    for rot in ROTATE_NOISE_VALS:
        for adc in ADC_CUTOFFS:
            for fft in FFT_CUTOFFS:
                for seed in SEEDS:
                    pkl = find_pkl(results_dir, adc, fft, rot, seed)
                    if pkl is None:
                        missing += 1
                        continue

                    with open(pkl, "rb") as f:
                        data = pickle.load(f)

                    if not needs_patch(data):
                        skipped += 1
                        continue

                    tag = profile_tag(adc, fft, rot)
                    print(f"  patching {tag} seed {seed}", file=sys.stderr)
                    ok = patch_pkl(pkl, dry_run=args.dry_run)
                    if ok:
                        patched += 1
                    else:
                        skipped += 1

    print(f"\nDone: {patched} patched, {skipped} skipped (already complete or no W&B), "
          f"{missing} pkl files not found", file=sys.stderr)


if __name__ == "__main__":
    main()
