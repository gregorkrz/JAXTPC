#!/usr/bin/env python
"""
Plot W&B optimization trajectories grouped by run tags.

For each requested tag, runs that contain that tag are fetched from a W&B project.
Each run is assigned to the first matching tag (in the provided tag order), and all
runs assigned to the same tag use the same color.

Optional ``--tag-labels`` or repeated ``--tag-label`` supply human-readable legend text
per tag (same order as ``--tags``); otherwise the raw W&B tag string is shown.

For each parameter, this script saves paired plots:
  - left: linear scale
  - right: log scale (or symlog when values include non-positive values)

Plot families (saved in separate directories):
  - gradients:         grads/<param>
  - physical_params:   params/<param>_physical
  - normalized_params: params/<param>_normalized
  - relative_errors:   abs((param_physical - GT) / GT)

Ground-truth reference lines (physical + normalized parameter panels only) come from the
same detector YAML and GT lifetime/velocity overrides as ``run_optimization.py``
(see ``--gt-config``).

Usage
-----
python src/plots/run_results.py \
  --tags baseline,noise0p1 \
  --entity your_entity \
  --project jaxtpc-optimization
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

plt = None
Line2D = None


_PLOTS_DIR = os.environ.get("PLOTS_DIR", "plots")
# Matplotlib's default savefig dpi is 100; use a higher default for W&B batch exports.
DEFAULT_PNG_DPI = 300

_KIND_TO_DIR = {
    "grads": "gradients",
    "params_physical": "physical_params",
    "params_normalized": "normalized_params",
    "params_relative_error": "relative_errors",
}

_KIND_TO_TITLE = {
    "grads": "gradient trajectories",
    "params_physical": "physical parameter trajectories",
    "params_normalized": "normalized parameter trajectories",
    "params_relative_error": "absolute relative error trajectories",
}

# Match ``run_optimization.py`` — GT lifetime/velocity overrides config file defaults.
GT_LIFETIME_US = 10_000.0
GT_VELOCITY_CM_US = 0.160

# Normalized W&B metrics are log(physical / scale); scales match run_optimization.TYPICAL_SCALES.

TYPICAL_SCALES = {
    "velocity_cm_us": 0.1,
    "lifetime_us": 10_000.0,
    "diffusion_trans_cm2_us": 1e-5,
    "diffusion_long_cm2_us": 1e-5,
    "recomb_alpha": 1.0,
    "recomb_beta": 0.2,
    "recomb_beta_90": 0.2,
    "recomb_R": 1.0,
}

_GT_PARAM_ORDER = (
    "velocity_cm_us",
    "lifetime_us",
    "diffusion_trans_cm2_us",
    "diffusion_long_cm2_us",
    "recomb_alpha",
    "recomb_beta",
    "recomb_beta_90",
    "recomb_R",
)


@dataclass
class Series:
    tag: str
    run_name: str
    run_id: str
    steps: np.ndarray
    values: np.ndarray


def _split_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _sanitize_for_path(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    safe = safe.strip("-")
    return safe or "tag"


def _build_default_output_dir(tags: List[str]) -> str:
    tags_part = "__".join(_sanitize_for_path(tag) for tag in tags)
    return os.path.join(_PLOTS_DIR, "wandb_run_results", tags_part)


def _build_color_map(tags: List[str], colors: Optional[List[str]]) -> Dict[str, str]:
    if colors is not None and len(colors) != len(tags):
        raise ValueError(
            f"--colors must contain exactly {len(tags)} values (got {len(colors)})"
        )
    if colors is not None:
        return {tag: color for tag, color in zip(tags, colors)}

    cmap = plt.cm.tab10
    return {
        tag: cmap(i / max(len(tags) - 1, 1))
        for i, tag in enumerate(tags)
    }


def _legend_order_tags(present_tags: List[str], tag_order: List[str]) -> List[str]:
    """Order tags for the legend: ``--tags`` order first, then any extras alphabetically."""
    present_set = set(present_tags)
    ordered = [t for t in tag_order if t in present_set]
    tail = sorted(present_set.difference(tag_order))
    return ordered + tail


def _pick_primary_tag(run_tags: List[str], requested_tags: List[str]) -> Optional[str]:
    run_tag_set = set(run_tags)
    for tag in requested_tags:
        if tag in run_tag_set:
            return tag
    return None


def _match_metric_key(key: str) -> Tuple[Optional[str], Optional[str]]:
    if key.startswith("grads/"):
        return "grads", key.split("/", 1)[1]
    if key.startswith("params/") and key.endswith("_physical"):
        param = key[len("params/") : -len("_physical")]
        return "params_physical", param
    if key.startswith("params/") and key.endswith("_normalized"):
        param = key[len("params/") : -len("_normalized")]
        return "params_normalized", param
    return None, None


def _is_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float, np.number))


def _extract_run_series(
    run: "wandb.apis.public.Run",
    assigned_tag: str,
) -> Dict[str, Dict[str, Series]]:
    # kind -> param -> list[(step, value)]
    values: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in run.scan_history():
        step = row.get("_step")
        if step is None:
            continue
        step_f = float(step)
        for key, val in row.items():
            kind, param = _match_metric_key(key)
            if kind is None or param is None or not _is_number(val):
                continue
            values[kind][param].append((step_f, float(val)))

    out: Dict[str, Dict[str, Series]] = defaultdict(dict)
    for kind, per_param in values.items():
        for param, points in per_param.items():
            if not points:
                continue
            points = sorted(points, key=lambda pair: pair[0])
            steps = np.array([p[0] for p in points], dtype=float)
            vals = np.array([p[1] for p in points], dtype=float)
            out[kind][param] = Series(
                tag=assigned_tag,
                run_name=run.name or run.id,
                run_id=run.id,
                steps=steps,
                values=vals,
            )
    return out


def _float_gt_physical(param_name: str, gt_params: Any, recomb_model: str) -> float:
    """Physical GT for one tunable name; mirrors ``run_optimization._get_gt_val``."""
    rp = gt_params.recomb_params
    if param_name == "velocity_cm_us":
        return float(gt_params.velocity_cm_us)
    if param_name == "lifetime_us":
        return float(gt_params.lifetime_us)
    if param_name == "diffusion_trans_cm2_us":
        return float(gt_params.diffusion_trans_cm2_us)
    if param_name == "diffusion_long_cm2_us":
        return float(gt_params.diffusion_long_cm2_us)
    if param_name == "recomb_alpha":
        return float(rp.alpha)
    if param_name == "recomb_beta":
        if recomb_model != "modified_box":
            raise ValueError("recomb_beta GT only for modified_box")
        return float(rp.beta)
    if param_name == "recomb_beta_90":
        if recomb_model != "emb":
            raise ValueError("recomb_beta_90 GT only for emb")
        return float(rp.beta_90)
    if param_name == "recomb_R":
        if recomb_model != "emb":
            raise ValueError("recomb_R GT only for emb")
        return float(rp.R)
    raise ValueError(f"Unknown param {param_name!r}")


def _load_ground_truth_tables(
    config_path: str,
    *,
    verbose: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Build physical and normalized (log-scale) GT from detector YAML + GT lifetime/velocity.

    Normalized trajectories in W&B are ``log(physical / TYPICAL_SCALES[name])``.
    Returns empty dicts if loading fails.
    """
    try:
        import jax.numpy as jnp

        from tools.config import create_sim_params
        from tools.geometry import generate_detector
    except Exception as exc:
        print(
            f"Warning: GT lines disabled (import error: {exc}). "
            f"Install jax and ensure repo root is on PYTHONPATH.",
            flush=True,
        )
        return {}, {}

    try:
        detector = generate_detector(config_path)
    except Exception as exc:
        print(f"Warning: could not load GT config {config_path!r} ({exc}); no GT lines.", flush=True)
        return {}, {}

    recomb_model = (
        detector.get("simulation", {})
        .get("charge_recombination", {})
        .get("model", "modified_box")
    )
    try:
        gt_params = create_sim_params(detector, recombination_model=recomb_model)
        gt_params = gt_params._replace(
            lifetime_us=jnp.array(GT_LIFETIME_US),
            velocity_cm_us=jnp.array(GT_VELOCITY_CM_US),
        )
    except Exception as exc:
        print(f"Warning: could not build SimParams for GT ({exc}); no GT lines.", flush=True)
        return {}, {}

    physical: Dict[str, float] = {}
    normalized: Dict[str, float] = {}
    for name in _GT_PARAM_ORDER:
        try:
            v = _float_gt_physical(name, gt_params, recomb_model)
        except ValueError:
            continue
        physical[name] = v
        scale = TYPICAL_SCALES.get(name)
        if scale is not None and v > 0.0 and scale > 0.0:
            normalized[name] = math.log(v / scale)

    if verbose:
        print(f"GT reference: config={config_path}  recomb_model={recomb_model!r}", flush=True)
        for name in _GT_PARAM_ORDER:
            if name not in physical:
                continue
            pn = normalized.get(name)
            pn_s = f"{pn:.6g}" if pn is not None and math.isfinite(pn) else "n/a"
            print(
                f"  {name}: physical={physical[name]:.6g}  log(p/scale)={pn_s}",
                flush=True,
            )

    return physical, normalized


def _resolve_entity_project(
    api,
    entity: Optional[str],
    project: Optional[str],
) -> Tuple[str, str]:
    final_entity = entity or os.environ.get("WANDB_ENTITY") or api.default_entity
    final_project = project or os.environ.get("WANDB_PROJECT") or "jaxtpc-optimization"
    if not final_entity:
        raise ValueError(
            "Could not resolve W&B entity. Set --entity or WANDB_ENTITY."
        )
    return final_entity, final_project


def _configure_log_axis(ax, all_values: np.ndarray, ylabel_base: str) -> None:
    if all_values.size == 0:
        ax.set_ylabel(f"{ylabel_base} (log)")
        ax.set_yscale("log")
        return
    finite_vals = all_values[np.isfinite(all_values)]
    if finite_vals.size == 0:
        ax.set_ylabel(f"{ylabel_base} (log)")
        ax.set_yscale("log")
        return

    if np.all(finite_vals > 0.0):
        ax.set_yscale("log")
        ax.set_ylabel(f"{ylabel_base} (log)")
        return

    abs_vals = np.abs(finite_vals)
    nonzero = abs_vals[abs_vals > 0.0]
    if nonzero.size == 0:
        linthresh = 1e-6
    else:
        linthresh = max(float(np.quantile(nonzero, 0.05)), 1e-8)
        if not math.isfinite(linthresh):
            linthresh = 1e-6
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_ylabel(f"{ylabel_base} (symlog)")


def _plot_kind(
    kind: str,
    series_by_param: Dict[str, List[Series]],
    color_by_tag: Dict[str, str],
    out_root: str,
    entity: str,
    project: str,
    *,
    tag_order: List[str],
    verbose: bool = False,
    gt_by_param: Optional[Dict[str, float]] = None,
    label_by_tag: Optional[Dict[str, str]] = None,
    dpi: int = DEFAULT_PNG_DPI,
) -> int:
    out_dir = os.path.join(out_root, _KIND_TO_DIR[kind])
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for param in sorted(series_by_param):
        runs = series_by_param[param]
        if not runs:
            continue

        fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
        all_vals = []
        present_tags = []
        for run_series in runs:
            color = color_by_tag[run_series.tag]
            ax_lin.plot(
                run_series.steps,
                run_series.values,
                color=color,
                lw=1.4,
                alpha=0.45,
            )
            ax_log.plot(
                run_series.steps,
                run_series.values,
                color=color,
                lw=1.4,
                alpha=0.45,
            )
            all_vals.append(run_series.values)
            present_tags.append(run_series.tag)

        tags_for_legend = _legend_order_tags(present_tags, tag_order)
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=color_by_tag[tag],
                lw=2.2,
                label=(
                    label_by_tag[tag]
                    if label_by_tag is not None and tag in label_by_tag
                    else tag
                ),
            )
            for tag in tags_for_legend
        ]

        gt_y: Optional[float] = None
        if kind in ("params_physical", "params_normalized") and gt_by_param:
            gt_y = gt_by_param.get(param)
        if gt_y is not None and math.isfinite(gt_y):
            ax_lin.axhline(
                y=gt_y,
                color="0.2",
                linestyle="--",
                linewidth=1.5,
                alpha=0.95,
                zorder=8,
            )
            ax_log.axhline(
                y=gt_y,
                color="0.2",
                linestyle="--",
                linewidth=1.5,
                alpha=0.95,
                zorder=8,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="0.2",
                    linestyle="--",
                    linewidth=1.5,
                    label="GT",
                )
            )

        ax_lin.set_title(f"{param} - linear")
        ax_lin.set_xlabel("step")
        ax_lin.set_ylabel(param)
        ax_lin.grid(True, alpha=0.25)
        ax_lin.legend(handles=legend_handles, fontsize=8)

        ax_log.set_title(f"{param} - log")
        ax_log.set_xlabel("step")
        ax_log.grid(True, which="both", alpha=0.25)
        log_vals = np.concatenate(all_vals)
        if gt_y is not None and math.isfinite(gt_y):
            log_vals = np.concatenate([log_vals, np.array([gt_y], dtype=float)])
        _configure_log_axis(ax_log, log_vals, param)

        fig.suptitle(
            f"W&B {_KIND_TO_TITLE[kind]} | {entity}/{project} | {param}",
            fontsize=11,
            y=1.02,
        )
        fig.tight_layout()

        out_name = f"{_sanitize_for_path(param)}.png"
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        saved += 1
        if verbose:
            print(f"    wrote {out_path}", flush=True)

    return saved


def _build_relative_error_series(
    physical_by_param: Dict[str, List[Series]],
    gt_physical: Dict[str, float],
) -> Dict[str, List[Series]]:
    """Build absolute relative-error trajectories from physical parameter traces.

    For each point: ``abs((value - gt) / gt)``.
    Parameters missing GT or with GT==0 are skipped.
    """
    rel_by_param: Dict[str, List[Series]] = defaultdict(list)
    for param, runs in physical_by_param.items():
        gt = gt_physical.get(param)
        if gt is None or (not math.isfinite(gt)) or gt == 0.0:
            continue
        for run_series in runs:
            vals = np.asarray(run_series.values, dtype=float)
            rel_vals = np.abs((vals - gt) / gt)
            rel_by_param[param].append(
                Series(
                    tag=run_series.tag,
                    run_name=run_series.run_name,
                    run_id=run_series.run_id,
                    steps=run_series.steps,
                    values=rel_vals,
                )
            )
    return rel_by_param


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tags",
        required=True,
        help="Comma-separated list of W&B tags to include.",
    )
    parser.add_argument(
        "--colors",
        default=None,
        help="Optional comma-separated colors, one per tag (same order as --tags).",
    )
    parser.add_argument(
        "--tag-labels",
        default=None,
        metavar="LABELS",
        help=(
            "Optional comma-separated legend labels in the same order as --tags. "
            "Labels containing commas require repeated --tag-label instead."
        ),
    )
    parser.add_argument(
        "--tag-label",
        action="append",
        default=None,
        dest="tag_label_each",
        metavar="LABEL",
        help=(
            "Legend label for one --tags entry (repeat once per tag, same order). "
            "Mutually exclusive with --tag-labels."
        ),
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity (default: WANDB_ENTITY or logged-in default entity).",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="W&B project (default: WANDB_PROJECT or jaxtpc-optimization).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root directory (default: $PLOTS_DIR/wandb_run_results/<tags>).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional cap on number of fetched runs (0 = no cap).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Per-tag run counts, per-parameter output paths, and metric kinds summary.",
    )
    _repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    _default_gt_cfg = os.path.join(_repo_root, "config", "cubic_wireplane_config.yaml")
    parser.add_argument(
        "--gt-config",
        default=None,
        metavar="YAML",
        help=(
            "Detector YAML for GT horizontal lines on physical/normalized panels "
            f"(default: {_default_gt_cfg}). "
            "Uses the same GT lifetime/velocity overrides as run_optimization.py."
        ),
    )
    parser.add_argument(
        "--no-gt-lines",
        action="store_true",
        help="Do not draw GT reference lines on physical/normalized parameter plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_PNG_DPI,
        metavar="N",
        help=f"PNG resolution (default: {DEFAULT_PNG_DPI}).",
    )
    return parser.parse_args()


def main() -> None:
    global plt, Line2D
    args = parse_args()

    try:
        import matplotlib.pyplot as _plt
        from matplotlib.lines import Line2D as _Line2D
    except ImportError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for this script. Install it with `pip install matplotlib`."
        ) from exc
    plt = _plt
    Line2D = _Line2D

    try:
        import wandb
    except ImportError as exc:
        raise ModuleNotFoundError(
            "wandb is required for this script. Install it with `pip install wandb`."
        ) from exc

    tags = _split_csv(args.tags)
    if not tags:
        raise ValueError("--tags must not be empty")

    str_labels = args.tag_labels is not None and str(args.tag_labels).strip()
    each_labels = args.tag_label_each
    if str_labels and each_labels:
        raise ValueError("Use either --tag-labels or repeated --tag-label, not both.")
    label_by_tag: Optional[Dict[str, str]] = None
    if each_labels:
        if len(each_labels) != len(tags):
            raise ValueError(
                f"Provide exactly one --tag-label per --tags entry ({len(tags)} tags), "
                f"got {len(each_labels)} label(s)."
            )
        label_by_tag = dict(zip(tags, each_labels))
    elif str_labels:
        lbls = [s.strip() for s in str(args.tag_labels).split(",")]
        if len(lbls) != len(tags):
            raise ValueError(
                f"--tag-labels must list exactly {len(tags)} name(s) (same count as --tags), "
                f"got {len(lbls)}."
            )
        label_by_tag = dict(zip(tags, lbls))

    colors = _split_csv(args.colors) if args.colors else None
    color_by_tag = _build_color_map(tags, colors)

    output_dir = args.output_dir or _build_default_output_dir(tags)
    os.makedirs(output_dir, exist_ok=True)

    api = wandb.Api()
    entity, project = _resolve_entity_project(api, args.entity, args.project)
    path = f"{entity}/{project}"

    print(f"W&B path      : {path}")
    print(f"Selected tags : {tags}")
    print(f"Output root   : {output_dir}")
    if label_by_tag:
        print("Legend labels :")
        for t in tags:
            print(f"  {t!r} -> {label_by_tag[t]!r}")
    if args.verbose:
        print(f"Max runs cap  : {args.max_runs if args.max_runs > 0 else 'none'}", flush=True)

    runs = api.runs(path=path, filters={"tags": {"$in": tags}})
    if args.max_runs > 0:
        runs = runs[: args.max_runs]

    # kind -> param -> list[Series]
    aggregated: Dict[str, Dict[str, List[Series]]] = defaultdict(
        lambda: defaultdict(list)
    )

    n_loaded_runs = 0
    n_used_runs = 0
    runs_assigned_per_tag: Counter[str] = Counter()
    runs_with_series_per_tag: Counter[str] = Counter()
    skipped_no_tag = 0
    for run in runs:
        n_loaded_runs += 1
        run_tags = list(getattr(run, "tags", []) or [])
        assigned_tag = _pick_primary_tag(run_tags, tags)
        if assigned_tag is None:
            skipped_no_tag += 1
            if args.verbose:
                print(
                    f"  skip run {run.id}: no tag order match among {run_tags!r}",
                    flush=True,
                )
            continue

        runs_assigned_per_tag[assigned_tag] += 1

        try:
            per_kind = _extract_run_series(run, assigned_tag)
        except Exception as exc:
            print(f"Warning: failed to parse run {run.id}: {exc}")
            continue

        has_data = False
        for kind, per_param in per_kind.items():
            for param, series in per_param.items():
                aggregated[kind][param].append(series)
                has_data = True

        if has_data:
            n_used_runs += 1
            runs_with_series_per_tag[assigned_tag] += 1

    print(f"Fetched runs  : {n_loaded_runs}")
    print(f"Usable runs   : {n_used_runs}")
    if args.verbose:
        print(f"Skipped (no primary tag in order): {skipped_no_tag}", flush=True)
        print(f"Runs assigned per tag (have primary tag): {dict(runs_assigned_per_tag)}", flush=True)
        print(
            f"Runs with trajectory data per tag: {dict(runs_with_series_per_tag)}",
            flush=True,
        )
        for kind in ("grads", "params_physical", "params_normalized", "params_relative_error"):
            params = sorted(aggregated.get(kind, {}).keys())
            print(
                f"Kind {_KIND_TO_DIR[kind]}: {len(params)} parameter(s)"
                + (f" — {params}" if params else ""),
                flush=True,
            )

    if n_used_runs == 0:
        print("No matching metric trajectories found for selected tags.")
        return

    _repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    _default_gt_cfg = os.path.join(_repo_root, "config", "cubic_wireplane_config.yaml")
    gt_config_path = args.gt_config or _default_gt_cfg
    gt_physical: Dict[str, float] = {}
    gt_normalized: Dict[str, float] = {}
    if not args.no_gt_lines:
        gt_physical, gt_normalized = _load_ground_truth_tables(
            gt_config_path,
            verbose=args.verbose,
        )
        if gt_physical:
            aggregated["params_relative_error"] = _build_relative_error_series(
                aggregated.get("params_physical", {}),
                gt_physical,
            )

    total_saved = 0
    for kind in ("grads", "params_physical", "params_normalized", "params_relative_error"):
        if args.verbose:
            print(f"Plotting {_KIND_TO_DIR[kind]} …", flush=True)
        gt_map: Optional[Dict[str, float]] = None
        if kind == "params_physical":
            gt_map = gt_physical
        elif kind == "params_normalized":
            gt_map = gt_normalized
        saved = _plot_kind(
            kind=kind,
            series_by_param=aggregated.get(kind, {}),
            color_by_tag=color_by_tag,
            out_root=output_dir,
            entity=entity,
            project=project,
            tag_order=tags,
            verbose=args.verbose,
            gt_by_param=gt_map,
            label_by_tag=label_by_tag,
            dpi=args.dpi,
        )
        total_saved += saved
        print(f"Saved {saved} figure(s) to: {os.path.join(output_dir, _KIND_TO_DIR[kind])}")

    print(f"Done. Total figures saved: {total_saved}")


if __name__ == "__main__":
    main()
