#!/usr/bin/env python
"""
Plot W&B optimization trajectories grouped by run tags.

For each requested tag, runs that contain that tag are fetched from a W&B project.
Each run is assigned to the first matching tag (in the provided tag order), and all
runs assigned to the same tag use the same color.

For each parameter, this script saves paired plots:
  - left: linear scale
  - right: log scale (or symlog when values include non-positive values)

Plot families (saved in separate directories):
  - gradients:         grads/<param>
  - physical_params:   params/<param>_physical
  - normalized_params: params/<param>_normalized

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
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

_KIND_TO_DIR = {
    "grads": "gradients",
    "params_physical": "physical_params",
    "params_normalized": "normalized_params",
}

_KIND_TO_TITLE = {
    "grads": "gradient trajectories",
    "params_physical": "physical parameter trajectories",
    "params_normalized": "normalized parameter trajectories",
}


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

        unique_tags = sorted(set(present_tags))
        legend_handles = [
            Line2D([0], [0], color=color_by_tag[tag], lw=2.2, label=tag)
            for tag in unique_tags
        ]

        ax_lin.set_title(f"{param} - linear")
        ax_lin.set_xlabel("step")
        ax_lin.set_ylabel(param)
        ax_lin.grid(True, alpha=0.25)
        ax_lin.legend(handles=legend_handles, fontsize=8)

        ax_log.set_title(f"{param} - log")
        ax_log.set_xlabel("step")
        ax_log.grid(True, which="both", alpha=0.25)
        _configure_log_axis(ax_log, np.concatenate(all_vals), param)

        fig.suptitle(
            f"W&B {_KIND_TO_TITLE[kind]} | {entity}/{project} | {param}",
            fontsize=11,
            y=1.02,
        )
        fig.tight_layout()

        out_name = f"{_sanitize_for_path(param)}.png"
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    return saved


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

    runs = api.runs(path=path, filters={"tags": {"$in": tags}})
    if args.max_runs > 0:
        runs = runs[: args.max_runs]

    # kind -> param -> list[Series]
    aggregated: Dict[str, Dict[str, List[Series]]] = defaultdict(
        lambda: defaultdict(list)
    )

    n_loaded_runs = 0
    n_used_runs = 0
    for run in runs:
        n_loaded_runs += 1
        run_tags = list(getattr(run, "tags", []) or [])
        assigned_tag = _pick_primary_tag(run_tags, tags)
        if assigned_tag is None:
            continue

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

    print(f"Fetched runs  : {n_loaded_runs}")
    print(f"Usable runs   : {n_used_runs}")

    if n_used_runs == 0:
        print("No matching metric trajectories found for selected tags.")
        return

    total_saved = 0
    for kind in ("grads", "params_physical", "params_normalized"):
        saved = _plot_kind(
            kind=kind,
            series_by_param=aggregated.get(kind, {}),
            color_by_tag=color_by_tag,
            out_root=output_dir,
            entity=entity,
            project=project,
        )
        total_saved += saved
        print(f"Saved {saved} figure(s) to: {os.path.join(output_dir, _KIND_TO_DIR[kind])}")

    print(f"Done. Total figures saved: {total_saved}")


if __name__ == "__main__":
    main()
