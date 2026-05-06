#!/usr/bin/env python3
"""Pull timing-study W&B runs and plot deposits vs timing, JAX memory, and avg step time.

Designed for jobs submitted via ``timing_study_diag50mev`` / ``timing_study_cont``
profiles (tags ``dep_<n>`` + study tag). Requires ``wandb`` login / ``WANDB_API_KEY``.

Examples::

    .venv/bin/python scripts/plot_timing_study_wandb.py \\
        --entity fcc_ml --project jaxtpc-optimization \\
        --output plots/timing_study.png

    .venv/bin/python scripts/plot_timing_study_wandb.py \\
        --study-tags timing_study_diag50mev,timing_study_cont \\
        --study-tag-labels "Diag 50 MeV,Continuous scan" \\
        --output plots/timing_study.png

Reference **finished** run ``x29don9r`` (``dep_5000``): summary includes ``_runtime``,
``sys/gpu/jax_peak_gb``, ``sys/gpu/jax_mem_gb``, sparse ``step_time_s`` when
``--log-interval 1000``.

Reference **failed** run ``awgtutt4`` (``dep_80000``): ``historyLineCount`` is 0,
summary usually only ``_runtime`` (~seconds until JAX ``RESOURCE_EXHAUSTED`` during
compile); no JAX GPU metrics — panels 2–3 show downward triangles; panel 4 marks
missing ``step_time_s`` the same way.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from typing import Any, Iterable

DEFAULT_PNG_DPI = 300

import matplotlib.pyplot as plt
import numpy as np


def _cfg_val(run: Any, key: str) -> Any:
    raw = run.config.get(key)
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


def _parse_dep_tag(tags: Iterable[str]) -> int | None:
    for t in tags:
        m = re.match(r"^dep_(\d+)$", str(t).strip())
        if m:
            return int(m.group(1))
    return None


def _gather_history_vectors(run: Any, keys: list[str]) -> dict[str, np.ndarray]:
    """Collect metric columns via scan_history (no pandas dependency)."""
    cols = {k: [] for k in keys}
    steps = []
    try:
        for row in run.scan_history(keys=keys):
            st = row.get("_step")
            steps.append(float(st) if st is not None else math.nan)
            for k in keys:
                v = row.get(k)
                if v is None:
                    cols[k].append(math.nan)
                else:
                    fv = float(v)
                    cols[k].append(fv if math.isfinite(fv) else math.nan)
    except Exception:
        pass
    out = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    out["_step"] = np.asarray(steps, dtype=float)
    return out


def _nanmean(a: np.ndarray) -> float:
    v = a[~np.isnan(a)]
    return float(np.mean(v)) if v.size else math.nan


def _nanmax(a: np.ndarray) -> float:
    v = a[~np.isnan(a)]
    return float(np.max(v)) if v.size else math.nan


def _csv_float(v: Any) -> str:
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return ""
    return f"{fv:.6g}" if math.isfinite(fv) else ""


def collect_runs(
    api: Any,
    entity: str,
    project: str,
    study_tags: list[str],
    *,
    limit: int,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Fetch runs whose tags intersect ``study_tags``; newest run wins per deposit."""
    path = f"{entity}/{project}"
    tagged: dict[int, dict[str, Any]] = {}

    seen_ids: set[str] = set()
    per_tag_counts: list[tuple[str, int]] = []
    for tag in study_tags:
        try:
            iterator = api.runs(path, filters={"tags": {"$in": [tag]}}, order="-created_at")
        except TypeError:
            iterator = api.runs(path, filters={"tags": {"$in": [tag]}})
        n_tag = 0
        for run in iterator:
            if limit > 0 and n_tag >= limit:
                break
            n_tag += 1
            rid = getattr(run, "id", None) or run.name
            if rid in seen_ids:
                continue
            ts = set(run.tags or [])
            if not ts.intersection(study_tags):
                continue
            seen_ids.add(str(rid))

            dep = _cfg_val(run, "max_num_deposits")
            if dep is None:
                dep = _parse_dep_tag(ts)
            if dep is None:
                continue
            dep = int(dep)

            max_steps = _cfg_val(run, "max_steps") or 1000
            try:
                max_steps = int(max_steps)
            except (TypeError, ValueError):
                max_steps = 1000

            summary = dict(run.summary or {})
            hist = _gather_history_vectors(
                run,
                ["step_time_s", "sys/gpu/jax_mem_gb", "sys/gpu/jax_peak_gb"],
            )

            step_times = hist["step_time_s"]
            jax_mem = hist["sys/gpu/jax_mem_gb"]
            jax_peak_h = hist["sys/gpu/jax_peak_gb"]

            mean_step_s = _nanmean(step_times)
            est_loop_s = (
                mean_step_s * max_steps if math.isfinite(mean_step_s) else math.nan
            )

            wall_s = summary.get("_runtime")
            if wall_s is None and isinstance(summary.get("_wandb"), dict):
                wall_s = summary["_wandb"].get("runtime")

            jax_peak_summary = summary.get("sys/gpu/jax_peak_gb")
            jax_peak = math.nan
            if jax_peak_summary is not None:
                try:
                    jax_peak = float(jax_peak_summary)
                except (TypeError, ValueError):
                    jax_peak = math.nan
            if not math.isfinite(jax_peak):
                jax_peak = _nanmax(jax_peak_h)

            jax_mem_avg = _nanmean(jax_mem)
            if not math.isfinite(jax_mem_avg) and summary.get("sys/gpu/jax_mem_gb") is not None:
                jax_mem_avg = float(summary["sys/gpu/jax_mem_gb"])

            jax_peak_store = jax_peak if math.isfinite(jax_peak) else math.nan
            jax_mem_store = jax_mem_avg if math.isfinite(jax_mem_avg) else math.nan
            jax_metrics_logged = math.isfinite(jax_peak_store) or math.isfinite(jax_mem_store)

            row = {
                "id": str(rid),
                "name": getattr(run, "name", rid),
                "state": getattr(run, "state", "unknown"),
                "deposits": dep,
                "max_steps": max_steps,
                "mean_step_time_s": mean_step_s,
                "est_optimizer_wall_s": est_loop_s,
                "job_wall_runtime_s": float(wall_s) if wall_s is not None else math.nan,
                "jax_peak_gb": jax_peak_store,
                "jax_mem_avg_gb": jax_mem_store,
                "jax_metrics_logged": jax_metrics_logged,
                "created_at": getattr(run, "created_at", None),
                "tags": sorted(ts),
            }

            prev = tagged.get(dep)
            if prev is None:
                tagged[dep] = row
                continue
            # Prefer newest completed run for duplicate deposits.
            ca_new = row["created_at"]
            ca_old = prev["created_at"]
            new_finished = row["state"] == "finished"
            old_finished = prev["state"] == "finished"
            replace = False
            if new_finished and not old_finished:
                replace = True
            elif new_finished == old_finished and ca_new and ca_old:
                replace = ca_new > ca_old
            elif ca_new and ca_old:
                replace = ca_new > ca_old
            if replace:
                tagged[dep] = row

        per_tag_counts.append((tag, n_tag))

    if verbose:
        print(f"W&B path: {path}", flush=True)
        print(f"Study tags ({len(study_tags)}): {study_tags}", flush=True)
        print(f"Fetch limit per tag: {limit}", flush=True)
        for t, n in per_tag_counts:
            print(f"  runs seen for tag {t!r}: {n}", flush=True)

    return sorted(tagged.values(), key=lambda r: r["deposits"])


def plot_rows(
    rows: list[dict[str, Any]],
    title: str,
    output_path: str,
    *,
    study_tag_labels: list[str] | None = None,
    study_tags: list[str] | None = None,
    verbose: bool = False,
    dpi: int = DEFAULT_PNG_DPI,
) -> None:
    if not rows:
        raise SystemExit("No matching runs — check entity/project/tags and wandb login.")

    if verbose:
        states = Counter(str(r["state"]) for r in rows)
        with_jax = sum(1 for r in rows if r.get("jax_metrics_logged"))
        print(
            f"Plotting {len(rows)} rows (one per deposit): "
            f"states {dict(states)}, with JAX metrics: {with_jax}",
            flush=True,
        )

    deps = np.array([r["deposits"] for r in rows], dtype=float)
    finished = np.array([r["state"] == "finished" for r in rows])

    est_loop = np.array([r["est_optimizer_wall_s"] for r in rows], dtype=float)
    wall = np.array([r["job_wall_runtime_s"] for r in rows], dtype=float)
    peak = np.array([r["jax_peak_gb"] for r in rows], dtype=float)
    mem_avg = np.array([r["jax_mem_avg_gb"] for r in rows], dtype=float)
    step_mean = np.array([r["mean_step_time_s"] for r in rows], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(9, 13), sharex=True)

    def _plot_xy(ax, y: np.ndarray, ylabel: str, *, show_legend: bool = True):
        mask_ok = finished & np.isfinite(y)
        mask_fail = (~finished) & np.isfinite(y)
        if np.any(mask_ok):
            ax.plot(deps[mask_ok], y[mask_ok], "o-", lw=2, ms=8, label="finished")
        if np.any(mask_fail):
            ax.scatter(
                deps[mask_fail],
                y[mask_fail],
                s=120,
                marker="x",
                c="crimson",
                zorder=5,
                label="not finished (metric present)",
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if show_legend:
            ax.legend(loc="best", fontsize=8)

    if study_tag_labels and study_tags and len(study_tag_labels) == len(study_tags):
        label_line = " · ".join(
            f"{lab} ({tag})" for lab, tag in zip(study_tag_labels, study_tags, strict=True)
        )
        axes[0].set_title(f"{title}\n{label_line}", fontsize=10)
    else:
        axes[0].set_title(title)
    _plot_xy(
        axes[0],
        est_loop,
        "Est. optimizer time (s)\nmean(step_time_s)×max_steps",
        show_legend=False,
    )
    # Job wall-clock: finished (gray) and failed e.g. compile-OOM with only _runtime (orange).
    ax0t = axes[0].twinx()
    mask_w_fin = finished & np.isfinite(wall)
    mask_w_fail = (~finished) & np.isfinite(wall)
    if np.any(mask_w_fin):
        ax0t.scatter(
            deps[mask_w_fin],
            wall[mask_w_fin],
            c="gray",
            marker="s",
            s=36,
            alpha=0.7,
            label="finished: job wall _runtime",
        )
    if np.any(mask_w_fail):
        ax0t.scatter(
            deps[mask_w_fail],
            wall[mask_w_fail],
            c="darkorange",
            marker="s",
            s=52,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.5,
            label="failed: job wall _runtime (e.g. awgtutt4)",
            zorder=6,
        )
    if np.any(mask_w_fin | mask_w_fail):
        ax0t.set_ylabel("Job wall runtime _runtime (s)", color="gray")
        ax0t.tick_params(axis="y", labelcolor="gray")
        ax0t.legend(loc="upper left", fontsize=7)

    h0, l0 = axes[0].get_legend_handles_labels()
    if h0:
        axes[0].legend(h0, l0, loc="upper right", fontsize=7)

    # Reference ymax for placing compile-OOM markers (failed before any JAX metric).
    peak_fin = peak[finished & np.isfinite(peak)]
    peak_ref = _nanmax(peak_fin)
    if not math.isfinite(peak_ref) or peak_ref <= 0:
        if verbose:
            print(
                "  peak_ref: no finished finite jax_peak_gb — using fallback 1.0 GB",
                flush=True,
            )
        peak_ref = 1.0
    elif verbose:
        print(f"  peak_ref from finished runs: {peak_ref:.6g} GB", flush=True)

    mem_fin = mem_avg[finished & np.isfinite(mem_avg)]
    mem_ref = _nanmax(mem_fin)
    if not math.isfinite(mem_ref) or mem_ref <= 0:
        if verbose:
            print(
                "  mem_ref: no finished finite jax_mem_avg_gb — using fallback 0.5 GB",
                flush=True,
            )
        mem_ref = 0.5
    elif verbose:
        print(f"  mem_ref from finished runs: {mem_ref:.6g} GB", flush=True)

    fail_no_jax = (~finished) & ~np.isfinite(peak) & ~np.isfinite(mem_avg)

    _plot_xy(
        axes[1],
        peak,
        "Peak JAX GPU memory (GB)\nsummary sys/gpu/jax_peak_gb",
        show_legend=False,
    )
    if np.any(fail_no_jax):
        axes[1].scatter(
            deps[fail_no_jax],
            np.full(np.sum(fail_no_jax), peak_ref * 0.07),
            s=110,
            marker="v",
            c="crimson",
            zorder=6,
            label="failed, no JAX metrics (compile OOM)",
        )
    h1, l1 = axes[1].get_legend_handles_labels()
    if h1:
        axes[1].legend(h1, l1, loc="best", fontsize=8)

    _plot_xy(
        axes[2],
        mem_avg,
        "Avg JAX in-use memory (GB)\nmean logged sys/gpu/jax_mem_gb",
        show_legend=False,
    )
    if np.any(fail_no_jax):
        axes[2].scatter(
            deps[fail_no_jax],
            np.full(np.sum(fail_no_jax), mem_ref * 0.07),
            s=110,
            marker="v",
            c="crimson",
            zorder=6,
            label="failed, no JAX metrics (compile OOM)",
        )
    h2, l2 = axes[2].get_legend_handles_labels()
    if h2:
        axes[2].legend(h2, l2, loc="best", fontsize=8)

    step_fin = step_mean[finished & np.isfinite(step_mean)]
    step_ref = _nanmax(step_fin)
    if not math.isfinite(step_ref) or step_ref <= 0:
        if verbose:
            print(
                "  step_ref: no finished finite mean_step_time — using fallback 0.2 s",
                flush=True,
            )
        step_ref = 0.2
    elif verbose:
        print(f"  step_ref from finished runs: {step_ref:.6g} s", flush=True)

    no_step = ~np.isfinite(step_mean)
    _plot_xy(
        axes[3],
        step_mean,
        "Avg step time (s)\nmean logged step_time_s",
        show_legend=False,
    )
    if np.any(no_step):
        axes[3].scatter(
            deps[no_step],
            np.full(np.sum(no_step), step_ref * 0.07),
            s=110,
            marker="v",
            c="crimson",
            zorder=6,
            label="no logged step_time_s (compile fail / no train loop)",
        )
    h3, l3 = axes[3].get_legend_handles_labels()
    if h3:
        axes[3].legend(h3, l3, loc="best", fontsize=8)

    axes[-1].set_xlabel("max_num_deposits (pad)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Saved plot: {output_path}")

    csv_path = os.path.splitext(output_path)[0] + "_runs.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        cols = [
            "deposits",
            "state",
            "run_id",
            "jax_metrics_logged",
            "mean_step_time_s",
            "est_optimizer_wall_s",
            "job_wall_runtime_s",
            "jax_peak_gb",
            "jax_mem_avg_gb",
            "tags",
        ]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(
                ",".join(
                    [
                        str(r["deposits"]),
                        str(r["state"]),
                        str(r["id"]),
                        "1" if r.get("jax_metrics_logged") else "0",
                        _csv_float(r["mean_step_time_s"]),
                        _csv_float(r["est_optimizer_wall_s"]),
                        _csv_float(r["job_wall_runtime_s"]),
                        _csv_float(r["jax_peak_gb"]),
                        _csv_float(r["jax_mem_avg_gb"]),
                        json.dumps(r["tags"]),
                    ]
                )
                + "\n"
            )
    print(f"Saved table: {csv_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "fcc_ml"))
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "jaxtpc-optimization"))
    p.add_argument(
        "--study-tags",
        default="timing_study_diag50mev,timing_study_cont",
        help="Comma-separated run tags (union); runs matching any tag are considered.",
    )
    p.add_argument(
        "--study-tag-labels",
        default=None,
        metavar="LABELS",
        help=(
            "Optional comma-separated human-readable names in the same order as "
            "--study-tags (shown under the plot title with each tag). "
            "Must provide exactly one label per tag. "
            "Cannot use together with repeated --study-tag-label."
        ),
    )
    p.add_argument(
        "--study-tag-label",
        action="append",
        default=None,
        dest="study_tag_label_each",
        metavar="LABEL",
        help=(
            "Human-readable name for the next --study-tags slot (repeat once per tag). "
            "Use this when a label contains commas. Mutually exclusive with --study-tag-labels."
        ),
    )
    p.add_argument(
        "--output",
        default=os.path.join(os.environ.get("PLOTS_DIR", "plots"), "timing_study_wandb.png"),
        help="PNG output path (CSV stem matched).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Approx max runs fetched per study tag (safety bound).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print W&B fetch counts, row summary, and reference scale choices.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_PNG_DPI,
        metavar="N",
        help=f"PNG resolution (default: {DEFAULT_PNG_DPI}).",
    )
    args = p.parse_args()

    study_tags = [t.strip() for t in args.study_tags.split(",") if t.strip()]
    study_tag_labels: list[str] | None = None
    str_labels = args.study_tag_labels is not None and str(args.study_tag_labels).strip()
    each_labels = args.study_tag_label_each
    if str_labels and each_labels:
        raise SystemExit("Use either --study-tag-labels or repeated --study-tag-label, not both.")
    if each_labels:
        study_tag_labels = list(each_labels)
        if len(study_tag_labels) != len(study_tags):
            raise SystemExit(
                f"Provide exactly one --study-tag-label per --study-tags entry "
                f"({len(study_tags)} tags), got {len(study_tag_labels)} label(s)."
            )
    elif str_labels:
        study_tag_labels = [s.strip() for s in str(args.study_tag_labels).split(",")]
        if len(study_tag_labels) != len(study_tags):
            raise SystemExit(
                f"--study-tag-labels must list exactly {len(study_tags)} name(s) "
                f"(same count as --study-tags), got {len(study_tag_labels)}."
            )

    import wandb

    if args.verbose:
        print(
            f"Query: entity={args.entity!r} project={args.project!r} "
            f"output={args.output!r}",
            flush=True,
        )

    api = wandb.Api(timeout=120)
    rows = collect_runs(
        api,
        args.entity,
        args.project,
        study_tags,
        limit=args.limit,
        verbose=args.verbose,
    )
    if args.verbose:
        deps_sorted = [r["deposits"] for r in rows]
        if deps_sorted:
            print(
                f"Unique deposits covered: {len(deps_sorted)} "
                f"(min={min(deps_sorted)}, max={max(deps_sorted)})",
                flush=True,
            )

    if study_tag_labels:
        title = f"{args.entity}/{args.project}  timing study"
    else:
        title = f"{args.entity}/{args.project}  timing study ({', '.join(study_tags)})"
    plot_rows(
        rows,
        title,
        args.output,
        study_tag_labels=study_tag_labels,
        study_tags=study_tags if study_tag_labels else None,
        verbose=args.verbose,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
