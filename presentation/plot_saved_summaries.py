#!/usr/bin/env python3
"""Plot saved RAIL-3 summaries; never evaluate predictions or estimate statistics.

Inputs are the six digest-bound local summary files in inputs/. This script
does not import rail3, load models, access targets, select cases, bootstrap,
compute metrics, classify hypotheses or read the historical research archive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_ORDER = ("G-small", "G-match", "Atom", "Rect", "Union", "Relative")
COLORS = ("#4477AA", "#CC8844", "#218B88", "#B66580", "#919342", "#78549A")
MARKERS = ("o", "s", "^", "v", "P", "D")
INPUTS = (
    "fidelity_contrasts_v2.json", "six_model_three_objectives.csv",
    "sparse_utility_coco.csv", "sparse_utility_voc_r3.csv",
    "coco_r3_interval.csv", "fit_sensitivity_v2.json",
)


def checked_inputs():
    expected = json.loads((HERE / "input_identities.json").read_text())
    values = {}
    for name in INPUTS:
        path = HERE / "inputs" / name
        raw = path.read_bytes()
        observed = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if observed != expected[name]:
            raise ValueError(f"Saved display input identity mismatch: {name}")
        if name.endswith(".json"):
            values[name] = json.loads(raw)
        else:
            values[name] = list(csv.DictReader(raw.decode().splitlines()))
    return values


def paired_interval(ax, point, low, high, y, *, coco=False, color="#22313C"):
    """Draw saved endpoints directly, without estimating or converting a CI."""
    ax.hlines(y, low, high, color=color, linewidth=1.2)
    ax.plot([low, high], [y, y], "|", color=color, markersize=7)
    ax.plot(point, y, "s" if coco else "o", color=color,
            markerfacecolor=color if coco else "white", markersize=4.5)


def fidelity(plt, data):
    source = data["fidelity_contrasts_v2.json"]
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.1), layout="constrained")
    for j, metric in enumerate(("absolute_residual_mae", "drre")):
        ax = axes[0, j]
        rows = [r for r in source["top_rows"] if r["metric"] == metric]
        rows.sort(key=lambda r: 0 if r["dataset"].startswith("VOC") else 1)
        for y, row in enumerate(rows):
            # Display-unit conversion only, applied to saved point and endpoints.
            point, low, high = (1000 * float(row[k]) for k in ("effect", "lower", "upper"))
            paired_interval(ax, point, low, high, y, coco=row["dataset"].startswith("COCO"))
        ax.set_yticks([0, 1], ["VOC FIT OOF", "COCO"])
        ax.set_ylim(1.5, -.5)
        ax.set_xlabel("Atom − G-match (×10⁻³)")
        ax.set_title(("(a) Absolute fidelity", "(b) STOP-relative fidelity")[j], loc="left")
        ax = axes[1, j]
        rows = [r for r in source["bottom_rows"] if r["metric"] == metric]
        rows.sort(key=lambda r: (r["right"] != "RECT_P", not r["dataset"].startswith("VOC")))
        ax.axvspan(-5, 5, color="#EEF1F3")
        ax.axvline(0, color="#87929A", linewidth=.7)
        for y, row in enumerate(rows):
            # These relative-display values were already saved; do not divide anew.
            point, low, high = (float(row["display_percent"][k]) for k in ("effect", "lower", "upper"))
            paired_interval(ax, point, low, high, y, coco=row["dataset"].startswith("COCO"))
        ax.set_yticks(range(4), ["VOC / Rect", "COCO / Rect", "VOC / Union", "COCO / Union"])
        ax.set_ylim(3.5, -.5)
        ax.set_xlim(-5.6, 5.6)
        ax.set_xticks([-5, 0, 5])
        ax.set_xlabel("Difference / comparator (%)")
        ax.set_title(("(c) Local MAE", "(d) Local DRRE")[j], loc="left")
    fig.suptitle("Paired 95% image-group intervals; frozen ±5% tie band", fontsize=10)
    return fig


def objectives(plt, data):
    rows = data["six_model_three_objectives.csv"]
    if [r["model"] for r in rows] != list(MODEL_ORDER):
        raise ValueError("Frozen six-model row order changed")
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.1), layout="constrained")
    columns = ("COCO_MAE", "COCO_DRRE", "COCO_normalized_regret")
    for ax, field, title in zip(axes, columns, ("Absolute MAE ↓", "STOP-relative DRRE ↓", "Normalized regret ↓")):
        for y, row in enumerate(rows):
            ax.scatter(float(row[field]), y, color=COLORS[y], marker=MARKERS[y], s=24)
        ax.set_yticks(range(6), MODEL_ORDER)
        ax.set_ylim(5.5, -.5)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color="#E2E7EB", linewidth=.5)
    axes[2].axvline(1, color="#87929A", linestyle="--", linewidth=.7)
    fig.suptitle("Saved printed precision; descriptive points, no model CIs", fontsize=10)
    return fig


def utility(plt, data):
    coco = data["sparse_utility_coco.csv"]
    voc = data["sparse_utility_voc_r3.csv"]
    interval = data["coco_r3_interval.csv"][0]
    if len(coco) != 18 or len(voc) != 6:
        raise ValueError("Frozen sparse display-cell inventory changed")
    fig = plt.figure(figsize=(6.3, 5.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(2, 1))
    ax = fig.add_subplot(grid[0, 0])
    for model, color, marker in zip(MODEL_ORDER, COLORS, MARKERS):
        rows = [r for r in coco if r["model"] == model]
        ax.plot([100 * float(r["budget"]) for r in rows],
                [100 * float(r["gain_capture"]) for r in rows],
                marker=marker, color=color, label=model, linewidth=1, markersize=4)
    ax.set_title("(a) COCO: all six models", loc="left")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.axhline(0, color="#87929A", linewidth=.7)
    ax.set_xticks([1, 2, 5])
    ax.set_xlabel("FORCED_K budget (%)")
    ax.set_ylabel("Signed gain capture (%)")
    ax = fig.add_subplot(grid[0, 1])
    for rows, label, color, marker in (
        ([r for r in voc if r["budget_semantics"] == "FORCED_K"], "VOC FIT OOF", "#52616D", "o"),
        ([r for r in coco if r["model"] == "Relative"], "COCO external", COLORS[-1], "D"),
    ):
        ax.plot([100 * float(r["budget"]) for r in rows],
                [100 * float(r["gain_capture"]) for r in rows],
                marker=marker, color=color, label=label, linewidth=1, markersize=4)
    ax.set_title("(b) Relative: two stages", loc="left")
    ax.axhline(0, color="#87929A", linewidth=.7)
    ax.legend(fontsize=7, frameon=False)
    ax.set_xticks([1, 2, 5])
    ax.set_xlabel("FORCED_K budget (%)")
    ax.set_ylabel("Signed gain capture (%)")
    ax = fig.add_subplot(grid[1, :])
    paired_interval(ax, float(interval["utility_improvement"]), float(interval["CI_low"]),
                    float(interval["CI_high"]), 0, coco=True, color=COLORS[-1])
    ax.axvline(0, color="#87929A", linewidth=.7)
    ax.set_yticks([])
    ax.set_xlabel("1 − normalized regret (predeclared COCO Relative)")
    ax.set_title("(c) Saved paired 95% interval; not raw gain or IoU", loc="left")
    fig.suptitle("Stage criteria and denominators differ; no domain-difference test", fontsize=10)
    return fig


def fit_sensitivity(plt, data):
    rows = data["fit_sensitivity_v2.json"]["original_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.9), layout="constrained")
    groups = (
        ("target_rich", "subset", (
            ("ALL_STATES", "All states"), ("REFERENCE_POSITIVE", "Reference positive"),
            ("A0_SOURCE_PRESENT", "A0 source present"),
            ("REFERENCE_POSITIVE_AND_A0_SOURCE_PRESENT", "Reference + A0 source")), "(a) Population composition"),
        ("utility_sparse", "utility", (
            ("full_image_pixel_mismatch", "Pixel mismatch"), ("one_minus_iou", "1 − IoU"),
            ("one_minus_dice", "1 − Dice"),
            ("foreground_normalized_symmetric_pixel_error", "Foreground error")), "(b) Alternative utilities"),
    )
    for ax, (source, field, labels, title) in zip(axes, groups):
        for j, (key, label) in enumerate(labels):
            selected = [r for r in rows[source] if r[field] == key]
            if any(r["semantics"] != "FORCED_K" for r in selected):
                raise ValueError("Unexpected sensitivity display semantics")
            ax.plot([100 * float(r["budget"]) for r in selected],
                    [float(r["gain capture"]) for r in selected],
                    color=COLORS[j], marker=MARKERS[j], label=label, linewidth=1, markersize=4)
        ax.axhline(0, color="#87929A", linewidth=.7)
        ax.set_xticks([1, 2, 5])
        ax.set_xlabel("FORCED_K budget (%)")
        ax.set_ylabel("Gain capture (fraction)")
        ax.set_title(title, loc="left")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("FIT-only, post-hoc sensitivity; separate RC-001 sources", fontsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true", help="Check local saved-input hashes without plotting")
    parser.add_argument("--output", type=Path, help="Required new output directory; existing paths are refused")
    args = parser.parse_args()
    if args.check_inputs:
        checked_inputs()
        print("Saved display-input identities match; no scientific analysis performed.")
        return
    if args.output is None:
        parser.error("provide --output for a new directory, or --check-inputs")
    output = args.output.resolve()
    if output.exists():
        parser.error("output already exists; choose a new directory")
    for protected in (ROOT / "evidence", HERE):
        if output == protected or protected in output.parents:
            parser.error("output must be separate from evidence and presentation sources")
    data = checked_inputs()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    output.mkdir(parents=True)
    for name, function in (("fidelity_contrasts", fidelity), ("six_model_objectives", objectives),
                           ("utility_and_transfer", utility), ("fit_sensitivity", fit_sensitivity)):
        fig = function(plt, data)
        for extension in ("svg", "pdf"):
            fig.savefig(output / f"{name}.{extension}")
        plt.close(fig)
    print("Re-rendered four saved-summary displays; no new metrics, CIs or decisions.")


if __name__ == "__main__":
    main()
