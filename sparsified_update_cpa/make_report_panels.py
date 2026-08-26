#!/usr/bin/env python3
"""Build report PNG panels from completed Naive Sparse CPA runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


DEFAULT_RUNS_ROOT = Path(
    "sparsified_update_cpa/outputs_v1/author_runs/imagenet/vgg16/attack/cp_direct/nodef"
)
DEFAULT_PATTERN = "01_naive_sparse_cpa_client_topk_global_topk_keep*_n64_steps10_trials3_fia8_uia"
DEFAULT_REPORT_DIR = Path("sparsified_update_cpa/outputs_v1/png_report/n64_steps10_trials3_fia8")


ITER_METRICS = {
    "gi": ["loss", "loss_decor", "loss_l1", "loss_nv"],
    "fi": ["loss", "loss_fi", "loss_tv"],
}
SUMMARY_METRICS = [
    ("lpips", "LPIPS", "lower is better"),
    ("psnr", "PSNR", "higher is better"),
    ("ssim", "SSIM", "higher is better"),
    ("embedding_cosine_similarity", "Embedding cosine", "higher is better"),
    ("reconstruction_success_rate", "Success rate", "higher is better"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def keep_ratio_from_name(name: str) -> float:
    match = re.search(r"keep([0-9p]+)_n", name)
    if not match:
        raise ValueError(f"Cannot parse keep ratio from {name}")
    return float(match.group(1).replace("p", "."))


def ratio_tag(keep_ratio: float) -> str:
    return f"{keep_ratio:.2f}".replace(".", "p")


def load_summary(run_dir: Path) -> dict[str, dict[str, str]]:
    with (run_dir / "metric_summary.csv").open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: row for row in csv.DictReader(handle)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_runs(runs_root: Path, pattern: str) -> list[Path]:
    runs = sorted(runs_root.glob(pattern), key=lambda path: keep_ratio_from_name(path.name), reverse=True)
    if not runs:
        raise FileNotFoundError(f"No runs matched {runs_root / pattern}")
    return runs


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate_iter_rows(run_dirs: list[Path]) -> dict[float, dict[str, dict[str, dict[int, float]]]]:
    result: dict[float, dict[str, dict[str, dict[int, float]]]] = {}
    for run_dir in run_dirs:
        keep = keep_ratio_from_name(run_dir.name)
        rows = load_jsonl(run_dir / "losses.jsonl")
        by_phase_metric_iter: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for row in rows:
            phase = str(row.get("phase"))
            if phase not in ITER_METRICS:
                continue
            iteration = int(row["iter"])
            for metric in ITER_METRICS[phase]:
                if metric in row:
                    by_phase_metric_iter[phase][metric][iteration].append(float(row[metric]))
        keep_result: dict[str, dict[str, dict[int, float]]] = {}
        for phase, metric_map in by_phase_metric_iter.items():
            keep_result[phase] = {}
            for metric, iter_map in metric_map.items():
                keep_result[phase][metric] = {
                    iteration: mean(values) for iteration, values in sorted(iter_map.items())
                }
        result[keep] = keep_result
    return result


def plot_iter_metric(
    aggregated: dict[float, dict[str, dict[str, dict[int, float]]]],
    *,
    phase: str,
    metric: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    for keep in sorted(aggregated.keys(), reverse=True):
        series = aggregated[keep].get(phase, {}).get(metric, {})
        if not series:
            continue
        xs = sorted(series)
        ys = [series[x] for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=2, label=f"keep {keep:.0%}")
    ax.set_title(f"{phase.upper()} {metric} by iteration")
    ax.set_xlabel("iteration")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_iter_panel(
    aggregated: dict[float, dict[str, dict[str, dict[int, float]]]],
    *,
    phase: str,
    out_path: Path,
) -> None:
    metrics = ITER_METRICS[phase]
    n_cols = 2
    n_rows = (len(metrics) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.5 * n_rows), dpi=150)
    axes_list = list(axes.flat if hasattr(axes, "flat") else [axes])
    for ax, metric in zip(axes_list, metrics):
        for keep in sorted(aggregated.keys(), reverse=True):
            series = aggregated[keep].get(phase, {}).get(metric, {})
            if not series:
                continue
            xs = sorted(series)
            ys = [series[x] for x in xs]
            ax.plot(xs, ys, marker="o", linewidth=2, label=f"keep {keep:.0%}")
        ax.set_title(metric)
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.35)
    for ax in axes_list[len(metrics) :]:
        ax.axis("off")
    axes_list[0].legend()
    fig.suptitle(f"{phase.upper()} losses by iteration")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def load_summary_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        keep = keep_ratio_from_name(run_dir.name)
        summary = load_summary(run_dir)
        row = {"keep_ratio": keep}
        for metric, _label, _direction in SUMMARY_METRICS:
            row[metric] = float(summary[metric]["mean"])
        rows.append(row)
    return sorted(rows, key=lambda row: row["keep_ratio"], reverse=True)


def plot_summary_metric(rows: list[dict[str, Any]], metric: str, label: str, direction: str, out_path: Path) -> None:
    xs = [f"{row['keep_ratio']:.0%}" for row in rows]
    ys = [row[metric] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(xs, ys, marker="o", linewidth=2)
    ax.set_title(f"{label} by keep ratio ({direction})")
    ax.set_xlabel("keep ratio")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_summary_panel(rows: list[dict[str, Any]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150)
    axes_list = list(axes.flat)
    labels = [f"{row['keep_ratio']:.0%}" for row in rows]
    for ax, (metric, label, direction) in zip(axes_list, SUMMARY_METRICS):
        values = [row[metric] for row in rows]
        ax.plot(labels, values, marker="o", linewidth=2)
        ax.set_title(f"{label} ({direction})")
        ax.set_xlabel("keep ratio")
        ax.grid(True, alpha=0.35)
    for ax in axes_list[len(SUMMARY_METRICS) :]:
        ax.axis("off")
    fig.suptitle("Client-level Naive Sparse CPA metrics by keep ratio")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def add_header(image: Image.Image, text: str) -> Image.Image:
    font = load_font(32)
    header_h = 64
    canvas = Image.new("RGB", (image.width, image.height + header_h), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 16), text, fill=(15, 23, 42), font=font)
    canvas.paste(image, (0, header_h))
    return canvas


def stack_compare_images(report_dir: Path, keep: float, out_path: Path) -> None:
    keep_dir = report_dir / f"keep_{ratio_tag(keep)}"
    compare_paths = sorted(keep_dir.glob("batch_*_compare.png"))
    if not compare_paths:
        raise FileNotFoundError(f"No compare PNGs found in {keep_dir}")
    images = [add_header(Image.open(path).convert("RGB"), path.stem.replace("_", " ")) for path in compare_paths]
    width = max(image.width for image in images)
    gap = 18
    height = sum(image.height for image in images) + gap * (len(images) + 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = gap
    for image in images:
        x = (width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.report_dir / "report_panels")
    run_dirs = discover_runs(args.runs_root, args.pattern)

    aggregated = aggregate_iter_rows(run_dirs)
    for phase, metrics in ITER_METRICS.items():
        plot_iter_panel(aggregated, phase=phase, out_path=output_dir / "iteration" / f"{phase}_loss_panel.png")
        for metric in metrics:
            plot_iter_metric(
                aggregated,
                phase=phase,
                metric=metric,
                out_path=output_dir / "iteration" / f"{phase}_{metric}_by_iteration.png",
            )

    summary_rows = load_summary_rows(run_dirs)
    plot_summary_panel(summary_rows, output_dir / "metrics" / "all_metrics_by_keep.png")
    for metric, label, direction in SUMMARY_METRICS:
        plot_summary_metric(
            summary_rows,
            metric,
            label,
            direction,
            output_dir / "metrics" / f"{metric}_by_keep.png",
        )

    for run_dir in run_dirs:
        keep = keep_ratio_from_name(run_dir.name)
        stack_compare_images(
            args.report_dir,
            keep,
            output_dir / "compare_by_keep" / f"keep_{ratio_tag(keep)}_all_batches_compare.png",
        )

    print(f"Report panels written to: {output_dir}")
    print(f"Iteration panels: {output_dir / 'iteration'}")
    print(f"Metric panels: {output_dir / 'metrics'}")
    print(f"Compare panels: {output_dir / 'compare_by_keep'}")


if __name__ == "__main__":
    main()
