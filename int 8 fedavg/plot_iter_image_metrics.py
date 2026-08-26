#!/usr/bin/env python3
"""Plot per-iteration PSNR/SSIM/LPIPS curves from FIA image metric CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "int 8 fedavg"
    / "imagenet"
    / "vgg16"
    / "clients3_rounds20_spc8_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot iter-wise image metrics produced by --log_image_metrics FIA runs."
    )
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        default=[],
        help="Optional label=path or path to fia_image_metrics_iter.csv. Can be repeated.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def parse_input_spec(spec: str) -> Tuple[Optional[str], Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), resolve_path(Path(path.strip()))
    return None, resolve_path(Path(spec))


def discover_inputs(run_dir: Path) -> List[Tuple[str, Path]]:
    paths = sorted(run_dir.glob("**/fia_image_metrics_iter.csv"))
    discovered: List[Tuple[str, Path]] = []
    for path in paths:
        if "metric_comparison_graphs" in path.parts:
            continue
        discovered.append((label_for_path(path, run_dir), path))
    return discovered


def label_for_path(path: Path, run_dir: Path) -> str:
    parts = path.relative_to(run_dir).parts if run_dir in path.parents else path.parts
    joined = "/".join(parts)
    if "fastica_dequantized_aggregate" in parts:
        return "naive_dequant_aggregate"
    for part in parts:
        if part.startswith("randomized_R"):
            return part
    if "naive" in parts:
        return "naive_client_qs"
    return joined.replace("/fia_image_metrics_iter.csv", "").replace("/", "_")


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def row_weight(row: Dict[str, str]) -> float:
    start = parse_float(row.get("chunk_start"))
    end = parse_float(row.get("chunk_end"))
    if start is None or end is None:
        return 1.0
    return max(1.0, end - start)


def load_curve(path: Path) -> List[Dict[str, float]]:
    by_iter: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    weights: Dict[int, float] = defaultdict(float)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            iter_value = parse_float(row.get("iter"))
            if iter_value is None:
                continue
            iteration = int(iter_value)
            weight = row_weight(row)
            any_metric = False
            for metric in ("psnr_mean", "ssim_mean", "lpips_mean"):
                value = parse_float(row.get(metric))
                if value is None:
                    continue
                by_iter[iteration][metric] += value * weight
                any_metric = True
            if any_metric:
                weights[iteration] += weight

    curve = []
    for iteration in sorted(by_iter):
        weight = weights[iteration]
        if weight <= 0:
            continue
        item = {"iter": float(iteration)}
        for metric, total in by_iter[iteration].items():
            item[metric] = total / weight
        curve.append(item)
    return curve


def write_summary(rows: Iterable[Dict[str, object]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "iter", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(curves: Dict[str, List[Dict[str, float]]], metric: str, output_path: Path) -> bool:
    import matplotlib.pyplot as plt

    plotted = False
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    for label, curve in curves.items():
        xs = [row["iter"] for row in curve if metric in row]
        ys = [row[metric] for row in curve if metric in row]
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.2, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("FIA iteration")
    ax.set_ylabel(metric.replace("_mean", "").upper())
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else run_dir / "metric_comparison_graphs" / "iter_image_metrics"
    )

    if args.inputs:
        inputs = []
        for spec in args.inputs:
            label, path = parse_input_spec(spec)
            inputs.append((label or label_for_path(path, run_dir), path))
    else:
        inputs = discover_inputs(run_dir)

    curves: Dict[str, List[Dict[str, float]]] = {}
    for label, path in inputs:
        if not path.is_file():
            print(f"skip missing: {path}")
            continue
        curve = load_curve(path)
        if not curve:
            print(f"skip empty: {path}")
            continue
        curves[label] = curve

    if not curves:
        raise FileNotFoundError(
            "No fia_image_metrics_iter.csv data found. Re-run FIA with --log_image_metrics first."
        )

    summary_rows = []
    for label, curve in curves.items():
        for row in curve:
            for metric in ("psnr_mean", "ssim_mean", "lpips_mean"):
                if metric in row:
                    summary_rows.append(
                        {
                            "label": label,
                            "iter": int(row["iter"]),
                            "metric": metric,
                            "value": row[metric],
                        }
                    )
    write_summary(summary_rows, output_dir / "iter_image_metric_summary.csv")

    outputs = {
        "psnr_mean": output_dir / "psnr_by_iter.png",
        "ssim_mean": output_dir / "ssim_by_iter.png",
        "lpips_mean": output_dir / "lpips_by_iter.png",
    }
    for metric, path in outputs.items():
        if plot_metric(curves, metric, path):
            print(f"saved: {path}")
        else:
            print(f"skip {metric}: no values")
    print(f"summary saved: {output_dir / 'iter_image_metric_summary.csv'}")


if __name__ == "__main__":
    main()
