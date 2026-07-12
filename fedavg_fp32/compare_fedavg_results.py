#!/usr/bin/env python3
"""Compare base, 8bit, and 4bit FedAvg result directories."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SKIP_METRICS = {"round", "quant_bits"}
METHOD_ORDER = {"base": 0, "8bit": 1, "4bit": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FedAvg quantization experiment outputs.")
    parser.add_argument("result_dirs", type=Path, nargs="+", help="Experiment output directories.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("fedavg_comparison"),
        help="Directory for comparison_summary.csv and plots.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional metric column names to plot. Defaults to every numeric metric in metrics.csv.",
    )
    return parser.parse_args()


def parse_number(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_metrics(result_dir: Path) -> List[Dict[str, Optional[float]]]:
    metrics_path = result_dir / "metrics.csv"
    with metrics_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    metrics: List[Dict[str, Optional[float]]] = []
    for row in rows:
        parsed = {key: parse_number(value) for key, value in row.items()}
        if parsed.get("round") is None:
            raise ValueError(f"Missing numeric round value in {metrics_path}")
        parsed["round"] = int(parsed["round"])
        metrics.append(parsed)
    return metrics


def read_summary(result_dir: Path) -> Dict:
    summary_path = result_dir / "summary.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def read_config(result_dir: Path) -> Dict:
    config_path = result_dir / "config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def quant_bits(summary: Dict, config: Dict, metrics: Sequence[Dict[str, Optional[float]]]) -> int:
    if "quant_bits" in summary:
        return int(summary["quant_bits"])
    if "quant_bits" in config:
        return int(config["quant_bits"])
    for row in metrics:
        if row.get("quant_bits") is not None:
            return int(row["quant_bits"])
    return 32


def method_name(bits: int) -> str:
    if bits == 32:
        return "base"
    return f"{bits}bit"


def metric_values(metrics: Sequence[Dict[str, Optional[float]]], metric: str) -> List[float]:
    return [float(row[metric]) for row in metrics if row.get(metric) is not None]


def final_metric(metrics: Sequence[Dict[str, Optional[float]]], metric: str) -> Optional[float]:
    values = metric_values(metrics, metric)
    return values[-1] if values else None


def best_accuracy(metrics: Sequence[Dict[str, Optional[float]]]) -> tuple[Optional[float], Optional[int]]:
    rows = [row for row in metrics if row.get("val_accuracy") is not None]
    if not rows:
        return None, None
    best = max(rows, key=lambda row: float(row["val_accuracy"]))
    return float(best["val_accuracy"]), int(best["round"])


def average_metric(metrics: Sequence[Dict[str, Optional[float]]], metric: str) -> Optional[float]:
    values = metric_values(metrics, metric)
    return sum(values) / len(values) if values else None


def summary_value(summary: Dict, key: str, fallback: Optional[float]) -> Optional[float]:
    return summary[key] if key in summary else fallback


def write_comparison_summary(experiments: Sequence[Dict], output_dir: Path) -> None:
    base = next((exp for exp in experiments if exp["method"] == "base"), None)
    base_accuracy = base["final_accuracy"] if base else None
    fields = [
        "method",
        "quant_bits",
        "final_accuracy",
        "best_accuracy",
        "best_round",
        "accuracy_drop_vs_fp32",
        "average_quant_mse",
        "average_relative_l2",
        "average_cosine_similarity",
        "average_saturation_ratio",
        "total_communication_bytes",
        "compression_ratio_vs_fp32",
    ]
    with (output_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for exp in experiments:
            final_accuracy = exp["final_accuracy"]
            accuracy_drop = "" if base_accuracy is None or final_accuracy is None else base_accuracy - final_accuracy
            writer.writerow(
                {
                    "method": exp["method"],
                    "quant_bits": exp["quant_bits"],
                    "final_accuracy": final_accuracy,
                    "best_accuracy": exp["best_accuracy"],
                    "best_round": exp["best_round"],
                    "accuracy_drop_vs_fp32": accuracy_drop,
                    "average_quant_mse": exp["average_quant_mse"],
                    "average_relative_l2": exp["average_relative_l2"],
                    "average_cosine_similarity": exp["average_cosine_similarity"],
                    "average_saturation_ratio": exp["average_saturation_ratio"],
                    "total_communication_bytes": exp["total_communication_bytes"],
                    "compression_ratio_vs_fp32": exp["compression_ratio_vs_fp32"],
                }
            )


def discover_metrics(experiments: Sequence[Dict]) -> List[str]:
    metrics: List[str] = []
    for exp in experiments:
        for row in exp["metrics"]:
            for key, value in row.items():
                if key in SKIP_METRICS or value is None or key in metrics:
                    continue
                metrics.append(key)
    return metrics


def ylabel(metric: str) -> str:
    return metric.replace("_", " ").title()


def plot_metric(experiments: Sequence[Dict], metric: str, output_path: Path) -> bool:
    has_values = False
    plt.figure(figsize=(9, 5.5))
    for exp in experiments:
        pairs = [
            (int(row["round"]), float(row[metric]))
            for row in exp["metrics"]
            if row.get(metric) is not None
        ]
        if not pairs:
            continue
        has_values = True
        rounds = [round_num for round_num, _ in pairs]
        values = [value for _, value in pairs]
        plt.plot(rounds, values, marker="o", linewidth=1.8, markersize=3.5, label=exp["method"])
    if not has_values:
        plt.close()
        return False
    plt.title(ylabel(metric))
    plt.xlabel("Round")
    plt.ylabel(ylabel(metric))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = []
    for result_dir in args.result_dirs:
        result_dir = result_dir.resolve()
        metrics = read_metrics(result_dir)
        summary = read_summary(result_dir)
        config = read_config(result_dir)
        bits = quant_bits(summary, config, metrics)
        best_acc, best_round = best_accuracy(metrics)
        experiments.append(
            {
                "dir": result_dir,
                "summary": summary,
                "config": config,
                "metrics": metrics,
                "quant_bits": bits,
                "method": method_name(bits),
                "final_accuracy": summary_value(summary, "final_accuracy", final_metric(metrics, "val_accuracy")),
                "best_accuracy": summary_value(summary, "best_accuracy", best_acc),
                "best_round": summary_value(summary, "best_round", best_round),
                "average_quant_mse": summary_value(summary, "average_quant_mse", average_metric(metrics, "quant_mse")),
                "average_relative_l2": summary_value(
                    summary, "average_relative_l2", average_metric(metrics, "quant_relative_l2")
                ),
                "average_cosine_similarity": summary_value(
                    summary, "average_cosine_similarity", average_metric(metrics, "quant_cosine_similarity")
                ),
                "average_saturation_ratio": summary_value(
                    summary, "average_saturation_ratio", average_metric(metrics, "quant_saturation_ratio")
                ),
                "total_communication_bytes": summary_value(
                    summary, "total_communication_bytes", sum(metric_values(metrics, "communication_bytes"))
                ),
                "compression_ratio_vs_fp32": summary_value(
                    summary, "compression_ratio_vs_fp32", final_metric(metrics, "compression_ratio_vs_fp32")
                ),
            }
        )
    experiments.sort(key=lambda exp: METHOD_ORDER.get(exp["method"], 99))

    write_comparison_summary(experiments, output_dir)
    requested_metrics = args.metrics if args.metrics else discover_metrics(experiments)
    written = []
    for metric in requested_metrics:
        output_path = output_dir / f"{metric}.png"
        if plot_metric(experiments, metric, output_path):
            written.append(output_path.name)

    present_methods = {exp["method"] for exp in experiments}
    missing_methods = [method for method in METHOD_ORDER if method not in present_methods]
    if missing_methods:
        print(f"Missing result dirs for: {', '.join(missing_methods)}")
    print(f"Comparison summary saved to {output_dir / 'comparison_summary.csv'}")
    print(f"Metric plots saved to {output_dir}: {', '.join(written)}")


if __name__ == "__main__":
    main()
