from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one slide-sized PNG for UIA/FedAvg M5 results.")
    parser.add_argument("--t-dir", type=Path, default=Path("results/m5_uia_T_robustness"))
    parser.add_argument("--beff-dir", type=Path, default=Path("results/m5_uia_beff_validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m5_uia_summary"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_rows = read_csv(Path(args.t_dir) / "aggregate_summary.csv")
    beff_rows = read_csv(Path(args.beff_dir) / "summary.csv")
    if not t_rows:
        raise ValueError(f"missing rows in {Path(args.t_dir) / 'aggregate_summary.csv'}")
    if not beff_rows:
        raise ValueError(f"missing rows in {Path(args.beff_dir) / 'summary.csv'}")
    render(t_rows, beff_rows, output_dir / "m5_uia_results_summary.png")
    print(output_dir / "m5_uia_results_summary.png")


def render(t_rows: list[dict[str, str]], beff_rows: list[dict[str, str]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )
    fig = plt.figure(figsize=(16, 9), dpi=180)
    gs = fig.add_gridspec(
        2,
        4,
        left=0.055,
        right=0.985,
        top=0.895,
        bottom=0.085,
        hspace=0.38,
        wspace=0.32,
    )
    fig.suptitle("UIA / FedAvg: Spectrum Bias and MP Prediction", fontsize=18, fontweight="bold", y=0.965)
    fig.text(
        0.055,
        0.925,
        "VGG-16 frozen conv, FC-only FedAvg, 4-bit stochastic per-row for T robustness; ICA+Hungarian only for B_eff validation",
        fontsize=10,
        color="#4b5563",
    )
    ax_cv = fig.add_subplot(gs[0, 0])
    ax_sp = fig.add_subplot(gs[0, 1])
    ax_beff = fig.add_subplot(gs[0, 2])
    ax_rank = fig.add_subplot(gs[0, 3])
    ax_scatter = fig.add_subplot(gs[1, 0:3])
    ax_bar = fig.add_subplot(gs[1, 3])

    plot_t_metric(ax_cv, t_rows, "b_norm_cv_mean", "b_norm_cv_ci95", "Norm CV", "CV")
    plot_t_metric(ax_sp, t_rows, "spearman_mean", "spearman_ci95", "Loss-Norm Spearman", "rho")
    ax_sp.set_ylim(0.84, 1.0)
    plot_t_metric(ax_beff, t_rows, "B_eff_mean", "B_eff_ci95", "Predicted B_eff", "B_eff")
    plot_rank(ax_rank, t_rows)
    plot_prediction_scatter(ax_scatter, beff_rows)
    plot_validation_summary(ax_bar, beff_rows)

    fig.savefig(output)
    plt.close(fig)


def plot_t_metric(
    ax: Any,
    rows: list[dict[str, str]],
    mean_key: str,
    ci_key: str,
    title: str,
    ylabel: str,
) -> None:
    colors = {16: "#2563eb", 32: "#dc2626"}
    for n in sorted({int(row["n"]) for row in rows}):
        group = [row for row in rows if int(row["n"]) == n]
        x = np.asarray([int(row["T"]) for row in group], dtype=np.float64)
        y = np.asarray([float(row[mean_key]) for row in group], dtype=np.float64)
        ci = np.asarray([float(row[ci_key]) for row in group], dtype=np.float64)
        color = colors.get(n, "#374151")
        ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=f"n={n}")
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.14)
    style_t_axis(ax, title, ylabel)


def plot_rank(ax: Any, rows: list[dict[str, str]]) -> None:
    colors = {16: "#2563eb", 32: "#dc2626"}
    for n in sorted({int(row["n"]) for row in rows}):
        group = [row for row in rows if int(row["n"]) == n]
        x = np.asarray([int(row["T"]) for row in group], dtype=np.float64)
        y = np.asarray([float(row["rank_mean"]) for row in group], dtype=np.float64)
        y_min = np.asarray([float(row["rank_min"]) for row in group], dtype=np.float64)
        y_max = np.asarray([float(row["rank_max"]) for row in group], dtype=np.float64)
        color = colors.get(n, "#374151")
        ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=f"n={n}")
        ax.fill_between(x, y_min, y_max, color=color, alpha=0.14)
        ax.axhline(n, linestyle=":", color=color, linewidth=1.0, alpha=0.55)
    style_t_axis(ax, "True Rank", "SVD rank")


def style_t_axis(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    ax.set_xlabel("local steps T")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)


def plot_prediction_scatter(ax: Any, rows: list[dict[str, str]]) -> None:
    colors = {"per_row": "#2563eb", "per_tensor": "#dc2626", "per_bucket": "#059669"}
    markers = {1: "o", 10: "s"}
    for row in rows:
        scale = str(row["scale_granularity"])
        T = int(row["T"])
        ax.scatter(
            int(row["predicted_B_eff"]),
            int(row["measured_recovered_cos09"]),
            color=colors.get(scale, "#6b7280"),
            marker=markers.get(T, "o"),
            s=54,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.45,
        )
    max_axis = max(
        max(int(row["n"]) for row in rows),
        max(int(row["predicted_B_eff"]) for row in rows),
        max(int(row["measured_recovered_cos09"]) for row in rows),
    ) + 2
    ax.plot([0, max_axis], [0, max_axis], linestyle="--", color="#111827", linewidth=1.0)
    ax.set_xlim(0, max_axis)
    ax.set_ylim(0, max_axis)
    ax.set_title("Predicted vs Measured B_eff")
    ax.set_xlabel("predicted B_eff from MP threshold")
    ax.set_ylabel("measured recovered count (cos > 0.9)")
    ax.grid(True, alpha=0.25)
    handles = [
        plt.Line2D([], [], linestyle="", marker="o", color=color, label=scale)
        for scale, color in colors.items()
    ]
    handles.extend(
        plt.Line2D([], [], linestyle="", marker=marker, color="#374151", label=f"T={T}")
        for T, marker in markers.items()
    )
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=2)


def plot_validation_summary(ax: Any, rows: list[dict[str, str]]) -> None:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        err = abs(int(row["predicted_B_eff"]) - int(row["measured_recovered_cos09"]))
        groups[str(row["scale_granularity"])].append(err)
    labels = ["per_row", "per_bucket", "per_tensor"]
    colors = ["#2563eb", "#059669", "#dc2626"]
    exact = [sum(err == 0 for err in groups[label]) / max(1, len(groups[label])) for label in labels]
    mae = [sum(groups[label]) / max(1, len(groups[label])) for label in labels]
    x = np.arange(len(labels))
    ax.bar(x, exact, color=colors, alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(["row", "bucket", "tensor"])
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Exact Prediction Rate")
    ax.set_ylabel("fraction exact")
    ax.grid(True, axis="y", alpha=0.25)
    for idx, (rate, mean_err) in enumerate(zip(exact, mae)):
        ax.text(idx, rate + 0.035, f"{rate:.2f}", ha="center", va="bottom", fontsize=10)
        ax.text(idx, 0.08, f"MAE {mean_err:.2f}", ha="center", va="center", fontsize=9, color="#111827")
    total = len(rows)
    exact_total = sum(
        int(row["predicted_B_eff"]) == int(row["measured_recovered_cos09"]) for row in rows
    )
    mae_total = np.mean(
        [
            abs(int(row["predicted_B_eff"]) - int(row["measured_recovered_cos09"]))
            for row in rows
        ]
    )
    ax.text(
        0.5,
        -0.24,
        f"overall: {exact_total}/{total} exact, MAE {mae_total:.3f}",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
