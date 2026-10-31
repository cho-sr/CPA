from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and plot UIA/FedAvg T-robustness results.")
    parser.add_argument("--input-dir", type=Path, default=Path("results/m5_uia_T_robustness"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m5_uia_T_robustness"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.input_dir) / "summary.csv")
    if not rows:
        raise ValueError(f"missing rows in {Path(args.input_dir) / 'summary.csv'}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_rows(rows)
    write_csv(output_dir / "aggregate_summary.csv", aggregate)
    render_figure(aggregate, output_dir / "uia_T_robustness_summary.png")
    print(output_dir / "aggregate_summary.csv")
    print(output_dir / "uia_T_robustness_summary.png")


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["n"]), int(row["T"]))].append(row)
    out: list[dict[str, Any]] = []
    for (n, T), group in sorted(groups.items()):
        ranks = np.asarray([float(row["rank_svd"]) for row in group], dtype=np.float64)
        beff = np.asarray([float(row["B_eff_pred"]) for row in group], dtype=np.float64)
        cv = np.asarray([float(row["b_norm_cv"]) for row in group], dtype=np.float64)
        gia_cv = np.asarray([float(row["gia_a_norm_cv"]) for row in group], dtype=np.float64)
        sp = np.asarray([float(row["spearman_final_loss_b_norm"]) for row in group], dtype=np.float64)
        out.append(
            {
                "n": n,
                "T": T,
                "replicates": len(group),
                "rank_min": float(np.min(ranks)),
                "rank_mean": float(np.mean(ranks)),
                "rank_max": float(np.max(ranks)),
                "rank_gt_n_count": int(np.sum(ranks > n)),
                "B_eff_mean": float(np.mean(beff)),
                "B_eff_ci95": ci95(beff),
                "B_eff_min": float(np.min(beff)),
                "B_eff_max": float(np.max(beff)),
                "b_norm_cv_mean": float(np.mean(cv)),
                "b_norm_cv_ci95": ci95(cv),
                "b_norm_cv_min": float(np.min(cv)),
                "b_norm_cv_max": float(np.max(cv)),
                "gia_a_norm_cv_mean": float(np.mean(gia_cv)),
                "spearman_mean": float(np.mean(sp)),
                "spearman_ci95": ci95(sp),
                "spearman_min": float(np.min(sp)),
                "spearman_max": float(np.max(sp)),
            }
        )
    return out


def render_figure(rows: list[dict[str, Any]], output: Path) -> None:
    n_values = sorted({int(row["n"]) for row in rows})
    colors = {16: "#2563eb", 32: "#dc2626", 64: "#059669"}
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    fig.suptitle("UIA / FedAvg T-Robustness (5 batches x 3 seeds)", fontsize=17, fontweight="bold")
    ax_rank, ax_beff, ax_cv, ax_sp = axes.ravel()
    for n in n_values:
        group = [row for row in rows if int(row["n"]) == n]
        T = np.asarray([int(row["T"]) for row in group], dtype=np.float64)
        color = colors.get(n)
        rank_mean = np.asarray([float(row["rank_mean"]) for row in group])
        rank_min = np.asarray([float(row["rank_min"]) for row in group])
        rank_max = np.asarray([float(row["rank_max"]) for row in group])
        ax_rank.plot(T, rank_mean, marker="o", color=color, linewidth=2.2, label=f"n={n}")
        ax_rank.fill_between(T, rank_min, rank_max, color=color, alpha=0.14)
        ax_rank.axhline(n, linestyle=":", linewidth=1.0, color=color, alpha=0.45)

        plot_ci(ax_beff, T, group, "B_eff_mean", "B_eff_ci95", color, f"n={n}")
        plot_ci(ax_cv, T, group, "b_norm_cv_mean", "b_norm_cv_ci95", color, f"UIA n={n}")
        gia = np.asarray([float(row["gia_a_norm_cv_mean"]) for row in group])
        ax_cv.plot(T, gia, linestyle="--", linewidth=1.4, color=color, alpha=0.7, label=f"GIA n={n}")
        plot_ci(ax_sp, T, group, "spearman_mean", "spearman_ci95", color, f"n={n}")

    ax_rank.set_title("True Rank of B_T Z")
    ax_rank.set_xlabel("local steps T")
    ax_rank.set_ylabel("SVD rank")
    ax_rank.grid(True, alpha=0.25)
    ax_rank.legend()

    ax_beff.set_title("Predicted B_eff")
    ax_beff.set_xlabel("local steps T")
    ax_beff.set_ylabel("predicted B_eff")
    ax_beff.grid(True, alpha=0.25)
    ax_beff.legend()

    ax_cv.set_title("Norm Bias CV")
    ax_cv.set_xlabel("local steps T")
    ax_cv.set_ylabel("coefficient of variation")
    ax_cv.grid(True, alpha=0.25)
    ax_cv.legend(ncol=2)

    ax_sp.set_title("Spearman(final loss, ||b_i||)")
    ax_sp.set_xlabel("local steps T")
    ax_sp.set_ylabel("Spearman rho")
    ax_sp.set_ylim(0.75, 1.0)
    ax_sp.grid(True, alpha=0.25)
    ax_sp.legend()

    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_ci(ax, T: np.ndarray, group: list[dict[str, Any]], mean_key: str, ci_key: str, color: str, label: str) -> None:
    mean = np.asarray([float(row[mean_key]) for row in group], dtype=np.float64)
    ci = np.asarray([float(row[ci_key]) for row in group], dtype=np.float64)
    ax.plot(T, mean, marker="o", color=color, linewidth=2.2, label=label)
    ax.fill_between(T, mean - ci, mean + ci, color=color, alpha=0.14)


def ci95(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(values.size))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
