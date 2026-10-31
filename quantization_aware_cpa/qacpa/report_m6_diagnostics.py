from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report M6 diagnostics from existing M5 UIA CSVs.")
    parser.add_argument("--t-summary", type=Path, default=Path("results/m5_uia_T_robustness/summary.csv"))
    parser.add_argument("--beff-summary", type=Path, default=Path("results/m5_uia_beff_validation/summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m6_diagnostics"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_rows = read_csv(Path(args.t_summary))
    beff_rows = read_csv(Path(args.beff_summary))
    if not t_rows:
        raise ValueError(f"missing rows in {args.t_summary}")
    if not beff_rows:
        raise ValueError(f"missing rows in {args.beff_summary}")

    tensor_error = per_tensor_error_direction(beff_rows)
    seed_std = t_seed_std(t_rows)
    rank_deficit = rank_deficit_rows(t_rows)

    write_csv(output_dir / "per_tensor_error_direction.csv", tensor_error)
    write_csv(output_dir / "T_seed_std.csv", seed_std)
    write_csv(output_dir / "rank_deficit_conditions.csv", rank_deficit)
    render(output_dir / "m6_diagnostics_summary.png", tensor_error, seed_std, rank_deficit)

    print(output_dir / "per_tensor_error_direction.csv")
    print(output_dir / "T_seed_std.csv")
    print(output_dir / "rank_deficit_conditions.csv")
    print(output_dir / "m6_diagnostics_summary.png")


def per_tensor_error_direction(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        scale = str(row["scale_granularity"])
        groups[(scale, int(row["n"]), int(row["T"]))].append(row)
    for (scale, n, T), group in sorted(groups.items()):
        err = np.asarray(
            [int(row["predicted_B_eff"]) - int(row["measured_recovered_cos09"]) for row in group],
            dtype=np.float64,
        )
        out.append(
            {
                "scale_granularity": scale,
                "n": n,
                "T": T,
                "rows": int(err.size),
                "mean_signed_error_pred_minus_measured": float(np.mean(err)),
                "mean_abs_error": float(np.mean(np.abs(err))),
                "overpredict_count": int(np.sum(err > 0)),
                "exact_count": int(np.sum(err == 0)),
                "underpredict_count": int(np.sum(err < 0)),
                "max_overpredict": int(np.max(err)) if err.size else 0,
                "max_underpredict": int(np.min(err)) if err.size else 0,
            }
        )
    return out


def t_seed_std(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    per_seed: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        per_seed[(int(row["n"]), int(row["T"]), int(row["seed"]))].append(row)

    seed_means: dict[tuple[int, int], list[dict[str, float]]] = defaultdict(list)
    for (n, T, seed), group in sorted(per_seed.items()):
        cv = np.asarray([float(row["b_norm_cv"]) for row in group], dtype=np.float64)
        sp = np.asarray([float(row["spearman_final_loss_b_norm"]) for row in group], dtype=np.float64)
        beff = np.asarray([float(row["B_eff_pred"]) for row in group], dtype=np.float64)
        rank = np.asarray([float(row["rank_svd"]) for row in group], dtype=np.float64)
        seed_means[(n, T)].append(
            {
                "seed": float(seed),
                "cv_mean": float(np.mean(cv)),
                "spearman_mean": float(np.mean(sp)),
                "B_eff_mean": float(np.mean(beff)),
                "rank_mean": float(np.mean(rank)),
            }
        )

    out: list[dict[str, Any]] = []
    for (n, T), group in sorted(seed_means.items()):
        cv = np.asarray([item["cv_mean"] for item in group], dtype=np.float64)
        sp = np.asarray([item["spearman_mean"] for item in group], dtype=np.float64)
        beff = np.asarray([item["B_eff_mean"] for item in group], dtype=np.float64)
        rank = np.asarray([item["rank_mean"] for item in group], dtype=np.float64)
        out.append(
            {
                "n": n,
                "T": T,
                "seeds": len(group),
                "cv_seed_mean": float(np.mean(cv)),
                "cv_seed_std": std(cv),
                "spearman_seed_mean": float(np.mean(sp)),
                "spearman_seed_std": std(sp),
                "B_eff_seed_mean": float(np.mean(beff)),
                "B_eff_seed_std": std(beff),
                "rank_seed_mean": float(np.mean(rank)),
                "rank_seed_std": std(rank),
            }
        )
    return out


def rank_deficit_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        n = int(row["n"])
        rank = float(row["rank_svd"])
        if rank < n:
            out.append(
                {
                    "run_tag": row["run_tag"],
                    "seed": int(row["seed"]),
                    "batch_index": int(row["batch_index"]),
                    "n": n,
                    "T": int(row["T"]),
                    "rank_svd": rank,
                    "rank_deficit": float(n - rank),
                    "B_eff_pred": int(row["B_eff_pred"]),
                    "b_norm_cv": float(row["b_norm_cv"]),
                    "spearman_final_loss_b_norm": float(row["spearman_final_loss_b_norm"]),
                }
            )
    return out


def render(
    output: Path,
    tensor_error: list[dict[str, Any]],
    seed_std: list[dict[str, Any]],
    rank_deficit: list[dict[str, Any]],
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    fig.suptitle("M6 Diagnostics from UIA/FedAvg M5 Runs", fontsize=17, fontweight="bold")
    ax_err, ax_cv, ax_sp, ax_rank = axes.ravel()

    scales = sorted({row["scale_granularity"] for row in tensor_error})
    x = np.arange(len(scales))
    signed = [
        np.mean(
            [
                row["mean_signed_error_pred_minus_measured"]
                for row in tensor_error
                if row["scale_granularity"] == scale
            ]
        )
        for scale in scales
    ]
    mae = [
        np.mean([row["mean_abs_error"] for row in tensor_error if row["scale_granularity"] == scale])
        for scale in scales
    ]
    ax_err.bar(x - 0.18, signed, width=0.36, label="signed")
    ax_err.bar(x + 0.18, mae, width=0.36, label="abs")
    ax_err.axhline(0.0, color="#111827", linewidth=1.0)
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(scales)
    ax_err.set_title("B_eff Prediction Error")
    ax_err.set_ylabel("predicted - measured")
    ax_err.legend(frameon=False)
    ax_err.grid(True, axis="y", alpha=0.25)

    plot_seed_std(ax_cv, seed_std, "cv_seed_mean", "cv_seed_std", "CV seed std", "CV")
    plot_seed_std(
        ax_sp,
        seed_std,
        "spearman_seed_mean",
        "spearman_seed_std",
        "Spearman seed std",
        "rho",
    )

    groups: dict[tuple[int, int], int] = defaultdict(int)
    for row in rank_deficit:
        groups[(int(row["n"]), int(row["T"]))] += 1
    labels = [f"n={n},T={T}" for (n, T) in sorted(groups)]
    counts = [groups[key] for key in sorted(groups)]
    if labels:
        ax_rank.bar(np.arange(len(labels)), counts, color="#dc2626")
        ax_rank.set_xticks(np.arange(len(labels)))
        ax_rank.set_xticklabels(labels, rotation=45, ha="right")
    else:
        ax_rank.text(0.5, 0.5, "No rank<n rows", ha="center", va="center", transform=ax_rank.transAxes)
    ax_rank.set_title("Rank Deficit Conditions")
    ax_rank.set_ylabel("replicate count")
    ax_rank.grid(True, axis="y", alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_seed_std(ax: Any, rows: list[dict[str, Any]], mean_key: str, std_key: str, title: str, ylabel: str) -> None:
    colors = {16: "#2563eb", 32: "#dc2626"}
    for n in sorted({int(row["n"]) for row in rows}):
        group = [row for row in rows if int(row["n"]) == n]
        x = np.asarray([int(row["T"]) for row in group], dtype=np.float64)
        y = np.asarray([float(row[mean_key]) for row in group], dtype=np.float64)
        s = np.asarray([float(row[std_key]) for row in group], dtype=np.float64)
        ax.plot(x, y, marker="o", linewidth=2.0, color=colors.get(n), label=f"n={n}")
        ax.fill_between(x, y - s, y + s, color=colors.get(n), alpha=0.14)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    ax.set_title(title)
    ax.set_xlabel("local steps T")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)


def std(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
