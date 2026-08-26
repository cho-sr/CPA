#!/usr/bin/env python3
"""Plot completed single-client CoRA metrics by keep ratio."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTHOR_ROOT = (
    REPO_ROOT
    / "sparsified_update_cpa"
    / "outputs_v1"
    / "author_runs"
    / "imagenet"
    / "vgg16"
    / "attack"
    / "cp_direct"
    / "nodef"
)
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs_v1"
    / "png_report"
    / "single_client_metric_ratio_graph"
)

RUNS = [
    {
        "keep": 0.8,
        "member": "m007",
        "run": AUTHOR_ROOT / "cora_single_client_cpa_keep0p8_n24_steps1_trials1_R8_seed1234_m007_uia",
        "oracle_summary": "metric_summary_cora_single_oracle_embedding.csv",
    },
    {
        "keep": 0.5,
        "member": "m001",
        "run": AUTHOR_ROOT / "cora_single_client_cpa_keep0p5_n24_steps1_trials1_R8_seed1234_m001_uia",
        "oracle_summary": "metric_summary_cora_single_keep0p5_oracle_embedding.csv",
    },
    {
        "keep": 0.3,
        "member": "m007",
        "run": AUTHOR_ROOT / "cora_single_client_cpa_keep0p3_n24_steps1_trials1_R8_seed1234_m007_uia",
        "oracle_summary": "metric_summary_cora_single_keep0p3_oracle_embedding.csv",
    },
]

METRIC_NAMES = ["lpips", "psnr", "ssim"]


def read_summary(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metric = row["metric"]
            rows[metric] = {
                "mean": float(row["mean"]) if row["mean"] else float("nan"),
                "std": float(row["std"]) if row["std"] else float("nan"),
                "n": float(row["n"]) if row["n"] else float("nan"),
            }
    return rows


def collect_rows() -> list[dict[str, object]]:
    collected: list[dict[str, object]] = []
    for item in RUNS:
        run_dir = Path(item["run"])
        direct = read_summary(run_dir / "metric_summary.csv")
        oracle = read_summary(run_dir / str(item["oracle_summary"]))
        for metric in METRIC_NAMES:
            collected.append(
                {
                    "keep_ratio": item["keep"],
                    "keep_percent": int(round(float(item["keep"]) * 100)),
                    "selected_member": item["member"],
                    "evaluation": "direct",
                    "metric": metric,
                    "mean": direct[metric]["mean"],
                    "std": direct[metric]["std"],
                    "n": int(direct[metric]["n"]),
                    "run_dir": str(run_dir),
                }
            )
            collected.append(
                {
                    "keep_ratio": item["keep"],
                    "keep_percent": int(round(float(item["keep"]) * 100)),
                    "selected_member": item["member"],
                    "evaluation": "oracle_embedding",
                    "metric": metric,
                    "mean": oracle[metric]["mean"],
                    "std": oracle[metric]["std"],
                    "n": int(oracle[metric]["n"]),
                    "run_dir": str(run_dir),
                }
            )
    return collected


def write_csv_summary(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "keep_ratio",
        "keep_percent",
        "selected_member",
        "evaluation",
        "metric",
        "mean",
        "std",
        "n",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def values_for(rows: list[dict[str, object]], metric: str, evaluation: str) -> tuple[list[float], list[float], list[float]]:
    filtered = [
        row
        for row in rows
        if row["metric"] == metric and row["evaluation"] == evaluation
    ]
    filtered.sort(key=lambda row: float(row["keep_ratio"]))
    xs = [float(row["keep_percent"]) for row in filtered]
    means = [float(row["mean"]) for row in filtered]
    stds = [float(row["std"]) for row in filtered]
    return xs, means, stds


def plot(rows: list[dict[str, object]], path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 21,
            "axes.labelsize": 17,
            "legend.fontsize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("white")
    panels = [
        ("lpips", "LPIPS", "lower is better"),
        ("psnr", "PSNR", "higher is better"),
        ("ssim", "SSIM", "higher is better"),
    ]
    colors = {"direct": "#2563eb", "oracle_embedding": "#16a34a"}
    labels = {"direct": "Direct pairing", "oracle_embedding": "Oracle embedding match"}

    for ax, (metric, title, subtitle) in zip(axes.reshape(-1), panels):
        for evaluation in ("direct", "oracle_embedding"):
            xs, means, stds = values_for(rows, metric, evaluation)
            ax.plot(
                xs,
                means,
                marker="o",
                linewidth=3.0,
                markersize=8,
                color=colors[evaluation],
                label=labels[evaluation],
            )
            ax.fill_between(
                xs,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                color=colors[evaluation],
                alpha=0.10,
                linewidth=0,
            )
            for x, y in zip(xs, means):
                label = f"{y:.3f}" if metric != "psnr" else f"{y:.2f}"
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=12)
        ax.set_title(f"{title}\n{subtitle}", pad=10)
        ax.set_xlabel("Keep ratio (%)")
        ax.set_xticks([30, 50, 80])
        ax.grid(True, axis="y", color="#cbd5e1", linewidth=1.0, alpha=0.75)
        ax.grid(True, axis="x", color="#e2e8f0", linewidth=0.8, alpha=0.65)
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
        if metric == "lpips":
            ax.set_ylim(0.30, 0.90)
        elif metric == "psnr":
            ax.set_ylim(9.0, 17.8)
        elif metric == "ssim":
            ax.set_ylim(0.0, 0.72)

    handles, legend_labels = axes.reshape(-1)[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(
        "Single-Client CoRA Metrics by Keep Ratio (ImageNet / VGG16 / n=24 / R=8)",
        fontsize=24,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.018,
        "Selected members: keep80=m007, keep50=m001, keep30=m007. keep100 is excluded because no same-observation result exists yet.",
        ha="center",
        fontsize=13,
        color="#475569",
    )
    fig.tight_layout(rect=(0.035, 0.05, 0.985, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> None:
    rows = collect_rows()
    csv_path = OUTPUT_DIR / "single_client_metric_by_keep_summary.csv"
    png_path = OUTPUT_DIR / "single_client_metric_ratio_graph.png"
    write_csv_summary(rows, csv_path)
    plot(rows, png_path)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
