from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare current paper figures and Milestone-4 figure schema.")
    parser.add_argument("--cmp-sweep-csv", type=Path, default=Path("results/v4_cmp_sweep_cpaica.csv"))
    parser.add_argument(
        "--m3-aggregated-csv",
        type=Path,
        default=Path("results/v7_m3_fallback_5batch/per_image_metrics_aggregated.csv"),
    )
    parser.add_argument(
        "--spearman-csv",
        type=Path,
        default=Path("results/v7_m3_fallback_5batch/spearman_by_batch.csv"),
    )
    parser.add_argument(
        "--image-grid",
        type=Path,
        default=Path("results/v9_stage2_saturation_diagnostics/figure_original_fp32_4bit_naive_4bit_A_fulln.png"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/v9_paper_figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_lambda_vs_cosine(args.cmp_sweep_csv, output_dir / "lambda_min_over_mp_edge_vs_worst_cosine.png")
    plot_loss_vs_lpips(
        args.m3_aggregated_csv,
        args.spearman_csv,
        output_dir / "loss_vs_lpips_scatter_4conditions_fulln.png",
    )
    write_heatmap_schema(output_dir / "m4_heatmap_input_schema.csv")
    if args.image_grid.is_file():
        shutil.copyfile(args.image_grid, output_dir / args.image_grid.name)
    print(f"prepared paper figure assets in {output_dir}", flush=True)


def plot_lambda_vs_cosine(input_csv: Path, output_path: Path) -> None:
    rows = read_csv(input_csv)
    colors = {1.0: "#4C78A8", 1.5: "#F58518", 3.0: "#54A24B", 6.0: "#B279A2", 12.0: "#E45756"}
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    for c_mp in sorted({float(row["c_mp"]) for row in rows}):
        group = [row for row in rows if float(row["c_mp"]) == c_mp]
        ax.scatter(
            [float(row["min_retained_lam_over_mp_edge"]) for row in group],
            [float(row["worst_source_cosine"]) for row in group],
            s=46,
            label=f"c_mp={c_mp:g}",
            color=colors.get(c_mp, "#777777"),
            edgecolors="white",
            linewidths=0.6,
            alpha=0.85,
        )
    ax.axvline(3.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.set_xscale("log")
    ax.set_xlabel("min retained lambda / MP edge")
    ax.set_ylabel("worst source cosine")
    ax.set_title("MP margin vs retained-source quality")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_loss_vs_lpips(metrics_csv: Path, spearman_csv: Path, output_path: Path) -> None:
    rows = read_csv(metrics_csv)
    spearman = {
        row["method"]: row
        for row in read_csv(spearman_csv)
        if row.get("scope") == "all"
    }
    colors = {
        "fp32": "#4C78A8",
        "naive": "#F58518",
        "A": "#54A24B",
        "A_oracle": "#B279A2",
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=160)
    for method in ("fp32", "naive", "A", "A_oracle"):
        group = [row for row in rows if row["method"] == method]
        stat = spearman.get(method, {})
        display_method = "CPA (unmodified) + naive dequant" if method == "naive" else method
        label = display_method
        if stat:
            label = (
                f"{display_method} rho={float(stat['spearman_loss_lpips']):.2f} "
                f"[{float(stat['spearman_ci_low']):.2f},{float(stat['spearman_ci_high']):.2f}]"
            )
        ax.scatter(
            [float(row["per_sample_loss"]) for row in group],
            [float(row["lpips"]) for row in group],
            s=28,
            alpha=0.78,
            color=colors[method],
            edgecolors="white",
            linewidths=0.5,
            label=label,
        )
    ax.set_xlabel("per-sample loss")
    ax.set_ylabel("per-image LPIPS")
    ax.set_title("Leakage bias after quantization")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_heatmap_schema(path: Path) -> None:
    rows = [
        {
            "B": "",
            "bits": "",
            "rounding": "",
            "seed": "",
            "scale_granularity": "",
            "method": "",
            "lpips_mean": "",
            "spearman_loss_lpips": "",
            "B_eff_mean": "",
            "mp_edge_mean": "",
            "mp_predicted_B_eff": "",
        }
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_list(raw: str) -> list[float]:
    return [float(item) for item in ast.literal_eval(raw)]


if __name__ == "__main__":
    main()
