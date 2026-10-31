from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate M3 per-image metrics and report Spearman statistics.")
    parser.add_argument("--m3-dirs", required=True, help="Comma-separated result directories containing method CSVs.")
    parser.add_argument("--methods", default="fp32,naive,A,A_oracle")
    parser.add_argument("--output-dir", type=Path, default=Path("results/v6_m3_spearman"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    m3_dirs = [Path(item.strip()) for item in args.m3_dirs.split(",") if item.strip()]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_all_rows(m3_dirs=m3_dirs, methods=methods)
    write_rows(output_dir / "per_image_metrics_aggregated.csv", rows)

    rng = np.random.default_rng(int(args.seed))
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        summary_rows.append(summarize(method_rows, method=method, batch_index=None, rng=rng, n_boot=int(args.bootstrap)))
        for batch_index in sorted({int(row["batch_index"]) for row in method_rows}):
            batch_rows = [row for row in method_rows if int(row["batch_index"]) == batch_index]
            summary_rows.append(
                summarize(batch_rows, method=method, batch_index=batch_index, rng=rng, n_boot=int(args.bootstrap))
            )
    write_summary_csv(output_dir / "spearman_by_batch.csv", summary_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "m3_dirs": [str(path) for path in m3_dirs],
                "methods": methods,
                "bootstrap": int(args.bootstrap),
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    make_scatter(rows, methods=methods, output_path=output_dir / "loss_vs_lpips_scatter_overlay.png")
    print(f"wrote {output_dir / 'spearman_by_batch.csv'}")
    print(f"wrote {output_dir / 'loss_vs_lpips_scatter_overlay.png'}")


def load_all_rows(*, m3_dirs: list[Path], methods: list[str]) -> list[dict[str, Any]]:
    by_method_gid: dict[tuple[str, int], dict[str, Any]] = {}
    for root in m3_dirs:
        for method in methods:
            path = root / method / "per_image_metrics.csv"
            if not path.is_file():
                continue
            with path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    converted = convert_row(row, result_dir=root)
                    by_method_gid[(method, int(converted["global_index"]))] = converted
    return [by_method_gid[key] for key in sorted(by_method_gid, key=lambda item: (item[0], item[1]))]


def convert_row(row: dict[str, str], *, result_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)
    for key in ("global_index", "batch_index", "source_index", "label"):
        out[key] = int(out[key])
    for key in ("per_sample_loss", "target_source_cosine", "psnr", "ssim", "lpips"):
        out[key] = float(out[key])
    out["result_dir"] = str(result_dir)
    return out


def summarize(
    rows: list[dict[str, Any]],
    *,
    method: str,
    batch_index: int | None,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, Any]:
    losses = np.array([float(row["per_sample_loss"]) for row in rows], dtype=np.float64)
    lpips = np.array([float(row["lpips"]) for row in rows], dtype=np.float64)
    psnr = np.array([float(row["psnr"]) for row in rows], dtype=np.float64)
    ssim = np.array([float(row["ssim"]) for row in rows], dtype=np.float64)
    rho, pvalue = safe_spearman(losses, lpips)
    ci_low, ci_high = bootstrap_ci(losses, lpips, rng=rng, n_boot=n_boot)
    return {
        "scope": "all" if batch_index is None else "batch",
        "batch_index": "" if batch_index is None else int(batch_index),
        "method": method,
        "n_images": int(len(rows)),
        "lpips_mean": float(np.mean(lpips)) if len(lpips) else float("nan"),
        "lpips_median": float(np.median(lpips)) if len(lpips) else float("nan"),
        "psnr_mean": float(np.mean(psnr)) if len(psnr) else float("nan"),
        "ssim_mean": float(np.mean(ssim)) if len(ssim) else float("nan"),
        "spearman_loss_lpips": rho,
        "spearman_pvalue": pvalue,
        "spearman_ci_low": ci_low,
        "spearman_ci_high": ci_high,
    }


def bootstrap_ci(
    losses: np.ndarray,
    lpips: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    n = int(losses.shape[0])
    if n < 3:
        return float("nan"), float("nan")
    values: list[float] = []
    for _ in range(int(n_boot)):
        indices = rng.integers(0, n, size=n)
        rho, _ = safe_spearman(losses[indices], lpips[indices])
        if np.isfinite(rho):
            values.append(float(rho))
    if not values:
        return float("nan"), float("nan")
    lo, hi = np.percentile(np.array(values, dtype=np.float64), [2.5, 97.5])
    return float(lo), float(hi)


def safe_spearman(losses: np.ndarray, lpips: np.ndarray) -> tuple[float, float]:
    if losses.shape[0] < 2 or np.unique(losses).shape[0] < 2 or np.unique(lpips).shape[0] < 2:
        return float("nan"), float("nan")
    result = spearmanr(losses, lpips)
    return float(result.statistic), float(result.pvalue)


def make_scatter(rows: list[dict[str, Any]], *, methods: list[str], output_path: Path) -> None:
    colors = {
        "fp32": "#4C78A8",
        "naive": "#F58518",
        "A": "#54A24B",
        "A_oracle": "#B279A2",
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=160)
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        x = [float(row["per_sample_loss"]) for row in method_rows]
        y = [float(row["lpips"]) for row in method_rows]
        ax.scatter(
            x,
            y,
            label=f"{method} (n={len(method_rows)})",
            s=28,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.5,
            color=colors.get(method),
        )
    ax.set_xlabel("per-sample loss")
    ax.set_ylabel("per-image LPIPS")
    ax.set_title("M3 per-image LPIPS vs per-sample loss")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
