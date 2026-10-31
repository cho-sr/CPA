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
    parser = argparse.ArgumentParser(description="Build a full-n M3 report with A falling back to naive rows.")
    parser.add_argument("--m3-dirs", required=True, help="Comma-separated result directories containing method CSVs.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/v7_m3_fallback_5batch"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    m3_dirs = [Path(item.strip()) for item in args.m3_dirs.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_key = load_rows(m3_dirs=m3_dirs, methods=["fp32", "naive", "A", "A_oracle"])
    global_indices = sorted(gid for (method, gid) in rows_by_key if method == "naive")
    if not global_indices:
        raise RuntimeError("no naive rows found; cannot define the full evaluation set")

    full_rows: list[dict[str, Any]] = []
    for method in ("fp32", "naive", "A", "A_oracle"):
        method_rows = build_method_rows(method=method, global_indices=global_indices, rows_by_key=rows_by_key)
        full_rows.extend(method_rows)
        write_csv(output_dir / method / "per_image_metrics.csv", method_rows)

    write_csv(output_dir / "per_image_metrics_aggregated.csv", full_rows)

    rng = np.random.default_rng(int(args.seed))
    summary_rows: list[dict[str, Any]] = []
    for method in ("fp32", "naive", "A", "A_oracle"):
        method_rows = [row for row in full_rows if row["method"] == method]
        summary_rows.append(summarize(method_rows, method=method, batch_index=None, rng=rng, n_boot=int(args.bootstrap)))
        for batch_index in sorted({int(row["batch_index"]) for row in method_rows}):
            batch_rows = [row for row in method_rows if int(row["batch_index"]) == batch_index]
            summary_rows.append(
                summarize(batch_rows, method=method, batch_index=batch_index, rng=rng, n_boot=int(args.bootstrap))
            )
    write_csv(output_dir / "spearman_by_batch.csv", summary_rows)

    recovery_rows = build_recovery_rows(full_rows)
    write_csv(output_dir / "recovery_by_batch.csv", recovery_rows)
    make_scatter(full_rows, output_dir / "loss_vs_lpips_scatter_overlay.png")

    summary = {
        "m3_dirs": [str(path) for path in m3_dirs],
        "global_indices": global_indices,
        "summary": summary_rows,
        "recovery_by_batch": recovery_rows,
        "A_retained_global_indices": [
            int(row["global_index"])
            for row in full_rows
            if row["method"] == "A" and row.get("A_source") == "stage2_retained"
        ],
        "A_fallback_global_indices": [
            int(row["global_index"])
            for row in full_rows
            if row["method"] == "A" and row.get("A_source") == "naive_fallback"
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for row in [item for item in summary_rows if item["scope"] == "all"]:
        print(
            "{method}: n={n} lpips={lpips:.4f} spearman={rho:.4f}".format(
                method=row["method"],
                n=row["n_images"],
                lpips=row["lpips_mean"],
                rho=row["spearman_loss_lpips"],
            ),
            flush=True,
        )
    for row in recovery_rows:
        print(
            "batch={batch} gap={gap:.4f} recovery={recovery:.3f}".format(
                batch=row["batch_index"],
                gap=row["fp32_naive_lpips_gap"],
                recovery=row["recovery_rate"],
            ),
            flush=True,
        )


def load_rows(*, m3_dirs: list[Path], methods: list[str]) -> dict[tuple[str, int], dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for root in m3_dirs:
        for method in methods:
            path = root / method / "per_image_metrics.csv"
            if not path.is_file():
                continue
            with path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    converted = convert_row(row, result_dir=root)
                    by_key[(method, int(converted["global_index"]))] = converted
    return by_key


def convert_row(row: dict[str, str], *, result_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)
    for key in ("global_index", "batch_index", "source_index", "label"):
        out[key] = int(out[key])
    for key in ("per_sample_loss", "target_source_cosine", "psnr", "ssim", "lpips"):
        out[key] = float(out[key])
    out["result_dir"] = str(result_dir)
    return out


def build_method_rows(
    *,
    method: str,
    global_indices: list[int],
    rows_by_key: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    for gid in global_indices:
        if method == "A":
            if ("A", gid) in rows_by_key:
                row = dict(rows_by_key[("A", gid)])
                row["A_source"] = "stage2_retained"
            else:
                row = dict(rows_by_key[("naive", gid)])
                row["A_source"] = "naive_fallback"
            row["method"] = "A"
            rows.append(row)
            continue
        key = (method, gid)
        if key not in rows_by_key:
            missing.append(gid)
            continue
        row = dict(rows_by_key[key])
        row["method"] = method
        rows.append(row)
    if missing:
        raise RuntimeError(f"{method} is missing global indices {missing}")
    return rows


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


def build_recovery_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [recovery_row(rows, batch_index="all")]
    for batch_index in sorted({int(row["batch_index"]) for row in rows}):
        batch_rows = [row for row in rows if int(row["batch_index"]) == batch_index]
        out.append(recovery_row(batch_rows, batch_index=int(batch_index)))
    return out


def recovery_row(rows: list[dict[str, Any]], *, batch_index: int | str) -> dict[str, Any]:
    by_method = {
        method: [row for row in rows if row["method"] == method]
        for method in ("fp32", "naive", "A", "A_oracle")
    }
    means = {method: float(np.mean([float(row["lpips"]) for row in method_rows])) for method, method_rows in by_method.items()}
    gap = means["naive"] - means["fp32"]
    gain = means["naive"] - means["A"]
    recovery = gain / gap if abs(gap) > 1.0e-12 else float("nan")
    return {
        "batch_index": batch_index,
        "fp32_lpips_mean": means["fp32"],
        "naive_lpips_mean": means["naive"],
        "A_lpips_mean": means["A"],
        "A_oracle_lpips_mean": means["A_oracle"],
        "fp32_naive_lpips_gap": gap,
        "naive_A_lpips_gain": gain,
        "recovery_rate": recovery,
    }


def safe_spearman(losses: np.ndarray, lpips: np.ndarray) -> tuple[float, float]:
    if losses.shape[0] < 2 or np.unique(losses).shape[0] < 2 or np.unique(lpips).shape[0] < 2:
        return float("nan"), float("nan")
    result = spearmanr(losses, lpips)
    return float(result.statistic), float(result.pvalue)


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


def make_scatter(rows: list[dict[str, Any]], output_path: Path) -> None:
    colors = {
        "fp32": "#4C78A8",
        "naive": "#F58518",
        "A": "#54A24B",
        "A_oracle": "#B279A2",
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=160)
    for method in ("fp32", "naive", "A", "A_oracle"):
        method_rows = [row for row in rows if row["method"] == method]
        x = [float(row["per_sample_loss"]) for row in method_rows]
        y = [float(row["lpips"]) for row in method_rows]
        ax.scatter(
            x,
            y,
            label=f"{'CPA (unmodified) + naive dequant' if method == 'naive' else method} (n={len(method_rows)})",
            s=28,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.5,
            color=colors[method],
        )
    ax.set_xlabel("per-sample loss")
    ax.set_ylabel("per-image LPIPS")
    ax.set_title("M3 full-n per-image LPIPS vs per-sample loss")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
