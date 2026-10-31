from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


METHODS = ("fp32", "naive", "A", "A_oracle", "A_naive_fallback_remeasured", "A_debias_fallback")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate M3 after replacing A fallback from naive to debias.")
    parser.add_argument(
        "--base-aggregated-csv",
        type=Path,
        default=Path("results/v7_m3_fallback_5batch/per_image_metrics_aggregated.csv"),
    )
    parser.add_argument("--fallback-dir", type=Path, default=Path("results/v10_m3_fallback_fia"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/v10_debias_fallback_report"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = read_csv(args.base_aggregated_csv)
    fallback_rows = load_fallback_rows(args.fallback_dir)
    full_rows = build_full_rows(base_rows, fallback_rows)
    write_csv(output_dir / "per_image_metrics_aggregated.csv", full_rows)
    summary_rows = summarize_all(full_rows)
    write_csv(output_dir / "summary_by_method.csv", summary_rows)
    batch_rows = summarize_batches(full_rows)
    write_csv(output_dir / "summary_by_batch.csv", batch_rows)
    fallback_compare = compare_fallback_sources(full_rows)
    write_csv(output_dir / "fallback_only_comparison.csv", fallback_compare)
    source_quality = read_csv(args.fallback_dir / "fallback_source_quality.csv")
    write_csv(output_dir / "fallback_source_quality.csv", source_quality)

    summary = {
        "summary_by_method": summary_rows,
        "summary_by_batch": batch_rows,
        "fallback_only_comparison": fallback_compare,
        "outputs": {
            "per_image_metrics_aggregated": str(output_dir / "per_image_metrics_aggregated.csv"),
            "summary_by_method": str(output_dir / "summary_by_method.csv"),
            "summary_by_batch": str(output_dir / "summary_by_batch.csv"),
            "fallback_only_comparison": str(output_dir / "fallback_only_comparison.csv"),
            "fallback_source_quality": str(output_dir / "fallback_source_quality.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    debias = row_by_method(summary_rows, "A_debias_fallback")
    old = row_by_method(summary_rows, "A")
    remeasured = row_by_method(summary_rows, "A_naive_fallback_remeasured")
    print(
        "A old={old:.4f}; A naive-fallback remeasured={newold:.4f}; "
        "A debias-fallback={debias:.4f}; recovery={recovery:.3f}".format(
            old=float(old["lpips_mean"]),
            newold=float(remeasured["lpips_mean"]),
            debias=float(debias["lpips_mean"]),
            recovery=float(debias["recovery_vs_fp32_gap"]),
        ),
        flush=True,
    )


def build_full_rows(
    base_rows: list[dict[str, str]],
    fallback_rows: dict[tuple[str, int], dict[str, str]],
) -> list[dict[str, Any]]:
    base_by_key = {(row["method"], int(row["global_index"])): row for row in base_rows}
    global_indices = sorted(int(row["global_index"]) for row in base_rows if row["method"] == "naive")
    out: list[dict[str, Any]] = []
    for method in ("fp32", "naive", "A", "A_oracle"):
        for gid in global_indices:
            row = dict(base_by_key[(method, gid)])
            row["method"] = method
            out.append(convert_row(row))
    for method, fallback_method in (
        ("A_naive_fallback_remeasured", "naive_full"),
        ("A_debias_fallback", "debias_full"),
    ):
        for gid in global_indices:
            base_A = base_by_key[("A", gid)]
            if base_A.get("A_source") == "stage2_retained":
                row = dict(base_A)
                row["A_source"] = "stage2_retained"
            else:
                row = dict(fallback_rows[(fallback_method, gid)])
                row["A_source"] = f"{fallback_method}_fallback"
            row["method"] = method
            out.append(convert_row(row))
    return out


def summarize_all(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [summary_row([row for row in rows if row["method"] == method], method=method, scope="all", batch="") for method in METHODS]
    means = {row["method"]: float(row["lpips_mean"]) for row in out}
    gap = means["naive"] - means["fp32"]
    for row in out:
        method = row["method"]
        if method in {"A", "A_naive_fallback_remeasured", "A_debias_fallback", "A_oracle"}:
            row["gain_vs_naive"] = means["naive"] - means[method]
            row["recovery_vs_fp32_gap"] = safe_div(means["naive"] - means[method], gap)
        else:
            row["gain_vs_naive"] = ""
            row["recovery_vs_fp32_gap"] = ""
    return out


def summarize_batches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for batch in sorted({int(row["batch_index"]) for row in rows}):
        batch_rows = [row for row in rows if int(row["batch_index"]) == batch]
        means = {
            method: float(np.mean([float(row["lpips"]) for row in batch_rows if row["method"] == method]))
            for method in METHODS
        }
        gap = means["naive"] - means["fp32"]
        row: dict[str, Any] = {"batch": batch, "fp32_naive_gap": gap}
        for method in METHODS:
            row[f"{method}_lpips_mean"] = means[method]
        for method in ("A", "A_naive_fallback_remeasured", "A_debias_fallback", "A_oracle"):
            row[f"{method}_gain_vs_naive"] = means["naive"] - means[method]
            row[f"{method}_recovery_vs_fp32_gap"] = safe_div(means["naive"] - means[method], gap)
        out.append(row)
    return out


def compare_fallback_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fallback_gids = sorted(
        {
            int(row["global_index"])
            for row in rows
            if row["method"] == "A_debias_fallback" and row["A_source"] == "debias_full_fallback"
        }
    )
    out: list[dict[str, Any]] = []
    for gid in fallback_gids:
        by_method = {row["method"]: row for row in rows if int(row["global_index"]) == gid}
        out.append(
            {
                "global_index": gid,
                "batch": int(by_method["naive"]["batch_index"]),
                "source_index": int(by_method["naive"]["source_index"]),
                "per_sample_loss": float(by_method["naive"]["per_sample_loss"]),
                "naive_full_lpips": float(by_method["A_naive_fallback_remeasured"]["lpips"]),
                "debias_full_lpips": float(by_method["A_debias_fallback"]["lpips"]),
                "A_oracle_lpips": float(by_method["A_oracle"]["lpips"]),
                "debias_minus_naive_lpips": float(by_method["A_debias_fallback"]["lpips"])
                - float(by_method["A_naive_fallback_remeasured"]["lpips"]),
                "oracle_minus_debias_lpips": float(by_method["A_oracle"]["lpips"])
                - float(by_method["A_debias_fallback"]["lpips"]),
                "naive_full_source_cosine": float(by_method["A_naive_fallback_remeasured"]["target_source_cosine"]),
                "debias_full_source_cosine": float(by_method["A_debias_fallback"]["target_source_cosine"]),
                "A_oracle_source_cosine": float(by_method["A_oracle"]["target_source_cosine"]),
            }
        )
    return out


def summary_row(rows: list[dict[str, Any]], *, method: str, scope: str, batch: int | str) -> dict[str, Any]:
    lpips = np.asarray([float(row["lpips"]) for row in rows], dtype=np.float64)
    psnr = np.asarray([float(row["psnr"]) for row in rows], dtype=np.float64)
    ssim = np.asarray([float(row["ssim"]) for row in rows], dtype=np.float64)
    losses = np.asarray([float(row["per_sample_loss"]) for row in rows], dtype=np.float64)
    rho, pvalue = safe_spearman(losses, lpips)
    return {
        "scope": scope,
        "batch": batch,
        "method": method,
        "n_images": len(rows),
        "lpips_mean": float(np.mean(lpips)),
        "psnr_mean": float(np.mean(psnr)),
        "ssim_mean": float(np.mean(ssim)),
        "spearman_loss_lpips": rho,
        "spearman_pvalue": pvalue,
        "fallback_count": int(sum(str(row.get("A_source", "")).endswith("_fallback") for row in rows)),
    }


def load_fallback_rows(root: Path) -> dict[tuple[str, int], dict[str, str]]:
    out: dict[tuple[str, int], dict[str, str]] = {}
    for method in ("naive_full", "debias_full"):
        path = Path(root) / method / "per_image_metrics.csv"
        for row in read_csv(path):
            out[(method, int(row["global_index"]))] = row
    return out


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("global_index", "batch_index", "source_index", "label"):
        out[key] = int(out[key])
    for key in ("per_sample_loss", "target_source_cosine", "psnr", "ssim", "lpips"):
        out[key] = float(out[key])
    return out


def row_by_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    for row in rows:
        if row["method"] == method:
            return row
    raise RuntimeError(f"missing method {method}")


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.shape[0] < 2 or np.unique(x).shape[0] < 2 or np.unique(y).shape[0] < 2:
        return float("nan"), float("nan")
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(float(den)) > 1.0e-12 else float("nan")


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
