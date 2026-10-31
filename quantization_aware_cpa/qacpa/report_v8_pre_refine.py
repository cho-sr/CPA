from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-refine diagnostics for M3 and retained source cosines.")
    parser.add_argument("--table-csv", type=Path, default=Path("results/v5_4bit_table_lsrescale_cpaica.csv"))
    parser.add_argument("--recovery-csv", type=Path, default=Path("results/v7_m3_fallback_5batch/recovery_by_batch.csv"))
    parser.add_argument("--spearman-csv", type=Path, default=Path("results/v7_m3_fallback_5batch/spearman_by_batch.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/v8_pre_refine_diagnostics"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = read_csv(args.table_csv)
    by_method_batch = {(row["method"], int(row["batch"])): row for row in table}
    retained_rows = retained_cosine_rows(by_method_batch)
    write_csv(output_dir / "retained_A_vs_A_oracle_cosines.csv", retained_rows)
    retained_summary = summarize_retained(retained_rows)
    write_csv(output_dir / "retained_A_vs_A_oracle_summary.csv", retained_summary)

    recovery_rows = read_csv(args.recovery_csv)
    batch_lpips_rows = batch_lpips_table(recovery_rows, by_method_batch)
    write_csv(output_dir / "batch_lpips_gap_recovery_table.csv", batch_lpips_rows)
    gap_rows = gap_correlation_rows(recovery_rows, batch_lpips_rows=batch_lpips_rows)
    write_csv(output_dir / "lpips_gap_recovery_correlation.csv", gap_rows)

    spearman_rows = read_csv(args.spearman_csv)
    full_n_spearman_rows = [row for row in spearman_rows if row["scope"] in {"all", "batch"}]
    write_csv(output_dir / "full_n_spearman_by_batch.csv", full_n_spearman_rows)

    summary = {
        "retained_summary": retained_summary,
        "lpips_gap_recovery_correlation": gap_rows,
        "outputs": {
            "retained_cosines": str(output_dir / "retained_A_vs_A_oracle_cosines.csv"),
            "retained_summary": str(output_dir / "retained_A_vs_A_oracle_summary.csv"),
            "gap_correlation": str(output_dir / "lpips_gap_recovery_correlation.csv"),
            "batch_lpips_gap_recovery": str(output_dir / "batch_lpips_gap_recovery_table.csv"),
            "full_n_spearman": str(output_dir / "full_n_spearman_by_batch.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    all_summary = next(row for row in retained_summary if row["scope"] == "all")
    print(
        "retained A mean={A:.6f} A_oracle mean={O:.6f} delta={D:.6f} oracle_gt_A={K}/{N}".format(
            A=all_summary["A_cos_mean"],
            O=all_summary["A_oracle_cos_mean"],
            D=all_summary["delta_mean"],
            K=all_summary["oracle_gt_A_count"],
            N=all_summary["n_retained"],
        ),
        flush=True,
    )
    corr = next(row for row in gap_rows if row["scope"] == "batch_only")
    print(
        "gap_vs_recovery pearson={p:.4f} spearman={s:.4f}".format(
            p=corr["pearson_gap_recovery"],
            s=corr["spearman_gap_recovery"],
        ),
        flush=True,
    )


def retained_cosine_rows(by_method_batch: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in sorted(batch for method, batch in by_method_batch if method == "A"):
        A = by_method_batch[("A", batch)]
        oracle = by_method_batch[("A_oracle", batch)]
        A_cos = parse_list(A["source_cos_per_truth"])
        oracle_cos = parse_list(oracle["source_cos_per_truth"])
        retained = [idx for idx, value in enumerate(A_cos) if abs(float(value)) > 0.0]
        for idx in retained:
            rows.append(
                {
                    "batch": int(batch),
                    "global_index": int(batch) * 8 + int(idx),
                    "source_index": int(idx),
                    "A_cosine": float(A_cos[idx]),
                    "A_oracle_cosine": float(oracle_cos[idx]),
                    "oracle_minus_A": float(oracle_cos[idx]) - float(A_cos[idx]),
                    "A_above_0_99": float(A_cos[idx]) > 0.99,
                    "oracle_above_0_99": float(oracle_cos[idx]) > 0.99,
                }
            )
    return rows


def summarize_retained(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.append(summarize_retained_scope(rows, scope="all", batch=""))
    for batch in sorted({int(row["batch"]) for row in rows}):
        batch_rows = [row for row in rows if int(row["batch"]) == batch]
        out.append(summarize_retained_scope(batch_rows, scope="batch", batch=batch))
    return out


def summarize_retained_scope(rows: list[dict[str, Any]], *, scope: str, batch: int | str) -> dict[str, Any]:
    A = np.array([float(row["A_cosine"]) for row in rows], dtype=np.float64)
    O = np.array([float(row["A_oracle_cosine"]) for row in rows], dtype=np.float64)
    D = O - A
    return {
        "scope": scope,
        "batch": batch,
        "n_retained": int(len(rows)),
        "A_cos_mean": float(np.mean(A)),
        "A_cos_min": float(np.min(A)),
        "A_oracle_cos_mean": float(np.mean(O)),
        "A_oracle_cos_min": float(np.min(O)),
        "delta_mean": float(np.mean(D)),
        "delta_min": float(np.min(D)),
        "delta_max": float(np.max(D)),
        "oracle_gt_A_count": int(np.sum(D > 0.0)),
        "A_lt_0_99_count": int(np.sum(A < 0.99)),
        "oracle_lt_0_99_count": int(np.sum(O < 0.99)),
    }


def gap_correlation_rows(rows: list[dict[str, str]], batch_lpips_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    batch_rows = [row for row in rows if row["batch_index"] != "all"]
    gaps = np.array([float(row["fp32_naive_lpips_gap"]) for row in batch_rows], dtype=np.float64)
    recovery = np.array([float(row["recovery_rate"]) for row in batch_rows], dtype=np.float64)
    naive = np.array([float(row["naive_lpips_mean"]) for row in batch_rows], dtype=np.float64)
    cond = np.array([], dtype=np.float64)
    if batch_lpips_rows is not None:
        cond = np.array([float(row["cond_A_true"]) for row in batch_lpips_rows], dtype=np.float64)
    pearson_gap = safe_corr_result(pearsonr, gaps, recovery)
    spearman_gap = safe_corr_result(spearmanr, gaps, recovery)
    pearson_naive = safe_corr_result(pearsonr, naive, recovery)
    spearman_naive = safe_corr_result(spearmanr, naive, recovery)
    pearson_cond = safe_corr_result(pearsonr, cond, recovery)
    spearman_cond = safe_corr_result(spearmanr, cond, recovery)
    return [
        {
            "scope": "batch_only",
            "n_batches": int(len(batch_rows)),
            "pearson_gap_recovery": pearson_gap[0],
            "pearson_gap_recovery_pvalue": pearson_gap[1],
            "spearman_gap_recovery": spearman_gap[0],
            "spearman_gap_recovery_pvalue": spearman_gap[1],
            "pearson_naive_recovery": pearson_naive[0],
            "pearson_naive_recovery_pvalue": pearson_naive[1],
            "spearman_naive_recovery": spearman_naive[0],
            "spearman_naive_recovery_pvalue": spearman_naive[1],
            "pearson_cond_recovery": pearson_cond[0],
            "pearson_cond_recovery_pvalue": pearson_cond[1],
            "spearman_cond_recovery": spearman_cond[0],
            "spearman_cond_recovery_pvalue": spearman_cond[1],
            "gap_values": " ".join(f"{x:.9g}" for x in gaps),
            "recovery_values": " ".join(f"{x:.9g}" for x in recovery),
            "naive_lpips_values": " ".join(f"{x:.9g}" for x in naive),
            "cond_A_true_values": " ".join(f"{x:.9g}" for x in cond) if cond.shape[0] else "",
        }
    ]


def batch_lpips_table(
    recovery_rows: list[dict[str, str]],
    by_method_batch: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in recovery_rows:
        if row["batch_index"] == "all":
            continue
        batch = int(row["batch_index"])
        A_row = by_method_batch.get(("A", batch), {})
        out.append(
            {
                "batch": batch,
                "fp32_lpips_mean": float(row["fp32_lpips_mean"]),
                "naive_lpips_mean": float(row["naive_lpips_mean"]),
                "A_lpips_mean": float(row["A_lpips_mean"]),
                "A_oracle_lpips_mean": float(row["A_oracle_lpips_mean"]),
                "fp32_naive_lpips_gap": float(row["fp32_naive_lpips_gap"]),
                "recovery_rate": float(row["recovery_rate"]),
                "cond_A_true": float(A_row.get("cond_A_true", "nan")),
                "B_eff": int(A_row.get("B_eff", 0)) if A_row else 0,
            }
        )
    return out


def safe_corr_result(fn: Any, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.shape[0] < 2 or np.unique(x).shape[0] < 2 or np.unique(y).shape[0] < 2:
        return float("nan"), float("nan")
    result = fn(x, y)
    return float(result.statistic), float(result.pvalue)


def parse_list(raw: str) -> list[float]:
    value = ast.literal_eval(raw)
    return [float(item) for item in value]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
