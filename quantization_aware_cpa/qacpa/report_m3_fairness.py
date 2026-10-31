from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report fair M3 comparisons on common image sets.")
    parser.add_argument("--m3-dir", type=Path, required=True)
    parser.add_argument("--methods", default="fp32,naive,A,A_oracle")
    parser.add_argument("--common-mode", choices=["intersection", "reference"], default="intersection")
    parser.add_argument("--reference-method", default="A")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    rows_by_method = {method: read_metric_rows(args.m3_dir / method / "per_image_metrics.csv") for method in methods}
    by_gid = {
        method: {int(row["global_index"]): row for row in rows}
        for method, rows in rows_by_method.items()
    }
    if args.common_mode == "reference":
        common = set(by_gid[str(args.reference_method)])
        for method in methods:
            common &= set(by_gid[method])
    else:
        common = set.intersection(*(set(by_gid[method]) for method in methods))
    common_indices = sorted(common)
    if not common_indices:
        raise RuntimeError("no common global_index values across requested methods")

    summary_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [by_gid[method][gid] for gid in common_indices]
        summary_rows.append(summarize_rows(method, method_rows, batch_index=None))
        for batch_index in sorted({int(row["batch_index"]) for row in method_rows}):
            subset = [row for row in method_rows if int(row["batch_index"]) == batch_index]
            batch_rows.append(summarize_rows(method, subset, batch_index=batch_index))

    output = {
        "m3_dir": str(args.m3_dir),
        "methods": methods,
        "common_mode": str(args.common_mode),
        "reference_method": str(args.reference_method),
        "common_global_indices": common_indices,
        "n_common": len(common_indices),
        "summary": summary_rows,
        "by_batch": batch_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
        writer.writerows(batch_rows)

    print(f"common n={len(common_indices)} indices={common_indices}")
    for row in summary_rows:
        print(
            "{method}: lpips={lpips_mean:.4f} psnr={psnr_mean:.3f} ssim={ssim_mean:.3f} "
            "spearman={spearman_loss_lpips:.3f}".format(**row),
            flush=True,
        )


def read_metric_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_rows(method: str, rows: list[dict[str, str]], *, batch_index: int | None) -> dict[str, Any]:
    losses = [float(row["per_sample_loss"]) for row in rows]
    lpips = [float(row["lpips"]) for row in rows]
    psnr = [float(row["psnr"]) for row in rows]
    ssim = [float(row["ssim"]) for row in rows]
    target_cos = [float(row["target_source_cosine"]) for row in rows]
    rho, pvalue = safe_spearman(losses, lpips)
    return {
        "scope": "all_common" if batch_index is None else "batch",
        "batch_index": "" if batch_index is None else int(batch_index),
        "method": method,
        "n_images": len(rows),
        "lpips_mean": float(np.mean(lpips)),
        "lpips_median": float(np.median(lpips)),
        "psnr_mean": float(np.mean(psnr)),
        "ssim_mean": float(np.mean(ssim)),
        "target_source_cosine_mean": float(np.mean(target_cos)),
        "spearman_loss_lpips": rho,
        "spearman_loss_lpips_pvalue": pvalue,
    }


def safe_spearman(losses: list[float], lpips: list[float]) -> tuple[float, float]:
    if len(losses) < 2 or len(set(losses)) < 2 or len(set(lpips)) < 2:
        return float("nan"), float("nan")
    result = spearmanr(losses, lpips)
    return float(result.statistic), float(result.pvalue)


if __name__ == "__main__":
    main()
