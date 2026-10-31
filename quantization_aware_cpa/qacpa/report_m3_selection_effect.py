from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check M3 Spearman selection effects on common global indices.")
    parser.add_argument("--m3-dirs", required=True, help="Comma-separated result directories containing method CSVs.")
    parser.add_argument("--selector-method", default="A")
    parser.add_argument("--methods", default="fp32,naive,A")
    parser.add_argument("--output-dir", type=Path, default=Path("results/v7_m3_selection_effect"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    m3_dirs = [Path(item.strip()) for item in args.m3_dirs.split(",") if item.strip()]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    rows = load_rows(m3_dirs=m3_dirs, methods=sorted(set(methods + [args.selector_method])))
    selector_gids = sorted(gid for (method, gid) in rows if method == args.selector_method)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for method in methods:
        available = [gid for gid in selector_gids if (method, gid) in rows]
        method_rows = [rows[(method, gid)] for gid in available]
        summary = summarize(method_rows, method=method, selector=args.selector_method, requested_gids=selector_gids)
        summary_rows.append(summary)
        for row in method_rows:
            out = dict(row)
            out["selection_selector"] = args.selector_method
            selected_rows.append(out)

    write_csv(output_dir / "selection_effect_summary.csv", summary_rows)
    write_csv(output_dir / "selection_effect_rows.csv", selected_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "m3_dirs": [str(path) for path in m3_dirs],
                "selector_method": args.selector_method,
                "selector_global_indices": selector_gids,
                "methods": methods,
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for row in summary_rows:
        print(
            "{method}@{n_available}/{n_requested}: lpips={lpips:.4f} spearman={rho:.4f}".format(
                method=row["method"],
                n_available=row["n_available"],
                n_requested=row["n_requested"],
                lpips=row["lpips_mean"],
                rho=row["spearman_loss_lpips"],
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


def summarize(
    rows: list[dict[str, Any]],
    *,
    method: str,
    selector: str,
    requested_gids: list[int],
) -> dict[str, Any]:
    losses = np.array([float(row["per_sample_loss"]) for row in rows], dtype=np.float64)
    lpips = np.array([float(row["lpips"]) for row in rows], dtype=np.float64)
    rho, pvalue = safe_spearman(losses, lpips)
    available_gids = sorted(int(row["global_index"]) for row in rows)
    missing_gids = [gid for gid in requested_gids if gid not in set(available_gids)]
    return {
        "selector_method": selector,
        "method": method,
        "n_requested": int(len(requested_gids)),
        "n_available": int(len(rows)),
        "missing_global_indices": " ".join(str(gid) for gid in missing_gids),
        "lpips_mean": float(np.mean(lpips)) if len(lpips) else float("nan"),
        "psnr_mean": float(np.mean([float(row["psnr"]) for row in rows])) if rows else float("nan"),
        "ssim_mean": float(np.mean([float(row["ssim"]) for row in rows])) if rows else float("nan"),
        "spearman_loss_lpips": rho,
        "spearman_pvalue": pvalue,
    }


def safe_spearman(losses: np.ndarray, lpips: np.ndarray) -> tuple[float, float]:
    if losses.shape[0] < 2 or np.unique(losses).shape[0] < 2 or np.unique(lpips).shape[0] < 2:
        return float("nan"), float("nan")
    result = spearmanr(losses, lpips)
    return float(result.statistic), float(result.pvalue)


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
