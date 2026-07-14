#!/usr/bin/env python3
"""Compare FP32 and quantized-update CPA result directories."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METRICS = ("lpips", "psnr", "ssim", "cs")


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _result_file(result_dir: Path, n_samples: int, suffix: str) -> Path:
    path = result_dir / f"{n_samples}_{suffix}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return path


def _summarize_frame(path: Path, stage: str, run_name: str) -> list[dict]:
    df = pd.read_pickle(path)
    rows = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "run": run_name,
                "stage": stage,
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
                "n": int(values.shape[0]),
            }
        )
    return rows


def _iter_rec_pairs(rec_dict: dict) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for inp, rec in zip(rec_dict["inp"], rec_dict["rec"]):
        inp_arr = np.asarray(inp, dtype=np.float32)
        rec_arr = np.asarray(rec, dtype=np.float32)
        n = min(inp_arr.shape[0], rec_arr.shape[0])
        for idx in range(n):
            yield inp_arr[idx].reshape(-1), rec_arr[idx].reshape(-1)


def _summarize_mse(path: Path, run_name: str) -> dict:
    rec_dict = _load_pickle(path)
    values = np.asarray(
        [np.mean((inp - rec) ** 2) for inp, rec in _iter_rec_pairs(rec_dict)],
        dtype=np.float64,
    )
    if values.size == 0:
        raise ValueError(f"No reconstruction pairs found in {path}")
    return {
        "run": run_name,
        "stage": "fia",
        "metric": "mse",
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "n": int(values.size),
    }


def _to_markdown(table: pd.DataFrame) -> str:
    headers = list(table.columns)
    rows = [headers] + table.astype(str).values.tolist()
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(headers))]

    def fmt(row: list[str]) -> str:
        cells = [row[idx].ljust(widths[idx]) for idx in range(len(headers))]
        return "| " + " | ".join(cells) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt(headers), sep] + [fmt(row) for row in rows[1:]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32_dir", required=True, type=Path)
    parser.add_argument("--int4_dir", required=True, type=Path)
    parser.add_argument("--n_samples", required=True, type=int)
    parser.add_argument(
        "--out_prefix",
        default="quantized_update_cpa/baseline_compare",
        type=Path,
        help="Output prefix for .csv and .md files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = (("fp32", args.fp32_dir), ("int4", args.int4_dir))

    rows = []
    for run_name, result_dir in runs:
        summary_file = _result_file(result_dir, args.n_samples, "summary")
        rows.extend(_summarize_frame(summary_file, "ica", run_name))

        fi_summary_file = _result_file(result_dir, args.n_samples, "fi_summary")
        rows.extend(_summarize_frame(fi_summary_file, "fia", run_name))

        rec_file = _result_file(result_dir, args.n_samples, "rec")
        rows.append(_summarize_mse(rec_file, run_name))

    table = pd.DataFrame(rows).sort_values(["stage", "metric", "run"])
    csv_path = args.out_prefix.with_suffix(".csv")
    md_path = args.out_prefix.with_suffix(".md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    md_path.write_text(_to_markdown(table) + "\n")

    print(_to_markdown(table))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
