from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize upstream CPA/GMA attack pickle outputs.")
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for root in args.roots:
        rows.extend(summarize_root(Path(root)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(args.output)


def summarize_root(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*_summary.pkl")):
        if summary_path.name.endswith("_fi_summary.pkl"):
            continue
        batch_size = summary_path.name.split("_", 1)[0]
        df = pd.read_pickle(summary_path)
        rows.append(summarize_df(root, summary_path, df, batch_size, stage="gi"))
        fi_path = root / f"{batch_size}_fi_summary.pkl"
        if fi_path.is_file():
            fi_df = pd.read_pickle(fi_path)
            rows.append(summarize_df(root, fi_path, fi_df, batch_size, stage="fi"))
    return rows


def summarize_df(root: Path, path: Path, df: pd.DataFrame, batch_size: str, stage: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "exp_path": str(root),
        "file": str(path),
        "batch_size": int(batch_size),
        "stage": stage,
        "n_rows": int(len(df)),
    }
    if "batch" in df.columns:
        row["n_batches"] = int(df["batch"].nunique())
        row["batches"] = ",".join(str(int(x)) for x in sorted(df["batch"].unique()))
    for metric in ("cs", "lpips", "psnr", "ssim"):
        if metric in df.columns and len(df[metric]) > 0:
            values = pd.Series([to_float(value) for value in df[metric]])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
    return row


def to_float(value: Any) -> float:
    arr = np.asarray(value)
    if arr.shape == ():
        return float(arr)
    return float(np.mean(arr.astype(np.float64)))


if __name__ == "__main__":
    main()
