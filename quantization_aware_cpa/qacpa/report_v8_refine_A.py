from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize v8 refine-A retained source cosines.")
    parser.add_argument("--runs-jsonl", type=Path, default=Path("results/runs.jsonl"))
    parser.add_argument("--table-csv", type=Path, default=Path("results/v5_4bit_table_lsrescale_cpaica.csv"))
    parser.add_argument("--run-tag", default="v8_refine_A")
    parser.add_argument("--output-dir", type=Path, default=Path("results/v8_refine_A"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = read_csv(args.table_csv)
    baseline = {(row["method"], int(row["batch"])): row for row in table}
    refine_rows_by_batch = load_refine_rows(args.runs_jsonl, run_tag=str(args.run_tag))

    rows: list[dict[str, Any]] = []
    step_logs: list[dict[str, Any]] = []
    for batch in sorted(refine_rows_by_batch):
        row = refine_rows_by_batch[batch]
        A_base = baseline[("A", batch)]
        oracle = baseline[("A_oracle", batch)]
        A_cos = parse_list(A_base["source_cos_per_truth"])
        oracle_cos = parse_list(oracle["source_cos_per_truth"])
        refined_cos = [float(x) for x in row["metrics"]["source_cos_per_truth"]]
        retained = [idx for idx, value in enumerate(A_cos) if abs(float(value)) > 0.0]
        for idx in retained:
            rows.append(
                {
                    "batch": int(batch),
                    "global_index": int(batch) * 8 + int(idx),
                    "source_index": int(idx),
                    "A_cosine": float(A_cos[idx]),
                    "A_oracle_cosine": float(oracle_cos[idx]),
                    "refine_A_cosine": float(refined_cos[idx]),
                    "oracle_minus_A": float(oracle_cos[idx]) - float(A_cos[idx]),
                    "refine_minus_A": float(refined_cos[idx]) - float(A_cos[idx]),
                    "refine_minus_oracle": float(refined_cos[idx]) - float(oracle_cos[idx]),
                }
            )
        for log in row.get("refine_A", {}).get("a_step_logs", []):
            out = dict(log)
            out["batch"] = int(batch)
            step_logs.append(out)

    summary_rows = summarize(rows, refine_rows_by_batch)
    write_csv(output_dir / "retained_A_vs_A_oracle_vs_refine_A_cosines.csv", rows)
    write_csv(output_dir / "refine_A_summary.csv", summary_rows)
    write_csv(output_dir / "refine_A_step_logs.csv", step_logs)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_tag": str(args.run_tag),
                "summary": summary_rows,
                "outputs": {
                    "cosines": str(output_dir / "retained_A_vs_A_oracle_vs_refine_A_cosines.csv"),
                    "summary": str(output_dir / "refine_A_summary.csv"),
                    "step_logs": str(output_dir / "refine_A_step_logs.csv"),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    all_row = next(row for row in summary_rows if row["scope"] == "all")
    print(
        "refine retained mean={R:.6f}; baseline A={A:.6f}; oracle={O:.6f}; refine_minus_A={D:.6f}".format(
            R=all_row["refine_A_cos_mean"],
            A=all_row["A_cos_mean"],
            O=all_row["A_oracle_cos_mean"],
            D=all_row["refine_minus_A_mean"],
        ),
        flush=True,
    )
    print(
        "refine worsened {worse}/{n} retained sources".format(
            worse=all_row["refine_lt_A_count"],
            n=all_row["n_retained"],
        ),
        flush=True,
    )


def load_refine_rows(path: Path, *, run_tag: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run_tag") != run_tag:
                continue
            if row.get("milestone") != "stage2_A":
                continue
            rows[int(row["batch_index"])] = row
    if not rows:
        raise RuntimeError(f"no rows found for run_tag={run_tag!r}")
    return rows


def summarize(rows: list[dict[str, Any]], refine_rows_by_batch: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    out = [summarize_scope(rows, scope="all", batch="", refine_rows_by_batch=refine_rows_by_batch)]
    for batch in sorted({int(row["batch"]) for row in rows}):
        batch_rows = [row for row in rows if int(row["batch"]) == batch]
        out.append(summarize_scope(batch_rows, scope="batch", batch=batch, refine_rows_by_batch=refine_rows_by_batch))
    return out


def summarize_scope(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    batch: int | str,
    refine_rows_by_batch: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    A = np.array([float(row["A_cosine"]) for row in rows], dtype=np.float64)
    O = np.array([float(row["A_oracle_cosine"]) for row in rows], dtype=np.float64)
    R = np.array([float(row["refine_A_cosine"]) for row in rows], dtype=np.float64)
    selected_refine_rows = (
        list(refine_rows_by_batch.values())
        if batch == ""
        else [refine_rows_by_batch[int(batch)]]
    )
    accepted = sum(int(row.get("refine_A", {}).get("n_a_steps_accepted", 0)) for row in selected_refine_rows)
    requested = sum(int(row.get("refine_A", {}).get("n_a_steps_requested", 0)) for row in selected_refine_rows)
    return {
        "scope": scope,
        "batch": batch,
        "n_retained": int(len(rows)),
        "A_cos_mean": float(np.mean(A)),
        "A_cos_min": float(np.min(A)),
        "A_oracle_cos_mean": float(np.mean(O)),
        "A_oracle_cos_min": float(np.min(O)),
        "refine_A_cos_mean": float(np.mean(R)),
        "refine_A_cos_min": float(np.min(R)),
        "oracle_minus_A_mean": float(np.mean(O - A)),
        "refine_minus_A_mean": float(np.mean(R - A)),
        "refine_minus_A_min": float(np.min(R - A)),
        "refine_minus_A_max": float(np.max(R - A)),
        "refine_lt_A_count": int(np.sum(R < A)),
        "refine_lt_0_99_count": int(np.sum(R < 0.99)),
        "a_steps_accepted": int(accepted),
        "a_steps_requested": int(requested),
    }


def parse_list(raw: str) -> list[float]:
    return [float(item) for item in ast.literal_eval(raw)]


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
