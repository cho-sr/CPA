from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V10 source diagnostics for oracle-extra and batch-2 retained gains.")
    parser.add_argument(
        "--stage-table-csv",
        type=Path,
        default=Path("results/v5_4bit_table_lsrescale_cpaica.csv"),
    )
    parser.add_argument(
        "--rank-split-source-csv",
        type=Path,
        default=Path("results/v6_diag_rank_split_fia/source_recovery.csv"),
    )
    parser.add_argument(
        "--component-csv",
        type=Path,
        default=Path("results/v9_stage2_saturation_diagnostics/component_lpips_contributions.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/v10_source_diagnostics"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_rows = read_csv(args.stage_table_csv)
    rank_rows = read_csv(args.rank_split_source_csv)
    component_rows = read_csv(args.component_csv)
    stage_by_method_batch = {(row["method"], int(row["batch"])): row for row in stage_rows}

    oracle_extra_rows = oracle_extra_vs_oracle_rows(rank_rows, stage_by_method_batch)
    write_csv(output_dir / "oracle_extra_vs_A_oracle_subthreshold.csv", oracle_extra_rows)
    oracle_extra_summary = summarize_oracle_extra(oracle_extra_rows)
    write_csv(output_dir / "oracle_extra_vs_A_oracle_summary.csv", oracle_extra_summary)

    batch2_rows = batch_retained_rows(component_rows)
    write_csv(output_dir / "retained_naive_cosine_by_batch.csv", batch2_rows)
    batch2_summary = summarize_retained_by_batch(batch2_rows)
    write_csv(output_dir / "retained_naive_cosine_summary.csv", batch2_summary)

    summary = {
        "oracle_extra_vs_A_oracle": oracle_extra_summary,
        "retained_naive_cosine_by_batch": batch2_summary,
        "outputs": {
            "oracle_extra_rows": str(output_dir / "oracle_extra_vs_A_oracle_subthreshold.csv"),
            "oracle_extra_summary": str(output_dir / "oracle_extra_vs_A_oracle_summary.csv"),
            "retained_rows": str(output_dir / "retained_naive_cosine_by_batch.csv"),
            "retained_summary": str(output_dir / "retained_naive_cosine_summary.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sub = next(row for row in oracle_extra_summary if row["scope"] == "subthreshold_all")
    b2 = next(row for row in batch2_summary if int(row["batch"]) == 2)
    print(
        "subthreshold oracle-extra minus A_oracle mean={delta:.4f}; "
        "batch2 retained naive_cos_mean={cos:.4f}, A_gain={gain:.4f}".format(
            delta=float(sub["oracle_extra_minus_A_oracle_mean"]),
            cos=float(b2["naive_cosine_mean"]),
            gain=float(b2["A_gain_vs_naive_lpips_mean"]),
        ),
        flush=True,
    )


def oracle_extra_vs_oracle_rows(
    rank_rows: list[dict[str, str]],
    stage_by_method_batch: dict[tuple[str, int], dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rank_rows:
        if str(row["mp_subthreshold"]).lower() != "true":
            continue
        batch = int(row["batch_index"])
        source = int(row["source_index"])
        oracle_cos = source_cos_from_table(stage_by_method_batch[("A_oracle", batch)], source)
        A_cos = source_cos_from_table(stage_by_method_batch[("A", batch)], source)
        naive_cos = source_cos_from_table(stage_by_method_batch[("naive", batch)], source)
        out.append(
            {
                "global_index": int(row["global_index"]),
                "batch": batch,
                "source_index": source,
                "per_sample_loss": float(row["per_sample_loss"]),
                "oracle_extra_initial_cosine": float(row["source_cosine_initialization"]),
                "oracle_extra_final_cosine": float(row["source_cosine_final"]),
                "A_oracle_cosine": oracle_cos,
                "A_cosine": A_cos,
                "naive_cosine": naive_cos,
                "oracle_extra_minus_A_oracle": float(row["source_cosine_final"]) - oracle_cos,
                "oracle_extra_selected_for_fia": str(row["selected_for_fia"]).lower() == "true",
            }
        )
    return out


def summarize_oracle_extra(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [summarize_oracle_scope(rows, "subthreshold_all")]
    selected = [row for row in rows if bool(row["oracle_extra_selected_for_fia"])]
    omitted = [row for row in rows if not bool(row["oracle_extra_selected_for_fia"])]
    out.append(summarize_oracle_scope(selected, "oracle_extra_selected_for_fia"))
    out.append(summarize_oracle_scope(omitted, "oracle_extra_omitted_by_0_9_threshold"))
    for batch in sorted({int(row["batch"]) for row in rows}):
        out.append(summarize_oracle_scope([row for row in rows if int(row["batch"]) == batch], f"batch_{batch}"))
    return out


def summarize_oracle_scope(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    return {
        "scope": scope,
        "n": len(rows),
        "oracle_extra_final_mean": mean(rows, "oracle_extra_final_cosine"),
        "A_oracle_mean": mean(rows, "A_oracle_cosine"),
        "oracle_extra_minus_A_oracle_mean": mean(rows, "oracle_extra_minus_A_oracle"),
        "oracle_extra_minus_A_oracle_min": min_value(rows, "oracle_extra_minus_A_oracle"),
        "oracle_extra_minus_A_oracle_max": max_value(rows, "oracle_extra_minus_A_oracle"),
        "selected_for_fia_count": int(sum(bool(row["oracle_extra_selected_for_fia"]) for row in rows)),
    }


def batch_retained_rows(component_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in component_rows:
        if str(row["A_source"]) != "stage2_retained":
            continue
        out.append(
            {
                "global_index": int(row["global_index"]),
                "batch": int(row["batch"]),
                "source_index": int(row["source_index"]),
                "per_sample_loss": float(row["per_sample_loss"]),
                "naive_cosine": float(row["naive_stage_source_cosine"]),
                "A_cosine": float(row["A_stage_source_cosine"]),
                "A_oracle_cosine": float(row["A_oracle_stage_source_cosine"]),
                "A_gain_vs_naive_lpips": float(row["A_gain_vs_naive_lpips"]),
                "A_oracle_vs_A_lpips_gain": float(row["A_oracle_vs_A_lpips_gain"]),
                "naive_lpips": float(row["naive_lpips"]),
                "A_lpips": float(row["A_lpips"]),
                "A_oracle_lpips": float(row["A_oracle_lpips"]),
            }
        )
    return out


def summarize_retained_by_batch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for batch in sorted({int(row["batch"]) for row in rows}):
        group = [row for row in rows if int(row["batch"]) == batch]
        out.append(
            {
                "batch": batch,
                "n_retained": len(group),
                "naive_cosine_mean": mean(group, "naive_cosine"),
                "naive_cosine_min": min_value(group, "naive_cosine"),
                "naive_cosine_lt_0_9_count": int(sum(float(row["naive_cosine"]) < 0.9 for row in group)),
                "A_cosine_mean": mean(group, "A_cosine"),
                "A_oracle_cosine_mean": mean(group, "A_oracle_cosine"),
                "A_gain_vs_naive_lpips_mean": mean(group, "A_gain_vs_naive_lpips"),
                "A_gain_vs_naive_lpips_min": min_value(group, "A_gain_vs_naive_lpips"),
                "A_gain_vs_naive_lpips_max": max_value(group, "A_gain_vs_naive_lpips"),
            }
        )
    return out


def source_cos_from_table(row: dict[str, str], source: int) -> float:
    values = parse_list(row["source_cos_per_truth"])
    return float(values[int(source)])


def parse_list(raw: str) -> list[float]:
    return [float(value) for value in ast.literal_eval(raw)]


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([float(row[key]) for row in rows]))


def min_value(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.min([float(row[key]) for row in rows]))


def max_value(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.max([float(row[key]) for row in rows]))


def read_csv(path: Path) -> list[dict[str, str]]:
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
