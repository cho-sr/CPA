from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


METHODS = ("fp32", "naive", "A", "A_oracle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose full-n M3 gains after closing Stage-2 image tuning.")
    parser.add_argument(
        "--aggregated-csv",
        type=Path,
        default=Path("results/v7_m3_fallback_5batch/per_image_metrics_aggregated.csv"),
    )
    parser.add_argument(
        "--stage-table-csv",
        type=Path,
        default=Path("results/v5_4bit_table_lsrescale_cpaica.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/v9_stage2_saturation_diagnostics"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_rows = read_csv(args.aggregated_csv)
    table_rows = read_csv(args.stage_table_csv)
    stage_by_method_batch = {(row["method"], int(row["batch"])): row for row in table_rows}
    by_method_gid = {(row["method"], int(row["global_index"])): row for row in image_rows}
    global_indices = sorted(int(row["global_index"]) for row in image_rows if row["method"] == "naive")

    paired_rows = build_component_rows(global_indices, by_method_gid, stage_by_method_batch)
    write_csv(output_dir / "component_lpips_contributions.csv", paired_rows)

    oracle_decomp = build_group_decomposition(paired_rows)
    write_csv(output_dir / "oracle_gain_decomposition.csv", oracle_decomp)

    batch_rows = build_batch_table(paired_rows, stage_by_method_batch)
    write_csv(output_dir / "batch2_context_table.csv", batch_rows)

    make_image_grid(
        output_dir / "figure_original_fp32_4bit_naive_4bit_A_fulln.png",
        paired_rows,
        sort_by_loss=False,
    )
    make_image_grid(
        output_dir / "figure_original_fp32_4bit_naive_4bit_A_fulln_loss_sorted.png",
        paired_rows,
        sort_by_loss=True,
    )

    summary = {
        "n_images": len(paired_rows),
        "overall": first(row for row in oracle_decomp if row["scope"] == "all"),
        "by_A_source": [row for row in oracle_decomp if row["scope"] == "A_source"],
        "batch2": first(row for row in batch_rows if int(row["batch"]) == 2),
        "outputs": {
            "component_lpips_contributions": str(output_dir / "component_lpips_contributions.csv"),
            "oracle_gain_decomposition": str(output_dir / "oracle_gain_decomposition.csv"),
            "batch2_context_table": str(output_dir / "batch2_context_table.csv"),
            "image_grid": str(output_dir / "figure_original_fp32_4bit_naive_4bit_A_fulln.png"),
            "image_grid_loss_sorted": str(output_dir / "figure_original_fp32_4bit_naive_4bit_A_fulln_loss_sorted.png"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    all_row = summary["overall"]
    fallback_row = first(row for row in oracle_decomp if row.get("A_source") == "naive_fallback")
    retained_row = first(row for row in oracle_decomp if row.get("A_source") == "stage2_retained")
    print(
        "oracle recovery={recovery:.3f}; A_oracle-A mean={gap:.4f}; "
        "retained contribution={ret:.4f}; fallback contribution={fb:.4f}".format(
            recovery=float(all_row["A_oracle_recovery_vs_fp32_gap"]),
            gap=float(all_row["A_oracle_vs_A_lpips_gain_mean"]),
            ret=float(retained_row["contribution_to_full_A_oracle_vs_A_mean"]),
            fb=float(fallback_row["contribution_to_full_A_oracle_vs_A_mean"]),
        ),
        flush=True,
    )
    b2 = summary["batch2"]
    print(
        "batch2 B_eff={beff}/8 fallback={fb}/8 recovery={rec:.3f} "
        "retained_A_gain={rgain:.4f}".format(
            beff=int(b2["B_eff"]),
            fb=int(b2["fallback_count"]),
            rec=float(b2["A_recovery_rate"]),
            rgain=float(b2["retained_A_gain_vs_naive_mean"]),
        ),
        flush=True,
    )


def build_component_rows(
    global_indices: list[int],
    by_method_gid: dict[tuple[str, int], dict[str, str]],
    stage_by_method_batch: dict[tuple[str, int], dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gid in global_indices:
        method_rows = {method: by_method_gid[(method, gid)] for method in METHODS}
        batch = int(method_rows["naive"]["batch_index"])
        source = int(method_rows["naive"]["source_index"])
        A_source = method_rows["A"].get("A_source", "stage2_retained")
        stage_A = stage_by_method_batch.get(("A", batch), {})
        stage_oracle = stage_by_method_batch.get(("A_oracle", batch), {})
        stage_naive = stage_by_method_batch.get(("naive", batch), {})
        A_stage_cos = source_cos_from_table(stage_A, source)
        oracle_stage_cos = source_cos_from_table(stage_oracle, source)
        naive_stage_cos = source_cos_from_table(stage_naive, source)
        fp32_lpips = as_float(method_rows["fp32"]["lpips"])
        naive_lpips = as_float(method_rows["naive"]["lpips"])
        A_lpips = as_float(method_rows["A"]["lpips"])
        oracle_lpips = as_float(method_rows["A_oracle"]["lpips"])
        rows.append(
            {
                "global_index": gid,
                "batch": batch,
                "source_index": source,
                "label": int(method_rows["naive"]["label"]),
                "A_source": A_source,
                "retained": A_source == "stage2_retained",
                "per_sample_loss": as_float(method_rows["naive"]["per_sample_loss"]),
                "fp32_lpips": fp32_lpips,
                "naive_lpips": naive_lpips,
                "A_lpips": A_lpips,
                "A_oracle_lpips": oracle_lpips,
                "recoverable_gap_lpips": naive_lpips - fp32_lpips,
                "A_gain_vs_naive_lpips": naive_lpips - A_lpips,
                "A_oracle_gain_vs_naive_lpips": naive_lpips - oracle_lpips,
                "A_oracle_vs_A_lpips_gain": A_lpips - oracle_lpips,
                "A_target_source_cosine": as_float(method_rows["A"]["target_source_cosine"]),
                "naive_target_source_cosine": as_float(method_rows["naive"]["target_source_cosine"]),
                "A_oracle_target_source_cosine": as_float(method_rows["A_oracle"]["target_source_cosine"]),
                "A_stage_source_cosine": A_stage_cos,
                "naive_stage_source_cosine": naive_stage_cos,
                "A_oracle_stage_source_cosine": oracle_stage_cos,
                "A_oracle_minus_A_stage_cosine": oracle_stage_cos - A_stage_cos,
                "orig_png": method_rows["fp32"]["orig_png"],
                "fp32_rec_png": method_rows["fp32"]["rec_png"],
                "naive_rec_png": method_rows["naive"]["rec_png"],
                "A_rec_png": method_rows["A"]["rec_png"],
            }
        )
    return rows


def build_group_decomposition(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_recoverable = sum(float(row["recoverable_gap_lpips"]) for row in rows)
    total_oracle_gain_vs_naive = sum(float(row["A_oracle_gain_vs_naive_lpips"]) for row in rows)
    total_oracle_vs_A = sum(float(row["A_oracle_vs_A_lpips_gain"]) for row in rows)
    total_A_gain_vs_naive = sum(float(row["A_gain_vs_naive_lpips"]) for row in rows)
    groups = [("all", "all", rows)]
    for source in sorted({str(row["A_source"]) for row in rows}):
        groups.append(("A_source", source, [row for row in rows if row["A_source"] == source]))

    out: list[dict[str, Any]] = []
    n_full = max(1, len(rows))
    for scope, name, group in groups:
        n = max(1, len(group))
        sum_recoverable = sum(float(row["recoverable_gap_lpips"]) for row in group)
        sum_oracle_gain_vs_naive = sum(float(row["A_oracle_gain_vs_naive_lpips"]) for row in group)
        sum_oracle_vs_A = sum(float(row["A_oracle_vs_A_lpips_gain"]) for row in group)
        sum_A_gain_vs_naive = sum(float(row["A_gain_vs_naive_lpips"]) for row in group)
        out.append(
            {
                "scope": scope,
                "A_source": "" if scope == "all" else name,
                "n": len(group),
                "fp32_lpips_mean": mean(group, "fp32_lpips"),
                "naive_lpips_mean": mean(group, "naive_lpips"),
                "A_lpips_mean": mean(group, "A_lpips"),
                "A_oracle_lpips_mean": mean(group, "A_oracle_lpips"),
                "recoverable_gap_lpips_mean": sum_recoverable / n,
                "A_gain_vs_naive_lpips_mean": sum_A_gain_vs_naive / n,
                "A_oracle_gain_vs_naive_lpips_mean": sum_oracle_gain_vs_naive / n,
                "A_oracle_vs_A_lpips_gain_mean": sum_oracle_vs_A / n,
                "contribution_to_full_A_gain_vs_naive_mean": sum_A_gain_vs_naive / n_full,
                "contribution_to_full_A_oracle_gain_vs_naive_mean": sum_oracle_gain_vs_naive / n_full,
                "contribution_to_full_A_oracle_vs_A_mean": sum_oracle_vs_A / n_full,
                "share_of_total_A_gain_vs_naive": safe_div(sum_A_gain_vs_naive, total_A_gain_vs_naive),
                "share_of_total_A_oracle_gain_vs_naive": safe_div(sum_oracle_gain_vs_naive, total_oracle_gain_vs_naive),
                "share_of_total_A_oracle_vs_A": safe_div(sum_oracle_vs_A, total_oracle_vs_A),
                "A_recovery_vs_fp32_gap": safe_div(sum_A_gain_vs_naive, total_recoverable)
                if scope == "all"
                else safe_div(sum_A_gain_vs_naive, sum_recoverable),
                "A_oracle_recovery_vs_fp32_gap": safe_div(sum_oracle_gain_vs_naive, total_recoverable)
                if scope == "all"
                else safe_div(sum_oracle_gain_vs_naive, sum_recoverable),
            }
        )
    return out


def build_batch_table(
    rows: list[dict[str, Any]],
    stage_by_method_batch: dict[tuple[str, int], dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for batch in sorted({int(row["batch"]) for row in rows}):
        group = [row for row in rows if int(row["batch"]) == batch]
        retained = [row for row in group if row["A_source"] == "stage2_retained"]
        fallback = [row for row in group if row["A_source"] == "naive_fallback"]
        stage_A = stage_by_method_batch.get(("A", batch), {})
        stage_oracle = stage_by_method_batch.get(("A_oracle", batch), {})
        B_eff = int(stage_A.get("B_eff", len(retained)))
        B = int(stage_A.get("B_requested", len(group)))
        gap = sum(float(row["recoverable_gap_lpips"]) for row in group)
        A_gain = sum(float(row["A_gain_vs_naive_lpips"]) for row in group)
        oracle_gain = sum(float(row["A_oracle_gain_vs_naive_lpips"]) for row in group)
        out.append(
            {
                "batch": batch,
                "B_eff": B_eff,
                "B_requested": B,
                "subthreshold_count": B - B_eff,
                "retained_count": len(retained),
                "fallback_count": len(fallback),
                "retained_ratio": safe_div(len(retained), len(group)),
                "fallback_ratio": safe_div(len(fallback), len(group)),
                "fp32_lpips_mean": mean(group, "fp32_lpips"),
                "naive_lpips_mean": mean(group, "naive_lpips"),
                "A_lpips_mean": mean(group, "A_lpips"),
                "A_oracle_lpips_mean": mean(group, "A_oracle_lpips"),
                "fp32_naive_gap": gap / max(1, len(group)),
                "A_gain_vs_naive_mean": A_gain / max(1, len(group)),
                "A_oracle_gain_vs_naive_mean": oracle_gain / max(1, len(group)),
                "A_recovery_rate": safe_div(A_gain, gap),
                "A_oracle_recovery_rate": safe_div(oracle_gain, gap),
                "retained_A_gain_vs_naive_mean": mean_delta(retained, "A_gain_vs_naive_lpips"),
                "fallback_A_gain_vs_naive_mean": mean_delta(fallback, "A_gain_vs_naive_lpips"),
                "retained_A_oracle_vs_A_gain_mean": mean_delta(retained, "A_oracle_vs_A_lpips_gain"),
                "fallback_A_oracle_vs_A_gain_mean": mean_delta(fallback, "A_oracle_vs_A_lpips_gain"),
                "A_subthr_gt_0_9": int_or_blank(stage_A.get("subthr_gt_0_9", "")),
                "A_subthr_gt_0_7": int_or_blank(stage_A.get("subthr_gt_0_7", "")),
                "A_subthr_gt_0_5": int_or_blank(stage_A.get("subthr_gt_0_5", "")),
                "A_oracle_subthr_gt_0_9": int_or_blank(stage_oracle.get("subthr_gt_0_9", "")),
                "A_oracle_subthr_gt_0_7": int_or_blank(stage_oracle.get("subthr_gt_0_7", "")),
                "A_oracle_subthr_gt_0_5": int_or_blank(stage_oracle.get("subthr_gt_0_5", "")),
                "cond_A_true": as_float(stage_A.get("cond_A_true", "nan")),
                "cond_A_hat": as_float(stage_A.get("cond_A_hat", "nan")),
            }
        )
    return out


def make_image_grid(path: Path, rows: list[dict[str, Any]], *, sort_by_loss: bool) -> None:
    selected = sorted(rows, key=lambda row: float(row["per_sample_loss"])) if sort_by_loss else list(rows)
    columns = [
        ("original", "orig_png"),
        ("fp32", "fp32_rec_png"),
        ("4-bit CPA (unmodified)\n+ naive dequant", "naive_rec_png"),
        ("4-bit A", "A_rec_png"),
    ]
    tile = 224
    header = 38
    height = header + tile * len(selected)
    width = tile * len(columns)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for col, (label, _key) in enumerate(columns):
        draw.text((col * tile + 6, 5), label, fill=(0, 0, 0))
    for row_idx, row in enumerate(selected):
        y = header + row_idx * tile
        for col, (_label, key) in enumerate(columns):
            image_path = resolve_existing_path(str(row[key]), base=Path.cwd())
            if image_path is None:
                continue
            image = Image.open(image_path).convert("RGB").resize((tile, tile))
            sheet.paste(image, (col * tile, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def resolve_existing_path(raw: str, *, base: Path) -> Path | None:
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(base / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def source_cos_from_table(row: dict[str, str], source_index: int) -> float:
    if not row:
        return float("nan")
    values = parse_list(row.get("source_cos_per_truth", "[]"))
    if source_index >= len(values):
        return float("nan")
    return float(values[source_index])


def parse_list(raw: str) -> list[float]:
    if not raw:
        return []
    value = ast.literal_eval(raw)
    return [float(item) for item in value]


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([float(row[key]) for row in rows]))


def mean_delta(rows: list[dict[str, Any]], key: str) -> float:
    return mean(rows, key) if rows else float("nan")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(float(den)) > 1.0e-12 else float("nan")


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def int_or_blank(value: Any) -> int | str:
    if value == "":
        return ""
    try:
        return int(value)
    except (TypeError, ValueError):
        return ""


def first(rows: Any) -> Any:
    for row in rows:
        return row
    raise RuntimeError("expected at least one row")


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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
