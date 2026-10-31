from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Milestone-4 Phase A and compare observed B_eff to MP predictions.")
    parser.add_argument("--manifest", type=Path, default=Path("results/v10_m4_phase_plan/phase_A_manifest.csv"))
    parser.add_argument("--grid-root", type=Path, default=Path("results/v10_m4_grid"))
    parser.add_argument("--prediction-csv", type=Path, default=Path("results/v10_mp_prediction_grid/mp_prediction_grid.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/v10_m4_phase_A_report"))
    parser.add_argument("--phase-b-seeds", default="0,1,2,3,4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    predictions = pd.read_csv(args.prediction_csv)
    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for _, job in manifest.iterrows():
        summary_path = Path(args.grid_root) / str(job["run_tag"]) / "summary.json"
        status = "done" if summary_path.is_file() else "missing"
        status_rows.append({**job.to_dict(), "status": status, "summary_json": str(summary_path)})
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        diag = load_batch_diag(Path(args.grid_root) / str(job["run_tag"]), summary)
        if str(job["job_kind"]) == "fp32_truth_fia":
            rows.append(method_row(job, summary, "fp32", {}, {}, predicted=None))
            continue
        pred = match_prediction(predictions, job)
        for method in sorted(summary.get("metrics", {}).keys()):
            rows.append(method_row(job, summary, method, diag, first_diag(diag), predicted=pred))

    observed = pd.DataFrame(rows)
    status_df = pd.DataFrame(status_rows)
    observed.to_csv(output_dir / "phase_A_observed_metrics.csv", index=False)
    status_df.to_csv(output_dir / "phase_A_job_status.csv", index=False)

    quant = (
        observed[observed["job_kind"].eq("quantized_m3_fia")].copy()
        if "job_kind" in observed.columns
        else pd.DataFrame()
    )
    if not quant.empty:
        make_prediction_figure(quant, output_dir / "mp_predicted_vs_observed_beff.png")
        make_lpips_tables(quant, output_dir)
        make_granularity_table(quant, output_dir / "granularity_tradeoff.csv")
        make_phase_b_manifest(quant, output_dir / "phase_B_selected_manifest.csv", phase_b_seeds=str(args.phase_b_seeds))

    result = {
        "n_jobs": int(len(manifest)),
        "n_done": int(status_df["status"].eq("done").sum()) if not status_df.empty else 0,
        "n_metric_rows": int(len(observed)),
        "outputs": {
            "observed_metrics": str(output_dir / "phase_A_observed_metrics.csv"),
            "job_status": str(output_dir / "phase_A_job_status.csv"),
            "prediction_figure": str(output_dir / "mp_predicted_vs_observed_beff.png"),
            "granularity_tradeoff": str(output_dir / "granularity_tradeoff.csv"),
            "phase_B_selected_manifest": str(output_dir / "phase_B_selected_manifest.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Phase A report: done_jobs={result['n_done']}/{result['n_jobs']} rows={result['n_metric_rows']}", flush=True)


def method_row(
    job: pd.Series,
    summary: dict[str, Any],
    method: str,
    diag_rows: list[dict[str, Any]],
    diag: dict[str, Any],
    *,
    predicted: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = summary.get("metrics", {}).get(method, {})
    observed_beff = as_float(diag.get("A_stage1_B_eff"))
    predicted_beff = as_float(predicted.get("B_eff")) if predicted else np.nan
    return {
        "job_index": int(job["job_index"]),
        "run_tag": str(job["run_tag"]),
        "job_kind": str(job["job_kind"]),
        "method": method,
        "B": int(job["B"]) if str(job["B"]) != "nan" else "",
        "bits": str(job["bits"]),
        "rounding": str(job["rounding"]),
        "seed": int(job["seed"]),
        "scale_granularity": str(job["scale_granularity"]),
        "require_distinct_labels": parse_bool(job["require_distinct_labels"]),
        "n_images": metrics.get("n_images", ""),
        "lpips_mean": metrics.get("lpips_mean", ""),
        "psnr_mean": metrics.get("psnr_mean", ""),
        "ssim_mean": metrics.get("ssim_mean", ""),
        "spearman_loss_lpips": metrics.get("spearman_loss_lpips", ""),
        "spearman_loss_lpips_pvalue": metrics.get("spearman_loss_lpips_pvalue", ""),
        "target_source_cosine_mean": metrics.get("target_source_cosine_mean", ""),
        "target_source_cosine_gt_0_5": metrics.get("target_source_cosine_gt_0_5", ""),
        "target_source_cosine_gt_0_7": metrics.get("target_source_cosine_gt_0_7", ""),
        "target_source_cosine_gt_0_9": metrics.get("target_source_cosine_gt_0_9", ""),
        "predicted_B_eff": predicted_beff,
        "observed_B_eff": observed_beff,
        "B_eff_delta_observed_minus_predicted": observed_beff - predicted_beff
        if np.isfinite(observed_beff) and np.isfinite(predicted_beff)
        else np.nan,
        "predicted_near_boundary_count_pm20pct": predicted.get("near_boundary_count_pm20pct", np.nan) if predicted else np.nan,
        "observed_kept_min_lam_over_mp_edge": as_float(diag.get("A_stage1_kept_min_lam_over_mp_edge")),
        "observed_discarded_max_lam_over_mp_edge": as_float(diag.get("A_stage1_discarded_max_lam_over_mp_edge")),
        "quant_mse": as_float(diag.get("quant_mse")),
        "quant_rel_l2": as_float(diag.get("quant_rel_l2")),
        "quant_max_abs_error": as_float(diag.get("quant_max_abs_error")),
        "labels_unique_count": as_float(diag.get("labels_unique_count")),
        "labels_duplicate_count": as_float(diag.get("labels_duplicate_count")),
        "batch_diag_count": int(len(diag_rows)),
        "summary_json": str(Path(job["run_tag"]) / "summary.json"),
    }


def make_prediction_figure(df: pd.DataFrame, path: Path) -> None:
    points = df[df["method"].eq("A")].copy()
    points = points[np.isfinite(points["predicted_B_eff"]) & np.isfinite(points["observed_B_eff"])]
    if points.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=180)
    bits_values = sorted(points["bits"].unique(), key=lambda x: int(x))
    cmap = plt.get_cmap("viridis", len(bits_values))
    markers = {"per_row": "o", "per_bucket": "s", "per_tensor": "^", "none": "x"}
    for b_idx, bits in enumerate(bits_values):
        subset_bits = points[points["bits"].eq(bits)]
        for scale, subset in subset_bits.groupby("scale_granularity"):
            ax.scatter(
                subset["predicted_B_eff"],
                subset["observed_B_eff"],
                s=58,
                alpha=0.85,
                marker=markers.get(str(scale), "o"),
                color=cmap(b_idx),
                edgecolor="black",
                linewidth=0.5,
                label=f"{bits}-bit {scale}",
            )
    upper = float(max(points["predicted_B_eff"].max(), points["observed_B_eff"].max()))
    ax.plot([0, upper], [0, upper], color="#333333", linewidth=1.2, linestyle="--", label="y=x")
    ax.set_xlabel("Predicted B_eff from MP spectrum")
    ax.set_ylabel("Observed B_eff in Stage 1")
    ax.set_title("MP Prediction vs Observed Stage-1 Rank")
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_xlim(left=0, right=upper * 1.05)
    ax.set_ylim(left=0, right=upper * 1.05)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_lpips_tables(df: pd.DataFrame, output_dir: Path) -> None:
    table = df.pivot_table(
        index=["B", "bits", "scale_granularity", "require_distinct_labels"],
        columns="method",
        values=["lpips_mean", "spearman_loss_lpips", "target_source_cosine_gt_0_9"],
        aggfunc="first",
    )
    table.columns = ["_".join(str(item) for item in col if str(item)) for col in table.columns]
    table.reset_index().to_csv(output_dir / "phase_A_metrics_pivot.csv", index=False)


def make_granularity_table(df: pd.DataFrame, path: Path) -> None:
    rows = []
    group_cols = ["B", "bits", "scale_granularity", "require_distinct_labels"]
    for key, group in df.groupby(group_cols):
        A_rows = group[group["method"].eq("A")]
        first = A_rows.iloc[0] if not A_rows.empty else group.iloc[0]
        row = {col: key[idx] for idx, col in enumerate(group_cols)}
        row.update(
            {
                "predicted_B_eff": first.get("predicted_B_eff", np.nan),
                "observed_B_eff": first.get("observed_B_eff", np.nan),
                "quant_mse": first.get("quant_mse", np.nan),
                "quant_rel_l2": first.get("quant_rel_l2", np.nan),
                "quant_max_abs_error": first.get("quant_max_abs_error", np.nan),
                "labels_unique_count": first.get("labels_unique_count", np.nan),
                "labels_duplicate_count": first.get("labels_duplicate_count", np.nan),
            }
        )
        for method, method_group in group.groupby("method"):
            method_first = method_group.iloc[0]
            row[f"{method}_lpips_mean"] = method_first.get("lpips_mean", np.nan)
            row[f"{method}_spearman_loss_lpips"] = method_first.get("spearman_loss_lpips", np.nan)
        rows.append(row)
    pd.DataFrame(rows).sort_values(group_cols).to_csv(path, index=False)


def make_phase_b_manifest(df: pd.DataFrame, path: Path, *, phase_b_seeds: str) -> None:
    points = df[df["method"].eq("A")].copy()
    if points.empty:
        return
    selected = points[
        (points["predicted_near_boundary_count_pm20pct"].fillna(0).astype(float) > 0)
        | (points["observed_discarded_max_lam_over_mp_edge"].fillna(0).astype(float) >= 2.4)
        | (points["observed_kept_min_lam_over_mp_edge"].fillna(99).astype(float) <= 3.6)
    ].copy()
    if selected.empty:
        selected = points.copy()
    rows = []
    for _, row in selected.iterrows():
        rows.append(
            {
                "include": "phase_B_seed5_boundary_cell",
                "B": int(row["B"]),
                "bits": str(row["bits"]),
                "rounding": str(row["rounding"]),
                "scale_granularity": str(row["scale_granularity"]),
                "require_distinct_labels": bool(row["require_distinct_labels"]),
                "seeds": phase_b_seeds,
                "predicted_B_eff": row["predicted_B_eff"],
                "observed_B_eff": row["observed_B_eff"],
                "predicted_near_boundary_count_pm20pct": row["predicted_near_boundary_count_pm20pct"],
                "observed_kept_min_lam_over_mp_edge": row["observed_kept_min_lam_over_mp_edge"],
                "observed_discarded_max_lam_over_mp_edge": row["observed_discarded_max_lam_over_mp_edge"],
            }
        )
    pd.DataFrame(rows).drop_duplicates().to_csv(path, index=False)


def match_prediction(predictions: pd.DataFrame, job: pd.Series) -> dict[str, Any] | None:
    mask = (
        predictions["B"].astype(int).eq(int(job["B"]))
        & predictions["bits"].astype(int).eq(int(job["bits"]))
        & predictions["scale_granularity"].astype(str).eq(str(job["scale_granularity"]))
        & predictions["require_distinct_labels"].map(parse_bool).eq(parse_bool(job["require_distinct_labels"]))
    )
    matches = predictions[mask]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def load_batch_diag(job_dir: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    csv_path = job_dir / "batch_diagnostics.csv"
    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    return list(summary.get("batch_diagnostics", []))


def first_diag(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[0] if rows else {}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    main()
