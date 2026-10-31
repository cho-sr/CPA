from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
M5_DIR = ROOT / "results" / "m5_uia_beff_validation"
M6_DIR = ROOT / "results" / "m6_round_dependence"
T_SPLIT_DIR = ROOT / "results" / "m5_beff_validation_T1_split"
SPECTRUM_DIR = ROOT / "results" / "m6_round_spectrum_decomposition"
N32_DIR = ROOT / "results" / "m6_n32_prediction_reconciliation"


def main() -> None:
    m5_rows = read_csv(M5_DIR / "summary.csv")
    m6_rows = read_csv(M6_DIR / "summary.csv")
    aggregate_rows = read_csv(M6_DIR / "aggregate_summary.csv")
    lambda_rows = read_csv(M6_DIR / "round_lambda_vectors.csv")
    training_rows = read_csv(M6_DIR / "training_loss.csv")

    write_t_split_report(m5_rows)
    render_beff_scatter(m5_rows, M5_DIR / "predicted_vs_measured_beff.png")
    render_beff_scatter(m5_rows, T_SPLIT_DIR / "predicted_vs_measured_beff_T_split_square.png")
    write_numeric_source_audit(m5_rows)
    write_round_spectrum_decomposition(aggregate_rows, lambda_rows, training_rows)
    write_round0_gap_report(m6_rows, training_rows)
    write_n32_reconciliation(m5_rows, m6_rows)
    write_slide11_png()


def write_t_split_report(rows: list[dict[str, str]]) -> None:
    T_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    full = split_summary(rows)
    core_rows = [row for row in rows if row["scale_granularity"] == "per_row"]
    core = split_summary(core_rows)
    write_csv(T_SPLIT_DIR / "t_split_summary_full_96.csv", full)
    write_csv(T_SPLIT_DIR / "t_split_summary_core_per_row.csv", core)

    overall = exact_summary(rows)
    core_t1 = exact_summary([row for row in core_rows if int(row["T"]) == 1])
    all_t1 = exact_summary([row for row in rows if int(row["T"]) == 1])
    payload = {
        "schema_version": 1,
        "run_tag": "m5_beff_validation_T1_split",
        "source_run_tag": "m5_uia_beff_validation",
        "source_summary_csv": str(M5_DIR / "summary.csv"),
        "confirmed_overall": overall,
        "core_per_row_T1_FedSGD": core_t1,
        "all_granularities_T1": all_t1,
        "paper_sentence_ko": (
            "예측식은 96 조건 중 81 에서 정확했다 (MAE 0.219). "
            "FedSGD 에 해당하는 T=1 부분격자에서는 30/36 가 정확했다."
        ),
        "definition_note": (
            "The 30/36 denominator is the core per-row grid "
            "n in {16,32,64} x bits in {2,3,4,8} x seed in {0,1,2}. "
            "Including the extra 4-bit per-tensor/per-bucket granularity checks, "
            "T=1 is 39/48."
        ),
        "artifacts": {
            "t_split_summary_full_96_csv": str(T_SPLIT_DIR / "t_split_summary_full_96.csv"),
            "t_split_summary_core_per_row_csv": str(T_SPLIT_DIR / "t_split_summary_core_per_row.csv"),
            "scatter_png": str(T_SPLIT_DIR / "predicted_vs_measured_beff_T_split_square.png"),
            "paper_sentence_md": str(T_SPLIT_DIR / "paper_sentence.md"),
        },
    }
    (T_SPLIT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (T_SPLIT_DIR / "paper_sentence.md").write_text(
        "\n".join(
            [
                'run_tag="m5_beff_validation_T1_split"',
                "",
                payload["paper_sentence_ko"],
                "",
                payload["definition_note"],
                "",
            ]
        ),
        encoding="utf-8",
    )


def split_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for label, subset in [
        ("T=1 (= FedSGD)", [row for row in rows if int(row["T"]) == 1]),
        ("T=10 (= FedAvg)", [row for row in rows if int(row["T"]) == 10]),
        ("overall", list(rows)),
    ]:
        summary = exact_summary(subset)
        out.append({"subset": label, **summary})
    return out


def exact_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    diffs = np.asarray(
        [int(row["predicted_B_eff"]) - int(row["measured_recovered_cos09"]) for row in rows],
        dtype=np.int64,
    )
    n = int(diffs.size)
    exact = int(np.sum(diffs == 0))
    return {
        "condition_count": n,
        "exact_count": exact,
        "exact_fraction": float(exact / n) if n else float("nan"),
        "mean_abs_error": float(np.mean(np.abs(diffs))) if n else float("nan"),
        "max_abs_error": int(np.max(np.abs(diffs))) if n else 0,
    }


def render_beff_scatter(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"per_row": "#2563eb", "per_tensor": "#dc2626", "per_bucket": "#059669"}
    markers = {1: "o", 10: "s"}
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    for row in rows:
        T = int(row["T"])
        ax.scatter(
            int(row["predicted_B_eff"]),
            int(row["measured_recovered_cos09"]),
            color=colors.get(row["scale_granularity"], "#6b7280"),
            marker=markers.get(T, "o"),
            s=70 if T == 1 else 58,
            alpha=0.84,
            edgecolor="#111827" if T == 1 else "white",
            linewidth=0.6,
        )
    max_axis = max(max(int(row["n"]) for row in rows), max(int(row["predicted_B_eff"]) for row in rows)) + 2
    ax.plot([0, max_axis], [0, max_axis], color="#111827", linestyle="--", linewidth=1.0)
    ax.set_xlim(0, max_axis)
    ax.set_ylim(0, max_axis)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("predicted B_eff")
    ax.set_ylabel("measured recovered count (cos > 0.9)")
    ax.set_title("UIA B_eff Prediction Validation")
    ax.grid(True, alpha=0.25)
    handles = [
        plt.Line2D([], [], linestyle="", marker="o", color=color, label=scale)
        for scale, color in colors.items()
    ]
    handles.extend(
        [
            plt.Line2D([], [], linestyle="", marker="o", color="#374151", label="T=1 (= FedSGD)"),
            plt.Line2D([], [], linestyle="", marker="s", color="#374151", label="T=10 (= FedAvg)"),
        ]
    )
    ax.legend(handles=handles, loc="lower right", frameon=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_numeric_source_audit(rows: list[dict[str, str]]) -> None:
    overall = exact_summary(rows)
    candidates = scan_prediction_csv_candidates()
    lines = [
        'run_tag="m5_beff_validation_T1_split"',
        "",
        "# Prediction Count Audit",
        "",
        f"Source CSV: `{M5_DIR / 'summary.csv'}`",
        f"Direct recount: {overall['exact_count']}/{overall['condition_count']} exact, "
        f"MAE {overall['mean_abs_error']:.6f}, max error {overall['max_abs_error']}.",
        "",
        "The authoritative overall result is 81/96 exact (MAE 0.219); no alternative overall count is used. "
        "The audit parsed prediction-summary CSV candidates under `quantization_aware_cpa/results`.",
        "",
        "Parsed prediction-summary candidates:",
    ]
    for item in candidates:
        lines.append(
            f"- `{item['path']}`: {item['exact_count']}/{item['condition_count']} exact, "
            f"MAE {item['mean_abs_error']:.6f}, run_tags={item['run_tags']}"
        )
    lines.append("")
    (T_SPLIT_DIR / "numeric_source_audit.md").write_text("\n".join(lines), encoding="utf-8")


def scan_prediction_csv_candidates() -> list[dict[str, Any]]:
    out = []
    for path in (ROOT / "results").rglob("*.csv"):
        try:
            rows = read_csv(path)
        except UnicodeDecodeError:
            continue
        if not rows:
            continue
        first = rows[0]
        if {"predicted_B_eff", "measured_recovered_cos09"}.issubset(first):
            summary = exact_summary(rows)
            out.append(
                {
                    "path": str(path),
                    "run_tags": sorted({row.get("run_tag", "") for row in rows}),
                    **summary,
                }
            )
    return out


def write_round_spectrum_decomposition(
    aggregate_rows: list[dict[str, str]],
    lambda_rows: list[dict[str, str]],
    training_rows: list[dict[str, str]],
) -> None:
    SPECTRUM_DIR.mkdir(parents=True, exist_ok=True)
    lambda_by_round: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in lambda_rows:
        lambda_by_round[int(row["round"])].append(row)
    train_by_round = {int(row["round"]): row for row in training_rows}
    out = []
    for row in aggregate_rows:
        round_idx = int(row["round"])
        group = lambda_by_round[round_idx]
        ratios = np.asarray([float(item["lambda_max"]) / float(item["mp_edge"]) for item in group], dtype=np.float64)
        out.append(
            {
                "run_tag": "m6_round_spectrum_decomposition",
                "round": round_idx,
                "training_loss_mean": float(train_by_round[round_idx]["training_loss_mean"]),
                "measurement_loss_mean": float(row["batch_loss_mean_mean"]),
                "B_eff_pred_mean": float(row["B_eff_pred_mean"]),
                "spearman_loss_a_norm_mean": float(row["spearman_loss_a_norm_mean"]),
                "sigma2_mean": float(row["sigma2_mean"]),
                "sigma2_std": float(row["sigma2_std"]),
                "mp_edge_mean": float(row["mp_edge_mean"]),
                "mp_edge_std": float(row["mp_edge_std"]),
                "lambda_max_mean": float(row["lambda_max_mean"]),
                "lambda_max_std": float(row["lambda_max_std"]),
                "lambda_max_over_mp_edge_mean": float(np.mean(ratios)),
                "lambda_max_over_mp_edge_std": float(np.std(ratios, ddof=1)) if ratios.size > 1 else 0.0,
                "lambda_max_over_mp_edge_from_aggregate": float(row["lambda_max_mean"])
                / float(row["mp_edge_mean"]),
            }
        )
    write_csv(SPECTRUM_DIR / "round_spectrum_decomposition.csv", out)
    render_round_spectrum(out, SPECTRUM_DIR / "round_spectrum_decomposition.png")

    first, last = out[0], out[-1]
    verdict = {
        "schema_version": 1,
        "run_tag": "m6_round_spectrum_decomposition",
        "source_files": {
            "aggregate_summary_csv": str(M6_DIR / "aggregate_summary.csv"),
            "round_lambda_vectors_csv": str(M6_DIR / "round_lambda_vectors.csv"),
        },
        "changes_round0_to_round100": {
            "training_loss_ratio": last["training_loss_mean"] / first["training_loss_mean"],
            "measurement_loss_ratio": last["measurement_loss_mean"] / first["measurement_loss_mean"],
            "sigma2_ratio": last["sigma2_mean"] / first["sigma2_mean"],
            "mp_edge_ratio": last["mp_edge_mean"] / first["mp_edge_mean"],
            "lambda_max_ratio": last["lambda_max_mean"] / first["lambda_max_mean"],
            "lambda_max_over_mp_edge_ratio": last["lambda_max_over_mp_edge_from_aggregate"]
            / first["lambda_max_over_mp_edge_from_aggregate"],
            "B_eff_pred_delta": last["B_eff_pred_mean"] - first["B_eff_pred_mean"],
            "spearman_delta": last["spearman_loss_a_norm_mean"] - first["spearman_loss_a_norm_mean"],
        },
        "verdict": (
            "No shrinking-scale cancellation is observed: sigma2, mp_edge, and lambda_max "
            "rise mildly while their ratio is nearly preserved. Because the measurement "
            "loss changes only slightly, hold the learning-stage-insensitivity paragraph."
        ),
        "artifacts": {
            "round_spectrum_decomposition_csv": str(SPECTRUM_DIR / "round_spectrum_decomposition.csv"),
            "round_spectrum_decomposition_png": str(SPECTRUM_DIR / "round_spectrum_decomposition.png"),
        },
    }
    (SPECTRUM_DIR / "summary.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    (SPECTRUM_DIR / "interpretation.md").write_text(
        "\n".join(
            [
                'run_tag="m6_round_spectrum_decomposition"',
                "",
                "# Round Spectrum Decomposition",
                "",
                verdict["verdict"],
                "",
                "Round 0 to 100 changes:",
                f"- train loss: {first['training_loss_mean']:.4f} -> {last['training_loss_mean']:.4f}",
                f"- measurement loss: {first['measurement_loss_mean']:.4f} -> {last['measurement_loss_mean']:.4f}",
                f"- sigma2: {first['sigma2_mean']:.3e} -> {last['sigma2_mean']:.3e}",
                f"- mp_edge: {first['mp_edge_mean']:.3e} -> {last['mp_edge_mean']:.3e}",
                f"- lambda_max: {first['lambda_max_mean']:.3e} -> {last['lambda_max_mean']:.3e}",
                f"- lambda_max/mp_edge: {first['lambda_max_over_mp_edge_from_aggregate']:.1f} "
                f"-> {last['lambda_max_over_mp_edge_from_aggregate']:.1f}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def render_round_spectrum(rows: list[dict[str, Any]], path: Path) -> None:
    rounds = np.asarray([int(row["round"]) for row in rows], dtype=np.float64)
    sigma2 = np.asarray([float(row["sigma2_mean"]) for row in rows], dtype=np.float64)
    mp_edge = np.asarray([float(row["mp_edge_mean"]) for row in rows], dtype=np.float64)
    lambda_max = np.asarray([float(row["lambda_max_mean"]) for row in rows], dtype=np.float64)
    ratio = np.asarray([float(row["lambda_max_over_mp_edge_from_aggregate"]) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(rounds, lambda_max, marker="o", linewidth=2.2, color="#2563eb", label="lambda_max")
    ax.plot(rounds, mp_edge, marker="s", linewidth=2.2, color="#dc2626", label="mp_edge")
    ax.plot(rounds, sigma2, marker="^", linewidth=2.2, color="#059669", label="sigma2")
    ax.set_yscale("log")
    ax.set_xlabel("round checkpoint")
    ax.set_ylabel("raw spectrum / noise scale (log)")
    ax.set_xticks(rounds)
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(rounds, ratio, marker="D", linewidth=2.2, color="#7c3aed", label="lambda_max/mp_edge")
    ax2.set_ylabel("lambda_max / mp_edge")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=True)
    ax.set_title("M6 Round Spectrum Decomposition")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_round0_gap_report(m6_rows: list[dict[str, str]], training_rows: list[dict[str, str]]) -> None:
    round0 = [row for row in m6_rows if int(row["round"]) == 0]
    train0 = next(row for row in training_rows if int(row["round"]) == 0)
    val_losses = [float(row["batch_loss_mean"]) for row in round0]
    cache_audit = audit_feature_caches()
    lines = [
        'run_tag="m6_round_spectrum_decomposition"',
        "",
        "# Round 0 Train/Val Gap Audit",
        "",
        f"Round 0 train loss is `{float(train0['training_loss_mean']):.4f}` and val/measurement "
        f"loss is `{float(train0['measurement_probe_loss_mean']):.4f}`.",
        "",
        "Findings from logs/code:",
        "- `training_loss_mean` is not a moving average. The code calls `mean_probe_loss` at each checkpoint.",
        "- Round 0 contains no optimizer step before measurement. The round loop measures checkpoint 0 before `train_one_round` is called.",
        "- Train and measurement splits are disjoint (`train` vs `val`) and `measurement_batches_are_used_for_training=False`.",
        "- The cached train and val batches use the same class-label sequence, so class distribution does not explain the gap.",
        "- The image paths differ (`train/...` vs `val/...`), so the remaining explanation visible in logs is split/image difficulty.",
        "- Per-train-batch checkpoint losses were not logged; only the five-batch train mean is available without recomputation.",
        "",
        f"Val batch losses at round 0: {', '.join(f'{value:.4f}' for value in val_losses)}.",
        "",
        "Cached label audit:",
        f"- train total/unique labels: {cache_audit['train_total_labels']}/{cache_audit['train_unique_labels']}",
        f"- val total/unique labels: {cache_audit['val_total_labels']}/{cache_audit['val_unique_labels']}",
        f"- train/val label overlap: {cache_audit['label_overlap']}",
        f"- same labels in each matched batch: {cache_audit['same_labels_each_batch']}",
        f"- same image paths in each matched batch: {cache_audit['same_paths_each_batch']}",
        "",
    ]
    (SPECTRUM_DIR / "round0_train_val_gap_report.md").write_text("\n".join(lines), encoding="utf-8")


def audit_feature_caches() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "train_total_labels": "torch_unavailable",
            "train_unique_labels": "torch_unavailable",
            "val_total_labels": "torch_unavailable",
            "val_unique_labels": "torch_unavailable",
            "label_overlap": "torch_unavailable",
            "same_labels_each_batch": "torch_unavailable",
            "same_paths_each_batch": "torch_unavailable",
        }

    train_labels_all: list[int] = []
    val_labels_all: list[int] = []
    same_labels: list[bool] = []
    same_paths: list[bool] = []
    for batch_idx in range(5):
        train_path = M6_DIR / "train_feature_cache" / f"training_features_seed0_splittrain_n32_batch{batch_idx}.pt"
        val_path = ROOT / "cache" / f"uia_features_seed0_n32_batch{batch_idx}.pt"
        train = torch.load(train_path, map_location="cpu")
        val = torch.load(val_path, map_location="cpu")
        train_labels = [int(value) for value in train["labels"].tolist()]
        val_labels = [int(value) for value in val["labels"].tolist()]
        train_labels_all.extend(train_labels)
        val_labels_all.extend(val_labels)
        same_labels.append(train_labels == val_labels)
        same_paths.append(list(train.get("image_paths", [])) == list(val.get("image_paths", [])))
    return {
        "train_total_labels": len(train_labels_all),
        "train_unique_labels": len(set(train_labels_all)),
        "val_total_labels": len(val_labels_all),
        "val_unique_labels": len(set(val_labels_all)),
        "label_overlap": len(set(train_labels_all) & set(val_labels_all)),
        "same_labels_each_batch": same_labels,
        "same_paths_each_batch": same_paths,
    }


def write_n32_reconciliation(m5_rows: list[dict[str, str]], m6_rows: list[dict[str, str]]) -> None:
    N32_DIR.mkdir(parents=True, exist_ok=True)
    m5_target = [
        row
        for row in m5_rows
        if int(row["n"]) == 32 and int(row["bits"]) == 4 and row["scale_granularity"] == "per_row"
    ]
    m6_round0 = [row for row in m6_rows if int(row["round"]) == 0]
    rows = []
    for row in sorted(m5_target, key=lambda item: (int(item["T"]), int(item["seed"]))):
        rows.append(
            {
                "source": "m5_uia_beff_validation",
                "round": 0,
                "seed": int(row["seed"]),
                "batch_index": int(row["batch_index"]),
                "n": int(row["n"]),
                "T": int(row["T"]),
                "bits": int(row["bits"]),
                "scale_granularity": row["scale_granularity"],
                "prediction": int(row["predicted_B_eff"]),
                "measured": int(row["measured_recovered_cos09"]),
                "aggregation_axis": "seeds_on_batch0",
            }
        )
    for row in sorted(m6_round0, key=lambda item: int(item["batch_index"])):
        rows.append(
            {
                "source": "m6_round_dependence",
                "round": int(row["round"]),
                "seed": int(row["seed"]),
                "batch_index": int(row["batch_index"]),
                "n": int(row["n"]),
                "T": 1,
                "bits": int(row["bits"]),
                "scale_granularity": row["scale_granularity"],
                "prediction": int(row["B_eff_pred"]),
                "measured": "",
                "aggregation_axis": "batches_on_seed0",
            }
        )
    write_csv(N32_DIR / "n32_prediction_reconciliation.csv", rows)

    m5_seed0_batch0 = next(row for row in m5_target if int(row["seed"]) == 0 and int(row["T"]) == 1)
    m6_batch0 = next(row for row in m6_round0 if int(row["batch_index"]) == 0)
    lines = [
        'run_tag="m6_round_spectrum_decomposition"',
        "",
        "# n=32 Prediction Reconciliation",
        "",
        f"Matched condition check: m5 seed0/batch0/T=1 predicts `{m5_seed0_batch0['predicted_B_eff']}` "
        f"and m6 round0/seed0/batch0 predicts `{m6_batch0['B_eff_pred']}`.",
        "",
        "The apparent 24/32 vs 21.2/32 mismatch is an aggregation-axis difference: "
        "24 is the single seed0,batch0 condition, while 21.2 is the m6 mean over five "
        "measurement batches at seed0 (`24,25,22,21,14`).",
        "",
    ]
    (N32_DIR / "interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def write_slide11_png() -> None:
    path = M6_DIR / "slide11_premise_validation.png"
    width, height = 2880, 1620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(68)
    subtitle_font = load_font(36)
    header_font = load_font(31)
    body_font = load_font(29)
    bold_font = load_font(30)
    note_font = load_font(27)
    foot_font = load_font(23)

    x0, y0 = 135, 70
    draw.text((x0, y0), "Slide 11 - Premise Validation", fill="#111827", font=title_font)
    draw.text(
        (x0, y0 + 92),
        "4-bit CPA/FIA trend: no-CPA and prior-only controls are required before claiming success.",
        fill="#374151",
        font=subtitle_font,
    )

    columns = [
        ("Run", 490),
        ("Batch / n", 250),
        ("Protocol", 470),
        ("Quantization", 470),
        ("LPIPS", 230),
        ("Interpretation", 650),
    ]
    rows = [
        ("Paper CPA+FIA fp32", "B=32", "upstream reference", "fp32", "0.483", "published reference"),
        ("Ported upstream fp32", "B=32", "official CPA code", "fp32", "0.4660", "reproduced / slightly better"),
        ("Upstream 4-bit", "B=16", "official CPA/FIA", "4-bit stoch per-row", "0.5340", "lower-B quantized point"),
        ("Local CPA (unmodified)", "B=8", "+ naive dequant", "4-bit stoch per-row", "0.5439", "between upstream B=16/32"),
        ("Upstream 4-bit", "B=32", "official CPA/FIA", "4-bit stoch per-row", "0.5615", "higher-B degradation"),
    ]
    table_x = x0
    table_y = 220
    row_h = 132
    header_h = 132
    table_w = sum(width for _label, width in columns)
    draw.rectangle((table_x, table_y, table_x + table_w, table_y + header_h), fill="#1f2937")
    col_x = table_x
    for label, col_w in columns:
        draw.text((col_x + 24, table_y + 48), label, fill="white", font=header_font)
        col_x += col_w
    for ridx, row in enumerate(rows):
        y = table_y + header_h + ridx * row_h
        draw.rectangle((table_x, y, table_x + table_w, y + row_h), fill="#f8fafc" if ridx % 2 == 0 else "white")
        draw.line((table_x, y, table_x + table_w, y), fill="#d1d5db", width=2)
        col_x = table_x
        for cidx, (value, (_label, col_w)) in enumerate(zip(row, columns)):
            font = bold_font if cidx == 4 and ridx in {1, 2, 3, 4} else body_font
            draw.text((col_x + 24, y + 47), value, fill="#1f2937", font=font)
            col_x += col_w
    draw.rectangle((table_x, table_y, table_x + table_w, table_y + header_h + len(rows) * row_h), outline="#cbd5e1", width=2)

    notes = [
        (
            "Conclusion",
            "B=8 CPA (unmodified) + naive dequant lies between upstream B=16 and B=32. This does not establish reconstruction success.",
            "#eef2ff",
            "#c7d2fe",
        ),
        (
            "Bridge",
            "At T=1, UIA and GIA are identical (rel_l2 = 0.0), so the UIA grid includes GIA as a special case. Prediction validation ran for T in {1, 10}.",
            "#ecfdf5",
            "#bbf7d0",
        ),
        (
            "Prediction",
            "B_eff exact in 81/96 conditions (MAE 0.219); FedSGD core T=1 grid exact in 30/36.",
            "#fff7ed",
            "#fed7aa",
        ),
        (
            "Round / threat",
            "All reported GIA/FedSGD reconstruction numbers before m6 are round0. Single transmission observed -> reported leakage is a lower bound.",
            "#f8fafc",
            "#cbd5e1",
        ),
    ]
    note_x = x0
    note_w = table_w
    note_y = table_y + header_h + len(rows) * row_h + 42
    note_h = 95
    label_w = 355
    for label, text, fill, outline in notes:
        draw.rectangle((note_x, note_y, note_x + note_w, note_y + note_h), fill=fill, outline=outline, width=2)
        draw.text((note_x + 42, note_y + 30), label, fill="#111827", font=bold_font)
        draw.text((note_x + label_w, note_y + 30), text, fill="#374151", font=note_font)
        note_y += note_h + 24
    draw.text(
        (x0, height - 85),
        "Footnote: B differs across rows (8/16/32), so this validates trend consistency rather than same-condition equality.",
        fill="#6b7280",
        font=foot_font,
    )
    image.save(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
