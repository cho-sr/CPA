"""Reproduce the frozen M7 consistency audit from existing CSV/JSON only.

Run from quantization_aware_cpa: python -m qacpa.report_m7_consistency_audit
No model, tensor cache, experiment runner, ICA, FIA, or training is loaded.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/m7_consistency_audit"
PROTOCOL = ROOT.parent / "codex/docs/M7_FOLLOWUP_ANALYSIS_PROTOCOL.json"
EXPECTED_HASH = "24a25980e58356591943559e29474bb777494f426b0834ca6d68f76aa2daae67"
EXCLUDED = {22, 24, 32, 36}
SOURCES: dict[str, str] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path: str | Path) -> Path:
    value = Path(path)
    value = value if value.is_absolute() else ROOT / value
    SOURCES[str(value.relative_to(ROOT.parent)).replace("\\", "/")] = digest(value)
    return value


def read_csv(path: str | Path) -> list[dict]:
    value = source(path)
    with value.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: str | Path) -> dict:
    return json.loads(source(path).read_text(encoding="utf-8-sig"))


def write_csv(name: str, rows: list[dict]) -> None:
    assert rows, name
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(name: str, value: dict) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def correlation(rows: list[dict], condition: str) -> dict:
    loss = np.array([float(row["per_sample_loss"]) for row in rows])
    lpips = np.array([float(row["lpips"]) for row in rows])
    assert np.isfinite(loss).all() and np.isfinite(lpips).all()
    result = spearmanr(loss, lpips, alternative="two-sided")
    return {"condition": condition, "n_images": len(rows), "lpips_mean": float(lpips.mean()),
            "spearman_rho": float(result.statistic), "pvalue_two_sided": float(result.pvalue),
            "pvalue_method": "scipy.stats.spearmanr asymptotic two-sided; descriptive image-level audit"}


def spearman_audit() -> tuple[list[dict], list[dict], dict]:
    paths = ["results/v5_m3_fia_4bit", "results/v6_m3_fia_4bit_batch2",
             "results/v6_m3_fia_4bit_batch3", "results/v6_m3_fia_4bit_batch4"]
    direct = {method: [] for method in ("fp32", "naive")}
    for folder, method in itertools.product(paths, direct):
        path = f"{folder}/{method}/per_image_metrics.csv"
        for line, raw in enumerate(read_csv(path), 2):
            direct[method].append({**raw, "source_csv": path, "source_csv_line": line})
    for method, rows in direct.items():
        ids = [int(row["global_index"]) for row in rows]
        assert len(ids) == len(set(ids)), (method, "duplicate images")
        assert set(ids) == set(range(40)) - (EXCLUDED if method == "fp32" else set())
        rows.sort(key=lambda row: int(row["global_index"]))
    aggregated = read_csv("results/v7_m3_fallback_5batch/per_image_metrics_aggregated.csv")
    older = read_csv("results/v6_m3_spearman_5batch/per_image_metrics_aggregated.csv")
    for collection in (aggregated, older):
        indexed = {(row["method"], int(row["global_index"])): row for row in collection}
        for method, rows in direct.items():
            for row in rows:
                counterpart = indexed[(method, int(row["global_index"]))]
                for key in ("per_sample_loss", "lpips", "target_source_cosine", "image_path", "label"):
                    assert row[key] == counterpart[key], (method, row["global_index"], key)
    pure = [row for row in aggregated if row["method"] == "fp32" and int(row["global_index"]) not in EXCLUDED]
    assert len(pure) == 36
    fills = read_csv("results/v7_m3_fp32_truth_missing/fp32/per_image_metrics.csv")
    assert {int(row["global_index"]) for row in fills} == EXCLUDED
    aggregate_fp = {int(row["global_index"]): row for row in aggregated if row["method"] == "fp32"}
    for row in fills:
        for key in ("lpips", "per_sample_loss"):
            assert row[key] == aggregate_fp[int(row["global_index"])][key]
    main = [correlation(direct["fp32"], "GIA fp32 CPA, pure36"),
            correlation(direct["naive"], "GIA 4-bit CPA + dequant, all40")]
    archived = correlation(list(aggregate_fp.values()), "historical fp32 hybrid40; archived provenance only")
    old_stats = read_csv("results/v6_m3_spearman_5batch/spearman_by_batch.csv")
    new_stats = read_csv("results/v7_m3_fallback_5batch/spearman_by_batch.csv")
    history = []
    for row, old, method, status in ((main[0], old_stats, "fp32", "canonical pure36"),
                                    (main[1], old_stats, "naive", "canonical all40"),
                                    (archived, new_stats, "fp32", "archived hybrid; excluded from main text")):
        logged = next(value for value in old if value["scope"] == "all" and value["method"] == method)
        assert abs(row["spearman_rho"] - float(logged["spearman_loss_lpips"])) < 1e-14
        assert abs(row["pvalue_two_sided"] - float(logged["spearman_pvalue"])) < 1e-14
        history.append({**row, "status": status,
                        "logged_source_csv": "results/" + ("v6_m3_spearman_5batch" if old is old_stats else "v7_m3_fallback_5batch") + "/spearman_by_batch.csv",
                        "raw_method_key": method, "logged_rho": logged["spearman_loss_lpips"],
                        "logged_pvalue": logged["spearman_pvalue"], "matches_logged": True})
    per_image = []
    for method, rows in direct.items():
        for row in rows:
            per_image.append({"condition": "gia_fp32_pure36" if method == "fp32" else "gia_4bit_all40",
                              **{key: row[key] for key in ("global_index", "batch_index", "source_index", "label",
                                  "image_path", "per_sample_loss", "lpips", "target_source_cosine", "source_csv", "source_csv_line")}})
    write_csv("spearman_main.csv", main)
    write_csv("spearman_history.csv", history)
    write_csv("spearman_per_image.csv", per_image)
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for rows, item, color, marker in zip(direct.values(), main, ("#2563eb", "#e17c20"), ("o", "s")):
        ax.scatter([float(row["per_sample_loss"]) for row in rows], [float(row["lpips"]) for row in rows],
                   color=color, marker=marker, s=44, alpha=.8,
                   label=f"{item['condition']}: rho={item['spearman_rho']:+.3f}")
    ax.set(xlabel="Per-sample victim loss", ylabel="LPIPS", title="Loss and reconstruction quality: canonical samples")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(OUT / "loss_vs_lpips_main.png", dpi=200)
    plt.close(fig)
    return main, history, {"direct_original_csv_vs_v6_v7_equal": True,
                          "excluded_truth_fill_indices": sorted(EXCLUDED), "main_fp32_images": 36,
                          "main_4bit_images": 40, "no_new_loss_or_LPIPS_measurement": True}


def cell_key(row: dict) -> tuple:
    return tuple(int(row[key]) for key in ("seed", "batch_index", "n", "T", "bits")) + (
        row["scale_granularity"], str(row["bucket_size"]))


def count_summary(rows: list[dict], subset: str) -> dict:
    errors = np.array([int(row["predicted_B_eff"]) - int(row["measured_recovered_cos09"]) for row in rows])
    return {"subset": subset, "condition_count": len(rows), "exact_count": int((errors == 0).sum()),
            "mean_abs_error": float(np.abs(errors).mean()), "max_abs_error": int(np.abs(errors).max()),
            "n_values": json.dumps(sorted({int(row["n"]) for row in rows})),
            "T_values": json.dumps(sorted({int(row["T"]) for row in rows})),
            "bits_values": json.dumps(sorted({int(row["bits"]) for row in rows})),
            "seeds": json.dumps(sorted({int(row["seed"]) for row in rows})),
            "batch_indices": json.dumps(sorted({int(row["batch_index"]) for row in rows})),
            "granularities": json.dumps(sorted({row["scale_granularity"] for row in rows}))}


def validation_audit() -> tuple[list[dict], list[dict]]:
    rows = read_csv("results/m5_uia_beff_validation/summary.csv")
    saved = read_json("results/m5_uia_beff_validation/summary.json")
    expected = {(seed, 0, n, T, bits, "per_row", "") for seed, n, T, bits in itertools.product(
        range(3), (16, 32, 64), (1, 10), (2, 3, 4, 8))}
    expected |= {(seed, 0, n, T, 4, granularity, "256" if granularity == "per_bucket" else "")
                 for seed, n, T, granularity in itertools.product(range(3), (16, 32), (1, 10), ("per_tensor", "per_bucket"))}
    assert len(rows) == len({cell_key(row) for row in rows}) == len(expected) == 96
    assert {cell_key(row) for row in rows} == expected
    assert saved["config"]["uia_beff_validation"]["n_conditions"] == 96
    assert {cell_key(row) for row in saved["config"]["uia_beff_validation"]["conditions"]} == expected
    assert saved["config"]["data"]["split"] == "val"
    cells = []
    for line, row in enumerate(rows, 2):
        ratios = json.loads(row["lam_over_mp_edge"])
        cosines = json.loads(row["source_cos_per_source"])
        predicted = sum(float(value) > float(row["c_mp"]) for value in ratios)
        measured = sum(float(value) > .9 for value in cosines)
        assert len(ratios) == int(row["n"])
        assert len(cosines) == int(row["stage1_B_eff"])
        assert predicted == int(row["predicted_B_eff"])
        assert measured == int(row["measured_recovered_cos09"])
        cells.append({**{key: row[key] for key in ("condition_index", "seed", "batch_index", "n", "T", "bits",
            "rounding", "scale_granularity", "bucket_size", "lr", "c_mp", "predicted_B_eff", "stage1_B_eff", "measured_recovered_cos09")},
            "round": 0, "split": "val", "protocol_label": "T=1 (= FedSGD)" if int(row["T"]) == 1 else "T=10 (= FedAvg)",
            "predicted_from_saved_eigenvalue_ratios": predicted, "measured_from_saved_source_cosines": measured,
            "exact": predicted == measured, "abs_error": abs(predicted - measured),
            "source_csv": "results/m5_uia_beff_validation/summary.csv", "source_csv_line": line})
    core = [row for row in rows if row["scale_granularity"] == "per_row"]
    summaries = []
    for name, selected in (("all granularities", rows), ("core per-row", core)):
        for T in (None, 1, 10):
            subset = selected if T is None else [row for row in selected if int(row["T"]) == T]
            summaries.append(count_summary(subset, name + ("; all T" if T is None else f"; T={T}")))
    write_csv("beff_cells.csv", cells)
    write_csv("beff_recount.csv", summaries)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    colors = {"per_row": "#2563eb", "per_tensor": "#d45b3e", "per_bucket": "#168568"}
    for T, marker in ((1, "o"), (10, "s")):
        for granularity, color in colors.items():
            subset = [row for row in rows if int(row["T"]) == T and row["scale_granularity"] == granularity]
            ax.scatter([int(row["predicted_B_eff"]) for row in subset],
                       [int(row["measured_recovered_cos09"]) for row in subset],
                       marker=marker, s=70 if T == 1 else 96, facecolors=color if T == 1 else "none",
                       edgecolors=color, linewidths=1.1, alpha=.75)
    ax.plot((0, 66), (0, 66), "--", color="#555555", linewidth=1)
    overall = summaries[0]
    ax.set(xlim=(0, 66), ylim=(0, 66), xlabel="Predicted B_eff", ylabel="Measured source count (cosine > 0.9)",
           title=f"B_eff validation: {overall['exact_count']}/{overall['condition_count']} exact; MAE {overall['mean_abs_error']:.5f}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.2)
    legend = [Line2D([], [], color=color, marker="o", linestyle="", label=key) for key, color in colors.items()]
    legend += [Line2D([], [], color="#333333", marker="o", linestyle="", label="T=1 (= FedSGD)"),
               Line2D([], [], color="#333333", marker="s", markerfacecolor="none", linestyle="", label="T=10 (= FedAvg)")]
    ax.legend(handles=legend, loc="lower right")
    fig.savefig(OUT / "predicted_vs_measured_beff_square.png", dpi=200)
    fig.savefig(OUT / "predicted_vs_measured_beff_square.pdf")
    plt.close(fig)
    return summaries, rows


def n32_audit(validation: list[dict]) -> dict:
    m5 = read_csv("results/m5_uia_spectrum/summary.csv")
    m6 = read_csv("results/m6_round_dependence/summary.csv")
    m6_json = read_json("results/m6_round_dependence/summary.json")
    assert m6_json["protocol"]["local_steps_observed"] == 1
    assert m6_json["protocol"]["measurement_split"] == "val"
    assert m6_json["protocol"]["measurement_batches_are_used_for_training"] is False
    spectra = [row for row in m5 if int(row["n"]) == 32]
    validation32 = [row for row in validation if int(row["n"]) == 32 and int(row["bits"]) == 4 and row["scale_granularity"] == "per_row"]
    rounds = [row for row in m6 if int(row["round"]) == 0]
    normalized = []
    for name, rows in (("m5_uia_spectrum", spectra), ("m5_uia_beff_validation", validation32), ("m6_round_dependence", rounds)):
        for row in rows:
            T = int(row.get("T", 1))
            quant_seed = row.get("quantization_seed")
            if quant_seed in (None, ""):
                quant_seed = int(row["seed"]) + 1_000_003 * int(row["batch_index"]) + 9_173 * int(row["n"]) + 1_009 * T + 101 * int(row["bits"])
            normalized.append({"source": name, "round": int(row.get("round", 0)), "seed": int(row["seed"]),
                "batch_index": int(row["batch_index"]), "n": int(row["n"]), "T": T, "bits": int(row["bits"]),
                "rounding": row["rounding"], "scale_granularity": row["scale_granularity"], "bucket_size": row["bucket_size"],
                "lr": float(row["lr"]), "c_mp": float(row["c_mp"]), "split": "val",
                "observation": "factorized gradient (T=1 bridge)" if T == 1 else "factorized cumulative local gradients",
                "prediction": int(row.get("B_eff_pred", row.get("predicted_B_eff"))),
                "quantization_seed": int(quant_seed), "quantization_seed_origin": "source code formula" if name == "m5_uia_beff_validation" else "logged CSV",
                "aggregation_axis": "single condition; no aggregation", "source_csv": f"results/{name}/summary.csv"})
    matched = [row for row in normalized if row["seed"] == row["batch_index"] == row["round"] == 0 and row["T"] == 1]
    assert len(matched) == 3
    assert len({row["prediction"] for row in matched}) == 1, "Prediction mismatch in the compared scientific condition"
    axes = ["round", "seed", "batch_index", "n", "T", "bits", "rounding", "scale_granularity", "bucket_size", "lr", "c_mp", "split", "observation", "quantization_seed"]
    comparison = [{"axis": key, **{row["source"]: row[key] for row in matched},
                   "same_across_runs": len({str(row[key]) for row in matched}) == 1} for key in axes]
    assert all(row["same_across_runs"] for row in comparison if row["axis"] != "quantization_seed")
    # Path identity is evidenced by existing source-level logs, without opening tensor caches.
    m5_paths = read_csv("results/m5_uia_spectrum/source_norms.csv")
    m6_paths = read_csv("results/m6_round_dependence/source_norms.csv")
    m5_paths = [row for row in m5_paths if int(row["n"]) == 32 and int(row["T"]) == 1 and int(row["seed"]) == int(row["batch_index"]) == 0]
    m6_paths = [row for row in m6_paths if int(row["round"]) == int(row["seed"]) == int(row["batch_index"]) == 0]
    assert {row["image_path"] for row in m5_paths} == {row["image_path"] for row in m6_paths}
    assert len(m5_paths) == len(m6_paths) == 32
    values = [int(row["B_eff_pred"]) for row in sorted(rounds, key=lambda row: int(row["batch_index"]))]
    aggregates = read_csv("results/m6_round_dependence/aggregate_summary.csv")
    mean = float(np.mean(values))
    assert mean == float(next(row for row in aggregates if int(row["round"]) == 0)["B_eff_pred_mean"])
    write_csv("n32_prediction_cells.csv", normalized)
    write_csv("n32_axis_comparison.csv", comparison)
    result = {"m6_round0_by_batch": values, "m6_round0_five_batch_mean": mean,
              "matched_scientific_axes_predictions": {row["source"]: row["prediction"] for row in matched},
              "quantization_seeds": {row["source"]: row["quantization_seed"] for row in matched},
              "same_quantized_input_realization": False, "m5_spectrum_m6_original32_image_paths_equal": True,
              "comparison_type": "prediction versus prediction; no measured count used in this reconciliation",
              "note": "24 is seed0/batch0/T1; 21.2 averages five batches. Quantizer seed schedules differ, so equal predictions are not a byte-identical-input reproducibility check. The M6 spectrum is T1; saved T10 dumps are separate."}
    return result


def main() -> None:
    assert digest(PROTOCOL) == EXPECTED_HASH, "Frozen follow-up protocol changed"
    protocol = read_json(PROTOCOL)
    lock = read_json(PROTOCOL.with_suffix(".lock.json"))
    assert lock["sha256"] == EXPECTED_HASH
    assert protocol["fp32_reference"]["exclude_global_indices"] == sorted(EXCLUDED)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("report_m3_spearman.py", "run_uia_beff_validation.py", "run_uia_spectrum.py", "run_round_dependence.py"):
        source(ROOT / "qacpa" / name)
    main_stats, history, checks = spearman_audit()
    recount, validation = validation_audit()
    n32 = n32_audit(validation)
    now = datetime.now(timezone.utc).isoformat()
    summary = {"schema_version": 1, "run_tag": "m7_consistency_audit", "completed_at_utc": now,
               "protocol_sha256": EXPECTED_HASH, "protocol_locked_at_utc": lock["locked_at_utc"],
               "spearman_main": main_stats, "spearman_provenance_checks": checks,
               "beff_recount": recount, "n32_prediction_reconciliation": n32,
               "historical_spearman_values": "spearman_history.csv only; not a main comparison",
               "execution": "existing CSV/JSON analysis only; no training, model inference, tensor loading, ICA or FIA",
               "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}}
    lines = ["# M7 consistency audit", "", "All calculations use existing CSV/JSON records under the frozen follow-up amendment. No experiment was rerun.", "",
             "## Canonical loss versus LPIPS correlation", "",
             "The main fp32 reference is the 36 actual CPA reconstructions. Global indices 22, 24, 32 and 36 are excluded because their historical fp32 entries used true embeddings. The 4-bit CPA + dequant reference includes all 40 images.", "",
             "| Condition | Images | Mean LPIPS | Spearman rho | Two-sided p |", "|---|---:|---:|---:|---:|"]
    for row in main_stats:
        lines.append(f"| {row['condition']} | {row['n_images']} | {row['lpips_mean']:.10f} | {row['spearman_rho']:+.12f} | {row['pvalue_two_sided']:.12g} |")
    lines += ["", "Original per-method CSVs agree exactly with the old aggregation and the filtered current reference. These are descriptive pooled image-level Spearman tests, with scipy's asymptotic p-values; they do not test a paired change in correlation or establish a batch-level causal claim. The methods have different sample counts. Historical alternative-scope values are retained only in `spearman_history.csv` for provenance.", "",
              "![Canonical loss and LPIPS](loss_vs_lpips_main.png)", "", "## B_eff source-count validation", "",
              "The 96 unique condition keys and saved condition configuration agree. Predicted counts were independently recounted from the saved eigenvalue/MP-edge ratios (> c_mp=3); measured counts were independently recounted from saved matched source cosines (>0.9). Both agree with every recorded count.", "",
              "Core grid: n={16,32,64}, bits={2,3,4,8}, seed={0,1,2}, T={1,10}, per-row scaling, batch0. Additional checks: n={16,32}, bits=4, the same seeds and T, per-tensor or per-bucket scaling (bucket size 256). All use ImageNet val and stochastic rounding.", "",
              "| Subset | Exact / conditions | MAE | Maximum error |", "|---|---:|---:|---:|"]
    for row in recount:
        lines.append(f"| {row['subset']} | {row['exact_count']}/{row['condition_count']} | {row['mean_abs_error']:.6f} | {row['max_abs_error']} |")
    lines += ["", "T=1 over all 48 conditions is the primary FedSGD subset. Its 36-condition core per-row subset is reported separately. These are condition cells, not 96 independent datasets.", "",
              "![Square B_eff plot](predicted_vs_measured_beff_square.png)", "",
              "## n=32 prediction reconciliation", "",
              f"M6 round0 predictions across batches0-4 are {n32['m6_round0_by_batch']}; their arithmetic mean is {n32['m6_round0_five_batch_mean']}. The matching scientific condition is n32, bits4, T1, victim seed0, batch0, round0, stochastic per-row quantization, c_mp3 and ImageNet val. All three source runs predict 24 for that condition. This compares predicted counts with predicted counts.", "",
              "The quantizer random seeds differ: " + ", ".join(f"{key}={value}" for key, value in n32['quantization_seeds'].items()) + ". Therefore this is count agreement across the listed scientific axes, not identical-input numerical reproduction. The m5 spectrum and m6 source logs identify the same 32 original image paths. M6's round-dependence spectrum is FedSGD/T1; its separately saved T10 UIA dumps were not used in this comparison.", "",
              "The full axis comparison is in `n32_axis_comparison.csv`; all selected cells, including the other m5 T and seed values, are in `n32_prediction_cells.csv`.", "",
              "## Reproduce", "", "From `quantization_aware_cpa`, run `python -m qacpa.report_m7_consistency_audit`. The script verifies the frozen protocol SHA-256 and validates source membership before writing results. `input_manifest.json` records every input file hash, dependency versions are in `summary.json`, and output hashes are in `output_manifest.json`.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json("summary.json", summary)
    source(Path(__file__))
    write_json("input_manifest.json", {"run_tag": "m7_consistency_audit", "inputs": SOURCES})
    names = ["spearman_main.csv", "spearman_history.csv", "spearman_per_image.csv", "beff_cells.csv", "beff_recount.csv",
             "n32_prediction_cells.csv", "n32_axis_comparison.csv", "loss_vs_lpips_main.png", "predicted_vs_measured_beff_square.png",
             "predicted_vs_measured_beff_square.pdf", "report.md", "summary.json", "input_manifest.json"]
    write_json("output_manifest.json", {"run_tag": "m7_consistency_audit", "outputs": {name: digest(OUT / name) for name in names}})
    print(json.dumps({"spearman_main": main_stats, "beff_recount": recount, "n32": n32}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
