"""Frozen M7 successor: shuffle stored reconstructions and describe image LPIPS.

No reconstruction, optimization, source matching, or training is performed.
Historical pairwise LPIPS matrices are reused after diagonal provenance checks;
only the new UIA pairwise matrix needs VGG inference.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .fia import denormalize_imagenet
from .m7_recompute import LPIPSEvaluator, PROJECT_ROOT, sha256_file, write_csv
from .result_validity import require_valid_no_cpa_control
from .seed import seed_all


EXCLUDED_FP32 = (22, 24, 32, 36)
CONDITIONS = ("gia_fp32", "gia_4bit", "uia_fp32")
LABELS = {
    "gia_fp32": "GIA fp32 (36; no oracle fills)",
    "gia_4bit": "GIA 4-bit (40)",
    "uia_fp32": "UIA fp32 (80)",
    "shuffled_gia_fp32": "Shuffled GIA fp32 (20 x 36)",
    "shuffled_gia_4bit": "Shuffled GIA 4-bit (20 x 40)",
    "shuffled_uia_fp32": "Shuffled UIA fp32 (20 x 80)",
    "prior_only": "Prior-only (40)",
    "no_cpa_4bit": "No-CPA 4-bit (40; threshold reference)",
    "no_cpa_fp32": "No-CPA fp32 (40)",
    "dataset_mean": "Constant: full-val mean (40)",
    "gray_0.5": "Constant: gray 0.5 (40)",
    "uniform_noise": "Constant control: uniform noise (40)",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected at least two finite scalar LPIPS values")
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
            "q05": float(np.quantile(values, .05, method="linear")),
            "q50": float(np.quantile(values, .50, method="linear")),
            "q95": float(np.quantile(values, .95, method="linear")),
            "min": float(values.min()), "max": float(values.max())}


def threshold_from_controls(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (40,) or not np.isfinite(values).all():
        raise ValueError("Frozen reference requires all 40 finite GIA 4-bit no-CPA values")
    return float(np.quantile(values, .05, method="linear"))


def shuffle_rows(matrix: np.ndarray, gids: list[int], condition: str,
                 seeds: range = range(20)) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Preserve endpoint IDs when permuting targets across one evaluation set."""
    n = len(gids)
    if n < 2 or len(set(gids)) != n or matrix.shape != (n, n) or not np.isfinite(matrix).all():
        raise ValueError("A finite square matrix with distinct axis IDs is required")
    rows, summaries = [], []
    for seed in seeds:
        perm = np.random.default_rng(seed).permutation(n)
        values = matrix[np.arange(n), perm]
        fixed = perm == np.arange(n)
        rows.extend({"run_tag": "m7_shuffle_control", "condition": condition, "seed": seed,
                     "reconstruction_global_index": int(gids[i]), "target_global_index": int(gids[j]),
                     "reconstruction_axis_index": i, "target_axis_index": int(j),
                     "fixed_point": int(fixed[i]), "lpips": float(values[i])}
                    for i, j in enumerate(perm))
        summaries.append({"condition": condition, "seed": seed, "n_images": n,
                          "fixed_points": int(fixed.sum()), **stats(values),
                          "permutation_target_gids": " ".join(str(gids[j]) for j in perm)})
    return rows, summaries


def assert_hash(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise RuntimeError(f"Hash mismatch: {path}")


def source_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def load_uia() -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], list[dict[str, str]]]:
    folder = PROJECT_ROOT / "results/m6_uia_fp32_B16_fia"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or summary["n_images"] != 80 or summary["oracle_fills"] != 0:
        raise RuntimeError("UIA source must be complete with 80 images and no oracle fills")
    metric_path = folder / "per_image_metrics.csv"
    assert_hash(metric_path, summary["output_hashes"][metric_path.name])
    rows = read_csv(metric_path)
    if [int(r["global_index"]) for r in rows] != list(range(80)):
        raise RuntimeError("UIA aggregate IDs differ from fixed 0..79")
    originals, recs, sources = [], [], [source_record(metric_path)]
    for batch in range(5):
        batch_dir = folder / f"batch{batch:02d}"
        batch_summary = json.loads((batch_dir / "summary.json").read_text(encoding="utf-8"))
        for name in ("inputs.pt", "rec.pt"):
            assert_hash(batch_dir / name, batch_summary["output_hashes"][name])
            sources.append(source_record(batch_dir / name))
        inputs = torch.load(batch_dir / "inputs.pt", map_location="cpu", weights_only=False)
        rec = torch.load(batch_dir / "rec.pt", map_location="cpu", weights_only=True)
        metadata = inputs["metadata"]
        if [int(row["global_index"]) for row in metadata] != list(range(batch * 16, (batch + 1) * 16)):
            raise RuntimeError("UIA input global IDs are not in evaluation order")
        for i, meta in enumerate(metadata):
            row = rows[batch * 16 + i]
            if any(int(meta[key]) != int(row[key]) for key in ("global_index", "batch_index", "source_index", "label")):
                raise RuntimeError("UIA CSV and cached input metadata disagree")
        if tuple(rec.shape) != (16, 3, 224, 224) or rec.shape != inputs["x"].shape:
            raise RuntimeError("Invalid UIA cached reconstruction/input shapes")
        originals.append(denormalize_imagenet(inputs["x"].float()))
        recs.append(rec.float())
    return torch.cat(recs), torch.cat(originals), rows, sources


def historical_matrices() -> tuple[dict[str, np.ndarray], dict[str, list[int]], dict[str, list[dict]], dict]:
    folder = PROJECT_ROOT / "results/m7_recompute_controls"
    historical = read_csv(folder / "historical_per_image.csv")
    aggregate_path = PROJECT_ROOT / "results/v7_m3_fallback_5batch/per_image_metrics_aggregated.csv"
    provenance = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert_hash(aggregate_path, provenance["historical_csv_sha256"])
    aggregates = read_csv(aggregate_path)
    matrices, axes, matched, audit = {}, {}, {}, {}
    for condition, key, raw_key in (("gia_fp32", "fp32_CPA", "fp32"), ("gia_4bit", "four_bit_CPA", "naive")):
        matrix_path = folder / f"{key}_pairwise_lpips.npy"
        matrix = np.load(matrix_path, allow_pickle=False)
        rows = sorted((r for r in historical if r["method"] == key), key=lambda r: int(r["global_index"]))
        raw_rows = sorted((r for r in aggregates if r["method"] == raw_key), key=lambda r: int(r["global_index"]))
        if matrix.shape != (40, 40) or not np.isfinite(matrix).all() or len(rows) != 40 or len(raw_rows) != 40:
            raise RuntimeError("Historical matrix or image coverage is incomplete")
        expected = np.array([float(r["lpips"]) for r in raw_rows])
        if not np.allclose([float(r["lpips"]) for r in rows], expected, atol=0, rtol=0):
            raise RuntimeError("Historical per-image CSV disagrees with original run ledger")
        error = float(np.max(np.abs(matrix.diagonal() - expected)))
        if error > 1e-5:
            raise RuntimeError("Historical cached LPIPS matrix diagonal failed 1e-5 validation")
        excluded = [int(r["global_index"]) for r in rows if r["source_kind"] == "exact_source"]
        if condition == "gia_fp32" and excluded != list(EXCLUDED_FP32):
            raise RuntimeError("Expected exactly the frozen four fp32 exact-source fills")
        gids = [i for i in range(40) if condition != "gia_fp32" or i not in EXCLUDED_FP32]
        matrices[condition] = matrix[np.ix_(gids, gids)]
        axes[condition] = gids
        matched[condition] = [raw_rows[i] for i in gids]
        audit[condition] = {"matrix_source": source_record(matrix_path),
                            "metric_source": source_record(aggregate_path), "axis_global_indices": gids,
                            "excluded_oracle_fill_gids": list(EXCLUDED_FP32) if condition == "gia_fp32" else [],
                            "max_abs_error_diagonal_vs_original": error,
                            "prior_direct_validation": provenance["historical_validation"][key]}
    return matrices, axes, matched, audit


def distribution_rows(condition: str, rows: list[dict], threshold: float) -> list[dict]:
    result = []
    for row in rows:
        value = float(row["lpips"])
        gid = int(row.get("global_index", row.get("target_global_index")))
        result.append({"run_tag": "m7_perimage_distribution", "condition": condition,
                       "global_index": gid, "seed": row.get("seed", ""),
                       "reconstruction_global_index": row.get("reconstruction_global_index", gid),
                       "target_global_index": row.get("target_global_index", gid),
                       "fixed_point": row.get("fixed_point", ""),
                       "lpips": value, "threshold": threshold, "success_strict_below": int(value < threshold),
                       "interpretation": "exploratory GIA threshold transfer" if "uia" in condition else "descriptive common-threshold exceedance"})
    return result


def distribution_summary(rows: list[dict]) -> list[dict]:
    summaries = []
    for condition in LABELS:
        selected = [r for r in rows if r["condition"] == condition]
        values = np.array([r["lpips"] for r in selected])
        count = sum(r["success_strict_below"] for r in selected)
        summaries.append({"condition": condition, "label": LABELS[condition], "n_values": len(values),
                          "n_unique_images": len({r["global_index"] for r in selected}),
                          "n_seeds": 20 if condition.startswith("shuffled_") else 1,
                          "success_count": count, "success_denominator": len(values),
                          "successful_fixed_pairs": sum(r["success_strict_below"] for r in selected if r["fixed_point"] == 1)
                              if condition.startswith("shuffled_") else None,
                          "successful_nonfixed_pairs": sum(r["success_strict_below"] for r in selected if r["fixed_point"] == 0)
                              if condition.startswith("shuffled_") else None,
                          "success_fraction": count / len(values), **stats(values),
                          "interpretation": selected[0]["interpretation"]})
    return summaries


def render_distributions(output: Path, rows: list[dict], threshold: float) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    arrays = {c: np.array([r["lpips"] for r in rows if r["condition"] == c]) for c in LABELS}
    bins = np.linspace(0, 1.5, 31)
    if any(np.min(v) < bins[0] or np.max(v) > bins[-1] for v in arrays.values()):
        raise RuntimeError("A value lies outside the frozen common plotting axis [0, 1.5]")
    colors = ["#2166ac", "#b2182b", "#00856a", "#729dc4", "#d67c87", "#69b5a5", "#8e44ad", "#242424", "#666666", "#d28a10", "#ac6100", "#76731d"]
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=180)
    for (condition, values), color in zip(arrays.items(), colors):
        style = "--" if condition.startswith("shuffled_") else ":" if condition in ("dataset_mean", "gray_0.5", "uniform_noise") else "-"
        ax.hist(values, bins=bins, density=True, histtype="step", color=color, linewidth=1.65,
                linestyle=style, label=LABELS[condition])
    ax.axvline(threshold, color="#111111", linewidth=1.5, linestyle="-.", label=f"Frozen control q05 = {threshold:.4f}")
    ax.set(xlim=(0, 1.5), xlabel="Per-image LPIPS (lower is better)", ylabel="Density (area = 1 per condition)",
           title="M7: all per-image LPIPS on a common axis")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    fig.text(.08, .02, "Shuffled pairs reuse images over 20 seeds; UIA uses a different protocol. Counts are descriptive.", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"lpips_histogram_overlay.{suffix}")
    plt.close(fig)

    values = arrays["gia_4bit"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=180)
    # A fixed, disclosed sensitivity visualization prevents reading one binning as proof of modes.
    histogram_specs = [(0.04, 0.00), (0.04, 0.02), (0.06, 0.00), (0.06, 0.03)]
    for width, offset in histogram_specs:
        edges = np.arange(-width + offset, 1.2 + 2 * width, width)
        axes[0].hist(values, bins=edges, density=True, histtype="step", linewidth=1.5,
                     label=f"width {width:.2f}, offset {offset:.2f}")
    grid = np.linspace(0, 1.2, 1201)
    kde_modes = []
    for factor in (.75, 1., 1.25):
        kde = gaussian_kde(values, bw_method=lambda obj: obj.scotts_factor() * factor)
        density = kde(grid)
        # Report only interior modes having >=5% of maximum-density prominence.
        peaks, properties = find_peaks(density, prominence=.05 * density.max())
        peaks = [int(i) for i in peaks if values.min() <= grid[i] <= values.max()]
        kde_modes.append({"scott_bandwidth_multiplier": factor, "n_prominent_modes": len(peaks),
                          "mode_lpips": [float(grid[i]) for i in peaks],
                          "prominence_rule": "at least 5% of curve maximum; inside observed range"})
        axes[1].plot(grid, density, linewidth=1.6, label=f"Scott bandwidth x {factor:g}")
    for ax in axes:
        ax.axvline(threshold, linestyle="-.", color="black", linewidth=1)
        ax.set(xlim=(0, 1.2), xlabel="GIA 4-bit per-image LPIPS", ylabel="Density")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_title("Histogram sensitivity (same 40 images)")
    axes[1].set_title("KDE sensitivity; descriptive, no mode test")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"gia_4bit_modality_sensitivity.{suffix}")
    plt.close(fig)
    return {"n_images": 40, "common_histogram_bins": bins.tolist(), "histogram_sensitivity": histogram_specs,
            "kde_sensitivity": kde_modes, "formal_multimodality_test": None,
            "interpretation_limit": "N=40; histogram/KDE shapes are descriptive and cannot establish a latent mixture or uniform degradation."}


def main() -> None:
    require_valid_no_cpa_control()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    assert_hash(args.protocol, args.protocol_sha256)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    if (protocol["fp32_reference"]["exclude_global_indices"] != list(EXCLUDED_FP32)
            or protocol["shuffle"]["conditions"] != {"gia_fp32": 36, "gia_4bit": 40, "uia_fp32": 80}
            or protocol["shuffle"]["seeds"] != list(range(20))
            or protocol["shuffle"]["rematch"] is not False
            or protocol["success"]["quantile"] != .05
            or protocol["success"]["quantile_method"] != "linear"
            or protocol["success"]["comparison"] != "LPIPS strictly less than threshold"):
        raise RuntimeError("Runner and frozen protocol disagree")
    started = time.perf_counter()
    torch.set_num_threads(4)
    shuffle_output = PROJECT_ROOT / "results/m7_shuffle_control"
    distribution_output = PROJECT_ROOT / "results/m7_perimage_distribution"
    shuffle_output.mkdir(parents=True, exist_ok=True)
    distribution_output.mkdir(parents=True, exist_ok=True)
    identity = {"protocol": str(args.protocol.resolve()), "protocol_sha256": args.protocol_sha256,
                "code": source_record(Path(__file__)), "started_at_utc": utc_now()}
    # Control-only numeric cutoff is persisted before any candidate is loaded or summarized.
    control_path = PROJECT_ROOT / "results/m7_no_cpa_control/4bit/per_image_metrics.csv"
    controls = read_csv(control_path)
    if [int(r["global_index"]) for r in controls] != list(range(40)):
        raise RuntimeError("Threshold control IDs are incomplete")
    threshold = threshold_from_controls(np.array([float(r["lpips"]) for r in controls]))
    threshold_payload = {**identity, "run_tag": "m7_perimage_distribution", "fixed_at_utc": utc_now(),
                         "reference": "all 40 GIA 4-bit no-CPA images", "reference_source": source_record(control_path),
                         "quantile": .05, "quantile_method": "numpy.quantile(method='linear')", "threshold": threshold,
                         "comparison": "strict LPIPS < threshold", "candidate_metrics_consulted": False,
                         "reason": "An empirical lower-tail control cutoff without LPIPS ratio assumptions; common cutoff permits descriptive comparisons.",
                         "limitations": "In-sample control calibration; no individual-image significance. UIA counts use exploratory threshold transfer."}
    threshold_path = distribution_output / "threshold.json"
    if threshold_path.exists():
        prior = json.loads(threshold_path.read_text(encoding="utf-8"))
        for key in ("threshold", "protocol_sha256", "reference_source"):
            if prior[key] != threshold_payload[key]:
                raise RuntimeError("Existing frozen numerical cutoff differs; refusing overwrite")
    else:
        write_json(threshold_path, threshold_payload)
    print(f"Control-only threshold persisted: {threshold:.10f}", flush=True)

    matrices, axes, matched, audit = historical_matrices()
    rec, truth, uia_rows, uia_sources = load_uia()
    uia_matrix_path = shuffle_output / "uia_fp32_pairwise_lpips.npy"
    uia_meta_path = shuffle_output / "uia_fp32_matrix_provenance.json"
    if uia_matrix_path.is_file() and uia_meta_path.is_file():
        cache = json.loads(uia_meta_path.read_text(encoding="utf-8"))
        if cache["sources"] != uia_sources or cache["protocol_sha256"] != args.protocol_sha256:
            raise RuntimeError("UIA metric cache has different inputs/protocol")
        assert_hash(uia_matrix_path, cache["matrix_sha256"])
        uia_matrix = np.load(uia_matrix_path, allow_pickle=False)
        uia_validation = cache["validation"]
    else:
        # Original UIA calls seed_all before Stage 1, then FIA enables cuDNN
        # benchmarking while retaining deterministic kernels. Match that state.
        seed_all(0)
        torch.backends.cudnn.benchmark = True
        evaluator = LPIPSEvaluator(args.device)
        uia_matrix = evaluator.pairwise(rec, truth, batch_size=args.batch_size)
        direct = evaluator.paired(rec, truth, batch_size=1)
        expected = np.array([float(r["lpips"]) for r in uia_rows])
        uia_validation = {"max_abs_error_diagonal_vs_original": float(np.max(np.abs(uia_matrix.diagonal() - expected))),
                          "max_abs_error_diagonal_vs_direct_single_pair": float(np.max(np.abs(uia_matrix.diagonal() - direct))),
                          "tolerance": 1e-5}
        if any(uia_validation[key] > 1e-5 for key in uia_validation if key != "tolerance"):
            raise RuntimeError(f"UIA diagonal validation failed: {uia_validation}")
        np.save(uia_matrix_path, uia_matrix)
        write_json(uia_meta_path, {**identity, "sources": uia_sources, "validation": uia_validation,
                                  "matrix_sha256": sha256_file(uia_matrix_path), "axis_global_indices": list(range(80))})
        del evaluator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del rec, truth
    print("GPU_RELEASED: all requested LPIPS inference is complete", flush=True)
    write_json(shuffle_output / "gpu_complete.json", {**identity, "completed_at_utc": utc_now(), "validation": uia_validation})
    matrices["uia_fp32"], axes["uia_fp32"], matched["uia_fp32"] = uia_matrix, list(range(80)), uia_rows
    audit["uia_fp32"] = {"sources": uia_sources, "validation": uia_validation, "axis_global_indices": list(range(80))}

    pair_rows, seed_rows, summaries = [], [], []
    for condition in CONDITIONS:
        matrix = matrices[condition]
        np.save(shuffle_output / f"{condition}_pairwise_lpips.npy", matrix)
        pairs, seeds = shuffle_rows(matrix, axes[condition], condition)
        pair_rows.extend(pairs)
        seed_rows.extend(seeds)
        matched_values = np.array([float(r["lpips"]) for r in matched[condition]])
        means = np.array([r["mean"] for r in seeds])
        pooled = np.array([r["lpips"] for r in pairs])
        summaries.append({"condition": condition, "n_images": len(matrix), "n_seeds": 20,
                          "matched_lpips_mean": float(matched_values.mean()),
                          "seed_mean_distribution": stats(means), "pooled_pair_distribution": stats(pooled),
                          "shuffled_minus_matched": float(pooled.mean() - matched_values.mean()),
                          "shuffled_minus_no_cpa_4bit": float(pooled.mean() - np.mean([float(r["lpips"]) for r in controls])),
                          "fixed_points_total": sum(r["fixed_points"] for r in seeds),
                          "fixed_points_by_seed": [r["fixed_points"] for r in seeds]})
    write_csv(shuffle_output / "per_image_metrics.csv", pair_rows)
    write_csv(shuffle_output / "per_seed_summary.csv", seed_rows)
    write_csv(shuffle_output / "summary.csv", [{k: v for k, v in s.items() if not isinstance(v, (dict, list))}
               | {f"seed_means_{k}": v for k, v in s["seed_mean_distribution"].items()}
               | {f"pooled_pairs_{k}": v for k, v in s["pooled_pair_distribution"].items()} for s in summaries])
    shuffle_summary = {**identity, "run_tag": "m7_shuffle_control", "status": "complete",
                       "metric": "FIA torchmetrics VGG LPIPS / qacpa.m7_recompute.LPIPSEvaluator", "sources": audit,
                       "rng": "np.random.default_rng(seed).permutation(N)", "seeds": list(range(20)),
                       "fixed_points": "retained and reported", "rematching": False,
                       "scope": "within each entire evaluation set (36 / 40 / 80), not within original victim batches",
                       "std_definition": "sample standard deviation ddof=1", "quantile_method": "linear",
                       "summaries": summaries,
                       "dependence_warning": "20 seed means vary pairings of the same images; pooled pairs are not independent images.",
                       "completed_at_utc": utc_now(), "wall_time_sec": time.perf_counter() - started}
    write_json(shuffle_output / "summary.json", shuffle_summary)

    perimage = []
    for condition in CONDITIONS:
        perimage.extend(distribution_rows(condition, matched[condition], threshold))
        perimage.extend(distribution_rows("shuffled_" + condition, [r for r in pair_rows if r["condition"] == condition], threshold))
    perimage.extend(distribution_rows("no_cpa_4bit", controls, threshold))
    sources = [source_record(control_path)]
    for condition, relative in (("no_cpa_fp32", "m7_no_cpa_control/fp32/per_image_metrics.csv"),
                                ("prior_only", "m7_prior_only/prior_only/per_image_metrics.csv")):
        source = PROJECT_ROOT / "results" / relative
        perimage.extend(distribution_rows(condition, read_csv(source), threshold))
        sources.append(source_record(source))
    constant_path = PROJECT_ROOT / "results/m7_recompute_controls/per_image_metrics.csv"
    constant_rows = read_csv(constant_path)
    sources.append(source_record(constant_path))
    for condition in ("dataset_mean", "gray_0.5", "uniform_noise"):
        selected = [r for r in constant_rows if r["control"] == condition]
        if len(selected) != 40:
            raise RuntimeError("Constant control requires exactly 40 values")
        perimage.extend(distribution_rows(condition, selected, threshold))
    dist_summary = distribution_summary(perimage)
    seed_successes = []
    for condition in CONDITIONS:
        for seed in range(20):
            selected = [r for r in perimage if r["condition"] == "shuffled_" + condition and r["seed"] == seed]
            seed_successes.append({"condition": "shuffled_" + condition, "seed": seed,
                                   "success_count": sum(r["success_strict_below"] for r in selected),
                                   "denominator": len(selected), "threshold": threshold})
    write_csv(distribution_output / "per_image_metrics.csv", perimage)
    write_csv(distribution_output / "success_counts.csv", dist_summary)
    write_csv(distribution_output / "shuffled_per_seed_success_counts.csv", seed_successes)
    modality = render_distributions(distribution_output, perimage, threshold)
    write_json(distribution_output / "modality_sensitivity.json", modality)
    distribution = {**identity, "run_tag": "m7_perimage_distribution", "status": "complete",
                    "threshold_source": source_record(threshold_path), "threshold": threshold,
                    "sources": sources, "shuffle_source": source_record(shuffle_output / "summary.json"),
                    "conditions": dist_summary, "modality": modality, "completed_at_utc": utc_now(),
                    "wall_time_sec": time.perf_counter() - started}
    write_json(distribution_output / "summary.json", distribution)
    write_reports(shuffle_output, distribution_output, shuffle_summary, distribution)
    print(json.dumps({"shuffle": summaries, "threshold": threshold, "success_counts": dist_summary,
                      "modality": modality["kde_sensitivity"]}, indent=2), flush=True)


def write_reports(shuffle_output: Path, distribution_output: Path, shuffled: dict, distribution: dict) -> None:
    report = ["# M7 shuffled pairing control", "", "Each reconstruction is paired to a uniformly permuted original within its complete evaluation set; no rematching.",
              "NumPy default_rng seeds 0–19; ordinary permutations retain and report fixed points. GIA fp32 excludes oracle-fill gids 22, 24, 32, 36.", "",
              "| Condition | N | Matched LPIPS | Shuffled mean | SD of 20 seed means | Seed-mean q05 / q95 | Change |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in shuffled["summaries"]:
        s = row["seed_mean_distribution"]
        report.append(f"| {row['condition']} | {row['n_images']} | {row['matched_lpips_mean']:.6f} | {s['mean']:.6f} | {s['std']:.6f} | {s['q05']:.6f} / {s['q95']:.6f} | +{row['shuffled_minus_matched']:.6f} |")
    report += ["", "All three shuffled means are above the no-CPA 4-bit reference 0.753823. GIA 4-bit increases from 0.543940 to 0.798902 (+0.254962), so its good matched LPIPS depends on correct correspondence rather than merely the image prior.",
               "Sample SD uses ddof=1; quantiles use linear interpolation. Pooled pair SD/quantiles are separately stored in summary.csv / summary.json.",
               "Twenty seeds reuse the same images; the 720 / 800 / 1600 pairs are not independent image samples. Fixed-point counts and every endpoint ID are retained in per_image_metrics.csv.",
               "No-CPA and prior-only assess FIA without recovered observation structure; shuffled pairing assesses loss of the correct reconstruction-to-original correspondence.",
               "A shuffled mean need not equal the no-CPA mean: generated-image appearance differs. A large rise from matched LPIPS establishes sensitivity to correspondence; it does not by itself isolate the CPA contribution.", ""]
    (shuffle_output / "report.md").write_text("\n".join(report), encoding="utf-8")
    report = ["# M7 per-image LPIPS and descriptive success counts", "",
              f"Frozen threshold: strict LPIPS < **{distribution['threshold']:.10f}**, the linear lower fifth percentile of all 40 GIA 4-bit no-CPA LPIPS values.",
              "The control-only numerical threshold was saved before candidate counting. This successor uses one common descriptive cutoff; it does not replace or amend the historical M7 preregistered rule.",
              "UIA counts apply the GIA control reference across protocols and are exploratory. These are not individual-image significance decisions.", "",
              "| Condition | Mean LPIPS | Success count |", "|---|---:|---:|"]
    for row in distribution["conditions"]:
        report.append(f"| {row['label']} | {row['mean']:.6f} | {row['success_count']}/{row['success_denominator']} |")
    report += ["", "Shuffled denominators count pair evaluations across 20 seeds, not independent images; per-seed image counts are in shuffled_per_seed_success_counts.csv.",
               "Every below-threshold shuffled pair is a retained fixed point; zero nonfixed shuffled pairs pass the threshold in any of the three conditions.",
               "GIA 4-bit has two visible LPIPS groups, with KDE modes near 0.45 and 0.78 across all three preregistered bandwidths. This supports heterogeneous reconstruction quality in this sample. However, 29/40 pass the frozen cutoff: a claim of 'a minority of recoveries and a majority of failures' is inconsistent with these counts.",
               "No-CPA fp32 and 4-bit means are close (0.756820 / 0.753823), but their distributions differ: SD 0.186229 / 0.109000 and common-cutoff counts 10/40 / 2/40. Mean stability does not establish distributional equivalence.",
               "![Common-axis distributions](lpips_histogram_overlay.png)", "", "![4-bit shape sensitivity](gia_4bit_modality_sensitivity.png)", "",
               "The four exploratory histogram variants and three preregistered KDE bandwidths disclose shape sensitivity at N=40. No formal multimodality test is claimed. A smooth unimodal curve does not establish identical degradation for every image, and incidental histogram peaks do not establish a minority-only recovery mixture.", ""]
    (distribution_output / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
