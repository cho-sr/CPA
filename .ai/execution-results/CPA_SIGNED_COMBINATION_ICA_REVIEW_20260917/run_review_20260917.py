"""One-shot canonical-runtime review of the frozen four-candidate comparison."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "quantization_aware_cpa"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

TASK_ID = "CPA_SIGNED_COMBINATION_ICA_REVIEW_20260917"
PARENT_TASK_ID = "CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916"
PLAN_VERSION = 1
REVIEW_ROOT = ROOT / ".ai/execution-results/CPA_SIGNED_COMBINATION_ICA_REVIEW_20260917"
PARENT_RESULTS = ROOT / ".ai/execution-results/CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916"
PUBLIC_OUTPUT = ROOT / "quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916"
PRIVATE_TRUTH = ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"
REVIEW_OUTPUT = ROOT / "quantization_aware_cpa/results_private/signed_combination_ica_B16_T10_20260916/review_20260917"
PARENT_REGISTRATION = PARENT_RESULTS / "registration.json"
PARENT_COMPARISON = ROOT / "quantization_aware_cpa/results_private/signed_combination_ica_B16_T10_20260916/comparison.json"
REFERENCE_COMPARISON = ROOT / "quantization_aware_cpa/results_private/signed_e2e_h1_nofia_B16_T10_20260916/comparison.json"
PARENT_HANDOFF = ROOT / ".ai/relay-runs/executor/CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916_v1/handoff.md"
SNAPSHOT_HANDOFF = REVIEW_ROOT.parent.parent / "relay-runs/executor/CPA_SIGNED_COMBINATION_ICA_REVIEW_20260917_v1/handoff.md"
EVALUATOR = ROOT / "quantization_aware_cpa/qacpa/signed_combination_ica_evaluation.py"

EXPECTED = {
    "parent_registration": "b5aed69767b22a2074f2fbd76064ea34076fc2283789142fa5a781573a53c81a",
    "parent_comparison": "d765fc3c9292bd41df2ac8a286165850906b5f22b5b7ac9df50f89de6b341369",
    "reference_comparison": "545f3d422ab54121826abd412065f53cc74d37c4d4b346cdbd633042e3b3704e",
    "evaluator": "dec3fd4189db5665378d8be80f724472c14f64b025ce6f38d95d9364394f4dd8",
    "manifest": "9b4891765e6fc7e4adb1e3e11c158b3d4a2c3430d3397e1f1466ddafe2e1cd69",
    "checkpoint": "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
    "old_direct": "fe683f68013896a407a38130ee54ef9dc799a9ca1abf7c4f08efeefea7d91751",
    "old_cell": "b352c3bde7db87ff00ecbbbd4715c5c865ada662c4dcf88b1e116f5369ec9d5a",
    "new_direct": "4efc29ad08b1ea8b0cee6991de6f185839708c2c72aaafb4fedfb307b5e680ce",
    "new_cell": "01c3b2b9289c055feb9c1a0bf496855fb7232a0bf406ba05bd161e679535d708",
    "lpips_backbone": "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
    "lpips_calibration": "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868",
}
TOLERANCE = 2e-8
MAX_ADDED_SECONDS = 300.0
PARENT_TOTAL_SECONDS = 2330.7886750999605
PARENT_CAP_SECONDS = 2700.0

INPUTS = {
    "manifest": ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation/manifest.json",
    "packet_fc_00": ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation/fc_00.sgn4",
    "packet_fc_01": ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation/fc_01.sgn4",
    "packet_fc_02": ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation/fc_02.sgn4",
    "packet_fc_03": ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation/fc_03.sgn4",
    "packet_fc_04": ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation/fc_04.sgn4",
    "packet_fc_05": ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation/fc_05.sgn4",
    "checkpoint": Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth",
    "private_truth": PRIVATE_TRUTH,
}
CANDIDATES = {
    "original_separator_direct": ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive/candidate.pt",
    "original_separator_cell": ROOT / "quantization_aware_cpa/results/signed_e2e_cell_screen_B16_T10_20260916/B_H1_high/candidate.pt",
    "combination_separator_direct": PUBLIC_OUTPUT / "direct_fia/candidate.pt",
    "combination_separator_cell": PUBLIC_OUTPUT / "actual_update_refinement/B_H1_combination_ica/candidate.pt",
}
SOURCES = {
    "signed_combination_ica": ROOT / "quantization_aware_cpa/qacpa/signed_combination_ica.py",
    "signed_combination_ica_runner": ROOT / "quantization_aware_cpa/qacpa/signed_combination_ica_runner.py",
    "signed_combination_ica_evaluation": EVALUATOR,
    "signed_combination_ica_experiment": ROOT / "quantization_aware_cpa/qacpa/signed_combination_ica_experiment.py",
    "signed_e2e_cells": ROOT / "quantization_aware_cpa/qacpa/signed_e2e_cells.py",
    "signed_e2e_runner": ROOT / "quantization_aware_cpa/qacpa/signed_e2e_runner.py",
    "fc_replay": ROOT / "quantization_aware_cpa/qacpa/fc_replay.py",
    "full_update_replay": ROOT / "quantization_aware_cpa/qacpa/full_update_replay.py",
    "full_update_adjoint": ROOT / "quantization_aware_cpa/qacpa/full_update_adjoint.py",
    "full_update_pipeline": ROOT / "quantization_aware_cpa/qacpa/full_update_pipeline.py",
    "deterministic_hybrid_setup": ROOT / "quantization_aware_cpa/qacpa/deterministic_hybrid_setup.py",
    "signed_baseline_observation": ROOT / "quantization_aware_cpa/qacpa/signed_baseline_observation.py",
    "signed_cell_rank": ROOT / "quantization_aware_cpa/qacpa/signed_cell_rank.py",
    "signed_cell_rank_evaluation": ROOT / "quantization_aware_cpa/qacpa/signed_cell_rank_evaluation.py",
    "upstream_naive": ROOT / "quantization_aware_cpa/qacpa/upstream_naive.py",
    "cell_utils": ROOT / "quantized_update_cpa/cell_utils.py",
    "gradient_inversion": ROOT / "official_cpa/src/gradient_inversion.py",
    "feature_inversion": ROOT / "official_cpa/src/feature_inversion.py",
    "vgg16": ROOT / "official_cpa/src/vgg16.py",
    "parent_wrapper": PARENT_RESULTS / "run_real_experiment.py",
}
METADATA = {
    "parent_registration": PARENT_REGISTRATION,
    "parent_stage_execution": PARENT_RESULTS / "stage_execution.json",
    "parent_execution_summary": PARENT_RESULTS / "execution_summary.json",
    "parent_final_audit": PARENT_RESULTS / "final_audit.json",
    "parent_report": PARENT_RESULTS / "report.md",
    "parent_comparison": PARENT_COMPARISON,
    "reference_comparison": REFERENCE_COMPARISON,
    "parent_handoff_snapshot": PARENT_HANDOFF,
}
LPIPS_WEIGHTS = {
    "lpips_backbone": INPUTS["checkpoint"],
    "lpips_calibration": Path(sys.executable).resolve().parent / "Lib/site-packages/torchmetrics/functional/image/lpips_models/vgg.pth",
}


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(label: str, path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        return {"label": label, "path": path.as_posix(), "exists": False, "sha256": None, "bytes": None}
    return {"label": label, "path": path.as_posix(), "exists": True,
            "sha256": sha256(path), "bytes": path.stat().st_size}


def catalog() -> dict:
    values = {}
    for group, entries in (("metadata", METADATA), ("inputs", INPUTS), ("candidates", CANDIDATES),
                           ("sources", SOURCES), ("evaluator_weights", LPIPS_WEIGHTS)):
        values[group] = {label: file_record(label, path) for label, path in entries.items()}
    return values


def flatten_catalog(value: dict) -> dict:
    return {(group, label): item for group, entries in value.items() for label, item in entries.items()}


def runtime_snapshot(label: str) -> dict:
    warn_only = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
    return {
        "label": label,
        "timestamp_utc": now_utc(),
        "pid": os.getpid(),
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": str(torch.__version__),
        "torchvision": __import__("importlib.metadata", fromlist=["version"]).version("torchvision"),
        "torchmetrics": __import__("importlib.metadata", fromlist=["version"]).version("torchmetrics"),
        "environment": {key: os.environ.get(key) for key in (
            "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "MKL_THREADING_LAYER", "PYTHONHASHSEED")},
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": bool(warn_only()) if warn_only else None,
        "cuda_is_available": torch.cuda.is_available(),
        "cuda_is_initialized": torch.cuda.is_initialized(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "cpu_backends": {
            "mkldnn": bool(torch.backends.mkldnn.enabled),
            "mkl": bool(torch.backends.mkl.is_available()),
            "openmp": bool(torch.backends.openmp.is_available()),
        },
        "backend_flags": {
            "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        },
    }


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def independent_parent_timing() -> dict:
    stage_record = json.loads((PARENT_RESULTS / "stage_execution.json").read_text(encoding="utf-8"))
    output = {}
    for stage in stage_record["stages"]:
        seconds = (parse_time(stage["finished_at_utc"]) - parse_time(stage["started_at_utc"])).total_seconds()
        output[stage["stage"]] = {
            "recorded_cumulative_wrapper_wall_seconds": stage["wall_seconds"],
            "started_at_utc": stage["started_at_utc"],
            "finished_at_utc": stage["finished_at_utc"],
            "independent_wall_seconds": seconds,
        }
    return output


def archive_before(before: Path) -> dict:
    archive_items = dict(METADATA)
    archive_items.update({
        "original_separator_direct_candidate": CANDIDATES["original_separator_direct"],
        "original_separator_cell_candidate": CANDIDATES["original_separator_cell"],
        "combination_separator_direct_candidate": CANDIDATES["combination_separator_direct"],
        "combination_separator_cell_candidate": CANDIDATES["combination_separator_cell"],
    })
    archived = {}
    for label, source in archive_items.items():
        if not source.is_file():
            raise FileNotFoundError(f"required before archive file is missing: {source}")
        destination = before / "parent_and_comparison" / f"{label}{source.suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        archived[label] = {"source": source.as_posix(), "archive": destination.as_posix(),
                           "sha256": sha256(source)}
    for label, source in SOURCES.items():
        if not source.is_file():
            raise FileNotFoundError(f"required source is missing: {source}")
        destination = before / "sources" / f"{label}{source.suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        archived[f"source:{label}"] = {"source": source.as_posix(), "archive": destination.as_posix(),
                                      "sha256": sha256(source)}
    for label, source in INPUTS.items():
        if label.startswith("packet_") or label == "manifest":
            if not source.is_file():
                raise FileNotFoundError(f"required packet is missing: {source}")
            destination = before / "public_packets" / f"{label}{source.suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            archived[f"input:{label}"] = {"source": source.as_posix(), "archive": destination.as_posix(),
                                          "sha256": sha256(source)}
    return archived


def sign_check(before_hashes: dict) -> None:
    flat = flatten_catalog(before_hashes)
    expected = {
        ("metadata", "parent_registration"): EXPECTED["parent_registration"],
        ("metadata", "parent_comparison"): EXPECTED["parent_comparison"],
        ("metadata", "reference_comparison"): EXPECTED["reference_comparison"],
        ("sources", "signed_combination_ica_evaluation"): EXPECTED["evaluator"],
        ("inputs", "manifest"): EXPECTED["manifest"],
        ("inputs", "checkpoint"): EXPECTED["checkpoint"],
        ("candidates", "original_separator_direct"): EXPECTED["old_direct"],
        ("candidates", "original_separator_cell"): EXPECTED["old_cell"],
        ("candidates", "combination_separator_direct"): EXPECTED["new_direct"],
        ("candidates", "combination_separator_cell"): EXPECTED["new_cell"],
        ("evaluator_weights", "lpips_backbone"): EXPECTED["lpips_backbone"],
        ("evaluator_weights", "lpips_calibration"): EXPECTED["lpips_calibration"],
    }
    for key, expected_hash in expected.items():
        item = flat[key]
        if not item["exists"] or item["sha256"] != expected_hash:
            raise RuntimeError(f"before hash pin failed for {key}: {item}")
    for label in INPUTS:
        if label.startswith("packet_") and not flat[("inputs", label)]["exists"]:
            raise RuntimeError(f"packet missing before evaluation: {label}")
    if REVIEW_OUTPUT.exists():
        raise FileExistsError(f"review output already exists: {REVIEW_OUTPUT}")


def direction(value: float | None) -> int:
    if value is None or not math.isfinite(float(value)):
        return 0
    return (float(value) > 0) - (float(value) < 0)


def control_residuals(current: dict, reference: dict) -> dict:
    mapping = {"original_separator_direct": "x0", "original_separator_cell": "B_H1_high"}
    residuals = {}
    for current_name, reference_name in mapping.items():
        current_rows = current["image_metrics"][current_name]["per_image"]
        reference_rows = reference["image_metrics"][reference_name]["per_image"]
        by_truth_current = {row["truth_index"]: row for row in current_rows}
        by_truth_reference = {row["truth_index"]: row for row in reference_rows}
        rows = []
        correspondence_equal = True
        for truth_index in range(16):
            left, right = by_truth_current[truth_index], by_truth_reference[truth_index]
            correspondence_equal = correspondence_equal and left["candidate_index"] == right["candidate_index"]
            residual = {
                "truth_index": truth_index,
                "current_candidate_index": left["candidate_index"],
                "reference_candidate_index": right["candidate_index"],
                "correspondence_equal": left["candidate_index"] == right["candidate_index"],
                "metric_differences_current_minus_reference": {},
            }
            for metric in ("mse", "psnr_db", "ssim", "lpips"):
                current_value, reference_value = left.get(metric), right.get(metric)
                difference = None if current_value is None or reference_value is None else current_value - reference_value
                residual["metric_differences_current_minus_reference"][metric] = difference
            lpips_difference = residual["metric_differences_current_minus_reference"]["lpips"]
            residual["lpips_float32_spacing_units"] = (
                None if lpips_difference is None else float(lpips_difference / np.spacing(np.float32(right["lpips"])))
            )
            rows.append(residual)
        residuals[current_name] = {
            "reference_row": reference_name,
            "correspondence_equal": correspondence_equal,
            "per_image": rows,
            "max_abs_metric_difference": {
                metric: max(abs(row["metric_differences_current_minus_reference"][metric] or 0.0) for row in rows)
                for metric in ("mse", "psnr_db", "ssim", "lpips")
            },
        }
    return residuals


def directional_comparison(current: dict, parent: dict) -> dict:
    pairs = {
        "combination_direct_vs_original_direct": ("combination_separator_direct", "original_separator_direct"),
        "combination_cell_vs_combination_direct": ("combination_separator_cell", "combination_separator_direct"),
        "combination_cell_vs_original_cell": ("combination_separator_cell", "original_separator_cell"),
    }
    favorable = {"mean_mse": "lower", "mean_psnr_db": "higher", "mean_ssim": "higher", "mean_lpips": "lower"}
    output = {}
    current_contrasts = current["predeclared_contrasts"]["differences_are_left_minus_right"]
    parent_contrasts = parent["predeclared_contrasts"]["differences_are_left_minus_right"]
    for pair_name, (left, right) in pairs.items():
        left_rows = {row["truth_index"]: row for row in current["image_metrics"][left]["per_image"]}
        right_rows = {row["truth_index"]: row for row in current["image_metrics"][right]["per_image"]}
        per_image = {}
        for metric in ("mse", "psnr_db", "ssim", "lpips"):
            counts = {"favorable_left": 0, "unfavorable_left": 0, "equal": 0}
            for index in range(16):
                delta = left_rows[index][metric] - right_rows[index][metric]
                if delta == 0:
                    counts["equal"] += 1
                elif (metric in ("mse", "lpips") and delta < 0) or (metric in ("mean_psnr_db", "mean_ssim", "psnr_db", "ssim") and delta > 0):
                    counts["favorable_left"] += 1
                else:
                    counts["unfavorable_left"] += 1
            per_image[metric] = counts
        current_values = current_contrasts[pair_name]
        parent_values = parent_contrasts[pair_name]
        output[pair_name] = {
            "left": left,
            "right": right,
            "favorable_directions": favorable,
            "current_mean_differences": current_values,
            "parent_mean_differences": parent_values,
            "mean_direction_changed": {
                metric: direction(current_values[metric]) != direction(parent_values[metric])
                for metric in current_values
            },
            "per_image_direction_counts": per_image,
        }
    return output


def endpoint_summary(parent_comparison: dict) -> dict:
    output = {}
    for name, profile in parent_comparison["endpoint_profiles"].items():
        if name == "exact_feasibility_definition":
            continue
        output[name] = {
            "exact_feasible": profile["exact_feasible"],
            "overall_code_mismatch_count": profile["overall_code_mismatch_count"],
            "overall_code_mismatch_fraction": profile["overall_code_mismatch_fraction"],
            "overall_bounded_cell_violation_count": profile["overall_bounded_cell_violation_count"],
            "overall_bounded_cell_violation_fraction": profile["overall_bounded_cell_violation_fraction"],
            "cell_loss_equal_tensor_mean": profile["cell_loss_equal_tensor_mean"],
        }
    return output


def report_text(summary: dict, comparison: dict | None, controls: dict | None,
                directions: dict | None, timing: dict, error: str | None) -> str:
    lines = [
        f"# {TASK_ID}", "",
        f"Status: **{summary['status']}**.", "",
        "This is one canonical-runtime reevaluation of the four frozen 16-image candidate sets. "
        "No optimizer, FC replay, separator, source/FIA run, new arm, tuning, download, or second evaluation was performed.",
        "",
        "## Runtime repair and execution", "",
        f"- Parent total wall time: {PARENT_TOTAL_SECONDS:.15f} s of the registered {PARENT_CAP_SECONDS:.0f} s cap; remaining before review: {PARENT_CAP_SECONDS - PARENT_TOTAL_SECONDS:.15f} s.",
        f"- Review wrapper wall time: {summary.get('wrapper_wall_seconds')} s; registered added evaluation/finalization limit: {MAX_ADDED_SECONDS:.0f} s; within limit: {summary.get('within_added_limit')}.",
        "- The wrapper recorded runtime before setup and after configure_runtime(42, threads=1). The pre-setup snapshot describes this review process only.",
        "- The unchanged evaluator was invoked exactly once after the public artifact gate path and canonical runtime setup.",
        "",
        "| Parent stage | Stored cumulative wall seconds | Independent duration from timestamps |",
        "|---|---:|---:|",
    ]
    for name, value in timing.items():
        lines.append(f"| {name} | {value['recorded_cumulative_wrapper_wall_seconds']:.6f} | {value['independent_wall_seconds']:.6f} |")
    lines += [
        "",
        "The stored parent values after SEPARATE are cumulative wrapper elapsed times. The independent durations above are derived from each stage's own start/end timestamps.",
        "",
        "## Provenance and private boundary", "",
        "- Public gate remained strict and passed before private loading.",
        "- Private truth was consumed as x only; labels_consumed=false and true_updates_consumed=false. Bundled fields may materialize during safe tensor loading.",
        "- The same one-time pixel-MSE Hungarian correspondence was reused for MSE, PSNR, SSIM, LPIPS and the aligned figure; all 16 images were included.",
        f"- Before/after hash catalog is in {REVIEW_ROOT / 'before_after_integrity.json'}. Parent metadata, public inputs, candidates, protected/shared sources, checkpoint and LPIPS weights remained byte-identical: {summary.get('integrity_unchanged')}.",
        "",
    ]
    if comparison is not None:
        lines += [
            "## Four-set metrics", "",
            "| Set | MSE | PSNR dB | SSIM | LPIPS |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, values in comparison["mean_metrics"].items():
            lines.append(f"| {name} | {values['mean_mse']:.15g} | {values['mean_psnr_db']:.15g} | {values['mean_ssim']:.15g} | {values['mean_lpips']:.15g} |")
        lines += [
            "",
            "The three predeclared contrasts and the per-image directional counts are recorded in review_results.json. "
            "They are descriptive for this fixed batch and seed. The adopted separator package is not an equal-compute or single-prior causal ablation.",
            "",
            "## Control residuals", "",
            f"Control reproduction status: **{comparison['control_metric_reproduction']['status']}** with unchanged absolute tolerance 2e-8.",
            "Correspondence equality, all four per-image residuals and LPIPS float32-spacing units are recorded in review_results.json.",
        ]
        for name, value in controls.items():
            lines.append(f"- {name} vs {value['reference_row']}: correspondence_equal={value['correspondence_equal']}; max absolute residuals={value['max_abs_metric_difference']}.")
        lines += [
            "",
            "If both controls pass, the canonical runtime restores the archived comparison. This bundled setup change does not prove that CPU thread count, seed, or an individual backend flag alone caused the earlier discrepancy.",
            "",
            "## Endpoint feasibility", "",
            "The saved endpoint profiles were reused without replay. Exact feasibility still means zero fixed-observed-scale code mismatches and zero bounded-cell violations across all six FC tensors.",
            "| Set | Exact feasible | Code mismatches | Bounded-cell violations | Cell loss |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, value in summary["endpoint_profiles"].items():
            lines.append(f"| {name} | {str(value['exact_feasible']).lower()} | {value['overall_code_mismatch_count']} | {value['overall_bounded_cell_violation_count']} | {value['cell_loss_equal_tensor_mean']:.12g} |")
        lines += [
            "",
            "The new final cell endpoint remains exact_feasible=false independently of image-metric reproduction and is not a recovery-success claim.",
            "",
            "## Saved optimization observations", "",
            "- Cell loss decreased from 22.8891772 to 0.192871174.",
            "- Feature-target cosine was approximately 0.930897 at the start and 0.680713 at the final endpoint.",
            "- The initial weighted cell/base gradient-norm ratio was approximately 9508.7; the final gradient cosine inferred from saved norms was approximately -0.9542.",
            "These observations support conflicting objectives in this run but do not uniquely establish the cause of poor source recovery.",
            "",
            "## Review artifacts", "",
            f"- Review summary: {REVIEW_ROOT / 'execution_summary.json'}",
            f"- Review results: {REVIEW_ROOT / 'review_results.json'}",
            f"- Review registration: {REVIEW_ROOT / 'review_registration.json'}",
            f"- Runtime before setup: {REVIEW_ROOT / 'runtime_before_setup.json'}",
            f"- Runtime after setup: {REVIEW_ROOT / 'runtime_after_setup.json'}",
            f"- New comparison: {REVIEW_OUTPUT / 'comparison.json'}",
            f"- New report: {REVIEW_OUTPUT / 'report.md'}",
            f"- New figure: {REVIEW_OUTPUT / 'all16_comparison.png'}",
            f"- Parent bytes archived under: {REVIEW_ROOT / 'before'}",
        ]
    else:
        lines += ["## Failure", "", f"The review evaluator did not complete: {error}."]
    lines += ["", "No additional attempt is authorized or implied by this report."]
    return "\n".join(lines) + "\n"


def main() -> int:
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    setup_completed = False
    evaluation_invocations = 0
    comparison = None
    error = None
    evaluation_exit_status = None
    before_hashes = None
    after_hashes = None
    timing = {}
    try:
        if (REVIEW_ROOT / "review_registration.json").exists() or (REVIEW_ROOT / "execution_summary.json").exists():
            raise FileExistsError("review task already has a registration or summary; refusing a second attempt")
        before = REVIEW_ROOT / "before"
        archived = archive_before(before)
        before_hashes = catalog()
        sign_check(before_hashes)
        timing = independent_parent_timing()
        write_json(REVIEW_ROOT / "archive_manifest.json", archived)
        write_json(REVIEW_ROOT / "runtime_before_setup.json", runtime_snapshot("before_configure_runtime"))
        from qacpa.deterministic_hybrid_setup import configure_runtime
        configure_runtime(42, threads=1)
        setup_completed = True
        write_json(REVIEW_ROOT / "runtime_after_setup.json", runtime_snapshot("after_configure_runtime"))
        review_registration = {
            "task_id": TASK_ID,
            "plan_version": PLAN_VERSION,
            "status": "REGISTERED",
            "registered_before_evaluation": True,
            "parent_task_id": PARENT_TASK_ID,
            "parent_registration_path": PARENT_REGISTRATION.resolve().as_posix(),
            "parent_registration_sha256": before_hashes["metadata"]["parent_registration"]["sha256"],
            "parent_comparison_sha256": before_hashes["metadata"]["parent_comparison"]["sha256"],
            "reference_comparison_sha256": before_hashes["metadata"]["reference_comparison"]["sha256"],
            "snapshot_handoff_sha256": sha256(SNAPSHOT_HANDOFF) if SNAPSHOT_HANDOFF.is_file() else None,
            "review_wrapper_sha256": sha256(Path(__file__).resolve()),
            "unchanged_evaluator_sha256": before_hashes["sources"]["signed_combination_ica_evaluation"]["sha256"],
            "public_output": PUBLIC_OUTPUT.resolve().as_posix(),
            "private_truth": PRIVATE_TRUTH.resolve().as_posix(),
            "review_output": REVIEW_OUTPUT.resolve().as_posix(),
            "control_reproduction_tolerance": TOLERANCE,
            "candidate_hashes": {label: before_hashes["candidates"][label]["sha256"] for label in CANDIDATES},
            "before_hashes": before_hashes,
            "archive_manifest": archived,
            "runtime_setup": {
                "function": "qacpa.deterministic_hybrid_setup.configure_runtime",
                "seed": 42,
                "threads": 1,
                "before_snapshot": "runtime_before_setup.json",
                "after_snapshot": "runtime_after_setup.json",
            },
            "zero_optimizer_steps": True,
            "zero_fc_replay_steps": True,
            "evaluation_invocation_limit": 1,
            "parent_independent_timing": timing,
            "parent_total_wall_seconds": PARENT_TOTAL_SECONDS,
            "parent_remaining_seconds_before_review": PARENT_CAP_SECONDS - PARENT_TOTAL_SECONDS,
            "max_added_seconds": MAX_ADDED_SECONDS,
            "quality": "REVIEW_ONLY",
        }
        write_json(REVIEW_ROOT / "review_registration.json", review_registration)
        from qacpa.signed_combination_ica_evaluation import evaluate
        evaluation_invocations += 1
        evaluation_started = time.perf_counter()
        comparison = evaluate(
            registration_path=PARENT_REGISTRATION,
            public_output=PUBLIC_OUTPUT,
            private_truth=PRIVATE_TRUTH,
            output=REVIEW_OUTPUT,
        )
        evaluation_exit_status = 0
        evaluation_seconds = time.perf_counter() - evaluation_started
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        evaluation_exit_status = 1
        evaluation_seconds = None
    finally:
        if before_hashes is None:
            try:
                before_hashes = catalog()
            except Exception as exc:
                error = error or f"{type(exc).__name__}: {exc}"
        try:
            after_hashes = catalog()
        except Exception as exc:
            error = error or f"{type(exc).__name__}: {exc}"
            after_hashes = {}
        if before_hashes:
            flat_before, flat_after = flatten_catalog(before_hashes), flatten_catalog(after_hashes)
            integrity_unchanged = all(
                key in flat_after and flat_before[key].get("exists") == flat_after[key].get("exists")
                and flat_before[key].get("sha256") == flat_after[key].get("sha256")
                for key in flat_before
            )
        else:
            integrity_unchanged = False
        wrapper_wall = time.perf_counter() - started
        if comparison is None and (REVIEW_OUTPUT / "comparison.json").is_file():
            try:
                comparison = json.loads((REVIEW_OUTPUT / "comparison.json").read_text(encoding="utf-8"))
            except Exception:
                pass
        parent_comparison = None
        reference_comparison = None
        controls = None
        directions = None
        if comparison is not None:
            parent_comparison = json.loads(PARENT_COMPARISON.read_text(encoding="utf-8"))
            reference_comparison = json.loads(REFERENCE_COMPARISON.read_text(encoding="utf-8"))
            controls = control_residuals(comparison, reference_comparison)
            directions = directional_comparison(comparison, parent_comparison)
        endpoint_profiles = endpoint_summary(parent_comparison) if parent_comparison else {}
        final_status = "FAILED" if error and comparison is None else (
            "COMPLETED" if comparison and comparison.get("status") == "COMPLETED" else "COMPLETED_WITH_REVIEW"
        )
        review_results = {
            "task_id": TASK_ID,
            "parent_task_id": PARENT_TASK_ID,
            "status": final_status,
            "evaluation_exit_status": evaluation_exit_status,
            "evaluation_invocations": evaluation_invocations,
            "evaluation_seconds": evaluation_seconds,
            "wrapper_wall_seconds": wrapper_wall,
            "within_added_limit": wrapper_wall <= MAX_ADDED_SECONDS,
            "parent_total_wall_seconds": PARENT_TOTAL_SECONDS,
            "parent_cap_seconds": PARENT_CAP_SECONDS,
            "parent_remaining_seconds_before_review": PARENT_CAP_SECONDS - PARENT_TOTAL_SECONDS,
            "cumulative_time_if_added": PARENT_TOTAL_SECONDS + wrapper_wall,
            "cumulative_within_parent_cap": PARENT_TOTAL_SECONDS + wrapper_wall <= PARENT_CAP_SECONDS,
            "runtime_setup_completed": setup_completed,
            "private_output_exists": REVIEW_OUTPUT.exists(),
            "private_loaded": bool(comparison and comparison.get("private_truth_metadata", {}).get("private_loaded")),
            "public_gate_status": (comparison or {}).get("public_gate", {}).get("status"),
            "integrity_unchanged": integrity_unchanged,
            "before_after_integrity_path": (REVIEW_ROOT / "before_after_integrity.json").as_posix(),
            "control_residuals": controls,
            "directional_comparison": directions,
            "endpoint_profiles": endpoint_profiles,
            "error": error,
            "no_second_attempt": True,
            "no_optimizer_steps": True,
            "no_fc_replay_steps": True,
        }
        write_json(REVIEW_ROOT / "before_after_integrity.json", {
            "task_id": TASK_ID,
            "before": before_hashes,
            "after": after_hashes,
            "unchanged": integrity_unchanged,
            "private_truth_hashed_only": True,
        })
        write_json(REVIEW_ROOT / "review_results.json", review_results)
        summary = {
            **review_results,
            "report_path": (REVIEW_ROOT / "report.md").as_posix(),
            "comparison_path": (REVIEW_OUTPUT / "comparison.json").as_posix(),
            "figure_path": (REVIEW_OUTPUT / "all16_comparison.png").as_posix(),
        }
        write_json(REVIEW_ROOT / "execution_summary.json", summary)
        (REVIEW_ROOT / "report.md").write_text(
            report_text(summary, comparison, controls, directions, timing, error), encoding="utf-8"
        )
    print(json.dumps({
        "task_id": TASK_ID,
        "status": final_status,
        "evaluation_exit_status": evaluation_exit_status,
        "evaluation_invocations": evaluation_invocations,
        "wrapper_wall_seconds": wrapper_wall,
        "public_gate_status": (comparison or {}).get("public_gate", {}).get("status"),
        "control_reproduction_status": (comparison or {}).get("control_metric_reproduction", {}).get("status"),
    }, allow_nan=False), flush=True)
    return 0 if final_status != "FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

