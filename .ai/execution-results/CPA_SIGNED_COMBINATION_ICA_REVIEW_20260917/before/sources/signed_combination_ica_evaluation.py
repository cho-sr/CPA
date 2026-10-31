"""Strict offline evaluator for the one registered combination-ICA run.

This module is deliberately separate from the historical screen evaluators.
It authenticates the new outer experiment and the two archived controls before
loading the bundled private truth.  Only ``x`` is consumed from that bundle;
the bundled labels and update are disclosed as materialized fields but are not
read or used for matching, selection, or attack decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from . import signed_combination_ica_runner as runner
from .full_update_evaluation import evaluate_candidate
from .signed_e2e_runner import (
    IMAGE_COUNT, REPO_ROOT, hash_files, protected_paths, write_json,
)


TASK_ID = "CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916"
PLAN_VERSION = 1
IMPLEMENTATION_TASK_ID = runner.TASK_ID
ROOT = REPO_ROOT
REPORT_ROOT = ROOT / ".ai/execution-results/CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916"
PUBLIC_ROOT = ROOT / "quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916"
PRIVATE_ROOT = ROOT / "quantization_aware_cpa/results_private/signed_combination_ica_B16_T10_20260916"
PRIVATE_BASELINE_ROOT = ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10"
PRIVATE_TRUTH = PRIVATE_BASELINE_ROOT / "observation/evaluation_truth.pt"
OLD_SCREEN_ROOT = ROOT / "quantization_aware_cpa/results/signed_e2e_cell_screen_B16_T10_20260916"
OLD_DIRECT_CANDIDATE = ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive/candidate.pt"
OLD_CELL_CANDIDATE = OLD_SCREEN_ROOT / "B_H1_high/candidate.pt"
OLD_PRIVATE_COMPARISON = ROOT / "quantization_aware_cpa/results_private/signed_e2e_h1_nofia_B16_T10_20260916/comparison.json"
REPAIR_INTEGRITY = ROOT / ".ai/execution-results/CPA_SIGNED_COMBINATION_ICA_REPAIR_20260916/source_integrity.json"
TASK_WRAPPER = REPORT_ROOT / "run_real_experiment.py"

EXPECTED_OLD_METADATA_HASHES = {
    "registration.json": "25ae8a5ef4d2e38110709b1fd003d1aa09b823dd63ed2c0a41391d49f79bbe69",
    "summary.json": "dca2618457914fc38298041dd0270c6f113fb899f5c7f88af6a48278c8557a00",
    "integrity.json": "a98f0b2e5526fe305a147f69c5e51d13478252eda927a8349b7f12f6fa530d65",
}
EXPECTED_CONTROL_HASHES = {
    "original_separator_direct": "fe683f68013896a407a38130ee54ef9dc799a9ca1abf7c4f08efeefea7d91751",
    "original_separator_cell": "b352c3bde7db87ff00ecbbbd4715c5c865ada662c4dcf88b1e116f5369ec9d5a",
}
CONTROL_NAMES = (
    "original_separator_direct",
    "original_separator_cell",
    "combination_separator_direct",
    "combination_separator_cell",
)
METRIC_NAMES = ("mean_mse", "mean_psnr_db", "mean_ssim", "mean_lpips")
CONTROL_REPRODUCTION_TOLERANCE = 2e-8


class EvaluationError(RuntimeError):
    """A public provenance or offline evaluation contract failed."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"invalid UTF-8 JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvaluationError(f"expected JSON object: {path}")
    return value


def _must_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise EvaluationError(f"required regular file is missing: {path}")


def _norm_map(values: Mapping[str, str]) -> dict[str, str]:
    return {str(key).replace("\\", "/").casefold(): str(value) for key, value in values.items()}


def _repo_key(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _code_hashes() -> dict[str, str]:
    paths = [
        ROOT / "quantization_aware_cpa/qacpa/signed_combination_ica.py",
        ROOT / "quantization_aware_cpa/qacpa/signed_combination_ica_runner.py",
        ROOT / "quantization_aware_cpa/qacpa/signed_combination_ica_experiment.py",
        ROOT / "quantization_aware_cpa/qacpa/signed_combination_ica_evaluation.py",
        TASK_WRAPPER,
    ]
    return {path.resolve().as_posix(): file_sha256(path) for path in paths}


def _public_input_hashes() -> dict[str, str]:
    return runner._public_packet_hashes(runner.PUBLIC_OBSERVATION) | {
        runner.CHECKPOINT.resolve().as_posix(): file_sha256(runner.CHECKPOINT),
    }


def _repair_integrity_snapshot() -> dict[str, Any]:
    repair = _read_json(REPAIR_INTEGRITY)
    protected = hash_files(protected_paths())
    shared = hash_files(runner.SHARED_PATHS)
    adapted_paths = [
        ROOT / "ICA-GA-main/ICA-GA-main/attacks/feature_separation/base_ica.py",
        ROOT / "ICA-GA-main/ICA-GA-main/attacks/feature_separation/fast_ica.py",
        ROOT / "ICA-GA-main/ICA-GA-main/attacks/feature_separation/gradient_descent_ica.py",
        ROOT / "ICA-GA-main/ICA-GA-main/attacks/feature_separation/combination_ica.py",
        ROOT / "ICA-GA-main/ICA-GA-main/LICENSE",
    ]
    adapted = {_repo_key(path): file_sha256(path) for path in adapted_paths}
    expected_protected = repair.get("protected_source_hashes_after", {})
    expected_shared = repair.get("shared_source_hashes_after", {})
    expected_adapted = repair.get("adapted_reference_hashes_after", {})
    if _norm_map(protected) != _norm_map(expected_protected):
        raise EvaluationError("protected source hashes differ from the completed repair")
    if _norm_map(shared) != _norm_map(expected_shared):
        raise EvaluationError("shared source hashes differ from the completed repair")
    if _norm_map(adapted) != _norm_map(expected_adapted):
        raise EvaluationError("adapted reference hashes differ from the completed repair")
    expected_new = repair.get("new_file_hashes_after_repair", {})
    current_new = {_repo_key(ROOT / path): file_sha256(ROOT / path) for path in expected_new}
    if _norm_map(current_new) != _norm_map(expected_new):
        raise EvaluationError("repaired source hashes differ from the completion record")
    return {
        "protected_source_hashes": protected,
        "shared_source_hashes": shared,
        "adapted_reference_hashes": adapted,
        "repair_new_file_hashes": current_new,
        "repair_record": REPAIR_INTEGRITY.resolve().as_posix(),
    }


def _validate_old_controls(registration: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {}
    for name, expected in EXPECTED_OLD_METADATA_HASHES.items():
        path = OLD_SCREEN_ROOT / name
        _must_file(path)
        actual = file_sha256(path)
        if actual != expected:
            raise EvaluationError(f"archived control metadata hash mismatch: {name}")
        metadata[name] = {"path": path.resolve().as_posix(), "sha256": actual}
    screen_registration = _read_json(OLD_SCREEN_ROOT / "registration.json")
    screen_summary = _read_json(OLD_SCREEN_ROOT / "summary.json")
    screen_integrity = _read_json(OLD_SCREEN_ROOT / "integrity.json")
    if (screen_registration.get("task_id") != "CPA_SIGNED_E2E_CELL_SCREEN_20260916"
            or screen_registration.get("status") != "REGISTERED"
            or screen_summary.get("status") != "COMPLETED"
            or screen_integrity.get("protected_sources_unchanged") is not True
            or screen_integrity.get("inputs_unchanged") is not True):
        raise EvaluationError("archived screen control registration is incomplete")
    for name, path in (("original_separator_direct", OLD_DIRECT_CANDIDATE),
                       ("original_separator_cell", OLD_CELL_CANDIDATE)):
        _must_file(path)
        actual = file_sha256(path)
        if actual != EXPECTED_CONTROL_HASHES[name]:
            raise EvaluationError(f"archived control candidate hash mismatch: {name}")
        runner._validate_candidate(path, actual)
    profiles = screen_summary.get("endpoint_profiles")
    if (not isinstance(profiles, dict) or not isinstance(profiles.get("x0"), dict)
            or not isinstance(profiles.get("B_H1_high"), dict)
            or not isinstance(profiles["x0"].get("H1"), dict)
            or not isinstance(profiles["B_H1_high"].get("H1"), dict)):
        raise EvaluationError("archived control endpoint profiles are incomplete")
    return {
        "metadata": metadata,
        "candidate_hashes": {name: EXPECTED_CONTROL_HASHES[name] for name in EXPECTED_CONTROL_HASHES},
        "paths": {
            "original_separator_direct": OLD_DIRECT_CANDIDATE.resolve().as_posix(),
            "original_separator_cell": OLD_CELL_CANDIDATE.resolve().as_posix(),
        },
        "screen_registration": screen_registration,
        "screen_summary": screen_summary,
        "screen_integrity": screen_integrity,
        "profiles": {
            "original_separator_direct": profiles["x0"]["H1"],
            "original_separator_cell": profiles["B_H1_high"]["H1"],
        },
    }


def _validate_stage_result(public_root: Path, stage: str) -> dict[str, Any]:
    if not public_root.is_dir():
        raise EvaluationError("new public output root is missing")
    if stage == "SEPARATE":
        source_registration, source, target = runner._load_sources(public_root)
        result = _read_json(public_root / "result.json")
        if result.get("status") != "COMPLETED" or result.get("stage") != "SEPARATE":
            raise EvaluationError("separation result is incomplete")
        if result.get("target_z_exact_abs_S_hat") is not True or not torch.equal(target, source.abs()):
            raise EvaluationError("separation target is not exact abs(S_hat)")
        return {"registration": source_registration, "result": result,
                "source_sha256": file_sha256(public_root / "sources.pt"),
                "target_z_sha256": runner.tensor_sha256(target)}
    if stage == "INVERT":
        source_root = public_root
        invert_root = public_root / "direct_fia"
        source_registration, _, target = runner._load_sources(source_root)
        registration = _read_json(invert_root / "registration.json")
        result = _read_json(invert_root / "result.json")
        if (registration.get("task_id") != IMPLEMENTATION_TASK_ID
                or registration.get("stage") != "INVERT"
                or registration.get("status") != "COMPLETED"
                or registration.get("fia_config") != runner.CANONICAL_FIA_CONFIG
                or result.get("status") != "COMPLETED"
                or result.get("target_z_sha256") != runner.tensor_sha256(target)
                or result.get("candidate_sha256") != registration.get("candidate_sha256")):
            raise EvaluationError("inversion result/config/target binding is incomplete")
        if result.get("direct_fia", {}).get("settings", {}).get("n_iter") != 25000:
            raise EvaluationError("Direct FIA did not record the registered 25000 steps")
        candidate = runner._validate_candidate(invert_root / "candidate.pt",
                                               registration.get("candidate_sha256"))
        return {"registration": registration, "result": result,
                "candidate_sha256": file_sha256(invert_root / "candidate.pt"),
                "candidate_shape": list(candidate.shape), "source_registration": source_registration}
    if stage == "REFINE":
        refine_root = public_root / "actual_update_refinement"
        registration = _read_json(refine_root / "registration.json")
        result = _read_json(refine_root / "result.json")
        source_registration, _, target = runner._load_sources(public_root)
        candidate_registration = _read_json(public_root / "direct_fia/registration.json")
        if (registration.get("task_id") != IMPLEMENTATION_TASK_ID
                or registration.get("stage") != "REFINE"
                or registration.get("status") != "COMPLETED"
                or registration.get("method") != runner.METHOD
                or registration.get("arm") != runner.REFINE_ARM_NAME
                or registration.get("label_hypothesis", {}).get("name") != "H1"
                or result.get("status") != "COMPLETED"
                or result.get("actual_update_result", {}).get("iterations") != 100
                or result.get("actual_update_result", {}).get("arm") != runner.REFINE_ARM_NAME
                or result.get("actual_update_result", {}).get("optimized_label_hypothesis") != "H1"
                or result.get("registration", {}).get("candidate_sha256") != candidate_registration.get("candidate_sha256")
                or result.get("registration", {}).get("target_z_sha256") != runner.tensor_sha256(target)):
            raise EvaluationError("refinement result identity/config/target binding is incomplete")
        preflight = runner._validate_preflight(result.get("preflight"))
        start_profile = runner._validate_endpoint_profile(result.get("start_profile"))
        final_profile = runner._validate_endpoint_profile(result.get("final_profile"))
        if result.get("feasibility", {}).get("software_completion") != "COMPLETED":
            raise EvaluationError("refinement software completion is not authenticated")
        final_path = refine_root / runner.REFINE_ARM_NAME / "candidate.pt"
        final_hash = result.get("actual_update_result", {}).get("candidate_sha256")
        runner._validate_candidate(final_path, final_hash)
        for profile in (start_profile, final_profile):
            if set(profile["per_tensor"]) != set(runner.FC_NAMES):
                raise EvaluationError("endpoint profile does not cover all six FC tensors")
        return {"registration": registration, "result": result, "preflight": preflight,
                "start_profile": start_profile, "final_profile": final_profile,
                "final_candidate_sha256": file_sha256(final_path),
                "source_registration": source_registration}
    raise EvaluationError(f"unknown public stage: {stage}")


def validate_public_gate(*, registration_path: Path, public_output: Path) -> dict[str, Any]:
    """Authenticate every public artifact before private deserialization."""
    registration = _read_json(registration_path)
    if (registration.get("task_id") != TASK_ID or registration.get("plan_version") != PLAN_VERSION
            or registration.get("status") != "REGISTERED"
            or registration.get("implementation_task_id") != IMPLEMENTATION_TASK_ID
            or registration.get("method") != runner.METHOD):
        raise EvaluationError("outer experiment registration identity is invalid")
    if registration.get("stage_order") != ["SEPARATE", "INVERT", "REFINE"]:
        raise EvaluationError("outer stage order is not the registered sequence")
    if registration.get("public_output_root") != public_output.resolve().as_posix():
        raise EvaluationError("outer public output binding changed")
    if public_output.resolve().exists() is False:
        raise EvaluationError("new public output root is missing")
    if _norm_map(registration.get("public_input_hashes", {})) != _norm_map(_public_input_hashes()):
        raise EvaluationError("public input hash changed before private evaluation")
    repair = _repair_integrity_snapshot()
    if _norm_map(registration.get("integrity_before", {}).get("protected_source_hashes", {})) != _norm_map(repair["protected_source_hashes"]):
        raise EvaluationError("registered protected-source hashes differ")
    if _norm_map(registration.get("integrity_before", {}).get("shared_source_hashes", {})) != _norm_map(repair["shared_source_hashes"]):
        raise EvaluationError("registered shared-source hashes differ")
    code_before = registration.get("code_hashes")
    if _norm_map(code_before or {}) != _norm_map(_code_hashes()):
        raise EvaluationError("experiment/evaluator code hash changed before private evaluation")
    controls = _validate_old_controls(registration)
    separate = _validate_stage_result(public_output, "SEPARATE")
    invert = _validate_stage_result(public_output, "INVERT")
    refine = _validate_stage_result(public_output, "REFINE")
    bound = registration.get("bound_artifacts", {})
    if (not isinstance(bound, Mapping)
            or bound.get("source_artifact_sha256") != separate["source_sha256"]):
        raise EvaluationError("outer source artifact hash binding changed")
    if bound.get("target_z_sha256") != separate["target_z_sha256"]:
        raise EvaluationError("outer target hash binding changed")
    if bound.get("direct_candidate_sha256") != invert["candidate_sha256"]:
        raise EvaluationError("outer direct candidate hash binding changed")
    if bound.get("cell_candidate_sha256") != refine["final_candidate_sha256"]:
        raise EvaluationError("outer cell candidate hash binding changed")
    after_protected = hash_files(protected_paths())
    after_shared = hash_files(runner.SHARED_PATHS)
    if _norm_map(after_protected) != _norm_map(repair["protected_source_hashes"]):
        raise EvaluationError("protected sources changed during public execution")
    if _norm_map(after_shared) != _norm_map(repair["shared_source_hashes"]):
        raise EvaluationError("shared sources changed during public execution")
    return {
        "task_id": TASK_ID, "status": "PUBLIC_GATE_PASSED",
        "public_output": public_output.resolve().as_posix(),
        "implementation_task_id": IMPLEMENTATION_TASK_ID,
        "stages": {"separate": separate, "invert": invert, "refine": refine},
        "controls": controls,
        "code_hashes": _code_hashes(),
        "integrity": {"protected_source_hashes": after_protected,
                       "shared_source_hashes": after_shared,
                       "public_input_hashes": _public_input_hashes(),
                       "unchanged": True},
        "private_loaded": False,
    }


def _load_truth_x(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    if path.resolve() != PRIVATE_TRUTH.resolve():
        raise EvaluationError("private truth path is outside the registered baseline observation")
    _must_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "x" not in payload:
        raise EvaluationError("private truth bundle lacks x")
    x = payload["x"]
    if (not isinstance(x, torch.Tensor) or x.dtype != torch.float32
            or tuple(x.shape) != (IMAGE_COUNT, 3, 224, 224)
            or not bool(torch.isfinite(x).all())):
        raise EvaluationError("private truth x has invalid FP32 [16,3,224,224] schema")
    return x.detach().contiguous(), {
        "private_loaded": True,
        "consumed_field": "x only",
        "labels_consumed": False,
        "true_updates_consumed": False,
        "bundled_fields_may_materialize": sorted(payload.keys()),
    }


def _candidate_paths(public_output: Path) -> dict[str, Path]:
    return {
        "original_separator_direct": OLD_DIRECT_CANDIDATE,
        "original_separator_cell": OLD_CELL_CANDIDATE,
        "combination_separator_direct": public_output / "direct_fia/candidate.pt",
        "combination_separator_cell": public_output / f"actual_update_refinement/{runner.REFINE_ARM_NAME}/candidate.pt",
    }


def _lpips_metrics(candidates: dict[str, torch.Tensor], truth_x: torch.Tensor,
                   image_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from .report_full_update_iterations import _make_lpips_metric
    metric, provenance = _make_lpips_metric()
    mean = truth_x.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = truth_x.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    originals = (truth_x * std + mean).clamp(0, 1)
    output = {}
    for name, report in image_reports.items():
        values = []
        for pair in report["per_image"]:
            i, j = pair["candidate_index"], pair["truth_index"]
            metric.reset()
            score = metric((candidates[name][i:i + 1] - .5) * 2,
                           (originals[j:j + 1] - .5) * 2)
            if score.numel() != 1 or not bool(torch.isfinite(score).all()):
                raise EvaluationError(f"nonfinite LPIPS score for {name}/{i}")
            pair["lpips"] = float(score.detach())
            values.append(pair["lpips"])
        report["mean_lpips"] = sum(values) / len(values)
        report["lpips_provenance"] = provenance
        output[name] = {"mean_lpips": report["mean_lpips"], "per_image": values}
    return {"metrics": output, "provenance": provenance}


def _aligned_candidates(candidates: dict[str, torch.Tensor], reports: dict[str, dict[str, Any]]) -> dict[str, torch.Tensor]:
    aligned = {}
    for name, values in candidates.items():
        ordered = torch.empty_like(values)
        for pair in reports[name]["per_image"]:
            ordered[pair["truth_index"]] = values[pair["candidate_index"]]
        aligned[name] = ordered
    return aligned


def _write_figure(output: Path, candidates: dict[str, torch.Tensor], reports: dict[str, dict[str, Any]], truth_x: torch.Tensor) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mean = truth_x.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = truth_x.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    originals = (truth_x * std + mean).clamp(0, 1)
    aligned = _aligned_candidates(candidates, reports)
    names = ["truth"] + list(aligned)
    columns = [originals] + [aligned[name] for name in aligned]
    figure, axes = plt.subplots(IMAGE_COUNT, len(names), figsize=(3 * len(names), 2.4 * IMAGE_COUNT))
    axes = axes.reshape(IMAGE_COUNT, len(names))
    try:
        for row in range(IMAGE_COUNT):
            for col, values in enumerate(columns):
                axes[row, col].imshow(values[row].permute(1, 2, 0).clamp(0, 1))
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(names[col])
        figure.tight_layout()
        path = output / "all16_comparison.png"
        figure.savefig(path, dpi=110)
    finally:
        plt.close(figure)
    return path.name


def _means(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    output = {}
    for name, report in reports.items():
        rows = report["per_image"]
        finite_psnr = [row["psnr_db"] for row in rows if row["psnr_db"] is not None]
        output[name] = {
            "mean_mse": sum(row["mse"] for row in rows) / IMAGE_COUNT,
            "mean_psnr_db": sum(finite_psnr) / IMAGE_COUNT if len(finite_psnr) == IMAGE_COUNT else None,
            "mean_ssim": sum(row["ssim"] for row in rows) / IMAGE_COUNT,
            "mean_lpips": sum(row["lpips"] for row in rows) / IMAGE_COUNT,
        }
    return output


def _control_reproduction(means: Mapping[str, Mapping[str, float | None]]) -> dict[str, Any]:
    prior = _read_json(OLD_PRIVATE_COMPARISON)
    prior_means = prior.get("mean_metrics", {})
    mapping = {"original_separator_direct": "x0", "original_separator_cell": "B_H1_high"}
    details = {}
    passed = True
    for current_name, old_name in mapping.items():
        expected = prior_means.get(old_name)
        actual = means.get(current_name)
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            raise EvaluationError("archived private comparison lacks control mean metrics")
        differences = {metric: (None if expected.get(metric) is None or actual.get(metric) is None
                                else actual[metric] - expected[metric]) for metric in METRIC_NAMES}
        within = {metric: (differences[metric] is not None
                           and abs(float(differences[metric])) <= CONTROL_REPRODUCTION_TOLERANCE)
                  for metric in METRIC_NAMES}
        passed = passed and all(within.values())
        details[current_name] = {"prior_row": old_name, "prior": expected,
                                 "current": actual, "difference": differences,
                                 "within_tolerance": within}
    return {"status": "PASSED" if passed else "NEEDS_REVIEW",
            "tolerance": {metric: CONTROL_REPRODUCTION_TOLERANCE for metric in METRIC_NAMES},
            "details": details,
            "scientific_conclusion_allowed": passed}


def _contrasts(means: Mapping[str, Mapping[str, float | None]]) -> dict[str, Any]:
    pairs = {
        "combination_direct_vs_original_direct": ("combination_separator_direct", "original_separator_direct"),
        "combination_cell_vs_combination_direct": ("combination_separator_cell", "combination_separator_direct"),
        "combination_cell_vs_original_cell": ("combination_separator_cell", "original_separator_cell"),
    }
    output = {}
    for name, (left, right) in pairs.items():
        output[name] = {metric: (None if means[left][metric] is None or means[right][metric] is None
                                 else means[left][metric] - means[right][metric])
                        for metric in METRIC_NAMES}
    return {"differences_are_left_minus_right": output,
            "favorable_directions": {"mean_mse": "lower", "mean_psnr_db": "higher",
                                      "mean_ssim": "higher", "mean_lpips": "lower"},
            "interpretation": "paired descriptive contrasts for this fixed batch; no threshold or generalization claim"}


def _profile_summary(gate: Mapping[str, Any]) -> dict[str, Any]:
    controls = gate["controls"]["profiles"]
    refine = gate["stages"]["refine"]
    return {
        "original_separator_direct": controls["original_separator_direct"],
        "original_separator_cell": controls["original_separator_cell"],
        "combination_separator_direct": refine["start_profile"],
        "combination_separator_cell": refine["final_profile"],
        "exact_feasibility_definition": "zero fixed-observed-scale code mismatches and zero bounded-cell violations over all six FC tensors",
    }


def evaluate(*, registration_path: Path, public_output: Path,
             private_truth: Path = PRIVATE_TRUTH, output: Path = PRIVATE_ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing preexisting private evaluation output: {output}")
    gate = validate_public_gate(registration_path=registration_path.resolve(),
                                public_output=public_output.resolve())
    # This is the only private phase boundary.  No private tensor is loaded by
    # validate_public_gate or the control authentication above.
    truth_x, private_meta = _load_truth_x(private_truth.resolve())
    output.mkdir(parents=True, exist_ok=False)
    paths = _candidate_paths(public_output.resolve())
    candidates = {}
    image_reports = {}
    candidate_records = {}
    for name in CONTROL_NAMES:
        path = paths[name]
        _must_file(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != {"pixels"}:
            raise EvaluationError(f"candidate schema is not exact for {name}")
        pixels = payload["pixels"]
        if (not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.float32
                or tuple(pixels.shape) != (IMAGE_COUNT, 3, 224, 224)
                or not bool(torch.isfinite(pixels).all())
                or bool((pixels < 0).any()) or bool((pixels > 1).any())):
            raise EvaluationError(f"candidate value schema is invalid for {name}")
        candidates[name] = pixels.detach().contiguous()
        candidate_records[name] = {"path": path.resolve().as_posix(),
                                   "file_sha256": file_sha256(path)}
        image_reports[name] = evaluate_candidate(path, private_truth.resolve(), include_labels=False)
    lpips = _lpips_metrics(candidates, truth_x, image_reports)
    means = _means(image_reports)
    control_reproduction = _control_reproduction(means)
    figure = _write_figure(output, candidates, image_reports, truth_x)
    protected_after = hash_files(protected_paths())
    shared_after = hash_files(runner.SHARED_PATHS)
    public_inputs_after = _public_input_hashes()
    registered = _read_json(registration_path)
    integrity_ok = (
        _norm_map(protected_after) == _norm_map(gate["integrity"]["protected_source_hashes"])
        and _norm_map(shared_after) == _norm_map(gate["integrity"]["shared_source_hashes"])
        and _norm_map(public_inputs_after) == _norm_map(registered.get("public_input_hashes", {}))
    )
    if not integrity_ok:
        raise EvaluationError("protected source or public input changed during offline evaluation")
    status = "COMPLETED" if control_reproduction["status"] == "PASSED" else "NEEDS_REVIEW"
    comparison = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": status,
        "evaluation_status": status, "implementation_task_id": IMPLEMENTATION_TASK_ID,
        "public_output": public_output.resolve().as_posix(),
        "private_truth_path": private_truth.resolve().as_posix(),
        "private_output": output.as_posix(), "labels_consumed": False,
        "true_updates_consumed": False, "private_truth_metadata": private_meta,
        "matching": "Hungarian_pixel_MSE_once_per_candidate_reused_for_SSIM_LPIPS_and_aligned_figure",
        "all_images_included": True, "candidate_order": list(CONTROL_NAMES),
        "candidates": candidate_records, "image_metrics": image_reports,
        "mean_metrics": means, "lpips": lpips,
        "control_metric_reproduction": control_reproduction,
        "predeclared_contrasts": _contrasts(means),
        "endpoint_profiles": _profile_summary(gate),
        "public_gate": gate, "figure_artifacts": [figure],
        "source_integrity": {"protected_source_hashes": protected_after,
                              "shared_source_hashes": shared_after,
                              "public_input_hashes": public_inputs_after,
                              "unchanged": True},
        "wall_seconds": time.perf_counter() - started,
        "scientific_quality": "EVALUATED" if status == "COMPLETED" else "NEEDS_REVIEW",
        "limitations": [
            "one fixed batch and seed",
            "separator iteration/optimizer packages differ, so this is not an equal-compute or single-prior causal ablation",
            "new coverage-derived H1 label slots may differ from archived controls",
            "exact cell feasibility is reported separately from image quality",
        ],
    }
    write_json(output / "comparison.json", comparison)
    write_json(output / "results.json", comparison)
    lines = [f"# {TASK_ID}", "", f"Status: **{status}**.", "",
             "The public provenance gate passed before private truth loading. "
             "The evaluator consumed x only; bundled labels and true updates were not consumed.", "",
             "| Set | MSE | PSNR dB | SSIM | LPIPS |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in CONTROL_NAMES:
        values = means[name]
        lines.append(f"| {name} | {values['mean_mse']:.12g} | "
                     f"{values['mean_psnr_db'] if values['mean_psnr_db'] is not None else 'null'} | "
                     f"{values['mean_ssim']:.12g} | {values['mean_lpips']:.12g} |")
    lines += ["", "Control reproduction: **" + control_reproduction["status"] + "**.",
              "Endpoint code/cell profiles for all six FC tensors are embedded in `comparison.json`.",
              "The aligned figure reuses each set's one pixel-MSE Hungarian correspondence."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", dest="registration_path", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--private-truth", type=Path, default=PRIVATE_TRUTH)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = evaluate(**vars(args))
    except Exception as error:
        print(f"{TASK_ID}: {type(error).__name__}: {error}", flush=True)
        return 1
    print(json.dumps({"task_id": TASK_ID, "status": result["status"],
                      "evaluation_status": result["evaluation_status"],
                      "wall_seconds": result["wall_seconds"]}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
