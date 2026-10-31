"""Finalize the saved H1 FIA-removal candidates without running optimization.

This entry point is intentionally a saved-artifact reader.  It authenticates
the failed parent's public receipt, recomputes the three endpoint profiles,
then opens the private image tensor only after the public gate has passed.
It has no optimizer, arm runner, or continuation path.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import torch

from .full_update_evaluation import evaluate_candidate
from .signed_baseline_observation import CHECKPOINT_SHA
from .signed_e2e_cells import load_public_cell_tables
from .signed_e2e_nofia_evaluation import (
    _load_private_images, _lpips, _means, _prior_control_means,
    _write_figure, align_candidates_to_truth,
)
from .signed_e2e_nofia_runner import (
    CHECKPOINT, CONTROL_ARM, H1_LABELS, LAMBDA_CELL, LAMBDA_FIA, NAIVE_ROOT,
    NEW_ARM, PREVIOUS_PRIVATE_ROOT, PREVIOUS_PUBLIC_ROOT, PUBLIC_ROOT, REPO_ROOT,
    PLAN_VERSION, TASK_ID, ExperimentError, _endpoint_profile, _hash_map_differences,
    _normalize_hash_key, _read_json, _tensor_sha, authenticate_previous_run,
    compare_canonical_profiles, file_sha256, hash_files, hash_public_inputs,
    load_model, load_public_start, protected_paths, validate_structured_provenance,
    write_json,
)
from .signed_e2e_runner import (
    MAX_CUDA_ALLOCATED, REPO_ROOT as SHARED_REPO_ROOT, TV_WEIGHT, _rel,
)
from .signed_e2e_nofia_runner import _device_tables
from .signed_e2e_runner import EXPECTED_CHECKPOINT


FINALIZATION_TASK_ID = "CPA_SIGNED_E2E_H1_NOFIA_FINALIZE_20260916"
FINAL_REPORT_ROOT = REPO_ROOT / ".ai/execution-results/CPA_SIGNED_E2E_H1_NOFIA_FINALIZE_20260916"
PARENT_PUBLIC_ROOT = REPO_ROOT / "quantization_aware_cpa/results/signed_e2e_h1_nofia_B16_T10_20260916"
PRIVATE_TRUTH = REPO_ROOT / (
    "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/"
    "observation/evaluation_truth.pt"
)
PRIVATE_OUTPUT_ROOT = REPO_ROOT / "quantization_aware_cpa/results_private/signed_e2e_h1_nofia_B16_T10_20260916"
EXPECTED_NEW_CANDIDATE_SHA = "5a20b87bf69473bf7696e46d9054c0f3d04a4c02f98ffde6ee3969374bf4e525"
EXPECTED_CONTROL_CANDIDATE_SHA = "b352c3bde7db87ff00ecbbbd4715c5c865ada662c4dcf88b1e116f5369ec9d5a"
PARENT_FAILED_SUMMARY_SHA = "379d79565906075db58da9fa4695e8adda907eb2d88a9ec59872443b8ba224be"
PARENT_REAL_SECONDS = 155.46602060005534
SOURCE_BEFORE = FINAL_REPORT_ROOT / "before/source_hashes.json"
PARENT_ARCHIVE = FINAL_REPORT_ROOT / "before/parent_public"

SHARED_PATHS = [
    REPO_ROOT / "quantization_aware_cpa/qacpa/signed_e2e_cells.py",
    REPO_ROOT / "quantization_aware_cpa/qacpa/signed_e2e_runner.py",
    REPO_ROOT / "quantization_aware_cpa/qacpa/fc_replay.py",
    REPO_ROOT / "quantization_aware_cpa/qacpa/full_update_replay.py",
    REPO_ROOT / "quantization_aware_cpa/qacpa/full_update_adjoint.py",
]
PARENT_FILES = [
    "registration.json", "full_size_numeric_check.json", "initial_calibration_comparison.json",
    "x0_candidate.pt", "x0_metadata.json", "failure.json", "summary.json",
    f"{NEW_ARM}/candidate.pt", f"{NEW_ARM}/attack_state.pt", f"{NEW_ARM}/result.json",
    f"{NEW_ARM}/optimization_log.json",
]


class FinalizationError(RuntimeError):
    """A saved-candidate provenance, public gate, or evaluation check failed."""


def _must_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise FinalizationError(f"required regular file is missing: {path}")


def _read_json_any(path: Path) -> Any:
    _must_file(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalizationError(f"invalid UTF-8 JSON: {path}: {error}") from error


def _validate_pixels(value: Any, name: str) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or tuple(value.shape) != (16, 3, 224, 224)
            or not bool(torch.isfinite(value).all())
            or bool((value < 0).any()) or bool((value > 1).any())):
        raise FinalizationError(f"{name} must be finite FP32 [16,3,224,224] pixels")
    return value.contiguous()


def _load_candidate(path: Path, expected_sha: str | None = None) -> torch.Tensor:
    _must_file(path)
    actual_sha = file_sha256(path)
    if expected_sha is not None and actual_sha != expected_sha:
        raise FinalizationError(
            f"candidate hash mismatch: {path}: expected {expected_sha}, actual {actual_sha}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"pixels"}:
        raise FinalizationError(f"candidate schema mismatch: {path}")
    return _validate_pixels(payload["pixels"], str(path))


def _hashes_match(expected: Any, actual: Any, label: str) -> bool:
    return not _hash_map_differences(expected, actual, label)


def _validate_saved_arm_result(result: Any, log: Any, *, expected_arm: str,
                               expected_hash: str, expected_lambda_fia: float) -> dict[str, Any]:
    """Validate the immutable result/log contract used by finalization tests and runtime."""
    errors: list[dict[str, Any]] = []
    if not isinstance(result, dict):
        errors.append({"path": "result", "expected": "object", "actual": type(result).__name__})
        result = {}
    if result.get("status") != "COMPLETED":
        errors.append({"path": "result.status", "expected": "COMPLETED", "actual": result.get("status")})
    if result.get("arm") != expected_arm:
        errors.append({"path": "result.arm", "expected": expected_arm, "actual": result.get("arm")})
    if result.get("iterations") != 100 or type(result.get("iterations")) is not int:
        errors.append({"path": "result.iterations", "expected": 100, "actual": result.get("iterations")})
    for key, expected in (("lambda_fia", expected_lambda_fia), ("lambda_tv", TV_WEIGHT),
                          ("lambda_cell", LAMBDA_CELL)):
        if result.get(key) != expected:
            errors.append({"path": f"result.{key}", "expected": expected, "actual": result.get(key)})
    if result.get("candidate_sha256") != expected_hash:
        errors.append({"path": "result.candidate_sha256", "expected": expected_hash,
                       "actual": result.get("candidate_sha256")})
    if not isinstance(log, list) or len(log) != 100:
        errors.append({"path": "optimization_log", "expected": "100 rows", 
                       "actual": len(log) if isinstance(log, list) else type(log).__name__})
    else:
        iterations = [row.get("iteration") for row in log if isinstance(row, dict)]
        if iterations != list(range(1, 101)):
            errors.append({"path": "optimization_log.iteration", "expected": list(range(1, 101)),
                           "actual": iterations})
        for row_number, row in enumerate(log, 1):
            if not isinstance(row, dict) or row.get("lambda_fia") != expected_lambda_fia:
                errors.append({"path": f"optimization_log[{row_number}].lambda_fia",
                               "expected": expected_lambda_fia,
                               "actual": row.get("lambda_fia") if isinstance(row, dict) else type(row).__name__})
                break
    if errors:
        raise FinalizationError("saved arm result/log contract failed: " +
                                json.dumps(errors, ensure_ascii=False, sort_keys=True))
    return {"passed": True, "arm": expected_arm, "iterations": 100,
            "candidate_sha256": expected_hash, "log_rows": 100}


def _validate_public_gate(integrity: dict[str, Any]) -> None:
    """Keep private loading behind an explicit, testable public completion gate."""
    required = ("protected_sources_unchanged", "shared_sources_unchanged",
                "inputs_unchanged", "previous_control_files_unchanged",
                "parent_artifacts_unchanged", "endpoint_profiles_authenticated")
    failed = [key for key in required if integrity.get(key) is not True]
    if failed:
        raise FinalizationError("public gate failed: " + ", ".join(failed))


def _verify_parent_artifacts() -> dict[str, Any]:
    if PARENT_PUBLIC_ROOT.resolve() != (REPO_ROOT /
            "quantization_aware_cpa/results/signed_e2e_h1_nofia_B16_T10_20260916").resolve():
        raise FinalizationError("unexpected parent public output root")
    for relative in PARENT_FILES:
        _must_file(PARENT_PUBLIC_ROOT / relative)
        archived = PARENT_ARCHIVE / relative
        _must_file(archived)
        actual_sha, archived_sha = file_sha256(PARENT_PUBLIC_ROOT / relative), file_sha256(archived)
        if actual_sha != archived_sha:
            raise FinalizationError(f"preserved parent artifact changed: {relative}: {actual_sha} != {archived_sha}")
    if file_sha256(PARENT_PUBLIC_ROOT / "summary.json") != PARENT_FAILED_SUMMARY_SHA:
        raise FinalizationError("parent failed summary is no longer the archived byte sequence")
    failure = _read_json(PARENT_PUBLIC_ROOT / "failure.json")
    summary = _read_json(PARENT_PUBLIC_ROOT / "summary.json")
    if (summary.get("task_id") != TASK_ID or summary.get("status") != "FAILED"
            or summary.get("error") != "archived B_H1_high endpoint profile mismatch"):
        raise FinalizationError("parent failure evidence is not the registered validator failure")
    if failure != summary:
        raise FinalizationError("parent failure.json and failed summary differ")
    return {"summary": summary, "failure": failure,
            "summary_sha256": file_sha256(PARENT_PUBLIC_ROOT / "summary.json"),
            "archived_parent_artifacts": {relative: file_sha256(PARENT_ARCHIVE / relative)
                                           for relative in PARENT_FILES}}


def _verify_parent_registration() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], torch.Tensor,
                                               torch.Tensor, torch.Tensor, dict[str, Any]]:
    parent = _read_json(PARENT_PUBLIC_ROOT / "registration.json")
    numeric = _read_json(PARENT_PUBLIC_ROOT / "full_size_numeric_check.json")
    calibration = _read_json(PARENT_PUBLIC_ROOT / "initial_calibration_comparison.json")
    x0, source, target = load_public_start(NAIVE_ROOT)
    if parent.get("task_id") != TASK_ID or parent.get("plan_version") != PLAN_VERSION:
        raise FinalizationError("parent registration task/version mismatch")
    if parent.get("status") != "REGISTERED" or parent.get("new_arm") != NEW_ARM:
        raise FinalizationError("parent registration is not the signed noFIA plan")
    if parent.get("control_arm_reused") != CONTROL_ARM or parent.get("control_retrained") is not False:
        raise FinalizationError("parent control provenance is not reuse-only")
    if parent.get("h1_labels") != list(H1_LABELS) or parent.get("h1_labels_are_hypotheses") is not True:
        raise FinalizationError("parent H1 labels changed")
    objective = parent.get("objective", {})
    if (objective.get("lambda_fia") != LAMBDA_FIA or objective.get("lambda_tv") != TV_WEIGHT
            or objective.get("lambda_cell") != LAMBDA_CELL):
        raise FinalizationError("parent objective coefficients changed")
    if numeric.get("terminal_deltas_torch_equal") is not True or numeric.get("feature_grad_enabled_equals_no_grad") is not True:
        raise FinalizationError("parent numerical preflight did not pass")
    if (calibration.get("current", {}).get("matches_previous_cell_loss_exact") is not True
            or calibration.get("current", {}).get("matches_previous_feature_norm_exact") is not True):
        raise FinalizationError("parent calibration comparison did not pass")
    recorded = parent.get("source_and_target_hashes", {})
    actual_source_hashes = {"x0_pixels": _tensor_sha(x0), "S_hat": _tensor_sha(source),
                            "target_z": _tensor_sha(target)}
    if recorded != actual_source_hashes:
        raise FinalizationError("parent source/target tensor hashes changed: " +
                                json.dumps({"expected": recorded, "actual": actual_source_hashes}, sort_keys=True))
    actual_inputs = hash_public_inputs(PUBLIC_ROOT, NAIVE_ROOT, CHECKPOINT)
    actual_protected = hash_files(protected_paths())
    actual_shared = hash_files(SHARED_PATHS)
    differences = []
    differences += _hash_map_differences(parent.get("input_hashes"), actual_inputs, "parent.input_hashes")
    differences += _hash_map_differences(parent.get("protected_source_hashes_before"), actual_protected,
                                         "parent.protected_source_hashes_before")
    differences += _hash_map_differences(parent.get("shared_source_hashes_before"), actual_shared,
                                         "parent.shared_source_hashes_before")
    if differences:
        raise FinalizationError("parent public provenance hash mismatch: " +
                                json.dumps(differences, ensure_ascii=False, sort_keys=True))
    result = _read_json(PARENT_PUBLIC_ROOT / NEW_ARM / "result.json")
    log = _read_json_any(PARENT_PUBLIC_ROOT / NEW_ARM / "optimization_log.json")
    arm_contract = _validate_saved_arm_result(result, log, expected_arm=NEW_ARM,
                                              expected_hash=EXPECTED_NEW_CANDIDATE_SHA,
                                              expected_lambda_fia=LAMBDA_FIA)
    state = torch.load(PARENT_PUBLIC_ROOT / NEW_ARM / "attack_state.pt", map_location="cpu", weights_only=True)
    if (not isinstance(state, dict) or set(state) != {"pixels", "labels", "lambda_fia", "lambda_tv",
                                                        "lambda_cell", "completed_iterations"}
            or state.get("lambda_fia") != LAMBDA_FIA or state.get("lambda_tv") != TV_WEIGHT
            or state.get("lambda_cell") != LAMBDA_CELL or state.get("completed_iterations") != 100):
        raise FinalizationError("parent attack state provenance/schema mismatch")
    new_pixels = _validate_pixels(state.get("pixels"), "saved noFIA state pixels")
    state_labels = state.get("labels")
    if (not isinstance(state_labels, torch.Tensor) or state_labels.dtype != torch.int64
            or not torch.equal(state_labels, torch.tensor(H1_LABELS, dtype=torch.int64))):
        raise FinalizationError("parent attack state H1 labels changed")
    candidate = _load_candidate(PARENT_PUBLIC_ROOT / NEW_ARM / "candidate.pt", EXPECTED_NEW_CANDIDATE_SHA)
    if not torch.equal(candidate, new_pixels):
        raise FinalizationError("parent candidate and attack state pixels differ")
    x0_candidate = _load_candidate(PARENT_PUBLIC_ROOT / "x0_candidate.pt")
    if not torch.equal(x0_candidate, x0):
        raise FinalizationError("saved x0 candidate differs from the registered naive pixels")
    x0_metadata = _read_json(PARENT_PUBLIC_ROOT / "x0_metadata.json")
    if x0_metadata.get("candidate_sha256") != file_sha256(PARENT_PUBLIC_ROOT / "x0_candidate.pt"):
        raise FinalizationError("saved x0 metadata file hash mismatch")
    return (parent, numeric, calibration, x0, source, target,
            {"inputs": actual_inputs, "protected": actual_protected, "shared": actual_shared,
             "arm_contract": arm_contract, "new_result": result, "new_log": log,
             "new_pixels": candidate, "parent_summary": _verify_parent_artifacts()})


def _diagnostic_comparison(actual: dict[str, Any], expected: dict[str, Any], *, name: str) -> dict[str, Any]:
    fields = ("feature_target_cosine_loss", "tv_normalized", "feature_norm_mean", "feature_norm_min",
              "feature_norm_max", "pixel_min", "pixel_max", "pixel_mean", "x0_displacement_l2")
    differences = []
    for field in fields:
        if field not in expected or field not in actual:
            continue
        left, right = float(actual[field]), float(expected[field])
        close = math.isclose(left, right, rel_tol=2e-7, abs_tol=2e-9)
        differences.append({"path": f"{name}.{field}", "expected": right, "actual": left,
                            "difference": left - right, "within_tolerance": close})
    return {"passed": all(item["within_tolerance"] for item in differences),
            "differences": differences, "float_tolerance": {"rel_tol": 2e-7, "abs_tol": 2e-9}}


def _source_change_record() -> dict[str, Any]:
    before = _read_json(SOURCE_BEFORE).get("sources", {})
    paths = {
        "quantization_aware_cpa/qacpa/signed_e2e_nofia_runner.py": REPO_ROOT / "quantization_aware_cpa/qacpa/signed_e2e_nofia_runner.py",
        "quantization_aware_cpa/qacpa/signed_e2e_nofia_evaluation.py": REPO_ROOT / "quantization_aware_cpa/qacpa/signed_e2e_nofia_evaluation.py",
        "quantization_aware_cpa/tests/test_signed_e2e_nofia.py": REPO_ROOT / "quantization_aware_cpa/tests/test_signed_e2e_nofia.py",
    }
    after = {key: file_sha256(path) for key, path in paths.items()}
    return {"before": before, "after": after,
            "changed": {key: before.get(key) != value for key, value in after.items()},
            "semantic_changes": [
                "structured UTF-8 JSON and recorded input/source hash authentication",
                "complete canonical profile schema with exact integer/bool/string checks and bounded float reports",
                "Hungarian correspondence is reused for the comparison figure",
                "saved-candidate finalization entry point and regression coverage",
            ]}


def _write_private_report(output: Path, comparison: dict[str, Any]) -> None:
    means = comparison["mean_metrics"]
    lines = [f"# {TASK_ID} finalization", "", "Status: **COMPLETED**.", "",
             "The parent wrapper exited 1 at its reporting gate after the C_H1_high_noFIA 100-step arm had completed. "
             "This finalization reused the saved x0, B_H1_high control, and C_H1_high_noFIA candidate. "
             "Private evaluation consumed image tensor `x` only; labels and true updates were not consumed.", "",
             "| Candidate | MSE | PSNR dB | SSIM | LPIPS |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in ("x0", CONTROL_ARM, NEW_ARM):
        values = means[name]
        lines.append(f"| {name} | {values['mean_mse']:.9g} | {values['mean_psnr_db']} | "
                     f"{values['mean_ssim']:.9g} | {values['mean_lpips']:.9g} |")
    lines.extend(["", "Metrics are descriptive for the fixed all16-image batch and do not change the registered decision rules.",
                  "The figure uses each candidate's recorded Hungarian pixel-MSE correspondence."])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_saved_candidates(*, report_root: Path = FINAL_REPORT_ROOT,
                              public_root: Path = PARENT_PUBLIC_ROOT,
                              private_truth: Path = PRIVATE_TRUTH,
                              private_output: Path = PRIVATE_OUTPUT_ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    report_root = report_root.resolve()
    public_root = public_root.resolve()
    private_output = private_output.resolve()
    if report_root != FINAL_REPORT_ROOT.resolve() or public_root != PARENT_PUBLIC_ROOT.resolve():
        raise FinalizationError("finalization roots must be the registered paths")
    if private_output.exists():
        raise FinalizationError(f"refusing preexisting private finalization output: {private_output}")
    stages: dict[str, dict[str, float]] = {}
    stage_start = time.perf_counter()
    archive = _verify_parent_artifacts()
    stages["parent_archive_and_failure_audit"] = {"wall_seconds": time.perf_counter() - stage_start}

    stage_start = time.perf_counter()
    parent, numeric, calibration, x0, source, target, parent_audit = _verify_parent_registration()
    previous = authenticate_previous_run()
    previous_provenance = previous["structured_provenance"]
    if not previous_provenance["passed"]:
        raise FinalizationError("archived control structured provenance failed")
    if previous["control_candidate_sha256"] != EXPECTED_CONTROL_CANDIDATE_SHA:
        raise FinalizationError("archived control candidate is not the registered artifact")
    control = _load_candidate(PREVIOUS_PUBLIC_ROOT / CONTROL_ARM / "candidate.pt", EXPECTED_CONTROL_CANDIDATE_SHA)
    previous_control_result = previous["control_result"]
    stages["structured_provenance_and_saved_candidate_audit"] = {"wall_seconds": time.perf_counter() - stage_start}

    stage_start = time.perf_counter()
    if file_sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise FinalizationError("checkpoint hash mismatch")
    configure_runtime = __import__("qacpa.deterministic_hybrid_setup", fromlist=["configure_runtime"]).configure_runtime
    configure_runtime(42, threads=1)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise FinalizationError("the registered runtime requires CUDA for endpoint replay")
    manifest, tables = load_public_cell_tables(PUBLIC_ROOT)
    model = load_model(CHECKPOINT, device)
    x0_device, target_device = x0.to(device), target.to(device)
    labels_device = torch.tensor(H1_LABELS, dtype=torch.int64, device=device)
    device_tables = _device_tables(tables, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    profiles: dict[str, dict[str, Any]] = {}
    for name, candidate_pixels in (("x0", x0), (CONTROL_ARM, control), (NEW_ARM, parent_audit["new_pixels"])):
        profiles[name] = _endpoint_profile(model, candidate_pixels.to(device), x0_device, target_device,
                                            labels_device, tables, device)
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        if peak_allocated > MAX_CUDA_ALLOCATED:
            raise FinalizationError(f"endpoint replay CUDA allocation cap exceeded: {peak_allocated}")
    else:
        peak_allocated = peak_reserved = 0
    stages["three_endpoint_replays"] = {"wall_seconds": time.perf_counter() - stage_start}

    control_profile_auth = compare_canonical_profiles(profiles[CONTROL_ARM], previous["control_endpoint_profile"])
    x0_profile_auth = compare_canonical_profiles(profiles["x0"], previous["x0_endpoint_profile"])
    control_diagnostics = _diagnostic_comparison(profiles[CONTROL_ARM], previous_control_result.get("final_metrics", {}),
                                                 name=CONTROL_ARM)
    if not control_profile_auth["passed"] or not x0_profile_auth["passed"] or not control_diagnostics["passed"]:
        mismatch = {"control_profile": control_profile_auth, "x0_profile": x0_profile_auth,
                    "control_diagnostics": control_diagnostics}
        write_json(report_root / "public_gate_mismatch.json", mismatch)
        raise FinalizationError("saved endpoint reproduction mismatch: " +
                                json.dumps(mismatch, ensure_ascii=False, sort_keys=True))
    for name, profile in profiles.items():
        schema_errors = __import__("qacpa.signed_e2e_nofia_runner", fromlist=["validate_canonical_profile"]).validate_canonical_profile(profile, name=name)
        if schema_errors:
            raise FinalizationError("new endpoint profile schema failed: " + json.dumps(schema_errors, sort_keys=True))
    endpoint_authentication = {
        "schema": {"canonical_root_fields": list(__import__("qacpa.signed_e2e_nofia_runner", fromlist=["CANONICAL_PROFILE_ROOT_FIELDS"]).CANONICAL_PROFILE_ROOT_FIELDS),
                    "canonical_tensor_fields": list(__import__("qacpa.signed_e2e_nofia_runner", fromlist=["CANONICAL_PROFILE_TENSOR_FIELDS"]).CANONICAL_PROFILE_TENSOR_FIELDS),
                    "allowed_root_diagnostics": list(__import__("qacpa.signed_e2e_nofia_runner", fromlist=["PROFILE_DIAGNOSTIC_FIELDS"]).PROFILE_DIAGNOSTIC_FIELDS)},
        "control": control_profile_auth, "x0": x0_profile_auth,
        "control_added_result_diagnostics": control_diagnostics,
        "archived_control_candidate_sha256": EXPECTED_CONTROL_CANDIDATE_SHA,
    }
    write_json(PARENT_PUBLIC_ROOT / "endpoint_profiles.json",
               {name: {"H1": profile} for name, profile in profiles.items()})
    write_json(PARENT_PUBLIC_ROOT / "archived_control_authentication.json", endpoint_authentication)

    current_protected = hash_files(protected_paths())
    current_shared = hash_files(SHARED_PATHS)
    current_inputs = hash_public_inputs(PUBLIC_ROOT, NAIVE_ROOT, CHECKPOINT)
    preserved_parent = {relative: file_sha256(PARENT_PUBLIC_ROOT / relative) ==
                        parent_audit["parent_summary"]["archived_parent_artifacts"][relative]
                        for relative in PARENT_FILES}
    final_integrity = {
        "finalization_task_id": FINALIZATION_TASK_ID,
        "parent_task_id": TASK_ID,
        "protected_sources_unchanged": _hashes_match(parent.get("protected_source_hashes_before"), current_protected, "protected"),
        "shared_sources_unchanged": _hashes_match(parent.get("shared_source_hashes_before"), current_shared, "shared"),
        "inputs_unchanged": _hashes_match(parent.get("input_hashes"), current_inputs, "inputs"),
        "previous_control_files_unchanged": previous["control_files_sha256"] ==
            {_path: file_sha256(Path(_path)) for _path in previous["control_files_sha256"]},
        "parent_artifacts_unchanged": all(preserved_parent.values()),
        "endpoint_profiles_authenticated": True,
        "hashes": {"parent_recorded_inputs": parent.get("input_hashes"), "current_inputs": current_inputs,
                    "parent_recorded_protected": parent.get("protected_source_hashes_before"),
                    "current_protected": current_protected, "parent_recorded_shared": parent.get("shared_source_hashes_before"),
                    "current_shared": current_shared, "parent_artifacts": preserved_parent},
        "candidate_hashes": {"x0_file": file_sha256(PARENT_PUBLIC_ROOT / "x0_candidate.pt"),
                             "B_H1_high": EXPECTED_CONTROL_CANDIDATE_SHA,
                             "C_H1_high_noFIA": EXPECTED_NEW_CANDIDATE_SHA},
    }
    # The finalization record is written before any private deserialization.
    write_json(PARENT_PUBLIC_ROOT / "finalization_integrity.json", final_integrity)
    public_summary = {
        "task_id": TASK_ID, "finalization_task_id": FINALIZATION_TASK_ID,
        "plan_version": PLAN_VERSION, "status": "COMPLETED",
        "stage": "SAVED_CANDIDATE_PUBLIC_FINALIZATION_COMPLETE",
        "recovery_provenance": {"parent_summary_status": "FAILED",
                                 "parent_failure": archive["failure"],
                                 "parent_failed_summary_sha256": PARENT_FAILED_SUMMARY_SHA,
                                 "parent_real_phase_seconds": PARENT_REAL_SECONDS},
        "candidates": {name: {"endpoint_profile": profiles[name]} for name in profiles},
        "endpoint_replay_count": {"this_finalization": 3, "parent_in_memory": 3, "total": 6},
        "endpoint_authentication": endpoint_authentication,
        "integrity": final_integrity,
        "private_evaluation": "PENDING_AFTER_PUBLIC_GATE",
        "resource": {"endpoint_peak_cuda_allocated_bytes": peak_allocated,
                      "endpoint_peak_cuda_reserved_bytes": peak_reserved,
                      "max_cuda_allocated_bytes": MAX_CUDA_ALLOCATED},
        "warnings": ["Parent optimization_log.json preserves pre_step_base_loss naming that includes diagnostic FIA despite lambda_fia=0.",
                     "Parent wrapper exit1 reflects the repaired endpoint schema gate, while its 100-step candidate remains immutable."],
    }
    write_json(PARENT_PUBLIC_ROOT / "finalized_summary.json", public_summary)
    _validate_public_gate(final_integrity)
    stages["public_authentication_and_artifact_write"] = {"wall_seconds": time.perf_counter() - stage_start}

    stage_start = time.perf_counter()
    private_output.mkdir(parents=True, exist_ok=False)
    truth, private_meta = _load_private_images(private_truth.resolve())
    private_candidates = {"x0": x0, CONTROL_ARM: control, NEW_ARM: parent_audit["new_pixels"]}
    private_paths = {"x0": private_output / "x0_candidate.pt",
                     CONTROL_ARM: private_output / f"{CONTROL_ARM}_candidate.pt",
                     NEW_ARM: private_output / f"{NEW_ARM}_candidate.pt"}
    source_paths = {"x0": PARENT_PUBLIC_ROOT / "x0_candidate.pt",
                    CONTROL_ARM: PREVIOUS_PUBLIC_ROOT / CONTROL_ARM / "candidate.pt",
                    NEW_ARM: PARENT_PUBLIC_ROOT / NEW_ARM / "candidate.pt"}
    for name in private_candidates:
        shutil.copy2(source_paths[name], private_paths[name])
    reports: dict[str, dict[str, Any]] = {}
    for name in ("x0", CONTROL_ARM, NEW_ARM):
        reports[name] = evaluate_candidate(private_paths[name], private_truth.resolve(), include_labels=False)
    lpips_info = _lpips(private_candidates, reports, truth)
    means = _means(reports)
    prior = _prior_control_means()
    control_diff = {key: means[CONTROL_ARM][key] - prior[key] for key in prior}
    control_tolerance = {key: 2e-8 for key in prior}
    control_matches = all(abs(control_diff[key]) <= control_tolerance[key] for key in control_diff)
    figure = _write_figure(private_output, private_candidates, truth, reports)
    protected_after_private = hash_files(protected_paths())
    inputs_after_private = hash_public_inputs(PUBLIC_ROOT, NAIVE_ROOT, CHECKPOINT)
    if protected_after_private != current_protected or not _hashes_match(parent.get("input_hashes"), inputs_after_private, "inputs_after_private"):
        raise FinalizationError("protected source or public input changed during private evaluation")
    if not control_matches:
        write_json(private_output / "control_reproduction_mismatch.json", {
            "prior_means": prior, "current_means": means[CONTROL_ARM],
            "difference": control_diff, "tolerance": control_tolerance,
        })
        raise FinalizationError("archived control private metrics differ beyond the unchanged tolerance")
    private_comparison = {
        "task_id": TASK_ID, "finalization_task_id": FINALIZATION_TASK_ID,
        "plan_version": PLAN_VERSION, "status": "COMPLETED", "evaluation_status": "COMPLETED",
        "public_output": PARENT_PUBLIC_ROOT.as_posix(), "private_output": private_output.as_posix(),
        "private_truth_path": private_truth.resolve().as_posix(), "private_truth_metadata": private_meta,
        "labels_consumed": False, "true_updates_consumed": False,
        "matching": "Hungarian_pixel_MSE_once_per_candidate_reused_for_SSIM_LPIPS_and_aligned_figure",
        "all_images_included": True, "candidates": {name: {"tensor_sha256": _tensor_sha(value),
                                                               "file_sha256": file_sha256(private_paths[name])}
                                                      for name, value in private_candidates.items()},
        "image_metrics": reports, "mean_metrics": means, "lpips": lpips_info,
        "figure_artifacts": [figure],
        "fixed_ablation": {"control": CONTROL_ARM, "new_arm": NEW_ARM, "lambda_fia_control": 1.0,
                           "lambda_fia_new": 0.0, "lambda_tv": TV_WEIGHT, "lambda_cell": LAMBDA_CELL,
                           "h1_labels": list(H1_LABELS)},
        "control_metric_reproduction": {"matches_prior_report_with_tolerance": control_matches,
                                         "prior_means": prior, "current_means": means[CONTROL_ARM],
                                         "difference": control_diff, "tolerance": control_tolerance},
        "public_gate": final_integrity, "source_integrity_after_private": {
            "protected_sources_unchanged": protected_after_private == current_protected,
            "inputs_unchanged": _hashes_match(parent.get("input_hashes"), inputs_after_private, "inputs_after_private")},
        "directional_interpretation": {
            "rule": "fixed-H1 paired descriptive FIA-weight contrast; no threshold, selection, or generalization",
            "private_metrics_not_used_for_attack_or_selection": True, "labels_consumed": False,
        },
    }
    write_json(private_output / "comparison.json", private_comparison)
    write_json(private_output / "results.json", private_comparison)
    _write_private_report(private_output, private_comparison)
    stages["offline_private_all16_evaluation"] = {"wall_seconds": time.perf_counter() - stage_start}

    public_summary["private_evaluation"] = {"status": "COMPLETED", "output": private_output.as_posix(),
                                            "labels_consumed": False, "all_images_included": True}
    write_json(PARENT_PUBLIC_ROOT / "finalized_summary.json", public_summary)
    result = {
        "task_id": FINALIZATION_TASK_ID, "scientific_task_id": TASK_ID, "status": "COMPLETED",
        "parent_wrapper_exit": 1, "finalizer_exit": 0, "evaluation_exit": 0,
        "parent_optimization_steps": 100, "new_optimizer_steps": 0,
        "candidate_hashes": private_comparison["candidates"],
        "public_summary": public_summary, "private_comparison": private_comparison,
        "stages": stages, "parent_real_phase_seconds": PARENT_REAL_SECONDS,
        "wall_seconds": time.perf_counter() - started,
        "source_changes": _source_change_record(),
        "archived_failed_summary_sha256": PARENT_FAILED_SUMMARY_SHA,
        "warnings": public_summary["warnings"],
    }
    write_json(report_root / "results.json", result)
    lines = [f"# {FINALIZATION_TASK_ID}", "", "Status: **COMPLETED**.", "",
             f"The saved {NEW_ARM} candidate was finalized after the parent wrapper's exit 1 at the endpoint schema gate. "
             f"No optimizer step was run during finalization; the original 100-step arm and all parent artifacts remain byte-identical.", "",
             "Public endpoint authentication passed for x0, B_H1_high, and C_H1_high_noFIA. "
             "The canonical H1 profile uses exact integer, boolean, string, and tensor-schema checks; bounded float differences are recorded in `endpoint_profiles.json`.", "",
             "The private all16 evaluation completed after the public gate. It consumed only private image tensor `x`; `labels_consumed=false` and `true_updates_consumed=false`.", "",
             f"Private report: `{private_output.as_posix()}/report.md`.",
             f"Archived parent failed summary SHA256: `{PARENT_FAILED_SUMMARY_SHA}`."]
    (report_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "completion_receipt.json", {
        "task_id": FINALIZATION_TASK_ID, "status": "COMPLETED", "finalizer_exit": 0,
        "evaluation_exit": 0, "report": (report_root / "report.md").as_posix(),
        "public_summary": (PARENT_PUBLIC_ROOT / "finalized_summary.json").as_posix(),
        "private_comparison": (private_output / "comparison.json").as_posix(),
        "new_optimizer_steps": 0,
    })
    write_json(report_root / "execution_summary.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-truth", type=Path, default=PRIVATE_TRUTH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = finalize_saved_candidates(private_truth=args.private_truth)
    except Exception as error:
        failure = {"task_id": FINALIZATION_TASK_ID, "scientific_task_id": TASK_ID,
                   "status": "PARTIAL", "finalizer_exit": 1, "error_type": type(error).__name__,
                   "error": str(error), "new_optimizer_steps": 0}
        FINAL_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        write_json(FINAL_REPORT_ROOT / "failure.json", failure)
        write_json(FINAL_REPORT_ROOT / "execution_summary.json", failure)
        print(json.dumps(failure, ensure_ascii=False, allow_nan=False), flush=True)
        return 1
    print(json.dumps({"task_id": FINALIZATION_TASK_ID, "status": result["status"],
                      "wall_seconds": result["wall_seconds"]}, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
