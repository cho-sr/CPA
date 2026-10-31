"""Single-factor H1 actual-update ablation with the FIA target removed.

This module is deliberately separate from the completed five-arm screen.  It
reuses the signed cell implementation and FC replay, starts once from the
saved naive pixels, and runs exactly one new 100-step arm.  The FIA cosine is
computed only as a diagnostic; the optimizer gradient is assembled from TV
and the signed-cell replay gradient with ``lambda_fia=0``.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor, nn

from .deterministic_hybrid_setup import configure_runtime
from .fc_replay import fc_recomputed_gradients
from .signed_baseline_observation import CHECKPOINT_SHA
from .signed_e2e_cells import CellTable, DeviceCellTable, load_public_cell_tables
from .signed_e2e_runner import (
    CELL_CHUNK, IMAGE_COUNT, IMAGE_LR, IMAGE_LR_FINAL, IMAGE_STEPS, LOCAL_LR,
    LOCAL_STEPS, MAX_CUDA_ALLOCATED, MAX_SECONDS, REPO_ROOT, TV_WEIGHT,
    _cell_callback, _device_tables, _feature_forward, _arm_metrics,
    canonical_profile, compare_replay_paths, cuda_resource, file_sha256,
    hash_files, hash_public_inputs, load_model, load_public_start, normalize_pixels,
    protected_paths, total_variation, write_json,
)


TASK_ID = "CPA_SIGNED_E2E_H1_NOFIA_20260916"
PLAN_VERSION = 1
PREVIOUS_TASK_ID = "CPA_SIGNED_E2E_CELL_SCREEN_20260916"
NEW_ARM = "C_H1_high_noFIA"
CONTROL_ARM = "B_H1_high"
LAMBDA_FIA = 0.0
LAMBDA_CELL = 0.24202997768915402
H1_LABELS = (507, 655, 452, 549, 231, 657, 160, 761,
             975, 389, 658, 761, 621, 124, 969, 290)

PUBLIC_ROOT = REPO_ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation"
NAIVE_ROOT = REPO_ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive"
CHECKPOINT = Path("C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
PREVIOUS_PUBLIC_ROOT = REPO_ROOT / "quantization_aware_cpa/results/signed_e2e_cell_screen_B16_T10_20260916"
PREVIOUS_PRIVATE_ROOT = REPO_ROOT / "quantization_aware_cpa/results_private/signed_e2e_cell_screen_B16_T10_20260916"


class ExperimentError(RuntimeError):
    """A fixed-input, provenance, numeric, resource, or budget violation."""


class Clock:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.stages: dict[str, dict[str, float]] = {}

    def seconds(self) -> float:
        return time.perf_counter() - self.started

    def check(self, label: str) -> None:
        elapsed = self.seconds()
        if elapsed > MAX_SECONDS:
            raise ExperimentError(f"45-minute real-phase deadline exceeded at {label}: {elapsed:.3f}s")

    def stage(self, name: str):
        from contextlib import contextmanager

        @contextmanager
        def _context():
            start = time.perf_counter()
            self.check(f"stage_start:{name}")
            try:
                yield
            finally:
                self.stages[name] = {"wall_seconds": time.perf_counter() - start,
                                     "total_seconds": self.seconds()}
        return _context()


def _path(path: Path) -> str:
    return path.resolve().as_posix()


def _tensor_sha(value: Tensor) -> str:
    import numpy as np

    array = value.detach().cpu().contiguous().numpy()
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(memoryview(array).cast("B"))
    return hasher.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExperimentError(f"expected JSON object: {path}")
    return value


def _must_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise ExperimentError(f"required regular file is missing: {path}")


CANONICAL_PROFILE_ROOT_FIELDS = (
    "per_tensor", "overall_numel", "overall_code_mismatch_count",
    "overall_code_mismatch_fraction", "overall_bounded_cell_violation_count",
    "overall_bounded_cell_violation_fraction", "cell_loss_equal_tensor_mean",
    "exact_feasible", "feature_norm_mean", "feature_norm_min", "feature_norm_max",
)
CANONICAL_PROFILE_TENSOR_FIELDS = (
    "name", "numel", "fixed_s", "code_mismatch_count", "code_mismatch_fraction",
    "bounded_cell_violation_count", "bounded_cell_violation_fraction",
    "mean_violation_in_s", "max_violation_in_s", "cell_penalty", "candidate_scale",
)
PROFILE_DIAGNOSTIC_FIELDS = (
    "label_hypothesis", "feature_target_cosine_loss", "tv_normalized", "pixel_min",
    "pixel_max", "pixel_mean", "x0_displacement_l2",
)
_ROOT_INTEGER_FIELDS = {
    "overall_numel", "overall_code_mismatch_count", "overall_bounded_cell_violation_count",
}
_ROOT_FLOAT_FIELDS = {
    "overall_code_mismatch_fraction", "overall_bounded_cell_violation_fraction",
    "cell_loss_equal_tensor_mean", "feature_norm_mean", "feature_norm_min", "feature_norm_max",
}
_TENSOR_INTEGER_FIELDS = {"numel", "code_mismatch_count", "bounded_cell_violation_count"}
_TENSOR_FLOAT_FIELDS = {
    "fixed_s", "code_mismatch_fraction", "bounded_cell_violation_fraction",
    "mean_violation_in_s", "max_violation_in_s", "cell_penalty", "candidate_scale",
}


def _is_json_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_json_float(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_canonical_profile(profile: Any, *, name: str = "profile") -> list[dict[str, Any]]:
    """Validate the complete canonical profile, permitting only diagnostics at root."""
    errors: list[dict[str, Any]] = []

    def error(path: str, expected: Any, actual: Any) -> None:
        errors.append({"path": path, "expected": expected, "actual": actual})

    if not isinstance(profile, dict):
        return [{"path": name, "expected": "object", "actual": type(profile).__name__}]
    required = set(CANONICAL_PROFILE_ROOT_FIELDS)
    allowed = required | set(PROFILE_DIAGNOSTIC_FIELDS)
    for key in sorted(required - set(profile)):
        error(f"{name}.{key}", "required", "missing")
    for key in sorted(set(profile) - allowed):
        error(f"{name}.{key}", "canonical field or documented diagnostic", "unexpected")
    if "per_tensor" in profile:
        tensors = profile["per_tensor"]
        if not isinstance(tensors, dict):
            error(f"{name}.per_tensor", "object", type(tensors).__name__)
        else:
            expected_names = {"classifier.0.bias", "classifier.0.weight", "classifier.3.bias",
                              "classifier.3.weight", "classifier.6.bias", "classifier.6.weight"}
            for tensor_name in sorted(expected_names - set(tensors)):
                error(f"{name}.per_tensor.{tensor_name}", "required", "missing")
            for tensor_name in sorted(set(tensors) - expected_names):
                error(f"{name}.per_tensor.{tensor_name}", "registered tensor name", "unexpected")
            for tensor_name in sorted(expected_names & set(tensors)):
                value = tensors[tensor_name]
                path = f"{name}.per_tensor.{tensor_name}"
                if not isinstance(value, dict):
                    error(path, "object", type(value).__name__)
                    continue
                keys = set(value)
                for key in sorted(set(CANONICAL_PROFILE_TENSOR_FIELDS) - keys):
                    error(f"{path}.{key}", "required", "missing")
                for key in sorted(keys - set(CANONICAL_PROFILE_TENSOR_FIELDS)):
                    error(f"{path}.{key}", "canonical tensor field", "unexpected")
                if "name" in value and value["name"] != tensor_name:
                    error(f"{path}.name", tensor_name, value["name"])
                for key in sorted(_TENSOR_INTEGER_FIELDS & keys):
                    if not _is_json_integer(value[key]):
                        error(f"{path}.{key}", "integer", value[key])
                for key in sorted(_TENSOR_FLOAT_FIELDS & keys):
                    if not _is_json_float(value[key]):
                        error(f"{path}.{key}", "finite number", value[key])
                if "name" in value and not isinstance(value["name"], str):
                    error(f"{path}.name", "string", value["name"])
    for key in sorted(_ROOT_INTEGER_FIELDS & set(profile)):
        if not _is_json_integer(profile[key]):
            error(f"{name}.{key}", "integer", profile[key])
    for key in sorted(_ROOT_FLOAT_FIELDS & set(profile)):
        if not _is_json_float(profile[key]):
            error(f"{name}.{key}", "finite number", profile[key])
    if "exact_feasible" in profile and not isinstance(profile["exact_feasible"], bool):
        error(f"{name}.exact_feasible", "boolean", profile["exact_feasible"])
    for key in set(PROFILE_DIAGNOSTIC_FIELDS) & set(profile):
        if key == "label_hypothesis":
            if not isinstance(profile[key], str):
                error(f"{name}.{key}", "string", profile[key])
        elif not _is_json_float(profile[key]):
            error(f"{name}.{key}", "finite number", profile[key])
    return errors


def _canonical_projection(profile: dict[str, Any]) -> dict[str, Any]:
    return {key: profile[key] for key in CANONICAL_PROFILE_ROOT_FIELDS}


def compare_canonical_profiles(actual: Any, expected: Any) -> dict[str, Any]:
    """Compare canonical fields exactly, while recording bounded float differences."""
    actual_errors = validate_canonical_profile(actual, name="actual")
    expected_errors = validate_canonical_profile(expected, name="expected")
    result: dict[str, Any] = {
        "passed": False, "schema_errors": {"actual": actual_errors, "expected": expected_errors},
        "mismatches": [], "float_differences": [],
        "float_tolerance": {"rel_tol": 2e-7, "abs_tol": 2e-9},
    }
    if actual_errors or expected_errors:
        return result

    def visit(left: Any, right: Any, path: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in left:
                visit(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, bool) or isinstance(right, bool):
            if type(left) is not type(right) or left != right:
                result["mismatches"].append({"path": path, "expected": right, "actual": left,
                                               "kind": "exact"})
            return
        if _is_json_integer(left) or _is_json_integer(right):
            if type(left) is not type(right) or left != right:
                result["mismatches"].append({"path": path, "expected": right, "actual": left,
                                               "kind": "exact_integer"})
            return
        if _is_json_float(left) and _is_json_float(right):
            difference = float(left) - float(right)
            close = math.isclose(float(left), float(right), rel_tol=2e-7, abs_tol=2e-9)
            result["float_differences"].append({"path": path, "expected": float(right),
                                                 "actual": float(left), "difference": difference,
                                                 "within_tolerance": close})
            if not close:
                result["mismatches"].append({"path": path, "expected": float(right),
                                               "actual": float(left), "difference": difference,
                                               "kind": "float_bound"})
            return
        if left != right:
            result["mismatches"].append({"path": path, "expected": right, "actual": left,
                                           "kind": "exact"})

    visit(_canonical_projection(actual), _canonical_projection(expected), "profile")
    result["passed"] = not result["mismatches"]
    return result


def _same_profile(left: Any, right: Any, path: str = "") -> tuple[bool, list[str]]:
    comparison = compare_canonical_profiles(left, right)
    differences = []
    for item in comparison["schema_errors"]["actual"] + comparison["schema_errors"]["expected"]:
        differences.append(f"{item['path']} expected={item['expected']!r} actual={item['actual']!r}")
    for item in comparison["mismatches"]:
        differences.append(f"{item['path']} expected={item['expected']!r} actual={item['actual']!r}")
    return bool(comparison["passed"]), differences


def _normalize_hash_key(key: Any) -> str:
    return str(key).replace("\\", "/").casefold()


def _hash_map_differences(expected: Any, actual: Any, label: str) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return [{"path": label, "expected": "object", "actual": type(expected).__name__ + "/" + type(actual).__name__}]
    expected_normalized = {_normalize_hash_key(key): value for key, value in expected.items()}
    actual_normalized = {_normalize_hash_key(key): value for key, value in actual.items()}
    for key in sorted(set(expected_normalized) | set(actual_normalized)):
        if key not in expected_normalized:
            differences.append({"path": f"{label}.{key}", "expected": "missing", "actual": actual_normalized[key]})
        elif key not in actual_normalized:
            differences.append({"path": f"{label}.{key}", "expected": expected_normalized[key], "actual": "missing"})
        elif expected_normalized[key] != actual_normalized[key]:
            differences.append({"path": f"{label}.{key}", "expected": expected_normalized[key],
                                "actual": actual_normalized[key]})
    return differences


def validate_structured_provenance(registration: Any, integrity: Any,
                                   actual_inputs: dict[str, str],
                                   actual_sources: dict[str, str]) -> dict[str, Any]:
    """Validate predecessor JSON records and their byte hashes without marker searches."""
    errors: list[dict[str, Any]] = []
    if not isinstance(registration, dict):
        errors.append({"path": "registration", "expected": "object", "actual": type(registration).__name__})
        registration = {}
    if not isinstance(integrity, dict):
        errors.append({"path": "integrity", "expected": "object", "actual": type(integrity).__name__})
        integrity = {}
    for path, expected in (("registration.task_id", PREVIOUS_TASK_ID),
                           ("registration.plan_version", PLAN_VERSION),
                           ("registration.status", "REGISTERED")):
        current = registration
        for part in path.split(".")[1:]:
            current = current.get(part) if isinstance(current, dict) else None
        if current != expected:
            errors.append({"path": path, "expected": expected, "actual": current})
    for path in ("protected_sources_unchanged", "inputs_unchanged"):
        if integrity.get(path) is not True:
            errors.append({"path": f"integrity.{path}", "expected": True, "actual": integrity.get(path)})
    registration_inputs = registration.get("input_hashes")
    registration_sources = registration.get("protected_source_hashes_before")
    integrity_inputs_before = integrity.get("input_hashes_before")
    integrity_inputs_after = integrity.get("input_hashes_after")
    integrity_sources_before = integrity.get("protected_source_hashes_before")
    integrity_sources_after = integrity.get("protected_source_hashes_after")
    for label, left, right in (
        ("input_hashes.registration_vs_integrity_before", registration_inputs, integrity_inputs_before),
        ("input_hashes.registration_vs_integrity_after", registration_inputs, integrity_inputs_after),
        ("source_hashes.registration_vs_integrity_before", registration_sources, integrity_sources_before),
        ("source_hashes.registration_vs_integrity_after", registration_sources, integrity_sources_after),
        ("input_hashes.recorded_vs_current", registration_inputs, actual_inputs),
        ("source_hashes.recorded_vs_current", registration_sources, actual_sources),
    ):
        errors.extend(_hash_map_differences(left, right, label))
    return {"passed": not errors, "mismatches": errors,
            "json_authentication": "structured_utf8_json_and_normalized_path_key_hashes"}


def authenticate_previous_run() -> dict[str, Any]:
    if not PREVIOUS_PUBLIC_ROOT.is_dir():
        raise ExperimentError("completed public control root is missing")
    summary = _read_json(PREVIOUS_PUBLIC_ROOT / "summary.json")
    labels = _read_json(PREVIOUS_PUBLIC_ROOT / "label_hypotheses.json")
    calibration = _read_json(PREVIOUS_PUBLIC_ROOT / "calibration.json")
    control_result = _read_json(PREVIOUS_PUBLIC_ROOT / CONTROL_ARM / "result.json")
    endpoint = _read_json(PREVIOUS_PUBLIC_ROOT / "endpoint_profiles.json")
    if summary.get("task_id") != PREVIOUS_TASK_ID or summary.get("status") != "COMPLETED":
        raise ExperimentError("previous public summary is not the completed signed screen")
    if labels.get("h1", {}).get("labels") != list(H1_LABELS):
        raise ExperimentError("previous H1 label vector does not match the signed handoff")
    if labels.get("labels_are_hypotheses") is not True:
        raise ExperimentError("previous H1 label vector is not marked as a hypothesis")
    if calibration.get("weights", {}).get("lambda_high") != LAMBDA_CELL:
        raise ExperimentError("previous high cell weight does not match the signed handoff")
    if calibration.get("H1", {}).get("cell_loss") != 22.991680145263672:
        raise ExperimentError("previous H1 calibration cell loss is unexpected")
    if calibration.get("H1", {}).get("feature_norm_mean") != 21.787342071533203:
        raise ExperimentError("previous H1 calibration feature norm is unexpected")
    if control_result.get("status") != "COMPLETED" or control_result.get("iterations") != 100:
        raise ExperimentError("archived B_H1_high control is incomplete")
    if control_result.get("lambda_cell") != LAMBDA_CELL:
        raise ExperimentError("archived control cell weight is unexpected")
    control_candidate = PREVIOUS_PUBLIC_ROOT / CONTROL_ARM / "candidate.pt"
    control_state = PREVIOUS_PUBLIC_ROOT / CONTROL_ARM / "attack_state.pt"
    _must_file(control_candidate)
    _must_file(control_state)
    if control_result.get("candidate_sha256") != file_sha256(control_candidate):
        raise ExperimentError("archived control candidate hash mismatch")
    if control_result.get("candidate_sha256") != "b352c3bde7db87ff00ecbbbd4715c5c865ada662c4dcf88b1e116f5369ec9d5a":
        raise ExperimentError("archived control candidate is not the registered artifact")
    if CONTROL_ARM not in endpoint or "H1" not in endpoint[CONTROL_ARM]:
        raise ExperimentError("archived control endpoint profile is missing")
    old_files = [
        PREVIOUS_PUBLIC_ROOT / "registration.json",
        PREVIOUS_PUBLIC_ROOT / "integrity.json",
        PREVIOUS_PUBLIC_ROOT / "summary.json",
        PREVIOUS_PUBLIC_ROOT / "label_hypotheses.json",
        PREVIOUS_PUBLIC_ROOT / "calibration.json",
        PREVIOUS_PUBLIC_ROOT / "endpoint_profiles.json",
        PREVIOUS_PUBLIC_ROOT / CONTROL_ARM / "candidate.pt",
        PREVIOUS_PUBLIC_ROOT / CONTROL_ARM / "attack_state.pt",
        PREVIOUS_PUBLIC_ROOT / CONTROL_ARM / "result.json",
    ]
    registration = _read_json(PREVIOUS_PUBLIC_ROOT / "registration.json")
    integrity = _read_json(PREVIOUS_PUBLIC_ROOT / "integrity.json")
    provenance = validate_structured_provenance(
        registration, integrity, hash_public_inputs(PUBLIC_ROOT, NAIVE_ROOT, CHECKPOINT),
        hash_files(protected_paths()))
    if not provenance["passed"]:
        raise ExperimentError("previous structured provenance authentication failed: " +
                              json.dumps(provenance, ensure_ascii=False, sort_keys=True))
    old_hashes = {_path(path): file_sha256(path) for path in old_files}
    return {
        "task_id": PREVIOUS_TASK_ID,
        "summary_status": summary["status"],
        "h1_labels": list(H1_LABELS),
        "lambda_cell": LAMBDA_CELL,
        "calibration": {"cell_loss": calibration["H1"]["cell_loss"],
                         "feature_norm_mean": calibration["H1"]["feature_norm_mean"]},
        "control_result": control_result,
        "control_candidate_sha256": file_sha256(control_candidate),
        "control_files_sha256": old_hashes,
        "registration": registration,
        "integrity": integrity,
        "structured_provenance": provenance,
        "control_endpoint_profile": endpoint[CONTROL_ARM]["H1"],
        "x0_endpoint_profile": endpoint["x0"]["H1"],
    }


def compose_optimizer_gradient(fia_gradient: Tensor, tv_gradient: Tensor,
                               cell_gradient: Tensor, *, lambda_fia: float = LAMBDA_FIA,
                               lambda_tv: float = TV_WEIGHT,
                               lambda_cell: float = LAMBDA_CELL) -> Tensor:
    """Assemble the actual image gradient; FIA can be measured but omitted."""
    if not (fia_gradient.shape == tv_gradient.shape == cell_gradient.shape):
        raise ExperimentError("objective component gradients have different shapes")
    if not all(value.dtype == torch.float32 for value in (fia_gradient, tv_gradient, cell_gradient)):
        raise ExperimentError("objective component gradients must be FP32")
    result = lambda_fia * fia_gradient + lambda_tv * tv_gradient + lambda_cell * cell_gradient
    if not bool(torch.isfinite(result).all()):
        raise ExperimentError("composed image gradient is nonfinite")
    return result


def _cell_gradient(model: nn.Module, pixels: Tensor, labels: Tensor,
                   device_tables: Mapping[str, DeviceCellTable]) -> tuple[Tensor, Tensor, Tensor]:
    features = _feature_forward(model, pixels)
    cell_loss, feature_gradient, target_gradient = fc_recomputed_gradients(
        model.classifier, features, labels, LOCAL_STEPS, LOCAL_LR,
        _cell_callback(device_tables))
    if target_gradient is not None:
        raise ExperimentError("hard-label replay returned an unexpected target gradient")
    image_gradient, = torch.autograd.grad(features, pixels, grad_outputs=feature_gradient)
    if not bool(torch.isfinite(image_gradient).all()):
        raise ExperimentError("cell image gradient is nonfinite")
    return cell_loss.detach(), image_gradient, features


def _endpoint_profile(model: nn.Module, pixels: Tensor, x0: Tensor, target: Tensor,
                      labels: Tensor, tables: Mapping[str, CellTable],
                      device: torch.device) -> dict[str, Any]:
    profile = canonical_profile(model, pixels, labels, tables, device)
    with torch.no_grad():
        features = _feature_forward(model, pixels)
        fia = (1.0 - ((features * target).sum(dim=1) /
                      (features.norm(dim=1) * target.norm(dim=1)))).mean()
        tv = total_variation(normalize_pixels(pixels))
    profile.update({
        "label_hypothesis": "H1",
        "feature_target_cosine_loss": float(fia),
        "tv_normalized": float(tv),
        "pixel_min": float(pixels.min()),
        "pixel_max": float(pixels.max()),
        "pixel_mean": float(pixels.mean()),
        "x0_displacement_l2": float((pixels - x0).norm()),
    })
    return profile


def _run_arm(model: nn.Module, x0_cpu: Tensor, target_cpu: Tensor, labels: Tensor,
             tables: Mapping[str, CellTable], device_tables: Mapping[str, DeviceCellTable],
             output: Path, device: torch.device, clock: Clock) -> dict[str, Any]:
    folder = output / NEW_ARM
    if folder.exists():
        raise ExperimentError("new arm output unexpectedly exists")
    folder.mkdir()
    pixels = nn.Parameter(x0_cpu.to(device=device).clone())
    x0 = x0_cpu.to(device=device)
    target = target_cpu.to(device=device)
    labels = labels.to(device=device)
    optimizer = torch.optim.Adam([pixels], lr=IMAGE_LR, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0.0)
    logs: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(IMAGE_STEPS):
        clock.check(f"{NEW_ARM}:step:{step + 1}")
        learning_rate = IMAGE_LR * (IMAGE_LR_FINAL / IMAGE_LR +
                                    (1 - IMAGE_LR_FINAL / IMAGE_LR) *
                                    (1 + math.cos(math.pi * step / (IMAGE_STEPS - 1))) / 2)
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        features = _feature_forward(model, pixels)
        fia_graph = (1.0 - ((features * target).sum(dim=1) /
                            (features.norm(dim=1) * target.norm(dim=1)))).mean()
        fia_gradient, = torch.autograd.grad(fia_graph, pixels, retain_graph=True)
        tv = total_variation(normalize_pixels(pixels))
        tv_gradient, = torch.autograd.grad(tv, pixels, retain_graph=True)
        cell_value, feature_gradient, _ = fc_recomputed_gradients(
            model.classifier, features, labels, LOCAL_STEPS, LOCAL_LR,
            _cell_callback(device_tables))
        cell_gradient, = torch.autograd.grad(features, pixels, grad_outputs=feature_gradient)
        total_gradient = compose_optimizer_gradient(fia_gradient, tv_gradient, cell_gradient)
        weighted_tv = TV_WEIGHT * tv_gradient
        weighted_cell = LAMBDA_CELL * cell_gradient
        total_from_components = weighted_tv + weighted_cell
        component_difference = float((total_gradient - total_from_components).abs().max())
        if component_difference != 0.0:
            raise ExperimentError(f"lambda_F=0 component gradient mismatch: {component_difference}")
        if not bool(torch.isfinite(total_gradient).all()):
            raise ExperimentError(f"nonfinite optimizer gradient at step {step + 1}")
        pixels.grad = total_gradient
        before = pixels.detach().clone()
        optimizer.step()
        with torch.no_grad():
            pixels.clamp_(0.0, 1.0)
            pixel_step = (pixels - before).norm()
            clamp_fraction = ((pixels == 0) | (pixels == 1)).float().mean()
        tv_norm = float(tv_gradient.norm())
        cell_norm = float(cell_gradient.norm())
        weighted_tv_norm = float(weighted_tv.norm())
        weighted_cell_norm = float(weighted_cell.norm())
        direction_cosine = None
        if weighted_tv_norm > 0 and weighted_cell_norm > 0:
            direction_cosine = float((weighted_tv * weighted_cell).sum() /
                                     (weighted_tv.norm() * weighted_cell.norm()))
        row = {
            "iteration": step + 1,
            "learning_rate": learning_rate,
            "pre_step_fia_loss": float(fia_graph.detach()),
            "pre_step_tv_normalized": float(tv.detach()),
            "pre_step_cell_loss": float(cell_value.detach()),
            "pre_step_base_loss": float(fia_graph.detach() + TV_WEIGHT * tv.detach()),
            "pre_step_total_loss": float(TV_WEIGHT * tv.detach() + LAMBDA_CELL * cell_value.detach()),
            "fia_gradient_norm_diagnostic": float(fia_gradient.norm()),
            "tv_gradient_norm": tv_norm,
            "cell_gradient_norm": cell_norm,
            "weighted_tv_gradient_norm": weighted_tv_norm,
            "weighted_cell_gradient_norm": weighted_cell_norm,
            "weighted_tv_cell_direction_cosine": direction_cosine,
            "total_gradient_norm": float(total_gradient.norm()),
            "lambda_fia": LAMBDA_FIA,
            "lambda_tv": TV_WEIGHT,
            "lambda_cell": LAMBDA_CELL,
            "optimizer_gradient_excludes_fia": True,
            "component_gradient_max_abs_difference": component_difference,
            "actual_pixel_step_l2": float(pixel_step),
            "post_step_clamp_fraction": float(clamp_fraction),
            "elapsed_seconds": time.perf_counter() - started,
        }
        logs.append(row)
        write_json(folder / "optimization_log.json", logs)
        if device.type == "cuda":
            cuda_resource(device, f"{NEW_ARM}:step:{step + 1}")
        del features, fia_graph, fia_gradient, tv, tv_gradient, cell_value, feature_gradient
        del cell_gradient, total_gradient, weighted_tv, weighted_cell, total_from_components, before
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    final_pixels = pixels.detach().clone()
    final_metrics = _arm_metrics(model, final_pixels, x0, target)
    torch.save({"pixels": final_pixels.cpu()}, folder / "candidate.pt")
    torch.save({"pixels": final_pixels.cpu(), "labels": labels.cpu(),
                "lambda_fia": LAMBDA_FIA, "lambda_tv": TV_WEIGHT,
                "lambda_cell": LAMBDA_CELL, "completed_iterations": IMAGE_STEPS},
               folder / "attack_state.pt")
    result = {
        "status": "COMPLETED", "task_id": TASK_ID, "arm": NEW_ARM,
        "optimized_label_hypothesis": "H1", "lambda_fia": LAMBDA_FIA,
        "lambda_tv": TV_WEIGHT, "lambda_cell": LAMBDA_CELL,
        "iterations": IMAGE_STEPS, "optimization_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": 0 if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": 0 if device.type != "cuda" else int(torch.cuda.max_memory_reserved(device)),
        "final_metrics": final_metrics, "candidate_sha256": file_sha256(folder / "candidate.pt"),
    }
    write_json(folder / "result.json", result)
    del optimizer, pixels, x0, target, labels, final_pixels
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_ablation(*, public_observation: Path = PUBLIC_ROOT, naive: Path = NAIVE_ROOT,
                 checkpoint: Path = CHECKPOINT, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    clock = Clock()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing preexisting noFIA output: {output}")
    if public_observation.resolve() != PUBLIC_ROOT.resolve() or naive.resolve() != NAIVE_ROOT.resolve():
        raise ExperimentError("this ablation accepts only the registered public observation and naive roots")
    if checkpoint.resolve() != CHECKPOINT.resolve() or file_sha256(checkpoint) != CHECKPOINT_SHA:
        raise ExperimentError("checkpoint path or hash does not match the signed packet")
    previous = authenticate_previous_run()
    manifest, tables = load_public_cell_tables(public_observation)
    input_hashes = hash_public_inputs(public_observation, naive, checkpoint)
    protected_before = hash_files(protected_paths())
    shared_paths = [
        REPO_ROOT / "quantization_aware_cpa/qacpa/signed_e2e_cells.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/signed_e2e_runner.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/fc_replay.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/full_update_replay.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/full_update_adjoint.py",
    ]
    shared_before = hash_files(shared_paths)
    x0, source, target = load_public_start(naive)
    configure_runtime(42, threads=1)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise ExperimentError("the authorized real phase requires CUDA")
    device_tables = _device_tables(tables, device)
    output.mkdir(parents=True, exist_ok=False)
    registration = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "REGISTERED",
        "registered_before_optimization": True, "registered_before_private_evaluation": True,
        "previous_task_id": PREVIOUS_TASK_ID, "new_arm": NEW_ARM,
        "control_arm_reused": CONTROL_ARM, "control_retrained": False,
        "public_observation": _path(public_observation), "naive": _path(naive),
        "checkpoint": _path(checkpoint), "checkpoint_sha256": CHECKPOINT_SHA,
        "public_manifest_sha256": file_sha256(public_observation / "manifest.json"),
        "input_hashes": input_hashes, "protected_source_hashes_before": protected_before,
        "shared_source_hashes_before": shared_before,
        "source_and_target_hashes": {"x0_pixels": _tensor_sha(x0), "S_hat": _tensor_sha(source),
                                      "target_z": _tensor_sha(target)},
        "h1_labels": list(H1_LABELS), "h1_labels_are_hypotheses": True,
        "previous_authentication": previous,
        "objective": {"lambda_fia": LAMBDA_FIA, "lambda_tv": TV_WEIGHT,
                       "lambda_cell": LAMBDA_CELL,
                       "formula": "0*FIA + 0.1*TV(normalize(x)) + 0.24202997768915402*L_C(DeltaTheta(x,H1))",
                       "fia_role": "diagnostic_only_no_optimizer_gradient"},
        "fixed_training": manifest.get("training", {}),
        "fixed_codec": manifest.get("codec_metadata", {}),
        "screen": {"image_steps": IMAGE_STEPS, "image_lr": IMAGE_LR,
                    "image_lr_final": IMAGE_LR_FINAL, "optimizer": "Adam",
                    "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.0,
                    "tv_space": "normalized_u=(x-mean)/std", "cell_chunk_size": CELL_CHUNK,
                    "max_seconds": MAX_SECONDS, "max_cuda_allocated_bytes": MAX_CUDA_ALLOCATED,
                    "seed": 42, "normalization_once": True},
        "label_policy": {"private_labels_used": False, "h1_frozen": True},
        "private_boundary": "no private truth accepted by runner",
    }
    model: nn.Module | None = None
    try:
        write_json(output / "registration.json", registration)
        with clock.stage("model_and_public_preflight"):
            model = load_model(checkpoint, device)
            with torch.no_grad():
                z0 = _feature_forward(model, x0.to(device))
            if tuple(z0.shape) != (IMAGE_COUNT, 25088):
                raise ExperimentError("x0 feature shape mismatch")
        with clock.stage("full_size_numeric_check"):
            numeric = compare_replay_paths(model, x0.to(device),
                                           torch.tensor(H1_LABELS, dtype=torch.int64, device=device),
                                           tables, device_tables, device)
            numeric["resource"] = cuda_resource(device, "full_size_numeric_check")
            write_json(output / "full_size_numeric_check.json", numeric)
        with clock.stage("initial_h1_calibration"):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            calibration_started = time.perf_counter()
            h1 = torch.tensor(H1_LABELS, dtype=torch.int64, device=device)
            calibration_loss, calibration_gradient, calibration_features = _cell_gradient(
                model, x0.to(device).clone().requires_grad_(True), h1, device_tables)
            current_calibration = {
                "cell_loss": float(calibration_loss),
                "cell_image_gradient_norm": float(calibration_gradient.norm()),
                "feature_norm_mean": float(calibration_features.norm(dim=1).mean()),
                "finite": bool(torch.isfinite(calibration_loss) and torch.isfinite(calibration_gradient).all()),
                "wall_seconds": time.perf_counter() - calibration_started,
                "resource": cuda_resource(device, "initial_h1_calibration"),
            }
            expected = previous["calibration"]
            current_calibration["matches_previous_cell_loss_exact"] = current_calibration["cell_loss"] == expected["cell_loss"]
            current_calibration["matches_previous_feature_norm_exact"] = current_calibration["feature_norm_mean"] == expected["feature_norm_mean"]
            current_calibration["cell_loss_difference"] = current_calibration["cell_loss"] - expected["cell_loss"]
            current_calibration["feature_norm_difference"] = current_calibration["feature_norm_mean"] - expected["feature_norm_mean"]
            write_json(output / "initial_calibration_comparison.json", {
                "current": current_calibration, "previous_recorded": expected,
                "separate_cpu_endpoint_mean_not_substituted": 22.991682647381136,
            })
            if not current_calibration["matches_previous_cell_loss_exact"] or not current_calibration["matches_previous_feature_norm_exact"]:
                raise ExperimentError("initial H1 calibration reproduction mismatch")
            del calibration_loss, calibration_gradient, calibration_features, h1
        registration["runtime"] = {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device), "device_name": "cuda",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "threads": torch.get_num_threads(),
        }
        write_json(output / "registration.json", registration)
        with torch.no_grad():
            x0_features = _feature_forward(model, x0.to(device))
            x0_fia = float((1.0 - ((x0_features * target.to(device)).sum(dim=1) /
                                   (x0_features.norm(dim=1) * target.to(device).norm(dim=1)))).mean())
            x0_tv = float(total_variation(normalize_pixels(x0.to(device))))
        torch.save({"pixels": x0}, output / "x0_candidate.pt")
        write_json(output / "x0_metadata.json", {"candidate_sha256": file_sha256(output / "x0_candidate.pt"),
                                                   "x0_tensor_sha256": _tensor_sha(x0),
                                                   "fia_diagnostic": x0_fia, "tv_normalized": x0_tv})
        with clock.stage("one_new_arm"):
            arm_result = _run_arm(model, x0, target, torch.tensor(H1_LABELS, dtype=torch.int64),
                                  tables, device_tables, output, device, clock)
        with clock.stage("public_endpoint_replay"):
            h1_device = torch.tensor(H1_LABELS, dtype=torch.int64, device=device)
            control_pixels = torch.load(PREVIOUS_PUBLIC_ROOT / CONTROL_ARM / "candidate.pt",
                                        map_location=device, weights_only=True)["pixels"].to(device)
            new_pixels = torch.load(output / NEW_ARM / "candidate.pt",
                                    map_location=device, weights_only=True)["pixels"].to(device)
            profiles = {
                "x0": _endpoint_profile(model, x0.to(device), x0.to(device), target.to(device), h1_device, tables, device),
                CONTROL_ARM: _endpoint_profile(model, control_pixels, x0.to(device), target.to(device), h1_device, tables, device),
                NEW_ARM: _endpoint_profile(model, new_pixels, x0.to(device), target.to(device), h1_device, tables, device),
            }
            control_comparison = compare_canonical_profiles(
                profiles[CONTROL_ARM], previous["control_endpoint_profile"])
            control_same = bool(control_comparison["passed"])
            control_differences = [
                f"{item['path']} expected={item.get('expected')!r} actual={item.get('actual')!r}"
                for item in control_comparison["schema_errors"]["actual"]
                + control_comparison["schema_errors"]["expected"]
                + control_comparison["mismatches"]
            ]
            profiles[CONTROL_ARM]["archived_control_authentication"] = {
                "matches_previous_endpoint_profile": control_same,
                "differences": control_differences,
                "comparison": control_comparison,
                "archived_candidate_sha256": previous["control_candidate_sha256"],
            }
            if not control_same:
                raise ExperimentError("archived B_H1_high endpoint profile mismatch: " +
                                      json.dumps(control_comparison, ensure_ascii=False, sort_keys=True))
            write_json(output / "endpoint_profiles.json", profiles)
            del control_pixels, new_pixels, h1_device
        protected_after = hash_files(protected_paths())
        shared_after = hash_files(shared_paths)
        input_after = hash_public_inputs(public_observation, naive, checkpoint)
        integrity = {
            "protected_source_hashes_before": protected_before,
            "protected_source_hashes_after": protected_after,
            "protected_sources_unchanged": protected_before == protected_after,
            "shared_source_hashes_before": shared_before,
            "shared_source_hashes_after": shared_after,
            "shared_sources_unchanged": shared_before == shared_after,
            "input_hashes_before": input_hashes, "input_hashes_after": input_after,
            "inputs_unchanged": input_hashes == input_after,
            "previous_control_files_unchanged": previous["control_files_sha256"] == {
                path.resolve().as_posix(): file_sha256(path)
                for path in [PREVIOUS_PUBLIC_ROOT / "registration.json", PREVIOUS_PUBLIC_ROOT / "integrity.json",
                             PREVIOUS_PUBLIC_ROOT / "summary.json", PREVIOUS_PUBLIC_ROOT / "label_hypotheses.json",
                             PREVIOUS_PUBLIC_ROOT / "calibration.json", PREVIOUS_PUBLIC_ROOT / "endpoint_profiles.json",
                             PREVIOUS_PUBLIC_ROOT / CONTROL_ARM / "candidate.pt", PREVIOUS_PUBLIC_ROOT / CONTROL_ARM / "attack_state.pt",
                             PREVIOUS_PUBLIC_ROOT / CONTROL_ARM / "result.json"]},
        }
        write_json(output / "integrity.json", integrity)
        if not all(integrity.values()):
            raise ExperimentError("source, public input, or archived control integrity changed")
        summary = {
            "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "COMPLETED",
            "stage": "PUBLIC_NOFIA_OPTIMIZATION_COMPLETE", "clock": {
                "total_wall_seconds": clock.seconds(), "stages": clock.stages},
            "objective": registration["objective"], "h1_labels": list(H1_LABELS),
            "control": {"arm": CONTROL_ARM, "reused": True, "retrained": False,
                        "candidate_sha256": previous["control_candidate_sha256"]},
            "new_arm": arm_result, "endpoint_profiles": profiles,
            "replay_counts": {"new_arm_reverse_gradient_steps": IMAGE_STEPS,
                               "endpoint_h1_replays": 3},
            "private_evaluation": "PENDING_SEPARATE_AUTHORIZED_EVALUATOR",
            "resource_rules": {"max_seconds": MAX_SECONDS, "max_cuda_allocated_bytes": MAX_CUDA_ALLOCATED},
            "label_policy": {"private_labels_used": False, "h1_frozen": True},
        }
        write_json(output / "summary.json", summary)
        return summary
    except Exception as error:
        failure = {"task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "FAILED",
                   "stage": "PUBLIC_NOFIA_RUNNER", "error_type": type(error).__name__,
                   "error": str(error), "completed_wall_seconds": clock.seconds(),
                   "stages": clock.stages}
        if output.exists():
            write_json(output / "failure.json", failure)
            write_json(output / "summary.json", failure)
        raise
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-observation", type=Path, default=PUBLIC_ROOT)
    parser.add_argument("--naive", type=Path, default=NAIVE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = run_ablation(**vars(args))
    except Exception as error:
        print(f"{TASK_ID}: {type(error).__name__}: {error}", flush=True)
        return 1
    print(json.dumps({"task_id": TASK_ID, "status": result["status"],
                      "wall_seconds": result["clock"]["total_wall_seconds"]}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
