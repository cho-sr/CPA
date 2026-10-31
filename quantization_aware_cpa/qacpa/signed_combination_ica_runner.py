"""Public-only CLI adapters for the isolated INT4 combination ICA method.

The commands are intentionally separate: ``separate`` creates signed public
features, ``invert`` calls the unchanged Direct FIA adapter, and ``refine``
connects the new target to the unchanged actual-image signed-cell path.  This
module does not accept private truth or labels.  No command is executed by the
implementation handoff; the CLI below is for a future explicitly authorized
research run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import torch
from torch import Tensor, nn

from .signed_baseline_observation import CHECKPOINT_SHA, FC_SHAPES, read_bundle
from .signed_combination_ica import (
    ICA_GA_METHOD, CombinationICAConfig, SeparationResult, separate_observation,
    tensor_sha256, tensor_stats,
)
from .signed_e2e_cells import FC_NAMES, CellTable, DeviceCellTable, load_public_cell_tables
from .signed_e2e_runner import (
    FEATURE_DIM as SHARED_FEATURE_DIM, IMAGE_COUNT, IMAGE_STEPS, REPO_ROOT, TV_WEIGHT,
    Clock, _device_tables, _feature_forward, compare_replay_paths, file_sha256,
    canonical_profile, configure_runtime, hash_files, labels_from_public_logits,
    load_model, protected_paths, run_arm, write_json,
)
from .upstream_naive import run_upstream_fia


TASK_ID = "CPA_SIGNED_COMBINATION_ICA_IMPLEMENTATION_20260916"
PLAN_VERSION = 1
METHOD = ICA_GA_METHOD
SOURCE_COUNT = IMAGE_COUNT
if FC_SHAPES["classifier.0.weight"][1] != SHARED_FEATURE_DIM:
    raise RuntimeError("signed observation width and shared feature dimension disagree")
FEATURE_DIM = SHARED_FEATURE_DIM
PUBLIC_OBSERVATION = REPO_ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation"
CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth"
AUTHORIZED_MANIFEST_SHA = "9b4891765e6fc7e4adb1e3e11c158b3d4a2c3430d3397e1f1466ddafe2e1cd69"
DEFAULT_SEPARATION_OUTPUT = REPO_ROOT / "quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916"
DEFAULT_INVERT_OUTPUT = DEFAULT_SEPARATION_OUTPUT / "direct_fia"
DEFAULT_REFINE_OUTPUT = DEFAULT_SEPARATION_OUTPUT / "actual_update_refinement"
DEFAULT_FIA_CONFIG = {
    "n_iter": 25000, "lr": 0.1, "tv": 0.1, "fi": 1.0,
    "optimizer": "adam", "scheduler": "cosine", "log_every": 5000,
}
FIXED_CELL_WEIGHT = 0.24202997768915402
REFINE_ARM_NAME = "B_H1_combination_ica"
REFINE_LABEL_NAME = "H1"
CANONICAL_SEPARATOR_CONFIG = CombinationICAConfig().as_dict()
CANONICAL_FIA_CONFIG = dict(DEFAULT_FIA_CONFIG)
CANONICAL_REFINE_OBJECTIVE = {
    "feature_weight": 1.0, "tv_weight": TV_WEIGHT, "cell_weight": FIXED_CELL_WEIGHT,
    "optimizer": "Adam", "image_steps": IMAGE_STEPS, "lr_initial": 0.001,
    "lr_final": 0.0001,
}
SHARED_PATHS = [
    REPO_ROOT / "quantization_aware_cpa/qacpa/signed_e2e_cells.py",
    REPO_ROOT / "quantization_aware_cpa/qacpa/signed_e2e_runner.py",
    REPO_ROOT / "quantization_aware_cpa/qacpa/fc_replay.py",
    REPO_ROOT / "quantization_aware_cpa/qacpa/full_update_replay.py",
    REPO_ROOT / "quantization_aware_cpa/qacpa/full_update_adjoint.py",
]


class PipelineError(RuntimeError):
    """Public artifact, provenance, or integration contract failed."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PipelineError(f"invalid UTF-8 JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise PipelineError(f"expected JSON object: {path}")
    return value


def _must_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise PipelineError(f"required regular file is missing: {path}")


def _hash_map(paths: list[Path]) -> dict[str, str]:
    return {path.resolve().as_posix(): file_sha256(path) for path in paths}


def _expected_source_shape() -> tuple[int, int]:
    return (SOURCE_COUNT, FEATURE_DIM)


def _canonical_config(config: CombinationICAConfig | None) -> CombinationICAConfig:
    config = config or CombinationICAConfig()
    if not isinstance(config, CombinationICAConfig):
        raise PipelineError("runner only accepts the registered separator configuration")
    if config.as_dict() != CANONICAL_SEPARATOR_CONFIG:
        raise PipelineError("runner only accepts the registered separator configuration")
    return config


def _runtime(device: torch.device | str, *, phase: str) -> dict[str, Any]:
    device = torch.device(device)
    return {
        "phase": phase, "device": str(device), "torch": torch.__version__,
        "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "dtype": "float32",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "threads": torch.get_num_threads(),
    }


def _public_input_hashes(observation: Path, checkpoint: Path) -> dict[str, str]:
    return _public_packet_hashes(observation) | {checkpoint.resolve().as_posix(): file_sha256(checkpoint)}


def _artifact_hashes(paths: list[Path]) -> dict[str, str]:
    for path in paths:
        _must_file(path)
    return _hash_map(paths)


def _stage_integrity_before(*, observation: Path, checkpoint: Path,
                            artifacts: list[Path] = ()) -> dict[str, Any]:
    return {
        "public_input_hashes": _public_input_hashes(observation, checkpoint),
        "artifact_hashes": _artifact_hashes(artifacts),
        "protected_source_hashes": hash_files(protected_paths()),
        "shared_source_hashes": hash_files(SHARED_PATHS),
    }


def _stage_integrity_after(*, observation: Path, checkpoint: Path,
                           artifacts: list[Path] = ()) -> dict[str, Any]:
    return {
        "public_input_hashes": _public_input_hashes(observation, checkpoint),
        "artifact_hashes": _artifact_hashes(artifacts),
        "protected_source_hashes": hash_files(protected_paths()),
        "shared_source_hashes": hash_files(SHARED_PATHS),
    }


def _integrity_equal(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return before == after


def _persist_and_require_integrity(output: Path, before: dict[str, Any], *,
                                   observation: Path, checkpoint: Path,
                                   artifacts: list[Path] = ()) -> dict[str, Any]:
    try:
        after = _stage_integrity_after(observation=observation, checkpoint=checkpoint,
                                       artifacts=artifacts)
        unchanged = _integrity_equal(before, after)
        record = {"before": before, "after": after, "unchanged": unchanged}
    except Exception as error:
        record = {"before": before, "after": {"error_type": type(error).__name__, "error": str(error)},
                  "unchanged": False}
    write_json(output / "integrity.json", record)
    if record["unchanged"] is not True:
        raise PipelineError("stage public/source/protected integrity changed")
    return record


def _validate_source_tensors(S_hat: Any, target_z: Any) -> tuple[Tensor, Tensor]:
    expected = _expected_source_shape()
    if (not isinstance(S_hat, Tensor) or S_hat.dtype != torch.float32
            or S_hat.layout != torch.strided or tuple(S_hat.shape) != expected
            or not bool(torch.isfinite(S_hat).all())):
        raise PipelineError(f"S_hat must be finite contiguous-compatible FP32 {expected}")
    if (not isinstance(target_z, Tensor) or target_z.dtype != torch.float32
            or target_z.layout != torch.strided or tuple(target_z.shape) != expected
            or not bool(torch.isfinite(target_z).all()) or bool((target_z < 0).any())):
        raise PipelineError(f"target_z must be finite nonnegative FP32 {expected}")
    if not torch.equal(target_z, S_hat.abs()):
        raise PipelineError("target_z must equal abs(S_hat) exactly")
    return S_hat.detach().contiguous(), target_z.detach().contiguous()


def _public_packet_hashes(observation: Path) -> dict[str, str]:
    manifest_path = observation / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [manifest_path]
    for descriptor in manifest["layers"].values():
        paths.append(observation / descriptor["file"])
    return _hash_map(paths)


def _source_registration(*, observation: Path, checkpoint: Path, config: CombinationICAConfig,
                         input_hashes: dict[str, str], source_hashes: dict[str, str],
                         shared_hashes: dict[str, str], runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "REGISTERED",
        "method": METHOD, "stage": "SEPARATE", "registered_before_execution": True,
        "public_observation": observation.resolve().as_posix(),
        "checkpoint": checkpoint.resolve().as_posix(), "checkpoint_sha256": CHECKPOINT_SHA,
        "packet_hashes": _public_packet_hashes(observation),
        "input_hashes": input_hashes, "adapted_source_hashes": source_hashes,
        "shared_source_hashes": shared_hashes, "config": config.as_dict(), "runtime": runtime,
        "source_count": SOURCE_COUNT, "source_shape": list(_expected_source_shape()),
        "native_estimated_order": True, "private_data_used": False,
        "generator_used": False, "quality": "NOT_EVALUATED",
    }


def _source_hashes() -> dict[str, str]:
    paths = [Path(__file__), Path(__file__).with_name("signed_combination_ica.py")]
    return {path.resolve().as_posix(): file_sha256(path) for path in paths}


def _write_separation_result(output: Path, separation: SeparationResult,
                             registration: dict[str, Any], integrity: dict[str, Any],
                             started: float) -> dict[str, Any]:
    S_hat, target_z = _validate_source_tensors(separation.S_hat, separation.target_z)
    S_hat = S_hat.detach().cpu().contiguous()
    target_z = target_z.detach().cpu().contiguous()
    sources_path = output / "sources.pt"
    torch.save({"S_hat": S_hat, "target_z": target_z}, sources_path)
    write_json(output / "refinement_log.json", separation.refinement.log)
    result = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "COMPLETED",
        "method": METHOD, "stage": "SEPARATE", "quality": "NOT_EVALUATED",
        "config": registration["config"],
        "public_manifest_sha256": file_sha256(Path(registration["public_observation"]) / "manifest.json"),
        "checkpoint_sha256": registration["checkpoint_sha256"],
        "source_count": SOURCE_COUNT, "source_shape": list(_expected_source_shape()),
        "native_estimated_order": True, "S_hat_sha256": tensor_sha256(S_hat),
        "target_z_sha256": tensor_sha256(target_z), "target_z_exact_abs_S_hat": True,
        "S_hat_stats": tensor_stats(S_hat), "target_z_stats": tensor_stats(target_z),
        "sources_file_sha256": file_sha256(sources_path),
        "fastica": separation.metadata["fastica"], "refinement": separation.metadata["refinement"],
        "whitening": separation.metadata["whitening"],
        "final_losses": separation.refinement.final_losses,
        "timing": {"wall_seconds": time.perf_counter() - started},
        "private_data_used": False, "generator_used": False, "integrity": integrity,
    }
    write_json(output / "result.json", result)
    write_json(output / "registration.json", {**registration, "status": "COMPLETED",
                                                 "output_hashes": {
                                                     "sources.pt": result["sources_file_sha256"],
                                                     "S_hat": result["S_hat_sha256"],
                                                     "target_z": result["target_z_sha256"],
                                                 }})
    return result


def separate(*, observation: Path = PUBLIC_OBSERVATION, checkpoint: Path = CHECKPOINT,
             output: Path = DEFAULT_SEPARATION_OUTPUT, config: CombinationICAConfig | None = None,
             device: torch.device | str = "cpu") -> dict[str, Any]:
    """Decode classifier.0.weight and create the new public source artifact."""
    observation, checkpoint, output = observation.resolve(), checkpoint.resolve(), output.resolve()
    if observation != PUBLIC_OBSERVATION.resolve() or checkpoint != CHECKPOINT.resolve():
        raise PipelineError("separate accepts only the registered public observation/checkpoint")
    if output.exists():
        raise FileExistsError(f"refusing preexisting separation output: {output}")
    configure_runtime(42, threads=1)
    if file_sha256(checkpoint) != CHECKPOINT_SHA:
        raise PipelineError("checkpoint hash mismatch")
    manifest, entries = read_bundle(observation)
    q, scale = entries["classifier.0.weight"]
    X = (q.to(dtype=torch.float32) * scale).contiguous()
    if tuple(X.shape) != FC_SHAPES["classifier.0.weight"]:
        raise PipelineError("classifier.0.weight packet shape changed")
    config = _canonical_config(config)
    output.mkdir(parents=True, exist_ok=False)
    input_hashes = _public_input_hashes(observation, checkpoint)
    integrity_before = _stage_integrity_before(observation=observation, checkpoint=checkpoint)
    registration = _source_registration(
        observation=observation, checkpoint=checkpoint, config=config,
        input_hashes=input_hashes, source_hashes=_source_hashes(),
        shared_hashes=hash_files(SHARED_PATHS), runtime=_runtime(device, phase="SEPARATE"))
    registration["integrity_before"] = integrity_before
    write_json(output / "registration.json", registration)
    started = time.perf_counter()
    try:
        separation = separate_observation(X, SOURCE_COUNT, config=config, device=device)
        integrity = _persist_and_require_integrity(output, integrity_before,
                                                   observation=observation, checkpoint=checkpoint)
        result = _write_separation_result(output, separation, registration, integrity, started)
        return result
    except Exception as error:
        write_json(output / "failure.json", {"task_id": TASK_ID, "method": METHOD,
                                              "stage": "SEPARATE", "status": "FAILED",
                                              "error_type": type(error).__name__, "error": str(error),
                                              "wall_seconds": time.perf_counter() - started})
        raise


def _load_sources(source_root: Path) -> tuple[dict[str, Any], Tensor, Tensor]:
    source_root = source_root.resolve()
    registration = _read_json(source_root / "registration.json")
    result = _read_json(source_root / "result.json")
    if (registration.get("task_id") != TASK_ID or registration.get("plan_version") != PLAN_VERSION
            or registration.get("method") != METHOD or registration.get("stage") != "SEPARATE"
            or registration.get("status") != "COMPLETED"):
        raise PipelineError("source artifact method/stage identity mismatch")
    if (result.get("task_id") != TASK_ID or result.get("plan_version") != PLAN_VERSION
            or result.get("status") != "COMPLETED" or result.get("method") != METHOD
            or result.get("stage") != "SEPARATE"):
        raise PipelineError("source artifact is not a completed combination ICA result")
    if (registration.get("native_estimated_order") is not True
            or registration.get("private_data_used") is not False
            or registration.get("generator_used") is not False
            or registration.get("source_count") != SOURCE_COUNT
            or registration.get("source_shape") != list(_expected_source_shape())
            or registration.get("checkpoint_sha256") != CHECKPOINT_SHA
            or registration.get("config") != CANONICAL_SEPARATOR_CONFIG):
        raise PipelineError("source registration provenance or shape contract failed")
    sources_path = source_root / "sources.pt"
    _must_file(sources_path)
    output_hashes = registration.get("output_hashes")
    if (not isinstance(output_hashes, Mapping)
            or set(output_hashes) != {"sources.pt", "S_hat", "target_z"}
            or output_hashes["sources.pt"] != file_sha256(sources_path)):
        raise PipelineError("source artifact file hash mismatch")
    payload = torch.load(sources_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"S_hat", "target_z"}:
        raise PipelineError("source artifact schema must be exactly S_hat and target_z")
    S_hat, target_z = _validate_source_tensors(payload["S_hat"], payload["target_z"])
    if (result.get("source_count") != SOURCE_COUNT
            or result.get("source_shape") != list(_expected_source_shape())
            or result.get("native_estimated_order") is not True
            or result.get("config") != CANONICAL_SEPARATOR_CONFIG
            or result.get("public_manifest_sha256") != AUTHORIZED_MANIFEST_SHA
            or result.get("checkpoint_sha256") != CHECKPOINT_SHA
            or result.get("private_data_used") is not False
            or result.get("generator_used") is not False
            or result.get("S_hat_sha256") != tensor_sha256(S_hat)
            or result.get("target_z_sha256") != tensor_sha256(target_z)
            or result.get("sources_file_sha256") != file_sha256(sources_path)
            or result.get("target_z_exact_abs_S_hat") is not True
            or not isinstance(result.get("integrity"), Mapping)
            or result["integrity"].get("unchanged") is not True):
        raise PipelineError("source tensor hash mismatch")
    return registration, S_hat, target_z


def _validate_public_binding(registration: dict[str, Any], observation: Path, checkpoint: Path) -> dict[str, str]:
    if observation.resolve() != PUBLIC_OBSERVATION.resolve() or checkpoint.resolve() != CHECKPOINT.resolve():
        raise PipelineError("public binding path changed")
    expected_observation = registration.get("public_observation")
    expected_checkpoint = registration.get("checkpoint")
    if expected_observation is not None and Path(expected_observation).resolve() != observation.resolve():
        raise PipelineError("registered public observation path changed")
    if expected_checkpoint is not None and Path(expected_checkpoint).resolve() != checkpoint.resolve():
        raise PipelineError("registered checkpoint path changed")
    if file_sha256(checkpoint) != CHECKPOINT_SHA:
        raise PipelineError("public checkpoint hash changed")
    if file_sha256(observation / "manifest.json") != AUTHORIZED_MANIFEST_SHA:
        raise PipelineError("public observation manifest hash changed")
    actual = _public_packet_hashes(observation) | {checkpoint.resolve().as_posix(): file_sha256(checkpoint)}
    expected = registration.get("input_hashes", {})
    if not isinstance(expected, Mapping):
        raise PipelineError("registered public input hashes are not an object")
    normalized = {str(key).replace("\\", "/").casefold(): value for key, value in expected.items()}
    for key, value in actual.items():
        if normalized.get(key.replace("\\", "/").casefold()) != value:
            raise PipelineError(f"public input hash mismatch: {key}")
    if set(normalized) != {key.replace("\\", "/").casefold() for key in actual}:
        raise PipelineError("public input hash set changed")
    return actual


def _require_same_hashes(expected: Mapping[str, str], actual: Mapping[str, str], *, label: str) -> None:
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        raise PipelineError(f"{label} hash binding changed")
    normalize = lambda values: {str(key).replace("\\", "/").casefold(): value
                                for key, value in values.items()}
    if normalize(expected) != normalize(actual):
        raise PipelineError(f"{label} hash binding changed")


def _validate_candidate(path: Path, expected_hash: str | None = None) -> Tensor:
    _must_file(path)
    actual = file_sha256(path)
    if expected_hash is not None and actual != expected_hash:
        raise PipelineError(f"candidate hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"pixels"}:
        raise PipelineError("candidate schema must contain only pixels")
    pixels = payload["pixels"]
    if (not isinstance(pixels, Tensor) or pixels.dtype != torch.float32
            or tuple(pixels.shape) != (SOURCE_COUNT, 3, 224, 224)
            or not bool(torch.isfinite(pixels).all()) or bool((pixels < 0).any())
            or bool((pixels > 1).any())):
        raise PipelineError("candidate pixels have invalid dtype, shape, finite state, or range")
    return pixels.detach().contiguous()


def invoke_direct_fia(*, target_z: Tensor, output_dir: Path, device: torch.device | str,
                      direct_fia: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Bridge that makes the new target the actual Direct FIA input."""
    direct_fia = run_upstream_fia if direct_fia is None else direct_fia
    return direct_fia(target_z=target_z, device=device, config=dict(CANONICAL_FIA_CONFIG),
                      output_dir=output_dir, seed=42)


def invert(*, source_root: Path, output: Path = DEFAULT_INVERT_OUTPUT,
           observation: Path = PUBLIC_OBSERVATION, checkpoint: Path = CHECKPOINT,
           device: torch.device | str = "cuda",
           direct_fia: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Authenticate the new public target and call unchanged Direct FIA."""
    configure_runtime(42, threads=1)
    source_registration, _, target_z = _load_sources(source_root)
    input_hashes = _validate_public_binding(source_registration, observation, checkpoint)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing preexisting inversion output: {output}")
    runtime_before = _runtime(device, phase="INVERT_PRE_DIRECT_FIA")
    integrity_before = _stage_integrity_before(
        observation=observation, checkpoint=checkpoint,
        artifacts=[source_root.resolve() / "registration.json",
                   source_root.resolve() / "result.json", source_root.resolve() / "sources.pt"])
    output.mkdir(parents=True, exist_ok=False)
    registration = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "REGISTERED",
        "method": METHOD, "stage": "INVERT", "registered_before_execution": True,
        "source_root": source_root.resolve().as_posix(),
        "source_artifact_sha256": file_sha256(source_root / "sources.pt"),
        "target_z_sha256": tensor_sha256(target_z), "input_hashes": input_hashes,
        "checkpoint_sha256": CHECKPOINT_SHA, "fia_config": dict(CANONICAL_FIA_CONFIG),
        "runtime_pre_adapter": runtime_before, "integrity_before": integrity_before,
        "private_data_used": False, "generator_used": False, "quality": "NOT_EVALUATED",
    }
    write_json(output / "registration.json", registration)
    started = time.perf_counter()
    try:
        fia = invoke_direct_fia(target_z=target_z, output_dir=output / "direct_fia",
                                device=device, direct_fia=direct_fia)
        rec = fia.get("rec")
        if (not isinstance(rec, Tensor) or rec.dtype != torch.float32
                or tuple(rec.shape) != (SOURCE_COUNT, 3, 224, 224)
                or not bool(torch.isfinite(rec).all()) or bool((rec < 0).any())
                or bool((rec > 1).any())):
            raise PipelineError("Direct FIA returned invalid public candidate")
        candidate_path = output / "candidate.pt"
        torch.save({"pixels": rec.detach().contiguous().cpu()}, candidate_path)
        runtime_after = _runtime(device, phase="INVERT_POST_DIRECT_FIA")
        integrity = _persist_and_require_integrity(
            output, integrity_before, observation=observation, checkpoint=checkpoint,
            artifacts=[source_root.resolve() / "registration.json",
                       source_root.resolve() / "result.json", source_root.resolve() / "sources.pt"])
        result = {"task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "COMPLETED",
                  "method": METHOD, "stage": "INVERT", "quality": "NOT_EVALUATED",
                  "source_artifact_sha256": registration["source_artifact_sha256"],
                  "checkpoint_sha256": registration["checkpoint_sha256"],
                  "fia_config": dict(CANONICAL_FIA_CONFIG),
                  "candidate_sha256": file_sha256(candidate_path),
                  "target_z_sha256": tensor_sha256(target_z), "runtime_pre_adapter": runtime_before,
                  "runtime_post_adapter": runtime_after, "integrity": integrity, "direct_fia": {
                      key: value for key, value in fia.items() if key != "rec"},
                  "timing": {"wall_seconds": time.perf_counter() - started},
                  "private_data_used": False, "generator_used": False}
        write_json(output / "result.json", result)
        write_json(output / "direct_fia_log.json", fia.get("log", []))
        write_json(output / "registration.json", {**registration, "status": "COMPLETED",
                                                     "candidate_sha256": result["candidate_sha256"],
                                                     "runtime_post_adapter": runtime_after,
                                                     "integrity": integrity})
        return result
    except Exception as error:
        write_json(output / "failure.json", {"task_id": TASK_ID, "method": METHOD,
                                              "stage": "INVERT", "status": "FAILED",
                                              "error_type": type(error).__name__, "error": str(error),
                                              "wall_seconds": time.perf_counter() - started})
        raise


def build_refinement_labels(model: nn.Module, candidate: Tensor,
                            tables: Mapping[str, CellTable], device: torch.device) -> tuple[dict[str, Any], Tensor]:
    """Derive one frozen H1 coverage assignment from this candidate's public logits."""
    with torch.no_grad():
        features = _feature_forward(model, _to_device(candidate, device))
        logits = model.classifier(features)
    evidence, labels = labels_from_public_logits(logits, tables["classifier.6.bias"].q_cpu)
    if evidence.get("status") != "COMPLETED" or "H1" not in labels:
        raise PipelineError("coverage-based public H1 label assignment failed")
    return evidence, labels["H1"].detach().to(device=device, dtype=torch.int64)


PROFILE_FIELDS = {
    "name", "numel", "fixed_s", "code_mismatch_count", "code_mismatch_fraction",
    "bounded_cell_violation_count", "bounded_cell_violation_fraction",
    "mean_violation_in_s", "max_violation_in_s", "cell_penalty", "candidate_scale",
}


def _validate_endpoint_profile(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict) or not isinstance(profile.get("per_tensor"), dict):
        raise PipelineError("endpoint profile must contain all six FC tensors")
    if set(profile["per_tensor"]) != set(FC_NAMES):
        raise PipelineError("endpoint profile must contain all six FC tensors")
    if type(profile.get("exact_feasible")) is not bool:
        raise PipelineError("endpoint profile exact_feasible must be a bool")
    for field in ("overall_numel", "overall_code_mismatch_count",
                  "overall_bounded_cell_violation_count"):
        if type(profile.get(field)) is not int or profile[field] < 0:
            raise PipelineError(f"endpoint profile integer field is invalid: {field}")
    for field in ("overall_code_mismatch_fraction", "overall_bounded_cell_violation_fraction",
                  "cell_loss_equal_tensor_mean", "feature_norm_mean", "feature_norm_min",
                  "feature_norm_max"):
        if (not isinstance(profile.get(field), (int, float))
                or isinstance(profile[field], bool)
                or not math.isfinite(float(profile[field]))):
            raise PipelineError(f"endpoint profile numeric field is invalid: {field}")
    for name in FC_NAMES:
        item = profile["per_tensor"][name]
        if not isinstance(item, dict) or not PROFILE_FIELDS.issubset(item):
            raise PipelineError(f"endpoint profile fields are incomplete for {name}")
        if item["name"] != name:
            raise PipelineError(f"endpoint profile name mismatch for {name}")
        for field in ("numel", "code_mismatch_count", "bounded_cell_violation_count"):
            if type(item[field]) is not int or item[field] < 0:
                raise PipelineError(f"endpoint profile integer field is invalid for {name}: {field}")
        for field in PROFILE_FIELDS - {"name", "numel", "code_mismatch_count", "bounded_cell_violation_count"}:
            if not isinstance(item[field], (int, float)) or isinstance(item[field], bool) or not math.isfinite(float(item[field])):
                raise PipelineError(f"endpoint profile numeric field is invalid for {name}: {field}")
    return profile


def _endpoint_profile(model: nn.Module, pixels: Tensor, labels: Tensor,
                      tables: Mapping[str, CellTable], device: torch.device) -> dict[str, Any]:
    profile = _validate_endpoint_profile(canonical_profile(model, pixels, labels, tables, device))
    profile = dict(profile)
    profile["scale_diagnostics"] = {
        name: {"fixed_s": profile["per_tensor"][name]["fixed_s"],
               "candidate_scale": profile["per_tensor"][name]["candidate_scale"]}
        for name in FC_NAMES
    }
    profile["feasibility"] = {
        "exact_code_and_bounded_cell": profile["exact_feasible"],
        "overall_code_mismatch_count": profile["overall_code_mismatch_count"],
        "overall_bounded_cell_violation_count": profile["overall_bounded_cell_violation_count"],
    }
    return profile


def _validate_preflight(preflight: Any) -> dict[str, Any]:
    if not isinstance(preflight, dict) or preflight.get("feature_grad_enabled_equals_no_grad") is not True:
        raise PipelineError("actual-update preflight feature path did not match")
    if preflight.get("terminal_deltas_torch_equal") is not True:
        raise PipelineError("actual-update preflight terminal delta path did not match")
    per_tensor = preflight.get("per_tensor")
    if not isinstance(per_tensor, dict) or set(per_tensor) != set(FC_NAMES):
        raise PipelineError("actual-update preflight must contain all six FC tensors")
    required = {"torch_equal", "max_abs_difference", "max_difference_in_s",
                "fixed_s_code_mismatch_count"}
    for name in FC_NAMES:
        if not isinstance(per_tensor[name], dict) or not required.issubset(per_tensor[name]):
            raise PipelineError(f"actual-update preflight fields are incomplete for {name}")
    return preflight


def _to_device(value: Tensor, device: torch.device) -> Tensor:
    return value.to(device=device)


def invoke_actual_update_refinement(*, model: nn.Module, candidate: Tensor, target_z: Tensor,
                                    labels: Tensor, tables: Mapping[str, CellTable],
                                    device_tables: Mapping[str, DeviceCellTable], output: Path,
                                    device: torch.device, clock: Any,
                                    image_refiner: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Bridge that passes the new target into the unchanged six-FC cell runner."""
    image_refiner = run_arm if image_refiner is None else image_refiner
    return image_refiner(model, REFINE_ARM_NAME, candidate, target_z, labels,
                         FIXED_CELL_WEIGHT, tables, device_tables, output, device, clock)


def refine(*, source_root: Path, candidate_root: Path,
           output: Path = DEFAULT_REFINE_OUTPUT, observation: Path = PUBLIC_OBSERVATION,
           naive: Path | None = None, checkpoint: Path = CHECKPOINT,
           device_name: str = "cuda", image_refiner: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Future-only bridge to the existing actual-image six-FC cell refinement."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing preexisting refinement output: {output}")
    configure_runtime(42, threads=1)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise PipelineError("future actual-update refinement requires CUDA")
    clock = Clock(time.perf_counter())
    source_root, candidate_root = source_root.resolve(), candidate_root.resolve()
    source_registration, _, target_z = _load_sources(source_root)
    input_hashes = _validate_public_binding(source_registration, observation, checkpoint)
    candidate_registration = _read_json(candidate_root / "registration.json")
    candidate_result = _read_json(candidate_root / "result.json")
    if (candidate_registration.get("task_id") != TASK_ID
            or candidate_registration.get("plan_version") != PLAN_VERSION
            or candidate_registration.get("status") != "COMPLETED"
            or candidate_registration.get("method") != METHOD
            or candidate_registration.get("stage") != "INVERT"
            or candidate_registration.get("fia_config") != CANONICAL_FIA_CONFIG
            or candidate_registration.get("checkpoint_sha256") != CHECKPOINT_SHA
            or candidate_registration.get("private_data_used") is not False
            or candidate_registration.get("generator_used") is not False):
        raise PipelineError("candidate is not an authenticated combination ICA inversion artifact")
    if (candidate_result.get("task_id") != TASK_ID
            or candidate_result.get("plan_version") != PLAN_VERSION
            or candidate_result.get("status") != "COMPLETED"
            or candidate_result.get("method") != METHOD
            or candidate_result.get("stage") != "INVERT"
            or candidate_result.get("fia_config") != CANONICAL_FIA_CONFIG
            or candidate_result.get("checkpoint_sha256") != CHECKPOINT_SHA
            or candidate_result.get("candidate_sha256") != candidate_registration.get("candidate_sha256")
            or candidate_result.get("source_artifact_sha256") != candidate_registration.get("source_artifact_sha256")
            or candidate_result.get("target_z_sha256") != candidate_registration.get("target_z_sha256")
            or candidate_result.get("private_data_used") is not False
            or candidate_result.get("generator_used") is not False
            or not isinstance(candidate_result.get("integrity"), Mapping)
            or candidate_result["integrity"].get("unchanged") is not True):
        raise PipelineError("candidate result is not a complete authenticated inversion result")
    if candidate_registration.get("source_artifact_sha256") != file_sha256(source_root / "sources.pt"):
        raise PipelineError("candidate source artifact binding does not match source artifact")
    _require_same_hashes(candidate_registration.get("input_hashes", {}), input_hashes,
                         label="candidate public input")
    if candidate_registration.get("target_z_sha256") != tensor_sha256(target_z):
        raise PipelineError("candidate target binding does not match source artifact")
    candidate_path = candidate_root / "candidate.pt"
    candidate = _validate_candidate(candidate_path, candidate_registration.get("candidate_sha256"))
    integrity_artifacts = [source_root / "registration.json", source_root / "result.json",
                           source_root / "sources.pt", candidate_root / "registration.json",
                           candidate_root / "result.json", candidate_path]
    integrity_before = _stage_integrity_before(observation=observation, checkpoint=checkpoint,
                                               artifacts=integrity_artifacts)
    _, tables = load_public_cell_tables(observation)
    model = None
    try:
        model = load_model(checkpoint, device)
        labels_evidence, labels = build_refinement_labels(model, candidate, tables, device)
        device_tables = _device_tables(tables, device)
        output.mkdir(parents=True, exist_ok=False)
        registration = {
            "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "REGISTERED",
            "method": METHOD, "stage": "REFINE", "registered_before_execution": True,
            "source_root": source_root.as_posix(), "candidate_root": candidate_root.as_posix(),
            "source_artifact_sha256": file_sha256(source_root / "sources.pt"),
            "target_z_sha256": tensor_sha256(target_z), "candidate_sha256": candidate_registration["candidate_sha256"],
            "input_hashes": input_hashes, "protected_source_hashes": integrity_before["protected_source_hashes"],
            "shared_source_hashes": integrity_before["shared_source_hashes"],
            "checkpoint_sha256": CHECKPOINT_SHA, "runtime": _runtime(device, phase="REFINE"),
            "integrity_before": integrity_before, "arm": REFINE_ARM_NAME,
            "label_hypothesis": {"name": REFINE_LABEL_NAME,
                "rule": "coverage-based H1 from this candidate public logits and positive bias-code classes",
                "frozen_before_optimization": True},
            "h1_labels": labels.detach().cpu().tolist(), "label_evidence": labels_evidence,
            "fia_config": dict(CANONICAL_FIA_CONFIG), "objective": dict(CANONICAL_REFINE_OBJECTIVE),
            "private_data_used": False, "generator_used": False, "quality": "NOT_EVALUATED",
        }
        write_json(output / "registration.json", registration)
        preflight = _validate_preflight(compare_replay_paths(
            model, _to_device(candidate, device), labels, tables, device_tables, device))
        # Persist this successful gate before any 100-step image optimization.
        write_json(output / "preflight.json", preflight)
        start_profile = _endpoint_profile(model, _to_device(candidate, device), labels, tables, device)
        write_json(output / "start_profile.json", start_profile)
        actual_update_result = invoke_actual_update_refinement(
            model=model, candidate=candidate, target_z=target_z, labels=labels, tables=tables,
            device_tables=device_tables, output=output, device=device, clock=clock,
            image_refiner=image_refiner)
        if (not isinstance(actual_update_result, dict)
                or actual_update_result.get("status") != "COMPLETED"
                or actual_update_result.get("arm") != REFINE_ARM_NAME
                or actual_update_result.get("optimized_label_hypothesis") != REFINE_LABEL_NAME):
            raise PipelineError("actual-update result identity or label hypothesis is invalid")
        final_candidate_path = output / REFINE_ARM_NAME / "candidate.pt"
        final_candidate_hash = actual_update_result.get("candidate_sha256")
        if not isinstance(final_candidate_hash, str) or len(final_candidate_hash) != 64:
            raise PipelineError("actual-update result is missing the final candidate hash")
        final_candidate = _validate_candidate(final_candidate_path, final_candidate_hash)
        final_profile = _endpoint_profile(model, _to_device(final_candidate, device), labels, tables, device)
        write_json(output / "final_profile.json", final_profile)
        integrity = _persist_and_require_integrity(
            output, integrity_before, observation=observation, checkpoint=checkpoint,
            artifacts=integrity_artifacts)
        feasibility = {"start_exact_feasible": start_profile["exact_feasible"],
                       "final_exact_feasible": final_profile["exact_feasible"],
                       "software_completion": "COMPLETED"}
        final = {"task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "COMPLETED",
                 "method": METHOD, "stage": "REFINE", "quality": "NOT_EVALUATED",
                 "registration": registration, "preflight": preflight,
                 "start_profile": start_profile, "final_profile": final_profile,
                 "feasibility": feasibility, "actual_update_result": actual_update_result,
                 "integrity": integrity, "private_data_used": False, "generator_used": False}
        write_json(output / "result.json", final)
        write_json(output / "registration.json", {**registration, "status": "COMPLETED",
                                                     "integrity": integrity, "feasibility": feasibility})
        return final
    except Exception as error:
        if output.exists():
            write_json(output / "failure.json", {
                "task_id": TASK_ID, "method": METHOD, "stage": "REFINE", "status": "FAILED",
                "error_type": type(error).__name__, "error": str(error),
                "preflight_persisted": (output / "preflight.json").is_file(),
                "start_profile_persisted": (output / "start_profile.json").is_file(),
                "final_profile_persisted": (output / "final_profile.json").is_file(),
            })
        raise
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    separate_parser = sub.add_parser("separate")
    separate_parser.add_argument("--output", type=Path, default=DEFAULT_SEPARATION_OUTPUT)
    separate_parser.add_argument("--device", default="cpu")
    invert_parser = sub.add_parser("invert")
    invert_parser.add_argument("--source-root", type=Path, required=True)
    invert_parser.add_argument("--output", type=Path, default=DEFAULT_INVERT_OUTPUT)
    invert_parser.add_argument("--device", default="cuda")
    refine_parser = sub.add_parser("refine")
    refine_parser.add_argument("--source-root", type=Path, required=True)
    refine_parser.add_argument("--candidate-root", type=Path, required=True)
    refine_parser.add_argument("--output", type=Path, default=DEFAULT_REFINE_OUTPUT)
    refine_parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "separate":
            result = separate(output=args.output, device=args.device)
        elif args.command == "invert":
            result = invert(source_root=args.source_root, output=args.output, device=args.device)
        else:
            result = refine(source_root=args.source_root, candidate_root=args.candidate_root,
                            output=args.output, device_name=args.device)
    except Exception as error:
        print(f"{TASK_ID}: {type(error).__name__}: {error}", flush=True)
        return 1
    print(json.dumps({"task_id": TASK_ID, "method": METHOD, "stage": result["stage"],
                      "status": result["status"], "quality": result.get("quality", "NOT_EVALUATED")},
                     ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
