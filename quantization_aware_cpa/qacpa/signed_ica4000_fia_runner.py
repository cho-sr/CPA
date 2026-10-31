"""Isolated public runner for the registered 4000-step ICA -> Direct FIA arm.

This module deliberately has only two commands.  It reuses the unchanged
combination ICA tensor core with the one registered change, then passes the
final exact ``abs(S_hat)`` target to the unchanged upstream Direct FIA adapter.
It has no private-data, label, cell, replay, generator, or candidate-selection
interface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor

from .deterministic_hybrid_setup import configure_runtime
from .signed_baseline_observation import CHECKPOINT_SHA, FC_SHAPES, read_bundle
from .signed_combination_ica import (
    CombinationICAConfig,
    SeparationResult,
    separate_observation,
    tensor_sha256,
    tensor_stats,
)
from .signed_e2e_runner import REPO_ROOT, hash_files, protected_paths, write_json
from .upstream_naive import run_upstream_fia, source_hashes as upstream_source_hashes


TASK_ID = "CPA_SIGNED_ICA4000_FIA_20260917"
PLAN_VERSION = 1
PARENT_TASK_ID = "CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916"
METHOD = "INT4_combination_ica_gd4000_direct_fia"
SOURCE_COUNT = 16
FEATURE_DIM = 25088
PUBLIC_OBSERVATION = REPO_ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation"
CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth"
OLD_ROOT = REPO_ROOT / "quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916"
OLD_DIRECT_CANDIDATE = OLD_ROOT / "direct_fia/candidate.pt"
OLD_PRIVATE_REVIEW = REPO_ROOT / "quantization_aware_cpa/results_private/signed_combination_ica_B16_T10_20260916/review_20260917/comparison.json"
REVIEWED_COMPARISON_SHA = "bc07e948dada0aba09e14b4b7ff8fa65d7ba988315fcaa1801c43cb7c7880d04"
AUTHORIZED_MANIFEST_SHA = "9b4891765e6fc7e4adb1e3e11c158b3d4a2c3430d3397e1f1466ddafe2e1cd69"
CORE_SHA = "1c7482e1ed77ca4641cd8822ebb6046c5150e794c3838aacc3e3d575f3d992d6"
OLD_RUNNER_SHA = "a7ca10cb59f59c76e50c3c20a564ca6d13ee1795385fade62b302a44854a124f"
OLD_ARTIFACT_SHA = {
    "registration.json": "70f48861153e5155c5d9fe1c4a20851dc8b4ca3ee9317dbef362148bf6d3d484",
    "result.json": "80946a388a649ebb8a99bfcbdb9301659788d89e1cb7562c441dca43f6b4e36e",
    "sources.pt": "408099e99bb6ad1d8b721ff518c040ec9bec332827f0f36d3b9715ffeb4af5a2",
    "refinement_log.json": "c20f6a62b2e5161c63fc89fcfcbe9e2dee3221eb3956262c8dc7c6a7cf8d9d2f",
    "direct_fia/candidate.pt": "4efc29ad08b1ea8b0cee6991de6f185839708c2c72aaafb4fedfb307b5e680ce",
    "direct_fia/result.json": "91c05c281135ffff7dc2fb40fa14e213675d371f72c528036ac8b3e69515a6da",
    "direct_fia/registration.json": "9b4d66d5c35588df5d9e4e7fae5362643569908177b65514348e32584c239657",
}
FIA_CONFIG = {
    "n_iter": 25000, "lr": 0.1, "tv": 0.1, "fi": 1.0,
    "optimizer": "adam", "scheduler": "cosine", "log_every": 5000,
}
MAX_SECONDS = 45 * 60
MAX_CUDA_ALLOCATED_BYTES = 12 * 1024**3


class RunnerError(RuntimeError):
    """A registered public artifact or numerical contract failed."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _must_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise RunnerError(f"required regular file is missing: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    _must_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RunnerError(f"invalid UTF-8 JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise RunnerError(f"expected a JSON object: {path}")
    return value


def _expected_shape() -> tuple[int, int]:
    return SOURCE_COUNT, FEATURE_DIM


def old_config() -> dict[str, Any]:
    return CombinationICAConfig().as_dict()


def new_config() -> dict[str, Any]:
    return CombinationICAConfig(refinement_iterations=4000).as_dict()


def config_difference() -> dict[str, tuple[Any, Any]]:
    before, after = old_config(), new_config()
    return {key: (before[key], after[key]) for key in before if before[key] != after[key]}


def _public_packet_hashes(observation: Path = PUBLIC_OBSERVATION) -> dict[str, str]:
    observation = observation.resolve()
    manifest_path = observation / "manifest.json"
    _must_file(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RunnerError(f"invalid public observation manifest: {error}") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("layers"), dict):
        raise RunnerError("public observation manifest has no layer map")
    paths = [manifest_path]
    for descriptor in manifest["layers"].values():
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("file"), str):
            raise RunnerError("public observation layer descriptor is invalid")
        paths.append(observation / descriptor["file"])
    for path in paths:
        _must_file(path)
    return {path.resolve().as_posix(): file_sha256(path) for path in paths}


def public_input_hashes() -> dict[str, str]:
    return _public_packet_hashes() | {CHECKPOINT.resolve().as_posix(): file_sha256(CHECKPOINT)}


def old_artifact_hashes() -> dict[str, str]:
    actual = {}
    for relative, expected in OLD_ARTIFACT_SHA.items():
        path = OLD_ROOT / relative
        _must_file(path)
        digest = file_sha256(path)
        if digest != expected:
            raise RunnerError(f"archived 1000-step artifact hash mismatch: {relative}")
        actual[relative] = digest
    return actual


def _validate_old_control() -> dict[str, Any]:
    if file_sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RunnerError("checkpoint hash mismatch")
    if file_sha256(PUBLIC_OBSERVATION / "manifest.json") != AUTHORIZED_MANIFEST_SHA:
        raise RunnerError("public observation manifest hash mismatch")
    artifacts = old_artifact_hashes()
    registration = _read_json(OLD_ROOT / "registration.json")
    result = _read_json(OLD_ROOT / "result.json")
    old = old_config()
    if (registration.get("status") != "COMPLETED" or registration.get("stage") != "SEPARATE"
            or registration.get("config") != old or registration.get("source_count") != SOURCE_COUNT
            or registration.get("source_shape") != list(_expected_shape())
            or result.get("status") != "COMPLETED" or result.get("stage") != "SEPARATE"
            or result.get("config") != old or result.get("target_z_exact_abs_S_hat") is not True):
        raise RunnerError("archived 1000-step separation metadata is not canonical")
    fia_registration = _read_json(OLD_ROOT / "direct_fia/registration.json")
    fia_result = _read_json(OLD_ROOT / "direct_fia/result.json")
    if (fia_registration.get("status") != "COMPLETED" or fia_registration.get("stage") != "INVERT"
            or fia_registration.get("fia_config") != FIA_CONFIG
            or fia_result.get("status") != "COMPLETED"
            or fia_result.get("candidate_sha256") != file_sha256(OLD_DIRECT_CANDIDATE)
            or fia_result.get("direct_fia", {}).get("settings", {}).get("n_iter") != 25000):
        raise RunnerError("archived 1000-step Direct FIA metadata is not canonical")
    return {"artifact_hashes": artifacts, "registration": registration, "result": result,
            "fia_registration": fia_registration, "fia_result": fia_result}


def protected_integrity() -> dict[str, str]:
    return hash_files(protected_paths())


def shared_integrity() -> dict[str, str]:
    shared = [
        REPO_ROOT / "quantization_aware_cpa/qacpa/signed_e2e_cells.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/signed_e2e_runner.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/fc_replay.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/full_update_replay.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/full_update_adjoint.py",
    ]
    return hash_files(shared)


def _runtime(device: torch.device | str, phase: str) -> dict[str, Any]:
    device = torch.device(device)
    return {
        "phase": phase, "device": str(device), "torch": torch.__version__,
        "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "dtype": "float32", "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _stage_integrity(*, source_artifacts: list[Path] = ()) -> dict[str, Any]:
    return {
        "public_input_hashes": public_input_hashes(),
        "source_artifact_hashes": {path.resolve().as_posix(): file_sha256(path)
                                    for path in source_artifacts},
        "protected_source_hashes": protected_integrity(),
        "shared_source_hashes": shared_integrity(),
    }


def _require_unchanged(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    record = {"before": before, "after": after, "unchanged": before == after}
    if record["unchanged"] is not True:
        raise RunnerError("public input/source/protected integrity changed")
    return record


def _validate_tensors(S_hat: Any, target_z: Any) -> tuple[Tensor, Tensor]:
    expected = _expected_shape()
    for name, value in (("S_hat", S_hat), ("target_z", target_z)):
        if (not isinstance(value, Tensor) or value.layout != torch.strided
                or value.dtype != torch.float32 or tuple(value.shape) != expected
                or not bool(torch.isfinite(value).all())):
            raise RunnerError(f"{name} must be finite FP32 with shape {expected}")
    if bool((target_z < 0).any()) or not torch.equal(target_z, S_hat.abs()):
        raise RunnerError("target_z must be the exact elementwise abs(S_hat)")
    return S_hat.detach().cpu().contiguous(), target_z.detach().cpu().contiguous()


def _validate_candidate(path: Path, expected_hash: str | None = None) -> Tensor:
    _must_file(path)
    actual = file_sha256(path)
    if expected_hash is not None and actual != expected_hash:
        raise RunnerError(f"candidate hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"pixels"}:
        raise RunnerError("candidate schema must contain only pixels")
    pixels = payload["pixels"]
    if (not isinstance(pixels, Tensor) or pixels.dtype != torch.float32
            or tuple(pixels.shape) != (SOURCE_COUNT, 3, 224, 224)
            or not bool(torch.isfinite(pixels).all())
            or bool((pixels < 0).any()) or bool((pixels > 1).any())):
        raise RunnerError("candidate must be finite FP32 [16,3,224,224] in [0,1]")
    return pixels.detach().contiguous()


def _write_registration(output: Path, *, stage: str, config: dict[str, Any],
                        input_integrity: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    registration = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "REGISTERED",
        "parent_task_id": PARENT_TASK_ID, "method": METHOD, "stage": stage,
        "registered_before_execution": True, "variant": "ICA_GD4000_then_Direct_FIA25000",
        "public_observation": PUBLIC_OBSERVATION.resolve().as_posix(),
        "checkpoint": CHECKPOINT.resolve().as_posix(), "checkpoint_sha256": CHECKPOINT_SHA,
        "public_manifest_sha256": AUTHORIZED_MANIFEST_SHA,
        "input_hashes": public_input_hashes(), "old_control_hashes": old_artifact_hashes(),
        "config": config, "config_difference_from_control": config_difference(),
        "runtime": runtime, "source_count": SOURCE_COUNT,
        "source_shape": list(_expected_shape()), "native_estimated_order": True,
        "target_contract": "target_z equals abs(S_hat) exactly in native estimated order",
        "private_data_used": False, "generator_used": False,
        "cell_or_replay_used": False, "quality": "NOT_EVALUATED",
        "integrity_before": input_integrity,
    }
    write_json(output / "registration.json", registration)
    return registration


def _load_source(source_root: Path) -> tuple[dict[str, Any], Tensor, Tensor]:
    source_root = source_root.resolve()
    registration = _read_json(source_root / "registration.json")
    result = _read_json(source_root / "result.json")
    expected_config = new_config()
    if (registration.get("task_id") != TASK_ID or registration.get("status") != "COMPLETED"
            or registration.get("stage") != "SEPARATE" or registration.get("method") != METHOD
            or registration.get("config") != expected_config
            or registration.get("source_count") != SOURCE_COUNT
            or registration.get("source_shape") != list(_expected_shape())
            or registration.get("private_data_used") is not False
            or registration.get("generator_used") is not False):
        raise RunnerError("4000 source registration is not authenticated")
    if (result.get("task_id") != TASK_ID or result.get("status") != "COMPLETED"
            or result.get("stage") != "SEPARATE" or result.get("config") != expected_config
            or result.get("target_z_exact_abs_S_hat") is not True):
        raise RunnerError("4000 source result is not authenticated")
    prefix = _read_json(source_root / "prefix_check.json")
    if (prefix.get("status") != "PASSED" or prefix.get("iterations_compared") != 1000
            or prefix.get("old_log_sha256") != OLD_ARTIFACT_SHA["refinement_log.json"]):
        raise RunnerError("authenticated 1000-step prefix check is missing")
    source_path = source_root / "sources.pt"
    _must_file(source_path)
    outputs = registration.get("output_hashes")
    if (not isinstance(outputs, Mapping) or outputs.get("sources.pt") != file_sha256(source_path)):
        raise RunnerError("4000 source file hash binding changed")
    output_artifacts = registration.get("output_artifact_hashes")
    if (not isinstance(output_artifacts, Mapping)
            or output_artifacts.get("refinement_log.json") != file_sha256(source_root / "refinement_log.json")
            or output_artifacts.get("prefix_check.json") != file_sha256(source_root / "prefix_check.json")):
        raise RunnerError("4000 source log/prefix hash binding changed")
    payload = torch.load(source_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"S_hat", "target_z"}:
        raise RunnerError("source payload must contain exactly S_hat and target_z")
    S_hat, target_z = _validate_tensors(payload["S_hat"], payload["target_z"])
    if (result.get("S_hat_sha256") != tensor_sha256(S_hat)
            or result.get("target_z_sha256") != tensor_sha256(target_z)
            or result.get("sources_file_sha256") != file_sha256(source_path)
            or result.get("integrity", {}).get("unchanged") is not True):
        raise RunnerError("4000 source tensor hash binding changed")
    return registration, S_hat, target_z


def _validate_public_binding(registration: Mapping[str, Any]) -> None:
    if registration.get("public_observation") != PUBLIC_OBSERVATION.resolve().as_posix():
        raise RunnerError("registered public observation path changed")
    if registration.get("checkpoint") != CHECKPOINT.resolve().as_posix():
        raise RunnerError("registered checkpoint path changed")
    actual = public_input_hashes()
    expected = registration.get("input_hashes")
    normalize = lambda values: {str(k).replace("\\", "/").casefold(): str(v)
                                for k, v in values.items()}
    if not isinstance(expected, Mapping) or normalize(expected) != normalize(actual):
        raise RunnerError("public input hash binding changed")
    if file_sha256(CHECKPOINT) != CHECKPOINT_SHA or file_sha256(PUBLIC_OBSERVATION / "manifest.json") != AUTHORIZED_MANIFEST_SHA:
        raise RunnerError("registered public input pin changed")


def _prefix_check(log: list[dict[str, Any]]) -> dict[str, Any]:
    old_log = json.loads((OLD_ROOT / "refinement_log.json").read_text(encoding="utf-8"))
    if not isinstance(old_log, list) or len(old_log) != 1000:
        raise RunnerError("archived refinement log is not the expected 1000-row list")
    if len(log) < 1000:
        raise RunnerError("4000 run did not persist the complete common prefix")
    passed = log[:1000] == old_log
    mismatch = None
    if not passed:
        for index, (actual, expected) in enumerate(zip(log[:1000], old_log)):
            if actual != expected:
                mismatch = {"zero_based_index": index, "actual": actual, "expected": expected}
                break
    return {
        "status": "PASSED" if passed else "FAILED",
        "iterations_compared": 1000, "comparison": "exact structured JSON values",
        "mismatch": mismatch, "old_log_sha256": OLD_ARTIFACT_SHA["refinement_log.json"],
    }


def separate(*, output: Path, device: str = "cpu") -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing preexisting public output: {output}")
    if device != "cpu":
        raise RunnerError("the registered separator device is CPU")
    started = time.perf_counter()
    try:
        configure_runtime(42, threads=1)
        if file_sha256(CHECKPOINT) != CHECKPOINT_SHA:
            raise RunnerError("checkpoint hash mismatch")
        if file_sha256(PUBLIC_OBSERVATION / "manifest.json") != AUTHORIZED_MANIFEST_SHA:
            raise RunnerError("public observation manifest hash mismatch")
        control = _validate_old_control()
        manifest, entries = read_bundle(PUBLIC_OBSERVATION)
        q, scale = entries["classifier.0.weight"]
        X = (q.to(dtype=torch.float32) * scale).contiguous()
        if tuple(X.shape) != (4096, FEATURE_DIM) or X.dtype != torch.float32:
            raise RunnerError("decoded classifier.0.weight is not [4096,25088] FP32")
        config = new_config()
        if config_difference() != {"refinement_iterations": (1000, 4000)}:
            raise RunnerError("4000 config differs from the control in more than one field")
        output.mkdir(parents=True, exist_ok=False)
        before = _stage_integrity()
        registration = _write_registration(output, stage="SEPARATE", config=config,
                                            input_integrity=before,
                                            runtime=_runtime("cpu", "SEPARATE_PRE_ICA"))
        write_json(output / "control_authentication.json", {
            "status": "PASSED", "old_control_hashes": control["artifact_hashes"],
            "old_control_task_id": control["registration"].get("task_id"),
            "old_control_config": control["registration"].get("config"),
            "public_manifest": manifest.get("format"),
        })
        separation = separate_observation(X, SOURCE_COUNT, config=CombinationICAConfig(**config), device="cpu")
        S_hat, target_z = _validate_tensors(separation.S_hat, separation.target_z)
        source_path = output / "sources.pt"
        torch.save({"S_hat": S_hat, "target_z": target_z}, source_path)
        write_json(output / "refinement_log.json", separation.refinement.log)
        prefix = _prefix_check(separation.refinement.log)
        write_json(output / "prefix_check.json", prefix)
        # Generated stage files are recorded separately; the before/after
        # integrity comparison covers only reusable public inputs and sources.
        after = _stage_integrity()
        integrity = _require_unchanged(before, after)
        if prefix["status"] != "PASSED":
            raise RunnerError("the common 1000-step refinement prefix does not reproduce exactly")
        result = {
            "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "COMPLETED",
            "parent_task_id": PARENT_TASK_ID, "method": METHOD, "stage": "SEPARATE",
            "quality": "NOT_EVALUATED", "config": config,
            "public_manifest_sha256": file_sha256(PUBLIC_OBSERVATION / "manifest.json"),
            "checkpoint_sha256": CHECKPOINT_SHA, "source_count": SOURCE_COUNT,
            "source_shape": list(_expected_shape()), "native_estimated_order": True,
            "S_hat_sha256": tensor_sha256(S_hat), "target_z_sha256": tensor_sha256(target_z),
            "target_z_exact_abs_S_hat": True, "S_hat_stats": tensor_stats(S_hat),
            "target_z_stats": tensor_stats(target_z), "sources_file_sha256": file_sha256(source_path),
            "fastica": separation.metadata["fastica"], "refinement": separation.metadata["refinement"],
            "whitening": separation.metadata["whitening"], "final_losses": separation.refinement.final_losses,
            "trajectory_checkpoints": {str(iteration): separation.refinement.log[iteration - 1]
                                        for iteration in (1000, 2000, 3000, 4000)},
            "prefix_check": prefix, "control_fastica": control["result"].get("fastica"),
            "output_artifact_hashes": {
                "refinement_log.json": file_sha256(output / "refinement_log.json"),
                "prefix_check.json": file_sha256(output / "prefix_check.json"),
            },
            "timing": {"wall_seconds": time.perf_counter() - started},
            "private_data_used": False, "generator_used": False, "cell_or_replay_used": False,
            "integrity": integrity,
        }
        write_json(output / "result.json", result)
        write_json(output / "registration.json", {**registration, "status": "COMPLETED",
                    "output_hashes": {"sources.pt": result["sources_file_sha256"],
                                      "S_hat": result["S_hat_sha256"], "target_z": result["target_z_sha256"]},
                    "prefix_check": prefix, "integrity": integrity,
                    "output_artifact_hashes": result["output_artifact_hashes"]})
        return result
    except Exception as error:
        if output.exists():
            write_json(output / "failure.json", {"task_id": TASK_ID, "stage": "SEPARATE",
                       "status": "FAILED", "error_type": type(error).__name__, "error": str(error),
                       "wall_seconds": time.perf_counter() - started})
        raise


def invoke_direct_fia(*, target_z: Tensor, output_dir: Path, device: str = "cuda",
                      direct_fia: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    if direct_fia is None:
        direct_fia = run_upstream_fia
    if device != "cuda":
        raise RunnerError("the registered FIA device is CUDA")
    return direct_fia(target_z=target_z, device=device, config=dict(FIA_CONFIG),
                      output_dir=output_dir, seed=42)


def invert(*, source_root: Path, output: Path, device: str = "cuda",
           direct_fia: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing preexisting FIA output: {output}")
    started = time.perf_counter()
    try:
        configure_runtime(42, threads=1)
        registration_source, _, target_z = _load_source(source_root)
        _validate_public_binding(registration_source)
        before = _stage_integrity(source_artifacts=[source_root.resolve() / "registration.json",
                                                    source_root.resolve() / "result.json",
                                                    source_root.resolve() / "sources.pt",
                                                    source_root.resolve() / "refinement_log.json",
                                                    source_root.resolve() / "prefix_check.json"])
        output.mkdir(parents=True, exist_ok=False)
        runtime_before = _runtime("cuda", "INVERT_PRE_DIRECT_FIA")
        registration = {
            "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "REGISTERED",
            "parent_task_id": PARENT_TASK_ID, "method": METHOD, "stage": "INVERT",
            "registered_before_execution": True, "source_root": source_root.resolve().as_posix(),
            "source_artifact_sha256": file_sha256(source_root / "sources.pt"),
            "target_z_sha256": tensor_sha256(target_z), "input_hashes": public_input_hashes(),
            "checkpoint_sha256": CHECKPOINT_SHA, "fia_config": dict(FIA_CONFIG),
            "runtime_pre_adapter": runtime_before, "integrity_before": before,
            "target_from_4000_source": True, "private_data_used": False,
            "generator_used": False, "cell_or_replay_used": False, "quality": "NOT_EVALUATED",
        }
        write_json(output / "registration.json", registration)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        fia = invoke_direct_fia(target_z=target_z, output_dir=output / "upstream_fia",
                                device=device, direct_fia=direct_fia)
        rec = fia.get("rec")
        if (not isinstance(rec, Tensor) or rec.dtype != torch.float32
                or tuple(rec.shape) != (SOURCE_COUNT, 3, 224, 224)
                or not bool(torch.isfinite(rec).all()) or bool((rec < 0).any())
                or bool((rec > 1).any())):
            raise RunnerError("Direct FIA returned an invalid public candidate")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_allocated())
        else:
            peak = 0
        if peak > MAX_CUDA_ALLOCATED_BYTES:
            raise RunnerError(f"CUDA allocated cap exceeded: {peak} > {MAX_CUDA_ALLOCATED_BYTES}")
        candidate_path = output / "candidate.pt"
        torch.save({"pixels": rec.detach().cpu().contiguous()}, candidate_path)
        runtime_after = _runtime("cuda", "INVERT_POST_DIRECT_FIA")
        after = _stage_integrity(source_artifacts=[source_root.resolve() / "registration.json",
                                                   source_root.resolve() / "result.json",
                                                   source_root.resolve() / "sources.pt",
                                                   source_root.resolve() / "refinement_log.json",
                                                   source_root.resolve() / "prefix_check.json"])
        integrity = _require_unchanged(before, after)
        result = {
            "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "COMPLETED",
            "parent_task_id": PARENT_TASK_ID, "method": METHOD, "stage": "INVERT",
            "quality": "NOT_EVALUATED", "source_artifact_sha256": registration["source_artifact_sha256"],
            "target_z_sha256": tensor_sha256(target_z), "target_from_4000_source": True,
            "checkpoint_sha256": CHECKPOINT_SHA, "fia_config": dict(FIA_CONFIG),
            "candidate_sha256": file_sha256(candidate_path), "runtime_pre_adapter": runtime_before,
            "runtime_post_adapter": runtime_after, "integrity": integrity,
            "direct_fia": {key: value for key, value in fia.items() if key != "rec"},
            "resource": {"cuda_max_memory_allocated_bytes": peak,
                         "cuda_allocated_cap_bytes": MAX_CUDA_ALLOCATED_BYTES},
            "timing": {"wall_seconds": time.perf_counter() - started},
            "private_data_used": False, "generator_used": False, "cell_or_replay_used": False,
        }
        write_json(output / "result.json", result)
        write_json(output / "direct_fia_log.json", fia.get("log", []))
        write_json(output / "registration.json", {**registration, "status": "COMPLETED",
                    "candidate_sha256": result["candidate_sha256"],
                    "runtime_post_adapter": runtime_after, "integrity": integrity,
                    "resource": result["resource"]})
        return result
    except Exception as error:
        if output.exists():
            write_json(output / "failure.json", {"task_id": TASK_ID, "stage": "INVERT",
                       "status": "FAILED", "error_type": type(error).__name__, "error": str(error),
                       "wall_seconds": time.perf_counter() - started})
        raise


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    separate_parser = sub.add_parser("separate")
    separate_parser.add_argument("--output", type=Path, required=True)
    separate_parser.add_argument("--device", default="cpu")
    invert_parser = sub.add_parser("invert")
    invert_parser.add_argument("--source-root", type=Path, required=True)
    invert_parser.add_argument("--output", type=Path, required=True)
    invert_parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "separate":
            result = separate(output=args.output, device=args.device)
        else:
            result = invert(source_root=args.source_root, output=args.output, device=args.device)
    except Exception as error:
        print(f"{TASK_ID}: {type(error).__name__}: {error}", flush=True)
        return 1
    print(json.dumps({"task_id": TASK_ID, "stage": result["stage"],
                      "status": result["status"], "quality": result["quality"]},
                     allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
