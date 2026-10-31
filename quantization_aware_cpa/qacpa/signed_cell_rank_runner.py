"""CLI boundary for the signed external rank/cell comparison arm.

``preprocess`` consumes only the existing public signed ``q,s`` observation and
writes a finalized CPU FP32 ``V`` artifact.  ``attack`` is a future public-only
entry point: it feeds that ``V`` to the unchanged original CPA and Direct FIA
adapters.  Neither command collects updates, loads private truth, estimates
labels, replays SGD, or changes the canonical CPA/FIA implementation.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import re
import time
from typing import Any

import torch

from . import signed_cell_rank as core
from . import upstream_naive
from .signed_baseline import (
    ATTACK_LAYER,
    CPA_CONFIG,
    FIA_CONFIG,
    SEED,
    configure,
    runtime,
    source_metrics,
    telemetry,
    verify_checkpoint,
)
from .signed_baseline_observation import CHECKPOINT_SHA, FC_SHAPES, digest, read_bundle
from quantized_update_cpa.trial_metrics import metric_record, new_trial_metrics


FORMAT = "signed_external_uncentered_gram_eigh_cell_rank_v1"
SCIENTIFIC_RANK = 16
SCIENTIFIC_AP_ITERATIONS = 5
DEFAULT_CHUNK_ROWS = 64
VALID_AP_ITERATIONS = (0, SCIENTIFIC_AP_ITERATIONS)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _resolve(path: str | Path) -> Path:
    return Path(path).resolve()


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _new_output(path: str | Path, *inputs: str | Path) -> Path:
    result = _resolve(path)
    if result.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {result}")
    for value in inputs:
        if _overlap(result, _resolve(value)):
            raise ValueError(f"output overlaps input: {result}")
    return result


def _public_observation(path: str | Path):
    folder = _resolve(path)
    manifest, entries = read_bundle(folder)
    if manifest.get("checkpoint_sha256") != CHECKPOINT_SHA:
        raise ValueError("public observation checkpoint digest mismatch")
    if ATTACK_LAYER not in entries:
        raise ValueError(f"public observation lacks {ATTACK_LAYER}")
    q, s = entries[ATTACK_LAYER]
    expected = tuple(FC_SHAPES[ATTACK_LAYER])
    if tuple(q.shape) != expected or q.dtype != torch.int8 or q.device.type != "cpu":
        raise ValueError("signed attack-layer codes have the wrong shape or dtype")
    if s.ndim != 0 or s.dtype != torch.float32 or s.device.type != "cpu":
        raise ValueError("signed attack-layer scale must be a CPU FP32 scalar")
    return folder, manifest, entries, q, s


def _layer_hashes(manifest: dict) -> dict[str, dict[str, Any]]:
    result = {}
    for name, descriptor in manifest.get("layers", {}).items():
        result[name] = {key: descriptor[key] for key in descriptor if key in {"file", "sha256", "shape", "numel", "file_bytes"}}
    return result


def _declared_ap_iterations(value: Any, *, label: str = "AP iterations") -> int:
    if type(value) is not int or type(value) is bool or value not in VALID_AP_ITERATIONS:
        raise ValueError(f"{label} must be explicitly 0 (AP0) or {SCIENTIFIC_AP_ITERATIONS} (AP5)")
    return value


def _require_preprocess_metadata(metadata: Any, *, label: str) -> dict[str, Any]:
    if type(metadata) is not dict:
        raise ValueError(f"{label} must be an object")
    required_types = {
        "format": str, "schema_version": int, "arm": str, "attack_layer": str,
        "rank": int, "ap_iterations": int, "centered": bool,
        "rank_backend": str, "projection_dtype": str, "projection_device": str,
        "final_selection": str, "scale_is_metric_only": bool,
    }
    for key, expected_type in required_types.items():
        if key not in metadata:
            raise ValueError(f"{label}.{key} is missing")
        if type(metadata[key]) is not expected_type:
            raise ValueError(f"{label}.{key} has the wrong type")
    if metadata["schema_version"] != 1 or metadata["format"] != FORMAT:
        raise ValueError(f"{label} format/schema mismatch")
    if metadata["arm"] != "INT4_cell_rank" or metadata["attack_layer"] != ATTACK_LAYER:
        raise ValueError(f"{label} arm/layer mismatch")
    if metadata["rank"] != SCIENTIFIC_RANK or metadata["centered"] is not False:
        raise ValueError(f"{label} rank/centering contract mismatch")
    if metadata["rank_backend"] != core.RANK_BACKEND or metadata["projection_dtype"] != "torch.float32":
        raise ValueError(f"{label} rank backend/dtype mismatch")
    if metadata["projection_device"] not in {"cpu", "cuda"}:
        raise ValueError(f"{label} projection device is invalid")
    if metadata["final_selection"] != "final_iteration_only" or metadata["scale_is_metric_only"] is not True:
        raise ValueError(f"{label} selection/scale contract mismatch")
    _declared_ap_iterations(metadata["ap_iterations"], label=f"{label}.ap_iterations")
    return metadata


def _validate_preprocessed_contract(registration: dict[str, Any], preprocessing: dict[str, Any],
                                    *, expected_ap_iterations: int, q: torch.Tensor,
                                    s: torch.Tensor) -> None:
    _require_preprocess_metadata(registration, label="registered_preprocess")
    _require_preprocess_metadata(preprocessing, label="preprocessing_result")
    expected_ap_iterations = _declared_ap_iterations(expected_ap_iterations, label="expected AP iterations")
    shared = ("format", "schema_version", "arm", "attack_layer", "rank", "ap_iterations",
              "centered", "rank_backend", "projection_dtype", "projection_device",
              "final_selection", "scale_is_metric_only")
    for key in shared:
        if registration[key] != preprocessing[key]:
            raise ValueError(f"preprocessing registration/result mismatch: {key}")
    if registration["ap_iterations"] != expected_ap_iterations:
        raise ValueError("preprocessed artifact AP iteration does not match the declared arm")
    if (type(registration.get("q_shape")) is not list
            or any(type(item) is not int for item in registration["q_shape"])
            or type(registration.get("s_shape")) is not list
            or any(type(item) is not int for item in registration["s_shape"])):
        raise ValueError("preprocessing q/s shapes have the wrong type")
    if registration["q_shape"] != list(q.shape) or registration["q_dtype"] != str(q.dtype):
        raise ValueError("preprocessing q metadata does not match the public observation")
    if registration["s_shape"] != list(s.shape) or registration["s_dtype"] != str(s.dtype):
        raise ValueError("preprocessing s metadata does not match the public observation")
    for key in ("public_manifest_sha256", "y_q_tensor_sha256", "v_file_sha256", "v_tensor_sha256",
                "diagnostics_sha256"):
        if key not in registration:
            raise ValueError(f"registered_preprocess.{key} is missing")
        if type(registration[key]) is not str or SHA256_RE.fullmatch(registration[key]) is None:
            raise ValueError(f"registered_preprocess.{key} is not a valid SHA256")
    if (type(registration.get("public_checkpoint_sha256")) is not str
            or SHA256_RE.fullmatch(registration["public_checkpoint_sha256"]) is None
            or registration["public_checkpoint_sha256"] != CHECKPOINT_SHA):
        raise ValueError("registered_preprocess public checkpoint binding is invalid")
    for key in ("status", "preprocess_status", "attack_status", "evaluation_status",
                "full_experiment_status", "final_v_file_sha256", "final_v_tensor_sha256"):
        if key not in preprocessing:
            raise ValueError(f"preprocessing_result.{key} is missing")
    if (preprocessing["status"] != "COMPLETED" or preprocessing["preprocess_status"] != "COMPLETED"
            or preprocessing["attack_status"] != "NOT_RUN"
            or preprocessing["evaluation_status"] != "NOT_RUN"
            or preprocessing["full_experiment_status"] != "NOT_RUN"):
        raise ValueError("preprocessing artifact stage status is not finalized")
    if preprocessing["final_v_file_sha256"] != registration["v_file_sha256"]:
        raise ValueError("preprocessing V file hash is not cross-consistent")
    if preprocessing["final_v_tensor_sha256"] != registration["v_tensor_sha256"]:
        raise ValueError("preprocessing V tensor hash is not cross-consistent")
    for key in ("public_manifest_sha256", "final_v_file_sha256", "final_v_tensor_sha256"):
        if type(preprocessing[key]) is not str or SHA256_RE.fullmatch(preprocessing[key]) is None:
            raise ValueError(f"preprocessing_result.{key} is not a valid SHA256")


def _preprocess_registration(*, public_folder: Path, manifest: dict, q: torch.Tensor, s: torch.Tensor,
                             y_q: torch.Tensor, rank: int, iterations: int, projection_device: str,
                             chunk_rows: int) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "schema_version": 1,
        "arm": "INT4_cell_rank",
        "attack_layer": ATTACK_LAYER,
        "public_observation": str(public_folder),
        "public_manifest_sha256": digest(public_folder / "manifest.json"),
        "public_checkpoint_sha256": manifest["checkpoint_sha256"],
        "public_layers": _layer_hashes(manifest),
        "q_shape": list(q.shape),
        "q_dtype": str(q.dtype),
        "s_shape": list(s.shape),
        "s_dtype": str(s.dtype),
        "codec_metadata": "quantized_update_cpa.cell_utils.CODEC_METADATA",
        "input": "Y_q = dequantize(q, s) = q*s, public signed per-tensor scale",
        "y_q_tensor_sha256": core.tensor_sha256(y_q),
        "rank": rank,
        "ap_iterations": iterations,
        "final_selection": "final_iteration_only",
        "rank_backend": core.RANK_BACKEND,
        "projection_device": projection_device,
        "projection_dtype": "torch.float32",
        "chunk_rows": chunk_rows,
        "centered": False,
        "scale_is_metric_only": True,
        "private_inputs_used": False,
        "full_experiment_status": "NOT_RUN",
        "preprocess_status": "COMPLETED",
        "attack_status": "NOT_RUN",
        "evaluation_status": "NOT_RUN",
    }


def preprocess(*, public_observation: str | Path, output: str | Path,
               ap_iterations: int = SCIENTIFIC_AP_ITERATIONS,
               projection_device: str = "cuda", chunk_rows: int = DEFAULT_CHUNK_ROWS):
    """Create a finalized external preprocessing artifact from public q,s."""
    output = _new_output(output, public_observation)
    public_folder, manifest, _entries, q, s = _public_observation(public_observation)
    y_q = (q.to(dtype=torch.float32) * s).contiguous()
    if tuple(y_q.shape) != tuple(FC_SHAPES[ATTACK_LAYER]):
        raise ValueError("decoded attack-layer shape mismatch")
    registration = _preprocess_registration(
        public_folder=public_folder, manifest=manifest, q=q, s=s, y_q=y_q,
        rank=SCIENTIFIC_RANK, iterations=ap_iterations,
        projection_device=str(torch.device(projection_device)), chunk_rows=chunk_rows)
    result = core.alternating_projection(
        y_q, q, s, rank=SCIENTIFIC_RANK, iterations=ap_iterations,
        projection_device=projection_device, chunk_rows=chunk_rows)
    final_fit = core.fixed_cell_metrics(result.final_v, q, s, chunk_rows=chunk_rows)
    if not final_fit["exact_code_match"] or not final_fit["bounded_cell_satisfied"]:
        raise ValueError("final V failed fixed observed-code/cell validation")
    output.mkdir(parents=True)
    torch.save(y_q, output / "Y_q.pt")
    torch.save(result.final_v, output / "V.pt")
    if result.final_m is not None:
        torch.save(result.final_m, output / "M.pt")
    _write_jsonl(output / "diagnostics.jsonl", result.diagnostics)
    registration.update({
        "y_q_file_sha256": core.file_sha256(output / "Y_q.pt"),
        "v_file_sha256": core.file_sha256(output / "V.pt"),
        "v_tensor_sha256": core.tensor_sha256(result.final_v),
        "m_file_sha256": core.file_sha256(output / "M.pt") if result.final_m is not None else None,
        "m_tensor_sha256": core.tensor_sha256(result.final_m) if result.final_m is not None else None,
        "diagnostics_sha256": core.file_sha256(output / "diagnostics.jsonl"),
    })
    write_json(output / "registered_preprocess.json", registration)
    preprocessing = {
        "status": "COMPLETED",
        "format": FORMAT,
        "schema_version": 1,
        "preprocess_status": "COMPLETED",
        "attack_status": "NOT_RUN",
        "evaluation_status": "NOT_RUN",
        "full_experiment_status": "NOT_RUN",
        "arm": "INT4_cell_rank",
        "attack_layer": ATTACK_LAYER,
        "rank": SCIENTIFIC_RANK,
        "ap_iterations": ap_iterations,
        "centered": False,
        "rank_backend": core.RANK_BACKEND,
        "projection_device": str(torch.device(projection_device)),
        "projection_dtype": "torch.float32",
        "final_selection": "final_iteration_only",
        "public_manifest_sha256": registration["public_manifest_sha256"],
        "y_q_tensor_sha256": registration["y_q_tensor_sha256"],
        "y_q_file_sha256": registration["y_q_file_sha256"],
        "final_v_tensor_sha256": registration["v_tensor_sha256"],
        "final_v_file_sha256": registration["v_file_sha256"],
        "final_m_tensor_sha256": registration["m_tensor_sha256"],
        "final_m_file_sha256": registration["m_file_sha256"],
        "final_v_fixed_scale_metrics": final_fit,
        "scale_is_metric_only": True,
        "diagnostic_count": len(result.diagnostics),
        "diagnostics_sha256": registration["diagnostics_sha256"],
        "runtime_seconds": sum(float(row.get("runtime_seconds", 0.0)) for row in result.diagnostics),
    }
    write_json(output / "preprocessing_result.json", preprocessing)
    return preprocessing


def _read_preprocessed(preprocessed: str | Path, public_folder: Path, q: torch.Tensor, s: torch.Tensor,
                       *, expected_ap_iterations: int = SCIENTIFIC_AP_ITERATIONS):
    folder = _resolve(preprocessed)
    registration = json.loads((folder / "registered_preprocess.json").read_text(encoding="utf-8"))
    preprocessing = json.loads((folder / "preprocessing_result.json").read_text(encoding="utf-8"))
    _validate_preprocessed_contract(registration, preprocessing,
                                    expected_ap_iterations=expected_ap_iterations, q=q, s=s)
    public_sha = digest(public_folder / "manifest.json")
    if registration.get("public_manifest_sha256") != public_sha or preprocessing.get("public_manifest_sha256") != public_sha:
        raise ValueError("preprocessing artifact belongs to another public observation")
    v_path = folder / "V.pt"
    if core.file_sha256(v_path) != preprocessing.get("final_v_file_sha256"):
        raise ValueError("preprocessed V file hash mismatch")
    value = torch.load(v_path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or tuple(value.shape) != tuple(q.shape):
        raise ValueError("preprocessed V has the wrong CPU FP32 shape")
    if core.tensor_sha256(value) != preprocessing.get("final_v_tensor_sha256"):
        raise ValueError("preprocessed V tensor hash mismatch")
    fit = core.fixed_cell_metrics(value, q, s)
    if not fit["exact_code_match"] or not fit["bounded_cell_satisfied"]:
        raise ValueError("preprocessed V is not fixed-code/cell feasible")
    if registration["v_file_sha256"] != preprocessing["final_v_file_sha256"]:
        raise ValueError("preprocessing V file hash is not cross-consistent")
    if registration["v_tensor_sha256"] != preprocessing["final_v_tensor_sha256"]:
        raise ValueError("preprocessing V tensor hash is not cross-consistent")
    return folder, registration, preprocessing, value, fit


def _arm_registration(*, output: Path, public_folder: Path, prep_folder: Path, prep_result: dict,
                       value: torch.Tensor, input_sha: str, source_hashes: dict[str, str],
                       expected_ap_iterations: int) -> dict[str, Any]:
    v_path = (prep_folder / "V.pt").resolve()
    preprocessing_path = (prep_folder / "preprocessing_result.json").resolve()
    input_info = {
        "arm": "INT4_cell_rank",
        "public_manifest_sha256": digest(public_folder / "manifest.json"),
        "preprocessed_result_sha256": core.file_sha256(prep_folder / "preprocessing_result.json"),
        "preprocessed_v_sha256": prep_result["final_v_file_sha256"],
        "preprocessed_v_path": str(v_path),
        "preprocessed_v_file_sha256": core.file_sha256(v_path),
        "preprocessed_v_tensor_sha256": core.tensor_sha256(value),
        "preprocessed_result_path": str(preprocessing_path),
        "private_inputs_used": False,
    }
    return {
        "input": input_info,
        "input_tensor_sha256": input_sha,
        "input_conversion": "raw_signed_dequantized_V_final_cell_projected_from_public_q_s",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "seed": SEED,
        "cpa": deepcopy(CPA_CONFIG),
        "fia": deepcopy(FIA_CONFIG),
        "source_adapter": "abs(S_hat); estimated ordering; no oracle correction",
        "private_inputs_used": False,
        "labels_used": False,
        "preprocess": {
            "format": FORMAT,
            "schema_version": 1,
            "rank": prep_result["rank"],
            "ap_iterations": prep_result["ap_iterations"],
            "expected_ap_iterations": expected_ap_iterations,
            "attack_layer": ATTACK_LAYER,
            "centered": False,
            "rank_backend": core.RANK_BACKEND,
            "projection_dtype": prep_result["projection_dtype"],
            "projection_device": prep_result["projection_device"],
            "final_selection": prep_result["final_selection"],
            "final_v_file_sha256": prep_result["final_v_file_sha256"],
            "final_v_tensor_sha256": prep_result["final_v_tensor_sha256"],
            "scale_is_metric_only": True,
        },
        "attack_status": "NOT_RUN",
        "evaluation_status": "NOT_RUN",
        "source_hashes": source_hashes,
        "arm_output": str(output),
    }


def _write_failed_attack(*, output: Path, registration: dict[str, Any], input_sha: str,
                         source_hashes: dict[str, str], cpa_status: str,
                         fia_status: str, error: Exception) -> None:
    """Persist a truthful stage result before surfacing an attack exception."""
    write_json(output / "result.json", {
        "status": "FAILED",
        "arm": "INT4_cell_rank",
        "input": registration["input"],
        "input_tensor_sha256": input_sha,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "seed": SEED,
        "preprocess_status": "COMPLETED",
        "preprocess": registration.get("preprocess"),
        "attack_status": "FAILED",
        "evaluation_status": "NOT_RUN",
        "cpa_status": cpa_status,
        "fia_status": fia_status,
        "full_experiment_status": "NOT_RUN",
        "private_inputs_used": False,
        "labels_used": False,
        "source_hashes": source_hashes,
        "error_type": type(error).__name__,
        "error": str(error),
    })


def attack(*, public_observation: str | Path, preprocessed: str | Path,
           checkpoint: str | Path, output: str | Path,
           expected_ap_iterations: int = SCIENTIFIC_AP_ITERATIONS):
    """Future public-only CPA/FIA arm using final cell-projected ``V``."""
    output = _new_output(output, public_observation, preprocessed)
    public_folder, manifest, _entries, q, s = _public_observation(public_observation)
    prep_folder, _prep_registration, prep_result, value, prep_fit = _read_preprocessed(
        preprocessed, public_folder, q, s, expected_ap_iterations=expected_ap_iterations)
    checkpoint = _resolve(checkpoint)
    verify_checkpoint(checkpoint)
    if digest(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("checkpoint digest mismatch")
    configure()
    output.mkdir(parents=True)
    (output / "cpa").mkdir()
    (output / "fia").mkdir()
    source_before = upstream_naive.source_hashes()
    # The CPA manifest uses upstream_naive's canonical NumPy-dtype hash. Keep
    # preprocessing tensor/file hashes separate from this execution fingerprint.
    input_sha = upstream_naive._tensor_hash(value)
    registration = _arm_registration(
        output=output, public_folder=public_folder, prep_folder=prep_folder,
        prep_result=prep_result, value=value, input_sha=input_sha,
        source_hashes=source_before, expected_ap_iterations=expected_ap_iterations)
    write_json(output / "registered_arm.json", registration)
    started = time.perf_counter()
    cpa_status = "NOT_RUN"
    fia_status = "NOT_RUN"
    # telemetry is the existing observation-only wrapper; it does not replace
    # either original algorithm and is easy to replace with a null context in
    # a CPU/mock integration test.
    cpa_status = "RUNNING"
    try:
        with telemetry(output / "cpa", "cpa"):
            cpa = upstream_naive.run_upstream_cpa(
                Y_hat=value, n_components=SCIENTIFIC_RANK, device="cuda",
                config=deepcopy(CPA_CONFIG), output_dir=output / "cpa", seed=SEED)
        sources = cpa.pop("S_hat")
        if not isinstance(sources, torch.Tensor) or tuple(sources.shape) != (SCIENTIFIC_RANK, value.shape[1]):
            raise ValueError("original CPA returned an invalid source matrix")
        if not bool(torch.isfinite(sources).all()):
            raise ValueError("original CPA returned nonfinite sources")
        target = sources.abs().contiguous()
        torch.save({"S_hat": sources, "target_z": target}, output / "sources.pt")
        write_json(output / "cpa_result.json", cpa)
        cpa_runtime = runtime()
        cpa_status = "COMPLETED"
    except Exception as error:
        cpa_status = "FAILED"
        _write_failed_attack(output=output, registration=registration, input_sha=input_sha,
                             source_hashes=source_before, cpa_status=cpa_status,
                             fia_status=fia_status, error=error)
        raise
    fia_status = "RUNNING"
    try:
        with telemetry(output / "fia", "fia"):
            fia = upstream_naive.run_upstream_fia(
                target_z=target, device="cuda", config=deepcopy(FIA_CONFIG),
                output_dir=output / "fia", seed=SEED)
        pixels = fia.pop("rec")
        if (not isinstance(pixels, torch.Tensor) or tuple(pixels.shape) != (SCIENTIFIC_RANK, 3, 224, 224)
                or not bool(torch.isfinite(pixels).all()) or bool(((pixels < 0) | (pixels > 1)).any())):
            raise ValueError("original Direct FIA returned invalid pixels")
        torch.save({"pixels": pixels}, output / "candidate.pt")
        write_json(output / "fia_result.json", fia)
        fia_status = "COMPLETED"
    except Exception as error:
        fia_status = "FAILED"
        _write_failed_attack(output=output, registration=registration, input_sha=input_sha,
                             source_hashes=source_before, cpa_status=cpa_status,
                             fia_status=fia_status, error=error)
        raise
    try:
        cpa_log = json.loads((output / "cpa_result.json").read_text(encoding="utf-8")).get("log", [])
        raw_metrics = source_metrics(sources, cpa_log) if cpa_log else new_trial_metrics()
        raw_metrics["raw_metrics"]["quantization_consistency"] = metric_record(
            prep_fit, status="COMPUTED", definition="Final cell-projected V at fixed observed signed q,s")
        raw_metrics["raw_metrics"]["scale_consistency"] = metric_record(
            prep_fit["scale_consistency"], status="COMPUTED", definition="Candidate V scale versus observed per-tensor s; metric only")
        write_json(output / "raw_metrics.json", raw_metrics)
        source_after = upstream_naive.source_hashes()
        if source_after != source_before:
            raise RuntimeError("upstream source changed during cell-rank arm")
    except Exception as error:
        _write_failed_attack(output=output, registration=registration, input_sha=input_sha,
                             source_hashes=source_before, cpa_status=cpa_status,
                             fia_status=fia_status, error=error)
        raise
    result = {
        "status": "COMPLETED",
        "arm": "INT4_cell_rank",
        "input": registration["input"],
        "input_tensor_sha256": input_sha,
        "preprocessed_v_sha256": prep_result["final_v_file_sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA,
        "seed": SEED,
        "cpa_iterations": int(CPA_CONFIG["cpa_ica_max_iter"]),
        "fia_iterations": int(FIA_CONFIG["n_iter"]),
        "image_count": SCIENTIFIC_RANK,
        "selection": "final_iteration_only_no_oracle_selection",
        "preprocess_status": "COMPLETED",
        "preprocess": registration["preprocess"],
        "attack_status": "COMPLETED",
        "evaluation_status": "NOT_RUN",
        "cpa_status": cpa_status,
        "fia_status": fia_status,
        "labels_used": False,
        "private_inputs_used": False,
        "source_adapter": "abs(S_hat); estimated ordering; no oracle correction",
        "candidate_sha256": core.file_sha256(output / "candidate.pt"),
        "sources_sha256": core.file_sha256(output / "sources.pt"),
        "source_hashes": source_after,
        "cpa_runtime": cpa_runtime,
        "final_runtime": runtime(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
        "wall_seconds": time.perf_counter() - started,
        "full_experiment_status": "NOT_RUN",
    }
    write_json(output / "result.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("preprocess")
    prep.add_argument("--public-observation", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--ap-iterations", type=int, default=SCIENTIFIC_AP_ITERATIONS)
    prep.add_argument("--projection-device", choices=("cpu", "cuda"), default="cuda")
    prep.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    arm = sub.add_parser("attack")
    arm.add_argument("--public-observation", type=Path, required=True)
    arm.add_argument("--preprocessed", type=Path, required=True)
    arm.add_argument("--checkpoint", type=Path, required=True)
    arm.add_argument("--output", type=Path, required=True)
    arm.add_argument("--expected-ap-iterations", type=int, choices=VALID_AP_ITERATIONS,
                     default=SCIENTIFIC_AP_ITERATIONS,
                     help="Declared preprocessing contract: AP5 by default, or explicit AP0 identity control")
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "preprocess":
        return preprocess(public_observation=args.public_observation, output=args.output,
                          ap_iterations=args.ap_iterations, projection_device=args.projection_device,
                          chunk_rows=args.chunk_rows)
    return attack(public_observation=args.public_observation, preprocessed=args.preprocessed,
                  checkpoint=args.checkpoint, output=args.output,
                  expected_ap_iterations=args.expected_ap_iterations)


if __name__ == "__main__":
    main()
