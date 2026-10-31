"""Strict public INT8 randomized-dequantization orchestration.

The sender, separation, each Direct FIA arm, and final evaluator are separate
processes in the task wrapper.  This runner itself accepts only public INT8
packets after the sender phase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from .deterministic_hybrid_setup import configure_runtime
from .signed_combination_ica import CombinationICAConfig, separate_observation, tensor_sha256, tensor_stats
from .signed_int8_observation import CHECKPOINT_SHA, FC_SHAPES, read_bundle
from .signed_int8_randomized import align_and_average, sample_matrix
from .upstream_naive import run_upstream_fia, source_hashes as upstream_source_hashes


TASK_ID = "CPA_SIGNED_INT8_RANDOMIZED8_FIA_20260917"
PLAN_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth"
PUBLIC_ROOT = REPO_ROOT / "quantization_aware_cpa/results/signed_int8_randomized8_fia_B16_T10_20260917"
PRIVATE_ROOT = REPO_ROOT / "quantization_aware_cpa/results_private/signed_int8_randomized8_fia_B16_T10_20260917"
SOURCE_COUNT = 16
FEATURE_DIM = 25088
OBSERVATION_ROWS = 4096
MEMBER_COUNT = 8
MEMBER_SEEDS = [1234 + 1000003 * index for index in range(MEMBER_COUNT)]
SEPARATION_CONFIG = CombinationICAConfig(refinement_iterations=4000)
FIA_CONFIG = {"n_iter": 25000, "lr": 0.1, "tv": 0.1, "fi": 1.0,
              "optimizer": "adam", "scheduler": "cosine", "log_every": 5000}
MAX_SECONDS = 90 * 60
MAX_CUDA_ALLOCATED_BYTES = 12 * 1024**3


class RunnerError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")
    temp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:
        raise RunnerError(f"invalid UTF-8 JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise RunnerError(f"expected JSON object: {path}")
    return value


def _must_file(path: Path) -> None:
    if not Path(path).is_file() or Path(path).is_symlink():
        raise RunnerError(f"missing regular file: {path}")


def _runtime(device: str, phase: str) -> dict[str, Any]:
    return {"phase": phase, "device": device, "torch": torch.__version__,
            "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "dtype": "float32", "threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def _code_hashes() -> dict[str, str]:
    paths = [Path(__file__), Path(__file__).with_name("signed_int8_randomized.py"),
             Path(__file__).with_name("signed_int8_observation.py"),
             Path(__file__).with_name("signed_combination_ica.py"),
             Path(__file__).with_name("upstream_naive.py")]
    return {path.resolve().as_posix(): file_sha256(path) for path in paths}


def _protected_hashes() -> dict[str, str]:
    paths = [REPO_ROOT / "quantization_aware_cpa/qacpa/signed_combination_ica.py",
             REPO_ROOT / "quantization_aware_cpa/qacpa/upstream_naive.py",
             REPO_ROOT / "quantization_aware_cpa/qacpa/deterministic_hybrid_setup.py",
             REPO_ROOT / "quantized_update_cpa/cell_utils.py",
             REPO_ROOT / "official_cpa/src/gradient_inversion.py",
             REPO_ROOT / "official_cpa/src/feature_inversion.py",
             REPO_ROOT / "official_cpa/src/vgg16.py"]
    return {path.resolve().as_posix(): file_sha256(path) for path in paths}


def public_input_hashes(observation: Path | None = None) -> dict[str, str]:
    observation = (observation or PUBLIC_ROOT / "observation").resolve()
    manifest = observation / "manifest.json"
    _must_file(manifest)
    value = _read_json(manifest)
    paths = [manifest] + [observation / item["file"] for item in value["layers"].values()]
    paths.append(CHECKPOINT)
    return {path.resolve().as_posix(): file_sha256(path) for path in paths}


def _tensor_hash(value: torch.Tensor) -> str:
    return tensor_sha256(value.detach().cpu().contiguous())


def _validate_sources(source_root: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    result = _read_json(source_root / "result.json")
    if source_root.name == "center":
        if result.get("index") != "center" or result.get("separation_call") != 1 or result.get("private_data_used") is not False:
            raise RunnerError("center separation result is not authenticated")
    elif (result.get("task_id") != TASK_ID or result.get("status") != "COMPLETED"
          or result.get("stage") != "SEPARATE" or result.get("separation_calls") != 9):
        raise RunnerError("aggregate separation result is not an authenticated nine-call result")
    payload = torch.load(source_root / "sources.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"S_hat", "target_z"}:
        raise RunnerError("aggregate sources payload schema mismatch")
    S_hat, target_z = payload["S_hat"], payload["target_z"]
    expected = (SOURCE_COUNT, FEATURE_DIM)
    for name, value in (("S_hat", S_hat), ("target_z", target_z)):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or tuple(value.shape) != expected or not bool(torch.isfinite(value).all())):
            raise RunnerError(f"invalid aggregate {name}")
    if bool((target_z < 0).any()) or not torch.equal(target_z, S_hat.abs()):
        raise RunnerError("aggregate target is not exact abs(S_hat)")
    if result.get("S_hat_sha256") != _tensor_hash(S_hat) or result.get("target_z_sha256") != _tensor_hash(target_z):
        raise RunnerError("aggregate tensor hash binding changed")
    return S_hat.contiguous(), target_z.contiguous(), result


def register(public_root: Path, private_root: Path) -> dict[str, Any]:
    public_root, private_root = Path(public_root).resolve(), Path(private_root).resolve()
    if public_root.exists() or private_root.exists() or public_root in private_root.parents or private_root in public_root.parents:
        raise FileExistsError("INT8 task output roots must be new and disjoint")
    if file_sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RunnerError("checkpoint hash mismatch")
    public_root.mkdir(parents=True, exist_ok=False)
    private_root.mkdir(parents=True, exist_ok=False)
    registration = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "REGISTERED",
        "method": "INT8_randomized_dequantization_M8_FastICA_GD4000_DirectFIA25000",
        "stage_order": ["SENDER", "SEPARATE", "INVERT_CENTER", "INVERT_RANDOMIZED8", "PUBLIC_GATE", "EVALUATE"],
        "public_output_root": public_root.as_posix(), "public_observation": (public_root / "observation").as_posix(),
        "separator_config": SEPARATION_CONFIG.as_dict(), "member_count": MEMBER_COUNT,
        "member_seeds": MEMBER_SEEDS, "fia_config": FIA_CONFIG,
        "codec": "signed_symmetric_int8_qplus127_nearest_even_cpu_fp32_v1",
        "source_count": SOURCE_COUNT, "source_shape": [SOURCE_COUNT, FEATURE_DIM],
        "checkpoint": CHECKPOINT.as_posix(), "checkpoint_sha256": CHECKPOINT_SHA,
        "public_input_hashes": {}, "code_hashes": _code_hashes(), "protected_source_hashes": _protected_hashes(),
        "private_data_policy": {"attack_reads_public_observation_only": True, "load_after_public_gate": True,
                                 "sender_true_updates_consumed": True, "attack_true_updates_consumed": False,
                                 "evaluator_consumed_field": "x only", "labels_consumed": False},
        "actual_image_cell_feasibility": "NOT_EVALUATED", "scientific_quality": "NOT_EVALUATED",
        "budget_seconds": MAX_SECONDS, "cuda_allocated_cap_bytes": MAX_CUDA_ALLOCATED_BYTES,
        "registered_before_execution": True,
    }
    write_json(public_root / "registration.json", registration)
    write_json(private_root / "task_boundary.json", {"task_id": TASK_ID, "true_updates_consumed_by_sender": True,
                                                       "labels_consumed": False, "images_consumed": False})
    return registration


def separate(public_observation: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing preexisting separation output: {output}")
    configure_runtime(42, threads=1)
    if file_sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RunnerError("checkpoint hash mismatch")
    manifest, entries = read_bundle(public_observation)
    q, scale = entries["classifier.0.weight"]
    X0 = (q.to(torch.float32) * scale).contiguous()
    if tuple(X0.shape) != (OBSERVATION_ROWS, FEATURE_DIM):
        raise RunnerError("public FC0 matrix has wrong shape")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "registration.json", {
        "task_id": TASK_ID, "status": "REGISTERED", "stage": "SEPARATE",
        "public_manifest_sha256": file_sha256(Path(public_observation) / "manifest.json"),
        "config": SEPARATION_CONFIG.as_dict(), "separation_calls_registered": 9,
        "member_count": MEMBER_COUNT, "member_seeds": MEMBER_SEEDS, "private_data_used": False,
        "runtime": _runtime("cpu", "SEPARATE_PRE_ICA"), "protected_source_hashes": _protected_hashes(),
    })
    records: list[dict[str, Any]] = []
    sources: list[torch.Tensor] = []
    center_dir = output / "center"
    center_dir.mkdir()
    center = separate_observation(X0, SOURCE_COUNT, config=SEPARATION_CONFIG, device="cpu")
    center_source = center.S_hat.detach().cpu().contiguous()
    torch.save({"S_hat": center_source, "target_z": center_source.abs().contiguous()}, center_dir / "sources.pt")
    write_json(center_dir / "refinement_log.json", center.refinement.log)
    write_json(center_dir / "result.json", {"index": "center", "separation_call": 1,
        "S_hat_sha256": _tensor_hash(center_source), "target_z_sha256": _tensor_hash(center_source.abs()),
        "fastica": center.metadata["fastica"], "refinement": center.metadata["refinement"],
        "whitening": center.metadata["whitening"], "final_losses": center.refinement.final_losses,
        "private_data_used": False})
    sources.append(center_source)
    records.append({"index": "center", "separation_call": 1, "input_sha256": _tensor_hash(X0),
                    "S_hat_sha256": _tensor_hash(center_source), "target_z_sha256": _tensor_hash(center_source.abs()),
                    "fastica": center.metadata["fastica"], "timing_seconds": time.perf_counter() - started})
    del center
    for member_index, seed in enumerate(MEMBER_SEEDS):
        sampled, sample_meta = sample_matrix(q, scale, seed=seed, chunk_rows=32)
        member_dir = output / f"member_{member_index:02d}"
        member_dir.mkdir()
        separation = separate_observation(sampled, SOURCE_COUNT, config=SEPARATION_CONFIG, device="cpu")
        source = separation.S_hat.detach().cpu().contiguous()
        target = source.abs().contiguous()
        torch.save({"S_hat": source, "target_z": target}, member_dir / "sources.pt")
        write_json(member_dir / "refinement_log.json", separation.refinement.log)
        write_json(member_dir / "sample.json", sample_meta)
        write_json(member_dir / "result.json", {"index": member_index, "separation_call": member_index + 2,
            "input_sha256": sample_meta["tensor_sha256"], "S_hat_sha256": _tensor_hash(source),
            "target_z_sha256": _tensor_hash(target), "fastica": separation.metadata["fastica"],
            "refinement": separation.metadata["refinement"], "whitening": separation.metadata["whitening"],
            "final_losses": separation.refinement.final_losses, "private_data_used": False})
        sources.append(source)
        records.append({"index": member_index, "separation_call": member_index + 2,
                        "input_sha256": sample_meta["tensor_sha256"], "sample": sample_meta,
                        "S_hat_sha256": _tensor_hash(source), "target_z_sha256": _tensor_hash(target),
                        "fastica": separation.metadata["fastica"],
                        "timing_seconds": time.perf_counter() - started})
        del sampled, separation, source, target
    aggregate, alignment = align_and_average(sources[0], sources[1:])
    aggregate_target = aggregate.abs().contiguous()
    agg_dir = output / "aggregate"
    agg_dir.mkdir()
    torch.save({"S_hat": aggregate, "target_z": aggregate_target}, agg_dir / "sources.pt")
    write_json(agg_dir / "alignment.json", alignment)
    result = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "COMPLETED", "stage": "SEPARATE",
        "public_manifest_sha256": file_sha256(Path(public_observation) / "manifest.json"),
        "config": SEPARATION_CONFIG.as_dict(), "source_shape": [SOURCE_COUNT, FEATURE_DIM],
        "separation_calls": 9, "center_is_reference_not_member": True,
        "member_count": MEMBER_COUNT, "member_seeds": MEMBER_SEEDS, "records": records,
        "alignment": alignment, "S_hat_sha256": _tensor_hash(aggregate),
        "target_z_sha256": _tensor_hash(aggregate_target), "target_z_exact_abs_S_hat": True,
        "S_hat_stats": tensor_stats(aggregate), "target_z_stats": tensor_stats(aggregate_target),
        "aggregate_sources_sha256": file_sha256(agg_dir / "sources.pt"),
        "center_sources_sha256": file_sha256(center_dir / "sources.pt"),
        "runtime": _runtime("cpu", "SEPARATE_POST_ICA"), "protected_source_hashes": _protected_hashes(),
        "upstream_source_hashes": upstream_source_hashes(), "private_data_used": False,
        "labels_consumed": False, "true_updates_consumed": False,
        "actual_image_cell_feasibility": "NOT_EVALUATED", "quality": "NOT_EVALUATED",
        "timing": {"wall_seconds": time.perf_counter() - started},
    }
    write_json(output / "result.json", result)
    write_json(agg_dir / "result.json", result)
    write_json(agg_dir / "registration.json", {
        "task_id": TASK_ID, "status": "COMPLETED", "stage": "SEPARATE",
        "config": SEPARATION_CONFIG.as_dict(), "separation_calls": 9,
        "S_hat_sha256": result["S_hat_sha256"], "target_z_sha256": result["target_z_sha256"],
        "sources_sha256": file_sha256(agg_dir / "sources.pt"), "private_data_used": False,
    })
    write_json(output / "registration.json", {**_read_json(output / "registration.json"), "status": "COMPLETED",
                                                "result_sha256": file_sha256(output / "result.json")})
    return result


def _initial_state_hash(device: str) -> str:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    init = torch.randn((SOURCE_COUNT, 3, 224, 224), device=device, dtype=torch.float32)
    return _tensor_hash(init)


def invert(source_root: Path, output: Path, arm: str) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing preexisting FIA output: {output}")
    configure_runtime(42, threads=1)
    if not torch.cuda.is_available():
        raise RunnerError("registered Direct FIA requires CUDA")
    S_hat, target_z, source_result = _validate_sources(Path(source_root).resolve())
    output.mkdir(parents=True, exist_ok=False)
    initial_hash = _initial_state_hash("cuda")
    registration = {"task_id": TASK_ID, "status": "REGISTERED", "stage": "INVERT", "arm": arm,
                    "source_root": Path(source_root).resolve().as_posix(),
                    "source_artifact_sha256": file_sha256(Path(source_root) / "sources.pt"),
                    "target_z_sha256": _tensor_hash(target_z), "fia_config": FIA_CONFIG,
                    "checkpoint_sha256": CHECKPOINT_SHA, "initial_image_state_sha256": initial_hash,
                    "runtime_pre_adapter": _runtime("cuda", "INVERT_PRE_DIRECT_FIA"),
                    "protected_source_hashes": _protected_hashes(), "upstream_source_hashes": upstream_source_hashes(),
                    "private_data_used": False, "labels_consumed": False, "true_updates_consumed": False,
                    "quality": "NOT_EVALUATED"}
    write_json(output / "registration.json", registration)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    fia = run_upstream_fia(target_z=target_z, device="cuda", config=dict(FIA_CONFIG),
                           output_dir=output / "upstream_fia", seed=42)
    rec = fia["rec"].detach().cpu().contiguous()
    if (rec.dtype != torch.float32 or tuple(rec.shape) != (SOURCE_COUNT, 3, 224, 224)
            or not bool(torch.isfinite(rec).all()) or bool((rec < 0).any()) or bool((rec > 1).any())):
        raise RunnerError("Direct FIA returned invalid candidate")
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated())
    if peak > MAX_CUDA_ALLOCATED_BYTES:
        raise RunnerError(f"CUDA allocated cap exceeded: {peak}")
    torch.save({"pixels": rec}, output / "candidate.pt")
    result = {"task_id": TASK_ID, "status": "COMPLETED", "stage": "INVERT", "arm": arm,
              "source_artifact_sha256": registration["source_artifact_sha256"], "target_z_sha256": _tensor_hash(target_z),
              "fia_config": FIA_CONFIG, "candidate_sha256": file_sha256(output / "candidate.pt"),
              "initial_image_state_sha256": initial_hash, "direct_fia": {k: v for k, v in fia.items() if k != "rec"},
              "runtime_post_adapter": _runtime("cuda", "INVERT_POST_DIRECT_FIA"),
              "resource": {"cuda_max_memory_allocated_bytes": peak, "cuda_allocated_cap_bytes": MAX_CUDA_ALLOCATED_BYTES},
              "protected_source_hashes": _protected_hashes(), "private_data_used": False,
              "labels_consumed": False, "true_updates_consumed": False, "quality": "NOT_EVALUATED",
              "timing": {"wall_seconds": time.perf_counter() - started}}
    write_json(output / "direct_fia_log.json", fia.get("log", []))
    write_json(output / "result.json", result)
    write_json(output / "registration.json", {**registration, "status": "COMPLETED",
                                               "candidate_sha256": result["candidate_sha256"],
                                               "resource": result["resource"]})
    return result


def finalize_public(public_root: Path) -> dict[str, Any]:
    public_root = Path(public_root).resolve()
    registration = _read_json(public_root / "registration.json")
    observation = public_root / "observation"
    separation = public_root / "separation"
    center = public_root / "fia_center"
    randomized = public_root / "fia_randomized8"
    for path in (observation / "manifest.json", separation / "result.json", center / "result.json",
                 randomized / "result.json", center / "candidate.pt", randomized / "candidate.pt"):
        _must_file(path)
    inputs = public_input_hashes(observation)
    if registration.get("code_hashes") != _code_hashes() or registration.get("protected_source_hashes") != _protected_hashes():
        raise RunnerError("registered source hashes changed")
    registration.update({"status": "PUBLIC_COMPLETED", "public_input_hashes": inputs,
                         "public_gate_ready": True, "public_completion_time": time.time(),
                         "protected_source_hashes_after": _protected_hashes(),
                         "private_data_policy": {**registration["private_data_policy"], "public_gate_passed": True}})
    write_json(public_root / "registration.json", registration)
    return registration


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("register"); p.add_argument("--public-root", type=Path, required=True); p.add_argument("--private-root", type=Path, required=True)
    p = sub.add_parser("separate"); p.add_argument("--public-observation", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("invert"); p.add_argument("--source-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--arm", required=True)
    p = sub.add_parser("finalize-public"); p.add_argument("--public-root", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "register": result = register(args.public_root, args.private_root)
        elif args.command == "separate": result = separate(args.public_observation, args.output)
        elif args.command == "invert": result = invert(args.source_root, args.output, args.arm)
        else: result = finalize_public(args.public_root)
    except Exception as error:
        print(f"{TASK_ID}: {type(error).__name__}: {error}", flush=True)
        return 1
    print(json.dumps({"task_id": TASK_ID, "status": result.get("status"), "stage": result.get("stage", "REGISTER"),
                      "quality": result.get("quality", "NOT_EVALUATED")}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
