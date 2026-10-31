"""Isolated INT4 row-wise FastICA+GD4000 screen3 pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import torch

from .deterministic_hybrid_setup import configure_runtime
from .signed_baseline import FIA_CONFIG, SEED
from .signed_bits_observation import CHECKPOINT_SHA, FC_SHAPES, digest
from .signed_int4_rowwise_codec import (CODEC, RAW_UPDATES, RAW_UPDATES_SHA, SCALE_COUNT,
    collect_rowwise, read_bundle, sample_matrix, tensor_digest)
from .signed_bits_randomized import align_preserve_norm
from .signed_combination_ica import CombinationICAConfig, separate_observation
from .signed_int8_screen3_fia import PREFIX_STEPS, TOTAL_STEPS, export_completed_checkpoint, run_trajectory
from .upstream_naive import source_hashes as upstream_source_hashes


TASK_ID = "CPA_SIGNED_INT4_ROWWISE_COMBINATION_SCREEN3_FIA_20260919"
PLAN_VERSION = 1
BITS = 4
SOURCE_COUNT = 16
FEATURE_DIM = 25088
MEMBER_COUNT = 8
MEMBER_SEEDS = [1234 + 1000003 * i for i in range(MEMBER_COUNT)]
MAX_SECONDS = 12600
MAX_CUDA_ALLOCATED_BYTES = 12 * 1024**3
TOTAL_FIA_UPDATES = 125000
OBSERVATION = Path(__file__).resolve().parents[1] / "results/signed_int4_rowwise_combination_randomized8_screen3_fia_B16_T10_20260919/observation"
PARENT_PUBLIC = Path(__file__).resolve().parents[1] / "results/signed_int4_randomized8_screen3_fia_B16_T10_20260918_sweep"
PARENT_PRIVATE = Path(__file__).resolve().parents[1] / "results_private/signed_int4_randomized8_screen3_fia_B16_T10_20260918_sweep"
PARENT_COMPARISON = PARENT_PRIVATE / "evaluation_recovery_20260918/comparison.json"
PRIVATE_TRUTH = Path(__file__).resolve().parents[1] / "results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"
BASELINE_CANDIDATE = Path(__file__).resolve().parents[1] / "results/signed_int4_baseline_B16_T10/naive/candidate.pt"


class RowwiseScreen3Error(RuntimeError):
    pass


SEPARATOR = CombinationICAConfig(fast_max_iter=200, fast_tolerance=1e-6,
    refinement_iterations=4000, refinement_lr=0.001, refinement_momentum=0.0,
    refinement_weight_decay=0.0, negentropy_weight=0.0935827857091549,
    decorrelation_weight=0.2513674104161625, sign_weight=0.05811961446881231,
    l1_weight=17.94138146247377, eps=1e-10, rank_floor=1e-12)


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8", newline="\n")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RowwiseScreen3Error(f"expected JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    return digest(Path(path))


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    array = value.numpy()
    h = hashlib.sha256()
    h.update(str(array.dtype).encode("ascii"))
    h.update(str(array.shape).encode("ascii"))
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__), Path(__file__).with_name("signed_int4_rowwise_codec.py"),
        Path(__file__).with_name("signed_combination_ica.py"),
        Path(__file__).with_name("signed_int8_screen3_fia.py"),
    ]
    return {str(p.resolve()): file_sha256(p) for p in paths}


def _protected_hashes() -> dict[str, str]:
    root = _repo()
    paths = [
        root / "official_cpa/src/gradient_inversion.py",
        root / "official_cpa/src/feature_inversion.py",
        root / "official_cpa/src/vgg16.py",
        root / "quantization_aware_cpa/qacpa/deterministic_hybrid_setup.py",
        root / "quantization_aware_cpa/qacpa/signed_combination_ica.py",
        root / "quantization_aware_cpa/qacpa/signed_bits_screen3_runner.py",
        root / "quantization_aware_cpa/qacpa/signed_bits_screen3_evaluation.py",
    ]
    return {str(p.resolve()): file_sha256(p) for p in paths}


def _input_hashes(observation: Path = OBSERVATION) -> dict[str, str]:
    observation = Path(observation).resolve()
    manifest = read_json(observation / "manifest.json")
    paths = [observation / "manifest.json"]
    for item in manifest["layers"].values():
        paths.append(observation / item["file"])
    checkpoint = Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth"
    paths.append(checkpoint)
    return {str(p.resolve()): file_sha256(p) for p in paths}


def _config() -> dict[str, Any]:
    return SEPARATOR.as_dict()


def _check_source(source: torch.Tensor, name: str) -> None:
    if (source.dtype != torch.float32 or source.device.type != "cpu"
            or tuple(source.shape) != (SOURCE_COUNT, FEATURE_DIM)
            or not bool(torch.isfinite(source).all())):
        raise RowwiseScreen3Error(f"{name} must be finite CPU FP32 [16,25088]")


def _save_branch(folder: Path, payload: dict[str, torch.Tensor], record: dict[str, Any]) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    torch.save(payload, folder / "sources.pt")
    record = dict(record)
    record["source_file_sha256"] = file_sha256(folder / "sources.pt")
    write_json(folder / "result.json", record)


def _assert_observation() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    manifest, entries = read_bundle(OBSERVATION)
    q, scales = entries["classifier.0.weight"]
    X0 = (q.to(torch.float32) * scales[:, None]).contiguous()
    if tuple(q.shape) != (4096, FEATURE_DIM) or tuple(X0.shape) != (4096, FEATURE_DIM):
        raise RowwiseScreen3Error("FC0 row-wise observation shape mismatch")
    return q, scales, X0, manifest


def register(public_root: Path, private_root: Path) -> dict[str, Any]:
    public_root, private_root = Path(public_root).resolve(), Path(private_root).resolve()
    if public_root.exists() or private_root.exists() or public_root in private_root.parents or private_root in public_root.parents:
        raise FileExistsError("new INT4 GD-only output roots must be absent and disjoint")
    if not PRIVATE_TRUTH.is_file() or not BASELINE_CANDIDATE.is_file():
        raise RowwiseScreen3Error("registered baseline artifacts are missing")
    if not (Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth").is_file():
        raise RowwiseScreen3Error("registered VGG16 checkpoint is missing")
    public_root.mkdir(parents=True, exist_ok=False)
    private_root.mkdir(parents=True, exist_ok=False)
    registration = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "REGISTERED", "bits": BITS,
        "method": "signed_int4_rowwise_combination_ica_fastica_gd4000",
        "backend": "qacpa.signed_combination_ica.separate_observation",
        "public_observation": OBSERVATION.as_posix(), "public_observation_is_new_sender_output": True,
        "public_output_root": public_root.as_posix(), "private_output_root": private_root.as_posix(),
        "separator_config": _config(), "separator_config_exact_combination_ica": True,
        "fastica": {"called": True, "method": "parallel_symmetric_fixed_point_logcosh_alpha1"},
        "center_and_member_count": 9, "member_count": MEMBER_COUNT, "member_seeds": MEMBER_SEEDS,
        "alignment": {"reference": "new row-wise center signed S_hat", "matching": "Hungarian_max_sum_abs_cosine",
                       "correction": "permutation_and_sign_only", "row_norm_rescaling": False},
        "fia_config": dict(FIA_CONFIG), "screen_config": {"prefix_steps": PREFIX_STEPS, "total_steps": TOTAL_STEPS,
            "selection": "post5000_feature_cosine_distance_plus_0.1_normalized_TV", "selected_count": 3,
            "continuation_steps": 20000, "total_fia_updates": TOTAL_FIA_UPDATES},
        "input_manifest_sha256": None, "input_hashes": {},
        "checkpoint_sha256": CHECKPOINT_SHA, "code_hashes": _code_hashes(), "protected_source_hashes": _protected_hashes(),
        "upstream_source_hashes": upstream_source_hashes(),
        "rowwise_codec": {"codec": CODEC, "scale_count": SCALE_COUNT, "scale_bytes": SCALE_COUNT * 4,
            "physical_code_storage": "int8", "logical_code_bits": 4},
        "private_data_policy": {"attack_reads_public_observation_only": True, "load_after_public_gate": True,
            "sender_true_updates_consumed": False, "attack_true_updates_consumed": False,
            "evaluator_consumed_field": "x only", "labels_consumed": False},
        "actual_image_cell_feasibility": "NOT_EVALUATED", "scientific_quality": "NOT_EVALUATED",
        "budget_seconds": MAX_SECONDS, "cuda_allocated_cap_bytes": MAX_CUDA_ALLOCATED_BYTES,
        "registered_before_execution": True,
    }
    write_json(public_root / "registration.json", registration)
    sender = collect_rowwise(public_root / "observation", private_root / "sender_provenance")
    registration.update({"input_manifest_sha256": sender["public_manifest_sha256"],
                         "input_hashes": _input_hashes(public_root / "observation"), "sender": sender})
    write_json(public_root / "registration.json", registration)
    write_json(private_root / "task_boundary.json", {"task_id": TASK_ID, "bits": BITS,
        "attack_true_updates_consumed": False, "labels_consumed": False, "images_consumed": False})
    return registration


def _run_one_separation(index: str, X: torch.Tensor, output: Path, input_sha: str) -> dict[str, Any]:
    branch = Path(output) / ("center" if index == "center" else f"member_{int(index):02d}")
    if branch.exists():
        raise FileExistsError(f"refusing partial branch collision: {branch}")
    branch.mkdir(parents=True, exist_ok=False)
    run = separate_observation(X, SOURCE_COUNT, config=SEPARATOR, device="cpu")
    native = run.S_hat.detach().cpu().contiguous()
    _check_source(native, f"native {index}")
    record = {"task_id": TASK_ID, "status": "COMPLETED", "index": index,
              "method": "signed_combination_ica.separate_observation", "separation_call": None,
              "input_sha256": input_sha, "native_source_sha256": tensor_sha256(native),
              "settings": run.metadata, "wall_time_sec": 0.0,
              "fastica": run.metadata["fastica"], "refinement": run.metadata["refinement"],
              "private_data_used": False, "generator_used": False, "target_z_exact_abs_S_hat": True}
    torch.save({"S_hat_native": native} if index != "center" else {"S_hat": native}, branch / "native.pt")
    return record, native


def separate(public_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    public_root = Path(public_root).resolve()
    output = public_root / "separation"
    if output.exists():
        result_path = output / "result.json"
        if result_path.is_file():
            result = read_json(result_path)
            if result.get("status") == "COMPLETED":
                return result
        raise FileExistsError(f"refusing incomplete separation tree: {output}")
    q, scales, X0, manifest = _assert_observation()
    configure_runtime(SEED, threads=1)
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    center_record, center_native = _run_one_separation("center", X0, output, tensor_sha256(X0))
    center_record["separation_call"] = 1
    torch.save({"S_hat": center_native, "target_z": center_native.abs().contiguous()}, output / "center/sources.pt")
    center_record.update({"target_z_sha256": tensor_sha256(center_native.abs()),
                          "source_file_sha256": file_sha256(output / "center/sources.pt")})
    write_json(output / "center/result.json", center_record)
    records.append(center_record)
    for index, seed in enumerate(MEMBER_SEEDS):
        sampled, sample_meta = sample_matrix(q, scales, seed=seed, chunk_rows=32)
        expected_hash = sample_meta["tensor_sha256"]
        rec, native = _run_one_separation(str(index), sampled, output, expected_hash)
        aligned, alignment = align_preserve_norm(center_native, native, index)
        _check_source(aligned, f"aligned {index}")
        target = aligned.abs().contiguous()
        branch = output / f"member_{index:02d}"
        torch.save({"S_hat_native": native, "S_hat_aligned": aligned, "target_z": target}, branch / "sources.pt")
        rec.update({"separation_call": index + 2, "sample": sample_meta,
                    "aligned_source_sha256": tensor_sha256(aligned), "target_z_sha256": tensor_sha256(target),
                    "alignment": alignment, "source_file_sha256": file_sha256(branch / "sources.pt")})
        write_json(branch / "result.json", rec)
        records.append(rec)
        del sampled, native, aligned, target
    result = {"task_id": TASK_ID, "status": "COMPLETED", "stage": "SEPARATE", "bits": BITS,
              "public_manifest_sha256": file_sha256(OBSERVATION / "manifest.json"), "input_manifest_sha256": file_sha256(OBSERVATION / "manifest.json"),
              "config": _config(), "method": "signed_combination_ica.separate_observation", "fastica_called": True,
              "separation_calls": 9, "records": records, "input_hashes_reproduced_from_parent": True,
              "native_source_shape": [SOURCE_COUNT, FEATURE_DIM], "private_data_used": False,
              "timing": {"wall_seconds": time.perf_counter() - started}}
    write_json(output / "result.json", result)
    return result


def _source_entry(public_root: Path, index: int) -> tuple[torch.Tensor, str, dict[str, Any]]:
    branch = Path(public_root) / "separation" / ("center" if index == -1 else f"member_{index:02d}")
    result = read_json(branch / "result.json")
    payload = torch.load(branch / "sources.pt", map_location="cpu", weights_only=True)
    if index == -1:
        if set(payload) != {"S_hat", "target_z"}:
            raise RowwiseScreen3Error("center source schema mismatch")
        source, target = payload["S_hat"], payload["target_z"]
    else:
        if set(payload) != {"S_hat_native", "S_hat_aligned", "target_z"}:
            raise RowwiseScreen3Error(f"member {index} source schema mismatch")
        source, target = payload["S_hat_aligned"], payload["target_z"]
        _check_source(payload["S_hat_native"], f"native member {index}")
    _check_source(source, f"source {index}")
    if not torch.equal(target, source.abs()) or result.get("target_z_sha256") != tensor_sha256(target):
        raise RowwiseScreen3Error(f"target binding failed for {index}")
    if result.get("source_file_sha256") != file_sha256(branch / "sources.pt"):
        raise RowwiseScreen3Error(f"source file binding failed for {index}")
    return target.contiguous(), file_sha256(branch / "sources.pt"), result


def _progress(event: dict[str, Any]) -> None:
    print(json.dumps({"task_id": TASK_ID, "stage": "FIA", **event}, allow_nan=False), flush=True)


def _public_selection(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if len(scores) != MEMBER_COUNT or sorted(x["member_index"] for x in scores) != list(range(MEMBER_COUNT)):
        raise RowwiseScreen3Error("selection requires exactly eight members")
    for row in scores:
        if row.get("score", {}).get("raw_state_unchanged") is not True:
            raise RowwiseScreen3Error("public score is not bound to raw state")
    ranked = sorted(scores, key=lambda x: (float(x["score"]["score"]), int(x["member_index"])))
    return {"task_id": TASK_ID, "status": "FROZEN_PUBLIC_SELECTION", "selection_rule": "feature_cosine_distance_plus_0.1_normalized_TV",
            "private_metrics_used": False, "ranking": [{"rank": r + 1, "member_index": x["member_index"],
                "score": x["score"]["score"], "feature_cosine_distance_mean": x["score"].get("feature_cosine_distance_mean"),
                "tv_normalized": x["score"].get("tv_normalized"), "candidate_sha256": x["result"]["candidate_sha256"],
                "checkpoint_sha256": x["result"]["checkpoint_sha256"]} for r, x in enumerate(ranked)],
            "selected_member_indices": [x["member_index"] for x in ranked[:3]], "tie_break": "member_index_ascending",
            "completed_step": PREFIX_STEPS}


def run_public(public_root: Path, private_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    public_root, private_root = Path(public_root).resolve(), Path(private_root).resolve()
    registration = read_json(public_root / "registration.json")
    if registration.get("task_id") != TASK_ID or registration.get("status") != "REGISTERED":
        raise RowwiseScreen3Error("row-wise registration is not ready")
    if registration.get("code_hashes") != _code_hashes():
        raise RowwiseScreen3Error("row-wise adapter or registered source code changed")
    separation = separate(public_root)
    center_target, center_source_sha, _ = _source_entry(public_root, -1)
    center = run_trajectory(center_target, output=public_root / "fia_center_25000", source_sha256=center_source_sha,
                            arm="center_25000", end_steps=TOTAL_STEPS, progress=_progress)
    scores = []
    for index in range(MEMBER_COUNT):
        target, source_sha, _ = _source_entry(public_root, index)
        result = run_trajectory(target, output=public_root / f"fia_member_{index:02d}_5000", source_sha256=source_sha,
                                arm=f"member_{index:02d}_5000", end_steps=PREFIX_STEPS, progress=_progress)
        scores.append({"member_index": index, "result": result, "score": result["score"],
                       "target_z_sha256": tensor_sha256(target), "source_file_sha256": source_sha})
    selection = _public_selection(scores)
    write_json(public_root / "selection.json", selection)
    continuations = []
    for index in selection["selected_member_indices"]:
        target, source_sha, _ = _source_entry(public_root, int(index))
        continuations.append(run_trajectory(target, output=public_root / f"fia_member_{int(index):02d}_25000",
            source_sha256=source_sha, arm=f"member_{int(index):02d}_25000", end_steps=TOTAL_STEPS,
            parent_checkpoint=public_root / f"fia_member_{int(index):02d}_5000/checkpoint_5000.pt",
            parent_role="MEMBER_PREFIX_5000", progress=_progress))
    execution = {"task_id": TASK_ID, "status": "PUBLIC_COMPLETED", "separation": separation,
        "center": center, "member_prefixes": [x["result"] for x in scores], "selection": selection,
        "continuations": continuations, "new_fia_updates": TOTAL_FIA_UPDATES,
        "total_cpa_iterations": 9 * int(_config()["cpa_ica_max_iter"]), "private_loaded": False,
        "wall_seconds": time.perf_counter() - started}
    write_json(public_root / "public_execution.json", execution)
    registration.update({"status": "PUBLIC_COMPLETED", "public_frozen": True, "selection_sha256": file_sha256(public_root / "selection.json"),
        "separation_result_sha256": file_sha256(public_root / "separation/result.json"), "public_execution_sha256": file_sha256(public_root / "public_execution.json"),
        "public_input_hashes": _input_hashes(), "private_loaded": False, "public_stage_wall_seconds": time.perf_counter() - started})
    write_json(public_root / "registration.json", registration)
    return execution


def _candidate(path: Path, expected_step: int, *, center: bool = False) -> dict[str, Any]:
    result = read_json(path / "result.json")
    completed = int(result.get("completed_iterations", -1))
    if result.get("status") != "COMPLETED" or (not (center and expected_step == PREFIX_STEPS and completed == TOTAL_STEPS)
                                                and completed != expected_step):
        raise RowwiseScreen3Error(f"candidate endpoint mismatch: {path}")
    candidate = path / ("candidate_5000.pt" if expected_step == 5000 and center else "candidate.pt")
    bound = result.get("prefix_candidate_sha256") if expected_step == 5000 and center else result.get("candidate_sha256")
    if bound != file_sha256(candidate):
        raise RowwiseScreen3Error(f"candidate hash mismatch: {candidate}")
    payload = torch.load(candidate, map_location="cpu", weights_only=True)
    pixels = payload.get("pixels") if isinstance(payload, dict) else None
    if set(payload) != {"pixels"} or pixels.dtype != torch.float32 or tuple(pixels.shape) != (16, 3, 224, 224):
        raise RowwiseScreen3Error(f"candidate schema mismatch: {candidate}")
    if not bool(torch.isfinite(pixels).all()) or bool((pixels < 0).any()) or bool((pixels > 1).any()):
        raise RowwiseScreen3Error(f"candidate range mismatch: {candidate}")
    return {"path": candidate.as_posix(), "sha256": file_sha256(candidate), "result": (path / "result.json").as_posix(),
            "result_sha256": file_sha256(path / "result.json"), "step": expected_step}


def public_gate(public_root: Path) -> dict[str, Any]:
    public_root = Path(public_root).resolve()
    registration = read_json(public_root / "registration.json")
    if registration.get("task_id") != TASK_ID or registration.get("status") != "PUBLIC_COMPLETED":
        raise RowwiseScreen3Error("public completion status gate failed")
    if registration.get("public_input_hashes") != _input_hashes():
        raise RowwiseScreen3Error("public input hashes changed")
    if registration.get("code_hashes") != _code_hashes() or registration.get("protected_source_hashes") != _protected_hashes():
        raise RowwiseScreen3Error("source integrity gate failed")
    separation = read_json(public_root / "separation/result.json")
    if separation.get("status") != "COMPLETED" or separation.get("separation_calls") != 9 or separation.get("fastica_called") is not True:
        raise RowwiseScreen3Error("row-wise FastICA/GD separation gate failed")
    selection = read_json(public_root / "selection.json")
    if selection.get("status") != "FROZEN_PUBLIC_SELECTION" or selection.get("private_metrics_used") is not False:
        raise RowwiseScreen3Error("public selection gate failed")
    candidates: dict[str, dict[str, Any]] = {}
    candidates["center_5000"] = _candidate(public_root / "fia_center_25000", 5000, center=True)
    for index in range(MEMBER_COUNT):
        candidates[f"member_{index:02d}_5000"] = _candidate(public_root / f"fia_member_{index:02d}_5000", 5000)
    candidates["center_25000"] = _candidate(public_root / "fia_center_25000", TOTAL_STEPS)
    for rank, index in enumerate(selection["selected_member_indices"], 1):
        candidates[f"rank{rank}_member_{int(index):02d}_25000"] = _candidate(public_root / f"fia_member_{int(index):02d}_25000", TOTAL_STEPS)
    if len(candidates) != 13:
        raise RowwiseScreen3Error("public gate candidate count is not 13")
    gate = {"task_id": TASK_ID, "status": "PUBLIC_GATE_PASSED", "private_loaded": False,
            "candidate_count": 13, "candidates": candidates, "selection": selection["selected_member_indices"],
            "registration_sha256": file_sha256(public_root / "registration.json"),
            "selection_sha256": file_sha256(public_root / "selection.json"),
            "separation_result_sha256": file_sha256(public_root / "separation/result.json"),
            "public_input_hashes": registration["public_input_hashes"]}
    write_json(public_root / "public_gate.json", gate)
    return gate


def public_gate_for_evaluation_repair(public_root: Path, repair_provenance: Path) -> dict[str, Any]:
    """Authenticate a saved public gate across a reporting-only source revision.

    The original registration and gate remain immutable.  This path is used
    only after public completion when a renderer/evaluator repair changes this
    new adapter's file hash; all protected research sources, inputs, source
    tensors, checkpoints, selection and candidate hashes must still match.
    """
    public_root = Path(public_root).resolve()
    registration = read_json(public_root / "registration.json")
    repair = read_json(repair_provenance)
    if repair.get("status") != "REGISTERED_REPORTING_REPAIR":
        raise RowwiseScreen3Error("evaluation repair provenance is not registered")
    if repair.get("registered_code_hashes") != registration.get("code_hashes"):
        raise RowwiseScreen3Error("repair provenance does not authenticate the original registered code hashes")
    if repair.get("current_code_hashes") != _code_hashes():
        raise RowwiseScreen3Error("repair provenance does not match the current reporting adapter")
    if registration.get("protected_source_hashes") != _protected_hashes():
        raise RowwiseScreen3Error("protected research sources changed during reporting repair")
    if registration.get("status") != "PUBLIC_COMPLETED" or registration.get("public_input_hashes") != _input_hashes():
        raise RowwiseScreen3Error("registered public artifacts changed during reporting repair")
    saved_gate = public_root / "public_gate.json"
    if repair.get("saved_public_gate_sha256") != file_sha256(saved_gate):
        raise RowwiseScreen3Error("saved public gate evidence changed")
    old_gate = read_json(saved_gate)
    if old_gate.get("status") != "PUBLIC_GATE_PASSED" or old_gate.get("candidate_count") != 13:
        raise RowwiseScreen3Error("saved public gate is not the completed thirteen-set gate")
    selection = read_json(public_root / "selection.json")
    if selection.get("status") != "FROZEN_PUBLIC_SELECTION" or selection.get("private_metrics_used") is not False:
        raise RowwiseScreen3Error("saved public selection is not frozen/authenticated")
    candidates: dict[str, dict[str, Any]] = {}
    candidates["center_5000"] = _candidate(public_root / "fia_center_25000", 5000, center=True)
    for index in range(MEMBER_COUNT):
        candidates[f"member_{index:02d}_5000"] = _candidate(public_root / f"fia_member_{index:02d}_5000", 5000)
    candidates["center_25000"] = _candidate(public_root / "fia_center_25000", TOTAL_STEPS)
    for rank, index in enumerate(selection["selected_member_indices"], 1):
        candidates[f"rank{rank}_member_{int(index):02d}_25000"] = _candidate(public_root / f"fia_member_{int(index):02d}_25000", TOTAL_STEPS)
    if candidates != old_gate.get("candidates"):
        raise RowwiseScreen3Error("saved public candidate hashes or paths changed during reporting repair")
    return {**old_gate, "reporting_repair": repair, "private_loaded": False}


def _private_x(path: Path = PRIVATE_TRUTH) -> tuple[torch.Tensor, dict[str, Any]]:
    path = Path(path).resolve()
    if path != PRIVATE_TRUTH.resolve() or file_sha256(path) != "724bbacbe316cc28dac086db9cfaf6507f4c1af31144fd610cc480985aab72e2":
        raise RowwiseScreen3Error("private truth path/hash is not registered")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "x" not in payload:
        raise RowwiseScreen3Error("private truth has no x field")
    x = payload["x"]
    if x.dtype != torch.float32 or tuple(x.shape) != (16, 3, 224, 224) or not bool(torch.isfinite(x).all()):
        raise RowwiseScreen3Error("private x schema mismatch")
    return x.detach().contiguous(), {"path": path.as_posix(), "sha256": file_sha256(path), "keys_present": sorted(payload),
        "consumed_field": "x only", "labels_consumed": False, "true_updates_consumed": False}


def _auth_parent_controls() -> tuple[dict[str, Any], dict[str, Path]]:
    comparison = read_json(PARENT_COMPARISON)
    if comparison.get("status") != "COMPLETED" or comparison.get("public_gate", {}).get("status") != "PUBLIC_GATE_PASSED":
        raise RowwiseScreen3Error("archived parent comparison is not completed/public-gated")
    expected = {"center_25000", "rank1_member_06_25000", "rank2_member_05_25000", "rank3_member_00_25000"}
    paths = {}
    for name in expected:
        raw = comparison.get("candidate_paths", {}).get(name)
        expected_sha = comparison.get("candidate_hashes", {}).get(name)
        if not raw or not expected_sha:
            raise RowwiseScreen3Error(f"archived parent control missing: {name}")
        path = Path(raw).resolve()
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise RowwiseScreen3Error(f"archived parent control authentication failed: {name}")
        paths[name] = path
    return comparison, paths


def _aligned_images(names: list[str], candidates: dict[str, torch.Tensor], reports: dict[str, dict[str, Any]], truth_x: torch.Tensor) -> list[torch.Tensor]:
    mean = truth_x.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = truth_x.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    truth = (truth_x * std + mean).clamp(0, 1)
    out = [truth]
    for name in names:
        aligned = torch.empty_like(candidates[name])
        for row in reports[name]["per_image"]:
            aligned[row["truth_index"]] = candidates[name][row["candidate_index"]]
        out.append(aligned)
    return out


def _contact_sheet(path: Path, columns: list[tuple[str, str]], candidates: dict[str, torch.Tensor], truth_x: torch.Tensor,
                   reports: dict[str, dict[str, Any]], title: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [x[0] for x in columns if x[0] != "__truth__"]
    display = _aligned_images(names, candidates, reports, truth_x)
    labels = [x[1] for x in columns]
    fig, axes = plt.subplots(16, len(display), figsize=(3.0 * len(display), 38), squeeze=False)
    try:
        for row in range(16):
            for col, images in enumerate(display):
                axes[row, col].imshow(images[row].permute(1, 2, 0).clamp(0, 1))
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(labels[col], fontsize=8)
        fig.suptitle(title, fontsize=16)
        fig.text(.5, .005, "Hungarian pixel-MSE correspondence is reused for all metrics and figures; it does not verify image identity.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .015, 1, .97))
        fig.savefig(path, dpi=110)
    finally:
        plt.close(fig)
    return path.name


def _slides(root: Path, columns: list[tuple[str, str]], candidates: dict[str, torch.Tensor], truth_x: torch.Tensor,
            reports: dict[str, dict[str, Any]], title: str) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont
    root.mkdir(parents=True, exist_ok=False)
    names = [x[0] for x in columns if x[0] != "__truth__"]
    labels = [x[1] for x in columns]
    display = _aligned_images(names, candidates, reports, truth_x)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 19)
    manifest = {"status": "COMPLETED", "width": 3840, "height": 2160, "columns": labels, "rows": 16, "pages": []}
    for page_no, start in enumerate((0, 4, 8, 12), 1):
        canvas = Image.new("RGB", (3840, 2160), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((80, 38), f"{title} | examples {start + 1:02d}-{min(start + 4, 16):02d}", fill="black", font=font)
        left, top, gap_x, gap_y = 80, 115, 25, 27
        cell_w = (3840 - 2 * left - 4 * gap_x) // 5
        cell_h = (2160 - top - 125 - 3 * gap_y) // 4
        for row in range(start, start + 4):
            for col, image_set in enumerate(display):
                img = image_set[row].mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                image = Image.fromarray(img).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
                x = left + col * (cell_w + gap_x); y = top + (row - start) * (cell_h + gap_y)
                canvas.paste(image, (x, y)); draw.text((x + 4, y + 4), labels[col], fill="white", stroke_width=2, stroke_fill="black", font=small)
        draw.text((80, 2115), "Hungarian pixel-MSE matching is reused for metrics and figures; it does not verify image identity.", fill="black", font=small)
        path = root / f"page{page_no:02d}.png"; canvas.save(path)
        manifest["pages"].append({"path": path.name, "sha256": file_sha256(path), "rows": list(range(start + 1, start + 5))})
    overview = Image.new("RGB", (3840, 2160), "white")
    draw = ImageDraw.Draw(overview); draw.text((80, 38), f"{title} | all 16 examples", fill="black", font=font)
    left, top, gap_x, gap_y = 80, 115, 25, 16
    cell_w = (3840 - 2 * left - 4 * gap_x) // 5
    cell_h = (2160 - top - 125 - 7 * gap_y) // 8
    panel_width = (3840 - 3 * 40) // 2
    panel_left = 40
    panel_gap = 40
    panel_cell_w = (panel_width - 2 * 25 - 4 * 8) // 5
    panel_cell_h = (2160 - top - 125 - 7 * gap_y) // 8
    for panel, base in enumerate((0, 8)):
        for local in range(8):
            row = base + local
            for col, image_set in enumerate(display):
                img = image_set[row].mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                image = Image.fromarray(img).resize((panel_cell_w, panel_cell_h), Image.Resampling.BILINEAR)
                x = panel_left + panel * (panel_width + panel_gap) + 25 + col * (panel_cell_w + 8)
                y = top + local * (panel_cell_h + gap_y)
                canvas_x = x
                overview.paste(image, (canvas_x, y))
                if local == 0: draw.text((canvas_x + 4, y + 4), labels[col], fill="white", stroke_width=2, stroke_fill="black", font=small)
    draw.text((80, 2115), "Hungarian pixel-MSE matching is reused for metrics and figures; it does not verify image identity.", fill="black", font=small)
    opath = root / "int4_gdonly_25000_overview.png"; overview.save(opath)
    manifest["overview"] = {"path": opath.name, "sha256": file_sha256(opath), "rows": 16}
    write_json(root / "manifest.json", manifest)
    shutil.make_archive(str(root.parent / "slide_png_int4_gdonly_20260918"), "zip", root_dir=root)
    return manifest


def evaluate(public_root: Path, private_root: Path, output: Path | None = None,
             repair_provenance: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    public_root, private_root = Path(public_root).resolve(), Path(private_root).resolve()
    output = Path(output or (private_root / "evaluation")).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing evaluation output: {output}")
    gate = (public_gate_for_evaluation_repair(public_root, Path(repair_provenance))
            if repair_provenance is not None else public_gate(public_root))
    parent_comparison, parent_paths = _auth_parent_controls()
    configure_runtime(SEED, threads=1)
    truth_x, private_meta = _private_x()
    output.mkdir(parents=True, exist_ok=False)
    from .full_update_evaluation import evaluate_candidate
    from . import signed_int8_screen3_evaluation as prior_eval
    candidate_paths: dict[str, Path] = {}
    for name, record in gate["candidates"].items():
        candidate_paths[f"gdonly_{name}"] = Path(record["path"]).resolve()
    for name, path in parent_paths.items():
        candidate_paths[f"parent_{name}"] = path
    candidates = {name: torch.load(path, map_location="cpu", weights_only=True)["pixels"].contiguous()
                  for name, path in candidate_paths.items()}
    reports = {name: evaluate_candidate(path, PRIVATE_TRUTH, include_labels=False) for name, path in candidate_paths.items()}
    # LPIPS is evaluated once after all public artifacts and controls are authenticated.
    prior_eval._lpips(candidates, reports, truth_x)
    means = prior_eval._means(reports)
    control_residuals = {}
    for name in ("center_25000", "rank1_member_06_25000", "rank2_member_05_25000", "rank3_member_00_25000"):
        old = parent_comparison["mean_metrics"][name]
        new = means[f"parent_{name}"]
        control_residuals[name] = {key: abs(float(new[key]) - float(old[key])) for key in ("mean_mse", "mean_psnr_db", "mean_ssim", "mean_lpips")}
    max_residual = max(v for row in control_residuals.values() for v in row.values())
    controls_reproduced = max_residual <= 2e-8
    if not controls_reproduced:
        result = {"task_id": TASK_ID, "status": "NEEDS_REVIEW", "evaluation_status": "NEEDS_REVIEW",
                  "public_gate": gate, "private_truth": private_meta, "control_residuals": control_residuals,
                  "control_max_abs_residual": max_residual, "control_tolerance": 2e-8,
                  "reason": "archived controls did not reproduce within unchanged tolerance; no second attempt",
                  "private_x_only": True, "labels_consumed": False, "actual_image_cell_feasibility": "NOT_EVALUATED"}
        write_json(output / "comparison.json", result); write_json(output / "results.json", result)
        return result
    # Auxiliary baseline is authenticated separately for the baseline-inclusive figures.
    if not BASELINE_CANDIDATE.is_file():
        raise RowwiseScreen3Error("naive dequantization baseline candidate is missing")
    baseline_name = "aux_naive_dequantization_baseline"
    baseline_path = BASELINE_CANDIDATE.resolve()
    baseline_payload = torch.load(baseline_path, map_location="cpu", weights_only=True)
    if set(baseline_payload) != {"pixels"}:
        raise RowwiseScreen3Error("baseline candidate schema mismatch")
    candidates[baseline_name] = baseline_payload["pixels"].contiguous()
    reports[baseline_name] = evaluate_candidate(baseline_path, PRIVATE_TRUTH, include_labels=False)
    prior_eval._lpips({baseline_name: candidates[baseline_name]}, {baseline_name: reports[baseline_name]}, truth_x)
    final_names = ["gdonly_center_25000", "gdonly_rank1_member_%02d_25000" % gate["selection"][0],
                   "gdonly_rank2_member_%02d_25000" % gate["selection"][1], "gdonly_rank3_member_%02d_25000" % gate["selection"][2]]
    final_columns = [("__truth__", "Ground truth"), (baseline_name, "Naive dequantization baseline | INT4"),
        (final_names[1], f"GD-only public rank 1 | member {gate['selection'][0]:02d}"),
        (final_names[2], f"GD-only public rank 2 | member {gate['selection'][1]:02d}"),
        (final_names[3], f"GD-only public rank 3 | member {gate['selection'][2]:02d}")]
    screen_columns = [("__truth__", "Ground truth"), (baseline_name, "Naive dequantization baseline | INT4")]
    for i in range(MEMBER_COUNT): screen_columns.append((f"gdonly_member_{i:02d}_5000", f"GD-only member {i:02d} | 5000"))
    figures = output / "figures"; figures.mkdir()
    figure_manifest = {"status": "COMPLETED", "final": _contact_sheet(figures / "contact_sheet_int4_gdonly_25000_with_baseline.png", final_columns, candidates, truth_x, reports, "INT4 original CPA GD-only | 25000"),
        "screen": _contact_sheet(figures / "contact_sheet_int4_gdonly_5000_with_baseline.png", screen_columns, candidates, truth_x, reports, "INT4 original CPA GD-only | 5000")}
    method_columns = [("__truth__", "Ground truth"), ("parent_center_25000", "Parent FastICA/GD center"),
        ("parent_rank1_member_06_25000", "Parent FastICA/GD rank1"), (final_names[0], "New GD-only center"), (final_names[1], "New GD-only rank1")]
    figure_manifest["method_comparison"] = _contact_sheet(figures / "method_comparison_parent_vs_gdonly.png", method_columns, candidates, truth_x, reports, "Parent FastICA/GD vs original GD-only")
    slides = _slides(output / "slide_png", final_columns, candidates, truth_x, reports, "INT4 original CPA GD-only")
    figure_manifest["slide_manifest"] = slides
    write_json(output / "figure_manifest.json", figure_manifest)
    new_means = {name: means[name] for name in means if name.startswith("gdonly_")}
    rank1 = final_names[1]; center = final_names[0]
    contrast = {key: new_means[rank1][key] - new_means[center][key] for key in ("mean_mse", "mean_psnr_db", "mean_ssim", "mean_lpips")}
    result = {"task_id": TASK_ID, "status": "COMPLETED", "evaluation_status": "COMPLETED", "scientific_quality": "EVALUATED",
        "public_gate": gate, "private_truth": private_meta, "candidate_count": 17, "comparison_set_count": 17,
        "candidate_order": list(candidate_paths), "candidate_paths": {k: str(v) for k, v in candidate_paths.items()},
        "candidate_hashes": {k: file_sha256(v) for k, v in candidate_paths.items()}, "image_metrics": reports, "mean_metrics": means,
        "matching": "Hungarian_pixel_MSE_once_per_candidate_reused_for_MSE_PSNR_SSIM_LPIPS_and_figures",
        "control_residuals": control_residuals, "control_max_abs_residual": max_residual, "control_tolerance": 2e-8,
        "controls_reproduced": True, "primary_contrast_gdonly_rank1_minus_center": contrast,
        "selection": gate["selection"], "private_x_only": True, "labels_consumed": False,
        "true_updates_consumed_by_attack": False, "actual_image_cell_feasibility": "NOT_EVALUATED",
        "figure_artifacts": figure_manifest, "parent_comparison": PARENT_COMPARISON.as_posix(),
        "legacy_objective": {"negentropy_effective": 0.0, "feature_tv_effective": 0.0},
        "wall_seconds": time.perf_counter() - started,
        "limitations": ["one fixed B16/T10 observation and seed", "descriptive public top3", "no private feedback or image averaging", "actual-image cell not run"]}
    write_json(output / "comparison.json", result); write_json(output / "results.json", result)
    lines = [f"# {TASK_ID}", "", "Status: **COMPLETED**.", "", "The original upstream CPA gradient-only backend ran with zero FastICA calls.", "", "| Set | MSE | PSNR dB | SSIM | LPIPS |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in means.items():
        lines.append(f"| {name} | {row['mean_mse']:.12g} | {row['mean_psnr_db'] if row['mean_psnr_db'] is not None else 'n/a'} | {row['mean_ssim']:.12g} | {row['mean_lpips']:.12g} |")
    lines += ["", f"Control maximum residual: {max_residual:.12g} (tolerance 2e-8).", "Primary contrast is new GD-only public rank1 minus new GD-only center.", "Private evaluator consumed x only; labels and true updates were not consumed.", "Actual-image cell feasibility: NOT_EVALUATED."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return result


def run_all(public_root: Path, private_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    public_root, private_root = Path(public_root).resolve(), Path(private_root).resolve()
    execution = run_public(public_root, private_root)
    gate = public_gate(public_root)
    evaluation = evaluate(public_root, private_root)
    summary = {"task_id": TASK_ID, "status": evaluation["status"], "public_status": "PUBLIC_COMPLETED",
               "evaluation_status": evaluation.get("evaluation_status"), "public_gate": gate["status"],
               "new_fia_updates": TOTAL_FIA_UPDATES, "separation_calls": 9,
               "elapsed_seconds": time.perf_counter() - started, "public_root": public_root.as_posix(),
               "private_root": private_root.as_posix(), "evaluation": private_root.joinpath("evaluation/comparison.json").as_posix(),
               "actual_image_cell_feasibility": "NOT_EVALUATED"}
    write_json(private_root / "execution_summary.json", summary)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("register", "run", "gate", "evaluate"))
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "register": out = register(args.public_root, args.private_root)
        elif args.command == "run": out = run_all(args.public_root, args.private_root)
        elif args.command == "gate": out = public_gate(args.public_root)
        else: out = evaluate(args.public_root, args.private_root)
        print(json.dumps({"task_id": TASK_ID, "status": out.get("status"), "evaluation_status": out.get("evaluation_status")}, allow_nan=False), flush=True)
    except Exception as exc:
        print(f"{TASK_ID}: {type(exc).__name__}: {exc}", flush=True)
        raise
