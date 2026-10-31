"""Isolated public 7/6/5/4 screen3 runner using the repaired FIA writer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from .deterministic_hybrid_setup import configure_runtime
from .signed_combination_ica import CombinationICAConfig, separate_observation, tensor_sha256, tensor_stats
from .signed_bits_observation import (
    CHECKPOINT_SHA, FC_SHAPES, SUPPORTED_BITS, collect, digest, read_bundle,
)
from .signed_bits_randomized import align_preserve_norm, sample_matrix
from .signed_int8_screen3_fia import FIA_CONFIG, PREFIX_STEPS, TOTAL_STEPS, run_trajectory
from .upstream_naive import source_hashes as upstream_source_hashes


TASK_ID = "CPA_SIGNED_BITS7654_SCREEN3_20260918"
PLAN_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth"
SOURCE_COUNT, FEATURE_DIM, OBSERVATION_ROWS = 16, 25088, 4096
MEMBER_COUNT = 8
MEMBER_SEEDS = [1234 + 1000003 * index for index in range(MEMBER_COUNT)]
SEPARATION_CONFIG = CombinationICAConfig(refinement_iterations=4000)
MAX_SECONDS_PER_BIT = 12600
MAX_CUDA_ALLOCATED_BYTES = 12 * 1024**3
TOTAL_FIA_UPDATES = 125000


class BitsScreen3Error(RuntimeError):
    pass


def bit_task_id(bits: int) -> str:
    if bits not in SUPPORTED_BITS:
        raise BitsScreen3Error(f"unsupported bit width: {bits}")
    return f"{TASK_ID}_B{bits}"


def file_sha256(path: Path) -> str:
    return digest(Path(path))


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")
    temp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BitsScreen3Error(f"expected JSON object: {path}")
    return value


def _tensor_hash(value: torch.Tensor) -> str:
    return tensor_sha256(value.detach().cpu().contiguous())


def _code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__), Path(__file__).with_name("signed_bits_observation.py"),
        Path(__file__).with_name("signed_bits_randomized.py"),
        Path(__file__).with_name("signed_bits_screen3_evaluation.py"),
        Path(__file__).with_name("signed_int8_screen3_fia.py"),
        Path(__file__).with_name("signed_combination_ica.py"),
        Path(__file__).with_name("upstream_naive.py"),
    ]
    return {path.resolve().as_posix(): file_sha256(path) for path in paths}


def _protected_hashes() -> dict[str, str]:
    paths = [
        REPO_ROOT / "quantization_aware_cpa/qacpa/signed_combination_ica.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/upstream_naive.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/deterministic_hybrid_setup.py",
        REPO_ROOT / "quantized_update_cpa/cell_utils.py",
        REPO_ROOT / "quantized_update_cpa/blind_alignment.py",
        REPO_ROOT / "official_cpa/src/gradient_inversion.py",
        REPO_ROOT / "official_cpa/src/feature_inversion.py",
        REPO_ROOT / "official_cpa/src/vgg16.py",
    ]
    return {path.resolve().as_posix(): file_sha256(path) for path in paths}


def _public_input_hashes(observation: Path) -> dict[str, str]:
    manifest = read_json(Path(observation) / "manifest.json")
    paths = [Path(observation) / "manifest.json"]
    paths += [Path(observation) / item["file"] for item in manifest["layers"].values()]
    paths.append(CHECKPOINT)
    return {path.resolve().as_posix(): file_sha256(path) for path in paths}


def register_bit(bits: int, public_root: Path, private_root: Path) -> dict[str, Any]:
    from .signed_bits_observation import codec_metadata, qmax_for_bits
    qmax_for_bits(bits)
    public_root, private_root = Path(public_root).resolve(), Path(private_root).resolve()
    if public_root.exists() or private_root.exists() or public_root in private_root.parents or private_root in public_root.parents:
        raise FileExistsError(f"bit {bits} output roots must be new and disjoint")
    if not CHECKPOINT.is_file() or file_sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise BitsScreen3Error("registered VGG16 checkpoint changed")
    public_root.mkdir(parents=True, exist_ok=False)
    private_root.mkdir(parents=True, exist_ok=False)
    registration = {
        "task_id": bit_task_id(bits), "schedule_task_id": TASK_ID, "plan_version": PLAN_VERSION,
        "status": "REGISTERED", "bits": bits, "qmax": qmax_for_bits(bits),
        "method": "bit_generic_randomized8_members_individual_FIA5000_public_top3_to_total25000",
        "stage_order": ["SENDER", "SEPARATE_9_CALLS", "ALIGN_PUBLIC", "FIA_CENTER_25000",
                        "FIA_MEMBERS_5000", "PUBLIC_SELECTION_TOP3", "FIA_TOP3_CONTINUATION_20000",
                        "PUBLIC_GATE", "EVALUATE_13_FROZEN_SETS"],
        "public_output_root": public_root.as_posix(), "private_output_root": private_root.as_posix(),
        "public_observation": (public_root / "observation").as_posix(),
        "separator_config": SEPARATION_CONFIG.as_dict(), "member_count": MEMBER_COUNT,
        "member_seeds": MEMBER_SEEDS, "fia_config": FIA_CONFIG,
        "screen_config": {"prefix_steps": PREFIX_STEPS, "total_steps": TOTAL_STEPS,
                          "screen_score": "feature_cosine_distance_plus_0.1_normalized_TV",
                          "tie_break": "member_index_ascending", "selected_count": 3,
                          "continuation_steps": 20000, "total_fia_updates": TOTAL_FIA_UPDATES},
        "alignment": {"reference": "public_center_signed_sources", "matching": "Hungarian_max_sum_abs_cosine",
                      "correction": "permutation_and_sign_only", "row_norm_rescaling": False,
                      "feature_average": False},
        "codec": codec_metadata(bits), "source_count": SOURCE_COUNT, "source_shape": [SOURCE_COUNT, FEATURE_DIM],
        "checkpoint": CHECKPOINT.as_posix(), "checkpoint_sha256": CHECKPOINT_SHA,
        "public_input_hashes": {}, "code_hashes": _code_hashes(), "protected_source_hashes": _protected_hashes(),
        "upstream_source_hashes": upstream_source_hashes(),
        "private_data_policy": {"attack_reads_public_observation_only": True, "load_after_public_gate": True,
                                "sender_true_updates_consumed": True, "attack_true_updates_consumed": False,
                                "evaluator_consumed_field": "x only", "labels_consumed": False},
        "actual_image_cell_feasibility": "NOT_EVALUATED", "scientific_quality": "NOT_EVALUATED",
        "budget_seconds": MAX_SECONDS_PER_BIT, "cuda_allocated_cap_bytes": MAX_CUDA_ALLOCATED_BYTES,
        "registered_before_execution": True,
    }
    write_json(public_root / "registration.json", registration)
    write_json(private_root / "task_boundary.json", {
        "task_id": bit_task_id(bits), "bits": bits, "true_updates_consumed_by_sender": True,
        "attack_true_updates_consumed": False, "labels_consumed": False, "images_consumed": False,
    })
    return registration


def _save_source(folder: Path, payload: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=False)
    torch.save(payload, folder / "sources.pt")
    write_json(folder / "result.json", metadata)


def _source_payload(path: Path, index: int) -> tuple[torch.Tensor, str, dict[str, Any]]:
    payload = torch.load(Path(path) / "sources.pt", map_location="cpu", weights_only=True)
    result = read_json(Path(path) / "result.json")
    if index == -1:
        if set(payload) != {"S_hat", "target_z"}:
            raise BitsScreen3Error("center source schema mismatch")
        source, target = payload["S_hat"], payload["target_z"]
    else:
        if set(payload) != {"S_hat_native", "S_hat_aligned", "target_z"}:
            raise BitsScreen3Error(f"member source schema mismatch: {index}")
        source, target = payload["S_hat_aligned"], payload["target_z"]
    if (source.dtype != torch.float32 or target.dtype != torch.float32
            or tuple(source.shape) != (SOURCE_COUNT, FEATURE_DIM)
            or tuple(target.shape) != (SOURCE_COUNT, FEATURE_DIM)
            or not bool(torch.isfinite(source).all()) or not torch.equal(target, source.abs())):
        raise BitsScreen3Error("invalid source target binding")
    if result.get("target_z_sha256") != _tensor_hash(target):
        raise BitsScreen3Error("source target hash mismatch")
    if index == -1 and result.get("S_hat_sha256") != _tensor_hash(source):
        raise BitsScreen3Error("center source hash mismatch")
    if index != -1 and (result.get("S_hat_native_sha256") != _tensor_hash(payload["S_hat_native"])
                        or result.get("S_hat_aligned_sha256") != _tensor_hash(source)):
        raise BitsScreen3Error("member native/aligned source hash mismatch")
    return target.contiguous(), file_sha256(Path(path) / "sources.pt"), result


def _source_entry(separation_root: Path, index: int) -> tuple[torch.Tensor, str, dict[str, Any]]:
    return _source_payload(Path(separation_root) / ("center" if index == -1 else f"member_{index:02d}"), index)


def separate(bits: int, public_observation: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    public_observation, output = Path(public_observation).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing separation output: {output}")
    configure_runtime(42, threads=1)
    manifest, entries = read_bundle(public_observation, bits)
    q, scale = entries["classifier.0.weight"]
    X0 = (q.to(torch.float32) * scale).contiguous()
    if tuple(X0.shape) != (OBSERVATION_ROWS, FEATURE_DIM):
        raise BitsScreen3Error("public FC0 matrix shape mismatch")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "registration.json", {"task_id": bit_task_id(bits), "status": "REGISTERED", "stage": "SEPARATE",
        "bits": bits, "public_manifest_sha256": file_sha256(public_observation / "manifest.json"),
        "config": SEPARATION_CONFIG.as_dict(), "separation_calls_registered": 9,
        "member_count": MEMBER_COUNT, "member_seeds": MEMBER_SEEDS, "private_data_used": False})
    records = []
    center = separate_observation(X0, SOURCE_COUNT, config=SEPARATION_CONFIG, device="cpu")
    center_source = center.S_hat.detach().cpu().contiguous()
    center_target = center_source.abs().contiguous()
    _save_source(output / "center", {"S_hat": center_source, "target_z": center_target}, {
        "task_id": bit_task_id(bits), "status": "COMPLETED", "index": "center", "separation_call": 1,
        "input_sha256": _tensor_hash(X0), "S_hat_sha256": _tensor_hash(center_source),
        "target_z_sha256": _tensor_hash(center_target), "fastica": center.metadata["fastica"],
        "refinement": center.metadata["refinement"], "whitening": center.metadata["whitening"],
        "final_losses": center.refinement.final_losses, "private_data_used": False})
    records.append({"index": "center", "separation_call": 1, "input_sha256": _tensor_hash(X0),
                    "S_hat_sha256": _tensor_hash(center_source), "target_z_sha256": _tensor_hash(center_target),
                    "fastica": center.metadata["fastica"]})
    member_records = []
    for member_index, seed in enumerate(MEMBER_SEEDS):
        sampled, sample_meta = sample_matrix(q, scale, bits=bits, seed=seed, chunk_rows=32)
        separated = separate_observation(sampled, SOURCE_COUNT, config=SEPARATION_CONFIG, device="cpu")
        native = separated.S_hat.detach().cpu().contiguous()
        aligned, alignment = align_preserve_norm(center_source, native, member_index)
        target = aligned.abs().contiguous()
        _save_source(output / f"member_{member_index:02d}",
                     {"S_hat_native": native, "S_hat_aligned": aligned, "target_z": target}, {
            "task_id": bit_task_id(bits), "status": "COMPLETED", "index": member_index,
            "separation_call": member_index + 2, "input_sha256": sample_meta["tensor_sha256"],
            "S_hat_native_sha256": _tensor_hash(native), "S_hat_aligned_sha256": _tensor_hash(aligned),
            "target_z_sha256": _tensor_hash(target), "sample": sample_meta, "alignment": alignment,
            "fastica": separated.metadata["fastica"], "refinement": separated.metadata["refinement"],
            "whitening": separated.metadata["whitening"], "final_losses": separated.refinement.final_losses,
            "private_data_used": False})
        member_records.append({"index": member_index, "separation_call": member_index + 2,
            "input_sha256": sample_meta["tensor_sha256"], "sample": sample_meta,
            "S_hat_native_sha256": _tensor_hash(native), "S_hat_aligned_sha256": _tensor_hash(aligned),
            "target_z_sha256": _tensor_hash(target), "alignment": alignment,
            "fastica": separated.metadata["fastica"]})
        records.append(member_records[-1])
        del sampled, separated, native, aligned, target
    result = {"task_id": bit_task_id(bits), "schedule_task_id": TASK_ID, "plan_version": PLAN_VERSION,
        "status": "COMPLETED", "stage": "SEPARATE", "bits": bits,
        "public_manifest_sha256": file_sha256(public_observation / "manifest.json"),
        "config": SEPARATION_CONFIG.as_dict(), "source_shape": [SOURCE_COUNT, FEATURE_DIM],
        "separation_calls": 9, "center_is_reference_not_member": True, "member_count": MEMBER_COUNT,
        "member_seeds": MEMBER_SEEDS, "records": records,
        "alignment": {"reference": "center signed S_hat", "correction": "permutation_sign_only",
                      "row_norm_rescaling": False, "feature_average": False, "members": member_records},
        "runtime": {"device": "cpu", "dtype": "float32", "threads": 1},
        "protected_source_hashes": _protected_hashes(), "upstream_source_hashes": upstream_source_hashes(),
        "private_data_used": False, "labels_consumed": False, "true_updates_consumed": False,
        "actual_image_cell_feasibility": "NOT_EVALUATED", "scientific_quality": "NOT_EVALUATED",
        "timing": {"wall_seconds": time.perf_counter() - started}}
    write_json(output / "result.json", result)
    write_json(output / "registration.json", {**read_json(output / "registration.json"),
                                               "status": "COMPLETED", "result_sha256": file_sha256(output / "result.json")})
    return result


def _progress(event: dict[str, Any]) -> None:
    print(json.dumps({"stage": "FIA", **event}, allow_nan=False), flush=True)


def freeze_public_selection(screen_results: list[dict[str, Any]], bits: int) -> dict[str, Any]:
    if len(screen_results) != MEMBER_COUNT or sorted(int(x.get("member_index", -1)) for x in screen_results) != list(range(MEMBER_COUNT)):
        raise BitsScreen3Error("selection requires exactly eight member scores")
    for item in screen_results:
        score = item.get("score", {})
        if (not isinstance(score.get("score"), (int, float))
                or not bool(torch.isfinite(torch.tensor(float(score["score"]))))
                or score.get("raw_state_unchanged") is not True):
            raise BitsScreen3Error(f"invalid public score for member {item.get('member_index')}")
    ranking = sorted(screen_results, key=lambda x: (float(x["score"]["score"]), int(x["member_index"])))
    return {"task_id": bit_task_id(bits), "status": "FROZEN_PUBLIC_SELECTION", "bits": bits,
        "completed_step": PREFIX_STEPS, "ranking": [
            {"rank": rank + 1, "member_index": item["member_index"], "score": item["score"]["score"],
             "feature_cosine_distance_mean": item["score"].get("feature_cosine_distance_mean"),
             "tv_normalized": item["score"].get("tv_normalized"), "checkpoint_sha256": item["result"]["checkpoint_sha256"],
             "candidate_sha256": item["result"]["candidate_sha256"], "target_z_sha256": item["target_z_sha256"],
             "source_file_sha256": item["source_file_sha256"]} for rank, item in enumerate(ranking)],
        "selected_member_indices": [item["member_index"] for item in ranking[:3]],
        "tie_break": "member_index_ascending", "selection_rule": "post5000_feature_cosine_distance_plus_0.1_normalized_TV",
        "private_metrics_used": False}


def _candidate(folder: Path, step: int) -> dict[str, Any]:
    folder = Path(folder)
    result = read_json(folder / "result.json")
    candidate = folder / ("candidate_5000.pt" if step == 5000 and (folder / "candidate_5000.pt").is_file() else "candidate.pt")
    expected_hash = result.get("prefix_candidate_sha256") if step == 5000 and folder.name == "fia_center_25000" else result.get("candidate_sha256")
    if result.get("status") != "COMPLETED" or result.get("completed_iterations") != (TOTAL_STEPS if step == 5000 and folder.name == "fia_center_25000" else step):
        raise BitsScreen3Error(f"candidate step/status mismatch: {folder}")
    if expected_hash != file_sha256(candidate):
        raise BitsScreen3Error(f"candidate hash mismatch: {candidate}")
    payload = torch.load(candidate, map_location="cpu", weights_only=True)
    pixels = payload.get("pixels") if isinstance(payload, dict) else None
    if set(payload) != {"pixels"} or pixels.dtype != torch.float32 or tuple(pixels.shape) != (16, 3, 224, 224):
        raise BitsScreen3Error(f"candidate schema mismatch: {candidate}")
    if not bool(torch.isfinite(pixels).all()) or bool((pixels < 0).any()) or bool((pixels > 1).any()):
        raise BitsScreen3Error(f"candidate range mismatch: {candidate}")
    return {"path": candidate.as_posix(), "sha256": file_sha256(candidate), "result": (folder / "result.json").as_posix(),
            "result_sha256": file_sha256(folder / "result.json"), "step": step}


def run_public(bits: int, public_root: Path, private_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    public_root, private_root = Path(public_root).resolve(), Path(private_root).resolve()
    registration = read_json(public_root / "registration.json")
    if registration.get("task_id") != bit_task_id(bits) or registration.get("status") != "REGISTERED":
        raise BitsScreen3Error("bit registration is not ready")
    if registration.get("code_hashes") != _code_hashes():
        raise BitsScreen3Error("bit source code hash changed after registration")
    observation = public_root / "observation"
    sender = collect(bits, output=observation, private_output=private_root / "sender")
    separation_root = public_root / "separation"
    separation = separate(bits, observation, separation_root)
    center_target, center_source_sha, _ = _source_entry(separation_root, -1)
    center_result = run_trajectory(center_target, output=public_root / "fia_center_25000",
                                   source_sha256=center_source_sha, arm="center_25000", end_steps=TOTAL_STEPS,
                                   progress=_progress)
    screen_results = []
    for index in range(MEMBER_COUNT):
        target, source_sha, _ = _source_entry(separation_root, index)
        branch = public_root / f"fia_member_{index:02d}_5000"
        result = run_trajectory(target, output=branch, source_sha256=source_sha,
                                arm=f"member_{index:02d}_5000", end_steps=PREFIX_STEPS, progress=_progress)
        score = result.get("score", {})
        screen_results.append({"member_index": index, "branch": branch.as_posix(), "result": result,
                               "score": score, "source_file_sha256": source_sha,
                               "target_z_sha256": tensor_sha256(target)})
    selection = freeze_public_selection(screen_results, bits)
    write_json(public_root / "selection.json", selection)
    continuation_results = []
    for index in selection["selected_member_indices"]:
        target, source_sha, _ = _source_entry(separation_root, int(index))
        continuation_results.append(run_trajectory(
            target, output=public_root / f"fia_member_{int(index):02d}_25000", source_sha256=source_sha,
            arm=f"member_{int(index):02d}_25000", end_steps=TOTAL_STEPS,
            parent_checkpoint=public_root / f"fia_member_{int(index):02d}_5000/checkpoint_5000.pt",
            parent_role="MEMBER_PREFIX_5000", progress=_progress))
    public_hashes = _public_input_hashes(observation)
    registration.update({"status": "PUBLIC_COMPLETED", "public_input_hashes": public_hashes,
        "sender_manifest_sha256": sender["public_manifest_sha256"], "separation_result_sha256": file_sha256(separation_root / "result.json"),
        "selection_sha256": file_sha256(public_root / "selection.json"), "public_frozen": True, "private_loaded": False,
        "public_stage_wall_seconds": time.perf_counter() - started, "scientific_quality": "NOT_EVALUATED"})
    write_json(public_root / "registration.json", registration)
    result = {"task_id": bit_task_id(bits), "schedule_task_id": TASK_ID, "bits": bits, "status": "PUBLIC_COMPLETED",
        "sender": sender, "separation": separation, "center": center_result, "screen_results": screen_results,
        "selection": selection, "continuations": continuation_results, "total_fia_updates": TOTAL_FIA_UPDATES,
        "public_input_hashes": public_hashes, "wall_seconds": time.perf_counter() - started,
        "private_data_used": False, "actual_image_cell_feasibility": "NOT_EVALUATED", "scientific_quality": "NOT_EVALUATED"}
    write_json(public_root / "public_execution.json", result)
    return result


def public_gate(bits: int, public_root: Path) -> dict[str, Any]:
    public_root = Path(public_root).resolve()
    registration = read_json(public_root / "registration.json")
    if registration.get("task_id") != bit_task_id(bits) or registration.get("status") != "PUBLIC_COMPLETED":
        raise BitsScreen3Error("bit public registration gate failed")
    if registration.get("code_hashes") != _code_hashes() or registration.get("protected_source_hashes") != _protected_hashes():
        raise BitsScreen3Error("bit source hash gate failed")
    observation = public_root / "observation"
    if registration.get("public_input_hashes") != _public_input_hashes(observation):
        raise BitsScreen3Error("bit public input hashes changed")
    separation = read_json(public_root / "separation/result.json")
    selection = read_json(public_root / "selection.json")
    if separation.get("status") != "COMPLETED" or separation.get("separation_calls") != 9:
        raise BitsScreen3Error("bit separation is incomplete")
    selected = selection.get("selected_member_indices")
    if (selection.get("status") != "FROZEN_PUBLIC_SELECTION"
            or selection.get("private_metrics_used") is not False
            or not isinstance(selected, list) or len(selected) != 3
            or sorted(int(x) for x in selected) != sorted(set(int(x) for x in selected))
            or any(int(x) not in range(MEMBER_COUNT) for x in selected)):
        raise BitsScreen3Error("bit selection is not frozen/public-only")
    candidates: dict[str, dict[str, Any]] = {}
    center = public_root / "fia_center_25000"
    candidates["center_5000"] = _candidate(center, 5000)
    for index in range(MEMBER_COUNT):
        candidates[f"member_{index:02d}_5000"] = _candidate(public_root / f"fia_member_{index:02d}_5000", 5000)
    candidates["center_25000"] = _candidate(center, TOTAL_STEPS)
    for rank, index in enumerate(selection["selected_member_indices"], 1):
        candidates[f"rank{rank}_member_{int(index):02d}_25000"] = _candidate(public_root / f"fia_member_{int(index):02d}_25000", TOTAL_STEPS)
    if len(candidates) != 13:
        raise BitsScreen3Error("bit public candidate count is not 13")
    return {"status": "PUBLIC_GATE_PASSED", "bits": bits, "private_loaded": False,
            "candidate_count": len(candidates), "candidates": candidates,
            "selection": selection["selected_member_indices"],
            "registration_sha256": file_sha256(public_root / "registration.json"),
            "selection_sha256": file_sha256(public_root / "selection.json"),
            "public_input_hashes": registration["public_input_hashes"],
            "source_code_hashes": registration["code_hashes"], "cumulative_fia_updates": TOTAL_FIA_UPDATES}


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register")
    reg.add_argument("--bits", type=int, required=True); reg.add_argument("--public-root", type=Path, required=True); reg.add_argument("--private-root", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--bits", type=int, required=True); run.add_argument("--public-root", type=Path, required=True); run.add_argument("--private-root", type=Path, required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("--bits", type=int, required=True); gate.add_argument("--public-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "register": result = register_bit(args.bits, args.public_root, args.private_root)
        elif args.command == "run": result = run_public(args.bits, args.public_root, args.private_root)
        else: result = public_gate(args.bits, args.public_root)
    except Exception as error:
        print(f"{TASK_ID}: {type(error).__name__}: {error}", flush=True)
        return 1
    print(json.dumps({"task_id": result["task_id"], "status": result["status"], "bits": result.get("bits")}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
