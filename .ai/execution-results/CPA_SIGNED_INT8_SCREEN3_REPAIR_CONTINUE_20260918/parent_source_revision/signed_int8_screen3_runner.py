"""Public-only INT8 eight-member FIA5000 screen and top-three continuation."""
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
from .signed_int8_randomized import sample_matrix
from .signed_int8_screen3_fia import FIA_CONFIG, PREFIX_STEPS, TOTAL_STEPS, run_trajectory
from .upstream_naive import source_hashes as upstream_source_hashes


TASK_ID = "CPA_SIGNED_INT8_RANDOMIZED8_SCREEN3_FIA_20260917"
PLAN_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth"
PUBLIC_ROOT = REPO_ROOT / "quantization_aware_cpa/results/signed_int8_randomized8_screen3_fia_B16_T10_20260917"
PRIVATE_ROOT = REPO_ROOT / "quantization_aware_cpa/results_private/signed_int8_randomized8_screen3_fia_B16_T10_20260917"
SOURCE_COUNT = 16
FEATURE_DIM = 25088
OBSERVATION_ROWS = 4096
MEMBER_COUNT = 8
MEMBER_SEEDS = [1234 + 1000003 * index for index in range(MEMBER_COUNT)]
SEPARATION_CONFIG = CombinationICAConfig(refinement_iterations=4000)
MAX_SECONDS = 210 * 60
MAX_CUDA_ALLOCATED_BYTES = 12 * 1024**3
TOTAL_FIA_UPDATES = 125000


class Screen3Error(RuntimeError):
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Screen3Error(f"expected JSON object: {path}")
    return value


def _must_file(path: Path) -> None:
    if not Path(path).is_file() or Path(path).is_symlink():
        raise Screen3Error(f"missing regular file: {path}")


def _tensor_hash(value: torch.Tensor) -> str:
    return tensor_sha256(value.detach().cpu().contiguous())


def _code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__), Path(__file__).with_name("signed_int8_screen3_fia.py"),
        Path(__file__).with_name("signed_int8_observation.py"),
        Path(__file__).with_name("signed_int8_randomized.py"),
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


def public_input_hashes(observation: Path) -> dict[str, str]:
    observation = Path(observation).resolve()
    manifest = observation / "manifest.json"
    _must_file(manifest)
    value = read_json(manifest)
    paths = [manifest] + [observation / item["file"] for item in value["layers"].values()]
    paths.append(CHECKPOINT)
    return {path.resolve().as_posix(): file_sha256(path) for path in paths}


def register(public_root: Path = PUBLIC_ROOT, private_root: Path = PRIVATE_ROOT) -> dict[str, Any]:
    public_root, private_root = Path(public_root).resolve(), Path(private_root).resolve()
    if public_root.exists() or private_root.exists() or public_root in private_root.parents or private_root in public_root.parents:
        raise FileExistsError("screen3 output roots must be new and disjoint")
    _must_file(CHECKPOINT)
    if file_sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise Screen3Error("registered VGG16 checkpoint changed")
    public_root.mkdir(parents=True, exist_ok=False)
    private_root.mkdir(parents=True, exist_ok=False)
    registration = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "REGISTERED",
        "method": "INT8_randomized8_individual_FIA5000_public_top3_to_total25000",
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
        "codec": "signed_symmetric_int8_qplus127_nearest_even_cpu_fp32_v1",
        "source_count": SOURCE_COUNT, "source_shape": [SOURCE_COUNT, FEATURE_DIM],
        "checkpoint": CHECKPOINT.as_posix(), "checkpoint_sha256": CHECKPOINT_SHA,
        "public_input_hashes": {}, "code_hashes": _code_hashes(),
        "protected_source_hashes": _protected_hashes(), "upstream_source_hashes": upstream_source_hashes(),
        "private_data_policy": {"attack_reads_public_observation_only": True, "load_after_public_gate": True,
                                 "sender_true_updates_consumed": True, "attack_true_updates_consumed": False,
                                 "evaluator_consumed_field": "x only", "labels_consumed": False},
        "actual_image_cell_feasibility": "NOT_EVALUATED", "scientific_quality": "NOT_EVALUATED",
        "budget_seconds": MAX_SECONDS, "cuda_allocated_cap_bytes": MAX_CUDA_ALLOCATED_BYTES,
        "registered_before_execution": True,
    }
    write_json(public_root / "registration.json", registration)
    write_json(private_root / "task_boundary.json", {
        "task_id": TASK_ID, "true_updates_consumed_by_sender": True,
        "attack_true_updates_consumed": False, "labels_consumed": False, "images_consumed": False,
    })
    return registration


def _save_source(folder: Path, payload: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=False)
    torch.save(payload, folder / "sources.pt")
    write_json(folder / "result.json", metadata)


def _align_preserve_norm(center: torch.Tensor, member: torch.Tensor, member_index: int) -> tuple[torch.Tensor, dict[str, Any]]:
    from scipy.optimize import linear_sum_assignment
    if center.shape != member.shape or center.dtype != torch.float32 or member.dtype != torch.float32:
        raise Screen3Error("source alignment shape/dtype mismatch")
    center64, member64 = center.double(), member.double()
    cn = torch.linalg.vector_norm(center64, dim=1)
    mn = torch.linalg.vector_norm(member64, dim=1)
    if bool((cn <= 0).any()) or bool((mn <= 0).any()) or not bool(torch.isfinite(cn).all() & torch.isfinite(mn).all()):
        raise Screen3Error("source alignment requires finite nonzero rows")
    cosine = (center64 / cn[:, None]) @ (member64 / mn[:, None]).T
    rows, cols = linear_sum_assignment((-cosine.abs()).numpy())
    aligned = torch.empty_like(member64)
    signs: list[int] = []
    scores: list[float] = []
    norm_ratios: list[float] = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        dot = torch.dot(center64[row], member64[col])
        sign = 1 if float(dot) >= 0 else -1
        aligned[row] = sign * member64[col]
        signs.append(sign)
        scores.append(float(cosine[row, col].abs()))
        norm_ratios.append(float(mn[col] / cn[row]))
    aligned = aligned.to(torch.float32).contiguous()
    if not bool(torch.isfinite(aligned).all()):
        raise Screen3Error("aligned source is nonfinite")
    return aligned, {
        "member_index": member_index, "permutation_member_columns": cols.tolist(),
        "signs": signs, "matched_abs_cosines": scores,
        "mean_abs_cosine": sum(scores) / len(scores), "min_abs_cosine": min(scores),
        "max_abs_cosine": max(scores), "native_to_reference_norm_ratios": norm_ratios,
        "row_norm_rescaling": False, "aligned_sha256": _tensor_hash(aligned),
    }


def _source_payload(path: Path, index: int) -> tuple[torch.Tensor, str, dict[str, Any]]:
    payload = torch.load(path / "sources.pt", map_location="cpu", weights_only=True)
    result = read_json(path / "result.json")
    if index == -1:
        if set(payload) != {"S_hat", "target_z"}:
            raise Screen3Error("center source schema mismatch")
        source, target = payload["S_hat"], payload["target_z"]
    else:
        if set(payload) != {"S_hat_native", "S_hat_aligned", "target_z"}:
            raise Screen3Error("member source schema mismatch")
        source, target = payload["S_hat_aligned"], payload["target_z"]
    if (source.dtype != torch.float32 or target.dtype != torch.float32
            or tuple(source.shape) != (SOURCE_COUNT, FEATURE_DIM)
            or tuple(target.shape) != (SOURCE_COUNT, FEATURE_DIM)
            or not bool(torch.isfinite(source).all()) or not torch.equal(target, source.abs())):
        raise Screen3Error("invalid source target binding")
    if result.get("target_z_sha256") != _tensor_hash(target):
        raise Screen3Error("source target hash mismatch")
    if index == -1:
        if result.get("S_hat_sha256") != _tensor_hash(source):
            raise Screen3Error("center source hash mismatch")
    else:
        if (result.get("S_hat_native_sha256") != _tensor_hash(payload["S_hat_native"])
                or result.get("S_hat_aligned_sha256") != _tensor_hash(source)):
            raise Screen3Error("member native/aligned source hash mismatch")
    return target.contiguous(), file_sha256(path / "sources.pt"), result


def separate(public_observation: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    public_observation, output = Path(public_observation).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing separation output: {output}")
    configure_runtime(42, threads=1)
    manifest, entries = read_bundle(public_observation)
    q, scale = entries["classifier.0.weight"]
    X0 = (q.to(torch.float32) * scale).contiguous()
    if tuple(X0.shape) != (OBSERVATION_ROWS, FEATURE_DIM):
        raise Screen3Error("public FC0 matrix shape mismatch")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "registration.json", {
        "task_id": TASK_ID, "status": "REGISTERED", "stage": "SEPARATE",
        "public_manifest_sha256": file_sha256(public_observation / "manifest.json"),
        "config": SEPARATION_CONFIG.as_dict(), "separation_calls_registered": 9,
        "member_count": MEMBER_COUNT, "member_seeds": MEMBER_SEEDS, "private_data_used": False,
    })
    records: list[dict[str, Any]] = []
    center_dir = output / "center"
    center = separate_observation(X0, SOURCE_COUNT, config=SEPARATION_CONFIG, device="cpu")
    center_source = center.S_hat.detach().cpu().contiguous()
    center_target = center_source.abs().contiguous()
    _save_source(center_dir, {"S_hat": center_source, "target_z": center_target}, {
        "task_id": TASK_ID, "status": "COMPLETED", "index": "center", "separation_call": 1,
        "input_sha256": _tensor_hash(X0), "S_hat_sha256": _tensor_hash(center_source),
        "target_z_sha256": _tensor_hash(center_target), "fastica": center.metadata["fastica"],
        "refinement": center.metadata["refinement"], "whitening": center.metadata["whitening"],
        "final_losses": center.refinement.final_losses, "private_data_used": False,
    })
    records.append({"index": "center", "separation_call": 1, "input_sha256": _tensor_hash(X0),
                    "S_hat_sha256": _tensor_hash(center_source), "target_z_sha256": _tensor_hash(center_target),
                    "fastica": center.metadata["fastica"]})
    member_sources: list[tuple[int, torch.Tensor, Path, dict[str, Any]]] = []
    for member_index, seed in enumerate(MEMBER_SEEDS):
        sampled, sample_meta = sample_matrix(q, scale, seed=seed, chunk_rows=32)
        member_dir = output / f"member_{member_index:02d}"
        separation = separate_observation(sampled, SOURCE_COUNT, config=SEPARATION_CONFIG, device="cpu")
        native = separation.S_hat.detach().cpu().contiguous()
        aligned, alignment = _align_preserve_norm(center_source, native, member_index)
        target = aligned.abs().contiguous()
        _save_source(member_dir, {"S_hat_native": native, "S_hat_aligned": aligned, "target_z": target}, {
            "task_id": TASK_ID, "status": "COMPLETED", "index": member_index,
            "separation_call": member_index + 2, "input_sha256": sample_meta["tensor_sha256"],
            "S_hat_native_sha256": _tensor_hash(native), "S_hat_aligned_sha256": _tensor_hash(aligned),
            "target_z_sha256": _tensor_hash(target), "sample": sample_meta, "alignment": alignment,
            "fastica": separation.metadata["fastica"], "refinement": separation.metadata["refinement"],
            "whitening": separation.metadata["whitening"], "final_losses": separation.refinement.final_losses,
            "private_data_used": False,
        })
        member_sources.append((member_index, aligned, member_dir, alignment))
        records.append({"index": member_index, "separation_call": member_index + 2,
                        "input_sha256": sample_meta["tensor_sha256"], "sample": sample_meta,
                        "S_hat_native_sha256": _tensor_hash(native), "S_hat_aligned_sha256": _tensor_hash(aligned),
                        "target_z_sha256": _tensor_hash(target), "alignment": alignment,
                        "fastica": separation.metadata["fastica"]})
        del sampled, separation, native, aligned, target
    result = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "COMPLETED", "stage": "SEPARATE",
        "public_manifest_sha256": file_sha256(public_observation / "manifest.json"),
        "config": SEPARATION_CONFIG.as_dict(), "source_shape": [SOURCE_COUNT, FEATURE_DIM],
        "separation_calls": 9, "center_is_reference_not_member": True, "member_count": MEMBER_COUNT,
        "member_seeds": MEMBER_SEEDS, "records": records,
        "alignment": {"reference": "center signed S_hat", "correction": "permutation_sign_only",
                       "row_norm_rescaling": False, "feature_average": False,
                       "members": [item[3] for item in member_sources]},
        "runtime": {"device": "cpu", "dtype": "float32", "threads": 1},
        "protected_source_hashes": _protected_hashes(), "upstream_source_hashes": upstream_source_hashes(),
        "private_data_used": False, "labels_consumed": False, "true_updates_consumed": False,
        "actual_image_cell_feasibility": "NOT_EVALUATED", "scientific_quality": "NOT_EVALUATED",
        "timing": {"wall_seconds": time.perf_counter() - started},
    }
    write_json(output / "result.json", result)
    write_json(output / "registration.json", {**read_json(output / "registration.json"),
                                               "status": "COMPLETED", "result_sha256": file_sha256(output / "result.json")})
    return result


def _source_entry(separation_root: Path, index: int) -> tuple[torch.Tensor, str, dict[str, Any]]:
    return _source_payload(separation_root / ("center" if index == -1 else f"member_{index:02d}"), index)


def _progress(event: dict[str, Any]) -> None:
    print(json.dumps({"stage": "FIA", **event}, allow_nan=False), flush=True)


def freeze_public_selection(screen_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate all eight public endpoint scores and freeze deterministic top3."""
    if len(screen_results) != MEMBER_COUNT:
        raise Screen3Error("selection requires exactly eight member scores")
    indices = [int(item.get("member_index", -1)) for item in screen_results]
    if sorted(indices) != list(range(MEMBER_COUNT)):
        raise Screen3Error("selection member indices are not exactly 0..7")
    for item in screen_results:
        score = item.get("score", {})
        if (not isinstance(score.get("score"), (int, float))
                or not torch.isfinite(torch.tensor(float(score["score"])))
                or score.get("raw_state_unchanged") is not True):
            raise Screen3Error(f"invalid/nonfinite public score for member {item.get('member_index')}")
    ranking = sorted(screen_results, key=lambda item: (float(item["score"]["score"]), int(item["member_index"])))
    return {
        "task_id": TASK_ID, "status": "FROZEN_PUBLIC_SELECTION", "completed_step": PREFIX_STEPS,
        "ranking": [{"rank": rank + 1, "member_index": item["member_index"],
                      "score": item["score"]["score"],
                      "feature_cosine_distance_mean": item["score"].get("feature_cosine_distance_mean"),
                      "tv_normalized": item["score"].get("tv_normalized"),
                      "checkpoint_sha256": item["result"]["checkpoint_sha256"],
                      "candidate_sha256": item["result"]["candidate_sha256"],
                      "target_z_sha256": item["target_z_sha256"],
                      "source_file_sha256": item["source_file_sha256"]}
                     for rank, item in enumerate(ranking)],
        "selected_member_indices": [item["member_index"] for item in ranking[:3]],
        "tie_break": "member_index_ascending",
        "selection_rule": "post5000_feature_cosine_distance_plus_0.1_normalized_TV",
        "private_metrics_used": False,
    }


def run_public(public_root: Path = PUBLIC_ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    public_root = Path(public_root).resolve()
    registration = read_json(public_root / "registration.json")
    if registration.get("task_id") != TASK_ID or registration.get("status") != "REGISTERED":
        raise Screen3Error("screen3 registration is not ready")
    observation = public_root / "observation"
    separation_root = public_root / "separation"
    if not separation_root.exists():
        separate(observation, separation_root)
    separation = read_json(separation_root / "result.json")
    if separation.get("separation_calls") != 9 or separation.get("status") != "COMPLETED":
        raise Screen3Error("nine-call separation is incomplete")
    center_target, center_source_file_sha, center_record = _source_entry(separation_root, -1)
    center_out = public_root / "fia_center_25000"
    center_result = run_trajectory(center_target, output=center_out, source_sha256=center_source_file_sha,
                                   arm="center_25000", end_steps=TOTAL_STEPS, progress=_progress)
    screen_results: list[dict[str, Any]] = []
    for index in range(MEMBER_COUNT):
        target, source_file_sha, record = _source_entry(separation_root, index)
        branch = public_root / f"fia_member_{index:02d}_5000"
        result = run_trajectory(target, output=branch, source_sha256=source_file_sha,
                                arm=f"member_{index:02d}_5000", end_steps=PREFIX_STEPS,
                                progress=_progress)
        score = result.get("score")
        if not score or not score.get("raw_state_unchanged") or not all(
                torch.isfinite(torch.tensor(score["feature_cosine_distances"]))):
            raise Screen3Error(f"invalid public screen score for member {index}")
        screen_results.append({"member_index": index, "branch": branch.as_posix(),
                               "result": result, "score": score, "source_file_sha256": source_file_sha,
                               "target_z_sha256": tensor_sha256(target)})
    selection = freeze_public_selection(screen_results)
    selected = selection["selected_member_indices"]
    write_json(public_root / "selection.json", selection)
    continuation_results = []
    for index in selected:
        target, source_file_sha, _ = _source_entry(separation_root, index)
        parent = public_root / f"fia_member_{index:02d}_5000/checkpoint_5000.pt"
        branch = public_root / f"fia_member_{index:02d}_25000"
        continuation_results.append(run_trajectory(
            target, output=branch, source_sha256=source_file_sha, arm=f"member_{index:02d}_25000",
            end_steps=TOTAL_STEPS, parent_checkpoint=parent, parent_role="MEMBER_PREFIX_5000",
            progress=_progress))
    public_hashes = public_input_hashes(observation)
    registration.update({
        "status": "PUBLIC_COMPLETED", "public_input_hashes": public_hashes,
        "separation_result_sha256": file_sha256(separation_root / "result.json"),
        "selection_sha256": file_sha256(public_root / "selection.json"),
        "public_frozen": True, "private_loaded": False,
        "public_stage_wall_seconds": time.perf_counter() - started,
        "scientific_quality": "NOT_EVALUATED",
    })
    write_json(public_root / "registration.json", registration)
    result = {
        "task_id": TASK_ID, "status": "PUBLIC_COMPLETED", "center": center_result,
        "screen_results": screen_results, "selection": selection,
        "continuations": continuation_results, "total_fia_updates": TOTAL_FIA_UPDATES,
        "public_input_hashes": public_hashes, "wall_seconds": time.perf_counter() - started,
        "private_data_used": False, "actual_image_cell_feasibility": "NOT_EVALUATED",
        "scientific_quality": "NOT_EVALUATED",
    }
    write_json(public_root / "public_execution.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register_parser = sub.add_parser("register")
    register_parser.add_argument("--public-root", type=Path, default=PUBLIC_ROOT)
    register_parser.add_argument("--private-root", type=Path, default=PRIVATE_ROOT)
    separate_parser = sub.add_parser("separate")
    separate_parser.add_argument("--observation", type=Path, required=True)
    separate_parser.add_argument("--output", type=Path, required=True)
    attack_parser = sub.add_parser("attack")
    attack_parser.add_argument("--public-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            result = register(args.public_root, args.private_root)
        elif args.command == "separate":
            result = separate(args.observation, args.output)
        else:
            result = run_public(args.public_root)
    except Exception as error:
        print(f"{TASK_ID}: {type(error).__name__}: {error}", flush=True)
        return 1
    print(json.dumps({"task_id": TASK_ID, "status": result["status"]}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
