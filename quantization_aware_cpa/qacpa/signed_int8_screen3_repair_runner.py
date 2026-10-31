"""Strict continuation/export orchestration for the failed INT8 screen3 run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from .deterministic_hybrid_setup import configure_runtime
from .signed_int8_screen3_fia import FIA_CONFIG, PREFIX_STEPS, TOTAL_STEPS, export_completed_checkpoint, run_trajectory, tensor_sha256
from . import signed_int8_screen3_runner as original


TASK_ID = "CPA_SIGNED_INT8_SCREEN3_REPAIR_CONTINUE_20260918"
PARENT_TASK_ID = "CPA_SIGNED_INT8_RANDOMIZED8_SCREEN3_FIA_20260917"
REPO_ROOT = Path(__file__).resolve().parents[2]
PARENT_PUBLIC_ROOT = REPO_ROOT / "quantization_aware_cpa/results/signed_int8_randomized8_screen3_fia_B16_T10_20260917"
PARENT_PRIVATE_ROOT = REPO_ROOT / "quantization_aware_cpa/results_private/signed_int8_randomized8_screen3_fia_B16_T10_20260917"
PARENT_SOURCE_REVISION_ROOT = REPO_ROOT / ".ai/execution-results/CPA_SIGNED_INT8_SCREEN3_REPAIR_CONTINUE_20260918/parent_source_revision"
PUBLIC_ROOT = REPO_ROOT / "quantization_aware_cpa/results/signed_int8_randomized8_screen3_repair_B16_T10_20260918"
PRIVATE_ROOT = REPO_ROOT / "quantization_aware_cpa/results_private/signed_int8_screen3_repair_B16_T10_20260918"
SOURCE_COUNT = 16
FEATURE_DIM = 25088
MEMBER_COUNT = 8
MAX_CUDA_ALLOCATED_BYTES = 12 * 1024**3
PARENT_REGISTRATION_SHA256 = "d974ad6075bdf3dbb5607ec914d992957a26d537a83483e98f70c370242ea3fb"
PARENT_SELECTION_SHA256 = "9c6a24bd73d0d9fb4ea752316b3a77e05779b6ae4152c090c229ca2b1d2a8683"
PARENT_FIA_SHA256 = "ab36aef2990fc7c0c11f4a0abcd1bc162e6a5896c91e7bbb3e71176c55849a95"
PARENT_RUNNER_SHA256 = "2b718a7203ca47e2b09773bdb344b7ee8b074ca5429d37a7cf58dced1f6b23e2"
MEMBER01_PREFIX_SHA256 = "8932029334060d05b9f50a75b8a4a3dd816d8726a1b41bbcbdd8bd470f033cb5"
MEMBER01_FINAL_SHA256 = "e48fee1e339d2f47c25cd308f3f78cad0ff44e2b6e6d78ea72d8669190f14353"
MEMBER07_PREFIX_SHA256 = "5e8f6ead3dd7a3e2f486fb6adb587c55411ebdaf2f28b40d1d41d8fa21c154da"
MEMBER05_PREFIX_SHA256 = "7bdccb705c2edb911ce33f283e68167cb41c160262e336d667f3fdd7f1ce3908"
CENTER5_SHA256 = "ae7942cfce971091a9cb0ca357ebde184fe9e9f03977cd4c7a2a16273e5be1b6"
CENTER25_SHA256 = "2e6f075d2097d0c663648564359aae729895b86ebf090eae107a2931f0c9c9fe"


class RepairError(RuntimeError):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RepairError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def _ensure_parent_source_snapshot() -> dict[str, Path]:
    """Materialize the authenticated pre-repair source revision beside this task.

    The repaired FIA module necessarily has a new hash.  The old parent bytes are
    reconstructed only from the exact narrow diff and are accepted only when the
    resulting bytes match the immutable hash recorded by the parent run.
    """
    current_fia = REPO_ROOT / "quantization_aware_cpa/qacpa/signed_int8_screen3_fia.py"
    current_text = current_fia.read_text(encoding="utf-8")
    # The exporter is appended after the parent revision.  Remove that
    # task-local addition before reversing the single child-writer diff.
    if "\ndef _validate_completed_checkpoint" not in current_text:
        raise RepairError("completed-checkpoint exporter is missing from repaired FIA source")
    current_text = current_text.split("\ndef _validate_completed_checkpoint", 1)[0].rstrip("\r\n") + "\n"
    new_block = (
        "        # A fresh 25000-step trajectory owns and must write its local 5000\n"
        "        # prefix.  A resumed child already has that authenticated parent and\n"
        "        # must link it instead of inventing a second prefix.\n"
        "        if parent_checkpoint is None and prefix_checkpoint_path is None:\n"
        "            raise StagedFIAError(\"25000 trajectory did not produce its 5000 checkpoint\")"
    )
    old_block = (
        "        if prefix_checkpoint_path is None:\n"
        "            raise StagedFIAError(\"25000 trajectory did not produce its 5000 checkpoint\")"
    )
    if current_text.count(new_block) != 1:
        raise RepairError("cannot establish the exact pre-repair FIA source diff")
    old_bytes = current_text.replace(new_block, old_block, 1).encode("utf-8")
    if hashlib.sha256(old_bytes).hexdigest() != PARENT_FIA_SHA256:
        raise RepairError("reconstructed parent FIA source does not match registered hash")
    PARENT_SOURCE_REVISION_ROOT.mkdir(parents=True, exist_ok=True)
    fia_snapshot = PARENT_SOURCE_REVISION_ROOT / "signed_int8_screen3_fia.py"
    if fia_snapshot.exists() and fia_snapshot.read_bytes() != old_bytes:
        raise RepairError("parent FIA source snapshot collision")
    fia_snapshot.write_bytes(old_bytes)

    runner_source = REPO_ROOT / "quantization_aware_cpa/qacpa/signed_int8_screen3_runner.py"
    runner_snapshot = PARENT_SOURCE_REVISION_ROOT / "signed_int8_screen3_runner.py"
    runner_bytes = runner_source.read_bytes()
    if hashlib.sha256(runner_bytes).hexdigest() != PARENT_RUNNER_SHA256:
        raise RepairError("parent runner source revision changed before repair migration")
    if runner_snapshot.exists() and runner_snapshot.read_bytes() != runner_bytes:
        raise RepairError("parent runner source snapshot collision")
    runner_snapshot.write_bytes(runner_bytes)
    return {"fia": fia_snapshot, "runner": runner_snapshot}


def _candidate(path: Path, result_path: Path, step: int, *, prefix_center: bool = False) -> dict[str, Any]:
    result = read(result_path)
    candidate = path
    if not candidate.is_file() or candidate.is_symlink():
        raise RepairError(f"missing candidate: {candidate}")
    expected_hash = result.get("prefix_candidate_sha256") if prefix_center else result.get("candidate_sha256")
    if expected_hash != sha(candidate):
        raise RepairError(f"candidate/result hash mismatch: {candidate}")
    expected_iterations = TOTAL_STEPS if prefix_center else step
    if result.get("status") != "COMPLETED" or result.get("completed_iterations") != expected_iterations:
        raise RepairError(f"candidate result step/status mismatch: {result_path}")
    payload = torch.load(candidate, map_location="cpu", weights_only=True)
    if set(payload) != {"pixels"} or payload["pixels"].dtype != torch.float32 or tuple(payload["pixels"].shape) != (16, 3, 224, 224):
        raise RepairError(f"candidate schema mismatch: {candidate}")
    if not bool(torch.isfinite(payload["pixels"]).all()) or bool((payload["pixels"] < 0).any()) or bool((payload["pixels"] > 1).any()):
        raise RepairError(f"candidate range mismatch: {candidate}")
    return {"path": candidate.as_posix(), "sha256": sha(candidate), "result": result_path.as_posix(),
            "result_sha256": sha(result_path), "step": step,
            "target_z_sha256": result.get("target_z_sha256"), "source_sha256": result.get("source_sha256"),
            "parent_checkpoint_sha256": result.get("parent_checkpoint_sha256")}


def _source(index: int) -> tuple[torch.Tensor, str, dict[str, Any], Path]:
    folder = PARENT_PUBLIC_ROOT / ("separation/center" if index == -1 else f"separation/member_{index:02d}")
    payload = torch.load(folder / "sources.pt", map_location="cpu", weights_only=True)
    result = read(folder / "result.json")
    if index == -1:
        if set(payload) != {"S_hat", "target_z"}:
            raise RepairError("inherited center source schema mismatch")
        target = payload["target_z"]
        source = payload["S_hat"]
        if result.get("S_hat_sha256") != tensor_sha256(source):
            raise RepairError("inherited center source hash mismatch")
    else:
        if set(payload) != {"S_hat_native", "S_hat_aligned", "target_z"}:
            raise RepairError(f"inherited member source schema mismatch: {index}")
        source = payload["S_hat_aligned"]
        target = payload["target_z"]
        if result.get("S_hat_native_sha256") != tensor_sha256(payload["S_hat_native"]):
            raise RepairError(f"inherited member native hash mismatch: {index}")
        if result.get("S_hat_aligned_sha256") != tensor_sha256(source):
            raise RepairError(f"inherited member aligned hash mismatch: {index}")
    if source.dtype != torch.float32 or target.dtype != torch.float32 or tuple(target.shape) != (16, 25088) or not torch.equal(target, source.abs()):
        raise RepairError(f"inherited source target mismatch: {index}")
    if result.get("target_z_sha256") != tensor_sha256(target):
        raise RepairError(f"inherited target hash mismatch: {index}")
    return target.contiguous(), sha(folder / "sources.pt"), result, folder / "sources.pt"


def _all_inherited_candidates() -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    center = PARENT_PUBLIC_ROOT / "fia_center_25000"
    candidates["center_5000"] = _candidate(center / "candidate_5000.pt", center / "result.json", 5000, prefix_center=True)
    for index in range(8):
        branch = PARENT_PUBLIC_ROOT / f"fia_member_{index:02d}_5000"
        candidates[f"member_{index:02d}_5000"] = _candidate(branch / "candidate.pt", branch / "result.json", 5000)
    candidates["center_25000"] = _candidate(center / "candidate.pt", center / "result.json", 25000)
    return candidates


def authenticate_inherited() -> dict[str, Any]:
    parent_sources = _ensure_parent_source_snapshot()
    registration_path = PARENT_PUBLIC_ROOT / "registration.json"
    selection_path = PARENT_PUBLIC_ROOT / "selection.json"
    if sha(registration_path) != PARENT_REGISTRATION_SHA256 or sha(selection_path) != PARENT_SELECTION_SHA256:
        raise RepairError("parent registration/selection bytes changed")
    registration = read(registration_path)
    selection = read(selection_path)
    if registration.get("task_id") != PARENT_TASK_ID or selection.get("selected_member_indices") != [1, 7, 5]:
        raise RepairError("parent task/selection binding mismatch")
    if sha(parent_sources["fia"]) != PARENT_FIA_SHA256 or sha(parent_sources["runner"]) != PARENT_RUNNER_SHA256:
        raise RepairError("parent source revision changed before repair migration")
    if not (PARENT_PUBLIC_ROOT / "observation/manifest.json").is_file():
        raise RepairError("parent public observation is missing")
    public_inputs = registration.get("public_input_hashes", {})
    inherited_files: list[dict[str, Any]] = []
    for raw_path, expected in public_inputs.items():
        path = Path(raw_path)
        if not path.is_file() or sha(path) != expected:
            raise RepairError(f"parent public input hash mismatch: {path}")
        inherited_files.append({"path": path.as_posix(), "sha256": expected, "role": "parent_public_input"})
    separation = read(PARENT_PUBLIC_ROOT / "separation/result.json")
    if separation.get("status") != "COMPLETED" or separation.get("separation_calls") != 9:
        raise RepairError("parent separation result is incomplete")
    inherited_files += [
        {"path": (PARENT_PUBLIC_ROOT / "registration.json").as_posix(), "sha256": PARENT_REGISTRATION_SHA256, "role": "parent_registration"},
        {"path": (PARENT_PUBLIC_ROOT / "selection.json").as_posix(), "sha256": PARENT_SELECTION_SHA256, "role": "frozen_public_selection"},
        {"path": (PARENT_PUBLIC_ROOT / "separation/result.json").as_posix(), "sha256": sha(PARENT_PUBLIC_ROOT / "separation/result.json"), "role": "nine_call_separation"},
        {"path": parent_sources["fia"].as_posix(), "sha256": PARENT_FIA_SHA256, "role": "parent_fia_source_revision"},
        {"path": parent_sources["runner"].as_posix(), "sha256": PARENT_RUNNER_SHA256, "role": "parent_runner_source_revision"},
    ]
    source_records = {}
    for index in [-1, *range(8)]:
        _, source_file_sha, result, source_path = _source(index)
        source_records["center" if index == -1 else f"member_{index:02d}"] = {
            "source_file": source_path.as_posix(), "source_file_sha256": source_file_sha,
            "target_z_sha256": result["target_z_sha256"], "source_result_sha256": sha(source_path.parent / "result.json"),
        }
        inherited_files.append({"path": source_path.as_posix(), "sha256": source_file_sha, "role": "inherited_source"})
        inherited_files.append({"path": (source_path.parent / "result.json").as_posix(), "sha256": sha(source_path.parent / "result.json"), "role": "inherited_source_result"})
    candidates = _all_inherited_candidates()
    critical = {
        "member01_prefix": PARENT_PUBLIC_ROOT / "fia_member_01_5000/checkpoint_5000.pt",
        "member01_final": PARENT_PUBLIC_ROOT / "fia_member_01_25000/checkpoint_25000.pt",
        "member07_prefix": PARENT_PUBLIC_ROOT / "fia_member_07_5000/checkpoint_5000.pt",
        "member05_prefix": PARENT_PUBLIC_ROOT / "fia_member_05_5000/checkpoint_5000.pt",
        "center_5000_checkpoint": PARENT_PUBLIC_ROOT / "fia_center_25000/checkpoint_5000.pt",
        "center_25000_checkpoint": PARENT_PUBLIC_ROOT / "fia_center_25000/checkpoint_25000.pt",
    }
    expected = {"member01_prefix": MEMBER01_PREFIX_SHA256, "member01_final": MEMBER01_FINAL_SHA256,
                "member07_prefix": MEMBER07_PREFIX_SHA256, "member05_prefix": MEMBER05_PREFIX_SHA256,
                "center_5000_checkpoint": CENTER5_SHA256, "center_25000_checkpoint": CENTER25_SHA256}
    for role, path in critical.items():
        if sha(path) != expected[role]:
            raise RepairError(f"critical inherited checkpoint mismatch: {role}")
        inherited_files.append({"path": path.as_posix(), "sha256": expected[role], "role": role})
    child = torch.load(critical["member01_final"], map_location="cpu", weights_only=False)
    if child.get("completed_iterations") != TOTAL_STEPS or child.get("parent_checkpoint_sha256") != MEMBER01_PREFIX_SHA256:
        raise RepairError("member01 completed child chain mismatch")
    return {
        "parent_task_id": PARENT_TASK_ID, "parent_registration_sha256": PARENT_REGISTRATION_SHA256,
        "parent_selection_sha256": PARENT_SELECTION_SHA256, "parent_code_hashes": registration.get("code_hashes", {}),
        "parent_fia_source_sha256": PARENT_FIA_SHA256, "parent_runner_source_sha256": PARENT_RUNNER_SHA256,
        "parent_source_revision": {"fia": parent_sources["fia"].as_posix(), "runner": parent_sources["runner"].as_posix()},
        "frozen_selection": selection, "source_records": source_records,
        "inherited_candidates": candidates, "inherited_files": inherited_files,
        "member01_final_checkpoint": critical["member01_final"].as_posix(),
        "member01_prefix_checkpoint": critical["member01_prefix"].as_posix(),
        "member07_prefix_checkpoint": critical["member07_prefix"].as_posix(),
        "member05_prefix_checkpoint": critical["member05_prefix"].as_posix(),
        "center_control": {"checkpoint_5000": sha(critical["center_5000_checkpoint"]),
                           "checkpoint_25000": sha(critical["center_25000_checkpoint"])},
    }


def repair_code_hashes() -> dict[str, str]:
    paths = [Path(__file__), Path(__file__).with_name("signed_int8_screen3_fia.py"),
             Path(__file__).with_name("signed_int8_screen3_repair_evaluation.py"),
             Path(__file__).with_name("signed_int8_screen3_evaluation.py"),
             Path(__file__).with_name("signed_int8_observation.py"),
             Path(__file__).with_name("signed_int8_randomized.py")]
    return {path.resolve().as_posix(): sha(path) for path in paths}


def register(public_root: Path = PUBLIC_ROOT, private_root: Path = PRIVATE_ROOT) -> dict[str, Any]:
    public_root, private_root = Path(public_root).resolve(), Path(private_root).resolve()
    if public_root.exists() or private_root.exists() or public_root in private_root.parents or private_root in public_root.parents:
        raise FileExistsError("repair output roots must be new and disjoint")
    inherited = authenticate_inherited()
    public_root.mkdir(parents=True, exist_ok=False)
    private_root.mkdir(parents=True, exist_ok=False)
    registration = {
        "task_id": TASK_ID, "parent_task_id": PARENT_TASK_ID, "status": "REGISTERED",
        "public_output_root": public_root.as_posix(), "private_output_root": private_root.as_posix(),
        "method": "INT8_screen3_repair_export_member01_and_continue_member07_member05",
        "inherited_parent_root": PARENT_PUBLIC_ROOT.as_posix(), "inherited_private_root": PARENT_PRIVATE_ROOT.as_posix(),
        "inherited_registration_sha256": PARENT_REGISTRATION_SHA256,
        "inherited_selection_sha256": PARENT_SELECTION_SHA256,
        "old_source_revision": {"fia": PARENT_FIA_SHA256, "runner": PARENT_RUNNER_SHA256,
                                 "snapshot_root": PARENT_SOURCE_REVISION_ROOT.as_posix()},
        "repair_code_hashes": repair_code_hashes(),
        "permitted_source_diff": {"signed_int8_screen3_fia.py": {"old_sha256": PARENT_FIA_SHA256,
            "new_sha256": sha(Path(__file__).with_name("signed_int8_screen3_fia.py")),
            "change": "child final checkpoint branch and zero-step completed-checkpoint exporter"},
            "signed_int8_screen3_repair_runner.py": {"old_sha256": None, "new": "new isolated adapter"}},
        "source_migration_policy": "old artifacts authenticated under old registration/source revision; repaired outputs under new revision",
        "fixed_config": FIA_CONFIG, "selection": {"member_indices": [1, 7, 5], "reranking": False},
        "remaining_new_optimizer_steps": 40000, "cumulative_fia_updates": {"before": 85000, "after": 125000},
        "cuda_allocated_cap_bytes": MAX_CUDA_ALLOCATED_BYTES, "private_data_loaded": False,
        "actual_image_cell_feasibility": "NOT_EVALUATED", "scientific_quality": "NOT_EVALUATED",
        "inherited_manifest_sha256_pending": True,
    }
    write(public_root / "registration.json", registration)
    write(public_root / "inherited_manifest.json", inherited)
    write(private_root / "task_boundary.json", {"task_id": TASK_ID, "sender_true_updates_consumed": True,
                                                  "repair_attack_true_updates_consumed": False,
                                                  "private_evaluation_consumed": "pending x only", "labels_consumed": False})
    registration["inherited_manifest_sha256"] = sha(public_root / "inherited_manifest.json")
    write(public_root / "registration.json", registration)
    return registration


def _target(index: int, inherited: dict[str, Any]) -> tuple[torch.Tensor, str]:
    record = inherited["source_records"]["center" if index == -1 else f"member_{index:02d}"]
    payload = torch.load(record["source_file"], map_location="cpu", weights_only=True)
    target = payload["target_z"]
    if tensor_sha256(target) != record["target_z_sha256"]:
        raise RepairError(f"target changed: {index}")
    return target.contiguous(), record["source_file_sha256"]


def _final_candidates(public_root: Path, inherited: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = dict(inherited["inherited_candidates"])
    entries["rank1_member_01_25000"] = _candidate(public_root / "fia_member_01_25000/candidate.pt",
                                                   public_root / "fia_member_01_25000/result.json", TOTAL_STEPS)
    entries["rank2_member_07_25000"] = _candidate(public_root / "fia_member_07_25000/candidate.pt",
                                                   public_root / "fia_member_07_25000/result.json", TOTAL_STEPS)
    entries["rank3_member_05_25000"] = _candidate(public_root / "fia_member_05_25000/candidate.pt",
                                                   public_root / "fia_member_05_25000/result.json", TOTAL_STEPS)
    return entries


def public_gate(public_root: Path) -> dict[str, Any]:
    public_root = Path(public_root).resolve()
    registration = read(public_root / "registration.json")
    if registration.get("task_id") != TASK_ID or registration.get("status") != "PUBLIC_COMPLETED":
        raise RepairError("repair registration is not PUBLIC_COMPLETED")
    inherited = authenticate_inherited()
    if sha(public_root / "inherited_manifest.json") != registration.get("inherited_manifest_sha256"):
        raise RepairError("repair inherited manifest hash changed")
    candidates = _final_candidates(public_root, inherited)
    expected_names = {"center_5000", *[f"member_{i:02d}_5000" for i in range(8)], "center_25000",
                      "rank1_member_01_25000", "rank2_member_07_25000", "rank3_member_05_25000"}
    if set(candidates) != expected_names:
        raise RepairError("repair candidate set is not exactly 13 frozen sets")
    if registration.get("repair_code_hashes") != repair_code_hashes():
        raise RepairError("repair source hash migration changed")
    return {"status": "PUBLIC_GATE_PASSED", "private_loaded": False,
            "inherited_registration_sha256": PARENT_REGISTRATION_SHA256,
            "inherited_selection_sha256": PARENT_SELECTION_SHA256,
            "candidate_count": len(candidates), "candidates": candidates,
            "repair_code_hashes": registration["repair_code_hashes"],
            "cumulative_fia_updates": 125000}


def run_repair(public_root: Path = PUBLIC_ROOT, private_root: Path = PRIVATE_ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    public_root, private_root = Path(public_root).resolve(), Path(private_root).resolve()
    registration = read(public_root / "registration.json")
    if registration.get("status") != "REGISTERED":
        raise RepairError("repair registration is not ready")
    if registration.get("repair_code_hashes") != repair_code_hashes():
        raise RepairError("repair source hash migration changed before continuation")
    inherited = read(public_root / "inherited_manifest.json")
    if sha(public_root / "inherited_manifest.json") != registration.get("inherited_manifest_sha256"):
        raise RepairError("inherited manifest hash changed before continuation")
    fresh = authenticate_inherited()
    if fresh["parent_registration_sha256"] != inherited["parent_registration_sha256"]:
        raise RepairError("inherited manifest no longer authenticates")
    # Member01 is recovered by export only; no optimizer step is permitted.
    member01_target, member01_source_sha = _target(1, inherited)
    export_completed_checkpoint(
        Path(inherited["member01_final_checkpoint"]),
        output=public_root / "fia_member_01_25000", target_z=member01_target,
        source_sha256=member01_source_sha, arm="member_01_25000",
        parent_checkpoint=Path(inherited["member01_prefix_checkpoint"]), device="cuda",
    )
    continuation_results = []
    for index, parent_key in ((7, "member07_prefix_checkpoint"), (5, "member05_prefix_checkpoint")):
        target, source_sha = _target(index, inherited)
        continuation_results.append(run_trajectory(
            target, output=public_root / f"fia_member_{index:02d}_25000", source_sha256=source_sha,
            arm=f"member_{index:02d}_25000", end_steps=TOTAL_STEPS,
            parent_checkpoint=Path(inherited[parent_key]), parent_role="MEMBER_PREFIX_5000",
            progress=lambda event: print(json.dumps({"stage": "REPAIR_FIA", **event}, allow_nan=False), flush=True),
        ))
    candidates = _final_candidates(public_root, inherited)
    write(public_root / "repair_public_manifest.json", {"task_id": TASK_ID, "candidate_count": 13,
        "candidates": candidates, "inherited_selection": [1, 7, 5],
        "optimizer_steps_executed_by_recovery": {"member01_export": 0, "member07": 20000, "member05": 20000},
        "cumulative_fia_updates": 125000})
    registration.update({"status": "PUBLIC_COMPLETED", "public_gate_pending": True,
                         "repair_public_manifest_sha256": sha(public_root / "repair_public_manifest.json"),
                         "cumulative_fia_updates": 125000,
                         "incremental_optimizer_steps": 40000,
                         "wall_seconds": time.perf_counter() - started})
    write(public_root / "registration.json", registration)
    gate = public_gate(public_root)
    registration.update({"status": "PUBLIC_COMPLETED", "public_gate_passed": True,
                         "gate_candidate_count": gate["candidate_count"], "private_loaded": False})
    write(public_root / "registration.json", registration)
    result = {"task_id": TASK_ID, "status": "PUBLIC_COMPLETED", "public_gate": gate,
              "member01_export": read(public_root / "fia_member_01_25000/result.json"),
              "continuations": continuation_results, "cumulative_fia_updates": 125000,
              "private_loaded": False, "wall_seconds": time.perf_counter() - started,
              "actual_image_cell_feasibility": "NOT_EVALUATED", "scientific_quality": "NOT_EVALUATED"}
    write(public_root / "repair_public_execution.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_register = sub.add_parser("register")
    p_register.add_argument("--public-root", type=Path, default=PUBLIC_ROOT)
    p_register.add_argument("--private-root", type=Path, default=PRIVATE_ROOT)
    p_run = sub.add_parser("continue")
    p_run.add_argument("--public-root", type=Path, required=True)
    p_run.add_argument("--private-root", type=Path, required=True)
    p_gate = sub.add_parser("gate")
    p_gate.add_argument("--public-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            result = register(args.public_root, args.private_root)
        elif args.command == "continue":
            result = run_repair(args.public_root, args.private_root)
        else:
            result = public_gate(args.public_root)
    except Exception as error:
        print(f"{TASK_ID}: {type(error).__name__}: {error}", flush=True)
        return 1
    print(json.dumps({"task_id": TASK_ID, "status": result["status"]}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
