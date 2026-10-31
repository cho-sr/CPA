"""Resumable same-scope recovery for the registered 7 -> 6 -> 5 -> 4 sweep."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


TASK_ID = "CPA_SIGNED_BITS7654_RECOVERY_20260918"
PARENT_TASK_ID = "CPA_SIGNED_BITS7654_SCREEN3_20260918"
EXECUTOR_THREAD = "01a0a799-f1ce-7b31-8c46-b4ddef40aeab"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RESULT_ROOT = REPO_ROOT / ".ai/execution-results" / TASK_ID
PARENT_RESULT_ROOT = REPO_ROOT / ".ai/execution-results" / PARENT_TASK_ID
PYTHON = Path(r"C:\miniconda\envs\DLG\python.exe")
BITS = (7, 6, 5, 4)
BIT7_BASE_USED = 11021.525099699968
AGGREGATE_BASE_USED = 11024.35411100008
BIT7_REMAINING = 1578.474900300032
AGGREGATE_REMAINING = 39375.64588899992
FRESH_BIT_CAP = 12600.0
CUDA_CAP_BYTES = 12 * 1024**3
PARENT_SUMMARY_SHA = "5c43c9c5fc58af6280cd2f7b4563e71c267e87699a1c4b9527edc0b53031882e"
BIT7_EXECUTION_SHA = "863e7ca84b5e805b22b45713b3b13b37b48f7558b70aa1d668a4412d24c75308"
BIT7_SELECTION_SHA = "e3e296851bc8316a851671a66643823f3edd6e863ffd76407e946d018efcda66"
BIT7_REGISTRATION_SHA = "cc4ccb524cf6b04bd7c71bd6d54531529e4d40263b0cea550ee2007a854eca8c"
REGISTERED_HASHES = {
    6: "88d745533e9a609e544b5662c5e381a59b863bebc4ba57c7f9423aae307e2e0b",
    5: "345d50f7e8c02dad6c2f489baa6f88033ad0682cb3489f07ea7df08504b269cd",
    4: "c53b6c0b93e220ded607bb76b989941132d78bc327d846f5ef264b9dd388209b",
}


class RecoveryError(RuntimeError):
    pass


def public_root(bits: int) -> Path:
    return REPO_ROOT / f"quantization_aware_cpa/results/signed_int{bits}_randomized8_screen3_fia_B16_T10_20260918_sweep"


def private_root(bits: int) -> Path:
    return REPO_ROOT / f"quantization_aware_cpa/results_private/signed_int{bits}_randomized8_screen3_fia_B16_T10_20260918_sweep"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def _bit_state(status: str = "QUEUED") -> dict[str, Any]:
    return {"status": status, "stages": {}, "stage_records": [], "optimizer_steps": 0,
            "private_data_loaded": False}


def new_state() -> dict[str, Any]:
    return {
        "task_id": TASK_ID, "parent_task_id": PARENT_TASK_ID, "status": "RUNNING",
        "claimed_by": EXECUTOR_THREAD, "assigned_role": "CODEX_EXECUTOR", "bits": list(BITS),
        "bit_status": {"7": "PENDING_SAVED_EVALUATION", "6": "QUEUED", "5": "QUEUED", "4": "QUEUED"},
        "bit_state": {"7": _bit_state("PENDING_SAVED_EVALUATION"), "6": _bit_state(),
                      "5": _bit_state(), "4": _bit_state()},
        "private_data_loaded": False, "scientific_quality": "NOT_EVALUATED",
        "actual_image_cell_feasibility": "NOT_EVALUATED", "new_optimizer_steps_bit7": 0,
        "parent_artifacts_preserved": True, "created_at_epoch": time.time(),
        "accounting": {"bit7_base_used_seconds": BIT7_BASE_USED,
                        "aggregate_base_used_seconds": AGGREGATE_BASE_USED,
                        "bit7_recovery_seconds": 0.0, "fresh_bit_recovery_seconds": 0.0,
                        "aggregate_recovery_seconds": 0.0,
                        "bit7_remaining_at_start_seconds": BIT7_REMAINING,
                        "aggregate_remaining_at_start_seconds": AGGREGATE_REMAINING},
        "bit_results": {}, "stages": [],
    }


def save_state(state: dict[str, Any]) -> None:
    state["updated_at_epoch"] = time.time()
    write_json(RESULT_ROOT / "execution_summary.json", state)
    write_json(RESULT_ROOT / "schedule_state.json", {
        "task_id": TASK_ID, "parent_task_id": PARENT_TASK_ID, "status": state["status"],
        "claimed_by": state["claimed_by"], "bit_status": state["bit_status"],
        "private_data_loaded": state["private_data_loaded"],
        "scientific_quality": state["scientific_quality"],
        "actual_image_cell_feasibility": state["actual_image_cell_feasibility"],
        "accounting": state["accounting"], "updated_at_epoch": state["updated_at_epoch"],
    })


def load_state() -> dict[str, Any]:
    path = RESULT_ROOT / "execution_summary.json"
    if not path.is_file():
        state = new_state()
        save_state(state)
        return state
    state = read_json(path)
    if state.get("task_id") != TASK_ID or state.get("claimed_by") != EXECUTOR_THREAD:
        raise RecoveryError("recovery state ownership mismatch")
    if state.get("status") == "COMPLETED":
        raise RecoveryError("recovery task is already finalized; refusing duplicate execution")
    if state.get("status") != "RUNNING":
        state["resumed_from_status"] = state.get("status")
        state["status"] = "RUNNING"
    return state


def verify_parent_evidence() -> None:
    summary_path = PARENT_RESULT_ROOT / "execution_summary.json"
    if sha256(summary_path) != PARENT_SUMMARY_SHA:
        raise RecoveryError("parent failed wrapper evidence changed")
    summary = read_json(summary_path)
    if summary.get("task_id") != PARENT_TASK_ID or summary.get("status") != "FAILED":
        raise RecoveryError("parent wrapper evidence is not the registered failed run")
    bit7_public = next((row for row in summary.get("stages", [])
                        if row.get("bit") == 7 and row.get("name") == "public"), None)
    bit7_gate = next((row for row in summary.get("stages", [])
                      if row.get("bit") == 7 and row.get("name") == "public_gate"), None)
    if not bit7_public or bit7_public.get("exit_code") != 0 or not bit7_gate or bit7_gate.get("exit_code") != 1:
        raise RecoveryError("parent failure evidence does not match the verified gate-only failure")
    root = public_root(7)
    if sha256(root / "public_execution.json") != BIT7_EXECUTION_SHA:
        raise RecoveryError("saved bit7 public execution changed")
    if sha256(root / "selection.json") != BIT7_SELECTION_SHA:
        raise RecoveryError("saved bit7 selection changed")
    if sha256(root / "registration.json") != BIT7_REGISTRATION_SHA:
        raise RecoveryError("saved bit7 registration changed")
    selection = read_json(root / "selection.json")
    if (selection.get("status") != "FROZEN_PUBLIC_SELECTION"
            or selection.get("selected_member_indices") != [0, 5, 3]
            or selection.get("private_metrics_used") is not False):
        raise RecoveryError("saved bit7 public selection is not the registered [0,5,3]")
    execution = read_json(root / "public_execution.json")
    if (execution.get("status") != "PUBLIC_COMPLETED"
            or execution.get("total_fia_updates") != 125000
            or execution.get("private_data_used") is not False):
        raise RecoveryError("saved bit7 public execution is incomplete or private")


def verify_registration(bits: int, *, allow_public_completed: bool = False) -> dict[str, Any]:
    root = public_root(bits)
    registration_path = root / "registration.json"
    if not registration_path.is_file():
        raise RecoveryError(f"bit {bits} registration is missing")
    registration = read_json(registration_path)
    from quantization_aware_cpa.qacpa import signed_bits_screen3_runner as runner
    status = registration.get("status")
    if bits == 7:
        if sha256(registration_path) != BIT7_REGISTRATION_SHA:
            raise RecoveryError("bit7 registration bytes changed")
    elif status == "REGISTERED":
        if sha256(registration_path) != REGISTERED_HASHES[bits]:
            raise RecoveryError(f"bit {bits} pre-run registration bytes changed")
    elif not (allow_public_completed and status == "PUBLIC_COMPLETED"):
        raise RecoveryError(f"bit {bits} registration status is not allowed: {status}")
    if (registration.get("task_id") != runner.bit_task_id(bits)
            or registration.get("bits") != bits
            or registration.get("code_hashes") != runner._code_hashes()
            or registration.get("protected_source_hashes") != runner._protected_hashes()):
        raise RecoveryError(f"bit {bits} registered source/config gate failed")
    if bits == 7 and status != "PUBLIC_COMPLETED":
        raise RecoveryError("bit7 saved registration is not PUBLIC_COMPLETED")
    if bits != 7 and status not in ({"REGISTERED", "PUBLIC_COMPLETED"} if allow_public_completed else {"REGISTERED"}):
        raise RecoveryError(f"bit {bits} is not an unused REGISTERED root")
    return registration


def verify_saved_bit7() -> None:
    verify_parent_evidence()
    verify_registration(7)
    from quantization_aware_cpa.qacpa import signed_bits_screen3_runner as runner
    root = public_root(7)
    separation = read_json(root / "separation" / "result.json")
    if separation.get("status") != "COMPLETED" or separation.get("separation_calls") != 9:
        raise RecoveryError("bit7 separation evidence is incomplete")
    for index in (-1, 0, 1, 2, 3, 4, 5, 6, 7):
        runner._source_entry(root / "separation", index)


def verify_fresh_root(bits: int) -> None:
    registration = verify_registration(bits, allow_public_completed=False)
    if (registration.get("status") != "REGISTERED"
            or registration.get("public_input_hashes") != {}
            or registration.get("public_frozen") not in (None, False)):
        raise RecoveryError(f"bit {bits} registration has unexpected pre-run state")
    if (public_root(bits) / "public_execution.json").exists() or (public_root(bits) / "observation").exists():
        raise RecoveryError(f"bit {bits} has unexpected executed public artifacts before resume")


def verify_saved_public_completed(bits: int) -> dict[str, Any]:
    registration = verify_registration(bits, allow_public_completed=True)
    execution_path = public_root(bits) / "public_execution.json"
    if not execution_path.is_file():
        raise RecoveryError(f"bit {bits} has PUBLIC_COMPLETED registration without public execution")
    execution = read_json(execution_path)
    if (execution.get("status") != "PUBLIC_COMPLETED"
            or execution.get("total_fia_updates") != 125000
            or execution.get("private_data_used") is not False
            or execution.get("scientific_quality") != "NOT_EVALUATED"):
        raise RecoveryError(f"bit {bits} saved public execution is not the registered 125000-update result")
    return registration


def accounted_seconds(state: dict[str, Any]) -> float:
    return sum(float(record.get("wall_seconds", 0.0)) for record in state.get("stages", []))


def remaining_seconds(state: dict[str, Any], bits: int) -> float:
    used = accounted_seconds(state)
    if bits == 7:
        bit_used = sum(float(record.get("wall_seconds", 0.0))
                       for record in state.get("stages", []) if record.get("bit") == 7)
        return BIT7_REMAINING - bit_used
    bit_used = sum(float(record.get("wall_seconds", 0.0))
                   for record in state.get("stages", []) if record.get("bit") == bits)
    return FRESH_BIT_CAP - bit_used


def _next_attempt(state: dict[str, Any], bits: int, name: str) -> int:
    return 1 + sum(1 for record in state.get("stages", [])
                   if record.get("bit") == bits and record.get("name") == name)


def run_stage(state: dict[str, Any], bits: int, name: str, args: list[str]) -> dict[str, Any]:
    remaining = min(remaining_seconds(state, bits), AGGREGATE_REMAINING - accounted_seconds(state))
    if remaining <= 0:
        raise RecoveryError(f"bit {bits} budget exhausted before {name}")
    attempt = _next_attempt(state, bits, name)
    stdout_path = RESULT_ROOT / f"b{bits}_{name}_attempt{attempt}.stdout.log"
    stderr_path = RESULT_ROOT / f"b{bits}_{name}_attempt{attempt}.stderr.log"
    record = {"bit": bits, "name": name, "attempt": attempt, "args": args,
              "status": "RUNNING", "started_at_epoch": time.time(),
              "stdout": stdout_path.as_posix(), "stderr": stderr_path.as_posix(),
              "timeout_seconds": remaining}
    state["stages"].append(record)
    save_state(state)
    started = time.perf_counter()
    timed_out = False
    exit_code = 1
    with stdout_path.open("w", encoding="utf-8", newline="") as stdout, \
         stderr_path.open("w", encoding="utf-8", newline="") as stderr:
        process = subprocess.Popen([str(PYTHON), "-B", *args], cwd=REPO_ROOT,
                                   stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        record["pid"] = process.pid
        save_state(state)
        try:
            exit_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
            exit_code = 124
    wall = time.perf_counter() - started
    record.update({"exit_code": exit_code, "timed_out": timed_out, "wall_seconds": wall,
                   "ended_at_epoch": time.time(), "status": "COMPLETED" if exit_code == 0 else "FAILED"})
    save_state(state)
    if exit_code != 0:
        raise RecoveryError(f"bit {bits} {name} failed with exit code {exit_code}")
    return record


def mark_stage(state: dict[str, Any], bits: int, name: str, status: str, **extra: Any) -> None:
    state["bit_state"][str(bits)]["stages"][name] = status
    state["bit_state"][str(bits)].update(extra)
    save_state(state)


def gate_output(bits: int) -> Path:
    return RESULT_ROOT / f"b{bits}_gate.json"


def evaluation_output(bits: int) -> Path:
    root = private_root(bits)
    base = root / "evaluation_recovery_20260918"
    if not base.exists():
        return base
    index = 2
    while (root / f"evaluation_recovery_20260918_retry{index}").exists():
        index += 1
    return root / f"evaluation_recovery_20260918_retry{index}"


def comparison_for(bits: int, output: Path) -> dict[str, Any]:
    comparison = read_json(output / "comparison.json")
    if (comparison.get("task_id") != f"CPA_SIGNED_BITS7654_SCREEN3_20260918_B{bits}"
            or comparison.get("evaluation_status") != "COMPLETED"
            or comparison.get("scientific_quality") != "EVALUATED"
            or comparison.get("private_x_only") is not True
            or comparison.get("labels_consumed") is not False):
        raise RecoveryError(f"bit {bits} evaluation result failed its strict contract")
    return comparison


def _set_public_completed(state: dict[str, Any], bits: int) -> None:
    state["bit_status"][str(bits)] = "PUBLIC_COMPLETED_PENDING_EVALUATION"
    save_state(state)


def refuse_partial_optimizer_stage(state: dict[str, Any], bits: int, name: str) -> None:
    status = state["bit_state"][str(bits)]["stages"].get(name)
    if status in {"RUNNING", "FAILED"}:
        raise RecoveryError(f"bit {bits} {name} stage is {status}; refusing to restart a partial optimizer stage")


def run_bit7(state: dict[str, Any]) -> None:
    verify_saved_bit7()
    state["bit_state"]["7"]["optimizer_steps"] = 0
    if state["bit_state"]["7"]["stages"].get("saved_public") != "COMPLETED":
        mark_stage(state, 7, "saved_public", "COMPLETED", optimizer_steps=0,
                   evidence="authenticated saved bit7 public_execution/selection/candidates")
    _set_public_completed(state, 7)
    if state["bit_state"]["7"]["stages"].get("gate") != "COMPLETED":
        output = gate_output(7)
        run_stage(state, 7, "gate", [str(RESULT_ROOT / "gate_adapter.py"), "--bits", "7",
                                      "--public-root", str(public_root(7)), "--output", str(output)])
        gate = read_json(output)
        if gate.get("task_id") != "CPA_SIGNED_BITS7654_SCREEN3_20260918_B7":
            raise RecoveryError("bit7 gate adapter task binding failed")
        mark_stage(state, 7, "gate", "COMPLETED", candidate_count=gate["candidate_count"],
                   private_loaded=False, optimizer_steps=0)
    else:
        gate = read_json(gate_output(7))
        if (gate.get("task_id") != "CPA_SIGNED_BITS7654_SCREEN3_20260918_B7"
                or gate.get("status") != "PUBLIC_GATE_PASSED"):
            raise RecoveryError("saved bit7 gate adapter evidence is invalid")
    output = evaluation_output(7)
    stage = state["bit_state"]["7"]["stages"]
    if stage.get("evaluation") != "COMPLETED":
        run_stage(state, 7, "evaluation", ["-m", "quantization_aware_cpa.qacpa.signed_bits_screen3_evaluation",
                                             "--bits", "7", "--public-root", str(public_root(7)),
                                             "--private-truth", str(REPO_ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"),
                                             "--output", str(output)])
        comparison = comparison_for(7, output)
        mark_stage(state, 7, "evaluation", "COMPLETED", evaluation_output=output.as_posix(),
                   candidate_count=comparison["public_gate"]["candidate_count"], optimizer_steps=0)
    else:
        comparison = comparison_for(7, Path(state["bit_state"]["7"]["evaluation_output"]))
    state["private_data_loaded"] = True
    state["bit_state"]["7"]["private_data_loaded"] = True
    state["bit_status"]["7"] = "COMPLETED"
    state["bit_results"]["7"] = summarize_comparison(
        comparison, output if stage.get("evaluation") != "COMPLETED"
        else Path(state["bit_state"]["7"]["evaluation_output"]))
    state["accounting"]["bit7_recovery_seconds"] = sum(float(x.get("wall_seconds", 0.0)) for x in state["stages"] if x.get("bit") == 7)
    state["accounting"]["aggregate_recovery_seconds"] = accounted_seconds(state)
    save_state(state)


def run_fresh_bit(state: dict[str, Any], bits: int) -> None:
    state["bit_status"][str(bits)] = "RUNNING"
    save_state(state)
    bit_state = state["bit_state"][str(bits)]
    if bit_state["stages"].get("public") != "COMPLETED":
        if (read_json(public_root(bits) / "registration.json").get("status") == "PUBLIC_COMPLETED"
                and (public_root(bits) / "public_execution.json").is_file()):
            registration = verify_saved_public_completed(bits)
            mark_stage(state, bits, "public", "COMPLETED", optimizer_steps=125000,
                       private_loaded=False, recovered_from_saved_public=True,
                       registered_hash_before_run=REGISTERED_HASHES[bits])
        else:
            refuse_partial_optimizer_stage(state, bits, "public")
            verify_fresh_root(bits)
            run_stage(state, bits, "public", ["-m", "quantization_aware_cpa.qacpa.signed_bits_screen3_runner", "run",
                                               "--bits", str(bits), "--public-root", str(public_root(bits)),
                                               "--private-root", str(private_root(bits))])
            registration = verify_registration(bits, allow_public_completed=True)
            if registration.get("status") != "PUBLIC_COMPLETED":
                raise RecoveryError(f"bit {bits} public runner exited successfully without PUBLIC_COMPLETED")
            mark_stage(state, bits, "public", "COMPLETED", optimizer_steps=125000, private_loaded=False,
                       registered_hash_before_run=REGISTERED_HASHES[bits])
    else:
        registration = verify_registration(bits, allow_public_completed=True)
        if registration.get("status") != "PUBLIC_COMPLETED":
            raise RecoveryError(f"bit {bits} saved public stage is not authenticated")
    _set_public_completed(state, bits)
    if bit_state["stages"].get("gate") != "COMPLETED":
        output = gate_output(bits)
        run_stage(state, bits, "gate", [str(RESULT_ROOT / "gate_adapter.py"), "--bits", str(bits),
                                         "--public-root", str(public_root(bits)), "--output", str(output)])
        gate = read_json(output)
        if gate.get("task_id") != f"CPA_SIGNED_BITS7654_SCREEN3_20260918_B{bits}":
            raise RecoveryError(f"bit {bits} gate adapter task binding failed")
        mark_stage(state, bits, "gate", "COMPLETED", candidate_count=gate["candidate_count"], private_loaded=False)
    output = evaluation_output(bits)
    if bit_state["stages"].get("evaluation") != "COMPLETED":
        run_stage(state, bits, "evaluation", ["-m", "quantization_aware_cpa.qacpa.signed_bits_screen3_evaluation",
                                               "--bits", str(bits), "--public-root", str(public_root(bits)),
                                               "--private-truth", str(REPO_ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"),
                                               "--output", str(output)])
        comparison = comparison_for(bits, output)
        mark_stage(state, bits, "evaluation", "COMPLETED", evaluation_output=output.as_posix(),
                   candidate_count=comparison["public_gate"]["candidate_count"])
    else:
        comparison = comparison_for(bits, Path(bit_state["evaluation_output"]))
        output = Path(bit_state["evaluation_output"])
    state["private_data_loaded"] = True
    bit_state["private_data_loaded"] = True
    bit_state["optimizer_steps"] = 125000
    state["bit_status"][str(bits)] = "COMPLETED"
    state["bit_results"][str(bits)] = summarize_comparison(comparison, output)
    state["accounting"]["fresh_bit_recovery_seconds"] = sum(float(x.get("wall_seconds", 0.0)) for x in state["stages"] if x.get("bit") != 7)
    state["accounting"]["aggregate_recovery_seconds"] = accounted_seconds(state)
    save_state(state)


def summarize_comparison(comparison: dict[str, Any], output: Path) -> dict[str, Any]:
    return {"evaluation_status": comparison["evaluation_status"], "scientific_quality": comparison["scientific_quality"],
            "candidate_count": comparison["public_gate"]["candidate_count"], "mean_metrics": comparison["mean_metrics"],
            "primary_contrast_rank1_minus_center25000": comparison["primary_contrast_rank1_minus_center25000"],
            "evaluation_output": output.as_posix(), "figure_artifacts": comparison.get("figure_artifacts", [])}


def write_report(state: dict[str, Any]) -> None:
    lines = [f"# {TASK_ID}", "", f"Status: **{state['status']}**.", "",
             "This recovery preserved the original failed wrapper evidence and used the saved bit7 computation.",
             "Bit7 used zero new optimizer steps; bits 6, 5 and 4 were processed serially with fresh per-bit observations.",
             "", "| Bit | Status | Evaluation | Candidate sets |", "| ---: | --- | --- | ---: |"]
    for bits in BITS:
        item = state.get("bit_results", {}).get(str(bits), {})
        lines.append(f"| {bits} | {state['bit_status'].get(str(bits))} | {item.get('evaluation_status', 'n/a')} | {item.get('candidate_count', 'n/a')} |")
    lines += ["", "Every private evaluator ran after its public gate and consumed x only; labels and true updates were not consumed by the attack.",
              "Actual-image cell feasibility: NOT_EVALUATED.",
              f"Recovery additional accounted stage time: {accounted_seconds(state):.6f} seconds.",
              f"Bit7 original remaining allowance: {BIT7_REMAINING:.6f} seconds; aggregate original remaining allowance: {AGGREGATE_REMAINING:.6f} seconds."]
    (RESULT_ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def finalize(state: dict[str, Any]) -> None:
    state["status"] = "COMPLETED"
    state["scientific_quality"] = "EVALUATED"
    state["finished_at_epoch"] = time.time()
    state["private_data_loaded"] = True
    save_state(state)
    write_json(RESULT_ROOT / "integrity.json", {
        "task_id": TASK_ID, "status": state["status"], "parent_task_id": PARENT_TASK_ID,
        "parent_failed_evidence_preserved": True, "parent_artifacts_preserved": True,
        "bit_status": state["bit_status"], "bit7_new_optimizer_steps": 0,
        "fresh_bits_optimizer_steps": {str(bits): state["bit_state"][str(bits)]["optimizer_steps"] for bits in (6, 5, 4)},
        "public_gate_before_each_private_evaluation": True, "private_x_only": True,
        "labels_consumed": False, "attack_true_updates_consumed": False,
        "actual_image_cell_feasibility": "NOT_EVALUATED", "cuda_allocated_cap_bytes": CUDA_CAP_BYTES,
        "accounting": state["accounting"],
    })
    write_json(RESULT_ROOT / "finalization_summary.json", {
        "task_id": TASK_ID, "status": state["status"], "scientific_quality": state["scientific_quality"],
        "bits": list(BITS), "bit_status": state["bit_status"], "bit_results": state["bit_results"],
        "parent_original_failure_preserved": True, "actual_image_cell_feasibility": "NOT_EVALUATED",
        "accounting": state["accounting"],
    })
    write_report(state)


def fail(state: dict[str, Any], error: BaseException, failed_bits: int | None) -> None:
    state["status"] = "PARTIAL" if any(value == "COMPLETED" for value in state["bit_status"].values()) else "FAILED"
    state["error_type"] = type(error).__name__
    state["error"] = str(error)
    state["finished_at_epoch"] = time.time()
    if failed_bits is not None:
        state["bit_status"][str(failed_bits)] = "FAILED"
        order = list(BITS)
        for bit in order[order.index(failed_bits) + 1:]:
            if state["bit_status"].get(str(bit)) not in {"COMPLETED", "FAILED"}:
                state["bit_status"][str(bit)] = "BLOCKED_BY_PREVIOUS_FAILURE"
    save_state(state)
    write_json(RESULT_ROOT / "integrity.json", {
        "task_id": TASK_ID, "status": state["status"], "error_type": type(error).__name__,
        "error": str(error), "bit_status": state["bit_status"], "parent_artifacts_preserved": True,
        "actual_image_cell_feasibility": "NOT_EVALUATED", "accounting": state["accounting"],
    })
    write_report(state)


def main() -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    state = load_state()
    failed_bits: int | None = 7
    try:
        verify_parent_evidence()
        save_state(state)
        run_bit7(state)
        for bits in (6, 5, 4):
            failed_bits = bits
            if state["bit_status"].get(str(bits)) == "COMPLETED":
                continue
            run_fresh_bit(state, bits)
        failed_bits = None
        finalize(state)
    except Exception as error:
        fail(state, error, failed_bits)
        return 1
    print(json.dumps({"task_id": TASK_ID, "status": state["status"], "bit_status": state["bit_status"]},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
