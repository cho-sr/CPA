"""Bounded one-shot continuation and offline evaluation wrapper."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


TASK_ID = "CPA_SIGNED_INT8_SCREEN3_REPAIR_CONTINUE_20260918"
REPO_ROOT = Path(r"C:\Users\조상제\Desktop\CPA")
PYTHON = Path(r"C:\miniconda\envs\DLG\python.exe")
RESULT_ROOT = REPO_ROOT / ".ai/execution-results" / TASK_ID
PUBLIC_ROOT = REPO_ROOT / "quantization_aware_cpa/results/signed_int8_randomized8_screen3_repair_B16_T10_20260918"
PRIVATE_ROOT = REPO_ROOT / "quantization_aware_cpa/results_private/signed_int8_screen3_repair_B16_T10_20260918"
PRIVATE_TRUTH = REPO_ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"
CAP_SECONDS = 5399.606889200048


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8", newline="\n")
    tmp.replace(path)


def _validate_ready() -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    from quantization_aware_cpa.qacpa import signed_int8_screen3_repair_runner as repair

    registration = repair.read(PUBLIC_ROOT / "registration.json")
    if registration.get("task_id") != TASK_ID or registration.get("status") != "REGISTERED":
        raise RuntimeError("repair registration is not REGISTERED at launch")
    if registration.get("repair_code_hashes") != repair.repair_code_hashes():
        raise RuntimeError("repair code hash migration is stale at launch")
    inherited = repair.authenticate_inherited()
    if repair.sha(PUBLIC_ROOT / "inherited_manifest.json") != registration.get("inherited_manifest_sha256"):
        raise RuntimeError("inherited manifest is not authenticated at launch")
    return {
        "status": "READY_VALIDATED", "task_id": TASK_ID,
        "parent_registration_sha256": repair.PARENT_REGISTRATION_SHA256,
        "parent_selection_sha256": repair.PARENT_SELECTION_SHA256,
        "parent_fia_source_sha256": repair.PARENT_FIA_SHA256,
        "parent_runner_source_sha256": repair.PARENT_RUNNER_SHA256,
        "inherited_manifest_sha256": registration["inherited_manifest_sha256"],
        "inherited_source_count": len(inherited["source_records"]),
        "inherited_candidate_count": len(inherited["inherited_candidates"]),
        "frozen_selection": [1, 7, 5],
        "private_loaded": False,
        "remaining_budget_seconds": CAP_SECONDS,
    }


def _run_stage(name: str, args: list[str], started: float, summary: dict[str, Any]) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    remaining = CAP_SECONDS - elapsed
    if remaining <= 0:
        raise TimeoutError("remaining original real-work budget is exhausted before " + name)
    stdout_path = RESULT_ROOT / f"{name}.stdout.log"
    stderr_path = RESULT_ROOT / f"{name}.stderr.log"
    stage_started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, \
         stderr_path.open("w", encoding="utf-8", newline="\n") as stderr:
        process = subprocess.Popen(
            [str(PYTHON), *args], cwd=REPO_ROOT, stdout=stdout, stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            exit_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            exit_code = 124
            timed_out = True
        else:
            timed_out = False
    result = {
        "name": name, "pid": process.pid, "exit_code": exit_code, "timed_out": timed_out,
        "started_at_monotonic": stage_started, "wall_seconds": time.perf_counter() - stage_started,
        "stdout": stdout_path.as_posix(), "stderr": stderr_path.as_posix(),
        "remaining_budget_after_seconds": CAP_SECONDS - (time.perf_counter() - started),
    }
    summary.setdefault("stages", []).append(result)
    _write(RESULT_ROOT / "execution_summary.json", summary)
    if exit_code != 0:
        raise RuntimeError(f"{name} failed with exit code {exit_code}")
    return result


def main() -> int:
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "task_id": TASK_ID, "status": "RUNNING", "wrapper_pid": __import__("os").getpid(),
        "started_at_epoch": time.time(), "cap_seconds": CAP_SECONDS,
        "parent_artifacts_preserved": True, "private_loaded": False,
        "stages": [], "scientific_quality": "NOT_EVALUATED",
        "actual_image_cell_feasibility": "NOT_EVALUATED",
        "remaining_optimizer_steps": {"member01_export": 0, "member07": 20000, "member05": 20000},
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write(RESULT_ROOT / "execution_summary.json", summary)
    try:
        summary["ready_validation"] = _validate_ready()
        _write(RESULT_ROOT / "ready_validation.json", summary["ready_validation"])
        _write(RESULT_ROOT / "execution_summary.json", summary)

        _run_stage(
            "continuation", [
                "-m", "quantization_aware_cpa.qacpa.signed_int8_screen3_repair_runner", "continue",
                "--public-root", str(PUBLIC_ROOT), "--private-root", str(PRIVATE_ROOT),
            ], started, summary,
        )
        _run_stage(
            "private_evaluation", [
                "-m", "quantization_aware_cpa.qacpa.signed_int8_screen3_repair_evaluation",
                "--public-root", str(PUBLIC_ROOT), "--private-truth", str(PRIVATE_TRUTH),
                "--output", str(PRIVATE_ROOT / "evaluation"),
            ], started, summary,
        )
        summary.update({
            "status": "COMPLETED", "scientific_quality": "EVALUATED", "private_loaded": True,
            "private_policy": "x_only_after_public_gate", "cumulative_fia_updates": 125000,
            "candidate_count": 13, "selection": [1, 7, 5],
            "finished_at_epoch": time.time(), "wall_seconds": time.perf_counter() - started,
        })
        _write(RESULT_ROOT / "integrity.json", {
            "status": "COMPLETED", "public_root": PUBLIC_ROOT.as_posix(),
            "private_evaluation_root": (PRIVATE_ROOT / "evaluation").as_posix(),
            "public_gate": "PASSED", "candidate_count": 13,
            "private_data_policy": "x_only_after_public_gate",
            "sender_true_updates_consumed": True, "attack_true_updates_consumed": False,
            "actual_image_cell_feasibility": "NOT_EVALUATED",
        })
        (RESULT_ROOT / "report.md").write_text(
            f"# {TASK_ID}\n\nStatus: **COMPLETED**.\n\n"
            "The saved member01 endpoint was exported with zero optimizer steps; only member07 and member05 "
            "continued by 20,000 steps from authenticated 5,000-step parent checkpoints. "
            "The 13-set public gate passed before canonical-runtime private x-only evaluation.\n\n"
            "Frozen public selection: [1, 7, 5]. Actual-image cell feasibility: NOT_EVALUATED.\n",
            encoding="utf-8", newline="\n",
        )
    except Exception as error:
        summary.update({
            "status": "FAILED", "error_type": type(error).__name__, "error": str(error),
            "finished_at_epoch": time.time(), "wall_seconds": time.perf_counter() - started,
        })
        _write(RESULT_ROOT / "integrity.json", {
            "status": "FAILED", "public_root": PUBLIC_ROOT.as_posix(),
            "private_loaded": summary.get("private_loaded", False),
            "completed_stages": [stage["name"] for stage in summary["stages"] if stage["exit_code"] == 0],
            "error_type": type(error).__name__, "error": str(error),
            "parent_artifacts_preserved": True,
        })
        (RESULT_ROOT / "report.md").write_text(
            f"# {TASK_ID}\n\nStatus: **FAILED**.\n\n{type(error).__name__}: {error}\n\n"
            "The wrapper stopped without retrying or restarting any parent stage.\n",
            encoding="utf-8", newline="\n",
        )
        _write(RESULT_ROOT / "execution_summary.json", summary)
        return 1
    _write(RESULT_ROOT / "execution_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
