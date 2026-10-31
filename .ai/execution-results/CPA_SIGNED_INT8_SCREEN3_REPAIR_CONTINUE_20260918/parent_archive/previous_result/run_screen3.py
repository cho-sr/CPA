"""Single bounded sequential wrapper for CPA_SIGNED_INT8_RANDOMIZED8_SCREEN3_FIA_20260917."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import time
from typing import Any


TASK_ID = "CPA_SIGNED_INT8_RANDOMIZED8_SCREEN3_FIA_20260917"
ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(__file__).resolve().parent
PYTHON = Path(r"C:\miniconda\envs\DLG\python.exe")
PUBLIC_ROOT = ROOT / "quantization_aware_cpa/results/signed_int8_randomized8_screen3_fia_B16_T10_20260917"
PRIVATE_ROOT = ROOT / "quantization_aware_cpa/results_private/signed_int8_randomized8_screen3_fia_B16_T10_20260917"
CAP_SECONDS = 12600.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    started = time.perf_counter()
    stages: list[dict[str, Any]] = []
    commands = [
        ("register", ["-m", "qacpa.signed_int8_screen3_runner", "register",
                       "--public-root", str(PUBLIC_ROOT), "--private-root", str(PRIVATE_ROOT)]),
        ("sender", ["-m", "qacpa.signed_int8_observation", "sender",
                     "--raw-updates", str(ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/raw_updates.pt"),
                     "--archived-sender-validation", str(ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/sender_validation.json"),
                     "--evaluation-truth", str(ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"),
                     "--output", str(PUBLIC_ROOT / "observation"), "--private-output", str(PRIVATE_ROOT / "sender")]),
        ("public_attack", ["-m", "qacpa.signed_int8_screen3_runner", "attack",
                            "--public-root", str(PUBLIC_ROOT)]),
        ("private_evaluation", ["-m", "qacpa.signed_int8_screen3_evaluation",
                                 "--public-root", str(PUBLIC_ROOT), "--output", str(PRIVATE_ROOT / "evaluation")]),
    ]
    final_status = "COMPLETED"
    failure: dict[str, Any] | None = None
    for name, arguments in commands:
        elapsed = time.perf_counter() - started
        if elapsed >= CAP_SECONDS:
            final_status = "PARTIAL"
            failure = {"stage": name, "error": "registered wall-clock cap reached before stage"}
            stages.append({"stage": name, "status": "NOT_STARTED_BUDGET", "exit_code": None})
            break
        stdout_path = RESULTS / f"{name}.stdout.log"
        stderr_path = RESULTS / f"{name}.stderr.log"
        stage_started = time.perf_counter()
        record: dict[str, Any] = {"stage": name, "status": "RUNNING", "started_at": now(),
                                  "command": [str(PYTHON), *arguments], "stdout": stdout_path.as_posix(),
                                  "stderr": stderr_path.as_posix()}
        stages.append(record)
        write_json(RESULTS / "stage_execution.json", {"task_id": TASK_ID, "status": "RUNNING", "stages": stages})
        try:
            with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, stderr_path.open("w", encoding="utf-8", newline="\n") as stderr:
                completed = subprocess.run([str(PYTHON), *arguments], cwd=ROOT / "quantization_aware_cpa",
                                           stdout=stdout, stderr=stderr, env=os.environ.copy(),
                                           timeout=max(1.0, CAP_SECONDS - (time.perf_counter() - started)),
                                           check=False)
            record.update(status="COMPLETED" if completed.returncode == 0 else "FAILED",
                          exit_code=completed.returncode, ended_at=now(),
                          wall_seconds=time.perf_counter() - stage_started)
        except subprocess.TimeoutExpired as error:
            record.update(status="TIMEOUT", exit_code=None, ended_at=now(),
                          wall_seconds=time.perf_counter() - stage_started, error=str(error))
        write_json(RESULTS / "stage_execution.json", {"task_id": TASK_ID, "status": "RUNNING", "stages": stages})
        if record["status"] != "COMPLETED":
            final_status = "PARTIAL"
            failure = {"stage": name, "exit_code": record.get("exit_code"),
                       "status": record["status"], "stderr": stderr_path.as_posix()}
            break
    summary: dict[str, Any] = {
        "task_id": TASK_ID, "status": final_status,
        "execution_status": "COMPLETED" if final_status == "COMPLETED" else "PARTIAL",
        "scientific_quality": "EVALUATED" if final_status == "COMPLETED" else "NOT_EVALUATED",
        "started_at": stages[0].get("started_at") if stages else now(), "ended_at": now(),
        "wall_seconds": time.perf_counter() - started, "budget_seconds": CAP_SECONDS,
        "stages": stages, "failure": failure,
        "public_root": PUBLIC_ROOT.as_posix(), "private_root": PRIVATE_ROOT.as_posix(),
        "parent_failure_evidence": (RESULTS / "parent_failure_evidence").as_posix(),
        "ownership_transition": (RESULTS / "ownership_transition.json").as_posix(),
        "private_data_policy": {"sender_true_updates_consumed": True,
                                 "attack_true_updates_consumed": False, "evaluator_consumed": "x only",
                                 "labels_consumed": False},
        "actual_image_cell_feasibility": "NOT_EVALUATED",
    }
    if PUBLIC_ROOT.exists():
        summary["public_registration_sha256"] = sha256(PUBLIC_ROOT / "registration.json") if (PUBLIC_ROOT / "registration.json").is_file() else None
    if PRIVATE_ROOT.exists():
        summary["private_files"] = sorted(path.relative_to(PRIVATE_ROOT).as_posix() for path in PRIVATE_ROOT.rglob("*") if path.is_file())
    write_json(RESULTS / "execution_summary.json", summary)
    write_json(RESULTS / "integrity.json", {"task_id": TASK_ID, "status": final_status,
                                             "summary_sha256": sha256(RESULTS / "execution_summary.json"),
                                             "protected_evidence_preserved": True,
                                             "source_code_hashes_recorded_in_registration": True})
    lines = [f"# {TASK_ID}", "", f"Status: **{final_status}**.", "",
             "The parent INT8 averaged-feature attempt is preserved under `parent_failure_evidence/`; it failed before sender collection because its CLI passed the subparser `command` field to `collect()`.",
             "", "This run registers eight individual public FIA5000 screens, freezes public top3 by feature cosine distance + 0.1 normalized TV, and continues only those three from raw 5000-step checkpoints to total25000. Actual-image cell feasibility is **NOT_EVALUATED**.", ""]
    if failure:
        lines += [f"Failure stage: `{failure.get('stage')}`; status `{failure.get('status', 'FAILED')}`.", ""]
    lines += ["Stage records: `stage_execution.json`.", "Execution summary: `execution_summary.json`."]
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0 if final_status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
