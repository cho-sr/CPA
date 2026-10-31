"""One serial persistent wrapper for the authorized fresh16 v4 sweep."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
TASK_ID = "CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921"
PLAN_VERSION = 4
AMENDMENT_ID = "CPA_PAPER_SPARSE_FRESH16_R8_AMENDMENT_20260921"
ROOT = PROJECT / "results/paper_sparse_fresh16_keep50_30_20_R8_20260921"
CONFIG = PROJECT / "configs/cpa_fia_fresh16_R8_S5k_K3_F25k.json"
STAGE_EXECUTION = ROOT / "stage_execution.json"
EXECUTION_RECORD = ROOT / "execution_record.json"
WRAPPER_STATUS = ROOT / "wrapper_status.json"
PYTHON = Path(sys.executable).resolve()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_records(record: dict) -> None:
    write_json(STAGE_EXECUTION, record)
    write_json(EXECUTION_RECORD, record)
    write_json(WRAPPER_STATUS, {
        "task_id": TASK_ID,
        "plan_version": PLAN_VERSION,
        "amendment_id": AMENDMENT_ID,
        "n_images": 16,
        "batch_indices": [0],
        "candidate_ids": list(range(8)),
        "wrapper_pid": os.getpid(),
        "status": record.get("status"),
        "current_stage": record.get("current_stage"),
        "exit_status": record.get("exit_status"),
        "updated_utc": now(),
        "stage_execution": str(STAGE_EXECUTION),
    })


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    command = [str(PYTHON), "-u", "-m", "qacpa.run_paper_sparse_fresh16_r8",
               "--config", str(CONFIG), "--phase", "run"]
    started = now()
    record = {
        "task_id": TASK_ID,
        "plan_version": PLAN_VERSION,
        "amendment_id": AMENDMENT_ID,
        "n_images": 16,
        "batch_indices": [0],
        "candidate_ids": list(range(8)),
        "wrapper_pid": os.getpid(),
        "python": str(PYTHON),
        "project": str(PROJECT),
        "workspace": str(WORKSPACE),
        "output_root": str(ROOT),
        "private_output_root": str(PROJECT / "results_private/paper_sparse_fresh16_keep50_30_20_R8_20260921"),
        "config": str(CONFIG),
        "command": subprocess.list2cmdline(command),
        "status": "RUNNING",
        "full_experiment_status": "IN_PROGRESS",
        "started_utc": started,
        "serial": True,
        "fresh_update_collection": True,
        "supersedes": [
            ".ai/execution-handoffs/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921_V1_CLAIMED_ARCHIVE_20260921.md",
            ".ai/execution-handoffs/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921_V2_R8_CLAIMED_ARCHIVE_20260921.md",
            ".ai/execution-handoffs/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921_V3_FAILED_ARCHIVE_20260921.md",
        ],
        "stages": [],
    }
    write_json(ROOT / "startup_receipt.json", {
        "task_id": TASK_ID,
        "plan_version": PLAN_VERSION,
        "amendment_id": AMENDMENT_ID,
        "wrapper_pid": os.getpid(),
        "started_utc": started,
        "python": str(PYTHON),
        "output_root": str(ROOT),
        "config": str(CONFIG),
        "command": subprocess.list2cmdline(command),
        "fresh_update_collection": True,
        "candidate_ids": list(range(8)),
        "n_images": 16,
        "batch_indices": [0],
    })
    _write_records(record)
    stdout_path = ROOT / "fresh16.stdout.log"
    stderr_path = ROOT / "fresh16.stderr.log"
    stage = {
        "name": "fresh16_v4_full_experiment",
        "command": subprocess.list2cmdline(command),
        "status": "RUNNING",
        "started_utc": now(),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    record["current_stage"] = stage["name"]
    record["stages"].append(stage)
    _write_records(record)
    started_monotonic = time.monotonic()
    code = 1
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
        code = int(completed.returncode)
        stage["exit_code"] = code
        stage["status"] = "COMPLETED" if code == 0 else "FAILED"
    except BaseException as error:
        stage["status"] = "FAILED"
        stage["exit_code"] = 1
        stage["error"] = repr(error)
        code = 1
    stage["duration_seconds"] = round(time.monotonic() - started_monotonic, 3)
    stage["ended_utc"] = now()
    if stage["status"] == "COMPLETED":
        record.update(status="COMPLETED", full_experiment_status="FULL_EXPERIMENT_COMPLETED",
                      current_stage=None, exit_status=0, ended_utc=now())
    else:
        record.update(status="FAILED", full_experiment_status="INCOMPLETE_STOPPED_ON_FAILURE",
                      exit_status=code or 1, ended_utc=now())
    _write_records(record)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
