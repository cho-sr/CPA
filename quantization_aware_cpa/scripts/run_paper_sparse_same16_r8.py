"""Serial persistent wrapper for the authorized v3/SAME16 paper sweep."""
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
AMENDMENT_ID = "CPA_PAPER_SPARSE_SAME16_R8_AMENDMENT_20260921"
PLAN_VERSION = 3
CANDIDATE_COUNT = 8
N_IMAGES = 16
SWEEP_ROOT = PROJECT / "results/paper_sparse_same16_keep50_30_20_R8_20260921"
STAGE_EXECUTION = SWEEP_ROOT / "stage_execution.json"
EXECUTION_RECORD = SWEEP_ROOT / "execution_record.json"
WRAPPER_STATUS = SWEEP_ROOT / "wrapper_status.json"
PYTHON = Path(sys.executable).resolve()
FRACTIONS = (50, 30, 20)
CONFIG_NAMES = tuple(
    f"cpa_fia_random_topk{percent}_same16_R8_S5k_K3_F25k.json"
    for percent in FRACTIONS
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_list() -> list[tuple[str, list[str]]]:
    stages: list[tuple[str, list[str]]] = []
    for percent, name in zip(FRACTIONS, CONFIG_NAMES):
        stages.append((f"prepare_topk{percent}", [str(PYTHON), "-u", "-m", "qacpa.run_paper_sparse_same16_r8",
                       "--config", f"configs/{name}", "--phase", "prepare"]))
    for percent, name in zip(FRACTIONS, CONFIG_NAMES):
        stages.append((f"migrate_topk{percent}", [str(PYTHON), "-u", "-m", "qacpa.run_paper_sparse_same16_r8",
                       "--config", f"configs/{name}", "--phase", "migrate"]))
    for percent, name in zip(FRACTIONS, CONFIG_NAMES):
        stages.append((f"run_topk{percent}", [str(PYTHON), "-u", "-m", "qacpa.run_paper_sparse_same16_r8",
                       "--config", f"configs/{name}", "--phase", "run"]))
    stages.append(("combined_report", [str(PYTHON), "-u", "-m", "qacpa.report_paper_sparse_same16_r8"]))
    return stages


def _write_records(record: dict) -> None:
    write_json(STAGE_EXECUTION, record)
    write_json(EXECUTION_RECORD, record)
    write_json(WRAPPER_STATUS, {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
        "candidate_count": CANDIDATE_COUNT, "candidate_ids": list(range(CANDIDATE_COUNT)),
        "n_images": N_IMAGES, "batch_indices": [0], "wrapper_pid": os.getpid(),
        "status": record.get("status"), "current_stage": record.get("current_stage"),
        "exit_status": record.get("exit_status"), "updated_utc": now(),
        "stage_execution": str(STAGE_EXECUTION),
    })


def _validate_configs() -> None:
    config_dir = PROJECT / "configs"
    expected = set(CONFIG_NAMES)
    actual = {path.name for path in config_dir.iterdir() if path.name in expected}
    if actual != expected:
        raise RuntimeError(f"Authorized SAME16 config set is incomplete: {sorted(actual)}")
    for percent, name in zip(FRACTIONS, CONFIG_NAMES):
        config = json.loads((config_dir / name).read_text(encoding="utf-8"))
        if (config.get("keep_percent"), config.get("candidate_count"), config.get("n_images"),
                config.get("batch_indices")) != (percent, CANDIDATE_COUNT, N_IMAGES, [0]):
            raise RuntimeError(f"SAME16 config scope mismatch: {name}")
        if config.get("candidate_ids") != list(range(CANDIDATE_COUNT)):
            raise RuntimeError(f"SAME16 candidate IDs mismatch: {name}")
        if config.get("plan_version") != PLAN_VERSION or config.get("amendment_id") != AMENDMENT_ID:
            raise RuntimeError(f"SAME16 amendment identity mismatch: {name}")


def main() -> int:
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    _validate_configs()
    stages = command_list()
    record: dict = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
        "candidate_count": CANDIDATE_COUNT, "candidate_ids": list(range(CANDIDATE_COUNT)),
        "n_images": N_IMAGES, "batch_indices": [0], "wrapper_pid": os.getpid(),
        "python": str(PYTHON), "project": str(PROJECT), "workspace": str(WORKSPACE),
        "output_root": str(SWEEP_ROOT), "status": "RUNNING", "full_experiment_status": "IN_PROGRESS",
        "started_utc": now(), "deadline_hours": 48, "serial": True,
        "supersedes": str(PROJECT / "results/paper_sparse_same80_keep50_30_20_R8_20260921/v2_supersession.json"),
        "commands": [subprocess.list2cmdline(args) for _, args in stages], "stages": [],
    }
    write_json(SWEEP_ROOT / "startup_receipt.json", {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
        "candidate_count": CANDIDATE_COUNT, "candidate_ids": list(range(CANDIDATE_COUNT)),
        "n_images": N_IMAGES, "batch_indices": [0], "wrapper_pid": os.getpid(),
        "started_utc": record["started_utc"], "python": str(PYTHON), "output_root": str(SWEEP_ROOT),
        "stage_execution": str(STAGE_EXECUTION), "command_count": len(stages),
    })
    _write_records(record)
    for name, args in stages:
        stdout_path = SWEEP_ROOT / f"{name}.stdout.log"
        stderr_path = SWEEP_ROOT / f"{name}.stderr.log"
        stage = {"name": name, "command": subprocess.list2cmdline(args), "status": "RUNNING",
                 "started_utc": now(), "stdout": str(stdout_path), "stderr": str(stderr_path)}
        record["current_stage"] = name
        record["stages"].append(stage)
        _write_records(record)
        started = time.monotonic()
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                completed = subprocess.run(args, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                           stdout=stdout, stderr=stderr,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
            stage["exit_code"] = int(completed.returncode)
            stage["status"] = "COMPLETED" if completed.returncode == 0 else "FAILED"
        except BaseException as error:
            stage["status"] = "FAILED"
            stage["exit_code"] = 1
            stage["error"] = repr(error)
        stage["duration_seconds"] = round(time.monotonic() - started, 3)
        stage["ended_utc"] = now()
        _write_records(record)
        if stage["status"] != "COMPLETED":
            record.update(status="FAILED", full_experiment_status="INCOMPLETE_STOPPED_ON_FAILURE",
                          exit_status=int(stage.get("exit_code", 1)) or 1, ended_utc=now())
            _write_records(record)
            return int(record["exit_status"])
    record.update(status="COMPLETED", full_experiment_status="FULL_EXPERIMENT_COMPLETED",
                  current_stage=None, exit_status=0, ended_utc=now())
    _write_records(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
