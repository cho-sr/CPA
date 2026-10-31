"""Persistent sequential launcher for the locked Same80 bits8..4 experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from qacpa.run_same80_bits_r100 import _resolve_project_path, _validate_config
from qacpa.same80_bits_r100_core import (
    MIN_FREE_BYTES, TASK_ID, TaskError, ensure_free_space, execution_lock,
    file_sha256, process_start_time_utc, read_json, utc_now, write_json,
)


DEFAULT_CONFIG = PROJECT / "configs/same80_bits87654_R100_S5k_K3_F25k.json"
PHASES = (
    ("prepare", "qacpa.run_same80_bits_r100"),
    ("run", "qacpa.run_same80_bits_r100"),
    ("gate", "qacpa.run_same80_bits_r100"),
    ("report", "qacpa.report_same80_bits_r100"),
)


def _runtime_check(public_root: Path) -> dict[str, Any]:
    required_python = Path("C:/miniconda/envs/DLG/python.exe")
    actual = Path(sys.executable).resolve()
    if os.name == "nt" and actual.as_posix().lower() != required_python.resolve().as_posix().lower():
        raise TaskError(f"research worker must use {required_python}, got {actual}")
    if not torch.cuda.is_available():
        raise TaskError("CUDA is unavailable; approved research execution cannot start")
    free_vram, total_vram = torch.cuda.mem_get_info()
    if free_vram <= 0 or total_vram <= 0:
        raise TaskError("CUDA reported invalid available/total device memory")
    disk_free = ensure_free_space(public_root, reserve_bytes=MIN_FREE_BYTES)
    return {"python": str(actual), "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0),
            "cuda_free_bytes_before_start": int(free_vram),
            "cuda_total_bytes": int(total_vram), "disk_free_bytes_before_start": disk_free,
            "minimum_free_disk_bytes": MIN_FREE_BYTES}


def _phase(config_path: Path, module: str, phase: str, *, log_dir: Path,
           status: dict[str, Any], status_path: Path) -> int:
    stdout_path, stderr_path = log_dir / f"{phase}.stdout.log", log_dir / f"{phase}.stderr.log"
    command = [sys.executable, "-u", "-m", module, "--config", str(config_path)]
    if module == "qacpa.run_same80_bits_r100":
        command += ["--phase", phase]
    process_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
                   "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    entry = {"phase": phase, "state": "RUNNING", "command": command,
             "started_at_utc": utc_now()}
    status["state"] = "RUNNING"
    status["current_phase"] = phase
    status["phases"].append(entry)
    write_json(status_path, status)
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(command, cwd=PROJECT, env=process_env,
                                   stdout=stdout, stderr=stderr,
                                   creationflags=flags)
        entry["pid"] = process.pid
        entry["stdout"] = stdout_path.name
        entry["stderr"] = stderr_path.name
        write_json(status_path, status)
        code = int(process.wait())
    entry.update(state="COMPLETED" if code == 0 else "FAILED",
                 exit_code=code, ended_at_utc=utc_now())
    write_json(status_path, status)
    return code


def run(config_path: Path) -> int:
    config_path = config_path.resolve()
    cfg = read_json(config_path)
    _validate_config(cfg)
    public_root = _resolve_project_path(cfg["public_root"])
    private_root = _resolve_project_path(cfg["private_root"])
    private_root.mkdir(parents=True, exist_ok=True)
    status_path = private_root / "worker_status.json"
    status: dict[str, Any] = {
        "task_id": TASK_ID, "plan_version": 1, "state": "STARTING",
        "worker_pid": os.getpid(), "worker_start_time_utc": process_start_time_utc(os.getpid()),
        "started_at_utc": utc_now(), "config_sha256": file_sha256(config_path),
        "worker_count": 1, "phases": [], "failure": None,
    }
    write_json(status_path, status)
    try:
        status["runtime"] = _runtime_check(public_root)
        with execution_lock(private_root / "orchestration_lock"):
            for phase, module in PHASES:
                code = _phase(config_path, module, phase, log_dir=private_root,
                              status=status, status_path=status_path)
                if code != 0:
                    stage_path = public_root / "stage_execution.json"
                    stage = read_json(stage_path) if stage_path.is_file() else {}
                    resource_review = (stage.get("state") == "NEEDS_REVIEW"
                                       or any(token in str(stage.get("error", "")).lower()
                                              for token in ("out of memory", "disk reserve", "cuda unavailable")))
                    status["state"] = "NEEDS_REVIEW" if resource_review else "FAILED"
                    status["failure"] = {"phase": phase, "exit_code": code,
                                          "stage_execution": stage,
                                          "recorded_at_utc": utc_now()}
                    return code
        gate_path = public_root / "public_gate.json"
        report_index = private_root / "evaluation/evaluation_index.json"
        status.update(state="COMPLETED", current_phase=None,
                      public_gate_sha256=file_sha256(gate_path),
                      evaluation_index_sha256=file_sha256(report_index),
                      finished_at_utc=utc_now())
        return 0
    except BaseException as exc:
        status.update(state="NEEDS_REVIEW" if isinstance(exc, TaskError)
                      and any(token in str(exc).lower() for token in ("disk", "cuda", "memory"))
                      else "FAILED",
                      failure={"error": f"{type(exc).__name__}: {exc}",
                               "recorded_at_utc": utc_now()},
                      finished_at_utc=utc_now())
        print(status["failure"]["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        status.setdefault("finished_at_utc", utc_now())
        write_json(status_path, status)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    return run(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
