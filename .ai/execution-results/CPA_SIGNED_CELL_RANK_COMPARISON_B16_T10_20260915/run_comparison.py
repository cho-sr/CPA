"""One-shot serial runner for the user-authorized AP5/CPA/FIA/comparison.

This execution-only wrapper never changes scientific arguments. It records each
child's stdout, stderr, exit code and timing, and stops immediately on failure.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PYTHON = Path(r"C:/miniconda/envs/DLG/python.exe")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_record(record: dict) -> None:
    target = OUT / "stage_execution.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def command_args() -> list[tuple[str, list[str]]]:
    return [
        (
            "preprocess",
            [
                str(PYTHON), "-m", "quantization_aware_cpa.qacpa.signed_cell_rank_runner",
                "preprocess", "--public-observation",
                "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation",
                "--output", "quantization_aware_cpa/results/signed_cell_rank_B16_T10/preprocess_ap5",
                "--ap-iterations", "5", "--projection-device", "cuda",
            ],
        ),
        (
            "attack",
            [
                str(PYTHON), "-m", "quantization_aware_cpa.qacpa.signed_cell_rank_runner",
                "attack", "--public-observation",
                "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation",
                "--preprocessed", "quantization_aware_cpa/results/signed_cell_rank_B16_T10/preprocess_ap5",
                "--checkpoint", r"C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth",
                "--output", "quantization_aware_cpa/results/signed_cell_rank_B16_T10/cell_rank_ap5",
                "--expected-ap-iterations", "5",
            ],
        ),
        (
            "evaluation",
            [
                str(PYTHON), "-m", "quantization_aware_cpa.qacpa.signed_cell_rank_evaluation",
                "--naive", "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive",
                "--cell-rank", "quantization_aware_cpa/results/signed_cell_rank_B16_T10/cell_rank_ap5",
                "--public-observation", "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation",
                "--private-observation", "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation",
                "--output", "quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/evaluation_ap5",
            ],
        ),
    ]


def main() -> int:
    record: dict = {
        "task_id": "CPA_SIGNED_CELL_RANK_COMPARISON_B16_T10_20260915",
        "plan_version": 1,
        "wrapper_pid": __import__("os").getpid(),
        "status": "RUNNING",
        "full_experiment_status": "IN_PROGRESS",
        "started_utc": now(),
        "stages": [],
    }
    write_record(record)
    for name, args in command_args():
        stdout_path = OUT / f"{name}.stdout.log"
        stderr_path = OUT / f"{name}.stderr.log"
        stage = {
            "name": name,
            "command": subprocess.list2cmdline(args),
            "status": "RUNNING",
            "started_utc": now(),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        record["stages"].append(stage)
        write_record(record)
        started = time.monotonic()
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                result = subprocess.run(
                    args, cwd=ROOT, stdout=stdout, stderr=stderr,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    check=False,
                )
            stage["exit_code"] = int(result.returncode)
            stage["duration_seconds"] = round(time.monotonic() - started, 3)
            stage["ended_utc"] = now()
            stage["status"] = "COMPLETED" if result.returncode == 0 else "FAILED"
        except BaseException as error:  # preserve exact wrapper failure for the report
            stage["status"] = "FAILED"
            stage["error"] = repr(error)
            stage["duration_seconds"] = round(time.monotonic() - started, 3)
            stage["ended_utc"] = now()
        write_record(record)
        if stage["status"] != "COMPLETED":
            record.update(
                status="FAILED",
                full_experiment_status="INCOMPLETE_STOPPED_ON_FAILURE",
                ended_utc=now(),
            )
            write_record(record)
            return int(stage.get("exit_code", 1)) or 1
    record.update(
        status="COMPLETED",
        full_experiment_status="FULL_EXPERIMENT_COMPLETED",
        ended_utc=now(),
    )
    write_record(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
