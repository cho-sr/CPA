"""Run only the authorized saved-result offline evaluator once."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PYTHON = Path(r"C:/miniconda/envs/DLG/python.exe")
ARGS = [
    str(PYTHON), "-m", "quantization_aware_cpa.qacpa.signed_cell_rank_evaluation",
    "--naive", "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive",
    "--cell-rank", "quantization_aware_cpa/results/signed_cell_rank_B16_T10/cell_rank_ap5",
    "--public-observation", "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation",
    "--private-observation", "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation",
    "--output", "quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/evaluation_ap5",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(value: dict) -> None:
    target = OUT / "evaluation_execution.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def main() -> int:
    record = {
        "task_id": "CPA_SIGNED_CELL_EVALUATION_REPAIR_20260915",
        "plan_version": 1,
        "status": "RUNNING",
        "started_utc": now(),
        "command": subprocess.list2cmdline(ARGS),
        "stdout": str(OUT / "offline_evaluation.stdout.log"),
        "stderr": str(OUT / "offline_evaluation.stderr.log"),
        "output": "quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/evaluation_ap5",
    }
    write(record)
    started = time.monotonic()
    try:
        with (OUT / "offline_evaluation.stdout.log").open("wb") as stdout, \
                (OUT / "offline_evaluation.stderr.log").open("wb") as stderr:
            result = subprocess.run(
                ARGS, cwd=ROOT, stdout=stdout, stderr=stderr,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
            )
        record["exit_code"] = int(result.returncode)
    except BaseException as error:
        record["exit_code"] = 1
        record["error"] = repr(error)
    record["duration_seconds"] = round(time.monotonic() - started, 3)
    record["ended_utc"] = now()
    record["status"] = "COMPLETED" if record["exit_code"] == 0 else "FAILED"
    record["full_evaluation_status"] = "COMPLETED" if record["exit_code"] == 0 else "INCOMPLETE_STOPPED_ON_FAILURE"
    write(record)
    return record["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
