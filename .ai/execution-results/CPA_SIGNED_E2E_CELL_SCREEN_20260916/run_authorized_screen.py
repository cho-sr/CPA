"""One-shot task wrapper for the authorized public screen and offline evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / ".ai/execution-results/CPA_SIGNED_E2E_CELL_SCREEN_20260916"
PYTHON = Path("C:/miniconda/envs/DLG/python.exe")
PUBLIC = ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation"
NAIVE = ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive"
CHECKPOINT = Path("C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
PUBLIC_OUTPUT = ROOT / "quantization_aware_cpa/results/signed_e2e_cell_screen_B16_T10_20260916"
PRIVATE_TRUTH = ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"
PRIVATE_OUTPUT = ROOT / "quantization_aware_cpa/results_private/signed_e2e_cell_screen_B16_T10_20260916"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run_stage(name: str, command: list[str], log_prefix: str) -> dict:
    started = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "quantization_aware_cpa")
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True)
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    (REPORT / f"{log_prefix}_stdout.log").write_text(stdout, encoding="utf-8")
    (REPORT / f"{log_prefix}_stderr.log").write_text(stderr, encoding="utf-8")
    record = {"task_id": "CPA_SIGNED_E2E_CELL_SCREEN_20260916", "stage": name,
              "command": command, "cwd": str(ROOT), "started_at_utc": now(),
              "exit_status": result.returncode, "wall_seconds": time.perf_counter() - started}
    write_json(REPORT / f"{log_prefix}_execution.json", record)
    return record


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    write_json(REPORT / "wrapper_start.json", {"task_id": "CPA_SIGNED_E2E_CELL_SCREEN_20260916",
              "pid": os.getpid(), "started_at_utc": now(), "cwd": str(ROOT),
              "python": str(PYTHON), "model_calls_while_running": 0})
    tests = [str(PYTHON), "-m", "pytest",
             "quantization_aware_cpa/tests/test_signed_e2e_cells.py",
             "quantization_aware_cpa/tests/test_fc_replay.py",
             "quantization_aware_cpa/tests/test_signed_stage_diagnostics.py",
             "quantization_aware_cpa/tests/test_signed_cell_rank.py",
             "quantization_aware_cpa/tests/test_signed_cell_rank_pipeline.py",
             "quantization_aware_cpa/tests/test_signed_cell_rank_validation.py",
             "-q", "--basetemp", ".tmp_pytest_signed_e2e_cell_screen_20260916"]
    records = {"tests": run_stage("focused_and_relevant_tests", tests, "tests")}
    if records["tests"]["exit_status"] != 0:
        records.update(status="FAILED_TESTS", wall_seconds=time.perf_counter() - started)
        write_json(REPORT / "execution_summary.json", records)
        return 1
    runner = [str(PYTHON), "-m", "quantization_aware_cpa.qacpa.signed_e2e_runner",
              "--public-observation", str(PUBLIC), "--naive", str(NAIVE),
              "--checkpoint", str(CHECKPOINT), "--output", str(PUBLIC_OUTPUT),
              "--device", "cuda"]
    records["runner"] = run_stage("public_optimization_and_replay_evaluation", runner, "runner")
    if records["runner"]["exit_status"] != 0:
        records.update(status="FAILED_PUBLIC_RUNNER", wall_seconds=time.perf_counter() - started)
        write_json(REPORT / "execution_summary.json", records)
        return 1
    evaluator = [str(PYTHON), "-m", "quantization_aware_cpa.qacpa.signed_e2e_evaluation",
                 "--public-output", str(PUBLIC_OUTPUT), "--public-observation", str(PUBLIC),
                 "--naive", str(NAIVE), "--checkpoint", str(CHECKPOINT),
                 "--private-truth", str(PRIVATE_TRUTH), "--output", str(PRIVATE_OUTPUT)]
    records["evaluation"] = run_stage("authorized_offline_private_evaluation", evaluator, "evaluation")
    records.update(status="COMPLETED" if records["evaluation"]["exit_status"] == 0 else "FAILED_EVALUATION",
                   wall_seconds=time.perf_counter() - started)
    write_json(REPORT / "execution_summary.json", records)
    return 0 if records["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
