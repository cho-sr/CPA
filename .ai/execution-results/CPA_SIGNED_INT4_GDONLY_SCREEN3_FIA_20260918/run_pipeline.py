from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback


TASK_ID = "CPA_SIGNED_INT4_GDONLY_SCREEN3_FIA_20260918"
ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / ".ai/execution-results" / TASK_ID
PUBLIC = ROOT / "quantization_aware_cpa/results/signed_int4_gdonly_randomized8_screen3_fia_B16_T10_20260918"
PRIVATE = ROOT / "quantization_aware_cpa/results_private/signed_int4_gdonly_randomized8_screen3_fia_B16_T10_20260918"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    startup = {"task_id": TASK_ID, "status": "STARTING", "pid": os.getpid(), "started_at": now(),
               "runtime": sys.executable, "cwd": str(ROOT), "public_root": str(PUBLIC), "private_root": str(PRIVATE),
               "model_calls": 0, "stage": "INITIALIZE"}
    write(RESULTS / "startup_receipt.json", startup)
    record = {"task_id": TASK_ID, "status": "RUNNING", "pid": os.getpid(), "started_at": startup["started_at"],
              "runtime": sys.executable, "public_root": str(PUBLIC), "private_root": str(PRIVATE),
              "exit_status": None, "model_calls": 0}
    write(RESULTS / "stage_execution.json", record)
    try:
        sys.path.insert(0, str(ROOT))
        from quantization_aware_cpa.qacpa import signed_int4_gdonly_screen3 as pipeline
        startup.update({"status": "RUNNING", "stage": "IMPLEMENTATION_READY", "cuda_available": bool(pipeline.torch.cuda.is_available()),
                        "torch": pipeline.torch.__version__, "cuda": pipeline.torch.version.cuda})
        write(RESULTS / "startup_receipt.json", startup)
        if not PUBLIC.exists() and not PRIVATE.exists():
            pipeline.register(PUBLIC, PRIVATE)
        elif not (PUBLIC.exists() and PRIVATE.exists()):
            raise RuntimeError("public/private output roots are in an incompatible partial state")
        startup.update({"stage": "REGISTERED", "registration": str(PUBLIC / "registration.json")})
        write(RESULTS / "startup_receipt.json", startup)
        summary = pipeline.run_all(PUBLIC, PRIVATE)
        record.update({"status": summary["status"], "exit_status": 0, "ended_at": now(), "summary": summary})
        write(RESULTS / "stage_execution.json", record)
        write(RESULTS / "execution_summary.json", {**summary, "stage_execution": str(RESULTS / "stage_execution.json")})
        (RESULTS / "report.md").write_text(
            f"# {TASK_ID}\n\nStatus: **{summary['status']}**.\n\n"
            f"Public root: `{PUBLIC}`\n\nPrivate report: `{PRIVATE / 'evaluation/comparison.json'}`\n",
            encoding="utf-8", newline="\n")
        return 0 if summary["status"] == "COMPLETED" else 1
    except BaseException as error:
        failure = {"task_id": TASK_ID, "status": "FAILED", "error_type": type(error).__name__,
                   "error": str(error), "traceback": traceback.format_exc(), "failed_at": now(),
                   "public_root": str(PUBLIC), "private_root": str(PRIVATE), "model_calls": 0}
        write(RESULTS / "execution_summary.json", failure)
        write(RESULTS / "failure.json", failure)
        record.update({"status": "FAILED", "exit_status": 1, "ended_at": now(), "error_type": type(error).__name__, "error": str(error)})
        write(RESULTS / "stage_execution.json", record)
        (RESULTS / "report.md").write_text(
            f"# {TASK_ID}\n\nStatus: **FAILED**.\n\nError: `{type(error).__name__}: {error}`\n\n"
            f"Completed artifacts, if any, remain under `{PUBLIC}` and `{PRIVATE}`.\n",
            encoding="utf-8", newline="\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
