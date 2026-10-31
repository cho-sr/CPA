import json, os, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[3]
record_path = Path(__file__).with_name("diagnostic_execution.json")
stdout_path = Path(__file__).with_name("diagnostic_stdout.log")
stderr_path = Path(__file__).with_name("diagnostic_stderr.log")
command = [
    r"C:/miniconda/envs/DLG/python.exe", "-m", "quantization_aware_cpa.qacpa.signed_stage_diagnostics",
    "--naive", "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive",
    "--cell-rank", "quantization_aware_cpa/results/signed_cell_rank_B16_T10/cell_rank_ap5",
    "--comparison", "quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/evaluation_ap5/comparison.json",
    "--public-observation", "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation",
    "--private-observation", "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation",
    "--checkpoint", r"C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth",
    "--method-plan", ".ai/proposals/SIGNED_CPA_FIA_STAGE_DIAGNOSTIC_20260915.md",
    "--expected-plan-sha256", "d24c271ae8461bdad3061c957b53f247cf852c7ba1bbf1d63933205511f0d14d",
    "--output", "quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/stage_diagnostic_v2",
    "--device", "cuda", "--eval-microbatch", "1", "--max-evaluation-seconds", "600", "--max-cuda-allocated-gib", "6",
]
start = time.time()
record = {"task_id": "CPA_SIGNED_STAGE_DIAGNOSTIC_V2_20260915", "stage": "diagnostic",
          "status": "RUNNING", "command": command, "pid": os.getpid(),
          "started_at_utc": datetime.now(timezone.utc).isoformat(), "cwd": str(root)}
record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
try:
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        completed = subprocess.run(command, cwd=root, stdout=out, stderr=err, check=False)
    record.update(status="COMPLETED" if completed.returncode == 0 else "FAILED",
                  exit_status=completed.returncode, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  wall_seconds=time.time()-start)
except Exception as exc:
    record.update(status="FAILED", exit_status=None, error=str(exc), traceback=traceback.format_exc(),
                  finished_at_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.time()-start)
record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
sys.exit(record.get("exit_status") or 1)
