import json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[3]
task = Path(__file__).parent
record_path = task / "diagnostic_execution.json"
command = [r"C:/miniconda/envs/DLG/python.exe", "-m", "quantization_aware_cpa.qacpa.signed_stage_diagnostics",
 "--naive", "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive", "--cell-rank", "quantization_aware_cpa/results/signed_cell_rank_B16_T10/cell_rank_ap5",
 "--comparison", "quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/evaluation_ap5/comparison.json", "--public-observation", "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation",
 "--private-observation", "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation", "--checkpoint", r"C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth",
 "--method-plan", ".ai/proposals/SIGNED_CPA_FIA_STAGE_DIAGNOSTIC_20260915.md", "--expected-plan-sha256", "d24c271ae8461bdad3061c957b53f247cf852c7ba1bbf1d63933205511f0d14d",
 "--output", "quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/stage_diagnostic_v2_repaired", "--device", "cuda", "--eval-microbatch", "1", "--max-evaluation-seconds", "600", "--max-cuda-allocated-gib", "6"]
record = {"task_id": "CPA_SIGNED_STAGE_DIAGNOSTIC_REPAIR_20260915", "stage": "corrected_offline_evaluation", "status": "RUNNING", "pid": os.getpid(), "command": command, "cwd": str(root), "started_at_utc": datetime.now(timezone.utc).isoformat()}
record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
start = time.perf_counter()
try:
    with (task / "diagnostic_stdout.log").open("w", encoding="utf-8") as out, (task / "diagnostic_stderr.log").open("w", encoding="utf-8") as err:
        result = subprocess.run(command, cwd=root, stdout=out, stderr=err, check=False)
    record.update(status="COMPLETED" if result.returncode == 0 else "FAILED", exit_status=result.returncode,
                  wall_seconds=time.perf_counter() - start, finished_at_utc=datetime.now(timezone.utc).isoformat())
except Exception as exc:
    record.update(status="FAILED", exit_status=1, error=str(exc), wall_seconds=time.perf_counter() - start,
                  finished_at_utc=datetime.now(timezone.utc).isoformat())
record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
raise SystemExit(record.get("exit_status", 1))
