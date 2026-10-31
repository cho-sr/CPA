from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
TASK = "CPA_SIGNED_INT4_GDONLY_SCREEN3_FIA_20260918"
RESULT = ROOT / ".ai/execution-results" / TASK
PUBLIC = ROOT / "quantization_aware_cpa/results/signed_int4_gdonly_randomized8_screen3_fia_B16_T10_20260918"
PRIVATE = ROOT / "quantization_aware_cpa/results_private/signed_int4_gdonly_randomized8_screen3_fia_B16_T10_20260918"
OUTPUT = PRIVATE / "evaluation_repair_20260918"


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from quantization_aware_cpa.qacpa import signed_int4_gdonly_screen3 as pipeline
    registration = pipeline.read_json(PUBLIC / "registration.json")
    repair = {
        "task_id": TASK,
        "status": "REGISTERED_REPORTING_REPAIR",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "repair_scope": "renderer_axis_and_offline_evaluation_output_only",
        "scientific_optimizer_steps_added": 0,
        "public_gate_before_repair": "PUBLIC_GATE_PASSED",
        "saved_public_gate_sha256": pipeline.file_sha256(PUBLIC / "public_gate.json"),
        "registered_code_hashes": registration["code_hashes"],
        "current_code_hashes": pipeline._code_hashes(),
        "registered_protected_source_hashes": registration["protected_source_hashes"],
        "current_protected_source_hashes": pipeline._protected_hashes(),
        "private_load_before_gate": False,
        "reason": "PIL received a doubly permuted CHW array during the first reporting pass; public artifacts and all optimizer states are preserved.",
    }
    provenance = RESULT / "evaluation_repair_provenance.json"
    write(provenance, repair)
    try:
        result = pipeline.evaluate(PUBLIC, PRIVATE, output=OUTPUT, repair_provenance=provenance)
        summary = {
            "task_id": TASK, "status": result["status"], "evaluation_status": result.get("evaluation_status"),
            "public_status": "PUBLIC_COMPLETED", "public_gate": result["public_gate"]["status"],
            "optimizer_steps_added": 0, "separation_calls_reused": 9, "new_fia_updates_reused": 125000,
            "evaluation_output": str(OUTPUT / "comparison.json"), "repair_provenance": str(provenance),
            "finished_at": datetime.now(timezone.utc).isoformat(), "actual_image_cell_feasibility": "NOT_EVALUATED",
        }
        write(RESULT / "execution_summary.json", summary)
        (RESULT / "report.md").write_text(
            f"# {TASK}\n\nStatus: **{result['status']}**.\n\n"
            "The public computation completed before a renderer-only repair. The saved public gate was re-authenticated with strict revision provenance; the offline comparison then consumed private x only.\n\n"
            f"Evaluation: `{OUTPUT / 'comparison.json'}`\n\n"
            f"Repair provenance: `{provenance}`\n",
            encoding="utf-8", newline="\n")
        return 0 if result["status"] == "COMPLETED" else 1
    except BaseException as error:
        failure = {"task_id": TASK, "status": "FAILED", "error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc(), "repair_provenance": str(provenance),
                   "finished_at": datetime.now(timezone.utc).isoformat(), "optimizer_steps_added": 0}
        write(RESULT / "evaluation_repair_failure.json", failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

