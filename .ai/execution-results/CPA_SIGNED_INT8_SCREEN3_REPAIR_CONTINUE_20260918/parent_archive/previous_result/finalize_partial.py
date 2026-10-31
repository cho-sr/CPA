"""Finalize an already failed screen3 wrapper from saved evidence only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import torch


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
TASK_ID = "CPA_SIGNED_INT8_RANDOMIZED8_SCREEN3_FIA_20260917"
PUBLIC = ROOT / "quantization_aware_cpa/results/signed_int8_randomized8_screen3_fia_B16_T10_20260917"
PRIVATE = ROOT / "quantization_aware_cpa/results_private/signed_int8_randomized8_screen3_fia_B16_T10_20260917"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def main() -> int:
    summary_path = REPORT / "execution_summary.json"
    report_path = REPORT / "report.md"
    shutil.copy2(summary_path, REPORT / "execution_summary_pre_audit.json")
    shutil.copy2(report_path, REPORT / "report_pre_audit.md")
    summary = read(summary_path)
    separation = read(PUBLIC / "separation/result.json")
    selection = read(PUBLIC / "selection.json")
    prefix_records = []
    for index in range(8):
        branch = PUBLIC / f"fia_member_{index:02d}_5000"
        prefix_records.append({
            "member_index": index,
            "checkpoint": (branch / "checkpoint_5000.pt").is_file(),
            "candidate": (branch / "candidate.pt").is_file(),
            "result": (branch / "result.json").is_file(),
            "checkpoint_sha256": sha(branch / "checkpoint_5000.pt"),
            "result_status": read(branch / "result.json")["status"],
            "completed_iterations": read(branch / "result.json")["completed_iterations"],
        })
    center = PUBLIC / "fia_center_25000"
    child = PUBLIC / "fia_member_01_25000/checkpoint_25000.pt"
    prefix = PUBLIC / "fia_member_01_5000/checkpoint_5000.pt"
    child_state = torch.load(child, map_location="cpu", weights_only=False)
    audit = {
        "task_id": TASK_ID,
        "status": "PARTIAL",
        "scientific_quality": "NOT_EVALUATED",
        "failure": {
            "stage": "public_attack",
            "error_type": "StagedFIAError",
            "error": "25000 trajectory did not produce its 5000 checkpoint",
            "meaning": "continuation child had an authenticated parent checkpoint and completed 20000 additional updates; the writer then incorrectly required a new prefix checkpoint before writing result/candidate",
            "no_restart": True,
        },
        "completed_stages": ["registration", "sender", "nine_separations", "public_alignment",
                             "center_25000", "eight_member_5000_screens", "public_top3_selection",
                             "member_01_continuation_updates_to_25000"],
        "not_run": ["member_07_continuation", "member_05_continuation", "private_gate", "private_evaluation", "figures"],
        "separation": {"calls": separation["separation_calls"],
                        "all_fastica_converged": all(row["fastica"]["converged"] for row in separation["records"])},
        "selection": {"selected_member_indices": selection["selected_member_indices"],
                      "ranking": [{"rank": row["rank"], "member_index": row["member_index"], "score": row["score"]}
                                  for row in selection["ranking"]],
                      "private_metrics_used": selection["private_metrics_used"]},
        "center": {
            "checkpoint_5000_sha256": sha(center / "checkpoint_5000.pt"),
            "checkpoint_25000_sha256": sha(center / "checkpoint_25000.pt"),
            "candidate_5000_sha256": sha(center / "candidate_5000.pt"),
            "candidate_25000_sha256": sha(center / "candidate.pt"),
        },
        "member_prefixes": prefix_records,
        "member_01_child": {
            "checkpoint_25000_sha256": sha(child),
            "completed_iterations": child_state["completed_iterations"],
            "parent_checkpoint_sha256": child_state["parent_checkpoint_sha256"],
            "parent_checkpoint_matches_member_01_prefix": child_state["parent_checkpoint_sha256"] == sha(prefix),
            "candidate_saved": (PUBLIC / "fia_member_01_25000/candidate.pt").is_file(),
            "result_saved": (PUBLIC / "fia_member_01_25000/result.json").is_file(),
        },
        "update_counts": {"center": 25000, "member_prefixes": 8 * 5000,
                           "member_continuation_additional": 20000, "total_executed": 85000,
                           "registered_total": 125000},
        "private_boundary": {"sender_true_updates_consumed": True, "attack_true_updates_consumed": False,
                              "evaluator_true_updates_consumed": False, "private_evaluation_started": False},
        "actual_image_cell_feasibility": "NOT_EVALUATED",
        "source_and_parent_evidence": {"parent_failure_evidence": (REPORT / "parent_failure_evidence").as_posix(),
                                        "ownership_transition": (REPORT / "ownership_transition.json").as_posix()},
    }
    write(REPORT / "final_audit.json", audit)
    summary.update({"status": "PARTIAL", "execution_status": "PARTIAL", "scientific_quality": "NOT_EVALUATED",
                    "failure_detail": audit["failure"], "artifact_audit": audit,
                    "final_audit": (REPORT / "final_audit.json").as_posix(),
                    "actual_update_counts": audit["update_counts"],
                    "private_evaluation_started": False})
    write(summary_path, summary)
    write(REPORT / "integrity.json", {"task_id": TASK_ID, "status": "PARTIAL",
                                       "summary_sha256": sha(summary_path),
                                       "final_audit_sha256": sha(REPORT / "final_audit.json"),
                                       "public_registration_sha256": sha(PUBLIC / "registration.json"),
                                       "public_observation_manifest_sha256": sha(PUBLIC / "observation/manifest.json"),
                                       "protected_evidence_preserved": True,
                                       "no_rerun_or_new_arm": True})
    lines = [
        f"# {TASK_ID}", "", "Status: **PARTIAL / FAILED DURING PUBLIC CONTINUATION**.", "",
        "Registration, sender conversion, all nine ICA separations, public alignment, center FIA25000, and all eight member FIA5000 screens completed.",
        "", "The frozen public ranking was member 01, member 07, member 05. Member 01 completed its 20000 continuation updates from the authenticated 5000-step raw checkpoint and reached a saved 25000-step raw checkpoint. The isolated writer then raised `StagedFIAError` because it incorrectly required a new prefix checkpoint for a continuation child; its candidate/result files were therefore not written.",
        "", "Executed optimizer updates: 85,000 of the registered 125,000. Member 07/member 05 continuation, private gate/evaluation, metrics and figures were not run. No restart, new arm, tuning or private feedback was performed.",
        "", "Sender consumption: archived true updates were consumed for INT8 conversion. Attack consumed public q/s and settings only. Private evaluation did not start. Actual-image cell feasibility: **NOT_EVALUATED**.",
        "", "See `final_audit.json`, `execution_summary.json`, and the preserved `parent_failure_evidence/` directory for hashes and stage evidence.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
