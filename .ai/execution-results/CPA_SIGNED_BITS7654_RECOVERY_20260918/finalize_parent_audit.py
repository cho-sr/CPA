from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / ".ai/execution-results/CPA_SIGNED_BITS7654_RECOVERY_20260918"
ACTIVE = REPO / ".ai/EXECUTION_HANDOFF.md"
BITS = {
    7: "signed_int7_randomized8_screen3_fia_B16_T10_20260918_sweep",
    6: "signed_int6_randomized8_screen3_fia_B16_T10_20260918_sweep",
    5: "signed_int5_randomized8_screen3_fia_B16_T10_20260918_sweep",
    4: "signed_int4_randomized8_screen3_fia_B16_T10_20260918_sweep",
}
SELECTIONS = {7: [0, 5, 3], 6: [1, 3, 4], 5: [5, 7, 6], 4: [6, 5, 0]}
EXPECTED_SOURCE_HASHES = {
    "signed_bits_screen3_runner.py": "0ab6bdac5e121b1713e7124bc8dcc958e9b962d6eb667676389823e87d4ff984",
    "signed_bits_screen3_evaluation.py": "811b2d45b0b5c98b46d018b11e0ff80dec8314364968cba9f43b266d76b51120",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


summary_path = ROOT / "execution_summary.json"
summary = read_json(summary_path)
if summary.get("task_id") != "CPA_SIGNED_BITS7654_RECOVERY_20260918":
    raise RuntimeError("wrong recovery summary")
if summary.get("status") != "COMPLETED" or summary.get("bit_status") != {str(b): "COMPLETED" for b in BITS}:
    raise RuntimeError("recovery summary does not claim all four completed")

active_text = ACTIVE.read_text(encoding="utf-8-sig")
if "TASK_ID: CPA_SIGNED_BITS7654_RECOVERY_20260918" not in active_text:
    raise RuntimeError("active handoff is not the recovery task")
shutil.copyfile(ACTIVE, ROOT / "active_handoff_before_finalization.md")

comparisons = {}
protected_candidate_hashes = {}
figure_records = {}
for bits, root_name in BITS.items():
    evaluation = REPO / f"quantization_aware_cpa/results_private/{root_name}/evaluation_recovery_20260918"
    comparison_path = evaluation / "comparison.json"
    comparison = read_json(comparison_path)
    gate = comparison.get("public_gate", {})
    required = {
        "status": "COMPLETED",
        "evaluation_status": "COMPLETED",
        "scientific_quality": "EVALUATED",
        "private_x_only": True,
        "labels_consumed": False,
    }
    if any(comparison.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"comparison status failed for bit {bits}")
    if gate.get("status") != "PUBLIC_GATE_PASSED" or gate.get("private_loaded") is not False or gate.get("candidate_count") != 13:
        raise RuntimeError(f"public gate failed for bit {bits}")
    if comparison.get("frozen_selection") != SELECTIONS[bits]:
        raise RuntimeError(f"selection mismatch for bit {bits}")
    if "Hungarian" not in str(comparison.get("matching")) or "reused" not in str(comparison.get("matching")):
        raise RuntimeError(f"matching provenance failed for bit {bits}")
    paths = comparison.get("candidate_paths", {})
    hashes = comparison.get("candidate_hashes", {})
    gate_candidates = gate.get("candidates", {})
    if len(paths) != 13 or set(paths) != set(hashes) or set(paths) != set(gate_candidates):
        raise RuntimeError(f"candidate binding count failed for bit {bits}")
    candidate_records = {}
    for name, raw_path in paths.items():
        path = Path(raw_path).resolve()
        actual = sha(path)
        record = gate_candidates[name]
        if actual != hashes[name] or actual != record.get("sha256", record.get("candidate_sha256")) or Path(record.get("path", "")).resolve() != path:
            raise RuntimeError(f"candidate hash failed for bit {bits}: {name}")
        candidate_records[name] = {"path": str(path), "sha256": actual}
    protected_candidate_hashes[str(bits)] = candidate_records
    figure_dir = evaluation / "figures_with_baseline_20260918"
    figure_manifest = figure_dir / "figure_manifest.json"
    if not figure_manifest.is_file():
        raise RuntimeError(f"corrected figure manifest missing for bit {bits}")
    figure = read_json(figure_manifest)
    final_columns = figure.get("column_contract", {}).get("final", [])
    if len(final_columns) != 5 or "Ground truth" not in final_columns[0] or "baseline" not in final_columns[1].lower():
        raise RuntimeError(f"baseline-inclusive figure contract failed for bit {bits}")
    figure_records[str(bits)] = {
        "directory": str(figure_dir),
        "manifest": str(figure_manifest),
        "final": figure.get("figures", {}).get("final"),
        "screen": figure.get("figures", {}).get("screen"),
    }
    comparisons[str(bits)] = {
        "comparison_path": str(comparison_path),
        "comparison_sha256": sha(comparison_path),
        "status": comparison["status"],
        "evaluation_status": comparison["evaluation_status"],
        "scientific_quality": comparison["scientific_quality"],
        "gate_status": gate["status"],
        "candidate_count": gate["candidate_count"],
        "selection": comparison["frozen_selection"],
        "mean_metrics": {name: comparison["mean_metrics"][name] for name in ["center_25000", *(f"rank{i}_member_{m:02d}_25000" for i, m in enumerate(SELECTIONS[bits], 1))]},
    }

source_hashes = {}
for name, expected in EXPECTED_SOURCE_HASHES.items():
    matches = list((REPO / "quantization_aware_cpa/qacpa").rglob(name))
    if not matches:
        raise RuntimeError(f"protected source missing: {name}")
    actual = sha(matches[0])
    source_hashes[name] = {"path": str(matches[0]), "sha256": actual, "expected_registered_sha256": expected, "unchanged": actual == expected}

audit = {
    "task_id": "CPA_SIGNED_BITS7654_RECOVERY_20260918",
    "status": "COMPLETED_ARTIFACTS_VERIFIED",
    "wrapper_exit_code": 0,
    "wrapper_summary_status": summary.get("status"),
    "wrapper_error_preserved": {"error_type": summary.get("error_type"), "error": summary.get("error")},
    "wrapper_error_interpretation": "The raw summary retains the late bit-6 registration-byte error. All four saved public gates, 13-set x-only comparisons, and corrected figure manifests independently pass; no optimizer rerun was performed.",
    "bits": comparisons,
    "candidate_hashes": protected_candidate_hashes,
    "corrected_figures": figure_records,
    "source_hashes": source_hashes,
    "optimizer_steps": {"7": 0, "6": 125000, "5": 125000, "4": 125000},
    "public_gate_before_private_evaluation": True,
    "private_x_only": True,
    "labels_consumed": False,
    "actual_image_cell_feasibility": "NOT_EVALUATED",
    "parent_artifacts_preserved": True,
    "original_failure_evidence_preserved": True,
    "active_handoff_before": str(ROOT / "active_handoff_before_finalization.md"),
    "finalized_at": datetime.now(timezone.utc).isoformat(),
}
write_json(ROOT / "final_audit.json", audit)

report_path = ROOT / "report.md"
report = report_path.read_text(encoding="utf-8")
report += """

## Post-exit artifact audit

The callback reported wrapper exit code 0. The raw `execution_summary.json` retains `error_type=RecoveryError` with `error=bit 6 registration bytes changed` from a late registration transition. That evidence is preserved. A separate final audit verified all four saved comparison files as `COMPLETED/EVALUATED`, each with `PUBLIC_GATE_PASSED`, `private_loaded=false`, 13 authenticated candidates and the registered stored Hungarian correspondence. No optimizer or replay step was rerun.

Corrected baseline-inclusive contact sheets were completed for bits 5 and 4 under their existing evaluation directories using saved candidates and stored matching. The combined available figure manifest now lists bits 8, 7, 6, 5 and 4 with no pending bits. The actual-image cell remains `NOT_EVALUATED`.

Final audit: `.ai/execution-results/CPA_SIGNED_BITS7654_RECOVERY_20260918/final_audit.json`.
"""
report_path.write_text(report, encoding="utf-8", newline="\n")

updated = active_text.replace("STATUS: RUNNING", "STATUS: COMPLETED", 1).replace("ACTIVE_CLAIM_STATUS: RUNNING", "ACTIVE_CLAIM_STATUS: COMPLETED", 1)
updated += "\n\nFINAL_AUDIT_STATUS: COMPLETED_ARTIFACTS_VERIFIED\nFINAL_AUDIT_PATH: .ai/execution-results/CPA_SIGNED_BITS7654_RECOVERY_20260918/final_audit.json\nWRAPPER_ERROR_PRESERVED: bit 6 registration bytes changed after all required stages; no rerun\n"
ACTIVE.write_text(updated, encoding="utf-8", newline="\n")
