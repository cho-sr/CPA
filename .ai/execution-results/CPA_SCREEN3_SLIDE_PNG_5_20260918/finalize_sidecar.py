from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OUTPUT = REPO / "quantization_aware_cpa/results_private/slide_png_5_20260918"
HANDOFF = REPO / ".ai/relay-runs/executor/CPA_SCREEN3_SLIDE_PNG_5_20260918_v1/handoff.md"
BASELINE = REPO / ".ai/execution-results/CPA_SCREEN3_BASELINE_FIGURE_REPAIR_20260918/render_baseline_figures.py"
SLIDE = REPO / ".ai/execution-results/CPA_SCREEN3_SLIDE_PNG_876_20260918/render_slide_pngs.py"
EVALUATION = REPO / (
    "quantization_aware_cpa/results_private/"
    "signed_int5_randomized8_screen3_fia_B16_T10_20260918_sweep/"
    "evaluation_recovery_20260918"
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


comparison_path = EVALUATION / "comparison.json"
comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
manifest_path = OUTPUT / "slide_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest["comparison_sha256"] != sha(comparison_path):
    raise RuntimeError("comparison hash changed after rendering")
if manifest["frozen_selection"] != [5, 7, 6] or manifest["public_gate_status"] != "PUBLIC_GATE_PASSED":
    raise RuntimeError("INT5 manifest gate/selection mismatch")

candidate_hashes = {
    name: {"path": str(Path(path).resolve()), "sha256": sha(Path(path).resolve())}
    for name, path in comparison["candidate_paths"].items()
}
output_hashes = {}
for path in sorted(OUTPUT.glob("*.png")):
    output_hashes[path.name] = {"sha256": sha(path), "bytes": path.stat().st_size}
for name in ("index.md", "slide_manifest.json", "slide_png_5_20260918.zip"):
    path = OUTPUT / name
    output_hashes[name] = {"sha256": sha(path), "bytes": path.stat().st_size}

integrity = {
    "task_id": "CPA_SCREEN3_SLIDE_PNG_5_20260918",
    "status": "COMPLETED",
    "public_authentication": "completed before private x loading",
    "private_consumption": "registered x only; labels, z and true updates not consumed",
    "source_hashes": {
        "baseline_renderer": {"path": str(BASELINE), "sha256": sha(BASELINE)},
        "slide_layout_renderer": {"path": str(SLIDE), "sha256": sha(SLIDE)},
        "comparison_json": {"path": str(comparison_path), "sha256": sha(comparison_path)},
        "candidate_files": candidate_hashes,
    },
    "output_hashes": output_hashes,
    "dimensions": [3840, 2160],
    "png_count": len(list(OUTPUT.glob("*.png"))),
    "rows": [[1, 16], [1, 4], [5, 8], [9, 12], [13, 16]],
    "roles": [
        "Ground truth",
        "Naive dequantization baseline",
        "Public rank 1 | member 05",
        "Public rank 2 | member 07",
        "Public rank 3 | member 06",
    ],
    "optimizer_steps": 0,
    "model_inference": False,
    "metrics_recomputed": False,
    "matching_recomputed": False,
    "active_parent_handoff_modified": False,
}
write_json(ROOT / "integrity.json", integrity)
shutil.copyfile(HANDOFF, ROOT / "immutable_sidecar_snapshot.md")

claim = {
    "task_id": "CPA_SCREEN3_SLIDE_PNG_5_20260918",
    "status": "COMPLETED",
    "assigned_role": "CODEX_EXECUTOR",
    "executor_thread": "01a0a799-f1ce-7b31-8c46-b4ddef40aeab",
    "executor_pid": 14236,
    "runtime": "Luna/xhigh",
    "handoff_sha256": sha(HANDOFF),
    "parent_task_id": "CPA_SIGNED_BITS7654_RECOVERY_20260918",
    "active_parent_not_replaced": True,
    "result_report": str(ROOT / "report.md"),
    "private_output_root": str(OUTPUT),
    "planner_thread": "01a0a799-d62c-7ab0-9c2a-628acf6d2c95",
    "originating_user_thread": "01a0a7ab-fa58-7d22-ac52-3247b1dd2255",
}
write_json(ROOT / "claim.json", claim)

report_lines = [
    "# CPA_SCREEN3_SLIDE_PNG_5_20260918",
    "",
    "Status: COMPLETED.",
    "",
    "Created the isolated INT5 slide export from the completed, evaluated and PUBLIC_GATE_PASSED frozen 25,000-step comparison. The output contains one all-16 overview and four detail pages covering 01–04, 05–08, 09–12 and 13–16. Every PNG is exactly 3840x2160.",
    "",
    "The adapter reused the strict public authentication, private-x loading, stored per-image Hungarian alignment and proven 16:9 layout helpers. It did not invoke the hardcoded INT8/7/6 main, modify prior outputs, recompute metrics/matching, run model inference or take optimizer steps. Labels, z and true updates were not consumed.",
    "",
    "## Output",
    "",
    f"- INT5 overview: `{OUTPUT / 'int5_25000_overview.png'}`",
    f"- ZIP: `{OUTPUT / 'slide_png_5_20260918.zip'}`",
    f"- Detail pages and index: `{OUTPUT}`",
    "",
    "Selection is [5, 7, 6], shown as original member IDs 05, 07 and 06. Every page uses Ground truth | same-bit naive baseline | public rank1 | rank2 | rank3, with the existing interpretation footer.",
    "",
    "## Validation",
    "",
    "- Focused INT5 adapter regression: 2 passed.",
    "- Actual PNG validation: 5 files, all 3840x2160.",
    "- Row coverage: overview 01–16; detail pages 01–04, 05–08, 09–12 and 13–16.",
    "- ZIP validation: exactly the five PNGs and index.md.",
    "- Visual inspection: INT5 overview and detail page 01 show five readable columns, member IDs 05/07/06, and unclipped headings.",
    "- Existing comparison, candidate, metric, registration and active parent handoff bytes were preserved; hashes are recorded in integrity.json and slide_manifest.json.",
    "",
    "Originating user thread: 01a0a7ab-fa58-7d22-ac52-3247b1dd2255.",
]
(ROOT / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")
