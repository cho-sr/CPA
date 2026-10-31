from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OUTPUT = REPO / "quantization_aware_cpa/results_private/slide_png_8_7_6_20260918"
HANDOFF = REPO / ".ai/relay-runs/executor/CPA_SCREEN3_SLIDE_PNG_876_20260918_v1/handoff.md"
BASELINE = REPO / ".ai/execution-results/CPA_SCREEN3_BASELINE_FIGURE_REPAIR_20260918/render_baseline_figures.py"
EVALS = {
    8: REPO / "quantization_aware_cpa/results_private/signed_int8_screen3_repair_B16_T10_20260918/evaluation_final_20260918",
    7: REPO / "quantization_aware_cpa/results_private/signed_int7_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918",
    6: REPO / "quantization_aware_cpa/results_private/signed_int6_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


manifest_path = OUTPUT / "slide_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
comparison_hashes = {bits: sha(EVALS[bits] / "comparison.json") for bits in EVALS}
if any(manifest["bits_detail"][str(bits)]["comparison_sha256"] != value for bits, value in comparison_hashes.items()):
    raise RuntimeError("comparison hash changed after rendering")

candidate_hashes = {}
for bits, evaluation in EVALS.items():
    comparison = json.loads((evaluation / "comparison.json").read_text(encoding="utf-8"))
    candidate_hashes[str(bits)] = {
        name: {
            "path": str(Path(path).resolve()),
            "sha256": sha(Path(path).resolve()),
        }
        for name, path in comparison["candidate_paths"].items()
    }

output_hashes = {}
for path in sorted(OUTPUT.glob("*.png")):
    output_hashes[path.name] = {"sha256": sha(path), "bytes": path.stat().st_size}
for name in ("index.md", "slide_manifest.json", "slide_png_8_7_6_20260918.zip"):
    path = OUTPUT / name
    output_hashes[name] = {"sha256": sha(path), "bytes": path.stat().st_size}

integrity = {
    "task_id": "CPA_SCREEN3_SLIDE_PNG_876_20260918",
    "status": "COMPLETED",
    "public_authentication": "completed before private x loading",
    "private_consumption": "registered x only; labels, z and true updates not consumed",
    "source_hashes": {
        "baseline_renderer": {"path": str(BASELINE), "sha256": sha(BASELINE)},
        "comparison_json": {str(bits): {"path": str(EVALS[bits] / "comparison.json"), "sha256": value} for bits, value in comparison_hashes.items()},
        "candidate_files": candidate_hashes,
    },
    "output_hashes": output_hashes,
    "dimensions": [3840, 2160],
    "png_count": len(list(OUTPUT.glob("*.png"))),
    "optimizer_steps": 0,
    "metrics_recomputed": False,
    "matching_recomputed": False,
    "active_parent_handoff_modified": False,
}
integrity_path = ROOT / "integrity.json"
write_json(integrity_path, integrity)

shutil.copyfile(HANDOFF, ROOT / "immutable_sidecar_snapshot.md")
claim = {
    "task_id": "CPA_SCREEN3_SLIDE_PNG_876_20260918",
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

overview_paths = {bits: OUTPUT / f"int{bits}_25000_overview.png" for bits in (8, 7, 6)}
report_lines = [
    "# CPA_SCREEN3_SLIDE_PNG_876_20260918",
    "",
    "Status: COMPLETED.",
    "",
    "Created an isolated CPU-only 16:9 presentation export from the authenticated frozen 25,000-step comparisons for INT8, INT7 and INT6. Each bit has one all-16 overview and four detail pages covering rows 01–04, 05–08, 09–12 and 13–16. Every page is exactly 3840x2160 PNG.",
    "",
    "The renderer authenticated public candidates and registrations before loading the registered private x bundle, reused the stored per-image Hungarian correspondence, and did not recompute metrics or matching. No optimizer/model inference ran; optimizer_steps=0. Labels, z and true updates were not consumed.",
    "",
    "## Output",
    "",
    f"- INT8 overview: `{overview_paths[8]}`",
    f"- INT7 overview: `{overview_paths[7]}`",
    f"- INT6 overview: `{overview_paths[6]}`",
    f"- ZIP: `{OUTPUT / 'slide_png_8_7_6_20260918.zip'}`",
    f"- Detail pages and index: `{OUTPUT}`",
    "",
    "Selections preserve original member IDs: INT8 [1, 7, 5], INT7 [0, 5, 3], INT6 [1, 3, 4]. Every page uses Ground truth | same-bit naive baseline | public rank1 | rank2 | rank3. The footer states: Rows matched by pixel error; source identity is not verified.",
    "",
    "## Validation",
    "",
    "- Focused slide-layout regression: 3 passed.",
    "- Actual PNG validation: 15 files, all 3840x2160.",
    "- ZIP validation: exactly the 15 PNGs and index.md.",
    "- Visual inspection: INT6 overview and INT6 detail page 01 show five columns, member IDs 01/03/04, and rows 01–16 / 01–04 respectively without overlap.",
    "- Existing comparison, candidate, metric, figure, registration and active parent handoff bytes were preserved; hashes are recorded in integrity.json and slide_manifest.json.",
    "",
    "Originating user thread: 01a0a7ab-fa58-7d22-ac52-3247b1dd2255.",
]
(ROOT / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")
