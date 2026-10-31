"""Isolated INT5 adapter for the proven screen3 slide renderer.

The shared INT8/7/6 entry point is intentionally not called because its
top-level mapping is restricted to those completed exports.  This adapter
imports its tested layout helpers and the strict baseline authentication
helper, then supplies the INT5 source, selection, and output root locally.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
from typing import Any

import torch


TASK_ID = "CPA_SCREEN3_SLIDE_PNG_5_20260918"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "quantization_aware_cpa/results_private/slide_png_5_20260918"
EVALUATION_DIR = REPO_ROOT / (
    "quantization_aware_cpa/results_private/"
    "signed_int5_randomized8_screen3_fia_B16_T10_20260918_sweep/"
    "evaluation_recovery_20260918"
)
BASELINE_RENDERER_PATH = REPO_ROOT / (
    ".ai/execution-results/CPA_SCREEN3_BASELINE_FIGURE_REPAIR_20260918/"
    "render_baseline_figures.py"
)
SLIDE_RENDERER_PATH = REPO_ROOT / (
    ".ai/execution-results/CPA_SCREEN3_SLIDE_PNG_876_20260918/"
    "render_slide_pngs.py"
)
SELECTION = [5, 7, 6]
EXPECTED_DIMENSIONS = (3840, 2160)
EXPECTED_DPI = 240


class Int5SlideError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def import_module(path: Path, name: str):
    if not path.is_file():
        raise Int5SlideError(f"missing reusable renderer: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Int5SlideError(f"could not import renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_int5_candidates(comparison: dict[str, Any], paths: dict[str, Path], baseline: Any) -> tuple[dict[str, torch.Tensor], list[str]]:
    selection = [int(value) for value in comparison.get("frozen_selection", [])]
    if selection != SELECTION:
        raise Int5SlideError(f"unexpected INT5 frozen selection: {selection}")
    names = [
        "center_25000",
        "rank1_member_05_25000",
        "rank2_member_07_25000",
        "rank3_member_06_25000",
    ]
    metrics = comparison["image_metrics"]
    loaded: dict[str, torch.Tensor] = {}
    for name in names:
        if name not in paths or name not in metrics:
            raise Int5SlideError(f"missing authenticated INT5 final candidate: {name}")
        payload = torch.load(paths[name], map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != {"pixels"}:
            raise Int5SlideError(f"candidate payload schema failed: {name}")
        pixels = payload["pixels"]
        if (not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.float32
                or tuple(pixels.shape) != (16, 3, 224, 224)
                or not bool(torch.isfinite(pixels).all())):
            raise Int5SlideError(f"candidate tensor schema failed: {name}")
        loaded[name] = baseline.aligned_candidate(pixels, metrics[name]["per_image"])
    return loaded, names


def main() -> int:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise Int5SlideError(f"refusing to overwrite non-empty output root: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    slide = import_module(SLIDE_RENDERER_PATH, "screen3_slide_renderer_for_int5")
    baseline = import_module(BASELINE_RENDERER_PATH, "screen3_baseline_renderer_for_int5")
    comparison_path = EVALUATION_DIR / "comparison.json"
    comparison, paths, public_roots = baseline.authenticate_comparison(comparison_path, 5)
    if (comparison.get("status") != "COMPLETED" or comparison.get("evaluation_status") != "COMPLETED"
            or comparison.get("scientific_quality") != "EVALUATED"):
        raise Int5SlideError("INT5 comparison is not completed and evaluated")
    # Public authentication is complete before this registered private bundle load.
    truth_x, private_meta = baseline.load_private_x(comparison)
    truth = slide._to_display_tensor(truth_x)
    candidates, names = load_int5_candidates(comparison, paths, baseline)

    overview = slide._draw_page(
        OUTPUT_ROOT / "int5_25000_overview.png",
        5,
        truth,
        candidates,
        names,
        SELECTION,
        None,
    )
    pages = []
    for page_number, (start, _) in enumerate(slide.page_ranges(), 1):
        pages.append(
            slide._draw_page(
                OUTPUT_ROOT / f"int5_25000_page{page_number:02d}.png",
                5,
                truth,
                candidates,
                names,
                SELECTION,
                start,
            )
        )
    pngs = [OUTPUT_ROOT / item["file"] for item in [overview, *pages]]
    index_lines = [
        "# Screen3 slide PNG export: INT5",
        "",
        "Frozen final comparison at 25,000 Direct FIA steps. The overview has all 16 examples; four detail pages cover 01–04, 05–08, 09–12 and 13–16.",
        "",
        "Roles: Ground truth | same-bit naive dequantization baseline | public rank 1 member 05 | public rank 2 member 07 | public rank 3 member 06.",
        "",
        "Interpretation: Rows matched by pixel error; source identity is not verified.",
        "",
        "Overview: `int5_25000_overview.png`",
        "",
        "Detail pages: " + ", ".join(f"`{item['file']}`" for item in pages),
        "",
        slide._metric_table(comparison, SELECTION),
    ]
    index_path = OUTPUT_ROOT / "index.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8", newline="\n")
    zip_path = OUTPUT_ROOT / "slide_png_5_20260918.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in [*pngs, index_path]:
            archive.write(path, arcname=path.name)
    with zipfile.ZipFile(zip_path) as archive:
        expected = {path.name for path in [*pngs, index_path]}
        if set(archive.namelist()) != expected:
            raise Int5SlideError("ZIP member set is not the five PNGs and index.md")

    manifest = {
        "task_id": TASK_ID,
        "status": "COMPLETED",
        "bit": 5,
        "dimensions": list(EXPECTED_DIMENSIONS),
        "dpi": EXPECTED_DPI,
        "optimizer_steps": 0,
        "model_inference": False,
        "metrics_recomputed": False,
        "matching_recomputed": False,
        "private_x_only": True,
        "labels_consumed": False,
        "private_meta": private_meta,
        "comparison_path": comparison_path.as_posix(),
        "comparison_sha256": sha256(comparison_path),
        "public_gate_status": comparison["public_gate"]["status"],
        "public_candidate_count": comparison["public_gate"]["candidate_count"],
        "public_roots": [path.as_posix() for path in public_roots],
        "frozen_selection": SELECTION,
        "roles": slide.role_names(SELECTION),
        "candidate_names": names,
        "candidate_hashes": {name: comparison["candidate_hashes"][name] for name in names},
        "overview": overview,
        "detail_pages": pages,
        "zip": {"file": zip_path.name, "sha256": sha256(zip_path), "members": [path.name for path in [*pngs, index_path]]},
        "original_files_preserved": True,
        "active_parent_handoff_modified": False,
    }
    write_json(OUTPUT_ROOT / "slide_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Int5SlideError as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)
