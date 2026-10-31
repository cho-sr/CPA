"""CPU-only 16:9 slide export for the frozen screen3 comparisons.

This sidecar deliberately delegates public authentication, private-x loading,
and stored Hungarian alignment to the completed baseline figure renderer.  It
only lays out already-saved tensors; it does not evaluate a model, recompute a
metric, or select a candidate.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
from typing import Any

import torch


TASK_ID = "CPA_SCREEN3_SLIDE_PNG_876_20260918"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "quantization_aware_cpa/results_private/slide_png_8_7_6_20260918"
BASELINE_RENDERER_PATH = (
    REPO_ROOT
    / ".ai/execution-results/CPA_SCREEN3_BASELINE_FIGURE_REPAIR_20260918/render_baseline_figures.py"
)
EVALUATIONS = {
    8: REPO_ROOT / "quantization_aware_cpa/results_private/signed_int8_screen3_repair_B16_T10_20260918/evaluation_final_20260918",
    7: REPO_ROOT / "quantization_aware_cpa/results_private/signed_int7_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918",
    6: REPO_ROOT / "quantization_aware_cpa/results_private/signed_int6_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918",
}
EXPECTED_SELECTIONS = {8: [1, 7, 5], 7: [0, 5, 3], 6: [1, 3, 4]}
EXPECTED_DIMENSIONS = (3840, 2160)
EXPECTED_DPI = 240
STEPS = 25000
FOOTER = "Rows matched by pixel error; source identity is not verified"


class SlideExportError(RuntimeError):
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


def load_baseline_renderer():
    if not BASELINE_RENDERER_PATH.is_file():
        raise SlideExportError(f"missing authenticated renderer: {BASELINE_RENDERER_PATH}")
    spec = importlib.util.spec_from_file_location("screen3_baseline_renderer", BASELINE_RENDERER_PATH)
    if spec is None or spec.loader is None:
        raise SlideExportError("could not load the baseline renderer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def page_ranges() -> list[tuple[int, int]]:
    return [(start, start + 4) for start in range(0, 16, 4)]


def role_names(selection: list[int]) -> list[str]:
    if len(selection) != 3 or len(set(selection)) != 3 or any(member not in range(8) for member in selection):
        raise SlideExportError(f"invalid frozen selection: {selection}")
    return [
        "Ground truth",
        "Naive dequantization baseline",
        f"Public rank 1 | member {selection[0]:02d}",
        f"Public rank 2 | member {selection[1]:02d}",
        f"Public rank 3 | member {selection[2]:02d}",
    ]


def candidate_names(selection: list[int]) -> list[str]:
    return [
        "center_25000",
        *(f"rank{rank}_member_{member:02d}_25000" for rank, member in enumerate(selection, 1)),
    ]


def _short_role_titles(selection: list[int], bits: int) -> list[str]:
    return [
        "Ground\ntruth",
        f"Naive\nbaseline\n{bits}-bit",
        f"Rank 1\nmember {selection[0]:02d}",
        f"Rank 2\nmember {selection[1]:02d}",
        f"Rank 3\nmember {selection[2]:02d}",
    ]


def _to_display_tensor(truth_x: torch.Tensor) -> torch.Tensor:
    mean = truth_x.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = truth_x.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    return (truth_x * std + mean).clamp(0, 1).detach().cpu()


def _load_final_images(comparison: dict[str, Any], paths: dict[str, Path], renderer: Any) -> tuple[dict[str, torch.Tensor], list[int], list[str]]:
    selection = [int(value) for value in comparison["frozen_selection"]]
    names = candidate_names(selection)
    if selection != EXPECTED_SELECTIONS[int(comparison["bits"])]:
        raise SlideExportError(f"unexpected selection in comparison: {selection}")
    candidates: dict[str, torch.Tensor] = {}
    metrics = comparison["image_metrics"]
    for name in names:
        payload = torch.load(paths[name], map_location="cpu", weights_only=True)
        pixels = payload["pixels"]
        if tuple(pixels.shape) != (16, 3, 224, 224) or pixels.dtype != torch.float32:
            raise SlideExportError(f"candidate tensor schema failed: {name}")
        candidates[name] = renderer.aligned_candidate(pixels, metrics[name]["per_image"])
    return candidates, selection, names


def _draw_page(
    output: Path,
    bits: int,
    truth: torch.Tensor,
    candidates: dict[str, torch.Tensor],
    names: list[str],
    selection: list[int],
    row_start: int | None,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = _short_role_titles(selection, bits)
    displayed = [truth, *(candidates[name] for name in names)]
    if row_start is None:
        rows = list(range(16))
        page_label = "Examples 01–16"
        figure = plt.figure(figsize=(16, 9), dpi=EXPECTED_DPI)
        outer = figure.add_gridspec(
            1, 2, left=0.035, right=0.995, top=0.875, bottom=0.105, wspace=0.035
        )
        for panel, panel_start in enumerate((0, 8)):
            grid = outer[panel].subgridspec(8, 5, wspace=0.025, hspace=0.06)
            for local_row, global_row in enumerate(range(panel_start, panel_start + 8)):
                for column, images in enumerate(displayed):
                    axis = figure.add_subplot(grid[local_row, column])
                    axis.imshow(images[global_row].permute(1, 2, 0).numpy())
                    axis.set_xticks([])
                    axis.set_yticks([])
                    for spine in axis.spines.values():
                        spine.set_visible(False)
                    if local_row == 0:
                        axis.set_title(titles[column], fontsize=7.5, pad=2, linespacing=0.9)
                    if column == 0:
                        axis.text(
                            -0.08,
                            0.5,
                            f"Ex {global_row + 1:02d}",
                            transform=axis.transAxes,
                            ha="right",
                            va="center",
                            fontsize=7.2,
                        )
            figure.text(
                0.267 if panel == 0 else 0.755,
                0.895,
                f"Examples {panel_start + 1:02d}–{panel_start + 8:02d}",
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
            )
    else:
        rows = list(range(row_start, row_start + 4))
        page_label = f"Examples {row_start + 1:02d}–{row_start + 4:02d}"
        figure = plt.figure(figsize=(16, 9), dpi=EXPECTED_DPI)
        grid = figure.add_gridspec(4, 5, left=0.025, right=0.995, top=0.835, bottom=0.105, wspace=0.025, hspace=0.12)
        for local_row, global_row in enumerate(rows):
            for column, images in enumerate(displayed):
                axis = figure.add_subplot(grid[local_row, column])
                axis.imshow(images[global_row].permute(1, 2, 0).numpy())
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
                if local_row == 0:
                    axis.set_title(titles[column], fontsize=14, pad=5, linespacing=0.95)
                if column == 0:
                    axis.text(
                        -0.055,
                        0.5,
                        f"Ex {global_row + 1:02d}",
                        transform=axis.transAxes,
                        ha="right",
                        va="center",
                        fontsize=12,
                    )
    figure.patch.set_facecolor("white")
    figure.suptitle(f"INT{bits} screen3 | {STEPS:,} steps | {page_label}", fontsize=18, y=0.965, weight="bold")
    figure.text(0.5, 0.035, FOOTER, ha="center", va="center", fontsize=10, color="#444444")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output, dpi=EXPECTED_DPI, facecolor="white")
    finally:
        plt.close(figure)
    from PIL import Image

    with Image.open(output) as image:
        dimensions = list(image.size)
    if tuple(dimensions) != EXPECTED_DIMENSIONS:
        raise SlideExportError(f"wrong PNG dimensions for {output}: {dimensions}")
    return {
        "file": output.name,
        "sha256": sha256(output),
        "dimensions": dimensions,
        "row_range": [rows[0] + 1, rows[-1] + 1],
        "roles": role_names(selection),
        "candidate_names": names,
        "matching_reused": True,
        "metrics_recomputed": False,
    }


def _metric_table(comparison: dict[str, Any], selection: list[int]) -> str:
    lines = [
        "| Set | MSE | PSNR dB | SSIM | LPIPS |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in candidate_names(selection):
        row = comparison["mean_metrics"][name]
        lines.append(
            f"| {name} | {row['mean_mse']:.12g} | {row['mean_psnr_db']:.12g} | "
            f"{row['mean_ssim']:.12g} | {row['mean_lpips']:.12g} |"
        )
    return "\n".join(lines)


def main() -> int:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise SlideExportError(f"refusing to overwrite non-empty output root: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    renderer = load_baseline_renderer()
    authenticated: dict[int, dict[str, Any]] = {}
    loaded_paths: dict[int, dict[str, Path]] = {}
    public_roots: dict[int, list[str]] = {}
    comparison_hashes: dict[int, str] = {}

    # Authenticate every public set before loading the registered private x.
    for bits in (8, 7, 6):
        evaluation_dir = EVALUATIONS[bits]
        comparison_path = evaluation_dir / "comparison.json"
        comparison, paths, roots = renderer.authenticate_comparison(comparison_path, bits)
        comparison["bits"] = bits
        if [int(value) for value in comparison["frozen_selection"]] != EXPECTED_SELECTIONS[bits]:
            raise SlideExportError(f"frozen selection mismatch for INT{bits}")
        authenticated[bits] = comparison
        loaded_paths[bits] = paths
        public_roots[bits] = [path.as_posix() for path in roots]
        comparison_hashes[bits] = renderer.sha256(comparison_path)

    truth_x, private_meta = renderer.load_private_x(authenticated[8])
    truth = _to_display_tensor(truth_x)
    all_bits: dict[str, Any] = {}
    png_paths: list[Path] = []
    for bits in (8, 7, 6):
        comparison = authenticated[bits]
        candidates, selection, names = _load_final_images(comparison, loaded_paths[bits], renderer)
        prefix = f"int{bits}_25000"
        overview = _draw_page(OUTPUT_ROOT / f"{prefix}_overview.png", bits, truth, candidates, names, selection, None)
        pages = [
            _draw_page(OUTPUT_ROOT / f"{prefix}_page{index:02d}.png", bits, truth, candidates, names, selection, start)
            for index, (start, _) in enumerate(page_ranges(), 1)
        ]
        png_paths.extend(OUTPUT_ROOT / entry["file"] for entry in [overview, *pages])
        all_bits[str(bits)] = {
            "comparison_path": (EVALUATIONS[bits] / "comparison.json").as_posix(),
            "comparison_sha256": comparison_hashes[bits],
            "public_gate_status": comparison["public_gate"]["status"],
            "public_candidate_count": comparison["public_gate"]["candidate_count"],
            "public_roots": public_roots[bits],
            "frozen_selection": selection,
            "roles": role_names(selection),
            "candidate_names": names,
            "candidate_hashes": {name: comparison["candidate_hashes"][name] for name in names},
            "overview": overview,
            "detail_pages": pages,
            "existing_mean_metrics": {
                name: comparison["mean_metrics"][name] for name in names
            },
        }

    index_lines = [
        "# Screen3 slide PNG export: INT8 / INT7 / INT6",
        "",
        "Frozen final comparisons at 25,000 Direct FIA steps. The overview has all 16 examples; each bit also has four detail pages covering four consecutive examples.",
        "",
        f"Interpretation: {FOOTER}.",
        "",
    ]
    for bits in (8, 7, 6):
        entry = all_bits[str(bits)]
        index_lines += [
            f"## INT{bits}",
            "",
            f"Public selection: {entry['frozen_selection']} (original member IDs).",
            "",
            f"Overview: `{entry['overview']['file']}`",
            "",
            "Detail pages: " + ", ".join(f"`{page['file']}`" for page in entry["detail_pages"]),
            "",
            _metric_table(authenticated[bits], entry["frozen_selection"]),
            "",
        ]
    index_path = OUTPUT_ROOT / "index.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8", newline="\n")

    zip_path = OUTPUT_ROOT / "slide_png_8_7_6_20260918.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in [*png_paths, index_path]:
            archive.write(path, arcname=path.name)
    with zipfile.ZipFile(zip_path) as archive:
        expected_members = {path.name for path in [*png_paths, index_path]}
        if set(archive.namelist()) != expected_members:
            raise SlideExportError("ZIP member set does not match the 15 PNGs and index")

    manifest = {
        "task_id": TASK_ID,
        "status": "COMPLETED",
        "bits": [8, 7, 6],
        "dimensions": list(EXPECTED_DIMENSIONS),
        "dpi": EXPECTED_DPI,
        "optimizer_steps": 0,
        "metrics_recomputed": False,
        "matching_recomputed": False,
        "private_x_only": True,
        "labels_consumed": False,
        "private_meta": private_meta,
        "output_root": OUTPUT_ROOT.as_posix(),
        "bits_detail": all_bits,
        "zip": {
            "file": zip_path.name,
            "sha256": sha256(zip_path),
            "members": [path.name for path in [*png_paths, index_path]],
        },
        "original_files_preserved": True,
        "active_parent_handoff_modified": False,
    }
    write_json(OUTPUT_ROOT / "slide_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SlideExportError as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)
