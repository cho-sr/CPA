"""Audit and render the completed raw-update-row FIA experiment.

This module performs no source separation or image optimization. It reads all
completed outputs before loading originals for reporting, and preserves every
224 x 224 reconstruction pixel in the PNG panels.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

PROJECT = Path(__file__).resolve().parents[1]
IMAGE_IDENTITY_FIELDS = ("global_index", "batch_index", "source_index", "image_path")
REQUIRED_BATCH_OUTPUTS = {
    "sources.pt", "target_z.pt", "source_order_rec.pt", "rec.pt",
    "per_image_metrics.csv", "fia_log.json", "target_metadata.json",
}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT / path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("An empty report table is not a completed experiment")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _verify_hashes(folder: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        path = folder / name
        if file_hash(path) != expected:
            raise RuntimeError(f"Frozen file changed: {path}")


def _validate_rows(rows: list[dict[str, Any]], n_images: int = 80) -> None:
    if [int(row["global_index"]) for row in rows] != list(range(n_images)):
        raise RuntimeError("Expected all images in global-index order, without duplicates")
    for row in rows:
        index = int(row["global_index"])
        if int(row["batch_index"]) != index // 16 or int(row["source_index"]) != index % 16:
            raise RuntimeError("Global and batch image indices disagree")
        if not row["image_path"]:
            raise RuntimeError("Missing source image identity")
        for metric in ("lpips", "psnr", "ssim", "source_abs_cosine"):
            if not np.isfinite(float(row[metric])):
                raise RuntimeError(f"Non-finite {metric} at image {index}")


def _validate_paired_identity(
    rows: list[dict[str, Any]], reference: list[dict[str, Any]],
) -> None:
    if len(rows) != len(reference):
        raise RuntimeError("Reference and raw-row image counts differ")
    for row, ref in zip(rows, reference):
        for key in IMAGE_IDENTITY_FIELDS:
            if str(row[key]) != str(ref[key]):
                raise RuntimeError(f"Reference image identity mismatch: {key}, g{row['global_index']}")
        for key in ("dataset_index", "label"):
            if key in row and key in ref and str(row[key]) != str(ref[key]):
                raise RuntimeError(f"Reference image identity mismatch: {key}, g{row['global_index']}")


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        **{f"{key}_mean": float(np.mean([float(row[key]) for row in rows]))
           for key in ("lpips", "psnr", "ssim")},
        "target_cos09_count": sum(float(row["source_abs_cosine"]) > .9 for row in rows),
    }


def _rgb(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    if tensor.shape != (3, 224, 224) or not bool(torch.isfinite(tensor).all()):
        raise ValueError("Expected finite RGB image tensors at native 224 x 224 resolution")
    return (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_grid(
    images: dict[str, torch.Tensor], rows: list[dict[str, Any]], indices: list[int],
    path: Path, columns: int, title: str,
) -> dict[str, Any]:
    """Render unresized tiles and verify saved PNG crops against their tensors."""
    if not indices or columns < 1 or len(set(indices)) != len(indices):
        raise ValueError("Grid needs unique image indices and at least one column")
    if any(index < 0 or index >= len(rows) for index in indices):
        raise ValueError("Grid image index is out of range")
    if len(images) not in (2, 3) or any(len(value) != len(rows) for value in images.values()):
        raise ValueError("Expected two or three complete image collections")
    tile, gap, header, labels = 224, 16, 120, 60
    panel_count = len(images)
    block_width = panel_count * tile + (panel_count - 1) * gap
    nrows = (len(indices) + columns - 1) // columns
    width = gap + columns * (block_width + gap)
    height = header + nrows * (tile + labels + gap)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font, small = _font(22), _font(19)
    draw.text((gap, 10), title, font=_font(28), fill="#0f172a")
    draw.text((gap, 47), "VGG16 | FedAvg n=16, T=10 | same per-row 4-bit-interval dither observations; not an integer codec",
              font=font, fill="#334155")
    selection = "all 80, original index order" if indices == list(range(80)) else "random 12, seed=42 fixed before results"
    draw.text((gap, 80), f"{selection} | not cherry-picked | raw rows: no CPA, debias, MP, LS-A or embedding refinement",
              font=small, fill="#475569")
    placements = []
    for position, index in enumerate(indices):
        left = gap + (position % columns) * (block_width + gap)
        top = header + (position // columns) * (tile + labels + gap)
        row = rows[index]
        draw.text((left, top), f"g{index:02d} | raw-row LPIPS {float(row['lpips']):.3f} | source cos {float(row['source_abs_cosine']):.3f}",
                  font=small, fill="#0f172a")
        for panel, (label, tensors) in enumerate(images.items()):
            x, y = left + panel * (tile + gap), top + labels
            draw.text((x, top + 28), label, font=small, fill="#0f172a")
            rgb = _rgb(tensors[index])
            canvas.paste(Image.fromarray(rgb), (x, y))
            placements.append({
                "global_index": index, "panel": label, "image_path": str(row["image_path"]),
                "box": [x, y, x + tile, y + tile],
                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    with Image.open(path) as saved:
        if saved.size != (width, height) or saved.mode != "RGB":
            raise RuntimeError("Saved grid dimensions or color mode changed")
        for placement in placements:
            array = np.asarray(saved.crop(tuple(placement["box"])))
            digest = hashlib.sha256(array.tobytes()).hexdigest()
            if digest != placement["rgb_sha256"]:
                raise RuntimeError("PNG tile does not match its source pixels")
    return {
        "file": path.name, "width": width, "height": height, "tile_size": tile,
        "image_indices": indices, "panels": list(images), "tile_placements": placements,
        "pixel_verification": "all saved PNG tiles match their source RGB tensors exactly",
        "file_sha256": file_hash(path),
    }


def build_report(cfg: dict[str, Any], manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    """Produce the final all80 report only after every registered batch passes."""
    output = Path(output)
    if cfg["batch_indices"] != list(range(5)) or cfg["n_images"] != 80 or (cfg["n"], cfg["T"]) != (16, 10):
        raise ValueError("The registered raw-row comparison requires all five n16/T10 batches")
    expected_random = np.random.default_rng(42).choice(80, 12, replace=False).tolist()
    if cfg["grid"] != {"seed": 42, "n": 12} or manifest["random_grid_indices"] != expected_random:
        raise RuntimeError("Random grid selection changed from its preregistered seed")
    for name, expected in manifest["inputs_sha256"].items():
        if file_hash(resolve(name)) != expected:
            raise RuntimeError(f"Frozen report input changed: {name}")

    batches: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    source, reference = resolve(cfg["source_output"]), resolve(cfg["reference_output"])
    reference_summary = read_json(reference / "summary.json")
    if reference_summary["status"] != "complete" or reference_summary["n_images"] != 80:
        raise RuntimeError("Completed all80 CPA reference is required")
    expected_csv_hash = reference_summary["output_hashes"]["per_image_metrics.csv"]
    if file_hash(reference / "per_image_metrics.csv") != expected_csv_hash:
        raise RuntimeError("CPA reference metric table changed")
    reference_rows = read_csv(reference / "per_image_metrics.csv")
    _validate_rows(reference_rows)

    # Validate every result before loading any original images or plotting.
    for batch in cfg["batch_indices"]:
        folder = output / f"batch{batch:02d}"
        done = read_json(folder / "summary.json")
        if done["identity"] != manifest["identity"] or done["n_images"] != 16 or done["batch_index"] != batch:
            raise RuntimeError("All five completed batches must have the frozen identity")
        if not REQUIRED_BATCH_OUTPUTS.issubset(done["output_hashes"]):
            raise RuntimeError("A completed batch is missing required output hashes")
        _verify_hashes(folder, done["output_hashes"])
        batch_rows = read_csv(folder / "per_image_metrics.csv")
        expected_indices = list(range(16 * batch, 16 * (batch + 1)))
        if [int(row["global_index"]) for row in batch_rows] != expected_indices:
            raise RuntimeError("Batch image coverage is incomplete")
        if sorted(int(row["estimated_source_index"]) for row in batch_rows) != list(range(16)):
            raise RuntimeError("Post-FIA correspondence is not a complete permutation")
        if any(row["run_tag"] != cfg["run_tag"] for row in batch_rows):
            raise RuntimeError("Per-image results have the wrong run identity")
        measured = _metrics(batch_rows)
        for name in ("lpips_mean", "psnr_mean", "ssim_mean", "target_cos09_count"):
            if not np.isclose(measured[name], done[name], rtol=1e-7, atol=1e-9):
                raise RuntimeError(f"Batch summary disagrees with per-image {name}")
        ref_folder = reference / f"batch{batch:02d}"
        ref_done = read_json(ref_folder / "summary.json")
        if ref_done["identity"] != reference_summary["identity"] or ref_done["n_images"] != 16:
            raise RuntimeError("CPA reference batch identity differs from its summary")
        if file_hash(ref_folder / "rec.pt") != ref_done["output_hashes"]["rec.pt"]:
            raise RuntimeError("CPA reference reconstruction changed")
        batches.append(done)
        rows.extend(batch_rows)
    _validate_rows(rows)
    _validate_paired_identity(rows, reference_rows)

    originals, recs, reference_recs = [], [], []
    for batch in cfg["batch_indices"]:
        inputs = torch.load(source / f"batch{batch:02d}/inputs.pt", map_location="cpu", weights_only=False, mmap=True)
        original = inputs["x"].detach().cpu()
        mean = original.new_tensor((.485, .456, .406))[None, :, None, None]
        std = original.new_tensor((.229, .224, .225))[None, :, None, None]
        originals.append((original * std + mean).clamp(0, 1))
        recs.append(torch.load(output / f"batch{batch:02d}/rec.pt", map_location="cpu", weights_only=True))
        reference_recs.append(torch.load(reference / f"batch{batch:02d}/rec.pt", map_location="cpu", weights_only=True))
    original_images, raw_images, cpa_images = (torch.cat(items) for items in (originals, recs, reference_recs))
    for images in (original_images, raw_images, cpa_images):
        if images.shape != (80, 3, 224, 224) or not bool(torch.isfinite(images).all()):
            raise RuntimeError("All80 finite native-resolution images are required")

    metrics, reference_metrics = _metrics(rows), _metrics(reference_rows)
    paired = []
    for row, ref in zip(rows, reference_rows):
        pair = {key: row[key] for key in IMAGE_IDENTITY_FIELDS}
        pair.update(observation_row=int(row["observation_row"]),
                    cpa_source_abs_cosine=float(ref["source_abs_cosine"]),
                    rawrow_source_abs_cosine=float(row["source_abs_cosine"]))
        if "fia_target_cosine" in row:
            pair["rawrow_fia_target_cosine"] = float(row["fia_target_cosine"])
        for key in ("lpips", "psnr", "ssim"):
            pair[f"cpa_fia_{key}"] = float(ref[key])
            pair[f"rawrow_fia_{key}"] = float(row[key])
            pair[f"delta_rawrow_minus_cpa_{key}"] = float(row[key]) - float(ref[key])
        paired.append(pair)
    write_csv(output / "per_image_metrics.csv", rows)
    write_csv(output / "paired_cpa_comparison.csv", paired)
    write_csv(output / "per_batch_summary.csv", [
        {**{key: batch[key] for key in ("batch_index", "n_images", "lpips_mean", "psnr_mean", "ssim_mean", "target_cos09_count")},
         "fia_runtime_sec": batch["runtime_sec"]["fia"],
         "invocation_runtime_sec": batch["runtime_sec"]["invocation"]}
        for batch in batches
    ])
    torch.save(raw_images, output / "all80_rec.pt")
    images = {"Original": original_images, "CPA + FIA (M8)": cpa_images, "Raw rows + abs + FIA": raw_images}
    grids = [
        render_grid(images, rows, list(range(80)), output / "all80_cpa_vs_rawrow_landscape.png", columns=8,
                    title="FedAvg 4-bit dither: CPA + FIA and direct raw-update-row FIA | all 80 images"),
        render_grid(images, rows, expected_random, output / "random12_not_cherry_picked.png", columns=4,
                    title="FedAvg 4-bit dither: CPA + FIA and direct raw-update-row FIA | random 12 of 80"),
        render_grid({"Original": original_images, "Raw rows + abs + FIA": raw_images}, rows,
                    list(range(80)), output / "all80_reconstruction_landscape.png", columns=10,
                    title="FedAvg 4-bit dither: direct raw-update-row FIA, without CPA | all 80 images"),
    ]
    write_json(output / "figure_manifest.json", {
        "run_tag": cfg["run_tag"], "identity": manifest["identity"],
        "selection": {"all80": "all samples in original global-index order",
                      "random12": {"seed": 42, "indices": expected_random, "selected_before_results": True}},
        "panels": {"Original": str(source), "CPA + FIA (M8)": str(reference),
                   "Raw rows + abs + FIA": str(output)},
        "evaluation_correspondence": "Hungarian feature-cosine correspondence after FIA only",
        "grids": grids,
    })
    runtime = {key: sum(float(batch["runtime_sec"][key]) for batch in batches) for key in ("fia", "invocation")}
    delta = metrics["lpips_mean"] - reference_metrics["lpips_mean"]
    report = [
        "# FedAvg direct raw-update-row FIA (no CPA)", "",
        "Completed all 80 images: VGG16, n=16, T=10. This experiment reuses exactly the cached observation used by the M8 CPA reference.", "",
        "| Method | Images | LPIPS | PSNR | SSIM | Source absolute cosine > 0.9 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Direct upstream CPA + FIA (M8) | 80 | {reference_metrics['lpips_mean']:.6f} | {reference_metrics['psnr_mean']:.3f} | {reference_metrics['ssim_mean']:.4f} | {reference_metrics['target_cos09_count']}/80 |",
        f"| Raw update rows + abs + FIA (M9) | 80 | {metrics['lpips_mean']:.6f} | {metrics['psnr_mean']:.3f} | {metrics['ssim_mean']:.4f} | {metrics['target_cos09_count']}/80 |", "",
        "The M9 input is the elementwise absolute value of selected observed update rows. These are mixed-update vectors used as FIA targets, not separated individual-image embeddings.",
        "For each batch, 16 of the 4096 rows are chosen without replacement by CPU torch.randperm with seed=base_seed+batch_index, fixed before FIA. Selection order and zero rows are preserved; no L2 normalization or data-dependent row replacement is applied.",
        "No CPA/ICA, debias, MP truncation or diagnostic, LS mixing-matrix estimate, embedding reoptimization, oracle fill, or new client training is used in M9.",
        "FIA uses the same original Direct implementation, fixed iteration budget and TV coefficient as M8. Original features/images are consulted only after FIA for correspondence and evaluation.",
        "The observation is the existing per-row 4-bit-interval uniform-dither model, not a transmitted integer 4-bit codec. This is raw-row feature inversion, not dequantization-only image reconstruction.", "",
        f"Mean paired LPIPS difference (raw rows minus CPA): {delta:+.6f}. This is a descriptive comparison of two feature-input construction methods; it does not isolate every implementation choice or establish a general CPA contribution.",
        "Source absolute cosine is computed between the signed raw update row and the original feature, using the same post-FIA correspondence criterion as M8. The FIA target itself is abs(raw row); its cosine is stored separately as fia_target_cosine. Source cosine > 0.9 is not an LPIPS success threshold and does not establish successful per-image source separation for raw rows.",
        "No invalidated historical no-CPA control, normalized signal percentage, floor-based threshold, or statistical significance claim is used.", "",
        f"FIA runtime: {runtime['fia']:.1f} seconds; total batch invocation runtime: {runtime['invocation']:.1f} seconds.",
        "All batch identities, output hashes, frozen input hashes, all80 image identities, and native-resolution PNG tile pixels were verified before completion.", "",
        "## Figures", "",
        "All80 figures contain every image in original index order. Random12 indices were fixed with seed42 before results; not cherry-picked. All panels preserve 224 x 224 pixels per image.", "",
        "![All80 CPA and raw-row comparison](all80_cpa_vs_rawrow_landscape.png)", "",
        "![Random12 not cherry-picked](random12_not_cherry_picked.png)", "",
        "![All80 original and raw-row reconstruction](all80_reconstruction_landscape.png)", "",
        "Per-image observations, metrics and correspondence are in `per_image_metrics.csv`; paired differences are in `paired_cpa_comparison.csv`; exact grid positions and pixel hashes are in `figure_manifest.json`.", "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    names = (
        "per_image_metrics.csv", "paired_cpa_comparison.csv", "per_batch_summary.csv", "all80_rec.pt",
        "all80_cpa_vs_rawrow_landscape.png", "random12_not_cherry_picked.png",
        "all80_reconstruction_landscape.png", "figure_manifest.json", "report.md",
    )
    result = {
        "run_tag": cfg["run_tag"], "status": "complete", "n_images": 80,
        "config": cfg, "identity": manifest["identity"], "metrics": metrics,
        "reference_run_tag": reference_summary["run_tag"], "reference_metrics": reference_metrics,
        "mean_paired_lpips_difference_rawrow_minus_cpa": delta,
        "runtime_sec": runtime, "oracle_fills": 0, "cpa": False, "debias": False,
        "mp_truncation": False, "ls_A": False, "stage2": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_hashes": {name: file_hash(output / name) for name in names},
    }
    write_json(output / "summary.json", result)
    return result
