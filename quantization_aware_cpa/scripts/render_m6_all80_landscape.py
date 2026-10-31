"""Render the archived FedAvg FP32 all80 tensors without running an attack.

Run from any directory with the experiment's Python environment. This script
only reads saved CPU tensors and writes its PNG plus its own audit manifest.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rgb_array(tensor: torch.Tensor) -> np.ndarray:
    if tensor.device.type != "cpu" or tuple(tensor.shape) != (3, 224, 224):
        raise RuntimeError("Expected one saved CPU image with shape (3,224,224)")
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError("Non-finite image tensor")
    return (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def main() -> None:
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    folder = root / "results/m6_uia_fp32_B16_fia"
    summary_path = folder / "summary.json"
    metrics_path = folder / "per_image_metrics.csv"
    summary = read_json(summary_path)
    if (summary["status"] != "complete" or summary["n_images"] != 80
            or summary["oracle_fills"] != 0):
        raise RuntimeError("A complete 80-image run without oracle fills is required")
    if sha256(metrics_path) != summary["output_hashes"][metrics_path.name]:
        raise RuntimeError("Full-run metric CSV hash mismatch")
    with metrics_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["global_index"]) for row in rows] != list(range(80)):
        raise RuntimeError("Expected all80 images in original global-index order")
    for row in rows:
        if (row["oracle_fill"].lower() != "false" or row["missing_source"].lower() != "false"
                or row["bits"] != "fp32" or int(row["T"]) != 10 or int(row["n"]) != 16):
            raise RuntimeError("Unexpected per-image protocol or missing/oracle source")

    tile, inner_gap, group_gap, margin = 224, 12, 20, 20
    columns, grid_rows, header, labels, row_gap = 10, 8, 110, 50, 18
    block = 2 * tile + inner_gap
    width = 2 * margin + columns * block + (columns - 1) * group_gap
    height = header + grid_rows * (tile + labels) + (grid_rows - 1) * row_gap + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 20)
    small = ImageFont.truetype(str(font_path), 17)
    title_font = ImageFont.truetype(str(font_path), 34)
    title = "FedAvg FP32 (32-bit) | n=16, T=10 | All 80 images"
    subtitle = ("Original (left) / FP32 reconstruction (right) | g00-g79 in original order | "
                "All images included; not cherry-picked | No oracle fills")
    draw.text((margin, 16), title, font=title_font, fill="#111827")
    draw.text((margin, 66), subtitle, font=font, fill="#475569")

    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    sources = []
    placements = []
    expected_tiles = []
    for batch in range(5):
        batch_folder = folder / f"batch{batch:02d}"
        batch_summary_path = batch_folder / "summary.json"
        batch_summary = read_json(batch_summary_path)
        if (batch_summary["batch_index"] != batch or batch_summary["n_images"] != 16
                or batch_summary["oracle_fills"] != 0):
            raise RuntimeError("Batch coverage or oracle-fill mismatch")
        source_entry = {"batch_index": batch,
                        "summary_path": str(batch_summary_path.relative_to(root)),
                        "summary_sha256": sha256(batch_summary_path), "tensors": {}}
        for name in ("inputs.pt", "rec.pt"):
            path = batch_folder / name
            digest = sha256(path)
            if digest != batch_summary["output_hashes"][name]:
                raise RuntimeError(f"Archived tensor hash mismatch: {path}")
            if digest != summary["per_victim_batch"][batch]["output_hashes"][name]:
                raise RuntimeError(f"Batch/full-run tensor hash disagreement: {path}")
            source_entry["tensors"][name] = {"path": str(path.relative_to(root)),
                                               "sha256": digest,
                                               "matches_archived_hash": True}
        sources.append(source_entry)
        inputs = torch.load(batch_folder / "inputs.pt", map_location="cpu", weights_only=False)
        originals = (inputs["x"].float() * std + mean).clamp(0, 1)
        recs = torch.load(batch_folder / "rec.pt", map_location="cpu", weights_only=True)
        if tuple(originals.shape) != (16, 3, tile, tile) or originals.shape != recs.shape:
            raise RuntimeError("Expected 16 aligned full-resolution pairs")
        if len(inputs["metadata"]) != 16:
            raise RuntimeError("Expected metadata for all16 batch images")
        for source in range(16):
            index = batch * 16 + source
            row = rows[index]
            meta = inputs["metadata"][source]
            for key, expected in (("global_index", index), ("batch_index", batch),
                                  ("source_index", source)):
                if int(row[key]) != expected or int(meta[key]) != expected:
                    raise RuntimeError(f"Metric/tensor metadata order mismatch: {key}")
            for key in ("dataset_index", "label"):
                if int(row[key]) != int(meta[key]):
                    raise RuntimeError(f"Metric/tensor metadata mismatch: {key}")
            if str(row["image_path"]).replace("\\", "/").lower() != str(meta["image_path"]).replace("\\", "/").lower():
                raise RuntimeError("Metric/tensor image-path mismatch")
            left = margin + index % columns * (block + group_gap)
            top = header + index // columns * (tile + labels + row_gap)
            label = f"g{index:02d}   LPIPS={float(row['lpips']):.3f}   cos={float(row['source_abs_cosine']):.3f}"
            draw.text((left, top), label, font=font, fill="#111827")
            entry = {"global_index": index, "batch_index": batch, "source_index": source,
                     "image_path": row["image_path"], "oracle_fill": False,
                     "lpips": float(row["lpips"]),
                     "source_abs_cosine": float(row["source_abs_cosine"]), "tiles": []}
            for column, (name, tensor) in enumerate((("Original", originals[source]),
                                                     ("FP32", recs[source]))):
                x = left + column * (tile + inner_gap)
                draw.text((x, top + 27), name, font=small, fill="#475569")
                array = rgb_array(tensor)
                bbox = [x, top + labels, x + tile, top + labels + tile]
                canvas.paste(Image.fromarray(array), (bbox[0], bbox[1]))
                entry["tiles"].append({"condition": name, "bbox_xyxy": bbox,
                                       "rgb8_sha256": hashlib.sha256(array.tobytes()).hexdigest()})
                expected_tiles.append((bbox, array))
            placements.append(entry)

    if [entry["global_index"] for entry in placements] != list(range(80)):
        raise RuntimeError("Final rendering does not cover all80 exactly once")
    output = folder / "all80_reconstruction_grid_landscape.png"
    canvas.save(output, dpi=(200, 200))
    with Image.open(output) as check:
        check.verify()
    with Image.open(output) as check:
        if check.mode != "RGB" or check.size != (width, height):
            raise RuntimeError("Unexpected PNG mode or dimensions")
        for bbox, expected in expected_tiles:
            if not np.array_equal(np.asarray(check.crop(tuple(bbox))), expected):
                raise RuntimeError(f"Saved PNG tile differs from archived tensor: {bbox}")
    if torch.cuda.is_initialized():
        raise RuntimeError("Rendering unexpectedly initialized CUDA")

    manifest = {
        "schema_version": 1, "artifact_type": "archived_reconstruction_layout",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": summary["run_tag"], "title": title, "subtitle": subtitle,
        "script": {"path": str(Path(__file__).resolve().relative_to(root)),
                   "sha256": sha256(Path(__file__).resolve())},
        "font": {"path": str(font_path), "sha256": sha256(font_path)},
        "source_summary": {"path": str(summary_path.relative_to(root)), "sha256": sha256(summary_path)},
        "source_metrics": {"path": str(metrics_path.relative_to(root)), "sha256": sha256(metrics_path)},
        "source_batches": sources,
        "output": {"path": str(output.relative_to(root)), "sha256": sha256(output),
                   "format": "PNG", "mode": "RGB", "width": width, "height": height},
        "layout": {"groups_across": columns, "group_rows": grid_rows,
                   "tile_width": tile, "tile_height": tile,
                   "pair_order": ["original", "FP32 reconstruction"]},
        "verification": {"image_pairs": 80, "exact_rgb8_tensor_tile_matches": len(expected_tiles),
                         "all_global_indices_once_in_original_order": True,
                         "archived_input_and_reconstruction_hashes_verified": 10,
                         "metadata_and_csv_correspondence_verified": True,
                         "png_decodes": True, "oracle_fills": 0,
                         "gpu_initialized": False, "new_attack_or_training": False},
        "placements": placements,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path),
                      "size": [width, height], "sha256": manifest["output"]["sha256"],
                      "verification": manifest["verification"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
