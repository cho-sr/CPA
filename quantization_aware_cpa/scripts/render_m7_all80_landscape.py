"""Render a landscape research figure from the saved all80 experiment tensors."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    folder = root / "results" / "m7_uia_4bit_B16"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or summary["n_images"] != 80:
        raise RuntimeError("The complete 80-image experiment is required")
    with (folder / "per_image_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["global_index"]) for row in rows] != list(range(80)):
        raise RuntimeError("Expected all images in original global-index order")

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
    title = ImageFont.truetype(str(font_path), 34)
    draw.text((margin, 16), "UIA 4-bit | n=16, T=10 | All 80 images", font=title, fill="#111827")
    draw.text((margin, 66), "Original (left) / 4-bit reconstruction (right) | g00-g79 in original order | All images included; not cherry-picked",
              font=font, fill="#475569")

    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    rendered = []
    for batch in range(5):
        batch_folder = folder / f"batch{batch:02d}"
        batch_summary = json.loads((batch_folder / "summary.json").read_text(encoding="utf-8"))
        for name in ("inputs.pt", "rec.pt"):
            actual_hash = hashlib.sha256((batch_folder / name).read_bytes()).hexdigest()
            if actual_hash != batch_summary["output_hashes"][name]:
                raise RuntimeError(f"Source tensor hash mismatch: {batch_folder / name}")
        inputs = torch.load(batch_folder / "inputs.pt", map_location="cpu", weights_only=False)
        truth = (inputs["x"] * std + mean).clamp(0, 1)
        recs = torch.load(batch_folder / "rec.pt", map_location="cpu", weights_only=True)
        if truth.shape != recs.shape or truth.shape[0] != 16:
            raise RuntimeError("Expected 16 aligned original/reconstruction pairs")
        for source in range(16):
            index = batch * 16 + source
            row = rows[index]
            if int(row["batch_index"]) != batch or int(row["source_index"]) != source:
                raise RuntimeError("Metric/tensor order mismatch")
            left = margin + index % columns * (block + group_gap)
            top = header + index // columns * (tile + labels + row_gap)
            label = f"g{index:02d}   LPIPS={float(row['lpips']):.3f}   cos={float(row['source_abs_cosine']):.3f}"
            draw.text((left, top), label, font=font, fill="#111827")
            for column, (name, tensor) in enumerate((("Original", truth[source]), ("4-bit", recs[source]))):
                x = left + column * (tile + inner_gap)
                draw.text((x, top + 27), name, font=small, fill="#475569")
                array = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                image = Image.fromarray(array)
                if image.size != (tile, tile):
                    image = image.resize((tile, tile), Image.Resampling.LANCZOS)
                canvas.paste(image, (x, top + labels))
            rendered.append(index)
    assert rendered == list(range(80))
    output = folder / "all80_reconstruction_grid_landscape.png"
    canvas.save(output, dpi=(200, 200))
    with Image.open(output) as check:
        check.verify()
    print(json.dumps({"output": str(output), "size": [width, height], "image_pairs": len(rendered),
                      "layout": [columns, grid_rows], "format": "PNG"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
