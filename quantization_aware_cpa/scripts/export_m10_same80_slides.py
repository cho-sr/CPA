"""Render verified experiment image tiles as 16:9 research figures.

Uses the existing PNG and its recorded crop coordinates. No model inference,
image synthesis, color correction, or experimental result files are changed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "results/m10_cpa_fia_packed4_same80_B16"
OUTPUT = SOURCE / "slides_16x9"
SIZE = (3840, 2160)
PANELS = ("Original", "Packed4 CPA + FIA")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size: int, bold: bool = False):
    return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", size)


def means(rows: list[dict]) -> dict:
    values = {key: math.fsum(float(row[key]) for row in rows) / len(rows) for key in ("lpips", "ssim", "psnr")}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Nonfinite image metric")
    return values


def render(rows: list[dict], tiles: dict, filename: str, batch: int | None) -> dict:
    width, height = SIZE
    canvas = Image.new("RGB", SIZE, "white")
    draw = ImageDraw.Draw(canvas)
    columns = 10 if batch is None else 4
    nrows = math.ceil(len(rows) / columns)
    label_height, gap_x, gap_y, pair_gap = 38, 24, 14, 10
    top, bottom, margin = 230, 80, 64
    tile = int(min(((width - 2 * margin - (columns - 1) * gap_x) / columns - pair_gap) / 2,
                   (height - top - bottom - (nrows - 1) * gap_y) / nrows - label_height))
    block_width, block_height = 2 * tile + pair_gap, tile + label_height
    grid_width = columns * block_width + (columns - 1) * gap_x
    grid_height = nrows * block_height + (nrows - 1) * gap_y
    x0 = (width - grid_width) // 2
    y0 = top + (height - top - bottom - grid_height) // 2
    coverage = "All 80 images" if batch is None else f"Batch {batch} | All 16 images"
    draw.text((64, 30), f"Packed 4-bit CPA + FIA | {coverage}", font=font(56, True), fill="#14243a")
    draw.text((64, 103), "VGG16 | 16 images per client update | 10 local SGD steps | Original order", font=font(31), fill="#526074")
    average = means(rows)
    metrics_text = f"Mean LPIPS  {average['lpips']:.4f}     |     Mean SSIM  {average['ssim']:.4f}     |     Mean PSNR  {average['psnr']:.4f} dB"
    draw.text((64, 158), metrics_text, font=font(38, True), fill="#1e5799")
    legend = "Each pair: ORIGINAL (left) / RECONSTRUCTION (right)"
    draw.text((width - 64, 116), legend, anchor="ra", font=font(27), fill="#526074")
    draw.line((64, 216, width - 64, 216), fill="#cbd5e1", width=2)
    placements = []
    for index, row in enumerate(rows):
        global_id = int(row["global_index"])
        x = x0 + (index % columns) * (block_width + gap_x)
        y = y0 + (index // columns) * (block_height + gap_y)
        text = f"g{global_id:02d} | LPIPS {float(row['lpips']):.3f} | cos {float(row['source_abs_cosine']):.3f}"
        label_font = font(20 if batch is None else 27)
        if draw.textbbox((0, 0), text, font=label_font)[2] > block_width:
            raise ValueError("Image label exceeds its pair width")
        draw.text((x, y), text, font=label_font, fill="#263449")
        for panel_index, panel in enumerate(PANELS):
            pixels = tiles[(global_id, panel)].resize((tile, tile), Image.Resampling.LANCZOS)
            left, upper = x + panel_index * (tile + pair_gap), y + label_height
            box = (left, upper, left + tile, upper + tile)
            if not (0 <= left < box[2] <= width and top <= upper < box[3] <= height - bottom):
                raise ValueError("Image tile exceeds slide content bounds")
            canvas.paste(pixels, (left, upper))
            placements.append({"global_index": global_id, "panel": panel, "box": box,
                               "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
    draw.text((64, height - 48), "All registered images retained | cos: matched source absolute cosine", font=font(26), fill="#526074")
    draw.text((width - 64, height - 48), "16:9  /  3840 x 2160", anchor="ra", font=font(26), fill="#526074")
    path = OUTPUT / filename
    canvas.save(path, dpi=(288, 288))
    with Image.open(path) as saved:
        assert saved.size == SIZE
        for placement in placements:
            if hashlib.sha256(saved.crop(placement["box"]).tobytes()).hexdigest() != placement["rgb_sha256"]:
                raise ValueError("Exported tile differs from the specified resampling")
    return {"file": filename, "sha256": sha256(path), "size": SIZE,
            "columns": columns, "rows": nrows, "tile_size": tile,
            "n_images": len(rows), "global_indices": [int(row["global_index"]) for row in rows],
            "mean_metrics": average, "placements": placements}


def main() -> None:
    summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((SOURCE / "figure_manifest.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or summary["n_images"] != 80:
        raise ValueError("Expected the complete 80-image experiment")
    for name in ("per_image_metrics.csv", "figure_manifest.json", "all80_reconstruction_landscape.png"):
        if sha256(SOURCE / name) != summary["output_hashes"][name]:
            raise ValueError(f"Source artifact hash mismatch: {name}")
    with (SOURCE / "per_image_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["global_index"]) for row in rows] != list(range(80)):
        raise ValueError("Expected all80 in the original order")
    average = means(rows)
    for key, value in average.items():
        if not math.isclose(value, summary["metrics"][f"{key}_mean"], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Mean does not match the registered result: {key}")
    tiles = {}
    with Image.open(SOURCE / "all80_reconstruction_landscape.png") as source:
        for placement in manifest["grid"]["placements"]:
            tile = source.crop(placement["box"]).convert("RGB")
            if tile.size != (224, 224) or hashlib.sha256(tile.tobytes()).hexdigest() != placement["rgb_sha256"]:
                raise ValueError("Original figure tile identity mismatch")
            key = (int(placement["global_index"]), placement["panel"])
            if key in tiles:
                raise ValueError("Duplicate source panel")
            tiles[key] = tile
    if set(tiles) != {(i, panel) for i in range(80) for panel in PANELS}:
        raise ValueError("Missing original/reconstruction image pair")
    OUTPUT.mkdir(exist_ok=True)
    outputs = [render(rows, tiles, "all80_reconstruction_3840x2160.png", None)]
    for batch in range(5):
        selected = rows[batch * 16:(batch + 1) * 16]
        if any(int(row["batch_index"]) != batch for row in selected):
            raise ValueError("Original batch grouping changed")
        outputs.append(render(selected, tiles, f"batch{batch:02d}_reconstruction_3840x2160.png", batch))
    with (OUTPUT / "mean_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "n_images", "mean"))
        writer.writeheader()
        writer.writerows({"metric": key, "n_images": 80, "mean": value} for key, value in average.items())
    provenance = {"run_tag": summary["run_tag"], "source_png_sha256": sha256(SOURCE / "all80_reconstruction_landscape.png"),
                  "source_csv_sha256": sha256(SOURCE / "per_image_metrics.csv"), "mean_metrics": average,
                  "rendering": "Existing verified RGB tiles; isotropic Lanczos resampling; no color correction or image synthesis",
                  "outputs": outputs}
    (OUTPUT / "export_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mean_metrics": average, "output_dir": str(OUTPUT),
                      "files": [output["file"] for output in outputs], "verified_source_tiles": len(tiles)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
