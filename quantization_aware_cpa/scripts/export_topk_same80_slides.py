"""Make one verified 16:9 original/reconstruction figure per Top-k percentage."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import zipfile

from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "exports/topk_same80_reconstruction_16x9"
PERCENTS = (50, 30, 20, 10)
SIZE = (3840, 2160)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    return digest(path.read_bytes())


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def font(size: int, bold: bool = False):
    return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", size)


def load_source(percent: int) -> tuple:
    folder = PROJECT / f"results/cpa_fia_topk{percent}_same80_B16"
    summary = read_json(folder / "summary.json")
    if summary["status"] != "complete" or summary["n_images"] != 80:
        raise RuntimeError(f"Top-k{percent} requires the completed all80 result")
    names = ("per_image_metrics.csv", "figure_manifest.json", "all80_reconstruction_landscape.png")
    for name in names:
        if file_hash(folder / name) != summary["output_hashes"][name]:
            raise RuntimeError(f"Source artifact changed: {folder / name}")
    with (folder / "per_image_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 80 or [int(row["global_index"]) for row in rows] != list(range(80)):
        raise RuntimeError("The figure must include every original image in its frozen order")
    for i, row in enumerate(rows):
        if (int(row["batch_index"]), int(row["source_index"])) != (i // 16, i % 16):
            raise RuntimeError("Original batch/source order changed")
    metrics = {key: math.fsum(float(row[key]) for row in rows) / 80 for key in ("lpips", "ssim", "psnr")}
    for key, value in metrics.items():
        if not math.isfinite(value) or not math.isclose(value, summary["metrics"][f"{key}_mean"], rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("Figure metrics do not match per-image measurements")
    manifest = read_json(folder / "figure_manifest.json")
    panels = {"Original": "Original", f"Top-k {percent}% + FIA": "Reconstruction"}
    tiles = {}
    with Image.open(folder / "all80_reconstruction_landscape.png") as source:
        for placement in manifest["grid"]["placements"]:
            tile = source.crop(placement["box"]).convert("RGB")
            if tile.size != (224, 224) or digest(tile.tobytes()) != placement["rgb_sha256"]:
                raise RuntimeError("Saved source tile differs from the recorded reconstruction")
            panel = panels[placement["panel"]]
            key = (int(placement["global_index"]), panel)
            if key in tiles:
                raise RuntimeError("Duplicate source image panel")
            tiles[key] = tile.copy()
    if set(tiles) != {(i, panel) for i in range(80) for panel in panels.values()}:
        raise RuntimeError("The full80 original/reconstruction pairs are required")
    return rows, tiles, metrics, {"run_tag": summary["run_tag"], "source_folder": str(folder.relative_to(PROJECT)),
                                 "summary_sha256": file_hash(folder / "summary.json"),
                                 "source_hashes": {name: file_hash(folder / name) for name in names}}


def render(percent: int, rows: list, tiles: dict, metrics: dict) -> dict:
    width, height = SIZE
    canvas = Image.new("RGB", SIZE, "white")
    draw = ImageDraw.Draw(canvas)
    margin = 68
    ink, muted, accent = "#15243B", "#536176", "#245FA0"
    draw.text((margin, 26), "TOP-K SPARSIFICATION  /  CPA + FIA", font=font(28, True), fill=accent)
    title = f"{percent}% retained  |  Original vs Reconstruction"
    draw.text((margin, 74), title, font=font(60, True), fill=ink)
    draw.text((width - margin, 38), "ALL 80 IMAGES", anchor="ra", font=font(34, True), fill=ink)
    draw.text((width - margin, 90), "Same 5 batches x 16 images  |  Local SGD: 10 steps", anchor="ra", font=font(28), fill=muted)
    metric_text = f"Mean LPIPS  {metrics['lpips']:.4f}     |     Mean SSIM  {metrics['ssim']:.4f}     |     Mean PSNR  {metrics['psnr']:.4f} dB"
    draw.text((margin, 162), metric_text, font=font(35, True), fill=accent)
    draw.text((width - margin, 170), "Each pair: Original (left) / Reconstruction (right)", anchor="ra", font=font(27), fill=muted)
    if draw.textbbox((margin, 74), title, font=font(60, True))[2] >= width - margin - 800:
        raise RuntimeError("Figure title overlaps the coverage description")
    draw.line((margin, 221, width - margin, 221), fill="#CBD5E1", width=2)
    tile_size, columns, pair_gap, gap_x = 174, 10, 8, 16
    label_height, gap_y = 32, 24
    pair_width = 2 * tile_size + pair_gap
    row_height = label_height + tile_size + gap_y
    grid_width = columns * pair_width + (columns - 1) * gap_x
    x0, y0 = (width - grid_width) // 2, 276
    placements = []
    for i, row in enumerate(rows):
        global_id = int(row["global_index"])
        left = x0 + (i % columns) * (pair_width + gap_x)
        top = y0 + (i // columns) * row_height
        for j, panel in enumerate(("Original", "Reconstruction")):
            x = left + j * (tile_size + pair_gap)
            text = f"{global_id:02d}  Original" if j == 0 else "Reconstruction"
            label_font = font(20, j == 1)
            if draw.textbbox((0, 0), text, font=label_font)[2] > tile_size:
                raise RuntimeError("Image label exceeds its tile width")
            draw.text((x, top), text, font=label_font, fill=muted if j == 0 else accent)
            tile = tiles[(global_id, panel)].resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            box = (x, top + label_height, x + tile_size, top + label_height + tile_size)
            if not (0 <= box[0] < box[2] <= width and 230 <= box[1] < box[3] < height - 60):
                raise RuntimeError("Tile exceeds the figure content bounds")
            canvas.paste(tile, box[:2])
            placements.append({"global_index": global_id, "panel": panel, "box": box,
                               "rgb_sha256": digest(tile.tobytes())})
    draw.text((margin, height - 46), "All 80 images in original order  |  VGG16 round 0  |  Retained percentage of the first FC update", font=font(25), fill=muted)
    draw.text((width - margin, height - 46), "16:9  /  3840 x 2160", anchor="ra", font=font(25), fill=muted)
    path = OUTPUT / f"topk{percent}_original_vs_reconstruction_16x9.png"
    canvas.save(path, dpi=(288, 288))
    with Image.open(path) as saved:
        if saved.size != SIZE or saved.width * 9 != saved.height * 16:
            raise RuntimeError("Export must be exactly16:9")
        for placement in placements:
            if digest(saved.crop(placement["box"]).tobytes()) != placement["rgb_sha256"]:
                raise RuntimeError("Exported tile differs from the specified resampling")
    return {"keep_percent": percent, "file": path.name, "sha256": file_hash(path), "size": SIZE,
            "n_images": 80, "n_panels": 160, "global_indices": list(range(80)),
            "mean_metrics": metrics, "tile_size": tile_size, "placements": placements}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records, metric_rows, original_hashes, identities = [], [], None, None
    for percent in PERCENTS:
        rows, tiles, metrics, provenance = load_source(percent)
        current = [digest(tiles[(i, "Original")].tobytes()) for i in range(80)]
        current_ids = [[row[k] for k in ("global_index", "batch_index", "source_index", "dataset_index", "label", "image_path")] for row in rows]
        if original_hashes is not None and (current != original_hashes or current_ids != identities):
            raise RuntimeError("Percentages must use exactly the same originals and identities")
        original_hashes, identities = current, current_ids
        record = render(percent, rows, tiles, metrics)
        records.append({**record, **provenance})
        metric_rows.append({"keep_percent": percent, "n_images": 80, **metrics})
        print(json.dumps({k: record[k] for k in ("keep_percent", "file", "size", "n_images", "mean_metrics")}), flush=True)
    with (OUTPUT / "mean_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    manifest = {"rendering": "Verified existing RGB tiles; isotropic Lanczos resize only; no image synthesis or color adjustment",
                "all80_originals_identical_across_percentages": True, "outputs": records}
    (OUTPUT / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    archive_path = OUTPUT / "topk50_30_20_10_reconstruction_16x9.zip"
    names = [record["file"] for record in records] + ["mean_metrics.csv", "export_manifest.json"]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(OUTPUT / name, arcname=name)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for name in names:
            assert digest(archive.read(name)) == file_hash(OUTPUT / name)


if __name__ == "__main__":
    main()
