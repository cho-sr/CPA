"""Lay out existing FedSGD reconstructions; do not run an attack or alter pixels."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "results/v9_stage2_saturation_diagnostics/component_lpips_contributions.csv"
OUTPUT_DIR = ROOT / "results/fedsgd_all40_fp32_4bit_landscape"
ORACLE_IDS = {22, 24, 32, 36}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)


def main() -> None:
    with SOURCE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if sorted(int(row["global_index"]) for row in rows) != list(range(40)):
        raise RuntimeError("Exactly the historical 40 distinct samples are required")
    rows.sort(key=lambda row: float(row["per_sample_loss"]))
    actual_oracle_ids = {
        int(row["global_index"])
        for row in rows
        if "v7_m3_fp32_truth_missing" in row["fp32_rec_png"]
    }
    if actual_oracle_ids != ORACLE_IDS:
        raise RuntimeError("Unexpected FP32 oracle supplement provenance")

    tile, inner_gap, group_gap, margin = 224, 12, 24, 32
    columns, grid_rows, header, label_height, row_gap = 10, 4, 168, 62, 28
    block_width = 2 * tile + inner_gap
    width = 2 * margin + columns * block_width + (columns - 1) * group_gap
    height = header + grid_rows * (tile + label_height) + (grid_rows - 1) * row_gap + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, text_font, label_font = font(36), font(21), font(18)
    draw.text((margin, 20), "FedSGD | FP32 (32-bit) vs 4-bit dither | All 40 images", font=title_font, fill="#111827")
    draw.text((margin, 72), "B=8, T=1 | Left: FP32 / Right: 4-bit | Per-sample loss ascending, matching the historical figure | All 40 included; not cherry-picked", font=text_font, fill="#475569")
    draw.text((margin, 108), "* Amber FP32 labels: true-embedding FIA supplements (g22, g24, g32, g36). The other 36 FP32 images use CPA + FIA.", font=text_font, fill="#92400e")

    manifest_rows = []
    expected_tiles = []
    for slot, row in enumerate(rows):
        gid = int(row["global_index"])
        left = margin + slot % columns * (block_width + group_gap)
        top = header + slot // columns * (tile + label_height + row_gap)
        draw.text((left, top), f"g{gid:02d}   loss={float(row['per_sample_loss']):.3g}", font=text_font, fill="#111827")
        item = {
            "display_index": slot,
            "global_index": gid,
            "batch_index": int(row["batch"]),
            "source_index": int(row["source_index"]),
            "per_sample_loss": float(row["per_sample_loss"]),
            "fp32_oracle_supplement": gid in ORACLE_IDS,
        }
        for column, (method, key) in enumerate((("fp32", "fp32_rec_png"), ("4bit", "naive_rec_png"))):
            source = (ROOT / row[key]).resolve()
            if ROOT not in source.parents or not source.is_file():
                raise RuntimeError(f"Missing or out-of-workspace source: {source}")
            if f"global{gid:02d}_rec.png" not in source.name:
                raise RuntimeError(f"Source/global index mismatch: {source}")
            if method == "4bit" and source.parent.parent.name != "naive":
                raise RuntimeError("The 4-bit column must use naive CPA, without embedding refinement")
            x, y = left + column * (tile + inner_gap), top + label_height
            label = "FP32* / oracle embedding" if method == "fp32" and gid in ORACLE_IDS else "FP32 (32-bit)" if method == "fp32" else "4-bit dither / naive CPA"
            if draw.textbbox((0, 0), label, font=label_font)[2] > tile:
                raise RuntimeError(f"Tile label exceeds image width: {label}")
            if method == "fp32" and gid in ORACLE_IDS:
                draw.rectangle((x, top + 31, x + tile - 1, y - 5), fill="#fef3c7")
            draw.text((x, top + 33), label, font=label_font, fill="#92400e" if method == "fp32" and gid in ORACLE_IDS else "#475569")
            with Image.open(source) as opened:
                image = opened.convert("RGB")
            if image.size != (tile, tile):
                raise RuntimeError(f"Expected native 224x224 reconstruction: {source}, got {image.size}")
            canvas.paste(image, (x, y))
            expected_tiles.append((source, (x, y, x + tile, y + tile)))
            item[f"{method}_source"] = str(source.relative_to(ROOT)).replace("\\", "/")
            item[f"{method}_source_sha256"] = sha256(source)
            item[f"{method}_tile_box"] = [x, y, x + tile, y + tile]
        manifest_rows.append(item)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "fedsgd_all40_fp32_vs_4bit_loss_sorted_landscape.png"
    canvas.save(output, dpi=(200, 200))
    with Image.open(output) as opened:
        opened.verify()
    with Image.open(output) as opened:
        verified = opened.convert("RGB")
    for source, box in expected_tiles:
        with Image.open(source) as opened:
            if ImageChops.difference(verified.crop(box), opened.convert("RGB")).getbbox() is not None:
                raise RuntimeError(f"Saved figure altered reconstruction pixels: {source}")
    manifest = {
        "schema_version": 1,
        "output": output.name,
        "output_sha256": sha256(output),
        "size_pixels": [width, height],
        "sample_count": 40,
        "image_tile_count": 80,
        "layout_sample_columns_rows": [columns, grid_rows],
        "tile_size_pixels": [tile, tile],
        "order": "per_sample_loss ascending; all 40 historical samples",
        "new_attack_or_training": False,
        "pixel_verification": "All 80 saved reconstruction tiles exactly match source PNG pixels; no resizing or contrast changes",
        "methods": {
            "fp32": "36 ported CPA+FIA reconstructions plus 4 explicitly marked true-embedding FIA supplements",
            "4bit": "Ported naive CPA+FIA on the historical per-row 4-bit dither observation model; no embedding refinement",
        },
        "fp32_oracle_supplement_ids": sorted(ORACLE_IDS),
        "source_csv": str(SOURCE_CSV.relative_to(ROOT)).replace("\\", "/"),
        "source_csv_sha256": sha256(SOURCE_CSV),
        "renderer": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "renderer_sha256": sha256(Path(__file__)),
        "samples": manifest_rows,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), "size": [width, height], "samples": 40, "exact_source_tiles_verified": len(expected_tiles)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
