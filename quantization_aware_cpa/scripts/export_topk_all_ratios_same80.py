"""Arrange the same 80 saved images and four zero-fill Top-k results in one PNG.

Reuses authenticated native RGB tiles and existing identity/metric records.
This is a figure export only; it does not invoke attacks or perform matching.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from export_topk_same80_slides import (
    PERCENTS, PROJECT, digest, file_hash, font, load_source,
)

OUTPUT = PROJECT / "exports/naive_vs_topk50_comparison"
FILENAME = "sparsification_original_keep50_30_20_10_all80.png"
IDENTITY_FIELDS = (
    "global_index", "batch_index", "source_index", "dataset_index", "label", "image_path",
)
TILE = 224
COLUMNS = 5
MARGIN = 40
PADDING = 16
TILE_GAP = 8
BLOCK_GAP_X = 24
BLOCK_GAP_Y = 18
BLOCK_WIDTH = 2 * PADDING + 5 * TILE + 4 * TILE_GAP
BLOCK_HEIGHT = 328
HEADER = 220
FOOTER = 80


def main() -> None:
    loaded = {}
    identities = None
    original_hashes = None
    for percent in PERCENTS:
        rows, tiles, metrics, provenance = load_source(percent)
        current_ids = [[row[key] for key in IDENTITY_FIELDS] for row in rows]
        current_hashes = [digest(tiles[(i, "Original")].tobytes()) for i in range(80)]
        if identities is not None:
            if identities != current_ids or original_hashes != current_hashes:
                raise RuntimeError("The four ratios must use identical images in identical order")
        identities, original_hashes = current_ids, current_hashes
        loaded[percent] = {
            "rows": rows, "tiles": tiles, "metrics": metrics, "provenance": provenance,
        }

    row_count = (80 + COLUMNS - 1) // COLUMNS
    width = 2 * MARGIN + COLUMNS * BLOCK_WIDTH + (COLUMNS - 1) * BLOCK_GAP_X
    height = HEADER + row_count * BLOCK_HEIGHT + (row_count - 1) * BLOCK_GAP_Y + FOOTER
    canvas = Image.new("RGB", (width, height), "#f1f5f9")
    draw = ImageDraw.Draw(canvas)
    ink, muted = "#15243b", "#536176"
    panels = ("Original", "Keep 50%", "Keep 30%", "Keep 20%", "Keep 10%")
    colors = (muted, "#245fa0", "#245fa0", "#245fa0", "#245fa0")
    fonts = {size: font(size, bold) for size, bold in ((56, True), (30, False), (26, False), (24, True), (21, True), (23, False))}

    draw.text((MARGIN, 28), "SPARSIFICATION | ALL 80 IMAGES", font=fonts[56], fill=ink)
    draw.text((width - MARGIN, 46), "Naive = zero-fill at every ratio", anchor="ra", font=fonts[30], fill=ink)
    draw.text((MARGIN, 104), "Each image: Original  >  Keep 50%  >  Keep 30%  >  Keep 20%  >  Keep 10%",
              font=fonts[30], fill=ink)
    draw.text((MARGIN, 153), "Percentages indicate retained coordinates in the first FC weight tensor | VGG16 | 5 batches x 16 images | T = 10",
              font=fonts[26], fill=muted)
    draw.text((width - MARGIN, 157), "Native 224 x 224 tiles | Original image order 00-79",
              anchor="ra", font=fonts[26], fill=muted)
    placements = []
    for index in range(80):
        left = MARGIN + (index % COLUMNS) * (BLOCK_WIDTH + BLOCK_GAP_X)
        top = HEADER + (index // COLUMNS) * (BLOCK_HEIGHT + BLOCK_GAP_Y)
        draw.rounded_rectangle((left, top, left + BLOCK_WIDTH, top + BLOCK_HEIGHT),
                               radius=12, fill="white", outline="#d4dde7", width=2)
        draw.text((left + PADDING, top + 14), f"Image {index:02d}", font=fonts[24], fill=ink)
        draw.text((left + BLOCK_WIDTH - PADDING, top + 16), f"Batch {index // 16} | Index {index % 16:02d}",
                  anchor="ra", font=fonts[21], fill=muted)
        images = [loaded[50]["tiles"][(index, "Original")]]
        images += [loaded[p]["tiles"][(index, "Reconstruction")] for p in PERCENTS]
        for panel_index, (label, pixels, color) in enumerate(zip(panels, images, colors)):
            x, y = left + PADDING + panel_index * (TILE + TILE_GAP), top + 86
            draw.text((x + TILE // 2, top + 54), label, anchor="ma", font=fonts[21], fill=color)
            box = (x, y, x + TILE, y + TILE)
            canvas.paste(pixels, (x, y))
            placements.append({
                "global_index": index, "panel": label, "box": box,
                "source_rgb_sha256": digest(pixels.tobytes()),
            })
    draw.text((MARGIN, height - 48), "All four methods: global magnitude Top-k + zero-fill + CPA/FIA | Stored RGB pixels preserved",
              font=fonts[23], fill=muted)
    draw.text((width - MARGIN, height - 48), f"80 originals + 320 reconstructions | {width} x {height} px",
              anchor="ra", font=fonts[23], fill=muted)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / FILENAME
    staging = target.with_suffix(".pending.png")
    canvas.save(staging, format="PNG", compress_level=6)
    del canvas, draw
    with Image.open(staging) as saved:
        saved.load()
        if saved.size != (width, height) or saved.mode != "RGB":
            raise RuntimeError("Unexpected saved image format")
        for placement in placements:
            if digest(saved.crop(placement["box"]).tobytes()) != placement["source_rgb_sha256"]:
                raise RuntimeError("A saved tile differs from the authenticated source pixels")
    if len(placements) != 400 or len({(p["global_index"], p["panel"]) for p in placements}) != 400:
        raise RuntimeError("The figure must contain each of the 400 panels exactly once")
    for source in loaded.values():
        provenance = source["provenance"]
        folder = PROJECT / provenance["source_folder"]
        for name, expected_hash in provenance["source_hashes"].items():
            if file_hash(folder / name) != expected_hash:
                raise RuntimeError("An input artifact changed while the figure was rendered")
    staging.replace(target)
    manifest = {
        "purpose": "Same80 sparsification comparison from completed saved figures",
        "file": FILENAME, "sha256": file_hash(target), "size": [width, height],
        "n_images": 80, "n_panels": 400, "panel_order": list(panels),
        "method": "Naive zero-fill for each global magnitude Top-k retained ratio",
        "tile_size": TILE, "pixel_verification": "All 400 native RGB tiles verified exactly",
        "all_original_identities_and_pixels_equal_across_ratios": True,
        "global_indices": list(range(80)), "identity_fields": IDENTITY_FIELDS,
        "image_identities": identities, "placements": placements,
        "source_runs": {str(p): loaded[p]["provenance"] for p in PERCENTS},
        "new_attack_steps": 0, "matching_recomputed": False,
        "rendering": "Native 224x224 RGB tiles; no resizing, enhancement, or synthesis",
    }
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"file": str(target), "size": [width, height], "verified_panels": len(placements)}), flush=True)


if __name__ == "__main__":
    main()
