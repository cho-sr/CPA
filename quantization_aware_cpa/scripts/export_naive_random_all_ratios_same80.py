"""Export saved same80 sparse results, with explicit gaps for unavailable runs.

Figure assembly only: no attacks, new matching, resizing, or image synthesis.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from export_topk_all_ratios_same80 import IDENTITY_FIELDS
from export_topk_same80_slides import PERCENTS, PROJECT, digest, file_hash, font, load_source, read_json

OUTPUT = PROJECT / "exports/naive_vs_topk50_comparison"
FILENAME = "sparsification_naive_random_keep50_30_20_10_all80.png"
RANDOM10 = PROJECT / "results/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k"
TILE, COLUMNS, MARGIN, PADDING = 224, 4, 40, 16
TILE_GAP, PAIR_GAP, BLOCK_GAP = 8, 20, 24
PAIR_WIDTH = 2 * TILE + TILE_GAP
BLOCK_WIDTH = 2 * PADDING + TILE + 4 * (PAIR_GAP + PAIR_WIDTH)
BLOCK_HEIGHT, HEADER, FOOTER = 368, 250, 108


def identities(rows: list) -> list:
    return [[row[key] for key in IDENTITY_FIELDS] for row in rows]


def load_random10(naive: dict) -> dict:
    summary = read_json(RANDOM10 / "summary.json")
    if summary["status"] != "complete" or summary["n_images"] != 80:
        raise RuntimeError("Random10 requires a completed same80 result")
    source_hashes = {name: summary["output_hashes"][name]
                     for name in ("all80_rec.pt", "per_image_metrics.csv")}
    source_hashes["summary.json"] = file_hash(RANDOM10 / "summary.json")
    for name, expected in source_hashes.items():
        if file_hash(RANDOM10 / name) != expected:
            raise RuntimeError(f"Random input hash mismatch: {name}")
    baseline_folder = PROJECT / naive["provenance"]["source_folder"]
    if file_hash(baseline_folder / "summary.json") != summary["identity"]["baseline_summary_sha256"]:
        raise RuntimeError("Random10 references a different baseline")
    baseline = read_json(baseline_folder / "summary.json")
    for key in ("n", "T", "batch_indices", "source_output", "local_training", "model", "stage1", "fia", "evaluation"):
        if baseline["config"][key] != summary["config"][key]:
            raise RuntimeError(f"Naive and Random protocol mismatch: {key}")
    if summary["config"]["keep_percent"] != 10 or summary["randomization"]["candidates"] != 20:
        raise RuntimeError("Unexpected Random10 keep ratio or candidate budget")
    packets = summary["identity"]["packets"]
    if [p["batch_index"] for p in packets] != list(range(5)):
        raise RuntimeError("Random10 packet coverage differs")
    for packet, stored in zip(packets, baseline["communication_per_batch"]):
        expected_path = baseline_folder / f"batch{packet['batch_index']:02d}/observation.bin"
        if Path(packet["path"]).resolve() != expected_path.resolve() or packet["sha256"] != stored["sha256"]:
            raise RuntimeError("Saved packet provenance differs between methods")
    with (RANDOM10 / "per_image_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if identities(rows) != identities(naive["rows"]):
        raise RuntimeError("Random10 uses different images or ordering")
    rec = torch.load(RANDOM10 / "all80_rec.pt", map_location="cpu", weights_only=True)
    if rec.shape != (80, 3, TILE, TILE) or not torch.isfinite(rec).all():
        raise RuntimeError("Invalid Random10 reconstruction tensor")
    if rec.min().item() < 0 or rec.max().item() > 1:
        raise RuntimeError("Expected saved Random10 pixels in [0,1]")
    rgb = (rec.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    return {
        "tiles": [Image.fromarray(pixels) for pixels in rgb],
        "provenance": {"run_tag": summary["run_tag"], "source_folder": str(RANDOM10.relative_to(PROJECT)),
                       "source_hashes": source_hashes, "randomization": summary["randomization"],
                       "baseline_summary_sha256": summary["identity"]["baseline_summary_sha256"],
                       "same_packet_hashes_in_saved_provenance": True},
    }


def missing_run_snapshot() -> dict:
    missing = {}
    for percent in (50, 30, 20):
        checked = []
        for candidates in (8, 20):
            folder = PROJECT / f"results/cpa_fia_random_topk{percent}_same80_R{candidates}_S5k_K3_F25k"
            summary_path = folder / "summary.json"
            summary = read_json(summary_path) if summary_path.exists() else {}
            if summary.get("status") == "complete" and summary.get("n_images") == 80:
                raise RuntimeError(f"New completed result found; review before marking N/A: {folder}")
            checked.append({"source_folder": str(folder.relative_to(PROJECT)),
                            "summary_exists": summary_path.exists(),
                            "all80_rec_exists": (folder / "all80_rec.pt").exists()})
        missing[str(percent)] = {"reason": "No completed protocol-matched same80 Random result", "checked": checked}
    return missing


def main() -> None:
    torch.set_num_threads(2)
    naive = {}
    for percent in PERCENTS:
        rows, tiles, metrics, provenance = load_source(percent)
        provenance["source_hashes"]["summary.json"] = provenance["summary_sha256"]
        naive[percent] = {"rows": rows, "tiles": tiles, "metrics": metrics, "provenance": provenance}
        if identities(rows) != identities(naive[50]["rows"]):
            raise RuntimeError("Naive image identities differ across ratios")
        for index in range(80):
            if tiles[index, "Original"].tobytes() != naive[50]["tiles"][index, "Original"].tobytes():
                raise RuntimeError("Original pixels differ across ratios")
    random10 = load_random10(naive[10])
    unavailable = missing_run_snapshot()
    width = 2 * MARGIN + COLUMNS * BLOCK_WIDTH + (COLUMNS - 1) * BLOCK_GAP
    height = HEADER + 20 * BLOCK_HEIGHT + 19 * BLOCK_GAP + FOOTER
    canvas = Image.new("RGB", (width, height), "#f1f5f9")
    draw = ImageDraw.Draw(canvas)
    ink, muted, blue, orange = "#15243b", "#536176", "#245fa0", "#ad571b"
    fonts = {size: font(size, size in (56, 30, 24, 22)) for size in (56, 30, 28, 26, 24, 22, 20, 18)}
    draw.text((MARGIN, 28), "SPARSIFICATION | NAIVE vs RANDOM | ALL 80 IMAGES", font=fonts[56], fill=ink)
    draw.text((MARGIN, 106), "Each image: Original  >  Keep 50% [Naive | Random]  >  Keep 30% [Naive | Random]  >  Keep 20% [Naive | Random]  >  Keep 10% [Naive | Random]",
              font=fonts[30], fill=ink)
    draw.text((MARGIN, 157), "Naive = zero-fill | Random = uniform random-fill | Keep = retained coordinates in FC1 | VGG16 | 5 batches x 16 | Local SGD T = 10",
              font=fonts[26], fill=muted)
    draw.text((MARGIN, 201), "Random 10%: completed R20 result. Random 50/30/20%: no completed same80 result; shown as N/A.", font=fonts[26], fill=orange)
    draw.text((width - MARGIN, 201), "Original image order 00-79 | Native 224 x 224 pixels", anchor="ra", font=fonts[26], fill=muted)

    placements = []
    for index in range(80):
        left = MARGIN + (index % COLUMNS) * (BLOCK_WIDTH + BLOCK_GAP)
        top = HEADER + (index // COLUMNS) * (BLOCK_HEIGHT + BLOCK_GAP)
        draw.rounded_rectangle((left, top, left + BLOCK_WIDTH, top + BLOCK_HEIGHT), radius=12,
                               fill="white", outline="#d4dde7", width=2)
        draw.text((left + PADDING, top + 14), f"Image {index:02d}", font=fonts[24], fill=ink)
        draw.text((left + BLOCK_WIDTH - PADDING, top + 16), f"Batch {index // 16} | Index {index % 16:02d}",
                  anchor="ra", font=fonts[22], fill=muted)
        original_x, tile_y = left + PADDING, top + 122
        draw.text((original_x + TILE // 2, top + 82), "Original", anchor="ma", font=fonts[22], fill=muted)
        panels = [("Original", None, original_x, naive[50]["tiles"][index, "Original"])]
        for ratio_index, percent in enumerate(PERCENTS):
            pair_x = original_x + TILE + PAIR_GAP + ratio_index * (PAIR_WIDTH + PAIR_GAP)
            draw.rounded_rectangle((pair_x, top + 47, pair_x + PAIR_WIDTH, top + 78), radius=5, fill="#eaf0f6")
            draw.text((pair_x + PAIR_WIDTH // 2, top + 50), f"Keep {percent}%", anchor="ma", font=fonts[22], fill=ink)
            for method_index, (method, color) in enumerate((("Naive", blue), ("Random", orange))):
                x = pair_x + method_index * (TILE + TILE_GAP)
                draw.text((x + TILE // 2, top + 87), method, anchor="ma", font=fonts[22], fill=color)
                pixels = naive[percent]["tiles"][index, "Reconstruction"] if method == "Naive" else (
                    random10["tiles"][index] if percent == 10 else None)
                panels.append((method, percent, x, pixels))
        for method, percent, x, pixels in panels:
            box = (x, tile_y, x + TILE, tile_y + TILE)
            available = pixels is not None
            if available:
                canvas.paste(pixels, box[:2])
            else:
                draw.rectangle((x, tile_y, x + TILE - 1, tile_y + TILE - 1), fill="#edf0f4", outline="#d4dde7")
                draw.text((x + TILE // 2, tile_y + 65), "N/A", anchor="ma", font=fonts[30], fill="#748195")
                draw.text((x + TILE // 2, tile_y + 112), "No completed", anchor="ma", font=fonts[20], fill=muted)
                draw.text((x + TILE // 2, tile_y + 140), "same80 result", anchor="ma", font=fonts[20], fill=muted)
            placements.append({"global_index": index, "method": method, "keep_percent": percent,
                               "box": box, "available": available,
                               "rgb_sha256": digest((pixels if available else canvas.crop(box)).tobytes())})
    draw.text((MARGIN, height - 85), "Search budget differs: Naive = 1 candidate, FIA 25k | Random 10% = 20 candidates at FIA 5k; top 3 continued to total 25k, selected by observed-update consistency.",
              font=fonts[26], fill=muted)
    draw.text((MARGIN, height - 43), "Saved reconstructions only | No new attacks or image matching | N/A is unavailable data, not a failed reconstruction", font=fonts[24], fill=muted)
    draw.text((width - MARGIN, height - 43), f"80 originals + 400 reconstructions + 240 N/A | {width} x {height} px", anchor="ra", font=fonts[24], fill=muted)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / FILENAME
    staging = target.with_suffix(".pending.png")
    canvas.save(staging, format="PNG", compress_level=6)
    del canvas, draw
    with Image.open(staging) as saved:
        saved.load()
        if saved.size != (width, height) or saved.mode != "RGB":
            raise RuntimeError("Unexpected output format")
        for item in placements:
            if digest(saved.crop(item["box"]).tobytes()) != item["rgb_sha256"]:
                raise RuntimeError("Saved tile pixels changed")
    unique = {(p["global_index"], p["method"], p["keep_percent"]) for p in placements}
    if len(placements) != 720 or len(unique) != 720 or sum(p["available"] for p in placements) != 480:
        raise RuntimeError("Unexpected image or placeholder coverage")
    sources = {f"naive_keep{p}": naive[p]["provenance"] for p in PERCENTS}
    sources["random_keep10"] = random10["provenance"]
    for source in sources.values():
        for name, expected in source["source_hashes"].items():
            if file_hash(PROJECT / source["source_folder"] / name) != expected:
                raise RuntimeError("An input changed during export")
    staging.replace(target)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "file": FILENAME, "sha256": file_hash(target), "size": [width, height],
        "n_images": 80, "n_panels": 720, "available_panels": 480, "unavailable_panels": 240,
        "panel_order": ["Original"] + [f"Keep {p}% {method}" for p in PERCENTS for method in ("Naive", "Random")],
        "identity_fields": IDENTITY_FIELDS, "image_identities": identities(naive[50]["rows"]),
        "placements": placements, "sources": sources, "unavailable_random": unavailable,
        "naive_definition": "Global magnitude Top-k followed by zero-fill and CPA/FIA",
        "random_definition": "Uniform random-fill of missing coordinates, preserving the same received packet",
        "search_compute_matched": False, "new_attack_steps": 0, "matching_recomputed": False,
        "rendering": "Native 224x224 RGB; no resizing, enhancement, or synthesis; missing results are explicit N/A",
        "verification": "All 720 slots checked; 480 image tiles match authenticated saved inputs exactly",
    }
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"file": str(target), "size": [width, height], "verified_tiles": 480, "missing_slots": 240}), flush=True)


if __name__ == "__main__":
    main()
