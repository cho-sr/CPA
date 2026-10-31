"""Compose existing CIFAR and matched ImageNet sparse results; no experiments."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import build_cifar100_selected_panel as selected
import build_comparison_panels as existing

ROOT, OUT = selected.ROOT, selected.OUT
KEEP = (50, 30, 20, 10)
INDICES = selected.SELECTED
TILES = OUT / "cifar100_sparse_examples11_24"
SOURCES = {}


def track(path):
    path = path.resolve()
    SOURCES[path] = selected.sha(path)
    return path


def cifar_rows():
    index = json.loads(track(ROOT / "cifar100_png/index.json").read_text(encoding="utf-8"))
    conditions = {row["condition"]: row for row in index["conditions"]}
    with Image.open(track(ROOT / "cifar100_png/bits8_24.png")) as image:
        quantization = image.convert("RGB")
    images, records, checked = {}, [], 0
    for keep in KEEP:
        assert conditions[f"topk{keep}"]["completed_images"] == 24
        source = track(ROOT / f"cifar100_png/topk{keep}_24.png")
        with Image.open(source) as image:
            full = image.convert("RGB")
        assert full.size == (840, 982)
        for batch in range(3):
            with Image.open(track(ROOT / f"cifar100_png/topk{keep}_batch{batch:02d}.png")) as image:
                single = image.convert("RGB")
            assert single.size == (840, 394)
            for ordinal in range(batch*8, batch*8+8):
                for method in range(3):
                    box = selected.crop_box(ordinal, method)
                    data = np.asarray(full.crop(box)).copy()
                    assert np.array_equal(data, np.asarray(single.crop(selected.crop_box(ordinal, method, True))))
                    checked += 1
                    if method == 0:
                        assert np.array_equal(data, np.asarray(quantization.crop(box)))
                    if ordinal in INDICES:
                        images[ordinal, keep, method] = data
                        records.append({
                            "display_ordinal_one_based": ordinal+1, "keep_percent": keep,
                            "method": ("original", "naive", "random_R8")[method],
                            "source": source.relative_to(ROOT).as_posix(), "crop_xyxy": box,
                            "rgb_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                        })
    TILES.mkdir(exist_ok=True)
    rows = []
    for ordinal in INDICES:
        row = [images[ordinal, 50, 0]]
        output = [(f"example{ordinal+1}_original.png", row[0])]
        for keep in KEEP:
            for method, name in ((1, "naive"), (2, "random")):
                tile = images[ordinal, keep, method]
                row.append(tile)
                output.append((f"example{ordinal+1}_keep{keep}_{name}.png", tile))
        for name, tile in output:
            path = TILES / name
            Image.fromarray(tile).save(path)
            with Image.open(path) as saved:
                assert np.array_equal(np.asarray(saved), tile)
        rows.append((str(ordinal+1), row))
    return rows, records, checked


def imagenet_rows():
    result_root = ROOT / "quantization_aware_cpa/results"
    source = existing.load_tensor(result_root / "m6_uia_fp32_B16_fia/batch00/inputs.pt")
    columns = [existing.unnormalize(source["x"])[existing.INDICES]]
    identities = None
    for keep in KEEP:
        folder = result_root / f"cpa_fia_topk{keep}_same80_B16"
        rows = existing.read_csv(folder / "per_image_metrics.csv")
        current = [(r["global_index"], r["dataset_index"], r["image_path"]) for r in rows]
        if identities is None:
            identities = current
        assert current == identities
        for i in existing.INDICES:
            assert int(rows[i]["dataset_index"]) == source["metadata"][i]["dataset_index"]
        columns.append(existing.load_tensor(folder / "all80_rec.pt")[existing.INDICES])
        if keep == 10:
            random_root = result_root / "cpa_fia_random_topk10_same80_R20_S5k_K3_F25k"
            random_rows = existing.read_csv(random_root / "per_image_metrics.csv")
            assert identities == [(r["global_index"], r["dataset_index"], r["image_path"]) for r in random_rows]
            columns.append(existing.load_tensor(random_root / "all80_rec.pt")[existing.INDICES])
        else:
            columns.append(None)
    SOURCES.update(existing.SOURCES)
    previous = json.loads(track(OUT / "comparison_manifest.json").read_text(encoding="utf-8"))
    for relative, digest in previous["sources_sha256"].items():
        path = (ROOT / relative).resolve()
        if path in SOURCES:
            assert SOURCES[path] == digest, "Previously displayed ImageNet data changed"
    return [(str(index), [existing.rgb(column[pos]) if column is not None else None for column in columns])
            for pos, index in enumerate(existing.INDICES)]


def fill_supplied_random(rows, directory):
    """Extract whole displayed tiles; keep supplied g00/g01 pairing, no rematching."""
    records = []
    for keep, column in ((50, 2), (30, 4), (20, 6)):
        path = track(directory / f"topk{keep}_randomR8_S5k_K3_F25k_batch00.png")
        full_path = track(directory / f"topk{keep}_randomR8_S5k_K3_F25k_all80.png")
        with Image.open(path) as image:
            sheet = image.convert("RGB")
        with Image.open(full_path) as image:
            full = image.convert("RGB")
        assert sheet.size == (1069, 649) and full.size == (1069, 3225)
        for index, (_, images) in zip(existing.INDICES, rows):
            assert index in (0, 1) and images[column] is None
            x = 5 + 266 * index
            original_box = (x, 33, x+128, 161)
            rec_box = (x+133, 33, x+261, 161)
            expected = Image.fromarray(np.rint(images[0]*255).astype(np.uint8)).resize(
                (128, 128), Image.Resampling.LANCZOS)
            assert np.array_equal(np.asarray(expected), np.asarray(sheet.crop(original_box))), "Original identity mismatch"
            for box in (original_box, rec_box):
                assert np.array_equal(np.asarray(sheet.crop(box)), np.asarray(full.crop(box))), "Batch/all80 tile mismatch"
            tile = np.asarray(sheet.crop(rec_box)).copy()
            images[column] = tile
            records.append({"global_index": index, "keep_percent": keep,
                            "method": "random_R8", "source": path.relative_to(ROOT).as_posix(),
                            "crop_xyxy": rec_box, "original_crop_xyxy": original_box,
                            "original_matches_existing_tensor_after_source_export_resize": True,
                            "batch_and_all80_tiles_equal": True,
                            "rgb_sha256": hashlib.sha256(tile.tobytes()).hexdigest()})
    return records


def vertical_panel(groups, levels, formats):
    """Reproduce the reference PDF's 86mm, five-column paired-row layout."""
    width = 86 / 25.4
    left = right = .02
    step = (width-left-right) / 5
    cell = step-.015
    heading, sample_gap, group_gap, footer = .22, .05, .12, .025
    height = heading + sum(len(rows)*(2*step+sample_gap)+group_gap for _, rows in groups) + footer
    fig = plt.figure(figsize=(width, height), facecolor="white")
    for col, level in enumerate(levels):
        fig.text((left+(col+1.5)*step)/width, 1-.085/height, level,
                 ha="center", va="center", fontsize=6, weight="bold")
    cursor = heading
    for _, rows in groups:
        for _, images in rows:
            for method in (1, 2):
                columns = [images[0]] + [images[method+2*k] for k in range(4)]
                for col, data in enumerate(columns):
                    assert data is not None
                    ax = fig.add_axes([(left+col*step)/width, 1-(cursor+cell)/height,
                                       cell/width, cell/height])
                    artist = ax.imshow(data, interpolation="none")
                    assert np.array_equal(np.asarray(artist.get_array()), data)
                    ax.set_xticks([]); ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_linewidth(.25); spine.set_edgecolor("#b0b0b0")
                cursor += step
            cursor += sample_gap
        cursor += group_gap
    for extension in formats:
        fig.savefig(OUT / f"sparsification_cifar100_imagenet.{extension}", dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png-only", action="store_true", help="Leave PDF assets and manuscript untouched")
    parser.add_argument("--imagenet-random-png-dir", type=Path,
                        help="Fill combined figure from supplied paired R8 contact sheets")
    parser.add_argument("--layout", choices=("horizontal", "reference-pdf"), default="horizontal")
    args = parser.parse_args()
    if args.imagenet_random_png_dir and not args.png_only and args.layout != "reference-pdf":
        parser.error("Horizontal supplied PNG composition requires --png-only")
    formats = ("png",) if args.png_only else ("png", "pdf")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    cifar, records, checked = cifar_rows()
    imagenet = imagenet_rows()
    supplied = fill_supplied_random(imagenet, args.imagenet_random_png_dir.resolve()) if args.imagenet_random_png_dir else []
    groups = [("CIFAR-100", cifar), ("ImageNet", imagenet)]
    levels = [f"Keep {keep}%" for keep in KEEP]
    if not supplied:
        selected.panel("cifar100_sparsification_examples11_24", groups[:1], levels, formats=formats)
    if args.layout == "reference-pdf":
        vertical_panel(groups, levels, formats)
    else:
        selected.panel("sparsification_cifar100_imagenet", groups, levels, formats=formats)
    for path, digest in SOURCES.items():
        assert selected.sha(path) == digest, f"Source changed: {path}"
    stems = ("sparsification_cifar100_imagenet",) if supplied else ("cifar100_sparsification_examples11_24", "sparsification_cifar100_imagenet")
    manifest = {
        "purpose": "Document composition from existing artifacts only",
        "selected_display_ordinals": [11, 24], "selection": "Same user-selected examples as the quantization figure; no sparse-specific reselection",
        "keep_percent": list(KEEP), "cifar_tiles": records,
        "layout": {"width_mm": 86, "dataset_titles": [], "image_numbers_shown": False,
                   "annotations_in_caption": ["datasets", "methods", "original column", "selection"],
                   "method_pairing": ("Original column, four condition columns; Naive above Random per example"
                                      if args.layout == "reference-pdf" else "Original, then Naive/Random adjacent columns for each condition"),
                   "layout_style": args.layout,
                   "generated_formats": list(formats), "pdf_updated": not args.png_only},
        "all_24_originals_equal_to_quantization": True,
        "full_vs_batch_tile_equalities": checked,
        "exported_tile_pixels_unchanged": True,
        "imagenet_indices": [0, 1], "imagenet_random_available_keep": list(KEEP) if supplied else [10],
        "imagenet_na_cells": 0 if supplied else 6,
        "imagenet_supplied_random_tiles": supplied,
        "protocols": ("CIFAR supplied Random R8; ImageNet keep50/30/20 supplied Random R8 PNGs, keep10 existing same80/B16/T10/R20. Candidate budgets differ across fractions."
                      if supplied else "CIFAR supplied Random R8; ImageNet existing matched same80/B16/T10/R20. Do not substitute fresh16 partial results or older incompatible experiments."),
        "matching": "Saved PNG pairings and already-aligned ImageNet tensors retained; supplied originals checked against existing originals, no candidate reselection",
        "provenance_limit": "Supplied CIFAR and new ImageNet PNG folders do not include raw checkpoints or full matching/configuration records; original identity and displayed pairing verified, not full protocol equivalence",
        "new_attack_steps": 0, "metrics_recomputed": False, "matching_recomputed": False,
        "enhancement": False, "averaging": False, "candidate_reselection": False,
        "sources_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in SOURCES.items()},
        "outputs_sha256": {f"paper_assets/{stem}.{ext}": selected.sha(OUT / f"{stem}.{ext}") for stem in stems for ext in formats},
    }
    (OUT / "cifar100_sparse_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"tile_equalities": checked, "selected_examples": [11, 24],
                      "sources_unchanged": True, "outputs": list(manifest["outputs_sha256"])}))


if __name__ == "__main__":
    main()
