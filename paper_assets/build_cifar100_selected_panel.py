"""Typeset supplied CIFAR PNG tiles and existing ImageNet artifacts, not attacks.

Examples 11 and 24 were selected post hoc by the user after visual inspection,
to illustrate low-bit Random versus Naive behavior across all bit widths.
No metric, matching, candidate selection, or reconstruction is recomputed.
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

import build_comparison_panels as existing

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
BITS = (8, 7, 6, 5, 4)
SELECTED = (10, 23)  # Zero-based display ordinals, not verified dataset IDs.
TILES = OUT / "cifar100_selected_examples11_24"
SOURCES = {}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def track(path):
    path = Path(path).resolve()
    SOURCES[path] = sha(path)
    return path


def crop_box(ordinal, method, batch_sheet=False):
    batch, column = divmod(ordinal, 8)
    x = 156 + 84 * column
    y = 89 + (0 if batch_sheet else 294 * batch) + 86 * method
    return (x, y, x + 80, y + 80)


def cifar_columns():
    index_path = track(ROOT / "cifar100_png/index.json")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    conditions = {row["condition"]: row for row in index["conditions"]}
    originals, tiles, provenance = {}, {}, []
    checked = 0
    for bits in BITS:
        assert conditions[f"bits{bits}"]["completed_images"] == 24
        full_path = track(ROOT / f"cifar100_png/bits{bits}_24.png")
        with Image.open(full_path) as loaded:
            sheet = loaded.convert("RGB")
        assert sheet.size == (840, 982), "Source layout changed; review coordinates"
        for batch in range(3):
            batch_path = track(ROOT / f"cifar100_png/bits{bits}_batch{batch:02d}.png")
            with Image.open(batch_path) as loaded:
                single = loaded.convert("RGB")
            assert single.size == (840, 394)
            for ordinal in range(batch * 8, batch * 8 + 8):
                for method in range(3):
                    tile = np.asarray(sheet.crop(crop_box(ordinal, method))).copy()
                    check = np.asarray(single.crop(crop_box(ordinal, method, True)))
                    assert np.array_equal(tile, check), "Batch and full sheet disagree"
                    checked += 1
                    if method == 0:
                        if ordinal in originals:
                            assert np.array_equal(originals[ordinal], tile), "Original identity changed across bits"
                        else:
                            originals[ordinal] = tile
                    if ordinal in SELECTED:
                        tiles[ordinal, bits, method] = tile
                        provenance.append({
                            "display_ordinal_one_based": ordinal+1,
                            "bits": bits, "method": ("original", "naive", "random_R8")[method],
                            "source": full_path.relative_to(ROOT).as_posix(),
                            "crop_xyxy": crop_box(ordinal, method),
                            "rgb_sha256": hashlib.sha256(tile.tobytes()).hexdigest(),
                        })
    TILES.mkdir(exist_ok=True)
    outputs, rows = [], []
    for ordinal in SELECTED:
        prefix = f"example{ordinal+1:02d}"
        outputs.append((f"{prefix}_original.png", tiles[ordinal, 8, 0]))
        columns = [tiles[ordinal, 8, 0]]
        for bits in BITS:
            for method, name in ((1, "naive"), (2, "random")):
                tile = tiles[ordinal, bits, method]
                outputs.append((f"{prefix}_bits{bits}_{name}.png", tile))
                columns.append(tile)
        rows.append(columns)
    for name, tile in outputs:
        path = TILES / name
        Image.fromarray(tile).save(path)
        with Image.open(path) as saved:
            assert np.array_equal(np.asarray(saved), tile)
    return rows, provenance, checked


def imagenet_columns():
    """Reuse the old figure's exact two identities and stored assignments."""
    columns, assignments, original = [], {}, None
    for bits in BITS:
        folder = (
            "signed_int8_screen3_repair_B16_T10_20260918/evaluation_final_20260918"
            if bits == 8 else
            f"signed_int{bits}_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918"
        )
        report = existing.read_json(ROOT / "quantization_aware_cpa/results_private" / folder / "comparison.json")
        truth = existing.unnormalize(existing.load_tensor(report["private_truth"]["path"])["x"])[existing.INDICES]
        if original is None:
            original = truth
            columns.append(truth)
        else:
            assert torch.equal(original, truth)
        names = ["center_25000", next(name for name in report["mean_metrics"] if name.startswith("rank1_"))]
        for name in names:
            candidate = existing.load_tensor(report["candidate_paths"][name])["pixels"]
            saved = {int(row["truth_index"]): int(row["candidate_index"])
                     for row in report["image_metrics"][name]["per_image"]}
            order = [saved[i] for i in existing.INDICES]
            columns.append(candidate[order])
            assignments[f"{bits}:{name}"] = order
    SOURCES.update(existing.SOURCES)
    previous = json.loads(track(OUT / "comparison_manifest.json").read_text(encoding="utf-8"))
    assert assignments == previous["quantization_saved_candidate_indices"]
    rows = [[existing.rgb(column[row]) for column in columns] for row in range(2)]
    return rows, assignments


def panel(filename, groups, levels=None, formats=("png", "pdf")):
    """Keep the horizontal method pairs; shrink the figure to one column."""
    width = 86 / 25.4  # spconf: (178mm text width - 6mm gutter) / 2.
    left, right = .02, .02
    levels = levels or [f"{bits}-bit" for bits in BITS]
    ncols = 1 + 2*len(levels)
    step = (width - left - right) / ncols
    cell = step - .01
    heading = .20
    group_heading = 0
    group_gap = .06
    sample_gap = .01
    footer = .025
    height = heading + sum(group_heading + len(rows)*(step+sample_gap) + group_gap
                           for _, rows in groups) + footer
    fig = plt.figure(figsize=(width, height), facecolor="white")
    def label(x, y, text, **kwargs):
        fig.text(x/width, 1-y/height, text, **kwargs)
    for col, level in enumerate(levels):
        label(left+(2+2*col)*step, .085, level, ha="center", va="center",
              fontsize=6, weight="bold")
    cursor = heading
    for _title, rows in groups:
        cursor += group_heading
        for _row_label, columns in rows:
            assert len(columns) == 1+2*len(levels)
            for col, data in enumerate(columns):
                ax = fig.add_axes([(left+col*step)/width, 1-(cursor+cell)/height, cell/width, cell/height])
                if data is None:
                    ax.set_facecolor("#f3f3f3")
                    ax.text(.5, .5, "N/A", ha="center", va="center", fontsize=5)
                else:
                    artist = ax.imshow(data, interpolation="none")
                    assert np.array_equal(np.asarray(artist.get_array()), data)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(.25); spine.set_edgecolor("#b0b0b0")
            cursor += step + sample_gap
        cursor += group_gap
    for extension in formats:
        fig.savefig(OUT / f"{filename}.{extension}", dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png-only", action="store_true", help="Leave PDF assets and manuscript untouched")
    args = parser.parse_args()
    formats = ("png",) if args.png_only else ("png", "pdf")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    cifar, provenance, checked = cifar_columns()
    cifar_group = ("CIFAR-100",
                   [(str(i+1), row) for i, row in zip(SELECTED, cifar)])
    panel("cifar100_quantization_selected_examples11_24", [cifar_group], formats=formats)
    imagenet, assignments = imagenet_columns()
    panel("quantization_cifar100_imagenet", [cifar_group,
          ("ImageNet", [(str(i), row) for i, row in enumerate(imagenet)])], formats=formats)
    for path, digest in SOURCES.items():
        assert sha(path) == digest, f"Source changed during rendering: {path}"
    manifest = {
        "purpose": "Manuscript asset composition only",
        "selection": {
            "display_ordinals_one_based": [i+1 for i in SELECTED],
            "display_indices_zero_based": list(SELECTED),
            "batch_and_column": [{"batch": i//8, "column_one_based": i%8+1} for i in SELECTED],
            "scope": "24 displayed examples, all five bit widths visually inspected",
            "rule": "User-selected examples 11 and 24 after visual inspection; low-bit qualitative comparison, not a quantitative or preregistered selection",
            "supersedes": "The initial editorial proposal of example03; example03 assets are preserved but not used in the manuscript",
            "quantitative_optimum_claimed": False,
            "dataset_sample_id": None, "class_label": None,
            "not_representative_of_full_set": True,
        },
        "bits": list(BITS), "cifar_tiles": provenance,
        "layout": {"width_mm": 86, "dataset_titles": [], "image_numbers_shown": False,
                   "annotations_in_caption": ["datasets", "methods", "original column", "selection"],
                   "method_pairing": "Original, then Naive/Random adjacent columns for each condition",
                   "generated_formats": list(formats), "pdf_updated": not args.png_only},
        "validation": {"full_vs_batch_tile_equalities": checked,
                       "all_24_originals_identical_across_bits": True,
                       "exported_tile_pixels_unchanged": True,
                       "imagenet_saved_assignments_unchanged": True},
        "imagenet_indices": existing.INDICES,
        "imagenet_saved_assignments": assignments,
        "provenance_limit": "CIFAR images are supplied rendered PNGs; underlying checkpoints, per-image metrics and matching records are not in this folder. Displayed pairing is preserved, not independently recomputed.",
        "new_attack_steps": 0, "metrics_recomputed": False, "matching_recomputed": False,
        "random_candidate_reselected": False,
        "rendering": "Whole 80x80 CIFAR tiles extracted losslessly; no image-content crop, color correction, enhancement, interpolation smoothing, or averaging. Existing ImageNet RGB and saved assignments reused.",
        "sources_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in SOURCES.items()},
        "outputs_sha256": {path.relative_to(ROOT).as_posix(): sha(path)
                           for path in [OUT/f"{stem}.{ext}" for stem in
                           ("cifar100_quantization_selected_examples11_24", "quantization_cifar100_imagenet")
                           for ext in formats]},
    }
    (OUT / "cifar100_selection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"selected_examples": [11, 24], "tile_equalities": checked,
                      "assets": list(manifest["outputs_sha256"]), "sources_unchanged": True}))


if __name__ == "__main__":
    main()
