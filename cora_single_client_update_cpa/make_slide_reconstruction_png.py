#!/usr/bin/env python3
"""Create 16:9 slide-friendly original/reconstruction metric panels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
SPARSE_SRC = REPO_ROOT / "sparsified_update_cpa"
if str(SPARSE_SRC) not in sys.path:
    sys.path.insert(0, str(SPARSE_SRC))

from common import read_pickle  # noqa: E402
from make_reconstruction_metric_png import (  # noqa: E402
    build_dataset_original_loader,
    build_update_file_original_loader,
    chw_to_pil,
    format_metric,
    infer_dataset_from_run_dir,
    load_font,
    load_manifest,
    load_metric_rows,
    metric_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--metrics_csv", type=Path, default=None)
    parser.add_argument("--ds", default=None)
    parser.add_argument("--source", choices=["dataset", "update_file"], default="dataset")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--canvas_width", type=int, default=2560)
    parser.add_argument("--canvas_height", type=int, default=1440)
    parser.add_argument("--samples_per_row", type=int, default=8)
    parser.add_argument("--max_samples_per_batch", type=int, default=24)
    parser.add_argument("--success_lpips_threshold", type=float, default=0.5)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def success_color(metric: Mapping[str, str], threshold: float) -> tuple[bool, tuple[int, int, int], tuple[int, int, int]]:
    value = metric.get("lpips")
    ok = value not in (None, "") and float(value) <= threshold
    if ok:
        return True, (22, 163, 74), (240, 253, 244)
    return False, (148, 163, 184), (248, 250, 252)


def draw_fit_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
) -> None:
    value = text
    while len(value) > 3 and draw.textbbox((0, 0), value, font=font)[2] > max_width:
        value = value[:-2]
    if value != text:
        value = value[:-1] + "."
    draw.text(xy, value, fill=fill, font=font)


def draw_sample_card(
    canvas: Image.Image,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    original: np.ndarray,
    reconstruction: np.ndarray,
    metric: Mapping[str, str],
    ds: str,
    threshold: float,
    fonts: Mapping[str, ImageFont.ImageFont],
) -> None:
    draw = ImageDraw.Draw(canvas)
    ok, border, fill = success_color(metric, threshold)
    radius = 8
    draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=border, width=4 if ok else 3)

    pad = 10
    header_h = 28
    label_h = 22
    metric_h = 76
    image_gap = 8
    image_size = min((w - 2 * pad - image_gap) // 2, h - 2 * pad - header_h - label_h - metric_h)
    image_size = max(64, int(image_size))

    sample = int(metric.get("sample", 0))
    sample_index = metric.get("sample_index", "")
    draw_fit_text(
        draw,
        (x + pad, y + 7),
        f"s{sample:02d} idx {sample_index}",
        font=fonts["small"],
        fill=(15, 23, 42),
        max_width=w - 2 * pad,
    )

    image_y = y + pad + header_h + label_h
    orig_x = x + pad
    rec_x = orig_x + image_size + image_gap
    draw.text((orig_x, image_y - label_h + 2), "original", fill=(51, 65, 85), font=fonts["tiny"])
    draw.text((rec_x, image_y - label_h + 2), "recon", fill=(51, 65, 85), font=fonts["tiny"])
    canvas.paste(chw_to_pil(original, ds=ds, normalized=True, tile_size=image_size), (orig_x, image_y))
    canvas.paste(chw_to_pil(reconstruction, ds=ds, normalized=False, tile_size=image_size), (rec_x, image_y))
    if ok:
        draw.rectangle((rec_x, image_y, rec_x + image_size - 1, image_y + image_size - 1), outline=border, width=5)

    metric_y = image_y + image_size + 10
    lpips_fill = (22, 101, 52) if ok else (15, 23, 42)
    draw.text((x + pad, metric_y), f"LPIPS {format_metric(metric, 'lpips', 3)}", fill=lpips_fill, font=fonts["metric"])
    draw.text((x + pad, metric_y + 26), f"PSNR {format_metric(metric, 'psnr', 2)}", fill=(15, 23, 42), font=fonts["small"])
    draw.text((x + pad, metric_y + 49), f"SSIM {format_metric(metric, 'ssim', 3)}", fill=(15, 23, 42), font=fonts["small"])


def make_slide_panel(
    *,
    entry: Mapping[str, Any],
    originals: np.ndarray,
    metrics: Mapping[tuple[int, int], Mapping[str, str]],
    ds: str,
    out_path: Path,
    title: str,
    canvas_width: int,
    canvas_height: int,
    samples_per_row: int,
    max_samples: int,
    success_threshold: float,
) -> None:
    batch = int(entry["batch"])
    rec = np.asarray(entry["rec_fi_attack_order"], dtype=np.float32)
    n = min(len(rec), len(originals), max_samples)
    columns = max(1, min(samples_per_row, n))
    rows = max(1, (n + columns - 1) // columns)
    margin_x = 34
    margin_top = 78
    margin_bottom = 28
    gap_x = 12
    gap_y = 14
    card_w = int((canvas_width - 2 * margin_x - (columns - 1) * gap_x) / columns)
    card_h = int((canvas_height - margin_top - margin_bottom - (rows - 1) * gap_y) / rows)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    fonts = {
        "title": load_font(34),
        "subtitle": load_font(20),
        "metric": load_font(22),
        "small": load_font(18),
        "tiny": load_font(16),
    }
    draw.text((margin_x, 20), title, fill=(15, 23, 42), font=fonts["title"])
    draw.text((margin_x, 56), "green border: LPIPS <= threshold", fill=(71, 85, 105), font=fonts["subtitle"])

    for sample in range(n):
        metric = metrics.get(metric_key(batch, sample))
        if metric is None:
            raise KeyError(f"Missing metric row for batch={batch}, sample={sample}")
        row = sample // columns
        col = sample % columns
        draw_sample_card(
            canvas,
            x=margin_x + col * (card_w + gap_x),
            y=margin_top + row * (card_h + gap_y),
            w=card_w,
            h=card_h,
            original=originals[sample],
            reconstruction=rec[sample],
            metric=metric,
            ds=ds,
            threshold=success_threshold,
            fonts=fonts,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    ds = args.ds or manifest.get("ds") or manifest.get("dataset") or infer_dataset_from_run_dir(run_dir)
    if not ds:
        raise ValueError("--ds is required when dataset cannot be inferred")
    metrics_csv = args.metrics_csv or (run_dir / "image_metrics_by_sample.csv")
    metrics = load_metric_rows(metrics_csv)
    rec_entries = read_pickle(run_dir / "reconstructions.pkl")
    load_originals = (
        build_dataset_original_loader(str(ds))
        if args.source == "dataset"
        else build_update_file_original_loader(manifest)
    )

    batch_paths: list[Path] = []
    for entry in rec_entries:
        batch = int(entry["batch"])
        rec = np.asarray(entry["rec_fi_attack_order"], dtype=np.float32)
        n = min(len(rec), args.max_samples_per_batch)
        rows = [metrics[metric_key(batch, sample)] for sample in range(n)]
        positions = [
            int(row["original_sample"]) if row.get("original_sample") not in (None, "") else sample
            for sample, row in enumerate(rows)
        ]
        originals = load_originals(entry, positions)
        title = args.title or f"{run_dir.name} batch {batch}"
        out_path = args.output_dir / f"batch_{batch:03d}_original_reconstruction_metrics.png"
        make_slide_panel(
            entry=entry,
            originals=originals,
            metrics=metrics,
            ds=str(ds),
            out_path=out_path,
            title=title,
            canvas_width=int(args.canvas_width),
            canvas_height=int(args.canvas_height),
            samples_per_row=int(args.samples_per_row),
            max_samples=int(args.max_samples_per_batch),
            success_threshold=float(args.success_lpips_threshold),
        )
        batch_paths.append(out_path)

    if len(batch_paths) == 1:
        Image.open(batch_paths[0]).save(args.output_dir / "all_batches_original_reconstruction_metrics.png")
    else:
        # Multiple batches are uncommon for the slide view. Keep one 16:9 slide per batch
        # and copy the first one to the all-batches path for compatibility.
        Image.open(batch_paths[0]).save(args.output_dir / "all_batches_original_reconstruction_metrics.png")
    print(f"wrote {args.output_dir / 'all_batches_original_reconstruction_metrics.png'}")
    for path in batch_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
