#!/usr/bin/env python3
"""Create 16:9 slide PNGs for original/reconstruction panels."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import filesystem_path, read_json, read_pickle


DS_MEAN = {
    "tiny_imagenet": [0.485, 0.456, 0.406],
    "imagenet": [0.485, 0.456, 0.406],
}
DS_STD = {
    "tiny_imagenet": [0.229, 0.224, 0.225],
    "imagenet": [0.229, 0.224, 0.225],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--metrics_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ds", default=None)
    parser.add_argument("--source", choices=["update_file"], default="update_file")
    parser.add_argument("--title", required=True)
    parser.add_argument("--success_lpips_threshold", type=float, default=0.5)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def denormalize_chw(x: np.ndarray, ds: str) -> np.ndarray:
    mean = np.asarray(DS_MEAN[ds], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(DS_STD[ds], dtype=np.float32).reshape(3, 1, 1)
    return np.clip(x.astype(np.float32) * std + mean, 0.0, 1.0)


def chw_to_pil(x: np.ndarray, *, ds: str, normalized: bool, tile_size: int) -> Image.Image:
    if normalized:
        x = denormalize_chw(x, ds)
    else:
        x = np.clip(x.astype(np.float32), 0.0, 1.0)
    arr = np.transpose(x, (1, 2, 0))
    image = Image.fromarray((arr * 255.0).round().astype(np.uint8))
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:  # pragma: no cover - old Pillow fallback
        resample = Image.BILINEAR
    return image.resize((tile_size, tile_size), resample)


def load_metric_rows(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    with open(filesystem_path(path), newline="", encoding="utf-8") as handle:
        return {
            (int(row["batch"]), int(row["sample"])): row
            for row in csv.DictReader(handle)
        }


def fmt(row: Mapping[str, str], key: str, digits: int) -> str:
    value = row.get(key)
    if value in (None, ""):
        return "NA"
    return f"{float(value):.{digits}f}"


def metric_position(row: Mapping[str, str], sample: int) -> int:
    value = row.get("original_sample")
    if value not in (None, ""):
        return int(value)
    return sample


def metric_sample_index(
    row: Mapping[str, str],
    *,
    update_sample_indices: list[int],
    position: int,
) -> str:
    for key in ("matched_sample_index", "sample_index"):
        value = row.get(key)
        if value not in (None, ""):
            return str(int(float(value)))
    if 0 <= position < len(update_sample_indices):
        return str(int(update_sample_indices[position]))
    return ""


def draw_card(
    canvas: Image.Image,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    original: np.ndarray,
    reconstruction: np.ndarray,
    metric: Mapping[str, str],
    sample_index: str,
    ds: str,
    success_threshold: float,
    fonts: Mapping[str, ImageFont.ImageFont],
) -> None:
    draw = ImageDraw.Draw(canvas)
    sample = int(metric.get("sample", 0))
    lpips = float(metric["lpips"]) if metric.get("lpips") not in (None, "") else math.inf
    success = lpips <= success_threshold
    border = (22, 163, 74) if success else (148, 163, 184)
    fill = (240, 253, 244) if success else (248, 250, 252)
    text = (15, 23, 42)
    muted = (71, 85, 105)
    tile = 136
    inner = 9
    image_y = y + 58
    orig_x = x + inner
    rec_x = x + width - inner - tile

    draw.rounded_rectangle((x, y, x + width, y + height), radius=6, fill=fill, outline=border, width=3)
    draw.text((x + inner, y + 8), f"s{sample:02d} idx {sample_index}", fill=text, font=fonts["small"])
    draw.text((orig_x, y + 36), "original", fill=muted, font=fonts["small"])
    draw.text((rec_x, y + 36), "recon", fill=muted, font=fonts["small"])
    canvas.paste(chw_to_pil(original, ds=ds, normalized=True, tile_size=tile), (orig_x, image_y))
    canvas.paste(chw_to_pil(reconstruction, ds=ds, normalized=False, tile_size=tile), (rec_x, image_y))
    if success:
        draw.rectangle((rec_x, image_y, rec_x + tile - 1, image_y + tile - 1), outline=border, width=5)

    metric_y = image_y + tile + 12
    draw.text(
        (x + inner, metric_y),
        f"LPIPS {fmt(metric, 'lpips', 3)}",
        fill=(21, 128, 61) if success else text,
        font=fonts["metric"],
    )
    draw.text((x + inner, metric_y + 28), f"PSNR {fmt(metric, 'psnr', 2)}", fill=text, font=fonts["body"])
    draw.text((x + inner, metric_y + 52), f"SSIM {fmt(metric, 'ssim', 3)}", fill=text, font=fonts["body"])


def make_slide(
    *,
    entry: Mapping[str, Any],
    metrics: Mapping[tuple[int, int], Mapping[str, str]],
    originals_all: np.ndarray,
    sample_indices_all: list[int],
    ds: str,
    title: str,
    out_path: Path,
    success_threshold: float,
) -> None:
    width, height = 2560, 1440
    margin_x = 34
    top = 78
    bottom = 30
    gap_x = 14
    gap_y = 15
    cols = 8
    rec = np.asarray(entry["rec_fi_attack_order"], dtype=np.float32)
    n = int(rec.shape[0])
    rows = max(1, math.ceil(n / cols))
    card_w = (width - 2 * margin_x - (cols - 1) * gap_x) // cols
    card_h = (height - top - bottom - (rows - 1) * gap_y) // rows
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    fonts = {
        "title": load_font(34),
        "subtitle": load_font(18),
        "small": load_font(17),
        "body": load_font(19),
        "metric": load_font(22),
    }
    draw.text((margin_x, 18), title, fill=(15, 23, 42), font=fonts["title"])
    draw.text(
        (margin_x, 55),
        f"green border: LPIPS <= {success_threshold:g}",
        fill=(71, 85, 105),
        font=fonts["subtitle"],
    )

    batch = int(entry["batch"])
    for sample in range(n):
        row = metrics[(batch, sample)]
        original_pos = metric_position(row, sample)
        sample_index = metric_sample_index(
            row,
            update_sample_indices=sample_indices_all,
            position=original_pos,
        )
        col = sample % cols
        row_idx = sample // cols
        draw_card(
            canvas,
            x=margin_x + col * (card_w + gap_x),
            y=top + row_idx * (card_h + gap_y),
            width=card_w,
            height=card_h,
            original=originals_all[original_pos],
            reconstruction=rec[sample],
            metric=row,
            sample_index=sample_index,
            ds=ds,
            success_threshold=success_threshold,
            fonts=fonts,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest = read_json(run_dir / "manifest.json")
    ds = args.ds or manifest.get("ds") or manifest.get("dataset")
    if not ds:
        raise ValueError("--ds is required when manifest has no dataset")
    update_file = Path(manifest["update_file"])
    update_data = read_pickle(update_file)
    recon_entries = read_pickle(run_dir / "reconstructions.pkl")
    metrics = load_metric_rows(args.metrics_csv)

    batch_paths: list[Path] = []
    for entry in recon_entries:
        batch = int(entry["batch"])
        originals_all = np.asarray(update_data["x"][batch], dtype=np.float32)
        sample_indices_nested = update_data.get("sample_indices", [])
        sample_indices_all = (
            [int(value) for value in sample_indices_nested[batch]]
            if len(sample_indices_nested) > batch
            else [int(value) for value in entry.get("sample_indices", [])]
        )
        out_path = args.output_dir / f"batch_{batch:03d}_original_reconstruction_metrics.png"
        make_slide(
            entry=entry,
            metrics=metrics,
            originals_all=originals_all,
            sample_indices_all=sample_indices_all,
            ds=str(ds),
            title=args.title,
            out_path=out_path,
            success_threshold=args.success_lpips_threshold,
        )
        batch_paths.append(out_path)

    if len(batch_paths) == 1:
        all_path = args.output_dir / "all_batches_original_reconstruction_metrics.png"
        Image.open(batch_paths[0]).save(all_path)
        print(f"wrote {all_path}")
    else:
        for path in batch_paths:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
