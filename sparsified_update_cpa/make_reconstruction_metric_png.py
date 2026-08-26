#!/usr/bin/env python3
"""Create original/reconstruction PNG panels annotated with saved image metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from common import get_dataloaders, read_json, read_pickle


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
    parser.add_argument("--metrics_csv", type=Path, default=None)
    parser.add_argument("--ds", default=None)
    parser.add_argument("--source", choices=["dataset", "update_file"], default="dataset")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--tile_size", type=int, default=150)
    parser.add_argument("--max_samples_per_batch", type=int, default=8)
    parser.add_argument(
        "--samples_per_row",
        type=int,
        default=None,
        help="Wrap each batch panel after this many samples. Defaults to max_samples_per_batch.",
    )
    parser.add_argument("--success_lpips_threshold", type=float, default=0.5)
    return parser.parse_args()


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    return read_json(path)


def infer_dataset_from_run_dir(run_dir: Path) -> str | None:
    parts = list(run_dir.parts)
    if "author_runs" in parts:
        index = parts.index("author_runs")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


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
    return image.resize((tile_size, tile_size), Image.BILINEAR)


def metric_key(batch: int, sample: int) -> tuple[int, int]:
    return int(batch), int(sample)


def load_metric_rows(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {metric_key(row["batch"], row["sample"]): row for row in csv.DictReader(handle)}


def build_dataset_original_loader(ds: str) -> Callable[[Mapping[str, Any], list[int]], np.ndarray]:
    _, dl_test = get_dataloaders(ds, batch_size=1, shuffle_test=False, num_workers=0)
    dataset = dl_test.dataset

    def load(entry: Mapping[str, Any], positions: list[int]) -> np.ndarray:
        all_indices = [int(value) for value in entry["sample_indices"]]
        sample_indices = [all_indices[position] for position in positions]
        xs = []
        for sample_index in sample_indices:
            item = dataset[sample_index]
            x = item[0] if isinstance(item, (tuple, list)) else item
            xs.append(torch.as_tensor(x, dtype=torch.float32).numpy())
        return np.stack(xs, axis=0)

    return load


def build_update_file_original_loader(manifest: Mapping[str, Any]) -> Callable[[Mapping[str, Any], list[int]], np.ndarray]:
    update_file_value = manifest.get("update_file")
    if not update_file_value:
        raise ValueError("manifest.json does not contain update_file")
    update_data = read_pickle(Path(update_file_value))

    def load(entry: Mapping[str, Any], positions: list[int]) -> np.ndarray:
        return np.asarray(update_data["x"][int(entry["batch"])], dtype=np.float32)[positions]

    return load


def format_metric(row: Mapping[str, str], key: str, digits: int) -> str:
    value = row.get(key)
    if value in (None, ""):
        return "NA"
    return f"{float(value):.{digits}f}"


def draw_sample_tile(
    canvas: Image.Image,
    *,
    x: int,
    y: int,
    original: np.ndarray,
    reconstruction: np.ndarray,
    metric: Mapping[str, str],
    ds: str,
    tile_size: int,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    success_threshold: float,
) -> None:
    draw = ImageDraw.Draw(canvas)
    sample = int(metric.get("sample", 0))
    sample_index = metric.get("sample_index", "")
    lpips_value = float(metric["lpips"]) if metric.get("lpips") not in (None, "") else None
    success = bool(lpips_value is not None and lpips_value <= success_threshold)
    border = (34, 197, 94) if success else (148, 163, 184)
    fill = (240, 253, 244) if success else (248, 250, 252)

    panel_w = tile_size
    header_h = 36
    label_h = 20
    text_h = 60
    panel_h = header_h + label_h + tile_size + label_h + tile_size + text_h
    draw.rounded_rectangle((x, y, x + panel_w, y + panel_h), radius=6, fill=fill, outline=border, width=3)
    draw.text((x + 6, y + 8), f"s{sample:02d} idx {sample_index}", fill=(15, 23, 42), font=small_font)

    y_orig_label = y + header_h
    y_orig = y_orig_label + label_h
    y_rec_label = y_orig + tile_size
    y_rec = y_rec_label + label_h
    draw.text((x + 6, y_orig_label + 2), "original", fill=(51, 65, 85), font=small_font)
    draw.text((x + 6, y_rec_label + 2), "reconstruction", fill=(51, 65, 85), font=small_font)
    canvas.paste(chw_to_pil(original, ds=ds, normalized=True, tile_size=tile_size), (x, y_orig))
    canvas.paste(chw_to_pil(reconstruction, ds=ds, normalized=False, tile_size=tile_size), (x, y_rec))
    if success:
        draw.rectangle((x, y_rec, x + tile_size - 1, y_rec + tile_size - 1), outline=(22, 163, 74), width=5)

    y_text = y_rec + tile_size + 6
    draw.text(
        (x + 6, y_text),
        f"LPIPS {format_metric(metric, 'lpips', 3)}",
        fill=(22, 101, 52) if success else (15, 23, 42),
        font=font,
    )
    draw.text((x + 6, y_text + 22), f"PSNR {format_metric(metric, 'psnr', 2)}", fill=(15, 23, 42), font=small_font)
    draw.text((x + 6, y_text + 40), f"SSIM {format_metric(metric, 'ssim', 3)}", fill=(15, 23, 42), font=small_font)


def make_batch_panel(
    *,
    entry: Mapping[str, Any],
    originals: np.ndarray,
    metrics: Mapping[tuple[int, int], Mapping[str, str]],
    ds: str,
    out_path: Path,
    title: str,
    tile_size: int,
    max_samples: int,
    samples_per_row: int | None,
    success_threshold: float,
) -> None:
    batch = int(entry["batch"])
    rec = np.asarray(entry["rec_fi_attack_order"], dtype=np.float32)
    n = min(len(rec), len(originals), max_samples)
    gap = 14
    title_h = 72
    panel_h = 36 + 20 + tile_size + 20 + tile_size + 60
    columns = max(1, min(int(samples_per_row or max_samples), max(1, n)))
    rows = max(1, (n + columns - 1) // columns)
    width = gap + columns * (tile_size + gap)
    height = title_h + rows * (panel_h + gap)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    font = load_font(16)
    small_font = load_font(13)
    draw.text((gap, 16), title, fill=(15, 23, 42), font=title_font)
    draw.text((gap, 46), "green border: LPIPS <= threshold", fill=(71, 85, 105), font=small_font)
    for sample in range(n):
        metric = metrics.get(metric_key(batch, sample))
        if metric is None:
            raise KeyError(f"Missing metric row for batch={batch}, sample={sample}")
        row = sample // columns
        column = sample % columns
        draw_sample_tile(
            canvas,
            x=gap + column * (tile_size + gap),
            y=title_h + row * (panel_h + gap),
            original=originals[sample],
            reconstruction=rec[sample],
            metric=metric,
            ds=ds,
            tile_size=tile_size,
            font=font,
            small_font=small_font,
            success_threshold=success_threshold,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def stack_panels(paths: list[Path], out_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    gap = 18
    width = max(image.width for image in images)
    height = gap + sum(image.height for image in images) + gap * len(images)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = gap
    for image in images:
        x = (width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    ds = args.ds or manifest.get("ds") or manifest.get("dataset") or infer_dataset_from_run_dir(run_dir)
    if not ds:
        raise ValueError("--ds is required when dataset cannot be inferred from run_dir")
    metrics_csv = args.metrics_csv or (run_dir / "image_metrics_by_sample.csv")
    if not metrics_csv.exists():
        raise FileNotFoundError(metrics_csv)
    output_dir = args.output_dir or (
        Path("sparsified_update_cpa")
        / "outputs_v1"
        / "png_report"
        / "reconstruction_metric_panels"
        / run_dir.name
    )

    rec_entries = read_pickle(run_dir / "reconstructions.pkl")
    metrics = load_metric_rows(metrics_csv)
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
        batch_metric_rows = [metrics[metric_key(batch, sample)] for sample in range(n)]
        positions = [
            int(row["original_sample"])
            if row.get("original_sample") not in (None, "")
            else sample
            for sample, row in enumerate(batch_metric_rows)
        ]
        originals = load_originals(entry, positions)
        out_path = output_dir / f"batch_{batch:03d}_original_reconstruction_metrics.png"
        make_batch_panel(
            entry=entry,
            originals=originals,
            metrics=metrics,
            ds=str(ds),
            out_path=out_path,
            title=f"{run_dir.name} batch {batch}",
            tile_size=args.tile_size,
            max_samples=args.max_samples_per_batch,
            samples_per_row=args.samples_per_row,
            success_threshold=args.success_lpips_threshold,
        )
        batch_paths.append(out_path)

    stack_panels(batch_paths, output_dir / "all_batches_original_reconstruction_metrics.png")
    print(f"wrote {output_dir / 'all_batches_original_reconstruction_metrics.png'}")
    for path in batch_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
