#!/usr/bin/env python3
"""Export Naive Sparse CPA reconstructions and metric plots as PNG files."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DS_MEAN = {
    "imagenet": np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1),
    "tiny_imagenet": np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1),
    "cifar10": np.asarray([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1),
    "cifar100": np.asarray([0.5071, 0.4867, 0.4408], dtype=np.float32).reshape(3, 1, 1),
}
DS_STD = {
    "imagenet": np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1),
    "tiny_imagenet": np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1),
    "cifar10": np.asarray([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(3, 1, 1),
    "cifar100": np.asarray([0.2675, 0.2565, 0.2761], dtype=np.float32).reshape(3, 1, 1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs_root",
        type=Path,
        default=Path("sparsified_update_cpa/outputs_v1/author_runs/imagenet/vgg16/attack/cp_direct/nodef"),
    )
    parser.add_argument(
        "--pattern",
        default="01_naive_sparse_cpa_client_topk_global_topk_keep*_n64_steps10_trials3_fia8_uia",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("sparsified_update_cpa/outputs_v1/png_report/n64_steps10_trials3_fia8"),
    )
    parser.add_argument("--ds", default="imagenet")
    parser.add_argument("--max_samples_per_batch", type=int, default=8)
    parser.add_argument("--tile_size", type=int, default=112)
    return parser.parse_args()


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def keep_ratio_from_name(name: str) -> float:
    match = re.search(r"(?:keep|_k)([0-9p]+)(?:_|$)", name)
    if not match:
        raise ValueError(f"Cannot parse keep ratio from {name}")
    return float(match.group(1).replace("p", "."))


def ratio_tag(keep_ratio: float) -> str:
    return f"{keep_ratio:.2f}".replace(".", "p")


def load_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def load_summary(run_dir: Path) -> dict[str, dict[str, str]]:
    with (run_dir / "metric_summary.csv").open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: row for row in csv.DictReader(handle)}


def denormalize_chw(image: np.ndarray, ds: str) -> np.ndarray:
    value = image.astype(np.float32, copy=False)
    if value.ndim != 3:
        raise ValueError(f"Expected CHW image, got {value.shape}")
    if ds in DS_MEAN:
        value = value * DS_STD[ds] + DS_MEAN[ds]
    value = np.clip(value, 0.0, 1.0)
    return value


def chw_to_pil(image: np.ndarray, *, ds: str, already_unit: bool = False, tile_size: int = 112) -> Image.Image:
    value = image.astype(np.float32, copy=False)
    if not already_unit:
        value = denormalize_chw(value, ds)
    else:
        value = np.clip(value, 0.0, 1.0)
    hwc = np.transpose(value, (1, 2, 0))
    pil = Image.fromarray((hwc * 255.0).round().astype(np.uint8), mode="RGB")
    if tile_size and pil.size != (tile_size, tile_size):
        pil = pil.resize((tile_size, tile_size), Image.Resampling.BICUBIC)
    return pil


def text_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[3] - box[1]


def make_pair_grid(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    *,
    ds: str,
    title: str,
    out_path: Path,
    max_samples: int,
    tile_size: int,
) -> None:
    n = min(len(originals), len(reconstructions), max_samples)
    if n < 1:
        return
    font = ImageFont.load_default()
    label_h = 18
    title_h = 28
    gap = 8
    width = n * tile_size + (n + 1) * gap
    height = title_h + label_h + 2 * tile_size + 3 * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 6), title, fill=(0, 0, 0), font=font)
    draw.text((gap, title_h), "original", fill=(0, 0, 0), font=font)
    draw.text((gap, title_h + label_h + tile_size + gap), "reconstruction", fill=(0, 0, 0), font=font)
    for idx in range(n):
        x = gap + idx * (tile_size + gap)
        y_orig = title_h + label_h
        y_rec = y_orig + tile_size + gap + label_h
        orig = chw_to_pil(originals[idx], ds=ds, already_unit=False, tile_size=tile_size)
        rec = chw_to_pil(reconstructions[idx], ds=ds, already_unit=True, tile_size=tile_size)
        canvas.paste(orig, (x, y_orig))
        canvas.paste(rec, (x, y_rec))
        draw.text((x + 2, title_h), f"{idx:02d}", fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def export_run(run_dir: Path, *, output_dir: Path, ds: str, max_samples_per_batch: int, tile_size: int) -> dict[str, Any]:
    keep_ratio = keep_ratio_from_name(run_dir.name)
    tag = ratio_tag(keep_ratio)
    manifest = load_manifest(run_dir)
    update_file = Path(manifest["update_file"])
    update_data = read_pickle(update_file)
    recon = read_pickle(run_dir / "reconstructions.pkl")
    run_png_dir = output_dir / f"keep_{tag}"
    for entry in recon:
        batch = int(entry["batch"])
        rec = np.asarray(entry["rec_fi_attack_order"], dtype=np.float32)
        originals = np.asarray(update_data["x"][batch], dtype=np.float32)[: len(rec)]
        make_pair_grid(
            originals,
            rec,
            ds=ds,
            title=f"Client-level Naive Sparse CPA keep={keep_ratio:.2f} batch={batch}",
            out_path=run_png_dir / f"batch_{batch:03d}_compare.png",
            max_samples=max_samples_per_batch,
            tile_size=tile_size,
        )
        indiv_dir = run_png_dir / f"batch_{batch:03d}_individual"
        indiv_dir.mkdir(parents=True, exist_ok=True)
        n = min(len(originals), len(rec), max_samples_per_batch)
        for idx in range(n):
            chw_to_pil(originals[idx], ds=ds, already_unit=False, tile_size=224).save(
                indiv_dir / f"{idx:03d}_original.png"
            )
            chw_to_pil(rec[idx], ds=ds, already_unit=True, tile_size=224).save(
                indiv_dir / f"{idx:03d}_reconstruction.png"
            )
    summary = load_summary(run_dir)
    return {
        "keep_ratio": keep_ratio,
        "run_dir": str(run_dir),
        "png_dir": str(run_png_dir),
        "lpips": float(summary["lpips"]["mean"]),
        "psnr": float(summary["psnr"]["mean"]),
        "ssim": float(summary["ssim"]["mean"]),
        "embedding_cosine_similarity": float(summary["embedding_cosine_similarity"]["mean"]),
        "reconstruction_success_rate": float(summary["reconstruction_success_rate"]["mean"]),
        "sparse_cosine_similarity_to_fp32": float(manifest["sparse_cosine_similarity_to_fp32"]),
        "sparse_relative_l2": float(manifest["sparse_relative_l2"]),
    }


def make_metric_plot(rows: list[dict[str, Any]], out_path: Path) -> None:
    rows = sorted(rows, key=lambda row: row["keep_ratio"], reverse=True)
    ratios = [row["keep_ratio"] for row in rows]
    labels = [f"{ratio:.0%}" for ratio in ratios]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    plots = [
        ("LPIPS lower is better", "lpips"),
        ("PSNR higher is better", "psnr"),
        ("SSIM higher is better", "ssim"),
        ("Embedding cosine higher is better", "embedding_cosine_similarity"),
    ]
    for ax, (title, key) in zip(axes.flat, plots):
        values = [row[key] for row in rows]
        ax.plot(labels, values, marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("keep ratio")
        ax.grid(True, alpha=0.35)
    fig.suptitle("Client-level Naive Sparse CPA Metrics")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def write_report_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "keep_ratio",
        "sparse_cosine_similarity_to_fp32",
        "sparse_relative_l2",
        "lpips",
        "psnr",
        "ssim",
        "embedding_cosine_similarity",
        "reconstruction_success_rate",
        "png_dir",
        "run_dir",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda value: value["keep_ratio"], reverse=True):
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    run_dirs = sorted(args.runs_root.glob(args.pattern), key=lambda path: keep_ratio_from_name(path.name), reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories matched {args.runs_root / args.pattern}")
    rows = [
        export_run(
            run_dir,
            output_dir=args.output_dir,
            ds=args.ds,
            max_samples_per_batch=args.max_samples_per_batch,
            tile_size=args.tile_size,
        )
        for run_dir in run_dirs
    ]
    make_metric_plot(rows, args.output_dir / "metric_summary.png")
    write_report_csv(rows, args.output_dir / "png_report_summary.csv")
    print(f"Exported PNG report: {args.output_dir}")
    for row in sorted(rows, key=lambda value: value["keep_ratio"], reverse=True):
        print(f"keep {row['keep_ratio']:.2f}: {row['png_dir']}")


if __name__ == "__main__":
    main()
