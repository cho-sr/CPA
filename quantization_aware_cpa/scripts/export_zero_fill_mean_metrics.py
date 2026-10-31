"""Export verified all80 LPIPS/PSNR/SSIM means for zero-fill Top-k30 and Top-k10."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "exports/zero_fill_topk30_topk10_mean_metrics_16x9"
SIZE = (3840, 2160)
METRICS = (("lpips", "LPIPS", "Lower is better", 6),
           ("psnr", "PSNR", "Higher is better / dB", 4),
           ("ssim", "SSIM", "Higher is better", 6))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size, bold=False):
    return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", size)


def load(percent):
    root = PROJECT / f"results/cpa_fia_topk{percent}_same80_B16"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or summary["n_images"] != 80:
        raise RuntimeError("Complete all80 results are required")
    path = root / "per_image_metrics.csv"
    if sha(path) != summary["output_hashes"][path.name]:
        raise RuntimeError("Metric source changed")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["global_index"]) for row in rows] != list(range(80)):
        raise RuntimeError("Exactly all80 images must be included in original order")
    values = {key: math.fsum(float(row[key]) for row in rows) / len(rows) for key, *_ in METRICS}
    for key, value in values.items():
        if not math.isfinite(value) or not math.isclose(value, summary["metrics"][f"{key}_mean"], rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("Mean differs from the completed report")
    return rows, values, {"source_summary": str((root / "summary.json").relative_to(PROJECT)),
                          "summary_sha256": sha(root / "summary.json"), "csv_sha256": sha(path)}


def render(percent, means):
    canvas = Image.new("RGB", SIZE, "#F7F9FC")
    draw = ImageDraw.Draw(canvas)
    ink, accent, muted = "#172A43", "#2563A6", "#53657B"
    draw.rectangle((0, 0, 3840, 22), fill=accent)
    draw.text((190, 170), "ZERO-FILL  /  CPA + FIA", font=font(62, True), fill=accent)
    draw.text((190, 300), f"Top-k {percent}% retained", font=font(150, True), fill=ink)
    draw.text((190, 510), "Mean reconstruction metrics across all 80 images", font=font(70), fill=muted)
    draw.text((3650, 205), f"{100-percent}% ZERO-FILLED", anchor="ra", font=font(52, True), fill=muted)
    draw.line((190, 680, 3650, 680), fill="#CBD5E1", width=3)
    boxes = []
    for index, (key, title, direction, decimals) in enumerate(METRICS):
        left = 190 + index * 1180
        box = (left, 840, left + 1100, 1600)
        draw.rounded_rectangle(box, radius=34, fill="white", outline="#DDE4ED", width=3)
        draw.rounded_rectangle((left + 55, 904, left + 68, 1004), radius=6, fill=accent)
        draw.text((left + 100, 900), title, font=font(88, True), fill=ink)
        label = f"{means[key]:.{decimals}f}"
        value_font = font(155, True)
        bounds = draw.textbbox((0, 0), label, font=value_font)
        if bounds[2] - bounds[0] > 980:
            raise RuntimeError("Mean value exceeds metric card")
        draw.text((left + 65, 1100), label, font=value_font, fill=accent)
        draw.text((left + 70, 1420), direction, font=font(52), fill=muted)
        boxes.append({"metric": key, "mean": means[key], "display": label, "box": box})
    draw.text((190, 1830), "VGG16 round 0  |  5 batches x 16 images  |  10 local SGD steps", font=font(56), fill=muted)
    draw.text((190, 1955), "Arithmetic mean over all 80 reconstructed images", font=font(44), fill=muted)
    draw.text((3650, 1955), "16:9  /  3840 x 2160", anchor="ra", font=font(44), fill=muted)
    path = OUTPUT / f"zero_fill_topk{percent}_mean_metrics_16x9.png"
    canvas.save(path, dpi=(288, 288))
    with Image.open(path) as image:
        assert image.size == SIZE and image.width * 9 == image.height * 16
    return {"keep_percent": percent, "n_images": 80, "file": path.name,
            "size": SIZE, "sha256": sha(path), "metrics": boxes}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results, previous = [], None
    for percent in (30, 10):
        rows, values, provenance = load(percent)
        identities = [[row[key] for key in ("global_index", "batch_index", "source_index", "dataset_index", "label", "image_path")] for row in rows]
        if previous is not None and identities != previous:
            raise RuntimeError("The percentages must use identical images and order")
        previous = identities
        results.append({**render(percent, values), **provenance})
        print(json.dumps({"keep_percent": percent, "means": values, "file": results[-1]["file"]}), flush=True)
    (OUTPUT / "export_manifest.json").write_text(json.dumps({"outputs": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
