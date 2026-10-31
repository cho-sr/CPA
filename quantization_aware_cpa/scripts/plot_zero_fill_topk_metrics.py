"""Plot verified same80 mean metrics for Top-k10/20/30 zero-fill CPA+FIA."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image

from export_zero_fill_mean_metrics import load

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "exports/zero_fill_topk10_20_30_metric_graph"
PERCENTS = (10, 20, 30)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    means, provenance, identities = [], [], None
    for percent in PERCENTS:
        rows, values, source = load(percent)
        current = [[row[key] for key in ("global_index", "batch_index", "source_index", "dataset_index", "label", "image_path")] for row in rows]
        if identities is not None and current != identities:
            raise RuntimeError("Conditions must have identical80 images in original order")
        identities = current
        means.append({"keep_percent": percent, "n_images": 80, **values})
        provenance.append({"keep_percent": percent, **source})
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": True,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 9))
    fig.subplots_adjust(left=.07, right=.97, bottom=.20, top=.73, wspace=.34)
    ink, muted = "#172A43", "#53657B"
    fig.text(.07, .925, "Zero-fill CPA + FIA", fontsize=30, fontweight="bold", color=ink)
    fig.text(.07, .862, "Mean reconstruction metrics | Same 80 images | Top-k 10%, 20%, 30% retained",
             fontsize=16, color=muted)
    colors = ("#97B8D7", "#5685B1", "#245A8D")
    specifications = (("lpips", "LPIPS", "Lower is better", .75),
                      ("psnr", "PSNR (dB)", "Higher is better", 26.),
                      ("ssim", "SSIM", "Higher is better", .4))
    for axis, (metric, title, direction, ymax) in zip(axes, specifications):
        values = [row[metric] for row in means]
        axis.bar(range(3), values, width=.62, color=colors, zorder=3)
        axis.set_xlim(-.65, 2.65)
        axis.set_ylim(0, ymax)
        axis.set_xticks(range(3), [f"{p}%" for p in PERCENTS], fontsize=18)
        axis.set_xlabel("Top-k retained fraction", fontsize=13, labelpad=16, color=muted)
        axis.tick_params(axis="x", length=0, pad=12)
        axis.tick_params(axis="y", length=0, pad=8, labelsize=13, colors=muted)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.grid(True, color="#E0E7EF", linewidth=1, zorder=0)
        for edge in ("top", "right", "left"):
            axis.spines[edge].set_visible(False)
        axis.spines["bottom"].set_color("#B7C4D2")
        axis.set_title(title, loc="left", fontsize=23, fontweight="bold", color=ink, pad=40)
        axis.text(0, 1.045, direction, transform=axis.transAxes, fontsize=13, color=muted)
        for index, value in enumerate(values):
            axis.text(index, value + ymax * .025, f"{value:.4f}", ha="center", va="bottom",
                      fontsize=17, fontweight="bold", color=ink)
    fig.text(.07, .075, "VGG16 round 0 | 5 batches x 16 images | 10 local SGD steps", fontsize=13, color=muted)
    fig.text(.97, .075, "All 80 images included", ha="right", fontsize=13, color=muted)
    path = OUTPUT / "zero_fill_topk10_20_30_mean_metrics_16x9.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    with Image.open(path) as image:
        assert image.size == (3840, 2160)
    with (OUTPUT / "mean_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(means[0]))
        writer.writeheader()
        writer.writerows(means)
    manifest = {"file": path.name, "size": [3840, 2160], "n_images_per_condition": 80,
                "keep_percents": list(PERCENTS), "mean_metrics": means, "source_provenance": provenance,
                "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUTPUT / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"file": path.name, "mean_metrics": means}), flush=True)


if __name__ == "__main__":
    main()
