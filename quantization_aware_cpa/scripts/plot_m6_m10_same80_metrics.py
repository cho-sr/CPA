"""Descriptive comparison of saved M6 FP32 and M10 packed4 image metrics."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
TAGS = ("m6_uia_fp32_B16_fia", "m10_cpa_fia_packed4_same80_B16")
OUTPUT = PROJECT / "results" / TAGS[1] / "comparison_m6_fp32"
METRICS = ("lpips", "ssim", "psnr")
COLORS = ("#3979b9", "#e59236")
LABELS = ("32-bit (FP32)", "4-bit")
IDENTITY_FIELDS = ("global_index", "batch_index", "source_index", "dataset_index", "label", "image_path")
NOTE = "Descriptive comparison: CPA implementation and FIA matching order differ between runs."
NOTE2 = "The observed differences do not isolate the effect of compression alone."


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict], metric: str) -> float:
    values = [float(row[metric]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite source metric")
    return math.fsum(values) / len(values)


def read_run(tag: str) -> tuple[dict, list[dict]]:
    root = PROJECT / "results" / tag
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or summary["n_images"] != 80 or summary["oracle_fills"] != 0:
        raise ValueError("Both runs must be complete, unfilled, full80 measurements")
    path = root / "per_image_metrics.csv"
    if sha256(path) != summary["output_hashes"][path.name]:
        raise ValueError(f"Source metric file changed: {tag}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["global_index"]) for row in rows] != list(range(80)):
        raise ValueError("Original IDs 0-79 must be present in order")
    if len({row["image_path"] for row in rows}) != 80:
        raise ValueError("Duplicated source image")
    for index, row in enumerate(rows):
        if (int(row["batch_index"]), int(row["source_index"]), int(row["n"]), int(row["T"])) != (index // 16, index % 16, 16, 10):
            raise ValueError("Original n16/T10 grouping changed")
        if any(str(row.get(flag, "False")).lower() not in ("false", "0") for flag in ("oracle_fill", "missing_source")):
            raise ValueError("The comparison requires all80 measured reconstructions")
    for metric in METRICS:
        if not math.isclose(mean(rows, metric), summary["metrics"][metric + "_mean"], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Recomputed mean disagrees with summary: {tag} {metric}")
    return summary, rows


def draw_metric(axis, metric: str, values: list[float], *, fontsize: int) -> None:
    limits = {"lpips": .70, "ssim": .45, "psnr": 27.5}
    axis.bar([0, 1], values, color=COLORS, width=.57, zorder=3)
    axis.set(xlim=(-.65, 1.65), ylim=(0, limits[metric]))
    axis.set_xticks([0, 1], LABELS, fontsize=fontsize - 2)
    axis.tick_params(axis="y", labelsize=fontsize - 3, length=0, pad=8)
    axis.tick_params(axis="x", length=0, pad=14)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.grid(True, color="#e0e6ee", linewidth=1, zorder=0)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color("#b5c0cf")
    title = metric.upper() + (" (dB)" if metric == "psnr" else "")
    direction = "Lower is better" if metric == "lpips" else "Higher is better"
    axis.set_title(title, loc="left", fontsize=fontsize + 5, fontweight="bold", pad=26, color="#14243a")
    axis.text(1, 1.055, direction, transform=axis.transAxes, ha="right", fontsize=fontsize - 4, color="#526074")
    for index, value in enumerate(values):
        axis.text(index, value + limits[metric] * .024, f"{value:.4f}", ha="center", va="bottom",
                  fontsize=fontsize + 2, fontweight="bold", color="#182a42")
    delta = values[1] - values[0]
    unit = " dB" if metric == "psnr" else ""
    axis.text(.5, -.22, f"4-bit - 32-bit: {delta:+.4f}{unit}", transform=axis.transAxes,
              ha="center", fontsize=fontsize - 1, color="#334155",
              bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f0f4f8", "edgecolor": "none"})


def plot(metrics: list[str], comparison: dict, filename: str) -> dict:
    combined = len(metrics) == 3
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": True,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 9), squeeze=False)
    fig.subplots_adjust(left=.065 if combined else .15, right=.975 if combined else .86,
                        bottom=.255, top=.72, wspace=.32)
    fig.text(.065, .94, "32-bit (FP32) vs 4-bit", fontsize=27,
             fontweight="bold", color="#14243a")
    fig.text(.065, .887, "Mean image metrics | Same 80 images | 5 batches of 16 | 10 local SGD steps",
             fontsize=15, color="#526074")
    handles = [Patch(facecolor=COLORS[0], label=LABELS[0]),
               Patch(facecolor=COLORS[1], label=LABELS[1])]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.065, .846), ncol=2,
               frameon=False, fontsize=13, borderaxespad=0, handlelength=1.5)
    for axis, metric in zip(axes.flat, metrics):
        row = comparison[metric]
        draw_metric(axis, metric, [row["m6_fp32_mean"], row["m10_packed4_mean"]], fontsize=17)
    fig.text(.065, .077, NOTE, fontsize=12, color="#526074")
    fig.text(.065, .045, NOTE2, fontsize=12, color="#526074")
    path = OUTPUT / filename
    fig.savefig(path, dpi=240 if combined else 120)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    with Image.open(path) as image:
        expected = (3840, 2160) if combined else (1920, 1080)
        if image.size != expected:
            raise ValueError("Chart must preserve its registered 16:9 dimensions")
    return {"file": path.name, "size": expected, "sha256": sha256(path),
            "svg": path.with_suffix(".svg").name, "metrics": metrics}


def main() -> None:
    summaries, runs = zip(*(read_run(tag) for tag in TAGS))
    for first, second in zip(*runs):
        if any(first[key] != second[key] for key in IDENTITY_FIELDS):
            raise ValueError("The two runs do not refer to the same image/batch identities")
    comparison = {}
    for metric in METRICS:
        first, second = (mean(rows, metric) for rows in runs)
        comparison[metric] = {"metric": metric, "n_images": 80, "m6_fp32_mean": first,
                              "m10_packed4_mean": second, "m10_minus_m6": second - first,
                              "preferred_direction": "lower" if metric == "lpips" else "higher"}
    OUTPUT.mkdir(exist_ok=True)
    write_csv(OUTPUT / "mean_metric_comparison.csv", list(comparison.values()))
    paired = []
    for first, second in zip(*runs):
        row = {key: first[key] for key in IDENTITY_FIELDS}
        for metric in METRICS:
            row["m6_" + metric] = float(first[metric])
            row["m10_" + metric] = float(second[metric])
            row[metric + "_m10_minus_m6"] = float(second[metric]) - float(first[metric])
        paired.append(row)
    write_csv(OUTPUT / "paired_image_metrics.csv", paired)
    batches = []
    for batch in range(5):
        for metric in METRICS:
            first, second = (mean(rows[batch * 16:(batch + 1) * 16], metric) for rows in runs)
            batches.append({"batch_index": batch, "metric": metric, "n_images": 16,
                            "m6_fp32_mean": first, "m10_packed4_mean": second, "m10_minus_m6": second - first})
    write_csv(OUTPUT / "per_batch_metric_comparison.csv", batches)
    outputs = [plot(list(METRICS), comparison, "m6_fp32_vs_m10_packed4_metrics_16x9.png")]
    outputs.extend(plot([metric], comparison, f"{metric}_comparison_16x9.png") for metric in METRICS)
    provenance = {"run_tags": TAGS, "n_images": 80, "paired_identity_fields": IDENTITY_FIELDS,
                  "means_recomputed_from_per_image_csv": True, "comparison": list(comparison.values()),
                  "interpretation": NOTE + " " + NOTE2,
                  "implementation_difference": "M6: ported Stage1 and permutation before FIA; M10: direct upstream CPA and matching after FIA",
                  "source_hashes": {tag: {name: sha256(PROJECT / "results" / tag / name)
                      for name in ("summary.json", "per_image_metrics.csv")} for tag in TAGS},
                  "outputs": outputs}
    (OUTPUT / "comparison_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"comparison": list(comparison.values()), "figures": [row["file"] for row in outputs]}))


if __name__ == "__main__":
    main()
