"""Read completed same80 results and export matched Top-k10 comparisons."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from qacpa.report_m10_packed4 import file_hash, read_json, read_csv, write_json, write_csv, _rgb

OUTPUT = PROJECT / "exports/zero_vs_random_topk10_same80"
RUNS = ("cpa_fia_topk10_same80_B16", "cpa_fia_random_topk10_same80_R20_S5k_K3_F25k")
LABELS = ("Zero fill", "Random fill")
KEYS = ("lpips", "ssim", "psnr")
IDENTITY = ("global_index", "batch_index", "source_index", "dataset_index", "label", "image_path")
SIZE = (3840, 2160)
INK, MUTED, BLUE, ORANGE = "#192E47", "#52657C", "#3877A7", "#D97C43"


def font(size, bold=False):
    return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", size)


def load():
    sources, tables, recs, provenance = [], [], [], []
    for tag in RUNS:
        folder = PROJECT / "results" / tag
        summary = read_json(folder / "summary.json")
        assert summary["status"] == "complete" and summary["n_images"] == 80
        for name in ("all80_rec.pt", "per_image_metrics.csv"):
            assert file_hash(folder / name) == summary["output_hashes"][name]
        rows = read_csv(folder / "per_image_metrics.csv")
        assert [int(row["global_index"]) for row in rows] == list(range(80))
        for key in KEYS:
            value = np.mean([float(row[key]) for row in rows])
            assert np.isclose(value, summary["metrics"][key + "_mean"], rtol=0, atol=1e-12)
        rec = torch.load(folder / "all80_rec.pt", map_location="cpu", weights_only=True)
        assert rec.shape == (80, 3, 224, 224) and torch.isfinite(rec).all()
        sources.append(summary)
        tables.append(rows)
        recs.append(rec)
        provenance.append({"run_tag": tag, "summary_sha256": file_hash(folder / "summary.json"),
                           "inputs": {name: summary["output_hashes"][name] for name in ("all80_rec.pt", "per_image_metrics.csv")}})
    assert [[row[k] for k in IDENTITY] for row in tables[0]] == [[row[k] for k in IDENTITY] for row in tables[1]]
    for key in ("n", "T", "batch_indices", "source_output", "local_training", "model", "stage1", "fia", "evaluation"):
        assert sources[0]["config"][key] == sources[1]["config"][key], key
    for b, reference in enumerate(sources[1]["identity"]["packets"]):
        baseline_packet = PROJECT / "results" / RUNS[0] / f"batch{b:02d}/observation.bin"
        assert Path(reference["path"]).resolve() == baseline_packet.resolve()
        assert file_hash(baseline_packet) == reference["sha256"]
    return sources, tables, recs, provenance


def metrics_plot(sources, comparison):
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 9))
    fig.subplots_adjust(left=.075, right=.975, top=.71, bottom=.27, wspace=.37)
    fig.text(.075, .92, "Top-k 10% | Zero fill vs Random fill", fontsize=29, weight="bold", color=INK)
    fig.text(.075, .85, "Mean reconstruction metrics | Same 80 images and identical received packets", fontsize=15, color=MUTED)
    specs = (("lpips", "LPIPS", "Lower is better", .75), ("ssim", "SSIM", "Higher is better", .35),
             ("psnr", "PSNR (dB)", "Higher is better", 26.))
    for ax, (key, title, direction, upper) in zip(axes, specs):
        values = [summary["metrics"][key + "_mean"] for summary in sources]
        ax.bar([0, 1], values, color=[BLUE, ORANGE], width=.58, zorder=3)
        ax.set(xlim=(-.6, 1.6), ylim=(0, upper))
        ax.set_xticks([0, 1], LABELS, fontsize=14)
        ax.tick_params(axis="both", length=0, pad=10, colors=MUTED)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.grid(True, color="#E0E7EF", zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#BBC9D8")
        ax.set_title(title, loc="left", fontsize=23, weight="bold", pad=39, color=INK)
        ax.text(0, 1.05, direction, transform=ax.transAxes, fontsize=12, color=MUTED)
        for i, value in enumerate(values):
            ax.text(i, value + upper * .028, f"{value:.6f}" if key != "psnr" else f"{value:.4f}",
                    ha="center", fontsize=15, weight="bold", color=INK)
        diff = next(row["random_minus_zero"] for row in comparison if row["metric"] == key)
        color = "#297A59" if (diff < 0 if key == "lpips" else diff > 0) else "#9E553A"
        ax.text(.5, -.24, f"Random - Zero: {diff:+.6f}" + (" dB" if key == "psnr" else ""),
                transform=ax.transAxes, ha="center", fontsize=12, color=color)
    fig.text(.075, .10, "Zero fill: 1 candidate at FIA 25k   |   Random fill: 20 at 5k, top 3 continued to total 25k", fontsize=12, color=MUTED)
    fig.text(.075, .058, "VGG16 round 0 | 5 batches x 16 | Local SGD 10 steps | Search compute differs; transmitted bytes are identical", fontsize=11, color=MUTED)
    path = OUTPUT / "zero_vs_random_topk10_metrics_16x9.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    assert Image.open(path).size == SIZE
    return {"file": path.name, "size": list(SIZE), "sha256": file_hash(path)}


def tile(canvas, tensor, box, global_id, panel, placements):
    pixels = Image.fromarray(_rgb(tensor)).resize((box[2]-box[0], box[3]-box[1]), Image.Resampling.LANCZOS)
    canvas.paste(pixels, box[:2])
    placements.append({"global_index": global_id, "panel": panel, "box": list(box),
                       "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})


def save_verified(canvas, path, placements):
    canvas.save(path)
    with Image.open(path) as saved:
        assert saved.size == SIZE
        for item in placements:
            assert hashlib.sha256(saved.crop(item["box"]).tobytes()).hexdigest() == item["rgb_sha256"]
    return {"file": path.name, "size": list(SIZE), "sha256": file_hash(path), "placements": placements}


def paired_grid(tables, recs):
    canvas = Image.new("RGB", SIZE, "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((68, 35), "TOP-K 10%  |  Zero fill vs Uniform random fill", font=font(57, True), fill=INK)
    draw.text((68, 118), "Same 80 images in original order  |  Each pair: Zero fill (left) / Random fill (right)", font=font(32), fill=MUTED)
    draw.text((68, 172), "Final candidates selected by observed-update consistency; no selection by ground-truth image metrics", font=font(27), fill=MUTED)
    placements = []
    for i in range(80):
        x, y = 68 + (i % 10) * 372, 253 + (i // 10) * 229
        for j, label in enumerate(LABELS):
            xx = x + j * 182
            draw.text((xx, y), f"{i:02d}  {label}", font=font(20, True), fill=(BLUE, ORANGE)[j])
            tile(canvas, recs[j][i], (xx, y+32, xx+174, y+206), i, label, placements)
    draw.text((68, 2110), "Identical FC1 packets and victim updates | Random fill uses a larger attack search budget", font=font(25), fill=MUTED)
    return save_verified(canvas, OUTPUT / "zero_vs_random_topk10_all80_16x9.png", placements)


def batch_grids(sources, tables, recs):
    source = PROJECT / sources[0]["config"]["source_output"]
    manifest = read_json(PROJECT / "results" / RUNS[1] / "input_manifest.json")
    results = []
    for batch in range(5):
        path = source / f"batch{batch:02d}/inputs.pt"
        assert file_hash(path) == manifest["inputs_sha256"][str(path.relative_to(PROJECT))]
        inputs = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        x = inputs["x"]
        original = (x * x.new_tensor((.229, .224, .225))[None, :, None, None]
                    + x.new_tensor((.485, .456, .406))[None, :, None, None]).clamp(0, 1)
        canvas = Image.new("RGB", SIZE, "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((76, 36), f"TOP-K 10%  |  Original / Zero fill / Random fill  |  Batch {batch}", font=font(54, True), fill=INK)
        draw.text((76, 122), "All 16 images in original order | LPIPS lower is better | Random fill: 20 at 5k, top 3 to 25k", font=font(31), fill=MUTED)
        placements = []
        for s in range(16):
            i = batch*16+s
            assert int(inputs["metadata"][s]["global_index"]) == i
            x0, y0 = 76 + (s % 4)*932, 238 + (s//4)*452
            draw.text((x0, y0), f"Image {i:02d}", font=font(29, True), fill=INK)
            for j, (label, tensor) in enumerate((("Original", original[s]), ("Zero fill", recs[0][i]), ("Random fill", recs[1][i]))):
                xx, yy = x0 + j*302, y0+86
                draw.text((xx, y0+43), label, font=font(28, True), fill=(MUTED, BLUE, ORANGE)[j])
                tile(canvas, tensor, (xx, yy, xx+288, yy+288), i, label, placements)
                if j:
                    draw.text((xx, yy+302), f"LPIPS {float(tables[j-1][i]['lpips']):.4f}", font=font(25), fill=MUTED)
        draw.text((76, 2106), "Same observed packet, images and update | Independent metric evaluation after whole-candidate selection", font=font(25), fill=MUTED)
        results.append(save_verified(canvas, OUTPUT / f"batch{batch:02d}_original_zero_random_16x9.png", placements))
    return results


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    sources, tables, recs, provenance = load()
    paired = []
    for i in range(80):
        paired.append({**{key: tables[0][i][key] for key in IDENTITY},
                       **{f"{key}_{label}": float(tables[j][i][key]) for key in KEYS for j, label in enumerate(("zero", "random"))},
                       **{f"{key}_random_minus_zero": float(tables[1][i][key])-float(tables[0][i][key]) for key in KEYS}})
    comparison = []
    for key in KEYS:
        zero, random = (float(s["metrics"][key+"_mean"]) for s in sources)
        diff = np.array([row[key+"_random_minus_zero"] for row in paired])
        improved = diff < -1e-12 if key == "lpips" else diff > 1e-12
        comparison.append({"metric": key, "zero_fill": zero, "random_fill": random, "random_minus_zero": random-zero,
                           "relative_change_percent": 100*(random-zero)/zero, "random_better_images": int(improved.sum()),
                           "ties": int((np.abs(diff) <= 1e-12).sum()), "n_images": 80})
    batch_rows = [{"batch_index": b, "n_images": 16,
                   **{f"{key}_{label}": float(np.mean([float(row[key]) for row in tables[j][b*16:(b+1)*16]]))
                      for key in KEYS for j, label in enumerate(("zero", "random"))}} for b in range(5)]
    write_csv(OUTPUT / "mean_metric_comparison.csv", comparison)
    write_csv(OUTPUT / "paired_per_image_metrics.csv", paired)
    write_csv(OUTPUT / "per_batch_comparison.csv", batch_rows)
    figures = [metrics_plot(sources, comparison), paired_grid(tables, recs), *batch_grids(sources, tables, recs)]
    zero_runtime = sum(row["runtime_sec"]["invocation"] for row in sources[0]["per_victim_batch"])
    random_runtime = sources[1]["wall_time_sec"]
    report = {"scope": "Top-k10 same80 zero-fill vs uniform-random fill; completed artifacts only",
              "n_images": 80, "means": comparison, "per_batch": batch_rows, "source_provenance": provenance,
              "identical_received_packets": True, "packet_bytes_per_client": sources[1]["communication"]["packet_bytes"]//5,
              "budget_per_batch": {"zero": {"cpa_iterations": 25000, "fia_iterations": 25000},
                                   "random": sources[1]["budget_per_batch"]},
              "runtime": {"zero_sum_batch_invocation_sec": zero_runtime, "random_elapsed_sec": random_runtime,
                          "note": "Different scopes: zero sums batch invocations; random includes report and any resume downtime"},
              "figures": figures, "no_new_training_or_attack": True, "script_sha256": file_hash(Path(__file__))}
    write_json(OUTPUT / "comparison_manifest.json", report)
    text = ["# Top-k10: Zero fill vs uniform random fill", "", "Same80 image identities/order, original n16x5/T10 updates and identical FC1 packets verified by hashes. Report reads completed results only.", "",
            "| Metric | Zero fill | Random fill | Random minus zero | Random better images |", "|---|---:|---:|---:|---:|"]
    text += [f"| {r['metric']} | {r['zero_fill']:.6f} | {r['random_fill']:.6f} | {r['random_minus_zero']:+.6f} | {r['random_better_images']}/80 |" for r in comparison]
    text += ["", "LPIPS is slightly lower; SSIM and PSNR are slightly lower. These descriptive means show no consistent improvement across metrics. All80 are included; counts are paired image comparisons, not independent experimental replicates or a significance test.", "",
             "Communication is identical:53,949,264 bytes per attacked FC1 packet,86.874994% below a same-shape dense FP32 packet. Random values are generated on the attacker side.", "",
             "Search compute differs by user request. Per batch zero fill uses one CPA25k/FIA25k candidate; random fill totals20 CPA25k,20 FIA5k plus3 additional FIA20k (500k CPA iterations and160k FIA iterations). Finalists and the final whole-batch candidate were selected by observed-update consistency, not image metrics. No additional control attacks were run.", "",
             "Figures: same80 paired zero/random images, five original/zero/random batch sheets, and a mean-metric graph. Every exported image tile was checked against the saved tensor rendered with the same RGB conversion and isotropic Lanczos resizing. There is no contrast enhancement or color adjustment.", ""]
    (OUTPUT / "report.md").write_text("\n".join(text), encoding="utf-8")
    with zipfile.ZipFile(OUTPUT / "zero_vs_random_topk10_comparison.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT.iterdir()):
            if path.suffix in (".png", ".csv", ".json", ".md"):
                archive.write(path, path.name)
    print(json.dumps({"output": str(OUTPUT), "metrics": comparison, "zero_seconds": zero_runtime,
                      "random_seconds": random_runtime, "figures": [item["file"] for item in figures]}, indent=2))


if __name__ == "__main__":
    main()
