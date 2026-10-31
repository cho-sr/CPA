"""Typeset existing measurements and images; no attack, evaluation, or matching.

Read saved RGB candidates and saved assignments only. The displayed examples
are fixed indices 0 and 1, never selected by reconstruction quality.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from build_results_table import build_table

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
INDICES = [0, 1]
SOURCES = {}
torch.set_num_threads(1)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def record(path):
    path = Path(path).resolve()
    if path not in SOURCES:
        with path.open("rb") as stream:
            SOURCES[path] = hashlib.file_digest(stream, "sha256").hexdigest()
    return path


def read_json(path):
    return json.loads(record(path).read_text(encoding="utf-8"))


def read_csv(path):
    with record(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_tensor(path):
    return torch.load(record(path), map_location="cpu", weights_only=True, mmap=True)


def rgb(tensor):
    return tensor.detach().clamp(0, 1).permute(1, 2, 0).numpy()


def unnormalize(tensor):
    mean = tensor.new_tensor([.485, .456, .406])[None, :, None, None]
    std = tensor.new_tensor([.229, .224, .225])[None, :, None, None]
    return (tensor * std + mean).clamp(0, 1)


def panel(filename, levels, columns, random_label):
    """Render unmodified RGB samples with vector labels and explicit gaps."""
    width = 170 / 25.4  # Preserve the existing asset width; LaTeX sets print width.
    ncols = len(columns)
    left, right = .035, .995
    cell_in = width * (right - left) / ncols
    height = 2 * cell_in + .63
    fig = plt.figure(figsize=(width, height))
    step = (right - left) / ncols
    header_y = 1 - .10 / height
    sub_y = 1 - .26 / height
    fig.text(left + step / 2, sub_y, "Original", ha="center", va="center")
    for j, level in enumerate(levels):
        x = left + (2 + 2*j) * step
        fig.text(x, header_y, level, ha="center", va="center", weight="bold")
        fig.text(x-step/2, sub_y, "Naive", ha="center", va="center")
        fig.text(x+step/2, sub_y, random_label, ha="center", va="center")
    for row, index in enumerate(INDICES):
        top = height - .35 - row * cell_in
        bottom = (top - cell_in) / height
        fig.text(.014, bottom + cell_in / height / 2, str(index),
                 ha="center", va="center")
        for col, images in enumerate(columns):
            ax = fig.add_axes([left + col*step + .002, bottom,
                               step - .004, cell_in / height * .97])
            ax.set_xticks([]); ax.set_yticks([])
            if images is None:
                ax.set_facecolor("#f3f3f3")
                ax.text(.5, .5, "N/A", ha="center", va="center")
            else:
                data = rgb(images[row])
                artist = ax.imshow(data, interpolation="none")
                assert np.array_equal(np.asarray(artist.get_array()), data)
            for spine in ax.spines.values():
                spine.set_color("#b0b0b0"); spine.set_linewidth(.35)
    fig.text(.5, .055 / height, "CIFAR-100: corresponding image panels reserved",
             ha="center", va="bottom", color="#555555")
    fig.savefig(OUT / f"{filename}.pdf")
    fig.savefig(OUT / f"{filename}.png", dpi=230)
    plt.close(fig)


def main():
    qcolumns, assignments = [], {}
    qtruth = None
    for bits in (8, 7, 6, 5, 4):
        if bits == 8:
            folder = "signed_int8_screen3_repair_B16_T10_20260918/evaluation_final_20260918"
        else:
            folder = f"signed_int{bits}_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918"
        report = read_json(ROOT / "quantization_aware_cpa/results_private" / folder / "comparison.json")
        truth = unnormalize(load_tensor(report["private_truth"]["path"])["x"])[INDICES]
        if qtruth is None:
            qtruth = truth
            qcolumns.append(truth)
        else:
            assert torch.equal(qtruth, truth), "Quantization identities changed"
        names = ["center_25000", next(k for k in report["mean_metrics"] if k.startswith("rank1_"))]
        for name in names:
            candidate = load_tensor(report["candidate_paths"][name])["pixels"]
            saved = {int(r["truth_index"]): int(r["candidate_index"])
                     for r in report["image_metrics"][name]["per_image"]}
            order = [saved[i] for i in INDICES]
            qcolumns.append(candidate[order])
            assignments[f"{bits}:{name}"] = order
    panel("quantization_naive_cora",
          [f"{b}-bit" for b in (8,7,6,5,4)], qcolumns, "CoRA")

    result_root = ROOT / "quantization_aware_cpa/results"
    source = load_tensor(result_root / "m6_uia_fp32_B16_fia/batch00/inputs.pt")
    scolumns = [unnormalize(source["x"])[INDICES]]
    sweep = read_csv(result_root / "cpa_fia_topk30_20_10_same80_B16/comparison_metrics.csv")
    sweep = {int(r["keep_percent"]): r for r in sweep}
    random_root = result_root / "cpa_fia_random_topk10_same80_R20_S5k_K3_F25k"
    random_summary = read_csv(random_root / "comparison_metrics.csv")
    zero = next(r for r in random_summary if r["method"].startswith("Zero"))
    for key in ("psnr_mean", "ssim_mean", "lpips_mean"):
        assert float(zero[key]) == float(sweep[10][key])
    identities = None
    for keep in (50, 30, 20, 10):
        folder = result_root / f"cpa_fia_topk{keep}_same80_B16"
        rows = read_csv(folder / "per_image_metrics.csv")
        current = [(r["global_index"],r["dataset_index"],r["image_path"]) for r in rows]
        if identities is None: identities = current
        else: assert identities == current, "Sparsification identities changed"
        for i in INDICES:
            assert int(rows[i]["dataset_index"]) == source["metadata"][i]["dataset_index"]
        scolumns.append(load_tensor(folder / "all80_rec.pt")[INDICES])
        if keep == 10:
            rows = read_csv(random_root / "per_image_metrics.csv")
            assert identities == [(r["global_index"],r["dataset_index"],r["image_path"]) for r in rows]
            scolumns.append(load_tensor(random_root / "all80_rec.pt")[INDICES])
        else:
            scolumns.append(None)
    panel("sparsification_naive_random",
          [f"Keep {k}%" for k in (50,30,20,10)], scolumns, "Random")

    build_table()
    manifest = {
        "purpose": "Document rendering from completed artifacts only",
        "new_attack_steps": 0, "metrics_recomputed": False, "matching_recomputed": False,
        "example_indices": INDICES, "selection": "first two registered identities, independent of quality",
        "quantization_saved_candidate_indices": assignments,
        "sparsification": "all80_rec tensors already aligned by saved evaluation",
        "missing": "CIFAR-100 results; protocol-matched same80/B16/T10/R20 random sparse filling at keep 50/30/20 percent. Older 50/30 percent CoRA results exist under different protocols; see PAPER_SPARSIFICATION_RESULT_AUDIT_20260921.md.",
        "rendering": "stored RGB, standard ImageNet unnormalization for originals, no enhancement or cropping",
        "sources_sha256": {str(p.relative_to(ROOT)): h for p,h in SOURCES.items()},
    }
    for path, digest in SOURCES.items():
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == digest
    (OUT / "comparison_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("Created combined_results.tex and paired quantization/sparsification panels.")
    print("Saved assignments reused; sources unchanged; no metrics or attacks recomputed.")


if __name__ == "__main__":
    main()
