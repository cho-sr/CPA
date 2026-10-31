"""CPU-only post-hoc explanation of the completed M7 FedAvg dither4 results."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch


ROOT = Path(__file__).resolve().parents[1] / "results"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    torch.set_num_threads(2)
    output = ROOT / "m7_fedavg_heterogeneity"
    output.mkdir(exist_ok=True)
    quant_path = ROOT / "m7_uia_4bit_B16/per_image_metrics.csv"
    fp32_path = ROOT / "m6_uia_fp32_B16_fia/per_image_metrics.csv"
    quant, fp32 = read_csv(quant_path), read_csv(fp32_path)
    if [int(r["global_index"]) for r in quant] != list(range(80)) or [r["image_path"] for r in quant] != [r["image_path"] for r in fp32]:
        raise RuntimeError("Expected the same80 originals in both precision conditions")
    rows, fingerprints = [], {}
    for path in (quant_path, fp32_path):
        fingerprints[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for batch in range(5):
        folder = ROOT / f"m6_uia_fp32_B16_fia/batch{batch:02d}"
        stored = torch.load(folder / "observation.pt", map_location="cpu", weights_only=False, mmap=True)
        observation = stored["observation"]
        if not isinstance(observation, dict):
            observation = vars(observation)
        inputs = torch.load(folder / "inputs.pt", map_location="cpu", weights_only=False, mmap=True)
        B, Z = observation["B_cumulative"], inputs["Z_evaluation_only"]
        for name, value in (("B_cumulative", B), ("Z_evaluation_only", Z)):
            fingerprints[f"batch{batch:02d}/{name}"] = hashlib.sha256(memoryview(value.contiguous().numpy()).cast("B")).hexdigest()
        b_norm = B.double().norm(dim=0).numpy()
        z_norm = Z.double().norm(dim=1).numpy()
        for index in range(16):
            g = batch * 16 + index
            q, f = quant[g], fp32[g]
            rows.append({"global_index": g, "batch_index": batch,
                         "b_norm": float(b_norm[index]), "z_norm": float(z_norm[index]),
                         "component_frobenius_norm": float(b_norm[index] * z_norm[index]),
                         "initial_loss": float(q["initial_per_sample_loss"]),
                         "final_loss": float(q["final_per_sample_loss"]),
                         "source_cosine_4bit": float(q["source_abs_cosine"]),
                         "source_cosine_fp32": float(f["source_abs_cosine"]),
                         "lpips_4bit": float(q["lpips"]), "lpips_fp32": float(f["lpips"]),
                         "lpips_delta": float(q["lpips"]) - float(f["lpips"])})
        del stored, observation, inputs, B, Z
    with (output / "per_image_analysis.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups = {}
    for name, selected in (("embedding_cos_gt_09", [r for r in rows if r["source_cosine_4bit"] > .9]),
                           ("embedding_cos_le_09", [r for r in rows if r["source_cosine_4bit"] <= .9])):
        groups[name] = {"n": len(selected)}
        for key in ("lpips_4bit", "lpips_fp32", "lpips_delta", "b_norm", "component_frobenius_norm", "initial_loss", "final_loss"):
            values = np.asarray([r[key] for r in selected])
            groups[name][key] = {"mean": float(values.mean()), "median": float(np.median(values))}
    correlations = {}
    for x, y in (("initial_loss", "b_norm"), ("b_norm", "source_cosine_4bit"), ("b_norm", "lpips_4bit"),
                 ("component_frobenius_norm", "lpips_4bit"), ("initial_loss", "lpips_4bit")):
        correlations[f"{x}:{y}"] = {"spearman": float(spearmanr([r[x] for r in rows], [r[y] for r in rows]).statistic),
                                    "per_batch": [float(spearmanr([r[x] for r in rows if r["batch_index"] == b],
                                                                  [r[y] for r in rows if r["batch_index"] == b]).statistic) for b in range(5)]}
    counts = {"cos_gt_09": sum(r["source_cosine_4bit"] > .9 for r in rows),
              "cos_lt_01": sum(r["source_cosine_4bit"] < .1 for r in rows),
              "intermediate": [r["global_index"] for r in rows if .1 <= r["source_cosine_4bit"] <= .9],
              "fp32_high_to_4bit_low": sum(r["source_cosine_fp32"] > .9 and r["source_cosine_4bit"] <= .9 for r in rows)}
    result = {"run_tag": "m7_fedavg_heterogeneity", "source_run": "m7_uia_4bit_B16", "analysis_type": "post_hoc_descriptive_cpu_only",
              "input_fingerprints": fingerprints, "groups": groups, "correlations": correlations, "counts": counts,
              "no_cpa_control_used": False, "image_success_threshold_used": False,
              "limitations": "Outcome-conditioned summaries; not causal ablation. Five batches,80 images. B norms do not capture conditioning or source similarity. No new training/ICA/FIA or GPU work."}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    x = np.asarray([r["b_norm"] for r in rows])
    cosine = np.asarray([r["source_cosine_4bit"] for r in rows])
    axes[0].scatter(x, cosine, c="#2563eb", s=28, alpha=.85)
    axes[0].set(xscale="log", xlabel="Cumulative mixing coefficient norm ||b_i||", ylabel="Recovered embedding |cosine|", title="CPA source separation")
    high = cosine > .9
    for mask, label, color in ((high, "Embedding |cos| > 0.9 (n=53)", "#2563eb"), (~high, "Embedding |cos| <= 0.9 (n=27)", "#dc2626")):
        axes[1].scatter(x[mask], np.asarray([r["lpips_4bit"] for r in rows])[mask], s=28, alpha=.85, label=label, color=color)
    axes[1].set(xscale="log", xlabel="Cumulative mixing coefficient norm ||b_i||", ylabel="Image LPIPS (lower is better)", title="FIA image reconstruction")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("FedAvg n16 T10, same80 dither4 observations | all images included | descriptive analysis")
    fig.savefig(output / "contribution_vs_reconstruction.png", dpi=180)
    fig.savefig(output / "contribution_vs_reconstruction.pdf")
    plt.close(fig)
    good, weak = groups["embedding_cos_gt_09"], groups["embedding_cos_le_09"]
    lines = ["# FedAvg reconstruction heterogeneity: existing M7 all80", "", "CPU-only post-hoc descriptive analysis; no new attacks or training. No no-CPA data or LPIPS success threshold.", "",
             "| Embedding group | n | fp32 LPIPS | dither4 LPIPS | Mean increase | Median ||b_i|| |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, g in (("abs cosine >0.9", good), ("abs cosine <=0.9", weak)):
        lines.append(f"| {name} | {g['n']} | {g['lpips_fp32']['mean']:.6f} | {g['lpips_4bit']['mean']:.6f} | {g['lpips_delta']['mean']:.6f} | {g['b_norm']['median']:.8f} |")
    lines += ["", f"53 embeddings have cosine>0.9,26 have cosine<0.1, and one (g50) has intermediate cosine0.877.26 images cross from high fp32 cosine to low4bit cosine.",
              f"The median coefficient norm is {good['b_norm']['median']/weak['b_norm']['median']:.2f} times larger in the high-cosine group.",
              "Observed update Y approximately equals sum_i b_i z_i^T. Per-row dither amplitude is fixed by the mixture row maximum, so small source contributions face larger relative corruption. Norm/quality association supports this explanation but does not isolate causality, conditioning, source similarity or optimizer effects.",
              "FIA remains imperfect even with accurate features: g48 has source cosine0.999246 but LPIPS0.563253. Image success/failure counts are not inferred from embedding thresholds.", "",
              "These results describe the previous ported M7 experiment. The running M8 direct-upstream experiment is separate.", "",
              "![Contribution and reconstruction](contribution_vs_reconstruction.png)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": counts, "groups": groups, "correlations": correlations}, ensure_ascii=True))


if __name__ == "__main__":
    main()
