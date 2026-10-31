"""Offline all-image evaluation of a completed deterministic FC pilot."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from .full_update_evaluation import evaluate_candidate
from .full_update_pipeline import denormalize_images, file_digest, write_json
from .report_full_update_iterations import _make_lpips_metric


def report(base: Path, truth: Path, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if output.exists():
        raise FileExistsError(output)
    # Only completed final artifacts are evaluated; no attack is invoked.
    outputs = {}
    for arm in ("fia", "hybrid"):
        folder = base / arm
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if result["status"] != "completed":
            raise ValueError(f"Incomplete arm: {arm}")
        evaluation = evaluate_candidate(folder / "candidate.pt", truth)
        candidate = torch.load(folder / "candidate.pt", map_location="cpu", weights_only=True)
        outputs[arm] = {"result": result, "evaluation": evaluation, "candidate": candidate,
                        "candidate_sha256": file_digest(folder / "candidate.pt")}
    private = torch.load(truth, map_location="cpu", weights_only=True)
    reference = denormalize_images(private["x"].float()).clamp(0,1)
    metric, lpips_provenance = _make_lpips_metric()
    with torch.no_grad():
        for arm, data in outputs.items():
            pixels = data["candidate"]["pixels"].float()
            values = []
            for pair in data["evaluation"]["per_image"]:
                i, j = pair["candidate_index"], pair["truth_index"]
                metric.reset()
                value = metric((pixels[i:i+1]-.5)*2, (reference[j:j+1]-.5)*2)
                if value.numel() != 1 or not torch.isfinite(value):
                    raise ValueError("Invalid LPIPS")
                pair["lpips"] = float(value)
                values.append(float(value))
            data["evaluation"]["mean_lpips"] = sum(values)/len(values)
            data["evaluation"]["lpips"] = "vgg_same_MSE_Hungarian_pairs"
            intersection = Counter(data["candidate"]["labels"].tolist()) & Counter(private["labels"].tolist())
            data["evaluation"]["label_multiset_overlap_count"] = sum(intersection.values())
    output.mkdir(parents=True)
    fig, axes = plt.subplots(16,3,figsize=(7.5,36))
    for j in range(16):
        axes[j,0].imshow(reference[j].permute(1,2,0))
        for column, arm in enumerate(("fia","hybrid"),1):
            data = outputs[arm]
            pair = next(row for row in data["evaluation"]["per_image"] if row["truth_index"] == j)
            pixels = data["candidate"]["pixels"][pair["candidate_index"]]
            axes[j,column].imshow(pixels.permute(1,2,0))
        for column in range(3):
            axes[j,column].axis("off")
    for column,title in enumerate(("Reference (offline only)","CPA + FIA","CPA + FIA + bin/radius")):
        axes[0,column].set_title(title,fontsize=9)
    fig.tight_layout(pad=.4)
    fig.savefig(output / "all16_comparison.png", dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(4,12,figsize=(22,7.6))
    for j in range(16):
        row, start = j//4, (j%4)*3
        axes[row,start].imshow(reference[j].permute(1,2,0))
        axes[row,start].set_title(f"{j+1}: reference",fontsize=8)
        for offset, arm in enumerate(("fia","hybrid"),1):
            data = outputs[arm]
            pair = next(item for item in data["evaluation"]["per_image"] if item["truth_index"] == j)
            pixels = data["candidate"]["pixels"][pair["candidate_index"]]
            axes[row,start+offset].imshow(pixels.permute(1,2,0))
            axes[row,start+offset].set_title(arm,fontsize=8)
        for offset in range(3):
            axes[row,start+offset].axis("off")
    fig.tight_layout(pad=.4)
    fig.savefig(output / "all16_compact.png",dpi=100)
    plt.close(fig)
    lines = ["Frozen Conv / all FC learning, deterministic uint4, single client, B16/T10.",
             "Same public CPA target, shared 500-step FIA warmup, 50 additional steps per arm.",
             "The added treatment is bin + radius with both image and label gradients; equal iterations, not equal time.", "",
             "| Arm | PSNR dB | SSIM | VGG LPIPS | Label overlap /16 | Optimization seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm, data in outputs.items():
        ev = data["evaluation"]
        lines.append(f"| {arm} | {ev['mean_psnr_db']:.5f} | {ev['mean_ssim']:.6f} | {ev['mean_lpips']:.6f} | {ev['label_multiset_overlap_count']} | {data['result']['optimization_time_sec']:.2f} |")
        del data["candidate"]
    deltas = {key: outputs["hybrid"]["evaluation"][key]-outputs["fia"]["evaluation"][key]
              for key in ("mean_psnr_db","mean_ssim","mean_lpips")}
    lines += ["", "All16 final images are included; matching is fixed independently per arm using pixel-MSE Hungarian.",
              "No private candidate selection, early stop, or success cutoff. A short single-batch pilot is not a general success/failure claim."]
    result = {"arms": outputs, "hybrid_minus_fia": deltas, "lpips_provenance": lpips_provenance,
              "truth_file_sha256": file_digest(truth), "success_threshold": None}
    write_json(output / "comparison.json", result)
    (output / "comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"hybrid_minus_fia":deltas, "output":str(output)}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,required=True)
    parser.add_argument("--truth",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(8)
    report(args.base,args.truth,args.output)


if __name__ == "__main__":
    main()
