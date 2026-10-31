"""Final offline evaluation for the bounded signed-cell image screen.

The public runner must finish every preregistered arm before this module opens
the bundled private evaluation truth.  The truth file may materialize labels
and oracle features when loaded, but this evaluator consumes only normalized
images; labels and true updates never enter matching, selection, or reporting.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from .full_update_evaluation import evaluate_candidate
from .report_full_update_iterations import _make_lpips_metric
from .signed_baseline_observation import CHECKPOINT_SHA, FC_SHAPES, digest
from .signed_e2e_runner import (
    IMAGE_COUNT, MAX_SECONDS, PLAN_VERSION, REPO_ROOT, TASK_ID, file_sha256,
    hash_files, hash_public_inputs, protected_paths, write_json,
)


PRIVATE_BASELINE_ROOT = REPO_ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10"
EXPECTED_PRIVATE_TRUTH = PRIVATE_BASELINE_ROOT / "observation/evaluation_truth.pt"
EXPECTED_PRIVATE_OUTPUT = REPO_ROOT / "quantization_aware_cpa/results_private/signed_e2e_cell_screen_B16_T10_20260916"


class EvaluationError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"expected JSON object: {path}")
    return value


def _validate_public_completion(public_output: Path, public_observation: Path,
                                naive: Path, checkpoint: Path) -> tuple[dict, dict, dict]:
    if not public_output.is_dir():
        raise EvaluationError("public optimization output is missing")
    registration = _read_json(public_output / "registration.json")
    summary = _read_json(public_output / "summary.json")
    integrity = _read_json(public_output / "integrity.json")
    if (registration.get("task_id") != TASK_ID or registration.get("plan_version") != PLAN_VERSION
            or registration.get("status") != "REGISTERED"):
        raise EvaluationError("public registration does not identify this task")
    if summary.get("status") != "COMPLETED":
        raise EvaluationError("public runner did not complete all registered stages")
    if integrity.get("protected_sources_unchanged") is not True or integrity.get("inputs_unchanged") is not True:
        raise EvaluationError("public integrity record does not prove unchanged protected inputs")
    actual_inputs = hash_public_inputs(public_observation, naive, checkpoint)
    if actual_inputs != registration.get("input_hashes"):
        raise EvaluationError("public input hashes changed before private evaluation")
    arms_plan = summary.get("arms_plan")
    if not isinstance(arms_plan, list) or not arms_plan or len(arms_plan) > 5:
        raise EvaluationError("public arm plan is missing or exceeds the registered maximum")
    for item in arms_plan:
        name = item.get("arm")
        folder = public_output / name
        if (not isinstance(name, str) or not name or not folder.is_dir()
                or not (folder / "candidate.pt").is_file() or not (folder / "result.json").is_file()
                or not (folder / "optimization_log.json").is_file()):
            raise EvaluationError(f"public arm is incomplete: {name}")
        result = _read_json(folder / "result.json")
        if result.get("status") != "COMPLETED" or result.get("iterations") != 100:
            raise EvaluationError(f"public arm is not a completed fixed 100-step arm: {name}")
        candidate = torch.load(folder / "candidate.pt", map_location="cpu", weights_only=True)
        if (not isinstance(candidate, dict) or set(candidate) != {"pixels"}
                or not isinstance(candidate["pixels"], torch.Tensor)
                or candidate["pixels"].dtype != torch.float32
                or tuple(candidate["pixels"].shape) != (IMAGE_COUNT, 3, 224, 224)
                or not bool(torch.isfinite(candidate["pixels"]).all())
                or bool((candidate["pixels"] < 0).any()) or bool((candidate["pixels"] > 1).any())):
            raise EvaluationError(f"public arm candidate schema/value failure: {name}")
        if result.get("candidate_sha256") != file_sha256(folder / "candidate.pt"):
            raise EvaluationError(f"public arm candidate hash failure: {name}")
    return registration, summary, integrity


def _load_truth_after_validation(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "x" not in payload:
        raise EvaluationError("private evaluation truth lacks x")
    images = payload["x"]
    if (not isinstance(images, torch.Tensor) or images.dtype != torch.float32
            or tuple(images.shape) != (IMAGE_COUNT, 3, 224, 224)
            or not bool(torch.isfinite(images).all())):
        raise EvaluationError("private truth x has the wrong FP32 shape or finite status")
    # Deliberately do not inspect or index labels/z. torch.load may materialize
    # bundled fields, so the fact is disclosed in the output metadata.
    return images.contiguous(), {"private_loaded": True, "labels_consumed": False,
                                 "bundled_fields_may_materialize": sorted(payload.keys()),
                                 "consumed_field": "x only"}


def _lpips_metrics(candidates: dict[str, torch.Tensor], truth_x: torch.Tensor,
                   image_reports: dict[str, dict]) -> tuple[dict[str, dict], dict]:
    metric, provenance = _make_lpips_metric()
    mean = truth_x.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = truth_x.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    originals = (truth_x * std + mean).clamp(0, 1)
    result = {}
    for name, report in image_reports.items():
        pixels = candidates[name]
        values = []
        for pair in report["per_image"]:
            i, j = pair["candidate_index"], pair["truth_index"]
            metric.reset()
            score = metric((pixels[i:i + 1] - .5) * 2, (originals[j:j + 1] - .5) * 2)
            if score.numel() != 1 or not bool(torch.isfinite(score).all()):
                raise EvaluationError(f"nonfinite LPIPS score for {name} image {i}")
            pair["lpips"] = float(score.detach())
            values.append(pair["lpips"])
        report["mean_lpips"] = sum(values) / len(values)
        report["lpips_provenance"] = provenance
        result[name] = {"mean_lpips": report["mean_lpips"], "per_image": values}
    return result, provenance


def _write_figure(output: Path, candidates: dict[str, torch.Tensor], truth_x: torch.Tensor) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mean = truth_x.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = truth_x.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    originals = (truth_x * std + mean).clamp(0, 1)
    names = ["originals"] + list(candidates)
    columns = [originals] + [candidates[name] for name in candidates]
    figure, axes = plt.subplots(IMAGE_COUNT, len(names), figsize=(3 * len(names), 2.4 * IMAGE_COUNT))
    axes = axes.reshape(IMAGE_COUNT, len(names))
    try:
        for row in range(IMAGE_COUNT):
            for col, values in enumerate(columns):
                axes[row, col].imshow(values[row].permute(1, 2, 0).clamp(0, 1))
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(names[col])
        figure.tight_layout()
        path = output / "all16_comparison.png"
        figure.savefig(path, dpi=110)
    finally:
        plt.close(figure)
    return path.name


def _directional_summary(image_reports: dict[str, dict]) -> dict[str, Any]:
    metrics = ("mean_mse", "mean_psnr_db", "mean_ssim", "mean_lpips")
    means = {name: {key: (None if key == "mean_psnr_db" and any(
        item["psnr_db"] is None for item in report["per_image"]) else
        sum(item[key.removeprefix("mean_")] for item in report["per_image"]) / IMAGE_COUNT)
                     for key in metrics} for name, report in image_reports.items()}
    return means


def evaluate(*, public_output: Path, public_observation: Path, naive: Path,
             checkpoint: Path, private_truth: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing preexisting private evaluation output: {output}")
    if not private_truth.resolve().is_relative_to(PRIVATE_BASELINE_ROOT.resolve()):
        raise EvaluationError("private truth must remain in the existing private observation tree")
    registration, summary, integrity = _validate_public_completion(
        public_output, public_observation, naive, checkpoint)
    protected_before = hash_files(protected_paths())
    output.mkdir(parents=True, exist_ok=False)
    # This is the phase boundary: all public candidates and their hashes were
    # validated before this first private tensor deserialization.
    truth_x, private_meta = _load_truth_after_validation(private_truth)
    candidates: dict[str, Tensor] = {}
    image_reports: dict[str, dict] = {}
    for item in summary["arms_plan"]:
        name = item["arm"]
        candidates[name] = torch.load(public_output / name / "candidate.pt",
                                      map_location="cpu", weights_only=True)["pixels"].float()
        image_reports[name] = evaluate_candidate(public_output / name / "candidate.pt",
                                                  private_truth, include_labels=False)
    x0_path = output / "x0_candidate.pt"
    x0 = torch.load(naive / "candidate.pt", map_location="cpu", weights_only=True)["pixels"].float()
    torch.save({"pixels": x0}, x0_path)
    candidates["x0"] = x0
    image_reports["x0"] = evaluate_candidate(x0_path, private_truth, include_labels=False)
    lpips_result, lpips_provenance = _lpips_metrics(candidates, truth_x, image_reports)
    means = _directional_summary(image_reports)
    figure = _write_figure(output, {name: candidates[name] for name in summary["arms_plan"]
                                    for name in [name["arm"]]}, truth_x)
    protected_after = hash_files(protected_paths())
    input_after = hash_public_inputs(public_observation, naive, checkpoint)
    if protected_before != protected_after or input_after != registration.get("input_hashes"):
        raise EvaluationError("protected source or public input changed during offline evaluation")
    comparison = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "COMPLETED",
        "evaluation_status": "COMPLETED", "public_output": str(public_output),
        "private_truth_path": str(private_truth), "private_output": str(output),
        "labels_consumed": False, "private_truth_metadata": private_meta,
        "matching": "Hungarian_pixel_MSE_once_for_each_final_candidate_and_x0; reused for SSIM/LPIPS",
        "all_images_included": True, "image_metrics": image_reports,
        "mean_metrics": means, "lpips": lpips_result, "lpips_provenance": lpips_provenance,
        "figure_artifacts": [figure], "public_summary": summary,
        "public_integrity": integrity,
        "source_integrity": {"before": protected_before, "after": protected_after,
                              "unchanged": protected_before == protected_after},
        "input_hashes": {"before": registration.get("input_hashes"), "after": input_after,
                         "unchanged": input_after == registration.get("input_hashes")},
        "directional_interpretation": {
            "rule": "paired descriptive contrasts only; no threshold, selection, or generalization",
            "arms_fixed_before_private_access": True,
            "x0_is_saved_naive_pixels": True,
        },
        "wall_seconds": time.perf_counter() - started,
        "max_global_real_phase_seconds": MAX_SECONDS,
    }
    write_json(output / "comparison.json", comparison)
    write_json(output / "results.json", comparison)
    lines = [f"# {TASK_ID}", "", "Status: **COMPLETED**.", "",
             "All scheduled public arms were finalized before private evaluation. "
             "Labels and true updates were not consumed.", "",
             "| Arm | MSE | PSNR dB | SSIM | LPIPS |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, values in means.items():
        lines.append(f"| {name} | {values['mean_mse']:.8g} | "
                     f"{values['mean_psnr_db'] if values['mean_psnr_db'] is not None else 'null'} | "
                     f"{values['mean_ssim']:.8g} | {values['mean_lpips']:.8g} |")
    lines += ["", "Exact code/cell feasibility and public replay profiles are in `comparison.json` "
              "and the public runner output. These image metrics are descriptive for this batch; "
              "no reconstruction-success threshold or generalization claim is applied."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--public-observation", type=Path, required=True)
    parser.add_argument("--naive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--private-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        result = evaluate(**vars(args))
    except Exception as error:
        print(f"{TASK_ID}: {type(error).__name__}: {error}", flush=True)
        return 1
    print(json.dumps({"task_id": TASK_ID, "status": result["status"],
                      "wall_seconds": result["wall_seconds"]}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

