"""Offline all-image comparison for the H1 FIA-removal ablation.

The public runner freezes x0, the archived B_H1_high control, and the new
C_H1_high_noFIA candidate before this module opens the private truth.  The
private payload is used for images only; bundled labels and true updates are
never inspected or consumed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import torch

from .full_update_evaluation import evaluate_candidate
from .report_full_update_iterations import _make_lpips_metric
from .signed_baseline_observation import CHECKPOINT_SHA
from .signed_e2e_nofia_runner import (
    CHECKPOINT, CONTROL_ARM, H1_LABELS, NEW_ARM, NAIVE_ROOT,
    PREVIOUS_PUBLIC_ROOT, PREVIOUS_PRIVATE_ROOT, PUBLIC_ROOT, REPO_ROOT,
    TASK_ID, LAMBDA_CELL, LAMBDA_FIA, PLAN_VERSION, file_sha256,
    hash_files, hash_public_inputs, protected_paths, write_json,
)


EXPECTED_PRIVATE_TRUTH = REPO_ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"
EXPECTED_PUBLIC_OUTPUT = REPO_ROOT / "quantization_aware_cpa/results/signed_e2e_h1_nofia_B16_T10_20260916"
EXPECTED_PRIVATE_OUTPUT = REPO_ROOT / "quantization_aware_cpa/results_private/signed_e2e_h1_nofia_B16_T10_20260916"


class EvaluationError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"expected JSON object: {path}")
    return value


def _validate_pixels(value: Any, name: str) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or tuple(value.shape) != (16, 3, 224, 224)
            or not bool(torch.isfinite(value).all())
            or bool((value < 0).any()) or bool((value > 1).any())):
        raise EvaluationError(f"{name} must be finite FP32 pixels with shape [16,3,224,224]")
    return value.contiguous()


def _candidate(path: Path, expected_hash: str | None = None) -> torch.Tensor:
    if not path.is_file() or path.is_symlink():
        raise EvaluationError(f"candidate is missing: {path}")
    if expected_hash is not None and file_sha256(path) != expected_hash:
        raise EvaluationError(f"candidate hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"pixels"}:
        raise EvaluationError(f"candidate schema mismatch: {path}")
    return _validate_pixels(payload["pixels"], str(path))


def _validate_public(public_output: Path, public_observation: Path, naive: Path,
                     checkpoint: Path) -> tuple[dict, dict, dict, dict[str, torch.Tensor]]:
    if public_output.resolve() != EXPECTED_PUBLIC_OUTPUT.resolve():
        raise EvaluationError("unexpected public output root")
    registration = _read_json(public_output / "registration.json")
    summary = _read_json(public_output / "summary.json")
    integrity = _read_json(public_output / "integrity.json")
    preflight = _read_json(public_output / "full_size_numeric_check.json")
    if registration.get("task_id") != TASK_ID or registration.get("plan_version") != PLAN_VERSION:
        raise EvaluationError("public registration task/version mismatch")
    if registration.get("status") != "REGISTERED" or registration.get("new_arm") != NEW_ARM:
        raise EvaluationError("public registration is not the signed noFIA plan")
    if summary.get("task_id") != TASK_ID or summary.get("status") != "COMPLETED":
        raise EvaluationError("public noFIA runner did not complete")
    if summary.get("control", {}).get("arm") != CONTROL_ARM or summary["control"].get("retrained") is not False:
        raise EvaluationError("archived control reuse is not authenticated")
    if summary.get("new_arm", {}).get("arm") != NEW_ARM or summary["new_arm"].get("iterations") != 100:
        raise EvaluationError("new arm is not exactly one completed 100-step candidate")
    if registration.get("h1_labels") != list(H1_LABELS):
        raise EvaluationError("H1 labels changed before private evaluation")
    objective = registration.get("objective", {})
    if (objective.get("lambda_fia") != LAMBDA_FIA or objective.get("lambda_cell") != LAMBDA_CELL
            or objective.get("lambda_tv") != 0.1):
        raise EvaluationError("objective coefficients do not match the one-factor ablation")
    if integrity.get("protected_sources_unchanged") is not True or integrity.get("inputs_unchanged") is not True:
        raise EvaluationError("public integrity record does not prove unchanged inputs")
    if integrity.get("shared_sources_unchanged") is not True or integrity.get("previous_control_files_unchanged") is not True:
        raise EvaluationError("shared or archived-control integrity check failed")
    if preflight.get("terminal_deltas_torch_equal") is not True or preflight.get("feature_grad_enabled_equals_no_grad") is not True:
        raise EvaluationError("public numerical path check failed")
    actual_inputs = hash_public_inputs(public_observation, naive, checkpoint)
    if actual_inputs != registration.get("input_hashes"):
        raise EvaluationError("public inputs changed before private evaluation")
    control_result = _read_json(PREVIOUS_PUBLIC_ROOT / CONTROL_ARM / "result.json")
    if control_result.get("candidate_sha256") != registration.get("previous_authentication", {}).get("control_candidate_sha256"):
        raise EvaluationError("archived control result hash does not match registered provenance")
    candidates = {
        "x0": _candidate(public_output / "x0_candidate.pt"),
        CONTROL_ARM: _candidate(PREVIOUS_PUBLIC_ROOT / CONTROL_ARM / "candidate.pt",
                                control_result["candidate_sha256"]),
        NEW_ARM: _candidate(public_output / NEW_ARM / "candidate.pt",
                            summary["new_arm"]["candidate_sha256"]),
    }
    for name in (NEW_ARM,):
        result = _read_json(public_output / name / "result.json")
        if result.get("lambda_fia") != 0.0 or result.get("lambda_cell") != LAMBDA_CELL:
            raise EvaluationError(f"new candidate objective provenance mismatch: {name}")
    return registration, summary, integrity, candidates


def _load_private_images(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    if path.resolve() != EXPECTED_PRIVATE_TRUTH.resolve():
        raise EvaluationError("private truth path is outside the registered observation")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "x" not in payload:
        raise EvaluationError("private truth does not contain x")
    images = _validate_truth_images(payload["x"])
    return images, {"private_loaded": True, "labels_consumed": False,
                    "bundled_fields_may_materialize": sorted(payload.keys()),
                    "consumed_field": "x only"}


def _validate_truth_images(value: Any) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or tuple(value.shape) != (16, 3, 224, 224)
            or not bool(torch.isfinite(value).all())):
        raise EvaluationError("private truth x must be finite FP32 [16,3,224,224]")
    return value.contiguous()


def _lpips(candidates: dict[str, torch.Tensor], reports: dict[str, dict], truth: torch.Tensor) -> dict[str, Any]:
    metric, provenance = _make_lpips_metric()
    mean = truth.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = truth.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    originals = (truth * std + mean).clamp(0, 1)
    result: dict[str, Any] = {}
    for name, report in reports.items():
        values = []
        for pair in report["per_image"]:
            i, j = pair["candidate_index"], pair["truth_index"]
            metric.reset()
            score = metric((candidates[name][i:i + 1] - .5) * 2,
                           (originals[j:j + 1] - .5) * 2)
            if score.numel() != 1 or not bool(torch.isfinite(score).all()):
                raise EvaluationError(f"nonfinite LPIPS score for {name}/{i}")
            pair["lpips"] = float(score)
            values.append(pair["lpips"])
        report["mean_lpips"] = sum(values) / len(values)
        report["lpips_provenance"] = provenance
        result[name] = {"mean_lpips": report["mean_lpips"], "per_image": values}
    return {"metrics": result, "provenance": provenance}


def _means(reports: dict[str, dict]) -> dict[str, dict[str, float | None]]:
    result = {}
    for name, report in reports.items():
        psnr = [item["psnr_db"] for item in report["per_image"]]
        result[name] = {
            "mean_mse": sum(item["mse"] for item in report["per_image"]) / 16,
            "mean_psnr_db": None if any(value is None for value in psnr) else sum(psnr) / 16,
            "mean_ssim": sum(item["ssim"] for item in report["per_image"]) / 16,
            "mean_lpips": report["mean_lpips"],
        }
    return result


def _prior_control_means() -> dict[str, float]:
    machine_readable = PREVIOUS_PRIVATE_ROOT / "comparison.json"
    if machine_readable.is_file():
        try:
            value = _read_json(machine_readable)
            means = value.get("mean_metrics", {}).get(CONTROL_ARM)
            if isinstance(means, dict) and all(key in means for key in
                                               ("mean_mse", "mean_psnr_db", "mean_ssim", "mean_lpips")):
                return {key: float(means[key]) for key in
                        ("mean_mse", "mean_psnr_db", "mean_ssim", "mean_lpips")}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    report = (PREVIOUS_PRIVATE_ROOT / "report.md").read_text(encoding="utf-8")
    for line in report.splitlines():
        if line.startswith("| B_H1_high |"):
            fields = [field.strip() for field in line.strip().strip("|").split("|")]
            if len(fields) != 5:
                break
            return {"mean_mse": float(fields[1]), "mean_psnr_db": float(fields[2]),
                    "mean_ssim": float(fields[3]), "mean_lpips": float(fields[4])}
    raise EvaluationError("prior private control report row is missing")


def align_candidates_to_truth(candidate: torch.Tensor, report: dict[str, Any],
                              originals: torch.Tensor) -> torch.Tensor:
    """Return candidate rows in the truth order recorded by Hungarian matching."""
    pairs = report.get("per_image")
    if not isinstance(pairs, list) or len(pairs) != len(originals):
        raise EvaluationError("metric correspondence is not a complete image permutation")
    candidate_indices = [item.get("candidate_index") for item in pairs]
    truth_indices = [item.get("truth_index") for item in pairs]
    expected = list(range(len(originals)))
    if sorted(candidate_indices) != expected or sorted(truth_indices) != expected:
        raise EvaluationError("metric correspondence is not a one-to-one permutation")
    aligned = torch.empty_like(candidate)
    for item in pairs:
        aligned[item["truth_index"]] = candidate[item["candidate_index"]]
    return aligned


def _write_figure(output: Path, candidates: dict[str, torch.Tensor], truth: torch.Tensor,
                  reports: dict[str, dict[str, Any]]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean = truth.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = truth.new_tensor((.229, .224, .406)).view(1, 3, 1, 1)
    # Correct the third channel explicitly; keeping this local avoids using
    # any evaluator state from the historical five-arm task.
    std = truth.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    originals = (truth * std + mean).clamp(0, 1)
    names = ["originals", *candidates.keys()]
    columns = [originals, *[align_candidates_to_truth(candidates[name], reports[name], originals)
                             for name in candidates]]
    figure, axes = plt.subplots(16, len(names), figsize=(3 * len(names), 2.4 * 16))
    axes = axes.reshape(16, len(names))
    try:
        for row in range(16):
            for col, values in enumerate(columns):
                axes[row, col].imshow(values[row].permute(1, 2, 0).clamp(0, 1))
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(names[col])
        figure.tight_layout()
        figure.savefig(output / "all16_comparison.png", dpi=110)
    finally:
        plt.close(figure)
    return "all16_comparison.png"


def evaluate(*, public_output: Path = EXPECTED_PUBLIC_OUTPUT,
             public_observation: Path = PUBLIC_ROOT, naive: Path = NAIVE_ROOT,
             checkpoint: Path = CHECKPOINT, private_truth: Path = EXPECTED_PRIVATE_TRUTH,
             output: Path = EXPECTED_PRIVATE_OUTPUT) -> dict[str, Any]:
    started = time.perf_counter()
    public_output = public_output.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing preexisting noFIA private output: {output}")
    if file_sha256(checkpoint) != CHECKPOINT_SHA:
        raise EvaluationError("checkpoint hash mismatch")
    registration, summary, integrity, candidates = _validate_public(
        public_output, public_observation.resolve(), naive.resolve(), checkpoint.resolve())
    protected_before = hash_files(protected_paths())
    output.mkdir(parents=True, exist_ok=False)
    # This is the private phase boundary.  All public candidates, hashes and
    # endpoint checks were completed before this first private deserialization.
    truth, private_meta = _load_private_images(private_truth.resolve())
    reports: dict[str, dict] = {}
    for name in ("x0", CONTROL_ARM, NEW_ARM):
        candidate_path = output / f"{name}_candidate.pt"
        torch.save({"pixels": candidates[name]}, candidate_path)
        reports[name] = evaluate_candidate(candidate_path, private_truth, include_labels=False)
    lpips_info = _lpips(candidates, reports, truth)
    means = _means(reports)
    prior = _prior_control_means()
    control_diff = {key: means[CONTROL_ARM][key] - prior[key] for key in prior}
    control_tolerance = {"mean_mse": 2e-8, "mean_psnr_db": 2e-8,
                         "mean_ssim": 2e-8, "mean_lpips": 2e-8}
    control_matches = all(abs(control_diff[key]) <= control_tolerance[key] for key in control_diff)
    figure = _write_figure(output, {"x0": candidates["x0"], CONTROL_ARM: candidates[CONTROL_ARM],
                                    NEW_ARM: candidates[NEW_ARM]}, truth, reports)
    protected_after = hash_files(protected_paths())
    input_after = hash_public_inputs(public_observation, naive, checkpoint)
    if protected_before != protected_after or input_after != registration.get("input_hashes"):
        raise EvaluationError("protected source or public input changed during private evaluation")
    comparison = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "COMPLETED",
        "evaluation_status": "COMPLETED", "public_output": public_output.as_posix(),
        "private_truth_path": private_truth.as_posix(), "private_output": output.as_posix(),
        "labels_consumed": False, "private_truth_metadata": private_meta,
        "matching": "Hungarian_pixel_MSE_once_per_candidate_reused_for_SSIM_LPIPS",
        "all_images_included": True, "candidates": {name: {"tensor_sha256": _tensor_hash(value)}
                                                        for name, value in candidates.items()},
        "image_metrics": reports, "mean_metrics": means,
        "lpips": lpips_info, "figure_artifacts": [figure],
        "fixed_ablation": {"control": CONTROL_ARM, "new_arm": NEW_ARM,
                           "lambda_fia_control": 1.0, "lambda_fia_new": 0.0,
                           "lambda_cell": LAMBDA_CELL, "h1_labels": list(H1_LABELS)},
        "control_metric_reproduction": {"matches_prior_report_with_tolerance": control_matches,
                                         "prior_means": prior, "current_means": means[CONTROL_ARM],
                                         "difference": control_diff, "tolerance": control_tolerance},
        "public_summary": summary, "public_integrity": integrity,
        "source_integrity": {"before": protected_before, "after": protected_after,
                              "unchanged": protected_before == protected_after},
        "input_hashes": {"before": registration.get("input_hashes"), "after": input_after,
                         "unchanged": input_after == registration.get("input_hashes")},
        "directional_interpretation": {
            "rule": "fixed-H1 paired descriptive FIA-weight contrast; no threshold, selection, or generalization",
            "private_metrics_not_used_for_attack_or_selection": True,
            "labels_consumed": False,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output / "comparison.json", comparison)
    write_json(output / "results.json", comparison)
    lines = [f"# {TASK_ID}", "", "Status: **COMPLETED**.", "",
             "The archived B_H1_high candidate was reused as the lambda_F=1 control; "
             "C_H1_high_noFIA is the single fresh lambda_F=0 arm. Labels and true updates "
             "were not consumed.", "", "| Candidate | MSE | PSNR dB | SSIM | LPIPS |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in ("x0", CONTROL_ARM, NEW_ARM):
        values = means[name]
        lines.append(f"| {name} | {values['mean_mse']:.9g} | {values['mean_psnr_db']} | "
                     f"{values['mean_ssim']:.9g} | {values['mean_lpips']:.9g} |")
    lines += ["", f"Archived-control metric reproduction within tolerance: **{control_matches}**.",
              "Exact code/cell feasibility and all public replay profiles are in `comparison.json` "
              "and the public output. Metrics are descriptive for this fixed batch/seed and do "
              "not establish reconstruction success, class recovery, or generalization."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison


def _tensor_hash(value: torch.Tensor) -> str:
    import hashlib
    import numpy as np

    array = value.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(array.dtype).encode("ascii"))
    h.update(str(array.shape).encode("ascii"))
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-output", type=Path, default=EXPECTED_PUBLIC_OUTPUT)
    parser.add_argument("--public-observation", type=Path, default=PUBLIC_ROOT)
    parser.add_argument("--naive", type=Path, default=NAIVE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--private-truth", type=Path, default=EXPECTED_PRIVATE_TRUTH)
    parser.add_argument("--output", type=Path, default=EXPECTED_PRIVATE_OUTPUT)
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
