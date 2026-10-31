"""Offline evaluator for the two frozen INT8 Direct FIA candidates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from .deterministic_hybrid_setup import configure_runtime
from .full_update_evaluation import evaluate_candidate
from .report_full_update_iterations import _make_lpips_metric
from . import signed_int8_randomized_runner as runner


TASK_ID = runner.TASK_ID
PRIVATE_TRUTH = runner.REPO_ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"
NAMES = ("center_int8", "randomized8_int8")


class EvaluationError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def _private_x(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    if path.resolve() != PRIVATE_TRUTH.resolve():
        raise EvaluationError("private truth path is outside the registered baseline bundle")
    if runner.file_sha256(path) != "724bbacbe316cc28dac086db9cfaf6507f4c1af31144fd610cc480985aab72e2":
        raise EvaluationError("private evaluation truth hash changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "x" not in payload:
        raise EvaluationError("private truth has no x field")
    x = payload["x"]
    if (not isinstance(x, torch.Tensor) or x.dtype != torch.float32
            or tuple(x.shape) != (16, 3, 224, 224) or not bool(torch.isfinite(x).all())):
        raise EvaluationError("private x is not finite FP32 [16,3,224,224]")
    return x.detach().contiguous(), {"path": path.resolve().as_posix(), "sha256": runner.file_sha256(path),
                                    "keys_materialized": sorted(payload), "consumed_field": "x only",
                                    "labels_consumed": False, "true_updates_consumed": False}


def public_gate(public_root: Path) -> dict[str, Any]:
    public_root = Path(public_root).resolve()
    registration = _read_json(public_root / "registration.json")
    if (registration.get("task_id") != TASK_ID or registration.get("status") != "PUBLIC_COMPLETED"
            or registration.get("public_output_root") != public_root.as_posix()
            or registration.get("private_data_policy", {}).get("load_after_public_gate") is not True):
        raise EvaluationError("outer public registration is not complete/authenticated")
    actual_inputs = runner.public_input_hashes(public_root / "observation")
    if registration.get("public_input_hashes") != actual_inputs:
        raise EvaluationError("public input hash binding changed")
    if registration.get("code_hashes") != runner._code_hashes() or registration.get("protected_source_hashes") != runner._protected_hashes():
        raise EvaluationError("registered source hash binding changed")
    separation = _read_json(public_root / "separation/result.json")
    if separation.get("status") != "COMPLETED" or separation.get("separation_calls") != 9:
        raise EvaluationError("nine-call public separation is incomplete")
    for name, arm in (("center_int8", "fia_center"), ("randomized8_int8", "fia_randomized8")):
        result = _read_json(public_root / arm / "result.json")
        candidate = public_root / arm / "candidate.pt"
        if (result.get("status") != "COMPLETED" or result.get("stage") != "INVERT"
                or result.get("fia_config") != runner.FIA_CONFIG
                or result.get("candidate_sha256") != runner.file_sha256(candidate)):
            raise EvaluationError(f"public candidate is not authenticated: {name}")
        payload = torch.load(candidate, map_location="cpu", weights_only=True)
        if set(payload) != {"pixels"} or payload["pixels"].dtype != torch.float32 or tuple(payload["pixels"].shape) != (16,3,224,224):
            raise EvaluationError(f"invalid public candidate schema: {name}")
    integrity = {"public_input_hashes": actual_inputs, "protected_source_hashes": runner._protected_hashes(),
                 "unchanged": True, "private_loaded": False}
    return {"status": "PUBLIC_GATE_PASSED", "registration": registration,
            "separation": separation, "integrity": integrity, "private_loaded": False}


def _mean_metrics(reports: dict[str, dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    values = {}
    for name, report in reports.items():
        finite_psnr = [row["psnr_db"] for row in report["per_image"] if row["psnr_db"] is not None]
        values[name] = {"mean_mse": sum(row["mse"] for row in report["per_image"]) / 16,
                        "mean_psnr_db": sum(finite_psnr) / 16 if len(finite_psnr) == 16 else None,
                        "mean_ssim": sum(row["ssim"] for row in report["per_image"]) / 16,
                        "mean_lpips": sum(row["lpips"] for row in report["per_image"]) / 16}
    return values


def _lpips(candidates: dict[str, torch.Tensor], reports: dict[str, dict[str, Any]], truth_x: torch.Tensor) -> dict[str, Any]:
    metric, provenance = _make_lpips_metric()
    mean = truth_x.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = truth_x.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    originals = (truth_x * std + mean).clamp(0, 1)
    for name, report in reports.items():
        for row in report["per_image"]:
            metric.reset()
            i, j = row["candidate_index"], row["truth_index"]
            score = metric((candidates[name][i:i+1] - .5) * 2, (originals[j:j+1] - .5) * 2)
            if score.numel() != 1 or not bool(torch.isfinite(score).all()):
                raise EvaluationError(f"nonfinite LPIPS score: {name}/{i}")
            row["lpips"] = float(score.detach())
    return {"provenance": provenance, "matching_reused": True}


def _figure(output: Path, candidates: dict[str, torch.Tensor], reports: dict[str, dict[str, Any]], truth_x: torch.Tensor) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mean = truth_x.new_tensor((.485, .456, .406)).view(1,3,1,1)
    std = truth_x.new_tensor((.229, .224, .225)).view(1,3,1,1)
    truth = (truth_x * std + mean).clamp(0, 1)
    aligned = {}
    for name, report in reports.items():
        ordered = torch.empty_like(candidates[name])
        for row in report["per_image"]:
            ordered[row["truth_index"]] = candidates[name][row["candidate_index"]]
        aligned[name] = ordered
    fig, axes = plt.subplots(16, 3, figsize=(9, 38), squeeze=False)
    try:
        for i in range(16):
            for j, values in enumerate((truth, aligned[NAMES[0]], aligned[NAMES[1]])):
                axes[i, j].imshow(values[i].permute(1,2,0).clamp(0,1))
                axes[i, j].axis("off")
                if i == 0: axes[i, j].set_title(("truth", NAMES[0], NAMES[1])[j])
        fig.tight_layout()
        fig.savefig(output / "all16_comparison.png", dpi=110)
    finally:
        plt.close(fig)
    return "all16_comparison.png"


def evaluate(public_root: Path, private_truth: Path = PRIVATE_TRUTH, output: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    public_root = Path(public_root).resolve()
    output = Path(output or (runner.PRIVATE_ROOT / "evaluation")).resolve()
    if output.exists():
        raise FileExistsError(f"refusing preexisting evaluation output: {output}")
    gate = public_gate(public_root)
    configure_runtime(42, threads=1)
    canonical_runtime = runner._runtime("cpu", "PRIVATE_METRICS_CANONICAL_RUNTIME")
    truth_x, private_meta = _private_x(Path(private_truth).resolve())
    output.mkdir(parents=True, exist_ok=False)
    paths = {"center_int8": public_root / "fia_center/candidate.pt",
             "randomized8_int8": public_root / "fia_randomized8/candidate.pt"}
    candidates = {name: torch.load(path, map_location="cpu", weights_only=True)["pixels"].contiguous()
                  for name, path in paths.items()}
    reports = {name: evaluate_candidate(path, Path(private_truth).resolve(), include_labels=False)
               for name, path in paths.items()}
    lpips = _lpips(candidates, reports, truth_x)
    means = _mean_metrics(reports)
    contrast = {metric: means[NAMES[1]][metric] - means[NAMES[0]][metric]
                if means[NAMES[1]][metric] is not None and means[NAMES[0]][metric] is not None else None
                for metric in ("mean_mse", "mean_psnr_db", "mean_ssim", "mean_lpips")}
    improvement = {}
    for metric, lower in (("mse", True), ("psnr_db", False), ("ssim", False), ("lpips", True)):
        old = {r["truth_index"]: r for r in reports[NAMES[0]]["per_image"]}
        new = {r["truth_index"]: r for r in reports[NAMES[1]]["per_image"]}
        deltas = [new[i][metric] - old[i][metric] for i in sorted(old)]
        improvement[metric] = {"improved": sum(d < 0 if lower else d > 0 for d in deltas),
                               "worsened": sum(d > 0 if lower else d < 0 for d in deltas),
                               "tied": sum(d == 0 for d in deltas),
                               "per_image_new_minus_center": [{"truth_index": i, "difference": new[i][metric] - old[i][metric]} for i in sorted(old)]}
    figure = _figure(output, candidates, reports, truth_x)
    result = {"task_id": TASK_ID, "status": "COMPLETED", "evaluation_status": "COMPLETED",
              "scientific_quality": "EVALUATED", "variant": "INT8_randomized8_minus_INT8_center",
              "public_gate": gate, "private_truth": private_meta, "canonical_runtime": canonical_runtime,
              "candidate_order": list(NAMES), "candidates": {name: {"path": path.as_posix(), "sha256": runner.file_sha256(path)} for name, path in paths.items()},
              "matching": "Hungarian_pixel_MSE_once_per_candidate_reused_for_PSNR_SSIM_LPIPS_and_aligned_figure",
              "all_images_included": True, "image_metrics": reports, "mean_metrics": means,
              "lpips": lpips, "contrast_randomized8_minus_center": contrast, "per_image_improvement": improvement,
              "labels_consumed": False, "true_updates_consumed": False,
              "sender_true_updates_consumed": True, "actual_image_cell_feasibility": "NOT_EVALUATED",
              "figure_artifacts": [figure], "wall_seconds": time.perf_counter() - started,
              "limitations": ["one fixed B16/T10 observation and seed", "M=8 is a first bounded screen, not an optimum claim",
                              "additional eight separation calls versus one INT8 center", "no private feedback to attack"]}
    _write_json(output / "comparison.json", result)
    _write_json(output / "results.json", result)
    lines = [f"# {TASK_ID}", "", "Status: **COMPLETED**.", "", "The public provenance gate passed before the private x-only evaluation.", "",
             "| Set | MSE | PSNR dB | SSIM | LPIPS |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in NAMES:
        row = means[name]
        lines.append(f"| {name} | {row['mean_mse']:.12g} | {row['mean_psnr_db']:.12g} | {row['mean_ssim']:.12g} | {row['mean_lpips']:.12g} |")
    lines += ["", "Contrast is randomized8 minus center; lower MSE/LPIPS and higher PSNR/SSIM are favorable.",
              "Exact actual-image cell feasibility: NOT_EVALUATED.", "The sender consumed archived true updates for conversion; attack/evaluator did not."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-truth", type=Path, default=PRIVATE_TRUTH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate(**vars(args))
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        _write_json(args.output / "failure.json", {"task_id": TASK_ID, "status": "FAILED",
                                                     "error_type": type(error).__name__, "error": str(error)})
        print(f"{TASK_ID}: {type(error).__name__}: {error}", flush=True)
        return 1
    print(json.dumps({"task_id": TASK_ID, "status": result["status"], "scientific_quality": result["scientific_quality"]}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
