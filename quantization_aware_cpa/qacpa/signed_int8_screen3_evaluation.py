"""Offline x-only evaluator for the frozen screen3 candidate sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from .deterministic_hybrid_setup import configure_runtime
from .full_update_evaluation import evaluate_candidate
from .report_full_update_iterations import _make_lpips_metric
from . import signed_int8_screen3_runner as runner


TASK_ID = runner.TASK_ID
PRIVATE_TRUTH = runner.REPO_ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"
TRUTH_SHA256 = "724bbacbe316cc28dac086db9cfaf6507f4c1af31144fd610cc480985aab72e2"


class Screen3EvaluationError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Screen3EvaluationError(f"expected object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def _private_x(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    path = Path(path).resolve()
    if path != PRIVATE_TRUTH.resolve() or runner.file_sha256(path) != TRUTH_SHA256:
        raise Screen3EvaluationError("private truth path/hash is not the registered baseline")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "x" not in payload:
        raise Screen3EvaluationError("private truth has no x field")
    x = payload["x"]
    if (not isinstance(x, torch.Tensor) or x.dtype != torch.float32
            or tuple(x.shape) != (16, 3, 224, 224) or not bool(torch.isfinite(x).all())):
        raise Screen3EvaluationError("private x schema mismatch")
    return x.detach().contiguous(), {
        "path": path.as_posix(), "sha256": runner.file_sha256(path),
        "keys_materialized": sorted(payload), "consumed_field": "x only",
        "labels_consumed": False, "true_updates_consumed": False,
    }


def _candidate(path: Path, expected_step: int) -> dict[str, Any]:
    result = _read(path / "result.json")
    candidate = path / ("candidate_5000.pt" if expected_step == 5000 and (path / "candidate_5000.pt").is_file() else "candidate.pt")
    if (result.get("status") != "COMPLETED" or result.get("completed_iterations") != expected_step
            or result.get("candidate_sha256") != runner.file_sha256(path / "candidate.pt")):
        # Center's result hash is its final candidate while its prefix is
        # separately bound below; member prefixes use candidate.pt directly.
        if not (expected_step == 5000 and path.name == "fia_center_25000"):
            raise Screen3EvaluationError(f"candidate result binding failed: {path}")
    if expected_step == 5000 and path.name == "fia_center_25000":
        prefix_hash = result.get("prefix_candidate_sha256")
        if prefix_hash != runner.file_sha256(candidate):
            raise Screen3EvaluationError("center 5000 candidate binding failed")
    payload = torch.load(candidate, map_location="cpu", weights_only=True)
    if set(payload) != {"pixels"} or payload["pixels"].dtype != torch.float32 or tuple(payload["pixels"].shape) != (16, 3, 224, 224):
        raise Screen3EvaluationError(f"candidate schema failed: {candidate}")
    return {"path": candidate, "sha256": runner.file_sha256(candidate), "result": result,
            "step": expected_step}


def public_gate(public_root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    public_root = Path(public_root).resolve()
    registration = _read(public_root / "registration.json")
    if (registration.get("task_id") != TASK_ID or registration.get("status") != "PUBLIC_COMPLETED"
            or registration.get("public_output_root") != public_root.as_posix()
            or registration.get("private_data_policy", {}).get("load_after_public_gate") is not True):
        raise Screen3EvaluationError("public registration gate failed")
    if registration.get("public_input_hashes") != runner.public_input_hashes(public_root / "observation"):
        raise Screen3EvaluationError("public input hashes changed")
    if registration.get("code_hashes") != runner._code_hashes() or registration.get("protected_source_hashes") != runner._protected_hashes():
        raise Screen3EvaluationError("source hash gate failed")
    separation = _read(public_root / "separation/result.json")
    selection = _read(public_root / "selection.json")
    if separation.get("status") != "COMPLETED" or separation.get("separation_calls") != 9:
        raise Screen3EvaluationError("nine separation calls are not complete")
    if (selection.get("status") != "FROZEN_PUBLIC_SELECTION"
            or len(selection.get("ranking", [])) != 8
            or len(selection.get("selected_member_indices", [])) != 3
            or selection.get("private_metrics_used") is not False):
        raise Screen3EvaluationError("public selection is not frozen/authenticated")
    candidates: dict[str, Path] = {}
    center = public_root / "fia_center_25000"
    _candidate(center, 5000)
    candidates["center_5000"] = center / "candidate_5000.pt"
    for index in range(8):
        branch = public_root / f"fia_member_{index:02d}_5000"
        _candidate(branch, 5000)
        candidates[f"member_{index:02d}_5000"] = branch / "candidate.pt"
    _candidate(center, 25000)
    candidates["center_25000"] = center / "candidate.pt"
    for rank, index in enumerate(selection["selected_member_indices"], 1):
        branch = public_root / f"fia_member_{int(index):02d}_25000"
        _candidate(branch, 25000)
        candidates[f"rank{rank}_member_{int(index):02d}_25000"] = branch / "candidate.pt"
    integrity = {
        "status": "PUBLIC_GATE_PASSED", "private_loaded": False,
        "public_input_hashes": registration["public_input_hashes"],
        "registration_sha256": runner.file_sha256(public_root / "registration.json"),
        "selection_sha256": runner.file_sha256(public_root / "selection.json"),
        "candidate_count": len(candidates), "candidate_paths": {key: value.as_posix() for key, value in candidates.items()},
    }
    return integrity, candidates


def _lpips(candidates: dict[str, torch.Tensor], reports: dict[str, dict[str, Any]], truth_x: torch.Tensor) -> dict[str, Any]:
    metric, provenance = _make_lpips_metric()
    mean = truth_x.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = truth_x.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    originals = (truth_x * std + mean).clamp(0, 1)
    for name, report in reports.items():
        for row in report["per_image"]:
            metric.reset()
            i, j = row["candidate_index"], row["truth_index"]
            value = metric((candidates[name][i:i + 1] - .5) * 2, (originals[j:j + 1] - .5) * 2)
            if value.numel() != 1 or not bool(torch.isfinite(value).all()):
                raise Screen3EvaluationError(f"nonfinite LPIPS: {name}/{i}")
            row["lpips"] = float(value.detach())
    return {"provenance": provenance, "matching_reused": True}


def _means(reports: dict[str, dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    result = {}
    for name, report in reports.items():
        rows = report["per_image"]
        psnr = [row["psnr_db"] for row in rows if row["psnr_db"] is not None]
        result[name] = {
            "mean_mse": sum(row["mse"] for row in rows) / 16,
            "mean_psnr_db": sum(psnr) / 16 if len(psnr) == 16 else None,
            "mean_ssim": sum(row["ssim"] for row in rows) / 16,
            "mean_lpips": sum(row["lpips"] for row in rows) / 16,
        }
    return result


def _contact_sheet(path: Path, names: list[str], candidates: dict[str, torch.Tensor], truth_x: torch.Tensor,
                   reports: dict[str, dict[str, Any]], title: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mean = truth_x.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = truth_x.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    truth = (truth_x * std + mean).clamp(0, 1)
    displayed = [truth]
    for name in names[1:]:
        aligned = torch.empty_like(candidates[name])
        for row in reports[name]["per_image"]:
            aligned[row["truth_index"]] = candidates[name][row["candidate_index"]]
        displayed.append(aligned)
    figure, axes = plt.subplots(16, len(displayed), figsize=(3 * len(displayed), 38), squeeze=False)
    try:
        for i in range(16):
            for j, images in enumerate(displayed):
                axes[i, j].imshow(images[i].permute(1, 2, 0).clamp(0, 1))
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(names[j])
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(path, dpi=110)
    finally:
        plt.close(figure)
    return path.name


def evaluate(public_root: Path, *, private_truth: Path = PRIVATE_TRUTH, output: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    public_root = Path(public_root).resolve()
    output = Path(output or (runner.PRIVATE_ROOT / "evaluation")).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing evaluation output: {output}")
    gate, paths = public_gate(public_root)
    configure_runtime(42, threads=1)
    runtime = {"threads": torch.get_num_threads(), "seed": 42, "device": "cpu"}
    truth_x, private_meta = _private_x(Path(private_truth))
    output.mkdir(parents=True, exist_ok=False)
    candidates = {name: torch.load(path, map_location="cpu", weights_only=True)["pixels"].contiguous()
                  for name, path in paths.items()}
    reports = {name: evaluate_candidate(path, Path(private_truth).resolve(), include_labels=False)
               for name, path in paths.items()}
    lpips = _lpips(candidates, reports, truth_x)
    means = _means(reports)
    center = means["center_25000"]
    rank1_name = next(name for name in means if name.startswith("rank1_"))
    contrast = {key: (means[rank1_name][key] - center[key]
                      if means[rank1_name][key] is not None and center[key] is not None else None)
                for key in ("mean_mse", "mean_psnr_db", "mean_ssim", "mean_lpips")}
    sheet5 = _contact_sheet(output / "contact_sheet_5000.png",
                            ["center_5000"] + [f"member_{i:02d}_5000" for i in range(8)],
                            candidates, truth_x, reports, "INT8 screen3 FIA5000")
    sheet25 = _contact_sheet(output / "contact_sheet_25000.png",
                             ["center_25000"] + [name for name in paths if name.startswith("rank")],
                             candidates, truth_x, reports, "INT8 screen3 FIA25000")
    result = {
        "task_id": TASK_ID, "status": "COMPLETED", "evaluation_status": "COMPLETED",
        "scientific_quality": "EVALUATED", "public_gate": gate, "private_truth": private_meta,
        "canonical_runtime": runtime, "candidate_order": list(paths), "candidate_paths": {key: str(value) for key, value in paths.items()},
        "candidate_hashes": {key: runner.file_sha256(value) for key, value in paths.items()},
        "matching": "Hungarian_pixel_MSE_once_per_candidate_reused_for_PSNR_SSIM_LPIPS_and_aligned_figures",
        "all_images_included": True, "image_metrics": reports, "mean_metrics": means, "lpips": lpips,
        "primary_contrast_rank1_minus_center25000": contrast, "screen_and_continuation": _read(public_root / "public_execution.json"),
        "figure_artifacts": [sheet5, sheet25], "private_x_only": True,
        "sender_true_updates_consumed": True, "attack_true_updates_consumed": False,
        "labels_consumed": False, "actual_image_cell_feasibility": "NOT_EVALUATED",
        "wall_seconds": time.perf_counter() - started,
        "limitations": ["one fixed B16/T10 observation and seed", "public FIA fit score selects top3",
                        "no private feedback or image averaging", "screen scores compare different targets",
                        "scientific contrast is descriptive and does not establish source recovery"],
    }
    _write(output / "comparison.json", result)
    _write(output / "results.json", result)
    lines = [f"# {TASK_ID}", "", "Status: **COMPLETED**.", "", "Public selection was frozen before private x-only evaluation.", "",
             "| Set | MSE | PSNR dB | SSIM | LPIPS |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in means.items():
        lines.append(f"| {name} | {row['mean_mse']:.12g} | {row['mean_psnr_db'] if row['mean_psnr_db'] is not None else 'n/a'} | {row['mean_ssim']:.12g} | {row['mean_lpips']:.12g} |")
    lines.extend(["", "Top3 ranking used public post5000 feature cosine distance + 0.1 normalized TV.",
                  "Exact actual-image cell feasibility: NOT_EVALUATED.",
                  "Sender true-update consumption is reported separately; attack and evaluator used no true updates."])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-truth", type=Path, default=PRIVATE_TRUTH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.public_root, private_truth=args.private_truth, output=args.output)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        _write(args.output / "failure.json", {"task_id": TASK_ID, "status": "FAILED",
                                                "error_type": type(error).__name__, "error": str(error)})
        print(f"{TASK_ID}: {type(error).__name__}: {error}", flush=True)
        return 1
    print(json.dumps({"task_id": TASK_ID, "status": result["status"],
                      "scientific_quality": result["scientific_quality"]}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
