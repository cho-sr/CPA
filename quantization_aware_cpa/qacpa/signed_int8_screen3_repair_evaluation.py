"""Strict offline evaluator for the repaired frozen INT8 screen3 outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from .deterministic_hybrid_setup import configure_runtime
from .full_update_evaluation import evaluate_candidate
from . import signed_int8_screen3_evaluation as prior_eval
from . import signed_int8_screen3_repair_runner as repair


TASK_ID = repair.TASK_ID
PRIVATE_TRUTH = repair.REPO_ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"
TRUTH_SHA256 = "724bbacbe316cc28dac086db9cfaf6507f4c1af31144fd610cc480985aab72e2"


class RepairEvaluationError(RuntimeError):
    pass


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def _private_x(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    path = Path(path).resolve()
    if path != PRIVATE_TRUTH.resolve() or repair.sha(path) != TRUTH_SHA256:
        raise RepairEvaluationError("private truth path/hash is not the registered baseline")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"x", "z", "labels"}:
        raise RepairEvaluationError("private truth schema changed or labels would be loaded")
    x = payload["x"]
    if (not isinstance(x, torch.Tensor) or x.dtype != torch.float32
            or tuple(x.shape) != (16, 3, 224, 224) or not bool(torch.isfinite(x).all())):
        raise RepairEvaluationError("private x schema mismatch")
    return x.detach().contiguous(), {
        "path": path.as_posix(), "sha256": repair.sha(path), "keys_present": sorted(payload),
        "consumed_field": "x only", "labels_consumed": False, "true_updates_consumed": False,
    }


def _candidate_paths(gate: dict[str, Any]) -> dict[str, Path]:
    raw = gate.get("candidates")
    if not isinstance(raw, dict) or set(raw) != {
        "center_5000", *[f"member_{i:02d}_5000" for i in range(8)], "center_25000",
        "rank1_member_01_25000", "rank2_member_07_25000", "rank3_member_05_25000",
    }:
        raise RepairEvaluationError("public gate did not provide exactly 13 candidates")
    paths: dict[str, Path] = {}
    for name, record in raw.items():
        if not isinstance(record, dict) or set(record) < {"path", "sha256", "step"}:
            raise RepairEvaluationError(f"candidate integrity record is incomplete: {name}")
        path = Path(record["path"]).resolve()
        if not path.is_file() or repair.sha(path) != record["sha256"]:
            raise RepairEvaluationError(f"candidate hash changed after public gate: {name}")
        if record["step"] not in {5000, 25000}:
            raise RepairEvaluationError(f"candidate step is invalid: {name}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if set(payload) != {"pixels"} or payload["pixels"].dtype != torch.float32:
            raise RepairEvaluationError(f"candidate schema mismatch: {name}")
        if tuple(payload["pixels"].shape) != (16, 3, 224, 224) or not bool(torch.isfinite(payload["pixels"]).all()):
            raise RepairEvaluationError(f"candidate tensor mismatch: {name}")
        paths[name] = path
    return paths


def evaluate(public_root: Path, *, private_truth: Path = PRIVATE_TRUTH,
             output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    public_root = Path(public_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing evaluation output: {output}")

    # This is the only operation before private loading.  It authenticates the
    # repaired registration, inherited parent bytes, and all 13 frozen sets.
    gate = repair.public_gate(public_root)
    paths = _candidate_paths(gate)
    configure_runtime(42, threads=1)
    runtime = {"threads": torch.get_num_threads(), "seed": 42, "device": "cpu"}
    truth_x, private_meta = _private_x(Path(private_truth))
    output.mkdir(parents=True, exist_ok=False)

    candidates = {}
    for name, path in paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        candidates[name] = payload["pixels"].contiguous()
    reports = {name: evaluate_candidate(path, Path(private_truth).resolve(), include_labels=False)
               for name, path in paths.items()}
    lpips = prior_eval._lpips(candidates, reports, truth_x)
    means = prior_eval._means(reports)
    center = means["center_25000"]
    rank1 = means["rank1_member_01_25000"]
    contrast = {key: (rank1[key] - center[key]
                      if rank1[key] is not None and center[key] is not None else None)
                for key in ("mean_mse", "mean_psnr_db", "mean_ssim", "mean_lpips")}
    sheet5 = prior_eval._contact_sheet(
        output / "contact_sheet_5000.png",
        ["center_5000"] + [f"member_{i:02d}_5000" for i in range(8)],
        candidates, truth_x, reports, "INT8 screen3 repaired FIA5000",
    )
    sheet25 = prior_eval._contact_sheet(
        output / "contact_sheet_25000.png",
        ["center_25000", "rank1_member_01_25000", "rank2_member_07_25000", "rank3_member_05_25000"],
        candidates, truth_x, reports, "INT8 screen3 repaired FIA25000",
    )
    result = {
        "task_id": TASK_ID, "status": "COMPLETED", "evaluation_status": "COMPLETED",
        "public_gate": gate, "canonical_runtime": runtime, "private_truth": private_meta,
        "candidate_order": list(paths), "candidate_paths": {key: value.as_posix() for key, value in paths.items()},
        "candidate_hashes": {key: repair.sha(value) for key, value in paths.items()},
        "matching": "Hungarian_pixel_MSE_once_per_candidate_reused_for_PSNR_SSIM_LPIPS_and_aligned_figures",
        "all_images_included": True, "image_metrics": reports, "mean_metrics": means, "lpips": lpips,
        "primary_contrast_rank1_member01_minus_center25000": contrast,
        "frozen_selection": [1, 7, 5], "private_x_only": True, "labels_consumed": False,
        "sender_true_updates_consumed": True, "attack_true_updates_consumed": False,
        "actual_image_cell_feasibility": "NOT_EVALUATED", "scientific_quality": "EVALUATED",
        "figure_artifacts": [sheet5, sheet25], "wall_seconds": time.perf_counter() - started,
        "limitations": ["one fixed B16/T10 observation and seed", "public top3 selection used no private metrics",
                        "no private feedback or image averaging", "actual-image cell was not run"],
    }
    _write(output / "comparison.json", result)
    _write(output / "results.json", result)
    lines = [f"# {TASK_ID}", "", "Status: **COMPLETED**.",
             "", "The public gate authenticated all 13 frozen sets before private x-only loading.", "",
             "| Set | MSE | PSNR dB | SSIM | LPIPS |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in means.items():
        lines.append(f"| {name} | {row['mean_mse']:.12g} | {row['mean_psnr_db'] if row['mean_psnr_db'] is not None else 'n/a'} | {row['mean_ssim']:.12g} | {row['mean_lpips']:.12g} |")
    lines.extend(["", "Primary contrast is rank1 member01 minus center25000.",
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
