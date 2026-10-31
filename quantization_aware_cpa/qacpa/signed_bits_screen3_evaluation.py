"""Strict x-only offline evaluator for one frozen bit-sweep screen3 result."""
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
from . import signed_bits_screen3_runner as runner
from .signed_bits_observation import EVALUATION_TRUTH, EVALUATION_TRUTH_SHA


class BitsEvaluationError(RuntimeError):
    pass


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def _private_x(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    path = Path(path).resolve()
    if path != EVALUATION_TRUTH.resolve() or runner.file_sha256(path) != EVALUATION_TRUTH_SHA:
        raise BitsEvaluationError("private truth path/hash is not registered")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"x", "z", "labels"}:
        raise BitsEvaluationError("private truth schema is not the registered x/z/labels bundle")
    x = payload["x"]
    if (not isinstance(x, torch.Tensor) or x.dtype != torch.float32
            or tuple(x.shape) != (16, 3, 224, 224) or not bool(torch.isfinite(x).all())):
        raise BitsEvaluationError("private x schema mismatch")
    return x.detach().contiguous(), {"path": path.as_posix(), "sha256": runner.file_sha256(path),
        "keys_present": sorted(payload), "consumed_field": "x only", "labels_consumed": False,
        "true_updates_consumed": False}


def _paths(gate: dict[str, Any]) -> dict[str, Path]:
    raw = gate.get("candidates")
    if not isinstance(raw, dict) or len(raw) != 13:
        raise BitsEvaluationError("public gate did not produce exactly 13 candidates")
    paths: dict[str, Path] = {}
    for name, record in raw.items():
        path = Path(record["path"]).resolve()
        if not path.is_file() or runner.file_sha256(path) != record["sha256"]:
            raise BitsEvaluationError(f"candidate changed after public gate: {name}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        pixels = payload.get("pixels") if isinstance(payload, dict) else None
        if set(payload) != {"pixels"} or pixels.dtype != torch.float32 or tuple(pixels.shape) != (16, 3, 224, 224):
            raise BitsEvaluationError(f"candidate schema mismatch: {name}")
        paths[name] = path
    return paths


def evaluate(bits: int, public_root: Path, *, output: Path,
             private_truth: Path = EVALUATION_TRUTH) -> dict[str, Any]:
    started = time.perf_counter()
    public_root, output = Path(public_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing bit evaluation output: {output}")
    gate = runner.public_gate(bits, public_root)
    paths = _paths(gate)
    configure_runtime(42, threads=1)
    truth_x, private_meta = _private_x(private_truth)
    runtime = {"threads": torch.get_num_threads(), "seed": 42, "device": "cpu"}
    output.mkdir(parents=True, exist_ok=False)
    candidates = {name: torch.load(path, map_location="cpu", weights_only=True)["pixels"].contiguous()
                  for name, path in paths.items()}
    reports = {name: evaluate_candidate(path, Path(private_truth).resolve(), include_labels=False)
               for name, path in paths.items()}
    lpips = prior_eval._lpips(candidates, reports, truth_x)
    means = prior_eval._means(reports)
    center = means["center_25000"]
    rank1_name = next(name for name in means if name.startswith("rank1_"))
    contrast = {key: (means[rank1_name][key] - center[key]
                      if means[rank1_name][key] is not None and center[key] is not None else None)
                for key in ("mean_mse", "mean_psnr_db", "mean_ssim", "mean_lpips")}
    sheet5 = prior_eval._contact_sheet(output / f"contact_sheet_b{bits}_5000.png",
        ["center_5000"] + [f"member_{i:02d}_5000" for i in range(8)],
        candidates, truth_x, reports, f"bits{bits} screen3 FIA5000")
    selected = [name for name in paths if name.startswith("rank")]
    sheet25 = prior_eval._contact_sheet(output / f"contact_sheet_b{bits}_25000.png",
        ["center_25000"] + selected, candidates, truth_x, reports, f"bits{bits} screen3 FIA25000")
    result = {
        "task_id": runner.bit_task_id(bits), "schedule_task_id": runner.TASK_ID, "bits": bits,
        "status": "COMPLETED", "evaluation_status": "COMPLETED", "public_gate": gate,
        "canonical_runtime": runtime, "private_truth": private_meta, "candidate_order": list(paths),
        "candidate_paths": {name: path.as_posix() for name, path in paths.items()},
        "candidate_hashes": {name: runner.file_sha256(path) for name, path in paths.items()},
        "matching": "Hungarian_pixel_MSE_once_per_candidate_reused_for_PSNR_SSIM_LPIPS_and_aligned_figures",
        "all_images_included": True, "image_metrics": reports, "mean_metrics": means, "lpips": lpips,
        "primary_contrast_rank1_minus_center25000": contrast,
        "frozen_selection": gate["selection"], "private_x_only": True, "labels_consumed": False,
        "sender_true_updates_consumed": True, "attack_true_updates_consumed": False,
        "actual_image_cell_feasibility": "NOT_EVALUATED", "scientific_quality": "EVALUATED",
        "figure_artifacts": [sheet5, sheet25], "wall_seconds": time.perf_counter() - started,
        "limitations": ["one fixed B16/T10 observation and seed", "public top3 only; no private selection",
                        "no private feedback or image averaging", "actual-image cell not run"],
    }
    _write(output / "comparison.json", result)
    _write(output / "results.json", result)
    lines = [f"# {runner.bit_task_id(bits)}", "", "Status: **COMPLETED**.", "",
             "The public gate authenticated all 13 frozen sets before private x-only loading.", "",
             "| Set | MSE | PSNR dB | SSIM | LPIPS |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in means.items():
        lines.append(f"| {name} | {row['mean_mse']:.12g} | {row['mean_psnr_db']:.12g} | {row['mean_ssim']:.12g} | {row['mean_lpips']:.12g} |")
    lines += ["", "Primary contrast is public rank1 minus center25000.",
              "Actual-image cell feasibility: NOT_EVALUATED.",
              "Private evaluator consumed x only; labels and true updates were not consumed."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-truth", type=Path, default=EVALUATION_TRUTH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.bits, args.public_root, output=args.output, private_truth=args.private_truth)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        _write(args.output / "failure.json", {"task_id": runner.bit_task_id(args.bits), "status": "FAILED",
                                               "error_type": type(error).__name__, "error": str(error)})
        print(f"{runner.TASK_ID}: {type(error).__name__}: {error}", flush=True)
        return 1
    print(json.dumps({"task_id": result["task_id"], "status": result["status"],
                      "scientific_quality": result["scientific_quality"]}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
