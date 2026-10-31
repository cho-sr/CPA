"""Private sender diagnostic: compare every local-SGD prefix with exact replay.

This standalone tool reads private images and hard labels solely to validate
the forward recurrence. It is never an attack input, optimizer, or candidate
selection step. Reports must stay in the isolated private truth directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn import functional as F

from .full_update_pipeline import (
    ContractError, denormalize_images, file_digest, load_starting_vgg16,
    load_tensor_file, read_observation, validate_contract, validate_design, write_json,
)
from .full_update_replay import differentiable_sgd


def compare_parameter_delta(replayed: torch.Tensor, reference: torch.Tensor,
                            initial: torch.Tensor, radius: torch.Tensor,
                            *, chunk_rows: int = 64) -> dict:
    """Compare raw deltas in bounded row chunks, using final observed radii.

    Chunkwise torch.equal is equivalent to equality of the entire tensor.
    Positive-radius error uses observed h=2r/15, calculated as error/r*7.5
    in FP64 to avoid underflow. Zero-radius rows have separate absolute errors.
    """
    if (replayed.shape != reference.shape or replayed.shape != initial.shape or
            replayed.dtype != reference.dtype or replayed.dtype != initial.dtype or
            replayed.device != reference.device or replayed.device != initial.device):
        raise ValueError("Compared parameters must have identical shape, dtype, and device")
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    rows = 1 if replayed.ndim == 1 else replayed.shape[0]
    if radius.shape != (rows,) or not bool(torch.isfinite(radius).all()) or bool((radius < 0).any()):
        raise ValueError("Observed radius does not match the parameter row layout")
    candidate_rows = replayed.detach().reshape(rows, -1)
    reference_rows = reference.detach().reshape(rows, -1)
    initial_rows = initial.detach().reshape(rows, -1)
    result = {"exact_torch_equal": True, "max_absolute_error": 0.0,
              "max_positive_radius_error_steps": 0.0, "max_zero_radius_error_abs": 0.0}
    with torch.no_grad():
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            actual_delta = reference_rows[start:stop] - initial_rows[start:stop]
            candidate_delta = candidate_rows[start:stop]
            if not bool(torch.isfinite(actual_delta).all()) or not bool(torch.isfinite(candidate_delta).all()):
                raise RuntimeError("Nonfinite parameter delta in replay validation")
            result["exact_torch_equal"] &= torch.equal(candidate_delta, actual_delta)
            error = (candidate_delta.double() - actual_delta.double()).abs()
            result["max_absolute_error"] = max(result["max_absolute_error"], float(error.max()))
            row_radius = radius[start:stop].to(device=error.device, dtype=torch.float64)
            positive = row_radius > 0
            if bool(positive.any()):
                error_steps = error[positive] / row_radius[positive, None] * 7.5
                result["max_positive_radius_error_steps"] = max(
                    result["max_positive_radius_error_steps"], float(error_steps.max()))
            if bool((~positive).any()):
                result["max_zero_radius_error_abs"] = max(
                    result["max_zero_radius_error_abs"], float(error[~positive].max()))
    return result


def validate_replay(*, config: Path, observation: Path, checkpoint: Path,
                    truth: Path, output: Path, device: torch.device,
                    threads: int = 2) -> dict:
    """Run ordinary SGD once and regenerate all t=1..T replay prefixes."""
    truth, observation, output = Path(truth).resolve(), Path(observation).resolve(), Path(output).resolve()
    private_folder = truth.parent
    if (private_folder == observation or private_folder in observation.parents or
            observation in private_folder.parents):
        raise ContractError("Private truth and public packets must be in separate directory trees")
    if private_folder not in output.parents:
        raise ContractError("Validation report must be inside the private truth directory")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite replay validation report: {output}")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no automatic device substitution")
    configuration = json.loads(Path(config).read_text(encoding="utf-8"))
    if not isinstance(configuration, dict) or set(configuration) != {"contract", "attack"}:
        raise ContractError("Config must contain exactly contract and attack")
    validate_design(configuration["attack"])
    spec = validate_contract(configuration["contract"])
    manifest, observations = read_observation(observation, checkpoint)
    if manifest["contract"] != configuration["contract"]:
        raise ContractError("Config contract differs from the observed packet")
    saved = load_tensor_file(truth)
    if not isinstance(saved, dict) or set(saved) != {"x", "labels"}:
        raise ContractError("Use isolated sender evaluation_truth.pt containing exactly x and labels")
    images, labels = saved["x"], saved["labels"]
    if (not isinstance(images, torch.Tensor) or images.dtype != torch.float32 or
            tuple(images.shape) != tuple(manifest["contract"]["image_shape"])):
        raise ContractError("Private image shape/dtype differs from the observed contract")
    if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.long or
            tuple(labels.shape) != (len(images),) or bool(((labels < 0) | (labels >= 1000)).any())):
        raise ContractError("Private labels must be original int64 ImageNet class indices")
    pixels = denormalize_images(images)
    if (not bool(torch.isfinite(images).all()) or bool((pixels < -1e-5).any()) or
            bool((pixels > 1.0 + 1e-5).any())):
        raise ContractError("Private images must retain the original ImageNet normalization")
    x, y = images.to(device), labels.to(device)
    del saved, images, labels, pixels
    base = load_starting_vgg16(checkpoint, device)
    reference = deepcopy(base)
    initial = {name: parameter.detach() for name, parameter in base.named_parameters()}
    reference_parameters = dict(reference.named_parameters())
    if set(initial) != set(observations):
        raise ContractError("Model parameters differ from the complete observed parameter scope")
    optimizer = torch.optim.SGD(reference.parameters(), lr=spec.lr, momentum=spec.momentum,
                                weight_decay=spec.weight_decay, foreach=False)
    report = {
        "status": "running", "diagnostic": "private_sender_forward_replay_prefix_validation",
        "uses_private_truth": True, "raw_updates_saved": False,
        "reconstruction_success": "not_evaluated", "config_sha256": file_digest(config),
        "checkpoint_sha256": manifest["checkpoint_sha256"], "truth_sha256": file_digest(truth),
        "observation_manifest_sha256": file_digest(observation / "manifest.json"),
        "contract_sha256": manifest["contract_sha256"],
        "image_shape": manifest["contract"]["image_shape"], "local_steps": len(spec.batch_order),
        "parameter_tensors": len(initial), "device": str(device), "torch_version": torch.__version__,
        "determinism": {"matmul_tf32": False, "cudnn_tf32": False, "cudnn_benchmark": False,
                        "cudnn_deterministic": True, "deterministic_algorithms": True,
                        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
        "comparison": "raw final-minus-initial deltas; torch.equal for every parameter",
        "error_steps_reference": "final transmitted row radius, h=2r/15; zero-radius absolute error separate",
        "replay_create_graph": False, "replay_checkpoint_forward": False, "prefixes": [],
    }
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    write_json(output, report)
    try:
        for prefix, batch in enumerate(spec.batch_order, start=1):
            index = torch.tensor(batch, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            reference_loss = F.cross_entropy(reference(x[index]), y[index], reduction="mean")
            reference_loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            del reference_loss, index
            replayed = differentiable_sgd(base, x, y, replace(spec, batch_order=spec.batch_order[:prefix]),
                                          create_graph=False, checkpoint_forward=False)
            if set(replayed) != set(initial):
                raise RuntimeError("Replay omitted or introduced model parameters")
            per_tensor = {
                name: compare_parameter_delta(replayed[name], reference_parameters[name], initial[name],
                                               observations[name].radius)
                for name in initial
            }
            del replayed
            step = {"prefix": prefix, "exact_all_tensors": all(s["exact_torch_equal"] for s in per_tensor.values()),
                    "per_tensor": per_tensor}
            for metric in ("max_absolute_error", "max_positive_radius_error_steps", "max_zero_radius_error_abs"):
                step[metric] = max(s[metric] for s in per_tensor.values())
            report["prefixes"].append(step)
            report["wall_time_sec"] = time.perf_counter() - started
            write_json(output, report)
            print(json.dumps({key: value for key, value in step.items() if key != "per_tensor"}), flush=True)
        report["all_prefixes_exact"] = all(step["exact_all_tensors"] for step in report["prefixes"])
        report["status"] = "exact_match" if report["all_prefixes_exact"] else "numerical_difference"
        for metric in ("max_absolute_error", "max_positive_radius_error_steps", "max_zero_radius_error_abs"):
            report[metric] = max(step[metric] for step in report["prefixes"])
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report["wall_time_sec"] = time.perf_counter() - started
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        write_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "observation", "checkpoint", "truth", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        report = validate_replay(**vars(args))
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "message": str(error)}),
              file=sys.stderr, flush=True)
        return 2
    print(json.dumps({key: value for key, value in report.items() if key != "prefixes"}), flush=True)
    return 0 if report["all_prefixes_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
