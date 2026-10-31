"""Bounded actual-image signed-cell screen for the fixed VGG16 FC update.

This runner consumes only the public signed q/s packet, the public starting
checkpoint, the saved public naive pixels, and the public FIA source target.
It creates a fresh pixel-space continuation and evaluates the preregistered
FIA/control and signed-cell arms.  Private sender tensors are deliberately not
accepted by this module; the separate evaluator owns the final offline step.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .deterministic_hybrid_setup import configure_runtime
from .fc_replay import fc_recomputed_gradients, fc_sgd_updates
from .full_update_pipeline import load_starting_vgg16
from .signed_baseline_observation import CHECKPOINT_SHA, FC_SHAPES, TRAINING, digest
from .signed_e2e_cells import (
    CELL_CHUNK_SIZE, FC_NAMES, CellTable, DeviceCellTable,
    load_public_cell_tables, signed_cell_loss, summarize_signed_update,
)


TASK_ID = "CPA_SIGNED_E2E_CELL_SCREEN_20260916"
PLAN_VERSION = 1
IMAGE_COUNT = 16
FEATURE_DIM = 25088
LOCAL_STEPS = 10
LOCAL_LR = 0.001
IMAGE_STEPS = 100
IMAGE_LR = 0.001
IMAGE_LR_FINAL = 0.0001
TV_WEIGHT = 0.1
CELL_CHUNK = CELL_CHUNK_SIZE
MAX_SECONDS = 45 * 60
MAX_CUDA_ALLOCATED = 12 * 1024**3
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
REPO_ROOT = Path(__file__).resolve().parents[2]
QACPA_ROOT = REPO_ROOT / "quantization_aware_cpa"
EXPECTED_CHECKPOINT = Path("C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
EXPECTED_PUBLIC = QACPA_ROOT / "results/signed_int4_baseline_B16_T10/observation"
EXPECTED_NAIVE = QACPA_ROOT / "results/signed_int4_baseline_B16_T10/naive"


class ExperimentError(RuntimeError):
    """A public contract, numerical path, budget, or resource check failed."""


class DeadlineExceeded(ExperimentError):
    pass


class ResourceCapExceeded(ExperimentError):
    pass


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    return digest(Path(path))


def tensor_sha256(value: Tensor) -> str:
    import numpy as np
    array = value.detach().cpu().contiguous().numpy()
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(memoryview(array).cast("B"))
    return hasher.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode("utf-8")).hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def protected_paths() -> list[Path]:
    return [
        REPO_ROOT / "quantization_aware_cpa/qacpa/fc_replay.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/full_update_replay.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/full_update_adjoint.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/full_update_pipeline.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/deterministic_hybrid_setup.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/signed_baseline_observation.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/signed_cell_rank.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/signed_cell_rank_evaluation.py",
        REPO_ROOT / "quantization_aware_cpa/qacpa/upstream_naive.py",
        REPO_ROOT / "quantized_update_cpa/cell_utils.py",
        REPO_ROOT / "official_cpa/src/gradient_inversion.py",
        REPO_ROOT / "official_cpa/src/feature_inversion.py",
        REPO_ROOT / "official_cpa/src/vgg16.py",
    ]


def hash_files(paths: list[Path]) -> dict[str, str]:
    return {_rel(path): file_sha256(path) for path in paths}


def hash_public_inputs(public: Path, naive: Path, checkpoint: Path) -> dict[str, str]:
    paths = [public / "manifest.json", naive / "candidate.pt", naive / "sources.pt", checkpoint]
    manifest = json.loads((public / "manifest.json").read_text(encoding="utf-8"))
    for descriptor in manifest["layers"].values():
        paths.append(public / descriptor["file"])
    return hash_files(paths)


def _require_tensor(value: Any, *, name: str, shape: tuple[int, ...], dtype: torch.dtype,
                    device: str = "cpu") -> Tensor:
    if (not isinstance(value, Tensor) or value.layout != torch.strided
            or value.dtype != dtype or value.device.type != device
            or tuple(value.shape) != shape or not bool(torch.isfinite(value).all())):
        raise ExperimentError(f"{name} must be finite {device} {dtype} with shape {shape}")
    return value.contiguous()


def load_public_start(naive: Path) -> tuple[Tensor, Tensor, Tensor]:
    candidate = torch.load(naive / "candidate.pt", map_location="cpu", weights_only=True)
    sources = torch.load(naive / "sources.pt", map_location="cpu", weights_only=True)
    if not isinstance(candidate, dict) or set(candidate) != {"pixels"}:
        raise ExperimentError("saved naive candidate must contain only pixels")
    if not isinstance(sources, dict) or set(sources) != {"S_hat", "target_z"}:
        raise ExperimentError("saved naive sources must contain only S_hat and target_z")
    pixels = _require_tensor(candidate["pixels"], name="x0",
                             shape=(IMAGE_COUNT, 3, 224, 224), dtype=torch.float32)
    if bool((pixels < 0).any()) or bool((pixels > 1).any()):
        raise ExperimentError("saved candidate pixels are outside [0,1]")
    source = _require_tensor(sources["S_hat"], name="S_hat",
                             shape=(IMAGE_COUNT, FEATURE_DIM), dtype=torch.float32)
    target = _require_tensor(sources["target_z"], name="target_z",
                             shape=(IMAGE_COUNT, FEATURE_DIM), dtype=torch.float32)
    if not torch.equal(target, source.abs()):
        raise ExperimentError("saved target_z is not exactly abs(S_hat)")
    if bool((target < 0).any()):
        raise ExperimentError("saved target_z is negative")
    return pixels, source, target


def normalize_pixels(pixels: Tensor) -> Tensor:
    mean = pixels.new_tensor(MEAN).view(1, 3, 1, 1)
    std = pixels.new_tensor(STD).view(1, 3, 1, 1)
    return (pixels - mean) / std


def total_variation(normalized: Tensor) -> Tensor:
    dx = torch.mean(torch.abs(normalized[:, :, :, :-1] - normalized[:, :, :, 1:]))
    dy = torch.mean(torch.abs(normalized[:, :, :-1, :] - normalized[:, :, 1:, :]))
    return dx + dy


def _feature_forward(model: nn.Module, pixels: Tensor) -> Tensor:
    normalized = normalize_pixels(pixels)
    features = model.avgpool(model.features(normalized)).flatten(1)
    if (features.ndim != 2 or tuple(features.shape) != (IMAGE_COUNT, FEATURE_DIM)
            or not bool(torch.isfinite(features).all()) or bool((features < 0).any())):
        raise ExperimentError("VGG feature output has an invalid shape, value, or ReLU sign")
    return features


def cosine_loss(features: Tensor, target: Tensor) -> Tensor:
    if features.shape != target.shape or not bool(torch.isfinite(target).all()):
        raise ExperimentError("FIA feature target shape or finite-value check failed")
    norms = torch.linalg.vector_norm(features, dim=1) * torch.linalg.vector_norm(target, dim=1)
    if bool((norms == 0).any()) or not bool(torch.isfinite(norms).all()):
        raise ExperimentError("FIA cosine has a zero or nonfinite vector norm")
    cosine = (features * target).sum(dim=1) / norms
    return (1 - cosine).mean()


def load_model(checkpoint: Path, device: torch.device) -> nn.Module:
    if file_sha256(checkpoint) != CHECKPOINT_SHA:
        raise ExperimentError("starting checkpoint hash mismatch")
    model = load_starting_vgg16(checkpoint, device)
    model.eval()
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    return model


def labels_from_public_logits(logits: Tensor, bias_q: Tensor) -> tuple[dict, dict[str, Tensor]]:
    if tuple(logits.shape) != (IMAGE_COUNT, 1000) or not bool(torch.isfinite(logits).all()):
        raise ExperimentError("public classifier logits have the wrong shape or finite status")
    h0 = logits.argmax(dim=1).to(torch.int64)
    positive_classes = [int(x) for x in torch.nonzero(bias_q > 0, as_tuple=False).flatten().tolist()]
    k = len(positive_classes)
    evidence: dict[str, Any] = {
        "positive_bias_code_classes_ascending": positive_classes,
        "positive_bias_code_count": k,
        "rule": "positive signed bias code is one-way evidence only; no absence or multiplicity claim",
        "h0": {"name": "H0", "rule": "public starting classifier argmax; first index wins exact ties",
               "labels": [int(x) for x in h0.tolist()]},
    }
    if k > IMAGE_COUNT:
        evidence["status"] = "MODEL_OR_OBSERVATION_LABEL_HINT_CONFLICT"
        return evidence, {"H0": h0}
    from scipy.optimize import linear_sum_assignment
    reward = torch.empty((IMAGE_COUNT, IMAGE_COUNT), dtype=torch.float64)
    if k:
        reward[:, :k] = logits[:, positive_classes].detach().double()
    if k < IMAGE_COUNT:
        reward[:, k:] = logits.detach().double().amax(dim=1, keepdim=True)
    rows, columns = linear_sum_assignment((-reward).numpy())
    assignment = [{"row": int(row), "column": int(column), "reward": float(reward[row, column])}
                  for row, column in zip(rows.tolist(), columns.tolist())]
    h1 = h0.clone()
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column < k:
            h1[row] = positive_classes[column]
    evidence.update({
        "status": "COMPLETED",
        "scipy_version": __import__("scipy").__version__,
        "assignment": assignment,
        "h1": {"name": "H1", "rule": "mandatory positive-code classes plus free slots using public logits",
               "labels": [int(x) for x in h1.tolist()]},
        "h1_equals_h0": bool(torch.equal(h0, h1)),
        "labels_are_hypotheses": True,
    })
    return evidence, {"H0": h0, "H1": h1}


class Clock:
    def __init__(self, started: float):
        self.started = started
        self.stages: dict[str, dict[str, float]] = {}

    def seconds(self) -> float:
        return time.perf_counter() - self.started

    def check(self, label: str) -> None:
        elapsed = self.seconds()
        if elapsed > MAX_SECONDS:
            raise DeadlineExceeded(f"global 45-minute deadline exceeded at {label}: {elapsed:.3f}s")

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        self.check(f"stage_start:{name}")
        try:
            yield
        finally:
            self.stages[name] = {"wall_seconds": time.perf_counter() - start,
                                 "total_seconds": self.seconds()}


def cuda_resource(device: torch.device, label: str) -> dict[str, Any]:
    if device.type != "cuda":
        return {"label": label, "allocated_bytes": 0, "reserved_bytes": 0, "cap_bytes": MAX_CUDA_ALLOCATED}
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    if allocated > MAX_CUDA_ALLOCATED:
        raise ResourceCapExceeded(f"PyTorch allocated CUDA memory cap exceeded at {label}: {allocated}")
    return {"label": label, "allocated_bytes": allocated, "reserved_bytes": reserved,
            "cap_bytes": MAX_CUDA_ALLOCATED}


def _device_tables(tables: Mapping[str, CellTable], device: torch.device) -> dict[str, DeviceCellTable]:
    return {name: table.on(device) for name, table in tables.items()}


def _cell_callback(device_tables: Mapping[str, DeviceCellTable]):
    def callback(updates: Mapping[str, Tensor]) -> Tensor:
        return signed_cell_loss(updates, device_tables, chunk_size=CELL_CHUNK)
    return callback


def recomputed_cell_gradient(model: nn.Module, pixels: Tensor, labels: Tensor,
                             device_tables: Mapping[str, DeviceCellTable]) -> tuple[Tensor, Tensor, Tensor]:
    features = _feature_forward(model, pixels)
    cell_value, feature_gradient, target_gradient = fc_recomputed_gradients(
        model.classifier, features, labels, LOCAL_STEPS, LOCAL_LR,
        _cell_callback(device_tables))
    if target_gradient is not None:
        raise ExperimentError("hard-label replay unexpectedly returned a target gradient")
    image_gradient, = torch.autograd.grad(features, pixels, grad_outputs=feature_gradient)
    if not bool(torch.isfinite(image_gradient).all()):
        raise ExperimentError("recomputed cell image gradient is nonfinite")
    return cell_value.detach(), image_gradient.detach(), features


def _cpu_fit(updates: Mapping[str, Tensor], tables: Mapping[str, CellTable]) -> dict[str, Any]:
    per_tensor = {}
    for name in FC_NAMES:
        per_tensor[name] = summarize_signed_update(updates[name].detach().cpu().float(), tables[name])
    total = sum(item["numel"] for item in per_tensor.values())
    mismatch = sum(item["code_mismatch_count"] for item in per_tensor.values())
    violations = sum(item["bounded_cell_violation_count"] for item in per_tensor.values())
    return {
        "per_tensor": per_tensor,
        "overall_numel": total,
        "overall_code_mismatch_count": mismatch,
        "overall_code_mismatch_fraction": mismatch / total,
        "overall_bounded_cell_violation_count": violations,
        "overall_bounded_cell_violation_fraction": violations / total,
        "cell_loss_equal_tensor_mean": sum(item["cell_penalty"] for item in per_tensor.values()) / len(per_tensor),
        "exact_feasible": mismatch == 0 and violations == 0,
    }


def canonical_profile(model: nn.Module, pixels: Tensor, labels: Tensor,
                      tables: Mapping[str, CellTable], device: torch.device) -> dict[str, Any]:
    with torch.no_grad():
        features = _feature_forward(model, pixels)
    updates = fc_sgd_updates(model.classifier, features.detach(), labels,
                             steps=LOCAL_STEPS, lr=LOCAL_LR, create_graph=False,
                             checkpoint_forward=False)
    fit = _cpu_fit(updates, tables)
    fit["feature_norm_mean"] = float(torch.linalg.vector_norm(features, dim=1).mean())
    fit["feature_norm_min"] = float(torch.linalg.vector_norm(features, dim=1).min())
    fit["feature_norm_max"] = float(torch.linalg.vector_norm(features, dim=1).max())
    return fit


def compare_replay_paths(model: nn.Module, pixels: Tensor, labels: Tensor,
                         tables: Mapping[str, CellTable], device_tables: Mapping[str, DeviceCellTable],
                         device: torch.device) -> dict[str, Any]:
    with torch.no_grad():
        no_grad_features = _feature_forward(model, pixels.detach())
    feature_pixels = pixels.detach().clone().requires_grad_(True)
    grad_features = _feature_forward(model, feature_pixels)
    feature_equal = torch.equal(no_grad_features, grad_features.detach())
    del feature_pixels, grad_features
    captured: dict[str, Tensor] = {}

    def capture_loss(updates: Mapping[str, Tensor]) -> Tensor:
        for name in FC_NAMES:
            captured[name] = updates[name].detach().cpu().clone()
        return signed_cell_loss(updates, device_tables, chunk_size=CELL_CHUNK)

    adjoint_pixels = pixels.detach().clone().requires_grad_(True)
    adjoint_features = _feature_forward(model, adjoint_pixels)
    adjoint_loss, _, _ = fc_recomputed_gradients(
        model.classifier, adjoint_features, labels, LOCAL_STEPS, LOCAL_LR, capture_loss)
    canonical = fc_sgd_updates(model.classifier, no_grad_features.detach(), labels,
                               steps=LOCAL_STEPS, lr=LOCAL_LR, create_graph=False,
                               checkpoint_forward=False)
    tensors = {}
    mismatch = False
    for name in FC_NAMES:
        left, right = canonical[name].detach().cpu(), captured[name]
        difference = (left - right).abs()
        scale = float(tables[name].scale_cpu)
        tensors[name] = {"torch_equal": bool(torch.equal(left, right)),
                         "max_abs_difference": float(difference.max()),
                         "max_difference_in_s": float(difference.max()) / scale,
                         "fixed_s_code_mismatch_count": int(
                             (torch.round(right / tables[name].scale_cpu).clamp(-7, 7).to(torch.int8)
                              != torch.round(left / tables[name].scale_cpu).clamp(-7, 7).to(torch.int8)).sum())}
        mismatch |= not tensors[name]["torch_equal"]
    result = {"feature_grad_enabled_equals_no_grad": feature_equal,
              "terminal_deltas_torch_equal": not mismatch,
              "adjoint_cell_loss": float(adjoint_loss), "per_tensor": tensors}
    if not feature_equal or mismatch:
        raise ExperimentError("NUMERICAL_PATH_MISMATCH: " + json.dumps(result, allow_nan=False))
    del no_grad_features, adjoint_pixels, adjoint_features, canonical, captured
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def initial_calibration(model: nn.Module, pixels_cpu: Tensor, target_cpu: Tensor,
                        labels: Mapping[str, Tensor], tables: Mapping[str, CellTable],
                        device_tables: Mapping[str, DeviceCellTable], device: torch.device,
                        clock: Clock) -> dict[str, Any]:
    result: dict[str, Any] = {}
    common = pixels_cpu.to(device=device).clone()
    common.requires_grad_(True)
    for name, label in labels.items():
        clock.check(f"calibration:{name}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        cell_value, image_gradient, features = recomputed_cell_gradient(
            model, common, label.to(device), device_tables)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result[name] = {
            "cell_loss": float(cell_value),
            "cell_image_gradient_norm": float(image_gradient.norm()),
            "feature_norm_mean": float(torch.linalg.vector_norm(features, dim=1).mean()),
            "finite": bool(torch.isfinite(cell_value) and torch.isfinite(image_gradient).all()),
            "wall_seconds": time.perf_counter() - started,
            "resource": cuda_resource(device, f"calibration:{name}"),
        }
        del cell_value, image_gradient, features
        if device.type == "cuda":
            torch.cuda.empty_cache()
    h0_loss = result["H0"]["cell_loss"]
    if not math.isfinite(h0_loss):
        raise ExperimentError("H0 initial cell loss is nonfinite")
    lambda_base = 1.0 / max(1.0, h0_loss)
    result["weights"] = {"lambda_base": lambda_base, "lambda_low": lambda_base,
                          "lambda_high": 10.0 * lambda_base,
                          "definition": "lambda_base=1/max(1,L_C0_H0), shared by both hypotheses"}
    return result


def arm_lr(step_zero_based: int) -> float:
    progress = step_zero_based / max(1, IMAGE_STEPS - 1)
    fraction = IMAGE_LR_FINAL / IMAGE_LR + (1 - IMAGE_LR_FINAL / IMAGE_LR) * (
        1 + math.cos(math.pi * progress)) / 2
    return IMAGE_LR * fraction


def _arm_metrics(model: nn.Module, pixels: Tensor, x0: Tensor, target: Tensor) -> dict[str, Any]:
    with torch.no_grad():
        features = _feature_forward(model, pixels)
        feature_loss = cosine_loss(features, target)
        tv = total_variation(normalize_pixels(pixels))
    return {"feature_target_cosine_loss": float(feature_loss), "tv_normalized": float(tv),
            "feature_norm_mean": float(torch.linalg.vector_norm(features, dim=1).mean()),
            "feature_norm_min": float(torch.linalg.vector_norm(features, dim=1).min()),
            "feature_norm_max": float(torch.linalg.vector_norm(features, dim=1).max()),
            "pixel_min": float(pixels.min()), "pixel_max": float(pixels.max()),
            "pixel_mean": float(pixels.mean()), "x0_displacement_l2": float((pixels - x0).norm())}


def run_arm(model: nn.Module, name: str, x0_cpu: Tensor, target_cpu: Tensor,
            labels: Tensor | None, lambda_cell: float, tables: Mapping[str, CellTable],
            device_tables: Mapping[str, DeviceCellTable], output: Path, device: torch.device,
            clock: Clock) -> dict[str, Any]:
    folder = output / name
    folder.mkdir(exist_ok=False)
    pixels = nn.Parameter(x0_cpu.to(device=device).clone())
    target = target_cpu.to(device=device)
    y = labels.to(device=device) if labels is not None else None
    optimizer = torch.optim.Adam([pixels], lr=IMAGE_LR, betas=(.9, .999), eps=1e-8, weight_decay=0.0)
    logs: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    use_cell = labels is not None
    for step in range(IMAGE_STEPS):
        clock.check(f"arm:{name}:step:{step + 1}")
        learning_rate = arm_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        features = _feature_forward(model, pixels)
        fia = cosine_loss(features, target)
        tv = total_variation(normalize_pixels(pixels))
        base = fia + TV_WEIGHT * tv
        base_gradient, = torch.autograd.grad(base, pixels, retain_graph=True)
        cell_value = None
        cell_gradient = None
        if use_cell:
            cell_value, feature_gradient, _ = fc_recomputed_gradients(
                model.classifier, features, y, LOCAL_STEPS, LOCAL_LR,
                _cell_callback(device_tables))
            cell_gradient, = torch.autograd.grad(features, pixels, grad_outputs=feature_gradient)
        total_gradient = base_gradient if cell_gradient is None else base_gradient + lambda_cell * cell_gradient
        if not bool(torch.isfinite(total_gradient).all()):
            raise ExperimentError(f"nonfinite image gradient in {name} step {step + 1}")
        pixels.grad = total_gradient
        before = pixels.detach().clone()
        optimizer.step()
        with torch.no_grad():
            pixels.clamp_(0.0, 1.0)
            displacement = float((pixels - before).norm())
            clamp_fraction = float(((pixels == 0) | (pixels == 1)).float().mean())
        row = {"iteration": step + 1, "learning_rate": learning_rate,
               "pre_step_fia_loss": float(fia), "pre_step_tv_normalized": float(tv),
               "pre_step_base_loss": float(base),
               "pre_step_cell_loss": None if cell_value is None else float(cell_value),
               "base_gradient_norm": float(base_gradient.norm()),
               "cell_gradient_norm": None if cell_gradient is None else float(cell_gradient.norm()),
               "total_gradient_norm": float(total_gradient.norm()),
               "actual_pixel_step_l2": displacement, "post_step_clamp_fraction": clamp_fraction,
               "elapsed_seconds": time.perf_counter() - started}
        logs.append(row)
        write_json(folder / "optimization_log.json", logs)
        if device.type == "cuda":
            cuda_resource(device, f"arm:{name}:step:{step + 1}")
        del features, fia, tv, base, base_gradient, cell_value, cell_gradient, total_gradient, before
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    final_pixels = pixels.detach().clone()
    final_metrics = _arm_metrics(model, final_pixels, x0_cpu.to(device), target)
    torch.save({"pixels": final_pixels.cpu()}, folder / "candidate.pt")
    torch.save({"pixels": final_pixels.cpu(), "labels": None if y is None else y.cpu(),
                "lambda_cell": lambda_cell, "completed_iterations": IMAGE_STEPS},
               folder / "attack_state.pt")
    result = {"status": "COMPLETED", "arm": name, "optimized_label_hypothesis": None if not use_cell else name.split("_")[1],
              "lambda_cell": lambda_cell, "iterations": IMAGE_STEPS,
              "optimization_seconds": time.perf_counter() - started,
              "peak_cuda_allocated_bytes": 0 if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device)),
              "peak_cuda_reserved_bytes": 0 if device.type != "cuda" else int(torch.cuda.max_memory_reserved(device)),
              "final_metrics": final_metrics, "candidate_sha256": file_sha256(folder / "candidate.pt")}
    write_json(folder / "result.json", result)
    del optimizer, pixels, final_pixels, target, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def public_histogram(tables: Mapping[str, CellTable]) -> dict[str, Any]:
    result = {}
    for name in FC_NAMES:
        q = tables[name].q_cpu
        counts = torch.bincount((q.to(torch.int64) + 7).flatten(), minlength=15)
        result[name] = {"scale": float(tables[name].scale_cpu),
                        "counts_by_code": {str(code): int(counts[code + 7]) for code in range(-7, 8)},
                        "positive_count": int((q > 0).sum()), "negative_count": int((q < 0).sum()),
                        "zero_count": int((q == 0).sum()), "numel": q.numel(),
                        "nonzero_fraction": float((q != 0).float().mean())}
    return result


def run_screen(*, public_observation: Path, naive: Path, checkpoint: Path,
               output: Path, device_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    clock = Clock(started)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing preexisting signed E2E output: {output}")
    if file_sha256(checkpoint) != CHECKPOINT_SHA:
        raise ExperimentError("checkpoint hash does not match signed observation")
    manifest, tables = load_public_cell_tables(public_observation)
    input_hashes = hash_public_inputs(public_observation, naive, checkpoint)
    protected_before = hash_files(protected_paths())
    x0, source, target = load_public_start(naive)
    configure_runtime(42, threads=1)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ExperimentError("the authorized real phase requires CUDA")
    device_tables = _device_tables(tables, device)
    output.mkdir(parents=True, exist_ok=False)
    registration = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "REGISTERED",
        "registered_before_optimization": True, "registered_before_private_evaluation": True,
        "public_observation": str(public_observation), "naive": str(naive),
        "checkpoint": str(checkpoint), "checkpoint_sha256": CHECKPOINT_SHA,
        "public_manifest_sha256": file_sha256(public_observation / "manifest.json"),
        "input_hashes": input_hashes, "protected_source_hashes_before": protected_before,
        "source_and_target_hashes": {"x0_pixels": tensor_sha256(x0), "S_hat": tensor_sha256(source),
                                      "target_z": tensor_sha256(target)},
        "fixed_training": TRAINING, "fixed_codec": manifest["codec_metadata"],
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "device": torch.cuda.get_device_name(device), "device_name": device_name,
                     "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                     "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                     "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                     "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                     "cudnn_benchmark": torch.backends.cudnn.benchmark,
                     "cudnn_deterministic": torch.backends.cudnn.deterministic,
                     "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                     "threads": torch.get_num_threads()},
        "screen": {"image_steps": IMAGE_STEPS, "image_lr": IMAGE_LR,
                    "image_lr_final": IMAGE_LR_FINAL, "optimizer": "Adam",
                    "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.0,
                    "tv_weight": TV_WEIGHT, "cell_chunk_size": CELL_CHUNK,
                    "max_seconds": MAX_SECONDS, "max_cuda_allocated_bytes": MAX_CUDA_ALLOCATED,
                    "pixel_space_variable": True, "normalization_once": True,
                    "tv_space": "normalized_u=(x-mean)/std"},
        "label_policy": {"private_labels_used": False, "hypotheses_frozen_before_optimization": True},
        "public_histogram": public_histogram(tables),
    }
    model = None
    try:
        write_json(output / "registration.json", registration)
        with clock.stage("model_and_public_initialization"):
            model = load_model(checkpoint, device)
            with torch.no_grad():
                # _feature_forward accepts pixels and normalizes exactly once.
                z0 = _feature_forward(model, x0.to(device))
                logits = model.classifier(z0)
            bias_q = tables["classifier.6.bias"].q_cpu
            label_evidence, labels = labels_from_public_logits(logits.cpu(), bias_q)
            write_json(output / "label_hypotheses.json", label_evidence)
            if label_evidence.get("status") != "COMPLETED":
                raise ExperimentError(label_evidence["status"])
            with torch.no_grad():
                target_device = target.to(device)
                _ = cosine_loss(z0, target_device)
            del z0, logits, target_device
        registration["label_evidence_sha256"] = file_sha256(output / "label_hypotheses.json")
        write_json(output / "registration.json", registration)
        clock.check("before_full_size_check")
        with clock.stage("full_size_numeric_check"):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            numeric = compare_replay_paths(model, x0.to(device), labels["H0"].to(device),
                                           tables, device_tables, device)
            numeric["resource"] = cuda_resource(device, "full_size_numeric_check")
            write_json(output / "full_size_numeric_check.json", numeric)
        with clock.stage("initial_calibration"):
            calibration = initial_calibration(model, x0, target, labels, tables, device_tables, device, clock)
            write_json(output / "calibration.json", calibration)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        # Endpoint profiles are public-only canonical replay checks.  The same
        # unique hypotheses are evaluated at x0 and after each final arm.
        unique_labels = {name: value for name, value in labels.items()}
        if torch.equal(unique_labels.get("H0"), unique_labels.get("H1")):
            unique_labels.pop("H1", None)
        endpoint_profiles = {"x0": {name: canonical_profile(model, x0.to(device), value.to(device), tables, device)
                                     for name, value in unique_labels.items()}}
        write_json(output / "endpoint_profiles.json", endpoint_profiles)
        weights = calibration["weights"]
        arms_plan = [("A_FIA", None, 0.0), ("B_H0_low", "H0", weights["lambda_low"]),
                     ("B_H0_high", "H0", weights["lambda_high"])]
        if not torch.equal(labels["H0"], labels["H1"]):
            arms_plan += [("B_H1_low", "H1", weights["lambda_low"]),
                          ("B_H1_high", "H1", weights["lambda_high"])]
        arm_results = {}
        for arm_name, label_name, lambda_cell in arms_plan:
            clock.check(f"before_arm:{arm_name}")
            label = None if label_name is None else labels[label_name]
            result = run_arm(model, arm_name, x0, target, label, lambda_cell,
                             tables, device_tables, output, device, clock)
            arm_results[arm_name] = result
            final_pixels = torch.load(output / arm_name / "candidate.pt", map_location=device,
                                      weights_only=True)["pixels"].to(device)
            endpoint_profiles[arm_name] = {
                name: canonical_profile(model, final_pixels, value.to(device), tables, device)
                for name, value in unique_labels.items()}
            write_json(output / "endpoint_profiles.json", endpoint_profiles)
            del final_pixels
            if device.type == "cuda":
                torch.cuda.empty_cache()
        protected_after = hash_files(protected_paths())
        input_after = hash_public_inputs(public_observation, naive, checkpoint)
        integrity = {"protected_source_hashes_before": protected_before,
                     "protected_source_hashes_after": protected_after,
                     "protected_sources_unchanged": protected_before == protected_after,
                     "input_hashes_before": input_hashes, "input_hashes_after": input_after,
                     "inputs_unchanged": input_hashes == input_after}
        write_json(output / "integrity.json", integrity)
        if not integrity["protected_sources_unchanged"] or not integrity["inputs_unchanged"]:
            raise ExperimentError("protected source or public input integrity changed")
        summary = {"task_id": TASK_ID, "status": "COMPLETED", "stage": "PUBLIC_OPTIMIZATION_COMPLETE",
                   "clock": {"total_wall_seconds": clock.seconds(), "stages": clock.stages},
                   "label_hypotheses": label_evidence, "calibration": calibration,
                   "arms_plan": [{"arm": a, "label": l, "lambda_cell": w} for a, l, w in arms_plan],
                   "arms": arm_results, "endpoint_profiles": endpoint_profiles,
                   "replay_counts": {"optimization_reverse_gradient_arms": sum(1 for a, _, _ in arms_plan if a != "A_FIA") * IMAGE_STEPS,
                                      "endpoint_full_batch_per_hypothesis": 1 + len(arms_plan),
                                      "endpoint_hypotheses": len(unique_labels)},
                   "private_evaluation": "PENDING_SEPARATE_AUTHORIZED_EVALUATOR",
                   "resource_rules": {"max_seconds": MAX_SECONDS, "max_cuda_allocated_bytes": MAX_CUDA_ALLOCATED}}
        write_json(output / "summary.json", summary)
        return summary
    except Exception as error:
        failure = {"task_id": TASK_ID, "status": "FAILED", "stage": "PUBLIC_RUNNER",
                   "error_type": type(error).__name__, "error": str(error),
                   "completed_wall_seconds": clock.seconds(), "stages": clock.stages}
        write_json(output / "failure.json", failure)
        write_json(output / "summary.json", failure)
        raise
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-observation", type=Path, required=True)
    parser.add_argument("--naive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_screen(public_observation=args.public_observation.resolve(),
                            naive=args.naive.resolve(), checkpoint=args.checkpoint.resolve(),
                            output=args.output, device_name=args.device)
    except Exception as error:
        print(f"{TASK_ID}: {type(error).__name__}: {error}", flush=True)
        return 1
    print(json.dumps({"task_id": TASK_ID, "status": result["status"],
                      "wall_seconds": result["clock"]["total_wall_seconds"]}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
