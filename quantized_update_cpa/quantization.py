"""Quantization primitives used by the 01--05 CPA experiments.

The implementation deliberately models a signed symmetric *15-level* int4
format: codes ``-7, ..., 7``.  It stores codes separately from dequantized
values so the attack never has to infer a scale from an already quantized
tensor.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Optional, Tuple

import torch


SUPPORTED_ROUNDING = {"nearest"}


def require_nearest_rounding(rounding: str) -> None:
    """Reject unsupported rounding explicitly instead of silently approximating it."""
    if rounding not in SUPPORTED_ROUNDING:
        raise ValueError(
            f"Only nearest rounding is supported by the corrected experiments; "
            f"got rounding={rounding!r}."
        )


def qrange(bits: int) -> Tuple[int, int]:
    if bits < 2:
        raise ValueError("bits must be >= 2 for signed symmetric quantization")
    qmax = (2 ** (bits - 1)) - 1
    return -qmax, qmax


def quantize_tensor_symmetric(
    tensor: torch.Tensor,
    *,
    bits: int = 4,
    rounding: str = "nearest",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Return integer codes, dequantization, scale and per-tensor statistics."""
    require_nearest_rounding(rounding)
    qmin, qmax = qrange(bits)
    max_abs = tensor.detach().abs().max()
    scale = max_abs / qmax if max_abs > 0 else torch.ones((), device=tensor.device)
    codes = torch.round(tensor / scale).clamp(qmin, qmax).to(torch.int8)
    dequantized = codes.to(dtype=tensor.dtype) * scale

    error = dequantized - tensor
    tensor_norm = torch.linalg.vector_norm(tensor.detach().reshape(-1))
    error_norm = torch.linalg.vector_norm(error.detach().reshape(-1))
    stats = {
        "mse": float(torch.mean(error.detach() ** 2).cpu()),
        "relative_l2": float(
            (error_norm / tensor_norm.clamp_min(torch.finfo(tensor.dtype).eps)).cpu()
        ),
        "scale": float(scale.detach().cpu()),
        "saturation_ratio": float((codes.abs() >= qmax).float().mean().cpu()),
        "qmin": qmin,
        "qmax": qmax,
    }
    return codes, dequantized.to(dtype=tensor.dtype), scale, stats


def quantize_update_symmetric(
    *,
    update: Dict[str, torch.Tensor],
    bits: int = 4,
    rounding: str = "nearest",
) -> Tuple[
    "OrderedDict[str, torch.Tensor]",
    "OrderedDict[str, torch.Tensor]",
    Dict[str, object],
]:
    """Quantize an ordered update dictionary with one scale per parameter tensor.

    Returns ``(codes, dequantized, quant_stats)``.  Codes are int8 tensors;
    the bit-width remains part of the metadata rather than the storage dtype.
    """
    require_nearest_rounding(rounding)
    codes_by_name: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    dequantized_by_name: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    per_tensor = []
    originals, dequantized = [], []

    for name, tensor in update.items():
        codes, dq, _scale, stats = quantize_tensor_symmetric(
            tensor, bits=bits, rounding=rounding
        )
        codes_by_name[name] = codes
        dequantized_by_name[name] = dq
        per_tensor.append({"name": name, **stats})
        originals.append(tensor.detach().reshape(-1).float())
        dequantized.append(dq.detach().reshape(-1).float())

    original_flat = torch.cat(originals)
    dq_flat = torch.cat(dequantized)
    error = dq_flat - original_flat
    qmin, qmax = qrange(bits)
    stats: Dict[str, object] = {
        "quant_bits": bits,
        "rounding": rounding,
        "zero_point": None,
        "qmin": qmin,
        "qmax": qmax,
        "levels": qmax - qmin + 1,
        "quant_mse": float(torch.mean(error**2).cpu()),
        "quant_relative_l2": float(
            (torch.linalg.vector_norm(error) / torch.linalg.vector_norm(original_flat).clamp_min(1e-12)).cpu()
        ),
        "quant_cosine_similarity": float(
            torch.nn.functional.cosine_similarity(original_flat, dq_flat, dim=0).cpu()
        ),
        "quant_scale_mean": sum(float(row["scale"]) for row in per_tensor) / len(per_tensor),
        "quant_scale_min": min(float(row["scale"]) for row in per_tensor),
        "quant_scale_max": max(float(row["scale"]) for row in per_tensor),
        "quant_saturation_ratio": sum(float(row["saturation_ratio"]) for row in per_tensor)
        / len(per_tensor),
        "per_tensor": per_tensor,
    }
    return codes_by_name, dequantized_by_name, stats


def bin_distance_squared(
    predicted: torch.Tensor,
    codes: torch.Tensor,
    scale: torch.Tensor | float,
    *,
    bits: int = 4,
) -> torch.Tensor:
    """Squared distance to nearest-rounding quantization bins.

    Interior codes use closed finite bins.  Saturated codes are one-sided:
    ``qmax -> [ (qmax-.5)s, +inf )`` and
    ``qmin -> ( -inf, (qmin+.5)s ]``.
    """
    _qmin, qmax = qrange(bits)
    qmin = -qmax
    scale_t = torch.as_tensor(scale, dtype=predicted.dtype, device=predicted.device)
    code_t = codes.to(device=predicted.device, dtype=predicted.dtype)
    lower = (code_t - 0.5) * scale_t
    upper = (code_t + 0.5) * scale_t

    below = torch.relu(lower - predicted)
    above = torch.relu(predicted - upper)
    distance = below + above
    # Saturation bins have only one finite boundary.
    distance = torch.where(code_t >= qmax, torch.relu(lower - predicted), distance)
    distance = torch.where(code_t <= qmin, torch.relu(predicted - upper), distance)
    return distance.square()


def randomized_dequantize_tensor(
    codes: torch.Tensor,
    scale: torch.Tensor | float,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Heuristic uniform-bin dequantization.

    Interior codes receive ``U(-s/2, s/2)``.  Saturated values are kept at
    ``q*s`` because the one-sided tails are unbounded without an additional
    update prior.  This is intentionally not posterior sampling.
    """
    qmin, qmax = qrange(4)
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=codes.device)
    base = codes.to(dtype=torch.float32) * scale_t
    noise = torch.empty_like(base).uniform_(-0.5, 0.5, generator=generator) * scale_t
    interior = (codes > qmin) & (codes < qmax)
    return base + noise * interior.to(dtype=base.dtype)


def quantize_dequantize_update(
    *,
    update: Dict[str, torch.Tensor],
    bits: int,
    rounding: str = "nearest",
    generator: Optional[torch.Generator] = None,
) -> Tuple["OrderedDict[str, torch.Tensor]", Dict[str, object]]:
    """Backward-compatible wrapper retained for older utility callers."""
    del generator
    _codes, dequantized, stats = quantize_update_symmetric(
        update=update, bits=bits, rounding=rounding
    )
    return dequantized, stats


def scales_from_stats(quant_stats: Dict[str, object]) -> Iterable[float]:
    """Yield scales in the same parameter order used by the stored update."""
    for row in quant_stats.get("per_tensor", []):
        yield float(row["scale"])
