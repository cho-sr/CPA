"""Fake quantization helpers for FedAvg update CPA experiments."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch


def _quantize_tensor_symmetric(
    tensor: torch.Tensor,
    *,
    bits: int,
    rounding: str,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if bits < 2:
        raise ValueError("bits must be >= 2 for signed symmetric quantization")
    if rounding not in {"nearest", "stochastic"}:
        raise ValueError("rounding must be 'nearest' or 'stochastic'")

    qmax = (2 ** (bits - 1)) - 1
    qmin = -qmax
    max_abs = tensor.detach().abs().max()
    scale = max_abs / qmax if max_abs > 0 else torch.tensor(1.0, device=tensor.device)

    scaled = tensor / scale
    if rounding == "nearest":
        quantized = torch.round(scaled)
    else:
        lower = torch.floor(scaled)
        prob = scaled - lower
        rand = torch.rand(
            prob.shape,
            dtype=prob.dtype,
            device=prob.device,
            generator=generator,
        )
        quantized = lower + (rand < prob).to(dtype=prob.dtype)

    quantized = quantized.clamp(qmin, qmax)
    dequantized = quantized * scale

    error = dequantized - tensor
    tensor_norm = torch.linalg.vector_norm(tensor.detach().reshape(-1))
    error_norm = torch.linalg.vector_norm(error.detach().reshape(-1))
    relative_l2 = error_norm / tensor_norm.clamp_min(torch.finfo(tensor.dtype).eps)
    mse = torch.mean(error.detach() ** 2)
    saturation = (quantized.abs() >= qmax).to(dtype=torch.float32).mean()

    return dequantized.to(dtype=tensor.dtype), {
        "mse": float(mse.detach().cpu()),
        "relative_l2": float(relative_l2.detach().cpu()),
        "scale": float(scale.detach().cpu()),
        "saturation_ratio": float(saturation.detach().cpu()),
    }


def quantize_dequantize_update(
    *,
    update: Dict[str, torch.Tensor],
    bits: int,
    rounding: str = "nearest",
    generator: Optional[torch.Generator] = None,
) -> Tuple["OrderedDict[str, torch.Tensor]", Dict[str, float]]:
    """Apply per-tensor signed symmetric fake quantization to a model update."""
    quantized_update: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    stats = []

    flat_original = []
    flat_quantized = []
    for name, tensor in update.items():
        quantized, tensor_stats = _quantize_tensor_symmetric(
            tensor,
            bits=bits,
            rounding=rounding,
            generator=generator,
        )
        quantized_update[name] = quantized
        stats.append(tensor_stats)
        flat_original.append(tensor.detach().reshape(-1).float())
        flat_quantized.append(quantized.detach().reshape(-1).float())

    original = torch.cat(flat_original)
    quantized = torch.cat(flat_quantized)
    error = quantized - original
    original_norm = torch.linalg.vector_norm(original)
    error_norm = torch.linalg.vector_norm(error)

    quant_stats = {
        "quant_bits": bits,
        "quant_mse": float(torch.mean(error ** 2).cpu()),
        "quant_relative_l2": float(
            (error_norm / original_norm.clamp_min(torch.finfo(original.dtype).eps)).cpu()
        ),
        "quant_cosine_similarity": float(
            torch.nn.functional.cosine_similarity(original, quantized, dim=0).cpu()
        ),
        "quant_scale_mean": sum(s["scale"] for s in stats) / len(stats),
        "quant_scale_min": min(s["scale"] for s in stats),
        "quant_scale_max": max(s["scale"] for s in stats),
        "quant_saturation_ratio": sum(s["saturation_ratio"] for s in stats) / len(stats),
    }
    return quantized_update, quant_stats
