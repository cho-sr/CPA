"""Fake quantization helpers for FedAvg client weight updates."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


EPS = 1e-12
SCALE_METADATA_BITS = 32


def _stats_from_tensors(
    tensor: torch.Tensor,
    dequantized: torch.Tensor,
    saturation_count: int,
    scale: float,
) -> Dict[str, float]:
    fp32 = tensor.detach().to(dtype=torch.float32)
    dq = dequantized.detach().to(dtype=torch.float32)
    error = fp32 - dq
    return {
        "numel": float(fp32.numel()),
        "sq_error_sum": float(torch.sum(error * error).item()),
        "original_norm_sq": float(torch.sum(fp32 * fp32).item()),
        "dequant_norm_sq": float(torch.sum(dq * dq).item()),
        "dot_sum": float(torch.sum(fp32 * dq).item()),
        "saturation_count": float(saturation_count),
        "scale_sum": float(scale),
        "scale_min": float(scale),
        "scale_max": float(scale),
        "scale_count": 1.0,
    }


def quantize_dequantize_tensor(
    tensor: torch.Tensor,
    bits: int,
    rounding: str,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return a float32 fake-quantized tensor and element-level statistics."""
    if bits not in {32, 8, 4}:
        raise ValueError(f"Unsupported quantization bits: {bits}")
    if rounding not in {"nearest", "stochastic"}:
        raise ValueError(f"Unsupported rounding mode: {rounding}")

    fp32 = tensor.detach().to(dtype=torch.float32)
    if bits == 32:
        dequantized = fp32.clone()
        return dequantized, _stats_from_tensors(fp32, dequantized, saturation_count=0, scale=0.0)

    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax
    max_abs = torch.max(torch.abs(fp32)).item() if fp32.numel() > 0 else 0.0
    if max_abs == 0.0:
        dequantized = torch.zeros_like(fp32, dtype=torch.float32)
        return dequantized, _stats_from_tensors(fp32, dequantized, saturation_count=0, scale=0.0)

    scale = float(max_abs / qmax)
    scaled = fp32 / scale
    if rounding == "nearest":
        q_unclamped = torch.round(scaled)
    else:
        lower = torch.floor(scaled)
        probability = torch.clamp(scaled - lower, min=0.0, max=1.0)
        bernoulli = torch.bernoulli(probability, generator=generator)
        q_unclamped = lower + bernoulli

    saturation_count = int(((q_unclamped < qmin) | (q_unclamped > qmax)).sum().item())
    q = torch.clamp(q_unclamped, qmin, qmax)
    dequantized = (q * scale).to(dtype=torch.float32)
    return dequantized, _stats_from_tensors(fp32, dequantized, saturation_count, scale)


def summarize_quant_stats(raw: Dict[str, float]) -> Dict[str, float]:
    numel = max(raw.get("numel", 0.0), 0.0)
    original_norm = raw.get("original_norm_sq", 0.0) ** 0.5
    dequant_norm = raw.get("dequant_norm_sq", 0.0) ** 0.5
    denominator = original_norm * dequant_norm
    if raw.get("sq_error_sum", 0.0) == 0.0:
        cosine = 1.0
    elif denominator <= EPS:
        cosine = 1.0
    else:
        cosine = raw.get("dot_sum", 0.0) / (denominator + EPS)
    scale_count = raw.get("scale_count", 0.0)
    if scale_count > 0:
        scale_mean = raw.get("scale_sum", 0.0) / scale_count
        scale_min = raw.get("scale_min", 0.0)
        scale_max = raw.get("scale_max", 0.0)
    else:
        scale_mean = 0.0
        scale_min = 0.0
        scale_max = 0.0

    return {
        "quant_mse": raw.get("sq_error_sum", 0.0) / numel if numel > 0 else 0.0,
        "quant_relative_l2": (raw.get("sq_error_sum", 0.0) ** 0.5) / (original_norm + EPS),
        "quant_cosine_similarity": cosine,
        "quant_saturation_ratio": raw.get("saturation_count", 0.0) / numel if numel > 0 else 0.0,
        "quant_scale_mean": scale_mean,
        "quant_scale_min": scale_min,
        "quant_scale_max": scale_max,
    }


def merge_raw_quant_stats(accumulator: Dict[str, float], raw: Dict[str, float]) -> None:
    for key in [
        "numel",
        "sq_error_sum",
        "original_norm_sq",
        "dequant_norm_sq",
        "dot_sum",
        "saturation_count",
        "scale_sum",
        "scale_count",
    ]:
        accumulator[key] = accumulator.get(key, 0.0) + raw.get(key, 0.0)

    scale_count = raw.get("scale_count", 0.0)
    if scale_count > 0:
        raw_min = raw.get("scale_min", 0.0)
        raw_max = raw.get("scale_max", 0.0)
        if accumulator.get("scale_count", 0.0) == scale_count:
            accumulator["scale_min"] = raw_min
            accumulator["scale_max"] = raw_max
        else:
            accumulator["scale_min"] = min(accumulator.get("scale_min", raw_min), raw_min)
            accumulator["scale_max"] = max(accumulator.get("scale_max", raw_max), raw_max)


def quantize_dequantize_update(
    update: Dict[str, torch.Tensor],
    bits: int,
    rounding: str,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Fake-quantize/dequantize every tensor in a client update."""
    compressed_update: Dict[str, torch.Tensor] = {}
    raw_stats: Dict[str, float] = {}
    for name, tensor in update.items():
        dequantized, tensor_stats = quantize_dequantize_tensor(
            tensor=tensor,
            bits=bits,
            rounding=rounding,
            generator=generator,
        )
        compressed_update[name] = dequantized.to(dtype=torch.float32)
        merge_raw_quant_stats(raw_stats, tensor_stats)

    raw_stats.update(summarize_quant_stats(raw_stats))
    raw_stats["num_tensors"] = float(len(update))
    raw_stats["communication_bits"] = float(communication_bits(update, bits))
    raw_stats["communication_bytes"] = raw_stats["communication_bits"] / 8.0
    fp32_bits = float(communication_bits(update, 32))
    raw_stats["compression_ratio_vs_fp32"] = fp32_bits / raw_stats["communication_bits"] if raw_stats["communication_bits"] else 1.0
    return compressed_update, raw_stats


def communication_bits(update: Dict[str, torch.Tensor], bits: int) -> int:
    if bits not in {32, 8, 4}:
        raise ValueError(f"Unsupported quantization bits: {bits}")
    parameter_bits = sum(tensor.numel() for tensor in update.values()) * bits
    if bits == 32:
        return int(parameter_bits)
    scale_bits = len(update) * SCALE_METADATA_BITS
    return int(parameter_bits + scale_bits)
