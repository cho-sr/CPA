"""Conditional public-packet observations for all single-client parameters.

This module requires public uint4 packets *including their FP32 row radii*.
It does not establish that those radii are public in a particular experiment.
Only the sender-side ``encode_updates`` accepts unquantized reference updates.
The observer and losses receive packet bytes and candidate updates only.

For stochastic rounding a code q has support closure [q-1, q+1] intersect
[0, 15], in coordinates t=(x/r+1)*7.5. This is a full-step interval, with
zero-probability endpoints when |t-q|=1. Closure fit is therefore weaker than
positive marginal rounding probability, and neither proves a joint RNG match.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import math
import operator

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .communication_codec import CODEC_ID, HEADER_BYTES, encode_uint4, packet_info


ZERO_SCALE_NORMALIZATION = 1.0  # Fixed update units; never estimated from truth.
DEFAULT_TOLERANCE_STEPS = 1e-6


@dataclass(frozen=True)
class PackedObservation:
    """Validated original shape, CPU uint8 code matrix and CPU FP32 row radius."""

    shape: tuple[int, ...]
    codes: torch.Tensor
    radius: torch.Tensor


def _shape(value) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError("shape must be a nonempty list/tuple of positive integers")
    if any(type(size) is not int or size <= 0 for size in value):
        raise ValueError("shape must contain positive integers, not booleans")
    return tuple(value)


def _matrix_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    # A bias vector shares one radius. A row per scalar would reveal |bias|.
    return (1, shape[0]) if len(shape) == 1 else (shape[0], math.prod(shape[1:]))


def _mapping(value, description: str):
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{description} must be a nonempty mapping")
    if any(not isinstance(name, str) or not name for name in value):
        raise ValueError(f"{description} names must be nonempty strings")


def _tensor(value, name: str):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.layout != torch.strided or not value.is_floating_point():
        raise ValueError(f"{name} must be a strided real floating tensor")
    _shape(list(value.shape))
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} must contain only finite values")


def encode_updates(updates: Mapping[str, torch.Tensor], seed: int) -> dict[str, dict]:
    """Sender only: encode FP32 updates without returning private residuals.

    Tensors use insertion order and seeds (seed + tensor_index) modulo 2**64.
    The seed is provenance, not observer input. Existing codec behavior is
    unchanged. Weights flatten all but their output dimension; biases are 1xN.
    """
    _mapping(updates, "updates")
    if isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    seed = operator.index(seed)
    if not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("seed must be in the uint64 range")
    entries = {}
    for index, (name, update) in enumerate(updates.items()):
        _tensor(update, name)
        if update.dtype != torch.float32:
            raise ValueError(f"{name}: sender update must be FP32; no implicit conversion")
        shape = tuple(update.shape)
        matrix = update.detach().to(device="cpu").reshape(_matrix_shape(shape))
        entries[name] = {"shape": list(shape),
                         "packet": encode_uint4(matrix, (seed + index) % (1 << 64))}
    return entries


def load_observations(entries: Mapping[str, dict]) -> OrderedDict[str, PackedObservation]:
    """Parse codes and transmitted radii directly, without decoded-center inference.

    The codec validates all wire bytes (including padding/zero-row canonicality)
    before allocation. Unknown entry fields are rejected so truth cannot be
    accidentally bundled into this observer-facing interface.
    """
    _mapping(entries, "entries")
    observations = OrderedDict()
    for name, entry in entries.items():
        if not isinstance(entry, Mapping) or set(entry) != {"shape", "packet"}:
            raise ValueError(f"{name}: entry must contain exactly shape and packet")
        shape = _shape(entry["shape"])
        packet = entry["packet"]
        info = packet_info(packet)
        if info["codec_id"] != CODEC_ID or info["bits"] != 4:
            raise ValueError(f"{name}: only the stochastic uint4 codec is supported")
        rows, cols = _matrix_shape(shape)
        if info["shape"] != [rows, cols]:
            raise ValueError(f"{name}: packet shape disagrees with canonical parameter layout")
        radius = np.frombuffer(packet, dtype="<f4", count=rows, offset=HEADER_BYTES).copy()
        packed = np.frombuffer(packet, dtype=np.uint8, offset=HEADER_BYTES + rows * 4)
        codes = np.empty(rows * cols, dtype=np.uint8)
        codes[0::2] = packed & 0x0F
        codes[1::2] = packed[:codes.size // 2] >> 4
        code_rows = codes.reshape(rows, cols)
        if np.any((radius > 0) & ~np.any((code_rows == 0) | (code_rows == 15), axis=1)):
            raise ValueError(f"{name}: positive-radius row must contain a radius endpoint code")
        observations[name] = PackedObservation(shape, torch.from_numpy(codes.reshape(rows, cols)),
                                               torch.from_numpy(radius))
    return observations


def _validate_pair(updates, observations, chunk_rows):
    _mapping(updates, "updates")
    _mapping(observations, "observations")
    if set(updates) != set(observations):
        raise ValueError("update and observation names must match exactly")
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    device = None
    for name, observation in observations.items():
        update = updates[name]
        _tensor(update, name)
        if not isinstance(observation, PackedObservation):
            raise TypeError(f"{name}: expected PackedObservation from load_observations")
        if tuple(update.shape) != observation.shape:
            raise ValueError(f"{name}: update shape must match observation exactly")
        rows, cols = _matrix_shape(_shape(observation.shape))
        codes, radius = observation.codes, observation.radius
        if (codes.dtype != torch.uint8 or tuple(codes.shape) != (rows, cols)
                or radius.dtype != torch.float32 or tuple(radius.shape) != (rows,)
                or codes.device.type != "cpu" or radius.device.type != "cpu"):
            raise ValueError(f"{name}: invalid observation tensor layout")
        if (bool((codes > 15).any()) or not bool(torch.isfinite(radius).all())
                or bool(torch.signbit(radius).any()) or bool((codes[radius == 0] != 0).any())):
            raise ValueError(f"{name}: invalid observation codes or radius")
        if bool(((radius > 0) & ~((codes == 0) | (codes == 15)).any(dim=1)).any()):
            raise ValueError(f"{name}: positive-radius row must contain a radius endpoint code")
        if device is not None and update.device != device:
            raise ValueError("candidate updates must share a device")
        device = update.device


def _chunk_values(matrix_chunk, observed_radius, observed_codes):
    values = matrix_chunk.double()
    radius = observed_radius.detach().to(device=matrix_chunk.device, dtype=torch.float64)
    codes = observed_codes.to(device=matrix_chunk.device, dtype=torch.float64)
    nonzero = radius > 0
    safe_radius = torch.where(nonzero, radius, torch.ones_like(radius))
    # Algebraically x/h + 7.5, but no FP32 h underflow for subnormal radii.
    coordinate = (values / safe_radius[:, None] + 1.0) * 7.5
    lower, upper = (codes - 1.0).clamp_min(0.0), (codes + 1.0).clamp_max(15.0)
    violation = torch.maximum(lower - coordinate, coordinate - upper).clamp_min(0.0)
    radius_delta = values.abs().amax(dim=1) - radius
    radius_steps = radius_delta / safe_radius * 7.5
    return values, radius, codes, nonzero, coordinate, violation, radius_delta, radius_steps


def _chunks(update, observation, chunk_rows):
    matrix = update.reshape(_matrix_shape(observation.shape))
    for start in range(0, matrix.shape[0], chunk_rows):
        stop = min(start + chunk_rows, matrix.shape[0])
        yield _chunk_values(matrix[start:stop], observation.radius[start:stop],
                            observation.codes[start:stop])


def _chunk_loss(matrix_chunk, observed_radius, observed_codes, mode):
    values, _, codes, nonzero, coordinate, violation, _, radius_steps = _chunk_values(
        matrix_chunk, observed_radius, observed_codes)
    error = violation if mode == "cell" else coordinate - codes
    value_sum = torch.where(nonzero[:, None], error.square(),
                            values.abs() / ZERO_SCALE_NORMALIZATION).sum()
    radius_sum = torch.where(nonzero, radius_steps.square(),
                             values.abs().amax(dim=1) / ZERO_SCALE_NORMALIZATION).sum()
    return value_sum, radius_sum


def observation_loss(updates: Mapping[str, torch.Tensor],
                     observations: Mapping[str, PackedObservation], *, mode: str = "cell",
                     radius_weight: float = 1.0, chunk_rows: int = 64) -> torch.Tensor:
    """Equal mean of tensor losses, using the fixed observed step h=2r/15.

    Positive-radius rows use mean squared interval violation (or center error)
    plus ``radius_weight`` times mean squared row-maximum radius error, both in
    observed-step units. Zero-radius rows instead use absolute value / 1.0
    update units, and absolute row maximum / 1.0 for the radius term. The fixed
    zero normalization never uses candidate scale or private reference values.

    Double arithmetic preserves subnormal coordinates and differentiability.
    Non-reentrant checkpoints recompute each whole-row loss chunk on backward,
    keeping its FP64 intermediate arrays out of the retained graph. Candidate
    updates and the replay graph still require their own memory budget.
    """
    if mode not in ("cell", "center"):
        raise ValueError("mode must be 'cell' or 'center'")
    if not math.isfinite(radius_weight) or radius_weight < 0:
        raise ValueError("radius_weight must be finite and nonnegative")
    _validate_pair(updates, observations, chunk_rows)
    losses = []
    for name, observation in observations.items():
        update = updates[name]
        matrix = update.reshape(_matrix_shape(observation.shape))
        value_sum = update.new_zeros((), dtype=torch.float64)
        radius_sum = update.new_zeros((), dtype=torch.float64)
        for start in range(0, matrix.shape[0], chunk_rows):
            stop = min(start + chunk_rows, matrix.shape[0])
            args = (matrix[start:stop], observation.radius[start:stop],
                    observation.codes[start:stop], mode)
            if torch.is_grad_enabled() and matrix.requires_grad:
                value_chunk, radius_chunk = checkpoint(_chunk_loss, *args,
                                                       use_reentrant=False, preserve_rng_state=False)
            else:
                value_chunk, radius_chunk = _chunk_loss(*args)
            value_sum = value_sum + value_chunk
            radius_sum = radius_sum + radius_chunk
        rows, _ = _matrix_shape(observation.shape)
        losses.append(value_sum / update.numel() + radius_weight * radius_sum / rows)
    return torch.stack(losses).mean()


@torch.no_grad()
def summarize_observation_fit(updates: Mapping[str, torch.Tensor],
                              observations: Mapping[str, PackedObservation], *,
                              tolerance_steps: float = DEFAULT_TOLERANCE_STEPS,
                              chunk_rows: int = 64) -> dict:
    """JSON diagnostics with a fixed tolerance, separate from the training loss.

    For positive radii the tolerance is in observed-step units. Zero radii have
    no quantization step: their separate absolute threshold is numerically
    ``tolerance_steps * ZERO_SCALE_NORMALIZATION`` update units, not a fraction
    of a nonexistent step. It is returned as ``zero_scale_tolerance_abs``.
    Exact and near zero-probability boundary counts are separate; no fixed-seed
    or joint-PRNG claim is made.
    """
    if not math.isfinite(tolerance_steps) or tolerance_steps < 0:
        raise ValueError("tolerance_steps must be finite and nonnegative")
    _validate_pair(updates, observations, chunk_rows)
    per_tensor = {}
    for name, observation in observations.items():
        stats = {"positive_radius_elements": 0, "zero_radius_elements": 0,
                 "positive_radius_rows": 0, "bin_violation_count": 0,
                 "max_bin_violation_steps": 0.0, "bin_violation_sum_steps": 0.0,
                 "max_radius_error_steps": 0.0, "radius_error_sum_steps": 0.0,
                 "max_radius_error_abs": 0.0, "max_zero_scale_error_abs": 0.0,
                 "zero_scale_error_sum_abs": 0.0,
                 "zero_probability_boundary_count": 0,
                 "near_zero_probability_boundary_count": 0,
                 "zero_marginal_probability_count": 0}
        for values, radius, codes, nonzero, coordinate, violation, radius_delta, radius_steps in _chunks(
                updates[name], observation, chunk_rows):
            active = violation[nonzero]
            zero_error = values[~nonzero].abs()
            radius_error = radius_steps[nonzero].abs()
            stats["positive_radius_elements"] += active.numel()
            stats["zero_radius_elements"] += zero_error.numel()
            stats["positive_radius_rows"] += radius_error.numel()
            stats["max_radius_error_abs"] = max(stats["max_radius_error_abs"],
                                                  float(radius_delta.abs().max()))
            if active.numel():
                distance = (coordinate[nonzero] - codes[nonzero]).abs()
                stats["max_bin_violation_steps"] = max(stats["max_bin_violation_steps"], float(active.max()))
                stats["bin_violation_sum_steps"] += float(active.sum())
                stats["bin_violation_count"] += int((active > tolerance_steps).sum())
                stats["max_radius_error_steps"] = max(stats["max_radius_error_steps"], float(radius_error.max()))
                stats["radius_error_sum_steps"] += float(radius_error.sum())
                stats["zero_probability_boundary_count"] += int(((active == 0) & (distance >= 1.0)).sum())
                stats["near_zero_probability_boundary_count"] += int(
                    ((active <= tolerance_steps) & (distance >= 1.0 - tolerance_steps)).sum())
                stats["zero_marginal_probability_count"] += int(((distance >= 1.0) | (active > 0)).sum())
            if zero_error.numel():
                stats["max_zero_scale_error_abs"] = max(stats["max_zero_scale_error_abs"], float(zero_error.max()))
                stats["zero_scale_error_sum_abs"] += float(zero_error.sum())
        stats["mean_bin_violation_steps"] = stats.pop("bin_violation_sum_steps") / max(1, stats["positive_radius_elements"])
        stats["bin_violation_rate"] = stats["bin_violation_count"] / max(1, stats["positive_radius_elements"])
        stats["mean_radius_error_steps"] = stats.pop("radius_error_sum_steps") / max(1, stats["positive_radius_rows"])
        stats["mean_zero_scale_error_abs"] = stats.pop("zero_scale_error_sum_abs") / max(1, stats["zero_radius_elements"])
        stats["closure_fit_within_tolerance"] = (
            stats["max_bin_violation_steps"] <= tolerance_steps
            and stats["max_radius_error_steps"] <= tolerance_steps
            and stats["max_zero_scale_error_abs"] <= tolerance_steps * ZERO_SCALE_NORMALIZATION)
        per_tensor[name] = stats
    all_stats = list(per_tensor.values())
    return {"tolerance_steps": tolerance_steps,
            "zero_scale_normalization_update_units": ZERO_SCALE_NORMALIZATION,
            "zero_scale_tolerance_abs": tolerance_steps * ZERO_SCALE_NORMALIZATION,
            "closure_fit_within_tolerance": all(s["closure_fit_within_tolerance"] for s in all_stats),
            "max_bin_violation_steps": max(s["max_bin_violation_steps"] for s in all_stats),
            "mean_bin_violation_steps": sum(s["mean_bin_violation_steps"] for s in all_stats) / len(all_stats),
            "mean_tensor_bin_violation_rate": sum(s["bin_violation_rate"] for s in all_stats) / len(all_stats),
            "bin_violation_rate": sum(s["bin_violation_count"] for s in all_stats)
            / max(1, sum(s["positive_radius_elements"] for s in all_stats)),
            "max_radius_error_steps": max(s["max_radius_error_steps"] for s in all_stats),
            "mean_radius_error_steps": sum(s["mean_radius_error_steps"] for s in all_stats) / len(all_stats),
            "max_radius_error_abs": max(s["max_radius_error_abs"] for s in all_stats),
            "max_zero_scale_error_abs": max(s["max_zero_scale_error_abs"] for s in all_stats),
            "mean_zero_scale_error_abs": sum(s["mean_zero_scale_error_abs"] for s in all_stats) / len(all_stats),
            "zero_probability_boundary_count": sum(s["zero_probability_boundary_count"] for s in all_stats),
            "near_zero_probability_boundary_count": sum(s["near_zero_probability_boundary_count"] for s in all_stats),
            "zero_marginal_probability_count": sum(s["zero_marginal_probability_count"] for s in all_stats),
            "exact_joint_rng_match": "not_checked",
            "probability_interpretation": "ideal coordinatewise stochastic rounding at observed radius; no joint seed claim",
            "mean_aggregation": "equal tensor means",
            "per_tensor": per_tensor}
