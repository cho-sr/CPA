"""Strict deterministic uint4 packets and half-step observation constraints.

Wire codec number 2 is distinct from the historical stochastic codec number 1.
The sender uses FP32 row abs-max radii and CPU FP64 nearest-even rounding onto
16 affine levels, c_q = r * (2*q/15 - 1). The FP64 encoder coordinate is
explicitly (u + r) * 7.5 / r, followed by clamp and torch.round (nearest-even).
Only codes and public radii are sent:
there is no source-dependent correction of decoded values. Weights use output
rows and biases share one row. A zero-radius row has canonical all-zero codes.

Closed half-step cells are optimization surrogates: their midpoint ties must
also satisfy the actual nearest-even encoder. Final diagnostics report exact
reencoding and radius matching separately from tolerant closed-cell fit.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .communication_codec import HEADER, HEADER_BYTES, MAGIC, FORMAT_VERSION, ROW_CHUNK_SIZE
from .full_update_observation import PackedObservation as _BaseObservation


CODEC_ID = "uniform_affine_uint4_nearest_even_v1"
CODEC_NUMBER = 2
ZERO_SCALE_NORMALIZATION = 1.0
DEFAULT_TOLERANCE_STEPS = 1e-6


@dataclass(frozen=True)
class PackedObservation(_BaseObservation):
    """Compatible shape/codes/radius fields, tagged for deterministic losses."""

    codec_id: str = CODEC_ID


def _mapping(value, role):
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{role} must be a nonempty mapping")
    if any(not isinstance(name, str) or not name for name in value):
        raise ValueError(f"{role} names must be nonempty strings")


def _shape(shape):
    if (not isinstance(shape, (list, tuple)) or not shape
            or any(type(n) is not int or n <= 0 for n in shape)):
        raise ValueError("shape must contain positive integers")
    return tuple(shape)


def _matrix_shape(shape):
    return (1, shape[0]) if len(shape) == 1 else (shape[0], math.prod(shape[1:]))


def _tensor(value, name):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or not value.is_floating_point()):
        raise ValueError(f"{name} must be a strided real floating tensor")
    _shape(tuple(value.shape))
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} must be finite")


def _round_codes(values, radius):
    safe = torch.where(radius > 0, radius, torch.ones_like(radius)).double()
    coordinate = ((values.double() + safe[:, None]) * 7.5 / safe[:, None]).clamp(0.0, 15.0)
    codes = torch.round(coordinate).to(torch.uint8)
    return torch.where((radius > 0)[:, None], codes, torch.zeros_like(codes))


def _encode_matrix(matrix):
    rows, cols = matrix.shape
    if rows > 0xFFFFFFFF or cols > 0xFFFFFFFF:
        raise ValueError("matrix dimensions must fit uint32")
    offset = HEADER_BYTES + rows * 4
    packet = bytearray(offset + (rows * cols + 1) // 2)
    packet[:HEADER_BYTES] = HEADER.pack(MAGIC, FORMAT_VERSION, CODEC_NUMBER, 4, 0,
                                        rows, cols, rows * cols, rows)
    radii = torch.empty(rows, dtype=torch.float32)
    for start in range(0, rows, ROW_CHUNK_SIZE):
        stop = min(rows, start + ROW_CHUNK_SIZE)
        values = matrix[start:stop]
        radius = values.abs().amax(dim=1)
        radii[start:stop] = radius
        flat = _round_codes(values, radius).flatten()
        packed = flat[0::2].clone()
        packed[:flat.numel() // 2] |= flat[1::2] << 4
        first = offset + start * cols // 2
        packet[first:first + packed.numel()] = packed.numpy().tobytes()
    packet[HEADER_BYTES:offset] = radii.numpy().astype("<f4", copy=False).tobytes()
    return bytes(packet)


def encode_updates(updates: Mapping[str, torch.Tensor]) -> dict[str, dict]:
    """Sender only: sorted-name FP32 updates to shape + immutable packet bytes."""
    _mapping(updates, "updates")
    entries = {}
    for name in sorted(updates):
        value = updates[name]
        _tensor(value, name)
        if value.dtype != torch.float32:
            raise ValueError("sender updates must be FP32; no implicit rounding or conversion")
        shape = tuple(value.shape)
        matrix = value.detach().to(device="cpu").reshape(_matrix_shape(shape))
        entries[name] = {"shape": list(shape), "packet": _encode_matrix(matrix)}
    return entries


def _parse(packet):
    if not isinstance(packet, bytes):
        raise TypeError("packet must be immutable bytes")
    if len(packet) < HEADER_BYTES:
        raise ValueError("packet is shorter than its header")
    magic, version, codec, bits, flags, rows, cols, count, nr = HEADER.unpack_from(packet)
    if (magic != MAGIC or version != FORMAT_VERSION or codec != CODEC_NUMBER
            or bits != 4 or flags != 0):
        raise ValueError("expected deterministic nearest-even uint4 codec; other codecs are rejected")
    if rows == 0 or cols == 0 or count != rows * cols or nr != rows:
        raise ValueError("invalid matrix shape, element count or radius count")
    code_bytes = (count + 1) // 2
    if len(packet) != HEADER_BYTES + rows * 4 + code_bytes:
        raise ValueError("packet length does not match its shape")
    radius = np.frombuffer(packet, dtype="<f4", count=rows, offset=HEADER_BYTES)
    if not np.isfinite(radius).all() or np.signbit(radius).any():
        raise ValueError("radius must be finite canonical nonnegative FP32")
    packed = np.frombuffer(packet, dtype=np.uint8, offset=HEADER_BYTES + rows * 4)
    if count % 2 and int(packed[-1]) & 0xF0:
        raise ValueError("nonzero high-nibble padding")
    codes = np.empty(count, dtype=np.uint8)
    codes[0::2] = packed & 0x0F
    codes[1::2] = packed[:count // 2] >> 4
    codes = codes.reshape(rows, cols)
    if np.any(codes[radius == 0] != 0):
        raise ValueError("nonzero codes in a zero-radius row")
    if np.any((radius > 0) & ~np.any((codes == 0) | (codes == 15), axis=1)):
        raise ValueError("positive-radius row lacks a deterministic radius endpoint code")
    reference_bytes = HEADER_BYTES + count * 4
    info = {"codec_id": CODEC_ID, "codec_number": CODEC_NUMBER,
            "format_version": FORMAT_VERSION, "bits": 4, "rounding": "nearest_even",
            "rows": rows, "cols": cols, "shape": [rows, cols], "elements": count,
            "radius_count": rows, "header_bytes": HEADER_BYTES, "radius_bytes": rows * 4,
            "code_bytes": code_bytes, "payload_bytes": rows * 4 + code_bytes,
            "packet_bytes": len(packet), "fp32_reference_packet_bytes": reference_bytes,
            "compression_ratio_vs_fp32": reference_bytes / len(packet),
            "reduction_fraction_vs_fp32": 1.0 - len(packet) / reference_bytes}
    return info, radius, codes


def packet_info(packet: bytes) -> dict:
    """Validate deterministic bytes and return JSON-compatible communication counts."""
    return _parse(packet)[0]


def decode_packet(packet: bytes) -> torch.Tensor:
    """Decode to a fresh CPU FP32 grid matrix using packet bytes only."""
    _, radius, codes = _parse(packet)
    r = torch.from_numpy(radius.astype(np.float64, copy=True))
    q = torch.from_numpy(codes).double()
    decoded = (r[:, None] * (2.0 * q / 15.0 - 1.0)).float()
    decoded[r == 0] = 0.0
    return decoded


def load_observations(entries: Mapping[str, dict]) -> OrderedDict[str, PackedObservation]:
    """Sorted, strict observer input. No source update, residual or seed is accepted."""
    _mapping(entries, "entries")
    observations = OrderedDict()
    for name in sorted(entries):
        entry = entries[name]
        if not isinstance(entry, Mapping) or set(entry) != {"shape", "packet"}:
            raise ValueError("entries must contain exactly shape and packet")
        shape = _shape(entry["shape"])
        info, radius, codes = _parse(entry["packet"])
        if tuple(info["shape"]) != _matrix_shape(shape):
            raise ValueError(f"{name}: noncanonical parameter/packet shape")
        observations[name] = PackedObservation(shape, torch.from_numpy(codes),
                                               torch.from_numpy(radius.copy()))
    return observations


def _validate(updates, observations, chunk_rows):
    _mapping(updates, "updates")
    _mapping(observations, "observations")
    if set(updates) != set(observations):
        raise ValueError("candidate and observation names must match exactly")
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    device = None
    for name, obs in observations.items():
        _tensor(updates[name], name)
        if not isinstance(obs, PackedObservation) or obs.codec_id != CODEC_ID:
            raise ValueError("deterministic observations are required; stochastic observations are not accepted")
        if tuple(updates[name].shape) != obs.shape:
            raise ValueError(f"{name}: candidate shape must match exactly")
        rows, cols = _matrix_shape(_shape(obs.shape))
        if (obs.codes.dtype != torch.uint8 or tuple(obs.codes.shape) != (rows, cols)
                or obs.radius.dtype != torch.float32 or tuple(obs.radius.shape) != (rows,)
                or obs.codes.device.type != "cpu" or obs.radius.device.type != "cpu"):
            raise ValueError("invalid observation tensor layout")
        if (bool((obs.codes > 15).any()) or not bool(torch.isfinite(obs.radius).all())
                or bool(torch.signbit(obs.radius).any()) or bool((obs.codes[obs.radius == 0] != 0).any())):
            raise ValueError("invalid codes or radius")
        if bool(((obs.radius > 0) & ~((obs.codes == 0) | (obs.codes == 15)).any(dim=1)).any()):
            raise ValueError("positive-radius row lacks an endpoint code")
        if device is not None and updates[name].device != device:
            raise ValueError("all candidates must share a device")
        device = updates[name].device


def _chunk_values(matrix, observed_radius, observed_codes):
    values = matrix.double()
    radius = observed_radius.detach().to(device=matrix.device, dtype=torch.float64)
    codes = observed_codes.to(device=matrix.device, dtype=torch.float64)
    nonzero = radius > 0
    safe = torch.where(nonzero, radius, torch.ones_like(radius))
    coordinate = (values + safe[:, None]) * 7.5 / safe[:, None]
    lower = (codes - .5).clamp_min(0)
    upper = (codes + .5).clamp_max(15)
    violation = torch.maximum(lower - coordinate, coordinate - upper).clamp_min(0)
    radius_delta = values.abs().amax(dim=1) - radius
    return values, radius, codes, nonzero, coordinate, violation, radius_delta


def _chunk_loss(matrix, observed_radius, observed_codes, scale_floor):
    values, radius, _, nonzero, _, violation, radius_delta = _chunk_values(
        matrix, observed_radius, observed_codes)
    h = radius * (2.0 / 15.0)
    zero_scale = ZERO_SCALE_NORMALIZATION if scale_floor is None else scale_floor
    denominator = h if scale_floor is None else h.clamp_min(scale_floor)
    denominator = torch.where(nonzero, denominator, torch.full_like(h, zero_scale))
    distance = violation * h[:, None] / denominator[:, None]
    value_loss = torch.where(nonzero[:, None], distance.square(), values.abs() / zero_scale).sum()
    radius_loss = torch.where(nonzero, (radius_delta / denominator).square(),
                              values.abs().amax(dim=1) / zero_scale).sum()
    return value_loss, radius_loss


def observation_loss(updates: Mapping[str, torch.Tensor], observations: Mapping[str, PackedObservation], *,
                     radius_weight: float = 1.0, chunk_rows: int = 64,
                     scale_floor: float | None = None) -> torch.Tensor:
    """Equal tensor means of squared half-cell and row-radius errors.

    Positive rows use observed h=2r/15, or max(h, scale_floor) when a fixed
    public-only floor is explicitly provided. The floor changes objective
    conditioning, never cell boundaries, radius metadata or final diagnostics.
    Zero rows use absolute error divided by the floor, or a fixed 1.0 update
    unit when no floor is supplied. Every tensor and row retains positive weight.
    Non-reentrant chunk checkpoints support higher-order autograd/HVPs.
    """
    if (not isinstance(radius_weight, (int, float)) or isinstance(radius_weight, bool)
            or not math.isfinite(radius_weight) or radius_weight < 0):
        raise ValueError("radius_weight must be finite and nonnegative")
    if scale_floor is not None and (not isinstance(scale_floor, (int, float))
                                   or isinstance(scale_floor, bool) or not math.isfinite(scale_floor)
                                   or scale_floor <= 0):
        raise ValueError("scale_floor must be a fixed positive finite number or None")
    _validate(updates, observations, chunk_rows)
    layers = []
    for name, obs in observations.items():
        matrix = updates[name].reshape(_matrix_shape(obs.shape))
        value_sum = matrix.new_zeros((), dtype=torch.float64)
        radius_sum = matrix.new_zeros((), dtype=torch.float64)
        for start in range(0, matrix.shape[0], chunk_rows):
            end = min(start + chunk_rows, matrix.shape[0])
            args = matrix[start:end], obs.radius[start:end], obs.codes[start:end], scale_floor
            if torch.is_grad_enabled() and matrix.requires_grad:
                values, radii = checkpoint(_chunk_loss, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                values, radii = _chunk_loss(*args)
            value_sum = value_sum + values
            radius_sum = radius_sum + radii
        layers.append(value_sum / matrix.numel() + radius_weight * radius_sum / matrix.shape[0])
    return torch.stack(layers).mean()


@torch.no_grad()
def summarize_observation_fit(updates: Mapping[str, torch.Tensor], observations: Mapping[str, PackedObservation], *,
                              tolerance_steps: float = DEFAULT_TOLERANCE_STEPS,
                              chunk_rows: int = 64) -> dict:
    """True half-bin diagnostics, unaffected by the optional optimization floor.

    Exact codes are recomputed at each candidate's own abs-max radius using the
    actual nearest-even rule. Radius equality is reported separately. Their
    conjunction establishes exact packet reencoding for FP32 candidates only.
    Float64 synthetic candidates get exact_packet_match=None. A zero-radius
    row uses a separate fixed absolute tolerance, tolerance_steps * 1.0 units.
    """
    if (not isinstance(tolerance_steps, (int, float)) or isinstance(tolerance_steps, bool)
            or not math.isfinite(tolerance_steps) or tolerance_steps < 0):
        raise ValueError("tolerance_steps must be finite and nonnegative")
    _validate(updates, observations, chunk_rows)
    per_tensor = {}
    for name, obs in observations.items():
        matrix = updates[name].reshape(_matrix_shape(obs.shape))
        stats = {"elements": matrix.numel(), "positive_radius_elements": 0,
                 "zero_radius_elements": 0, "positive_radius_rows": 0,
                 "bin_violation_count": 0, "max_bin_violation_steps": 0.0,
                 "max_radius_error_steps": 0.0, "max_radius_error_abs": 0.0,
                 "max_zero_scale_error_abs": 0.0, "code_mismatch_count": 0,
                 "radius_mismatch_count": 0, "zero_scale_nonzero_count": 0,
                 "midpoint_tie_count": 0, "midpoint_wrong_code_count": 0}
        bin_sum = radius_sum = zero_sum = 0.0
        for start in range(0, matrix.shape[0], chunk_rows):
            end = min(start + chunk_rows, matrix.shape[0])
            values, radius, codes, nonzero, coordinate, violation, radius_delta = _chunk_values(
                matrix[start:end], obs.radius[start:end], obs.codes[start:end])
            active, zero_error = violation[nonzero], values[~nonzero].abs()
            radius_steps = (radius_delta[nonzero] / radius[nonzero] * 7.5).abs()
            candidate_radius = values.abs().amax(dim=1)
            # Exact-code diagnostics use the sender's declared CPU arithmetic,
            # regardless of the device used by the candidate training replay.
            cpu_values = values.cpu()
            reencoded = _round_codes(cpu_values, cpu_values.abs().amax(dim=1))
            stats["code_mismatch_count"] += int((reencoded != obs.codes[start:end]).sum())
            stats["radius_mismatch_count"] += int((candidate_radius != radius).sum())
            stats["max_radius_error_abs"] = max(stats["max_radius_error_abs"], float(radius_delta.abs().max()))
            stats["positive_radius_elements"] += active.numel()
            stats["zero_radius_elements"] += zero_error.numel()
            stats["positive_radius_rows"] += radius_steps.numel()
            if active.numel():
                stats["max_bin_violation_steps"] = max(stats["max_bin_violation_steps"], float(active.max()))
                stats["bin_violation_count"] += int((active > tolerance_steps).sum())
                stats["max_radius_error_steps"] = max(stats["max_radius_error_steps"], float(radius_steps.max()))
                bin_sum += float(active.sum())
                radius_sum += float(radius_steps.sum())
                t = coordinate[nonzero]
                ties = (t >= 0) & (t <= 15) & ((t - torch.floor(t)) == .5)
                mismatch = torch.round(t).clamp(0, 15) != codes[nonzero]
                stats["midpoint_tie_count"] += int(ties.sum())
                stats["midpoint_wrong_code_count"] += int((ties & mismatch).sum())
            if zero_error.numel():
                stats["max_zero_scale_error_abs"] = max(stats["max_zero_scale_error_abs"], float(zero_error.max()))
                stats["zero_scale_nonzero_count"] += int((zero_error != 0).sum())
                zero_sum += float(zero_error.sum())
        stats["mean_bin_violation_steps"] = bin_sum / max(1, stats["positive_radius_elements"])
        stats["bin_violation_rate"] = stats["bin_violation_count"] / max(1, stats["positive_radius_elements"])
        stats["mean_radius_error_steps"] = radius_sum / max(1, stats["positive_radius_rows"])
        stats["mean_zero_scale_error_abs"] = zero_sum / max(1, stats["zero_radius_elements"])
        stats["exact_code_match"] = stats["code_mismatch_count"] == 0
        stats["exact_radius_match"] = stats["radius_mismatch_count"] == 0
        stats["exact_packet_match"] = (stats["exact_code_match"] and stats["exact_radius_match"]
                                        if updates[name].dtype == torch.float32 else None)
        stats["closure_fit_within_tolerance"] = (
            stats["max_bin_violation_steps"] <= tolerance_steps
            and stats["max_radius_error_steps"] <= tolerance_steps
            and stats["max_zero_scale_error_abs"] <= tolerance_steps * ZERO_SCALE_NORMALIZATION)
        per_tensor[name] = stats
    rows = list(per_tensor.values())
    return {"codec_id": CODEC_ID, "rounding": "nearest_even", "tolerance_steps": tolerance_steps,
            "zero_scale_tolerance_abs": tolerance_steps * ZERO_SCALE_NORMALIZATION,
            "zero_scale_normalization_update_units": ZERO_SCALE_NORMALIZATION,
            "closure_fit_within_tolerance": all(row["closure_fit_within_tolerance"] for row in rows),
            "exact_code_match": all(row["exact_code_match"] for row in rows),
            "exact_radius_match": all(row["exact_radius_match"] for row in rows),
            "exact_packet_match": (all(row["exact_packet_match"] for row in rows)
                                     if all(updates[name].dtype == torch.float32 for name in updates) else None),
            "code_mismatch_count": sum(row["code_mismatch_count"] for row in rows),
            "radius_mismatch_count": sum(row["radius_mismatch_count"] for row in rows),
            "midpoint_tie_count": sum(row["midpoint_tie_count"] for row in rows),
            "midpoint_wrong_code_count": sum(row["midpoint_wrong_code_count"] for row in rows),
            "max_bin_violation_steps": max(row["max_bin_violation_steps"] for row in rows),
            "mean_bin_violation_steps": sum(row["mean_bin_violation_steps"] for row in rows) / len(rows),
            "bin_violation_rate": sum(row["bin_violation_count"] for row in rows)
            / max(1, sum(row["positive_radius_elements"] for row in rows)),
            "mean_tensor_bin_violation_rate": sum(row["bin_violation_rate"] for row in rows) / len(rows),
            "max_radius_error_steps": max(row["max_radius_error_steps"] for row in rows),
            "mean_radius_error_steps": sum(row["mean_radius_error_steps"] for row in rows) / len(rows),
            "max_radius_error_abs": max(row["max_radius_error_abs"] for row in rows),
            "max_zero_scale_error_abs": max(row["max_zero_scale_error_abs"] for row in rows),
            "mean_zero_scale_error_abs": sum(row["mean_zero_scale_error_abs"] for row in rows) / len(rows),
            "zero_scale_nonzero_count": sum(row["zero_scale_nonzero_count"] for row in rows),
            "mean_aggregation": "equal tensor means", "per_tensor": per_tensor}
