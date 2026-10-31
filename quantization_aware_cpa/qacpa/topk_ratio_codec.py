"""Exact magnitude Top-k with a configured retained percentage: row-major bitmap + unchanged FP32 values.

Selection is global within the supplied FC1 tensor. Ties retain lower flat
indices first. No quantization, rescaling, error feedback or imputation.
The 32-byte little-endian header is magic, rows, columns, kept count, reserved.
"""
from __future__ import annotations

import math
import struct

import numpy as np
import torch

CODEC_ID = "topk_global_bitmap_fp32_v1"
HEADER = struct.Struct("<8sIIQQ")
MAGIC = b"QACTOPK1"
CHUNK = 1_048_576


def encode_topk(tensor: torch.Tensor, keep_percent: int) -> bytes:
    if tensor.ndim != 2 or tensor.dtype != torch.float32 or tensor.numel() < 2:
        raise ValueError("Expected a nonempty 2D FP32 tensor with at least two elements")
    if type(keep_percent) is not int or not 1 <= keep_percent < 100:
        raise ValueError("keep_percent must be an integer in 1..99")
    if tensor.numel() * keep_percent // 100 < 1:
        raise ValueError("The requested fraction retains no coordinates")
    flat = tensor.detach().cpu().contiguous().numpy().reshape(-1)
    if not np.isfinite(flat).all():
        raise ValueError("Top-k input must be finite")
    n, kept = flat.size, flat.size * keep_percent // 100
    magnitudes = np.abs(flat)
    magnitudes.partition(n - kept)
    threshold = float(magnitudes[n - kept])
    del magnitudes
    mask = np.zeros(n, dtype=np.bool_)
    for start in range(0, n, CHUNK):
        mask[start:start + CHUNK] = np.abs(flat[start:start + CHUNK]) > threshold
    needed = kept - int(mask.sum())
    for start in range(0, n, CHUNK):
        if not needed:
            break
        ties = np.flatnonzero(np.abs(flat[start:start + CHUNK]) == threshold)
        take = min(needed, len(ties))
        mask[start + ties[:take]] = True
        needed -= take
    if needed or int(mask.sum()) != kept:
        raise RuntimeError("Top-k must retain exactly floor(N * keep_percent / 100) coordinates")
    bitmap = np.packbits(mask, bitorder="little").tobytes()
    values = flat[mask].astype("<f4", copy=False).tobytes()
    return HEADER.pack(MAGIC, *tensor.shape, kept, 0) + bitmap + values


def _parse(packet: bytes) -> tuple[dict, np.ndarray, np.ndarray]:
    if len(packet) < HEADER.size:
        raise ValueError("Truncated sparse header")
    magic, rows, columns, kept, reserved = HEADER.unpack_from(packet)
    n = rows * columns
    if magic != MAGIC or reserved != 0 or not rows or not columns or n < 2 or not 0 < kept < n:
        raise ValueError("Invalid Top-k packet header")
    bitmap_bytes = (n + 7) // 8
    expected = HEADER.size + bitmap_bytes + 4 * kept
    if len(packet) != expected:
        raise ValueError("Sparse packet length does not match its header")
    bitmap = np.frombuffer(packet, dtype=np.uint8, count=bitmap_bytes, offset=HEADER.size)
    if n % 8 and int(bitmap[-1]) >> (n % 8):
        raise ValueError("Nonzero bitmap padding")
    mask = np.unpackbits(bitmap, bitorder="little", count=n).view(np.bool_)
    if int(mask.sum()) != kept:
        raise ValueError("Sparse bitmap count differs from header")
    values = np.frombuffer(packet, dtype="<f4", count=kept, offset=HEADER.size + bitmap_bytes)
    if not np.isfinite(values).all():
        raise ValueError("Sparse values must be finite")
    info = {"codec_id": CODEC_ID, "shape": [rows, columns], "elements": n,
            "kept_elements": kept, "keep_fraction": kept / n, "value_bits": 32,
            "value_bytes": 4 * kept, "bitmap_bytes": bitmap_bytes,
            "header_bytes": HEADER.size, "packet_bytes": expected,
            "fp32_reference_packet_bytes": HEADER.size + 4 * n}
    return info, mask, values


def packet_info(packet: bytes) -> dict:
    return _parse(packet)[0]


def decode_packet(packet: bytes) -> torch.Tensor:
    """Receiver takes only transmitted bytes and fills discarded coordinates with zero."""
    info, mask, values = _parse(packet)
    dense = np.zeros(info["elements"], dtype=np.float32)
    dense[mask] = values
    return torch.from_numpy(dense.reshape(info["shape"]))


def sparsification_diagnostics(actual: torch.Tensor, packet: bytes, decoded: torch.Tensor) -> dict:
    """Audit exact retained values, top-k ordering and zero fill, outside the attack."""
    info, mask, _ = _parse(packet)
    if actual.dtype != torch.float32 or decoded.dtype != torch.float32 or tuple(actual.shape) != tuple(info["shape"]) or actual.shape != decoded.shape:
        raise ValueError("Diagnostic input shape or dtype differs from packet")
    original = actual.detach().cpu().contiguous().numpy().reshape(-1)
    received = decoded.detach().cpu().contiguous().numpy().reshape(-1)
    total_energy = kept_energy = error_energy = 0.
    kept_min, dropped_max = math.inf, 0.
    nonzero = 0
    last_kept_tie = first_dropped_tie = None
    for start in range(0, len(mask), CHUNK):
        a, b, m = (array[start:start + CHUNK] for array in (original, received, mask))
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError("Sparse diagnostics require finite values")
        if not np.array_equal(a[m].view(np.uint32), b[m].view(np.uint32)) or np.any(b[~m] != 0):
            raise RuntimeError("Sparse receiver changed retained FP32 bits or failed zero fill")
        aa, bb = a.astype(np.float64), b.astype(np.float64)
        total_energy += float(aa @ aa)
        kept_energy += float(bb @ bb)
        diff = aa - bb
        error_energy += float(diff @ diff)
        nonzero += int(np.count_nonzero(b))
        if m.any():
            kept_min = min(kept_min, float(np.abs(a[m]).min()))
        if (~m).any():
            dropped_max = max(dropped_max, float(np.abs(a[~m]).max()))
    if kept_min < dropped_max:
        raise RuntimeError("Packet violates global magnitude Top-k ordering")
    if kept_min == dropped_max:
        for start in range(0, len(mask), CHUNK):
            ties = np.abs(original[start:start + CHUNK]) == kept_min
            m = mask[start:start + CHUNK]
            kept_ids, dropped_ids = np.flatnonzero(ties & m), np.flatnonzero(ties & ~m)
            if len(kept_ids):
                last_kept_tie = start + int(kept_ids[-1])
            if len(dropped_ids) and first_dropped_tie is None:
                first_dropped_tie = start + int(dropped_ids[0])
        if last_kept_tie is not None and first_dropped_tie is not None and last_kept_tie > first_dropped_tie:
            raise RuntimeError("Top-k cutoff ties must prefer lower flat indices")
    return {**info, "scope": "global within attacked classifier.0.weight tensor",
            "selection_rule": "absolute magnitude descending, flat index ascending for ties",
            "retained_fp32_bits_exact": True, "discarded_values_zero": True,
            "minimum_retained_abs": kept_min, "maximum_discarded_abs": dropped_max,
            "decoded_nonzero_elements": nonzero, "decoded_nonzero_fraction": nonzero / len(mask),
            "retained_energy_fraction": kept_energy / total_energy if total_energy else None,
            "relative_l2_error": math.sqrt(error_energy / total_energy) if total_energy else 0.,
            "quantization": False, "rescaling": False, "error_feedback": False}
