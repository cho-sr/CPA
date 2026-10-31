"""Task-local reader for the historical 50% and ratio Top-k packets.

The 50% packet was produced by ``sparsification_codec.py`` and carries the
historical codec label ``topk50_global_bitmap_fp32_v1``.  Its wire layout is
the same as the ratio codec used by 30% and 20%.  This adapter verifies that
fact before the packet-only attack uses the shared random-fill and
recompression implementation.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

from . import sparsification_codec as historical50
from . import topk_ratio_codec as ratio


N_ELEMENTS = 4096 * 25088
SHAPE = (4096, 25088)
SOURCE_CODECS = {
    50: historical50.CODEC_ID,
    30: ratio.CODEC_ID,
    20: ratio.CODEC_ID,
}


def expected_packet_bytes(keep_percent: int) -> int:
    if keep_percent not in SOURCE_CODECS:
        raise ValueError(f"Unsupported paper sparse fraction: {keep_percent}")
    kept = N_ELEMENTS * keep_percent // 100
    return ratio.HEADER.size + (N_ELEMENTS + 7) // 8 + 4 * kept


def _same_bits(left: np.ndarray, right: np.ndarray) -> bool:
    return np.array_equal(left.astype("<f4", copy=False).view(np.uint32),
                          right.astype("<f4", copy=False).view(np.uint32))


def validate_received_packet(packet: bytes, keep_percent: int,
                             source_codec: str | None = None) -> dict[str, Any]:
    """Validate one saved packet and prove the 50% schema equivalence.

    This function accepts only serialized packet bytes.  It never reads the
    uncompressed update or any image/evaluation artifact.
    """
    if keep_percent not in SOURCE_CODECS:
        raise ValueError(f"Unsupported paper sparse fraction: {keep_percent}")
    expected_source = SOURCE_CODECS[keep_percent]
    if source_codec is not None and source_codec != expected_source:
        raise ValueError(f"Unexpected source codec label: {source_codec}")

    generic_info, generic_mask, generic_values = ratio._parse(packet)
    expected_kept = N_ELEMENTS * keep_percent // 100
    if tuple(generic_info["shape"]) != SHAPE or generic_info["kept_elements"] != expected_kept:
        raise RuntimeError("Received packet has the wrong FC1 shape or retained count")
    if len(packet) != expected_packet_bytes(keep_percent):
        raise RuntimeError("Received packet has the wrong serialized length")

    source_info = generic_info
    source_mask = generic_mask
    source_values = generic_values
    equivalence: dict[str, Any] = {"historical_50_schema_checked": False}
    if keep_percent == 50:
        old_info, old_mask, old_values = historical50._parse(packet)
        if old_info["kept_elements"] != generic_info["kept_elements"]:
            raise RuntimeError("Historical 50% retained count differs from ratio parser")
        if not np.array_equal(old_mask, generic_mask) or not _same_bits(old_values, generic_values):
            raise RuntimeError("Historical 50% mask or FP32 payload differs from ratio parser")
        old_decoded = historical50.decode_packet(packet)
        generic_decoded = ratio.decode_packet(packet)
        if not torch.equal(old_decoded, generic_decoded):
            raise RuntimeError("Historical 50% decoded tensor differs from ratio codec")
        if ratio.encode_topk(generic_decoded, 50) != packet:
            raise RuntimeError("Generic 50% recompression does not preserve packet bytes")
        source_info = old_info
        source_mask = old_mask
        source_values = old_values
        equivalence = {
            "historical_50_schema_checked": True,
            "mask_equal": True,
            "retained_fp32_bits_equal": True,
            "decoded_tensor_equal": True,
            "generic_recompression_equal": True,
        }

    return {
        "source_codec": expected_source,
        "generic_codec": ratio.CODEC_ID,
        "shape": list(source_info["shape"]),
        "elements": int(source_info["elements"]),
        "kept_elements": int(source_info["kept_elements"]),
        "keep_percent": int(keep_percent),
        "packet_bytes": int(len(packet)),
        "bitmap_bytes": int(source_info["bitmap_bytes"]),
        "value_bytes": int(source_info["value_bytes"]),
        "header_bytes": int(source_info["header_bytes"]),
        "expected_packet_bytes": expected_packet_bytes(keep_percent),
        "mask_sha256": hashlib.sha256(np.packbits(source_mask, bitorder="little").tobytes()).hexdigest(),
        "retained_values_sha256": hashlib.sha256(source_values.astype("<f4", copy=False).tobytes()).hexdigest(),
        "retained_fp32_bits_exact": True,
        "row_major_little_bit_order": True,
        "tie_break": "flat index ascending",
        **equivalence,
    }


def decode_for_attack(packet: bytes, keep_percent: int) -> torch.Tensor:
    """Decode through the generic reader after the schema check."""
    validate_received_packet(packet, keep_percent)
    return ratio.decode_packet(packet)


def recompress_for_ranking(update: torch.Tensor, keep_percent: int) -> bytes:
    """Use the shared generic Top-k serializer for observable ranking."""
    if tuple(update.shape) != SHAPE or update.dtype != torch.float32:
        raise ValueError("Expected the FC1 FP32 update shape for recompression")
    return ratio.encode_topk(update, keep_percent)
