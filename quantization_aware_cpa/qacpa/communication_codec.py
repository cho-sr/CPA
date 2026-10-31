"""Versioned communication packets, independent of the historical dither model.

The uint4 encoder uses transmitted FP32 row radii, CPU float64 arithmetic and
one private RNG call per 64-row chunk (including zero rows). Decoding receives
only packet bytes. No source-dependent correction or quantization residual is
transmitted. Both wire formats are specified by m10_compression_only_B16.
"""
from __future__ import annotations

import operator
import struct

import numpy as np
import torch


CODEC_ID = "uniform_affine_uint4_stochastic_v1"
FP32_CODEC_ID = "fp32_identity_v1"
ROW_CHUNK_SIZE = 64
FORMAT_VERSION = 1
MAGIC = b"CPACOMM1"
HEADER = struct.Struct("<8sBBBBIIQI")
HEADER_BYTES = HEADER.size


def _input(Y: torch.Tensor) -> tuple[int, int]:
    if not isinstance(Y, torch.Tensor):
        raise TypeError("Y must be a torch.Tensor")
    if Y.device.type != "cpu" or Y.dtype != torch.float32 or Y.layout != torch.strided:
        raise ValueError("Y must be a strided CPU torch.float32 tensor")
    if Y.ndim != 2 or any(size <= 0 or size > 0xFFFFFFFF for size in Y.shape):
        raise ValueError("Y must have two positive uint32 dimensions")
    return int(Y.shape[0]), int(Y.shape[1])


def _header(rows: int, cols: int, codec: int) -> bytes:
    return HEADER.pack(MAGIC, FORMAT_VERSION, codec, 4 if codec else 32, 0,
                       rows, cols, rows * cols, rows if codec else 0)


def encode_uint4(Y: torch.Tensor, seed: int) -> bytes:
    """Encode a finite FP32 matrix into stochastic packed uint4 + row radii.

    The caller retains the encoder seed/version as experiment provenance. The
    decoder does not need either the seed or a copy of the original tensor.
    This function does not alter the input, CPU global RNG or GPU RNG.
    """
    rows, cols = _input(Y)
    if isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    try:
        seed = operator.index(seed)
    except TypeError as error:
        raise TypeError("seed must be an integer") from error
    if not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("seed must be in the uint64 range")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    radii = torch.empty(rows, dtype=torch.float32)
    codes_offset = HEADER_BYTES + rows * 4
    packet = bytearray(codes_offset + (rows * cols + 1) // 2)
    packet[:HEADER_BYTES] = _header(rows, cols, 1)
    for start in range(0, rows, ROW_CHUNK_SIZE):
        stop = min(start + ROW_CHUNK_SIZE, rows)
        values = Y[start:stop].detach()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Y must contain only finite FP32 values")
        radius = values.abs().amax(dim=1)
        radii[start:stop] = radius
        zero = radius == 0
        safe_radius = torch.where(zero, torch.ones_like(radius), radius).double()
        coordinate = ((values.double() / safe_radius[:, None] + 1.0) * 7.5).clamp_(0.0, 15.0)
        lower = torch.floor(coordinate)
        uniform = torch.rand(values.shape, generator=rng, dtype=torch.float64, device="cpu")
        codes = (lower + (uniform < coordinate - lower)).to(torch.uint8)
        # Random draws for zero rows have already occurred; do not skip them.
        codes[zero] = 0
        flat = codes.flatten()
        packed = flat[0::2].clone()
        packed[:flat.numel() // 2] |= flat[1::2] << 4
        byte_start = codes_offset + start * cols // 2
        # Chunk starts are even element offsets because ROW_CHUNK_SIZE is 64.
        packet[byte_start:byte_start + packed.numel()] = packed.numpy().tobytes()
    packet[HEADER_BYTES:codes_offset] = radii.numpy().astype("<f4", copy=False).tobytes()
    return bytes(packet)


def encode_fp32(Y: torch.Tensor) -> bytes:
    """Identity packet for codec validation; preserves all finite FP32 bits."""
    rows, cols = _input(Y)
    packet = bytearray(HEADER_BYTES + rows * cols * 4)
    packet[:HEADER_BYTES] = _header(rows, cols, 0)
    for start in range(0, rows, ROW_CHUNK_SIZE):
        stop = min(start + ROW_CHUNK_SIZE, rows)
        values = Y[start:stop].detach()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Y must contain only finite FP32 values")
        raw = values.contiguous().numpy().astype("<f4", copy=False).tobytes()
        offset = HEADER_BYTES + start * cols * 4
        packet[offset:offset + len(raw)] = raw
    return bytes(packet)


def _parse(packet: bytes) -> tuple[dict, np.ndarray | None, np.ndarray]:
    if not isinstance(packet, bytes):
        raise TypeError("packet must be immutable bytes")
    if len(packet) < HEADER_BYTES:
        raise ValueError("packet is shorter than its header")
    magic, version, codec, bits, flags, rows, cols, count, radius_count = HEADER.unpack_from(packet)
    if magic != MAGIC or version != FORMAT_VERSION or flags != 0:
        raise ValueError("unsupported packet magic, format version or flags")
    if codec not in (0, 1) or bits != (4 if codec else 32):
        raise ValueError("unsupported codec or bit width")
    if rows == 0 or cols == 0 or count != rows * cols:
        raise ValueError("invalid shape or element count")
    if radius_count != (rows if codec else 0):
        raise ValueError("invalid radius count")
    radius_bytes = radius_count * 4
    data_bytes = (count + 1) // 2 if codec else count * 4
    if len(packet) != HEADER_BYTES + radius_bytes + data_bytes:
        raise ValueError("packet length does not match its shape")
    radii = None
    if codec:
        radii = np.frombuffer(packet, dtype="<f4", count=rows, offset=HEADER_BYTES)
        if not np.isfinite(radii).all() or np.signbit(radii).any():
            raise ValueError("radii must be finite and canonical nonnegative FP32")
        data = np.frombuffer(packet, dtype=np.uint8, count=data_bytes,
                             offset=HEADER_BYTES + radius_bytes)
        if count % 2 and int(data[-1]) & 0xF0:
            raise ValueError("nonzero high-nibble padding")
        # All-zero rows have one canonical code representation, including when
        # odd column counts put their first/last nibble beside another row.
        for row in np.flatnonzero(radii == 0):
            first, end = int(row) * cols, (int(row) + 1) * cols
            if first % 2:
                if int(data[first // 2]) & 0xF0:
                    raise ValueError("nonzero code in a zero-radius row")
                first += 1
            if end % 2:
                if int(data[end // 2]) & 0x0F:
                    raise ValueError("nonzero code in a zero-radius row")
                end -= 1
            if np.any(data[first // 2:end // 2]):
                raise ValueError("nonzero code in a zero-radius row")
    else:
        data = np.frombuffer(packet, dtype="<f4", count=count, offset=HEADER_BYTES)
        for start in range(0, count, ROW_CHUNK_SIZE * cols):
            if not np.isfinite(data[start:start + ROW_CHUNK_SIZE * cols]).all():
                raise ValueError("FP32 packet contains nonfinite values")
    reference_bytes = HEADER_BYTES + count * 4
    info = {
        "codec_id": CODEC_ID if codec else FP32_CODEC_ID,
        "codec_number": codec, "format_version": version, "rows": rows,
        "cols": cols, "shape": [rows, cols], "elements": count, "bits": bits,
        "radius_count": radius_count, "header_bytes": HEADER_BYTES,
        "radius_bytes": radius_bytes, "code_bytes": data_bytes if codec else 0,
        "fp32_data_bytes": 0 if codec else data_bytes,
        "payload_bytes": radius_bytes + data_bytes, "packet_bytes": len(packet),
        "fp32_reference_packet_bytes": reference_bytes,
        "compression_ratio_vs_fp32": reference_bytes / len(packet),
        "reduction_fraction_vs_fp32": 1.0 - len(packet) / reference_bytes,
    }
    return info, radii, data


def packet_info(packet: bytes) -> dict:
    """Validate the complete packet and return JSON-compatible wire metadata."""
    return _parse(packet)[0]


def decode_packet(packet: bytes) -> torch.Tensor:
    """Validate and decode packet bytes to a fresh contiguous CPU FP32 matrix."""
    info, radii, data = _parse(packet)
    rows, cols = info["rows"], info["cols"]
    if radii is None:
        return torch.from_numpy(data.astype(np.float32, copy=True).reshape(rows, cols))
    result = torch.empty((rows, cols), dtype=torch.float32, device="cpu")
    for start in range(0, rows, ROW_CHUNK_SIZE):
        stop = min(start + ROW_CHUNK_SIZE, rows)
        count = (stop - start) * cols
        first = start * cols // 2
        packed = data[first:first + (count + 1) // 2]
        codes = np.empty(count, dtype=np.uint8)
        codes[0::2] = packed & 0x0F
        codes[1::2] = packed[:count // 2] >> 4
        values = torch.from_numpy(codes.reshape(stop - start, cols)).double()
        radius = torch.from_numpy(radii[start:stop].astype(np.float64, copy=True))
        values = radius[:, None] * (2.0 * values / 15.0 - 1.0)
        values[radius == 0] = 0.0
        result[start:stop] = values.float()
    return result
