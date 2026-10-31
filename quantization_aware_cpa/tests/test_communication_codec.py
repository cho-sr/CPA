"""CPU codec gates: actual bytes, independent decoding, numerical behavior."""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest
import torch

from qacpa.communication_codec import (
    CODEC_ID, HEADER, HEADER_BYTES, MAGIC, ROW_CHUNK_SIZE,
    decode_packet, encode_fp32, encode_uint4, packet_info,
)


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def replace_header(packet, **values):
    names = ("magic", "version", "codec", "bits", "flags", "rows", "cols", "count", "radii")
    fields = dict(zip(names, HEADER.unpack_from(packet)))
    fields.update(values)
    return HEADER.pack(*(fields[name] for name in names)) + packet[HEADER_BYTES:]


def test_wire_contract_matches_frozen_protocol():
    path = Path(__file__).resolve().parents[1] / "configs/m10_compression_only_B16.protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    wire = protocol["wire_format"]
    assert HEADER.format == wire["header_struct"]
    assert HEADER_BYTES == wire["header_bytes"] == 32
    assert MAGIC.decode("ascii") == wire["magic_ascii"]
    assert CODEC_ID == protocol["codec"]["id"]
    assert ROW_CHUNK_SIZE == protocol["codec"]["encoder_rng"]["row_chunk_size"]


def test_known_nibbles_all_levels_and_odd_padding():
    # r=15 makes all 16 reconstruction levels exactly representable FP32.
    grid = torch.arange(-15, 16, 2, dtype=torch.float32).reshape(1, 16)
    packet = encode_uint4(grid, seed=7)
    assert packet[32:36] == struct.pack("<f", 15.0)
    assert packet[36:] == bytes((0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE))
    assert torch.equal(decode_packet(packet), grid)
    odd = encode_uint4(torch.tensor([[-3., 3., -3.]]), seed=0)
    assert odd[36:] == b"\xf0\x00"
    assert torch.equal(decode_packet(odd), torch.tensor([[-3., 3., -3.]]))


def test_cpu_fp32_bitwise_identity_and_fresh_decoder_storage():
    tiny = torch.nextafter(torch.tensor(0.), torch.tensor(1.)).item()
    values = torch.tensor([[0., -0., tiny], [-tiny, torch.finfo(torch.float32).max, -1.]]).T
    packet = encode_fp32(values)
    decoded = decode_packet(packet)
    assert decoded.is_contiguous() and decoded.device.type == "cpu"
    assert torch.equal(decoded.view(torch.int32), values.contiguous().view(torch.int32))
    assert len(packet) == 32 + values.numel() * 4
    decoded.zero_()
    assert torch.equal(decode_packet(packet).view(torch.int32), values.contiguous().view(torch.int32))


def test_chunk_rng_includes_zero_rows_and_preserves_global_rng_and_input():
    values = torch.linspace(-1, 1, 65 * 3).reshape(65, 3)
    values[0] = 0
    values[63] = 0
    original = values.clone()
    rng_state = torch.random.get_rng_state().clone()
    packet = encode_uint4(values, seed=173)
    assert packet == encode_uint4(values, seed=173)
    assert torch.equal(values, original)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    # Independent scalar code decisions from the declared draws, followed by
    # a separate Python nibble pack. This also crosses the 64-row boundary.
    generator = torch.Generator().manual_seed(173)
    uniforms = torch.cat([torch.rand((64, 3), dtype=torch.float64, generator=generator),
                          torch.rand((1, 3), dtype=torch.float64, generator=generator)])
    codes = []
    for row in range(65):
        radius = float(values[row].abs().max())
        for col in range(3):
            t = min(15., max(0., (float(values[row, col]) / radius + 1.) * 7.5)) if radius else 0.
            lower = math.floor(t)
            codes.append(lower + int(float(uniforms[row, col]) < t - lower) if radius else 0)
    reference = bytes(codes[k] | ((codes[k + 1] if k + 1 < len(codes) else 0) << 4)
                      for k in range(0, len(codes), 2))
    assert packet[32 + 65 * 4:] == reference
    assert torch.equal(decode_packet(packet)[[0, 63]], torch.zeros((2, 3)))


def test_scalar_odd_rows_zero_rows_and_subnormal_radii():
    tiny = torch.nextafter(torch.tensor(0.), torch.tensor(1.)).item()
    maximum = torch.finfo(torch.float32).max
    values = torch.tensor([[0., 0., 0.], [-tiny, 0., tiny], [0., 0., 0.],
                           [-maximum, 0., maximum]], dtype=torch.float32)
    decoded = decode_packet(encode_uint4(values, seed=90))
    assert torch.isfinite(decoded).all()
    assert torch.equal(decoded[:, [0, 2]], values[:, [0, 2]])
    assert torch.equal(decoded[[0, 2]], torch.zeros((2, 3)))
    assert torch.equal(decode_packet(encode_uint4(torch.zeros((1, 1)), 0)), torch.zeros((1, 1)))
    assert torch.equal(decode_packet(encode_uint4(torch.tensor([[2.]]), 0)), torch.tensor([[2.]]))


def test_stochastic_mean_variance_and_full_delta_error_bound():
    # Pre-fixed 65,536 independent draws per coordinate, seed 314159. With r=15
    # the grid spacing is 2: x=.5 has p=.75 and x=0 has p=.5.
    repeats = 65536
    values = torch.tensor([[-15., 15., .5, 0.]]).repeat(repeats, 1)
    decoded = decode_packet(encode_uint4(values, seed=314159)).double()
    for col, mean, variance in ((2, .5, .75), (3, 0., 1.)):
        actual = decoded[:, col]
        assert abs(float(actual.mean()) - mean) < 6.0 * math.sqrt(variance / repeats)
        assert abs(float(actual.var(unbiased=False)) - variance) < .025
    error = (decoded - values.double()).abs()
    assert float(error.max()) <= 2.
    assert bool((error > 1.).any()), "Actual stochastic rounding is not a half-delta cell model"


def test_decoder_isolation_in_new_process(tmp_path):
    values = torch.tensor([[-15., -1., 1., 15.], [0., 0., 0., 0.]])
    path = tmp_path / "packet.bin"
    path.write_bytes(encode_uint4(values, seed=3))
    expected = decode_packet(path.read_bytes())
    script = ("from pathlib import Path; import sys,torch; "
              "from qacpa.communication_codec import decode_packet; "
              "value=decode_packet(Path(sys.argv[1]).read_bytes()); "
              "sys.stdout.buffer.write(value.numpy().tobytes())")
    result = subprocess.run([sys.executable, "-c", script, str(path)], check=True, capture_output=True,
                            cwd=Path(__file__).resolve().parents[1])
    assert result.stdout == expected.numpy().tobytes()


@pytest.mark.parametrize("changes", [
    {"magic": b"BADMAGIC"}, {"version": 2}, {"codec": 2}, {"bits": 8},
    {"flags": 1}, {"rows": 0}, {"cols": 0}, {"count": 2}, {"radii": 0},
    {"rows": 0xFFFFFFFF, "cols": 0xFFFFFFFF, "count": 0xFFFFFFFE00000001,
     "radii": 0xFFFFFFFF},
])
def test_malformed_headers_rejected_before_allocation(changes):
    packet = replace_header(encode_uint4(torch.tensor([[-1., 1., 0.]]), 0), **changes)
    with pytest.raises(ValueError):
        packet_info(packet)
    with pytest.raises(ValueError):
        decode_packet(packet)


@pytest.mark.parametrize("radius", [-1., -0., float("nan"), float("inf")])
def test_invalid_radii_rejected(radius):
    packet = encode_uint4(torch.tensor([[-1., 1.]]), 0)
    packet = packet[:32] + struct.pack("<f", radius) + packet[36:]
    with pytest.raises(ValueError):
        decode_packet(packet)


def test_truncation_padding_noncanonical_zero_codes_and_nonfinite_fp32():
    packet = encode_uint4(torch.tensor([[-1., 1., 0.]]), 0)
    malformed = [b"", packet[:31], packet[:-1], packet + b"\0", packet[:-1] + bytes([packet[-1] | 0xF0])]
    zero_packet = encode_uint4(torch.zeros((3, 1)), 0)
    malformed += [zero_packet[:-2] + b"\x10\x00", zero_packet[:-1] + b"\x01"]
    fp32 = encode_fp32(torch.zeros((1, 1)))
    malformed += [fp32[:32] + struct.pack("<f", float("nan"))]
    for item in malformed:
        with pytest.raises(ValueError):
            decode_packet(item)


@pytest.mark.parametrize("values", [
    torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]]),
    torch.tensor([[1.]], dtype=torch.float64), torch.tensor([[1]], dtype=torch.int32),
    torch.zeros(3), torch.zeros((0, 3)), torch.zeros((2, 0)),
])
def test_invalid_inputs_rejected(values):
    for encoder in (lambda x: encode_uint4(x, 0), encode_fp32):
        with pytest.raises(ValueError):
            encoder(values)


@pytest.mark.parametrize("seed", [-1, 2**64, .5, True])
def test_invalid_seeds_rejected(seed):
    with pytest.raises((TypeError, ValueError)):
        encode_uint4(torch.ones((1, 1)), seed)


def test_packet_metadata_counts_actual_packed_bytes():
    values = torch.ones((65, 5))
    packet = encode_uint4(values, 0)
    info = packet_info(packet)
    assert info["codec_id"] == CODEC_ID
    assert info["shape"] == [65, 5]
    assert info["radius_bytes"] == 260
    assert info["code_bytes"] == 163
    assert info["packet_bytes"] == len(packet) == 455
    assert info["fp32_reference_packet_bytes"] == 1332
    assert info["compression_ratio_vs_fp32"] == 1332 / 455
    assert packet_info(encode_fp32(values))["compression_ratio_vs_fp32"] == 1.0
