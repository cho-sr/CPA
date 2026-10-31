import struct

import numpy as np
import pytest
import torch

from qacpa.topk_ratio_codec import (
    HEADER, MAGIC, encode_topk, decode_packet, packet_info, sparsification_diagnostics,
)
from qacpa.sparsification_codec import encode_topk50


@pytest.mark.parametrize("percent", [30, 20, 10, 50])
@pytest.mark.parametrize("shape", [(1, 10), (3, 7), (7, 10), (32, 64)])
def test_real_packet_matches_independent_sort_and_preserves_fp32_bits(percent, shape):
    array = np.random.default_rng(42).integers(-4, 5, size=shape).astype(np.float32) / 8
    x = torch.from_numpy(array.copy())
    packet = encode_topk(x, percent)
    assert packet == encode_topk(x, percent)
    flat = array.reshape(-1)
    indices = sorted(range(flat.size), key=lambda i: (-abs(float(flat[i])), i))[:flat.size * percent // 100]
    expected = np.zeros_like(flat)
    expected[indices] = flat[indices]
    decoded = decode_packet(packet)
    np.testing.assert_array_equal(decoded.numpy().reshape(-1).view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(x.numpy().view(np.uint32), array.view(np.uint32))
    info = packet_info(packet)
    assert info["kept_elements"] == len(indices)
    assert len(packet) == 32 + (flat.size + 7) // 8 + 4 * len(indices)
    diag = sparsification_diagnostics(x, packet, decoded)
    assert diag["retained_fp32_bits_exact"] and diag["discarded_values_zero"]
    expected_error = np.linalg.norm(flat.astype(float) - expected) / np.linalg.norm(flat.astype(float))
    assert diag["relative_l2_error"] == pytest.approx(expected_error)
    if percent == 50:
        assert packet == encode_topk50(x)


@pytest.mark.parametrize("percent", [30, 20, 10])
def test_magnitude_global_selection_ties_and_signed_zero(percent):
    x = torch.tensor([[-100., -90., 80., 70., 60.], [1., -2., 3., -4., 5.]])
    decoded = decode_packet(encode_topk(x, percent))
    assert torch.count_nonzero(decoded[1]) == 0
    assert torch.count_nonzero(decoded) == percent // 10
    zeros = torch.tensor([[-0., 0., -0., 0., 0., 0., 0., 0., 0., 0.]])
    packet = encode_topk(zeros, percent)
    assert packet[HEADER.size] == (1 << (percent // 10)) - 1
    assert torch.signbit(decode_packet(packet)[0, 0])


@pytest.mark.parametrize("percent", [0, 100, -10, 10.0, True, "30"])
def test_invalid_retained_percentage(percent):
    with pytest.raises(ValueError):
        encode_topk(torch.ones(2, 10), percent)


@pytest.mark.parametrize("fault", ["truncate", "append", "magic", "count", "bitmap", "padding", "nan"])
def test_corrupt_receiver_packet_is_rejected(fault):
    packet = bytearray(encode_topk(torch.arange(21, dtype=torch.float32).reshape(3, 7), 30))
    if fault == "truncate":
        packet.pop()
    elif fault == "append":
        packet.append(0)
    elif fault == "magic":
        packet[0] ^= 1
    elif fault == "count":
        packet[:32] = HEADER.pack(MAGIC, 3, 7, 5, 0)
    elif fault == "bitmap":
        packet[32:35] = bytes(3)
    elif fault == "padding":
        packet[34] |= 128
    else:
        packet[-4:] = struct.pack("<f", float("nan"))
    with pytest.raises(ValueError):
        decode_packet(bytes(packet))


def test_audit_rejects_valid_packet_with_wrong_topk_selection():
    x = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    packet = HEADER.pack(MAGIC, 1, 10, 3, 0) + bytes([7, 0]) + struct.pack("<fff", 0., 1., 2.)
    with pytest.raises(RuntimeError, match="Top-k ordering"):
        sparsification_diagnostics(x, packet, decode_packet(packet))
