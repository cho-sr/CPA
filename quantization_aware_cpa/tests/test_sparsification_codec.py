import struct

import numpy as np
import pytest
import torch

from qacpa.sparsification_codec import (
    HEADER, MAGIC, encode_topk50, decode_packet, packet_info, sparsification_diagnostics,
)


@pytest.mark.parametrize("shape", [(1, 2), (3, 3), (7, 10), (32, 64)])
def test_packet_matches_independent_sorted_topk_and_preserves_input_bits(shape):
    rng = np.random.default_rng(42)
    # Repeated magnitudes test exact count and deterministic tie breaking.
    array = rng.integers(-4, 5, size=shape).astype(np.float32) / 8
    x = torch.from_numpy(array.copy())
    packet = encode_topk50(x)
    assert packet == encode_topk50(x)
    info = packet_info(packet)
    flat = array.reshape(-1)
    indices = sorted(range(flat.size), key=lambda i: (-abs(float(flat[i])), i))[:flat.size // 2]
    expected = np.zeros_like(flat)
    expected[indices] = flat[indices]
    decoded = decode_packet(packet)
    np.testing.assert_array_equal(decoded.numpy().reshape(-1).view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(x.numpy().view(np.uint32), array.view(np.uint32))
    assert info["kept_elements"] == len(indices)
    assert len(packet) == 32 + (flat.size + 7) // 8 + 4 * len(indices)
    diagnostics = sparsification_diagnostics(x, packet, decoded)
    assert diagnostics["retained_fp32_bits_exact"] and diagnostics["discarded_values_zero"]


def test_topk_is_global_over_tensor_and_uses_absolute_magnitude():
    x = torch.tensor([[-100., -90., 80., 70.], [1., -2., 3., -4.]])
    decoded = decode_packet(encode_topk50(x))
    assert torch.equal(decoded[0], x[0])
    assert torch.count_nonzero(decoded[1]) == 0


def test_ties_and_zero_values_keep_exact_coordinates_and_signed_zero_bits():
    x = torch.tensor([[-0., 0., -0., 0., 0., 0., 0., 0.]])
    packet = encode_topk50(x)
    assert packet[HEADER.size] == 0b00001111
    decoded = decode_packet(packet)
    assert torch.signbit(decoded[0, 0]) and torch.signbit(decoded[0, 2])
    diagnostics = sparsification_diagnostics(x, packet, decoded)
    assert diagnostics["kept_elements"] == 4 and diagnostics["decoded_nonzero_elements"] == 0
    assert diagnostics["relative_l2_error"] == 0


@pytest.mark.parametrize("x", [torch.ones(1), torch.ones(2, 2, dtype=torch.float64),
                                  torch.ones(1, 1), torch.tensor([[float('nan'), 1.]]),
                                  torch.tensor([[float('inf'), 1.]])])
def test_invalid_inputs_are_rejected(x):
    with pytest.raises(ValueError):
        encode_topk50(x)


@pytest.mark.parametrize("fault", ["truncate", "append", "magic", "count", "bitmap", "padding", "nan"])
def test_receiver_rejects_corrupt_packets(fault):
    packet = bytearray(encode_topk50(torch.arange(6, dtype=torch.float32).reshape(2, 3)))
    if fault == "truncate":
        packet.pop()
    elif fault == "append":
        packet.append(0)
    elif fault == "magic":
        packet[0] ^= 1
    elif fault == "count":
        packet[:32] = HEADER.pack(MAGIC, 2, 3, 2, 0)
    elif fault == "bitmap":
        packet[32] = 0
    elif fault == "padding":
        packet[32] |= 128
    else:
        packet[-4:] = struct.pack('<f', float('nan'))
    with pytest.raises(ValueError):
        decode_packet(bytes(packet))


def test_audit_rejects_wrong_values_and_nonzero_discarded_coordinates():
    x = torch.tensor([[1., 2., -3., 4.]])
    packet = encode_topk50(x)
    decoded = decode_packet(packet)
    decoded[0, 0] = .1
    with pytest.raises(RuntimeError, match="bits or failed zero fill"):
        sparsification_diagnostics(x, packet, decoded)


def test_audit_rejects_structurally_valid_non_topk_packet():
    x = torch.tensor([[1., 2., 3., 4.]])
    packet = HEADER.pack(MAGIC, 1, 4, 2, 0) + bytes([3]) + struct.pack('<ff', 1., 2.)
    with pytest.raises(RuntimeError, match="Top-k ordering"):
        sparsification_diagnostics(x, packet, decode_packet(packet))
