"""CPU gates for deterministic codes, exact ties and differentiable half-cells."""
import json
import struct

import pytest
import torch

from qacpa import deterministic_observation as det
from qacpa import full_update_observation as stochastic
from qacpa.communication_codec import HEADER, HEADER_BYTES, encode_fp32, packet_info as old_packet_info


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def observe(values):
    return det.load_observations(det.encode_updates(values))


def test_actual_four_bit_grid_bytes_codec_isolation_and_decoder_only_input():
    source = {"weight": torch.arange(-15., 16., 2.).reshape(1, 16)}
    entry = det.encode_updates(source)["weight"]
    packet = entry["packet"]
    assert packet[HEADER_BYTES:HEADER_BYTES + 4] == struct.pack("<f", 15.)
    assert packet[HEADER_BYTES + 4:] == bytes((0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE))
    assert det.packet_info(packet)["codec_number"] == 2
    assert det.packet_info(packet)["code_bytes"] == 8
    assert det.packet_info(packet)["packet_bytes"] == HEADER_BYTES + 4 + 8
    assert torch.equal(det.decode_packet(packet), source["weight"])
    # Neither observer silently treats the other wire format as its own codec.
    with pytest.raises(ValueError):
        old_packet_info(packet)
    with pytest.raises(ValueError):
        stochastic.load_observations({"weight": entry})
    for wrong in (stochastic.encode_updates(source, 42)["weight"]["packet"],
                  encode_fp32(source["weight"])):
        with pytest.raises(ValueError, match="deterministic"):
            det.load_observations({"weight": {"shape": [1, 16], "packet": wrong}})


def test_nearest_even_midpoint_ties_are_actual_encoder_decisions():
    # With r=15 and h=2, all midpoints are exactly representable. Code is
    # nearest-even in the affine integer coordinate, not signed input units.
    values = torch.cat([torch.tensor([-15.]), torch.arange(-14., 15., 2.), torch.tensor([15.])]).reshape(1, -1)
    observations = observe({"weight": values})
    expected = [0] + [int(round(i + .5)) for i in range(15)] + [15]
    assert observations["weight"].codes.tolist() == [expected]
    fit = det.summarize_observation_fit({"weight": values}, observations)
    assert fit["exact_code_match"] and fit["exact_radius_match"] and fit["exact_packet_match"]
    assert fit["midpoint_tie_count"] == 15 and fit["midpoint_wrong_code_count"] == 0
    assert det.observation_loss({"weight": values}, observations).item() == 0


def test_closed_half_cell_does_not_approve_wrong_midpoint_tie():
    observations = observe({"weight": torch.tensor([[-15., 15., -13.]])})
    assert observations["weight"].codes.tolist() == [[0, 15, 1]]
    candidate = {"weight": torch.tensor([[-15., 15., -14.]])}
    assert det.observation_loss(candidate, observations).item() == 0
    fit = det.summarize_observation_fit(candidate, observations, tolerance_steps=0.)
    assert fit["closure_fit_within_tolerance"]
    assert not fit["exact_code_match"] and fit["exact_radius_match"]
    assert not fit["exact_packet_match"]
    assert fit["midpoint_wrong_code_count"] == 1
    assert fit["code_mismatch_count"] == 1


@pytest.mark.parametrize("midpoint,lower_code,upper_code", [(-14., 0, 1), (-12., 1, 2), (0., 7, 8), (14., 14, 15)])
def test_adjacent_fp32_values_on_either_side_of_ties(midpoint, lower_code, upper_code):
    center = torch.tensor(midpoint, dtype=torch.float32)
    below = torch.nextafter(center, torch.tensor(float("-inf")))
    above = torch.nextafter(center, torch.tensor(float("inf")))
    # Around zero the nearest subnormal values cannot affect a FP64 sum with
    # radius15. Use an ordinary representable perturbation for that boundary;
    # the encoder's finite-arithmetic convention is explicit in the packet spec.
    if midpoint == 0.:
        below, above = torch.tensor(-1e-5), torch.tensor(1e-5)
    values = torch.stack([torch.tensor(-15.), torch.tensor(15.), below, center, above]).reshape(1, 5)
    obs = observe({"weight": values})
    midpoint_code = lower_code if lower_code % 2 == 0 else upper_code
    assert obs["weight"].codes[0, 2:].tolist() == [lower_code, midpoint_code, upper_code]


def test_half_step_cells_reject_values_only_stochastic_full_step_would_accept():
    observations = observe({"weight": torch.tensor([[-15., 15., 1.]])})
    candidate = {"weight": torch.tensor([[-15., 15., -.5]])}
    # q8 has deterministic closure [0,2], not stochastic closure [-1,3].
    assert det.observation_loss(candidate, observations, radius_weight=0).item() == pytest.approx(.25 ** 2 / 3)
    assert not det.summarize_observation_fit(candidate, observations)["closure_fit_within_tolerance"]


def test_radius_consistency_is_separate_from_codes_and_closed_cells():
    observations = observe({"weight": torch.tensor([[-15., 15.]])})
    candidate = {"weight": torch.tensor([[-14.5, 14.5]])}
    assert det.observation_loss(candidate, observations, radius_weight=0).item() == 0
    assert det.observation_loss(candidate, observations).item() == pytest.approx(.25 ** 2)
    fit = det.summarize_observation_fit(candidate, observations)
    assert fit["exact_code_match"]
    assert not fit["exact_radius_match"] and not fit["exact_packet_match"]
    assert fit["max_radius_error_steps"] == .25


def test_sorted_names_conv_fc_bias_layouts_actual_source_containment_and_odd_chunks():
    generator = torch.Generator().manual_seed(321)
    values = {"z.weight": torch.randn((65, 3), generator=generator),
              "a.weight": torch.randn((3, 2, 3, 3), generator=generator),
              "b.bias": torch.randn(7, generator=generator)}
    original = {name: value.clone() for name, value in values.items()}
    rng = torch.random.get_rng_state().clone()
    entries = det.encode_updates(values)
    assert list(entries) == sorted(values)
    assert entries == det.encode_updates(dict(reversed(list(values.items()))))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(value, original[name]) for name, value in values.items())
    observations = det.load_observations(entries)
    assert observations["a.weight"].codes.shape == (3, 18)
    assert observations["b.bias"].codes.shape == (1, 7)
    assert observations["b.bias"].radius.shape == (1,)
    for name, obs in observations.items():
        assert obs.shape == tuple(values[name].shape)
        assert obs.codec_id == det.CODEC_ID
        decoded = det.decode_packet(entries[name]["packet"])
        r, q = obs.radius.double(), obs.codes.double()
        expected = (r[:, None] * (2 * q / 15 - 1)).float()
        assert torch.equal(decoded, expected)
    assert det.observation_loss(values, observations).item() == 0
    fit = det.summarize_observation_fit(values, observations)
    assert fit["exact_packet_match"] and fit["closure_fit_within_tolerance"]
    json.dumps(fit, allow_nan=False)


def test_same_code_and_radius_always_decode_same_center_without_source_correction():
    # Close-to-boundary inputs used to trigger private source-based corrections
    # in the legacy quantizer. Both now produce exactly the same packet/center.
    left = {"weight": torch.tensor([[-15., 15., 1.999995]])}
    right = {"weight": torch.tensor([[-15., 15., 1.999997]])}
    packets = [det.encode_updates(value)["weight"]["packet"] for value in (left, right)]
    assert packets[0] == packets[1]
    assert det.decode_packet(packets[0])[0, 2].item() == pytest.approx(1.)


def test_zero_and_subnormal_radius_and_maximum_fp32():
    tiny = torch.nextafter(torch.tensor(0.), torch.tensor(1.)).item()
    maximum = torch.finfo(torch.float32).max
    values = {"weight": torch.tensor([[0., -0., 0.], [-tiny, 0., tiny], [-maximum, 0., maximum]])}
    entries = det.encode_updates(values)
    observations = det.load_observations(entries)
    assert observations["weight"].codes[0].tolist() == [0, 0, 0]
    assert observations["weight"].codes[1].tolist() == [0, 8, 15]
    decoded = det.decode_packet(entries["weight"]["packet"])
    assert torch.isfinite(decoded).all()
    assert not bool(torch.signbit(decoded[0]).any())
    assert det.observation_loss(values, observations).item() == 0
    assert det.summarize_observation_fit(values, observations)["exact_packet_match"]
    candidate = values["weight"].double().clone().requires_grad_()
    with torch.no_grad():
        candidate[1, 0] = -2 * tiny
    loss = det.observation_loss({"weight": candidate}, observations)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(candidate.grad).all()
    assert candidate.grad[1, 0] != 0
    assert det.summarize_observation_fit({"weight": candidate}, observations)["exact_packet_match"] is None


def test_zero_row_absolute_rule_and_public_scale_floor():
    observations = observe({"bias": torch.zeros(3)})
    candidate = torch.tensor([.25, -.5, .75], dtype=torch.float64, requires_grad=True)
    baseline = det.observation_loss({"bias": candidate}, observations)
    floored = det.observation_loss({"bias": candidate}, observations, scale_floor=.5)
    assert baseline.item() == pytest.approx(.5 + .75)
    assert floored.item() == pytest.approx(2 * baseline.item())
    floored.backward()
    assert torch.isfinite(candidate.grad).all() and bool((candidate.grad != 0).all())
    fit = det.summarize_observation_fit({"bias": candidate}, observations)
    assert fit["zero_scale_nonzero_count"] == 3 and not fit["exact_radius_match"]
    assert fit["zero_scale_tolerance_abs"] == 1e-6


def test_scale_floor_changes_weights_but_not_true_half_cells_or_radius():
    observations = observe({"weight": torch.tensor([[-15., 15., 1.]])})
    candidate = {"weight": torch.tensor([[-18., 15., -1.]])}
    before = det.summarize_observation_fit(candidate, observations)
    loss = det.observation_loss(candidate, observations)
    floored = det.observation_loss(candidate, observations, scale_floor=4.)
    assert floored.item() == pytest.approx(loss.item() / 4)
    assert det.summarize_observation_fit(candidate, observations) == before
    assert not before["closure_fit_within_tolerance"]


def test_equal_tensor_weighting_not_parameter_count_weighting():
    observations = observe({"large": torch.zeros(100), "small": torch.zeros(1)})
    values = {"large": torch.ones(100), "small": torch.tensor([2.])}
    assert det.observation_loss(values, observations).item() == pytest.approx((2 + 4) / 2)


def test_finite_nonzero_gradient_hvp_and_chunk_equivalence():
    source = {"weight": torch.tensor([[-15., 15., 1.]]).repeat(65, 1)}
    observations = observe(source)
    a = torch.tensor([[-18., 16., -2.]], dtype=torch.float64).repeat(65, 1).requires_grad_()
    b = a.detach().clone().requires_grad_()
    loss_a = det.observation_loss({"weight": a}, observations, chunk_rows=1)
    loss_b = det.observation_loss({"weight": b}, observations, chunk_rows=64)
    grad_a, = torch.autograd.grad(loss_a, a, create_graph=True)
    grad_b, = torch.autograd.grad(loss_b, b, create_graph=True)
    direction = torch.linspace(.1, .9, a.numel(), dtype=torch.float64).reshape(a.shape)
    hvp_a, = torch.autograd.grad((grad_a * direction).sum(), a)
    hvp_b, = torch.autograd.grad((grad_b * direction).sum(), b)
    for first, second in ((loss_a, loss_b), (grad_a, grad_b), (hvp_a, hvp_b)):
        torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)
        assert torch.isfinite(first).all()
    assert bool((grad_a != 0).all()) and bool((hvp_a != 0).any())


def test_gradient_through_nonlinear_update_graph_and_finite_difference():
    observations = observe({"weight": torch.tensor([[-15., 15., 1.]])})
    # Keep the row maximum unique: amax has no classical Hessian at tied maxima.
    variable = torch.tensor([[-4.1, 4., -2.]], dtype=torch.float64, requires_grad=True)
    def objective(x):
        update = x * x.abs() * 1.2
        return det.observation_loss({"weight": update}, observations, scale_floor=3.)
    assert torch.autograd.gradcheck(objective, (variable,), eps=1e-6, atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradgradcheck(objective, (variable,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_strict_names_shapes_codec_tags_and_no_truth_entry():
    values = {"bias": torch.tensor([-1., 0., 1.])}
    entries = det.encode_updates(values)
    observations = det.load_observations(entries)
    with pytest.raises(ValueError, match="names"):
        det.observation_loss({"other": values["bias"]}, observations)
    with pytest.raises(ValueError, match="shape"):
        det.observation_loss({"bias": values["bias"].reshape(1, 3)}, observations)
    with pytest.raises(ValueError, match="exactly"):
        det.load_observations({"bias": {**entries["bias"], "true_update": values["bias"]}})
    with pytest.raises(ValueError, match="canonical"):
        det.load_observations({"bias": {**entries["bias"], "shape": [3, 1]}})
    with pytest.raises(ValueError, match="stochastic"):
        det.observation_loss(values, stochastic.load_observations(stochastic.encode_updates(values, 0)))


@pytest.mark.parametrize("bad_radius", [-1., -0., float("nan"), float("inf")])
def test_invalid_radius_wire_bytes_rejected(bad_radius):
    entry = det.encode_updates({"bias": torch.tensor([-1., 0., 1.])})["bias"]
    packet = entry["packet"]
    malformed = packet[:HEADER_BYTES] + struct.pack("<f", bad_radius) + packet[HEADER_BYTES + 4:]
    with pytest.raises(ValueError, match="radius"):
        det.packet_info(malformed)


def test_padding_length_zero_codes_and_missing_endpoint_are_rejected():
    packet = det.encode_updates({"bias": torch.tensor([-1., 0., 1.])})["bias"]["packet"]
    zero = det.encode_updates({"bias": torch.zeros(3)})["bias"]["packet"]
    malformed = [packet[:-1], packet + b"\0", packet[:-1] + bytes([packet[-1] | 0xF0]),
                 zero[:-1] + b"\x01", packet[:HEADER_BYTES + 4] + b"\x88\x08"]
    for value in malformed:
        with pytest.raises(ValueError):
            det.packet_info(value)
    header = list(HEADER.unpack_from(packet))
    header[7] += 1
    with pytest.raises(ValueError):
        det.packet_info(HEADER.pack(*header) + packet[HEADER_BYTES:])


@pytest.mark.parametrize("value", [torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                                   torch.ones(1, dtype=torch.int64), torch.tensor(1.), torch.zeros(0)])
def test_invalid_sender_and_candidate_tensors_rejected(value):
    with pytest.raises(ValueError):
        det.encode_updates({"bias": value})
    with pytest.raises(ValueError):
        det.observation_loss({"bias": value}, observe({"bias": torch.ones(1)}))


@pytest.mark.parametrize("kwargs", [{"scale_floor": 0}, {"scale_floor": -1},
                                    {"scale_floor": float("nan")}, {"scale_floor": True},
                                    {"radius_weight": -1}, {"chunk_rows": 0}])
def test_invalid_loss_settings_rejected(kwargs):
    values = {"bias": torch.ones(1)}
    with pytest.raises(ValueError):
        det.observation_loss(values, observe(values), **kwargs)
