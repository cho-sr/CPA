"""Tiny CPU gates for packet-only, full-parameter stochastic observations."""
from collections import OrderedDict
import json
import struct

import pytest
import torch

from qacpa.communication_codec import HEADER_BYTES, encode_fp32, packet_info
from qacpa.full_update_observation import (
    encode_updates, load_observations, observation_loss, summarize_observation_fit,
)


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def observe(updates):
    return load_observations(encode_updates(updates, seed=173))


def test_conv_fc_bias_layouts_and_actual_source_containment():
    generator = torch.Generator().manual_seed(19)
    updates = OrderedDict([
        ("features.0.weight", torch.randn((3, 2, 3, 3), generator=generator)),
        ("features.0.bias", torch.randn(3, generator=generator)),
        ("classifier.0.weight", torch.randn((65, 3), generator=generator)),
        ("classifier.0.bias", torch.randn(65, generator=generator)),
    ])
    entries = encode_updates(updates, seed=173)
    observations = load_observations(entries)
    assert list(observations) == list(updates)
    assert observations["features.0.weight"].codes.shape == (3, 18)
    assert observations["features.0.bias"].codes.shape == (1, 3)
    assert observations["classifier.0.bias"].radius.shape == (1,)
    for name, value in updates.items():
        assert observations[name].shape == tuple(value.shape)
        assert set(entries[name]) == {"shape", "packet"}
        assert packet_info(entries[name]["packet"])["bits"] == 4
    assert observation_loss(updates, observations).item() == 0.0
    summary = summarize_observation_fit(updates, observations)
    assert summary["closure_fit_within_tolerance"]
    assert summary["zero_marginal_probability_count"] == 0
    assert summary["exact_joint_rng_match"] == "not_checked"
    json.dumps(summary, allow_nan=False)


def test_encoding_reproducible_without_changing_input_or_global_rng():
    updates = {"a": torch.tensor([[-15., 15., .5, 0.]]).repeat(65, 1),
               "b": torch.tensor([[-1., 1., 0.]])}
    snapshot = {name: value.clone() for name, value in updates.items()}
    rng = torch.random.get_rng_state().clone()
    assert encode_updates(updates, 72) == encode_updates(updates, 72)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(torch.equal(value, snapshot[name]) for name, value in updates.items())


def test_full_step_support_contains_actual_halfstep_counterexample():
    updates = {"weight": torch.tensor([[-15., 15., .5]]).repeat(64, 1)}
    observations = observe(updates)
    q = observations["weight"].codes[:, 2].double()
    center = q * 2.0 - 15.0
    assert bool(((center - .5).abs() > 1.0).any()), "real encoder can exceed half a step"
    assert observation_loss(updates, observations).item() == 0.0
    assert observation_loss(updates, observations, mode="center").item() > 0.0


def test_closed_cell_boundary_has_zero_marginal_probability():
    source = {"weight": torch.tensor([[-15., 15., 1.]])}
    observations = observe(source)
    assert observations["weight"].codes.tolist() == [[0, 15, 8]]
    candidate = {"weight": torch.tensor([[-15., 15., -1.]], dtype=torch.float64)}
    assert observation_loss(candidate, observations).item() == 0.0
    fit = summarize_observation_fit(candidate, observations, tolerance_steps=0.0)
    assert fit["closure_fit_within_tolerance"]
    assert fit["zero_probability_boundary_count"] == 1
    assert fit["zero_marginal_probability_count"] == 1
    assert fit["exact_joint_rng_match"] == "not_checked"
    # A mathematical endpoint with a nonzero marginal probability is different.
    source_fit = summarize_observation_fit(source, observations, tolerance_steps=0.0)
    assert source_fit["zero_probability_boundary_count"] == 0


def test_support_intersects_observed_radius_and_radius_term_is_separate():
    observations = observe({"weight": torch.tensor([[-15., 15., 1.]])})
    outside = {"weight": torch.tensor([[-16., 15., 1.]])}
    assert observation_loss(outside, observations, radius_weight=0).item() == pytest.approx(.25 / 3)
    assert observation_loss(outside, observations, radius_weight=1).item() == pytest.approx(.25 / 3 + .25)
    # All entries can stay in code closures while their maximum misses r.
    shrunken = {"weight": torch.tensor([[-14., 14., 1.]])}
    assert observation_loss(shrunken, observations, radius_weight=0).item() == 0.0
    assert observation_loss(shrunken, observations).item() == pytest.approx(.25)
    fit = summarize_observation_fit(shrunken, observations)
    assert fit["max_bin_violation_steps"] == 0.0
    assert fit["max_radius_error_steps"] == pytest.approx(.5)
    assert not fit["closure_fit_within_tolerance"]


@pytest.mark.parametrize("mode", ["cell", "center"])
def test_candidate_gradients_finite_and_nonzero_outside_bins(mode):
    observations = observe({"weight": torch.tensor([[-15., 15., 1.]])})
    candidate = torch.tensor([[-18., 16., -3.]], dtype=torch.float64, requires_grad=True)
    loss = observation_loss({"weight": candidate}, observations, mode=mode)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(candidate.grad).all()
    assert bool((candidate.grad != 0).all())
    assert candidate.grad[0, 0] < 0 and candidate.grad[0, 1] > 0 and candidate.grad[0, 2] < 0


def test_zero_radius_uses_fixed_absolute_error_and_no_division():
    observations = observe({"bias": torch.zeros(3)})
    candidate = torch.tensor([.25, -.5, .75], dtype=torch.float64, requires_grad=True)
    loss = observation_loss({"bias": candidate}, observations)
    assert loss.item() == pytest.approx(.5 + .75)
    loss.backward()
    assert torch.isfinite(candidate.grad).all() and bool((candidate.grad != 0).all())
    fit = summarize_observation_fit({"bias": candidate}, observations)
    assert fit["max_zero_scale_error_abs"] == .75
    assert fit["zero_scale_normalization_update_units"] == 1.0
    assert fit["zero_scale_tolerance_abs"] == 1e-6
    assert not fit["closure_fit_within_tolerance"]
    zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    zero_loss = observation_loss({"bias": zero}, observations)
    zero_loss.backward()
    assert zero_loss.item() == 0.0 and torch.isfinite(zero.grad).all()
    tiny_candidate = {"bias": torch.full((3,), 5e-7, dtype=torch.float64)}
    assert summarize_observation_fit(tiny_candidate, observations)["closure_fit_within_tolerance"]
    assert not summarize_observation_fit(tiny_candidate, observations,
                                         tolerance_steps=1e-8)["closure_fit_within_tolerance"]


def test_subnormal_and_maximum_fp32_radii_use_double_coordinates():
    tiny = torch.nextafter(torch.tensor(0.), torch.tensor(1.)).item()
    maximum = torch.finfo(torch.float32).max
    source = {"weight": torch.tensor([[-tiny, 0., tiny], [0., 0., 0.], [-maximum, 0., maximum]])}
    observations = observe(source)
    assert observation_loss(source, observations).item() == 0.0
    assert summarize_observation_fit(source, observations)["closure_fit_within_tolerance"]
    # Double candidates allow a finite derivative in inverse-subnormal units.
    candidate = source["weight"].double().clone().requires_grad_()
    with torch.no_grad():
        candidate[0, 0] = -2 * tiny
    loss = observation_loss({"weight": candidate}, observations)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(candidate.grad).all()
    assert candidate.grad[0, 0] != 0
    json.dumps(summarize_observation_fit({"weight": candidate}, observations), allow_nan=False)


def test_equal_tensor_weighting_not_parameter_count_weighting():
    observations = observe({"small": torch.zeros(1), "large": torch.zeros(100)})
    candidates = {"small": torch.tensor([2.]), "large": torch.ones(100)}
    # Each layer has a mean absolute coordinate term plus one row-max term.
    assert observation_loss(candidates, observations).item() == pytest.approx((4.0 + 2.0) / 2)
    single = observation_loss({"small": candidates["small"]}, {"small": observations["small"]})
    large = observation_loss({"large": candidates["large"]}, {"large": observations["large"]})
    assert observation_loss(candidates, observations).item() == ((single + large) / 2).item()


def test_row_chunking_preserves_values_and_gradients():
    source = {"weight": torch.tensor([[-15., 15., 1.]]).repeat(65, 1)}
    observations = observe(source)
    one = (source["weight"].double() * 1.5).requires_grad_()
    many = one.detach().clone().requires_grad_()
    loss_one = observation_loss({"weight": one}, observations, chunk_rows=1)
    loss_many = observation_loss({"weight": many}, observations, chunk_rows=64)
    loss_one.backward()
    loss_many.backward()
    torch.testing.assert_close(loss_one, loss_many, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(one.grad, many.grad, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("mode", ["cell", "center"])
@pytest.mark.parametrize("radius_weight", [0.0, 0.7])
def test_loss_checkpoint_preserves_scalar_and_gradients(monkeypatch, mode, radius_weight):
    import qacpa.full_update_observation as module

    source = torch.linspace(-1., 1., 128).repeat(129, 1)
    source[0].zero_()
    observations = observe({"weight": source, "bias": torch.zeros(3)})
    candidates = {"weight": (source.double() * 1.7 + 0.13).requires_grad_(),
                  "bias": torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64, requires_grad=True)}
    checked_loss = observation_loss(candidates, observations, mode=mode, radius_weight=radius_weight)
    checked_gradients = torch.autograd.grad(checked_loss, tuple(candidates.values()))
    with monkeypatch.context() as context:
        context.setattr(module, "checkpoint", lambda function, *args, **kwargs: function(*args))
        direct_loss = observation_loss(candidates, observations, mode=mode, radius_weight=radius_weight)
        direct_gradients = torch.autograd.grad(direct_loss, tuple(candidates.values()))
    torch.testing.assert_close(checked_loss, direct_loss, atol=0, rtol=0)
    for checked, direct in zip(checked_gradients, direct_gradients):
        torch.testing.assert_close(checked, direct, atol=0, rtol=0)


def test_loss_checkpoint_does_not_retain_all_double_chunk_arrays(monkeypatch):
    import qacpa.full_update_observation as module

    source = torch.linspace(-1., 1., 128).repeat(256, 1)
    observations = observe({"weight": source})
    candidate = (source * 1.7 + 0.13).requires_grad_()

    def retained_storage_bytes():
        storages = set()

        def pack(tensor):
            storage = tensor.untyped_storage()
            storages.add((str(tensor.device), storage.data_ptr(), storage.nbytes()))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss = observation_loss({"weight": candidate}, observations)
        torch.autograd.grad(loss, candidate)
        return sum(item[2] for item in storages)

    checked_bytes = retained_storage_bytes()
    with monkeypatch.context() as context:
        context.setattr(module, "checkpoint", lambda function, *args, **kwargs: function(*args))
        direct_bytes = retained_storage_bytes()
    # Count backing storage once: repeated slices of the same FP32 candidate
    # and uint8 observation must not be mistaken for independent allocations.
    assert checked_bytes < direct_bytes / 2


def test_name_shape_and_observer_only_contract_are_strict():
    entries = encode_updates({"bias": torch.tensor([-1., 0., 1.])}, 0)
    observations = load_observations(entries)
    with pytest.raises(ValueError, match="names"):
        observation_loss({"other": torch.zeros(3)}, observations)
    with pytest.raises(ValueError, match="shape"):
        observation_loss({"bias": torch.zeros((1, 3))}, observations)
    with pytest.raises(ValueError, match="names"):
        observation_loss({"bias": torch.zeros(3), "extra": torch.zeros(1)}, observations)
    with pytest.raises(ValueError, match="exactly"):
        load_observations({"bias": {**entries["bias"], "true_update": torch.zeros(3)}})
    with pytest.raises(ValueError, match="canonical"):
        load_observations({"bias": {**entries["bias"], "shape": [3, 1]}})
    for shape in ([], [True, 3], [0, 3], [1.0, 3]):
        with pytest.raises(ValueError):
            load_observations({"bias": {**entries["bias"], "shape": shape}})


def test_non4bit_malformed_and_nonfinite_packets_rejected():
    source = torch.tensor([[-1., 0., 1.]])
    with pytest.raises(ValueError, match="uint4"):
        load_observations({"weight": {"shape": [1, 3], "packet": encode_fp32(source)}})
    entry = encode_updates({"weight": source}, 0)["weight"]
    packet = entry["packet"]
    for radius in (float("nan"), float("inf"), -1., -0.):
        bad = packet[:HEADER_BYTES] + struct.pack("<f", radius) + packet[HEADER_BYTES + 4:]
        with pytest.raises(ValueError):
            load_observations({"weight": {**entry, "packet": bad}})
    for bad in (packet[:-1], packet + b"\0", packet[:-1] + bytes([packet[-1] | 0xF0])):
        with pytest.raises(ValueError):
            load_observations({"weight": {**entry, "packet": bad}})
    # A nonzero abs-max radius must correspond to at least one endpoint code.
    # This has valid codec bytes but cannot be emitted by its abs-max sender.
    no_endpoint = packet[:HEADER_BYTES + 4] + b"\x88\x08"
    with pytest.raises(ValueError, match="endpoint"):
        load_observations({"weight": {**entry, "packet": no_endpoint}})
    zero = encode_updates({"bias": torch.zeros(3)}, 0)["bias"]
    bad = zero["packet"][:-1] + b"\x01"
    with pytest.raises(ValueError):
        load_observations({"bias": {**zero, "packet": bad}})


@pytest.mark.parametrize("bad", [torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                                 torch.tensor(1.), torch.zeros(0), torch.ones(3, dtype=torch.int64)])
def test_nonfinite_or_invalid_candidates_and_sender_values_rejected(bad):
    observations = observe({"bias": torch.ones(1)})
    with pytest.raises((ValueError, TypeError)):
        encode_updates({"bias": bad}, 0)
    with pytest.raises((ValueError, TypeError)):
        observation_loss({"bias": bad}, observations)


def test_observer_input_is_fresh_and_does_not_require_source_or_seed():
    packet = encode_updates({"bias": torch.tensor([-1., 0., 1.])}, 321)["bias"]["packet"]
    entries = {"bias": {"shape": [3], "packet": packet}}
    observations = load_observations(entries)
    observations["bias"].codes.zero_()
    fresh = load_observations(entries)
    assert fresh["bias"].codes[0, 2] == 15
    assert fresh["bias"].radius.item() == 1.0


@pytest.mark.parametrize("kwargs", [{"mode": "bad"}, {"radius_weight": -1},
                                    {"radius_weight": float("nan")}, {"chunk_rows": 0},
                                    {"chunk_rows": True}])
def test_loss_options_rejected(kwargs):
    source = {"bias": torch.ones(1)}
    with pytest.raises(ValueError):
        observation_loss(source, observe(source), **kwargs)


@pytest.mark.parametrize("tolerance", [-1., float("nan"), float("inf")])
def test_diagnostic_tolerance_is_explicit_and_validated(tolerance):
    source = {"bias": torch.ones(1)}
    with pytest.raises(ValueError):
        summarize_observation_fit(source, observe(source), tolerance_steps=tolerance)
