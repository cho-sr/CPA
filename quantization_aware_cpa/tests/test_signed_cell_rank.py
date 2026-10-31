"""Small CPU-only tests for the external signed rank/cell projection."""
from __future__ import annotations

import pytest
import torch

from qacpa import signed_cell_rank as rank
from quantized_update_cpa import cell_utils


def encoded(value: torch.Tensor):
    q, s = cell_utils.encode(value)
    return q, cell_utils.dequantize(q, s), s


def test_gram_eigh_matches_float64_direct_svd_without_centering():
    value = torch.tensor([
        [2.0, -1.0, 0.5, 3.0, 1.0, -2.0, .25],
        [-1.5, 2.5, 1.0, 0.0, -3.0, .5, 2.0],
        [.25, 1.0, -2.0, 1.5, 2.25, .75, -1.0],
        [3.0, -2.0, .5, -1.0, 1.0, 2.0, .125],
        [.5, .75, 1.25, -2.5, .5, -1.25, 3.0],
        [-2.0, .25, 2.5, 1.0, -1.5, .5, .75],
    ], dtype=torch.float32)
    rank_target = 3
    u, singular, vh = torch.linalg.svd(value.double(), full_matrices=False)
    expected = (u[:, :rank_target] * singular[:rank_target]) @ vh[:rank_target]
    actual = rank.gram_eigh(value, rank_target).double()
    assert torch.allclose(actual, expected, atol=3e-4, rtol=3e-4)
    # Nonzero row means are retained: the projector is deliberately uncentered.
    assert torch.allclose(actual.mean(dim=1), expected.mean(dim=1), atol=3e-4, rtol=3e-4)


def test_ap0_is_identity_and_does_not_mutate_observation():
    original = torch.tensor([
        [-3.2, -1.0, 0.0, 2.0, 4.1],
        [1.5, -2.2, 3.0, .25, -4.1],
        [.75, 2.1, -1.2, 3.3, -2.9],
        [-.5, .25, 1.0, -3.0, 2.5],
    ], dtype=torch.float32)
    q, v0, s = encoded(original)
    q_before, s_before, v_before = q.clone(), s.clone(), v0.clone()
    result = rank.alternating_projection(v0, q, s, rank=2, iterations=0, projection_device="cpu")
    assert torch.equal(result.final_v, v0)
    assert result.final_m is None
    assert len(result.diagnostics) == 1
    assert torch.equal(q, q_before) and torch.equal(s, s_before)
    assert torch.equal(v0, v_before)


def test_cell_feasible_v_and_rank_limited_m_are_reported_separately():
    original = torch.tensor([
        [-7.0, 0.0, 7.0, 3.0, -2.0, 1.0],
        [1.0, 5.0, -4.0, 2.0, -1.0, 6.0],
        [2.0, -3.0, 4.0, -5.0, 1.0, 0.0],
        [-1.0, 2.0, -2.0, 3.0, 4.0, -6.0],
    ], dtype=torch.float32)
    q, v0, s = encoded(original)
    result = rank.alternating_projection(v0, q, s, rank=1, iterations=2, projection_device="cpu")
    fit = rank.fixed_cell_metrics(result.final_v, q, s)
    assert fit["exact_code_match"] and fit["bounded_cell_satisfied"]
    assert result.final_m is not None
    assert int(torch.linalg.matrix_rank(result.final_m).item()) <= 1
    assert len(result.diagnostics) == 3
    assert all(row["gradient_norm"]["status"] == "NOT_APPLICABLE" for row in result.diagnostics)
    assert all(row["V_next"] is not None for row in result.diagnostics[1:])
    assert all(row["V_next"]["fixed_scale"]["exact_code_match"] for row in result.diagnostics[1:])


def test_zero_and_float32_endpoint_codec_fixtures_remain_valid():
    zero = torch.zeros((2, 4), dtype=torch.float32)
    q0, v0, s0 = encoded(zero)
    result0 = rank.alternating_projection(v0, q0, s0, rank=1, iterations=2, projection_device="cpu")
    assert torch.equal(result0.final_v, zero)
    assert rank.fixed_cell_metrics(result0.final_v, q0, s0)["exact_code_match"]

    maximum = torch.tensor(torch.finfo(torch.float32).max, dtype=torch.float32)
    edge = torch.stack((-maximum, torch.tensor(0., dtype=torch.float32), maximum)).reshape(1, 3)
    qe, ve, se = encoded(edge)
    assert torch.equal(qe, torch.tensor([[-7, 0, 7]], dtype=torch.int8))
    # The finite-edge codec check is an identity/AP0 fixture: squaring the
    # largest FP32 value would overflow a float32 row Gram by design.
    result_edge = rank.alternating_projection(ve, qe, se, rank=1, iterations=0, projection_device="cpu")
    fit = rank.fixed_cell_metrics(result_edge.final_v, qe, se)
    assert fit["exact_code_match"] and fit["bounded_cell_satisfied"]
    assert fit["scale_consistency"]["candidate_supported"]


def test_off_rank_fixture_moves_and_repeated_cpu_run_is_deterministic():
    original = torch.tensor([
        [1., 2., 3., 4., 5.],
        [5., 4., 3., 2., 1.],
        [2., -1., 4., 0., 3.],
        [-2., 3., 1., 5., -4.],
    ], dtype=torch.float32)
    q, v0, s = encoded(original)
    first = rank.alternating_projection(v0, q, s, rank=1, iterations=2, projection_device="cpu")
    second = rank.alternating_projection(v0, q, s, rank=1, iterations=2, projection_device="cpu")
    assert not torch.equal(first.final_v, v0)
    assert torch.equal(first.final_v, second.final_v)
    assert torch.equal(first.final_m, second.final_m)
    for first_row, second_row in zip(first.diagnostics, second.diagnostics):
        assert {k: v for k, v in first_row.items() if k != "runtime_seconds"} == {
            k: v for k, v in second_row.items() if k != "runtime_seconds"}


def test_movement_from_previous_v_uses_consecutive_retained_iterates(monkeypatch):
    original = torch.tensor([
        [1., 2., 3., 4., 5.],
        [5., 4., 3., 2., 1.],
        [2., -1., 4., 0., 3.],
        [-2., 3., 1., 5., -4.],
    ], dtype=torch.float32)
    q, v0, s = encoded(original)
    retained = []
    record = rank._record_step

    def spy(iteration, before, projected, after, *args, **kwargs):
        retained.append((before.detach().clone(), after.detach().clone()))
        return record(iteration, before, projected, after, *args, **kwargs)

    monkeypatch.setattr(rank, "_record_step", spy)
    result = rank.alternating_projection(v0, q, s, rank=1, iterations=3, projection_device="cpu")
    assert len(retained) == 3
    for (before, after), diagnostic in zip(retained, result.diagnostics[1:]):
        expected = rank._movement(after, before)
        assert diagnostic["movement_from_previous_V"] == expected
        assert diagnostic["V_next"]["fixed_scale"]["exact_code_match"]
    assert torch.equal(result.initial_v, v0)


def test_invalid_rank_dtype_and_nonfinite_inputs_fail_explicitly():
    value = torch.ones((3, 5), dtype=torch.float32)
    q, v0, s = encoded(value)
    with pytest.raises(ValueError, match="FP32"):
        rank.gram_eigh(value.double(), 1)
    with pytest.raises(ValueError, match="rank"):
        rank.alternating_projection(v0, q, s, rank=6, iterations=0)
    bad = v0.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        rank.alternating_projection(bad, q, s, rank=1, iterations=0)
    with pytest.raises(ValueError, match="nonnegative"):
        rank.alternating_projection(v0, q, s, rank=1, iterations=-1)
