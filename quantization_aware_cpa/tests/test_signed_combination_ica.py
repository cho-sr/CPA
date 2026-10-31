"""Focused CPU contracts for the adopted two-stage public ICA separator."""
from __future__ import annotations

import inspect

import pytest
import torch

import qacpa.signed_combination_ica as ica


def test_fastica_uses_rowwise_derivative_broadcast():
    torch.manual_seed(7)
    W = torch.tensor([[1.0, 0.2, -0.1], [-0.3, 1.1, 0.4], [0.5, -0.2, 0.9]])
    Xw = torch.randn(3, 37)
    actual, a = ica._fastica_update(W, Xw)
    g = torch.tanh(W @ Xw)
    expected_pre = (g @ Xw.T) / Xw.shape[1] - a[:, None] * W
    old_pre = (g @ Xw.T) / Xw.shape[1] - a * W
    torch.testing.assert_close(actual, ica.symmetric_decorrelation(expected_pre))
    assert float((a[:, None] * W - a * W).abs().max()) > 1e-5
    assert not torch.allclose(expected_pre, old_pre)


def test_whitening_is_cpu_row_centered_and_rejects_bad_rank_or_values():
    torch.manual_seed(11)
    X = torch.randn(3, 32)
    whitening = ica.whiten_observation(X, 3)
    assert whitening.source_device == "cpu"
    assert whitening.input_shape == (3, 32)
    assert whitening.mu.shape == (3, 1)
    assert whitening.K.shape == (3, 3)
    assert whitening.Xw.shape == (3, 32)
    torch.testing.assert_close(whitening.mu, X.mean(dim=1, keepdim=True))
    assert torch.isfinite(whitening.Xw).all()

    dependent = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    with pytest.raises(ica.ICAError, match="rank"):
        ica.whiten_observation(dependent, 2)
    with pytest.raises(ica.ICAError, match="nonfinite"):
        ica.whiten_observation(torch.tensor([[1.0, float("nan")], [0.0, 1.0]]), 2)
    with pytest.raises(ica.ICAError, match="FP32"):
        ica.whiten_observation(X.double(), 3)


def test_first_step_convergence_is_accepted(monkeypatch):
    X = torch.randn(2, 20)
    whitening = ica.whiten_observation(X, 2)
    config = ica.CombinationICAConfig(fast_max_iter=1, fast_tolerance=1e-6,
                                      refinement_iterations=1)

    def fixed_update(W, Xw):
        return W.clone(), torch.ones(W.shape[0])

    monkeypatch.setattr(ica, "_fastica_update", fixed_update)
    result = ica.fastica_initialize(whitening, config)
    assert result.converged is True
    assert result.iterations == 1
    assert result.residual == 0.0


def test_max_iteration_nonconvergence_is_reported_with_last_finite_iterate(monkeypatch):
    whitening = ica.whiten_observation(torch.randn(2, 20), 2)
    config = ica.CombinationICAConfig(fast_max_iter=1, refinement_iterations=1)

    def nonconverging_update(W, Xw):
        return torch.zeros_like(W), torch.ones(W.shape[0])

    monkeypatch.setattr(ica, "_fastica_update", nonconverging_update)
    result = ica.fastica_initialize(whitening, config)
    assert result.converged is False
    assert result.iterations == 1
    assert result.residual == 1.0
    assert torch.isfinite(result.W).all()


def test_whitening_object_is_shared_between_fastica_and_refinement(monkeypatch):
    torch.manual_seed(13)
    seen = []
    config = ica.CombinationICAConfig(refinement_iterations=1)

    def fake_fast(whitening, config):
        seen.append(("fast", id(whitening)))
        return ica.FastICAResult(torch.eye(2), True, 1, 0.0,
                                 config.fast_max_iter, config.fast_tolerance)

    def fake_refine(whitening, fast, config):
        seen.append(("refine", id(whitening)))
        s = torch.ones(2, 20)
        return ica.RefinementResult(torch.eye(2), s, s.abs(), 1,
                                    {"total": 0.0}, [])

    monkeypatch.setattr(ica, "fastica_initialize", fake_fast)
    monkeypatch.setattr(ica, "refine_from_fastica", fake_refine)
    result = ica.separate_observation(torch.randn(2, 20), 2, config=config)
    assert [stage for stage, _ in seen] == ["fast", "refine"]
    assert seen[0][1] == seen[1][1] == id(result.whitening)
    assert torch.equal(result.target_z, result.S_hat.abs())


def test_stable_logcosh_matches_direct_moderate_values_and_handles_large_values():
    values = torch.tensor([-3.0, -0.25, 0.0, 0.25, 3.0], requires_grad=True)
    stable = ica.stable_logcosh_plus_eps(values)
    direct = torch.log(torch.cosh(values) + ica.EPS)
    torch.testing.assert_close(stable, direct, atol=1e-6, rtol=1e-6)
    stable.sum().backward()
    assert torch.isfinite(values.grad).all()
    assert torch.isfinite(ica.stable_logcosh_plus_eps(torch.tensor([-1000.0, 1000.0]))).all()


def test_prior_negentropy_has_an_active_gradient_and_defaults_are_registered():
    config = ica.CombinationICAConfig()
    assert config.negentropy_weight == pytest.approx(0.0935827857091549)
    assert config.decorrelation_weight == pytest.approx(0.2513674104161625)
    assert config.sign_weight == pytest.approx(0.05811961446881231)
    assert config.l1_weight == pytest.approx(17.94138146247377)
    assert config.refinement_lr == 0.001
    assert config.refinement_momentum == 0.0
    assert config.refinement_weight_decay == 0.0
    whitening = ica.whiten_observation(torch.randn(2, 24), 2)
    W = torch.eye(2, requires_grad=True)
    losses = ica.prior_losses(W, whitening, config)
    assert losses["negentropy"].requires_grad
    losses["total"].backward()
    assert torch.isfinite(W.grad).all()
    assert float(config.negentropy_weight) > 0


def test_two_stage_separator_is_finite_and_target_is_exact_abs_in_native_order():
    torch.manual_seed(23)
    independent = torch.stack([
        torch.rand(192) * 2.0 - 1.0,
        torch.sign(torch.randn(192)) * torch.sqrt(torch.rand(192) + 0.05),
        torch.randn(192).clamp(-2.0, 2.0),
    ])
    mixing = torch.tensor([[1.0, 0.25, -0.15], [0.1, 0.9, 0.2], [-0.2, 0.15, 1.1]])
    observed = (mixing @ independent).float()
    config = ica.CombinationICAConfig(fast_max_iter=40, refinement_iterations=8)
    result = ica.separate_observation(observed, 3, config=config)
    assert result.S_hat.dtype == torch.float32
    assert result.S_hat.is_contiguous()
    assert result.S_hat.shape == observed.shape
    assert torch.isfinite(result.S_hat).all()
    assert torch.equal(result.target_z, result.S_hat.abs())
    assert result.metadata["native_estimated_order"] is True
    assert result.metadata["quality"] == "NOT_EVALUATED"
    assert result.metadata["private_data"]["truth_loaded"] is False


def test_core_interface_has_no_truth_or_private_inputs():
    signature = inspect.signature(ica.separate_observation)
    names = set(signature.parameters)
    assert names == {"X", "source_count", "config", "device"}
    assert not names.intersection({"truth", "labels", "private", "images", "model", "generator"})
    assert "from .ica_ga" not in inspect.getsource(ica)
