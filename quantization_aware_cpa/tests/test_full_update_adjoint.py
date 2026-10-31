"""Exact-gradient algorithm fixtures, not VGG16 reconstruction measurements."""

import pytest
import torch
from torch import nn

from qacpa.full_update_adjoint import recomputed_image_gradients
from qacpa.full_update_replay import SGDReplaySpec, differentiable_sgd


class SmallConvFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 2)
        self.fc = nn.Linear(8, 3)
        self.register_buffer("gain", torch.tensor(0.8))

    def forward(self, images):
        return self.fc(torch.tanh(self.conv(images) * self.gain).flatten(1))


def mixed_loss(deltas):
    terms = [((index + 1) * delta.square() + 0.03 * delta).sum()
             for index, delta in enumerate(deltas.values())]
    # Couple tensors too, so the terminal adjoint is not tensor-separable.
    return sum(terms) + 0.7 * sum(delta.mean() for delta in deltas.values()).square()


@pytest.fixture
def case():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(927)
        model = SmallConvFC().double().eval()
        images = torch.randn(3, 1, 3, 3, dtype=torch.float64)
    return model, images, torch.tensor([0, 2, 1])


@pytest.mark.parametrize("steps", [3, 10])
@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_recomputed_gradients_match_full_unroll(case, steps, soft, dtype):
    model, images, hard_targets = case
    model = model.to(dtype=dtype)
    images = images.to(dtype=dtype).requires_grad_(True)
    targets = (torch.tensor([[0.6, 0.1, 0.3], [0.15, 0.65, 0.2], [0.35, 0.4, 0.25]],
                            dtype=dtype, requires_grad=True) if soft else hard_targets)
    batches = ((0, 2), (1,), (2, 0, 1), (0,))
    spec = SGDReplaySpec(0.09, tuple(batches[i % len(batches)] for i in range(steps)), weight_decay=0.13)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.4)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    modes = [module.training for module in model.modules()]
    rng = torch.random.get_rng_state().clone()

    reference_loss = mixed_loss(differentiable_sgd(model, images, targets, spec))
    variables = (images, targets) if soft else (images,)
    reference_gradients = torch.autograd.grad(reference_loss, variables)
    actual_loss, image_gradient, target_gradient = recomputed_image_gradients(model, images, targets, spec, mixed_loss)
    atol, rtol = (2e-7, 2e-5) if dtype == torch.float32 else (2e-14, 1e-10)
    torch.testing.assert_close(actual_loss, reference_loss, atol=0, rtol=0)
    torch.testing.assert_close(image_gradient, reference_gradients[0], atol=atol, rtol=rtol)
    assert image_gradient.norm() > 0 and not image_gradient.requires_grad
    if soft:
        torch.testing.assert_close(target_gradient, reference_gradients[1], atol=atol, rtol=rtol)
        assert target_gradient.norm() > 0 and not target_gradient.requires_grad
    else:
        assert target_gradient is None
    assert not actual_loss.requires_grad
    assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())
    assert all(torch.equal(parameter.grad, torch.full_like(parameter, 0.4)) for parameter in model.parameters())
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert images.grad is None and targets.grad is None


def test_no_decay_and_unused_parameter_match_unroll(case):
    model, images, targets = case
    model.register_parameter("unused", nn.Parameter(torch.ones(2, dtype=images.dtype)))
    images.requires_grad_(True)
    spec = SGDReplaySpec(0.1, ((0,), (1, 2), (0, 2)))
    reference_loss = mixed_loss(differentiable_sgd(model, images, targets, spec))
    reference_gradient, = torch.autograd.grad(reference_loss, images)
    actual_loss, actual_gradient, _ = recomputed_image_gradients(model, images, targets, spec, mixed_loss)
    torch.testing.assert_close(actual_loss, reference_loss, atol=0, rtol=0)
    torch.testing.assert_close(actual_gradient, reference_gradient, atol=2e-14, rtol=1e-10)


def test_constant_loss_produces_zero_gradients(case):
    model, images, targets = case
    spec = SGDReplaySpec(0.1, ((0, 1, 2),))
    loss, gradient, target_gradient = recomputed_image_gradients(
        model, images, targets, spec, lambda deltas: images.new_tensor(4.0))
    assert loss == 4 and torch.equal(gradient, torch.zeros_like(images))
    assert target_gradient is None


def test_linear_loss_expanded_terminal_gradients_are_supported(case):
    model, images, targets = case
    images.requires_grad_(True)
    spec = SGDReplaySpec(0.1, ((0, 1), (2, 0)))
    objective = lambda deltas: sum(delta.sum() for delta in deltas.values())
    expected_loss = objective(differentiable_sgd(model, images, targets, spec))
    expected_gradient, = torch.autograd.grad(expected_loss, images)
    actual_loss, actual_gradient, _ = recomputed_image_gradients(model, images, targets, spec, objective)
    torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-14, rtol=1e-10)


def test_momentum_rejected_explicitly(case):
    model, images, targets = case
    spec = SGDReplaySpec(0.1, ((0, 1, 2),), momentum=0.9)
    with pytest.raises(ValueError, match="requires momentum=0"):
        recomputed_image_gradients(model, images, targets, spec, mixed_loss)


def test_stochastic_custom_forward_rejected_and_rng_preserved(case):
    class Stochastic(SmallConvFC):
        def forward(self, images):
            return super().forward(images) + torch.rand(())

    _, images, targets = case
    model = Stochastic().double().eval()
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="explicit RNG replay"):
        recomputed_image_gradients(model, images, targets, SGDReplaySpec(0.1, ((0,),)), mixed_loss)
    assert torch.equal(rng, torch.random.get_rng_state())
