"""Small-model algorithm fixtures; these do not measure VGG16 attack quality."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from qacpa.full_update_replay import SGDReplaySpec, differentiable_sgd


class SmallConvFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2)
        self.fc = nn.Linear(8, 3)
        self.register_buffer("gain", torch.tensor(0.8))

    def forward(self, images):
        features = torch.tanh(self.conv(images) * self.gain)
        return self.fc(features.flatten(1))


@pytest.fixture
def case():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(421)
        model = SmallConvFC().double().eval()
        images = torch.randn(3, 1, 3, 3, dtype=torch.float64)
    return model, images, torch.tensor([0, 2, 1])


def actual_sgd(model, images, targets, spec):
    victim = deepcopy(model)
    initial = {name: p.detach().clone() for name, p in victim.named_parameters()}
    optimizer = torch.optim.SGD(
        victim.parameters(), lr=spec.lr, momentum=spec.momentum,
        weight_decay=spec.weight_decay, dampening=spec.dampening,
        nesterov=spec.nesterov, foreach=False,
    )
    for batch in spec.batch_order:
        indices = torch.tensor(batch)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(victim(images[indices]), targets[indices], reduction="mean")
        loss.backward()
        optimizer.step()
    return {name: p.detach() - initial[name] for name, p in victim.named_parameters()}


@pytest.mark.parametrize("optimizer", [
    {}, {"weight_decay": 0.13},
    {"momentum": 0.8, "weight_decay": 0.13, "dampening": 0.2},
    {"momentum": 0.8, "weight_decay": 0.13, "nesterov": True},
])
@pytest.mark.parametrize("steps", [1, 2, 3])
def test_every_step_matches_actual_sgd_and_preserves_model(case, optimizer, steps):
    model, images, targets = case
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.42)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    modes = [module.training for module in model.modules()]
    rng = torch.random.get_rng_state().clone()
    spec = SGDReplaySpec(0.07, ((0, 2), (1,), (2, 1, 0))[:steps], **optimizer)
    expected = actual_sgd(model, images, targets, spec)
    for create_graph in (True, False):
        actual = differentiable_sgd(model, images, targets, spec, create_graph=create_graph)
        assert actual.keys() == expected.keys()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=3e-16, rtol=1e-12)
        assert actual["conv.weight"].abs().max() > 0
        assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())
        assert all(torch.equal(p.grad, gradients[name]) for name, p in model.named_parameters())
        assert [module.training for module in model.modules()] == modes
        assert torch.equal(torch.random.get_rng_state(), rng)
        if not create_graph:
            assert all(not delta.requires_grad for delta in actual.values())


def delta_loss(deltas):
    return sum((index + 1) * delta.square().sum() for index, delta in enumerate(deltas.values()))


@pytest.mark.parametrize("optimizer", [
    {}, {"momentum": 0.8, "weight_decay": 0.13, "dampening": 0.2},
    {"momentum": 0.8, "weight_decay": 0.13, "nesterov": True},
])
def test_fp32_sgd_operation_order_is_exact(case, optimizer):
    # Matching add(alpha=...) matters when quantization steps approach an ULP.
    model, images, targets = case
    model, images = model.float(), images.float()
    spec = SGDReplaySpec(0.07, ((0, 1, 2),) * 10, **optimizer)
    actual = differentiable_sgd(model, images, targets, spec)
    expected = actual_sgd(model, images, targets, spec)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


@pytest.mark.parametrize("checkpoint_forward", [False, True])
def test_image_hypergradient_matches_directional_finite_difference(case, checkpoint_forward):
    model, images, targets = case
    images = images.requires_grad_(True)
    spec = SGDReplaySpec(0.12, ((0, 1), (2, 0), (1, 2)), momentum=0.65, weight_decay=0.09, nesterov=True)
    loss = delta_loss(differentiable_sgd(model, images, targets, spec, checkpoint_forward=checkpoint_forward))
    gradient, = torch.autograd.grad(loss, images)
    direction = torch.linspace(-0.7, 0.9, images.numel(), dtype=images.dtype).reshape_as(images)
    direction = direction / direction.norm()
    epsilon = 1e-5
    upper = delta_loss(differentiable_sgd(model, images.detach() + epsilon * direction, targets, spec, create_graph=False))
    lower = delta_loss(differentiable_sgd(model, images.detach() - epsilon * direction, targets, spec, create_graph=False))
    numeric = (upper - lower) / (2 * epsilon)
    analytic = (gradient * direction).sum()
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    torch.testing.assert_close(analytic, numeric, atol=2e-10, rtol=2e-6)


def test_soft_labels_and_checkpoint_match_gradients(case):
    model, images, _ = case
    images = images.requires_grad_(True)
    label_logits = torch.tensor([[0.4, -0.3, 1.1], [0.5, 0.1, -0.7], [-0.2, 0.4, 0.1]], dtype=images.dtype, requires_grad=True)
    spec = SGDReplaySpec(0.08, ((0, 2), (2, 1), (1, 0)), momentum=0.7, dampening=0.1, weight_decay=0.08)
    results = []
    gradients = []
    for checkpoint_forward in (False, True):
        probabilities = label_logits.softmax(dim=1)
        deltas = differentiable_sgd(model, images, probabilities, spec, checkpoint_forward=checkpoint_forward)
        expected = actual_sgd(model, images.detach(), probabilities.detach(), spec)
        for name in expected:
            torch.testing.assert_close(deltas[name], expected[name], atol=3e-16, rtol=1e-12)
        gradients.append(torch.autograd.grad(delta_loss(deltas), (images, label_logits)))
        results.append(deltas)
    for name in results[0]:
        torch.testing.assert_close(results[0][name], results[1][name], atol=1e-14, rtol=1e-12)
    for original, checked in zip(*gradients):
        assert torch.isfinite(original).all() and original.norm() > 0
        torch.testing.assert_close(original, checked, atol=1e-13, rtol=1e-11)


@pytest.mark.parametrize("arguments", [
    {"lr": -0.1}, {"lr": float("nan")}, {"momentum": -0.2},
    {"weight_decay": -0.2}, {"dampening": float("inf")},
    {"nesterov": True}, {"nesterov": True, "momentum": 0.3, "dampening": 0.1},
    {"batch_order": ()}, {"batch_order": ((),)},
    {"batch_order": ((-1,),)}, {"batch_order": ((True,),)},
])
def test_rejects_invalid_optimizer_context(arguments):
    options = {"lr": 0.1, "batch_order": ((0,),)}
    options.update(arguments)
    with pytest.raises(ValueError):
        SGDReplaySpec(**options)


def test_rejects_frozen_parameters_and_invalid_data(case):
    model, images, targets = case
    spec = SGDReplaySpec(0.1, ((0, 1),))
    model.conv.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="all model parameters"):
        differentiable_sgd(model, images, targets, spec)
    model.conv.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="index exceeds"):
        differentiable_sgd(model, images, targets, SGDReplaySpec(0.1, ((3,),)))
    with pytest.raises(ValueError, match="sum to one"):
        differentiable_sgd(model, images, torch.full((3, 3), 0.2, dtype=images.dtype), spec)
    with pytest.raises(ValueError, match="number of classes"):
        differentiable_sgd(model, images, torch.full((3, 4), 0.25, dtype=images.dtype), spec)
    with pytest.raises(ValueError, match="number of classes"):
        differentiable_sgd(model, images, torch.tensor([7, 0, 0]), spec)


def test_active_dropout_and_inplace_activations_rejected(case):
    _, images, targets = case
    for activation, message in [(nn.Dropout(0.5), "active dropout"), (nn.ReLU(inplace=True), "inplace")]:
        model = nn.Sequential(nn.Flatten(), activation, nn.Linear(9, 3)).double().train()
        with pytest.raises(ValueError, match=message):
            differentiable_sgd(model, images, targets, SGDReplaySpec(0.1, ((0,),)))


def test_changing_buffers_rejected_and_rng_restored(case):
    class Stateful(SmallConvFC):
        def forward(self, images):
            self.gain.add_(torch.rand(()))
            return super().forward(images)

    _, images, targets = case
    model = Stateful().double().eval()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="changed buffer"):
        differentiable_sgd(model, images, targets, SGDReplaySpec(0.1, ((0,),)))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())


def test_functional_randomness_rejected_and_rng_restored(case):
    class Stochastic(SmallConvFC):
        def forward(self, images):
            return super().forward(images) + torch.randn((), dtype=images.dtype)

    _, images, targets = case
    model = Stochastic().double().eval()
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="explicit RNG replay"):
        differentiable_sgd(model, images, targets, SGDReplaySpec(0.1, ((0,),)))
    assert torch.equal(rng, torch.random.get_rng_state())


def test_inactive_dropout_and_frozen_batchnorm_buffers_supported(case):
    _, images, targets = case
    model = nn.Sequential(nn.BatchNorm2d(1), nn.Flatten(), nn.Dropout(0.5), nn.Linear(9, 3)).double().eval()
    spec = SGDReplaySpec(0.1, ((0, 1), (2,)))
    actual = differentiable_sgd(model, images, targets, spec)
    expected = actual_sgd(model, images, targets, spec)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=3e-16, rtol=1e-12)
    model[0].train()
    with pytest.raises(ValueError, match="changing buffers"):
        differentiable_sgd(model, images, targets, spec)


def test_unused_parameter_follows_sgd_skip_semantics(case):
    model, images, targets = case
    model.register_parameter("unused", nn.Parameter(torch.ones(2, dtype=images.dtype)))
    spec = SGDReplaySpec(0.1, ((0, 1), (2,)), momentum=0.6, weight_decay=0.2)
    actual = differentiable_sgd(model, images, targets, spec)
    expected = actual_sgd(model, images, targets, spec)
    assert torch.equal(actual["unused"], torch.zeros_like(actual["unused"]))
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=3e-16, rtol=1e-12)
