"""FC recurrence and gradient fixtures, not VGG16 reconstruction claims."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from qacpa.fc_replay import fc_recomputed_gradients, fc_sgd_updates


@pytest.fixture
def case():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(618)
        classifier = nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Dropout(0), nn.Linear(4, 3)).double().train()
        features = torch.randn(4, 5, dtype=torch.float64)
    return classifier, features, torch.tensor([0, 2, 1, 0])


def reference_updates(classifier, features, targets, steps, lr):
    model = deepcopy(classifier)
    start = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0, foreach=False)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(features), targets, reduction="mean").backward()
        optimizer.step()
    return {"classifier." + name: parameter.detach() - start[name] for name, parameter in model.named_parameters()}


def objective(updates):
    return sum((index + 1) * delta.square().sum() + 0.04 * delta.sum()
               for index, delta in enumerate(updates.values()))


@pytest.mark.parametrize("steps", [1, 3, 10])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fc_forward_matches_actual_full_batch_sgd_exactly(case, steps, dtype):
    classifier, features, targets = case
    classifier, features = classifier.to(dtype=dtype), features.to(dtype=dtype)
    expected = reference_updates(classifier, features, targets, steps, 0.001)
    for create_graph in (False, True):
        actual = fc_sgd_updates(classifier, features, targets, steps=steps, lr=0.001,
                                 create_graph=create_graph)
        assert set(actual) == set(expected)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        if not create_graph:
            assert all(not value.requires_grad for value in actual.values())


@pytest.mark.parametrize("steps", [3, 10])
@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fc_recomputation_matches_unrolled_feature_and_label_gradients(case, steps, soft, dtype):
    classifier, features, hard_targets = case
    classifier = classifier.to(dtype=dtype)
    features = features.to(dtype=dtype).requires_grad_(True)
    targets = (torch.tensor([[.6, .1, .3], [.2, .5, .3], [.1, .1, .8], [.4, .2, .4]],
                            dtype=dtype, requires_grad=True) if soft else hard_targets)
    full_loss = objective(fc_sgd_updates(classifier, features, targets, steps=steps, lr=.03))
    expected = torch.autograd.grad(full_loss, (features, targets) if soft else (features,))
    loss, feature_gradient, target_gradient = fc_recomputed_gradients(
        classifier, features, targets, steps, .03, objective)
    torch.testing.assert_close(loss, full_loss, atol=0, rtol=0)
    atol, rtol = (2e-8, 2e-5) if dtype == torch.float32 else (2e-14, 1e-10)
    torch.testing.assert_close(feature_gradient, expected[0], atol=atol, rtol=rtol)
    assert feature_gradient.norm() > 0
    assert not loss.requires_grad and not feature_gradient.requires_grad
    if soft:
        torch.testing.assert_close(target_gradient, expected[1], atol=atol, rtol=rtol)
        assert target_gradient.norm() > 0 and not target_gradient.requires_grad
    else:
        assert target_gradient is None


def test_feature_gradient_composes_with_frozen_conv_and_softmax():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(721)
        conv = nn.Conv2d(1, 2, 2).double().requires_grad_(False)
        classifier = nn.Sequential(nn.Linear(8, 4), nn.Tanh(), nn.Linear(4, 3)).double()
        images = torch.randn(3, 1, 3, 3, dtype=torch.float64, requires_grad=True)
        logits = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    full_loss = objective(fc_sgd_updates(classifier, conv(images).flatten(1), logits.softmax(1),
                                        steps=10, lr=.03))
    expected_image_gradient, expected_logit_gradient = torch.autograd.grad(full_loss, (images, logits))
    features, probabilities = conv(images).flatten(1), logits.softmax(1)
    loss, feature_gradient, probability_gradient = fc_recomputed_gradients(
        classifier, features, probabilities, 10, .03, objective)
    actual_image_gradient, = torch.autograd.grad(features, images, grad_outputs=feature_gradient)
    actual_logit_gradient, = torch.autograd.grad(probabilities, logits, grad_outputs=probability_gradient)
    torch.testing.assert_close(loss, full_loss, atol=0, rtol=0)
    torch.testing.assert_close(actual_image_gradient, expected_image_gradient, atol=2e-14, rtol=1e-10)
    torch.testing.assert_close(actual_logit_gradient, expected_logit_gradient, atol=2e-14, rtol=1e-10)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in conv.parameters())


def test_classifier_state_gradients_modes_and_rng_preserved(case):
    classifier, features, targets = case
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.grad = torch.full_like(parameter, .37)
    state = {name: value.clone() for name, value in classifier.state_dict().items()}
    modes = [module.training for module in classifier.modules()]
    rng = torch.random.get_rng_state().clone()
    fc_sgd_updates(classifier, features, targets, steps=3, lr=.001)
    fc_recomputed_gradients(classifier, features, targets, 3, .001, objective)
    assert all(torch.equal(value, state[name]) for name, value in classifier.state_dict().items())
    assert all(torch.equal(parameter.grad, torch.full_like(parameter, .37)) for parameter in classifier.parameters())
    assert [module.training for module in classifier.modules()] == modes
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_invalid_local_step_count_rejected(case, steps):
    classifier, features, targets = case
    with pytest.raises(ValueError, match="positive integer"):
        fc_sgd_updates(classifier, features, targets, steps=steps, lr=.001)


def test_conv_model_and_nonmatrix_features_rejected(case):
    classifier, features, targets = case
    with pytest.raises(ValueError, match="fixed features"):
        fc_sgd_updates(classifier, features.unsqueeze(1), targets, steps=10, lr=.001)
    with pytest.raises(ValueError, match="classifier, not a Conv"):
        fc_sgd_updates(nn.Conv2d(1, 2, 2), features, targets, steps=10, lr=.001)


def test_active_dropout_and_frozen_classifier_rejected(case):
    classifier, features, targets = case
    classifier[2].p = .5
    with pytest.raises(ValueError, match="active dropout"):
        fc_recomputed_gradients(classifier, features, targets, 10, .001, objective)
    classifier[2].p = 0
    classifier[0].weight.requires_grad_(False)
    with pytest.raises(ValueError, match="all model parameters"):
        fc_sgd_updates(classifier, features, targets, steps=10, lr=.001)
