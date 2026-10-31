"""Synthetic CPU metric/derivative gates; no real packet, model, or GPU run."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from qacpa.spectral_fia_metric import metric_weights, spectral_cosine_loss


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def case(dtype=torch.float64):
    generator = torch.Generator().manual_seed(515)
    features = torch.randn(3, 7, generator=generator, dtype=dtype)
    target = torch.randn(3, 7, generator=generator, dtype=dtype)
    vectors, _ = torch.linalg.qr(torch.randn(7, 3, generator=generator, dtype=dtype))
    weights = torch.tensor([.1, .65, .99], dtype=dtype)
    return features, target, vectors, weights


def dense_loss(features, target, vectors, weights, epsilon):
    h = epsilon * torch.eye(features.shape[1], dtype=features.dtype)
    h = h + (1-epsilon) * (vectors * weights) @ vectors.T
    inner = ((features @ h) * target).sum(dim=1)
    fn = ((features @ h) * features).sum(dim=1).clamp_min(1e-16).sqrt()
    tn = ((target @ h) * target).sum(dim=1).clamp_min(1e-16).sqrt()
    return (1-inner/fn/tn).mean()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dense_h_forward_and_feature_target_gradients(dtype):
    features, target, vectors, weights = case(dtype)
    features.requires_grad_()
    target.requires_grad_()
    actual = spectral_cosine_loss(features, target, vectors, weights, .1)
    expected = dense_loss(features, target, vectors, weights, .1)
    actual_grad = torch.autograd.grad(actual, (features, target))
    expected_grad = torch.autograd.grad(expected, (features, target))
    atol, rtol = (2e-7, 2e-6) if dtype == torch.float32 else (2e-15, 2e-13)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)


def test_metric_gradcheck():
    features, target, vectors, weights = case()
    features.requires_grad_()
    target.requires_grad_()
    assert torch.autograd.gradcheck(
        lambda x, y: spectral_cosine_loss(x, y, vectors, weights, .1),
        (features, target), eps=1e-6, atol=2e-6, rtol=2e-5)


def test_equal_weights_flat_equivalence_and_rotated_basis():
    features, target, vectors, weights = case()
    equal = torch.full_like(weights, .6)
    rotation = torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]], dtype=weights.dtype)
    actual = spectral_cosine_loss(features, target, vectors, equal, .1)
    flat = spectral_cosine_loss(features, target, vectors, equal.mean().expand_as(equal), .1)
    rotated = spectral_cosine_loss(features, target, vectors @ rotation, equal, .1)
    torch.testing.assert_close(actual, flat, atol=0, rtol=0)
    torch.testing.assert_close(actual, rotated, atol=2e-15, rtol=0)


@pytest.mark.parametrize("epsilon,zero_weights", [(1., False), (.1, True), (.8, True)])
def test_identity_and_zero_weights_reduce_to_original_cosine(epsilon, zero_weights):
    features, target, vectors, weights = case()
    if zero_weights:
        weights = torch.zeros_like(weights)
    features.requires_grad_()
    actual = spectral_cosine_loss(features, target, vectors, weights, epsilon)
    expected = (1-F.cosine_similarity(features, target, dim=1, eps=1e-8)).mean()
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=0)
    left, = torch.autograd.grad(actual, features)
    right, = torch.autograd.grad(expected, features)
    torch.testing.assert_close(left, right, atol=2e-15, rtol=2e-13)


def test_global_h_scaling_invariance_away_from_floor():
    features, target, vectors, weights = case()
    epsilon, scale = .1, 2.
    scaled_epsilon = scale * epsilon
    scaled_weights = scale*(1-epsilon)/(1-scaled_epsilon)*weights
    actual = spectral_cosine_loss(features, target, vectors, weights, epsilon)
    scaled = spectral_cosine_loss(features, target, vectors, scaled_weights, scaled_epsilon)
    torch.testing.assert_close(actual, scaled, atol=2e-15, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_and_tiny_vectors_have_finite_gradients_and_explicit_floor(dtype):
    features, target, vectors, weights = case(dtype)
    features = torch.zeros_like(features).requires_grad_()
    target = (target*1e-12).requires_grad_()
    loss = spectral_cosine_loss(features, target, vectors, weights, .1)
    assert float(loss.detach()) == 1.
    for gradient in torch.autograd.grad(loss, (features, target)):
        assert torch.isfinite(gradient).all()
    tiny = torch.full((1, 2), 1e-12, dtype=dtype)
    v = torch.eye(2, dtype=dtype)
    w = torch.ones(2, dtype=dtype)
    expected = 1 - tiny.square().sum()/1e-16
    torch.testing.assert_close(spectral_cosine_loss(tiny, tiny, v, w, .1), expected)


def test_metric_weights_and_mean_matched_flat_trace():
    eigenvalues = torch.tensor([10., 5., 2., 1.], dtype=torch.float64)
    weights = metric_weights(eigenvalues, 2.)
    torch.testing.assert_close(weights, torch.tensor([.8, .6, 0., 0.], dtype=eigenvalues.dtype))
    assert weights.dtype == eigenvalues.dtype
    assert weights.sum() == weights.mean().expand_as(weights).sum()
    torch.testing.assert_close(metric_weights(eigenvalues, 0.), torch.ones_like(eigenvalues))


def test_joint_frozen_conv_fc_replay_hybrid_derivatives_match_dense_unroll():
    from qacpa.deterministic_observation import encode_updates, load_observations, observation_loss
    from qacpa.fc_replay import fc_recomputed_gradients, fc_sgd_updates

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(918)
        conv = nn.Sequential(nn.Conv2d(1, 2, 2), nn.Tanh()).double().requires_grad_(False)
        classifier = nn.Sequential(nn.Linear(8, 4), nn.Tanh(), nn.Linear(4, 3)).double().eval()
        images = torch.randn(3, 1, 3, 3, dtype=torch.float64, requires_grad=True)
        logits = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        target = torch.randn(3, 8, dtype=torch.float64)
        vectors, _ = torch.linalg.qr(torch.randn(8, 3, dtype=torch.float64))
    weights = torch.tensor([.2, .6, .95], dtype=torch.float64)
    fixture_updates = {"classifier."+name: torch.linspace(-.03, .04, p.numel()).reshape(p.shape)
                       for name, p in classifier.named_parameters()}
    observations = load_observations(encode_updates(fixture_updates))

    def update_loss(updates):
        return observation_loss(updates, observations, scale_floor=.001, chunk_rows=2)

    def tv(x):
        return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + \
               (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()

    features = conv(images).flatten(1)
    full_q = update_loss(fc_sgd_updates(classifier, features, logits.softmax(1), steps=10, lr=.03))
    full = 1.7*dense_loss(features, target, vectors, weights, .1) + .1*tv(images) + .003*full_q
    expected = torch.autograd.grad(full, (images, logits))

    features = conv(images).flatten(1)
    prior = 1.7*spectral_cosine_loss(features, target, vectors, weights, .1) + .1*tv(images)
    prior_image, = torch.autograd.grad(prior, images, retain_graph=True)
    loss, dz, dp = fc_recomputed_gradients(classifier, features.detach(), logits.detach().softmax(1),
                                         10, .03, update_loss)
    update_image, = torch.autograd.grad(features, images, grad_outputs=dz)
    probabilities = logits.detach().softmax(1)
    actual = (prior_image + .003*update_image,
              .003*probabilities*(dp-(probabilities*dp).sum(1, keepdim=True)))
    torch.testing.assert_close(loss, full_q, atol=1e-12, rtol=1e-12)
    for left, right in zip(actual, expected):
        assert torch.isfinite(left).all() and left.norm() > 0
        torch.testing.assert_close(left, right, atol=2e-10, rtol=2e-9)
    assert all(not p.requires_grad and p.grad is None for p in conv.parameters())


@pytest.mark.parametrize("epsilon", [0., -1., 1.01, float("nan"), float("inf"), True])
def test_bad_epsilon_rejected(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        spectral_cosine_loss(*case(), epsilon)


@pytest.mark.parametrize("mutation", ["nonfinite", "negative_weights", "bad_shape", "different_dtype", "nonorthogonal"])
def test_bad_tensor_inputs_rejected(mutation):
    features, target, vectors, weights = case()
    if mutation == "nonfinite":
        features[0,0] = float("nan")
    elif mutation == "negative_weights":
        weights[0] = -.1
    elif mutation == "bad_shape":
        target = target[:2]
    elif mutation == "different_dtype":
        target = target.float()
    else:
        vectors = vectors*2
    with pytest.raises(ValueError):
        spectral_cosine_loss(features, target, vectors, weights, .1)


@pytest.mark.parametrize("values,edge", [([0., 1.], 1.), ([float("inf")], 1.),
                                         ([1.], -1.), ([1.], float("nan")), ([], 1.)])
def test_bad_weight_inputs_rejected(values, edge):
    with pytest.raises(ValueError):
        metric_weights(torch.tensor(values, dtype=torch.float64), edge)
