"""CPU composition fixtures for FIA + FC-SGD quantization penalty + TV.

These small networks and synthetic public packets verify derivatives only;
they are not a substitute for the VGG16 pilot or its reconstruction metrics.
"""

import pytest
import torch
from torch import nn

from qacpa.deterministic_observation import encode_updates, load_observations, observation_loss
from qacpa.fc_replay import fc_recomputed_gradients, fc_sgd_updates
from qacpa.full_update_pipeline import normalize_pixels, total_variation
from qacpa.run_deterministic_hybrid import balanced_weight, cosine_loss


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("steps", [3, 10])
def test_separate_hybrid_gradients_equal_one_joint_objective(dtype, steps):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(142)
        conv = nn.Conv2d(3, 2, 2).to(dtype=dtype).requires_grad_(False)
        classifier = nn.Sequential(nn.Linear(8, 5), nn.Tanh(), nn.Linear(5, 3)).to(dtype=dtype)
        pixels = torch.rand(3, 3, 3, 3, dtype=dtype, requires_grad=True)
        logits = torch.randn(3, 3, dtype=dtype, requires_grad=True)
        anchor = torch.rand(3, 8, dtype=dtype) + .1
        # Packet inputs are fixed public tensors, separate from the candidate.
        packet_source = {"classifier." + name: torch.randn(parameter.shape, dtype=torch.float32) * .001
                         for name, parameter in classifier.named_parameters()}
    observations = load_observations(encode_updates(packet_source))
    public_steps = torch.cat([obs.radius[obs.radius > 0].double() * (2 / 15)
                              for obs in observations.values()])
    scale_floor = float(public_steps.median())

    def feature_model(images):
        return conv(normalize_pixels(images)).tanh().flatten(1)

    def update_loss(updates):
        return observation_loss(updates, observations, radius_weight=1., scale_floor=scale_floor)

    features = feature_model(pixels)
    probabilities = logits.softmax(1)
    feature_loss, tv = cosine_loss(features, anchor), total_variation(pixels)
    feature_gradient, = torch.autograd.grad(feature_loss, pixels, retain_graph=True)
    tv_gradient, = torch.autograd.grad(tv, pixels, retain_graph=True)
    q_loss, dz, dp = fc_recomputed_gradients(classifier, features.detach(), probabilities.detach(),
                                             steps, .001, update_loss)
    q_image_gradient, = torch.autograd.grad(features, pixels, grad_outputs=dz)
    lambda_tv = balanced_weight(feature_gradient, tv_gradient, .1)
    lambda_q = balanced_weight(feature_gradient, q_image_gradient, .25)
    composed_images = feature_gradient + lambda_tv * tv_gradient + lambda_q * q_image_gradient
    detached_p = probabilities.detach()
    composed_logits = lambda_q * detached_p * (dp - (detached_p * dp).sum(1, keepdim=True))
    composed_loss = feature_loss.detach() + lambda_tv * tv.detach() + lambda_q * q_loss

    # Independently build a single complete objective with unrolled FC SGD.
    reference_features = feature_model(pixels)
    reference_loss = (cosine_loss(reference_features, anchor) + lambda_tv * total_variation(pixels)
                      + lambda_q * update_loss(fc_sgd_updates(classifier, reference_features, logits.softmax(1),
                                                              steps=steps, lr=.001)))
    reference_images, reference_logits = torch.autograd.grad(reference_loss, (pixels, logits))
    atol, rtol = (3e-8, 3e-5) if dtype == torch.float32 else (2e-13, 1e-10)
    torch.testing.assert_close(composed_loss, reference_loss, atol=0, rtol=0)
    torch.testing.assert_close(composed_images, reference_images, atol=atol, rtol=rtol)
    torch.testing.assert_close(composed_logits, reference_logits, atol=atol, rtol=rtol)
    assert float((lambda_tv * tv_gradient).norm() / feature_gradient.norm()) == pytest.approx(.1, rel=2e-6)
    assert float((lambda_q * q_image_gradient).norm() / feature_gradient.norm()) == pytest.approx(.25, rel=2e-6)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in conv.parameters())


@pytest.mark.parametrize("reference, auxiliary", [
    (torch.zeros(3), torch.ones(3)), (torch.ones(3), torch.zeros(3)),
    (torch.tensor([float("nan")]), torch.ones(1)),
    (torch.ones(1), torch.tensor([float("inf")])),
])
def test_public_gradient_calibration_rejects_degenerate_values(reference, auxiliary):
    with pytest.raises(ValueError, match="finite, nonzero"):
        balanced_weight(reference, auxiliary, .25)
