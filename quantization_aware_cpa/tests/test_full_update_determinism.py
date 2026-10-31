"""Verify the fixed VGG16 pooling identity without a nondeterministic kernel."""

import pytest
import torch
from torch import nn

from qacpa.full_update_pipeline import VGGFixedSpatialPool


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_fixed_pool_exact_forward_first_and_second_derivatives(dtype, noncontiguous):
    generator = torch.Generator().manual_seed(193)
    source = torch.randn((2, 3, 7, 7), dtype=dtype, generator=generator)
    if noncontiguous:
        source = source.transpose(-1, -2)
        assert not source.is_contiguous()
    direction = torch.linspace(-0.4, 0.8, source.numel(), dtype=dtype).reshape_as(source)
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    outputs = []
    try:
        torch.use_deterministic_algorithms(True)
        for pool in (nn.AdaptiveAvgPool2d((7, 7)), VGGFixedSpatialPool()):
            values = source.detach().clone().requires_grad_(True)
            output = pool(values)
            loss = (output.sin().square() + 0.03 * output.pow(4)).sum()
            first, = torch.autograd.grad(loss, values, create_graph=True)
            second, = torch.autograd.grad(first, values, grad_outputs=direction)
            outputs.append((output.detach(), first.detach(), second.detach()))
        for reference, replacement in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(reference, replacement, atol=0, rtol=0)
    finally:
        torch.use_deterministic_algorithms(previous_determinism, warn_only=previous_warn_only)


@pytest.mark.parametrize("shape", [(2, 3, 6, 7), (2, 3, 7, 8), (2, 3, 14, 14), (3, 7, 7), (2, 3, 1, 7, 7)])
def test_fixed_pool_rejects_shapes_where_adaptive_pooling_is_not_the_contract_identity(shape):
    with pytest.raises(ValueError):
        VGGFixedSpatialPool()(torch.zeros(shape))


def test_contract_224_vgg16_features_have_seven_by_seven_spatial_shape():
    from torchvision.models import vgg16

    with torch.device("meta"):
        model = vgg16(weights=None, init_weights=False)
        features = model.features(torch.empty((16, 3, 224, 224)))
        output = VGGFixedSpatialPool()(features)
    assert tuple(features.shape) == (16, 512, 7, 7)
    assert output is features
