"""FC-only local SGD with fixed, differentiable feature inputs.

Every local step reuses the same feature matrix and the entire batch. Only the
supplied classifier is registered with the replay engine: a Conv feature model
is never trained, copied, or invoked by this module. The caller computes
``features = frozen_conv(candidate_images)`` and owns that image-to-feature
graph. Quantizer choice belongs to the observation callback, not this module.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

from .full_update_adjoint import recomputed_image_gradients
from .full_update_replay import SGDReplaySpec, differentiable_sgd


class _ClassifierOnly(nn.Module):
    """Register the original classifier under canonical VGG parameter names."""

    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        self.classifier = classifier

    def forward(self, features: Tensor) -> Tensor:
        return self.classifier(features)


def _context(classifier: nn.Module, features: Tensor, steps: int, lr: float):
    if not isinstance(classifier, nn.Module):
        raise TypeError("classifier must be a torch.nn.Module")
    if not isinstance(features, Tensor) or features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("fixed features must have shape (batch_size, feature_dimension)")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if any(isinstance(module, nn.modules.conv._ConvNd) for module in classifier.modules()):
        raise ValueError("FC-only replay must receive the classifier, not a Conv model")
    batch = tuple(range(features.shape[0]))
    spec = SGDReplaySpec(lr=lr, batch_order=(batch,) * steps)
    return _ClassifierOnly(classifier), spec


def fc_sgd_updates(
    classifier: nn.Module,
    features: Tensor,
    targets: Tensor,
    *,
    steps: int,
    lr: float,
    create_graph: bool = True,
    checkpoint_forward: bool = True,
) -> dict[str, Tensor]:
    """Return raw final-minus-initial updates for every classifier parameter.

    Names are prefixed with ``classifier.``; for torchvision VGG16 these are
    ``classifier.0.weight``, ``classifier.0.bias``, etc. Mean cross entropy uses
    hard class indices or probability targets. SGD has zero momentum and weight
    decay, a fresh optimizer, and a repeated full batch. All classifier
    parameters must be trainable and the supplied model mode is preserved.

    ``create_graph=True`` preserves derivatives to features and soft targets,
    but stores a complete FC trajectory. Use ``fc_recomputed_gradients`` for
    the memory-constrained attack. ``False`` returns detached diagnostic deltas.
    Dropout must be inactive, buffers fixed, and activations out of place.
    """
    model, spec = _context(classifier, features, steps, lr)
    return differentiable_sgd(model, features, targets, spec, create_graph=create_graph,
                              checkpoint_forward=checkpoint_forward)


def fc_recomputed_gradients(
    classifier: nn.Module,
    features: Tensor,
    targets: Tensor,
    steps: int,
    lr: float,
    lossfn: Callable[[dict[str, Tensor]], Tensor],
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Return detached update loss, feature gradient, and probability gradient.

    This is exact reverse recomputation of every FC SGD step, without keeping
    the whole trajectory graph. T=10 costs 55 ordinary FC prefix steps and ten
    reverse Hessian-vector steps; Conv computation is outside this recurrence.
    Full classifier states and gradient vectors still need memory.

    ``lossfn`` may depend only on the candidate updates and fixed public
    observations. Compute FIA/TV priors separately. The caller backpropagates
    the returned feature gradient through its frozen Conv graph; any returned
    target gradient is with respect to probabilities, so a softmax Jacobian is
    needed when optimizing logits. Hard labels return ``None`` as the third
    value. The callback sees no private images, labels, or true update.
    """
    model, spec = _context(classifier, features, steps, lr)
    return recomputed_image_gradients(model, features, targets, spec, lossfn)
