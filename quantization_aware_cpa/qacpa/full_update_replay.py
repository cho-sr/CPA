"""Differentiable, single-client SGD over every model parameter.

This module consumes candidate images and explicitly supplied training context.
It neither loads observations nor accesses victim images or true updates. The
returned tensors are raw ``theta_final - theta_initial`` parameter differences,
without learning-rate normalization or quantization.

Supported context: mean cross entropy, one SGD parameter group, a fresh optimizer
(no initial momentum buffers), explicit minibatches, and a deterministic model
whose buffers do not change. The caller must set the actual training/evaluation
mode beforehand. Active dropout, state-changing buffers, and inplace activation
modules are rejected; they are never silently disabled. Checkpointing saves
forward activations only and does not remove the differentiable SGD trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class SGDReplaySpec:
    """A fresh SGD optimizer and the exact sample indices for each local step.

    ``batch_order=((0, 1), (1, 0))`` performs two mean-loss minibatch steps.
    Repeated indices are allowed and represent repeated samples. Warm-started
    optimizer state and parameter-specific optimizer settings are unsupported.
    """

    lr: float
    batch_order: tuple[tuple[int, ...], ...]
    momentum: float = 0.0
    weight_decay: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False

    def __post_init__(self) -> None:
        for name in ("lr", "momentum", "weight_decay", "dampening"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite nonnegative real number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative real number")
        if not isinstance(self.nesterov, bool):
            raise ValueError("nesterov must be a bool")
        if self.nesterov and (self.momentum <= 0 or self.dampening != 0):
            raise ValueError("Nesterov momentum requires positive momentum and zero dampening")
        if not isinstance(self.batch_order, tuple) or not self.batch_order:
            raise ValueError("batch_order must be a nonempty tuple of minibatch tuples")
        for batch in self.batch_order:
            if not isinstance(batch, tuple) or not batch:
                raise ValueError("each minibatch must be a nonempty tuple of indices")
            if any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in batch):
                raise ValueError("minibatch indices must be nonnegative integers")


def _validate_inputs(model: nn.Module, images: Tensor, targets: Tensor, spec: SGDReplaySpec) -> None:
    if not isinstance(spec, SGDReplaySpec):
        raise TypeError("spec must be an SGDReplaySpec")
    if images.ndim < 2 or images.shape[0] == 0 or not images.is_floating_point():
        raise ValueError("images must be a nonempty floating-point batch")
    if not bool(torch.isfinite(images).all()):
        raise ValueError("images must be finite")
    if max(max(batch) for batch in spec.batch_order) >= len(images):
        raise ValueError("minibatch index exceeds the number of candidate images")
    if targets.device != images.device:
        raise ValueError("targets and images must be on the same device")
    if targets.ndim not in (1, 2) or targets.shape[0] != len(images):
        raise ValueError("targets must have shape (N,) or (N, number_of_classes)")
    if targets.ndim == 1:
        if targets.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("hard targets must contain integer class indices")
        if bool((targets < 0).any()):
            raise ValueError("hard targets must contain nonnegative class indices")
    else:
        if not targets.is_floating_point() or targets.shape[1] < 1:
            raise ValueError("soft targets must contain floating-point class probabilities")
        if not bool(torch.isfinite(targets).all()) or bool((targets < 0).any()):
            raise ValueError("soft targets must contain finite nonnegative probabilities")
        sums = targets.detach().sum(dim=1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5):
            raise ValueError("soft target probabilities must sum to one per image")

    named_parameters = dict(model.named_parameters())
    if not named_parameters or any(not p.requires_grad for p in named_parameters.values()):
        raise ValueError("full-update replay requires all model parameters to be trainable")
    for name, parameter in named_parameters.items():
        if parameter.device != images.device or parameter.dtype != images.dtype:
            raise ValueError(f"parameter {name} must share the images' device and dtype")
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.dropout._DropoutNd) and module.training and module.p > 0:
            raise ValueError(f"active dropout at {name or '<root>'} requires explicit RNG replay")
        if getattr(module, "inplace", False):
            raise ValueError(f"inplace module at {name or '<root>'} is unsupported; use out-of-place activations")
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training and module.track_running_stats:
            raise ValueError(f"changing buffers at {name or '<root>'} are unsupported")


def differentiable_sgd(
    model: nn.Module,
    images: Tensor,
    targets: Tensor,
    spec: SGDReplaySpec,
    *,
    create_graph: bool = True,
    checkpoint_forward: bool = False,
) -> dict[str, Tensor]:
    """Replay all local SGD steps and return every named parameter's raw delta.

    With ``create_graph=True``, all inner updates retain their dependence on
    ``images`` and probability ``targets``. The initial model is fixed public
    context, so its parameters are cloned and detached once. The supplied model,
    parameter gradients, buffers, module modes, and global Torch RNG states are
    preserved. ``create_graph=False`` explicitly returns detached diagnostic
    deltas and must not be used as an image-optimization surrogate.

    The model must remain unchanged until a subsequent backward completes when
    checkpointing is enabled, because recomputation invokes its forward method.
    Arbitrary Python side effects in custom modules are outside this API's scope.
    """
    _validate_inputs(model, images, targets, spec)
    if not isinstance(create_graph, bool) or not isinstance(checkpoint_forward, bool):
        raise ValueError("create_graph and checkpoint_forward must be bools")

    initial = {name: p.detach().clone().requires_grad_(True) for name, p in model.named_parameters()}
    parameters = dict(initial)
    names = tuple(parameters)
    reference_buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    modes = tuple((module, module.training) for module in model.modules())
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []

    def forward(batch_images: Tensor, *values: Tensor) -> Tensor:
        # Parameters are explicit checkpoint inputs, not loop-variable captures.
        # Recomputed forwards therefore use that step's functional parameters.
        local_parameters = dict(zip(names, values, strict=True))
        local_buffers = {name: b.clone() for name, b in reference_buffers.items()}
        parameter_versions = tuple(p._version for p in values)
        buffer_versions = {name: b._version for name, b in local_buffers.items()}
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = {device: torch.cuda.get_rng_state(device) for device in cuda_devices}
        output = torch.func.functional_call(
            model, (local_parameters, local_buffers), (batch_images,), strict=True
        )
        if any(module.training != mode for module, mode in modes):
            for module, mode in modes:
                module.training = mode
            raise ValueError("model forward changed module training modes")
        if any(p._version != version for p, version in zip(values, parameter_versions)):
            raise ValueError("model forward mutated a parameter")
        if any(local_parameters[name] is not value for name, value in zip(names, values, strict=True)):
            raise ValueError("model forward replaced a parameter")
        for name, buffer in local_buffers.items():
            if buffer._version != buffer_versions[name] or not torch.equal(buffer, reference_buffers[name]):
                raise ValueError(f"model forward changed buffer {name}; buffer replay is unsupported")
        if not torch.equal(cpu_rng, torch.random.get_rng_state()) or any(
            not torch.equal(state, torch.cuda.get_rng_state(device)) for device, state in cuda_rng.items()
        ):
            raise ValueError("stochastic model forward requires explicit RNG replay")
        if not isinstance(output, Tensor) or output.ndim != 2 or output.shape[0] != len(batch_images):
            raise ValueError("model must return logits of shape (batch_size, number_of_classes)")
        return output

    momentum_buffers: dict[str, Tensor] = {}
    try:
        # fork_rng also restores global state if an unsupported forward fails.
        with torch.random.fork_rng(devices=cuda_devices), torch.enable_grad():
            for batch in spec.batch_order:
                indices = torch.tensor(batch, dtype=torch.long, device=images.device)
                batch_images = images.index_select(0, indices)
                batch_targets = targets.index_select(0, indices)
                values = tuple(parameters.values())
                if checkpoint_forward:
                    logits = checkpoint(forward, batch_images, *values, use_reentrant=False, preserve_rng_state=True)
                else:
                    logits = forward(batch_images, *values)
                if targets.ndim == 1:
                    if bool((batch_targets >= logits.shape[1]).any()):
                        raise ValueError("hard target class index exceeds the model's number of classes")
                    batch_targets = batch_targets.long()
                else:
                    if batch_targets.shape != logits.shape:
                        raise ValueError("soft targets must match the model's number of classes")
                    batch_targets = batch_targets.to(dtype=logits.dtype)
                loss = F.cross_entropy(logits, batch_targets, reduction="mean")
                gradients = torch.autograd.grad(loss, values, create_graph=create_graph, allow_unused=True)
                updated: dict[str, Tensor] = {}
                for (name, parameter), gradient in zip(parameters.items(), gradients, strict=True):
                    # SGD skips parameters unused by this step, including decay
                    # and momentum. They remain in the complete returned delta.
                    if gradient is None:
                        updated[name] = parameter
                        continue
                    # Keep PyTorch SGD's add(alpha=...) operation order: a
                    # separate multiply/add can alter FP32 updates by an ULP.
                    direction = torch.add(gradient, parameter, alpha=spec.weight_decay) if spec.weight_decay else gradient
                    if spec.momentum:
                        if name not in momentum_buffers:
                            # PyTorch's first buffer is the un-dampened gradient.
                            buffer = direction
                        else:
                            buffer = torch.add(momentum_buffers[name] * spec.momentum, direction,
                                               alpha=1 - spec.dampening)
                        momentum_buffers[name] = buffer if create_graph else buffer.detach()
                        direction = torch.add(direction, buffer, alpha=spec.momentum) if spec.nesterov else buffer
                    next_parameter = torch.add(parameter, direction, alpha=-spec.lr)
                    updated[name] = next_parameter if create_graph else next_parameter.detach().requires_grad_(True)
                parameters = updated
            deltas = {name: parameters[name] - start for name, start in initial.items()}
            return deltas if create_graph else {name: delta.detach() for name, delta in deltas.items()}
    finally:
        for module, mode in modes:
            module.training = mode
