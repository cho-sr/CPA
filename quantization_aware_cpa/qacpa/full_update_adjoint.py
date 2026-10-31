"""Exact reverse recomputation for full-model, momentum-free local SGD.

The algorithm trades O(T squared) local-step computation for a single local
step's differentiable graph. It regenerates every needed parameter state from
the fixed starting model, without truncation, a first-order approximation, or
storing the complete training trajectory. Model parameters and the adjoint
vectors still occupy full-model memory; this is not a constant-byte algorithm.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .full_update_replay import SGDReplaySpec, _validate_inputs


def recomputed_image_gradients(
    model: nn.Module,
    images: Tensor,
    targets: Tensor,
    spec: SGDReplaySpec,
    update_loss: Callable[[dict[str, Tensor]], Tensor],
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Return detached update loss and exact image/probability-target gradients.

    ``update_loss`` must be a scalar function of the supplied raw parameter
    deltas and fixed public observation data only. Direct image priors belong
    outside this callable and must have their derivatives added by the caller.
    For floating targets, the final return value differentiates with respect to
    the supplied probabilities, not their upstream logits. The caller can pass
    that derivative through a softmax separately. Hard labels return ``None``.

    Supports mean cross entropy, explicit minibatches, fresh SGD, momentum=0,
    and weight decay. Active randomness, changing buffers, frozen parameters,
    and inplace activations are rejected exactly as in differentiable replay.
    Returns numerical first derivatives, not a graph for further derivatives.
    """
    _validate_inputs(model, images, targets, spec)
    if spec.momentum != 0 or spec.nesterov:
        raise ValueError("reverse recomputation currently requires momentum=0 and no Nesterov")
    if not callable(update_loss):
        raise TypeError("update_loss must be a callable of candidate parameter deltas")

    # The starting values remain fixed. All functional training uses cloned
    # state, so retaining detached references here cannot mutate the model.
    initial = {name: parameter.detach() for name, parameter in model.named_parameters()}
    names = tuple(initial)
    original_buffers = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
    modes = tuple((module, module.training) for module in model.modules())
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    candidate_images = images.detach().clone().requires_grad_(True)
    candidate_targets = targets.detach().clone()
    soft_targets = targets.is_floating_point()
    if soft_targets:
        candidate_targets.requires_grad_(True)

    def forward(batch_images: Tensor, *values: Tensor) -> Tensor:
        parameters = dict(zip(names, values, strict=True))
        buffers = {name: buffer.clone() for name, buffer in original_buffers.items()}
        parameter_versions = tuple(value._version for value in values)
        buffer_versions = {name: buffer._version for name, buffer in buffers.items()}
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = {device: torch.cuda.get_rng_state(device) for device in cuda_devices}
        output = torch.func.functional_call(model, (parameters, buffers), (batch_images,), strict=True)
        if any(module.training != mode for module, mode in modes):
            for module, mode in modes:
                module.training = mode
            raise ValueError("model forward changed module training modes")
        if any(value._version != version for value, version in zip(values, parameter_versions)):
            raise ValueError("model forward mutated a parameter")
        if any(parameters[name] is not value for name, value in zip(names, values, strict=True)):
            raise ValueError("model forward replaced a parameter")
        for name, buffer in buffers.items():
            if buffer._version != buffer_versions[name] or not torch.equal(buffer, original_buffers[name]):
                raise ValueError(f"model forward changed buffer {name}; buffer replay is unsupported")
        if not torch.equal(cpu_rng, torch.random.get_rng_state()) or any(
            not torch.equal(state, torch.cuda.get_rng_state(device)) for device, state in cuda_rng.items()
        ):
            raise ValueError("stochastic model forward requires explicit RNG replay")
        if not isinstance(output, Tensor) or output.ndim != 2 or output.shape[0] != len(batch_images):
            raise ValueError("model must return logits of shape (batch_size, number_of_classes)")
        return output

    def training_loss(parameters: dict[str, Tensor], batch: tuple[int, ...], *, recompute_forward: bool) -> Tensor:
        indices = torch.tensor(batch, dtype=torch.long, device=images.device)
        batch_images = candidate_images.index_select(0, indices)
        batch_targets = candidate_targets.index_select(0, indices)
        values = tuple(parameters.values())
        if recompute_forward:
            logits = checkpoint(forward, batch_images, *values, use_reentrant=False, preserve_rng_state=True)
        else:
            logits = forward(batch_images, *values)
        if soft_targets:
            if batch_targets.shape != logits.shape:
                raise ValueError("soft targets must match the model's number of classes")
            batch_targets = batch_targets.to(dtype=logits.dtype)
        else:
            if bool((batch_targets >= logits.shape[1]).any()):
                raise ValueError("hard target class index exceeds the model's number of classes")
            batch_targets = batch_targets.long()
        return F.cross_entropy(logits, batch_targets, reduction="mean")

    def advance(parameters: dict[str, Tensor], batch: tuple[int, ...]) -> dict[str, Tensor]:
        loss = training_loss(parameters, batch, recompute_forward=False)
        gradients = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True)
        result = {}
        for (name, parameter), gradient in zip(parameters.items(), gradients, strict=True):
            if gradient is None:
                result[name] = parameter
                continue
            direction = torch.add(gradient, parameter, alpha=spec.weight_decay) if spec.weight_decay else gradient
            result[name] = torch.add(parameter, direction, alpha=-spec.lr).detach().requires_grad_(True)
        return result

    def regenerate(step: int) -> dict[str, Tensor]:
        parameters = {name: value.clone().requires_grad_(True) for name, value in initial.items()}
        for batch in spec.batch_order[:step]:
            parameters = advance(parameters, batch)
        return parameters

    def reverse_step(parameters: dict[str, Tensor], batch: tuple[int, ...], adjoint: dict[str, Tensor]):
        values = tuple(parameters.values())
        loss = training_loss(parameters, batch, recompute_forward=True)
        gradients = torch.autograd.grad(loss, values, create_graph=True, allow_unused=True)
        directions, vectors = [], []
        for (name, parameter), gradient in zip(parameters.items(), gradients, strict=True):
            if gradient is None:
                # Actual SGD skips unused parameters, including weight decay.
                continue
            direction = torch.add(gradient, parameter, alpha=spec.weight_decay) if spec.weight_decay else gradient
            if direction.requires_grad:
                directions.append(direction)
                vectors.append(adjoint[name])
        inputs = values + (candidate_images,) + ((candidate_targets,) if soft_targets else ())
        if directions:
            hessian_products = torch.autograd.grad(directions, inputs, grad_outputs=vectors, allow_unused=True)
        else:
            hessian_products = (None,) * len(inputs)
        # All Hessian/cross products are now numerical tensors. Release local
        # gradient outputs before updating the adjoint, and reuse its storage
        # rather than allocating another full model's worth of vectors.
        del gradients, directions, vectors, loss
        for name, product in zip(names, hessian_products[:len(values)], strict=True):
            if product is not None:
                adjoint[name].add_(product, alpha=-spec.lr)
        image_product = hessian_products[len(values)]
        target_product = hessian_products[len(values) + 1] if soft_targets else None
        return adjoint, image_product, target_product

    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.enable_grad():
            final_state = regenerate(len(spec.batch_order))
            deltas = {name: parameter - initial[name] for name, parameter in final_state.items()}
            loss = update_loss(deltas)
            if not isinstance(loss, Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ValueError("update_loss must return one finite scalar tensor")
            detached_loss = loss.detach()
            if loss.requires_grad:
                terminal_gradients = torch.autograd.grad(loss, tuple(final_state.values()), allow_unused=True)
            else:
                terminal_gradients = (None,) * len(final_state)
            # Gradients of a linear loss can be expanded or share storage.
            # Own these numerical vectors before their in-place recurrence.
            adjoint = {name: gradient.detach().clone() if gradient is not None else torch.zeros_like(parameter)
                       for (name, parameter), gradient in zip(final_state.items(), terminal_gradients, strict=True)}
            del final_state, deltas, loss, terminal_gradients

            image_gradient = torch.zeros_like(candidate_images)
            target_gradient = torch.zeros_like(candidate_targets) if soft_targets else None
            for step in reversed(range(len(spec.batch_order))):
                state = regenerate(step)
                adjoint, image_product, target_product = reverse_step(state, spec.batch_order[step], adjoint)
                if image_product is not None:
                    image_gradient.add_(image_product, alpha=-spec.lr)
                if target_product is not None:
                    target_gradient.add_(target_product, alpha=-spec.lr)
                del state, image_product, target_product
            if not bool(torch.isfinite(image_gradient).all()) or (
                target_gradient is not None and not bool(torch.isfinite(target_gradient).all())
            ):
                raise RuntimeError("Nonfinite reverse-recomputation gradient")
            return detached_loss, image_gradient.detach(), target_gradient.detach() if soft_targets else None
    finally:
        for module, mode in modes:
            module.training = mode
