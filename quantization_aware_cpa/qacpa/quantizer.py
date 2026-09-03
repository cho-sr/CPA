from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


ROUNDINGS = {"stochastic", "deterministic"}


@dataclass(frozen=True)
class QuantizedObservation:
    Y_hat: torch.Tensor
    delta: torch.Tensor
    bits: int
    rounding: str


def quantize(
    Y: torch.Tensor,
    bits: int,
    rounding: str,
    *,
    generator: torch.Generator | None = None,
) -> QuantizedObservation:
    """Per-row symmetric quantization observation.

    For deterministic rounding, `Y_hat` is the nearest grid dequantization.
    For stochastic rounding, `Y_hat` is a subtractive-dithered observation with
    independent U(-delta/2, delta/2) error. This keeps the exact cell-membership
    gate valid while making the covariance debias term exact.
    """
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D, got shape={tuple(Y.shape)}")
    if bits < 2:
        raise ValueError("bits must be >= 2")
    if rounding not in ROUNDINGS:
        raise ValueError(f"rounding must be one of {sorted(ROUNDINGS)}, got {rounding!r}")

    Y_float = Y.detach()
    levels = 2**int(bits) - 1
    qmax = levels // 2
    r = Y_float.abs().amax(dim=1, keepdim=True)
    delta = 2.0 * r / float(levels)
    nonzero = delta > 0
    delta_safe = torch.where(nonzero, delta, torch.ones_like(delta))

    if rounding == "deterministic":
        q = torch.round(Y_float / delta_safe).clamp(-qmax, qmax)
        Y_hat = q * delta
        Y_hat = torch.where(nonzero, Y_hat, torch.zeros_like(Y_hat))
    else:
        noise = (
            torch.rand(
                Y_float.shape,
                dtype=Y_float.dtype,
                device=Y_float.device,
                generator=generator,
            )
            - 0.5
        ) * delta
        Y_hat = torch.where(nonzero, Y_float + noise, torch.zeros_like(Y_float))

    Y_hat = _project_observation_into_cell(Y_float, Y_hat, delta)
    return QuantizedObservation(Y_hat=Y_hat.to(dtype=Y.dtype), delta=delta.squeeze(1).to(dtype=Y.dtype), bits=bits, rounding=rounding)


def _project_observation_into_cell(
    Y: torch.Tensor,
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Remove sub-ulp boundary overshoots from saturated quantization cells."""
    half_width = delta / 2.0
    diff = Y - Y_hat
    inward = torch.minimum(
        half_width * 1.0e-3,
        torch.full_like(half_width, 1.0e-5),
    )
    target_half_width = (half_width - inward).clamp_min(0.0)
    too_far = diff.abs() > target_half_width
    if not bool(too_far.any()):
        return Y_hat
    boundary = Y - diff.sign() * target_half_width
    return torch.where(too_far, boundary, Y_hat)


def quantization_slack(
    Y: torch.Tensor,
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    return delta[:, None] / 2.0 - (Y - Y_hat).abs()


def assert_cell_membership(
    Y: torch.Tensor,
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
    *,
    atol: float = 1.0e-6,
) -> None:
    slack = quantization_slack(Y, Y_hat, delta)
    min_slack = float(slack.min().detach().cpu())
    if min_slack < -atol:
        raise AssertionError(
            "quantization cell invariant failed: "
            f"min_slack={min_slack:.9g}, atol={atol}"
        )


def hinge_residual(
    predicted: torch.Tensor,
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    return F.relu((predicted - Y_hat).abs() - delta[:, None] / 2.0)


def hinge_loss(
    A: torch.Tensor,
    S: torch.Tensor,
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    return hinge_residual(A @ S, Y_hat, delta).square().mean()


def consistency_rate(
    A: torch.Tensor,
    S: torch.Tensor,
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
    *,
    atol: float = 1.0e-6,
) -> float:
    inside = (A @ S - Y_hat).abs() <= (delta[:, None] / 2.0 + atol)
    return float(inside.float().mean().detach().cpu())


def mean_hinge(
    A: torch.Tensor,
    S: torch.Tensor,
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
) -> float:
    return float(hinge_residual(A @ S, Y_hat, delta).mean().detach().cpu())
