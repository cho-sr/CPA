"""Public signed-update feature separation using the adopted ICA-GA stages.

Only an observed matrix is accepted here.  The implementation keeps the
reference ICA-GA whitening and four prior terms, but fixes its row-wise
FastICA broadcast, shares one whitening object between stages, uses stable
``log(cosh(x) + eps)``, and accepts valid convergence on the first iteration.
No images, labels, true sources, model, generator, or evaluation callback is
available to this tensor core.

The algebra is adapted from ICA-GA-main/ICA-GA-main/attacks/feature_separation
under the MIT notice in ``ICA_GA_MIT_NOTICE.txt``.  The imported coefficients
are source defaults; they are not a quality claim for the signed INT4 packet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


ICA_GA_METHOD = "INT4_combination_ica"
ICA_GA_SOURCE = "ICA-GA-main/ICA-GA-main/attacks/feature_separation/{fast_ica,gradient_descent_ica,combination_ica}.py"
EPS = 1e-10
DEFAULT_NEGENTROPY = 0.0935827857091549
DEFAULT_DECORRELATION = 0.2513674104161625
DEFAULT_SIGN = 0.05811961446881231
DEFAULT_L1 = 17.94138146247377


class ICAError(ValueError):
    """Invalid public matrix or nonfinite ICA state."""


@dataclass(frozen=True)
class CombinationICAConfig:
    """Explicit numerical settings for the two-stage public separator."""

    fast_max_iter: int = 200
    fast_tolerance: float = 1e-6
    refinement_iterations: int = 1000
    refinement_lr: float = 0.001
    refinement_momentum: float = 0.0
    refinement_weight_decay: float = 0.0
    negentropy_weight: float = DEFAULT_NEGENTROPY
    decorrelation_weight: float = DEFAULT_DECORRELATION
    sign_weight: float = DEFAULT_SIGN
    l1_weight: float = DEFAULT_L1
    eps: float = EPS
    rank_floor: float = 1e-12

    def __post_init__(self) -> None:
        integer_fields = ("fast_max_iter", "refinement_iterations")
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ICAError(f"{name} must be a positive integer")
        positive_fields = ("fast_tolerance", "refinement_lr", "eps", "rank_floor")
        for name in positive_fields:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
                raise ICAError(f"{name} must be a finite positive number")
        for name in ("refinement_momentum", "refinement_weight_decay",
                     "negentropy_weight", "decorrelation_weight", "sign_weight", "l1_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ICAError(f"{name} must be a finite nonnegative number")

    def as_dict(self) -> dict[str, Any]:
        return {
            "fast_max_iter": self.fast_max_iter,
            "fast_tolerance": self.fast_tolerance,
            "refinement_iterations": self.refinement_iterations,
            "refinement_lr": self.refinement_lr,
            "refinement_momentum": self.refinement_momentum,
            "refinement_weight_decay": self.refinement_weight_decay,
            "negentropy_weight": self.negentropy_weight,
            "decorrelation_weight": self.decorrelation_weight,
            "sign_weight": self.sign_weight,
            "l1_weight": self.l1_weight,
            "eps": self.eps,
            "rank_floor": self.rank_floor,
        }


@dataclass(frozen=True)
class WhitenedObservation:
    """One immutable whitening basis shared by FastICA and refinement."""

    mu: Tensor
    K: Tensor
    Xw: Tensor
    input_shape: tuple[int, int]
    source_count: int
    source_device: str = "cpu"

    def on(self, device: torch.device | str) -> "WhitenedObservation":
        target = torch.device(device)
        return WhitenedObservation(
            mu=self.mu.to(target), K=self.K.to(target), Xw=self.Xw.to(target),
            input_shape=self.input_shape, source_count=self.source_count,
            source_device=target.type,
        )


@dataclass(frozen=True)
class FastICAResult:
    W: Tensor
    converged: bool
    iterations: int
    residual: float
    max_iter: int
    tolerance: float


@dataclass(frozen=True)
class RefinementResult:
    W: Tensor
    S_hat: Tensor
    target_z: Tensor
    iterations: int
    final_losses: dict[str, float]
    log: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SeparationResult:
    S_hat: Tensor
    target_z: Tensor
    W_fast: Tensor
    W_final: Tensor
    whitening: WhitenedObservation
    fast: FastICAResult
    refinement: RefinementResult
    metadata: dict[str, Any]


def tensor_sha256(value: Tensor) -> str:
    """Hash dtype, shape and contiguous CPU tensor bytes."""
    import numpy as np

    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def tensor_stats(value: Tensor) -> dict[str, float]:
    """Return finite descriptive statistics for an emitted CPU tensor."""
    detached = value.detach().float().cpu()
    return {
        "min": float(detached.min()),
        "max": float(detached.max()),
        "mean": float(detached.mean()),
        "std_population": float(detached.std(unbiased=False)),
    }


def _require_observation(X: Tensor, source_count: int) -> Tensor:
    if not isinstance(X, Tensor) or X.layout != torch.strided or X.ndim != 2:
        raise ICAError("X must be a strided rank-2 tensor with shape [m,d]")
    if X.dtype != torch.float32:
        raise ICAError(f"X must use FP32, got {X.dtype}")
    if type(source_count) is not int or source_count <= 0:
        raise ICAError("source_count must be a positive integer")
    m, d = X.shape
    if d < 2:
        raise ICAError("X needs at least two columns for covariance denominator d-1")
    if source_count > min(m, d):
        raise ICAError(f"source_count={source_count} exceeds min(X.shape)={min(m, d)}")
    if not bool(torch.isfinite(X).all()):
        raise ICAError("X contains nonfinite values")
    return X.detach().to(device="cpu", dtype=torch.float32).contiguous()


def whiten_observation(X: Tensor, source_count: int, *, rank_floor: float = EPS) -> WhitenedObservation:
    """Compute the ICA-GA CPU whitening algebra exactly once."""
    X_cpu = _require_observation(X, source_count)
    if not math.isfinite(float(rank_floor)) or rank_floor <= 0:
        raise ICAError("rank_floor must be a finite positive number")
    m, d = X_cpu.shape
    mu = X_cpu.mean(dim=1, keepdim=True)
    centered = X_cpu - mu
    covariance = centered @ centered.T / (d - 1)
    if not bool(torch.isfinite(covariance).all()):
        raise ICAError("observation covariance is nonfinite")
    eigenvalues_complex, eigenvectors_complex = torch.linalg.eig(covariance)
    eigenvalues = eigenvalues_complex.real.float()
    eigenvectors = eigenvectors_complex.real.float()
    indices = torch.topk(eigenvalues.abs(), source_count).indices
    selected = eigenvalues.abs()[indices]
    if not bool(torch.isfinite(selected).all()) or bool((selected <= rank_floor).any()):
        raise ICAError("observation has insufficient usable covariance rank for source_count")
    lamb_inv_sqrt = torch.diag(1.0 / (torch.sqrt(selected) + EPS)).float()
    K = (lamb_inv_sqrt @ eigenvectors.T[indices]).float()
    Xw = K @ centered
    for name, value in (("mu", mu), ("K", K), ("Xw", Xw)):
        if not bool(torch.isfinite(value).all()):
            raise ICAError(f"whitening output {name} is nonfinite")
    if tuple(K.shape) != (source_count, m) or tuple(Xw.shape) != (source_count, d):
        raise ICAError("whitening output shape is invalid")
    return WhitenedObservation(mu=mu, K=K, Xw=Xw, input_shape=(m, d),
                               source_count=source_count, source_device="cpu")


def symmetric_decorrelation(W: Tensor) -> Tensor:
    """Compute the finite-positive-floor symmetric orthogonalization."""
    if W.ndim != 2 or W.shape[0] != W.shape[1] or W.dtype != torch.float32:
        raise ICAError("W must be square FP32 for symmetric decorrelation")
    if not bool(torch.isfinite(W).all()):
        raise ICAError("W is nonfinite before symmetric decorrelation")
    values, vectors = torch.linalg.eigh(W @ W.T)
    values = torch.clamp(values, min=torch.finfo(W.dtype).tiny)
    result = (vectors * (1.0 / torch.sqrt(values)).unsqueeze(0)) @ vectors.T @ W
    if not bool(torch.isfinite(result).all()):
        raise ICAError("symmetric decorrelation produced nonfinite W")
    return result


def _fastica_update(W: Tensor, Xw: Tensor) -> tuple[Tensor, Tensor]:
    """One parallel FastICA update with the corrected row broadcast."""
    if W.shape[1] != Xw.shape[0]:
        raise ICAError("FastICA W/Xw shape mismatch")
    d = Xw.shape[1]
    U = W @ Xw
    g = torch.tanh(U)
    a = torch.mean(1.0 - g * g, dim=1)
    # a is one derivative mean per W row; a[:, None] is essential.
    Wnew = symmetric_decorrelation((g @ Xw.T) / d - a[:, None] * W)
    return Wnew, a


def fastica_initialize(whitening: WhitenedObservation, config: CombinationICAConfig,
                       *, device: torch.device | str | None = None) -> FastICAResult:
    """Run parallel/symmetric fixed-point FastICA from an identity matrix."""
    prepared = whitening if device is None else whitening.on(device)
    n = prepared.source_count
    W = symmetric_decorrelation(torch.eye(n, dtype=torch.float32, device=prepared.Xw.device))
    residual = math.inf
    converged = False
    iterations = 0
    for iteration in range(1, config.fast_max_iter + 1):
        Wnew, _ = _fastica_update(W, prepared.Xw)
        dot = torch.sum(Wnew * W, dim=1)
        residual_tensor = torch.max(torch.abs(torch.abs(dot) - 1.0))
        if not bool(torch.isfinite(residual_tensor)):
            raise ICAError(f"FastICA nonfinite convergence residual at iteration {iteration}")
        residual = float(residual_tensor)
        W = Wnew
        iterations = iteration
        if residual < config.fast_tolerance:
            converged = True
            break
    if not bool(torch.isfinite(W).all()):
        raise ICAError("FastICA returned a nonfinite W")
    return FastICAResult(W=W.detach(), converged=converged, iterations=iterations,
                         residual=residual, max_iter=config.fast_max_iter,
                         tolerance=config.fast_tolerance)


def stable_logcosh_plus_eps(value: Tensor, eps: float = EPS) -> Tensor:
    """Stable algebraic equivalent of ``log(cosh(value) + eps)``."""
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ICAError("eps must be finite and positive")
    absolute = value.abs()
    log_cosh = absolute + torch.log1p(torch.exp(-2.0 * absolute)) - math.log(2.0)
    result = torch.logaddexp(log_cosh, value.new_tensor(math.log(eps)))
    if not bool(torch.isfinite(result).all()):
        raise ICAError("stable logcosh produced a nonfinite value")
    return result


def prior_losses(W: Tensor, whitening: WhitenedObservation,
                 config: CombinationICAConfig) -> dict[str, Tensor]:
    """Return all four active ICA-GA prior terms and their weighted total."""
    if W.shape != (whitening.source_count, whitening.source_count):
        raise ICAError("refinement W shape does not match source_count")
    Wbar = W / (torch.linalg.vector_norm(W, dim=1, keepdim=True) + config.eps)
    T = Wbar @ whitening.Xw
    S = T + Wbar @ whitening.K @ whitening.mu
    Lne = -(stable_logcosh_plus_eps(T, config.eps).mean(dim=1) ** 2).mean()
    identity = torch.eye(whitening.source_count, dtype=W.dtype, device=W.device)
    Ldecor = ((Wbar @ Wbar.T).abs() - identity).square().sum()
    Lsign = torch.minimum(F.relu(-S).norm(dim=1), F.relu(S).norm(dim=1)).mean()
    Ll1 = S.abs().mean()
    total = (config.negentropy_weight * Lne + config.decorrelation_weight * Ldecor
             + config.sign_weight * Lsign + config.l1_weight * Ll1)
    values = {"negentropy": Lne, "decorrelation": Ldecor, "sign": Lsign,
              "l1": Ll1, "total": total}
    if not all(bool(torch.isfinite(value)) for value in values.values()):
        raise ICAError("ICA prior produced a nonfinite loss")
    return values


def _recover_sources(W: Tensor, whitening: WhitenedObservation, eps: float) -> Tensor:
    Wbar = W / (torch.linalg.vector_norm(W, dim=1, keepdim=True) + eps)
    result = Wbar @ whitening.Xw + Wbar @ whitening.K @ whitening.mu
    if not bool(torch.isfinite(result).all()):
        raise ICAError("recovered signed sources are nonfinite")
    return result


def refine_from_fastica(whitening: WhitenedObservation, fast: FastICAResult,
                        config: CombinationICAConfig) -> RefinementResult:
    """Run the exact 1000-step plain-SGD prior refinement from FastICA W."""
    W = fast.W.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD([W], lr=config.refinement_lr,
                                momentum=config.refinement_momentum,
                                weight_decay=config.refinement_weight_decay)
    log: list[dict[str, Any]] = []
    for iteration in range(1, config.refinement_iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = prior_losses(W, whitening, config)
        losses["total"].backward()
        if W.grad is None or not bool(torch.isfinite(W.grad).all()):
            raise ICAError(f"ICA prior gradient is nonfinite at iteration {iteration}")
        optimizer.step()
        log.append({"iteration": iteration, **{name: float(value.detach()) for name, value in losses.items()},
                    "gradient_norm": float(W.grad.detach().norm())})
    with torch.no_grad():
        final_losses = {name: float(value.detach()) for name, value in prior_losses(W, whitening, config).items()}
        final_W = W.detach().clone()
        S_hat = _recover_sources(final_W, whitening, config.eps).detach().contiguous()
        target_z = S_hat.abs().detach().contiguous()
    return RefinementResult(W=final_W, S_hat=S_hat.cpu(), target_z=target_z.cpu(),
                            iterations=config.refinement_iterations,
                            final_losses=final_losses, log=log)


def separate_observation(X: Tensor, source_count: int, *, config: CombinationICAConfig | None = None,
                         device: torch.device | str = "cpu") -> SeparationResult:
    """Run FastICA then prior refinement using one shared whitening object."""
    config = config or CombinationICAConfig()
    whitening_cpu = whiten_observation(X, source_count, rank_floor=config.rank_floor)
    whitening = whitening_cpu.on(device)
    # ``whitening`` is transferred once and the same object is passed to both
    # stages; FastICA must not silently build a second basis.
    fast = fastica_initialize(whitening, config)
    refinement = refine_from_fastica(whitening, fast, config)
    if not torch.equal(refinement.target_z, refinement.S_hat.abs()):
        raise ICAError("target_z is not the exact elementwise absolute value of S_hat")
    metadata = {
        "method": ICA_GA_METHOD,
        "source": ICA_GA_SOURCE,
        "input_shape": list(whitening.input_shape),
        "source_count": source_count,
        "native_estimated_order": True,
        "whitening": {"algorithm": "row_centered_covariance_torch.linalg.eig_top_abs_eigenvalues",
                       "computed_once_on": "cpu", "shared_between_stages": True,
                       "mu_shape": list(whitening.mu.shape), "K_shape": list(whitening.K.shape),
                       "Xw_shape": list(whitening.Xw.shape), "requested_device": str(device),
                       "dtype": "float32", "rank_floor": config.rank_floor},
        "fastica": {"algorithm": "parallel_symmetric_fixed_point_logcosh_alpha1",
                    "corrected_derivative_broadcast": "a[:,None] * W",
                    "converged": fast.converged, "iterations": fast.iterations,
                    "residual": fast.residual, "max_iter": fast.max_iter,
                    "tolerance": fast.tolerance},
        "refinement": {"optimizer": "SGD", "iterations": refinement.iterations,
                       "lr": config.refinement_lr, "momentum": config.refinement_momentum,
                       "weight_decay": config.refinement_weight_decay, "scheduler": None,
                       "active_terms": ["negentropy", "decorrelation", "sign", "l1"],
                       "weights": {"negentropy": config.negentropy_weight,
                                   "decorrelation": config.decorrelation_weight,
                                   "sign": config.sign_weight, "l1": config.l1_weight},
                       "stable_logcosh": True},
        "outputs": {"S_hat_dtype": "float32", "S_hat_shape": list(refinement.S_hat.shape),
                    "S_hat_sha256": tensor_sha256(refinement.S_hat),
                    "target_z_sha256": tensor_sha256(refinement.target_z),
                    "S_hat_stats": tensor_stats(refinement.S_hat),
                    "target_z_stats": tensor_stats(refinement.target_z),
                    "target_z_exact_abs_S_hat": True,
                    "S_hat_finite": bool(torch.isfinite(refinement.S_hat).all()),
                    "target_z_finite": bool(torch.isfinite(refinement.target_z).all()),
                    "target_z_nonnegative": bool((refinement.target_z >= 0).all())},
        "quality": "NOT_EVALUATED",
        "private_data": {"truth_loaded": False, "labels_loaded": False, "generator_used": False},
    }
    return SeparationResult(S_hat=refinement.S_hat, target_z=refinement.target_z,
                            W_fast=fast.W.detach().cpu(), W_final=refinement.W.detach().cpu(),
                            whitening=whitening, fast=fast, refinement=refinement,
                            metadata=metadata)


def loss_dict_to_float(losses: dict[str, Tensor]) -> dict[str, float]:
    return {name: float(value.detach()) for name, value in losses.items()}
