"""Low-rank positive-metric FIA cosine, with no observation or model access.

For orthonormal columns V and nonnegative weights w, use
H = epsilon*I + (1-epsilon)*V*diag(w)*V.T, with 0 < epsilon <= 1.
Only B-by-k projections are formed; a d-by-d H is never materialized.
Each metric norm is floored at 1e-8 separately, matching the per-vector norm
convention of torch cosine_similarity.  Flooring the squared norm before its
square root also gives finite derivatives at a zero vector.

An ideal cosine is invariant under global positive scaling of H.  The fixed
norm floor breaks that invariance for sufficiently tiny metric norms.  This
module does not select components, calibrate loss weights, or infer recovery
quality from a spectrum.  H=I controls should use the original FIA loss in the
runner to preserve its original operation order.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor


METRIC_NORM_FLOOR = 1e-8
_DTYPES = (torch.float32, torch.float64)


def _tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or value.layout != torch.strided or value.dtype not in _DTYPES:
        raise ValueError(f"{name} must be a strided float32 or float64 tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def metric_weights(eigenvalues: Tensor, edge: float) -> Tensor:
    """Return max(0, 1-edge/lambda) for finite positive eigenvalues.

    Eigenvalues must be a nonempty vector; edge is a finite nonnegative scalar.
    This explicit heuristic does not identify trustworthy feature directions.
    A mean-matched flat control uses weights.mean().expand_as(weights).
    """
    _tensor("eigenvalues", eigenvalues)
    if eigenvalues.ndim != 1 or eigenvalues.numel() == 0 or not bool((eigenvalues > 0).all()):
        raise ValueError("eigenvalues must be a nonempty vector of strictly positive values")
    if isinstance(edge, bool) or not isinstance(edge, Real) or not math.isfinite(edge) or edge < 0:
        raise ValueError("edge must be a finite nonnegative scalar")
    # edge/lambda can overflow for otherwise valid tiny eigenvalues, but such
    # coordinates are exactly zero after the declared positive-part operation.
    ratio = float(edge) / eigenvalues
    return (1.0 - ratio).clamp_min(0.0)


def spectral_cosine_loss(features: Tensor, target: Tensor, vectors: Tensor,
                         weights: Tensor, epsilon: float) -> Tensor:
    """Return the batch mean of 1-cos_H(features_i, target_i).

    Shapes are features/target [B,d], orthonormal vectors [d,k], weights [k],
    with nonempty B,d,k and k<=d.  All tensors share dtype and device.  Weights
    are nonnegative; values above one are mathematically allowed.  V and w are
    normally fixed public inputs, but the arithmetic retains their graphs.
    Validation never changes caller-owned tensors or random state.
    """
    for name, value in (("features", features), ("target", target),
                        ("vectors", vectors), ("weights", weights)):
        _tensor(name, value)
    if features.ndim != 2 or min(features.shape) == 0 or target.shape != features.shape:
        raise ValueError("features and target must have the same nonempty [B,d] shape")
    if (vectors.ndim != 2 or vectors.shape[0] != features.shape[1] or
            not 0 < vectors.shape[1] <= vectors.shape[0] or weights.shape != (vectors.shape[1],)):
        raise ValueError("vectors must have shape [d,k] and weights shape [k], with 0<k<=d")
    if any(value.dtype != features.dtype or value.device != features.device
           for value in (target, vectors, weights)):
        raise ValueError("all tensors must share dtype and device")
    if not bool((weights >= 0).all()):
        raise ValueError("weights must be nonnegative")
    if (isinstance(epsilon, bool) or not isinstance(epsilon, Real) or
            not math.isfinite(epsilon) or not 0 < epsilon <= 1):
        raise ValueError("epsilon must be a finite scalar with 0 < epsilon <= 1")
    with torch.no_grad():
        k = vectors.shape[1]
        gram = vectors.T @ vectors
        identity = torch.eye(k, dtype=vectors.dtype, device=vectors.device)
        tolerance = 10 * torch.finfo(vectors.dtype).eps * k
        if not torch.allclose(gram, identity, rtol=0.0, atol=tolerance):
            raise ValueError("vectors must have orthonormal columns")

    feature_projection, target_projection = features @ vectors, target @ vectors
    correction = (1.0 - float(epsilon)) * weights
    inner = float(epsilon) * (features * target).sum(dim=1)
    inner = inner + (feature_projection * target_projection * correction).sum(dim=1)
    feature_squared = float(epsilon) * features.square().sum(dim=1)
    feature_squared = feature_squared + (feature_projection.square() * correction).sum(dim=1)
    target_squared = float(epsilon) * target.square().sum(dim=1)
    target_squared = target_squared + (target_projection.square() * correction).sum(dim=1)
    feature_norm = feature_squared.clamp_min(METRIC_NORM_FLOOR**2).sqrt()
    target_norm = target_squared.clamp_min(METRIC_NORM_FLOOR**2).sqrt()
    cosine = (inner / feature_norm) / target_norm
    if not bool(torch.isfinite(cosine).all()):
        raise ValueError("nonfinite spectral cosine arithmetic")
    return (1.0 - cosine).mean()
