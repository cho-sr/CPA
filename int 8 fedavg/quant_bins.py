"""Exact bounded int8 bin utilities for client q,s experiments."""

from __future__ import annotations

from typing import Tuple

import numpy as np


QMIN = -127
QMAX = 127
LEVELS = QMAX - QMIN + 1


def validate_codes(codes: np.ndarray, *, qmin: int = QMIN, qmax: int = QMAX) -> None:
    if np.any(codes < qmin) or np.any(codes > qmax):
        raise ValueError(f"codes must be in [{qmin}, {qmax}]")


def bin_bounds(
    codes: np.ndarray,
    scale: float,
    max_abs: float,
    *,
    qmin: int = QMIN,
    qmax: int = QMAX,
    validate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return exact bounded nearest-rounding bins for signed symmetric int8.

    Interior codes use [(q-.5)s, (q+.5)s]. Saturated codes are bounded by the
    known source range implied by scale=max_abs/127:
    q=+127 -> [126.5s, 127s], q=-127 -> [-127s, -126.5s].
    """
    codes_i = np.asarray(codes, dtype=np.int8)
    if validate:
        validate_codes(codes_i, qmin=qmin, qmax=qmax)

    scale_f = float(scale)
    max_abs_f = float(max_abs)
    if max_abs_f == 0.0 or scale_f == 0.0:
        zeros = np.zeros(codes_i.shape, dtype=np.float32)
        return zeros, zeros
    if scale_f < 0.0 or max_abs_f < 0.0:
        raise ValueError("scale and max_abs must be non-negative")

    code_f = codes_i.astype(np.float32)
    lower = (code_f - 0.5) * scale_f
    upper = (code_f + 0.5) * scale_f
    lower = np.maximum(lower, -max_abs_f).astype(np.float32, copy=False)
    upper = np.minimum(upper, max_abs_f).astype(np.float32, copy=False)
    return lower, upper


def naive_dequantize(codes: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(codes, dtype=np.int8).astype(np.float32) * float(scale)


def randomized_dequantize(
    codes: np.ndarray,
    scale: float,
    max_abs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    lower, upper = bin_bounds(codes, scale, max_abs)
    width = upper - lower
    draws = rng.random(size=lower.shape).astype(np.float32, copy=False)
    return lower + draws * width


def assert_within_bins(
    samples: np.ndarray,
    codes: np.ndarray,
    scale: float,
    max_abs: float,
    *,
    atol: float = 1e-7,
) -> None:
    lower, upper = bin_bounds(codes, scale, max_abs)
    values = np.asarray(samples, dtype=np.float32)
    below = values < (lower - atol)
    above = values > (upper + atol)
    if np.any(below) or np.any(above):
        raise ValueError(
            "randomized samples left their quantization bins: "
            f"below={int(np.count_nonzero(below))}, above={int(np.count_nonzero(above))}"
        )
