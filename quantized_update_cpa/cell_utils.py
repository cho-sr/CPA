"""CPU FP32 cells for the existing signed, per-tensor, nearest-even codec.

This validates quantization.py's utility; it does not define a future collector
or reinterpret an affine packet. Bounds are *inclusive representable FP32*
endpoints, not real-valued half-step closures. Binary searches over ordered
nonnegative FP32 bit patterns use the actual FP32 divide/round/clamp operation.

The finite cap is the largest finite m for which FP32(m / 7) == observed s.
Thus scale rounding cannot exclude a self-encoded maximum. Only an upper cap
is imposed: projection neither fixes an anchor nor enforces scale equality.
Scale consistency is a separate diagnostic. Zero input keeps the reference's
q=0, s=1 convention; these cells also admit nonzero values of code zero.
"""

from __future__ import annotations

import torch


CODEC_METADATA = {
    "codec": "signed_symmetric_int4_nearest_even_cpu_fp32_utility_v1",
    "provenance": "quantized_update_cpa/quantization.py:quantize_tensor_symmetric",
    "bits": 4,
    "qmin": -7,
    "qmax": 7,
    "scale_scope": "per_parameter_tensor_absmax_scalar",
    "input_dtype": "float32",
    "scale_dtype": "float32",
    "code_dtype": "int8",
    "device": "cpu",
    "rounding": "nearest_even",
    "zero_tensor": "q=0,s=1",
    "scale_operation": "float32(max_abs/7)",
    "code_operation": "round(float32(G/s)).clamp(-7,7).int8",
    "edge_policy": "largest_finite_fp32_absmax_rounding_to_observed_scale",
}

_MAX_FINITE_BITS = 0x7F7FFFFF


class UnsupportedCodecInput(ValueError):
    """An input is outside this utility's explicit CPU FP32 support domain."""


def _float_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise UnsupportedCodecInput(f"{name} must be CPU float32; no implicit cast")
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise UnsupportedCodecInput(f"{name} must be nonempty and finite")


def _scale(s: torch.Tensor) -> None:
    _float_tensor(s, "s")
    if s.ndim != 0 or not bool(s > 0):
        raise UnsupportedCodecInput("s must be a positive scalar; zero-scale underflow unsupported")


def _codes(q: torch.Tensor) -> None:
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    if q.device.type != "cpu" or q.dtype != torch.int8 or q.numel() == 0:
        raise UnsupportedCodecInput("q must be nonempty CPU int8")
    if bool(((q < -7) | (q > 7)).any()):
        raise UnsupportedCodecInput("q must be signed codes in [-7,7]")


def _reference_scale(G: torch.Tensor) -> torch.Tensor:
    max_abs = G.detach().abs().max()
    return max_abs / 7 if max_abs > 0 else torch.ones((), device=G.device)


def _fixed_codes(G: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return torch.round(G / s).clamp(-7, 7).to(torch.int8)


def encode(G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exactly the reference q,s on supported inputs, without eps clamps.

    Nonzero tensors whose FP32 absmax/7 underflows to zero are rejected rather
    than silently repairing the reference's undefined division/cast behavior.
    """
    _float_tensor(G, "G")
    s = _reference_scale(G)
    _scale(s)
    return _fixed_codes(G, s), s


def dequantize(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """FP32 q*s, in the same order and dtype as the reference."""
    _codes(q)
    _scale(s)
    return q.to(dtype=torch.float32) * s


def _from_bits(bits: torch.Tensor) -> torch.Tensor:
    return bits.to(torch.int32).view(torch.float32)


def _finite_cap(s: torch.Tensor) -> tuple[torch.Tensor, int]:
    # Float32 positive finite values are ordered by their positive int32 bits.
    lo, hi = 0, _MAX_FINITE_BITS + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if bool(_from_bits(torch.tensor(mid)) / 7 <= s):
            lo = mid + 1
        else:
            hi = mid
    cap_bits = lo - 1
    cap = _from_bits(torch.tensor(cap_bits))
    if not torch.equal(cap / 7, s):
        raise UnsupportedCodecInput("s cannot arise from a finite FP32 absmax/7")
    return cap, cap_bits


def get_cells(
    q: torch.Tensor, s: torch.Tensor, codec_metadata: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounded exact-code L,U of q's shape; require explicit provenance.

    Ideal boundaries (q +/- .5)*s describe real arithmetic. Here both division
    rounding and nearest-even rounding decide endpoint ownership. At the two
    extreme codes the scale-compatible cap replaces the unbounded clipping
    tail. Every returned endpoint is checked with the fixed-scale encoder;
    adjacent representable numbers must leave its cell or exceed that cap.
    """
    if not isinstance(codec_metadata, dict) or codec_metadata != CODEC_METADATA:
        raise UnsupportedCodecInput("unsupported or ambiguous codec metadata")
    _codes(q)
    _scale(s)
    cap, cap_bits = _finite_cap(s)
    thresholds = torch.arange(1, 8, dtype=torch.int64)
    lo = torch.zeros(7, dtype=torch.int64)
    hi = torch.full((7,), cap_bits + 1, dtype=torch.int64)
    while bool((lo < hi).any()):
        mid = (lo + hi) // 2
        below = _fixed_codes(_from_bits(mid), s).to(torch.int64) < thresholds
        active = lo < hi
        lo = torch.where(active & below, mid + 1, lo)
        hi = torch.where(active & ~below, mid, hi)
    if bool((lo > cap_bits).any()):
        raise UnsupportedCodecInput("some signed codes have no bounded FP32 preimage")
    positive_lower = _from_bits(lo)
    preceding = _from_bits(lo - 1)
    # Index 0..14 corresponds to codes -7..7; negative side follows odd symmetry.
    lower = torch.cat((-torch.cat((preceding[1:], cap.reshape(1))).flip(0),
                       -preceding[:1], positive_lower))
    upper = torch.cat((-positive_lower.flip(0), preceding[:1],
                       preceding[1:], cap.reshape(1)))
    table_codes = torch.arange(-7, 8, dtype=torch.int8)
    if not (torch.equal(_fixed_codes(lower, s), table_codes)
            and torch.equal(_fixed_codes(upper, s), table_codes)):
        raise UnsupportedCodecInput("FP32 runtime does not preserve endpoint code ownership")
    left = torch.nextafter(lower, torch.full_like(lower, -torch.inf))
    right = torch.nextafter(upper, torch.full_like(upper, torch.inf))
    if bool(((left >= -cap) & (_fixed_codes(left, s) == table_codes)).any()
            | ((right <= cap) & (_fixed_codes(right, s) == table_codes)).any()):
        raise UnsupportedCodecInput("endpoint is not maximal in representable FP32 order")
    indices = q.to(torch.int64) + 7
    return lower[indices], upper[indices]


def project_to_cells(G: torch.Tensor, L: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Coordinatewise bounded projection only; no scale/anchor optimization."""
    for name, value in (("G", G), ("L", L), ("U", U)):
        _float_tensor(value, name)
    if G.shape != L.shape or G.shape != U.shape or bool((L > U).any()):
        raise ValueError("G,L,U must have identical shapes and L <= U")
    return torch.maximum(L, torch.minimum(G, U))


def check_code_consistency(G: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> dict:
    """Exact fixed-observation-scale agreement; never recompute s from G."""
    _float_tensor(G, "G")
    _codes(q)
    _scale(s)
    if G.shape != q.shape:
        raise ValueError("G and q must have identical shapes")
    mismatch = int((_fixed_codes(G, s) != q).sum().item())
    return {"all_equal": mismatch == 0, "mismatch_count": mismatch,
            "numel": q.numel(), "mismatch_fraction": mismatch / q.numel()}


def scale_consistency(G: torch.Tensor, s: torch.Tensor) -> dict:
    """Reference candidate scale error, separately from code consistency.

    Python float differences are diagnostic scalars only; scale calculation
    remains FP32. An underflowed candidate is reported explicitly, not clamped.
    """
    _float_tensor(G, "G")
    _scale(s)
    candidate = _reference_scale(G)
    error = abs(float(candidate) - float(s))
    valid = bool(candidate > 0)
    return {"candidate_scale": float(candidate), "observed_scale": float(s),
            "absolute_error": error, "relative_error": error / float(s),
            "exact_equal": torch.equal(candidate, s),
            "candidate_supported": valid,
            "reason": None if valid else "NONZERO_ABSMAX_DIV7_UNDERFLOW"}
