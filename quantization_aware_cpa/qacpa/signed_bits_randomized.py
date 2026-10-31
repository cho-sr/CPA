"""Public bounded-cell samplers for the isolated 7/6/5/4 codec."""
from __future__ import annotations

import hashlib
from typing import Any

import torch

from .signed_bits_observation import SUPPORTED_BITS, codec_metadata, qmax_for_bits


_MAX_FINITE_BITS = 0x7F7FFFFF


class RandomizedBitsError(ValueError):
    pass


def _from_bits(bits: torch.Tensor) -> torch.Tensor:
    return bits.to(torch.int32).view(torch.float32)


def _fixed_codes(value: torch.Tensor, scale: torch.Tensor, qmax: int) -> torch.Tensor:
    return torch.round(value / scale).clamp(-qmax, qmax).to(torch.int8)


def _finite_cap(scale: torch.Tensor, qmax: int) -> tuple[torch.Tensor, int]:
    lo, hi = 0, _MAX_FINITE_BITS + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if bool(_from_bits(torch.tensor(mid, dtype=torch.int32)) / qmax <= scale):
            lo = mid + 1
        else:
            hi = mid
    bits = lo - 1
    cap = _from_bits(torch.tensor(bits, dtype=torch.int32))
    if not torch.equal(cap / qmax, scale):
        raise RandomizedBitsError("observed scale cannot arise from finite FP32 max_abs/qmax")
    return cap, bits


def cell_bounds_table(scale: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = qmax_for_bits(bits)
    if (not isinstance(scale, torch.Tensor) or scale.device.type != "cpu" or scale.dtype != torch.float32
            or scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0)):
        raise RandomizedBitsError("scale must be a positive CPU FP32 scalar")
    cap, cap_bits = _finite_cap(scale, qmax)
    thresholds = torch.arange(1, qmax + 1, dtype=torch.int64)
    lo = torch.zeros(qmax, dtype=torch.int64)
    hi = torch.full((qmax,), cap_bits + 1, dtype=torch.int64)
    while bool((lo < hi).any()):
        mid = (lo + hi) // 2
        below = _fixed_codes(_from_bits(mid), scale, qmax).to(torch.int64) < thresholds
        active = lo < hi
        lo = torch.where(active & below, mid + 1, lo)
        hi = torch.where(active & ~below, mid, hi)
    if bool((lo > cap_bits).any()):
        raise RandomizedBitsError("some code has no bounded FP32 preimage")
    positive_lower = _from_bits(lo)
    preceding = _from_bits(lo - 1)
    lower = torch.cat((-torch.cat((preceding[1:], cap.reshape(1))).flip(0), -preceding[:1], positive_lower))
    upper = torch.cat((-positive_lower.flip(0), preceding[:1], preceding[1:], cap.reshape(1)))
    codes = torch.arange(-qmax, qmax + 1, dtype=torch.int8)
    if not (torch.equal(_fixed_codes(lower, scale, qmax), codes)
            and torch.equal(_fixed_codes(upper, scale, qmax), codes)):
        raise RandomizedBitsError("endpoint ownership check failed")
    left = torch.nextafter(lower, torch.full_like(lower, -torch.inf))
    right = torch.nextafter(upper, torch.full_like(upper, torch.inf))
    if bool((((left >= -cap) & (_fixed_codes(left, scale, qmax) == codes)).any()
             | ((right <= cap) & (_fixed_codes(right, scale, qmax) == codes)).any())):
        raise RandomizedBitsError("endpoint is not maximal in representable order")
    return lower, upper


def sample_matrix(q: torch.Tensor, scale: torch.Tensor, *, bits: int, seed: int,
                  chunk_rows: int = 32) -> tuple[torch.Tensor, dict[str, Any]]:
    qmax = qmax_for_bits(bits)
    if type(seed) is not int or seed < 0:
        raise RandomizedBitsError("seed must be a nonnegative integer")
    if q.ndim != 2 or q.device.type != "cpu" or q.dtype != torch.int8:
        raise RandomizedBitsError("sample q must be a CPU rank-2 int8 tensor")
    if bool((q < -qmax).any()) or bool((q > qmax).any()):
        raise RandomizedBitsError("sample q is outside the selected bit range")
    lower_table, upper_table = cell_bounds_table(scale, bits)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = torch.empty(q.shape, dtype=torch.float32)
    for start in range(0, q.shape[0], chunk_rows):
        stop = min(start + chunk_rows, q.shape[0])
        indices = q[start:stop].to(torch.int16).to(torch.int64) + qmax
        lower, upper = lower_table[indices].double(), upper_table[indices].double()
        noise = torch.rand((stop - start, q.shape[1]), dtype=torch.float64, generator=generator)
        result[start:stop] = (lower + noise * (upper - lower)).to(torch.float32)
    anchor = torch.nonzero(q.abs().reshape(-1) == qmax, as_tuple=False)
    if anchor.numel() == 0:
        raise RandomizedBitsError("public tensor has no extremal anchor")
    flat = int(anchor[0, 0])
    row, col = divmod(flat, q.shape[1])
    cap, _ = _finite_cap(scale, qmax)
    result[row, col] = torch.copysign(cap, q[row, col].float())
    mismatch = _fixed_codes(result, scale, qmax) != q
    if bool(mismatch.any()) or not bool(torch.isfinite(result).all()):
        raise RandomizedBitsError("sample failed exact fixed-scale code consistency")
    if not torch.equal(result.abs().max() / qmax, scale):
        raise RandomizedBitsError("anchor did not preserve observed scale")
    array = result.numpy()
    digest = hashlib.sha256(str(array.dtype).encode('ascii') + str(array.shape).encode('ascii') + memoryview(array).cast('B')).hexdigest()
    return result.contiguous(), {
        "bits": bits, "seed": seed, "chunk_rows": chunk_rows, "generator": "torch.Generator(cpu)",
        "sampler": "uniform_fp64_interpolation_cast_fp32_inclusive_registered_cells",
        "codec_metadata": codec_metadata(bits), "anchor": {"flat_index": flat, "row": row, "column": col,
            "q": int(q[row, col]), "value": float(result[row, col]), "cap": float(cap)},
        "scale": float(scale), "scale_reproduced": True, "code_consistency": {
            "all_equal": True, "mismatch_count": 0, "numel": q.numel(), "mismatch_fraction": 0.0},
        "tensor_sha256": digest,
    }


def align_preserve_norm(center: torch.Tensor, member: torch.Tensor, member_index: int) -> tuple[torch.Tensor, dict[str, Any]]:
    from scipy.optimize import linear_sum_assignment
    if center.shape != member.shape or center.dtype != torch.float32 or member.dtype != torch.float32:
        raise RandomizedBitsError("source alignment shape/dtype mismatch")
    c, m = center.double(), member.double()
    cn, mn = torch.linalg.vector_norm(c, dim=1), torch.linalg.vector_norm(m, dim=1)
    if bool((cn <= 0).any()) or bool((mn <= 0).any()) or not bool(torch.isfinite(cn).all() & torch.isfinite(mn).all()):
        raise RandomizedBitsError("alignment requires finite nonzero rows")
    cosine = (c / cn[:, None]) @ (m / mn[:, None]).T
    rows, cols = linear_sum_assignment((-cosine.abs()).numpy())
    aligned = torch.empty_like(m)
    signs, scores, ratios = [], [], []
    for row, col in zip(rows.tolist(), cols.tolist()):
        sign = 1 if float(torch.dot(c[row], m[col])) >= 0 else -1
        aligned[row] = sign * m[col]
        signs.append(sign); scores.append(float(cosine[row, col].abs())); ratios.append(float(mn[col] / cn[row]))
    aligned = aligned.float().contiguous()
    if not bool(torch.isfinite(aligned).all()):
        raise RandomizedBitsError("aligned source is nonfinite")
    return aligned, {"member_index": member_index, "permutation_member_columns": cols.tolist(), "signs": signs,
                     "matched_abs_cosines": scores, "mean_abs_cosine": sum(scores)/len(scores),
                     "min_abs_cosine": min(scores), "max_abs_cosine": max(scores),
                     "native_to_reference_norm_ratios": ratios, "row_norm_rescaling": False}
