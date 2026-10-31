"""Public-only INT8 bounded-cell sampler and blind source alignment."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import torch

from .signed_int8_observation import CODEC_METADATA, QMAX, QMIN

_MAX_FINITE_BITS = 0x7F7FFFFF


class RandomizedINT8Error(ValueError):
    pass


def _from_bits(bits: torch.Tensor) -> torch.Tensor:
    return bits.to(torch.int32).view(torch.float32)


def _fixed_codes(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.round(value / scale).clamp(QMIN, QMAX).to(torch.int8)


def _finite_cap(scale: torch.Tensor) -> tuple[torch.Tensor, int]:
    lo, hi = 0, _MAX_FINITE_BITS + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if bool(_from_bits(torch.tensor(mid, dtype=torch.int32)) / QMAX <= scale):
            lo = mid + 1
        else:
            hi = mid
    bits = lo - 1
    cap = _from_bits(torch.tensor(bits, dtype=torch.int32))
    if not torch.equal(cap / QMAX, scale):
        raise RandomizedINT8Error("observed scale cannot arise from finite FP32 max_abs/127")
    return cap, bits


def cell_bounds_table(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact inclusive FP32 endpoint table for all 255 signed codes."""
    if (not isinstance(scale, torch.Tensor) or scale.device.type != "cpu"
            or scale.dtype != torch.float32 or scale.ndim != 0
            or not bool(torch.isfinite(scale)) or not bool(scale > 0)):
        raise RandomizedINT8Error("scale must be a positive CPU FP32 scalar")
    cap, cap_bits = _finite_cap(scale)
    thresholds = torch.arange(1, QMAX + 1, dtype=torch.int64)
    lo = torch.zeros(QMAX, dtype=torch.int64)
    hi = torch.full((QMAX,), cap_bits + 1, dtype=torch.int64)
    while bool((lo < hi).any()):
        mid = (lo + hi) // 2
        below = _fixed_codes(_from_bits(mid), scale).to(torch.int64) < thresholds
        active = lo < hi
        lo = torch.where(active & below, mid + 1, lo)
        hi = torch.where(active & ~below, mid, hi)
    if bool((lo > cap_bits).any()):
        raise RandomizedINT8Error("some INT8 code has no bounded FP32 preimage")
    positive_lower = _from_bits(lo)
    preceding = _from_bits(lo - 1)
    lower = torch.cat((-torch.cat((preceding[1:], cap.reshape(1))).flip(0),
                       -preceding[:1], positive_lower))
    upper = torch.cat((-positive_lower.flip(0), preceding[:1], preceding[1:], cap.reshape(1)))
    codes = torch.arange(QMIN, QMAX + 1, dtype=torch.int8)
    if not (torch.equal(_fixed_codes(lower, scale), codes)
            and torch.equal(_fixed_codes(upper, scale), codes)):
        raise RandomizedINT8Error("INT8 endpoint ownership check failed")
    left = torch.nextafter(lower, torch.full_like(lower, -torch.inf))
    right = torch.nextafter(upper, torch.full_like(upper, torch.inf))
    if bool((((left >= -cap) & (_fixed_codes(left, scale) == codes)).any()
             | ((right <= cap) & (_fixed_codes(right, scale) == codes)).any())):
        raise RandomizedINT8Error("INT8 endpoint is not maximal in representable order")
    return lower, upper


def get_cells(q: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (not isinstance(q, torch.Tensor) or q.device.type != "cpu"
            or q.dtype != torch.int8 or q.numel() == 0
            or bool((q < QMIN).any()) or bool((q > QMAX).any())):
        raise RandomizedINT8Error("q must be nonempty CPU int8 in [-127,127]")
    lower, upper = cell_bounds_table(scale)
    index = q.to(torch.int16).to(torch.int64) - QMIN
    return lower[index], upper[index]


def code_consistency(value: torch.Tensor, q: torch.Tensor, scale: torch.Tensor) -> dict[str, Any]:
    if value.shape != q.shape or value.dtype != torch.float32 or q.dtype != torch.int8:
        raise RandomizedINT8Error("sample and q shapes/dtypes differ")
    mismatch = int((_fixed_codes(value, scale) != q).sum())
    return {"all_equal": mismatch == 0, "mismatch_count": mismatch,
            "numel": q.numel(), "mismatch_fraction": mismatch / q.numel()}


def _tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    array = value.numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def sample_matrix(q: torch.Tensor, scale: torch.Tensor, *, seed: int,
                  chunk_rows: int = 32) -> tuple[torch.Tensor, dict[str, Any]]:
    """Sample one matrix in bounded chunks with one independent CPU generator."""
    if type(seed) is not int or seed < 0:
        raise RandomizedINT8Error("seed must be a nonnegative integer")
    if q.ndim != 2 or q.device.type != "cpu" or q.dtype != torch.int8:
        raise RandomizedINT8Error("sample q must be a CPU rank-2 int8 tensor")
    lower_table, upper_table = cell_bounds_table(scale)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = torch.empty(q.shape, dtype=torch.float32)
    for start in range(0, q.shape[0], chunk_rows):
        stop = min(start + chunk_rows, q.shape[0])
        indices = q[start:stop].to(torch.int16).to(torch.int64) - QMIN
        lower = lower_table[indices].double()
        upper = upper_table[indices].double()
        noise = torch.rand((stop - start, q.shape[1]), dtype=torch.float64, generator=generator)
        result[start:stop] = (lower + noise * (upper - lower)).to(torch.float32)
    anchor = torch.nonzero(q.abs().reshape(-1) == QMAX, as_tuple=False)
    if anchor.numel() == 0:
        raise RandomizedINT8Error("public INT8 tensor has no extremal anchor")
    flat = int(anchor[0, 0])
    row, col = divmod(flat, q.shape[1])
    cap, _ = _finite_cap(scale)
    result[row, col] = torch.copysign(cap, q[row, col].float())
    consistency = code_consistency(result, q, scale)
    if consistency["all_equal"] is not True:
        raise RandomizedINT8Error("sample failed exact fixed-scale code consistency")
    if not bool(torch.isfinite(result).all()):
        raise RandomizedINT8Error("sample contains nonfinite values")
    max_abs = result.abs().max()
    scale_reproduced = torch.equal(max_abs / QMAX, scale)
    if not scale_reproduced:
        raise RandomizedINT8Error("anchor did not preserve the observed scale")
    meta = {
        "seed": seed, "chunk_rows": chunk_rows, "generator": "torch.Generator(cpu)",
        "sampler": "uniform_fp64_interpolation_cast_fp32_inclusive_INT8_cells",
        "codec_metadata": CODEC_METADATA, "anchor": {"flat_index": flat, "row": row, "column": col,
            "q": int(q[row, col]), "value": float(result[row, col]), "cap": float(cap)},
        "scale": float(scale), "scale_reproduced": scale_reproduced,
        "code_consistency": consistency, "tensor_sha256": _tensor_sha256(result),
    }
    return result.contiguous(), meta


def align_and_average(center: torch.Tensor, members: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
    """Blind abs-cosine Hungarian alignment, row-norm scaling, fixed average."""
    from scipy.optimize import linear_sum_assignment
    if (not isinstance(center, torch.Tensor) or center.ndim != 2 or center.dtype != torch.float32
            or not bool(torch.isfinite(center).all()) or len(members) != 8):
        raise RandomizedINT8Error("alignment requires finite FP32 center and exactly eight members")
    n = center.shape[0]
    center64 = center.double()
    center_norm = torch.linalg.vector_norm(center64, dim=1)
    if bool((center_norm <= 0).any()) or not bool(torch.isfinite(center_norm).all()):
        raise RandomizedINT8Error("center contains zero/nonfinite source rows")
    aligned: list[torch.Tensor] = []
    records = []
    for index, member in enumerate(members):
        if member.shape != center.shape or member.dtype != torch.float32 or not bool(torch.isfinite(member).all()):
            raise RandomizedINT8Error(f"invalid member {index}")
        member64 = member.double()
        member_norm = torch.linalg.vector_norm(member64, dim=1)
        if bool((member_norm <= 0).any()) or not bool(torch.isfinite(member_norm).all()):
            raise RandomizedINT8Error(f"member {index} contains zero/nonfinite source rows")
        cosine = (center64 / center_norm[:, None]) @ (member64 / member_norm[:, None]).T
        rows, cols = linear_sum_assignment((-cosine.abs()).numpy())
        signs = []
        transformed = torch.empty_like(member64)
        scores = []
        for row, col in zip(rows.tolist(), cols.tolist()):
            dot = torch.dot(center64[row], member64[col])
            sign = 1.0 if float(dot) >= 0 else -1.0
            signs.append(sign)
            scores.append(float(cosine[row, col].abs()))
            transformed[row] = center_norm[row] * sign * member64[col] / member_norm[col]
        if not bool(torch.isfinite(transformed).all()):
            raise RandomizedINT8Error(f"member {index} alignment is nonfinite")
        aligned.append(transformed)
        records.append({"member_index": index, "permutation_member_columns": cols.tolist(),
                        "signs": [int(x) for x in signs], "mean_abs_cosine": sum(scores) / len(scores),
                        "min_abs_cosine": min(scores), "max_abs_cosine": max(scores)})
    aggregate = (sum((value.double() for value in aligned), start=torch.zeros_like(center64)) / 8).to(torch.float32)
    target = aggregate.abs().contiguous()
    if not bool(torch.isfinite(aggregate).all()) or not torch.equal(target, aggregate.abs()):
        raise RandomizedINT8Error("aggregate/target is nonfinite or target is not exact abs")
    return aggregate, {"reference": "center_fixed_public_S_hat", "matching": "Hungarian_max_sum_abs_cosine",
                       "cosine_dtype": "float64", "row_norm_matching": True, "member_count": 8,
                       "records": records, "aggregate_dtype": "FP64_sum_div8_cast_FP32",
                       "aggregate_sha256": _tensor_sha256(aggregate),
                       "target_z_sha256": _tensor_sha256(target), "target_z_exact_abs": True}

