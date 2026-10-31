"""External uncentered rank/cell preprocessing for the signed INT4 update.

This module deliberately has no model, private-data, CPA, or FIA dependency.
It operates on a public decoded update and the immutable signed codec cells.
The alternating map is::

    V_0 = dequantize(q, s)
    M_t = best_rank_r(V_t)
    V_(t+1) = project_to_cells(M_t, L(q, s), U(q, s))

``M`` is the numerical rank projection and ``V`` is the cell-feasible matrix;
they are never conflated.  The scientific runner uses uncentered FP32 data and
``rank=16``.  The smaller ranks accepted here are for CPU fixtures only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any

import torch

from quantized_update_cpa import cell_utils


RANK_BACKEND = "gram_eigh"
DEFAULT_CHUNK_ROWS = 64


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash dtype, shape, and contiguous CPU tensor bytes deterministically."""
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise TypeError("value must be a strided torch.Tensor")
    cpu = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(repr(tuple(cpu.shape)).encode("ascii"))
    digest.update(memoryview(cpu.numpy()).cast("B"))
    return digest.hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_matrix(value: torch.Tensor, name: str, *, cpu: bool | None = None) -> None:
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ValueError(f"{name} must be a strided tensor")
    if value.ndim != 2 or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty rank-2 matrix")
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must be FP32; implicit dtype fallback is forbidden")
    if cpu is True and value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if cpu is False and value.device.type not in {"cpu", "cuda"}:
        raise ValueError(f"{name} must be on CPU or CUDA")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite values")


def _require_codes(q: torch.Tensor, shape: tuple[int, int] | None = None) -> None:
    if not isinstance(q, torch.Tensor) or q.device.type != "cpu" or q.dtype != torch.int8:
        raise ValueError("q must be a CPU int8 tensor")
    if q.ndim != 2 or q.numel() == 0:
        raise ValueError("q must be a nonempty rank-2 matrix")
    if shape is not None and tuple(q.shape) != tuple(shape):
        raise ValueError("q shape must match the update matrix")
    if bool(((q < -7) | (q > 7)).any()):
        raise ValueError("q codes must lie in [-7, 7]")


def _require_scale(s: torch.Tensor) -> None:
    if not isinstance(s, torch.Tensor) or s.device.type != "cpu" or s.dtype != torch.float32:
        raise ValueError("s must be a CPU FP32 scalar")
    if s.ndim != 0 or not bool(torch.isfinite(s)) or not bool(s > 0):
        raise ValueError("s must be a finite positive scalar")


def _validate_rank(rank: int, rows: int, cols: int) -> None:
    if type(rank) is not int or rank < 0 or rank > min(rows, cols):
        raise ValueError(f"rank must be an integer in [0, {min(rows, cols)}]")


@torch.no_grad()
def gram_eigh(value: torch.Tensor, rank: int) -> torch.Tensor:
    """Return the uncentered Frobenius best-rank projection.

    The eigendecomposition is only of the row Gram ``V V.T``.  This avoids a
    feature-dimension covariance and does not construct a full feature-space
    projector.  ``torch.linalg.eigh`` is intentionally external to canonical
    CPA, which continues to use its own whitening implementation.
    """
    _require_matrix(value, "value", cpu=False)
    rows, cols = value.shape
    _validate_rank(rank, rows, cols)
    if rank == 0:
        return torch.zeros_like(value)
    gram = value.matmul(value.transpose(0, 1))
    gram = (gram + gram.transpose(0, 1)) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    # eigh returns ascending eigenvalues; for a PSD Gram, the final ``rank``
    # vectors are the largest algebraic eigenvalues.  Ties may choose a
    # different basis, but the reconstructed matrix is the artifact of record.
    basis = eigenvectors[:, -rank:]
    coefficients = basis.transpose(0, 1).matmul(value)
    return basis.matmul(coefficients).contiguous()


def _cell_tables(q: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _require_codes(q)
    _require_scale(s)
    table = torch.arange(-7, 8, dtype=torch.int8)
    return cell_utils.get_cells(table, s, dict(cell_utils.CODEC_METADATA))


def project_to_observed_cells(
    value: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    lower_table: torch.Tensor | None = None,
    upper_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """Coordinatewise projection using fixed observed ``q,s`` cells.

    Bounds are expanded one row chunk at a time.  The operation never changes
    ``q`` or ``s`` and does not impose a candidate-scale anchor.
    """
    _require_matrix(value, "value", cpu=True)
    _require_codes(q, tuple(value.shape))
    _require_scale(s)
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    if lower_table is None or upper_table is None:
        lower_table, upper_table = _cell_tables(q, s)
    if tuple(lower_table.shape) != (15,) or tuple(upper_table.shape) != (15,):
        raise ValueError("cell tables must contain exactly 15 code bounds")
    output = torch.empty_like(value)
    for start in range(0, value.shape[0], chunk_rows):
        stop = min(start + chunk_rows, value.shape[0])
        indices = q[start:stop].to(dtype=torch.int64) + 7
        lower = lower_table[indices]
        upper = upper_table[indices]
        output[start:stop] = cell_utils.project_to_cells(value[start:stop], lower, upper)
    if not bool(torch.isfinite(output).all()):
        raise ValueError("cell projection produced nonfinite values")
    return output.contiguous()


def fixed_cell_metrics(
    value: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    lower_table: torch.Tensor | None = None,
    upper_table: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Report exact fixed-scale code and bounded-cell agreement."""
    _require_matrix(value, "value", cpu=True)
    _require_codes(q, tuple(value.shape))
    _require_scale(s)
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    if lower_table is None or upper_table is None:
        lower_table, upper_table = _cell_tables(q, s)
    mismatch = 0
    violations = 0
    total = value.numel()
    for start in range(0, value.shape[0], chunk_rows):
        stop = min(start + chunk_rows, value.shape[0])
        chunk = value[start:stop]
        indices = q[start:stop].to(dtype=torch.int64) + 7
        lower = lower_table[indices]
        upper = upper_table[indices]
        mismatch += int(cell_utils.check_code_consistency(chunk, q[start:stop], s)["mismatch_count"])
        violations += int(((chunk < lower) | (chunk > upper)).sum().item())
    scale = cell_utils.scale_consistency(value, s)
    return {
        "exact_code_match": mismatch == 0,
        "exact_code_ratio": 1.0 - mismatch / total,
        "code_mismatch_count": mismatch,
        "numel": total,
        "bounded_cell_satisfied": violations == 0,
        "bounded_cell_violation_count": violations,
        "bounded_cell_violation_fraction": violations / total,
        "scale_consistency": scale,
        "scale_is_metric_only": True,
        "codec_scope": "fixed_observed_signed_symmetric_int4_scale",
    }


def _finite_scalar(value: torch.Tensor | float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _norm_record(value: torch.Tensor, denominator: torch.Tensor | None = None, *, zero_reason: str) -> dict[str, Any]:
    numerator = torch.linalg.vector_norm(value)
    n = _finite_scalar(numerator)
    if n is None:
        return {"value": None, "reason": "NONFINITE_NORM"}
    if denominator is None:
        return {"value": n, "reason": None}
    d = _finite_scalar(torch.linalg.vector_norm(denominator))
    if d is None:
        return {"value": None, "reason": "NONFINITE_REFERENCE_NORM"}
    if d == 0.0:
        return {"value": None, "reason": zero_reason}
    ratio = n / d
    return {"value": ratio if math.isfinite(ratio) else None,
            "reason": None if math.isfinite(ratio) else "NONFINITE_RELATIVE_NORM"}


def _matrix_stats(value: torch.Tensor) -> dict[str, Any]:
    norm = _finite_scalar(torch.linalg.vector_norm(value))
    maximum = _finite_scalar(value.abs().max())
    return {
        "l2_norm": norm,
        "l2_norm_reason": None if norm is not None else "NONFINITE_NORM",
        "max_abs": maximum,
        "max_abs_reason": None if maximum is not None else "NONFINITE_MAX_ABS",
        "finite": bool(torch.isfinite(value).all()),
        "shape": list(value.shape),
    }


def _movement(current: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    delta = current - reference
    return {
        "relative_l2": _norm_record(delta, reference, zero_reason="ZERO_REFERENCE_NORM"),
        "max_abs": _finite_scalar(delta.abs().max()),
        "max_abs_reason": None if _finite_scalar(delta.abs().max()) is not None else "NONFINITE_MAX_ABS",
        "changed_coordinate_count": int((current != reference).sum().item()),
    }


def _pair_residual(left: torch.Tensor | None, right: torch.Tensor | None) -> dict[str, Any]:
    if left is None or right is None:
        return {"l2_norm": None, "reason": "NOT_APPLICABLE_INITIAL_RECORD"}
    delta = left - right
    n = _finite_scalar(torch.linalg.vector_norm(delta))
    return {"l2_norm": n, "reason": None if n is not None else "NONFINITE_NORM"}


def _record_initial(v0: torch.Tensor, q: torch.Tensor, s: torch.Tensor, *, rank: int,
                    projection_device: torch.device, chunk_rows: int) -> dict[str, Any]:
    return {
        "iteration": 0,
        "phase": "initial",
        "V_t": {"matrix": _matrix_stats(v0), "fixed_scale": fixed_cell_metrics(
            v0, q, s, chunk_rows=chunk_rows)},
        "M_t": None,
        "V_next": None,
        "movement_from_V0": {"relative_l2": {"value": 0.0, "reason": None},
                              "max_abs": 0.0, "max_abs_reason": None,
                              "changed_coordinate_count": 0},
        "movement_from_previous_V": {"relative_l2": {"value": 0.0, "reason": None},
                                      "max_abs": 0.0, "max_abs_reason": None,
                                      "changed_coordinate_count": 0},
        "pair_residual_Vt_minus_Mt": _pair_residual(None, None),
        "pair_residual_Vnext_minus_Mt": _pair_residual(None, None),
        "rank_target": rank,
        "rank_backend": RANK_BACKEND,
        "projection_dtype": str(v0.dtype),
        "projection_device": str(projection_device),
        "gradient_norm": {"value": None, "status": "NOT_APPLICABLE",
                          "reason": "alternating_projection_has_no_autograd"},
        "nonfinite_status": "PASS",
        "runtime_seconds": 0.0,
    }


def _record_step(iteration: int, before: torch.Tensor, projected: torch.Tensor,
                 after: torch.Tensor, v0: torch.Tensor,
                 q: torch.Tensor, s: torch.Tensor, *, rank: int,
                 projection_device: torch.device, chunk_rows: int,
                 runtime_seconds: float) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "phase": "completed_step",
        "V_t": {"matrix": _matrix_stats(before), "fixed_scale": fixed_cell_metrics(
            before, q, s, chunk_rows=chunk_rows)},
        "M_t": {"matrix": _matrix_stats(projected), "fixed_scale": fixed_cell_metrics(
            projected, q, s, chunk_rows=chunk_rows)},
        "V_next": {"matrix": _matrix_stats(after), "fixed_scale": fixed_cell_metrics(
            after, q, s, chunk_rows=chunk_rows)},
        "movement_from_V0": _movement(after, v0),
        # Consecutive movement is explicitly V_(t+1)-V_t.  Keep movement from
        # V0 as the independent cumulative diagnostic above.
        "movement_from_previous_V": _movement(after, before),
        "pair_residual_Vt_minus_Mt": _pair_residual(before, projected),
        "pair_residual_Vnext_minus_Mt": _pair_residual(after, projected),
        "rank_target": rank,
        "rank_backend": RANK_BACKEND,
        "projection_dtype": str(projected.dtype),
        "projection_device": str(projection_device),
        "gradient_norm": {"value": None, "status": "NOT_APPLICABLE",
                          "reason": "alternating_projection_has_no_autograd"},
        "nonfinite_status": "PASS",
        "runtime_seconds": runtime_seconds,
    }


@dataclass(frozen=True)
class AlternatingProjectionResult:
    """Final external-projection artifacts and scalar diagnostics."""

    initial_v: torch.Tensor
    final_v: torch.Tensor
    final_m: torch.Tensor | None
    diagnostics: tuple[dict[str, Any], ...]
    rank: int
    iterations: int
    projection_device: str
    projection_dtype: str = "torch.float32"


def alternating_projection(
    v0: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    *,
    rank: int,
    iterations: int,
    projection_device: str | torch.device = "cpu",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> AlternatingProjectionResult:
    """Run the prescribed uncentered alternating rank/cell map.

    ``v0``, ``q``, and ``s`` remain CPU FP32/INT8 codec artifacts.  Rank work
    may run on the explicitly requested CPU/CUDA device, but ``M`` is returned
    to CPU before the exact existing cell utility is used.  No backend or dtype
    fallback is performed.
    """
    _require_matrix(v0, "v0", cpu=True)
    _require_codes(q, tuple(v0.shape))
    _require_scale(s)
    if type(iterations) is not int or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    _validate_rank(rank, v0.shape[0], v0.shape[1])
    device = torch.device(projection_device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("projection_device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA projection requested but CUDA is unavailable")
    if device.type == "cuda" and bool(torch.backends.cuda.matmul.allow_tf32):
        raise RuntimeError("CUDA projection requires TF32 to be disabled explicitly")

    lower_table, upper_table = _cell_tables(q, s)
    initial = v0.detach().clone().contiguous()
    value = initial.clone()
    records: list[dict[str, Any]] = [_record_initial(
        value, q, s, rank=rank, projection_device=device, chunk_rows=chunk_rows)]
    final_m: torch.Tensor | None = None
    for step in range(iterations):
        started = time.perf_counter()
        projected_device = gram_eigh(value.to(device=device), rank)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        projected = projected_device.to(device="cpu", dtype=torch.float32).contiguous()
        del projected_device
        after = project_to_observed_cells(
            projected, q, s, chunk_rows=chunk_rows,
            lower_table=lower_table, upper_table=upper_table)
        fit = fixed_cell_metrics(after, q, s, chunk_rows=chunk_rows,
                                 lower_table=lower_table, upper_table=upper_table)
        if (not fit["exact_code_match"] or not fit["bounded_cell_satisfied"]
                or not bool(torch.isfinite(after).all())):
            raise ValueError("cell projection failed fixed-code or finite-value validation")
        elapsed = time.perf_counter() - started
        records.append(_record_step(
            step + 1, value, projected, after, initial, q, s,
            rank=rank, projection_device=device, chunk_rows=chunk_rows,
            runtime_seconds=elapsed))
        value, final_m = after, projected
    return AlternatingProjectionResult(
        initial_v=initial, final_v=value, final_m=final_m,
        diagnostics=tuple(records), rank=rank, iterations=iterations,
        projection_device=str(device),
    )
