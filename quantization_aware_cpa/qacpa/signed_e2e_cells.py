"""Memory-bounded signed INT4 cell loss for actual image-generated FC updates.

The public signed observation fixes one scalar FP32 scale and one signed code
per FC coordinate.  This module turns those observations into the exact
inclusive FP32 cells supplied by ``quantized_update_cpa.cell_utils`` and
provides a differentiable penalty for a replayed image update.  It does not
project, recode, change the scale, or read private sender data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from quantized_update_cpa import cell_utils
from .signed_baseline_observation import FC_SHAPES, read_bundle


CELL_CHUNK_SIZE = 65_536
FC_NAMES = tuple(sorted(FC_SHAPES))


class CellContractError(ValueError):
    """The signed public observation or a replay tensor violates its contract."""


@dataclass(frozen=True)
class CellTable:
    """One immutable signed-code cell table prepared from a public q/s pair."""

    name: str
    q_cpu: Tensor
    scale_cpu: Tensor
    lower_cpu: Tensor
    upper_cpu: Tensor

    def on(self, device: torch.device) -> "DeviceCellTable":
        return DeviceCellTable(
            name=self.name,
            q=self.q_cpu.to(device=device),
            scale=self.scale_cpu.to(device=device),
            lower=self.lower_cpu.to(device=device),
            upper=self.upper_cpu.to(device=device),
        )


@dataclass(frozen=True)
class DeviceCellTable:
    name: str
    q: Tensor
    scale: Tensor
    lower: Tensor
    upper: Tensor


def _require_cpu_qs(name: str, q: Tensor, scale: Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(q, Tensor) or q.layout != torch.strided or q.device.type != "cpu":
        raise CellContractError(f"{name} q must be a CPU strided tensor")
    if q.dtype != torch.int8 or tuple(q.shape) != shape or not q.numel():
        raise CellContractError(f"{name} q has the wrong dtype or shape")
    if bool(((q < -7) | (q > 7)).any()):
        raise CellContractError(f"{name} q is outside [-7,7]")
    if (not isinstance(scale, Tensor) or scale.layout != torch.strided
            or scale.device.type != "cpu" or scale.dtype != torch.float32
            or scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0)):
        raise CellContractError(f"{name} scale must be a finite positive CPU FP32 scalar")


def prepare_cell_tables(entries: Mapping[str, tuple[Tensor, Tensor]]) -> dict[str, CellTable]:
    """Compute exact CPU FP32 bounds once for all six public FC tensors."""
    if set(entries) != set(FC_NAMES):
        raise CellContractError("the signed cell loss requires exactly all six FC tensors")
    tables: dict[str, CellTable] = {}
    for name in FC_NAMES:
        q, scale = entries[name]
        shape = tuple(FC_SHAPES[name])
        _require_cpu_qs(name, q, scale, shape)
        table_q = torch.arange(-7, 8, dtype=torch.int8)
        lower, upper = cell_utils.get_cells(
            table_q, scale, dict(cell_utils.CODEC_METADATA))
        tables[name] = CellTable(
            name=name,
            q_cpu=q.contiguous(),
            scale_cpu=scale.contiguous(),
            lower_cpu=lower.contiguous(),
            upper_cpu=upper.contiguous(),
        )
    return tables


def load_public_cell_tables(public_observation: str | Path) -> tuple[dict, dict[str, CellTable]]:
    """Read and validate only the public signed q/s bundle."""
    manifest, entries = read_bundle(Path(public_observation))
    return manifest, prepare_cell_tables(entries)


class _ChunkedSignedCellPenalty(torch.autograd.Function):
    """Penalty with a bounded forward and a residual-recomputing backward.

    The q tensor is saved as one byte per coordinate.  Lower and upper bounds
    are gathered from the 15-entry table in each chunk, so no full L/U arrays
    are materialized on the device and no chunk computation graph is retained.
    PyTorch's zero derivative at an inclusive cell endpoint is used explicitly.
    """

    @staticmethod
    def forward(ctx, delta: Tensor, q: Tensor, lower: Tensor, upper: Tensor,
                scale: Tensor, chunk_size: int) -> Tensor:
        if delta.dtype != torch.float32 or delta.layout != torch.strided:
            raise CellContractError("replayed delta must be strided FP32")
        if q.dtype != torch.int8 or q.device != delta.device or q.shape != delta.shape:
            raise CellContractError("cell q must match replayed delta")
        if lower.dtype != torch.float32 or upper.dtype != torch.float32:
            raise CellContractError("cell bounds must be FP32")
        if lower.shape != (15,) or upper.shape != (15,) or lower.device != delta.device:
            raise CellContractError("cell bounds must be device-local 15-entry tables")
        if scale.dtype != torch.float32 or scale.ndim != 0 or scale.device != delta.device:
            raise CellContractError("cell scale must be a device-local FP32 scalar")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise CellContractError("chunk_size must be a positive integer")
        if not bool(torch.isfinite(delta).all()):
            raise CellContractError("replayed delta is nonfinite")
        if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()
                   and torch.isfinite(scale) and scale > 0):
            raise CellContractError("cell bounds or scale are nonfinite/invalid")

        ctx.save_for_backward(delta, q, lower, upper, scale)
        ctx.chunk_size = chunk_size
        flat_delta, flat_q = delta.reshape(-1), q.reshape(-1)
        total = torch.zeros((), dtype=delta.dtype, device=delta.device)
        n = flat_delta.numel()
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            values = flat_delta[start:stop]
            indices = flat_q[start:stop].to(torch.int64) + 7
            lo = lower.index_select(0, indices)
            hi = upper.index_select(0, indices)
            residual = torch.relu(lo - values) + torch.relu(values - hi)
            total = total + torch.sum((residual / scale).square(), dtype=delta.dtype)
        return total / n

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        delta, q, lower, upper, scale = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        flat_delta, flat_q = delta.reshape(-1), q.reshape(-1)
        gradient = torch.empty_like(flat_delta)
        n = flat_delta.numel()
        factor = grad_output.to(dtype=delta.dtype) / (n * scale.square())
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            values = flat_delta[start:stop]
            indices = flat_q[start:stop].to(torch.int64) + 7
            lo = lower.index_select(0, indices)
            hi = upper.index_select(0, indices)
            below = values < lo
            above = values > hi
            residual = torch.where(below, lo - values,
                                   torch.where(above, values - hi,
                                               torch.zeros_like(values)))
            direction = torch.where(below, -torch.ones_like(values),
                                    torch.where(above, torch.ones_like(values),
                                                torch.zeros_like(values)))
            gradient[start:stop] = factor * 2 * residual * direction
        return gradient.reshape(delta.shape), None, None, None, None, None


def signed_cell_penalty(delta: Tensor, table: DeviceCellTable,
                        *, chunk_size: int = CELL_CHUNK_SIZE) -> Tensor:
    """Return one tensor's mean squared outside-cell penalty."""
    if not isinstance(table, DeviceCellTable):
        raise TypeError("table must be a DeviceCellTable")
    if tuple(delta.shape) != tuple(table.q.shape):
        raise CellContractError(f"{table.name} replay shape differs from public q")
    return _ChunkedSignedCellPenalty.apply(
        delta, table.q, table.lower, table.upper, table.scale, chunk_size)


def signed_cell_loss(updates: Mapping[str, Tensor], tables: Mapping[str, DeviceCellTable],
                    *, chunk_size: int = CELL_CHUNK_SIZE) -> Tensor:
    """Return ``(1/6) * sum(mean((outside_distance/s)^2))`` for all six tensors."""
    if set(updates) != set(FC_NAMES) or set(tables) != set(FC_NAMES):
        raise CellContractError("signed cell loss requires exactly all six FC tensors")
    values = []
    for name in FC_NAMES:
        delta = updates[name]
        table = tables[name]
        if (not isinstance(delta, Tensor) or delta.device != table.q.device
                or delta.dtype != torch.float32 or tuple(delta.shape) != tuple(FC_SHAPES[name])):
            raise CellContractError(f"{name} replay delta has the wrong device, dtype or shape")
        values.append(signed_cell_penalty(delta, table, chunk_size=chunk_size))
    return torch.stack(values).mean()


def summarize_signed_update(update: Tensor, table: CellTable,
                            *, chunk_size: int = CELL_CHUNK_SIZE) -> dict:
    """Report exact fixed-s code and bounded-cell fit on a detached CPU FP32 update."""
    if (not isinstance(update, Tensor) or update.device.type != "cpu"
            or update.dtype != torch.float32 or tuple(update.shape) != tuple(table.q_cpu.shape)):
        raise CellContractError(f"{table.name} diagnostic update must be CPU FP32 with public shape")
    if not bool(torch.isfinite(update).all()):
        raise CellContractError(f"{table.name} diagnostic update is nonfinite")
    flat, q = update.reshape(-1), table.q_cpu.reshape(-1)
    lower, upper = table.lower_cpu.reshape(-1), table.upper_cpu.reshape(-1)
    mismatch = 0
    violations = 0
    violation_sum = 0.0
    violation_max = 0.0
    penalty_sum = 0.0
    for start in range(0, flat.numel(), chunk_size):
        stop = min(start + chunk_size, flat.numel())
        values, codes = flat[start:stop], q[start:stop]
        predicted = torch.round(values / table.scale_cpu).clamp(-7, 7).to(torch.int8)
        mismatch += int((predicted != codes).sum())
        indices = codes.to(torch.int64) + 7
        lo, hi = lower[indices], upper[indices]
        distance = torch.relu(lo - values) + torch.relu(values - hi)
        violations += int((distance > 0).sum())
        violation_sum += float(distance.sum(dtype=torch.float64))
        penalty_sum += float(((distance / table.scale_cpu).square()).sum(dtype=torch.float64))
        if distance.numel():
            violation_max = max(violation_max, float(distance.max()))
    n = flat.numel()
    scale = float(table.scale_cpu)
    return {
        "name": table.name,
        "numel": n,
        "fixed_s": scale,
        "code_mismatch_count": mismatch,
        "code_mismatch_fraction": mismatch / n,
        "bounded_cell_violation_count": violations,
        "bounded_cell_violation_fraction": violations / n,
        "mean_violation_in_s": violation_sum / n / scale,
        "max_violation_in_s": violation_max / scale,
        "cell_penalty": penalty_sum / n,
        "candidate_scale": float(update.detach().abs().max() / 7)
        if bool(update.detach().abs().max() > 0) else 1.0,
    }
