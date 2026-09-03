from __future__ import annotations

import math

import torch


def condition_number(matrix: torch.Tensor, *, eps: float = 1.0e-12) -> float:
    values = torch.linalg.svdvals(matrix.detach().float())
    if values.numel() == 0:
        return math.nan
    return float((values.max() / values.min().clamp_min(eps)).cpu())


def column_normalized(matrix: torch.Tensor, *, eps: float = 1.0e-12) -> torch.Tensor:
    return matrix / matrix.norm(dim=0, keepdim=True).clamp_min(eps)


def row_normalized(matrix: torch.Tensor, *, eps: float = 1.0e-12) -> torch.Tensor:
    return matrix / matrix.norm(dim=1, keepdim=True).clamp_min(eps)


def relative_l2(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1.0e-12) -> float:
    return float((a - b).norm().div(b.norm().clamp_min(eps)).detach().cpu())
