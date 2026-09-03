from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .linalg import row_normalized


@dataclass(frozen=True)
class Alignment:
    true_indices: list[int]
    estimated_indices: list[int]
    unmatched_true_indices: list[int]
    unmatched_estimated_indices: list[int]
    signs: list[float]
    scales: list[float]
    mean_abs_cosine: float
    per_source_abs_cosine: list[float]


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("scipy is required for Hungarian matching.") from exc
    return linear_sum_assignment(cost)


def match_sources_by_abs_cosine(S_true: torch.Tensor, S_est: torch.Tensor) -> Alignment:
    if S_true.ndim != 2 or S_est.ndim != 2 or S_true.shape[1] != S_est.shape[1]:
        raise ValueError(f"source shape mismatch: {tuple(S_true.shape)} vs {tuple(S_est.shape)}")

    similarity = row_normalized(S_true).matmul(row_normalized(S_est).T)
    sim_np = similarity.detach().cpu().numpy()
    rows, cols = _linear_sum_assignment(-np.abs(sim_np))
    matched_true = {int(x) for x in rows}
    matched_est = {int(x) for x in cols}

    signs: list[float] = []
    scales: list[float] = []
    abs_cosines: list[float] = []
    for true_idx, est_idx in zip(rows, cols):
        true_row = S_true[int(true_idx)].double()
        est_row = S_est[int(est_idx)].double()
        denom = est_row.dot(est_row).clamp_min(1.0e-12)
        signed_scale = float((true_row.dot(est_row) / denom).detach().cpu())
        signs.append(1.0 if signed_scale >= 0.0 else -1.0)
        scales.append(abs(signed_scale))
        abs_cosines.append(float(abs(sim_np[int(true_idx), int(est_idx)])))

    return Alignment(
        true_indices=[int(x) for x in rows],
        estimated_indices=[int(x) for x in cols],
        unmatched_true_indices=[idx for idx in range(S_true.shape[0]) if idx not in matched_true],
        unmatched_estimated_indices=[idx for idx in range(S_est.shape[0]) if idx not in matched_est],
        signs=signs,
        scales=scales,
        mean_abs_cosine=float(np.mean(abs_cosines)) if abs_cosines else float("nan"),
        per_source_abs_cosine=abs_cosines,
    )


def align_estimate_to_truth(
    A_est: torch.Tensor,
    S_est: torch.Tensor,
    S_true: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Alignment]:
    alignment = match_sources_by_abs_cosine(S_true, S_est)
    order = torch.tensor(alignment.estimated_indices, dtype=torch.long, device=S_est.device)
    signs = torch.tensor(alignment.signs, dtype=S_est.dtype, device=S_est.device)
    scales = torch.tensor(alignment.scales, dtype=S_est.dtype, device=S_est.device).clamp_min(1.0e-12)
    signed_scales = signs * scales

    S_ordered = S_est.index_select(0, order) * signed_scales[:, None]
    A_ordered = A_est.index_select(1, order) / signed_scales[None, :]
    return A_ordered, S_ordered, alignment


def positive_mean_sign_fix(A_est: torch.Tensor, S_est: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    means = S_est.mean(dim=1)
    signs = torch.where(means < 0, -torch.ones_like(means), torch.ones_like(means))
    return A_est / signs[None, :], S_est * signs[:, None]
