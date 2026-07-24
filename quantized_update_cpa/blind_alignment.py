"""Blind ensemble alignment and evaluation-only oracle assignment.

The public functions used by the attack path deliberately accept only
recovered sources.  Ground-truth embeddings are accepted solely by the
explicitly named evaluation helper.
"""

from __future__ import annotations

from typing import Iterable, Literal, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _flat_unit_rows(values: np.ndarray | torch.Tensor, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(torch.as_tensor(values).detach().cpu(), dtype=np.float64)
    array = array.reshape(array.shape[0], -1)
    return array / (np.linalg.norm(array, axis=1, keepdims=True) + eps)


def align_member_to_reference(
    reference_sources: np.ndarray | torch.Tensor,
    member_sources: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, dict[str, float]]:
    """Permutation/sign-align one recovered member to another without oracle data."""
    reference = np.asarray(torch.as_tensor(reference_sources).detach().cpu())
    member = np.asarray(torch.as_tensor(member_sources).detach().cpu())
    if reference.shape != member.shape:
        raise ValueError(f"Ensemble member shape mismatch: {reference.shape} vs {member.shape}")
    similarity = _flat_unit_rows(reference) @ _flat_unit_rows(member).T
    rows, cols = linear_sum_assignment(-np.abs(similarity))
    aligned = np.empty_like(reference)
    selected = []
    for row, col in zip(rows, cols):
        sign = 1.0 if similarity[row, col] >= 0 else -1.0
        aligned[row] = member[col] * sign
        selected.append(abs(float(similarity[row, col])))
    return aligned, {
        "mean_abs_cosine": float(np.mean(selected)),
        "min_abs_cosine": float(np.min(selected)),
        "max_abs_cosine": float(np.max(selected)),
    }


def aggregate_blind_ensemble(
    recovered_members: Sequence[np.ndarray | torch.Tensor],
    *,
    aggregate: Literal["mean", "median"] = "mean",
    reference: Literal["first", "medoid"] = "first",
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Blindly align and aggregate recovered source matrices.

    The medoid, if requested, is selected by member-to-member cosine agreement;
    it never refers to private inputs, labels, or true embeddings.
    """
    if not recovered_members:
        raise ValueError("At least one ensemble member is required")
    arrays = [np.asarray(torch.as_tensor(member).detach().cpu()) for member in recovered_members]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("All ensemble members must have the same shape")
    reference_index = 0
    if reference == "medoid" and len(arrays) > 1:
        scores = []
        for candidate in arrays:
            per_other = []
            for other in arrays:
                _aligned, stats = align_member_to_reference(candidate, other)
                per_other.append(float(stats["mean_abs_cosine"]))
            scores.append(float(np.mean(per_other)))
        reference_index = int(np.argmax(scores))
    elif reference != "first":
        raise ValueError(f"Unknown reference policy: {reference}")

    ref = arrays[reference_index]
    aligned, scores = [ref], [1.0]
    for index, member in enumerate(arrays):
        if index == reference_index:
            continue
        member_aligned, stats = align_member_to_reference(ref, member)
        aligned.append(member_aligned)
        scores.append(float(stats["mean_abs_cosine"]))
    stack = np.stack(aligned, axis=0)
    if aggregate == "mean":
        result = stack.mean(axis=0)
    elif aggregate == "median":
        result = np.median(stack, axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregate}")
    return result.astype(ref.dtype, copy=False), {
        "reference_member": reference_index,
        "members": len(arrays),
        "alignment_mean_abs_cosine": float(np.mean(scores)),
        "alignment_min_abs_cosine": float(np.min(scores)),
    }


def oracle_permutation_aligned_cosine_similarity(
    recovered_sources: np.ndarray | torch.Tensor,
    true_embeddings: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, dict[str, object]]:
    """Evaluation-only Hungarian assignment against private embeddings.

    This function must not be called by source recovery, ensemble alignment, or
    FIA initialization.  The name is intentionally explicit for result labels.
    """
    recovered = np.asarray(torch.as_tensor(recovered_sources).detach().cpu())
    truth = np.asarray(torch.as_tensor(true_embeddings).detach().cpu())
    if recovered.shape != truth.shape:
        raise ValueError(f"Oracle evaluation shape mismatch: {recovered.shape} vs {truth.shape}")
    similarity = _flat_unit_rows(truth) @ _flat_unit_rows(recovered).T
    rows, cols = linear_sum_assignment(-np.abs(similarity))
    ordered = np.empty_like(recovered)
    cosines = np.empty(len(rows), dtype=np.float64)
    signs = np.empty(len(rows), dtype=np.float64)
    for row, col in zip(rows, cols):
        sign = 1.0 if similarity[row, col] >= 0 else -1.0
        ordered[row] = recovered[col] * sign
        cosines[row] = abs(similarity[row, col])
        signs[row] = sign
    return ordered, {
        "metric": "Oracle permutation-aligned cosine similarity, evaluation only",
        "mean_cs": float(cosines.mean()),
        "per_sample_cs": cosines.tolist(),
        "permutation": cols.tolist(),
        "signs": signs.tolist(),
    }
