"""Evaluation-only oracle assignment for recovered CPA embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost)
    except ModuleNotFoundError:
        return _hungarian_rectangular(cost)


def _hungarian_rectangular(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Small dependency-free Hungarian fallback for rectangular matrices."""
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("cost must be a 2D matrix")
    transposed = False
    if matrix.shape[0] > matrix.shape[1]:
        matrix = matrix.T
        transposed = True

    n_rows, n_cols = matrix.shape
    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=np.int64)
    way = np.zeros(n_cols + 1, dtype=np.int64)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    rows = []
    cols = []
    for j in range(1, n_cols + 1):
        if p[j] != 0:
            rows.append(int(p[j] - 1))
            cols.append(int(j - 1))
    row_arr = np.asarray(rows, dtype=np.int64)
    col_arr = np.asarray(cols, dtype=np.int64)
    if transposed:
        return col_arr, row_arr
    return row_arr, col_arr


def _flat_unit_rows(values: np.ndarray | torch.Tensor, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(torch.as_tensor(values).detach().cpu(), dtype=np.float64)
    if array.ndim < 2:
        raise ValueError("values must have a leading sample dimension")
    array = array.reshape(array.shape[0], -1)
    return array / (np.linalg.norm(array, axis=1, keepdims=True) + eps)


def _as_source_array(values: np.ndarray | torch.Tensor) -> np.ndarray:
    array = np.asarray(torch.as_tensor(values).detach().cpu(), dtype=np.float32)
    if array.ndim < 2:
        raise ValueError("source matrix must have a leading sample dimension")
    return array


def align_to_reference(
    reference_sources: np.ndarray | torch.Tensor,
    member_sources: np.ndarray | torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Permutation/sign/scale-align one recovered source set to a blind reference."""
    reference = _as_source_array(reference_sources)
    member = _as_source_array(member_sources)
    if reference.shape != member.shape:
        raise ValueError(f"source shape mismatch: {reference.shape} vs {member.shape}")

    similarity = _flat_unit_rows(reference) @ _flat_unit_rows(member).T
    rows, cols = _linear_sum_assignment(-np.abs(similarity))
    aligned = np.empty_like(reference)
    scales = np.zeros(reference.shape[0], dtype=np.float64)
    matched_abs_cosines = np.zeros(reference.shape[0], dtype=np.float64)
    permutation = np.zeros(reference.shape[0], dtype=np.int64)

    reference_flat = reference.reshape(reference.shape[0], -1).astype(np.float64, copy=False)
    member_flat = member.reshape(member.shape[0], -1).astype(np.float64, copy=False)
    for row, col in zip(rows, cols):
        scale = float(np.dot(member_flat[col], reference_flat[row]) / (np.dot(member_flat[col], member_flat[col]) + eps))
        aligned[row] = (member[col] * scale).astype(np.float32, copy=False)
        scales[row] = scale
        matched_abs_cosines[row] = abs(float(similarity[row, col]))
        permutation[row] = int(col)

    return aligned, {
        "mean_abs_cosine": float(np.mean(matched_abs_cosines)) if matched_abs_cosines.size else None,
        "min_abs_cosine": float(np.min(matched_abs_cosines)) if matched_abs_cosines.size else None,
        "max_abs_cosine": float(np.max(matched_abs_cosines)) if matched_abs_cosines.size else None,
        "permutation_reference_to_member": permutation.tolist(),
        "least_squares_scales": scales.tolist(),
    }


def choose_medoid(members: list[np.ndarray | torch.Tensor]) -> tuple[int, dict[str, Any]]:
    if not members:
        raise ValueError("at least one source matrix is required")
    arrays = [_as_source_array(member) for member in members]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("all source matrices must have the same shape")

    scores = []
    for candidate in arrays:
        pair_scores = []
        for other in arrays:
            _aligned, stats = align_to_reference(candidate, other)
            pair_scores.append(float(stats["mean_abs_cosine"]))
        scores.append(float(np.mean(pair_scores)))
    medoid_index = int(np.argmax(scores))
    return medoid_index, {
        "policy": "blind_medoid_by_member_to_member_abs_cosine",
        "medoid_index": medoid_index,
        "medoid_scores": scores,
    }


def blind_global_sign_fix(sources: np.ndarray | torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
    """Fix remaining row signs with a no-oracle positive-energy rule."""
    array = _as_source_array(sources).copy()
    flat = array.reshape(array.shape[0], -1)
    signs = np.ones(array.shape[0], dtype=np.float32)
    positive_energy = np.sum(np.where(flat > 0.0, flat * flat, 0.0), axis=1)
    negative_energy = np.sum(np.where(flat < 0.0, flat * flat, 0.0), axis=1)
    flip = negative_energy > positive_energy
    array[flip] *= -1.0
    signs[flip] = -1.0
    return array, {
        "policy": "blind_positive_energy_global_sign",
        "signs": signs.astype(float).tolist(),
        "flipped_count": int(np.count_nonzero(flip)),
    }


def blind_medoid_align_and_average(
    members: list[np.ndarray | torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Align recovered members to a blind medoid, average them, then fix signs."""
    arrays = [_as_source_array(member) for member in members]
    medoid_index, medoid_stats = choose_medoid(arrays)
    reference = arrays[medoid_index]

    aligned = []
    rows = []
    for index, member in enumerate(arrays):
        if index == medoid_index:
            aligned_member = reference.astype(np.float32, copy=True)
            stats = {
                "mean_abs_cosine": 1.0,
                "min_abs_cosine": 1.0,
                "max_abs_cosine": 1.0,
                "permutation_reference_to_member": list(range(reference.shape[0])),
                "least_squares_scales": [1.0] * reference.shape[0],
            }
        else:
            aligned_member, stats = align_to_reference(reference, member)
        aligned.append(aligned_member)
        rows.append({"member": index, **stats})

    stack = np.stack(aligned, axis=0).astype(np.float32, copy=False)
    mean_sources = stack.mean(axis=0)
    final_sources, sign_stats = blind_global_sign_fix(mean_sources)
    return final_sources.astype(np.float32, copy=False), stack, {
        **medoid_stats,
        "member_alignment": rows,
        "global_sign": sign_stats,
        "members": len(arrays),
        "alignment_mean_abs_cosine": float(np.mean([row["mean_abs_cosine"] for row in rows])),
        "alignment_min_abs_cosine": float(np.min([row["min_abs_cosine"] for row in rows])),
        "final_embeddings_policy": "blind_medoid_alignment_then_mean_and_positive_energy_sign",
        "oracle_inputs_used": False,
    }


def oracle_permutation_aligned_cosine_similarity(
    recovered_sources: np.ndarray | torch.Tensor,
    true_embeddings: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Hungarian assignment against private embeddings for evaluation only."""
    recovered = np.asarray(torch.as_tensor(recovered_sources).detach().cpu())
    truth = np.asarray(torch.as_tensor(true_embeddings).detach().cpu())
    if recovered.ndim < 2 or truth.ndim < 2:
        raise ValueError("recovered_sources and true_embeddings need sample dimensions")
    if recovered.reshape(recovered.shape[0], -1).shape[1] != truth.reshape(truth.shape[0], -1).shape[1]:
        raise ValueError(f"Feature dimension mismatch: {recovered.shape} vs {truth.shape}")

    similarity = _flat_unit_rows(truth) @ _flat_unit_rows(recovered).T
    rows, cols = _linear_sum_assignment(-np.abs(similarity))
    ordered = np.empty((len(rows),) + recovered.shape[1:], dtype=recovered.dtype)
    cosines = np.empty(len(rows), dtype=np.float64)
    signs = np.empty(len(rows), dtype=np.float64)
    for out_idx, (row, col) in enumerate(zip(rows, cols)):
        sign = 1.0 if similarity[row, col] >= 0 else -1.0
        ordered[out_idx] = recovered[col] * sign
        cosines[out_idx] = abs(float(similarity[row, col]))
        signs[out_idx] = sign

    return ordered, {
        "metric": "Oracle permutation-aligned cosine similarity, evaluation only",
        "mean_cs": float(cosines.mean()) if cosines.size else None,
        "min_cs": float(cosines.min()) if cosines.size else None,
        "max_cs": float(cosines.max()) if cosines.size else None,
        "per_sample_cs": cosines.tolist(),
        "truth_rows": rows.tolist(),
        "recovered_cols": cols.tolist(),
        "signs": signs.tolist(),
    }
