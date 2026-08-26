"""Blind medoid alignment for repeated FastICA source matrices."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

from assignment import linear_sum_assignment


def _as_2d_float(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"source matrix must be 2D, got shape={array.shape}")
    return array


def _unit_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = _as_2d_float(values).astype(np.float64, copy=False)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def align_to_reference(
    reference_sources: np.ndarray,
    member_sources: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Permutation/sign/scale-align one source matrix to a blind reference."""
    reference = _as_2d_float(reference_sources)
    member = _as_2d_float(member_sources)
    if reference.shape != member.shape:
        raise ValueError(f"source shape mismatch: {reference.shape} vs {member.shape}")

    similarity = _unit_rows(reference) @ _unit_rows(member).T
    rows, cols = linear_sum_assignment(-np.abs(similarity))
    aligned = np.empty_like(reference)
    scales = np.zeros(reference.shape[0], dtype=np.float64)
    matched_abs_cosines = np.zeros(reference.shape[0], dtype=np.float64)
    permutation = np.zeros(reference.shape[0], dtype=np.int64)

    for row, col in zip(rows, cols):
        ref_row = reference[row].astype(np.float64, copy=False)
        member_row = member[col].astype(np.float64, copy=False)
        scale = float(np.dot(member_row, ref_row) / (np.dot(member_row, member_row) + eps))
        aligned[row] = (member[col] * scale).astype(np.float32, copy=False)
        scales[row] = scale
        matched_abs_cosines[row] = abs(float(similarity[row, col]))
        permutation[row] = int(col)

    return aligned, {
        "mean_abs_cosine": float(np.mean(matched_abs_cosines)),
        "min_abs_cosine": float(np.min(matched_abs_cosines)),
        "max_abs_cosine": float(np.max(matched_abs_cosines)),
        "permutation_reference_to_member": permutation.tolist(),
        "least_squares_scales": scales.tolist(),
    }


def choose_medoid(members: Sequence[np.ndarray]) -> Tuple[int, Dict[str, Any]]:
    if not members:
        raise ValueError("at least one source matrix is required")
    arrays = [_as_2d_float(member) for member in members]
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


def blind_global_sign_fix(sources: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fix the remaining global sign ambiguity without using true embeddings."""
    array = _as_2d_float(sources).copy()
    signs = np.ones(array.shape[0], dtype=np.float32)
    positive_energy = np.sum(np.where(array > 0.0, array * array, 0.0), axis=1)
    negative_energy = np.sum(np.where(array < 0.0, array * array, 0.0), axis=1)
    flip = negative_energy > positive_energy
    array[flip] *= -1.0
    signs[flip] = -1.0
    return array, {
        "policy": "blind_positive_energy_global_sign",
        "signs": signs.astype(float).tolist(),
        "flipped_count": int(np.count_nonzero(flip)),
    }


def blind_medoid_align_and_average(
    members: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Align R FastICA runs to a blind medoid, average matched sources, fix sign."""
    arrays = [_as_2d_float(member) for member in members]
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
        rows.append({"trial": index, **stats})

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
    }
