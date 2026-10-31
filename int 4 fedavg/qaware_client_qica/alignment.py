"""Oracle-free centered alignment for repeated QICA source matrices."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

from assignment import linear_sum_assignment


def _as_2d_float(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"source matrix must be 2D, got shape={array.shape}")
    return array


def _center_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = _as_2d_float(values)
    means = array.mean(axis=1, keepdims=True)
    return array - means, means


def _unit_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = _as_2d_float(values).astype(np.float64, copy=False)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def max_abs_positive_sign_fix(sources: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
    """Resolve row sign using the largest-magnitude element in each row."""
    array = _as_2d_float(sources).copy()
    signs = np.ones(array.shape[0], dtype=np.float32)
    max_indices = np.argmax(np.abs(array), axis=1)
    for row, col in enumerate(max_indices):
        if array[row, col] < 0:
            array[row] *= -1.0
            signs[row] = -1.0
    return array.astype(np.float32, copy=False), {
        "policy": "row_max_abs_entry_positive",
        "signs": signs.astype(float).tolist(),
        "flipped_count": int(np.count_nonzero(signs < 0)),
    }


def align_to_reference_centered(
    reference_mean_restored: np.ndarray,
    member_mean_restored: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Match using centered rows and apply the same transform to mean-restored rows."""
    reference = _as_2d_float(reference_mean_restored)
    member = _as_2d_float(member_mean_restored)
    if reference.shape != member.shape:
        raise ValueError(f"source shape mismatch: {reference.shape} vs {member.shape}")

    reference_centered, _ref_mean = _center_rows(reference)
    member_centered, _member_mean = _center_rows(member)
    similarity = _unit_rows(reference_centered) @ _unit_rows(member_centered).T
    rows, cols = linear_sum_assignment(-np.abs(similarity))

    aligned_centered = np.empty_like(reference_centered)
    aligned_mean_restored = np.empty_like(reference)
    scales = np.zeros(reference.shape[0], dtype=np.float64)
    matched_abs_cosines = np.zeros(reference.shape[0], dtype=np.float64)
    permutation = np.zeros(reference.shape[0], dtype=np.int64)

    for row, col in zip(rows, cols):
        ref_row = reference_centered[row].astype(np.float64, copy=False)
        member_row = member_centered[col].astype(np.float64, copy=False)
        scale = float(np.dot(member_row, ref_row) / (np.dot(member_row, member_row) + eps))
        aligned_centered[row] = (member_centered[col] * scale).astype(np.float32, copy=False)
        aligned_mean_restored[row] = (member[col] * scale).astype(np.float32, copy=False)
        scales[row] = scale
        matched_abs_cosines[row] = abs(float(similarity[row, col]))
        permutation[row] = int(col)

    return aligned_centered, aligned_mean_restored, {
        "mean_abs_cosine": float(np.mean(matched_abs_cosines)),
        "min_abs_cosine": float(np.min(matched_abs_cosines)),
        "max_abs_cosine": float(np.max(matched_abs_cosines)),
        "permutation_reference_to_member": permutation.tolist(),
        "least_squares_scales": scales.tolist(),
    }


def choose_centered_medoid(members: Sequence[np.ndarray]) -> tuple[int, Dict[str, Any]]:
    if not members:
        raise ValueError("at least one source matrix is required")
    arrays = [_as_2d_float(member) for member in members]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("all source matrices must have the same shape")

    scores = []
    for candidate in arrays:
        pair_scores = []
        for other in arrays:
            _aligned_c, _aligned_m, stats = align_to_reference_centered(candidate, other)
            pair_scores.append(float(stats["mean_abs_cosine"]))
        scores.append(float(np.mean(pair_scores)))
    medoid_index = int(np.argmax(scores))
    return medoid_index, {
        "policy": "blind_centered_medoid_by_member_to_member_abs_cosine",
        "medoid_index": medoid_index,
        "medoid_scores": scores,
    }


def blind_centered_align_and_average(
    members: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Center-match trials, apply transforms to mean-restored rows, then average."""
    arrays = [_as_2d_float(member) for member in members]
    medoid_index, medoid_stats = choose_centered_medoid(arrays)
    reference = arrays[medoid_index]
    reference_centered, _reference_mean = _center_rows(reference)

    aligned_centered = []
    aligned_mean_restored = []
    rows = []
    for index, member in enumerate(arrays):
        if index == medoid_index:
            centered_member = reference_centered.astype(np.float32, copy=True)
            mean_restored_member = reference.astype(np.float32, copy=True)
            stats = {
                "mean_abs_cosine": 1.0,
                "min_abs_cosine": 1.0,
                "max_abs_cosine": 1.0,
                "permutation_reference_to_member": list(range(reference.shape[0])),
                "least_squares_scales": [1.0] * reference.shape[0],
            }
        else:
            centered_member, mean_restored_member, stats = align_to_reference_centered(reference, member)
        aligned_centered.append(centered_member)
        aligned_mean_restored.append(mean_restored_member)
        rows.append({"trial": index, **stats})

    centered_stack = np.stack(aligned_centered, axis=0).astype(np.float32, copy=False)
    mean_restored_stack = np.stack(aligned_mean_restored, axis=0).astype(np.float32, copy=False)
    final_sources = mean_restored_stack.mean(axis=0)
    final_sources, sign_stats = max_abs_positive_sign_fix(final_sources)
    return (
        final_sources.astype(np.float32, copy=False),
        centered_stack,
        mean_restored_stack,
        {
            **medoid_stats,
            "member_alignment": rows,
            "global_sign": sign_stats,
            "members": len(arrays),
            "alignment_mean_abs_cosine": float(np.mean([row["mean_abs_cosine"] for row in rows])),
            "alignment_min_abs_cosine": float(np.min([row["min_abs_cosine"] for row in rows])),
            "scale_alignment": "least_squares_scale_on_centered_sources_applied_to_mean_restored_sources",
            "final_embeddings_policy": "mean_restored_average_after_oracle_free_centered_alignment",
        },
    )
