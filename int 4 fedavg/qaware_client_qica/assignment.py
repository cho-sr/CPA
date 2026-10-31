"""Linear-sum assignment with a SciPy-free fallback."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def linear_sum_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment

        return scipy_linear_sum_assignment(cost_matrix)
    except ImportError:
        return _hungarian_fallback(cost_matrix)


def _hungarian_fallback(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"cost matrix must be 2D, got shape={cost.shape}")
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    transposed = False
    if n_rows > n_cols:
        cost = cost.T
        n_rows, n_cols = cost.shape
        transposed = True

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
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
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
            rows.append(p[j] - 1)
            cols.append(j - 1)
    rows_a = np.asarray(rows, dtype=np.int64)
    cols_a = np.asarray(cols, dtype=np.int64)
    order = np.argsort(rows_a)
    rows_a = rows_a[order]
    cols_a = cols_a[order]
    if transposed:
        return cols_a, rows_a
    return rows_a, cols_a
