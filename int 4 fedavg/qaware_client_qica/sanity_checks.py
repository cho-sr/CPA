#!/usr/bin/env python3
"""Fast sanity checks for client-wise bin-constrained QICA."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import torch

from alignment import blind_centered_align_and_average
from qica_core import (
    ClientWiseBinConstrainedQICA,
    bin_distance_squared,
    compute_fixed_whitening,
    finite_bin_bounds,
    project_to_finite_bins_,
    weighted_aggregate,
)


def record(rows: List[Dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    rows.append({"name": name, "passed": bool(passed), "detail": detail})


def synthetic_artifact() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    codes = torch.tensor(
        [
            [
                [-7, -3, 0, 4, 7, 1],
                [0, 1, 2, 3, 4, 5],
                [-1, -2, -3, -4, -5, -6],
                [7, 6, 5, 4, 3, 2],
            ],
            [
                [-6, -4, 0, 2, 6, 7],
                [1, 2, 3, 4, 5, 6],
                [-7, -5, -3, -1, 1, 3],
                [0, 0, 0, 0, 0, 0],
            ],
        ],
        dtype=torch.int8,
    )
    scales = torch.tensor([0.5, 0.25], dtype=torch.float32)
    max_abs = scales * 7.0
    weights = torch.tensor([0.4, 0.6], dtype=torch.float32)
    return codes, scales, max_abs, weights


def make_whitening(codes: torch.Tensor, scales: torch.Tensor, weights: torch.Tensor, m: int = 2):
    init = codes.float() * scales.view(-1, 1, 1)
    aggregate = weighted_aggregate(init, weights)
    return compute_fixed_whitening(
        aggregate,
        n_components=m,
        eig_floor_rel=1e-5,
        rank_tol_rel=1e-7,
        eps=1e-6,
        column_count=None,
        seed=42,
    )


def test_endpoint_bins(rows: List[Dict[str, Any]]) -> None:
    codes, scales, max_abs, _weights = synthetic_artifact()
    lower, upper = finite_bin_bounds(codes, scales, max_abs)
    passed = bool(
        torch.isclose(lower[0, 0, 0], torch.tensor(-3.5))
        and torch.isclose(upper[0, 0, 0], torch.tensor(-3.25))
        and torch.isclose(lower[0, 0, 4], torch.tensor(3.25))
        and torch.isclose(upper[0, 0, 4], torch.tensor(3.5))
    )
    record(rows, "finite_endpoint_bins_for_pm7", passed, {"lower_pm7": [float(lower[0, 0, 0]), float(lower[0, 0, 4])]})


def test_projection(rows: List[Dict[str, Any]]) -> None:
    codes, scales, max_abs, _weights = synthetic_artifact()
    values = codes.float() * scales.view(-1, 1, 1)
    values[0, 0, 0] = -10.0
    values[0, 0, 4] = 10.0
    project_to_finite_bins_(values, codes, scales, max_abs)
    dist = bin_distance_squared(values, codes, scales, max_abs)
    record(rows, "projection_returns_values_to_bins", bool(torch.all(dist <= 1e-12)), float(dist.max()))


def test_qon_gradient(rows: List[Dict[str, Any]]) -> None:
    codes, scales, max_abs, weights = synthetic_artifact()
    whitening = make_whitening(codes, scales, weights, m=2)
    runner = ClientWiseBinConstrainedQICA(
        codes=codes,
        scales=scales,
        max_abs=max_abs,
        weights=weights,
        whitening=whitening,
        n_components=2,
        use_q_loss=True,
        use_ne=False,
        train_observation=True,
        lr=1e-2,
        eps=1e-6,
        decor_weight=0.0,
        decor_temperature=1.0,
        l1_weight=0.0,
        nv_weight=0.0,
        q_weight=1.0,
        seed=1,
    )
    with torch.no_grad():
        runner.X_client_hat[0, 0, 1] += 0.4
    row = runner.step(iter_idx=0, columns_per_step=None)
    record(
        rows,
        "qon_train_observation_has_x_gradient",
        bool(row["x_client_grad_norm"] is not None and row["x_client_grad_norm"] > 0),
        row["x_client_grad_norm"],
    )


def test_fail_fast(rows: List[Dict[str, Any]]) -> None:
    codes, scales, max_abs, weights = synthetic_artifact()
    whitening = make_whitening(codes, scales, weights, m=2)
    try:
        ClientWiseBinConstrainedQICA(
            codes=codes,
            scales=scales,
            max_abs=max_abs,
            weights=weights,
            whitening=whitening,
            n_components=2,
            use_q_loss=True,
            use_ne=False,
            train_observation=False,
            lr=1e-2,
            eps=1e-6,
            decor_weight=0.0,
            decor_temperature=1.0,
            l1_weight=0.0,
            nv_weight=0.0,
            q_weight=1.0,
            seed=1,
        )
    except ValueError as exc:
        record(rows, "qon_no_train_observation_fail_fast", True, str(exc))
        return
    record(rows, "qon_no_train_observation_fail_fast", False, "no ValueError")


def test_qoff_w_only(rows: List[Dict[str, Any]]) -> None:
    codes, scales, max_abs, weights = synthetic_artifact()
    whitening = make_whitening(codes, scales, weights, m=2)
    runner = ClientWiseBinConstrainedQICA(
        codes=codes,
        scales=scales,
        max_abs=max_abs,
        weights=weights,
        whitening=whitening,
        n_components=2,
        use_q_loss=False,
        use_ne=True,
        train_observation=True,
        lr=1e-2,
        eps=1e-6,
        decor_weight=0.0,
        decor_temperature=1.0,
        l1_weight=0.0,
        nv_weight=0.0,
        q_weight=1.0,
        seed=1,
    )
    record(
        rows,
        "qoff_observation_is_fixed_and_w_only",
        bool(not isinstance(runner.X_client_hat, torch.nn.Parameter) and len(runner.optimizer.param_groups[0]["params"]) == 1),
        str(type(runner.X_client_hat)),
    )


def test_rank_fail_fast(rows: List[Dict[str, Any]]) -> None:
    codes, scales, _max_abs, weights = synthetic_artifact()
    init = codes.float() * scales.view(-1, 1, 1)
    aggregate = weighted_aggregate(init, weights)
    try:
        compute_fixed_whitening(
            aggregate,
            n_components=4,
            eig_floor_rel=1e-5,
            rank_tol_rel=1.1,
            eps=1e-6,
            column_count=None,
            seed=42,
        )
    except ValueError as exc:
        record(rows, "m_lte_numerical_rank_fail_fast", True, str(exc))
        return
    record(rows, "m_lte_numerical_rank_fail_fast", False, "no ValueError")


def test_centered_alignment_mean_restored(rows: List[Dict[str, Any]]) -> None:
    base = np.asarray([[1, 2, 3, 4], [-4, 1, 0, 2]], dtype=np.float32)
    member = np.asarray([[-8, 2, 0, 4], [-2, -4, -6, -8]], dtype=np.float32)
    final, centered_stack, restored_stack, metrics = blind_centered_align_and_average([base, member])
    passed = bool(final.shape == base.shape and centered_stack.shape[0] == 2 and restored_stack.shape[0] == 2)
    record(rows, "centered_matching_applies_transform_to_mean_restored", passed, metrics["scale_alignment"])


def main() -> None:
    rows: List[Dict[str, Any]] = []
    test_endpoint_bins(rows)
    test_projection(rows)
    test_qon_gradient(rows)
    test_fail_fast(rows)
    test_qoff_w_only(rows)
    test_rank_fail_fast(rows)
    test_centered_alignment_mean_restored(rows)
    passed = all(row["passed"] for row in rows)
    print(json.dumps({"passed": passed, "tests": rows}, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
