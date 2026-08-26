#!/usr/bin/env python3
"""Fast sanity checks for CoRA sparse helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPARSE_SRC = ROOT / "sparsified_update_cpa"
if str(SPARSE_SRC) not in sys.path:
    sys.path.insert(0, str(SPARSE_SRC))

from sparsification import pack_bool_mask  # noqa: E402

from run_cora_sparse_cpa import (  # noqa: E402
    _randomized_client_attack_layer,
    make_member_seed_plan,
    sparse_distance_to_observation,
)


def assert_close(name: str, got: float, expected: float, atol: float = 1e-6) -> None:
    if abs(got - expected) > atol:
        raise AssertionError(f"{name}: got {got}, expected {expected}")


def test_seed_plan() -> None:
    plan = make_member_seed_plan(4, 1234)
    seeds = [row["desparsification_seed"] for row in plan]
    if len(set(seeds)) != 4:
        raise AssertionError(f"member desparsification seeds are not unique: {seeds}")
    if {row["cpa_seed"] for row in plan} != {"fixed_from_args_seed"}:
        raise AssertionError("CPA seed should be fixed across CoRA members")


def test_sparse_distance_matches_dense() -> None:
    pred = np.array([[1.0, 0.0, -2.0], [0.5, 3.0, 0.0]], dtype=np.float32)
    mask_np = np.array([[True, False, True], [False, True, False]])
    dense_obs = np.zeros_like(pred)
    dense_obs[mask_np] = np.array([0.5, -1.5, 2.0], dtype=np.float32)
    observation = {
        "mask": pack_bool_mask(torch.as_tensor(mask_np)),
        "values": dense_obs[mask_np].astype(np.float32),
    }
    got = sparse_distance_to_observation(pred, observation, observation_index=None)
    expected_l2 = float(np.linalg.norm(pred.reshape(-1) - dense_obs.reshape(-1)))
    expected_rel = expected_l2 / max(float(np.linalg.norm(dense_obs.reshape(-1))), 1e-12)
    assert_close("distance_l2", got["distance_l2"], expected_l2)
    assert_close("relative_l2", got["relative_l2"], expected_rel)


def test_retained_rescaled_desparse_bounds_and_keeps() -> None:
    mask_np = np.array([[True, False, True], [False, False, True]])
    observed = np.zeros(mask_np.shape, dtype=np.float32)
    observed[mask_np] = np.array([2.0, -4.0, 6.0], dtype=np.float32)
    tau = 0.75
    observation = {
        "mask": pack_bool_mask(torch.as_tensor(mask_np)),
        "values": observed[mask_np].astype(np.float32),
        "tau": tau,
    }
    sample = _randomized_client_attack_layer(
        observation,
        shape=mask_np.shape,
        rng=np.random.default_rng(123),
        sampler="retained_rescaled",
    )
    if not np.allclose(sample[mask_np], observed[mask_np]):
        raise AssertionError("observed sparse entries changed during desparsification")
    dropped = sample[~mask_np]
    if np.any(dropped < -tau) or np.any(dropped > tau):
        raise AssertionError(f"dropped values left [-tau, tau]: {dropped}")


def main() -> None:
    test_seed_plan()
    test_sparse_distance_matches_dense()
    test_retained_rescaled_desparse_bounds_and_keeps()
    print("CoRA sparse sanity checks passed")


if __name__ == "__main__":
    main()
