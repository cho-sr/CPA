#!/usr/bin/env python3
"""Fast sanity checks for client q,s randomized FastICA utilities."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict

import numpy as np

from alignment import blind_medoid_align_and_average
from quant_bins import bin_bounds, naive_dequantize, randomized_dequantize
from run_naive_and_randomized_fastica import branch_manifest, make_trial_seed_plan


def record(results: Dict[str, Any], name: str, passed: bool, detail: Any) -> None:
    results[name] = {"passed": bool(passed), "detail": detail}
    print(f"{name}: {'PASS' if passed else 'FAIL'}")


def test_bin_bounds(results: Dict[str, Any]) -> None:
    codes = np.asarray([-7, 0, 7], dtype=np.int8)
    lower, upper = bin_bounds(codes, scale=2.0, max_abs=14.0)
    expected_lower = np.asarray([-14.0, -1.0, 13.0], dtype=np.float32)
    expected_upper = np.asarray([-13.0, 1.0, 14.0], dtype=np.float32)
    passed = bool(np.allclose(lower, expected_lower) and np.allclose(upper, expected_upper))
    record(results, "test_1_exact_bounded_bins", passed, {"lower": lower.tolist(), "upper": upper.tolist()})


def test_randomized_samples_inside_bins(results: Dict[str, Any]) -> None:
    codes = np.asarray([-7, -6, 0, 6, 7] * 1000, dtype=np.int8)
    rng = np.random.default_rng(123)
    samples = randomized_dequantize(codes, scale=0.25, max_abs=1.75, rng=rng)
    lower, upper = bin_bounds(codes, scale=0.25, max_abs=1.75)
    passed = bool(np.all(samples >= lower - 1e-7) and np.all(samples <= upper + 1e-7))
    record(
        results,
        "test_2_randomized_samples_stay_in_bins",
        passed,
        {"sample_min": float(samples.min()), "sample_max": float(samples.max())},
    )


def test_zero_max_abs(results: Dict[str, Any]) -> None:
    codes = np.zeros((4, 4), dtype=np.int8)
    rng = np.random.default_rng(456)
    naive = naive_dequantize(codes, scale=0.0)
    randomized = randomized_dequantize(codes, scale=0.0, max_abs=0.0, rng=rng)
    passed = bool(np.all(naive == 0.0) and np.all(randomized == 0.0))
    record(results, "test_3_zero_max_abs_is_deterministic_zero", passed, randomized.tolist())


def test_weighted_aggregate(results: Dict[str, Any]) -> None:
    codes_a = np.asarray([[1, 2], [3, 4]], dtype=np.int8)
    codes_b = np.asarray([[-1, -2], [-3, -4]], dtype=np.int8)
    got = 0.25 * naive_dequantize(codes_a, 0.5) + 0.75 * naive_dequantize(codes_b, 2.0)
    expected = np.asarray([[-1.375, -2.75], [-4.125, -5.5]], dtype=np.float32)
    record(results, "test_4_weighted_fedavg_from_same_qs", bool(np.allclose(got, expected)), got.tolist())


def test_blind_alignment(results: Dict[str, Any]) -> None:
    base = np.eye(4, dtype=np.float32)
    member = np.stack([base[2] * 3.0, -base[0] * 2.0, base[3] * 0.5, -base[1] * 4.0], axis=0)
    final, aligned, stats = blind_medoid_align_and_average([base, member])
    signature = inspect.signature(blind_medoid_align_and_average)
    forbidden = [name for name in signature.parameters if "true" in name or "label" in name or "z" == name]
    passed = (
        not forbidden
        and final.shape == base.shape
        and aligned.shape == (2, 4, 4)
        and stats["members"] == 2
        and stats["global_sign"]["policy"] == "blind_positive_energy_global_sign"
    )
    record(
        results,
        "test_5_blind_medoid_alignment_no_oracle_args",
        passed,
        {"signature": str(signature), "stats": stats},
    )


def test_trial_seed_policy(results: Dict[str, Any]) -> None:
    plan = make_trial_seed_plan(8, noise_seed=1234, fastica_random_state=42)
    fastica_seeds = {row["fastica_random_state"] for row in plan}
    dequant_seeds = {row["dequantization_seed"] for row in plan}
    passed = len(fastica_seeds) == 1 and len(dequant_seeds) == 8
    record(results, "test_6_fastica_seed_fixed_noise_seed_varies", passed, plan)


def test_branch_manifest_same_qs_hash(results: Dict[str, Any]) -> None:
    naive = branch_manifest(
        branch="naive",
        q_s_artifact_hash="codes-hash",
        q_s_meta_hash="meta-hash",
        fastica_random_state=42,
    )
    randomized = branch_manifest(
        branch="randomized",
        q_s_artifact_hash="codes-hash",
        q_s_meta_hash="meta-hash",
        fastica_random_state=42,
    )
    passed = (
        naive["q_s_codes_sha256"] == randomized["q_s_codes_sha256"]
        and naive["q_s_meta_sha256"] == randomized["q_s_meta_sha256"]
        and "final eval only" in naive["true_z_policy"]
    )
    record(results, "test_7_naive_randomized_share_qs_hash", passed, {"naive": naive, "randomized": randomized})


def main() -> None:
    results: Dict[str, Any] = {}
    test_bin_bounds(results)
    test_randomized_samples_inside_bins(results)
    test_zero_max_abs(results)
    test_weighted_aggregate(results)
    test_blind_alignment(results)
    test_trial_seed_policy(results)
    test_branch_manifest_same_qs_hash(results)

    failed = [name for name, row in results.items() if not row["passed"]]
    if failed:
        raise SystemExit(f"Sanity checks failed: {failed}")
    print(f"All sanity checks passed from {Path(__file__).parent}")


if __name__ == "__main__":
    main()
