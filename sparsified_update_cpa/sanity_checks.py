#!/usr/bin/env python3
"""Sanity checks for Naive Sparse CPA artifacts and metric contracts."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blind_alignment import blind_medoid_align_and_average
from metrics import summarize_metric_rows, write_csv
from sparsification import (
    attack_layer_spectrum_stats,
    compare_update_to_reference,
    derive_sparse_update_pickle,
    global_topk_attack_layer_observation,
    identity_global_topk_stats,
    randomized_desparsify_sparse_observation,
    sparsify_update_global_topk,
    summarize_client_sparse_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sparsified_update_cpa/outputs_v1/sanity/sanity_results.json"),
    )
    return parser.parse_args()


def _record(results: dict[str, Any], name: str, passed: bool, detail: Any) -> None:
    results[name] = {"passed": bool(passed), "detail": detail}
    print(f"{name}: {'PASS' if passed else 'FAIL'}")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _synthetic_fp32_artifact(path: Path) -> Path:
    grad = [
        [
            np.asarray([[0.0, 1.0, -2.0, 3.0]], dtype=np.float32),
            np.asarray([4.0, 0.0, -5.0, 6.0], dtype=np.float32),
        ]
    ]
    labels = [np.asarray([0, 1], dtype=np.int64)]
    sample_indices = [[10, 11]]
    artifact = {
        "schema_version": 2,
        "x": [np.zeros((2, 3, 4, 4), dtype=np.float32)],
        "y": labels,
        "z": [np.eye(2, dtype=np.float32)],
        "grad": grad,
        "param_names": ["layer.weight", "layer.bias"],
        "parameter_schema": [
            {"name": "layer.weight", "shape": [1, 4], "dtype": "torch.float32"},
            {"name": "layer.bias", "shape": [4], "dtype": "torch.float32"},
        ],
        "attack_layer": {"index": 0, "name": "layer.weight"},
        "sample_indices": sample_indices,
        "local_batch_orders": [[0, 1]],
        "quant_codes": None,
        "quant_stats": [{"quant_bits": 32, "rounding": "none"}],
        "metadata": {
            "quant_bits": 32,
            "rounding": "nearest",
            "attack_trials": 1,
            "n_samples": 2,
            "local_epochs": 1,
            "local_steps": 1,
            "local_batch_size": 2,
            "micro_batch_size": 2,
            "lr": 0.001,
            "optimizer": "SGD",
            "momentum": 0.0,
            "weight_decay": 0.0,
            "seed": 42,
            "checkpoint": {"path": None, "sha256": None},
            "sample_indices_hash": _sha(json.dumps(sample_indices).encode("utf-8")),
            "label_hash": _sha(labels[0].tobytes()),
            "update_tensor_hash": _sha(grad[0][0].tobytes() + grad[0][1].tobytes()),
        },
    }
    _write_pickle(path, artifact)
    return path


def test_global_topk_exact_mask_count(results: dict[str, Any]) -> None:
    update = OrderedDict(
        [
            ("a", torch.tensor([[0.0, 1.0, -2.0, 3.0]], dtype=torch.float32)),
            ("b", torch.tensor([4.0, 0.0, -5.0, 6.0], dtype=torch.float32)),
        ]
    )
    sparse, _masks, stats = sparsify_update_global_topk(update, keep_ratio=0.5)
    values = torch.cat([tensor.reshape(-1) for tensor in sparse.values()])
    passed = (
        stats["mask_kept_elements"] == 4
        and int((values != 0).sum().item()) == 4
        and set(values[values != 0].tolist()) == {3.0, 4.0, -5.0, 6.0}
        and stats["topk_threshold_abs"] == 3.0
    )
    _record(results, "test_1_global_topk_exact_mask_count", passed, stats)


def test_selected_original_zeros_are_allowed(results: dict[str, Any]) -> None:
    update = OrderedDict([("a", torch.tensor([0.0, 1.0, 0.0, -2.0], dtype=torch.float32))])
    sparse, _masks, stats = sparsify_update_global_topk(update, keep_ratio=1.0)
    values = next(iter(sparse.values()))
    passed = stats["mask_kept_elements"] == 4 and int((values != 0).sum().item()) == 2
    _record(results, "test_2_mask_count_authoritative_when_original_zeros_exist", passed, stats)


def test_derived_artifact_contract(base_dir: Path, results: dict[str, Any]) -> Path:
    fp32_path = _synthetic_fp32_artifact(base_dir / "synthetic_fp32.pickle")
    sparse_path = derive_sparse_update_pickle(
        fp32_update_file=fp32_path,
        output_file=base_dir / "synthetic_sparse_keep0p5.pickle",
        keep_ratio=0.5,
        sparse_policy="global_topk",
        author_source_hash="synthetic-author",
    )
    sparse = _read_pickle(sparse_path)
    grad_arrays = sparse["grad"][0]
    dense_arrays = all(isinstance(value, np.ndarray) and value.dtype == np.float32 for value in grad_arrays)
    dense_zero_filled = any(int(np.count_nonzero(value == 0.0)) > 0 for value in grad_arrays)
    shapes_match = [list(value.shape) for value in grad_arrays] == [[1, 4], [4]]
    metadata_keys = {
        "sparsification_policy",
        "keep_ratio",
        "observed_zero_policy",
        "sparse_nonzero_fraction",
        "sparse_relative_l2",
        "sparse_cosine_similarity_to_fp32",
        "source_fp32_update_hash",
    }
    passed = (
        dense_arrays
        and dense_zero_filled
        and shapes_match
        and bool(sparse.get("sparse_masks"))
        and metadata_keys.issubset(set(sparse["metadata"]))
    )
    _record(
        results,
        "test_3_sparse_artifact_dense_zero_filled_contract",
        passed,
        {
            "sparse_path": str(sparse_path),
            "dense_arrays": dense_arrays,
            "dense_zero_filled": dense_zero_filled,
            "shapes": [list(value.shape) for value in grad_arrays],
            "metadata_subset": {key: sparse["metadata"].get(key) for key in sorted(metadata_keys)},
        },
    )
    return sparse_path


def test_metric_summary_contract(base_dir: Path, results: dict[str, Any]) -> None:
    image_rows = [
        {"batch": 0, "sample": 0, "lpips": 0.40, "psnr": 22.0, "ssim": 0.30},
        {"batch": 0, "sample": 1, "lpips": 0.60, "psnr": 18.0, "ssim": 0.10},
    ]
    eval_entries = [{"batch": 0, "embedding": {"per_sample_cs": [0.9, 0.8], "mean_cs": 0.85}}]
    summary = summarize_metric_rows(
        image_rows=image_rows,
        eval_entries=eval_entries,
        success_lpips_threshold=0.50,
    )
    csv_path = base_dir / "metric_summary.csv"
    write_csv(csv_path, summary)
    by_metric = {row["metric"]: row for row in summary}
    passed = (
        {"lpips", "psnr", "ssim", "embedding_cosine_similarity", "reconstruction_success_rate"}
        .issubset(by_metric)
        and by_metric["reconstruction_success_rate"]["mean"] == 0.5
        and csv_path.exists()
    )
    _record(results, "test_4_metric_summary_emits_required_metrics", passed, summary)


def test_client_then_aggregate_stats_contract(results: dict[str, Any]) -> None:
    client_updates = [
        OrderedDict(
            [
                ("attack.weight", torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -4.0]])),
                ("tail.bias", torch.tensor([0.25, -0.75])),
            ]
        ),
        OrderedDict(
            [
                ("attack.weight", torch.tensor([[2.0, -1.0, 0.0], [1.0, -3.0, 4.0]])),
                ("tail.bias", torch.tensor([-0.5, 0.5])),
            ]
        ),
    ]
    names = list(client_updates[0])
    fp32_aggregate = OrderedDict(
        (name, sum(update[name] for update in client_updates) / len(client_updates)) for name in names
    )
    sparse_accum = OrderedDict((name, torch.zeros_like(value)) for name, value in fp32_aggregate.items())
    client_stats = []
    packed_masks_empty = True
    for update in client_updates:
        sparse_update, masks, stats = sparsify_update_global_topk(update, keep_ratio=0.5, pack_masks=False)
        client_stats.append(stats)
        for name in names:
            sparse_accum[name].add_(sparse_update[name] / len(client_updates))
        packed_masks_empty = packed_masks_empty and not masks

    identity_stats = [identity_global_topk_stats(update) for update in client_updates]
    client_level = summarize_client_sparse_stats(client_stats, attack_layer_index=0)
    aggregate_level = compare_update_to_reference(sparse_accum, fp32_aggregate, attack_layer_index=0)
    spectrum = attack_layer_spectrum_stats(sparse_accum["attack.weight"])
    passed = (
        client_level["num_clients"] == 2
        and all(row["sparse_mask_fraction"] == 1.0 for row in identity_stats)
        and all(row["mask_kept_elements"] == 4 for row in client_stats)
        and "global" in aggregate_level
        and "attack_layer" in aggregate_level
        and packed_masks_empty
        and spectrum["numerical_rank"] >= 1
        and bool(spectrum["singular_values_descending"])
        and bool(spectrum["covariance_eigenvalues_descending"])
    )
    _record(
        results,
        "test_5_client_then_aggregate_stats_and_spectrum_contract",
        passed,
        {
            "client_level": client_level,
            "aggregate_level": aggregate_level,
            "spectrum_keys": sorted(spectrum.keys()),
            "packed_masks_empty": packed_masks_empty,
            "identity_mask_fractions": [row["sparse_mask_fraction"] for row in identity_stats],
        },
    )


def test_randomized_desparse_observed_and_bounds(results: dict[str, Any]) -> None:
    observed = np.asarray([[10.0, 0.0, -2.0], [0.0, 4.0, 0.0]], dtype=np.float32)
    mask = np.asarray([[True, False, True], [False, True, False]])
    tau = 0.25
    sample = randomized_desparsify_sparse_observation(observed, mask, tau, np.random.default_rng(123))
    dropped = sample[~mask]
    passed = bool(
        np.allclose(sample[mask], observed[mask])
        and np.all(dropped >= -tau - 1e-7)
        and np.all(dropped <= tau + 1e-7)
    )
    _record(
        results,
        "test_6_randomized_desparse_preserves_observed_and_bounds_dropped",
        passed,
        {"sample": sample.tolist(), "tau": tau},
    )


def test_attack_layer_only_topk_matches_full_sparse(results: dict[str, Any]) -> None:
    update = OrderedDict(
        [
            ("attack.weight", torch.tensor([[1.0, -7.0, 3.0], [0.25, -5.0, 2.0]])),
            ("tail.bias", torch.tensor([4.0, -6.0, 0.5, 8.0])),
        ]
    )
    sparse, _masks, stats, raw_masks = sparsify_update_global_topk(
        update, keep_ratio=0.5, pack_masks=False, return_masks=True
    )
    observation = global_topk_attack_layer_observation(
        update,
        keep_ratio=0.5,
        attack_layer_index=0,
    )
    attack_mask = raw_masks[0]
    expected_values = sparse["attack.weight"][attack_mask].detach().cpu().numpy()
    passed = bool(
        torch.equal(observation["mask"], attack_mask)
        and np.array_equal(observation["values"], expected_values.astype(np.float32))
        and observation["tau"] == stats["topk_threshold_abs"]
        and observation["observed_elements"] == int(attack_mask.sum().item())
    )
    _record(
        results,
        "test_7_attack_layer_only_topk_matches_full_sparse",
        passed,
        {
            "tau": observation["tau"],
            "mask": observation["mask"].tolist(),
            "values": observation["values"].tolist(),
        },
    )


def test_randomized_desparse_seed_policy(results: dict[str, Any]) -> None:
    observed = np.asarray([1.0, 0.0, -3.0, 0.0], dtype=np.float32)
    mask = np.asarray([True, False, True, False])
    a = randomized_desparsify_sparse_observation(observed, mask, 0.5, np.random.default_rng(7))
    b = randomized_desparsify_sparse_observation(observed, mask, 0.5, np.random.default_rng(7))
    c = randomized_desparsify_sparse_observation(observed, mask, 0.5, np.random.default_rng(8))
    passed = bool(np.allclose(a, b) and not np.allclose(a[~mask], c[~mask]))
    _record(
        results,
        "test_8_randomized_desparse_same_seed_reproducible_different_seed_varies",
        passed,
        {"seed7": a.tolist(), "seed8": c.tolist()},
    )


def test_randomized_desparse_keep1_identity(results: dict[str, Any]) -> None:
    observed = np.asarray([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    mask = np.ones_like(observed, dtype=bool)
    sample = randomized_desparsify_sparse_observation(observed, mask, 999.0, np.random.default_rng(123))
    passed = bool(np.array_equal(sample, observed))
    _record(results, "test_9_randomized_desparse_keep1_degenerates_to_identity", passed, sample.tolist())


def test_blind_medoid_alignment_no_oracle_args(results: dict[str, Any]) -> None:
    base = np.eye(4, dtype=np.float32)
    member = np.stack([base[2] * 3.0, -base[0] * 2.0, base[3] * 0.5, -base[1] * 4.0], axis=0)
    final, aligned, stats = blind_medoid_align_and_average([base, member])
    signature = inspect.signature(blind_medoid_align_and_average)
    forbidden = [name for name in signature.parameters if "true" in name or "label" in name or name == "z"]
    passed = (
        not forbidden
        and final.shape == base.shape
        and aligned.shape == (2, 4, 4)
        and stats["oracle_inputs_used"] is False
        and stats["global_sign"]["policy"] == "blind_positive_energy_global_sign"
    )
    _record(
        results,
        "test_10_blind_medoid_alignment_no_oracle_args",
        passed,
        {"signature": str(signature), "stats": stats},
    )


def main() -> None:
    args = parse_args()
    base_dir = args.output.parent
    results: dict[str, Any] = {}
    test_global_topk_exact_mask_count(results)
    test_selected_original_zeros_are_allowed(results)
    test_derived_artifact_contract(base_dir, results)
    test_metric_summary_contract(base_dir, results)
    test_client_then_aggregate_stats_contract(results)
    test_randomized_desparse_observed_and_bounds(results)
    test_attack_layer_only_topk_matches_full_sparse(results)
    test_randomized_desparse_seed_policy(results)
    test_randomized_desparse_keep1_identity(results)
    test_blind_medoid_alignment_no_oracle_args(results)
    _write_json(args.output, results)
    if not all(row["passed"] for row in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
