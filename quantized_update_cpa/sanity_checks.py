#!/usr/bin/env python3
"""Sanity checks for corrected quantized-update CPA experiments."""

from __future__ import annotations

import argparse
import inspect
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from blind_alignment import aggregate_blind_ensemble
from common import (
    add_collect_args,
    build_experiment_model,
    derive_quantized_update_pickle,
    ensure_shared_fp32_update,
    read_pickle,
    resolve_local_batch_size,
    validate_parameter_mapping,
    write_json,
)
from fedavg_refinement import reproduce_fp32_update, validate_05_supported
from quantization import bin_distance_squared, quantize_update_symmetric, require_nearest_rounding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corrected 01--05 sanity checks.")
    add_collect_args(parser)
    parser.add_argument("--rounding", type=str, default="nearest")
    parser.add_argument("--output", type=Path, default=Path("quantized_update_cpa/outputs_v2/sanity/sanity_results.json"))
    parser.add_argument("--fp32_repro_max_abs_tol", type=float, default=1e-6)
    parser.add_argument("--fp32_repro_rel_l2_tol", type=float, default=1e-5)
    return parser.parse_args()


def _record(results: dict[str, Any], name: str, passed: bool, detail: Any) -> None:
    results[name] = {"passed": bool(passed), "detail": detail}
    print(f"{name}: {'PASS' if passed else 'FAIL'}")


def test_fp32_reproduction(args: argparse.Namespace, results: dict[str, Any]) -> Path:
    fp32_update_file = ensure_shared_fp32_update(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_experiment_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        device=device,
        checkpoint_path=args.global_checkpoint,
        seed=args.seed,
    )
    repro = reproduce_fp32_update(args=args, fp32_update_file=fp32_update_file, batch=0, model=model)
    _record(results, "test_1_fp32_local_update_reproduction", bool(repro["passed"]), repro)
    return fp32_update_file


def test_parameter_mapping(args: argparse.Namespace, fp32_update_file: Path, results: dict[str, Any]) -> None:
    data = read_pickle(fp32_update_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_experiment_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        device=device,
        checkpoint_path=args.global_checkpoint,
        seed=args.seed,
    )
    validate_parameter_mapping(data["param_names"], data["grad"][0], model)
    detail = {
        "parameter_count": len(data["param_names"]),
        "attack_layer": data["attack_layer"],
        "first_parameter": data["parameter_schema"][0],
        "last_parameter": data["parameter_schema"][-1],
    }
    _record(results, "test_2_parameter_mapping", True, detail)


def test_quantizer_roundtrip(fp32_update_file: Path, results: dict[str, Any]) -> Path:
    int4_update_file = fp32_update_file.parent / "sanity_int4_nearest.pickle"
    int4_update_file = derive_quantized_update_pickle(
        fp32_update_file=fp32_update_file,
        output_file=int4_update_file,
        rounding="nearest",
    )
    fp32 = read_pickle(fp32_update_file)
    int4 = read_pickle(int4_update_file)
    names = fp32["param_names"]
    recomputed = OrderedDict(
        (name, torch.as_tensor(value).float()) for name, value in zip(names, fp32["grad"][0])
    )
    codes, _dq, _stats = quantize_update_symmetric(update=recomputed, bits=4, rounding="nearest")
    mismatches = []
    for idx, name in enumerate(names):
        stored = torch.as_tensor(int4["quant_codes"][0][idx], dtype=torch.int8)
        if not torch.equal(stored, codes[name].cpu()):
            mismatches.append(name)
    detail = {
        "int4_update_file": str(int4_update_file),
        "qmin": int4["quant_stats"][0]["qmin"],
        "qmax": int4["quant_stats"][0]["qmax"],
        "levels": int4["quant_stats"][0]["levels"],
        "mismatches": mismatches,
    }
    _record(results, "test_3_quantizer_codes_exact", len(mismatches) == 0, detail)
    return int4_update_file


def test_bin_loss(results: dict[str, Any]) -> None:
    pred = torch.tensor([0.5, -2.0, 2.0, 20.0, 12.0, -20.0, -12.0])
    codes = torch.tensor([0, 0, 0, 7, 7, -7, -7], dtype=torch.int8)
    got = bin_distance_squared(pred, codes, 2.0)
    expected = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    _record(
        results,
        "test_4_bin_loss_cases",
        bool(torch.allclose(got, expected)),
        {"got": got.tolist(), "expected": expected.tolist()},
    )


def test_blind_ensemble(results: dict[str, Any]) -> None:
    sig = inspect.signature(aggregate_blind_ensemble)
    forbidden = [name for name in sig.parameters if "true" in name or "label" in name or "embedding" in name]
    base = torch.eye(4)
    member = torch.stack([-base[2], base[0], base[3], -base[1]], dim=0)
    aggregate, stats = aggregate_blind_ensemble([base, member], aggregate="mean", reference="first")
    passed = not forbidden and aggregate.shape == tuple(base.shape) and stats["members"] == 2
    _record(
        results,
        "test_5_blind_ensemble_no_oracle_args",
        passed,
        {"signature": str(sig), "forbidden_parameters": forbidden, "stats": stats},
    )


def test_sample_counts(args: argparse.Namespace, fp32_update_file: Path, results: dict[str, Any]) -> None:
    data = read_pickle(fp32_update_file)
    metadata = data["metadata"]
    detail = {
        "attack_trials": metadata["attack_trials"],
        "stored_batches": len(data["grad"]),
        "samples_per_trial": metadata["n_samples"],
        "sample_index_batches": len(data["sample_indices"]),
        "label_batches": len(data["y"]),
        "local_steps": metadata["local_steps"],
        "expected_local_steps": args.local_epochs
        * ((args.n_samples + resolve_local_batch_size(args) - 1) // resolve_local_batch_size(args)),
    }
    passed = (
        detail["attack_trials"] == detail["stored_batches"]
        == detail["sample_index_batches"]
        == detail["label_batches"]
        and detail["samples_per_trial"] == args.n_samples
        and detail["local_steps"] == detail["expected_local_steps"]
    )
    _record(results, "test_6_sample_count_metadata", passed, detail)


def test_unsupported_05(args: argparse.Namespace, int4_update_file: Path, results: dict[str, Any]) -> None:
    data = read_pickle(int4_update_file)
    model = torch.nn.Sequential(torch.nn.Dropout(p=0.2))
    failures = []
    bad_rounding = False
    try:
        require_nearest_rounding("stochastic")
    except ValueError:
        bad_rounding = True
    failures.append(("stochastic_rounding", bad_rounding))

    for key, value in (("momentum", 0.9), ("weight_decay", 1e-4)):
        mutated = dict(data)
        mutated["metadata"] = dict(data["metadata"])
        mutated["metadata"][key] = value
        rejected = False
        try:
            validate_05_supported(args, mutated, torch.nn.Sequential())
        except ValueError:
            rejected = True
        failures.append((key, rejected))

    dropout_rejected = False
    try:
        validate_05_supported(args, data, model)
    except ValueError:
        dropout_rejected = True
    failures.append(("active_dropout", dropout_rejected))

    _record(results, "test_7_unsupported_05_conditions", all(flag for _name, flag in failures), dict(failures))


def main() -> None:
    args = parse_args()
    results: dict[str, Any] = {}
    fp32_update_file = test_fp32_reproduction(args, results)
    test_parameter_mapping(args, fp32_update_file, results)
    int4_update_file = test_quantizer_roundtrip(fp32_update_file, results)
    test_bin_loss(results)
    test_blind_ensemble(results)
    test_sample_counts(args, fp32_update_file, results)
    test_unsupported_05(args, int4_update_file, results)
    write_json(args.output, results)
    if not all(row["passed"] for row in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
