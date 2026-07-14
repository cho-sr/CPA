#!/usr/bin/env python3
"""Stage 3: randomized dequantization ensemble for 4-bit update CPA."""

from __future__ import annotations

import argparse
import copy
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    add_attack_args,
    add_collect_args,
    collect_fedavg_update_pickle,
    output_pickle_path,
    resolve_local_batch_size,
    run_attack_with_update_file,
)


EXPERIMENT_NAME = "03_randomized_dequant_ensemble_cpa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect naive 4-bit FedAvg updates, generate randomized "
            "dequantization ensemble members, and optionally run CPA/FIA."
        )
    )
    add_collect_args(parser)
    parser.add_argument(
        "--rounding",
        type=str,
        default="nearest",
        choices=["nearest", "stochastic"],
        help="Rounding mode used for the base 4-bit fake quantization.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=8,
        help="Number of randomized dequantization members K.",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=1.0,
        help="Multiplier for U(-delta/2, delta/2) dequantization noise.",
    )
    parser.add_argument(
        "--noise_seed",
        type=int,
        default=1234,
        help="Seed for randomized dequantization noise.",
    )
    parser.add_argument(
        "--start_member",
        type=int,
        default=0,
        help="First ensemble member index to process, useful for resuming.",
    )
    add_attack_args(parser)
    return parser.parse_args()


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_pickle(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _fallback_scale(round_stats: dict, tensor: np.ndarray) -> float:
    if "quant_scale_mean" in round_stats:
        return float(round_stats["quant_scale_mean"])
    nonzero = np.abs(tensor[np.nonzero(tensor)])
    if nonzero.size == 0:
        return 0.0
    return float(np.percentile(nonzero, 5))


def _tensor_scale(round_stats: dict, tensor_idx: int, tensor: np.ndarray) -> float:
    per_tensor = round_stats.get("per_tensor")
    if per_tensor is not None and tensor_idx < len(per_tensor):
        return float(per_tensor[tensor_idx]["scale"])
    return _fallback_scale(round_stats, tensor)


def randomized_dequant_pickle(
    *,
    base_update_file: Path,
    output_file: Path,
    member_idx: int,
    noise_seed: int,
    noise_scale: float,
) -> Path:
    data = _read_pickle(base_update_file)
    randomized = copy.deepcopy(data)
    rng = np.random.default_rng(noise_seed + member_idx)

    for round_idx, grad_list in enumerate(randomized["grad"]):
        round_stats = randomized["quant_stats"][round_idx]
        for tensor_idx, tensor in enumerate(grad_list):
            scale = _tensor_scale(round_stats, tensor_idx, tensor)
            if scale <= 0.0:
                continue
            half_width = 0.5 * noise_scale * scale
            noise = rng.uniform(
                low=-half_width,
                high=half_width,
                size=tensor.shape,
            ).astype(tensor.dtype, copy=False)
            grad_list[tensor_idx] = tensor + noise

    metadata = randomized.setdefault("metadata", {})
    metadata.update(
        {
            "randomized_dequantization": True,
            "randomized_dequantization_base": str(base_update_file),
            "ensemble_member": member_idx,
            "ensemble_noise_seed": noise_seed,
            "ensemble_noise_scale": noise_scale,
            "ensemble_noise_distribution": "uniform(-delta/2, delta/2)",
        }
    )
    _write_pickle(randomized, output_file)
    return output_file


def main() -> None:
    args = parse_args()
    if args.ensemble_size < 1:
        raise ValueError("--ensemble_size must be >= 1")
    if args.start_member < 0 or args.start_member >= args.ensemble_size:
        raise ValueError("--start_member must be in [0, ensemble_size)")
    if args.noise_scale < 0:
        raise ValueError("--noise_scale must be non-negative")

    experiment_name = f"{EXPERIMENT_NAME}_{args.rounding}"
    base_update_file = output_pickle_path(
        experiment_name,
        args.ds,
        args.model,
        args.n_samples,
    )

    if not args.reuse_existing or not base_update_file.exists():
        collect_fedavg_update_pickle(
            ds=args.ds,
            model_name=args.model,
            h_dim=args.h_dim,
            n_samples=args.n_samples,
            n_rounds=args.n_rounds,
            local_epochs=args.local_epochs,
            local_batch_size=resolve_local_batch_size(args),
            lr=args.local_lr,
            quant_bits=4,
            rounding=args.rounding,
            output_file=base_update_file,
            seed=args.seed,
            global_checkpoint=args.global_checkpoint,
        )

    member_dir = base_update_file.parent / "randomized_members"
    for member_idx in range(args.start_member, args.ensemble_size):
        member_file = member_dir / f"{args.n_samples}_k{member_idx:02d}.pickle"
        if not args.reuse_existing or not member_file.exists():
            randomized_dequant_pickle(
                base_update_file=base_update_file,
                output_file=member_file,
                member_idx=member_idx,
                noise_seed=args.noise_seed,
                noise_scale=args.noise_scale,
            )

        if args.run_attack:
            run_attack_with_update_file(
                args,
                exp_name=f"{experiment_name}_k{member_idx:02d}",
                update_file=member_file,
            )


if __name__ == "__main__":
    main()
