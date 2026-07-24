#!/usr/bin/env python3
"""04: Quantization-aware ICA objective ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    OUTPUT_ROOT,
    add_attack_args,
    add_collect_args,
    derive_quantized_update_pickle,
    ensure_shared_fp32_update,
)
from quant_aware_ica import ABLATIONS, run_quant_aware_ica


EXPERIMENT_NAME = "04_quantization_aware_ica"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a separate quantization-aware ICA proposal: whitened observation "
            "+ unmixing W, bin-distance quantization consistency, standardized "
            "source negentropy, and Q/NE ablations."
        )
    )
    add_collect_args(parser)
    parser.add_argument("--rounding", type=str, default="nearest", choices=["nearest"])
    parser.add_argument("--qica_ablation", type=str, default="all", choices=["all", *ABLATIONS.keys()])
    parser.add_argument("--qica_n_iter", type=int, default=1000)
    parser.add_argument("--qica_n_log", type=int, default=100)
    parser.add_argument("--qica_lr", type=float, default=1e-3)
    parser.add_argument("--qica_q_weight", type=float, default=1.0)
    parser.add_argument("--qica_columns_per_step", type=int, default=8192)
    parser.add_argument("--qica_eps", type=float, default=1e-6)
    parser.add_argument(
        "--qica_train_observation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the attack-layer observation candidate to move inside observed quantization bins.",
    )
    parser.add_argument(
        "--run_fia",
        action="store_true",
        help="After 04 source recovery, run a separate author Direct FIA stage.",
    )
    add_attack_args(parser)
    return parser.parse_args()


def int4_update_path(args: argparse.Namespace, fp32_update_file: Path) -> Path:
    return (
        OUTPUT_ROOT
        / "02_naive_int4_update_cpa"
        / args.ds
        / args.model
        / "updates"
        / fp32_update_file.parent.name
        / "int4_nearest.pickle"
    )


def main() -> None:
    args = parse_args()
    fp32_update_file = ensure_shared_fp32_update(args)
    int4_update_file = derive_quantized_update_pickle(
        fp32_update_file=fp32_update_file,
        output_file=int4_update_path(args, fp32_update_file),
        rounding=args.rounding,
    )
    output_dirs = run_quant_aware_ica(args, update_file=int4_update_file)
    for path in output_dirs:
        print(f"04 output: {path}")


if __name__ == "__main__":
    main()
