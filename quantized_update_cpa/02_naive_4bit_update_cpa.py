#!/usr/bin/env python3
"""02: Naive symmetric int4 dequantized FedAvg-update CPA."""

from __future__ import annotations

import argparse
from pathlib import Path

from author_runner import run_author_cpa_with_update_file
from common import (
    OUTPUT_ROOT,
    add_attack_args,
    add_collect_args,
    derive_quantized_update_pickle,
    ensure_shared_fp32_update,
    run_tag,
)


EXPERIMENT_NAME = "02_naive_int4_update_cpa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive per-parameter symmetric int4 updates from the shared FP32 "
            "FedAvg artifact, then run the public author CPA classes."
        )
    )
    add_collect_args(parser)
    parser.add_argument("--rounding", type=str, default="nearest", choices=["nearest"])
    add_attack_args(parser)
    return parser.parse_args()


def int4_update_path(args: argparse.Namespace, fp32_update_file: Path) -> Path:
    fp32_hash_tag = fp32_update_file.parent.name
    return (
        OUTPUT_ROOT
        / EXPERIMENT_NAME
        / args.ds
        / args.model
        / "updates"
        / fp32_hash_tag
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
    if args.run_attack:
        run_author_cpa_with_update_file(
            args,
            exp_name=f"{EXPERIMENT_NAME}_nearest_{run_tag(args)}",
            update_file=int4_update_file,
            method="Naive symmetric int4 dequantized FedAvg-update CPA",
        )
    else:
        print(f"int4 update artifact: {int4_update_file}")


if __name__ == "__main__":
    main()
