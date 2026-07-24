#!/usr/bin/env python3
"""01: Unmodified author CPA attack on our FP32 FedAvg updates."""

from __future__ import annotations

import argparse

from author_runner import run_author_cpa_with_update_file
from common import add_attack_args, add_collect_args, ensure_shared_fp32_update, run_tag


EXPERIMENT_NAME = "01_author_fp32_update_cpa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the public author CPA classes on shared FP32 FedAvg local updates."
    )
    add_collect_args(parser)
    add_attack_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fp32_update_file = ensure_shared_fp32_update(args)
    if args.run_attack:
        run_author_cpa_with_update_file(
            args,
            exp_name=f"{EXPERIMENT_NAME}_{run_tag(args)}",
            update_file=fp32_update_file,
            method="Unmodified author CPA attack on our FP32 FedAvg updates",
        )
    else:
        print(f"FP32 update artifact: {fp32_update_file}")


if __name__ == "__main__":
    main()
