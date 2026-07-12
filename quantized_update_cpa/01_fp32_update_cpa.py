#!/usr/bin/env python3
"""Stage 1: FP32 FedAvg local-update CPA baseline."""

from __future__ import annotations

import argparse

from common import (
    add_attack_args,
    add_collect_args,
    collect_fedavg_update_pickle,
    output_pickle_path,
    resolve_local_batch_size,
    run_attack_with_update_file,
)


EXPERIMENT_NAME = "01_fp32_update_cpa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect FP32 FedAvg local updates and optionally run CPA."
    )
    add_collect_args(parser)
    add_attack_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = output_pickle_path(
        EXPERIMENT_NAME,
        args.ds,
        args.model,
        args.n_samples,
    )

    if not args.reuse_existing or not output_file.exists():
        collect_fedavg_update_pickle(
            ds=args.ds,
            model_name=args.model,
            h_dim=args.h_dim,
            n_samples=args.n_samples,
            n_rounds=args.n_rounds,
            local_epochs=args.local_epochs,
            local_batch_size=resolve_local_batch_size(args),
            lr=args.local_lr,
            quant_bits=32,
            rounding="nearest",
            output_file=output_file,
            seed=args.seed,
            global_checkpoint=args.global_checkpoint,
        )

    if args.run_attack:
        run_attack_with_update_file(
            args,
            exp_name=EXPERIMENT_NAME,
            update_file=output_file,
        )


if __name__ == "__main__":
    main()
