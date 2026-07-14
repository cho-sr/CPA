#!/usr/bin/env python3
"""Stage 3: quantization-aware ICA objective with Q(A S) consistency."""

from __future__ import annotations

import argparse

from common import (
    add_attack_args,
    add_collect_args,
    collect_fedavg_update_pickle,
    output_pickle_path,
    resolve_local_batch_size,
)
from qas_attack import run_quant_aware_as_attack_with_update_file


EXPERIMENT_NAME = "04_quant_aware_ica_objective_cpa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect 4-bit dequantized FedAvg updates and optimize a "
            "quantization-aware ICA objective: min_{A,S} ||D(Q(A S)) - Y_q||^2 + R(S)."
        )
    )
    add_collect_args(parser)
    parser.add_argument(
        "--rounding",
        type=str,
        default="nearest",
        choices=["nearest", "stochastic"],
        help="Rounding mode for 4-bit update fake quantization.",
    )
    parser.add_argument(
        "--qas_rows_per_step",
        type=int,
        default=512,
        help="Number of attack-layer rows sampled per Q(A S) consistency step.",
    )
    parser.add_argument(
        "--qas_qcons",
        type=float,
        default=10.0,
        help="Weight for normalized quantized-gradient consistency.",
    )
    parser.add_argument(
        "--qas_lr",
        type=float,
        default=1e-5,
        help="Learning rate for the quantization-aware A,S objective.",
    )
    parser.add_argument(
        "--qas_ind",
        type=float,
        default=1.0,
        help="Weight for source independence penalty.",
    )
    parser.add_argument(
        "--qas_ne",
        type=float,
        default=0.01,
        help="Weight for non-Gaussianity reward.",
    )
    parser.add_argument(
        "--qas_l1",
        type=float,
        default=0.001,
        help="Weight for source sparsity prior.",
    )
    parser.add_argument(
        "--qas_nv",
        type=float,
        default=0.001,
        help="Weight for sign-ambiguity/non-negativity prior.",
    )
    parser.add_argument(
        "--qas_a_decor",
        type=float,
        default=0.1,
        help="Weight for mixing-column decorrelation.",
    )
    add_attack_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.quant_bits = 4
    args.attack_lr = args.qas_lr
    experiment_name = f"{EXPERIMENT_NAME}_{args.rounding}_qas"
    output_file = output_pickle_path(
        experiment_name,
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
            quant_bits=4,
            rounding=args.rounding,
            output_file=output_file,
            seed=args.seed,
            global_checkpoint=args.global_checkpoint,
        )

    if args.run_attack:
        run_quant_aware_as_attack_with_update_file(
            args,
            exp_name=experiment_name,
            update_file=output_file,
        )


if __name__ == "__main__":
    main()
