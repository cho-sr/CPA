#!/usr/bin/env python3
"""01: Naive client-level Global Top-k sparse update CPA.

One-sample client updates are sparsified before uniform aggregation.  The
resulting dense zero-filled aggregate tensors are fed to CPA without masks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from author_runner import run_author_cpa_with_update_file
from common import (
    AUTHOR_SOURCE_HASH,
    OUTPUT_ROOT,
    add_attack_args,
    add_collect_args,
    client_sparse_collector_config,
    collect_client_then_aggregate_sparse_pickles,
    hash_json,
    run_tag,
)
from sparsification import (
    OBSERVED_ZERO_POLICY,
    format_keep_ratio,
    parse_keep_ratio_or_percent,
    require_global_topk,
    validate_keep_ratio,
)


EXPERIMENT_NAME = "01_naive_sparse_cpa"
DEFAULT_KEEP_RATIOS = [1.0, 0.80, 0.20, 0.10, 0.05, 0.01]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build dense zero-filled observations by applying full-model Global "
            "Top-k to each one-sample client update before uniform aggregation, "
            "then run official CPA/FIA without sparse-aware correction."
        )
    )
    add_collect_args(parser)
    parser.add_argument("--keep_ratios", type=parse_keep_ratio_or_percent, nargs="+", default=DEFAULT_KEEP_RATIOS)
    parser.add_argument("--sparse_policy", type=str, default="global_topk", choices=["global_topk"])
    parser.add_argument("--success_lpips_threshold", type=float, default=0.50)
    add_attack_args(parser)
    return parser.parse_args()


def client_sparse_group_tag(args: argparse.Namespace) -> str:
    config = client_sparse_collector_config(
        ds=args.ds,
        model_name=args.model,
        h_dim=args.h_dim,
        n_samples=args.n_samples,
        attack_trials=args.attack_trials,
        local_epochs=args.local_epochs,
        lr=args.local_lr,
        seed=args.seed,
        checkpoint_path=args.global_checkpoint,
    )
    return (
        f"client_then_aggregate_n{args.n_samples}_steps{args.local_epochs}"
        f"_trials{args.attack_trials}_{hash_json(config)[:16]}"
    )


def sparse_update_path(args: argparse.Namespace, keep_ratio: float) -> Path:
    ratio_tag = format_keep_ratio(keep_ratio)
    return (
        OUTPUT_ROOT
        / EXPERIMENT_NAME
        / args.ds
        / args.model
        / "updates"
        / client_sparse_group_tag(args)
        / "client_then_aggregate"
        / args.sparse_policy
        / f"keep_{ratio_tag}.pickle"
    )


def main() -> None:
    args = parse_args()
    require_global_topk(args.sparse_policy)
    for keep_ratio in args.keep_ratios:
        validate_keep_ratio(keep_ratio)
    if args.success_lpips_threshold <= 0:
        raise ValueError("--success_lpips_threshold must be positive")
    args.observed_zero_policy = OBSERVED_ZERO_POLICY
    args.sparsification_application = "client_then_aggregate"
    args.client_granularity = "one_sample_per_client"

    output_files = {float(keep_ratio): sparse_update_path(args, keep_ratio) for keep_ratio in args.keep_ratios}
    sparse_files = collect_client_then_aggregate_sparse_pickles(
        ds=args.ds,
        model_name=args.model,
        h_dim=args.h_dim,
        n_samples=args.n_samples,
        local_epochs=args.local_epochs,
        lr=args.local_lr,
        output_files=output_files,
        seed=args.seed,
        global_checkpoint=args.global_checkpoint,
        sparse_policy=args.sparse_policy,
        author_source_hash=AUTHOR_SOURCE_HASH,
        attack_trials=args.attack_trials,
        collect_log_every=args.collect_log_every,
    )
    for keep_ratio in args.keep_ratios:
        sparse_file = sparse_files[float(keep_ratio)]
        if args.run_attack:
            ratio_tag = format_keep_ratio(keep_ratio)
            run_author_cpa_with_update_file(
                args,
                exp_name=f"{EXPERIMENT_NAME}_client_topk_{args.sparse_policy}_keep{ratio_tag}_{run_tag(args)}",
                update_file=sparse_file,
                method=(
                    "Naive Sparse CPA: one-sample client updates are full-model "
                    "Global Top-k sparsified before uniform aggregation; CPA "
                    "uses Y_obs = [delta_theta_obs]_attack_fc ~= A Z."
                ),
            )

    if not args.run_attack:
        print("sparse update artifacts:")
        for keep_ratio in args.keep_ratios:
            print(sparse_files[float(keep_ratio)])


if __name__ == "__main__":
    main()
