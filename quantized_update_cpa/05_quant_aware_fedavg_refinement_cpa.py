#!/usr/bin/env python3
"""05: CPA-initialized quantization-aware FedAvg update refinement."""

from __future__ import annotations

import argparse
import copy
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
from fedavg_refinement import run_fedavg_refinement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine CPA/FIA initialized images by differentiably unrolling the "
            "same full-batch local SGD used by the FedAvg collector and matching "
            "theta_T - theta_0 to observed int4 quantization bins."
        )
    )
    add_collect_args(parser)
    parser.add_argument("--rounding", type=str, default="nearest", choices=["nearest"])
    parser.add_argument("--label_mode", type=str, default="known", choices=["known", "unknown"])
    parser.add_argument("--init_run_path", type=Path, default=None)
    parser.add_argument(
        "--run_init_attack",
        action="store_true",
        help="Run author CPA+Direct FIA first with n_sample_fi forced to n_samples.",
    )
    parser.add_argument("--refine_n_iter", type=int, default=100)
    parser.add_argument("--refine_n_log", type=int, default=10)
    parser.add_argument("--refine_lr", type=float, default=1e-2)
    parser.add_argument("--refine_update_weight", type=float, default=1.0)
    parser.add_argument("--refine_tv_weight", type=float, default=1e-4)
    parser.add_argument("--display_samples", type=int, default=16)
    parser.add_argument("--fp32_repro_max_abs_tol", type=float, default=1e-6)
    parser.add_argument("--fp32_repro_rel_l2_tol", type=float, default=1e-5)
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

    init_run_path = args.init_run_path
    if init_run_path is None:
        if not args.run_init_attack:
            raise SystemExit(
                "05 needs a CPA/FIA initialization with all n_samples images. "
                "Pass --init_run_path or use --run_init_attack for a small smoke run."
            )
        init_args = copy.copy(args)
        init_args.n_sample_fi = args.n_samples
        init_args.fi_method = "direct"
        init_run_path = run_author_cpa_with_update_file(
            init_args,
            exp_name=f"05_init_author_cpa_{run_tag(init_args)}",
            update_file=int4_update_file,
            method="Author CPA+Direct FIA initialization for 05 FedAvg refinement",
        )

    output_dir = run_fedavg_refinement(
        args,
        int4_update_file=int4_update_file,
        fp32_update_file=fp32_update_file,
        init_run_path=Path(init_run_path),
    )
    print(f"05 output: {output_dir}")


if __name__ == "__main__":
    main()
