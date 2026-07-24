#!/usr/bin/env python3
"""03: Heuristic randomized dequantization ensemble for author CPA."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from author_runner import run_author_cpa_with_update_file, run_direct_fia_from_embeddings
from blind_alignment import aggregate_blind_ensemble
from common import (
    OUTPUT_ROOT,
    add_attack_args,
    add_collect_args,
    assert_matching_manifest,
    derive_quantized_update_pickle,
    ensure_shared_fp32_update,
    hash_tensor_sequence,
    read_pickle,
    run_tag,
    sha256_bytes,
    sha256_file,
    write_json,
)
from quantization import randomized_dequantize_tensor


EXPERIMENT_NAME = "03_randomized_dequant_ensemble_cpa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create heuristic uniform-bin dequantized int4 update members and "
            "run the public author CPA classes without oracle ensemble alignment."
        )
    )
    add_collect_args(parser)
    parser.add_argument("--rounding", type=str, default="nearest", choices=["nearest"])
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--noise_seed", type=int, default=1234)
    parser.add_argument(
        "--perturb_scope",
        type=str,
        default="all",
        choices=["all", "attack_layer"],
        help="Perturb every parameter tensor or only the CPA attack layer.",
    )
    parser.add_argument("--aggregate", type=str, default="mean", choices=["mean", "median"])
    parser.add_argument("--reference", type=str, default="first", choices=["first", "medoid"])
    parser.add_argument("--run_aggregate_fia", action="store_true")
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


def member_update_path(args: argparse.Namespace, int4_update_file: Path, member_idx: int) -> Path:
    return (
        OUTPUT_ROOT
        / EXPERIMENT_NAME
        / args.ds
        / args.model
        / "updates"
        / int4_update_file.parent.name
        / f"{args.perturb_scope}_seed{args.noise_seed}"
        / f"member_{member_idx:03d}.pickle"
    )


def _save_pickle(path: Path, value: Any) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def randomized_dequant_pickle(
    *,
    int4_update_file: Path,
    output_file: Path,
    member_idx: int,
    noise_seed: int,
    perturb_scope: str,
) -> Path:
    data = read_pickle(int4_update_file)
    if data.get("metadata", {}).get("quant_bits") != 4:
        raise ValueError("Randomized dequantization requires an int4 artifact")
    if data.get("metadata", {}).get("rounding") != "nearest":
        raise ValueError("Randomized dequantization currently supports nearest rounding only")
    if not data.get("quant_codes"):
        raise ValueError("int4 artifact is missing quant_codes")

    manifest = {
        "schema_version": 2,
        "method": "heuristic_uniform_bin_randomized_dequantization",
        "source_int4_path": str(int4_update_file.resolve()),
        "source_int4_hash": sha256_file(int4_update_file),
        "member_idx": member_idx,
        "noise_seed": noise_seed,
        "perturb_scope": perturb_scope,
        "saturation_policy": "codes -7 and 7 kept at q*s; exact posterior is not claimed",
    }
    manifest_file = output_file.parent / f"member_{member_idx:03d}.manifest.json"
    if output_file.exists():
        assert_matching_manifest(manifest_file, manifest)
        return output_file

    randomized = copy.deepcopy(data)
    attack_layer_index = int(data["attack_layer"]["index"])
    randomized["grad"] = []
    generator = torch.Generator(device="cpu").manual_seed(noise_seed + 1_000_003 * member_idx)
    for trial_idx, (codes_by_tensor, grad_by_tensor) in enumerate(zip(data["quant_codes"], data["grad"])):
        stats = data["quant_stats"][trial_idx]["per_tensor"]
        trial_grad = []
        for tensor_idx, (codes_np, grad_np) in enumerate(zip(codes_by_tensor, grad_by_tensor)):
            if perturb_scope == "attack_layer" and tensor_idx != attack_layer_index:
                trial_grad.append(np.asarray(grad_np, dtype=np.float32))
                continue
            scale = float(stats[tensor_idx]["scale"])
            codes = torch.as_tensor(codes_np, dtype=torch.int8)
            dequant = randomized_dequantize_tensor(codes, scale, generator=generator)
            trial_grad.append(dequant.cpu().numpy().astype(np.float32, copy=False))
        randomized["grad"].append(trial_grad)

    randomized["metadata"].update(
        {
            "randomized_dequantization": True,
            "randomized_dequantization_interpretation": "heuristic uniform-bin perturbation, not exact posterior sampling",
            "randomized_dequantization_base": str(int4_update_file.resolve()),
            "source_int4_update_hash": sha256_file(int4_update_file),
            "ensemble_member": member_idx,
            "ensemble_noise_seed": noise_seed,
            "ensemble_perturb_scope": perturb_scope,
            "ensemble_saturation_policy": "q=-7 and q=7 are not sampled from unbounded tails",
            "update_tensor_hash": hash_tensor_sequence(randomized["grad"]),
            "randomized_member_hash": sha256_bytes(
                f"{sha256_file(int4_update_file)}:{member_idx}:{noise_seed}:{perturb_scope}".encode("utf-8")
            ),
        }
    )
    _save_pickle(output_file, randomized)
    write_json(manifest_file, manifest)
    return output_file


def _load_member_embeddings(run_paths: list[Path], n_batches: int) -> list[list[np.ndarray]]:
    per_member = []
    for path in run_paths:
        rec_path = path / "reconstructions.pkl"
        if not rec_path.exists():
            raise FileNotFoundError(f"Missing member reconstruction file: {rec_path}")
        rows = read_pickle(rec_path)
        if len(rows) < n_batches:
            raise ValueError(f"Member {path} has {len(rows)} batches, expected {n_batches}")
        per_member.append([np.asarray(rows[batch]["rec_emb_attack_order"]) for batch in range(n_batches)])
    return per_member


def _aggregate_member_embeddings(
    *,
    args: argparse.Namespace,
    run_paths: list[Path],
    int4_update_file: Path,
) -> tuple[Path, list[np.ndarray]]:
    data = read_pickle(int4_update_file)
    n_batches = min(args.attack_n_batch or int(data["metadata"]["attack_trials"]), int(data["metadata"]["attack_trials"]))
    per_member = _load_member_embeddings(run_paths, n_batches)
    aggregated_batches = []
    stats_rows = []
    for batch in range(n_batches):
        batch_members = [member[batch] for member in per_member]
        aggregate, stats = aggregate_blind_ensemble(
            batch_members, aggregate=args.aggregate, reference=args.reference
        )
        aggregated_batches.append(aggregate)
        stats_rows.append({"batch": batch, **stats})

    output_dir = (
        OUTPUT_ROOT
        / EXPERIMENT_NAME
        / args.ds
        / args.model
        / "blind_aggregates"
        / f"{run_tag(args)}_{args.perturb_scope}_k{args.ensemble_size}_{args.aggregate}_{args.reference}"
    )
    output = output_dir / "blind_ensemble_embeddings.pkl"
    _save_pickle(
        output,
        {
            "embedding_batches": aggregated_batches,
            "stats": stats_rows,
            "policy": "blind abs-cosine Hungarian alignment with sign correction",
            "oracle_inputs_used": False,
        },
    )
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 2,
            "method": "blind_ensemble_alignment",
            "source_run_paths": [str(path.resolve()) for path in run_paths],
            "source_int4_update_hash": sha256_file(int4_update_file),
            "aggregate": args.aggregate,
            "reference": args.reference,
            "ensemble_size": args.ensemble_size,
            "perturb_scope": args.perturb_scope,
            "oracle_inputs_used": False,
        },
    )
    return output, aggregated_batches


def main() -> None:
    args = parse_args()
    if args.ensemble_size < 1:
        raise ValueError("--ensemble_size must be >= 1")

    fp32_update_file = ensure_shared_fp32_update(args)
    int4_update_file = derive_quantized_update_pickle(
        fp32_update_file=fp32_update_file,
        output_file=int4_update_path(args, fp32_update_file),
        rounding=args.rounding,
    )

    member_paths = []
    run_paths = []
    for member_idx in range(args.ensemble_size):
        member_file = randomized_dequant_pickle(
            int4_update_file=int4_update_file,
            output_file=member_update_path(args, int4_update_file, member_idx),
            member_idx=member_idx,
            noise_seed=args.noise_seed,
            perturb_scope=args.perturb_scope,
        )
        member_paths.append(member_file)
        if args.run_attack:
            run_path = run_author_cpa_with_update_file(
                args,
                exp_name=(
                    f"{EXPERIMENT_NAME}_nearest_{run_tag(args)}_"
                    f"{args.perturb_scope}_k{member_idx:03d}"
                ),
                update_file=member_file,
                method="Heuristic randomized dequantization ensemble member CPA",
            )
            run_paths.append(run_path)

    if args.run_attack and args.ensemble_size > 1:
        aggregate_file, aggregated = _aggregate_member_embeddings(
            args=args,
            run_paths=run_paths,
            int4_update_file=int4_update_file,
        )
        print(f"blind ensemble aggregate: {aggregate_file}")
        if args.run_aggregate_fia:
            run_direct_fia_from_embeddings(
                args,
                exp_name=(
                    f"{EXPERIMENT_NAME}_blindagg_{run_tag(args)}_"
                    f"{args.perturb_scope}_k{args.ensemble_size}_{args.aggregate}_{args.reference}"
                ),
                update_file=int4_update_file,
                recovered_embeddings=aggregated,
                method="Direct FIA from blind randomized-dequantization ensemble aggregate",
            )
    else:
        print("member update artifacts:")
        for path in member_paths:
            print(path)


if __name__ == "__main__":
    main()
