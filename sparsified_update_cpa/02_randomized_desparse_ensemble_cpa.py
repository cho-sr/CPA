#!/usr/bin/env python3
"""02: Fixed sparse observation randomized desparsification ensemble CPA."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from author_runner import run_author_cpa_with_update_file, run_direct_fia_from_embeddings
from blind_alignment import blind_medoid_align_and_average
from common import (
    AUTHOR_SOURCE_HASH,
    OUTPUT_ROOT,
    add_attack_args,
    add_collect_args,
    client_sparse_collector_config,
    collect_client_then_aggregate_sparse_pickles,
    hash_json,
    read_pickle,
    run_tag,
    sha256_bytes,
    sha256_file,
    write_json,
    write_pickle,
)
from sparsification import (
    OBSERVED_ZERO_POLICY,
    RANDOMIZED_DESPARSE_POLICY,
    format_keep_ratio,
    parse_keep_ratio_or_percent,
    require_global_topk,
    unpack_bool_mask,
    validate_keep_ratio,
)


BASELINE_EXPERIMENT_NAME = "01_naive_sparse_cpa"
EXPERIMENT_NAME = "02_randomized_desparse_ensemble_cpa"
DEFAULT_KEEP_RATIOS = [0.10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fix each client's Global Top-k sparse observation, randomly fill only "
            "dropped attack-FC coordinates from U(-tau_k, tau_k), run member CPA, "
            "then blind-align and average recovered sources."
        )
    )
    add_collect_args(parser)
    parser.add_argument("--keep_ratios", type=parse_keep_ratio_or_percent, nargs="+", default=DEFAULT_KEEP_RATIOS)
    parser.add_argument("--sparse_policy", type=str, default="global_topk", choices=["global_topk"])
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--noise_seed", type=int, default=1234)
    parser.add_argument("--run_aggregate_fia", action="store_true")
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
        / BASELINE_EXPERIMENT_NAME
        / args.ds
        / args.model
        / "updates"
        / client_sparse_group_tag(args)
        / "client_then_aggregate"
        / args.sparse_policy
        / f"keep_{ratio_tag}.pickle"
    )


def observation_path(args: argparse.Namespace, keep_ratio: float) -> Path:
    ratio_tag = format_keep_ratio(keep_ratio)
    return (
        OUTPUT_ROOT
        / EXPERIMENT_NAME
        / args.ds
        / args.model
        / "fixed_observations"
        / client_sparse_group_tag(args)
        / "client_then_aggregate"
        / args.sparse_policy
        / f"keep_{ratio_tag}_attack_layer_observations.pickle"
    )


def ensemble_output_dir(args: argparse.Namespace, keep_ratio: float) -> Path:
    ratio_tag = format_keep_ratio(keep_ratio)
    return (
        OUTPUT_ROOT
        / EXPERIMENT_NAME
        / args.ds
        / args.model
        / "blind_aggregates"
        / client_sparse_group_tag(args)
        / args.sparse_policy
        / f"keep_{ratio_tag}"
        / f"R{args.ensemble_size}_seed{args.noise_seed}_medoid_mean"
    )


def make_member_seed_plan(ensemble_size: int, noise_seed: int) -> list[dict[str, Any]]:
    return [
        {
            "member": member_idx,
            "desparsification_seed": int(noise_seed + 1_000_003 * member_idx),
            "cpa_seed": "fixed_from_args_seed",
            "trial_varying_randomness": "desparsification_noise_only",
        }
        for member_idx in range(ensemble_size)
    ]


def _hash_arrays(values: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.asarray(value)
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _member_update_hash(base_hash: str, attack_layer_hash: str, member_idx: int, member_seed: int) -> str:
    return sha256_bytes(
        json.dumps(
            {
                "base_update_tensor_hash": base_hash,
                "randomized_attack_layer_hash": attack_layer_hash,
                "member": member_idx,
                "desparsification_seed": member_seed,
            },
            sort_keys=True,
        ).encode("utf-8")
    )


def _load_observation_payload(
    observation: Mapping[str, Any],
    *,
    observation_index: Path | None,
) -> tuple[Mapping[str, Any], np.ndarray]:
    if "mask" in observation and "values" in observation:
        return observation["mask"], np.asarray(observation["values"], dtype=np.float32)
    shard_file = observation.get("shard_file")
    if shard_file is None:
        raise ValueError("observation row must contain inline mask/values or shard_file")
    if observation_index is None:
        raise ValueError("sharded observation rows require an observation index path")
    shard_path = observation_index.parent / str(shard_file)
    with np.load(shard_path) as payload:
        mask = {
            "shape": payload["mask_shape"].astype(np.int64).tolist(),
            "packed_bits": payload["mask_packed_bits"].astype(np.uint8, copy=False),
        }
        values = payload["values"].astype(np.float32, copy=False)
    return mask, values


def _randomized_client_attack_layer(
    observation: Mapping[str, Any],
    *,
    shape: tuple[int, ...],
    rng: np.random.Generator,
    observation_index: Path | None = None,
) -> np.ndarray:
    packed_mask, values = _load_observation_payload(observation, observation_index=observation_index)
    mask = unpack_bool_mask(packed_mask)
    if tuple(mask.shape) != shape:
        raise ValueError(f"attack-layer mask shape mismatch: {tuple(mask.shape)} vs {shape}")
    observed_count = int(mask.sum())
    if values.size != observed_count:
        raise ValueError(f"observed value count mismatch: values={values.size}, mask={observed_count}")

    if observed_count == int(mask.size):
        return values.reshape(shape).astype(np.float32, copy=True)

    tau = float(observation["tau"])
    if tau < 0.0 or not np.isfinite(tau):
        raise ValueError("tau must be finite and non-negative")
    if tau == 0.0:
        sampled = np.zeros(shape, dtype=np.float32)
    else:
        sampled = rng.uniform(-tau, tau, size=shape).astype(np.float32, copy=False)
    sampled[mask] = values
    return sampled


def randomized_attack_layer_aggregate(
    observation_data: Mapping[str, Any],
    *,
    batch_idx: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    shape = tuple(int(value) for value in observation_data["attack_layer"]["shape"])
    rows = observation_data["observations"][batch_idx]
    if not rows:
        raise ValueError(f"no client observations for batch {batch_idx}")

    aggregate = np.zeros(shape, dtype=np.float32)
    tau_values = []
    observed_fractions = []
    rng = np.random.default_rng(int(seed))
    observation_index_value = observation_data.get("_index_path")
    observation_index = Path(str(observation_index_value)) if observation_index_value else None
    for row in rows:
        client_sample = _randomized_client_attack_layer(
            row,
            shape=shape,
            rng=rng,
            observation_index=observation_index,
        )
        weight = float(row["aggregation_weight"])
        aggregate += client_sample * weight
        tau_values.append(float(row["tau"]))
        observed_fractions.append(float(row["observed_elements"]) / max(1.0, float(row["total_elements"])))
    tau_array = np.asarray(tau_values, dtype=np.float64)
    observed_array = np.asarray(observed_fractions, dtype=np.float64)
    return aggregate.astype(np.float32, copy=False), {
        "batch": batch_idx,
        "clients": len(rows),
        "desparsification_seed": int(seed),
        "sampler": "uniform",
        "missing_value_abs_bound_policy": "per_client_global_topk_min_observed_abs_tau",
        "tau_mean": float(tau_array.mean()),
        "tau_min": float(tau_array.min()),
        "tau_max": float(tau_array.max()),
        "attack_layer_observed_fraction_mean": float(observed_array.mean()),
        "aggregate_min": float(aggregate.min()),
        "aggregate_max": float(aggregate.max()),
    }


def materialize_member_update_data(
    *,
    base_data: Mapping[str, Any],
    base_update_file: Path,
    base_update_file_hash: str,
    observation_file: Path | None,
    observation_file_hash: str | None,
    observation_data: Mapping[str, Any] | None,
    keep_ratio: float,
    member_idx: int,
    member_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attack_layer_index = int(base_data["attack_layer"]["index"])
    member_grad = []
    attack_layers = []
    batch_stats = []
    for batch_idx, batch_grad in enumerate(base_data["grad"]):
        batch_values = list(batch_grad)
        if observation_data is None:
            attack_layer = np.asarray(batch_values[attack_layer_index], dtype=np.float32)
            stats = {
                "batch": batch_idx,
                "desparsification_seed": int(member_seed + 9_176_521 * batch_idx),
                "sampler": "deterministic_identity",
                "keep_ratio": float(keep_ratio),
                "note": "keep_ratio=1.0 has no dropped coordinates",
            }
        else:
            attack_layer, stats = randomized_attack_layer_aggregate(
                observation_data,
                batch_idx=batch_idx,
                seed=int(member_seed + 9_176_521 * batch_idx),
            )
        batch_values[attack_layer_index] = attack_layer
        member_grad.append(batch_values)
        attack_layers.append(attack_layer)
        batch_stats.append(stats)

    attack_layer_hash = _hash_arrays(attack_layers)
    metadata = copy.deepcopy(base_data["metadata"])
    metadata.update(
        {
            "randomized_desparsification": observation_data is not None,
            "randomized_desparsification_policy": RANDOMIZED_DESPARSE_POLICY,
            "fixed_sparse_observation": True,
            "desparsification_scope": "attack_layer",
            "missing_value_sampler": "uniform",
            "missing_value_abs_bound_policy": "per_client_global_topk_min_observed_abs_tau",
            "ensemble_member": int(member_idx),
            "ensemble_desparsification_seed": int(member_seed),
            "source_sparse_update_file": str(base_update_file.resolve()),
            "source_sparse_update_hash": base_update_file_hash,
            "source_client_attack_observation_file": str(observation_file.resolve()) if observation_file else None,
            "source_client_attack_observation_hash": observation_file_hash,
            "update_file_hash_override": base_update_file_hash,
            "randomized_attack_layer_hash": attack_layer_hash,
            "update_tensor_hash": _member_update_hash(
                str(base_data["metadata"]["update_tensor_hash"]), attack_layer_hash, member_idx, member_seed
            ),
        }
    )

    member_data = dict(base_data)
    member_data["metadata"] = metadata
    member_data["grad"] = member_grad
    return member_data, batch_stats


def _load_member_sources(run_paths: Sequence[Path], n_batches: int) -> tuple[list[list[np.ndarray]], str]:
    per_member = []
    source_kind: str | None = None
    for path in run_paths:
        rows = read_pickle(path / "reconstructions.pkl")
        if len(rows) < n_batches:
            raise ValueError(f"Member {path} has {len(rows)} batches, expected {n_batches}")
        member_batches = []
        for batch in range(n_batches):
            rec_emb = np.asarray(rows[batch]["rec_emb_attack_order"], dtype=np.float32)
            if rec_emb.size > 0:
                current_kind = "rec_emb_attack_order"
                value = rec_emb
            else:
                current_kind = "rec_gi"
                value = np.asarray(rows[batch]["rec_gi"], dtype=np.float32)
            if source_kind is None:
                source_kind = current_kind
            elif source_kind != current_kind:
                raise ValueError(f"Mixed member source kinds: {source_kind} and {current_kind}")
            member_batches.append(value)
        per_member.append(member_batches)
    if source_kind is None:
        raise ValueError("no member sources found")
    return per_member, source_kind


def aggregate_member_sources(
    *,
    args: argparse.Namespace,
    keep_ratio: float,
    base_update_file: Path,
    base_update_file_hash: str,
    observation_file: Path | None,
    observation_file_hash: str | None,
    run_paths: Sequence[Path],
    seed_plan: Sequence[Mapping[str, Any]],
) -> tuple[Path, list[np.ndarray], str]:
    base_data = read_pickle(base_update_file)
    n_batches = min(args.attack_n_batch or int(base_data["metadata"]["attack_trials"]), int(base_data["metadata"]["attack_trials"]))
    per_member, source_kind = _load_member_sources(run_paths, n_batches)
    final_batches = []
    aligned_batches = []
    stats_rows = []
    for batch in range(n_batches):
        member_sources = [member[batch] for member in per_member]
        final_sources, aligned_stack, stats = blind_medoid_align_and_average(member_sources)
        final_batches.append(final_sources)
        aligned_batches.append(aligned_stack)
        stats_rows.append({"batch": batch, **stats})

    output_dir = ensemble_output_dir(args, keep_ratio)
    output_file = output_dir / "blind_ensemble_sources.pkl"
    write_pickle(
        output_file,
        {
            "source_batches": final_batches,
            "aligned_member_batches": aligned_batches,
            "alignment_stats": stats_rows,
            "source_kind": source_kind,
            "policy": "blind medoid permutation/sign/least-squares-scale alignment followed by mean",
            "oracle_inputs_used": False,
        },
    )
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "method": "fixed_sparse_observation_randomized_desparsification_blind_ensemble",
            "source_sparse_update_file": str(base_update_file.resolve()),
            "source_sparse_update_hash": base_update_file_hash,
            "source_client_attack_observation_file": str(observation_file.resolve()) if observation_file else None,
            "source_client_attack_observation_hash": observation_file_hash,
            "source_run_paths": [str(path.resolve()) for path in run_paths],
            "keep_ratio": float(keep_ratio),
            "ensemble_size": int(args.ensemble_size),
            "noise_seed": int(args.noise_seed),
            "seed_plan": list(seed_plan),
            "desparsification_scope": "attack_layer",
            "missing_value_sampler": "uniform",
            "missing_value_abs_bound_policy": "per_client_global_topk_min_observed_abs_tau",
            "alignment": "blind_medoid_align_then_mean",
            "source_kind": source_kind,
            "oracle_inputs_used": False,
        },
    )
    return output_file, final_batches, source_kind


def main() -> None:
    args = parse_args()
    require_global_topk(args.sparse_policy)
    for keep_ratio in args.keep_ratios:
        validate_keep_ratio(keep_ratio)
    if args.ensemble_size < 1:
        raise ValueError("--ensemble_size must be >= 1")
    if args.success_lpips_threshold <= 0:
        raise ValueError("--success_lpips_threshold must be positive")

    args.observed_zero_policy = RANDOMIZED_DESPARSE_POLICY
    args.sparsification_application = "client_then_aggregate"
    args.client_granularity = "one_sample_per_client"

    output_files = {float(keep_ratio): sparse_update_path(args, keep_ratio) for keep_ratio in args.keep_ratios}
    observation_files = {
        float(keep_ratio): observation_path(args, keep_ratio)
        for keep_ratio in args.keep_ratios
        if float(keep_ratio) != 1.0
    }
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
        client_attack_observation_files=observation_files,
        collect_log_every=args.collect_log_every,
    )

    seed_plan = make_member_seed_plan(args.ensemble_size, args.noise_seed)
    for keep_ratio in args.keep_ratios:
        ratio = float(keep_ratio)
        ratio_tag = format_keep_ratio(ratio)
        sparse_file = sparse_files[ratio]
        observation_file = observation_files.get(ratio)

        if not args.run_attack:
            print(f"deterministic sparse artifact keep={ratio:g}: {sparse_file}")
            if observation_file is not None:
                print(f"fixed attack-layer observation keep={ratio:g}: {observation_file}")
            continue

        base_data = read_pickle(sparse_file)
        observation_data = read_pickle(observation_file) if observation_file is not None else None
        if observation_data is not None:
            observation_data["_index_path"] = str(observation_file)
        sparse_file_hash = sha256_file(sparse_file)
        observation_file_hash = sha256_file(observation_file) if observation_file is not None else None
        run_paths = []
        for seed_row in seed_plan:
            member_idx = int(seed_row["member"])
            member_seed = int(seed_row["desparsification_seed"])
            print(
                f"[02-member] keep={ratio:g} member {member_idx + 1}/{args.ensemble_size} "
                f"desparse_seed={member_seed} start",
                flush=True,
            )
            member_data, batch_stats = materialize_member_update_data(
                base_data=base_data,
                base_update_file=sparse_file,
                base_update_file_hash=sparse_file_hash,
                observation_file=observation_file,
                observation_file_hash=observation_file_hash,
                observation_data=observation_data,
                keep_ratio=ratio,
                member_idx=member_idx,
                member_seed=member_seed,
            )
            exp_name = (
                f"{EXPERIMENT_NAME}_keep{ratio_tag}_{run_tag(args, include_fia=False)}"
                f"_R{args.ensemble_size}_seed{args.noise_seed}_m{member_idx:03d}"
            )
            run_path = run_author_cpa_with_update_file(
                args,
                exp_name=exp_name,
                update_file=sparse_file,
                update_data=member_data,
                run_fia=False,
                method=(
                    "Fixed sparse observation randomized desparsification member CPA: "
                    "observed attack-layer entries are exact, dropped entries are sampled "
                    "from U(-tau_k, tau_k) before uniform aggregation."
                ),
                extra_manifest={
                    "update_file_role": "baseline sparse context artifact; attack layer overridden in memory",
                    "randomized_desparsification_policy": RANDOMIZED_DESPARSE_POLICY,
                    "desparsification_scope": "attack_layer",
                    "missing_value_sampler": "uniform",
                    "missing_value_abs_bound_policy": "per_client_global_topk_min_observed_abs_tau",
                    "ensemble_member": member_idx,
                    "ensemble_size": int(args.ensemble_size),
                    "ensemble_noise_seed": int(args.noise_seed),
                    "ensemble_desparsification_seed": member_seed,
                    "desparsification_batch_stats": batch_stats,
                },
                attack_order_policy=(
                    "member CPA order; no oracle permutation; source will be aligned only by blind ensemble"
                ),
            )
            run_paths.append(run_path)
            print(
                f"[02-member] keep={ratio:g} member {member_idx + 1}/{args.ensemble_size} done: {run_path}",
                flush=True,
            )

        ensemble_file, aggregated_sources, source_kind = aggregate_member_sources(
            args=args,
            keep_ratio=ratio,
            base_update_file=sparse_file,
            base_update_file_hash=sparse_file_hash,
            observation_file=observation_file,
            observation_file_hash=observation_file_hash,
            run_paths=run_paths,
            seed_plan=seed_plan,
        )
        print(f"blind ensemble sources keep={ratio:g}: {ensemble_file}")

        if args.run_aggregate_fia:
            if source_kind != "rec_emb_attack_order":
                raise ValueError(
                    "--run_aggregate_fia requires a conv model with recovered embedding sources; "
                    f"got source_kind={source_kind!r}"
                )
            run_direct_fia_from_embeddings(
                args,
                exp_name=(
                    f"{EXPERIMENT_NAME}_blindagg_keep{ratio_tag}_{run_tag(args)}"
                    f"_R{args.ensemble_size}_seed{args.noise_seed}"
                ),
                update_file=sparse_file,
                recovered_embeddings=aggregated_sources,
                method="Direct FIA from blind fixed-sparse randomized-desparsification ensemble",
                extra_manifest={
                    "aggregate_fia_input": "blind_ensemble_recovered_embeddings",
                    "gradient_role": "runner compatibility/context only; Direct FIA optimizes against recovered embeddings",
                    "source_blind_ensemble_file": str(ensemble_file.resolve()),
                    "randomized_desparsification_policy": RANDOMIZED_DESPARSE_POLICY,
                    "desparsification_scope": "attack_layer",
                    "missing_value_sampler": "uniform",
                    "missing_value_abs_bound_policy": "per_client_global_topk_min_observed_abs_tau",
                    "ensemble_size": int(args.ensemble_size),
                    "ensemble_noise_seed": int(args.noise_seed),
                },
            )


if __name__ == "__main__":
    main()
