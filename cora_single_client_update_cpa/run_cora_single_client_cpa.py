#!/usr/bin/env python3
"""CoRA for a fixed single-client compressed update observation.

The observed signal is one client's compressed local update:

    Y_k = S_r(Delta theta_k)

Each randomized member keeps observed attack-FC coordinates exact, fills only
dropped attack-FC coordinates, runs CPA + Direct FIA, then optional consistency
ranking recomputes one client update from the recovered batch and compares it to
the same compressed observation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SPARSE_SRC = REPO_ROOT / "sparsified_update_cpa"
if str(SPARSE_SRC) not in sys.path:
    sys.path.insert(0, str(SPARSE_SRC))

from author_runner import run_author_cpa_with_update_file  # noqa: E402
from cora_ica import require_negentropy_run  # noqa: E402
from common import (  # noqa: E402
    AUTHOR_SOURCE_HASH,
    AUTHOR_RUN_ROOT,
    _load_indexed_test_batches,
    _named_update_dict,
    add_attack_args,
    add_collect_args,
    build_experiment_model,
    checkpoint_manifest,
    get_device,
    get_model,
    hash_json,
    hash_tensor_sequence,
    make_attack_namespace,
    parameter_schema,
    read_pickle,
    resolve_local_batch_size,
    resolve_micro_batch_size,
    run_tag,
    sha256_bytes,
    sha256_file,
    write_json,
    write_pickle,
)
from metrics import write_csv  # noqa: E402
from sparsification import (  # noqa: E402
    OBSERVED_ZERO_POLICY,
    RANDOMIZED_DESPARSE_POLICY,
    format_keep_ratio,
    identity_global_topk_stats,
    parse_keep_ratio_or_percent,
    require_global_topk,
    sparsify_update_global_topk,
    validate_keep_ratio,
)
import utils as author_utils  # noqa: E402


EXPERIMENT_NAME = "cora_single_client_cpa"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs_v1"
SINGLE_UPDATE_ROOT = OUTPUT_ROOT / "single_client_updates"
DEFAULT_KEEP_RATIOS = [0.8]
IMAGE_MEAN_STD = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "tiny_imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_collect_args(parser)
    parser.set_defaults(n_samples=24, attack_trials=1, local_epochs=1)
    parser.add_argument("--keep_ratios", type=parse_keep_ratio_or_percent, nargs="+", default=DEFAULT_KEEP_RATIOS)
    parser.add_argument("--sparse_policy", type=str, default="global_topk", choices=["global_topk"])
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--noise_seed", type=int, default=1234)
    parser.add_argument(
        "--desparse_sampler",
        type=str,
        default="retained_rescaled",
        choices=["retained_rescaled", "uniform_tau"],
    )
    parser.add_argument(
        "--tau_policy",
        type=str,
        default="topk_threshold",
        choices=["topk_threshold", "min_abs_nonzero_observed"],
        help="Bound for dropped coordinates. topk_threshold matches the paper-style Top-k threshold.",
    )
    parser.add_argument("--success_lpips_threshold", type=float, default=0.50)
    parser.add_argument(
        "--consistency_label_mode",
        type=str,
        default="model_pred",
        choices=["model_pred", "zero"],
    )
    parser.add_argument("--skip_consistency_ranking", action="store_true")
    add_attack_args(parser)
    return parser.parse_args()


def local_steps(args: argparse.Namespace) -> int:
    local_batch_size = resolve_local_batch_size(args)
    return int(args.local_epochs) * int(math.ceil(int(args.n_samples) / local_batch_size))


def single_client_config(args: argparse.Namespace) -> dict[str, Any]:
    local_batch_size = resolve_local_batch_size(args)
    micro_batch_size = resolve_micro_batch_size(args)
    return {
        "schema_version": 1,
        "signal": "single_client_compressed_update",
        "update_definition": "Y_k = S_r(theta_k_local - theta_global)",
        "ds": args.ds,
        "model": args.model,
        "h_dim": args.h_dim,
        "n_samples": int(args.n_samples),
        "attack_trials": int(args.attack_trials),
        "local_epochs": int(args.local_epochs),
        "local_steps": local_steps(args),
        "local_batch_size": int(local_batch_size),
        "micro_batch_size": int(micro_batch_size),
        "clients_per_trial": 1,
        "client_granularity": f"one_client_batch_{int(args.n_samples)}",
        "aggregation_weights": "none_single_client",
        "optimizer": "SGD",
        "lr": float(args.local_lr),
        "momentum": 0.0,
        "weight_decay": 0.0,
        "seed": int(args.seed),
        "checkpoint": checkpoint_manifest(args.global_checkpoint),
        "author_source_hash": AUTHOR_SOURCE_HASH,
    }


def group_tag(args: argparse.Namespace) -> str:
    config = single_client_config(args)
    return (
        f"single_client_n{args.n_samples}_steps{config['local_steps']}"
        f"_trials{args.attack_trials}_{hash_json(config)[:16]}"
    )


def source_update_path(args: argparse.Namespace, keep_ratio: float) -> Path:
    ratio_tag = format_keep_ratio(keep_ratio)
    return (
        SINGLE_UPDATE_ROOT
        / args.ds
        / args.model
        / group_tag(args)
        / args.sparse_policy
        / f"keep_{ratio_tag}.pickle"
    )


def observation_path(args: argparse.Namespace, keep_ratio: float) -> Path:
    ratio_tag = format_keep_ratio(keep_ratio)
    return (
        SINGLE_UPDATE_ROOT
        / args.ds
        / args.model
        / group_tag(args)
        / args.sparse_policy
        / f"keep_{ratio_tag}_attack_layer_observation.pickle"
    )


def cora_output_dir(args: argparse.Namespace, keep_ratio: float) -> Path:
    ratio_tag = format_keep_ratio(keep_ratio)
    return (
        OUTPUT_ROOT
        / args.ds
        / args.model
        / group_tag(args)
        / args.sparse_policy
        / f"keep_{ratio_tag}"
        / f"R{args.ensemble_size}_seed{args.noise_seed}_{args.consistency_label_mode}"
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


def author_member_run_path(args: argparse.Namespace, exp_name: str) -> Path:
    author_utils.exp_path_base = str(AUTHOR_RUN_ROOT)
    attack_args = make_attack_namespace(args, exp_name=exp_name)
    return Path(author_utils.get_attack_exp_path(attack_args))


def clone_model_from_global(global_model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    local_model = get_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        load_path=None,
        dataparallel=False,
    )
    state = {name: value.detach().cpu() for name, value in global_model.state_dict().items()}
    local_model.load_state_dict(state, strict=True)
    return local_model.to(device=device, dtype=torch.float32)


def train_single_client_update(
    *,
    global_model: torch.nn.Module,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    log_prefix: str,
) -> OrderedDict[str, torch.Tensor]:
    print(f"{log_prefix} local model init", flush=True)
    local_model = clone_model_from_global(global_model, args, device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=float(args.local_lr), momentum=0.0, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    local_batch_size = resolve_local_batch_size(args)
    micro_batch_size = resolve_micro_batch_size(args)
    loader = DataLoader(TensorDataset(x_cpu, y_cpu), batch_size=local_batch_size, shuffle=False, num_workers=0)
    print(
        f"{log_prefix} local train start steps={local_steps(args)} "
        f"batch={local_batch_size} micro={micro_batch_size}",
        flush=True,
    )
    for _epoch in range(int(args.local_epochs)):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device=device, dtype=torch.float32)
            y_batch = y_batch.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            logical_n = int(x_batch.shape[0])
            for start in range(0, logical_n, micro_batch_size):
                end = min(start + micro_batch_size, logical_n)
                loss = criterion(local_model(x_batch[start:end]), y_batch[start:end])
                (loss * ((end - start) / float(logical_n))).backward()
            optimizer.step()
    update = OrderedDict(
        (name, value.detach().cpu().float().contiguous())
        for name, value in _named_update_dict(local_model, global_model).items()
    )
    del optimizer, local_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return update


def pack_mask(mask: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(mask, dtype=np.bool_).reshape(-1)
    return {
        "shape": list(mask.shape),
        "packed_bits": np.packbits(flat.astype(np.uint8)),
    }


def unpack_mask(mask_shape: Sequence[int], packed_bits: np.ndarray) -> np.ndarray:
    total = int(np.prod(mask_shape, dtype=np.int64))
    bits = np.unpackbits(np.asarray(packed_bits, dtype=np.uint8), count=total)
    return bits.astype(np.bool_).reshape(tuple(int(v) for v in mask_shape))


def write_observation_shard(
    *,
    index_path: Path,
    batch_idx: int,
    keep_ratio: float,
    mask: np.ndarray,
    values: np.ndarray,
    tau: float,
    tau_policy: str,
    global_total: int,
    global_kept: int,
    selection_mode: str,
) -> dict[str, Any]:
    shard_dir = index_path.with_suffix("").parent / f"{index_path.with_suffix('').name}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_name = f"batch_{batch_idx:03d}_client_0000.npz"
    shard_path = shard_dir / shard_name
    packed = pack_mask(mask)
    values = np.asarray(values, dtype=np.float32)
    np.savez(
        shard_path,
        mask_shape=np.asarray(packed["shape"], dtype=np.int64),
        mask_packed_bits=np.asarray(packed["packed_bits"], dtype=np.uint8),
        values=values,
    )
    digest = hashlib.sha256()
    digest.update(str(tuple(packed["shape"])).encode("ascii"))
    digest.update(np.asarray(packed["packed_bits"], dtype=np.uint8).tobytes())
    digest.update(str(tuple(values.shape)).encode("ascii"))
    digest.update(values.tobytes())
    return {
        "batch": int(batch_idx),
        "client": 0,
        "aggregation_weight": 1.0,
        "tau": float(tau),
        "tau_policy": tau_policy,
        "shard_file": str(shard_path.relative_to(index_path.parent)),
        "observation_hash": digest.hexdigest(),
        "observed_elements": int(mask.sum()),
        "total_elements": int(mask.size),
        "global_total_elements": int(global_total),
        "global_kept_elements": int(global_kept),
        "selection_mode": selection_mode,
        "storage": "sharded_npz",
        "keep_ratio": float(keep_ratio),
    }


def load_observation_row(row: Mapping[str, Any], *, observation_file: Path) -> tuple[np.ndarray, np.ndarray]:
    shard_path = observation_file.parent / str(row["shard_file"])
    with np.load(shard_path) as data:
        mask = unpack_mask(data["mask_shape"], data["mask_packed_bits"])
        values = np.asarray(data["values"], dtype=np.float32)
    return mask, values


def tau_from_policy(stats: Mapping[str, Any], attack_values: np.ndarray, policy: str) -> float:
    if policy == "topk_threshold":
        value = stats.get("topk_threshold_abs")
        return float(value) if value is not None else 0.0
    if policy == "min_abs_nonzero_observed":
        nonzero_abs = np.abs(np.asarray(attack_values, dtype=np.float32))
        nonzero_abs = nonzero_abs[nonzero_abs > 0.0]
        return float(nonzero_abs.min()) if nonzero_abs.size else 0.0
    raise ValueError(f"unknown tau policy: {policy}")


def build_empty_data(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    keep_ratio: float,
    names: Sequence[str],
    schema: Sequence[Mapping[str, Any]],
    attack_layer_index: int,
    sample_indices: list[list[int]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "x": [],
        "y": [],
        "z": [],
        "grad": [],
        "param_names": list(names),
        "parameter_schema": list(schema),
        "attack_layer": {"index": int(attack_layer_index), "name": names[attack_layer_index]},
        "sample_indices": sample_indices,
        "local_batch_orders": [],
        "quant_codes": None,
        "quant_stats": [],
        "sparse_masks": None,
        "sparse_stats": [],
        "whitening_pre_stats": [],
        "metadata": {
            **dict(config),
            "quant_bits": 32,
            "rounding": "nearest",
            "fake_quantization": False,
            "dequantized_before_attack": False,
            "sparsified_observation": True,
            "sparsification_policy": args.sparse_policy,
            "sparsification_application": "single_client_update",
            "keep_ratio": float(keep_ratio),
            "control_role": "single_client_fp32_control" if float(keep_ratio) >= 1.0 else None,
            "observed_zero_policy": OBSERVED_ZERO_POLICY,
            "trial_semantics": "independent_single_client_batches_from_same_w0",
            "attack_fc_observation_model": "Y_k_attack_fc ~= A Z for one compressed client update",
            "az_scope": "attack_fc_layer_only_approximate",
            "attacker_observation": "one_client_compressed_update",
            "attacker_knows_removed_values": False,
            "client_masks_available_to_attacker": True,
            "client_thresholds_available_to_attacker": True,
            "compression_scheme_known": True,
            "tau_policy": args.tau_policy,
        },
    }


def collect_single_client_sparse_pickles(args: argparse.Namespace) -> dict[float, Path]:
    require_global_topk(args.sparse_policy)
    ratios = sorted({float(r) for r in args.keep_ratios}, reverse=True)
    for ratio in ratios:
        validate_keep_ratio(ratio)
    source_files = {ratio: source_update_path(args, ratio) for ratio in ratios}
    obs_files = {ratio: observation_path(args, ratio) for ratio in ratios if ratio < 1.0}
    if (
        args.reuse_existing
        and all(path.exists() for path in source_files.values())
        and all(path.exists() for path in obs_files.values())
    ):
        for ratio, path in source_files.items():
            print(f"[single-collect] reuse keep={ratio:g}: {path}", flush=True)
        return source_files

    torch.manual_seed(int(args.seed))
    device = get_device()
    config = single_client_config(args)
    global_model = build_experiment_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        device=device,
        checkpoint_path=args.global_checkpoint,
        seed=args.seed,
    )
    global_model.eval()
    xs, ys, sample_indices = _load_indexed_test_batches(
        ds=args.ds,
        n_samples=int(args.n_samples),
        attack_trials=int(args.attack_trials),
        seed=int(args.seed),
    )
    named_params = list(global_model.named_parameters())
    names = [name for name, _param in named_params]
    schema = parameter_schema(global_model)
    attack_layer_index = int(getattr(global_model, "attack_index", 0))
    if attack_layer_index >= len(names):
        raise ValueError("Model attack_index is outside named parameter mapping")

    data_by_ratio = {
        ratio: build_empty_data(
            args=args,
            config=config,
            keep_ratio=ratio,
            names=names,
            schema=schema,
            attack_layer_index=attack_layer_index,
            sample_indices=sample_indices,
        )
        for ratio in ratios
    }
    obs_by_ratio: dict[float, dict[str, Any]] = {
        ratio: {
            "schema_version": 1,
            "metadata": {
                **dict(config),
                "method": "single_client_attack_layer_fixed_sparse_observation_collector",
                "fixed_sparse_observation": True,
                "sparsification_policy": args.sparse_policy,
                "sparsification_application": "single_client_update",
                "keep_ratio": float(ratio),
                "desparsification_scope": "attack_layer",
                "tau_policy": args.tau_policy,
                "observed_zero_policy": OBSERVED_ZERO_POLICY,
            },
            "attack_layer": {
                "index": int(attack_layer_index),
                "name": names[attack_layer_index],
                "shape": list(named_params[attack_layer_index][1].shape),
                "dtype": "torch.float32",
            },
            "param_names": names,
            "sample_indices": sample_indices,
            "observations": [],
            "storage": "sharded_npz",
        }
        for ratio in obs_files
    }

    print(
        "[single-collect] "
        f"ds={args.ds} model={args.model} trials={args.attack_trials} "
        f"client_batch={args.n_samples} local_steps={config['local_steps']} "
        f"keep_ratios={' '.join(format(r, 'g') for r in ratios)}",
        flush=True,
    )

    for batch_idx, (x_cpu, y_cpu) in enumerate(zip(xs, ys)):
        print(f"[single-collect] batch {batch_idx + 1}/{len(xs)} start", flush=True)
        for ratio in ratios:
            data_by_ratio[ratio]["x"].append(x_cpu.numpy())
            data_by_ratio[ratio]["y"].append(y_cpu.numpy())
        with torch.no_grad():
            z_value = None
            if getattr(global_model, "model_type", None) == "conv":
                chunks = []
                feature_chunk = max(1, min(resolve_micro_batch_size(args), len(x_cpu)))
                for start in range(0, len(x_cpu), feature_chunk):
                    _, z_chunk = global_model(x_cpu[start : start + feature_chunk].to(device), return_z=True)
                    chunks.append(z_chunk.detach().cpu())
                z_value = torch.cat(chunks, dim=0).numpy()
        if z_value is not None:
            for ratio in ratios:
                data_by_ratio[ratio]["z"].append(z_value)

        update = train_single_client_update(
            global_model=global_model,
            x_cpu=x_cpu,
            y_cpu=y_cpu,
            args=args,
            device=device,
            log_prefix=f"[single-collect] batch {batch_idx + 1}/{len(xs)}",
        )
        source_fp32_hash = hash_tensor_sequence(list(update.values()))
        for ratio in ratios:
            print(f"[single-collect] batch {batch_idx + 1}/{len(xs)} keep={ratio:g} sparsify", flush=True)
            if ratio >= 1.0:
                sparse_update = OrderedDict((name, value.clone()) for name, value in update.items())
                sparse_stats = identity_global_topk_stats(update)
                raw_masks = [torch.ones_like(value, dtype=torch.bool) for value in update.values()]
            else:
                sparse_update, _packed, sparse_stats, raw_masks = sparsify_update_global_topk(
                    update,
                    keep_ratio=float(ratio),
                    pack_masks=False,
                    return_masks=True,
                )
            data = data_by_ratio[ratio]
            data["grad"].append([value.detach().cpu().numpy().astype(np.float32, copy=False) for value in sparse_update.values()])
            data["quant_stats"].append({"quant_bits": 32, "rounding": "none"})
            data["sparse_stats"].append(sparse_stats)
            data["metadata"].update(
                {
                    "source_fp32_update_hash": source_fp32_hash,
                    "sparse_nonzero_fraction": sparse_stats.get("sparse_nonzero_fraction"),
                    "sparse_relative_l2": sparse_stats.get("sparse_relative_l2"),
                    "sparse_cosine_similarity_to_fp32": sparse_stats.get("sparse_cosine_similarity_to_fp32"),
                    "sparse_retained_l2_norm_ratio": sparse_stats.get("sparse_retained_l2_norm_ratio"),
                }
            )
            if ratio in obs_by_ratio:
                mask = raw_masks[attack_layer_index].detach().cpu().numpy().astype(np.bool_, copy=False)
                attack_tensor = sparse_update[names[attack_layer_index]].detach().cpu().numpy().astype(np.float32, copy=False)
                values = attack_tensor[mask].astype(np.float32, copy=True)
                tau = tau_from_policy(sparse_stats, values, args.tau_policy)
                row = write_observation_shard(
                    index_path=obs_files[ratio],
                    batch_idx=batch_idx,
                    keep_ratio=ratio,
                    mask=mask,
                    values=values,
                    tau=tau,
                    tau_policy=args.tau_policy,
                    global_total=int(sparse_stats.get("total_elements", 0)),
                    global_kept=int(sparse_stats.get("mask_kept_elements", 0)),
                    selection_mode=str(next((r.get("selection_mode") for r in []), "")),
                )
                obs_by_ratio[ratio]["observations"].append([row])
                print(
                    f"[single-collect] batch {batch_idx + 1}/{len(xs)} keep={ratio:g} "
                    f"attack_obs={row['observed_elements']}/{row['total_elements']} tau={tau:.6g}",
                    flush=True,
                )
        del update
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for ratio, data in data_by_ratio.items():
        data["metadata"]["sample_indices_hash"] = sha256_bytes(json.dumps(sample_indices).encode("utf-8"))
        data["metadata"]["label_hash"] = hash_tensor_sequence(data["y"])
        data["metadata"]["update_tensor_hash"] = hash_tensor_sequence(data["grad"])
        output_file = source_files[ratio]
        write_pickle(output_file, data)
        write_json(
            output_file.with_suffix(".manifest.json"),
            {
                **single_client_config(args),
                "method": "single_client_sparse_update_collector",
                "sparse_policy": args.sparse_policy,
                "keep_ratio": float(ratio),
                "tau_policy": args.tau_policy,
                "observed_zero_policy": OBSERVED_ZERO_POLICY,
            },
        )
        print(f"[single-collect] wrote keep={ratio:g}: {output_file}", flush=True)

    for ratio, observation in obs_by_ratio.items():
        observation_file = obs_files[ratio]
        observation["metadata"]["source_sparse_update_file"] = str(source_files[ratio].resolve())
        observation["metadata"]["source_sparse_update_hash"] = sha256_file(source_files[ratio])
        observation["metadata"]["sample_indices_hash"] = sha256_bytes(json.dumps(sample_indices).encode("utf-8"))
        observation["metadata"]["client_attack_observation_hash"] = hash_observation_rows(observation["observations"])
        write_pickle(observation_file, observation)
        write_json(observation_file.with_suffix(".manifest.json"), observation["metadata"])
        print(f"[single-collect] wrote observation keep={ratio:g}: {observation_file}", flush=True)

    return source_files


def hash_observation_rows(observations: Sequence[Sequence[Mapping[str, Any]]]) -> str:
    digest = hashlib.sha256()
    for rows in observations:
        digest.update(f"batch:{len(rows)}".encode("ascii"))
        digest.update(b"\0")
        for row in rows:
            for key in ("batch", "client", "tau", "observed_elements", "total_elements", "observation_hash"):
                digest.update(str(row.get(key)).encode("utf-8"))
                digest.update(b"\0")
    return digest.hexdigest()


def _hash_arrays(values: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.asarray(value)
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def sample_missing_values(
    *,
    shape: Sequence[int],
    mask: np.ndarray,
    values: np.ndarray,
    tau: float,
    rng: np.random.Generator,
    sampler: str,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.bool_).reshape(tuple(int(v) for v in shape))
    values = np.asarray(values, dtype=np.float32)
    if bool(mask.all()):
        out = np.zeros(mask.shape, dtype=np.float32)
        out[mask] = values
        return out
    out = np.zeros(mask.shape, dtype=np.float32)
    dropped = ~mask
    dropped_count = int(np.count_nonzero(dropped))
    if tau <= 0.0:
        pass
    elif sampler == "uniform_tau":
        out[dropped] = rng.uniform(-tau, tau, size=dropped_count).astype(np.float32)
    elif sampler == "retained_rescaled":
        if values.size == 0:
            out[dropped] = rng.uniform(-tau, tau, size=dropped_count).astype(np.float32)
        else:
            observed_abs = np.abs(values.astype(np.float32, copy=False))
            abs_min = float(observed_abs.min())
            abs_max = float(observed_abs.max())
            if abs_max > abs_min:
                retained_unit = ((observed_abs - abs_min) / (abs_max - abs_min)).astype(np.float32, copy=False)
                retained_mean = float(np.mean(retained_unit))
                retained_var = float(np.var(retained_unit))
                beta_scale = retained_mean * (1.0 - retained_mean) / max(retained_var, 1e-12) - 1.0
                if beta_scale > 0.0 and np.isfinite(beta_scale):
                    alpha = max(retained_mean * beta_scale, 1e-3)
                    beta = max((1.0 - retained_mean) * beta_scale, 1e-3)
                    magnitudes = rng.beta(alpha, beta, size=dropped_count).astype(np.float32)
                    magnitudes *= tau
                else:
                    magnitudes = rng.uniform(0.0, tau, size=dropped_count).astype(np.float32)
            else:
                magnitudes = rng.uniform(0.0, tau, size=dropped_count).astype(np.float32)
            positive_fraction = float(np.mean(values > 0.0)) if values.size else 0.5
            signs = rng.random(size=dropped_count) < positive_fraction
            magnitudes[~signs] *= -1.0
            out[dropped] = magnitudes.astype(np.float32, copy=False)
    else:
        raise ValueError(f"unknown sampler: {sampler}")
    out[mask] = values
    return out.astype(np.float32, copy=False)


def materialize_member_update_data(
    *,
    base_data: Mapping[str, Any],
    base_update_file: Path,
    base_update_file_hash: str,
    observation_file: Path | None,
    observation_data: Mapping[str, Any] | None,
    keep_ratio: float,
    member_idx: int,
    member_seed: int,
    sampler: str,
) -> dict[str, Any]:
    attack_idx = int(base_data["attack_layer"]["index"])
    member_grad = []
    attack_layers = []
    member_stats = []
    rng = np.random.default_rng(int(member_seed))
    for batch_idx, batch_grad in enumerate(base_data["grad"]):
        batch_values = list(batch_grad)
        if observation_data is None or keep_ratio >= 1.0:
            attack_layer = np.asarray(batch_values[attack_idx], dtype=np.float32)
            stats = {"batch": int(batch_idx), "sampler": "deterministic_identity", "tau": 0.0}
        else:
            row = observation_data["observations"][batch_idx][0]
            mask, values = load_observation_row(row, observation_file=observation_file)
            attack_layer = sample_missing_values(
                shape=observation_data["attack_layer"]["shape"],
                mask=mask,
                values=values,
                tau=float(row["tau"]),
                rng=rng,
                sampler=sampler,
            )
            observed_fraction = float(row["observed_elements"]) / max(1.0, float(row["total_elements"]))
            stats = {
                "batch": int(batch_idx),
                "sampler": sampler,
                "tau": float(row["tau"]),
                "observed_fraction": observed_fraction,
                "observation_hash": row.get("observation_hash"),
            }
            print(
                f"[single-member] keep={keep_ratio:g} member {member_idx + 1} "
                f"batch {batch_idx + 1}/{len(base_data['grad'])} "
                f"observed={observed_fraction:.4f} tau={float(row['tau']):.6g}",
                flush=True,
            )
        batch_values[attack_idx] = attack_layer
        member_grad.append(batch_values)
        attack_layers.append(np.asarray(attack_layer, dtype=np.float32))
        member_stats.append(stats)
    attack_hash = _hash_arrays(attack_layers)
    metadata = copy.deepcopy(base_data["metadata"])
    metadata.update(
        {
            "cora_variant": "single_client_compressed_update",
            "cora_stage": "independent_member_restart",
            "cora_wrapper": True,
            "cora_candidate_generation": "single_client_fixed_sparse_observation_randomized_desparsification",
            "fixed_sparse_observation": True,
            "randomized_desparsification": observation_data is not None and keep_ratio < 1.0,
            "randomized_desparsification_policy": RANDOMIZED_DESPARSE_POLICY,
            "desparsification_scope": "attack_layer",
            "missing_value_sampler": sampler if keep_ratio < 1.0 else "deterministic_identity",
            "ensemble_member": int(member_idx),
            "ensemble_size": None,
            "ensemble_desparsification_seed": int(member_seed),
            "source_sparse_update_file": str(base_update_file.resolve()),
            "source_sparse_update_hash": base_update_file_hash,
            "source_attack_layer_observation_file": str(observation_file.resolve()) if observation_file else None,
            "member_desparsification_stats": member_stats,
            "randomized_attack_layer_hash": attack_hash,
            "update_tensor_hash": sha256_bytes(
                json.dumps(
                    {
                        "base_update_tensor_hash": base_data["metadata"].get("update_tensor_hash"),
                        "attack_layer_hash": attack_hash,
                        "member": member_idx,
                        "seed": member_seed,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ),
        }
    )
    member_data = dict(base_data)
    member_data["metadata"] = metadata
    member_data["grad"] = member_grad
    return member_data


def normalize_unit_image(x_unit: torch.Tensor, ds: str) -> torch.Tensor:
    if ds not in IMAGE_MEAN_STD:
        raise ValueError(f"Unsupported dataset for ranking normalization: {ds}")
    mean_values, std_values = IMAGE_MEAN_STD[ds]
    mean = torch.tensor(mean_values, device=x_unit.device, dtype=x_unit.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std_values, device=x_unit.device, dtype=x_unit.dtype).view(1, 3, 1, 1)
    return (x_unit.clamp(0.0, 1.0) - mean) / std


def choose_labels(mode: str, model: torch.nn.Module, x_norm: torch.Tensor, batch_size: int) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros(int(x_norm.shape[0]), dtype=torch.long, device=x_norm.device)
    if mode == "model_pred":
        preds = []
        with torch.no_grad():
            for start in range(0, int(x_norm.shape[0]), batch_size):
                end = min(start + batch_size, int(x_norm.shape[0]))
                preds.append(torch.argmax(model(x_norm[start:end]), dim=1).detach())
        return torch.cat(preds, dim=0)
    raise ValueError(f"unknown label mode: {mode}")


def recompress_reconstruction_attack_layer(
    *,
    args: argparse.Namespace,
    global_model: torch.nn.Module,
    rec_fi: np.ndarray,
    keep_ratio: float,
    attack_layer_name: str,
) -> np.ndarray:
    device = next(global_model.parameters()).device
    rec = torch.as_tensor(rec_fi, device=device, dtype=torch.float32)
    x_norm = normalize_unit_image(rec, args.ds)
    y_pred = choose_labels(
        args.consistency_label_mode,
        global_model,
        x_norm,
        batch_size=max(1, min(resolve_micro_batch_size(args), int(x_norm.shape[0]))),
    )
    update = train_single_client_update(
        global_model=global_model,
        x_cpu=x_norm.detach().cpu(),
        y_cpu=y_pred.detach().cpu(),
        args=args,
        device=device,
        log_prefix="[single-rank]",
    )
    sparse_update, _packed, _stats = sparsify_update_global_topk(update, keep_ratio=float(keep_ratio), pack_masks=False)
    attack = sparse_update[attack_layer_name].detach().cpu().numpy().astype(np.float32, copy=False)
    del update, sparse_update
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return attack


def relative_l2_chunked(pred: np.ndarray, observed: np.ndarray, chunk_elems: int = 2_000_000) -> float:
    pred_flat = np.asarray(pred, dtype=np.float32).reshape(-1)
    obs_flat = np.asarray(observed, dtype=np.float32).reshape(-1)
    if pred_flat.shape != obs_flat.shape:
        raise ValueError(f"shape mismatch: {pred_flat.shape} vs {obs_flat.shape}")
    diff_sq = 0.0
    obs_sq = 0.0
    for start in range(0, int(pred_flat.size), int(chunk_elems)):
        end = min(start + int(chunk_elems), int(pred_flat.size))
        pred64 = pred_flat[start:end].astype(np.float64, copy=False)
        obs64 = obs_flat[start:end].astype(np.float64, copy=False)
        delta = pred64 - obs64
        diff_sq += float(np.dot(delta, delta))
        obs_sq += float(np.dot(obs64, obs64))
    return float(math.sqrt(max(diff_sq, 0.0)) / max(math.sqrt(max(obs_sq, 0.0)), 1e-12))


def rank_members_by_consistency(
    *,
    args: argparse.Namespace,
    keep_ratio: float,
    source_data: Mapping[str, Any],
    run_paths: Sequence[Path],
) -> Path:
    device = get_device()
    model = build_experiment_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        device=device,
        checkpoint_path=args.global_checkpoint,
        seed=args.seed,
    )
    model.eval()
    attack_idx = int(source_data["attack_layer"]["index"])
    attack_name = str(source_data["attack_layer"]["name"])
    observed_attack = [np.asarray(batch[attack_idx], dtype=np.float32) for batch in source_data["grad"]]
    n_batches = min(args.attack_n_batch or len(observed_attack), len(observed_attack))
    ranked = []
    for member_idx, run_path in enumerate(run_paths):
        recon_entries = read_pickle(run_path / "reconstructions.pkl")
        costs = []
        for batch_idx, entry in enumerate(recon_entries[:n_batches]):
            rec_fi = np.asarray(entry.get("rec_fi_attack_order"), dtype=np.float32)
            if rec_fi.size == 0:
                raise ValueError("single-client consistency ranking requires Direct FIA reconstructions")
            full_n = int(source_data["metadata"]["n_samples"])
            if int(rec_fi.shape[0]) < full_n:
                raise ValueError(
                    "Exact single-client ranking needs FIA reconstructions for the full observed client batch. "
                    f"Got {rec_fi.shape[0]}/{full_n}. Use --n_sample_fi {full_n}."
                )
            print(
                f"[single-rank] member {member_idx + 1}/{len(run_paths)} "
                f"batch {batch_idx + 1}/{n_batches} recompress n={full_n}",
                flush=True,
            )
            pred_attack = recompress_reconstruction_attack_layer(
                args=args,
                global_model=model,
                rec_fi=rec_fi[:full_n],
                keep_ratio=keep_ratio,
                attack_layer_name=attack_name,
            )
            costs.append(relative_l2_chunked(pred_attack, observed_attack[int(entry["batch"])]))
            del pred_attack
        mean_cost = float(np.mean(costs)) if costs else float("inf")
        ranked.append(
            {
                "member": int(member_idx),
                "score": -mean_cost,
                "mean_relative_l2_to_observed_client_update": mean_cost,
                "batches": int(len(costs)),
                "source_run_path": str(run_path.resolve()),
                "ranking_observation": "one_client_compressed_update",
            }
        )
    ranked.sort(key=lambda row: float(row["mean_relative_l2_to_observed_client_update"]))
    output_dir = cora_output_dir(args, keep_ratio)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "consistency_member_ranking.csv", ranked)
    best = ranked[0]
    write_pickle(output_dir / "selected_reconstructions.pkl", read_pickle(Path(best["source_run_path"]) / "reconstructions.pkl"))
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "method": "CoRA single-client compressed-update consistency ranking",
            "keep_ratio": float(keep_ratio),
            "ensemble_size": int(len(run_paths)),
            "selected_member": int(best["member"]),
            "selected_member_run_path": best["source_run_path"],
            "ranking_observation": "one_client_compressed_update",
            "exact_single_client_batch_ranking": True,
            "member_ranking_file": "consistency_member_ranking.csv",
        },
    )
    return output_dir


def main() -> None:
    args = parse_args()
    if args.fi_method != "direct":
        raise ValueError("Single-client CoRA supports Direct FIA only")
    if args.use_labels:
        raise ValueError("--use_labels is disallowed for the no-oracle attack setting")
    args.local_batch_size = args.local_batch_size or args.n_samples
    args.sparsification_application = "single_client_update"
    require_global_topk(args.sparse_policy)

    source_files = collect_single_client_sparse_pickles(args)
    seed_plan = make_member_seed_plan(int(args.ensemble_size), int(args.noise_seed))

    for ratio in sorted(source_files, reverse=True):
        source_file = source_files[ratio]
        source_data = read_pickle(source_file)
        source_hash = sha256_file(source_file)
        observation_file = observation_path(args, ratio) if ratio < 1.0 else None
        observation_data = read_pickle(observation_file) if observation_file and observation_file.exists() else None
        member_count = 1 if ratio >= 1.0 else int(args.ensemble_size)
        run_paths: list[Path] = []
        output_dir = cora_output_dir(args, ratio)
        output_dir.mkdir(parents=True, exist_ok=True)
        for member_idx in range(member_count):
            member_seed = int(seed_plan[member_idx]["desparsification_seed"]) if ratio < 1.0 else int(args.noise_seed)
            ratio_tag = format_keep_ratio(ratio)
            exp_name = (
                f"{EXPERIMENT_NAME}_keep{ratio_tag}_{run_tag(args, include_fia=False)}"
                f"_R{member_count}_seed{args.noise_seed}_m{member_idx:03d}"
            )
            existing_run = author_member_run_path(args, exp_name)
            if args.reuse_existing and (existing_run / "reconstructions.pkl").exists():
                require_negentropy_run(existing_run, args.ne)
                print(f"[single-member] reuse completed member {member_idx + 1}/{member_count}: {existing_run}", flush=True)
                run_paths.append(existing_run)
                continue
            member_data = materialize_member_update_data(
                base_data=source_data,
                base_update_file=source_file,
                base_update_file_hash=source_hash,
                observation_file=observation_file,
                observation_data=observation_data,
                keep_ratio=float(ratio),
                member_idx=member_idx,
                member_seed=member_seed,
                sampler=args.desparse_sampler,
            )
            member_data["metadata"]["ensemble_size"] = member_count
            if args.run_attack:
                run_path = run_author_cpa_with_update_file(
                    args,
                    enable_negentropy=True,
                    exp_name=exp_name,
                    update_file=source_file,
                    update_data=member_data,
                    method=(
                        "CoRA single-client compressed update: fixed sparse observation, "
                        "attack-FC randomized desparsification, author CPA, Direct FIA"
                    ),
                    extra_manifest={
                        "cora_variant": "single_client_compressed_update",
                        "ensemble_member": int(member_idx),
                        "ensemble_size": int(member_count),
                        "desparsification_seed": int(member_seed),
                        "desparse_sampler": args.desparse_sampler if ratio < 1.0 else "deterministic_identity",
                        "paper_threat_model_alignment": (
                            "server observes one client's compressed update and does not know removed values"
                        ),
                    },
                    attack_order_policy="single-client compressed update; no oracle permutation",
                )
                run_paths.append(run_path)
        write_json(
            output_dir / "observation_manifest.json",
            {
                "schema_version": 1,
                "source_file": str(source_file.resolve()),
                "source_hash": source_hash,
                "observation_file": str(observation_file.resolve()) if observation_file else None,
                "observation_hash": sha256_file(observation_file) if observation_file else None,
                "keep_ratio": float(ratio),
                "ensemble_size": int(member_count),
                "noise_seed": int(args.noise_seed),
                "desparse_sampler": args.desparse_sampler if ratio < 1.0 else "deterministic_identity",
                "tau_policy": args.tau_policy,
                "observation": "one_client_compressed_update",
            },
        )
        if args.run_attack and not args.skip_consistency_ranking:
            rank_dir = rank_members_by_consistency(
                args=args,
                keep_ratio=float(ratio),
                source_data=source_data,
                run_paths=run_paths,
            )
            print(f"[single-rank] keep={ratio:g} ranking saved: {rank_dir}", flush=True)
        elif args.run_attack:
            print("[single-rank] skipped", flush=True)
        else:
            print(f"[single-cora] artifact ready: {source_file}", flush=True)


if __name__ == "__main__":
    main()
