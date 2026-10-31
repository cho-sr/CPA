#!/usr/bin/env python3
"""CoRA for one compressed client update sampled from a FedAvg round.

The observed signal remains one client's compressed local update, but the
global model is produced by a real FedAvg sparse-communication trajectory:

    Y_k^t = S_r(theta_k^{t,local} - theta^t)

Each randomized member keeps observed attack-FC coordinates exact, fills only
dropped attack-FC coordinates, runs CPA + Direct FIA, then optional consistency
ranking recomputes that same client update at theta^t and compares it to the
same compressed observation.  The attacker does not observe the round aggregate.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SPARSE_SRC = REPO_ROOT / "sparsified_update_cpa"
if str(SPARSE_SRC) not in sys.path:
    sys.path.insert(0, str(SPARSE_SRC))

from author_runner import run_author_cpa_with_update_file  # noqa: E402
from cora_ica import require_negentropy_run  # noqa: E402
from common import (  # noqa: E402
    AUTHOR_SOURCE_HASH,
    AUTHOR_RUN_ROOT,
    _named_update_dict,
    add_attack_args,
    add_collect_args,
    build_experiment_model,
    checkpoint_manifest,
    get_dataloaders,
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


EXPERIMENT_NAME = "cora_fedavg_single_client_cpa"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs_v1"
SINGLE_UPDATE_ROOT = OUTPUT_ROOT / "fedavg_single_client_updates"
DEFAULT_KEEP_RATIOS = [0.8]
IMAGE_MEAN_STD = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "tiny_imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


def windows_long_path(path: Path) -> str:
    text = str(path if path.is_absolute() else path.resolve())
    if sys.platform != "win32" or text.startswith("\\\\?\\"):
        return text
    text = text.replace("/", "\\")
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text.lstrip("\\")
    return "\\\\?\\" + text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_collect_args(parser)
    parser.set_defaults(n_samples=24, attack_trials=1, local_epochs=1)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--observe_round", type=int, default=None)
    parser.add_argument("--samples_per_client", type=int, default=None)
    parser.add_argument("--target_client_id", type=int, default=0)
    parser.add_argument("--private_split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--keep_ratios", type=parse_keep_ratio_or_percent, nargs="+", default=DEFAULT_KEEP_RATIOS)
    parser.add_argument("--sparse_policy", type=str, default="global_topk", choices=["global_topk"])
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--noise_seed", type=int, default=1234)
    parser.add_argument(
        "--desparse_sampler",
        type=str,
        default="retained_rescaled",
        choices=["zero_fill", "retained_rescaled", "uniform_tau"],
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
        "--prepared_source_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing matched keep_<ratio>.pickle artifacts created from one shared FP32 update. "
            "When supplied, skip FedAvg trajectory collection."
        ),
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default=None,
        help="Optional ASCII tag used to keep prepared-control attack outputs separate from prior runs.",
    )
    parser.add_argument(
        "--consistency_label_mode",
        type=str,
        default="model_pred",
        choices=["model_pred", "zero"],
    )
    parser.add_argument("--skip_consistency_ranking", action="store_true")
    parser.add_argument(
        "--fia_screen_iter",
        type=int,
        default=0,
        help="Run every ensemble member to this FIA iteration, rank them, then continue only --fia_screen_top_k members.",
    )
    parser.add_argument(
        "--fia_screen_top_k",
        type=int,
        default=0,
        help="Number of screened FIA members to continue to --n_iter_fi. Requires --fia_screen_iter > 0.",
    )
    add_attack_args(parser)
    return parser.parse_args()


def fedavg_start_checkpoint(args: argparse.Namespace) -> Path | None:
    return getattr(args, "fedavg_start_checkpoint", args.global_checkpoint)


def resolve_samples_per_client(args: argparse.Namespace) -> int:
    return int(args.samples_per_client or args.n_samples)


def resolve_observe_round(args: argparse.Namespace) -> int:
    return int(args.observe_round or args.rounds)


def local_steps(args: argparse.Namespace) -> int:
    local_batch_size = resolve_local_batch_size(args)
    return int(args.local_epochs) * int(math.ceil(int(args.n_samples) / local_batch_size))


def single_client_config(args: argparse.Namespace) -> dict[str, Any]:
    local_batch_size = resolve_local_batch_size(args)
    micro_batch_size = resolve_micro_batch_size(args)
    samples_per_client = resolve_samples_per_client(args)
    observe_round = resolve_observe_round(args)
    return {
        "schema_version": 1,
        "signal": "fedavg_context_single_client_compressed_update",
        "update_definition": "Y_k^t = S_r(theta_k^{t,local} - theta^t)",
        "fedavg_context": True,
        "attacker_observes_round_aggregate": False,
        "ds": args.ds,
        "model": args.model,
        "h_dim": args.h_dim,
        "n_samples": int(samples_per_client),
        "attack_trials": 1,
        "rounds": int(args.rounds),
        "observe_round": int(observe_round),
        "num_clients": int(args.num_clients),
        "samples_per_client": int(samples_per_client),
        "target_client_id": int(args.target_client_id),
        "private_split": args.private_split,
        "local_epochs": int(args.local_epochs),
        "local_steps": local_steps(args),
        "local_batch_size": int(local_batch_size),
        "micro_batch_size": int(micro_batch_size),
        "clients_per_trial": int(args.num_clients),
        "client_granularity": f"{int(samples_per_client)}_samples_per_client",
        "aggregation_weights": "FedAvg trajectory only; observed value is one selected client update",
        "optimizer": "SGD",
        "lr": float(args.local_lr),
        "momentum": 0.0,
        "weight_decay": 0.0,
        "seed": int(args.seed),
        "initial_checkpoint": checkpoint_manifest(fedavg_start_checkpoint(args)),
        "author_source_hash": AUTHOR_SOURCE_HASH,
    }


def group_tag(args: argparse.Namespace) -> str:
    config = single_client_config(args)
    return (
        f"fedavg_single_client_clients{args.num_clients}_round{config['observe_round']}of{args.rounds}"
        f"_spc{config['samples_per_client']}_target{args.target_client_id}"
        f"_local{args.local_epochs}_{hash_json(config)[:16]}"
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
    if args.prepared_source_dir is not None:
        control_tag = args.experiment_tag or "shared_fp32"
        return (
            OUTPUT_ROOT
            / "shared_attacks"
            / control_tag
            / f"keep_{ratio_tag}"
            / f"R{args.ensemble_size}_seed{args.noise_seed}_{args.consistency_label_mode}"
        )
    output_dir = (
        OUTPUT_ROOT
        / args.ds
        / args.model
        / group_tag(args)
        / args.sparse_policy
        / f"keep_{ratio_tag}"
        / f"R{args.ensemble_size}_seed{args.noise_seed}_{args.consistency_label_mode}"
    )
    return output_dir / args.experiment_tag if args.experiment_tag else output_dir


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


def member_exp_name(args: argparse.Namespace, keep_ratio: float, member_count: int, member_idx: int) -> str:
    ratio_tag = format_keep_ratio(keep_ratio)
    if args.prepared_source_dir is not None:
        control_tag = args.experiment_tag or "shared_fp32"
        return f"cora_shared_{control_tag}_k{ratio_tag}_R{member_count}_s{args.noise_seed}_m{member_idx:03d}"
    experiment_name = EXPERIMENT_NAME if not args.experiment_tag else f"{EXPERIMENT_NAME}_{args.experiment_tag}"
    return (
        f"{experiment_name}_keep{ratio_tag}_{run_tag(args, include_fia=False)}"
        f"_R{member_count}_seed{args.noise_seed}_m{member_idx:03d}"
    )


def prepared_source_files(args: argparse.Namespace) -> dict[float, Path]:
    if args.prepared_source_dir is None:
        raise ValueError("prepared_source_files requires --prepared_source_dir")
    source_dir = args.prepared_source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Prepared source directory does not exist: {source_dir}")

    source_files: dict[float, Path] = {}
    for ratio in sorted({float(value) for value in args.keep_ratios}, reverse=True):
        validate_keep_ratio(ratio)
        source_file = source_dir / f"keep_{format_keep_ratio(ratio)}.pickle"
        if not source_file.exists():
            raise FileNotFoundError(f"Missing prepared source for keep={ratio:g}: {source_file}")
        source_data = read_pickle(source_file)
        metadata = source_data.get("metadata", {})
        if float(metadata.get("keep_ratio", -1.0)) != float(ratio):
            raise RuntimeError(
                f"Prepared source keep ratio mismatch for {source_file}: "
                f"metadata={metadata.get('keep_ratio')} requested={ratio}"
            )
        if not metadata.get("shared_fp32_control"):
            raise RuntimeError(f"Prepared source is missing shared_fp32_control metadata: {source_file}")
        if ratio < 1.0:
            observation_file = source_file.with_name(f"keep_{format_keep_ratio(ratio)}_attack_layer_observation.pickle")
            if not observation_file.exists():
                raise FileNotFoundError(f"Missing prepared observation for keep={ratio:g}: {observation_file}")
        source_files[ratio] = source_file
    return source_files


def source_observation_path(args: argparse.Namespace, keep_ratio: float, source_file: Path) -> Path | None:
    if keep_ratio >= 1.0:
        return None
    if args.prepared_source_dir is not None:
        return source_file.with_name(f"keep_{format_keep_ratio(keep_ratio)}_attack_layer_observation.pickle")
    return observation_path(args, keep_ratio)


def checkpoint_iteration(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(checkpoint, Mapping) or "iteration" not in checkpoint:
        return None
    return int(checkpoint["iteration"])


def member_has_completed_fia(run_path: Path, target_iter: int) -> bool:
    if not (run_path / "reconstructions.pkl").exists():
        return False
    if int(target_iter) <= 0:
        return True
    iteration = checkpoint_iteration(run_path / "batch_000_fi.pt")
    return iteration is not None and iteration >= int(target_iter)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_fedavg_clients(args: argparse.Namespace) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    dl_train, dl_test = get_dataloaders(
        args.ds,
        batch_size=1,
        augment=False,
        shuffle_train=False,
        shuffle_test=False,
        num_workers=int(args.num_workers),
    )
    dataset = dl_train.dataset if args.private_split == "train" else dl_test.dataset
    samples_per_client = resolve_samples_per_client(args)
    total = int(args.num_clients) * int(samples_per_client)
    if total > len(dataset):
        raise ValueError(f"Need {total} private samples, but split has only {len(dataset)}")
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    selected = torch.randperm(len(dataset), generator=generator).tolist()[:total]
    xs, ys, indices = [], [], []
    for start in range(0, total, samples_per_client):
        subset_indices = selected[start : start + samples_per_client]
        loader = DataLoader(
            Subset(dataset, subset_indices),
            batch_size=len(subset_indices),
            shuffle=False,
            num_workers=int(args.num_workers),
        )
        batch = next(iter(loader))
        x, y = batch[:2]
        idx = batch[2] if len(batch) > 2 else torch.tensor(subset_indices, dtype=torch.long)
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
        indices.append(idx.detach().cpu())
    return xs, ys, indices


def named_parameters_cpu(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, param.detach().cpu().clone().float())
        for name, param in model.named_parameters()
    )


def zero_update_like(reference: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, torch.zeros_like(value, dtype=torch.float32)) for name, value in reference.items())


def add_weighted_update_(acc: OrderedDict[str, torch.Tensor], update: Mapping[str, torch.Tensor], weight: float) -> None:
    for name in acc:
        acc[name].add_(update[name].float(), alpha=float(weight))


def apply_parameter_update(
    model: nn.Module,
    global_params: Mapping[str, torch.Tensor],
    update: Mapping[str, torch.Tensor],
) -> None:
    state = model.state_dict()
    for name, delta in update.items():
        state[name] = (global_params[name] + delta).to(device=state[name].device, dtype=state[name].dtype)
    model.load_state_dict(state, strict=True)


def sparse_update_and_masks(
    update: OrderedDict[str, torch.Tensor],
    keep_ratio: float,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any], list[torch.Tensor]]:
    if float(keep_ratio) >= 1.0:
        return (
            OrderedDict((name, value.clone()) for name, value in update.items()),
            identity_global_topk_stats(update),
            [torch.ones_like(value, dtype=torch.bool) for value in update.values()],
        )
    sparse_update, _packed, sparse_stats, raw_masks = sparsify_update_global_topk(
        update,
        keep_ratio=float(keep_ratio),
        pack_masks=False,
        return_masks=True,
    )
    return sparse_update, sparse_stats, raw_masks


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
        windows_long_path(shard_path),
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
    with np.load(windows_long_path(shard_path)) as data:
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
            "sparsification_application": "fedavg_client_update_at_observed_round",
            "keep_ratio": float(keep_ratio),
            "control_role": "fedavg_single_client_fp32_control" if float(keep_ratio) >= 1.0 else None,
            "observed_zero_policy": OBSERVED_ZERO_POLICY,
            "trial_semantics": "one_selected_client_update_from_sparse_fedavg_round",
            "attack_fc_observation_model": "Y_k^t_attack_fc ~= A_t Z for one compressed client update at FedAvg round t",
            "az_scope": "attack_fc_layer_only_approximate",
            "attacker_observation": "one_client_compressed_update_from_fedavg_round",
            "attacker_observes_round_aggregate": False,
            "attacker_knows_removed_values": False,
            "client_masks_available_to_attacker": True,
            "client_thresholds_available_to_attacker": True,
            "compression_scheme_known": True,
            "tau_policy": args.tau_policy,
        },
    }


def collect_fedavg_single_client_sparse_pickles(args: argparse.Namespace) -> dict[float, Path]:
    require_global_topk(args.sparse_policy)
    ratios = sorted({float(r) for r in args.keep_ratios}, reverse=True)
    for ratio in ratios:
        validate_keep_ratio(ratio)
    samples_per_client = resolve_samples_per_client(args)
    args.n_samples = int(samples_per_client)
    args.attack_trials = 1
    observe_round = resolve_observe_round(args)
    if observe_round < 1 or observe_round > int(args.rounds):
        raise ValueError("--observe_round must be in [1, --rounds]")
    if int(args.target_client_id) < 0 or int(args.target_client_id) >= int(args.num_clients):
        raise ValueError("--target_client_id must be in [0, --num_clients)")

    source_files = {ratio: source_update_path(args, ratio) for ratio in ratios}
    obs_files = {ratio: observation_path(args, ratio) for ratio in ratios if ratio < 1.0}
    if (
        args.reuse_existing
        and all(path.exists() for path in source_files.values())
        and all(path.exists() for path in obs_files.values())
    ):
        for ratio, path in source_files.items():
            print(f"[fedavg-single-collect] reuse keep={ratio:g}: {path}", flush=True)
        return source_files

    set_seed(int(args.seed))
    device = get_device()
    config = single_client_config(args)
    xs, ys, original_indices = load_fedavg_clients(args)
    target_id = int(args.target_client_id)
    target_sample_indices = [[int(value) for value in original_indices[target_id].tolist()]]
    client_sizes = [int(x.shape[0]) for x in xs]
    weights = [size / float(sum(client_sizes)) for size in client_sizes]

    print(
        "[fedavg-single-collect] "
        f"ds={args.ds} model={args.model} rounds={args.rounds} observe_round={observe_round} "
        f"clients={args.num_clients} samples_per_client={samples_per_client} "
        f"target_client={target_id} local_steps={config['local_steps']} "
        f"keep_ratios={' '.join(format(r, 'g') for r in ratios)}",
        flush=True,
    )

    for ratio in ratios:
        output_file = source_files[ratio]
        observation_file = obs_files.get(ratio)
        print(f"[fedavg-single-collect] keep={ratio:g} trajectory start", flush=True)
        global_model = build_experiment_model(
            model_name=args.model,
            ds=args.ds,
            h_dim=args.h_dim,
            device=device,
            checkpoint_path=fedavg_start_checkpoint(args),
            seed=args.seed,
        )
        global_model.eval()
        named_params = list(global_model.named_parameters())
        names = [name for name, _param in named_params]
        schema = parameter_schema(global_model)
        attack_layer_index = int(getattr(global_model, "attack_index", 0))
        if attack_layer_index >= len(names):
            raise ValueError("Model attack_index is outside named parameter mapping")

        data = build_empty_data(
            args=args,
            config=config,
            keep_ratio=ratio,
            names=names,
            schema=schema,
            attack_layer_index=attack_layer_index,
            sample_indices=target_sample_indices,
        )
        observation: dict[str, Any] | None = None
        if observation_file is not None:
            observation = {
                "schema_version": 1,
                "metadata": {
                    **dict(config),
                    "method": "fedavg_single_client_attack_layer_fixed_sparse_observation_collector",
                    "fixed_sparse_observation": True,
                    "sparsification_policy": args.sparse_policy,
                    "sparsification_application": "fedavg_client_update_at_observed_round",
                    "keep_ratio": float(ratio),
                    "desparsification_scope": "attack_layer",
                    "tau_policy": args.tau_policy,
                    "observed_zero_policy": OBSERVED_ZERO_POLICY,
                    "attacker_observation": "one_client_compressed_update_from_fedavg_round",
                    "attacker_observes_round_aggregate": False,
                },
                "attack_layer": {
                    "index": int(attack_layer_index),
                    "name": names[attack_layer_index],
                    "shape": list(named_params[attack_layer_index][1].shape),
                    "dtype": "torch.float32",
                },
                "param_names": names,
                "sample_indices": target_sample_indices,
                "observations": [],
                "storage": "sharded_npz",
            }

        checkpoint_before_round: Path | None = None
        captured = False
        round_metrics: list[dict[str, Any]] = []
        for round_idx in range(1, int(args.rounds) + 1):
            start_time = time.time()
            global_params = named_parameters_cpu(global_model)
            if round_idx == observe_round:
                checkpoint_before_round = output_file.with_suffix(".fedavg_round_before.pt")
                checkpoint_before_round.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": global_model.state_dict(),
                        "metadata": {
                            **dict(config),
                            "checkpoint_role": "theta_t_before_observed_client_update",
                            "keep_ratio": float(ratio),
                        },
                    },
                    checkpoint_before_round,
                )
                data["x"].append(xs[target_id].numpy())
                data["y"].append(ys[target_id].numpy())
                if getattr(global_model, "model_type", None) == "conv":
                    chunks = []
                    feature_chunk = max(1, min(resolve_micro_batch_size(args), int(xs[target_id].shape[0])))
                    with torch.no_grad():
                        for start in range(0, int(xs[target_id].shape[0]), feature_chunk):
                            _, z_chunk = global_model(xs[target_id][start : start + feature_chunk].to(device), return_z=True)
                            chunks.append(z_chunk.detach().cpu())
                    data["z"].append(torch.cat(chunks, dim=0).numpy())

            aggregate = zero_update_like(global_params)
            client_sparse_stats = []
            for client_id, (x_client, y_client, weight) in enumerate(zip(xs, ys, weights)):
                update = train_single_client_update(
                    global_model=global_model,
                    x_cpu=x_client,
                    y_cpu=y_client,
                    args=args,
                    device=device,
                    log_prefix=f"[fedavg-single-collect] keep={ratio:g} round {round_idx}/{args.rounds} client {client_id + 1}/{args.num_clients}",
                )
                source_fp32_hash = hash_tensor_sequence(list(update.values()))
                sparse_update, sparse_stats, raw_masks = sparse_update_and_masks(update, float(ratio))
                client_sparse_stats.append(sparse_stats)
                add_weighted_update_(aggregate, sparse_update, weight)

                if round_idx == observe_round and client_id == target_id:
                    data["grad"].append([
                        value.detach().cpu().numpy().astype(np.float32, copy=False)
                        for value in sparse_update.values()
                    ])
                    data["quant_stats"].append({"quant_bits": 32, "rounding": "none"})
                    data["sparse_stats"].append(sparse_stats)
                    data["metadata"].update(
                        {
                            "checkpoint": checkpoint_manifest(checkpoint_before_round),
                            "source_fp32_update_hash": source_fp32_hash,
                            "sparse_nonzero_fraction": sparse_stats.get("sparse_nonzero_fraction"),
                            "sparse_relative_l2": sparse_stats.get("sparse_relative_l2"),
                            "sparse_cosine_similarity_to_fp32": sparse_stats.get("sparse_cosine_similarity_to_fp32"),
                            "sparse_retained_l2_norm_ratio": sparse_stats.get("sparse_retained_l2_norm_ratio"),
                            "observed_round": int(round_idx),
                            "observed_client_id": int(client_id),
                        }
                    )
                    if observation is not None and observation_file is not None:
                        mask = raw_masks[attack_layer_index].detach().cpu().numpy().astype(np.bool_, copy=False)
                        attack_tensor = sparse_update[names[attack_layer_index]].detach().cpu().numpy().astype(
                            np.float32,
                            copy=False,
                        )
                        values = attack_tensor[mask].astype(np.float32, copy=True)
                        tau = tau_from_policy(sparse_stats, values, args.tau_policy)
                        row = write_observation_shard(
                            index_path=observation_file,
                            batch_idx=0,
                            keep_ratio=ratio,
                            mask=mask,
                            values=values,
                            tau=tau,
                            tau_policy=args.tau_policy,
                            global_total=int(sparse_stats.get("total_elements", 0)),
                            global_kept=int(sparse_stats.get("mask_kept_elements", 0)),
                            selection_mode="fedavg_target_client_global_topk",
                        )
                        observation["observations"].append([row])
                        print(
                            f"[fedavg-single-collect] keep={ratio:g} observed round={round_idx} "
                            f"client={client_id} attack_obs={row['observed_elements']}/{row['total_elements']} "
                            f"tau={tau:.6g}",
                            flush=True,
                        )
                    captured = True
                del update, sparse_update
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            apply_parameter_update(global_model, global_params, aggregate)
            metric = {
                "round": int(round_idx),
                "keep_ratio": float(ratio),
                "aggregate_update_l2": float(sum(float((tensor * tensor).sum().item()) for tensor in aggregate.values()) ** 0.5),
                "client_sparse_topk_threshold_abs_mean": (
                    float(np.mean([float(row["topk_threshold_abs"]) for row in client_sparse_stats if row.get("topk_threshold_abs") is not None]))
                    if any(row.get("topk_threshold_abs") is not None for row in client_sparse_stats)
                    else None
                ),
                "client_sparse_nonzero_fraction_mean": float(np.mean([float(row["sparse_nonzero_fraction"]) for row in client_sparse_stats])),
                "round_time": float(time.time() - start_time),
            }
            round_metrics.append(metric)
            print(
                f"[fedavg-single-collect] keep={ratio:g} round {round_idx}/{args.rounds} "
                f"l2={metric['aggregate_update_l2']:.6g} time={metric['round_time']:.1f}s",
                flush=True,
            )
            del global_params, aggregate
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not captured or checkpoint_before_round is None:
            raise RuntimeError("failed to capture target client update at observed FedAvg round")

        data["metadata"]["sample_indices_hash"] = sha256_bytes(json.dumps(target_sample_indices).encode("utf-8"))
        data["metadata"]["label_hash"] = hash_tensor_sequence(data["y"])
        data["metadata"]["update_tensor_hash"] = hash_tensor_sequence(data["grad"])
        data["metadata"]["fedavg_round_metrics"] = round_metrics
        output_file.parent.mkdir(parents=True, exist_ok=True)
        write_pickle(output_file, data)
        write_json(
            output_file.with_suffix(".manifest.json"),
            {
                **single_client_config(args),
                "method": "fedavg_single_client_sparse_update_collector",
                "sparse_policy": args.sparse_policy,
                "keep_ratio": float(ratio),
                "tau_policy": args.tau_policy,
                "observed_zero_policy": OBSERVED_ZERO_POLICY,
                "checkpoint": checkpoint_manifest(checkpoint_before_round),
                "source_update_file": str(output_file.resolve()),
            },
        )
        print(f"[fedavg-single-collect] wrote keep={ratio:g}: {output_file}", flush=True)

        if observation is not None and observation_file is not None:
            observation["metadata"]["checkpoint"] = checkpoint_manifest(checkpoint_before_round)
            observation["metadata"]["source_sparse_update_file"] = str(output_file.resolve())
            observation["metadata"]["source_sparse_update_hash"] = sha256_file(output_file)
            observation["metadata"]["sample_indices_hash"] = sha256_bytes(json.dumps(target_sample_indices).encode("utf-8"))
            observation["metadata"]["client_attack_observation_hash"] = hash_observation_rows(observation["observations"])
            write_pickle(observation_file, observation)
            write_json(observation_file.with_suffix(".manifest.json"), observation["metadata"])
            print(f"[fedavg-single-collect] wrote observation keep={ratio:g}: {observation_file}", flush=True)

        del global_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
    if sampler == "zero_fill" or tau <= 0.0:
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
    randomized_desparsification = (
        observation_data is not None and keep_ratio < 1.0 and sampler != "zero_fill"
    )
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
    candidate_generation = (
        "fedavg_context_single_client_fixed_sparse_observation_deterministic_zero_fill"
        if not randomized_desparsification
        else "fedavg_context_single_client_fixed_sparse_observation_randomized_desparsification"
    )
    metadata = copy.deepcopy(base_data["metadata"])
    metadata.update(
        {
            "cora_variant": "fedavg_context_single_client_compressed_update" if randomized_desparsification else None,
            "cora_stage": "independent_member_restart" if randomized_desparsification else "baseline",
            "cora_wrapper": randomized_desparsification,
            "cora_candidate_generation": candidate_generation,
            "fixed_sparse_observation": True,
            "randomized_desparsification": randomized_desparsification,
            "randomized_desparsification_policy": RANDOMIZED_DESPARSE_POLICY if randomized_desparsification else "none",
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


def ranking_output_names(output_prefix: str) -> tuple[str, str, str]:
    if output_prefix == "consistency":
        return "consistency_member_ranking.csv", "selected_reconstructions.pkl", "manifest.json"
    return (
        f"{output_prefix}_member_ranking.csv",
        f"{output_prefix}_selected_reconstructions.pkl",
        f"{output_prefix}_manifest.json",
    )


def read_member_ranking_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, Any] = dict(row)
            for key in ("member", "batches"):
                if parsed.get(key) not in {None, ""}:
                    parsed[key] = int(parsed[key])
            for key in ("score", "mean_relative_l2_to_observed_client_update"):
                if parsed.get(key) not in {None, ""}:
                    parsed[key] = float(parsed[key])
            rows.append(parsed)
    rows.sort(key=lambda row: float(row["mean_relative_l2_to_observed_client_update"]))
    return rows


def rank_members_by_consistency(
    *,
    args: argparse.Namespace,
    keep_ratio: float,
    source_data: Mapping[str, Any],
    run_paths: Sequence[Path],
    member_ids: Sequence[int] | None = None,
    output_prefix: str = "consistency",
    manifest_extra: Mapping[str, Any] | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    if not run_paths:
        raise ValueError("No member run paths provided for consistency ranking")
    if member_ids is None:
        member_ids = list(range(len(run_paths)))
    if len(member_ids) != len(run_paths):
        raise ValueError("member_ids must have the same length as run_paths")
    device = get_device()
    checkpoint_info = source_data.get("metadata", {}).get("checkpoint", {})
    checkpoint_path = Path(checkpoint_info["path"]) if checkpoint_info.get("path") else fedavg_start_checkpoint(args)
    model = build_experiment_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        device=device,
        checkpoint_path=checkpoint_path,
        seed=args.seed,
    )
    model.eval()
    attack_idx = int(source_data["attack_layer"]["index"])
    attack_name = str(source_data["attack_layer"]["name"])
    observed_attack = [np.asarray(batch[attack_idx], dtype=np.float32) for batch in source_data["grad"]]
    n_batches = min(args.attack_n_batch or len(observed_attack), len(observed_attack))
    ranked = []
    for position, run_path in enumerate(run_paths):
        member_idx = int(member_ids[position])
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
                f"[single-rank] member m{member_idx:03d} ({position + 1}/{len(run_paths)}) "
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
                "ranking_observation": "one_client_compressed_update_from_fedavg_round",
            }
        )
    ranked.sort(key=lambda row: float(row["mean_relative_l2_to_observed_client_update"]))
    output_dir = cora_output_dir(args, keep_ratio)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_name, selected_name, manifest_name = ranking_output_names(output_prefix)
    write_csv(output_dir / ranking_name, ranked)
    best = ranked[0]
    write_pickle(output_dir / selected_name, read_pickle(Path(best["source_run_path"]) / "reconstructions.pkl"))
    manifest = {
        "schema_version": 1,
        "method": "CoRA FedAvg-context single-client compressed-update consistency ranking",
        "keep_ratio": float(keep_ratio),
        "ensemble_size": int(len(run_paths)),
        "ranked_member_ids": [int(value) for value in member_ids],
        "selected_member": int(best["member"]),
        "selected_member_run_path": best["source_run_path"],
        "ranking_observation": "one_client_compressed_update_from_fedavg_round",
        "model_checkpoint_for_attack_and_ranking": str(checkpoint_path.resolve()) if checkpoint_path else None,
        "exact_single_client_batch_ranking": True,
        "member_ranking_file": ranking_name,
    }
    if output_prefix != "consistency":
        manifest["ranking_stage"] = output_prefix
    if manifest_extra:
        manifest.update(dict(manifest_extra))
    write_json(output_dir / manifest_name, manifest)
    return output_dir, ranked


def validate_fia_screening_args(args: argparse.Namespace) -> None:
    screen_iter = int(args.fia_screen_iter)
    screen_top_k = int(args.fia_screen_top_k)
    if screen_iter < 0:
        raise ValueError("--fia_screen_iter must be >= 0")
    if screen_top_k < 0:
        raise ValueError("--fia_screen_top_k must be >= 0")
    if screen_iter == 0 and screen_top_k == 0:
        return
    if screen_iter <= 0 or screen_top_k <= 0:
        raise ValueError("--fia_screen_iter and --fia_screen_top_k must be set together")
    if not args.run_attack:
        raise ValueError("--fia_screen_iter requires --run_attack")
    if args.skip_consistency_ranking:
        raise ValueError("--fia_screen_iter requires consistency ranking")
    if int(args.n_iter_fi) <= 0:
        raise ValueError("--fia_screen_iter requires --n_iter_fi > 0")
    if screen_iter >= int(args.n_iter_fi):
        raise ValueError("--fia_screen_iter must be smaller than --n_iter_fi")


def use_fia_screening(args: argparse.Namespace, keep_ratio: float, member_count: int) -> bool:
    return (
        int(args.fia_screen_iter) > 0
        and int(args.fia_screen_top_k) > 0
        and float(keep_ratio) < 1.0
        and int(member_count) > 1
    )


def run_cora_member_attack(
    *,
    args: argparse.Namespace,
    attack_args: argparse.Namespace,
    source_data: Mapping[str, Any],
    source_file: Path,
    source_hash: str,
    observation_file: Path | None,
    observation_data: Mapping[str, Any] | None,
    keep_ratio: float,
    member_count: int,
    member_idx: int,
    member_seed: int,
    attack_checkpoint: Path | None,
    target_fia_iter: int,
    phase: str,
    fresh_start: bool,
) -> Path:
    member_args = copy.copy(attack_args)
    member_args.fia_stop_iter = int(target_fia_iter)
    member_args.fresh_start = bool(fresh_start)
    exp_name = member_exp_name(member_args, keep_ratio, member_count, member_idx)
    existing_run = author_member_run_path(member_args, exp_name)
    zero_fill_baseline = args.desparse_sampler == "zero_fill"
    is_cora = bool(keep_ratio < 1.0 and member_count > 1 and not zero_fill_baseline)
    if args.reuse_existing and not fresh_start and member_has_completed_fia(existing_run, int(target_fia_iter)):
        if is_cora:
            require_negentropy_run(existing_run, member_args.ne)
        print(
            f"[fedavg-single-member] reuse {phase} member m{member_idx:03d} "
            f"target_fia_iter={target_fia_iter}: {existing_run}",
            flush=True,
        )
        return existing_run

    member_data = materialize_member_update_data(
        base_data=source_data,
        base_update_file=source_file,
        base_update_file_hash=source_hash,
        observation_file=observation_file,
        observation_data=observation_data,
        keep_ratio=float(keep_ratio),
        member_idx=member_idx,
        member_seed=member_seed,
        sampler=args.desparse_sampler,
    )
    naive_single_sample = bool(keep_ratio < 1.0 and member_count == 1 and not zero_fill_baseline)
    member_data["metadata"].update(
        {
            "ensemble_size": member_count,
            "cora_variant": "fedavg_context_single_client_compressed_update" if is_cora else None,
            "cora_stage": "independent_member_restart" if is_cora else "baseline",
            "cora_wrapper": is_cora,
        }
    )
    if zero_fill_baseline:
        method = "FedAvg-context single-client compressed update: deterministic zero-fill sparse baseline, author CPA, Direct FIA"
        attack_variant = "zero_fill_sparse_baseline"
    elif naive_single_sample:
        method = "FedAvg-context single-client compressed update: single-sample naive desparsification, author CPA, Direct FIA"
        attack_variant = "naive_desparsification_r1"
    elif is_cora:
        method = (
            "CoRA FedAvg-context single-client compressed update: fixed sparse observation, "
            "attack-FC randomized desparsification, author CPA, Direct FIA"
        )
        attack_variant = "cora"
    else:
        method = "FedAvg-context single-client FP32 reference: author CPA, Direct FIA"
        attack_variant = "fp32_reference"
    return run_author_cpa_with_update_file(
        member_args,
        enable_negentropy=is_cora,
        exp_name=exp_name,
        update_file=source_file,
        update_data=member_data,
        method=method,
        extra_manifest={
            "cora_variant": "fedavg_context_single_client_compressed_update" if is_cora else None,
            "attack_variant": attack_variant,
            "ensemble_member": int(member_idx),
            "ensemble_size": int(member_count),
            "desparsification_seed": int(member_seed),
            "desparse_sampler": args.desparse_sampler if keep_ratio < 1.0 else "deterministic_identity",
            "model_checkpoint_for_attack": str(attack_checkpoint.resolve()) if attack_checkpoint else None,
            "paper_threat_model_alignment": (
                "server observes one client's compressed update from a FedAvg round and does not know removed values"
            ),
        },
        attack_order_policy="FedAvg-context single-client compressed update; no oracle permutation",
    )


def main() -> None:
    args = parse_args()
    if args.fi_method != "direct":
        raise ValueError("FedAvg-context single-client CoRA supports Direct FIA only")
    if args.use_labels:
        raise ValueError("--use_labels is disallowed for the no-oracle attack setting")
    validate_fia_screening_args(args)
    args.fedavg_start_checkpoint = args.global_checkpoint
    args.samples_per_client = resolve_samples_per_client(args)
    args.n_samples = int(args.samples_per_client)
    args.attack_trials = 1
    args.local_batch_size = args.local_batch_size or args.n_samples
    args.sparsification_application = "fedavg_client_update_at_observed_round"
    require_global_topk(args.sparse_policy)
    if args.experiment_tag and not all(char.isalnum() or char in {"-", "_"} for char in args.experiment_tag):
        raise ValueError("--experiment_tag may contain only letters, digits, '-' and '_'")
    if args.prepared_source_dir is not None:
        args.prepared_source_dir = args.prepared_source_dir.resolve()

    source_files = (
        prepared_source_files(args)
        if args.prepared_source_dir is not None
        else collect_fedavg_single_client_sparse_pickles(args)
    )
    seed_plan = make_member_seed_plan(int(args.ensemble_size), int(args.noise_seed))

    for ratio in sorted(source_files, reverse=True):
        source_file = source_files[ratio]
        source_data = read_pickle(source_file)
        source_hash = sha256_file(source_file)
        checkpoint_info = source_data.get("metadata", {}).get("checkpoint", {})
        attack_checkpoint = Path(checkpoint_info["path"]) if checkpoint_info.get("path") else fedavg_start_checkpoint(args)
        attack_args = copy.copy(args)
        attack_args.global_checkpoint = attack_checkpoint
        attack_args.n_samples = int(source_data["metadata"]["n_samples"])
        observation_file = source_observation_path(args, ratio, source_file)
        observation_data = read_pickle(observation_file) if observation_file and observation_file.exists() else None
        member_count = 1 if ratio >= 1.0 else int(args.ensemble_size)
        run_paths: list[Path] = []
        run_member_ids: list[int] = []
        final_ranking_extra: dict[str, Any] = {}
        output_dir = cora_output_dir(args, ratio)
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.run_attack and use_fia_screening(args, float(ratio), member_count):
            screen_iter = int(args.fia_screen_iter)
            screen_top_k = min(int(args.fia_screen_top_k), int(member_count))
            screen_prefix = f"fia_screen_iter{screen_iter}"
            screen_ranking_name, _screen_selected_name, _screen_manifest_name = ranking_output_names(screen_prefix)
            screen_ranking_file = output_dir / screen_ranking_name
            screen_paths: list[Path] = []
            screen_member_ids: list[int] = []
            print(
                f"[fedavg-single-screen] keep={ratio:g} run {member_count} members "
                f"to FIA iter {screen_iter}, then continue top {screen_top_k}",
                flush=True,
            )
            for member_idx in range(member_count):
                member_seed = (
                    int(seed_plan[member_idx]["desparsification_seed"]) if ratio < 1.0 else int(args.noise_seed)
                )
                run_path = run_cora_member_attack(
                    args=args,
                    attack_args=attack_args,
                    source_data=source_data,
                    source_file=source_file,
                    source_hash=source_hash,
                    observation_file=observation_file,
                    observation_data=observation_data,
                    keep_ratio=float(ratio),
                    member_count=member_count,
                    member_idx=member_idx,
                    member_seed=member_seed,
                    attack_checkpoint=attack_checkpoint,
                    target_fia_iter=screen_iter,
                    phase="screen",
                    fresh_start=bool(args.fresh_start),
                )
                screen_paths.append(run_path)
                screen_member_ids.append(member_idx)

            if args.reuse_existing and not args.fresh_start and screen_ranking_file.exists():
                screen_ranked = read_member_ranking_csv(screen_ranking_file)
                screen_rank_dir = output_dir
                print(f"[fedavg-single-screen] reuse screening ranking: {screen_ranking_file}", flush=True)
            else:
                screen_rank_dir, screen_ranked = rank_members_by_consistency(
                    args=attack_args,
                    keep_ratio=float(ratio),
                    source_data=source_data,
                    run_paths=screen_paths,
                    member_ids=screen_member_ids,
                    output_prefix=screen_prefix,
                    manifest_extra={
                        "screen_fia_iter": int(screen_iter),
                        "screen_top_k": int(screen_top_k),
                        "screened_ensemble_size": int(member_count),
                        "final_fia_iter": int(args.n_iter_fi),
                    },
                )
            if len(screen_ranked) < screen_top_k:
                raise RuntimeError(f"Screening produced {len(screen_ranked)} ranked members, expected at least {screen_top_k}")
            selected_member_ids = [int(row["member"]) for row in screen_ranked[:screen_top_k]]
            print(
                "[fedavg-single-screen] selected "
                + " ".join(f"m{member_idx:03d}" for member_idx in selected_member_ids)
                + f" from {screen_rank_dir}",
                flush=True,
            )

            for member_idx in selected_member_ids:
                member_seed = (
                    int(seed_plan[member_idx]["desparsification_seed"]) if ratio < 1.0 else int(args.noise_seed)
                )
                run_path = run_cora_member_attack(
                    args=args,
                    attack_args=attack_args,
                    source_data=source_data,
                    source_file=source_file,
                    source_hash=source_hash,
                    observation_file=observation_file,
                    observation_data=observation_data,
                    keep_ratio=float(ratio),
                    member_count=member_count,
                    member_idx=member_idx,
                    member_seed=member_seed,
                    attack_checkpoint=attack_checkpoint,
                    target_fia_iter=int(args.n_iter_fi),
                    phase="final",
                    fresh_start=False,
                )
                run_paths.append(run_path)
                run_member_ids.append(member_idx)
            final_ranking_extra = {
                "screening_enabled": True,
                "screen_fia_iter": int(screen_iter),
                "screen_top_k": int(screen_top_k),
                "screened_ensemble_size": int(member_count),
                "continued_member_ids": [int(value) for value in selected_member_ids],
                "screening_member_ranking_file": screen_ranking_name,
            }
        elif args.run_attack:
            for member_idx in range(member_count):
                member_seed = (
                    int(seed_plan[member_idx]["desparsification_seed"]) if ratio < 1.0 else int(args.noise_seed)
                )
                run_path = run_cora_member_attack(
                    args=args,
                    attack_args=attack_args,
                    source_data=source_data,
                    source_file=source_file,
                    source_hash=source_hash,
                    observation_file=observation_file,
                    observation_data=observation_data,
                    keep_ratio=float(ratio),
                    member_count=member_count,
                    member_idx=member_idx,
                    member_seed=member_seed,
                    attack_checkpoint=attack_checkpoint,
                    target_fia_iter=int(args.n_iter_fi),
                    phase="full",
                    fresh_start=bool(args.fresh_start),
                )
                run_paths.append(run_path)
                run_member_ids.append(member_idx)

        observation_manifest = {
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
            "observation": "one_client_compressed_update_from_fedavg_round",
            "attacker_observes_round_aggregate": False,
            "model_checkpoint_for_attack": str(attack_checkpoint.resolve()) if attack_checkpoint else None,
        }
        if final_ranking_extra:
            observation_manifest.update(final_ranking_extra)
        write_json(output_dir / "observation_manifest.json", observation_manifest)
        if args.run_attack and not args.skip_consistency_ranking:
            rank_dir, _ranked = rank_members_by_consistency(
                args=attack_args,
                keep_ratio=float(ratio),
                source_data=source_data,
                run_paths=run_paths,
                member_ids=run_member_ids,
                manifest_extra=final_ranking_extra,
            )
            print(f"[fedavg-single-rank] keep={ratio:g} ranking saved: {rank_dir}", flush=True)
        elif args.run_attack:
            print("[fedavg-single-rank] skipped", flush=True)
        else:
            print(f"[fedavg-single-cora] artifact ready: {source_file}", flush=True)


if __name__ == "__main__":
    main()
