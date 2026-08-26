#!/usr/bin/env python3
"""CoRA wrapper for aggregate sparse FedAvg round-update observations.

This runner differs from ``cora_sparse_update_cpa`` in the observation model.
It never uses per-client sparse observations for candidate generation or
ranking.  It collects or loads a CPA-format round update whose attack-visible
gradient is the sparse aggregate communication result:

    Y_round = sum_k alpha_k S_r(theta_k_local - theta_global_before_round)

Then each CoRA member fills only dropped aggregate attack-FC coordinates before
running the author CPA/FIA implementation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
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
SPARSE_CPA_ROOT = REPO_ROOT / "sparsified_update_cpa"
if str(SPARSE_CPA_ROOT) not in sys.path:
    sys.path.insert(0, str(SPARSE_CPA_ROOT))

from author_runner import run_author_cpa_with_update_file  # noqa: E402
from common import (  # noqa: E402
    AUTHOR_SOURCE_HASH,
    checkpoint_manifest,
    get_dataloaders,
    hash_json,
    hash_tensor_sequence,
    parameter_schema,
    read_pickle,
    sha256_bytes,
    sha256_file,
    validate_parameter_mapping,
    write_json,
    write_pickle,
)
from models import get_model  # noqa: E402
from sparsification import (  # noqa: E402
    OBSERVED_ZERO_POLICY,
    format_keep_ratio,
    parse_keep_ratio_or_percent,
    sparsify_update_global_topk,
    validate_keep_ratio,
)
from utils import get_device  # noqa: E402


EXPERIMENT_NAME = "cora_round_update_cpa"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs_v1"
SPARSE_AUTHOR_RUN_ROOT = SPARSE_CPA_ROOT / "outputs_v1" / "author_runs"
IMAGE_MEAN_STD = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "tiny_imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds", type=str, default="imagenet")
    parser.add_argument("--model", type=str, default="vgg16")
    parser.add_argument("--h_dim", type=int, default=256)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--samples_per_client", type=int, default=8)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--local_batch_size", type=int, default=None)
    parser.add_argument("--local_lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--private_split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--global_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--round_update_file",
        type=Path,
        default=None,
        help=(
            "Optional CPA-format sparse aggregate round update. If omitted, the "
            "runner collects one sparse FedAvg trajectory per keep ratio."
        ),
    )
    parser.add_argument("--reuse_existing", action="store_true")
    parser.add_argument("--fresh_start", action="store_true")
    parser.add_argument("--run_attack", action="store_true")
    parser.add_argument("--keep_ratios", type=parse_keep_ratio_or_percent, nargs="+", default=[0.8, 0.5, 0.3, 0.1])
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--noise_seed", type=int, default=1234)
    parser.add_argument("--desparse_sampler", type=str, default="retained_rescaled", choices=["retained_rescaled", "uniform_tau"])
    parser.add_argument("--success_lpips_threshold", type=float, default=0.50)
    parser.add_argument("--consistency_label_mode", type=str, default="model_pred", choices=["model_pred", "zero"])
    parser.add_argument("--skip_consistency_ranking", action="store_true")
    parser.add_argument(
        "--allow_partial_aggregate_ranking",
        action="store_true",
        help="Allow approximate aggregate ranking when FIA reconstructed fewer than all attacked samples.",
    )
    parser.add_argument("--attack_n_batch", type=int, default=None)
    parser.add_argument("--attack_n_iter", type=int, default=25000)
    parser.add_argument("--attack_n_log", type=int, default=1000)
    parser.add_argument("--attack_lr", type=float, default=1e-3)
    parser.add_argument("--decor", type=float, default=5.3)
    parser.add_argument("--T", type=float, default=7.7)
    parser.add_argument("--tv", type=float, default=0.1)
    parser.add_argument("--nv", type=float, default=0.13)
    parser.add_argument("--l1", type=float, default=5.0)
    parser.add_argument("--ne", type=float, default=1.0)
    parser.add_argument("--n_iter_fi", type=int, default=25000)
    parser.add_argument("--n_log_fi", type=int, default=2000)
    parser.add_argument("--lr_fi", type=float, default=1e-1)
    parser.add_argument("--n_sample_fi", type=int, default=16)
    parser.add_argument("--fia_micro_batch_size", type=int, default=None)
    parser.add_argument("--fi", type=float, default=1.0)
    parser.add_argument("--fi_method", type=str, default="direct", choices=["direct", "gm"])
    parser.add_argument("--use_labels", action="store_true")
    parser.add_argument("--ideal_emb_rec", action="store_true")
    parser.add_argument("--project", type=str, default="cpa_round_cora")
    return parser.parse_args()


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str | None) -> torch.device:
    if requested is None:
        return get_device()
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(requested)


def round_group_tag(args: argparse.Namespace) -> str:
    config = {
        "ds": args.ds,
        "model": args.model,
        "h_dim": args.h_dim,
        "num_clients": args.num_clients,
        "rounds": args.rounds,
        "samples_per_client": args.samples_per_client,
        "local_epochs": args.local_epochs,
        "local_batch_size": args.local_batch_size or args.samples_per_client,
        "local_lr": args.local_lr,
        "private_split": args.private_split,
        "seed": args.seed,
        "global_checkpoint": str(resolve_path(args.global_checkpoint)) if args.global_checkpoint else None,
    }
    return (
        f"round_observed_clients{args.num_clients}_rounds{args.rounds}"
        f"_spc{args.samples_per_client}_local{args.local_epochs}_{hash_json(config)[:16]}"
    )


def round_update_path(args: argparse.Namespace, keep_ratio: float) -> Path:
    return (
        OUTPUT_ROOT
        / args.ds
        / args.model
        / "round_updates"
        / round_group_tag(args)
        / "client_topk_then_aggregate_observed_only"
        / "global_topk"
        / f"keep_{format_keep_ratio(keep_ratio)}.pickle"
    )


def cora_output_dir(args: argparse.Namespace, keep_ratio: float) -> Path:
    return (
        OUTPUT_ROOT
        / args.ds
        / args.model
        / round_group_tag(args)
        / "client_topk_then_aggregate_observed_only"
        / "global_topk"
        / f"keep_{format_keep_ratio(keep_ratio)}"
        / f"R{args.ensemble_size}_seed{args.noise_seed}_{args.consistency_label_mode}"
    )


def author_member_run_path(args: argparse.Namespace, exp_name: str) -> Path:
    return SPARSE_AUTHOR_RUN_ROOT / args.ds / args.model / "attack" / "cp_direct" / "nodef" / f"{exp_name}_uia"


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


def apply_parameter_update(model: nn.Module, global_params: Mapping[str, torch.Tensor], update: Mapping[str, torch.Tensor]) -> None:
    state = model.state_dict()
    for name, delta in update.items():
        state[name] = (global_params[name] + delta).to(device=state[name].device, dtype=state[name].dtype)
    model.load_state_dict(state, strict=True)


def compute_update(local_params: Mapping[str, torch.Tensor], global_params: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, local_params[name].sub(global_params[name]).float()) for name in global_params)


def global_topk_attack_layer_observation_from_params_np(
    local_params: Mapping[str, torch.Tensor],
    global_params: Mapping[str, torch.Tensor],
    *,
    names: Sequence[str],
    attack_layer_index: int,
    keep_ratio: float,
) -> dict[str, Any]:
    """Return the attack-layer slice of full-model Global Top-k using NumPy.

    Ranking only needs the recompressed attack-FC layer.  Avoiding PyTorch's
    huge CPU kthvalue/sub tensors sidesteps c10 native crashes on low-RAM hosts.
    """
    validate_keep_ratio(float(keep_ratio))
    if attack_layer_index < 0 or attack_layer_index >= len(names):
        raise ValueError("attack_layer_index is outside the update parameter mapping")

    shapes = [tuple(global_params[name].shape) for name in names]
    sizes = [int(np.prod(shape, dtype=np.int64)) for shape in shapes]
    total_elements = int(sum(sizes))
    if total_elements < 1:
        raise ValueError("update tensors contain no elements")
    kept_elements = int(math.ceil(total_elements * float(keep_ratio)))
    kept_elements = max(1, min(total_elements, kept_elements))

    def fill_delta(name: str, out: np.ndarray) -> None:
        local_np = local_params[name].detach().cpu().numpy().reshape(-1)
        global_np = global_params[name].detach().cpu().numpy().reshape(-1)
        np.subtract(local_np, global_np, out=out, casting="unsafe")

    def fill_delta_abs(name: str, out: np.ndarray) -> None:
        fill_delta(name, out)
        np.abs(out, out=out)

    if kept_elements == total_elements:
        attack_name = names[attack_layer_index]
        attack_delta = (
            local_params[attack_name].detach().cpu().numpy()
            - global_params[attack_name].detach().cpu().numpy()
        ).astype(np.float32, copy=False)
        return {
            "name": attack_name,
            "mask": np.ones(attack_delta.shape, dtype=np.bool_),
            "values": attack_delta.reshape(-1).astype(np.float32, copy=True),
            "selection_mode": "identity",
        }

    flat_abs = np.empty(total_elements, dtype=np.float32)
    offset = 0
    for name, size in zip(names, sizes):
        fill_delta_abs(name, flat_abs[offset : offset + size])
        offset += size

    drop_elements = total_elements - kept_elements
    if drop_elements <= kept_elements:
        selection_mode = "drop_smallest"
        threshold_index = drop_elements - 1
        threshold_value = float(np.partition(flat_abs, threshold_index)[threshold_index])
        strict_selected = int(np.count_nonzero(flat_abs < threshold_value))
        tie_needed = int(drop_elements - strict_selected)
    else:
        selection_mode = "keep_largest"
        threshold_index = total_elements - kept_elements
        threshold_value = float(np.partition(flat_abs, threshold_index)[threshold_index])
        strict_selected = int(np.count_nonzero(flat_abs > threshold_value))
        tie_needed = int(kept_elements - strict_selected)
    del flat_abs

    remaining_ties = tie_needed
    attack_name = names[attack_layer_index]
    for index, name in enumerate(names):
        abs_delta_flat = np.empty(sizes[index], dtype=np.float32)
        fill_delta_abs(name, abs_delta_flat)
        abs_delta = abs_delta_flat.reshape(shapes[index])
        tie_mask = abs_delta == threshold_value
        tie_count = int(np.count_nonzero(tie_mask))
        if selection_mode == "drop_smallest":
            drop_ties = min(int(remaining_ties), tie_count)
            if index == attack_layer_index:
                mask = abs_delta > threshold_value
                if tie_count and drop_ties < tie_count:
                    tie_indices = np.flatnonzero(tie_mask.reshape(-1))
                    mask.reshape(-1)[tie_indices[drop_ties:]] = True
                fill_delta(attack_name, abs_delta_flat)
                attack_delta = abs_delta_flat.reshape(shapes[index])
                values = attack_delta[mask].astype(np.float32, copy=True)
                return {"name": attack_name, "mask": mask, "values": values, "selection_mode": selection_mode}
            remaining_ties -= drop_ties
        else:
            keep_ties = min(int(remaining_ties), tie_count)
            if index == attack_layer_index:
                mask = abs_delta > threshold_value
                if keep_ties > 0:
                    tie_indices = np.flatnonzero(tie_mask.reshape(-1))
                    mask.reshape(-1)[tie_indices[:keep_ties]] = True
                fill_delta(attack_name, abs_delta_flat)
                attack_delta = abs_delta_flat.reshape(shapes[index])
                values = attack_delta[mask].astype(np.float32, copy=True)
                return {"name": attack_name, "mask": mask, "values": values, "selection_mode": selection_mode}
            remaining_ties -= keep_ties

    raise RuntimeError("failed to construct attack-layer Top-k observation mask")


def train_one_client(
    global_model: nn.Module,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    *,
    round_idx: int,
    client_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[OrderedDict[str, torch.Tensor], float]:
    print(f"[round-train] client {client_id} model init", flush=True)
    local_model = get_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        load_path=None,
        dataparallel=False,
    )
    global_state = {name: value.detach().cpu() for name, value in global_model.state_dict().items()}
    local_model.load_state_dict(global_state, strict=True)
    del global_state
    local_model.to(device)
    local_model.train()
    print(f"[round-train] client {client_id} optimization start", flush=True)
    previous_cudnn_enabled = torch.backends.cudnn.enabled
    if device.type == "cuda" and args.ds == "imagenet" and args.model == "vgg16":
        torch.cuda.empty_cache()
        torch.backends.cudnn.enabled = False
    optimizer = torch.optim.SGD(local_model.parameters(), lr=float(args.local_lr), momentum=0.0, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + int(round_idx) * 1000 + int(client_id))
    loader = DataLoader(
        TensorDataset(x_cpu, y_cpu),
        batch_size=int(args.local_batch_size or args.samples_per_client),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    total_loss = 0.0
    total_seen = 0
    for _epoch in range(int(args.local_epochs)):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device=device, dtype=torch.float32)
            y_batch = y_batch.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            batch_n = int(x_batch.shape[0])
            micro_batch_size = batch_n
            if args.ds == "imagenet" and args.model == "vgg16" and batch_n > 4:
                micro_batch_size = 4
            batch_loss = 0.0
            for start in range(0, batch_n, micro_batch_size):
                end = min(start + micro_batch_size, batch_n)
                loss = criterion(local_model(x_batch[start:end]), y_batch[start:end])
                chunk_n = int(end - start)
                (loss * (chunk_n / float(batch_n))).backward()
                batch_loss += float(loss.detach().cpu().item()) * chunk_n
            optimizer.step()
            total_loss += batch_loss
            total_seen += batch_n
    if total_seen == 0:
        raise RuntimeError("client had no local samples")
    local_params = named_parameters_cpu(local_model)
    del optimizer, local_model
    torch.backends.cudnn.enabled = previous_cudnn_enabled
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return local_params, total_loss / total_seen


def load_private_clients(args: argparse.Namespace) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    dl_train, dl_test = get_dataloaders(
        args.ds,
        batch_size=1,
        augment=False,
        shuffle_train=False,
        shuffle_test=False,
        num_workers=int(args.num_workers),
    )
    dataset = dl_train.dataset if args.private_split == "train" else dl_test.dataset
    total = int(args.num_clients) * int(args.samples_per_client)
    if total > len(dataset):
        raise ValueError(f"Need {total} private samples, but split has only {len(dataset)}")
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    selected = torch.randperm(len(dataset), generator=generator).tolist()[:total]
    xs, ys, indices = [], [], []
    for start in range(0, total, int(args.samples_per_client)):
        subset_indices = selected[start : start + int(args.samples_per_client)]
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


def extract_embeddings(model: nn.Module, x_cpu: torch.Tensor, *, device: torch.device, batch_size: int) -> np.ndarray:
    was_training = model.training
    model.eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for (x_batch,) in DataLoader(TensorDataset(x_cpu), batch_size=max(1, int(batch_size)), shuffle=False):
            _logits, z = model(x_batch.to(device=device, dtype=torch.float32), return_z=True)
            chunks.append(z.detach().cpu())
    if was_training:
        model.train()
    return torch.cat(chunks, dim=0).numpy()


def dense_identity_stats(update: Mapping[str, torch.Tensor], keep_ratio: float) -> dict[str, Any]:
    total = sum(int(tensor.numel()) for tensor in update.values())
    nonzero = sum(int((tensor != 0).sum().item()) for tensor in update.values())
    return {
        "keep_ratio": float(keep_ratio),
        "total_elements": int(total),
        "mask_kept_elements": int(total),
        "stored_nonzero_elements": int(nonzero),
        "sparse_mask_fraction": 1.0,
        "sparse_nonzero_fraction": nonzero / total if total else 0.0,
        "sparse_relative_l2": 0.0,
        "sparse_cosine_similarity_to_fp32": 1.0,
        "sparse_energy_retained": 1.0,
        "topk_threshold_abs": 0.0,
    }


def sparsify_client_update(update: OrderedDict[str, torch.Tensor], keep_ratio: float) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any]]:
    if float(keep_ratio) >= 1.0:
        return update, dense_identity_stats(update, keep_ratio)
    sparse_update, _packed_masks, stats = sparsify_update_global_topk(update, keep_ratio=float(keep_ratio), pack_masks=False)
    return sparse_update, stats


def mean_or_none(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def collect_sparse_round_update(args: argparse.Namespace, keep_ratio: float, output_file: Path) -> Path:
    if output_file.exists() and args.reuse_existing:
        print(f"[round-collect] reuse keep={keep_ratio:g}: {output_file}", flush=True)
        return output_file
    set_seed(int(args.seed))
    device = choose_device(args.device)
    args.local_batch_size = args.local_batch_size or args.samples_per_client
    xs, ys, original_indices = load_private_clients(args)
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    original_index_all = torch.cat(original_indices, dim=0)
    n_samples = int(x_all.shape[0])
    args.n_samples = n_samples
    args.attack_trials = 1

    model = get_model(model_name=args.model, ds=args.ds, h_dim=args.h_dim, load_path=None, dataparallel=False).to(device)
    if args.global_checkpoint is not None:
        checkpoint = torch.load(resolve_path(args.global_checkpoint), map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    names = [name for name, _param in model.named_parameters()]
    attack_layer_index = int(getattr(model, "attack_index", 0))
    schema = parameter_schema(model)
    client_sizes = [int(x.shape[0]) for x in xs]
    weights = [size / float(sum(client_sizes)) for size in client_sizes]
    metrics = []
    final_aggregate: OrderedDict[str, torch.Tensor] | None = None
    final_z: np.ndarray | None = None
    checkpoint_before_round: Path | None = None

    print(
        f"[round-collect] keep={keep_ratio:g} FedAvg sparse trajectory start "
        f"rounds={args.rounds} clients={args.num_clients} spc={args.samples_per_client}",
        flush=True,
    )
    for round_idx in range(1, int(args.rounds) + 1):
        start = time.time()
        global_params = named_parameters_cpu(model)
        if round_idx == int(args.rounds):
            checkpoint_before_round = output_file.with_suffix(".round_model_before.pt")
            checkpoint_before_round.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_before_round)
            final_z = extract_embeddings(model, x_all, device=device, batch_size=int(args.local_batch_size))
        aggregate = zero_update_like(global_params)
        client_losses = []
        client_sparse_stats = []
        for client_id, (x_client, y_client, weight) in enumerate(zip(xs, ys, weights)):
            local_params, loss = train_one_client(
                model,
                x_client,
                y_client,
                round_idx=round_idx,
                client_id=client_id,
                args=args,
                device=device,
            )
            dense_update = compute_update(local_params, global_params)
            sparse_update, sparse_stats = sparsify_client_update(dense_update, float(keep_ratio))
            add_weighted_update_(aggregate, sparse_update, weight)
            client_losses.append(loss)
            client_sparse_stats.append(sparse_stats)
            del local_params, dense_update, sparse_update
            if device.type == "cuda":
                torch.cuda.empty_cache()
        apply_parameter_update(model, global_params, aggregate)
        metric = {
            "round": int(round_idx),
            "keep_ratio": float(keep_ratio),
            "weighted_train_loss": float(sum(loss * size for loss, size in zip(client_losses, client_sizes)) / sum(client_sizes)),
            "client_train_losses": [float(value) for value in client_losses],
            "aggregate_update_l2": float(sum(float((tensor * tensor).sum().item()) for tensor in aggregate.values()) ** 0.5),
            "client_sparse_topk_threshold_abs_mean": mean_or_none(
                [float(row["topk_threshold_abs"]) for row in client_sparse_stats if row.get("topk_threshold_abs") is not None]
            ),
            "client_sparse_nonzero_fraction_mean": mean_or_none(
                [float(row["sparse_nonzero_fraction"]) for row in client_sparse_stats]
            ),
            "round_time": float(time.time() - start),
        }
        metrics.append(metric)
        print(
            f"[round-collect] keep={keep_ratio:g} round {round_idx}/{args.rounds} "
            f"loss={metric['weighted_train_loss']:.4f} l2={metric['aggregate_update_l2']:.6g} "
            f"time={metric['round_time']:.1f}s",
            flush=True,
        )
        if round_idx == int(args.rounds):
            final_aggregate = OrderedDict((name, value.detach().cpu().clone().float()) for name, value in aggregate.items())
        del global_params, aggregate
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if final_aggregate is None or final_z is None or checkpoint_before_round is None:
        raise RuntimeError("failed to collect final sparse round update")
    grad = [[final_aggregate[name].detach().cpu().numpy().astype(np.float32, copy=False) for name in names]]
    sample_indices = [[int(value) for value in original_index_all.tolist()]]
    metadata = {
        "schema_version": 1,
        "signal": "fedavg_sparse_round_update_observed_aggregate_only",
        "update_definition": "Y_round = sum_k alpha_k S_r(theta_k_local - theta_global_before_round)",
        "attacker_observation": "aggregate_sparse_round_update_only",
        "client_masks_available_to_attacker": False,
        "client_thresholds_available_to_attacker": False,
        "ds": args.ds,
        "model": args.model,
        "checkpoint": checkpoint_manifest(checkpoint_before_round),
        "round_index": int(args.rounds),
        "rounds": int(args.rounds),
        "n_samples": n_samples,
        "attack_trials": 1,
        "num_clients": int(args.num_clients),
        "samples_per_client": int(args.samples_per_client),
        "clients_per_trial": int(args.num_clients),
        "client_granularity": f"{int(args.samples_per_client)}_samples_per_client",
        "local_epochs": int(args.local_epochs),
        "local_batch_size": int(args.local_batch_size),
        "local_steps": int(args.local_epochs) * math.ceil(int(args.samples_per_client) / int(args.local_batch_size)),
        "lr": float(args.local_lr),
        "optimizer": "SGD",
        "momentum": 0.0,
        "weight_decay": 0.0,
        "aggregation": "weighted_mean_by_client_sample_count",
        "aggregation_weights": [float(value) for value in weights],
        "seed": int(args.seed),
        "sparsified_observation": True,
        "sparsification_policy": "global_topk",
        "sparsification_application": "client_topk_then_aggregate_round_observed_only",
        "keep_ratio": float(keep_ratio),
        "observed_zero_policy": OBSERVED_ZERO_POLICY,
        "desparsification_scope": "aggregate_attack_layer",
        "aggregate_only_tau_policy": "min_abs_nonzero_observed_aggregate_attack_layer",
        "trial_semantics": "single_fedavg_round_from_sparse_communication_trajectory",
        "attack_fc_observation_model": "Y_round_attack_fc ~= A Z after client-sparse FedAvg aggregation",
        "az_scope": "attack_fc_layer_only_approximate",
        "author_source_hash": AUTHOR_SOURCE_HASH,
        "sample_indices_hash": sha256_bytes(str(sample_indices).encode("utf-8")),
        "label_hash": hash_tensor_sequence([y_all.numpy()]),
        "update_tensor_hash": hash_tensor_sequence(grad),
        "round_metrics": metrics,
    }
    data = {
        "schema_version": 3,
        "x": [x_all.numpy()],
        "y": [y_all.numpy()],
        "z": [final_z],
        "grad": grad,
        "index": [original_index_all.numpy()],
        "param_names": names,
        "parameter_schema": schema,
        "attack_layer": {"index": attack_layer_index, "name": names[attack_layer_index]},
        "sample_indices": sample_indices,
        "metadata": metadata,
        "whitening_pre_stats": [],
        "sparse_stats": [],
    }
    write_pickle(output_file, data)
    write_json(output_file.with_suffix(".manifest.json"), metadata)
    print(f"[round-collect] wrote keep={keep_ratio:g}: {output_file}", flush=True)
    return output_file


def normalize_loaded_round_data(args: argparse.Namespace, path: Path) -> dict[str, Any]:
    data = read_pickle(path)
    if "metadata" in data and "param_names" in data and "attack_layer" in data:
        metadata = data["metadata"]
        args.n_samples = int(metadata["n_samples"])
        args.attack_trials = int(metadata.get("attack_trials", len(data["grad"])))
        return data

    device = torch.device("cpu")
    model = get_model(model_name=args.model, ds=args.ds, h_dim=args.h_dim, load_path=None, dataparallel=False).to(device)
    names = [name for name, _param in model.named_parameters()]
    attack_layer_index = int(getattr(model, "attack_index", 0))
    validate_parameter_mapping(names, data["grad"][0], model)
    n_batches = len(data["grad"])
    n_samples = int(np.asarray(data["y"][0]).shape[0])
    args.n_samples = n_samples
    args.attack_trials = n_batches
    sample_indices = [
        [int(value) for value in np.asarray(batch_index).reshape(-1).tolist()]
        for batch_index in data.get("index", [list(range(n_samples)) for _ in range(n_batches)])
    ]
    metadata = {
        "schema_version": 1,
        "signal": "external_round_update_cpa_pickle_normalized",
        "update_definition": data.get("meta", [{}])[0].get("update_definition", "external aggregate round update"),
        "attacker_observation": "aggregate_round_update_only",
        "client_masks_available_to_attacker": False,
        "client_thresholds_available_to_attacker": False,
        "ds": args.ds,
        "model": args.model,
        "checkpoint": checkpoint_manifest(resolve_path(args.global_checkpoint)),
        "n_samples": n_samples,
        "attack_trials": n_batches,
        "local_epochs": int(data.get("meta", [{}])[0].get("local_epochs", args.local_epochs)),
        "local_batch_size": int(data.get("meta", [{}])[0].get("local_batch_size", args.local_batch_size or n_samples)),
        "local_steps": int(data.get("meta", [{}])[0].get("local_epochs", args.local_epochs)),
        "lr": float(data.get("meta", [{}])[0].get("lr", args.local_lr)),
        "optimizer": "SGD",
        "momentum": 0.0,
        "weight_decay": 0.0,
        "aggregation": data.get("meta", [{}])[0].get("aggregation", "unknown"),
        "aggregation_weights": data.get("meta", [{}])[0].get("aggregation_weights"),
        "clients_per_trial": data.get("meta", [{}])[0].get("num_clients"),
        "samples_per_client": data.get("meta", [{}])[0].get("samples_per_client"),
        "seed": int(args.seed),
        "sparsified_observation": True,
        "sparsification_policy": "external_or_precomputed",
        "sparsification_application": "aggregate_round_observed_only",
        "keep_ratio": None,
        "observed_zero_policy": OBSERVED_ZERO_POLICY,
        "desparsification_scope": "aggregate_attack_layer",
        "source_update_file": str(path.resolve()),
        "source_update_hash": sha256_file(path),
        "sample_indices_hash": sha256_bytes(str(sample_indices).encode("utf-8")),
        "label_hash": hash_tensor_sequence(data["y"]),
        "update_tensor_hash": hash_tensor_sequence(data["grad"]),
    }
    normalized = dict(data)
    normalized["param_names"] = names
    normalized["parameter_schema"] = parameter_schema(model)
    normalized["attack_layer"] = {"index": attack_layer_index, "name": names[attack_layer_index]}
    normalized["sample_indices"] = sample_indices
    normalized["metadata"] = metadata
    normalized.setdefault("whitening_pre_stats", [])
    return normalized


def attack_layer_observation(data: Mapping[str, Any], batch_idx: int, *, keep_ratio: float) -> dict[str, Any]:
    attack_index = int(data["attack_layer"]["index"])
    observed = np.asarray(data["grad"][batch_idx][attack_index], dtype=np.float32)
    mask = np.ones(observed.shape, dtype=np.bool_) if float(keep_ratio) >= 1.0 else (observed != 0.0)
    values = observed[mask].astype(np.float32, copy=True)
    tau = float(np.min(np.abs(values))) if values.size else 0.0
    return {
        "batch": int(batch_idx),
        "shape": list(observed.shape),
        "mask": mask,
        "values": values,
        "tau": tau,
        "observed_elements": int(mask.sum()),
        "total_elements": int(mask.size),
        "observed_fraction": float(mask.sum() / max(1, mask.size)),
        "tau_policy": "min_abs_nonzero_observed_aggregate_attack_layer",
    }


def randomized_aggregate_attack_layer(observation: Mapping[str, Any], *, rng: np.random.Generator, sampler: str) -> np.ndarray:
    shape = tuple(int(value) for value in observation["shape"])
    mask = np.asarray(observation["mask"], dtype=np.bool_).reshape(shape)
    values = np.asarray(observation["values"], dtype=np.float32)
    tau = float(observation["tau"])
    if bool(mask.all()):
        return values.reshape(shape).astype(np.float32, copy=True)
    sampled = np.zeros(shape, dtype=np.float32)
    dropped = ~mask
    dropped_count = int(np.count_nonzero(dropped))
    if tau <= 0.0:
        pass
    elif sampler == "uniform_tau":
        sampled[dropped] = rng.uniform(-tau, tau, size=dropped_count).astype(np.float32)
    elif sampler == "retained_rescaled":
        if values.size == 0:
            sampled[dropped] = rng.uniform(-tau, tau, size=dropped_count).astype(np.float32)
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
            sampled[dropped] = magnitudes.astype(np.float32, copy=False)
    else:
        raise ValueError(f"unknown sampler: {sampler}")
    sampled[mask] = values
    return sampled.astype(np.float32, copy=False)


def materialize_member_update_data(
    base_data: Mapping[str, Any],
    *,
    keep_ratio: float,
    member_idx: int,
    member_seed: int,
    sampler: str,
    source_file: Path,
    source_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attack_index = int(base_data["attack_layer"]["index"])
    rng = np.random.default_rng(int(member_seed))
    member_grad = []
    observations = []
    attack_layers = []
    for batch_idx, batch_grad in enumerate(base_data["grad"]):
        observation = attack_layer_observation(base_data, batch_idx, keep_ratio=keep_ratio)
        print(
            f"[round-member] keep={keep_ratio:g} member {member_idx + 1} "
            f"batch {batch_idx + 1}/{len(base_data['grad'])} "
            f"observed={observation['observed_fraction']:.4f} tau={observation['tau']:.6g}",
            flush=True,
        )
        attack_layer = randomized_aggregate_attack_layer(observation, rng=rng, sampler=sampler)
        batch_values = list(batch_grad)
        batch_values[attack_index] = attack_layer
        member_grad.append(batch_values)
        attack_layers.append(attack_layer)
        observations.append({key: value for key, value in observation.items() if key not in {"mask", "values"}})
    metadata = copy.deepcopy(base_data["metadata"])
    candidate_hash = hash_tensor_sequence(attack_layers)
    metadata.update(
        {
            "cora_candidate_generation": "aggregate_round_observation_randomized_desparsification",
            "fixed_sparse_observation": True,
            "attacker_observation": "aggregate_sparse_round_update_only",
            "desparsification_scope": "aggregate_attack_layer",
            "missing_value_sampler": sampler,
            "missing_value_abs_bound_policy": "aggregate_observed_min_abs_tau",
            "keep_ratio": float(keep_ratio),
            "ensemble_member": int(member_idx),
            "ensemble_desparsification_seed": int(member_seed),
            "source_round_update_file": str(source_file.resolve()),
            "source_round_update_hash": source_hash,
            "randomized_attack_layer_hash": candidate_hash,
            "update_tensor_hash": sha256_bytes(
                f"{base_data['metadata'].get('update_tensor_hash')}:{candidate_hash}:{member_idx}:{member_seed}".encode("utf-8")
            ),
            "update_file_hash_override": candidate_hash,
        }
    )
    member_data = dict(base_data)
    member_data["metadata"] = metadata
    member_data["grad"] = member_grad
    return member_data, observations


def normalize_unit_image(x_unit: torch.Tensor, ds: str) -> torch.Tensor:
    if ds not in IMAGE_MEAN_STD:
        return x_unit.clamp(0.0, 1.0)
    mean_values, std_values = IMAGE_MEAN_STD[ds]
    mean = torch.tensor(mean_values, device=x_unit.device, dtype=x_unit.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std_values, device=x_unit.device, dtype=x_unit.dtype).view(1, 3, 1, 1)
    return (x_unit.clamp(0.0, 1.0) - mean) / std


def choose_labels(mode: str, model: nn.Module, x_norm: torch.Tensor, *, batch_size: int | None = None) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros(int(x_norm.shape[0]), dtype=torch.long, device=x_norm.device)
    if mode == "model_pred":
        chunk_size = max(1, min(int(batch_size or x_norm.shape[0]), int(x_norm.shape[0])))
        preds = []
        with torch.no_grad():
            for start in range(0, int(x_norm.shape[0]), chunk_size):
                end = min(start + chunk_size, int(x_norm.shape[0]))
                preds.append(torch.argmax(model(x_norm[start:end]), dim=1).detach())
        return torch.cat(preds, dim=0)
    raise ValueError(f"unknown consistency label mode: {mode}")


def recompress_reconstruction_round_attack_layer(
    *,
    args: argparse.Namespace,
    global_model: nn.Module,
    rec_fi: np.ndarray,
    keep_ratio: float,
    attack_layer_name: str,
) -> np.ndarray:
    device = next(global_model.parameters()).device
    rec = torch.as_tensor(rec_fi, device=device, dtype=torch.float32)
    x_norm = normalize_unit_image(rec, args.ds)
    label_batch_size = max(1, min(int(args.local_batch_size or args.samples_per_client), int(rec.shape[0])))
    print(
        f"[round-rank-detail] label inference mode={args.consistency_label_mode} "
        f"n={int(rec.shape[0])} batch_size={label_batch_size}",
        flush=True,
    )
    y_pred = choose_labels(args.consistency_label_mode, global_model, x_norm, batch_size=label_batch_size)
    samples_per_client = int(args.samples_per_client)
    num_clients = int(math.ceil(int(rec.shape[0]) / samples_per_client))
    weights = [samples_per_client / float(int(rec.shape[0])) for _ in range(num_clients)]
    if num_clients > 0:
        weights[-1] = (int(rec.shape[0]) - samples_per_client * (num_clients - 1)) / float(int(rec.shape[0]))
    global_params = named_parameters_cpu(global_model)
    names = list(global_params.keys())
    if attack_layer_name not in global_params:
        raise ValueError(f"attack layer {attack_layer_name!r} is not in the model parameters")
    attack_layer_index = names.index(attack_layer_name)
    aggregate_attack = np.zeros(tuple(global_params[attack_layer_name].shape), dtype=np.float32)
    for client_id in range(num_clients):
        start = client_id * samples_per_client
        end = min(start + samples_per_client, int(rec.shape[0]))
        print(
            f"[round-rank-detail] client {client_id + 1}/{num_clients} "
            f"samples={end - start} local train start",
            flush=True,
        )
        local_params, _loss = train_one_client(
            global_model,
            x_norm[start:end].detach().cpu(),
            y_pred[start:end].detach().cpu(),
            round_idx=int(args.rounds),
            client_id=client_id,
            args=args,
            device=device,
        )
        print(f"[round-rank-detail] client {client_id + 1}/{num_clients} topk attack-layer start", flush=True)
        observation = global_topk_attack_layer_observation_from_params_np(
            local_params,
            global_params,
            names=names,
            keep_ratio=float(keep_ratio),
            attack_layer_index=attack_layer_index,
        )
        mask_value = observation["mask"]
        if isinstance(mask_value, torch.Tensor):
            mask_np = mask_value.detach().cpu().numpy().astype(np.bool_, copy=False)
        else:
            mask_np = np.asarray(mask_value, dtype=np.bool_)
        values_np = np.asarray(observation["values"], dtype=np.float32)
        aggregate_attack[mask_np] += values_np * float(weights[client_id])
        print(f"[round-rank-detail] client {client_id + 1}/{num_clients} done", flush=True)
        del local_params, observation, mask_np, values_np
    attack = aggregate_attack.astype(np.float32, copy=False)
    del global_params, aggregate_attack
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return attack


def relative_l2(pred: np.ndarray, observed: np.ndarray) -> float:
    pred_flat = np.asarray(pred, dtype=np.float32).reshape(-1).astype(np.float64)
    obs_flat = np.asarray(observed, dtype=np.float32).reshape(-1).astype(np.float64)
    denom = max(float(np.linalg.norm(obs_flat)), 1e-12)
    return float(np.linalg.norm(pred_flat - obs_flat) / denom)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rank_members_by_aggregate_consistency(
    *,
    args: argparse.Namespace,
    keep_ratio: float,
    source_data: Mapping[str, Any],
    run_paths: Sequence[Path],
) -> Path:
    device = choose_device(args.device)
    model = get_model(model_name=args.model, ds=args.ds, h_dim=args.h_dim, load_path=None, dataparallel=False).to(device)
    if args.global_checkpoint is not None:
        checkpoint = torch.load(resolve_path(args.global_checkpoint), map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    attack_index = int(source_data["attack_layer"]["index"])
    attack_name = str(source_data["attack_layer"]["name"])
    observed_attack = [np.asarray(batch[attack_index], dtype=np.float32) for batch in source_data["grad"]]
    n_batches = min(args.attack_n_batch or len(observed_attack), len(observed_attack))
    ranked = []
    for member_idx, run_path in enumerate(run_paths):
        recon_entries = read_pickle(run_path / "reconstructions.pkl")
        costs = []
        for batch_idx, entry in enumerate(recon_entries[:n_batches]):
            rec_fi = np.asarray(entry.get("rec_fi_attack_order"), dtype=np.float32)
            if rec_fi.size == 0:
                raise ValueError("aggregate consistency ranking requires Direct FIA reconstructions")
            full_n = int(source_data["metadata"]["n_samples"])
            if int(rec_fi.shape[0]) < full_n and not args.allow_partial_aggregate_ranking:
                raise ValueError(
                    "Exact aggregate ranking needs FIA reconstructions for all attacked samples. "
                    f"Got {rec_fi.shape[0]}/{full_n}. Set --n_sample_fi {full_n} or use "
                    "--allow_partial_aggregate_ranking for an approximate score."
                )
            rec_for_score = rec_fi[:full_n]
            print(
                f"[round-rank] member {member_idx + 1}/{len(run_paths)} "
                f"batch {batch_idx + 1}/{n_batches} recon={rec_for_score.shape[0]}",
                flush=True,
            )
            pred_attack = recompress_reconstruction_round_attack_layer(
                args=args,
                global_model=model,
                rec_fi=rec_for_score,
                keep_ratio=keep_ratio,
                attack_layer_name=attack_name,
            )
            costs.append(relative_l2(pred_attack, observed_attack[int(entry["batch"])]))
            del pred_attack
        mean_cost = float(np.mean(costs)) if costs else float("inf")
        ranked.append(
            {
                "member": int(member_idx),
                "score": -mean_cost,
                "mean_relative_l2_to_observed_round_update": mean_cost,
                "batches": int(len(costs)),
                "source_run_path": str(run_path.resolve()),
                "ranking_observation": "aggregate_sparse_round_update_only",
                "partial_aggregate_ranking": bool(args.allow_partial_aggregate_ranking),
            }
        )
    ranked.sort(key=lambda row: float(row["mean_relative_l2_to_observed_round_update"]))
    output_dir = cora_output_dir(args, keep_ratio)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "consistency_member_ranking.csv", ranked)
    best = ranked[0]
    best_recon = read_pickle(Path(best["source_run_path"]) / "reconstructions.pkl")
    write_pickle(output_dir / "selected_reconstructions.pkl", best_recon)
    manifest = {
        "schema_version": 1,
        "method": "CoRA round-update aggregate consistency ranking",
        "keep_ratio": float(keep_ratio),
        "ensemble_size": int(args.ensemble_size),
        "selected_member": int(best["member"]),
        "selected_member_run_path": best["source_run_path"],
        "ranking_observation": "aggregate_sparse_round_update_only",
        "per_client_observations_used": False,
        "exact_aggregate_ranking": not bool(args.allow_partial_aggregate_ranking),
        "member_ranking_file": "consistency_member_ranking.csv",
    }
    write_json(output_dir / "manifest.json", manifest)
    return output_dir


def main() -> None:
    args = parse_args()
    args.round_update_file = resolve_path(args.round_update_file)
    args.global_checkpoint = resolve_path(args.global_checkpoint)
    requested_global_checkpoint = args.global_checkpoint
    args.local_batch_size = args.local_batch_size or args.samples_per_client
    if args.ensemble_size < 1:
        raise ValueError("--ensemble_size must be >= 1")
    for keep_ratio in args.keep_ratios:
        validate_keep_ratio(float(keep_ratio))
    if args.fi_method != "direct":
        raise ValueError("Round-update CoRA supports Direct FIA only")
    args.observed_zero_policy = OBSERVED_ZERO_POLICY
    if args.device == "cpu":
        cpu_threads = max(1, min(4, os.cpu_count() or 1))
        torch.set_num_threads(cpu_threads)
        print(f"[round-cora] cpu torch threads={cpu_threads}", flush=True)

    for keep_ratio in args.keep_ratios:
        args.global_checkpoint = requested_global_checkpoint
        ratio = float(keep_ratio)
        if args.round_update_file is None:
            source_file = collect_sparse_round_update(args, ratio, round_update_path(args, ratio))
        else:
            source_file = args.round_update_file
        source_hash = sha256_file(source_file)
        source_data = normalize_loaded_round_data(args, source_file)
        source_data["metadata"]["keep_ratio"] = ratio
        args.n_samples = int(source_data["metadata"]["n_samples"])
        args.attack_trials = int(source_data["metadata"].get("attack_trials", len(source_data["grad"])))
        checkpoint = source_data["metadata"].get("checkpoint", {})
        old_checkpoint = requested_global_checkpoint
        if isinstance(checkpoint, Mapping) and checkpoint.get("path"):
            args.global_checkpoint = Path(str(checkpoint["path"]))

        run_paths: list[Path] = []
        observation_summaries = []
        for member_idx in range(int(args.ensemble_size)):
            member_seed = int(args.noise_seed) + 1_000_003 * member_idx
            ratio_tag = format_keep_ratio(ratio)
            exp_name = (
                f"{EXPERIMENT_NAME}_roundagg_keep{ratio_tag}_"
                f"n{args.n_samples}_round{args.rounds}_R{args.ensemble_size}"
                f"_seed{args.noise_seed}_m{member_idx:03d}"
            )
            existing_run_path = author_member_run_path(args, exp_name)
            if (
                args.run_attack
                and not bool(getattr(args, "fresh_start", False))
                and (existing_run_path / "reconstructions.pkl").exists()
            ):
                print(
                    f"[round-member] reuse completed member {member_idx + 1}/{args.ensemble_size}: "
                    f"{existing_run_path}",
                    flush=True,
                )
                run_paths.append(existing_run_path)
                observation_summaries.append(
                    {"member": member_idx, "reused_completed_run": str(existing_run_path.resolve())}
                )
                continue
            member_data, observation_rows = materialize_member_update_data(
                source_data,
                keep_ratio=ratio,
                member_idx=member_idx,
                member_seed=member_seed,
                sampler=args.desparse_sampler,
                source_file=source_file,
                source_hash=source_hash,
            )
            observation_summaries.append({"member": member_idx, "observations": observation_rows})
            if args.run_attack:
                run_path = run_author_cpa_with_update_file(
                    args,
                    exp_name=exp_name,
                    update_file=source_file,
                    update_data=member_data,
                    method=(
                        "CoRA round-update: aggregate sparse FedAvg round observation, "
                        "attack-FC randomized desparsification, author CPA, Direct FIA"
                    ),
                    extra_manifest={
                        "cora_variant": "round_update_aggregate_observed_only",
                        "per_client_observations_used": False,
                        "ensemble_member": int(member_idx),
                        "ensemble_size": int(args.ensemble_size),
                        "desparsification_seed": int(member_seed),
                        "desparse_sampler": args.desparse_sampler,
                    },
                    attack_order_policy="aggregate sparse round observation only; no per-client oracle permutation",
                )
                run_paths.append(run_path)
        output_dir = cora_output_dir(args, ratio)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "aggregate_observation_manifest.json",
            {
                "schema_version": 1,
                "source_file": str(source_file.resolve()),
                "source_hash": source_hash,
                "keep_ratio": ratio,
                "ensemble_size": int(args.ensemble_size),
                "noise_seed": int(args.noise_seed),
                "desparse_sampler": args.desparse_sampler,
                "per_client_observations_used": False,
                "observation": "aggregate_sparse_round_update_only",
                "member_observation_summaries": observation_summaries,
            },
        )
        if args.run_attack and not args.skip_consistency_ranking:
            rank_dir = rank_members_by_aggregate_consistency(
                args=args,
                keep_ratio=ratio,
                source_data=source_data,
                run_paths=run_paths,
            )
            print(f"[round-rank] keep={ratio:g} ranking saved: {rank_dir}", flush=True)
        elif args.run_attack:
            print("[round-rank] skipped", flush=True)
        else:
            print(f"[round-cora] artifact ready: {source_file}", flush=True)
        args.global_checkpoint = old_checkpoint


if __name__ == "__main__":
    main()
