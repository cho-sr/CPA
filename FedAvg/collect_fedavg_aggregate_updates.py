#!/usr/bin/env python3
"""Collect a final-round FedAvg aggregate update in CPA-compatible format."""

from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SRC = REPO_ROOT / "baseline" / "src"
if str(BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINE_SRC))

from datasets import get_dataloaders  # noqa: E402
from models import get_model  # noqa: E402
from utils import get_model_type  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-client FedAvg and save the final server aggregate update for CPA."
    )
    parser.add_argument("--ds", type=str, default="imagenet", choices=["imagenet"])
    parser.add_argument("--model", type=str, default="vgg16", choices=["vgg16"])
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--samples_per_client", type=int, default=8)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--local_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--h_dim", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument(
        "--private_split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split used as client-private samples. Default matches CPA evaluation data.",
    )
    parser.add_argument(
        "--save_final_only",
        action="store_true",
        help="Compatibility flag; this collector always writes only the final aggregate pickle.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for the final aggregate pickle and metadata.",
    )
    parser.add_argument(
        "--export_baseline_update",
        type=Path,
        default=None,
        help="Optional CPA baseline update path, e.g. exp/imagenet/vgg16/updates/nodef/24.pickle.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_clients < 1:
        raise ValueError("--num_clients must be >= 1")
    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if args.samples_per_client < 1:
        raise ValueError("--samples_per_client must be >= 1")
    if args.local_epochs < 1:
        raise ValueError("--local_epochs must be >= 1")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0")
    if args.local_batch_size is None:
        args.local_batch_size = args.samples_per_client
    if args.local_batch_size < 1 or args.local_batch_size > args.samples_per_client:
        raise ValueError("--local_batch_size must be in [1, samples_per_client]")
    if args.output_dir is None:
        args.output_dir = (
            REPO_ROOT
            / "FedAvg"
            / args.ds
            / args.model
            / f"clients{args.num_clients}_rounds{args.rounds}_spc{args.samples_per_client}_seed{args.seed}"
        )
    else:
        args.output_dir = resolve_path(args.output_dir)
    if args.export_baseline_update is not None:
        args.export_baseline_update = resolve_path(args.export_baseline_update)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: Optional[str]) -> torch.device:
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(requested)


def write_pickle(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_metrics_csv(metrics: Sequence[Dict[str, Any]], path: Path) -> None:
    if not metrics:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(metrics[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)


def build_private_dataset(args: argparse.Namespace):
    dl_train, dl_test = get_dataloaders(
        args.ds,
        batch_size=1,
        augment=False,
        shuffle_train=False,
        shuffle_test=False,
        num_workers=args.num_workers,
        return_indices=True,
    )
    return dl_train.dataset if args.private_split == "train" else dl_test.dataset


def make_client_indices(
    dataset_size: int,
    *,
    num_clients: int,
    samples_per_client: int,
    seed: int,
) -> List[List[int]]:
    total = num_clients * samples_per_client
    if total > dataset_size:
        raise ValueError(f"Need {total} private samples, but split has only {dataset_size}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = torch.randperm(dataset_size, generator=generator).tolist()[:total]
    return [
        selected[start : start + samples_per_client]
        for start in range(0, total, samples_per_client)
    ]


def load_client_tensors(
    dataset,
    client_indices: Sequence[Sequence[int]],
    *,
    num_workers: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    original_indices: List[torch.Tensor] = []
    for indices in client_indices:
        loader = DataLoader(
            Subset(dataset, list(indices)),
            batch_size=len(indices),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
        x, y, idx = next(iter(loader))
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
        original_indices.append(idx.detach().cpu())
    return xs, ys, original_indices


def extract_embeddings(
    model: nn.Module,
    x_cpu: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    was_training = model.training
    model.eval()
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        loader = DataLoader(TensorDataset(x_cpu), batch_size=batch_size, shuffle=False)
        for (x_batch,) in loader:
            _, z = model(x_batch.to(device=device), return_z=True)
            chunks.append(z.detach().cpu())
    if was_training:
        model.train()
    return torch.cat(chunks, dim=0).numpy()


def named_parameters_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone().to(dtype=torch.float32)
        for name, parameter in model.named_parameters()
    }


def train_one_client(
    global_model: nn.Module,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    *,
    round_idx: int,
    client_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], float]:
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator(device="cpu").manual_seed(
        args.seed + round_idx * 1000 + client_id
    )
    loader = DataLoader(
        TensorDataset(x_cpu, y_cpu),
        batch_size=args.local_batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    total_loss = 0.0
    total_seen = 0
    for _epoch in range(args.local_epochs):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device=device, dtype=torch.float32)
            y_batch = y_batch.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = local_model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_n = int(x_batch.size(0))
            total_loss += float(loss.detach().cpu().item()) * batch_n
            total_seen += batch_n

    if total_seen == 0:
        raise RuntimeError(f"Client {client_id} produced no training samples")
    local_params = named_parameters_cpu(local_model)
    del optimizer, local_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return local_params, total_loss / total_seen


def compute_update(
    local_params: Dict[str, torch.Tensor],
    global_params: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        name: local_params[name].sub(global_params[name]).to(dtype=torch.float32)
        for name in global_params
    }


def aggregate_updates(
    client_updates: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float],
) -> Dict[str, torch.Tensor]:
    if not client_updates:
        raise ValueError("No client updates to aggregate")
    if len(client_updates) != len(weights):
        raise ValueError("client_updates and weights must have the same length")
    if abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError(f"Aggregation weights sum to {sum(weights)}")

    aggregated: Dict[str, torch.Tensor] = {}
    names = list(client_updates[0].keys())
    for name in names:
        update = torch.zeros_like(client_updates[0][name], dtype=torch.float32)
        for weight, client_update in zip(weights, client_updates):
            update.add_(client_update[name].to(dtype=torch.float32), alpha=float(weight))
        aggregated[name] = update
    return aggregated


def apply_parameter_update(
    model: nn.Module,
    global_params: Dict[str, torch.Tensor],
    aggregate_update: Dict[str, torch.Tensor],
) -> None:
    next_state = model.state_dict()
    for name, update in aggregate_update.items():
        next_state[name] = global_params[name].add(update).to(dtype=torch.float32)
    model.load_state_dict(next_state, strict=True)


def l2_norm(update: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for tensor in update.values():
        total += float(torch.sum(tensor.float() * tensor.float()).item())
    return float(total ** 0.5)


def make_manifest(
    args: argparse.Namespace,
    *,
    device: torch.device,
    param_names: Sequence[str],
    client_indices: Sequence[Sequence[int]],
    original_indices: Sequence[torch.Tensor],
) -> Dict[str, Any]:
    total_samples = args.num_clients * args.samples_per_client
    return {
        "schema_version": 1,
        "signal": "fedavg_server_aggregate_update",
        "update_definition": "aggregate_delta_w = weighted_mean(client_w_after - global_w_before_round)",
        "ds": args.ds,
        "model": args.model,
        "num_clients": args.num_clients,
        "rounds": args.rounds,
        "saved_rounds": [args.rounds],
        "samples_per_client": args.samples_per_client,
        "n_samples": total_samples,
        "local_epochs": args.local_epochs,
        "local_batch_size": args.local_batch_size,
        "lr": args.lr,
        "optimizer": "SGD",
        "aggregation": "weighted_mean_by_client_sample_count",
        "aggregation_weights": [1.0 / args.num_clients for _ in range(args.num_clients)],
        "client_sampling": "all_clients_each_round",
        "private_split": args.private_split,
        "seed": args.seed,
        "device": str(device),
        "param_names": list(param_names),
        "attack_layer_index": 26,
        "client_subset_indices": [list(map(int, indices)) for indices in client_indices],
        "client_original_indices": [
            [int(value) for value in tensor.tolist()] for tensor in original_indices
        ],
    }


def make_cpa_pickle(
    *,
    args: argparse.Namespace,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    z_all: np.ndarray,
    original_index_all: torch.Tensor,
    aggregate_update: Dict[str, torch.Tensor],
    param_names: Sequence[str],
    weights: Sequence[float],
    metrics: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    total_samples = int(x_all.shape[0])
    grad_list = [aggregate_update[name].detach().cpu().numpy() for name in param_names]
    return {
        "x": [x_all.numpy()],
        "y": [y_all.numpy()],
        "z": [z_all],
        "grad": [grad_list],
        "index": [original_index_all.numpy()],
        "meta": [
            {
                "round": args.rounds,
                "global_round_before": args.rounds - 1,
                "global_round_after": args.rounds,
                "num_clients": args.num_clients,
                "samples_per_client": args.samples_per_client,
                "n_samples": total_samples,
                "local_epochs": args.local_epochs,
                "n_epochs": args.local_epochs,
                "local_batch_size": args.local_batch_size,
                "lr": args.lr,
                "aggregation_weights": list(weights),
                "aggregation": "weighted_mean_by_client_sample_count",
                "update_definition": "server aggregate of final FedAvg round",
                "source_model_for_z": "global model before final round",
                "private_split": args.private_split,
                "final_round_train_loss": metrics[-1]["weighted_train_loss"] if metrics else None,
            }
        ],
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"FedAvg aggregate collection | ds={args.ds} model={args.model} "
        f"clients={args.num_clients} rounds={args.rounds} "
        f"samples/client={args.samples_per_client} device={device}",
        flush=True,
    )

    dataset = build_private_dataset(args)
    client_indices = make_client_indices(
        len(dataset),
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        seed=args.seed,
    )
    xs, ys, original_indices = load_client_tensors(
        dataset,
        client_indices,
        num_workers=args.num_workers,
    )
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    original_index_all = torch.cat(original_indices, dim=0)

    model = get_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        dataparallel=False,
    ).to(device)
    if get_model_type(model) != "conv":
        raise ValueError("This collector expects a conv model with return_z=True")
    param_names = [name for name, _parameter in model.named_parameters()]
    if len(param_names) <= 26:
        raise ValueError("VGG16 attack layer index 26 is outside the parameter list")

    client_sizes = [len(x) for x in xs]
    total_client_samples = float(sum(client_sizes))
    weights = [size / total_client_samples for size in client_sizes]
    manifest = make_manifest(
        args,
        device=device,
        param_names=param_names,
        client_indices=client_indices,
        original_indices=original_indices,
    )
    manifest["aggregation_weights"] = weights
    write_json(manifest, args.output_dir / "manifest.json")

    metrics: List[Dict[str, Any]] = []
    final_aggregate_update: Optional[Dict[str, torch.Tensor]] = None
    final_z: Optional[np.ndarray] = None

    for round_idx in range(1, args.rounds + 1):
        round_start = time.time()
        global_params = named_parameters_cpu(model)

        if round_idx == args.rounds:
            final_z = extract_embeddings(
                model,
                x_all,
                device=device,
                batch_size=args.local_batch_size,
            )

        client_updates: List[Dict[str, torch.Tensor]] = []
        client_losses: List[float] = []
        for client_id, (x_client, y_client) in enumerate(zip(xs, ys)):
            local_params, train_loss = train_one_client(
                model,
                x_client,
                y_client,
                round_idx=round_idx,
                client_id=client_id,
                args=args,
                device=device,
            )
            client_updates.append(compute_update(local_params, global_params))
            client_losses.append(train_loss)
            del local_params

        aggregate_update = aggregate_updates(client_updates, weights)
        apply_parameter_update(model, global_params, aggregate_update)

        weighted_train_loss = sum(
            loss * size for loss, size in zip(client_losses, client_sizes)
        ) / sum(client_sizes)
        metric = {
            "round": round_idx,
            "weighted_train_loss": weighted_train_loss,
            "client_train_losses": client_losses,
            "aggregate_update_l2": l2_norm(aggregate_update),
            "round_time": time.time() - round_start,
        }
        metrics.append(metric)
        print(
            "round {round:03d} | train_loss {weighted_train_loss:.4f} | "
            "agg_l2 {aggregate_update_l2:.4f} | time {round_time:.1f}s".format(**metric),
            flush=True,
        )

        if round_idx == args.rounds:
            final_aggregate_update = {
                name: tensor.detach().cpu().clone() for name, tensor in aggregate_update.items()
            }

        del global_params, client_updates, aggregate_update
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if final_aggregate_update is None or final_z is None:
        raise RuntimeError("Final aggregate update was not captured")

    final_pickle = make_cpa_pickle(
        args=args,
        x_all=x_all,
        y_all=y_all,
        z_all=final_z,
        original_index_all=original_index_all,
        aggregate_update=final_aggregate_update,
        param_names=param_names,
        weights=weights,
        metrics=metrics,
    )
    output_pickle = args.output_dir / f"round_{args.rounds:03d}_aggregate_cpa.pickle"
    write_pickle(final_pickle, output_pickle)
    write_json(metrics, args.output_dir / "metrics.json")
    write_metrics_csv(metrics, args.output_dir / "metrics.csv")

    if args.export_baseline_update is not None:
        write_pickle(final_pickle, args.export_baseline_update)

    print(f"Saved final aggregate CPA pickle: {output_pickle}", flush=True)
    if args.export_baseline_update is not None:
        print(f"Exported baseline-compatible update: {args.export_baseline_update}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
