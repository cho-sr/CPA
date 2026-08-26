#!/usr/bin/env python3
"""Measure FedAvg utility with client-update Global Top-k sparsification.

Each keep ratio gets an independent FedAvg trajectory from the same initial
model and the same fixed client split. Sparsification is applied to each
client update after local training and before server aggregation:

    delta_k = theta_k_local - theta_global
    y_k = S_r(delta_k)
    theta_global <- theta_global + sum_k alpha_k y_k
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SRC = REPO_ROOT / "baseline" / "src"
for path in (REPO_ROOT, BASELINE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from datasets import get_dataloaders  # noqa: E402
from models import get_model  # noqa: E402
from sparsified_update_cpa.sparsification import (  # noqa: E402
    format_keep_ratio,
    sparsify_update_global_topk,
    validate_keep_ratio,
)
from utils import get_model_type  # noqa: E402


METRIC_FIELDS = [
    "round",
    "keep_ratio",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_top1",
    "val_top5",
    "eval_samples",
    "aggregate_update_l2",
    "sparse_mask_fraction_mean",
    "sparse_nonzero_fraction_mean",
    "sparse_relative_l2_mean",
    "sparse_cosine_similarity_to_fp32_mean",
    "sparse_energy_retained_mean",
    "topk_threshold_abs_mean",
    "round_time",
]


def parse_keep_ratio(value: str) -> float:
    keep_ratio = float(value)
    if keep_ratio > 1.0:
        keep_ratio = keep_ratio / 100.0
    validate_keep_ratio(keep_ratio)
    return float(keep_ratio)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ImageNet/VGG16 FedAvg with client-update sparsification and report acc/loss."
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
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--private_split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=0,
        help="0 means evaluate the full split; positive values use a deterministic subset.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Evaluate every N rounds. Final round is always evaluated. Use 0 for final only.",
    )
    parser.add_argument(
        "--keep_ratios",
        type=parse_keep_ratio,
        nargs="+",
        default=[1.0, 0.8, 0.5],
        help="Fractions or percents are accepted, e.g. 1.0 0.8 0.5 or 100 80 50.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Base output directory. Per-keep-ratio subdirectories are created under it.",
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
    if args.lr <= 0.0 or not math.isfinite(args.lr):
        raise ValueError("--lr must be a finite positive value")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0")
    if args.local_batch_size is None:
        args.local_batch_size = args.samples_per_client
    if args.local_batch_size < 1 or args.local_batch_size > args.samples_per_client:
        raise ValueError("--local_batch_size must be in [1, samples_per_client]")
    if args.eval_batch_size < 1:
        raise ValueError("--eval_batch_size must be >= 1")
    if args.eval_max_samples < 0:
        raise ValueError("--eval_max_samples must be >= 0")
    if args.eval_every < 0:
        raise ValueError("--eval_every must be >= 0")
    args.keep_ratios = sorted(set(float(value) for value in args.keep_ratios), reverse=True)
    for keep_ratio in args.keep_ratios:
        validate_keep_ratio(keep_ratio)
    if args.output_dir is None:
        eval_tag = "all" if args.eval_max_samples == 0 else str(args.eval_max_samples)
        args.output_dir = (
            REPO_ROOT
            / "sparsified_fedavg_utility"
            / "outputs"
            / args.ds
            / args.model
            / (
                f"clients{args.num_clients}_rounds{args.rounds}"
                f"_spc{args.samples_per_client}_local{args.local_epochs}"
                f"_private{args.private_split}_eval{args.eval_split}{eval_tag}"
                f"_seed{args.seed}"
            )
        )
    else:
        args.output_dir = resolve_path(args.output_dir)


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


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_metrics_csv(metrics: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)


def build_dataset(args: argparse.Namespace, split: str, *, return_indices: bool):
    dl_train, dl_test = get_dataloaders(
        args.ds,
        batch_size=1,
        augment=False,
        shuffle_train=False,
        shuffle_test=False,
        num_workers=args.num_workers,
        return_indices=return_indices,
    )
    return dl_train.dataset if split == "train" else dl_test.dataset


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


def make_eval_loader(args: argparse.Namespace, dataset, *, device: torch.device) -> Tuple[DataLoader, int]:
    eval_dataset = dataset
    if args.eval_max_samples > 0 and args.eval_max_samples < len(dataset):
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 100_000)
        indices = torch.randperm(len(dataset), generator=generator).tolist()[: args.eval_max_samples]
        eval_dataset = Subset(dataset, indices)
    loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return loader, len(eval_dataset)


def named_parameters_cpu(model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict(
        (
            name,
            parameter.detach().cpu().clone().to(dtype=torch.float32),
        )
        for name, parameter in model.named_parameters()
    )


def train_one_client(
    global_model: nn.Module,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    *,
    round_idx: int,
    client_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple["OrderedDict[str, torch.Tensor]", float]:
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

            batch_size = int(x_batch.size(0))
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_seen += batch_size

    if total_seen == 0:
        raise RuntimeError(f"Client {client_id} produced no training samples")
    local_params = named_parameters_cpu(local_model)
    del optimizer, local_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return local_params, total_loss / total_seen


def compute_update(
    local_params: "OrderedDict[str, torch.Tensor]",
    global_params: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict(
        (name, local_params[name].sub(global_params[name]).to(dtype=torch.float32))
        for name in global_params
    )


def zero_update_like(
    global_params: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict(
        (name, torch.zeros_like(tensor, dtype=torch.float32))
        for name, tensor in global_params.items()
    )


def add_weighted_update_(
    accumulator: "OrderedDict[str, torch.Tensor]",
    update: "OrderedDict[str, torch.Tensor]",
    weight: float,
) -> None:
    for name in accumulator:
        accumulator[name].add_(update[name].to(dtype=torch.float32), alpha=float(weight))


def apply_parameter_update(
    model: nn.Module,
    global_params: "OrderedDict[str, torch.Tensor]",
    aggregate_update: "OrderedDict[str, torch.Tensor]",
) -> None:
    next_state = model.state_dict()
    for name, update in aggregate_update.items():
        next_value = global_params[name].add(update).to(dtype=next_state[name].dtype)
        next_state[name] = next_value.to(device=next_state[name].device)
    model.load_state_dict(next_state, strict=True)


def l2_norm(update: "OrderedDict[str, torch.Tensor]") -> float:
    total = 0.0
    for tensor in update.values():
        total += float(torch.sum(tensor.float() * tensor.float()).item())
    return float(total ** 0.5)


def dense_identity_stats(update: "OrderedDict[str, torch.Tensor]", keep_ratio: float) -> Dict[str, float]:
    total_elements = sum(int(tensor.numel()) for tensor in update.values())
    stored_nonzero = sum(int((tensor != 0).sum().item()) for tensor in update.values())
    return {
        "keep_ratio": float(keep_ratio),
        "total_elements": float(total_elements),
        "mask_kept_elements": float(total_elements),
        "stored_nonzero_elements": float(stored_nonzero),
        "sparse_mask_fraction": 1.0,
        "sparse_nonzero_fraction": stored_nonzero / total_elements if total_elements else 0.0,
        "sparse_relative_l2": 0.0,
        "sparse_cosine_similarity_to_fp32": 1.0,
        "sparse_energy_retained": 1.0,
        "topk_threshold_abs": 0.0,
    }


def sparsify_client_update(
    update: "OrderedDict[str, torch.Tensor]",
    *,
    keep_ratio: float,
) -> Tuple["OrderedDict[str, torch.Tensor]", Dict[str, Any]]:
    if keep_ratio >= 1.0:
        return update, dense_identity_stats(update, keep_ratio)
    sparse_update, _packed_masks, stats = sparsify_update_global_topk(
        update,
        keep_ratio=keep_ratio,
        pack_masks=False,
    )
    return sparse_update, stats


def mean_stat(stats: Sequence[Dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in stats if row.get(key) is not None]
    return float(sum(values) / len(values)) if values else 0.0


def aggregate_sparse_stats(stats: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "sparse_mask_fraction_mean": mean_stat(stats, "sparse_mask_fraction"),
        "sparse_nonzero_fraction_mean": mean_stat(stats, "sparse_nonzero_fraction"),
        "sparse_relative_l2_mean": mean_stat(stats, "sparse_relative_l2"),
        "sparse_cosine_similarity_to_fp32_mean": mean_stat(
            stats,
            "sparse_cosine_similarity_to_fp32",
        ),
        "sparse_energy_retained_mean": mean_stat(stats, "sparse_energy_retained"),
        "topk_threshold_abs_mean": mean_stat(stats, "topk_threshold_abs"),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_seen = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch[:2]
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = int(inputs.size(0))
            maxk = min(5, int(outputs.size(1)))
            _, pred = outputs.topk(maxk, dim=1)
            correct = pred.eq(targets.view(-1, 1).expand_as(pred))
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_top1 += int(correct[:, :1].sum().item())
            total_top5 += int(correct.any(dim=1).sum().item())
            total_seen += batch_size
    if total_seen == 0:
        raise ValueError("Evaluation loader produced no samples")
    return (
        total_loss / total_seen,
        total_top1 / total_seen,
        total_top5 / total_seen,
    )


def should_evaluate(round_idx: int, args: argparse.Namespace) -> bool:
    if round_idx == args.rounds:
        return True
    return args.eval_every > 0 and round_idx % args.eval_every == 0


def make_run_manifest(
    args: argparse.Namespace,
    *,
    keep_ratio: float,
    device: torch.device,
    param_names: Sequence[str],
    client_indices: Sequence[Sequence[int]],
    original_indices: Sequence[torch.Tensor],
    eval_samples: int,
) -> Dict[str, Any]:
    total_samples = args.num_clients * args.samples_per_client
    return {
        "schema_version": 1,
        "signal": "sparsified_fedavg_utility",
        "update_definition": "delta_obs_k = S_r(theta_k_local - theta_global_before_round)",
        "sparsification_scope": "full_client_update_before_server_aggregation",
        "sparsification_policy": "global_topk",
        "keep_ratio": float(keep_ratio),
        "ds": args.ds,
        "model": args.model,
        "num_clients": args.num_clients,
        "rounds": args.rounds,
        "samples_per_client": args.samples_per_client,
        "n_samples": total_samples,
        "local_epochs": args.local_epochs,
        "local_batch_size": args.local_batch_size,
        "lr": args.lr,
        "optimizer": "SGD",
        "aggregation": "weighted_mean_by_client_sample_count",
        "client_sampling": "all_clients_each_round",
        "private_split": args.private_split,
        "eval_split": args.eval_split,
        "eval_samples": eval_samples,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "device": str(device),
        "param_names": list(param_names),
        "client_subset_indices": [list(map(int, indices)) for indices in client_indices],
        "client_original_indices": [
            [int(value) for value in tensor.tolist()] for tensor in original_indices
        ],
    }


def write_summary(metrics: Sequence[Dict[str, Any]], output_dir: Path, *, keep_ratio: float) -> None:
    evaluated = [
        metric
        for metric in metrics
        if not math.isnan(float(metric["val_accuracy"]))
    ]
    if not evaluated:
        summary = {
            "keep_ratio": float(keep_ratio),
            "final_accuracy": None,
            "best_accuracy": None,
            "best_round": None,
            "final_val_loss": None,
            "final_train_loss": metrics[-1]["train_loss"] if metrics else None,
        }
    else:
        final_metric = evaluated[-1]
        best_metric = max(evaluated, key=lambda metric: float(metric["val_accuracy"]))
        summary = {
            "keep_ratio": float(keep_ratio),
            "final_accuracy": final_metric["val_accuracy"],
            "final_top5": final_metric["val_top5"],
            "best_accuracy": best_metric["val_accuracy"],
            "best_top5": best_metric["val_top5"],
            "best_round": best_metric["round"],
            "final_val_loss": final_metric["val_loss"],
            "final_train_loss": metrics[-1]["train_loss"] if metrics else None,
            "average_sparse_relative_l2": mean_stat(metrics, "sparse_relative_l2_mean"),
            "average_sparse_cosine_similarity_to_fp32": mean_stat(
                metrics,
                "sparse_cosine_similarity_to_fp32_mean",
            ),
            "average_sparse_energy_retained": mean_stat(metrics, "sparse_energy_retained_mean"),
        }
    write_json(summary, output_dir / "summary.json")


def run_keep_ratio(
    args: argparse.Namespace,
    *,
    keep_ratio: float,
    device: torch.device,
    xs: Sequence[torch.Tensor],
    ys: Sequence[torch.Tensor],
    client_indices: Sequence[Sequence[int]],
    original_indices: Sequence[torch.Tensor],
    eval_loader: DataLoader,
    eval_samples: int,
) -> Dict[str, Any]:
    set_seed(args.seed)
    keep_dir = args.output_dir / f"keep_{format_keep_ratio(keep_ratio)}"
    keep_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        dataparallel=False,
    ).to(device)
    if get_model_type(model) != "conv":
        raise ValueError("This runner expects a conv model")

    param_names = [name for name, _parameter in model.named_parameters()]
    client_sizes = [len(x) for x in xs]
    total_client_samples = float(sum(client_sizes))
    weights = [size / total_client_samples for size in client_sizes]

    manifest = make_run_manifest(
        args,
        keep_ratio=keep_ratio,
        device=device,
        param_names=param_names,
        client_indices=client_indices,
        original_indices=original_indices,
        eval_samples=eval_samples,
    )
    manifest["aggregation_weights"] = weights
    write_json(manifest, keep_dir / "manifest.json")

    criterion = nn.CrossEntropyLoss()
    metrics: List[Dict[str, Any]] = []

    print(
        f"[sparse-fedavg] keep={keep_ratio:.6g} start | "
        f"rounds={args.rounds} clients={args.num_clients} device={device}",
        flush=True,
    )

    for round_idx in range(1, args.rounds + 1):
        round_start = time.time()
        global_params = named_parameters_cpu(model)
        aggregate_update = zero_update_like(global_params)

        client_losses: List[float] = []
        sparse_stats_rows: List[Dict[str, Any]] = []
        for client_id, (x_client, y_client, weight) in enumerate(zip(xs, ys, weights)):
            local_params, train_loss = train_one_client(
                model,
                x_client,
                y_client,
                round_idx=round_idx,
                client_id=client_id,
                args=args,
                device=device,
            )
            dense_update = compute_update(local_params, global_params)
            sparse_update, sparse_stats = sparsify_client_update(
                dense_update,
                keep_ratio=keep_ratio,
            )
            add_weighted_update_(aggregate_update, sparse_update, weight)
            client_losses.append(train_loss)
            sparse_stats_rows.append(sparse_stats)

            del local_params, dense_update, sparse_update
            if device.type == "cuda":
                torch.cuda.empty_cache()

        apply_parameter_update(model, global_params, aggregate_update)
        train_loss = sum(
            loss * size for loss, size in zip(client_losses, client_sizes)
        ) / sum(client_sizes)

        if should_evaluate(round_idx, args):
            val_loss, val_top1, val_top5 = evaluate(model, eval_loader, criterion, device)
        else:
            val_loss, val_top1, val_top5 = float("nan"), float("nan"), float("nan")

        sparse_summary = aggregate_sparse_stats(sparse_stats_rows)
        metric = {
            "round": round_idx,
            "keep_ratio": float(keep_ratio),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_top1,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "eval_samples": eval_samples if should_evaluate(round_idx, args) else 0,
            "aggregate_update_l2": l2_norm(aggregate_update),
            **sparse_summary,
            "round_time": time.time() - round_start,
        }
        metrics.append(metric)
        write_json(metrics, keep_dir / "metrics.json")
        write_metrics_csv(metrics, keep_dir / "metrics.csv")
        write_summary(metrics, keep_dir, keep_ratio=keep_ratio)

        val_text = (
            f"val_loss {val_loss:.4f} | val_acc {val_top1:.4f} | val_top5 {val_top5:.4f}"
            if not math.isnan(val_top1)
            else "val skipped"
        )
        print(
            "keep {keep:.6g} round {round:03d} | train_loss {train:.4f} | "
            "{val} | sparse_l2 {sparse_l2:.4f} | time {time:.1f}s".format(
                keep=keep_ratio,
                round=round_idx,
                train=train_loss,
                val=val_text,
                sparse_l2=sparse_summary["sparse_relative_l2_mean"],
                time=metric["round_time"],
            ),
            flush=True,
        )

        del global_params, aggregate_update
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_summary(metrics, keep_dir, keep_ratio=keep_ratio)
    return {
        "keep_ratio": float(keep_ratio),
        "output_dir": str(keep_dir),
        "metrics_csv": str(keep_dir / "metrics.csv"),
        "summary_json": str(keep_dir / "summary.json"),
    }


def write_comparison_summary(run_summaries: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for run in run_summaries:
        summary_path = Path(run["summary_json"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "keep_ratio": summary["keep_ratio"],
                "final_accuracy": summary.get("final_accuracy"),
                "final_top5": summary.get("final_top5"),
                "best_accuracy": summary.get("best_accuracy"),
                "best_top5": summary.get("best_top5"),
                "best_round": summary.get("best_round"),
                "final_val_loss": summary.get("final_val_loss"),
                "final_train_loss": summary.get("final_train_loss"),
                "average_sparse_relative_l2": summary.get("average_sparse_relative_l2"),
                "average_sparse_cosine_similarity_to_fp32": summary.get(
                    "average_sparse_cosine_similarity_to_fp32"
                ),
            }
        )

    fields = list(rows[0].keys()) if rows else []
    with (output_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(rows, output_dir / "comparison_summary.json")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[sparse-fedavg] ds={args.ds} model={args.model} "
        f"private={args.private_split} eval={args.eval_split} "
        f"keep_ratios={' '.join(format_keep_ratio(r) for r in args.keep_ratios)}",
        flush=True,
    )

    private_dataset = build_dataset(args, args.private_split, return_indices=True)
    client_indices = make_client_indices(
        len(private_dataset),
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        seed=args.seed,
    )
    xs, ys, original_indices = load_client_tensors(
        private_dataset,
        client_indices,
        num_workers=args.num_workers,
    )

    eval_dataset = build_dataset(args, args.eval_split, return_indices=False)
    eval_loader, eval_samples = make_eval_loader(args, eval_dataset, device=device)

    top_manifest = {
        "schema_version": 1,
        "runner": "sparsified_fedavg_utility/run_sparse_fedavg_utility.py",
        "ds": args.ds,
        "model": args.model,
        "keep_ratios": args.keep_ratios,
        "output_dir": str(args.output_dir),
        "private_split": args.private_split,
        "eval_split": args.eval_split,
        "eval_samples": eval_samples,
        "num_clients": args.num_clients,
        "rounds": args.rounds,
        "samples_per_client": args.samples_per_client,
        "local_epochs": args.local_epochs,
        "local_batch_size": args.local_batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
    }
    write_json(top_manifest, args.output_dir / "manifest.json")

    run_summaries = []
    for keep_ratio in args.keep_ratios:
        run_summaries.append(
            run_keep_ratio(
                args,
                keep_ratio=keep_ratio,
                device=device,
                xs=xs,
                ys=ys,
                client_indices=client_indices,
                original_indices=original_indices,
                eval_loader=eval_loader,
                eval_samples=eval_samples,
            )
        )

    write_comparison_summary(run_summaries, args.output_dir)
    print(f"[sparse-fedavg] wrote comparison: {args.output_dir / 'comparison_summary.csv'}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
