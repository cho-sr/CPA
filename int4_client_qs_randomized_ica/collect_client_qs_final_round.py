#!/usr/bin/env python3
"""Collect final-round client int4 q,s artifacts for fair FastICA comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
FEDAVG_SRC = REPO_ROOT / "FedAvg"
if str(FEDAVG_SRC) not in sys.path:
    sys.path.insert(0, str(FEDAVG_SRC))

import collect_fedavg_aggregate_updates as fp32_fedavg  # noqa: E402


QMIN = -7
QMAX = 7
ATTACK_LAYER_NAME = "classifier.0.weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run int4 FedAvg and save final-round client q codes and per-client "
            "scales before server aggregation."
        )
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
    parser.add_argument("--private_split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--store_scope",
        type=str,
        default="all",
        choices=["all", "attack_layer"],
        help="Default stores every parameter tensor; attack_layer is a lighter smoke option.",
    )
    parser.add_argument("--fresh_start", action="store_true")
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
            / "int4_client_qs_randomized_ica"
            / args.ds
            / args.model
            / f"clients{args.num_clients}_rounds{args.rounds}_spc{args.samples_per_client}_seed{args.seed}"
        )
    else:
        args.output_dir = resolve_path(args.output_dir)


def safe_rmtree(path: Path, *, allowed_root: Path) -> None:
    root = allowed_root.resolve()
    target = path.resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"refusing to delete outside {root}: {target}")
    if target.exists():
        shutil.rmtree(target)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def write_pickle(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_metrics_csv(metrics: Sequence[Dict[str, Any]], path: Path) -> None:
    if not metrics:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantize_tensor_int4(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    original = tensor.detach().to(dtype=torch.float32)
    max_abs = original.abs().max()
    max_abs_value = float(max_abs.detach().cpu().item())
    if max_abs_value > 0.0:
        scale = max_abs / QMAX
        codes = torch.round(original / scale).clamp(QMIN, QMAX).to(torch.int8)
        dequantized = codes.to(dtype=torch.float32) * scale
        scale_value = float(scale.detach().cpu().item())
    else:
        codes = torch.zeros_like(original, dtype=torch.int8)
        dequantized = torch.zeros_like(original, dtype=torch.float32)
        scale_value = 0.0

    error = dequantized - original
    numel = int(original.numel())
    error_sq = float(torch.sum(error * error).item())
    original_sq = float(torch.sum(original * original).item())
    dequant_sq = float(torch.sum(dequantized * dequantized).item())
    dot = float(torch.sum(original * dequantized).item())
    stats = {
        "scale": scale_value,
        "max_abs": max_abs_value,
        "mse": error_sq / max(numel, 1),
        "relative_l2": (error_sq**0.5) / max(original_sq**0.5, 1e-12),
        "cosine_similarity": dot / max((original_sq**0.5) * (dequant_sq**0.5), 1e-12),
        "saturation_ratio": float((codes.abs() >= QMAX).float().mean().item()),
        "qmin": QMIN,
        "qmax": QMAX,
        "numel": numel,
    }
    return codes, dequantized, stats


def quantize_update_int4(
    update: Dict[str, torch.Tensor],
) -> Tuple[
    "OrderedDict[str, torch.Tensor]",
    "OrderedDict[str, torch.Tensor]",
    Dict[str, Any],
]:
    codes_by_name: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    dequantized_by_name: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    per_tensor: List[Dict[str, Any]] = []
    total_values = 0
    total_error_sq = 0.0
    total_original_sq = 0.0
    total_dequant_sq = 0.0
    total_dot = 0.0
    saturation_values = 0.0
    scales = []

    for name, tensor in update.items():
        codes, dequantized, stats = _quantize_tensor_int4(tensor)
        codes_by_name[name] = codes.detach().cpu()
        dequantized_by_name[name] = dequantized.detach().cpu()
        row = {"name": name, **stats}
        per_tensor.append(row)
        scales.append(float(stats["scale"]))
        numel = int(stats["numel"])
        total_values += numel
        total_error_sq += float(stats["mse"]) * numel
        saturation_values += float(stats["saturation_ratio"]) * numel
        original = tensor.detach().reshape(-1).float()
        dequant_flat = dequantized.detach().reshape(-1).float()
        total_original_sq += float(torch.sum(original * original).item())
        total_dequant_sq += float(torch.sum(dequant_flat * dequant_flat).item())
        total_dot += float(torch.sum(original * dequant_flat).item())

    stats_all = {
        "quant_bits": 4,
        "rounding": "nearest",
        "quantization": "signed_symmetric_int4",
        "scale_scope": "per_client_per_parameter_tensor",
        "zero_point": None,
        "qmin": QMIN,
        "qmax": QMAX,
        "levels": QMAX - QMIN + 1,
        "quant_mse": total_error_sq / max(total_values, 1),
        "quant_relative_l2": (total_error_sq**0.5) / max(total_original_sq**0.5, 1e-12),
        "quant_cosine_similarity": total_dot
        / max((total_original_sq**0.5) * (total_dequant_sq**0.5), 1e-12),
        "quant_scale_mean": float(sum(scales) / len(scales)),
        "quant_scale_min": float(min(scales)),
        "quant_scale_max": float(max(scales)),
        "quant_saturation_ratio": saturation_values / max(total_values, 1),
        "per_tensor": per_tensor,
    }
    return codes_by_name, dequantized_by_name, stats_all


def tensor_key(client_id: int, param_index: int) -> str:
    return f"client{client_id:03d}_param{param_index:02d}"


def make_root_manifest(
    args: argparse.Namespace,
    *,
    device: torch.device,
    param_names: Sequence[str],
    client_indices: Sequence[Sequence[int]],
    original_indices: Sequence[torch.Tensor],
    weights: Sequence[float],
    attack_layer_index: int,
) -> Dict[str, Any]:
    total_samples = args.num_clients * args.samples_per_client
    return {
        "schema_version": 1,
        "experiment": "client_qs_naive_vs_randomized_fastica",
        "stage": "client_qs_collection",
        "signal": "final_round_client_int4_qs_before_server_aggregation",
        "update_definition": "client_delta_w = w_client_after_local_train - w_global_before_final_round",
        "server_aggregation_input": "client_qs_artifact_not_final_aggregate",
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
        "aggregation_weights": list(weights),
        "client_sampling": "all_clients_each_round",
        "private_split": args.private_split,
        "seed": args.seed,
        "device": str(device),
        "param_names": list(param_names),
        "attack_layer_index": attack_layer_index,
        "attack_layer_name": param_names[attack_layer_index],
        "quant_bits": 4,
        "rounding": "nearest",
        "quantization": "signed_symmetric_int4",
        "scale_definition": "scale=max_abs/7; scale=0 when max_abs=0",
        "qmin": QMIN,
        "qmax": QMAX,
        "store_scope": args.store_scope,
        "client_subset_indices": [list(map(int, indices)) for indices in client_indices],
        "client_original_indices": [
            [int(value) for value in tensor.tolist()] for tensor in original_indices
        ],
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.fresh_start:
        safe_rmtree(args.output_dir, allowed_root=REPO_ROOT / "int4_client_qs_randomized_ica")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client_qs_dir = args.output_dir / "client_qs"
    client_qs_dir.mkdir(parents=True, exist_ok=True)

    fp32_fedavg.set_seed(args.seed)
    device = fp32_fedavg.choose_device(args.device)

    print(
        f"client q,s collection | ds={args.ds} model={args.model} "
        f"clients={args.num_clients} rounds={args.rounds} "
        f"samples/client={args.samples_per_client} device={device}",
        flush=True,
    )

    dataset = fp32_fedavg.build_private_dataset(args)
    client_indices = fp32_fedavg.make_client_indices(
        len(dataset),
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        seed=args.seed,
    )
    xs, ys, original_indices = fp32_fedavg.load_client_tensors(
        dataset,
        client_indices,
        num_workers=args.num_workers,
    )
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    original_index_all = torch.cat(original_indices, dim=0)

    model = fp32_fedavg.get_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        dataparallel=False,
    ).to(device)
    if fp32_fedavg.get_model_type(model) != "conv":
        raise ValueError("This collector expects a conv model with return_z=True")
    param_names = [name for name, _parameter in model.named_parameters()]
    if ATTACK_LAYER_NAME not in param_names:
        raise ValueError(f"{ATTACK_LAYER_NAME} not found in model parameters")
    attack_layer_index = int(param_names.index(ATTACK_LAYER_NAME))

    client_sizes = [len(x) for x in xs]
    total_client_samples = float(sum(client_sizes))
    weights = [size / total_client_samples for size in client_sizes]
    root_manifest = make_root_manifest(
        args,
        device=device,
        param_names=param_names,
        client_indices=client_indices,
        original_indices=original_indices,
        weights=weights,
        attack_layer_index=attack_layer_index,
    )
    write_json(root_manifest, args.output_dir / "manifest.json")

    metrics: List[Dict[str, Any]] = []
    final_z: Optional[np.ndarray] = None
    final_global_state: Optional[Dict[str, torch.Tensor]] = None
    final_client_codes: List[OrderedDict[str, torch.Tensor]] = []
    final_client_stats: List[Dict[str, Any]] = []

    for round_idx in range(1, args.rounds + 1):
        round_start = time.time()
        global_params = fp32_fedavg.named_parameters_cpu(model)
        if round_idx == args.rounds:
            final_z = fp32_fedavg.extract_embeddings(
                model,
                x_all,
                device=device,
                batch_size=args.local_batch_size,
            )
            final_global_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        dequantized_client_updates: List[Dict[str, torch.Tensor]] = []
        client_losses: List[float] = []
        client_quant_stats: List[Dict[str, Any]] = []
        round_client_codes: List[OrderedDict[str, torch.Tensor]] = []
        for client_id, (x_client, y_client) in enumerate(zip(xs, ys)):
            local_params, train_loss = fp32_fedavg.train_one_client(
                model,
                x_client,
                y_client,
                round_idx=round_idx,
                client_id=client_id,
                args=args,
                device=device,
            )
            fp32_update = fp32_fedavg.compute_update(local_params, global_params)
            codes, dequantized_update, quant_stats = quantize_update_int4(fp32_update)
            quant_stats["round"] = round_idx
            quant_stats["client_id"] = client_id
            round_client_codes.append(codes)
            dequantized_client_updates.append(dict(dequantized_update))
            client_quant_stats.append(quant_stats)
            client_losses.append(train_loss)
            del local_params, fp32_update

        aggregate_update = fp32_fedavg.aggregate_updates(dequantized_client_updates, weights)
        fp32_fedavg.apply_parameter_update(model, global_params, aggregate_update)

        weighted_train_loss = sum(
            loss * size for loss, size in zip(client_losses, client_sizes)
        ) / sum(client_sizes)
        metric = {
            "round": round_idx,
            "weighted_train_loss": weighted_train_loss,
            "client_train_losses": client_losses,
            "aggregate_update_l2": fp32_fedavg.l2_norm(aggregate_update),
            "quant_mse_mean": sum(s["quant_mse"] for s in client_quant_stats) / len(client_quant_stats),
            "quant_relative_l2_mean": sum(s["quant_relative_l2"] for s in client_quant_stats)
            / len(client_quant_stats),
            "quant_cosine_similarity_mean": sum(s["quant_cosine_similarity"] for s in client_quant_stats)
            / len(client_quant_stats),
            "quant_saturation_ratio_mean": sum(s["quant_saturation_ratio"] for s in client_quant_stats)
            / len(client_quant_stats),
            "round_time": time.time() - round_start,
        }
        metrics.append(metric)
        print(
            "round {round:03d} | train_loss {weighted_train_loss:.4f} | "
            "agg_l2 {aggregate_update_l2:.4f} | q_rel_l2 {quant_relative_l2_mean:.4f} | "
            "time {round_time:.1f}s".format(**metric),
            flush=True,
        )

        if round_idx == args.rounds:
            final_client_codes = round_client_codes
            final_client_stats = client_quant_stats

        del global_params, dequantized_client_updates, aggregate_update
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if final_z is None or final_global_state is None or not final_client_codes:
        raise RuntimeError("final-round client q,s artifact was not captured")

    arrays: Dict[str, np.ndarray] = {}
    clients_meta: List[Dict[str, Any]] = []
    for client_id, (codes_by_name, quant_stats) in enumerate(zip(final_client_codes, final_client_stats)):
        tensors_meta = []
        for param_index, name in enumerate(param_names):
            if args.store_scope == "attack_layer" and param_index != attack_layer_index:
                continue
            key = tensor_key(client_id, param_index)
            codes_np = codes_by_name[name].cpu().numpy().astype(np.int8, copy=False)
            arrays[key] = codes_np
            tensor_stats = dict(quant_stats["per_tensor"][param_index])
            tensors_meta.append(
                {
                    "param_index": param_index,
                    "name": name,
                    "key": key,
                    "shape": list(codes_np.shape),
                    "dtype": "int8",
                    "scale": float(tensor_stats["scale"]),
                    "max_abs": float(tensor_stats["max_abs"]),
                    "qmin": QMIN,
                    "qmax": QMAX,
                    "mse": float(tensor_stats["mse"]),
                    "relative_l2": float(tensor_stats["relative_l2"]),
                    "saturation_ratio": float(tensor_stats["saturation_ratio"]),
                }
            )
        clients_meta.append(
            {
                "client_id": client_id,
                "aggregation_weight": float(weights[client_id]),
                "quant_stats": {
                    key: value for key, value in quant_stats.items() if key != "per_tensor"
                },
                "tensors": tensors_meta,
            }
        )

    codes_path = client_qs_dir / "final_round_client_codes.npz"
    np.savez_compressed(codes_path, **arrays)

    model_state_path = client_qs_dir / "final_round_global_model_before.pt"
    torch.save(final_global_state, model_state_path)

    context_path = client_qs_dir / "final_round_context.pkl"
    write_pickle(
        {
            "x": x_all.numpy(),
            "y": y_all.numpy(),
            "z": final_z,
            "index": original_index_all.numpy(),
            "param_names": param_names,
            "attack_layer": {"index": attack_layer_index, "name": ATTACK_LAYER_NAME},
            "aggregation_weights": weights,
            "source_model_for_z": "global model before final round",
            "source_model_state_path": str(model_state_path.resolve()),
            "client_original_indices": [
                [int(value) for value in tensor.tolist()] for tensor in original_indices
            ],
        },
        context_path,
    )

    quant_meta = {
        "schema_version": 1,
        "artifact_type": "final_round_client_qs",
        "code_file": str(codes_path.resolve()),
        "context_file": str(context_path.resolve()),
        "source_model_state_file": str(model_state_path.resolve()),
        "source": "client q,s before final-round server aggregation",
        "not_source": "final aggregate pickle",
        "quant_bits": 4,
        "rounding": "nearest",
        "qmin": QMIN,
        "qmax": QMAX,
        "scale_definition": "scale=max_abs/7; scale=0 when max_abs=0",
        "bin_policy_for_randomized": "bounded exact bins: q=7 [6.5s,7s], q=-7 [-7s,-6.5s]",
        "param_names": param_names,
        "attack_layer": {"index": attack_layer_index, "name": ATTACK_LAYER_NAME},
        "aggregation_weights": weights,
        "clients": clients_meta,
        "codes_sha256": sha256_file(codes_path),
        "source_model_state_sha256": sha256_file(model_state_path),
        "context_sha256": sha256_file(context_path),
    }
    write_json(quant_meta, client_qs_dir / "final_round_client_quant_meta.json")
    write_json(metrics, args.output_dir / "metrics.json")
    write_metrics_csv(metrics, args.output_dir / "metrics.csv")

    root_manifest["client_qs"] = {
        "code_file": str(codes_path.resolve()),
        "context_file": str(context_path.resolve()),
        "source_model_state_file": str(model_state_path.resolve()),
        "quant_meta_file": str((client_qs_dir / "final_round_client_quant_meta.json").resolve()),
        "codes_sha256": quant_meta["codes_sha256"],
        "source_model_state_sha256": quant_meta["source_model_state_sha256"],
        "context_sha256": quant_meta["context_sha256"],
    }
    write_json(root_manifest, args.output_dir / "manifest.json")

    print(f"Saved client q,s codes: {codes_path}", flush=True)
    print(f"Saved client q,s metadata: {client_qs_dir / 'final_round_client_quant_meta.json'}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
