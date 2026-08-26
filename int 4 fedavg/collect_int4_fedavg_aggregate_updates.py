#!/usr/bin/env python3
"""Collect a final-round int4 FedAvg aggregate update in CPA-compatible format."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
FEDAVG_SRC = REPO_ROOT / "FedAvg"
if str(FEDAVG_SRC) not in sys.path:
    sys.path.insert(0, str(FEDAVG_SRC))

import collect_fedavg_aggregate_updates as fp32_fedavg  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-client FedAvg where each client quantizes its update to "
            "signed int4 before server-side dequantization and aggregation."
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
    parser.add_argument(
        "--save_final_only",
        action="store_true",
        help="Compatibility flag; this collector always writes only the final aggregate pickle.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for the final int4/dequant aggregate pickle and metadata.",
    )
    parser.add_argument(
        "--export_baseline_update",
        type=Path,
        default=None,
        help="Optional CPA baseline update path, e.g. exp/imagenet/vgg16/updates/nodef/24.pickle.",
    )
    return parser.parse_args()


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
            / "int 4 fedavg"
            / args.ds
            / args.model
            / f"clients{args.num_clients}_rounds{args.rounds}_spc{args.samples_per_client}_seed{args.seed}"
        )
    else:
        args.output_dir = fp32_fedavg.resolve_path(args.output_dir)
    if args.export_baseline_update is not None:
        args.export_baseline_update = fp32_fedavg.resolve_path(args.export_baseline_update)


def quantize_dequantize_update_int4(
    update: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Per-client, per-parameter signed symmetric int4 quantization.

    The server receives dequantized tensors, so the aggregate update below is
    a weighted mean of dequantized client updates, not original FP32 updates.
    """
    qmin, qmax = -7, 7
    dequantized: Dict[str, torch.Tensor] = {}
    per_tensor: List[Dict[str, Any]] = []

    total_values = 0
    total_error_sq = 0.0
    total_original_sq = 0.0
    total_dequant_sq = 0.0
    total_dot = 0.0
    saturation_values = 0.0
    scales: List[float] = []

    for name, tensor in update.items():
        original = tensor.detach().to(dtype=torch.float32)
        max_abs = original.abs().max()
        if float(max_abs.item()) > 0.0:
            scale = max_abs / qmax
        else:
            scale = torch.ones((), dtype=torch.float32)

        codes = torch.round(original / scale).clamp(qmin, qmax).to(torch.int8)
        dq = codes.to(dtype=torch.float32) * scale
        error = dq - original

        numel = int(original.numel())
        error_sq = float(torch.sum(error * error).item())
        original_sq = float(torch.sum(original * original).item())
        dequant_sq = float(torch.sum(dq * dq).item())
        dot = float(torch.sum(original * dq).item())
        saturation_ratio = float((codes.abs() >= qmax).float().mean().item())
        scale_value = float(scale.detach().cpu().item())

        total_values += numel
        total_error_sq += error_sq
        total_original_sq += original_sq
        total_dequant_sq += dequant_sq
        total_dot += dot
        saturation_values += saturation_ratio * numel
        scales.append(scale_value)
        dequantized[name] = dq
        per_tensor.append(
            {
                "name": name,
                "scale": scale_value,
                "mse": error_sq / max(numel, 1),
                "relative_l2": math.sqrt(error_sq) / max(math.sqrt(original_sq), 1e-12),
                "saturation_ratio": saturation_ratio,
                "qmin": qmin,
                "qmax": qmax,
            }
        )

    cosine_denominator = max(math.sqrt(total_original_sq) * math.sqrt(total_dequant_sq), 1e-12)
    stats: Dict[str, Any] = {
        "quant_bits": 4,
        "rounding": "nearest",
        "quantization": "signed_symmetric_int4",
        "scale_scope": "per_client_per_parameter_tensor",
        "zero_point": None,
        "qmin": qmin,
        "qmax": qmax,
        "levels": qmax - qmin + 1,
        "quant_mse": total_error_sq / max(total_values, 1),
        "quant_relative_l2": math.sqrt(total_error_sq) / max(math.sqrt(total_original_sq), 1e-12),
        "quant_cosine_similarity": total_dot / cosine_denominator,
        "quant_scale_mean": float(sum(scales) / len(scales)),
        "quant_scale_min": float(min(scales)),
        "quant_scale_max": float(max(scales)),
        "quant_saturation_ratio": saturation_values / max(total_values, 1),
        "per_tensor": per_tensor,
    }
    return dequantized, stats


def make_int4_manifest(
    args: argparse.Namespace,
    *,
    device: torch.device,
    param_names: Sequence[str],
    client_indices: Sequence[Sequence[int]],
    original_indices: Sequence[torch.Tensor],
    weights: Sequence[float],
) -> Dict[str, Any]:
    manifest = fp32_fedavg.make_manifest(
        args,
        device=device,
        param_names=param_names,
        client_indices=client_indices,
        original_indices=original_indices,
    )
    manifest.update(
        {
            "signal": "int4_fedavg_server_dequantized_aggregate_update",
            "update_definition": (
                "client_delta_w = w_client - w_global_before_round; "
                "client_delta_w is quantized with each client's per-parameter int4 scale; "
                "server dequantizes each client update before weighted aggregation"
            ),
            "quant_bits": 4,
            "rounding": "nearest",
            "quantization": "signed_symmetric_int4",
            "scale_scope": "per_client_per_parameter_tensor",
            "server_aggregation_input": "dequantized_client_updates",
            "saved_rounds": [args.rounds],
            "aggregation_weights": list(weights),
        }
    )
    return manifest


def main() -> None:
    args = parse_args()
    validate_args(args)
    fp32_fedavg.set_seed(args.seed)
    device = fp32_fedavg.choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"int4 FedAvg aggregate collection | ds={args.ds} model={args.model} "
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
    if len(param_names) <= 26:
        raise ValueError("VGG16 attack layer index 26 is outside the parameter list")

    client_sizes = [len(x) for x in xs]
    total_client_samples = float(sum(client_sizes))
    weights = [size / total_client_samples for size in client_sizes]
    manifest = make_int4_manifest(
        args,
        device=device,
        param_names=param_names,
        client_indices=client_indices,
        original_indices=original_indices,
        weights=weights,
    )
    fp32_fedavg.write_json(manifest, args.output_dir / "manifest.json")

    metrics: List[Dict[str, Any]] = []
    final_aggregate_update: Optional[Dict[str, torch.Tensor]] = None
    final_z: Optional[np.ndarray] = None
    final_quant_stats: Optional[List[Dict[str, Any]]] = None

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

        dequantized_client_updates: List[Dict[str, torch.Tensor]] = []
        client_losses: List[float] = []
        client_quant_stats: List[Dict[str, Any]] = []
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
            dequantized_update, quant_stats = quantize_dequantize_update_int4(fp32_update)
            quant_stats["round"] = round_idx
            quant_stats["client_id"] = client_id
            dequantized_client_updates.append(dequantized_update)
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
            final_aggregate_update = {
                name: tensor.detach().cpu().clone() for name, tensor in aggregate_update.items()
            }
            final_quant_stats = client_quant_stats

        del global_params, dequantized_client_updates, aggregate_update
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if final_aggregate_update is None or final_z is None or final_quant_stats is None:
        raise RuntimeError("Final int4 aggregate update was not captured")

    final_pickle = fp32_fedavg.make_cpa_pickle(
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
    final_pickle["meta"][0].update(
        {
            "quant_bits": 4,
            "rounding": "nearest",
            "quantization": "signed_symmetric_int4",
            "scale_scope": "per_client_per_parameter_tensor",
            "server_aggregation_input": "dequantized_client_updates",
            "update_definition": "server aggregate of final int4/dequantized FedAvg round",
        }
    )

    output_pickle = args.output_dir / f"round_{args.rounds:03d}_int4_dequant_aggregate_cpa.pickle"
    fp32_fedavg.write_pickle(final_pickle, output_pickle)
    fp32_fedavg.write_json(metrics, args.output_dir / "metrics.json")
    fp32_fedavg.write_metrics_csv(metrics, args.output_dir / "metrics.csv")
    fp32_fedavg.write_json(final_quant_stats, args.output_dir / "final_round_client_quant_stats.json")

    if args.export_baseline_update is not None:
        fp32_fedavg.write_pickle(final_pickle, args.export_baseline_update)

    print(f"Saved final int4/dequant aggregate CPA pickle: {output_pickle}", flush=True)
    if args.export_baseline_update is not None:
        print(f"Exported baseline-compatible update: {args.export_baseline_update}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
