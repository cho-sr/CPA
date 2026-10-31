#!/usr/bin/env python3
"""Collect a real FedAvg update observation with random mixed 4/8-bit clients.

Each client trains locally from the current global model.  Immediately after
the local update is computed, this collector samples either 4 or 8 bits,
quantizes that client update, dequantizes it on the server side, and aggregates
the dequantized client updates.  Therefore the whole FL trajectory, including
the final observed update, is produced under the mixed-bit communication rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
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


ATTACK_LAYER_NAME = "classifier.0.weight"
SUPPORTED_BITS = (4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--quant_bits",
        type=int,
        default=None,
        choices=SUPPORTED_BITS,
        help="Fixed bit width for all client updates. Overrides --quant_bits_choices.",
    )
    parser.add_argument("--quant_bits_choices", type=int, nargs="+", default=[4, 8], choices=SUPPORTED_BITS)
    parser.add_argument(
        "--bit_selection_seed",
        type=int,
        default=1234,
        help="Seed for selecting 4bit/8bit client updates.",
    )
    parser.add_argument(
        "--bit_selection_granularity",
        type=str,
        default="client_round",
        choices=["client_round", "client", "round"],
        help="Whether bits are sampled per client update, fixed per client, or fixed per round.",
    )
    parser.add_argument(
        "--store_scope",
        type=str,
        default="attack_layer",
        choices=["none", "attack_layer", "all"],
        help="Optional final-round client q/s storage scope.",
    )
    parser.add_argument(
        "--observation_scope",
        type=str,
        default="aggregate",
        choices=["aggregate", "single_client"],
        help="Attack observation to save: final server aggregate or one final-round client update.",
    )
    parser.add_argument(
        "--observed_client_id",
        type=int,
        default=0,
        help="Client id to observe when --observation_scope single_client.",
    )
    parser.add_argument("--fresh_start", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--export_baseline_update",
        type=Path,
        default=None,
        help="Optional CPA baseline update path, e.g. exp/imagenet/vgg16/updates/nodef/24.pickle.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def bit_tag(choices: Sequence[int]) -> str:
    return "_".join(str(value) for value in sorted(set(int(v) for v in choices)))


def unique_quant_bits(args: argparse.Namespace) -> List[int]:
    return sorted(set(int(value) for value in args.quant_bits_choices))


def fixed_quant_bits(args: argparse.Namespace) -> Optional[int]:
    choices = unique_quant_bits(args)
    return choices[0] if len(choices) == 1 else None


def quant_bits_metadata(args: argparse.Namespace) -> int | str:
    fixed = fixed_quant_bits(args)
    return int(fixed) if fixed is not None else "mixed"


def quant_tag(args: argparse.Namespace) -> str:
    fixed = fixed_quant_bits(args)
    return f"int{fixed}" if fixed is not None else f"mixed{bit_tag(args.quant_bits_choices)}"


def quantization_name(args: argparse.Namespace) -> str:
    fixed = fixed_quant_bits(args)
    if fixed is not None:
        return f"signed_symmetric_int{fixed}"
    return "random_mixed_signed_symmetric_int4_int8"


def bit_selection_policy(args: argparse.Namespace) -> str:
    if fixed_quant_bits(args) is not None:
        return "fixed_bit_width"
    return f"random_choice_per_{args.bit_selection_granularity}"


def validate_args(args: argparse.Namespace) -> None:
    if args.quant_bits is not None:
        args.quant_bits_choices = [int(args.quant_bits)]
    else:
        args.quant_bits_choices = [int(value) for value in args.quant_bits_choices]
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
    if not args.quant_bits_choices:
        raise ValueError("--quant_bits_choices must not be empty")
    if any(int(value) not in SUPPORTED_BITS for value in args.quant_bits_choices):
        raise ValueError("--quant_bits_choices supports only 4 and 8")
    if args.observed_client_id < 0 or args.observed_client_id >= args.num_clients:
        raise ValueError("--observed_client_id must be in [0, --num_clients)")
    if args.output_dir is None:
        bit_schedule_tag = (
            ""
            if fixed_quant_bits(args) is not None
            else f"_{args.bit_selection_granularity}_bitseed{args.bit_selection_seed}"
        )
        observation_tag = (
            ""
            if args.observation_scope == "aggregate"
            else f"_obsclient{int(args.observed_client_id)}"
        )
        args.output_dir = (
            REPO_ROOT
            / "mixed_bit_fedavg"
            / args.ds
            / args.model
            / (
                f"clients{args.num_clients}_rounds{args.rounds}_spc{args.samples_per_client}"
                f"_seed{args.seed}_{quant_tag(args)}"
                f"{bit_schedule_tag}"
                f"{observation_tag}"
            )
        )
    else:
        args.output_dir = resolve_path(args.output_dir)
    if args.export_baseline_update is not None:
        args.export_baseline_update = resolve_path(args.export_baseline_update)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_tensor_sequence(values: Sequence[Any]) -> str:
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            digest.update(f"sequence:{len(value)}".encode("ascii"))
            digest.update(b"\0")
            for item in value:
                update(item)
            return
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
        digest.update(b"\0")

    update(values)
    return digest.hexdigest()


def checkpoint_manifest(checkpoint_path: Optional[Path]) -> Dict[str, Optional[str]]:
    if checkpoint_path is None:
        return {"path": None, "sha256": None}
    path = checkpoint_path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def parameter_schema(model: torch.nn.Module) -> List[Dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in model.named_parameters()
    ]


def qrange(bits: int) -> Tuple[int, int]:
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"unsupported quantization bits: {bits}")
    qmax = (2 ** (bits - 1)) - 1
    return -qmax, qmax


def quantize_tensor_symmetric(
    tensor: torch.Tensor,
    *,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    qmin, qmax = qrange(bits)
    original = tensor.detach().to(dtype=torch.float32)
    max_abs = original.abs().max()
    max_abs_value = float(max_abs.detach().cpu().item())
    if max_abs_value > 0.0:
        scale = max_abs / qmax
        codes = torch.round(original / scale).clamp(qmin, qmax).to(torch.int8)
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
        "quant_bits": int(bits),
        "scale": scale_value,
        "max_abs": max_abs_value,
        "mse": error_sq / max(numel, 1),
        "relative_l2": math.sqrt(error_sq) / max(math.sqrt(original_sq), 1e-12),
        "cosine_similarity": dot / max(math.sqrt(original_sq) * math.sqrt(dequant_sq), 1e-12),
        "saturation_ratio": float((codes.abs() >= qmax).float().mean().item()) if numel else 0.0,
        "qmin": qmin,
        "qmax": qmax,
        "numel": numel,
    }
    return codes, dequantized, stats


def quantize_update_symmetric(
    update: Dict[str, torch.Tensor],
    *,
    bits: int,
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
    scales: List[float] = []

    for name, tensor in update.items():
        codes, dequantized, stats = quantize_tensor_symmetric(tensor, bits=bits)
        codes_by_name[name] = codes.detach().cpu()
        dequantized_by_name[name] = dequantized.detach().cpu()
        row = {"name": name, **stats}
        per_tensor.append(row)
        numel = int(stats["numel"])
        total_values += numel
        total_error_sq += float(stats["mse"]) * numel
        saturation_values += float(stats["saturation_ratio"]) * numel
        scales.append(float(stats["scale"]))

        original = tensor.detach().reshape(-1).float()
        dequant_flat = dequantized.detach().reshape(-1).float()
        total_original_sq += float(torch.sum(original * original).item())
        total_dequant_sq += float(torch.sum(dequant_flat * dequant_flat).item())
        total_dot += float(torch.sum(original * dequant_flat).item())

    qmin, qmax = qrange(bits)
    stats_all = {
        "quant_bits": int(bits),
        "rounding": "nearest",
        "quantization": f"signed_symmetric_int{int(bits)}",
        "scale_scope": "per_client_per_parameter_tensor",
        "zero_point": None,
        "qmin": qmin,
        "qmax": qmax,
        "levels": qmax - qmin + 1,
        "quant_mse": total_error_sq / max(total_values, 1),
        "quant_relative_l2": math.sqrt(total_error_sq) / max(math.sqrt(total_original_sq), 1e-12),
        "quant_cosine_similarity": total_dot
        / max(math.sqrt(total_original_sq) * math.sqrt(total_dequant_sq), 1e-12),
        "quant_scale_mean": float(sum(scales) / len(scales)),
        "quant_scale_min": float(min(scales)),
        "quant_scale_max": float(max(scales)),
        "quant_saturation_ratio": saturation_values / max(total_values, 1),
        "per_tensor": per_tensor,
    }
    return codes_by_name, dequantized_by_name, stats_all


def make_bit_schedule(args: argparse.Namespace) -> Tuple[List[Dict[str, int]], Dict[Tuple[int, int], int]]:
    choices = [int(value) for value in args.quant_bits_choices]
    rng = random.Random(int(args.bit_selection_seed))
    rows: List[Dict[str, int]] = []

    client_bits: Optional[List[int]] = None
    round_bits: Optional[Dict[int, int]] = None
    if args.bit_selection_granularity == "client":
        client_bits = [rng.choice(choices) for _client_id in range(args.num_clients)]
    elif args.bit_selection_granularity == "round":
        round_bits = {round_idx: rng.choice(choices) for round_idx in range(1, args.rounds + 1)}

    schedule_by_pair: Dict[Tuple[int, int], int] = {}
    for round_idx in range(1, args.rounds + 1):
        for client_id in range(args.num_clients):
            if client_bits is not None:
                bits = client_bits[client_id]
            elif round_bits is not None:
                bits = round_bits[round_idx]
            else:
                bits = rng.choice(choices)
            rows.append({"round": round_idx, "client_id": client_id, "quant_bits": int(bits)})
            schedule_by_pair[(round_idx, client_id)] = int(bits)
    return rows, schedule_by_pair


def count_bits(values: Sequence[int]) -> Dict[str, int]:
    return {str(bit): int(sum(1 for value in values if int(value) == bit)) for bit in sorted(set(values))}


def observation_signal(args: argparse.Namespace) -> str:
    tag = quant_tag(args)
    if args.observation_scope == "single_client":
        return f"{tag}_fedavg_single_client_dequantized_update"
    return f"{tag}_fedavg_server_dequantized_aggregate_update"


def observation_update_definition(args: argparse.Namespace) -> str:
    fixed = fixed_quant_bits(args)
    quant_desc = (
        f"quantized to signed symmetric int{fixed}"
        if fixed is not None
        else "randomly quantized to signed symmetric int4 or int8"
    )
    if args.observation_scope == "single_client":
        return (
            "client_delta_w = w_client - w_global_before_round; observed client update is "
            f"{quant_desc}, then dequantized before attack"
        )
    return (
        "client_delta_w = w_client - w_global_before_round; each client update is "
        f"{quant_desc} before server dequantization and weighted aggregation"
    )


def clone_update_cpu(update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in update.items()}


def resolve_attack_layer_index(param_names: Sequence[str]) -> int:
    if ATTACK_LAYER_NAME in param_names:
        return int(list(param_names).index(ATTACK_LAYER_NAME))
    if len(param_names) <= 26:
        raise ValueError("VGG16 attack layer index 26 is outside the parameter list")
    return 26


def tensor_key(client_id: int, param_index: int) -> str:
    return f"client{client_id:03d}_param{param_index:02d}"


def make_manifest(
    args: argparse.Namespace,
    *,
    device: torch.device,
    param_names: Sequence[str],
    client_indices: Sequence[Sequence[int]],
    original_indices: Sequence[torch.Tensor],
    weights: Sequence[float],
    attack_layer_index: int,
    bit_schedule: Sequence[Dict[str, int]],
) -> Dict[str, Any]:
    manifest = fp32_fedavg.make_manifest(
        args,
        device=device,
        param_names=param_names,
        client_indices=client_indices,
        original_indices=original_indices,
    )
    total_fedavg_samples = int(args.num_clients) * int(args.samples_per_client)
    observed_client_id = int(args.observed_client_id)
    observed_sample_count = (
        int(args.samples_per_client)
        if args.observation_scope == "single_client"
        else total_fedavg_samples
    )
    fixed_bits = fixed_quant_bits(args)
    qbits = quant_bits_metadata(args)
    manifest.update(
        {
            "signal": observation_signal(args),
            "update_definition": observation_update_definition(args),
            "observation_scope": args.observation_scope,
            "attacker_observation": (
                f"one_client_{quant_tag(args)}_dequantized_update_from_fedavg_round"
                if args.observation_scope == "single_client"
                else f"final_round_{quant_tag(args)}_dequantized_fedavg_aggregate_update"
            ),
            "attacker_observes_round_aggregate": args.observation_scope == "aggregate",
            "observed_client_id": observed_client_id if args.observation_scope == "single_client" else None,
            "observe_round": int(args.rounds),
            "n_samples": observed_sample_count,
            "observed_sample_count": observed_sample_count,
            "total_fedavg_samples": total_fedavg_samples,
            "quant_bits": qbits,
            "fixed_quant_bits": fixed_bits,
            "mixed_quant_bits": None if fixed_bits is not None else unique_quant_bits(args),
            "quant_bits_choices": [int(v) for v in args.quant_bits_choices],
            "rounding": "nearest",
            "quantization": quantization_name(args),
            "scale_scope": "per_client_per_parameter_tensor",
            "server_aggregation_input": "dequantized_client_updates",
            "client_update_quantized_before_server_aggregation": True,
            "aggregate_requantized": False,
            "bit_selection_policy": bit_selection_policy(args),
            "bit_selection_random": fixed_bits is None,
            "bit_selection_seed": int(args.bit_selection_seed),
            "bit_selection_granularity": args.bit_selection_granularity,
            "bit_selection_schedule": list(bit_schedule),
            "bit_selection_counts": count_bits([row["quant_bits"] for row in bit_schedule]),
            "aggregation_weights": list(weights),
            "fedavg_aggregation_weights": list(weights),
            "attack_layer_index": int(attack_layer_index),
            "attack_layer_name": param_names[attack_layer_index],
            "store_scope": args.store_scope,
        }
    )
    if args.observation_scope == "single_client":
        manifest["observed_client_subset_indices"] = [
            int(value) for value in client_indices[observed_client_id]
        ]
        manifest["observed_client_original_indices"] = [
            int(value) for value in original_indices[observed_client_id].tolist()
        ]
    return manifest


def save_client_qs_artifacts(
    *,
    args: argparse.Namespace,
    client_qs_dir: Path,
    final_client_codes: Sequence[OrderedDict[str, torch.Tensor]],
    final_client_stats: Sequence[Dict[str, Any]],
    final_global_state: Dict[str, torch.Tensor],
    final_z: np.ndarray,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    original_index_all: torch.Tensor,
    original_indices: Sequence[torch.Tensor],
    param_names: Sequence[str],
    attack_layer_index: int,
    weights: Sequence[float],
    final_round_bits: Sequence[int],
) -> Dict[str, Any]:
    client_qs_dir.mkdir(parents=True, exist_ok=True)
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
                    "param_index": int(param_index),
                    "name": name,
                    "key": key,
                    "shape": list(codes_np.shape),
                    "dtype": "int8",
                    "quant_bits": int(quant_stats["quant_bits"]),
                    "scale": float(tensor_stats["scale"]),
                    "max_abs": float(tensor_stats["max_abs"]),
                    "qmin": int(tensor_stats["qmin"]),
                    "qmax": int(tensor_stats["qmax"]),
                    "mse": float(tensor_stats["mse"]),
                    "relative_l2": float(tensor_stats["relative_l2"]),
                    "saturation_ratio": float(tensor_stats["saturation_ratio"]),
                }
            )
        clients_meta.append(
            {
                "client_id": int(client_id),
                "aggregation_weight": float(weights[client_id]),
                "quant_bits": int(final_round_bits[client_id]),
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
    fp32_fedavg.write_pickle(
        {
            "x": x_all.numpy(),
            "y": y_all.numpy(),
            "z": final_z,
            "index": original_index_all.numpy(),
            "param_names": list(param_names),
            "attack_layer": {"index": int(attack_layer_index), "name": param_names[attack_layer_index]},
            "aggregation_weights": list(weights),
            "observation_scope": args.observation_scope,
            "observed_client_id": (
                int(args.observed_client_id) if args.observation_scope == "single_client" else None
            ),
            "attacker_observes_round_aggregate": args.observation_scope == "aggregate",
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
        "source": "mixed 4/8-bit client q,s before final-round server aggregation",
        "not_source": "post-hoc FP32 quantization",
        "quant_bits": "mixed",
        "mixed_quant_bits": sorted(set(int(value) for value in final_round_bits)),
        "rounding": "nearest",
        "qrange_by_bits": {"4": {"qmin": -7, "qmax": 7}, "8": {"qmin": -127, "qmax": 127}},
        "scale_definition": "scale=max_abs/qmax; scale=0 when max_abs=0",
        "bin_policy_for_randomized": "bounded exact bins using each client's selected bit width",
        "param_names": list(param_names),
        "attack_layer": {"index": int(attack_layer_index), "name": param_names[attack_layer_index]},
        "aggregation_weights": list(weights),
        "observation_scope": args.observation_scope,
        "observed_client_id": (
            int(args.observed_client_id) if args.observation_scope == "single_client" else None
        ),
        "attacker_observes_round_aggregate": args.observation_scope == "aggregate",
        "final_round_quant_bits_by_client": [
            {"client_id": int(client_id), "quant_bits": int(bits)}
            for client_id, bits in enumerate(final_round_bits)
        ],
        "clients": clients_meta,
        "codes_sha256": sha256_file(codes_path),
        "source_model_state_sha256": sha256_file(model_state_path),
        "context_sha256": sha256_file(context_path),
    }
    write_json(quant_meta, client_qs_dir / "final_round_client_quant_meta.json")
    return {
        "code_file": str(codes_path.resolve()),
        "context_file": str(context_path.resolve()),
        "source_model_state_file": str(model_state_path.resolve()),
        "quant_meta_file": str((client_qs_dir / "final_round_client_quant_meta.json").resolve()),
        "codes_sha256": quant_meta["codes_sha256"],
        "source_model_state_sha256": quant_meta["source_model_state_sha256"],
        "context_sha256": quant_meta["context_sha256"],
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.fresh_start:
        safe_rmtree(args.output_dir, allowed_root=REPO_ROOT / "mixed_bit_fedavg")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fp32_fedavg.set_seed(args.seed)
    device = fp32_fedavg.choose_device(args.device)
    bit_schedule, schedule_by_pair = make_bit_schedule(args)

    print(
        f"{quant_tag(args)} FedAvg collection | ds={args.ds} model={args.model} "
        f"clients={args.num_clients} rounds={args.rounds} "
        f"samples/client={args.samples_per_client} bits={args.quant_bits_choices} "
        f"bit_policy={bit_selection_policy(args)} observation={args.observation_scope} device={device}",
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
    param_schema = parameter_schema(model)
    attack_layer_index = resolve_attack_layer_index(param_names)

    client_sizes = [len(x) for x in xs]
    total_client_samples = float(sum(client_sizes))
    weights = [size / total_client_samples for size in client_sizes]
    manifest = make_manifest(
        args,
        device=device,
        param_names=param_names,
        client_indices=client_indices,
        original_indices=original_indices,
        weights=weights,
        attack_layer_index=attack_layer_index,
        bit_schedule=bit_schedule,
    )
    write_json(manifest, args.output_dir / "manifest.json")
    write_json(list(bit_schedule), args.output_dir / "bit_selection_schedule.json")

    metrics: List[Dict[str, Any]] = []
    final_observed_update: Optional[Dict[str, torch.Tensor]] = None
    final_z_all: Optional[np.ndarray] = None
    final_global_state: Optional[Dict[str, torch.Tensor]] = None
    final_client_codes: List[OrderedDict[str, torch.Tensor]] = []
    final_client_stats: List[Dict[str, Any]] = []
    final_round_bits: List[int] = []
    observed_client_id = int(args.observed_client_id)

    for round_idx in range(1, args.rounds + 1):
        round_start = time.time()
        global_params = fp32_fedavg.named_parameters_cpu(model)

        if round_idx == args.rounds:
            final_z_all = fp32_fedavg.extract_embeddings(
                model,
                x_all,
                device=device,
                batch_size=args.local_batch_size,
            )
            final_global_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }

        dequantized_client_updates: List[Dict[str, torch.Tensor]] = []
        client_losses: List[float] = []
        client_quant_stats: List[Dict[str, Any]] = []
        round_client_codes: List[OrderedDict[str, torch.Tensor]] = []
        round_bits: List[int] = []
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
            quant_bits = schedule_by_pair[(round_idx, client_id)]
            codes, dequantized_update, quant_stats = quantize_update_symmetric(
                fp32_update,
                bits=quant_bits,
            )
            quant_stats["round"] = int(round_idx)
            quant_stats["client_id"] = int(client_id)
            round_client_codes.append(codes)
            dequantized_client_updates.append(dict(dequantized_update))
            client_quant_stats.append(quant_stats)
            client_losses.append(train_loss)
            round_bits.append(int(quant_bits))
            del local_params, fp32_update

        aggregate_update = fp32_fedavg.aggregate_updates(dequantized_client_updates, weights)
        fp32_fedavg.apply_parameter_update(model, global_params, aggregate_update)

        weighted_train_loss = sum(
            loss * size for loss, size in zip(client_losses, client_sizes)
        ) / sum(client_sizes)
        metric = {
            "round": int(round_idx),
            "weighted_train_loss": weighted_train_loss,
            "client_train_losses": client_losses,
            "aggregate_update_l2": fp32_fedavg.l2_norm(aggregate_update),
            "quant_bits_by_client": list(round_bits),
            "quant_bit_counts": count_bits(round_bits),
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
            "round {round:03d} | bits {quant_bits_by_client} | train_loss "
            "{weighted_train_loss:.4f} | agg_l2 {aggregate_update_l2:.4f} | "
            "q_rel_l2 {quant_relative_l2_mean:.4f} | time {round_time:.1f}s".format(**metric),
            flush=True,
        )

        if round_idx == args.rounds:
            if args.observation_scope == "single_client":
                final_observed_update = clone_update_cpu(dequantized_client_updates[observed_client_id])
            else:
                final_observed_update = clone_update_cpu(aggregate_update)
            final_client_codes = round_client_codes
            final_client_stats = client_quant_stats
            final_round_bits = round_bits

        del global_params, dequantized_client_updates, aggregate_update
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if final_observed_update is None or final_z_all is None or final_global_state is None:
        raise RuntimeError("Final mixed-bit aggregate update was not captured")

    if args.observation_scope == "single_client":
        offset = sum(int(xs[idx].shape[0]) for idx in range(observed_client_id))
        observed_count = int(xs[observed_client_id].shape[0])
        observed_x = xs[observed_client_id]
        observed_y = ys[observed_client_id]
        observed_z = final_z_all[offset : offset + observed_count]
        observed_original_index = original_indices[observed_client_id]
        observed_weights = [1.0]
    else:
        observed_x = x_all
        observed_y = y_all
        observed_z = final_z_all
        observed_original_index = original_index_all
        observed_weights = weights

    if args.observation_scope == "single_client":
        output_pickle = (
            args.output_dir
            / f"round_{args.rounds:03d}_client{observed_client_id:03d}_"
            f"{quant_tag(args)}_dequant_update_cpa.pickle"
        )
    else:
        output_pickle = args.output_dir / f"round_{args.rounds:03d}_{quant_tag(args)}_dequant_aggregate_cpa.pickle"

    client_qs = None
    if args.store_scope != "none":
        client_qs = save_client_qs_artifacts(
            args=args,
            client_qs_dir=args.output_dir / "client_qs",
            final_client_codes=final_client_codes,
            final_client_stats=final_client_stats,
            final_global_state=final_global_state,
            final_z=final_z_all,
            x_all=x_all,
            y_all=y_all,
            original_index_all=original_index_all,
            original_indices=original_indices,
            param_names=param_names,
            attack_layer_index=attack_layer_index,
            weights=weights,
            final_round_bits=final_round_bits,
        )
        manifest["client_qs"] = client_qs

    checkpoint_path = (
        Path(str(client_qs["source_model_state_file"]))
        if client_qs is not None and client_qs.get("source_model_state_file")
        else None
    )
    observed_quant_codes = None
    observed_quant_stats: List[Dict[str, Any]] = []
    if args.observation_scope == "single_client":
        observed_codes = final_client_codes[observed_client_id]
        observed_quant_codes = [[value.cpu().numpy() for value in observed_codes.values()]]
        observed_quant_stats = [final_client_stats[observed_client_id]]

    final_pickle = fp32_fedavg.make_cpa_pickle(
        args=args,
        x_all=observed_x,
        y_all=observed_y,
        z_all=observed_z,
        original_index_all=observed_original_index,
        aggregate_update=final_observed_update,
        param_names=param_names,
        weights=observed_weights,
        metrics=metrics,
    )
    fixed_bits = fixed_quant_bits(args)
    qbits = quant_bits_metadata(args)
    final_meta = {
        "signal": observation_signal(args),
        "quant_bits": qbits,
        "fixed_quant_bits": fixed_bits,
        "mixed_quant_bits": None if fixed_bits is not None else unique_quant_bits(args),
        "final_round_quant_bits_by_client": list(final_round_bits),
        "observed_client_quant_bits": (
            int(final_round_bits[observed_client_id])
            if args.observation_scope == "single_client"
            else None
        ),
        "rounding": "nearest",
        "quantization": quantization_name(args),
        "scale_scope": "per_client_per_parameter_tensor",
        "server_aggregation_input": "dequantized_client_updates",
        "client_update_quantized_before_server_aggregation": True,
        "aggregate_requantized": False,
        "bit_selection_policy": bit_selection_policy(args),
        "bit_selection_random": fixed_bits is None,
        "bit_selection_seed": int(args.bit_selection_seed),
        "bit_selection_granularity": args.bit_selection_granularity,
        "observation_scope": args.observation_scope,
        "attacker_observation": (
            f"one_client_{quant_tag(args)}_dequantized_update_from_fedavg_round"
            if args.observation_scope == "single_client"
            else f"final_round_{quant_tag(args)}_dequantized_fedavg_aggregate_update"
        ),
        "attacker_observes_round_aggregate": args.observation_scope == "aggregate",
        "observed_client_id": observed_client_id if args.observation_scope == "single_client" else None,
        "update_definition": (
            f"one final-round client's {quant_tag(args)} dequantized FedAvg update"
            if args.observation_scope == "single_client"
            else f"server aggregate of final {quant_tag(args)} dequantized FedAvg round"
        ),
        "fedavg_aggregation_weights": list(weights),
        "observed_sample_count": int(observed_x.shape[0]),
        "total_fedavg_samples": int(x_all.shape[0]),
    }
    final_pickle["meta"][0].update(final_meta)
    final_pickle["param_names"] = list(param_names)
    final_pickle["parameter_schema"] = param_schema
    final_pickle["attack_layer"] = {
        "index": int(attack_layer_index),
        "name": param_names[attack_layer_index],
        "shape": list(final_observed_update[param_names[attack_layer_index]].shape),
        "dtype": "torch.float32",
    }
    final_pickle["sample_indices"] = [
        [int(value) for value in observed_original_index.detach().cpu().numpy().reshape(-1).tolist()]
    ]
    final_pickle["local_batch_orders"] = []
    final_pickle["quant_codes"] = observed_quant_codes
    final_pickle["quant_stats"] = observed_quant_stats
    final_pickle["whitening_pre_stats"] = []
    metadata = {
        "schema_version": 2,
        **final_pickle["meta"][0],
        "ds": args.ds,
        "model": args.model,
        "h_dim": int(args.h_dim),
        "n_samples": int(observed_x.shape[0]),
        "attack_trials": 1,
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "local_steps": int(args.local_epochs)
        * int(math.ceil(int(args.samples_per_client) / max(1, int(args.local_batch_size)))),
        "local_batch_size": int(args.local_batch_size),
        "micro_batch_size": int(args.local_batch_size),
        "optimizer": "SGD",
        "lr": float(args.lr),
        "momentum": 0.0,
        "weight_decay": 0.0,
        "seed": int(args.seed),
        "checkpoint": checkpoint_manifest(checkpoint_path),
        "fake_quantization": True,
        "dequantized_before_attack": True,
        "quantization_bin_observation": args.observation_scope == "single_client",
        "quantization_missing_value_policy": (
            "quantized code and scale define a nearest-rounding bin; randomized runner samples inside the bin"
            if args.observation_scope == "single_client"
            else None
        ),
        "clients_per_trial": 1 if args.observation_scope == "single_client" else int(args.num_clients),
        "fedavg_num_clients": int(args.num_clients),
        "client_granularity": f"{int(args.samples_per_client)}_samples_per_client",
    }
    metadata["sample_indices_hash"] = sha256_bytes(
        json.dumps(final_pickle["sample_indices"], sort_keys=True).encode("utf-8")
    )
    metadata["label_hash"] = hash_tensor_sequence(final_pickle["y"])
    metadata["update_tensor_hash"] = hash_tensor_sequence(final_pickle["grad"])
    if observed_quant_stats:
        metadata["quantization_scales_hash"] = sha256_bytes(
            json.dumps(observed_quant_stats, sort_keys=True).encode("utf-8")
        )
    final_pickle["metadata"] = metadata

    manifest["source_update_file"] = str(output_pickle.resolve())
    manifest["source_update_hash"] = None
    manifest["observed_client_quant_bits"] = metadata["observed_client_quant_bits"]
    manifest["quantization_scales_hash"] = metadata.get("quantization_scales_hash")
    write_json(metrics, args.output_dir / "metrics.json")
    fp32_fedavg.write_metrics_csv(metrics, args.output_dir / "metrics.csv")
    write_json(final_client_stats, args.output_dir / "final_round_client_quant_stats.json")
    fp32_fedavg.write_pickle(final_pickle, output_pickle)
    manifest["source_update_hash"] = sha256_file(output_pickle)
    write_json(manifest, args.output_dir / "manifest.json")

    if args.export_baseline_update is not None:
        fp32_fedavg.write_pickle(final_pickle, args.export_baseline_update)

    print(f"Saved final {quant_tag(args)}/dequant {args.observation_scope} CPA pickle: {output_pickle}", flush=True)
    if args.store_scope != "none":
        print(f"Saved final-round client q/s: {args.output_dir / 'client_qs'}", flush=True)
    if args.export_baseline_update is not None:
        print(f"Exported baseline-compatible update: {args.export_baseline_update}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
