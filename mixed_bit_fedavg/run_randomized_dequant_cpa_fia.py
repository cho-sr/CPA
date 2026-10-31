#!/usr/bin/env python3
"""Run sparsification-style randomized dequantization on fixed-bit FL updates.

The collected observation must contain quantization codes and per-tensor scales.
For each ensemble member, the runner samples values uniformly inside the
nearest-rounding quantization bin, then runs the author CPA path.  This is the
quantization analogue of sparse desparsification: the observed code is fixed,
and only the information lost inside each bin is randomized.
"""

from __future__ import annotations

import argparse
import hashlib
import copy
import json
import math
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SPARSE_SRC = REPO_ROOT / "sparsified_update_cpa"
if str(SPARSE_SRC) in sys.path:
    sys.path.remove(str(SPARSE_SRC))
sys.path.insert(0, str(SPARSE_SRC))

import author_runner as sparse_author_runner  # noqa: E402
import common as sparse_common  # noqa: E402
from blind_alignment import blind_medoid_align_and_average  # noqa: E402
from metrics import write_csv  # noqa: E402
from run_mixed_bit_cpa_fia import (  # noqa: E402
    MIXED_AUTHOR_RUN_ROOT,
    _checkpoint_path,
    _manifest_path,
    convert_update_schema,
    hydrate_args,
    read_json,
    resolve_path,
)


ENSEMBLE_ROOT = REPO_ROOT / "mixed_bit_fedavg" / "outputs_v1" / "randomized_dequant"
IMAGE_MEAN_STD = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "tiny_imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update_file", type=Path, required=True)
    parser.add_argument(
        "--manifest_file",
        type=Path,
        default=None,
        help="Defaults to manifest.json next to --update_file.",
    )
    parser.add_argument(
        "--global_checkpoint",
        type=Path,
        default=None,
        help="Defaults to client_qs/final_round_global_model_before.pt next to --update_file.",
    )
    parser.add_argument("--ds", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--h_dim", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--attack_trials", type=int, default=None)
    parser.add_argument("--local_epochs", type=int, default=None)
    parser.add_argument("--local_batch_size", type=int, default=None)
    parser.add_argument("--local_lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--success_lpips_threshold", type=float, default=0.50)
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--noise_seed", type=int, default=1234)
    parser.add_argument(
        "--perturb_scope",
        type=str,
        default="attack_layer",
        choices=["attack_layer", "all"],
        help="Sample only the CPA attack layer or every parameter tensor.",
    )
    parser.add_argument(
        "--run_member_fia",
        action="store_true",
        help="Run Direct FIA for each randomized member instead of CPA-only members.",
    )
    parser.add_argument(
        "--run_aggregate_fia",
        action="store_true",
        help="Blind-align member CPA embeddings, average them, then run one Direct FIA.",
    )
    parser.add_argument(
        "--run_consistency_ranking",
        action="store_true",
        help=(
            "CoRA-style selection: each member must have Direct FIA reconstructions; "
            "recompute the selected client's update from each reconstruction, quantize/dequantize "
            "it, and rank by relative L2 to the observed quantized update."
        ),
    )
    parser.add_argument(
        "--fia_screen_iter",
        type=int,
        default=0,
        help=(
            "Run all randomized members only to this Direct-FIA iteration, rank them, "
            "then continue only --fia_screen_top_k members to --n_iter_fi."
        ),
    )
    parser.add_argument(
        "--fia_screen_top_k",
        type=int,
        default=0,
        help="Number of screening-ranked members to continue. Requires --fia_screen_iter > 0.",
    )
    parser.add_argument(
        "--consistency_label_mode",
        type=str,
        default="model_pred",
        choices=["model_pred", "zero"],
        help="Labels used only while recomputing updates for no-oracle consistency ranking.",
    )
    parser.add_argument(
        "--ranking_micro_batch_size",
        type=int,
        default=None,
        help="Memory-only micro-batch size for ranking update recomputation.",
    )
    parser.add_argument(
        "--fia_stop_iter",
        type=int,
        default=None,
        help="Stop member Direct FIA early. Aggregate FIA still uses --n_iter_fi.",
    )
    sparse_common.add_attack_args(parser)
    parser.set_defaults(project="cpa_quantized_random_dequant")
    return parser.parse_args()


def qrange(bits: int) -> tuple[int, int]:
    if bits not in {4, 8}:
        raise ValueError(f"fixed-bit randomized dequantization supports only 4 or 8 bits, got {bits}")
    qmax = (2 ** (bits - 1)) - 1
    return -qmax, qmax


def member_seed_plan(ensemble_size: int, noise_seed: int) -> list[dict[str, Any]]:
    return [
        {
            "member": member_idx,
            "dequantization_seed": int(noise_seed + 1_000_003 * member_idx),
            "cpa_seed": "fixed_from_args_seed",
            "trial_varying_randomness": "quantization_bin_noise_only",
        }
        for member_idx in range(int(ensemble_size))
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


def _quant_bits(data: Mapping[str, Any]) -> int:
    metadata = data.get("metadata", {})
    value = metadata.get("quant_bits")
    if value == "mixed" or value is None:
        value = metadata.get("observed_client_quant_bits")
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if value not in {4, 8}:
        raise ValueError(
            "Randomized dequantization needs a fixed 4bit or 8bit observation. "
            f"metadata quant_bits={metadata.get('quant_bits')!r}, "
            f"observed_client_quant_bits={metadata.get('observed_client_quant_bits')!r}"
        )
    return int(value)


def _randomized_dequantize_array(
    codes: np.ndarray,
    *,
    scale: float,
    bits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    qmin, qmax = qrange(bits)
    codes_i = np.asarray(codes, dtype=np.int16)
    scale_f = float(scale)
    base = codes_i.astype(np.float32) * scale_f
    if scale_f == 0.0:
        return base.astype(np.float32, copy=False)
    noise = rng.uniform(-0.5, 0.5, size=base.shape).astype(np.float32, copy=False) * scale_f
    interior = (codes_i > qmin) & (codes_i < qmax)
    return (base + noise * interior.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def _quantize_dequantize_array(
    values: np.ndarray,
    *,
    bits: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    qmin, qmax = qrange(bits)
    original = np.asarray(values, dtype=np.float32)
    max_abs = float(np.max(np.abs(original))) if original.size else 0.0
    if max_abs > 0.0:
        scale = max_abs / float(qmax)
        codes = np.rint(original / scale).clip(qmin, qmax).astype(np.int8, copy=False)
        dequantized = (codes.astype(np.float32, copy=False) * scale).astype(np.float32, copy=False)
    else:
        scale = 0.0
        codes = np.zeros_like(original, dtype=np.int8)
        dequantized = np.zeros_like(original, dtype=np.float32)
    error = dequantized.astype(np.float64, copy=False) - original.astype(np.float64, copy=False)
    original64 = original.astype(np.float64, copy=False)
    dequant64 = dequantized.astype(np.float64, copy=False)
    error_sq = float(np.dot(error.reshape(-1), error.reshape(-1)))
    original_sq = float(np.dot(original64.reshape(-1), original64.reshape(-1)))
    dequant_sq = float(np.dot(dequant64.reshape(-1), dequant64.reshape(-1)))
    dot = float(np.dot(original64.reshape(-1), dequant64.reshape(-1)))
    stats = {
        "quant_bits": int(bits),
        "scale": float(scale),
        "max_abs": float(max_abs),
        "mse": error_sq / max(int(original.size), 1),
        "relative_l2": math.sqrt(error_sq) / max(math.sqrt(original_sq), 1e-12),
        "cosine_similarity": dot / max(math.sqrt(original_sq) * math.sqrt(dequant_sq), 1e-12),
        "saturation_ratio": float(np.mean(np.abs(codes) >= qmax)) if codes.size else 0.0,
        "qmin": int(qmin),
        "qmax": int(qmax),
        "numel": int(original.size),
    }
    return codes, dequantized, stats


def _requantize_with_scale(values: np.ndarray, *, scale: float, bits: int) -> np.ndarray:
    qmin, qmax = qrange(bits)
    original = np.asarray(values, dtype=np.float32)
    scale_f = float(scale)
    if scale_f <= 0.0:
        return np.zeros_like(original, dtype=np.int8)
    return np.rint(original / scale_f).clip(qmin, qmax).astype(np.int8, copy=False)


def _relative_l2_chunked(pred: np.ndarray, observed: np.ndarray, chunk_elems: int = 2_000_000) -> float:
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


def _resolve_ranking_micro_batch_size(args: argparse.Namespace) -> int:
    local_batch_size = int(args.local_batch_size or args.n_samples)
    requested = args.ranking_micro_batch_size
    if requested is None and args.ds == "imagenet" and args.model == "vgg16":
        value = min(local_batch_size, sparse_common.AUTO_VGG16_IMAGENET_COLLECT_MICRO_BATCH_SIZE)
    else:
        value = int(requested or local_batch_size)
    if value < 1 or value > local_batch_size:
        raise ValueError("--ranking_micro_batch_size must be in [1, local_batch_size]")
    return value


def _local_steps(args: argparse.Namespace) -> int:
    local_batch_size = int(args.local_batch_size or args.n_samples)
    return int(args.local_epochs) * int(math.ceil(int(args.n_samples) / max(1, local_batch_size)))


def _clone_model_from_global(
    global_model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    local_model = sparse_common.get_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        load_path=None,
        dataparallel=False,
    )
    state = {name: value.detach().cpu() for name, value in global_model.state_dict().items()}
    local_model.load_state_dict(state, strict=True)
    return local_model.to(device=device, dtype=torch.float32)


def _normalize_unit_image(x_unit: torch.Tensor, ds: str) -> torch.Tensor:
    if ds not in IMAGE_MEAN_STD:
        raise ValueError(f"Unsupported dataset for ranking normalization: {ds}")
    mean_values, std_values = IMAGE_MEAN_STD[ds]
    mean = torch.tensor(mean_values, device=x_unit.device, dtype=x_unit.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std_values, device=x_unit.device, dtype=x_unit.dtype).view(1, 3, 1, 1)
    return (x_unit.clamp(0.0, 1.0) - mean) / std


def _choose_ranking_labels(
    mode: str,
    model: torch.nn.Module,
    x_norm: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
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


def _train_single_client_update_from_reconstruction(
    *,
    global_model: torch.nn.Module,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    log_prefix: str,
) -> OrderedDict[str, torch.Tensor]:
    print(f"{log_prefix} local model init", flush=True)
    local_model = _clone_model_from_global(global_model, args, device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=float(args.local_lr), momentum=0.0, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    local_batch_size = int(args.local_batch_size or args.n_samples)
    micro_batch_size = _resolve_ranking_micro_batch_size(args)
    loader = DataLoader(TensorDataset(x_cpu, y_cpu), batch_size=local_batch_size, shuffle=False, num_workers=0)
    print(
        f"{log_prefix} local train start steps={_local_steps(args)} "
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
        for name, value in sparse_common._named_update_dict(local_model, global_model).items()
    )
    del optimizer, local_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return update


def _recompress_reconstruction_quantized_attack_layer(
    *,
    args: argparse.Namespace,
    global_model: torch.nn.Module,
    rec_fi: np.ndarray,
    attack_layer_name: str,
    bits: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    device = next(global_model.parameters()).device
    rec = torch.as_tensor(rec_fi, device=device, dtype=torch.float32)
    x_norm = _normalize_unit_image(rec, args.ds)
    ranking_micro_batch_size = _resolve_ranking_micro_batch_size(args)
    y_pred = _choose_ranking_labels(
        args.consistency_label_mode,
        global_model,
        x_norm,
        batch_size=max(1, min(ranking_micro_batch_size, int(x_norm.shape[0]))),
    )
    update = _train_single_client_update_from_reconstruction(
        global_model=global_model,
        x_cpu=x_norm.detach().cpu(),
        y_cpu=y_pred.detach().cpu(),
        args=args,
        device=device,
        log_prefix="[quant-rank]",
    )
    attack = update[attack_layer_name].detach().cpu().numpy().astype(np.float32, copy=False)
    _codes, dequantized, stats = _quantize_dequantize_array(attack, bits=bits)
    raw_attack = np.array(attack, dtype=np.float32, copy=True)
    del update, attack
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return raw_attack, dequantized, stats


def materialize_member_update_data(
    *,
    base_data: Mapping[str, Any],
    base_update_file: Path,
    base_update_file_hash: str,
    member_idx: int,
    member_seed: int,
    ensemble_size: int,
    noise_seed: int,
    perturb_scope: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not base_data.get("quant_codes"):
        raise ValueError("update file is missing quant_codes; recollect with --quant_bits 4 or --quant_bits 8")
    if not base_data.get("quant_stats"):
        raise ValueError("update file is missing quant_stats; scales are required for bin sampling")

    bits = _quant_bits(base_data)
    attack_layer_index = int(base_data["attack_layer"]["index"])
    member_grad = []
    randomized_arrays = []
    batch_stats = []
    for batch_idx, batch_grad in enumerate(base_data["grad"]):
        batch_values = [np.asarray(value, dtype=np.float32) for value in batch_grad]
        codes_by_tensor = base_data["quant_codes"][batch_idx]
        stats_by_tensor = base_data["quant_stats"][batch_idx]["per_tensor"]
        rng = np.random.default_rng(int(member_seed + 9_176_521 * batch_idx))
        randomized_tensors = 0
        sampled_values = 0
        for tensor_idx, codes_np in enumerate(codes_by_tensor):
            if perturb_scope == "attack_layer" and tensor_idx != attack_layer_index:
                continue
            scale = float(stats_by_tensor[tensor_idx]["scale"])
            sampled = _randomized_dequantize_array(codes_np, scale=scale, bits=bits, rng=rng)
            batch_values[tensor_idx] = sampled
            randomized_arrays.append(sampled)
            randomized_tensors += 1
            sampled_values += int(sampled.size)
        member_grad.append(batch_values)
        batch_stats.append(
            {
                "batch": int(batch_idx),
                "dequantization_seed": int(member_seed + 9_176_521 * batch_idx),
                "quant_bits": int(bits),
                "perturb_scope": perturb_scope,
                "randomized_tensors": int(randomized_tensors),
                "sampled_values": int(sampled_values),
                "sampler": "uniform_inside_nearest_rounding_bin",
                "saturation_policy": "qmin/qmax saturation bins kept at q*scale",
            }
        )

    randomized_hash = _hash_arrays(randomized_arrays)
    metadata = copy.deepcopy(dict(base_data["metadata"]))
    metadata.update(
        {
            "randomized_dequantization": True,
            "randomized_dequantization_policy": "uniform_inside_nearest_rounding_quantization_bin",
            "randomized_dequantization_interpretation": "heuristic missing-precision fill, not exact posterior sampling",
            "dequantization_scope": perturb_scope,
            "missing_value_sampler": "uniform",
            "quantization_missing_value": "continuous residual inside observed quantization bin",
            "ensemble_member": int(member_idx),
            "ensemble_size": int(ensemble_size),
            "ensemble_noise_seed": int(noise_seed),
            "ensemble_dequantization_seed": int(member_seed),
            "source_quantized_update_file": str(base_update_file.resolve()),
            "source_quantized_update_hash": base_update_file_hash,
            "update_file_hash_override": base_update_file_hash,
            "randomized_dequantized_values_hash": randomized_hash,
        }
    )
    member_data = dict(base_data)
    member_data["metadata"] = metadata
    member_data["grad"] = member_grad
    member_data["metadata"]["update_tensor_hash"] = sparse_common.hash_tensor_sequence(member_grad)
    return member_data, batch_stats


def _load_member_sources(run_paths: Sequence[Path], n_batches: int) -> tuple[list[list[np.ndarray]], str]:
    per_member = []
    source_kind: Optional[str] = None
    for path in run_paths:
        rows = sparse_common.read_pickle(path / "reconstructions.pkl")
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
    base_data: Mapping[str, Any],
    base_update_file: Path,
    base_update_file_hash: str,
    run_paths: Sequence[Path],
    seed_plan: Sequence[Mapping[str, Any]],
    run_id: str,
) -> tuple[Path, list[np.ndarray], str]:
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

    output_dir = ENSEMBLE_ROOT / run_id
    output_file = output_dir / "blind_ensemble_sources.pkl"
    sparse_common.write_pickle(
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
    sparse_common.write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "method": "fixed_bit_quantized_update_randomized_dequant_blind_ensemble",
            "source_quantized_update_file": str(base_update_file.resolve()),
            "source_quantized_update_hash": base_update_file_hash,
            "source_run_paths": [str(path.resolve()) for path in run_paths],
            "quant_bits": _quant_bits(base_data),
            "ensemble_size": int(args.ensemble_size),
            "noise_seed": int(args.noise_seed),
            "seed_plan": list(seed_plan),
            "dequantization_scope": args.perturb_scope,
            "missing_value_sampler": "uniform",
            "missing_value_policy": "sample continuous residual inside quantization bin",
            "alignment": "blind_medoid_align_then_mean",
            "source_kind": source_kind,
            "oracle_inputs_used": False,
        },
    )
    return output_file, final_batches, source_kind


def rank_members_by_quantized_consistency(
    *,
    args: argparse.Namespace,
    base_data: Mapping[str, Any],
    base_update_file: Path,
    base_update_file_hash: str,
    run_paths: Sequence[Path],
    seed_plan: Sequence[Mapping[str, Any]],
    run_id: str,
    output_suffix: str = "consistency_ranked",
    manifest_extra: Optional[Mapping[str, Any]] = None,
) -> tuple[Path, list[dict[str, Any]]]:
    if not run_paths:
        raise ValueError("No member run paths provided for consistency ranking")
    bits = _quant_bits(base_data)
    device = sparse_common.get_device()
    model = sparse_common.build_experiment_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        device=device,
        checkpoint_path=args.global_checkpoint,
        seed=args.seed,
    )
    model.eval()
    attack_idx = int(base_data["attack_layer"]["index"])
    attack_name = str(base_data["attack_layer"]["name"])
    observed_attack = [np.asarray(batch[attack_idx], dtype=np.float32) for batch in base_data["grad"]]
    observed_codes = [
        np.asarray(batch_codes[attack_idx], dtype=np.int8)
        for batch_codes in base_data.get("quant_codes", [])
    ]
    observed_scales = [
        float(batch_stats["per_tensor"][attack_idx]["scale"])
        for batch_stats in base_data.get("quant_stats", [])
    ]
    n_batches = min(args.attack_n_batch or len(observed_attack), len(observed_attack))
    full_n = int(base_data["metadata"]["n_samples"])
    ranked: list[dict[str, Any]] = []
    for position, run_path in enumerate(run_paths):
        member_idx = int(seed_plan[position]["member"]) if position < len(seed_plan) else int(position)
        recon_path = run_path / "reconstructions.pkl"
        if not recon_path.exists():
            raise FileNotFoundError(recon_path)
        recon_entries = sparse_common.read_pickle(recon_path)
        costs: list[float] = []
        code_matches: list[float] = []
        scale_errors: list[float] = []
        pred_quant_errors: list[float] = []
        for batch_pos, entry in enumerate(recon_entries[:n_batches]):
            rec_fi = np.asarray(entry.get("rec_fi_attack_order"), dtype=np.float32)
            if rec_fi.size == 0:
                raise ValueError(
                    "quantized consistency ranking requires Direct FIA reconstructions for each member. "
                    "Rerun with --run_member_fia --run_consistency_ranking."
                )
            if int(rec_fi.shape[0]) < full_n:
                raise ValueError(
                    "Exact quantized consistency ranking needs FIA reconstructions for the full observed client batch. "
                    f"Got {rec_fi.shape[0]}/{full_n}. Use --n_sample_fi {full_n}."
                )
            batch = int(entry["batch"])
            print(
                f"[quant-rank] member m{member_idx:03d} ({position + 1}/{len(run_paths)}) "
                f"batch {batch_pos + 1}/{n_batches} recompress n={full_n}",
                flush=True,
            )
            pred_raw_attack, pred_attack, pred_stats = _recompress_reconstruction_quantized_attack_layer(
                args=args,
                global_model=model,
                rec_fi=rec_fi[:full_n],
                attack_layer_name=attack_name,
                bits=bits,
            )
            costs.append(_relative_l2_chunked(pred_attack, observed_attack[batch]))
            pred_quant_errors.append(float(pred_stats["relative_l2"]))
            if batch < len(observed_codes) and batch < len(observed_scales):
                obs_scale = float(observed_scales[batch])
                pred_codes_on_observed_scale = _requantize_with_scale(
                    pred_raw_attack,
                    scale=obs_scale,
                    bits=bits,
                )
                obs_codes = observed_codes[batch]
                if pred_codes_on_observed_scale.shape == obs_codes.shape:
                    code_matches.append(float(np.mean(pred_codes_on_observed_scale == obs_codes)))
                pred_scale = float(pred_stats["scale"])
                scale_errors.append(abs(pred_scale - obs_scale) / max(abs(obs_scale), 1e-12))
            del pred_raw_attack, pred_attack
        mean_cost = float(np.mean(costs)) if costs else float("inf")
        ranked.append(
            {
                "member": int(member_idx),
                "score": -mean_cost,
                "mean_relative_l2_to_observed_quantized_update": mean_cost,
                "mean_requantized_code_match_observed_scale": (
                    float(np.mean(code_matches)) if code_matches else None
                ),
                "mean_attack_layer_scale_relative_error": (
                    float(np.mean(scale_errors)) if scale_errors else None
                ),
                "mean_predicted_update_quant_relative_l2": (
                    float(np.mean(pred_quant_errors)) if pred_quant_errors else None
                ),
                "batches": int(len(costs)),
                "source_run_path": str(run_path.resolve()),
                "ranking_observation": "one_client_fixed_bit_quantized_update_from_fedavg_round",
                "ranking_label_mode": args.consistency_label_mode,
                "ranking_compression": f"signed_symmetric_int{bits}_per_tensor_quantize_dequantize",
            }
        )
    ranked.sort(key=lambda row: float(row["mean_relative_l2_to_observed_quantized_update"]))
    suffix = str(output_suffix).strip("_")
    if not suffix:
        raise ValueError("output_suffix must not be empty")
    output_dir = ENSEMBLE_ROOT / f"{run_id}_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "consistency_member_ranking.csv", ranked)
    best = ranked[0]
    best_path = Path(str(best["source_run_path"]))
    for name in (
        "reconstructions.pkl",
        "image_metrics_by_sample.csv",
        "metric_summary.csv",
        "oracle_evaluation.json",
    ):
        source = best_path / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
    sparse_common.write_pickle(
        output_dir / "selected_reconstructions.pkl",
        sparse_common.read_pickle(best_path / "reconstructions.pkl"),
    )
    ranking_manifest = {
        "schema_version": 1,
        "method": "CoRA-style fixed-bit randomized dequantization consistency ranking",
        "source_quantized_update_file": str(base_update_file.resolve()),
        "source_quantized_update_hash": base_update_file_hash,
        "quant_bits": int(bits),
        "ensemble_size": int(len(run_paths)),
        "noise_seed": int(args.noise_seed),
        "seed_plan": list(seed_plan),
        "dequantization_scope": args.perturb_scope,
        "missing_value_sampler": "uniform",
        "missing_value_policy": "sample continuous residual inside quantization bin",
        "ranking_observation": "one_client_fixed_bit_quantized_update_from_fedavg_round",
        "ranking_score": "mean relative L2 between recomputed quantized/dequantized attack layer and observed dequantized attack layer",
        "ranking_label_mode": args.consistency_label_mode,
        "oracle_inputs_used_for_ranking": False,
        "selected_member": int(best["member"]),
        "selected_member_run_path": str(best_path.resolve()),
        "member_ranking_file": "consistency_member_ranking.csv",
        "model_checkpoint_for_attack_and_ranking": str(Path(args.global_checkpoint).resolve())
        if args.global_checkpoint
        else None,
        "update_file": str(base_update_file.resolve()),
        "ds": args.ds,
        "model": args.model,
        "h_dim": int(args.h_dim),
        "n_samples": int(args.n_samples),
        "attack_trials": int(args.attack_trials),
        "local_epochs": int(args.local_epochs),
        "local_batch_size": int(args.local_batch_size),
        "local_lr": float(args.local_lr),
    }
    if manifest_extra:
        ranking_manifest.update(dict(manifest_extra))
    sparse_common.write_json(output_dir / "manifest.json", ranking_manifest)
    return output_dir, ranked


def run_id(args: argparse.Namespace, data: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    del manifest
    metadata = data["metadata"]
    bits = _quant_bits(data)
    observation_scope = str(metadata.get("observation_scope", "unknown"))
    if observation_scope == "single_client":
        obs_tag = f"client{int(metadata.get('observed_client_id', 0))}"
    else:
        obs_tag = "aggregate"
    n_sample_fi = args.n_samples if args.n_sample_fi in {-1, 0} else min(args.n_sample_fi, args.n_samples)
    return (
        f"random_dequant_int{bits}_{obs_tag}_n{args.n_samples}_steps"
        f"{metadata.get('local_steps', args.local_epochs)}_trials{args.attack_trials}"
        f"_R{args.ensemble_size}_seed{args.noise_seed}_{args.perturb_scope}_fia{n_sample_fi}"
    )


def validate_screening_args(args: argparse.Namespace) -> None:
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
    if not args.run_member_fia:
        raise ValueError("--fia_screen_iter requires --run_member_fia")
    if not args.run_consistency_ranking:
        raise ValueError("--fia_screen_iter requires --run_consistency_ranking")
    if screen_top_k > int(args.ensemble_size):
        raise ValueError("--fia_screen_top_k must be <= --ensemble_size")
    if screen_iter >= int(args.n_iter_fi):
        raise ValueError("--fia_screen_iter must be smaller than --n_iter_fi")
    n_sample_fi = args.n_samples if args.n_sample_fi in {-1, 0} else min(args.n_sample_fi, args.n_samples)
    if int(n_sample_fi) < int(args.n_samples):
        raise ValueError("screening/ranking requires full-batch FIA; use --n_sample_fi equal to n_samples")


def use_screening(args: argparse.Namespace) -> bool:
    return int(args.fia_screen_iter) > 0 and int(args.fia_screen_top_k) > 0


def run_quant_member_attack(
    *,
    args: argparse.Namespace,
    base_data: Mapping[str, Any],
    base_update_file: Path,
    base_update_file_hash: str,
    bits: int,
    tag: str,
    seed_row: Mapping[str, Any],
) -> Path:
    member_idx = int(seed_row["member"])
    member_seed = int(seed_row["dequantization_seed"])
    print(
        f"[quant-rand] int{bits} member {member_idx + 1}/{args.ensemble_size} "
        f"seed={member_seed} start",
        flush=True,
    )
    member_data, batch_stats = materialize_member_update_data(
        base_data=base_data,
        base_update_file=base_update_file,
        base_update_file_hash=base_update_file_hash,
        member_idx=member_idx,
        member_seed=member_seed,
        ensemble_size=args.ensemble_size,
        noise_seed=args.noise_seed,
        perturb_scope=args.perturb_scope,
    )
    exp_name = f"{tag}_m{member_idx:03d}"
    run_path = sparse_author_runner.run_author_cpa_with_update_file(
        args,
        exp_name=exp_name,
        update_file=base_update_file,
        update_data=member_data,
        run_fia=bool(args.run_member_fia),
        method=(
            f"Fixed int{bits} FL update randomized dequantization member: "
            "observed q-code and scale are fixed; missing precision is sampled "
            "uniformly inside the nearest-rounding bin before author CPA."
        ),
        extra_manifest={
            "update_file_role": "fixed-bit quantized FL observation; grad overridden in memory by randomized dequantization",
            "randomized_dequantization_policy": "uniform_inside_nearest_rounding_quantization_bin",
            "dequantization_scope": args.perturb_scope,
            "missing_value_sampler": "uniform",
            "missing_value_policy": "sample continuous residual inside quantization bin",
            "ensemble_member": member_idx,
            "ensemble_size": int(args.ensemble_size),
            "ensemble_noise_seed": int(args.noise_seed),
            "ensemble_dequantization_seed": member_seed,
            "dequantization_batch_stats": batch_stats,
        },
        attack_order_policy=(
            "member CPA order; no oracle permutation; source will be aligned only by blind ensemble"
        ),
    )
    print(f"[quant-rand] int{bits} member {member_idx + 1}/{args.ensemble_size} done: {run_path}", flush=True)
    return run_path


def run_quant_member_attacks(
    *,
    args: argparse.Namespace,
    base_data: Mapping[str, Any],
    base_update_file: Path,
    base_update_file_hash: str,
    bits: int,
    tag: str,
    seed_plan: Sequence[Mapping[str, Any]],
) -> list[Path]:
    return [
        run_quant_member_attack(
            args=args,
            base_data=base_data,
            base_update_file=base_update_file,
            base_update_file_hash=base_update_file_hash,
            bits=bits,
            tag=tag,
            seed_row=seed_row,
        )
        for seed_row in seed_plan
    ]


def ranked_seed_plan(
    seed_plan: Sequence[Mapping[str, Any]],
    ranked: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
) -> list[Mapping[str, Any]]:
    by_member = {int(row["member"]): row for row in seed_plan}
    selected = []
    for row in ranked[: int(top_k)]:
        member = int(row["member"])
        if member not in by_member:
            raise ValueError(f"screening selected unknown member {member}")
        selected.append(by_member[member])
    return selected


def main() -> None:
    args = parse_args()
    if args.ensemble_size < 1:
        raise ValueError("--ensemble_size must be >= 1")

    update_file = resolve_path(args.update_file)
    if not update_file.is_file():
        raise FileNotFoundError(update_file)
    manifest_file = _manifest_path(update_file, args.manifest_file)
    manifest = read_json(manifest_file) if manifest_file.is_file() else {}
    checkpoint_path = _checkpoint_path(update_file, args.global_checkpoint, manifest)
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Could not find final-round global model checkpoint. Pass --global_checkpoint "
            "or collect with --store_scope attack_layer/all."
        )

    raw_data = sparse_common.read_pickle(update_file)
    args = hydrate_args(
        args,
        update_data=raw_data,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
    )
    base_data = convert_update_schema(
        update_data=raw_data,
        manifest=manifest,
        update_file=update_file,
        checkpoint_path=checkpoint_path,
        args=args,
    )
    bits = _quant_bits(base_data)
    if base_data["metadata"].get("observation_scope") != "single_client":
        raise ValueError("This runner expects --observation_scope single_client fixed-bit observations")
    if not base_data.get("quant_codes"):
        raise ValueError("No quant_codes found. Recollect the observation with --quant_bits 4 or --quant_bits 8.")
    validate_screening_args(args)

    sparse_author_runner.AUTHOR_RUN_ROOT = MIXED_AUTHOR_RUN_ROOT
    seed_plan = member_seed_plan(args.ensemble_size, args.noise_seed)
    base_hash = sparse_common.sha256_file(update_file)
    tag = run_id(args, base_data, manifest)

    final_seed_plan: Sequence[Mapping[str, Any]] = seed_plan
    if use_screening(args):
        screen_args = copy.copy(args)
        screen_args.fia_stop_iter = int(args.fia_screen_iter)
        print(
            f"[quant-screen] run {args.ensemble_size} members to FIA iter "
            f"{args.fia_screen_iter}, then continue top {args.fia_screen_top_k}",
            flush=True,
        )
        screen_paths = run_quant_member_attacks(
            args=screen_args,
            base_data=base_data,
            base_update_file=update_file,
            base_update_file_hash=base_hash,
            bits=bits,
            tag=tag,
            seed_plan=seed_plan,
        )
        screen_dir, screen_ranked = rank_members_by_quantized_consistency(
            args=screen_args,
            base_data=base_data,
            base_update_file=update_file,
            base_update_file_hash=base_hash,
            run_paths=screen_paths,
            seed_plan=seed_plan,
            run_id=tag,
            output_suffix=f"fia_screen_iter{int(args.fia_screen_iter)}_ranked",
            manifest_extra={
                "ranking_stage": "fia_screen",
                "screen_fia_iter": int(args.fia_screen_iter),
                "screen_top_k": int(args.fia_screen_top_k),
                "screened_ensemble_size": int(args.ensemble_size),
            },
        )
        final_seed_plan = ranked_seed_plan(
            seed_plan,
            screen_ranked,
            top_k=int(args.fia_screen_top_k),
        )
        selected_members = [int(row["member"]) for row in final_seed_plan]
        print(
            "[quant-screen] selected "
            + " ".join(f"m{member:03d}" for member in selected_members)
            + f" from {screen_dir}",
            flush=True,
        )
        args.fia_stop_iter = None
        args.fresh_start = False

    run_paths = run_quant_member_attacks(
        args=args,
        base_data=base_data,
        base_update_file=update_file,
        base_update_file_hash=base_hash,
        bits=bits,
        tag=tag,
        seed_plan=final_seed_plan,
    )

    if args.ensemble_size > 1:
        ensemble_file, aggregated_sources, source_kind = aggregate_member_sources(
            args=args,
            base_data=base_data,
            base_update_file=update_file,
            base_update_file_hash=base_hash,
            run_paths=run_paths,
            seed_plan=final_seed_plan,
            run_id=tag,
        )
        print(f"blind ensemble sources: {ensemble_file}", flush=True)
        if args.run_aggregate_fia:
            if source_kind != "rec_emb_attack_order":
                raise ValueError(
                    "--run_aggregate_fia requires a conv model with recovered embedding sources; "
                    f"got source_kind={source_kind!r}"
                )
            sparse_author_runner.run_direct_fia_from_embeddings(
                args,
                exp_name=f"{tag}_blindagg",
                update_file=update_file,
                recovered_embeddings=aggregated_sources,
                method=(
                    f"Direct FIA from blind fixed-int{bits} randomized-dequantization "
                    "ensemble aggregate"
                ),
                extra_manifest={
                    "aggregate_fia_input": "blind_ensemble_recovered_embeddings",
                    "gradient_role": "runner compatibility/context only; Direct FIA optimizes against recovered embeddings",
                    "source_blind_ensemble_file": str(ensemble_file.resolve()),
                    "randomized_dequantization_policy": "uniform_inside_nearest_rounding_quantization_bin",
                    "dequantization_scope": args.perturb_scope,
                    "missing_value_sampler": "uniform",
                    "missing_value_policy": "sample continuous residual inside quantization bin",
                    "ensemble_size": int(args.ensemble_size),
                    "ensemble_noise_seed": int(args.noise_seed),
                },
            )

    if args.run_consistency_ranking:
        ranking_dir, ranked = rank_members_by_quantized_consistency(
            args=args,
            base_data=base_data,
            base_update_file=update_file,
            base_update_file_hash=base_hash,
            run_paths=run_paths,
            seed_plan=final_seed_plan,
            run_id=tag,
            manifest_extra={
                "screening_enabled": bool(use_screening(args)),
                "screen_fia_iter": int(args.fia_screen_iter),
                "screen_top_k": int(args.fia_screen_top_k),
                "screened_ensemble_size": int(args.ensemble_size) if use_screening(args) else None,
                "continued_member_ids": [int(row["member"]) for row in final_seed_plan],
            },
        )
        best = ranked[0]
        print(
            "[quant-rank] selected "
            f"m{int(best['member']):03d} "
            f"mean_relative_l2={float(best['mean_relative_l2_to_observed_quantized_update']):.6g} "
            f"ranking_dir={ranking_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
