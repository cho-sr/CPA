#!/usr/bin/env python3
"""CoRA wrapper for fixed sparse-observation CPA.

This runner keeps the sparse observation fixed, generates calibrated randomized
desparsification candidates, runs CPA+Direct FIA independently for each member,
then ranks members by recompressing reconstructed images and comparing them to
the observed sparse attack-layer client updates.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SPARSE_SRC = REPO_ROOT / "sparsified_update_cpa"
if str(SPARSE_SRC) not in sys.path:
    sys.path.insert(0, str(SPARSE_SRC))

from author_runner import run_author_cpa_with_update_file  # noqa: E402
from blind_alignment import _linear_sum_assignment  # noqa: E402
from common import (  # noqa: E402
    AUTHOR_SOURCE_HASH,
    AUTHOR_RUN_ROOT,
    OUTPUT_ROOT as SPARSE_OUTPUT_ROOT,
    _named_update_dict,
    add_attack_args,
    add_collect_args,
    build_experiment_model,
    client_sparse_collector_config,
    collect_client_then_aggregate_sparse_pickles,
    get_device,
    hash_json,
    make_attack_namespace,
    read_pickle,
    run_tag,
    sha256_bytes,
    sha256_file,
    write_json,
    write_pickle,
)
from metrics import write_csv  # noqa: E402
from sparsification import (  # noqa: E402
    RANDOMIZED_DESPARSE_POLICY,
    format_keep_ratio,
    parse_keep_ratio_or_percent,
    require_global_topk,
    sparsify_update_global_topk,
    unpack_bool_mask,
    validate_keep_ratio,
)
import utils as author_utils  # noqa: E402


EXPERIMENT_NAME = "cora_sparse_cpa"
BASELINE_EXPERIMENT_NAME = "01_naive_sparse_cpa"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs_v1"
DEFAULT_KEEP_RATIOS = [0.30]
IMAGE_MEAN_STD = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "tiny_imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_collect_args(parser)
    parser.add_argument("--keep_ratios", type=parse_keep_ratio_or_percent, nargs="+", default=DEFAULT_KEEP_RATIOS)
    parser.add_argument("--sparse_policy", type=str, default="global_topk", choices=["global_topk"])
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--noise_seed", type=int, default=1234)
    parser.add_argument(
        "--desparse_sampler",
        type=str,
        default="retained_rescaled",
        choices=["retained_rescaled", "uniform_tau"],
        help=(
            "Calibrated missing-coordinate sampler. retained_rescaled uses the retained "
            "attack-layer magnitude/sign distribution rescaled into [0, tau]; uniform_tau "
            "uses U(-tau, tau)."
        ),
    )
    parser.add_argument("--success_lpips_threshold", type=float, default=0.50)
    parser.add_argument(
        "--consistency_label_mode",
        type=str,
        default="model_pred",
        choices=["model_pred", "zero"],
        help="Blind label policy used only for recompression consistency scoring.",
    )
    parser.add_argument(
        "--consistency_max_recon_per_batch",
        type=int,
        default=0,
        help="Score at most this many reconstructions per batch; 0 means all reconstructed FIA images.",
    )
    parser.add_argument(
        "--consistency_max_clients",
        type=int,
        default=0,
        help="Compare against at most this many observed clients per batch; 0 means all clients.",
    )
    parser.add_argument(
        "--skip_consistency_ranking",
        action="store_true",
        help="Run member CPA+FIA only and skip CoRA ranking.",
    )
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
        SPARSE_OUTPUT_ROOT
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
        SPARSE_OUTPUT_ROOT
        / "02_randomized_desparse_ensemble_cpa"
        / args.ds
        / args.model
        / "fixed_observations"
        / client_sparse_group_tag(args)
        / "client_then_aggregate"
        / args.sparse_policy
        / f"keep_{ratio_tag}_attack_layer_observations.pickle"
    )


def cora_output_dir(args: argparse.Namespace, keep_ratio: float) -> Path:
    ratio_tag = format_keep_ratio(keep_ratio)
    return (
        OUTPUT_ROOT
        / args.ds
        / args.model
        / client_sparse_group_tag(args)
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


def completed_member_run_path(args: argparse.Namespace, exp_name: str) -> Path | None:
    author_utils.exp_path_base = str(AUTHOR_RUN_ROOT)
    attack_args = make_attack_namespace(args, exp_name=exp_name)
    run_path = Path(author_utils.get_attack_exp_path(attack_args))
    if (run_path / "reconstructions.pkl").exists():
        return run_path
    return None


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
    sampler: str,
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
    elif sampler == "uniform_tau":
        sampled = rng.uniform(-tau, tau, size=shape).astype(np.float32)
    elif sampler == "retained_rescaled":
        observed_abs = np.abs(values.astype(np.float32, copy=False))
        if observed_abs.size == 0:
            sampled = rng.uniform(-tau, tau, size=shape).astype(np.float32)
        else:
            abs_min = float(observed_abs.min())
            abs_max = float(observed_abs.max())
            if abs_max > abs_min:
                retained_unit = ((observed_abs - abs_min) / (abs_max - abs_min)).astype(np.float64, copy=False)
                retained_mean = float(np.mean(retained_unit))
                retained_var = float(np.var(retained_unit))
                beta_scale = retained_mean * (1.0 - retained_mean) / max(retained_var, 1e-12) - 1.0
                if beta_scale > 0.0 and np.isfinite(beta_scale):
                    alpha = max(retained_mean * beta_scale, 1e-3)
                    beta = max((1.0 - retained_mean) * beta_scale, 1e-3)
                    magnitudes = (rng.beta(alpha, beta, size=shape).astype(np.float32) * tau).astype(
                        np.float32,
                        copy=False,
                    )
                else:
                    magnitudes = rng.uniform(0.0, tau, size=shape).astype(np.float32)
            else:
                magnitudes = rng.uniform(0.0, tau, size=shape).astype(np.float32)
            positive_fraction = float(np.mean(values > 0.0)) if values.size else 0.5
            sign_draws = rng.random(size=shape) < positive_fraction
            magnitudes[~sign_draws] *= -1.0
            sampled = magnitudes
    else:
        raise ValueError(f"unknown desparse sampler: {sampler}")
    sampled[mask] = values
    return sampled


def randomized_attack_layer_aggregate(
    observation_data: Mapping[str, Any],
    *,
    batch_idx: int,
    seed: int,
    sampler: str,
    progress_prefix: str | None = None,
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
    for row_idx, row in enumerate(rows):
        if progress_prefix and (row_idx == 0 or (row_idx + 1) % 8 == 0 or row_idx + 1 == len(rows)):
            print(f"{progress_prefix} client {row_idx + 1}/{len(rows)}", flush=True)
        client_sample = _randomized_client_attack_layer(
            row,
            shape=shape,
            rng=rng,
            sampler=sampler,
            observation_index=observation_index,
        )
        aggregate += client_sample * float(row["aggregation_weight"])
        tau_values.append(float(row["tau"]))
        observed_fractions.append(float(row["observed_elements"]) / max(1.0, float(row["total_elements"])))
    tau_array = np.asarray(tau_values, dtype=np.float64)
    observed_array = np.asarray(observed_fractions, dtype=np.float64)
    return aggregate.astype(np.float32, copy=False), {
        "batch": int(batch_idx),
        "clients": len(rows),
        "desparsification_seed": int(seed),
        "sampler": sampler,
        "missing_value_abs_bound_policy": "per_client_global_topk_min_observed_abs_tau",
        "tau_mean": float(tau_array.mean()),
        "tau_min": float(tau_array.min()),
        "tau_max": float(tau_array.max()),
        "attack_layer_observed_fraction_mean": float(observed_array.mean()),
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
    desparse_sampler: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attack_layer_index = int(base_data["attack_layer"]["index"])
    member_grad = []
    attack_layers = []
    batch_stats = []
    total_batches = len(base_data["grad"])
    for batch_idx, batch_grad in enumerate(base_data["grad"]):
        print(
            f"[CoRA-materialize] keep={keep_ratio:g} member {member_idx + 1} "
            f"batch {batch_idx + 1}/{total_batches} start",
            flush=True,
        )
        batch_values = list(batch_grad)
        if observation_data is None:
            attack_layer = np.asarray(batch_values[attack_layer_index], dtype=np.float32)
            stats = {
                "batch": int(batch_idx),
                "desparsification_seed": int(member_seed + 9_176_521 * batch_idx),
                "sampler": "deterministic_identity",
                "keep_ratio": float(keep_ratio),
            }
        else:
            attack_layer, stats = randomized_attack_layer_aggregate(
                observation_data,
                batch_idx=batch_idx,
                seed=int(member_seed + 9_176_521 * batch_idx),
                sampler=desparse_sampler,
                progress_prefix=(
                    f"[CoRA-materialize] keep={keep_ratio:g} member {member_idx + 1} "
                    f"batch {batch_idx + 1}/{total_batches}"
                ),
            )
        batch_values[attack_layer_index] = attack_layer
        member_grad.append(batch_values)
        attack_layers.append(attack_layer)
        batch_stats.append(stats)
        print(
            f"[CoRA-materialize] keep={keep_ratio:g} member {member_idx + 1} "
            f"batch {batch_idx + 1}/{total_batches} done",
            flush=True,
        )

    attack_layer_hash = _hash_arrays(attack_layers)
    metadata = copy.deepcopy(base_data["metadata"])
    metadata.update(
        {
            "cora_candidate_generation": "fixed_sparse_observation_randomized_desparsification",
            "randomized_desparsification": observation_data is not None,
            "randomized_desparsification_policy": RANDOMIZED_DESPARSE_POLICY,
            "fixed_sparse_observation": True,
            "desparsification_scope": "attack_layer",
            "missing_value_sampler": desparse_sampler,
            "missing_value_abs_bound_policy": "per_client_global_topk_min_observed_abs_tau",
            "ensemble_member": int(member_idx),
            "ensemble_desparsification_seed": int(member_seed),
            "source_sparse_update_file": str(base_update_file.resolve()),
            "source_sparse_update_hash": base_update_file_hash,
            "source_client_attack_observation_file": str(observation_file.resolve()) if observation_file else None,
            "source_client_attack_observation_hash": observation_file_hash,
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


def normalize_unit_image(x_unit: torch.Tensor, ds: str) -> torch.Tensor:
    if ds not in IMAGE_MEAN_STD:
        raise ValueError(f"Unsupported image normalization dataset for CoRA ranking: {ds}")
    mean_values, std_values = IMAGE_MEAN_STD[ds]
    mean = torch.tensor(mean_values, device=x_unit.device, dtype=x_unit.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std_values, device=x_unit.device, dtype=x_unit.dtype).view(1, 3, 1, 1)
    return (x_unit.clamp(0.0, 1.0) - mean) / std


def choose_consistency_label(
    *,
    mode: str,
    global_model: torch.nn.Module,
    x_norm: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if mode == "zero":
        return torch.zeros(1, dtype=torch.long, device=x_norm.device), {"label_mode": mode, "label": 0}
    if mode == "model_pred":
        with torch.no_grad():
            logits = global_model(x_norm)
            label = int(torch.argmax(logits, dim=1).detach().cpu().item())
            confidence = float(torch.softmax(logits, dim=1).max(dim=1).values.detach().cpu().item())
        return (
            torch.tensor([label], dtype=torch.long, device=x_norm.device),
            {"label_mode": mode, "label": label, "confidence": confidence},
        )
    raise ValueError(f"unknown consistency label mode: {mode}")


def recompress_reconstruction_attack_layer(
    *,
    args: argparse.Namespace,
    global_model: torch.nn.Module,
    x_unit_np: np.ndarray,
    keep_ratio: float,
    attack_layer_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = next(global_model.parameters()).device
    x_unit = torch.as_tensor(x_unit_np, device=device, dtype=torch.float32).unsqueeze(0)
    x_norm = normalize_unit_image(x_unit, args.ds)
    y, label_meta = choose_consistency_label(
        mode=args.consistency_label_mode,
        global_model=global_model,
        x_norm=x_norm,
    )

    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=args.local_lr, momentum=0.0, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    for _epoch in range(int(args.local_epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(local_model(x_norm), y)
        loss.backward()
        optimizer.step()

    update = OrderedDict(
        (name, value.detach().cpu().float().contiguous())
        for name, value in _named_update_dict(local_model, global_model).items()
    )
    sparse_update, _packed_masks, sparse_stats = sparsify_update_global_topk(
        update,
        keep_ratio=keep_ratio,
        pack_masks=False,
    )
    attack = sparse_update[attack_layer_name].detach().cpu().numpy().astype(np.float32, copy=False)
    meta = {
        **label_meta,
        "topk_threshold_abs": sparse_stats.get("topk_threshold_abs"),
        "sparse_nonzero_fraction": sparse_stats.get("sparse_nonzero_fraction"),
        "sparse_mask_fraction": sparse_stats.get("sparse_mask_fraction"),
    }
    del local_model, update, sparse_update
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return attack, meta


def sparse_distance_to_observation(
    pred_attack: np.ndarray,
    observation: Mapping[str, Any],
    *,
    observation_index: Path | None,
    pred_norm_sq: float | None = None,
    observed_norm_sq: float | None = None,
) -> dict[str, float]:
    pred_flat = np.asarray(pred_attack, dtype=np.float32).reshape(-1)
    packed_mask, values = _load_observation_payload(observation, observation_index=observation_index)
    mask = unpack_bool_mask(packed_mask).reshape(-1)
    values_flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if int(mask.sum()) != int(values_flat.size):
        raise ValueError("observation mask/value size mismatch while scoring")
    if pred_norm_sq is None:
        pred_norm_sq = float(np.dot(pred_flat.astype(np.float64), pred_flat.astype(np.float64)))
    if observed_norm_sq is None:
        observed_norm_sq = float(np.dot(values_flat.astype(np.float64), values_flat.astype(np.float64)))
    dot = float(np.dot(pred_flat[mask].astype(np.float64), values_flat.astype(np.float64)))
    sq = max(pred_norm_sq + observed_norm_sq - 2.0 * dot, 0.0)
    l2 = math.sqrt(sq)
    rel_l2 = l2 / max(math.sqrt(observed_norm_sq), 1e-12)
    return {
        "distance_l2": float(l2),
        "relative_l2": float(rel_l2),
        "pred_norm": float(math.sqrt(max(pred_norm_sq, 0.0))),
        "observed_norm": float(math.sqrt(max(observed_norm_sq, 0.0))),
    }


def observation_norm_sq(
    observation: Mapping[str, Any],
    *,
    observation_index: Path | None,
) -> float:
    _packed_mask, values = _load_observation_payload(observation, observation_index=observation_index)
    values_flat = np.asarray(values, dtype=np.float32).reshape(-1)
    return float(np.dot(values_flat.astype(np.float64), values_flat.astype(np.float64)))


def score_member_run(
    *,
    args: argparse.Namespace,
    run_path: Path,
    member_idx: int,
    keep_ratio: float,
    observation_data: Mapping[str, Any],
    global_model: torch.nn.Module,
    attack_layer_name: str,
    n_batches: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    recon_entries = read_pickle(run_path / "reconstructions.pkl")
    observation_index_value = observation_data.get("_index_path")
    observation_index = Path(str(observation_index_value)) if observation_index_value else None
    match_rows: list[dict[str, Any]] = []
    selected_costs: list[float] = []

    for batch_idx, entry in enumerate(recon_entries[:n_batches]):
        rec_fi = np.asarray(entry.get("rec_fi_attack_order"), dtype=np.float32)
        if rec_fi.size == 0:
            raise ValueError(
                f"{run_path} has no rec_fi_attack_order; CoRA ranking requires member FIA "
                "with --n_iter_fi > 0"
            )
        n_rec = int(rec_fi.shape[0])
        if args.consistency_max_recon_per_batch > 0:
            n_rec = min(n_rec, int(args.consistency_max_recon_per_batch))
        observed_rows = list(observation_data["observations"][int(entry["batch"])])
        if args.consistency_max_clients > 0:
            observed_rows = observed_rows[: int(args.consistency_max_clients)]
        if not observed_rows:
            raise ValueError(f"no observed sparse clients for batch {entry['batch']}")

        print(
            f"[CoRA-rank] member {member_idx + 1} batch {batch_idx + 1}/{n_batches} "
            f"recon={n_rec} clients={len(observed_rows)}",
            flush=True,
        )
        observed_norm_cache = [
            observation_norm_sq(obs, observation_index=observation_index)
            for obs in observed_rows
        ]
        cost = np.zeros((n_rec, len(observed_rows)), dtype=np.float64)
        candidate_meta: list[dict[str, Any]] = []
        distance_meta: list[list[dict[str, float]]] = []
        for rec_sample in range(n_rec):
            print(
                f"[CoRA-rank] member {member_idx + 1} batch {batch_idx + 1}/{n_batches} "
                f"recon {rec_sample + 1}/{n_rec}",
                flush=True,
            )
            pred_attack, meta = recompress_reconstruction_attack_layer(
                args=args,
                global_model=global_model,
                x_unit_np=rec_fi[rec_sample],
                keep_ratio=keep_ratio,
                attack_layer_name=attack_layer_name,
            )
            candidate_meta.append(meta)
            row_distances = []
            pred_flat = np.asarray(pred_attack, dtype=np.float32).reshape(-1)
            pred_norm_sq = float(np.dot(pred_flat.astype(np.float64), pred_flat.astype(np.float64)))
            for obs_idx, obs in enumerate(observed_rows):
                dist = sparse_distance_to_observation(
                    pred_attack,
                    obs,
                    observation_index=observation_index,
                    pred_norm_sq=pred_norm_sq,
                    observed_norm_sq=observed_norm_cache[obs_idx],
                )
                cost[rec_sample, obs_idx] = dist["relative_l2"]
                row_distances.append(dist)
            distance_meta.append(row_distances)
            del pred_attack

        rec_rows, obs_cols = _linear_sum_assignment(cost)
        sample_indices = [int(value) for value in entry.get("sample_indices", [])]
        for rec_row, obs_col in zip(rec_rows.tolist(), obs_cols.tolist()):
            obs = observed_rows[obs_col]
            dist = distance_meta[rec_row][obs_col]
            selected_costs.append(float(cost[rec_row, obs_col]))
            client = int(obs.get("client", obs_col))
            match_rows.append(
                {
                    "member": int(member_idx),
                    "batch": int(entry["batch"]),
                    "rec_sample": int(rec_row),
                    "matched_client": client,
                    "matched_sample_index": sample_indices[client] if client < len(sample_indices) else None,
                    "relative_l2": float(cost[rec_row, obs_col]),
                    "distance_l2": dist["distance_l2"],
                    "pred_norm": dist["pred_norm"],
                    "observed_norm": dist["observed_norm"],
                    "label_mode": candidate_meta[rec_row].get("label_mode"),
                    "label": candidate_meta[rec_row].get("label"),
                    "label_confidence": candidate_meta[rec_row].get("confidence"),
                    "candidate_topk_threshold_abs": candidate_meta[rec_row].get("topk_threshold_abs"),
                    "candidate_sparse_nonzero_fraction": candidate_meta[rec_row].get("sparse_nonzero_fraction"),
                    "source_run_path": str(run_path.resolve()),
                }
            )

    mean_relative_l2 = float(np.mean(selected_costs)) if selected_costs else None
    summary = {
        "member": int(member_idx),
        "score": -mean_relative_l2 if mean_relative_l2 is not None else None,
        "mean_relative_l2": mean_relative_l2,
        "median_relative_l2": float(np.median(selected_costs)) if selected_costs else None,
        "min_relative_l2": float(np.min(selected_costs)) if selected_costs else None,
        "max_relative_l2": float(np.max(selected_costs)) if selected_costs else None,
        "matched_pairs": int(len(selected_costs)),
        "source_run_path": str(run_path.resolve()),
    }
    return summary, match_rows


def rank_members_by_consistency(
    *,
    args: argparse.Namespace,
    keep_ratio: float,
    sparse_file: Path,
    observation_file: Path,
    observation_data: Mapping[str, Any],
    run_paths: Sequence[Path],
) -> Path:
    base_data = read_pickle(sparse_file)
    n_batches = min(args.attack_n_batch or int(base_data["metadata"]["attack_trials"]), int(base_data["metadata"]["attack_trials"]))
    device = get_device()
    torch.manual_seed(int(args.seed))
    global_model = build_experiment_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        device=device,
        checkpoint_path=args.global_checkpoint,
        seed=args.seed,
    )
    global_model.eval()
    attack_layer_name = str(base_data["attack_layer"]["name"])

    summaries: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    for member_idx, run_path in enumerate(run_paths):
        print(f"[CoRA-rank] keep={keep_ratio:g} member {member_idx + 1}/{len(run_paths)}", flush=True)
        summary, rows = score_member_run(
            args=args,
            run_path=run_path,
            member_idx=member_idx,
            keep_ratio=keep_ratio,
            observation_data=observation_data,
            global_model=global_model,
            attack_layer_name=attack_layer_name,
            n_batches=n_batches,
        )
        summaries.append(summary)
        matches.extend(rows)

    ranked = sorted(
        summaries,
        key=lambda row: float(row["mean_relative_l2"]) if row["mean_relative_l2"] is not None else float("inf"),
    )
    if not ranked:
        raise ValueError("no members were scored")
    best = ranked[0]
    output_dir = cora_output_dir(args, keep_ratio)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "consistency_member_ranking.csv", ranked)
    write_csv(output_dir / "consistency_matches.csv", matches)
    best_path = Path(str(best["source_run_path"]))
    selected_recon = read_pickle(best_path / "reconstructions.pkl")
    write_pickle(
        output_dir / "selected_reconstructions.pkl",
        {
            "selected_member": int(best["member"]),
            "selection_score": best,
            "source_run_path": str(best_path.resolve()),
            "reconstructions": selected_recon,
        },
    )
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "method": "CoRA sparse wrapper: calibrated candidates, independent CPA+FIA restarts, recompression consistency ranking",
            "source_sparse_update_file": str(sparse_file.resolve()),
            "source_sparse_update_hash": sha256_file(sparse_file),
            "source_client_attack_observation_file": str(observation_file.resolve()),
            "source_client_attack_observation_hash": sha256_file(observation_file),
            "source_run_paths": [str(path.resolve()) for path in run_paths],
            "selected_member": int(best["member"]),
            "selected_source_run_path": str(best_path.resolve()),
            "keep_ratio": float(keep_ratio),
            "ensemble_size": int(args.ensemble_size),
            "noise_seed": int(args.noise_seed),
            "candidate_generation": {
                "policy": RANDOMIZED_DESPARSE_POLICY,
                "fixed_sparse_observation": True,
                "desparsification_scope": "attack_layer",
                "missing_value_sampler": args.desparse_sampler,
                "missing_value_abs_bound_policy": "per_client_global_topk_min_observed_abs_tau",
            },
            "independent_restarts": {
                "attack": "existing CPA wrapped by CoRA",
                "fia": "Direct FIA per member before ranking",
                "trial_varying_randomness": "desparsification_noise_only",
            },
            "consistency_ranking": {
                "score": "negative mean matched relative L2",
                "recompression": "x_hat -> local SGD update -> same global Top-k sparsification -> attack-layer sparse observation",
                "matching": "Hungarian assignment between reconstructed images and observed sparse client attack-layer updates",
                "compression_operator": "global_topk",
                "label_mode": args.consistency_label_mode,
                "oracle_inputs_used": False,
                "max_recon_per_batch": int(args.consistency_max_recon_per_batch),
                "max_clients": int(args.consistency_max_clients),
            },
            "best_member_summary": best,
        },
    )
    return output_dir


def main() -> None:
    args = parse_args()
    require_global_topk(args.sparse_policy)
    for keep_ratio in args.keep_ratios:
        validate_keep_ratio(float(keep_ratio))
    if args.ensemble_size < 1:
        raise ValueError("--ensemble_size must be >= 1")
    if args.n_iter_fi <= 0 and args.run_attack and not args.skip_consistency_ranking:
        raise ValueError("CoRA consistency ranking requires --n_iter_fi > 0 so each member has reconstructions")
    args.observed_zero_policy = RANDOMIZED_DESPARSE_POLICY
    args.sparsification_application = "client_then_aggregate"
    args.client_granularity = "one_sample_per_client"

    output_files = {float(keep_ratio): sparse_update_path(args, float(keep_ratio)) for keep_ratio in args.keep_ratios}
    observation_files = {
        float(keep_ratio): observation_path(args, float(keep_ratio))
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
            print(f"deterministic sparse artifact keep={ratio:g}: {sparse_file}", flush=True)
            if observation_file is not None:
                print(f"fixed attack-layer observation keep={ratio:g}: {observation_file}", flush=True)
            continue
        if observation_file is None:
            raise ValueError("CoRA sparse ranking currently requires keep_ratio < 1.0 fixed observations")

        base_data = read_pickle(sparse_file)
        observation_data = read_pickle(observation_file)
        observation_data["_index_path"] = str(observation_file)
        sparse_file_hash = sha256_file(sparse_file)
        observation_file_hash = sha256_file(observation_file)
        run_paths = []
        for seed_row in seed_plan:
            member_idx = int(seed_row["member"])
            member_seed = int(seed_row["desparsification_seed"])
            print(
                f"[CoRA-member] keep={ratio:g} member {member_idx + 1}/{args.ensemble_size} "
                f"desparse_seed={member_seed} start",
                flush=True,
            )
            exp_name = (
                f"{EXPERIMENT_NAME}_keep{ratio_tag}_{run_tag(args)}"
                f"_R{args.ensemble_size}_seed{args.noise_seed}_m{member_idx:03d}"
            )
            completed_path = completed_member_run_path(args, exp_name)
            if completed_path is not None:
                run_path = completed_path
                run_paths.append(run_path)
                print(
                    f"[CoRA-member] keep={ratio:g} member {member_idx + 1}/{args.ensemble_size} "
                    f"resume-skip existing reconstructions: {run_path}",
                    flush=True,
                )
                print(
                    f"[CoRA-member] keep={ratio:g} member {member_idx + 1}/{args.ensemble_size} done: {run_path}",
                    flush=True,
                )
                continue
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
                desparse_sampler=args.desparse_sampler,
            )
            run_path = run_author_cpa_with_update_file(
                args,
                exp_name=exp_name,
                update_file=sparse_file,
                update_data=member_data,
                run_fia=True,
                method=(
                    "CoRA member: fixed sparse observation randomized desparsification, "
                    "CPA, and Direct FIA before recompression consistency ranking."
                ),
                extra_manifest={
                    "cora_wrapper": True,
                    "cora_stage": "independent_member_restart",
                    "update_file_role": "baseline sparse context artifact; attack layer overridden in memory",
                    "randomized_desparsification_policy": RANDOMIZED_DESPARSE_POLICY,
                    "desparsification_scope": "attack_layer",
                    "missing_value_sampler": args.desparse_sampler,
                    "missing_value_abs_bound_policy": "per_client_global_topk_min_observed_abs_tau",
                    "ensemble_member": member_idx,
                    "ensemble_size": int(args.ensemble_size),
                    "ensemble_noise_seed": int(args.noise_seed),
                    "ensemble_desparsification_seed": member_seed,
                    "desparsification_batch_stats": batch_stats,
                    "consistency_ranking_policy": (
                        "post-hoc blind recompression score; not used during this member CPA/FIA optimization"
                    ),
                },
                attack_order_policy=(
                    "member CPA/FIA order; CoRA ranking later uses sparse-observation consistency, not oracle labels"
                ),
            )
            run_paths.append(run_path)
            print(
                f"[CoRA-member] keep={ratio:g} member {member_idx + 1}/{args.ensemble_size} done: {run_path}",
                flush=True,
            )

        if args.skip_consistency_ranking:
            print("CoRA member runs complete; consistency ranking skipped.", flush=True)
            continue

        output_dir = rank_members_by_consistency(
            args=args,
            keep_ratio=ratio,
            sparse_file=sparse_file,
            observation_file=observation_file,
            observation_data=observation_data,
            run_paths=run_paths,
        )
        print(f"CoRA selected artifact keep={ratio:g}: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
