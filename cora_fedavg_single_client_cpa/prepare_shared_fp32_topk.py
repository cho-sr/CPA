#!/usr/bin/env python3
"""Create a matched FP32 control and Top-k observations from one lossless source.

The existing FedAvg runner collects a separate sparse trajectory for each keep
ratio.  This utility instead treats a verified lossless source artifact as the
shared dense update, writes an explicit keep=1 FP32 control, and derives every
requested Top-k observation from that same tensor and checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SPARSE_SRC = REPO_ROOT / "sparsified_update_cpa"
if str(SPARSE_SRC) not in sys.path:
    sys.path.insert(0, str(SPARSE_SRC))

from common import (  # noqa: E402
    build_experiment_model,
    get_device,
    hash_tensor_sequence,
    read_pickle,
    sha256_file,
    write_json,
    write_pickle,
)
from sparsification import format_keep_ratio, validate_keep_ratio  # noqa: E402
from cora_fedavg_single_client_cpa.CoRA_sparsification import (  # noqa: E402
    OBSERVED_ZERO_POLICY,
    set_seed,
    sparse_update_and_masks,
    tau_from_policy,
    train_single_client_update,
    write_observation_shard,
)


DEFAULT_KEEP_RATIOS = [1.0, 0.8, 0.5, 0.3, 0.1]


def parse_keep_ratio(value: str) -> float:
    ratio = float(value)
    if ratio > 1.0:
        ratio /= 100.0
    validate_keep_ratio(ratio)
    return ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_file", type=Path, required=True, help="Lossless sparse artifact used as the dense source.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for keep_<ratio>.pickle artifacts.")
    parser.add_argument("--keep_ratios", type=parse_keep_ratio, nargs="+", default=DEFAULT_KEEP_RATIOS)
    parser.add_argument(
        "--tau_policy",
        choices=["topk_threshold", "min_abs_nonzero_observed"],
        default="topk_threshold",
    )
    parser.add_argument(
        "--allow_nonlossless_source",
        action="store_true",
        help="Allow a source whose recorded sparse-vs-FP32 metrics are not exactly lossless.",
    )
    parser.add_argument(
        "--reconstruct_from_checkpoint",
        action="store_true",
        help=(
            "Recreate the dense client update from the source artifact's saved checkpoint and private batch. "
            "Use this when the source is sparse or its original checkpoint, rather than a lossless update, is required."
        ),
    )
    return parser.parse_args()


def require_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")


def as_dense_update(source_data: Mapping[str, Any]) -> OrderedDict[str, torch.Tensor]:
    if len(source_data.get("grad", [])) != 1:
        raise ValueError("Shared-control preparation currently expects exactly one observed client batch")
    names = list(source_data["param_names"])
    values = list(source_data["grad"][0])
    if len(names) != len(values):
        raise ValueError("Parameter names and observed update tensors do not match")
    return OrderedDict(
        (name, torch.as_tensor(value, dtype=torch.float32).detach().cpu().contiguous())
        for name, value in zip(names, values)
    )


def lossless_source_metadata(source_data: Mapping[str, Any], allow_nonlossless: bool) -> dict[str, Any]:
    metadata = dict(source_data.get("metadata", {}))
    relative_l2 = metadata.get("sparse_relative_l2")
    cosine = metadata.get("sparse_cosine_similarity_to_fp32")
    retained = metadata.get("sparse_retained_l2_norm_ratio")
    is_lossless = relative_l2 == 0.0 and cosine == 1.0 and retained == 1.0
    if not is_lossless and not allow_nonlossless:
        raise ValueError(
            "The source is not recorded as lossless relative to FP32. "
            "Use a keep80 artifact with relative_l2=0, cosine=1, retained_norm=1, "
            "or explicitly pass --allow_nonlossless_source."
        )
    return {
        "recorded_sparse_relative_l2": relative_l2,
        "recorded_sparse_cosine_similarity_to_fp32": cosine,
        "recorded_sparse_retained_l2_norm_ratio": retained,
        "source_recorded_lossless": is_lossless,
    }


def update_comparison(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    diff_sq = 0.0
    reference_sq = 0.0
    candidate_sq = 0.0
    dot = 0.0
    for name, reference_value in reference.items():
        candidate_value = candidate[name]
        diff = candidate_value.float() - reference_value.float()
        diff_sq += float((diff * diff).sum().item())
        reference_sq += float((reference_value.float() * reference_value.float()).sum().item())
        candidate_sq += float((candidate_value.float() * candidate_value.float()).sum().item())
        dot += float((reference_value.float() * candidate_value.float()).sum().item())
    return {
        "relative_l2": float(diff_sq**0.5 / max(reference_sq**0.5, 1e-12)),
        "cosine_similarity": float(dot / max((reference_sq * candidate_sq) ** 0.5, 1e-12)),
    }


def reconstruct_dense_update(source_data: Mapping[str, Any]) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any]]:
    metadata = source_data["metadata"]
    checkpoint_info = metadata.get("checkpoint", {})
    checkpoint_path_text = checkpoint_info.get("path")
    if not checkpoint_path_text:
        raise ValueError("Source artifact has no saved checkpoint for dense-update reconstruction")
    checkpoint_path = Path(checkpoint_path_text)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Saved source checkpoint does not exist: {checkpoint_path}")
    if len(source_data.get("x", [])) != 1 or len(source_data.get("y", [])) != 1:
        raise ValueError("Dense-update reconstruction currently expects exactly one saved private batch")

    local_args = argparse.Namespace(
        model=metadata["model"],
        ds=metadata["ds"],
        h_dim=int(metadata["h_dim"]),
        local_lr=float(metadata["lr"]),
        local_epochs=int(metadata["local_epochs"]),
        local_batch_size=int(metadata["local_batch_size"]),
        micro_batch_size=int(metadata["micro_batch_size"]),
        n_samples=int(metadata["n_samples"]),
    )
    set_seed(int(metadata["seed"]))
    device = get_device()
    global_model = build_experiment_model(
        model_name=local_args.model,
        ds=local_args.ds,
        h_dim=local_args.h_dim,
        device=device,
        checkpoint_path=checkpoint_path,
        seed=int(metadata["seed"]),
    )
    global_model.eval()
    dense_update = train_single_client_update(
        global_model=global_model,
        x_cpu=torch.as_tensor(source_data["x"][0], dtype=torch.float32).detach().cpu(),
        y_cpu=torch.as_tensor(source_data["y"][0], dtype=torch.long).detach().cpu(),
        args=local_args,
        device=device,
        log_prefix="[shared-fp32-replay]",
    )
    source_sparse = as_dense_update(source_data)
    replay_sparse, _stats, _masks = sparse_update_and_masks(dense_update, float(metadata["keep_ratio"]))
    comparison = update_comparison(source_sparse, replay_sparse)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dense_update, {
        "dense_source": "reconstructed_from_saved_checkpoint_and_private_batch",
        "replay_checkpoint": str(checkpoint_path.resolve()),
        "replay_sparse_vs_stored": comparison,
    }


def derived_data(
    *,
    source_data: Mapping[str, Any],
    sparse_update: Mapping[str, torch.Tensor],
    sparse_stats: Mapping[str, Any],
    keep_ratio: float,
    source_file: Path,
    source_file_hash: str,
    source_fp32_hash: str,
    control_id: str,
) -> dict[str, Any]:
    data = copy.deepcopy(source_data)
    data["grad"] = [
        [value.detach().cpu().numpy().astype(np.float32, copy=True) for value in sparse_update.values()]
    ]
    data["sparse_stats"] = [dict(sparse_stats)]
    data["sparse_masks"] = None

    metadata = copy.deepcopy(source_data["metadata"])
    metadata.update(
        {
            "keep_ratio": float(keep_ratio),
            "control_role": "fedavg_single_client_fp32_control" if keep_ratio >= 1.0 else None,
            "sparsified_observation": keep_ratio < 1.0,
            "sparsification_policy": "global_topk",
            "sparsification_application": "derived_offline_from_shared_fp32_control",
            "source_fp32_update_hash": source_fp32_hash,
            "sparse_nonzero_fraction": sparse_stats.get("sparse_nonzero_fraction"),
            "sparse_relative_l2": sparse_stats.get("sparse_relative_l2"),
            "sparse_cosine_similarity_to_fp32": sparse_stats.get("sparse_cosine_similarity_to_fp32"),
            "sparse_retained_l2_norm_ratio": sparse_stats.get("sparse_retained_l2_norm_ratio"),
            "shared_fp32_control": True,
            "shared_fp32_control_id": control_id,
            "shared_fp32_control_file": str(source_file.resolve()),
            "shared_fp32_control_file_hash": source_file_hash,
            "shared_fp32_derivation": "offline_global_topk_from_lossless_source_artifact",
            "observed_zero_policy": OBSERVED_ZERO_POLICY,
            "update_tensor_hash": hash_tensor_sequence(data["grad"]),
        }
    )
    data["metadata"] = metadata
    return data


def build_attack_layer_observation(
    *,
    data: Mapping[str, Any],
    raw_masks: list[torch.Tensor],
    keep_ratio: float,
    tau_policy: str,
    output_file: Path,
) -> dict[str, Any]:
    attack_idx = int(data["attack_layer"]["index"])
    attack_name = str(data["attack_layer"]["name"])
    attack_values = np.asarray(data["grad"][0][attack_idx], dtype=np.float32)
    mask = raw_masks[attack_idx].detach().cpu().numpy().astype(np.bool_, copy=False)
    observed_values = attack_values[mask].astype(np.float32, copy=True)
    stats = data["sparse_stats"][0]
    tau = tau_from_policy(stats, observed_values, tau_policy)
    metadata = copy.deepcopy(data["metadata"])
    metadata.update(
        {
            "method": "shared_fp32_topk_attack_layer_observation_collector",
            "fixed_sparse_observation": True,
            "desparsification_scope": "attack_layer",
            "tau_policy": tau_policy,
            "attacker_observation": "one_client_compressed_update_from_fedavg_round",
            "attacker_observes_round_aggregate": False,
        }
    )
    observation = {
        "schema_version": 1,
        "metadata": metadata,
        "attack_layer": {
            "index": attack_idx,
            "name": attack_name,
            "shape": list(attack_values.shape),
            "dtype": "torch.float32",
        },
        "param_names": list(data["param_names"]),
        "sample_indices": copy.deepcopy(data["sample_indices"]),
        "observations": [],
        "storage": "sharded_npz",
    }
    row = write_observation_shard(
        index_path=output_file,
        batch_idx=0,
        keep_ratio=keep_ratio,
        mask=mask,
        values=observed_values,
        tau=tau,
        tau_policy=tau_policy,
        global_total=int(stats.get("total_elements", 0)),
        global_kept=int(stats.get("mask_kept_elements", 0)),
        selection_mode="shared_fp32_offline_global_topk",
    )
    observation["observations"].append([row])
    return observation


def main() -> None:
    args = parse_args()
    source_file = args.source_file.resolve()
    output_dir = args.output_dir.resolve()
    if not source_file.is_file():
        raise FileNotFoundError(f"Missing source artifact: {source_file}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite output directory: {output_dir}")

    source_data = read_pickle(source_file)
    if args.reconstruct_from_checkpoint:
        dense_update, dense_source_info = reconstruct_dense_update(source_data)
        lossless_info = {
            "recorded_sparse_relative_l2": source_data["metadata"].get("sparse_relative_l2"),
            "recorded_sparse_cosine_similarity_to_fp32": source_data["metadata"].get(
                "sparse_cosine_similarity_to_fp32"
            ),
            "recorded_sparse_retained_l2_norm_ratio": source_data["metadata"].get(
                "sparse_retained_l2_norm_ratio"
            ),
            "source_recorded_lossless": False,
        }
    else:
        lossless_info = lossless_source_metadata(source_data, bool(args.allow_nonlossless_source))
        dense_update = as_dense_update(source_data)
        dense_source_info = {"dense_source": "verified_lossless_source_artifact"}
    source_file_hash = sha256_file(source_file)
    source_fp32_hash = hash_tensor_sequence(list(dense_update.values()))

    ratios = sorted({float(ratio) for ratio in args.keep_ratios}, reverse=True)
    if 1.0 not in ratios:
        ratios.insert(0, 1.0)
    control_id = f"shared_fp32_{source_fp32_hash[:16]}"
    spec = {
        "schema_version": 1,
        "control_id": control_id,
        "source_file": str(source_file),
        "source_file_hash": source_file_hash,
        "source_fp32_update_hash": source_fp32_hash,
        "recorded_source_fp32_update_hash": source_data["metadata"].get("source_fp32_update_hash"),
        "checkpoint": source_data["metadata"].get("checkpoint"),
        "sample_indices_hash": source_data["metadata"].get("sample_indices_hash"),
        "keep_ratios": ratios,
        "tau_policy": args.tau_policy,
        **lossless_info,
        **dense_source_info,
    }

    for ratio in ratios:
        sparse_update, sparse_stats, raw_masks = sparse_update_and_masks(dense_update, ratio)
        data = derived_data(
            source_data=source_data,
            sparse_update=sparse_update,
            sparse_stats=sparse_stats,
            keep_ratio=ratio,
            source_file=source_file,
            source_file_hash=source_file_hash,
            source_fp32_hash=source_fp32_hash,
            control_id=control_id,
        )
        update_file = output_dir / f"keep_{format_keep_ratio(ratio)}.pickle"
        require_new_output(update_file)
        write_pickle(update_file, data)
        write_json(update_file.with_suffix(".manifest.json"), {**spec, "keep_ratio": ratio, "update_file": str(update_file)})

        if ratio < 1.0:
            observation_file = output_dir / f"keep_{format_keep_ratio(ratio)}_attack_layer_observation.pickle"
            require_new_output(observation_file)
            observation = build_attack_layer_observation(
                data=data,
                raw_masks=raw_masks,
                keep_ratio=ratio,
                tau_policy=args.tau_policy,
                output_file=observation_file,
            )
            write_pickle(observation_file, observation)
            write_json(
                observation_file.with_suffix(".manifest.json"),
                {
                    **spec,
                    "keep_ratio": ratio,
                    "observation_file": str(observation_file),
                    "source_sparse_update_file": str(update_file),
                },
            )
        print(
            f"[shared-fp32] keep={ratio:g} relative_l2={data['metadata']['sparse_relative_l2']} "
            f"output={update_file}",
            flush=True,
        )

    write_json(output_dir / "shared_control_manifest.json", spec)
    print(f"[shared-fp32] ready: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
