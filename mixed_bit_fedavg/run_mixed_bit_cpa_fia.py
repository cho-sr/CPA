#!/usr/bin/env python3
"""Run sparsification-style author CPA + Direct FIA on mixed-bit FL updates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SPARSE_SRC = REPO_ROOT / "sparsified_update_cpa"
if str(SPARSE_SRC) in sys.path:
    sys.path.remove(str(SPARSE_SRC))
sys.path.insert(0, str(SPARSE_SRC))

import author_runner as sparse_author_runner  # noqa: E402
import common as sparse_common  # noqa: E402


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "mixed_bit_fedavg"
    / "imagenet"
    / "vgg16"
    / "clients3_rounds20_spc8_seed42_bits4_8_client_round_bitseed1234"
)
DEFAULT_UPDATE_FILE = DEFAULT_RUN_DIR / "round_020_mixed4_8_dequant_aggregate_cpa.pickle"
MIXED_AUTHOR_RUN_ROOT = REPO_ROOT / "mixed_bit_fedavg" / "outputs_v1" / "author_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update_file", type=Path, default=DEFAULT_UPDATE_FILE)
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
    parser.add_argument(
        "--fia_stop_iter",
        type=int,
        default=None,
        help="Stop Direct FIA early while keeping checkpoint resume compatibility.",
    )
    parser.add_argument("--cpa_only", action="store_true")
    sparse_common.add_attack_args(parser)
    parser.set_defaults(project="cpa_mixed_bit")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int_list(values: Any) -> list[int]:
    array = np.asarray(values)
    return [int(value) for value in array.reshape(-1).tolist()]


def _first_meta(update_data: Mapping[str, Any]) -> dict[str, Any]:
    rows = update_data.get("meta") or []
    return dict(rows[0]) if rows else {}


def _manifest_path(update_file: Path, value: Optional[Path]) -> Path:
    if value is not None:
        return resolve_path(value)
    return update_file.parent / "manifest.json"


def _checkpoint_path(update_file: Path, value: Optional[Path], manifest: Mapping[str, Any]) -> Optional[Path]:
    if value is not None:
        return resolve_path(value)
    manifest_qs = manifest.get("client_qs")
    if isinstance(manifest_qs, Mapping) and manifest_qs.get("source_model_state_file"):
        candidate = Path(str(manifest_qs["source_model_state_file"]))
        return candidate if candidate.is_absolute() else resolve_path(candidate)
    candidate = update_file.parent / "client_qs" / "final_round_global_model_before.pt"
    return candidate if candidate.exists() else None


def _local_steps(meta: Mapping[str, Any]) -> int:
    local_epochs = int(meta.get("local_epochs", meta.get("n_epochs", 1)))
    local_batch_size = int(meta.get("local_batch_size", meta.get("samples_per_client", 1)))
    samples_per_client = int(meta.get("samples_per_client", local_batch_size))
    return local_epochs * int(np.ceil(samples_per_client / max(1, local_batch_size)))


def _param_names(update_data: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    names = update_data.get("param_names") or manifest.get("param_names")
    if not names:
        raise ValueError("param_names not found in update pickle or manifest")
    return [str(name) for name in names]


def _attack_layer(param_names: list[str], manifest: Mapping[str, Any]) -> dict[str, Any]:
    if "attack_layer" in manifest and isinstance(manifest["attack_layer"], Mapping):
        row = dict(manifest["attack_layer"])
        return {"index": int(row["index"]), "name": str(row["name"])}
    index = int(manifest.get("attack_layer_index", 26))
    if index >= len(param_names):
        raise ValueError(f"attack_layer_index={index} is outside param_names")
    return {"index": index, "name": str(manifest.get("attack_layer_name", param_names[index]))}


def _parameter_schema(param_names: list[str], grad_row: list[Any]) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
        for name, value in zip(param_names, grad_row)
    ]


def hydrate_args(
    args: argparse.Namespace,
    *,
    update_data: Mapping[str, Any],
    manifest: Mapping[str, Any],
    checkpoint_path: Optional[Path],
) -> argparse.Namespace:
    meta = _first_meta(update_data)
    args.ds = args.ds or str(manifest.get("ds") or "imagenet")
    args.model = args.model or str(manifest.get("model") or "vgg16")
    args.n_samples = int(args.n_samples or meta.get("n_samples") or manifest.get("n_samples"))
    args.attack_trials = int(args.attack_trials or len(update_data["grad"]))
    args.local_epochs = int(args.local_epochs or meta.get("local_epochs") or meta.get("n_epochs") or 1)
    args.local_batch_size = int(args.local_batch_size or meta.get("local_batch_size") or args.n_samples)
    args.local_lr = float(args.local_lr or meta.get("lr") or manifest.get("lr") or 1e-3)
    args.seed = int(args.seed if args.seed is not None else manifest.get("seed", 42))
    args.global_checkpoint = checkpoint_path
    args.mixed_bit_source_manifest = manifest
    return args


def convert_update_schema(
    *,
    update_data: Mapping[str, Any],
    manifest: Mapping[str, Any],
    update_file: Path,
    checkpoint_path: Optional[Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if "metadata" in update_data and "attack_layer" in update_data and "param_names" in update_data:
        return dict(update_data)

    meta = _first_meta(update_data)
    param_names = _param_names(update_data, manifest)
    attack_layer = _attack_layer(param_names, manifest)
    sample_indices = [
        _as_int_list(row)
        for row in update_data.get("index", [np.arange(args.n_samples, dtype=np.int64)])
    ]
    local_steps = _local_steps(meta)
    observation_scope = str(
        meta.get("observation_scope", manifest.get("observation_scope", "aggregate"))
    )
    observed_client_id = meta.get("observed_client_id", manifest.get("observed_client_id"))
    observes_aggregate = bool(
        meta.get(
            "attacker_observes_round_aggregate",
            manifest.get("attacker_observes_round_aggregate", observation_scope == "aggregate"),
        )
    )
    attacker_observation = str(
        meta.get(
            "attacker_observation",
            manifest.get(
                "attacker_observation",
                "one_client_mixed_bit_dequantized_update_from_fedavg_round"
                if observation_scope == "single_client"
                else "final_round_mixed_bit_dequantized_fedavg_aggregate_update",
            ),
        )
    )

    converted = dict(update_data)
    converted["param_names"] = param_names
    converted["parameter_schema"] = _parameter_schema(param_names, list(update_data["grad"][0]))
    converted["attack_layer"] = attack_layer
    converted["sample_indices"] = sample_indices
    converted["local_batch_orders"] = []
    converted["quant_codes"] = update_data.get("quant_codes")
    converted["quant_stats"] = update_data.get("quant_stats", [])
    converted["whitening_pre_stats"] = []

    metadata = {
        "schema_version": 2,
        "signal": meta.get(
            "signal", manifest.get("signal", "mixed_bit_fedavg_server_dequantized_aggregate_update")
        ),
        "update_definition": meta.get("update_definition") or manifest.get("update_definition"),
        "observation_scope": observation_scope,
        "observed_client_id": observed_client_id,
        "attacker_observes_round_aggregate": observes_aggregate,
        "ds": args.ds,
        "model": args.model,
        "h_dim": int(args.h_dim),
        "n_samples": int(args.n_samples),
        "attack_trials": int(args.attack_trials),
        "rounds": int(meta.get("round", manifest.get("rounds", args.attack_trials))),
        "local_epochs": int(args.local_epochs),
        "local_steps": int(local_steps),
        "local_batch_size": int(args.local_batch_size),
        "micro_batch_size": int(args.local_batch_size),
        "optimizer": "SGD",
        "lr": float(args.local_lr),
        "momentum": 0.0,
        "weight_decay": 0.0,
        "seed": int(args.seed),
        "checkpoint": sparse_common.checkpoint_manifest(checkpoint_path),
        "quant_bits": meta.get("quant_bits", manifest.get("quant_bits", "mixed")),
        "mixed_quant_bits": meta.get("mixed_quant_bits", manifest.get("mixed_quant_bits")),
        "rounding": meta.get("rounding", manifest.get("rounding")),
        "quantization": meta.get("quantization", manifest.get("quantization")),
        "scale_scope": meta.get("scale_scope", manifest.get("scale_scope")),
        "fake_quantization": True,
        "dequantized_before_attack": True,
        "server_aggregation_input": meta.get(
            "server_aggregation_input", manifest.get("server_aggregation_input")
        ),
        "client_update_quantized_before_server_aggregation": bool(
            meta.get(
                "client_update_quantized_before_server_aggregation",
                manifest.get("client_update_quantized_before_server_aggregation", True),
            )
        ),
        "aggregate_requantized": bool(meta.get("aggregate_requantized", manifest.get("aggregate_requantized", False))),
        "bit_selection_seed": meta.get("bit_selection_seed", manifest.get("bit_selection_seed")),
        "bit_selection_granularity": meta.get(
            "bit_selection_granularity", manifest.get("bit_selection_granularity")
        ),
        "final_round_quant_bits_by_client": meta.get(
            "final_round_quant_bits_by_client", manifest.get("final_round_quant_bits_by_client")
        ),
        "observed_client_quant_bits": meta.get(
            "observed_client_quant_bits", manifest.get("observed_client_quant_bits")
        ),
        "aggregation_weights": meta.get("aggregation_weights", manifest.get("aggregation_weights")),
        "fedavg_aggregation_weights": meta.get(
            "fedavg_aggregation_weights", manifest.get("fedavg_aggregation_weights")
        ),
        "clients_per_trial": 1
        if observation_scope == "single_client"
        else int(meta.get("num_clients", manifest.get("num_clients", 1))),
        "fedavg_num_clients": int(meta.get("num_clients", manifest.get("num_clients", 1))),
        "client_granularity": f"{int(meta.get('samples_per_client', 1))}_samples_per_client",
        "attacker_observation": attacker_observation,
        "attack_fc_observation_model": (
            "Y_mixed_client_attack_fc ~= A Z for one mixed-bit FedAvg client update"
            if observation_scope == "single_client"
            else "Y_mixed_attack_fc ~= A Z after mixed-bit FedAvg aggregation"
        ),
        "az_scope": "attack_fc_layer_only_approximate",
        "source_mixed_bit_update_file": str(update_file.resolve()),
        "source_mixed_bit_manifest_file": str((update_file.parent / "manifest.json").resolve()),
    }
    metadata["sample_indices_hash"] = sparse_common.sha256_bytes(
        json.dumps(sample_indices).encode("utf-8")
    )
    metadata["label_hash"] = sparse_common.hash_tensor_sequence(converted["y"])
    metadata["update_tensor_hash"] = sparse_common.hash_tensor_sequence(converted["grad"])
    if converted.get("quant_stats"):
        metadata["quantization_scales_hash"] = sparse_common.sha256_bytes(
            json.dumps(converted["quant_stats"], sort_keys=True).encode("utf-8")
        )
    converted["metadata"] = metadata
    return converted


def run_tag(args: argparse.Namespace, manifest: Mapping[str, Any]) -> str:
    local_steps = int(args.local_epochs)
    granularity = manifest.get("bit_selection_granularity", "mixed")
    bit_seed = manifest.get("bit_selection_seed", args.seed)
    quant_bits = manifest.get("quant_bits", "mixed")
    if isinstance(quant_bits, int) or str(quant_bits).isdigit():
        bit_tag = f"int{int(quant_bits)}"
    else:
        choices = manifest.get("mixed_quant_bits") or manifest.get("quant_bits_choices") or [4, 8]
        bit_tag = "mixed_bits" + "_".join(str(int(value)) for value in choices)
    observation_scope = str(manifest.get("observation_scope", "aggregate"))
    obs_tag = (
        f"client{int(manifest.get('observed_client_id', 0))}"
        if observation_scope == "single_client"
        else "aggregate"
    )
    n_sample_fi = args.n_samples if args.n_sample_fi in {-1, 0} else min(args.n_sample_fi, args.n_samples)
    return (
        f"{bit_tag}_{granularity}_bitseed{bit_seed}_{obs_tag}_"
        f"n{args.n_samples}_steps{local_steps}_trials{args.attack_trials}_fia{n_sample_fi}"
    )


def main() -> None:
    args = parse_args()
    update_file = resolve_path(args.update_file)
    if not update_file.is_file():
        raise FileNotFoundError(update_file)
    manifest_file = _manifest_path(update_file, args.manifest_file)
    manifest = read_json(manifest_file) if manifest_file.is_file() else {}
    checkpoint_path = _checkpoint_path(update_file, args.global_checkpoint, manifest)
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Could not find final-round global model checkpoint. Pass --global_checkpoint "
            "or run the mixed-bit collector with --store_scope attack_layer/all."
        )

    update_data = sparse_common.read_pickle(update_file)
    args = hydrate_args(
        args,
        update_data=update_data,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
    )
    attack_data = convert_update_schema(
        update_data=update_data,
        manifest=manifest,
        update_file=update_file,
        checkpoint_path=checkpoint_path,
        args=args,
    )

    sparse_author_runner.AUTHOR_RUN_ROOT = MIXED_AUTHOR_RUN_ROOT
    exp_name = f"mixed_bit_fedavg_cpa_fia_{run_tag(args, manifest)}"
    attack_meta = attack_data["metadata"]
    observation_scope = str(attack_meta.get("observation_scope", "aggregate"))
    if observation_scope == "single_client":
        method = (
            "Mixed-bit FedAvg CPA/FIA: the FedAvg trajectory uses client updates "
            "randomly quantized to 4bit or 8bit, but the attacker observes one "
            "selected client's dequantized update; author CPA then Direct FIA "
            "follows the sparsification runner path."
        )
        attack_order_policy = "no oracle permutation; mixed-bit single-client update observation order"
    else:
        method = (
            "Mixed-bit FedAvg CPA/FIA: client updates are randomly quantized to "
            "4bit or 8bit before server dequantization and aggregation; author "
            "CPA then Direct FIA follows the sparsification runner path."
        )
        attack_order_policy = "no oracle permutation; mixed-bit aggregate observation order"
    run_path = sparse_author_runner.run_author_cpa_with_update_file(
        args,
        exp_name=exp_name,
        update_file=update_file,
        update_data=attack_data,
        method=method,
        run_fia=not args.cpa_only,
        extra_manifest={
            "mixed_bit_cpa_wrapper": str(Path(__file__).resolve()),
            "source_mixed_bit_manifest": str(manifest_file.resolve()) if manifest_file.exists() else None,
            "final_round_global_checkpoint": str(checkpoint_path.resolve()),
            "attack_order_policy": attack_order_policy,
            "observation_scope": observation_scope,
            "attacker_observation": attack_meta.get("attacker_observation"),
            "attacker_observes_round_aggregate": attack_meta.get("attacker_observes_round_aggregate"),
            "observed_client_id": attack_meta.get("observed_client_id"),
            "observed_client_quant_bits": attack_meta.get("observed_client_quant_bits"),
            "fedavg_num_clients": attack_meta.get("fedavg_num_clients"),
            "fedavg_aggregation_weights": attack_meta.get("fedavg_aggregation_weights"),
        },
        attack_order_policy=attack_order_policy,
    )
    print(f"mixed-bit CPA/FIA output: {run_path}", flush=True)


if __name__ == "__main__":
    main()
