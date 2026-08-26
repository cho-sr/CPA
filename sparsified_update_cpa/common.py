"""Shared infrastructure for sparse-update CPA experiments.

The sparse experiments are intentionally separated from ``quantized_update_cpa``
so in-progress quantization files are not modified.  The immutable author CPA
snapshot is read from the local ``official_cpa/src`` checkout when present, with
a fallback to the tracked ``cocktail_party_attack/src`` snapshot.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import shutil
import sys
import types
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

for _thread_env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_var, "1")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

try:
    torch.set_num_threads(1)
except RuntimeError:
    pass
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


REPO_ROOT = Path(__file__).resolve().parents[1]
CPA_SRC_CANDIDATES = (
    REPO_ROOT / "official_cpa" / "src",
    REPO_ROOT / "cocktail_party_attack" / "src",
    REPO_ROOT / "src",
)
CPA_SRC = next((candidate for candidate in CPA_SRC_CANDIDATES if candidate.is_dir()), CPA_SRC_CANDIDATES[0])
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs_v1"
SHARED_UPDATE_ROOT = OUTPUT_ROOT / "shared_updates"
AUTHOR_RUN_ROOT = OUTPUT_ROOT / "author_runs"
DATASET_ROOT = REPO_ROOT / "datasets"
AUTO_VGG16_IMAGENET_COLLECT_MICRO_BATCH_SIZE = 4
AUTO_VGG16_IMAGENET_FIA_MICRO_BATCH_SIZE = 2

if not CPA_SRC.is_dir():
    searched = ", ".join(str(candidate) for candidate in CPA_SRC_CANDIDATES)
    raise FileNotFoundError(f"Author CPA snapshot not found. Searched: {searched}")

_cpa_src_str = str(CPA_SRC)
if _cpa_src_str in sys.path:
    sys.path.remove(_cpa_src_str)
sys.path.insert(0, _cpa_src_str)


def _install_image_only_torchtext_stub() -> None:
    """Avoid importing broken torchtext binaries for image-only CPA datasets."""
    if "torchtext" in sys.modules:
        return

    torchtext_mod = types.ModuleType("torchtext")
    data_mod = types.ModuleType("torchtext.data")
    data_utils_mod = types.ModuleType("torchtext.data.utils")
    data_functional_mod = types.ModuleType("torchtext.data.functional")
    vocab_mod = types.ModuleType("torchtext.vocab")

    def get_tokenizer(*_args, **_kwargs):
        return lambda text: str(text).split()

    def to_map_style_dataset(dataset):
        return list(dataset)

    class GloVe:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("torchtext is stubbed because sparse CPA uses image datasets only")

    data_utils_mod.get_tokenizer = get_tokenizer
    data_functional_mod.to_map_style_dataset = to_map_style_dataset
    vocab_mod.GloVe = GloVe
    data_mod.utils = data_utils_mod
    data_mod.functional = data_functional_mod
    torchtext_mod.data = data_mod
    torchtext_mod.vocab = vocab_mod

    sys.modules["torchtext"] = torchtext_mod
    sys.modules["torchtext.data"] = data_mod
    sys.modules["torchtext.data.utils"] = data_utils_mod
    sys.modules["torchtext.data.functional"] = data_functional_mod
    sys.modules["torchtext.vocab"] = vocab_mod


_install_image_only_torchtext_stub()

import datasets as cpa_datasets  # noqa: E402
from datasets import get_dataloaders  # noqa: E402
from models import get_model  # noqa: E402
from utils import get_device  # noqa: E402

cpa_datasets.ds_root = str(DATASET_ROOT)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


AUTHOR_SOURCE_HASH = source_tree_hash(CPA_SRC)


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


def hash_json(value: Mapping[str, Any]) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_matching_manifest(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.exists():
        return
    current = read_json(path)
    if current != expected:
        raise RuntimeError(
            f"Manifest mismatch at {path}. Refusing to reuse an artifact generated "
            "with different data, source, or optimization settings. Use a new output "
            "directory or --fresh_start."
        )


def safe_rmtree(path: Path, *, allowed_root: Path) -> None:
    resolved_root = allowed_root.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing to delete outside experiment root: {resolved_path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def _checkpoint_state_dict(checkpoint_path: Optional[Path]) -> Optional[Dict[str, torch.Tensor]]:
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def checkpoint_manifest(checkpoint_path: Optional[Path]) -> Dict[str, Optional[str]]:
    if checkpoint_path is None:
        return {"path": None, "sha256": None}
    return {"path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path)}


def build_experiment_model(
    *,
    model_name: str,
    ds: str,
    h_dim: int,
    device: torch.device,
    checkpoint_path: Optional[Path] = None,
    seed: Optional[int] = None,
) -> torch.nn.Module:
    if checkpoint_path is None and seed is not None:
        torch.manual_seed(seed)
    state_dict = _checkpoint_state_dict(checkpoint_path)
    model = get_model(
        model_name=model_name,
        ds=ds,
        h_dim=h_dim,
        load_path=None,
        dataparallel=False,
    )
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model.to(device=device, dtype=torch.float32)


def _named_update_dict(
    local_model: torch.nn.Module,
    global_model: torch.nn.Module,
) -> "OrderedDict[str, torch.Tensor]":
    local_params = OrderedDict(local_model.named_parameters())
    global_params = OrderedDict(global_model.named_parameters())
    if list(local_params) != list(global_params):
        raise ValueError("Local/global parameter names do not match")
    return OrderedDict(
        (name, (local_params[name].detach() - global_params[name].detach()).float())
        for name in global_params
    )


def parameter_schema(model: torch.nn.Module) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
        for name, parameter in model.named_parameters()
    ]


def validate_parameter_mapping(names: Sequence[str], values: Sequence[Any], model: torch.nn.Module) -> None:
    expected = list(model.named_parameters())
    if len(names) != len(expected) or len(values) != len(expected):
        raise ValueError("Parameter count does not match collector/model mapping")
    for index, (name, value, (expected_name, parameter)) in enumerate(zip(names, values, expected)):
        tensor = torch.as_tensor(value)
        if name != expected_name:
            raise ValueError(f"Parameter {index}: expected name {expected_name}, got {name}")
        if tuple(tensor.shape) != tuple(parameter.shape):
            raise ValueError(
                f"Parameter {name}: expected shape {tuple(parameter.shape)}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.float32:
            raise ValueError(f"Parameter {name}: expected stored dtype float32, got {tensor.dtype}")


def _sample_trial_indices(dataset_size: int, n_samples: int, attack_trials: int, seed: int) -> list[list[int]]:
    if n_samples > dataset_size:
        raise ValueError(f"n_samples={n_samples} exceeds dataset size={dataset_size}")
    result: list[list[int]] = []
    for trial in range(attack_trials):
        generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003 * trial)
        result.append(torch.randperm(dataset_size, generator=generator)[:n_samples].tolist())
    return result


def _load_indexed_test_batches(
    *, ds: str, n_samples: int, attack_trials: int, seed: int
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[int]]]:
    _, dl_test = get_dataloaders(ds, batch_size=1, shuffle_test=False, num_workers=0)
    indices_by_trial = _sample_trial_indices(len(dl_test.dataset), n_samples, attack_trials, seed)
    xs, ys = [], []
    for indices in indices_by_trial:
        loader = DataLoader(
            Subset(dl_test.dataset, indices), batch_size=n_samples, shuffle=False, num_workers=0
        )
        x, y = next(iter(loader))
        xs.append(x)
        ys.append(y)
    return xs, ys, indices_by_trial


def _local_batch_orders(n_samples: int, local_batch_size: int, local_epochs: int) -> list[list[int]]:
    if local_batch_size < 1 or local_batch_size > n_samples:
        raise ValueError("local_batch_size must be in [1, n_samples]")
    base = list(range(n_samples))
    one_epoch = [base[start : start + local_batch_size] for start in range(0, n_samples, local_batch_size)]
    return [batch for _ in range(local_epochs) for batch in one_epoch]


def collector_config(
    *,
    ds: str,
    model_name: str,
    h_dim: int,
    n_samples: int,
    attack_trials: int,
    local_epochs: int,
    local_batch_size: int,
    micro_batch_size: int,
    lr: float,
    seed: int,
    checkpoint_path: Optional[Path],
) -> dict[str, Any]:
    local_steps = local_epochs * ((n_samples + local_batch_size - 1) // local_batch_size)
    return {
        "schema_version": 2,
        "signal": "fedavg_local_update",
        "update_definition": "delta_w = w_local - w_global",
        "ds": ds,
        "model": model_name,
        "h_dim": h_dim,
        "n_samples": n_samples,
        "attack_trials": attack_trials,
        "local_epochs": local_epochs,
        "local_steps": local_steps,
        "local_batch_size": local_batch_size,
        "micro_batch_size": micro_batch_size,
        "optimizer": "SGD",
        "lr": lr,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "seed": seed,
        "checkpoint": checkpoint_manifest(checkpoint_path),
        "author_source_hash": AUTHOR_SOURCE_HASH,
    }


def shared_fp32_update_path(config: Mapping[str, Any]) -> Path:
    config_id = hash_json(config)[:16]
    return (
        SHARED_UPDATE_ROOT
        / str(config["ds"])
        / str(config["model"])
        / f"n{config['n_samples']}_steps{config['local_steps']}_trials{config['attack_trials']}_{config_id}"
        / "fp32.pickle"
    )


def client_sparse_collector_config(
    *,
    ds: str,
    model_name: str,
    h_dim: int,
    n_samples: int,
    attack_trials: int,
    local_epochs: int,
    lr: float,
    seed: int,
    checkpoint_path: Optional[Path],
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "signal": "client_then_aggregate_sparse_fedavg_update",
        "update_definition": "delta_obs = mean_k S_r(delta_w_k), one sample per client",
        "ds": ds,
        "model": model_name,
        "h_dim": h_dim,
        "n_samples": n_samples,
        "attack_trials": attack_trials,
        "local_epochs": local_epochs,
        "local_steps": local_epochs,
        "local_batch_size": 1,
        "micro_batch_size": 1,
        "clients_per_trial": n_samples,
        "client_granularity": "one_sample_per_client",
        "aggregation_weights": "uniform_mean",
        "optimizer": "SGD",
        "lr": lr,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "seed": seed,
        "checkpoint": checkpoint_manifest(checkpoint_path),
        "author_source_hash": AUTHOR_SOURCE_HASH,
    }


def collect_fedavg_update_pickle(
    *,
    ds: str,
    model_name: str,
    h_dim: int,
    n_samples: int,
    local_epochs: int,
    local_batch_size: int,
    micro_batch_size: int,
    lr: float,
    output_file: Path,
    seed: int,
    global_checkpoint: Optional[Path] = None,
    attack_trials: Optional[int] = None,
    n_rounds: Optional[int] = None,
) -> Path:
    if attack_trials is None:
        if n_rounds is None:
            raise TypeError("attack_trials is required")
        attack_trials = n_rounds
    if n_rounds is not None and n_rounds != attack_trials:
        raise ValueError("n_rounds compatibility alias conflicts with attack_trials")
    if attack_trials < 1:
        raise ValueError("attack_trials must be >= 1")
    if micro_batch_size < 1 or micro_batch_size > local_batch_size:
        raise ValueError("micro_batch_size must be in [1, local_batch_size]")

    config = collector_config(
        ds=ds,
        model_name=model_name,
        h_dim=h_dim,
        n_samples=n_samples,
        attack_trials=attack_trials,
        local_epochs=local_epochs,
        local_batch_size=local_batch_size,
        micro_batch_size=micro_batch_size,
        lr=lr,
        seed=seed,
        checkpoint_path=global_checkpoint,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_manifest = output_file.parent / "manifest.json"
    expected_manifest = {**config, "method": "collector", "quant_bits": 32, "rounding": "nearest"}
    if output_file.exists():
        assert_matching_manifest(output_manifest, expected_manifest)
        return output_file

    torch.manual_seed(seed)
    device = get_device()
    global_model = build_experiment_model(
        model_name=model_name,
        ds=ds,
        h_dim=h_dim,
        device=device,
        checkpoint_path=global_checkpoint,
        seed=seed,
    )
    global_model.eval()
    xs, ys, sample_indices = _load_indexed_test_batches(
        ds=ds, n_samples=n_samples, attack_trials=attack_trials, seed=seed
    )
    names = [name for name, _ in global_model.named_parameters()]
    schema = parameter_schema(global_model)
    attack_layer_index = int(getattr(global_model, "attack_index", 0))
    if attack_layer_index >= len(names):
        raise ValueError("Model attack_index is outside named parameter mapping")
    batch_orders = _local_batch_orders(n_samples, local_batch_size, local_epochs)
    criterion = nn.CrossEntropyLoss()
    data: dict[str, Any] = {
        "schema_version": 2,
        "x": [],
        "y": [],
        "z": [],
        "grad": [],
        "param_names": names,
        "parameter_schema": schema,
        "attack_layer": {"index": attack_layer_index, "name": names[attack_layer_index]},
        "sample_indices": sample_indices,
        "local_batch_orders": batch_orders,
        "quant_codes": None,
        "quant_stats": [],
        "metadata": {
            **config,
            "quant_bits": 32,
            "rounding": "nearest",
            "fake_quantization": False,
            "dequantized_before_attack": False,
            "trial_semantics": "independent_batches_from_same_w0",
            "multi_step_cpa_factorization": "empirical_baseline_approximation",
            "gradient_accumulation": {
                "enabled": micro_batch_size < local_batch_size,
                "micro_batch_size": micro_batch_size,
                "logical_batch_size": local_batch_size,
            },
        },
    }

    for x_cpu, y_cpu in zip(xs, ys):
        data["x"].append(x_cpu.numpy())
        data["y"].append(y_cpu.numpy())
        with torch.no_grad():
            if getattr(global_model, "model_type", None) == "conv":
                z_chunks = []
                for start in range(0, len(x_cpu), micro_batch_size):
                    _, z_chunk = global_model(x_cpu[start : start + micro_batch_size].to(device), return_z=True)
                    z_chunks.append(z_chunk.detach().cpu())
                data["z"].append(torch.cat(z_chunks, dim=0).numpy())

        local_model = copy.deepcopy(global_model).to(device)
        local_model.train()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
        local_dataset = TensorDataset(x_cpu, y_cpu)
        local_loader = DataLoader(local_dataset, batch_size=local_batch_size, shuffle=False, num_workers=0)
        for _epoch in range(local_epochs):
            for x_batch, y_batch in local_loader:
                optimizer.zero_grad(set_to_none=True)
                logical_batch_size = len(x_batch)
                for start in range(0, logical_batch_size, micro_batch_size):
                    x_micro = x_batch[start : start + micro_batch_size].to(device)
                    y_micro = y_batch[start : start + micro_batch_size].to(device)
                    loss = criterion(local_model(x_micro), y_micro)
                    loss = loss * (len(x_micro) / logical_batch_size)
                    loss.backward()
                optimizer.step()

        update = _named_update_dict(local_model, global_model)
        data["quant_stats"].append({"quant_bits": 32, "rounding": "none"})
        data["grad"].append([value.detach().cpu().numpy() for value in update.values()])
        del local_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    data["metadata"]["sample_indices_hash"] = sha256_bytes(json.dumps(sample_indices).encode("utf-8"))
    data["metadata"]["label_hash"] = hash_tensor_sequence(data["y"])
    data["metadata"]["update_tensor_hash"] = hash_tensor_sequence(data["grad"])
    write_pickle(output_file, data)
    write_json(output_manifest, expected_manifest)
    return output_file


def _mean_row_value(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.asarray(values, dtype=np.float64).mean())


def _hash_client_attack_observations(observations: Sequence[Sequence[Mapping[str, Any]]]) -> str:
    digest = hashlib.sha256()
    for trial_rows in observations:
        digest.update(f"trial:{len(trial_rows)}".encode("ascii"))
        digest.update(b"\0")
        for row in trial_rows:
            for key in ("batch", "client", "aggregation_weight", "tau", "observed_elements", "total_elements"):
                digest.update(str(row[key]).encode("utf-8"))
                digest.update(b"\0")
            if "observation_hash" in row:
                digest.update(str(row["observation_hash"]).encode("ascii"))
                digest.update(b"\0")
                continue
            mask = row["mask"]
            digest.update(str(tuple(mask["shape"])).encode("ascii"))
            digest.update(np.asarray(mask["packed_bits"], dtype=np.uint8).tobytes())
            values = np.asarray(row["values"], dtype=np.float32)
            digest.update(str(tuple(values.shape)).encode("ascii"))
            digest.update(values.tobytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _attack_observation_shard_dir(index_path: Path) -> Path:
    return index_path.with_suffix("").parent / f"{index_path.with_suffix('').name}_shards"


def _write_attack_observation_shard(
    *,
    index_path: Path,
    ratio: float,
    batch: int,
    client: int,
    aggregation_weight: float,
    observation: Mapping[str, Any],
    pack_bool_mask: Any,
) -> dict[str, Any]:
    shard_dir = _attack_observation_shard_dir(index_path)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_name = f"batch_{batch:03d}_client_{client:04d}.npz"
    shard_path = shard_dir / shard_name
    mask = pack_bool_mask(observation["mask"])
    values = np.asarray(observation["values"], dtype=np.float32)
    np.savez(
        shard_path,
        mask_shape=np.asarray(mask["shape"], dtype=np.int64),
        mask_packed_bits=np.asarray(mask["packed_bits"], dtype=np.uint8),
        values=values,
    )
    digest = hashlib.sha256()
    digest.update(str(tuple(mask["shape"])).encode("ascii"))
    digest.update(np.asarray(mask["packed_bits"], dtype=np.uint8).tobytes())
    digest.update(str(tuple(values.shape)).encode("ascii"))
    digest.update(values.tobytes())
    observation_hash = digest.hexdigest()
    return {
        "batch": int(batch),
        "client": int(client),
        "aggregation_weight": float(aggregation_weight),
        "tau": float(observation["tau"]),
        "tau_policy": "min_abs_value_among_global_topk_observed_coordinates",
        "shard_file": str(shard_path.relative_to(index_path.parent)),
        "observation_hash": observation_hash,
        "observed_elements": int(observation["observed_elements"]),
        "total_elements": int(observation["total_elements"]),
        "global_total_elements": int(observation.get("global_total_elements", 0)),
        "global_kept_elements": int(observation.get("global_kept_elements", 0)),
        "selection_mode": str(observation.get("selection_mode", "")),
        "storage": "sharded_npz",
        "keep_ratio": float(ratio),
    }


def _build_sparse_trial_data(
    *,
    config: Mapping[str, Any],
    keep_ratio: float,
    sparse_policy: str,
    names: Sequence[str],
    schema: Sequence[Mapping[str, Any]],
    attack_layer_index: int,
    sample_indices: list[list[int]],
    observed_zero_policy: str,
) -> dict[str, Any]:
    control_role = "one_sample_per_client_fp32_control" if float(keep_ratio) == 1.0 else None
    return {
        "schema_version": 3,
        "x": [],
        "y": [],
        "z": [],
        "grad": [],
        "param_names": list(names),
        "parameter_schema": list(schema),
        "attack_layer": {"index": attack_layer_index, "name": names[attack_layer_index]},
        "sample_indices": sample_indices,
        "local_batch_orders": _local_batch_orders(int(config["n_samples"]), 1, int(config["local_epochs"])),
        "quant_codes": None,
        "quant_stats": [],
        "sparse_masks": [],
        "sparse_stats": [],
        "client_sparse_stats": [],
        "aggregate_sparse_stats": [],
        "whitening_pre_stats": [],
        "metadata": {
            **dict(config),
            "quant_bits": 32,
            "rounding": "nearest",
            "fake_quantization": False,
            "dequantized_before_attack": False,
            "sparsified_observation": True,
            "sparsification_policy": sparse_policy,
            "sparsification_application": "client_then_aggregate",
            "keep_ratio": float(keep_ratio),
            "control_role": control_role,
            "observed_zero_policy": observed_zero_policy,
            "client_granularity": "one_sample_per_client",
            "aggregation_weights": "uniform_mean",
            "client_masks_stored": False,
            "trial_semantics": "independent_batches_from_same_w0",
            "multi_step_cpa_factorization": "client_level_sparse_before_uniform_aggregate",
            "attack_fc_observation_model": "Y_obs = [delta_theta_obs]_attack_fc ~= A Z",
            "az_scope": "attack_fc_layer_only_approximate",
        },
    }


def _build_client_attack_observation_data(
    *,
    config: Mapping[str, Any],
    keep_ratio: float,
    sparse_policy: str,
    names: Sequence[str],
    schema: Sequence[Mapping[str, Any]],
    attack_layer_index: int,
    sample_indices: list[list[int]],
    observed_zero_policy: str,
) -> dict[str, Any]:
    attack_schema = dict(schema[attack_layer_index])
    return {
        "schema_version": 2,
        "param_names": list(names),
        "attack_layer": {
            "index": attack_layer_index,
            "name": names[attack_layer_index],
            "shape": attack_schema["shape"],
            "dtype": attack_schema["dtype"],
        },
        "sample_indices": sample_indices,
        "storage": "inline",
        "observations": [],
        "metadata": {
            **dict(config),
            "sparsified_observation": True,
            "fixed_sparse_observation": True,
            "sparsification_policy": sparse_policy,
            "sparsification_application": "client_then_aggregate",
            "keep_ratio": float(keep_ratio),
            "observed_zero_policy": observed_zero_policy,
            "client_granularity": "one_sample_per_client",
            "aggregation_weights": "uniform_mean",
            "desparsification_scope": "attack_layer",
            "missing_value_abs_bound_policy": "per_client_global_topk_min_observed_abs_tau",
            "client_attack_layer_masks_stored": True,
            "client_attack_layer_values_stored": True,
            "client_attack_layer_storage": "inline",
            "trial_semantics": "independent_batches_from_same_w0",
        },
    }


def _collect_client_attack_observation_shards_only(
    *,
    ds: str,
    model_name: str,
    h_dim: int,
    n_samples: int,
    local_epochs: int,
    lr: float,
    seed: int,
    global_checkpoint: Optional[Path],
    sparse_policy: str,
    author_source_hash: str | None,
    attack_trials: int,
    normalized_output_files: Mapping[float, Path],
    normalized_observation_files: Mapping[float, Path],
    observation_manifests: Mapping[float, Mapping[str, Any]],
    config: Mapping[str, Any],
    collect_log_every: int | None = None,
) -> None:
    from sparsification import (
        OBSERVED_ZERO_POLICY,
        global_topk_attack_layer_observation,
        pack_bool_mask,
        require_global_topk,
        validate_keep_ratio,
    )

    require_global_topk(sparse_policy)
    missing_ratios = sorted(
        ratio
        for ratio, path in normalized_observation_files.items()
        if float(ratio) != 1.0 and not path.exists()
    )
    if not missing_ratios:
        return
    for ratio in missing_ratios:
        validate_keep_ratio(ratio)

    torch.manual_seed(seed)
    device = get_device()
    global_model = build_experiment_model(
        model_name=model_name,
        ds=ds,
        h_dim=h_dim,
        device=device,
        checkpoint_path=global_checkpoint,
        seed=seed,
    )
    global_model.eval()
    xs, ys, sample_indices = _load_indexed_test_batches(
        ds=ds, n_samples=n_samples, attack_trials=attack_trials, seed=seed
    )
    named_params = list(global_model.named_parameters())
    names = [name for name, _ in named_params]
    schema = parameter_schema(global_model)
    attack_layer_index = int(getattr(global_model, "attack_index", 0))
    if attack_layer_index >= len(names):
        raise ValueError("Model attack_index is outside named parameter mapping")

    observation_data_by_ratio = {
        ratio: _build_client_attack_observation_data(
            config=config,
            keep_ratio=ratio,
            sparse_policy=sparse_policy,
            names=names,
            schema=schema,
            attack_layer_index=attack_layer_index,
            sample_indices=sample_indices,
            observed_zero_policy=OBSERVED_ZERO_POLICY,
        )
        for ratio in missing_ratios
    }
    for ratio, observation_data in observation_data_by_ratio.items():
        path = normalized_observation_files[ratio]
        shard_dir = _attack_observation_shard_dir(path)
        if shard_dir.exists() and not path.exists():
            shutil.rmtree(shard_dir)
        observation_data["storage"] = "sharded_npz"
        observation_data["shard_root"] = str(shard_dir.relative_to(path.parent))
        observation_data["metadata"]["client_attack_layer_storage"] = "sharded_npz"
        observation_data["metadata"]["client_attack_layer_values_stored"] = True
        observation_data["metadata"]["client_attack_layer_masks_stored"] = True
        observation_data["metadata"]["source_sparse_update_file"] = str(
            normalized_output_files[ratio].resolve()
        )
        observation_data["metadata"]["source_sparse_update_hash"] = sha256_file(
            normalized_output_files[ratio]
        )
        observation_data["metadata"]["source_sparse_update_tensor_hash"] = None
        observation_data["metadata"]["source_sparse_update_tensor_hash_note"] = (
            "not loaded in sidecar-only collection to avoid materializing large baseline pickle"
        )

    criterion = nn.CrossEntropyLoss()
    progress_every = max(1, int(collect_log_every)) if collect_log_every else max(1, min(8, n_samples))
    print(
        "[sparse-observe] "
        f"reusing existing aggregate artifacts; writing sharded fixed attack-layer observations "
        f"ds={ds} model={model_name} trials={attack_trials} clients_per_trial={n_samples} "
        f"local_epochs={local_epochs} keep_ratios={' '.join(format(ratio, 'g') for ratio in missing_ratios)}",
        flush=True,
    )

    label_batches = []
    for trial_idx, (x_cpu, y_cpu) in enumerate(zip(xs, ys)):
        label_batches.append(y_cpu.numpy())
        for observation_data in observation_data_by_ratio.values():
            observation_data["observations"].append([])
        print(f"[sparse-observe] trial {trial_idx + 1}/{attack_trials} start", flush=True)
        for client_idx in range(n_samples):
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
            x_one = x_cpu[client_idx : client_idx + 1].to(device)
            y_one = y_cpu[client_idx : client_idx + 1].to(device)
            for _epoch in range(local_epochs):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(local_model(x_one), y_one)
                loss.backward()
                optimizer.step()

            update = OrderedDict(
                (name, value.detach().cpu().float().contiguous())
                for name, value in _named_update_dict(local_model, global_model).items()
            )
            for ratio in missing_ratios:
                observation = global_topk_attack_layer_observation(
                    update,
                    keep_ratio=ratio,
                    attack_layer_index=attack_layer_index,
                )
                row = _write_attack_observation_shard(
                    index_path=normalized_observation_files[ratio],
                    ratio=ratio,
                    batch=trial_idx,
                    client=client_idx,
                    aggregation_weight=1.0 / float(n_samples),
                    observation=observation,
                    pack_bool_mask=pack_bool_mask,
                )
                observation_data_by_ratio[ratio]["observations"][trial_idx].append(row)
                del observation

            del local_model, update
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (
                client_idx == 0
                or (client_idx + 1) % progress_every == 0
                or (client_idx + 1) == n_samples
            ):
                print(
                    "[sparse-observe] "
                    f"trial {trial_idx + 1}/{attack_trials} client {client_idx + 1}/{n_samples}",
                    flush=True,
                )

    label_hash = hash_tensor_sequence(label_batches)
    sample_indices_hash = sha256_bytes(json.dumps(sample_indices).encode("utf-8"))
    for ratio, observation_data in observation_data_by_ratio.items():
        metadata = observation_data["metadata"]
        metadata["sample_indices_hash"] = sample_indices_hash
        metadata["label_hash"] = label_hash
        metadata["author_source_hash"] = author_source_hash
        metadata["client_attack_observation_hash"] = _hash_client_attack_observations(
            observation_data["observations"]
        )
        metadata["client_attack_observation_file_role"] = (
            "sharded fixed per-client sparse attack-layer observation for randomized desparsification"
        )
        tau_values = [
            float(row["tau"])
            for trial_rows in observation_data["observations"]
            for row in trial_rows
        ]
        observed_fractions = [
            float(row["observed_elements"]) / max(1.0, float(row["total_elements"]))
            for trial_rows in observation_data["observations"]
            for row in trial_rows
        ]
        metadata["tau_summary"] = {
            "mean": float(np.asarray(tau_values, dtype=np.float64).mean()) if tau_values else None,
            "min": float(np.asarray(tau_values, dtype=np.float64).min()) if tau_values else None,
            "max": float(np.asarray(tau_values, dtype=np.float64).max()) if tau_values else None,
        }
        metadata["attack_layer_observed_fraction_summary"] = {
            "mean": (
                float(np.asarray(observed_fractions, dtype=np.float64).mean())
                if observed_fractions
                else None
            ),
            "min": (
                float(np.asarray(observed_fractions, dtype=np.float64).min())
                if observed_fractions
                else None
            ),
            "max": (
                float(np.asarray(observed_fractions, dtype=np.float64).max())
                if observed_fractions
                else None
            ),
        }
        path = normalized_observation_files[ratio]
        write_pickle(path, observation_data)
        write_json(path.with_suffix(".manifest.json"), observation_manifests[ratio])
        print(f"[sparse-observe] wrote keep={ratio:g} sharded attack-layer observations: {path}", flush=True)


def collect_client_then_aggregate_sparse_pickles(
    *,
    ds: str,
    model_name: str,
    h_dim: int,
    n_samples: int,
    local_epochs: int,
    lr: float,
    output_files: Mapping[float, Path],
    seed: int,
    global_checkpoint: Optional[Path],
    sparse_policy: str,
    author_source_hash: str | None = None,
    attack_trials: int,
    client_attack_observation_files: Optional[Mapping[float, Path]] = None,
    collect_log_every: int | None = None,
) -> dict[float, Path]:
    from sparsification import (
        OBSERVED_ZERO_POLICY,
        attack_layer_spectrum_stats,
        compare_update_to_reference,
        identity_global_topk_stats,
        pack_bool_mask,
        require_global_topk,
        sparsify_update_global_topk,
        summarize_client_sparse_stats,
        validate_keep_ratio,
    )

    require_global_topk(sparse_policy)
    requested_ratios = sorted({float(ratio) for ratio in output_files}, reverse=True)
    if not requested_ratios:
        raise ValueError("At least one keep ratio is required")
    for ratio in requested_ratios:
        validate_keep_ratio(ratio)
    if attack_trials < 1:
        raise ValueError("attack_trials must be >= 1")

    work_ratios = sorted(set(requested_ratios + [1.0]), reverse=True)
    config = client_sparse_collector_config(
        ds=ds,
        model_name=model_name,
        h_dim=h_dim,
        n_samples=n_samples,
        attack_trials=attack_trials,
        local_epochs=local_epochs,
        lr=lr,
        seed=seed,
        checkpoint_path=global_checkpoint,
    )
    expected_manifests = {
        ratio: {
            **config,
            "method": "client_then_aggregate_sparse_collector",
            "sparse_policy": sparse_policy,
            "keep_ratio": ratio,
            "observed_zero_policy": OBSERVED_ZERO_POLICY,
            "author_source_hash": author_source_hash,
        }
        for ratio in requested_ratios
    }
    normalized_output_files = {float(ratio): path for ratio, path in output_files.items()}
    normalized_observation_files = (
        {float(ratio): path for ratio, path in client_attack_observation_files.items()}
        if client_attack_observation_files
        else {}
    )
    unknown_observation_ratios = set(normalized_observation_files) - set(normalized_output_files)
    if unknown_observation_ratios:
        raise ValueError(
            "client_attack_observation_files contains ratios missing from output_files: "
            f"{sorted(unknown_observation_ratios)}"
        )
    observation_manifests = {
        ratio: {
            **config,
            "method": "client_attack_layer_fixed_sparse_observation_collector",
            "sparse_policy": sparse_policy,
            "keep_ratio": ratio,
            "observed_zero_policy": OBSERVED_ZERO_POLICY,
            "desparsification_scope": "attack_layer",
            "author_source_hash": author_source_hash,
        }
        for ratio in normalized_observation_files
        if float(ratio) != 1.0
    }
    for ratio, path in normalized_output_files.items():
        if path.exists():
            assert_matching_manifest(path.with_suffix(".manifest.json"), expected_manifests[ratio])
    for ratio, path in normalized_observation_files.items():
        if float(ratio) == 1.0:
            continue
        if path.exists():
            assert_matching_manifest(path.with_suffix(".manifest.json"), observation_manifests[ratio])
    if all(path.exists() for path in normalized_output_files.values()) and all(
        float(ratio) == 1.0 or path.exists() for ratio, path in normalized_observation_files.items()
    ):
        return normalized_output_files
    if all(path.exists() for path in normalized_output_files.values()) and any(
        float(ratio) != 1.0 and not path.exists()
        for ratio, path in normalized_observation_files.items()
    ):
        _collect_client_attack_observation_shards_only(
            ds=ds,
            model_name=model_name,
            h_dim=h_dim,
            n_samples=n_samples,
            local_epochs=local_epochs,
            lr=lr,
            seed=seed,
            global_checkpoint=global_checkpoint,
            sparse_policy=sparse_policy,
            author_source_hash=author_source_hash,
            attack_trials=attack_trials,
            normalized_output_files=normalized_output_files,
            normalized_observation_files=normalized_observation_files,
            observation_manifests=observation_manifests,
            config=config,
            collect_log_every=collect_log_every,
        )
        if all(
            float(ratio) == 1.0 or path.exists()
            for ratio, path in normalized_observation_files.items()
        ):
            return normalized_output_files

    torch.manual_seed(seed)
    device = get_device()
    global_model = build_experiment_model(
        model_name=model_name,
        ds=ds,
        h_dim=h_dim,
        device=device,
        checkpoint_path=global_checkpoint,
        seed=seed,
    )
    global_model.eval()
    xs, ys, sample_indices = _load_indexed_test_batches(
        ds=ds, n_samples=n_samples, attack_trials=attack_trials, seed=seed
    )
    named_params = list(global_model.named_parameters())
    names = [name for name, _ in named_params]
    schema = parameter_schema(global_model)
    param_shapes = [tuple(parameter.shape) for _, parameter in named_params]
    attack_layer_index = int(getattr(global_model, "attack_index", 0))
    if attack_layer_index >= len(names):
        raise ValueError("Model attack_index is outside named parameter mapping")

    data_by_ratio = {
        ratio: _build_sparse_trial_data(
            config=config,
            keep_ratio=ratio,
            sparse_policy=sparse_policy,
            names=names,
            schema=schema,
            attack_layer_index=attack_layer_index,
            sample_indices=sample_indices,
            observed_zero_policy=OBSERVED_ZERO_POLICY,
        )
        for ratio in work_ratios
    }
    observation_data_by_ratio = {
        ratio: _build_client_attack_observation_data(
            config=config,
            keep_ratio=ratio,
            sparse_policy=sparse_policy,
            names=names,
            schema=schema,
            attack_layer_index=attack_layer_index,
            sample_indices=sample_indices,
            observed_zero_policy=OBSERVED_ZERO_POLICY,
        )
        for ratio, path in normalized_observation_files.items()
        if float(ratio) != 1.0 and not path.exists()
    }

    criterion = nn.CrossEntropyLoss()
    progress_every = max(1, int(collect_log_every)) if collect_log_every else max(1, min(8, n_samples))
    print(
        "[sparse-collect] "
        f"ds={ds} model={model_name} trials={attack_trials} "
        f"clients_per_trial={n_samples} local_epochs={local_epochs} "
        f"keep_ratios={' '.join(format(ratio, 'g') for ratio in requested_ratios)}",
        flush=True,
    )

    def empty_accumulator() -> list[torch.Tensor]:
        return [torch.zeros(shape, dtype=torch.float32) for shape in param_shapes]

    for trial_idx, (x_cpu, y_cpu) in enumerate(zip(xs, ys)):
        print(f"[sparse-collect] trial {trial_idx + 1}/{attack_trials} start", flush=True)
        with torch.no_grad():
            z_value = None
            if getattr(global_model, "model_type", None) == "conv":
                z_chunks = []
                feature_chunk_size = max(1, min(4, len(x_cpu)))
                for start in range(0, len(x_cpu), feature_chunk_size):
                    _, z_chunk = global_model(x_cpu[start : start + feature_chunk_size].to(device), return_z=True)
                    z_chunks.append(z_chunk.detach().cpu())
                z_value = torch.cat(z_chunks, dim=0).numpy()

        for ratio in work_ratios:
            data_by_ratio[ratio]["x"].append(x_cpu.numpy())
            data_by_ratio[ratio]["y"].append(y_cpu.numpy())
            if z_value is not None:
                data_by_ratio[ratio]["z"].append(z_value)
            if ratio in observation_data_by_ratio:
                observation_data_by_ratio[ratio]["observations"].append([])

        fp32_accum = empty_accumulator()
        sparse_accums = {ratio: empty_accumulator() for ratio in work_ratios if ratio != 1.0}
        client_stats_by_ratio: dict[float, list[dict[str, Any]]] = {ratio: [] for ratio in work_ratios}

        for client_idx in range(n_samples):
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
            x_one = x_cpu[client_idx : client_idx + 1].to(device)
            y_one = y_cpu[client_idx : client_idx + 1].to(device)
            for _epoch in range(local_epochs):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(local_model(x_one), y_one)
                loss.backward()
                optimizer.step()

            update = OrderedDict(
                (name, value.detach().cpu().float().contiguous())
                for name, value in _named_update_dict(local_model, global_model).items()
            )
            for param_idx, value in enumerate(update.values()):
                fp32_accum[param_idx].add_(value)

            for ratio in work_ratios:
                if ratio == 1.0:
                    stats = identity_global_topk_stats(update)
                else:
                    if ratio in observation_data_by_ratio:
                        sparse_update, _packed_masks, stats, raw_masks = sparsify_update_global_topk(
                            update, keep_ratio=ratio, pack_masks=False, return_masks=True
                        )
                    else:
                        sparse_update, _packed_masks, stats = sparsify_update_global_topk(
                            update, keep_ratio=ratio, pack_masks=False
                        )
                        raw_masks = None
                    for param_idx, value in enumerate(sparse_update.values()):
                        sparse_accums[ratio][param_idx].add_(value)
                    if ratio in observation_data_by_ratio:
                        attack_mask = raw_masks[attack_layer_index]
                        attack_tensor = sparse_update[names[attack_layer_index]]
                        observed_values = attack_tensor[attack_mask].detach().cpu().numpy().astype(
                            np.float32, copy=False
                        )
                        observation_data_by_ratio[ratio]["observations"][trial_idx].append(
                            {
                                "batch": trial_idx,
                                "client": client_idx,
                                "aggregation_weight": 1.0 / float(n_samples),
                                "tau": float(stats.get("topk_threshold_abs") or 0.0),
                                "tau_policy": "min_abs_value_among_global_topk_observed_coordinates",
                                "mask": pack_bool_mask(attack_mask),
                                "values": observed_values,
                                "observed_elements": int(attack_mask.sum().item()),
                                "total_elements": int(attack_mask.numel()),
                            }
                        )
                    if raw_masks is not None:
                        del raw_masks
                    del sparse_update
                client_stats_by_ratio[ratio].append(stats)

            del local_model, update
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (
                client_idx == 0
                or (client_idx + 1) % progress_every == 0
                or (client_idx + 1) == n_samples
            ):
                print(
                    "[sparse-collect] "
                    f"trial {trial_idx + 1}/{attack_trials} "
                    f"client {client_idx + 1}/{n_samples}",
                    flush=True,
                )

        scale = 1.0 / float(n_samples)
        for value in fp32_accum:
            value.mul_(scale)
        for accum in sparse_accums.values():
            for value in accum:
                value.mul_(scale)

        reference_update = OrderedDict((name, value) for name, value in zip(names, fp32_accum))
        for ratio in work_ratios:
            aggregate_tensors = fp32_accum if ratio == 1.0 else sparse_accums[ratio]
            aggregate_update = OrderedDict((name, value) for name, value in zip(names, aggregate_tensors))
            aggregate_stats = compare_update_to_reference(
                aggregate_update,
                reference_update,
                attack_layer_index=attack_layer_index,
            )
            spectrum = attack_layer_spectrum_stats(aggregate_tensors[attack_layer_index])
            client_level = summarize_client_sparse_stats(
                client_stats_by_ratio[ratio], attack_layer_index=attack_layer_index
            )
            trial_sparse_stats = {
                "batch": trial_idx,
                "sparsification_policy": sparse_policy,
                "sparsification_application": "client_then_aggregate",
                "keep_ratio": ratio,
                "control_role": "one_sample_per_client_fp32_control" if ratio == 1.0 else None,
                "observed_zero_policy": OBSERVED_ZERO_POLICY,
                "num_clients": n_samples,
                "client_level": client_level,
                "aggregate_level": aggregate_stats,
                "attack_layer_spectrum": spectrum,
                "attack_fc_observation_model": "Y_obs = [delta_theta_obs]_attack_fc ~= A Z",
                "az_scope": "attack_fc_layer_only_approximate",
            }
            data_by_ratio[ratio]["grad"].append(
                [value.detach().cpu().numpy().astype(np.float32, copy=False) for value in aggregate_tensors]
            )
            data_by_ratio[ratio]["quant_stats"].append({"quant_bits": 32, "rounding": "none"})
            data_by_ratio[ratio]["sparse_stats"].append(trial_sparse_stats)
            data_by_ratio[ratio]["client_sparse_stats"].append(client_level)
            data_by_ratio[ratio]["aggregate_sparse_stats"].append(aggregate_stats)
            data_by_ratio[ratio]["whitening_pre_stats"].append(
                {
                    "batch": trial_idx,
                    "attack_layer": {"index": attack_layer_index, "name": names[attack_layer_index]},
                    "observation_model": "Y_obs = [delta_theta_obs]_attack_fc ~= A Z",
                    "spectrum": spectrum,
                }
            )

        del fp32_accum, sparse_accums
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[sparse-collect] trial {trial_idx + 1}/{attack_trials} aggregated", flush=True)

    client_fp32_update_hash = hash_tensor_sequence(data_by_ratio[1.0]["grad"])
    for ratio, data in data_by_ratio.items():
        aggregate_rows = [row["global"] for row in data["aggregate_sparse_stats"]]
        attack_rows = [row["attack_layer"] for row in data["aggregate_sparse_stats"]]
        rank_rows = [row["spectrum"] for row in data["whitening_pre_stats"]]
        metadata = data["metadata"]
        metadata["sample_indices_hash"] = sha256_bytes(json.dumps(sample_indices).encode("utf-8"))
        metadata["label_hash"] = hash_tensor_sequence(data["y"])
        metadata["update_tensor_hash"] = hash_tensor_sequence(data["grad"])
        metadata["source_fp32_update_hash"] = client_fp32_update_hash
        metadata["source_fp32_update_file"] = (
            str(normalized_output_files[1.0].resolve()) if 1.0 in normalized_output_files else None
        )
        metadata["sparse_nonzero_fraction"] = _mean_row_value(aggregate_rows, "nonzero_fraction")
        metadata["sparse_relative_l2"] = _mean_row_value(aggregate_rows, "relative_l2_to_fp32")
        metadata["sparse_cosine_similarity_to_fp32"] = _mean_row_value(
            aggregate_rows, "cosine_similarity_to_fp32"
        )
        metadata["sparse_retained_l2_norm_ratio"] = _mean_row_value(
            aggregate_rows, "retained_l2_norm_ratio"
        )
        metadata["aggregate_attack_layer_nonzero_fraction"] = _mean_row_value(
            attack_rows, "nonzero_fraction"
        )
        metadata["aggregate_attack_layer_relative_l2_to_fp32"] = _mean_row_value(
            attack_rows, "relative_l2_to_fp32"
        )
        metadata["aggregate_attack_layer_cosine_similarity_to_fp32"] = _mean_row_value(
            attack_rows, "cosine_similarity_to_fp32"
        )
        metadata["aggregate_attack_layer_retained_l2_norm_ratio"] = _mean_row_value(
            attack_rows, "retained_l2_norm_ratio"
        )
        metadata["attack_layer_numerical_rank_mean"] = _mean_row_value(rank_rows, "numerical_rank")
        metadata["attack_layer_effective_rank_mean"] = _mean_row_value(rank_rows, "effective_rank")
        metadata["client_level_sparsity"] = data["client_sparse_stats"]
        metadata["aggregate_level_sparsity"] = data["aggregate_sparse_stats"]
        metadata["whitening_pre_stats_file"] = "whitening_pre_stats.json"
        metadata["sparse_stats_hash"] = sha256_bytes(
            json.dumps(data["sparse_stats"], sort_keys=True, default=str).encode("utf-8")
        )

    for ratio in requested_ratios:
        path = normalized_output_files[ratio]
        write_pickle(path, data_by_ratio[ratio])
        write_json(path.with_suffix(".manifest.json"), expected_manifests[ratio])
        print(f"[sparse-collect] wrote keep={ratio:g} artifact: {path}", flush=True)

    for ratio, observation_data in observation_data_by_ratio.items():
        path = normalized_observation_files[ratio]
        metadata = observation_data["metadata"]
        metadata["sample_indices_hash"] = sha256_bytes(json.dumps(sample_indices).encode("utf-8"))
        metadata["label_hash"] = data_by_ratio[ratio]["metadata"]["label_hash"]
        metadata["source_sparse_update_file"] = str(normalized_output_files[ratio].resolve())
        metadata["source_sparse_update_hash"] = sha256_file(normalized_output_files[ratio])
        metadata["source_sparse_update_tensor_hash"] = data_by_ratio[ratio]["metadata"]["update_tensor_hash"]
        metadata["client_attack_observation_hash"] = _hash_client_attack_observations(
            observation_data["observations"]
        )
        metadata["client_attack_observation_file_role"] = (
            "fixed per-client sparse attack-layer observation for randomized desparsification"
        )
        write_pickle(path, observation_data)
        write_json(path.with_suffix(".manifest.json"), observation_manifests[ratio])
        print(f"[sparse-collect] wrote keep={ratio:g} attack-layer observations: {path}", flush=True)
    return normalized_output_files


def ensure_shared_fp32_update(args: argparse.Namespace) -> Path:
    config = collector_config(
        ds=args.ds,
        model_name=args.model,
        h_dim=args.h_dim,
        n_samples=args.n_samples,
        attack_trials=args.attack_trials,
        local_epochs=args.local_epochs,
        local_batch_size=resolve_local_batch_size(args),
        micro_batch_size=resolve_micro_batch_size(args),
        lr=args.local_lr,
        seed=args.seed,
        checkpoint_path=args.global_checkpoint,
    )
    path = shared_fp32_update_path(config)
    return collect_fedavg_update_pickle(
        ds=args.ds,
        model_name=args.model,
        h_dim=args.h_dim,
        n_samples=args.n_samples,
        attack_trials=args.attack_trials,
        local_epochs=args.local_epochs,
        local_batch_size=resolve_local_batch_size(args),
        micro_batch_size=resolve_micro_batch_size(args),
        lr=args.local_lr,
        output_file=path,
        seed=args.seed,
        global_checkpoint=args.global_checkpoint,
    )


def run_tag(args: argparse.Namespace, *, include_fia: bool = True) -> str:
    if getattr(args, "sparsification_application", None) == "client_then_aggregate":
        local_steps = args.local_epochs
    else:
        local_batch_size = resolve_local_batch_size(args)
        local_steps = args.local_epochs * ((args.n_samples + local_batch_size - 1) // local_batch_size)
    tag = f"n{args.n_samples}_steps{local_steps}_trials{args.attack_trials}"
    if include_fia:
        n_sample_fi = args.n_samples if args.n_sample_fi in {-1, 0} else min(args.n_sample_fi, args.n_samples)
        tag += f"_fia{n_sample_fi}"
    return tag


def update_experiment_manifest(
    *,
    experiment_id: str,
    method: str,
    update_file: Path,
    args: argparse.Namespace,
    evaluated_samples: int,
    extra: Optional[Mapping[str, Any]] = None,
    update_data: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    data = update_data if update_data is not None else read_pickle(update_file)
    metadata = data["metadata"]
    update_file_hash = metadata.get("update_file_hash_override") or sha256_file(update_file)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "method": method,
        "author_source": str(CPA_SRC.resolve()),
        "author_source_hash": AUTHOR_SOURCE_HASH,
        "checkpoint": metadata["checkpoint"],
        "update_file": str(update_file.resolve()),
        "update_file_hash": update_file_hash,
        "update_tensor_hash": metadata["update_tensor_hash"],
        "sample_indices_hash": metadata["sample_indices_hash"],
        "label_hash": metadata["label_hash"],
        "model_architecture": args.model,
        "attacked_layer": data["attack_layer"],
        "local_steps": metadata["local_steps"],
        "local_epochs": metadata["local_epochs"],
        "batch_size": metadata["local_batch_size"],
        "learning_rate": metadata["lr"],
        "optimizer": metadata["optimizer"],
        "momentum": metadata["momentum"],
        "weight_decay": metadata["weight_decay"],
        "bit_width": metadata.get("quant_bits", 32),
        "rounding_mode": metadata.get("rounding"),
        "sparsification_policy": metadata.get("sparsification_policy"),
        "keep_ratio": metadata.get("keep_ratio"),
        "observed_zero_policy": metadata.get("observed_zero_policy"),
        "sparse_nonzero_fraction": metadata.get("sparse_nonzero_fraction"),
        "sparse_relative_l2": metadata.get("sparse_relative_l2"),
        "sparse_cosine_similarity_to_fp32": metadata.get("sparse_cosine_similarity_to_fp32"),
        "sparse_retained_l2_norm_ratio": metadata.get("sparse_retained_l2_norm_ratio"),
        "source_fp32_update_hash": metadata.get("source_fp32_update_hash"),
        "sparsification_application": metadata.get("sparsification_application"),
        "client_granularity": metadata.get("client_granularity"),
        "clients_per_trial": metadata.get("clients_per_trial"),
        "aggregation_weights": metadata.get("aggregation_weights"),
        "control_role": metadata.get("control_role"),
        "aggregate_attack_layer_nonzero_fraction": metadata.get("aggregate_attack_layer_nonzero_fraction"),
        "aggregate_attack_layer_relative_l2_to_fp32": metadata.get("aggregate_attack_layer_relative_l2_to_fp32"),
        "aggregate_attack_layer_cosine_similarity_to_fp32": metadata.get(
            "aggregate_attack_layer_cosine_similarity_to_fp32"
        ),
        "aggregate_attack_layer_retained_l2_norm_ratio": metadata.get(
            "aggregate_attack_layer_retained_l2_norm_ratio"
        ),
        "attack_layer_numerical_rank_mean": metadata.get("attack_layer_numerical_rank_mean"),
        "attack_layer_effective_rank_mean": metadata.get("attack_layer_effective_rank_mean"),
        "whitening_pre_stats_file": metadata.get("whitening_pre_stats_file"),
        "attack_fc_observation_model": metadata.get("attack_fc_observation_model"),
        "az_scope": metadata.get("az_scope"),
        "seed": metadata["seed"],
        "attack_trials": metadata["attack_trials"],
        "samples_per_trial": metadata["n_samples"],
        "total_attacked_samples": metadata["attack_trials"] * metadata["n_samples"],
        "total_reconstructed_samples": metadata["attack_trials"] * evaluated_samples,
        "total_evaluated_samples": metadata["attack_trials"] * evaluated_samples,
        "evaluated_samples_per_trial": evaluated_samples,
        "oracle_policy": "evaluation_only_hungarian_assignment",
    }
    if extra:
        manifest.update(dict(extra))
    return manifest


def add_collect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ds", type=str, default="imagenet")
    parser.add_argument("--model", type=str, default="vgg16")
    parser.add_argument("--h_dim", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument(
        "--attack_trials", "--n_rounds", dest="attack_trials", type=int, default=10,
        help="Independent batches from the same w0; not sequential global rounds.",
    )
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--local_batch_size", type=int, default=None)
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=None,
        help=(
            "Memory-only micro-batch size for accumulating each logical local batch. "
            "Defaults to 4 for imagenet/vgg16 and local_batch_size otherwise."
        ),
    )
    parser.add_argument("--local_lr", type=float, default=1e-3)
    parser.add_argument("--global_checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse_existing", action="store_true", help="Only reuse when manifests match exactly.")
    parser.add_argument("--run_attack", action="store_true")
    parser.add_argument(
        "--collect_log_every",
        type=int,
        default=None,
        help="Print sparse collection progress every N clients. Defaults to every 8 clients for n>=8.",
    )


def add_attack_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument(
        "--fia_micro_batch_size",
        type=int,
        default=None,
        help=(
            "Memory-only micro-batch size for Direct FIA forward/backward. "
            "Defaults to 2 for imagenet/vgg16 and n_sample_fi otherwise."
        ),
    )
    parser.add_argument("--fi", type=float, default=1.0)
    parser.add_argument("--fi_method", type=str, default="direct", choices=["direct", "gm"])
    parser.add_argument("--use_labels", action="store_true")
    parser.add_argument("--ideal_emb_rec", action="store_true")
    parser.add_argument("--fresh_start", action="store_true")
    parser.add_argument("--project", type=str, default="cpa_sparse")


def make_attack_namespace(args: argparse.Namespace, *, exp_name: str) -> argparse.Namespace:
    n_sample_fi = args.n_samples if args.n_sample_fi in {-1, 0} else min(args.n_sample_fi, args.n_samples)
    fia_micro_batch_size = resolve_fia_micro_batch_size(args, n_sample_fi)
    return argparse.Namespace(
        model=args.model, h_dim=args.h_dim, ds=args.ds, batch_size=args.n_samples, attack="cp",
        decor=args.decor, tv=args.tv, ne=args.ne, l1=args.l1, nv=args.nv, T=args.T,
        n_batch=args.attack_n_batch or args.attack_trials, n_iter=args.attack_n_iter,
        n_log=args.attack_n_log, opt="adam", lr=args.attack_lr, lr_N=1e-6, sch="none",
        fi_method=args.fi_method, n_iter_fi=args.n_iter_fi, n_log_fi=args.n_log_fi,
        opt_fi="adam", lr_fi=args.lr_fi, sch_fi="cosine", n_sample_fi=n_sample_fi,
        fia_micro_batch_size=fia_micro_batch_size,
        gm=0.0, fi=args.fi, fl_alg="fedavg", C=1.0, sigma=0.0, defense="nodef",
        exp_name=exp_name, project=args.project, fresh_start=args.fresh_start, dry_run=True,
        submitit=False, n_gpu=1, timeout=600, wait=False, disable_pbar=True,
        use_labels=args.use_labels, ideal_emb_rec=args.ideal_emb_rec,
    )


def resolve_local_batch_size(args: argparse.Namespace) -> int:
    return args.local_batch_size or args.n_samples


def resolve_micro_batch_size(args: argparse.Namespace) -> int:
    local_batch_size = resolve_local_batch_size(args)
    if args.micro_batch_size is None and args.ds == "imagenet" and args.model == "vgg16":
        micro_batch_size = min(local_batch_size, AUTO_VGG16_IMAGENET_COLLECT_MICRO_BATCH_SIZE)
    else:
        micro_batch_size = args.micro_batch_size or local_batch_size
    if micro_batch_size < 1 or micro_batch_size > local_batch_size:
        raise ValueError("micro_batch_size must be in [1, local_batch_size]")
    return micro_batch_size


def resolve_fia_micro_batch_size(args: argparse.Namespace, n_sample_fi: int) -> int:
    if args.fia_micro_batch_size is None and args.ds == "imagenet" and args.model == "vgg16":
        micro_batch_size = min(n_sample_fi, AUTO_VGG16_IMAGENET_FIA_MICRO_BATCH_SIZE)
    else:
        micro_batch_size = args.fia_micro_batch_size or n_sample_fi
    if micro_batch_size < 1 or micro_batch_size > n_sample_fi:
        raise ValueError("fia_micro_batch_size must be in [1, n_sample_fi]")
    return micro_batch_size


def _install_optional_dependency_stubs() -> None:
    if "submitit" not in sys.modules:
        try:
            __import__("submitit")
        except ModuleNotFoundError:
            sys.modules["submitit"] = types.ModuleType("submitit")
    if "wandb" not in sys.modules:
        try:
            __import__("wandb")
        except ModuleNotFoundError:
            stub = types.ModuleType("wandb")
            stub.init = lambda *args, **kwargs: types.SimpleNamespace(id="dry-run", get_url=lambda: "disabled")
            stub.log = lambda *args, **kwargs: None
            stub.config = types.SimpleNamespace(update=lambda *args, **kwargs: None)
            stub.Image = lambda *args, **kwargs: None
            stub.run = types.SimpleNamespace(id="dry-run", get_url=lambda: "disabled")
            sys.modules["wandb"] = stub
