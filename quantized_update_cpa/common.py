#!/usr/bin/env python3
"""Shared, provenance-safe infrastructure for the corrected CPA experiments.

``src/`` is the immutable public author snapshot.  Nothing in this module
modifies it: the module only imports author classes and supplies external
FedAvg artifacts, manifests, and experiment wrappers.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

for _thread_env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_var, "1")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from quantization import quantize_update_symmetric, require_nearest_rounding

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


REPO_ROOT = Path(__file__).resolve().parents[1]
# This is deliberately the immutable author snapshot at repository root.
CPA_SRC = REPO_ROOT / "src"
FEDAVG_SRC = REPO_ROOT / "fedavg_fp32"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs_v2"
SHARED_UPDATE_ROOT = OUTPUT_ROOT / "shared_updates"
AUTHOR_RUN_ROOT = OUTPUT_ROOT / "author_runs"
DATASET_ROOT = REPO_ROOT / "datasets"

if not CPA_SRC.is_dir():
    raise FileNotFoundError(f"Author CPA snapshot not found: {CPA_SRC}")
# Put the snapshot at the front before importing *any* author module.
# Keep ``src/`` first even if adjacent utility folders contain same-named modules.
for _path in (FEDAVG_SRC, CPA_SRC):
    _path_str = str(_path)
    if _path_str in sys.path:
        sys.path.remove(_path_str)
    sys.path.insert(0, _path_str)

import datasets as cpa_datasets  # noqa: E402
from datasets import get_dataloaders, nclasses_dict  # noqa: E402
from models import get_model  # noqa: E402
from utils import exp_path_base, get_device, write_pickle  # noqa: E402
from vgg16 import vgg16  # noqa: E402

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
    """Stable content hash for an immutable source snapshot."""
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


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_matching_manifest(path: Path, expected: Mapping[str, Any]) -> None:
    """Refuse accidental reuse of artifacts generated with another setting."""
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
    """Delete only a resolved experiment directory below an explicit root."""
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


def _checkpoint_num_classes(state_dict: Optional[Dict[str, torch.Tensor]]) -> Optional[int]:
    if state_dict is None:
        return None
    for name in ("classifier.6.weight", "module.classifier.6.weight"):
        if name in state_dict:
            return int(state_dict[name].shape[0])
    return None


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
    """Build the public CPA model with external checkpoint support."""
    if checkpoint_path is None and seed is not None:
        torch.manual_seed(seed)
    state_dict = _checkpoint_state_dict(checkpoint_path)
    checkpoint_classes = _checkpoint_num_classes(state_dict)
    if model_name == "vgg16" and ds != "imagenet":
        model = vgg16(pretrained=True)
        final_layer = model.classifier[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError(f"Expected VGG16 classifier[-1], found {type(final_layer)}")
        out_features = checkpoint_classes or nclasses_dict[ds]
        if final_layer.out_features != out_features:
            model.classifier[-1] = nn.Linear(final_layer.in_features, out_features)
    else:
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


def parameter_schema(model: torch.nn.Module) -> List[Dict[str, Any]]:
    return [
        {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
        for name, parameter in model.named_parameters()
    ]


def validate_parameter_mapping(
    names: Sequence[str], values: Sequence[Any], model: torch.nn.Module
) -> None:
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


def _sample_trial_indices(dataset_size: int, n_samples: int, attack_trials: int, seed: int) -> List[List[int]]:
    if n_samples > dataset_size:
        raise ValueError(f"n_samples={n_samples} exceeds dataset size={dataset_size}")
    result: List[List[int]] = []
    for trial in range(attack_trials):
        generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003 * trial)
        result.append(torch.randperm(dataset_size, generator=generator)[:n_samples].tolist())
    return result


def _load_indexed_test_batches(
    *, ds: str, n_samples: int, attack_trials: int, seed: int
) -> tuple[List[torch.Tensor], List[torch.Tensor], List[List[int]]]:
    # num_workers=0 and explicit Subset order make trial identities reproducible.
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


def _local_batch_orders(n_samples: int, local_batch_size: int, local_epochs: int) -> List[List[int]]:
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
    lr: float,
    seed: int,
    checkpoint_path: Optional[Path],
) -> Dict[str, Any]:
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


def collect_fedavg_update_pickle(
    *,
    ds: str,
    model_name: str,
    h_dim: int,
    n_samples: int,
    local_epochs: int,
    local_batch_size: int,
    lr: float,
    quant_bits: int,
    rounding: str,
    output_file: Path,
    seed: int,
    global_checkpoint: Optional[Path] = None,
    attack_trials: Optional[int] = None,
    n_rounds: Optional[int] = None,
) -> Path:
    """Collect reproducible independent FedAvg attack trials.

    ``n_rounds`` remains an input-only compatibility alias.  These are not
    sequential global rounds; all trials start from the same checkpoint.
    """
    if attack_trials is None:
        if n_rounds is None:
            raise TypeError("attack_trials is required")
        attack_trials = n_rounds
    if n_rounds is not None and n_rounds != attack_trials:
        raise ValueError("n_rounds compatibility alias conflicts with attack_trials")
    if quant_bits not in {4, 32}:
        raise ValueError("Only FP32 (32) and symmetric int4 (4) are supported")
    if quant_bits == 4:
        require_nearest_rounding(rounding)
    if attack_trials < 1:
        raise ValueError("attack_trials must be >= 1")

    config = collector_config(
        ds=ds,
        model_name=model_name,
        h_dim=h_dim,
        n_samples=n_samples,
        attack_trials=attack_trials,
        local_epochs=local_epochs,
        local_batch_size=local_batch_size,
        lr=lr,
        seed=seed,
        checkpoint_path=global_checkpoint,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_manifest = output_file.parent / "manifest.json"
    expected_manifest = {**config, "method": "collector", "quant_bits": quant_bits, "rounding": rounding}
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
    data: Dict[str, Any] = {
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
            "quant_bits": quant_bits,
            "rounding": rounding,
            "fake_quantization": quant_bits != 32,
            "dequantized_before_attack": quant_bits != 32,
            "trial_semantics": "independent_batches_from_same_w0",
            "multi_step_cpa_factorization": "empirical_baseline_approximation",
        },
    }

    for trial, (x_cpu, y_cpu) in enumerate(zip(xs, ys)):
        data["x"].append(x_cpu.numpy())
        data["y"].append(y_cpu.numpy())
        with torch.no_grad():
            if getattr(global_model, "model_type", None) == "conv":
                _, z = global_model(x_cpu.to(device), return_z=True)
                data["z"].append(z.detach().cpu().numpy())

        local_model = copy.deepcopy(global_model).to(device)
        local_model.train()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
        local_dataset = TensorDataset(x_cpu, y_cpu)
        local_loader = DataLoader(local_dataset, batch_size=local_batch_size, shuffle=False, num_workers=0)
        for _epoch in range(local_epochs):
            for x_batch, y_batch in local_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(local_model(x_batch.to(device)), y_batch.to(device))
                loss.backward()
                optimizer.step()

        update = _named_update_dict(local_model, global_model)
        if quant_bits == 32:
            stored_update = list(update.values())
            data["quant_stats"].append({"quant_bits": 32, "rounding": "none"})
        else:
            codes, dequantized, stats = quantize_update_symmetric(
                update=update, bits=4, rounding=rounding
            )
            stored_update = list(dequantized.values())
            if data["quant_codes"] is None:
                data["quant_codes"] = []
            data["quant_codes"].append([value.cpu().numpy() for value in codes.values()])
            data["quant_stats"].append(stats)
        data["grad"].append([value.detach().cpu().numpy() for value in stored_update])
        del local_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    data["metadata"]["sample_indices_hash"] = sha256_bytes(
        json.dumps(sample_indices).encode("utf-8")
    )
    data["metadata"]["label_hash"] = hash_tensor_sequence(data["y"])
    data["metadata"]["update_tensor_hash"] = hash_tensor_sequence(data["grad"])
    write_pickle(data, str(output_file))
    write_json(output_manifest, expected_manifest)
    return output_file


def ensure_shared_fp32_update(args: argparse.Namespace) -> Path:
    config = collector_config(
        ds=args.ds,
        model_name=args.model,
        h_dim=args.h_dim,
        n_samples=args.n_samples,
        attack_trials=args.attack_trials,
        local_epochs=args.local_epochs,
        local_batch_size=resolve_local_batch_size(args),
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
        lr=args.local_lr,
        quant_bits=32,
        rounding="nearest",
        output_file=path,
        seed=args.seed,
        global_checkpoint=args.global_checkpoint,
    )


def derive_quantized_update_pickle(
    *, fp32_update_file: Path, output_file: Path, rounding: str = "nearest"
) -> Path:
    """Derive int4 from the exact shared FP32 artifact, never recollect samples."""
    require_nearest_rounding(rounding)
    fp32 = read_pickle(fp32_update_file)
    if fp32.get("metadata", {}).get("quant_bits") != 32:
        raise ValueError("Int4 artifacts must be derived from an FP32 collector artifact")
    manifest = {
        "schema_version": 2,
        "method": "symmetric_int4_from_shared_fp32",
        "source_fp32_path": str(fp32_update_file.resolve()),
        "source_fp32_hash": sha256_file(fp32_update_file),
        "rounding": rounding,
        "author_source_hash": AUTHOR_SOURCE_HASH,
    }
    manifest_file = output_file.parent / "manifest.json"
    if output_file.exists():
        assert_matching_manifest(manifest_file, manifest)
        return output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result = copy.deepcopy(fp32)
    names = result["param_names"]
    result["grad"] = []
    result["quant_codes"] = []
    result["quant_stats"] = []
    for trial_update in fp32["grad"]:
        update = OrderedDict((name, torch.as_tensor(value).float()) for name, value in zip(names, trial_update))
        codes, dequantized, stats = quantize_update_symmetric(update=update, bits=4, rounding=rounding)
        result["grad"].append([value.cpu().numpy() for value in dequantized.values()])
        result["quant_codes"].append([value.cpu().numpy() for value in codes.values()])
        result["quant_stats"].append(stats)
    result["metadata"].update(
        {
            "quant_bits": 4,
            "rounding": rounding,
            "fake_quantization": True,
            "dequantized_before_attack": True,
            "quantization_scheme": "symmetric_per_parameter_tensor_zero_point_none_codes_-7_to_7",
            "source_fp32_update_file": str(fp32_update_file.resolve()),
            "source_fp32_update_hash": sha256_file(fp32_update_file),
            "update_tensor_hash": hash_tensor_sequence(result["grad"]),
            "quantization_scales_hash": sha256_bytes(
                json.dumps(result["quant_stats"], sort_keys=True).encode("utf-8")
            ),
        }
    )
    write_pickle(result, str(output_file))
    write_json(manifest_file, manifest)
    return output_file


def output_pickle_path(
    experiment_name: str, ds: str, model: str, n_samples: int, *, local_steps: Optional[int] = None,
    attack_trials: Optional[int] = None,
) -> Path:
    suffix = f"n{n_samples}"
    if local_steps is not None:
        suffix += f"_steps{local_steps}"
    if attack_trials is not None:
        suffix += f"_trials{attack_trials}"
    return OUTPUT_ROOT / experiment_name / ds / model / "updates" / f"{suffix}.pickle"


def run_tag(args: argparse.Namespace, *, include_fia: bool = True) -> str:
    """Human-readable setting tag for output directories and files."""
    local_batch_size = resolve_local_batch_size(args)
    local_steps = args.local_epochs * ((args.n_samples + local_batch_size - 1) // local_batch_size)
    tag = f"n{args.n_samples}_steps{local_steps}_trials{args.attack_trials}"
    if include_fia:
        n_sample_fi = args.n_samples if args.n_sample_fi in {-1, 0} else min(args.n_sample_fi, args.n_samples)
        tag += f"_fia{n_sample_fi}"
    return tag


def update_experiment_manifest(
    *, experiment_id: str, method: str, update_file: Path, args: argparse.Namespace,
    evaluated_samples: int, extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    data = read_pickle(update_file)
    metadata = data["metadata"]
    manifest: Dict[str, Any] = {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "method": method,
        "author_source_hash": AUTHOR_SOURCE_HASH,
        "checkpoint": metadata["checkpoint"],
        "update_file": str(update_file.resolve()),
        "update_file_hash": sha256_file(update_file),
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
        "bit_width": metadata["quant_bits"],
        "rounding_mode": metadata["rounding"],
        "quantization_scales_hash": metadata.get("quantization_scales_hash"),
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
    parser.add_argument("--local_lr", type=float, default=1e-3)
    parser.add_argument("--global_checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse_existing", action="store_true", help="Only reuse when manifests match exactly.")
    parser.add_argument("--run_attack", action="store_true")


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
    parser.add_argument("--fi", type=float, default=1.0)
    parser.add_argument("--fi_method", type=str, default="direct", choices=["direct", "gm"])
    parser.add_argument("--use_labels", action="store_true")
    parser.add_argument("--ideal_emb_rec", action="store_true")
    parser.add_argument("--fresh_start", action="store_true")
    parser.add_argument("--project", type=str, default="cpa_test")


def make_attack_namespace(args: argparse.Namespace, *, exp_name: str) -> argparse.Namespace:
    n_sample_fi = args.n_samples if args.n_sample_fi in {-1, 0} else min(args.n_sample_fi, args.n_samples)
    return argparse.Namespace(
        model=args.model, h_dim=args.h_dim, ds=args.ds, batch_size=args.n_samples, attack="cp",
        decor=args.decor, tv=args.tv, ne=args.ne, l1=args.l1, nv=args.nv, T=args.T,
        n_batch=args.attack_n_batch or args.attack_trials, n_iter=args.attack_n_iter,
        n_log=args.attack_n_log, opt="adam", lr=args.attack_lr, lr_N=1e-6, sch="none",
        fi_method=args.fi_method, n_iter_fi=args.n_iter_fi, n_log_fi=args.n_log_fi,
        opt_fi="adam", lr_fi=args.lr_fi, sch_fi="cosine", n_sample_fi=n_sample_fi,
        gm=0.0, fi=args.fi, fl_alg="fedavg", C=1.0, sigma=0.0, defense="nodef",
        exp_name=exp_name, project=args.project, fresh_start=args.fresh_start, dry_run=True,
        submitit=False, n_gpu=1, timeout=600, wait=False, disable_pbar=True,
        use_labels=args.use_labels, ideal_emb_rec=args.ideal_emb_rec,
    )


def resolve_local_batch_size(args: argparse.Namespace) -> int:
    return args.local_batch_size or args.n_samples


def _install_optional_dependency_stubs() -> None:
    """Allow author classes to import in dry-run environments without wandb."""
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
