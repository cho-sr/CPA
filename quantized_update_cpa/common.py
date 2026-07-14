#!/usr/bin/env python3
"""Shared utilities for FedAvg-update CPA experiments."""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import types
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

for thread_env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ.setdefault(thread_env_var, "1")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


REPO_ROOT = Path(__file__).resolve().parents[1]
CPA_SRC = REPO_ROOT / "cocktail_party_attack" / "src"
FEDAVG_SRC = REPO_ROOT / "fedavg_fp32"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DATASET_ROOT = REPO_ROOT / "datasets"

for path in (CPA_SRC, FEDAVG_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import datasets as cpa_datasets  # noqa: E402
from datasets import get_dataloaders, nclasses_dict  # noqa: E402
from models import get_model  # noqa: E402
from quantization import quantize_dequantize_update  # noqa: E402
from utils import exp_path_base, get_device, write_pickle  # noqa: E402
from vgg16 import vgg16  # noqa: E402

cpa_datasets.ds_root = str(DATASET_ROOT)


def output_pickle_path(
    experiment_name: str,
    ds: str,
    model: str,
    n_samples: int,
) -> Path:
    return OUTPUT_ROOT / experiment_name / ds / model / "updates" / f"{n_samples}.pickle"


def _as_numpy_list(update: Sequence[torch.Tensor]) -> List:
    return [tensor.detach().cpu().numpy() for tensor in update]


def _named_update_dict(
    local_model: torch.nn.Module,
    global_model: torch.nn.Module,
) -> "OrderedDict[str, torch.Tensor]":
    local_params = OrderedDict(local_model.named_parameters())
    global_params = OrderedDict(global_model.named_parameters())
    if local_params.keys() != global_params.keys():
        raise ValueError("Local and global model parameter names do not match")

    update: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name in global_params:
        update[name] = (
            local_params[name].detach() - global_params[name].detach()
        ).to(dtype=torch.float32)
    return update


def _update_dict_to_ordered_list(
    update: Dict[str, torch.Tensor],
    names: Sequence[str],
) -> List[torch.Tensor]:
    return [update[name].detach().to(dtype=torch.float32) for name in names]


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
    final_weight = state_dict.get("classifier.6.weight")
    if final_weight is None:
        final_weight = state_dict.get("module.classifier.6.weight")
    if final_weight is None:
        return None
    return int(final_weight.shape[0])


def build_experiment_model(
    *,
    model_name: str,
    ds: str,
    h_dim: int,
    device: torch.device,
    checkpoint_path: Optional[Path] = None,
) -> torch.nn.Module:
    """Build the CPA model, with FedAvg Tiny-ImageNet VGG16 checkpoint support."""
    state_dict = _checkpoint_state_dict(checkpoint_path)
    checkpoint_classes = _checkpoint_num_classes(state_dict)

    if model_name == "vgg16" and ds != "imagenet":
        model = vgg16(pretrained=True)
        final_layer = model.classifier[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError(f"Expected VGG16 classifier[-1] to be nn.Linear, found {type(final_layer)}")
        out_features = checkpoint_classes or nclasses_dict[ds]
        if final_layer.out_features != out_features:
            model.classifier[-1] = nn.Linear(final_layer.in_features, out_features)
    else:
        model = get_model(
            model_name=model_name,
            ds=ds,
            h_dim=h_dim,
            load_path=f"{exp_path_base}/{ds}/{model_name}/model.pt",
            dataparallel=False,
        )
        if checkpoint_classes is not None and hasattr(model, "classifier"):
            final_layer = model.classifier[-1]
            if isinstance(final_layer, nn.Linear) and final_layer.out_features != checkpoint_classes:
                model.classifier[-1] = nn.Linear(final_layer.in_features, checkpoint_classes)

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model.to(device=device, dtype=torch.float32)


def collect_fedavg_update_pickle(
    *,
    ds: str,
    model_name: str,
    h_dim: int,
    n_samples: int,
    n_rounds: int,
    local_epochs: int,
    local_batch_size: int,
    lr: float,
    quant_bits: int,
    rounding: str,
    output_file: Path,
    seed: int,
    global_checkpoint: Optional[Path] = None,
) -> Path:
    """Collect FedAvg local updates in the pickle format consumed by CPA."""
    if quant_bits not in {32, 8, 4}:
        raise ValueError("--quant_bits must be one of 32, 8, or 4")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    logger = logging.getLogger("quantized_update_cpa.collect")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = get_device()

    logger.info(
        "Collecting %s %s FedAvg updates: n_samples=%s n_rounds=%s "
        "local_epochs=%s local_batch_size=%s quant_bits=%s",
        ds,
        model_name,
        n_samples,
        n_rounds,
        local_epochs,
        local_batch_size,
        quant_bits,
    )

    global_model = build_experiment_model(
        model_name=model_name,
        ds=ds,
        h_dim=h_dim,
        device=device,
        checkpoint_path=global_checkpoint,
    )
    global_model.eval()

    _, dl_test = get_dataloaders(ds, batch_size=n_samples, shuffle_test=True)
    test_iter = iter(dl_test)
    criterion = nn.CrossEntropyLoss()

    data = {
        "x": [],
        "y": [],
        "z": [],
        "grad": [],
        "metadata": {
            "signal": "fedavg_local_update",
            "update_definition": "delta_w = w_local - w_global",
            "quant_bits": quant_bits,
            "rounding": rounding,
            "fake_quantization": quant_bits != 32,
            "dequantized_before_attack": True,
            "n_samples": n_samples,
            "n_rounds": n_rounds,
            "local_epochs": local_epochs,
            "local_batch_size": local_batch_size,
            "lr": lr,
            "seed": seed,
            "dataset_root": str(DATASET_ROOT),
            "global_checkpoint": str(global_checkpoint) if global_checkpoint else None,
        },
        "quant_stats": [],
    }

    for round_idx in range(n_rounds):
        try:
            x_cpu, y_cpu = next(test_iter)
        except StopIteration:
            test_iter = iter(dl_test)
            x_cpu, y_cpu = next(test_iter)

        data["x"].append(x_cpu.numpy())
        data["y"].append(y_cpu.numpy())

        x_full = x_cpu.to(device)
        with torch.no_grad():
            if getattr(global_model, "model_type", None) == "conv":
                _, z = global_model(x_full, return_z=True)
                data["z"].append(z.detach().cpu().numpy())

        local_model = copy.deepcopy(global_model).to(device)
        local_model.train()
        opt = torch.optim.SGD(local_model.parameters(), lr=lr)

        local_ds = TensorDataset(x_cpu, y_cpu)
        local_loader = DataLoader(local_ds, batch_size=local_batch_size, shuffle=True)
        for _ in range(local_epochs):
            for x_batch, y_batch in local_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                opt.zero_grad(set_to_none=True)
                pred = local_model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                opt.step()

        update = _named_update_dict(local_model, global_model)
        names = list(update.keys())
        if quant_bits == 32:
            saved_update = _update_dict_to_ordered_list(update, names)
            quant_stats = {
                "quant_bits": 32,
                "quant_mse": 0.0,
                "quant_relative_l2": 0.0,
                "quant_cosine_similarity": 1.0,
            }
        else:
            generator = None
            if rounding == "stochastic":
                generator = torch.Generator(device=device).manual_seed(
                    seed + round_idx * 1_000_003
                )
            quantized_update, quant_stats = quantize_dequantize_update(
                update=update,
                bits=quant_bits,
                rounding=rounding,
                generator=generator,
            )
            saved_update = _update_dict_to_ordered_list(quantized_update, names)

        data["grad"].append(_as_numpy_list(saved_update))
        data["quant_stats"].append(quant_stats)
        logger.info("Collected round %s/%s", round_idx + 1, n_rounds)

        del local_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_pickle(data, str(output_file))
    logger.info("Saved update pickle: %s", output_file)
    return output_file


def add_collect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ds", type=str, default="imagenet")
    parser.add_argument("--model", type=str, default="vgg16")
    parser.add_argument("--h_dim", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--n_rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--local_batch_size", type=int, default=None)
    parser.add_argument("--local_lr", type=float, default=1e-3)
    parser.add_argument(
        "--global_checkpoint",
        type=Path,
        default=None,
        help="Optional FedAvg global checkpoint .pt to use as w_global.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse_existing", action="store_true")
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
    parser.add_argument("--gm", type=float, default=1.0)
    parser.add_argument("--fi", type=float, default=1.0)
    parser.add_argument(
        "--fi_method",
        type=str,
        default="direct",
        choices=["direct", "gm", "qgm"],
    )
    parser.add_argument("--quant_bits", type=int, default=4)
    parser.add_argument(
        "--qgm_metric",
        type=str,
        default="relative_l2",
        choices=["relative_l2", "cosine"],
        help="Consistency metric for quantized-gradient matching FIA.",
    )
    parser.add_argument("--use_labels", action="store_true")
    parser.add_argument("--ideal_emb_rec", action="store_true")
    parser.add_argument(
        "--dry_run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Disable wandb logging. This is the default for these wrappers.",
    )
    parser.add_argument(
        "--no_dry_run",
        dest="dry_run",
        action="store_false",
        help="Enable wandb logging if wandb is installed and configured.",
    )
    parser.add_argument("--fresh_start", action="store_true")
    parser.add_argument("--project", type=str, default="cpa_test")


def make_attack_namespace(
    args: argparse.Namespace,
    *,
    exp_name: str,
    quantized_update_file: Path,
) -> argparse.Namespace:
    n_sample_fi = args.n_sample_fi
    if n_sample_fi == -1 or n_sample_fi > args.n_samples:
        n_sample_fi = args.n_samples

    attack_args = argparse.Namespace(
        model=args.model,
        h_dim=args.h_dim,
        ds=args.ds,
        batch_size=args.n_samples,
        attack="cp",
        decor=args.decor,
        tv=args.tv,
        ne=args.ne,
        l1=args.l1,
        nv=args.nv,
        T=args.T,
        n_batch=args.attack_n_batch or args.n_rounds,
        n_iter=args.attack_n_iter,
        n_log=args.attack_n_log,
        opt="adam",
        lr=args.attack_lr,
        lr_N=1e-6,
        sch="none",
        fi_method=args.fi_method,
        quant_bits=args.quant_bits,
        qgm_metric=args.qgm_metric,
        n_iter_fi=args.n_iter_fi,
        n_log_fi=args.n_log_fi,
        opt_fi="adam",
        lr_fi=args.lr_fi,
        sch_fi="cosine",
        n_sample_fi=n_sample_fi,
        gm=args.gm,
        fi=args.fi,
        fl_alg="fedavg",
        C=1.0,
        sigma=0.0,
        defense="nodef",
        exp_name=exp_name,
        project=args.project,
        fresh_start=args.fresh_start,
        dry_run=args.dry_run,
        submitit=False,
        n_gpu=1,
        timeout=600,
        wait=False,
        disable_pbar=True,
        use_labels=args.use_labels,
        ideal_emb_rec=args.ideal_emb_rec,
    )

    return attack_args


def run_attack_with_update_file(
    args: argparse.Namespace,
    *,
    exp_name: str,
    update_file: Path,
) -> None:
    _install_optional_dependency_stubs()
    try:
        import attack as attack_module
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise RuntimeError(
            "Running CPA requires the original attack dependencies. Install them with "
            "`pip install -r cocktail_party_attack/requirements.txt` and retry. "
            f"Missing module: {missing}"
        ) from exc

    attack_args = make_attack_namespace(
        args,
        exp_name=exp_name,
        quantized_update_file=update_file,
    )
    original_get_updates_file = attack_module.get_updates_file
    original_get_model = attack_module.get_model
    attack_module.get_updates_file = lambda ds, model, n_samples: str(update_file)
    if args.global_checkpoint is not None or (args.model == "vgg16" and args.ds != "imagenet"):
        attack_module.get_model = (
            lambda model_name, ds, h_dim=256, load_path=None, dataparallel=True: build_experiment_model(
                model_name=model_name,
                ds=ds,
                h_dim=h_dim,
                device=get_device(),
                checkpoint_path=args.global_checkpoint,
            )
        )
    try:
        attack_module.attack(attack_args)
    finally:
        attack_module.get_updates_file = original_get_updates_file
        attack_module.get_model = original_get_model


def resolve_local_batch_size(args: argparse.Namespace) -> int:
    return args.local_batch_size or args.n_samples


def _install_optional_dependency_stubs() -> None:
    """Allow local dry-run attacks when optional logging/Slurm deps are absent."""
    if "submitit" not in sys.modules:
        try:
            __import__("submitit")
        except ModuleNotFoundError:
            submitit_stub = types.ModuleType("submitit")

            class _AutoExecutor:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError(
                        "submitit is not installed; run without --submitit or install requirements.txt"
                    )

            submitit_stub.AutoExecutor = _AutoExecutor
            sys.modules["submitit"] = submitit_stub

    if "wandb" not in sys.modules:
        try:
            __import__("wandb")
        except ModuleNotFoundError:
            wandb_stub = types.ModuleType("wandb")

            class _Config:
                def update(self, *args, **kwargs):
                    return None

            class _Run:
                id = "dry-run"

                def get_url(self):
                    return "wandb-disabled"

            def _init(*args, **kwargs):
                wandb_stub.run = _Run()
                return wandb_stub.run

            def _log(*args, **kwargs):
                return None

            class _Image:
                def __init__(self, *args, **kwargs):
                    self.args = args
                    self.kwargs = kwargs

            wandb_stub.config = _Config()
            wandb_stub.run = _Run()
            wandb_stub.init = _init
            wandb_stub.log = _log
            wandb_stub.Image = _Image
            sys.modules["wandb"] = wandb_stub
