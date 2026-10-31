#!/usr/bin/env python3
"""End-to-end quant-aware CPA from final-round aggregated int4 update.

This runner optimizes pixels directly against the observed server-side
aggregated FedAvg update.  It does not use recovered embeddings or per-client
q/s bins as reconstruction targets.  The main path is:

    aggregated Delta theta_obs -> L(theta_agg(x, y) - Delta theta_obs) -> x, y

where theta_agg is produced by differentiably unrolling each client's local SGD
and applying the stored FedAvg aggregation weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_SRC = REPO_ROOT / "baseline" / "src"
QICA_SRC = REPO_ROOT / "int 4 fedavg" / "qaware_client_qica"
for path in (BASELINE_SRC, QICA_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from datasets import ds_mean, ds_std, nclasses_dict  # noqa: E402
from models import get_model  # noqa: E402
from qica_core import finite_bin_bounds  # noqa: E402

try:  # noqa: E402
    from torch.func import functional_call
except ImportError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call  # type: ignore


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "int 4 fedavg"
    / "imagenet"
    / "vgg16"
    / "clients3_rounds20_spc8_seed42"
)


@dataclass
class TensorTarget:
    name: str
    observed: torch.Tensor
    lower: torch.Tensor | None
    upper: torch.Tensor | None
    consistency: str
    edge_code_ratio: float | None
    numel: int


@dataclass
class ClientBatch:
    client_id: int
    weight: float
    sample_slice: slice
    global_indices: List[int]


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct aggregated-update -> (x,y) quant-aware CPA for int4 FedAvg artifacts."
    )
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--aggregate_pickle", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--client_ids", type=str, default="all", help="'all' or comma-separated client ids")
    parser.add_argument("--loss_scope", type=str, default="attack_layer", choices=["attack_layer", "all", "names"])
    parser.add_argument("--loss_param_names", type=str, default="", help="Comma-separated names when --loss_scope names")
    parser.add_argument("--label_mode", type=str, default="known", choices=["known", "unknown"])
    parser.add_argument(
        "--agg_consistency",
        type=str,
        default="client_bin_sum",
        choices=["client_bin_sum", "aggregate_requant", "mse"],
        help="Observed aggregate consistency: FedAvg-summed client bins, aggregate requantized bin, or MSE baseline.",
    )
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--n_log", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--q_weight", type=float, default=1.0)
    parser.add_argument("--tv_weight", type=float, default=1e-4)
    parser.add_argument(
        "--mi_weight",
        type=float,
        default=0.0,
        help="Weight for reconstructed-image dependence penalty; uses pairwise squared correlation as a MI proxy.",
    )
    parser.add_argument(
        "--mi_downsample",
        type=int,
        default=32,
        help="Spatial size for the MI proxy; 0 computes on the full image.",
    )
    parser.add_argument(
        "--ne_weight",
        type=float,
        default=0.0,
        help="Weight for non-Gaussianity prior; minimizes negative logcosh negentropy proxy on image gradients.",
    )
    parser.add_argument(
        "--ne_alpha",
        type=float,
        default=1.0,
        help="Alpha for logcosh non-Gaussianity proxy.",
    )
    parser.add_argument(
        "--local_steps_override",
        type=int,
        default=None,
        help="Use fewer differentiable local-SGD steps than the artifact for memory-limited approximation runs.",
    )
    parser.add_argument(
        "--inner_microbatch_size",
        type=int,
        default=0,
        help="Microbatch size for the differentiable full-batch local-SGD gradient; 0 disables microbatching.",
    )
    parser.add_argument(
        "--unroll_param_scope",
        type=str,
        default="loss_params",
        choices=["loss_params", "full"],
        help="Update only loss-scope parameters in the differentiable inner loop by default; full is exact but memory-heavy.",
    )
    parser.add_argument(
        "--q_row_chunk",
        type=int,
        default=128,
        help="Rows per chunk when evaluating large 2D aggregated update tensors; 0 computes at once.",
    )
    parser.add_argument(
        "--q_row_sample",
        type=int,
        default=128,
        help="Deterministic row subset for large 2D aggregated update losses; 0 uses all rows.",
    )
    parser.add_argument("--init_pixels", type=str, default="random", choices=["random"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--fresh_start", action="store_true")
    parser.add_argument("--q_grad_check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad_nonzero_eps", type=float, default=1e-12)
    parser.add_argument("--max_png_items", type=int, default=24)
    parser.add_argument("--skip_ssim", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def safe_rmtree(path: Path, *, allowed_root: Path) -> None:
    root = allowed_root.resolve()
    target = path.resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"refusing to delete outside {root}: {target}")
    if target.exists():
        shutil.rmtree(target)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)


def write_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def write_pickle(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except ModuleNotFoundError:
            handle.seek(0)
            return CompatUnpickler(handle).load()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_client_ids(value: str, n_clients: int) -> List[int]:
    if value == "all":
        return list(range(n_clients))
    ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not ids:
        raise ValueError("--client_ids selected no clients")
    bad = [idx for idx in ids if idx < 0 or idx >= n_clients]
    if bad:
        raise ValueError(f"--client_ids outside [0,{n_clients - 1}]: {bad}")
    return ids


def selected_param_names(args: argparse.Namespace, run_manifest: Mapping[str, Any], quant_meta: Mapping[str, Any]) -> List[str]:
    if args.loss_scope == "attack_layer":
        return [str(quant_meta["attack_layer"]["name"])]
    all_names = [str(name) for name in run_manifest["param_names"]]
    if args.loss_scope == "all":
        return all_names
    names = [name.strip() for name in args.loss_param_names.split(",") if name.strip()]
    missing = sorted(set(names) - set(all_names))
    if missing:
        raise ValueError(f"--loss_param_names not in q/s artifact: {missing}")
    return names


def validate_metadata(run_manifest: Mapping[str, Any], quant_meta: Mapping[str, Any]) -> None:
    if int(run_manifest.get("quant_bits", 4)) != 4:
        raise ValueError("This runner requires int4 quantized updates")
    if str(run_manifest.get("rounding", quant_meta.get("rounding", "nearest"))) != "nearest":
        raise ValueError("This runner supports nearest rounding only")
    if str(run_manifest.get("optimizer", "SGD")) != "SGD":
        raise ValueError("This runner currently supports only SGD")
    if float(run_manifest.get("momentum", 0.0)) != 0.0:
        raise ValueError("This runner requires momentum=0")
    if float(run_manifest.get("weight_decay", 0.0)) != 0.0:
        raise ValueError("This runner requires weight_decay=0")
    if int(run_manifest["local_batch_size"]) != int(run_manifest["samples_per_client"]):
        raise ValueError("This runner requires full-batch local SGD: local_batch_size == samples_per_client")
    if str(quant_meta.get("artifact_type")) != "final_round_client_qs":
        raise ValueError("Expected final_round_client_qs artifact")
    for client in quant_meta["clients"]:
        stats = client.get("quant_stats", {})
        if int(stats.get("quant_bits", 4)) != 4:
            raise ValueError("All client q/s tensors must be int4")
        if str(stats.get("rounding", "nearest")) != "nearest":
            raise ValueError("All client q/s tensors must use nearest rounding")


def validate_model_supported(model: torch.nn.Module) -> None:
    active_dropout = []
    batchnorm = []
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and float(module.p) > 0.0:
            active_dropout.append(module)
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            batchnorm.append(module)
    if active_dropout:
        raise ValueError("Deterministic local-SGD unroll requires no active dropout")
    if batchnorm:
        raise ValueError("This runner refuses BatchNorm/stateful normalization layers")


def load_model(quant_meta: Mapping[str, Any], device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    state_path = resolve_path(Path(str(quant_meta["source_model_state_file"])))
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    model = get_model(model_name="vgg16", ds="imagenet", h_dim=None, dataparallel=False)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    validate_model_supported(model)
    return model, {"checkpoint_path": str(state_path), "checkpoint_sha256": sha256_file(state_path)}


def _state_dict_params(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, p.detach().clone().requires_grad_(True)) for name, p in model.named_parameters())


def _param_tuple(params: OrderedDict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    return tuple(params.values())


def _make_param_dict(names: Sequence[str], values: Sequence[torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, value) for name, value in zip(names, values))


def normalize_pixels(pixels: torch.Tensor, ds: str = "imagenet") -> torch.Tensor:
    mean = torch.tensor(ds_mean[ds], device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    std = torch.tensor(ds_std[ds], device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    return (pixels - mean) / std


def total_variation(pixels: torch.Tensor) -> torch.Tensor:
    dx = torch.mean(torch.abs(pixels[:, :, :, :-1] - pixels[:, :, :, 1:]))
    dy = torch.mean(torch.abs(pixels[:, :, :-1, :] - pixels[:, :, 1:, :]))
    return dx + dy


def image_mi_proxy_loss(pixels: torch.Tensor, downsample: int) -> torch.Tensor:
    """Penalize dependence between reconstructed images with pairwise corr^2.

    For Gaussian variables, mutual information is monotone in correlation. This
    is a cheap differentiable proxy, not a full nonparametric MI estimator.
    """
    n = int(pixels.shape[0])
    if n < 2:
        return torch.zeros((), device=pixels.device, dtype=pixels.dtype)
    if downsample > 0 and (pixels.shape[-2] != downsample or pixels.shape[-1] != downsample):
        pooled = F.adaptive_avg_pool2d(pixels, (downsample, downsample))
    else:
        pooled = pixels
    flat = pooled.flatten(start_dim=1)
    flat = flat - flat.mean(dim=1, keepdim=True)
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    corr = flat @ flat.t()
    mask = ~torch.eye(n, dtype=torch.bool, device=pixels.device)
    return corr[mask].square().mean()


def non_gaussianity_prior_loss(pixels: torch.Tensor, alpha: float) -> torch.Tensor:
    """Encourage heavy-tailed natural-image gradients with a logcosh NE proxy."""
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError("--ne_alpha must be positive")
    gx = pixels[:, :, :, 1:] - pixels[:, :, :, :-1]
    gy = pixels[:, :, 1:, :] - pixels[:, :, :-1, :]
    values = torch.cat((gx.flatten(), gy.flatten()))
    values = values - values.mean()
    values = values / values.std(unbiased=False).clamp_min(1e-6)
    # Minimized as a negative prior: larger logcosh response means more
    # non-Gaussian/heavy-tailed image-gradient statistics.
    scaled = alpha * values
    log_cosh = scaled + F.softplus(-2.0 * scaled) - math.log(2.0)
    return -log_cosh.mean() / alpha


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    return -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def forward_with_params(model: torch.nn.Module, params: OrderedDict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return functional_call(model, params, (x,))


def unroll_full_batch_sgd(
    *,
    model: torch.nn.Module,
    theta0: OrderedDict[str, torch.Tensor],
    x_norm: torch.Tensor,
    labels: torch.Tensor | None,
    label_probs: torch.Tensor | None,
    steps: int,
    lr: float,
    create_graph: bool,
    microbatch_size: int,
    update_names: Sequence[str] | None,
) -> OrderedDict[str, torch.Tensor]:
    names = list(theta0.keys())
    params = OrderedDict((name, value) for name, value in theta0.items())
    if update_names is None:
        grad_names = names
    else:
        missing = sorted(set(update_names) - set(names))
        if missing:
            raise ValueError(f"inner-loop update params are missing from model: {missing}")
        grad_names = list(update_names)
    for _step in range(steps):
        param_tuple = tuple(params[name] for name in grad_names)
        n_samples = int(x_norm.shape[0])
        mb = int(microbatch_size)
        if mb <= 0 or mb >= n_samples:
            pred = forward_with_params(model, params, x_norm)
            if label_probs is not None:
                loss = soft_cross_entropy(pred, label_probs)
            elif labels is not None:
                loss = F.cross_entropy(pred, labels)
            else:
                raise ValueError("Either labels or label_probs are required")
            grads = torch.autograd.grad(
                loss,
                param_tuple,
                create_graph=create_graph,
                retain_graph=create_graph,
            )
        else:
            grad_sums = None
            for start in range(0, n_samples, mb):
                end = min(start + mb, n_samples)
                weight = float(end - start) / float(n_samples)
                pred = forward_with_params(model, params, x_norm[start:end])
                if label_probs is not None:
                    loss = soft_cross_entropy(pred, label_probs[start:end]) * weight
                elif labels is not None:
                    loss = F.cross_entropy(pred, labels[start:end]) * weight
                else:
                    raise ValueError("Either labels or label_probs are required")
                part_grads = torch.autograd.grad(
                    loss,
                    param_tuple,
                    create_graph=create_graph,
                    retain_graph=create_graph,
                )
                if grad_sums is None:
                    grad_sums = list(part_grads)
                else:
                    grad_sums = [acc + grad for acc, grad in zip(grad_sums, part_grads)]
            if grad_sums is None:
                raise ValueError("empty local-SGD batch")
            grads = tuple(grad_sums)
        next_params = OrderedDict((name, value) for name, value in params.items())
        for name, grad in zip(grad_names, grads):
            next_params[name] = params[name] - lr * grad
        params = next_params
    return params


def client_sample_slices(context: Mapping[str, Any], client_ids: Sequence[int]) -> Tuple[Dict[int, slice], List[int]]:
    per_client = context.get("client_original_indices")
    if per_client is None:
        raise ValueError("final_round_context.pkl is missing client_original_indices")
    offsets = []
    cursor = 0
    for rows in per_client:
        offsets.append(cursor)
        cursor += len(rows)
    slices: Dict[int, slice] = {}
    selected: List[int] = []
    for client_id in client_ids:
        start = offsets[client_id]
        end = start + len(per_client[client_id])
        slices[client_id] = slice(len(selected), len(selected) + end - start)
        selected.extend(range(start, end))
    return slices, selected


def tensor_meta_by_name(client: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(row["name"]): row for row in client.get("tensors", [])}


def load_clients(
    *,
    quant_meta: Mapping[str, Any],
    context: Mapping[str, Any],
    client_ids: Sequence[int],
) -> Tuple[List[ClientBatch], List[int]]:
    sample_slices, selected_global_indices = client_sample_slices(context, client_ids)
    clients: List[ClientBatch] = []
    for client_id in client_ids:
        client_meta = quant_meta["clients"][client_id]
        sl = sample_slices[client_id]
        clients.append(
            ClientBatch(
                client_id=client_id,
                weight=float(client_meta["aggregation_weight"]),
                sample_slice=sl,
                global_indices=selected_global_indices[sl.start : sl.stop],
            )
        )
    return clients, selected_global_indices


def default_aggregate_pickle(run_dir: Path) -> Path:
    manifest = read_json(run_dir / "manifest.json")
    rounds = int(manifest.get("rounds", 20))
    return run_dir / f"round_{rounds:03d}_int4_dequant_aggregate_cpa.pickle"


def load_aggregate_targets(
    *,
    args: argparse.Namespace,
    aggregate_pickle: Path,
    run_manifest: Mapping[str, Any],
    quant_meta: Mapping[str, Any],
    client_qs_dir: Path,
    names: Sequence[str],
    device: torch.device,
) -> Tuple[List[TensorTarget], Dict[str, Any]]:
    data = read_pickle(aggregate_pickle)
    grad_rows = data["grad"][0]
    param_names = [str(name) for name in run_manifest["param_names"]]
    codes_path = client_qs_dir / "final_round_client_codes.npz"
    targets: List[TensorTarget] = []
    with np.load(codes_path) as npz:
        for name in names:
            index = param_names.index(name)
            observed = torch.as_tensor(np.asarray(grad_rows[index], dtype=np.float32), device=device)
            lower: torch.Tensor | None = None
            upper: torch.Tensor | None = None
            edge_code_ratio: float | None = None
            if args.agg_consistency == "client_bin_sum":
                lower_sum = torch.zeros_like(observed)
                upper_sum = torch.zeros_like(observed)
                edge_ratio_sum = 0.0
                weight_sum = 0.0
                for client in quant_meta["clients"]:
                    tensor_meta = tensor_meta_by_name(client)[name]
                    codes_np = np.asarray(npz[str(tensor_meta["key"])], dtype=np.int8)
                    codes = torch.as_tensor(codes_np, dtype=torch.int8, device=device)
                    edge_ratio = float((np.abs(codes_np) == 7).mean())
                    client_lower, client_upper = finite_bin_bounds(
                        codes,
                        float(tensor_meta["scale"]),
                        float(tensor_meta["max_abs"]),
                    )
                    weight = float(client["aggregation_weight"])
                    lower_sum = lower_sum + weight * client_lower
                    upper_sum = upper_sum + weight * client_upper
                    edge_ratio_sum += weight * edge_ratio
                    weight_sum += weight
                lower = lower_sum
                upper = upper_sum
                edge_code_ratio = edge_ratio_sum / max(weight_sum, 1e-12)
            elif args.agg_consistency == "aggregate_requant":
                max_abs = float(observed.detach().abs().max().cpu())
                scale = max_abs / 7.0 if max_abs > 0.0 else 0.0
                if scale > 0.0:
                    codes = torch.round(observed / scale).clamp(-7, 7).to(torch.int8)
                else:
                    codes = torch.zeros_like(observed, dtype=torch.int8)
                edge_code_ratio = float((codes.detach().abs() == 7).float().mean().cpu())
                lower, upper = finite_bin_bounds(codes, scale, max_abs)
            elif args.agg_consistency != "mse":
                raise ValueError(f"unknown agg consistency: {args.agg_consistency}")
            targets.append(
                TensorTarget(
                    name=name,
                    observed=observed,
                    lower=lower,
                    upper=upper,
                    consistency=args.agg_consistency,
                    edge_code_ratio=edge_code_ratio,
                    numel=int(observed.numel()),
                )
            )
    meta = data.get("meta", [{}])[0]
    return targets, {
        "path": str(aggregate_pickle),
        "sha256": sha256_file(aggregate_pickle),
        "meta": meta,
        "signal": meta.get("update_definition", "server aggregate update"),
        "agg_consistency": args.agg_consistency,
        "target_edge_code_ratios": {target.name: target.edge_code_ratio for target in targets},
        "bin_source": (
            "FedAvg weighted sum of each client's exact int4 bins"
            if args.agg_consistency == "client_bin_sum"
            else "aggregate update requantized to int4"
            if args.agg_consistency == "aggregate_requant"
            else "MSE to stored dequantized aggregate update"
        ),
    }


def select_rows_for_target(tensor: torch.Tensor, row_sample: int) -> Tuple[torch.Tensor, torch.Tensor | None]:
    if tensor.ndim >= 2 and row_sample > 0 and row_sample < int(tensor.shape[0]):
        rows = torch.linspace(0, int(tensor.shape[0]) - 1, steps=int(row_sample), device=tensor.device)
        rows = rows.round().to(torch.long).unique(sorted=True)
        return tensor.index_select(0, rows), rows
    return tensor, None


def selected_row_indices_for_targets(
    targets: Sequence[TensorTarget],
    row_sample: int,
    device: torch.device,
) -> Dict[str, torch.Tensor | None]:
    rows_by_name: Dict[str, torch.Tensor | None] = {}
    for target in targets:
        observed = target.observed
        if observed.ndim >= 2 and row_sample > 0 and row_sample < int(observed.shape[0]):
            rows = torch.linspace(0, int(observed.shape[0]) - 1, steps=int(row_sample), device=device)
            rows_by_name[target.name] = rows.round().to(torch.long).unique(sorted=True)
        else:
            rows_by_name[target.name] = None
    return rows_by_name


def aggregate_target_loss_and_metrics(
    predicted: torch.Tensor,
    target: TensorTarget,
    *,
    selected_rows: torch.Tensor | None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    source_elements = int(target.observed.numel())
    predicted_sel = predicted
    observed_sel = target.observed
    if selected_rows is not None:
        observed_sel = observed_sel.index_select(0, selected_rows)
    center = observed_sel
    if target.consistency == "mse":
        err = predicted_sel - observed_sel
        loss = err.square().mean()
        outside = err.abs()
        outside_bin_l2 = float(outside.norm().detach().cpu())
        within_count = 0
        bin_satisfaction = 0.0
        mean_bin_violation = float(outside.mean().detach().cpu())
    else:
        if target.lower is None or target.upper is None:
            raise ValueError(f"{target.consistency} requires aggregate bin bounds")
        lower_sel = target.lower
        upper_sel = target.upper
        if selected_rows is not None:
            lower_sel = lower_sel.index_select(0, selected_rows)
            upper_sel = upper_sel.index_select(0, selected_rows)
        center = 0.5 * (lower_sel + upper_sel)
        outside = torch.relu(lower_sel - predicted_sel) + torch.relu(predicted_sel - upper_sel)
        loss = outside.square().mean()
        with torch.no_grad():
            within = outside <= 1e-8
            within_count = int(within.sum().detach().cpu())
            bin_satisfaction = float(within.float().mean().detach().cpu())
            mean_bin_violation = float(outside.mean().detach().cpu())
            outside_bin_l2 = float(outside.norm().detach().cpu())
        err = predicted_sel - center
    with torch.no_grad():
        obs_norm = torch.linalg.vector_norm(center).clamp_min(1e-12)
        err_norm = torch.linalg.vector_norm(err)
        pred_norm = torch.linalg.vector_norm(predicted_sel).clamp_min(1e-12)
        cosine = torch.sum(predicted_sel * center) / (pred_norm * obs_norm)
    return loss, {
        "agg_consistency": target.consistency,
        "agg_mse_to_center": float((predicted_sel - center).square().mean().detach().cpu()),
        "agg_rel_l2_to_center": float((err_norm / obs_norm).detach().cpu()),
        "agg_cosine_to_center": float(cosine.detach().cpu()),
        "agg_mean_abs_error_to_center": float(err.abs().mean().detach().cpu()),
        "agg_max_abs_error_to_center": float(err.abs().max().detach().cpu()),
        "bin_satisfaction": bin_satisfaction,
        "mean_bin_violation": mean_bin_violation,
        "outside_bin_l2": outside_bin_l2,
        "edge_code_ratio": target.edge_code_ratio,
        "within_bin_elements": float(within_count),
        "total_elements": int(predicted_sel.numel()),
        "source_elements": source_elements,
        "sampled_rows": int(selected_rows.numel()) if selected_rows is not None else int(target.observed.shape[0]) if target.observed.ndim >= 2 else None,
    }


def labels_for_slice(
    *,
    args: argparse.Namespace,
    y_true: torch.Tensor,
    label_logits: torch.nn.Parameter | None,
    sl: slice,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    if args.label_mode == "known":
        return y_true[sl], None
    if label_logits is None:
        raise ValueError("label_logits missing for unknown label mode")
    return None, F.softmax(label_logits[sl], dim=-1)


def client_delta_for_targets(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    theta0: OrderedDict[str, torch.Tensor],
    pixels: torch.Tensor,
    y_true: torch.Tensor,
    label_logits: torch.nn.Parameter | None,
    client: ClientBatch,
    local_steps: int,
    local_lr: float,
    create_graph: bool,
    update_names: Sequence[str] | None,
    selected_rows_by_name: Mapping[str, torch.Tensor | None],
) -> Dict[str, torch.Tensor]:
    sl = client.sample_slice
    labels, label_probs = labels_for_slice(args=args, y_true=y_true, label_logits=label_logits, sl=sl)
    theta_t = unroll_full_batch_sgd(
        model=model,
        theta0=theta0,
        x_norm=normalize_pixels(pixels[sl]),
        labels=labels,
        label_probs=label_probs,
        steps=local_steps,
        lr=local_lr,
        create_graph=create_graph,
        microbatch_size=args.inner_microbatch_size,
        update_names=update_names,
    )
    names = list(update_names) if update_names is not None else list(theta0.keys())
    deltas: Dict[str, torch.Tensor] = {}
    for name in names:
        delta = theta_t[name] - theta0[name]
        selected_rows = selected_rows_by_name.get(name)
        if selected_rows is not None:
            delta = delta.index_select(0, selected_rows)
        deltas[name] = delta
    return deltas


def aggregate_update_objective(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    theta0: OrderedDict[str, torch.Tensor],
    pixels: torch.Tensor,
    y_true: torch.Tensor,
    label_logits: torch.nn.Parameter | None,
    clients: Sequence[ClientBatch],
    targets: Sequence[TensorTarget],
    local_steps: int,
    local_lr: float,
    update_names: Sequence[str] | None,
    create_graph: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    aggregate: Dict[str, torch.Tensor] = {}
    selected_rows_by_name = selected_row_indices_for_targets(targets, args.q_row_sample, pixels.device)
    for client in clients:
        deltas = client_delta_for_targets(
            args=args,
            model=model,
            theta0=theta0,
            pixels=pixels,
            y_true=y_true,
            label_logits=label_logits,
            client=client,
            local_steps=local_steps,
            local_lr=local_lr,
            create_graph=create_graph,
            update_names=update_names,
            selected_rows_by_name=selected_rows_by_name,
        )
        for name, delta in deltas.items():
            weighted = float(client.weight) * delta
            aggregate[name] = weighted if name not in aggregate else aggregate[name] + weighted

    loss_sum = torch.zeros((), device=pixels.device)
    total_elems = 0
    metric_accum = {
        "agg_mse_to_center": 0.0,
        "agg_rel_l2_to_center": 0.0,
        "agg_cosine_to_center": 0.0,
        "agg_mean_abs_error_to_center": 0.0,
        "agg_max_abs_error_to_center": 0.0,
        "bin_satisfaction": 0.0,
        "mean_bin_violation": 0.0,
        "outside_bin_l2_sq": 0.0,
        "edge_code_ratio": 0.0,
        "within_bin_elements": 0.0,
        "source_elements": 0.0,
        "sampled_rows": 0.0,
    }
    consistency = targets[0].consistency if targets else args.agg_consistency
    for target in targets:
        target_loss, metrics = aggregate_target_loss_and_metrics(
            aggregate[target.name],
            target,
            selected_rows=selected_rows_by_name.get(target.name),
        )
        elems = int(metrics["total_elements"])
        loss_sum = loss_sum + target_loss * elems
        total_elems += elems
        for key in (
            "agg_mse_to_center",
            "agg_rel_l2_to_center",
            "agg_cosine_to_center",
            "agg_mean_abs_error_to_center",
            "agg_max_abs_error_to_center",
            "bin_satisfaction",
            "mean_bin_violation",
            "edge_code_ratio",
        ):
            value = metrics[key]
            if value is not None:
                metric_accum[key] += float(value) * elems
        metric_accum["outside_bin_l2_sq"] += float(metrics["outside_bin_l2"]) ** 2
        metric_accum["within_bin_elements"] += float(metrics["within_bin_elements"])
        metric_accum["source_elements"] += float(metrics["source_elements"])
        if metrics["sampled_rows"] is not None:
            metric_accum["sampled_rows"] = float(metrics["sampled_rows"])
    loss = loss_sum / max(1, total_elems)
    return loss, {
        "loss_q": float(loss.detach().cpu()),
        "agg_consistency": consistency,
        "agg_mse_to_center": metric_accum["agg_mse_to_center"] / max(1, total_elems),
        "agg_rel_l2_to_center": metric_accum["agg_rel_l2_to_center"] / max(1, total_elems),
        "agg_cosine_to_center": metric_accum["agg_cosine_to_center"] / max(1, total_elems),
        "agg_mean_abs_error_to_center": metric_accum["agg_mean_abs_error_to_center"] / max(1, total_elems),
        "agg_max_abs_error_to_center": metric_accum["agg_max_abs_error_to_center"] / max(1, total_elems),
        "bin_satisfaction": metric_accum["bin_satisfaction"] / max(1, total_elems),
        "mean_bin_violation": metric_accum["mean_bin_violation"] / max(1, total_elems),
        "outside_bin_l2": float(metric_accum["outside_bin_l2_sq"] ** 0.5),
        "edge_code_ratio": metric_accum["edge_code_ratio"] / max(1, total_elems),
        "within_bin_elements": metric_accum["within_bin_elements"],
        "total_q_elements": float(total_elems),
        "source_elements": metric_accum["source_elements"],
        "sampled_rows": metric_accum["sampled_rows"],
    }


def q_gradient_probe(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    theta0: OrderedDict[str, torch.Tensor],
    pixel_logits: torch.nn.Parameter,
    y_true: torch.Tensor,
    label_logits: torch.nn.Parameter | None,
    clients: Sequence[ClientBatch],
    targets: Sequence[TensorTarget],
    local_steps: int,
    local_lr: float,
    update_names: Sequence[str] | None,
) -> Dict[str, Any]:
    pixels = torch.sigmoid(pixel_logits)
    loss_q, metrics = aggregate_update_objective(
        args=args,
        model=model,
        theta0=theta0,
        pixels=pixels,
        y_true=y_true,
        label_logits=label_logits,
        clients=clients,
        targets=targets,
        local_steps=local_steps,
        local_lr=local_lr,
        update_names=update_names,
        create_graph=True,
    )
    grad = torch.autograd.grad(loss_q, pixel_logits, allow_unused=True, retain_graph=False)[0]
    if grad is None:
        grad_norm = 0.0
        nonzero = False
    else:
        grad_norm = float(grad.detach().norm().cpu())
        nonzero = bool(grad_norm > args.grad_nonzero_eps)
    status = "ok"
    if float(loss_q.detach().cpu()) <= 1e-20:
        status = "zero_gradient_possible_because_aggregate_loss_is_zero"
    elif not nonzero:
        status = "failed_nonzero_aggregate_loss_but_zero_pixel_gradient"
    return {
        "loss_q": float(loss_q.detach().cpu()),
        "pixel_grad_norm": grad_norm,
        "nonzero_pixel_gradient": nonzero,
        "status": status,
        **metrics,
    }


def to_uint8_chw(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return (np.transpose(image, (1, 2, 0)) * 255.0).round().astype(np.uint8)


def save_image_chw(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8_chw(image)).save(path)


def make_pair_grid(originals: np.ndarray, reconstructions: np.ndarray, path: Path, *, max_items: int) -> None:
    n = min(max_items, int(originals.shape[0]), int(reconstructions.shape[0]))
    if n <= 0:
        return
    tile_h = int(originals.shape[-2])
    tile_w = int(originals.shape[-1])
    pair_cols = min(4, n)
    rows = math.ceil(n / pair_cols)
    label_h = 18
    canvas = Image.new("RGB", (pair_cols * tile_w * 2, rows * (tile_h + label_h)), "white")
    for idx in range(n):
        row, col = divmod(idx, pair_cols)
        x0 = col * tile_w * 2
        y0 = row * (tile_h + label_h) + label_h
        canvas.paste(Image.fromarray(to_uint8_chw(originals[idx])), (x0, y0))
        canvas.paste(Image.fromarray(to_uint8_chw(reconstructions[idx])), (x0 + tile_w, y0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


class ImageMetricComputer:
    def __init__(self, device: torch.device, *, skip_ssim: bool, skip_lpips: bool) -> None:
        self.ssim_fn = None
        self.ssim_error: str | None = None
        if not skip_ssim:
            try:
                from torchmetrics.functional import structural_similarity_index_measure

                self.ssim_fn = structural_similarity_index_measure
            except Exception as exc:
                self.ssim_error = str(exc)
        self.lpips_model = None
        self.lpips_error: str | None = None
        if not skip_lpips:
            try:
                from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

                self.lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
                self.lpips_model.eval()
            except Exception as exc:
                self.lpips_error = str(exc)

    def compute(self, originals: torch.Tensor, reconstructions: torch.Tensor) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        rows: List[Dict[str, Any]] = []
        mse_values: List[float] = []
        psnr_values: List[float] = []
        ssim_values: List[float] = []
        lpips_values: List[float] = []
        with torch.no_grad():
            for idx, (orig, rec) in enumerate(zip(originals, reconstructions)):
                orig_b = orig.unsqueeze(0)
                rec_b = rec.unsqueeze(0)
                mse = torch.mean((orig_b - rec_b) ** 2).clamp_min(1e-12)
                psnr = 10.0 * torch.log10(1.0 / mse)
                row: Dict[str, Any] = {
                    "sample": idx,
                    "mse": float(mse.detach().cpu()),
                    "psnr": float(psnr.detach().cpu()),
                    "ssim": None,
                    "lpips": None,
                }
                mse_values.append(float(row["mse"]))
                psnr_values.append(float(row["psnr"]))
                if self.ssim_fn is not None:
                    value = float(self.ssim_fn(rec_b, orig_b).detach().cpu().item())
                    row["ssim"] = value
                    ssim_values.append(value)
                if self.lpips_model is not None:
                    value = float(self.lpips_model(rec_b * 2.0 - 1.0, orig_b * 2.0 - 1.0).detach().cpu().item())
                    row["lpips"] = value
                    lpips_values.append(value)
                rows.append(row)
        return {
            "mse_mean": float(np.mean(mse_values)),
            "mse_min": float(np.min(mse_values)),
            "mse_max": float(np.max(mse_values)),
            "psnr_mean": float(np.mean(psnr_values)),
            "psnr_min": float(np.min(psnr_values)),
            "psnr_max": float(np.max(psnr_values)),
            "ssim_mean": float(np.mean(ssim_values)) if ssim_values else None,
            "ssim_min": float(np.min(ssim_values)) if ssim_values else None,
            "ssim_max": float(np.max(ssim_values)) if ssim_values else None,
            "ssim_error": self.ssim_error,
            "lpips_mean": float(np.mean(lpips_values)) if lpips_values else None,
            "lpips_min": float(np.min(lpips_values)) if lpips_values else None,
            "lpips_max": float(np.max(lpips_values)) if lpips_values else None,
            "lpips_error": self.lpips_error,
        }, rows


def default_output_dir(args: argparse.Namespace, run_dir: Path, sample_count: int) -> Path:
    if args.output_dir is not None:
        return resolve_path(args.output_dir)
    prior_tag = ""
    if args.mi_weight > 0.0:
        prior_tag += f"_mi{args.mi_weight:g}"
    if args.ne_weight > 0.0:
        prior_tag += f"_ne{args.ne_weight:g}"
    tag = (
        f"direct_agg_update_cpa_{args.agg_consistency}_{args.loss_scope}_{args.label_mode}_"
        f"n{sample_count}_it{args.n_iter}_random{prior_tag}"
    )
    return run_dir / "quant_aware_cpa_grad_loss" / tag


def optimize(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    theta0: OrderedDict[str, torch.Tensor],
    x_true: torch.Tensor,
    y_true: torch.Tensor,
    clients: Sequence[ClientBatch],
    targets: Sequence[TensorTarget],
    local_steps: int,
    local_lr: float,
    update_names: Sequence[str] | None,
    output_dir: Path,
) -> Tuple[torch.Tensor, torch.nn.Parameter | None, List[Dict[str, Any]], Dict[str, Any]]:
    pixel_logits = torch.nn.Parameter(torch.logit(torch.rand_like(x_true).clamp(1e-4, 1 - 1e-4)))
    trainable: List[torch.nn.Parameter] = [pixel_logits]
    label_logits = None
    if args.label_mode == "unknown":
        label_logits = torch.nn.Parameter(torch.zeros(x_true.shape[0], int(nclasses_dict["imagenet"]), device=x_true.device))
        trainable.append(label_logits)
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    log_path = output_dir / "losses.jsonl"
    if args.fresh_start and log_path.exists():
        log_path.unlink()

    initial_probe = None
    if args.q_grad_check:
        initial_probe = q_gradient_probe(
            args=args,
            model=model,
            theta0=theta0,
            pixel_logits=pixel_logits,
            y_true=y_true,
            label_logits=label_logits,
            clients=clients,
            targets=targets,
            local_steps=local_steps,
            local_lr=local_lr,
            update_names=update_names,
        )

    iter_rows: List[Dict[str, Any]] = []
    start = time.time()
    last = start
    for iteration in range(args.n_iter):
        optimizer.zero_grad(set_to_none=True)
        pixels = torch.sigmoid(pixel_logits)
        loss_q, q_metrics = aggregate_update_objective(
            args=args,
            model=model,
            theta0=theta0,
            pixels=pixels,
            y_true=y_true,
            label_logits=label_logits,
            clients=clients,
            targets=targets,
            local_steps=local_steps,
            local_lr=local_lr,
            update_names=update_names,
            create_graph=True,
        )
        loss_tv = total_variation(pixels) if args.tv_weight > 0 else torch.zeros((), device=x_true.device)
        loss_mi = (
            image_mi_proxy_loss(pixels, args.mi_downsample)
            if args.mi_weight > 0
            else torch.zeros((), device=x_true.device)
        )
        loss_ne = (
            non_gaussianity_prior_loss(pixels, args.ne_alpha)
            if args.ne_weight > 0
            else torch.zeros((), device=x_true.device)
        )
        loss = (
            args.q_weight * loss_q
            + args.tv_weight * loss_tv
            + args.mi_weight * loss_mi
            + args.ne_weight * loss_ne
        )
        loss.backward()
        optimizer.step()

        should_log = iteration == 0 or iteration == args.n_iter - 1 or ((iteration + 1) % args.n_log == 0)
        if should_log:
            now = time.time()
            row = {
                "iter": iteration,
                "loss": float(loss.detach().cpu()),
                "loss_q": q_metrics["loss_q"],
                "loss_tv": float(loss_tv.detach().cpu()),
                "loss_mi": float(loss_mi.detach().cpu()),
                "loss_ne": float(loss_ne.detach().cpu()),
                "q_weight": args.q_weight,
                "tv_weight": args.tv_weight,
                "mi_weight": args.mi_weight,
                "ne_weight": args.ne_weight,
                "agg_consistency": q_metrics["agg_consistency"],
                "bin_satisfaction": q_metrics["bin_satisfaction"],
                "mean_bin_violation": q_metrics["mean_bin_violation"],
                "outside_bin_l2": q_metrics["outside_bin_l2"],
                "edge_code_ratio": q_metrics["edge_code_ratio"],
                "within_bin_elements": q_metrics["within_bin_elements"],
                "agg_mse_to_center": q_metrics["agg_mse_to_center"],
                "agg_rel_l2_to_center": q_metrics["agg_rel_l2_to_center"],
                "agg_cosine_to_center": q_metrics["agg_cosine_to_center"],
                "agg_mean_abs_error_to_center": q_metrics["agg_mean_abs_error_to_center"],
                "agg_max_abs_error_to_center": q_metrics["agg_max_abs_error_to_center"],
                "total_q_elements": q_metrics["total_q_elements"],
                "source_elements": q_metrics["source_elements"],
                "sampled_rows": q_metrics["sampled_rows"],
                "seconds_since_last_log": now - last,
                "seconds_total": now - start,
            }
            last = now
            iter_rows.append(row)
            write_jsonl(log_path, row)
            write_json(output_dir / "progress.json", {"status": "running", "latest": row})
            print(
                "iter {iter} | loss {loss:.6g} | update {loss_q:.3g} | "
                "tv {loss_tv:.3g} | mi {loss_mi:.3g} | ne {loss_ne:.3g} | "
                "bin {bin_satisfaction:.4f} | out_l2 {outside_bin_l2:.3g} | edge {edge_code_ratio:.4f} | "
                "rel_l2 {agg_rel_l2_to_center:.4f} | "
                "cos {agg_cosine_to_center:.4f} | total {seconds_total:.1f}s".format(**row),
                flush=True,
            )

    with torch.no_grad():
        rec_pixels = torch.sigmoid(pixel_logits).detach()

    final_probe = None
    if args.q_grad_check:
        final_probe = q_gradient_probe(
            args=args,
            model=model,
            theta0=theta0,
            pixel_logits=pixel_logits,
            y_true=y_true,
            label_logits=label_logits,
            clients=clients,
            targets=targets,
            local_steps=local_steps,
            local_lr=local_lr,
            update_names=update_names,
        )
    diagnostics = {
        "initial_q_gradient_probe": initial_probe,
        "final_q_gradient_probe": final_probe,
    }
    return rec_pixels, label_logits, iter_rows, diagnostics


def main() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    client_qs_dir = run_dir / "client_qs"
    quant_meta_path = client_qs_dir / "final_round_client_quant_meta.json"
    context_path = client_qs_dir / "final_round_context.pkl"
    manifest_path = run_dir / "manifest.json"
    if not quant_meta_path.is_file() or not context_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing required int4 FedAvg artifact under {run_dir}")

    quant_meta = read_json(quant_meta_path)
    run_manifest = read_json(manifest_path)
    validate_metadata(run_manifest, quant_meta)
    context = read_pickle(context_path)

    device = choose_device(args.device)
    seed = int(args.seed if args.seed is not None else run_manifest.get("seed", 42))
    set_seed(seed)

    client_ids = parse_client_ids(args.client_ids, len(quant_meta["clients"]))
    all_client_ids = list(range(len(quant_meta["clients"])))
    if client_ids != all_client_ids:
        raise ValueError(
            "Aggregated-gradient observation requires all clients. "
            f"Use --client_ids all for this artifact; got {client_ids}."
        )
    names = selected_param_names(args, run_manifest, quant_meta)
    aggregate_pickle = resolve_path(args.aggregate_pickle) if args.aggregate_pickle is not None else default_aggregate_pickle(run_dir)
    targets, aggregate_info = load_aggregate_targets(
        args=args,
        aggregate_pickle=aggregate_pickle,
        run_manifest=run_manifest,
        quant_meta=quant_meta,
        client_qs_dir=client_qs_dir,
        names=names,
        device=device,
    )
    model, model_info = load_model(quant_meta, device)
    theta0 = _state_dict_params(model)
    clients, selected_indices = load_clients(
        quant_meta=quant_meta,
        context=context,
        client_ids=client_ids,
    )

    x_np = np.asarray(context["x"], dtype=np.float32)[np.asarray(selected_indices, dtype=np.int64)]
    y_np = np.asarray(context["y"], dtype=np.int64)[np.asarray(selected_indices, dtype=np.int64)]
    x_true = torch.as_tensor(x_np, device=device, dtype=torch.float32)
    y_true = torch.as_tensor(y_np, device=device, dtype=torch.long)
    output_dir = default_output_dir(args, run_dir, int(x_true.shape[0]))
    if args.fresh_start:
        safe_rmtree(output_dir, allowed_root=run_dir / "quant_aware_cpa_grad_loss")
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_local_steps = int(run_manifest["local_epochs"])
    local_steps = artifact_local_steps if args.local_steps_override is None else int(args.local_steps_override)
    if local_steps <= 0:
        raise ValueError("--local_steps_override must be positive")
    if local_steps > artifact_local_steps:
        raise ValueError(
            f"--local_steps_override={local_steps} exceeds artifact local_epochs={artifact_local_steps}"
        )
    local_lr = float(run_manifest["lr"])
    manifest = {
        "method": "direct_aggregated_update_to_xy_quant_aware_cpa",
        "formula": (
            "q_weight * L_update(FedAvg_k(theta_T_k(x,y)-theta_0), observed_aggregate_constraint) "
            "+ tv_weight * TV(x) + mi_weight * MI_proxy(x) + ne_weight * NE_prior(x)"
        ),
        "agg_consistency": args.agg_consistency,
        "run_dir": str(run_dir),
        "client_qs_dir": str(client_qs_dir),
        "aggregate_observation": aggregate_info,
        "client_ids": client_ids,
        "selected_param_names": names,
        "loss_scope": args.loss_scope,
        "label_mode": args.label_mode,
        "init_pixels": args.init_pixels,
        "n_iter": args.n_iter,
        "n_log": args.n_log,
        "lr": args.lr,
        "q_weight": args.q_weight,
        "tv_weight": args.tv_weight,
        "mi_weight": args.mi_weight,
        "mi_downsample": args.mi_downsample,
        "ne_weight": args.ne_weight,
        "ne_alpha": args.ne_alpha,
        "local_steps_override": args.local_steps_override,
        "inner_microbatch_size": args.inner_microbatch_size,
        "unroll_param_scope": args.unroll_param_scope,
        "q_row_chunk": args.q_row_chunk,
        "q_row_sample": args.q_row_sample,
        "device": str(device),
        "seed": seed,
        "local_sgd": {
            "steps": local_steps,
            "artifact_steps": artifact_local_steps,
            "lr": local_lr,
            "full_batch": True,
            "truncated_unroll": local_steps != artifact_local_steps,
        },
        "model": model_info,
        "artifact_hashes": {
            "quant_meta_sha256": sha256_file(quant_meta_path),
            "context_sha256": sha256_file(context_path),
            "codes_sha256": quant_meta.get("codes_sha256"),
        },
        "priors": {
            "tv": "anisotropic total variation on reconstructed pixels",
            "mi": "pairwise squared centered-correlation penalty between reconstructed images",
            "ne": "negative logcosh non-Gaussianity proxy on reconstructed image gradients",
        },
        "note": "No recovered embedding is used; the server-side aggregated gradient/update layer directly constrains x.",
    }
    write_json(output_dir / "manifest.json", manifest)

    print(
        f"run_dir={run_dir} | clients={client_ids} | samples={x_true.shape[0]} | "
        f"params={names} | local_steps={local_steps} | output={output_dir}",
        flush=True,
    )
    update_names = None if args.unroll_param_scope == "full" else names
    rec_pixels, label_logits, iter_rows, diagnostics = optimize(
        args=args,
        model=model,
        theta0=theta0,
        x_true=x_true,
        y_true=y_true,
        clients=clients,
        targets=targets,
        local_steps=local_steps,
        local_lr=local_lr,
        update_names=update_names,
        output_dir=output_dir,
    )
    write_json(output_dir / "diagnostics.json", diagnostics)

    with torch.enable_grad():
        final_q_loss, final_q = aggregate_update_objective(
            args=args,
            model=model,
            theta0=theta0,
            pixels=rec_pixels,
            y_true=y_true,
            label_logits=label_logits,
            clients=clients,
            targets=targets,
            local_steps=local_steps,
            local_lr=local_lr,
            update_names=update_names,
            create_graph=False,
        )
    final_q["loss_q"] = float(final_q_loss.detach().cpu())
    final_prior = {
        "loss_tv": float(total_variation(rec_pixels).detach().cpu()),
        "loss_mi": float(image_mi_proxy_loss(rec_pixels, args.mi_downsample).detach().cpu()),
        "loss_ne": float(non_gaussianity_prior_loss(rec_pixels, args.ne_alpha).detach().cpu()),
        "q_weight": args.q_weight,
        "tv_weight": args.tv_weight,
        "mi_weight": args.mi_weight,
        "ne_weight": args.ne_weight,
    }
    orig_cpu = x_true.detach().cpu()
    rec_cpu = rec_pixels.detach().cpu()
    metric_computer = ImageMetricComputer(device, skip_ssim=args.skip_ssim, skip_lpips=args.skip_lpips)
    metric_summary, metric_rows = metric_computer.compute(orig_cpu.to(device), rec_cpu.to(device))
    write_csv(output_dir / "image_metrics_by_sample.csv", metric_rows)

    orig_np = orig_cpu.numpy()
    rec_np = rec_cpu.numpy()
    make_pair_grid(orig_np, rec_np, output_dir / "png" / "compare.png", max_items=args.max_png_items)
    rec_dir = output_dir / "png" / "reconstructed"
    rec_index = []
    for idx, image in enumerate(rec_np):
        path = rec_dir / f"rec_{idx:03d}_label{int(y_np[idx])}.png"
        save_image_chw(path, image)
        rec_index.append({"sample": idx, "label": int(y_np[idx]), "path": str(path)})
    write_csv(rec_dir / "index.csv", rec_index)
    write_pickle(
        output_dir / "reconstructions.pkl",
        {
            "rec_pixels": rec_np,
            "x_true": orig_np,
            "y_true": y_np,
            "selected_indices": list(map(int, selected_indices)),
            "client_ids": client_ids,
        },
    )
    metrics = {
        **metric_summary,
        "final_q": final_q,
        "final_prior": final_prior,
        "last_logged": iter_rows[-1] if iter_rows else None,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "progress.json", {"status": "complete", "latest": iter_rows[-1] if iter_rows else None})
    print(f"complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
