#!/usr/bin/env python3
"""QICA-conditioned quantization-aware FIA from final-round client int4 q/s.

This runner starts from recovered QICA embeddings as the FIA target.  During FIA
it also reconstructs the observed attack-layer update through a per-client
mixing matrix and penalizes values that leave the stored client int4 bins.  The
main gradient path is ``L_fi + L_q -> z(x) -> x``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_SRC = REPO_ROOT / "baseline" / "src"
for path in (SCRIPT_DIR, BASELINE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from datasets import ds_mean, ds_std  # noqa: E402
from models import get_model  # noqa: E402
from assignment import linear_sum_assignment  # noqa: E402
from qica_core import bin_distance_squared, finite_bin_bounds, requantize  # noqa: E402


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
    codes: torch.Tensor
    scale: float
    max_abs: float
    numel: int


@dataclass
class ClientBatch:
    client_id: int
    weight: float
    sample_slice: slice
    global_indices: List[int]
    targets: List[TensorTarget]


@dataclass
class OptimizationResult:
    tag: str
    pixel_logits: torch.nn.Parameter
    rec_pixels: torch.Tensor
    final_z_hat: torch.Tensor
    iter_rows: List[Dict[str, Any]]
    final_fi_loss: float
    final_q_loss: float
    final_mix_l2: float
    final_q_metrics: Dict[str, float]


class CompatUnpickler(pickle.Unpickler):
    """Read NumPy 2 pickles from NumPy 1 environments when possible."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "QICA-conditioned quantization-aware FIA: match recovered embeddings "
            "while reconstructed attack-layer updates satisfy stored int4 q/s bins."
        )
    )
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--client_ids", type=str, default="all", help="'all' or comma-separated client ids")
    parser.add_argument("--loss_scope", type=str, default="attack_layer", choices=["attack_layer", "all", "names"])
    parser.add_argument("--loss_param_names", type=str, default="", help="Comma-separated names when --loss_scope names")
    parser.add_argument(
        "--label_mode",
        type=str,
        default="known",
        choices=["known", "unknown"],
        help="Kept for output compatibility; the mixing objective does not use labels.",
    )
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--n_log", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--opt", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--sch", type=str, default="cosine", choices=["cosine", "none"])
    parser.add_argument(
        "--fi_weight",
        type=float,
        default=1.0,
        help="Weight for FIA feature loss: 1 - cosine(z(x), recovered QICA embedding).",
    )
    parser.add_argument("--q_weight", type=float, default=1.0)
    parser.add_argument("--tv_weight", type=float, default=1e-4)
    parser.add_argument("--mix_l2_weight", type=float, default=0.0)
    parser.add_argument("--mixing_ridge", type=float, default=1e-3)
    parser.add_argument("--mixing_init", type=str, default="least_squares", choices=["least_squares", "random", "zero"])
    parser.add_argument("--train_mixing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--embedding_path", type=Path, required=True, help="Recovered QICA final_embeddings.npy target.")
    parser.add_argument(
        "--embedding_alignment",
        type=str,
        default="none",
        choices=["none", "oracle_context"],
        help="Use oracle_context only for diagnostics/upper-bound runs; it aligns QICA rows to context['z'].",
    )
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--q_grad_check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail_on_zero_q_grad", action="store_true")
    parser.add_argument("--grad_nonzero_eps", type=float, default=1e-12)
    parser.add_argument(
        "--compare_fia_only",
        action="store_true",
        help="Also run a same-init FIA-only baseline with q_weight=0 and compare against frozen-A q-aware FIA.",
    )
    parser.add_argument("--init_pixels", type=Path, default=None, help=".npy or .pkl containing pixel reconstructions")
    parser.add_argument("--seed", type=int, default=None, help="Override manifest seed for initialization")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--fresh_start", action="store_true")
    parser.add_argument("--max_png_items", type=int, default=24)
    parser.add_argument("--skip_eval_png", action="store_true")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except ModuleNotFoundError:
            handle.seek(0)
            return CompatUnpickler(handle).load()


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)


def write_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def write_pickle(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_output_dir(args: argparse.Namespace, run_dir: Path, n_samples: int) -> Path:
    if args.output_dir is not None:
        return resolve_path(args.output_dir)
    scope = args.loss_scope
    if scope == "names":
        scope = "names_" + "_".join(name.replace(".", "-") for name in parse_name_list(args.loss_param_names))
    alignment_suffix = "" if args.embedding_alignment == "none" else f"_{args.embedding_alignment}"
    comparison_suffix = "_cmp_fia_only" if args.compare_fia_only else ""
    return (
        run_dir
        / "qaware_client_qica"
        / f"qica_embedding_qaware_fia_{scope}{alignment_suffix}{comparison_suffix}_n{n_samples}_it{args.n_iter}"
    )


def parse_client_ids(value: str, n_clients: int) -> List[int]:
    if value.strip().lower() == "all":
        return list(range(n_clients))
    ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not ids:
        raise ValueError("--client_ids selected no clients")
    bad = [client_id for client_id in ids if client_id < 0 or client_id >= n_clients]
    if bad:
        raise ValueError(f"--client_ids outside [0,{n_clients - 1}]: {bad}")
    return ids


def parse_name_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def selected_param_names(args: argparse.Namespace, quant_meta: Mapping[str, Any]) -> List[str]:
    stored_names = {
        row["name"]
        for client in quant_meta["clients"]
        for row in client.get("tensors", [])
    }
    if args.loss_scope == "attack_layer":
        names = [quant_meta["attack_layer"]["name"]]
    elif args.loss_scope == "all":
        names = list(quant_meta["param_names"])
    else:
        names = parse_name_list(args.loss_param_names)
        if not names:
            raise ValueError("--loss_scope names requires --loss_param_names")

    missing = [name for name in names if name not in stored_names]
    if missing:
        raise ValueError(
            "Selected loss tensors are not stored in final_round_client_codes.npz: "
            + ", ".join(missing)
        )
    return names


def validate_model_supported(model: torch.nn.Module) -> None:
    active_dropout = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.Dropout) and float(module.p) > 0.0
    ]
    batchnorm = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    if active_dropout:
        raise ValueError("End-to-end FIA requires deterministic feature extraction; active dropout was found")
    if batchnorm:
        raise ValueError("End-to-end FIA currently rejects BatchNorm/stateful normalization layers")


def load_model(quant_meta: Mapping[str, Any], device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    state_path = resolve_path(Path(str(quant_meta["source_model_state_file"])))
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    model = get_model(model_name="vgg16", ds="imagenet", h_dim=None, dataparallel=False)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=torch.float32)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    validate_model_supported(model)
    return model, {
        "checkpoint_path": str(state_path),
        "checkpoint_sha256": sha256_file(state_path),
    }


def tensor_meta_by_name(client: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {row["name"]: row for row in client.get("tensors", [])}


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
        slices[client_id] = slice(len(selected), len(selected) + (end - start))
        selected.extend(range(start, end))
    return slices, selected


def load_clients(
    *,
    client_qs_dir: Path,
    quant_meta: Mapping[str, Any],
    context: Mapping[str, Any],
    client_ids: Sequence[int],
    names: Sequence[str],
    device: torch.device,
) -> Tuple[List[ClientBatch], List[int]]:
    codes_path = client_qs_dir / "final_round_client_codes.npz"
    if not codes_path.is_file():
        raise FileNotFoundError(codes_path)
    sample_slices, selected_global_indices = client_sample_slices(context, client_ids)
    clients: List[ClientBatch] = []
    with np.load(codes_path) as npz:
        for client_id in client_ids:
            client_meta = quant_meta["clients"][client_id]
            by_name = tensor_meta_by_name(client_meta)
            targets = []
            for name in names:
                meta = by_name[name]
                codes_np = np.asarray(npz[meta["key"]], dtype=np.int8)
                codes = torch.as_tensor(codes_np, device=device, dtype=torch.int8)
                targets.append(
                    TensorTarget(
                        name=name,
                        codes=codes,
                        scale=float(meta["scale"]),
                        max_abs=float(meta["max_abs"]),
                        numel=int(codes.numel()),
                    )
                )
            clients.append(
                ClientBatch(
                    client_id=client_id,
                    weight=float(client_meta["aggregation_weight"]),
                    sample_slice=sample_slices[client_id],
                    global_indices=selected_global_indices[
                        sample_slices[client_id].start : sample_slices[client_id].stop
                    ],
                    targets=targets,
                )
            )
    weight_sum = sum(client.weight for client in clients)
    if weight_sum <= 0:
        raise ValueError("selected clients have non-positive total aggregation weight")
    for client in clients:
        client.weight = client.weight / weight_sum
    return clients, selected_global_indices


def infer_image_shape(context: Mapping[str, Any], selected_indices: Sequence[int]) -> Tuple[int, int, int, int]:
    x = np.asarray(context["x"])
    sample = x[np.asarray(selected_indices, dtype=np.int64)]
    if sample.ndim != 4:
        raise ValueError(f"context x must be NCHW, got shape {sample.shape}")
    return tuple(int(value) for value in sample.shape)


def normalize_pixels(pixels: torch.Tensor, ds: str = "imagenet") -> torch.Tensor:
    mean = torch.tensor(ds_mean[ds], device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    std = torch.tensor(ds_std[ds], device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    return (pixels - mean) / std


def unnormalize_images(x_norm: torch.Tensor, ds: str = "imagenet") -> torch.Tensor:
    mean = torch.tensor(ds_mean[ds], device=x_norm.device, dtype=x_norm.dtype).view(1, 3, 1, 1)
    std = torch.tensor(ds_std[ds], device=x_norm.device, dtype=x_norm.dtype).view(1, 3, 1, 1)
    return torch.clamp(x_norm * std + mean, 0.0, 1.0)


def total_variation(pixels: torch.Tensor) -> torch.Tensor:
    dx = torch.mean(torch.abs(pixels[:, :, :, :-1] - pixels[:, :, :, 1:]))
    dy = torch.mean(torch.abs(pixels[:, :, :-1, :] - pixels[:, :, 1:, :]))
    return dx + dy


def make_optimizer(params: Sequence[torch.nn.Parameter], args: argparse.Namespace):
    if args.opt == "adam":
        return torch.optim.Adam(params, lr=args.lr)
    if args.opt == "sgd":
        return torch.optim.SGD(params, lr=args.lr)
    raise ValueError(f"unknown optimizer {args.opt}")


def make_scheduler(optimizer, args: argparse.Namespace):
    if args.sch == "none":
        return None
    if args.sch == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_iter, last_epoch=-1)
    raise ValueError(f"unknown scheduler {args.sch}")


def matrix_bin_loss(update_hat: torch.Tensor, target: TensorTarget) -> Tuple[torch.Tensor, Dict[str, float]]:
    dist = bin_distance_squared(update_hat, target.codes, target.scale, target.max_abs)
    loss = dist.mean()
    with torch.no_grad():
        lower, upper = finite_bin_bounds(target.codes, target.scale, target.max_abs)
        outside = torch.relu(lower - update_hat) + torch.relu(update_hat - upper)
        q = requantize(update_hat.detach(), target.scale)
        elems = int(update_hat.numel())
        return loss, {
            "bin_satisfaction": float((outside <= 1e-8).float().mean().detach().cpu()),
            "requantization_match": float((q == target.codes).float().mean().detach().cpu()),
            "mean_bin_violation": float(outside.mean().detach().cpu()),
            "matched_elements": int((q == target.codes).sum().detach().cpu()),
            "total_elements": elems,
        }


def q_objective_from_features(
    z_hat: torch.Tensor,
    clients: Sequence[ClientBatch],
    mixing_params: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    device = z_hat.device
    loss_q = torch.zeros((), device=device)
    loss_mix = torch.zeros((), device=device)
    metric_accum = {
        "bin_satisfaction_weighted": 0.0,
        "requantization_match_weighted": 0.0,
        "mean_bin_violation_weighted": 0.0,
        "total_elements": 0,
    }
    for mix, client in zip(mixing_params, clients):
        z_client = z_hat[client.sample_slice]
        update_hat = mix.matmul(z_client)
        client_loss, client_metrics = matrix_bin_loss(update_hat, client.targets[0])
        loss_q = loss_q + float(client.weight) * client_loss
        loss_mix = loss_mix + float(client.weight) * mix.square().mean()
        metric_accum["bin_satisfaction_weighted"] += float(client.weight) * client_metrics["bin_satisfaction"]
        metric_accum["requantization_match_weighted"] += float(client.weight) * client_metrics["requantization_match"]
        metric_accum["mean_bin_violation_weighted"] += float(client.weight) * client_metrics["mean_bin_violation"]
        metric_accum["total_elements"] += int(client_metrics["total_elements"])
    return loss_q, loss_mix, {
        "bin_satisfaction": metric_accum["bin_satisfaction_weighted"],
        "requantization_match": metric_accum["requantization_match_weighted"],
        "mean_bin_violation": metric_accum["mean_bin_violation_weighted"],
        "total_q_elements": float(metric_accum["total_elements"]),
    }


def target_dequantized(target: TensorTarget) -> torch.Tensor:
    return target.codes.to(dtype=torch.float32) * float(target.scale)


def initialize_mixing_matrix(
    *,
    z_client: torch.Tensor,
    target: TensorTarget,
    init: str,
    ridge: float,
    seed: int,
    client_id: int,
) -> torch.Tensor:
    h_dim = int(target.codes.shape[0])
    n_client = int(z_client.shape[0])
    device = z_client.device
    if init == "zero":
        return torch.zeros((h_dim, n_client), device=device, dtype=torch.float32)
    if init == "random":
        generator = torch.Generator(device=device).manual_seed(seed + 7919 * (client_id + 1))
        return torch.randn((h_dim, n_client), generator=generator, device=device, dtype=torch.float32) * 1e-4
    if init != "least_squares":
        raise ValueError(f"unknown mixing init: {init}")

    z = z_client.detach().to(dtype=torch.float32)
    x_target = target_dequantized(target).to(device=device, dtype=torch.float32)
    gram = z.matmul(z.T)
    eye = torch.eye(n_client, device=device, dtype=torch.float32)
    ridge_value = float(ridge) * float(gram.diagonal().mean().detach().abs().clamp_min(1.0).cpu())
    inv = torch.linalg.pinv(gram + ridge_value * eye)
    return x_target.matmul(z.T).matmul(inv).detach()


def initialize_mixing_parameters(
    *,
    clients: Sequence[ClientBatch],
    z_source: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> torch.nn.ParameterList:
    params = torch.nn.ParameterList()
    for client in clients:
        if len(client.targets) != 1:
            raise ValueError("mixing FIA currently expects exactly one attack-layer target per client")
        z_client = z_source[client.sample_slice]
        init = initialize_mixing_matrix(
            z_client=z_client,
            target=client.targets[0],
            init=args.mixing_init,
            ridge=args.mixing_ridge,
            seed=seed,
            client_id=client.client_id,
        )
        parameter = torch.nn.Parameter(init, requires_grad=bool(args.train_mixing))
        params.append(parameter)
    return params


def load_embedding_array(path: Path) -> np.ndarray:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32, copy=False)


def load_embedding_target_none(
    embeddings: np.ndarray,
    selected_indices: Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if len(embeddings) >= max(selected_indices) + 1:
        selected = embeddings[np.asarray(selected_indices, dtype=np.int64)]
        selected_embedding_rows = list(map(int, selected_indices))
        row_policy = "embedding_rows_assumed_to_match_context_attack_order"
    elif len(embeddings) == len(selected_indices):
        selected = embeddings
        selected_embedding_rows = list(range(len(selected_indices)))
        row_policy = "embedding_rows_assumed_to_match_selected_client_order"
    else:
        raise ValueError(
            f"embedding target has {len(embeddings)} rows, cannot select {len(selected_indices)} samples"
        )
    return torch.as_tensor(selected, device=device, dtype=torch.float32), {
        "policy": "none",
        "oracle_context_used": False,
        "row_policy": row_policy,
        "selected_embedding_rows": selected_embedding_rows,
    }


def unit_rows_np(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def align_embedding_rows_to_reference(
    recovered: np.ndarray,
    reference: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    recovered_f = np.asarray(recovered, dtype=np.float32)
    reference_f = np.asarray(reference, dtype=np.float32)
    if recovered_f.ndim != 2 or reference_f.ndim != 2:
        raise ValueError(f"embedding/reference must be 2D, got {recovered_f.shape} and {reference_f.shape}")
    if recovered_f.shape[1] != reference_f.shape[1]:
        raise ValueError(f"embedding dim mismatch: recovered={recovered_f.shape}, reference={reference_f.shape}")
    if recovered_f.shape[0] < reference_f.shape[0]:
        raise ValueError(
            f"not enough recovered rows for oracle alignment: recovered={recovered_f.shape[0]} "
            f"reference={reference_f.shape[0]}"
        )

    similarity = unit_rows_np(reference_f) @ unit_rows_np(recovered_f).T
    rows, cols = linear_sum_assignment(-np.abs(similarity))
    order = np.argsort(rows)
    rows = rows[order]
    cols = cols[order]
    if not np.array_equal(rows, np.arange(reference_f.shape[0], dtype=np.int64)):
        raise RuntimeError("oracle alignment failed to match every reference row")

    aligned = np.empty_like(reference_f)
    scales = np.zeros(reference_f.shape[0], dtype=np.float64)
    signed_cosines = np.zeros(reference_f.shape[0], dtype=np.float64)
    abs_cosines = np.zeros(reference_f.shape[0], dtype=np.float64)
    for ref_row, rec_row in enumerate(cols):
        rec = recovered_f[rec_row].astype(np.float64, copy=False)
        ref = reference_f[ref_row].astype(np.float64, copy=False)
        scale = float(np.dot(rec, ref) / (np.dot(rec, rec) + eps))
        aligned[ref_row] = (rec * scale).astype(np.float32, copy=False)
        signed_cosines[ref_row] = float(similarity[ref_row, rec_row])
        abs_cosines[ref_row] = abs(signed_cosines[ref_row])
        scales[ref_row] = scale

    return aligned.astype(np.float32, copy=False), {
        "policy": "oracle_context_hungarian_abs_cosine_with_least_squares_sign_scale",
        "oracle_context_used": True,
        "context_row_to_embedding_row": cols.astype(int).tolist(),
        "least_squares_sign_scales": scales.astype(float).tolist(),
        "mean_abs_cosine": float(abs_cosines.mean()),
        "median_abs_cosine": float(np.median(abs_cosines)),
        "min_abs_cosine": float(abs_cosines.min()),
        "max_abs_cosine": float(abs_cosines.max()),
        "mean_signed_cosine": float(signed_cosines.mean()),
    }


def load_embedding_target(
    path: Path,
    selected_indices: Sequence[int],
    context: Mapping[str, Any],
    device: torch.device,
    alignment: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    embeddings = load_embedding_array(path)
    if alignment == "none":
        z_target, diagnostics = load_embedding_target_none(embeddings, selected_indices, device)
    elif alignment == "oracle_context":
        context_z = np.asarray(context["z"], dtype=np.float32)
        selected = np.asarray(selected_indices, dtype=np.int64)
        if len(embeddings) >= len(context_z):
            aligned, alignment_diag = align_embedding_rows_to_reference(embeddings, context_z)
            z_selected = aligned[selected]
            selected_embedding_rows = [
                int(alignment_diag["context_row_to_embedding_row"][int(row)])
                for row in selected.tolist()
            ]
            selected_scales = [
                float(alignment_diag["least_squares_sign_scales"][int(row)])
                for row in selected.tolist()
            ]
            row_policy = "full_context_oracle_alignment_then_select_attack_order_rows"
        elif len(embeddings) == len(selected_indices):
            reference = context_z[selected]
            z_selected, alignment_diag = align_embedding_rows_to_reference(embeddings, reference)
            selected_embedding_rows = list(map(int, alignment_diag["context_row_to_embedding_row"]))
            selected_scales = list(map(float, alignment_diag["least_squares_sign_scales"]))
            row_policy = "selected_subset_oracle_alignment_to_selected_attack_order_rows"
        else:
            raise ValueError(
                f"embedding target has {len(embeddings)} rows, cannot oracle-align "
                f"against context rows {len(context_z)} or selected rows {len(selected_indices)}"
            )
        z_target = torch.as_tensor(z_selected, device=device, dtype=torch.float32)
        diagnostics = {
            **alignment_diag,
            "row_policy": row_policy,
            "selected_attack_order_rows": selected.astype(int).tolist(),
            "selected_embedding_rows": selected_embedding_rows,
            "selected_least_squares_sign_scales": selected_scales,
            "note": "oracle_context uses context['z']; use it for diagnostics/upper-bound, not a blind attack claim.",
        }
    else:
        raise ValueError(f"unknown embedding alignment: {alignment}")

    diagnostics.update(
        {
            "embedding_path": str(resolve_path(path)),
            "embedding_shape": list(map(int, embeddings.shape)),
            "selected_count": int(len(selected_indices)),
        }
    )
    return z_target, diagnostics


def read_init_pixels(path: Path, selected_indices: Sequence[int]) -> np.ndarray:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32, copy=False)
    elif path.suffix.lower() in {".pkl", ".pickle"}:
        data = read_pickle(path)
        for key in ("rec_attack_order", "rec_eval_order", "reconstructions", "rec"):
            if key in data:
                arr = np.asarray(data[key], dtype=np.float32)
                break
        else:
            raise ValueError(f"{path} does not contain a recognized reconstruction key")
    else:
        raise ValueError("--init_pixels must be .npy, .pkl, or .pickle")

    if len(arr) >= max(selected_indices) + 1:
        return arr[np.asarray(selected_indices, dtype=np.int64)]
    if len(arr) == len(selected_indices):
        return arr
    raise ValueError(f"init pixels has {len(arr)} rows, expected {len(selected_indices)} or global-indexable rows")


def initialize_pixel_logits(
    *,
    args: argparse.Namespace,
    shape: Tuple[int, int, int, int],
    selected_indices: Sequence[int],
    device: torch.device,
    seed: int,
) -> torch.nn.Parameter:
    eps = 1e-4
    if args.init_pixels is not None:
        pixels_np = read_init_pixels(args.init_pixels, selected_indices)
        if tuple(pixels_np.shape) != tuple(shape):
            raise ValueError(f"init pixel shape {pixels_np.shape} does not match expected {shape}")
        pixels = torch.as_tensor(pixels_np, device=device, dtype=torch.float32).clamp(eps, 1.0 - eps)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
        pixels = torch.rand(shape, generator=generator, device=device, dtype=torch.float32).clamp(eps, 1.0 - eps)
    logits = torch.log(pixels / (1.0 - pixels))
    return torch.nn.Parameter(logits)


def client_order_diagnostics(
    clients: Sequence[ClientBatch],
    selected_indices: Sequence[int],
    mixing_params: Sequence[torch.Tensor],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    selected_rows = list(map(int, selected_indices))
    context_index = np.asarray(context.get("index")) if "index" in context else None
    rows: List[Dict[str, Any]] = []
    all_ok = True
    for client, mix in zip(clients, mixing_params):
        sl = client.sample_slice
        slice_rows = selected_rows[sl.start : sl.stop]
        expected_original = list(map(int, context["client_original_indices"][client.client_id]))
        if context_index is not None:
            dataset_original = [int(context_index[row]) for row in slice_rows]
        else:
            dataset_original = []
        shape_ok = (
            int(mix.shape[0]) == int(client.targets[0].codes.shape[0])
            and int(mix.shape[1]) == len(slice_rows)
        )
        slice_ok = list(map(int, client.global_indices)) == slice_rows
        original_ok = (not dataset_original) or dataset_original == expected_original
        ok = bool(shape_ok and slice_ok and original_ok)
        all_ok = all_ok and ok
        rows.append(
            {
                "client_id": int(client.client_id),
                "ok": ok,
                "sample_slice": [int(sl.start), int(sl.stop)],
                "num_samples": int(len(slice_rows)),
                "a_shape": list(map(int, mix.shape)),
                "target_shape": list(map(int, client.targets[0].codes.shape)),
                "shape_ok": bool(shape_ok),
                "slice_matches_selected_rows": bool(slice_ok),
                "context_original_index_order_ok": bool(original_ok),
                "a_column_meaning": "A_k[:, j] multiplies z_target/client_feature row j in this client slice.",
                "attack_order_rows": slice_rows,
                "dataset_original_indices": dataset_original,
                "expected_client_original_indices": expected_original,
            }
        )
    return {
        "all_ok": bool(all_ok),
        "selected_attack_order_rows": selected_rows,
        "clients": rows,
        "note": "The selected row index is the attack-order row in context['x'], context['z'], and final_embeddings.npy after any requested alignment.",
    }


def sample_axis_inverse_transform_diagnostics(
    clients: Sequence[ClientBatch],
    z_source: torch.Tensor,
    mixing_params: Sequence[torch.Tensor],
    *,
    seed: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    all_ok = True
    for offset, (client, mix) in enumerate(zip(clients, mixing_params)):
        z_client = z_source[client.sample_slice].detach()
        mix_detached = mix.detach()
        n_samples = int(z_client.shape[0])
        if n_samples == 0:
            continue
        generator = torch.Generator(device="cpu").manual_seed(seed + 97_531 * (offset + 1))
        perm_cpu = torch.randperm(n_samples, generator=generator)
        sign_cpu = torch.where(
            torch.rand(n_samples, generator=generator) >= 0.5,
            torch.ones(n_samples),
            -torch.ones(n_samples),
        )
        scale_cpu = torch.exp(torch.randn(n_samples, generator=generator) * 0.35).clamp_min(1e-4)
        perm = perm_cpu.to(device=z_client.device)
        factor = (sign_cpu * scale_cpu).to(device=z_client.device, dtype=z_client.dtype)

        transformed_z = z_client.index_select(0, perm) * factor.view(-1, 1)
        transformed_mix = mix_detached.index_select(1, perm) / factor.view(1, -1)
        base_update = mix_detached.matmul(z_client)
        transformed_update = transformed_mix.matmul(transformed_z)
        diff = transformed_update - base_update
        rel_l2 = float(
            torch.linalg.vector_norm(diff).detach().cpu()
            / torch.linalg.vector_norm(base_update).detach().cpu().clamp_min(1e-12)
        )
        max_abs = float(diff.abs().max().detach().cpu())
        ok = rel_l2 <= 1e-5 or max_abs <= 1e-7
        all_ok = all_ok and ok
        rows.append(
            {
                "client_id": int(client.client_id),
                "ok": bool(ok),
                "relative_l2_error": rel_l2,
                "max_abs_error": max_abs,
                "permutation_new_row_to_old_row": perm_cpu.numpy().astype(int).tolist(),
                "sign_scale_new_row": factor.detach().cpu().numpy().astype(float).tolist(),
                "inverse_rule": "if z'_j = d_j * z_perm[j], then A'_[:,j] = A[:,perm[j]] / d_j",
            }
        )
    return {
        "all_ok": bool(all_ok),
        "axis": "sample_rows",
        "note": (
            "A_k is [attack_layer_rows, client_samples], so its columns can invert "
            "sample-row permutation/sign/scale. Feature-column transforms of z are not inverted by A_k."
        ),
        "clients": rows,
    }


def q_gradient_probe(
    model: torch.nn.Module,
    pixel_logits: torch.Tensor,
    clients: Sequence[ClientBatch],
    mixing_params: Sequence[torch.Tensor],
    *,
    eps: float,
) -> Dict[str, Any]:
    pixels = torch.sigmoid(pixel_logits)
    x_norm = normalize_pixels(pixels, "imagenet")
    _logits, z_hat = model(x_norm, return_z=True)
    loss_q, _loss_mix, q_metrics = q_objective_from_features(z_hat, clients, mixing_params)
    grad_z, grad_pixels = torch.autograd.grad(
        loss_q,
        (z_hat, pixel_logits),
        allow_unused=True,
        retain_graph=False,
    )

    def grad_stats(grad: torch.Tensor | None) -> Dict[str, float]:
        if grad is None:
            return {"l2": 0.0, "max_abs": 0.0, "nonzero_fraction": 0.0}
        grad_detached = grad.detach()
        return {
            "l2": float(torch.linalg.vector_norm(grad_detached).cpu()),
            "max_abs": float(grad_detached.abs().max().cpu()),
            "nonzero_fraction": float((grad_detached.abs() > eps).float().mean().cpu()),
        }

    z_stats = grad_stats(grad_z)
    pixel_stats = grad_stats(grad_pixels)
    loss_q_value = float(loss_q.detach().cpu())
    nonzero = pixel_stats["max_abs"] > eps
    if nonzero:
        status = "ok_nonzero_pixel_gradient"
    elif loss_q_value <= eps:
        status = "zero_gradient_because_q_loss_is_zero_or_inside_bins"
    else:
        status = "failed_nonzero_q_loss_but_zero_pixel_gradient"
    return {
        "status": status,
        "loss_q": loss_q_value,
        "q_metrics": q_metrics,
        "z_grad": z_stats,
        "pixel_logit_grad": pixel_stats,
        "nonzero_pixel_gradient": bool(nonzero),
        "eps": float(eps),
    }


def build_diagnostics(
    *,
    clients: Sequence[ClientBatch],
    selected_indices: Sequence[int],
    mixing_params: Sequence[torch.Tensor],
    z_target: torch.Tensor,
    embedding_diagnostics: Dict[str, Any],
    context: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    with torch.no_grad():
        init_q_loss, init_mix_l2, init_q_metrics = q_objective_from_features(z_target, clients, mixing_params)
    return {
        "embedding_alignment": embedding_diagnostics,
        "client_sample_and_a_column_order": client_order_diagnostics(
            clients,
            selected_indices,
            mixing_params,
            context,
        ),
        "sample_axis_permutation_sign_scale_inverse": sample_axis_inverse_transform_diagnostics(
            clients,
            z_target,
            mixing_params,
            seed=seed,
        ),
        "mixing_init_on_z_qica": {
            "loss_q": float(init_q_loss.detach().cpu()),
            "loss_mix_l2": float(init_mix_l2.detach().cpu()),
            **init_q_metrics,
            "note": "This checks A_k @ z_qica before image optimization.",
        },
        "feature_axis_note": (
            "This factorization uses A_k @ z with A_k acting on sample rows. "
            "It can absorb sample-row permutation/sign/scale through A_k columns, "
            "but it cannot undo arbitrary feature-column permutation/sign/scale."
        ),
    }


def to_uint8_chw(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return (np.transpose(image, (1, 2, 0)) * 255.0).round().astype(np.uint8)


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
            except Exception as exc:  # pragma: no cover - depends on local torchmetrics install
                self.ssim_error = str(exc)

        self.lpips_model = None
        self.lpips_error: str | None = None
        if not skip_lpips:
            try:
                from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

                self.lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
                self.lpips_model.eval()
            except Exception as exc:  # pragma: no cover - depends on local torchmetrics/assets
                self.lpips_error = str(exc)

    def compute(
        self,
        originals: torch.Tensor,
        reconstructions: torch.Tensor,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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

        summary: Dict[str, Any] = {
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
        }
        return summary, rows


def metric_batch_from_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in ("mse", "psnr", "ssim", "lpips"):
        values = [row.get(key) for row in rows]
        result[key] = values if any(value is not None for value in values) else None
    return result


def optimize_pixels(
    *,
    tag: str,
    model: torch.nn.Module,
    initial_pixel_logits: torch.Tensor,
    z_target: torch.Tensor,
    clients: Sequence[ClientBatch],
    mixing_params: Sequence[torch.Tensor],
    args: argparse.Namespace,
    q_weight: float,
    output_dir: Path,
    log_filename: str,
) -> OptimizationResult:
    pixel_logits = torch.nn.Parameter(initial_pixel_logits.detach().clone())
    trainable: List[torch.nn.Parameter] = [pixel_logits]
    if args.train_mixing:
        trainable.extend([mix for mix in mixing_params if isinstance(mix, torch.nn.Parameter) and mix.requires_grad])

    optimizer = make_optimizer(trainable, args)
    scheduler = make_scheduler(optimizer, args)
    log_path = output_dir / log_filename
    if args.fresh_start and log_path.exists():
        log_path.unlink()

    iter_rows: List[Dict[str, Any]] = []
    start_time = time.time()
    last_log = start_time
    for iteration in range(args.n_iter):
        optimizer.zero_grad(set_to_none=True)

        pixels = torch.sigmoid(pixel_logits)
        x_norm = normalize_pixels(pixels, "imagenet")
        _logits, z_hat = model(x_norm, return_z=True)
        loss_fi = (1.0 - F.cosine_similarity(z_hat, z_target, dim=-1, eps=1e-10)).mean()
        loss_q, loss_mix, q_metrics = q_objective_from_features(z_hat, clients, mixing_params)
        loss_tv = total_variation(pixels) if args.tv_weight > 0 else torch.zeros((), device=pixels.device)

        loss = (
            float(q_weight) * loss_q
            + args.tv_weight * loss_tv
            + args.fi_weight * loss_fi
            + args.mix_l2_weight * loss_mix
        )
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        should_log = (
            iteration == 0
            or iteration == args.n_iter - 1
            or (args.n_log > 0 and iteration % args.n_log == 0)
        )
        if should_log:
            now = time.time()
            row = {
                "run_tag": tag,
                "iter": iteration,
                "loss": float(loss.detach().cpu()),
                "loss_q": float(loss_q.detach().cpu()),
                "loss_fi": float(loss_fi.detach().cpu()),
                "loss_tv": float(loss_tv.detach().cpu()),
                "loss_mix_l2": float(loss_mix.detach().cpu()),
                "fi_weight": float(args.fi_weight),
                "q_weight": float(q_weight),
                "tv_weight": float(args.tv_weight),
                "mix_l2_weight": float(args.mix_l2_weight),
                **q_metrics,
                "seconds_since_last_log": now - last_log,
                "seconds_total": now - start_time,
            }
            iter_rows.append(row)
            write_jsonl(log_path, row)
            print(
                "[{run_tag}] iter {iter:05d} | loss {loss:.6g} | fi {loss_fi:.3g} | "
                "q {loss_q:.3g} | tv {loss_tv:.3g} | mix {loss_mix_l2:.3g} | "
                "bin {bin_satisfaction:.4f} | "
                "req {requantization_match:.4f} | total {seconds_total:.1f}s".format(**row),
                flush=True,
            )
            last_log = now
            write_json(
                {
                    "status": "running",
                    "run_tag": tag,
                    "iter": iteration,
                    "n_iter": args.n_iter,
                    "latest": row,
                },
                output_dir / f"{tag}_progress.json",
            )

    with torch.no_grad():
        rec_pixels = torch.sigmoid(pixel_logits).detach()
        x_norm = normalize_pixels(rec_pixels, "imagenet")
        _logits, final_z_hat = model(x_norm, return_z=True)
        final_fi_loss = (
            1.0 - F.cosine_similarity(final_z_hat, z_target, dim=-1, eps=1e-10)
        ).mean()
        final_q_loss, final_mix_l2, final_q_metrics = q_objective_from_features(
            final_z_hat,
            clients,
            mixing_params,
        )

    return OptimizationResult(
        tag=tag,
        pixel_logits=pixel_logits,
        rec_pixels=rec_pixels.detach(),
        final_z_hat=final_z_hat.detach(),
        iter_rows=iter_rows,
        final_fi_loss=float(final_fi_loss.detach().cpu()),
        final_q_loss=float(final_q_loss.detach().cpu()),
        final_mix_l2=float(final_mix_l2.detach().cpu()),
        final_q_metrics=final_q_metrics,
    )


def result_arrays_and_metrics(
    result: OptimizationResult,
    original_pixels: torch.Tensor,
    metric_computer: ImageMetricComputer,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    metrics_summary, metrics_by_sample = metric_computer.compute(original_pixels, result.rec_pixels)
    rec_np = result.rec_pixels.detach().cpu().numpy().astype(np.float32, copy=False)
    z_np = result.final_z_hat.detach().cpu().numpy().astype(np.float32, copy=False)
    return rec_np, z_np, metrics_summary, metrics_by_sample


def comparison_summary(
    *,
    fia_only_metrics: Dict[str, Any],
    qaware_metrics: Dict[str, Any],
    fia_only_result: OptimizationResult,
    qaware_result: OptimizationResult,
) -> Dict[str, Any]:
    ssim_gain = None
    if fia_only_metrics.get("ssim_mean") is not None and qaware_metrics.get("ssim_mean") is not None:
        ssim_gain = float(qaware_metrics["ssim_mean"] - fia_only_metrics["ssim_mean"])
    lpips_reduction = None
    if fia_only_metrics.get("lpips_mean") is not None and qaware_metrics.get("lpips_mean") is not None:
        lpips_reduction = float(fia_only_metrics["lpips_mean"] - qaware_metrics["lpips_mean"])
    return {
        "image_oracle_metrics": {
            "fia_only": fia_only_metrics,
            "qaware_frozen_a": qaware_metrics,
            "mse_mean_reduction_fia_only_minus_qaware": float(
                fia_only_metrics["mse_mean"] - qaware_metrics["mse_mean"]
            ),
            "psnr_mean_gain_qaware_minus_fia_only": float(
                qaware_metrics["psnr_mean"] - fia_only_metrics["psnr_mean"]
            ),
            "ssim_mean_gain_qaware_minus_fia_only": ssim_gain,
            "lpips_mean_reduction_fia_only_minus_qaware": lpips_reduction,
        },
        "q_consistency_metrics": {
            "fia_only": {
                "loss_q": fia_only_result.final_q_loss,
                **fia_only_result.final_q_metrics,
            },
            "qaware_frozen_a": {
                "loss_q": qaware_result.final_q_loss,
                **qaware_result.final_q_metrics,
            },
            "loss_q_reduction_fia_only_minus_qaware": float(
                fia_only_result.final_q_loss - qaware_result.final_q_loss
            ),
            "bin_satisfaction_gain_qaware_minus_fia_only": float(
                qaware_result.final_q_metrics["bin_satisfaction"]
                - fia_only_result.final_q_metrics["bin_satisfaction"]
            ),
            "mean_bin_violation_reduction_fia_only_minus_qaware": float(
                fia_only_result.final_q_metrics["mean_bin_violation"]
                - qaware_result.final_q_metrics["mean_bin_violation"]
            ),
        },
        "fairness": "Both branches start from identical pixel logits and use the same frozen A_k.",
    }


def run(args: argparse.Namespace) -> Path:
    run_dir = resolve_path(args.run_dir)
    client_qs_dir = run_dir / "client_qs"
    quant_meta_path = client_qs_dir / "final_round_client_quant_meta.json"
    context_path = client_qs_dir / "final_round_context.pkl"
    root_manifest_path = run_dir / "manifest.json"
    for path in (quant_meta_path, context_path, root_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    quant_meta = read_json(quant_meta_path)
    root_manifest = read_json(root_manifest_path)
    if quant_meta.get("quant_bits") != 4 or quant_meta.get("rounding") != "nearest":
        raise ValueError("End-to-end q-aware FIA requires nearest int4 q/s artifacts")
    context = read_pickle(context_path)

    n_clients = len(quant_meta["clients"])
    client_ids = parse_client_ids(args.client_ids, n_clients)
    selected_names = selected_param_names(args, quant_meta)
    attack_layer_name = str(quant_meta["attack_layer"]["name"])
    if selected_names != [attack_layer_name]:
        raise ValueError(
            "QICA-conditioned q-aware FIA currently reconstructs only "
            f"the attack layer {attack_layer_name!r}. Use --loss_scope attack_layer."
        )

    seed = int(root_manifest["seed"] if args.seed is None else args.seed)

    set_seed(seed)
    device = choose_device(args.device)
    model, model_info = load_model(quant_meta, device)

    clients, selected_global_indices = load_clients(
        client_qs_dir=client_qs_dir,
        quant_meta=quant_meta,
        context=context,
        client_ids=client_ids,
        names=selected_names,
        device=device,
    )
    image_shape = infer_image_shape(context, selected_global_indices)
    output_dir = default_output_dir(args, run_dir, image_shape[0])
    if args.fresh_start:
        safe_rmtree(output_dir, allowed_root=output_dir.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_context = np.asarray(context["x"], dtype=np.float32)[np.asarray(selected_global_indices, dtype=np.int64)]
    y_context = np.asarray(context["y"])[np.asarray(selected_global_indices, dtype=np.int64)]
    z_target, embedding_diagnostics = load_embedding_target(
        args.embedding_path,
        selected_global_indices,
        context,
        device,
        args.embedding_alignment,
    )

    pixel_logits = initialize_pixel_logits(
        args=args,
        shape=image_shape,
        selected_indices=selected_global_indices,
        device=device,
        seed=seed,
    )
    initial_pixel_logits = pixel_logits.detach().clone()
    del pixel_logits
    mixing_params = initialize_mixing_parameters(
        clients=clients,
        z_source=z_target,
        args=args,
        seed=seed,
    )
    diagnostics: Dict[str, Any] = {}
    if args.diagnostics:
        diagnostics = build_diagnostics(
            clients=clients,
            selected_indices=selected_global_indices,
            mixing_params=mixing_params,
            z_target=z_target,
            embedding_diagnostics=embedding_diagnostics,
            context=context,
            seed=seed,
        )

    print(
        "QICA embedding-conditioned q-aware FIA | "
        f"run_dir={run_dir} | clients={client_ids} | samples={image_shape[0]} | "
        f"attack_layer={attack_layer_name} | mixing_init={args.mixing_init} | "
        f"train_mixing={args.train_mixing} | embedding_alignment={args.embedding_alignment} | "
        f"compare_fia_only={args.compare_fia_only} | embedding={resolve_path(args.embedding_path)} | "
        f"device={device}",
        flush=True,
    )

    if args.q_grad_check:
        probe_logits = initial_pixel_logits.detach().clone().requires_grad_(True)
        initial_q_gradient_probe = q_gradient_probe(
            model,
            probe_logits,
            clients,
            mixing_params,
            eps=float(args.grad_nonzero_eps),
        )
        diagnostics["initial_lq_pixel_gradient_probe"] = initial_q_gradient_probe
        if (
            args.fail_on_zero_q_grad
            and initial_q_gradient_probe["status"] == "failed_nonzero_q_loss_but_zero_pixel_gradient"
        ):
            raise RuntimeError(f"L_q gradient probe failed: {initial_q_gradient_probe}")

    fia_only_result: OptimizationResult | None = None
    qaware_result: OptimizationResult
    was_training = model.training
    model.eval()
    try:
        if args.compare_fia_only:
            fia_only_result = optimize_pixels(
                tag="fia_only",
                model=model,
                initial_pixel_logits=initial_pixel_logits,
                z_target=z_target,
                clients=clients,
                mixing_params=mixing_params,
                args=args,
                q_weight=0.0,
                output_dir=output_dir,
                log_filename="fia_only_losses.jsonl",
            )
        qaware_result = optimize_pixels(
            tag="qaware_train_a" if args.train_mixing else "qaware_frozen_a",
            model=model,
            initial_pixel_logits=initial_pixel_logits,
            z_target=z_target,
            clients=clients,
            mixing_params=mixing_params,
            args=args,
            q_weight=float(args.q_weight),
            output_dir=output_dir,
            log_filename="losses.jsonl",
        )
        if args.q_grad_check:
            final_q_gradient_probe = q_gradient_probe(
                model,
                qaware_result.pixel_logits,
                clients,
                mixing_params,
                eps=float(args.grad_nonzero_eps),
            )
            diagnostics["final_lq_pixel_gradient_probe"] = final_q_gradient_probe
    finally:
        model.train(was_training)

    with torch.no_grad():
        original_pixels = unnormalize_images(torch.as_tensor(x_context, device=device, dtype=torch.float32), "imagenet")
        orig_np = original_pixels.detach().cpu().numpy().astype(np.float32, copy=False)
        metric_computer = ImageMetricComputer(
            device,
            skip_ssim=bool(args.skip_ssim),
            skip_lpips=bool(args.skip_lpips),
        )
        rec_np, final_z_np, metrics_summary, metrics_by_sample = result_arrays_and_metrics(
            qaware_result,
            original_pixels,
            metric_computer,
        )
        z_target_np = z_target.detach().cpu().numpy().astype(np.float32, copy=False)
        mixing_np = [
            mix.detach().cpu().numpy().astype(np.float32, copy=False)
            for mix in mixing_params
        ]

    comparison: Dict[str, Any] | None = None
    if fia_only_result is not None:
        fia_rec_np, fia_z_np, fia_metrics_summary, fia_metrics_by_sample = result_arrays_and_metrics(
            fia_only_result,
            original_pixels,
            metric_computer,
        )
        write_csv(fia_only_result.iter_rows, output_dir / "fia_only_iter.csv")
        write_csv(fia_metrics_by_sample, output_dir / "fia_only_image_metrics_by_sample.csv")
        write_json(
            {
                "run_tag": "fia_only",
                "metrics_summary": fia_metrics_summary,
                "metrics_by_sample": fia_metrics_by_sample,
                "metrics_batch": metric_batch_from_rows(fia_metrics_by_sample),
                "final_fi_loss": fia_only_result.final_fi_loss,
                "final_q_loss": fia_only_result.final_q_loss,
                "final_q_metrics": fia_only_result.final_q_metrics,
            },
            output_dir / "fia_only_metrics.json",
        )
        write_pickle(
            {
                "run_dir": str(run_dir),
                "client_ids": client_ids,
                "selected_global_indices": selected_global_indices,
                "selected_param_names": selected_names,
                "rec_attack_order": fia_rec_np,
                "x_original": orig_np,
                "z_target_qica": z_target_np,
                "z_hat_final": fia_z_np,
                "labels": y_context,
                "mixing_matrices": mixing_np,
                "iter_rows": fia_only_result.iter_rows,
                "metrics_summary": fia_metrics_summary,
                "metrics_by_sample": fia_metrics_by_sample,
                "metrics_batch": metric_batch_from_rows(fia_metrics_by_sample),
                "final_fi_loss": fia_only_result.final_fi_loss,
                "final_q_loss": fia_only_result.final_q_loss,
                "final_q_metrics": fia_only_result.final_q_metrics,
                "model_info": model_info,
            },
            output_dir / "fia_only_recon.pkl",
        )
        if not args.skip_eval_png:
            make_pair_grid(orig_np, fia_rec_np, output_dir / "png" / "fia_only_compare.png", max_items=args.max_png_items)
        comparison = comparison_summary(
            fia_only_metrics=fia_metrics_summary,
            qaware_metrics=metrics_summary,
            fia_only_result=fia_only_result,
            qaware_result=qaware_result,
        )
        write_json(comparison, output_dir / "comparison_fia_only_vs_qaware.json")

    write_csv(qaware_result.iter_rows, output_dir / "e2e_iter.csv")
    write_csv(metrics_by_sample, output_dir / "image_metrics_by_sample.csv")
    write_json(
        {
            "run_tag": qaware_result.tag,
            "metrics_summary": metrics_summary,
            "metrics_by_sample": metrics_by_sample,
            "metrics_batch": metric_batch_from_rows(metrics_by_sample),
            "final_fi_loss": qaware_result.final_fi_loss,
            "final_q_loss": qaware_result.final_q_loss,
            "final_q_metrics": qaware_result.final_q_metrics,
        },
        output_dir / "fia_metrics.json",
    )
    write_pickle(
        {
            "run_dir": str(run_dir),
            "client_ids": client_ids,
            "selected_global_indices": selected_global_indices,
            "selected_param_names": selected_names,
            "rec_attack_order": rec_np,
            "x_original": orig_np,
            "z_target_qica": z_target_np,
            "z_hat_final": final_z_np,
            "labels": y_context,
            "mixing_matrices": mixing_np,
            "iter_rows": qaware_result.iter_rows,
            "metrics_summary": metrics_summary,
            "metrics_by_sample": metrics_by_sample,
            "metrics_batch": metric_batch_from_rows(metrics_by_sample),
            "final_fi_loss": qaware_result.final_fi_loss,
            "final_q_loss": qaware_result.final_q_loss,
            "final_q_metrics": qaware_result.final_q_metrics,
            "diagnostics": diagnostics,
            "comparison_fia_only_vs_qaware": comparison,
            "model_info": model_info,
        },
        output_dir / "end_to_end_qaware_fia_recon.pkl",
    )
    if not args.skip_eval_png:
        make_pair_grid(orig_np, rec_np, output_dir / "png" / "e2e_qaware_fia_compare.png", max_items=args.max_png_items)
    if diagnostics:
        write_json(diagnostics, output_dir / "diagnostics.json")

    manifest = {
        "schema_version": 1,
        "stage": "end_to_end_quantization_aware_fia",
        "method": "qica_embedding_conditioned_fia_with_attack_layer_int4_bin_consistency",
        "attack_inputs": {
            "quant_meta_path": str(quant_meta_path.resolve()),
            "quant_meta_sha256": sha256_file(quant_meta_path),
            "codes_path": str((client_qs_dir / "final_round_client_codes.npz").resolve()),
            "codes_sha256": quant_meta.get("codes_sha256"),
            "checkpoint_path": model_info["checkpoint_path"],
            "checkpoint_sha256": model_info["checkpoint_sha256"],
            "embedding_path": str(resolve_path(args.embedding_path)),
            "embedding_used_as": "required_fia_feature_target",
            "embedding_alignment": embedding_diagnostics,
            "init_pixels": str(resolve_path(args.init_pixels)) if args.init_pixels is not None else None,
        },
        "label_policy": "labels are not used by the QICA-conditioned q-consistency objective",
        "gradient_path": (
            "L_fi -> z_hat(x) -> pixels; "
            "L_q -> A_k @ z_hat_k(x) -> z_hat_k(x) -> pixels"
        ),
        "loss": {
            "formula": "fi_weight * (1 - cosine(z_hat(x), z_qica)) + q_weight * attack_layer_bin_loss(A_k @ z_hat_k(x)) + tv_weight * TV(pixels) + mix_l2_weight * ||A_k||^2",
            "bin_loss": "mean squared distance outside finite nearest-rounding bins",
            "fia_feature_loss": "1 - cosine(z_hat(x), recovered QICA embedding)",
            "main_objective": "FIA image recovery conditioned on recovered QICA embeddings, with attack-layer int4 consistency",
        },
        "mixing_factorization": {
            "source": "current model feature z_hat(x), classifier.0 input",
            "reconstruction": "for each client k, update_hat_k = A_k @ z_hat_k(x)",
            "mixing_init": args.mixing_init,
            "mixing_ridge": args.mixing_ridge,
            "mixing_initialized_from": "recovered QICA embedding target z_qica",
            "optimized_mixing": bool(args.train_mixing),
            "optimized_pixels": True,
        },
        "diagnostics": diagnostics,
        "comparison_fia_only_vs_qaware": comparison,
        "selection": {
            "client_ids": client_ids,
            "selected_global_indices": selected_global_indices,
            "loss_scope": args.loss_scope,
            "selected_param_names": selected_names,
        },
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "metrics_summary": metrics_summary,
    }
    write_json(manifest, output_dir / "manifest.json")
    write_json({"status": "complete", "output_dir": str(output_dir), "metrics_summary": metrics_summary}, output_dir / "progress.json")
    return output_dir


def main() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    args = parse_args()
    if args.n_iter < 1:
        raise ValueError("--n_iter must be >= 1")
    if args.n_log < 0:
        raise ValueError("--n_log must be >= 0")
    if args.compare_fia_only and args.train_mixing:
        raise ValueError("--compare_fia_only requires frozen A_k; use --no-train_mixing")
    output_dir = run(args)
    print(f"End-to-end q-aware FIA output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
