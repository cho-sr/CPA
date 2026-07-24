#!/usr/bin/env python3
"""Experiment 05: quantization-aware FedAvg update-matching refinement."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

import common as common_mod
from common import (
    OUTPUT_ROOT,
    assert_matching_manifest,
    build_experiment_model,
    hash_tensor_sequence,
    parameter_schema,
    read_pickle,
    run_tag,
    safe_rmtree,
    sha256_file,
    update_experiment_manifest,
    validate_parameter_mapping,
    write_json,
)
from quantization import bin_distance_squared, require_nearest_rounding

common_mod._install_optional_dependency_stubs()

import utils as author_utils  # noqa: E402
from datasets import nclasses_dict  # noqa: E402

try:  # noqa: E402
    from torch.func import functional_call
except ImportError:  # pragma: no cover - older PyTorch fallback
    from torch.nn.utils.stateless import functional_call  # type: ignore


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), default=_json_default, sort_keys=True) + "\n")


def _save_pickle(path: Path, value: Any) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _state_dict_params(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, p.detach().clone().requires_grad_(True)) for name, p in model.named_parameters())


def _param_tuple(params: OrderedDict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(params.values())


def _make_param_dict(names: list[str], values: tuple[torch.Tensor, ...]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, value) for name, value in zip(names, values))


def _normalize_pixels(pixels: torch.Tensor, ds: str) -> torch.Tensor:
    mean = torch.tensor(author_utils.ds_mean[ds], dtype=pixels.dtype, device=pixels.device).view(1, 3, 1, 1)
    std = torch.tensor(author_utils.ds_std[ds], dtype=pixels.dtype, device=pixels.device).view(1, 3, 1, 1)
    return (pixels - mean) / std


def _total_variation(pixels: torch.Tensor) -> torch.Tensor:
    dx = torch.mean(torch.abs(pixels[:, :, :, :-1] - pixels[:, :, :, 1:]))
    dy = torch.mean(torch.abs(pixels[:, :, :-1, :] - pixels[:, :, 1:, :]))
    return dx + dy


def _soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    return -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def _forward_with_params(model: torch.nn.Module, params: OrderedDict[str, torch.Tensor], x: torch.Tensor):
    return functional_call(model, params, (x,))


def _forward_with_params_return_z(
    model: torch.nn.Module, params: OrderedDict[str, torch.Tensor], x: torch.Tensor
):
    return functional_call(model, params, (x,), {"return_z": True})


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
) -> OrderedDict[str, torch.Tensor]:
    names = list(theta0.keys())
    params = OrderedDict((name, value) for name, value in theta0.items())
    for _step in range(steps):
        pred = _forward_with_params(model, params, x_norm)
        if label_probs is not None:
            loss = _soft_cross_entropy(pred, label_probs)
        elif labels is not None:
            loss = F.cross_entropy(pred, labels)
        else:
            raise ValueError("Either hard labels or soft label probabilities are required")
        grads = torch.autograd.grad(
            loss,
            _param_tuple(params),
            create_graph=create_graph,
            retain_graph=create_graph,
        )
        params = _make_param_dict(names, tuple(p - lr * g for p, g in zip(params.values(), grads)))
    return params


def validate_05_supported(args, update_data: Mapping[str, Any], model: torch.nn.Module) -> None:
    require_nearest_rounding(args.rounding)
    metadata = update_data["metadata"]
    if metadata["quant_bits"] != 4:
        raise ValueError("05 requires int4 quantized updates")
    if metadata["rounding"] != "nearest":
        raise ValueError("05 supports nearest rounding only")
    if metadata["optimizer"] != "SGD":
        raise ValueError("05 currently supports only SGD collector updates")
    if float(metadata["momentum"]) != 0.0:
        raise ValueError("05 requires momentum=0")
    if float(metadata["weight_decay"]) != 0.0:
        raise ValueError("05 requires weight_decay=0")
    if int(metadata["local_batch_size"]) != int(metadata["n_samples"]):
        raise ValueError("05 requires full-batch local SGD: local_batch_size == n_samples")
    if int(metadata["local_steps"]) != int(args.local_epochs):
        raise ValueError("05 expects local_steps == local_epochs under full-batch SGD")
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            raise ValueError("05 refuses models with BatchNorm/stateful normalization layers")
        if isinstance(module, torch.nn.Dropout) and float(module.p) > 0.0:
            raise ValueError("05 refuses active dropout; dropout must be absent or p=0")


def reproduce_fp32_update(
    *,
    args,
    fp32_update_file: Path,
    batch: int,
    model: torch.nn.Module,
) -> dict[str, Any]:
    data = read_pickle(fp32_update_file)
    validate_parameter_mapping(data["param_names"], data["grad"][batch], model)
    device = next(model.parameters()).device
    theta0 = _state_dict_params(model)
    x = torch.as_tensor(data["x"][batch], device=device, dtype=torch.float32)
    y = torch.as_tensor(data["y"][batch], device=device, dtype=torch.long)
    theta_t = unroll_full_batch_sgd(
        model=model,
        theta0=theta0,
        x_norm=x,
        labels=y,
        label_probs=None,
        steps=int(data["metadata"]["local_steps"]),
        lr=float(data["metadata"]["lr"]),
        create_graph=False,
    )
    max_abs = 0.0
    mean_abs_num = 0.0
    mean_abs_den = 0
    rel_num = 0.0
    rel_den = 0.0
    per_param = []
    for name, stored in zip(data["param_names"], data["grad"][batch]):
        predicted = theta_t[name] - theta0[name]
        target = torch.as_tensor(stored, device=device, dtype=predicted.dtype)
        err = predicted - target
        max_err = float(err.abs().max().detach().cpu())
        mean_err = float(err.abs().mean().detach().cpu())
        rel_num += float(torch.sum(err.detach() ** 2).cpu())
        rel_den += float(torch.sum(target.detach() ** 2).cpu())
        mean_abs_num += float(err.abs().sum().detach().cpu())
        mean_abs_den += err.numel()
        per_param.append({"name": name, "max_abs_error": max_err, "mean_abs_error": mean_err})
        max_abs = max(max_abs, max_err)
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs_num / max(1, mean_abs_den),
        "relative_l2_error": (rel_num / max(rel_den, 1e-24)) ** 0.5,
        "per_parameter": per_param,
        "passed": max_abs <= args.fp32_repro_max_abs_tol and (rel_num / max(rel_den, 1e-24)) ** 0.5 <= args.fp32_repro_rel_l2_tol,
    }


def _load_initial_pixels(
    *,
    init_run_path: Path,
    batch: int,
    n_samples: int,
    device: torch.device,
) -> torch.Tensor:
    rec_path = init_run_path / "reconstructions.pkl"
    if not rec_path.exists():
        raise FileNotFoundError(f"Missing CPA initialization reconstructions: {rec_path}")
    rows = read_pickle(rec_path)
    if batch >= len(rows):
        raise ValueError(f"Initialization path has only {len(rows)} batches, missing batch {batch}")
    rec = torch.as_tensor(rows[batch]["rec_fi_attack_order"], device=device, dtype=torch.float32)
    if rec.numel() == 0:
        rec = torch.as_tensor(rows[batch]["rec_gi"], device=device, dtype=torch.float32)
    if rec.ndim != 4:
        raise ValueError("CPA initialization does not contain image-shaped reconstructions")
    if int(rec.shape[0]) != int(n_samples):
        raise ValueError(
            f"05 requires initialization for all {n_samples} samples; got {int(rec.shape[0])}. "
            "Run the init CPA/FIA stage with --n_sample_fi equal to --n_samples."
        )
    return rec.clamp(1e-4, 1 - 1e-4)


class FedAvgRefiner:
    def __init__(
        self,
        *,
        args,
        model: torch.nn.Module,
        update_data: Mapping[str, Any],
        batch: int,
        init_pixels: torch.Tensor,
    ) -> None:
        self.args = args
        self.model = model
        self.update_data = update_data
        self.batch = batch
        self.device = init_pixels.device
        self.theta0 = _state_dict_params(model)
        self.param_names = list(self.theta0.keys())
        self.pixel_logits = torch.nn.Parameter(torch.logit(init_pixels.clamp(1e-4, 1 - 1e-4)))
        self.label_logits = None
        if args.label_mode == "unknown":
            n_classes = int(nclasses_dict[args.ds])
            self.label_logits = torch.nn.Parameter(torch.zeros(init_pixels.shape[0], n_classes, device=self.device))
            params = [self.pixel_logits, self.label_logits]
        elif args.label_mode == "known":
            params = [self.pixel_logits]
        else:
            raise ValueError("label_mode must be known or unknown")
        self.optimizer = torch.optim.Adam(params, lr=args.refine_lr)
        self.start_iter = 0

    def state_dict(self) -> dict[str, Any]:
        state = {
            "pixel_logits": self.pixel_logits.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.label_logits is not None:
            state["label_logits"] = self.label_logits.detach().cpu()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.pixel_logits.data = state["pixel_logits"].to(self.device, dtype=self.pixel_logits.dtype)
        if self.label_logits is not None and "label_logits" in state:
            self.label_logits.data = state["label_logits"].to(self.device, dtype=self.label_logits.dtype)
        self.optimizer.load_state_dict(state["optimizer"])

    def _labels(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.args.label_mode == "known":
            labels = torch.as_tensor(self.update_data["y"][self.batch], device=self.device, dtype=torch.long)
            return labels, None
        assert self.label_logits is not None
        return None, F.softmax(self.label_logits, dim=-1)

    def step(self) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        pixels = torch.sigmoid(self.pixel_logits)
        x_norm = _normalize_pixels(pixels, self.args.ds)
        labels, label_probs = self._labels()
        theta_t = unroll_full_batch_sgd(
            model=self.model,
            theta0=self.theta0,
            x_norm=x_norm,
            labels=labels,
            label_probs=label_probs,
            steps=int(self.update_data["metadata"]["local_steps"]),
            lr=float(self.update_data["metadata"]["lr"]),
            create_graph=True,
        )
        loss_update = torch.tensor(0.0, device=self.device)
        elems = 0
        for name, target_codes_np, stats in zip(
            self.param_names,
            self.update_data["quant_codes"][self.batch],
            self.update_data["quant_stats"][self.batch]["per_tensor"],
        ):
            delta_hat = theta_t[name] - self.theta0[name]
            codes = torch.as_tensor(target_codes_np, device=self.device, dtype=torch.int8)
            dist = bin_distance_squared(delta_hat, codes, float(stats["scale"]))
            loss_update = loss_update + dist.sum()
            elems += dist.numel()
        loss_update = loss_update / max(1, elems)

        loss_tv = _total_variation(pixels) if self.args.refine_tv_weight > 0 else torch.tensor(0.0, device=self.device)
        loss = (self.args.refine_update_weight * loss_update) + (self.args.refine_tv_weight * loss_tv)
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "loss_update_bin": float(loss_update.detach().cpu()),
            "loss_tv": float(loss_tv.detach().cpu()),
        }

    def get_pixels(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.pixel_logits).detach()


def _load_ckpt(path: Path) -> dict[str, Any] | None:
    if path.exists():
        return torch.load(path, map_location="cpu")
    return None


def _save_ckpt(path: Path, *, iteration: int, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"iteration": iteration, "state": dict(state)}, path)


def run_fedavg_refinement(
    args,
    *,
    int4_update_file: Path,
    fp32_update_file: Path,
    init_run_path: Path,
) -> Path:
    require_nearest_rounding(args.rounding)
    update_data = read_pickle(int4_update_file)
    device = author_utils.get_device()
    model = build_experiment_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        device=device,
        checkpoint_path=args.global_checkpoint,
        seed=args.seed,
    )
    model.eval()
    validate_parameter_mapping(update_data["param_names"], update_data["grad"][0], model)
    validate_05_supported(args, update_data, model)

    n_batches = min(args.attack_n_batch or int(update_data["metadata"]["attack_trials"]), int(update_data["metadata"]["attack_trials"]))
    display_n = min(args.display_samples, args.n_samples)
    exp_name = f"05_fedavg_refinement_{args.label_mode}_{run_tag(args, include_fia=False)}_it{args.refine_n_iter}"
    output_dir = OUTPUT_ROOT / "05_quant_aware_fedavg_refinement" / args.ds / args.model / exp_name
    if getattr(args, "fresh_start", False):
        safe_rmtree(output_dir, allowed_root=OUTPUT_ROOT / "05_quant_aware_fedavg_refinement")

    extra = {
        "method_label": "CPA-initialized quantization-aware FedAvg update-matching refinement",
        "label_mode": args.label_mode,
        "known_label_policy": "true labels fixed" if args.label_mode == "known" else "trainable logits with softmax probabilities",
        "update_matching_samples": args.n_samples,
        "display_samples": display_n,
        "full_batch_local_sgd_unroll": True,
        "create_graph": True,
        "unsupported_conditions_error": True,
        "init_run_path": str(init_run_path.resolve()),
        "source_fp32_update_file": str(fp32_update_file.resolve()),
        "source_fp32_update_hash": sha256_file(fp32_update_file),
        "executed_attack_trials": n_batches,
        "total_reconstructed_samples": n_batches * args.n_samples,
        "total_evaluated_samples": n_batches * display_n,
    }
    manifest = update_experiment_manifest(
        experiment_id=exp_name,
        method="CPA-initialized quantization-aware FedAvg update-matching refinement",
        update_file=int4_update_file,
        args=args,
        evaluated_samples=args.n_samples,
        extra=extra,
    )
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists():
        assert_matching_manifest(manifest_path, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)

    repro = reproduce_fp32_update(args=args, fp32_update_file=fp32_update_file, batch=0, model=model)
    write_json(output_dir / "fp32_reproduction_check.json", repro)
    if not repro["passed"]:
        raise RuntimeError("FP32 update reproduction check failed; refusing to run 05")

    log_path = output_dir / "losses.jsonl"
    recon_rows = []
    for batch in range(n_batches):
        init_pixels = _load_initial_pixels(
            init_run_path=init_run_path,
            batch=batch,
            n_samples=args.n_samples,
            device=device,
        )
        refiner = FedAvgRefiner(
            args=args,
            model=model,
            update_data=update_data,
            batch=batch,
            init_pixels=init_pixels,
        )
        ckpt_path = output_dir / f"batch_{batch:03d}_refine.pt"
        ckpt = _load_ckpt(ckpt_path)
        if ckpt is not None:
            refiner.load_state_dict(ckpt["state"])
            refiner.start_iter = int(ckpt["iteration"])
        for iter_idx in range(refiner.start_iter, args.refine_n_iter):
            loss_dict = refiner.step()
            if (iter_idx % args.refine_n_log == 0) or (iter_idx == args.refine_n_iter - 1):
                _write_jsonl(log_path, {"batch": batch, "iter": iter_idx, **loss_dict})
                _save_ckpt(ckpt_path, iteration=iter_idx + 1, state=refiner.state_dict())
        pixels = refiner.get_pixels().detach().cpu().numpy()
        recon_rows.append(
            {
                "batch": batch,
                "sample_indices": update_data["sample_indices"][batch],
                "optimized_pixels_all_samples": pixels,
                "display_pixels": pixels[:display_n],
                "label_mode": args.label_mode,
            }
        )
    _save_pickle(output_dir / "refined_reconstructions.pkl", recon_rows)
    write_json(
        output_dir / "run_summary.json",
        {
            "optimized_batches": n_batches,
            "optimized_samples_per_batch": args.n_samples,
            "display_samples_per_batch": display_n,
            "optimized_tensor_hash": hash_tensor_sequence(
                [row["optimized_pixels_all_samples"] for row in recon_rows]
            ),
            "note": "Oracle labels are used only in known-label threat mode; unknown labels use trainable logits softmax.",
        },
    )
    return output_dir
