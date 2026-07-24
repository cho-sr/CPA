#!/usr/bin/env python3
"""Quantization-aware ICA objective used by experiment 04.

This is intentionally not a patch to the public author source.  It is a
separate proposal: optimize an ICA unmixing matrix on a candidate attack-layer
observation that is allowed to move inside the observed int4 bins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

import common as common_mod
from author_runner import run_direct_fia_from_embeddings
from blind_alignment import oracle_permutation_aligned_cosine_similarity
from common import (
    OUTPUT_ROOT,
    assert_matching_manifest,
    build_experiment_model,
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


ABLATIONS = {
    "qoff_neoff": {"use_q_loss": False, "use_ne": False},
    "qoff_neon": {"use_q_loss": False, "use_ne": True},
    "qon_neoff": {"use_q_loss": True, "use_ne": False},
    "qon_neon": {"use_q_loss": True, "use_ne": True},
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
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


def _whiten(
    x: torch.Tensor, eps: float, *, n_components: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_mu = x.mean(dim=-1, keepdim=True)
    x_zc = x - x_mu
    cov = x_zc.matmul(x_zc.T) / max(1, x_zc.shape[1] - 1)
    cov = 0.5 * (cov + cov.T)
    jitter = eps * cov.diagonal().abs().mean().clamp_min(1.0)
    cov = cov + torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device) * jitter
    eig_vals, eig_vecs = torch.linalg.eigh(cov)
    topk = torch.topk(eig_vals.abs(), k=min(n_components, cov.shape[0]))[1]
    lamb = eig_vals[topk].abs().clamp_min(eps)
    whitening = torch.diag(torch.rsqrt(lamb)).matmul(eig_vecs.T[topk])
    return whitening.matmul(x_zc), whitening, x_mu


def standardized_negentropy_loss(sources: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Minimization loss: negative standardized log-cosh negentropy contrast."""
    centered = sources - sources.mean(dim=-1, keepdim=True)
    standardized = centered / (centered.std(dim=-1, keepdim=True, unbiased=False) + eps)
    g_s = torch.log(torch.cosh(standardized.clamp(-10, 10))).mean(dim=-1)
    # Stable Monte Carlo-free constant for v~N(0,1), close enough for the contrast.
    normal_ref = torch.tensor(0.374567, dtype=sources.dtype, device=sources.device)
    contrast = (g_s - normal_ref).square()
    return -contrast.mean()


class WhitenedQuantAwareICA:
    def __init__(
        self,
        *,
        observation: torch.Tensor,
        codes: torch.Tensor,
        scale: float,
        args,
        use_q_loss: bool,
        use_ne: bool,
    ) -> None:
        self.args = args
        self.device = observation.device
        self.eps = float(args.qica_eps)
        self.codes = codes.to(self.device)
        self.scale = float(scale)
        self.use_q_loss = use_q_loss
        self.use_ne = use_ne
        self.n_components = min(int(args.n_samples), int(observation.shape[0]))
        self.learn_observation = bool(use_q_loss and args.qica_train_observation)
        self.X_hat = torch.nn.Parameter(observation.detach().clone()) if self.learn_observation else observation.detach()
        self.W_hat = torch.nn.Parameter(torch.eye(self.n_components, device=self.device, dtype=observation.dtype))
        params: list[torch.Tensor] = [self.W_hat]
        if isinstance(self.X_hat, torch.nn.Parameter):
            params.append(self.X_hat)
        self.optimizer = torch.optim.Adam(params, lr=args.qica_lr)
        self.start_iter = 0

    def _source(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X_hat if isinstance(self.X_hat, torch.Tensor) else torch.as_tensor(self.X_hat, device=self.device)
        x_w, whitening, x_mu = _whiten(x, self.eps, n_components=self.n_components)
        w_norm = self.W_hat / (self.W_hat.norm(dim=-1, keepdim=True) + self.eps)
        source = w_norm.matmul(x_w)
        source = source + w_norm.matmul(whitening).matmul(x_mu)
        return source, x

    def state_dict(self) -> dict[str, Any]:
        state = {
            "W_hat": self.W_hat.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
            "learn_observation": self.learn_observation,
        }
        if isinstance(self.X_hat, torch.nn.Parameter):
            state["X_hat"] = self.X_hat.detach().cpu()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.W_hat.data = state["W_hat"].to(self.device, dtype=self.W_hat.dtype)
        if isinstance(self.X_hat, torch.nn.Parameter) and "X_hat" in state:
            self.X_hat.data = state["X_hat"].to(self.device, dtype=self.X_hat.dtype)
        self.optimizer.load_state_dict(state["optimizer"])

    def step(self) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        source, x = self._source()
        loss_q = torch.tensor(0.0, device=self.device)
        loss_ne = torch.tensor(0.0, device=self.device)
        loss_decor = torch.tensor(0.0, device=self.device)
        loss_l1 = torch.tensor(0.0, device=self.device)
        loss_nv = torch.tensor(0.0, device=self.device)

        if self.use_q_loss:
            if self.args.qica_columns_per_step and self.args.qica_columns_per_step < x.shape[1]:
                cols = torch.randint(x.shape[1], (self.args.qica_columns_per_step,), device=self.device)
                loss_q = bin_distance_squared(x[:, cols], self.codes[:, cols], self.scale).mean()
            else:
                loss_q = bin_distance_squared(x, self.codes, self.scale).mean()
        if self.use_ne:
            loss_ne = standardized_negentropy_loss(source, eps=self.eps)
        if self.args.decor > 0:
            w_norm = self.W_hat / (self.W_hat.norm(dim=-1, keepdim=True) + self.eps)
            cos_matrix = w_norm.matmul(w_norm.T).abs()
            loss_decor = (torch.exp(cos_matrix * self.args.T) - 1).mean()
        if self.args.l1 > 0:
            loss_l1 = source.abs().mean()
        if self.args.nv > 0:
            loss_nv = torch.minimum(F.relu(-source).norm(dim=-1), F.relu(source).norm(dim=-1)).mean()

        loss = (
            (self.args.qica_q_weight * loss_q)
            + loss_ne
            + (self.args.decor * loss_decor)
            + (self.args.l1 * loss_l1)
            + (self.args.nv * loss_nv)
        )
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "loss_qbin": float(loss_q.detach().cpu()),
            "loss_ne": float(loss_ne.detach().cpu()),
            "loss_decor": float(loss_decor.detach().cpu()),
            "loss_l1": float(loss_l1.detach().cpu()),
            "loss_nv": float(loss_nv.detach().cpu()),
        }

    def get_sources(self) -> torch.Tensor:
        with torch.no_grad():
            source, _x = self._source()
        return source.detach()


def _load_ckpt(path: Path) -> dict[str, Any] | None:
    if path.exists():
        return torch.load(path, map_location="cpu")
    return None


def _save_ckpt(path: Path, *, iteration: int, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"iteration": iteration, "state": dict(state)}, path)


def _prepare_output_dir(args, *, exp_name: str, update_file: Path, ablation: str) -> tuple[Path, dict[str, Any]]:
    output_dir = OUTPUT_ROOT / "04_quantization_aware_ica" / args.ds / args.model / exp_name / ablation
    if getattr(args, "fresh_start", False):
        safe_rmtree(output_dir, allowed_root=OUTPUT_ROOT / "04_quantization_aware_ica")
    extra = {
        "ablation": ablation,
        "qica_objective": "whitened observation plus unmixing W; no free A,S factorization",
        "quantization_consistency": "nearest-rounding bin distance with one-sided saturation bins",
        "source_negentropy": "standardized per source against Gaussian log-cosh reference",
        "qica_train_observation": bool(args.qica_train_observation),
        "qica_columns_per_step": args.qica_columns_per_step,
    }
    manifest = update_experiment_manifest(
        experiment_id=f"{exp_name}_{ablation}",
        method="Quantization-aware ICA objective",
        update_file=update_file,
        args=args,
        evaluated_samples=args.n_samples,
        extra=extra,
    )
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists():
        assert_matching_manifest(manifest_path, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    return output_dir, manifest


def run_quant_aware_ica_ablation(args, *, update_file: Path, ablation: str) -> tuple[Path, list[np.ndarray]]:
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown 04 ablation: {ablation}")
    require_nearest_rounding(args.rounding)
    data = read_pickle(update_file)
    if data.get("metadata", {}).get("quant_bits") != 4:
        raise ValueError("04 requires int4 update artifacts with stored quant_codes")
    if data.get("metadata", {}).get("rounding") != "nearest":
        raise ValueError("04 supports nearest rounding only")
    if not data.get("quant_codes"):
        raise ValueError("04 requires stored integer quantization codes")

    author_utils.exp_path_base = str(OUTPUT_ROOT / "04_quantization_aware_ica")
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
    validate_parameter_mapping(data["param_names"], data["grad"][0], model)

    exp_name = f"04_qaware_ica_nearest_{run_tag(args, include_fia=False)}_it{args.qica_n_iter}"
    output_dir, _manifest = _prepare_output_dir(args, exp_name=exp_name, update_file=update_file, ablation=ablation)
    log_path = output_dir / "losses.jsonl"
    if getattr(args, "fresh_start", False) and log_path.exists():
        log_path.unlink()

    attack_layer_index = int(data["attack_layer"]["index"])
    n_batches = min(args.attack_n_batch or int(data["metadata"]["attack_trials"]), int(data["metadata"]["attack_trials"]))
    recovered_batches: list[np.ndarray] = []
    eval_rows = []
    flags = ABLATIONS[ablation]
    for batch in range(n_batches):
        observation = torch.as_tensor(data["grad"][batch][attack_layer_index], device=device, dtype=torch.float32)
        codes = torch.as_tensor(data["quant_codes"][batch][attack_layer_index], device=device, dtype=torch.int8)
        scale = float(data["quant_stats"][batch]["per_tensor"][attack_layer_index]["scale"])
        runner = WhitenedQuantAwareICA(
            observation=observation,
            codes=codes,
            scale=scale,
            args=args,
            use_q_loss=flags["use_q_loss"],
            use_ne=flags["use_ne"],
        )
        ckpt_path = output_dir / f"batch_{batch:03d}_qica.pt"
        ckpt = _load_ckpt(ckpt_path)
        if ckpt is not None:
            runner.load_state_dict(ckpt["state"])
            runner.start_iter = int(ckpt["iteration"])

        for iter_idx in range(runner.start_iter, args.qica_n_iter):
            loss_dict = runner.step()
            if (iter_idx % args.qica_n_log == 0) or (iter_idx == args.qica_n_iter - 1):
                _write_jsonl(log_path, {"batch": batch, "iter": iter_idx, "ablation": ablation, **loss_dict})
                _save_ckpt(ckpt_path, iteration=iter_idx + 1, state=runner.state_dict())
        sources = runner.get_sources().abs().detach().cpu().numpy()
        recovered_batches.append(sources)
        if len(data.get("z", [])) > batch:
            _ordered, cs = oracle_permutation_aligned_cosine_similarity(sources, data["z"][batch])
            eval_rows.append({"batch": batch, "embedding": cs})

    _save_pickle(
        output_dir / "recovered_embeddings.pkl",
        {
            "embedding_batches": recovered_batches,
            "ablation": ablation,
            "attack_order_policy": "no oracle permutation; abs sign convention only",
        },
    )
    write_json(
        output_dir / "oracle_evaluation.json",
        {"metric_note": "Oracle permutation-aligned cosine similarity, evaluation only", "batches": eval_rows},
    )
    return output_dir, recovered_batches


def selected_ablations(value: str) -> Iterable[str]:
    if value == "all":
        return ABLATIONS.keys()
    if value not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {value}")
    return [value]


def run_quant_aware_ica(args, *, update_file: Path) -> list[Path]:
    output_dirs = []
    for ablation in selected_ablations(args.qica_ablation):
        output_dir, recovered = run_quant_aware_ica_ablation(
            args, update_file=update_file, ablation=ablation
        )
        output_dirs.append(output_dir)
        if args.run_fia:
            run_direct_fia_from_embeddings(
                args,
                exp_name=f"04_qaware_ica_fia_{ablation}_{run_tag(args)}",
                update_file=update_file,
                recovered_embeddings=recovered,
                method=f"Separate author Direct FIA from 04 recovered embeddings ({ablation})",
            )
    return output_dirs
