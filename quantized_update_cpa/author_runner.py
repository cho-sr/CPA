#!/usr/bin/env python3
"""Oracle-free wrapper around the immutable public CPA implementation.

The public ``src/attack.py`` function evaluates and reorders recovered sources
inside the attack loop, then feeds the reordered source into FIA.  This module
keeps the author classes themselves intact while removing that oracle data path:
source recovery, optional direct FIA, and final evaluation are separate phases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

import common as common_mod
from blind_alignment import oracle_permutation_aligned_cosine_similarity
from common import (
    AUTHOR_RUN_ROOT,
    AUTHOR_SOURCE_HASH,
    assert_matching_manifest,
    build_experiment_model,
    make_attack_namespace,
    read_pickle,
    safe_rmtree,
    update_experiment_manifest,
    validate_parameter_mapping,
    write_json,
)

common_mod._install_optional_dependency_stubs()

import utils as author_utils  # noqa: E402
from feature_inversion import Direct  # noqa: E402
from gradient_inversion import CocktailPartyAttack  # noqa: E402


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


def _load_checkpoint(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def _save_checkpoint(path: Path, *, iteration: int, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"iteration": iteration, "state": dict(state)}, path)


def _set_author_output_root() -> None:
    AUTHOR_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    author_utils.exp_path_base = str(AUTHOR_RUN_ROOT)


def _prepare_exp_path(
    args,
    *,
    exp_name: str,
    update_file: Path,
    evaluated_samples: int,
    method: str,
    extra_manifest: Optional[Mapping[str, Any]] = None,
) -> tuple[Any, Path, dict[str, Any]]:
    _set_author_output_root()
    attack_args = make_attack_namespace(args, exp_name=exp_name)
    exp_path = Path(author_utils.get_attack_exp_path(attack_args))
    extra = {
        "source_runner": str(Path(__file__).resolve()),
        "source_runner_import": __name__,
        "author_gradient_inversion_import": CocktailPartyAttack.__module__,
        "author_feature_inversion_import": Direct.__module__,
        "author_loss_ne_behavior": "public_author_code_sets_loss_ne_to_zero_after_computing_it",
        "oracle_removed_from_attack_pipeline": True,
        "attack_pipeline_ground_truth_inputs": "none; labels only when an explicit known-label threat model is enabled",
    }
    if extra_manifest:
        extra.update(dict(extra_manifest))
    manifest = update_experiment_manifest(
        experiment_id=exp_name,
        method=method,
        update_file=update_file,
        args=args,
        evaluated_samples=evaluated_samples,
        extra=extra,
    )
    manifest_file = exp_path / "manifest.json"
    if getattr(args, "fresh_start", False):
        safe_rmtree(exp_path, allowed_root=AUTHOR_RUN_ROOT)
    elif exp_path.exists():
        assert_matching_manifest(manifest_file, manifest)
    exp_path.mkdir(parents=True, exist_ok=True)
    write_json(manifest_file, manifest)
    return attack_args, exp_path, manifest


def _tensor_list(values: list[Any], device: torch.device) -> list[torch.Tensor]:
    return [torch.as_tensor(value, device=device, dtype=torch.float32) for value in values]


def _labels_for_attack(n_samples: int, device: torch.device, true_labels: Optional[Any] = None) -> torch.Tensor:
    if true_labels is None:
        return torch.zeros(n_samples, dtype=torch.long, device=device)
    return torch.as_tensor(true_labels, dtype=torch.long, device=device)


def _image_eval(rec: torch.Tensor, truth_normalized: torch.Tensor, ds: str) -> dict[str, float]:
    truth = author_utils.normalize(truth_normalized, method="ds", ds=ds)
    n = min(int(rec.shape[0]), int(truth.shape[0]))
    if n == 0:
        return {"evaluated_images": 0}
    rec_eval = rec[:n].detach().float().clamp(0, 1)
    truth_eval = truth[:n].detach().float().clamp(0, 1)
    mse = torch.mean((rec_eval - truth_eval) ** 2, dim=tuple(range(1, rec_eval.ndim)))
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    return {
        "evaluated_images": int(n),
        "mse_mean": float(mse.mean().cpu()),
        "psnr_mean": float(psnr.mean().cpu()),
    }


def run_author_cpa_with_update_file(
    args,
    *,
    exp_name: str,
    update_file: Path,
    method: str,
    allow_true_labels: bool = False,
) -> Path:
    """Run author CPA classes without oracle reordering in the attack path."""
    if args.fi_method != "direct":
        raise ValueError(
            "The corrected 01--03 runner supports original direct FIA only. "
            "The author GradientMatching FIA has unconstrained unknown labels."
        )
    if args.use_labels and not allow_true_labels:
        raise ValueError("--use_labels is allowed only for an explicit known-label threat model")

    data = read_pickle(update_file)
    attack_trials = int(data["metadata"]["attack_trials"])
    n_batches = min(args.attack_n_batch or attack_trials, attack_trials)
    evaluated_samples = args.n_samples if args.n_sample_fi in {-1, 0} else min(args.n_sample_fi, args.n_samples)
    attack_args, exp_path, _manifest = _prepare_exp_path(
        args,
        exp_name=exp_name,
        update_file=update_file,
        evaluated_samples=evaluated_samples,
        method=method,
        extra_manifest={
            "executed_attack_trials": n_batches,
            "total_reconstructed_samples": n_batches * evaluated_samples,
            "total_evaluated_samples": n_batches * evaluated_samples,
        },
    )
    device = author_utils.get_device()
    torch.manual_seed(int(args.seed))
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

    log_path = exp_path / "losses.jsonl"
    if getattr(args, "fresh_start", False) and log_path.exists():
        log_path.unlink()

    rec_entries = []
    eval_entries = []
    for batch in range(n_batches):
        grads = _tensor_list(data["grad"][batch], device=device)
        labels = _labels_for_attack(args.n_samples, device, data["y"][batch] if args.use_labels else None)

        gi = CocktailPartyAttack(model, grads, labels, attack_args, batch)
        gi_ckpt = exp_path / f"batch_{batch:03d}_gi.pt"
        ckpt = _load_checkpoint(gi_ckpt)
        if ckpt is not None:
            gi.set_attack_state(ckpt["state"])
            gi.start_iter = int(ckpt["iteration"])

        for iter_idx in range(gi.start_iter, attack_args.n_iter):
            loss_dict = gi.step()
            if (iter_idx % attack_args.n_log == 0) or (iter_idx == attack_args.n_iter - 1):
                _write_jsonl(log_path, {"phase": "gi", "batch": batch, "iter": iter_idx, **loss_dict})
                _save_checkpoint(gi_ckpt, iteration=iter_idx + 1, state=gi.get_attack_state())

        rec_gi = gi.get_rec().detach()
        rec_emb = rec_gi.abs() if getattr(model, "model_type", None) == "conv" else torch.empty(0, device=device)
        rec_fi = torch.empty(0, device=device)

        if getattr(model, "model_type", None) == "conv" and attack_args.n_iter_fi > 0:
            n_fi = min(attack_args.n_sample_fi, rec_emb.shape[0])
            rec_emb_fi = rec_emb[:n_fi].detach()
            fi = Direct(rec_emb_fi, model, attack_args, grads, labels)
            fi_ckpt = exp_path / f"batch_{batch:03d}_fi.pt"
            ckpt = _load_checkpoint(fi_ckpt)
            if ckpt is not None:
                fi.set_attack_state(ckpt["state"])
                fi.start_iter = int(ckpt["iteration"])

            for iter_fi in range(fi.start_iter, attack_args.n_iter_fi):
                loss_dict_fi = fi.step()
                if (iter_fi % attack_args.n_log_fi == 0) or (iter_fi == attack_args.n_iter_fi - 1):
                    _write_jsonl(log_path, {"phase": "fi", "batch": batch, "iter": iter_fi, **loss_dict_fi})
                    _save_checkpoint(fi_ckpt, iteration=iter_fi + 1, state=fi.get_attack_state())
            rec_fi = fi.get_rec().detach()

        batch_entry = {
            "batch": batch,
            "sample_indices": data["sample_indices"][batch],
            "rec_gi": rec_gi.detach().cpu().numpy(),
            "rec_emb_attack_order": rec_emb.detach().cpu().numpy(),
            "rec_fi_attack_order": rec_fi.detach().cpu().numpy(),
            "attack_order_policy": "no oracle permutation; original recovered order",
        }
        rec_entries.append(batch_entry)

        eval_entry: dict[str, Any] = {
            "batch": batch,
            "metric_note": "Oracle metrics are evaluation only and are not fed back to CPA/FIA.",
        }
        if len(data.get("z", [])) > batch and rec_emb.numel() > 0:
            _ordered, cs = oracle_permutation_aligned_cosine_similarity(
                rec_emb.detach().cpu(), data["z"][batch]
            )
            eval_entry["embedding"] = cs
        if rec_fi.numel() > 0:
            truth_x = torch.as_tensor(data["x"][batch], device=device, dtype=torch.float32)
            eval_entry["image"] = _image_eval(rec_fi, truth_x, args.ds)
        eval_entries.append(eval_entry)

    _save_pickle(exp_path / "reconstructions.pkl", rec_entries)
    write_json(exp_path / "oracle_evaluation.json", {"batches": eval_entries})
    return exp_path


def run_direct_fia_from_embeddings(
    args,
    *,
    exp_name: str,
    update_file: Path,
    recovered_embeddings: list[np.ndarray],
    method: str,
) -> Path:
    """Run author direct FIA from blind aggregated embeddings."""
    data = read_pickle(update_file)
    attack_trials = int(data["metadata"]["attack_trials"])
    n_batches = min(args.attack_n_batch or attack_trials, attack_trials)
    if len(recovered_embeddings) < n_batches:
        raise ValueError("Not enough recovered embedding batches for aggregate FIA")
    evaluated_samples = args.n_samples if args.n_sample_fi in {-1, 0} else min(args.n_sample_fi, args.n_samples)
    attack_args, exp_path, _manifest = _prepare_exp_path(
        args,
        exp_name=exp_name,
        update_file=update_file,
        evaluated_samples=evaluated_samples,
        method=method,
        extra_manifest={
            "aggregate_fia_input": "blind_ensemble_recovered_embeddings",
            "executed_attack_trials": n_batches,
            "total_reconstructed_samples": n_batches * evaluated_samples,
            "total_evaluated_samples": n_batches * evaluated_samples,
        },
    )
    device = author_utils.get_device()
    torch.manual_seed(int(args.seed))
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

    log_path = exp_path / "losses.jsonl"
    rec_entries = []
    eval_entries = []
    for batch in range(n_batches):
        grads = _tensor_list(data["grad"][batch], device=device)
        labels = _labels_for_attack(args.n_samples, device, None)
        rec_emb = torch.as_tensor(recovered_embeddings[batch], device=device, dtype=torch.float32)
        n_fi = min(attack_args.n_sample_fi, rec_emb.shape[0])
        rec_emb_fi = rec_emb[:n_fi].detach()
        fi = Direct(rec_emb_fi, model, attack_args, grads, labels)
        fi_ckpt = exp_path / f"batch_{batch:03d}_fi.pt"
        ckpt = _load_checkpoint(fi_ckpt)
        if ckpt is not None:
            fi.set_attack_state(ckpt["state"])
            fi.start_iter = int(ckpt["iteration"])
        for iter_fi in range(fi.start_iter, attack_args.n_iter_fi):
            loss_dict_fi = fi.step()
            if (iter_fi % attack_args.n_log_fi == 0) or (iter_fi == attack_args.n_iter_fi - 1):
                _write_jsonl(log_path, {"phase": "fi", "batch": batch, "iter": iter_fi, **loss_dict_fi})
                _save_checkpoint(fi_ckpt, iteration=iter_fi + 1, state=fi.get_attack_state())
        rec_fi = fi.get_rec().detach()
        rec_entries.append(
            {
                "batch": batch,
                "sample_indices": data["sample_indices"][batch],
                "rec_emb_attack_order": rec_emb.detach().cpu().numpy(),
                "rec_fi_attack_order": rec_fi.detach().cpu().numpy(),
                "attack_order_policy": "blind ensemble alignment only; no oracle permutation",
            }
        )
        truth_x = torch.as_tensor(data["x"][batch], device=device, dtype=torch.float32)
        eval_entries.append(
            {
                "batch": batch,
                "metric_note": "Oracle image metrics are evaluation only.",
                "image": _image_eval(rec_fi, truth_x, args.ds),
            }
        )
    _save_pickle(exp_path / "reconstructions.pkl", rec_entries)
    write_json(exp_path / "oracle_evaluation.json", {"batches": eval_entries})
    return exp_path
