#!/usr/bin/env python3
"""Oracle-free wrapper around the official CPA implementation for sparse updates."""

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
    assert_matching_manifest,
    build_experiment_model,
    make_attack_namespace,
    read_json,
    read_pickle,
    safe_rmtree,
    update_experiment_manifest,
    validate_parameter_mapping,
    write_json,
    write_pickle,
)
from metrics import summarize_metric_rows, write_csv

common_mod._install_optional_dependency_stubs()

import utils as author_utils  # noqa: E402
from feature_inversion import Direct  # noqa: E402
from gradient_inversion import CocktailPartyAttack  # noqa: E402
from cora_ica import CoRACocktailPartyAttack, source_hash as cora_ica_source_hash  # noqa: E402


IMAGE_METRIC_FIELDS = ("batch", "sample", "sample_index", "mse", "psnr", "ssim", "lpips", "success_lpips")
SUMMARY_FIELDS = ("metric", "mean", "std", "min", "max", "n", "threshold")


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


def _load_checkpoint(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    return torch.load(path, map_location=author_utils.get_device())


def _save_checkpoint(path: Path, *, iteration: int, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"iteration": iteration, "state": dict(state)}, path)


def _resolve_fia_stop_iter(args, attack_args) -> int:
    target = int(attack_args.n_iter_fi)
    stop_iter = getattr(args, "fia_stop_iter", None)
    if stop_iter in {None, 0}:
        return target
    stop_iter = int(stop_iter)
    if stop_iter < 1:
        raise ValueError("fia_stop_iter must be >= 1")
    return min(stop_iter, target)


RESUME_COMPATIBLE_MANIFEST_FIELDS = {"fia_micro_batch_size", "member_cpa_only"}
RESUME_COMPATIBLE_MANIFEST_METADATA_FIELDS = {
    "resume_compatible_overrides",
    "fia_micro_batch_size_resume_note",
}


def _manifest_without_resume_compatible_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    filtered = dict(value)
    for key in RESUME_COMPATIBLE_MANIFEST_FIELDS | RESUME_COMPATIBLE_MANIFEST_METADATA_FIELDS:
        filtered.pop(key, None)
    return filtered


def _prepare_resume_compatible_manifest(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(expected)
    current = read_json(path)
    if current == expected:
        return dict(expected)
    if _manifest_without_resume_compatible_fields(current) != _manifest_without_resume_compatible_fields(expected):
        assert_matching_manifest(path, expected)
    changed = {
        key: {"from": current.get(key), "to": expected.get(key)}
        for key in RESUME_COMPATIBLE_MANIFEST_FIELDS
        if current.get(key) != expected.get(key)
    }
    if not changed:
        return current
    merged = dict(expected)
    history = list(current.get("resume_compatible_overrides", []))
    history.append(
        {
            "changed_fields": changed,
            "reason": (
                "Resume from existing CPA/FIA checkpoints when only resume-compatible "
                "fields change. member_cpa_only may change when Direct FIA is added "
                "after a completed CPA-only member; fia_micro_batch_size controls only "
                "memory chunking. Optimization objective, samples, seeds, and iteration "
                "counts are unchanged."
            ),
        }
    )
    merged["resume_compatible_overrides"] = history
    merged["fia_micro_batch_size_resume_note"] = (
        "Some earlier FIA checkpoint iterations in this run may have used a previous "
        "fia_micro_batch_size; this field controls only memory chunking during Direct FIA."
    )
    return merged


class MicroBatchDirect(Direct):
    """Direct FIA with memory-only chunked model forwards."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        chunk_size = getattr(self.args, "fia_micro_batch_size", None)
        self.fia_micro_batch_size = self.n if chunk_size is None else int(chunk_size)
        if self.fia_micro_batch_size < 1:
            raise ValueError("fia_micro_batch_size must be >= 1")
        self.fia_micro_batch_size = min(self.fia_micro_batch_size, self.n)

    def step(self):
        loss_gm = torch.tensor(0.0, device=self.device)
        loss_fi_value = torch.tensor(0.0, device=self.device)
        loss_tv_value = torch.tensor(0.0, device=self.device)

        self.S_hat.data = torch.max(
            torch.min(self.S_hat, (1 - self.mean) / self.std),
            -self.mean / self.std,
        )
        self.opt.zero_grad()

        if self.fi > 0:
            for start in range(0, self.n, self.fia_micro_batch_size):
                end = min(start + self.fia_micro_batch_size, self.n)
                _, z_hat = self.model(self.S_hat[start:end], return_z=True)
                chunk_loss_fi = (1 - self.cosine_similarity(self.Z[start:end], z_hat)).mean()
                weight = (end - start) / self.n
                (self.fi * chunk_loss_fi * weight).backward()
                loss_fi_value = loss_fi_value + chunk_loss_fi.detach() * weight

        if self.tv > 0:
            loss_tv = self.total_variation(self.S_hat)
            (self.tv * loss_tv).backward()
            loss_tv_value = loss_tv.detach()

        self.opt.step()
        if self.sch:
            self.sch.step()

        loss_value = self.fi * loss_fi_value + self.tv * loss_tv_value
        return {
            "loss": loss_value.detach().cpu().item(),
            "loss_fi": loss_fi_value.detach().cpu().item(),
            "loss_gm": loss_gm.detach().cpu().item(),
            "loss_tv": loss_tv_value.detach().cpu().item(),
        }


def _make_direct_fia(z: torch.Tensor, model, attack_args, grads, labels):
    chunk_size = getattr(attack_args, "fia_micro_batch_size", None)
    if chunk_size is not None and int(chunk_size) < int(z.shape[0]):
        return MicroBatchDirect(z, model, attack_args, grads, labels)
    return Direct(z, model, attack_args, grads, labels)


class ImageMetricComputer:
    def __init__(self, device: torch.device, *, success_lpips_threshold: float) -> None:
        self.device = device
        self.success_lpips_threshold = float(success_lpips_threshold)
        self.ssim_fn = None
        self.ssim_error: str | None = None
        self.lpips_model = None
        self.lpips_error: str | None = None
        try:
            from torchmetrics.functional import structural_similarity_index_measure

            self.ssim_fn = structural_similarity_index_measure
        except Exception as exc:  # pragma: no cover - depends on local optional deps
            self.ssim_error = str(exc)
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            self.lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
            self.lpips_model.eval()
        except Exception as exc:  # pragma: no cover - depends on local optional deps
            self.lpips_error = str(exc)

    def dependency_status(self) -> dict[str, Any]:
        return {
            "ssim_available": self.ssim_fn is not None,
            "ssim_error": self.ssim_error,
            "lpips_available": self.lpips_model is not None,
            "lpips_error": self.lpips_error,
            "success_lpips_threshold": self.success_lpips_threshold,
        }

    def compute_batch(
        self,
        *,
        batch: int,
        rec: torch.Tensor,
        truth_normalized: torch.Tensor,
        ds: str,
        sample_indices: list[int],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        truth = author_utils.normalize(truth_normalized, method="ds", ds=ds)
        n = min(int(rec.shape[0]), int(truth.shape[0]))
        rec_eval = rec[:n].detach().float().clamp(0, 1)
        truth_eval = truth[:n].detach().float().clamp(0, 1)
        rows: list[dict[str, Any]] = []
        psnr_values: list[float] = []
        ssim_values: list[float] = []
        lpips_values: list[float] = []

        with torch.no_grad():
            for sample in range(n):
                rec_b = rec_eval[sample : sample + 1]
                truth_b = truth_eval[sample : sample + 1]
                mse = torch.mean((rec_b - truth_b) ** 2)
                psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
                row: dict[str, Any] = {
                    "batch": batch,
                    "sample": sample,
                    "sample_index": sample_indices[sample] if sample < len(sample_indices) else None,
                    "mse": float(mse.detach().cpu()),
                    "psnr": float(psnr.detach().cpu()),
                    "ssim": None,
                    "lpips": None,
                    "success_lpips": None,
                }
                psnr_values.append(float(row["psnr"]))
                if self.ssim_fn is not None:
                    try:
                        value = float(self.ssim_fn(rec_b, truth_b).detach().cpu().item())
                        row["ssim"] = value
                        ssim_values.append(value)
                    except Exception as exc:  # pragma: no cover - data/dependency specific
                        self.ssim_error = str(exc)
                        self.ssim_fn = None
                if self.lpips_model is not None:
                    try:
                        value = float(self.lpips_model(rec_b * 2.0 - 1.0, truth_b * 2.0 - 1.0).detach().cpu().item())
                        row["lpips"] = value
                        row["success_lpips"] = bool(value <= self.success_lpips_threshold)
                        lpips_values.append(value)
                    except Exception as exc:  # pragma: no cover - data/dependency specific
                        self.lpips_error = str(exc)
                        self.lpips_model = None
                rows.append(row)

        summary = {
            "evaluated_images": int(n),
            "mse_mean": _mean_or_none([float(row["mse"]) for row in rows]),
            "psnr_mean": _mean_or_none(psnr_values),
            "ssim_mean": _mean_or_none(ssim_values),
            "lpips_mean": _mean_or_none(lpips_values),
            "success_lpips_threshold": self.success_lpips_threshold,
            "reconstruction_success_rate": (
                float(sum(value <= self.success_lpips_threshold for value in lpips_values) / len(lpips_values))
                if lpips_values
                else None
            ),
        }
        return rows, summary


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _set_author_output_root() -> None:
    AUTHOR_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    author_utils.exp_path_base = str(AUTHOR_RUN_ROOT)


def _prepare_exp_path(
    args,
    *,
    exp_name: str,
    update_file: Path,
    update_data: Optional[Mapping[str, Any]] = None,
    evaluated_samples: int,
    method: str,
    extra_manifest: Optional[Mapping[str, Any]] = None,
) -> tuple[Any, Path, dict[str, Any]]:
    _set_author_output_root()
    attack_args = make_attack_namespace(args, exp_name=exp_name)
    extra = {
        "source_runner": str(Path(__file__).resolve()),
        "source_runner_import": __name__,
        "author_gradient_inversion_import": CocktailPartyAttack.__module__,
        "author_feature_inversion_import": Direct.__module__,
        "oracle_removed_from_attack_pipeline": True,
        "attack_pipeline_ground_truth_inputs": "none; labels only when an explicit known-label threat model is enabled",
        "observed_zero_policy": getattr(args, "observed_zero_policy", None),
        "success_lpips_threshold": getattr(args, "success_lpips_threshold", 0.5),
    }
    if getattr(attack_args, "fia_micro_batch_size", None) is not None:
        extra["fia_micro_batch_size"] = int(attack_args.fia_micro_batch_size)
        extra["feature_inversion_runner"] = "Direct FIA with memory-only chunked model forwards"
    if extra_manifest:
        extra.update(dict(extra_manifest))
    manifest = update_experiment_manifest(
        experiment_id=exp_name,
        method=method,
        update_file=update_file,
        args=args,
        evaluated_samples=evaluated_samples,
        extra=extra,
        update_data=update_data,
    )
    exp_path = Path(author_utils.get_attack_exp_path(attack_args))
    manifest_file = exp_path / "manifest.json"
    if getattr(args, "fresh_start", False):
        safe_rmtree(exp_path, allowed_root=AUTHOR_RUN_ROOT)
    elif exp_path.exists():
        manifest = _prepare_resume_compatible_manifest(manifest_file, manifest)
    exp_path.mkdir(parents=True, exist_ok=True)
    write_json(manifest_file, manifest)
    return attack_args, exp_path, manifest


def _tensor_list(values: list[Any], device: torch.device) -> list[torch.Tensor]:
    return [torch.as_tensor(value, device=device, dtype=torch.float32) for value in values]


def _labels_for_attack(n_samples: int, device: torch.device, true_labels: Optional[Any] = None) -> torch.Tensor:
    if true_labels is None:
        return torch.zeros(n_samples, dtype=torch.long, device=device)
    return torch.as_tensor(true_labels, dtype=torch.long, device=device)


def _write_metric_outputs(
    exp_path: Path,
    *,
    image_rows: list[dict[str, Any]],
    eval_entries: list[dict[str, Any]],
    success_lpips_threshold: float,
) -> list[dict[str, Any]]:
    write_csv(exp_path / "image_metrics_by_sample.csv", image_rows, fieldnames=IMAGE_METRIC_FIELDS)
    summary_rows = summarize_metric_rows(
        image_rows=image_rows,
        eval_entries=eval_entries,
        success_lpips_threshold=success_lpips_threshold,
    )
    write_csv(exp_path / "metric_summary.csv", summary_rows, fieldnames=SUMMARY_FIELDS)
    return summary_rows


def _write_whitening_pre_stats(exp_path: Path, data: Mapping[str, Any]) -> None:
    if data.get("whitening_pre_stats"):
        write_json(
            exp_path / "whitening_pre_stats.json",
            {
                "note": "Pre-whitening diagnostics for the attack FC layer only.",
                "observation_model": data.get("metadata", {}).get("attack_fc_observation_model"),
                "az_scope": data.get("metadata", {}).get("az_scope"),
                "batches": data["whitening_pre_stats"],
            },
        )


def run_author_cpa_with_update_file(
    args,
    *,
    exp_name: str,
    update_file: Path,
    method: str,
    allow_true_labels: bool = False,
    update_data: Optional[Mapping[str, Any]] = None,
    run_fia: bool = True,
    enable_negentropy: bool = False,
    extra_manifest: Optional[Mapping[str, Any]] = None,
    attack_order_policy: str = "no oracle permutation; sparse zeros are treated as observed values",
) -> Path:
    """Run CPA without oracle reordering; CoRA explicitly restores negentropy."""
    if args.fi_method != "direct":
        raise ValueError(
            "The sparse runner supports original direct FIA only. "
            "The author GradientMatching FIA has unconstrained unknown labels."
        )
    if args.use_labels and not allow_true_labels:
        raise ValueError("--use_labels is allowed only for an explicit known-label threat model")

    data = read_pickle(update_file) if update_data is None else update_data
    attack_trials = int(data["metadata"]["attack_trials"])
    n_batches = min(args.attack_n_batch or attack_trials, attack_trials)
    evaluated_samples = args.n_samples if args.n_sample_fi in {-1, 0} else min(args.n_sample_fi, args.n_samples)
    success_lpips_threshold = float(getattr(args, "success_lpips_threshold", 0.5))
    manifest_updates = {
        "executed_attack_trials": n_batches,
        "total_reconstructed_samples": n_batches * evaluated_samples,
        "total_evaluated_samples": n_batches * evaluated_samples,
        "member_cpa_only": not bool(run_fia),
    }
    if data.get("whitening_pre_stats"):
        manifest_updates["whitening_pre_stats_file"] = "whitening_pre_stats.json"
        manifest_updates["attack_fc_observation_model"] = data["metadata"].get("attack_fc_observation_model")
        manifest_updates["az_scope"] = data["metadata"].get("az_scope")
    if extra_manifest:
        manifest_updates.update(dict(extra_manifest))
    cpa_class = CoRACocktailPartyAttack if enable_negentropy else CocktailPartyAttack
    if enable_negentropy:
        manifest_updates.update({
            "negentropy_enabled": float(args.ne) > 0,
            "negentropy_behavior": "upstream_logcosh_contrast_without_zero_overwrite",
            "cora_ica_source_sha256": cora_ica_source_hash(),
        })
    attack_args, exp_path, _manifest = _prepare_exp_path(
        args,
        exp_name=exp_name,
        update_file=update_file,
        update_data=data,
        evaluated_samples=evaluated_samples,
        method=method,
        extra_manifest=manifest_updates,
    )
    _write_whitening_pre_stats(exp_path, data)
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
    metric_computer = (
        ImageMetricComputer(device, success_lpips_threshold=success_lpips_threshold) if run_fia else None
    )

    log_path = exp_path / "losses.jsonl"
    if getattr(args, "fresh_start", False) and log_path.exists():
        log_path.unlink()
    fia_stop_iter = _resolve_fia_stop_iter(args, attack_args)

    print(
        f"[attack] start exp={exp_path.name} batches={n_batches} "
        f"cpa_iter={attack_args.n_iter} fia_iter={attack_args.n_iter_fi} "
        f"fia_stop_iter={fia_stop_iter}",
        flush=True,
    )
    rec_entries: list[dict[str, Any]] = []
    eval_entries: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    for batch in range(n_batches):
        print(f"[attack] exp={exp_path.name} batch {batch + 1}/{n_batches} CPA start", flush=True)
        grads = _tensor_list(data["grad"][batch], device=device)
        labels = _labels_for_attack(args.n_samples, device, data["y"][batch] if args.use_labels else None)

        gi = cpa_class(model, grads, labels, attack_args, batch)
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
                print(
                    f"[attack] exp={exp_path.name} batch {batch + 1}/{n_batches} "
                    f"CPA iter {iter_idx + 1}/{attack_args.n_iter} loss={loss_dict.get('loss')}",
                    flush=True,
                )

        rec_gi = gi.get_rec().detach()
        rec_emb = rec_gi.abs() if getattr(model, "model_type", None) == "conv" else torch.empty(0, device=device)
        rec_fi = torch.empty(0, device=device)

        if run_fia and getattr(model, "model_type", None) == "conv" and attack_args.n_iter_fi > 0:
            n_fi = min(attack_args.n_sample_fi, rec_emb.shape[0])
            rec_emb_fi = rec_emb[:n_fi].detach()
            fi = _make_direct_fia(rec_emb_fi, model, attack_args, grads, labels)
            print(
                f"[attack] exp={exp_path.name} batch {batch + 1}/{n_batches} "
                f"Direct-FIA start samples={n_fi}",
                flush=True,
            )
            fi_ckpt = exp_path / f"batch_{batch:03d}_fi.pt"
            ckpt = _load_checkpoint(fi_ckpt)
            if ckpt is not None:
                fi.set_attack_state(ckpt["state"])
                fi.start_iter = int(ckpt["iteration"])

            for iter_fi in range(fi.start_iter, fia_stop_iter):
                loss_dict_fi = fi.step()
                if (iter_fi % attack_args.n_log_fi == 0) or (iter_fi == fia_stop_iter - 1):
                    _write_jsonl(log_path, {"phase": "fi", "batch": batch, "iter": iter_fi, **loss_dict_fi})
                    _save_checkpoint(fi_ckpt, iteration=iter_fi + 1, state=fi.get_attack_state())
                    print(
                        f"[attack] exp={exp_path.name} batch {batch + 1}/{n_batches} "
                        f"Direct-FIA iter {iter_fi + 1}/{attack_args.n_iter_fi} "
                        f"loss={loss_dict_fi.get('loss')}",
                        flush=True,
                    )
            rec_fi = fi.get_rec().detach()

        rec_entries.append(
            {
                "batch": batch,
                "sample_indices": data["sample_indices"][batch],
                "rec_gi": rec_gi.detach().cpu().numpy(),
                "rec_emb_attack_order": rec_emb.detach().cpu().numpy(),
                "rec_fi_attack_order": rec_fi.detach().cpu().numpy(),
                "attack_order_policy": attack_order_policy,
            }
        )

        eval_entry: dict[str, Any] = {
            "batch": batch,
            "metric_note": "Oracle metrics are evaluation only and are not fed back to CPA/FIA.",
        }
        if len(data.get("z", [])) > batch and rec_emb.numel() > 0:
            _ordered, cs = oracle_permutation_aligned_cosine_similarity(
                rec_emb.detach().cpu(), data["z"][batch]
            )
            eval_entry["embedding"] = cs
        if metric_computer is not None and rec_fi.numel() > 0:
            truth_x = torch.as_tensor(data["x"][batch], device=device, dtype=torch.float32)
            rows, summary = metric_computer.compute_batch(
                batch=batch,
                rec=rec_fi,
                truth_normalized=truth_x,
                ds=args.ds,
                sample_indices=list(data["sample_indices"][batch]),
            )
            image_rows.extend(rows)
            eval_entry["image"] = summary
        eval_entries.append(eval_entry)

    write_pickle(exp_path / "reconstructions.pkl", rec_entries)
    summary_rows = _write_metric_outputs(
        exp_path,
        image_rows=image_rows,
        eval_entries=eval_entries,
        success_lpips_threshold=success_lpips_threshold,
    )
    write_json(
        exp_path / "oracle_evaluation.json",
        {
            "batches": eval_entries,
            "image_metric_dependencies": (
                metric_computer.dependency_status()
                if metric_computer is not None
                else {"skipped": True, "reason": "member CPA-only run"}
            ),
            "metric_summary": summary_rows,
        },
    )
    return exp_path


def run_direct_fia_from_embeddings(
    args,
    *,
    exp_name: str,
    update_file: Path,
    recovered_embeddings: list[np.ndarray],
    method: str,
    extra_manifest: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Run official Direct FIA from externally recovered embeddings."""
    data = read_pickle(update_file)
    attack_trials = int(data["metadata"]["attack_trials"])
    n_batches = min(args.attack_n_batch or attack_trials, attack_trials)
    if len(recovered_embeddings) < n_batches:
        raise ValueError("Not enough recovered embedding batches for aggregate FIA")
    evaluated_samples = args.n_samples if args.n_sample_fi in {-1, 0} else min(args.n_sample_fi, args.n_samples)
    success_lpips_threshold = float(getattr(args, "success_lpips_threshold", 0.5))
    manifest_updates = {
        "aggregate_fia_input": "external_recovered_embeddings",
        "executed_attack_trials": n_batches,
        "total_reconstructed_samples": n_batches * evaluated_samples,
        "total_evaluated_samples": n_batches * evaluated_samples,
    }
    if data.get("whitening_pre_stats"):
        manifest_updates["whitening_pre_stats_file"] = "whitening_pre_stats.json"
        manifest_updates["attack_fc_observation_model"] = data["metadata"].get("attack_fc_observation_model")
        manifest_updates["az_scope"] = data["metadata"].get("az_scope")
    if extra_manifest:
        manifest_updates.update(dict(extra_manifest))
    attack_args, exp_path, _manifest = _prepare_exp_path(
        args,
        exp_name=exp_name,
        update_file=update_file,
        update_data=data,
        evaluated_samples=evaluated_samples,
        method=method,
        extra_manifest=manifest_updates,
    )
    _write_whitening_pre_stats(exp_path, data)
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
    metric_computer = ImageMetricComputer(device, success_lpips_threshold=success_lpips_threshold)

    log_path = exp_path / "losses.jsonl"
    print(
        f"[aggregate-fia] start exp={exp_path.name} batches={n_batches} "
        f"fia_iter={attack_args.n_iter_fi}",
        flush=True,
    )
    rec_entries: list[dict[str, Any]] = []
    eval_entries: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    for batch in range(n_batches):
        print(f"[aggregate-fia] exp={exp_path.name} batch {batch + 1}/{n_batches} start", flush=True)
        grads = _tensor_list(data["grad"][batch], device=device)
        labels = _labels_for_attack(args.n_samples, device, None)
        rec_emb = torch.as_tensor(recovered_embeddings[batch], device=device, dtype=torch.float32)
        n_fi = min(attack_args.n_sample_fi, rec_emb.shape[0])
        rec_emb_fi = rec_emb[:n_fi].detach()
        fi = _make_direct_fia(rec_emb_fi, model, attack_args, grads, labels)
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
                print(
                    f"[aggregate-fia] exp={exp_path.name} batch {batch + 1}/{n_batches} "
                    f"iter {iter_fi + 1}/{attack_args.n_iter_fi} loss={loss_dict_fi.get('loss')}",
                    flush=True,
                )
        rec_fi = fi.get_rec().detach()
        rec_entries.append(
            {
                "batch": batch,
                "sample_indices": data["sample_indices"][batch],
                "rec_emb_attack_order": rec_emb.detach().cpu().numpy(),
                "rec_fi_attack_order": rec_fi.detach().cpu().numpy(),
                "attack_order_policy": "external embedding order; no oracle permutation in FIA input",
            }
        )
        truth_x = torch.as_tensor(data["x"][batch], device=device, dtype=torch.float32)
        rows, summary = metric_computer.compute_batch(
            batch=batch,
            rec=rec_fi,
            truth_normalized=truth_x,
            ds=args.ds,
            sample_indices=list(data["sample_indices"][batch]),
        )
        image_rows.extend(rows)
        eval_entry: dict[str, Any] = {
            "batch": batch,
            "metric_note": "Oracle metrics are evaluation only.",
            "image": summary,
        }
        if len(data.get("z", [])) > batch and rec_emb.numel() > 0:
            _ordered, cs = oracle_permutation_aligned_cosine_similarity(
                rec_emb.detach().cpu(), data["z"][batch]
            )
            eval_entry["embedding"] = cs
        eval_entries.append(eval_entry)

    write_pickle(exp_path / "reconstructions.pkl", rec_entries)
    summary_rows = _write_metric_outputs(
        exp_path,
        image_rows=image_rows,
        eval_entries=eval_entries,
        success_lpips_threshold=success_lpips_threshold,
    )
    write_json(
        exp_path / "oracle_evaluation.json",
        {
            "batches": eval_entries,
            "image_metric_dependencies": metric_computer.dependency_status(),
            "metric_summary": summary_rows,
        },
    )
    return exp_path
