"""Locked M6 UIA fp32, B=16, T=10 CPA + unchanged 25k FIA runner.

Only the actual observed weight difference enters CPA. Ideal B_T @ Z is saved
for update diagnostics. The prepare phase audits existing CPU caches only;
the run phase requires completed M7 artifacts before any GPU work.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .alignment import match_sources_by_abs_cosine
from .fia import denormalize_imagenet, run_direct_fia
from .seed import seed_all
from .stage1 import estimate_mixing_and_sources
from .uia import UIAFeatureBatch, load_or_collect_uia_feature_batch, run_fc_only_fedavg_on_features
from .victim import VictimBatch, load_imagenet_dataset, select_label_indices


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / "configs/m6_uia_fp32_B16_fia.json"
RUN_TAG = "m6_uia_fp32_B16_fia"
N = 16
BATCHES = list(range(5))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Undefined zero-noise MP ratios are null, never JSON NaN/Infinity."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty measurement CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def read_locked_config(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    lock_path = Path(path).with_suffix(".lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    if file_hash(path) != lock["sha256"] or config["protocol_id"] != lock["protocol_id"]:
        raise RuntimeError("UIA fp32 protocol changed after locking")
    validate_config(config)
    return config, lock


def validate_config(cfg: dict[str, Any]) -> None:
    checks = {
        "run tag": cfg["run_tag"] == RUN_TAG,
        "victim seed": cfg["seed"] == 0,
        "80 originals in five B16 batches": cfg["data"]["batch_size"] == N and cfg["data"]["batch_indices"] == BATCHES
            and cfg["data"]["n_images"] == 80 and cfg["data"]["require_distinct_labels"] is True,
        "round0 frozen VGG16": cfg["model"]["name"] == "vgg16" and cfg["model"]["weights"] == "VGG16_Weights.IMAGENET1K_V1"
            and cfg["model"]["attacked_checkpoint_round"] == 0 and cfg["model"]["conv_frozen"] is True,
        "no quantizer": cfg["quantization"]["bits"] == "fp32" and cfg["quantization"]["enabled"] is False
            and cfg["quantization"]["delta"] == 0 and cfg["quantization"]["scale_granularity"] == "per_row",
        "actual observed T10 SGD": cfg["uia"]["steps"] == 10 and cfg["uia"]["lr"] == .001
            and cfg["uia"]["optimizer"] == "sgd" and cfg["uia"]["momentum"] == 0 and cfg["uia"]["weight_decay"] == 0
            and cfg["uia"]["observation"] == "Y_delta_over_lr" and cfg["uia"]["uses_Y_factor_as_attack_input"] is False
            and cfg["uia"]["effective_minibatch_size"] == N,
        "full16 unchanged CPA": cfg["stage1"]["n_components"] == N and cfg["stage1"]["ica_method"] == "cpa"
            and cfg["stage1"]["debias"] is False and cfg["stage1"]["mp_truncation"] is False,
        "unchanged FIA": all(cfg["fia"][key] == value for key, value in
            {"n_iter": 25000, "lr": .1, "tv": .1, "fi": 1., "optimizer": "adam", "scheduler": "cosine", "seed": 42, "chunk_size": N}.items()),
        "no future 4bit": cfg["decision"]["automatic_4bit"] is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Unsupported UIA fp32 protocol changes: {failed}")


def check_dependency(cfg: dict[str, Any]) -> None:
    dependency = cfg["dependency"]
    execution_path, report_path = resolve_path(dependency["execution_status"]), resolve_path(dependency["report_summary"])
    if not execution_path.is_file() or not report_path.is_file():
        raise RuntimeError("M7 completion artifacts are required before UIA GPU work")
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (execution.get("state") != dependency["required_execution_state"]
            or report.get("report_status") != dependency["required_report_status"]
            or report.get("preregistration_sha256") != dependency["preregistration_sha256"]):
        raise RuntimeError("M7 must complete its locked report before UIA GPU work")


def implementation_identity(lock: dict[str, Any]) -> dict[str, Any]:
    names = ("run_m6_uia_fp32_fia.py", "uia.py", "victim.py", "stage1.py", "fia.py", "alignment.py", "seed.py")
    weights = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
    if not weights.is_file():
        raise RuntimeError(f"The established VGG16 weights must already exist: {weights}")
    return {"protocol_sha256": lock["sha256"], "code_sha256": {name: file_hash(PROJECT / "qacpa" / name) for name in names},
            "weights_path": str(weights), "weights_sha256": file_hash(weights)}


def save_checkpoint(folder: Path, name: str, payload: dict[str, Any], identity: dict[str, Any]) -> None:
    path = folder / f"{name}.pt"
    temporary = path.with_suffix(".pt.tmp")
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temporary)
    temporary.replace(path)
    write_json(folder / f"{name}.meta.json", {"identity": identity, "file_sha256": file_hash(path), "saved_at_utc": utc_now()})


def load_checkpoint(folder: Path, name: str, identity: dict[str, Any]) -> dict[str, Any] | None:
    marker = folder / f"{name}.meta.json"
    if not marker.is_file():
        return None
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    path = folder / f"{name}.pt"
    if metadata.get("identity") != identity or not path.is_file() or file_hash(path) != metadata["file_sha256"]:
        raise RuntimeError(f"Checkpoint protocol/input/code/integrity mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def audit_cached_batch(cfg: dict[str, Any], batch: int, dataset: Any) -> dict[str, Any]:
    """Read existing artifacts and original transforms; never create a model/cache."""
    started = time.perf_counter()
    cache = resolve_path(cfg["paths"]["cache_dir"])
    victim_path, feature_path = cache / f"victim_seed0_B16_batch{batch}.pt", cache / f"uia_features_seed0_n16_batch{batch}.pt"
    if not victim_path.is_file() or not feature_path.is_file():
        raise RuntimeError(f"CPU prepare requires both existing victim and UIA caches for batch {batch}")
    victim = VictimBatch.from_cache_dict(torch.load(victim_path, map_location="cpu", weights_only=False))
    # Existing-file checks above make the legacy loader strictly read-only here.
    legacy_cfg = {**cfg, "paths": {**cfg["paths"], "cache_dir": str(cache)}}
    feature = load_or_collect_uia_feature_batch(legacy_cfg, n=N, batch_index=batch)
    root = resolve_path(cfg["data"]["root"])
    for source in (victim, feature):
        if (source.seed != cfg["seed"] or source.batch_index != batch or source.split != cfg["data"]["split"]
                or Path(source.data_root).resolve() != root or len(source.labels) != N):
            raise RuntimeError(f"Legacy cache identity mismatch for batch {batch}")
    if (feature.n != N or tuple(victim.x.shape) != (N, 3, 224, 224) or tuple(feature.Z.shape) != (N, 25088)
            or len(set(victim.labels.tolist())) != N):
        raise RuntimeError("B16 source/image/label shape gate failed")
    for left, right in ((feature.Z, victim.S_true), (feature.labels, victim.labels), (feature.gia_A, victim.A_true),
                        (feature.gia_Y, victim.Y), (feature.initial_logits, victim.logits),
                        (feature.initial_per_sample_loss, victim.per_sample_loss)):
        if not torch.equal(left, right):
            raise RuntimeError(f"UIA feature cache differs from its victim cache in batch {batch}")
    indices = select_label_indices(dataset, B=N, seed=cfg["seed"], batch_index=batch, require_distinct_labels=True)
    samples = [dataset[index] for index in indices]
    current_x = torch.stack([sample[0] for sample in samples])
    current_labels = torch.tensor([int(sample[1]) for sample in samples])
    current_paths = [str(Path(dataset.samples[index][0]).resolve()) for index in indices]
    if (not torch.equal(current_x, victim.x) or not torch.equal(current_labels, victim.labels)
            or current_paths != [str(Path(path).resolve()) for path in victim.image_paths]
            or feature.image_paths != victim.image_paths):
        raise RuntimeError(f"Current selected originals differ from cached batch {batch}")
    if not bool(torch.isfinite(feature.Z).all()) or not bool(torch.isfinite(victim.x).all()):
        raise RuntimeError("Nonfinite cached images/features")
    metadata = [{"global_index": batch * N + i, "batch_index": batch, "source_index": i,
                 "dataset_index": indices[i], "label": int(victim.labels[i]), "image_path": current_paths[i],
                 "initial_per_sample_loss": float(victim.per_sample_loss[i])} for i in range(N)]
    return {"x": victim.x.contiguous(), "Z_evaluation_only": feature.Z.contiguous(), "metadata": metadata,
            "feature_cache_path": str(feature_path), "feature_cache_sha256": file_hash(feature_path),
            "victim_cache_path": str(victim_path), "victim_cache_sha256": file_hash(victim_path),
            "wall_time_sec": time.perf_counter() - started}


def prepare(cfg: dict[str, Any], identity: dict[str, Any], output: Path) -> dict[str, Any]:
    dataset = None
    files, image_paths = [], []
    for batch in BATCHES:
        folder = output / f"batch{batch:02d}"
        batch_identity = {**identity, "batch_index": batch, "stage": "cached_input_audit"}
        inputs = load_checkpoint(folder, "inputs", batch_identity)
        if inputs is None:
            if dataset is None:
                dataset = load_imagenet_dataset(resolve_path(cfg["data"]["root"]), cfg["data"]["split"])
            inputs = audit_cached_batch(cfg, batch, dataset)
            save_checkpoint(folder, "inputs", inputs, batch_identity)
        for kind in ("feature", "victim"):
            if file_hash(Path(inputs[f"{kind}_cache_path"])) != inputs[f"{kind}_cache_sha256"]:
                raise RuntimeError(f"Underlying {kind} cache changed after input registration")
        image_paths.extend(row["image_path"] for row in inputs["metadata"])
        files.append({"batch_index": batch, "path": str(folder / "inputs.pt"), "sha256": file_hash(folder / "inputs.pt"),
                      "feature_cache_path": inputs["feature_cache_path"], "feature_cache_sha256": inputs["feature_cache_sha256"],
                      "victim_cache_path": inputs["victim_cache_path"], "victim_cache_sha256": inputs["victim_cache_sha256"]})
        print(f"CPU cache audit complete: batch={batch}, n={N}", flush=True)
    if len(image_paths) != 80 or len(set(image_paths)) != 80:
        raise RuntimeError("The locked five batches must contain 80 unique originals")
    manifest = {"run_tag": RUN_TAG, "identity": identity, "n_images": 80, "prepared_files": files,
                "prepare_phase": "CPU existing-cache and original-transform audit only; no observation collection, ICA, or FIA",
                "prepared_at_utc": utc_now()}
    write_json(output / "input_manifest.json", manifest)
    return manifest


def actual_observation(observation: Any, *, maximum_relative_error: float) -> torch.Tensor:
    """Never repair/substitute the ideal factorization for a transmitted update."""
    values = observation if isinstance(observation, dict) else vars(observation)
    actual = values["Y_delta_over_lr"]
    if tuple(actual.shape) != (4096, 25088) or not bool(torch.isfinite(actual).all()):
        raise RuntimeError("Actual observed update shape/finiteness gate failed")
    error = float(values["update_factor_rel_l2"])
    if not math.isfinite(error) or error > maximum_relative_error or error < 0:
        raise RuntimeError(f"Actual-update/factor residual gate failed: {error} > {maximum_relative_error}")
    return actual


def rank_diagnostics(eigenvalues: list[float], *, dimension: int, rel_tol: float, abs_tol: float) -> dict[str, Any]:
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.shape != (N,) or not np.isfinite(values).all():
        raise ValueError("Rank diagnostics require all 16 finite observed centered covariance eigenvalues")
    singular = np.sqrt(dimension * np.maximum(values, 0))
    tolerance = max(abs_tol, rel_tol * float(singular.max()))
    return {"predicted_B_eff": int(np.count_nonzero(values > 0)), "nominal_zero_noise_MP_ceiling": N,
            "sigma2": 0., "mp_edge": 0., "mp_threshold": 0., "lam_over_mp_edge": [None] * N,
            "numerical_rank_top16_observed_centered": int(np.count_nonzero(singular > tolerance)),
            "numerical_rank_tolerance": tolerance, "observed_centered_top16_singular_values": singular.tolist(),
            "observed_centered_top16_eigenvalues": values.tolist(),
            "rank_scope": "top16 only; diagnostic, never an ICA truncation rule"}


def fia_targets(sources: torch.Tensor, truth: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """M3 abs adapter and permutation only; no oracle scales or missing-source fill."""
    if sources.shape != truth.shape or sources.ndim != 2 or sources.shape[0] != N:
        raise RuntimeError("Full 16 estimated sources are required; oracle/reduced-rank fallback is prohibited")
    if not bool(torch.isfinite(sources).all()):
        raise RuntimeError("Nonfinite estimated source")
    alignment = match_sources_by_abs_cosine(truth, sources)
    if alignment.true_indices != list(range(N)) or sorted(alignment.estimated_indices) != list(range(N)):
        raise RuntimeError("Incomplete full16 Hungarian correspondence; no fill is permitted")
    order = torch.tensor(alignment.estimated_indices, dtype=torch.long)
    targets = sources.detach().cpu().index_select(0, order).abs().contiguous()
    adapted_cosines = F.cosine_similarity(truth.cpu(), targets, dim=1, eps=1e-12)
    source_rows = [{"source_index": index, "estimated_source_index": alignment.estimated_indices[index],
                    "source_abs_cosine": float(alignment.per_source_abs_cosine[index]),
                    "fia_abs_target_cosine": float(adapted_cosines[index]),
                    "measured_recovered_cos09": int(alignment.per_source_abs_cosine[index] > .9),
                    "missing_source": False, "oracle_fill": False} for index in range(N)]
    return targets, source_rows


def stage1_call(Y: torch.Tensor, cfg: dict[str, Any], batch: int, device: torch.device) -> Any:
    stage = cfg["stage1"]
    seed_all(int(cfg["seed"]) + batch)
    return estimate_mixing_and_sources(
        Y, torch.zeros(Y.shape[0], dtype=torch.float32), n_components=N, rounding="fp32", debias=False,
        seed=int(cfg["seed"]) + batch, max_iter=1000, tol=1e-4, mp_truncation=False, ica_method="cpa",
        c_mp=float(stage["c_mp"]), whitening_tolerance=float(stage["whitening_tolerance"]), device=device,
        **{key: stage[key] for key in ("cpa_ica_max_iter", "cpa_ica_lr", "cpa_ica_optimizer", "cpa_ica_decor",
                                      "cpa_ica_T", "cpa_ica_ne", "cpa_ica_nv", "cpa_ica_l1", "cpa_ica_tv", "cpa_ica_log_every")})


def decision_label(mean_lpips: float, cfg: dict[str, Any]) -> dict[str, Any]:
    if not math.isfinite(mean_lpips):
        raise ValueError("A finite, unrounded mean LPIPS is required")
    decision = cfg["decision"]
    for name in ("reference_range_met", "protocol_difference_review", "pipeline_review"):
        band = decision[name]
        if mean_lpips >= band["minimum_inclusive"] and mean_lpips <= band.get("maximum_inclusive", math.inf):
            return {"classification": name, "action": band["action"], "unrounded_mean_lpips": mean_lpips,
                    "automatic_4bit": False}
    return {"classification": "unclassified_review", "action": decision["other_values"], "unrounded_mean_lpips": mean_lpips,
            "automatic_4bit": False}


def update_status(output: Path, **values: Any) -> None:
    write_json(output / "execution_status.json", {"run_tag": RUN_TAG, "pid": os.getpid(), "updated_at_utc": utc_now(), **values})


def run_batch(cfg: dict[str, Any], identity: dict[str, Any], batch: int, output: Path, device: torch.device) -> dict[str, Any]:
    folder = output / f"batch{batch:02d}"
    started = time.perf_counter()
    inputs_identity = {**identity, "batch_index": batch, "stage": "cached_input_audit"}
    inputs = load_checkpoint(folder, "inputs", inputs_identity)
    if inputs is None:
        raise RuntimeError("Input audit must finish before batch execution")
    input_hash = file_hash(folder / "inputs.pt")
    batch_identity = {**identity, "batch_index": batch, "inputs_sha256": input_hash}
    completed = folder / "summary.json"
    if completed.is_file():
        summary = json.loads(completed.read_text(encoding="utf-8"))
        if summary["identity"] != batch_identity:
            raise RuntimeError(f"Completed batch identity mismatch: {folder}")
        for name, digest in summary["output_hashes"].items():
            if file_hash(folder / name) != digest:
                raise RuntimeError(f"Completed batch artifact changed: {folder / name}")
        return summary
    update_status(output, state="running", batch_index=batch, stage="actual_UIA_update_collection")
    observation_identity = {**batch_identity, "stage": "actual_UIA_update"}
    observed = load_checkpoint(folder, "observation", observation_identity)
    if observed is None:
        stage_start = time.perf_counter()
        seed_all(int(cfg["seed"]) + batch)
        feature = UIAFeatureBatch.from_cache_dict(torch.load(inputs["feature_cache_path"], map_location="cpu", weights_only=False))
        observation = run_fc_only_fedavg_on_features(feature, lr=float(cfg["uia"]["lr"]), steps=int(cfg["uia"]["steps"]), device=device)
        actual_observation(observation, maximum_relative_error=float(cfg["uia"]["update_factor_rel_l2_gate"]))
        observed = {"observation": vars(observation), "wall_time_sec": time.perf_counter() - stage_start,
                    "observation_used_for_attack": "Y_delta_over_lr", "definition": "(W_initial-W_after_10_local_steps)/lr",
                    "author_update_convention": "W_after-W_initial; ours is negative author update divided by lr",
                    "factor_usage": "B_cumulative and Y_factor are diagnostics only"}
        save_checkpoint(folder, "observation", observed, observation_identity)
        del observation, feature
    Y = actual_observation(observed["observation"], maximum_relative_error=float(cfg["uia"]["update_factor_rel_l2_gate"]))
    update_status(output, state="running", batch_index=batch, stage="CPA_full16")
    source_identity = {**batch_identity, "stage": "CPA_full16", "observation_sha256": file_hash(folder / "observation.pt")}
    source_payload = load_checkpoint(folder, "stage1", source_identity)
    if source_payload is None:
        stage_start = time.perf_counter()
        result = stage1_call(Y, cfg, batch, device)
        if result.B_eff != N or tuple(result.S_hat.shape) != (N, 25088):
            raise RuntimeError("CPA did not return full16 sources; no fallback is permitted")
        diag = {field.name: getattr(result, field.name) for field in fields(result) if field.name not in {"A_hat", "S_hat", "Y_recon"}}
        ranks = rank_diagnostics(
            result.lam_deb_topB, dimension=25088, rel_tol=float(cfg["rank"]["rel_tol"]), abs_tol=float(cfg["rank"]["abs_tol"]))
        targets, source_rows = fia_targets(result.S_hat, inputs["Z_evaluation_only"])
        source_payload = {"S_hat": result.S_hat, "A_hat": result.A_hat, "fia_targets": targets,
                          "source_rows": source_rows, "diagnostics": json_safe(diag), "rank": ranks,
                          "wall_time_sec": time.perf_counter() - stage_start}
        save_checkpoint(folder, "stage1", source_payload, source_identity)
        del result
    write_json(folder / "stage1_diagnostics.json", {"identity": source_identity, **source_payload["diagnostics"], **source_payload["rank"]})
    del Y
    observation_diagnostics = {key: observed["observation"][key] for key in ("update_factor_rel_l2", "update_factor_max_abs")}
    final_losses = observed["observation"]["final_per_sample_loss"].tolist()
    collection_wall = float(observed["wall_time_sec"])
    del observed
    if device.type == "cuda":
        torch.cuda.empty_cache()
    update_status(output, state="running", batch_index=batch, stage="FIA_25000")
    fia_identity = {**batch_identity, "stage": "unchanged_FIA_25000", "stage1_sha256": file_hash(folder / "stage1.pt")}
    reconstruction = load_checkpoint(folder, "fia", fia_identity)
    if reconstruction is None:
        fia = cfg["fia"]
        result = run_direct_fia(target_z=source_payload["fia_targets"], x_true=inputs["x"], device=device,
                                n_iter=fia["n_iter"], log_every=fia["log_every"], lr=fia["lr"], tv=fia["tv"], fi=fia["fi"],
                                optimizer=fia["optimizer"], scheduler=fia["scheduler"], seed=fia["seed"],
                                progress_label=f"UIA_fp32_B16_batch{batch}")
        reconstruction = {"rec": result.rec, "metrics": vars(result.metrics), "log": result.log, "wall_time_sec": result.wall_time_sec}
        save_checkpoint(folder, "fia", reconstruction, fia_identity)
        del result
    rec = reconstruction["rec"]
    metrics = reconstruction["metrics"]
    if tuple(rec.shape) != (N, 3, 224, 224) or not bool(torch.isfinite(rec).all()):
        raise RuntimeError("FIA reconstruction completeness/finiteness gate failed")
    if any(len(metrics[key]) != N or not np.isfinite(metrics[key]).all() for key in ("lpips", "psnr", "ssim")):
        raise RuntimeError("All 16 per-image metrics must be finite")
    update_status(output, state="running", batch_index=batch, stage="batch_artifacts")
    source_rows = [{**inputs["metadata"][i], **row} for i, row in enumerate(source_payload["source_rows"])]
    rows = [{**source_rows[i], "run_tag": RUN_TAG, "bits": "fp32", "T": 10, "n": N, "scale_granularity": "per_row",
             "final_per_sample_loss": float(final_losses[i]), **{key: float(metrics[key][i]) for key in ("lpips", "psnr", "ssim")}}
            for i in range(N)]
    write_csv(folder / "per_image_metrics.csv", rows)
    write_csv(folder / "source_cosines.csv", source_rows)
    torch.save(rec, folder / "rec.pt")
    write_json(folder / "fia_log.json", reconstruction["log"])
    render_grid(inputs["x"], rec, rows, folder / "reconstruction_grid.png", title=f"UIA fp32 B16 batch {batch}: all 16 originals")
    summary = {"schema_version": 1, "run_tag": RUN_TAG, "record_type": "batch_result", "identity": batch_identity,
               "batch_index": batch, "n_images": N, "bits": "fp32", "scale_granularity": "per_row", "T": 10,
               **{f"{key}_mean": float(np.mean(metrics[key])) for key in ("lpips", "psnr", "ssim")},
               "source_cos_per_source": [row["source_abs_cosine"] for row in source_rows],
               "source_cos_mean": float(np.mean([row["source_abs_cosine"] for row in source_rows])),
               "measured_B_eff_cos09": sum(row["measured_recovered_cos09"] for row in source_rows),
               **source_payload["rank"], "stage1_B_eff": N, **observation_diagnostics,
               "observation": "actual (W_initial-W_after_10_steps)/lr; Y_factor diagnostic only",
               "runtime_sec": {"cached_input_audit": inputs["wall_time_sec"], "update_collection": collection_wall,
                               "stage1": source_payload["wall_time_sec"], "fia": reconstruction["wall_time_sec"],
                               "current_batch_invocation": time.perf_counter() - started},
               "output_hashes": {name: file_hash(folder / name) for name in ("rec.pt", "per_image_metrics.csv", "source_cosines.csv",
                                  "fia_log.json", "reconstruction_grid.png", "stage1_diagnostics.json", "observation.pt", "stage1.pt", "fia.pt", "inputs.pt")},
               "completed_at_utc": utc_now(), "oracle_fills": 0}
    write_json(completed, summary)
    print(f"COMPLETE UIA fp32 batch={batch}: LPIPS={summary['lpips_mean']:.8f}, measured B_eff={summary['measured_B_eff_cos09']}/16", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def render_grid(x: torch.Tensor, rec: torch.Tensor, rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    from PIL import Image, ImageDraw, ImageFont
    truth = denormalize_imagenet(x)
    tile, gap, label_height, header, columns = 112, 12, 27, 58, 4
    block = 2 * tile + gap
    canvas = Image.new("RGB", (columns * (block + gap) + gap, header + math.ceil(len(rows) / columns) * (tile + label_height + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 14) if font_path.is_file() else ImageFont.load_default()
    draw.text((gap, 8), title, fill="black", font=font)
    draw.text((gap, 30), "Original (left) / reconstruction (right); every image included, no selection by quality", fill="#475569", font=font)
    for index, row in enumerate(rows):
        left = gap + index % columns * (block + gap)
        top = header + index // columns * (tile + label_height + gap)
        draw.text((left, top), f"g{int(row['global_index']):02d} LPIPS={float(row['lpips']):.3f} cos={float(row['source_abs_cosine']):.3f}", fill="black", font=font)
        for column, image in enumerate((truth[index], rec[index])):
            array = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            thumb = Image.fromarray(array).resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (left + column * (tile + gap), top + label_height))
    canvas.save(path)


def summarize(cfg: dict[str, Any], identity: dict[str, Any], output: Path) -> dict[str, Any]:
    batch_summaries, rows, originals, recs = [], [], [], []
    for batch in BATCHES:
        folder = output / f"batch{batch:02d}"
        marker = folder / "summary.json"
        if not marker.is_file():
            raise RuntimeError("All five completed batches are required; no partial aggregate")
        summary = json.loads(marker.read_text(encoding="utf-8"))
        expected = {**identity, "batch_index": batch, "inputs_sha256": file_hash(folder / "inputs.pt")}
        if summary["identity"] != expected:
            raise RuntimeError("Completed batch has a different protocol/code/input identity")
        for name, digest in summary["output_hashes"].items():
            if file_hash(folder / name) != digest:
                raise RuntimeError(f"Completed aggregate input changed: {folder / name}")
        with (folder / "per_image_metrics.csv").open(newline="", encoding="utf-8") as handle:
            selected = list(csv.DictReader(handle))
        for row in selected:
            for key in ("global_index", "batch_index", "source_index", "estimated_source_index", "label", "measured_recovered_cos09"):
                row[key] = int(row[key])
            for key in ("lpips", "psnr", "ssim", "source_abs_cosine", "fia_abs_target_cosine"):
                row[key] = float(row[key])
        if len(selected) != N or [row["global_index"] for row in selected] != list(range(batch * N, (batch + 1) * N)):
            raise RuntimeError("Per-image global index coverage is incomplete")
        inputs = torch.load(folder / "inputs.pt", map_location="cpu", weights_only=False)
        originals.append(inputs["x"])
        recs.append(torch.load(folder / "rec.pt", map_location="cpu", weights_only=True))
        batch_summaries.append(summary)
        rows.extend(selected)
    if len(rows) != 80 or len({row["image_path"] for row in rows}) != 80:
        raise RuntimeError("Final metrics must contain all 80 unique originals")
    write_csv(output / "per_image_metrics.csv", rows)
    source_rows = [{key: row[key] for key in ("global_index", "batch_index", "source_index", "estimated_source_index",
                   "source_abs_cosine", "fia_abs_target_cosine", "measured_recovered_cos09", "missing_source", "oracle_fill")} for row in rows]
    write_csv(output / "source_cosines.csv", source_rows)
    batch_rows = [{"batch_index": item["batch_index"], "n_images": N, "predicted_B_eff": item["predicted_B_eff"],
                   "numeric_rank_top16": item["numerical_rank_top16_observed_centered"], "measured_B_eff_cos09": item["measured_B_eff_cos09"],
                   **{f"{key}_mean": item[f"{key}_mean"] for key in ("lpips", "psnr", "ssim")},
                   **{f"runtime_{key}_sec": value for key, value in item["runtime_sec"].items()}} for item in batch_summaries]
    write_csv(output / "per_batch_summary.csv", batch_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
    for batch in BATCHES:
        selected = [row for row in rows if row["batch_index"] == batch]
        axes[0].hist([row["lpips"] for row in selected], bins=np.linspace(0, max(1.1, max(row["lpips"] for row in rows) * 1.03), 31),
                     histtype="step", linewidth=1.8, label=f"batch {batch} (n=16)")
        axes[1].hist([row["source_abs_cosine"] for row in selected], bins=np.linspace(0, 1.000001, 31), histtype="step", linewidth=1.8)
    axes[0].set(xlabel="Per-image LPIPS", ylabel="Image count")
    axes[1].set(xlabel="Matched absolute ICA source cosine", ylabel="Source count")
    axes[1].axvline(.9, linestyle="--", color="black", linewidth=1)
    axes[0].legend(frameon=False)
    fig.suptitle("Actual fp32 UIA: full16 CPA + unchanged FIA; all 80 images")
    fig.tight_layout()
    fig.savefig(output / "lpips_source_cosine_histograms.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 3.4), dpi=170)
    ax.axis("off")
    table_rows = [[row["batch_index"], N, row["predicted_B_eff"], row["numeric_rank_top16"], row["measured_B_eff_cos09"],
                   f"{row['lpips_mean']:.4f}", f"{row['psnr_mean']:.2f}", f"{row['ssim_mean']:.4f}", f"{row['runtime_fia_sec']/3600:.2f}"] for row in batch_rows]
    table = ax.table(cellText=table_rows, colLabels=["Batch", "Images", "Pred B_eff", "Rank top16", "Cos>0.9", "LPIPS", "PSNR", "SSIM", "FIA hours"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    ax.set_title("UIA fp32 B16, T10: five independent round-0 client updates")
    fig.tight_layout()
    fig.savefig(output / "per_batch_table.png")
    plt.close(fig)
    render_grid(torch.cat(originals), torch.cat(recs), rows, output / "all80_reconstruction_grid.png", title="UIA fp32 B16, T10: all 80 originals")
    means = {key: float(np.mean([row[key] for row in rows])) for key in ("lpips", "psnr", "ssim", "source_abs_cosine")}
    decision = decision_label(means["lpips"], cfg)
    from .run import git_sha
    sha = git_sha()
    summary = {"schema_version": 1, "record_type": "full_run", "run_tag": RUN_TAG, "git_sha": sha, "identity": identity,
               "config": cfg, "config_sha256": identity["protocol_sha256"], "status": "complete", "n_images": 80, "n_victim_batches": 5,
               "metrics": {**{f"{key}_mean": value for key, value in means.items()},
                           "measured_B_eff_cos09_total": sum(row["measured_recovered_cos09"] for row in rows),
                           "source_cos_per_source": [row["source_abs_cosine"] for row in rows]},
               "per_victim_batch": batch_summaries, "decision": decision,
               "runtime_sec": {key: sum(item["runtime_sec"][key] for item in batch_summaries)
                               for key in ("cached_input_audit", "update_collection", "stage1", "fia")},
               "gia_reference_is_unpaired": True, "gia_reference": cfg["evaluation"]["gia_reference"],
               "published_table5_reference": {"n": 16, "lpips": .410, "measured_minus_reference": means["lpips"] - .410,
                                               "comparison": "published protocol reference, not a paired-image effect estimate"},
               "oracle_fills": 0, "automatic_4bit": False, "completed_at_utc": utc_now(),
               "output_hashes": {name: file_hash(output / name) for name in ("per_image_metrics.csv", "source_cosines.csv", "per_batch_summary.csv",
                                  "lpips_source_cosine_histograms.png", "per_batch_table.png", "all80_reconstruction_grid.png")}}
    report = ["# M6 UIA fp32 B16 + CPA/FIA", "", "Completed: 80 images, five independent B=16 client updates from round 0; T=10, SGD lr=0.001.", "",
              "| Metric | Value |", "|---|---:|", f"| Mean LPIPS | {means['lpips']:.8f} |", f"| Mean PSNR | {means['psnr']:.4f} |",
              f"| Mean SSIM | {means['ssim']:.6f} |", f"| ICA sources with absolute cosine > 0.9 | {summary['metrics']['measured_B_eff_cos09_total']}/80 |", "",
              f"Published Table 5 n=16 reference LPIPS: **0.410**; observed mean minus reference: **{means['lpips']-.410:+.8f}**. This compares protocol-level results on different images, not paired reconstruction outcomes.", "",
              f"Fixed decision band: **{decision['classification']}**. {decision['action']}", "",
              "CPA consumes the actual observed (W_initial - W_after)/lr. B_T @ Z is retained only to diagnose floating-point update error.",
              "Full 16 CPA components, zero quantization delta, no MP truncation, no debiasing; missing sources/failed whitening abort. No oracle-source fills.",
              "FIA uses the historical abs(source) adapter and permutation-only matching, 25,000 iterations, TV=0.1, FI=1, Adam lr=0.1, cosine schedule, seed 42, batches of 16.", "",
              "The GIA reference 0.44481556 used B=8 and 40 different batch compositions, including four exact-source fills. It is an unpaired contextual reference, not a controlled UIA effect estimate.",
              "The 80 fixed distinct-label validation images differ from the authors' shuffled sampling protocol; local n=16 fits inside nominal minibatch 32. The model is reset before each independent update.", "",
              "![Per-image distributions](lpips_source_cosine_histograms.png)", "![Per-batch table](per_batch_table.png)",
              "![Every reconstruction](all80_reconstruction_grid.png)", "", "No automatic 4-bit, Stage 2, refinement, or GMA task is launched.", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    summary["output_hashes"]["report.md"] = file_hash(output / "report.md")
    ledger = resolve_path(cfg["paths"]["results_jsonl"])
    exists = False
    if ledger.is_file():
        with ledger.open(encoding="utf-8") as handle:
            for line in handle:
                if RUN_TAG in line:
                    record = json.loads(line)
                    if record.get("record_type") == "full_run" and record.get("identity") == identity:
                        exists = True
                        break
    if not exists:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(summary), ensure_ascii=False, allow_nan=False) + "\n")
    # This is the queue's completion marker: every requested artifact and the
    # ledger record must already exist before status='complete' becomes visible.
    write_json(output / "summary.json", summary)
    update_status(output, state="complete", report_path=str(output / "report.md"), protocol_sha256=identity["protocol_sha256"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("prepare", "run", "report"), default="run")
    args = parser.parse_args()
    cfg, lock = read_locked_config(args.config)
    output = resolve_path(cfg["paths"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    # Creating the containing directory/lock file is the only mutation before
    # ownership; no prepared input, status, checkpoint, or model precedes it.
    process_lock = (output / ".execution.lock").open("a+b")
    process_lock.seek(0, 2)
    if process_lock.tell() == 0:
        process_lock.write(b"0")
        process_lock.flush()
    process_lock.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(process_lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if args.phase == "run":
            check_dependency(cfg)
        torch.set_num_threads(4)
        identity = implementation_identity(lock)
        if args.phase == "report":
            summarize(cfg, identity, output)
            return
        prepare(cfg, identity, output)
        if args.phase == "prepare":
            print("CPU cached-input preparation complete; no UIA observation, ICA, FIA, or GPU work performed", flush=True)
            return
        device = torch.device(cfg["device"])
        if device.type == "cuda":
            if not torch.cuda.is_available() or torch.cuda.get_device_capability(device) != (12, 0):
                raise RuntimeError("Expected the established sm_120 GPU environment")
        for batch in BATCHES:
            run_batch(cfg, identity, batch, output, device)
        summarize(cfg, identity, output)
    except BaseException as exc:
        update_status(output, state="failed", phase=args.phase, error=str(exc), traceback=traceback.format_exc(),
                      protocol_sha256=lock["sha256"])
        raise
    finally:
        process_lock.close()


if __name__ == "__main__":
    main()
