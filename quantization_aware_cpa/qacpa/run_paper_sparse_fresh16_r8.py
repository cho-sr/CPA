"""Fresh 16-image paper sparse naive versus CoRA8 experiment.

This driver owns the v4 replacement experiment.  It reads only the registered
original batch00 images and public VGG16 weights, collects a new B16/T10 local
update, creates new Top-k packets, and runs the naive and eight-candidate
random-fill arms from new output directories.  Historical attack artifacts,
metrics, checkpoints, and reconstructions are never used as inputs here.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import paper_sparse_codec
from .fia import denormalize_imagenet, evaluate_images
from .paper_sparse_codec import SOURCE_CODECS, expected_packet_bytes, validate_received_packet
from .random_fill import candidate_seed, consistency_score, random_fill_packet, rank_candidates
from .run_m8_fedavg_naive import (
    PROJECT,
    WORKSPACE,
    evaluate_after_fia,
    execution_lock,
    file_hash,
    read_csv,
    read_json,
    resolve,
    status as write_status,
    utc_now,
    write_csv,
    write_json,
)
from .seed import seed_all
from .staged_fia import run_staged_fia
from .topk_ratio_codec import decode_packet, encode_topk, sparsification_diagnostics
from .upstream_naive import VGG16FeatureAdapter, _save_checkpoint, run_upstream_cpa, _tensor_hash
from .victim import disable_inplace_relu


TASK_ID = "CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921"
PLAN_VERSION = 4
AMENDMENT_ID = "CPA_PAPER_SPARSE_FRESH16_R8_AMENDMENT_20260921"
PERCENTS = (50, 30, 20)
CANDIDATES = tuple(range(8))
N_IMAGES = 16
BATCH_INDEX = 0
T_STEPS = 10
FC1_SHAPE = (4096, 25088)
INPUT_RELATIVE = "results/m6_uia_fp32_B16_fia/batch00/inputs.pt"
INPUT_SHA256 = "24995487511a3e9d8d4665fdcd704f7a0f792249a9ba79880638d13bc4740141"
OUTPUT_RELATIVE = "results/paper_sparse_fresh16_keep50_30_20_R8_20260921"
PRIVATE_RELATIVE = "results_private/paper_sparse_fresh16_keep50_30_20_R8_20260921"
FRESH_RANDOMIZATION = {
    "sampler": "uniform_tau_pcg64_v1",
    "tau": "min(abs(transmitted FP32 values))",
    "candidates": 8,
    "candidate_ids": list(CANDIDATES),
    "screen_iterations": 5000,
    "advance_top_k": 3,
    "final_iterations": 25000,
    "cosine_horizon": 25000,
    "fill_seed": "100000 + candidate_id; batch00",
    "cpa_seed": "0; identical for all candidates",
    "fia_seed": "42; identical for all candidates",
    "ranking": "relative L2 of recompressed reconstructed-data SGD update to received packet",
    "ranking_labels": "public round0 model argmax; no true labels",
    "tie_break": "candidate_index ascending",
    "selection_unit": "whole16-image candidate for original batch00; candidate IDs 0..7 only",
    "final_selection": "lowest same observable relative L2 among three final candidates",
    "true_metrics_for_selection": False,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _tensor_hash_cpu(value: torch.Tensor) -> str:
    return _tensor_hash(value.detach().cpu().contiguous())


def _public_weights_path() -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"


def _code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__),
        PROJECT / "qacpa/paper_sparse_codec.py",
        PROJECT / "qacpa/topk_ratio_codec.py",
        PROJECT / "qacpa/random_fill.py",
        PROJECT / "qacpa/staged_fia.py",
        PROJECT / "qacpa/upstream_naive.py",
        PROJECT / "qacpa/run_m8_fedavg_naive.py",
        PROJECT / "qacpa/fia.py",
        PROJECT / "qacpa/victim.py",
    ]
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in paths}


def _identity_rows(metadata: list[dict[str, Any]]) -> list[tuple[str, ...]]:
    fields = ("global_index", "batch_index", "source_index", "dataset_index", "label", "image_path")
    return [tuple(str(row[field]) for field in fields) for row in metadata]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {f"{key}_mean": float(np.mean([float(row[key]) for row in rows])) for key in ("lpips", "psnr", "ssim")}


def _checked_marker(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    saved = read_json(path)
    if saved.get("identity") != identity:
        raise RuntimeError(f"Fresh v4 artifact identity differs: {path}")
    for name, expected in saved.get("output_hashes", {}).items():
        artifact = path.parent / name
        if not artifact.is_file() or file_hash(artifact) != expected:
            raise RuntimeError(f"Fresh v4 artifact hash differs: {artifact}")
    return saved


def _save_marker(path: Path, identity: dict[str, Any], names: list[str], **values: Any) -> dict[str, Any]:
    result = {
        "identity": identity,
        **values,
        "output_hashes": {name: file_hash(path.parent / name) for name in names},
    }
    write_json(path, result)
    return result


def _static_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "plan_version": PLAN_VERSION,
        "amendment_id": AMENDMENT_ID,
        "run_tag": "paper_sparse_fresh16_keep50_30_20_R8_20260921",
        "output_dir": OUTPUT_RELATIVE,
        "private_output_dir": PRIVATE_RELATIVE,
        "source_input": INPUT_RELATIVE,
        "device": "cuda",
        "seed": 0,
        "n": N_IMAGES,
        "T": T_STEPS,
        "n_images": N_IMAGES,
        "batch_indices": [0],
        "keep_percents": list(PERCENTS),
        "candidate_count": len(CANDIDATES),
        "candidate_ids": list(CANDIDATES),
        "selection": "original batch00 in original global_index order; no LPIPS selection",
        "local_training": {
            "lr": 0.001,
            "steps": 10,
            "optimizer": "sgd",
            "momentum": 0.0,
            "weight_decay": 0.0,
            "classifier_dropout": False,
            "conv_frozen": True,
            "updated_layers": "all classifier parameters",
            "model_reset_each_update": True,
            "independent_client_updates": 1,
            "provenance": "fresh v4 B16/T10 collection from original batch00 pixels",
        },
        "model": {
            "name": "vgg16",
            "weights": "IMAGENET1K_V1",
            "attacked_checkpoint_round": 0,
            "conv_frozen": True,
            "updated_layers": "all classifier parameters",
            "attack_layer": "classifier.0.weight",
        },
        "stage1": {
            "cpa_ica_max_iter": 25000,
            "cpa_ica_lr": 0.001,
            "cpa_ica_optimizer": "adam",
            "cpa_ica_scheduler": "none",
            "cpa_ica_decor": 5.3,
            "cpa_ica_T": 7.7,
            "cpa_ica_ne": 1.0,
            "cpa_ica_nv": 0.13,
            "cpa_ica_l1": 5.0,
            "cpa_ica_tv": 0.1,
            "cpa_ica_log_every": 5000,
        },
        "fia": {
            "n_iter": 25000,
            "log_every": 5000,
            "lr": 0.1,
            "tv": 0.1,
            "fi": 1.0,
            "optimizer": "adam",
            "scheduler": "cosine",
            "seed": 42,
        },
        "sparsification": {
            "selection": "global magnitude within classifier.0.weight",
            "tie_break": "flat index ascending",
            "value_dtype": "float32",
            "positions": "row-major bitmap, little bit order",
            "discarded_fill": 0.0,
            "quantization": False,
            "rescaling": False,
            "error_feedback": False,
        },
        "randomization": FRESH_RANDOMIZATION,
        "evaluation": "post-FIA signed-source absolute-cosine Hungarian within original batch00; fresh x/Z only after public selection",
        "compression_scope": "attacked classifier.0.weight packet only",
        "new_training": True,
        "new_update_collection": True,
        "old_attack_artifacts_imported": False,
        "old_metrics_imported": False,
        "historical_scope": "audit-only original identities and public model; no old update/packet/checkpoint/source/score/rec/metric reuse",
    }


def validate_config(cfg: dict[str, Any], config_path: Path) -> None:
    expected = _static_config()
    if cfg != expected:
        raise ValueError("The v4 config differs from the locked fresh16/R8 protocol")
    if cfg["candidate_ids"] != list(CANDIDATES) or cfg["candidate_count"] != len(CANDIDATES):
        raise ValueError("Fresh v4 candidate IDs must be exactly 0..7")
    if cfg["keep_percents"] != list(PERCENTS) or cfg["batch_indices"] != [0]:
        raise ValueError("Fresh v4 scope must be keep50/30/20 and original batch00 only")
    if cfg["stage1"]["cpa_ica_max_iter"] != 25000 or cfg["fia"]["n_iter"] != 25000:
        raise ValueError("Fresh v4 attack horizons changed")
    if cfg["randomization"] != FRESH_RANDOMIZATION:
        raise ValueError("Fresh v4 randomization protocol changed")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)


def _load_registered_input() -> tuple[Path, dict[str, Any], torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    path = resolve(INPUT_RELATIVE)
    if file_hash(path) != INPUT_SHA256:
        raise RuntimeError("The authenticated original batch00 inputs.pt changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    x = payload["x"].detach().cpu().float().contiguous()
    metadata = list(payload["metadata"])
    labels = torch.tensor([int(row["label"]) for row in metadata], dtype=torch.long)
    if x.shape != (N_IMAGES, 3, 224, 224) or labels.shape != (N_IMAGES,):
        raise RuntimeError("Registered batch00 tensors do not have B16 shapes")
    if not torch.isfinite(x).all() or len(set(int(value) for value in labels.tolist())) != N_IMAGES:
        raise RuntimeError("Registered batch00 tensors failed finite/distinct-label checks")
    if [int(row["global_index"]) for row in metadata] != list(range(N_IMAGES)):
        raise RuntimeError("Registered batch00 metadata is not global_index0..15")
    if len(metadata) != N_IMAGES or len(_identity_rows(metadata)) != N_IMAGES:
        raise RuntimeError("Registered batch00 metadata is incomplete")
    return path, payload, x, labels, metadata


def preflight(config_path: Path, cfg: dict[str, Any], output: Path, private: Path) -> dict[str, Any]:
    source_path, _, x, labels, metadata = _load_registered_input()
    weights = _public_weights_path()
    if not weights.is_file():
        raise RuntimeError(f"Public VGG16 checkpoint is missing: {weights}")
    weights_sha = file_hash(weights)
    output.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    identity = {
        "task_id": TASK_ID,
        "plan_version": PLAN_VERSION,
        "amendment_id": AMENDMENT_ID,
        "config_sha256": file_hash(config_path),
        "runner_sha256": file_hash(Path(__file__)),
        "code_sha256": _code_hashes(),
        "source_input_path": str(source_path),
        "source_input_sha256": INPUT_SHA256,
        "original_x_sha256": _tensor_hash_cpu(x),
        "original_labels_sha256": _tensor_hash_cpu(labels),
        "weights_path": str(weights),
        "weights_sha256": weights_sha,
        "n_images": N_IMAGES,
        "batch_indices": [0],
        "keep_percents": list(PERCENTS),
        "candidate_ids": list(CANDIDATES),
        "new_update_collection": True,
        "old_attack_artifacts_imported": False,
    }
    manifest_path = output / "input_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("identity") != identity:
            raise RuntimeError("Existing fresh v4 input manifest has incompatible identity")
        if _identity_rows(manifest.get("metadata", [])) != _identity_rows(metadata):
            raise RuntimeError("Existing fresh v4 input identity changed")
        return manifest
    supersession = {
        "task_id": TASK_ID,
        "replacement_plan_version": PLAN_VERSION,
        "amendment_id": AMENDMENT_ID,
        "amendment_receipt": ".ai/execution-handoffs/CPA_PAPER_SPARSE_FRESH16_R8_AMENDMENT_20260921.md",
        "superseded_versions": [1, 2, 3],
        "superseded_handoff_archive": ".ai/execution-handoffs/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921_V3_FAILED_ARCHIVE_20260921.md",
        "historical_same80_and_same16_reuse": "audit_only",
        "old_attack_artifacts_imported": False,
        "old_callbacks_resumed": False,
        "old_worker_running_at_registration": False,
        "new_worker_scope": "one fresh16/R8 serial worker; no batches01..04; no parallel arm",
        "recorded_at_utc": _now(),
    }
    write_json(output / "supersession_record.json", supersession)
    manifest = {
        "schema_version": 4,
        "identity": identity,
        "metadata": metadata,
        "n_images": N_IMAGES,
        "batch_indices": [0],
        "source_input": str(source_path),
        "weights_path": str(weights),
        "weights_sha256": weights_sha,
        "new_training": True,
        "new_update_collection": True,
        "fresh_private_output": str(private),
        "created_at_utc": _now(),
        "scope": "original batch00 global_index0..15 only",
    }
    write_csv(output / "image_manifest.csv", metadata)
    manifest["image_manifest_sha256"] = file_hash(output / "image_manifest.csv")
    write_json(manifest_path, manifest)
    write_json(output / "preflight.json", {
        "identity": identity,
        "input_manifest_sha256": file_hash(manifest_path),
        "source_input_sha256": INPUT_SHA256,
        "weights_sha256": weights_sha,
        "x_shape": list(x.shape),
        "labels_shape": list(labels.shape),
        "metadata_global_indices": list(range(N_IMAGES)),
        "fresh_update_required": True,
        "validated_at_utc": _now(),
    })
    return manifest


def _collect_fresh_update(cfg: dict[str, Any], manifest: dict[str, Any], private: Path,
                          device: torch.device) -> dict[str, Any]:
    source_path, _, x, labels, metadata = _load_registered_input()
    update_path = private / "fresh_victim_update.pt"
    truth_path = private / "fresh_truth.pt"
    manifest_path = private / "fresh_update_manifest.json"
    if update_path.is_file() or truth_path.is_file() or manifest_path.is_file():
        if not (update_path.is_file() and truth_path.is_file() and manifest_path.is_file()):
            raise RuntimeError("Fresh private collection is partially present; refusing mixed state")
        saved = read_json(manifest_path)
        if saved.get("input_manifest_identity") != manifest["identity"]:
            raise RuntimeError("Fresh private collection belongs to a different input identity")
        for path_key in ("update_path", "truth_path"):
            path = Path(saved[path_key])
            if file_hash(path) != saved[f"{path_key}_sha256"]:
                raise RuntimeError(f"Fresh private collection changed: {path}")
        return saved

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The authorized fresh experiment requires the established CUDA device")
    from torchvision.models import VGG16_Weights, vgg16

    started = time.perf_counter()
    seed_all(0)
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
    disable_inplace_relu(model)
    model.eval()
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    classifier = model.classifier
    attacked = classifier[0]
    if not isinstance(attacked, torch.nn.Linear) or tuple(attacked.weight.shape) != FC1_SHAPE:
        raise RuntimeError("Unexpected VGG16 classifier.0.weight shape")
    x_device = x.to(device=device, dtype=torch.float32)
    labels_device = labels.to(device=device, dtype=torch.long)
    with torch.no_grad():
        features = model.features(x_device)
        z = model.avgpool(features).flatten(1).contiguous()
        initial_logits = classifier(z)
        initial_per_sample_loss = F.cross_entropy(initial_logits, labels_device, reduction="none")
    W_initial = attacked.weight.detach().clone()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0)
    cumulative = torch.zeros((FC1_SHAPE[0], N_IMAGES), device=device, dtype=torch.float32)
    step_losses: list[float] = []
    with torch.enable_grad():
        for step in range(T_STEPS):
            optimizer.zero_grad(set_to_none=True)
            hidden = attacked(z)
            hidden.retain_grad()
            logits = classifier[1:](hidden)
            per_sample_loss = F.cross_entropy(logits, labels_device, reduction="none")
            loss = per_sample_loss.mean()
            loss.backward()
            if hidden.grad is None:
                raise RuntimeError("Fresh update collection did not retain classifier.0 hidden gradients")
            cumulative += hidden.grad.detach().T
            optimizer.step()
            step_losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        final_logits = classifier(z)
        final_per_sample_loss = F.cross_entropy(final_logits, labels_device, reduction="none")
    Y = ((W_initial - attacked.weight.detach()) / 0.001).detach().cpu().float().contiguous()
    Z = z.detach().cpu().float().contiguous()
    cumulative_cpu = cumulative.detach().cpu().float().contiguous()
    Y_factor = cumulative_cpu.matmul(Z)
    rel_error = float((Y - Y_factor).norm() / Y_factor.norm().clamp_min(1.0e-12))
    max_error = float((Y - Y_factor).abs().max())
    if Y.shape != FC1_SHAPE or not torch.isfinite(Y).all() or not math.isfinite(rel_error) or rel_error > 1.0e-3:
        raise RuntimeError(f"Fresh observed update failed shape/factor gate: rel={rel_error}")
    if not torch.isfinite(Z).all() or Z.shape != (N_IMAGES, FC1_SHAPE[1]):
        raise RuntimeError("Fresh evaluation feature collection failed")
    truth_payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "plan_version": PLAN_VERSION,
        "amendment_id": AMENDMENT_ID,
        "x": x,
        "labels": labels,
        "Z": Z,
        "metadata": metadata,
        "source_input_sha256": INPUT_SHA256,
        "weights_sha256": manifest["identity"]["weights_sha256"],
        "created_at_utc": _now(),
    }
    update_payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "plan_version": PLAN_VERSION,
        "amendment_id": AMENDMENT_ID,
        "Y": Y,
        "B_cumulative": cumulative_cpu,
        "Y_factor": Y_factor,
        "initial_logits": initial_logits.detach().cpu().float(),
        "final_logits": final_logits.detach().cpu().float(),
        "initial_per_sample_loss": initial_per_sample_loss.detach().cpu().float(),
        "final_per_sample_loss": final_per_sample_loss.detach().cpu().float(),
        "step_losses": step_losses,
        "lr": 0.001,
        "steps": T_STEPS,
        "optimizer": "sgd",
        "momentum": 0.0,
        "weight_decay": 0.0,
        "observation_convention": "Y=(W_initial-W_after10)/0.001",
        "update_factor_rel_l2": rel_error,
        "update_factor_max_abs": max_error,
        "source_input_sha256": INPUT_SHA256,
        "weights_sha256": manifest["identity"]["weights_sha256"],
        "created_at_utc": _now(),
    }
    _atomic_torch_save(truth_path, truth_payload)
    _atomic_torch_save(update_path, update_payload)
    record = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "plan_version": PLAN_VERSION,
        "amendment_id": AMENDMENT_ID,
        "input_manifest_identity": manifest["identity"],
        "source_input_path": str(source_path),
        "source_input_sha256": INPUT_SHA256,
        "update_path": str(update_path),
        "truth_path": str(truth_path),
        "update_sha256": file_hash(update_path),
        "truth_sha256": file_hash(truth_path),
        "Y_sha256": _tensor_hash_cpu(Y),
        "Z_sha256": _tensor_hash_cpu(Z),
        "Y_shape": list(Y.shape),
        "Z_shape": list(Z.shape),
        "lr": 0.001,
        "steps": T_STEPS,
        "fresh_collection": True,
        "historical_update_imported": False,
        "update_factor_rel_l2": rel_error,
        "update_factor_max_abs": max_error,
        "runtime_sec": time.perf_counter() - started,
        "collected_at_utc": _now(),
    }
    write_json(manifest_path, record)
    del model, classifier, x_device, labels_device, features, z, Y, cumulative, Y_factor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def _load_update(record: dict[str, Any]) -> torch.Tensor:
    path = Path(record["update_path"])
    if file_hash(path) != record["update_sha256"]:
        raise RuntimeError("Fresh private update hash changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("observation_convention") != "Y=(W_initial-W_after10)/0.001":
        raise RuntimeError("Fresh update convention changed")
    Y = payload["Y"].detach().cpu().float().contiguous()
    if Y.shape != FC1_SHAPE or not torch.isfinite(Y).all():
        raise RuntimeError("Fresh update payload is invalid")
    return Y


def _load_truth(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["truth_path"])
    if file_hash(path) != record["truth_sha256"]:
        raise RuntimeError("Fresh private truth hash changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    x = payload["x"].float().contiguous()
    Z = payload["Z"].float().contiguous()
    if x.shape != (N_IMAGES, 3, 224, 224) or Z.shape != (N_IMAGES, FC1_SHAPE[1]):
        raise RuntimeError("Fresh truth shape changed")
    return payload


def _packet_setup(cfg: dict[str, Any], manifest: dict[str, Any], update_record: dict[str, Any],
                  Y: torch.Tensor, output: Path) -> dict[int, dict[str, Any]]:
    packets: dict[int, dict[str, Any]] = {}
    for percent in PERCENTS:
        fraction = output / f"keep{percent}"
        packet_dir = fraction / "packet"
        packet_dir.mkdir(parents=True, exist_ok=True)
        packet_path = packet_dir / "observation.bin"
        packet = encode_topk(Y, percent)
        audit = validate_received_packet(packet, percent, SOURCE_CODECS[percent])
        decoded = decode_packet(packet)
        diagnostics = sparsification_diagnostics(Y, packet, decoded)
        if len(packet) != expected_packet_bytes(percent):
            raise RuntimeError(f"Fresh Top-k{percent} packet has an unexpected byte count")
        if packet_path.exists() and packet_path.read_bytes() != packet:
            raise RuntimeError(f"Existing fresh Top-k{percent} packet differs from the new update")
        if not packet_path.exists():
            temporary = packet_path.with_suffix(".bin.tmp")
            temporary.write_bytes(packet)
            temporary.replace(packet_path)
        communication = {
            "task_id": TASK_ID,
            "plan_version": PLAN_VERSION,
            "amendment_id": AMENDMENT_ID,
            "keep_percent": percent,
            "codec": SOURCE_CODECS[percent],
            "generic_codec": "topk_global_bitmap_fp32_v1",
            "value_bits": 32,
            "keep_fraction": percent / 100,
            "shape": list(FC1_SHAPE),
            "n_values": int(Y.numel()),
            "kept_elements": int(Y.numel() * percent // 100),
            "fp32_packet_bytes": 32 + 4 * Y.numel(),
            "value_bytes": 4 * int(Y.numel() * percent // 100),
            "bitmap_bytes": (int(Y.numel()) + 7) // 8,
            "header_bytes": 32,
            "packet_bytes": len(packet),
            "compression_ratio": (32 + 4 * int(Y.numel())) / len(packet),
            "saved_percent": 100.0 * (1.0 - len(packet) / (32 + 4 * int(Y.numel()))),
            "retained_fp32_bits_exact": True,
            "scope": "attacked classifier.0.weight packet only",
            "selection": "global magnitude within fresh Y; flat index ascending ties",
            "quantization": False,
            "rescaling": False,
            "error_feedback": False,
            "transmitted_tensor_convention": "Y=(W_initial-W_after10)/0.001",
            "fresh_update_sha256": update_record["update_sha256"],
            "fresh_Y_sha256": update_record["Y_sha256"],
            "sha256": file_hash(packet_path),
            "created_at_utc": _now(),
        }
        write_json(packet_dir / "communication.json", communication)
        diagnostics.update({
            "task_id": TASK_ID,
            "plan_version": PLAN_VERSION,
            "amendment_id": AMENDMENT_ID,
            "keep_percent": percent,
            "fresh_update_sha256": update_record["update_sha256"],
            "fresh_observation": True,
        })
        write_json(packet_dir / "sparsification_diagnostics.json", diagnostics)
        packet_manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "plan_version": PLAN_VERSION,
            "amendment_id": AMENDMENT_ID,
            "keep_percent": percent,
            "packet_path": str(packet_path),
            "packet_sha256": file_hash(packet_path),
            "packet_bytes": len(packet),
            "source_codec": SOURCE_CODECS[percent],
            "fresh_update_sha256": update_record["update_sha256"],
            "fresh_Y_sha256": update_record["Y_sha256"],
            "shared_by": ["naive_zero_fill", "uniform_random_fill_candidates_0_to_7"],
            "old_packet_imported": False,
            "validation": audit,
            "communication_sha256": file_hash(packet_dir / "communication.json"),
            "diagnostics_sha256": file_hash(packet_dir / "sparsification_diagnostics.json"),
            "created_at_utc": _now(),
        }
        write_json(packet_dir / "packet_manifest.json", packet_manifest)
        packets[percent] = packet_manifest
        del packet, decoded
    return packets


def _fraction_identity(root_identity: dict[str, Any], percent: int, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        **root_identity,
        "keep_percent": percent,
        "source_codec": packet["source_codec"],
        "packet_sha256": packet["packet_sha256"],
        "packet_bytes": packet["packet_bytes"],
        "fresh_update_sha256": packet["fresh_update_sha256"],
        "fresh_Y_sha256": packet["fresh_Y_sha256"],
        "attack_artifacts_imported": False,
    }


def _status(output: Path, cfg: dict[str, Any], **values: Any) -> None:
    write_status(output, cfg, task_id=TASK_ID, plan_version=PLAN_VERSION,
                 amendment_id=AMENDMENT_ID, n_images=N_IMAGES,
                 candidate_ids=list(CANDIDATES), keep_percents=list(PERCENTS), **values)


def _verify_packet(packet_manifest: dict[str, Any]) -> bytes:
    path = Path(packet_manifest["packet_path"])
    packet = path.read_bytes()
    if file_hash(path) != packet_manifest["packet_sha256"] or len(packet) != packet_manifest["packet_bytes"]:
        raise RuntimeError(f"Fresh packet changed: {path}")
    return packet


def _run_naive_attack(cfg: dict[str, Any], identity: dict[str, Any], packet_manifest: dict[str, Any],
                      folder: Path, device: torch.device, update_status_fn) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    done = _checked_marker(folder / "summary.json", identity)
    if done is not None:
        return done
    packet = _verify_packet(packet_manifest)
    percent = int(identity["keep_percent"])
    cpa_marker = _checked_marker(folder / "cpa_result.json", identity)
    if cpa_marker is None:
        update_status_fn("fresh_naive_CPA", None, 25000)
        zero = paper_sparse_codec.decode_for_attack(packet, percent)
        write_json(folder / "zero_fill_audit.json", {
            "task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
            "sampler": "zero_fill", "packet_sha256": identity["packet_sha256"],
            "fresh_update_sha256": identity["fresh_update_sha256"], "truth_used": False,
        })
        cpa = run_upstream_cpa(Y_hat=zero, n_components=N_IMAGES, device=device,
                               config=cfg["stage1"], output_dir=folder / "cpa", seed=0)
        sources = cpa.pop("S_hat").detach().cpu().float().contiguous()
        if sources.shape != (N_IMAGES, FC1_SHAPE[1]) or not torch.isfinite(sources).all():
            raise RuntimeError("Fresh naive CPA did not return all16 finite sources")
        _save_checkpoint(folder / "sources.pt", sources)
        write_json(folder / "cpa_log.json", cpa)
        cpa_marker = _save_marker(folder / "cpa_result.json", identity,
                                   ["sources.pt", "zero_fill_audit.json", "cpa_log.json"],
                                   method="naive_zero_fill", candidate=None,
                                   completed_iterations=25000, truth_used=False,
                                   wall_time_sec=cpa["wall_time_sec"])
        del zero, cpa, sources
        if device.type == "cuda":
            torch.cuda.empty_cache()
    sources = torch.load(folder / "sources.pt", map_location="cpu", weights_only=True)
    fia_marker = _checked_marker(folder / "fia_result.json", identity)
    if fia_marker is None:
        update_status_fn("fresh_naive_FIA", None, 25000)
        fia = run_staged_fia(target_z=sources.abs().contiguous(), device=device,
                             config=cfg["fia"], output_dir=folder / "fia", seed=42, stop=25000)
        source_rec = fia.pop("rec").detach().cpu().float().contiguous()
        if source_rec.shape != (N_IMAGES, 3, 224, 224) or not torch.isfinite(source_rec).all():
            raise RuntimeError("Fresh naive FIA did not return all16 finite reconstructions")
        _save_checkpoint(folder / "source_order_rec.pt", source_rec)
        write_json(folder / "fia_log.json", fia)
        fia_marker = _save_marker(folder / "fia_result.json", identity,
                                  ["source_order_rec.pt", "fia_log.json", "sources.pt", "cpa_result.json"],
                                  method="naive_zero_fill", completed_iterations=25000,
                                  truth_used=False, wall_time_sec=fia["wall_time_sec"])
        del fia, source_rec
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"identity": identity, "method": "naive_zero_fill", "folder": str(folder),
            "sources_path": str(folder / "sources.pt"), "rec_path": str(folder / "source_order_rec.pt"),
            "attack_complete": True}


def _run_random_candidate(cfg: dict[str, Any], identity: dict[str, Any], packet_manifest: dict[str, Any],
                          folder: Path, candidate: int, stop: int, device: torch.device,
                          update_status_fn) -> dict[str, Any]:
    if candidate not in CANDIDATES:
        raise ValueError("Fresh v4 candidate IDs are exactly 0..7")
    phase = "screen" if stop == 5000 else "final"
    folder.mkdir(parents=True, exist_ok=True)
    marker_path = folder / f"{phase}.json"
    previous = _checked_marker(marker_path, identity)
    if previous is not None:
        return previous
    packet = _verify_packet(packet_manifest)
    percent = int(identity["keep_percent"])
    cpa_marker = _checked_marker(folder / "cpa_result.json", identity)
    if cpa_marker is None:
        update_status_fn(f"random_candidate{candidate:02d}_CPA", candidate, stop)
        matrix, fill_audit = random_fill_packet(packet, candidate_seed(0, candidate))
        fill_audit.update({
            "task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
            "keep_percent": percent, "source_codec": identity["source_codec"],
            "generic_recompression_codec": "topk_global_bitmap_fp32_v1",
            "fresh_update_sha256": identity["fresh_update_sha256"], "truth_used": False,
        })
        write_json(folder / "fill_audit.json", fill_audit)
        cpa = run_upstream_cpa(Y_hat=matrix, n_components=N_IMAGES, device=device,
                               config=cfg["stage1"], output_dir=folder / "cpa", seed=0)
        sources = cpa.pop("S_hat").detach().cpu().float().contiguous()
        if sources.shape != (N_IMAGES, FC1_SHAPE[1]) or not torch.isfinite(sources).all():
            raise RuntimeError("Fresh random CPA did not return all16 finite sources")
        _save_checkpoint(folder / "sources.pt", sources)
        write_json(folder / "cpa_log.json", cpa)
        cpa_marker = _save_marker(folder / "cpa_result.json", identity,
                                   ["sources.pt", "fill_audit.json", "cpa_log.json"],
                                   method="uniform_random_fill", candidate=candidate,
                                   fill_seed=candidate_seed(0, candidate), completed_iterations=25000,
                                   truth_used=False, wall_time_sec=cpa["wall_time_sec"])
        del matrix, fill_audit, cpa, sources
        if device.type == "cuda":
            torch.cuda.empty_cache()
    sources = torch.load(folder / "sources.pt", map_location="cpu", weights_only=True)
    update_status_fn(f"random_candidate{candidate:02d}_FIA_{phase}", candidate, stop)
    fia = run_staged_fia(target_z=sources.abs().contiguous(), device=device,
                         config=cfg["fia"], output_dir=folder / "fia", seed=42, stop=stop)
    rec = fia.pop("rec").detach().cpu().float().contiguous()
    if rec.shape != (N_IMAGES, 3, 224, 224) or not torch.isfinite(rec).all():
        raise RuntimeError("Fresh random FIA did not return all16 finite reconstructions")
    _save_checkpoint(folder / f"{phase}_rec.pt", rec)
    write_json(folder / f"{phase}_fia_log.json", fia)
    update_status_fn(f"random_candidate{candidate:02d}_ranking_{phase}", candidate, stop)
    rank_started = time.perf_counter()
    score = consistency_score(rec=rec, packet=packet, keep_percent=percent, device=device)
    result = _save_marker(marker_path, identity,
                          [f"{phase}_rec.pt", f"{phase}_fia_log.json", "sources.pt", "cpa_result.json"],
                          method="uniform_random_fill", candidate=candidate, batch_index=0,
                          completed_iterations=stop, fill_seed=candidate_seed(0, candidate), **score,
                          ranking_wall_time_sec=time.perf_counter() - rank_started,
                          fia_optimizer_wall_time_sec=fia["optimizer_wall_time_sec"],
                          completed_at_utc=_now())
    del rec, sources, fia, packet
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _run_random_selection(cfg: dict[str, Any], identity: dict[str, Any], packet_manifest: dict[str, Any],
                          folder: Path, device: torch.device, update_status_fn) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    screens = [_run_random_candidate(cfg, identity, packet_manifest,
                                     folder / f"candidate{candidate:02d}", candidate, 5000,
                                     device, update_status_fn) for candidate in CANDIDATES]
    ranked = rank_candidates(screens, list(CANDIDATES))
    advanced = [int(row["candidate"]) for row in ranked[:3]]
    selection = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
        "keep_percent": identity["keep_percent"], "candidate_ids": list(CANDIDATES),
        "ranked_candidates": [int(row["candidate"]) for row in ranked],
        "advanced": advanced,
        "scores": [{"candidate": int(row["candidate"]), "relative_l2": float(row["relative_l2"])} for row in ranked],
        "rule": FRESH_RANDOMIZATION["ranking"], "truth_used": False, "metric_selection": False,
    }
    saved = _checked_marker(folder / "screen_selection.json", identity)
    if saved is not None:
        for key, value in selection.items():
            if saved.get(key) != value:
                raise RuntimeError("Fresh screen selection changed")
    else:
        _save_marker(folder / "screen_selection.json", identity,
                     [f"candidate{candidate:02d}/screen.json" for candidate in CANDIDATES], **selection)
    print(f"FRESH SCREEN COMPLETE keep={identity['keep_percent']} advanced={advanced}", flush=True)
    finals = [_run_random_candidate(cfg, identity, packet_manifest,
                                    folder / f"candidate{candidate:02d}", candidate, 25000,
                                    device, update_status_fn) for candidate in advanced]
    final_ranked = rank_candidates(finals, advanced)
    selected = int(final_ranked[0]["candidate"])
    final_selection = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
        "keep_percent": identity["keep_percent"], "candidate_ids": list(CANDIDATES),
        "selected_candidate": selected,
        "ranked_candidates": [int(row["candidate"]) for row in final_ranked],
        "scores": [{"candidate": int(row["candidate"]), "relative_l2": float(row["relative_l2"])} for row in final_ranked],
        "truth_used": False, "metric_selection": False,
        "continuation": "screen checkpoint 5000 resumed to total 25000; additional 20000",
    }
    saved_final = _checked_marker(folder / "final_selection.json", identity)
    if saved_final is not None:
        for key, value in final_selection.items():
            if saved_final.get(key) != value:
                raise RuntimeError("Fresh final selection changed")
    else:
        _save_marker(folder / "final_selection.json", identity,
                     [f"candidate{candidate:02d}/final.json" for candidate in advanced], **final_selection)
    return {"selected_candidate": selected, "advanced": advanced, "screen": screens, "final": finals,
            "selection": selection, "final_selection": final_selection}


def _evaluate_method(identity: dict[str, Any], truth: dict[str, Any], method: str, folder: Path,
                     source_path: Path, rec_path: Path, output_path: Path, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, float], torch.Tensor]:
    sources = torch.load(source_path, map_location="cpu", weights_only=True)
    source_recs = torch.load(rec_path, map_location="cpu", weights_only=True)
    aligned_recs, correspondence = evaluate_after_fia(sources, source_recs, truth["Z"])
    metrics = evaluate_images(x_true=truth["x"], rec=aligned_recs.to(device), device=device)
    rows = []
    for index in range(N_IMAGES):
        row = {
            **truth["metadata"][index],
            **correspondence[index],
            "task_id": TASK_ID,
            "plan_version": PLAN_VERSION,
            "amendment_id": AMENDMENT_ID,
            "keep_percent": identity["keep_percent"],
            "method": method,
            "selected_candidate": identity.get("selected_candidate"),
            "lpips": float(metrics.lpips[index]),
            "psnr": float(metrics.psnr[index]),
            "ssim": float(metrics.ssim[index]),
            "truth_used_for_evaluation_only": True,
        }
        rows.append(row)
    write_csv(output_path, rows)
    _atomic_torch_save(folder / ("all16_rec.pt" if method == "uniform_random_fill" else "naive_all16_rec.pt"), aligned_recs)
    return rows, _metrics(rows), aligned_recs


def _render_contact_sheet(original: torch.Tensor, naive: torch.Tensor, random: torch.Tensor,
                          random_rows: list[dict[str, Any]], naive_rows: list[dict[str, Any]],
                          path: Path, indices: list[int], title: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    tile, gap, header, label_height, columns = 160, 10, 42, 26, 4
    cell_width = 3 * tile + 2 * gap
    cell_height = tile + label_height
    rows = (len(indices) + columns - 1) // columns
    canvas = Image.new("RGB", (gap + columns * (cell_width + gap), header + gap + rows * (cell_height + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
        small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 11)
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((gap, 10), title, fill="black", font=font)
    truth_images = denormalize_imagenet(original).cpu().clamp(0, 1)
    for pos, index in enumerate(indices):
        left = gap + (pos % columns) * (cell_width + gap)
        top = header + gap + (pos // columns) * (cell_height + gap)
        images = (truth_images[index], naive[index].cpu().clamp(0, 1), random[index].cpu().clamp(0, 1))
        labels = ("Original", "Naive", "CoRA8")
        for col, (image, label) in enumerate(zip(images, labels)):
            array = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            canvas.paste(Image.fromarray(array).resize((tile, tile), Image.Resampling.LANCZOS),
                         (left + col * (tile + gap), top))
            draw.text((left + col * (tile + gap), top + tile + 3), label, fill="black", font=small)
        draw.text((left, top + tile + 15),
                  f"g{int(random_rows[index]['global_index']):02d} R LPIPS {float(random_rows[index]['lpips']):.3f} N {float(naive_rows[index]['lpips']):.3f}",
                  fill="#334155", font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _finalize_fraction(cfg: dict[str, Any], manifest: dict[str, Any], identity: dict[str, Any],
                       packet_manifest: dict[str, Any], selection: dict[str, Any],
                       naive_attack: dict[str, Any], fraction: Path, truth_record: dict[str, Any],
                       device: torch.device) -> dict[str, Any]:
    summary_path = fraction / "summary.json"
    existing = _checked_marker(summary_path, identity)
    if existing is not None:
        return existing
    truth = _load_truth(truth_record)
    selected = int(selection["selected_candidate"])
    random_folder = fraction / "random"
    random_candidate = random_folder / f"candidate{selected:02d}"
    random_rows, random_metrics, random_rec = _evaluate_method(
        {**identity, "selected_candidate": selected}, truth, "uniform_random_fill", fraction,
        random_candidate / "sources.pt", random_candidate / "final_rec.pt",
        fraction / "per_image_metrics_random.csv", device)
    naive_rows, naive_metrics, naive_rec = _evaluate_method(
        identity, truth, "naive_zero_fill", fraction, fraction / "naive/sources.pt",
        fraction / "naive/source_order_rec.pt", fraction / "per_image_metrics_naive.csv", device)
    comparison_rows = []
    for index in range(N_IMAGES):
        comparison_rows.append({
            **truth["metadata"][index],
            "keep_percent": identity["keep_percent"],
            "random_candidate": selected,
            "random_lpips": random_rows[index]["lpips"],
            "random_psnr": random_rows[index]["psnr"],
            "random_ssim": random_rows[index]["ssim"],
            "naive_lpips": naive_rows[index]["lpips"],
            "naive_psnr": naive_rows[index]["psnr"],
            "naive_ssim": naive_rows[index]["ssim"],
            "random_minus_naive_lpips": random_rows[index]["lpips"] - naive_rows[index]["lpips"],
            "random_minus_naive_psnr": random_rows[index]["psnr"] - naive_rows[index]["psnr"],
            "random_minus_naive_ssim": random_rows[index]["ssim"] - naive_rows[index]["ssim"],
        })
    write_csv(fraction / "per_image_comparison.csv", comparison_rows)
    _render_contact_sheet(truth["x"], naive_rec, random_rec, random_rows, naive_rows,
                          fraction / "all16_original_vs_naive_vs_cora8.png", list(range(N_IMAGES)),
                          f"Fresh16 keep{identity['keep_percent']}% | original / naive / CoRA8")
    _render_contact_sheet(truth["x"], naive_rec, random_rec, random_rows, naive_rows,
                          fraction / "fixed_index0_1_original_vs_naive_vs_cora8.png", [0, 1],
                          f"Fresh16 keep{identity['keep_percent']}% | fixed global indices 0 and 1")
    delta = {
        "lpips_mean_random_minus_naive": random_metrics["lpips_mean"] - naive_metrics["lpips_mean"],
        "psnr_mean_random_minus_naive": random_metrics["psnr_mean"] - naive_metrics["psnr_mean"],
        "ssim_mean_random_minus_naive": random_metrics["ssim_mean"] - naive_metrics["ssim_mean"],
    }
    report = [
        f"# Fresh16 paper sparse comparison: keep {identity['keep_percent']}%",
        "",
        "This report uses a newly collected B16/T10 VGG16 classifier update on original batch00 global indices 0..15. Naive zero fill and the selected CoRA8 candidate are evaluated on those same fresh per-image rows.",
        "",
        "| Method | LPIPS (lower) | PSNR dB (higher) | SSIM (higher) |",
        "|---|---:|---:|---:|",
        f"| Fresh naive zero fill | {naive_metrics['lpips_mean']:.6f} | {naive_metrics['psnr_mean']:.4f} | {naive_metrics['ssim_mean']:.6f} |",
        f"| Fresh CoRA8 candidate {selected} | {random_metrics['lpips_mean']:.6f} | {random_metrics['psnr_mean']:.4f} | {random_metrics['ssim_mean']:.6f} |",
        "",
        f"Selection used only the public recompressed-update relative-L2 score; candidate {selected} was selected after screen5k and top3 continuation to25k.",
        f"Random minus naive: LPIPS {delta['lpips_mean_random_minus_naive']:+.6f}, PSNR {delta['psnr_mean_random_minus_naive']:+.4f} dB, SSIM {delta['ssim_mean_random_minus_naive']:+.6f}.",
        "",
        f"The fresh packet is `{packet_manifest['packet_bytes']:,}` bytes and is shared byte-for-byte by naive and all eight candidates. No historical update, packet, CPA/FIA checkpoint, source, score, reconstruction or metric is an input.",
        "",
        "![Fresh16 contact sheet](all16_original_vs_naive_vs_cora8.png)",
        "",
        "![Fixed examples 0 and 1](fixed_index0_1_original_vs_naive_vs_cora8.png)",
        "",
    ]
    (fraction / "report.md").write_text("\n".join(report), encoding="utf-8")
    names = [
        "per_image_metrics_random.csv", "per_image_metrics_naive.csv", "per_image_comparison.csv",
        "all16_rec.pt", "naive_all16_rec.pt", "all16_original_vs_naive_vs_cora8.png",
        "fixed_index0_1_original_vs_naive_vs_cora8.png", "report.md", "random/screen_selection.json",
        "random/final_selection.json", "packet/observation.bin", "packet/packet_manifest.json",
    ]
    result = {
        "schema_version": 4,
        "task_id": TASK_ID,
        "plan_version": PLAN_VERSION,
        "amendment_id": AMENDMENT_ID,
        "status": "complete",
        "identity": identity,
        "n_images": N_IMAGES,
        "batch_indices": [0],
        "keep_percent": identity["keep_percent"],
        "candidate_ids": list(CANDIDATES),
        "selected_candidate": selected,
        "advanced_candidates": selection["advanced"],
        "screen_selection": selection,
        "final_selection": selection["final_selection"],
        "naive_metrics": naive_metrics,
        "random_metrics": random_metrics,
        "quality_delta_random_minus_naive": delta,
        "packet": packet_manifest,
        "fresh_update_sha256": truth_record["update_sha256"],
        "fresh_truth_sha256": truth_record["truth_sha256"],
        "truth_used_for_selection": False,
        "attack_artifacts_imported": False,
        "budget": {
            "naive_cpa_trajectories": 1,
            "random_cpa_trajectories": 8,
            "cpa_iterations": 9 * 25000,
            "random_fia_screen_iterations": 8 * 5000,
            "random_fia_continuation_iterations": 3 * 20000,
            "naive_fia_iterations": 25000,
            "ranking_replays": 11,
        },
        "completed_at_utc": _now(),
        "output_hashes": {name: file_hash(fraction / name) for name in names},
    }
    write_json(summary_path, result)
    return result


def _run_fraction(cfg: dict[str, Any], manifest: dict[str, Any], packets: dict[int, dict[str, Any]],
                  update_record: dict[str, Any], output: Path, private: Path,
                  device: torch.device) -> dict[str, Any]:
    percent = int(cfg["keep_percent"])
    packet_manifest = packets[percent]
    identity = _fraction_identity(manifest["identity"], percent, packet_manifest)
    fraction = output / f"keep{percent}"
    summary = _checked_marker(fraction / "summary.json", identity)
    if summary is not None:
        return summary
    random_folder = fraction / "random"

    def update_stage(stage: str, candidate: int | None, stop: int) -> None:
        _status(output, cfg, state="running", stage=stage, keep_percent=percent,
                candidate=candidate, fia_stop=stop, fia_horizon=25000,
                truth_used_for_selection=False)

    selection = _run_random_selection(cfg, identity, packet_manifest, random_folder, device, update_stage)
    naive = _run_naive_attack(cfg, identity, packet_manifest, fraction / "naive", device, update_stage)
    _status(output, cfg, state="evaluating_after_public_selection", keep_percent=percent,
            selected_candidate=selection["selected_candidate"], truth_used_for_selection=False)
    result = _finalize_fraction(cfg, manifest, identity, packet_manifest, selection, naive,
                                fraction, update_record, device)
    print(f"FRESH COMPLETE keep={percent} selected={result['selected_candidate']}", flush=True)
    return result


def _write_combined_report(cfg: dict[str, Any], manifest: dict[str, Any], output: Path,
                           private: Path, fraction_results: list[dict[str, Any]], truth_record: dict[str, Any]) -> dict[str, Any]:
    truth = _load_truth(truth_record)
    fraction_results = sorted(fraction_results, key=lambda row: int(row["keep_percent"]), reverse=True)
    combined = []
    per_image = []
    columns: list[tuple[str, torch.Tensor]] = [("Original", denormalize_imagenet(truth["x"]).cpu())]
    for result in fraction_results:
        percent = int(result["keep_percent"])
        fraction = output / f"keep{percent}"
        random_rows = read_csv(fraction / "per_image_metrics_random.csv")
        naive_rows = read_csv(fraction / "per_image_metrics_naive.csv")
        for method, values in (("naive_zero_fill", result["naive_metrics"]), ("uniform_random_fill_cora8", result["random_metrics"])):
            combined.append({
                "task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
                "keep_percent": percent, "n_images": N_IMAGES, "batch_indices": "[0]",
                "method": method, "selected_candidate": result["selected_candidate"] if "random" in method else "",
                "lpips_mean": values["lpips_mean"], "psnr_mean": values["psnr_mean"], "ssim_mean": values["ssim_mean"],
                "packet_bytes": result["packet"]["packet_bytes"], "fresh_update_sha256": result["fresh_update_sha256"],
            })
        per_image.extend([{**row, "condition": "random", "keep_percent": percent} for row in random_rows])
        per_image.extend([{**row, "condition": "naive", "keep_percent": percent} for row in naive_rows])
        naive_rec = torch.load(fraction / "naive_all16_rec.pt", map_location="cpu", weights_only=True)
        random_rec = torch.load(fraction / "all16_rec.pt", map_location="cpu", weights_only=True)
        columns.extend([(f"{percent}% Naive", naive_rec[:2]), (f"{percent}% CoRA8", random_rec[:2])])
    write_csv(output / "combined_comparison_metrics.csv", combined)
    write_csv(output / "combined_per_image_metrics.csv", per_image)
    _render_multi_sheet(columns, output / "fixed_index0_1_comparison.png",
                        "Fresh16 fixed illustration identities global_index 0 and 1")
    table = [
        f"# CPA paper sparse fresh16 sweep ({TASK_ID}, v4/R8)",
        "",
        "The v4 experiment freshly collected one B16/T10 update from original paper batch00 global indices 0..15, encoded new keep50/30/20 packets, and ran one fresh naive arm plus eight fresh CoRA candidates per fraction. The three public top3 selections continued from screen5k to total25k.",
        "",
        "| Retained | Method | LPIPS | PSNR dB | SSIM | Selected candidate |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in combined:
        candidate = row["selected_candidate"] if row["selected_candidate"] != "" else "—"
        table.append(f"| {row['keep_percent']}% | {row['method']} | {float(row['lpips_mean']):.6f} | {float(row['psnr_mean']):.4f} | {float(row['ssim_mean']):.6f} | {candidate} |")
    table += [
        "",
        "All values are fresh16 means over the same original per-image rows. Public ranking used only the recompressed-update relative-L2 score and public round0 labels; true labels, fresh images and fresh features were loaded only for evaluation after final selection.",
        "",
        f"Fresh update manifest: `{private / 'fresh_update_manifest.json'}`.",
        f"Combined metrics: `{output / 'combined_comparison_metrics.csv'}`.",
        f"All fresh per-image rows: `{output / 'combined_per_image_metrics.csv'}`.",
        f"Fixed examples 0 and 1: `{output / 'fixed_index0_1_comparison.png'}`.",
        "",
    ]
    report_path = output / "report.md"
    report_path.write_text("\n".join(table), encoding="utf-8")
    result = {
        "schema_version": 4,
        "task_id": TASK_ID,
        "plan_version": PLAN_VERSION,
        "amendment_id": AMENDMENT_ID,
        "status": "complete",
        "n_images": N_IMAGES,
        "batch_indices": [0],
        "keep_percents": list(PERCENTS),
        "candidate_ids": list(CANDIDATES),
        "fresh_update_collection": str(private / "fresh_update_manifest.json"),
        "fractions": [{"keep_percent": row["keep_percent"], "summary": str(output / f"keep{row['keep_percent']}" / "summary.json"),
                       "summary_sha256": file_hash(output / f"keep{row['keep_percent']}" / "summary.json"),
                       "selected_candidate": row["selected_candidate"]} for row in fraction_results],
        "output_hashes": {
            "combined_comparison_metrics.csv": file_hash(output / "combined_comparison_metrics.csv"),
            "combined_per_image_metrics.csv": file_hash(output / "combined_per_image_metrics.csv"),
            "fixed_index0_1_comparison.png": file_hash(output / "fixed_index0_1_comparison.png"),
            "report.md": file_hash(report_path),
        },
        "completed_at_utc": _now(),
    }
    write_json(output / "summary.json", result)
    return result


def _render_multi_sheet(columns: list[tuple[str, torch.Tensor]], path: Path, title: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    tile, gap, header = 224, 12, 52
    widths = len(columns) * tile + (len(columns) + 1) * gap
    canvas = Image.new("RGB", (widths, header + 2 * (tile + 34 + gap) + gap), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
        small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((gap, 14), title, fill="black", font=font)
    for row_index in range(2):
        top = header + gap + row_index * (tile + 34 + gap)
        for index, (label, tensor) in enumerate(columns):
            x = gap + index * (tile + gap)
            image = tensor[row_index].cpu().clamp(0, 1)
            array = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            canvas.paste(Image.fromarray(array).resize((tile, tile), Image.Resampling.LANCZOS), (x, top))
            draw.text((x, top + tile + 8), f"g{row_index} {label}", fill="black", font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def run_all(config_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    output = resolve(cfg["output_dir"])
    private = resolve(cfg["private_output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    with execution_lock(output):
        manifest = preflight(config_path, cfg, output, private)
        _status(output, cfg, state="running", stage="fresh_B16_T10_update_collection",
                new_update_collection=True, truth_used_for_selection=False)
        device = torch.device(cfg["device"])
        update_record = _collect_fresh_update(cfg, manifest, private, device)
        Y = _load_update(update_record)
        packets = _packet_setup(cfg, manifest, update_record, Y, output)
        root_manifest = dict(manifest)
        root_manifest.update({
            "fresh_update_manifest": str(private / "fresh_update_manifest.json"),
            "fresh_update_manifest_sha256": file_hash(private / "fresh_update_manifest.json"),
            "fresh_update_sha256": update_record["update_sha256"],
            "fresh_truth_sha256": update_record["truth_sha256"],
            "packet_manifests": packets,
            "packets_created_at_utc": _now(),
        })
        write_json(output / "fresh16_manifest.json", root_manifest)
        results = []
        for percent in PERCENTS:
            local_cfg = dict(cfg)
            local_cfg["keep_percent"] = percent
            results.append(_run_fraction(local_cfg, manifest, packets, update_record,
                                          output, private, device))
        _status(output, cfg, state="writing_combined_report", truth_used_for_selection=False)
        combined = _write_combined_report(cfg, manifest, output, private, results, update_record)
        _status(output, cfg, state="complete", metrics={
            str(row["keep_percent"]): {"naive": row["naive_metrics"], "random": row["random_metrics"],
                                       "selected_candidate": row["selected_candidate"]} for row in results},
                combined_report_sha256=combined["output_hashes"]["report.md"],
                fresh_update_sha256=update_record["update_sha256"],
                truth_used_for_selection=False)
        return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/cpa_fia_fresh16_R8_S5k_K3_F25k.json")
    parser.add_argument("--phase", choices=("validate", "run", "report"), default="run")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    cfg = read_json(config_path)
    validate_config(cfg, config_path)
    output = resolve(cfg["output_dir"])
    private = resolve(cfg["private_output_dir"])
    try:
        if args.phase == "validate":
            manifest = preflight(config_path, cfg, output, private)
            print(json.dumps({"status": "validated", "input_manifest": str(output / "input_manifest.json"),
                              "identity": manifest["identity"]}, ensure_ascii=False), flush=True)
            return 0
        if args.phase == "report":
            root = read_json(output / "fresh16_manifest.json")
            update = read_json(private / "fresh_update_manifest.json")
            results = [read_json(output / f"keep{percent}/summary.json") for percent in PERCENTS]
            combined = _write_combined_report(cfg, root, output, private, results, update)
            print(json.dumps(combined, ensure_ascii=False), flush=True)
            return 0
        torch.set_num_threads(4)
        result = run_all(config_path, cfg)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except BaseException as exc:
        output.mkdir(parents=True, exist_ok=True)
        _status(output, cfg, state="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
