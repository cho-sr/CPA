"""Prepare, run and publicly gate the locked same80 8/7/6/5/4-bit screen."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from . import signed_bits_observation as established_codec
from . import signed_bits_randomized as established_sampler
from . import signed_int8_screen3_fia as screen_fia
from . import staged_fia
from . import upstream_naive
from .same80_bits_r100_core import (
    BITS, CHECKPOINT_SHA256, FC1_SHAPE, FIA, MIN_FREE_BYTES, N_BATCHES,
    N_CANDIDATES, N_IMAGES, PLAN_VERSION, PREFIX_STEPS, PUBLIC_SCORE,
    SAMPLER_CHUNK_ROWS, STAGE1, TASK_ID, TOP_K, TOTAL_STEPS, TaskError,
    StreamingTop3, candidate_seed, cell_bounds_table, codec_metadata,
    code_identity, encode_tensor, ensure_free_space, execution_lock,
    expected_budget, file_sha256, qmax_for_bits, rank_full,
    read_json, remove_task_disposable, sample_matrix, save_marker,
    tensor_sha256, utc_now, verify_gate_counts, verify_hash_map,
    verify_identity, verify_marker, write_json, write_tensor,
)


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
PUBLIC_ROOT = PROJECT / "results/same80_bits87654_R100_S5k_K3_F25k"
PRIVATE_ROOT = PROJECT / "results_private/same80_bits87654_R100_S5k_K3_F25k"
CHECKPOINT_PATH = Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth"
SOURCE_ROOT = PROJECT / "results/m6_uia_fp32_B16_fia"
SAME80_IMAGE_MANIFEST = PROJECT / "results/cpa_fia_topk50_same80_B16/image_manifest.csv"
REPAIR_HANDOFF_SHA256 = "56fb37ce5f3bc06dc646782880b177ecd213bb49d1d75dedc87c9df258a738e5"
REPAIR_TRANSITION_PATH = WORKSPACE / ".ai/relay-runs/supervisors" / TASK_ID / "reviewed_repair_transition.json"
REPAIR_TRANSITION_SCHEMA = "CPA_SAME80_REVIEWED_REPAIR_TRANSITION_V1"
LEGACY_FIA_ATTESTATION_SCHEMA = "CPA_SAME80_REVIEWED_LEGACY_FIA_V1"

STAGE1_CONFIG = {
    "n_components": 16, "cpa_ica_max_iter": 25000, "cpa_ica_lr": 0.001,
    "cpa_ica_optimizer": "adam", "cpa_ica_scheduler": "none",
    "cpa_ica_decor": 5.3, "cpa_ica_T": 7.7, "cpa_ica_ne": 1.0,
    "cpa_ica_nv": 0.13, "cpa_ica_l1": 5.0, "cpa_ica_tv": 0.1,
    "cpa_ica_log_every": 5000, "debias": False, "mp_truncation": False,
    "refine_A": False, "stage2": False,
}


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def _validate_config(cfg: dict[str, Any]) -> None:
    expected = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION,
        "bits": list(BITS), "batch_indices": list(range(N_BATCHES)),
        "source_root": "results/m6_uia_fp32_B16_fia",
        "image_manifest": "results/cpa_fia_topk50_same80_B16/image_manifest.csv",
        "image_manifest_sha256": "d09d274c22f5a5c0b6d5a2bcb73d464d81eef85a6af543d3c3b0289128a0fda4",
        "checkpoint_path": str(CHECKPOINT_PATH), "checkpoint_sha256": CHECKPOINT_SHA256,
        "public_root": "results/same80_bits87654_R100_S5k_K3_F25k",
        "private_root": "results_private/same80_bits87654_R100_S5k_K3_F25k",
        "device": "cuda", "n": 16, "T": 10,
        "local_training": {"scope": "conv_frozen_all_fc", "round": 0, "optimizer": "sgd",
                            "lr": 0.001, "steps": 10, "momentum": 0.0,
                            "weight_decay": 0.0, "dropout": "disabled",
                            "preprocessing": "existing_same80_imagenet"},
        "observation": {"layer": "classifier.0.weight", "shape": list(FC1_SHAPE),
                        "field": "observation.Y_delta_over_lr",
                        "definition": "(W_initial-W_after10)/0.001"},
        "codec": {"family": "signed_symmetric_per_tensor_absmax_fp32_v1",
                  "rounding": "nearest_even", "bits": list(BITS),
                  "zero_tensor": "q=0,s=1", "physical_code_storage": "int8",
                  "packed_bandwidth_claim": False},
        "sampler": {"name": "finite_fp32_registered_cells_extremal_anchor_v1",
                    "chunk_rows": SAMPLER_CHUNK_ROWS, "candidate_count": N_CANDIDATES,
                    "candidate_ids": list(range(N_CANDIDATES)),
                    "seed": "1234+1000003*(100*batch_index+candidate_index)",
                    "shared_across_bits": True},
        "stage1": STAGE1_CONFIG,
        "fia": {**FIA, "public_screen_score": PUBLIC_SCORE,
                "tie_break": "candidate_index_ascending",
                "source_order": "estimated_source_order_abs_only",
                "private_scaling_or_alignment_before_FIA": False},
        "selection": {"screen_top_k": TOP_K, "final_top_k": 1,
                       "screen_score": PUBLIC_SCORE, "final_score": PUBLIC_SCORE,
                       "final_candidate_unit": "whole16_images",
                       "private_metrics_used": False},
        "resource": {"workers": 1, "minimum_free_disk_bytes": MIN_FREE_BYTES,
                     "first_measurement": "first_scheduled_naive_candidate_no_extra_pilot",
                     "estimate_uncertainty_fraction": 0.5},
    }
    if cfg != expected:
        raise ValueError("Configuration differs from the approved immutable v1 contract")


def _validate_parent_code_manifest(parent: dict[str, Any]) -> None:
    mapping = {
        "run_m6_uia_fp32_fia.py": PROJECT / "qacpa/run_m6_uia_fp32_fia.py",
        "uia.py": PROJECT / "qacpa/uia.py",
        "victim.py": PROJECT / "qacpa/victim.py",
        "stage1.py": PROJECT / "qacpa/stage1.py",
        "fia.py": PROJECT / "qacpa/fia.py",
        "alignment.py": PROJECT / "qacpa/alignment.py",
        "seed.py": PROJECT / "qacpa/seed.py",
    }
    for name, expected in parent["identity"]["code_sha256"].items():
        path = mapping.get(name)
        if path is None or file_sha256(path) != expected:
            raise TaskError(f"historical m6 code identity changed: {name}")


def _image_manifest_rows(path: Path, expected_sha256: str) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected_sha256:
        raise TaskError("original same80 image manifest hash mismatch")
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 80 or [int(row["global_index"]) for row in rows] != list(range(80)):
        raise TaskError("same80 manifest must preserve original global indices 0..79")
    for index, row in enumerate(rows):
        if int(row["batch_index"]) != index // 16 or int(row["source_index"]) != index % 16:
            raise TaskError("same80 manifest batch membership/order changed")
    if len({row["image_path"] for row in rows}) != 80:
        raise TaskError("same80 manifest contains duplicate source images")
    return rows


def _load_input_metadata(path: Path, expected_sha: str) -> list[dict[str, Any]]:
    if file_sha256(path) != expected_sha:
        raise TaskError(f"historical input hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, list) or len(metadata) != N_IMAGES:
        raise TaskError("cached source metadata is not one original B16")
    return metadata


def _verify_source_artifacts(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, Any]]]:
    root = _resolve_project_path(cfg["source_root"])
    parent = read_json(root / "input_manifest.json")
    if parent.get("run_tag") != "m6_uia_fp32_B16_fia" or parent.get("n_images") != 80:
        raise TaskError("unexpected historical m6 input manifest")
    _validate_parent_code_manifest(parent)
    manifest_path = _resolve_project_path(cfg["image_manifest"])
    base_manifest_path = PROJECT / "results/cpa_fia_topk50_same80_B16/input_manifest.json"
    base_manifest = read_json(base_manifest_path)
    if base_manifest.get("image_manifest_sha256") != cfg["image_manifest_sha256"]:
        raise TaskError("historical same80 manifest registration changed")
    image_rows = _image_manifest_rows(manifest_path, cfg["image_manifest_sha256"])
    result_rows: list[dict[str, Any]] = []
    prepared = parent.get("prepared_files")
    if not isinstance(prepared, list) or len(prepared) != N_BATCHES:
        raise TaskError("historical m6 manifest must authenticate five input caches")
    for batch, record in enumerate(prepared):
        input_path = root / f"batch{batch:02d}/inputs.pt"
        observation_path = root / f"batch{batch:02d}/observation.pt"
        input_meta_path = input_path.with_name("inputs.meta.json")
        observation_meta_path = observation_path.with_name("observation.meta.json")
        input_meta, observation_meta = read_json(input_meta_path), read_json(observation_meta_path)
        batch_summary = read_json(root / f"batch{batch:02d}/summary.json")
        stage_meta = read_json(root / f"batch{batch:02d}/stage1.meta.json")
        input_sha = record["sha256"]
        if Path(record["path"]).resolve() != input_path.resolve():
            raise TaskError("historical input path differs from its registered batch")
        if file_sha256(input_path) != input_sha or input_meta.get("file_sha256") != input_sha:
            raise TaskError(f"input cache hash mismatch for batch{batch:02d}")
        if input_meta.get("identity") != {**parent["identity"], "batch_index": batch,
                                          "stage": "cached_input_audit"}:
            raise TaskError(f"input metadata identity mismatch for batch{batch:02d}")
        observation_sha = observation_meta.get("file_sha256")
        expected_stage_identity = {**parent["identity"], "batch_index": batch,
                                   "inputs_sha256": input_sha,
                                   "stage": "CPA_full16", "observation_sha256": observation_sha}
        expected_observation_identity = {
            **parent["identity"], "batch_index": batch, "inputs_sha256": input_sha,
            "stage": "actual_UIA_update",
        }
        if (not observation_sha or file_sha256(observation_path) != observation_sha
                or observation_meta.get("identity") != expected_observation_identity
                or batch_summary.get("identity") != {**parent["identity"], "batch_index": batch,
                                                       "inputs_sha256": input_sha}
                or batch_summary.get("output_hashes", {}).get("observation.pt") != observation_sha
                or batch_summary.get("output_hashes", {}).get("inputs.pt") != input_sha
                or stage_meta.get("identity") != expected_stage_identity
                or stage_meta.get("file_sha256") != batch_summary.get("output_hashes", {}).get("stage1.pt")):
            raise TaskError(f"actual update artifact/meta mismatch for batch{batch:02d}")
        batch_rows = _load_input_metadata(input_path, input_sha)
        if [int(row["global_index"]) for row in batch_rows] != list(range(batch * 16, (batch + 1) * 16)):
            raise TaskError(f"cached original identity/order changed in batch{batch:02d}")
        for local, row in enumerate(batch_rows):
            expected = image_rows[batch * 16 + local]
            for key in ("global_index", "batch_index", "source_index", "dataset_index", "image_path"):
                if str(row[key]) != str(expected[key]):
                    raise TaskError(f"same80 image identity differs at global index {batch*16+local}: {key}")
        result_rows.append({"batch_index": batch, "input_path": str(input_path),
                            "input_sha256": input_sha,
                            "observation_path": str(observation_path),
                            "observation_sha256": observation_sha,
                            "input_metadata_sha256": file_sha256(input_meta_path),
                            "observation_metadata_sha256": file_sha256(observation_meta_path),
                            "global_indices": list(range(batch * 16, (batch + 1) * 16))})
    checkpoint = _resolve_project_path(cfg["checkpoint_path"])
    if file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise TaskError("public VGG16 checkpoint hash mismatch")
    return parent, image_rows, result_rows


def _public_source_records(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in record.items()
             if key not in {"input_path", "observation_path"}}
            for record in sources]


def _load_actual_y(record: dict[str, Any], batch: int) -> torch.Tensor:
    payload = torch.load(record["observation_path"], map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or not {"observation", "observation_used_for_attack", "definition"}.issubset(payload):
        raise TaskError("cached update schema does not contain actual_observation provenance")
    if (payload["observation_used_for_attack"] != "Y_delta_over_lr"
            or payload["definition"] != "(W_initial-W_after_10_local_steps)/lr"):
        raise TaskError("actual observation sign/scale convention is not the approved one")
    observation = payload["observation"]
    value = observation.get("Y_delta_over_lr")
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or tuple(value.shape) != FC1_SHAPE or not bool(torch.isfinite(value).all())):
        raise TaskError(f"invalid actual classifier.0.weight observation in batch{batch:02d}")
    if observation.get("update_factor_rel_l2") is None:
        raise TaskError("source update-factor provenance is missing")
    return value.detach().cpu().contiguous()


def _write_packet(folder: Path, *, q: torch.Tensor, scale: torch.Tensor, bits: int,
                  source_hash: str, source_record: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    packet_path = folder / "codes_int8.pt"
    manifest_path = folder / "packet.json"
    expected = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "bits": bits,
        "batch_index": int(source_record["batch_index"]),
        "logical_code_bits": bits, "physical_code_storage": "int8",
        "codec_metadata": codec_metadata(bits), "shape": list(q.shape),
        "q_dtype": "int8", "scale_dtype": "float32", "scale_fp32_hex": scale.numpy().tobytes().hex(),
        "source_actual_observation_sha256": source_record["observation_sha256"],
        "source_Y_delta_over_lr_sha256": source_hash,
        "input_sha256": source_record["input_sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "registration_identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "code_sha256": identity["code_sha256"],
    }
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if saved.get("packet_identity") != expected or file_sha256(packet_path) != saved.get("codes_file_sha256"):
            raise TaskError(f"existing public packet cannot be authenticated: {folder}")
        loaded = torch.load(packet_path, map_location="cpu", weights_only=True)
        if not torch.equal(loaded, q):
            raise TaskError(f"existing codes differ from deterministic re-encoding: {folder}")
        return saved
    if packet_path.exists() and not packet_path.is_file():
        raise TaskError(f"invalid packet path: {packet_path}")
    from .same80_bits_r100_core import write_tensor
    write_tensor(packet_path, q)
    manifest = {"packet_identity": expected, "scale": float(scale),
                "scale_fp32_hex": expected["scale_fp32_hex"],
                "codes_file": packet_path.name, "codes_file_sha256": file_sha256(packet_path),
                "physical_code_bytes": q.numel(), "logical_code_bits": bits * q.numel(),
                "packed_bandwidth_claim": False}
    write_json(manifest_path, manifest)
    return manifest


def prepare(config_path: Path, cfg: dict[str, Any], public_root: Path = PUBLIC_ROOT,
            private_root: Path = PRIVATE_ROOT) -> dict[str, Any]:
    _validate_config(cfg)
    public_root, private_root = Path(public_root).resolve(), Path(private_root).resolve()
    if public_root == private_root or public_root in private_root.parents or private_root in public_root.parents:
        raise TaskError("public and private output roots must be disjoint")
    parent, image_rows, sources = _verify_source_artifacts(cfg)
    base = code_identity(config_path.resolve(), cfg)
    public_sources = _public_source_records(sources)
    identity = {**base, "source_input_manifest_sha256": file_sha256(SOURCE_ROOT / "input_manifest.json"),
                "same80_image_manifest_sha256": cfg["image_manifest_sha256"],
                "source_batches": public_sources,
                "source_observation_sign_scale": "actual_observation.Y_delta_over_lr=(W_initial-W_after10)/0.001"}
    if public_root.exists():
        registration_path = public_root / "registration.json"
        if not registration_path.is_file():
            # main() acquires the task lock before prepare(), which creates
            # these two bookkeeping files in a new root. Accept only that
            # empty locked root; arbitrary pre-existing output remains fatal.
            allowed = {".execution.lock", ".execution_owner.json"}
            unexpected = {path.name for path in public_root.iterdir()} - allowed
            lock_path = public_root / ".execution.lock"
            owner_path = public_root / ".execution_owner.json"
            owner = read_json(owner_path) if owner_path.is_file() and not owner_path.is_symlink() else {}
            if (unexpected or not lock_path.is_file() or lock_path.is_symlink()
                    or owner.get("pid") != os.getpid()
                    or Path(owner.get("root", "")).resolve() != public_root):
                raise TaskError("public output root exists without a task registration or current lock")
        else:
            existing = read_json(registration_path)
            verify_identity(existing.get("identity", {}), identity)
            if existing.get("status") == "PUBLIC_COMPLETED":
                raise TaskError("public selection is frozen; prepare cannot reopen a completed task")
    else:
        public_root.mkdir(parents=True)
    private_root.mkdir(parents=True, exist_ok=True)
    private_manifest_path = private_root / "private_manifest.json"
    if not private_manifest_path.is_file():
        write_json(private_root / "private_manifest.json", {
            "task_id": TASK_ID, "plan_version": PLAN_VERSION,
            "source_input_manifest": str(SOURCE_ROOT / "input_manifest.json"),
            "source_input_manifest_sha256": identity["source_input_manifest_sha256"],
            "same80_image_manifest": str(SAME80_IMAGE_MANIFEST),
            "same80_image_manifest_sha256": cfg["image_manifest_sha256"],
            "source_batches": sources, "n_images": 80,
            "private_values_loaded_by_attack_process": False,
        })
    else:
        private_manifest = read_json(private_manifest_path)
        if (private_manifest.get("task_id") != TASK_ID
                or private_manifest.get("source_input_manifest_sha256") != identity["source_input_manifest_sha256"]
                or private_manifest.get("same80_image_manifest_sha256") != cfg["image_manifest_sha256"]):
            raise TaskError("existing private task provenance differs from this registration")
    registration_path = public_root / "registration.json"
    if not registration_path.exists():
        write_json(registration_path, {
            "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "PREPARED",
            "identity": identity, "public_input_only": True, "private_data_used_by_attack": False,
            "checkpoint_path": str(_resolve_project_path(cfg["checkpoint_path"])),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "protocol": {"bits": list(BITS), "batches": list(range(N_BATCHES)),
                         "candidate_ids": list(range(N_CANDIDATES)), "candidate_count": N_CANDIDATES,
                         "sampler_chunk_rows": SAMPLER_CHUNK_ROWS, "selection_score": PUBLIC_SCORE},
            "expected_budget": expected_budget(), "packets": [], "private_evaluation": "NOT_STARTED",
            "created_at_utc": utc_now(),
        })
    registration = read_json(registration_path)
    verify_identity(registration.get("identity", {}), identity)
    packet_rows = []
    for record in sources:
        batch = int(record["batch_index"])
        y = _load_actual_y(record, batch)
        y_hash = tensor_sha256(y)
        for bits in BITS:
            ensure_free_space(public_root, reserve_bytes=MIN_FREE_BYTES)
            q, scale = encode_tensor(y, bits)
            if tuple(q.shape) != FC1_SHAPE or q.dtype != torch.int8 or not bool(torch.isfinite(scale)):
                raise TaskError("registered quantizer returned invalid codes/scale")
            qmax = qmax_for_bits(bits)
            if bool((q < -qmax).any()) or bool((q > qmax).any()):
                raise TaskError("registered code escaped the signed logical bit range")
            packet = _write_packet(public_root / f"bit{bits:02d}" / f"batch{batch:02d}" / "observation",
                                   q=q, scale=scale, bits=bits, source_hash=y_hash,
                                   source_record=record, identity=identity)
            packet_rows.append({"bits": bits, "batch_index": batch,
                                "packet_path": str((public_root / f"bit{bits:02d}" / f"batch{batch:02d}" / "observation").relative_to(public_root)),
                                "packet_sha256": file_sha256(public_root / f"bit{bits:02d}" / f"batch{batch:02d}" / "observation/packet.json"),
                                "codes_file_sha256": packet["codes_file_sha256"],
                                "source_Y_sha256": y_hash, "scale": packet["scale"],
                                "source_observation_sha256": record["observation_sha256"]})
            del q, scale
        del y
        gc.collect()
    registration["packets"] = packet_rows
    registration["status"] = "PREPARED"
    registration["prepared_at_utc"] = registration.get("prepared_at_utc", utc_now())
    registration["packet_count"] = len(packet_rows)
    registration["physical_int8_code_bytes"] = sum(int(row["bits"] and FC1_SHAPE[0] * FC1_SHAPE[1]) for row in packet_rows)
    registration["logical_code_bits"] = sum(int(row["bits"]) * FC1_SHAPE[0] * FC1_SHAPE[1] for row in packet_rows)
    write_json(registration_path, registration)
    write_json(public_root / "input_manifest.json", {
        "identity": identity, "public_packets": packet_rows,
        "source_update_sign_and_scale": "actual_observation.Y_delta_over_lr=(W_initial-W_after10)/0.001",
        "private_paths_or_tensors_embedded": False,
    })
    write_json(private_root / "task_boundary.json", {
        "task_id": TASK_ID, "attack_reads_private_inputs": False,
        "evaluator_may_read_after_public_gate": True,
        "true_update_used_only_by_prepare_encoder": True,
        "labels_used_for_attack_or_selection": False,
    })
    return registration


def _reviewed_repair_transition(saved: dict[str, Any], current_code: dict[str, str],
                                registration_path: Path) -> dict[str, Any] | None:
    saved_code = saved.get("code_sha256")
    if not isinstance(saved_code, dict) or set(saved_code) != set(current_code):
        raise TaskError("registered code path set changed")
    changed = {name for name, digest in saved_code.items() if current_code[name] != digest}
    transition_path = REPAIR_TRANSITION_PATH
    if not transition_path.is_file():
        if changed:
            raise TaskError(f"registered code hash changed without reviewed transition: {sorted(changed)}")
        verify_hash_map(saved_code)
        return None
    transition = read_json(transition_path)
    registration = read_json(registration_path)
    identity_sha = hashlib.sha256(json.dumps(saved, sort_keys=True, allow_nan=False).encode()).hexdigest()
    runner_key = Path(__file__).resolve().as_posix()
    changes = transition.get("allowed_code_changes", {}) if isinstance(transition, dict) else {}
    change = changes.get(runner_key, {}) if isinstance(changes, dict) else {}
    report = transition.get("validation_report", {}) if isinstance(transition, dict) else {}
    if (not isinstance(transition, dict)
            or transition.get("schema") != REPAIR_TRANSITION_SCHEMA
            or transition.get("status") != "VALIDATED"
            or transition.get("task_id") != TASK_ID
            or transition.get("plan_version") != PLAN_VERSION
            or transition.get("amendment_id") != "CPA_SAME80_SCORE_REPAIR_20260926"
            or transition.get("authorization_handoff_sha256") != REPAIR_HANDOFF_SHA256
            or transition.get("registration_identity_sha256") != identity_sha
            or transition.get("config_sha256") != saved.get("config_sha256")
            or set(changes) != {runner_key}
            or changed != {runner_key}
            or change.get("before_sha256") != saved_code.get(runner_key)
            or change.get("after_sha256") != current_code.get(runner_key)
            or not isinstance(report, dict)
            or report.get("path") != (Path(".ai/relay-runs/supervisors") / TASK_ID / "repair_validation.json").as_posix()
            or not (WORKSPACE / report["path"]).is_file()
            or file_sha256(WORKSPACE / report["path"]) != report.get("sha256")
            or not isinstance(transition.get("checkpoint_provenance"), dict)
            or transition["checkpoint_provenance"].get("checkpoint_sha256")
               != "b40e114e85317563996bad4b8e923c6ed1ed0548a3d0034ac9d54d154c65e6ff"
            or not registration
            or registration.get("identity") != saved):
        raise TaskError("reviewed repair code transition is absent, malformed, or mismatched")
    verify_hash_map({name: digest for name, digest in saved_code.items() if name != runner_key})
    return transition


def _base_identity_matches(saved: dict[str, Any], config_path: Path, cfg: dict[str, Any],
                           registration_path: Path) -> None:
    base = code_identity(config_path.resolve(), cfg)
    for key, value in base.items():
        if key != "code_sha256" and saved.get(key) != value:
            raise TaskError(f"public task registration changed at {key}")
    _reviewed_repair_transition(saved, base["code_sha256"], registration_path)


def _load_registration(public_root: Path, config_path: Path, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_config(cfg)
    registration = read_json(public_root / "registration.json")
    identity = registration.get("identity")
    if not isinstance(identity, dict):
        raise TaskError("public registration has no immutable identity")
    _base_identity_matches(identity, config_path, cfg, public_root / "registration.json")
    if registration.get("status") not in {"PREPARED", "RUNNING", "PUBLIC_COMPLETED"}:
        raise TaskError("public registration is not in a resumable state")
    return registration, identity


def _load_packet(public_root: Path, bits: int, batch: int, identity: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    folder = public_root / f"bit{bits:02d}/batch{batch:02d}/observation"
    manifest = read_json(folder / "packet.json")
    packet_identity = manifest.get("packet_identity", {})
    registration = read_json(public_root / "registration.json")
    registered = [row for row in registration.get("packets", [])
                  if int(row.get("bits", -1)) == bits and int(row.get("batch_index", -1)) == batch]
    if (packet_identity.get("task_id") != TASK_ID or packet_identity.get("bits") != bits
            or packet_identity.get("batch_index") != batch
            or packet_identity.get("registration_identity_sha256") != hashlib.sha256(
                json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
            or packet_identity.get("code_sha256") != identity["code_sha256"]
            or packet_identity.get("source_actual_observation_sha256")
               != identity["source_batches"][batch]["observation_sha256"]
            or packet_identity.get("input_sha256") != identity["source_batches"][batch]["input_sha256"]
            or identity.get("source_observation_sign_scale")
               != "actual_observation.Y_delta_over_lr=(W_initial-W_after10)/0.001"
            or len(registered) != 1
            or registered[0].get("packet_path") != f"bit{bits:02d}/batch{batch:02d}/observation"
            or file_sha256(folder / "packet.json") != registered[0].get("packet_sha256")):
        raise TaskError("public packet identity mismatch")
    code_path = folder / manifest["codes_file"]
    if file_sha256(code_path) != manifest["codes_file_sha256"]:
        raise TaskError("public code packet hash mismatch")
    q = torch.load(code_path, map_location="cpu", weights_only=True)
    scale = torch.tensor(np.frombuffer(bytes.fromhex(manifest["scale_fp32_hex"]), dtype=np.float32).copy()[0], dtype=torch.float32)
    qmax = qmax_for_bits(bits)
    if (q.dtype != torch.int8 or tuple(q.shape) != FC1_SHAPE or bool((q < -qmax).any())
            or bool((q > qmax).any()) or not bool(torch.isfinite(scale)) or not bool(scale > 0)):
        raise TaskError("invalid public codes or scalar scale")
    return q, scale, manifest


def _attempts(path: Path) -> list[dict[str, Any]]:
    return read_json(path).get("attempts", []) if path.is_file() else []


def _start_attempt(folder: Path, name: str, *, prior_steps: int = 0,
                   checkpoint_valid: bool = False, wasted_upper_bound: int | None = None) -> dict[str, Any]:
    path = folder / f"{name}_attempts.json"
    attempts = _attempts(path)
    if attempts and attempts[-1].get("status") in {"FAILED_NUMERICAL", "FAILED_ERROR"}:
        raise TaskError(f"automatic rerun is prohibited after recorded {name} failure")
    interrupted = bool(attempts and attempts[-1].get("status") == "RUNNING")
    upper = max(prior_steps, wasted_upper_bound if wasted_upper_bound is not None else prior_steps)
    entry = {"attempt": len(attempts) + 1, "status": "RUNNING", "started_at_utc": utc_now(),
             "pid": os.getpid(), "resumed_after_interruption": interrupted,
             "valid_checkpoint_reused": bool(interrupted and checkpoint_valid),
             "wasted_work_steps_lower_bound": int(prior_steps if interrupted and not checkpoint_valid else 0),
             "wasted_work_steps_upper_bound": int(upper if interrupted and not checkpoint_valid else 0),
             "wasted_work_exact": bool(not interrupted or checkpoint_valid or upper == prior_steps)}
    attempts.append(entry)
    write_json(path, {"attempts": attempts})
    return entry


def _finish_attempt(folder: Path, name: str, attempt: dict[str, Any], *, status: str,
                    completed_steps: int = 0, checkpoint_sha256: str | None = None,
                    error: str | None = None) -> None:
    path = folder / f"{name}_attempts.json"
    attempts = _attempts(path)
    item = attempts[-1]
    if item.get("attempt") != attempt["attempt"]:
        raise TaskError("attempt ledger was concurrently changed")
    item.update({"status": status, "ended_at_utc": utc_now(),
                 "completed_steps": int(completed_steps), "checkpoint_sha256": checkpoint_sha256,
                 "error": error})
    write_json(path, {"attempts": attempts})


def _integrity_sidecar(checkpoint: Path, *, horizon: int, identity_sha: str,
                       kind: str = "fia") -> dict[str, Any] | None:
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".integrity.json")
    if not checkpoint.is_file() or not sidecar.is_file():
        return None
    item = read_json(sidecar)
    if item.get("checkpoint_sha256") != file_sha256(checkpoint) or item.get("horizon") != horizon or item.get("identity_sha256") != identity_sha:
        raise TaskError(f"checkpoint integrity/authentication failed: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    completed = int(state.get("completed_iterations", -1))
    common_valid = (0 <= completed <= horizon and isinstance(state.get("attack_state"), dict))
    if kind == "fia":
        common_valid = (common_valid and state.get("horizon") == horizon
                        and all(key in state for key in ("rng_python", "rng_numpy", "rng_torch", "rng_cuda"))
                        and all(key in state["attack_state"] for key in ("S_hat", "opt", "sch")))
    elif kind == "cpa":
        common_valid = common_valid and all(key in state for key in ("log", "wall_time_sec"))
    else:
        raise ValueError("unknown checkpoint kind")
    if not common_valid:
        raise TaskError(f"checkpoint state is incomplete: {checkpoint}")
    item["completed_iterations"] = completed
    return item


def _legacy_fia_provenance(output: Path, *, source_sha: str,
                           identity_sha: str) -> dict[str, Any]:
    output = Path(output).resolve()
    try:
        output_rel = output.relative_to(PUBLIC_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise TaskError("reviewed legacy checkpoint is outside the registered public root") from exc
    if output_rel != "bit08/batch00/naive/fia":
        raise TaskError("legacy checkpoint reuse is allowlisted only for bit08/batch00 Naive/FIA")
    config_path = PROJECT / "configs/same80_bits87654_R100_S5k_K3_F25k.json"
    cfg = read_json(config_path)
    registration, registered_identity = _load_registration(PUBLIC_ROOT, config_path, cfg)
    if registration.get("packet_count") != len(BITS) * N_BATCHES:
        raise TaskError("legacy FIA checkpoint registration is incomplete")
    q, scale, _ = _load_packet(PUBLIC_ROOT, 8, 0, registered_identity)
    if _naive_stage_id(registered_identity, q, scale, 8, 0) != identity_sha:
        raise TaskError("legacy FIA stage identity differs from the registered Naive baseline")
    cp = output / "fia_checkpoint.pt"
    ledger_path = output / "naive_fia_attempts.json"
    manifest_path = output / "fia_manifest.json"
    cpa_path = output.parent / "cpa_result.json"
    sources_path = output.parent / "sources.pt"
    for path in (cp, ledger_path, manifest_path, cpa_path, sources_path):
        if not path.is_file() or path.is_symlink():
            raise TaskError(f"legacy FIA provenance input is missing or not regular: {path}")
    ledger = read_json(ledger_path).get("attempts", [])
    if len(ledger) != 1:
        raise TaskError("legacy Naive/FIA completion ledger is not the original single attempt")
    attempt = ledger[0]
    checkpoint_sha = file_sha256(cp)
    if (attempt.get("attempt") != 1 or attempt.get("status") != "COMPLETED"
            or attempt.get("completed_steps") != TOTAL_STEPS or attempt.get("error") is not None
            or attempt.get("checkpoint_sha256") != checkpoint_sha):
        raise TaskError("legacy Naive/FIA completion record does not authenticate this checkpoint")
    state = torch.load(cp, map_location="cpu", weights_only=False)
    attack_state = state.get("attack_state")
    required_rng = {"rng_python", "rng_numpy", "rng_torch", "rng_cuda"}
    optimizer_state = attack_state.get("opt") if isinstance(attack_state, dict) else None
    scheduler_state = attack_state.get("sch") if isinstance(attack_state, dict) else None
    if (state.get("completed_iterations") != TOTAL_STEPS or state.get("horizon") != TOTAL_STEPS
            or not isinstance(attack_state, dict)
            or not {"S_hat", "opt", "sch"}.issubset(attack_state)
            or not isinstance(optimizer_state, dict)
            or not isinstance(optimizer_state.get("state"), dict)
            or not isinstance(optimizer_state.get("param_groups"), list)
            or not optimizer_state["param_groups"]
            or not isinstance(scheduler_state, dict)
            or scheduler_state.get("T_max") != TOTAL_STEPS
            or scheduler_state.get("last_epoch") != TOTAL_STEPS
            or not required_rng.issubset(state)
            or not isinstance(state.get("log"), list) or not state["log"]
            or int(state["log"][-1].get("iter", -1)) != TOTAL_STEPS - 1
            or not np.isfinite(float(state.get("wall_time_sec", float("nan"))))):
        raise TaskError("legacy FIA checkpoint schema, endpoint, RNG, or optimizer state is incomplete")
    saved_image = attack_state["S_hat"]
    if (not isinstance(saved_image, torch.Tensor) or saved_image.dtype != torch.float32
            or tuple(saved_image.shape) != (N_IMAGES, 3, 224, 224)
            or not bool(torch.isfinite(saved_image).all())):
        raise TaskError("legacy FIA checkpoint image state is invalid")
    cpa = read_json(cpa_path)
    sources = torch.load(sources_path, map_location="cpu", weights_only=True)
    target_sha = tensor_sha256(sources.abs().contiguous())
    if (cpa.get("sources_file_sha256") != file_sha256(sources_path)
            or cpa.get("sources_tensor_sha256") != tensor_sha256(sources)
            or cpa.get("private_data_used") is not False
            or target_sha != source_sha):
        raise TaskError("legacy FIA source artifact does not match the completed CPA record")
    manifest = read_json(manifest_path)
    fingerprint = manifest.get("fingerprint", {})
    expected_settings = {
        "seed": int(FIA["seed"]), "device": "cuda", "n_iter": TOTAL_STEPS,
        "lr": float(FIA["lr"]), "tv": float(FIA["tv"]), "fi": float(FIA["fi"]),
        "optimizer": str(FIA["optimizer"]), "scheduler": str(FIA["scheduler"]),
        "log_every": int(FIA["log_every"]),
        "backend": "official_cpa/src/feature_inversion.py:Direct",
        "driver": "staged_fia_v1; fixed full cosine horizon; original Direct.step",
        "cudnn_benchmark": True, "cudnn_deterministic": True,
    }
    if (fingerprint.get("kind") != "fia" or fingerprint.get("input_sha256") != source_sha
            or fingerprint.get("settings") != expected_settings
            or fingerprint.get("source_hashes") != upstream_naive.source_hashes()):
        raise TaskError("legacy FIA manifest no longer matches its saved source/settings identity")
    current_identity = code_identity(
        (PROJECT / "configs/same80_bits87654_R100_S5k_K3_F25k.json").resolve(),
        read_json(PROJECT / "configs/same80_bits87654_R100_S5k_K3_F25k.json"))
    transition = _reviewed_repair_transition(
        read_json(PUBLIC_ROOT / "registration.json")["identity"],
        current_identity["code_sha256"], PUBLIC_ROOT / "registration.json")
    if transition is None:
        raise TaskError("reviewed code transition is required for legacy checkpoint reuse")
    transition_sha = file_sha256(REPAIR_TRANSITION_PATH)
    return {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION,
        "authorization_handoff_sha256": REPAIR_HANDOFF_SHA256,
        "repair_transition_sha256": transition_sha,
        "registration_identity_sha256": hashlib.sha256(json.dumps(
            read_json(PUBLIC_ROOT / "registration.json")["identity"],
            sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "stage_identity_sha256": identity_sha,
        "source_sha256": source_sha,
        "sources_file_sha256": file_sha256(sources_path),
        "sources_tensor_sha256": tensor_sha256(sources),
        "cpa_record_sha256": file_sha256(cpa_path),
        "fia_manifest_sha256": file_sha256(manifest_path),
        "attempt_ledger_sha256": file_sha256(ledger_path),
        "attempt_number": 1,
        "completed_iterations": TOTAL_STEPS,
        "horizon": TOTAL_STEPS,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_image_sha256": tensor_sha256(saved_image),
        "checkpoint_wall_time_sec": float(state["wall_time_sec"]),
    }


def _legacy_fia_attestation_path(checkpoint: Path) -> Path:
    checkpoint = Path(checkpoint)
    return checkpoint.with_name(checkpoint.name + ".reviewed_recovery.json")


def _write_legacy_fia_attestation(output: Path, *, source_sha: str,
                                  identity_sha: str, artifact_name: str) -> dict[str, Any]:
    base = _legacy_fia_provenance(output, source_sha=source_sha, identity_sha=identity_sha)
    artifact = Path(output) / artifact_name
    if not artifact.is_file() or file_sha256(artifact) != base["checkpoint_sha256"]:
        raise TaskError("recovered checkpoint artifact differs from its completed original")
    source_attestation = _legacy_fia_attestation_path(Path(output) / "fia_checkpoint.pt")
    value = {
        "schema": LEGACY_FIA_ATTESTATION_SCHEMA,
        **base,
        "artifact_path": artifact.relative_to(PUBLIC_ROOT.resolve()).as_posix(),
        "artifact_sha256": file_sha256(artifact),
        "origin_attestation_sha256": (file_sha256(source_attestation)
                                      if artifact_name != "fia_checkpoint.pt" and source_attestation.is_file()
                                      else None),
        "attested_at_utc": utc_now(),
    }
    path = _legacy_fia_attestation_path(artifact)
    if path.exists():
        saved = read_json(path)
        stable_keys = set(value) - {"attested_at_utc"}
        if any(saved.get(key) != value[key] for key in stable_keys):
            raise TaskError(f"existing reviewed FIA attestation differs: {path}")
        return {**saved, "attestation_sha256": file_sha256(path)}
    write_json(path, value)
    return {**value, "attestation_sha256": file_sha256(path)}


def _verify_legacy_fia_attestation(output: Path, *, source_sha: str,
                                   identity_sha: str, artifact_name: str) -> dict[str, Any]:
    artifact = Path(output) / artifact_name
    path = _legacy_fia_attestation_path(artifact)
    if not path.is_file():
        raise TaskError("completed legacy FIA checkpoint lacks its separate reviewed recovery attestation")
    saved = read_json(path)
    base = _legacy_fia_provenance(output, source_sha=source_sha, identity_sha=identity_sha)
    expected_path = artifact.relative_to(PUBLIC_ROOT.resolve()).as_posix()
    source_attestation = _legacy_fia_attestation_path(Path(output) / "fia_checkpoint.pt")
    expected_origin = (file_sha256(source_attestation)
                       if artifact_name != "fia_checkpoint.pt" and source_attestation.is_file()
                       else None)
    for key, value in base.items():
        if saved.get(key) != value:
            raise TaskError(f"reviewed FIA checkpoint provenance mismatch at {key}")
    if (saved.get("schema") != LEGACY_FIA_ATTESTATION_SCHEMA
            or saved.get("artifact_path") != expected_path
            or saved.get("artifact_sha256") != base["checkpoint_sha256"]
            or saved.get("origin_attestation_sha256") != expected_origin
            or not saved.get("attested_at_utc")
            or not artifact.is_file()
            or file_sha256(artifact) != base["checkpoint_sha256"]):
        raise TaskError("reviewed FIA checkpoint artifact attestation is invalid")
    return {**saved, "attestation_sha256": file_sha256(path)}


def _recording_saver(module: Any, checkpoint_dir: Path, *, horizon: int,
                     identity_sha: str, kind: str):
    original = module._save_checkpoint

    def save(path: Path, payload: dict[str, Any]) -> None:
        if Path(path).name in {"fia_checkpoint.pt", "cpa_checkpoint.pt"}:
            sidecar_path = Path(path).with_suffix(Path(path).suffix + ".integrity.json")
            if sidecar_path.exists():
                sidecar_path.unlink()
        original(path, payload)
        if Path(path).name in {"fia_checkpoint.pt", "cpa_checkpoint.pt"}:
            sidecar = {"checkpoint_sha256": file_sha256(path),
                       "completed_iterations": int(payload.get("completed_iterations", -1)),
                       "horizon": horizon, "identity_sha256": identity_sha,
                       "kind": kind,
                       "recorded_at_utc": utc_now()}
            write_json(Path(path).with_suffix(Path(path).suffix + ".integrity.json"), sidecar)

    module._save_checkpoint = save
    return original


def _cpa_config(batch: int) -> dict[str, Any]:
    return {**STAGE1_CONFIG, "n_components": 16,
            "cpa_ica_log_every": 5000}


def _cleanup_cpa_work(folder: Path, task_root: Path) -> None:
    manifest = folder / "cpa_manifest.json"
    if manifest.is_file():
        value = read_json(manifest)
        cache = Path(value["cache_root"]).resolve()
        if cache.exists():
            remove_task_disposable(cache, task_root, allowed_names={cache.name})
    for name in ("cpa_checkpoint.pt", "cpa_checkpoint.pt.integrity.json", "cpa_manifest.json"):
        path = folder / name
        if path.exists():
            remove_task_disposable(path, task_root, allowed_names={name})
    for path in folder.glob("untrusted_cpa_checkpoint_*.pt"):
        remove_task_disposable(path, task_root, allowed_names={path.name})
    try:
        folder.rmdir()
    except OSError:
        pass


def _run_cpa(matrix: torch.Tensor, *, stage_dir: Path, identity: dict[str, Any],
             batch: int, device: torch.device, label: str, measure: bool,
             task_root: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    result_path, source_path = stage_dir / "upstream_cpa_result.json", stage_dir / "sources.pt"
    stage_identity = {"identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                      "label": label, "batch_index": batch,
                      "matrix_sha256": tensor_sha256(matrix), "cpa_seed": batch,
                      "config": _cpa_config(batch), "trajectory_steps": 25000}
    if result_path.exists():
        record = read_json(result_path)
        if record.get("stage_identity") != stage_identity or file_sha256(source_path) != record.get("sources_sha256"):
            raise TaskError(f"completed CPA artifact changed: {stage_dir}")
        _cleanup_cpa_work(stage_dir / "cpa_work", task_root)
        return torch.load(source_path, map_location="cpu", weights_only=True), record
    stage_dir.mkdir(parents=True, exist_ok=True)
    ensure_free_space(stage_dir, reserve_bytes=MIN_FREE_BYTES)
    work = stage_dir / "cpa_work"
    checkpoint = work / "cpa_checkpoint.pt"
    previous_steps = 0
    checkpoint_valid = False
    if checkpoint.is_file():
        sidecar = checkpoint.with_suffix(checkpoint.suffix + ".integrity.json")
        if sidecar.is_file():
            previous = _integrity_sidecar(checkpoint, horizon=25000,
                identity_sha=hashlib.sha256(json.dumps(stage_identity, sort_keys=True, allow_nan=False).encode()).hexdigest(), kind="cpa")
            previous_steps = int(previous["completed_iterations"])
            checkpoint_valid = True
        else:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            previous_steps = int(saved.get("completed_iterations", 0))
            if _attempts(stage_dir / "cpa_attempts.json") and _attempts(stage_dir / "cpa_attempts.json")[-1].get("status") == "RUNNING":
                quarantine = work / f"untrusted_cpa_checkpoint_{int(time.time())}.pt"
                checkpoint.replace(quarantine)
                checkpoint_valid = False
            else:
                raise TaskError("CPA checkpoint lacks a trusted interruption record")
    attempt = _start_attempt(stage_dir, "cpa", prior_steps=previous_steps,
                             checkpoint_valid=checkpoint_valid, wasted_upper_bound=25000)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    saver_identity = hashlib.sha256(json.dumps(stage_identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    original_saver = _recording_saver(upstream_naive, work, horizon=25000,
                                      identity_sha=saver_identity, kind="cpa")
    try:
        value = upstream_naive.run_upstream_cpa(Y_hat=matrix, n_components=16, device=device,
                  config=_cpa_config(batch), output_dir=work, seed=batch)
        if tuple(value["S_hat"].shape) != (16, 25088) or not bool(torch.isfinite(value["S_hat"]).all()):
            raise TaskError("original CPA returned invalid source matrix")
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        sources = value["S_hat"].detach().cpu().contiguous()
        from .same80_bits_r100_core import write_tensor
        write_tensor(source_path, sources)
        record = {"status": "COMPLETED", "stage_identity": stage_identity,
                  "sources_sha256": file_sha256(source_path), "source_tensor_sha256": tensor_sha256(sources),
                  "settings": value["settings"], "source_code_hashes": value["source_hashes"],
                  "optimizer_log": value["log"], "wall_seconds": elapsed,
                  "peak_cuda_allocated_bytes": int(peak), "cache_path_before_cleanup": value.get("cache_path"),
                  "attempt": attempt["attempt"], "completed_iterations": 25000,
                  "resource_measurement": bool(measure), "completed_at_utc": utc_now()}
        write_json(result_path, record)
        checkpoint_sha = file_sha256(checkpoint) if checkpoint.is_file() else None
        _finish_attempt(stage_dir, "cpa", attempt, status="COMPLETED", completed_steps=25000,
                        checkpoint_sha256=checkpoint_sha)
        _cleanup_cpa_work(work, task_root)
        return sources, record
    except BaseException as exc:
        message = f"{type(exc).__name__}: {exc}"
        numerical = any(word in message.lower() for word in ("nonfinite", "nan", "inf", "numerical"))
        _finish_attempt(stage_dir, "cpa", attempt,
                        status="FAILED_NUMERICAL" if numerical else "FAILED_ERROR",
                        completed_steps=previous_steps, error=message)
        raise
    finally:
        upstream_naive._save_checkpoint = original_saver


def _score_checkpoint(target_z: torch.Tensor, checkpoint: Path, *, output: Path,
                      source_sha: str, identity_sha: str, device: torch.device,
                      task_root: Path) -> dict[str, Any]:
    score_started = time.perf_counter()
    if not checkpoint.is_file():
        raise TaskError(f"FIA checkpoint missing before public scoring: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    completed = int(state.get("completed_iterations", -1))
    if completed not in {PREFIX_STEPS, TOTAL_STEPS} or state.get("horizon") != TOTAL_STEPS:
        raise TaskError("FIA score checkpoint has the wrong step/horizon")
    from .signed_int8_screen3_fia import construct_attack, score_attack
    score_dir = output / f"score_work_{completed}"
    if score_dir.exists():
        _cleanup_score_work(score_dir, task_root)
    attack, _ = construct_attack(target_z, output=score_dir, seed=42, device=device)
    attack.set_attack_state(screen_fia._device_state(state["attack_state"], torch.device(device)))
    if screen_fia.tensor_sha256(attack.S_hat) != screen_fia.tensor_sha256(state["attack_state"]["S_hat"]):
        raise TaskError("public score state restoration changed pixels")
    score = screen_fia.score_attack(attack, target_z)
    if not score.get("raw_state_unchanged"):
        raise TaskError("public score mutated FIA state")
    cache_manifest = score_dir / "upstream_cache/fia_manifest.json"
    if cache_manifest.is_file():
        cache_root = Path(read_json(cache_manifest)["cache_root"]).resolve()
        if cache_root.exists():
            remove_task_disposable(cache_root, task_root, allowed_names={cache_root.name})
    if cache_manifest.exists():
        remove_task_disposable(cache_manifest, task_root, allowed_names={cache_manifest.name})
    if cache_manifest.parent.exists():
        try:
            cache_manifest.parent.rmdir()
        except OSError as exc:
            raise TaskError(f"temporary score cache was not fully released: {cache_manifest.parent}") from exc
    try:
        score_dir.rmdir()
    except OSError:
        raise TaskError(f"temporary scoring adapter was not fully released: {score_dir}")
    score["checkpoint_sha256"] = file_sha256(checkpoint)
    score["completed_iterations"] = completed
    score["source_sha256"] = source_sha
    score["selection_identity_sha256"] = identity_sha
    score["score_wall_seconds"] = time.perf_counter() - score_started
    return score


def _cleanup_score_work(score_dir: Path, task_root: Path) -> None:
    score_dir = Path(score_dir).resolve()
    if not score_dir.exists():
        return
    cache_manifest = score_dir / "upstream_cache/fia_manifest.json"
    if cache_manifest.is_file():
        cache = Path(read_json(cache_manifest)["cache_root"]).resolve()
        if cache.exists():
            remove_task_disposable(cache, task_root, allowed_names={cache.name})
        remove_task_disposable(cache_manifest, task_root, allowed_names={cache_manifest.name})
    parent = score_dir / "upstream_cache"
    if parent.exists():
        try:
            parent.rmdir()
        except OSError as exc:
            raise TaskError(f"stale score work contains unrecognized files: {parent}") from exc
    try:
        score_dir.rmdir()
    except OSError as exc:
        raise TaskError(f"stale score work contains unrecognized files: {score_dir}") from exc


def _fia_attempt_folder(output: Path, *, identity_sha: str, source_sha: str) -> dict[str, Any] | None:
    checkpoint = output / "fia_checkpoint.pt"
    manifest = output / "fia_manifest.json"
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".integrity.json")
    attempt_names = ("screen_fia_attempts.json", "continuation_fia_attempts.json", "naive_fia_attempts.json")
    last_attempt = None
    for name in attempt_names:
        records = _attempts(output / name)
        if records:
            last_attempt = records[-1]
    interrupted = bool(last_attempt and last_attempt.get("status") == "RUNNING")
    if not checkpoint.is_file():
        if last_attempt and last_attempt.get("status") == "COMPLETED":
            raise TaskError("completed FIA checkpoint is missing; refusing to repeat its optimization")
        if sidecar.exists() and interrupted:
            sidecar.replace(output / f"untrusted_fia_sidecar_{time.time_ns()}.json")
        elif sidecar.exists():
            raise TaskError("FIA checkpoint sidecar exists without its checkpoint")
        return None
    reviewed_legacy = None
    if not sidecar.is_file():
        if interrupted:
            try:
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                wasted = max(0, int(state.get("completed_iterations", 0)))
            except BaseException:
                wasted = 0
            checkpoint.replace(output / f"untrusted_fia_checkpoint_{time.time_ns()}.pt")
            return {"completed_iterations": wasted, "checkpoint_valid": False}
        reviewed_legacy = _verify_legacy_fia_attestation(
            output, source_sha=source_sha, identity_sha=identity_sha,
            artifact_name="fia_checkpoint.pt")
        side = None
    else:
        side = read_json(sidecar)
        if (side.get("checkpoint_sha256") != file_sha256(checkpoint) or side.get("horizon") != TOTAL_STEPS
                or side.get("identity_sha256") != identity_sha or side.get("kind") != "fia"):
            raise TaskError("FIA checkpoint hash/horizon mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    attack_state = state.get("attack_state")
    if (state.get("horizon") != TOTAL_STEPS or not 0 <= int(state.get("completed_iterations", -1)) <= TOTAL_STEPS
            or not isinstance(attack_state, dict) or not {"S_hat", "opt", "sch"}.issubset(attack_state)):
        raise TaskError("FIA checkpoint is incomplete")
    if manifest.is_file():
        fp = read_json(manifest).get("fingerprint", {})
        if fp.get("input_sha256") != source_sha or fp.get("kind") != "fia":
            raise TaskError("FIA resume manifest target hash changed")
    result = side or reviewed_legacy or {}
    result["completed_iterations"] = int(state["completed_iterations"])
    result["checkpoint_valid"] = True
    if reviewed_legacy:
        result["reviewed_legacy_completed_checkpoint"] = True
        result["checkpoint_wall_time_sec"] = float(state["wall_time_sec"])
    return result


def _run_fia(target_z: torch.Tensor, *, output: Path, stop: int, source_sha: str,
             identity_sha: str, device: torch.device,
             attempt_name: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ensure_free_space(output.parent, reserve_bytes=MIN_FREE_BYTES)
    output.mkdir(parents=True, exist_ok=True)
    cp = output / "fia_checkpoint.pt"
    previous = _fia_attempt_folder(output, identity_sha=identity_sha, source_sha=source_sha)
    valid_cp = bool(previous and previous.get("checkpoint_valid", True))
    previous_steps = int(previous["completed_iterations"]) if previous else 0
    reviewed_reuse = bool(previous and previous.get("reviewed_legacy_completed_checkpoint"))
    if reviewed_reuse and (previous_steps != TOTAL_STEPS or stop != TOTAL_STEPS):
        raise TaskError("reviewed completed FIA checkpoint may only be reused at its original full horizon")
    attempt = None if reviewed_reuse else _start_attempt(
        output, attempt_name, prior_steps=previous_steps,
        checkpoint_valid=valid_cp, wasted_upper_bound=stop)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    original_saver = _recording_saver(staged_fia, output, horizon=TOTAL_STEPS,
                                      identity_sha=identity_sha, kind="fia")
    try:
        result = staged_fia.run_staged_fia(target_z=target_z, device=device,
                 config=screen_fia.FIA_CONFIG, output_dir=output, seed=int(FIA["seed"]), stop=stop)
        if reviewed_reuse and int(result.get("completed_iterations", -1)) != TOTAL_STEPS:
            raise TaskError("staged FIA did not restore the authenticated completed horizon")
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        rec_path = output / f"rec_{stop}.pt"
        from .same80_bits_r100_core import write_tensor
        recovered_rec = result["rec"].detach().cpu().contiguous()
        if rec_path.is_file() and reviewed_reuse:
            saved_rec = torch.load(rec_path, map_location="cpu", weights_only=True)
            if tensor_sha256(saved_rec) != tensor_sha256(recovered_rec):
                raise TaskError("restored completed FIA reconstruction differs from preserved output")
        else:
            write_tensor(rec_path, recovered_rec)
        cp_sha = file_sha256(cp)
        if attempt is not None:
            _finish_attempt(output, attempt_name, attempt, status="COMPLETED", completed_steps=stop,
                            checkpoint_sha256=cp_sha)
        result["rec"] = recovered_rec
        result["runner_wall_seconds"] = (float(previous["checkpoint_wall_time_sec"])
                                         if reviewed_reuse else elapsed)
        result["peak_cuda_allocated_bytes"] = None if reviewed_reuse else int(peak)
        result["resource_measurement_available"] = not reviewed_reuse
        result["reviewed_completed_checkpoint_reused"] = reviewed_reuse
        result["optimizer_steps_executed"] = 0 if reviewed_reuse else max(0, stop - previous_steps)
        if reviewed_reuse:
            result["recovery_rehydrate_wall_seconds"] = elapsed
            result["reviewed_recovery_attestation_sha256"] = previous["attestation_sha256"]
        result["rec_path"] = rec_path
        return result, attempt
    except BaseException as exc:
        if attempt is not None:
            msg = f"{type(exc).__name__}: {exc}"
            numerical = any(word in msg.lower() for word in ("nonfinite", "nan", "inf", "numerical"))
            _finish_attempt(output, attempt_name, attempt,
                            status="FAILED_NUMERICAL" if numerical else "FAILED_ERROR",
                            completed_steps=previous_steps, error=msg)
        raise
    finally:
        staged_fia._save_checkpoint = original_saver


def _progress(public_root: Path, **state: Any) -> None:
    write_json(public_root / "stage_execution.json", {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "pid": os.getpid(),
        "updated_at_utc": utc_now(), **state})


def _copy_atomic(source: Path, target: Path) -> None:
    source, target = Path(source), Path(target)
    if not source.is_file() or source.is_symlink():
        raise TaskError(f"atomic copy source is not a regular file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with source.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _disk_and_progress(public_root: Path, **state: Any) -> None:
    free = ensure_free_space(public_root, reserve_bytes=MIN_FREE_BYTES)
    _progress(public_root, free_disk_bytes=free, **state)


def _naive_stage_id(identity: dict[str, Any], q: torch.Tensor, scale: torch.Tensor,
                    bits: int, batch: int) -> str:
    value = {"identity": identity, "bits": bits, "batch": batch,
             "arm": "naive", "q_sha": tensor_sha256(q),
             "scale": scale.numpy().tobytes().hex()}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _save_sample(candidate_dir: Path, q: torch.Tensor, scale: torch.Tensor,
                 *, bits: int, batch: int, candidate: int, identity: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    seed = candidate_seed(batch, candidate)
    sample, audit = sample_matrix(q, scale, bits=bits, seed=seed, chunk_rows=SAMPLER_CHUNK_ROWS)
    sample_record = {"bits": bits, "batch_index": batch, "candidate_index": candidate,
                     "seed": seed, "sample_sha256": tensor_sha256(sample),
                     "packet_scale": float(scale), "sample_scale_reproduced": audit["scale_reproduced"],
                     "code_consistency": audit["code_consistency"], "sampler": audit["sampler"],
                     "chunk_rows": audit["chunk_rows"], "anchor": audit["anchor"],
                     "zero_tensor_convention": audit["zero_tensor_convention"],
                     "registration_identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()}
    audit_path = candidate_dir / "sample.json"
    if audit_path.exists():
        old = read_json(audit_path)
        if old != sample_record:
            raise TaskError(f"deterministic sample changed on resume: {candidate_dir}")
    else:
        write_json(audit_path, sample_record)
    return sample, sample_record


def _run_naive(public_root: Path, q: torch.Tensor, scale: torch.Tensor, *, bits: int,
               batch: int, identity: dict[str, Any], device: torch.device) -> dict[str, Any]:
    folder = public_root / f"bit{bits:02d}/batch{batch:02d}/naive"
    stage_id = _naive_stage_id(identity, q, scale, bits, batch)
    done = verify_marker(folder / "naive_complete.json", {"stage_sha256": stage_id})
    if done:
        return done
    dequantized = (q.to(torch.float32) * scale).contiguous()
    audit = {"bits": bits, "batch_index": batch, "input": "observed_codes_times_observed_scale",
             "sample_sha256": tensor_sha256(dequantized), "sampler": "none_center_baseline",
             "code_consistency": True, "scale_exact": True, "seed": batch}
    write_json(folder / "input.json", audit)
    source_path = folder / "sources.pt"
    cpa_result = folder / "cpa_result.json"
    if cpa_result.is_file():
        rec = read_json(cpa_result)
        if rec.get("input_sha256") != audit["sample_sha256"] or file_sha256(source_path) != rec["sources_file_sha256"]:
            raise TaskError("completed naive CPA artifact changed")
        sources = torch.load(source_path, map_location="cpu", weights_only=True)
        if tensor_sha256(sources) != rec.get("sources_tensor_sha256"):
            raise TaskError("completed naive CPA tensor hash changed")
    else:
        measure = not (public_root / "resource_measurement.json").is_file()
        sources, result = _run_cpa(dequantized, stage_dir=folder, identity=identity,
                                   batch=batch, device=device, label="naive_center", measure=measure,
                                   task_root=public_root)
        record = {"input_sha256": audit["sample_sha256"], "sources_file_sha256": file_sha256(source_path),
                  "sources_tensor_sha256": tensor_sha256(sources), "settings": result["settings"],
                  "optimizer_log": result["optimizer_log"], "wall_seconds": result["wall_seconds"],
                  "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
                  "resource_measurement": result["resource_measurement"],
                  "actual_upstream_behavior_preserved": True, "private_data_used": False}
        write_json(cpa_result, record)
    if not bool(torch.isfinite(sources).all()):
        raise TaskError("naive CPA sources contain nonfinite values")
    target_z = sources.abs().contiguous()
    target_sha = tensor_sha256(target_z)
    fia_out = folder / "fia"
    if (fia_out / "final_result.json").is_file():
        final = read_json(fia_out / "final_result.json")
        common_valid = (final.get("target_z_sha256") == target_sha
                        and file_sha256(fia_out / "rec_25000.pt") == final.get("rec_file_sha256")
                        and file_sha256(fia_out / "checkpoint_25000.pt") == final.get("checkpoint_25000_sha256"))
        if final.get("checkpoint_recovery_attestation_sha256"):
            source_att = _verify_legacy_fia_attestation(
                fia_out, source_sha=target_sha, identity_sha=stage_id,
                artifact_name="fia_checkpoint.pt")
            final_att = _verify_legacy_fia_attestation(
                fia_out, source_sha=target_sha, identity_sha=stage_id,
                artifact_name="checkpoint_25000.pt")
            common_valid = (common_valid
                            and not (fia_out / "fia_checkpoint.pt.integrity.json").exists()
                            and not (fia_out / "checkpoint_25000.pt.integrity.json").exists()
                            and source_att["attestation_sha256"] == final.get("checkpoint_recovery_attestation_sha256")
                            and final_att["attestation_sha256"] == final.get("checkpoint_25000_recovery_attestation_sha256"))
        else:
            common_valid = (common_valid
                            and file_sha256(fia_out / "checkpoint_25000.pt.integrity.json")
                                == final.get("checkpoint_integrity_sha256")
                            and file_sha256(fia_out / "fia_checkpoint.pt.integrity.json")
                                == final.get("checkpoint_integrity_sha256"))
        if not common_valid:
            raise TaskError("completed naive FIA artifacts changed")
    else:
        measure = not (public_root / "resource_measurement.json").is_file()
        result, _ = _run_fia(target_z, output=fia_out, stop=TOTAL_STEPS,
                             source_sha=target_sha, identity_sha=stage_id,
                             device=device, attempt_name="naive_fia")
        score = _score_checkpoint(target_z, fia_out / "fia_checkpoint.pt", output=fia_out,
                                  source_sha=tensor_sha256(sources), identity_sha=stage_id,
                                  device=device, task_root=public_root)
        final_path = fia_out / "rec_25000.pt"
        checkpoint_dest = fia_out / "checkpoint_25000.pt"
        if not checkpoint_dest.exists():
            _copy_atomic(fia_out / "fia_checkpoint.pt", checkpoint_dest)
        if file_sha256(checkpoint_dest) != file_sha256(fia_out / "fia_checkpoint.pt"):
            raise TaskError("Naive final checkpoint copy differs from its source")
        recovered_completed = bool(result.get("reviewed_completed_checkpoint_reused"))
        source_attestation = final_attestation = None
        if recovered_completed:
            source_attestation = _verify_legacy_fia_attestation(
                fia_out, source_sha=target_sha, identity_sha=stage_id,
                artifact_name="fia_checkpoint.pt")
            final_attestation = _write_legacy_fia_attestation(
                fia_out, source_sha=target_sha, identity_sha=stage_id,
                artifact_name="checkpoint_25000.pt")
            if ((fia_out / "fia_checkpoint.pt.integrity.json").exists()
                    or (fia_out / "checkpoint_25000.pt.integrity.json").exists()):
                raise TaskError("legacy checkpoint recovery must not fabricate an original integrity sidecar")
        else:
            sidecar_dest = fia_out / "checkpoint_25000.pt.integrity.json"
            if not sidecar_dest.exists():
                _copy_atomic(fia_out / "fia_checkpoint.pt.integrity.json", sidecar_dest)
            if file_sha256(sidecar_dest) != file_sha256(fia_out / "fia_checkpoint.pt.integrity.json"):
                raise TaskError("Naive final checkpoint integrity sidecar changed")
        resource_measurement = bool(measure and result.get("resource_measurement_available", True))
        final = {"status": "COMPLETED", "completed_iterations": TOTAL_STEPS,
                 "target_z_sha256": target_sha, "sources_sha256": tensor_sha256(sources),
                 "fia_identity_sha256": stage_id,
                 "rec_file_sha256": file_sha256(final_path), "score": score,
                 "fia_settings": result["settings"], "optimizer_log": result["log"],
                 "wall_seconds": result["runner_wall_seconds"],
                 "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
                 "checkpoint_25000_sha256": file_sha256(checkpoint_dest),
                 "resource_measurement": resource_measurement, "private_data_used": False}
        if recovered_completed:
            final.update({
                "checkpoint_recovery_attestation_sha256": source_attestation["attestation_sha256"],
                "checkpoint_25000_recovery_attestation_sha256": final_attestation["attestation_sha256"],
                "reviewed_completed_checkpoint_reused": True,
                "optimizer_steps_executed_during_recovery": 0,
                "peak_cuda_allocated_bytes_available": False,
                "resource_measurement_status": "UNAVAILABLE_FOR_ORIGINAL_COMPLETED_FIA; next scheduled Naive candidate will be measured",
            })
        else:
            final["checkpoint_integrity_sha256"] = file_sha256(sidecar_dest)
            final["peak_cuda_allocated_bytes_available"] = True
        write_json(fia_out / "final_result.json", final)
    if not (public_root / "resource_measurement.json").is_file():
        cpa_record = read_json(cpa_result)
        if cpa_record.get("resource_measurement") and final.get("resource_measurement"):
            _record_first_resource_measurement(public_root, cpa_result, final)
    _cleanup_upstream_cache(fia_out, public_root)
    complete = {"stage_sha256": stage_id, "status": "COMPLETED", "bits": bits,
                "batch_index": batch, "naive_trajectories": 1,
                "source_sha256": tensor_sha256(sources),
                "final_rec_sha256": file_sha256(fia_out / "rec_25000.pt"),
                "cpa_result_sha256": file_sha256(cpa_result),
                "fia_result_sha256": file_sha256(fia_out / "final_result.json"),
                "settings": {"cpa": STAGE1_CONFIG, "fia": FIA},
                "private_data_used": False}
    output_files = ["input.json", "sources.pt", "cpa_result.json", "fia/rec_25000.pt",
                    "fia/final_result.json", "fia/checkpoint_25000.pt"]
    if final.get("checkpoint_recovery_attestation_sha256"):
        output_files.extend(["fia/fia_checkpoint.pt.reviewed_recovery.json",
                             "fia/checkpoint_25000.pt.reviewed_recovery.json"])
    else:
        output_files.append("fia/checkpoint_25000.pt.integrity.json")
    save_marker(folder / "naive_complete.json", {"stage_sha256": stage_id},
                output_files,
                **{key: value for key, value in complete.items() if key != "stage_sha256"})
    del dequantized, sources, target_z
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return complete


def _record_first_resource_measurement(public_root: Path, cpa_path: Path,
                                       fia_result: dict[str, Any]) -> None:
    cpa = read_json(cpa_path)
    cpa_step_sec = float(cpa["wall_seconds"]) / 25000
    fia_step_sec = float(fia_result["wall_seconds"]) / 25000
    budget = expected_budget()
    completed_cpa = 0
    for path in public_root.rglob("cpa_result.json"):
        record = read_json(path)
        log = record.get("optimizer_log", [])
        if (log and int(log[-1].get("iter", -1)) == 24999
                and int(record.get("settings", {}).get("n_iter", 25000)) == 25000):
            completed_cpa += 1
    completed_naive_fia = 0
    for path in public_root.glob("bit*/batch*/naive/fia/final_result.json"):
        if read_json(path).get("completed_iterations") == TOTAL_STEPS:
            completed_naive_fia += 1
    completed_screens = sum(
        1 for path in public_root.glob("bit*/batch*/member*/screen_result.json")
        if read_json(path).get("status") == "COMPLETED"
        and read_json(path).get("screen_steps") == PREFIX_STEPS)
    completed_finals = sum(
        1 for path in public_root.glob("bit*/batch*/member*/final_result.json")
        if read_json(path).get("status") == "COMPLETED"
        and read_json(path).get("additional_steps") == TOTAL_STEPS - PREFIX_STEPS)
    completed_cpa_steps = completed_cpa * 25000
    completed_fia_steps = (completed_naive_fia * TOTAL_STEPS
                           + completed_screens * PREFIX_STEPS
                           + completed_finals * (TOTAL_STEPS - PREFIX_STEPS))
    remaining_cpa = max(0, budget["cpa_steps"] - completed_cpa_steps)
    remaining_fia = max(0, budget["fia_steps"] - completed_fia_steps)
    estimate = remaining_cpa * cpa_step_sec + remaining_fia * fia_step_sec
    try:
        rel = Path(cpa_path).parent.relative_to(Path(public_root))
        bit_label, batch_label = rel.parts[0], rel.parts[1]
        measured_candidate = f"Naive center {bit_label} {batch_label}"
    except (ValueError, IndexError):
        measured_candidate = "scheduled Naive center trajectory"
    skipped_first_measurement = read_json(
        public_root / "bit08/batch00/naive/fia/final_result.json")
    recovery_note = None
    if skipped_first_measurement.get("reviewed_completed_checkpoint_reused"):
        recovery_note = ("bit08 batch00 optimization was already COMPLETED, but its peak allocation was not persisted; "
                         "the saved checkpoint was reused with zero optimizer steps and the next scheduled Naive "
                         "candidate supplied this forward measurement")
    write_json(public_root / "resource_measurement.json", {
        "measured_candidate": measured_candidate,
        "cpa_wall_seconds": cpa["wall_seconds"], "cpa_steps": 25000,
        "cpa_peak_cuda_allocated_bytes": cpa["peak_cuda_allocated_bytes"],
        "fia_wall_seconds": fia_result["wall_seconds"], "fia_steps": 25000,
        "fia_peak_cuda_allocated_bytes": fia_result["peak_cuda_allocated_bytes"],
        "remaining_cpa_steps_after_measurement": remaining_cpa,
        "remaining_fia_steps_after_measurement": remaining_fia,
        "completed_work_counts_at_measurement": {
            "cpa_trajectories": completed_cpa,
            "naive_fia_trajectories": completed_naive_fia,
            "screen_fia_prefixes": completed_screens,
            "final_fia_continuations": completed_finals,
        },
        "projected_remaining_wall_seconds": estimate,
        "estimate_uncertainty_fraction": 0.5,
        "projection_basis": "one measured scheduled candidate; per-step linear extrapolation; not completed work",
        "recovery_provenance_note": recovery_note,
        "available_disk_bytes_at_measurement": shutil.disk_usage(public_root).free,
        "reserved_disk_bytes": MIN_FREE_BYTES,
        "measurement_time_utc": utc_now(),
    })


def _validate_cpa_record(record: dict[str, Any], batch: int) -> None:
    settings = record.get("settings", {})
    expected = {"n_components": 16, "seed": batch, "n_iter": 25000, "lr": 0.001,
                "opt": "adam", "sch": "none", "decor": 5.3, "T": 7.7,
                "ne": 1.0, "nv": 0.13, "l1": 5.0, "tv": 0.1,
                "log_every": 5000, "debias": False, "mp_truncation": False,
                "mixing_matrix_computed": False,
                "backend": "official_cpa/src/gradient_inversion.py:CocktailPartyAttack"}
    if any(settings.get(key) != value for key, value in expected.items()):
        raise TaskError("CPA trajectory does not match the original locked settings")
    log = record.get("optimizer_log", [])
    if not log or int(log[-1].get("iter", -1)) != 24999:
        raise TaskError("CPA trajectory did not complete exactly 25000 updates")


def _validate_fia_record(record: dict[str, Any], *, completed: int) -> None:
    settings = record.get("fia_settings", {})
    expected = {"n_iter": TOTAL_STEPS, "lr": 0.1, "tv": 0.1, "fi": 1.0,
                "optimizer": "adam", "scheduler": "cosine", "log_every": 5000,
                "seed": 42, "backend": "official_cpa/src/feature_inversion.py:Direct"}
    if any(settings.get(key) != value for key, value in expected.items()):
        raise TaskError("Direct FIA trajectory does not match the locked optimizer/horizon")
    log = record.get("optimizer_log", [])
    if not log or int(log[-1].get("iter", -1)) != completed - 1:
        raise TaskError("Direct FIA trajectory did not complete the registered endpoint")
    score = record.get("score", {})
    try:
        feature = float(score["feature_cosine_distance_mean"])
        tv = float(score["tv_normalized"])
        actual = float(score["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskError("public FIA score is incomplete") from exc
    if (not np.isfinite([feature, tv, actual]).all()
            or not np.isclose(actual, feature + 0.1 * tv, rtol=1e-10, atol=1e-12)
            or score.get("raw_state_unchanged") is not True):
        raise TaskError("public FIA score changed or is nonfinite")


def _screen_identity(identity: dict[str, Any], bits: int, batch: int,
                     candidate: int, sample: dict[str, Any], source_sha: str) -> dict[str, Any]:
    return {"task_identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "bits": bits, "batch_index": batch, "candidate_index": candidate,
            "seed": candidate_seed(batch, candidate), "sample_sha256": sample["sample_sha256"],
            "source_sha256": source_sha, "fia": FIA}


def _fia_identity_sha(identity: dict[str, Any], bits: int, batch: int,
                      candidate: int, source_sha: str, target_sha: str) -> str:
    payload = {"identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest(),
               "bits": bits, "batch_index": batch, "candidate_index": candidate,
               "source_sha256": source_sha, "target_z_sha256": target_sha,
               "fia": FIA}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _screen_candidate(public_root: Path, q: torch.Tensor, scale: torch.Tensor, *, bits: int,
                      batch: int, candidate: int, identity: dict[str, Any],
                      device: torch.device) -> dict[str, Any]:
    folder = public_root / f"bit{bits:02d}/batch{batch:02d}/member{candidate:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    sample, sample_record = _save_sample(folder, q, scale, bits=bits, batch=batch,
                                         candidate=candidate, identity=identity)
    source_path = folder / "sources.pt"
    cpa_path = folder / "cpa_result.json"
    if cpa_path.is_file():
        cpa_record = read_json(cpa_path)
        if cpa_record.get("input_sha256") != sample_record["sample_sha256"] or file_sha256(source_path) != cpa_record.get("sources_file_sha256"):
            raise TaskError("completed randomized CPA source record changed")
        sources = torch.load(source_path, map_location="cpu", weights_only=True)
        if tensor_sha256(sources) != cpa_record.get("sources_tensor_sha256"):
            raise TaskError("completed randomized CPA tensor hash changed")
    else:
        sources, cpa_result = _run_cpa(sample, stage_dir=folder, identity=identity,
                    batch=batch, device=device, label=f"random_{candidate:03d}", measure=False,
                    task_root=public_root)
        cpa_record = {"input_sha256": sample_record["sample_sha256"],
                      "sample_audit_sha256": file_sha256(folder / "sample.json"),
                      "sources_file_sha256": file_sha256(source_path),
                      "sources_tensor_sha256": tensor_sha256(sources),
                      "settings": cpa_result["settings"], "optimizer_log": cpa_result["optimizer_log"],
                      "wall_seconds": cpa_result["wall_seconds"],
                      "peak_cuda_allocated_bytes": cpa_result["peak_cuda_allocated_bytes"],
                      "private_data_used": False}
        write_json(cpa_path, cpa_record)
    del sample
    target_z = sources.abs().contiguous()
    target_sha = tensor_sha256(target_z)
    identity_sha = _fia_identity_sha(identity, bits, batch, candidate,
                                    cpa_record["sources_tensor_sha256"], target_sha)
    screen_path = folder / "screen_result.json"
    screen_identity = _screen_identity(identity, bits, batch, candidate, sample_record,
                                       cpa_record["sources_tensor_sha256"])
    if screen_path.is_file():
        result = read_json(screen_path)
        if (result.get("target_z_sha256") != target_sha
                or result.get("fia_identity_sha256") != identity_sha
                or file_sha256(folder / "screen_5000_rec.pt") != result.get("screen_rec_sha256")
                or file_sha256(source_path) != result.get("sources_file_sha256")):
            raise TaskError("screen marker/source mismatch")
        marker = verify_marker(folder / "screen_complete.json", screen_identity)
        if marker is None:
            save_marker(folder / "screen_complete.json", screen_identity,
                        ["sample.json", "sources.pt", "cpa_result.json", "screen_5000_rec.pt",
                         "screen_result.json"],
                        completed_iterations=PREFIX_STEPS, recovered_after_interrupted_write=True)
        checkpoint = folder / "fia/fia_checkpoint.pt"
        sidecar = folder / "fia/fia_checkpoint.pt.integrity.json"
        final_result_path = folder / "final_result.json"
        if checkpoint.is_file():
            expected_checkpoint_sha = result.get("checkpoint_5000_sha256")
            if final_result_path.is_file():
                expected_checkpoint_sha = read_json(final_result_path).get("final_checkpoint_sha256")
            expected_sidecar = checkpoint.with_suffix(checkpoint.suffix + ".integrity.json")
            if (file_sha256(checkpoint) != expected_checkpoint_sha or not expected_sidecar.is_file()):
                raise TaskError("screen/final checkpoint changed after its recorded endpoint")
        return result
    fia_dir = folder / "fia"
    result, attempt = _run_fia(target_z, output=fia_dir, stop=PREFIX_STEPS,
             source_sha=target_sha, identity_sha=identity_sha,
             device=device, attempt_name="screen_fia")
    screen_rec_path = folder / "screen_5000_rec.pt"
    from .same80_bits_r100_core import write_tensor
    write_tensor(screen_rec_path, result["rec"])
    score = _score_checkpoint(target_z, fia_dir / "fia_checkpoint.pt", output=fia_dir,
                source_sha=cpa_record["sources_tensor_sha256"], identity_sha=identity_sha,
                device=device, task_root=public_root)
    record = {"status": "COMPLETED", "bits": bits, "batch_index": batch,
              "candidate_index": candidate, "candidate_id": f"{candidate:03d}",
              "seed": candidate_seed(batch, candidate), "sample_sha256": sample_record["sample_sha256"],
              "sources_sha256": cpa_record["sources_tensor_sha256"], "target_z_sha256": target_sha,
              "sources_file_sha256": file_sha256(source_path), "fia_identity_sha256": identity_sha,
              "screen_rec_sha256": file_sha256(screen_rec_path), "screen_steps": PREFIX_STEPS,
              "screen_score": float(score["score"]), "score": score,
              "checkpoint_5000_sha256": file_sha256(fia_dir / "fia_checkpoint.pt"),
              "checkpoint_integrity_sha256": file_sha256(fia_dir / "fia_checkpoint.pt.integrity.json"),
              "fia_settings": result["settings"], "optimizer_log": result["log"],
              "wall_seconds": result["runner_wall_seconds"],
              "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
              "private_data_used": False, "selection_metric_based_on_truth": False}
    write_json(screen_path, record)
    save_marker(folder / "screen_complete.json", screen_identity,
                ["sample.json", "sources.pt", "cpa_result.json", "screen_5000_rec.pt",
                 "screen_result.json"],
                completed_iterations=PREFIX_STEPS)
    del sources, target_z
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def _rank_progress(public_root: Path, *, bits: int, batch: int, identity: dict[str, Any],
                   q: torch.Tensor, scale: torch.Tensor, device: torch.device) -> tuple[list[dict[str, Any]], list[int]]:
    cell = public_root / f"bit{bits:02d}/batch{batch:02d}"
    progress_path = cell / "screen_selection_progress.json"
    results = []
    selector = StreamingTop3()
    processed = []
    for candidate in range(N_CANDIDATES):
        folder = cell / f"member{candidate:03d}"
        result_path = folder / "screen_result.json"
        if result_path.is_file():
            record = _screen_candidate(public_root, q, scale, bits=bits, batch=batch,
                                       candidate=candidate, identity=identity, device=device)
            if int(record.get("candidate_index", -1)) != candidate or record.get("status") != "COMPLETED":
                raise TaskError("candidate screen record id/status mismatch")
        else:
            _disk_and_progress(public_root, state="RUNNING", stage="random_candidate_screen",
                               bits=bits, batch_index=batch, candidate_index=candidate,
                               total_candidates=N_CANDIDATES)
            record = _screen_candidate(public_root, q, scale, bits=bits, batch=batch,
                                       candidate=candidate, identity=identity, device=device)
        results.append(record)
        after, displaced = selector.add(record)
        processed.append(candidate)
        state = {"status": "RUNNING", "bits": bits, "batch_index": batch,
                 "processed_candidate_ids": processed,
                 "top3": [{"candidate_index": int(row["candidate_index"]),
                           "screen_score": float(row["screen_score"]),
                           "checkpoint_sha256": row["checkpoint_5000_sha256"]}
                          for row in selector.rows],
                 "all_scores": [{"candidate_index": int(row["candidate_index"]),
                                 "screen_score": float(row["screen_score"])} for row in results],
                 "private_data_used": False, "updated_at_utc": utc_now()}
        write_json(progress_path, state)
        # Ranking is durably updated before any displaced checkpoint is removed.
        cleanup_ids = set(displaced) | ({candidate} if candidate not in set(after) else set())
        for index in cleanup_ids:
            old = cell / f"member{index:03d}/fia/fia_checkpoint.pt"
            for name in ("fia_checkpoint.pt", "fia_checkpoint.pt.integrity.json"):
                path = old.with_name(name)
                if path.exists():
                    remove_task_disposable(path, public_root, allowed_names={name})
    full = rank_full(results, list(range(N_CANDIDATES)), "screen_score")
    if [int(x["candidate_index"]) for x in full[:TOP_K]] != [int(x["candidate_index"]) for x in selector.rows]:
        raise TaskError("streaming top3 differs from full public score sort")
    selection = {"status": "FROZEN_PUBLIC_SCREEN_SELECTION", "bits": bits,
                 "batch_index": batch, "ranking": [{"rank": rank + 1,
                    "candidate_index": int(row["candidate_index"]),
                    "screen_score": float(row["screen_score"]),
                    "checkpoint_5000_sha256": row["checkpoint_5000_sha256"],
                    "screen_rec_sha256": row["screen_rec_sha256"]}
                    for rank, row in enumerate(full)],
                 "selected_candidate_ids": [int(row["candidate_index"]) for row in full[:TOP_K]],
                 "tie_break": "candidate_index_ascending", "selection_score": PUBLIC_SCORE,
                 "selected_count": TOP_K, "candidate_count": N_CANDIDATES,
                 "private_data_used": False, "metric_based_selection": False,
                 "frozen_at_utc": utc_now()}
    final_selection_path = cell / "selection_5000.json"
    if final_selection_path.exists():
        old = read_json(final_selection_path)
        if old != selection:
            # Timestamp is not a selection input; retain the first frozen record only.
            comparable_old = {key: value for key, value in old.items() if key != "frozen_at_utc"}
            comparable_new = {key: value for key, value in selection.items() if key != "frozen_at_utc"}
            if comparable_old != comparable_new:
                raise TaskError("frozen 5000-step ranking changed on resume")
            selection = old
    else:
        write_json(final_selection_path, selection)
    kept = set(selection["selected_candidate_ids"])
    # Final selection is durable before releasing every non-finalist checkpoint.
    for candidate in range(N_CANDIDATES):
        if candidate not in kept:
            fia_dir = cell / f"member{candidate:03d}/fia"
            for name in ("fia_checkpoint.pt", "fia_checkpoint.pt.integrity.json"):
                path = fia_dir / name
                if path.exists():
                    remove_task_disposable(path, public_root, allowed_names={name})
            _cleanup_upstream_cache(fia_dir, public_root)
    write_json(progress_path, {**state, "status": "FROZEN", "selected_candidate_ids": selection["selected_candidate_ids"],
                               "selection_sha256": file_sha256(final_selection_path), "frozen_at_utc": utc_now()})
    return results, selection["selected_candidate_ids"]


def _cleanup_upstream_cache(folder: Path, task_root: Path) -> None:
    manifest = folder / "fia_manifest.json"
    if manifest.is_file():
        value = read_json(manifest)
        cache = Path(value["cache_root"]).resolve()
        if cache.exists():
            if (not cache.name.startswith("upstream_fia_")
                    or Path(folder).resolve() not in cache.parents):
                raise TaskError("unexpected upstream cache path; refusing cleanup")
            remove_task_disposable(cache, task_root, allowed_names={cache.name})


def _run_top3(public_root: Path, *, bits: int, batch: int, identity: dict[str, Any],
              selected: list[int], device: torch.device) -> tuple[list[dict[str, Any]], int]:
    cell = public_root / f"bit{bits:02d}/batch{batch:02d}"
    finals = []
    for candidate in selected:
        folder = cell / f"member{candidate:03d}"
        final_record_path = folder / "final_result.json"
        if final_record_path.is_file():
            saved = read_json(final_record_path)
            cp_final = folder / "fia/checkpoint_25000.pt"
            rec_final = folder / "final_25000_rec.pt"
            if (saved.get("candidate_index") != candidate
                    or not cp_final.is_file() or file_sha256(cp_final) != saved.get("final_checkpoint_sha256")
                    or not (folder / "fia/checkpoint_25000.pt.integrity.json").is_file()
                    or file_sha256(folder / "fia/checkpoint_25000.pt.integrity.json")
                       != saved.get("final_checkpoint_integrity_sha256")
                    or not rec_final.is_file() or file_sha256(rec_final) != saved.get("final_rec_sha256")
                    or not (folder / "fia/checkpoint_5000.pt").is_file()
                    or file_sha256(folder / "fia/checkpoint_5000.pt") != saved.get("screen_checkpoint_5000_sha256")
                    or not (folder / "fia/checkpoint_5000.pt.integrity.json").is_file()
                    or file_sha256(folder / "fia/checkpoint_5000.pt.integrity.json")
                       != saved.get("screen_checkpoint_integrity_sha256")):
                raise TaskError("completed continuation record/checkpoint hash mismatch")
            _cleanup_upstream_cache(folder / "fia", public_root)
            finals.append(saved)
            continue
        target_sources = torch.load(folder / "sources.pt", map_location="cpu", weights_only=True)
        target_z = target_sources.abs().contiguous()
        fia_dir = folder / "fia"
        checkpoint = fia_dir / "fia_checkpoint.pt"
        prefix_checkpoint = fia_dir / "checkpoint_5000.pt"
        prefix_sidecar = fia_dir / "checkpoint_5000.pt.integrity.json"
        screen_record = read_json(folder / "screen_result.json")
        if not prefix_checkpoint.exists() and checkpoint.exists():
            _copy_atomic(checkpoint, prefix_checkpoint)
            _copy_atomic(checkpoint.with_suffix(checkpoint.suffix + ".integrity.json"), prefix_sidecar)
        if not prefix_checkpoint.exists() or file_sha256(prefix_checkpoint) != screen_record.get("checkpoint_5000_sha256"):
            raise TaskError("selected candidate lost its authenticated 5000-step checkpoint")
        if not checkpoint.exists():
            _copy_atomic(prefix_checkpoint, checkpoint)
            _copy_atomic(prefix_sidecar, checkpoint.with_suffix(checkpoint.suffix + ".integrity.json"))
        identity_sha = screen_record["fia_identity_sha256"]
        result, attempt = _run_fia(target_z, output=fia_dir, stop=TOTAL_STEPS,
                   source_sha=tensor_sha256(target_z), identity_sha=identity_sha,
                   device=device, attempt_name="continuation_fia")
        rec_path = fia_dir / "rec_25000.pt"
        score = _score_checkpoint(target_z, fia_dir / "fia_checkpoint.pt", output=fia_dir,
                    source_sha=tensor_sha256(target_sources), identity_sha=identity_sha,
                    device=device, task_root=public_root)
        final_path = folder / "final_25000_rec.pt"
        if not final_path.exists():
            _copy_atomic(rec_path, final_path)
        checkpoint_final = fia_dir / "checkpoint_25000.pt"
        if not checkpoint_final.exists():
            _copy_atomic(fia_dir / "fia_checkpoint.pt", checkpoint_final)
            sidecar = fia_dir / "fia_checkpoint.pt.integrity.json"
            _copy_atomic(sidecar, fia_dir / "checkpoint_25000.pt.integrity.json")
        if (file_sha256(checkpoint_final) != file_sha256(fia_dir / "fia_checkpoint.pt")
                or file_sha256(fia_dir / "checkpoint_25000.pt.integrity.json")
                   != file_sha256(fia_dir / "fia_checkpoint.pt.integrity.json")):
            raise TaskError("final checkpoint copy differs from its authenticated source")
        record = {"status": "COMPLETED", "bits": bits, "batch_index": batch,
                  "candidate_index": candidate, "candidate_id": f"{candidate:03d}",
                  "completed_iterations": TOTAL_STEPS, "continued_from_step": PREFIX_STEPS,
                  "fia_identity_sha256": identity_sha,
                  "additional_steps": TOTAL_STEPS-PREFIX_STEPS,
                  "screen_checkpoint_5000_sha256": file_sha256(fia_dir / "checkpoint_5000.pt"),
                  "screen_checkpoint_integrity_sha256": file_sha256(fia_dir / "checkpoint_5000.pt.integrity.json"),
                  "final_checkpoint_sha256": file_sha256(checkpoint_final),
                  "final_checkpoint_integrity_sha256": file_sha256(fia_dir / "checkpoint_25000.pt.integrity.json"),
                  "final_rec_sha256": file_sha256(final_path),
                  "sources_sha256": tensor_sha256(target_sources),
                  "sample_sha256": read_json(folder / "sample.json")["sample_sha256"],
                  "target_z_sha256": tensor_sha256(target_z),
                  "final_score": float(score["score"]), "score": score,
                  "fia_settings": result["settings"], "optimizer_log": result["log"],
                  "wall_seconds": result["runner_wall_seconds"],
                  "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
                  "optimizer_state_continued": True, "private_data_used": False,
                  "metric_based_selection": False}
        write_json(folder / "final_result.json", record)
        _cleanup_upstream_cache(fia_dir, public_root)
        finals.append(record)
        del target_sources, target_z
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ranked = rank_full(finals, selected, "final_score")
    winner = int(ranked[0]["candidate_index"])
    final_selection = {"status": "FROZEN_PUBLIC_FINAL_SELECTION", "bits": bits,
                       "batch_index": batch, "screen_selected_candidate_ids": selected,
                       "final_ranking": [{"rank": rank + 1, "candidate_index": int(row["candidate_index"]),
                                          "final_score": float(row["final_score"]),
                                          "final_rec_sha256": row["final_rec_sha256"]}
                                         for rank, row in enumerate(ranked)],
                       "selected_candidate_index": winner, "selection_score": PUBLIC_SCORE,
                       "tie_break": "candidate_index_ascending", "private_data_used": False,
                       "metric_based_selection": False, "frozen_at_utc": utc_now()}
    path = cell / "final_selection.json"
    if path.exists():
        saved = read_json(path)
        for key in set(final_selection) - {"frozen_at_utc"}:
            if saved.get(key) != final_selection[key]:
                raise TaskError("25000-step public selection changed on resume")
        final_selection = saved
    else:
        write_json(path, final_selection)
    return finals, winner


def _verify_cell(public_root: Path, bits: int, batch: int, identity: dict[str, Any],
                 device: torch.device) -> dict[str, Any]:
    q, scale, packet = _load_packet(public_root, bits, batch, identity)
    cell = public_root / f"bit{bits:02d}/batch{batch:02d}"
    naive = _run_naive(public_root, q, scale, bits=bits, batch=batch,
                       identity=identity, device=device)
    screens, selected = _rank_progress(public_root, bits=bits, batch=batch,
                                       identity=identity, q=q, scale=scale, device=device)
    finals, winner = _run_top3(public_root, bits=bits, batch=batch, identity=identity,
                               selected=selected, device=device)
    if len(screens) != N_CANDIDATES or len(finals) != TOP_K or winner not in selected:
        raise TaskError("cell screen/final trajectory totals are invalid")
    if sorted(int(row["candidate_index"]) for row in screens) != list(range(N_CANDIDATES)):
        raise TaskError("cell candidate IDs must be exactly 000..099")
    cell_summary = {"status": "PUBLIC_CELL_COMPLETED", "bits": bits, "batch_index": batch,
                    "screen_candidates": N_CANDIDATES, "selected_finals": TOP_K,
                    "naive_trajectories": 1, "screen_selection_frozen": True,
                    "final_selection_frozen": True, "private_data_used": False,
                    "metric_based_selection": False, "naive_winner": "separate_center_baseline",
                    "screen_selected_candidate_ids": selected, "final_selected_candidate_index": winner,
                    "packet_sha256": file_sha256(cell / "observation/packet.json"),
                    "packet_codes_sha256": packet["codes_file_sha256"],
                    "naive_completion_sha256": file_sha256(cell / "naive/naive_complete.json"),
                    "selection_5000_sha256": file_sha256(cell / "selection_5000.json"),
                    "final_selection_sha256": file_sha256(cell / "final_selection.json"),
                    "screen_result_hashes": [file_sha256(cell / f"member{i:03d}/screen_result.json") for i in range(N_CANDIDATES)],
                    "final_result_hashes": [file_sha256(cell / f"member{i:03d}/final_result.json") for i in selected],
                    "winner_final_rec_sha256": file_sha256(cell / f"member{winner:03d}/final_25000_rec.pt"),
                    "naive_final_rec_sha256": naive["final_rec_sha256"],
                    "source_sha256": packet["packet_identity"]["source_Y_delta_over_lr_sha256"],
                    "storage_bytes_at_cell_completion": _tree_bytes(public_root),
                    "available_disk_bytes_at_cell_completion": shutil.disk_usage(public_root).free,
                    "updated_at_utc": utc_now()}
    write_json(cell / "cell_summary.json", cell_summary)
    del q, scale
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cell_summary


def _tree_bytes(root: Path) -> int:
    total = 0
    for path in Path(root).rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def run_public(config_path: Path, cfg: dict[str, Any], public_root: Path = PUBLIC_ROOT) -> dict[str, Any]:
    public_root = Path(public_root).resolve()
    registration, identity = _load_registration(public_root, config_path, cfg)
    if registration.get("status") == "PUBLIC_COMPLETED":
        return gate(public_root, config_path, cfg)
    if registration.get("packet_count") != len(BITS) * N_BATCHES:
        raise TaskError("prepare must create all 25 authenticated public packets")
    checkpoint = _resolve_project_path(cfg["checkpoint_path"])
    if file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise TaskError("checkpoint hash changed before attack")
    device = torch.device(cfg["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise TaskError("approved run requires the configured CUDA device")
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    _progress(public_root, state="RUNNING", stage="public_attack", worker_count=1,
              expected_cells=len(BITS)*N_BATCHES, private_data_used=False,
              started_at_utc=utc_now())
    cells = []
    try:
        for bits in BITS:
            for batch in range(N_BATCHES):
                _disk_and_progress(public_root, state="RUNNING", stage="cell_start",
                                   bits=bits, batch_index=batch, private_data_used=False)
                summary_path = public_root / f"bit{bits:02d}/batch{batch:02d}/cell_summary.json"
                if summary_path.is_file():
                    saved = read_json(summary_path)
                    if saved.get("status") != "PUBLIC_CELL_COMPLETED" or saved.get("bits") != bits or saved.get("batch_index") != batch:
                        raise TaskError("saved cell summary is invalid")
                    cells.append(saved)
                    continue
                cells.append(_verify_cell(public_root, bits, batch, identity, device))
        counts = verify_gate_counts(cells)
        gate = {"task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "PUBLIC_GATE_PASSED",
                "cells": [{"bits": row["bits"], "batch_index": row["batch_index"],
                           "cell_summary_sha256": file_sha256(public_root / f"bit{row['bits']:02d}/batch{row['batch_index']:02d}/cell_summary.json"),
                           "screen_candidates": row["screen_candidates"],
                           "selected_finals": row["selected_finals"],
                           "naive_trajectories": row["naive_trajectories"],
                           "screen_selection_frozen": row["screen_selection_frozen"],
                           "final_selection_frozen": row["final_selection_frozen"],
                           "private_data_used": row["private_data_used"]} for row in cells],
                "counts": counts, "public_selection_only": True,
                "private_data_loaded": False, "gated_at_utc": utc_now()}
        write_json(public_root / "public_gate.json", gate)
        registration["status"] = "PUBLIC_COMPLETED"
        registration["public_frozen"] = True
        registration["public_gate_sha256"] = file_sha256(public_root / "public_gate.json")
        registration["public_completed_at_utc"] = utc_now()
        write_json(public_root / "registration.json", registration)
        _progress(public_root, state="PUBLIC_GATE_PASSED", stage="complete",
                  completed_cells=len(cells), worker_count=1, private_data_used=False,
                  completed_at_utc=utc_now())
        return gate
    except BaseException as exc:
        _progress(public_root, state="FAILED", stage="public_attack",
                  error=f"{type(exc).__name__}: {exc}", worker_count=1,
                  failed_at_utc=utc_now())
        raise


def gate(public_root: Path = PUBLIC_ROOT, config_path: Path | None = None,
         cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    public_root = Path(public_root).resolve()
    registration = read_json(public_root / "registration.json")
    if config_path is not None and cfg is not None:
        _load_registration(public_root, config_path, cfg)
    identity = registration.get("identity", {})
    if not isinstance(identity, dict) or not isinstance(identity.get("code_sha256"), dict):
        raise TaskError("public registration has no authenticated code identity")
    verify_hash_map(identity["code_sha256"])
    if registration.get("status") != "PUBLIC_COMPLETED" or registration.get("public_frozen") is not True:
        raise TaskError("public selection has not been frozen")
    saved_gate = read_json(public_root / "public_gate.json")
    if (saved_gate.get("status") != "PUBLIC_GATE_PASSED"
            or saved_gate.get("private_data_loaded") is not False
            or registration.get("public_gate_sha256") != file_sha256(public_root / "public_gate.json")):
        raise TaskError("public gate record is invalid")
    cells = []
    for bits in BITS:
        for batch in range(N_BATCHES):
            folder = public_root / f"bit{bits:02d}/batch{batch:02d}"
            row = read_json(folder / "cell_summary.json")
            q, scale, packet = _load_packet(public_root, bits, batch, identity)
            if (row.get("status") != "PUBLIC_CELL_COMPLETED" or row.get("bits") != bits
                    or row.get("batch_index") != batch):
                raise TaskError("missing or mismatched bit/batch cell summary")
            if (row.get("packet_sha256") != file_sha256(folder / "observation/packet.json")
                    or row.get("packet_codes_sha256") != packet.get("codes_file_sha256")):
                raise TaskError("cell summary no longer matches its public code packet")
            for candidate in range(N_CANDIDATES):
                candidate_folder = folder / f"member{candidate:03d}"
                path = candidate_folder / "screen_result.json"
                if not path.is_file():
                    raise TaskError(f"missing screened candidate: {path}")
                result = read_json(path)
                if result.get("candidate_index") != candidate or result.get("screen_steps") != PREFIX_STEPS:
                    raise TaskError("candidate screen step/id gate failed")
                _validate_fia_record(result, completed=PREFIX_STEPS)
                if result.get("private_data_used") is not False or result.get("selection_metric_based_on_truth") is not False:
                    raise TaskError("screen result declares private or truth-based selection")
                sample = read_json(candidate_folder / "sample.json")
                cpa_record = read_json(candidate_folder / "cpa_result.json")
                _validate_cpa_record(cpa_record, batch)
                if (sample.get("seed") != candidate_seed(batch, candidate)
                        or sample.get("chunk_rows") != SAMPLER_CHUNK_ROWS
                        or sample.get("packet_scale") != float(scale)
                        or sample.get("sample_scale_reproduced") is not True
                        or sample.get("code_consistency", {}).get("all_equal") is not True
                        or sample.get("sample_sha256") != cpa_record.get("input_sha256")):
                    raise TaskError("sample seed, scale, or exact code reproduction gate failed")
                screen_identity = _screen_identity(identity, bits, batch, candidate, sample,
                                                   cpa_record["sources_tensor_sha256"])
                if verify_marker(candidate_folder / "screen_complete.json", screen_identity) is None:
                    raise TaskError("candidate screen completion marker is missing")
                if (file_sha256(candidate_folder / "sources.pt") != cpa_record.get("sources_file_sha256")
                        or file_sha256(candidate_folder / "screen_5000_rec.pt") != result.get("screen_rec_sha256")):
                    raise TaskError("screen reconstruction hash mismatch")
            selection = read_json(folder / "selection_5000.json")
            finalists = selection.get("selected_candidate_ids", [])
            if len(finalists) != TOP_K or len(set(finalists)) != TOP_K:
                raise TaskError("top-three screen selection missing")
            screen_rows = [read_json(folder / f"member{i:03d}/screen_result.json") for i in range(N_CANDIDATES)]
            expected_screen_order = [int(row["candidate_index"]) for row in rank_full(screen_rows, list(range(N_CANDIDATES)), "screen_score")]
            if [int(row["candidate_index"]) for row in selection.get("ranking", [])] != expected_screen_order:
                raise TaskError("frozen screen ranking does not match the full finite public sort")
            if finalists != expected_screen_order[:TOP_K] or selection.get("private_data_used") is not False:
                raise TaskError("screen top-three selection used wrong candidates or private data")
            final_selection = read_json(folder / "final_selection.json")
            if final_selection.get("screen_selected_candidate_ids") != finalists:
                raise TaskError("final ranking candidates differ from frozen screen top-three")
            final_rows = [read_json(folder / f"member{int(i):03d}/final_result.json") for i in finalists]
            expected_final_order = [int(row["candidate_index"]) for row in rank_full(final_rows, finalists, "final_score")]
            if ([int(row["candidate_index"]) for row in final_selection.get("final_ranking", [])] != expected_final_order
                    or final_selection.get("selected_candidate_index") != expected_final_order[0]
                    or final_selection.get("private_data_used") is not False):
                raise TaskError("final public reranking changed or used private data")
            for candidate in finalists:
                cfolder = folder / f"member{int(candidate):03d}"
                result = read_json(cfolder / "final_result.json")
                screen_result = read_json(cfolder / "screen_result.json")
                _validate_fia_record(result, completed=TOTAL_STEPS)
                prefix_integrity = _integrity_sidecar(cfolder / "fia/checkpoint_5000.pt",
                    horizon=TOTAL_STEPS, identity_sha=result["fia_identity_sha256"], kind="fia")
                final_integrity = _integrity_sidecar(cfolder / "fia/checkpoint_25000.pt",
                    horizon=TOTAL_STEPS, identity_sha=result["fia_identity_sha256"], kind="fia")
                if (prefix_integrity is None or prefix_integrity["completed_iterations"] != PREFIX_STEPS
                        or final_integrity is None or final_integrity["completed_iterations"] != TOTAL_STEPS
                        or result.get("screen_checkpoint_5000_sha256")
                           != screen_result.get("checkpoint_5000_sha256")):
                    raise TaskError("FIA continuation checkpoint state/RNG/horizon gate failed")
                if (result.get("completed_iterations") != TOTAL_STEPS
                        or result.get("additional_steps") != TOTAL_STEPS-PREFIX_STEPS
                        or result.get("optimizer_state_continued") is not True
                        or result.get("private_data_used") is not False
                        or result.get("metric_based_selection") is not False
                        or result.get("sources_sha256") != screen_result.get("sources_sha256")
                        or result.get("sample_sha256") != screen_result.get("sample_sha256")
                        or file_sha256(cfolder / "final_25000_rec.pt") != result.get("final_rec_sha256")
                        or file_sha256(cfolder / "fia/checkpoint_5000.pt") != result.get("screen_checkpoint_5000_sha256")
                        or file_sha256(cfolder / "fia/checkpoint_5000.pt.integrity.json")
                           != result.get("screen_checkpoint_integrity_sha256")
                        or file_sha256(cfolder / "fia/checkpoint_25000.pt") != result.get("final_checkpoint_sha256")
                        or file_sha256(cfolder / "fia/checkpoint_25000.pt.integrity.json")
                           != result.get("final_checkpoint_integrity_sha256")):
                    raise TaskError("selected final endpoint/checkpoint validation failed")
            naive_dir = folder / "naive"
            naive_input = read_json(naive_dir / "input.json")
            center = (q.to(torch.float32) * scale).contiguous()
            center_sha = tensor_sha256(center)
            del center
            if (naive_input.get("input") != "observed_codes_times_observed_scale"
                    or naive_input.get("sample_sha256") != center_sha
                    or naive_input.get("scale_exact") is not True
                    or naive_input.get("code_consistency") is not True):
                raise TaskError("Naive center is not the exact observed q times scale")
            naive_marker = verify_marker(naive_dir / "naive_complete.json",
                                         {"stage_sha256": _naive_stage_id(identity, q, scale, bits, batch)})
            if naive_marker is None:
                raise TaskError("separate Naive center trajectory is missing")
            naive = read_json(naive_dir / "naive_complete.json")
            naive_result = read_json(naive_dir / "fia/final_result.json")
            naive_cpa = read_json(naive_dir / "cpa_result.json")
            _validate_cpa_record(naive_cpa, batch)
            _validate_fia_record(naive_result, completed=TOTAL_STEPS)
            naive_checkpoint = _integrity_sidecar(naive_dir / "fia/checkpoint_25000.pt",
                horizon=TOTAL_STEPS, identity_sha=_naive_stage_id(identity, q, scale, bits, batch),
                kind="fia")
            if naive_checkpoint is None or naive_checkpoint["completed_iterations"] != TOTAL_STEPS:
                raise TaskError("Naive Direct FIA endpoint checkpoint state/horizon gate failed")
            if (naive.get("naive_trajectories") != 1
                    or not (folder / "naive/fia/checkpoint_25000.pt").is_file()
                    or file_sha256(folder / "naive/fia/checkpoint_25000.pt")
                       != naive_result.get("checkpoint_25000_sha256")
                    or file_sha256(folder / "naive/fia/checkpoint_25000.pt.integrity.json")
                       != naive_result.get("checkpoint_integrity_sha256")):
                raise TaskError("Naive 25000 endpoint/checkpoint validation failed")
            del q, scale
            cells.append(row)
    counts = verify_gate_counts(cells)
    if saved_gate.get("counts") != counts:
        raise TaskError("public gate counts changed or do not match the frozen plan")
    saved_ids = {(int(item["bits"]), int(item["batch_index"])) for item in saved_gate.get("cells", [])}
    expected_ids = {(bits, batch) for bits in BITS for batch in range(N_BATCHES)}
    if len(saved_gate.get("cells", [])) != len(expected_ids) or saved_ids != expected_ids:
        raise TaskError("public gate cell coverage changed or is incomplete")
    resource = read_json(public_root / "resource_measurement.json")
    projected = float(resource.get("projected_remaining_wall_seconds", float("nan")))
    if (resource.get("measured_candidate") != "first scheduled Naive center trajectory bit08 batch00"
            or not np.isfinite(projected) or projected < 0
            or resource.get("estimate_uncertainty_fraction") != 0.5):
        raise TaskError("first-candidate resource measurement/projection is missing or invalid")
    for entry in saved_gate.get("cells", []):
        path = public_root / f"bit{int(entry['bits']):02d}/batch{int(entry['batch_index']):02d}/cell_summary.json"
        if file_sha256(path) != entry.get("cell_summary_sha256"):
            raise TaskError("gated cell summary changed after freeze")
    return {"status": "PUBLIC_GATE_PASSED", "counts": counts,
            "gate_sha256": file_sha256(public_root / "public_gate.json"),
            "private_data_loaded": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "run", "gate"), required=True)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    cfg = read_json(config_path)
    _validate_config(cfg)
    public_root = _resolve_project_path(cfg["public_root"])
    private_root = _resolve_project_path(cfg["private_root"])
    with execution_lock(public_root):
        if args.phase == "prepare":
            result = prepare(config_path, cfg, public_root, private_root)
        elif args.phase == "run":
            result = run_public(config_path, cfg, public_root)
        else:
            result = gate(public_root, config_path, cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
