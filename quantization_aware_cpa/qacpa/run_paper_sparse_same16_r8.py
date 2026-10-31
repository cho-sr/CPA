"""Execute the authorized v3 SAME16 random-fill paper sweep.

The task keeps the authenticated Top-k 50/30/20 packets and the public
protocol, while replacing the evaluated image scope with original batch00
global indices 0..15.  The completed v2 adapter is used only for the attack
kernel; this module owns the v3 identity, input manifest, migration, budget,
selection and batch loop.
"""
from __future__ import annotations

from contextlib import contextmanager
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import run_paper_sparse_same80 as legacy
from . import run_paper_sparse_same80_r8 as r8_adapter
from .run_m8_fedavg_naive import (
    PROJECT, WORKSPACE, read_json, write_json, read_csv, write_csv, file_hash,
    resolve, utc_now, status as _status, execution_lock, evaluate_after_fia,
)
from .random_fill import candidate_seed, rank_candidates
from .paper_sparse_codec import SOURCE_CODECS, expected_packet_bytes, validate_received_packet
from .fia import evaluate_images


TASK_ID = "CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921"
PLAN_VERSION = 3
AMENDMENT_ID = "CPA_PAPER_SPARSE_SAME16_R8_AMENDMENT_20260921"
BASE_PLAN_VERSION = 1
PERCENTS = (50, 30, 20)
CANDIDATE_COUNT = 8
CANDIDATES = tuple(range(CANDIDATE_COUNT))
N_BATCHES = 1
N_IMAGES = 16
N_PER_BATCH = 16
T_STEPS = 10
FC1_SHAPE = (4096, 25088)
BATCH_INDICES = (0,)
SAME16_INPUT_SHA256 = "24995487511a3e9d8d4665fdcd704f7a0f792249a9ba79880638d13bc4740141"
R20_TAG = "results/cpa_fia_random_topk{percent}_same80_R20_S5k_K3_F25k"
R8_SAME80_TAG = "results/cpa_fia_random_topk{percent}_same80_R8_S5k_K3_F25k"
SAME16_TAG = "cpa_fia_random_topk{percent}_same16_R8_S5k_K3_F25k"
SWEEP_OUTPUT_ROOT = "results/paper_sparse_same16_keep50_30_20_R8_20260921"
AMENDMENT_PATH = WORKSPACE / ".ai/execution-handoffs/CPA_PAPER_SPARSE_SAME16_R8_AMENDMENT_20260921.md"
V1_ARCHIVE_PATH = WORKSPACE / ".ai/execution-handoffs/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921_V1_CLAIMED_ARCHIVE_20260921.md"
V2_ARCHIVE_PATH = WORKSPACE / ".ai/execution-handoffs/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921_V2_R8_CLAIMED_ARCHIVE_20260921.md"

RANDOMIZATION = {
    "sampler": "uniform_tau_pcg64_v1",
    "tau": "min(abs(transmitted FP32 values))",
    "candidates": CANDIDATE_COUNT,
    "screen_iterations": 5000,
    "advance_top_k": 3,
    "final_iterations": 25000,
    "cosine_horizon": 25000,
    "fill_seed": "100000 + 100*batch_index + candidate_index; same plan across fractions",
    "cpa_seed": "baseline seed + batch_index, identical for all candidates",
    "fia_seed": "baseline FIA seed42, identical for all candidates",
    "ranking": "relative L2 of recompressed reconstructed-data SGD update to received packet",
    "ranking_labels": "public round0 model argmax; no true labels",
    "tie_break": "candidate_index ascending",
    "selection_unit": "whole16-image candidate for original batch00; candidate IDs 0..7 only",
    "final_selection": "lowest same observable relative L2 among three final candidates",
    "true_metrics_for_selection": False,
}


def task_status(output: Path, cfg: dict[str, Any], **kwargs: Any) -> None:
    _status(output, cfg, task_id=TASK_ID, plan_version=PLAN_VERSION,
            amendment_id=AMENDMENT_ID, candidate_count=CANDIDATE_COUNT,
            keep_percent=cfg.get("keep_percent"), **kwargs)


def _baseline_config(percent: int) -> dict[str, Any]:
    return read_json(PROJECT / f"configs/cpa_fia_topk{percent}_same80_B16.json")


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    percent = cfg.get("keep_percent")
    if type(percent) is not int or percent not in PERCENTS:
        raise ValueError("Only retained fractions 50, 30 and 20 are authorized")
    baseline = _baseline_config(percent)
    legacy._validate_common_baseline(baseline, percent)
    changed = {
        "run_tag", "output_dir", "observation", "execution_scope", "keep_percent",
        "baseline_output", "randomization", "source_codec", "plan_version",
        "amendment_id", "candidate_count", "candidate_ids", "n_images",
        "batch_indices", "grid", "selection", "local_training",
    }
    expected_keys = set(baseline) | changed
    if set(cfg) != expected_keys:
        raise ValueError("Unexpected paper sparse SAME16 configuration fields")
    for key, value in baseline.items():
        if key not in changed and cfg.get(key) != value:
            raise ValueError(f"Matched baseline condition changed: {key}")
    tag = SAME16_TAG.format(percent=percent)
    if cfg["run_tag"] != tag or cfg["output_dir"] != f"results/{tag}":
        raise ValueError("Paper sparse SAME16 results must use a task-local output")
    if cfg["baseline_output"] != baseline["output_dir"]:
        raise ValueError("The SAME16 naive comparison must use its matched same80 baseline rows")
    if cfg["source_codec"] != SOURCE_CODECS[percent] or cfg["codec"] != baseline["codec"]:
        raise ValueError("The historical source codec label changed")
    if cfg.get("expected_packet_bytes") != expected_packet_bytes(percent):
        raise ValueError("The configured packet byte count is wrong")
    if cfg["plan_version"] != PLAN_VERSION or cfg["amendment_id"] != AMENDMENT_ID:
        raise ValueError("The v3 SAME16 amendment identity is missing")
    if cfg["candidate_count"] != CANDIDATE_COUNT or cfg["candidate_ids"] != list(CANDIDATES):
        raise ValueError("SAME16 requires exactly candidate IDs 0..7")
    if cfg.get("n") != 16 or cfg.get("T") != 10:
        raise ValueError("The B16/T10 attack grouping is fixed")
    if cfg.get("n_images") != N_IMAGES or cfg.get("batch_indices") != [0]:
        raise ValueError("SAME16 evaluates original batch00 only")
    if cfg.get("grid") != {"n": 16, "selection": "original batch00; global_index0..15"}:
        raise ValueError("The SAME16 grid scope is invalid")
    if cfg.get("selection") != "original batch00 in original global_index order; no LPIPS selection":
        raise ValueError("SAME16 selection scope is invalid")
    expected_local = dict(baseline["local_training"])
    expected_local["independent_client_updates"] = 1
    if cfg.get("local_training") != expected_local:
        raise ValueError("The authorized SAME16 local-training provenance is invalid")
    if cfg["sparsification"]["keep_fraction"] != percent / 100:
        raise ValueError("Only the retained fraction may vary")
    if cfg["randomization"] != RANDOMIZATION:
        raise ValueError("SAME16 random-fill selection protocol differs from the amendment")
    return baseline


def _actual_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE / path


def _verify_code_hashes(code_hashes: dict[str, str], *, label: str) -> None:
    for rel, expected in code_hashes.items():
        # v2 manifests retain v1 provenance keys such as
        # ``v1_snapshot::quantization_aware_cpa\\qacpa\\...``.
        path = _actual_path(rel.split("::")[-1])
        if not path.exists() or file_hash(path) != expected:
            raise RuntimeError(f"{label} dependency changed: {rel}")


def dependency_hashes(baselines: dict[int, dict[str, Any]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for summary in baselines.values():
        for rel, expected in summary["identity"]["code_sha256"].items():
            path = _actual_path(rel)
            if not path.exists() or file_hash(path) != expected:
                raise RuntimeError(f"Frozen naive dependency changed: {rel}")
            hashes[rel] = expected
    for percent in PERCENTS:
        for tag in (R20_TAG, R8_SAME80_TAG):
            old_manifest = read_json(resolve(tag.format(percent=percent)) / "input_manifest.json")
            _verify_code_hashes(old_manifest["identity"]["code_sha256"], label=f"Frozen {tag} Top-k{percent}")
            for rel, expected in old_manifest["identity"]["code_sha256"].items():
                hashes[f"provenance::{tag}::{rel}"] = expected
    new_paths = [
        PROJECT / "qacpa/paper_sparse_codec.py",
        PROJECT / "qacpa/run_paper_sparse_same80.py",
        PROJECT / "qacpa/run_paper_sparse_same80_r8.py",
        PROJECT / "qacpa/report_paper_sparse_same80_r8.py",
        PROJECT / "qacpa/run_paper_sparse_same16_r8.py",
        PROJECT / "qacpa/report_paper_sparse_same16_r8.py",
        PROJECT / "scripts/run_paper_sparse_same16_r8.py",
        PROJECT / "docs/RANDOM_FILL_SAME80.md",
        PROJECT / "qacpa/random_fill.py",
        PROJECT / "qacpa/staged_fia.py",
        PROJECT / "qacpa/upstream_naive.py",
        AMENDMENT_PATH, V1_ARCHIVE_PATH, V2_ARCHIVE_PATH,
    ]
    for path in new_paths:
        if not path.exists():
            raise RuntimeError(f"Missing v3 dependency: {path}")
        hashes[str(path.relative_to(WORKSPACE))] = file_hash(path)
    return hashes


@contextmanager
def _adapter_scope():
    """Run the authenticated v2 attack kernel with the v3 identity."""
    saved = {
        "TASK_ID": r8_adapter.TASK_ID,
        "PLAN_VERSION": r8_adapter.PLAN_VERSION,
        "AMENDMENT_ID": r8_adapter.AMENDMENT_ID,
        "RANDOMIZATION": r8_adapter.RANDOMIZATION,
        "validate_config": r8_adapter.validate_config,
        "dependency_hashes": r8_adapter.dependency_hashes,
    }
    r8_adapter.TASK_ID = TASK_ID
    r8_adapter.PLAN_VERSION = PLAN_VERSION
    r8_adapter.AMENDMENT_ID = AMENDMENT_ID
    r8_adapter.RANDOMIZATION = RANDOMIZATION
    r8_adapter.validate_config = validate_config
    r8_adapter.dependency_hashes = dependency_hashes
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(r8_adapter, key, value)


def _source_input_hashes(source_manifest: dict[str, Any]) -> dict[str, str]:
    values = {
        rel: digest for rel, digest in source_manifest["inputs_sha256"].items()
        if "batch00" in rel.lower()
    }
    input_key = next((rel for rel in values if rel.lower().endswith("batch00\\inputs.pt")), None)
    if input_key is None or values[input_key] != SAME16_INPUT_SHA256:
        raise RuntimeError("The authenticated original batch00 inputs.pt hash is missing")
    return values


def _verify_source_batch(source_manifest: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    if len(source_manifest.get("metadata", [])) != 80:
        raise RuntimeError("The authenticated source manifest must retain its historical 80-row provenance")
    input_path = resolve(cfg["source_output"]) / "batch00/inputs.pt"
    if file_hash(input_path) != SAME16_INPUT_SHA256:
        raise RuntimeError("Original batch00 inputs.pt changed")
    inputs = torch.load(input_path, map_location="cpu", weights_only=False, mmap=True)
    if tuple(inputs["x"].shape) != (16, 3, 224, 224) or tuple(inputs["Z_evaluation_only"].shape) != (16, 25088):
        raise RuntimeError("Invalid original batch00 tensors")
    if legacy._identity_rows(inputs["metadata"]) != legacy._identity_rows(source_manifest["metadata"][:16]):
        raise RuntimeError("Original batch00 identity/order changed")
    if [int(row["global_index"]) for row in inputs["metadata"]] != list(range(16)):
        raise RuntimeError("SAME16 must use original global indices 0..15")
    weights = Path(source_manifest["weights_path"])
    if file_hash(weights) != source_manifest["weights_sha256"]:
        raise RuntimeError("Public round0 checkpoint changed")
    return inputs


def verify_inputs(manifest: dict[str, Any]) -> None:
    for rel, expected in manifest["inputs_sha256"].items():
        path = resolve(rel)
        if file_hash(path) != expected:
            raise RuntimeError(f"Frozen SAME16 source changed: {rel}")
    weights = _actual_path(manifest["weights_path"])
    if file_hash(weights) != manifest["weights_sha256"]:
        raise RuntimeError("Frozen public checkpoint changed")


def prepare(config_path: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    validate_config(cfg)
    all_baselines: dict[int, dict[str, Any]] = {}
    all_manifests: dict[int, dict[str, Any]] = {}
    for fraction in PERCENTS:
        summary, manifest = legacy._verify_baseline_complete(fraction)
        all_baselines[fraction] = summary
        all_manifests[fraction] = manifest
    metadata = all_manifests[50]["metadata"][:N_IMAGES]
    for fraction in PERCENTS:
        if legacy._identity_rows(all_manifests[fraction]["metadata"][:N_IMAGES]) != legacy._identity_rows(metadata):
            raise RuntimeError("Original batch00 identities differ across sparse conditions")
    source_manifest = all_manifests[cfg["keep_percent"]]
    _verify_source_batch(source_manifest, cfg)
    hashes = dependency_hashes(all_baselines)
    source = resolve(cfg["baseline_output"])
    folder = source / "batch00"
    packet_path = folder / "observation.bin"
    packet = packet_path.read_bytes()
    comm = legacy._communication(folder, 0, cfg["keep_percent"])
    batch_summary = read_json(folder / "summary.json")
    expected_sha = batch_summary["output_hashes"]["observation.bin"]
    actual_sha = file_hash(packet_path)
    if actual_sha != expected_sha or actual_sha != comm["sha256"]:
        raise RuntimeError("Saved batch00 packet differs from the matched baseline")
    if comm["packet_bytes"] != expected_packet_bytes(cfg["keep_percent"]):
        raise RuntimeError("Unexpected batch00 packet byte count")
    audit = validate_received_packet(packet, cfg["keep_percent"], cfg["source_codec"])
    audit.update({
        "batch_index": 0, "path": str(packet_path), "sha256": actual_sha,
        "baseline_summary_sha256": file_hash(folder / "summary.json"),
        "baseline_communication_sha256": file_hash(folder / "communication.json"),
    })
    packet_ref = {
        "batch_index": 0, "path": str(packet_path),
        "relative_path": str(packet_path.relative_to(PROJECT)),
        "sha256": actual_sha, "bytes": len(packet), "source_codec": cfg["source_codec"],
        "retained_elements": audit["kept_elements"],
    }
    source_inputs = _source_input_hashes(source_manifest)
    identity = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
        "keep_percent": cfg["keep_percent"], "config_sha256": file_hash(config_path),
        "code_sha256": hashes,
        "baseline_summary_sha256": file_hash(source / "summary.json"),
        "baseline_manifest_sha256": file_hash(source / "input_manifest.json"),
        "source_inputs_sha256": source_inputs, "weights_sha256": source_manifest["weights_sha256"],
        "weights_path": source_manifest["weights_path"], "packets": [packet_ref],
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "input_manifest.json"
    audit_path = output / "packet_audit.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("identity") != identity:
            raise RuntimeError("Incompatible SAME16 resume: frozen input/code/config changed")
        if legacy._identity_rows(manifest.get("metadata", [])) != legacy._identity_rows(metadata):
            raise RuntimeError("Original batch00 identity changed")
        if file_hash(audit_path) != manifest.get("packet_audit_sha256"):
            raise RuntimeError("Frozen SAME16 packet audit changed")
        verify_inputs(manifest)
        return manifest
    packet_audit = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
        "keep_percent": cfg["keep_percent"], "source_codec": cfg["source_codec"],
        "batches": [audit],
    }
    write_json(audit_path, packet_audit)
    manifest = {
        "schema_version": 3, "task_id": TASK_ID, "plan_version": PLAN_VERSION,
        "amendment_id": AMENDMENT_ID, "run_tag": cfg["run_tag"], "identity": identity,
        "git_sha": source_manifest.get("git_sha", ""), "inputs_sha256": source_inputs,
        "weights_path": source_manifest["weights_path"], "weights_sha256": source_manifest["weights_sha256"],
        "metadata": metadata, "n_images": N_IMAGES, "n": N_PER_BATCH, "T": T_STEPS,
        "batch_indices": [0], "created_at_utc": utc_now(), "new_training": False,
        "new_update_collection": False, "source_codec": cfg["source_codec"],
        "randomization": RANDOMIZATION, "communication_per_batch": [comm],
        "packet_audits": [audit], "packet_audit_sha256": file_hash(audit_path),
        "baseline_output": cfg["baseline_output"],
        "baseline_summary_sha256": file_hash(source / "summary.json"),
        "baseline_manifest_sha256": file_hash(source / "input_manifest.json"),
        "scope": "original batch00 only; global_index0..15; batches01..04 prohibited",
    }
    write_csv(output / "image_manifest.csv", metadata)
    write_json(manifest_path, manifest)
    write_json(config_path.with_suffix(".lock.json"), {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
        "config_sha256": identity["config_sha256"], "input_manifest_sha256": file_hash(manifest_path),
        "packet_audit_sha256": file_hash(audit_path), "frozen_at_utc": utc_now(),
    })
    return manifest


def checked_marker(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    return legacy.checked_marker(path, identity)


def save_marker(path: Path, identity: dict[str, Any], names: list[str], **values: Any) -> dict[str, Any]:
    return legacy.save_marker(path, identity, names, **values)


def _copy(source: Path, target: Path) -> str:
    if not source.exists():
        raise RuntimeError(f"Required migration artifact is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return file_hash(target)


def _source_manifest(root: Path, percent: int, manifest: dict[str, Any]) -> dict[str, Any]:
    path = root / "input_manifest.json"
    if not path.exists():
        raise RuntimeError(f"Authenticated source manifest is missing: {path}")
    source = read_json(path)
    if source.get("task_id") != TASK_ID or source.get("plan_version") not in (BASE_PLAN_VERSION, 2):
        raise RuntimeError(f"Unexpected source plan for Top-k{percent}: {path}")
    if source.get("n_images") != 80 or len(source.get("metadata", [])) != 80:
        raise RuntimeError(f"Source candidate manifest is not historical same80: {path}")
    if legacy._identity_rows(source["metadata"][:16]) != legacy._identity_rows(manifest["metadata"]):
        raise RuntimeError(f"Source and SAME16 batch00 identities differ: {path}")
    old_packet = source["identity"]["packets"][0]
    new_packet = manifest["identity"]["packets"][0]
    if (old_packet["sha256"], old_packet["bytes"]) != (new_packet["sha256"], new_packet["bytes"]):
        raise RuntimeError(f"Source and SAME16 packet identities differ: {path}")
    _verify_code_hashes(source["identity"]["code_sha256"], label=f"Frozen source Top-k{percent}")
    return source


def _verify_importable(source_dir: Path, source_manifest: dict[str, Any], cfg: dict[str, Any], candidate: int) -> dict[str, Any]:
    return r8_adapter._verify_importable_candidate(source_dir, cfg, source_manifest, 0, candidate)


def _import_one(source_root: Path, new_dir: Path, source_manifest: dict[str, Any], cfg: dict[str, Any], candidate: int,
                manifest: dict[str, Any]) -> dict[str, Any]:
    source_dir = source_root / "batch00" / f"candidate{candidate:02d}"
    if not source_dir.exists():
        return {"batch_index": 0, "candidate": candidate, "status": "fresh_no_authenticated_artifact"}
    verified = _verify_importable(source_dir, source_manifest, cfg, candidate)
    new_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in ("sources.pt", "fill_audit.json", "cpa_log.json"):
        copied[name] = _copy(source_dir / name, new_dir / name)
    provenance = dict(verified["provenance"])
    provenance.update({
        "source_root": str(source_root), "source_manifest_sha256": file_hash(source_root / "input_manifest.json"),
        "source_plan_version": source_manifest["plan_version"], "source_scope": "authenticated same80 batch00 candidate state",
        "v3_scope": "same16 original batch00; candidate IDs0..7",
    })
    cpa_values = {key: value for key, value in verified["cpa"].items() if key not in ("identity", "output_hashes")}
    cpa_values["imported_artifact_provenance"] = provenance
    save_marker(new_dir / "cpa_result.json", manifest["identity"],
                ["sources.pt", "fill_audit.json", "cpa_log.json"], **cpa_values)
    checkpoint = verified["checkpoint"]
    if checkpoint is None:
        return {"batch_index": 0, "candidate": candidate, "status": "imported_cpa_only",
                "provenance": provenance, "new_cpa_marker_sha256": file_hash(new_dir / "cpa_result.json")}
    _copy(checkpoint, new_dir / "fia/fia_checkpoint.pt")
    source_fia_manifest = source_dir / "fia/fia_manifest.json"
    if source_fia_manifest.exists():
        _copy(source_fia_manifest, new_dir / "fia/fia_manifest.json")
    source_progress = source_dir / "fia/progress.json"
    if source_progress.exists():
        _copy(source_progress, new_dir / "fia/progress.json")
    result = {
        "batch_index": 0, "candidate": candidate,
        "provenance": provenance,
        "new_cpa_marker_sha256": file_hash(new_dir / "cpa_result.json"),
        "new_checkpoint_sha256": file_hash(new_dir / "fia/fia_checkpoint.pt"),
    }
    if verified["screen"] is not None:
        for name in ("screen_rec.pt", "screen_fia_log.json"):
            _copy(source_dir / name, new_dir / name)
        screen_values = {key: value for key, value in verified["screen"].items()
                         if key not in ("identity", "output_hashes")}
        screen_values["imported_artifact_provenance"] = provenance
        save_marker(new_dir / "screen.json", manifest["identity"],
                    ["screen_rec.pt", "screen_fia_log.json", "sources.pt", "cpa_result.json"], **screen_values)
        result.update({"status": "imported_cpa_and_completed_screen",
                       "new_screen_marker_sha256": file_hash(new_dir / "screen.json")})
    else:
        result["status"] = "imported_cpa_and_partial_screen_checkpoint"
        result["new_screen_marker_sha256"] = None
    return result


def migrate_fraction(cfg: dict[str, Any], manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    path = output / "migration_manifest.json"
    if path.exists():
        saved = read_json(path)
        if saved.get("identity") != manifest["identity"] or saved.get("plan_version") != PLAN_VERSION:
            raise RuntimeError(f"Incompatible existing SAME16 migration manifest: {path}")
        return saved
    percent = int(cfg["keep_percent"])
    v2_root = resolve(R8_SAME80_TAG.format(percent=percent))
    v1_root = resolve(R20_TAG.format(percent=percent))
    source_roots = []
    # Prefer the original authenticated R20 directory because it retains the
    # complete upstream CPA/FIA manifests; the v2 migration remains a valid
    # provenance source but intentionally copied only resume artifacts.
    for root in (v1_root, v2_root):
        if (root / "input_manifest.json").exists():
            source_roots.append((root, _source_manifest(root, percent, manifest)))
    if not source_roots:
        raise RuntimeError(f"No authenticated same80 candidate source exists for Top-k{percent}")
    records = []
    for candidate in CANDIDATES:
        imported = None
        for source_root, source_manifest in source_roots:
            source_dir = source_root / "batch00" / f"candidate{candidate:02d}"
            if source_dir.exists():
                imported = _import_one(source_root, output / "batch00" / f"candidate{candidate:02d}",
                                       source_manifest, cfg, candidate, manifest)
                break
        if imported is None:
            imported = {"batch_index": 0, "candidate": candidate, "status": "fresh_no_authenticated_artifact"}
        records.append(imported)
    result = {
        "schema_version": 2, "task_id": TASK_ID, "plan_version": PLAN_VERSION,
        "amendment_id": AMENDMENT_ID, "keep_percent": percent,
        "candidate_ids": list(CANDIDATES), "batch_indices": [0], "n_images": N_IMAGES,
        "identity": manifest["identity"],
        "source_roots": [str(root) for root, _ in source_roots],
        "records": records, "v1_selection_reused": False, "v2_selection_reused": False,
        "old_candidates_8_19_used": False, "batches01_04_used": False,
        "created_at_utc": utc_now(),
        "method_change": "same80 replaced by original same16 batch00; candidates remain exactly 0..7",
    }
    write_json(path, result)
    return result


def run_candidate(*, cfg: dict[str, Any], batch: int, candidate: int, packet: bytes,
                  folder: Path, identity: dict[str, Any], stop: int,
                  device: torch.device, notify_stage) -> dict[str, Any]:
    if batch != 0 or candidate not in CANDIDATES:
        raise ValueError("SAME16 permits only batch00 and candidate IDs0..7")
    with _adapter_scope():
        return r8_adapter.run_candidate(cfg=cfg, batch=batch, candidate=candidate, packet=packet,
                                        folder=folder, identity=identity, stop=stop,
                                        device=device, notify_stage=notify_stage)


def run_batch(cfg: dict[str, Any], manifest: dict[str, Any], batch: int,
              output: Path, device: torch.device) -> None:
    if batch != 0:
        raise ValueError("SAME16 must not run batches01..04")
    folder = output / "batch00"
    folder.mkdir(parents=True, exist_ok=True)
    identity = manifest["identity"]
    if checked_marker(folder / "summary.json", identity) is not None:
        return
    started = time.perf_counter()
    packet_ref = identity["packets"][0]
    packet_path = resolve(packet_ref["relative_path"])
    if file_hash(packet_path) != packet_ref["sha256"]:
        raise RuntimeError("Received batch00 packet changed")
    packet = packet_path.read_bytes()

    def notify_stage(stage, candidate, stop):
        task_status(output, cfg, state="running", batch_index=0, candidate=candidate,
                    candidates_total=CANDIDATE_COUNT, stage=stage, fia_stop=stop, fia_horizon=25000)

    screens = [run_candidate(cfg=cfg, batch=0, candidate=candidate, packet=packet,
                             folder=folder / f"candidate{candidate:02d}", identity=identity,
                             stop=5000, device=device, notify_stage=notify_stage)
               for candidate in CANDIDATES]
    ranked = rank_candidates(screens, list(CANDIDATES))
    advanced = [row["candidate"] for row in ranked[:3]]
    selection = {
        "ranked_candidates": [row["candidate"] for row in ranked], "advanced": advanced,
        "scores": [{"candidate": row["candidate"], "relative_l2": row["relative_l2"]} for row in ranked],
        "rule": RANDOMIZATION["ranking"], "truth_used": False, "metric_selection": False,
        "keep_percent": cfg["keep_percent"], "candidate_ids": list(CANDIDATES),
        "batch_indices": [0], "n_images": N_IMAGES, "plan_version": PLAN_VERSION,
    }
    saved_selection = checked_marker(folder / "screen_selection.json", identity)
    if saved_selection is not None and any(saved_selection.get(key) != value for key, value in selection.items()):
        raise RuntimeError("Frozen SAME16 screen selection changed")
    if saved_selection is None:
        save_marker(folder / "screen_selection.json", identity,
                    [f"candidate{i:02d}/screen.json" for i in CANDIDATES], **selection)
    print(f"SAME16 SCREEN COMPLETE keep={cfg['keep_percent']} batch=0 advanced={advanced}", flush=True)
    finals = [run_candidate(cfg=cfg, batch=0, candidate=candidate, packet=packet,
                            folder=folder / f"candidate{candidate:02d}", identity=identity,
                            stop=25000, device=device, notify_stage=notify_stage)
              for candidate in advanced]
    final_ranked = rank_candidates(finals, advanced)
    selected = int(final_ranked[0]["candidate"])
    final_selection = {
        "selected_candidate": selected,
        "ranked_candidates": [row["candidate"] for row in final_ranked],
        "scores": [{"candidate": row["candidate"], "relative_l2": row["relative_l2"]} for row in final_ranked],
        "truth_used": False, "metric_selection": False, "keep_percent": cfg["keep_percent"],
        "candidate_ids": list(CANDIDATES), "batch_indices": [0], "n_images": N_IMAGES,
        "plan_version": PLAN_VERSION,
    }
    saved_final = checked_marker(folder / "final_selection.json", identity)
    if saved_final is not None and any(saved_final.get(key) != value for key, value in final_selection.items()):
        raise RuntimeError("Frozen SAME16 final selection changed")
    if saved_final is None:
        save_marker(folder / "final_selection.json", identity,
                    [f"candidate{i:02d}/final.json" for i in advanced], **final_selection)
    del packet
    notify_stage("evaluation_after_final_selection", selected, 25000)
    verify_inputs(manifest)
    inputs = torch.load(resolve(cfg["source_output"]) / "batch00/inputs.pt",
                        map_location="cpu", weights_only=False, mmap=True)
    selected_folder = folder / f"candidate{selected:02d}"
    sources = torch.load(selected_folder / "sources.pt", map_location="cpu", weights_only=True)
    source_recs = torch.load(selected_folder / "final_rec.pt", map_location="cpu", weights_only=True)
    recs, correspondence = evaluate_after_fia(sources, source_recs, inputs["Z_evaluation_only"])
    metrics = evaluate_images(x_true=inputs["x"], rec=recs.to(device), device=device)
    rows = [{**inputs["metadata"][i], **correspondence[i], "run_tag": cfg["run_tag"],
             "n": 16, "T": 10, "selected_candidate": selected, "keep_percent": cfg["keep_percent"],
             "source_codec": cfg["source_codec"], "ls_A": False, "oracle_fill": False,
             **{key: float(getattr(metrics, key)[i]) for key in ("lpips", "psnr", "ssim")}}
            for i in range(N_IMAGES)]
    legacy._save_checkpoint(folder / "source_order_rec.pt", source_recs)
    legacy._save_checkpoint(folder / "rec.pt", recs)
    write_csv(folder / "per_image_metrics.csv", rows)
    cpa_time = sum(read_json(folder / f"candidate{i:02d}/cpa_result.json")["wall_time_sec"] for i in CANDIDATES)
    final_by_id = {row["candidate"]: row for row in finals}
    fia_time = sum(final_by_id.get(row["candidate"], row)["fia_optimizer_wall_time_sec"] for row in screens)
    ranking_time = sum(row["ranking_wall_time_sec"] for row in screens + finals)
    done = save_marker(folder / "summary.json", identity,
                       ["screen_selection.json", "final_selection.json", "source_order_rec.pt",
                        "rec.pt", "per_image_metrics.csv"],
                       task_id=TASK_ID, plan_version=PLAN_VERSION, amendment_id=AMENDMENT_ID,
                       batch_index=0, batch_indices=[0], n_images=N_IMAGES, n=16, T=10,
                       keep_percent=cfg["keep_percent"], source_codec=cfg["source_codec"],
                       packet_bytes=packet_ref["bytes"], retained_elements=packet_ref["retained_elements"],
                       candidate_ids=list(CANDIDATES), selected_candidate=selected,
                       advanced_candidates=advanced, oracle_fills=0, ls_A=False,
                       source_cos_per_source=[row["source_abs_cosine"] for row in rows],
                       embedding_cos09=sum(row["measured_recovered_cos09"] for row in rows),
                       **{f"{key}_mean": float(np.mean([row[key] for row in rows]))
                          for key in ("lpips", "ssim", "psnr")},
                       runtime_sec={"cpa": cpa_time, "fia_optimizer": fia_time,
                                    "ranking": ranking_time,
                                    "current_invocation": time.perf_counter() - started},
                       budget={"cpa_runs": CANDIDATE_COUNT, "cpa_iterations_total": CANDIDATE_COUNT * 25000,
                               "fia_screen_iterations_total": CANDIDATE_COUNT * 5000,
                               "fia_final_continuation_iterations_total": 3 * 20000,
                               "fia_iterations_total": CANDIDATE_COUNT * 5000 + 3 * 20000,
                               "ranking_replays": CANDIDATE_COUNT + 3},
                       selection_rule=RANDOMIZATION["ranking"], truth_used=False,
                       completed_at_utc=utc_now())
    print(f"SAME16 COMPLETE keep={cfg['keep_percent']} batch=0 selected={selected} LPIPS={done['lpips_mean']:.6f}", flush=True)
    del inputs, metrics, recs, source_recs, sources
    legacy.free_memory()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "migrate", "run"), default="prepare")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = read_json(config_path)
    validate_config(cfg)
    output = resolve(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    with execution_lock(output):
        try:
            manifest = prepare(config_path, cfg, output)
            if args.phase == "prepare":
                task_status(output, cfg, state="prepared", n_images=N_IMAGES, packets=1,
                            batch_indices=[0], attack_started=False, candidate_ids=list(CANDIDATES))
                return
            migration = migrate_fraction(cfg, manifest, output)
            if args.phase == "migrate":
                task_status(output, cfg, state="migrated", n_images=N_IMAGES, packets=1,
                            batch_indices=[0], attack_started=False,
                            migration_sha256=file_hash(output / "migration_manifest.json"),
                            imported_records=len(migration["records"]))
                return
            device = torch.device(cfg["device"])
            if device.type != "cuda" or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
                raise RuntimeError("Expected the established sm_120 CUDA device")
            if not (output / "run_clock.json").exists():
                write_json(output / "run_clock.json", {
                    "task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
                    "keep_percent": cfg["keep_percent"], "candidate_ids": list(CANDIDATES),
                    "batch_indices": [0], "n_images": N_IMAGES, "first_started_at_utc": utc_now(),
                    "note": "SAME16 original batch00 only; elapsed wall time includes authenticated resumes",
                })
            task_status(output, cfg, state="running", n_images=N_IMAGES, packets=1,
                        batch_indices=[0], attack_started=True,
                        migration_sha256=file_hash(output / "migration_manifest.json"))
            for batch in cfg["batch_indices"]:
                run_batch(cfg, manifest, int(batch), output, device)
            from .report_paper_sparse_same16_r8 import summarize
            result = summarize(cfg, manifest, output)
            task_status(output, cfg, state="complete", n_images=N_IMAGES, packets=1,
                        batch_indices=[0], metrics=result["metrics"],
                        random_summary_sha256=file_hash(output / "summary.json"))
        except BaseException as exc:
            task_status(output, cfg, state="failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
