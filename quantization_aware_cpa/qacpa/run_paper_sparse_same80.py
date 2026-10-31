"""Paper same80 random-fill sweep for retained Top-k 50%, 30% and 20%.

This runner is deliberately isolated from the completed Top-k10 driver.  It
reuses the packet-only random fill, upstream CPA, staged FIA and evaluation
adapters while registering the exact historical inputs and the three newly
authorized fractions in a task-local manifest.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .run_m8_fedavg_naive import (
    PROJECT, WORKSPACE, read_json, write_json, read_csv, write_csv, file_hash,
    resolve, utc_now, status as _status, execution_lock, evaluate_after_fia,
)
from .random_fill import candidate_seed, random_fill_packet, consistency_score, rank_candidates
from .paper_sparse_codec import SOURCE_CODECS, expected_packet_bytes, validate_received_packet
from .staged_fia import run_staged_fia
from .upstream_naive import run_upstream_cpa, _save_checkpoint
from .fia import evaluate_images


TASK_ID = "CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921"
PLAN_VERSION = 1
PERCENTS = (50, 30, 20)
N_BATCHES = 5
N_IMAGES = 80
N_PER_BATCH = 16
T_STEPS = 10
FC1_SHAPE = (4096, 25088)

RANDOMIZATION = {
    "sampler": "uniform_tau_pcg64_v1", "tau": "min(abs(transmitted FP32 values))",
    "candidates": 20, "screen_iterations": 5000, "advance_top_k": 3,
    "final_iterations": 25000, "cosine_horizon": 25000,
    "fill_seed": "100000 + 100*batch_index + candidate_index; same plan across fractions",
    "cpa_seed": "baseline seed + batch_index, identical for all candidates",
    "fia_seed": "baseline FIA seed42, identical for all candidates",
    "ranking": "relative L2 of recompressed reconstructed-data SGD update to received packet",
    "ranking_labels": "public round0 model argmax; no true labels",
    "tie_break": "candidate_index ascending",
    "selection_unit": "whole16-image candidate per victim batch",
    "final_selection": "lowest same observable relative L2 among three final candidates",
    "true_metrics_for_selection": False,
}


def _communication(folder: Path, batch: int, keep_percent: int) -> dict[str, Any]:
    """Validate a matched baseline communication record for any three fractions."""
    recorded = read_json(folder / "communication.json")
    packet_path = folder / "observation.bin"
    packet_bytes = packet_path.stat().st_size
    kept = FC1_SHAPE[0] * FC1_SHAPE[1] * keep_percent // 100
    expected = {
        "codec": SOURCE_CODECS[keep_percent], "value_bits": 32,
        "keep_fraction": keep_percent / 100, "shape": list(FC1_SHAPE),
        "n_values": FC1_SHAPE[0] * FC1_SHAPE[1], "kept_elements": kept,
        "fp32_payload_bytes": 4 * FC1_SHAPE[0] * FC1_SHAPE[1],
        "fp32_packet_bytes": 32 + 4 * FC1_SHAPE[0] * FC1_SHAPE[1],
        "value_bytes": 4 * kept, "bitmap_bytes": (FC1_SHAPE[0] * FC1_SHAPE[1] + 7) // 8,
        "header_bytes": 32, "packet_bytes": expected_packet_bytes(keep_percent),
        "scope": "attacked classifier.0.weight packet only", "quantization": False,
        "error_feedback": False, "source_reconstruction_only_from_packet": True,
    }
    if any(recorded.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Matched packet communication declaration changed at batch {batch}")
    if packet_bytes != expected["packet_bytes"] or recorded.get("sha256") != file_hash(packet_path):
        raise RuntimeError(f"Matched packet bytes or hash changed at batch {batch}")
    ratio = expected["fp32_packet_bytes"] / expected["packet_bytes"]
    if not np.isclose(recorded.get("compression_ratio"), ratio) or not np.isclose(
            recorded.get("saved_percent"), 100 * (1 - 1 / ratio)):
        raise RuntimeError(f"Matched packet savings changed at batch {batch}")
    return {"batch_index": batch, **recorded}


def task_status(output: Path, cfg: dict[str, Any], **kwargs: Any) -> None:
    _status(output, cfg, task_id=TASK_ID, plan_version=PLAN_VERSION,
            keep_percent=cfg.get("keep_percent"), **kwargs)


def _baseline_config(percent: int) -> dict[str, Any]:
    return read_json(PROJECT / f"configs/cpa_fia_topk{percent}_same80_B16.json")


def _validate_common_baseline(baseline: dict[str, Any], percent: int) -> None:
    if baseline.get("run_tag") != f"cpa_fia_topk{percent}_same80_B16":
        raise ValueError("Unexpected matched naive baseline run tag")
    if baseline.get("output_dir") != f"results/cpa_fia_topk{percent}_same80_B16":
        raise ValueError("Unexpected matched naive baseline output")
    if (baseline.get("n"), baseline.get("T"), baseline.get("n_images"),
            baseline.get("batch_indices")) != (16, 10, 80, list(range(5))):
        raise ValueError("Paper sparse runs require the original five B16 batches")
    if baseline.get("source_output") != "results/m6_uia_fp32_B16_fia":
        raise ValueError("The cached same80 input source is fixed")
    local = baseline.get("local_training", {})
    expected_local = {
        "lr": 0.001, "steps": 10, "optimizer": "sgd", "momentum": 0,
        "weight_decay": 0, "classifier_dropout": False,
        "model_reset_each_update": True, "independent_client_updates": 5,
        "provenance": "unchanged cached M6 updates; no retraining",
    }
    if local != expected_local:
        raise ValueError("Matched baseline local-training protocol changed")
    model = baseline.get("model", {})
    if (model.get("name"), model.get("weights"), model.get("attacked_checkpoint_round"),
            model.get("conv_frozen"), model.get("updated_layers"),
            model.get("attack_layer")) != (
                "vgg16", "IMAGENET1K_V1", 0, True, "all classifier parameters",
                "classifier.0.weight"):
        raise ValueError("Matched baseline model protocol changed")
    stage1 = baseline.get("stage1", {})
    if (stage1.get("cpa_ica_max_iter"), stage1.get("cpa_ica_lr"),
            stage1.get("cpa_ica_optimizer"), stage1.get("cpa_ica_scheduler"),
            stage1.get("cpa_ica_decor"), stage1.get("cpa_ica_T"),
            stage1.get("cpa_ica_ne"), stage1.get("cpa_ica_nv"),
            stage1.get("cpa_ica_l1"), stage1.get("cpa_ica_tv"),
            stage1.get("cpa_ica_log_every")) != (
                25000, 0.001, "adam", "none", 5.3, 7.7, 1, 0.13, 5, 0.1, 5000):
        raise ValueError("Upstream CPA protocol changed")
    fia = baseline.get("fia", {})
    if (fia.get("n_iter"), fia.get("log_every"), fia.get("lr"), fia.get("tv"),
            fia.get("fi"), fia.get("optimizer"), fia.get("scheduler"),
            fia.get("seed")) != (25000, 5000, 0.1, 0.1, 1, "adam", "cosine", 42):
        raise ValueError("Staged FIA protocol changed")
    for key in ("new_training", "new_update_collection", "ls_A", "debias",
                "mp_truncation", "mp_diagnostic", "stage2", "refine_A",
                "oracle_fills", "fp32_attack", "raw_row_attack"):
        if baseline.get(key) is not False:
            raise ValueError(f"Matched baseline flag {key} must be false")
    sparse = baseline.get("sparsification", {})
    if (sparse.get("keep_fraction"), sparse.get("selection"), sparse.get("tie_break"),
            sparse.get("value_dtype"), sparse.get("positions"), sparse.get("discarded_fill"),
            sparse.get("quantization"), sparse.get("rescaling"), sparse.get("error_feedback")) != (
                percent / 100, "global magnitude within classifier.0.weight",
                "flat index ascending", "float32", "row-major bitmap, little bit order",
                0.0, False, False, False):
        raise ValueError("Matched Top-k protocol changed")
    if baseline.get("stage1", {}).get("cpa_ica_max_iter") != 25000 or baseline.get("fia", {}).get("n_iter") != 25000:
        raise ValueError("The registered attack budget changed")


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    percent = cfg.get("keep_percent")
    if type(percent) is not int or percent not in PERCENTS:
        raise ValueError("Only retained fractions 50, 30 and 20 are authorized")
    baseline = _baseline_config(percent)
    _validate_common_baseline(baseline, percent)
    changed = {"run_tag", "output_dir", "observation", "execution_scope",
               "keep_percent", "baseline_output", "randomization", "source_codec"}
    expected_keys = set(baseline) | {"keep_percent", "baseline_output", "randomization", "source_codec"}
    if set(cfg) != expected_keys:
        raise ValueError("Unexpected paper sparse configuration fields")
    for key, value in baseline.items():
        if key not in changed and cfg.get(key) != value:
            raise ValueError(f"Matched baseline condition changed: {key}")
    tag = f"cpa_fia_random_topk{percent}_same80_R20_S5k_K3_F25k"
    if cfg["run_tag"] != tag or cfg["output_dir"] != f"results/{tag}":
        raise ValueError("Paper sparse results must use a task-local output")
    if cfg["baseline_output"] != baseline["output_dir"]:
        raise ValueError("The random-fill comparison must use its matched naive output")
    if cfg["source_codec"] != SOURCE_CODECS[percent] or cfg["codec"] != baseline["codec"]:
        raise ValueError("The historical source codec label changed")
    expected_bytes = expected_packet_bytes(percent)
    if cfg.get("expected_packet_bytes") != expected_bytes:
        raise ValueError("The configured packet byte count is wrong")
    if cfg["sparsification"]["keep_fraction"] != percent / 100:
        raise ValueError("Only the retained fraction may vary")
    if cfg["randomization"] != RANDOMIZATION:
        raise ValueError("Random-fill selection protocol differs from the handoff")
    if cfg.get("n") != 16 or cfg.get("T") != 10 or cfg.get("batch_indices") != list(range(5)):
        raise ValueError("Only same80 B16/T10 grouping is authorized")
    return baseline


def dependency_hashes(baselines: dict[int, dict[str, Any]]) -> dict[str, str]:
    """Verify all frozen baseline dependencies before recording new code."""
    hashes: dict[str, str] = {}
    for percent, summary in baselines.items():
        for rel, expected in summary["identity"]["code_sha256"].items():
            path = WORKSPACE / rel
            actual = file_hash(path)
            if actual != expected:
                raise RuntimeError(f"Frozen baseline dependency changed: {rel}")
            if rel in hashes and hashes[rel] != expected:
                raise RuntimeError(f"Baseline dependency hashes disagree: {rel}")
            hashes[rel] = expected
    new_paths = [
        PROJECT / "qacpa/paper_sparse_codec.py",
        PROJECT / "qacpa/run_paper_sparse_same80.py",
        PROJECT / "qacpa/report_paper_sparse_same80.py",
        PROJECT / "scripts/run_paper_sparse_same80.py",
        PROJECT / "docs/RANDOM_FILL_SAME80.md",
    ]
    for path in new_paths:
        hashes[str(path.relative_to(WORKSPACE))] = file_hash(path)
    return hashes


def _identity_rows(rows: list[dict[str, Any]]) -> list[tuple[str, ...]]:
    fields = ("global_index", "batch_index", "source_index", "dataset_index", "label", "image_path")
    return [tuple(str(row[key]) for key in fields) for row in rows]


def _verify_baseline_complete(percent: int) -> tuple[dict[str, Any], dict[str, Any]]:
    output = resolve(f"results/cpa_fia_topk{percent}_same80_B16")
    summary_path = output / "summary.json"
    manifest_path = output / "input_manifest.json"
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    if summary.get("status") != "complete" or summary.get("n_images") != 80:
        raise RuntimeError(f"Matched naive Top-k{percent} baseline is not complete")
    if summary.get("config") != _baseline_config(percent):
        raise RuntimeError(f"Matched naive Top-k{percent} configuration changed")
    if manifest.get("metadata") is None or len(manifest["metadata"]) != 80:
        raise RuntimeError(f"Matched naive Top-k{percent} manifest lacks all80 identities")
    if [int(row["global_index"]) for row in manifest["metadata"]] != list(range(80)):
        raise RuntimeError("The matched baseline global order changed")
    for name in ("per_image_metrics.csv", "all80_rec.pt"):
        digest = summary.get("output_hashes", {}).get(name)
        if not digest or file_hash(output / name) != digest:
            raise RuntimeError(f"Matched baseline artifact changed: {output / name}")
    rows = read_csv(output / "per_image_metrics.csv")
    if _identity_rows(rows) != _identity_rows(manifest["metadata"]):
        raise RuntimeError(f"Matched baseline Top-k{percent} identity table changed")
    return summary, manifest


def verify_inputs(manifest: dict[str, Any], batch: int | None = None) -> None:
    for rel, expected in manifest["inputs_sha256"].items():
        path = resolve(rel)
        if batch is None or path.parent.name == f"batch{batch:02d}":
            if file_hash(path) != expected:
                raise RuntimeError(f"Frozen source changed: {rel}")


def _verify_source_manifest(manifest: dict[str, Any]) -> None:
    if len(manifest.get("metadata", [])) != 80:
        raise RuntimeError("The source manifest must contain all80 identities")
    if [int(row["global_index"]) for row in manifest["metadata"]] != list(range(80)):
        raise RuntimeError("The source manifest global order changed")
    verify_inputs(manifest)
    weights = Path(manifest["weights_path"])
    if file_hash(weights) != manifest["weights_sha256"]:
        raise RuntimeError("Public round0 checkpoint changed")
    for batch in range(N_BATCHES):
        source = resolve("results/m6_uia_fp32_B16_fia") / f"batch{batch:02d}/inputs.pt"
        inputs = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
        if tuple(inputs["x"].shape) != (16, 3, 224, 224) or tuple(inputs["Z_evaluation_only"].shape) != (16, 25088):
            raise RuntimeError(f"Invalid cached same80 tensors in batch {batch}")
        if _identity_rows(inputs["metadata"]) != _identity_rows(manifest["metadata"][batch * 16:(batch + 1) * 16]):
            raise RuntimeError(f"Cached same80 identity/order changed in batch {batch}")
        del inputs


def prepare(config_path: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    baseline = validate_config(cfg)
    percent = cfg["keep_percent"]
    all_baselines: dict[int, dict[str, Any]] = {}
    all_manifests: dict[int, dict[str, Any]] = {}
    for fraction in PERCENTS:
        summary, manifest = _verify_baseline_complete(fraction)
        all_baselines[fraction] = summary
        all_manifests[fraction] = manifest
    metadata = all_manifests[50]["metadata"]
    for fraction in PERCENTS:
        if _identity_rows(all_manifests[fraction]["metadata"]) != _identity_rows(metadata):
            raise RuntimeError("Original same80 identities differ across sparse conditions")
    source_manifest = all_manifests[percent]
    _verify_source_manifest(source_manifest)
    hashes = dependency_hashes(all_baselines)
    source = resolve(cfg["baseline_output"])
    packets, communications, audits = [], [], []
    for batch in range(N_BATCHES):
        folder = source / f"batch{batch:02d}"
        packet_path = folder / "observation.bin"
        packet = packet_path.read_bytes()
        batch_summary = read_json(folder / "summary.json")
        comm = _communication(folder, batch, percent)
        expected_sha = batch_summary["output_hashes"]["observation.bin"]
        actual_sha = file_hash(packet_path)
        if actual_sha != expected_sha or actual_sha != comm["sha256"]:
            raise RuntimeError(f"Saved packet hash differs from matched baseline at batch {batch}")
        if comm["packet_bytes"] != expected_packet_bytes(percent):
            raise RuntimeError(f"Unexpected packet byte count at batch {batch}")
        audit = validate_received_packet(packet, percent, cfg["source_codec"])
        audit.update({"batch_index": batch, "path": str(packet_path), "sha256": actual_sha,
                      "baseline_summary_sha256": file_hash(folder / "summary.json"),
                      "baseline_communication_sha256": file_hash(folder / "communication.json")})
        packets.append({"batch_index": batch, "path": str(packet_path),
                        "relative_path": str(packet_path.relative_to(PROJECT)),
                        "sha256": actual_sha, "bytes": len(packet),
                        "source_codec": cfg["source_codec"],
                        "retained_elements": audit["kept_elements"]})
        communications.append(comm)
        audits.append(audit)
    identity = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "keep_percent": percent,
        "config_sha256": file_hash(config_path), "code_sha256": hashes,
        "baseline_summary_sha256": file_hash(source / "summary.json"),
        "baseline_manifest_sha256": file_hash(source / "input_manifest.json"),
        "source_inputs_sha256": source_manifest["inputs_sha256"],
        "weights_sha256": source_manifest["weights_sha256"],
        "packets": packets,
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "input_manifest.json"
    packet_audit_path = output / "packet_audit.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("identity") != identity:
            raise RuntimeError("Incompatible resume: frozen input/code/config changed")
        if _identity_rows(manifest.get("metadata", [])) != _identity_rows(metadata):
            raise RuntimeError("Original data identity changed")
        if file_hash(packet_audit_path) != manifest.get("packet_audit_sha256"):
            raise RuntimeError("Frozen packet audit changed")
        verify_inputs(manifest)
        return manifest
    packet_audit = {"task_id": TASK_ID, "plan_version": PLAN_VERSION,
                    "keep_percent": percent, "source_codec": cfg["source_codec"],
                    "batches": audits}
    write_json(packet_audit_path, packet_audit)
    manifest = {
        "schema_version": 2, "task_id": TASK_ID, "plan_version": PLAN_VERSION,
        "run_tag": cfg["run_tag"], "identity": identity,
        "git_sha": source_manifest.get("git_sha", ""),
        "inputs_sha256": source_manifest["inputs_sha256"],
        "metadata": metadata, "n_images": N_IMAGES, "n": N_PER_BATCH, "T": T_STEPS,
        "created_at_utc": utc_now(), "new_training": False,
        "new_update_collection": False, "source_codec": cfg["source_codec"],
        "randomization": RANDOMIZATION, "communication_per_batch": communications,
        "packet_audits": audits, "packet_audit_sha256": file_hash(packet_audit_path),
        "baseline_output": cfg["baseline_output"],
        "baseline_summary_sha256": file_hash(source / "summary.json"),
        "baseline_manifest_sha256": file_hash(source / "input_manifest.json"),
    }
    write_csv(output / "image_manifest.csv", metadata)
    write_json(manifest_path, manifest)
    write_json(config_path.with_suffix(".lock.json"), {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION,
        "config_sha256": identity["config_sha256"],
        "input_manifest_sha256": file_hash(manifest_path),
        "packet_audit_sha256": file_hash(packet_audit_path),
        "frozen_at_utc": utc_now(),
    })
    return manifest


def checked_marker(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    saved = read_json(path)
    if saved.get("identity") != identity:
        raise RuntimeError(f"Artifact identity differs: {path}")
    for name, expected in saved.get("output_hashes", {}).items():
        if file_hash(path.parent / name) != expected:
            raise RuntimeError(f"Artifact hash differs: {path.parent / name}")
    return saved


def save_marker(path: Path, identity: dict[str, Any], names: list[str], **values) -> dict[str, Any]:
    result = {"identity": identity, **values,
              "output_hashes": {name: file_hash(path.parent / name) for name in names}}
    write_json(path, result)
    return result


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_candidate(*, cfg: dict[str, Any], batch: int, candidate: int, packet: bytes,
                  folder: Path, identity: dict[str, Any], stop: int,
                  device: torch.device, notify_stage) -> dict[str, Any]:
    """Attack boundary: only a received packet enters this function."""
    folder.mkdir(parents=True, exist_ok=True)
    phase = "screen" if stop == RANDOMIZATION["screen_iterations"] else "final"
    previous = checked_marker(folder / f"{phase}.json", identity)
    if previous is not None:
        return previous
    cpa_done = checked_marker(folder / "cpa_result.json", identity)
    if cpa_done is None:
        notify_stage("random_fill_CPA", candidate, stop)
        matrix, audit = random_fill_packet(packet, candidate_seed(batch, candidate))
        if tuple(matrix.shape) != FC1_SHAPE or matrix.dtype != torch.float32:
            raise RuntimeError("Expected the same FC1 observation shape")
        audit.update({"keep_percent": cfg["keep_percent"],
                      "source_codec": cfg["source_codec"],
                      "generic_recompression_codec": "topk_global_bitmap_fp32_v1"})
        write_json(folder / "fill_audit.json", audit)
        cpa = run_upstream_cpa(Y_hat=matrix, n_components=16, device=device,
                               config=cfg["stage1"], output_dir=folder / "cpa",
                               seed=cfg["seed"] + batch)
        sources = cpa.pop("S_hat")
        if tuple(sources.shape) != (16, 25088) or not torch.isfinite(sources).all():
            raise RuntimeError("CPA must retain all16 finite sources")
        _save_checkpoint(folder / "sources.pt", sources)
        write_json(folder / "cpa_log.json", cpa)
        cpa_done = save_marker(folder / "cpa_result.json", identity,
                               ["sources.pt", "fill_audit.json", "cpa_log.json"],
                               candidate=candidate, batch_index=batch,
                               wall_time_sec=cpa["wall_time_sec"])
        del matrix, cpa, sources
        free_memory()
    sources = torch.load(folder / "sources.pt", map_location="cpu", weights_only=True)
    notify_stage(f"FIA_{phase}", candidate, stop)
    fia = run_staged_fia(target_z=sources.abs().contiguous(), device=device,
                         config=cfg["fia"], output_dir=folder / "fia",
                         seed=cfg["fia"]["seed"], stop=stop)
    rec = fia.pop("rec")
    _save_checkpoint(folder / f"{phase}_rec.pt", rec)
    write_json(folder / f"{phase}_fia_log.json", fia)
    free_memory()
    notify_stage(f"observable_ranking_{phase}", candidate, stop)
    started = time.perf_counter()
    score = consistency_score(rec=rec, packet=packet,
                              keep_percent=cfg["keep_percent"], device=device)
    result = save_marker(folder / f"{phase}.json", identity,
                         [f"{phase}_rec.pt", f"{phase}_fia_log.json", "sources.pt", "cpa_result.json"],
                         candidate=candidate, batch_index=batch,
                         completed_iterations=stop, **score,
                         ranking_wall_time_sec=time.perf_counter() - started,
                         fia_optimizer_wall_time_sec=fia["optimizer_wall_time_sec"],
                         completed_at_utc=utc_now())
    del rec, sources, fia
    free_memory()
    return result


def run_batch(cfg: dict[str, Any], manifest: dict[str, Any], batch: int,
              output: Path, device: torch.device) -> None:
    folder = output / f"batch{batch:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    identity = manifest["identity"]
    if checked_marker(folder / "summary.json", identity) is not None:
        return
    started = time.perf_counter()
    packet_ref = identity["packets"][batch]
    packet_path = Path(packet_ref["path"])
    if file_hash(packet_path) != packet_ref["sha256"]:
        raise RuntimeError("Received packet changed")
    packet = packet_path.read_bytes()

    def notify_stage(stage, candidate, stop):
        task_status(output, cfg, state="running", batch_index=batch, candidate=candidate,
                    candidates_total=20, stage=stage, fia_stop=stop, fia_horizon=25000)

    screens = [run_candidate(cfg=cfg, batch=batch, candidate=candidate, packet=packet,
                             folder=folder / f"candidate{candidate:02d}", identity=identity,
                             stop=5000, device=device, notify_stage=notify_stage)
               for candidate in range(20)]
    ranked = rank_candidates(screens, list(range(20)))
    advanced = [row["candidate"] for row in ranked[:3]]
    selection = {
        "ranked_candidates": [row["candidate"] for row in ranked],
        "advanced": advanced,
        "scores": [{"candidate": row["candidate"], "relative_l2": row["relative_l2"]} for row in ranked],
        "rule": RANDOMIZATION["ranking"], "truth_used": False,
        "metric_selection": False, "keep_percent": cfg["keep_percent"],
    }
    saved_selection = checked_marker(folder / "screen_selection.json", identity)
    if saved_selection is not None and any(saved_selection.get(key) != value for key, value in selection.items()):
        raise RuntimeError("Frozen screen selection changed")
    if saved_selection is None:
        save_marker(folder / "screen_selection.json", identity,
                    [f"candidate{i:02d}/screen.json" for i in range(20)], **selection)
    print(f"SCREEN COMPLETE keep={cfg['keep_percent']} batch={batch} advanced={advanced}", flush=True)
    finals = [run_candidate(cfg=cfg, batch=batch, candidate=candidate, packet=packet,
                            folder=folder / f"candidate{candidate:02d}", identity=identity,
                            stop=25000, device=device, notify_stage=notify_stage)
              for candidate in advanced]
    final_ranked = rank_candidates(finals, advanced)
    selected = int(final_ranked[0]["candidate"])
    final_selection = {
        "selected_candidate": selected,
        "ranked_candidates": [row["candidate"] for row in final_ranked],
        "scores": [{"candidate": row["candidate"], "relative_l2": row["relative_l2"]} for row in final_ranked],
        "truth_used": False, "metric_selection": False,
        "keep_percent": cfg["keep_percent"],
    }
    saved_final = checked_marker(folder / "final_selection.json", identity)
    if saved_final is not None and any(saved_final.get(key) != value for key, value in final_selection.items()):
        raise RuntimeError("Frozen final selection changed")
    if saved_final is None:
        save_marker(folder / "final_selection.json", identity,
                    [f"candidate{i:02d}/final.json" for i in advanced], **final_selection)
    del packet
    # Private evaluation truth is loaded only after screen and final selection markers exist.
    notify_stage("evaluation_after_final_selection", selected, 25000)
    verify_inputs(manifest, batch)
    inputs = torch.load(resolve(cfg["source_output"]) / f"batch{batch:02d}/inputs.pt",
                        map_location="cpu", weights_only=False, mmap=True)
    selected_folder = folder / f"candidate{selected:02d}"
    sources = torch.load(selected_folder / "sources.pt", map_location="cpu", weights_only=True)
    source_recs = torch.load(selected_folder / "final_rec.pt", map_location="cpu", weights_only=True)
    recs, correspondence = evaluate_after_fia(sources, source_recs, inputs["Z_evaluation_only"])
    metrics = evaluate_images(x_true=inputs["x"], rec=recs.to(device), device=device)
    rows = [{**inputs["metadata"][i], **correspondence[i],
             "run_tag": cfg["run_tag"], "n": 16, "T": 10,
             "selected_candidate": selected, "keep_percent": cfg["keep_percent"],
             "source_codec": cfg["source_codec"], "ls_A": False, "oracle_fill": False,
             **{key: float(getattr(metrics, key)[i]) for key in ("lpips", "psnr", "ssim")}}
            for i in range(16)]
    _save_checkpoint(folder / "source_order_rec.pt", source_recs)
    _save_checkpoint(folder / "rec.pt", recs)
    write_csv(folder / "per_image_metrics.csv", rows)
    cpa_time = sum(read_json(folder / f"candidate{i:02d}/cpa_result.json")["wall_time_sec"] for i in range(20))
    final_by_id = {row["candidate"]: row for row in finals}
    fia_time = sum(final_by_id.get(row["candidate"], row)["fia_optimizer_wall_time_sec"] for row in screens)
    ranking_time = sum(row["ranking_wall_time_sec"] for row in screens + finals)
    done = save_marker(folder / "summary.json", identity,
                       ["screen_selection.json", "final_selection.json", "source_order_rec.pt",
                        "rec.pt", "per_image_metrics.csv"],
                       task_id=TASK_ID, plan_version=PLAN_VERSION, batch_index=batch,
                       n_images=16, n=16, T=10, keep_percent=cfg["keep_percent"],
                       source_codec=cfg["source_codec"], packet_bytes=packet_ref["bytes"],
                       retained_elements=packet_ref["retained_elements"],
                       selected_candidate=selected, advanced_candidates=advanced,
                       oracle_fills=0, ls_A=False,
                       source_cos_per_source=[row["source_abs_cosine"] for row in rows],
                       embedding_cos09=sum(row["measured_recovered_cos09"] for row in rows),
                       **{f"{key}_mean": float(np.mean([row[key] for row in rows]))
                          for key in ("lpips", "ssim", "psnr")},
                       runtime_sec={"cpa": cpa_time, "fia_optimizer": fia_time,
                                    "ranking": ranking_time,
                                    "current_invocation": time.perf_counter() - started},
                       budget={"cpa_runs": 20, "cpa_iterations_total": 500000,
                               "fia_screen_iterations_total": 100000,
                               "fia_final_continuation_iterations_total": 60000,
                               "fia_iterations_total": 160000, "ranking_replays": 23},
                       selection_rule=RANDOMIZATION["ranking"], truth_used=False,
                       completed_at_utc=utc_now())
    print(f"COMPLETE keep={cfg['keep_percent']} batch={batch} selected={selected} LPIPS={done['lpips_mean']:.6f}", flush=True)
    del inputs, metrics, recs, source_recs, sources
    free_memory()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "run", "report"), default="prepare")
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
                task_status(output, cfg, state="prepared", n_images=80, packets=5,
                            attack_started=False)
                return
            if args.phase == "run":
                device = torch.device(cfg["device"])
                if device.type != "cuda" or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
                    raise RuntimeError("Expected the established sm_120 CUDA device")
                if not (output / "run_clock.json").exists():
                    write_json(output / "run_clock.json", {
                        "task_id": TASK_ID, "keep_percent": cfg["keep_percent"],
                        "first_started_at_utc": utc_now(),
                        "note": "elapsed wall time includes downtime between authenticated resumes",
                    })
                task_status(output, cfg, state="running", n_images=80, packets=5,
                            attack_started=True)
                for batch in cfg["batch_indices"]:
                    run_batch(cfg, manifest, batch, output, device)
            from .report_paper_sparse_same80 import summarize
            result = summarize(cfg, manifest, output)
            task_status(output, cfg, state="complete", n_images=80, metrics=result["metrics"],
                        random_summary_sha256=file_hash(output / "summary.json"))
        except BaseException as exc:
            task_status(output, cfg, state="failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
