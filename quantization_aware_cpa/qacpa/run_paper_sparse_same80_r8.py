"""Versioned R8 replacement for the paper same80 sparse sweep.

This module owns the v2 candidate-count amendment.  It uses the established
R20 runner only as an implementation adapter inside this process; all v2
configs, identities, markers, migration records and budgets are task-local.
The old R20 files and output tree are never rewritten by this module.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch

from . import run_paper_sparse_same80 as legacy
from .run_m8_fedavg_naive import PROJECT, WORKSPACE, read_json, write_json, read_csv, write_csv, file_hash, resolve, utc_now, execution_lock, evaluate_after_fia
from .random_fill import rank_candidates, candidate_seed
from .fia import evaluate_images


TASK_ID = legacy.TASK_ID
PLAN_VERSION = 2
AMENDMENT_ID = "CPA_PAPER_SPARSE_SAME80_K503020_R8_AMENDMENT_20260921"
BASE_PLAN_VERSION = 1
PERCENTS = (50, 30, 20)
CANDIDATE_COUNT = 8
CANDIDATES = tuple(range(CANDIDATE_COUNT))
N_BATCHES = 5
N_IMAGES = 80
N_PER_BATCH = 16
T_STEPS = 10
FC1_SHAPE = (4096, 25088)
R20_TAG = "results/cpa_fia_random_topk{percent}_same80_R20_S5k_K3_F25k"
R8_TAG = "cpa_fia_random_topk{percent}_same80_R8_S5k_K3_F25k"
AMENDMENT_PATH = WORKSPACE / ".ai/execution-handoffs/CPA_PAPER_SPARSE_SAME80_K503020_R8_AMENDMENT_20260921.md"
V1_ARCHIVE_PATH = WORKSPACE / ".ai/execution-handoffs/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921_V1_CLAIMED_ARCHIVE_20260921.md"

RANDOMIZATION = dict(legacy.RANDOMIZATION)
RANDOMIZATION["candidates"] = CANDIDATE_COUNT
RANDOMIZATION["selection_unit"] = "whole16-image candidate per victim batch; candidate IDs 0..7 only"


def task_status(output: Path, cfg: dict[str, Any], **kwargs: Any) -> None:
    legacy._status(output, cfg, task_id=TASK_ID, plan_version=PLAN_VERSION,
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
        "amendment_id", "candidate_count", "candidate_ids",
    }
    expected_keys = set(baseline) | changed - {"run_tag", "output_dir", "observation", "execution_scope"}
    if set(cfg) != expected_keys:
        raise ValueError("Unexpected paper sparse R8 configuration fields")
    for key, value in baseline.items():
        if key not in changed and cfg.get(key) != value:
            raise ValueError(f"Matched baseline condition changed: {key}")
    tag = R8_TAG.format(percent=percent)
    if cfg["run_tag"] != tag or cfg["output_dir"] != f"results/{tag}":
        raise ValueError("Paper sparse R8 results must use a task-local output")
    if cfg["baseline_output"] != baseline["output_dir"]:
        raise ValueError("The R8 comparison must use its matched naive output")
    if cfg["source_codec"] != legacy.SOURCE_CODECS[percent] or cfg["codec"] != baseline["codec"]:
        raise ValueError("The historical source codec label changed")
    if cfg["plan_version"] != PLAN_VERSION or cfg["amendment_id"] != AMENDMENT_ID:
        raise ValueError("The v2 amendment identity is missing")
    if cfg["candidate_count"] != CANDIDATE_COUNT or cfg["candidate_ids"] != list(CANDIDATES):
        raise ValueError("R8 requires exactly candidate IDs 0..7")
    if cfg.get("expected_packet_bytes") != legacy.expected_packet_bytes(percent):
        raise ValueError("The configured packet byte count is wrong")
    if cfg["sparsification"]["keep_fraction"] != percent / 100:
        raise ValueError("Only the retained fraction may vary")
    if cfg["randomization"] != RANDOMIZATION:
        raise ValueError("R8 random-fill selection protocol differs from the amendment")
    if cfg.get("n") != 16 or cfg.get("T") != 10 or cfg.get("batch_indices") != list(range(5)):
        raise ValueError("Only same80 B16/T10 grouping is authorized")
    return baseline


def _actual_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE / path


def _verify_code_hashes(code_hashes: dict[str, str], *, label: str) -> None:
    for rel, expected in code_hashes.items():
        path = _actual_path(rel)
        if not path.exists() or file_hash(path) != expected:
            raise RuntimeError(f"{label} dependency changed: {rel}")


def dependency_hashes(baselines: dict[int, dict[str, Any]]) -> dict[str, str]:
    """Freeze active v2 code plus authenticated v1 and baseline dependencies."""
    hashes: dict[str, str] = {}
    for summary in baselines.values():
        for rel, expected in summary["identity"]["code_sha256"].items():
            path = _actual_path(rel)
            actual = file_hash(path)
            if actual != expected:
                raise RuntimeError(f"Frozen baseline dependency changed: {rel}")
            hashes[rel] = expected
    for percent in PERCENTS:
        old_manifest = read_json(resolve(R20_TAG.format(percent=percent)) / "input_manifest.json")
        _verify_code_hashes(old_manifest["identity"]["code_sha256"], label=f"Frozen v1 Top-k{percent}")
        for rel, expected in old_manifest["identity"]["code_sha256"].items():
            hashes[f"v1_snapshot::{rel}"] = expected
    new_paths = [
        PROJECT / "qacpa/paper_sparse_codec.py",
        PROJECT / "qacpa/run_paper_sparse_same80_r8.py",
        PROJECT / "qacpa/report_paper_sparse_same80_r8.py",
        PROJECT / "scripts/run_paper_sparse_same80_r8.py",
        PROJECT / "docs/RANDOM_FILL_SAME80.md",
        PROJECT / "qacpa/random_fill.py",
        PROJECT / "qacpa/staged_fia.py",
        PROJECT / "qacpa/upstream_naive.py",
        AMENDMENT_PATH,
        V1_ARCHIVE_PATH,
    ]
    for path in new_paths:
        hashes[str(path.relative_to(WORKSPACE))] = file_hash(path)
    return hashes


@contextmanager
def _legacy_scope():
    """Let the frozen implementation adapter serve v2 without changing files."""
    saved = {
        "TASK_ID": legacy.TASK_ID,
        "PLAN_VERSION": legacy.PLAN_VERSION,
        "RANDOMIZATION": legacy.RANDOMIZATION,
        "validate_config": legacy.validate_config,
        "dependency_hashes": legacy.dependency_hashes,
    }
    legacy.TASK_ID = TASK_ID
    legacy.PLAN_VERSION = PLAN_VERSION
    legacy.RANDOMIZATION = RANDOMIZATION
    legacy.validate_config = validate_config
    legacy.dependency_hashes = dependency_hashes
    try:
        yield
    finally:
        legacy.TASK_ID = saved["TASK_ID"]
        legacy.PLAN_VERSION = saved["PLAN_VERSION"]
        legacy.RANDOMIZATION = saved["RANDOMIZATION"]
        legacy.validate_config = saved["validate_config"]
        legacy.dependency_hashes = saved["dependency_hashes"]


def prepare(config_path: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    validate_config(cfg)
    with _legacy_scope():
        return legacy.prepare(config_path, cfg, output)


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


def _verify_old_manifest(percent: int, new_manifest: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    old_root = resolve(R20_TAG.format(percent=percent))
    old_manifest_path = old_root / "input_manifest.json"
    old_manifest = read_json(old_manifest_path)
    if old_manifest.get("task_id") != TASK_ID or old_manifest.get("plan_version") != BASE_PLAN_VERSION:
        raise RuntimeError(f"R20 manifest identity is not the authenticated v1 snapshot: Top-k{percent}")
    if legacy._identity_rows(old_manifest["metadata"]) != legacy._identity_rows(new_manifest["metadata"]):
        raise RuntimeError(f"R20 and R8 same80 identities differ: Top-k{percent}")
    if old_manifest["inputs_sha256"] != new_manifest["inputs_sha256"]:
        raise RuntimeError(f"R20 and R8 source input hashes differ: Top-k{percent}")
    old_packets = old_manifest["identity"]["packets"]
    new_packets = new_manifest["identity"]["packets"]
    if [(p["batch_index"], p["sha256"], p["bytes"]) for p in old_packets] != [(p["batch_index"], p["sha256"], p["bytes"]) for p in new_packets]:
        raise RuntimeError(f"R20 and R8 packet identities differ: Top-k{percent}")
    _verify_code_hashes(old_manifest["identity"]["code_sha256"], label=f"Frozen v1 Top-k{percent}")
    return old_root, old_manifest


def _expected_cpa_settings(cfg: dict[str, Any], batch: int) -> dict[str, Any]:
    stage = cfg["stage1"]
    return {
        "n_components": 16, "seed": int(cfg["seed"] + batch), "device": str(cfg["device"]),
        "n_iter": int(stage["cpa_ica_max_iter"]), "lr": float(stage["cpa_ica_lr"]),
        "opt": stage["cpa_ica_optimizer"], "sch": stage["cpa_ica_scheduler"],
        "decor": float(stage["cpa_ica_decor"]), "T": float(stage["cpa_ica_T"]),
        "ne": float(stage["cpa_ica_ne"]), "nv": float(stage["cpa_ica_nv"]),
        "l1": float(stage["cpa_ica_l1"]), "tv": float(stage["cpa_ica_tv"]),
        "log_every": int(stage["cpa_ica_log_every"]), "debias": False,
        "mp_truncation": False, "mixing_matrix_computed": False,
    }


def _expected_fia_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    fia = cfg["fia"]
    return {
        "seed": int(fia["seed"]), "device": str(cfg["device"]),
        "n_iter": int(fia["n_iter"]), "lr": float(fia["lr"]),
        "tv": float(fia["tv"]), "fi": float(fia["fi"]),
        "optimizer": fia["optimizer"], "scheduler": fia["scheduler"],
        "log_every": int(fia["log_every"]),
        "driver": "staged_fia_v1; fixed full cosine horizon; original Direct.step",
    }


def _verify_checkpoint(path: Path, *, expected_stop: int | None = None) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    completed = int(state["completed_iterations"])
    if int(state["horizon"]) != 25000 or completed <= 0 or completed > 25000:
        raise RuntimeError(f"Invalid imported FIA checkpoint: {path}")
    if expected_stop is not None and completed != expected_stop:
        raise RuntimeError(f"Imported FIA checkpoint is not at the recorded screen boundary: {path}")
    return {"completed_iterations": completed, "horizon": int(state["horizon"])}


def _verify_importable_candidate(old_dir: Path, cfg: dict[str, Any], manifest: dict[str, Any],
                                 batch: int, candidate: int) -> dict[str, Any]:
    old_cpa = checked_marker(old_dir / "cpa_result.json", manifest["identity"])
    if old_cpa is None:
        raise RuntimeError(f"Candidate directory exists without authenticated CPA marker: {old_dir}")
    fill = read_json(old_dir / "fill_audit.json")
    packet_sha = manifest["identity"]["packets"][batch]["sha256"]
    if (fill.get("sampler"), fill.get("seed"), fill.get("packet_sha256"), fill.get("keep_percent"), fill.get("source_codec")) != (
            "uniform_tau_pcg64_v1", candidate_seed(batch, candidate), packet_sha,
            cfg["keep_percent"], cfg["source_codec"]):
        raise RuntimeError(f"Imported fill provenance differs: {old_dir}")
    cpa_manifest = read_json(old_dir / "cpa/cpa_manifest.json")
    settings = cpa_manifest["fingerprint"]["settings"]
    expected_cpa = _expected_cpa_settings(cfg, batch)
    for key, expected in expected_cpa.items():
        if settings.get(key) != expected:
            raise RuntimeError(f"Imported CPA settings differ at {old_dir}: {key}")
    _verify_code_hashes(cpa_manifest["fingerprint"]["source_hashes"], label="Imported CPA")
    imported = {
        "old_directory": str(old_dir),
        "old_cpa_marker_sha256": file_hash(old_dir / "cpa_result.json"),
        "old_sources_sha256": file_hash(old_dir / "sources.pt"),
        "old_fill_audit_sha256": file_hash(old_dir / "fill_audit.json"),
        "old_cpa_manifest_sha256": file_hash(old_dir / "cpa/cpa_manifest.json"),
        "candidate": candidate, "batch_index": batch,
        "seed": candidate_seed(batch, candidate),
    }
    old_screen = checked_marker(old_dir / "screen.json", manifest["identity"])
    checkpoint = old_dir / "fia/fia_checkpoint.pt"
    fia_manifest_path = old_dir / "fia/fia_manifest.json"
    if old_screen is not None:
        if old_screen.get("completed_iterations") != 5000 or old_screen.get("truth_used"):
            raise RuntimeError(f"Imported screen marker is not a truth-free 5k result: {old_dir}")
        expected_stop = 5000
    elif not checkpoint.exists():
        expected_stop = None
    else:
        expected_stop = None
    if checkpoint.exists():
        fia_manifest = read_json(fia_manifest_path)
        fia_settings = fia_manifest["fingerprint"]["settings"]
        for key, expected in _expected_fia_settings(cfg).items():
            if fia_settings.get(key) != expected:
                raise RuntimeError(f"Imported FIA settings differ at {old_dir}: {key}")
        _verify_code_hashes(fia_manifest["fingerprint"]["source_hashes"], label="Imported FIA")
        checkpoint_state = _verify_checkpoint(checkpoint, expected_stop=expected_stop)
        imported.update({
            "old_fia_manifest_sha256": file_hash(fia_manifest_path),
            "old_fia_checkpoint_sha256": file_hash(checkpoint),
            "checkpoint": checkpoint_state,
        })
    if old_screen is not None:
        imported.update({
            "old_screen_marker_sha256": file_hash(old_dir / "screen.json"),
            "old_screen_rec_sha256": file_hash(old_dir / "screen_rec.pt"),
            "old_screen_log_sha256": file_hash(old_dir / "screen_fia_log.json"),
        })
    return {"cpa": old_cpa, "screen": old_screen, "checkpoint": checkpoint if checkpoint.exists() else None,
            "provenance": imported}


def _import_one(old_dir: Path, new_dir: Path, cfg: dict[str, Any], old_manifest: dict[str, Any],
                new_manifest: dict[str, Any], batch: int, candidate: int) -> dict[str, Any]:
    if not old_dir.exists():
        return {"batch_index": batch, "candidate": candidate, "status": "fresh_no_v1_artifact"}
    verified = _verify_importable_candidate(old_dir, cfg, old_manifest, batch, candidate)
    new_dir.mkdir(parents=True, exist_ok=True)
    for name in ("sources.pt", "fill_audit.json", "cpa_log.json"):
        _copy(old_dir / name, new_dir / name)
    cpa_values = {key: value for key, value in verified["cpa"].items()
                  if key not in ("identity", "output_hashes")}
    cpa_values["imported_artifact_provenance"] = verified["provenance"]
    save_marker(new_dir / "cpa_result.json", new_manifest["identity"],
                ["sources.pt", "fill_audit.json", "cpa_log.json"], **cpa_values)
    checkpoint = verified["checkpoint"]
    if checkpoint is None:
        return {"batch_index": batch, "candidate": candidate, "status": "imported_cpa_only",
                "provenance": verified["provenance"]}
    _copy(checkpoint, new_dir / "fia/fia_checkpoint.pt")
    if verified["screen"] is not None:
        for name in ("screen_rec.pt", "screen_fia_log.json"):
            _copy(old_dir / name, new_dir / name)
        screen_values = {key: value for key, value in verified["screen"].items()
                         if key not in ("identity", "output_hashes")}
        screen_values["imported_artifact_provenance"] = verified["provenance"]
        save_marker(new_dir / "screen.json", new_manifest["identity"],
                    ["screen_rec.pt", "screen_fia_log.json", "sources.pt", "cpa_result.json"],
                    **screen_values)
        status = "imported_cpa_and_completed_screen"
    else:
        status = "imported_cpa_and_partial_screen_checkpoint"
    return {"batch_index": batch, "candidate": candidate, "status": status,
            "provenance": verified["provenance"],
            "new_cpa_marker_sha256": file_hash(new_dir / "cpa_result.json"),
            "new_checkpoint_sha256": file_hash(new_dir / "fia/fia_checkpoint.pt"),
            "new_screen_marker_sha256": file_hash(new_dir / "screen.json") if (new_dir / "screen.json").exists() else None}


def migrate_fraction(cfg: dict[str, Any], manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    path = output / "migration_manifest.json"
    old_root, old_manifest = _verify_old_manifest(cfg["keep_percent"], manifest)
    if path.exists():
        saved = read_json(path)
        if saved.get("identity") != manifest["identity"] or saved.get("plan_version") != PLAN_VERSION:
            raise RuntimeError(f"Incompatible existing R8 migration manifest: {path}")
        return saved
    records = []
    for batch in range(N_BATCHES):
        for candidate in CANDIDATES:
            old_dir = old_root / f"batch{batch:02d}/candidate{candidate:02d}"
            new_dir = output / f"batch{batch:02d}/candidate{candidate:02d}"
            records.append(_import_one(old_dir, new_dir, cfg, old_manifest, manifest, batch, candidate))
    result = {
        "schema_version": 1, "task_id": TASK_ID, "plan_version": PLAN_VERSION,
        "amendment_id": AMENDMENT_ID, "keep_percent": cfg["keep_percent"],
        "candidate_ids": list(CANDIDATES), "identity": manifest["identity"],
        "v1_root": str(old_root), "v1_manifest_sha256": file_hash(old_root / "input_manifest.json"),
        "v1_identity": old_manifest["identity"], "v1_selection_reused": False,
        "records": records, "old_candidates_8_19_used": False,
        "created_at_utc": utc_now(),
        "method_change": "candidate count 20 to exactly 8; all other scientific settings unchanged",
    }
    write_json(path, result)
    return result


def run_candidate(*, cfg: dict[str, Any], batch: int, candidate: int, packet: bytes,
                  folder: Path, identity: dict[str, Any], stop: int,
                  device: torch.device, notify_stage) -> dict[str, Any]:
    if candidate not in CANDIDATES:
        raise ValueError("R8 candidate IDs are exactly 0..7")
    with _legacy_scope():
        return legacy.run_candidate(cfg=cfg, batch=batch, candidate=candidate, packet=packet,
                                    folder=folder, identity=identity, stop=stop,
                                    device=device, notify_stage=notify_stage)


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
                    candidates_total=CANDIDATE_COUNT, stage=stage, fia_stop=stop,
                    fia_horizon=25000)

    screens = [run_candidate(cfg=cfg, batch=batch, candidate=candidate, packet=packet,
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
        "plan_version": PLAN_VERSION,
    }
    saved_selection = checked_marker(folder / "screen_selection.json", identity)
    if saved_selection is not None and any(saved_selection.get(key) != value for key, value in selection.items()):
        raise RuntimeError("Frozen R8 screen selection changed")
    if saved_selection is None:
        save_marker(folder / "screen_selection.json", identity,
                    [f"candidate{i:02d}/screen.json" for i in CANDIDATES], **selection)
    print(f"R8 SCREEN COMPLETE keep={cfg['keep_percent']} batch={batch} advanced={advanced}", flush=True)
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
        "truth_used": False, "metric_selection": False, "keep_percent": cfg["keep_percent"],
        "candidate_ids": list(CANDIDATES), "plan_version": PLAN_VERSION,
    }
    saved_final = checked_marker(folder / "final_selection.json", identity)
    if saved_final is not None and any(saved_final.get(key) != value for key, value in final_selection.items()):
        raise RuntimeError("Frozen R8 final selection changed")
    if saved_final is None:
        save_marker(folder / "final_selection.json", identity,
                    [f"candidate{i:02d}/final.json" for i in advanced], **final_selection)
    del packet
    notify_stage("evaluation_after_final_selection", selected, 25000)
    legacy.verify_inputs(manifest, batch)
    inputs = torch.load(resolve(cfg["source_output"]) / f"batch{batch:02d}/inputs.pt",
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
            for i in range(16)]
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
                       batch_index=batch, n_images=16, n=16, T=10,
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
    print(f"R8 COMPLETE keep={cfg['keep_percent']} batch={batch} selected={selected} LPIPS={done['lpips_mean']:.6f}", flush=True)
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
                task_status(output, cfg, state="prepared", n_images=80, packets=5,
                            attack_started=False, candidate_ids=list(CANDIDATES))
                return
            migration = migrate_fraction(cfg, manifest, output)
            if args.phase == "migrate":
                task_status(output, cfg, state="migrated", n_images=80, packets=5,
                            attack_started=False, migration_sha256=file_hash(output / "migration_manifest.json"),
                            imported_records=len(migration["records"]))
                return
            device = torch.device(cfg["device"])
            if device.type != "cuda" or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
                raise RuntimeError("Expected the established sm_120 CUDA device")
            if not (output / "run_clock.json").exists():
                write_json(output / "run_clock.json", {
                    "task_id": TASK_ID, "plan_version": PLAN_VERSION,
                    "amendment_id": AMENDMENT_ID, "keep_percent": cfg["keep_percent"],
                    "candidate_ids": list(CANDIDATES), "first_started_at_utc": utc_now(),
                    "note": "elapsed wall time includes downtime between authenticated resumes",
                })
            task_status(output, cfg, state="running", n_images=80, packets=5,
                        attack_started=True, migration_sha256=file_hash(output / "migration_manifest.json"))
            for batch in cfg["batch_indices"]:
                run_batch(cfg, manifest, batch, output, device)
            from .report_paper_sparse_same80_r8 import summarize
            result = summarize(cfg, manifest, output)
            task_status(output, cfg, state="complete", n_images=80, metrics=result["metrics"],
                        random_summary_sha256=file_hash(output / "summary.json"))
        except BaseException as exc:
            task_status(output, cfg, state="failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
