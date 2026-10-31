"""Same80 sparse packet -> 20 random fills -> CPA -> FIA5k -> top3 -> FIA25k."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import numpy as np
import torch

from .run_m8_fedavg_naive import (
    PROJECT, WORKSPACE, read_json, write_json, read_csv, write_csv, file_hash,
    resolve, utc_now, status, execution_lock, evaluate_after_fia,
)
from .run_topk_ratios_same80 import validate_config as validate_baseline, verify_inputs
from .report_topk_ratios_same80 import _communication
from .random_fill import candidate_seed, random_fill_packet, consistency_score, rank_candidates
from .staged_fia import run_staged_fia
from .upstream_naive import run_upstream_cpa, _save_checkpoint
from .fia import evaluate_images

PERCENTS = (10,)
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


def validate_config(cfg: dict) -> dict:
    percent = cfg.get("keep_percent")
    if type(percent) is not int or percent not in PERCENTS:
        raise ValueError("User selected Top-k10 only for random filling")
    baseline = read_json(PROJECT / f"configs/cpa_fia_topk{percent}_same80_B16.json")
    validate_baseline(baseline)
    added = {"randomization", "baseline_output"}
    changed = {"run_tag", "output_dir", "observation", "execution_scope"}
    if set(cfg) != set(baseline) | added:
        raise ValueError("Unexpected random-fill config fields")
    for key, value in baseline.items():
        if key not in changed and cfg[key] != value:
            raise ValueError(f"Baseline condition changed: {key}")
    tag = f"cpa_fia_random_topk{percent}_same80_R20_S5k_K3_F25k"
    if cfg["run_tag"] != tag or cfg["output_dir"] != f"results/{tag}":
        raise ValueError("Random-fill results must have a separate output")
    if cfg["baseline_output"] != baseline["output_dir"] or cfg["randomization"] != RANDOMIZATION:
        raise ValueError("Randomization/ranking protocol differs from the frozen design")
    return baseline


def code_hashes(baseline: dict) -> dict:
    hashes = dict(baseline["identity"]["code_sha256"])
    # Include every reused reporting dependency as well as the new driver.
    for name in ("random_fill.py", "staged_fia.py", "run_random_fill_same80.py",
                 "report_random_fill_same80.py",
                 "report_m10_packed4.py", "communication_codec.py", "errors.py"):
        path = PROJECT / "qacpa" / name
        hashes[str(path.relative_to(WORKSPACE))] = file_hash(path)
    for name in ("scripts/run_random_fill_topk10_same80.bat", "docs/RANDOM_FILL_SAME80.md"):
        path = PROJECT / name
        hashes[str(path.relative_to(WORKSPACE))] = file_hash(path)
    for rel, expected in hashes.items():
        if file_hash(WORKSPACE / rel) != expected:
            raise RuntimeError(f"Frozen baseline/new implementation changed: {rel}")
    return hashes


def prepare(config_path: Path, cfg: dict, output: Path) -> dict:
    baseline_cfg = validate_config(cfg)
    source = resolve(cfg["baseline_output"])
    baseline = read_json(source / "summary.json")
    old_manifest = read_json(source / "input_manifest.json")
    if baseline["status"] != "complete" or baseline["n_images"] != 80 or baseline["config"] != baseline_cfg:
        raise RuntimeError("Expected the completed matched zero-fill baseline")
    if old_manifest["identity"] != baseline["identity"]:
        raise RuntimeError("Baseline input identity changed")
    hashes = code_hashes(baseline)
    verify_inputs(old_manifest)
    weights = Path(old_manifest["weights_path"])
    if file_hash(weights) != old_manifest["weights_sha256"]:
        raise RuntimeError("Public round0 checkpoint changed")
    packets, communications = [], []
    for batch in range(5):
        folder = source / f"batch{batch:02d}"
        comm = _communication(folder, batch, cfg["keep_percent"])
        done = read_json(folder / "summary.json")
        if done["identity"] != baseline["identity"] or done["output_hashes"]["observation.bin"] != comm["sha256"]:
            raise RuntimeError("Packet differs from the one attacked by zero fill")
        packets.append({"batch_index": batch, "path": str(folder / "observation.bin"),
                        "sha256": comm["sha256"], "bytes": comm["packet_bytes"]})
        communications.append(comm)
    identity = {"config_sha256": file_hash(config_path), "code_sha256": hashes,
                "baseline_summary_sha256": file_hash(source / "summary.json"),
                "baseline_manifest_sha256": file_hash(source / "input_manifest.json"),
                "weights_sha256": old_manifest["weights_sha256"],
                "packets": packets, "versions": {"torch": torch.__version__, "numpy": np.__version__}}
    path = output / "input_manifest.json"
    if path.exists():
        manifest = read_json(path)
        if manifest["identity"] != identity:
            raise RuntimeError("Incompatible resume: frozen input/code/config changed")
        if manifest["inputs_sha256"] != old_manifest["inputs_sha256"] or manifest["metadata"] != old_manifest["metadata"]:
            raise RuntimeError("Original data identity changed")
        return manifest
    manifest = {"schema_version": 1, "run_tag": cfg["run_tag"], "identity": identity,
                "git_sha": old_manifest["git_sha"], "inputs_sha256": old_manifest["inputs_sha256"],
                "metadata": old_manifest["metadata"], "communication_per_batch": communications,
                "n_images": 80, "n": 16, "T": 10, "created_at_utc": utc_now(),
                "new_training": False, "new_update_collection": False,
                "randomization": RANDOMIZATION}
    write_csv(output / "image_manifest.csv", manifest["metadata"])
    write_json(path, manifest)
    write_json(config_path.with_suffix(".lock.json"), {"config_sha256": identity["config_sha256"],
               "input_manifest_sha256": file_hash(path), "frozen_at_utc": utc_now()})
    return manifest


def checked_marker(path: Path, identity: dict) -> dict | None:
    if not path.exists():
        return None
    saved = read_json(path)
    if saved["identity"] != identity:
        raise RuntimeError(f"Artifact identity differs: {path}")
    for name, expected in saved["output_hashes"].items():
        if file_hash(path.parent / name) != expected:
            raise RuntimeError(f"Artifact hash differs: {path.parent / name}")
    return saved


def save_marker(path: Path, identity: dict, names: list[str], **values) -> dict:
    result = {"identity": identity, **values,
              "output_hashes": {name: file_hash(path.parent / name) for name in names}}
    write_json(path, result)
    return result


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_candidate(*, cfg: dict, batch: int, candidate: int, packet: bytes,
                   folder: Path, identity: dict, stop: int, device: torch.device,
                   notify_stage) -> dict:
    """Attack boundary: received packet and public protocol only; no source cache."""
    folder.mkdir(parents=True, exist_ok=True)
    phase = "screen" if stop == RANDOMIZATION["screen_iterations"] else "final"
    previous = checked_marker(folder / f"{phase}.json", identity)
    if previous is not None:
        return previous
    cpa_done = checked_marker(folder / "cpa_result.json", identity)
    if cpa_done is None:
        notify_stage("random_fill_CPA", candidate, stop)
        matrix, audit = random_fill_packet(packet, candidate_seed(batch, candidate))
        if matrix.shape != (4096, 25088):
            raise RuntimeError("Expected the same FC1 observation shape")
        write_json(folder / "fill_audit.json", audit)
        cpa = run_upstream_cpa(Y_hat=matrix, n_components=16, device=device,
                               config=cfg["stage1"], output_dir=folder / "cpa", seed=cfg["seed"] + batch)
        sources = cpa.pop("S_hat")
        if sources.shape != (16, 25088) or not torch.isfinite(sources).all():
            raise RuntimeError("CPA must retain all16 sources")
        _save_checkpoint(folder / "sources.pt", sources)
        write_json(folder / "cpa_log.json", cpa)
        cpa_done = save_marker(folder / "cpa_result.json", identity,
                              ["sources.pt", "fill_audit.json", "cpa_log.json"],
                              candidate=candidate, batch_index=batch, wall_time_sec=cpa["wall_time_sec"])
        del matrix, cpa, sources
        free_memory()
    sources = torch.load(folder / "sources.pt", map_location="cpu", weights_only=True)
    notify_stage(f"FIA_{phase}", candidate, stop)
    fia = run_staged_fia(target_z=sources.abs().contiguous(), device=device, config=cfg["fia"],
                         output_dir=folder / "fia", seed=cfg["fia"]["seed"], stop=stop)
    rec = fia.pop("rec")
    _save_checkpoint(folder / f"{phase}_rec.pt", rec)
    write_json(folder / f"{phase}_fia_log.json", fia)
    free_memory()
    notify_stage(f"observable_ranking_{phase}", candidate, stop)
    started = time.perf_counter()
    score = consistency_score(rec=rec, packet=packet, keep_percent=cfg["keep_percent"], device=device)
    result = save_marker(folder / f"{phase}.json", identity,
                         [f"{phase}_rec.pt", f"{phase}_fia_log.json", "sources.pt", "cpa_result.json"],
                         candidate=candidate, batch_index=batch, completed_iterations=stop, **score,
                         ranking_wall_time_sec=time.perf_counter() - started,
                         fia_optimizer_wall_time_sec=fia["optimizer_wall_time_sec"],
                         completed_at_utc=utc_now())
    del rec, sources, fia
    free_memory()
    return result


def run_batch(cfg: dict, manifest: dict, batch: int, output: Path, device: torch.device) -> None:
    folder = output / f"batch{batch:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    identity = manifest["identity"]
    if checked_marker(folder / "summary.json", identity) is not None:
        return
    started = time.perf_counter()
    packet_ref = identity["packets"][batch]
    if file_hash(Path(packet_ref["path"])) != packet_ref["sha256"]:
        raise RuntimeError("Received packet changed")
    packet = Path(packet_ref["path"]).read_bytes()

    def notify_stage(stage, candidate, stop):
        status(output, cfg, state="running", batch_index=batch, candidate=candidate,
               candidates_total=20, stage=stage, fia_stop=stop, fia_horizon=25000)

    screens = [run_candidate(cfg=cfg, batch=batch, candidate=candidate, packet=packet,
                            folder=folder / f"candidate{candidate:02d}", identity=identity,
                            stop=5000, device=device, notify_stage=notify_stage) for candidate in range(20)]
    ranked = rank_candidates(screens, list(range(20)))
    advanced = [row["candidate"] for row in ranked[:3]]
    selection = {"ranked_candidates": [row["candidate"] for row in ranked], "advanced": advanced,
                 "scores": [{"candidate": row["candidate"], "relative_l2": row["relative_l2"]} for row in ranked],
                 "rule": RANDOMIZATION["ranking"], "truth_used": False}
    saved_selection = checked_marker(folder / "screen_selection.json", identity)
    if saved_selection is not None and any(saved_selection[key] != value for key, value in selection.items()):
        raise RuntimeError("Frozen screen selection changed")
    if saved_selection is None:
        save_marker(folder / "screen_selection.json", identity,
                    [f"candidate{i:02d}/screen.json" for i in range(20)], **selection)
    print(f"SCREEN COMPLETE batch={batch} advanced={advanced}", flush=True)
    finals = [run_candidate(cfg=cfg, batch=batch, candidate=candidate, packet=packet,
                           folder=folder / f"candidate{candidate:02d}", identity=identity,
                           stop=25000, device=device, notify_stage=notify_stage) for candidate in advanced]
    final_ranked = rank_candidates(finals, advanced)
    selected = int(final_ranked[0]["candidate"])
    save_marker(folder / "final_selection.json", identity,
                [f"candidate{i:02d}/final.json" for i in advanced],
                selected_candidate=selected, ranked_candidates=[row["candidate"] for row in final_ranked],
                scores=[{"candidate": row["candidate"], "relative_l2": row["relative_l2"]} for row in final_ranked],
                truth_used=False)
    del packet
    # Truth is first loaded after all candidate selection is permanently recorded.
    notify_stage("evaluation_after_final_selection", selected, 25000)
    verify_inputs(manifest, batch)
    inputs = torch.load(resolve(cfg["source_output"]) / f"batch{batch:02d}/inputs.pt",
                        map_location="cpu", weights_only=False, mmap=True)
    selected_folder = folder / f"candidate{selected:02d}"
    sources = torch.load(selected_folder / "sources.pt", map_location="cpu", weights_only=True)
    source_recs = torch.load(selected_folder / "final_rec.pt", map_location="cpu", weights_only=True)
    recs, correspondence = evaluate_after_fia(sources, source_recs, inputs["Z_evaluation_only"])
    metrics = evaluate_images(x_true=inputs["x"], rec=recs.to(device), device=device)
    rows = [{**inputs["metadata"][i], **correspondence[i], "run_tag": cfg["run_tag"], "n": 16, "T": 10,
             "selected_candidate": selected, "ls_A": False, "oracle_fill": False,
             **{key: float(getattr(metrics, key)[i]) for key in ("lpips", "psnr", "ssim")}} for i in range(16)]
    _save_checkpoint(folder / "source_order_rec.pt", source_recs)
    _save_checkpoint(folder / "rec.pt", recs)
    write_csv(folder / "per_image_metrics.csv", rows)
    cpa_time = sum(read_json(folder / f"candidate{i:02d}/cpa_result.json")["wall_time_sec"] for i in range(20))
    final_by_id = {row["candidate"]: row for row in finals}
    fia_time = sum(final_by_id.get(row["candidate"], row)["fia_optimizer_wall_time_sec"] for row in screens)
    ranking_time = sum(row["ranking_wall_time_sec"] for row in screens + finals)
    done = save_marker(folder / "summary.json", identity,
                       ["screen_selection.json", "final_selection.json", "source_order_rec.pt", "rec.pt", "per_image_metrics.csv"],
                       batch_index=batch, n_images=16, n=16, T=10, selected_candidate=selected, advanced_candidates=advanced,
                       oracle_fills=0, ls_A=False, source_cos_per_source=[row["source_abs_cosine"] for row in rows],
                       embedding_cos09=sum(row["measured_recovered_cos09"] for row in rows),
                       **{f"{key}_mean": float(np.mean([row[key] for row in rows])) for key in ("lpips", "ssim", "psnr")},
                       runtime_sec={"cpa": cpa_time, "fia_optimizer": fia_time, "ranking": ranking_time,
                                    "current_invocation": time.perf_counter() - started},
                       budget={"cpa_runs": 20, "cpa_iterations_total": 500000,
                               "fia_iterations_total": 160000, "ranking_replays": 23}, completed_at_utc=utc_now())
    print(f"COMPLETE batch={batch} selected={selected} LPIPS={done['lpips_mean']:.6f}", flush=True)
    del inputs, metrics, recs, source_recs, sources
    free_memory()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "run", "report"), default="prepare")
    args = parser.parse_args()
    cfg = read_json(args.config)
    validate_config(cfg)
    output = resolve(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    with execution_lock(output):
        try:
            manifest = prepare(args.config.resolve(), cfg, output)
            if args.phase == "prepare":
                status(output, cfg, state="prepared", n_images=80, packets=5, attack_started=False)
                return
            if args.phase == "run":
                device = torch.device(cfg["device"])
                if device.type != "cuda" or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
                    raise RuntimeError("Expected established sm_120 CUDA device")
                if not (output / "run_clock.json").exists():
                    write_json(output / "run_clock.json", {"first_started_at_utc": utc_now(),
                               "note": "elapsed wall time includes any downtime between resumed processes"})
                for batch in cfg["batch_indices"]:
                    run_batch(cfg, manifest, batch, output, device)
            from .report_random_fill_same80 import summarize
            result = summarize(cfg, manifest, output)
            status(output, cfg, state="complete", n_images=80, metrics=result["metrics"])
        except BaseException as exc:
            status(output, cfg, state="failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
