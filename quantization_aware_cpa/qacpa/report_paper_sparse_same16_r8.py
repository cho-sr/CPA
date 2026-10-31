"""Paper-ready reporting for the v3 SAME16 sparse sweep."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from . import report_paper_sparse_same80 as legacy_report
from . import run_paper_sparse_same16_r8 as r16
from .run_m8_fedavg_naive import read_json, read_csv, write_json, write_csv, resolve, file_hash, utc_now


TASK_ID = r16.TASK_ID
PLAN_VERSION = r16.PLAN_VERSION
AMENDMENT_ID = r16.AMENDMENT_ID
PERCENTS = r16.PERCENTS
CANDIDATES = r16.CANDIDATES
SWEEP_ROOT = r16.SWEEP_OUTPUT_ROOT
RESULT_REPORT = ".ai/execution-results/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921/report.md"


def _validate_migration(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    migration = read_json(output / "migration_manifest.json")
    if migration.get("task_id") != TASK_ID or migration.get("plan_version") != PLAN_VERSION:
        raise RuntimeError("SAME16 migration identity differs")
    if migration.get("identity") != manifest.get("identity"):
        raise RuntimeError("SAME16 migration and input identities differ")
    if migration.get("candidate_ids") != list(CANDIDATES):
        raise RuntimeError("SAME16 migration candidate scope is invalid")
    if migration.get("batch_indices") != [0] or migration.get("n_images") != 16:
        raise RuntimeError("SAME16 migration must cover original batch00 only")
    if migration.get("v1_selection_reused") or migration.get("v2_selection_reused"):
        raise RuntimeError("Historical selection decisions must not be reused")
    if migration.get("old_candidates_8_19_used") or migration.get("batches01_04_used"):
        raise RuntimeError("Unauthorized historical candidates or batches were used")
    records = migration.get("records", [])
    if {(row.get("batch_index"), row.get("candidate")) for row in records} != {(0, i) for i in CANDIDATES}:
        raise RuntimeError("SAME16 migration record coverage is incomplete")
    return migration


def _validate_naive_same16(cfg: dict[str, Any], manifest: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor, list[dict[str, str]]]:
    baseline_root = resolve(cfg["baseline_output"])
    baseline = read_json(baseline_root / "summary.json")
    if file_hash(baseline_root / "summary.json") != manifest["identity"]["baseline_summary_sha256"]:
        raise RuntimeError("Matched naive baseline summary changed")
    baseline_rows = read_csv(baseline_root / "per_image_metrics.csv")
    if len(baseline_rows) != 80:
        raise RuntimeError("Matched naive source must retain its historical per-image rows")
    naive_manifest = read_json(baseline_root / "input_manifest.json")
    if legacy_report._identity_rows(baseline_rows) != legacy_report._identity_rows(naive_manifest["metadata"]):
        raise RuntimeError("Matched naive identity table changed")
    if legacy_report._identity_rows(baseline_rows[:16]) != legacy_report._identity_rows(manifest["metadata"]):
        raise RuntimeError("Naive SAME16 rows do not match original batch00 identities")
    all_images = torch.load(baseline_root / "all80_rec.pt", map_location="cpu", weights_only=True, mmap=True)
    legacy_report._assert_images(all_images, 80, "matched naive historical reconstruction")
    expected_hash = baseline.get("output_hashes", {}).get("all80_rec.pt")
    if expected_hash and file_hash(baseline_root / "all80_rec.pt") != expected_hash:
        raise RuntimeError("Matched naive historical reconstruction changed")
    return baseline, all_images[:16].contiguous(), baseline_rows[:16]


def summarize(cfg: dict[str, Any], manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    baseline_cfg = r16.validate_config(cfg)
    percent = int(cfg["keep_percent"])
    if manifest.get("task_id") != TASK_ID or manifest.get("plan_version") != PLAN_VERSION:
        raise RuntimeError("SAME16 manifest task identity differs from the amendment")
    if manifest.get("amendment_id") != AMENDMENT_ID or manifest.get("run_tag") != cfg["run_tag"]:
        raise RuntimeError("SAME16 manifest and report configuration differ")
    r16.verify_inputs(manifest)
    migration = _validate_migration(output, manifest)
    packet_audit = output / "packet_audit.json"
    if file_hash(packet_audit) != manifest["packet_audit_sha256"]:
        raise RuntimeError("SAME16 packet audit changed")
    baseline, naive_all16, naive_rows = _validate_naive_same16(cfg, manifest)
    identity = manifest["identity"]
    rows: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    communications: list[dict[str, Any]] = []
    originals: list[torch.Tensor] = []
    random_recs: list[torch.Tensor] = []
    for batch in r16.BATCH_INDICES:
        folder = output / f"batch{batch:02d}"
        if any((output / f"batch{other:02d}").exists() for other in range(1, 5)):
            raise RuntimeError("SAME16 output contains an unauthorized later batch")
        done = r16.checked_marker(folder / "summary.json", identity)
        if done is None or (done.get("n_images"), done.get("n"), done.get("T"), done.get("batch_index")) != (16, 16, 10, 0):
            raise RuntimeError("Incomplete SAME16 batch00")
        if done.get("candidate_ids") != list(CANDIDATES) or done.get("batch_indices") != [0]:
            raise RuntimeError("SAME16 candidate/batch coverage is invalid")
        screens = [r16.checked_marker(folder / f"candidate{i:02d}/screen.json", identity) for i in CANDIDATES]
        if any(row is None or row.get("completed_iterations") != 5000 or row.get("truth_used") for row in screens):
            raise RuntimeError("All eight SAME16 observable screen candidates must be complete without truth")
        ranked = r16.rank_candidates(screens, list(CANDIDATES))
        advanced = [row["candidate"] for row in ranked[:3]]
        screen_selection = r16.checked_marker(folder / "screen_selection.json", identity)
        if (screen_selection is None or screen_selection.get("advanced") != advanced
                or screen_selection.get("truth_used") or screen_selection.get("candidate_ids") != list(CANDIDATES)):
            raise RuntimeError("Frozen SAME16 screen selection is invalid")
        finals = [r16.checked_marker(folder / f"candidate{i:02d}/final.json", identity) for i in advanced]
        if any(row is None or row.get("completed_iterations") != 25000 or row.get("truth_used") for row in finals):
            raise RuntimeError("All three SAME16 finalists must be complete without truth")
        final_ranked = r16.rank_candidates(finals, advanced)
        selected = int(final_ranked[0]["candidate"])
        final_selection = r16.checked_marker(folder / "final_selection.json", identity)
        if (final_selection is None or final_selection.get("selected_candidate") != selected
                or final_selection.get("truth_used") or final_selection.get("candidate_ids") != list(CANDIDATES)):
            raise RuntimeError("Frozen SAME16 final selection is invalid")
        if done.get("selected_candidate") != selected or done.get("advanced_candidates") != advanced:
            raise RuntimeError("SAME16 batch summary disagrees with observable selection")
        if any(r16.checked_marker(folder / f"candidate{i:02d}/cpa_result.json", identity) is None for i in CANDIDATES):
            raise RuntimeError("Missing completed SAME16 CPA candidate")
        batch_rows = read_csv(folder / "per_image_metrics.csv")
        inputs = torch.load(resolve(cfg["source_output"]) / "batch00/inputs.pt",
                            map_location="cpu", weights_only=False, mmap=True)
        if len(batch_rows) != 16 or legacy_report._identity_rows(batch_rows) != legacy_report._identity_rows(inputs["metadata"]):
            raise RuntimeError("SAME16 reconstruction identity/order changed")
        if legacy_report._identity_rows(batch_rows) != legacy_report._identity_rows(manifest["metadata"]):
            raise RuntimeError("SAME16 reconstruction differs from frozen batch00 identities")
        if any(int(row["selected_candidate"]) != selected for row in batch_rows):
            raise RuntimeError("Per-image candidate selection is forbidden")
        random_rec = torch.load(folder / "rec.pt", map_location="cpu", weights_only=True)
        source_rec = torch.load(folder / "source_order_rec.pt", map_location="cpu", weights_only=True)
        legacy_report._assert_images(random_rec, 16, "SAME16 random reconstruction")
        legacy_report._assert_images(source_rec, 16, "SAME16 source-order reconstruction")
        order = torch.tensor([int(row["estimated_source_index"]) for row in batch_rows], dtype=torch.long)
        if sorted(order.tolist()) != list(range(16)) or not torch.equal(random_rec, source_rec.index_select(0, order)):
            raise RuntimeError("SAME16 evaluation must only permute saved final images")
        measured = legacy_report._metrics(batch_rows)
        for key in ("lpips_mean", "psnr_mean", "ssim_mean", "embedding_cos09"):
            if not np.isclose(measured[key], done[key], rtol=1e-7, atol=1e-9):
                raise RuntimeError(f"SAME16 batch summary differs for {key}")
        originals.append(legacy_report._unnormalize(inputs["x"]).cpu())
        random_recs.append(random_rec)
        rows.extend(batch_rows)
        communications.append(legacy_report._communication(manifest, batch))
        batches.append({
            "batch_index": 0, "n_images": 16, "selected_candidate": selected,
            "advanced_candidates": advanced, "candidate_ids": list(CANDIDATES),
            "random_metrics": measured,
            "naive_metrics": legacy_report._metrics(naive_rows),
            "random_runtime_sec": done["runtime_sec"], "budget": done["budget"],
            "packet_bytes": communications[-1]["packet_bytes"],
        })
        del inputs, random_rec, source_rec
    if len(rows) != 16 or legacy_report._identity_rows(rows) != legacy_report._identity_rows(manifest["metadata"]):
        raise RuntimeError("SAME16 report must cover exactly original batch00 identities")
    random_all16 = torch.cat(random_recs)
    original_all16 = torch.cat(originals)
    legacy_report._assert_images(random_all16, 16, "SAME16 random reconstruction")
    legacy_report._assert_images(original_all16, 16, "SAME16 original reconstruction")
    legacy_report._assert_images(naive_all16, 16, "SAME16 naive reconstruction")
    write_csv(output / "per_image_metrics.csv", rows)
    per_batch_rows = []
    for batch in batches:
        naive, random = batch["naive_metrics"], batch["random_metrics"]
        per_batch_rows.append({
            "batch_index": batch["batch_index"], "n_images": 16,
            "selected_candidate": batch["selected_candidate"], "packet_bytes": batch["packet_bytes"],
            "naive_lpips_mean": naive["lpips_mean"], "naive_psnr_mean": naive["psnr_mean"],
            "naive_ssim_mean": naive["ssim_mean"], "naive_embedding_cos09": naive["embedding_cos09"],
            "random_lpips_mean": random["lpips_mean"], "random_psnr_mean": random["psnr_mean"],
            "random_ssim_mean": random["ssim_mean"], "random_embedding_cos09": random["embedding_cos09"],
            "runtime_sec": batch["random_runtime_sec"],
        })
    random_metrics = legacy_report._metrics(rows)
    naive_metrics = legacy_report._metrics(naive_rows)
    comparison = [
        {"keep_percent": percent, "n_images": 16, "condition": "naive_zero_fill",
         "method": "Zero fill (matched original batch00 rows)", "run_tag": baseline_cfg["run_tag"],
         "lpips_mean": naive_metrics["lpips_mean"], "psnr_mean": naive_metrics["psnr_mean"],
         "ssim_mean": naive_metrics["ssim_mean"], "embedding_cos09": naive_metrics["embedding_cos09"],
         "packet_bytes": sum(c["packet_bytes"] for c in communications),
         "retained_elements": sum(c["retained_elements"] for c in communications), "extra_bytes": 0},
        {"keep_percent": percent, "n_images": 16, "condition": "uniform_random_fill",
         "method": "Uniform random fill (8 at5k, top3 to25k)", "run_tag": cfg["run_tag"],
         "lpips_mean": random_metrics["lpips_mean"], "psnr_mean": random_metrics["psnr_mean"],
         "ssim_mean": random_metrics["ssim_mean"], "embedding_cos09": random_metrics["embedding_cos09"],
         "packet_bytes": sum(c["packet_bytes"] for c in communications),
         "retained_elements": sum(c["retained_elements"] for c in communications), "extra_bytes": 0},
    ]
    write_csv(output / "per_batch_metrics.csv", per_batch_rows)
    write_csv(output / "comparison_metrics.csv", comparison)
    torch.save(naive_all16, output / "naive_all16_rec.pt")
    torch.save(random_all16, output / "all16_rec.pt")
    report_rows = [{**row, "naive_lpips": naive_rows[i]["lpips"],
                    "naive_psnr": naive_rows[i]["psnr"], "naive_ssim": naive_rows[i]["ssim"]}
                   for i, row in enumerate(rows)]
    write_csv(output / "per_image_comparison.csv", report_rows)
    figure_manifest = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
        "keep_percent": percent, "candidate_ids": list(CANDIDATES), "n_images": 16,
        "batch_indices": [0], "migration_manifest_sha256": file_hash(output / "migration_manifest.json"),
        "same16": legacy_report._render_columns(
            [("Original", original_all16), ("Naive", naive_all16), ("Random R8", random_all16)],
            rows, output / "all16_original_vs_naive_vs_random.png",
            title=f"Same16 retained {percent}% | original / matched naive / R8 random fill", columns_per_row=4),
        "fixed_indices": legacy_report._render_columns(
            [("Original", original_all16[:2]), ("Naive", naive_all16[:2]), ("Random R8", random_all16[:2])],
            rows[:2], output / "fixed_index0_1_original_vs_naive_vs_random.png",
            title="SAME16 fixed illustration identities global_index 0 and 1", columns_per_row=1),
    }
    write_json(output / "figure_manifest.json", figure_manifest)
    packet_total = sum(c["packet_bytes"] for c in communications)
    dense_total = sum(c["fp32_packet_bytes"] for c in communications)
    delta = {
        "lpips_mean_random_minus_naive": random_metrics["lpips_mean"] - naive_metrics["lpips_mean"],
        "psnr_mean_random_minus_naive": random_metrics["psnr_mean"] - naive_metrics["psnr_mean"],
        "ssim_mean_random_minus_naive": random_metrics["ssim_mean"] - naive_metrics["ssim_mean"],
    }
    text = [
        f"# SAME16 retained Top-k {percent}%: matched naive versus R8 random fill", "",
        "This v3 amendment replaces the evaluated image scope with the original batch00 global indices 0..15. It keeps exactly candidate IDs0..7, the authenticated Top-k packet, screen5k, public top3 continuation to25k and truth-free public ranking.", "",
        "Naive and random metrics are aggregated over the same 16 original per-image rows. The historical same80 naive file is used only as a source for its batch00 rows and first16 saved reconstruction; no historical all-image mean is compared.", "",
        "No victim retraining, update collection, sampling, quantization, rescale, error feedback or extra bytes were used. Batches01..04 are outside this replacement scope.", "",
        "| Method | LPIPS (lower) | PSNR dB (higher) | SSIM (higher) | cos>0.9 |", "|---|---:|---:|---:|---:|",
        f"| {comparison[0]['method']} | {naive_metrics['lpips_mean']:.6f} | {naive_metrics['psnr_mean']:.4f} | {naive_metrics['ssim_mean']:.6f} | {naive_metrics['embedding_cos09']}/16 |",
        f"| {comparison[1]['method']} | {random_metrics['lpips_mean']:.6f} | {random_metrics['psnr_mean']:.4f} | {random_metrics['ssim_mean']:.6f} | {random_metrics['embedding_cos09']}/16 |", "",
        f"Quality changes are random minus naive: LPIPS {delta['lpips_mean_random_minus_naive']:+.6f}, PSNR {delta['psnr_mean_random_minus_naive']:+.4f} dB, SSIM {delta['ssim_mean_random_minus_naive']:+.6f}.", "",
        f"The original batch00 packet is {packet_total:,} bytes with {sum(c['retained_elements'] for c in communications):,} retained values and zero additional random-fill bytes. The same-shape FP32 serialization reference is {dense_total:,} bytes; packet reduction is {100*(1-packet_total/dense_total):.6f}%.", "",
        "SAME16 metrics use the unchanged signed-source absolute-cosine Hungarian correspondence saved by the attack run, followed by the existing PSNR, SSIM and VGG LPIPS implementations. Plotting reads saved aligned tensors and does not recompute matching.", "",
        "![SAME16 original, naive and R8 reconstructions](all16_original_vs_naive_vs_random.png)", "",
        "![Fixed registered indices 0 and 1](fixed_index0_1_original_vs_naive_vs_random.png)", "",
        "Artifacts include the input and migration manifests, packet audit, per-image/per-batch/paired CSVs, all16 reconstructions, candidate markers, selections and figure manifest.", "",
    ]
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")
    names = ["per_image_metrics.csv", "per_image_comparison.csv", "per_batch_metrics.csv",
             "comparison_metrics.csv", "all16_rec.pt", "naive_all16_rec.pt",
             "all16_original_vs_naive_vs_random.png", "fixed_index0_1_original_vs_naive_vs_random.png",
             "figure_manifest.json", "report.md"]
    result = {
        "schema_version": 4, "task_id": TASK_ID, "plan_version": PLAN_VERSION,
        "amendment_id": AMENDMENT_ID, "run_tag": cfg["run_tag"], "status": "complete",
        "git_sha": manifest["git_sha"], "config": cfg, "baseline_config": baseline_cfg,
        "identity": identity, "n_images": 16, "n": 16, "T": 10, "batch_indices": [0],
        "keep_percent": percent, "source_codec": cfg["source_codec"],
        "candidate_ids": list(CANDIDATES), "metrics": random_metrics, "naive_metrics": naive_metrics,
        "quality_delta_random_minus_naive": delta, "per_image_metrics": rows,
        "per_victim_batch": batches, "migration_manifest_sha256": file_hash(output / "migration_manifest.json"),
        "communication": {"scope": "attacked classifier.0.weight packet only", "packet_bytes": packet_total,
                          "fp32_packet_bytes": dense_total, "retained_elements": sum(c["retained_elements"] for c in communications),
                          "n_packets": 1, "packet_reduction_fraction": 1 - packet_total / dense_total,
                          "extra_bytes_for_random_fill": 0},
        "communication_per_batch": communications, "randomization": r16.RANDOMIZATION,
        "source_paths": {"baseline_output": str(resolve(cfg["baseline_output"])),
                         "source_input_output": str(resolve(cfg["source_output"])),
                         "packet_audit": str(output / "packet_audit.json"), "migration": str(output / "migration_manifest.json"),
                         "report": str(output / "report.md")},
        "budget_per_batch": batches[0]["budget"],
        "runtime_sec": {key: sum(float(batch["random_runtime_sec"].get(key, 0)) for batch in batches)
                        for key in {key for batch in batches for key in batch["random_runtime_sec"]}},
        "oracle_fills": 0, "ls_A": False, "debias": False, "mp_truncation": False,
        "stage2": False, "new_training": False, "new_update_collection": False,
        "completed_at_utc": utc_now(), "output_hashes": {name: file_hash(output / name) for name in names},
    }
    legacy_report._append_run_once(resolve(cfg.get("results_jsonl", "results/runs.jsonl")), result)
    write_json(output / "summary.json", result)
    return result


def _read_execution_record() -> dict[str, Any]:
    path = resolve(SWEEP_ROOT) / "execution_record.json"
    return read_json(path) if path.exists() else {"python": sys.executable, "commands": []}


def summarize_sweep(config_paths: list[Path] | None = None) -> dict[str, Any]:
    config_paths = config_paths or [PROJECT / f"configs/cpa_fia_random_topk{p}_same16_R8_S5k_K3_F25k.json" for p in PERCENTS]
    results = []
    for config_path in config_paths:
        cfg = read_json(config_path)
        output = resolve(cfg["output_dir"])
        manifest, summary = read_json(output / "input_manifest.json"), read_json(output / "summary.json")
        if summary.get("status") != "complete" or summary.get("task_id") != TASK_ID or summary.get("plan_version") != PLAN_VERSION:
            raise RuntimeError(f"SAME16 fraction {cfg.get('keep_percent')} is not complete")
        if summary.get("n_images") != 16 or summary.get("batch_indices") != [0] or summary.get("identity") != manifest.get("identity"):
            raise RuntimeError("Combined SAME16 sweep identity differs")
        results.append((cfg, manifest, summary, output))
    results.sort(key=lambda item: int(item[0]["keep_percent"]), reverse=True)
    metadata = results[0][1]["metadata"]
    for _, manifest, _, _ in results[1:]:
        if legacy_report._identity_rows(manifest["metadata"]) != legacy_report._identity_rows(metadata):
            raise RuntimeError("Combined SAME16 original identities differ")
    sweep = resolve(SWEEP_ROOT)
    sweep.mkdir(parents=True, exist_ok=True)
    combined = []
    for cfg, _, summary, _ in results:
        percent = int(cfg["keep_percent"])
        for condition, method, metrics, run_tag in (
            ("naive_zero_fill", "Zero fill (matched original batch00 rows)", summary["naive_metrics"], summary["baseline_config"]["run_tag"]),
            ("uniform_random_fill", "Uniform random fill (8 at5k, top3 to25k)", summary["metrics"], cfg["run_tag"]),
        ):
            combined.append({"task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
                             "keep_percent": percent, "n_images": 16, "batch_indices": [0],
                             "pair_id": f"same16_batch00_keep{percent}", "condition": condition,
                             "method": method, "run_tag": run_tag, "candidate_count": len(CANDIDATES),
                             "lpips_mean": metrics["lpips_mean"], "psnr_mean": metrics["psnr_mean"],
                             "ssim_mean": metrics["ssim_mean"], "embedding_cos09": metrics["embedding_cos09"],
                             "packet_bytes": summary["communication"]["packet_bytes"],
                             "retained_elements": summary["communication"]["retained_elements"], "random_extra_bytes": 0})
    write_csv(sweep / "combined_comparison_metrics.csv", combined)
    inputs = torch.load(resolve("results/m6_uia_fp32_B16_fia/batch00/inputs.pt"),
                        map_location="cpu", weights_only=False, mmap=True)
    original = legacy_report._unnormalize(inputs["x"]).cpu()
    legacy_report._assert_images(original, 16, "SAME16 original batch00")
    fixed_rows = inputs["metadata"][:2]
    fixed_columns: list[tuple[str, torch.Tensor]] = [("Original", original[:2])]
    sweep_manifest = {"task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
                      "n_images": 16, "batch_indices": [0], "example_indices": [0, 1],
                      "selection": "fixed original batch00 global indices 0 and 1",
                      "candidate_ids": list(CANDIDATES), "conditions": []}
    first_naive = None
    for cfg, manifest, summary, output in results:
        naive = torch.load(output / "naive_all16_rec.pt", map_location="cpu", weights_only=True, mmap=True)
        random = torch.load(output / "all16_rec.pt", map_location="cpu", weights_only=True, mmap=True)
        legacy_report._assert_images(naive, 16, f"naive SAME16 reconstruction keep{cfg['keep_percent']}")
        legacy_report._assert_images(random, 16, f"R8 SAME16 reconstruction keep{cfg['keep_percent']}")
        if first_naive is None:
            first_naive = naive
        elif not torch.equal(first_naive, naive):
            raise RuntimeError("Matched naive SAME16 reconstruction changed across fractions")
        percent = int(cfg["keep_percent"])
        fixed_columns.extend([(f"{percent}% Naive", naive[:2]), (f"{percent}% R8 Random", random[:2])])
        figure = legacy_report._render_columns(
            [("Original", original), ("Naive", naive), ("R8 Random", random)], metadata,
            sweep / f"same16_comparison_keep{percent}.png",
            title=f"Paper SAME16 retained {percent}% | original / naive / R8 random", columns_per_row=4)
        sweep_manifest["conditions"].append({"keep_percent": percent, "summary": str(output / "summary.json"),
                                              "summary_sha256": file_hash(output / "summary.json"), "figure": figure,
                                              "migration_manifest_sha256": summary["migration_manifest_sha256"]})
    sweep_manifest["fixed_figure"] = legacy_report._render_columns(
        fixed_columns, fixed_rows, sweep / "fixed_index0_1_comparison.png",
        title="SAME16 fixed illustration identities global_index 0 and 1", columns_per_row=1)
    historical10 = resolve("results/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k")
    historical10_naive = resolve("results/cpa_fia_topk10_same80_B16")
    sweep_manifest["historical_topk10"] = {
        "random_summary": str(historical10 / "summary.json"),
        "random_summary_sha256": file_hash(historical10 / "summary.json"),
        "random_batch00_rows": str(historical10 / "batch00/per_image_metrics.csv"),
        "random_batch00_reconstruction": str(historical10 / "batch00/rec.pt"),
        "naive_summary": str(historical10_naive / "summary.json"),
        "naive_summary_sha256": file_hash(historical10_naive / "summary.json"),
        "naive_batch00_rows": str(historical10_naive / "batch00/per_image_metrics.csv"),
        "note": "Historical Top-k10 remains its original R20 budget; these paths are provenance only and are not re-aggregated here.",
    }
    sweep_manifest["v2_supersession"] = str(resolve("results/paper_sparse_same80_keep50_30_20_R8_20260921/v2_supersession.json"))
    write_json(sweep / "sweep_manifest.json", sweep_manifest)
    record = _read_execution_record()
    lines = [
        f"# CPA paper sparse SAME16 sweep report ({TASK_ID}, v3/R8)", "",
        "Status: completed the three authorized retained fractions 50%, 30% and 20% over original batch00 global indices0..15 with exactly candidates0..7. The v2 same80 wrapper was safely superseded and preserved as partial evidence.", "",
        "Naive and random values are paired on the same 16 original per-image rows. The public packet, local SGD, CPA/FIA settings, screen5k, public top3 continuation25k, ranking and truth-free selection boundary are unchanged. Batches01..04 were not run.", "",
        "| Retained | Naive LPIPS | R8 LPIPS | Naive PSNR | R8 PSNR | Naive SSIM | R8 SSIM |", "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cfg, _, summary, _ in results:
        lines.append(f"| {cfg['keep_percent']}% | {summary['naive_metrics']['lpips_mean']:.6f} | {summary['metrics']['lpips_mean']:.6f} | {summary['naive_metrics']['psnr_mean']:.4f} | {summary['metrics']['psnr_mean']:.4f} | {summary['naive_metrics']['ssim_mean']:.6f} | {summary['metrics']['ssim_mean']:.6f} |")
    lines += ["", "SAME16 uses 24 candidate CPA trajectories, 600,000 CPA iterations, 300,000 FIA iterations and 33 public ranking replays across the three fractions, with authenticated batch00 candidate state reused only when its settings and hashes match.", "", "## Actual commands", "", "```text"]
    lines += [str(command) for command in record.get("commands", [])]
    lines += ["```", "", "## Validation and artifacts", "",
              "Each SAME16 migration manifest records the authenticated source root, imported artifact hashes, batch00/global-index coverage, candidate IDs0..7 and no reused selection decisions.",
              f"Combined metrics: `{sweep / 'combined_comparison_metrics.csv'}`.",
              f"Fixed examples: `{sweep / 'fixed_index0_1_comparison.png'}`; SAME16 figures: `{sweep / 'same16_comparison_keep50.png'}`, `{sweep / 'same16_comparison_keep30.png'}`, `{sweep / 'same16_comparison_keep20.png'}`.",
              f"v2 supersession: `{resolve('results/paper_sparse_same80_keep50_30_20_R8_20260921/v2_supersession.json')}`; historical Top-k10 provenance is linked in `{sweep / 'sweep_manifest.json'}`.", ""]
    (sweep / "report.md").write_text("\n".join(lines), encoding="utf-8")
    result_report = resolve(RESULT_REPORT)
    result_report.parent.mkdir(parents=True, exist_ok=True)
    result_report.write_text("\n".join(lines), encoding="utf-8")
    del inputs
    return {"task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
            "status": "complete", "fractions": [int(cfg["keep_percent"]) for cfg, *_ in results],
            "candidate_ids": list(CANDIDATES), "n_images": 16,
            "combined_metrics": str(sweep / "combined_comparison_metrics.csv"),
            "sweep_manifest": str(sweep / "sweep_manifest.json"), "report": str(result_report),
            "report_sha256": file_hash(result_report)}


if __name__ == "__main__":
    summarize_sweep()
