"""Paper-ready reporting for the v2/R8 same80 sparse sweep."""
from __future__ import annotations

from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch

from . import report_paper_sparse_same80 as legacy_report
from . import run_paper_sparse_same80_r8 as r8
from .run_m8_fedavg_naive import read_json, read_csv, write_json, write_csv, resolve, file_hash, utc_now


TASK_ID = r8.TASK_ID
PLAN_VERSION = r8.PLAN_VERSION
AMENDMENT_ID = r8.AMENDMENT_ID
PERCENTS = r8.PERCENTS
CANDIDATES = r8.CANDIDATES
SWEEP_ROOT = "results/paper_sparse_same80_keep50_30_20_R8_20260921"
RESULT_REPORT = ".ai/execution-results/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921/report.md"


def _validate_migration(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    migration = read_json(output / "migration_manifest.json")
    if migration.get("task_id") != TASK_ID or migration.get("plan_version") != PLAN_VERSION:
        raise RuntimeError("R8 migration identity differs")
    if migration.get("identity") != manifest.get("identity"):
        raise RuntimeError("R8 migration and input identities differ")
    if migration.get("candidate_ids") != list(CANDIDATES) or migration.get("v1_selection_reused"):
        raise RuntimeError("R8 migration candidate scope or selection provenance is invalid")
    return migration


def summarize(cfg: dict[str, Any], manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    baseline_cfg = r8.validate_config(cfg)
    percent = int(cfg["keep_percent"])
    if manifest.get("task_id") != TASK_ID or manifest.get("plan_version") != PLAN_VERSION:
        raise RuntimeError("R8 manifest task identity differs from the amendment")
    if manifest.get("run_tag") != cfg["run_tag"] or manifest.get("source_codec") != cfg["source_codec"]:
        raise RuntimeError("R8 manifest and report configuration differ")
    r8.legacy.verify_inputs(manifest)
    migration = _validate_migration(output, manifest)
    packet_audit = output / "packet_audit.json"
    if file_hash(packet_audit) != manifest["packet_audit_sha256"]:
        raise RuntimeError("R8 packet audit changed")
    baseline, naive_all80, baseline_rows = legacy_report._validate_baseline(cfg, manifest)
    identity = manifest["identity"]
    rows: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    communications = []
    originals, random_recs = [], []
    for batch in range(r8.N_BATCHES):
        folder = output / f"batch{batch:02d}"
        done = r8.checked_marker(folder / "summary.json", identity)
        if done is None or (done.get("n_images"), done.get("n"), done.get("T"), done.get("batch_index")) != (16, 16, 10, batch):
            raise RuntimeError(f"Incomplete R8 batch {batch}")
        if done.get("candidate_ids") != list(CANDIDATES):
            raise RuntimeError("R8 batch candidate coverage is not exactly 0..7")
        screens = [r8.checked_marker(folder / f"candidate{i:02d}/screen.json", identity) for i in CANDIDATES]
        if any(row is None or row.get("completed_iterations") != 5000 or row.get("truth_used") for row in screens):
            raise RuntimeError("All eight R8 observable screen candidates must be complete without truth")
        ranked = r8.rank_candidates(screens, list(CANDIDATES))
        advanced = [row["candidate"] for row in ranked[:3]]
        screen_selection = r8.checked_marker(folder / "screen_selection.json", identity)
        if (screen_selection is None or screen_selection.get("advanced") != advanced
                or screen_selection.get("truth_used") or screen_selection.get("candidate_ids") != list(CANDIDATES)):
            raise RuntimeError("Frozen R8 screen selection is invalid")
        finals = [r8.checked_marker(folder / f"candidate{i:02d}/final.json", identity) for i in advanced]
        if any(row is None or row.get("completed_iterations") != 25000 or row.get("truth_used") for row in finals):
            raise RuntimeError("All three R8 finalists must be complete without truth")
        final_ranked = r8.rank_candidates(finals, advanced)
        selected = int(final_ranked[0]["candidate"])
        final_selection = r8.checked_marker(folder / "final_selection.json", identity)
        if (final_selection is None or final_selection.get("selected_candidate") != selected
                or final_selection.get("truth_used") or final_selection.get("candidate_ids") != list(CANDIDATES)):
            raise RuntimeError("Frozen R8 final selection is invalid")
        if done.get("selected_candidate") != selected or done.get("advanced_candidates") != advanced:
            raise RuntimeError("R8 batch summary disagrees with observable selection")
        if any(r8.checked_marker(folder / f"candidate{i:02d}/cpa_result.json", identity) is None for i in CANDIDATES):
            raise RuntimeError("Missing completed R8 CPA candidate")
        batch_rows = read_csv(folder / "per_image_metrics.csv")
        inputs = torch.load(resolve(cfg["source_output"]) / f"batch{batch:02d}/inputs.pt",
                            map_location="cpu", weights_only=False, mmap=True)
        if len(batch_rows) != 16 or r8.legacy._identity_rows(batch_rows) != r8.legacy._identity_rows(inputs["metadata"]):
            raise RuntimeError("R8 reconstruction identity/order changed")
        if r8.legacy._identity_rows(batch_rows) != r8.legacy._identity_rows(manifest["metadata"][batch * 16:(batch + 1) * 16]):
            raise RuntimeError("R8 reconstruction differs from frozen all80 identities")
        if any(int(row["selected_candidate"]) != selected for row in batch_rows):
            raise RuntimeError("Per-image candidate selection is forbidden")
        random_rec = torch.load(folder / "rec.pt", map_location="cpu", weights_only=True)
        source_rec = torch.load(folder / "source_order_rec.pt", map_location="cpu", weights_only=True)
        legacy_report._assert_images(random_rec, 16, f"R8 random batch{batch} reconstruction")
        legacy_report._assert_images(source_rec, 16, f"R8 source-order batch{batch} reconstruction")
        order = torch.tensor([int(row["estimated_source_index"]) for row in batch_rows], dtype=torch.long)
        if sorted(order.tolist()) != list(range(16)) or not torch.equal(random_rec, source_rec.index_select(0, order)):
            raise RuntimeError("R8 evaluation must only permute saved final images")
        naive_batch = read_json(resolve(cfg["baseline_output"]) / f"batch{batch:02d}/summary.json")
        measured = legacy_report._metrics(batch_rows)
        for key in ("lpips_mean", "psnr_mean", "ssim_mean", "embedding_cos09"):
            if not np.isclose(measured[key], done[key], rtol=1e-7, atol=1e-9):
                raise RuntimeError(f"R8 batch summary differs for {key}")
        originals.append(legacy_report._unnormalize(inputs["x"]).cpu())
        random_recs.append(random_rec)
        rows.extend(batch_rows)
        communications.append(legacy_report._communication(manifest, batch))
        batches.append({
            "batch_index": batch, "n_images": 16, "selected_candidate": selected,
            "advanced_candidates": advanced, "candidate_ids": list(CANDIDATES),
            "random_metrics": measured,
            "naive_metrics": {key: naive_batch[f"{key}_mean"] for key in ("lpips", "psnr", "ssim")},
            "random_runtime_sec": done["runtime_sec"], "budget": done["budget"],
            "packet_bytes": communications[-1]["packet_bytes"],
        })
        del inputs, random_rec, source_rec
    if len(rows) != 80 or r8.legacy._identity_rows(rows) != r8.legacy._identity_rows(manifest["metadata"]):
        raise RuntimeError("R8 report must cover all80 original identities")
    random_all80 = torch.cat(random_recs)
    original_all80 = torch.cat(originals)
    legacy_report._assert_images(random_all80, 80, "R8 random all80 reconstruction")
    write_csv(output / "per_image_metrics.csv", rows)
    per_batch_rows = []
    for batch in batches:
        n, random = batch["naive_metrics"], batch["random_metrics"]
        per_batch_rows.append({
            "batch_index": batch["batch_index"], "selected_candidate": batch["selected_candidate"],
            "packet_bytes": batch["packet_bytes"], "naive_lpips_mean": n["lpips"],
            "naive_psnr_mean": n["psnr"], "naive_ssim_mean": n["ssim"],
            "random_lpips_mean": random["lpips_mean"], "random_psnr_mean": random["psnr_mean"],
            "random_ssim_mean": random["ssim_mean"], "random_embedding_cos09": random["embedding_cos09"],
            "runtime_sec": batch["random_runtime_sec"],
        })
    random_metrics = legacy_report._metrics(rows)
    naive_metrics = {key: baseline["metrics"][key] for key in ("lpips_mean", "psnr_mean", "ssim_mean", "embedding_cos09")}
    comparison = [
        {"keep_percent": percent, "condition": "naive_zero_fill", "method": "Zero fill (completed matched baseline)",
         "run_tag": baseline["config"]["run_tag"], "lpips_mean": naive_metrics["lpips_mean"],
         "psnr_mean": naive_metrics["psnr_mean"], "ssim_mean": naive_metrics["ssim_mean"],
         "embedding_cos09": naive_metrics["embedding_cos09"],
         "packet_bytes": sum(c["packet_bytes"] for c in communications),
         "retained_elements": sum(c["retained_elements"] for c in communications), "extra_bytes": 0},
        {"keep_percent": percent, "condition": "uniform_random_fill", "method": "Uniform random fill (8 at5k, top3 to25k)",
         "run_tag": cfg["run_tag"], "lpips_mean": random_metrics["lpips_mean"],
         "psnr_mean": random_metrics["psnr_mean"], "ssim_mean": random_metrics["ssim_mean"],
         "embedding_cos09": random_metrics["embedding_cos09"],
         "packet_bytes": sum(c["packet_bytes"] for c in communications),
         "retained_elements": sum(c["retained_elements"] for c in communications), "extra_bytes": 0},
    ]
    write_csv(output / "per_batch_metrics.csv", per_batch_rows)
    write_csv(output / "comparison_metrics.csv", comparison)
    shutil.copyfile(resolve(cfg["baseline_output"]) / "all80_rec.pt", output / "naive_all80_rec.pt")
    torch.save(random_all80, output / "all80_rec.pt")
    report_rows = [{**row, "naive_lpips": baseline_rows[i]["lpips"],
                    "naive_psnr": baseline_rows[i]["psnr"], "naive_ssim": baseline_rows[i]["ssim"]}
                   for i, row in enumerate(rows)]
    write_csv(output / "per_image_comparison.csv", report_rows)
    figure_manifest = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
        "keep_percent": percent, "candidate_ids": list(CANDIDATES),
        "migration_manifest_sha256": file_hash(output / "migration_manifest.json"),
        "all80": legacy_report._render_columns(
            [("Original", original_all80), ("Naive", naive_all80), ("Random R8", random_all80)],
            rows, output / "all80_original_vs_naive_vs_random.png",
            title=f"Same80 retained {percent}% | original / completed naive / R8 random fill", columns_per_row=5),
        "fixed_indices": legacy_report._render_fixed(
            [("Original", original_all80[:2]), ("Naive", naive_all80[:2]), ("Random R8", random_all80[:2])],
            rows[:2], output / "fixed_index0_1_original_vs_naive_vs_random.png"),
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
        f"# Same80 retained Top-k {percent}%: matched naive versus R8 random fill", "",
        f"This v2 amendment uses exactly candidate IDs0..7. It reuses only authenticated v1 artifacts for candidates0..7 after hash and optimizer-setting verification; v1 screen/final selections are not reused. V1 supersession evidence is recorded in `{resolve('results/paper_sparse_same80_keep50_30_20_R20_20260921/v1_supersession.json')}`.", "",
        "The exact existing ImageNet same80 data, five B16 batches, received packet and public checkpoint are unchanged. No victim retraining, update collection, sampling, quantization, rescale, error feedback or extra bytes were used.", "",
        f"Candidates screen at FIA5k, the public top3 continue by20k to total25k, and one whole16 candidate is selected by the same observable recompressed public-model relative-L2 score. Private evaluation truth is loaded only after selections are frozen.", "",
        "| Method | LPIPS (lower) | PSNR dB (higher) | SSIM (higher) | cos>0.9 |", "|---|---:|---:|---:|---:|",
        f"| {comparison[0]['method']} | {naive_metrics['lpips_mean']:.6f} | {naive_metrics['psnr_mean']:.4f} | {naive_metrics['ssim_mean']:.6f} | {naive_metrics['embedding_cos09']}/80 |",
        f"| {comparison[1]['method']} | {random_metrics['lpips_mean']:.6f} | {random_metrics['psnr_mean']:.4f} | {random_metrics['ssim_mean']:.6f} | {random_metrics['embedding_cos09']}/80 |", "",
        f"Quality changes are random minus naive: LPIPS {delta['lpips_mean_random_minus_naive']:+.6f}, PSNR {delta['psnr_mean_random_minus_naive']:+.4f} dB, SSIM {delta['ssim_mean_random_minus_naive']:+.6f}. Compute budgets differ and are disclosed.", "",
        f"Actual five-packet communication is {packet_total:,} bytes, with {sum(c['retained_elements'] for c in communications):,} retained values and zero additional random-fill bytes. The same-shape FP32 serialization reference is {dense_total:,} bytes; packet reduction is {100*(1-packet_total/dense_total):.6f}%.", "",
        "All80 metrics use the unchanged signed-source absolute-cosine Hungarian correspondence saved by the attack run, followed by the existing PSNR, SSIM and VGG LPIPS implementations. Plotting reads saved aligned tensors and does not recompute matching.", "",
        "![All80 original, naive and R8 reconstructions](all80_original_vs_naive_vs_random.png)", "",
        "![Fixed registered indices 0 and 1](fixed_index0_1_original_vs_naive_vs_random.png)", "",
        "Artifacts: `input_manifest.json`, `migration_manifest.json`, `packet_audit.json`, per-image/per-batch comparisons, saved reconstructions, candidate markers, selections and figure manifest.", "",
    ]
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")
    names = ["per_image_metrics.csv", "per_image_comparison.csv", "per_batch_metrics.csv",
             "comparison_metrics.csv", "all80_rec.pt", "naive_all80_rec.pt",
             "all80_original_vs_naive_vs_random.png", "fixed_index0_1_original_vs_naive_vs_random.png",
             "figure_manifest.json", "report.md"]
    result = {
        "schema_version": 3, "task_id": TASK_ID, "plan_version": PLAN_VERSION,
        "amendment_id": AMENDMENT_ID, "run_tag": cfg["run_tag"], "status": "complete",
        "git_sha": manifest["git_sha"], "config": cfg, "baseline_config": baseline_cfg,
        "identity": identity, "n_images": 80, "n": 16, "T": 10, "keep_percent": percent,
        "source_codec": cfg["source_codec"], "candidate_ids": list(CANDIDATES),
        "metrics": random_metrics, "naive_metrics": naive_metrics,
        "quality_delta_random_minus_naive": delta, "per_image_metrics": rows,
        "per_victim_batch": batches, "migration_manifest_sha256": file_hash(output / "migration_manifest.json"),
        "communication": {"scope": "attacked classifier.0.weight packet only",
                          "packet_bytes": packet_total, "fp32_packet_bytes": dense_total,
                          "retained_elements": sum(c["retained_elements"] for c in communications),
                          "n_packets": 5, "packet_reduction_fraction": 1 - packet_total / dense_total,
                          "extra_bytes_for_random_fill": 0},
        "communication_per_batch": communications, "randomization": r8.RANDOMIZATION,
        "source_paths": {"baseline_output": str(resolve(cfg["baseline_output"])),
                         "source_input_output": str(resolve(cfg["source_output"])),
                         "packet_audit": str(packet_audit), "migration": str(output / "migration_manifest.json"),
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
    config_paths = config_paths or [PROJECT / f"configs/cpa_fia_random_topk{p}_same80_R8_S5k_K3_F25k.json" for p in PERCENTS]
    results = []
    for config_path in config_paths:
        cfg = read_json(config_path)
        output = resolve(cfg["output_dir"])
        manifest, summary = read_json(output / "input_manifest.json"), read_json(output / "summary.json")
        if summary.get("status") != "complete" or summary.get("task_id") != TASK_ID or summary.get("plan_version") != PLAN_VERSION:
            raise RuntimeError(f"R8 fraction {cfg.get('keep_percent')} is not complete")
        if summary.get("identity") != manifest.get("identity"):
            raise RuntimeError("R8 fraction summary/manifest identity differs")
        results.append((cfg, manifest, summary, output))
    results.sort(key=lambda item: int(item[0]["keep_percent"]), reverse=True)
    metadata = results[0][1]["metadata"]
    for _, manifest, _, _ in results[1:]:
        if r8.legacy._identity_rows(manifest["metadata"]) != r8.legacy._identity_rows(metadata):
            raise RuntimeError("Combined R8 sweep original identities differ")
    sweep = resolve(SWEEP_ROOT)
    sweep.mkdir(parents=True, exist_ok=True)
    combined = []
    for cfg, _, summary, _ in results:
        percent = int(cfg["keep_percent"])
        for condition, method, metrics, run_tag in (
            ("naive_zero_fill", "Zero fill (completed matched baseline)", summary["naive_metrics"], summary["baseline_config"]["run_tag"]),
            ("uniform_random_fill", "Uniform random fill (8 at5k, top3 to25k)", summary["metrics"], cfg["run_tag"]),
        ):
            combined.append({"task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
                             "keep_percent": percent, "pair_id": f"same80_keep{percent}",
                             "condition": condition, "method": method, "run_tag": run_tag,
                             "candidate_count": CANDIDATE_COUNT, "lpips_mean": metrics["lpips_mean"],
                             "psnr_mean": metrics["psnr_mean"], "ssim_mean": metrics["ssim_mean"],
                             "embedding_cos09": metrics["embedding_cos09"],
                             "packet_bytes": summary["communication"]["packet_bytes"],
                             "retained_elements": summary["communication"]["retained_elements"],
                             "random_extra_bytes": 0})
    write_csv(sweep / "combined_comparison_metrics.csv", combined)
    source_batches = [torch.load(resolve(f"results/m6_uia_fp32_B16_fia/batch{batch:02d}/inputs.pt"),
                                 map_location="cpu", weights_only=False, mmap=True) for batch in range(5)]
    original = legacy_report._unnormalize(source_batches[0]["x"]).cpu()
    original_all80 = torch.cat([legacy_report._unnormalize(batch["x"]).cpu() for batch in source_batches])
    legacy_report._assert_images(original, 16, "fixed original batch")
    legacy_report._assert_images(original_all80, 80, "R8 all80 original")
    fixed_rows = source_batches[0]["metadata"][:2]
    fixed_columns: list[tuple[str, torch.Tensor]] = [("Original", original[:2])]
    all80_manifest = {"task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
                      "example_indices": [0, 1], "selection": "fixed registered global indices 0 and 1",
                      "candidate_ids": list(CANDIDATES), "conditions": []}
    first_naive = None
    for cfg, manifest, summary, output in results:
        naive = torch.load(output / "naive_all80_rec.pt", map_location="cpu", weights_only=True, mmap=True)
        random = torch.load(output / "all80_rec.pt", map_location="cpu", weights_only=True, mmap=True)
        legacy_report._assert_images(naive, 80, f"naive keep{cfg['keep_percent']} reconstruction")
        legacy_report._assert_images(random, 80, f"R8 random keep{cfg['keep_percent']} reconstruction")
        if first_naive is None:
            first_naive = naive
        elif not torch.equal(first_naive, naive):
            raise RuntimeError("Matched naive reconstruction changed across R8 fraction reports")
        percent = int(cfg["keep_percent"])
        fixed_columns.extend([(f"{percent}% Naive", naive[:2]), (f"{percent}% R8 Random", random[:2])])
        figure = legacy_report._render_columns(
            [("Original", original_all80), ("Naive", naive), ("R8 Random", random)], metadata,
            sweep / f"all80_comparison_keep{percent}.png",
            title=f"Paper same80 retained {percent}% | original / naive / R8 random", columns_per_row=5)
        all80_manifest["conditions"].append({"keep_percent": percent, "summary": str(output / "summary.json"),
                                              "summary_sha256": file_hash(output / "summary.json"), "figure": figure,
                                              "migration_manifest_sha256": summary["migration_manifest_sha256"]})
    all80_manifest["fixed_figure"] = legacy_report._render_fixed(
        fixed_columns, fixed_rows, sweep / "fixed_index0_1_comparison.png")
    unchanged10 = resolve("results/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k")
    all80_manifest["unchanged10"] = {
        "random_summary": str(unchanged10 / "summary.json"),
        "random_summary_sha256": file_hash(unchanged10 / "summary.json"),
        "random_input_manifest": str(unchanged10 / "input_manifest.json"),
        "random_input_manifest_sha256": file_hash(unchanged10 / "input_manifest.json"),
        "naive_summary": str(resolve("results/cpa_fia_topk10_same80_B16") / "summary.json"),
        "naive_summary_sha256": file_hash(resolve("results/cpa_fia_topk10_same80_B16") / "summary.json"),
        "note": "unchanged completed Top-k10 reference remains its original R20 budget; it was not relabeled R8",
    }
    all80_manifest["v1_supersession"] = str(resolve("results/paper_sparse_same80_keep50_30_20_R20_20260921/v1_supersession.json"))
    write_json(sweep / "sweep_manifest.json", all80_manifest)
    record = _read_execution_record()
    lines = [
        f"# CPA paper sparse same80 sweep report ({TASK_ID}, v2/R8)", "",
        "Status: completed the three authorized retained fractions 50%, 30% and 20% with exactly candidates0..7. V1 R20 was safely superseded at its authenticated checkpoint and remains preserved as partial evidence.", "",
        "The dataset, packets, public checkpoint, local SGD, CPA/FIA settings, screen5k, top3 continuation25k, public ranking and private evaluation boundary are unchanged by the amendment. Only candidate count changed from20 to8.", "",
        "| Retained | Naive LPIPS | R8 LPIPS | Naive PSNR | R8 PSNR | Naive SSIM | R8 SSIM |", "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cfg, _, summary, _ in results:
        lines.append(f"| {cfg['keep_percent']}% | {summary['naive_metrics']['lpips_mean']:.6f} | {summary['metrics']['lpips_mean']:.6f} | {summary['naive_metrics']['psnr_mean']:.4f} | {summary['metrics']['psnr_mean']:.4f} | {summary['naive_metrics']['ssim_mean']:.6f} | {summary['metrics']['ssim_mean']:.6f} |")
    lines += ["", "R8 uses 120 candidate CPA trajectories, 1,500,000 FIA iterations and 165 public ranking replays including authenticated reused candidate work. The paired quality differences are descriptive and the attack compute budgets differ from naive.", "", "## Actual commands", "", "```text"]
    lines += [str(command) for command in record.get("commands", [])]
    lines += ["```", "", "## Validation and artifacts", "",
              "Each R8 migration manifest records the v1 source identity, imported artifact hashes, candidate IDs0..7, optimizer settings and checkpoint provenance. V1 selection decisions and candidates8..19 were excluded.",
              f"Combined metrics: `{sweep / 'combined_comparison_metrics.csv'}`.",
              f"Fixed examples: `{sweep / 'fixed_index0_1_comparison.png'}`; all80 figures: `{sweep / 'all80_comparison_keep50.png'}`, `{sweep / 'all80_comparison_keep30.png'}`, `{sweep / 'all80_comparison_keep20.png'}`.",
              f"V1 supersession: `{resolve('results/paper_sparse_same80_keep50_30_20_R20_20260921/v1_supersession.json')}`; unchanged Top-k10 reference is linked in `{sweep / 'sweep_manifest.json'}`.", ""]
    (sweep / "report.md").write_text("\n".join(lines), encoding="utf-8")
    result_report = resolve(RESULT_REPORT)
    result_report.parent.mkdir(parents=True, exist_ok=True)
    result_report.write_text("\n".join(lines), encoding="utf-8")
    return {"task_id": TASK_ID, "plan_version": PLAN_VERSION, "amendment_id": AMENDMENT_ID,
            "status": "complete", "fractions": [int(cfg["keep_percent"]) for cfg, *_ in results],
            "candidate_ids": list(CANDIDATES), "combined_metrics": str(sweep / "combined_comparison_metrics.csv"),
            "sweep_manifest": str(sweep / "sweep_manifest.json"), "report": str(result_report),
            "report_sha256": file_hash(result_report)}


if __name__ == "__main__":
    summarize_sweep()
