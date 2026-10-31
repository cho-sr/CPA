"""Paper-ready reporting for the registered same80 random-fill sweep.

The report consumes saved candidate selections, saved reconstructions and the
completed matched naive outputs.  It performs no attack optimization and never
recomputes source matching during plotting.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from .run_m8_fedavg_naive import read_json, read_csv, write_json, write_csv, resolve, file_hash, utc_now
from .report_m10_packed4 import _append_run_once, _metrics, _font, _rgb
from .run_paper_sparse_same80 import (
    PROJECT, TASK_ID, PLAN_VERSION, PERCENTS, RANDOMIZATION, checked_marker,
    validate_config, verify_inputs, _identity_rows,
)


SWEEP_ROOT = "results/paper_sparse_same80_keep50_30_20_R20_20260921"
RESULT_REPORT = ".ai/execution-results/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921/report.md"
IDENTITY_FIELDS = ("global_index", "batch_index", "source_index", "dataset_index", "label", "image_path")


def _same_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(str(left[key]) == str(right[key]) for key in IDENTITY_FIELDS)


def _communication(manifest: dict[str, Any], batch: int) -> dict[str, Any]:
    value = dict(manifest["communication_per_batch"][batch])
    audit = manifest["packet_audits"][batch]
    packet = Path(manifest["identity"]["packets"][batch]["path"])
    if file_hash(packet) != manifest["identity"]["packets"][batch]["sha256"]:
        raise RuntimeError("Frozen packet changed during reporting")
    if int(value["packet_bytes"]) != packet.stat().st_size:
        raise RuntimeError("Packet accounting differs from binary length")
    return {
        "batch_index": batch, "source_codec": manifest["source_codec"],
        "packet_sha256": manifest["identity"]["packets"][batch]["sha256"],
        "retained_elements": int(audit["kept_elements"]),
        "packet_bytes": int(value["packet_bytes"]),
        "fp32_packet_bytes": int(value["fp32_packet_bytes"]),
        "value_bytes": int(value["value_bytes"]),
        "bitmap_bytes": int(value["bitmap_bytes"]),
        "header_bytes": int(value["header_bytes"]),
        "keep_fraction": float(value["keep_fraction"]),
        "packet_compression_ratio": float(value["fp32_packet_bytes"] / value["packet_bytes"]),
        "packet_reduction_fraction": float(1 - value["packet_bytes"] / value["fp32_packet_bytes"]),
        "random_fill_extra_bytes": 0,
    }


def _unnormalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor((.485, .456, .406))[None, :, None, None]
    std = x.new_tensor((.229, .224, .225))[None, :, None, None]
    return (x * std + mean).clamp(0, 1)


def _assert_images(tensor: torch.Tensor, n: int, label: str) -> None:
    if tuple(tensor.shape) != (n, 3, 224, 224) or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"Invalid {label} tensor")


def _render_columns(columns: list[tuple[str, torch.Tensor]], rows: list[dict[str, Any]],
                    path: Path, *, title: str, columns_per_row: int = 5) -> dict[str, Any]:
    if not columns or any(tensor.shape[0] != len(rows) for _, tensor in columns):
        raise RuntimeError("Comparison columns have inconsistent lengths")
    tile, gap, header, labels = 168, 10, 116, 44
    block = len(columns) * tile + (len(columns) - 1) * gap
    nrows = (len(rows) + columns_per_row - 1) // columns_per_row
    width = gap + columns_per_row * (block + gap)
    height = header + nrows * (tile + labels + gap)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 12), title, font=_font(26), fill="#0f172a")
    draw.text((gap, 50), "Fixed original identities; saved native tensors; no quality-based example selection", font=_font(18), fill="#475569")
    placements = []
    for index, row in enumerate(rows):
        left = gap + (index % columns_per_row) * (block + gap)
        top = header + (index // columns_per_row) * (tile + labels + gap)
        draw.text((left, top), f"g{int(row['global_index']):02d} | b{int(row['batch_index'])}/s{int(row['source_index']):02d}",
                  font=_font(17), fill="#0f172a")
        for col, (label, tensors) in enumerate(columns):
            x = left + col * (tile + gap)
            draw.text((x, top + 22), label, font=_font(15), fill="#334155")
            pixels = _rgb(tensors[index])
            image = Image.fromarray(pixels).resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, top + labels))
            placements.append({"global_index": int(row["global_index"]), "panel": label,
                               "box": [x, top + labels, x + tile, top + labels + tile],
                               "native_rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {"file": path.name, "n_images": len(rows), "columns": [label for label, _ in columns],
            "size": [width, height], "placements": placements,
            "rendering": "saved RGB tensors, clamp and isotropic Lanczos display resize"}


def _render_fixed(columns: list[tuple[str, torch.Tensor]], rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    return _render_columns(columns, rows, path, title="Paper same80 fixed illustration identities g0 and g1", columns_per_row=1)


def _validate_baseline(cfg: dict[str, Any], manifest: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor, list[dict[str, str]]]:
    baseline_root = resolve(cfg["baseline_output"])
    baseline = read_json(baseline_root / "summary.json")
    if file_hash(baseline_root / "summary.json") != manifest["baseline_summary_sha256"]:
        raise RuntimeError("Matched naive baseline summary changed")
    baseline_rows = read_csv(baseline_root / "per_image_metrics.csv")
    if len(baseline_rows) != 80 or _identity_rows(baseline_rows) != _identity_rows(manifest["metadata"]):
        raise RuntimeError("Matched naive baseline identity table differs")
    baseline_tensor = torch.load(baseline_root / "all80_rec.pt", map_location="cpu", weights_only=True, mmap=True)
    _assert_images(baseline_tensor, 80, "matched naive all80 reconstruction")
    expected_hash = baseline["output_hashes"].get("all80_rec.pt")
    if expected_hash and file_hash(baseline_root / "all80_rec.pt") != expected_hash:
        raise RuntimeError("Matched naive all80 reconstruction changed")
    return baseline, baseline_tensor, baseline_rows


def summarize(cfg: dict[str, Any], manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    baseline_cfg = validate_config(cfg)
    percent = int(cfg["keep_percent"])
    if manifest.get("task_id") != TASK_ID or manifest.get("plan_version") != PLAN_VERSION:
        raise RuntimeError("Manifest task identity differs from the handoff")
    if manifest.get("run_tag") != cfg["run_tag"] or manifest.get("source_codec") != cfg["source_codec"]:
        raise RuntimeError("Manifest and report configuration differ")
    verify_inputs(manifest)
    packet_audit = output / "packet_audit.json"
    if file_hash(packet_audit) != manifest["packet_audit_sha256"]:
        raise RuntimeError("Packet audit changed")
    baseline, naive_all80, baseline_rows = _validate_baseline(cfg, manifest)
    identity = manifest["identity"]
    rows: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    communications = []
    originals, random_recs = [], []
    for batch in range(5):
        folder = output / f"batch{batch:02d}"
        done = checked_marker(folder / "summary.json", identity)
        if done is None or (done.get("n_images"), done.get("n"), done.get("T"), done.get("batch_index")) != (16, 16, 10, batch):
            raise RuntimeError(f"Incomplete paper sparse batch {batch}")
        screens = [checked_marker(folder / f"candidate{i:02d}/screen.json", identity) for i in range(20)]
        if any(row is None or row.get("completed_iterations") != 5000 or row.get("truth_used") for row in screens):
            raise RuntimeError("All20 observable screen candidates must be complete without truth")
        ranked = rank_candidates(screens, list(range(20)))
        advanced = [row["candidate"] for row in ranked[:3]]
        screen_selection = checked_marker(folder / "screen_selection.json", identity)
        if screen_selection is None or screen_selection.get("advanced") != advanced or screen_selection.get("truth_used"):
            raise RuntimeError("Frozen screen selection is invalid")
        finals = [checked_marker(folder / f"candidate{i:02d}/final.json", identity) for i in advanced]
        if any(row is None or row.get("completed_iterations") != 25000 or row.get("truth_used") for row in finals):
            raise RuntimeError("All three observable finalists must be complete without truth")
        final_ranked = rank_candidates(finals, advanced)
        selected = int(final_ranked[0]["candidate"])
        final_selection = checked_marker(folder / "final_selection.json", identity)
        if final_selection is None or final_selection.get("selected_candidate") != selected or final_selection.get("truth_used"):
            raise RuntimeError("Frozen final selection is invalid")
        if done.get("selected_candidate") != selected or done.get("advanced_candidates") != advanced:
            raise RuntimeError("Batch summary disagrees with observable selection")
        if any(checked_marker(folder / f"candidate{i:02d}/cpa_result.json", identity) is None for i in range(20)):
            raise RuntimeError("Missing completed CPA candidate")
        batch_rows = read_csv(folder / "per_image_metrics.csv")
        inputs = torch.load(resolve(cfg["source_output"]) / f"batch{batch:02d}/inputs.pt",
                            map_location="cpu", weights_only=False, mmap=True)
        if len(batch_rows) != 16 or _identity_rows(batch_rows) != _identity_rows(inputs["metadata"]):
            raise RuntimeError("Random reconstruction identity/order changed")
        if _identity_rows(batch_rows) != _identity_rows(manifest["metadata"][batch * 16:(batch + 1) * 16]):
            raise RuntimeError("Random reconstruction differs from frozen all80 identities")
        if any(int(row["selected_candidate"]) != selected for row in batch_rows):
            raise RuntimeError("Per-image candidate selection is forbidden")
        random_rec = torch.load(folder / "rec.pt", map_location="cpu", weights_only=True)
        source_rec = torch.load(folder / "source_order_rec.pt", map_location="cpu", weights_only=True)
        _assert_images(random_rec, 16, f"random batch{batch} reconstruction")
        _assert_images(source_rec, 16, f"random batch{batch} source-order reconstruction")
        order = torch.tensor([int(row["estimated_source_index"]) for row in batch_rows], dtype=torch.long)
        if sorted(order.tolist()) != list(range(16)) or not torch.equal(random_rec, source_rec.index_select(0, order)):
            raise RuntimeError("Random evaluation must only permute saved final images")
        naive_batch = read_json(resolve(cfg["baseline_output"]) / f"batch{batch:02d}/summary.json")
        measured = _metrics(batch_rows)
        for key in ("lpips_mean", "psnr_mean", "ssim_mean", "embedding_cos09"):
            if not np.isclose(measured[key], done[key], rtol=1e-7, atol=1e-9):
                raise RuntimeError(f"Random batch summary differs for {key}")
        originals.append(_unnormalize(inputs["x"]).cpu())
        random_recs.append(random_rec)
        rows.extend(batch_rows)
        communications.append(_communication(manifest, batch))
        batches.append({
            "batch_index": batch, "n_images": 16, "selected_candidate": selected,
            "advanced_candidates": advanced, "random_metrics": measured,
            "naive_metrics": {key: naive_batch[f"{key}_mean"] for key in ("lpips", "psnr", "ssim")},
            "random_runtime_sec": done["runtime_sec"], "budget": done["budget"],
            "packet_bytes": communications[-1]["packet_bytes"],
        })
        del inputs, random_rec, source_rec
    if len(rows) != 80 or _identity_rows(rows) != _identity_rows(manifest["metadata"]):
        raise RuntimeError("Paper sparse report must cover all80 original identities")
    random_all80 = torch.cat(random_recs)
    original_all80 = torch.cat(originals)
    _assert_images(random_all80, 80, "random all80 reconstruction")
    write_csv(output / "per_image_metrics.csv", rows)
    per_batch_rows = []
    for batch in batches:
        n = batch["naive_metrics"]
        r = batch["random_metrics"]
        per_batch_rows.append({
            "batch_index": batch["batch_index"], "selected_candidate": batch["selected_candidate"],
            "packet_bytes": batch["packet_bytes"],
            "naive_lpips_mean": n["lpips"], "naive_psnr_mean": n["psnr"], "naive_ssim_mean": n["ssim"],
            "random_lpips_mean": r["lpips_mean"], "random_psnr_mean": r["psnr_mean"],
            "random_ssim_mean": r["ssim_mean"], "random_embedding_cos09": r["embedding_cos09"],
            "runtime_sec": batch["random_runtime_sec"],
        })
    random_metrics = _metrics(rows)
    naive_metrics = {key: baseline["metrics"][key] for key in ("lpips_mean", "psnr_mean", "ssim_mean", "embedding_cos09")}
    comparison = [
        {"keep_percent": percent, "condition": "naive_zero_fill", "method": "Zero fill (completed matched baseline)",
         "run_tag": baseline["config"]["run_tag"], "lpips_mean": naive_metrics["lpips_mean"],
         "psnr_mean": naive_metrics["psnr_mean"], "ssim_mean": naive_metrics["ssim_mean"],
         "embedding_cos09": naive_metrics["embedding_cos09"], "packet_bytes": sum(c["packet_bytes"] for c in communications),
         "retained_elements": sum(c["retained_elements"] for c in communications), "extra_bytes": 0},
        {"keep_percent": percent, "condition": "uniform_random_fill", "method": "Uniform random fill (20 at5k, top3 to25k)",
         "run_tag": cfg["run_tag"], "lpips_mean": random_metrics["lpips_mean"],
         "psnr_mean": random_metrics["psnr_mean"], "ssim_mean": random_metrics["ssim_mean"],
         "embedding_cos09": random_metrics["embedding_cos09"], "packet_bytes": sum(c["packet_bytes"] for c in communications),
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
        "task_id": TASK_ID, "keep_percent": percent,
        "all80": _render_columns(
            [("Original", original_all80), ("Naive", naive_all80), ("Random", random_all80)],
            rows, output / "all80_original_vs_naive_vs_random.png",
            title=f"Same80 retained {percent}% | original / completed naive / random fill", columns_per_row=5),
        "fixed_indices": _render_fixed(
            [("Original", original_all80[:2]), ("Naive", naive_all80[:2]), ("Random", random_all80[:2])],
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
        f"# Same80 retained Top-k {percent}%: matched naive versus random fill", "",
        "This task reuses the exact existing ImageNet same80 sample, five original B16 batches and saved received packets. It does not retrain the victim or recollect updates.", "",
        f"The source packet label is `{cfg['source_codec']}`. Retained coordinates are floor(N*{percent}/100), with unchanged FP32 values and ascending flat-index cutoff ties. Random filling samples only missing coordinates from U(-tau,tau), with tau equal to the minimum retained absolute value. Random filling adds zero transmitted bytes.", "",
        "All20 candidates use the saved packet, fixed CPA/FIA seeds and the upstream CPA/Direct FIA implementation. Candidates screen at FIA5k, the observable top3 continue to total25k, and one whole16 candidate is selected per batch by recompressed public-model update consistency. Private images and labels are loaded only after selections are frozen.", "",
        "| Method | LPIPS (lower) | PSNR dB (higher) | SSIM (higher) | cos>0.9 |", "|---|---:|---:|---:|---:|",
        f"| {comparison[0]['method']} | {naive_metrics['lpips_mean']:.6f} | {naive_metrics['psnr_mean']:.4f} | {naive_metrics['ssim_mean']:.6f} | {naive_metrics['embedding_cos09']}/80 |",
        f"| {comparison[1]['method']} | {random_metrics['lpips_mean']:.6f} | {random_metrics['psnr_mean']:.4f} | {random_metrics['ssim_mean']:.6f} | {random_metrics['embedding_cos09']}/80 |", "",
        f"Quality changes are random minus naive: LPIPS {delta['lpips_mean_random_minus_naive']:+.6f}, PSNR {delta['psnr_mean_random_minus_naive']:+.4f} dB, SSIM {delta['ssim_mean_random_minus_naive']:+.6f}. The random condition also uses the registered search budget: per batch 20 CPA25k, 20 FIA5k plus 3 continuations to20k, and23 ranking replays; the completed naive baseline uses one CPA25k/FIA25k attempt. These compute budgets differ.", "",
        f"Actual five-packet communication is {packet_total:,} bytes, with {sum(c['retained_elements'] for c in communications):,} retained values and zero additional random-fill bytes. The same-shape FP32 serialization reference is {dense_total:,} bytes; the packet reduction is {100*(1-packet_total/dense_total):.6f}%.", "",
        "All80 metrics use the unchanged signed-source absolute-cosine Hungarian correspondence saved by the attack run, followed by the existing PSNR, SSIM and VGG LPIPS implementations. Plotting reads the saved aligned tensors and does not recompute matching.", "",
        "![All80 original, naive and random reconstructions](all80_original_vs_naive_vs_random.png)", "",
        "![Fixed registered indices 0 and 1](fixed_index0_1_original_vs_naive_vs_random.png)", "",
        "Artifacts: `input_manifest.json`, `packet_audit.json`, `per_image_metrics.csv`, `per_image_comparison.csv`, `per_batch_metrics.csv`, `comparison_metrics.csv`, `all80_rec.pt`, `naive_all80_rec.pt`, candidate directories and `figure_manifest.json`.", "",
    ]
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")
    names = ["per_image_metrics.csv", "per_image_comparison.csv", "per_batch_metrics.csv",
             "comparison_metrics.csv", "all80_rec.pt", "naive_all80_rec.pt",
             "all80_original_vs_naive_vs_random.png", "fixed_index0_1_original_vs_naive_vs_random.png",
             "figure_manifest.json", "report.md"]
    result = {
        "schema_version": 2, "task_id": TASK_ID, "plan_version": PLAN_VERSION,
        "run_tag": cfg["run_tag"], "status": "complete", "git_sha": manifest["git_sha"],
        "config": cfg, "baseline_config": baseline_cfg, "identity": identity,
        "n_images": 80, "n": 16, "T": 10, "keep_percent": percent,
        "source_codec": cfg["source_codec"], "metrics": random_metrics,
        "naive_metrics": naive_metrics, "quality_delta_random_minus_naive": delta,
        "per_image_metrics": rows, "per_victim_batch": batches,
        "communication": {"scope": "attacked classifier.0.weight packet only",
                          "packet_bytes": packet_total, "fp32_packet_bytes": dense_total,
                          "retained_elements": sum(c["retained_elements"] for c in communications),
                          "n_packets": 5, "packet_reduction_fraction": 1 - packet_total / dense_total,
                          "extra_bytes_for_random_fill": 0},
        "communication_per_batch": communications, "randomization": RANDOMIZATION,
        "source_paths": {"baseline_output": str(resolve(cfg["baseline_output"])),
                         "source_input_output": str(resolve(cfg["source_output"])),
                         "packet_audit": str(packet_audit), "report": str(output / "report.md")},
        "budget_per_batch": batches[0]["budget"],
        "runtime_sec": {key: sum(float(batch["random_runtime_sec"].get(key, 0)) for batch in batches)
                        for key in {key for batch in batches for key in batch["random_runtime_sec"]}},
        "oracle_fills": 0, "ls_A": False, "debias": False, "mp_truncation": False,
        "stage2": False, "new_training": False, "new_update_collection": False,
        "completed_at_utc": utc_now(), "output_hashes": {name: file_hash(output / name) for name in names},
    }
    _append_run_once(resolve(cfg.get("results_jsonl", "results/runs.jsonl")), result)
    write_json(output / "summary.json", result)
    return result


def _read_execution_record() -> dict[str, Any]:
    path = resolve(SWEEP_ROOT) / "execution_record.json"
    if path.exists():
        return read_json(path)
    return {"python": sys.executable, "commands": []}


def summarize_sweep(config_paths: list[Path] | None = None) -> dict[str, Any]:
    """Create the combined three-fraction artifacts after all conditions finish."""
    config_paths = config_paths or [PROJECT / f"configs/cpa_fia_random_topk{p}_same80_R20_S5k_K3_F25k.json" for p in PERCENTS]
    results = []
    for config_path in config_paths:
        cfg = read_json(config_path)
        output = resolve(cfg["output_dir"])
        manifest = read_json(output / "input_manifest.json")
        summary = read_json(output / "summary.json")
        if summary.get("status") != "complete" or summary.get("task_id") != TASK_ID:
            raise RuntimeError(f"Fraction {cfg.get('keep_percent')} is not complete")
        if summary.get("identity") != manifest.get("identity"):
            raise RuntimeError("Fraction summary/manifest identity differs")
        results.append((cfg, manifest, summary, output))
    results.sort(key=lambda item: int(item[0]["keep_percent"]), reverse=True)
    metadata = results[0][1]["metadata"]
    for _, manifest, _, _ in results[1:]:
        if _identity_rows(manifest["metadata"]) != _identity_rows(metadata):
            raise RuntimeError("Combined sweep original identities differ")
    sweep = resolve(SWEEP_ROOT)
    sweep.mkdir(parents=True, exist_ok=True)
    combined = []
    for cfg, _, summary, _ in results:
        percent = int(cfg["keep_percent"])
        naive = summary["naive_metrics"]
        random = summary["metrics"]
        packet_bytes = summary["communication"]["packet_bytes"]
        retained = summary["communication"]["retained_elements"]
        for condition, method, metrics, run_tag in (
            ("naive_zero_fill", "Zero fill (completed matched baseline)", naive, summary["baseline_config"]["run_tag"]),
            ("uniform_random_fill", "Uniform random fill (20 at5k, top3 to25k)", random, cfg["run_tag"]),
        ):
            combined.append({"task_id": TASK_ID, "keep_percent": percent, "pair_id": f"same80_keep{percent}",
                             "condition": condition, "method": method, "run_tag": run_tag,
                             "lpips_mean": metrics["lpips_mean"], "psnr_mean": metrics["psnr_mean"],
                             "ssim_mean": metrics["ssim_mean"], "embedding_cos09": metrics["embedding_cos09"],
                             "packet_bytes": packet_bytes, "retained_elements": retained,
                             "random_extra_bytes": 0})
    write_csv(sweep / "combined_comparison_metrics.csv", combined)
    source_batches = [torch.load(resolve(f"results/m6_uia_fp32_B16_fia/batch{batch:02d}/inputs.pt"),
                                 map_location="cpu", weights_only=False, mmap=True)
                      for batch in range(5)]
    original = _unnormalize(source_batches[0]["x"]).cpu()
    original_all80 = torch.cat([_unnormalize(batch["x"]).cpu() for batch in source_batches])
    _assert_images(original, 16, "fixed original batch")
    _assert_images(original_all80, 80, "all80 original")
    fixed_rows = source_batches[0]["metadata"][:2]
    fixed_columns: list[tuple[str, torch.Tensor]] = [("Original", original[:2])]
    all80_manifest = {"task_id": TASK_ID, "plan_version": PLAN_VERSION, "example_indices": [0, 1],
                      "selection": "fixed registered global indices 0 and 1", "conditions": []}
    first_naive = None
    for cfg, manifest, summary, output in results:
        naive = torch.load(output / "naive_all80_rec.pt", map_location="cpu", weights_only=True, mmap=True)
        random = torch.load(output / "all80_rec.pt", map_location="cpu", weights_only=True, mmap=True)
        _assert_images(naive, 80, f"naive keep{cfg['keep_percent']} reconstruction")
        _assert_images(random, 80, f"random keep{cfg['keep_percent']} reconstruction")
        if first_naive is None:
            first_naive = naive
        elif not torch.equal(first_naive, naive):
            raise RuntimeError("Matched naive reconstruction changed across fraction reports")
        percent = int(cfg["keep_percent"])
        fixed_columns.extend([(f"{percent}% Naive", naive[:2]), (f"{percent}% Random", random[:2])])
        figure = _render_columns([("Original", original_all80), ("Naive", naive), ("Random", random)],
                                 metadata, sweep / f"all80_comparison_keep{percent}.png",
                                 title=f"Paper same80 retained {percent}% | original / naive / random", columns_per_row=5)
        all80_manifest["conditions"].append({"keep_percent": percent, "summary": str(output / "summary.json"),
                                              "summary_sha256": file_hash(output / "summary.json"),
                                              "figure": figure})
    fixed = _render_fixed(fixed_columns, fixed_rows, sweep / "fixed_index0_1_comparison.png")
    all80_manifest["fixed_figure"] = fixed
    unchanged10 = resolve("results/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k")
    all80_manifest["unchanged10"] = {
        "random_summary": str(unchanged10 / "summary.json"),
        "random_summary_sha256": file_hash(unchanged10 / "summary.json"),
        "random_input_manifest": str(unchanged10 / "input_manifest.json"),
        "random_input_manifest_sha256": file_hash(unchanged10 / "input_manifest.json"),
        "naive_summary": str(resolve("results/cpa_fia_topk10_same80_B16") / "summary.json"),
        "naive_summary_sha256": file_hash(resolve("results/cpa_fia_topk10_same80_B16") / "summary.json"),
        "note": "unchanged completed Top-k10 reference; no fourth new fraction was run",
    }
    write_json(sweep / "sweep_manifest.json", all80_manifest)
    record = _read_execution_record()
    lines = [
        f"# CPA paper sparse same80 sweep report ({TASK_ID})", "",
        "Status: completed three authorized retained fractions 50%, 30% and 20% using the exact existing paper same80 ImageNet data and saved packets.", "",
        "No new victim training, update collection, data sampling, fraction search, extra baseline restart or additional attack arm was used. Each condition uses its matched completed naive zero-fill summary and reconstruction for comparison.", "",
        "| Retained | Naive LPIPS | Random LPIPS | Naive PSNR | Random PSNR | Naive SSIM | Random SSIM |", "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cfg, _, summary, output in results:
        lines.append(f"| {cfg['keep_percent']}% | {summary['naive_metrics']['lpips_mean']:.6f} | {summary['metrics']['lpips_mean']:.6f} | {summary['naive_metrics']['psnr_mean']:.4f} | {summary['metrics']['psnr_mean']:.4f} | {summary['naive_metrics']['ssim_mean']:.6f} | {summary['metrics']['ssim_mean']:.6f} |")
    lines += ["", "The paired differences are descriptive quality changes. Random and naive conditions have different attack compute budgets, so changes are not attributed to missing-value imputation alone.", "", "## Actual commands", "", "```text"]
    lines += [str(command) for command in record.get("commands", [])]
    lines += ["```", "", "## Validation and artifacts", "",
              "The task-local manifests record all80 identities, source/input hashes, baseline hashes, packet hashes, historical codec labels, retained counts and communication bytes. The 50% historical codec was checked against the generic wire parser before production. Candidate ranking used public-model replay consistency and froze screen/final selections before private evaluation truth was loaded.", "",
              f"Combined metrics: `{sweep / 'combined_comparison_metrics.csv'}`.",
              f"Fixed examples: `{sweep / 'fixed_index0_1_comparison.png'}`; all80 figures: `{sweep / 'all80_comparison_keep50.png'}`, `{sweep / 'all80_comparison_keep30.png'}`, `{sweep / 'all80_comparison_keep20.png'}`.",
              f"Unchanged Top-k10 reference is linked in `{sweep / 'sweep_manifest.json'}`.", ""]
    (sweep / "report.md").write_text("\n".join(lines), encoding="utf-8")
    result_report = resolve(RESULT_REPORT)
    result_report.parent.mkdir(parents=True, exist_ok=True)
    result_report.write_text("\n".join(lines), encoding="utf-8")
    return {"task_id": TASK_ID, "status": "complete", "fractions": [int(cfg["keep_percent"]) for cfg, *_ in results],
            "combined_metrics": str(sweep / "combined_comparison_metrics.csv"),
            "sweep_manifest": str(sweep / "sweep_manifest.json"),
            "report": str(result_report), "report_sha256": file_hash(result_report)}


if __name__ == "__main__":
    summarize_sweep()
