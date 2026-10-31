"""Post-gate private evaluation and all-image reporting for same80 bits8..4."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .fia import evaluate_images
from .run_m8_fedavg_naive import evaluate_after_fia
from .run_same80_bits_r100 import (
    PROJECT, PRIVATE_ROOT, PUBLIC_ROOT, SAME80_IMAGE_MANIFEST, SOURCE_ROOT,
    _resolve_project_path, _validate_config, gate,
)
from .same80_bits_r100_core import (
    BITS, CHECKPOINT_SHA256, N_BATCHES, TASK_ID, TaskError, ensure_free_space,
    execution_lock, file_sha256, read_json, tensor_sha256, utc_now, write_json,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise TaskError(f"refusing to write empty report CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rgb(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != 3 or tuple(image.shape[-2:]) != (224, 224):
        raise TaskError("comparison PNG requires native 3x224x224 images")
    array = image.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().numpy()
    return np.ascontiguousarray(array)


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf")):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def render_comparison(originals: torch.Tensor, naive: torch.Tensor, cora: torch.Tensor,
                      rows: list[dict[str, Any]], path: Path, *, title: str,
                      columns: int = 4) -> dict[str, Any]:
    n = len(rows)
    expected = (n, 3, 224, 224)
    if any(tuple(value.shape) != expected or not bool(torch.isfinite(value).all())
           for value in (originals, naive, cora)):
        raise TaskError(f"comparison PNG inputs must each have shape {expected} and finite pixels")
    tile, margin, gap, header, caption = 224, 18, 8, 72, 42
    block = 3 * tile + 2 * gap
    row_h = tile + caption + gap
    size = (margin + columns * (block + gap),
            header + ((n + columns - 1) // columns) * (row_h + gap) + margin)
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), title, font=_font(25), fill="#0f172a")
    draw.text((margin, 45), "Original image order | independent B16 victim batches | native 224x224 tiles",
              font=_font(16), fill="#475569")
    placements = []
    panel_values = (("Original", originals), ("Naive", naive), ("Public CoRA", cora))
    for index, row in enumerate(rows):
        left = margin + (index % columns) * (block + gap)
        top = header + (index // columns) * (row_h + gap)
        global_index = int(row["global_index"])
        draw.text((left, top),
                  f"g{global_index:02d} b{int(row['batch_index'])}/s{int(row['source_index']):02d}  "
                  f"LPIPS N {float(row['naive_lpips']):.3f} / C {float(row['cora_lpips']):.3f}",
                  font=_font(14), fill="#0f172a")
        for panel, (label, tensor) in enumerate(panel_values):
            x, y = left + panel * (tile + gap), top + caption
            draw.text((x, top + 20), label, font=_font(15), fill="#334155")
            pixels = _rgb(tensor[index])
            canvas.paste(Image.fromarray(pixels), (x, y))
            placements.append({"global_index": global_index, "panel": label,
                               "box": [x, y, x + tile, y + tile],
                               "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    canvas.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)
    with Image.open(path) as saved:
        for item in placements:
            actual = hashlib.sha256(np.asarray(saved.crop(tuple(item["box"]))).tobytes()).hexdigest()
            if actual != item["rgb_sha256"]:
                raise TaskError("saved comparison PNG changed a native RGB image tile")
    return {"file": path.name, "n_images": n, "size": list(size),
            "tile_size": [224, 224], "placements": placements,
            "native_pixels_verified": True}


def _load_original_batch(batch: int, registration: dict[str, Any]) -> dict[str, Any]:
    source = SOURCE_ROOT / f"batch{batch:02d}/inputs.pt"
    if file_sha256(source) != registration["identity"]["source_batches"][batch]["input_sha256"]:
        raise TaskError(f"private cached input hash changed in batch{batch:02d}")
    payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or "x" not in payload or "Z_evaluation_only" not in payload:
        raise TaskError("post-gate evaluation requires original images and evaluation-only features")
    x, truth, metadata = payload["x"], payload["Z_evaluation_only"], payload.get("metadata")
    if (tuple(x.shape) != (16, 3, 224, 224) or tuple(truth.shape) != (16, 25088)
            or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(truth).all())
            or not isinstance(metadata, list) or len(metadata) != 16):
        raise TaskError(f"invalid post-gate evaluation inputs in batch{batch:02d}")
    return {"x": x.detach().cpu(), "truth": truth.detach().cpu(), "metadata": metadata}


def _metric_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, float]:
    return {f"{arm}_{name}_mean": float(np.mean([float(row[f"{arm}_{name}"]) for row in rows]))
            for name in ("lpips", "psnr", "ssim", "source_abs_cosine")}


def _selection_history(public_cell: Path) -> dict[str, Any]:
    screen = read_json(public_cell / "selection_5000.json")
    final = read_json(public_cell / "final_selection.json")
    detailed_screen = []
    for entry in screen["ranking"]:
        candidate = int(entry["candidate_index"])
        folder = public_cell / f"member{candidate:03d}"
        sample = read_json(folder / "sample.json")
        cpa = read_json(folder / "cpa_result.json")
        result = read_json(folder / "screen_result.json")
        detailed_screen.append({
            **entry, "candidate_id": f"{candidate:03d}", "seed": sample["seed"],
            "sample_sha256": sample["sample_sha256"], "source_sha256": result["sources_sha256"],
            "screen_rec_sha256": result["screen_rec_sha256"],
            "checkpoint_5000_sha256": result["checkpoint_5000_sha256"],
            "cpa_wall_seconds": cpa["wall_seconds"],
            "cpa_peak_cuda_allocated_bytes": cpa["peak_cuda_allocated_bytes"],
            "screen_wall_seconds": result["wall_seconds"],
            "screen_peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
            "cpa_attempts": read_json(folder / "cpa_attempts.json").get("attempts", []),
            "screen_fia_attempts": read_json(folder / "fia/screen_fia_attempts.json").get("attempts", []),
        })
    detailed_final = []
    for entry in final["final_ranking"]:
        candidate = int(entry["candidate_index"])
        folder = public_cell / f"member{candidate:03d}"
        result = read_json(folder / "final_result.json")
        sample = read_json(folder / "sample.json")
        detailed_final.append({**entry, "candidate_id": f"{candidate:03d}",
                               "seed": sample["seed"], "sample_sha256": sample["sample_sha256"],
                               "source_sha256": result["sources_sha256"],
                               "final_checkpoint_sha256": result["final_checkpoint_sha256"],
                               "final_checkpoint_integrity_sha256": result["final_checkpoint_integrity_sha256"],
                               "final_rec_sha256": result["final_rec_sha256"],
                               "fia_identity_sha256": result["fia_identity_sha256"],
                               "wall_seconds": result["wall_seconds"],
                               "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
                               "continuation_attempts": read_json(folder / "fia/continuation_fia_attempts.json").get("attempts", [])})
    return {
        "screen_status": screen["status"],
        "screen_candidate_count": screen["candidate_count"],
        "screen_ranking": detailed_screen,
        "screen_selected_candidate_ids": screen["selected_candidate_ids"],
        "screen_selection_sha256": file_sha256(public_cell / "selection_5000.json"),
        "final_status": final["status"],
        "final_ranking": detailed_final,
        "selected_candidate_index": final["selected_candidate_index"],
        "final_selection_sha256": file_sha256(public_cell / "final_selection.json"),
        "private_data_used": False,
    }


def _evaluate_bit(public_root: Path, registration: dict[str, Any], bits: int,
                  device: torch.device, output: Path) -> dict[str, Any]:
    x_parts, truth_parts, naive_recs, cora_recs, original_rgb_parts = [], [], [], [], []
    metadata_all: list[dict[str, Any]] = []
    correspondence_naive: list[dict[str, Any]] = []
    correspondence_cora: list[dict[str, Any]] = []
    batch_provenance = []
    selection_history = {}
    for batch in range(N_BATCHES):
        started = time.perf_counter()
        private = _load_original_batch(batch, registration)
        cell = public_root / f"bit{bits:02d}/batch{batch:02d}"
        naive_dir = cell / "naive"
        naive_sources = torch.load(naive_dir / "sources.pt", map_location="cpu", weights_only=True)
        naive_record = read_json(naive_dir / "fia/final_result.json")
        naive_rec_path = naive_dir / "fia/rec_25000.pt"
        if (tensor_sha256(naive_sources) != naive_record.get("sources_sha256")
                or file_sha256(naive_rec_path) != naive_record.get("rec_file_sha256")):
            raise TaskError("Naive endpoint/source hashes changed after the public gate")
        naive_source_rec = torch.load(naive_rec_path, map_location="cpu", weights_only=True)

        final_selection = read_json(cell / "final_selection.json")
        winner = int(final_selection["selected_candidate_index"])
        winner_dir = cell / f"member{winner:03d}"
        winner_result = read_json(winner_dir / "final_result.json")
        winner_sources = torch.load(winner_dir / "sources.pt", map_location="cpu", weights_only=True)
        winner_rec_path = winner_dir / "final_25000_rec.pt"
        if (file_sha256(winner_rec_path) != winner_result.get("final_rec_sha256")
                or tensor_sha256(winner_sources) != winner_result.get("sources_sha256")):
            raise TaskError("publicly selected CoRA endpoint/source hashes changed after the public gate")
        winner_source_rec = torch.load(winner_rec_path, map_location="cpu", weights_only=True)

        if [int(row["global_index"]) for row in private["metadata"]] != list(range(batch * 16, (batch + 1) * 16)):
            raise TaskError(f"private metadata order changed in batch{batch:02d}")
        naive_aligned, naive_match = evaluate_after_fia(naive_sources, naive_source_rec,
                                                        private["truth"])
        cora_aligned, cora_match = evaluate_after_fia(winner_sources, winner_source_rec,
                                                      private["truth"])
        x_parts.append(private["x"])
        truth_parts.append(private["truth"])
        naive_recs.append(naive_aligned.cpu())
        cora_recs.append(cora_aligned.cpu())
        mean = private["x"].new_tensor((.485, .456, .406))[None, :, None, None]
        std = private["x"].new_tensor((.229, .224, .225))[None, :, None, None]
        original_rgb_parts.append((private["x"] * std + mean).clamp(0, 1))
        metadata_all.extend(private["metadata"])
        correspondence_naive.extend(naive_match)
        correspondence_cora.extend(cora_match)
        selection_history[f"batch{batch:02d}"] = _selection_history(cell)
        cell_summary = read_json(cell / "cell_summary.json")
        batch_provenance.append({"batch_index": batch, "selected_candidate_index": winner,
                                 "naive_sources_sha256": naive_record["sources_sha256"],
                                 "naive_rec_sha256": naive_record["rec_file_sha256"],
                                 "cora_sources_sha256": winner_result["sources_sha256"],
                                 "cora_rec_sha256": winner_result["final_rec_sha256"],
                                 "naive_fia_wall_seconds": naive_record["wall_seconds"],
                                 "naive_cpa_wall_seconds": read_json(naive_dir / "cpa_result.json")["wall_seconds"],
                                 "naive_cpa_peak_cuda_allocated_bytes": read_json(naive_dir / "cpa_result.json")["peak_cuda_allocated_bytes"],
                                 "naive_fia_peak_cuda_allocated_bytes": naive_record["peak_cuda_allocated_bytes"],
                                 "cora_fia_wall_seconds": winner_result["wall_seconds"],
                                 "cora_fia_peak_cuda_allocated_bytes": winner_result["peak_cuda_allocated_bytes"],
                                 "naive_cpa_attempts": read_json(naive_dir / "cpa_attempts.json").get("attempts", []),
                                 "naive_fia_attempts": read_json(naive_dir / "fia/naive_fia_attempts.json").get("attempts", []),
                                 "evaluation_input_sha256": registration["identity"]["source_batches"][batch]["input_sha256"],
                                 "public_storage_bytes_at_cell_completion": cell_summary["storage_bytes_at_cell_completion"],
                                 "available_disk_bytes_at_cell_completion": cell_summary["available_disk_bytes_at_cell_completion"],
                                 "evaluation_seconds": time.perf_counter() - started})

    x_all = torch.cat(x_parts)
    naive_all = torch.cat(naive_recs)
    cora_all = torch.cat(cora_recs)
    original_rgb = torch.cat(original_rgb_parts)
    naive_metrics = evaluate_images(x_true=x_all, rec=naive_all.to(device), device=device)
    cora_metrics = evaluate_images(x_true=x_all, rec=cora_all.to(device), device=device)
    rows = []
    for index in range(80):
        metadata = metadata_all[index]
        if int(metadata["global_index"]) != index:
            raise TaskError("report rows are not in original global image order")
        base = {key: metadata[key] for key in ("global_index", "batch_index", "source_index", "dataset_index", "label", "image_path")}
        row = {**base, "bits": bits,
               "selected_candidate_index": selection_history[f"batch{index // 16:02d}"]["selected_candidate_index"],
               "naive_estimated_source_index": int(correspondence_naive[index]["estimated_source_index"]),
               "naive_source_abs_cosine": float(correspondence_naive[index]["source_abs_cosine"]),
               "naive_lpips": float(naive_metrics.lpips[index]),
               "naive_psnr": float(naive_metrics.psnr[index]),
               "naive_ssim": float(naive_metrics.ssim[index]),
               "cora_estimated_source_index": int(correspondence_cora[index]["estimated_source_index"]),
               "cora_source_abs_cosine": float(correspondence_cora[index]["source_abs_cosine"]),
               "cora_lpips": float(cora_metrics.lpips[index]),
               "cora_psnr": float(cora_metrics.psnr[index]),
               "cora_ssim": float(cora_metrics.ssim[index]),
               "same_correspondence_applied_to_all_image_metrics": True,
               "private_values_used_for_selection": False}
        rows.append(row)
    batches = []
    for batch in range(N_BATCHES):
        subset = rows[batch * 16:(batch + 1) * 16]
        batches.append({"bits": bits, "batch_index": batch, "n_images": 16,
                        **_metric_summary(subset, "naive"), **_metric_summary(subset, "cora"),
                        "selected_candidate_index": int(subset[0]["selected_candidate_index"])})

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_image_metrics.csv", rows)
    _write_csv(output / "per_batch_metrics.csv", batches)
    write_json(output / "selection_history.json", selection_history)
    from .same80_bits_r100_core import write_tensor
    write_tensor(output / "naive_all80_rec.pt", naive_all)
    write_tensor(output / "cora_all80_rec.pt", cora_all)
    figures = {}
    figures["all80"] = render_comparison(original_rgb, naive_all, cora_all, rows,
        output / "all80_original_naive_cora.png", title=f"Signed quantized CoRA bits{bits} | all80 public-selected comparison")
    for batch in range(N_BATCHES):
        sl = slice(batch * 16, (batch + 1) * 16)
        figures[f"batch{batch:02d}"] = render_comparison(original_rgb[sl], naive_all[sl], cora_all[sl],
            rows[sl], output / f"batch{batch:02d}_original_naive_cora.png",
            title=f"Signed quantized CoRA bits{bits} | original batch{batch:02d}")
    means = {**_metric_summary(rows, "naive"), **_metric_summary(rows, "cora")}
    resource_measurement_path = public_root / "resource_measurement.json"
    if not resource_measurement_path.is_file():
        raise TaskError("public first-candidate resource measurement is missing")
    summary = {"task_id": TASK_ID, "plan_version": registration["plan_version"],
               "status": "PRIVATE_EVALUATION_COMPLETED", "bits": bits, "n_images": 80,
               "batch_indices": list(range(N_BATCHES)), "metrics": means,
               "public_gate_sha256": registration["public_gate_sha256"],
               "config_sha256": registration["identity"]["config_sha256"],
               "checkpoint_sha256": CHECKPOINT_SHA256,
               "resource_measurement": read_json(resource_measurement_path),
               "per_batch_metrics": batches, "batch_provenance": batch_provenance,
               "selection_history": selection_history, "figures": figures,
               "search_budget_disclosure": "CoRA selects one public winner after 100 random candidates per bit/batch; Naive runs one center candidate. Search budgets differ.",
               "storage_disclosure": "Logical code width is the reported bit setting; physical packet storage remains int8 and no packed-bandwidth claim is made.",
               "private_evaluation_after_public_freeze": True,
               "same_hungarian_permutation_used_for_psnr_ssim_lpips": True,
               "private_values_used_for_selection": False,
               "completed_at_utc": utc_now()}
    report = [
        f"# Same80 signed quantized CoRA: {bits}-bit public-selected comparison", "",
        "This report evaluates all 80 original images in their original order and five independent B16 batches. Public selection was frozen before this private evaluation.", "",
        "| Arm | Images | LPIPS (lower) | PSNR dB (higher) | SSIM (higher) | Mean source |abs cosine| |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Naive observed-code center | 80 | {means['naive_lpips_mean']:.6f} | {means['naive_psnr_mean']:.3f} | {means['naive_ssim_mean']:.4f} | {means['naive_source_abs_cosine_mean']:.6f} |",
        f"| Publicly selected CoRA | 80 | {means['cora_lpips_mean']:.6f} | {means['cora_psnr_mean']:.3f} | {means['cora_ssim_mean']:.4f} | {means['cora_source_abs_cosine_mean']:.6f} |", "",
        "The CoRA row uses the one 16-image candidate selected by the public 25,000-step score for each B16 batch. No per-image best candidate or private metric entered selection. The Naive arm uses the dequantized observed codes and one separate center CPA/FIA trajectory per batch.", "",
        "The search budgets differ: CoRA screens 100 random candidates and continues the best public top three per batch; Naive evaluates one center candidate. Logical bit width describes the signed quantization codes; the stored packet uses int8 and is not a packed-bandwidth claim.", "",
        "The 5,000-step stage is reported separately through the public feature-cosine-plus-TV screening scores and complete 100-candidate ranking history. The image metrics above apply only to the selected 25,000-step endpoints.", "",
        "After the public freeze, source-absolute-cosine Hungarian correspondence against `Z_evaluation_only` was applied once to each arm, and that same permutation was used for PSNR, SSIM and VGG LPIPS. No source scaling was applied.", "",
        "| Original B16 batch | Naive LPIPS | Naive PSNR | Naive SSIM | CoRA LPIPS | CoRA PSNR | CoRA SSIM | CoRA candidate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[f"| {row['batch_index']} | {row['naive_lpips_mean']:.6f} | {row['naive_psnr_mean']:.3f} | {row['naive_ssim_mean']:.4f} | {row['cora_lpips_mean']:.6f} | {row['cora_psnr_mean']:.3f} | {row['cora_ssim_mean']:.4f} | {row['selected_candidate_index']:03d} |" for row in batches], "",
        "![All original images, Naive reconstructions and public CoRA reconstructions](all80_original_naive_cora.png)", "",
        *[f"![Batch {batch:02d} comparison](batch{batch:02d}_original_naive_cora.png)" for batch in range(N_BATCHES)], "",
        "The all-80 mean row is in `summary.csv`; per-image correspondence and metrics are in `per_image_metrics.csv`; five-batch means are in `per_batch_metrics.csv`; full 100-candidate public ranking histories, seeds, hashes, attempts, timings and CUDA peaks are in `selection_history.json`. Reconstruction tensors retain original global image order.", "",
    ]
    _write_text(output / "report.md", "\n".join(report))
    _write_csv(output / "summary.csv", [{"bits": bits, "n_images": 80, **means,
        "selected_batches": N_BATCHES, "random_candidates_per_batch": 100,
        "selected_finals_per_batch": 3, "naive_candidates_per_batch": 1,
        "physical_code_storage": "int8", "packed_bandwidth_claim": False}])
    names = ["summary.csv", "per_image_metrics.csv", "per_batch_metrics.csv", "selection_history.json",
             "naive_all80_rec.pt", "cora_all80_rec.pt", "report.md",
             "all80_original_naive_cora.png",
             *[f"batch{batch:02d}_original_naive_cora.png" for batch in range(N_BATCHES)]]
    summary["output_hashes"] = {name: file_sha256(output / name) for name in names}
    write_json(output / "summary.json", summary)
    return summary


def run_report(config_path: Path, cfg: dict[str, Any], public_root: Path = PUBLIC_ROOT,
               private_root: Path = PRIVATE_ROOT) -> dict[str, Any]:
    _validate_config(cfg)
    public_root, private_root = Path(public_root).resolve(), Path(private_root).resolve()
    gate(public_root, config_path, cfg)
    registration = read_json(public_root / "registration.json")
    private_manifest = read_json(private_root / "private_manifest.json")
    if (private_manifest.get("task_id") != TASK_ID
            or private_manifest.get("private_values_loaded_by_attack_process") is not False
            or private_manifest.get("source_input_manifest_sha256")
               != registration["identity"]["source_input_manifest_sha256"]):
        raise TaskError("post-gate private evaluation provenance does not match the frozen run")
    if file_sha256(SAME80_IMAGE_MANIFEST) != registration["identity"]["same80_image_manifest_sha256"]:
        raise TaskError("original same80 identity manifest changed before evaluation")
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TaskError("registered evaluation device CUDA is unavailable")
    output_root = private_root / "evaluation"
    ensure_free_space(private_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    with execution_lock(output_root):
        for bits in BITS:
            summaries.append(_evaluate_bit(public_root, registration, bits, device,
                                           output_root / f"bit{bits:02d}"))
        index = {"task_id": TASK_ID, "status": "PRIVATE_EVALUATION_COMPLETED",
                 "public_gate_sha256": registration["public_gate_sha256"],
                 "bits": [item["bits"] for item in summaries],
                 "reports": [{"bits": item["bits"], "directory": f"bit{item['bits']:02d}",
                              "summary_sha256": file_sha256(output_root / f"bit{item['bits']:02d}/summary.json"),
                              "metrics": item["metrics"]} for item in summaries],
                 "private_evaluation_started_after_public_freeze": True,
                 "completed_at_utc": utc_now()}
        write_json(output_root / "evaluation_index.json", index)
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    cfg = read_json(config_path)
    result = run_report(config_path, cfg,
                        _resolve_project_path(cfg["public_root"]),
                        _resolve_project_path(cfg["private_root"]))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
