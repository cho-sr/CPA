"""Verify and report Top-k 50% sparse CPA + FIA on the original five historical batches.

This CPU-only report reads all 80 original M6 inputs and completed attack
artifacts. It never selects images using reconstruction scores or retrains.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from .report_m10_packed4 import (
    REQUIRED_OUTPUTS as PACKED4_REQUIRED_OUTPUTS, _append_run_once, _distribution, _font,
    _metrics, _rgb, file_hash, read_csv, read_json, write_csv, write_json,
)

from .sparsification_codec import CODEC_ID, packet_info

REQUIRED_OUTPUTS = (PACKED4_REQUIRED_OUTPUTS - {"quantization_diagnostics.json"}) | {"sparsification_diagnostics.json", "packet_manifest.json"}

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_OUTPUT = "results/m6_uia_fp32_B16_fia"
SELECTION_LABEL = "All 80 historical images; original five n=16, T=10 batches"
IDENTITY_FIELDS = ("global_index", "batch_index", "source_index", "dataset_index", "label", "image_path")


def _communication(folder: Path, batch: int) -> dict:
    recorded = read_json(folder / "communication.json")
    packet_path = folder / "observation.bin"
    info = packet_info(packet_path.read_bytes())
    expected = {
        "codec": CODEC_ID, "value_bits": 32, "keep_fraction": .5,
        "shape": [4096, 25088], "n_values": 102760448, "kept_elements": 51380224,
        "fp32_payload_bytes": 411041792, "fp32_packet_bytes": 411041824,
        "value_bytes": 205520896, "bitmap_bytes": 12845056,
        "header_bytes": 32, "packet_bytes": 218365984,
        "scope": "attacked classifier.0.weight packet only", "quantization": False,
        "error_feedback": False, "source_reconstruction_only_from_packet": True,
    }
    if any(recorded.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Sparse packet communication declaration changed")
    if info["packet_bytes"] != expected["packet_bytes"] or info["shape"] != expected["shape"]:
        raise RuntimeError("Sparse packet format differs from the registered experiment")
    if recorded["sha256"] != file_hash(packet_path):
        raise RuntimeError("Sparse packet hash differs from communication record")
    ratio = expected["fp32_packet_bytes"] / expected["packet_bytes"]
    if not np.isclose(recorded["compression_ratio"], ratio) or not np.isclose(recorded["saved_percent"], 100 * (1 - 1 / ratio)):
        raise RuntimeError("Sparse packet savings calculation is inconsistent")
    diagnostics = read_json(folder / "sparsification_diagnostics.json")
    if (diagnostics["retained_fp32_bits_exact"] is not True or diagnostics["discarded_values_zero"] is not True
            or diagnostics["kept_elements"] != expected["kept_elements"]
            or diagnostics["minimum_retained_abs"] < diagnostics["maximum_discarded_abs"]):
        raise RuntimeError("Sparse codec audit failed")
    return {"batch_index": batch, **recorded}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT / path


def _same_identity(actual: dict[str, Any], expected: dict[str, Any], context: str) -> None:
    for key in IDENTITY_FIELDS:
        if key not in actual or key not in expected or str(actual[key]) != str(expected[key]):
            raise RuntimeError(f"Historical image identity differs in {context}: {key}")


def _images(tensor: torch.Tensor, n: int, context: str) -> None:
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (n, 3, 224, 224):
        raise RuntimeError(f"Expected all {n} native-resolution RGB images: {context}")
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"Non-finite image tensors: {context}")


def render_grid(originals: torch.Tensor, recs: torch.Tensor, rows: list[dict[str, Any]],
                path: Path, *, columns: int = 5) -> dict[str, Any]:
    """Show every supplied image in original batch/source order, without resizing."""
    _images(originals, len(rows), "grid originals")
    _images(recs, len(rows), "grid reconstructions")
    tile, gap, header, labels = 224, 16, 132, 62
    block = 2 * tile + gap
    size = (gap + columns * (block + gap),
            header + ((len(rows) + columns - 1) // columns) * (tile + labels + gap))
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    batch_ids = sorted({int(row["batch_index"]) for row in rows})
    coverage = "all 80 historical images" if len(rows) == 80 else f"batch {batch_ids[0]} | all 16 original images"
    draw.text((gap, 12), f"TOP-K 50% (FP32) | CPA + FIA | {coverage}", font=_font(28), fill="#0f172a")
    draw.text((gap, 51), "Original historical images and cached client updates; no score-based selection", font=_font(23), fill="#334155")
    draw.text((gap, 89), "VGG16 round 0 | n=16, T=10 per batch | original / reconstruction | original batch/source order", font=_font(22), fill="#475569")
    placements = []
    for index, row in enumerate(rows):
        left = gap + (index % columns) * (block + gap)
        top = header + (index // columns) * (tile + labels + gap)
        global_id, batch_id, source_id = (int(row[key]) for key in IDENTITY_FIELDS[:3])
        label = f"g{global_id:02d} b{batch_id}/s{source_id:02d} | LPIPS {float(row['lpips']):.3f} | cos {float(row['source_abs_cosine']):.3f}"
        draw.text((left, top), label, font=_font(18), fill="#0f172a")
        for panel, (title, tensors) in enumerate((("Original", originals), ("Top-k 50% + FIA", recs))):
            x, y = left + panel * (tile + gap), top + labels
            draw.text((x, top + 29), title, font=_font(19), fill="#334155")
            pixels = _rgb(tensors[index])
            canvas.paste(Image.fromarray(pixels), (x, y))
            placements.append({"global_index": global_id, "batch_index": batch_id,
                               "source_index": source_id, "panel": title,
                               "box": [x, y, x + tile, y + tile],
                               "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
    canvas.save(path)
    with Image.open(path) as saved:
        for placement in placements:
            actual = hashlib.sha256(np.asarray(saved.crop(tuple(placement["box"]))).tobytes()).hexdigest()
            if actual != placement["rgb_sha256"]:
                raise RuntimeError("Saved PNG does not preserve its native source tile")
    return {"file": path.name, "n_images": len(rows), "size": list(size), "tile_size": tile,
            "selection": SELECTION_LABEL, "batch_indices": batch_ids, "placements": placements,
            "pixel_verification": "all native RGB tiles verified"}


def render_distributions(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    labels = ("LPIPS (lower is better)", "PSNR, dB (higher is better)",
              "SSIM (higher is better)", "Matched signed-source absolute cosine")
    for axis, key, label in zip(axes.flat, ("lpips", "psnr", "ssim", "source_abs_cosine"), labels):
        values = np.sort([float(row[key]) for row in rows])
        axis.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color="#2563eb")
        axis.set(xlabel=label, ylabel="Fraction of all 80 historical images", ylim=(0, 1.04))
        axis.grid(alpha=.2)
    fig.suptitle("Top-k 50% sparse CPA + FIA: all80 empirical distributions\n" + SELECTION_LABEL, fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summarize(cfg: dict[str, Any], manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    """Validate all five completed batches and publish one full80 report/ledger row."""
    from .run_topk50_same80 import validate_config
    validate_config(cfg)
    output = Path(output)
    if (cfg["n"], cfg["n_images"], cfg["T"], cfg["batch_indices"]) != (16, 80, 10, [0, 1, 2, 3, 4]):
        raise ValueError("This report requires the original five n16/T10 batches and all80 images")
    if Path(cfg["source_output"]).as_posix() != SOURCE_OUTPUT:
        raise ValueError("The original M6 input cache is required")
    for key in ("new_training", "new_update_collection", "ls_A", "debias", "mp_truncation",
                "mp_diagnostic", "stage2", "refine_A", "oracle_fills", "fp32_attack", "raw_row_attack"):
        if cfg.get(key) is not False:
            raise ValueError(f"This report does not authorize {key}")
    if manifest.get("run_tag") != cfg["run_tag"]:
        raise RuntimeError("Manifest and report run tags differ")
    identity = manifest["identity"]
    git_sha = manifest.get("git_sha", identity.get("git_sha", ""))
    if re.fullmatch(r"[0-9a-fA-F]{40}", git_sha) is None:
        raise RuntimeError("A real 40-character git SHA is required")
    metadata = manifest["metadata"]
    if len(metadata) != 80 or [int(row["global_index"]) for row in metadata] != list(range(80)):
        raise RuntimeError("The input manifest must retain all80 original identities in order")
    frozen_inputs = {resolve(name).resolve(): expected for name, expected in manifest["inputs_sha256"].items()}
    for path, expected in frozen_inputs.items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Frozen report input changed: {path}")
    source = resolve(cfg["source_output"])
    if any((source / f"batch{batch:02d}/inputs.pt").resolve() not in frozen_inputs for batch in range(5)):
        raise RuntimeError("All five original input caches must have frozen hashes")

    rows, batches, communications, diagnostics, originals, recs = [], [], [], [], [], []
    for batch in cfg["batch_indices"]:
        folder = output / f"batch{batch:02d}"
        done = read_json(folder / "summary.json")
        if (done.get("identity") != identity or done.get("n_images") != 16 or done.get("n") != 16
                or done.get("T") != 10 or done.get("batch_index") != batch
                or done.get("oracle_fills") != 0 or done.get("ls_A") is not False):
            raise RuntimeError("Completed batch does not have the full registered input and identity")
        if not REQUIRED_OUTPUTS.issubset(done["output_hashes"]):
            raise RuntimeError("Required completed batch artifacts are missing from its hashes")
        for name, expected in done["output_hashes"].items():
            if file_hash(folder / name) != expected:
                raise RuntimeError(f"Completed batch artifact changed: {folder / name}")
        batch_rows = read_csv(folder / "per_image_metrics.csv")
        if [int(row["global_index"]) for row in batch_rows] != list(range(batch * 16, (batch + 1) * 16)):
            raise RuntimeError("Original image coverage is incomplete, duplicated, or reordered")
        order = [int(row["estimated_source_index"]) for row in batch_rows]
        if sorted(order) != list(range(16)):
            raise RuntimeError("Evaluation correspondence must retain every estimated source")
        inputs = torch.load(source / f"batch{batch:02d}/inputs.pt", map_location="cpu", weights_only=False, mmap=True)
        if len(inputs["metadata"]) != 16:
            raise RuntimeError("Original cached batch must retain all16 image identities")
        for index, (row, original_row) in enumerate(zip(batch_rows, inputs["metadata"])):
            _same_identity(row, original_row, f"batch{batch} reconstructed row{index}")
            _same_identity(original_row, metadata[batch * 16 + index], f"batch{batch} frozen row{index}")
            if row["run_tag"] != cfg["run_tag"] or int(row["batch_index"]) != batch or int(row["source_index"]) != index:
                raise RuntimeError("Per-image run and batch identities disagree")
            if int(row["n"]) != 16 or int(row["T"]) != 10:
                raise RuntimeError("Per-image client schedule differs from n16/T10")
            for flag in ("oracle_fill", "ls_A"):
                if str(row.get(flag, "False")).lower() not in ("false", "0"):
                    raise RuntimeError(f"Per-image {flag} is outside this experiment")
        measured = _metrics(batch_rows)
        for key in ("lpips_mean", "psnr_mean", "ssim_mean", "embedding_cos09"):
            if not np.isclose(measured[key], done[key], rtol=1e-7, atol=1e-9):
                raise RuntimeError(f"Batch summary differs from its per-image {key}")
        reconstruction = torch.load(folder / "rec.pt", map_location="cpu", weights_only=True)
        source_recs = torch.load(folder / "source_order_rec.pt", map_location="cpu", weights_only=True)
        for tensor, context in ((inputs["x"], "originals"), (reconstruction, "reconstructions"), (source_recs, "source-order reconstructions")):
            _images(tensor, 16, context)
        if not torch.equal(reconstruction, source_recs.index_select(0, torch.tensor(order))):
            raise RuntimeError("Evaluation reconstructions must be the saved FIA images with permutation only")
        original = inputs["x"].detach().cpu()
        mean = original.new_tensor((.485, .456, .406))[None, :, None, None]
        std = original.new_tensor((.229, .224, .225))[None, :, None, None]
        originals.append((original * std + mean).clamp(0, 1))
        recs.append(reconstruction)
        rows.extend(batch_rows)
        communications.append(_communication(folder, batch))
        diagnostics.append({"batch_index": batch, **read_json(folder / "sparsification_diagnostics.json")})
        runtime = done["runtime_sec"]
        if not runtime or any(not np.isfinite(float(value)) or float(value) < 0 for value in runtime.values()):
            raise RuntimeError("Completed batch runtimes must be finite and nonnegative")
        batches.append({"batch_index": batch, "n_images": 16, "n": 16, "T": 10,
                        **measured, "runtime_sec": runtime})
        del inputs, source_recs, original
    if len(rows) != 80 or len({row["image_path"] for row in rows}) != 80 or len({int(row["dataset_index"]) for row in rows}) != 80:
        raise RuntimeError("Every historical original must be distinct and evaluated")
    original_images, reconstructed = torch.cat(originals), torch.cat(recs)
    del originals, recs
    metrics, distributions = _metrics(rows), _distribution(rows)
    write_csv(output / "per_image_metrics.csv", rows)
    write_csv(output / "per_batch_metrics.csv", [{**{key: value for key, value in batch.items() if key != "runtime_sec"},
        **{f"runtime_{key}_sec": value for key, value in batch["runtime_sec"].items()}} for batch in batches])
    write_csv(output / "communication.csv", communications)
    write_csv(output / "sparsification_diagnostics.csv", diagnostics)
    write_csv(output / "metric_distributions.csv", distributions)
    torch.save(reconstructed, output / "all80_rec.pt")
    grid = render_grid(original_images, reconstructed, rows, output / "all80_reconstruction_landscape.png")
    batch_grids = [render_grid(original_images[batch * 16:(batch + 1) * 16],
                  reconstructed[batch * 16:(batch + 1) * 16], rows[batch * 16:(batch + 1) * 16],
                  output / f"batch{batch:02d}_reconstruction_landscape.png", columns=4) for batch in range(5)]
    write_json(output / "figure_manifest.json", {"run_tag": cfg["run_tag"], "identity": identity,
               "grid": grid, "batch_grids": batch_grids})
    render_distributions(rows, output / "metric_distributions.png")
    byte_keys = ("fp32_payload_bytes", "fp32_packet_bytes", "value_bytes", "bitmap_bytes", "header_bytes", "packet_bytes")
    communication = {key: sum(value[key] for value in communications) for key in byte_keys}
    communication.update(scope="attacked classifier.0.weight packet only", n_packets=5,
                         packet_compression_ratio=communication["fp32_packet_bytes"] / communication["packet_bytes"],
                         packet_reduction_fraction=1 - communication["packet_bytes"] / communication["fp32_packet_bytes"])
    runtime_keys = set().union(*(batch["runtime_sec"] for batch in batches))
    runtime = {key: sum(float(batch["runtime_sec"].get(key, 0)) for batch in batches) for key in sorted(runtime_keys)}
    report = [
        "# Top-k 50% sparse CPA + FIA: original historical80", "", SELECTION_LABEL + ".", "",
        "This run reuses the exact 80 historical M6/M8 images, in their original order and five 16-image batches. Each batch uses its cached actual (W_initial - W_after10)/lr update from the pretrained VGG16 round-0 classifier. The five independent client updates were already collected; this run does not retrain or regroup the clients. No historical LPIPS scores are used to select images.", "",
        "| Method | Images | LPIPS | PSNR (dB) | SSIM | Source absolute cosine > 0.9 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Top-k 50% sparse + upstream CPA + FIA | 80 | {metrics['lpips_mean']:.6f} | {metrics['psnr_mean']:.3f} | {metrics['ssim_mean']:.4f} | {metrics['embedding_cos09']}/80 |", "",
        "The attacker receives only the zero-filled observation decoded from each real sparse binary packet. Global magnitude Top-k within the attacked FC1 tensor retains exactly 50% of coordinates, breaking cutoff ties by ascending flat index. Selected values remain bit-exact FP32 and are serialized with a row-major bitmap. There is no quantization, rescaling, error feedback or missing-value estimation. All 16 CPA source estimates pass through abs(S_hat) into the original Direct FIA for each batch. Signed-source absolute-cosine Hungarian matching is applied only after FIA for evaluation, without truth scaling or source replacement.", "",
        f"The five attacked first-FC weight packets total **{communication['packet_bytes']:,} bytes**: {communication['value_bytes']:,} FP32 value bytes, {communication['bitmap_bytes']:,} position bitmap bytes and {communication['header_bytes']:,} header bytes. The computed same-shape FP32 packets total {communication['fp32_packet_bytes']:,} bytes, giving **{communication['packet_compression_ratio']:.4f}x** compression and **{100 * communication['packet_reduction_fraction']:.4f}%** fewer packet bytes.",
        "The FP32 byte count is a serialization calculation, not an FP32 attack result. Compression covers classifier.0.weight only. The same80 packed 4-bit run uses the same source updates, CPA/FIA implementation, seeds and matching order; only compression is changed. Historical 32-bit results use a different CPA implementation and cannot isolate compression alone. This condition does not measure full-model traffic, network latency or multi-round training utility.", "",
        "| Original batch | Images | LPIPS | PSNR (dB) | SSIM | Source cosine > 0.9 | Packet bytes |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[f"| {batch['batch_index']} | 16 | {batch['lpips_mean']:.6f} | {batch['psnr_mean']:.3f} | {batch['ssim_mean']:.4f} | {batch['embedding_cos09']}/16 | {comm['packet_bytes']:,} |" for batch, comm in zip(batches, communications)], "",
        f"CPA iterations per batch: {cfg['stage1']['cpa_ica_max_iter']:,}; FIA iterations per batch: {cfg['fia']['n_iter']:,}. Summed batch runtimes in seconds: " + ", ".join(f"{key}={value:.1f}" for key, value in runtime.items()) + ".",
        "All80 images are evaluated. Per-source cosine is logged individually; the >0.9 count is an embedding diagnostic, not an image-success threshold. Image metrics remain directly measured values without historical control normalization.", "",
        "![All80 originals and reconstructions in original batch/source order](all80_reconstruction_landscape.png)", "",
        *[f"[Original batch {batch}: all16 originals and reconstructions](batch{batch:02d}_reconstruction_landscape.png)" for batch in range(5)], "",
        "![All80 empirical metric distributions](metric_distributions.png)", "",
        "Per-image measurements: `per_image_metrics.csv`; per-batch measurements: `per_batch_metrics.csv`; packet accounting per batch: `communication.csv`; codec diagnostics: `sparsification_diagnostics.csv`; distributions: `metric_distributions.csv`. `all80_rec.pt` retains all float reconstructions in original order; `figure_manifest.json` records native tile hashes.", "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    names = ["per_image_metrics.csv", "per_batch_metrics.csv", "communication.csv", "sparsification_diagnostics.csv",
             "metric_distributions.csv", "all80_rec.pt", "all80_reconstruction_landscape.png", "figure_manifest.json",
             "metric_distributions.png", "report.md", *[f"batch{batch:02d}_reconstruction_landscape.png" for batch in range(5)]]
    result = {"run_tag": cfg["run_tag"], "status": "complete", "n_images": 80, "n": 16, "T": 10,
              "git_sha": git_sha, "schema_version": manifest.get("schema_version", cfg.get("schema_version", 1)),
              "config": cfg, "identity": identity, "inputs_sha256": manifest["inputs_sha256"],
              "metrics": metrics, "selection_label": SELECTION_LABEL, "selection": cfg.get("selection"),
              "per_image_metrics": rows, "per_victim_batch": batches,
              "communication": communication, "communication_per_batch": communications,
              "sparsification_diagnostics_per_batch": diagnostics, "metric_distributions": distributions,
              "runtime_sec": runtime, "wall_time_sec": runtime.get("invocation"),
              "oracle_fills": 0, "cpa": True, "ls_A": False, "debias": False, "mp_truncation": False,
              "stage2": False, "quantization": False, "keep_fraction": .5, "new_training": False, "new_update_collection": False,
              "diagnostics_not_applicable": ["MP noise model", "Stage2 feasibility", "gamma", "LS mixing matrix"],
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "output_hashes": {name: file_hash(output / name) for name in names}}
    _append_run_once(resolve(cfg.get("results_jsonl", "results/runs.jsonl")), result)
    write_json(output / "summary.json", result)
    write_json(output / "execution_status.json", {"run_tag": cfg["run_tag"], "state": "complete", "n_images": 80,
               "metrics": metrics, "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    return result
