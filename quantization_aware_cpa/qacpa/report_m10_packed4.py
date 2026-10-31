"""Report the selected best10 actual packed4 CPA + FIA experiment.

Reporting reads completed CPU artifacts only. Historical M8 scores select the
input sample; they are never used as a current FP32 or attack-control baseline.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

PROJECT = Path(__file__).resolve().parents[1]
SELECTION_LABEL = "Historical M8 lowest-LPIPS 10 of 80; selected exploratory sample"
REQUIRED_OUTPUTS = {
    "sources.pt", "source_order_rec.pt", "rec.pt", "per_image_metrics.csv",
    "cpa_log.json", "fia_log.json", "communication.json",
    "quantization_diagnostics.json", "observation.bin",
}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT / path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("A completed report requires nonempty tables")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = {key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
              for key in ("lpips", "psnr", "ssim", "source_abs_cosine")}
    if any(not np.isfinite(value).all() for value in values.values()):
        raise RuntimeError("Non-finite per-image measurements")
    cosines = values["source_abs_cosine"]
    if np.any(cosines < 0) or np.any(cosines > 1 + 1e-5):
        raise RuntimeError("Absolute source cosine is outside its numerical range")
    return {
        **{f"{key}_mean": float(value.mean()) for key, value in values.items()},
        "embedding_cos09": int((cosines > .9).sum()),
        "source_cos_per_source": cosines.tolist(),
    }


def _distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for key in ("lpips", "psnr", "ssim", "source_abs_cosine"):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result.append({"metric": key, "n": len(values), "mean": float(values.mean()),
                       "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
                       "min": float(values.min()), "p25": float(np.quantile(values, .25)),
                       "median": float(np.median(values)), "p75": float(np.quantile(values, .75)),
                       "max": float(values.max())})
    return result


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _rgb(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    if tuple(tensor.shape) != (3, 224, 224) or not bool(torch.isfinite(tensor).all()):
        raise RuntimeError("Expected finite native-resolution RGB tensors")
    return (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def render_grid(originals: torch.Tensor, recs: torch.Tensor, rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    """Display every selected input; preserve and check every 224x224 tile."""
    n_images, columns = len(rows), 5
    tile, gap, header, labels = 224, 16, 132, 62
    block = 2 * tile + gap
    size = (gap + columns * (block + gap), header + ((n_images + columns - 1) // columns) * (tile + labels + gap))
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 12), f"ACTUAL PACKED4 | CPA + FIA | all {n_images} selected images", font=_font(28), fill="#0f172a")
    draw.text((gap, 51), SELECTION_LABEL, font=_font(23), fill="#334155")
    draw.text((gap, 89), "VGG16 | n=10, T=10 | one rebuilt client update | original / reconstruction | local source order", font=_font(22), fill="#475569")
    placements = []
    for index, row in enumerate(rows):
        left = gap + (index % columns) * (block + gap)
        top = header + (index // columns) * (tile + labels + gap)
        historical = row.get("historical_global_index", row.get("original_global_index", "?"))
        draw.text((left, top), f"i{index:02d} (M8 g{historical}) | LPIPS {float(row['lpips']):.3f} | cos {float(row['source_abs_cosine']):.3f}", font=_font(19), fill="#0f172a")
        for panel, (label, tensors) in enumerate((("Original", originals), ("Packed4 CPA + FIA", recs))):
            x, y = left + panel * (tile + gap), top + labels
            draw.text((x, top + 29), label, font=_font(19), fill="#334155")
            pixels = _rgb(tensors[index])
            canvas.paste(Image.fromarray(pixels), (x, y))
            placements.append({"global_index": index, "historical_global_index": historical,
                               "panel": label, "box": [x, y, x + tile, y + tile],
                               "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
    canvas.save(path)
    with Image.open(path) as saved:
        for placement in placements:
            actual = hashlib.sha256(np.asarray(saved.crop(tuple(placement["box"]))).tobytes()).hexdigest()
            if actual != placement["rgb_sha256"]:
                raise RuntimeError("Saved PNG does not preserve its native source tile")
    return {"file": path.name, "size": list(size), "tile_size": tile, "selection": SELECTION_LABEL,
            "placements": placements, "pixel_verification": "all native RGB tiles verified"}


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
        axis.scatter(values, np.arange(1, len(values) + 1) / len(values), color="#2563eb", s=22)
        axis.set(xlabel=label, ylabel="Fraction of selected images", ylim=(0, 1.04))
        axis.grid(alpha=.2)
    fig.suptitle("Actual packed4 CPA + FIA: selected best10 empirical distributions\n" + SELECTION_LABEL, fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _communication(folder: Path, batch: int) -> dict[str, Any]:
    value = read_json(folder / "communication.json")
    fields = ("fp32_payload_bytes", "fp32_packet_bytes", "packed_payload_bytes", "scale_bytes", "header_bytes", "packet_bytes", "n_values")
    for key in fields:
        if key not in value or isinstance(value[key], bool) or int(value[key]) != value[key] or value[key] < 0:
            raise RuntimeError(f"Missing or invalid communication byte count: {key}")
    if value["packet_bytes"] != (folder / "observation.bin").stat().st_size:
        raise RuntimeError("Reported packet length differs from the actual binary file")
    if value["packet_bytes"] != value["packed_payload_bytes"] + value["scale_bytes"] + value["header_bytes"]:
        raise RuntimeError("Packet byte components do not add up")
    if value["fp32_payload_bytes"] != 4 * value["n_values"] or value["packed_payload_bytes"] != (value["n_values"] + 1) // 2:
        raise RuntimeError("Payload is not two four-bit codes per byte against FP32")
    if value["fp32_packet_bytes"] != value["fp32_payload_bytes"] + value["header_bytes"]:
        raise RuntimeError("FP32 packet accounting must include its same-format header")
    if not 0 < value["packet_bytes"] < value["fp32_payload_bytes"]:
        raise RuntimeError("Measured packet does not reduce FP32 payload bytes")
    return {"batch_index": batch, **value,
            "packet_compression_ratio": value["fp32_packet_bytes"] / value["packet_bytes"],
            "packet_reduction_fraction": 1 - value["packet_bytes"] / value["fp32_packet_bytes"],
            "fp32_data_to_packet_ratio": value["fp32_payload_bytes"] / value["packet_bytes"]}


def _append_run_once(path: Path, result: dict[str, Any]) -> None:
    """A report retry must not append the same completed run a second time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("run_tag") == result["run_tag"]:
                        existing.append(row)
    if existing:
        if len(existing) != 1 or existing[0].get("identity") != result["identity"]:
            raise RuntimeError("Run registry already contains a duplicate or conflicting run identity")
        for key in ("metrics", "per_victim_batch", "communication"):
            if existing[0].get(key) != result[key]:
                raise RuntimeError(f"Completed run registry disagrees with regenerated {key}")
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def summarize(cfg: dict[str, Any], manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    """Verify and report the single selected best10 run, without other arms."""
    output = Path(output)
    n, n_images = int(cfg["n"]), int(cfg["n_images"])
    if (n, n_images, int(cfg["T"]), cfg["batch_indices"]) != (10, 10, 10, [0]):
        raise ValueError("Current authorization is one n10/T10 packed4 CPA + FIA run")
    git_sha = manifest.get("git_sha", manifest["identity"].get("git_sha", ""))
    if re.fullmatch(r"[0-9a-fA-F]{40}", git_sha) is None:
        raise RuntimeError("A real 40-character git SHA is required")
    for name, expected in manifest["inputs_sha256"].items():
        if file_hash(resolve(name)) != expected:
            raise RuntimeError(f"Frozen report input changed: {name}")
    rows, batches, communications, diagnostics = [], [], [], []
    for batch in cfg["batch_indices"]:
        folder = output / f"batch{batch:02d}"
        done = read_json(folder / "summary.json")
        if done["identity"] != manifest["identity"] or done["n_images"] != n or done["batch_index"] != batch or done["oracle_fills"]:
            raise RuntimeError("Completed batch does not have the full registered input and identity")
        if not REQUIRED_OUTPUTS.issubset(done["output_hashes"]):
            raise RuntimeError("Required completed batch artifacts are missing from its hashes")
        for name, expected in done["output_hashes"].items():
            if file_hash(folder / name) != expected:
                raise RuntimeError(f"Completed batch artifact changed: {folder / name}")
        batch_rows = read_csv(folder / "per_image_metrics.csv")
        if [int(row["global_index"]) for row in batch_rows] != list(range(batch * n, (batch + 1) * n)):
            raise RuntimeError("Selected image coverage is incomplete or duplicated")
        if sorted(int(row["estimated_source_index"]) for row in batch_rows) != list(range(n)):
            raise RuntimeError("Evaluation correspondence must retain every estimated source")
        for index, row in enumerate(batch_rows):
            if row["run_tag"] != cfg["run_tag"] or int(row["batch_index"]) != batch or int(row["source_index"]) != index:
                raise RuntimeError("Per-image run and batch identities disagree")
            if str(row.get("oracle_fill", "False")).lower() not in ("false", "0"):
                raise RuntimeError("Oracle-filled reconstructions are outside this experiment")
        measured = _metrics(batch_rows)
        for key in ("lpips_mean", "psnr_mean", "ssim_mean", "embedding_cos09"):
            if not np.isclose(measured[key], done[key], rtol=1e-7, atol=1e-9):
                raise RuntimeError(f"Batch summary differs from its per-image {key}")
        rows.extend(batch_rows)
        communications.append(_communication(folder, batch))
        diagnostics.append({"batch_index": batch, **read_json(folder / "quantization_diagnostics.json")})
        batches.append({"batch_index": batch, "n_images": n, **measured, "runtime_sec": done["runtime_sec"]})
    if len(rows) != n_images or len({row["image_path"] for row in rows}) != n_images:
        raise RuntimeError("Every selected original must be distinct and evaluated")
    originals, recs = [], []
    for batch in cfg["batch_indices"]:
        inputs = torch.load(resolve(cfg["source_output"]) / f"batch{batch:02d}/inputs.pt", map_location="cpu", weights_only=False, mmap=True)
        metadata = inputs["metadata"]
        if len(metadata) != n:
            raise RuntimeError("Selected source metadata must cover every original")
        for row, original_row in zip(rows[batch * n:(batch + 1) * n], metadata):
            for key in ("global_index", "batch_index", "source_index", "dataset_index", "label", "image_path"):
                if str(row[key]) != str(original_row[key]):
                    raise RuntimeError(f"Reconstruction and original identities disagree: {key}")
            historical = original_row.get("historical_global_index", original_row.get("original_global_index"))
            if historical is None or not 0 <= int(historical) < 80:
                raise RuntimeError("Original M8 selection identity is required for each selected image")
            if "historical_global_index" in row and int(row["historical_global_index"]) != int(historical):
                raise RuntimeError("Historical selection index disagrees with the original metadata")
            row["historical_global_index"] = int(historical)
        original = inputs["x"].detach().cpu()
        mean = original.new_tensor((.485, .456, .406))[None, :, None, None]
        std = original.new_tensor((.229, .224, .225))[None, :, None, None]
        originals.append((original * std + mean).clamp(0, 1))
        recs.append(torch.load(output / f"batch{batch:02d}/rec.pt", map_location="cpu", weights_only=True))
    original_images, reconstructed = torch.cat(originals), torch.cat(recs)
    if len({row["historical_global_index"] for row in rows}) != n_images:
        raise RuntimeError("Historical selection contains a duplicate original")
    for images in (original_images, reconstructed):
        if tuple(images.shape) != (n_images, 3, 224, 224) or not bool(torch.isfinite(images).all()):
            raise RuntimeError("All selected native-resolution images must be present")
    metrics = _metrics(rows)
    distributions = _distribution(rows)
    write_csv(output / "per_image_metrics.csv", rows)
    write_csv(output / "per_batch_metrics.csv", [{**{key: value for key, value in batch.items() if key != "runtime_sec"},
        **{f"runtime_{key}_sec": value for key, value in batch["runtime_sec"].items()}} for batch in batches])
    write_csv(output / "communication.csv", communications)
    write_csv(output / "quantization_diagnostics.csv", diagnostics)
    write_csv(output / "metric_distributions.csv", distributions)
    torch.save(reconstructed, output / "all10_rec.pt")
    grid = render_grid(original_images, reconstructed, rows, output / "all10_reconstruction_landscape.png")
    write_json(output / "figure_manifest.json", {"run_tag": cfg["run_tag"], "identity": manifest["identity"], "grid": grid})
    render_distributions(rows, output / "metric_distributions.png")
    byte_keys = ("fp32_payload_bytes", "fp32_packet_bytes", "packed_payload_bytes", "scale_bytes", "header_bytes", "packet_bytes")
    communication = {key: sum(value[key] for value in communications) for key in byte_keys}
    communication.update(scope="attacked classifier.0.weight packet only",
                         packet_compression_ratio=communication["fp32_packet_bytes"] / communication["packet_bytes"],
                         packet_reduction_fraction=1 - communication["packet_bytes"] / communication["fp32_packet_bytes"])
    runtime_keys = set().union(*(batch["runtime_sec"] for batch in batches))
    runtime = {key: sum(float(batch["runtime_sec"].get(key, 0)) for batch in batches) for key in sorted(runtime_keys)}
    report = [
        "# Actual packed4 CPA + FIA: selected best10", "", SELECTION_LABEL + ".", "",
        "The input data were selected using historical M8 LPIPS before this run. The ten cached image features formed one rebuilt n=10, T=10 client update from the pretrained VGG16 round-0 classifier. These measurements describe this selected exploratory sample; they do not estimate reconstruction quality over the original 80 images.", "",
        "| Method | Images | LPIPS | PSNR (dB) | SSIM | Source absolute cosine > 0.9 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Actual packed4 + upstream CPA + FIA | {n_images} | {metrics['lpips_mean']:.6f} | {metrics['psnr_mean']:.3f} | {metrics['ssim_mean']:.4f} | {metrics['embedding_cos09']}/{n_images} |", "",
        "The observation is decoded from the saved binary packet: per-row stochastic 16-level integer quantization, two four-bit codes per byte, and decoder-visible FP32 radii. The attacker receives the decoded packet observation. Source estimates pass through abs(S_hat) into the original Direct FIA; Hungarian absolute-cosine correspondence is applied after FIA for evaluation.", "",
        f"The attacked first-FC weight packet contains **{communication['packet_bytes']:,} bytes**: {communication['packed_payload_bytes']:,} code bytes, {communication['scale_bytes']:,} radius bytes and {communication['header_bytes']:,} header bytes. The same-shape FP32 packet would contain {communication['fp32_packet_bytes']:,} bytes ({communication['fp32_payload_bytes']:,} tensor-data bytes), giving **{communication['packet_compression_ratio']:.4f}x** compression and **{100 * communication['packet_reduction_fraction']:.4f}%** fewer packet bytes.",
        "This byte comparison uses a computed same-shape FP32 serialization size; it is not an additional FP32 attack experiment. Byte savings cover classifier.0.weight only. Full-model traffic, networking latency and multi-round training utility were not measured.", "",
        f"CPA iterations: {cfg['stage1']['cpa_ica_max_iter']:,}; FIA iterations: {cfg['fia']['n_iter']:,}. Batch runtimes in seconds: " + ", ".join(f"{key}={value:.1f}" for key, value in runtime.items()) + ".",
        "All ten selected images are retained. Per-source cosine is reported individually; its >0.9 count is an embedding diagnostic, not an image-success threshold. LPIPS, PSNR and SSIM remain directly measured values without historical control normalization.", "",
        "![All ten selected originals and reconstructions](all10_reconstruction_landscape.png)", "",
        "![Selected-sample empirical metric distributions](metric_distributions.png)", "",
        "Per-image measurements: `per_image_metrics.csv`; per-batch measurements: `per_batch_metrics.csv`; measured packet accounting: `communication.csv`; codec error diagnostics: `quantization_diagnostics.csv`; distribution statistics: `metric_distributions.csv`.", "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    names = ("per_image_metrics.csv", "per_batch_metrics.csv", "communication.csv", "quantization_diagnostics.csv",
             "metric_distributions.csv", "all10_rec.pt", "all10_reconstruction_landscape.png", "figure_manifest.json",
             "metric_distributions.png", "report.md")
    result = {"run_tag": cfg["run_tag"], "status": "complete", "n_images": n_images,
              "git_sha": git_sha, "schema_version": manifest.get("schema_version", cfg.get("schema_version", 1)),
              "config": cfg, "identity": manifest["identity"], "metrics": metrics,
              "selection_label": SELECTION_LABEL, "selection": manifest.get("selection", cfg.get("selection")),
              "per_victim_batch": batches, "communication": communication, "communication_per_batch": communications,
              "quantization_diagnostics_per_batch": diagnostics, "metric_distributions": distributions,
              "runtime_sec": runtime, "wall_time_sec": runtime.get("invocation"),
              "oracle_fills": 0, "cpa": True, "ls_A": False, "debias": False,
              "mp_truncation": False, "stage2": False,
              "diagnostics_not_applicable": ["MP noise model", "Stage2 feasibility", "gamma", "LS mixing matrix"],
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "output_hashes": {name: file_hash(output / name) for name in names}}
    _append_run_once(resolve(cfg.get("results_jsonl", "results/runs.jsonl")), result)
    write_json(output / "summary.json", result)
    write_json(output / "execution_status.json", {"run_tag": cfg["run_tag"], "state": "complete", "n_images": n_images,
               "metrics": metrics, "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    return result
