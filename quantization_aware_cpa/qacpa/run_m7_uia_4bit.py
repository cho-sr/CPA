"""Paired UIA 4-bit replay of five stored n16/T10 actual updates, without training.

Prepare locks code, protocol and all existing inputs before quantization. Run
resumes only intact checkpoints from that identity. MP prediction is diagnostic;
CPA retains all16 axes without debiasing or truncation, followed by unchanged FIA.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch

from .run_m6_uia_fp32_fia import (
    PROJECT, N, BATCHES, actual_observation, file_hash, fia_targets, json_safe,
    load_checkpoint, render_grid, resolve_path, save_checkpoint, utc_now,
    write_csv, write_json,
)
from .fia import denormalize_imagenet, run_direct_fia
from .quantizer import assert_cell_membership, quantize
from .seed import seed_all
from .stage1 import estimate_mixing_and_sources
from .uia import mp_prediction_from_quantized

RUN_TAG = "m7_uia_4bit_B16"
DEFAULT_CONFIG = PROJECT / "configs/m7_uia_4bit_B16.json"
FOLLOWUP_SHA256 = "24a25980e58356591943559e29474bb777494f426b0834ca6d68f76aa2daae67"
CODE_FILES = (
    "qacpa/run_m7_uia_4bit.py", "qacpa/run_m6_uia_fp32_fia.py", "qacpa/stage1.py",
    "qacpa/fia.py", "qacpa/uia.py", "qacpa/quantizer.py", "qacpa/alignment.py",
    "qacpa/seed.py", "qacpa/linalg.py", "scripts/launch_m7_uia_4bit.ps1",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    import csv
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_config(cfg: dict[str, Any]) -> None:
    reference = json.loads(resolve_path(cfg["source_fp32_config"]).read_text(encoding="utf-8"))
    followup_path = resolve_path(cfg["followup_protocol"])
    if file_hash(followup_path) != FOLLOWUP_SHA256:
        raise RuntimeError("Follow-up protocol changed after registration")
    expected_quant = {"enabled": True, "bits": 4, "rounding": "stochastic", "scale_granularity": "per_row",
                      "seed_rule": "0+1000003*batch_index", "seeds": [1000003 * b for b in BATCHES],
                      "implementation": "qacpa.quantizer.quantize: subtractive-dither observation Y+U(-delta/2,delta/2), with inward cell projection; not conventional integer stochastic rounding"}
    checks = {
        "run tag": cfg["run_tag"] == RUN_TAG,
        "same model/data/actual observation/stage1/FIA": all(cfg[key] == reference[key] for key in ("seed", "model", "data", "uia", "stage1", "fia")),
        "4bit quantizer": cfg["quantization"] == expected_quant,
        "no training or oracle fills": cfg["new_training"] is False and cfg["oracle_fills"] is False,
        "no Stage2 or refinement": cfg["stage2"] is False and cfg["refine_A"] is False,
        "full16 ICA": cfg["stage1"]["n_components"] == N and not cfg["stage1"]["debias"] and not cfg["stage1"]["mp_truncation"],
        "diagnostic MP": cfg["prediction"] == {"implementation": "qacpa.uia.mp_prediction_from_quantized", "c_mp": 3.0, "debias_diagnostic_covariance": True, "influences_attack": False},
        "random grid": cfg["grid"] == {"seed": 42, "n": 12, "selection": "numpy.default_rng(42).choice(80,12,replace=False)", "label": "not cherry-picked"},
    }
    failed = [key for key, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"Unsupported UIA4bit protocol changes: {failed}")


def read_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def input_files(cfg: dict[str, Any]) -> dict[str, str]:
    source = resolve_path(cfg["source_fp32_output"])
    paths = [source / "summary.json", source / "per_image_metrics.csv",
             resolve_path(cfg["source_fp32_config"]), resolve_path(cfg["gia_reference_csv"]),
             resolve_path(cfg["followup_protocol"])]
    for batch in BATCHES:
        folder = source / f"batch{batch:02d}"
        paths.extend(folder / name for name in ("inputs.pt", "inputs.meta.json", "observation.pt", "observation.meta.json", "rec.pt", "summary.json"))
    return {str(path): file_hash(path) for path in paths}


def implementation_identity(config_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    weights = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
    return {"protocol_sha256": file_hash(config_path), "followup_sha256": FOLLOWUP_SHA256,
            "code_sha256": {name: file_hash(PROJECT / name) for name in CODE_FILES},
            "weights_sha256": file_hash(weights), "input_sha256": input_files(cfg),
            "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version}}


def verify_source_batch(cfg: dict[str, Any], batch: int) -> dict[str, Any]:
    folder = resolve_path(cfg["source_fp32_output"]) / f"batch{batch:02d}"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    for name in ("inputs", "observation"):
        metadata = json.loads((folder / f"{name}.meta.json").read_text(encoding="utf-8"))
        digest = file_hash(folder / f"{name}.pt")
        if digest != metadata["file_sha256"] or digest != summary["output_hashes"][f"{name}.pt"]:
            raise RuntimeError(f"Stored fp32 {name} fails original integrity gate")
        if metadata["identity"]["batch_index"] != batch:
            raise RuntimeError("Stored fp32 batch identity mismatch")
    if summary["oracle_fills"] != 0 or summary["n_images"] != N or summary["T"] != 10:
        raise RuntimeError("Stored fp32 protocol mismatch")
    inputs = torch.load(folder / "inputs.pt", map_location="cpu", weights_only=False)
    if tuple(inputs["x"].shape) != (N, 3, 224, 224) or tuple(inputs["Z_evaluation_only"].shape) != (N, 25088):
        raise RuntimeError("Stored input shape mismatch")
    if [row["global_index"] for row in inputs["metadata"]] != list(range(batch * N, (batch + 1) * N)):
        raise RuntimeError("Stored input coverage mismatch")
    # mmap avoids copying the oracle diagnostic Y_factor; only the actual
    # transmitted update is accessed. No model or training function is called.
    stored = torch.load(folder / "observation.pt", map_location="cpu", weights_only=False, mmap=True)
    actual_observation(stored["observation"], maximum_relative_error=cfg["uia"]["update_factor_rel_l2_gate"])
    if stored["observation_used_for_attack"] != "Y_delta_over_lr" or stored["observation"]["steps"] != 10:
        raise RuntimeError("Stored observation is not the registered actual T10 update")
    return inputs


def prepare(config_path: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    identity = implementation_identity(config_path, cfg)
    manifest_path = output / "input_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["identity"] != identity:
            raise RuntimeError("Frozen protocol/code/input/weights changed; refusing resume or relock")
        return existing
    source = json.loads((resolve_path(cfg["source_fp32_output"]) / "summary.json").read_text(encoding="utf-8"))
    if source["status"] != "complete" or source["n_images"] != 80 or source["oracle_fills"] != 0:
        raise RuntimeError("Complete pure80 fp32 source required")
    if source["config_sha256"] != file_hash(resolve_path(cfg["source_fp32_config"])):
        raise RuntimeError("Source fp32 config integrity mismatch")
    for name in ("stage1.py", "fia.py", "alignment.py", "seed.py", "uia.py"):
        if source["identity"]["code_sha256"][name] != identity["code_sha256"][f"qacpa/{name}"]:
            raise RuntimeError(f"Paired baseline implementation changed: {name}")
    all_rows = []
    for batch in BATCHES:
        inputs = verify_source_batch(cfg, batch)
        all_rows.extend(inputs["metadata"])
    if len({row["image_path"] for row in all_rows}) != 80:
        raise RuntimeError("Expected80 unique originals")
    manifest = {"run_tag": RUN_TAG, "identity": identity, "frozen_at_utc": utc_now(),
                "phase": "CPU existing-artifact integrity checks only; before new quantization, ICA, FIA or metrics",
                "n_images": 80, "metadata": all_rows,
                "random_grid_global_indices": np.random.default_rng(42).choice(80, 12, replace=False).tolist(),
                "estimated_runtime_sec": source["runtime_sec"]["stage1"] + source["runtime_sec"]["fia"]}
    write_json(manifest_path, manifest)
    write_json(config_path.with_suffix(".lock.json"), {"protocol_id": cfg["protocol_id"], "sha256": identity["protocol_sha256"],
               "frozen_at_utc": manifest["frozen_at_utc"], "input_manifest_sha256": file_hash(manifest_path)})
    update_status(output, state="prepared", protocol_sha256=identity["protocol_sha256"], no_gpu_work=True)
    return manifest


def update_status(output: Path, **kwargs: Any) -> None:
    write_json(output / "execution_status.json", {"run_tag": RUN_TAG, "pid": os.getpid(), "updated_at_utc": utc_now(), **kwargs})


def stage1_call(Y_hat: torch.Tensor, delta: torch.Tensor, cfg: dict[str, Any], batch: int, device: torch.device) -> Any:
    stage = cfg["stage1"]
    seed_all(cfg["seed"] + batch)
    return estimate_mixing_and_sources(Y_hat, delta, n_components=N, rounding="stochastic", debias=False,
        seed=cfg["seed"] + batch, max_iter=1000, tol=1e-4, mp_truncation=False, ica_method="cpa",
        c_mp=stage["c_mp"], whitening_tolerance=stage["whitening_tolerance"], device=device,
        **{key: stage[key] for key in ("cpa_ica_max_iter", "cpa_ica_lr", "cpa_ica_optimizer", "cpa_ica_decor", "cpa_ica_T",
                                      "cpa_ica_ne", "cpa_ica_nv", "cpa_ica_l1", "cpa_ica_tv", "cpa_ica_log_every")})


def quantize_actual(actual: torch.Tensor, batch: int) -> dict[str, Any]:
    started = time.perf_counter()
    quant_seed = 1000003 * batch
    qobs = quantize(actual, bits=4, rounding="stochastic", scale_granularity="per_row",
                    generator=torch.Generator(device="cpu").manual_seed(quant_seed))
    assert_cell_membership(actual, qobs.Y_hat, qobs.delta)
    residual = qobs.Y_hat - actual
    diagnostics = {"quantization_seed": quant_seed, "relative_l2": float(residual.norm() / actual.norm()),
                   "mse": float(residual.square().mean()), "max_abs": float(residual.abs().max()),
                   "empirical_error_mean": float(residual.mean()), "theoretical_mean_variance": float((qobs.delta.square() / 12).mean()),
                   "delta_min": float(qobs.delta.min()), "delta_mean": float(qobs.delta.mean()), "delta_max": float(qobs.delta.max()),
                   "zero_scale_rows": int((qobs.delta == 0).sum()), "cell_membership": True}
    return {"Y_hat": qobs.Y_hat, "delta": qobs.delta, "diagnostics": diagnostics, "wall_time_sec": time.perf_counter() - started}


def validate_completed(folder: Path, identity: dict[str, Any]) -> dict[str, Any]:
    result = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if result["identity"] != identity:
        raise RuntimeError("Completed batch identity mismatch")
    for name, digest in result["output_hashes"].items():
        if file_hash(folder / name) != digest:
            raise RuntimeError(f"Completed output changed: {folder / name}")
    return result


def run_batch(cfg: dict[str, Any], identity: dict[str, Any], batch: int, output: Path, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    folder = output / f"batch{batch:02d}"
    source = resolve_path(cfg["source_fp32_output"]) / f"batch{batch:02d}"
    batch_identity = {**identity, "batch_index": batch}
    if (folder / "summary.json").exists():
        return validate_completed(folder, batch_identity)
    inputs = verify_source_batch(cfg, batch)
    input_identity = {**batch_identity, "stage": "reused_fp32_inputs"}
    cached_inputs = load_checkpoint(folder, "inputs", input_identity)
    if cached_inputs is None:
        save_checkpoint(folder, "inputs", inputs, input_identity)
    update_status(output, state="running", batch_index=batch, stage="quantization_of_stored_actual_update")
    q_identity = {**batch_identity, "stage": "4bit_stochastic_per_row"}
    quantized = load_checkpoint(folder, "quantized", q_identity)
    stored = torch.load(source / "observation.pt", map_location="cpu", weights_only=False, mmap=True)
    actual = actual_observation(stored["observation"], maximum_relative_error=cfg["uia"]["update_factor_rel_l2_gate"])
    if quantized is None:
        quantized = quantize_actual(actual, batch)
        save_checkpoint(folder, "quantized", quantized, q_identity)
    observation_diag = {key: stored["observation"][key] for key in ("update_factor_rel_l2", "update_factor_max_abs")}
    final_losses = stored["observation"]["final_per_sample_loss"].tolist()
    del actual, stored
    mp_identity = {**batch_identity, "stage": "diagnostic_MP_only", "quantized_sha256": file_hash(folder / "quantized.pt")}
    prediction = load_checkpoint(folder, "mp_prediction", mp_identity)
    if prediction is None:
        update_status(output, state="running", batch_index=batch, stage="diagnostic_MP_only")
        mp_started = time.perf_counter()
        prediction = mp_prediction_from_quantized(quantized["Y_hat"], quantized["delta"], requested_rank=N,
                      rounding="stochastic", c_mp=cfg["prediction"]["c_mp"], device=device)
        prediction["wall_time_sec"] = time.perf_counter() - mp_started
        save_checkpoint(folder, "mp_prediction", prediction, mp_identity)
    update_status(output, state="running", batch_index=batch, stage="CPA_full16")
    cpa_identity = {**batch_identity, "stage": "CPA_full16_no_debias_no_truncation", "quantized_sha256": mp_identity["quantized_sha256"]}
    cpa = load_checkpoint(folder, "stage1", cpa_identity)
    if cpa is None:
        cpa_started = time.perf_counter()
        result = stage1_call(quantized["Y_hat"], quantized["delta"], cfg, batch, device)
        if result.B_eff != N or tuple(result.S_hat.shape) != (N, 25088):
            raise RuntimeError("CPA must return all16 sources; no fallback permitted")
        targets, source_rows = fia_targets(result.S_hat, inputs["Z_evaluation_only"])
        cpa = {"S_hat": result.S_hat, "A_hat": result.A_hat, "fia_targets": targets, "source_rows": source_rows,
               "diagnostics": {f.name: getattr(result, f.name) for f in fields(result) if f.name not in {"S_hat", "A_hat", "Y_recon"}},
               "wall_time_sec": time.perf_counter() - cpa_started}
        save_checkpoint(folder, "stage1", cpa, cpa_identity)
        del result
    write_json(folder / "stage1_diagnostics.json", {"identity": cpa_identity, "attack": cpa["diagnostics"],
        "MP_diagnostic_only": prediction, "quantization": quantized["diagnostics"], "actual_update_residual": observation_diag})
    quant_runtime, quant_diag = quantized["wall_time_sec"], quantized["diagnostics"]
    del quantized
    if device.type == "cuda":
        torch.cuda.empty_cache()
    update_status(output, state="running", batch_index=batch, stage="FIA_25000")
    fia_identity = {**batch_identity, "stage": "unchanged_FIA_25000", "stage1_sha256": file_hash(folder / "stage1.pt")}
    reconstruction = load_checkpoint(folder, "fia", fia_identity)
    if reconstruction is None:
        fia = cfg["fia"]
        result = run_direct_fia(target_z=cpa["fia_targets"], x_true=inputs["x"], device=device,
                  **{key: fia[key] for key in ("n_iter", "log_every", "lr", "tv", "fi", "optimizer", "scheduler", "seed")},
                  progress_label=f"UIA_4bit_B16_batch{batch}")
        reconstruction = {"rec": result.rec, "metrics": vars(result.metrics), "log": result.log, "wall_time_sec": result.wall_time_sec}
        save_checkpoint(folder, "fia", reconstruction, fia_identity)
        del result
    rec, metrics = reconstruction["rec"], reconstruction["metrics"]
    if tuple(rec.shape) != (N, 3, 224, 224) or not bool(torch.isfinite(rec).all()):
        raise RuntimeError("Incomplete or nonfinite reconstruction")
    if any(len(metrics[key]) != N or not np.isfinite(metrics[key]).all() for key in ("lpips", "psnr", "ssim")):
        raise RuntimeError("All16 image metrics must be finite")
    rows = [{**inputs["metadata"][i], **cpa["source_rows"][i], "run_tag": RUN_TAG, "n": N, "T": 10, "bits": 4,
             "scale_granularity": "per_row", "final_per_sample_loss": float(final_losses[i]),
             **{key: float(metrics[key][i]) for key in ("lpips", "psnr", "ssim")}} for i in range(N)]
    write_csv(folder / "per_image_metrics.csv", rows)
    write_csv(folder / "source_cosines.csv", [{**inputs["metadata"][i], **cpa["source_rows"][i]} for i in range(N)])
    torch.save(rec, folder / "rec.pt")
    write_json(folder / "fia_log.json", reconstruction["log"])
    render_grid(inputs["x"], rec, rows, folder / "reconstruction_grid.png", title=f"UIA4bit B16 T10 batch{batch}: all16 images")
    summary = {"run_tag": RUN_TAG, "identity": batch_identity, "batch_index": batch, "n_images": N,
               "predicted_B_eff": prediction["B_eff_pred"], "measured_B_eff_cos09": sum(row["measured_recovered_cos09"] for row in rows),
               "stage1_B_eff": N, "oracle_fills": 0, "update_collection_executed": False,
               **{f"{key}_mean": float(np.mean(metrics[key])) for key in ("lpips", "psnr", "ssim")},
               "quantization": quant_diag, "prediction": prediction, "actual_update_residual": observation_diag,
               "runtime_sec": {"update_collection": 0., "quantization": quant_runtime, "MP_diagnostic": prediction["wall_time_sec"],
                               "stage1": cpa["wall_time_sec"], "fia": reconstruction["wall_time_sec"], "current_invocation": time.perf_counter() - started},
               "completed_at_utc": utc_now(), "output_hashes": {name: file_hash(folder / name) for name in
               ("inputs.pt", "quantized.pt", "mp_prediction.pt", "stage1.pt", "fia.pt", "rec.pt", "per_image_metrics.csv",
                "source_cosines.csv", "stage1_diagnostics.json", "fia_log.json", "reconstruction_grid.png")}}
    write_json(folder / "summary.json", summary)
    print(f"COMPLETE UIA4bit batch={batch}: LPIPS={summary['lpips_mean']:.8f}, pred={summary['predicted_B_eff']}/16, measured={summary['measured_B_eff_cos09']}/16", flush=True)
    return summary


def comparison_rows(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fp32 = read_csv(resolve_path(cfg["source_fp32_output"]) / "per_image_metrics.csv")
    gia = [r for r in read_csv(resolve_path(cfg["gia_reference_csv"])) if r["method"] == "naive"]
    if len(fp32) != 80 or len(gia) != 40 or len(rows) != 80:
        raise RuntimeError("3row comparison requires complete80/80/40 rows")
    paired = []
    for a, b in zip(fp32, rows):
        if int(a["global_index"]) != int(b["global_index"]) or a["image_path"] != b["image_path"]:
            raise RuntimeError("Paired80 comparison image identity differs")
        paired.append({"global_index": int(b["global_index"]), "image_path": b["image_path"],
            **{f"{key}_{suffix}": float(row[key]) for key in ("lpips", "psnr", "ssim") for suffix, row in (("fp32", a), ("4bit", b))},
            "lpips_4bit_minus_fp32": float(b["lpips"]) - float(a["lpips"])})
    comparisons = []
    for label, n, T, selected, note in (("UIA fp32", 16, 10, fp32, "paired same80 actual updates"),
                                     ("UIA 4-bit", 16, 10, rows, "paired same80 actual updates"),
                                     ("GIA 4-bit", 8, 1, gia, "historical40; different batch protocol; contextual comparison")):
        cos_key = "target_source_cosine" if label.startswith("GIA") else "source_abs_cosine"
        comparisons.append({"condition": label, "n": n, "T": T, "n_images": len(selected),
             **{f"{key}_mean": float(np.mean([float(row[key]) for row in selected])) for key in ("lpips", "psnr", "ssim")},
             "embedding_cos09": sum(float(row[cos_key]) > .9 for row in selected), "note": note})
    return comparisons, paired


def random_grid(originals: torch.Tensor, fp32: torch.Tensor, quantized: torch.Tensor,
                rows: list[dict[str, Any]], selected: list[int], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    truth = denormalize_imagenet(originals)
    tile, gap, width, height = 112, 10, 1450, 580
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    draw.text((12, 10), "UIA n16 T10: Original / fp32 / 4-bit | 12 of80, seed42, not cherry-picked", fill="black", font=font)
    for i, index in enumerate(selected):
        left, top = 12 + (i % 4) * 360, 50 + (i // 4) * 174
        draw.text((left, top), f"g{index:02d} 4-bit LPIPS={float(rows[index]['lpips']):.3f}", fill="black", font=font)
        for j, tensor in enumerate((truth[index], fp32[index], quantized[index])):
            array = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            canvas.paste(Image.fromarray(array).resize((tile, tile), Image.Resampling.LANCZOS), (left + j * (tile + gap), top + 30))
    canvas.save(path)


def summarize(cfg: dict[str, Any], manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    rows, batches, originals, recs, fp32_recs = [], [], [], [], []
    for batch in BATCHES:
        folder = output / f"batch{batch:02d}"
        batches.append(validate_completed(folder, {**manifest["identity"], "batch_index": batch}))
        rows.extend(read_csv(folder / "per_image_metrics.csv"))
        originals.append(torch.load(folder / "inputs.pt", map_location="cpu", weights_only=False)["x"])
        recs.append(torch.load(folder / "rec.pt", map_location="cpu", weights_only=True))
        fp32_recs.append(torch.load(resolve_path(cfg["source_fp32_output"]) / f"batch{batch:02d}/rec.pt", map_location="cpu", weights_only=True))
    if [int(row["global_index"]) for row in rows] != list(range(80)):
        raise RuntimeError("Incomplete all80 metric coverage")
    comparisons, paired = comparison_rows(cfg, rows)
    write_csv(output / "per_image_metrics.csv", rows)
    write_csv(output / "source_cosines.csv", [{key: row[key] for key in ("global_index", "batch_index", "source_index", "source_abs_cosine", "fia_abs_target_cosine", "measured_recovered_cos09", "oracle_fill")} for row in rows])
    write_csv(output / "paired_fp32_4bit_metrics.csv", paired)
    write_csv(output / "three_condition_comparison.csv", comparisons)
    write_csv(output / "per_batch_summary.csv", [{key: row[key] for key in ("batch_index", "n_images", "predicted_B_eff", "measured_B_eff_cos09", "lpips_mean", "psnr_mean", "ssim_mean")} | {f"runtime_{key}_sec": value for key, value in row["runtime_sec"].items()} for row in batches])
    selected = manifest["random_grid_global_indices"]
    write_csv(output / "random_grid_selection.csv", [rows[i] for i in selected])
    random_grid(torch.cat(originals), torch.cat(fp32_recs), torch.cat(recs), rows, selected, output / "reconstruction_grid_not_cherry_picked.png")
    render_grid(torch.cat(originals), torch.cat(recs), rows, output / "all80_reconstruction_grid.png", title="UIA4bit n16 T10: all80 images")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)
    for label, values, column in (("UIA fp32", read_csv(resolve_path(cfg["source_fp32_output"]) / "per_image_metrics.csv"), "source_abs_cosine"), ("UIA 4-bit", rows, "source_abs_cosine")):
        axes[0].hist([float(row["lpips"]) for row in values], bins=np.arange(0, 1.50001, .05), histtype="step", label=label, linewidth=1.8)
        axes[1].hist([float(row[column]) for row in values], bins=np.linspace(0, 1.00001, 31), histtype="step", label=label, linewidth=1.8)
    axes[0].set(xlabel="Per-image LPIPS", ylabel="Image count")
    axes[1].set(xlabel="Matched absolute source cosine", ylabel="Source count")
    axes[1].axvline(.9, color="black", linestyle="--", linewidth=1)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "lpips_source_cosine_histograms.png")
    plt.close(fig)
    mean_metrics = {f"{key}_mean": float(np.mean([float(row[key]) for row in rows])) for key in ("lpips", "psnr", "ssim", "source_abs_cosine")}
    summary = {"schema_version": 1, "run_tag": RUN_TAG, "status": "complete", "identity": manifest["identity"],
         "config": cfg, "config_sha256": manifest["identity"]["protocol_sha256"], "n_images": 80, "oracle_fills": 0,
         "metrics": mean_metrics | {"measured_B_eff_cos09_total": sum(row["measured_B_eff_cos09"] for row in batches),
              "predicted_B_eff_total": sum(row["predicted_B_eff"] for row in batches),
              "source_cos_per_source": [float(row["source_abs_cosine"]) for row in rows]},
         "per_victim_batch": batches, "comparison": comparisons,
         "paired_lpips_delta_4bit_minus_fp32": float(np.mean([r["lpips_4bit_minus_fp32"] for r in paired])),
         "runtime_sec": {key: sum(row["runtime_sec"][key] for row in batches) for key in ("update_collection", "quantization", "MP_diagnostic", "stage1", "fia")},
         "completed_at_utc": utc_now()}
    report = ["# M7 UIA4bit n16 T10", "", "All80 original images and the exact five stored fp32 actual client updates were reused. No new training or update collection occurred.", "",
        "| Condition | n / T | Images | LPIPS | PSNR | SSIM | Embedding cos>0.9 |", "|---|---:|---:|---:|---:|---:|---:|"]
    report.extend(f"| {r['condition']} | {r['n']} / {r['T']} | {r['n_images']} | {r['lpips_mean']:.4f} | {r['psnr_mean']:.3f} | {r['ssim_mean']:.4f} | {r['embedding_cos09']}/{r['n_images']} |" for r in comparisons)
    report.extend(["", "UIA rows are paired on the same80 originals and actual updates. GIA4bit is a historical40-image B8 gradient experiment; it is a cross-protocol reference.",
        f"Paired mean LPIPS increase (4-bit minus fp32): {summary['paired_lpips_delta_4bit_minus_fp32']:.8f}.",
        f"MP predicted B_eff: {summary['metrics']['predicted_B_eff_total']}/80; measured matched |cosine|>0.9: {summary['metrics']['measured_B_eff_cos09_total']}/80.", "",
        "The repository's stochastic quantizer is its subtractive-dither observation model Y+Uniform(-delta/2,delta/2), with per-row delta=2max(abs(Y))/15 and inward cell projection. It is not conventional integer stochastic rounding or a measured deployed integer codec. Quantization applies only to the saved actual Y_delta_over_lr; ideal B_T@Z is never an attack input.",
        "The existing MP predictor debiases its diagnostic covariance and uses c_MP=3. CPA separately keeps full16 raw quantized axes with no debiasing or MP truncation. The diagnostic does not choose targets or alter the attack.",
        "CPA and FIA each use25000 iterations, with all source adaptation, matching, budgets and FIA TV0.1/FI1/Adam0.1/cosine/seed42 held equal to UIAfp32. Matching permutes abs(S_hat) only. Every image is scored; no oracle fills, Stage2, refinement or new training.",
        "UIAfp32 LPIPS0.4546 lies0.0066 above the upper end of the published CPA+FIA Table5 batch16–128 range0.410–0.448; this is a range-boundary comparison, not identical-protocol reproduction.", "",
        "![Random12: not cherry-picked](reconstruction_grid_not_cherry_picked.png)", "![All80](all80_reconstruction_grid.png)", "![Distributions](lpips_source_cosine_histograms.png)", ""])
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    names = ("per_image_metrics.csv", "source_cosines.csv", "paired_fp32_4bit_metrics.csv", "three_condition_comparison.csv", "per_batch_summary.csv", "random_grid_selection.csv", "reconstruction_grid_not_cherry_picked.png", "all80_reconstruction_grid.png", "lpips_source_cosine_histograms.png", "report.md")
    summary["output_hashes"] = {name: file_hash(output / name) for name in names}
    # All report artifacts precede the terminal marker. The local ledger avoids
    # interleaving writes with concurrently running independent analyses.
    (output / "runs.jsonl").write_text(json.dumps(json_safe(summary), ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_json(output / "summary.json", summary)
    update_status(output, state="complete", protocol_sha256=summary["config_sha256"], report_path=str(output / "report.md"))
    return summary


def worker(config_path: Path, cfg: dict[str, Any], output: Path) -> int:
    """Use Python's integer subprocess returncode, not a nullable PS ExitCode."""
    queue_path = output / "queue_status.json"
    state = {"run_tag": RUN_TAG, "worker_pid": os.getpid(), "state": "starting", "started_at_utc": utc_now(),
             "command": [sys.executable, "-u", "-m", "qacpa.run_m7_uia_4bit", "--config", str(config_path), "--phase", "run"]}
    report_script = PROJECT / "qacpa/report_m7_followup_review.py"
    state["report_generator_sha256_at_launch"] = file_hash(report_script) if report_script.is_file() else None
    write_json(queue_path, state)
    try:
        with (output / "run.stdout.log").open("a", encoding="utf-8") as stdout, (output / "run.stderr.log").open("a", encoding="utf-8") as stderr:
            process = subprocess.Popen(state["command"], cwd=PROJECT, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
            state.update(state="running", runner_pid=process.pid)
            write_json(queue_path, state)
            code = process.wait()
        state["runner_exit_code"] = int(code)
        if code != 0:
            raise RuntimeError(f"Runner exited with integer code{code}; see run.stderr.log")
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "complete" or summary["n_images"] != 80 or summary["config_sha256"] != file_hash(config_path):
            raise RuntimeError("Runner exited without the complete registered80-image result")
        for name, digest in summary["output_hashes"].items():
            if file_hash(output / name) != digest:
                raise RuntimeError("Completed report integrity verification failed")
        state.update(state="complete", completed_at_utc=utc_now())
        write_json(queue_path, state)
        # Documentation is a separately recorded publication step. Its failure
        # cannot turn already verified scientific artifacts into a failed run.
        postprocess = {"state": "running", "started_at_utc": utc_now(),
                       "report_generator_sha256_at_launch": state["report_generator_sha256_at_launch"]}
        write_json(output / "postprocess_status.json", postprocess)
        try:
            if not report_script.is_file() or file_hash(report_script) != state["report_generator_sha256_at_launch"]:
                raise RuntimeError("Report generator missing or changed since launch")
            with (output / "postprocess.stdout.log").open("a", encoding="utf-8") as stdout, (output / "postprocess.stderr.log").open("a", encoding="utf-8") as stderr:
                rendered = subprocess.run([sys.executable, "-m", "qacpa.report_m7_followup_review"], cwd=PROJECT,
                    stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}, check=False)
            postprocess.update(exit_code=int(rendered.returncode), state="complete" if rendered.returncode == 0 else "failed")
        except Exception as exc:
            postprocess.update(state="failed", error=str(exc))
        postprocess["finished_at_utc"] = utc_now()
        write_json(output / "postprocess_status.json", postprocess)
        return 0
    except BaseException as exc:
        state.update(state="failed", error=str(exc), failed_at_utc=utc_now())
        write_json(queue_path, state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("prepare", "run", "report", "worker"), default="prepare")
    args = parser.parse_args()
    cfg = read_config(args.config)
    output = resolve_path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "worker":
        raise SystemExit(worker(args.config, cfg, output))
    guard = (output / ".execution.lock").open("a+b")
    guard.seek(0, 2)
    if guard.tell() == 0:
        guard.write(b"0")
        guard.flush()
    guard.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        torch.set_num_threads(4)
        manifest = prepare(args.config, cfg, output)
        if args.phase == "prepare":
            print("CPU identity freeze complete; no quantization, new metrics or GPU work performed", flush=True)
            return
        if args.phase == "run":
            release_path = output / "gpu_release.json"
            release = json.loads(release_path.read_text(encoding="utf-8"))
            if release.get("released") is not True or release.get("protocol_sha256") != manifest["identity"]["protocol_sha256"]:
                raise RuntimeError("Explicit scheduler GPU release for this frozen protocol required")
            device = torch.device(cfg["device"])
            if device.type != "cuda" or not torch.cuda.is_available() or torch.cuda.get_device_capability(device) != (12, 0):
                raise RuntimeError("Expected established sm120 CUDA environment")
            for batch in BATCHES:
                run_batch(cfg, manifest["identity"], batch, output, device)
        summarize(cfg, manifest, output)
    except BaseException as exc:
        update_status(output, state="failed", phase=args.phase, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        guard.close()


if __name__ == "__main__":
    main()
