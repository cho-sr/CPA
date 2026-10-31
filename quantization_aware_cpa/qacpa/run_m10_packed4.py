"""Run only upstream CPA+FIA on a real packed4 update of historical best10.

The selected images/features are client inputs for a new n10/T10 update.
Only a decoder's packet output enters CPA; truth is loaded after FIA to score.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .communication_codec import CODEC_ID, encode_uint4, decode_packet, packet_info
from .run_m8_fedavg_naive import (
    PROJECT, WORKSPACE, file_hash, read_json, write_json, read_csv, write_csv,
    utc_now, resolve, status, execution_lock, evaluate_after_fia,
)
from .run_m6_uia_fp32_fia import actual_observation, save_checkpoint, load_checkpoint
from .uia import run_fc_only_fedavg_on_features
from .upstream_naive import run_upstream_cpa, run_upstream_fia
from .fia import evaluate_images
from .seed import seed_all

RUN_TAG = "m10_cpa_fia_packed4_best10"
DEFAULT_CONFIG = PROJECT / f"configs/{RUN_TAG}.json"


@dataclass
class SelectedClientFeatures:
    Z: torch.Tensor
    labels: torch.Tensor
    initial_per_sample_loss: torch.Tensor


def select_best10(rows: list[dict]) -> list[dict]:
    if len(rows) != 80 or {int(r["global_index"]) for r in rows} != set(range(80)):
        raise ValueError("Selection requires all80 uniquely indexed historical M8 results")
    if not all(math.isfinite(float(r["lpips"])) for r in rows):
        raise ValueError("Historical LPIPS must be finite")
    return sorted(rows, key=lambda r: (float(r["lpips"]), int(r["global_index"])))[:10]


def validate_config(cfg: dict) -> None:
    ref = read_json(PROJECT / "configs/m8_fedavg_naive_cpa_dither4_B16.json")
    if (cfg["run_tag"], cfg["n"], cfg["T"], cfg["n_images"], cfg["batch_indices"]) != (RUN_TAG, 10, 10, 10, [0]):
        raise ValueError("Only the selected n10/T10 packed4 CPA+FIA arm is authorized")
    if cfg["stage1"] != ref["stage1"] or cfg["fia"] != ref["fia"]:
        raise ValueError("The original M8 CPA and FIA settings must remain exact")
    if cfg["codec"] != CODEC_ID or cfg["codec_seed"] != 0 or cfg["seed"] != 0:
        raise ValueError("Unexpected codec or seed")
    if cfg["selection_reference"] != "results/m8_fedavg_naive_cpa_dither4_B16" or cfg["source_fp32_output"] != "results/m6_uia_fp32_B16_fia":
        raise ValueError("Use the audited historical M8 scores and M6 cached images/features")
    for key in ("ls_A", "debias", "mp_truncation", "mp_diagnostic", "stage2", "refine_A", "oracle_fills", "fp32_attack", "raw_row_attack"):
        if cfg.get(key) is not False:
            raise ValueError(f"{key} must be explicitly false")
    if not cfg["new_training"] or cfg["local_training"] != {"lr": .001, "steps": 10, "optimizer": "sgd", "momentum": 0., "weight_decay": 0., "classifier_dropout": False, "seed": 0}:
        raise ValueError("The selected client requires the unchanged FC-only T10 SGD schedule")


def code_hashes() -> dict:
    names = ["run_m10_packed4.py", "communication_codec.py", "report_m10_packed4.py", "upstream_naive.py", "uia.py", "victim.py", "seed.py", "fia.py", "run_m8_fedavg_naive.py", "run_m6_uia_fp32_fia.py"]
    paths = [PROJECT / "qacpa" / name for name in names]
    paths += sorted((WORKSPACE / "official_cpa/src").glob("*.py"))
    paths += [PROJECT / "scripts/launch_m10_packed4.ps1"]
    return {str(p.relative_to(WORKSPACE)): file_hash(p) for p in paths}


def prepare(config_path: Path, cfg: dict, output: Path) -> dict:
    reference = resolve(cfg["selection_reference"])
    historical = read_json(reference / "summary.json")
    csv_path = reference / "per_image_metrics.csv"
    csv_hash = file_hash(csv_path)
    if historical["status"] != "complete" or historical["n_images"] != 80 or csv_hash != historical["output_hashes"]["per_image_metrics.csv"]:
        raise RuntimeError("Historical M8 selection source is incomplete or changed")
    weights = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
    weights_hash = file_hash(weights)
    if weights_hash != read_json(resolve(cfg["source_fp32_output"]) / "summary.json")["identity"]["weights_sha256"]:
        raise RuntimeError("Pretrained weights do not match the original client")
    identity = {"config_sha256": file_hash(config_path), "code_sha256": code_hashes(), "selection_csv_sha256": csv_hash,
                "weights_sha256": weights_hash, "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version}}
    # The direct upstream implementation must still match the audited design.
    design = read_json(PROJECT / "configs/m10_compression_only_B16.protocol.json")
    for rel, expected in design["code_sha256_at_design"].items():
        if file_hash(WORKSPACE / rel) != expected:
            raise RuntimeError(f"Previously audited code/config changed: {rel}")
    manifest_path = output / "input_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest["identity"] != identity:
            raise RuntimeError("Frozen code/config/selection changed; refusing incompatible resume")
        for rel, expected in manifest["inputs_sha256"].items():
            if file_hash(resolve(rel)) != expected:
                raise RuntimeError(f"Frozen input changed: {rel}")
        return manifest
    selected = select_best10(read_csv(csv_path))
    if len({int(r["label"]) for r in selected}) != 10:
        raise RuntimeError("Selected best10 do not preserve distinct labels; do not silently replace any")
    tensors_x, tensors_z, metadata = [], [], []
    inputs_hashes = {str(csv_path.relative_to(PROJECT)): csv_hash}
    for rank, row in enumerate(selected):
        b, i = int(row["batch_index"]), int(row["source_index"])
        folder = resolve(cfg["source_fp32_output"]) / f"batch{b:02d}"
        path = folder / "inputs.pt"
        rel = str(path.relative_to(PROJECT))
        if rel not in inputs_hashes:
            digest = file_hash(path)
            if digest != read_json(folder / "inputs.meta.json")["file_sha256"] or digest != read_json(folder / "summary.json")["output_hashes"]["inputs.pt"]:
                raise RuntimeError("Stored source images/features failed integrity checks")
            inputs_hashes[rel] = digest
        source = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        old = source["metadata"][i]
        for key in ("global_index", "batch_index", "source_index", "dataset_index", "label", "image_path"):
            if str(old[key]) != str(row[key]):
                raise RuntimeError(f"Historical image identity mismatch: {key}")
        tensors_x.append(source["x"][i].clone())
        tensors_z.append(source["Z_evaluation_only"][i].clone())
        metadata.append({**old, "historical_global_index": int(old["global_index"]), "historical_batch_index": b,
                         "historical_source_index": i, "historical_m8_lpips": float(row["lpips"]),
                         "selection_rank": rank + 1, "global_index": rank, "batch_index": 0, "source_index": rank})
        del source
    inputs = {"x": torch.stack(tensors_x), "Z_evaluation_only": torch.stack(tensors_z), "metadata": metadata,
              "selection": "historical M8 lowest-LPIPS10 of80; client recollection inputs and post-FIA evaluation only"}
    if inputs["x"].shape != (10, 3, 224, 224) or inputs["Z_evaluation_only"].shape != (10, 25088):
        raise RuntimeError("Selected input shape mismatch")
    if not all(bool(torch.isfinite(inputs[key]).all()) for key in ("x", "Z_evaluation_only")):
        raise RuntimeError("Selected inputs contain nonfinite values")
    source_path = resolve(cfg["source_output"]) / "batch00/inputs.pt"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(inputs, source_path)
    inputs_hashes[str(source_path.relative_to(PROJECT))] = file_hash(source_path)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True).strip()
    manifest = {"run_tag": cfg["run_tag"], "schema_version": 1, "git_sha": git_sha, "identity": identity,
                "inputs_sha256": inputs_hashes, "weights_path": str(weights), "weights_sha256": weights_hash,
                "created_at_utc": utc_now(), "n_images": 10, "selection": {"method": "historical M8 LPIPS ascending, then global_index ascending; first10",
                "historical_global_indices": [int(r["global_index"]) for r in selected], "source_csv_sha256": csv_hash,
                "scope": "selected exploratory sample; new n10 update; no paired historical n16 claim"},
                "metadata": metadata, "random_grid_indices": list(range(10))}
    write_json(manifest_path, manifest)
    write_csv(output / "selection.csv", metadata)
    write_json(config_path.with_suffix(".lock.json"), {"config_sha256": identity["config_sha256"], "input_manifest_sha256": file_hash(manifest_path), "frozen_at_utc": utc_now()})
    print(f"Selected historical global IDs: {manifest['selection']['historical_global_indices']}", flush=True)
    return manifest


def collect_selected_update(cfg: dict, manifest: dict, output: Path, device: torch.device) -> dict:
    folder = output / "batch00"
    collection_identity = {**manifest["identity"], "selected_inputs_sha256": file_hash(resolve(cfg["source_output"]) / "batch00/inputs.pt"), "stage": "selected_n10_actual_T10_update"}
    saved = load_checkpoint(folder, "client_update", collection_identity)
    if saved is not None:
        return saved
    status(output, cfg, state="running", batch_index=0, stage="selected_n10_client_update_T10")
    inputs = torch.load(resolve(cfg["source_output"]) / "batch00/inputs.pt", map_location="cpu", weights_only=False)
    feature = SelectedClientFeatures(inputs["Z_evaluation_only"], torch.tensor([r["label"] for r in inputs["metadata"]], dtype=torch.long),
                                    torch.tensor([r["initial_per_sample_loss"] for r in inputs["metadata"]], dtype=torch.float32))
    seed_all(cfg["local_training"]["seed"])
    started = time.perf_counter()
    observed = run_fc_only_fedavg_on_features(feature, lr=.001, steps=10, device=device)
    Y = actual_observation(observed, maximum_relative_error=.001)
    saved = {"Y_delta_over_lr": Y, "wall_time_sec": time.perf_counter() - started,
             "diagnostics": {"update_factor_rel_l2": observed.update_factor_rel_l2, "update_factor_max_abs": observed.update_factor_max_abs,
                             "initial_per_sample_loss": observed.initial_per_sample_loss.tolist(), "final_per_sample_loss": observed.final_per_sample_loss.tolist(),
                             "n": 10, "T": 10, "observation_definition": "(W_initial-W_final)/lr", "new_update_collection": True}}
    save_checkpoint(folder, "client_update", saved, collection_identity)
    del observed, inputs, feature
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return saved


def quantization_diagnostics(Y: torch.Tensor, decoded: torch.Tensor) -> dict:
    # Chunked float64 accumulation avoids extra full-matrix diagnostic copies.
    sums = {"error": 0., "error2": 0., "original2": 0., "decoded2": 0., "dot": 0.}
    maximum, zero_values, zero_rows, positive_codes, endpoint_elements = 0., 0, 0, 0, 0
    for start in range(0, Y.shape[0], 64):
        x = Y[start:start + 64].double()
        z = decoded[start:start + 64].double()
        err = z - x
        sums["error"] += err.sum().item()
        sums["error2"] += err.square().sum().item()
        sums["original2"] += x.square().sum().item()
        sums["decoded2"] += z.square().sum().item()
        sums["dot"] += (x * z).sum().item()
        maximum = max(maximum, err.abs().max().item())
        radius = x.abs().amax(dim=1, keepdim=True)
        zero_rows += int((radius == 0).sum())
        zero_values += int((x == 0).sum())
        nonzero = radius > 0
        positive_codes += int(nonzero.sum()) * x.shape[1]
        endpoint_elements += int(((z.abs() == radius) & nonzero).sum())
        tolerance = 8 * torch.finfo(torch.float32).eps * radius + torch.finfo(torch.float32).tiny
        if bool((err.abs() > 2 * radius / 15 + tolerance).any()):
            raise RuntimeError("Real stochastic quantization exceeded its full-delta error bound")
    count = Y.numel()
    return {"relative_l2": math.sqrt(sums["error2"] / max(sums["original2"], 1e-300)), "mse": sums["error2"] / count,
            "max_abs": maximum, "empirical_error_mean": sums["error"] / count,
            "cosine": sums["dot"] / max(math.sqrt(sums["original2"] * sums["decoded2"]), 1e-300),
            "zero_fraction": zero_values / count, "zero_scale_rows": zero_rows,
            "endpoint_fraction_nonzero_rows": endpoint_elements / max(positive_codes, 1), "range_overflow_clipping": 0,
            "code_min": 0, "code_max": 15, "code_range_fields": "codec alphabet bounds, not measured extrema",
            "roundtrip_verified": True, "decoder_original_input_required": False, "legacy_half_delta_gate_used": False}


def prepare_packet(cfg: dict, manifest: dict, output: Path, device: torch.device) -> None:
    folder = output / "batch00"
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "packet_manifest.json"
    if marker.is_file():
        saved = read_json(marker)
        if saved["identity"] != manifest["identity"]:
            raise RuntimeError("Packet identity changed")
        for name, expected in saved["output_hashes"].items():
            if file_hash(folder / name) != expected:
                raise RuntimeError("Prepared packet or diagnostics changed")
        return
    update = collect_selected_update(cfg, manifest, output, device)
    status(output, cfg, state="running", batch_index=0, stage="actual_uint4_encode_and_decoder_validation")
    started = time.perf_counter()
    packet = encode_uint4(update["Y_delta_over_lr"], seed=cfg["codec_seed"])
    packet_path = folder / "observation.bin"
    temporary = packet_path.with_suffix(".bin.tmp")
    temporary.write_bytes(packet)
    temporary.replace(packet_path)
    # This decoder receives a fresh disk read, with no source or RNG argument.
    received = packet_path.read_bytes()
    if received != packet:
        raise RuntimeError("Written packet differs from transmitted bytes")
    decoded = decode_packet(received)
    info = packet_info(received)
    if info["packet_bytes"] != packet_path.stat().st_size or info["packet_bytes"] != 51396640:
        raise RuntimeError("Unexpected real packed4 byte budget")
    diag = quantization_diagnostics(update["Y_delta_over_lr"], decoded)
    diag["quantization_seed"] = cfg["codec_seed"]
    diag["update_collection_sec"] = update["wall_time_sec"]
    diag["update_factor_rel_l2"] = update["diagnostics"]["update_factor_rel_l2"]
    elapsed = time.perf_counter() - started
    communication = {"codec": CODEC_ID, "bits": 4, "shape": [4096, 25088], "n_values": 4096 * 25088,
                     "fp32_payload_bytes": 411041792, "fp32_packet_bytes": 411041824,
                     "packed_payload_bytes": info["code_bytes"], "scale_bytes": info["radius_bytes"],
                     "header_bytes": info["header_bytes"], "packet_bytes": info["packet_bytes"],
                     "compression_ratio": 411041824 / info["packet_bytes"], "saved_percent": 100 * (1 - info["packet_bytes"] / 411041824),
                     "scope": "attacked classifier.0.weight packet only", "codec_wall_time_sec": elapsed,
                     "transmitted_tensor_convention": "Y=(W_initial-W_final)/lr", "server_weight_delta": "-0.001*decoded_Y",
                     "sha256": hashlib.sha256(received).hexdigest(), "source_reconstruction_only_from_packet": True}
    write_json(folder / "communication.json", communication)
    write_json(folder / "quantization_diagnostics.json", diag)
    write_json(marker, {"identity": manifest["identity"], "output_hashes": {name: file_hash(folder / name) for name in ("observation.bin", "communication.json", "quantization_diagnostics.json")}})
    print(f"PACKET VERIFIED bytes={info['packet_bytes']} ratio={communication['compression_ratio']:.6f} relative_l2={diag['relative_l2']:.6f}", flush=True)
    del update, packet, received, decoded
    gc.collect()


def attack_from_packet(cfg: dict, output: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    """No original image/embedding/update is accepted by this attack boundary."""
    folder = output / "batch00"
    Y_hat = decode_packet((folder / "observation.bin").read_bytes())
    if Y_hat.shape != (4096, 25088):
        raise RuntimeError("Decoded observation shape mismatch")
    status(output, cfg, state="running", batch_index=0, stage="original_CPA_full10_from_received_packet")
    cpa = run_upstream_cpa(Y_hat=Y_hat, n_components=10, device=device, config=cfg["stage1"], output_dir=folder / "cpa", seed=0)
    sources = cpa["S_hat"].detach().cpu()
    if sources.shape != (10, 25088) or not bool(torch.isfinite(sources).all()):
        raise RuntimeError("CPA must return all10 finite sources")
    del Y_hat
    torch.save(sources, folder / "sources.pt")
    status(output, cfg, state="running", batch_index=0, stage="original_FIA_25000_source_order")
    fia = run_upstream_fia(target_z=sources.abs().contiguous(), device=device, config=cfg["fia"], output_dir=folder / "fia", seed=42)
    source_recs = fia["rec"].detach().cpu()
    if source_recs.shape != (10, 3, 224, 224) or not bool(torch.isfinite(source_recs).all()):
        raise RuntimeError("FIA must return all10 finite images")
    torch.save(source_recs, folder / "source_order_rec.pt")
    return sources, source_recs, cpa, fia


def run_batch(cfg: dict, manifest: dict, output: Path, device: torch.device) -> None:
    folder = output / "batch00"
    marker = folder / "summary.json"
    if marker.is_file():
        previous = read_json(marker)
        if previous["identity"] != manifest["identity"] or previous["n_images"] != 10:
            raise RuntimeError("Completed batch identity mismatch")
        for name, expected in previous["output_hashes"].items():
            if file_hash(folder / name) != expected:
                raise RuntimeError("Completed batch output changed")
        return
    started = time.perf_counter()
    prepare_packet(cfg, manifest, output, device)
    sources, source_recs, cpa, fia = attack_from_packet(cfg, output, device)
    status(output, cfg, state="running", batch_index=0, stage="evaluation_after_FIA")
    inputs = torch.load(resolve(cfg["source_output"]) / "batch00/inputs.pt", map_location="cpu", weights_only=False)
    recs, correspondence = evaluate_after_fia(sources, source_recs, inputs["Z_evaluation_only"])
    metrics = evaluate_images(x_true=inputs["x"], rec=recs.to(device), device=device)
    rows = [{**inputs["metadata"][i], **correspondence[i], "run_tag": cfg["run_tag"], "n": 10, "T": 10,
             "observation_model": CODEC_ID, "ls_A": False, "oracle_fill": False,
             **{key: float(getattr(metrics, key)[i]) for key in ("lpips", "psnr", "ssim")}} for i in range(10)]
    torch.save(recs, folder / "rec.pt")
    write_csv(folder / "per_image_metrics.csv", rows)
    write_json(folder / "cpa_log.json", cpa["log"])
    write_json(folder / "fia_log.json", fia["log"])
    communication = read_json(folder / "communication.json")
    quant = read_json(folder / "quantization_diagnostics.json")
    names = ("sources.pt", "source_order_rec.pt", "rec.pt", "per_image_metrics.csv", "cpa_log.json", "fia_log.json", "observation.bin", "communication.json", "quantization_diagnostics.json", "packet_manifest.json", "client_update.pt", "client_update.meta.json")
    done = {"identity": manifest["identity"], "batch_index": 0, "n_images": 10, "n": 10, "T": 10, "oracle_fills": 0, "ls_A": False,
            "embedding_cos09": sum(r["measured_recovered_cos09"] for r in rows),
            **{f"{key}_mean": float(np.mean([r[key] for r in rows])) for key in ("lpips", "psnr", "ssim")},
            "runtime_sec": {"update_collection": quant["update_collection_sec"], "codec": communication["codec_wall_time_sec"], "cpa": cpa["wall_time_sec"], "fia": fia["wall_time_sec"], "invocation": time.perf_counter() - started},
            "completed_at_utc": utc_now(), "output_hashes": {name: file_hash(folder / name) for name in names}}
    write_json(marker, done)
    print(f"COMPLETE selected10: LPIPS={done['lpips_mean']:.6f}, cosine09={done['embedding_cos09']}/10", flush=True)


def publish_status(cfg: dict, output: Path, state: str) -> None:
    path = WORKSPACE / "codex/STATUS.md"
    document = path.read_text(encoding="utf-8")
    begin, end = "<!-- M10_PACKED4_STATUS_BEGIN -->", "<!-- M10_PACKED4_STATUS_END -->"
    if document.count(begin) != 1 or document.count(end) != 1:
        raise RuntimeError("Expected one M10 status block")
    lines = [begin, "## M10 selected best10 CPA + FIA with actual packed4", "", f"- State: **{state.upper()}**; run tag `{cfg['run_tag']}`.",
             "- Historical M8 lowest-LPIPS10 of80; selected exploratory sample. One new n10/T10 client update from stored features.",
             "- Only packed4 CPA+FIA; upstream algorithms and budgets unchanged. No FP32 or raw-row attack.",
             f"- [Execution status](../quantization_aware_cpa/results/{cfg['run_tag']}/execution_status.json)."]
    if state == "complete":
        metrics = read_json(output / "summary.json")["metrics"]
        lines += [f"- All10: LPIPS **{metrics['lpips_mean']:.6f}**, embedding cosine>0.9 **{metrics['embedding_cos09']}/10**.",
                  f"- [Report and figures](../quantization_aware_cpa/results/{cfg['run_tag']}/report.md)."]
    lines += [end]
    start, finish = document.index(begin), document.index(end) + len(end)
    path.write_text(document[:start] + "\n".join(lines) + document[finish:], encoding="utf-8")


def worker(config_path: Path, cfg: dict, output: Path) -> int:
    state = {"run_tag": cfg["run_tag"], "state": "starting", "worker_pid": os.getpid(), "started_at_utc": utc_now()}
    write_json(output / "queue_status.json", state)
    command = [sys.executable, "-u", "-m", "qacpa.run_m10_packed4", "--config", str(config_path), "--phase", "run"]
    code = 1
    try:
        with (output / "run.stdout.log").open("a", encoding="utf-8") as out, (output / "run.stderr.log").open("a", encoding="utf-8") as err:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=out, stderr=err, env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            state.update(state="running", runner_pid=process.pid)
            write_json(output / "queue_status.json", state)
            code = int(process.wait())
        state.update(runner_exit_code=code, state="failed")
        if code == 0:
            result = read_json(output / "summary.json")
            if result["status"] != "complete" or result["n_images"] != 10 or result["identity"]["config_sha256"] != file_hash(config_path):
                raise RuntimeError("Missing complete registered10 result")
            if not all(file_hash(output / name) == expected for name, expected in result["output_hashes"].items()):
                raise RuntimeError("Final output hash check failed")
            state["state"] = "complete"
    except BaseException as exc:
        state.update(state="failed", error=f"{type(exc).__name__}: {exc}")
    state["finished_at_utc"] = utc_now()
    write_json(output / "queue_status.json", state)
    try:
        publish_status(cfg, output, state["state"])
    except Exception as exc:
        write_json(output / "postprocess_status.json", {"state": "failed", "error": str(exc)})
    return 0 if state["state"] == "complete" else (code or 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("prepare", "packet", "run", "report", "worker"), default="prepare")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = read_json(config_path)
    validate_config(cfg)
    output = resolve(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "worker":
        raise SystemExit(worker(config_path, cfg, output))
    torch.set_num_threads(4)
    with execution_lock(output):
        try:
            manifest = prepare(config_path, cfg, output)
            if args.phase == "prepare":
                status(output, cfg, state="prepared", selected_images=10)
                publish_status(cfg, output, "prepared")
                return
            device = torch.device(cfg["device"])
            if args.phase in ("packet", "run"):
                if device.type != "cuda" or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
                    raise RuntimeError("Expected the established sm_120 CUDA device")
                publish_status(cfg, output, "running")
                if args.phase == "packet":
                    prepare_packet(cfg, manifest, output, device)
                    status(output, cfg, state="packet_ready", selected_images=10)
                    return
                run_batch(cfg, manifest, output, device)
            from .report_m10_packed4 import summarize
            result = summarize(cfg, manifest, output)
            status(output, cfg, state="complete", n_images=10, metrics=result["metrics"])
            publish_status(cfg, output, "complete")
        except BaseException as exc:
            status(output, cfg, state="failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
