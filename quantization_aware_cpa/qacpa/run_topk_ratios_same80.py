"""Actual Top-k 30%, 20% or 10% sparse CPA + FIA using the unchanged five historical n16/T10 updates.

The default prepare phase verifies the original 80 cached images and updates.
Packet creation is CPU-only; --phase run explicitly starts CPA and FIA on CUDA.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .topk_ratio_codec import CODEC_ID, encode_topk, decode_packet, packet_info, sparsification_diagnostics
from .run_m6_uia_fp32_fia import actual_observation
from .run_m8_fedavg_naive import (
    PROJECT, WORKSPACE, file_hash, read_json, write_json, read_csv, write_csv,
    utc_now, resolve, status, execution_lock, evaluate_after_fia,
)
from .upstream_naive import run_upstream_cpa, run_upstream_fia
from .fia import evaluate_images

KEEP_PERCENTS = (30, 20, 10)
DEFAULT_CONFIG = PROJECT / "configs/cpa_fia_topk30_same80_B16.json"
IDENTITY_KEYS = ("global_index", "batch_index", "source_index", "dataset_index", "label", "image_path")


def validate_config(cfg: dict) -> None:
    baseline = read_json(PROJECT / "configs/cpa_fia_topk50_same80_B16.json")
    percent = cfg.get("keep_percent")
    if type(percent) is not int or percent not in KEEP_PERCENTS:
        raise ValueError("Only the authorized keep percentages 30, 20 and 10 are allowed")
    changed = {"run_tag", "output_dir", "codec", "observation", "expected_packet_bytes",
               "execution_scope", "sparsification"}
    if set(cfg) != set(baseline) | {"keep_percent"}:
        raise ValueError("Unexpected configuration fields")
    for key, value in baseline.items():
        if key not in changed and cfg.get(key) != value:
            raise ValueError(f"Keep the Top-k50 experiment's non-compression setting: {key}")
    tag = f"cpa_fia_topk{percent}_same80_B16"
    if cfg["run_tag"] != tag or cfg["output_dir"] != f"results/{tag}":
        raise ValueError("Use a separate run/output for each retained percentage")
    expected_bytes = 32 + 12845056 + 4 * (102760448 * percent // 100)
    if cfg["codec"] != CODEC_ID or cfg["expected_packet_bytes"] != expected_bytes:
        raise ValueError("Use the registered bitmap + unchanged FP32 packet")
    sparse = {**baseline["sparsification"], "keep_fraction": percent / 100}
    if cfg["sparsification"] != sparse:
        raise ValueError("Only the retained fraction may change within sparsification")



def code_hashes() -> dict:
    names = ("run_topk_ratios_same80.py", "report_topk_ratios_same80.py", "topk_ratio_codec.py",
             "queue_topk_ratios_same80.py", "run_topk50_same80.py", "report_topk50_same80.py", "run_m10_packed4.py",
             "report_m10_packed4.py", "sparsification_codec.py", "run_m6_uia_fp32_fia.py",
             "run_m8_fedavg_naive.py", "upstream_naive.py", "uia.py", "victim.py", "fia.py", "seed.py")
    paths = [PROJECT / "qacpa" / name for name in names]
    paths += sorted((WORKSPACE / "official_cpa/src").glob("*.py"))
    paths += [PROJECT / "scripts/run_topk50_same80.bat", PROJECT / "scripts/run_topk30_20_10_same80.bat"]
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in paths}


def validate_source_images(inputs: dict, historical: list[dict], batch: int) -> list[dict]:
    """Identity/order checks never rank the images using historical scores."""
    n = 16
    metadata = inputs["metadata"]
    if len(metadata) != n or len(historical) != n:
        raise RuntimeError("Every original batch must contain exactly16 images")
    for index, (row, old) in enumerate(zip(metadata, historical)):
        if (int(row["global_index"]), int(row["batch_index"]), int(row["source_index"])) != (batch * n + index, batch, index):
            raise RuntimeError("Original image order or batch assignment changed")
        if any(str(row[key]) != str(old[key]) for key in IDENTITY_KEYS):
            raise RuntimeError("Original image identity differs from historical M8")
    if len({int(row["label"]) for row in metadata}) != n:
        raise RuntimeError("Original batch labels must remain distinct")
    for key, shape in (("x", (n, 3, 224, 224)), ("Z_evaluation_only", (n, 25088))):
        tensor = inputs[key]
        if tuple(tensor.shape) != shape or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"Invalid original cached tensor: {key}")
    return metadata


def verify_inputs(manifest: dict, batch: int | None = None) -> None:
    for rel, expected in manifest["inputs_sha256"].items():
        path = resolve(rel)
        if batch is None or path.parent.name == f"batch{batch:02d}":
            if file_hash(path) != expected:
                raise RuntimeError(f"Frozen source changed: {rel}")


def prepare(config_path: Path, cfg: dict, output: Path) -> dict:
    baseline_path = PROJECT / "results/m10_cpa_fia_packed4_same80_B16/summary.json"
    baseline = read_json(baseline_path)
    if baseline["status"] != "complete" or baseline["n_images"] != 80:
        raise RuntimeError("The same80 packed4 reference must be complete")
    for rel, expected in baseline["identity"]["code_sha256"].items():
        if file_hash(WORKSPACE / rel) != expected:
            raise RuntimeError(f"The paired attack implementation changed: {rel}")
    if baseline["config"] != read_json(PROJECT / "configs/m10_cpa_fia_packed4_same80_B16.json"):
        raise RuntimeError("The paired packed4 configuration changed")
    topk50_path = PROJECT / "results/cpa_fia_topk50_same80_B16/summary.json"
    topk50 = read_json(topk50_path)
    if topk50["status"] != "complete" or topk50["n_images"] != 80:
        raise RuntimeError("The paired Top-k50 reference must be complete")
    if topk50["config"] != read_json(PROJECT / "configs/cpa_fia_topk50_same80_B16.json"):
        raise RuntimeError("The paired Top-k50 configuration changed")
    for rel, expected in topk50["identity"]["code_sha256"].items():
        if file_hash(WORKSPACE / rel) != expected:
            raise RuntimeError(f"The paired Top-k50 implementation changed: {rel}")
    source = resolve(cfg["source_output"])
    reference = resolve(cfg["historical_reference"])
    summary = read_json(source / "summary.json")
    old_summary = read_json(reference / "summary.json")
    if summary["status"] != "complete" or summary["n_images"] != 80 or old_summary["status"] != "complete" or old_summary["n_images"] != 80:
        raise RuntimeError("Complete historical80 source and M8 runs are required")
    source_cfg = summary["config"]
    if source_cfg["data"]["batch_indices"] != cfg["batch_indices"] or source_cfg["data"]["batch_size"] != 16:
        raise RuntimeError("Source batch configuration differs from the original16x5")
    training = source_cfg["uia"]
    if training["steps"] != 10 or training["lr"] != .001 or training["observation"] != "Y_delta_over_lr":
        raise RuntimeError("Expected saved actual T10 SGD0.001 updates")
    design_path = PROJECT / "configs/m10_compression_only_B16.protocol.json"
    design = read_json(design_path)
    for rel, expected in design["code_sha256_at_design"].items():
        if file_hash(WORKSPACE / rel) != expected:
            raise RuntimeError(f"Previously audited implementation changed: {rel}")
    weights = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
    weights_hash = file_hash(weights)
    if weights_hash != summary["identity"]["weights_sha256"]:
        raise RuntimeError("Expected the same pretrained VGG16 weights")
    identity = {"config_sha256": file_hash(config_path), "code_sha256": code_hashes(),
                "paired_packed4_summary_sha256": file_hash(baseline_path),
                "paired_topk50_summary_sha256": file_hash(topk50_path),
                "source_summary_sha256": file_hash(source / "summary.json"),
                "historical_summary_sha256": file_hash(reference / "summary.json"),
                "design_sha256": file_hash(design_path), "weights_sha256": weights_hash,
                "versions": {"torch": torch.__version__, "numpy": np.__version__}}
    manifest_path = output / "input_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest["identity"] != identity:
            raise RuntimeError("Frozen code/config/source changed; incompatible resume")
        verify_inputs(manifest)
        if file_hash(output / "image_manifest.csv") != manifest["image_manifest_sha256"]:
            raise RuntimeError("Frozen original image list changed")
        return manifest
    historical_path = reference / "per_image_metrics.csv"
    if file_hash(historical_path) != old_summary["output_hashes"]["per_image_metrics.csv"]:
        raise RuntimeError("Historical M8 identity table changed")
    historical = sorted(read_csv(historical_path), key=lambda row: int(row["global_index"]))
    if [int(row["global_index"]) for row in historical] != list(range(80)):
        raise RuntimeError("Historical table must cover exactly original IDs0-79")
    hashes = {str(path.relative_to(PROJECT)): file_hash(path) for path in
              (source / "summary.json", reference / "summary.json", historical_path, design_path, baseline_path, topk50_path)}
    metadata, batches = [], []
    declared = {int(row["batch_index"]): row for row in design["source_inputs"]}
    for batch in cfg["batch_indices"]:
        folder = source / f"batch{batch:02d}"
        done = read_json(folder / "summary.json")
        if done["n_images"] != 16 or done["oracle_fills"] != 0:
            raise RuntimeError("Each source batch must retain all16 original images")
        for name, design_key in (("inputs.pt", "declared_inputs_sha256"), ("observation.pt", "declared_observation_sha256")):
            path = folder / name
            meta_path = path.with_suffix(".meta.json")
            meta = read_json(meta_path)
            digest = file_hash(path)
            if digest != done["output_hashes"][name] or digest != meta["file_sha256"] or digest != declared[batch][design_key]:
                raise RuntimeError(f"Source checkpoint integrity failed: {path}")
            source_identity = meta["identity"]
            if source_identity["batch_index"] != batch or any(source_identity[key] != value for key, value in summary["identity"].items()):
                raise RuntimeError("Source checkpoint belongs to a different client/model")
            if name == "observation.pt" and source_identity["inputs_sha256"] != done["output_hashes"]["inputs.pt"]:
                raise RuntimeError("Saved update was collected on different images")
            hashes[str(path.relative_to(PROJECT))] = digest
            hashes[str(meta_path.relative_to(PROJECT))] = file_hash(meta_path)
        hashes[str((folder / "summary.json").relative_to(PROJECT))] = file_hash(folder / "summary.json")
        inputs = torch.load(folder / "inputs.pt", map_location="cpu", weights_only=False, mmap=True)
        metadata.extend(validate_source_images(inputs, historical[batch * 16:(batch + 1) * 16], batch))
        del inputs
        saved = torch.load(folder / "observation.pt", map_location="cpu", weights_only=False, mmap=True)
        actual_observation(saved["observation"], maximum_relative_error=.001)
        observation = saved["observation"]
        if observation["steps"] != 10 or observation["lr"] != .001:
            raise RuntimeError("Cached update training schedule changed")
        batches.append({"batch_index": batch, "global_indices": list(range(batch * 16, (batch + 1) * 16)),
                        "update_factor_rel_l2": float(observation["update_factor_rel_l2"]),
                        "keep_fraction": cfg["sparsification"]["keep_fraction"]})
        del saved, observation
        print(f"Verified unchanged original images and actual update: batch={batch}, n=16", flush=True)
    if len({row["image_path"] for row in metadata}) != 80 or len({int(row["dataset_index"]) for row in metadata}) != 80:
        raise RuntimeError("All80 source images must be distinct")
    write_csv(output / "image_manifest.csv", metadata)
    manifest = {"schema_version": 1, "run_tag": cfg["run_tag"], "n_images": 80,
                "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True).strip(),
                "identity": identity, "inputs_sha256": hashes, "metadata": metadata, "batches": batches,
                "selection": cfg["selection"], "weights_path": str(weights), "weights_sha256": weights_hash,
                "image_manifest_sha256": file_hash(output / "image_manifest.csv"),
                "new_training": False, "new_update_collection": False, "created_at_utc": utc_now()}
    write_json(manifest_path, manifest)
    write_json(config_path.with_suffix(".lock.json"), {"config_sha256": identity["config_sha256"],
               "input_manifest_sha256": file_hash(manifest_path), "frozen_at_utc": utc_now()})
    return manifest


def verify_completed(folder: Path, marker: str, identity: dict) -> dict | None:
    path = folder / marker
    if not path.is_file():
        return None
    saved = read_json(path)
    if saved["identity"] != identity or saved.get("batch_index") != int(folder.name.removeprefix("batch")):
        raise RuntimeError("Completed artifact identity or batch changed")
    required = {"observation.bin", "communication.json", "sparsification_diagnostics.json"}
    if marker == "summary.json":
        required |= {"sources.pt", "source_order_rec.pt", "rec.pt", "per_image_metrics.csv", "cpa_log.json", "fia_log.json", "packet_manifest.json"}
        if (saved.get("n"), saved.get("n_images"), saved.get("T"), saved.get("oracle_fills"), saved.get("ls_A")) != (16, 16, 10, 0, False):
            raise RuntimeError("Completed batch must retain the original full16 condition")
    if not required.issubset(saved["output_hashes"]):
        raise RuntimeError("Completed marker is missing required output hashes")
    for name, expected in saved["output_hashes"].items():
        if file_hash(folder / name) != expected:
            raise RuntimeError(f"Completed artifact changed: {folder / name}")
    return saved


def prepare_packet(cfg: dict, manifest: dict, batch: int, output: Path) -> None:
    folder = output / f"batch{batch:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    source = resolve(cfg["source_output"]) / f"batch{batch:02d}/observation.pt"
    source_hash = manifest["inputs_sha256"][str(source.relative_to(PROJECT))]
    previous = verify_completed(folder, "packet_manifest.json", manifest["identity"])
    if previous is not None:
        if previous.get("source_observation_sha256") != source_hash:
            raise RuntimeError("Prepared packet belongs to a different original update")
        return
    verify_inputs(manifest, batch)
    status(output, cfg, state="preparing_packets", batch_index=batch)
    started = time.perf_counter()
    saved = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    actual = actual_observation(saved["observation"], maximum_relative_error=.001)
    packet = encode_topk(actual, cfg["keep_percent"])
    packet_path = folder / "observation.bin"
    temporary = packet_path.with_suffix(".bin.tmp")
    temporary.write_bytes(packet)
    temporary.replace(packet_path)
    received = packet_path.read_bytes()
    if received != packet:
        raise RuntimeError("Written packet differs from encoded bytes")
    info = packet_info(received)
    if info["packet_bytes"] != cfg["expected_packet_bytes"] or info["codec_id"] != CODEC_ID:
        raise RuntimeError("Unexpected sparse packet format or size")
    decoded = decode_packet(received)
    diagnostics = sparsification_diagnostics(actual, received, decoded)
    diagnostics.update(update_collection_sec=0., new_update_collection=False,
                       update_factor_rel_l2=float(saved["observation"]["update_factor_rel_l2"]))
    elapsed = time.perf_counter() - started
    reference = info["fp32_reference_packet_bytes"]
    communication = {"codec": CODEC_ID, "value_bits": 32, "keep_fraction": cfg["sparsification"]["keep_fraction"],
                     "shape": info["shape"], "n_values": info["elements"], "kept_elements": info["kept_elements"],
                     "fp32_payload_bytes": 4 * info["elements"], "fp32_packet_bytes": reference,
                     "value_bytes": info["value_bytes"], "bitmap_bytes": info["bitmap_bytes"],
                     "header_bytes": info["header_bytes"], "packet_bytes": len(received),
                     "compression_ratio": reference / len(received), "saved_percent": 100 * (1 - len(received) / reference),
                     "scope": "attacked classifier.0.weight packet only", "codec_wall_time_sec": elapsed,
                     "transmitted_tensor_convention": "Y=(W_initial-W_final)/lr", "server_weight_delta": "-0.001*decoded_Y",
                     "sha256": hashlib.sha256(received).hexdigest(), "source_reconstruction_only_from_packet": True,
                     "quantization": False, "error_feedback": False}
    write_json(folder / "communication.json", communication)
    write_json(folder / "sparsification_diagnostics.json", diagnostics)
    write_json(folder / "packet_manifest.json", {"identity": manifest["identity"], "batch_index": batch,
               "source_observation_sha256": source_hash,
               "output_hashes": {name: file_hash(folder / name) for name in ("observation.bin", "communication.json", "sparsification_diagnostics.json")}})
    print(f"PACKET VERIFIED batch={batch}, bytes={len(received)}, retained={info['kept_elements']}/{info['elements']}", flush=True)
    del saved, actual, packet, received, decoded
    gc.collect()


def attack_from_packet(cfg: dict, batch: int, output: Path, device: torch.device) -> tuple:
    """This attack boundary accepts no true images, embeddings, or FP32 update."""
    folder, n = output / f"batch{batch:02d}", cfg["n"]
    decoded = decode_packet((folder / "observation.bin").read_bytes())
    if tuple(decoded.shape) != (4096, 25088):
        raise RuntimeError("Decoded observation shape mismatch")
    status(output, cfg, state="running", batch_index=batch, stage="CPA_from_received_packet")
    cpa = run_upstream_cpa(Y_hat=decoded, n_components=n, device=device, config=cfg["stage1"], output_dir=folder / "cpa", seed=cfg["seed"] + batch)
    sources = cpa["S_hat"].detach().cpu()
    if tuple(sources.shape) != (n, 25088) or not bool(torch.isfinite(sources).all()):
        raise RuntimeError("CPA must retain every finite source")
    del decoded
    torch.save(sources, folder / "sources.pt")
    status(output, cfg, state="running", batch_index=batch, stage="FIA_in_estimated_source_order")
    fia = run_upstream_fia(target_z=sources.abs().contiguous(), device=device, config=cfg["fia"], output_dir=folder / "fia", seed=cfg["fia"]["seed"])
    recs = fia["rec"].detach().cpu()
    if tuple(recs.shape) != (n, 3, 224, 224) or not bool(torch.isfinite(recs).all()):
        raise RuntimeError("FIA must retain every finite reconstruction")
    torch.save(recs, folder / "source_order_rec.pt")
    return sources, recs, cpa, fia


def run_batch(cfg: dict, manifest: dict, batch: int, output: Path, device: torch.device) -> None:
    folder = output / f"batch{batch:02d}"
    if verify_completed(folder, "summary.json", manifest["identity"]) is not None:
        return
    started = time.perf_counter()
    prepare_packet(cfg, manifest, batch, output)
    sources, source_recs, cpa, fia = attack_from_packet(cfg, batch, output, device)
    status(output, cfg, state="running", batch_index=batch, stage="evaluation_after_FIA")
    verify_inputs(manifest, batch)
    inputs = torch.load(resolve(cfg["source_output"]) / f"batch{batch:02d}/inputs.pt", map_location="cpu", weights_only=False, mmap=True)
    recs, correspondence = evaluate_after_fia(sources, source_recs, inputs["Z_evaluation_only"])
    metrics = evaluate_images(x_true=inputs["x"], rec=recs.to(device), device=device)
    rows = [{**inputs["metadata"][i], **correspondence[i], "run_tag": cfg["run_tag"], "n": cfg["n"], "T": cfg["T"],
             "observation_model": CODEC_ID, "ls_A": False, "oracle_fill": False,
             **{key: float(getattr(metrics, key)[i]) for key in ("lpips", "psnr", "ssim")}} for i in range(cfg["n"])]
    torch.save(recs, folder / "rec.pt")
    write_csv(folder / "per_image_metrics.csv", rows)
    write_json(folder / "cpa_log.json", cpa["log"])
    write_json(folder / "fia_log.json", fia["log"])
    communication = read_json(folder / "communication.json")
    names = ("sources.pt", "source_order_rec.pt", "rec.pt", "per_image_metrics.csv", "cpa_log.json", "fia_log.json",
             "observation.bin", "communication.json", "sparsification_diagnostics.json", "packet_manifest.json")
    done = {"identity": manifest["identity"], "batch_index": batch, "n_images": cfg["n"], "n": cfg["n"], "T": cfg["T"],
            "oracle_fills": 0, "ls_A": False, "embedding_cos09": sum(row["measured_recovered_cos09"] for row in rows),
            "source_cos_per_source": [row["source_abs_cosine"] for row in rows],
            **{f"{key}_mean": float(np.mean([row[key] for row in rows])) for key in ("lpips", "psnr", "ssim")},
            "runtime_sec": {"update_collection": 0., "codec": communication["codec_wall_time_sec"], "cpa": cpa["wall_time_sec"],
                            "fia": fia["wall_time_sec"], "invocation": time.perf_counter() - started},
            "completed_at_utc": utc_now(), "output_hashes": {name: file_hash(folder / name) for name in names}}
    write_json(folder / "summary.json", done)
    print(f"COMPLETE batch={batch}: LPIPS={done['lpips_mean']:.6f}, cosine09={done['embedding_cos09']}/{cfg['n']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("prepare", "packet", "run", "report"), default="prepare")
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
                status(output, cfg, state="prepared", n_images=80, batch_indices=cfg["batch_indices"], new_training=False)
                return
            if args.phase == "packet":
                for batch in cfg["batch_indices"]:
                    prepare_packet(cfg, manifest, batch, output)
                status(output, cfg, state="packet_ready", n_images=80, packets=5, new_training=False, attack_started=False)
                return
            if args.phase == "run":
                device = torch.device(cfg["device"])
                if device.type != "cuda" or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
                    raise RuntimeError("Expected the established sm_120 CUDA device")
                for batch in cfg["batch_indices"]:
                    run_batch(cfg, manifest, batch, output, device)
            from .report_topk_ratios_same80 import summarize
            result = summarize(cfg, manifest, output)
            status(output, cfg, state="complete", n_images=80, metrics=result["metrics"])
        except BaseException as exc:
            status(output, cfg, state="failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
