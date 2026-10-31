"""M7: raw observation rows -> unchanged FIA, with no source estimation.

Run from quantization_aware_cpa with ``python -m qacpa.run_m7_controls``.
All scientific parameters are read from the locked preregistration. Completed
FIA chunks can be resumed only with identical inputs, protocol and FIA code.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from .alignment import match_sources_by_abs_cosine
from .fia import denormalize_imagenet, evaluate_images, run_direct_fia
from .m7_protocol import PROJECT, file_sha256, read_protocol, write_json
from .run import append_jsonl, git_sha


NO_CPA = PROJECT / "results" / "m7_no_cpa_control"
PRIOR = PROJECT / "results" / "m7_prior_only"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def raw_observation_targets(observation: torch.Tensor, batch_index: int, seed: int = 0):
    """Only select predetermined rows and normalize; no truth argument exists."""
    if tuple(observation.shape) != (4096, 25088):
        raise ValueError(f"unexpected observation shape {tuple(observation.shape)}")
    generator = torch.Generator(device="cpu").manual_seed(seed + batch_index)
    row_indices = torch.randperm(4096, generator=generator)[:8]
    raw = observation.detach().cpu().index_select(0, row_indices).float().contiguous()
    return F.normalize(raw, p=2, dim=1, eps=1e-12), row_indices, raw.norm(dim=1)


def random_prior_targets() -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(42)
    return F.normalize(torch.rand(40, 25088, generator=generator), p=2, dim=1, eps=1e-12)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write empty metrics")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ensure_ledger_row(summary: dict) -> None:
    """Recover a completed run if an earlier process died before ledger append."""
    ledger = PROJECT / "results" / "runs.jsonl"
    if ledger.exists():
        with ledger.open(encoding="utf-8") as handle:
            for line in handle:
                if summary["run_tag"] not in line:
                    continue
                row = json.loads(line)
                if row.get("method") == summary["method"] and row.get("identity") == summary["identity"]:
                    return
    append_jsonl(ledger, summary)


def prepare() -> dict:
    protocol, lock = read_protocol()
    manifest_path = NO_CPA / "input_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["preregistration"]["sha256"] != lock["sha256"]:
            raise RuntimeError("existing M7 inputs use a different protocol")
        for saved in manifest["prepared_files"]:
            if file_sha256(Path(saved["path"])) != saved["sha256"]:
                raise RuntimeError(f"prepared input modified: {saved['path']}")
        return manifest

    NO_CPA.mkdir(parents=True, exist_ok=True)
    PRIOR.mkdir(parents=True, exist_ok=True)
    all_targets = {"fp32": [], "4bit": []}
    truth, true_z, metadata, row_ids, raw_norms = [], [], [], [], {"fp32": [], "4bit": []}
    sources = []
    for batch in range(5):
        victim_path = PROJECT / "cache" / f"victim_seed0_B8_batch{batch}.pt"
        quant_path = PROJECT / "cache" / f"quant_seed0_B8_batch{batch}_bits4_stochastic.pt"
        victim = torch.load(victim_path, map_location="cpu", weights_only=False)
        quant = torch.load(quant_path, map_location="cpu", weights_only=False)
        if int(victim["seed"]) != 0 or int(victim["batch_index"]) != batch:
            raise RuntimeError("historical victim cache identity mismatch")
        if len(set(victim["labels"].tolist())) != 8 or tuple(victim["x"].shape) != (8, 3, 224, 224):
            raise RuntimeError("historical B=8/distinct-label gate failed")
        if int(quant["bits"]) != 4 or quant["rounding"] != "stochastic":
            raise RuntimeError("historical 4-bit observation mismatch")
        if int(quant["seed"]) != 0 or int(quant["batch_index"]) != batch:
            raise RuntimeError("historical quantized cache identity mismatch")
        fp_target, fp_rows, fp_norm = raw_observation_targets(victim["Y"], batch)
        q_target, q_rows, q_norm = raw_observation_targets(quant["Y_hat"], batch)
        if not torch.equal(fp_rows, q_rows):
            raise AssertionError("precision conditions chose different rows")
        all_targets["fp32"].append(fp_target)
        all_targets["4bit"].append(q_target)
        raw_norms["fp32"].extend(fp_norm.tolist())
        raw_norms["4bit"].extend(q_norm.tolist())
        row_ids.extend(fp_rows.tolist())
        truth.append(victim["x"].float())
        true_z.append(victim["S_true"].float())
        per_sample_loss = victim.get("per_sample_loss")
        if per_sample_loss is None:
            per_sample_loss = F.cross_entropy(victim["logits"].float(), victim["labels"].long(), reduction="none")
        for index in range(8):
            metadata.append({
                "global_index": 8 * batch + index, "batch_index": batch,
                "source_index": index, "label": int(victim["labels"][index]),
                "image_path": victim["image_paths"][index],
                "per_sample_loss": float(per_sample_loss[index]),
            })
        sources.extend({"path": str(path), "sha256": file_sha256(path)} for path in (victim_path, quant_path))
        del victim, quant
        print(f"prepared victim batch {batch}: rows={fp_rows.tolist()}", flush=True)

    x = torch.cat(truth).contiguous()
    true_features = torch.cat(true_z).contiguous()
    # Verify identity against the archived float originals before any new FIA.
    from .m7_recompute import load_historical
    historical = load_historical()
    for value in historical.values():
        if not torch.equal(x, value["x"]):
            raise RuntimeError("M7 victim originals differ from historical 40-image row")
    del historical
    prepared_files = []
    for method in ("fp32", "4bit", "prior_only"):
        folder = PRIOR / method if method == "prior_only" else NO_CPA / method
        folder.mkdir(parents=True, exist_ok=True)
        targets = random_prior_targets() if method == "prior_only" else torch.cat(all_targets[method])
        payload = {
            "z": targets, "x": x, "S_true_evaluation_only": true_features,
            "metadata": metadata, "observation_rows": row_ids if method != "prior_only" else [-1] * 40,
            "raw_row_norms": raw_norms[method] if method != "prior_only" else [None] * 40,
            "protocol_sha256": lock["sha256"], "target_sha256": tensor_hash(targets),
        }
        path = folder / "targets.pt"
        torch.save(payload, path)
        prepared_files.append({"path": str(path), "sha256": file_sha256(path), "target_sha256": payload["target_sha256"]})
    truth_path = NO_CPA / "truth.pt"
    torch.save(denormalize_imagenet(x), truth_path)
    prepared_files.append({"path": str(truth_path), "sha256": file_sha256(truth_path)})
    manifest = {
        "schema_version": 1, "prepared_at_utc": utc_now(), "preregistration": lock,
        "protocol": protocol, "source_files": sources, "prepared_files": prepared_files,
        "row_indices_in_raw_order": row_ids,
        "grid_global_indices": np.random.default_rng(42).choice(40, 12, replace=False).tolist(),
        "pipeline": "observation row selection + row normalization + unmodified FIA",
        "no_cpa_zero_rows": {key: sum(norm == 0 for norm in values) for key, values in raw_norms.items()},
        "forbidden_stages_executed": [],
    }
    write_json(manifest_path, manifest)
    return manifest


def run_condition(method: str, device: torch.device, manifest: dict) -> None:
    start = time.perf_counter()
    protocol, lock = read_protocol()
    folder = PRIOR / method if method == "prior_only" else NO_CPA / method
    tag = "m7_prior_only" if method == "prior_only" else "m7_no_cpa_control"
    targets = torch.load(folder / "targets.pt", map_location="cpu", weights_only=False)
    fia_hash = file_sha256(PROJECT / "qacpa" / "fia.py")
    identity = {"protocol_sha256": lock["sha256"], "target_sha256": targets["target_sha256"], "fia_sha256": fia_hash}
    if tensor_hash(targets["z"]) != identity["target_sha256"]:
        raise RuntimeError("FIA target tensor hash mismatch")
    if (folder / "summary.json").exists():
        previous = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if previous["identity"] != identity:
            raise RuntimeError("existing condition has incompatible inputs/code")
        for filename, digest in previous["output_hashes"].items():
            if file_sha256(folder / filename) != digest:
                raise RuntimeError(f"completed condition output modified: {filename}")
        ensure_ledger_row(previous)
        print(f"verified completed condition {method}; skipping FIA", flush=True)
        return

    rec_chunks, logs = [], []
    for begin, end in protocol["fia"]["chunk_global_ranges"]:
        chunk_path = folder / f"chunk_{begin:02d}_{end:02d}.pt"
        write_json(NO_CPA / "execution_status.json", {
            "state": "running", "condition": method, "chunk_begin": begin, "chunk_end": end,
            "completed_images_in_condition": begin, "pid": __import__("os").getpid(),
            "updated_at_utc": utc_now(), "run_tag": tag,
        })
        chunk_identity = {**identity, "global_range": [begin, end]}
        if chunk_path.exists():
            saved = torch.load(chunk_path, map_location="cpu", weights_only=False)
            if saved["identity"] != chunk_identity or tensor_hash(saved["rec"]) != saved["rec_sha256"]:
                raise RuntimeError(f"checkpoint identity/integrity mismatch: {chunk_path}")
            print(f"resuming completed {method} chunk {begin}:{end}", flush=True)
        else:
            fi = protocol["fia"]
            result = run_direct_fia(
                target_z=targets["z"][begin:end], x_true=targets["x"][begin:end], device=device,
                n_iter=fi["n_iter"], log_every=fi["log_every"], lr=fi["lr"], tv=fi["tv"], fi=fi["fi"],
                optimizer=fi["optimizer"], scheduler=fi["scheduler"], seed=fi["seed"],
                progress_label=f"{tag}/{method}/{begin}:{end}",
            )
            saved = {"identity": chunk_identity, "rec": result.rec, "rec_sha256": tensor_hash(result.rec),
                     "log": result.log, "wall_time_sec": result.wall_time_sec}
            pending = chunk_path.with_suffix(".pt.tmp")
            torch.save(saved, pending)
            pending.replace(chunk_path)
            del result
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rec_chunks.append(saved["rec"])
        logs.append({"range": [begin, end], "wall_time_sec": saved["wall_time_sec"], "log": saved["log"]})

    raw_rec = torch.cat(rec_chunks).contiguous()
    # Evaluation assignment only, AFTER all targets have been inverted. This
    # existing helper's signs/scales are deliberately discarded.
    order, cosines, cosine_abs = [], [], []
    for batch in range(5):
        begin, end = batch * 8, batch * 8 + 8
        truth_z = targets["S_true_evaluation_only"][begin:end]
        target_z = targets["z"][begin:end]
        assignment = match_sources_by_abs_cosine(truth_z, target_z)
        if assignment.true_indices != list(range(8)):
            raise RuntimeError("incomplete B=8 evaluation assignment")
        selected = assignment.estimated_indices
        order.extend(begin + index for index in selected)
        signed = F.cosine_similarity(truth_z, target_z[selected], dim=1, eps=1e-12)
        cosines.extend(signed.tolist())
        cosine_abs.extend(signed.abs().tolist())
    aligned_rec = raw_rec[order].contiguous()
    matched_metrics = evaluate_images(x_true=targets["x"], rec=aligned_rec, device=device)
    positional_metrics = evaluate_images(x_true=targets["x"], rec=raw_rec, device=device)
    torch.save(raw_rec, folder / "rec_raw.pt")
    torch.save(aligned_rec, folder / "rec.pt")
    rows = []
    for index in range(40):
        row_position = order[index]
        rows.append({
            **targets["metadata"][index], "run_tag": tag, "method": method,
            "lpips": matched_metrics.lpips[index], "lpips_positional": positional_metrics.lpips[index],
            "psnr": matched_metrics.psnr[index], "ssim": matched_metrics.ssim[index],
            "source_cos": cosines[index], "source_cos_abs": cosine_abs[index],
            "raw_target_global_index": row_position,
            "observation_row": targets["observation_rows"][row_position],
            "raw_row_norm": targets["raw_row_norms"][row_position],
            "pairing": "evaluation-only permutation Hungarian abs target cosine; no sign/scale correction",
        })
    write_csv(folder / "per_image_metrics.csv", rows)
    write_json(folder / "fia_log.json", logs)
    per_batch = [{"batch_index": b, "n_images": 8,
                  "lpips_mean": float(np.mean(matched_metrics.lpips[b*8:(b+1)*8])),
                  "lpips_positional_mean": float(np.mean(positional_metrics.lpips[b*8:(b+1)*8])),
                  "source_cos_per_source": cosines[b*8:(b+1)*8]} for b in range(5)]
    summary = {
        "schema_version": 1, "run_tag": tag, "method": method, "git_sha": git_sha(),
        "identity": identity, "config": protocol, "preregistration": lock,
        "completed_at_utc": utc_now(), "n_images": 40,
        "lpips_mean": float(np.mean(matched_metrics.lpips)),
        "lpips_positional_mean": float(np.mean(positional_metrics.lpips)),
        "source_cos_per_source": cosines, "per_victim_batch": per_batch,
        "fia_wall_time_sec": sum(item["wall_time_sec"] for item in logs),
        "invocation_wall_time_sec": time.perf_counter() - start,
        "stage1": "not applicable: omitted", "stage2": "not applicable: prohibited",
        "input_manifest": str(NO_CPA / "input_manifest.json"),
        "output_hashes": {name: file_sha256(folder / name) for name in
                          ("rec.pt", "rec_raw.pt", "per_image_metrics.csv", "fia_log.json")},
    }
    write_json(folder / "summary.json", summary)
    ensure_ledger_row(summary)
    print(f"COMPLETE {method}: LPIPS={summary['lpips_mean']:.8f}, n=40", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "prepare", "no-cpa", "prior-only"), default="all")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    # An OS-held lock prevents accidentally starting concurrent M7 FIA queues;
    # it is released automatically even after a crash, so resume needs no cleanup.
    NO_CPA.mkdir(parents=True, exist_ok=True)
    process_lock = (NO_CPA / ".execution.lock").open("a+b")
    process_lock.seek(0, 2)
    if process_lock.tell() == 0:
        process_lock.write(b"0")
        process_lock.flush()
    process_lock.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(process_lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = utc_now()
    try:
        manifest = prepare()
        if args.phase == "prepare":
            return
        device = torch.device(args.device)
        if device.type == "cuda" and torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("M7 expected registered RTX 5060 Ti sm_120 device")
        for method in (("fp32", "4bit") if args.phase == "no-cpa" else
                       ("prior_only",) if args.phase == "prior-only" else ("fp32", "4bit", "prior_only")):
            run_condition(method, device, manifest)
        write_json(NO_CPA / "execution_status.json", {
            "state": "fia_complete", "phase": args.phase, "started_at_utc": started,
            "updated_at_utc": utc_now(), "pid": __import__("os").getpid(),
        })
        # Reporting contains no FIA, Stage 1, or Stage 2. Missing conditions stay pending.
        from .m7_report import main as report_main
        import sys
        sys.argv = ["qacpa.m7_report"]
        report_main()
    except BaseException as exc:
        write_json(NO_CPA / "execution_status.json", {
            "state": "failed", "phase": args.phase, "started_at_utc": started,
            "updated_at_utc": utc_now(), "error": str(exc), "traceback": traceback.format_exc(),
        })
        raise
    finally:
        process_lock.close()


if __name__ == "__main__":
    main()
