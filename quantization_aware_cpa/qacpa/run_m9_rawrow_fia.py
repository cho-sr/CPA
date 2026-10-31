"""Observed update rows + abs + original FIA; no source separation.

M9 reuses M8's exact observations and FIA backend. It is a declared raw-row
inversion baseline, not dequantization alone or a uniquely matched CPA ablation.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .fia import evaluate_images
from .run_m8_fedavg_naive import (
    PROJECT, WORKSPACE, evaluate_after_fia, execution_lock, file_hash,
    read_json, resolve, utc_now, write_csv, write_json,
)
from .upstream_naive import run_upstream_fia


DEFAULT_CONFIG = PROJECT / "configs/m9_fedavg_rawrow_dither4_B16.json"
FORBIDDEN = ("cpa", "whitening", "ls_A", "debias", "mp_truncation", "mp_diagnostic", "stage2", "refine_A", "new_training", "oracle_fills")


def raw_row_targets(Y_hat: torch.Tensor, batch_index: int, seed: int = 0, n: int = 16):
    """Select observed rows without reading truth, fitting a model, or normalizing."""
    if Y_hat.ndim != 2 or not 0 < n <= Y_hat.shape[0] or Y_hat.device.type != "cpu":
        raise ValueError("Expected a CPU observation matrix and a valid row count")
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + int(batch_index))
    row_ids = torch.randperm(Y_hat.shape[0], generator=generator)[:n]
    raw = Y_hat.detach().index_select(0, row_ids).float().contiguous()
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("Selected observation rows contain non-finite values")
    return raw, raw.abs().contiguous(), row_ids


def validate_config(cfg: dict[str, Any]) -> None:
    if any(cfg.get(key) is not False for key in FORBIDDEN):
        raise ValueError("CPA, whitening, LS-A, debias, MP, refinement and training must all be false")
    if (cfg["n"], cfg["T"], cfg["n_images"], cfg["batch_indices"], cfg["seed"]) != (16, 10, 80, list(range(5)), 0):
        raise ValueError("Keep the registered same80 n16/T10 sampling and row-selection seed")
    expected = {"n_iter": 25000, "log_every": 5000, "lr": .1, "tv": .1, "fi": 1., "optimizer": "adam", "scheduler": "cosine", "seed": 42}
    if cfg["fia"] != expected or cfg["grid"] != {"seed": 42, "n": 12}:
        raise ValueError("FIA and figure sampling settings must remain fixed")


def code_hashes() -> dict[str, str]:
    paths = [Path(__file__), PROJECT / "qacpa/report_m9_rawrow_fia.py", PROJECT / "qacpa/run_m8_fedavg_naive.py",
             PROJECT / "qacpa/upstream_naive.py", PROJECT / "qacpa/seed.py", PROJECT / "qacpa/fia.py",
             PROJECT / "scripts/launch_m9_rawrow_fia.ps1"]
    paths.extend(sorted((WORKSPACE / "official_cpa/src").glob("*.py")))
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in paths}


def status(output: Path, cfg: dict[str, Any], **state: Any) -> None:
    write_json(output / "execution_status.json", {"run_tag": cfg["run_tag"], "pid": os.getpid(), "updated_at_utc": utc_now(), **state})


def verify_frozen_code(config_path: Path, cfg: dict[str, Any], manifest: dict[str, Any]) -> None:
    identity = manifest["identity"]
    if (file_hash(config_path) != identity["config_sha256"]
            or file_hash(resolve(cfg["protocol"])) != identity["protocol_sha256"]
            or code_hashes() != identity["code_sha256"]):
        raise RuntimeError("Frozen code, configuration or protocol changed during execution")


def prepare(config_path: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    source, reference = resolve(cfg["source_output"]), resolve(cfg["reference_output"])
    source_summary, ref_summary = read_json(source / "summary.json"), read_json(reference / "summary.json")
    ref_manifest = read_json(reference / "input_manifest.json")
    for summary in (source_summary, ref_summary):
        if summary["status"] != "complete" or summary["n_images"] != 80 or summary["oracle_fills"] != 0:
            raise RuntimeError("Complete unfilled same80 source/reference runs are required")
    if cfg["fia"] != ref_summary["config"]["fia"] or cfg["source_output"] != ref_summary["config"]["source_output"]:
        raise RuntimeError("M9 must share M8's exact observation source and FIA settings")
    protocol = resolve(cfg["protocol"])
    identity = {"config_sha256": file_hash(config_path), "code_sha256": code_hashes(),
                "protocol_sha256": file_hash(protocol), "source_summary_sha256": file_hash(source / "summary.json"),
                "reference_summary_sha256": file_hash(reference / "summary.json")}
    for relative, expected in ref_manifest["identity"]["code_sha256"].items():
        if file_hash(WORKSPACE / relative) != expected:
            raise RuntimeError(f"M8 backend changed: {relative}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "input_manifest.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing["identity"] != identity:
            raise RuntimeError("Frozen M9 code/config/protocol or references changed")
        for relative, expected in existing["inputs_sha256"].items():
            if file_hash(resolve(relative)) != expected:
                raise RuntimeError(f"Frozen M9 input changed: {relative}")
        return existing
    inputs: dict[str, str] = {}
    def freeze(path: Path, expected: str | None = None) -> str:
        digest = file_hash(path)
        if expected is not None and digest != expected:
            raise RuntimeError(f"Archived input hash mismatch: {path}")
        inputs[str(path.relative_to(PROJECT))] = digest
        return digest
    frozen_rows = {}
    for batch in cfg["batch_indices"]:
        folder = source / f"batch{batch:02d}"
        bs = read_json(folder / "summary.json")
        if bs["n_images"] != 16 or bs["oracle_fills"] != 0:
            raise RuntimeError("Incomplete observation batch")
        for name in ("inputs.pt", "quantized.pt"):
            path = folder / name
            digest = freeze(path, bs["output_hashes"][name])
            if digest != ref_manifest["inputs_sha256"][str(path.relative_to(PROJECT))]:
                raise RuntimeError("M9 observation differs from M8 frozen observation")
            meta = folder / name.replace(".pt", ".meta.json")
            if read_json(meta)["file_sha256"] != digest:
                raise RuntimeError("Input metadata/hash mismatch")
            freeze(meta)
        freeze(folder / "summary.json")
        ref_folder = reference / f"batch{batch:02d}"
        rb = read_json(ref_folder / "summary.json")
        if rb["n_images"] != 16 or rb["identity"] != ref_summary["identity"]:
            raise RuntimeError("Incomplete M8 reference batch")
        freeze(ref_folder / "summary.json")
        for name in ("rec.pt", "per_image_metrics.csv"):
            freeze(ref_folder / name, rb["output_hashes"][name])
        generator = torch.Generator(device="cpu").manual_seed(cfg["seed"] + batch)
        frozen_rows[str(batch)] = torch.randperm(4096, generator=generator)[:16].tolist()
        print(f"Verified identical M8 observation and preregistered rows: batch={batch}", flush=True)
    for folder, summary in ((source, source_summary), (reference, ref_summary)):
        freeze(folder / "summary.json")
        freeze(folder / "per_image_metrics.csv", summary["output_hashes"]["per_image_metrics.csv"])
    freeze(reference / "input_manifest.json")
    freeze(reference / "all80_rec.pt", ref_summary["output_hashes"]["all80_rec.pt"])
    weights = Path(ref_manifest["weights_path"])
    if file_hash(weights) != ref_manifest["weights_sha256"]:
        raise RuntimeError("Pretrained weights changed")
    manifest = {"run_tag": cfg["run_tag"], "identity": identity, "created_at_utc": utc_now(),
                "inputs_sha256": inputs, "weights_path": str(weights), "weights_sha256": ref_manifest["weights_sha256"],
                "frozen_row_indices": frozen_rows, "row_selection_seed": cfg["seed"],
                "random_grid_indices": np.random.default_rng(42).choice(80, 12, replace=False).tolist(),
                "forbidden_operations": list(FORBIDDEN), "truth_inputs_used_for_selection": False}
    write_json(manifest_path, manifest)
    return manifest


def verify_batch_source(cfg: dict[str, Any], manifest: dict[str, Any], batch: int) -> Path:
    folder = resolve(cfg["source_output"]) / f"batch{batch:02d}"
    for relative, expected in manifest["inputs_sha256"].items():
        path = resolve(relative)
        if path.parent == folder and file_hash(path) != expected:
            raise RuntimeError(f"Frozen source changed: {path}")
    return folder


def run_batch(cfg: dict[str, Any], manifest: dict[str, Any], batch: int, output: Path, device: torch.device) -> dict[str, Any]:
    folder = output / f"batch{batch:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "summary.json"
    if marker.exists():
        done = read_json(marker)
        if done["identity"] != manifest["identity"] or done["n_images"] != 16:
            raise RuntimeError("Completed batch has incompatible identity")
        for name, digest in done["output_hashes"].items():
            if file_hash(folder / name) != digest:
                raise RuntimeError("Completed batch output hash mismatch")
        return done
    source = verify_batch_source(cfg, manifest, batch)
    started = time.perf_counter()
    status(output, cfg, state="running", batch_index=batch, stage="fixed_raw_row_selection_no_CPA")
    observation = torch.load(source / "quantized.pt", map_location="cpu", weights_only=False, mmap=True)
    Y_hat = observation["Y_hat"]
    if tuple(Y_hat.shape) != (4096, 25088):
        raise RuntimeError("Expected saved actual 4096x25088 FedAvg observation")
    sources, targets, row_ids = raw_row_targets(Y_hat, batch, cfg["seed"], cfg["n"])
    if row_ids.tolist() != manifest["frozen_row_indices"][str(batch)]:
        raise RuntimeError("Row selector differs from the preregistered indices")
    zero_rows = sources.abs().amax(dim=1) == 0
    target_metadata = {
        "batch_index": batch, "selection_seed": cfg["seed"] + batch, "observation_rows": row_ids.tolist(),
        "zero_raw_rows": zero_rows.tolist(), "zero_raw_row_count": int(zero_rows.sum()),
        "raw_row_l2_norms_diagnostic_only": sources.norm(dim=1).tolist(),
        "row_selection_informed_by_truth": False, "adapter": "abs(raw rows); no normalization",
        "forbidden_operations_executed": [], "input_sha256": file_hash(source / "quantized.pt"),
    }
    torch.save(sources, folder / "sources.pt")
    torch.save(targets, folder / "target_z.pt")
    write_json(folder / "target_metadata.json", target_metadata)
    del observation, Y_hat
    status(output, cfg, state="running", batch_index=batch, stage="original_FIA_on_abs_raw_rows")
    # No original image or true embedding has been loaded before this call.
    fia = run_upstream_fia(target_z=targets, device=device, config=cfg["fia"], output_dir=folder / "fia", seed=cfg["fia"]["seed"])
    source_recs = fia["rec"].detach().cpu()
    if tuple(source_recs.shape) != (16, 3, 224, 224) or not bool(torch.isfinite(source_recs).all()):
        raise RuntimeError("Expected all16 finite FIA reconstructions")
    torch.save(source_recs, folder / "source_order_rec.pt")
    write_json(folder / "fia_log.json", fia["log"])
    status(output, cfg, state="running", batch_index=batch, stage="evaluation_after_FIA")
    inputs = torch.load(source / "inputs.pt", map_location="cpu", weights_only=False, mmap=True)
    recs, correspondence = evaluate_after_fia(sources, source_recs, inputs["Z_evaluation_only"])
    metrics = evaluate_images(x_true=inputs["x"], rec=recs.to(device), device=device)
    order = torch.tensor([r["estimated_source_index"] for r in correspondence])
    abs_target_cos = F.cosine_similarity(inputs["Z_evaluation_only"].float(), targets[order], dim=1, eps=1e-12)
    rows = []
    for i in range(16):
        candidate = correspondence[i]["estimated_source_index"]
        rows.append({**inputs["metadata"][i], **correspondence[i], "run_tag": cfg["run_tag"], "n": 16, "T": 10,
                     "observation_model": "per_row_dither4", "cpa": False, "ls_A": False, "debias": False, "mp": False,
                     "oracle_fill": False, "observation_row": int(row_ids[candidate]), "zero_raw_row": bool(zero_rows[candidate]),
                     "fia_target_cosine": float(abs_target_cos[i]),
                     **{key: float(getattr(metrics, key)[i]) for key in ("lpips", "psnr", "ssim")}})
    if [int(r["global_index"]) for r in rows] != list(range(batch * 16, (batch + 1) * 16)):
        raise RuntimeError("Unexpected evaluated image identity/order")
    torch.save(recs, folder / "rec.pt")
    write_csv(folder / "per_image_metrics.csv", rows)
    names = ("sources.pt", "target_z.pt", "source_order_rec.pt", "rec.pt", "per_image_metrics.csv", "fia_log.json", "target_metadata.json")
    done = {"identity": manifest["identity"], "batch_index": batch, "n_images": 16, "oracle_fills": 0,
            "target_cos09_count": sum(float(r["source_abs_cosine"]) > .9 for r in rows),
            "zero_raw_row_count": int(zero_rows.sum()), "forbidden_operations_executed": [],
            **{f"{key}_mean": float(np.mean([r[key] for r in rows])) for key in ("lpips", "psnr", "ssim")},
            "runtime_sec": {"fia": fia["wall_time_sec"], "invocation": time.perf_counter() - started},
            "completed_at_utc": utc_now(), "output_hashes": {name: file_hash(folder / name) for name in names}}
    write_json(marker, done)
    print(f"COMPLETE raw-row batch={batch}: LPIPS={done['lpips_mean']:.6f}, all16 retained", flush=True)
    return done


def publish_status(cfg: dict[str, Any], output: Path, state: str) -> None:
    path = WORKSPACE / "codex/STATUS.md"
    document = path.read_text(encoding="utf-8")
    begin, end = "<!-- M9_RAWROW_STATUS_BEGIN -->", "<!-- M9_RAWROW_STATUS_END -->"
    if document.count(begin) != 1 or document.count(end) != 1:
        raise RuntimeError("Expected one preregistered M9 status block")
    lines = [begin, "## M9 FedAvg raw-row FIA", "", f"- State: **{state.upper()}**; run tag `{cfg['run_tag']}`.",
             "- Same80 n16/T10 cached M8 dither4 observations; fixed raw-row selection + abs + original FIA.",
             "- No CPA, whitening, debias, MP, LS-A, embedding refinement, oracle fills, or new training.",
             "- [Preregistered protocol](docs/FEDAVG_RAWROW_FIA_PROTOCOL.md).",
             "- Original-image correspondence is evaluated only after FIA; this is a declared raw-row inversion baseline.",
             "- The invalid historical M7 no-CPA results remain withdrawn."]
    if state == "complete":
        result = read_json(output / "summary.json")
        lines.append(f"- All80 results and landscape PNGs: [report](../quantization_aware_cpa/results/{cfg['run_tag']}/report.md).")
        if "metrics" in result:
            values = result["metrics"]
            lines.append(f"- LPIPS **{values['lpips_mean']:.6f}**, PSNR **{values['psnr_mean']:.3f}**, SSIM **{values['ssim_mean']:.4f}**.")
    lines.append(end)
    start, finish = document.index(begin), document.index(end) + len(end)
    path.write_text(document[:start] + "\n".join(lines) + document[finish:], encoding="utf-8")


def publish_status_safely(cfg: dict[str, Any], output: Path, state: str) -> None:
    """Document publication must not turn valid experiment outputs into a failure."""
    publication = {"started_at_utc": utc_now(), "experiment_state": state}
    try:
        publish_status(cfg, output, state)
        publication["state"] = "complete"
    except Exception as exc:
        publication.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        print(f"STATUS publication failed; experiment state is preserved: {exc}", flush=True)
    publication["finished_at_utc"] = utc_now()
    write_json(output / "postprocess_status.json", publication)


def worker(config_path: Path, cfg: dict[str, Any], output: Path) -> int:
    state = {"run_tag": cfg["run_tag"], "state": "starting", "worker_pid": os.getpid(), "started_at_utc": utc_now()}
    write_json(output / "queue_status.json", state)
    command = [sys.executable, "-u", "-m", "qacpa.run_m9_rawrow_fia", "--config", str(config_path), "--phase", "run"]
    code = 1
    try:
        with (output / "run.stdout.log").open("a", encoding="utf-8") as out, (output / "run.stderr.log").open("a", encoding="utf-8") as err:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=out, stderr=err,
                                       env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            state.update(state="running", runner_pid=process.pid)
            write_json(output / "queue_status.json", state)
            code = int(process.wait())
        state.update(runner_exit_code=code, state="failed")
        if code == 0:
            result = read_json(output / "summary.json")
            if result["status"] != "complete" or result["n_images"] != 80 or result["identity"]["config_sha256"] != file_hash(config_path):
                raise RuntimeError("Runner did not produce the complete registered result")
            if not all(file_hash(output / name) == digest for name, digest in result["output_hashes"].items()):
                raise RuntimeError("Final output hash verification failed")
            state["state"] = "complete"
    except BaseException as exc:
        state.update(state="failed", error=f"{type(exc).__name__}: {exc}")
    state["finished_at_utc"] = utc_now()
    write_json(output / "queue_status.json", state)
    postprocess = {"started_at_utc": utc_now()}
    try:
        publish_status(cfg, output, state["state"])
        postprocess.update(state="complete", finished_at_utc=utc_now())
    except Exception as exc:
        postprocess.update(state="failed", error=str(exc), finished_at_utc=utc_now())
    write_json(output / "postprocess_status.json", postprocess)
    return 0 if state["state"] == "complete" else (code or 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("prepare", "run", "report", "worker"), default="prepare")
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
                status(output, cfg, state="prepared")
                publish_status_safely(cfg, output, "prepared")
                print("Prepared raw-row baseline; all80 row IDs frozen before FIA", flush=True)
                return
            if args.phase == "run":
                device = torch.device(cfg["device"])
                if device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable")
                if file_hash(Path(manifest["weights_path"])) != manifest["weights_sha256"]:
                    raise RuntimeError("Frozen pretrained weights changed")
                publish_status_safely(cfg, output, "running")
                for batch in cfg["batch_indices"]:
                    verify_frozen_code(config_path, cfg, manifest)
                    run_batch(cfg, manifest, batch, output, device)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            verify_frozen_code(config_path, cfg, manifest)
            from .report_m9_rawrow_fia import build_report
            result = build_report(cfg, manifest, output)
            status(output, cfg, state="complete", n_images=80, metrics=result.get("metrics", {}))
            publish_status_safely(cfg, output, "complete")
        except BaseException as exc:
            status(output, cfg, state="failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
