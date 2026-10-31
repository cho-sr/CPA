"""FedAvg naive CPA: direct upstream CPA/FIA, with no mixing-matrix estimate.

Reuses the exact M7 dither observations. This is not an integer quantizer test.
True sources and images are used only after image optimization for evaluation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from .fia import denormalize_imagenet, evaluate_images
from .upstream_naive import run_upstream_cpa, run_upstream_fia

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DEFAULT_CONFIG = PROJECT / "configs/m8_fedavg_naive_cpa_dither4_B16.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_config(cfg: dict[str, Any]) -> None:
    for name in ("ls_A", "debias", "mp_truncation", "mp_diagnostic", "stage2", "refine_A", "new_training", "oracle_fills"):
        if cfg.get(name) is not False:
            raise ValueError(f"{name} must be explicitly false for this baseline")
    if (cfg["n"], cfg["T"], cfg["n_images"], cfg["batch_indices"]) != (16, 10, 80, list(range(5))):
        raise ValueError("This registered baseline requires the existing five n16/T10 batches")
    if cfg["stage1"]["cpa_ica_max_iter"] != 25000 or cfg["fia"]["n_iter"] != 25000 or cfg["fia"]["tv"] != .1:
        raise ValueError("Keep the existing CPA/FIA budget and FIA TV fixed")
    if cfg["grid"] != {"seed": 42, "n": 12}:
        raise ValueError("Grid selection must remain independent of results")


def code_hashes() -> dict[str, str]:
    paths = [Path(__file__), PROJECT / "qacpa/upstream_naive.py", PROJECT / "qacpa/seed.py", PROJECT / "qacpa/fia.py", PROJECT / "scripts/launch_m8_fedavg_naive.ps1"]
    paths.extend(sorted((WORKSPACE / "official_cpa/src").glob("*.py")))
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in paths}


def prepare(config_path: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    source = resolve(cfg["source_output"])
    summary = read_json(source / "summary.json")
    if summary["status"] != "complete" or summary["n_images"] != 80 or summary["oracle_fills"] != 0:
        raise RuntimeError("Complete, unfilled 80-image source run is required")
    source_cfg = summary["config"]
    if source_cfg["uia"]["observation"] != "Y_delta_over_lr" or source_cfg["uia"]["steps"] != 10:
        raise RuntimeError("Expected cached actual FedAvg updates")
    if source_cfg["quantization"]["bits"] != 4 or source_cfg["quantization"]["scale_granularity"] != "per_row":
        raise RuntimeError("Expected the existing per-row dither4 observations")
    output.mkdir(parents=True, exist_ok=True)
    identity = {"config_sha256": file_hash(config_path), "code_sha256": code_hashes(), "source_summary_sha256": file_hash(source / "summary.json")}
    manifest_path = output / "input_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest["identity"] != identity:
            raise RuntimeError("Code/config/source changed after freeze; use a new output directory")
        return manifest
    weights = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
    weights_hash = file_hash(weights)
    if weights_hash != summary["identity"]["weights_sha256"]:
        raise RuntimeError("Expected the same cached pretrained VGG16 weights")
    inputs = {}
    for batch in cfg["batch_indices"]:
        folder = source / f"batch{batch:02d}"
        batch_summary = read_json(folder / "summary.json")
        if batch_summary["n_images"] != 16 or batch_summary["oracle_fills"] != 0:
            raise RuntimeError("Incomplete source batch")
        for name in ("inputs.pt", "quantized.pt"):
            digest = file_hash(folder / name)
            meta = read_json(folder / name.replace(".pt", ".meta.json"))
            if digest != batch_summary["output_hashes"][name] or digest != meta["file_sha256"]:
                raise RuntimeError(f"Source hash mismatch: {folder / name}")
            inputs[str((folder / name).relative_to(PROJECT))] = digest
        for name in ("summary.json", "inputs.meta.json", "quantized.meta.json"):
            inputs[str((folder / name).relative_to(PROJECT))] = file_hash(folder / name)
        print(f"Verified same cached observation: batch={batch}", flush=True)
    for name, expected in summary["output_hashes"].items():
        if name == "per_image_metrics.csv" and file_hash(source / name) != expected:
            raise RuntimeError("Historical metric file changed")
    inputs[str((source / "per_image_metrics.csv").relative_to(PROJECT))] = file_hash(source / "per_image_metrics.csv")
    manifest = {"run_tag": cfg["run_tag"], "identity": identity, "inputs_sha256": inputs,
                "weights_path": str(weights), "weights_sha256": weights_hash, "created_at_utc": utc_now(),
                "random_grid_indices": np.random.default_rng(42).choice(80, 12, replace=False).tolist()}
    write_json(manifest_path, manifest)
    return manifest


def verify_batch_source(cfg: dict[str, Any], manifest: dict[str, Any], batch: int) -> Path:
    folder = resolve(cfg["source_output"]) / f"batch{batch:02d}"
    for path_string, expected in manifest["inputs_sha256"].items():
        path = resolve(path_string)
        if path.parent == folder and file_hash(path) != expected:
            raise RuntimeError(f"Frozen input changed: {path}")
    return folder


def evaluate_after_fia(sources: torch.Tensor, source_recs: torch.Tensor, truth: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Evaluation correspondence only: never return scaled/replaced source targets."""
    if sources.shape != truth.shape or len(sources) != len(source_recs):
        raise ValueError("Every estimated source and reconstruction must be present")
    similarity = (F.normalize(truth.detach().cpu(), dim=1, eps=1e-12) @ F.normalize(sources.detach().cpu(), dim=1, eps=1e-12).T).abs()
    true_indices, estimated_indices = linear_sum_assignment(-similarity.numpy())
    if true_indices.tolist() != list(range(len(truth))) or sorted(estimated_indices.tolist()) != list(range(len(truth))):
        raise RuntimeError("Incomplete evaluation correspondence")
    order = torch.tensor(estimated_indices, dtype=torch.long)
    aligned_recs = source_recs.index_select(0, order)
    rows = [{"source_index": i, "estimated_source_index": int(estimated_indices[i]),
             "source_abs_cosine": float(similarity[i, estimated_indices[i]]),
             "measured_recovered_cos09": int(similarity[i, estimated_indices[i]] > .9)} for i in range(len(truth))]
    return aligned_recs, rows


def status(output: Path, cfg: dict[str, Any], **kwargs: Any) -> None:
    write_json(output / "execution_status.json", {"run_tag": cfg["run_tag"], "pid": os.getpid(), "updated_at_utc": utc_now(), **kwargs})


def run_batch(cfg: dict[str, Any], manifest: dict[str, Any], batch: int, output: Path, device: torch.device) -> dict[str, Any]:
    folder = output / f"batch{batch:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "summary.json"
    if marker.is_file():
        done = read_json(marker)
        if done["identity"] != manifest["identity"] or done["n_images"] != 16:
            raise RuntimeError("Completed batch has different identity")
        for name, digest in done["output_hashes"].items():
            if file_hash(folder / name) != digest:
                raise RuntimeError("Completed batch changed")
        return done
    source = verify_batch_source(cfg, manifest, batch)
    started = time.perf_counter()
    observation = torch.load(source / "quantized.pt", map_location="cpu", weights_only=False, mmap=True)
    Y_hat = observation["Y_hat"]
    if tuple(Y_hat.shape) != (4096, 25088) or not bool(torch.isfinite(Y_hat).all()):
        raise RuntimeError("Invalid cached observation")
    status(output, cfg, state="running", batch_index=batch, stage="original_CPA_no_A")
    cpa = run_upstream_cpa(Y_hat=Y_hat, n_components=16, device=device, config=cfg["stage1"], output_dir=folder / "cpa", seed=cfg["seed"] + batch)
    sources = cpa["S_hat"].detach().cpu()
    if tuple(sources.shape) != (16, 25088) or not bool(torch.isfinite(sources).all()):
        raise RuntimeError("Original CPA must return all16 finite sources")
    del observation, Y_hat
    torch.save(sources, folder / "sources.pt")
    status(output, cfg, state="running", batch_index=batch, stage="original_FIA_source_order")
    # This is the entire attack target construction. No ground truth is loaded yet.
    fia = run_upstream_fia(target_z=sources.abs().contiguous(), device=device, config=cfg["fia"], output_dir=folder / "fia", seed=cfg["fia"]["seed"])
    source_recs = fia["rec"].detach().cpu()
    if tuple(source_recs.shape) != (16, 3, 224, 224) or not bool(torch.isfinite(source_recs).all()):
        raise RuntimeError("Original FIA must return all16 finite reconstructions")
    torch.save(source_recs, folder / "source_order_rec.pt")
    status(output, cfg, state="running", batch_index=batch, stage="evaluation_after_FIA")
    inputs = torch.load(source / "inputs.pt", map_location="cpu", weights_only=False, mmap=True)
    recs, correspondence = evaluate_after_fia(sources, source_recs, inputs["Z_evaluation_only"])
    metrics = evaluate_images(x_true=inputs["x"], rec=recs.to(device), device=device)
    rows = [{**inputs["metadata"][i], **correspondence[i], "run_tag": cfg["run_tag"], "n": 16, "T": 10,
             "observation_model": "per_row_dither4", "ls_A": False, "oracle_fill": False,
             **{key: float(getattr(metrics, key)[i]) for key in ("lpips", "psnr", "ssim")}} for i in range(16)]
    if [int(r["global_index"]) for r in rows] != list(range(batch * 16, (batch + 1) * 16)):
        raise RuntimeError("Unexpected image identity/order")
    torch.save(recs, folder / "rec.pt")
    write_csv(folder / "per_image_metrics.csv", rows)
    write_json(folder / "cpa_log.json", cpa["log"])
    write_json(folder / "fia_log.json", fia["log"])
    names = ("sources.pt", "source_order_rec.pt", "rec.pt", "per_image_metrics.csv", "cpa_log.json", "fia_log.json")
    done = {"identity": manifest["identity"], "batch_index": batch, "n_images": 16, "oracle_fills": 0, "ls_A": False,
            "embedding_cos09": sum(r["measured_recovered_cos09"] for r in rows),
            **{f"{key}_mean": float(np.mean([r[key] for r in rows])) for key in ("lpips", "psnr", "ssim")},
            "runtime_sec": {"cpa": cpa["wall_time_sec"], "fia": fia["wall_time_sec"], "invocation": time.perf_counter() - started},
            "completed_at_utc": utc_now(), "output_hashes": {name: file_hash(folder / name) for name in names}}
    write_json(marker, done)
    print(f"COMPLETE batch={batch}: LPIPS={done['lpips_mean']:.6f}, cos09={done['embedding_cos09']}/16", flush=True)
    return done


def render_grid(originals: torch.Tensor, recs: torch.Tensor, rows: list[dict[str, Any]], indices: list[int], path: Path, columns: int) -> None:
    from PIL import Image, ImageDraw, ImageFont
    tile, gap, header, labels = 224, 16, 96, 35
    block = 2 * tile + gap
    nrows = (len(indices) + columns - 1) // columns
    canvas = Image.new("RGB", (gap + columns * (block + gap), header + nrows * (tile + labels + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    draw.text((gap, 12), f"FedAvg naive CPA + FIA | n16 T10 | {len(indices)} of80 | original / reconstruction", font=font, fill="black")
    draw.text((gap, 45), "Direct upstream classes; dither4 observation | no LS-A, debias, MP or Stage2 | not cherry-picked", font=font, fill="#475569")
    truth = denormalize_imagenet(originals)
    for pos, index in enumerate(indices):
        left = gap + (pos % columns) * (block + gap)
        top = header + (pos // columns) * (tile + labels + gap)
        draw.text((left, top), f"g{index:02d} LPIPS={float(rows[index]['lpips']):.3f} cos={float(rows[index]['source_abs_cosine']):.3f}", font=font, fill="black")
        for col, tensor in enumerate((truth[index], recs[index])):
            array = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            canvas.paste(Image.fromarray(array).resize((tile, tile), Image.Resampling.LANCZOS), (left + col * (tile + gap), top + labels))
    canvas.save(path)


def summarize(cfg: dict[str, Any], manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    rows, batches, originals, recs = [], [], [], []
    for name, expected in manifest["inputs_sha256"].items():
        path = resolve(name)
        if path.name in ("inputs.pt", "per_image_metrics.csv") and file_hash(path) != expected:
            raise RuntimeError(f"Evaluation input changed after freeze: {path}")
    for batch in cfg["batch_indices"]:
        folder = output / f"batch{batch:02d}"
        done = read_json(folder / "summary.json")
        if done["identity"] != manifest["identity"] or done["n_images"] != 16 or done["oracle_fills"]:
            raise RuntimeError("All80 completed images required")
        for name, digest in done["output_hashes"].items():
            if file_hash(folder / name) != digest:
                raise RuntimeError(f"Completed output changed: {folder / name}")
        batches.append(done)
        rows.extend(read_csv(folder / "per_image_metrics.csv"))
        originals.append(torch.load(resolve(cfg["source_output"]) / f"batch{batch:02d}/inputs.pt", map_location="cpu", weights_only=False, mmap=True)["x"])
        recs.append(torch.load(folder / "rec.pt", map_location="cpu", weights_only=True))
    if [int(r["global_index"]) for r in rows] != list(range(80)):
        raise RuntimeError("Incomplete global-index coverage")
    metrics = {f"{key}_mean": float(np.mean([float(r[key]) for r in rows])) for key in ("lpips", "psnr", "ssim")}
    metrics["embedding_cos09"] = sum(int(r["measured_recovered_cos09"]) for r in rows)
    old = read_csv(resolve(cfg["source_output"]) / "per_image_metrics.csv")
    if [r["global_index"] for r in old] != [r["global_index"] for r in rows]:
        raise RuntimeError("Historical comparison image IDs differ")
    paired = [{"global_index": i, "old_ported_lpips": float(old[i]["lpips"]), "upstream_naive_lpips": float(rows[i]["lpips"]),
               "delta_new_minus_old": float(rows[i]["lpips"]) - float(old[i]["lpips"])} for i in range(80)]
    write_csv(output / "per_image_metrics.csv", rows)
    write_csv(output / "paired_old_ported_comparison.csv", paired)
    write_csv(output / "per_batch_summary.csv", [{k: b[k] for k in ("batch_index", "n_images", "lpips_mean", "psnr_mean", "ssim_mean", "embedding_cos09")} for b in batches])
    torch.save(torch.cat(recs), output / "all80_rec.pt")
    render_grid(torch.cat(originals), torch.cat(recs), rows, list(range(80)), output / "all80_reconstruction_landscape.png", columns=10)
    render_grid(torch.cat(originals), torch.cat(recs), rows, manifest["random_grid_indices"], output / "random12_not_cherry_picked.png", columns=4)
    report = ["# FedAvg naive CPA + FIA (direct upstream)", "", "Completed all80 originals: VGG16, n16, T10, same cached actual-update dither4 observations.", "",
              "| Condition | LPIPS | PSNR | SSIM | Embedding cos>0.9 |", "|---|---:|---:|---:|---:|",
              f"| Direct upstream CPA + FIA | {metrics['lpips_mean']:.6f} | {metrics['psnr_mean']:.3f} | {metrics['ssim_mean']:.4f} | {metrics['embedding_cos09']}/80 |",
              "| Earlier ported UIA4bit | 0.558584 | 21.926 | 0.2976 | 53/80 |", "",
              "No LS-A calculation, debias, MP diagnostics/truncation, Stage2, refinement, oracle fills or new client training.",
              "The original CocktailPartyAttack and Direct methods are invoked directly. Original whitening uses float32 covariance/(d-1) and eig; it differs from the earlier port's float64 covariance/d and eigh.",
              "FIA takes abs(S_hat) in estimated source order. Hungarian correspondence is applied after FIA only; this evaluation adapter differs from upstream's greedy correspondence before FIA and changes the assignment of initial image noise to sources.",
              "The backend and evaluation adapter both differ from the earlier port. Any performance change is not an isolated effect of removing LS-A.",
              "Observation is the same per-row 4-bit-interval uniform-dither model, not a transmitted integer 4-bit codec.", "",
              "![All80 landscape](all80_reconstruction_landscape.png)", "![Random12 not cherry-picked](random12_not_cherry_picked.png)", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    names = ("per_image_metrics.csv", "paired_old_ported_comparison.csv", "per_batch_summary.csv", "all80_rec.pt", "all80_reconstruction_landscape.png", "random12_not_cherry_picked.png", "report.md")
    result = {"run_tag": cfg["run_tag"], "status": "complete", "n_images": 80, "config": cfg, "identity": manifest["identity"],
              "metrics": metrics, "oracle_fills": 0, "ls_A": False, "completed_at_utc": utc_now(),
              "runtime_sec": {key: sum(b["runtime_sec"][key] for b in batches) for key in ("cpa", "fia")},
              "output_hashes": {name: file_hash(output / name) for name in names}}
    write_json(output / "summary.json", result)
    status(output, cfg, state="complete", n_images=80, metrics=metrics)
    return result


@contextmanager
def execution_lock(output: Path):
    with (output / ".execution.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def publish_status(cfg: dict[str, Any], output: Path, state: str) -> None:
    """Update only the new experiment's bounded status block."""
    path = WORKSPACE / "codex/STATUS.md"
    document = path.read_text(encoding="utf-8")
    begin, end = "<!-- M8_NAIVE_STATUS_BEGIN -->", "<!-- M8_NAIVE_STATUS_END -->"
    if document.count(begin) != 1 or document.count(end) != 1:
        raise RuntimeError("Expected one M8 status block")
    lines = [begin, "## M8 FedAvg naive CPA", "", f"- State: **{state.upper()}**; run tag `{cfg['run_tag']}`.",
             "- Direct original CPA and FIA classes; no LS-A, debias, MP or Stage2 calculation.",
             "- Same80 cached n16/T10 dither4 observations; no new client training.",
             "- Original-image/source correspondence is applied after FIA only.",
             "- [Protocol and code entry points](docs/FEDAVG_NAIVE_CPA_PROTOCOL.md)."]
    if state == "complete":
        result = read_json(output / "summary.json")
        values = result["metrics"]
        lines.append(f"- All80 complete: LPIPS **{values['lpips_mean']:.6f}**, PSNR **{values['psnr_mean']:.3f}**, SSIM **{values['ssim_mean']:.4f}**, embedding cos>0.9 **{values['embedding_cos09']}/80**.")
        lines.append(f"- [Results and PNGs](../quantization_aware_cpa/results/{cfg['run_tag']}/report.md).")
    lines.append(end)
    start, finish = document.index(begin), document.index(end) + len(end)
    path.write_text(document[:start] + "\n".join(lines) + document[finish:], encoding="utf-8")


def worker(config_path: Path, cfg: dict[str, Any], output: Path) -> int:
    state = {"run_tag": cfg["run_tag"], "state": "starting", "worker_pid": os.getpid(), "started_at_utc": utc_now()}
    write_json(output / "queue_status.json", state)
    command = [sys.executable, "-u", "-m", "qacpa.run_m8_fedavg_naive", "--config", str(config_path), "--phase", "run"]
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
                publish_status(cfg, output, "prepared")
                print("Prepared direct upstream FedAvg naive baseline; no GPU optimization performed", flush=True)
                return
            if args.phase == "run":
                device = torch.device(cfg["device"])
                if device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable")
                if file_hash(Path(manifest["weights_path"])) != manifest["weights_sha256"]:
                    raise RuntimeError("Frozen pretrained weights changed")
                publish_status(cfg, output, "running")
                for batch in cfg["batch_indices"]:
                    run_batch(cfg, manifest, batch, output, device)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            summarize(cfg, manifest, output)
        except BaseException as exc:
            status(output, cfg, state="failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
