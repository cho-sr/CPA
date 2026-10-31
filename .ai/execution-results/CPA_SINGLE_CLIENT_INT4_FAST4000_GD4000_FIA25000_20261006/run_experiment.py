"""One authorized INT4 single-client run; reuse the existing ICA/FIA/math helpers."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "quantization_aware_cpa"), str(ROOT)]
import torch
from qacpa import signed_combination_ica as core
from qacpa.deterministic_hybrid_setup import configure_runtime
from qacpa.signed_baseline_observation import CHECKPOINT_SHA, TRUTH_SHA, read_bundle
from qacpa.upstream_naive import run_upstream_fia, source_hashes

TASK = "CPA_SINGLE_CLIENT_INT4_FAST4000_GD4000_FIA25000_20261006"
HERE = Path(__file__).resolve().parent
PUBLIC = ROOT / "quantization_aware_cpa/results/single_client_int4_fast4000_gd4000_fia25000_20261006"
PRIVATE = ROOT / "quantization_aware_cpa/results_private/single_client_int4_fast4000_gd4000_fia25000_20261006"
OBSERVATION = ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation"
CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth"
TRUTH = ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"
MANIFEST_SHA = "9b4891765e6fc7e4adb1e3e11c158b3d4a2c3430d3397e1f1466ddafe2e1cd69"
CONFIG = core.CombinationICAConfig(fast_max_iter=4000, refinement_iterations=4000)
FIA = dict(n_iter=25000, lr=0.1, tv=0.1, fi=1.0, optimizer="adam", scheduler="cosine", log_every=5000)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def input_hashes():
    if digest(CHECKPOINT) != CHECKPOINT_SHA or digest(OBSERVATION / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Registered checkpoint or observation manifest changed")
    manifest = read(OBSERVATION / "manifest.json")
    if manifest["observation_source"] != "single_client_single_local_update" or manifest["codec_metadata"]["bits"] != 4:
        raise ValueError("Expected a single-client INT4 observation")
    result = {str(CHECKPOINT): CHECKPOINT_SHA, str(OBSERVATION / "manifest.json"): MANIFEST_SHA}
    for entry in manifest["layers"].values():
        path = OBSERVATION / entry["file"]
        if path.resolve().parent != OBSERVATION.resolve() or path.is_symlink() or digest(path) != entry["sha256"]:
            raise ValueError("Public packet hash/path mismatch")
        result[str(path)] = entry["sha256"]
    return result


def code_hashes():
    files = [Path(__file__), Path(core.__file__), ROOT / "quantization_aware_cpa/qacpa/signed_baseline_observation.py",
             ROOT / "quantization_aware_cpa/qacpa/deterministic_hybrid_setup.py",
             ROOT / "quantization_aware_cpa/qacpa/upstream_naive.py",
             ROOT / "quantization_aware_cpa/qacpa/full_update_evaluation.py",
             ROOT / "quantization_aware_cpa/qacpa/report_full_update_iterations.py"]
    return {str(path): digest(path) for path in files} | {"upstream:" + str(k): v for k, v in source_hashes().items()}


def fixed_fastica(whitening, config=CONFIG):
    """Exactly the requested update count; report convergence without breaking."""
    W = core.symmetric_decorrelation(torch.eye(whitening.source_count, dtype=torch.float32, device=whitening.Xw.device))
    history, first_converged = [], None
    for iteration in range(1, config.fast_max_iter + 1):
        Wnew, _ = core._fastica_update(W, whitening.Xw)
        residual = float((torch.sum(Wnew * W, dim=1).abs() - 1).abs().max())
        if not math.isfinite(residual):
            raise core.ICAError(f"Nonfinite FastICA residual at {iteration}")
        W = Wnew
        if residual < config.fast_tolerance and first_converged is None:
            first_converged = iteration
        history.append({"iteration": iteration, "residual": residual})
    return core.FastICAResult(W.detach(), residual < config.fast_tolerance, iteration, residual,
                             config.fast_max_iter, config.fast_tolerance), history, first_converged


def candidate_valid(pixels):
    if (not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.float32 or tuple(pixels.shape) != (16, 3, 224, 224)
            or not bool(torch.isfinite(pixels).all()) or bool(((pixels < 0) | (pixels > 1)).any())):
        raise ValueError("Invalid final FIA pixels")


def prepare():
    configure_runtime(42, threads=1)
    if PUBLIC.exists() or PRIVATE.exists():
        raise FileExistsError("Preserve existing results; this task does not overwrite or restart")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        raise RuntimeError("CUDA with at least 8 GiB available is required")
    from qacpa.report_full_update_iterations import _verified_lpips_weights
    registration = {"task_id": TASK, "status": "PREPARED", "session_role": "CODEX_EXECUTOR",
                    "authorized_thread": os.environ.get("CODEX_THREAD_ID"),
                    "authorization": "User: ICA 각 4천번씩 FIA 25k, 4비트 업데이트, single client; 그냥 여기서 실험 넣어줘",
                    "executor_exception": "Current conversation explicitly authorized; no model switch or Luna dispatch",
                    "input_hashes": input_hashes(), "code_hashes": code_hashes(),
                    "observation": str(OBSERVATION), "config": CONFIG.as_dict(), "fast_early_stop": False,
                    "fia_config": FIA, "source_count": 16, "seed": 42, "input_shape": [4096, 25088],
                    "observation_dtype": "INT4 codes and per-tensor FP32 scale; decoded q*s for attack",
                    "generator_used": False, "private_feedback": False, "new_observation": False,
                    "cell_or_replay_optimization": False, "candidate_selection": "final_only",
                    "evaluation": "All 16 images; pixel-MSE Hungarian pairing once for MSE/PSNR/SSIM/LPIPS; no success cutoff",
                    "lpips_weights": _verified_lpips_weights(),
                    "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                                "gpu": torch.cuda.get_device_name(0), "threads": 1},
                    "estimated_minutes": [35, 45], "guardrail": "One run only; nonfinite values or provenance failure abort dependent stages"}
    PUBLIC.mkdir(parents=True)
    write(PUBLIC / "registration.json", registration)
    write(HERE / "execution_record.json", {"task_id": TASK, "status": "PREPARED", "public_output": str(PUBLIC), "private_output": str(PRIVATE)})
    print(json.dumps({"task_id": TASK, "status": "PREPARED"}), flush=True)


def public_gate():
    registration, result = read(PUBLIC / "registration.json"), read(PUBLIC / "result.json")
    if registration["input_hashes"] != input_hashes() or registration["code_hashes"] != code_hashes():
        raise ValueError("Input/source provenance changed")
    if result["status"] != "COMPLETED" or result["fast_iterations"] != 4000 or result["gd_iterations"] != 4000 or result["fia_iterations"] != 25000:
        raise ValueError("Actual execution count or completion mismatch")
    for name, expected in result["artifact_hashes"].items():
        if digest(PUBLIC / name) != expected:
            raise ValueError(f"Final artifact changed: {name}")
    pixels = torch.load(PUBLIC / "candidate.pt", map_location="cpu", weights_only=True)["pixels"]
    candidate_valid(pixels)
    return pixels


def run():
    started = time.perf_counter()
    if (HERE / "started.json").exists():
        raise FileExistsError("A launch was already claimed; no automatic restart")
    with (HERE / "started.json").open("x", encoding="utf-8") as stream:
        json.dump({"task_id": TASK, "pid": os.getpid(), "started_unix": time.time()}, stream)
    registration = read(PUBLIC / "registration.json")
    if registration["input_hashes"] != input_hashes() or registration["code_hashes"] != code_hashes():
        raise ValueError("Prepared input/source identity changed")
    configure_runtime(42, threads=1)
    write(HERE / "execution_record.json", {"task_id": TASK, "status": "RUNNING", "stage": "SEPARATION", "pid": os.getpid()})
    _, bundle = read_bundle(OBSERVATION)
    q, scale = bundle["classifier.0.weight"]
    X = q.float() * scale
    del bundle, q, scale
    whitening = core.whiten_observation(X, 16, rank_floor=CONFIG.rank_floor)
    del X
    fast, fast_log, first_converged = fixed_fastica(whitening)
    refinement = core.refine_from_fastica(whitening, fast, CONFIG)
    if len(fast_log) != 4000 or refinement.iterations != 4000 or len(refinement.log) != 4000:
        raise ValueError("The requested exact 4000/4000 counts were not completed")
    if tuple(refinement.target_z.shape) != (16, 25088) or not torch.equal(refinement.target_z, refinement.S_hat.abs()):
        raise ValueError("Separated target contract mismatch")
    torch.save({"S_hat": refinement.S_hat, "target_z": refinement.target_z}, PUBLIC / "sources.pt")
    write(PUBLIC / "fastica_log.json", fast_log)
    write(PUBLIC / "gdica_log.json", refinement.log)
    separation_seconds = time.perf_counter() - started
    write(PUBLIC / "separation.json", {"status": "COMPLETED", "fast_iterations": 4000, "gd_iterations": 4000,
          "first_converged_iteration": first_converged, "final_fast_residual": fast.residual, "fast_early_stop": False,
          "final_gd_losses": refinement.final_losses, "target_z_sha256": core.tensor_sha256(refinement.target_z),
          "wall_seconds": separation_seconds, "native_estimated_order": True, "private_loaded": False})
    target = refinement.target_z
    del whitening, refinement, fast, fast_log
    gc.collect()
    write(HERE / "execution_record.json", {"task_id": TASK, "status": "RUNNING", "stage": "FIA_25000", "pid": os.getpid(), "separation_seconds": separation_seconds})
    torch.cuda.reset_peak_memory_stats()
    fia = run_upstream_fia(target_z=target, device="cuda", config=FIA, output_dir=PUBLIC / "fia", seed=42)
    pixels = fia.pop("rec")
    candidate_valid(pixels)
    if fia["settings"]["n_iter"] != 25000 or not fia["log"] or fia["log"][-1]["iter"] + 1 != 25000:
        raise ValueError("FIA did not finish exactly 25000 steps")
    torch.save({"pixels": pixels}, PUBLIC / "candidate.pt")
    write(PUBLIC / "fia_result.json", fia)
    if registration["input_hashes"] != input_hashes() or registration["code_hashes"] != code_hashes():
        raise ValueError("Input/source identity changed during optimization")
    artifacts = ["sources.pt", "fastica_log.json", "gdica_log.json", "separation.json", "candidate.pt", "fia_result.json"]
    write(PUBLIC / "result.json", {"task_id": TASK, "status": "COMPLETED", "quality": "NOT_EVALUATED",
          "fast_iterations": 4000, "gd_iterations": 4000, "fia_iterations": 25000,
          "separation_seconds": separation_seconds, "fia_seconds": fia["wall_time_sec"],
          "attack_seconds": time.perf_counter() - started, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
          "artifact_hashes": {name: digest(PUBLIC / name) for name in artifacts}})
    write(HERE / "execution_record.json", {"task_id": TASK, "status": "RUNNING", "stage": "OFFLINE_EVALUATION", "pid": os.getpid()})
    completed = subprocess.run([sys.executable, "-B", str(Path(__file__)), "evaluate"], cwd=ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(f"Frozen-image evaluation exited {completed.returncode}; preserve completed attack")
    write(HERE / "execution_record.json", {"task_id": TASK, "status": "COMPLETED", "exit_status": 0,
          "pid": os.getpid(), "wall_seconds": time.perf_counter() - started,
          "public_output": str(PUBLIC), "private_output": str(PRIVATE), "report": str(PRIVATE / "report.md")})
    print("COMPLETED " + str(PRIVATE / "report.md"), flush=True)


def evaluate():
    configure_runtime(42, threads=1)
    pixels = public_gate()  # No private deserialization before this gate.
    if digest(TRUTH) != TRUTH_SHA:
        raise ValueError("Private evaluation truth identity changed")
    from qacpa.full_update_evaluation import evaluate_candidate
    from qacpa.report_full_update_iterations import _make_lpips_metric
    metrics = evaluate_candidate(PUBLIC / "candidate.pt", TRUTH, include_labels=False)
    payload = torch.load(TRUTH, map_location="cpu", weights_only=True)
    originals = (payload["x"] * pixels.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
                 + pixels.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)).clamp(0, 1)
    keys_materialized = sorted(payload)
    del payload
    metric, provenance = _make_lpips_metric()
    for pair in metrics["per_image"]:
        i, j = pair["candidate_index"], pair["truth_index"]
        metric.reset()
        score = float(metric((pixels[i:i+1] - .5) * 2, (originals[j:j+1] - .5) * 2))
        if not math.isfinite(score):
            raise ValueError("Nonfinite final LPIPS")
        pair["lpips"] = score
    metrics.update(mean_lpips=sum(p["lpips"] for p in metrics["per_image"]) / 16,
                   lpips_provenance=provenance, private_keys_materialized=keys_materialized,
                   private_fields_consumed=["x"], labels_consumed=False, feature_truth_consumed=False)
    PRIVATE.mkdir(parents=True, exist_ok=False)
    write(PRIVATE / "metrics.json", metrics)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    for slot, pair in enumerate(sorted(metrics["per_image"], key=lambda p: p["truth_index"])):
        row, col = divmod(slot, 4)
        i, j = pair["candidate_index"], pair["truth_index"]
        for offset, tensor, label in [(0, originals[j], f"Original {j}"), (1, pixels[i], f"Recovered {i}")]:
            axis = axes[row, 2*col+offset]
            axis.imshow(tensor.permute(1, 2, 0).numpy())
            axis.set_title(label, fontsize=9)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(PRIVATE / "comparison.png", dpi=140)
    plt.close(fig)
    result = read(PUBLIC / "result.json")
    report = (f"# {TASK}\n\nStatus: COMPLETED.\n\n"
              "Single-client B16/T10 signed symmetric deterministic per-tensor INT4; frozen VGG16 Conv, trained FC; seed42.\n\n"
              "FastICA exactly4000 (early stop disabled), GD-ICA exactly4000, Direct FIA exactly25000. No generator, cell optimization, new observation, candidate selection or private feedback.\n\n"
              f"PSNR: {metrics['mean_psnr_db']:.6f} dB; SSIM: {metrics['mean_ssim']:.6f}; VGG LPIPS: {metrics['mean_lpips']:.6f}; MSE: {metrics['mean_mse']:.8f}.\n\n"
              f"Separation: {result['separation_seconds']:.2f}s; FIA: {result['fia_seconds']:.2f}s.\n\n"
              "All16 images evaluated. Pixel-MSE Hungarian matching is computed once and shared by every metric. Only x is consumed from the deserialized evaluation bundle; bundled keys are disclosed in metrics.json. No quantitative success threshold was supplied; software completion alone is not proof of successful reconstruction. No actual-update cell feasibility claim.\n\n"
              "![Original and reconstructed images](comparison.png)\n")
    (PRIVATE / "report.md").write_text(report, encoding="utf-8")
    (HERE / "report.md").write_text(report.replace("comparison.png", str(PRIVATE / "comparison.png")), encoding="utf-8")
    print(json.dumps({"status": "EVALUATED", "mean_psnr_db": metrics["mean_psnr_db"], "mean_ssim": metrics["mean_ssim"], "mean_lpips": metrics["mean_lpips"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "evaluate"])
    args = parser.parse_args()
    try:
        {"prepare": prepare, "run": run, "evaluate": evaluate}[args.command]()
    except Exception as error:
        traceback.print_exc()
        write(HERE / "failure.json", {"task_id": TASK, "command": args.command, "type": type(error).__name__, "error": str(error)})
        if args.command == "run":
            write(HERE / "execution_record.json", {"task_id": TASK, "status": "FAILED", "exit_status": 1, "error": str(error), "pid": os.getpid()})
        sys.exit(1)
