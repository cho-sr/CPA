"""Two authorized same-update arms: FP32 and INT8; exact ICA4000/4000/FIA25000."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TASK = "CPA_SINGLE_CLIENT_FP32_INT8_FAST4000_GD4000_FIA25000_20261006"
OLD_HERE = ROOT / ".ai/execution-results/CPA_SINGLE_CLIENT_INT4_FAST4000_GD4000_FIA25000_20261006"
spec = importlib.util.spec_from_file_location("int4_reference_runner", OLD_HERE / "run_experiment.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
from qacpa import signed_int8_observation as int8
from quantized_update_cpa.cell_utils import encode as encode4

torch, core = base.torch, base.core
read, write, digest = base.read, base.write, base.digest
CONFIG, FIA = base.CONFIG, base.FIA
ARMS = ("fp32", "int8")
PUBLIC = ROOT / "quantization_aware_cpa/results/single_client_fp32_int8_fast4000_gd4000_fia25000_20261006"
PRIVATE = ROOT / "quantization_aware_cpa/results_private/single_client_fp32_int8_fast4000_gd4000_fia25000_20261006"
RAW = base.TRUTH.with_name("raw_updates.pt")
SENDER = base.TRUTH.with_name("sender_validation.json")
TRUTH = base.TRUTH
OLD_PUBLIC = base.PUBLIC
OLD_METRICS = base.PRIVATE / "metrics.json"
ARM_BITS = {"fp32": 32, "int8": 8}


def require_arm(arm):
    if arm not in ARMS:
        raise ValueError("Only the two authorized arms fp32/int8 may run")


def code_hashes():
    return base.code_hashes() | {str(Path(__file__)): digest(Path(__file__)),
                               str(Path(int8.__file__)): digest(int8.__file__),
                               str(ROOT / "quantized_update_cpa/cell_utils.py"): digest(ROOT / "quantized_update_cpa/cell_utils.py")}


def input_hashes(arm):
    require_arm(arm)
    observation = PUBLIC / arm / "observation"
    manifest = observation / "manifest.json"
    result = {str(base.CHECKPOINT): digest(base.CHECKPOINT), str(manifest): digest(manifest)}
    if result[str(base.CHECKPOINT)] != base.CHECKPOINT_SHA:
        raise ValueError("Checkpoint identity changed")
    entries = read(manifest)["layers"].values()
    for entry in entries:
        path = observation / entry["file"]
        if path.is_symlink() or path.resolve().parent != observation.resolve() or digest(path) != entry["sha256"]:
            raise ValueError("Public observation path/hash mismatch")
        result[str(path)] = entry["sha256"]
    result[str(PUBLIC / "sender_provenance.json")] = digest(PUBLIC / "sender_provenance.json")
    return result


def prepare():
    base.configure_runtime(42, threads=1)
    if PUBLIC.exists() or PRIVATE.exists() or (HERE / "started.json").exists():
        raise FileExistsError("Preserve prior work; no duplicate preparation/restart")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        raise RuntimeError("CUDA with at least8GiB free is required")
    if (digest(RAW) != int8.RAW_UPDATES_SHA or digest(SENDER) != int8.ARCHIVED_SENDER_VALIDATION_SHA
            or digest(TRUTH) != int8.EVALUATION_TRUTH_SHA):
        raise ValueError("Archived same-client inputs are not canonical")
    base.public_gate()  # Existing INT4 optimization and provenance remain intact.
    sender = read(SENDER)
    if sender["public_manifest_sha256"] != base.MANIFEST_SHA or sender["checkpoint_sha256"] != base.CHECKPOINT_SHA:
        raise ValueError("Archived update is not linked to the completed INT4 baseline")
    payload = torch.load(RAW, map_location="cpu", weights_only=True)
    if (set(payload) != {"updates", "delta_convention", "checkpoint_sha256"}
            or payload["checkpoint_sha256"] != base.CHECKPOINT_SHA
            or payload["delta_convention"] != "theta_after10_minus_theta_start"
            or set(payload["updates"]) != set(int8.FC_SHAPES)):
        raise ValueError("Canonical raw update schema mismatch")
    updates = payload["updates"]
    _, baseline_codes = base.read_bundle(base.OBSERVATION)
    for name, shape in int8.FC_SHAPES.items():
        value = int8._require_tensor(updates[name], name)
        if tuple(value.shape) != shape:
            raise ValueError("Raw update shape mismatch")
        q, scale = encode4(value)
        old_q, old_scale = baseline_codes[name]
        if not torch.equal(q, old_q) or not torch.equal(scale, old_scale):
            raise ValueError("Raw update does not reproduce the exact existing INT4 packet")
    del baseline_codes, q, scale, old_q, old_scale, value
    PUBLIC.mkdir(parents=True)
    fp32_folder = PUBLIC / "fp32/observation"
    fp32_folder.mkdir(parents=True)
    fp32_file = fp32_folder / "updates.pt"
    torch.save(payload, fp32_file)
    file_sha = digest(fp32_file)
    manifest32 = {
        "format": "unquantized_cpu_fp32_update_v1", "bits": 32, "quantization": "none",
        "observation_source": "single_client_single_local_update_archived_raw_update",
        "checkpoint_sha256": base.CHECKPOINT_SHA, "training": sender["training"],
        "batch_id": "batch00", "labels": "private", "new_local_training": False,
        "layers": {name: {"shape": list(shape), "file": "updates.pt", "sha256": file_sha,
                          "dtype": "float32", "value_bytes": updates[name].numel()*4}
                   for name, shape in int8.FC_SHAPES.items()},
        "communication": {"value_bytes": sum(x.numel()*4 for x in updates.values()),
                          "physical_update_file_bytes": fp32_file.stat().st_size},
    }
    write(fp32_folder / "manifest.json", manifest32)
    manifest8 = int8.write_bundle(PUBLIC / "int8/observation", updates)
    # Existing strict reader authenticates every code packet and metadata field.
    _, codes8 = int8.read_bundle(PUBLIC / "int8/observation")
    for name, value in updates.items():
        q, scale = int8.encode(value)
        if not torch.equal(q, codes8[name][0]) or not torch.equal(scale, codes8[name][1]):
            raise ValueError("INT8 sender/reader contract mismatch")
    del payload, updates, codes8, q, scale, value
    gc.collect()
    write(PUBLIC / "sender_provenance.json", {
        "source_raw_updates_sha256": int8.RAW_UPDATES_SHA,
        "archived_sender_validation_sha256": int8.ARCHIVED_SENDER_VALIDATION_SHA,
        "evaluation_truth_sha256": int8.EVALUATION_TRUTH_SHA,
        "same_update_as_int4": True, "all_six_int4_q_and_scales_reproduced": True,
        "training": sender["training"], "raw_update_consumed_by_sender_only": True,
        "truth_images_or_labels_loaded_by_sender": False, "new_local_training": False,
        "fp32_communication": manifest32["communication"], "int8_communication": manifest8["communication"],
    })
    from qacpa.report_full_update_iterations import _verified_lpips_weights
    registration = {
        "task_id": TASK, "status": "PREPARED", "session_role": "CODEX_EXECUTOR",
        "authorized_thread": os.environ.get("CODEX_THREAD_ID"),
        "authorization": "User: 32비트랑 8비트에 대해서 돌려줘; same prior ICA4000/GD4000/FIA25000 conditions",
        "arms": list(ARMS), "arm_bits": ARM_BITS, "arm_order": list(ARMS),
        "code_hashes": code_hashes(), "input_hashes": {a: input_hashes(a) for a in ARMS},
        "config": CONFIG.as_dict(), "fast_early_stop": False, "fia_config": FIA,
        "source_count": 16, "seed": 42, "input_shape": [4096, 25088],
        "same_client_update_as_int4": True, "new_local_training": False,
        "generator_used": False, "private_feedback": False, "cell_or_replay_optimization": False,
        "candidate_selection": "final_only", "evaluation_after_both_attacks": True,
        "evaluation": "All16; pixel-MSE Hungarian matching once per arm for MSE/PSNR/SSIM/VGG LPIPS",
        "lpips_weights": _verified_lpips_weights(),
        "cached_int4_metrics": {"path": str(OLD_METRICS), "sha256": digest(OLD_METRICS)},
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0), "threads": 1},
        "estimated_total_minutes": [70, 90], "restart": "forbidden",
    }
    write(PUBLIC / "registration.json", registration)
    write(HERE / "execution_record.json", {"task_id": TASK, "status": "PREPARED", "arms": {a: "PREPARED" for a in ARMS}})
    print(json.dumps({"task_id": TASK, "status": "PREPARED", "same_int4_update_verified": True}), flush=True)


def provenance_gate(arm):
    require_arm(arm)
    registration = read(PUBLIC / "registration.json")
    if registration["code_hashes"] != code_hashes() or registration["input_hashes"][arm] != input_hashes(arm):
        raise ValueError("Sealed input/source identity changed")
    if (registration["arms"] != list(ARMS) or registration["config"] != CONFIG.as_dict()
            or registration["fia_config"] != FIA or registration["fast_early_stop"] is not False):
        raise ValueError("Registered scope or settings changed")
    return registration


def public_gate(arm):
    provenance_gate(arm)
    folder = PUBLIC / arm
    result = read(folder / "result.json")
    if (result["status"] != "COMPLETED" or result["bits"] != ARM_BITS[arm]
            or [result[k] for k in ["fast_iterations", "gd_iterations", "fia_iterations"]] != [4000, 4000, 25000]):
        raise ValueError("Final saved counts/completion mismatch")
    for name, expected in result["artifact_hashes"].items():
        if digest(folder / name) != expected:
            raise ValueError("Frozen artifact identity changed")
    for name in ["fastica_log.json", "gdica_log.json"]:
        if [row["iteration"] for row in read(folder / name)] != list(range(1, 4001)):
            raise ValueError("Full ICA update log mismatch")
    fia = read(folder / "fia_result.json")
    if fia["settings"]["n_iter"] != 25000 or fia["log"][-1]["iter"] != 24999:
        raise ValueError("FIA final zero-based iteration mismatch")
    sources = torch.load(folder / "sources.pt", map_location="cpu", weights_only=True)
    target = sources["target_z"]
    if (tuple(target.shape) != (16, 25088) or not torch.equal(target, sources["S_hat"].abs())
            or not bool(torch.isfinite(target).all())
            or core.tensor_sha256(target) != read(folder / "separation.json")["target_z_sha256"]):
        raise ValueError("Saved ICA feature contract mismatch")
    pixels = torch.load(folder / "candidate.pt", map_location="cpu", weights_only=True)["pixels"]
    base.candidate_valid(pixels)
    return pixels


def load_observation(arm):
    require_arm(arm)
    folder = PUBLIC / arm
    if arm == "fp32":
        payload = torch.load(folder / "observation/updates.pt", map_location="cpu", weights_only=True)
        X = payload["updates"]["classifier.0.weight"]
        del payload
    else:
        _, bundle = int8.read_bundle(folder / "observation")
        q, scale = bundle["classifier.0.weight"]
        X = int8.dequantize(q, scale)
        del bundle, q, scale
    if (tuple(X.shape) != (4096, 25088) or X.dtype != torch.float32
            or X.device.type != "cpu" or not bool(torch.isfinite(X).all())):
        raise ValueError("Attack observation contract mismatch")
    return X


def attack(arm):
    provenance_gate(arm)
    folder = PUBLIC / arm
    with (folder / "started.json").open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_unix": time.time()}, stream)
    base.configure_runtime(42, threads=1)
    started = time.perf_counter()
    X = load_observation(arm)
    whitening = core.whiten_observation(X, 16, rank_floor=CONFIG.rank_floor)
    del X
    fast, fast_log, first_converged = base.fixed_fastica(whitening, CONFIG)
    refinement = core.refine_from_fastica(whitening, fast, CONFIG)
    if len(fast_log) != 4000 or refinement.iterations != 4000 or len(refinement.log) != 4000:
        raise ValueError("Exact ICA budgets not completed")
    if (tuple(refinement.target_z.shape) != (16, 25088)
            or not torch.equal(refinement.target_z, refinement.S_hat.abs())
            or not bool(torch.isfinite(refinement.target_z).all())):
        raise ValueError("Separated feature contract mismatch before FIA")
    torch.save({"S_hat": refinement.S_hat, "target_z": refinement.target_z}, folder / "sources.pt")
    write(folder / "fastica_log.json", fast_log)
    write(folder / "gdica_log.json", refinement.log)
    separation_seconds = time.perf_counter() - started
    write(folder / "separation.json", {"status": "COMPLETED", "fast_iterations": 4000, "gd_iterations": 4000,
          "first_converged_iteration": first_converged, "final_fast_residual": fast.residual,
          "fast_early_stop": False, "final_gd_losses": refinement.final_losses,
          "target_z_sha256": core.tensor_sha256(refinement.target_z), "wall_seconds": separation_seconds,
          "native_estimated_order": True, "private_loaded": False})
    target = refinement.target_z
    del whitening, refinement, fast, fast_log
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    fia = base.run_upstream_fia(target_z=target, device="cuda", config=FIA, output_dir=folder / "fia", seed=42)
    pixels = fia.pop("rec")
    base.candidate_valid(pixels)
    if fia["settings"]["n_iter"] != 25000 or fia["log"][-1]["iter"] != 24999:
        raise ValueError("FIA did not finish exactly25000 steps")
    torch.save({"pixels": pixels}, folder / "candidate.pt")
    write(folder / "fia_result.json", fia)
    provenance_gate(arm)
    artifacts = ["sources.pt", "fastica_log.json", "gdica_log.json", "separation.json", "candidate.pt", "fia_result.json"]
    write(folder / "result.json", {"task_id": TASK, "arm": arm, "bits": ARM_BITS[arm], "status": "COMPLETED",
          "quality": "NOT_EVALUATED", "fast_iterations": 4000, "gd_iterations": 4000, "fia_iterations": 25000,
          "separation_seconds": separation_seconds, "fia_seconds": fia["wall_time_sec"],
          "attack_seconds": time.perf_counter() - started, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
          "artifact_hashes": {name: digest(folder / name) for name in artifacts}})
    public_gate(arm)
    print(f"ATTACK_COMPLETED {arm}", flush=True)


def evaluate(arm):
    public_gate(arm)  # Authenticate completed public artifacts before private loading.
    if digest(TRUTH) != int8.EVALUATION_TRUTH_SHA:
        raise ValueError("Canonical evaluation bundle identity changed")
    before = read(PUBLIC / arm / "result.json")["artifact_hashes"]
    # Reuse the exact previous metric/image evaluator with only arm paths and
    # the independently registered evaluation-bundle SHA supplied in memory.
    base.PUBLIC, base.PRIVATE, base.HERE = PUBLIC / arm, PRIVATE / arm, HERE / arm
    base.TASK = TASK + "_" + arm.upper()
    base.TRUTH_SHA = int8.EVALUATION_TRUTH_SHA
    base.public_gate = lambda: public_gate(arm)
    base.evaluate()
    metrics_path = PRIVATE / arm / "metrics.json"
    metrics = read(metrics_path)
    metrics.update(lpips="computed_offline_vgg", private_truth_sha256=int8.EVALUATION_TRUTH_SHA,
                   observation_bits=ARM_BITS[arm], same_update_as_int4=True)
    write(metrics_path, metrics)
    result = read(PUBLIC / arm / "result.json")
    report_path = PRIVATE / arm / "report.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace("Single-client B16/T10 signed symmetric deterministic per-tensor INT4;",
                            "Single-client B16/T10 " + ("unquantized FP32;" if arm == "fp32" else "signed symmetric deterministic per-tensor INT8;"))
    report += f"\nAttack wall time: {result['attack_seconds']:.3f}s; peak CUDA allocated: {result['peak_cuda_allocated_bytes']/1024**3:.3f}GiB. Same archived single-client update as the existing INT4 arm.\n"
    report_path.write_text(report, encoding="utf-8")
    (HERE / arm / "report.md").write_text(report.replace("comparison.png", str(PRIVATE / arm / "comparison.png")), encoding="utf-8")
    public_gate(arm)
    if before != read(PUBLIC / arm / "result.json")["artifact_hashes"]:
        raise ValueError("Completed attack changed during evaluation")
    write(PRIVATE / arm / "evaluation_record.json", {"status": "COMPLETED", "all_images": 16,
          "optimization_rerun": False, "public_artifacts_unchanged": True,
          "output_hashes": {name: digest(PRIVATE / arm / name) for name in ["metrics.json", "comparison.png", "report.md"]}})


def summary(states):
    registration = read(PUBLIC / "registration.json")
    cached = registration["cached_int4_metrics"]
    if digest(Path(cached["path"])) != cached["sha256"]:
        raise ValueError("Cached completed INT4 metrics changed")
    rows = []
    for arm in ARMS:
        if states[arm].get("evaluation_exit_status") == 0:
            metrics = read(PRIVATE / arm / "metrics.json")
            result = read(PUBLIC / arm / "result.json")
            rows.append({"bits": ARM_BITS[arm], "existing_result": False,
                         **{key: metrics[key] for key in ["mean_psnr_db", "mean_ssim", "mean_lpips", "mean_mse"]},
                         "attack_seconds": result["attack_seconds"], "fast_iterations": 4000,
                         "gd_iterations": 4000, "fia_iterations": 25000,
                         "image": str(PRIVATE / arm / "comparison.png")})
    old_metrics, old_result = read(OLD_METRICS), read(OLD_PUBLIC / "result.json")
    rows.append({"bits": 4, "existing_result": True,
                 **{key: old_metrics[key] for key in ["mean_psnr_db", "mean_ssim", "mean_lpips", "mean_mse"]},
                 "attack_seconds": old_result["attack_seconds"], "fast_iterations": 4000,
                 "gd_iterations": 4000, "fia_iterations": 25000,
                 "image": str(OLD_METRICS.with_name("comparison.png"))})
    write(PRIVATE / "comparison.json", {"task_id": TASK, "same_update": True, "rows": rows,
          "states": states, "int4_reused_without_rerun": True, "success_threshold": "NOT_REGISTERED"})
    lines = [f"# {TASK}", "", "Same single-client B16/T10 update, VGG16 Conv frozen/FC trained, seed42.", "",
             "FastICA4000 + GD-ICA4000 + Direct FIA25000 per new arm. Existing INT4 result reused without rerunning.", "",
             "| Bits | PSNR dB | SSIM | VGG LPIPS | Attack seconds |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['bits']} | {row['mean_psnr_db']:.6f} | {row['mean_ssim']:.6f} | {row['mean_lpips']:.6f} | {row['attack_seconds']:.3f} |")
    lines += ["", "All16 final images; pixel-MSE Hungarian matching once per arm shared by every metric. Truth x used only after both attacks; no generator, private feedback, tuning or intermediate selection.", "",
              "Software completion alone does not establish successful reconstruction; no numerical success cutoff registered.", ""]
    for row in rows:
        lines += [f"[Bits{row['bits']} image comparison]({row['image']})", ""]
    text = "\n".join(lines)
    (PRIVATE / "report.md").write_text(text, encoding="utf-8")
    (HERE / "report.md").write_text(text, encoding="utf-8")
    return rows


def run():
    with (HERE / "started.json").open("x", encoding="utf-8") as stream:
        json.dump({"task_id": TASK, "pid": os.getpid(), "started_unix": time.time()}, stream)
    started, states = time.perf_counter(), {}
    for arm in ARMS:
        provenance_gate(arm)
        write(HERE / "execution_record.json", {"task_id": TASK, "status": "RUNNING", "stage": "ATTACK", "arm": arm, "pid": os.getpid(), "arms": states})
        completed = subprocess.run([sys.executable, "-B", str(Path(__file__)), "attack", "--arm", arm], cwd=ROOT)
        states[arm] = {"attack_exit_status": completed.returncode}
    # No private metrics are available to either attack process.
    for arm in ARMS:
        if states[arm]["attack_exit_status"] == 0:
            write(HERE / "execution_record.json", {"task_id": TASK, "status": "RUNNING", "stage": "OFFLINE_EVALUATION", "arm": arm, "pid": os.getpid(), "arms": states})
            completed = subprocess.run([sys.executable, "-B", str(Path(__file__)), "evaluate", "--arm", arm], cwd=ROOT)
            states[arm]["evaluation_exit_status"] = completed.returncode
    rows = summary(states)
    ok = all(states[a].get("attack_exit_status") == 0 and states[a].get("evaluation_exit_status") == 0 for a in ARMS)
    record = {"task_id": TASK, "status": "COMPLETED" if ok else "FAILED", "exit_status": 0 if ok else 1,
              "pid": os.getpid(), "arms": states, "wall_seconds": time.perf_counter()-started,
              "public_output": str(PUBLIC), "private_output": str(PRIVATE), "report": str(PRIVATE / "report.md"),
              "optimization_restarts": 0, "new_local_training": False,
              "new_bitwidth_representations": True, "result_rows": rows}
    write(HERE / "execution_record.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "attack", "evaluate"])
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    try:
        if args.command in ("attack", "evaluate"):
            require_arm(args.arm)
            {"attack": attack, "evaluate": evaluate}[args.command](args.arm)
        else:
            {"prepare": prepare, "run": run}[args.command]()
    except Exception as error:
        traceback.print_exc()
        failure = {"task_id": TASK, "command": args.command, "arm": args.arm,
                   "type": type(error).__name__, "error": str(error), "optimization_restart": False}
        write(HERE / (f"{args.arm}_failure.json" if args.arm else "failure.json"), failure)
        if args.command == "run":
            write(HERE / "execution_record.json", failure | {"status": "FAILED", "exit_status": 1, "pid": os.getpid()})
        sys.exit(1)
