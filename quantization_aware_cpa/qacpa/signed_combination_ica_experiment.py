"""One-shot outer orchestration for the authorized repaired ICA experiment.

The staged implementation runner remains the owner of the scientific method.
This module only binds its three public stages, enforces the registered
sequence and wall-clock budget, then invokes the isolated offline evaluator.
It has no private-data or generator interface.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from . import signed_combination_ica_evaluation as evaluation
from . import signed_combination_ica_runner as runner
from .signed_e2e_runner import REPO_ROOT, write_json


TASK_ID = "CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916"
PLAN_VERSION = 1
IMPLEMENTATION_TASK_ID = runner.TASK_ID
ROOT = REPO_ROOT
REPORT_ROOT = ROOT / ".ai/execution-results/CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916"
PUBLIC_ROOT = ROOT / "quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916"
PRIVATE_ROOT = ROOT / "quantization_aware_cpa/results_private/signed_combination_ica_B16_T10_20260916"
PRIVATE_TRUTH = evaluation.PRIVATE_TRUTH
TASK_WRAPPER = REPORT_ROOT / "run_real_experiment.py"
PYTHON = Path("C:/miniconda/envs/DLG/python.exe")
MAX_SECONDS = 45 * 60
MAX_CUDA_ALLOCATED_BYTES = 12 * 1024**3
STAGES = ("SEPARATE", "INVERT", "REFINE", "PRIVATE_EVALUATION")


class ExperimentError(RuntimeError):
    """The registered orchestration cannot safely continue."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExperimentError(f"expected JSON object: {path}")
    return value


def _stage_commands(public_root: Path) -> dict[str, list[str]]:
    root = public_root.as_posix()
    return {
        "SEPARATE": [PYTHON.as_posix(), "-m", "qacpa.signed_combination_ica_runner",
                     "separate", "--device", "cpu", "--output", root],
        "INVERT": [PYTHON.as_posix(), "-m", "qacpa.signed_combination_ica_runner",
                   "invert", "--device", "cuda", "--source-root", root,
                   "--output", f"{root}/direct_fia"],
        "REFINE": [PYTHON.as_posix(), "-m", "qacpa.signed_combination_ica_runner",
                   "refine", "--device", "cuda", "--source-root", root,
                   "--candidate-root", f"{root}/direct_fia",
                   "--output", f"{root}/actual_update_refinement"],
    }


def _short_controls() -> dict[str, Any]:
    controls = evaluation._validate_old_controls({})
    return {"metadata": controls["metadata"], "candidate_hashes": controls["candidate_hashes"],
            "paths": controls["paths"]}


def _assert_expected_repair_state() -> dict[str, Any]:
    return evaluation._repair_integrity_snapshot()


def register(*, report: Path = REPORT_ROOT, public_output: Path = PUBLIC_ROOT,
             private_output: Path = PRIVATE_ROOT) -> dict[str, Any]:
    """Write the complete pre-computation outer registration."""
    report = report.resolve()
    public_output, private_output = public_output.resolve(), private_output.resolve()
    if report.exists():
        existing = {path.resolve() for path in report.iterdir()}
        allowed = {TASK_WRAPPER.resolve()} if TASK_WRAPPER.is_file() else set()
        # py_compile may leave a task-local cache beside the wrapper; it is
        # generated code metadata and carries no run or candidate evidence.
        cache = report / "__pycache__"
        if cache.is_dir():
            allowed.add(cache.resolve())
        if existing - allowed:
            raise FileExistsError(f"refusing preexisting experiment report directory: {report}")
    if public_output.exists() or private_output.exists():
        raise FileExistsError("registered public/private output root already exists")
    if not PYTHON.is_file():
        raise ExperimentError(f"registered Python runtime is missing: {PYTHON}")
    report.mkdir(parents=True, exist_ok=True)
    repair = _assert_expected_repair_state()
    controls = _short_controls()
    commands = _stage_commands(public_output)
    registration = {
        "task_id": TASK_ID, "plan_version": PLAN_VERSION, "status": "REGISTERED",
        "implementation_task_id": IMPLEMENTATION_TASK_ID,
        "method": runner.METHOD, "registered_before_execution": True,
        "stage_order": ["SEPARATE", "INVERT", "REFINE"],
        "public_output_root": public_output.as_posix(),
        "private_output_root": private_output.as_posix(),
        "private_truth_path": PRIVATE_TRUTH.resolve().as_posix(),
        "public_observation": runner.PUBLIC_OBSERVATION.resolve().as_posix(),
        "checkpoint": runner.CHECKPOINT.resolve().as_posix(),
        "public_input_hashes": evaluation._public_input_hashes(),
        "checkpoint_sha256": runner.CHECKPOINT_SHA,
        "public_manifest_sha256": runner.AUTHORIZED_MANIFEST_SHA,
        "separator_config": dict(runner.CANONICAL_SEPARATOR_CONFIG),
        "fia_config": dict(runner.CANONICAL_FIA_CONFIG),
        "refine_objective": dict(runner.CANONICAL_REFINE_OBJECTIVE),
        "refine_arm": runner.REFINE_ARM_NAME,
        "label_hypothesis": "coverage-based H1 assignment from the new public candidate, frozen before image optimization",
        "target_contract": "target_z equals abs(S_hat) exactly in native estimated order",
        "fixed_scientific_conditions": {
            "seed": 42, "separator_device": "cpu", "separator_dtype": "float32",
            "fastica_max_iter": 200, "fastica_tolerance": 1e-6,
            "separator_refinement_steps": 1000, "direct_fia_steps": 25000,
            "image_refinement_steps": 100, "source_count": 16,
            "feature_dim": 25088, "all_six_fc_tensors": True,
            "generator_used": False, "private_data_used": False,
        },
        "commands": {name: command for name, command in commands.items()},
        "resources": {"max_total_real_phase_seconds": MAX_SECONDS,
                      "max_cuda_allocated_bytes": MAX_CUDA_ALLOCATED_BYTES,
                      "controlled_termination": "owned experiment child only at hard cap"},
        "controls": controls,
        "repair_integrity": repair,
        "integrity_before": repair,
        "code_hashes": evaluation._code_hashes(),
        "bound_artifacts": {},
        "private_policy": {"load_after_public_gate": True, "consumed_field": "x only",
                            "labels_consumed": False, "true_updates_consumed": False,
                            "no_private_feedback": True},
        "quality": "NOT_EVALUATED_UNTIL_PRIVATE_PHASE",
    }
    write_json(report / "registration.json", registration)
    write_json(report / "registration_initial.json", registration)
    return registration


def _update_registration(report: Path, **fields: Any) -> dict[str, Any]:
    path = report / "registration.json"
    value = _json(path)
    value.update(fields)
    write_json(path, value)
    return value


def _remaining(started: float) -> float:
    return MAX_SECONDS - (time.perf_counter() - started)


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "quantization_aware_cpa"), str(ROOT)))
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def _run_process(stage: str, command: list[str], report: Path, started: float) -> dict[str, Any]:
    log_path = report / f"{stage.lower()}_stdout.log"
    record: dict[str, Any] = {
        "task_id": TASK_ID, "implementation_task_id": IMPLEMENTATION_TASK_ID,
        "stage": stage, "command": command, "cwd": ROOT.as_posix(),
        "started_at_utc": now(), "status": "RUNNING", "exit_status": None,
        "hard_cap_seconds": MAX_SECONDS, "model_calls_while_running": 0,
    }
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command argv: " + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=_environment(), stdout=log,
                                   stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        record["pid"] = process.pid
        try:
            import psutil
            record["process_created_ms"] = int(psutil.Process(process.pid).create_time() * 1000)
        except Exception:
            record["process_created_ms"] = None
        remaining = _remaining(started)
        try:
            exit_code = process.wait(timeout=max(0.1, remaining))
            record["exit_status"] = exit_code
            record["status"] = "COMPLETED" if exit_code == 0 else "FAILED"
        except subprocess.TimeoutExpired:
            record["status"] = "TIMEOUT"
            record["termination_requested"] = True
            process.terminate()
            try:
                exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = process.wait(timeout=30)
            record["exit_status"] = exit_code
            record["controlled_termination"] = True
    record["finished_at_utc"] = now()
    record["wall_seconds"] = time.perf_counter() - started
    write_json(report / f"{stage.lower()}_execution.json", record)
    return record


def _validate_stage(stage: str, public_output: Path) -> dict[str, Any]:
    if stage == "SEPARATE":
        return evaluation._validate_stage_result(public_output, "SEPARATE")
    if stage == "INVERT":
        return evaluation._validate_stage_result(public_output, "INVERT")
    if stage == "REFINE":
        return evaluation._validate_stage_result(public_output, "REFINE")
    raise ExperimentError(f"unknown stage: {stage}")


def _bind_stage_artifacts(report: Path, stage: str, validation: Mapping[str, Any]) -> None:
    registration = _json(report / "registration.json")
    bound = dict(registration.get("bound_artifacts", {}))
    if stage == "SEPARATE":
        bound.update({"source_artifact_sha256": validation["source_sha256"],
                      "target_z_sha256": validation["target_z_sha256"]})
    elif stage == "INVERT":
        bound["direct_candidate_sha256"] = validation["candidate_sha256"]
    elif stage == "REFINE":
        bound["cell_candidate_sha256"] = validation["final_candidate_sha256"]
    _update_registration(report, bound_artifacts=bound)


def _finalize(report: Path, record: dict[str, Any], status: str, *, error: str | None = None,
              started: float | None = None) -> int:
    record["status"] = status
    record["finished_at_utc"] = now()
    if started is not None:
        record["wall_seconds"] = time.perf_counter() - started
    else:
        record["wall_seconds"] = record.get("wall_seconds", 0.0)
    if error:
        record["error"] = error
    _write_outer_report(report, record)
    write_json(report / "stage_execution.json", record)
    write_json(report / "execution_summary.json", record)
    return 0 if status in {"COMPLETED", "COMPLETED_WITH_REVIEW"} else 1


def _write_outer_report(report: Path, record: Mapping[str, Any]) -> None:
    """Leave a truthful task report even when a later stage fails."""
    registration = {}
    try:
        registration = _json(report / "registration.json")
    except (OSError, ValueError, ExperimentError):
        pass
    lines = [f"# {TASK_ID}", "", f"Status: **{record.get('status', 'UNKNOWN')}**.", "",
             f"Implementation task: `{IMPLEMENTATION_TASK_ID}`.",
             "The wrapper registered exactly one SEPARATE -> INVERT -> REFINE sequence "
             "and one offline evaluation phase; no extra arm, seed, tuning, or rerun was registered.", "",
             "## Phase records", ""]
    for item in record.get("stages", []):
        lines.append(f"- `{item.get('stage')}`: {item.get('status')} (exit {item.get('exit_status')})")
    lines += ["", "Prelaunch tests and exact commands are recorded in `registration.json` and `registration_initial.json`.",
              "Public/private evidence is incremental in this directory; the private evaluator writes its own `report.md`."]
    if registration.get("bound_artifacts"):
        lines += ["", "Bound public artifact hashes:", ""]
        for key, value in registration["bound_artifacts"].items():
            lines.append(f"- `{key}`: `{value}`")
    private_report = registration.get("private_output_root")
    if private_report:
        lines += ["", f"Private comparison path: `{private_report}/comparison.json`. "]
    (report / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_registered(*, report: Path = REPORT_ROOT, public_output: Path = PUBLIC_ROOT,
                   private_output: Path = PRIVATE_ROOT,
                   private_truth: Path = PRIVATE_TRUTH,
                   process_runner: Callable[[str, list[str], Path, float], dict[str, Any]] | None = None) -> int:
    """Execute exactly the registered sequence once and return its exit code."""
    report, public_output, private_output = report.resolve(), public_output.resolve(), private_output.resolve()
    registration = _json(report / "registration.json")
    if registration.get("status") != "REGISTERED" or registration.get("task_id") != TASK_ID:
        raise ExperimentError("outer experiment is not a registered fresh run")
    if public_output.exists() or private_output.exists():
        raise FileExistsError("refusing to rerun or overwrite an experiment output root")
    if evaluation._norm_map(registration.get("code_hashes", {})) != evaluation._norm_map(evaluation._code_hashes()):
        raise ExperimentError("registered experiment/evaluator code changed before launch")
    if private_truth.resolve() != PRIVATE_TRUTH.resolve():
        raise ExperimentError("private truth path is not the registered baseline path")
    started = time.perf_counter()
    record: dict[str, Any] = {
        "task_id": TASK_ID, "implementation_task_id": IMPLEMENTATION_TASK_ID,
        "status": "RUNNING", "wrapper_pid": os.getpid(), "started_at_utc": now(),
        "stage_order": list(STAGES), "stages": [], "model_calls_while_running": 0,
        "budget_seconds": MAX_SECONDS, "cuda_allocated_cap_bytes": MAX_CUDA_ALLOCATED_BYTES,
        "private_phase_started": False,
    }
    write_json(report / "wrapper_start.json", {"task_id": TASK_ID, "pid": os.getpid(),
              "started_at_utc": record["started_at_utc"], "cwd": ROOT.as_posix(),
              "python": PYTHON.as_posix(), "model_calls_while_running": 0})
    write_json(report / "stage_execution.json", record)
    runner_fn = process_runner or _run_process
    commands = _stage_commands(public_output)
    for stage in ("SEPARATE", "INVERT", "REFINE"):
        if _remaining(started) <= 0:
            return _finalize(report, record, "FAILED_HARD_CAP", error="45-minute cap before stage", started=started)
        try:
            stage_record = runner_fn(stage, commands[stage], report, started)
        except Exception as error:
            record["stages"].append({"stage": stage, "status": "FAILED", "error": str(error)})
            return _finalize(report, record, "FAILED_STAGE_LAUNCH", error=str(error), started=started)
        record["stages"].append(stage_record)
        write_json(report / "stage_execution.json", record)
        if stage_record.get("exit_status") != 0:
            return _finalize(report, record, "FAILED_" + stage,
                             error=f"stage exit {stage_record.get('exit_status')}", started=started)
        try:
            validation = _validate_stage(stage, public_output)
            write_json(report / f"{stage.lower()}_validation.json", validation)
            _bind_stage_artifacts(report, stage, validation)
        except Exception as error:
            stage_record["artifact_validation"] = {"status": "FAILED", "error": str(error)}
            write_json(report / "stage_execution.json", record)
            return _finalize(report, record, "FAILED_" + stage + "_VALIDATION",
                             error=str(error), started=started)
    try:
        gate = evaluation.validate_public_gate(
            registration_path=report / "registration.json", public_output=public_output)
        write_json(report / "public_gate.json", gate)
        record["public_gate"] = {"status": "PASSED", "private_loaded": False}
        write_json(report / "stage_execution.json", record)
    except Exception as error:
        write_json(report / "public_gate_failure.json", {"task_id": TASK_ID, "status": "FAILED",
                   "private_loaded": False, "error_type": type(error).__name__, "error": str(error)})
        return _finalize(report, record, "FAILED_PUBLIC_GATE", error=str(error), started=started)
    if _remaining(started) <= 0:
        return _finalize(report, record, "FAILED_HARD_CAP",
                         error="45-minute cap before private phase", started=started)
    evaluation_command = [PYTHON.as_posix(), "-m", "qacpa.signed_combination_ica_evaluation",
                          "--registration", (report / "registration.json").as_posix(),
                          "--public-output", public_output.as_posix(),
                          "--private-truth", private_truth.as_posix(),
                          "--output", private_output.as_posix()]
    record["private_phase_started"] = True
    write_json(report / "stage_execution.json", record)
    private_record = runner_fn("PRIVATE_EVALUATION", evaluation_command, report, started)
    record["stages"].append(private_record)
    if private_record.get("exit_status") != 0:
        return _finalize(report, record, "FAILED_PRIVATE_EVALUATION",
                         error=f"private evaluator exit {private_record.get('exit_status')}", started=started)
    comparison = _json(private_output / "comparison.json")
    final_status = "COMPLETED" if comparison.get("status") == "COMPLETED" else "COMPLETED_WITH_REVIEW"
    record["evaluation_status"] = comparison.get("status")
    record["private_loaded"] = True
    record["scientific_quality"] = comparison.get("scientific_quality", "EVALUATED")
    record["report_path"] = (report / "report.md").as_posix()
    record["private_report_path"] = (private_output / "report.md").as_posix()
    return _finalize(report, record, final_status, started=started)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_ROOT)
    parser.add_argument("--public-output", type=Path, default=PUBLIC_ROOT)
    parser.add_argument("--private-output", type=Path, default=PRIVATE_ROOT)
    parser.add_argument("--private-truth", type=Path, default=PRIVATE_TRUTH)
    args = parser.parse_args(argv)
    try:
        return run_registered(**vars(args))
    except Exception as error:
        report = args.report.resolve()
        report.mkdir(parents=True, exist_ok=True)
        write_json(report / "wrapper_failure.json", {"task_id": TASK_ID, "status": "FAILED",
                   "error_type": type(error).__name__, "error": str(error), "finished_at_utc": now()})
        print(f"{TASK_ID}: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
