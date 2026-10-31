"""One-shot bounded wrapper for the registered ICA4000 -> FIA25000 run."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# A task-local script is launched directly by Python, before its child
# environment is constructed. Bind the repository package for registration
# and orchestration imports first; stage subprocesses receive the same path
# through env().
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "quantization_aware_cpa"))
sys.path.insert(0, str(ROOT))

from qacpa import signed_ica4000_fia_evaluation as evaluation
from qacpa import signed_ica4000_fia_runner as runner


TASK_ID = runner.TASK_ID
ROOT = runner.REPO_ROOT
REPORT = ROOT / ".ai/execution-results/CPA_SIGNED_ICA4000_FIA_20260917"
PUBLIC = ROOT / "quantization_aware_cpa/results/signed_ica4000_fia_B16_T10_20260917"
PRIVATE = ROOT / "quantization_aware_cpa/results_private/signed_ica4000_fia_B16_T10_20260917"
PYTHON = Path("C:/miniconda/envs/DLG/python.exe")
MAX_SECONDS = runner.MAX_SECONDS
STAGES = ("SEPARATE", "INVERT", "PRIVATE_EVALUATION")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def env() -> dict[str, str]:
    value = os.environ.copy()
    value["PYTHONPATH"] = os.pathsep.join((str(ROOT / "quantization_aware_cpa"), str(ROOT)))
    value["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return value


def code_hashes() -> dict[str, str]:
    return evaluation._code_hashes()


def registration() -> dict:
    if not PYTHON.is_file():
        raise RuntimeError(f"registered Python runtime is missing: {PYTHON}")
    if REPORT.exists():
        allowed = {Path(__file__).resolve(), REPORT / "__pycache__",
                   REPORT / "prelaunch_validation.json",
                   REPORT / "startup_failure_before_repair.json",
                   REPORT / "startup_failure_before_repair.stdout.log",
                   REPORT / "startup_failure_before_repair.stderr.log",
                   REPORT / "wrapper_stdout.log", REPORT / "wrapper_stderr.log"}
        existing = {path.resolve() for path in REPORT.iterdir()}
        if existing - allowed:
            raise RuntimeError(f"refusing preexisting report evidence: {REPORT}")
    if PUBLIC.exists() or PRIVATE.exists():
        raise RuntimeError("refusing preexisting new public/private output roots")
    REPORT.mkdir(parents=True, exist_ok=True)
    # Authenticate the archived control and public inputs before any new step.
    old = runner._validate_old_control()
    public_hashes = runner.public_input_hashes()
    integrity = {
        "public_input_hashes": public_hashes,
        "protected_source_hashes": runner.protected_integrity(),
        "shared_source_hashes": runner.shared_integrity(),
        "old_control_hashes": old["artifact_hashes"],
    }
    value = {
        "task_id": TASK_ID, "plan_version": runner.PLAN_VERSION, "status": "REGISTERED",
        "parent_task_id": runner.PARENT_TASK_ID, "method": runner.METHOD,
        "registered_before_execution": True,
        "stage_order": ["SEPARATE", "INVERT"],
        "public_output_root": PUBLIC.resolve().as_posix(),
        "private_output_root": PRIVATE.resolve().as_posix(),
        "private_truth_path": evaluation.PRIVATE_TRUTH.resolve().as_posix(),
        "public_observation": runner.PUBLIC_OBSERVATION.resolve().as_posix(),
        "checkpoint": runner.CHECKPOINT.resolve().as_posix(),
        "public_input_hashes": public_hashes,
        "checkpoint_sha256": runner.CHECKPOINT_SHA,
        "public_manifest_sha256": runner.AUTHORIZED_MANIFEST_SHA,
        "separator_config": runner.new_config(), "config_difference": runner.config_difference(),
        "fia_config": dict(runner.FIA_CONFIG),
        "target_contract": "target_z equals abs(S_hat) exactly in native estimated order",
        "fixed_scientific_conditions": {
            "seed": 42, "separator_device": "cpu", "separator_dtype": "float32",
            "separator_threads": 1, "fastica_max_iter": 200, "fastica_tolerance": 1e-6,
            "separator_refinement_steps": 4000, "direct_fia_steps": 25000,
            "source_count": 16, "feature_dim": 25088, "cell_optimization": False,
            "fc_replay": False, "generator_used": False, "private_data_used": False,
        },
        "old_control": {"root": runner.OLD_ROOT.resolve().as_posix(),
                        "artifacts": old["artifact_hashes"]},
        "review_reference": {"path": runner.OLD_PRIVATE_REVIEW.resolve().as_posix(),
                             "sha256": runner.REVIEWED_COMPARISON_SHA,
                             "row": "combination_separator_direct"},
        "integrity_before": integrity,
        "code_hashes": code_hashes(),
        "wrapper_sha256": evaluation.file_sha256(Path(__file__).resolve()),
        "resources": {"max_total_real_phase_seconds": MAX_SECONDS,
                      "max_cuda_allocated_bytes": runner.MAX_CUDA_ALLOCATED_BYTES},
        "private_data_policy": {"load_after_public_gate": True, "consumed_field": "x only",
                                 "labels_consumed": False, "true_updates_consumed": False,
                                 "no_private_feedback": True},
        "quality": "NOT_EVALUATED_UNTIL_PRIVATE_PHASE",
    }
    write_json(REPORT / "registration.json", value)
    write_json(REPORT / "registration_initial.json", value)
    return value


def remaining(started: float) -> float:
    return MAX_SECONDS - (time.perf_counter() - started)


def commands() -> dict[str, list[str]]:
    public = PUBLIC.as_posix()
    return {
        "SEPARATE": [PYTHON.as_posix(), "-m", "qacpa.signed_ica4000_fia_runner", "separate",
                     "--device", "cpu", "--output", public],
        "INVERT": [PYTHON.as_posix(), "-m", "qacpa.signed_ica4000_fia_runner", "invert",
                   "--device", "cuda", "--source-root", public,
                   "--output", (PUBLIC / "direct_fia").as_posix()],
        "PRIVATE_EVALUATION": [PYTHON.as_posix(), "-m", "qacpa.signed_ica4000_fia_evaluation",
                               "--registration", (REPORT / "registration.json").as_posix(),
                               "--public-output", public,
                               "--private-truth", evaluation.PRIVATE_TRUTH.resolve().as_posix(),
                               "--output", PRIVATE.as_posix()],
    }


def run_process(stage: str, command: list[str], started: float) -> dict:
    log_path = REPORT / f"{stage.lower()}_stdout.log"
    record = {"task_id": TASK_ID, "stage": stage, "command": command,
              "cwd": ROOT.as_posix(), "started_at_utc": now(), "status": "RUNNING",
              "exit_status": None, "model_calls_while_running": 0,
              "hard_cap_seconds": MAX_SECONDS}
    with log_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("Command argv: " + json.dumps(command, ensure_ascii=False) + "\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=env(), stdout=stream,
                                   stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        record["pid"] = process.pid
        record["process_created_ms"] = None
        try:
            import psutil
            record["process_created_ms"] = int(psutil.Process(process.pid).create_time() * 1000)
        except Exception:
            pass
        try:
            code = process.wait(timeout=max(.1, remaining(started)))
            record["exit_status"] = code
            record["status"] = "COMPLETED" if code == 0 else "FAILED"
        except subprocess.TimeoutExpired:
            record["status"] = "TIMEOUT"
            record["termination_requested"] = True
            process.terminate()
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait(timeout=30)
            record["exit_status"] = code
            record["controlled_termination"] = True
    record["finished_at_utc"] = now()
    record["wall_seconds"] = time.perf_counter() - started
    write_json(REPORT / f"{stage.lower()}_execution.json", record)
    return record


def validate_stage(stage: str) -> dict:
    if stage == "SEPARATE":
        reg, source, target = runner._load_source(PUBLIC)
        return {"status": "PASSED", "source_sha256": runner.file_sha256(PUBLIC / "sources.pt"),
                "S_hat_sha256": runner.tensor_sha256(source), "target_z_sha256": runner.tensor_sha256(target),
                "config": reg["config"], "prefix_check": read_json(PUBLIC / "prefix_check.json")}
    if stage == "INVERT":
        reg, _, target = runner._load_source(PUBLIC)
        direct = read_json(PUBLIC / "direct_fia/registration.json")
        result = read_json(PUBLIC / "direct_fia/result.json")
        candidate = runner._validate_candidate(PUBLIC / "direct_fia/candidate.pt", direct.get("candidate_sha256"))
        if result.get("target_z_sha256") != runner.tensor_sha256(target):
            raise RuntimeError("stage validation target binding changed")
        return {"status": "PASSED", "candidate_sha256": runner.file_sha256(PUBLIC / "direct_fia/candidate.pt"),
                "candidate_shape": list(candidate.shape), "target_z_sha256": runner.tensor_sha256(target),
                "source_sha256": runner.file_sha256(PUBLIC / "sources.pt"),
                "fia_config": direct.get("fia_config"), "resource": result.get("resource")}
    raise RuntimeError(stage)


def report(record: dict) -> None:
    lines = [f"# {TASK_ID}", "", f"Status: **{record.get('status', 'UNKNOWN')}**.", "",
             "This wrapper registered exactly one 4000-step CPU ICA stage, one unchanged Direct FIA25000 stage, "
             "and one offline two-set evaluation. Cell optimization, replay, labels, private feedback, extra arms, "
             "and continuation were not registered.", "", "## Phase records", ""]
    for stage in record.get("stages", []):
        lines.append(f"- `{stage.get('stage')}`: {stage.get('status')} (exit {stage.get('exit_status')}); "
                     f"{stage.get('wall_seconds', 0):.3f}s")
    if record.get("public_gate"):
        lines += ["", f"Public gate: **{record['public_gate'].get('status')}**; "
                  f"private_loaded={record['public_gate'].get('private_loaded')}."]
    if record.get("error"):
        lines += ["", f"Error: `{record['error']}`."]
    lines += ["", "The detailed comparison, per-image metrics, control residuals, trajectories, and aligned figure "
              "are in the private output directory when the private phase completed."]
    (REPORT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def finalize(record: dict, status: str, started: float, error: str | None = None) -> int:
    record["status"] = status
    record["finished_at_utc"] = now()
    record["wall_seconds"] = time.perf_counter() - started
    if error:
        record["error"] = error
    write_json(REPORT / "stage_execution.json", record)
    write_json(REPORT / "execution_summary.json", record)
    report(record)
    return 0 if status in {"COMPLETED", "COMPLETED_WITH_REVIEW"} else 1


def main() -> int:
    try:
        registration()
        started = time.perf_counter()
        record = {"task_id": TASK_ID, "status": "RUNNING", "wrapper_pid": os.getpid(),
                  "started_at_utc": now(), "stage_order": list(STAGES), "stages": [],
                  "model_calls_while_running": 0, "budget_seconds": MAX_SECONDS,
                  "cuda_allocated_cap_bytes": runner.MAX_CUDA_ALLOCATED_BYTES,
                  "private_phase_started": False}
        write_json(REPORT / "wrapper_start.json", {"task_id": TASK_ID, "pid": os.getpid(),
                  "started_at_utc": record["started_at_utc"], "cwd": ROOT.as_posix(),
                  "python": PYTHON.as_posix(), "model_calls_while_running": 0})
        write_json(REPORT / "stage_execution.json", record)
        cmd = commands()
        for stage in ("SEPARATE", "INVERT"):
            if remaining(started) <= 0:
                return finalize(record, "FAILED_HARD_CAP", started, "45-minute cap before public stage")
            stage_record = run_process(stage, cmd[stage], started)
            record["stages"].append(stage_record)
            write_json(REPORT / "stage_execution.json", record)
            if stage_record.get("exit_status") != 0:
                return finalize(record, "FAILED_" + stage, started,
                                f"stage exit {stage_record.get('exit_status')}")
            try:
                validation = validate_stage(stage)
                write_json(REPORT / f"{stage.lower()}_validation.json", validation)
                if stage == "SEPARATE":
                    registration_value = read_json(REPORT / "registration.json")
                    registration_value["bound_artifacts"] = {
                        "source_sha256": validation["source_sha256"],
                        "target_z_sha256": validation["target_z_sha256"],
                    }
                    write_json(REPORT / "registration.json", registration_value)
                else:
                    registration_value = read_json(REPORT / "registration.json")
                    registration_value["bound_artifacts"] = {
                        **registration_value.get("bound_artifacts", {}),
                        "direct_candidate_sha256": validation["candidate_sha256"],
                    }
                    write_json(REPORT / "registration.json", registration_value)
            except Exception as error:
                return finalize(record, "FAILED_" + stage + "_VALIDATION", started, str(error))
        registration_value = read_json(REPORT / "registration.json")
        registration_value["status"] = "PUBLIC_COMPLETED"
        write_json(REPORT / "registration.json", registration_value)
        try:
            gate = evaluation.validate_public_gate(registration_path=REPORT / "registration.json",
                                                   public_output=PUBLIC)
            write_json(REPORT / "public_gate.json", gate)
            record["public_gate"] = {"status": "PASSED", "private_loaded": False}
            write_json(REPORT / "stage_execution.json", record)
        except Exception as error:
            return finalize(record, "FAILED_PUBLIC_GATE", started, str(error))
        if remaining(started) <= 0:
            return finalize(record, "FAILED_HARD_CAP", started, "45-minute cap before private phase")
        record["private_phase_started"] = True
        write_json(REPORT / "stage_execution.json", record)
        private_record = run_process("PRIVATE_EVALUATION", cmd["PRIVATE_EVALUATION"], started)
        record["stages"].append(private_record)
        if private_record.get("exit_status") not in (0, 2):
            return finalize(record, "FAILED_PRIVATE_EVALUATION", started,
                            f"private evaluator exit {private_record.get('exit_status')}")
        comparison = read_json(PRIVATE / "comparison.json")
        final_status = "COMPLETED" if comparison.get("status") == "COMPLETED" else "COMPLETED_WITH_REVIEW"
        record["evaluation_status"] = comparison.get("status")
        record["private_loaded"] = True
        record["scientific_quality"] = comparison.get("scientific_quality", "NEEDS_REVIEW")
        record["private_report_path"] = (PRIVATE / "report.md").as_posix()
        registration_value = read_json(REPORT / "registration.json")
        registration_value["status"] = "COMPLETED"
        registration_value["evaluation_status"] = comparison.get("status")
        write_json(REPORT / "registration.json", registration_value)
        return finalize(record, final_status, started)
    except Exception as error:
        REPORT.mkdir(parents=True, exist_ok=True)
        write_json(REPORT / "wrapper_failure.json", {"task_id": TASK_ID, "status": "FAILED",
                   "error_type": type(error).__name__, "error": str(error), "finished_at_utc": now()})
        print(f"{TASK_ID}: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
