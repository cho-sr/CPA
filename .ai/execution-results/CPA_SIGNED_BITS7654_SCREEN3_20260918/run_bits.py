"""One-shot strict serial wrapper for the authorized 7 -> 6 -> 5 -> 4 sweep."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


TASK_ID = "CPA_SIGNED_BITS7654_SCREEN3_20260918"
REPO_ROOT = Path(r"C:\Users\조상제\Desktop\CPA")
PYTHON = Path(r"C:\miniconda\envs\DLG\python.exe")
RESULT_ROOT = REPO_ROOT / ".ai/execution-results" / TASK_ID
BITS = (7, 6, 5, 4)
PER_BIT_CAP = 12600.0
AGGREGATE_CAP = 50400.0
PRIVATE_TRUTH = REPO_ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"


def public_root(bits: int) -> Path:
    return REPO_ROOT / f"quantization_aware_cpa/results/signed_int{bits}_randomized8_screen3_fia_B16_T10_20260918_sweep"


def private_root(bits: int) -> Path:
    return REPO_ROOT / f"quantization_aware_cpa/results_private/signed_int{bits}_randomized8_screen3_fia_B16_T10_20260918_sweep"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")
    temp.replace(path)


def _state() -> dict[str, Any]:
    path = RESULT_ROOT / "schedule_state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "task_id": TASK_ID, "status": "RUNNING", "bits": list(BITS),
        "bit_status": {str(bit): "READY" for bit in BITS},
        "private_data_loaded": False, "scientific_quality": "NOT_EVALUATED",
        "actual_image_cell_feasibility": "NOT_EVALUATED",
    }


def _save_state(state: dict[str, Any]) -> None:
    _write(RESULT_ROOT / "schedule_state.json", state)


def _register_all() -> list[dict[str, Any]]:
    sys.path.insert(0, str(REPO_ROOT))
    from quantization_aware_cpa.qacpa import signed_bits_screen3_runner as runner
    registrations = []
    for bit in BITS:
        if public_root(bit).exists() or private_root(bit).exists():
            raise RuntimeError(f"bit {bit} output collision at startup")
        registrations.append(runner.register_bit(bit, public_root(bit), private_root(bit)))
    return registrations


def _run_stage(bit: int, name: str, args: list[str], bit_started: float,
               overall_started: float, summary: dict[str, Any]) -> dict[str, Any]:
    bit_remaining = PER_BIT_CAP - (time.perf_counter() - bit_started)
    overall_remaining = AGGREGATE_CAP - (time.perf_counter() - overall_started)
    timeout = min(bit_remaining, overall_remaining)
    if timeout <= 0:
        raise TimeoutError(f"bit {bit} real-work budget exhausted before {name}")
    stdout_path = RESULT_ROOT / f"b{bit}_{name}.stdout.log"
    stderr_path = RESULT_ROOT / f"b{bit}_{name}.stderr.log"
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, \
         stderr_path.open("w", encoding="utf-8", newline="\n") as stderr:
        process = subprocess.Popen([str(PYTHON), *args], cwd=REPO_ROOT,
                                   stdout=stdout, stderr=stderr,
                                   stdin=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            exit_code, timed_out = 124, True
        else:
            timed_out = False
    result = {"bit": bit, "name": name, "pid": process.pid, "exit_code": exit_code,
              "timed_out": timed_out, "wall_seconds": time.perf_counter() - started,
              "stdout": stdout_path.as_posix(), "stderr": stderr_path.as_posix(),
              "bit_elapsed_seconds": time.perf_counter() - bit_started,
              "aggregate_elapsed_seconds": time.perf_counter() - overall_started}
    summary.setdefault("stages", []).append(result)
    _write(RESULT_ROOT / "execution_summary.json", summary)
    if exit_code != 0:
        raise RuntimeError(f"bit {bit} {name} failed with exit code {exit_code}")
    return result


def _write_report(summary: dict[str, Any]) -> None:
    lines = [f"# {TASK_ID}", "", f"Status: **{summary['status']}**.", "",
             "Execution is strictly serial in the registered order 7 -> 6 -> 5 -> 4.",
             "Each bit uses a fresh observation from the same archived raw update; no candidate, target, or optimizer state crosses bit widths.",
             "", "| Bit | Status | Evaluation |", "| ---: | --- | --- |"]
    for bit in BITS:
        state = summary.get("bit_status", {}).get(str(bit), "BLOCKED")
        result = summary.get("bit_results", {}).get(str(bit), {})
        lines.append(f"| {bit} | {state} | {result.get('evaluation_status', 'n/a')} |")
    lines += ["", "Each public top3 was selected before private loading. Private evaluation consumes x only.",
              "Actual-image cell feasibility: NOT_EVALUATED.",
              "A failed bit stops all dependent lower-bit jobs; successful artifacts are preserved."]
    (RESULT_ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    overall_started = time.perf_counter()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "task_id": TASK_ID, "status": "RUNNING", "wrapper_pid": os.getpid(),
        "started_at_epoch": time.time(), "bits": list(BITS), "per_bit_cap_seconds": PER_BIT_CAP,
        "aggregate_cap_seconds": AGGREGATE_CAP, "stages": [], "bit_status": {str(bit): "READY" for bit in BITS},
        "bit_results": {}, "private_data_loaded": False, "scientific_quality": "NOT_EVALUATED",
        "actual_image_cell_feasibility": "NOT_EVALUATED", "parent_completed_before_launch": True,
    }
    _write(RESULT_ROOT / "execution_summary.json", summary)
    try:
        registrations = _register_all()
        summary["registrations"] = [{"bits": reg["bits"], "task_id": reg["task_id"],
                                     "public_root": reg["public_output_root"], "private_root": reg["private_output_root"],
                                     "code_hashes": reg["code_hashes"], "registered": True}
                                    for reg in registrations]
        summary["startup_receipt"] = {"status": "READY_VALIDATED", "bits": list(BITS),
                                      "all_four_registered_before_private_loading": True,
                                      "private_data_loaded": False, "cuda_cap_bytes": 12 * 1024**3}
        _write(RESULT_ROOT / "startup_receipt.json", summary["startup_receipt"])
        _write(RESULT_ROOT / "execution_summary.json", summary)
        for bit in BITS:
            bit_started = time.perf_counter()
            summary["bit_status"][str(bit)] = "RUNNING"
            _save_state(summary)
            runner_args = ["-m", "quantization_aware_cpa.qacpa.signed_bits_screen3_runner", "run",
                           "--bits", str(bit), "--public-root", str(public_root(bit)),
                           "--private-root", str(private_root(bit))]
            _run_stage(bit, "public", runner_args, bit_started, overall_started, summary)
            _run_stage(bit, "public_gate", [
                "-m", "quantization_aware_cpa.qacpa.signed_bits_screen3_runner", "gate",
                "--bits", str(bit), "--public-root", str(public_root(bit))],
                bit_started, overall_started, summary)
            eval_output = private_root(bit) / "evaluation"
            _run_stage(bit, "evaluation", [
                "-m", "quantization_aware_cpa.qacpa.signed_bits_screen3_evaluation",
                "--bits", str(bit), "--public-root", str(public_root(bit)),
                "--private-truth", str(PRIVATE_TRUTH), "--output", str(eval_output)],
                bit_started, overall_started, summary)
            comparison = json.loads((eval_output / "comparison.json").read_text(encoding="utf-8"))
            summary["bit_status"][str(bit)] = "COMPLETED"
            summary["bit_results"][str(bit)] = {
                "evaluation_status": comparison["evaluation_status"], "scientific_quality": comparison["scientific_quality"],
                "mean_metrics": comparison["mean_metrics"],
                "primary_contrast": comparison["primary_contrast_rank1_minus_center25000"],
                "candidate_count": comparison["public_gate"]["candidate_count"],
                "wall_seconds": time.perf_counter() - bit_started,
            }
            _save_state(summary)
            _write(RESULT_ROOT / "execution_summary.json", summary)
        summary.update({"status": "COMPLETED", "scientific_quality": "EVALUATED", "private_data_loaded": True,
                        "finished_at_epoch": time.time(), "wall_seconds": time.perf_counter() - overall_started})
        _write(RESULT_ROOT / "integrity.json", {
            "status": "COMPLETED", "bits": list(BITS), "bit_status": summary["bit_status"],
            "public_gate_before_each_private_eval": True, "private_x_only": True,
            "labels_consumed": False, "attack_true_updates_consumed": False,
            "parent_artifacts_preserved": True, "actual_image_cell_feasibility": "NOT_EVALUATED",
        })
        _write(RESULT_ROOT / "execution_summary.json", summary)
        _write_report(summary)
        return 0
    except Exception as error:
        summary.update({"status": "PARTIAL" if any(v == "COMPLETED" for v in summary["bit_status"].values()) else "FAILED",
                        "error_type": type(error).__name__, "error": str(error),
                        "finished_at_epoch": time.time(), "wall_seconds": time.perf_counter() - overall_started})
        for bit in BITS:
            if summary["bit_status"][str(bit)] in {"READY", "RUNNING"}:
                summary["bit_status"][str(bit)] = "BLOCKED_BY_PREVIOUS_FAILURE"
        _write(RESULT_ROOT / "integrity.json", {"status": summary["status"], "bit_status": summary["bit_status"],
            "error_type": type(error).__name__, "error": str(error), "parent_artifacts_preserved": True})
        _write(RESULT_ROOT / "execution_summary.json", summary)
        _write_report(summary)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
