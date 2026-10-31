"""One-shot bounded wrapper for CPA_SIGNED_INT8_RANDOMIZED8_FIA_20260917."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "CPA_SIGNED_INT8_RANDOMIZED8_FIA_20260917"
RESULT_ROOT = ROOT / ".ai/execution-results" / TASK_ID
PYTHON = Path("C:/miniconda/envs/DLG/python.exe")
PUBLIC_ROOT = ROOT / "quantization_aware_cpa/results/signed_int8_randomized8_fia_B16_T10_20260917"
PRIVATE_ROOT = ROOT / "quantization_aware_cpa/results_private/signed_int8_randomized8_fia_B16_T10_20260917"
RAW = ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/raw_updates.pt"
SENDER_VALIDATION = RAW.parent / "sender_validation.json"
TRUTH = RAW.parent / "evaluation_truth.pt"
CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth"
MAX_SECONDS = 90 * 60


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "quantization_aware_cpa"), str(ROOT), env.get("PYTHONPATH", "")])
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    stages = []

    def stage(name: str, module: str, args: list[str]) -> bool:
        elapsed = time.perf_counter() - started
        remaining = MAX_SECONDS - elapsed
        log = RESULT_ROOT / f"{len(stages)+1:02d}_{name}.log"
        record = {"name": name, "module": module, "args": args, "started_at": time.time(),
                  "log": log.as_posix(), "remaining_seconds_before": remaining}
        if remaining <= 0:
            record.update({"status": "NOT_STARTED", "exit_code": None, "error": "budget exhausted"})
            stages.append(record)
            return False
        command = [str(PYTHON), "-m", module, *args]
        try:
            with log.open("w", encoding="utf-8", newline="\n") as stream:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                           timeout=max(1, int(remaining)))
            record["exit_code"] = completed.returncode
            record["status"] = "COMPLETED" if completed.returncode == 0 else "FAILED"
        except subprocess.TimeoutExpired as error:
            record.update({"status": "TIMEOUT", "exit_code": None, "error": str(error)})
        except Exception as error:
            record.update({"status": "FAILED", "exit_code": None, "error": f"{type(error).__name__}: {error}"})
        record["finished_at"] = time.time()
        record["wall_seconds"] = time.perf_counter() - started
        stages.append(record)
        return record["status"] == "COMPLETED"

    success = stage("register", "qacpa.signed_int8_randomized_runner", [
        "register", "--public-root", str(PUBLIC_ROOT), "--private-root", str(PRIVATE_ROOT)])
    if success:
        success = stage("sender", "qacpa.signed_int8_observation", [
            "sender", "--raw-updates", str(RAW), "--archived-sender-validation", str(SENDER_VALIDATION),
            "--evaluation-truth", str(TRUTH), "--output", str(PUBLIC_ROOT / "observation"),
            "--private-output", str(PRIVATE_ROOT / "sender")])
    if success:
        success = stage("separate_center_and_members", "qacpa.signed_int8_randomized_runner", [
            "separate", "--public-observation", str(PUBLIC_ROOT / "observation"),
            "--output", str(PUBLIC_ROOT / "separation")])
    if success:
        success = stage("direct_fia_center", "qacpa.signed_int8_randomized_runner", [
            "invert", "--source-root", str(PUBLIC_ROOT / "separation/center"),
            "--output", str(PUBLIC_ROOT / "fia_center"), "--arm", "center_int8"])
    if success:
        success = stage("direct_fia_randomized8", "qacpa.signed_int8_randomized_runner", [
            "invert", "--source-root", str(PUBLIC_ROOT / "separation/aggregate"),
            "--output", str(PUBLIC_ROOT / "fia_randomized8"), "--arm", "randomized8_int8"])
    if success:
        success = stage("public_gate_finalize", "qacpa.signed_int8_randomized_runner", [
            "finalize-public", "--public-root", str(PUBLIC_ROOT)])
    if success:
        success = stage("offline_evaluation", "qacpa.signed_int8_randomized_evaluation", [
            "--public-root", str(PUBLIC_ROOT), "--private-truth", str(TRUTH),
            "--output", str(PRIVATE_ROOT / "evaluation")])

    status = "COMPLETED" if success else ("PARTIAL" if any(s["status"] == "COMPLETED" for s in stages) else "FAILED")
    artifacts = {}
    for path in (PUBLIC_ROOT / "registration.json", PUBLIC_ROOT / "observation/manifest.json",
                 PUBLIC_ROOT / "separation/result.json", PUBLIC_ROOT / "fia_center/result.json",
                 PUBLIC_ROOT / "fia_randomized8/result.json", PRIVATE_ROOT / "sender/sender_validation.json",
                 PRIVATE_ROOT / "evaluation/comparison.json", PRIVATE_ROOT / "evaluation/report.md"):
        if path.is_file():
            artifacts[path.as_posix()] = sha256(path)
    summary = {"task_id": TASK_ID, "status": status, "stage": "real_experiment",
               "scientific_quality": "EVALUATED" if status == "COMPLETED" else "NOT_EVALUATED",
               "started_at": started, "finished_at": time.time(),
               "wall_seconds": time.perf_counter() - started, "budget_seconds": MAX_SECONDS,
               "stages": stages, "artifacts": artifacts,
               "public_root": PUBLIC_ROOT.as_posix(), "private_root": PRIVATE_ROOT.as_posix(),
               "report_path": (PRIVATE_ROOT / "evaluation/report.md").as_posix() if (PRIVATE_ROOT / "evaluation/report.md").is_file() else None,
               "no_restart_or_extra_arm": True, "actual_image_cell_feasibility": "NOT_EVALUATED"}
    write_json(RESULT_ROOT / "execution_summary.json", summary)
    write_json(RESULT_ROOT / "integrity.json", {"task_id": TASK_ID, "status": status, "artifacts": artifacts,
                                                   "raw_input_sha256": sha256(RAW) if RAW.is_file() else None,
                                                   "checkpoint_sha256": sha256(CHECKPOINT) if CHECKPOINT.is_file() else None})
    if (PRIVATE_ROOT / "evaluation/report.md").is_file():
        shutil.copy2(PRIVATE_ROOT / "evaluation/report.md", RESULT_ROOT / "report.md")
    else:
        lines = [f"# {TASK_ID}", "", f"Status: **{status}**.", "", "Stage execution did not reach completed offline evaluation.", ""]
        for item in stages:
            lines.append(f"- {item['name']}: {item['status']} (exit={item.get('exit_code')})")
        (RESULT_ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0 if status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
