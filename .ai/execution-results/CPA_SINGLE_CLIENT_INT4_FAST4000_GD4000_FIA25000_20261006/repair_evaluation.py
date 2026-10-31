"""Repair the truth-file identity check; evaluate frozen artifacts only."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("frozen_task_runner", HERE / "run_experiment.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

EXPECTED_TRUTH_SHA = "724bbacbe316cc28dac086db9cfaf6507f4c1af31144fd610cc480985aab72e2"
EVIDENCE = {
    "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/sender_validation.json":
        "6e78ec5ff2376b01c23a3606fa18b3d9e60d1209f55504211aaf811919d2e244",
    "quantization_aware_cpa/results_private/signed_ica4000_fia_B16_T10_20260917/results.json":
        "a640905ce85effc93f1912b8ac0fcaa2a869330e5db46e7145901bb342e33023",
}


def main():
    started = time.perf_counter()
    if (HERE / "evaluation_repair.json").exists() or runner.PRIVATE.exists():
        raise FileExistsError("Preserve existing evaluation; this repair is one-time only")
    # Authenticate all sealed public inputs, production code and saved outputs.
    pixels = runner.public_gate()
    result = runner.read(runner.PUBLIC / "result.json")
    before = {name: runner.digest(runner.PUBLIC / name)
              for name in ["registration.json", "result.json", *result["artifact_hashes"]]}
    counts = {}
    for name in ["fastica_log.json", "gdica_log.json"]:
        rows = runner.read(runner.PUBLIC / name)
        if [row["iteration"] for row in rows] != list(range(1, 4001)):
            raise ValueError(f"Saved log is not exactly 4000 contiguous updates: {name}")
        counts[name] = len(rows)
    fia = runner.read(runner.PUBLIC / "fia_result.json")
    if fia["settings"]["n_iter"] != 25000 or fia["log"][-1]["iter"] != 24999:
        raise ValueError("Saved FIA settings/final zero-based iteration mismatch")
    sources = runner.torch.load(runner.PUBLIC / "sources.pt", map_location="cpu", weights_only=True)
    if (tuple(sources["target_z"].shape) != (16, 25088)
            or not runner.torch.equal(sources["target_z"], sources["S_hat"].abs())
            or not bool(runner.torch.isfinite(sources["target_z"]).all())):
        raise ValueError("Saved ICA sources/absolute-value target contract mismatch")
    separation = runner.read(runner.PUBLIC / "separation.json")
    if runner.core.tensor_sha256(sources["target_z"]) != separation["target_z_sha256"]:
        raise ValueError("Saved target tensor identity mismatch")
    del sources, pixels

    evidence = []
    for relative, expected in EVIDENCE.items():
        path = runner.ROOT / relative
        if runner.digest(path) != expected:
            raise ValueError("Historical provenance evidence changed")
        evidence.append(runner.read(path))
    sender, old_evaluation = evidence
    if (sender["source_truth_sha256"] != runner.TRUTH_SHA
            or sender["evaluation_truth_sha256"] != EXPECTED_TRUTH_SHA
            or sender["public_manifest_sha256"] != runner.MANIFEST_SHA
            or sender["checkpoint_sha256"] != runner.CHECKPOINT_SHA
            or old_evaluation["private_truth_metadata"]["sha256"] != EXPECTED_TRUTH_SHA
            or runner.digest(runner.TRUTH) != EXPECTED_TRUTH_SHA):
        raise ValueError("Evaluation bundle identity is not established by both historical records")

    original_record = runner.read(HERE / "execution_record.json")
    archive = HERE / "original_worker_execution_record.json"
    if archive.exists():
        raise FileExistsError("Original worker record archive already exists")
    archive.write_bytes((HERE / "execution_record.json").read_bytes())
    repair = {
        "task_id": runner.TASK, "status": "EVALUATING_FROZEN_CANDIDATE",
        "authorization": "User one-time completion; necessary reporting repair only",
        "original_worker_exit_status": original_record["exit_status"],
        "cause": "Source-data TRUTH_SHA was incorrectly compared to the derived evaluation bundle",
        "incorrect_expected_truth_sha256": runner.TRUTH_SHA,
        "correct_evaluation_truth_sha256": EXPECTED_TRUTH_SHA,
        "evidence_file_hashes": EVIDENCE,
        "evidence_fields": ["sender_validation.source_truth_sha256",
                            "sender_validation.evaluation_truth_sha256",
                            "20260917 results.private_truth_metadata.sha256"],
        "repair_script_sha256": runner.digest(Path(__file__)),
        "original_runner_sha256": runner.digest(HERE / "run_experiment.py"),
        "public_artifact_hashes_before": before,
        "verified_counts": {"fastica": 4000, "gdica": 4000, "fia": 25000},
        "optimization_rerun": False, "public_gate_unchanged": True,
        "evaluation_algorithm_unchanged": True,
    }
    runner.write(HERE / "evaluation_repair.json", repair)
    # Correct only the evaluator's expected file identity in memory. The sealed
    # original runner remains byte-for-byte intact, including its public gate.
    runner.TRUTH_SHA = EXPECTED_TRUTH_SHA
    runner.evaluate()
    runner.public_gate()
    after = {name: runner.digest(runner.PUBLIC / name) for name in before}
    if before != after:
        raise ValueError("Frozen public artifacts changed during evaluation")
    metrics = runner.read(runner.PRIVATE / "metrics.json")
    if len(metrics["per_image"]) != 16:
        raise ValueError("Final evaluation did not cover all 16 images")

    note = (
        "\nEvaluation/reporting repair: original worker exited 1 after completing all optimization. "
        "Its evaluator compared the source-data SHA to the derived evaluation bundle. "
        "The evaluation bundle SHA matches both original sender validation and the 2026-09-17 "
        "evaluation record. Only this file-identity expectation was corrected in memory; "
        "the same frozen candidate, original public gate and metric code were used. "
        "No optimization was restarted; all sealed public artifacts and source files are unchanged.\n\n"
        f"Attack wall time: {result['attack_seconds']:.3f}s "
        f"({result['attack_seconds']/60:.3f}min). Peak CUDA allocated memory: "
        f"{result['peak_cuda_allocated_bytes']/1024**3:.3f}GiB.\n\n"
        f"FastICA first residual below tolerance: iteration {separation['first_converged_iteration']}; "
        f"final residual {separation['final_fast_residual']:.9g} is above the 1e-6 tolerance. "
        "The run nevertheless completed exactly 4000 updates with early stopping disabled.\n"
    )
    for path in [runner.PRIVATE / "report.md", HERE / "report.md"]:
        path.write_text(path.read_text(encoding="utf-8") + note, encoding="utf-8")
    repair.update(status="COMPLETED", reporting_exit_status=0,
                  public_artifact_hashes_after=after, frozen_artifacts_unchanged=True,
                  evaluation_wall_seconds=time.perf_counter() - started,
                  output_hashes={name: runner.digest(runner.PRIVATE / name)
                                 for name in ["metrics.json", "comparison.png", "report.md"]})
    runner.write(HERE / "evaluation_repair.json", repair)
    runner.write(HERE / "execution_record.json", {
        "task_id": runner.TASK, "status": "COMPLETED", "completion": "EVALUATION_REPAIRED",
        "exit_status": original_record["exit_status"], "exit_status_scope": "original worker",
        "worker_status": "FAILED_IN_OFFLINE_EVALUATION", "attack_status": "COMPLETED",
        "reporting_exit_status": 0, "optimization_rerun": False,
        "fast_iterations": 4000, "gd_iterations": 4000, "fia_iterations": 25000,
        "attack_seconds": result["attack_seconds"], "pid": original_record["pid"],
        "public_output": str(runner.PUBLIC), "private_output": str(runner.PRIVATE),
        "report": str(runner.PRIVATE / "report.md"), "repair_record": str(HERE / "evaluation_repair.json"),
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        runner.write(HERE / "reporting_repair_failure.json", {
            "task_id": runner.TASK, "type": type(error).__name__, "error": str(error),
            "optimization_rerun": False,
        })
        sys.exit(1)
