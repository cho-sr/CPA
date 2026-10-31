"""Verify frozen results and report FP32's pre-ICA failure without rerunning."""
from pathlib import Path
import importlib.util
import math
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("immutable_two_arm_runner", HERE / "run_experiment.py")
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)


def main():
    if (HERE / "completion_verification.json").exists():
        raise FileExistsError("One-time reporting verification already saved")
    r.provenance_gate("fp32")
    r.public_gate("int8")
    r.base.public_gate()
    registration = r.read(r.PUBLIC / "registration.json")
    cached = registration["cached_int4_metrics"]
    if r.digest(Path(cached["path"])) != cached["sha256"]:
        raise ValueError("Cached INT4 metrics changed")
    record = r.read(HERE / "execution_record.json")
    failure_path = HERE / "fp32_failure.json"
    failure = r.read(failure_path)
    if (record["exit_status"] != 1 or record["arms"]["fp32"]["attack_exit_status"] != 1
            or record["arms"]["int8"] != {"attack_exit_status": 0, "evaluation_exit_status": 0}
            or failure["type"] != "ICAError"
            or failure["error"] != "observation has insufficient usable covariance rank for source_count"):
        raise ValueError("Saved failure/status does not match the reviewed traceback")
    absent = ["result.json", "separation.json", "sources.pt", "fastica_log.json",
              "gdica_log.json", "fia_result.json", "candidate.pt"]
    if any((r.PUBLIC / "fp32" / name).exists() for name in absent):
        raise ValueError("Unexpected FP32 optimization artifacts")

    checks, rows = {}, []
    for bits, folder, metrics_path in [(8, r.PUBLIC / "int8", r.PRIVATE / "int8/metrics.json"),
                                       (4, r.OLD_PUBLIC, r.OLD_METRICS)]:
        result, metrics = r.read(folder / "result.json"), r.read(metrics_path)
        for name in ["fastica_log.json", "gdica_log.json"]:
            if [row["iteration"] for row in r.read(folder / name)] != list(range(1, 4001)):
                raise ValueError("Saved ICA log does not prove4000 updates")
        fia = r.read(folder / "fia_result.json")
        if fia["settings"]["n_iter"] != 25000 or fia["log"][-1]["iter"] != 24999:
            raise ValueError("FIA count mismatch")
        pairs = metrics["per_image"]
        if (len(pairs) != 16 or sorted(p["truth_index"] for p in pairs) != list(range(16))
                or sorted(p["candidate_index"] for p in pairs) != list(range(16))
                or metrics["private_fields_consumed"] != ["x"] or metrics["labels_consumed"] is not False
                or not all(math.isfinite(p[k]) for p in pairs for k in ["mse", "psnr_db", "ssim", "lpips"])):
            raise ValueError("Incomplete or invalid all16-image evaluation")
        checks[str(bits)] = {"actual_counts": [4000, 4000, 25000], "all16_evaluated": True,
                            "metrics_sha256": r.digest(metrics_path), "artifact_hashes": result["artifact_hashes"]}
        rows.append({"bits": bits, "status": "COMPLETED", "existing_result": bits == 4,
                     **{key: metrics[key] for key in ["mean_psnr_db", "mean_ssim", "mean_lpips", "mean_mse"]},
                     "attack_seconds": result["attack_seconds"], "fast_iterations": 4000,
                     "gd_iterations": 4000, "fia_iterations": 25000,
                     "image": str(metrics_path.with_name("comparison.png"))})
    eval_record = r.read(r.PRIVATE / "int8/evaluation_record.json")
    for name, expected in eval_record["output_hashes"].items():
        if r.digest(r.PRIVATE / "int8" / name) != expected:
            raise ValueError("INT8 evaluation output identity changed")

    for source, archived in [(HERE / "execution_record.json", HERE / "original_worker_execution_record.json"),
                             (r.PRIVATE / "comparison.json", HERE / "original_worker_comparison.json"),
                             (r.PRIVATE / "report.md", HERE / "original_worker_report.md")]:
        with archived.open("xb") as stream:
            stream.write(source.read_bytes())
    failed_row = {"bits": 32, "status": "FAILED_BEFORE_ICA", "existing_result": False,
                  "failure_stage": "WHITENING", "error": failure["error"],
                  "fast_iterations": 0, "gd_iterations": 0, "fia_iterations": 0,
                  "mean_psnr_db": None, "mean_ssim": None, "mean_lpips": None, "mean_mse": None,
                  "attack_seconds": None, "image": None, "metrics_available": False}
    rows.insert(0, failed_row)
    fp32_private = r.PRIVATE / "fp32"
    fp32_private.mkdir(exist_ok=False)
    r.write(fp32_private / "failure_summary.json", failed_row | {
        "source_failure_sha256": r.digest(failure_path), "input_provenance_gate_passed": True,
        "configured_rank_floor": r.CONFIG.rank_floor, "configured_source_count": 16,
        "eigenvalues_saved": False, "true_mathematical_rank_established": False,
        "evaluation_performed": False, "optimization_restart": False,
    })
    failure_text = (
        "# FP32 run stopped before ICA\n\n"
        "The saved traceback places the failure in whiten_observation(X,16), before FastICA, GD-ICA or FIA. "
        "The input/source provenance gate passes. No final candidate or iteration logs exist. "
        "Actual optimization counts: FastICA0, GD-ICA0, FIA0; configured budgets were4000/4000/25000.\n\n"
        "Error: observation has insufficient usable covariance rank for source_count. "
        "The existing guard rejects nonfinite selected eigenvalues or any selected absolute eigenvalue "
        "at or below1e-12. Eigenvalues were not saved; this report does not establish the mathematical "
        "rank or whether scaling, numerical precision or data structure triggered the guard.\n\n"
        "No reconstruction metrics are available. This execution failure does not establish that "
        "FP32 updates cannot be reconstructed. No whitening settings were changed and no optimization was restarted.\n"
    )
    (fp32_private / "report.md").write_text(failure_text, encoding="utf-8")
    visual = ("Visual review: INT8 recovers recognizable outlines in some images, including the child, "
              "people holding a fish, lobster, dog and tree scene, but images remain faint and several "
              "outputs are dominated by colored textures. The cached INT4 image has poorer recognizable structure. "
              "This is a qualitative observation on one batch/seed; no quantitative success cutoff was registered.")
    comparison = r.read(r.PRIVATE / "comparison.json")
    comparison.update(status="PARTIAL_COMPLETION", rows=rows, fp32_failure=failed_row,
                      observed_eight_minus_four={key: rows[1][key]-rows[2][key]
                                                for key in ["mean_psnr_db", "mean_ssim", "mean_lpips", "mean_mse"]},
                      interpretation=visual, fp32_vs_int8_quality_comparison="NOT_AVAILABLE",
                      optimization_restarts=0, reporting_only_update=True)
    r.write(r.PRIVATE / "comparison.json", comparison)
    lines = [f"# {r.TASK}", "", "Status: PARTIAL_COMPLETION; original worker exit1.", "",
             "Same single-client VGG16 B16/T10 update, Conv frozen/FC trained, seed42; Generator-free Direct FIA.", "",
             "| Bits | Actual FastICA / GD-ICA / FIA | Status | PSNR dB | SSIM | VGG LPIPS | Attack time |",
             "|---|---|---|---:|---:|---:|---|",
             "| FP32 | 0 / 0 / 0 | Whitening guard stopped execution | unavailable | unavailable | unavailable | unavailable |"]
    for row in rows[1:]:
        sec = row["attack_seconds"]
        lines.append(f"| {row['bits']} | 4000 / 4000 / 25000 | " +
                     ("Completed existing run" if row["existing_result"] else "Completed") +
                     f" | {row['mean_psnr_db']:.6f} | {row['mean_ssim']:.6f} | {row['mean_lpips']:.6f} | {int(sec)//60}m {sec%60:.1f}s |")
    lines += ["", failure_text.split("\n\n", 1)[1],
              f"New worker total time: {record['wall_seconds']:.3f}s ({record['wall_seconds']/60:.3f}min), including the failed FP32 attempt, INT8 attack and evaluation. The prior INT4 run was reused without rerunning.", "",
              "All16 final images evaluated for INT8/INT4. Pixel-MSE Hungarian matching is computed once per arm and shared by MSE/PSNR/SSIM/LPIPS. Only truth x is consumed after both attack attempts; no private feedback or intermediate selection.", "",
              visual, "",
              "INT8 FastICA first crossed tolerance at27, but final residual2.02655792e-6 exceeds1e-6. Exact4000 updates completed with early stopping disabled.", "",
              "[INT8 original/reconstructed comparison](int8/comparison.png)", "",
              "[Existing INT4 comparison](../single_client_int4_fast4000_gd4000_fia25000_20261006/comparison.png)", "",
              "[FP32 failure report](fp32/report.md)", ""]
    report = "\n".join(lines)
    (r.PRIVATE / "report.md").write_text(report, encoding="utf-8")
    (HERE / "report.md").write_text(report.replace("(int8/comparison.png)", f"({(r.PRIVATE/'int8/comparison.png').as_posix()})")
                                  .replace("(../single_client_int4_fast4000_gd4000_fia25000_20261006/comparison.png)", f"({r.OLD_METRICS.with_name('comparison.png').as_posix()})")
                                  .replace("(fp32/report.md)", f"({(fp32_private/'report.md').as_posix()})"), encoding="utf-8")
    record.update(completion="PARTIAL_COMPLETION_VERIFIED", reporting_verified=True, result_rows=rows,
                  reporting_exit_status=0, optimization_restarts=0,
                  actual_counts={"fp32": [0, 0, 0], "int8": [4000, 4000, 25000], "cached_int4": [4000, 4000, 25000]},
                  failure_stage="FP32_WHITENING", frozen_artifacts_unchanged=True)
    r.write(HERE / "execution_record.json", record)
    # Read only hashes/counts: this does not execute whitening or optimization.
    r.public_gate("int8")
    r.provenance_gate("fp32")
    r.base.public_gate()
    r.write(HERE / "completion_verification.json", {
        "task_id": r.TASK, "status": "VERIFIED_PARTIAL_COMPLETION", "original_worker_exit_status": 1,
        "reporting_exit_status": 0, "optimization_rerun": False, "new_experiment": False,
        "polling_performed": False, "historical_peer_message_sent": False,
        "fp32_actual_counts": [0, 0, 0], "fp32_failure_sha256": r.digest(failure_path),
        "verified_completed_arms": checks, "int8_evaluation_output_hashes": eval_record["output_hashes"],
        "finalizer_sha256": r.digest(Path(__file__)),
        "reporting_outputs": {str(p): r.digest(p) for p in [r.PRIVATE / "report.md", r.PRIVATE / "comparison.json", fp32_private / "report.md", fp32_private / "failure_summary.json"]},
    })
    print("VERIFIED: FP32 whitening failed before0/0/0; INT8 and cached INT4 exactly4000/4000/25000; no reruns")


if __name__ == "__main__":
    main()
