"""Scientific inference checks; synthetic LPIPS only, no image reconstruction."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from qacpa import m7_report as report


def rows(value: float = 0.8) -> list[dict]:
    return [{"global_index": gid, "batch_index": gid // 8, "lpips": value} for gid in range(40)]


def test_sign_flip_enumerates_32_patterns_and_includes_ties() -> None:
    positive = report.exact_sign_flip([1, 2, 3, 4, 5])
    assert positive["total_sign_patterns"] == 32
    assert positive["p_one_sided"] == 1 / 32
    assert report.exact_sign_flip([0, 0, 0, 0, 0])["p_one_sided"] == 1.0
    assert report.exact_sign_flip([-1, -2, -3, -4, -5])["p_one_sided"] == 1.0
    # A zero-effect batch gives two equally extreme sign assignments.
    assert report.exact_sign_flip([1, 1, 1, 1, 0])["p_one_sided"] == 2 / 32
    with pytest.raises(ValueError, match="five finite"):
        report.exact_sign_flip([1, 2, 3, 4])


def test_iut_requires_both_positive_significant_effects() -> None:
    cpa = rows(0.5)
    both = report.group_comparison(cpa, rows(0.8), rows(0.9))
    assert both["iut_p"] == 1 / 32
    assert both["passes_preregistered_rule"]
    only_one = report.group_comparison(cpa, rows(0.8), rows(0.5))
    assert only_one["iut_p"] == 1.0
    assert not only_one["passes_preregistered_rule"]


def test_calibration_holds_out_whole_batch_and_preserves_precision() -> None:
    fp32, quantized, prior = rows(0.7), rows(0.8), rows(0.9)
    for row in fp32[:8]:
        row["lpips"] = 0.1
    thresholds = report.calibrate_thresholds({"fp32": fp32, "4bit": quantized}, prior, prereg_sha256="synthetic")
    assert thresholds["calibration_uses_candidate_values"] is False
    first = thresholds["by_precision"]["fp32"][0]
    assert first["strict_lpips_upper_bound"] == 0.7
    assert first["n_reference_each"] == 32
    assert len(first["reference_lpips"]["no_cpa"]) == 32
    assert thresholds["by_precision"]["fp32"][1]["strict_lpips_upper_bound"] == 0.1
    assert thresholds["by_precision"]["4bit"][0]["strict_lpips_upper_bound"] == 0.8


def test_rank_success_is_strict_and_ties_fail() -> None:
    thresholds = report.calibrate_thresholds({"fp32": rows(0.7), "4bit": rows(0.8)}, rows(0.9), prereg_sha256="synthetic")
    below = report.candidate_success(rows(np.nextafter(0.7, 0.0)), method="fp32_cpa", precision="fp32", thresholds=thresholds)
    ties = report.candidate_success(rows(0.7), method="fp32_cpa", precision="fp32", thresholds=thresholds)
    assert all(row["success_control_calibrated"] == 1 for row in below)
    assert all(row["rank_p_no_cpa"] == 1 / 33 for row in below)
    assert all(row["success_control_calibrated"] == 0 for row in ties)


def test_duplicate_or_partial_controls_cannot_calibrate() -> None:
    duplicate = rows()
    duplicate[-1] = dict(duplicate[-2])
    assert not report.complete_rows(duplicate)
    bad_batch = rows()
    bad_batch[-1]["batch_index"] = 0
    assert not report.complete_rows(bad_batch)
    with pytest.raises(ValueError, match="no partial calibration"):
        report.calibrate_thresholds({"fp32": rows(), "4bit": rows()[:-1]}, rows(), prereg_sha256="synthetic")
    with pytest.raises(ValueError, match="incomplete candidate"):
        report.candidate_success(rows()[:-1], method="fp32_cpa", precision="fp32", thresholds={})


def test_same36_sensitivity_pairs_identical_originals_and_keeps_five_batches() -> None:
    cpa = rows(0.5)
    for gid in (22, 24, 32, 36):
        cpa[gid]["lpips"] = 0.0
    selected = set(range(40)) - {22, 24, 32, 36}
    result = report.group_comparison(cpa, rows(0.8), rows(0.9), ids=selected)
    assert result["n_images"] == 36
    assert result["n_batches"] == 5
    assert result["mean_lpips_by_condition"]["cpa"] == 0.5
    assert result["no_cpa"]["batch_effects_control_minus_cpa"] == pytest.approx([0.3] * 5)


def test_csv_without_completion_marker_is_pending(tmp_path) -> None:
    report.write_csv(tmp_path / "per_image_metrics.csv", rows())
    assert report.load_completed_control(tmp_path, protocol_hash="locked") == []
    report.write_json(tmp_path / "summary.json", {"identity": {"protocol_sha256": "changed"}})
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        report.load_completed_control(tmp_path, protocol_hash="locked")


def test_partial_pipeline_never_emits_zero_success_counts(tmp_path, monkeypatch) -> None:
    prereg = json.loads(report.PREREG.read_text(encoding="utf-8"))
    spec = tmp_path / "M7_CONTROL_PREREGISTRATION.json"
    spec.write_text(json.dumps(prereg), encoding="utf-8")
    report.write_json(tmp_path / "M7_CONTROL_PREREGISTRATION.lock.json", {"sha256": hashlib.sha256(spec.read_bytes()).hexdigest()})
    historical = []
    for method, mean_key in (("fp32", "fp32_requested_mean"), ("naive", "4bit_requested_mean")):
        historical.extend({**row, "method": method} for row in rows(prereg["historical"][mean_key]))
    report.write_csv(tmp_path / prereg["historical"]["csv"], historical)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "load_completed_control", lambda folder, **kwargs: rows() if folder.name == "fp32" else [])
    monkeypatch.setattr(report, "histogram", lambda *args, **kwargs: None)
    monkeypatch.setattr(report, "random_grid", lambda *args, **kwargs: {"global_indices": list(range(12)), "available_columns": ["original"]})
    output = tmp_path / "no_cpa"
    summary = report.generate_report(output, tmp_path / "prior", tmp_path / "recompute", spec)
    assert summary["report_status"] == "pending"
    assert summary["primary_claim_supported"] is None
    assert summary["group_inference"] == {}
    assert all(row["success_count"] == "PENDING" for row in summary["grid_2x2"])
    assert not (output / "thresholds.json").exists()
    assert report.read_csv(output / "image_success.csv") == []


def test_complete_report_serializes_thresholds_before_counts_and_closes_only_finished_fia(tmp_path, monkeypatch) -> None:
    prereg = json.loads(report.PREREG.read_text(encoding="utf-8"))
    spec = tmp_path / "M7_CONTROL_PREREGISTRATION.json"
    spec.write_text(json.dumps(prereg), encoding="utf-8")
    report.write_json(tmp_path / "M7_CONTROL_PREREGISTRATION.lock.json", {"sha256": hashlib.sha256(spec.read_bytes()).hexdigest()})
    historical = []
    for method, mean_key in (("fp32", "fp32_requested_mean"), ("naive", "4bit_requested_mean")):
        historical.extend({**row, "method": method} for row in rows(prereg["historical"][mean_key]))
    report.write_csv(tmp_path / prereg["historical"]["csv"], historical)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "load_completed_control", lambda folder, **kwargs: rows(0.9 if folder.name == "prior_only" else 0.8))
    monkeypatch.setattr(report, "histogram", lambda *args, **kwargs: None)
    monkeypatch.setattr(report, "random_grid", lambda *args, **kwargs: {"global_indices": list(range(12)), "available_columns": ["original", "fp32", "4bit"]})
    monkeypatch.setattr(report, "load_recompute_distributions", lambda *args, **kwargs: {
        key: [0.8] * (800 if "shuffled" in key else 40)
        for key in ("fp32_shuffled", "4bit_shuffled", "dataset_mean", "gray", "noise")
    })
    monkeypatch.setattr(report, "shuffled_seed_distribution", lambda *args: {})
    output = tmp_path / "no_cpa"
    original_counter = report.candidate_success
    def assert_already_serialized(*args, **kwargs):
        frozen = json.loads((output / "thresholds.json").read_text(encoding="utf-8"))
        assert frozen == kwargs["thresholds"]
        return original_counter(*args, **kwargs)
    monkeypatch.setattr(report, "candidate_success", assert_already_serialized)
    execution = output / "execution_status.json"
    report.write_json(execution, {"state": "running", "pid": 42})
    summary = report.generate_report(output, tmp_path / "prior", tmp_path / "recompute", spec)
    assert summary["report_status"] == "complete"
    assert summary["primary_claim_supported"] is True
    assert [row["success_count"] for row in summary["grid_2x2"]] == [40, 0, 40, 0]
    assert len(report.read_csv(output / "image_success.csv")) == 160
    assert json.loads(execution.read_text(encoding="utf-8"))["state"] == "running"
    report.write_json(execution, {"state": "fia_complete", "pid": 42, "started_at_utc": "preserved"})
    report.generate_report(output, tmp_path / "prior", tmp_path / "recompute", spec)
    finished = json.loads(execution.read_text(encoding="utf-8"))
    assert finished["state"] == "complete"
    assert finished["pid"] == 42
    assert finished["started_at_utc"] == "preserved"
    assert "report_completed_at_utc" in finished
