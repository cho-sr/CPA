from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import qacpa.signed_combination_ica_evaluation as evaluation
import qacpa.signed_combination_ica_experiment as experiment
import qacpa.signed_combination_ica_runner as runner


def _profile(exact: bool) -> dict:
    per_tensor = {}
    for name in runner.FC_NAMES:
        per_tensor[name] = {
            "name": name, "numel": 4, "fixed_s": 0.5,
            "code_mismatch_count": 0 if exact else 1,
            "code_mismatch_fraction": 0.0 if exact else 0.25,
            "bounded_cell_violation_count": 0 if exact else 1,
            "bounded_cell_violation_fraction": 0.0 if exact else 0.25,
            "mean_violation_in_s": 0.0 if exact else 0.1,
            "max_violation_in_s": 0.0 if exact else 0.2,
            "cell_penalty": 0.0 if exact else 0.1,
            "candidate_scale": 0.5,
        }
    return {
        "per_tensor": per_tensor, "overall_numel": 24,
        "overall_code_mismatch_count": 0 if exact else 6,
        "overall_code_mismatch_fraction": 0.0 if exact else 0.25,
        "overall_bounded_cell_violation_count": 0 if exact else 6,
        "overall_bounded_cell_violation_fraction": 0.0 if exact else 0.25,
        "cell_loss_equal_tensor_mean": 0.0 if exact else 0.1,
        "exact_feasible": exact, "feature_norm_mean": 1.0,
        "feature_norm_min": 1.0, "feature_norm_max": 1.0,
    }


def test_endpoint_profile_validation_retains_all_six_and_infeasibility():
    profile = runner._validate_endpoint_profile(_profile(False))
    assert set(profile["per_tensor"]) == set(runner.FC_NAMES)
    assert profile["exact_feasible"] is False
    assert profile["overall_code_mismatch_count"] == 6
    assert profile["overall_bounded_cell_violation_count"] == 6


def test_aligned_figure_order_reuses_one_hungarian_correspondence():
    candidates = {"set": torch.tensor([[[[10.0]]], [[[20.0]]], [[[30.0]]]])}
    reports = {"set": {"per_image": [
        {"candidate_index": 0, "truth_index": 2},
        {"candidate_index": 1, "truth_index": 0},
        {"candidate_index": 2, "truth_index": 1},
    ]}}
    aligned = evaluation._aligned_candidates(candidates, reports)["set"]
    assert aligned[:, 0, 0, 0].tolist() == [20.0, 30.0, 10.0]


def test_outer_registration_explicitly_binds_implementation_and_experiment_ids(
        tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "_assert_expected_repair_state", lambda: {
        "protected_source_hashes": {"protected": "a"},
        "shared_source_hashes": {"shared": "b"},
        "adapted_reference_hashes": {}, "repair_new_file_hashes": {},
        "repair_record": "repair.json",
    })
    monkeypatch.setattr(experiment, "_short_controls", lambda: {
        "metadata": {}, "candidate_hashes": {}, "paths": {},
    })
    monkeypatch.setattr(experiment.evaluation, "_public_input_hashes", lambda: {"input": "hash"})
    monkeypatch.setattr(experiment.evaluation, "_code_hashes", lambda: {"code": "hash"})
    registration = experiment.register(
        report=tmp_path / "report", public_output=tmp_path / "public",
        private_output=tmp_path / "private")
    assert registration["task_id"] == experiment.TASK_ID
    assert registration["implementation_task_id"] == runner.TASK_ID
    assert registration["stage_order"] == ["SEPARATE", "INVERT", "REFINE"]
    assert registration["private_policy"]["load_after_public_gate"] is True
    assert json.loads((tmp_path / "report/registration_initial.json").read_text(encoding="utf-8"))["status"] == "REGISTERED"


def test_stage_failure_prevents_dependent_stages_and_private_entry(tmp_path, monkeypatch):
    report = tmp_path / "report"
    report.mkdir()
    public = tmp_path / "public"
    private = tmp_path / "private"
    registration = {
        "task_id": experiment.TASK_ID, "status": "REGISTERED",
        "code_hashes": {"code": "hash"},
    }
    (report / "registration.json").write_text(json.dumps(registration), encoding="utf-8")
    monkeypatch.setattr(experiment.evaluation, "_code_hashes", lambda: {"code": "hash"})
    calls = []

    def fake_process(stage, command, output, started):
        calls.append(stage)
        return {"stage": stage, "status": "FAILED", "exit_status": 9}

    result = experiment.run_registered(report=report, public_output=public,
                                       private_output=private, process_runner=fake_process)
    assert result == 1
    assert calls == ["SEPARATE"]
    summary = json.loads((report / "execution_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED_SEPARATE"
    assert not private.exists()


def test_mocked_success_keeps_public_gate_before_private_phase(tmp_path, monkeypatch):
    report = tmp_path / "report"
    report.mkdir()
    public = tmp_path / "public"
    private = tmp_path / "private"
    registration = {
        "task_id": experiment.TASK_ID, "status": "REGISTERED",
        "code_hashes": {"code": "hash"},
    }
    (report / "registration.json").write_text(json.dumps(registration), encoding="utf-8")
    monkeypatch.setattr(experiment.evaluation, "_code_hashes", lambda: {"code": "hash"})
    order = []
    validation = {
        "source_sha256": "source", "target_z_sha256": "target",
        "candidate_sha256": "direct", "final_candidate_sha256": "cell",
    }
    monkeypatch.setattr(experiment, "_validate_stage", lambda stage, root: order.append("validate:" + stage) or validation)

    def fake_gate(**kwargs):
        order.append("public_gate")
        return {"status": "PUBLIC_GATE_PASSED", "private_loaded": False}

    monkeypatch.setattr(experiment.evaluation, "validate_public_gate", fake_gate)

    def fake_process(stage, command, output, started):
        order.append(stage)
        if stage == "PRIVATE_EVALUATION":
            private.mkdir()
            (private / "comparison.json").write_text(
                json.dumps({"status": "COMPLETED", "scientific_quality": "EVALUATED"}),
                encoding="utf-8")
            (private / "report.md").write_text("done\n", encoding="utf-8")
        return {"stage": stage, "status": "COMPLETED", "exit_status": 0}

    result = experiment.run_registered(report=report, public_output=public,
                                       private_output=private, process_runner=fake_process)
    assert result == 0
    assert order == ["SEPARATE", "validate:SEPARATE", "INVERT", "validate:INVERT",
                     "REFINE", "validate:REFINE", "public_gate", "PRIVATE_EVALUATION"]
    summary = json.loads((report / "execution_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETED"
    assert summary["private_loaded"] is True


def test_mutated_public_binding_is_rejected_before_private_truth_reader(tmp_path, monkeypatch):
    public = tmp_path / "public"
    public.mkdir()
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps({
        "task_id": evaluation.TASK_ID, "plan_version": 1, "status": "REGISTERED",
        "implementation_task_id": evaluation.IMPLEMENTATION_TASK_ID,
        "method": runner.METHOD, "stage_order": ["SEPARATE", "INVERT", "REFINE"],
        "public_output_root": public.resolve().as_posix(),
        "public_input_hashes": {"input": "registered"},
        "integrity_before": {"protected_source_hashes": {}, "shared_source_hashes": {}},
        "code_hashes": {},
    }), encoding="utf-8")
    monkeypatch.setattr(evaluation, "_public_input_hashes", lambda: {"input": "changed"})
    with pytest.raises(evaluation.EvaluationError, match="public input hash changed"):
        evaluation.validate_public_gate(registration_path=registration_path,
                                        public_output=public)


def test_contrast_directions_are_fixed_and_infeasible_profile_is_reportable():
    means = {
        "original_separator_direct": {name: 1.0 for name in evaluation.METRIC_NAMES},
        "original_separator_cell": {name: 1.0 for name in evaluation.METRIC_NAMES},
        "combination_separator_direct": {name: 2.0 for name in evaluation.METRIC_NAMES},
        "combination_separator_cell": {name: 3.0 for name in evaluation.METRIC_NAMES},
    }
    contrasts = evaluation._contrasts(means)
    assert contrasts["favorable_directions"]["mean_mse"] == "lower"
    assert contrasts["differences_are_left_minus_right"][
        "combination_cell_vs_combination_direct"]["mean_mse"] == 1.0
    assert "no threshold" in contrasts["interpretation"]
