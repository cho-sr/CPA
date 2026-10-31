from copy import deepcopy
import inspect
import json

import pytest
import torch

from qacpa.signed_e2e_nofia_runner import (
    H1_LABELS, LAMBDA_CELL, LAMBDA_FIA, NEW_ARM, compare_canonical_profiles,
    compose_optimizer_gradient, validate_structured_provenance,
)
from qacpa.signed_e2e_nofia_evaluation import align_candidates_to_truth
from qacpa.signed_e2e_nofia_finalizer import (
    _validate_public_gate, _validate_saved_arm_result,
)


def _canonical_profile():
    tensor_fields = {
        "name": "{name}", "numel": 10, "fixed_s": 0.25,
        "code_mismatch_count": 0, "code_mismatch_fraction": 0.0,
        "bounded_cell_violation_count": 0, "bounded_cell_violation_fraction": 0.0,
        "mean_violation_in_s": 0.0, "max_violation_in_s": 0.0,
        "cell_penalty": 0.0, "candidate_scale": 0.25,
    }
    names = ("classifier.0.bias", "classifier.0.weight", "classifier.3.bias",
             "classifier.3.weight", "classifier.6.bias", "classifier.6.weight")
    return {
        "per_tensor": {name: {**tensor_fields, "name": name} for name in names},
        "overall_numel": 60, "overall_code_mismatch_count": 0,
        "overall_code_mismatch_fraction": 0.0,
        "overall_bounded_cell_violation_count": 0,
        "overall_bounded_cell_violation_fraction": 0.0,
        "cell_loss_equal_tensor_mean": 0.0, "exact_feasible": True,
        "feature_norm_mean": 1.0, "feature_norm_min": 0.5, "feature_norm_max": 1.5,
    }


def test_zero_fia_gradient_is_excluded_but_tv_and_cell_remain_active():
    pixels = torch.tensor([[0.2, 0.4], [0.6, 0.8]], dtype=torch.float32, requires_grad=True)
    fia = (3.0 * pixels).sum()
    tv = (pixels.square()).sum()
    cell = ((pixels - 0.5).square()).sum()
    fia_gradient, = torch.autograd.grad(fia, pixels, retain_graph=True)
    tv_gradient, = torch.autograd.grad(tv, pixels, retain_graph=True)
    cell_gradient, = torch.autograd.grad(cell, pixels)
    actual = compose_optimizer_gradient(fia_gradient, tv_gradient, cell_gradient)
    expected = 0.1 * tv_gradient + LAMBDA_CELL * cell_gradient
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual) > 0
    assert not torch.equal(actual, compose_optimizer_gradient(
        fia_gradient, tv_gradient, cell_gradient, lambda_fia=1.0))


def test_fia_diagnostic_instrumentation_cannot_change_adam_state():
    fia_gradient = torch.tensor([1.0, -2.0], dtype=torch.float32)
    tv_gradient = torch.tensor([0.4, 0.2], dtype=torch.float32)
    cell_gradient = torch.tensor([-0.3, 0.6], dtype=torch.float32)
    one = torch.nn.Parameter(torch.tensor([0.5, 0.5]))
    two = torch.nn.Parameter(torch.tensor([0.5, 0.5]))
    opt_one = torch.optim.Adam([one], lr=0.001, betas=(.9, .999), eps=1e-8, weight_decay=0.0)
    opt_two = torch.optim.Adam([two], lr=0.001, betas=(.9, .999), eps=1e-8, weight_decay=0.0)
    gradient = compose_optimizer_gradient(fia_gradient, tv_gradient, cell_gradient)
    one.grad = gradient
    two.grad = 0.1 * tv_gradient + LAMBDA_CELL * cell_gradient
    opt_one.step()
    opt_two.step()
    assert torch.equal(one, two)
    one_state, two_state = opt_one.state_dict()["state"], opt_two.state_dict()["state"]
    assert one_state.keys() == two_state.keys()
    for key in one_state:
        assert one_state[key].keys() == two_state[key].keys()
        for field in one_state[key]:
            left, right = one_state[key][field], two_state[key][field]
            assert torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right


def test_h1_and_single_arm_provenance_are_fixed():
    assert H1_LABELS == (507, 655, 452, 549, 231, 657, 160, 761,
                         975, 389, 658, 761, 621, 124, 969, 290)
    assert LAMBDA_FIA == 0.0
    assert NEW_ARM == "C_H1_high_noFIA"
    assert LAMBDA_CELL == 0.24202997768915402


def test_complete_canonical_profile_allows_only_documented_diagnostics():
    expected = _canonical_profile()
    actual = deepcopy(expected)
    actual.update({"label_hypothesis": "H1", "feature_target_cosine_loss": 0.2,
                   "tv_normalized": 0.1, "pixel_min": 0.0, "pixel_max": 1.0,
                   "pixel_mean": 0.5, "x0_displacement_l2": 0.0})
    comparison = compare_canonical_profiles(actual, expected)
    assert comparison["passed"] is True
    assert comparison["mismatches"] == []

    missing_tensor = deepcopy(expected)
    del missing_tensor["per_tensor"]["classifier.6.weight"]
    assert compare_canonical_profiles(missing_tensor, expected)["passed"] is False

    unexpected = deepcopy(expected)
    unexpected["unregistered"] = 1
    assert compare_canonical_profiles(unexpected, expected)["passed"] is False


@pytest.mark.parametrize("field", ["overall_numel", "exact_feasible"])
def test_canonical_required_integer_and_boolean_types_are_exact(field):
    expected = _canonical_profile()
    actual = deepcopy(expected)
    actual[field] = 1 if field == "overall_numel" else 1
    comparison = compare_canonical_profiles(actual, expected)
    assert comparison["passed"] is False
    if field == "overall_numel":
        assert any(item["path"] == "profile.overall_numel" for item in comparison["mismatches"])
    else:
        assert any(item["path"] == f"actual.{field}" for item in comparison["schema_errors"]["actual"])


def test_canonical_float_difference_is_reported_without_changing_bounds():
    expected = _canonical_profile()
    within = deepcopy(expected)
    within["feature_norm_mean"] += 1e-8
    comparison = compare_canonical_profiles(within, expected)
    assert comparison["passed"] is True
    diff = next(item for item in comparison["float_differences"] if item["path"] == "profile.feature_norm_mean")
    assert diff["difference"] == pytest.approx(1e-8)
    outside = deepcopy(expected)
    outside["feature_norm_mean"] += 1e-3
    assert compare_canonical_profiles(outside, expected)["passed"] is False


def test_structured_json_authentication_checks_actual_hashes_and_windows_paths():
    path = r"C:\Users\조상제\Desktop\CPA\data\manifest.json"
    parsed = json.loads(json.dumps({"checkpoint": path}, ensure_ascii=False))
    assert parsed["checkpoint"] == path
    registration = {"task_id": "CPA_SIGNED_E2E_CELL_SCREEN_20260916", "plan_version": 1,
                    "status": "REGISTERED", "input_hashes": {path: "a"},
                    "protected_source_hashes_before": {"C:\\code.py": "b"}}
    integrity = {"protected_sources_unchanged": True, "inputs_unchanged": True,
                 "input_hashes_before": {path: "a"}, "input_hashes_after": {path: "a"},
                 "protected_source_hashes_before": {"C:\\code.py": "b"},
                 "protected_source_hashes_after": {"C:\\code.py": "b"}}
    actual_inputs = {path.replace("\\", "/"): "a"}
    actual_sources = {"C:/code.py": "b"}
    assert validate_structured_provenance(registration, integrity, actual_inputs, actual_sources)["passed"] is True
    changed = validate_structured_provenance(registration, integrity,
                                             {path: "changed"}, actual_sources)
    assert changed["passed"] is False
    assert any("recorded_vs_current" in item["path"] for item in changed["mismatches"])


def test_figure_alignment_reuses_recorded_correspondence():
    candidates = torch.tensor([[10.0], [20.0], [30.0]])
    originals = torch.tensor([[100.0], [200.0], [300.0]])
    report = {"per_image": [{"candidate_index": 1, "truth_index": 0},
                             {"candidate_index": 2, "truth_index": 1},
                             {"candidate_index": 0, "truth_index": 2}]}
    aligned = align_candidates_to_truth(candidates, report, originals)
    assert torch.equal(aligned, torch.tensor([[20.0], [30.0], [10.0]]))


def test_finalizer_has_no_optimization_entry_and_rejects_non100_saved_arm():
    import qacpa.signed_e2e_nofia_finalizer as finalizer
    source = inspect.getsource(finalizer.finalize_saved_candidates)
    assert "run_ablation(" not in source
    assert "_run_arm(" not in source
    assert "torch.optim.Adam" not in source
    result = {"status": "COMPLETED", "arm": NEW_ARM, "iterations": 99,
              "lambda_fia": 0.0, "lambda_tv": 0.1, "lambda_cell": LAMBDA_CELL,
              "candidate_sha256": "x"}
    with pytest.raises(finalizer.FinalizationError):
        _validate_saved_arm_result(result, [], expected_arm=NEW_ARM, expected_hash="x",
                                   expected_lambda_fia=0.0)


def test_private_phase_gate_rejects_failed_public_provenance_first():
    with pytest.raises(Exception, match="public gate failed"):
        _validate_public_gate({"protected_sources_unchanged": False})
