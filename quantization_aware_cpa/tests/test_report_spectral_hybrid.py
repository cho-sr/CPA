"""Offline report contract tests; no research candidates/truth or LPIPS runs."""
from copy import deepcopy
import hashlib
import json

import pytest

import qacpa.report_spectral_hybrid as reporting


def _metrics(psnr=10., ssim=.2, lpips=.7):
    return {"mean_psnr_db": psnr, "mean_ssim": ssim, "mean_lpips": lpips}


@pytest.mark.parametrize("candidate,expected,directions", [
    (_metrics(11., .3, .6), "directional_improvement_in_this_pilot", ["improved"] * 3),
    (_metrics(11., .1, .6), "mixed", ["improved", "worsened", "improved"]),
    (_metrics(9., .1, .8), "no_established_improvement", ["worsened"] * 3),
    (_metrics(), "no_established_improvement", ["unchanged"] * 3),
    (_metrics(10., .3, .6), "no_established_improvement", ["unchanged", "improved", "improved"]),
])
def test_direction_rule_requires_all_three_strict_improvements(candidate, expected, directions):
    result = reporting.compare_evaluations(candidate, _metrics())
    assert result["interpretation"] == expected
    assert list(result["metric_directions"].values()) == directions
    assert result["all_three_metrics_strictly_improved"] == (expected == "directional_improvement_in_this_pilot")
    assert result["success_threshold"] is None
    assert result["statistical_significance"] == "not_tested"
    json.dumps(result, allow_nan=False)


def test_signed_differences_are_candidate_minus_reference_and_lpips_is_lower_better():
    result = reporting.compare_evaluations(_metrics(10.1, .22, .68), _metrics())
    assert result["differences_candidate_minus_reference"] == pytest.approx(
        {"mean_psnr_db": .1, "mean_ssim": .02, "mean_lpips": -.02})
    assert result["primary_metric"] == "mean_lpips_lower_is_better"


def test_perfect_psnr_null_retains_infinity_meaning_without_nonjson_numbers():
    result = reporting.compare_evaluations(_metrics(None, .3, .6), _metrics())
    assert result["metric_directions"]["mean_psnr_db"] == "improved"
    assert result["differences_candidate_minus_reference"]["mean_psnr_db"] is None
    assert result["interpretation"] == "directional_improvement_in_this_pilot"
    equal_infinite = reporting.compare_evaluations(_metrics(None), _metrics(None))
    assert equal_infinite["metric_directions"]["mean_psnr_db"] == "unchanged"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "10"])
def test_nonfinite_or_invalid_metrics_are_rejected(bad):
    with pytest.raises(reporting.ReportError):
        reporting.compare_evaluations(_metrics(psnr=bad), _metrics())
    with pytest.raises(reporting.ReportError):
        reporting.compare_evaluations(_metrics(lpips=bad), _metrics())


def test_missing_psnr_is_not_interpreted_as_perfect_match():
    missing = _metrics()
    del missing["mean_psnr_db"]
    with pytest.raises(reporting.ReportError, match="Missing"):
        reporting.compare_evaluations(missing, _metrics())


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _completed_fixture(tmp_path):
    base = tmp_path / "attack"
    base.mkdir()
    common_bytes = b"synthetic public common initialization placeholder"
    (base / "common_initial.pt").write_bytes(common_bytes)
    bindings = {key: "a" * 64 for key in reporting.BINDINGS}
    bindings["common_initial_sha256"] = hashlib.sha256(common_bytes).hexdigest()
    plan_bytes = b'{"scope":"synthetic offline report contract"}\n'
    (base / "registered_plan.json").write_bytes(plan_bytes)
    bindings["plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    arms = {}
    for arm in reporting.ARMS:
        folder = base / arm
        folder.mkdir()
        (folder / "candidate.pt").write_bytes(b"synthetic final artifact placeholder")
        (folder / "attack_state.pt").write_bytes(b"synthetic saved state placeholder")
        fit = {"per_tensor": {key: {} for key in reporting.FC_NAMES},
               "objective_at_same_post_step_candidate": 1.}
        result = {"status": "completed", "arm": arm, "iterations": 50, **bindings,
                  "candidate_sha256": reporting.file_digest(folder / "candidate.pt"),
                  "observation_term_enabled": True, "label_update_rule": "observation_gradient",
                  "fits": {"soft": deepcopy(fit), "hard": deepcopy(fit)}}
        arms[arm] = result
        _write_json(folder / "result.json", result)
    summary = {"status": "completed", "arms": arms, "input_and_source_hashes_unchanged": True, **bindings}
    _write_json(base / "summary.json", summary)
    return base, summary


def test_completion_guard_accepts_matching_three_arm_public_metadata(tmp_path):
    base, _ = _completed_fixture(tmp_path)
    summary, arms = reporting._completed_arms(base)
    assert set(arms) == set(reporting.ARMS)
    assert summary["input_and_source_hashes_unchanged"] is True


@pytest.mark.parametrize("field,value", [("status", "running"), ("iterations", 49),
                                       ("plan_sha256", "b" * 64),
                                       ("common_initial_sha256", "b" * 64),
                                       ("spectrum_npz_sha256", "b" * 64),
                                       ("observation_term_enabled", False),
                                       ("label_update_rule", "fixed")])
def test_invalid_arm_is_blocked_before_candidate_or_truth_load(tmp_path, monkeypatch, field, value):
    base, summary = _completed_fixture(tmp_path)
    summary["arms"]["mp"][field] = value
    _write_json(base / "mp/result.json", summary["arms"]["mp"])
    _write_json(base / "summary.json", summary)

    def forbidden(*args, **kwargs):
        pytest.fail("An incomplete/unmatched run must not load candidates or private truth")

    monkeypatch.setattr(reporting.torch, "load", forbidden)
    monkeypatch.setattr(reporting, "evaluate_candidate", forbidden)
    with pytest.raises(reporting.ReportError):
        reporting.report(base, tmp_path / "never_read_truth.pt", tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_unchanged_hash_flag_and_common_file_are_not_accepted_on_metadata_alone(tmp_path):
    base, summary = _completed_fixture(tmp_path)
    (base / "common_initial.pt").write_bytes(b"changed after optimization")
    with pytest.raises(reporting.ReportError, match="Common initial"):
        reporting._completed_arms(base)
    summary["input_and_source_hashes_unchanged"] = False
    _write_json(base / "summary.json", summary)
    with pytest.raises(reporting.ReportError, match="unchanged"):
        reporting._completed_arms(base)


@pytest.mark.parametrize("relative_path", ["registered_plan.json", "mp/candidate.pt"])
def test_changed_plan_or_candidate_file_is_blocked_before_tensor_or_truth_reads(tmp_path, monkeypatch, relative_path):
    base, _ = _completed_fixture(tmp_path)
    (base / relative_path).write_bytes(b"changed final artifact")

    def forbidden(*args, **kwargs):
        pytest.fail("Hash-mismatched artifacts must be rejected before tensor/truth reads")

    monkeypatch.setattr(reporting.torch, "load", forbidden)
    monkeypatch.setattr(reporting, "evaluate_candidate", forbidden)
    with pytest.raises(reporting.ReportError, match="recorded hash"):
        reporting.report(base, tmp_path / "never_read_truth.pt", tmp_path / "report")


@pytest.mark.parametrize("binding", reporting.BINDINGS)
@pytest.mark.parametrize("mutation", ["missing", "different"])
def test_every_candidate_payload_binding_is_required_and_matches_result(tmp_path, monkeypatch, binding, mutation):
    base, summary = _completed_fixture(tmp_path)
    _, arms = reporting._completed_arms(base)
    payload = {key: summary[key] for key in reporting.BINDINGS}
    if mutation == "missing":
        del payload[binding]
    else:
        payload[binding] = "b" * 64
    monkeypatch.setattr(reporting.torch, "load", lambda *args, **kwargs: payload)
    with pytest.raises(reporting.ReportError, match="observation|binding"):
        reporting._load_candidates(arms, summary["observation_manifest_sha256"])


def test_candidate_changed_between_completion_and_loading_is_rejected(tmp_path, monkeypatch):
    base, summary = _completed_fixture(tmp_path)
    _, arms = reporting._completed_arms(base)
    (base / "identity/candidate.pt").write_bytes(b"changed after completion check")

    def forbidden(*args, **kwargs):
        pytest.fail("Changed candidate must be rejected before tensor loading")

    monkeypatch.setattr(reporting.torch, "load", forbidden)
    with pytest.raises(reporting.ReportError, match="changed before loading"):
        reporting._load_candidates(arms, summary["observation_manifest_sha256"])


def test_missing_fc_tensor_or_final_label_kind_is_rejected(tmp_path):
    base, summary = _completed_fixture(tmp_path)
    del summary["arms"]["flat"]["fits"]["hard"]["per_tensor"]["classifier.0.weight"]
    _write_json(base / "flat/result.json", summary["arms"]["flat"])
    _write_json(base / "summary.json", summary)
    with pytest.raises(reporting.ReportError, match="every FC tensor"):
        reporting._completed_arms(base)


def test_output_overwrite_refused_before_any_inputs_are_opened(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Existing output should be rejected before reads")

    monkeypatch.setattr(reporting, "_completed_arms", forbidden)
    with pytest.raises(FileExistsError):
        reporting.report(tmp_path / "absent", tmp_path / "never_read.pt", output)
    assert marker.read_text(encoding="utf-8") == "keep"
