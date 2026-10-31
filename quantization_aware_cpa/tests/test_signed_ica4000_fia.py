from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from qacpa import signed_ica4000_fia_evaluation as evaluation
from qacpa import signed_ica4000_fia_runner as runner
from qacpa.signed_combination_ica import CombinationICAConfig, separate_observation


def test_only_refinement_iteration_changes_between_registered_configs():
    assert runner.config_difference() == {"refinement_iterations": (1000, 4000)}
    assert runner.new_config()["refinement_iterations"] == 4000
    assert {key: value for key, value in runner.new_config().items()
            if key != "refinement_iterations"} == {
                key: value for key, value in runner.old_config().items()
                if key != "refinement_iterations"
            }


def test_pure_core_honors_supplied_iteration_count_on_small_fixture():
    torch.manual_seed(42)
    observation = torch.randn(4, 80, dtype=torch.float32)
    config = CombinationICAConfig(fast_max_iter=3, refinement_iterations=7)
    result = separate_observation(observation, 4, config=config, device="cpu")
    assert result.refinement.iterations == 7
    assert len(result.refinement.log) == 7
    assert result.refinement.log[-1]["iteration"] == 7
    assert torch.equal(result.target_z, result.S_hat.abs())


def test_signed_abs_artifact_contract_is_exact_for_full_shape():
    signed = torch.randn((runner.SOURCE_COUNT, runner.FEATURE_DIM), dtype=torch.float32)
    target = signed.abs().contiguous()
    got_signed, got_target = runner._validate_tensors(signed, target)
    assert got_signed.shape == (16, 25088)
    assert got_signed.dtype == torch.float32
    assert torch.equal(got_target, got_signed.abs())
    with pytest.raises(runner.RunnerError, match="exact elementwise abs"):
        runner._validate_tensors(signed, target + torch.finfo(torch.float32).eps)


def test_fia_bridge_passes_the_exact_new_target_and_registered_settings():
    target = torch.rand((runner.SOURCE_COUNT, runner.FEATURE_DIM), dtype=torch.float32)
    captured = {}

    def fake_fia(*, target_z, device, config, output_dir, seed):
        captured.update(target=target_z, device=device, config=config,
                        output_dir=output_dir, seed=seed)
        return {"rec": torch.zeros((16, 3, 224, 224), dtype=torch.float32), "log": []}

    result = runner.invoke_direct_fia(target_z=target, output_dir=Path("mock-output"),
                                      device="cuda", direct_fia=fake_fia)
    assert torch.equal(captured["target"], target)
    assert captured["device"] == "cuda"
    assert captured["config"] == runner.FIA_CONFIG
    assert captured["seed"] == 42
    assert result["rec"].shape == (16, 3, 224, 224)


def test_collision_and_shape_guards_fail_without_touching_inputs(tmp_path):
    collision = tmp_path / "existing"
    collision.mkdir()
    with pytest.raises(FileExistsError):
        runner.separate(output=collision)
    with pytest.raises(runner.RunnerError, match="shape"):
        runner._validate_tensors(torch.zeros((16, 4)), torch.zeros((16, 4)))


def test_new_orchestration_has_no_cell_or_replay_entry_point():
    wrapper = (runner.REPO_ROOT / ".ai/execution-results/CPA_SIGNED_ICA4000_FIA_20260917/run_real_ica4000_fia.py").read_text(encoding="utf-8")
    assert 'STAGES = ("SEPARATE", "INVERT", "PRIVATE_EVALUATION")' in wrapper
    assert "actual_update_refinement" not in wrapper
    assert "qacpa.fc_replay" not in wrapper
    assert "labels_from_public_logits" not in wrapper
    assert "build_refinement_labels" not in wrapper


def test_public_gate_is_called_before_private_loader(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(evaluation, "validate_public_gate",
                        lambda **kwargs: events.append("public_gate") or {"registration": {}})
    monkeypatch.setattr(evaluation, "_load_private_x",
                        lambda path: events.append("private_load") or
                        (torch.zeros((16, 3, 224, 224)), {"consumed_field": "x only"}))
    monkeypatch.setattr(evaluation, "_lpips", lambda *args: {})
    monkeypatch.setattr(evaluation, "_write_figure", lambda *args: "figure.png")
    monkeypatch.setattr(evaluation, "_control_reproduction", lambda *args: {
        "status": "PASSED", "tolerance": {}, "difference": {},
        "within_tolerance": {}, "per_image_residuals": [],
    })
    monkeypatch.setattr(evaluation, "_image_improvement", lambda *args: {})
    monkeypatch.setattr(evaluation, "_contrasts", lambda *args: {})
    monkeypatch.setattr(evaluation, "_load_reviewed", lambda: {
        "mean_metrics": {"combination_separator_direct": {name: 0.0 for name in evaluation.METRIC_NAMES}}
    })
    monkeypatch.setattr(evaluation, "_norm_map", lambda value: value)
    monkeypatch.setattr(evaluation.runner, "protected_integrity", lambda: {})
    monkeypatch.setattr(evaluation.runner, "shared_integrity", lambda: {})
    monkeypatch.setattr(evaluation.runner, "public_input_hashes", lambda: {})
    monkeypatch.setattr(evaluation, "file_sha256", lambda path: "hash")
    monkeypatch.setattr(evaluation.runner, "_validate_candidate",
                        lambda path, expected_hash=None: torch.zeros((16, 3, 224, 224)))
    monkeypatch.setattr(evaluation, "evaluate_candidate", lambda *args, **kwargs: {
        "per_image": [{"candidate_index": i, "truth_index": i, "mse": 0.0,
                        "psnr_db": 0.0, "ssim": 0.0} for i in range(16)]
    })
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"task_id": runner.TASK_ID}), encoding="utf-8")
    # The gate itself is mocked because this test checks ordering only.
    with pytest.raises(Exception):
        evaluation.evaluate(registration_path=registration, public_output=tmp_path / "public",
                            private_truth=evaluation.PRIVATE_TRUTH, output=tmp_path / "private")
    assert events[:2] == ["public_gate", "private_load"]
