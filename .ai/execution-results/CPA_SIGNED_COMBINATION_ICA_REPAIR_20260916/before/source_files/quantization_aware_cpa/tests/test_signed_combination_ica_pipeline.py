"""Mocked integration contracts for the future public ICA pipeline."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import qacpa.signed_combination_ica_runner as runner


def test_direct_fia_bridge_passes_new_target_and_registered_defaults(tmp_path):
    target = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    captured = {}

    def fake_direct_fia(**kwargs):
        captured.update(kwargs)
        return {"rec": torch.zeros(16, 3, 224, 224), "log": []}

    result = runner.invoke_direct_fia(target_z=target, output_dir=tmp_path / "fia",
                                      device="cpu", direct_fia=fake_direct_fia)
    assert torch.equal(captured["target_z"], target)
    assert captured["seed"] == 42
    assert captured["config"] == runner.DEFAULT_FIA_CONFIG
    assert result["rec"].shape == (16, 3, 224, 224)


def test_actual_update_bridge_passes_new_target_candidate_and_all_fc_tables(tmp_path):
    names = tuple(runner._device_tables.__globals__["FC_NAMES"])
    tables = {name: object() for name in names}
    device_tables = {name: object() for name in names}
    candidate = torch.full((2, 3, 4, 4), 0.25)
    target = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    labels = torch.tensor([3, 7])
    captured = {}

    def fake_refiner(*args):
        captured["args"] = args
        return {"status": "MOCK_COMPLETED"}

    result = runner.invoke_actual_update_refinement(
        model=object(), candidate=candidate, target_z=target, labels=labels,
        tables=tables, device_tables=device_tables, output=tmp_path,
        device=torch.device("cpu"), clock=object(), image_refiner=fake_refiner)
    args = captured["args"]
    assert args[1] == runner.METHOD
    assert torch.equal(args[2], candidate)
    assert torch.equal(args[3], target)
    assert torch.equal(args[4], labels)
    assert args[5] == runner.FIXED_CELL_WEIGHT
    assert set(args[6]) == set(names)
    assert set(args[7]) == set(names)
    assert result["status"] == "MOCK_COMPLETED"


def test_public_h1_labels_are_recomputed_from_current_candidate(monkeypatch):
    candidate = torch.rand(16, 3, 2, 2)
    seen = {}

    def fake_features(model, pixels):
        seen["candidate"] = pixels.detach().clone()
        return torch.zeros(16, 4)

    def fake_labels(logits, bias_q):
        seen["logits"] = logits
        return {"status": "COMPLETED", "assignment": []}, {"H1": torch.arange(16)}

    monkeypatch.setattr(runner, "_feature_forward", fake_features)
    monkeypatch.setattr(runner, "labels_from_public_logits", fake_labels)
    tables = {"classifier.6.bias": SimpleNamespace(q_cpu=torch.zeros(1000, dtype=torch.int8))}
    model = SimpleNamespace(classifier=lambda features: torch.zeros(16, 1000))
    evidence, labels = runner.build_refinement_labels(model, candidate, tables,
                                                       torch.device("cpu"))
    assert torch.equal(seen["candidate"], candidate)
    assert evidence["status"] == "COMPLETED"
    assert torch.equal(labels, torch.arange(16, dtype=torch.int64))


def test_candidate_schema_and_hash_contract_reject_invalid_artifacts(tmp_path):
    path = tmp_path / "candidate.pt"
    torch.save({"pixels": torch.zeros(1)}, path)
    with pytest.raises(runner.PipelineError, match="invalid dtype, shape"):
        runner._validate_candidate(path)
    with pytest.raises(runner.PipelineError, match="candidate hash mismatch"):
        runner._validate_candidate(path, "0" * 64)


def test_hash_binding_is_exact_and_does_not_accept_missing_or_extra_inputs():
    expected = {"A": "1", "B": "2"}
    runner._require_same_hashes(expected, {"a": "1", "b": "2"}, label="test")
    with pytest.raises(runner.PipelineError, match="hash binding changed"):
        runner._require_same_hashes(expected, {"A": "1"}, label="test")
