"""Mocked integration contracts for the future public ICA pipeline."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import qacpa.signed_combination_ica_runner as runner
from qacpa.signed_e2e_cells import FC_NAMES


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
    names = tuple(FC_NAMES)
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
    assert args[1] == runner.REFINE_ARM_NAME
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


def _configure_small_public_fixture(monkeypatch, tmp_path, *, source_count=2, feature_dim=4):
    observation = tmp_path / "observation"
    observation.mkdir()
    (observation / "packet.bin").write_bytes(b"synthetic-public-packet")
    manifest = {"layers": {"classifier.0.weight": {"file": "packet.bin"}}}
    (observation / "manifest.json").write_text(
        __import__("json").dumps(manifest), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    monkeypatch.setattr(runner, "PUBLIC_OBSERVATION", observation)
    monkeypatch.setattr(runner, "CHECKPOINT", checkpoint)
    monkeypatch.setattr(runner, "CHECKPOINT_SHA", runner.file_sha256(checkpoint))
    monkeypatch.setattr(runner, "AUTHORIZED_MANIFEST_SHA",
                        runner.file_sha256(observation / "manifest.json"))
    monkeypatch.setattr(runner, "SOURCE_COUNT", source_count)
    monkeypatch.setattr(runner, "FEATURE_DIM", feature_dim)
    monkeypatch.setitem(runner.FC_SHAPES, "classifier.0.weight", (source_count, feature_dim))
    packet_manifest = manifest
    entries = {"classifier.0.weight": (
        torch.zeros(source_count, feature_dim, dtype=torch.int8),
        torch.tensor(0.5, dtype=torch.float32),
    )}
    monkeypatch.setattr(runner, "read_bundle", lambda path: (packet_manifest, entries))
    return observation, checkpoint


def _write_small_source_artifact(monkeypatch, tmp_path):
    source = torch.linspace(-1.0, 1.0, runner.SOURCE_COUNT * runner.FEATURE_DIM).reshape(
        runner.SOURCE_COUNT, runner.FEATURE_DIM).contiguous()
    target = source.abs()
    separation = SimpleNamespace(
        S_hat=source, target_z=target,
        refinement=SimpleNamespace(log=[], final_losses={"total": 0.0}),
        metadata={"fastica": {}, "refinement": {}, "whitening": {}},
    )
    monkeypatch.setattr(runner, "separate_observation", lambda *args, **kwargs: separation)
    output = tmp_path / "sources"
    result = runner.separate(observation=runner.PUBLIC_OBSERVATION,
                             checkpoint=runner.CHECKPOINT,
                             output=output, device="cpu")
    registration, loaded_source, loaded_target = runner._load_sources(output)
    return output, result, registration, loaded_source, loaded_target


def test_real_source_writer_loader_and_separate_to_invert_round_trip(tmp_path, monkeypatch):
    observation, checkpoint = _configure_small_public_fixture(monkeypatch, tmp_path)
    source_root, separate_result, registration, source, target = _write_small_source_artifact(
        monkeypatch, tmp_path)
    assert source.shape == (2, 4)
    assert torch.equal(target, source.abs())
    assert registration["source_shape"] == [2, 4]
    assert separate_result["source_shape"] == [2, 4]
    assert (source_root / "integrity.json").is_file()
    assert json_load(source_root / "result.json")["source_shape"] == [2, 4]

    captured = {}
    candidate = torch.full((2, 3, 224, 224), 0.25, dtype=torch.float32)

    def fake_direct_fia(**kwargs):
        captured.update(kwargs)
        return {"rec": candidate, "log": [], "settings": {"mock": True}}

    inversion_root = tmp_path / "inversion"
    inverted = runner.invert(source_root=source_root, output=inversion_root,
                             observation=observation, checkpoint=checkpoint,
                             device="cpu", direct_fia=fake_direct_fia)
    assert torch.equal(captured["target_z"], target)
    assert inverted["status"] == "COMPLETED"
    assert inverted["integrity"]["unchanged"] is True
    assert json_load(inversion_root / "registration.json")["fia_config"] == runner.CANONICAL_FIA_CONFIG
    assert runner._validate_candidate(inversion_root / "candidate.pt").shape == (2, 3, 224, 224)
    candidate_registration = json_load(inversion_root / "registration.json")
    candidate_path = inversion_root / "candidate.pt"
    candidate_path.write_bytes(candidate_path.read_bytes() + b"changed")
    with pytest.raises(runner.PipelineError, match="candidate hash mismatch"):
        runner._validate_candidate(candidate_path, candidate_registration["candidate_sha256"])
    with pytest.raises(FileExistsError, match="preexisting inversion output"):
        runner.invert(source_root=source_root, output=inversion_root,
                      observation=observation, checkpoint=checkpoint,
                      device="cpu", direct_fia=fake_direct_fia)


def json_load(path):
    return __import__("json").loads(path.read_text(encoding="utf-8"))


def test_source_validator_rejects_wrong_dimensions_nonfinite_and_mismatched_abs():
    original_count, original_dim = runner.SOURCE_COUNT, runner.FEATURE_DIM
    try:
        runner.SOURCE_COUNT, runner.FEATURE_DIM = 16, 25088
        valid = torch.zeros(16, 25088)
        with pytest.raises(runner.PipelineError, match="FP32"):
            runner._validate_source_tensors(torch.zeros(16, 25087), valid)
        with pytest.raises(runner.PipelineError, match="nonnegative"):
            runner._validate_source_tensors(valid, torch.full_like(valid, -1.0))
        bad = valid.clone()
        bad[0, 0] = float("nan")
        with pytest.raises(runner.PipelineError, match="finite"):
            runner._validate_source_tensors(bad, bad.abs())
        mismatched = valid.clone()
        mismatched[0, 0] = 1.0
        with pytest.raises(runner.PipelineError, match="abs.*exactly"):
            runner._validate_source_tensors(valid, mismatched)
    finally:
        runner.SOURCE_COUNT, runner.FEATURE_DIM = original_count, original_dim


def _profile_fixture(*, exact_feasible: bool) -> dict:
    per_tensor = {}
    for name in FC_NAMES:
        per_tensor[name] = {
            "name": name,
            "numel": 4,
            "fixed_s": 0.5,
            "code_mismatch_count": 0 if exact_feasible else 1,
            "code_mismatch_fraction": 0.0 if exact_feasible else 0.25,
            "bounded_cell_violation_count": 0 if exact_feasible else 1,
            "bounded_cell_violation_fraction": 0.0 if exact_feasible else 0.25,
            "mean_violation_in_s": 0.0 if exact_feasible else 0.125,
            "max_violation_in_s": 0.0 if exact_feasible else 0.25,
            "cell_penalty": 0.0 if exact_feasible else 0.125,
            "candidate_scale": 0.5,
        }
    return {
        "per_tensor": per_tensor,
        "overall_numel": 24,
        "overall_code_mismatch_count": 0 if exact_feasible else 6,
        "overall_code_mismatch_fraction": 0.0 if exact_feasible else 0.25,
        "overall_bounded_cell_violation_count": 0 if exact_feasible else 6,
        "overall_bounded_cell_violation_fraction": 0.0 if exact_feasible else 0.25,
        "cell_loss_equal_tensor_mean": 0.0 if exact_feasible else 0.125,
        "exact_feasible": exact_feasible,
        "feature_norm_mean": 1.0,
        "feature_norm_min": 1.0,
        "feature_norm_max": 1.0,
    }


def _preflight_fixture() -> dict:
    return {
        "feature_grad_enabled_equals_no_grad": True,
        "terminal_deltas_torch_equal": True,
        "adjoint_cell_loss": 0.0,
        "per_tensor": {
            name: {
                "torch_equal": True,
                "max_abs_difference": 0.0,
                "max_difference_in_s": 0.0,
                "fixed_s_code_mismatch_count": 0,
            }
            for name in FC_NAMES
        },
    }


def _prepare_refine_fixture(monkeypatch, tmp_path):
    observation, checkpoint = _configure_small_public_fixture(
        monkeypatch, tmp_path, source_count=16, feature_dim=25088)
    source_root, _, _, _, target = _write_small_source_artifact(monkeypatch, tmp_path)
    _, loaded_source, loaded_target = runner._load_sources(source_root)
    assert loaded_source.shape == (16, 25088)
    assert loaded_target.shape == (16, 25088)
    candidate = torch.full((16, 3, 224, 224), 0.25, dtype=torch.float32)

    def fake_direct_fia(**kwargs):
        return {"rec": candidate, "log": [], "settings": {"mock": True}}

    candidate_root = tmp_path / "inversion"
    runner.invert(source_root=source_root, output=candidate_root,
                  observation=observation, checkpoint=checkpoint, device="cpu",
                  direct_fia=fake_direct_fia)
    return observation, checkpoint, source_root, candidate_root, target, candidate


@pytest.mark.parametrize("final_feasible", [True, False])
def test_real_refine_orchestration_persists_ordered_evidence_and_feasibility(
        tmp_path, monkeypatch, final_feasible):
    observation, checkpoint, source_root, candidate_root, target, candidate = \
        _prepare_refine_fixture(monkeypatch, tmp_path)
    events = []
    original_configure_runtime = runner.configure_runtime

    def configured(*args, **kwargs):
        events.append("runtime")
        return original_configure_runtime(*args, **kwargs)

    monkeypatch.setattr(runner, "configure_runtime", configured)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(runner, "_to_device", lambda value, device: value)
    tables = {name: object() for name in FC_NAMES}
    monkeypatch.setattr(runner, "load_public_cell_tables",
                        lambda path: (object(), tables))
    monkeypatch.setattr(runner, "load_model",
                        lambda path, device: events.append("model") or object())

    def fake_labels(model, current_candidate, current_tables, device):
        events.append("labels")
        assert torch.equal(current_candidate, candidate)
        assert current_tables is tables
        return {"status": "COMPLETED", "rule": "coverage-based H1"}, \
            torch.arange(16, dtype=torch.int64)

    monkeypatch.setattr(runner, "build_refinement_labels", fake_labels)
    monkeypatch.setattr(runner, "_device_tables",
                        lambda current_tables, device: events.append("device_tables") or tables)
    monkeypatch.setattr(runner, "compare_replay_paths",
                        lambda *args: events.append("preflight") or _preflight_fixture())
    profile_calls = []

    def fake_profile(*args):
        profile_calls.append(True)
        events.append(f"profile_{len(profile_calls)}")
        return _profile_fixture(exact_feasible=True if len(profile_calls) == 1
                                else final_feasible)

    monkeypatch.setattr(runner, "canonical_profile", fake_profile)
    output = tmp_path / "refinement"

    def fake_refiner(model, arm, current_candidate, current_target, labels,
                     cell_weight, current_tables, device_tables, arm_output, device, clock):
        events.append("refiner")
        assert arm == runner.REFINE_ARM_NAME
        assert torch.equal(current_candidate, candidate)
        assert torch.equal(current_target, target)
        assert (arm_output / "preflight.json").is_file()
        assert (arm_output / "start_profile.json").is_file()
        folder = arm_output / arm
        folder.mkdir()
        final_path = folder / "candidate.pt"
        torch.save({"pixels": current_candidate}, final_path)
        return {"status": "COMPLETED", "arm": arm,
                "optimized_label_hypothesis": "H1",
                "candidate_sha256": runner.file_sha256(final_path)}

    result = runner.refine(source_root=source_root, candidate_root=candidate_root,
                           output=output, observation=observation, checkpoint=checkpoint,
                           device_name="cuda", image_refiner=fake_refiner)
    assert events == ["runtime", "model", "labels", "device_tables", "preflight",
                      "profile_1", "refiner", "profile_2"]
    assert result["status"] == "COMPLETED"
    assert result["method"] == runner.METHOD
    assert result["feasibility"] == {
        "start_exact_feasible": True,
        "final_exact_feasible": final_feasible,
        "software_completion": "COMPLETED",
    }
    assert result["integrity"]["unchanged"] is True
    assert set(result["start_profile"]["per_tensor"]) == set(FC_NAMES)
    assert set(result["final_profile"]["per_tensor"]) == set(FC_NAMES)
    assert set(result["final_profile"]["scale_diagnostics"]) == set(FC_NAMES)
    assert set(result["final_profile"]["feasibility"]) == {
        "exact_code_and_bounded_cell", "overall_code_mismatch_count",
        "overall_bounded_cell_violation_count",
    }
    for name in FC_NAMES:
        item = result["final_profile"]["per_tensor"][name]
        assert {"code_mismatch_count", "code_mismatch_fraction",
                "bounded_cell_violation_count", "bounded_cell_violation_fraction",
                "fixed_s", "candidate_scale"}.issubset(item)
    registration = json.loads((output / "registration.json").read_text(encoding="utf-8"))
    assert registration["status"] == "COMPLETED"
    assert registration["arm"] == "B_H1_combination_ica"
    assert registration["method"] == "INT4_combination_ica"
    assert registration["label_hypothesis"]["name"] == "H1"
    assert registration["label_hypothesis"]["frozen_before_optimization"] is True
    assert (output / "preflight.json").is_file()
    assert (output / "start_profile.json").is_file()
    assert (output / "final_profile.json").is_file()
    assert (output / "result.json").is_file()


def test_refine_failure_preserves_preflight_and_start_evidence(tmp_path, monkeypatch):
    observation, checkpoint, source_root, candidate_root, _, _ = \
        _prepare_refine_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(runner, "_to_device", lambda value, device: value)
    tables = {name: object() for name in FC_NAMES}
    monkeypatch.setattr(runner, "load_public_cell_tables",
                        lambda path: (object(), tables))
    monkeypatch.setattr(runner, "load_model", lambda path, device: object())
    monkeypatch.setattr(runner, "build_refinement_labels",
                        lambda *args: ({"status": "COMPLETED"}, torch.arange(16)))
    monkeypatch.setattr(runner, "_device_tables", lambda *args: tables)
    monkeypatch.setattr(runner, "compare_replay_paths", lambda *args: _preflight_fixture())
    monkeypatch.setattr(runner, "canonical_profile",
                        lambda *args: _profile_fixture(exact_feasible=False))

    def failing_refiner(*args):
        output = args[8]
        assert (output / "preflight.json").is_file()
        assert (output / "start_profile.json").is_file()
        raise RuntimeError("mock downstream failure")

    output = tmp_path / "failed_refinement"
    with pytest.raises(RuntimeError, match="mock downstream failure"):
        runner.refine(source_root=source_root, candidate_root=candidate_root,
                      output=output, observation=observation, checkpoint=checkpoint,
                      device_name="cuda", image_refiner=failing_refiner)
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "FAILED"
    assert failure["preflight_persisted"] is True
    assert failure["start_profile_persisted"] is True
    assert failure["final_profile_persisted"] is False
    assert not (output / "result.json").exists()
    registration = json.loads((output / "registration.json").read_text(encoding="utf-8"))
    assert registration["status"] == "REGISTERED"


def test_integrity_mismatch_is_saved_and_blocks_success(tmp_path, monkeypatch):
    before = {"public_input_hashes": {"input": "a"}, "artifact_hashes": {},
              "protected_source_hashes": {"source": "a"},
              "shared_source_hashes": {"shared": "a"}}
    after = {**before, "public_input_hashes": {"input": "changed"}}
    monkeypatch.setattr(runner, "_stage_integrity_after", lambda **kwargs: after)
    with pytest.raises(runner.PipelineError, match="integrity changed"):
        runner._persist_and_require_integrity(
            tmp_path, before, observation=tmp_path, checkpoint=tmp_path / "checkpoint")
    record = json.loads((tmp_path / "integrity.json").read_text(encoding="utf-8"))
    assert record["unchanged"] is False
    assert record["after"] == after


@pytest.mark.parametrize("changed_field", [
    "artifact_hashes", "protected_source_hashes", "shared_source_hashes",
])
def test_consumed_artifact_and_source_integrity_mismatches_block_success(
        tmp_path, monkeypatch, changed_field):
    before = {"public_input_hashes": {"input": "a"}, "artifact_hashes": {"artifact": "a"},
              "protected_source_hashes": {"source": "a"},
              "shared_source_hashes": {"shared": "a"}}
    after = {key: dict(value) for key, value in before.items()}
    only = next(iter(after[changed_field]))
    after[changed_field][only] = "changed"
    monkeypatch.setattr(runner, "_stage_integrity_after", lambda **kwargs: after)
    with pytest.raises(runner.PipelineError, match="integrity changed"):
        runner._persist_and_require_integrity(
            tmp_path, before, observation=tmp_path, checkpoint=tmp_path / "checkpoint")
    assert json.loads((tmp_path / "integrity.json").read_text(encoding="utf-8"))["unchanged"] is False


def test_registered_separator_and_input_bindings_reject_mutations(tmp_path, monkeypatch):
    observation, checkpoint = _configure_small_public_fixture(monkeypatch, tmp_path)
    source_root, _, registration, _, _ = _write_small_source_artifact(monkeypatch, tmp_path)
    with pytest.raises(runner.PipelineError, match="registered separator configuration"):
        runner._canonical_config(runner.CombinationICAConfig(refinement_iterations=2))
    with pytest.raises(runner.PipelineError, match="source registration provenance"):
        registration_path = source_root / "registration.json"
        broken = dict(registration)
        broken["config"] = {"mutated": True}
        registration_path.write_text(json.dumps(broken), encoding="utf-8")
        runner._load_sources(source_root)

    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    packet = observation / "packet.bin"
    packet.write_bytes(packet.read_bytes() + b"changed")
    with pytest.raises(runner.PipelineError, match="public input hash mismatch"):
        runner._validate_public_binding(registration, observation, checkpoint)

    packet.write_bytes(b"synthetic-public-packet")
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(runner.PipelineError, match="public checkpoint hash changed"):
        runner._validate_public_binding(registration, observation, checkpoint)


def test_incomplete_source_result_and_source_file_hash_are_rejected(tmp_path, monkeypatch):
    _configure_small_public_fixture(monkeypatch, tmp_path)
    source_root, _, registration, _, _ = _write_small_source_artifact(monkeypatch, tmp_path)
    result_path = source_root / "result.json"
    complete_result = json.loads(result_path.read_text(encoding="utf-8"))
    incomplete_result = dict(complete_result)
    del incomplete_result["target_z_exact_abs_S_hat"]
    result_path.write_text(json.dumps(incomplete_result), encoding="utf-8")
    with pytest.raises(runner.PipelineError, match="source tensor hash mismatch"):
        runner._load_sources(source_root)

    result_path.write_text(json.dumps(complete_result), encoding="utf-8")
    sources_path = source_root / "sources.pt"
    sources_path.write_bytes(sources_path.read_bytes() + b"changed")
    with pytest.raises(runner.PipelineError, match="source artifact file hash mismatch"):
        runner._load_sources(source_root)
