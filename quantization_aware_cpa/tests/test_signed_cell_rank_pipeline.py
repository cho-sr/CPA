"""CPU/mock integration and provenance gates for the signed cell-rank arm."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
import torch

from qacpa import signed_cell_rank_evaluation as evaluation
from qacpa import signed_cell_rank_runner as runner
from qacpa import signed_cell_rank as core
from qacpa.signed_baseline import ATTACK_LAYER
from qacpa.signed_baseline_observation import CHECKPOINT_SHA
from quantized_update_cpa import cell_utils


def _manifest(checkpoint=CHECKPOINT_SHA):
    return {"checkpoint_sha256": checkpoint, "layers": {}}


def test_preprocess_and_attack_use_final_v_and_original_adapters(monkeypatch, tmp_path):
    public = tmp_path / "public_observation"
    public.mkdir()
    (public / "manifest.json").write_text(json.dumps(_manifest()) + "\n", encoding="utf-8")
    source = torch.linspace(-3., 3., 256, dtype=torch.float32).reshape(16, 16)
    q, s = cell_utils.encode(source)
    entries = {ATTACK_LAYER: (q, s)}
    monkeypatch.setattr(runner, "read_bundle", lambda _folder: (_manifest(), entries))
    monkeypatch.setattr(runner, "FC_SHAPES", {ATTACK_LAYER: (16, 16)})
    monkeypatch.setattr(evaluation, "read_bundle", lambda _folder: (_manifest(), entries))
    monkeypatch.setattr(evaluation, "FC_SHAPES", {ATTACK_LAYER: (16, 16)})

    prep = tmp_path / "preprocess_ap0"
    prep_result = runner.preprocess(
        public_observation=public, output=prep, ap_iterations=0,
        projection_device="cpu", chunk_rows=4)
    assert prep_result["status"] == "COMPLETED"
    assert prep_result["full_experiment_status"] == "NOT_RUN"
    final_v = torch.load(prep / "V.pt", map_location="cpu", weights_only=True)
    assert torch.equal(final_v, cell_utils.dequantize(q, s))
    assert core.fixed_cell_metrics(final_v, q, s)["exact_code_match"]

    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"mock-checkpoint")
    original_digest = runner.digest
    monkeypatch.setattr(runner, "digest", lambda path: runner.CHECKPOINT_SHA
                        if Path(path).resolve() == checkpoint.resolve() else original_digest(path))
    monkeypatch.setattr(runner, "verify_checkpoint", lambda _path: None)
    monkeypatch.setattr(runner, "configure", lambda: None)
    monkeypatch.setattr(runner, "telemetry", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(runner, "runtime", lambda: {
        "cudnn_benchmark": True, "cudnn_deterministic": True,
        "deterministic_algorithms": True, "deterministic_warn_only": True,
        "matmul_tf32": False, "cudnn_tf32": False,
        "cublas_workspace_config": ":4096:8", "threads": 1,
    })
    source_hashes = {"mock_upstream.py": "a" * 64}
    monkeypatch.setattr(runner.upstream_naive, "source_hashes", lambda: source_hashes)
    cpa_calls, fia_calls = [], []

    def fake_cpa(**kwargs):
        cpa_calls.append(kwargs)
        assert torch.equal(kwargs["Y_hat"], final_v)
        assert kwargs["n_components"] == 16
        assert kwargs["config"] == runner.CPA_CONFIG
        assert kwargs["seed"] == runner.SEED
        runner.upstream_naive._experiment(
            kwargs["output_dir"], "cpa", kwargs["Y_hat"],
            deepcopy(evaluation.CPA_EXECUTED_SETTINGS))
        return {"S_hat": torch.ones((16, 16), dtype=torch.float32),
                "log": [{"loss": 1.0}],
                "settings": deepcopy(evaluation.CPA_EXECUTED_SETTINGS),
                "source_hashes": source_hashes}

    def fake_fia(**kwargs):
        fia_calls.append(kwargs)
        assert torch.equal(kwargs["target_z"], torch.ones((16, 16)))
        assert kwargs["config"] == runner.FIA_CONFIG
        assert kwargs["seed"] == runner.SEED
        runner.upstream_naive._experiment(
            kwargs["output_dir"], "fia", kwargs["target_z"],
            deepcopy(evaluation.FIA_EXECUTED_SETTINGS))
        (Path(kwargs["output_dir"]) / "fia_initial.json").write_text(json.dumps({
            "tensor_sha256": "a" * 64,
            "shape": [16, 3, 224, 224],
            "runtime": runner.runtime(),
            "seed": runner.SEED,
        }) + "\n", encoding="utf-8")
        return {"rec": torch.zeros((16, 3, 224, 224), dtype=torch.float32),
                "log": [], "settings": deepcopy(evaluation.FIA_EXECUTED_SETTINGS),
                "source_hashes": source_hashes}

    monkeypatch.setattr(runner.upstream_naive, "run_upstream_cpa", fake_cpa)
    monkeypatch.setattr(runner.upstream_naive, "run_upstream_fia", fake_fia)
    arm = tmp_path / "cell_rank_ap0"
    result = runner.attack(public_observation=public, preprocessed=prep,
                           checkpoint=checkpoint, output=arm, expected_ap_iterations=0)
    assert result["arm"] == "INT4_cell_rank"
    assert result["full_experiment_status"] == "NOT_RUN"
    assert result["attack_status"] == "COMPLETED"
    assert result["evaluation_status"] == "NOT_RUN"
    assert result["labels_used"] is False and result["private_inputs_used"] is False
    registered = json.loads((arm / "registered_arm.json").read_text(encoding="utf-8"))
    assert registered["input_conversion"] == "raw_signed_dequantized_V_final_cell_projected_from_public_q_s"
    assert registered["source_adapter"] == "abs(S_hat); estimated ordering; no oracle correction"
    assert registered["input_tensor_sha256"] == runner.upstream_naive._tensor_hash(final_v)
    cpa_manifest = json.loads((arm / "cpa" / "cpa_manifest.json").read_text(encoding="utf-8"))
    assert cpa_manifest["fingerprint"]["input_sha256"] == registered["input_tensor_sha256"]
    saved = torch.load(arm / "sources.pt", map_location="cpu", weights_only=True)
    assert torch.equal(saved["target_z"], saved["S_hat"].abs())
    assert len(cpa_calls) == 1 and len(fia_calls) == 1


def test_offline_evaluator_rejects_incomplete_arm_before_private_torch_load(monkeypatch, tmp_path):
    baseline_public = tmp_path / "baseline_public"
    cell_public = tmp_path / "cell_public"
    baseline_private = tmp_path / "baseline_private"
    cell_private = tmp_path / "cell_private"
    for path in (baseline_public, cell_public, baseline_private, cell_private):
        path.mkdir()
    monkeypatch.setattr(evaluation, "BASELINE_PUBLIC_ROOT", baseline_public)
    monkeypatch.setattr(evaluation, "CELL_PUBLIC_ROOT", cell_public)
    monkeypatch.setattr(evaluation, "BASELINE_PRIVATE_ROOT", baseline_private)
    monkeypatch.setattr(evaluation, "CELL_PRIVATE_ROOT", cell_private)
    observation = baseline_public / "observation"
    observation.mkdir()
    manifest = _manifest()
    (observation / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    public_sha = core.file_sha256(observation / "manifest.json")
    private = baseline_private / "observation"
    private.mkdir()
    raw = private / "raw_updates.pt"
    truth = private / "evaluation_truth.pt"
    raw.write_bytes(b"raw-private-placeholder")
    truth.write_bytes(b"truth-private-placeholder")
    sender = {
        "status": "COMPLETED", "checkpoint_sha256": CHECKPOINT_SHA,
        "source_truth_sha256": evaluation.TRUTH_SHA,
        "public_manifest_sha256": public_sha,
        "raw_updates_sha256": core.file_sha256(raw),
        "evaluation_truth_sha256": core.file_sha256(truth),
    }
    (private / "sender_validation.json").write_text(json.dumps(sender) + "\n", encoding="utf-8")
    naive = baseline_public / "naive"
    cell_rank = cell_public / "cell_rank_ap0"
    for path in (naive, cell_rank):
        path.mkdir()
    (naive / "result.json").write_text(json.dumps({"status": "FAILED"}) + "\n", encoding="utf-8")
    (cell_rank / "result.json").write_text(json.dumps({"status": "FAILED"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(evaluation.upstream_naive, "source_hashes", lambda: {})
    monkeypatch.setattr(evaluation.torch, "load", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("private tensor was loaded before finalization")))
    with pytest.raises(evaluation.ProvenanceError, match="completed"):
        evaluation.validate_provenance(
            naive=naive, cell_rank=cell_rank, public_observation=observation,
            private_observation=private, output=cell_private / "evaluation_ap0")


def test_same_arm_and_overlapping_paths_are_rejected_without_loading():
    path = Path("/tmp/signed-cell-rank")
    with pytest.raises(evaluation.ProvenanceError):
        evaluation._validate_paths(
            naive=path, cell_rank=path, public_observation=path / "obs",
            private_observation=path / "private", output=path / "out")


def _mock_settings_and_runtime(monkeypatch):
    source_hashes = {"mock_upstream.py": "a" * 64}
    monkeypatch.setattr(runner.upstream_naive, "source_hashes", lambda: source_hashes)
    monkeypatch.setattr(runner, "verify_checkpoint", lambda _path: None)
    monkeypatch.setattr(runner, "configure", lambda: None)
    monkeypatch.setattr(runner, "telemetry", lambda *_args, **_kwargs: nullcontext())
    runtime = {
        "cudnn_benchmark": True, "cudnn_deterministic": True,
        "deterministic_algorithms": True, "deterministic_warn_only": True,
        "matmul_tf32": False, "cudnn_tf32": False,
        "cublas_workspace_config": ":4096:8", "threads": 1,
    }
    monkeypatch.setattr(runner, "runtime", lambda: runtime)
    return source_hashes, runtime


def _prepare_mock_comparison(monkeypatch, tmp_path, *, fail_fia=False, ap_iterations=0):
    baseline_public = tmp_path / "baseline_public"
    cell_public = tmp_path / "cell_public"
    baseline_private = tmp_path / "baseline_private"
    cell_private = tmp_path / "cell_private"
    for path in (baseline_public, cell_public, baseline_private, cell_private):
        path.mkdir()
    monkeypatch.setattr(evaluation, "BASELINE_PUBLIC_ROOT", baseline_public)
    monkeypatch.setattr(evaluation, "CELL_PUBLIC_ROOT", cell_public)
    monkeypatch.setattr(evaluation, "BASELINE_PRIVATE_ROOT", baseline_private)
    monkeypatch.setattr(evaluation, "CELL_PRIVATE_ROOT", cell_private)

    public = baseline_public / "observation"
    public.mkdir()
    manifest = _manifest()
    (public / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    source = torch.linspace(-3., 3., 256, dtype=torch.float32).reshape(16, 16)
    q, s = cell_utils.encode(source)
    entries = {ATTACK_LAYER: (q, s)}
    monkeypatch.setattr(runner, "read_bundle", lambda _folder: (_manifest(), entries))
    monkeypatch.setattr(runner, "FC_SHAPES", {ATTACK_LAYER: (16, 16)})
    monkeypatch.setattr(evaluation, "read_bundle", lambda _folder: (_manifest(), entries))
    monkeypatch.setattr(evaluation, "FC_SHAPES", {ATTACK_LAYER: (16, 16)})
    source_hashes, runtime = _mock_settings_and_runtime(monkeypatch)

    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"mock-checkpoint")
    original_digest = runner.digest
    monkeypatch.setattr(runner, "digest", lambda path: runner.CHECKPOINT_SHA
                        if Path(path).resolve() == checkpoint.resolve() else original_digest(path))

    def fake_cpa(**kwargs):
        runner.upstream_naive._experiment(
            kwargs["output_dir"], "cpa", kwargs["Y_hat"],
            deepcopy(evaluation.CPA_EXECUTED_SETTINGS))
        return {
            "S_hat": torch.ones((16, 16), dtype=torch.float32),
            "log": [{"loss": 1.0}],
            "settings": deepcopy(evaluation.CPA_EXECUTED_SETTINGS),
            "source_hashes": source_hashes,
        }

    def fake_fia(**kwargs):
        if fail_fia:
            raise RuntimeError("mock FIA failure")
        runner.upstream_naive._experiment(
            kwargs["output_dir"], "fia", kwargs["target_z"],
            deepcopy(evaluation.FIA_EXECUTED_SETTINGS))
        (Path(kwargs["output_dir"]) / "fia_initial.json").write_text(json.dumps({
            "tensor_sha256": "a" * 64, "shape": [16, 3, 224, 224],
            "runtime": runtime, "seed": runner.SEED,
        }) + "\n", encoding="utf-8")
        return {
            "rec": torch.zeros((16, 3, 224, 224), dtype=torch.float32),
            "log": [], "settings": deepcopy(evaluation.FIA_EXECUTED_SETTINGS),
            "source_hashes": source_hashes,
        }

    monkeypatch.setattr(runner.upstream_naive, "run_upstream_cpa", fake_cpa)
    monkeypatch.setattr(runner.upstream_naive, "run_upstream_fia", fake_fia)
    prep = cell_public / f"preprocess_ap{ap_iterations}"
    runner.preprocess(public_observation=public, output=prep, ap_iterations=ap_iterations,
                      projection_device="cpu", chunk_rows=4)
    cell_arm = cell_public / f"cell_rank_ap{ap_iterations}"
    try:
        runner.attack(public_observation=public, preprocessed=prep,
                      checkpoint=checkpoint, output=cell_arm,
                      expected_ap_iterations=ap_iterations)
    except RuntimeError:
        if not fail_fia:
            raise
        return {"public": public, "cell_arm": cell_arm, "cell_private": cell_private,
                "prep": prep, "checkpoint": checkpoint}

    # Keep the cloned legacy fixture's caches inside its own output directory.
    naive = baseline_public / "naive"
    shutil.copytree(cell_arm, naive)
    for kind in ("cpa", "fia"):
        manifest_path = naive / kind / f"{kind}_manifest.json"
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        cache = naive / kind / "mock_cache"
        cache.mkdir()
        manifest_value["cache_root"] = str(cache)
        manifest_path.write_text(json.dumps(manifest_value) + "\n", encoding="utf-8")
    registration_path = naive / "registered_arm.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    public_sha = evaluation.digest(public / "manifest.json")
    registration["input"] = {
        "arm": "INT4_naive", "public_manifest_sha256": public_sha,
        "checkpoint_sha256": runner.CHECKPOINT_SHA, "private_inputs_used": False,
    }
    registration["input_conversion"] = "identity_raw_Delta_or_raw_s_q; no sign/lr/T/B transformation"
    registration.pop("preprocess", None)
    registration_path.write_text(json.dumps(registration) + "\n", encoding="utf-8")
    result_path = naive / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["arm"] = "INT4_naive"
    result["input"] = registration["input"]
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")

    private = baseline_private / "observation"
    private.mkdir()
    raw = private / "raw_updates.pt"
    truth = private / "evaluation_truth.pt"
    true_g = torch.full((16, 16), 0.25, dtype=torch.float32)
    torch.save({"updates": {ATTACK_LAYER: true_g},
                "delta_convention": evaluation.RAW_DELTA_CONVENTION,
                "checkpoint_sha256": runner.CHECKPOINT_SHA}, raw)
    images = torch.zeros((16, 3, 224, 224), dtype=torch.float32)
    torch.save({"x": images, "z": torch.eye(16, dtype=torch.float32),
                "labels": torch.arange(16, dtype=torch.int64)}, truth)
    sender = {
        "status": "COMPLETED", "checkpoint_sha256": runner.CHECKPOINT_SHA,
        "source_truth_sha256": evaluation.TRUTH_SHA,
        "public_manifest_sha256": public_sha,
        "raw_updates_sha256": evaluation.digest(raw),
        "evaluation_truth_sha256": evaluation.digest(truth),
    }
    (private / "sender_validation.json").write_text(json.dumps(sender) + "\n", encoding="utf-8")
    return {"public": public, "naive": naive, "cell_arm": cell_arm,
            "private": private, "cell_private": cell_private, "prep": prep,
            "checkpoint": checkpoint,
            "source_hashes": source_hashes}


def _fake_image_evaluation(folder, truth_path, originals):
    candidate = torch.load(folder / "candidate.pt", map_location="cpu", weights_only=True)
    pixels = candidate["pixels"].float()
    per_image = []
    for index in range(16):
        mse = float((pixels[index] - originals[index]).square().mean())
        per_image.append({"candidate_index": index, "truth_index": index,
                          "mse": mse, "psnr_db": -10.0, "ssim": 0.5,
                          "lpips": 0.25, "perfect_pixel_match": False})
    return {"matching": "mock_pixel_MSE_Hungarian_once", "n_images": 16,
            "all_images_included": True, "per_image": per_image,
            "mean_mse": sum(x["mse"] for x in per_image) / 16,
            "mean_psnr_db": -10.0, "mean_ssim": 0.5, "mean_lpips": 0.25,
            "labels_consumed": False}


def test_successful_mock_evaluator_writes_all16_figures_and_status(monkeypatch, tmp_path):
    fixture = _prepare_mock_comparison(monkeypatch, tmp_path)
    monkeypatch.setattr(evaluation, "_image_evaluation", _fake_image_evaluation)
    output = fixture["cell_private"] / "evaluation_ap0"
    result = evaluation.evaluate(naive=fixture["naive"], cell_rank=fixture["cell_arm"],
                                 public_observation=fixture["public"],
                                 private_observation=fixture["private"], output=output)
    assert result["status"] == "COMPLETED"
    assert result["evaluation_status"] == "COMPLETED"
    assert result["plot_status"] == "COMPLETED"
    assert result["all_images_included"]
    assert result["offline_private_loaded"] is True
    assert result["offline_update_error"]["naive_sq"]["relative_l2_error_status"] == "COMPLETED"
    assert result["offline_update_error"]["cell_rank_V"]["selection_used"] is False
    assert (output / "all16_comparison.png").is_file()
    assert (output / "all16_compact.png").is_file()
    saved = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert saved["status"] == "COMPLETED"


def test_legacy_naive_without_result_checkpoint_reaches_offline_evaluation(monkeypatch, tmp_path):
    """The completed legacy naive schema uses its validated registration hash."""
    fixture = _prepare_mock_comparison(monkeypatch, tmp_path)
    result_path = fixture["naive"] / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.pop("checkpoint_sha256")
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    monkeypatch.setattr(evaluation, "_image_evaluation", _fake_image_evaluation)
    output = fixture["cell_private"] / "legacy_naive_compatibility"
    evaluated = evaluation.evaluate(
        naive=fixture["naive"], cell_rank=fixture["cell_arm"],
        public_observation=fixture["public"], private_observation=fixture["private"],
        output=output)
    assert evaluated["status"] == "COMPLETED"
    assert evaluated["evaluation_status"] == "COMPLETED"
    assert evaluated["offline_private_loaded"] is True
    assert evaluated["offline_update_error"]["naive_sq"]["status"] == "COMPLETED"


def test_cell_arm_without_result_checkpoint_rejected_before_private_load(monkeypatch, tmp_path):
    fixture = _prepare_mock_comparison(monkeypatch, tmp_path)
    result_path = fixture["cell_arm"] / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.pop("checkpoint_sha256")
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    monkeypatch.setattr(evaluation.torch, "load", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("private artifact loaded before checkpoint rejection")))
    with pytest.raises(evaluation.ProvenanceError, match="result checkpoint_sha256 is missing"):
        evaluation.evaluate(
            naive=fixture["naive"], cell_rank=fixture["cell_arm"],
            public_observation=fixture["public"], private_observation=fixture["private"],
            output=fixture["cell_private"] / "missing_cell_checkpoint")


@pytest.mark.parametrize("target,field,value,match", [
    ("naive", "registered", "b" * 64, "registered checkpoint/seed mismatch"),
    ("cell_arm", "registered", "b" * 64, "registered checkpoint/seed mismatch"),
    ("naive", "result", "not-a-sha256", "result checkpoint_sha256 must be exactly"),
    ("cell_arm", "result", "b" * 64, "registered checkpoint/seed mismatch"),
])
def test_checkpoint_tampering_rejected_before_private_load(monkeypatch, tmp_path,
                                                            target, field, value, match):
    fixture = _prepare_mock_comparison(monkeypatch, tmp_path)
    folder = fixture[target]
    filename = "registered_arm.json" if field == "registered" else "result.json"
    path = folder / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = "checkpoint_sha256"
    payload[key] = value
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(evaluation.torch, "load", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("private artifact loaded before checkpoint rejection")))
    with pytest.raises(evaluation.ProvenanceError, match=match):
        evaluation.evaluate(
            naive=fixture["naive"], cell_rank=fixture["cell_arm"],
            public_observation=fixture["public"], private_observation=fixture["private"],
            output=fixture["cell_private"] / f"bad_{target}_{field}")


def test_plot_failure_preserves_numeric_results_without_false_completion(monkeypatch, tmp_path):
    fixture = _prepare_mock_comparison(monkeypatch, tmp_path)
    monkeypatch.setattr(evaluation, "_image_evaluation", _fake_image_evaluation)
    import matplotlib.figure
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected savefig failure")))
    output = fixture["cell_private"] / "evaluation_plot_failure"
    result = evaluation.evaluate(naive=fixture["naive"], cell_rank=fixture["cell_arm"],
                                 public_observation=fixture["public"],
                                 private_observation=fixture["private"], output=output)
    assert result["status"] == "FAILED"
    assert result["evaluation_status"] == "FAILED"
    assert result["plot_status"] == "FAILED"
    assert result["numeric_results_status"] == "COMPLETED"
    saved = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert saved["status"] == "FAILED" and "savefig failure" in saved["plot_error"]
    assert "Numerical comparison JSON was committed" in (output / "report.md").read_text(encoding="utf-8")


def test_canonical_input_and_executed_learning_rate_tampering_are_rejected(monkeypatch, tmp_path):
    fixture = _prepare_mock_comparison(monkeypatch, tmp_path)
    manifest_path = fixture["cell_arm"] / "cpa" / "cpa_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fingerprint"]["input_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(evaluation.ProvenanceError, match="input fingerprint"):
        evaluation.validate_provenance(
            naive=fixture["naive"], cell_rank=fixture["cell_arm"],
            public_observation=fixture["public"], private_observation=fixture["private"],
            output=fixture["cell_private"] / "bad_hash")

    manifest["fingerprint"]["input_sha256"] = json.loads(
        (fixture["cell_arm"] / "registered_arm.json").read_text(encoding="utf-8"))["input_tensor_sha256"]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    cpa_result_path = fixture["cell_arm"] / "cpa_result.json"
    cpa_result = json.loads(cpa_result_path.read_text(encoding="utf-8"))
    cpa_result["settings"]["lr"] = 0.123
    cpa_result_path.write_text(json.dumps(cpa_result) + "\n", encoding="utf-8")
    with pytest.raises(evaluation.ProvenanceError, match="settings mismatch"):
        evaluation.validate_provenance(
            naive=fixture["naive"], cell_rank=fixture["cell_arm"],
            public_observation=fixture["public"], private_observation=fixture["private"],
            output=fixture["cell_private"] / "bad_lr")


def test_failed_attack_persists_stage_specific_status(monkeypatch, tmp_path):
    fixture = _prepare_mock_comparison(monkeypatch, tmp_path, fail_fia=True)
    result = json.loads((fixture["cell_arm"] / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAILED"
    assert result["attack_status"] == "FAILED"
    assert result["cpa_status"] == "COMPLETED"
    assert result["fia_status"] == "FAILED"
    assert result["evaluation_status"] == "NOT_RUN"


def test_declared_ap_mismatch_rejected_before_original_cpa(monkeypatch, tmp_path):
    fixture = _prepare_mock_comparison(monkeypatch, tmp_path, ap_iterations=0)
    calls = []
    original = runner.upstream_naive.run_upstream_cpa
    monkeypatch.setattr(runner.upstream_naive, "run_upstream_cpa",
                        lambda **kwargs: (calls.append(kwargs), original(**kwargs))[1])
    with pytest.raises(ValueError, match="declared arm"):
        runner.attack(public_observation=fixture["public"], preprocessed=fixture["prep"],
                      checkpoint=fixture["checkpoint"], output=fixture["cell_private"] / "bad_ap5",
                      expected_ap_iterations=5)
    assert calls == []


def test_preprocess_registration_result_metadata_mismatch_rejected(monkeypatch, tmp_path):
    fixture = _prepare_mock_comparison(monkeypatch, tmp_path, ap_iterations=0)
    path = fixture["prep"] / "registered_preprocess.json"
    registration = json.loads(path.read_text(encoding="utf-8"))
    registration["rank"] = 15
    path.write_text(json.dumps(registration) + "\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(runner.upstream_naive, "run_upstream_cpa",
                        lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="rank"):
        runner.attack(public_observation=fixture["public"], preprocessed=fixture["prep"],
                      checkpoint=fixture["checkpoint"], output=fixture["cell_private"] / "bad_metadata",
                      expected_ap_iterations=0)
    assert calls == []


@pytest.mark.parametrize(("field", "value"), [
    ("rank", 15), ("centered", True), ("rank_backend", "svd"),
    ("ap_iterations", 5),
])
def test_both_preprocess_metadata_and_declared_arm_mismatch_rejected_before_cpa(
        monkeypatch, tmp_path, field, value):
    fixture = _prepare_mock_comparison(monkeypatch, tmp_path, ap_iterations=0)
    for filename in ("registered_preprocess.json", "preprocessing_result.json"):
        path = fixture["prep"] / filename
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata[field] = value
        path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(runner.upstream_naive, "run_upstream_cpa",
                        lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError):
        runner.attack(public_observation=fixture["public"], preprocessed=fixture["prep"],
                      checkpoint=fixture["checkpoint"], output=fixture["cell_private"] / f"bad_{field}",
                      expected_ap_iterations=0)
    assert calls == []


def test_explicit_ap5_contract_is_accepted_on_cpu_fixture(monkeypatch, tmp_path):
    fixture = _prepare_mock_comparison(monkeypatch, tmp_path, ap_iterations=5)
    registration = json.loads((fixture["prep"] / "registered_preprocess.json").read_text(encoding="utf-8"))
    assert registration["ap_iterations"] == 5
    assert json.loads((fixture["cell_arm"] / "result.json").read_text(encoding="utf-8"))["status"] == "COMPLETED"


@pytest.mark.parametrize("tamper", ["sender_raw_hash", "cell_v_file_hash", "cell_v_tensor_hash"])
def test_true_g_binding_tamper_rejected_before_private_load(monkeypatch, tmp_path, tamper):
    fixture = _prepare_mock_comparison(monkeypatch, tmp_path)
    if tamper == "sender_raw_hash":
        path = fixture["private"] / "sender_validation.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["raw_updates_sha256"] = "0" * 64
    else:
        path = fixture["cell_arm"] / "registered_arm.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        key = "preprocessed_v_file_sha256" if tamper == "cell_v_file_hash" else "preprocessed_v_tensor_sha256"
        value["input"][key] = "0" * 64
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    monkeypatch.setattr(evaluation.torch, "load", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("private true-G loaded before binding validation")))
    with pytest.raises(evaluation.ProvenanceError):
        evaluation.evaluate(naive=fixture["naive"], cell_rank=fixture["cell_arm"],
                            public_observation=fixture["public"],
                            private_observation=fixture["private"],
                            output=fixture["cell_private"] / f"bad_{tamper}")
