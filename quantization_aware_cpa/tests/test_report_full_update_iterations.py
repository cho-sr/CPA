"""Completed-artifact/provenance gates; synthetic CPU rendering only."""
from copy import deepcopy
import json

import pytest
import torch

from qacpa import report_full_update_iterations as reporter


def metadata():
    manifest = {"contract": {"image_shape": [16, 3, 224, 224]},
                "contract_sha256": "contract", "checkpoint_sha256": "checkpoint"}
    arms = {}
    for key, (folder, mode, budget) in reporter.ARMS.items():
        design = {"attack": {"iterations": budget, "seed": 42, "image_lr": 0.1}, "mode": mode,
                  "probe_only": False, "method": "direct_image_optimization_without_CPA",
                  "implementation_sha256": {"full_update_pipeline.py": "same implementation"},
                  "runtime": {"device": "cuda", "deterministic_algorithms": True},
                  "observation_manifest_sha256": "manifest", "contract_sha256": "contract",
                  "checkpoint_sha256": "checkpoint"}
        logs = [{"iteration": iteration, "loss": 100.0 / iteration + (mode == "center"),
                 "observation_loss": 99.0 / iteration, "tv": 0.01, "image_gradient_norm": 1.0,
                 "elapsed_sec": iteration * (1.0 if budget == 3 else 2.0)}
                for iteration in range(1, budget + 1)]
        arms[key] = {"design": design, "logs": logs}
    return arms, manifest


@pytest.mark.parametrize("change", ["seed", "implementation", "runtime", "checkpoint", "manifest", "prefix"])
def test_rejects_provenance_or_prefix_mismatch(change):
    arms, manifest = metadata()
    target = arms["cell_100"]
    if change == "seed":
        target["design"]["attack"]["seed"] = 43
    elif change == "implementation":
        target["design"]["implementation_sha256"]["full_update_pipeline.py"] = "changed"
    elif change == "runtime":
        target["design"]["runtime"]["deterministic_algorithms"] = False
    elif change == "checkpoint":
        target["design"]["checkpoint_sha256"] = "other checkpoint"
    elif change == "manifest":
        target["design"]["observation_manifest_sha256"] = "other observation"
    else:
        target["logs"][1]["loss"] += 1e-7
    with pytest.raises(reporter.ReportError):
        reporter._check_provenance(arms, manifest, "manifest")


def test_only_iterations_and_elapsed_time_can_change_from_pilot():
    arms, manifest = metadata()
    result = reporter._check_provenance(arms, manifest, "manifest")
    assert result["first_three_numeric_logs_exact"] is True
    assert result["only_budget_change_from_pilot"] == {"iterations": [3, 100]}
    assert result["common_attack_settings"] == {"seed": 42, "image_lr": 0.1}


def test_existing_report_is_not_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "comparison.json"
    marker.write_text("preserve previous report", encoding="utf-8")
    with pytest.raises(FileExistsError):
        reporter.build_report(base=tmp_path / "missing_base", truth=tmp_path / "private/truth.pt", output=output)
    assert marker.read_text(encoding="utf-8") == "preserve previous report"


def test_incomplete_runs_are_rejected_before_any_tensor_load(tmp_path, monkeypatch):
    def no_load(*args, **kwargs):
        pytest.fail("Incomplete-run reporter read a candidate or private truth tensor")
    monkeypatch.setattr(torch, "load", no_load)
    with pytest.raises(reporter.ReportError, match="incomplete"):
        reporter.build_report(base=tmp_path / "incomplete", truth=tmp_path / "private/truth.pt",
                              output=tmp_path / "new_report")
    assert not (tmp_path / "new_report").exists()


def test_candidate_must_be_bound_to_the_same_observation(tmp_path):
    folder = tmp_path / "wrong_candidate"
    folder.mkdir()
    torch.save({"pixels": torch.zeros(16, 3, 224, 224), "labels": torch.arange(16),
                "observation_manifest_sha256": "wrong manifest"}, folder / "candidate.pt")
    with pytest.raises(reporter.ReportError, match="different observation"):
        reporter._verify_evaluations({"cell_100": {"path": folder}}, tmp_path / "unopened_truth.pt", "expected")


def test_evaluation_revalidation_allows_only_float_roundoff():
    expected = {"per_image": [{"truth_index": 1, "label_correct": True, "mse": .125}], "mean_psnr_db": 9.0}
    near = deepcopy(expected)
    near["per_image"][0]["mse"] += 1e-15
    assert reporter._evaluation_matches(expected, near)
    for field, replacement in (("truth_index", 2), ("label_correct", 1), ("mse", .126)):
        changed = deepcopy(expected)
        changed["per_image"][0][field] = replacement
        assert not reporter._evaluation_matches(expected, changed)


def test_optional_lpips_uses_existing_pairs_and_minus_one_to_one_inputs(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        truth_pixels = torch.linspace(0, 1, 16).view(16, 1, 1, 1).expand(16, 3, 224, 224).clone()
        mean = torch.tensor((.485, .456, .406))[None, :, None, None]
        std = torch.tensor((.229, .224, .225))[None, :, None, None]
        truth_path = tmp_path / "truth.pt"
        torch.save({"x": (truth_pixels - mean) / std, "labels": torch.arange(16)}, truth_path)
        arms = {}
        for key in reporter.ARMS:
            directory = tmp_path / key
            directory.mkdir()
            torch.save({"pixels": truth_pixels.flip(0), "labels": torch.arange(16).flip(0)}, directory / "candidate.pt")
            arms[key] = {"path": directory, "evaluation": {"per_image": [
                {"candidate_index": 15 - i, "truth_index": i} for i in reversed(range(16))]}}
        calls, resets = [], []

        class MockMetric:
            def reset(self):
                resets.append(True)

            def __call__(self, candidate, reference):
                assert tuple(candidate.shape) == tuple(reference.shape) == (1, 3, 224, 224)
                assert candidate.device.type == reference.device.type == "cpu"
                assert candidate.dtype == reference.dtype == torch.float32
                assert not torch.is_grad_enabled()
                calls.append((candidate.clone(), reference.clone()))
                return (candidate - reference).square().mean()

        monkeypatch.setattr(reporter, "_make_lpips_metric", lambda: (MockMetric(), {"test_fixture": True}))
        supplement = reporter._lpips_supplement(arms, truth_path)
        assert len(calls) == len(resets) == 64
        for index, (candidate, reference) in enumerate(calls):
            expected = torch.full_like(candidate, float(truth_pixels[index % 16, 0, 0, 0] * 2 - 1))
            torch.testing.assert_close(candidate, expected, atol=2e-7, rtol=0)
            torch.testing.assert_close(reference, expected, atol=2e-7, rtol=0)
        assert supplement["threshold"] == "none_registered_or_applied"
        for values in supplement["arms"].values():
            assert values["n_images"] == 16
            assert [row["truth_index"] for row in values["per_image"]] == list(range(16))
            assert [row["candidate_index"] for row in values["per_image"]] == list(reversed(range(16)))
            assert values["mean_lpips"] < 1e-12
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("corrupt", [False, True], ids=["missing_backbone", "wrong_backbone_hash"])
def test_lpips_preflight_rejects_missing_or_changed_local_weights(tmp_path, monkeypatch, corrupt):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    if corrupt:
        checkpoint = tmp_path / "checkpoints/vgg16-397923af.pth"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"not the audited VGG16 weights")
    with pytest.raises(reporter.ReportError, match="missing|SHA256 mismatch"):
        reporter._verified_lpips_weights()


def test_real_cached_vgg_lpips_small_cpu_inference():
    from pathlib import Path
    if not (Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth").is_file():
        pytest.skip("Optional local-weight package smoke requires a preexisting VGG cache; no downloads")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        metric, metadata = reporter._make_lpips_metric()
        assert metadata["net_type"] == "vgg" and metadata["device"] == "cpu"
        assert metadata["downloads_allowed"] is False
        assert metadata["weights"]["backbone"]["sha256"] == reporter.LPIPS_BACKBONE_SHA256
        assert metadata["weights"]["calibration"]["sha256"] == reporter.LPIPS_CALIBRATION_SHA256
        assert all(parameter.device.type == "cpu" for parameter in metric.parameters())
        with torch.no_grad():
            image = torch.linspace(-1, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
            same = metric(image, image)
            metric.reset()
            changed = metric(image, image.flip(-1))
        assert torch.isfinite(same) and float(same) == 0
        assert torch.isfinite(changed) and float(changed) > 0
    finally:
        torch.set_num_threads(previous_threads)


def test_synthetic_completed_artifacts_generate_bound_json_markdown_and_pngs(tmp_path):
    from qacpa.full_update_evaluation import evaluate_candidate
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        arms, manifest = metadata()
        base = tmp_path / "batch00"
        observation = base / "observation"
        observation.mkdir(parents=True)
        manifest_path = observation / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_hash = reporter.digest(manifest_path)
        truth_path = tmp_path / "private/evaluation_truth.pt"
        truth_path.parent.mkdir()
        truth_pixels = (torch.arange(16, dtype=torch.float32) / 32 + 0.1).view(16, 1, 1, 1).expand(16, 3, 224, 224).clone()
        mean = torch.tensor((.485, .456, .406))[None, :, None, None]
        std = torch.tensor((.229, .224, .225))[None, :, None, None]
        torch.save({"x": (truth_pixels - mean) / std, "labels": torch.arange(16)}, truth_path)
        pixels = (truth_pixels + .015).flip(0)
        evaluation = None
        for key, (folder, mode, budget) in reporter.ARMS.items():
            directory = base / folder
            directory.mkdir()
            arm = deepcopy(arms[key])
            arm["design"]["observation_manifest_sha256"] = manifest_hash
            candidate = directory / "candidate.pt"
            torch.save({"pixels": pixels, "labels": torch.arange(16).flip(0),
                        "observation_manifest_sha256": manifest_hash}, candidate)
            if evaluation is None:
                evaluation = evaluate_candidate(candidate, truth_path)
            fit = {name: 0.0 for name in reporter.HARD_FIT_KEYS}
            fit.update(tolerance_steps=1e-6, closure_fit_within_tolerance=False,
                       zero_marginal_probability_count=0, exact_joint_rng_match="not_checked")
            result = {"status": "candidate_generated", "iterations": budget, "mode": mode,
                      "hard_label_observation_fit": fit, "optimization_time_sec": budget,
                      "wall_time_sec": budget + .5, "peak_cuda_allocated_bytes": 0}
            for name, value in (("registered_design.json", arm["design"]), ("result.json", result),
                                ("optimization_log.json", arm["logs"]), ("evaluation.json", evaluation)):
                (directory / name).write_text(json.dumps(value), encoding="utf-8")
        output = tmp_path / "report"
        report = reporter.build_report(base=base, truth=truth_path, output=output, image_grid=True)
        assert report["truth_sha256"] == reporter.digest(truth_path)
        assert report["existing_evaluations_revalidated"] is True
        assert report["comparison_checks"]["first_three_numeric_logs_exact"] is True
        for key, arm in report["arms"].items():
            assert arm["artifact_sha256"]["candidate.pt"] == reporter.digest(base / reporter.ARMS[key][0] / "candidate.pt")
            assert arm["final_metrics"]["n_images"] == 16
        assert set(path.name for path in output.iterdir()) == {
            "comparison.json", "comparison.md", "convergence.png", "final_psnr.png", "all16_by_truth.png"}
        assert (output / "convergence.png").read_bytes().startswith(b"\x89PNG")
        assert (output / "final_psnr.png").read_bytes().startswith(b"\x89PNG")
        assert (output / "all16_by_truth.png").read_bytes().startswith(b"\x89PNG")
        assert json.loads((output / "comparison.json").read_text()) == report
        assert all(value == 0 for delta in report["deltas_100_minus_3"].values() for value in delta.values())
    finally:
        torch.set_num_threads(previous_threads)
