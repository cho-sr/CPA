"""CPU protocol gates for direct observed-row FIA without source separation."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def runner():
    from qacpa import run_m9_rawrow_fia

    return run_m9_rawrow_fia


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def reject_source_recovery(monkeypatch, runner):
    from qacpa import stage1, stage2, uia, upstream_naive, victim

    def forbidden(*args, **kwargs):
        raise AssertionError("Raw-row FIA invoked CPA/debias/MP/LS-A/Stage2/training")

    for module, names in (
        (stage1, ("estimate_mixing_and_sources", "_least_squares_mixing", "_run_cpa_ica")),
        (stage2, ("solve_stage2_consistency", "oracle_pinv_initialization")),
        (uia, ("mp_prediction_from_quantized", "run_fc_only_fedavg_on_features", "run_fc_only_fedavg_checkpoints")),
        (upstream_naive, ("run_upstream_cpa",)),
        (runner, ("run_upstream_cpa", "estimate_stage1", "estimate_mixing_and_sources")),
        (torch.linalg, ("eigh", "eig", "svd", "pinv", "solve", "lstsq")),
    ):
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    for name in ("collect_victim_batch", "collect_uia_observation"):
        for module in (uia, victim):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)


def test_selection_is_seeded_exact_rows_without_global_rng_or_input_changes(
    runner, reject_source_recovery
):
    observation = torch.arange(-70., 70.).reshape(20, 7)
    before = observation.clone()
    rng_before = torch.random.get_rng_state().clone()
    signed, target, rows = runner.raw_row_targets(observation, batch_index=3, seed=41, n=16)
    expected_rows = torch.randperm(20, generator=torch.Generator().manual_seed(44))[:16]

    assert torch.equal(rows, expected_rows)
    assert len(set(rows.tolist())) == 16
    assert torch.equal(signed, observation[expected_rows])
    assert torch.equal(target, observation[expected_rows].abs())
    assert torch.equal(observation, before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert not torch.allclose(target.norm(dim=1), torch.ones(16))
    repeated = runner.raw_row_targets(observation, batch_index=3, seed=41, n=16)
    assert all(torch.equal(left, right) for left, right in zip((signed, target, rows), repeated))


def test_zero_rows_are_retained_and_do_not_get_replaced_by_priors(
    runner, reject_source_recovery
):
    observation = torch.tensor([[0., 0., 0.], [-100., 2., 3.], [4., -5., 6.], [0., 0., 0.]])
    signed, target, rows = runner.raw_row_targets(observation, batch_index=0, seed=0, n=4)

    assert sorted(rows.tolist()) == [0, 1, 2, 3]
    assert torch.equal(signed, observation[rows])
    assert torch.equal(target, observation[rows].abs())
    assert (target.abs().sum(dim=1) == 0).sum().item() == 2
    assert target.max().item() == 100., "Magnitude must not be clipped or normalized"


def test_rawrow_fia_finishes_before_truth_load_and_preserves_all_rows(
    runner, tmp_path, monkeypatch, reject_source_recovery
):
    from qacpa import run_m8_fedavg_naive as m8

    cfg = runner.read_json(runner.DEFAULT_CONFIG)
    m8_cfg = m8.read_json(m8.DEFAULT_CONFIG)
    assert cfg["fia"] == m8_cfg["fia"], "The reference FIA settings must be held fixed"
    source = tmp_path / "source"
    source.mkdir()
    (source / "quantized.pt").write_bytes(b"synthetic observation hash marker")
    output = tmp_path / "result"
    manifest = {"identity": {"synthetic": "raw-row-truth-boundary"}}
    # Preserve real matrix dimensions with a single stored scalar; no 411MB fixture.
    observation = torch.tensor(-.25).expand(4096, 25088)
    selected, target, row_ids = runner.raw_row_targets(
        observation, batch_index=0, seed=cfg.get("row_seed", cfg.get("seed", 0)), n=16
    )
    manifest["frozen_row_indices"] = {"0": row_ids.tolist()}
    source_recs = (torch.arange(16).float() / 15).view(16, 1, 1, 1).expand(16, 3, 224, 224).clone()
    truth = torch.zeros(16, 25088)
    truth[:, :16] = torch.eye(16)
    inputs = {
        "x": torch.zeros(16, 3, 224, 224), "Z_evaluation_only": truth,
        "metadata": [{"global_index": i, "batch_index": 0, "source_index": i,
                      "image_path": f"synthetic/image{i}.jpg"} for i in range(16)],
    }
    events = []
    actual_load = torch.load
    actual_evaluation = runner.evaluate_after_fia

    def load(path, *args, **kwargs):
        if Path(path) == source / "quantized.pt":
            events.append("observation")
            return {"Y_hat": observation}
        if Path(path) == source / "inputs.pt":
            assert events == ["observation", "fia"], "Ground truth was loaded before FIA finished"
            events.append("truth")
            return inputs
        return actual_load(path, *args, **kwargs)

    def fia(**kwargs):
        assert events == ["observation"]
        assert torch.equal(kwargs["target_z"], target)
        assert tuple(kwargs["target_z"].shape) == (16, 25088)
        assert "x_true" not in kwargs and "truth" not in kwargs
        assert kwargs["config"] == m8_cfg["fia"]
        assert kwargs["seed"] == m8_cfg["fia"]["seed"]
        events.append("fia")
        return {"rec": source_recs, "log": [], "wall_time_sec": 2.}

    def evaluate(estimated, reconstructed, originals):
        assert events == ["observation", "fia", "truth"]
        assert torch.equal(estimated, selected), "Match the signed raw rows only after FIA"
        assert torch.equal(reconstructed, source_recs)
        assert originals is truth
        events.append("matching")
        return actual_evaluation(estimated, reconstructed, originals)

    def metrics(**kwargs):
        assert events[-1] == "matching"
        assert len(kwargs["rec"]) == 16, "Every FIA output must be evaluated"
        events.append("metrics")
        return SimpleNamespace(lpips=[.8] * 16, psnr=[20.] * 16, ssim=[.2] * 16)

    monkeypatch.setattr(runner, "verify_batch_source", lambda *args: source)
    monkeypatch.setattr(runner.torch, "load", load)
    monkeypatch.setattr(runner, "run_upstream_fia", fia)
    monkeypatch.setattr(runner, "evaluate_after_fia", evaluate)
    monkeypatch.setattr(runner, "evaluate_images", metrics)
    result = runner.run_batch(cfg, manifest, 0, output, torch.device("cpu"))
    assert events == ["observation", "fia", "truth", "matching", "metrics"]
    assert result["n_images"] == 16 and result["oracle_fills"] == 0
    assert result["forbidden_operations_executed"] == []
    assert torch.equal(actual_load(output / "batch00/sources.pt", weights_only=True), selected)
    assert torch.equal(actual_load(output / "batch00/target_z.pt", weights_only=True), target)
    metadata = runner.read_json(output / "batch00/target_metadata.json")
    assert metadata["observation_rows"] == row_ids.tolist()
    assert metadata["row_selection_informed_by_truth"] is False
    assert torch.equal(actual_load(output / "batch00/source_order_rec.pt", weights_only=True), source_recs)
    assert runner.run_batch(cfg, manifest, 0, output, torch.device("cpu")) == result
    assert len(events) == 5, "A completed batch reran FIA or evaluation"
    with (output / "batch00/per_image_metrics.csv").open("a", encoding="utf-8") as handle:
        handle.write("changed")
    with pytest.raises(RuntimeError, match="Completed batch output hash mismatch"):
        runner.run_batch(cfg, manifest, 0, output, torch.device("cpu"))


def test_comparison_png_preserves_pixels_labels_and_selected_image_identity(tmp_path):
    import hashlib

    import numpy as np
    from PIL import Image
    from qacpa import report_m9_rawrow_fia as report

    ramp = torch.linspace(0., 1., 224 * 224).reshape(1, 1, 224, 224)
    originals = ramp.expand(3, 3, 224, 224).clone()
    raw_recs = originals.flip(-1)
    cpa_recs = 1 - originals
    images = {"Original": originals, "CPA + FIA": cpa_recs, "Raw-row FIA": raw_recs}
    rows = [{"global_index": i, "image_path": f"image{i}.jpg", "lpips": .1 + i / 10,
             "source_abs_cosine": .2 + i / 10} for i in range(3)]
    path = tmp_path / "comparison.png"
    result = report.render_grid(images, rows, [2, 0], path, columns=2, title="Synthetic protocol check")

    assert result["image_indices"] == [2, 0]
    assert result["tile_size"] == 224 and len(result["tile_placements"]) == 6
    assert result["panels"] == list(images)
    assert result["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as saved:
        assert saved.width > saved.height
        for placement in result["tile_placements"]:
            index = placement["global_index"]
            assert placement["image_path"] == rows[index]["image_path"]
            expected = (images[placement["panel"]][index].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            actual = np.asarray(saved.crop(tuple(placement["box"])))
            assert actual.shape == (224, 224, 3)
            assert np.array_equal(actual, expected)
    with pytest.raises(ValueError, match="unique"):
        report.render_grid(images, rows, [0, 0], path, columns=2, title="Duplicate")


def test_comparison_rejects_same_indices_with_different_originals():
    from qacpa import report_m9_rawrow_fia as report

    current = [{"global_index": 0, "batch_index": 0, "source_index": 0,
                "image_path": "image-a.jpg", "label": 1}]
    reference = [{**current[0], "image_path": "image-b.jpg"}]
    with pytest.raises(RuntimeError, match="image identity mismatch"):
        report._validate_paired_identity(current, reference)


def test_frozen_config_protocol_and_code_drift_are_rejected(runner, tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    protocol_path = tmp_path / "protocol.md"
    code_path = tmp_path / "runner.py"
    config_path.write_text('{"seed": 0}', encoding="utf-8")
    protocol_path.write_text("Fixed raw rows and FIA only\n", encoding="utf-8")
    code_path.write_text("raw_rows = observation[row_ids]\n", encoding="utf-8")
    cfg = {"protocol": str(protocol_path)}
    monkeypatch.setattr(runner, "code_hashes", lambda: {"runner.py": runner.file_hash(code_path)})
    manifest = {"identity": {
        "config_sha256": runner.file_hash(config_path),
        "protocol_sha256": runner.file_hash(protocol_path),
        "code_sha256": runner.code_hashes(),
    }}
    runner.verify_frozen_code(config_path, cfg, manifest)
    for changed_path in (config_path, protocol_path, code_path):
        original = changed_path.read_bytes()
        changed_path.write_bytes(original + b"\nchanged after registration")
        with pytest.raises(RuntimeError, match="Frozen code, configuration or protocol changed"):
            runner.verify_frozen_code(config_path, cfg, manifest)
        changed_path.write_bytes(original)
    runner.verify_frozen_code(config_path, cfg, manifest)


def test_document_publication_failure_preserves_complete_experiment(runner, tmp_path, monkeypatch):
    cfg = {"run_tag": "synthetic_publication_gate"}
    runner.status(tmp_path, cfg, state="complete", n_images=80)
    marker = tmp_path / "execution_status.json"
    original_marker = marker.read_bytes()

    def fail_to_publish(*args, **kwargs):
        raise OSError("synthetic document write failure")

    monkeypatch.setattr(runner, "publish_status", fail_to_publish)
    runner.publish_status_safely(cfg, tmp_path, "complete")
    assert marker.read_bytes() == original_marker
    assert runner.read_json(marker)["state"] == "complete"
    publication = runner.read_json(tmp_path / "postprocess_status.json")
    assert publication["state"] == "failed"
    assert publication["experiment_state"] == "complete"
    assert "synthetic document write failure" in publication["error"]
    assert publication["started_at_utc"] and publication["finished_at_utc"]
