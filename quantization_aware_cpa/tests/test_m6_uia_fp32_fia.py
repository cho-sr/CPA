"""CPU tests of the queued UIA scientific boundary, checkpoints and report.

The full-path test replaces observation collection, CPA and FIA with explicit
synthetic results. No VGG model, GPU, real measurement or dataset is used.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qacpa import run_m6_uia_fp32_fia as runner


@pytest.fixture
def cfg():
    return runner.read_locked_config()[0]


def test_locked_fp32_configuration_rejects_quantization_and_ideal_substitution(cfg):
    config, lock = runner.read_locked_config()
    assert lock["sha256"] == "deaf2865732e28d02351b40cd1920f3561b1ea7088805fa00b6c3030fec3ff4e"
    assert config["uia"]["observation"] == "Y_delta_over_lr"
    for group, key, value in (("quantization", "enabled", True), ("uia", "uses_Y_factor_as_attack_input", True),
                              ("stage1", "mp_truncation", True), ("stage1", "n_components", 15)):
        changed = deepcopy(cfg)
        changed[group][key] = value
        with pytest.raises(ValueError, match="Unsupported"):
            runner.validate_config(changed)


def test_actual_update_selection_and_residual_gate():
    observed = torch.tensor(2.).expand(4096, 25088)
    ideal = torch.tensor(9.).expand(4096, 25088)
    payload = {"Y_delta_over_lr": observed, "Y_factor": ideal, "update_factor_rel_l2": 1e-4}
    assert runner.actual_observation(payload, maximum_relative_error=.001) is observed
    payload["update_factor_rel_l2"] = .01
    with pytest.raises(RuntimeError, match="residual gate"):
        runner.actual_observation(payload, maximum_relative_error=.001)
    payload["update_factor_rel_l2"] = float("nan")
    with pytest.raises(RuntimeError, match="residual gate"):
        runner.actual_observation(payload, maximum_relative_error=.001)


def test_rank_keeps_zero_noise_prediction_separate_from_numerical_rank():
    values = [1.] * 15 + [1e-12]
    result = runner.rank_diagnostics(values, dimension=25088, rel_tol=1e-5, abs_tol=1e-8)
    assert result["predicted_B_eff"] == 16
    assert result["nominal_zero_noise_MP_ceiling"] == 16
    assert result["numerical_rank_top16_observed_centered"] == 15
    assert result["lam_over_mp_edge"] == [None] * 16
    values[-1] = 0
    assert runner.rank_diagnostics(values, dimension=25088, rel_tol=1e-5, abs_tol=1e-8)["predicted_B_eff"] == 15


def test_fia_adapter_is_abs_and_permutation_without_oracle_scaling():
    truth = torch.eye(16)
    permutation = torch.tensor([5, 2, 9, 1, 4, 8, 0, 3, 7, 6, 15, 11, 10, 14, 12, 13])
    magnitude = torch.arange(2., 18.)
    estimated = -truth[permutation] * magnitude[:, None]
    targets, rows = runner.fia_targets(estimated, truth)
    for true_index, row in enumerate(rows):
        estimated_index = int(torch.nonzero(permutation == true_index)[0])
        assert row["estimated_source_index"] == estimated_index
        assert targets[true_index, true_index] == magnitude[estimated_index]
        assert row["source_abs_cosine"] == 1 and not row["oracle_fill"]
    with pytest.raises(RuntimeError, match="Full 16"):
        runner.fia_targets(estimated[:15], truth)


def test_stage1_call_passes_actual_tensor_and_fixed_full16_parameters(cfg, monkeypatch):
    captured = {}
    actual = torch.ones(4, 9)
    def fake(Y, delta, **kwargs):
        captured.update(Y=Y, delta=delta, **kwargs)
        return "unchanged_backend"
    monkeypatch.setattr(runner, "estimate_mixing_and_sources", fake)
    monkeypatch.setattr(runner, "seed_all", lambda seed: captured.update(global_seed=seed))
    assert runner.stage1_call(actual, cfg, 3, torch.device("cpu")) == "unchanged_backend"
    assert captured["Y"] is actual
    assert captured["n_components"] == 16 and captured["rounding"] == "fp32"
    assert captured["global_seed"] == 3
    assert captured["debias"] is False and captured["mp_truncation"] is False
    assert torch.count_nonzero(captured["delta"]) == 0
    assert captured["cpa_ica_max_iter"] == 25000 and captured["whitening_tolerance"] == .001


def test_nonfinite_mp_diagnostics_become_json_null(tmp_path):
    output = tmp_path / "stage1.json"
    runner.write_json(output, {"ratios": [float("inf"), float("nan"), np.float64(-np.inf), 2.]})
    text = output.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["ratios"] == [None, None, None, 2.]


def test_checkpoint_rejects_changed_identity_and_bytes(tmp_path):
    identity = {"config_sha256": "original", "stage": "FIA"}
    runner.save_checkpoint(tmp_path, "test", {"rec": torch.ones(2)}, identity)
    assert torch.equal(runner.load_checkpoint(tmp_path, "test", identity)["rec"], torch.ones(2))
    with pytest.raises(RuntimeError, match="Checkpoint"):
        runner.load_checkpoint(tmp_path, "test", {**identity, "config_sha256": "modified"})
    with (tmp_path / "test.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(RuntimeError, match="Checkpoint"):
        runner.load_checkpoint(tmp_path, "test", identity)


@pytest.mark.parametrize("value,expected", [(.410, "reference_range_met"), (.448, "reference_range_met"),
    (.44800001, "unclassified_review"), (.45, "protocol_difference_review"), (.50, "protocol_difference_review"),
    (.55, "pipeline_review"), (.7, "pipeline_review")])
def test_frozen_decision_bands_use_unrounded_mean_and_never_launch_4bit(cfg, value, expected):
    decision = runner.decision_label(value, cfg)
    assert decision["classification"] == expected
    assert decision["automatic_4bit"] is False


def test_dependency_requires_completed_matching_m7_report(cfg, tmp_path):
    cfg = deepcopy(cfg)
    cfg["dependency"]["execution_status"] = str(tmp_path / "execution.json")
    cfg["dependency"]["report_summary"] = str(tmp_path / "report.json")
    runner.write_json(tmp_path / "execution.json", {"state": "running"})
    runner.write_json(tmp_path / "report.json", {"report_status": "complete", "preregistration_sha256": cfg["dependency"]["preregistration_sha256"]})
    with pytest.raises(RuntimeError, match="M7 must complete"):
        runner.check_dependency(cfg)
    runner.write_json(tmp_path / "execution.json", {"state": "complete"})
    runner.check_dependency(cfg)
    runner.write_json(tmp_path / "report.json", {"report_status": "complete", "preregistration_sha256": "wrong"})
    with pytest.raises(RuntimeError, match="M7 must complete"):
        runner.check_dependency(cfg)


@dataclass
class SyntheticStage1:
    A_hat: torch.Tensor
    S_hat: torch.Tensor
    Y_recon: torch.Tensor
    lam_deb_topB: list
    B_eff: int = 16
    lam_over_mp_edge_all: tuple = (float("inf"), float("nan"))
    ica_n_iter: int = 25000


def test_all80_synthetic_cpu_pipeline_resume_and_strict_json(cfg, tmp_path, monkeypatch):
    torch.set_num_threads(2)
    cfg = deepcopy(cfg)
    cfg["paths"]["output_dir"] = str(tmp_path)
    cfg["paths"]["results_jsonl"] = str(tmp_path / "runs.jsonl")
    identity = {"protocol_sha256": "synthetic_config_hash", "code_sha256": {"fia.py": "unchanged"}}
    feature_path = tmp_path / "synthetic_feature.pt"
    torch.save({}, feature_path)
    truth_z = torch.zeros(16, 25088)
    truth_z[:, :16] = torch.eye(16)
    counters = {"update": 0, "stage1": 0, "fia": 0}
    monkeypatch.setattr(runner.UIAFeatureBatch, "from_cache_dict", classmethod(lambda cls, data: data))
    monkeypatch.setattr(runner, "actual_observation", lambda obs, **kw: obs["Y_delta_over_lr"] if isinstance(obs, dict) else obs.Y_delta_over_lr)
    def fake_update(feature, **kwargs):
        counters["update"] += 1
        return SimpleNamespace(Y_delta_over_lr=torch.tensor([[2.]]), Y_factor=torch.tensor([[9.]]),
            B_cumulative=torch.zeros(1), final_per_sample_loss=torch.zeros(16), update_factor_rel_l2=.0001,
            update_factor_max_abs=1e-6)
    def fake_stage1(Y, cfg, batch, device):
        counters["stage1"] += 1
        assert Y.item() == 2., "Ideal Y_factor substituted for observed update"
        return SyntheticStage1(A_hat=torch.zeros(2, 16), S_hat=-truth_z * 3., Y_recon=torch.zeros(1), lam_deb_topB=[1.] * 15 + [1e-12])
    def fake_fia(**kwargs):
        counters["fia"] += 1
        assert kwargs["device"].type == "cpu"
        assert kwargs["n_iter"] == 25000 and kwargs["tv"] == .1 and kwargs["seed"] == 42
        assert torch.equal(kwargs["target_z"], truth_z * 3.)
        metrics = SimpleNamespace(lpips=[.44] * 16, psnr=[20.] * 16, ssim=[.5] * 16)
        return SimpleNamespace(rec=torch.full((16, 3, 224, 224), .5), metrics=metrics, log=[{"iter": 24999, "loss": .1}], wall_time_sec=10.)
    monkeypatch.setattr(runner, "run_fc_only_fedavg_on_features", fake_update)
    monkeypatch.setattr(runner, "stage1_call", fake_stage1)
    monkeypatch.setattr(runner, "run_direct_fia", fake_fia)
    for batch in range(5):
        metadata = [{"global_index": batch * 16 + i, "batch_index": batch, "source_index": i,
                     "label": i, "dataset_index": batch * 16 + i, "image_path": f"synthetic/image_{batch*16+i}.jpg",
                     "initial_per_sample_loss": 0.} for i in range(16)]
        payload = {"x": torch.zeros(16, 3, 224, 224), "Z_evaluation_only": truth_z,
                   "metadata": metadata, "feature_cache_path": str(feature_path), "wall_time_sec": .01}
        runner.save_checkpoint(tmp_path / f"batch{batch:02d}", "inputs", payload,
                               {**identity, "batch_index": batch, "stage": "cached_input_audit"})
        runner.run_batch(cfg, identity, batch, tmp_path, torch.device("cpu"))
    assert counters == {"update": 5, "stage1": 5, "fia": 5}
    # A completed condition must not recollect, optimize, or recompute LPIPS.
    runner.run_batch(cfg, identity, 0, tmp_path, torch.device("cpu"))
    assert counters == {"update": 5, "stage1": 5, "fia": 5}
    summary = runner.summarize(cfg, identity, tmp_path)
    assert summary["status"] == "complete" and summary["n_images"] == 80
    assert summary["config_sha256"] == identity["protocol_sha256"]
    assert summary["metrics"]["measured_B_eff_cos09_total"] == 80
    assert summary["published_table5_reference"]["measured_minus_reference"] == pytest.approx(.03)
    assert summary["decision"]["automatic_4bit"] is False
    assert len(summary["per_victim_batch"]) == 5
    for path in tmp_path.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "NaN" not in text and "Infinity" not in text
        json.loads(text)
    assert (tmp_path / "all80_reconstruction_grid.png").is_file()
    assert (tmp_path / "per_batch_table.png").is_file()
    assert len((tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()) == 1
