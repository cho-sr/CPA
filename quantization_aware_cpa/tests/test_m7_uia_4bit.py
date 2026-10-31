"""Meaningful CPU gates for paired observations, full16 attack and strict resume."""
from copy import deepcopy
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest
import torch

from qacpa import run_m7_uia_4bit as runner


@pytest.fixture
def cfg():
    return runner.read_config(runner.DEFAULT_CONFIG)


@pytest.mark.parametrize("group,key,value", [("fia", "tv", .2), ("fia", "n_iter", 100),
    ("stage1", "debias", True), ("stage1", "mp_truncation", True), ("stage1", "n_components", 15),
    ("quantization", "rounding", "deterministic"), ("quantization", "seeds", [42] * 5),
    ("uia", "uses_Y_factor_as_attack_input", True)])
def test_paired_protocol_changes_rejected(cfg, group, key, value):
    changed = deepcopy(cfg)
    changed[group][key] = value
    with pytest.raises(ValueError, match="Unsupported"):
        runner.validate_config(changed)


def test_seeded_quantization_uses_actual_and_preserves_zero_rows():
    actual = torch.tensor([[0., 0., 0., 0.], [.1, -.4, .3, 1.]])
    a = runner.quantize_actual(actual, 2)
    b = runner.quantize_actual(actual, 2)
    assert torch.equal(a["Y_hat"], b["Y_hat"])
    assert not torch.equal(a["Y_hat"], actual)
    assert torch.count_nonzero(a["Y_hat"][0]) == 0
    assert a["diagnostics"]["quantization_seed"] == 2000006
    assert a["diagnostics"]["cell_membership"] is True
    assert a["diagnostics"]["zero_scale_rows"] == 1


def test_stage1_full16_receives_quantized_input_without_diagnostic_truncation(cfg, monkeypatch):
    captured = {}
    Y, delta = torch.full((4, 9), 2.), torch.ones(4)
    monkeypatch.setattr(runner, "seed_all", lambda seed: captured.update(seed_all=seed))
    def fake(actual, d, **kwargs):
        captured.update(Y=actual, delta=d, **kwargs)
        return "sentinel"
    monkeypatch.setattr(runner, "estimate_mixing_and_sources", fake)
    assert runner.stage1_call(Y, delta, cfg, 3, torch.device("cpu")) == "sentinel"
    assert captured["Y"] is Y and captured["delta"] is delta
    assert captured["n_components"] == 16 and captured["rounding"] == "stochastic"
    assert captured["debias"] is False and captured["mp_truncation"] is False
    assert captured["cpa_ica_max_iter"] == 25000 and captured["seed_all"] == 3


@dataclass
class SyntheticStage1:
    A_hat: torch.Tensor
    S_hat: torch.Tensor
    Y_recon: torch.Tensor
    B_eff: int = 16


def test_batch_reuses_actual_observation_has_no_truth_fill_and_resumes(cfg, tmp_path, monkeypatch):
    torch.set_num_threads(2)
    cfg = deepcopy(cfg)
    source = tmp_path / "source"
    cfg["source_fp32_output"] = str(source)
    source.joinpath("batch00").mkdir(parents=True)
    truth_z = torch.zeros(16, 25088)
    truth_z[:, :16] = torch.eye(16)
    inputs = {"x": torch.zeros(16, 3, 224, 224), "Z_evaluation_only": truth_z,
              "metadata": [{"global_index": i, "batch_index": 0, "source_index": i, "label": i,
                            "image_path": f"synthetic/image{i}.jpg"} for i in range(16)]}
    observed = {"Y_delta_over_lr": torch.full((3, 12), 2.), "Y_factor": torch.full((3, 12), 9.),
                "final_per_sample_loss": torch.zeros(16), "update_factor_rel_l2": .0001, "update_factor_max_abs": .00001}
    torch.save({"observation": observed}, source / "batch00/observation.pt")
    monkeypatch.setattr(runner, "verify_source_batch", lambda *args: inputs)
    monkeypatch.setattr(runner, "actual_observation", lambda values, **kwargs: values["Y_delta_over_lr"])
    calls = {"quant": 0, "mp": 0, "cpa": 0, "fia": 0}
    quantizer = runner.quantize_actual
    def quant(actual, batch):
        calls["quant"] += 1
        assert torch.all(actual == 2), "Ideal factor substituted for the stored actual update"
        return quantizer(actual, batch)
    def mp(Y, delta, **kwargs):
        calls["mp"] += 1
        assert not torch.all(Y == 2)
        return {"B_eff_pred": 3, "lam_top": [1.] * 16}
    def stage1(Y, delta, config, batch, device):
        calls["cpa"] += 1
        assert Y.shape == (3, 12)
        return SyntheticStage1(torch.zeros(3, 16), -truth_z * 7., torch.zeros(1))
    def fia(**kwargs):
        calls["fia"] += 1
        assert torch.equal(kwargs["target_z"], truth_z * 7.)
        assert kwargs["n_iter"] == 25000 and kwargs["tv"] == .1 and kwargs["seed"] == 42
        return SimpleNamespace(rec=torch.full((16, 3, 224, 224), .5),
            metrics=SimpleNamespace(lpips=[.5] * 16, psnr=[20.] * 16, ssim=[.3] * 16), log=[], wall_time_sec=1.)
    monkeypatch.setattr(runner, "quantize_actual", quant)
    monkeypatch.setattr(runner, "mp_prediction_from_quantized", mp)
    monkeypatch.setattr(runner, "stage1_call", stage1)
    monkeypatch.setattr(runner, "run_direct_fia", fia)
    out = tmp_path / "result"
    identity = {"protocol_sha256": "synthetic"}
    first = runner.run_batch(cfg, identity, 0, out, torch.device("cpu"))
    assert first["predicted_B_eff"] == 3 and first["stage1_B_eff"] == 16
    assert first["measured_B_eff_cos09"] == 16 and first["oracle_fills"] == 0
    assert first["runtime_sec"]["update_collection"] == 0
    assert calls == {"quant": 1, "mp": 1, "cpa": 1, "fia": 1}
    runner.run_batch(cfg, identity, 0, out, torch.device("cpu"))
    assert calls == {"quant": 1, "mp": 1, "cpa": 1, "fia": 1}
    for path in out.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "NaN" not in text and "Infinity" not in text
        json.loads(text)
    with (out / "batch00/per_image_metrics.csv").open("a", encoding="utf-8") as handle:
        handle.write("corrupt")
    with pytest.raises(RuntimeError, match="Completed output changed"):
        runner.run_batch(cfg, identity, 0, out, torch.device("cpu"))


def test_three_condition_table_requires_same80_pairing(cfg, monkeypatch):
    fp32 = [{"global_index": str(i), "image_path": f"image{i}", "lpips": ".45", "psnr": "22", "ssim": ".3", "source_abs_cosine": ".99"} for i in range(80)]
    quant = [{**row, "lpips": ".55"} for row in fp32]
    gia = [{"method": "naive", "lpips": ".54", "psnr": "21", "ssim": ".2", "target_source_cosine": ".8"} for _ in range(40)]
    monkeypatch.setattr(runner, "read_csv", lambda path: gia if "aggregated" in str(path) else fp32)
    table, paired = runner.comparison_rows(cfg, quant)
    assert [row["n_images"] for row in table] == [80, 80, 40]
    assert [row["embedding_cos09"] for row in table] == [80, 80, 0]
    assert all(row["lpips_4bit_minus_fp32"] == pytest.approx(.1) for row in paired)
    quant[0]["image_path"] = "mismatched"
    with pytest.raises(RuntimeError, match="identity differs"):
        runner.comparison_rows(cfg, quant)


def test_worker_integer_exit_and_doc_failure_leave_scientific_completion(cfg, tmp_path, monkeypatch):
    result = {"status": "complete", "n_images": 80, "config_sha256": runner.file_hash(runner.DEFAULT_CONFIG), "output_hashes": {}}
    runner.write_json(tmp_path / "summary.json", result)
    class FakeProcess:
        pid = 4321
        def wait(self):
            return 0
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=17))
    assert runner.worker(runner.DEFAULT_CONFIG, cfg, tmp_path) == 0
    queue = json.loads((tmp_path / "queue_status.json").read_text(encoding="utf-8"))
    assert queue["state"] == "complete" and queue["runner_exit_code"] == 0
    postprocess = json.loads((tmp_path / "postprocess_status.json").read_text(encoding="utf-8"))
    assert postprocess["state"] == "failed"
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_complete80_report_includes_paired_table_diagnostics_and_random_grid(cfg, tmp_path, monkeypatch):
    cfg = deepcopy(cfg)
    out, source = tmp_path / "result", tmp_path / "fp32"
    cfg["source_fp32_output"] = str(source)
    cfg["gia_reference_csv"] = str(tmp_path / "gia.csv")
    all_fp32 = []
    for batch in range(5):
        folder, fp_folder = out / f"batch{batch:02d}", source / f"batch{batch:02d}"
        folder.mkdir(parents=True)
        fp_folder.mkdir(parents=True)
        rec = torch.full((16, 3, 224, 224), .5)
        torch.save({"x": torch.zeros_like(rec)}, folder / "inputs.pt")
        torch.save(rec, folder / "rec.pt")
        torch.save(rec, fp_folder / "rec.pt")
        rows = [{"global_index": batch * 16 + i, "batch_index": batch, "source_index": i,
                 "image_path": f"image{batch*16+i}", "lpips": .55, "psnr": 21., "ssim": .3,
                 "source_abs_cosine": .95, "fia_abs_target_cosine": .95,
                 "measured_recovered_cos09": 1, "oracle_fill": False} for i in range(16)]
        runner.write_csv(folder / "per_image_metrics.csv", rows)
        all_fp32.extend({**row, "lpips": .45} for row in rows)
    runner.write_csv(source / "per_image_metrics.csv", all_fp32)
    runner.write_csv(tmp_path / "gia.csv", [{"method": "naive", "lpips": .54, "psnr": 20., "ssim": .2, "target_source_cosine": .95} for _ in range(40)])
    def completed(folder, identity):
        return {"batch_index": identity["batch_index"], "n_images": 16, "predicted_B_eff": 15,
                "measured_B_eff_cos09": 16, "lpips_mean": .55, "psnr_mean": 21., "ssim_mean": .3,
                "runtime_sec": {"update_collection": 0., "quantization": 1., "MP_diagnostic": 1., "stage1": 2., "fia": 3.}}
    monkeypatch.setattr(runner, "validate_completed", completed)
    manifest = {"identity": {"protocol_sha256": "synthetic"}, "random_grid_global_indices": list(range(12))}
    summary = runner.summarize(cfg, manifest, out)
    assert summary["status"] == "complete" and summary["n_images"] == 80
    assert summary["metrics"]["predicted_B_eff_total"] == 75
    assert summary["metrics"]["measured_B_eff_cos09_total"] == 80
    assert summary["paired_lpips_delta_4bit_minus_fp32"] == pytest.approx(.1)
    assert len(runner.read_csv(out / "per_image_metrics.csv")) == 80
    assert len(runner.read_csv(out / "three_condition_comparison.csv")) == 3
    assert (out / "reconstruction_grid_not_cherry_picked.png").is_file()
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "subtractive-dither" in report and "0.0066" in report
    assert json.loads((out / "execution_status.json").read_text(encoding="utf-8"))["state"] == "complete"
