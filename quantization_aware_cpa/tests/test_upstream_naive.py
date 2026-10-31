"""Small CPU gates for the original CPA/Direct path and truth-free attack input."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from qacpa import upstream_naive as backend


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings():
    return json.loads((PROJECT / "configs/m7_uia_4bit_B16.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def reject_local_attack(monkeypatch):
    from qacpa import stage1, stage2, uia, victim

    def forbidden(*args, **kwargs):
        raise AssertionError("Original naive path called local LS/Stage2/MP or victim training")

    for module, names in (
        (stage1, ("estimate_mixing_and_sources", "_least_squares_mixing", "_run_cpa_ica")),
        (stage2, ("solve_stage2_consistency", "oracle_pinv_initialization")),
        (uia, ("mp_prediction_from_quantized", "run_fc_only_fedavg_on_features", "run_fc_only_fedavg_checkpoints")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    for name in ("collect_victim_batch", "collect_uia_observation"):
        for module in (uia, victim):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)


def test_methods_are_loaded_from_original_source_files():
    modules = backend.load_upstream_modules()
    official = PROJECT.parent / "official_cpa/src"
    for cls, filename, methods in (
        (modules.gradient_inversion.CocktailPartyAttack, "gradient_inversion.py", ("whiten", "step", "get_rec")),
        (modules.feature_inversion.Direct, "feature_inversion.py", ("step", "get_rec")),
    ):
        for method in methods:
            assert Path(inspect.getsourcefile(getattr(cls, method))).resolve() == (official / filename).resolve()


def test_cpa_invokes_original_steps_on_observation_without_ls(
    settings, tmp_path, monkeypatch, reject_local_attack
):
    modules = backend.load_upstream_modules()
    cls = modules.gradient_inversion.CocktailPartyAttack
    original_step, original_rec = cls.step, cls.get_rec
    original_device = modules.gradient_inversion.get_device
    steps, reconstructions = [], []
    observation = torch.randn(4, 29, generator=torch.Generator().manual_seed(31))
    before_observation = observation.clone()
    config = deepcopy(settings["stage1"])
    config.update(n_components=3, cpa_ica_max_iter=4, cpa_ica_log_every=2)

    def step(attack):
        assert torch.equal(attack.X, observation)
        assert list(attack.model.parameters()) == []
        assert len(attack.opt.param_groups) == 1
        assert len(attack.opt.param_groups[0]["params"]) == 1
        assert attack.opt.param_groups[0]["params"][0] is attack.W_hat
        before = attack.W_hat.detach().clone()
        losses = original_step(attack)
        steps.append((before, attack.W_hat.detach().clone(), losses))
        return losses

    def get_rec(attack):
        result = original_rec(attack)
        reconstructions.append(result.detach().cpu().clone())
        return result

    monkeypatch.setattr(cls, "step", step)
    monkeypatch.setattr(cls, "get_rec", get_rec)
    result = backend.run_upstream_cpa(
        Y_hat=observation, n_components=3, device=torch.device("cpu"),
        config=config, output_dir=tmp_path, seed=7,
    )
    assert len(steps) == 4 and reconstructions
    assert any(not torch.equal(before, after) for before, after, _ in steps)
    assert all(losses["loss_ne"] == 0 for _, _, losses in steps), "Preserve the original disabled NE term"
    assert torch.equal(result["S_hat"], reconstructions[-1])
    assert result["S_hat"].shape == (3, 29) and torch.isfinite(result["S_hat"]).all()
    assert torch.equal(observation, before_observation)
    assert "A_hat" not in result
    assert modules.gradient_inversion.get_device is original_device
    resumed = backend.run_upstream_cpa(
        Y_hat=observation, n_components=3, device=torch.device("cpu"),
        config=config, output_dir=tmp_path, seed=7,
    )
    assert len(steps) == 4, "A complete upstream optimizer checkpoint was unnecessarily rerun"
    assert torch.equal(resumed["S_hat"], result["S_hat"])
    with pytest.raises(RuntimeError, match="incompatible upstream checkpoint/cache"):
        backend.run_upstream_cpa(
            Y_hat=observation + .01, n_components=3, device=torch.device("cpu"),
            config=config, output_dir=tmp_path, seed=7,
        )


class TinyFeatureModel(torch.nn.Module):
    instances = []

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([
            [1., .2, .1], [.1, 1., .2], [.2, .1, 1.], [.3, .4, .5], [.5, .3, .4],
        ]))
        self.initial_weight = self.weight.detach().clone()
        self.forward_calls = 0
        self.instances.append(self)

    def forward(self, x, return_z=False):
        self.forward_calls += 1
        assert not self.training
        z = x.mean(dim=(-2, -1)) @ self.weight.T
        return (None, z) if return_z else z


def test_direct_invokes_original_pixel_optimizer_with_frozen_model_and_no_truth(
    settings, tmp_path, monkeypatch, reject_local_attack
):
    modules = backend.load_upstream_modules()
    cls = modules.feature_inversion.Direct
    original_step, original_rec = cls.step, cls.get_rec
    original_device = modules.feature_inversion.get_device
    monkeypatch.setitem(modules.datasets.xshape_dict, "imagenet", [3, 8, 8])
    monkeypatch.setattr(backend, "VGG16FeatureAdapter", TinyFeatureModel)
    TinyFeatureModel.instances.clear()
    target = torch.arange(1, 16, dtype=torch.float32).reshape(3, 5) * 7
    before_target = target.clone()
    config = deepcopy(settings["fia"])
    config.update(n_iter=3, log_every=1)
    steps, reconstructions = [], []

    def step(attack):
        assert torch.equal(attack.Z.cpu(), target), "Targets were normalized or truth-rescaled"
        assert all(not p.requires_grad for p in attack.model.parameters())
        assert len(attack.opt.param_groups) == 1
        assert len(attack.opt.param_groups[0]["params"]) == 1
        assert attack.opt.param_groups[0]["params"][0] is attack.S_hat
        before = attack.S_hat.detach().clone()
        losses = original_step(attack)
        steps.append((before, attack.S_hat.detach().clone(), losses))
        return losses

    def get_rec(attack):
        result = original_rec(attack)
        reconstructions.append(result.detach().cpu().clone())
        return result

    monkeypatch.setattr(cls, "step", step)
    monkeypatch.setattr(cls, "get_rec", get_rec)
    result = backend.run_upstream_fia(
        target_z=target, device=torch.device("cpu"), config=config,
        output_dir=tmp_path, seed=42,
    )
    assert len(steps) == 3 and reconstructions
    assert any(not torch.equal(before, after) for before, after, _ in steps)
    assert torch.equal(result["rec"], reconstructions[-1])
    assert result["rec"].shape == (3, 3, 8, 8)
    assert torch.isfinite(result["rec"]).all()
    assert result["rec"].min() >= 0 and result["rec"].max() <= 1
    model = TinyFeatureModel.instances[-1]
    assert model.forward_calls == 3
    assert torch.equal(model.weight, model.initial_weight) and model.weight.grad is None
    assert torch.equal(target, before_target)
    assert modules.feature_inversion.get_device is original_device


def test_evaluation_reorders_only_images_preserving_signed_scaled_sources():
    from qacpa import run_m8_fedavg_naive as runner

    truth = torch.eye(4)
    permutation = torch.tensor([2, 0, 3, 1])
    sources = truth[permutation] * torch.tensor([-7., 2., 11., -3.]).unsqueeze(1)
    source_recs = permutation.float().view(4, 1, 1, 1).expand(4, 3, 2, 2).clone()
    original_sources, original_recs = sources.clone(), source_recs.clone()
    aligned, rows = runner.evaluate_after_fia(sources, source_recs, truth)
    assert torch.equal(aligned[:, 0, 0, 0], torch.arange(4, dtype=torch.float32))
    assert [r["estimated_source_index"] for r in rows] == [1, 3, 0, 2]
    assert all(r["measured_recovered_cos09"] == 1 for r in rows)
    assert torch.equal(sources, original_sources)
    assert torch.equal(source_recs, original_recs)
    with pytest.raises(ValueError, match="Every estimated source"):
        runner.evaluate_after_fia(sources[:-1], source_recs, truth)


def test_batch_completes_fia_before_loading_truth_and_preserves_source_order(
    tmp_path, monkeypatch, reject_local_attack
):
    from qacpa import run_m8_fedavg_naive as runner

    cfg = runner.read_json(runner.DEFAULT_CONFIG)
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "result"
    manifest = {"identity": {"synthetic": "source-order-gate"}}
    # Expanded input preserves production dimensions without allocating a 411MB matrix.
    observation = torch.tensor(.25).expand(4096, 25088)
    truth = torch.zeros(16, 25088)
    truth[:, :16] = torch.eye(16)
    permutation = torch.arange(16).roll(3)
    signed_scales = torch.arange(1., 17.) * torch.tensor([1., -1.] * 8)
    sources = truth[permutation] * signed_scales.unsqueeze(1)
    source_recs = (permutation.float() / 15).view(16, 1, 1, 1).expand(16, 3, 224, 224).clone()
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
            assert events == ["observation", "cpa", "fia"], "Truth was loaded before image optimization"
            events.append("truth")
            return inputs
        return actual_load(path, *args, **kwargs)

    def cpa(**kwargs):
        assert events == ["observation"]
        assert kwargs["Y_hat"] is observation and kwargs["n_components"] == 16
        events.append("cpa")
        return {"S_hat": sources, "log": [], "wall_time_sec": 1.}

    def fia(**kwargs):
        assert events == ["observation", "cpa"], "Evaluation correspondence leaked into FIA"
        assert torch.equal(kwargs["target_z"], sources.abs()), "FIA source order or magnitude was changed"
        assert "x_true" not in kwargs and "truth" not in kwargs
        assert kwargs["config"]["n_iter"] == 25000 and kwargs["config"]["tv"] == .1
        events.append("fia")
        return {"rec": source_recs, "log": [], "wall_time_sec": 2.}

    def evaluate(estimated, reconstructed, originals):
        assert events == ["observation", "cpa", "fia", "truth"]
        events.append("matching")
        return actual_evaluation(estimated, reconstructed, originals)

    def metrics(**kwargs):
        assert events[-1] == "matching"
        assert torch.equal(kwargs["rec"][:, 0, 0, 0], torch.arange(16, dtype=torch.float32) / 15)
        events.append("metrics")
        return SimpleNamespace(lpips=[.2] * 16, psnr=[25.] * 16, ssim=[.5] * 16)

    monkeypatch.setattr(runner, "verify_batch_source", lambda *args: source)
    monkeypatch.setattr(runner.torch, "load", load)
    monkeypatch.setattr(runner, "run_upstream_cpa", cpa)
    monkeypatch.setattr(runner, "run_upstream_fia", fia)
    monkeypatch.setattr(runner, "evaluate_after_fia", evaluate)
    monkeypatch.setattr(runner, "evaluate_images", metrics)
    result = runner.run_batch(cfg, manifest, 0, output, torch.device("cpu"))
    assert events == ["observation", "cpa", "fia", "truth", "matching", "metrics"]
    assert result["embedding_cos09"] == 16 and result["oracle_fills"] == 0 and result["ls_A"] is False
    assert torch.equal(actual_load(output / "batch00/sources.pt", weights_only=True), sources)
    assert torch.equal(actual_load(output / "batch00/source_order_rec.pt", weights_only=True), source_recs)
    assert runner.run_batch(cfg, manifest, 0, output, torch.device("cpu")) == result
    assert len(events) == 6, "A completed batch reran the attack or evaluation"
    with (output / "batch00/per_image_metrics.csv").open("a", encoding="utf-8") as handle:
        handle.write("changed")
    with pytest.raises(RuntimeError, match="Completed batch changed"):
        runner.run_batch(cfg, manifest, 0, output, torch.device("cpu"))
