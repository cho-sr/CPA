"""Small CPU regression checks for CoRA's restored negentropy objective."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import author_runner as runner
from cora_ica import CoRACocktailPartyAttack, require_negentropy_run, source_hash


def make_attack(cls, *, ne=1.0, magnitude=1.0):
    attack = cls.__new__(cls)
    attack.device = torch.device("cpu")
    attack.eps = torch.tensor(1e-20, dtype=torch.float64)
    attack.W_hat = torch.tensor([[1.0, .4], [.3, 1.0]], dtype=torch.float64, requires_grad=True)
    attack.X_w = magnitude * torch.tensor(
        [[-2.0, -1.0, .3, 1.0, 3.0], [4.0, .2, -1.0, 2.0, -3.0]], dtype=torch.float64
    )
    attack.W_w = torch.eye(2, dtype=torch.float64)
    attack.X_mu = torch.zeros(2, 1, dtype=torch.float64)
    attack.ne, attack.a, attack.decor, attack.T = ne, 1, .25, 2.0
    attack.tv, attack.nv, attack.l1 = 0.0, .1, .2
    attack.inp_shape = [2, -1]
    attack.opt = torch.optim.SGD([attack.W_hat], lr=.001)
    attack.sch = None
    return attack


def test_restored_contrast_matches_upstream_formula_and_changes_gradient():
    attack = make_attack(CoRACocktailPartyAttack)
    baseline = make_attack(runner.CocktailPartyAttack)
    W = attack.W_hat / (attack.W_hat.norm(dim=-1, keepdim=True) + attack.eps)
    expected = -(torch.log(torch.cosh(W @ attack.X_w) + attack.eps).mean(dim=-1) ** 2).mean()
    ne_gradient, = torch.autograd.grad(expected, attack.W_hat)
    losses, old_losses = attack.step(), baseline.step()
    assert losses["loss_ne"] == pytest.approx(expected.item(), abs=1e-12)
    assert losses["loss_ne"] < 0 and old_losses["loss_ne"] == 0
    assert losses["loss"] - old_losses["loss"] == pytest.approx(expected.item(), abs=1e-12)
    assert torch.allclose(attack.W_hat.grad - baseline.W_hat.grad, ne_gradient, atol=1e-12)
    assert ne_gradient.abs().sum() > 0
    assert not torch.equal(attack.W_hat, baseline.W_hat)


def test_explicit_ne_zero_preserves_upstream_step():
    attack = make_attack(CoRACocktailPartyAttack, ne=0)
    baseline = make_attack(runner.CocktailPartyAttack, ne=0)
    assert attack.step() == baseline.step()
    assert torch.equal(attack.W_hat, baseline.W_hat)
    assert torch.equal(attack.W_hat.grad, baseline.W_hat.grad)


def test_large_sources_have_finite_negentropy_and_gradient():
    attack = make_attack(CoRACocktailPartyAttack, magnitude=1000)
    losses = attack.step()
    assert losses["loss_ne"] < 0
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert torch.isfinite(attack.W_hat.grad).all()
    assert torch.isfinite(attack.W_hat).all()


@pytest.mark.parametrize("enabled", [False, True])
def test_wrapper_dispatches_requested_objective_and_records_it(monkeypatch, tmp_path, enabled):
    selected, manifests = [], []

    class Selected(Exception):
        pass

    class Original:
        def __init__(self, *args, **kwargs):
            selected.append("original")
            raise Selected

    class CoRA:
        def __init__(self, *args, **kwargs):
            selected.append("cora")
            raise Selected

    def prepare(args, **kwargs):
        manifests.append(kwargs["extra_manifest"])
        return SimpleNamespace(n_iter=1, n_iter_fi=1), tmp_path, {}

    monkeypatch.setattr(runner, "CocktailPartyAttack", Original)
    monkeypatch.setattr(runner, "CoRACocktailPartyAttack", CoRA)
    monkeypatch.setattr(runner, "_prepare_exp_path", prepare)
    monkeypatch.setattr(runner, "_write_whitening_pre_stats", lambda *args: None)
    monkeypatch.setattr(runner, "build_experiment_model", lambda **kwargs: torch.nn.Identity())
    monkeypatch.setattr(runner, "validate_parameter_mapping", lambda *args: None)
    monkeypatch.setattr(runner.author_utils, "get_device", lambda: torch.device("cpu"))
    args = SimpleNamespace(fi_method="direct", use_labels=False, attack_n_batch=1,
                           n_samples=2, n_sample_fi=2, seed=42, model="vgg16",
                           ds="imagenet", h_dim=2, global_checkpoint=None, ne=1.0)
    data = {"metadata": {"attack_trials": 1}, "param_names": [], "grad": [[]]}
    with pytest.raises(Selected):
        runner.run_author_cpa_with_update_file(
            args, exp_name="synthetic", update_file=tmp_path / "unused.pkl",
            update_data=data, method="CoRA", run_fia=False, enable_negentropy=enabled,
        )
    assert selected == ["cora" if enabled else "original"]
    assert manifests[0].get("negentropy_enabled", False) == enabled
    if enabled:
        assert manifests[0]["cora_ica_source_sha256"] == source_hash()


def test_reuse_rejects_old_disabled_results_and_accepts_matching_objective(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"ne": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="different negentropy objective"):
        require_negentropy_run(tmp_path, 1)
    path.write_text(json.dumps({
        "negentropy_enabled": True,
        "negentropy_behavior": "upstream_logcosh_contrast_without_zero_overwrite",
        "cora_ica_source_sha256": source_hash(),
    }), encoding="utf-8")
    require_negentropy_run(tmp_path, 1)
    with pytest.raises(ValueError):
        require_negentropy_run(tmp_path, 0)
