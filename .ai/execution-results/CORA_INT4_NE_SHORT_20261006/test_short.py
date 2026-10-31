import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location("short_run", Path(__file__).with_name("run_short.py"))
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_task_loads_restored_class_and_executes_exact_steps(tmp_path):
    upstream = run.backend.load_upstream_modules()
    cls = run.load_cora(upstream)
    args = SimpleNamespace(n_iter=3, lr=.001, opt="adam", sch="none", decor=.5, T=2.,
                           ne=1., nv=.13, l1=5., tv=.1, use_labels=False, batch_size=3,
                           sigma=0., C=0., ds="imagenet", model="vgg16", defense="observed",
                           exp_name="synthetic", attack="cp", fi_method="direct", fl_alg="fedavg")
    observation = torch.randn(4, 29, generator=torch.Generator().manual_seed(17))
    with run.backend._runtime_environment(upstream, torch.device("cpu"), tmp_path):
        Path(upstream.utils.get_attack_exp_path(args)).mkdir(parents=True)
        attack = cls(run.backend._ObservedMatrixModel(), [observation], None, args, 0)
    log, _ = run.backend._optimize(attack, 3, 1, tmp_path, "cpa", torch.device("cpu"))
    assert len(log) == 3 and all(entry["loss_ne"] < 0 for entry in log)
    assert torch.isfinite(attack.W_hat.grad).all() and attack.W_hat.grad.norm() > 0
    checkpoint = torch.load(tmp_path / "cpa_checkpoint.pt", weights_only=False)
    assert checkpoint["completed_iterations"] == 3
    assert attack.get_rec().shape == (3, 29)


@pytest.fixture
def complete_member(tmp_path):
    for kind, count in [("cpa", run.CPA_STEPS), ("fia", run.FIA_STEPS)]:
        folder = tmp_path / kind
        folder.mkdir()
        torch.save({"completed_iterations": count}, folder / f"{kind}_checkpoint.pt")
    torch.save({"pixels": torch.zeros(16, 3, 224, 224)}, tmp_path / "candidate.pt")
    record = {"cpa_completed": run.CPA_STEPS, "fia_completed": run.FIA_STEPS,
              "sampler": {"code_consistency": {"all_equal": True}, "scale_reproduced": True},
              "negentropy_last": -.1, "candidate_sha256": run.digest(tmp_path / "candidate.pt")}
    run.write(tmp_path / "result.json", record)
    return tmp_path, record


def test_gate_rejects_shortened_optimization_before_evaluation(complete_member):
    folder, _ = complete_member
    torch.save({"completed_iterations": 499}, folder / "fia/fia_checkpoint.pt")
    with pytest.raises(ValueError, match="Incomplete"):
        run.validate_member(folder)


def test_gate_rejects_disabled_negentropy(complete_member):
    folder, record = complete_member
    record["negentropy_last"] = 0
    run.write(folder / "result.json", record)
    with pytest.raises(ValueError, match="Negentropy was not active"):
        run.validate_member(folder)


def test_gate_rejects_changed_final_candidate(complete_member):
    folder, _ = complete_member
    with (folder / "candidate.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        run.validate_member(folder)
