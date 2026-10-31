"""Scientific gates for packet-only fills, staged Direct FIA, and selection."""
from copy import deepcopy
import inspect
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from qacpa import random_fill as fill
from qacpa import staged_fia as staged
from qacpa import upstream_naive as original
from qacpa import run_random_fill_same80 as runner
from qacpa.topk_ratio_codec import encode_topk, decode_packet, _parse


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("percent", [10, 20, 30, 50])
def test_packet_only_fill_exact_bits_cutoff_and_reproducibility(percent):
    actual = torch.linspace(-1, 1, 400).reshape(20, 20)
    packet = encode_topk(actual, percent)
    _, mask, values = _parse(packet)
    a, audit = fill.random_fill_packet(packet, 10)
    b, again = fill.random_fill_packet(packet, 10)
    c, _ = fill.random_fill_packet(packet, 11)
    assert torch.equal(a, b) and audit == again
    assert not torch.equal(a, c)
    flat = a.numpy().reshape(-1)
    assert np.array_equal(flat[mask].view(np.uint32), values.view(np.uint32))
    assert np.all(np.abs(flat[~mask]) <= audit["tau"])
    assert audit["tau"] == float(np.abs(values).min())
    assert audit["extra_transmitted_bytes"] == 0
    assert packet == encode_topk(actual, percent)
    assert packet == encode_topk(a, percent)


def test_only_observed_packet_can_affect_fills():
    a = torch.tensor([[10., 9., .1, -.2, .3, .4, .5, .6, .7, .8]])
    b = a.clone()
    b[0, 2:] = torch.linspace(-.7, .7, 8)
    pa, pb = encode_topk(a, 20), encode_topk(b, 20)
    assert pa == pb
    assert torch.equal(fill.random_fill_packet(pa, 55)[0], fill.random_fill_packet(pb, 55)[0])


def test_zero_cutoff_and_negative_zero_bits():
    packet = encode_topk(torch.tensor([[1., -0., 0., 0.]]), 50)
    a, audit = fill.random_fill_packet(packet, 9)
    _, mask, values = _parse(packet)
    assert audit["tau"] == 0
    assert torch.equal(a, decode_packet(packet))
    assert np.array_equal(a.numpy().reshape(-1)[mask].view(np.uint32), values.view(np.uint32))


def test_sampler_chunk_boundaries_and_uniform_moments():
    n = fill.CHUNK + 16
    a = torch.linspace(-1, 1, n).reshape(2, -1)
    packet = encode_topk(a, 10)
    result, audit = fill.random_fill_packet(packet, 99)
    _, mask, _ = _parse(packet)
    dropped = result.numpy().reshape(-1)[~mask]
    expected = np.random.Generator(np.random.PCG64(99)).uniform(-audit["tau"], audit["tau"], len(dropped)).astype(np.float32)
    interior = np.nextafter(np.float32(audit["tau"]), np.float32(0))
    np.clip(expected, -interior, interior, out=expected)
    assert np.array_equal(dropped, expected)
    assert abs(dropped.mean()) < .005 * audit["tau"]
    assert np.isclose(dropped.var(), audit["tau"] ** 2 / 3, rtol=.01)


def test_rank_uses_only_score_with_fixed_ties():
    rows = [{"candidate": i, "relative_l2": 1. if i < 4 else 2., "lpips": 1-i/20} for i in reversed(range(20))]
    assert [row["candidate"] for row in fill.rank_candidates(rows, list(range(20)))[:3]] == [0, 1, 2]
    for row in rows:
        row["lpips"] *= -100
    assert [row["candidate"] for row in fill.rank_candidates(rows, list(range(20)))[:3]] == [0, 1, 2]
    with pytest.raises(ValueError):
        fill.rank_candidates(rows[:-1], list(range(20)))
    rows[0]["relative_l2"] = float("nan")
    with pytest.raises(ValueError):
        fill.rank_candidates(rows, list(range(20)))


def test_seed_plan_unique_and_scope_frozen():
    assert len({fill.candidate_seed(b, c) for b in range(5) for c in range(20)}) == 100
    cfg = runner.read_json(runner.PROJECT / "configs/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k.json")
    baseline = runner.validate_config(cfg)
    assert cfg["local_training"] == baseline["local_training"]
    assert cfg["stage1"] == baseline["stage1"] and cfg["fia"] == baseline["fia"]
    for key, value in (("keep_percent", 20), ("n", 10), ("T", 1)):
        changed = deepcopy(cfg)
        changed[key] = value
        with pytest.raises(ValueError):
            runner.validate_config(changed)
    changed = deepcopy(cfg)
    changed["randomization"]["advance_top_k"] = 5
    with pytest.raises(ValueError):
        runner.validate_config(changed)


def test_consistency_norm_and_replay_matches_manual_sgd():
    torch.manual_seed(19)
    model = torch.nn.Sequential(torch.nn.Linear(7, 6), torch.nn.ReLU(),
                                torch.nn.Dropout(.5), torch.nn.Linear(6, 4))
    reference = deepcopy(model).eval()
    z = torch.randn(5, 7)
    labels = torch.tensor([0, 1, 2, 3, 0])
    initial = reference[0].weight.detach().clone()
    for _ in range(10):
        loss = F.cross_entropy(reference(z), labels, reduction="none").mean()
        grads = torch.autograd.grad(loss, reference.parameters())
        with torch.no_grad():
            for p, g in zip(reference.parameters(), grads):
                p.add_(g, alpha=-.001)
    expected = (initial - reference[0].weight.detach()) / .001
    actual = fill.replay_classifier_update(model, z, labels)
    assert torch.equal(actual, expected)
    for a, b in zip(model.parameters(), reference.parameters()):
        assert torch.equal(a, b)
    assert fill.relative_l2(actual, actual) == 0
    assert math.isclose(fill.relative_l2(actual * 2, actual), 1., abs_tol=1e-7)


class TinyFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weights", torch.tensor([[1., .2, .3], [.2, 1., .1], [.3, .2, 1.]]))

    def forward(self, x, return_z=False):
        assert return_z
        return None, x.mean(dim=(2, 3)) @ self.weights


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


def test_original_direct_staged_equals_continuous_and_original_driver(tmp_path, monkeypatch):
    monkeypatch.setattr(staged, "VGG16FeatureAdapter", TinyFeatures)
    monkeypatch.setattr(original, "VGG16FeatureAdapter", TinyFeatures)
    config = {"n_iter": 10, "log_every": 2, "lr": .1, "tv": .1, "fi": 1., "optimizer": "adam", "scheduler": "cosine"}
    target = torch.tensor([[1., 2., 3.], [3., 2., 1.]])
    args = dict(target_z=target, device=torch.device("cpu"), config=config, seed=42)
    short = staged.run_staged_fia(**args, output_dir=tmp_path / "split", stop=2)
    state = torch.load(tmp_path / "split/fia_checkpoint.pt", weights_only=False)
    assert state["attack_state"]["sch"]["T_max"] == 10
    assert math.isclose(state["attack_state"]["opt"]["param_groups"][0]["lr"], .1 * (1 + math.cos(math.pi * 2/10)) / 2)
    assert short["completed_iterations"] == 2
    resumed = staged.run_staged_fia(**args, output_dir=tmp_path / "split", stop=10)
    continuous = staged.run_staged_fia(**args, output_dir=tmp_path / "continuous", stop=10)
    unchanged = original.run_upstream_fia(**args, output_dir=tmp_path / "original")
    assert torch.equal(resumed["rec"], continuous["rec"])
    assert torch.equal(resumed["rec"], unchanged["rec"])
    split_state = torch.load(tmp_path / "split/fia_checkpoint.pt", weights_only=False)
    full_state = torch.load(tmp_path / "continuous/fia_checkpoint.pt", weights_only=False)
    original_state = torch.load(tmp_path / "original/fia_checkpoint.pt", weights_only=False)
    assert_state_equal(split_state["attack_state"], full_state["attack_state"])
    assert_state_equal(split_state["attack_state"], original_state["attack_state"])
    with pytest.raises(RuntimeError, match="rewind"):
        staged.run_staged_fia(**args, output_dir=tmp_path / "split", stop=2)
    cfg2 = {**config, "n_iter": 12}
    with pytest.raises(RuntimeError, match="incompatible"):
        staged.run_staged_fia(**{**args, "config": cfg2}, output_dir=tmp_path / "split", stop=12)


def test_attack_boundary_and_evaluation_order():
    for function in (runner.run_candidate, fill.random_fill_packet, fill.consistency_score):
        text = inspect.getsource(function)
        assert "inputs.pt" not in text and "Z_evaluation_only" not in text
        assert "evaluate_images(" not in text and "evaluate_after_fia(" not in text
    text = inspect.getsource(runner.run_batch)
    assert text.index('"final_selection.json"') < text.index('inputs = torch.load')
    assert text.index('inputs = torch.load') < text.index('evaluate_after_fia(')


def test_changed_completed_candidate_cannot_be_reused(tmp_path):
    (tmp_path / "a.txt").write_text("before")
    identity = {"experiment": 1}
    runner.save_marker(tmp_path / "done.json", identity, ["a.txt"], candidate=0)
    assert runner.checked_marker(tmp_path / "done.json", identity)["candidate"] == 0
    (tmp_path / "a.txt").write_text("after")
    with pytest.raises(RuntimeError, match="hash"):
        runner.checked_marker(tmp_path / "done.json", identity)


def test_whole_batch_screen_top3_final_choice_and_completed_resume(tmp_path, monkeypatch):
    from types import SimpleNamespace
    cfg = runner.read_json(runner.PROJECT / "configs/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k.json")
    source = tmp_path / "source"
    (source / "batch00").mkdir(parents=True)
    cfg["source_output"] = str(source)
    inputs = {"x": torch.zeros(16, 3, 224, 224), "Z_evaluation_only": torch.eye(16, 25088),
              "metadata": [{"global_index": i, "batch_index": 0, "source_index": i,
                            "dataset_index": i, "label": i, "image_path": f"image{i}.jpg"} for i in range(16)]}
    torch.save(inputs, source / "batch00/inputs.pt")
    packet_path = tmp_path / "packet.bin"
    packet_path.write_bytes(b"mock-packet")
    identity = {"packets": [{"path": str(packet_path), "sha256": runner.file_hash(packet_path)}]}
    manifest = {"identity": identity}
    calls = []

    def fake_candidate(**kwargs):
        folder, candidate, stop = kwargs["folder"], kwargs["candidate"], kwargs["stop"]
        folder.mkdir(parents=True, exist_ok=True)
        calls.append((candidate, stop))
        phase = "screen" if stop == 5000 else "final"
        # Screen winners17,18,19; final winner18. Scores never read originals.
        score = (candidate-17 if candidate >= 17 else 100+candidate) if phase == "screen" else abs(candidate-18)
        runner.write_json(folder / "cpa_result.json", {"wall_time_sec": 1.})
        if phase == "final":
            assert (folder.parent / "screen_selection.json").exists()
            assert len([item for item in calls if item[1] == 5000]) == 20
            torch.save(torch.eye(16, 25088), folder / "sources.pt")
            torch.save(torch.full((16, 3, 224, 224), candidate / 20), folder / "final_rec.pt")
        return runner.save_marker(folder / f"{phase}.json", identity, [], candidate=candidate,
                                  relative_l2=float(score), ranking_wall_time_sec=1.,
                                  fia_optimizer_wall_time_sec=stop/1000)

    def evaluate_images(**kwargs):
        selected = runner.read_json(tmp_path / "out/batch00/final_selection.json")
        assert selected["selected_candidate"] == 18
        assert torch.all(kwargs["rec"] == .9)
        return SimpleNamespace(lpips=[.5] * 16, psnr=[22.] * 16, ssim=[.3] * 16)

    monkeypatch.setattr(runner, "run_candidate", fake_candidate)
    monkeypatch.setattr(runner, "verify_inputs", lambda *args: None)
    monkeypatch.setattr(runner, "evaluate_images", evaluate_images)
    output = tmp_path / "out"
    output.mkdir()
    runner.run_batch(cfg, manifest, 0, output, torch.device("cpu"))
    assert calls[:20] == [(i, 5000) for i in range(20)]
    assert calls[20:] == [(i, 25000) for i in (17, 18, 19)]
    rows = runner.read_csv(output / "batch00/per_image_metrics.csv")
    assert len(rows) == 16 and {int(row["selected_candidate"]) for row in rows} == {18}
    assert runner.read_json(output / "batch00/summary.json")["budget"]["fia_iterations_total"] == 160000
    runner.run_batch(cfg, manifest, 0, output, torch.device("cpu"))
    assert len(calls) == 23
