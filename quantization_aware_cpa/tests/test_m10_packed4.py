"""CPU checks for selected-client provenance and the packet-only attack boundary."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from qacpa import run_m10_packed4 as runner


@pytest.fixture
def cfg():
    return runner.read_json(runner.DEFAULT_CONFIG)


def historical_rows():
    return [{"global_index": str(i), "lpips": str(i / 100)} for i in range(80)]


def test_selection_sorts_numeric_lpips_and_breaks_ties_by_global_index():
    rows = historical_rows()
    for row in rows:
        row["lpips"] = "0.5"
    rows[73]["lpips"] = "0.1"
    rows.reverse()
    original = deepcopy(rows)
    selected = runner.select_best10(rows)
    assert [int(r["global_index"]) for r in selected] == [73, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert rows == original


@pytest.mark.parametrize("fault", ["missing", "duplicate", "out_of_range", "nan", "infinity"])
def test_selection_rejects_incomplete_or_invalid_historical_population(fault):
    rows = historical_rows()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1]["global_index"] = "0"
    elif fault == "out_of_range":
        rows[-1]["global_index"] = "80"
    else:
        rows[-1]["lpips"] = "nan" if fault == "nan" else "inf"
    with pytest.raises(ValueError):
        runner.select_best10(rows)


def test_fixed_protocol_is_valid(cfg):
    runner.validate_config(cfg)


@pytest.mark.parametrize("path,value", [
    (("n",), 16), (("T",), 1), (("batch_indices",), [0, 1]),
    (("stage1", "cpa_ica_max_iter"), 10), (("fia", "tv"), .2),
    (("codec_seed",), 42), (("codec",), "legacy_dither"),
    (("new_training",), False), (("local_training", "momentum"), .9),
    (("debias",), True), (("fp32_attack",), True), (("raw_row_attack",), True),
    (("selection_reference",), "results/different_scores"),
    (("source_fp32_output",), "results/different_client"),
])
def test_protocol_rejects_unrequested_attack_or_training_changes(cfg, path, value):
    target = cfg
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        runner.validate_config(cfg)


def test_attack_reads_only_packet_then_passes_exact_decode_and_absolute_sources(cfg, tmp_path, monkeypatch):
    folder = tmp_path / "batch00"
    folder.mkdir()
    packet = b"synthetic received bytes"
    (folder / "observation.bin").write_bytes(packet)
    # An expanded scalar has the real observation shape without allocating its
    # 392 MiB of data. The mock CPA must receive this exact decoder object.
    decoded = torch.tensor(-3.5).expand(4096, 25088)
    sources = torch.arange(10, dtype=torch.float32).sub(4.5)[:, None].expand(10, 25088)
    rec = torch.tensor(.25).expand(10, 3, 224, 224)
    calls, saved = [], {}

    def decode(received):
        assert received == packet
        calls.append("decode")
        return decoded

    def cpa(**kwargs):
        assert calls == ["decode"]
        assert set(kwargs) == {"Y_hat", "n_components", "device", "config", "output_dir", "seed"}
        assert kwargs["Y_hat"] is decoded
        assert kwargs["n_components"] == 10 and kwargs["seed"] == 0
        assert kwargs["config"] == cfg["stage1"]
        calls.append("cpa")
        return {"S_hat": sources, "log": [], "wall_time_sec": 1.}

    def fia(**kwargs):
        assert calls == ["decode", "cpa"]
        assert set(kwargs) == {"target_z", "device", "config", "output_dir", "seed"}
        assert torch.equal(kwargs["target_z"], sources.abs())
        assert kwargs["target_z"].is_contiguous()
        assert kwargs["seed"] == 42 and kwargs["config"] == cfg["fia"]
        calls.append("fia")
        return {"rec": rec, "log": [], "wall_time_sec": 1.}

    def forbidden(*args, **kwargs):
        pytest.fail("Client truth/original checkpoint must not be read by the attack")

    monkeypatch.setattr(runner, "decode_packet", decode)
    monkeypatch.setattr(runner, "run_upstream_cpa", cpa)
    monkeypatch.setattr(runner, "run_upstream_fia", fia)
    monkeypatch.setattr(runner, "status", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.torch, "load", forbidden)
    monkeypatch.setattr(runner, "evaluate_after_fia", forbidden)
    monkeypatch.setattr(runner, "actual_observation", forbidden)
    monkeypatch.setattr(runner.torch, "save", lambda value, path: saved.update({path.name: value}))
    returned_sources, returned_recs, _, _ = runner.attack_from_packet(cfg, tmp_path, torch.device("cpu"))
    assert calls == ["decode", "cpa", "fia"]
    assert torch.equal(returned_sources, sources)
    assert torch.equal(returned_recs, rec)
    assert set(saved) == {"sources.pt", "source_order_rec.pt"}


def test_failed_actual_update_residual_aborts_before_checkpoint_or_compression(cfg, tmp_path, monkeypatch):
    cfg["source_output"] = str(tmp_path / "selected_client")
    inputs = {
        "Z_evaluation_only": torch.zeros(10, 25088),
        "metadata": [{"label": i, "initial_per_sample_loss": 1.} for i in range(10)],
    }
    actual = torch.zeros(1).expand(4096, 25088)
    observed = SimpleNamespace(Y_delta_over_lr=actual, update_factor_rel_l2=.01,
                               Y_factor=torch.tensor(12345.))
    called = []

    def collect(feature, **kwargs):
        assert feature.Z is inputs["Z_evaluation_only"]
        assert feature.labels.tolist() == list(range(10))
        assert kwargs == {"lr": .001, "steps": 10, "device": torch.device("cpu")}
        called.append("collection")
        return observed

    def forbidden(*args, **kwargs):
        pytest.fail("An invalid actual update must not be saved or replaced/encoded")

    monkeypatch.setattr(runner, "file_hash", lambda path: "selected-input-hash")
    monkeypatch.setattr(runner, "load_checkpoint", lambda *args: None)
    monkeypatch.setattr(runner.torch, "load", lambda *args, **kwargs: inputs)
    monkeypatch.setattr(runner, "status", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "seed_all", lambda seed: None)
    monkeypatch.setattr(runner, "run_fc_only_fedavg_on_features", collect)
    monkeypatch.setattr(runner, "save_checkpoint", forbidden)
    monkeypatch.setattr(runner, "encode_uint4", forbidden)
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with pytest.raises(RuntimeError, match="Actual-update/factor residual gate failed"):
            runner.collect_selected_update(cfg, {"identity": {}}, tmp_path, torch.device("cpu"))
    finally:
        torch.set_num_threads(old_threads)
    assert called == ["collection"]
