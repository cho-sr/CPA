"""CPU checks for exact historical image reuse and the received-packet boundary."""
from copy import deepcopy

import pytest
import torch

from qacpa import run_topk_ratios_same80 as runner


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=(30, 20, 10))
def cfg(request):
    return runner.read_json(runner.PROJECT / f"configs/cpa_fia_topk{request.param}_same80_B16.json")


def historical_batch(batch):
    metadata = [
        {"global_index": batch * 16 + index, "batch_index": batch,
         "source_index": index, "dataset_index": 1000 + batch * 16 + index,
         "label": 100 + index, "image_path": f"original/{batch}/{index}.JPEG"}
        for index in range(16)
    ]
    inputs = {"metadata": metadata,
              "x": torch.tensor(.25).expand(16, 3, 224, 224),
              "Z_evaluation_only": torch.tensor(2.).expand(16, 25088)}
    historical = [{**{key: str(value) for key, value in row.items()},
                   "lpips": str((16 - index) / 100)}
                  for index, row in enumerate(metadata)]
    return inputs, historical


@pytest.mark.parametrize("batch", [0, 4])
def test_original_order_is_preserved_regardless_of_historical_lpips(batch):
    inputs, historical = historical_batch(batch)
    original = deepcopy(inputs["metadata"])
    result = runner.validate_source_images(inputs, historical, batch)
    assert result == original
    assert [row["global_index"] for row in result] == list(range(batch * 16, (batch + 1) * 16))
    assert inputs["metadata"] == original


@pytest.mark.parametrize("field", runner.IDENTITY_KEYS)
def test_any_original_identity_corruption_is_rejected(field):
    inputs, historical = historical_batch(2)
    inputs["metadata"][5][field] = "substituted.JPEG" if field == "image_path" else 9999
    with pytest.raises(RuntimeError, match="(identity|order or batch)"):
        runner.validate_source_images(inputs, historical, 2)


def test_matching_reordered_tables_still_reject_changed_original_order():
    inputs, historical = historical_batch(1)
    inputs["metadata"][0], inputs["metadata"][1] = inputs["metadata"][1], inputs["metadata"][0]
    historical[0], historical[1] = historical[1], historical[0]
    with pytest.raises(RuntimeError, match="order or batch"):
        runner.validate_source_images(inputs, historical, 1)


def test_matching_tables_cannot_silently_duplicate_a_label():
    inputs, historical = historical_batch(0)
    inputs["metadata"][1]["label"] = inputs["metadata"][0]["label"]
    historical[1]["label"] = historical[0]["label"]
    with pytest.raises(RuntimeError, match="labels must remain distinct"):
        runner.validate_source_images(inputs, historical, 0)


@pytest.mark.parametrize("fault", ["missing", "wrong_shape", "nonfinite"])
def test_invalid_cached_images_abort_instead_of_replacing_them(fault):
    inputs, historical = historical_batch(3)
    if fault == "missing":
        inputs["metadata"].pop()
        historical.pop()
    elif fault == "wrong_shape":
        inputs["x"] = inputs["x"][:15]
    else:
        inputs["Z_evaluation_only"] = torch.tensor(float("nan")).expand(16, 25088)
    with pytest.raises(RuntimeError):
        runner.validate_source_images(inputs, historical, 3)


def test_source_hash_guard_detects_changed_checkpoint_bytes(tmp_path):
    folder = tmp_path / "batch03"
    folder.mkdir()
    source = folder / "observation.pt"
    source.write_bytes(b"frozen source checkpoint")
    manifest = {"inputs_sha256": {str(source): runner.file_hash(source)}}
    runner.verify_inputs(manifest, 3)
    source.write_bytes(b"different source checkpoint")
    with pytest.raises(RuntimeError, match="Frozen source changed"):
        runner.verify_inputs(manifest, 3)


def test_registered_same80_configuration_is_valid(cfg):
    runner.validate_config(cfg)


@pytest.mark.parametrize("n,n_images,batches", [
    (80, 80, [0]), (10, 80, list(range(8))), (16, 64, list(range(4))),
    (16, 80, [1, 0, 2, 3, 4]),
])
def test_config_rejects_changed_client_grouping(cfg, n, n_images, batches):
    cfg.update(n=n, n_images=n_images, batch_indices=batches)
    with pytest.raises(ValueError, match="non-compression setting"):
        runner.validate_config(cfg)


def write_packet_marker(folder, identity, batch, source_hash):
    folder.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name in ("observation.bin", "communication.json", "sparsification_diagnostics.json"):
        path = folder / name
        path.write_bytes(f"batch {batch} artifact {name}".encode())
        outputs[name] = runner.file_hash(path)
    marker = {"identity": identity, "batch_index": batch,
              "source_observation_sha256": source_hash, "output_hashes": outputs}
    runner.write_json(folder / "packet_manifest.json", marker)
    return marker


def test_valid_packet_artifacts_copied_to_another_batch_are_rejected(tmp_path):
    identity = {"config_sha256": "shared-across-five-batches"}
    original = tmp_path / "batch00"
    marker = write_packet_marker(original, identity, 0, "source-zero")
    assert runner.verify_completed(original, "packet_manifest.json", identity) == marker
    copied = tmp_path / "batch01"
    copied.mkdir()
    for name in (*marker["output_hashes"], "packet_manifest.json"):
        (copied / name).write_bytes((original / name).read_bytes())
    with pytest.raises(RuntimeError, match="identity or batch"):
        runner.verify_completed(copied, "packet_manifest.json", identity)


def test_packet_resume_rejects_wrong_original_update_without_reencoding(cfg, tmp_path, monkeypatch):
    batch = 2
    identity = {"config_sha256": "same80"}
    source = runner.resolve(cfg["source_output"]) / f"batch{batch:02d}/observation.pt"
    source_key = str(source.relative_to(runner.PROJECT))
    manifest = {"identity": identity, "inputs_sha256": {source_key: "expected-update-sha256"}}
    folder = tmp_path / f"batch{batch:02d}"
    marker = write_packet_marker(folder, identity, batch, "wrong-update-sha256")

    def forbidden(*args, **kwargs):
        pytest.fail("A mismatched packet must fail before source loading or reencoding")

    monkeypatch.setattr(runner.torch, "load", forbidden)
    monkeypatch.setattr(runner, "encode_topk", forbidden)
    with pytest.raises(RuntimeError, match="different original update"):
        runner.prepare_packet(cfg, manifest, batch, tmp_path)
    assert runner.read_json(folder / "packet_manifest.json") == marker


@pytest.mark.parametrize("batch", range(5))
def test_attack_uses_only_received_packet_and_keeps_all16_with_fixed_seeds(cfg, batch, tmp_path, monkeypatch):
    folder = tmp_path / f"batch{batch:02d}"
    folder.mkdir()
    packet = f"received batch {batch}".encode()
    (folder / "observation.bin").write_bytes(packet)
    # The full observation shape is a scalar view, without a 392 MiB allocation.
    decoded = torch.tensor(-3.5).expand(4096, 25088)
    sources = torch.arange(16, dtype=torch.float32).sub(7.5)[:, None].expand(16, 25088)
    recs = torch.tensor(.25).expand(16, 3, 224, 224)
    calls, saved = [], {}

    def decode(received):
        assert received == packet
        calls.append("decode")
        return decoded

    def cpa(**kwargs):
        assert calls == ["decode"]
        assert set(kwargs) == {"Y_hat", "n_components", "device", "config", "output_dir", "seed"}
        assert kwargs["Y_hat"] is decoded
        assert kwargs["n_components"] == 16 and kwargs["seed"] == batch
        assert kwargs["config"] == cfg["stage1"]
        assert kwargs["output_dir"] == folder / "cpa"
        calls.append("cpa")
        return {"S_hat": sources, "log": [], "wall_time_sec": 1.}

    def fia(**kwargs):
        assert calls == ["decode", "cpa"]
        assert set(kwargs) == {"target_z", "device", "config", "output_dir", "seed"}
        assert torch.equal(kwargs["target_z"], sources.abs())
        assert kwargs["target_z"].is_contiguous()
        assert kwargs["seed"] == 42 and kwargs["config"] == cfg["fia"]
        assert kwargs["output_dir"] == folder / "fia"
        calls.append("fia")
        return {"rec": recs, "log": [], "wall_time_sec": 1.}

    def forbidden(*args, **kwargs):
        pytest.fail("Packet-only attack must not access client truth, matching, or FP32 checkpoints")

    monkeypatch.setattr(runner, "decode_packet", decode)
    monkeypatch.setattr(runner, "run_upstream_cpa", cpa)
    monkeypatch.setattr(runner, "run_upstream_fia", fia)
    monkeypatch.setattr(runner, "status", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.torch, "load", forbidden)
    monkeypatch.setattr(runner, "evaluate_after_fia", forbidden)
    monkeypatch.setattr(runner, "actual_observation", forbidden)
    monkeypatch.setattr(runner.torch, "save", lambda value, path: saved.update({path.name: value}))
    returned_sources, returned_recs, _, _ = runner.attack_from_packet(cfg, batch, tmp_path, torch.device("cpu"))
    assert calls == ["decode", "cpa", "fia"]
    assert torch.equal(returned_sources, sources)
    assert torch.equal(returned_recs, recs)
    assert set(saved) == {"sources.pt", "source_order_rec.pt"}


def test_attack_and_evaluation_are_unchanged_from_packed4():
    import ast
    import inspect
    from qacpa import run_m10_packed4_same80 as reference
    assert ast.dump(ast.parse(inspect.getsource(runner.attack_from_packet))) == ast.dump(ast.parse(inspect.getsource(reference.attack_from_packet)))
    expected = inspect.getsource(reference.run_batch).replace("quantization_diagnostics.json", "sparsification_diagnostics.json")
    assert ast.dump(ast.parse(inspect.getsource(runner.run_batch))) == ast.dump(ast.parse(expected))


@pytest.mark.parametrize("path,value", [("stage1.cpa_ica_lr", .002), ("fia.seed", 43),
    ("local_training.steps", 11), ("sparsification.keep_fraction", .25),
    ("sparsification.quantization", True), ("source_adapter", "truth-scaled")])
def test_sparse_run_rejects_changes_beyond_compression(cfg, path, value):
    parent = cfg
    keys = path.split(".")
    for key in keys[:-1]:
        parent = parent[key]
    parent[keys[-1]] = value
    with pytest.raises(ValueError):
        runner.validate_config(cfg)
