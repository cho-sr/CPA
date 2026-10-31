"""CPU-only sender/public receiver boundaries; toy modules are not VGG evidence."""
from copy import deepcopy
import json

import pytest
import torch
from torch import nn

from qacpa import deterministic_hybrid_setup as setup
from qacpa.deterministic_observation import decode_packet
from qacpa.full_update_pipeline import normalize_pixels


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    setup.configure_runtime(7)
    model = nn.Module()
    model.features = nn.AdaptiveAvgPool2d((1, 1))
    model.avgpool = nn.Identity()
    model.classifier = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Dropout(0),
                                     nn.Linear(4, 4), nn.ReLU(), nn.Dropout(0), nn.Linear(4, 1000))
    shapes = {"classifier." + name: tuple(p.shape) for name, p in model.classifier.named_parameters()}
    monkeypatch.setattr(setup, "_expected_shapes", lambda: shapes)
    monkeypatch.setattr(setup, "load_starting_vgg16", lambda checkpoint, device: deepcopy(model).to(device))
    checkpoint = tmp_path / "start.pt"
    torch.save(model.state_dict(), checkpoint)
    source_dir = tmp_path / "original_private"
    source_dir.mkdir()
    source = source_dir / "evaluation_truth.pt"
    torch.save({"x": normalize_pixels(torch.rand(16, 3, 224, 224)),
                "labels": torch.arange(16, dtype=torch.int64)}, source)
    return {"checkpoint": checkpoint, "source": source, "public": tmp_path / "public",
            "private": tmp_path / "private", "cpa": tmp_path / "cpa", "shapes": shapes}


@pytest.fixture
def collected(tiny):
    setup.collect(tiny["source"], tiny["checkpoint"], tiny["public"], tiny["private"], device="cpu")
    return tiny


def _manifest(case):
    path = case["public"] / "manifest.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_collect_checks_actual_sgd_and_codec_and_publishes_only_packets(collected):
    case = collected
    manifest, observations = setup.read_observation(case["public"], case["checkpoint"])
    assert set(observations) == set(case["shapes"])
    assert len(observations) == 6
    assert manifest["image_shape"] == [16, 3, 224, 224]
    assert manifest["training"]["steps"] == 10
    assert manifest["training"]["lr"] == .001
    assert manifest["permissions"]["labels"] == "private"
    assert {p.name for p in case["public"].iterdir()} == {"manifest.json"} | {
        f"packet_{index:02d}.bin" for index in range(6)}
    counts = manifest["communication"]
    assert counts["total_public_bytes"] == sum(p.stat().st_size for p in case["public"].iterdir())
    report = json.loads((case["private"] / "sender_validation.json").read_text())
    assert report["all_raw_deltas_torch_equal"] is True
    assert all(item["max_absolute_error"] == 0 for item in report["replay_comparison"].values())
    assert report["sender_codec_fit"]["exact_packet_match"] is True
    assert report["raw_updates_saved"] is False and report["features_saved"] is False
    saved = setup.load_tensor_file(case["private"] / "evaluation_truth.pt")
    assert set(saved) == {"x", "labels"}
    original = setup.load_tensor_file(case["source"])
    assert torch.equal(saved["x"], original["x"]) and torch.equal(saved["labels"], original["labels"])


def test_cpa_uses_only_new_public_packet_and_restores_strict_runtime(collected, monkeypatch):
    case = collected
    _, manifest = _manifest(case)
    packet = (case["public"] / manifest["layers"]["classifier.0.weight"]["packet"]).read_bytes()
    expected = -decode_packet(packet) / .001
    # Simulate permanently unavailable private data before the receiver runs.
    case["source"].parent.rename(case["source"].parent.with_name("offline_original"))
    case["private"].rename(case["private"].with_name("offline_private"))
    seen = {}

    def fake_cpa(**kwargs):
        seen.update(kwargs)
        torch.testing.assert_close(kwargs["Y_hat"], expected, atol=0, rtol=0)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        return {"S_hat": -torch.ones(16, 3), "log": [{"iter": 0, "loss": 1.}],
                "settings": {"n_iter": kwargs["config"]["cpa_ica_max_iter"]}, "source_hashes": {},
                "wall_time_sec": 1., "setup_wall_time_sec": .2, "optimizer_wall_time_sec": .8}

    monkeypatch.setattr(setup, "run_upstream_cpa", fake_cpa)
    report = setup.prepare_cpa(case["public"], case["cpa"], iterations=3, seed=42, device="cpu")
    assert seen["n_components"] == 16 and seen["seed"] == 42
    assert "debias" not in seen["config"]
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cuda.matmul.allow_tf32 is False
    target = setup.load_tensor_file(case["cpa"] / "target.pt")
    assert set(target) == {"target_z", "observation_manifest_sha256"}
    assert torch.equal(target["target_z"], torch.ones(16, 3))
    assert target["observation_manifest_sha256"] == setup.file_digest(case["public"] / "manifest.json")
    assert report["private_inputs_used"] is False
    assert report["upstream_negentropy_disabled_by_implementation"] is True
    assert report["upstream_tv_disabled_when_nv_positive"] is True


@pytest.mark.parametrize("mutation", ["labels", "missing_permission", "null_permission", "numeric_permission",
                                      "training", "shape", "missing_layer", "path", "counts"])
def test_rejects_manifest_tamper(collected, mutation):
    path, manifest = _manifest(collected)
    name = "classifier.0.weight"
    if mutation == "labels":
        manifest["labels"] = [0] * 16
    elif mutation == "missing_permission":
        del manifest["permissions"]["starting_checkpoint_public"]
    elif mutation == "null_permission":
        manifest["permissions"] = None
    elif mutation == "numeric_permission":
        manifest["permissions"]["starting_checkpoint_public"] = 1
    elif mutation == "training":
        manifest["training"]["steps"] = 1
    elif mutation == "shape":
        manifest["layers"][name]["shape"] = [3, 4]
    elif mutation == "missing_layer":
        del manifest["layers"][name]
    elif mutation == "path":
        manifest["layers"][name]["packet"] = "../private/evaluation_truth.pt"
    elif mutation == "counts":
        manifest["communication"]["code_bytes"] += 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(setup.ContractError):
        setup.read_observation(collected["public"], collected["checkpoint"])


def test_rejects_packet_corruption_and_changed_checkpoint(collected):
    original_checkpoint = collected["checkpoint"].read_bytes()
    collected["checkpoint"].write_bytes(original_checkpoint + b"changed")
    with pytest.raises(setup.ContractError, match="Checkpoint"):
        setup.read_observation(collected["public"], collected["checkpoint"])
    collected["checkpoint"].write_bytes(original_checkpoint)
    _, manifest = _manifest(collected)
    packet_path = collected["public"] / manifest["layers"]["classifier.0.weight"]["packet"]
    raw = bytearray(packet_path.read_bytes())
    raw[-1] ^= 1
    packet_path.write_bytes(raw)
    with pytest.raises(setup.ContractError, match="integrity"):
        setup.read_observation(collected["public"], collected["checkpoint"])


@pytest.mark.parametrize("which", ["public", "private"])
def test_collection_refuses_existing_outputs_without_changing_them(tiny, which):
    target = tiny[which]
    target.mkdir()
    old = target / "failure.json"
    old.write_bytes(b"old failure must remain")
    with pytest.raises(FileExistsError):
        setup.collect(tiny["source"], tiny["checkpoint"], tiny["public"], tiny["private"], device="cpu")
    assert old.read_bytes() == b"old failure must remain"
    assert not tiny["private" if which == "public" else "public"].exists()


@pytest.mark.parametrize("which", ["public_inside_private", "private_inside_public", "public_in_source"])
def test_collection_rejects_overlapping_trees(tiny, which):
    public, private = tiny["public"], tiny["private"]
    if which == "public_inside_private":
        public = private / "public"
    elif which == "private_inside_public":
        private = public / "private"
    else:
        public = tiny["source"].parent / "public"
    with pytest.raises(setup.ContractError, match="separate"):
        setup.collect(tiny["source"], tiny["checkpoint"], public, private, device="cpu")
    assert not public.exists() and not private.exists()


def test_fractional_private_labels_are_rejected_before_outputs(tiny):
    # Do not keep an mmap of the file open while replacing it on Windows.
    saved = torch.load(tiny["source"], weights_only=True, map_location="cpu")
    saved["labels"] = saved["labels"].float() + .5
    torch.save(saved, tiny["source"])
    with pytest.raises(setup.ContractError, match="int64"):
        setup.collect(tiny["source"], tiny["checkpoint"], tiny["public"], tiny["private"], device="cpu")
    assert not tiny["public"].exists() and not tiny["private"].exists()


def test_mismatched_replay_is_private_failure_and_never_completed_observation(tiny, monkeypatch):
    original = setup.fc_sgd_updates

    def incorrect(*args, **kwargs):
        result = original(*args, **kwargs)
        result["classifier.0.weight"] = result["classifier.0.weight"] + 1.
        return result

    monkeypatch.setattr(setup, "fc_sgd_updates", incorrect)
    with pytest.raises(RuntimeError, match="SGD/replay mismatch"):
        setup.collect(tiny["source"], tiny["checkpoint"], tiny["public"], tiny["private"], device="cpu")
    assert not (tiny["public"] / "manifest.json").exists()
    assert not (tiny["private"] / "sender_validation.json").exists()
    assert (tiny["private"] / "failure.json").is_file()


def test_cpa_existing_output_guard_precedes_read(tiny):
    tiny["cpa"].mkdir()
    marker = tiny["cpa"] / "report.json"
    marker.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        setup.prepare_cpa(tiny["public"], tiny["cpa"], device="cpu")
    assert marker.read_bytes() == b"original"


def test_runtime_requires_pre_cuda_workspace_and_resets_strict_flags(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "wrong")
    with pytest.raises(RuntimeError, match="before CUDA"):
        setup.configure_runtime()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    setup.configure_runtime()
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cudnn.deterministic is True


def test_cpa_rejects_truth_argument_on_cli():
    with pytest.raises(SystemExit) as error:
        setup.main(["cpa", "--observation", "public", "--output", "cpa", "--truth-source", "private"])
    assert error.value.code == 2
