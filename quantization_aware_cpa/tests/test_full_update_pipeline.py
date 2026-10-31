"""CPU integration gates with a toy Conv+FC model, never a VGG resource result.

Only model loading and expected parameter shapes are substituted. The sender,
packet files, contract, receiver, differentiable replay and optimization remain
the production implementations. Image and label dimensions retain the contract.
"""
from copy import deepcopy
import inspect
import json

import pytest
import torch
from torch import nn

from qacpa import full_update_pipeline as pipeline
from qacpa.full_update_observation import encode_updates, summarize_observation_fit
from qacpa.full_update_replay import differentiable_sgd


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ToyConvFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 2, 1, stride=32), nn.Tanh())
        self.classifier = nn.Sequential(nn.Linear(2, 1000))

    def forward(self, images):
        return self.classifier(self.features(images).mean(dim=(2, 3)))


@pytest.fixture
def contract():
    return {
        "schema_version": 1, "model": "vgg16", "training_scope": "conv_and_fc",
        "observation_scope": "all_parameters", "observation_source": "single_client",
        "bits": 4, "codec": pipeline.CODEC_ID, "delta_convention": "final_minus_initial",
        "row_layout": "weights_output_rows_bias_single_row", "num_classes": 1000,
        "normalization": "imagenet", "dropout_probability": 0.0,
        "image_shape": [1, 3, 224, 224],
        "permissions": {
            "packet_includes_public_radii": True,
            "bandwidth_accounting": "4bit_codes_plus_reported_metadata",
            "starting_checkpoint_public": True, "training_rules_public": True,
            "labels": "private",
        },
        "training": {"optimizer": "sgd", "lr": 0.02, "steps": 2,
                     "batch_order": [[0], [0]], "momentum": 0.0, "weight_decay": 0.0},
    }


@pytest.fixture
def toy(tmp_path, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(902)
        model = ToyConvFC().train()
    checkpoint = tmp_path / "public_start.pt"
    torch.save(model.state_dict(), checkpoint)
    shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}

    def load_model(path, device):
        loaded = deepcopy(model)
        loaded.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        return loaded.to(device).train()

    monkeypatch.setattr(pipeline, "vgg16_shapes", lambda: dict(shapes))
    monkeypatch.setattr(pipeline, "load_starting_vgg16", load_model)
    generator = torch.Generator().manual_seed(808)
    updates = {name: torch.randn(shape, generator=generator) * 0.03
               for name, shape in shapes.items()}
    return {"checkpoint": checkpoint, "shapes": shapes, "updates": updates,
            "entries": encode_updates(updates, seed=314), "load_model": load_model}


@pytest.fixture
def bundle(tmp_path, toy, contract):
    folder = tmp_path / "public_observation"
    manifest = pipeline.write_observation(folder, toy["entries"], contract,
                                          pipeline.file_digest(toy["checkpoint"]))
    return folder, manifest


def replace_manifest(folder, manifest):
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("permission", [
    "packet_includes_public_radii", "bandwidth_accounting", "starting_checkpoint_public",
    "training_rules_public", "labels",
])
@pytest.mark.parametrize("missing", [True, False], ids=["missing", "null"])
def test_each_permission_requires_an_explicit_value(contract, permission, missing):
    if missing:
        del contract["permissions"][permission]
    else:
        contract["permissions"][permission] = None
    with pytest.raises(pipeline.ContractError):
        pipeline.validate_contract(contract)


@pytest.mark.parametrize("permissions", [None, {}, True])
def test_permissions_container_is_never_defaulted(contract, permissions):
    contract["permissions"] = permissions
    with pytest.raises(pipeline.ContractError):
        pipeline.validate_contract(contract)


@pytest.mark.parametrize("permission", [
    "packet_includes_public_radii", "starting_checkpoint_public", "training_rules_public",
])
@pytest.mark.parametrize("value", [False, 1, "true"])
def test_public_information_requires_boolean_true(contract, permission, value):
    contract["permissions"][permission] = value
    with pytest.raises(pipeline.ContractError):
        pipeline.validate_contract(contract)


@pytest.mark.parametrize("field,value", [
    ("training_scope", "fc_only"), ("observation_scope", "classifier.0.weight"),
    ("observation_source", "aggregate"), ("bits", 8),
    ("delta_convention", "initial_minus_final_over_lr"),
    ("image_shape", [1, 3, 32, 32]), ("num_classes", 10),
])
def test_full_update_contract_rejects_incompatible_conditions(contract, field, value):
    contract[field] = value
    with pytest.raises(pipeline.ContractError):
        pipeline.validate_contract(contract)


@pytest.mark.parametrize("change", ["missing_conv", "missing_fc", "extra", "shape"])
def test_scope_requires_all_conv_and_fc_weights_biases(toy, change):
    entries = deepcopy(toy["entries"])
    if change == "missing_conv":
        del entries["features.0.bias"]
    elif change == "missing_fc":
        del entries["classifier.0.weight"]
    elif change == "extra":
        entries["unexpected.weight"] = entries["features.0.weight"]
    else:
        entries["features.0.weight"]["shape"] = [2, 3, 3, 3]
    with pytest.raises(pipeline.ContractError):
        pipeline.validate_parameter_scope(entries, toy["shapes"])


def test_packet_bundle_roundtrip_preserves_all_layers_and_public_accounting(bundle, toy, contract):
    folder, written = bundle
    manifest, observations = pipeline.read_observation(folder, toy["checkpoint"])
    assert manifest == written
    assert manifest["contract_sha256"] == pipeline.json_digest(contract)
    assert manifest["checkpoint_sha256"] == pipeline.file_digest(toy["checkpoint"])
    assert manifest["public_labels"] is None
    assert set(observations) == set(toy["shapes"])
    assert all(observations[name].shape == shape for name, shape in toy["shapes"].items())
    assert summarize_observation_fit(toy["updates"], observations)["closure_fit_within_tolerance"]
    packet_bytes = sum((folder / layer["file"]).stat().st_size for layer in manifest["layers"])
    assert manifest["communication"]["layer_packet_bytes"] == packet_bytes
    assert manifest["communication"]["elements"] == sum(p.numel() for p in toy["updates"].values())
    assert manifest["communication"]["radius_bytes"] > 0
    assert {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()} == {
        "manifest.json", *(layer["file"] for layer in manifest["layers"])
    }


def test_wrong_starting_checkpoint_is_rejected(bundle, toy, tmp_path):
    wrong = tmp_path / "wrong_start.pt"
    wrong.write_bytes(toy["checkpoint"].read_bytes() + b"different checkpoint bytes")
    with pytest.raises(pipeline.ContractError, match="Checkpoint"):
        pipeline.read_observation(bundle[0], wrong)


def test_packet_byte_corruption_is_rejected(bundle, toy):
    folder, manifest = bundle
    packet = folder / manifest["layers"][0]["file"]
    corrupted = bytearray(packet.read_bytes())
    corrupted[-1] ^= 1
    packet.write_bytes(corrupted)
    with pytest.raises(pipeline.ContractError, match="Packet integrity"):
        pipeline.read_observation(folder, toy["checkpoint"])


@pytest.mark.parametrize("tamper", [
    "contract", "shape", "name", "duplicate_name", "path", "packet_bytes",
    "code_bytes", "total_bytes", "private_labels", "truth_field",
])
def test_manifest_tampering_is_rejected(bundle, toy, tamper):
    folder, original = bundle
    manifest = deepcopy(original)
    if tamper == "contract":
        manifest["contract"]["training"]["lr"] *= 2
    elif tamper == "shape":
        manifest["layers"][0]["shape"] = [2, 3, 2, 2]
    elif tamper == "name":
        manifest["layers"][0]["name"] = "features.99.weight"
    elif tamper == "duplicate_name":
        manifest["layers"][1]["name"] = manifest["layers"][0]["name"]
    elif tamper == "path":
        manifest["layers"][0]["file"] = "../private/evaluation_truth.pt"
    elif tamper == "packet_bytes":
        manifest["layers"][0]["packet_bytes"] += 1
    elif tamper == "code_bytes":
        manifest["communication"]["code_bytes"] += 1
    elif tamper == "total_bytes":
        manifest["communication"]["layer_packet_bytes"] += 1
    elif tamper == "private_labels":
        manifest["public_labels"] = [12]
    else:
        manifest["private_inputs"] = "../private/evaluation_truth.pt"
    replace_manifest(folder, manifest)
    with pytest.raises(pipeline.ContractError):
        pipeline.read_observation(folder, toy["checkpoint"])


@pytest.mark.parametrize("visibility,labels", [("private", torch.tensor([12])), ("public", None)])
def test_writer_enforces_label_visibility_before_creating_files(tmp_path, toy, contract, visibility, labels):
    contract["permissions"]["labels"] = visibility
    output = tmp_path / "invalid_public_bundle"
    with pytest.raises(pipeline.ContractError, match="Public labels"):
        pipeline.write_observation(output, toy["entries"], contract,
                                   pipeline.file_digest(toy["checkpoint"]), labels)
    assert not output.exists()


def test_explicit_public_labels_roundtrip(tmp_path, toy, contract):
    contract["permissions"]["labels"] = "public"
    folder = tmp_path / "public_labels"
    pipeline.write_observation(folder, toy["entries"], contract,
                               pipeline.file_digest(toy["checkpoint"]), torch.tensor([999]))
    manifest, _ = pipeline.read_observation(folder, toy["checkpoint"])
    assert manifest["public_labels"] == [999]


def test_existing_observation_is_not_overwritten(bundle, toy, contract):
    folder, _ = bundle
    original = {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        pipeline.write_observation(folder, toy["entries"], contract, "different digest")
    assert {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()} == original


@pytest.mark.parametrize("relation", ["same", "private_inside_public", "public_inside_private"])
def test_sender_requires_separate_public_and_private_trees(tmp_path, toy, contract, relation):
    public, private = tmp_path / "public", tmp_path / "private"
    if relation == "same":
        private = public
    elif relation == "private_inside_public":
        private = public / "private"
    else:
        public = private / "public"
    with pytest.raises(pipeline.ContractError, match="separate directory trees"):
        pipeline.collect_observation(inputs=tmp_path / "nonexistent_truth.pt",
                                     checkpoint=toy["checkpoint"], contract=contract,
                                     output=public, private_output=private,
                                     codec_seed=0, device=torch.device("cpu"))
    assert not public.exists() and not private.exists()


@pytest.mark.parametrize("existing", ["public", "private"])
def test_sender_refuses_existing_outputs_before_reading_truth(tmp_path, toy, contract, existing):
    outputs = {"public": tmp_path / "public", "private": tmp_path / "private"}
    outputs[existing].mkdir()
    marker = outputs[existing] / "keep.txt"
    marker.write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        pipeline.collect_observation(inputs=tmp_path / "nonexistent_truth.pt",
                                     checkpoint=toy["checkpoint"], contract=contract,
                                     output=outputs["public"], private_output=outputs["private"],
                                     codec_seed=0, device=torch.device("cpu"))
    assert marker.read_text(encoding="utf-8") == "preserve me"


def test_receiver_optimizes_after_private_sources_are_removed(tmp_path, toy, contract, monkeypatch):
    source = tmp_path / "private_inputs.pt"
    pixels = torch.linspace(0.15, 0.85, 3 * 224 * 224).reshape(1, 3, 224, 224)
    torch.save({"x": pipeline.normalize_pixels(pixels), "metadata": [{"label": 12}]}, source)
    public, private = tmp_path / "packets", tmp_path / "sender_private"
    sent = pipeline.collect_observation(inputs=source, checkpoint=toy["checkpoint"], contract=contract,
                                        output=public, private_output=private,
                                        codec_seed=708, device=torch.device("cpu"))
    assert sent["training_steps"] == 2
    assert (private / "evaluation_truth.pt").is_file()
    provenance = json.loads((private / "sender_provenance.json").read_text(encoding="utf-8"))
    assert provenance["full_precision_update_saved"] is False
    assert provenance["sender_codec_fit"]["closure_fit_within_tolerance"]
    manifest, observations = pipeline.read_observation(public, toy["checkpoint"])
    assert manifest["public_labels"] is None
    assert set(observations) == set(toy["shapes"])

    # Remove both the source and sender artifacts from their original paths.
    source.rename(tmp_path / "unavailable_inputs.pt")
    private.rename(tmp_path / "unavailable_private")
    assert not source.exists() and not private.exists()
    accepted = set(inspect.signature(pipeline.attack_observation).parameters)
    assert accepted == {"observation", "checkpoint", "design", "output", "device", "mode", "probe"}
    assert set(inspect.signature(pipeline.read_observation).parameters) == {"folder", "checkpoint"}

    original_load = torch.load
    opened = []

    def only_public_checkpoint(path, *args, **kwargs):
        assert str(path) == str(toy["checkpoint"]), "Attack attempted a private tensor load"
        opened.append(str(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", only_public_checkpoint)
    replay_targets = []

    def track_replay_targets(model, images, targets, spec, **kwargs):
        replay_targets.append((tuple(targets.shape), targets.dtype, targets.requires_grad))
        return differentiable_sgd(model, images, targets, spec, **kwargs)

    monkeypatch.setattr(pipeline, "differentiable_sgd", track_replay_targets)
    design = {"iterations": 1, "seed": 909, "image_lr": 0.01, "label_lr": 0.01,
              "tv_weight": 0.0, "radius_weight": 1.0, "checkpoint_forward": True,
              "saved_tensors_on_cpu": False, "log_every": 1, "fit_tolerance_steps": 1e-6}
    output = tmp_path / "attack"
    result = pipeline.attack_observation(observation=public, checkpoint=toy["checkpoint"],
                                          design=design, output=output, device=torch.device("cpu"))
    assert opened
    assert result["status"] == "candidate_generated"
    assert result["iterations"] == 1 and result["image_gradient_nonzero"]
    assert result["label_optimization"] == "soft_relaxation"
    assert result["reconstruction_success"] == "not_evaluated"
    assert replay_targets == [((1, 1000), torch.float32, True), ((1,), torch.long, False)]
    candidate = original_load(output / "candidate.pt", weights_only=True, map_location="cpu")
    assert tuple(candidate["pixels"].shape) == (1, 3, 224, 224)
    assert torch.isfinite(candidate["pixels"]).all()
    initial_pixels = torch.rand(contract["image_shape"], generator=torch.Generator().manual_seed(design["seed"]))
    assert not torch.equal(candidate["pixels"], initial_pixels)
    assert bool(((candidate["pixels"] >= 0) & (candidate["pixels"] <= 1)).all())
    assert tuple(candidate["labels"].shape) == (1,) and candidate["labels"].dtype == torch.long
    assert bool(((candidate["labels"] >= 0) & (candidate["labels"] < 1000)).all())
    hard_updates = differentiable_sgd(toy["load_model"](toy["checkpoint"], torch.device("cpu")),
                                      pipeline.normalize_pixels(candidate["pixels"]), candidate["labels"],
                                      pipeline.validate_contract(contract), create_graph=False)
    assert result["hard_label_observation_fit"] == summarize_observation_fit(
        hard_updates, observations, tolerance_steps=design["fit_tolerance_steps"])
    original_result = (output / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        pipeline.attack_observation(observation=public, checkpoint=toy["checkpoint"],
                                    design=design, output=output, device=torch.device("cpu"))
    assert (output / "result.json").read_bytes() == original_result


@pytest.mark.parametrize("mode", ["cell", "center"])
def test_reverse_strategy_matches_unrolled_pixel_and_label_logit_step(
        tmp_path, toy, bundle, monkeypatch, mode):
    """Exercise ImageNet normalization, TV addition, and softmax chain rule."""
    original_adam = torch.optim.Adam
    steps = []

    class CapturedAdam(original_adam):
        def step(self, closure=None):
            parameters = [p for group in self.param_groups for p in group["params"]]
            snapshot = {"before": [p.detach().clone() for p in parameters],
                        "gradients": [p.grad.detach().clone() for p in parameters]}
            value = super().step(closure)
            snapshot["after"] = [p.detach().clone() for p in parameters]
            steps.append(snapshot)
            return value

    monkeypatch.setattr(torch.optim, "Adam", CapturedAdam)
    design = {"iterations": 1, "seed": 822, "image_lr": 0.01, "label_lr": 0.02,
              "tv_weight": 0.13, "radius_weight": 0.7, "checkpoint_forward": True,
              "saved_tensors_on_cpu": False, "log_every": 1, "fit_tolerance_steps": 1e-6}
    outputs = []
    for strategy in ("unrolled", "recompute_reverse"):
        output = tmp_path / strategy
        result = pipeline.attack_observation(
            observation=bundle[0], checkpoint=toy["checkpoint"],
            design={**design, "gradient_strategy": strategy}, output=output,
            device=torch.device("cpu"), mode=mode)
        assert result["status"] == "candidate_generated"
        assert result["label_optimization"] == "soft_relaxation"
        outputs.append(torch.load(output / "candidate.pt", weights_only=True, map_location="cpu"))
        registered = json.loads((output / "registered_design.json").read_text(encoding="utf-8"))
        assert registered["attack"]["gradient_strategy"] == strategy
        assert registered["method"] == "direct_image_optimization_without_CPA"
    assert len(steps) == 2
    assert [tuple(value.shape) for value in steps[0]["before"]] == [(1, 3, 224, 224), (1, 1000)]
    for first, second in zip(steps[0]["before"], steps[1]["before"], strict=True):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    for first, second in zip(steps[0]["gradients"], steps[1]["gradients"], strict=True):
        assert torch.isfinite(first).all() and first.norm() > 0
        # FP32 reverse accumulation has a different summation order. Check
        # global relative error as well as individual near-zero coordinates.
        assert float((first - second).norm() / first.norm()) < 5e-6
        torch.testing.assert_close(first, second, atol=1e-8, rtol=5e-5)
    for first, second in zip(steps[0]["after"], steps[1]["after"], strict=True):
        torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(outputs[0]["pixels"], outputs[1]["pixels"], atol=2e-6, rtol=2e-5)
    assert torch.equal(outputs[0]["labels"], outputs[1]["labels"])


@pytest.mark.parametrize("legacy", [False, True], ids=["zip_checkpoint", "legacy_checkpoint"])
def test_tensor_loader_preserves_exact_legacy_weights_and_file_digest(tmp_path, monkeypatch, legacy):
    checkpoint = tmp_path / "checkpoint.pt"
    state = {"features.0.weight": torch.linspace(-0.71, 0.89, 24).reshape(2, 3, 2, 2),
             "classifier.0.bias": torch.arange(1000, dtype=torch.float64) / 7}
    torch.save(state, checkpoint, _use_new_zipfile_serialization=not legacy)
    before = pipeline.file_digest(checkpoint)
    original_load, attempts = torch.load, []

    def record_load(path, **kwargs):
        attempts.append((path, dict(kwargs)))
        return original_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", record_load)
    loaded = pipeline.load_tensor_file(checkpoint)
    assert set(loaded) == set(state)
    for name in state:
        torch.testing.assert_close(loaded[name], state[name], atol=0, rtol=0)
        assert loaded[name].dtype == state[name].dtype
    assert pipeline.file_digest(checkpoint) == before
    assert len(attempts) == (2 if legacy else 1)
    assert attempts[0] == (checkpoint, {"map_location": "cpu", "weights_only": True, "mmap": True})
    if legacy:
        assert attempts[1] == (checkpoint, {"map_location": "cpu", "weights_only": True})


@pytest.mark.parametrize("error", [RuntimeError("corrupted archive"), OSError("unreadable checkpoint")])
def test_tensor_loader_does_not_retry_unrelated_failures(tmp_path, monkeypatch, error):
    calls = []

    def fail(path, **kwargs):
        calls.append((path, kwargs))
        raise error

    monkeypatch.setattr(torch, "load", fail)
    with pytest.raises(type(error), match=str(error)) as raised:
        pipeline.load_tensor_file(tmp_path / "corrupt.pt")
    assert raised.value is error and len(calls) == 1


@pytest.mark.parametrize("source_key", ["labels", "y", "metadata"])
@pytest.mark.parametrize("invalid_label", [12.75, True], ids=["fractional", "boolean"])
def test_sender_rejects_noninteger_labels_without_rounding(
        tmp_path, toy, contract, monkeypatch, source_key, invalid_label):
    source = tmp_path / "invalid_labels.pt"
    images = pipeline.normalize_pixels(torch.full((1, 3, 224, 224), 0.5))
    labels = [{"label": invalid_label}] if source_key == "metadata" else torch.tensor([invalid_label])
    torch.save({"x": images, source_key: labels}, source)
    public, private = tmp_path / "invalid_public", tmp_path / "invalid_private"

    def reject_model_loading(*args, **kwargs):
        pytest.fail("Sender loaded a model before rejecting noninteger labels")

    monkeypatch.setattr(pipeline, "load_starting_vgg16", reject_model_loading)
    with pytest.raises(pipeline.ContractError, match="integers|integer tensor"):
        pipeline.collect_observation(inputs=source, checkpoint=toy["checkpoint"], contract=contract,
                                     output=public, private_output=private,
                                     codec_seed=0, device=torch.device("cpu"))
    assert not public.exists() and not private.exists()
