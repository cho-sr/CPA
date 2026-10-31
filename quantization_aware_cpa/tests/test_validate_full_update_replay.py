"""Private prefix-validation gates on a CPU toy; no VGG resource claim."""
from copy import deepcopy
import json

import pytest
import torch
from torch import nn

from qacpa import full_update_pipeline as pipeline
from qacpa import validate_full_update_replay as diagnostic
from qacpa.full_update_observation import encode_updates


@pytest.fixture(autouse=True)
def restore_runtime():
    threads = torch.get_num_threads()
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.are_deterministic_algorithms_enabled())
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
     torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = flags[:4]
    torch.use_deterministic_algorithms(flags[4])


class TinyConvFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 2, 1, stride=32), nn.Tanh())
        self.classifier = nn.Sequential(nn.Linear(2, 1000))

    def forward(self, images):
        return self.classifier(self.features(images).mean(dim=(2, 3)))


@pytest.fixture
def case(tmp_path, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(514)
        model = TinyConvFC().train()
    checkpoint = tmp_path / "public_start.pt"
    torch.save(model.state_dict(), checkpoint)
    shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
    contract = {
        "schema_version": 1, "model": "vgg16", "training_scope": "conv_and_fc",
        "observation_scope": "all_parameters", "observation_source": "single_client",
        "bits": 4, "codec": pipeline.CODEC_ID, "delta_convention": "final_minus_initial",
        "row_layout": "weights_output_rows_bias_single_row", "num_classes": 1000,
        "normalization": "imagenet", "dropout_probability": 0.0, "image_shape": [1, 3, 224, 224],
        "permissions": {"packet_includes_public_radii": True,
                        "bandwidth_accounting": "4bit_codes_plus_reported_metadata",
                        "starting_checkpoint_public": True, "training_rules_public": True, "labels": "private"},
        "training": {"optimizer": "sgd", "lr": 0.02, "steps": 3, "batch_order": [[0], [0], [0]],
                     "momentum": 0.0, "weight_decay": 0.03},
    }
    design = {"iterations": 1, "seed": 0, "image_lr": 0.01, "label_lr": 0.01,
              "tv_weight": 0.0, "radius_weight": 1.0, "checkpoint_forward": True,
              "saved_tensors_on_cpu": False, "log_every": 1, "fit_tolerance_steps": 1e-6}
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"contract": contract, "attack": design}), encoding="utf-8")
    monkeypatch.setattr(pipeline, "vgg16_shapes", lambda: shapes)
    monkeypatch.setattr(diagnostic, "load_starting_vgg16", lambda path, device: deepcopy(model).to(device))
    generator = torch.Generator().manual_seed(90)
    updates = {name: torch.randn(shape, generator=generator) * 0.02 for name, shape in shapes.items()}
    observation = tmp_path / "public_packets"
    pipeline.write_observation(observation, encode_updates(updates, seed=909), contract,
                               pipeline.file_digest(checkpoint))
    private = tmp_path / "private"
    private.mkdir()
    truth = private / "evaluation_truth.pt"
    pixels = torch.linspace(0.1, 0.9, 3 * 224 * 224).reshape(1, 3, 224, 224)
    torch.save({"x": pipeline.normalize_pixels(pixels), "labels": torch.tensor([17])}, truth)
    return {"config": config, "observation": observation, "checkpoint": checkpoint,
            "truth": truth, "output": private / "replay_validation.json",
            "device": torch.device("cpu"), "threads": 1}


@pytest.mark.parametrize("chunk_rows", [1, 64])
def test_delta_errors_use_observed_steps_and_separate_zero_radii(chunk_rows):
    reference = torch.tensor([[0.2, -0.4], [0.0, 0.0]])
    replayed = torch.tensor([[0.24, -0.39], [0.03, -0.04]])
    stats = diagnostic.compare_parameter_delta(replayed, reference, torch.zeros_like(reference),
                                                torch.tensor([0.4, 0.0]), chunk_rows=chunk_rows)
    assert stats["exact_torch_equal"] is False
    assert stats["max_absolute_error"] == pytest.approx(0.04, abs=1e-8)
    assert stats["max_positive_radius_error_steps"] == pytest.approx(0.75, abs=2e-7)
    assert stats["max_zero_radius_error_abs"] == pytest.approx(0.04, abs=1e-8)


def test_bias_has_one_radius_and_subnormal_steps_do_not_underflow():
    tiny = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    actual = torch.tensor([0.0, 0.0])
    replayed = torch.stack([tiny, -tiny])
    stats = diagnostic.compare_parameter_delta(replayed, actual, actual, tiny.reshape(1))
    assert stats["max_positive_radius_error_steps"] == 7.5
    assert stats["max_zero_radius_error_abs"] == 0
    assert stats["max_absolute_error"] == float(tiny)


def test_all_prefixes_match_actual_sgd_without_saving_raw_updates(case, monkeypatch):
    original = diagnostic.differentiable_sgd
    calls = []

    def record_prefix(model, images, labels, spec, **kwargs):
        calls.append((len(spec.batch_order), kwargs, labels.dtype, labels.tolist()))
        return original(model, images, labels, spec, **kwargs)

    monkeypatch.setattr(diagnostic, "differentiable_sgd", record_prefix)
    before = {path: pipeline.file_digest(path) for path in (case["truth"], case["checkpoint"])}
    report = diagnostic.validate_replay(**case)
    assert [call[0] for call in calls] == [1, 2, 3]
    assert all(call[1] == {"create_graph": False, "checkpoint_forward": False} for call in calls)
    assert all(call[2:] == (torch.long, [17]) for call in calls)
    assert report["status"] == "exact_match" and report["all_prefixes_exact"] is True
    assert report["raw_updates_saved"] is False and report["uses_private_truth"] is True
    assert report["max_absolute_error"] == report["max_positive_radius_error_steps"] == 0
    assert report["reconstruction_success"] == "not_evaluated"
    assert [prefix["prefix"] for prefix in report["prefixes"]] == [1, 2, 3]
    assert all(prefix["exact_all_tensors"] and len(prefix["per_tensor"]) == 4 for prefix in report["prefixes"])
    assert json.loads(case["output"].read_text(encoding="utf-8")) == report
    assert {p.name for p in case["truth"].parent.iterdir()} == {"evaluation_truth.pt", "replay_validation.json"}
    assert {path: pipeline.file_digest(path) for path in before} == before


def test_one_corrupted_prefix_is_reported_without_hiding_later_prefixes(case, monkeypatch):
    original = diagnostic.differentiable_sgd

    def perturb(model, images, labels, spec, **kwargs):
        updates = original(model, images, labels, spec, **kwargs)
        if len(spec.batch_order) == 2:
            updates["features.0.weight"].flatten()[0] += 0.001
        return updates

    monkeypatch.setattr(diagnostic, "differentiable_sgd", perturb)
    report = diagnostic.validate_replay(**case)
    assert report["status"] == "numerical_difference" and report["all_prefixes_exact"] is False
    assert [prefix["exact_all_tensors"] for prefix in report["prefixes"]] == [True, False, True]
    assert report["max_absolute_error"] == pytest.approx(0.001, abs=1e-8)
    assert report["max_positive_radius_error_steps"] > 0


@pytest.mark.parametrize("location", ["public_tree", "outside_private", "existing"])
def test_report_stays_private_and_never_overwrites(case, location):
    if location == "public_tree":
        case["output"] = case["observation"] / "validation.json"
    elif location == "outside_private":
        case["output"] = case["truth"].parent.parent / "validation.json"
    else:
        case["output"].write_text("preserve previous validation", encoding="utf-8")
    with pytest.raises((pipeline.ContractError, FileExistsError)):
        diagnostic.validate_replay(**case)
    if location == "existing":
        assert case["output"].read_text(encoding="utf-8") == "preserve previous validation"
    else:
        assert not case["output"].exists()


def test_failure_preserves_completed_prefixes_only_in_private_report(case, monkeypatch):
    original = diagnostic.differentiable_sgd

    def fail_second_prefix(model, images, labels, spec, **kwargs):
        if len(spec.batch_order) == 2:
            raise RuntimeError("controlled replay diagnostic failure")
        return original(model, images, labels, spec, **kwargs)

    monkeypatch.setattr(diagnostic, "differentiable_sgd", fail_second_prefix)
    with pytest.raises(RuntimeError, match="controlled replay diagnostic failure"):
        diagnostic.validate_replay(**case)
    report = json.loads(case["output"].read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["error_type"] == "RuntimeError"
    assert [prefix["prefix"] for prefix in report["prefixes"]] == [1]
    assert not (case["observation"] / "replay_validation.json").exists()
