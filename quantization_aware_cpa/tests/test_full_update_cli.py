"""CPU CLI gates: real meta VGG counts, configuration, and run preservation.

Execution is stubbed when testing failure handling; these are not VGG training
or memory measurements. Inspect uses the actual meta-device architecture.
"""
from copy import deepcopy
import json

import pytest
import torch

from qacpa import run_full_update as cli
from qacpa.communication_codec import CODEC_ID


@pytest.fixture(autouse=True)
def preserve_runtime_flags():
    threads = torch.get_num_threads()
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
     torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = flags


@pytest.fixture
def config():
    return {
        "contract": {
            "schema_version": 1, "model": "vgg16", "training_scope": "conv_and_fc",
            "observation_scope": "all_parameters", "observation_source": "single_client",
            "bits": 4, "codec": CODEC_ID, "delta_convention": "final_minus_initial",
            "row_layout": "weights_output_rows_bias_single_row", "num_classes": 1000,
            "normalization": "imagenet", "dropout_probability": 0.0,
            "image_shape": [16, 3, 224, 224],
            "permissions": {
                "packet_includes_public_radii": True,
                "bandwidth_accounting": "4bit_codes_plus_reported_metadata",
                "starting_checkpoint_public": True, "training_rules_public": True,
                "labels": "private",
            },
            "training": {"optimizer": "sgd", "lr": 0.001, "steps": 10,
                         "batch_order": [list(range(16)) for _ in range(10)],
                         "momentum": 0.0, "weight_decay": 0.0},
        },
        "attack": {"iterations": 1, "seed": 909, "image_lr": 0.01, "label_lr": 0.01,
                   "tv_weight": 0.0, "radius_weight": 1.0, "checkpoint_forward": True,
                   "saved_tensors_on_cpu": False, "log_every": 1, "fit_tolerance_steps": 1e-6},
    }


def save_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def receiver_args(tmp_path, config, command="probe"):
    path = save_config(tmp_path, config)
    observation = tmp_path / "observation"
    observation.mkdir()
    (observation / "manifest.json").write_text(json.dumps({"contract": config["contract"]}), encoding="utf-8")
    checkpoint = tmp_path / "public_start.pt"
    checkpoint.write_bytes(b"not loaded by CLI failure fixtures")
    output = tmp_path / "run"
    argv = ["--threads", "1", command, "--config", str(path), "--checkpoint", str(checkpoint),
            "--observation", str(observation), "--output", str(output), "--device", "cpu"]
    return argv, output


def unexpected_execution(**kwargs):
    pytest.fail("A validation or inspect command executed training/attack")


@pytest.mark.parametrize("pending", [False, True], ids=["explicit_contract", "pending_permissions"])
def test_inspect_counts_actual_meta_vgg16_without_execution(tmp_path, config, monkeypatch, capsys, pending):
    if pending:
        config["contract"]["permissions"]["starting_checkpoint_public"] = None
    path = save_config(tmp_path, config)
    checkpoint = tmp_path / "metadata_only_checkpoint.pth"
    hashed = []

    def digest(received):
        hashed.append(received)
        return "a" * 64

    monkeypatch.setattr(cli, "file_digest", digest)
    monkeypatch.setattr(cli, "collect_observation", unexpected_execution)
    monkeypatch.setattr(cli, "attack_observation", unexpected_execution)
    assert cli.main(["--threads", "1", "inspect", "--config", str(path),
                     "--checkpoint", str(checkpoint)]) == 0
    captured = capsys.readouterr()
    assert not captured.err
    report = json.loads(captured.out)
    assert report["status"] == ("pending_contract" if pending else "contract_valid")
    assert bool(report["pending_reason"]) is pending
    assert report["parameter_tensors"] == 32
    assert report["conv_tensors"] == 26 and report["fc_tensors"] == 6
    assert report["elements"] == 138_357_544
    assert report["code_bytes"] == 69_178_772
    assert report["radius_bytes"] == 53_728
    assert report["packet_header_bytes"] == 1_024
    assert report["layer_packet_bytes"] == 69_233_524
    assert report["training_or_attack_executed"] is False
    assert report["checkpoint_sha256"] == "a" * 64 and hashed == [checkpoint]
    assert {p.name for p in tmp_path.iterdir()} == {"config.json"}


@pytest.mark.parametrize("command", ["probe", "attack"])
@pytest.mark.parametrize("existing_state", ["empty", "failed", "complete"])
def test_existing_receiver_run_is_rejected_without_changing_artifacts(
        tmp_path, config, monkeypatch, capsys, command, existing_state):
    argv, output = receiver_args(tmp_path, config, command)
    output.mkdir()
    if existing_state != "empty":
        (output / "registered_design.json").write_text('{"keep": "old design"}', encoding="utf-8")
        (output / "failure.json").write_text('{"keep": "old failure"}', encoding="utf-8")
    if existing_state == "complete":
        (output / "result.json").write_text('{"keep": "old result"}', encoding="utf-8")
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    monkeypatch.setattr(cli, "attack_observation", unexpected_execution)
    assert cli.main(argv) == 2
    captured = capsys.readouterr()
    assert not captured.out
    report = json.loads(captured.err)
    assert report["error_type"] == "FileExistsError"
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


@pytest.mark.parametrize("invalid", [
    "extra_top_level", "missing_attack", "zero_iterations", "non_boolean_checkpoint",
    "wrong_bits", "null_permission", "null_config", "null_contract", "null_attack", "null_training",
])
def test_invalid_config_fails_before_sender_execution(tmp_path, config, monkeypatch, capsys, invalid):
    if invalid == "extra_top_level":
        config["private_truth"] = "must not be accepted"
    elif invalid == "missing_attack":
        del config["attack"]
    elif invalid == "zero_iterations":
        config["attack"]["iterations"] = 0
    elif invalid == "non_boolean_checkpoint":
        config["attack"]["checkpoint_forward"] = 1
    elif invalid == "wrong_bits":
        config["contract"]["bits"] = 8
    elif invalid == "null_permission":
        config["contract"]["permissions"]["packet_includes_public_radii"] = None
    elif invalid == "null_config":
        config = None
    elif invalid == "null_contract":
        config["contract"] = None
    elif invalid == "null_attack":
        config["attack"] = None
    else:
        config["contract"]["training"] = None
    path = save_config(tmp_path, config)
    output, private = tmp_path / "public", tmp_path / "private"
    monkeypatch.setattr(cli, "collect_observation", unexpected_execution)
    argv = ["--threads", "1", "collect", "--config", str(path),
            "--checkpoint", str(tmp_path / "unopened_checkpoint.pt"),
            "--inputs", str(tmp_path / "unopened_truth.pt"),
            "--output", str(output), "--private-output", str(private),
            "--codec-seed", "0", "--device", "cpu"]
    assert cli.main(argv) == 2
    report = json.loads(capsys.readouterr().err)
    assert report["status"] == "failed"
    assert report["conditions_automatically_reduced"] is False
    assert not output.exists() and not private.exists()


def test_packet_contract_mismatch_is_rejected_before_attack(tmp_path, config, monkeypatch, capsys):
    argv, output = receiver_args(tmp_path, config)
    differing = deepcopy(config["contract"])
    differing["training"]["lr"] = 0.02
    (tmp_path / "observation/manifest.json").write_text(json.dumps({"contract": differing}), encoding="utf-8")
    monkeypatch.setattr(cli, "attack_observation", unexpected_execution)
    assert cli.main(argv) == 2
    report = json.loads(capsys.readouterr().err)
    assert report["error_type"] == "ContractError"
    assert "differs from the observed packet" in report["message"]
    assert not output.exists()


@pytest.mark.parametrize("command", ["probe", "attack"])
@pytest.mark.parametrize("out_of_memory", [False, True], ids=["runtime_failure", "oom_failure"])
def test_registered_receiver_failure_is_saved_with_original_conditions(
        tmp_path, config, monkeypatch, capsys, command, out_of_memory):
    argv, output = receiver_args(tmp_path, config, command)

    def fail_after_registration(**kwargs):
        assert kwargs["probe"] is (command == "probe")
        assert kwargs["device"] == torch.device("cpu")
        assert kwargs["design"] == config["attack"]
        assert kwargs["mode"] == "cell"
        cli.write_json(kwargs["output"] / "registered_design.json", {"attack": kwargs["design"]})
        error = torch.OutOfMemoryError if out_of_memory else RuntimeError
        raise error("controlled failure after registration")

    monkeypatch.setattr(cli, "attack_observation", fail_after_registration)
    assert cli.main(argv) == 2
    report = json.loads(capsys.readouterr().err)
    assert report["status"] == ("out_of_memory" if out_of_memory else "failed")
    assert report["conditions_automatically_reduced"] is False
    assert report["reconstruction_success"] == "not_established"
    assert json.loads((output / "failure.json").read_text(encoding="utf-8")) == report
    assert json.loads((output / "registered_design.json").read_text(encoding="utf-8")) == {"attack": config["attack"]}
    assert not (output / "result.json").exists()
    assert not (output / "candidate.pt").exists()


@pytest.mark.parametrize("registered", [False, True], ids=["before_registration", "after_success_result"])
def test_failure_artifact_only_marks_an_incomplete_registered_run(
        tmp_path, config, monkeypatch, capsys, registered):
    argv, output = receiver_args(tmp_path, config)

    def fail_at_boundary(**kwargs):
        if registered:
            cli.write_json(output / "registered_design.json", {"attack": kwargs["design"]})
            cli.write_json(output / "result.json", {"status": "preserve completed result"})
        raise RuntimeError("controlled boundary failure")

    monkeypatch.setattr(cli, "attack_observation", fail_at_boundary)
    assert cli.main(argv) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "failed"
    assert not (output / "failure.json").exists()
    if registered:
        assert json.loads((output / "result.json").read_text(encoding="utf-8")) == {"status": "preserve completed result"}
    else:
        assert not output.exists()
