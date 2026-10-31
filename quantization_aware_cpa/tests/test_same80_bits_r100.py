from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qacpa import run_same80_bits_r100 as runner
from qacpa import same80_bits_r100_core as core


def test_all_bits_quantizer_ties_ranges_and_zero_edge():
    for bits in core.BITS:
        qmax = core.qmax_for_bits(bits)
        values = torch.tensor([qmax * 16, 8, 24, -8, -24], dtype=torch.float32)
        q, scale = core.encode_tensor(values, bits)
        assert scale.item() == 16.0
        assert q.tolist() == [qmax, 0, 2, 0, -2]
        assert int(q.min()) >= -qmax and int(q.max()) <= qmax
        zero_q, zero_scale = core.encode_tensor(torch.zeros(3, 4), bits)
        assert torch.equal(zero_q, torch.zeros_like(zero_q))
        assert zero_scale.item() == 1.0
        zero, audit = core.sample_matrix(zero_q, zero_scale, bits=bits, seed=77)
        assert torch.equal(zero, torch.zeros_like(zero))
        assert audit["anchor"] is None and audit["scale_reproduced"] is True


@pytest.mark.parametrize("bits", core.BITS)
def test_bounded_sampler_replays_anchor_scale_and_exact_codes(bits: int):
    source = torch.tensor([[127.0, -12.0, 0.0, 1.0],
                           [4.0, -3.0, 2.0, -1.0]], dtype=torch.float32)
    q, scale = core.encode_tensor(source, bits)
    sample, first = core.sample_matrix(q, scale, bits=bits, seed=9123, chunk_rows=32)
    replay, second = core.sample_matrix(q, scale, bits=bits, seed=9123, chunk_rows=32)
    assert torch.equal(sample, replay)
    assert first["tensor_sha256"] == second["tensor_sha256"]
    assert first["code_consistency"]["all_equal"] is True
    assert torch.equal(torch.round(sample / scale).clamp(-core.qmax_for_bits(bits),
                                                           core.qmax_for_bits(bits)).to(torch.int8), q)
    assert torch.equal(sample.abs().max() / core.qmax_for_bits(bits), scale)
    assert first["anchor"]["q"] in (-core.qmax_for_bits(bits), core.qmax_for_bits(bits))
    assert sample.reshape(-1)[first["anchor"]["flat_index"]].abs().item() == first["anchor"]["cap"]


def test_eight_bit_adapter_matches_established_int8_sampler():
    from qacpa.signed_int8_randomized import sample_matrix as established_sample

    source = torch.tensor([[127.0, -12.0, 0.0, 1.0],
                           [4.0, -3.0, 2.0, -1.0]], dtype=torch.float32)
    q, scale = core.encode_tensor(source, 8)
    actual, audit = core.sample_matrix(q, scale, bits=8, seed=9123, chunk_rows=32)
    expected, expected_audit = established_sample(q, scale, seed=9123, chunk_rows=32)
    assert torch.equal(actual, expected)
    assert audit["tensor_sha256"] == expected_audit["tensor_sha256"]
    assert audit["codec_metadata"]["logical_code_bits"] == 8
    assert audit["code_consistency"]["all_equal"] is True


def test_candidate_seeds_and_100_item_streaming_top3_match_full_sort():
    assert core.candidate_seed(0, 0) == 1234
    assert core.candidate_seed(4, 99) == 1234 + 1000003 * 499
    rows = [{"candidate_index": index, "screen_score": float(index % 7)} for index in range(100)]
    selector = core.StreamingTop3()
    for row in reversed(rows):
        selector.add(row)
    ranking = core.rank_full(rows, list(range(100)), "screen_score")
    assert [row["candidate_index"] for row in selector.rows] == [row["candidate_index"] for row in ranking[:3]]
    assert [row["candidate_index"] for row in ranking[:3]] == [0, 7, 14]
    final_rows = [{"candidate_index": index, "final_score": 4.0 - index}
                  for index in (0, 7, 14)]
    assert core.rank_full(final_rows, [0, 7, 14], "final_score")[0]["candidate_index"] == 14
    with pytest.raises(core.TaskError, match="nonfinite"):
        core.rank_full([{"candidate_index": i, "score": float("nan")} for i in range(100)],
                       list(range(100)), "score")
    with pytest.raises(core.TaskError, match="incomplete"):
        core.rank_full(rows[:-1], list(range(100)), "screen_score")


def test_locked_config_and_public_identity_hide_private_paths():
    config = core.read_json(runner.PROJECT / "configs/same80_bits87654_R100_S5k_K3_F25k.json")
    runner._validate_config(config)
    changed = json.loads(json.dumps(config))
    changed["fia"]["screen_steps"] += 1
    with pytest.raises(ValueError, match="immutable"):
        runner._validate_config(changed)
    records = [{"batch_index": 0, "input_path": "private/inputs.pt",
                "observation_path": "private/observation.pt", "input_sha256": "a" * 64}]
    public = runner._public_source_records(records)
    assert public == [{"batch_index": 0, "input_sha256": "a" * 64}]
    assert "path" not in json.dumps(public)


def test_cpa_adapter_calls_original_entrypoint_with_frozen_settings(tmp_path, monkeypatch):
    matrix = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    source = torch.zeros((16, 25088), dtype=torch.float32)
    expected_settings = {
        "n_components": 16, "seed": 2, "device": "cpu", "n_iter": 25000,
        "lr": 0.001, "opt": "adam", "sch": "none", "decor": 5.3,
        "T": 7.7, "ne": 1.0, "nv": 0.13, "l1": 5.0, "tv": 0.1,
        "log_every": 5000, "whitening": "original", "backend": "official_cpa/src/gradient_inversion.py:CocktailPartyAttack",
        "debias": False, "mp_truncation": False, "mixing_matrix_computed": False,
    }
    calls = []

    def fake_original(*, Y_hat, n_components, device, config, output_dir, seed):
        calls.append((Y_hat, n_components, device, config, seed))
        return {"S_hat": source, "settings": expected_settings,
                "source_hashes": {"fixture": "abc"},
                "log": [{"iter": 24999, "loss": 0.0}], "wall_time_sec": 1.0,
                "peak_cuda_allocated_bytes": 0, "cache_path": None}

    monkeypatch.setattr(runner.upstream_naive, "run_upstream_cpa", fake_original)
    actual, record = runner._run_cpa(matrix, stage_dir=tmp_path / "candidate000",
        identity={"test": "identity"}, batch=2, device=torch.device("cpu"),
        label="fixture", measure=False, task_root=tmp_path)
    assert len(calls) == 1
    assert calls[0][0] is matrix and calls[0][1] == 16 and calls[0][4] == 2
    assert calls[0][3] == runner._cpa_config(2)
    assert torch.equal(actual, source)
    assert record["completed_iterations"] == 25000
    assert record["settings"] == expected_settings
    # Reuse is authenticated and does not invoke the optimizer again.
    resumed, reused = runner._run_cpa(matrix, stage_dir=tmp_path / "candidate000",
        identity={"test": "identity"}, batch=2, device=torch.device("cpu"),
        label="fixture", measure=False, task_root=tmp_path)
    assert len(calls) == 1 and torch.equal(resumed, actual) and reused["status"] == "COMPLETED"


def test_fia_endpoint_score_and_original_25000_horizon_gate():
    score = {"feature_cosine_distance_mean": 0.25, "tv_normalized": 0.5,
             "score": 0.3, "raw_state_unchanged": True}
    settings = {"n_iter": 25000, "lr": 0.1, "tv": 0.1, "fi": 1.0,
                "optimizer": "adam", "scheduler": "cosine", "log_every": 5000,
                "seed": 42, "backend": "official_cpa/src/feature_inversion.py:Direct"}
    record = {"fia_settings": settings, "optimizer_log": [{"iter": 24999}], "score": score}
    runner._validate_fia_record(record, completed=25000)
    bad = json.loads(json.dumps(record))
    bad["fia_settings"]["n_iter"] = 5000
    with pytest.raises(core.TaskError, match="locked optimizer/horizon"):
        runner._validate_fia_record(bad, completed=25000)


def test_staged_fia_continuation_preserves_pixels_optimizer_scheduler_and_rng(tmp_path):
    from qacpa import staged_fia

    assert runner.FIA["n_iter"] == 25000
    optimize_until = staged_fia.optimize_until

    class TinyDirect:
        def __init__(self):
            self.S_hat = torch.nn.Parameter(torch.randn(()))
            self.opt = torch.optim.Adam([self.S_hat], lr=0.1)
            self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, 12)

        def step(self):
            self.opt.zero_grad(set_to_none=True)
            noise = torch.rand(()) + random.random() + float(np.random.random())
            loss = (self.S_hat - noise).square()
            loss.backward()
            self.opt.step()
            self.sch.step()
            return {"loss": float(loss.detach())}

        def get_attack_state(self):
            return {"S_hat": self.S_hat.detach().clone(), "opt": self.opt.state_dict(),
                    "sch": self.sch.state_dict()}

        def set_attack_state(self, state):
            with torch.no_grad():
                self.S_hat.copy_(state["S_hat"])
            self.opt.load_state_dict(state["opt"])
            self.sch.load_state_dict(state["sch"])

    def reset(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    reset(823)
    full = TinyDirect()
    optimize_until(full, stop=12, horizon=12, log_every=4,
                   output=full_dir, device=torch.device("cpu"))
    full_torch_rng = torch.get_rng_state().clone()
    full_python_rng, full_numpy_rng = random.getstate(), np.random.get_state()

    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    reset(823)
    prefix = TinyDirect()
    optimize_until(prefix, stop=4, horizon=12, log_every=4,
                   output=staged_dir, device=torch.device("cpu"))
    prefix_state = torch.load(staged_dir / "fia_checkpoint.pt", map_location="cpu", weights_only=False)
    assert prefix_state["completed_iterations"] == 4 and prefix_state["horizon"] == 12
    assert set(("rng_python", "rng_numpy", "rng_torch", "rng_cuda")).issubset(prefix_state)

    reset(991)  # Construction can consume a different stream; resume restores the checkpoint RNG.
    resumed = TinyDirect()
    optimize_until(resumed, stop=12, horizon=12, log_every=4,
                   output=staged_dir, device=torch.device("cpu"))
    assert torch.equal(full.S_hat, resumed.S_hat)
    assert full.opt.state_dict()["param_groups"] == resumed.opt.state_dict()["param_groups"]
    assert torch.equal(full.opt.state_dict()["state"][0]["exp_avg"],
                       resumed.opt.state_dict()["state"][0]["exp_avg"])
    assert full.sch.state_dict() == resumed.sch.state_dict()
    assert torch.equal(torch.get_rng_state(), full_torch_rng)
    assert random.getstate() == full_python_rng
    assert np.array_equal(np.random.get_state()[1], full_numpy_rng[1])
    with pytest.raises(RuntimeError, match="cosine horizon"):
        optimize_until(resumed, stop=12, horizon=13, log_every=4,
                       output=staged_dir, device=torch.device("cpu"))


def test_atomic_tensor_write_marker_integrity_and_safe_cleanup(tmp_path, monkeypatch):
    path = tmp_path / "candidate000/sources.pt"
    core.write_tensor(path, torch.tensor([1.0]))
    original_sha = core.file_sha256(path)
    actual_save = torch.save

    def interrupted_save(value, target):
        Path(target).write_bytes(b"partial")
        raise OSError("simulated interrupted atomic write")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted"):
        core.write_tensor(path, torch.tensor([2.0]))
    monkeypatch.setattr(torch, "save", actual_save)
    assert core.file_sha256(path) == original_sha
    assert not list(path.parent.glob("*.tmp"))
    marker = path.parent / "complete.json"
    core.save_marker(marker, {"id": "one"}, ["sources.pt"], completed_iterations=1)
    assert core.verify_marker(marker, {"id": "one"})["completed_iterations"] == 1
    path.write_bytes(b"tampered")
    with pytest.raises(core.TaskError, match="hash mismatch"):
        core.verify_marker(marker, {"id": "one"})
    disposable = path.parent / "fia_checkpoint.pt"
    disposable.write_bytes(b"checkpoint")
    core.remove_task_disposable(disposable, tmp_path, allowed_names={"fia_checkpoint.pt"})
    assert not disposable.exists()
    outside = tmp_path.parent / "unrelated.pt"
    outside.write_bytes(b"keep")
    with pytest.raises(core.TaskError, match="outside task root"):
        core.remove_task_disposable(outside, tmp_path, allowed_names={"unrelated.pt"})
    outside.unlink()


def test_task_lock_excludes_second_worker_and_changed_code_hash_is_rejected(tmp_path):
    with core.execution_lock(tmp_path):
        with pytest.raises(core.TaskError, match="another worker"):
            with core.execution_lock(tmp_path):
                pass
    source = tmp_path / "code.py"
    source.write_text("first", encoding="utf-8")
    expected = {str(source.resolve()): core.file_sha256(source)}
    core.verify_hash_map(expected)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(core.TaskError, match="hash changed"):
        core.verify_hash_map(expected)


def test_public_comparison_png_retains_every_input_pixel(tmp_path):
    originals = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
    originals[0, 0] = 1.0
    originals[1, 1] = 0.5
    naive = originals.clone()
    cora = originals.flip(0)
    rows = [{"global_index": i, "batch_index": i // 2, "source_index": i,
             "naive_lpips": 0.1, "cora_lpips": 0.2} for i in range(2)]
    record = __import__("qacpa.report_same80_bits_r100", fromlist=["render_comparison"]).render_comparison(
        originals, naive, cora, rows, tmp_path / "comparison.png", title="fixture", columns=2)
    assert record["n_images"] == 2 and record["native_pixels_verified"] is True
    assert (tmp_path / "comparison.png").is_file()


def test_attack_runner_has_no_private_evaluation_input_path():
    import inspect

    source = inspect.getsource(runner.run_public)
    assert "inputs.pt" not in source
    assert "Z_evaluation_only" not in source
    assert "evaluate_images(" not in source
    assert "evaluate_after_fia(" not in source


def test_exact_expected_budget_and_no_private_gate_shortcuts():
    budget = core.expected_budget()
    assert budget == {"cells": 25, "random_candidates": 2500, "naive_trajectories": 25,
                     "cpa_trajectories": 2525, "cpa_steps": 63125000,
                     "random_fia_steps": 14000000, "naive_fia_steps": 625000,
                     "fia_steps": 14625000, "random_finals": 75, "naive_finals": 25}
    rows = [{"bits": bits, "batch_index": batch, "screen_candidates": 100,
             "selected_finals": 3, "naive_trajectories": 1,
             "screen_selection_frozen": True, "final_selection_frozen": True,
             "private_data_used": False}
            for bits in core.BITS for batch in range(core.N_BATCHES)]
    assert core.verify_gate_counts(rows) == budget
    rows[0]["private_data_used"] = True
    with pytest.raises(core.TaskError, match="selection freeze"):
        core.verify_gate_counts(rows)


def test_recording_saver_installs_sidecar_wrapper_and_restores_original(tmp_path):
    checkpoint = tmp_path / "fia_checkpoint.pt"

    def original(path, payload):
        Path(path).write_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))

    module = SimpleNamespace(_save_checkpoint=original)
    saved_original = runner._recording_saver(
        module, tmp_path, horizon=25000, identity_sha="identity", kind="fia")
    assert module._save_checkpoint is not original
    module._save_checkpoint(checkpoint, {"completed_iterations": 25000})
    sidecar = runner.read_json(checkpoint.with_suffix(".pt.integrity.json"))
    assert sidecar["checkpoint_sha256"] == core.file_sha256(checkpoint)
    assert sidecar["completed_iterations"] == 25000
    assert sidecar["identity_sha256"] == "identity"
    module._save_checkpoint = saved_original
    assert module._save_checkpoint is original


def test_actual_staged_checkpoint_schema_scores_without_mutating_saved_pixels(tmp_path, monkeypatch):
    from qacpa import signed_int8_screen3_fia as score_adapter

    class ScoringAttack:
        def __init__(self):
            self.S_hat = torch.nn.Parameter(torch.zeros((16, 3, 224, 224), dtype=torch.float32))
            self.mean = torch.zeros((1, 3, 1, 1), dtype=torch.float32)
            self.std = torch.ones((1, 3, 1, 1), dtype=torch.float32)
            self.device = torch.device("cpu")
            self.model = lambda value, return_z=False: (
                None, value.mean(dim=(1, 2, 3))[:, None].repeat(1, 2))

        def set_attack_state(self, state):
            with torch.no_grad():
                self.S_hat.copy_(state["S_hat"])

        @staticmethod
        def cosine_similarity(left, right):
            return torch.nn.functional.cosine_similarity(left, right, dim=1)

        @staticmethod
        def total_variation(value):
            return (value[:, :, 1:, :] - value[:, :, :-1, :]).abs().mean()

    raw_pixels = torch.full((16, 3, 224, 224), 0.375, dtype=torch.float32)
    checkpoint = tmp_path / "fia_checkpoint.pt"
    # This is the real staged_fia.optimize_until checkpoint schema: notably,
    # it has no raw_image_sha256 field.
    torch.save({
        "completed_iterations": 25000,
        "horizon": 25000,
        "attack_state": {"S_hat": raw_pixels, "opt": {}, "sch": {}},
        "log": [{"iter": i * 5000 - 1} for i in range(1, 6)],
        "wall_time_sec": 123.0,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": [],
    }, checkpoint)
    attack = ScoringAttack()

    def construct(target, *, output, seed, device):
        output.mkdir(parents=True, exist_ok=True)
        return attack, {"seed": seed, "device": str(device)}

    monkeypatch.setattr(score_adapter, "construct_attack", construct)
    target_z = torch.ones((16, 2), dtype=torch.float32)
    score = runner._score_checkpoint(
        target_z, checkpoint, output=tmp_path / "fia", source_sha="source",
        identity_sha="identity", device=torch.device("cpu"), task_root=tmp_path)
    expected = score_adapter.tensor_sha256(raw_pixels)
    assert score["raw_state_unchanged"] is True
    assert score["raw_image_sha256_before"] == expected
    assert score["raw_image_sha256_after"] == expected
    assert score["checkpoint_sha256"] == core.file_sha256(checkpoint)
    assert score["completed_iterations"] == 25000


def test_reviewed_completed_checkpoint_reuse_executes_zero_optimizer_steps(tmp_path, monkeypatch):
    from qacpa import staged_fia

    output = tmp_path / "fia"
    output.mkdir()
    original_checkpoint = output / "fia_checkpoint.pt"
    rec_path = output / "rec_25000.pt"
    saved_rec = torch.full((2, 3), 0.25, dtype=torch.float32)
    torch.save(saved_rec, rec_path)

    class TinyDirect:
        step_calls = 0

        def __init__(self):
            self.S_hat = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
            self.opt = torch.optim.Adam([self.S_hat], lr=0.1)
            self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, 25000)

        def set_attack_state(self, state):
            with torch.no_grad():
                self.S_hat.copy_(state["S_hat"])
            self.opt.load_state_dict(state["opt"])
            self.sch.load_state_dict(state["sch"])

        def step(self):
            type(self).step_calls += 1
            raise AssertionError("a reviewed completed checkpoint must not optimize")

        def get_rec(self):
            return saved_rec.clone()

    attack = TinyDirect()
    torch.save({
        "completed_iterations": 25000, "horizon": 25000,
        "attack_state": {"S_hat": attack.S_hat.detach().clone(),
                         "opt": attack.opt.state_dict(), "sch": attack.sch.state_dict()},
        "log": [{"iter": i * 5000 - 1} for i in range(1, 6)],
        "wall_time_sec": 456.0,
        "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(), "rng_cuda": [],
    }, original_checkpoint)
    checkpoint_sha = core.file_sha256(original_checkpoint)
    ledger = output / "naive_fia_attempts.json"
    ledger.write_text('{"attempts":[{"attempt":1,"status":"COMPLETED"}]}\n', encoding="utf-8")
    ledger_bytes = ledger.read_bytes()
    original_saver = staged_fia._save_checkpoint

    def run_staged_fia(*, target_z, device, config, output_dir, seed, stop):
        restored = TinyDirect()
        log, runtime = staged_fia.optimize_until(
            restored, stop=stop, horizon=config["n_iter"],
            log_every=config["log_every"], output=output_dir, device=device)
        return {"rec": restored.get_rec(), "log": log,
                "completed_iterations": stop, "settings": dict(config),
                "optimizer_wall_time_sec": runtime}

    monkeypatch.setattr(runner, "ensure_free_space", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_fia_attempt_folder", lambda *args, **kwargs: {
        "completed_iterations": 25000, "checkpoint_valid": True,
        "reviewed_legacy_completed_checkpoint": True,
        "checkpoint_wall_time_sec": 456.0, "attestation_sha256": "a" * 64,
    })
    monkeypatch.setattr(staged_fia, "run_staged_fia", run_staged_fia)
    result, attempt = runner._run_fia(
        torch.zeros((1, 1)), output=output, stop=25000, source_sha="source",
        identity_sha="identity", device=torch.device("cpu"), attempt_name="naive_fia")
    assert attempt is None
    assert result["optimizer_steps_executed"] == 0
    assert result["reviewed_completed_checkpoint_reused"] is True
    assert result["peak_cuda_allocated_bytes"] is None
    assert TinyDirect.step_calls == 0
    assert core.file_sha256(original_checkpoint) == checkpoint_sha
    assert ledger.read_bytes() == ledger_bytes
    assert core.tensor_sha256(torch.load(rec_path, map_location="cpu", weights_only=True)) == core.tensor_sha256(saved_rec)
    assert staged_fia._save_checkpoint is original_saver


def test_legacy_completed_checkpoint_provenance_rejects_lineage_mismatch(tmp_path, monkeypatch):
    output = tmp_path / "public/bit08/batch00/naive/fia"
    output.mkdir(parents=True)
    monkeypatch.setattr(runner, "PUBLIC_ROOT", tmp_path / "public")
    monkeypatch.setattr(runner, "PROJECT", tmp_path / "project")
    monkeypatch.setattr(runner, "REPAIR_TRANSITION_PATH", tmp_path / "transition.json")
    runner.REPAIR_TRANSITION_PATH.write_text("{}\n", encoding="utf-8")
    config_path = runner.PROJECT / "configs/same80_bits87654_R100_S5k_K3_F25k.json"
    config_path.parent.mkdir(parents=True)
    runner.write_json(config_path, {})
    runner.write_json(runner.PUBLIC_ROOT / "registration.json", {"identity": {"fixture": "identity"}})
    monkeypatch.setattr(runner, "_load_registration", lambda *args, **kwargs: (
        {"packet_count": 25}, {"fixture": "identity"}))
    monkeypatch.setattr(runner, "_load_packet", lambda *args, **kwargs: (
        torch.zeros(1), torch.tensor(1.0), {}))
    monkeypatch.setattr(runner, "_naive_stage_id", lambda *args, **kwargs: "stage-identity")
    transition_calls = []
    monkeypatch.setattr(runner, "_reviewed_repair_transition",
                        lambda saved, current, registration_path: (
                            transition_calls.append(current) or {"status": "VALIDATED"}))
    monkeypatch.setattr(runner, "code_identity", lambda *args, **kwargs: {
        "code_sha256": {"fixture.py": "registered-hash"}})
    monkeypatch.setattr(runner.upstream_naive, "source_hashes", lambda: {"official": "source-code"})

    sources = torch.tensor([[1.0, -2.0], [0.5, -0.25]], dtype=torch.float32)
    sources_path = output.parent / "sources.pt"
    torch.save(sources, sources_path)
    target_sha = core.tensor_sha256(sources.abs().contiguous())
    cpa_path = output.parent / "cpa_result.json"
    runner.write_json(cpa_path, {"sources_file_sha256": core.file_sha256(sources_path),
        "sources_tensor_sha256": core.tensor_sha256(sources), "private_data_used": False})
    checkpoint = output / "fia_checkpoint.pt"
    attack = torch.full((16, 3, 224, 224), 0.5, dtype=torch.float32)
    torch.save({"completed_iterations": 25000, "horizon": 25000,
        "attack_state": {"S_hat": attack,
                         "opt": {"state": {0: {}}, "param_groups": [{"params": [0]}]},
                         "sch": {"T_max": 25000, "last_epoch": 25000}},
        "log": [{"iter": 24999}], "wall_time_sec": 12.0,
        "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(), "rng_cuda": []}, checkpoint)
    ledger_path = output / "naive_fia_attempts.json"
    runner.write_json(ledger_path, {"attempts": [{"attempt": 1, "status": "COMPLETED",
        "completed_steps": 25000, "error": None,
        "checkpoint_sha256": core.file_sha256(checkpoint)}]})
    settings = {"seed": int(runner.FIA["seed"]), "device": "cuda",
        "n_iter": runner.TOTAL_STEPS, "lr": float(runner.FIA["lr"]),
        "tv": float(runner.FIA["tv"]), "fi": float(runner.FIA["fi"]),
        "optimizer": str(runner.FIA["optimizer"]), "scheduler": str(runner.FIA["scheduler"]),
        "log_every": int(runner.FIA["log_every"]),
        "backend": "official_cpa/src/feature_inversion.py:Direct",
        "driver": "staged_fia_v1; fixed full cosine horizon; original Direct.step",
        "cudnn_benchmark": True, "cudnn_deterministic": True}
    runner.write_json(output / "fia_manifest.json", {"fingerprint": {
        "kind": "fia", "input_sha256": target_sha, "settings": settings,
        "source_hashes": {"official": "source-code"}}})

    proof = runner._legacy_fia_provenance(output, source_sha=target_sha,
                                           identity_sha="stage-identity")
    assert proof["checkpoint_sha256"] == core.file_sha256(checkpoint)
    assert proof["completed_iterations"] == 25000
    assert proof["attempt_ledger_sha256"] == core.file_sha256(ledger_path)
    assert transition_calls == [{"fixture.py": "registered-hash"}]
    attestation = runner._write_legacy_fia_attestation(
        output, source_sha=target_sha, identity_sha="stage-identity",
        artifact_name="fia_checkpoint.pt")
    attestation_path = runner._legacy_fia_attestation_path(checkpoint)
    assert attestation["attestation_sha256"] == core.file_sha256(attestation_path)
    assert runner._write_legacy_fia_attestation(
        output, source_sha=target_sha, identity_sha="stage-identity",
        artifact_name="fia_checkpoint.pt")["attestation_sha256"] == attestation["attestation_sha256"]

    bad_ledger = runner.read_json(ledger_path)
    bad_ledger["attempts"][0]["checkpoint_sha256"] = "f" * 64
    runner.write_json(ledger_path, bad_ledger)
    with pytest.raises(core.TaskError, match="completion record"):
        runner._legacy_fia_provenance(output, source_sha=target_sha,
                                      identity_sha="stage-identity")


def test_runner_repair_transition_rejects_unrelated_registered_code_change(tmp_path, monkeypatch):
    import hashlib

    runner_key = Path(runner.__file__).resolve().as_posix()
    other_path = tmp_path / "unchanged.py"
    other_path.write_text("unchanged", encoding="utf-8")
    other_key = other_path.resolve().as_posix()
    saved = {"task_id": runner.TASK_ID, "plan_version": runner.PLAN_VERSION,
             "config_sha256": "config-hash",
             "code_sha256": {runner_key: "a" * 64, other_key: core.file_sha256(other_path)}}
    current_code = {runner_key: "c" * 64, other_key: core.file_sha256(other_path)}
    registration_path = tmp_path / "public/registration.json"
    registration_path.parent.mkdir(parents=True)
    runner.write_json(registration_path, {"identity": saved})
    workspace = tmp_path / "workspace"
    report_rel = (Path(".ai/relay-runs/supervisors") / runner.TASK_ID / "repair_validation.json").as_posix()
    report_path = workspace / report_rel
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"tests":"passed"}\n', encoding="utf-8")
    transition_path = workspace / ".ai/relay-runs/supervisors" / runner.TASK_ID / "reviewed_repair_transition.json"
    transition = {
        "schema": runner.REPAIR_TRANSITION_SCHEMA, "status": "VALIDATED",
        "task_id": runner.TASK_ID, "plan_version": runner.PLAN_VERSION,
        "amendment_id": "CPA_SAME80_SCORE_REPAIR_20260926",
        "authorization_handoff_sha256": runner.REPAIR_HANDOFF_SHA256,
        "registration_identity_sha256": hashlib.sha256(
            json.dumps(saved, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "config_sha256": "config-hash",
        "allowed_code_changes": {runner_key: {
            "before_sha256": "a" * 64, "after_sha256": "c" * 64}},
        "checkpoint_provenance": {"checkpoint_sha256":
            "b40e114e85317563996bad4b8e923c6ed1ed0548a3d0034ac9d54d154c65e6ff"},
        "validation_report": {"path": report_rel,
                              "sha256": core.file_sha256(report_path)},
    }
    runner.write_json(transition_path, transition)
    monkeypatch.setattr(runner, "WORKSPACE", workspace)
    monkeypatch.setattr(runner, "REPAIR_TRANSITION_PATH", transition_path)
    assert runner._reviewed_repair_transition(saved, current_code, registration_path) == transition
    current_code[other_key] = "d" * 64
    with pytest.raises(core.TaskError, match="malformed, or mismatched"):
        runner._reviewed_repair_transition(saved, current_code, registration_path)
