from pathlib import Path

import torch

from quantization_aware_cpa.qacpa import signed_int4_gdonly_screen3 as gd


def test_registered_backend_is_exact_original_without_fastica():
    config = gd._config()
    assert config["n_components"] == 16
    assert config["cpa_ica_max_iter"] == 25000
    assert config["cpa_ica_optimizer"] == "adam"
    assert config["cpa_ica_scheduler"] == "none"
    assert gd.CPA_CONFIG["cpa_ica_ne"] == 1.0
    assert gd.CPA_CONFIG["cpa_ica_nv"] == 0.13


def test_mocked_original_call_writes_native_and_exact_abs_target(monkeypatch, tmp_path):
    source = torch.arange(16 * gd.FEATURE_DIM, dtype=torch.float32).reshape(16, gd.FEATURE_DIM) / 1000
    seen = {}

    def fake_run_upstream_cpa(**kwargs):
        seen.update(kwargs)
        return {
            "S_hat": source,
            "log": [{"iter": 0, "loss": 1.0}],
            "wall_time_sec": 1.0,
            "setup_wall_time_sec": 0.1,
            "optimizer_wall_time_sec": 0.9,
            "source_hashes": {"fake": "hash"},
            "settings": {"backend": "official_cpa/src/gradient_inversion.py:CocktailPartyAttack",
                         "n_iter": 25000, "device": "cuda"},
        }

    monkeypatch.setattr(gd, "run_upstream_cpa", fake_run_upstream_cpa)
    separation_root = tmp_path / "separation"
    record, native = gd._run_one_separation("center", torch.ones(4096, gd.FEATURE_DIM), separation_root, "input")
    assert seen["config"]["n_components"] == 16
    assert seen["config"]["cpa_ica_max_iter"] == 25000
    assert record["fastica"]["called"] is False
    branch = separation_root / "center"
    torch.save({"S_hat": native, "target_z": native.abs().contiguous()}, branch / "sources.pt")
    record["target_z_sha256"] = gd.tensor_sha256(native.abs())
    record["source_file_sha256"] = gd.file_sha256(branch / "sources.pt")
    gd.write_json(branch / "result.json", record)
    target, _, loaded = gd._source_entry(tmp_path, -1)
    assert torch.equal(target, native.abs())
    assert loaded["fastica"]["called"] is False


def test_public_alignment_changes_only_permutation_and_sign():
    center = torch.randn(16, gd.FEATURE_DIM)
    permutation = torch.tensor([3, 1, 12, 0, 15, 8, 6, 9, 2, 4, 10, 14, 5, 7, 11, 13])
    member = center[permutation].clone()
    aligned, meta = gd.align_preserve_norm(center, member, 0)
    assert torch.allclose(aligned.abs(), center.abs())
    assert meta["row_norm_rescaling"] is False
    assert meta["member_index"] == 0


def test_public_selection_is_member_id_tie_broken():
    scores = []
    for i in range(8):
        scores.append({"member_index": i, "score": {"score": 1.0, "raw_state_unchanged": True},
                       "result": {"candidate_sha256": str(i), "checkpoint_sha256": str(i)}})
    selection = gd._public_selection(scores)
    assert selection["selected_member_indices"] == [0, 1, 2]


def test_center_completed_prefix_candidate_contract(tmp_path):
    branch = tmp_path / "fia_center_25000"
    branch.mkdir()
    pixels = torch.zeros(16, 3, 224, 224)
    torch.save({"pixels": pixels}, branch / "candidate_5000.pt")
    torch.save({"pixels": pixels}, branch / "candidate.pt")
    record = {"status": "COMPLETED", "completed_iterations": 25000,
              "prefix_candidate_sha256": gd.file_sha256(branch / "candidate_5000.pt"),
              "candidate_sha256": gd.file_sha256(branch / "candidate.pt")}
    gd.write_json(branch / "result.json", record)
    checked = gd._candidate(branch, 5000, center=True)
    assert checked["step"] == 5000
