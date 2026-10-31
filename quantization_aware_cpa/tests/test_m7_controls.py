"""CPU checks for the scientific boundary of the no-unmixing controls.

Expanded views represent the required observation shape without allocating a
4096 x 25088 gradient. These tests never invoke FIA or read new M7 results.
"""
from __future__ import annotations

import json
import math

import pytest
import torch

from qacpa import m7_protocol
from qacpa.run_m7_controls import random_prior_targets, raw_observation_targets


@pytest.mark.parametrize("batch_index", [0, 1, 4])
def test_raw_targets_preserve_selected_signed_and_zero_rows(batch_index: int) -> None:
    expected_rows = torch.randperm(
        4096, generator=torch.Generator(device="cpu").manual_seed(batch_index)
    )[:8]
    levels = torch.linspace(-4.0, 4.0, 4096)
    levels[expected_rows[0]] = 0.0
    levels[expected_rows[1]] = -2.0
    levels[expected_rows[2]] = 3.0
    original_levels = levels.clone()
    observation = levels[:, None].expand(4096, 25088).requires_grad_(True)
    assert observation.untyped_storage().nbytes() == 4096 * 4

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13579)
        state_before = torch.random.get_rng_state().clone()
        targets, selected_rows, raw_norms = raw_observation_targets(
            observation, batch_index=batch_index
        )
        assert torch.equal(torch.random.get_rng_state(), state_before)

    assert torch.equal(selected_rows, expected_rows)
    assert torch.equal(levels, original_levels), "target preparation mutated the observation"
    assert targets.shape == (8, 25088)
    assert not targets.requires_grad
    assert bool(torch.isfinite(targets).all())
    assert torch.count_nonzero(targets[0]) == 0, "a sampled zero row must remain present"
    assert bool((targets[1] < 0).all()), "negative observations must not be sign-fixed"
    assert bool((targets[2] > 0).all())
    expected_levels = levels.index_select(0, expected_rows)
    expected_targets = expected_levels.sign()[:, None].expand(8, 25088) / math.sqrt(25088)
    # The unchanged FP32 L2 reduction sums 25,088 entries; compare the analytic
    # norm at its reduction precision, while sign/zero checks above are exact.
    torch.testing.assert_close(targets, expected_targets, rtol=5e-5, atol=1e-8)
    torch.testing.assert_close(raw_norms, expected_levels.abs() * math.sqrt(25088), rtol=5e-5, atol=1e-6)


def test_precision_pairing_uses_the_same_rows_despite_observation_values() -> None:
    # This catches row quality/sign/zero filtering separately from row normalization.
    positive = torch.ones(1, 1).expand(4096, 25088)
    negative = (-torch.ones(1, 1)).expand(4096, 25088)
    fp_targets, fp_rows, _ = raw_observation_targets(positive, batch_index=3)
    q_targets, q_rows, _ = raw_observation_targets(negative, batch_index=3)
    assert torch.equal(fp_rows, q_rows)
    torch.testing.assert_close(fp_targets, -q_targets, rtol=0, atol=0)


def test_raw_targets_reject_transposed_or_reduced_observations() -> None:
    with pytest.raises(ValueError, match="unexpected observation shape"):
        raw_observation_targets(torch.zeros(1, 1).expand(25088, 4096), batch_index=0)
    with pytest.raises(ValueError, match="unexpected observation shape"):
        raw_observation_targets(torch.zeros(8, 4), batch_index=0)


def test_prior_targets_have_an_independent_reproducible_rng() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(98765)
        before = torch.random.get_rng_state().clone()
        first = random_prior_targets()
        assert torch.equal(torch.random.get_rng_state(), before)
        # Arbitrary unrelated work must neither influence nor be influenced by prior sampling.
        torch.rand(113)
        before_second = torch.random.get_rng_state().clone()
        second = random_prior_targets()
        assert torch.equal(torch.random.get_rng_state(), before_second)

    assert first.shape == (40, 25088)
    assert torch.equal(first, second)
    assert bool((first >= 0).all())
    assert bool(torch.isfinite(first).all())
    torch.testing.assert_close(first.norm(dim=1), torch.ones(40), rtol=1e-5, atol=1e-6)
    # Seed/distribution must agree with the locked positive-feature prior.
    raw = torch.rand(40, 25088, generator=torch.Generator(device="cpu").manual_seed(42))
    torch.testing.assert_close(first, raw / raw.norm(dim=1, keepdim=True), rtol=0, atol=0)


def test_locked_protocol_preserves_the_historical_budget_and_control_boundary() -> None:
    protocol, lock = m7_protocol.read_protocol()
    assert protocol["protocol_id"] == lock["protocol_id"]
    expected_fia = {
        "n_iter": 25000, "log_every": 5000, "lr": 0.1, "tv": 0.1, "fi": 1.0,
        "optimizer": "adam", "scheduler": "cosine", "seed": 42,
        "chunk_global_ranges": [[0, 16], [16, 24], [24, 32], [32, 40]],
    }
    assert {key: protocol["fia"][key] for key in expected_fia} == expected_fia
    assert protocol["victim"]["seed"] == 0
    assert protocol["victim"]["batch_size"] == 8
    assert protocol["victim"]["global_indices"] == list(range(40))
    for operation in ("selection_excludes_zero_rows", "sign_flip", "abs", "centering",
                      "whitening", "ica", "unmixing"):
        assert protocol["no_cpa"][operation] is False
    assert protocol["prior_only"]["embedding_seed"] == 42
    assert protocol["prior_only"]["uses_observation_or_true_embedding"] is False
    assert protocol["success"]["alpha"] == 0.05


def test_protocol_change_after_lock_is_rejected(tmp_path, monkeypatch) -> None:
    spec_path = tmp_path / "protocol.json"
    lock_path = tmp_path / "protocol.lock.json"
    spec_path.write_text(json.dumps({"protocol_id": "m7_test", "tv": 0.1}), encoding="utf-8")
    lock_path.write_text(json.dumps({
        "protocol_id": "m7_test", "sha256": m7_protocol.file_sha256(spec_path),
    }), encoding="utf-8")
    monkeypatch.setattr(m7_protocol, "SPEC", spec_path)
    monkeypatch.setattr(m7_protocol, "LOCK", lock_path)
    m7_protocol.read_protocol()
    spec_path.write_text(json.dumps({"protocol_id": "m7_test", "tv": 0.2}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after locking"):
        m7_protocol.read_protocol()
