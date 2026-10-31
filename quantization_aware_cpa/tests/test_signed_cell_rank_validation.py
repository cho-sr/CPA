"""Focused V1 initialization and V3 true-G validation tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from qacpa import signed_cell_rank_evaluation as evaluation


def _initial():
    return {
        "tensor_sha256": "a" * 64,
        "shape": [16, 3, 224, 224],
        "seed": evaluation.SEED,
        "runtime": copy.deepcopy(evaluation.RUNTIME_CONTRACT),
    }


def test_valid_baseline_initialization_preserves_warn_only_true():
    left, right = _initial(), _initial()
    checked = evaluation._validate_initialization_pair(left, right)
    assert checked[0]["runtime"]["deterministic_warn_only"] is True


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("field", ["tensor_sha256", "shape", "seed", "runtime"])
def test_missing_initialization_evidence_on_either_side_rejected(side, field):
    left, right = _initial(), _initial()
    (left if side == "left" else right).pop(field)
    with pytest.raises(evaluation.ProvenanceError, match="missing"):
        evaluation._validate_initialization_pair(left, right)


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(tensor_sha256="z" * 64),
    lambda value: value.update(tensor_sha256=None),
    lambda value: value.update(seed=True),
    lambda value: value["runtime"].update(threads=True),
    lambda value: value["runtime"].update(deterministic_warn_only=False),
])
def test_initialization_type_sha_and_fixed_runtime_guards(mutation):
    value = _initial()
    mutation(value)
    with pytest.raises(evaluation.ProvenanceError):
        evaluation._validate_initialization(value, "synthetic")


def test_true_g_error_positive_norm_and_zero_norm_conventions():
    true_g = torch.tensor([[3., 4.]], dtype=torch.float32)
    zero = torch.zeros_like(true_g)
    positive = evaluation.update_error_metrics(
        zero, true_g, arm="synthetic_sq", estimate_input_sha256="a" * 64,
        delta_convention=evaluation.RAW_DELTA_CONVENTION)
    assert positive["absolute_l2_error"] == pytest.approx(5.0)
    assert positive["reference_l2_norm"] == pytest.approx(5.0)
    assert positive["relative_l2_error"] == pytest.approx(1.0)
    zero_both = evaluation.update_error_metrics(
        zero, zero, arm="synthetic_zero", estimate_input_sha256="b" * 64,
        delta_convention=evaluation.RAW_DELTA_CONVENTION)
    assert zero_both["absolute_l2_error"] == 0.0
    assert zero_both["reference_l2_norm"] == 0.0
    assert zero_both["relative_l2_error"] is None
    assert zero_both["relative_l2_error_status"] == "UNDEFINED_ZERO_REFERENCE_NORM"
    nonzero_against_zero = evaluation.update_error_metrics(
        torch.ones_like(true_g), zero, arm="synthetic_nonzero", estimate_input_sha256="c" * 64,
        delta_convention=evaluation.RAW_DELTA_CONVENTION)
    assert nonzero_against_zero["relative_l2_error"] is None
    assert nonzero_against_zero["absolute_l2_error"] == pytest.approx(2 ** .5)


@pytest.mark.parametrize("estimate,true_g", [
    (torch.ones((1, 2), dtype=torch.float64), torch.ones((1, 2), dtype=torch.float32)),
    (torch.ones((1, 2), dtype=torch.float32), torch.ones((1, 3), dtype=torch.float32)),
    (torch.tensor([[float("nan"), 0.]], dtype=torch.float32), torch.zeros((1, 2), dtype=torch.float32)),
])
def test_true_g_error_rejects_invalid_tensors(estimate, true_g):
    with pytest.raises(evaluation.ProvenanceError):
        evaluation.update_error_metrics(
            estimate, true_g, arm="synthetic", estimate_input_sha256="d" * 64,
            delta_convention=evaluation.RAW_DELTA_CONVENTION)


def test_raw_true_g_schema_binding_rejects_wrong_checkpoint_layer_and_convention(tmp_path):
    path = tmp_path / "raw_updates.pt"
    good = {"updates": {evaluation.ATTACK_LAYER: torch.zeros((2, 2), dtype=torch.float32)},
            "delta_convention": evaluation.RAW_DELTA_CONVENTION,
            "checkpoint_sha256": evaluation.CHECKPOINT_SHA}
    evaluation.torch.save(good, path)
    assert evaluation._load_true_update(path, expected_shape=(2, 2))[0].shape == (2, 2)
    for mutation in (
        lambda value: value.update(checkpoint_sha256="0" * 64),
        lambda value: value.update(delta_convention="wrong"),
        lambda value: value.update(updates={"other": torch.zeros((2, 2), dtype=torch.float32)}),
        lambda value: value.update(updates={evaluation.ATTACK_LAYER: torch.zeros((2, 3), dtype=torch.float32)}),
    ):
        changed = copy.deepcopy(good)
        mutation(changed)
        evaluation.torch.save(changed, path)
        with pytest.raises(evaluation.ProvenanceError):
            evaluation._load_true_update(path, expected_shape=(2, 2))
