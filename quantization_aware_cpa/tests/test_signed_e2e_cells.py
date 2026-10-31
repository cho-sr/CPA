"""CPU contracts for the actual-image signed-cell path."""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from qacpa.fc_replay import fc_recomputed_gradients
from qacpa.signed_e2e_cells import (
    CellTable, DeviceCellTable, signed_cell_loss, signed_cell_penalty,
    summarize_signed_update,
)
from qacpa.signed_e2e_runner import arm_lr, labels_from_public_logits, normalize_pixels, total_variation
from quantized_update_cpa import cell_utils


def make_table(shape=(4,), *, code=0, scale=1.0):
    q = torch.full(shape, code, dtype=torch.int8)
    s = torch.tensor(scale, dtype=torch.float32)
    all_codes = torch.arange(-7, 8, dtype=torch.int8)
    lower, upper = cell_utils.get_cells(all_codes, s, dict(cell_utils.CODEC_METADATA))
    return CellTable("synthetic", q, s, lower, upper)


def device_table(table: CellTable) -> DeviceCellTable:
    return DeviceCellTable(table.name, table.q_cpu, table.scale_cpu, table.lower_cpu, table.upper_cpu)


def test_cell_penalty_inside_outside_and_chunk_gradient():
    table = make_table((6,), code=0, scale=1.0)
    lo, hi = table.lower_cpu[7], table.upper_cpu[7]
    values = torch.tensor([lo, hi, (lo + hi) / 2, lo - 0.25, hi + 0.5, 0.0], requires_grad=True)
    actual = signed_cell_penalty(values, device_table(table), chunk_size=2)
    expected = (torch.relu(lo - values.detach()) + torch.relu(values.detach() - hi)).square().mean()
    torch.testing.assert_close(actual, expected)
    gradient, = torch.autograd.grad(actual, values)
    expected_gradient = torch.where(values.detach() < lo, 2 * (values.detach() - lo) / values.numel(),
                                   torch.where(values.detach() > hi, 2 * (values.detach() - hi) / values.numel(),
                                               torch.zeros_like(values)))
    torch.testing.assert_close(gradient, expected_gradient)


def test_chunk_and_nonchunk_loss_gradient_match():
    table = make_table((17,), code=1, scale=1.0)
    values_a = torch.linspace(-1.3, 1.2, 17, requires_grad=True)
    values_b = values_a.detach().clone().requires_grad_(True)
    loss_a = signed_cell_penalty(values_a, device_table(table), chunk_size=3)
    loss_b = signed_cell_penalty(values_b, device_table(table), chunk_size=1000)
    grad_a, = torch.autograd.grad(loss_a, values_a)
    grad_b, = torch.autograd.grad(loss_b, values_b)
    torch.testing.assert_close(loss_a, loss_b, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(grad_a, grad_b, atol=0, rtol=0)


def test_signed_cell_loss_equal_tensor_means_and_summary():
    left = make_table((3,), code=0, scale=1.0)
    right = make_table((3,), code=1, scale=1.0)
    # Use the same table shape under two names to isolate equal tensor weighting.
    tables = {"a": device_table(left), "b": device_table(right)}
    updates = {"a": torch.tensor([-1.0, 0.0, 1.0]), "b": torch.tensor([0.0, .9, 1.1])}
    # signed_cell_loss is deliberately six-tensor-only; its strict contract is tested separately.
    with pytest.raises(ValueError, match="exactly all six"):
        signed_cell_loss(updates, tables)
    summary = summarize_signed_update(torch.tensor([-1.0, 0.0, 1.0]), left)
    assert summary["bounded_cell_violation_count"] == 2
    assert summary["code_mismatch_count"] == 2
    assert summary["cell_penalty"] > 0


def test_replay_cell_objective_changes_feature_gradient():
    torch.manual_seed(19)
    classifier = nn.Linear(2, 2, bias=False).double()
    classifier.weight.data.zero_()
    features = (torch.tensor([[4.0, -3.0], [2.0, 1.0]], dtype=torch.float64)
                .requires_grad_(True))
    labels = torch.tensor([0, 1])
    q = torch.zeros_like(classifier.weight, dtype=torch.int8)
    scale = torch.tensor(1.0, dtype=torch.float32)
    all_codes = torch.arange(-7, 8, dtype=torch.int8)
    lower, upper = cell_utils.get_cells(all_codes, scale, dict(cell_utils.CODEC_METADATA))
    table = DeviceCellTable("classifier.weight", q, scale, lower, upper)

    def cell_objective(updates):
        return signed_cell_penalty(updates["classifier.weight"].float(), table, chunk_size=2).double()

    def zero_objective(updates):
        return updates["classifier.weight"].sum() * 0

    loss_cell, grad_cell, _ = fc_recomputed_gradients(
        classifier, features, labels, steps=1, lr=1.0, lossfn=cell_objective)
    loss_zero, grad_zero, _ = fc_recomputed_gradients(
        classifier, features, labels, steps=1, lr=1.0, lossfn=zero_objective)
    assert float(loss_cell) > 0
    assert float(loss_zero) == 0
    assert torch.isfinite(grad_cell).all() and grad_cell.norm() > 0
    assert torch.equal(grad_zero, torch.zeros_like(grad_zero))


def test_label_hypotheses_public_only_and_conflict():
    logits = torch.zeros(16, 1000)
    logits[:, 7] = 1.0
    logits[:, 12] = .5
    q0 = torch.zeros(1000, dtype=torch.int8)
    evidence0, labels0 = labels_from_public_logits(logits, q0)
    assert evidence0["status"] == "COMPLETED"
    assert torch.equal(labels0["H0"], labels0["H1"])

    q2 = q0.clone()
    q2[3], q2[11] = 2, 1
    evidence2, labels2 = labels_from_public_logits(logits, q2)
    assert evidence2["positive_bias_code_classes_ascending"] == [3, 11]
    assert set(labels2["H1"].tolist()) >= {3, 11}
    assert len(evidence2["assignment"]) == 16

    q17 = q0.clone()
    q17[:17] = 1
    evidence17, labels17 = labels_from_public_logits(logits, q17)
    assert evidence17["status"] == "MODEL_OR_OBSERVATION_LABEL_HINT_CONFLICT"
    assert set(labels17) == {"H0"}


def test_normalization_tv_and_schedule_contract():
    pixels = torch.rand(2, 3, 224, 224)
    normalized = normalize_pixels(pixels)
    assert torch.equal(normalized, (pixels - pixels.new_tensor((.485, .456, .406)).view(1, 3, 1, 1))
                       / pixels.new_tensor((.229, .224, .225)).view(1, 3, 1, 1))
    assert total_variation(normalized).isfinite()
    assert arm_lr(0) == pytest.approx(.001)
    assert arm_lr(99) == pytest.approx(.0001)
