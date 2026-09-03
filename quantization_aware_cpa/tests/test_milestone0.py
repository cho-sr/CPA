from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from qacpa.gates import (
    factorization_gate,
    high_bits_convergence_gate,
    oracle_hinge_gate,
    quantizer_invariant_suite,
)
from qacpa.quantizer import quantize


def test_quantizer_invariant_all_required_modes_and_bits() -> None:
    checks = quantizer_invariant_suite(seed=0)
    assert len(checks) == 8
    assert all(check.passed for check in checks)


def test_oracle_hinge_zero_for_synthetic_factorization() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    A_true = torch.randn(17, 5, generator=generator)
    S_true = torch.empty(5, 101).exponential_(generator=generator)
    Y = A_true @ S_true

    for rounding in ("deterministic", "stochastic"):
        qobs = quantize(Y, bits=4, rounding=rounding, generator=generator)
        check = oracle_hinge_gate(A_true, S_true, qobs.Y_hat, qobs.delta)
        assert check.passed
        assert check.details["hinge_loss"] == 0.0


def test_factorization_gate_synthetic() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2)
    A_true = torch.randn(13, 4, generator=generator)
    S_true = torch.empty(4, 77).exponential_(generator=generator)
    Y = A_true @ S_true
    check = factorization_gate(Y, A_true, S_true)
    assert check.passed
    assert check.details["max_abs_err"] == 0.0


def test_bits16_debiased_stage1_matches_naive_solution() -> None:
    pytest.importorskip("sklearn")
    check = high_bits_convergence_gate(seed=3)
    assert check.passed
    assert check.details["relative_l2"] <= 1.0e-3


@pytest.mark.integration
def test_vgg16_imagenet_factorization_matches_autograd() -> None:
    imagenet_root = os.environ.get("QACPA_IMAGENET_ROOT")
    if not imagenet_root:
        pytest.skip("set QACPA_IMAGENET_ROOT to run the VGG16/ImageNet gate")

    from qacpa.victim import (
        collect_victim_batch,
        resolve_device,
    )

    victim = collect_victim_batch(
        data_root=Path(imagenet_root),
        split="val",
        B=2,
        seed=0,
        batch_index=0,
        require_distinct_labels=True,
        device=resolve_device("auto"),
    )
    assert factorization_gate(victim.Y, victim.A_true, victim.S_true).passed
