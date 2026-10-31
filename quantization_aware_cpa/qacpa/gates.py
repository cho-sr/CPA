from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .errors import GateFailure
from .linalg import relative_l2
from .quantizer import assert_cell_membership, hinge_loss, quantize
from .stage1 import estimate_mixing_and_sources


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    details: dict[str, Any]


def quantizer_invariant_suite(*, seed: int = 0) -> list[GateCheck]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    checks: list[GateCheck] = []
    base = torch.randn(33, 257, generator=generator)
    base[0].zero_()
    base[1, 0] = 100.0
    base[1, 1] = -100.0

    for bits in (2, 3, 4, 8):
        for rounding in ("deterministic", "stochastic"):
            qgen = torch.Generator(device="cpu")
            qgen.manual_seed(int(seed) + 1000 * bits + (1 if rounding == "stochastic" else 0))
            qobs = quantize(base, bits=bits, rounding=rounding, generator=qgen)
            assert_cell_membership(base, qobs.Y_hat, qobs.delta)
            max_over = float(
                ((base - qobs.Y_hat).abs() - qobs.delta[:, None] / 2.0).max().cpu()
            )
            checks.append(
                GateCheck(
                    name=f"quantizer_invariant_bits{bits}_{rounding}",
                    passed=True,
                    details={"bits": bits, "rounding": rounding, "max_over_cell": max_over},
                )
            )
    return checks


def oracle_hinge_gate(
    A_true: torch.Tensor,
    S_true: torch.Tensor,
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
) -> GateCheck:
    loss = float(hinge_loss(A_true, S_true, Y_hat, delta).detach().cpu())
    max_hinge = float(
        (
            (A_true.matmul(S_true) - Y_hat).abs()
            - delta[:, None] / 2.0
        )
        .clamp_min(0.0)
        .max()
        .detach()
        .cpu()
    )
    if max_hinge > 1.0e-6:
        raise GateFailure(f"oracle hinge loss must be exactly 0, got {loss:.12g}")
    return GateCheck("oracle_hinge_zero", True, {"hinge_loss": 0.0, "raw_hinge_loss": loss, "max_hinge": max_hinge})


def factorization_gate(Y: torch.Tensor, A_true: torch.Tensor, S_true: torch.Tensor) -> GateCheck:
    product = A_true.matmul(S_true)
    max_abs = float((Y - product).abs().max().detach().cpu())
    rel = relative_l2(product, Y)
    if not torch.allclose(Y, product, rtol=1.0e-4, atol=1.0e-6):
        raise GateFailure(f"Y_true != A_true @ S_true: max_abs_err={max_abs:.9g}, rel_l2={rel:.9g}")
    return GateCheck("factorization_matches_autograd", True, {"max_abs_err": max_abs, "rel_l2": rel})


def high_bits_convergence_gate(*, seed: int = 0) -> GateCheck:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    B, N, d = 4, 24, 2048
    A = torch.randn(N, B, generator=generator)
    S = torch.empty(B, d).exponential_(generator=generator)
    Y = A.matmul(S)

    qobs = quantize(Y, bits=16, rounding="stochastic", generator=generator)
    naive = estimate_mixing_and_sources(
        qobs.Y_hat,
        qobs.delta,
        n_components=B,
        rounding="stochastic",
        debias=False,
        seed=seed,
        max_iter=500,
        tol=1.0e-5,
        ica_method="cpa",
        cpa_ica_max_iter=500,
    )
    ours = estimate_mixing_and_sources(
        qobs.Y_hat,
        qobs.delta,
        n_components=B,
        rounding="stochastic",
        debias=True,
        seed=seed,
        max_iter=500,
        tol=1.0e-5,
        ica_method="cpa",
        cpa_ica_max_iter=500,
    )
    rel = relative_l2(ours.Y_recon, naive.Y_recon)
    if rel > 1.0e-3:
        raise GateFailure(f"bits->16 debiased stage1 diverged from naive: rel_l2={rel:.9g}")
    return GateCheck(
        "bits16_debiased_stage1_matches_naive",
        True,
        {
            "relative_l2": rel,
            "whitened_cov_offdiag_abs_max": ours.whitened_cov_offdiag_abs_max,
            "note": "Stage 1 convergence proxy with stochastic bits=16 and diagonal debias.",
        },
    )


def run_static_milestone0_gates(*, seed: int = 0) -> list[GateCheck]:
    checks = []
    checks.extend(quantizer_invariant_suite(seed=seed))
    checks.append(high_bits_convergence_gate(seed=seed))
    return checks
