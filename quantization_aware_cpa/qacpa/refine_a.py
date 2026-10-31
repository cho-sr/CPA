from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch

from .stage2 import Stage2Result, solve_stage2_consistency


@dataclass
class RefineAResult:
    A_star: torch.Tensor
    S_star: torch.Tensor
    stage2_result: Stage2Result
    a_step_logs: list[dict[str, float | bool | int]]
    n_a_steps_requested: int
    n_a_steps_accepted: int
    a_step_size_scale: float
    a_chunk_size: int
    wall_time_sec: float


def solve_stage2_with_A_refinement(
    *,
    A: torch.Tensor,
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
    S_init: torch.Tensor,
    max_iter: int,
    chunk_size: int,
    device: torch.device,
    lam_R: float = 0.0,
    mu: float = 1.0e3,
    eps: float = 1.0e-3,
    stop_tolerance: float = 1.0e-10,
    check_every: int = 25,
    phase1_max_iter: int | None = None,
    phase2_max_iter: int | None = None,
    phase1_acceleration: bool = True,
    n_a_steps: int = 5,
    a_step_size_scale: float = 1.0,
    a_chunk_size: int | None = None,
) -> RefineAResult:
    start = time.perf_counter()
    A_current = A.detach().cpu().float().contiguous()
    Y_cpu = Y_hat.detach().cpu().float().contiguous()
    delta_cpu = delta.detach().cpu().float().clamp_min(1.0e-12).contiguous()
    S_current = S_init.detach().cpu().float().clamp_min(0.0).contiguous()
    logs: list[dict[str, float | bool | int]] = []
    accepted = 0

    final_stage2: Stage2Result | None = None
    for step in range(int(n_a_steps)):
        final_stage2 = solve_stage2_consistency(
            A=A_current,
            Y_hat=Y_cpu,
            delta=delta_cpu,
            S_init=S_current,
            max_iter=int(max_iter),
            chunk_size=int(chunk_size),
            device=device,
            lam_R=float(lam_R),
            mu=float(mu),
            eps=float(eps),
            stop_tolerance=float(stop_tolerance),
            check_every=int(check_every),
            phase1_max_iter=phase1_max_iter,
            phase2_max_iter=phase2_max_iter,
            phase1_acceleration=bool(phase1_acceleration),
        )
        S_current = final_stage2.S_star.detach().cpu().float().clamp_min(0.0).contiguous()
        A_next, step_log = refine_A_hinge_step(
            A=A_current,
            S=S_current,
            Y_hat=Y_cpu,
            delta=delta_cpu,
            device=device,
            chunk_size=int(chunk_size if a_chunk_size is None else a_chunk_size),
            step_size_scale=float(a_step_size_scale),
        )
        if bool(step_log["accepted"]):
            A_current = A_next
            accepted += 1
        step_log["outer_step"] = int(step)
        logs.append(step_log)

    final_stage2 = solve_stage2_consistency(
        A=A_current,
        Y_hat=Y_cpu,
        delta=delta_cpu,
        S_init=S_current,
        max_iter=int(max_iter),
        chunk_size=int(chunk_size),
        device=device,
        lam_R=float(lam_R),
        mu=float(mu),
        eps=float(eps),
        stop_tolerance=float(stop_tolerance),
        check_every=int(check_every),
        phase1_max_iter=phase1_max_iter,
        phase2_max_iter=phase2_max_iter,
        phase1_acceleration=bool(phase1_acceleration),
    )
    return RefineAResult(
        A_star=A_current.contiguous(),
        S_star=final_stage2.S_star.detach().cpu().float().contiguous(),
        stage2_result=final_stage2,
        a_step_logs=logs,
        n_a_steps_requested=int(n_a_steps),
        n_a_steps_accepted=int(accepted),
        a_step_size_scale=float(a_step_size_scale),
        a_chunk_size=int(chunk_size if a_chunk_size is None else a_chunk_size),
        wall_time_sec=float(time.perf_counter() - start),
    )


def refine_A_hinge_step(
    *,
    A: torch.Tensor,
    S: torch.Tensor,
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
    device: torch.device,
    chunk_size: int,
    step_size_scale: float,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    A_cpu = A.detach().cpu().float().contiguous()
    S_cpu = S.detach().cpu().float().contiguous()
    Y_cpu = Y_hat.detach().cpu().float().contiguous()
    delta_cpu = delta.detach().cpu().float().clamp_min(1.0e-12).contiguous()

    A_scaled = (A_cpu / delta_cpu[:, None]).to(device=device, dtype=torch.float32)
    S_dev = S_cpu.to(device=device, dtype=torch.float32)
    Y_scaled = (Y_cpu / delta_cpu[:, None]).to(device=device, dtype=torch.float32)
    spectral = torch.linalg.matrix_norm(S_dev, ord=2)
    d = int(Y_cpu.shape[1])
    lipschitz_t = (2.0 * spectral.square() / float(d)).clamp_min(1.0e-12)
    eta = float(step_size_scale) / lipschitz_t
    grad = torch.zeros_like(A_scaled)

    H_before_sum = torch.zeros((), device=device, dtype=torch.float32)
    active_count = 0
    total_count = 0
    for start_col in range(0, d, int(chunk_size)):
        end_col = min(start_col + int(chunk_size), d)
        S_chunk = S_dev[:, start_col:end_col]
        residual = A_scaled.matmul(S_chunk) - Y_scaled[:, start_col:end_col]
        hinge = (residual.abs() - 0.5).clamp_min(0.0)
        active = hinge * residual.sign()
        grad.add_(2.0 * active.matmul(S_chunk.T) / float(d))
        H_before_sum.add_(hinge.square().sum())
        active_count += int((hinge > 0.0).sum().detach().cpu())
        total_count += int(hinge.numel())

    A_scaled_next = A_scaled - eta * grad
    H_after_sum = torch.zeros((), device=device, dtype=torch.float32)
    for start_col in range(0, d, int(chunk_size)):
        end_col = min(start_col + int(chunk_size), d)
        residual = A_scaled_next.matmul(S_dev[:, start_col:end_col]) - Y_scaled[:, start_col:end_col]
        hinge = (residual.abs() - 0.5).clamp_min(0.0)
        H_after_sum.add_(hinge.square().sum())

    H_before = float((H_before_sum / float(d)).detach().cpu())
    H_after = float((H_after_sum / float(d)).detach().cpu())
    accepted = H_after <= H_before
    A_out_scaled = A_scaled_next if accepted else A_scaled
    A_out = (A_out_scaled.detach().cpu() * delta_cpu[:, None]).float().contiguous()
    return A_out, {
        "H_before": H_before,
        "H_after": H_after,
        "accepted": bool(accepted),
        "eta": float(eta.detach().cpu()),
        "lipschitz": float(lipschitz_t.detach().cpu()),
        "grad_norm": float(grad.norm().detach().cpu()),
        "A_scaled_norm": float(A_scaled.norm().detach().cpu()),
        "active_fraction": float(active_count / max(1, total_count)),
    }
