from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from .quantizer import consistency_rate, mean_hinge


@dataclass
class Stage2Result:
    S_star: torch.Tensor
    per_column_hinge_mean: torch.Tensor
    per_column_squared_hinge_mean: torch.Tensor
    initial_consistency_rate: float
    final_consistency_rate: float
    initial_residual_violation: float
    final_residual_violation: float
    lipschitz: float
    step_size: float
    max_iter: int
    chunk_size: int
    stop_tolerance: float
    check_every: int
    chunk_iterations: list[int]
    wall_time_sec: float


def oracle_pinv_initialization(A_true: torch.Tensor, Y_hat: torch.Tensor) -> torch.Tensor:
    return torch.linalg.pinv(A_true.detach().cpu().float()).matmul(
        Y_hat.detach().cpu().float()
    ).clamp_min(0.0)


def solve_stage2_consistency(
    *,
    A: torch.Tensor,
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
    S_init: torch.Tensor,
    max_iter: int,
    chunk_size: int,
    device: torch.device,
    stop_tolerance: float = 1.0e-10,
    check_every: int = 25,
) -> Stage2Result:
    if A.ndim != 2 or Y_hat.ndim != 2 or S_init.ndim != 2:
        raise ValueError("A, Y_hat, and S_init must be rank-2 tensors")
    if A.shape[0] != Y_hat.shape[0]:
        raise ValueError(f"A/Y row mismatch: {tuple(A.shape)} vs {tuple(Y_hat.shape)}")
    if A.shape[1] != S_init.shape[0] or Y_hat.shape[1] != S_init.shape[1]:
        raise ValueError(f"incompatible shapes: A={tuple(A.shape)}, Y={tuple(Y_hat.shape)}, S={tuple(S_init.shape)}")

    start = time.perf_counter()
    A_dev = A.detach().to(device=device, dtype=torch.float32)
    delta_dev = delta.detach().to(device=device, dtype=torch.float32)
    half_delta = delta_dev[:, None] / 2.0
    lipschitz_t = 2.0 * torch.linalg.matrix_norm(A_dev, ord=2) ** 2
    lipschitz = float(lipschitz_t.detach().cpu())
    if lipschitz <= 0.0:
        raise ValueError("Stage-2 Lipschitz constant must be positive")
    eta = float((1.0 / lipschitz_t).detach().cpu())

    S_cpu = torch.empty_like(S_init.detach().cpu().float())
    hinge_mean_cpu = torch.empty(Y_hat.shape[1], dtype=torch.float32)
    hinge_sq_mean_cpu = torch.empty(Y_hat.shape[1], dtype=torch.float32)
    chunk_iterations: list[int] = []
    check_every = max(1, int(check_every))

    initial_consistency = consistency_rate(A.detach().cpu().float(), S_init.detach().cpu().float(), Y_hat, delta)
    initial_violation = mean_hinge(A.detach().cpu().float(), S_init.detach().cpu().float(), Y_hat, delta)

    with torch.no_grad():
        for start_col in range(0, Y_hat.shape[1], int(chunk_size)):
            end_col = min(start_col + int(chunk_size), Y_hat.shape[1])
            Y_chunk = Y_hat[:, start_col:end_col].detach().to(device=device, dtype=torch.float32)
            S_chunk = S_init[:, start_col:end_col].detach().to(device=device, dtype=torch.float32).clamp_min(0.0)
            iterations_run = 0

            for iteration in range(int(max_iter)):
                R = A_dev.matmul(S_chunk) - Y_chunk
                V = (R.abs() - half_delta).clamp_min(0.0) * R.sign()
                G = 2.0 * A_dev.T.matmul(V)
                S_chunk = (S_chunk - float(eta) * G).clamp_min(0.0)
                iterations_run = iteration + 1

                if iterations_run % check_every == 0 or iterations_run == int(max_iter):
                    residual = (A_dev.matmul(S_chunk) - Y_chunk).abs() - half_delta
                    max_hinge_sq = float(residual.clamp_min(0.0).square().mean(dim=0).max().detach().cpu())
                    if max_hinge_sq <= float(stop_tolerance):
                        break

            residual = (A_dev.matmul(S_chunk) - Y_chunk).abs() - half_delta
            hinge = residual.clamp_min(0.0)
            S_cpu[:, start_col:end_col] = S_chunk.detach().cpu()
            hinge_mean_cpu[start_col:end_col] = hinge.mean(dim=0).detach().cpu()
            hinge_sq_mean_cpu[start_col:end_col] = hinge.square().mean(dim=0).detach().cpu()
            chunk_iterations.append(iterations_run)

    final_consistency = consistency_rate(A.detach().cpu().float(), S_cpu, Y_hat, delta)
    final_violation = mean_hinge(A.detach().cpu().float(), S_cpu, Y_hat, delta)
    return Stage2Result(
        S_star=S_cpu.contiguous(),
        per_column_hinge_mean=hinge_mean_cpu,
        per_column_squared_hinge_mean=hinge_sq_mean_cpu,
        initial_consistency_rate=initial_consistency,
        final_consistency_rate=final_consistency,
        initial_residual_violation=initial_violation,
        final_residual_violation=final_violation,
        lipschitz=lipschitz,
        step_size=eta,
        max_iter=int(max_iter),
        chunk_size=int(chunk_size),
        stop_tolerance=float(stop_tolerance),
        check_every=check_every,
        chunk_iterations=chunk_iterations,
        wall_time_sec=time.perf_counter() - start,
    )
