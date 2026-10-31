from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from .quantizer import consistency_rate, mean_hinge


@dataclass
class Stage2Result:
    S_phase1: torch.Tensor
    S_star: torch.Tensor
    per_column_hinge_mean: torch.Tensor
    per_column_squared_hinge_mean: torch.Tensor
    per_column_H_initial: torch.Tensor
    per_column_H_phase1: torch.Tensor
    per_column_H_final: torch.Tensor
    per_column_l1_raw_phase1: torch.Tensor
    per_column_l1_raw_final: torch.Tensor
    per_column_l1_normalized_phase1: torch.Tensor
    per_column_l1_normalized_final: torch.Tensor
    initial_consistency_rate: float
    phase1_consistency_rate: float
    final_consistency_rate: float
    initial_residual_violation: float
    phase1_residual_violation: float
    final_residual_violation: float
    initial_l1_mean: float
    phase1_l1_mean: float
    final_l1_mean: float
    initial_l1_normalized_mean: float
    phase1_l1_normalized_mean: float
    final_l1_normalized_mean: float
    phase1_H_mean: float
    phase1_H_max: float
    final_H_mean: float
    final_H_max: float
    phase2_excess_mean: float
    phase2_excess_max: float
    initial_objective: float
    phase1_objective: float
    final_objective: float
    lam_R: float
    mu: float
    eps: float
    lipschitz: float
    step_size: float
    max_iter: int
    phase1_max_iter: int
    phase2_max_iter: int
    phase1_acceleration: bool
    chunk_size: int
    stop_tolerance: float
    check_every: int
    phase1_chunk_iterations: list[int]
    phase2_chunk_iterations: list[int]
    phase2_guard_reject_fraction: float
    input_guard_reject_fraction: float
    s_scale_min: float
    s_scale_median: float
    s_scale_max: float
    row_update_scale_min: float
    row_update_scale_median: float
    row_update_scale_max: float
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
    lam_R: float = 0.0,
    mu: float = 1.0e3,
    eps: float = 1.0e-3,
    stop_tolerance: float = 1.0e-10,
    check_every: int = 25,
    phase1_max_iter: int | None = None,
    phase2_max_iter: int | None = None,
    phase1_acceleration: bool = True,
    row_update_scale: torch.Tensor | None = None,
) -> Stage2Result:
    if A.ndim != 2 or Y_hat.ndim != 2 or S_init.ndim != 2:
        raise ValueError("A, Y_hat, and S_init must be rank-2 tensors")
    if A.shape[0] != Y_hat.shape[0]:
        raise ValueError(f"A/Y row mismatch: {tuple(A.shape)} vs {tuple(Y_hat.shape)}")
    if A.shape[1] != S_init.shape[0] or Y_hat.shape[1] != S_init.shape[1]:
        raise ValueError(
            f"incompatible shapes: A={tuple(A.shape)}, Y={tuple(Y_hat.shape)}, S={tuple(S_init.shape)}"
        )
    if delta.numel() != A.shape[0]:
        raise ValueError(f"delta must have one value per row: A={tuple(A.shape)}, delta={tuple(delta.shape)}")

    start = time.perf_counter()
    max_iter = int(max_iter)
    phase1_max_iter = int(max_iter if phase1_max_iter is None else phase1_max_iter)
    phase2_max_iter = int(max_iter if phase2_max_iter is None else phase2_max_iter)
    chunk_size = int(chunk_size)
    check_every = max(1, int(check_every))
    lam_R = float(lam_R)
    mu = float(mu)
    eps = float(eps)

    A_cpu = A.detach().cpu().float()
    Y_cpu = Y_hat.detach().cpu().float()
    delta_cpu = delta.detach().cpu().float().clamp_min(1.0e-12)
    S_init_cpu = S_init.detach().cpu().float().clamp_min(0.0)
    s_scale_cpu = S_init_cpu.abs().amax(dim=0).clamp_min(1.0e-12)
    if row_update_scale is None:
        row_update_scale_cpu = torch.ones(A_cpu.shape[1], dtype=torch.float32)
    else:
        row_update_scale_cpu = row_update_scale.detach().cpu().float().flatten()
        if row_update_scale_cpu.numel() != A_cpu.shape[1]:
            raise ValueError(
                "row_update_scale must have one value per source row: "
                f"A={tuple(A_cpu.shape)}, row_update_scale={tuple(row_update_scale_cpu.shape)}"
            )
        if bool((row_update_scale_cpu < 0.0).any()) or bool((row_update_scale_cpu > 1.0).any()):
            raise ValueError("row_update_scale values must lie in [0, 1]")

    A_dev = A_cpu.to(device=device, dtype=torch.float32)
    delta_dev = delta_cpu.to(device=device, dtype=torch.float32)
    row_update_scale_dev = row_update_scale_cpu.to(device=device, dtype=torch.float32)
    A_scaled = A_dev / delta_dev[:, None]
    spectral_norm = torch.linalg.matrix_norm(A_scaled, ord=2)
    scale_max = s_scale_cpu.max().to(device=device)
    lipschitz_t = 2.0 * (spectral_norm * scale_max) ** 2
    lipschitz = float(lipschitz_t.detach().cpu())
    if lipschitz <= 0.0:
        raise ValueError("Stage-2 Lipschitz constant must be positive")
    eta = float((1.0 / lipschitz_t).detach().cpu())

    S_phase1_cpu = torch.empty_like(S_init_cpu)
    S_final_cpu = torch.empty_like(S_init_cpu)
    H_initial_cpu = torch.empty(Y_cpu.shape[1], dtype=torch.float32)
    H_phase1_cpu = torch.empty(Y_cpu.shape[1], dtype=torch.float32)
    H_final_cpu = torch.empty(Y_cpu.shape[1], dtype=torch.float32)
    hinge_mean_cpu = torch.empty(Y_cpu.shape[1], dtype=torch.float32)
    hinge_sq_mean_cpu = torch.empty(Y_cpu.shape[1], dtype=torch.float32)
    l1_raw_phase1_cpu = torch.empty(Y_cpu.shape[1], dtype=torch.float32)
    l1_raw_final_cpu = torch.empty(Y_cpu.shape[1], dtype=torch.float32)
    l1_norm_phase1_cpu = torch.empty(Y_cpu.shape[1], dtype=torch.float32)
    l1_norm_final_cpu = torch.empty(Y_cpu.shape[1], dtype=torch.float32)
    phase2_excess_cpu = torch.empty(Y_cpu.shape[1], dtype=torch.float32)
    phase2_guard_rejects = 0
    phase2_guard_total = 0
    input_guard_rejects = 0
    input_guard_total = 0
    phase1_chunk_iterations: list[int] = []
    phase2_chunk_iterations: list[int] = []

    initial_consistency = consistency_rate(A_cpu, S_init_cpu, Y_cpu, delta_cpu)
    initial_violation = mean_hinge(A_cpu, S_init_cpu, Y_cpu, delta_cpu)
    initial_l1_mean = float(S_init_cpu.abs().mean())
    initial_l1_norm_mean = float((S_init_cpu / s_scale_cpu[None, :]).abs().mean())

    with torch.no_grad():
        for start_col in range(0, Y_cpu.shape[1], chunk_size):
            end_col = min(start_col + chunk_size, Y_cpu.shape[1])
            Y_scaled_chunk = (
                Y_cpu[:, start_col:end_col].to(device=device, dtype=torch.float32)
                / delta_dev[:, None]
            )
            scale_chunk = s_scale_cpu[start_col:end_col].to(device=device, dtype=torch.float32)
            U_chunk = (
                S_init_cpu[:, start_col:end_col].to(device=device, dtype=torch.float32)
                / scale_chunk[None, :]
            ).clamp_min(0.0)
            S_init_chunk = U_chunk * scale_chunk[None, :]
            H_initial, _, _ = _dimensionless_hinge_stats(A_scaled, Y_scaled_chunk, S_init_chunk)

            phase1_iterations = _run_phase1(
                U_chunk=U_chunk,
                scale_chunk=scale_chunk,
                A_scaled=A_scaled,
                Y_scaled_chunk=Y_scaled_chunk,
                eta=eta,
                max_iter=phase1_max_iter,
                check_every=check_every,
                stop_tolerance=float(stop_tolerance),
                acceleration=bool(phase1_acceleration),
                row_update_scale=row_update_scale_dev,
            )
            S_phase1 = U_chunk * scale_chunk[None, :]
            H_phase1, _, _ = _dimensionless_hinge_stats(A_scaled, Y_scaled_chunk, S_phase1)
            h0 = H_phase1.detach().clone()

            phase2_iterations, guard_rejects, guard_total = _run_phase2(
                U_chunk=U_chunk,
                scale_chunk=scale_chunk,
                A_scaled=A_scaled,
                Y_scaled_chunk=Y_scaled_chunk,
                h0=h0,
                eta=eta,
                lam_R=lam_R,
                mu=mu,
                eps=eps,
                max_iter=phase2_max_iter,
                row_update_scale=row_update_scale_dev,
            )
            phase2_guard_rejects += guard_rejects
            phase2_guard_total += guard_total
            S_final = U_chunk * scale_chunk[None, :]
            H_final, hinge_final, _ = _dimensionless_hinge_stats(A_scaled, Y_scaled_chunk, S_final)
            fallback_to_input = H_final > H_initial
            if bool(fallback_to_input.any()):
                S_final = torch.where(fallback_to_input[None, :], S_init_chunk, S_final)
                H_final, hinge_final, _ = _dimensionless_hinge_stats(A_scaled, Y_scaled_chunk, S_final)
                U_chunk.copy_(S_final / scale_chunk[None, :])
            input_guard_rejects += int(fallback_to_input.sum().detach().cpu())
            input_guard_total += int(fallback_to_input.numel())
            excess_final = (H_final - h0 - eps).clamp_min(0.0)

            S_phase1_cpu[:, start_col:end_col] = S_phase1.detach().cpu()
            S_final_cpu[:, start_col:end_col] = S_final.detach().cpu()
            H_initial_cpu[start_col:end_col] = H_initial.detach().cpu()
            H_phase1_cpu[start_col:end_col] = H_phase1.detach().cpu()
            H_final_cpu[start_col:end_col] = H_final.detach().cpu()
            hinge_mean_cpu[start_col:end_col] = hinge_final.mean(dim=0).detach().cpu()
            hinge_sq_mean_cpu[start_col:end_col] = hinge_final.square().mean(dim=0).detach().cpu()
            l1_raw_phase1_cpu[start_col:end_col] = S_phase1.abs().sum(dim=0).detach().cpu()
            l1_raw_final_cpu[start_col:end_col] = S_final.abs().sum(dim=0).detach().cpu()
            l1_norm_phase1_cpu[start_col:end_col] = (S_phase1 / scale_chunk[None, :]).abs().sum(dim=0).detach().cpu()
            l1_norm_final_cpu[start_col:end_col] = (S_final / scale_chunk[None, :]).abs().sum(dim=0).detach().cpu()
            phase2_excess_cpu[start_col:end_col] = excess_final.detach().cpu()
            phase1_chunk_iterations.append(phase1_iterations)
            phase2_chunk_iterations.append(phase2_iterations)

    phase1_consistency = consistency_rate(A_cpu, S_phase1_cpu, Y_cpu, delta_cpu)
    final_consistency = consistency_rate(A_cpu, S_final_cpu, Y_cpu, delta_cpu)
    phase1_violation = mean_hinge(A_cpu, S_phase1_cpu, Y_cpu, delta_cpu)
    final_violation = mean_hinge(A_cpu, S_final_cpu, Y_cpu, delta_cpu)
    phase1_l1_mean = float(S_phase1_cpu.abs().mean())
    final_l1_mean = float(S_final_cpu.abs().mean())
    phase1_l1_norm_mean = float((S_phase1_cpu / s_scale_cpu[None, :]).abs().mean())
    final_l1_norm_mean = float((S_final_cpu / s_scale_cpu[None, :]).abs().mean())
    phase1_H_mean = float(H_phase1_cpu.mean())
    phase1_H_max = float(H_phase1_cpu.max())
    final_H_mean = float(H_final_cpu.mean())
    final_H_max = float(H_final_cpu.max())
    phase2_excess_mean = float(phase2_excess_cpu.mean())
    phase2_excess_max = float(phase2_excess_cpu.max())

    return Stage2Result(
        S_phase1=S_phase1_cpu.contiguous(),
        S_star=S_final_cpu.contiguous(),
        per_column_hinge_mean=hinge_mean_cpu,
        per_column_squared_hinge_mean=hinge_sq_mean_cpu,
        per_column_H_initial=H_initial_cpu,
        per_column_H_phase1=H_phase1_cpu,
        per_column_H_final=H_final_cpu,
        per_column_l1_raw_phase1=l1_raw_phase1_cpu,
        per_column_l1_raw_final=l1_raw_final_cpu,
        per_column_l1_normalized_phase1=l1_norm_phase1_cpu,
        per_column_l1_normalized_final=l1_norm_final_cpu,
        initial_consistency_rate=initial_consistency,
        phase1_consistency_rate=phase1_consistency,
        final_consistency_rate=final_consistency,
        initial_residual_violation=initial_violation,
        phase1_residual_violation=phase1_violation,
        final_residual_violation=final_violation,
        initial_l1_mean=initial_l1_mean,
        phase1_l1_mean=phase1_l1_mean,
        final_l1_mean=final_l1_mean,
        initial_l1_normalized_mean=initial_l1_norm_mean,
        phase1_l1_normalized_mean=phase1_l1_norm_mean,
        final_l1_normalized_mean=final_l1_norm_mean,
        phase1_H_mean=phase1_H_mean,
        phase1_H_max=phase1_H_max,
        final_H_mean=final_H_mean,
        final_H_max=final_H_max,
        phase2_excess_mean=phase2_excess_mean,
        phase2_excess_max=phase2_excess_max,
        initial_objective=float("nan"),
        phase1_objective=float(phase1_H_mean),
        final_objective=float(lam_R * final_l1_norm_mean + mu * phase2_excess_cpu.square().mean().item()),
        lam_R=lam_R,
        mu=mu,
        eps=eps,
        lipschitz=lipschitz,
        step_size=eta,
        max_iter=max_iter,
        phase1_max_iter=phase1_max_iter,
        phase2_max_iter=phase2_max_iter,
        phase1_acceleration=bool(phase1_acceleration),
        chunk_size=chunk_size,
        stop_tolerance=float(stop_tolerance),
        check_every=check_every,
        phase1_chunk_iterations=phase1_chunk_iterations,
        phase2_chunk_iterations=phase2_chunk_iterations,
        phase2_guard_reject_fraction=float(phase2_guard_rejects / max(1, phase2_guard_total)),
        input_guard_reject_fraction=float(input_guard_rejects / max(1, input_guard_total)),
        s_scale_min=float(s_scale_cpu.min()),
        s_scale_median=float(s_scale_cpu.median()),
        s_scale_max=float(s_scale_cpu.max()),
        row_update_scale_min=float(row_update_scale_cpu.min()),
        row_update_scale_median=float(row_update_scale_cpu.median()),
        row_update_scale_max=float(row_update_scale_cpu.max()),
        wall_time_sec=time.perf_counter() - start,
    )


def _run_phase1(
    *,
    U_chunk: torch.Tensor,
    scale_chunk: torch.Tensor,
    A_scaled: torch.Tensor,
    Y_scaled_chunk: torch.Tensor,
    eta: float,
    max_iter: int,
    check_every: int,
    stop_tolerance: float,
    acceleration: bool,
    row_update_scale: torch.Tensor,
) -> int:
    iterations_run = 0
    if bool(acceleration):
        z_chunk = U_chunk.clone()
        t_value = 1.0
    else:
        z_chunk = U_chunk
        t_value = 1.0
    for iteration in range(int(max_iter)):
        S_chunk = z_chunk * scale_chunk[None, :]
        H_col, hinge, scaled_residual = _dimensionless_hinge_stats(A_scaled, Y_scaled_chunk, S_chunk)
        active = hinge * scaled_residual.sign()
        grad_s = 2.0 * A_scaled.T.matmul(active)
        grad_u = grad_s * scale_chunk[None, :]
        grad_u = grad_u * row_update_scale[:, None]
        next_u = (z_chunk - float(eta) * grad_u).clamp_min(0.0)
        if bool(acceleration):
            next_t = 0.5 * (1.0 + (1.0 + 4.0 * t_value * t_value) ** 0.5)
            momentum = (t_value - 1.0) / next_t
            z_chunk = (next_u + momentum * (next_u - U_chunk)).clamp_min(0.0)
            U_chunk.copy_(next_u)
            t_value = next_t
        else:
            U_chunk.copy_(next_u)
        iterations_run = iteration + 1
        if iterations_run % check_every == 0 or iterations_run == int(max_iter):
            if float(H_col.max().detach().cpu()) <= float(stop_tolerance):
                break
    return iterations_run


def _run_phase2(
    *,
    U_chunk: torch.Tensor,
    scale_chunk: torch.Tensor,
    A_scaled: torch.Tensor,
    Y_scaled_chunk: torch.Tensor,
    h0: torch.Tensor,
    eta: float,
    lam_R: float,
    mu: float,
    eps: float,
    max_iter: int,
    row_update_scale: torch.Tensor,
) -> tuple[int, int, int]:
    iterations_run = 0
    guard_rejects = 0
    guard_total = 0
    for iteration in range(int(max_iter)):
        S_chunk = U_chunk * scale_chunk[None, :]
        H_col, hinge, scaled_residual = _dimensionless_hinge_stats(A_scaled, Y_scaled_chunk, S_chunk)
        excess = (H_col - h0 - float(eps)).clamp_min(0.0)
        if float(mu) > 0.0 and bool((excess > 0).any()):
            active = hinge * scaled_residual.sign()
            grad_s = 2.0 * A_scaled.T.matmul(active)
            grad_u = grad_s * scale_chunk[None, :]
            grad_u = grad_u * row_update_scale[:, None]
            grad_u = grad_u * (2.0 * float(mu) * excess)[None, :]
        else:
            grad_u = torch.zeros_like(U_chunk)
        shrink = float(eta) * float(lam_R) * row_update_scale[:, None]
        candidate = (U_chunk - float(eta) * grad_u - shrink).clamp_min(0.0)
        candidate_H, _, _ = _dimensionless_hinge_stats(
            A_scaled,
            Y_scaled_chunk,
            candidate * scale_chunk[None, :],
        )
        accept = torch.isfinite(candidate_H) & (candidate_H <= h0 + float(eps))
        guard_rejects += int((~accept).sum().detach().cpu())
        guard_total += int(accept.numel())
        U_chunk.copy_(torch.where(accept[None, :], candidate, U_chunk))
        iterations_run = iteration + 1
    return iterations_run, guard_rejects, guard_total


def _dimensionless_hinge_stats(
    A_scaled: torch.Tensor,
    Y_scaled_chunk: torch.Tensor,
    S_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scaled_residual = A_scaled.matmul(S_chunk) - Y_scaled_chunk
    hinge = (scaled_residual.abs() - 0.5).clamp_min(0.0)
    H_col = hinge.square().sum(dim=0)
    return H_col, hinge, scaled_residual
