from __future__ import annotations

from dataclasses import dataclass
import time

import torch


@dataclass
class BPlusResult:
    S_star: torch.Tensor
    sigma_per_source: list[float]
    radius_scale: float
    lam_R: float
    step_size: float
    max_iter: int
    chunk_size: int
    initial_l1_mean: float
    final_l1_mean: float
    initial_l1_normalized_mean: float
    final_l1_normalized_mean: float
    s_scale_min: float
    s_scale_median: float
    s_scale_max: float
    lower_bound_hit_fraction: float
    wall_time_sec: float


def bplus_sigma(A_hat: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    unmixing = torch.linalg.pinv(A_hat.detach().cpu().float())
    noise_variance = delta.detach().cpu().float().square() / 12.0
    return torch.sqrt(unmixing.square().matmul(noise_variance).clamp_min(0.0))


def solve_bplus_l1(
    *,
    A_hat: torch.Tensor,
    S_hat: torch.Tensor,
    delta: torch.Tensor,
    lam_R: float,
    radius_scale: float,
    max_iter: int,
    chunk_size: int,
    step_size: float = 1.0,
) -> BPlusResult:
    start = time.perf_counter()
    A_cpu = A_hat.detach().cpu().float()
    center = S_hat.detach().cpu().float().clamp_min(0.0)
    sigma = bplus_sigma(A_cpu, delta)
    radius = float(radius_scale) * sigma[:, None]
    lower = (center - radius).clamp_min(0.0)
    upper = center + radius
    s_scale = center.abs().amax(dim=0).clamp_min(1.0e-12)
    lower_u = lower / s_scale[None, :]
    upper_u = upper / s_scale[None, :]
    shrink = float(step_size) * float(lam_R)

    S_star = torch.empty_like(center)
    max_iter = int(max_iter)
    chunk_size = int(chunk_size)
    for start_col in range(0, center.shape[1], chunk_size):
        end_col = min(start_col + chunk_size, center.shape[1])
        scale_chunk = s_scale[start_col:end_col]
        U_chunk = center[:, start_col:end_col] / scale_chunk[None, :]
        lower_chunk = lower_u[:, start_col:end_col]
        upper_chunk = upper_u[:, start_col:end_col]
        for _ in range(max_iter):
            U_chunk = (U_chunk - shrink).clamp_min(0.0)
            U_chunk = torch.minimum(torch.maximum(U_chunk, lower_chunk), upper_chunk)
        S_star[:, start_col:end_col] = U_chunk * scale_chunk[None, :]

    lower_hits = torch.isclose(S_star, lower, rtol=1.0e-5, atol=1.0e-7)
    center_u = center / s_scale[None, :]
    final_u = S_star / s_scale[None, :]
    return BPlusResult(
        S_star=S_star.contiguous(),
        sigma_per_source=[float(x) for x in sigma.tolist()],
        radius_scale=float(radius_scale),
        lam_R=float(lam_R),
        step_size=float(step_size),
        max_iter=max_iter,
        chunk_size=chunk_size,
        initial_l1_mean=float(center.abs().mean()),
        final_l1_mean=float(S_star.abs().mean()),
        initial_l1_normalized_mean=float(center_u.abs().mean()),
        final_l1_normalized_mean=float(final_u.abs().mean()),
        s_scale_min=float(s_scale.min()),
        s_scale_median=float(s_scale.median()),
        s_scale_max=float(s_scale.max()),
        lower_bound_hit_fraction=float(lower_hits.float().mean()),
        wall_time_sec=time.perf_counter() - start,
    )
