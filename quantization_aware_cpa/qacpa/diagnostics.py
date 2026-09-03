from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .alignment import Alignment, align_estimate_to_truth
from .linalg import condition_number
from .quantizer import consistency_rate, mean_hinge


@dataclass
class Diagnostics:
    B_requested: int
    B_eff: int
    rho: float
    gamma: float
    mean_axis_extent: float
    mean_axis_extent_normalized_by: str
    ica_rmse: float
    normalized_ica_rmse: float
    consistency_rate: float
    residual_violation: float
    cond_A_hat: float
    cond_A_true: float
    cond_A_true_matched: float
    source_cos_per_source: list[float]
    n_sources_recovered: int
    recovered_over_beff: float
    recovered_over_B: float
    a_hat_norm_per_source: list[float]
    a_true_norm_per_source: list[float]
    a_true_norm_matched: list[float]
    a_true_norm_unmatched: list[float]
    matched_true_indices: list[int]
    unmatched_true_indices: list[int]
    feasible_column_fraction: float
    infeasible_column_fraction: float
    gamma_columns: int
    gamma_infeasible_lps: int
    gamma_failed_lps: int
    alignment: Alignment


def _import_linprog():
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        raise RuntimeError("scipy is required for gamma LP diagnostics.") from exc
    return linprog


def compute_diagnostics(
    *,
    A_hat: torch.Tensor,
    S_hat: torch.Tensor,
    A_true: torch.Tensor,
    S_true: torch.Tensor,
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
    seed: int,
    gamma_columns: int,
    lp_method: str = "highs",
    lp_feasibility_tol: float = 1.0e-7,
) -> Diagnostics:
    rate = consistency_rate(A_hat, S_hat, Y_hat, delta)
    rho = 1.0 - rate
    residual = mean_hinge(A_hat, S_hat, Y_hat, delta)
    cond_hat = condition_number(A_hat)
    cond_true = condition_number(A_true)

    A_aligned, S_aligned, alignment = align_estimate_to_truth(A_hat, S_hat, S_true)
    B_requested = int(S_true.shape[0])
    B_eff = int(S_hat.shape[0])
    source_cos = alignment.per_source_abs_cosine
    n_sources_recovered = int(sum(value > 0.9 for value in source_cos))
    recovered_over_beff = float(n_sources_recovered / max(1, B_eff))
    recovered_over_B = float(n_sources_recovered / max(1, B_requested))
    a_hat_norm = [float(x) for x in A_aligned.norm(dim=0).detach().cpu()]
    a_true_norm_tensor = A_true.norm(dim=0).detach().cpu()
    a_true_norm = [float(x) for x in a_true_norm_tensor]
    a_true_norm_matched = [float(a_true_norm_tensor[idx]) for idx in alignment.true_indices]
    a_true_norm_unmatched = [float(a_true_norm_tensor[idx]) for idx in alignment.unmatched_true_indices]
    A_true_matched = A_true.index_select(
        1,
        torch.tensor(alignment.true_indices, dtype=torch.long, device=A_true.device),
    )
    cond_true_matched = condition_number(A_true_matched)
    columns = _sample_columns(S_true.shape[1], gamma_columns, seed)
    S_true_matched = S_true.index_select(
        0,
        torch.tensor(alignment.true_indices, dtype=torch.long, device=S_true.device),
    )
    ica_rmse = float(
        (S_aligned[:, columns] - S_true_matched[:, columns]).square().mean().sqrt().detach().cpu()
    )
    column_norms = S_aligned[:, columns].norm(dim=0).clamp_min(1.0e-12)
    normalized_ica_rmse = float(
        (
            (S_aligned[:, columns] - S_true_matched[:, columns])
            .div(column_norms[None, :])
            .square()
            .mean()
            .sqrt()
            .detach()
            .cpu()
        )
    )

    extent_result = estimate_axis_extent(
        A_aligned=A_aligned,
        S_reference=S_aligned,
        Y_hat=Y_hat,
        delta=delta,
        columns=columns,
        method=lp_method,
        feasibility_tol=lp_feasibility_tol,
    )
    mean_extent = extent_result["mean_axis_extent"]
    gamma = (
        float(mean_extent / normalized_ica_rmse)
        if normalized_ica_rmse > 0 and math.isfinite(mean_extent)
        else math.nan
    )

    return Diagnostics(
        B_requested=B_requested,
        B_eff=B_eff,
        rho=rho,
        gamma=gamma,
        mean_axis_extent=mean_extent,
        mean_axis_extent_normalized_by=str(extent_result["mean_axis_extent_normalized_by"]),
        ica_rmse=ica_rmse,
        normalized_ica_rmse=normalized_ica_rmse,
        consistency_rate=rate,
        residual_violation=residual,
        cond_A_hat=cond_hat,
        cond_A_true=cond_true,
        cond_A_true_matched=cond_true_matched,
        source_cos_per_source=source_cos,
        n_sources_recovered=n_sources_recovered,
        recovered_over_beff=recovered_over_beff,
        recovered_over_B=recovered_over_B,
        a_hat_norm_per_source=a_hat_norm,
        a_true_norm_per_source=a_true_norm,
        a_true_norm_matched=a_true_norm_matched,
        a_true_norm_unmatched=a_true_norm_unmatched,
        matched_true_indices=list(alignment.true_indices),
        unmatched_true_indices=list(alignment.unmatched_true_indices),
        feasible_column_fraction=extent_result["feasible_column_fraction"],
        infeasible_column_fraction=extent_result["infeasible_column_fraction"],
        gamma_columns=len(columns),
        gamma_infeasible_lps=extent_result["infeasible_lps"],
        gamma_failed_lps=extent_result["failed_lps"],
        alignment=alignment,
    )


def _sample_columns(d: int, requested: int, seed: int) -> np.ndarray:
    n_cols = min(int(requested), int(d), 512)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(d, size=n_cols, replace=False)).astype(np.int64)


def estimate_axis_extent(
    *,
    A_aligned: torch.Tensor,
    S_reference: torch.Tensor,
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
    columns: np.ndarray,
    method: str,
    feasibility_tol: float,
) -> dict[str, float | int]:
    linprog = _import_linprog()
    A_np = A_aligned.detach().cpu().double().numpy()
    S_np = S_reference.detach().cpu().double().numpy()
    Y_np = Y_hat.detach().cpu().double().numpy()
    delta_np = delta.detach().cpu().double().numpy()
    B = A_np.shape[1]
    A_ub_base = np.vstack([A_np, -A_np])
    bounds = [(0.0, None)] * B
    options = {
        "primal_feasibility_tolerance": float(feasibility_tol),
        "dual_feasibility_tolerance": float(feasibility_tol),
    }

    extents: list[float] = []
    feasible_columns = 0
    infeasible_columns = 0
    failed_columns = 0
    infeasible_lps = 0
    failed_lps = 0

    for col in columns:
        denominator = max(float(np.linalg.norm(S_np[:, int(col)])), 1.0e-12)
        upper = Y_np[:, int(col)] + delta_np / 2.0
        lower = Y_np[:, int(col)] - delta_np / 2.0
        b_ub = np.concatenate([upper, -lower])
        column_extents: list[float] = []
        column_infeasible = False
        column_failed = False
        for axis in range(B):
            c = np.zeros(B, dtype=np.float64)
            c[axis] = 1.0
            min_res = linprog(
                c,
                A_ub=A_ub_base,
                b_ub=b_ub,
                bounds=bounds,
                method=method,
                options=options,
            )
            max_res = linprog(
                -c,
                A_ub=A_ub_base,
                b_ub=b_ub,
                bounds=bounds,
                method=method,
                options=options,
            )
            if min_res.status == 2 or max_res.status == 2:
                infeasible_lps += int(min_res.status == 2) + int(max_res.status == 2)
                column_infeasible = True
                break
            if not min_res.success or not max_res.success:
                failed_lps += int(not min_res.success) + int(not max_res.success)
                column_failed = True
                break
            column_extents.append(float((-max_res.fun - min_res.fun) / denominator))
        if column_infeasible:
            infeasible_columns += 1
        elif column_failed:
            failed_columns += 1
        else:
            feasible_columns += 1
            extents.extend(column_extents)

    mean_extent = float(np.mean(extents)) if extents else math.nan
    total_columns = max(1, len(columns))
    return {
        "mean_axis_extent": mean_extent,
        "mean_axis_extent_normalized_by": "per-column ||s_hat[:, k]||",
        "feasible_column_fraction": float(feasible_columns / total_columns),
        "infeasible_column_fraction": float(infeasible_columns / total_columns),
        "failed_column_fraction": float(failed_columns / total_columns),
        "infeasible_lps": infeasible_lps,
        "failed_lps": failed_lps,
    }


def estimate_column_feasibility(
    *,
    A: torch.Tensor,
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
    columns: np.ndarray,
    method: str,
    feasibility_tol: float,
) -> dict[str, float | int]:
    linprog = _import_linprog()
    A_np = A.detach().cpu().double().numpy()
    Y_np = Y_hat.detach().cpu().double().numpy()
    delta_np = delta.detach().cpu().double().numpy()
    B = A_np.shape[1]
    A_ub_base = np.vstack([A_np, -A_np])
    bounds = [(0.0, None)] * B
    options = {
        "primal_feasibility_tolerance": float(feasibility_tol),
        "dual_feasibility_tolerance": float(feasibility_tol),
    }
    c = np.zeros(B, dtype=np.float64)

    feasible_columns = 0
    infeasible_columns = 0
    failed_columns = 0
    for col in columns:
        upper = Y_np[:, int(col)] + delta_np / 2.0
        lower = Y_np[:, int(col)] - delta_np / 2.0
        b_ub = np.concatenate([upper, -lower])
        result = linprog(
            c,
            A_ub=A_ub_base,
            b_ub=b_ub,
            bounds=bounds,
            method=method,
            options=options,
        )
        if result.status == 2:
            infeasible_columns += 1
        elif not result.success:
            failed_columns += 1
        else:
            feasible_columns += 1

    total_columns = max(1, len(columns))
    return {
        "feasible_column_fraction": float(feasible_columns / total_columns),
        "infeasible_column_fraction": float(infeasible_columns / total_columns),
        "failed_column_fraction": float(failed_columns / total_columns),
        "feasible_columns": int(feasible_columns),
        "infeasible_columns": int(infeasible_columns),
        "failed_columns": int(failed_columns),
        "total_columns": int(len(columns)),
    }
