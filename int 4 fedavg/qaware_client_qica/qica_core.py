"""Core client-wise bin-constrained QICA routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

QMIN = -7
QMAX = 7


@dataclass
class FixedWhitening:
    mean: torch.Tensor
    whitening: torch.Tensor
    eigvals: torch.Tensor
    numerical_rank: int
    min_eigvalue: float
    max_eigvalue: float
    selected_columns: Optional[torch.Tensor]


def finite_bin_bounds(
    codes: torch.Tensor,
    scale: torch.Tensor | float,
    max_abs: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite nearest-rounding bins for max-based symmetric int4."""
    code_f = codes.to(dtype=torch.float32)
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=code_f.device)
    max_abs_t = torch.as_tensor(max_abs, dtype=torch.float32, device=code_f.device)
    while scale_t.ndim < code_f.ndim:
        scale_t = scale_t.unsqueeze(-1)
    while max_abs_t.ndim < code_f.ndim:
        max_abs_t = max_abs_t.unsqueeze(-1)

    lower = (code_f - 0.5) * scale_t
    upper = (code_f + 0.5) * scale_t
    lower = torch.where(code_f <= QMIN, -max_abs_t, lower)
    upper = torch.where(code_f >= QMAX, max_abs_t, upper)
    zero = scale_t <= 0
    lower = torch.where(zero, torch.zeros_like(lower), lower)
    upper = torch.where(zero, torch.zeros_like(upper), upper)
    return lower, upper


def bin_distance_squared(
    predicted: torch.Tensor,
    codes: torch.Tensor,
    scale: torch.Tensor | float,
    max_abs: torch.Tensor | float,
) -> torch.Tensor:
    lower, upper = finite_bin_bounds(codes, scale, max_abs)
    return (torch.relu(lower - predicted) + torch.relu(predicted - upper)).square()


def project_to_finite_bins_(
    values: torch.Tensor,
    codes: torch.Tensor,
    scale: torch.Tensor | float,
    max_abs: torch.Tensor | float,
) -> None:
    lower, upper = finite_bin_bounds(codes, scale, max_abs)
    values.copy_(torch.minimum(torch.maximum(values, lower), upper))


def requantize(values: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    scale_t = torch.as_tensor(scale, dtype=values.dtype, device=values.device)
    while scale_t.ndim < values.ndim:
        scale_t = scale_t.unsqueeze(-1)
    q = torch.round(values / scale_t.clamp_min(torch.finfo(values.dtype).eps)).clamp(QMIN, QMAX)
    q = torch.where(scale_t <= 0, torch.zeros_like(q), q)
    return q.to(torch.int8)


def bin_satisfaction(values: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor, max_abs: torch.Tensor) -> float:
    dist = bin_distance_squared(values, codes, scale, max_abs)
    return float((dist <= 1e-16).float().mean().detach().cpu())


def requantization_match(values: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor) -> float:
    q = requantize(values, scale)
    return float((q == codes).float().mean().detach().cpu())


def weighted_aggregate(client_values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (client_values * weights.view(-1, 1, 1)).sum(dim=0)


def _select_columns(dim: int, count: Optional[int], device: torch.device, seed: int) -> Optional[torch.Tensor]:
    if count is None or count <= 0 or count >= dim:
        return None
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cols = torch.randperm(dim, generator=generator)[:count].to(device=device)
    return torch.sort(cols).values


def compute_fixed_whitening(
    aggregate: torch.Tensor,
    *,
    n_components: int,
    eig_floor_rel: float,
    rank_tol_rel: float,
    eps: float,
    column_count: Optional[int],
    seed: int,
) -> FixedWhitening:
    if aggregate.ndim != 2:
        raise ValueError(f"aggregate must be [H, D], got shape={tuple(aggregate.shape)}")
    h_dim, d_dim = aggregate.shape
    if n_components > h_dim:
        raise ValueError(f"M={n_components} exceeds attack-layer row count H={h_dim}")
    cols = _select_columns(d_dim, column_count, aggregate.device, seed)
    mean = aggregate.mean(dim=-1, keepdim=True)
    sample = aggregate if cols is None else aggregate.index_select(1, cols)
    sample_centered = sample - mean
    denom = max(1, sample_centered.shape[1] - 1)
    cov = sample_centered.matmul(sample_centered.T) / denom
    cov = 0.5 * (cov + cov.T)
    jitter = eps * cov.diagonal().abs().mean().clamp_min(1.0)
    cov = cov + torch.eye(h_dim, dtype=cov.dtype, device=cov.device) * jitter
    eig_vals, eig_vecs = torch.linalg.eigh(cov)
    eig_vals = eig_vals.clamp_min(0.0)
    max_eig = float(eig_vals.max().detach().cpu())
    if max_eig <= 0.0:
        raise ValueError("aggregate observation has zero numerical energy")
    rank_threshold = max_eig * float(rank_tol_rel)
    numerical_rank = int((eig_vals >= rank_threshold).sum().detach().cpu())
    if n_components > numerical_rank:
        raise ValueError(
            f"M={n_components} exceeds numerical rank={numerical_rank} "
            f"(rank_tol_rel={rank_tol_rel})"
        )
    top_indices = torch.topk(eig_vals, k=n_components).indices
    top_vals = eig_vals[top_indices]
    eig_floor = max_eig * float(eig_floor_rel)
    top_vals_clamped = top_vals.clamp_min(eig_floor)
    whitening = torch.diag(torch.rsqrt(top_vals_clamped)).matmul(eig_vecs.T[top_indices])
    return FixedWhitening(
        mean=mean.detach(),
        whitening=whitening.detach(),
        eigvals=top_vals.detach(),
        numerical_rank=numerical_rank,
        min_eigvalue=float(top_vals.min().detach().cpu()),
        max_eigvalue=float(top_vals.max().detach().cpu()),
        selected_columns=cols.detach() if cols is not None else None,
    )


def standardized_negentropy_loss(sources: torch.Tensor, *, eps: float) -> torch.Tensor:
    centered = sources - sources.mean(dim=-1, keepdim=True)
    standardized = centered / (centered.std(dim=-1, keepdim=True, unbiased=False) + eps)
    g_s = torch.log(torch.cosh(standardized.clamp(-10, 10))).mean(dim=-1)
    normal_ref = torch.tensor(0.374567, dtype=sources.dtype, device=sources.device)
    return -(g_s - normal_ref).square().mean()


def offdiag_mean_abs(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[0] <= 1:
        return torch.tensor(0.0, device=matrix.device, dtype=matrix.dtype)
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask].abs().mean()


def source_covariance_stats(sources: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    if sources.shape[1] <= 1:
        cov = torch.zeros((sources.shape[0], sources.shape[0]), device=sources.device, dtype=sources.dtype)
    else:
        centered = sources - sources.mean(dim=-1, keepdim=True)
        cov = centered.matmul(centered.T) / max(1, centered.shape[1] - 1)
    diag = cov.diagonal()
    return {
        "source_cov_diag_mean": float(diag.mean().detach().cpu()) if diag.numel() else 0.0,
        "source_cov_offdiag_mean_abs": float(offdiag_mean_abs(cov).detach().cpu()),
        "source_cov_fro": float(torch.linalg.vector_norm(cov).detach().cpu()),
    }


class ClientWiseBinConstrainedQICA:
    def __init__(
        self,
        *,
        codes: torch.Tensor,
        scales: torch.Tensor,
        max_abs: torch.Tensor,
        weights: torch.Tensor,
        whitening: FixedWhitening,
        n_components: int,
        use_q_loss: bool,
        use_ne: bool,
        train_observation: bool,
        lr: float,
        eps: float,
        decor_weight: float,
        decor_temperature: float,
        l1_weight: float,
        nv_weight: float,
        q_weight: float,
        seed: int,
    ) -> None:
        if use_q_loss and not train_observation:
            raise ValueError("Q-on ablations require qica_train_observation=True")
        self.codes = codes
        self.scales = scales
        self.max_abs = max_abs
        self.weights = weights
        self.whitening = whitening
        self.use_q_loss = bool(use_q_loss)
        self.use_ne = bool(use_ne)
        self.train_observation = bool(train_observation and use_q_loss)
        self.eps = float(eps)
        self.decor_weight = float(decor_weight)
        self.decor_temperature = float(decor_temperature)
        self.l1_weight = float(l1_weight)
        self.nv_weight = float(nv_weight)
        self.q_weight = float(q_weight)
        self.n_components = int(n_components)
        self.device = codes.device

        init = codes.to(dtype=torch.float32) * scales.view(-1, 1, 1)
        self.X_client_hat = torch.nn.Parameter(init) if self.train_observation else init.detach()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        eye = torch.eye(n_components, device=self.device, dtype=torch.float32)
        noise = torch.randn((n_components, n_components), generator=generator, device=self.device) * 1e-3
        self.W_hat = torch.nn.Parameter(eye + noise)
        params = [self.W_hat]
        if isinstance(self.X_client_hat, torch.nn.Parameter):
            params.append(self.X_client_hat)
        # SGD avoids allocating Adam states for the very large client observation tensor.
        self.optimizer = torch.optim.SGD(params, lr=lr)
        self.last_grad_norm_x: Optional[float] = None

    def _normalized_w(self) -> torch.Tensor:
        return self.W_hat / (self.W_hat.norm(dim=-1, keepdim=True) + self.eps)

    def _current_client(self) -> torch.Tensor:
        return self.X_client_hat if isinstance(self.X_client_hat, torch.Tensor) else torch.as_tensor(self.X_client_hat)

    def choose_step_columns(self, columns_per_step: Optional[int], step: int) -> Optional[torch.Tensor]:
        dim = int(self.codes.shape[-1])
        if columns_per_step is None or columns_per_step <= 0 or columns_per_step >= dim:
            return None
        generator = torch.Generator(device="cpu").manual_seed(10_000_019 + step)
        cols = torch.randperm(dim, generator=generator)[: int(columns_per_step)].to(device=self.device)
        return torch.sort(cols).values

    def source(self, cols: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._current_client()
        x_sel = x if cols is None else x.index_select(2, cols)
        aggregate = weighted_aggregate(x_sel, self.weights)
        centered = aggregate - self.whitening.mean
        w_norm = self._normalized_w()
        centered_source = w_norm.matmul(self.whitening.whitening).matmul(centered)
        mean_source = w_norm.matmul(self.whitening.whitening).matmul(self.whitening.mean)
        return centered_source + mean_source, aggregate

    def step(self, *, iter_idx: int, columns_per_step: Optional[int]) -> Dict[str, float]:
        cols = self.choose_step_columns(columns_per_step, iter_idx)
        self.optimizer.zero_grad(set_to_none=True)
        x = self._current_client()
        x_sel = x if cols is None else x.index_select(2, cols)
        codes_sel = self.codes if cols is None else self.codes.index_select(2, cols)
        pre_step_violation = bin_distance_squared(x_sel, codes_sel, self.scales, self.max_abs)

        source, _aggregate = self.source(cols)
        loss_q = torch.tensor(0.0, device=self.device)
        if self.use_q_loss:
            dist = bin_distance_squared(x_sel, codes_sel, self.scales, self.max_abs)
            per_client = dist.mean(dim=(1, 2))
            loss_q = (per_client * self.weights).sum()
        loss_ne = standardized_negentropy_loss(source, eps=self.eps) if self.use_ne else torch.tensor(0.0, device=self.device)
        loss_decor = torch.tensor(0.0, device=self.device)
        if self.decor_weight > 0:
            w_norm = self._normalized_w()
            cos_matrix = w_norm.matmul(w_norm.T)
            loss_decor = (torch.exp(cos_matrix.abs() * self.decor_temperature) - 1.0)
            loss_decor = offdiag_mean_abs(loss_decor)
        loss_l1 = source.abs().mean() if self.l1_weight > 0 else torch.tensor(0.0, device=self.device)
        loss_nv = (
            torch.minimum(F.relu(-source).norm(dim=-1), F.relu(source).norm(dim=-1)).mean()
            if self.nv_weight > 0
            else torch.tensor(0.0, device=self.device)
        )
        loss = (
            self.q_weight * loss_q
            + loss_ne
            + self.decor_weight * loss_decor
            + self.l1_weight * loss_l1
            + self.nv_weight * loss_nv
        )
        loss.backward()
        self.last_grad_norm_x = None
        if isinstance(self.X_client_hat, torch.nn.Parameter) and self.X_client_hat.grad is not None:
            grad = self.X_client_hat.grad if cols is None else self.X_client_hat.grad.index_select(2, cols)
            self.last_grad_norm_x = float(torch.linalg.vector_norm(grad.detach()).cpu())
        self.optimizer.step()

        pre_projection_violation = 0.0
        if self.use_q_loss and isinstance(self.X_client_hat, torch.nn.Parameter):
            with torch.no_grad():
                target = self.X_client_hat.data if cols is None else self.X_client_hat.data.index_select(2, cols)
                target_codes = self.codes if cols is None else self.codes.index_select(2, cols)
                before = bin_distance_squared(target, target_codes, self.scales, self.max_abs)
                pre_projection_violation = float(before.mean().detach().cpu())
                projected = target.clone()
                project_to_finite_bins_(projected, target_codes, self.scales, self.max_abs)
                if cols is None:
                    self.X_client_hat.data.copy_(projected)
                else:
                    self.X_client_hat.data[:, :, cols] = projected

        x_post = self._current_client()
        x_post_sel = x_post if cols is None else x_post.index_select(2, cols)
        source_post, _aggregate_post = self.source(cols)
        cov_stats = source_covariance_stats(source_post, eps=self.eps)
        return {
            "loss": float(loss.detach().cpu()),
            "loss_q": float(loss_q.detach().cpu()),
            "loss_ne": float(loss_ne.detach().cpu()),
            "loss_decor": float(loss_decor.detach().cpu()),
            "loss_l1": float(loss_l1.detach().cpu()),
            "loss_nv": float(loss_nv.detach().cpu()),
            "bin_satisfaction": bin_satisfaction(x_post_sel, codes_sel, self.scales, self.max_abs),
            "requantization_match": requantization_match(x_post_sel, codes_sel, self.scales),
            "pre_step_violation": float(pre_step_violation.mean().detach().cpu()),
            "pre_projection_violation": pre_projection_violation,
            "x_client_grad_norm": self.last_grad_norm_x,
            "q_gradient_enabled": bool(self.use_q_loss and self.train_observation),
            "whitening_min_eigenvalue": self.whitening.min_eigvalue,
            "whitening_max_eigenvalue": self.whitening.max_eigvalue,
            "whitening_numerical_rank": self.whitening.numerical_rank,
            **cov_stats,
        }

    def full_sources(self) -> torch.Tensor:
        with torch.no_grad():
            source, _aggregate = self.source(None)
        return source.detach()


def oracle_embedding_eval(recovered: np.ndarray, truth: np.ndarray) -> Dict[str, Any]:
    from assignment import linear_sum_assignment

    recovered_f = np.asarray(recovered, dtype=np.float32)
    truth_f = np.asarray(truth, dtype=np.float32)
    if recovered_f.shape != truth_f.shape:
        return {
            "skipped": True,
            "reason": f"shape mismatch recovered={list(recovered_f.shape)} truth={list(truth_f.shape)}",
        }
    rec = recovered_f / np.maximum(np.linalg.norm(recovered_f, axis=1, keepdims=True), 1e-12)
    tru = truth_f / np.maximum(np.linalg.norm(truth_f, axis=1, keepdims=True), 1e-12)
    similarity = tru.astype(np.float64) @ rec.astype(np.float64).T
    rows, cols = linear_sum_assignment(-np.abs(similarity))
    abs_cosines = np.asarray([abs(similarity[row, col]) for row, col in zip(rows, cols)], dtype=np.float64)
    signed_cosines = np.asarray([similarity[row, col] for row, col in zip(rows, cols)], dtype=np.float64)
    return {
        "skipped": False,
        "policy": "oracle_hungarian_abs_cosine_eval_only",
        "mean_abs_cosine": float(abs_cosines.mean()),
        "median_abs_cosine": float(np.median(abs_cosines)),
        "min_abs_cosine": float(abs_cosines.min()),
        "max_abs_cosine": float(abs_cosines.max()),
        "mean_signed_cosine": float(signed_cosines.mean()),
        "recovered_order_for_true_index": cols.astype(int).tolist(),
        "per_sample_abs_cosine": abs_cosines.tolist(),
        "per_sample_signed_cosine": signed_cosines.tolist(),
    }
