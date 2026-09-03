from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import torch

from .alignment import positive_mean_sign_fix
from .linalg import condition_number, relative_l2


@dataclass
class Stage1Result:
    A_hat: torch.Tensor
    S_hat: torch.Tensor
    Y_recon: torch.Tensor
    B_requested: int
    B_eff: int
    cond_A_hat: float
    debias_variance: float
    debias_diag_summary: dict[str, float]
    explained_eigenvalues: list[float]
    lam_deb_topB: list[float]
    lam_deb_kept: list[float]
    sigma2: float
    mp_edge: float
    c_mp: float
    mp_threshold: float
    mp_truncation: bool
    lam_over_mp_edge_all: list[float]
    lam_over_mp_edge_kept: list[float]
    kept_component_indices: list[int]
    discarded_component_indices: list[int]
    whitening_gain_max: float
    whitened_cov_offdiag_abs_max: float
    whitened_cov_diag_min: float
    whitened_cov_diag_max: float
    fastica_n_iter: int
    fastica_converged: bool
    fastica_algorithm: str
    reconstruction_rel_l2: float
    warnings: list[str]


def _import_fastica():
    try:
        from sklearn.decomposition import FastICA
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for FastICA.") from exc
    return FastICA


def estimate_mixing_and_sources(
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
    *,
    n_components: int,
    rounding: str,
    debias: bool,
    seed: int,
    max_iter: int,
    tol: float,
    fun: str = "logcosh",
    c_mp: float = 1.5,
    mp_truncation: bool = True,
    fastica_algorithm: str = "parallel",
) -> Stage1Result:
    if Y_hat.ndim != 2:
        raise ValueError(f"Y_hat must be 2D, got {tuple(Y_hat.shape)}")
    N, d = Y_hat.shape
    if not (0 < n_components <= min(N, d)):
        raise ValueError(f"invalid n_components={n_components} for Y_hat={tuple(Y_hat.shape)}")

    Y = Y_hat.detach().cpu().float()
    delta_cpu = delta.detach().cpu().float()
    sigma2 = float((delta_cpu.square() / 12.0).mean().cpu())
    mp_edge = float(sigma2 * (1.0 + (float(N) / float(d)) ** 0.5) ** 2)
    mp_threshold = float(c_mp) * mp_edge
    row_mean = Y.mean(dim=1, keepdim=True)
    Y_zc = Y - row_mean
    cov_raw = Y_zc.matmul(Y_zc.T) / float(d)
    cov = cov_raw.clone()
    debias_variance = 0.0
    noise_variance = torch.zeros(N, dtype=cov.dtype)
    if debias and rounding == "stochastic":
        noise_variance = delta_cpu.square() / 12.0
        debias_variance = float(noise_variance.mean().cpu())
        cov = cov - torch.diag(noise_variance)

    cov = (cov + cov.T) * 0.5
    eig_vals, eig_vecs = torch.linalg.eigh(cov)
    top_idx = torch.argsort(eig_vals, descending=True)[:n_components]
    top_vals = eig_vals.index_select(0, top_idx)
    P_top = eig_vecs.index_select(1, top_idx).T.contiguous()

    if bool(mp_truncation):
        keep_mask = top_vals > mp_threshold
    else:
        keep_mask = torch.ones_like(top_vals, dtype=torch.bool)
    if int(keep_mask.sum().item()) == 0:
        raise RuntimeError(
            "MP truncation removed every Stage-1 component: "
            f"mp_edge={mp_edge:.9g}, c_mp={float(c_mp):.6g}, threshold={mp_threshold:.9g}, "
            f"lam_deb_topB={[float(x) for x in top_vals.detach().cpu()]}"
        )

    kept_positions = torch.nonzero(keep_mask, as_tuple=False).flatten()
    discarded_positions = torch.nonzero(~keep_mask, as_tuple=False).flatten()
    top_vals_kept = top_vals.index_select(0, kept_positions)
    if bool((top_vals_kept <= 0).any()):
        raise RuntimeError(
            "Stage-1 retained a nonpositive eigenvalue; refusing to clamp it. "
            f"retained={ [float(x) for x in top_vals_kept.detach().cpu()] }"
        )
    P = P_top.index_select(0, kept_positions)
    B_eff = int(top_vals_kept.numel())

    lam_inv_sqrt = torch.rsqrt(top_vals_kept)
    lam_sqrt = torch.sqrt(top_vals_kept)
    whitening = lam_inv_sqrt[:, None] * P
    Y_white = whitening.matmul(Y_zc)
    whitened_cov = whitening.matmul(cov_raw - torch.diag(noise_variance)).matmul(whitening.T)
    identity = torch.eye(B_eff, dtype=whitened_cov.dtype)
    offdiag = whitened_cov - torch.diag(torch.diagonal(whitened_cov))
    offdiag_abs_max = float(offdiag.abs().max().cpu())
    diag_values = torch.diagonal(whitened_cov)
    diag_min = float(diag_values.min().cpu())
    diag_max = float(diag_values.max().cpu())
    if offdiag_abs_max > 1.0e-3 or not torch.allclose(whitened_cov, identity, atol=1.0e-3, rtol=1.0e-3):
        raise RuntimeError(
            "Stage-1 whitening failed: "
            f"offdiag_abs_max={offdiag_abs_max:.6g}, diag_min={diag_min:.6g}, diag_max={diag_max:.6g}"
        )

    FastICA = _import_fastica()
    ica = FastICA(
        algorithm=str(fastica_algorithm),
        whiten=False,
        fun=fun,
        max_iter=int(max_iter),
        tol=float(tol),
        random_state=int(seed),
    )

    warning_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ica.fit_transform(np.ascontiguousarray(Y_white.T.numpy()))
    for item in caught:
        warning_messages.append(str(item.message))

    unmixing_ica = torch.as_tensor(ica.components_, dtype=torch.float32)
    A_hat_red = torch.linalg.pinv(unmixing_ica)
    unmixing_full = unmixing_ica.matmul(whitening)
    S_hat = unmixing_full.matmul(Y)
    A_hat = P.T.matmul(lam_sqrt[:, None] * A_hat_red)

    A_hat = A_hat.float().contiguous()
    S_hat = S_hat.float().contiguous()
    A_hat, S_hat = positive_mean_sign_fix(A_hat, S_hat)
    Y_recon = A_hat.matmul(S_hat)
    debias_diag_summary = {
        "min": float(noise_variance.min().cpu()),
        "max": float(noise_variance.max().cpu()),
        "mean": float(noise_variance.mean().cpu()),
        "median": float(noise_variance.median().cpu()),
    }

    raw_n_iter = getattr(ica, "n_iter_", -1)
    if isinstance(raw_n_iter, (list, tuple, np.ndarray)):
        n_iter = int(np.max(raw_n_iter)) if len(raw_n_iter) else -1
    else:
        n_iter = int(raw_n_iter)
    mp_den = mp_edge if mp_edge > 0.0 else float("nan")
    return Stage1Result(
        A_hat=A_hat,
        S_hat=S_hat,
        Y_recon=Y_recon,
        B_requested=int(n_components),
        B_eff=B_eff,
        cond_A_hat=condition_number(A_hat),
        debias_variance=debias_variance,
        debias_diag_summary=debias_diag_summary,
        explained_eigenvalues=[float(x) for x in top_vals_kept.detach().cpu()],
        lam_deb_topB=[float(x) for x in top_vals.detach().cpu()],
        lam_deb_kept=[float(x) for x in top_vals_kept.detach().cpu()],
        sigma2=sigma2,
        mp_edge=mp_edge,
        c_mp=float(c_mp),
        mp_threshold=mp_threshold,
        mp_truncation=bool(mp_truncation),
        lam_over_mp_edge_all=[float(x / mp_den) for x in top_vals.detach().cpu()],
        lam_over_mp_edge_kept=[float(x / mp_den) for x in top_vals_kept.detach().cpu()],
        kept_component_indices=[int(x) for x in kept_positions.detach().cpu()],
        discarded_component_indices=[int(x) for x in discarded_positions.detach().cpu()],
        whitening_gain_max=float(lam_inv_sqrt.max().cpu()),
        whitened_cov_offdiag_abs_max=offdiag_abs_max,
        whitened_cov_diag_min=diag_min,
        whitened_cov_diag_max=diag_max,
        fastica_n_iter=n_iter,
        fastica_converged=bool(n_iter < int(max_iter)) if n_iter >= 0 else False,
        fastica_algorithm=str(fastica_algorithm),
        reconstruction_rel_l2=relative_l2(Y_recon, Y_hat.detach().cpu().float()),
        warnings=warning_messages,
    )
