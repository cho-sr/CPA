from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import torch
import torch.nn.functional as F

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
    ica_method: str
    ica_n_iter: int
    ica_converged: bool
    cpa_ica_optimizer: str
    cpa_ica_lr: float
    cpa_ica_decor: float
    cpa_ica_T: float
    cpa_ica_ne: float
    cpa_ica_nv: float
    cpa_ica_l1: float
    cpa_ica_tv: float
    cpa_ica_loss_initial: dict[str, float]
    cpa_ica_loss_final: dict[str, float]
    A_hat_reconstruction_method: str
    pre_ls_reconstruction_rel_l2: float
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
    c_mp: float = 3.0,
    mp_truncation: bool = True,
    fastica_algorithm: str = "parallel",
    ica_method: str = "cpa",
    cpa_ica_max_iter: int | None = None,
    cpa_ica_lr: float = 1.0e-3,
    cpa_ica_optimizer: str = "adam",
    cpa_ica_decor: float = 5.3,
    cpa_ica_T: float = 7.7,
    cpa_ica_ne: float = 1.0,
    cpa_ica_nv: float = 0.13,
    cpa_ica_l1: float = 5.0,
    cpa_ica_tv: float = 0.1,
    cpa_ica_log_every: int = 0,
    whitening_tolerance: float = 1.0e-3,
    device: torch.device | str | None = None,
) -> Stage1Result:
    if Y_hat.ndim != 2:
        raise ValueError(f"Y_hat must be 2D, got {tuple(Y_hat.shape)}")
    N, d = Y_hat.shape
    if not (0 < n_components <= min(N, d)):
        raise ValueError(f"invalid n_components={n_components} for Y_hat={tuple(Y_hat.shape)}")

    Y = Y_hat.detach().cpu().float()
    delta_cpu = delta.detach().cpu().float()
    Y_cov = Y.double()
    delta_cov = delta_cpu.double()
    sigma2 = float((delta_cov.square() / 12.0).mean().cpu())
    mp_edge = float(sigma2 * (1.0 + (float(N) / float(d)) ** 0.5) ** 2)
    mp_threshold = float(c_mp) * mp_edge
    row_mean = Y.mean(dim=1, keepdim=True)
    Y_zc = Y - row_mean
    row_mean_cov = Y_cov.mean(dim=1, keepdim=True)
    Y_zc_cov = Y_cov - row_mean_cov
    cov_raw = Y_zc_cov.matmul(Y_zc_cov.T) / float(d)
    cov = cov_raw.clone()
    debias_variance = 0.0
    noise_variance = torch.zeros(N, dtype=cov.dtype)
    if debias and rounding == "stochastic":
        noise_variance = delta_cov.square() / 12.0
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
    whitening_double = lam_inv_sqrt[:, None] * P
    whitening = whitening_double.float()
    P_float = P.float()
    lam_sqrt_float = lam_sqrt.float()
    Y_white = whitening.matmul(Y_zc)
    whitened_cov = whitening_double.matmul(cov_raw - torch.diag(noise_variance)).matmul(whitening_double.T)
    identity = torch.eye(B_eff, dtype=whitened_cov.dtype)
    offdiag = whitened_cov - torch.diag(torch.diagonal(whitened_cov))
    offdiag_abs_max = float(offdiag.abs().max().cpu())
    diag_values = torch.diagonal(whitened_cov)
    diag_min = float(diag_values.min().cpu())
    diag_max = float(diag_values.max().cpu())
    whitening_tol = float(whitening_tolerance)
    if offdiag_abs_max > whitening_tol or not torch.allclose(
        whitened_cov,
        identity,
        atol=whitening_tol,
        rtol=whitening_tol,
    ):
        raise RuntimeError(
            "Stage-1 whitening failed: "
            f"offdiag_abs_max={offdiag_abs_max:.6g}, diag_min={diag_min:.6g}, diag_max={diag_max:.6g}"
        )

    warning_messages: list[str] = []
    method = str(ica_method).lower()
    if method in {"fastica", "sklearn"}:
        FastICA = _import_fastica()
        ica = FastICA(
            algorithm=str(fastica_algorithm),
            whiten=False,
            fun=fun,
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=int(seed),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ica.fit_transform(np.ascontiguousarray(Y_white.T.numpy()))
        for item in caught:
            warning_messages.append(str(item.message))

        unmixing_ica = torch.as_tensor(ica.components_, dtype=torch.float32)
        raw_n_iter = getattr(ica, "n_iter_", -1)
        if isinstance(raw_n_iter, (list, tuple, np.ndarray)):
            n_iter = int(np.max(raw_n_iter)) if len(raw_n_iter) else -1
        else:
            n_iter = int(raw_n_iter)
        ica_converged = bool(n_iter < int(max_iter)) if n_iter >= 0 else False
        cpa_initial_loss: dict[str, float] = {}
        cpa_final_loss: dict[str, float] = {}
    elif method == "cpa":
        cpa_max_iter = int(25000 if cpa_ica_max_iter is None else cpa_ica_max_iter)
        cpa_result = _run_cpa_ica(
            Y_white=Y_white,
            whitening=whitening,
            row_mean=row_mean,
            max_iter=cpa_max_iter,
            lr=float(cpa_ica_lr),
            optimizer=str(cpa_ica_optimizer),
            decor=float(cpa_ica_decor),
            T=float(cpa_ica_T),
            ne=float(cpa_ica_ne),
            nv=float(cpa_ica_nv),
            l1=float(cpa_ica_l1),
            tv=float(cpa_ica_tv),
            log_every=int(cpa_ica_log_every),
            device=device,
        )
        unmixing_ica = cpa_result["unmixing"]
        n_iter = int(cpa_result["n_iter"])
        ica_converged = bool(cpa_result["converged"])
        cpa_initial_loss = cpa_result["initial_loss"]
        cpa_final_loss = cpa_result["final_loss"]
    else:
        raise ValueError(f"unknown Stage-1 ICA method {ica_method!r}; expected 'cpa' or 'fastica'")

    unmixing_full = unmixing_ica.matmul(whitening)
    S_hat = unmixing_full.matmul(Y)
    A_hat_red = torch.linalg.pinv(unmixing_ica)
    A_hat_pre_ls = P_float.T.matmul(lam_sqrt_float[:, None] * A_hat_red)
    Y_recon_pre_ls = A_hat_pre_ls.matmul(S_hat)
    pre_ls_reconstruction_rel_l2 = relative_l2(Y_recon_pre_ls, Y)
    A_hat = _least_squares_mixing(Y, S_hat)

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
        fastica_converged=ica_converged,
        fastica_algorithm=str(fastica_algorithm) if method != "cpa" else "cpa_adam_soft_penalty",
        ica_method=method,
        ica_n_iter=n_iter,
        ica_converged=ica_converged,
        cpa_ica_optimizer=str(cpa_ica_optimizer),
        cpa_ica_lr=float(cpa_ica_lr),
        cpa_ica_decor=float(cpa_ica_decor),
        cpa_ica_T=float(cpa_ica_T),
        cpa_ica_ne=float(cpa_ica_ne),
        cpa_ica_nv=float(cpa_ica_nv),
        cpa_ica_l1=float(cpa_ica_l1),
        cpa_ica_tv=float(cpa_ica_tv),
        cpa_ica_loss_initial=cpa_initial_loss,
        cpa_ica_loss_final=cpa_final_loss,
        A_hat_reconstruction_method="least_squares_from_observed_Y_hat",
        pre_ls_reconstruction_rel_l2=pre_ls_reconstruction_rel_l2,
        reconstruction_rel_l2=relative_l2(Y_recon, Y_hat.detach().cpu().float()),
        warnings=warning_messages,
    )


def _least_squares_mixing(Y: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    gram = S.matmul(S.T)
    rhs = S.matmul(Y.T)
    try:
        return torch.linalg.solve(gram, rhs).T
    except RuntimeError:
        return (torch.linalg.pinv(gram).matmul(rhs)).T


def _run_cpa_ica(
    *,
    Y_white: torch.Tensor,
    whitening: torch.Tensor,
    row_mean: torch.Tensor,
    max_iter: int,
    lr: float,
    optimizer: str,
    decor: float,
    T: float,
    ne: float,
    nv: float,
    l1: float,
    tv: float,
    log_every: int,
    device: torch.device | str | None,
) -> dict[str, object]:
    if Y_white.ndim != 2:
        raise ValueError(f"Y_white must be 2D, got {tuple(Y_white.shape)}")
    B_eff = int(Y_white.shape[0])
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    X_w = Y_white.detach().to(device=dev, dtype=torch.float32)
    W_w = whitening.detach().to(device=dev, dtype=torch.float32)
    X_mu = row_mean.detach().to(device=dev, dtype=torch.float32)
    W_hat = torch.empty((B_eff, B_eff), dtype=torch.float32, device=dev, requires_grad=True)
    torch.nn.init.eye_(W_hat)

    opt_name = str(optimizer).lower()
    if opt_name == "adam":
        opt = torch.optim.Adam([W_hat], lr=float(lr))
    elif opt_name == "sgd":
        opt = torch.optim.SGD([W_hat], lr=float(lr))
    else:
        raise ValueError(f"unknown CPA ICA optimizer {optimizer!r}")

    eps = torch.tensor(1.0e-20, dtype=torch.float32, device=dev)
    initial_loss = _cpa_ica_loss_dict(
        W_hat=W_hat,
        X_w=X_w,
        W_w=W_w,
        X_mu=X_mu,
        eps=eps,
        decor=float(decor),
        T=float(T),
        ne=float(ne),
        nv=float(nv),
        l1=float(l1),
        tv=float(tv),
    )
    final_loss = initial_loss
    max_iter = int(max_iter)
    for iteration in range(max_iter):
        opt.zero_grad(set_to_none=True)
        loss_values = _cpa_ica_loss_tensors(
            W_hat=W_hat,
            X_w=X_w,
            W_w=W_w,
            X_mu=X_mu,
            eps=eps,
            decor=float(decor),
            T=float(T),
            ne=float(ne),
            nv=float(nv),
            l1=float(l1),
            tv=float(tv),
        )
        loss_values["loss"].backward()
        opt.step()
        if int(log_every) > 0 and (iteration == 0 or (iteration + 1) % int(log_every) == 0):
            final_loss = _detach_loss_dict(loss_values)

    final_loss = _cpa_ica_loss_dict(
        W_hat=W_hat,
        X_w=X_w,
        W_w=W_w,
        X_mu=X_mu,
        eps=eps,
        decor=float(decor),
        T=float(T),
        ne=float(ne),
        nv=float(nv),
        l1=float(l1),
        tv=float(tv),
    )
    with torch.no_grad():
        W_norm = W_hat / (W_hat.norm(dim=-1, keepdim=True) + eps)
    return {
        "unmixing": W_norm.detach().cpu().float().contiguous(),
        "n_iter": max_iter,
        "converged": True,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
    }


def _cpa_ica_loss_tensors(
    *,
    W_hat: torch.Tensor,
    X_w: torch.Tensor,
    W_w: torch.Tensor,
    X_mu: torch.Tensor,
    eps: torch.Tensor,
    decor: float,
    T: float,
    ne: float,
    nv: float,
    l1: float,
    tv: float,
) -> dict[str, torch.Tensor]:
    zero = torch.tensor(0.0, dtype=torch.float32, device=W_hat.device)
    loss_ne = loss_decor = loss_nv = loss_tv = loss_l1 = zero
    W_hat_norm = W_hat / (W_hat.norm(dim=-1, keepdim=True) + eps)

    S_hat_centered = torch.matmul(W_hat_norm, X_w)
    if torch.isnan(S_hat_centered).any():
        raise ValueError("CPA ICA S_hat has NaN")

    if float(ne) > 0.0:
        loss_ne = -(((1.0 * torch.log(torch.cosh(S_hat_centered) + eps).mean(dim=-1)) ** 2).mean())
        loss_ne = torch.tensor(0.0, dtype=torch.float32, device=W_hat.device)

    S_hat = S_hat_centered + torch.matmul(torch.matmul(W_hat_norm, W_w), X_mu)
    if float(decor) > 0.0:
        cos_matrix = torch.matmul(W_hat_norm, W_hat_norm.T).abs()
        loss_decor = (torch.exp(cos_matrix * float(T)) - 1.0).mean()
    if float(tv) > 0.0 and float(nv) == 0.0:
        raise ValueError("CPA ICA TV prior needs image-shaped sources; VGG embedding runs use nv>0 and skip TV.")
    if float(nv) > 0.0:
        loss_nv = torch.minimum(F.relu(-S_hat).norm(dim=-1), F.relu(S_hat).norm(dim=-1)).mean()
    if float(l1) > 0.0:
        loss_l1 = torch.abs(S_hat).mean()

    loss = loss_ne + float(decor) * loss_decor + float(tv) * loss_tv + float(nv) * loss_nv + float(l1) * loss_l1
    return {
        "loss": loss,
        "loss_ne": loss_ne,
        "loss_decor": loss_decor,
        "loss_tv": loss_tv,
        "loss_nv": loss_nv,
        "loss_l1": loss_l1,
    }


def _cpa_ica_loss_dict(**kwargs: object) -> dict[str, float]:
    return _detach_loss_dict(_cpa_ica_loss_tensors(**kwargs))  # type: ignore[arg-type]


def _detach_loss_dict(loss_values: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in loss_values.items()}
