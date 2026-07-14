#!/usr/bin/env python3
"""Quantization-aware ICA attack runner isolated from the original CPA code."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from datasets import ds_type_dict
from eval_utils import get_eval
from feature_inversion import get_fi
from gradient_inversion import CocktailPartyAttack
from log_utils import AttackLog
from utils import (
    get_attack_exp_path,
    get_device,
    get_opt,
    get_pbar,
    get_sch,
    normalize,
    read_pickle,
    setup,
    subsample,
)

from common import build_experiment_model, make_attack_namespace


class QuantizationAwareCocktailPartyAttack(CocktailPartyAttack):
    """CPA/FastICA with a bounded dequantization residual on the attack layer."""

    def __init__(self, model, grads, labels, args, batch_id, quant_stats=None):
        self.qica_whitening = getattr(args, "qica_whitening", "fixed")
        super().__init__(model, grads, labels, args, batch_id)
        self.quant_stats = quant_stats
        self.qica_residual_l2 = float(getattr(args, "qica_residual_l2", 0.01))
        self.qica_half_width = self._get_quant_half_width()
        if self.qica_half_width is None:
            raise ValueError(
                "Quantization-aware ICA requires quant_stats with a positive scale "
                f"for attack layer index {self.model.attack_index}."
            )

        self.X_noise_param = torch.nn.Parameter(torch.zeros_like(self.X))
        self.opt = get_opt([self.W_hat, self.X_noise_param], self.args.opt, lr=self.args.lr)
        self.sch = get_sch(self.args.sch, self.opt, epochs=self.args.n_iter)

    def whiten(self, x):
        # Build the whitening matrix from the smaller sample covariance
        # instead of an SVD of the full VGG16 fc1 gradient matrix.
        with torch.no_grad():
            x_float = x.detach().float()
            cov = torch.matmul(x_float, x_float.T) / (x.shape[1] - 1)
            cov = 0.5 * (cov + cov.T)
            jitter = 1e-6 * cov.diagonal().abs().mean().clamp_min(1.0)
            cov = cov + torch.eye(cov.shape[0], device=x.device, dtype=cov.dtype) * jitter
            eig_vals, eig_vecs = torch.linalg.eigh(cov)
            topk_indices = torch.topk(eig_vals.abs(), self.n_comp)[1]
            lamb = eig_vals[topk_indices].abs().clamp_min(1e-6)
            W = torch.diag(torch.rsqrt(lamb)).matmul(eig_vecs.T[topk_indices])
            W = W.to(dtype=x.dtype)
        return torch.matmul(W, x), W

    def _get_quant_half_width(self):
        if self.quant_stats is None:
            return None
        per_tensor = self.quant_stats.get("per_tensor")
        scale = None
        if per_tensor is not None and self.model.attack_index < len(per_tensor):
            scale = per_tensor[self.model.attack_index].get("scale")
        if scale is None:
            scale = self.quant_stats.get("quant_scale_mean")
        if scale is None or scale <= 0:
            return None
        return torch.tensor(0.5 * float(scale), dtype=self.X.dtype, device=self.device)

    def _bounded_residual(self):
        return self.qica_half_width * torch.tanh(self.X_noise_param)

    def _current_ica_inputs(self):
        residual = self._bounded_residual()
        x_eff = self.X + residual
        if not torch.isfinite(x_eff).all():
            raise ValueError("X_eff has non-finite values")
        x_zc, x_mu = self.zero_center(x_eff)
        if self.qica_whitening == "fixed":
            w_w = self.W_w
            x_w = torch.matmul(w_w, x_zc)
        elif self.qica_whitening == "dynamic_cov":
            x_w, w_w = self.whiten(x_zc)
        else:
            raise ValueError(f"Unknown qica_whitening mode: {self.qica_whitening}")
        residual_unit = residual / self.qica_half_width.clamp_min(self.eps)
        loss_qica_residual = torch.mean(residual_unit * residual_unit)
        return x_w, w_w, x_mu, loss_qica_residual

    def get_attack_state(self):
        state_dict = {
            "W_hat": self.W_hat.detach().cpu(),
            "X_noise_param": self.X_noise_param.detach().cpu().half(),
        }
        if self.sch is not None:
            state_dict["sch"] = self.sch.state_dict()
        return state_dict

    def set_attack_state(self, state_dict):
        self.W_hat.data = state_dict["W_hat"].to(self.device, dtype=self.W_hat.dtype).data
        if "X_noise_param" in state_dict:
            self.X_noise_param.data = state_dict["X_noise_param"].to(
                self.device,
                dtype=self.X_noise_param.dtype,
            ).data
        if self.sch is not None and "sch" in state_dict:
            self.sch.load_state_dict(state_dict["sch"])

    def step(self):
        loss_ne = loss_decor = loss_nv = loss_tv = loss_l1 = loss_qica_residual = torch.tensor(
            0.0, device=self.device
        )

        self.opt.zero_grad()
        W_hat_norm = self.W_hat / (self.W_hat.norm(dim=-1, keepdim=True) + self.eps)
        X_w, W_w, X_mu, loss_qica_residual = self._current_ica_inputs()
        S_hat = torch.matmul(W_hat_norm, X_w)

        if torch.isnan(S_hat).any():
            raise ValueError("S_hat has NaN")

        if self.ne > 0:
            loss_ne = -(
                (
                    (1 / self.a)
                    * torch.log(torch.cosh(self.a * S_hat) + self.eps).mean(dim=-1)
                )
                ** 2
            ).mean()
            loss_ne = torch.tensor(0.0, device=self.device)

        S_hat = S_hat + torch.matmul(torch.matmul(W_hat_norm, W_w), X_mu)

        if self.decor > 0:
            cos_matrix = torch.matmul(W_hat_norm, W_hat_norm.T).abs()
            loss_decor = (torch.exp(cos_matrix * self.T) - 1).mean()

        if self.tv > 0 and self.nv == 0:
            loss_tv = self.total_variation(S_hat.view(self.inp_shape))

        if self.nv > 0:
            loss_nv = torch.minimum(
                F.relu(-S_hat).norm(dim=-1), F.relu(S_hat).norm(dim=-1)
            ).mean()

        if self.l1 > 0:
            loss_l1 = torch.abs(S_hat).mean()

        loss = (
            loss_ne
            + (self.decor * loss_decor)
            + (self.tv * loss_tv)
            + (self.nv * loss_nv)
            + (self.l1 * loss_l1)
            + (self.qica_residual_l2 * loss_qica_residual)
        )

        loss.backward()
        self.opt.step()
        if self.sch:
            self.sch.step()

        loss_dict = self.make_dict(
            [
                "loss",
                "loss_ne",
                "loss_decor",
                "loss_tv",
                "loss_nv",
                "loss_l1",
                "loss_qica_residual",
            ],
            [
                loss,
                loss_ne,
                loss_decor,
                loss_tv,
                loss_nv,
                loss_l1,
                loss_qica_residual,
            ],
        )
        return loss_dict

    def get_rec(self):
        with torch.no_grad():
            W_hat_norm = self.W_hat / (self.W_hat.norm(dim=-1, keepdim=True) + self.eps)
            X_w, W_w, X_mu, _ = self._current_ica_inputs()
            S_hat = torch.matmul(W_hat_norm, X_w)
            S_hat = S_hat + torch.matmul(torch.matmul(W_hat_norm, W_w), X_mu)
            S_hat = S_hat.detach().view(self.inp_shape)
            if self.inp_type == "image":
                S_hat = normalize(S_hat, method="infer")
        return S_hat


def _enable_qica_logging(attack_log: AttackLog) -> None:
    if "loss_qica_residual" in attack_log.tags_iter:
        return
    insert_at = attack_log.tags_iter.index("cs") if "cs" in attack_log.tags_iter else len(attack_log.tags_iter) - 1
    attack_log.tags_iter.insert(insert_at, "loss_qica_residual")
    attack_log.df_iter.insert(insert_at, "loss_qica_residual", [])


def run_quant_aware_ica_attack_with_update_file(
    args,
    *,
    exp_name: str,
    update_file: Path,
) -> None:
    attack_args = make_attack_namespace(
        args,
        exp_name=exp_name,
        quantized_update_file=update_file,
    )
    attack_args.fi_method = "direct"
    attack_args.qica_residual_l2 = args.qica_residual_l2
    attack_args.qica_whitening = args.qica_whitening

    exp_path = get_attack_exp_path(attack_args)
    logger = setup(exp_path=exp_path, log_file=f"{attack_args.batch_size}.log")
    grad_data = read_pickle(str(update_file))
    device = get_device()

    model = build_experiment_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        device=device,
        checkpoint_path=args.global_checkpoint,
    )
    model.eval()

    attack_log = AttackLog(
        args=attack_args,
        exp_path=exp_path,
        project=attack_args.project,
        model=model,
    )
    _enable_qica_logging(attack_log)
    logger.info(f"\nReading gradient data from {update_file}")

    for batch in range(attack_log.batch, attack_args.n_batch):
        inp = torch.tensor(grad_data["x"][batch], device=device)
        emb = torch.tensor(
            grad_data["z"][batch] if len(grad_data["z"]) > 0 else [], device=device
        )
        labels = torch.tensor(grad_data["y"][batch], device=device)
        grads = [torch.tensor(g, device=device) for g in grad_data["grad"][batch]]
        quant_stats = grad_data["quant_stats"][batch]

        attack_log.update_batch(batch, inp, emb, grads)

        if attack_log.attack_mode == "gi":
            gi = QuantizationAwareCocktailPartyAttack(
                model,
                grads,
                labels,
                attack_args,
                batch,
                quant_stats=quant_stats,
            )
            if attack_log.restore_required:
                gi.set_attack_state(attack_log.attack_state)
                attack_log.restore_required = False
                gi.start_iter = attack_log.iter

            pbar = get_pbar(range(gi.start_iter, attack_args.n_iter), disable=True)
            eval_gi = get_eval(
                inp,
                emb,
                model.model_type,
                ds_type_dict[attack_args.ds],
                attack_args.attack,
                fi=False,
            )

            for iter_idx in pbar:
                loss_dict = gi.step()
                if (iter_idx % attack_args.n_log == 0) or (iter_idx == attack_args.n_iter - 1):
                    rec_gi = gi.get_rec()
                    eval_avg, eval_batch, rec_gi_reord = eval_gi(rec_gi)
                    attack_log.update_iter(iter_idx, rec_gi_reord, loss_dict, eval_avg)
                    attack_log.checkpoint(gi.get_attack_state())
                    if iter_idx == attack_args.n_iter - 1:
                        attack_log.update_summary(eval_batch)

            rec_emb = rec_gi_reord.abs()
            rec = rec_gi_reord

        if model.model_type == "conv" and attack_args.n_iter_fi > 0:
            attack_log.attack_mode = "fi"
            attack_log.rec_emb = rec_emb

            inp_fi, emb_fi, rec_emb_fi = subsample(
                [inp, emb, rec_emb],
                n=attack_args.batch_size,
                n_sample=attack_args.n_sample_fi,
            )
            eval_fi = get_eval(
                inp_fi,
                emb_fi,
                model.model_type,
                ds_type_dict[attack_args.ds],
                attack_args.attack,
                fi=True,
            )
            attack_log.update_batch(batch, inp=inp_fi, fi=True)

            fi = get_fi(
                "direct",
                rec_emb_fi,
                model,
                attack_args,
                grads,
                labels,
                attack_log=attack_log,
            )
            pbar = get_pbar(range(fi.start_iter, attack_args.n_iter_fi), disable=True)
            for iter_fi in pbar:
                loss_dict_fi = fi.step()
                if (iter_fi % attack_args.n_log_fi == 0) or (iter_fi == attack_args.n_iter_fi - 1):
                    rec_fi = fi.get_rec()
                    eval_avg_fi, eval_batch_fi, _ = eval_fi(rec_fi)
                    attack_log.update_iter(
                        iter_fi,
                        rec_fi,
                        loss_dict_fi,
                        eval_avg_fi,
                        fi=True,
                    )
                    attack_log.checkpoint(fi.get_attack_state())
                    if iter_fi == attack_args.n_iter_fi - 1:
                        attack_log.update_summary(eval_batch_fi, fi=True)
            rec = rec_fi

        if ds_type_dict[attack_args.ds] == "image":
            inp = normalize(inp, method="ds", ds=attack_args.ds)

        attack_log.update_rec(
            inp.cpu().numpy(),
            emb.cpu().numpy(),
            rec_emb.cpu().numpy(),
            rec.cpu().numpy(),
        )
        attack_log.attack_mode = "gi"

    attack_log.save_to_disk()
