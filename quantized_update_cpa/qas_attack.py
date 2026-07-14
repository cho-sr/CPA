#!/usr/bin/env python3
"""Quantization-aware A,S factorization attack for 4-bit CPA updates."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from datasets import ds_type_dict
from eval_utils import get_eval
from feature_inversion import get_fi
from log_utils import AttackLog
from utils import (
    get_attack_exp_path,
    get_device,
    get_opt,
    get_pbar,
    normalize,
    read_pickle,
    setup,
    subsample,
)

from common import build_experiment_model, make_attack_namespace


class QuantizationAwareASAttack:
    """Optimize min_{A,S} ||D(Q(A S)) - Y_q||^2 + lambda R(S)."""

    def __init__(self, model, grads, labels, args, batch_id, quant_stats):
        self.model = model
        self.model.train()
        self.grads = grads
        self.labels = labels
        self.args = args
        self.batch_id = batch_id
        self.device = get_device()
        self.eps = torch.tensor(1e-20, device=self.device)
        self.n_comp = args.batch_size
        self.start_iter = 0
        self.Y_q = grads[model.attack_index].float()
        self.n_rows, self.n_features = self.Y_q.shape
        self.qmax = (2 ** (int(args.quant_bits) - 1)) - 1
        self.qmin = -self.qmax
        self.scale = self._get_quant_scale(quant_stats)
        self.rows_per_step = min(int(args.qas_rows_per_step), self.n_rows)
        self.row_generator = torch.Generator(device=self.device).manual_seed(
            int(args.seed) + 10_000 * batch_id
        )

        a_init, s_init = self._svd_init()
        self.A_raw = torch.nn.Parameter(a_init)
        self.S = torch.nn.Parameter(s_init)
        self.opt = get_opt([self.A_raw, self.S], self.args.opt, lr=self.args.lr)

    def _get_quant_scale(self, quant_stats):
        per_tensor = quant_stats.get("per_tensor")
        scale = None
        if per_tensor is not None and self.model.attack_index < len(per_tensor):
            scale = per_tensor[self.model.attack_index].get("scale")
        if scale is None:
            max_abs = self.Y_q.detach().abs().max()
            if max_abs > 0:
                scale = float((max_abs / self.qmax).detach().cpu())
        if scale is None:
            scale = quant_stats.get("quant_scale_mean")
        if scale is None or scale <= 0:
            raise ValueError(
                "QAS requires quant_stats with a positive scale "
                f"for attack layer index {self.model.attack_index}."
            )
        return torch.tensor(float(scale), dtype=self.Y_q.dtype, device=self.device)

    def _svd_init(self):
        with torch.no_grad():
            y = self.Y_q
            y_mu = y.mean(dim=-1, keepdim=True)
            y_zc = y - y_mu
            cov = torch.matmul(y_zc, y_zc.T) / max(1, y_zc.shape[1] - 1)
            cov = 0.5 * (cov + cov.T)
            jitter = 1e-6 * cov.diagonal().abs().mean().clamp_min(1.0)
            cov = cov + torch.eye(cov.shape[0], device=self.device) * jitter
            eig_vals, eig_vecs = torch.linalg.eigh(cov.float())
            topk = torch.topk(eig_vals.abs(), self.n_comp)[1]
            singular = torch.sqrt(eig_vals[topk].abs().clamp_min(1e-8) * max(1, y_zc.shape[1] - 1))
            u = eig_vecs[:, topk].to(dtype=y.dtype)
            v_t = torch.matmul(u.T, y_zc) / singular.to(dtype=y.dtype).unsqueeze(1)
            root = torch.sqrt(singular).to(dtype=y.dtype)
            a_init = u * root.unsqueeze(0)
            s_init = root.unsqueeze(1) * v_t
            col_norm = a_init.norm(dim=0).clamp_min(float(self.eps.detach().cpu()))
            s_init = col_norm.unsqueeze(1) * s_init
        return a_init, s_init

    def _normalized_A(self):
        return self.A_raw / (self.A_raw.norm(dim=0, keepdim=True) + self.eps)

    def _fake_quantize_ste(self, tensor):
        scaled = tensor / self.scale
        quantized = torch.round(scaled).clamp(self.qmin, self.qmax)
        dequantized = quantized * self.scale
        return tensor + (dequantized - tensor).detach()

    def _sample_rows(self):
        if self.rows_per_step >= self.n_rows:
            return torch.arange(self.n_rows, device=self.device)
        return torch.randperm(
            self.n_rows,
            device=self.device,
            generator=self.row_generator,
        )[: self.rows_per_step]

    def _source_priors(self, s_hat):
        s_centered = s_hat - s_hat.mean(dim=1, keepdim=True)
        s_unit = s_centered / (s_centered.norm(dim=1, keepdim=True) + self.eps)
        gram = torch.matmul(s_unit, s_unit.T).abs()
        offdiag = gram - torch.diag(torch.diag(gram))
        loss_ind = (offdiag * offdiag).mean()
        loss_ne = -torch.log(torch.cosh(s_centered.clamp(-10, 10)) + self.eps).mean()
        loss_l1 = s_hat.abs().mean()
        loss_nv = torch.minimum(
            F.relu(-s_hat).norm(dim=-1),
            F.relu(s_hat).norm(dim=-1),
        ).mean()
        a_unit = self._normalized_A()
        a_gram = torch.matmul(a_unit.T, a_unit).abs()
        a_offdiag = a_gram - torch.diag(torch.diag(a_gram))
        loss_a_decor = (a_offdiag * a_offdiag).mean()
        return loss_ind, loss_ne, loss_l1, loss_nv, loss_a_decor

    def step(self):
        self.opt.zero_grad(set_to_none=True)
        rows = self._sample_rows()
        a = self._normalized_A()
        y_hat = torch.matmul(a[rows], self.S)
        y_hat_q = self._fake_quantize_ste(y_hat)
        target = self.Y_q[rows]
        loss_qcons = torch.mean(((y_hat_q - target) / self.scale.clamp_min(self.eps)) ** 2)
        loss_ind, loss_ne, loss_l1, loss_nv, loss_a_decor = self._source_priors(self.S)
        loss = (
            self.args.qas_qcons * loss_qcons
            + self.args.qas_ind * loss_ind
            + self.args.qas_ne * loss_ne
            + self.args.qas_l1 * loss_l1
            + self.args.qas_nv * loss_nv
            + self.args.qas_a_decor * loss_a_decor
        )
        loss.backward()
        self.opt.step()
        return {
            "loss": loss.detach().cpu().item(),
            "loss_qcons": loss_qcons.detach().cpu().item(),
            "loss_ind": loss_ind.detach().cpu().item(),
            "loss_ne": loss_ne.detach().cpu().item(),
            "loss_decor": loss_a_decor.detach().cpu().item(),
            "loss_l1": loss_l1.detach().cpu().item(),
            "loss_nv": loss_nv.detach().cpu().item(),
            "loss_a_decor": loss_a_decor.detach().cpu().item(),
        }

    def get_rec(self):
        return self.S.detach().view(self.n_comp, -1)

    def get_attack_state(self):
        return {
            "A_raw": self.A_raw.detach().cpu().half(),
            "S": self.S.detach().cpu().half(),
        }

    def set_attack_state(self, state_dict):
        if "A_raw" in state_dict:
            self.A_raw.data = state_dict["A_raw"].to(self.device, dtype=self.A_raw.dtype).data
        if "S" in state_dict:
            self.S.data = state_dict["S"].to(self.device, dtype=self.S.dtype).data


def _enable_qas_logging(attack_log: AttackLog) -> None:
    extra = [
        "loss_qcons",
        "loss_ind",
        "loss_ne",
        "loss_l1",
        "loss_nv",
        "loss_a_decor",
    ]
    insert_at = attack_log.tags_iter.index("cs") if "cs" in attack_log.tags_iter else len(attack_log.tags_iter) - 1
    for tag in reversed(extra):
        if tag not in attack_log.tags_iter:
            attack_log.tags_iter.insert(insert_at, tag)
            attack_log.df_iter.insert(insert_at, tag, [])


def run_quant_aware_as_attack_with_update_file(args, *, exp_name: str, update_file: Path) -> None:
    attack_args = make_attack_namespace(
        args,
        exp_name=exp_name,
        quantized_update_file=update_file,
    )
    attack_args.fi_method = "direct"
    attack_args.qas_qcons = args.qas_qcons
    attack_args.qas_ind = args.qas_ind
    attack_args.qas_ne = args.qas_ne
    attack_args.qas_l1 = args.qas_l1
    attack_args.qas_nv = args.qas_nv
    attack_args.qas_a_decor = args.qas_a_decor
    attack_args.qas_rows_per_step = args.qas_rows_per_step
    attack_args.seed = args.seed

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
    _enable_qas_logging(attack_log)
    logger.info(f"\nReading gradient data from {update_file}")
    logger.info(
        "QAS objective: min ||D(Q(A S)) - Y_q||^2 + lambda R(S), "
        f"rows_per_step={attack_args.qas_rows_per_step}"
    )

    for batch in range(attack_log.batch, attack_args.n_batch):
        inp = torch.tensor(grad_data["x"][batch], device=device)
        emb = torch.tensor(
            grad_data["z"][batch] if len(grad_data["z"]) > 0 else [], device=device
        )
        labels = torch.tensor(grad_data["y"][batch], device=device)
        grads = [torch.tensor(g, device=device) for g in grad_data["grad"][batch]]
        quant_stats = grad_data["quant_stats"][batch]

        attack_log.update_batch(batch, inp, emb, grads)
        rec_gi_reord = None

        if attack_log.attack_mode == "gi":
            gi = QuantizationAwareASAttack(
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

            eval_gi = get_eval(
                inp,
                emb,
                model.model_type,
                ds_type_dict[attack_args.ds],
                attack_args.attack,
                fi=False,
            )
            pbar = get_pbar(range(gi.start_iter, attack_args.n_iter), disable=True)
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
