"""CoRA's CPA step with the upstream negentropy zero overwrite removed."""

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from gradient_inversion import CocktailPartyAttack


def source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def require_negentropy_run(run_path: Path, ne: float) -> None:
    """Reject reuse of results produced with CoRA's previous disabled objective."""
    manifest_path = Path(run_path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if (manifest.get("negentropy_enabled") is not (float(ne) > 0)
            or manifest.get("negentropy_behavior") != "upstream_logcosh_contrast_without_zero_overwrite"
            or manifest.get("cora_ica_source_sha256") != source_hash()):
        raise ValueError(
            f"Existing CoRA run has a different negentropy objective: {run_path}. "
            "Use a new output path or explicitly request --fresh_start."
        )


class CoRACocktailPartyAttack(CocktailPartyAttack):
    """Keep upstream whitening, priors and optimizers; restore its NE contrast.

    As in upstream, ``ne > 0`` enables the contrast and ``ne = 0`` disables it.
    The log-cosh expression is evaluated in an algebraically equivalent stable
    form so enabling the term does not introduce avoidable cosh overflow.
    """

    def step(self):
        loss_ne = loss_decor = loss_nv = loss_tv = loss_l1 = torch.tensor(
            0.0, device=self.device
        )
        self.opt.zero_grad()
        W_hat_norm = self.W_hat / (self.W_hat.norm(dim=-1, keepdim=True) + self.eps)
        S_hat = torch.matmul(W_hat_norm, self.X_w)
        if not torch.isfinite(S_hat).all():
            raise ValueError("CoRA S_hat has non-finite values")

        if self.ne > 0:
            absolute = (self.a * S_hat).abs()
            log_cosh = absolute + F.softplus(-2.0 * absolute) - math.log(2.0)
            log_cosh_plus_eps = torch.logaddexp(log_cosh, self.eps.log())
            loss_ne = -((log_cosh_plus_eps.mean(dim=-1) / self.a) ** 2).mean()

        S_hat = S_hat + torch.matmul(torch.matmul(W_hat_norm, self.W_w), self.X_mu)
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

        loss = (loss_ne + self.decor * loss_decor + self.tv * loss_tv
                + self.nv * loss_nv + self.l1 * loss_l1)
        if not torch.isfinite(loss):
            raise ValueError("CoRA ICA loss has non-finite values")
        loss.backward()
        self.opt.step()
        if self.sch:
            self.sch.step()
        return self.make_dict(
            ["loss", "loss_ne", "loss_decor", "loss_tv", "loss_nv", "loss_l1"],
            [loss, loss_ne, loss_decor, loss_tv, loss_nv, loss_l1],
        )
