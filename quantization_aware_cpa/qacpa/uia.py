from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .errors import GateFailure
from .victim import VictimBatch, disable_inplace_relu, load_or_collect_victim_batch, resolve_device


@dataclass
class UIAFeatureBatch:
    labels: torch.Tensor
    image_paths: list[str]
    Z: torch.Tensor
    gia_A: torch.Tensor
    gia_Y: torch.Tensor
    initial_logits: torch.Tensor
    initial_loss: float
    initial_per_sample_loss: torch.Tensor
    seed: int
    n: int
    batch_index: int
    data_root: str
    split: str

    def as_cache_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "labels": self.labels.cpu(),
            "image_paths": list(self.image_paths),
            "Z": self.Z.cpu(),
            "gia_A": self.gia_A.cpu(),
            "gia_Y": self.gia_Y.cpu(),
            "initial_logits": self.initial_logits.cpu(),
            "initial_loss": float(self.initial_loss),
            "initial_per_sample_loss": self.initial_per_sample_loss.cpu(),
            "seed": int(self.seed),
            "n": int(self.n),
            "batch_index": int(self.batch_index),
            "data_root": str(self.data_root),
            "split": str(self.split),
        }

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any]) -> "UIAFeatureBatch":
        return cls(
            labels=data["labels"].long(),
            image_paths=list(data.get("image_paths", [])),
            Z=data["Z"].float(),
            gia_A=data["gia_A"].float(),
            gia_Y=data["gia_Y"].float(),
            initial_logits=data["initial_logits"].float(),
            initial_loss=float(data["initial_loss"]),
            initial_per_sample_loss=data["initial_per_sample_loss"].float(),
            seed=int(data["seed"]),
            n=int(data["n"]),
            batch_index=int(data["batch_index"]),
            data_root=str(data["data_root"]),
            split=str(data["split"]),
        )


@dataclass
class UIAObservation:
    B_cumulative: torch.Tensor
    Y_delta_over_lr: torch.Tensor
    Y_factor: torch.Tensor
    initial_per_sample_loss: torch.Tensor
    final_logits: torch.Tensor
    final_per_sample_loss: torch.Tensor
    lr: float
    steps: int
    update_factor_rel_l2: float
    update_factor_max_abs: float


def uia_feature_cache_path(
    cache_dir: str | Path,
    *,
    seed: int,
    n: int,
    batch_index: int,
    require_distinct_labels: bool = True,
) -> Path:
    suffix = "" if require_distinct_labels else "_labelsmixed"
    return Path(cache_dir) / f"uia_features_seed{seed}_n{n}_batch{batch_index}{suffix}.pt"


def load_or_collect_uia_feature_batch(
    config: dict[str, Any],
    *,
    n: int,
    batch_index: int,
    force: bool = False,
) -> UIAFeatureBatch:
    cfg = uia_condition_config(config, n=n)
    seed = int(cfg["seed"])
    data_cfg = cfg["data"]
    require_distinct = bool(data_cfg.get("require_distinct_labels", True))
    path = uia_feature_cache_path(
        cfg["paths"]["cache_dir"],
        seed=seed,
        n=n,
        batch_index=batch_index,
        require_distinct_labels=require_distinct,
    )
    if path.is_file() and not force:
        return UIAFeatureBatch.from_cache_dict(torch.load(path, map_location="cpu"))

    victim = load_or_collect_victim_batch(cfg, batch_index=batch_index, force=force)
    feature_batch = feature_batch_from_victim(victim, n=n)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feature_batch.as_cache_dict(), path)
    return feature_batch


def uia_condition_config(config: dict[str, Any], *, n: int) -> dict[str, Any]:
    cfg = deepcopy(config)
    cfg.setdefault("data", {})["batch_size"] = int(n)
    cfg["data"]["n_batches"] = 1
    cfg["data"]["microbatch_size"] = min(32, int(n))
    cfg.setdefault("stage1", {})["n_components"] = int(n)
    return cfg


def feature_batch_from_victim(victim: VictimBatch, *, n: int) -> UIAFeatureBatch:
    return UIAFeatureBatch(
        labels=victim.labels.cpu().long(),
        image_paths=list(victim.image_paths),
        Z=victim.S_true.cpu().float(),
        gia_A=victim.A_true.cpu().float(),
        gia_Y=victim.Y.cpu().float(),
        initial_logits=victim.logits.cpu().float(),
        initial_loss=float(victim.loss),
        initial_per_sample_loss=victim.per_sample_loss.cpu().float(),
        seed=int(victim.seed),
        n=int(n),
        batch_index=int(victim.batch_index),
        data_root=str(victim.data_root),
        split=str(victim.split),
    )


def run_fc_only_fedavg_on_features(
    feature_batch: UIAFeatureBatch,
    *,
    lr: float,
    steps: int,
    device: torch.device,
) -> UIAObservation:
    return run_fc_only_fedavg_checkpoints(
        feature_batch,
        lr=lr,
        checkpoints=[int(steps)],
        device=device,
    )[int(steps)]


def run_fc_only_fedavg_checkpoints(
    feature_batch: UIAFeatureBatch,
    *,
    lr: float,
    checkpoints: list[int],
    device: torch.device,
) -> dict[int, UIAObservation]:
    checkpoint_values = sorted({int(value) for value in checkpoints})
    if not checkpoint_values:
        raise ValueError("checkpoints must not be empty")
    max_steps = int(checkpoint_values[-1])
    if any(value <= 0 for value in checkpoint_values):
        raise ValueError(f"checkpoints must be positive, got {checkpoints}")
    results: dict[int, UIAObservation] = {}
    classifier = load_vgg16_classifier(device)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(True)

    Z = feature_batch.Z.to(device=device, dtype=torch.float32)
    labels = feature_batch.labels.to(device=device)
    attacked = classifier[0]
    if not isinstance(attacked, torch.nn.Linear):
        raise GateFailure(f"unexpected VGG16 classifier[0]: {type(attacked)!r}")

    W0 = attacked.weight.detach().clone()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=float(lr), momentum=0.0, weight_decay=0.0)
    B_cumulative = torch.zeros((attacked.out_features, Z.shape[0]), device=device, dtype=torch.float32)

    with torch.enable_grad():
        for step in range(1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            hidden = attacked(Z)
            hidden.retain_grad()
            logits = classifier[1:](hidden)
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
            loss = per_sample_loss.mean()
            loss.backward()
            if hidden.grad is None:
                raise GateFailure("missing UIA attacked-layer hidden gradients")
            B_cumulative += hidden.grad.detach().T
            optimizer.step()
            if step in checkpoint_values:
                results[step] = make_uia_observation(
                    classifier=classifier,
                    attacked=attacked,
                    W0=W0,
                    Z=Z,
                    labels=labels,
                    B_cumulative=B_cumulative,
                    feature_batch=feature_batch,
                    lr=float(lr),
                    steps=step,
                )
    return results


def make_uia_observation(
    *,
    classifier: torch.nn.Sequential,
    attacked: torch.nn.Linear,
    W0: torch.Tensor,
    Z: torch.Tensor,
    labels: torch.Tensor,
    B_cumulative: torch.Tensor,
    feature_batch: UIAFeatureBatch,
    lr: float,
    steps: int,
) -> UIAObservation:
    with torch.no_grad():
        final_logits = classifier(Z)
        final_per_sample_loss = F.cross_entropy(final_logits, labels, reduction="none")

    Y_delta_over_lr = (W0 - attacked.weight.detach()) / float(lr)
    B_snapshot = B_cumulative.detach().clone()
    Y_factor = B_snapshot.matmul(Z.detach())
    diff = Y_delta_over_lr - Y_factor
    denom = Y_factor.norm().clamp_min(torch.tensor(1.0e-12, device=Y_factor.device))
    rel_l2 = float((diff.norm() / denom).detach().cpu())
    max_abs = float(diff.abs().max().detach().cpu())

    return UIAObservation(
        B_cumulative=B_snapshot.cpu().float().contiguous(),
        Y_delta_over_lr=Y_delta_over_lr.detach().cpu().float().contiguous(),
        Y_factor=Y_factor.detach().cpu().float().contiguous(),
        initial_per_sample_loss=feature_batch.initial_per_sample_loss.cpu().float(),
        final_logits=final_logits.detach().cpu().float().contiguous(),
        final_per_sample_loss=final_per_sample_loss.detach().cpu().float().contiguous(),
        lr=float(lr),
        steps=int(steps),
        update_factor_rel_l2=rel_l2,
        update_factor_max_abs=max_abs,
    )


def _unused_single_step_reference(
    feature_batch: UIAFeatureBatch,
    *,
    lr: float,
    steps: int,
    device: torch.device,
) -> UIAObservation:
    if int(steps) <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if float(lr) <= 0.0:
        raise ValueError(f"lr must be positive, got {lr}")

    classifier = load_vgg16_classifier(device)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(True)

    Z = feature_batch.Z.to(device=device, dtype=torch.float32)
    labels = feature_batch.labels.to(device=device)
    attacked = classifier[0]
    if not isinstance(attacked, torch.nn.Linear):
        raise GateFailure(f"unexpected VGG16 classifier[0]: {type(attacked)!r}")

    W0 = attacked.weight.detach().clone()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=float(lr), momentum=0.0, weight_decay=0.0)
    B_cumulative = torch.zeros((attacked.out_features, Z.shape[0]), device=device, dtype=torch.float32)

    with torch.enable_grad():
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            hidden = attacked(Z)
            hidden.retain_grad()
            logits = classifier[1:](hidden)
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
            loss = per_sample_loss.mean()
            loss.backward()
            if hidden.grad is None:
                raise GateFailure("missing UIA attacked-layer hidden gradients")
            B_cumulative += hidden.grad.detach().T
            optimizer.step()

    with torch.no_grad():
        final_logits = classifier(Z)
        final_per_sample_loss = F.cross_entropy(final_logits, labels, reduction="none")

    Y_delta_over_lr = (W0 - attacked.weight.detach()) / float(lr)
    Y_factor = B_cumulative.matmul(Z.detach())
    diff = Y_delta_over_lr - Y_factor
    denom = Y_factor.norm().clamp_min(torch.tensor(1.0e-12, device=device))
    rel_l2 = float((diff.norm() / denom).detach().cpu())
    max_abs = float(diff.abs().max().detach().cpu())

    return UIAObservation(
        B_cumulative=B_cumulative.detach().cpu().float().contiguous(),
        Y_delta_over_lr=Y_delta_over_lr.detach().cpu().float().contiguous(),
        Y_factor=Y_factor.detach().cpu().float().contiguous(),
        initial_per_sample_loss=feature_batch.initial_per_sample_loss.cpu().float(),
        final_logits=final_logits.detach().cpu().float().contiguous(),
        final_per_sample_loss=final_per_sample_loss.detach().cpu().float().contiguous(),
        lr=float(lr),
        steps=int(steps),
        update_factor_rel_l2=rel_l2,
        update_factor_max_abs=max_abs,
    )


def load_vgg16_classifier(device: torch.device) -> torch.nn.Sequential:
    from torchvision.models import VGG16_Weights, vgg16

    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
    disable_inplace_relu(model)
    classifier = model.classifier
    return classifier


def singular_values_from_row_gram(Y: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    work_device = device if device.type == "cuda" else torch.device("cpu")
    Y_work = Y.detach().to(device=work_device, dtype=torch.float32)
    gram = Y_work.matmul(Y_work.T)
    gram = (gram + gram.T) * 0.5
    eig_vals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    return torch.sqrt(eig_vals.flip(0)).detach().cpu().float()


def singular_values_from_factors(B: torch.Tensor, Z: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    work_device = device if device.type == "cuda" else torch.device("cpu")
    B_work = B.detach().to(device=work_device, dtype=torch.float32)
    Z_work = Z.detach().to(device=work_device, dtype=torch.float32)
    q_b, r_b = torch.linalg.qr(B_work, mode="reduced")
    del q_b
    q_z, r_z = torch.linalg.qr(Z_work.T, mode="reduced")
    del q_z
    small = r_b.matmul(r_z.T)
    return torch.linalg.svdvals(small).detach().cpu().float()


def numerical_rank_from_singular_values(
    singular_values: torch.Tensor,
    *,
    rel_tol: float = 1.0e-5,
    abs_tol: float = 1.0e-8,
) -> tuple[int, float]:
    if singular_values.numel() == 0:
        return 0, float(abs_tol)
    max_sv = float(singular_values.max().item())
    tol = max(float(abs_tol), float(rel_tol) * max_sv)
    rank = int((singular_values > tol).sum().item())
    return rank, tol


def mp_prediction_from_quantized(
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
    *,
    requested_rank: int,
    rounding: str,
    c_mp: float,
    device: torch.device,
) -> dict[str, Any]:
    work_device = device if device.type == "cuda" else torch.device("cpu")
    Y_work = Y_hat.detach().to(device=work_device, dtype=torch.float32)
    delta_work = delta.detach().to(device=work_device, dtype=torch.float32)
    N, d = Y_work.shape
    Y_zc = Y_work - Y_work.mean(dim=1, keepdim=True)
    cov = Y_zc.matmul(Y_zc.T) / float(d)
    if rounding == "stochastic":
        cov = cov - torch.diag(delta_work.square() / 12.0)
    cov = (cov + cov.T) * 0.5
    eig_vals = torch.linalg.eigvalsh(cov).flip(0).detach().cpu().float()
    lam_top = eig_vals[: int(requested_rank)].contiguous()
    sigma2 = float((delta_work.square() / 12.0).mean().detach().cpu())
    mp_edge = float(sigma2 * (1.0 + (float(N) / float(d)) ** 0.5) ** 2)
    threshold = float(c_mp) * mp_edge
    ratios = [
        float(value / mp_edge) if mp_edge > 0.0 else float("inf")
        for value in lam_top.tolist()
    ]
    B_eff = int(sum(value > float(c_mp) for value in ratios))
    return {
        "N": int(N),
        "d": int(d),
        "sigma2": sigma2,
        "mp_edge": mp_edge,
        "mp_threshold": threshold,
        "B_eff_pred": B_eff,
        "lam_top": [float(value) for value in lam_top.tolist()],
        "lam_over_mp_edge": ratios,
        "min_lam_over_mp_edge": float(min(ratios)) if ratios else float("nan"),
        "kept_min_lam_over_mp_edge": float(min(ratios[:B_eff])) if B_eff > 0 else float("nan"),
        "discarded_max_lam_over_mp_edge": float(max(ratios[B_eff:])) if B_eff < len(ratios) else float("nan"),
    }


def resolve_uia_device(config: dict[str, Any]) -> torch.device:
    return resolve_device(str(config.get("device", "auto")))
