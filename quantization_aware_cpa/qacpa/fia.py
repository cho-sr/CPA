from __future__ import annotations

from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class FIAMetrics:
    psnr: list[float]
    ssim: list[float]
    lpips: list[float]


@dataclass
class FIAResult:
    rec: torch.Tensor
    metrics: FIAMetrics
    log: list[dict[str, float]]
    wall_time_sec: float


class VGG16FeatureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward_z(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.features(x)
        pooled = self.model.avgpool(features)
        return torch.flatten(pooled, 1)


def run_direct_fia(
    *,
    target_z: torch.Tensor,
    x_true: torch.Tensor,
    device: torch.device,
    n_iter: int = 25000,
    log_every: int = 5000,
    lr: float = 1.0e-1,
    tv: float = 1.0e-1,
    fi: float = 1.0,
    optimizer: str = "adam",
    scheduler: str = "cosine",
    seed: int = 42,
    progress_label: str | None = None,
) -> FIAResult:
    start = time.perf_counter()
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    model = VGG16FeatureModel().to(device)
    z = target_z.detach().to(device=device, dtype=torch.float32)
    x_true = x_true.detach().to(device=device, dtype=torch.float32)
    x_hat = torch.nn.Parameter(torch.randn((z.shape[0], 3, 224, 224), device=device))

    opt_name = str(optimizer).lower()
    if opt_name == "adam":
        opt = torch.optim.Adam([x_hat], lr=float(lr))
    elif opt_name == "sgd":
        opt = torch.optim.SGD([x_hat], lr=float(lr))
    else:
        raise ValueError(f"unknown FIA optimizer {optimizer!r}")

    sch_name = str(scheduler).lower()
    if sch_name in {"none", ""}:
        sch = None
    elif sch_name == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(n_iter), last_epoch=-1)
    else:
        raise ValueError(f"unknown FIA scheduler {scheduler!r}")

    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    lower = -mean / std
    upper = (1.0 - mean) / std
    log: list[dict[str, float]] = []

    n_iter = int(n_iter)
    for iteration in range(n_iter):
        with torch.no_grad():
            x_hat.clamp_(min=lower, max=upper)
        opt.zero_grad(set_to_none=True)
        z_hat = model.forward_z(x_hat)
        loss_fi = (1.0 - F.cosine_similarity(z, z_hat, dim=-1, eps=1.0e-10)).mean()
        loss_tv = total_variation(x_hat) if float(tv) > 0.0 else torch.zeros((), device=device)
        loss = float(fi) * loss_fi + float(tv) * loss_tv
        loss.backward()
        opt.step()
        if sch is not None:
            sch.step()

        if iteration == 0 or (int(log_every) > 0 and (iteration + 1) % int(log_every) == 0) or iteration == n_iter - 1:
            entry = {
                "iter": float(iteration),
                "loss": float(loss.detach().cpu()),
                "loss_fi": float(loss_fi.detach().cpu()),
                "loss_tv": float(loss_tv.detach().cpu()),
                "lr": float(opt.param_groups[0]["lr"]),
                "time_sec": float(time.perf_counter() - start),
            }
            log.append(entry)
            if progress_label is not None:
                print(
                    "{label} iter={iteration} loss={loss:.4g} loss_fi={loss_fi:.4g} "
                    "loss_tv={loss_tv:.4g} time={time_sec:.1f}s".format(
                        label=progress_label,
                        iteration=int(iteration),
                        loss=entry["loss"],
                        loss_fi=entry["loss_fi"],
                        loss_tv=entry["loss_tv"],
                        time_sec=entry["time_sec"],
                    ),
                    flush=True,
                )

    with torch.no_grad():
        x_hat.clamp_(min=lower, max=upper)
        rec = denormalize_imagenet(x_hat.detach()).cpu().contiguous()
    metrics = evaluate_images(x_true=x_true, rec=rec.to(device), device=device)
    return FIAResult(
        rec=rec,
        metrics=metrics,
        log=log,
        wall_time_sec=float(time.perf_counter() - start),
    )


def evaluate_images(*, x_true: torch.Tensor, rec: torch.Tensor, device: torch.device) -> FIAMetrics:
    truth = denormalize_imagenet(x_true.detach().to(device=device, dtype=torch.float32))
    pred = rec.detach().to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    with torch.no_grad():
        for pred_i, truth_i in zip(pred.unsqueeze(1), truth.unsqueeze(1)):
            psnr_values.append(float(peak_signal_noise_ratio(pred_i.reshape(-1), truth_i.reshape(-1)).cpu()))
            ssim_values.append(float(structural_similarity_index_measure(pred_i, truth_i).cpu()))
            lpips_values.append(float(lpips_metric(normalize_lpips(pred_i), normalize_lpips(truth_i)).cpu()))
    return FIAMetrics(psnr=psnr_values, ssim=ssim_values, lpips=lpips_values)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return ((x * std) + mean).clamp(0.0, 1.0)


def normalize_lpips(x: torch.Tensor) -> torch.Tensor:
    return (x - 0.5) * 2.0
