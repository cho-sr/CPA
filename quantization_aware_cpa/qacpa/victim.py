from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F

from .errors import GateFailure


ATTACK_LAYER = "classifier.0.weight"
ATTACK_LAYER_SHAPE = (4096, 25088)


@dataclass
class VictimBatch:
    x: torch.Tensor
    labels: torch.Tensor
    image_paths: list[str]
    A_true: torch.Tensor
    S_true: torch.Tensor
    Y: torch.Tensor
    logits: torch.Tensor
    loss: float
    per_sample_loss: torch.Tensor
    seed: int
    batch_index: int
    data_root: str
    split: str

    def as_cache_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "attack_layer": ATTACK_LAYER,
            "x": self.x.cpu(),
            "labels": self.labels.cpu(),
            "image_paths": self.image_paths,
            "A_true": self.A_true.cpu(),
            "S_true": self.S_true.cpu(),
            "Y": self.Y.cpu(),
            "logits": self.logits.cpu(),
            "loss": self.loss,
            "per_sample_loss": self.per_sample_loss.cpu(),
            "seed": self.seed,
            "batch_index": self.batch_index,
            "data_root": self.data_root,
            "split": self.split,
        }

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any]) -> "VictimBatch":
        labels = data["labels"].long()
        logits = data["logits"].float()
        per_sample_loss = data.get("per_sample_loss")
        if per_sample_loss is None:
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
        return cls(
            x=data["x"],
            labels=labels,
            image_paths=list(data.get("image_paths", [])),
            A_true=data["A_true"],
            S_true=data["S_true"],
            Y=data["Y"],
            logits=logits,
            loss=float(data["loss"]),
            per_sample_loss=per_sample_loss.float(),
            seed=int(data["seed"]),
            batch_index=int(data["batch_index"]),
            data_root=str(data["data_root"]),
            split=str(data["split"]),
        )


def victim_cache_path(
    cache_dir: str | Path,
    *,
    seed: int,
    B: int,
    batch_index: int,
    require_distinct_labels: bool = True,
) -> Path:
    suffix = "" if require_distinct_labels else "_labelsmixed"
    return Path(cache_dir) / f"victim_seed{seed}_B{B}_batch{batch_index}{suffix}.pt"


def load_or_collect_victim_batch(
    config: dict[str, Any],
    *,
    batch_index: int,
    force: bool = False,
) -> VictimBatch:
    seed = int(config["seed"])
    data_cfg = config["data"]
    B = int(data_cfg["batch_size"])
    require_distinct = bool(data_cfg.get("require_distinct_labels", True))
    cache_dir = Path(config["paths"]["cache_dir"])
    path = victim_cache_path(
        cache_dir,
        seed=seed,
        B=B,
        batch_index=batch_index,
        require_distinct_labels=require_distinct,
    )
    if path.is_file() and not force:
        return VictimBatch.from_cache_dict(torch.load(path, map_location="cpu"))

    cache_dir.mkdir(parents=True, exist_ok=True)
    batch = collect_victim_batch(
        data_root=Path(data_cfg["root"]),
        split=str(data_cfg.get("split", "val")),
        B=B,
        seed=seed,
        batch_index=batch_index,
        require_distinct_labels=require_distinct,
        microbatch_size=int(data_cfg.get("microbatch_size", B)),
        device=resolve_device(str(config.get("device", "auto"))),
    )
    torch.save(batch.as_cache_dict(), path)
    return batch


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def collect_victim_batch(
    *,
    data_root: Path,
    split: str,
    B: int,
    seed: int,
    batch_index: int,
    device: torch.device,
    require_distinct_labels: bool = True,
    microbatch_size: int | None = None,
) -> VictimBatch:
    dataset = load_imagenet_dataset(data_root, split)
    indices = select_label_indices(
        dataset,
        B=B,
        seed=seed,
        batch_index=batch_index,
        require_distinct_labels=require_distinct_labels,
    )
    samples = [dataset[int(idx)] for idx in indices]
    x = torch.stack([sample[0] for sample in samples], dim=0)
    labels = torch.tensor([int(sample[1]) for sample in samples], dtype=torch.long)
    if require_distinct_labels and len(set(int(label) for label in labels.tolist())) != B:
        raise GateFailure(f"batch labels are not distinct: {labels.tolist()}")
    paths = [_sample_path(dataset, int(idx)) for idx in indices]

    A_true, S_true, Y, logits, loss, per_sample_loss = compute_vgg16_attack_layer_factorization(
        x=x,
        labels=labels,
        device=device,
        microbatch_size=microbatch_size,
    )
    return VictimBatch(
        x=x.cpu().float(),
        labels=labels.cpu(),
        image_paths=paths,
        A_true=A_true,
        S_true=S_true,
        Y=Y,
        logits=logits,
        loss=loss,
        per_sample_loss=per_sample_loss,
        seed=seed,
        batch_index=batch_index,
        data_root=str(data_root),
        split=split,
    )


def load_imagenet_dataset(data_root: Path, split: str):
    from torchvision import datasets
    from torchvision.models import VGG16_Weights

    transform = VGG16_Weights.IMAGENET1K_V1.transforms()
    try:
        return datasets.ImageNet(str(data_root), split=split, transform=transform)
    except Exception:
        folder_root = data_root / split
        if not folder_root.is_dir():
            raise
        return datasets.ImageFolder(str(folder_root), transform=transform)


def select_distinct_label_indices(dataset, *, B: int, seed: int, batch_index: int) -> list[int]:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [int(sample[1]) for sample in dataset.samples]

    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, target in enumerate(targets):
        by_label[int(target)].append(int(idx))
    labels = [label for label, values in by_label.items() if values]
    if len(labels) < B:
        raise ValueError(f"need at least {B} distinct labels, found {len(labels)}")

    rng = random.Random(int(seed))
    labels = sorted(labels)
    rng.shuffle(labels)
    start = int(batch_index) * int(B)
    selected_labels = [labels[(start + offset) % len(labels)] for offset in range(B)]
    if len(set(selected_labels)) != B:
        raise ValueError("selected labels are not distinct")

    selected_indices: list[int] = []
    for offset, label in enumerate(selected_labels):
        candidates = by_label[label]
        choice = (int(seed) + 9973 * int(batch_index) + offset) % len(candidates)
        selected_indices.append(candidates[choice])
    return selected_indices


def select_label_indices(dataset, *, B: int, seed: int, batch_index: int, require_distinct_labels: bool) -> list[int]:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [int(sample[1]) for sample in dataset.samples]

    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, target in enumerate(targets):
        by_label[int(target)].append(int(idx))
    labels = [label for label, values in by_label.items() if values]
    if not labels:
        raise ValueError("dataset contains no labeled samples")
    if require_distinct_labels:
        return select_distinct_label_indices(dataset, B=B, seed=seed, batch_index=batch_index)

    all_indices = list(range(len(targets)))
    rng = random.Random(int(seed))
    rng.shuffle(all_indices)
    start = int(batch_index) * int(B)
    return [all_indices[(start + offset) % len(all_indices)] for offset in range(int(B))]


def _sample_path(dataset, idx: int) -> str:
    samples = getattr(dataset, "samples", None) or getattr(dataset, "imgs", None)
    if samples is None:
        return ""
    return str(samples[idx][0])


def compute_vgg16_attack_layer_factorization(
    *,
    x: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    microbatch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
    from torchvision.models import VGG16_Weights, vgg16

    weights = VGG16_Weights.IMAGENET1K_V1
    model = vgg16(weights=weights).to(device)
    disable_inplace_relu(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    attacked = model.classifier[0]
    attacked.weight.requires_grad_(True)

    B = int(x.shape[0])
    chunk_size = int(B if microbatch_size is None else microbatch_size)
    if chunk_size <= 0:
        raise ValueError(f"microbatch_size must be positive, got {microbatch_size}")
    chunk_size = min(chunk_size, B)

    z_chunks: list[torch.Tensor] = []
    a_chunks: list[torch.Tensor] = []
    logits_chunks: list[torch.Tensor] = []
    loss_chunks: list[torch.Tensor] = []
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        for start in range(0, B, chunk_size):
            stop = min(start + chunk_size, B)
            x_dev = x[start:stop].to(device)
            labels_dev = labels[start:stop].to(device)
            features = model.features(x_dev)
            pooled = model.avgpool(features)
            z = torch.flatten(pooled, 1)
            hidden = attacked(z)
            hidden.retain_grad()
            logits = model.classifier[1:](hidden)
            per_sample_loss_chunk = F.cross_entropy(logits, labels_dev, reduction="none")
            loss_chunk = per_sample_loss_chunk.sum() / float(B)
            loss_chunk.backward()
            if hidden.grad is None:
                raise GateFailure("missing attacked-layer hidden gradients")
            z_chunks.append(z.detach().cpu().float())
            a_chunks.append(hidden.grad.detach().T.cpu().float())
            logits_chunks.append(logits.detach().cpu().float())
            loss_chunks.append(per_sample_loss_chunk.detach().cpu().float())

    if attacked.weight.grad is None:
        raise GateFailure("missing attacked-layer gradients")

    S_true = torch.cat(z_chunks, dim=0).contiguous()
    if float(S_true.min()) < -1.0e-7:
        raise GateFailure(f"VGG16 source nonnegativity failed: min={float(S_true.min())}")

    A_true = torch.cat(a_chunks, dim=1).contiguous()
    Y = attacked.weight.grad.detach().cpu().float().contiguous()
    logits = torch.cat(logits_chunks, dim=0).contiguous()
    per_sample_loss = torch.cat(loss_chunks, dim=0).contiguous()
    Y_factor = A_true.matmul(S_true)
    if Y.shape != ATTACK_LAYER_SHAPE:
        raise GateFailure(f"unexpected attacked-layer shape: {tuple(Y.shape)}")
    if not torch.allclose(Y, Y_factor, rtol=1.0e-4, atol=1.0e-6):
        max_err = float((Y - Y_factor).abs().max())
        raise GateFailure(f"Y != A_true @ S_true: max_abs_err={max_err:.9g}")

    return (
        A_true,
        S_true,
        Y,
        logits.detach().cpu().float(),
        float(per_sample_loss.mean().detach().cpu()),
        per_sample_loss.detach().cpu().float(),
    )


def disable_inplace_relu(module: torch.nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, torch.nn.ReLU):
            child.inplace = False
