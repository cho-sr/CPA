#!/usr/bin/env python3
"""FP32 FedAvg baseline on Tiny ImageNet-200 using the CPA FC2 model."""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tv_datasets
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
CPA_SRC = REPO_ROOT / "cocktail_party_attack" / "src"
if str(CPA_SRC) not in sys.path:
    sys.path.insert(0, str(CPA_SRC))

from models import get_model  # noqa: E402


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 200


class TinyImageNetValAnnotations(Dataset):
    """Tiny ImageNet validation dataset for val/images + val_annotations.txt."""

    def __init__(
        self,
        val_root: Path,
        class_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.val_root = Path(val_root)
        self.images_root = self.val_root / "images"
        self.annotation_file = self.val_root / "val_annotations.txt"
        self.class_to_idx = class_to_idx
        self.transform = transform

        if not self.images_root.is_dir():
            raise FileNotFoundError(f"Validation images directory not found: {self.images_root}")
        if not self.annotation_file.is_file():
            raise FileNotFoundError(f"Validation annotation file not found: {self.annotation_file}")

        samples: List[Tuple[Path, int]] = []
        with self.annotation_file.open("r", encoding="utf-8") as f:
            for line in f:
                fields = line.strip().split("\t")
                if len(fields) < 2:
                    continue
                image_name, wnid = fields[0], fields[1]
                if wnid not in class_to_idx:
                    raise ValueError(f"Validation class {wnid} is absent from train classes")
                image_path = self.images_root / image_name
                if image_path.is_file():
                    samples.append((image_path, class_to_idx[wnid]))

        if not samples:
            raise ValueError(f"No validation samples found under {self.images_root}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FP32 weighted FedAvg baseline on Tiny ImageNet-200 with CPA FC2."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=REPO_ROOT / "cocktail_party_attack" / "datasets" / "tiny-imagenet-200",
        help="Tiny ImageNet-200 root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "fedavg_fp32" / "outputs",
        help="Directory for metrics and checkpoints.",
    )
    parser.add_argument(
        "--split_dir",
        type=Path,
        default=REPO_ROOT / "fedavg_fp32" / "client_splits",
        help="Directory where reusable IID client split files are stored.",
    )
    parser.add_argument(
        "--client_split_path",
        type=Path,
        default=None,
        help="Optional explicit client split JSON path to load/save.",
    )
    parser.add_argument(
        "--overwrite_split",
        action="store_true",
        help="Regenerate and overwrite the client split file if it already exists.",
    )
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--h_dim", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument(
        "--extra_save_rounds",
        type=str,
        default="25",
        help="Comma-separated extra checkpoint rounds in addition to 0/save_every/final.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_clients <= 0:
        raise ValueError("--num_clients must be positive")
    if args.rounds < 0:
        raise ValueError("--rounds must be non-negative")
    if args.local_epochs <= 0:
        raise ValueError("--local_epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.save_every < 0:
        raise ValueError("--save_every must be non-negative")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_fp32() -> None:
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def validate_train_dataset(dataset: tv_datasets.ImageFolder) -> None:
    if len(dataset.classes) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} train classes, found {len(dataset.classes)}")
    if not dataset.samples:
        raise ValueError("No training samples were found")


def build_datasets(
    data_root: Path,
) -> Tuple[tv_datasets.ImageFolder, Dataset, str]:
    train_transform, val_transform = build_transforms()
    train_root = data_root / "train"
    val_root = data_root / "val"

    if not train_root.is_dir():
        raise FileNotFoundError(f"Train directory not found: {train_root}")
    if not val_root.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {val_root}")

    train_dataset = tv_datasets.ImageFolder(root=str(train_root), transform=train_transform)
    validate_train_dataset(train_dataset)

    class_dirs = [
        child
        for child in val_root.iterdir()
        if child.is_dir() and child.name in train_dataset.class_to_idx
    ]
    annotation_file = val_root / "val_annotations.txt"
    images_dir = val_root / "images"

    if class_dirs:
        val_dataset: Dataset = tv_datasets.ImageFolder(root=str(val_root), transform=val_transform)
        val_mode = "imagefolder"
        if len(getattr(val_dataset, "classes", [])) != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} validation classes, found {len(val_dataset.classes)}"
            )
    elif images_dir.is_dir() and annotation_file.is_file():
        val_dataset = TinyImageNetValAnnotations(
            val_root=val_root,
            class_to_idx=train_dataset.class_to_idx,
            transform=val_transform,
        )
        val_mode = "val_annotations"
    else:
        raise ValueError(
            "Validation directory must either contain class subdirectories or "
            "val/images with val_annotations.txt"
        )

    return train_dataset, val_dataset, val_mode


def select_indices(dataset_size: int, max_samples: Optional[int], seed: int) -> List[int]:
    if max_samples is None:
        return list(range(dataset_size))
    if max_samples <= 0:
        raise ValueError("--max_train_samples/--max_test_samples must be positive when set")
    if max_samples > dataset_size:
        raise ValueError(f"Requested {max_samples} samples from a dataset of size {dataset_size}")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(dataset_size, generator=generator).tolist()[:max_samples]


def default_split_path(args: argparse.Namespace, train_size: int) -> Path:
    max_part = "full" if args.max_train_samples is None else str(args.max_train_samples)
    filename = (
        f"tiny_imagenet_iid_seed_{args.seed}_clients_{args.num_clients}_"
        f"train_{train_size}_max_{max_part}.json"
    )
    return resolve_path(args.split_dir) / filename


def make_iid_client_indices(indices: Sequence[int], num_clients: int, seed: int) -> List[List[int]]:
    if num_clients <= 0:
        raise ValueError("--num_clients must be positive")
    if len(indices) < num_clients:
        raise ValueError("Number of selected training samples must be >= num_clients")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(indices), generator=generator).tolist()
    shuffled = [int(indices[i]) for i in order]
    splits = np.array_split(np.array(shuffled, dtype=np.int64), num_clients)
    return [split.astype(int).tolist() for split in splits]


def split_sha256(split_payload: Dict) -> str:
    payload = {
        "seed": split_payload["seed"],
        "num_clients": split_payload["num_clients"],
        "train_dataset_size": split_payload["train_dataset_size"],
        "selected_train_indices": split_payload["selected_train_indices"],
        "client_indices": split_payload["client_indices"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_or_create_client_split(
    args: argparse.Namespace,
    train_dataset_size: int,
) -> Tuple[List[List[int]], Dict, Path]:
    selected_train_indices = select_indices(train_dataset_size, args.max_train_samples, args.seed)
    split_path = resolve_path(args.client_split_path) if args.client_split_path else default_split_path(args, train_dataset_size)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    if split_path.is_file() and not args.overwrite_split:
        with split_path.open("r", encoding="utf-8") as f:
            split_payload = json.load(f)
        required = {
            "seed": args.seed,
            "num_clients": args.num_clients,
            "train_dataset_size": train_dataset_size,
            "max_train_samples": args.max_train_samples,
        }
        for key, expected_value in required.items():
            actual_value = split_payload.get(key)
            if actual_value != expected_value:
                raise ValueError(
                    f"Client split mismatch for {key}: expected {expected_value}, "
                    f"found {actual_value} in {split_path}"
                )
        client_indices = split_payload["client_indices"]
    else:
        client_indices = make_iid_client_indices(
            selected_train_indices,
            num_clients=args.num_clients,
            seed=args.seed,
        )
        split_payload = {
            "version": 1,
            "dataset": "tiny_imagenet",
            "split": "iid",
            "seed": args.seed,
            "num_clients": args.num_clients,
            "train_dataset_size": train_dataset_size,
            "max_train_samples": args.max_train_samples,
            "selected_train_indices": [int(i) for i in selected_train_indices],
            "client_indices": client_indices,
            "client_num_samples": [len(indices) for indices in client_indices],
            "created_at_unix": time.time(),
        }
        split_payload["sha256"] = split_sha256(split_payload)
        save_json(split_payload, split_path)

    split_payload["sha256"] = split_sha256(split_payload)
    client_sizes = [len(indices) for indices in client_indices]
    if sum(client_sizes) != len(selected_train_indices):
        raise ValueError("Client split sizes do not sum to the selected train sample count")
    if any(size <= 0 for size in client_sizes):
        raise ValueError("Every client must receive at least one sample")
    split_payload["client_num_samples"] = client_sizes

    return client_indices, split_payload, split_path


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def build_model(h_dim: int, device: torch.device) -> nn.Module:
    model = get_model(
        model_name="fc2",
        ds="tiny_imagenet",
        h_dim=h_dim,
        dataparallel=False,
    )
    return model.to(device=device, dtype=torch.float32)


def state_dict_to_device(state_dict: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().to(device=device, dtype=torch.float32).clone() for name, tensor in state_dict.items()}


def train_one_client(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    lr: float,
    local_epochs: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], float, int]:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    total_loss = 0.0
    total_seen = 0

    for _ in range(local_epochs):
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_seen += batch_size

    if total_seen == 0:
        raise ValueError("Client loader produced no samples")
    return deepcopy(model.state_dict()), total_loss / total_seen, len(loader.dataset)


def compute_update(
    local_state: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        name: (local_state[name].detach() - global_state[name].detach()).to(dtype=torch.float32)
        for name in global_state
    }


def aggregate_updates(
    client_updates: Sequence[Dict[str, torch.Tensor]],
    client_sizes: Sequence[int],
    global_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    total_samples = int(sum(client_sizes))
    if total_samples <= 0:
        raise ValueError("Total client sample count must be positive")
    weights = [size / total_samples for size in client_sizes]
    if abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError(f"Weighted aggregation coefficients sum to {sum(weights)}")

    aggregated_state: Dict[str, torch.Tensor] = {}
    for name, global_tensor in global_state.items():
        update = torch.zeros_like(global_tensor, dtype=torch.float32)
        for weight, client_update in zip(weights, client_updates):
            update.add_(client_update[name].to(device=global_tensor.device, dtype=torch.float32), alpha=weight)
        aggregated_state[name] = (global_tensor + update).to(dtype=torch.float32)
    return aggregated_state


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_seen += batch_size
    if total_seen == 0:
        raise ValueError("Validation loader produced no samples")
    return total_loss / total_seen, total_correct / total_seen


def metric_row(
    round_idx: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    round_time: float,
    lr: float,
) -> Dict[str, float]:
    return {
        "round": round_idx,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "round_time": round_time,
        "learning_rate": lr,
    }


def write_metrics(metrics: Sequence[Dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    json_path = output_dir / "metrics.json"
    fields = ["round", "train_loss", "val_loss", "val_accuracy", "round_time", "learning_rate"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(list(metrics), f, indent=2)


def parse_extra_save_rounds(extra_save_rounds: str) -> List[int]:
    if not extra_save_rounds.strip():
        return []
    rounds = []
    for value in extra_save_rounds.split(","):
        value = value.strip()
        if value:
            rounds.append(int(value))
    return rounds


def checkpoint_rounds(rounds: int, save_every: int, extra_save_rounds: Iterable[int]) -> set:
    save_rounds = {0, rounds}
    if save_every > 0:
        save_rounds.update(range(save_every, rounds + 1, save_every))
    save_rounds.update(round_idx for round_idx in extra_save_rounds if 0 <= round_idx <= rounds)
    return save_rounds


def ensure_fp32_state(state_dict: Dict[str, torch.Tensor]) -> None:
    non_fp32 = [
        (name, str(tensor.dtype))
        for name, tensor in state_dict.items()
        if torch.is_floating_point(tensor) and tensor.dtype != torch.float32
    ]
    if non_fp32:
        raise TypeError(f"Found non-FP32 floating tensors in checkpoint: {non_fp32[:5]}")


def save_checkpoint(
    model: nn.Module,
    round_idx: int,
    config: Dict,
    metrics_for_round: Optional[Dict],
    split_payload: Dict,
    split_path: Path,
    output_dir: Path,
    latest: bool = False,
) -> None:
    state_dict = {name: tensor.detach().cpu().to(dtype=torch.float32) for name, tensor in model.state_dict().items()}
    ensure_fp32_state(state_dict)
    checkpoint = {
        "round": round_idx,
        "model_state_dict": state_dict,
        "config": config,
        "metrics": metrics_for_round,
        "seed": config["seed"],
        "client_split": {
            "path": str(split_path),
            "sha256": split_payload["sha256"],
            "split": split_payload["split"],
            "num_clients": split_payload["num_clients"],
            "client_num_samples": split_payload["client_num_samples"],
            "selected_train_indices": split_payload["selected_train_indices"],
            "client_indices": split_payload["client_indices"],
        },
        "dataset_path": config["data_root"],
    }
    round_path = output_dir / f"global_round_{round_idx:03d}.pt"
    torch.save(checkpoint, round_path)
    if latest:
        torch.save(checkpoint, output_dir / "global_latest.pt")


def make_config(args: argparse.Namespace, data_root: Path, output_dir: Path, split_path: Path, val_mode: str) -> Dict:
    return {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "split_path": str(split_path),
        "validation_mode": val_mode,
        "model": "fc2",
        "dataset": "tiny_imagenet",
        "num_classes": NUM_CLASSES,
        "input_shape": [3, 64, 64],
        "h_dim": args.h_dim,
        "num_clients": args.num_clients,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "save_every": args.save_every,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "num_workers": args.num_workers,
        "fp32": True,
        "tf32_disabled": True,
        "aggregation": "weighted_fedavg",
        "client_sampling": "all_clients_each_round",
        "optimizer": "sgd",
    }


def print_round(metric: Dict[str, float]) -> None:
    print(
        "round {round:03d} | train_loss {train_loss:.4f} | "
        "val_loss {val_loss:.4f} | val_acc {val_accuracy:.4f} | "
        "time {round_time:.1f}s".format(**metric),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    configure_fp32()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_dataset, val_dataset, val_mode = build_datasets(data_root)
    client_indices, split_payload, split_path = load_or_create_client_split(args, len(train_dataset))

    val_indices = select_indices(len(val_dataset), args.max_test_samples, args.seed + 1)
    val_subset = Subset(val_dataset, val_indices)
    val_loader = make_loader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 10_000,
        num_workers=args.num_workers,
        device=device,
    )

    client_sizes = [len(indices) for indices in client_indices]
    weights = [size / sum(client_sizes) for size in client_sizes]
    if abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError(f"Weighted aggregation coefficients sum to {sum(weights)}")

    config = make_config(args, data_root, output_dir, split_path, val_mode)
    config["train_dataset_size"] = len(train_dataset)
    config["val_dataset_size"] = len(val_dataset)
    config["used_train_samples"] = int(sum(client_sizes))
    config["used_val_samples"] = len(val_subset)
    config["client_num_samples"] = client_sizes
    config["aggregation_weights"] = weights

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    model = build_model(args.h_dim, device)
    criterion = nn.CrossEntropyLoss()
    metrics: List[Dict[str, float]] = []
    save_rounds = checkpoint_rounds(
        args.rounds,
        args.save_every,
        parse_extra_save_rounds(args.extra_save_rounds),
    )

    if 0 in save_rounds:
        save_checkpoint(
            model=model,
            round_idx=0,
            config=config,
            metrics_for_round=None,
            split_payload=split_payload,
            split_path=split_path,
            output_dir=output_dir,
            latest=False,
        )

    print(
        f"Starting FP32 FedAvg on {device} | train samples {sum(client_sizes)} | "
        f"val samples {len(val_subset)} | clients {args.num_clients}",
        flush=True,
    )
    print(f"Client split: {split_path}", flush=True)

    for round_idx in range(1, args.rounds + 1):
        round_start = time.time()
        global_state = state_dict_to_device(model.state_dict(), device)
        client_updates: List[Dict[str, torch.Tensor]] = []
        client_train_losses: List[float] = []

        for client_id, indices in enumerate(client_indices):
            client_seed = args.seed + round_idx * 1000 + client_id
            client_subset = Subset(train_dataset, indices)
            client_loader = make_loader(
                client_subset,
                batch_size=args.batch_size,
                shuffle=True,
                seed=client_seed,
                num_workers=args.num_workers,
                device=device,
            )
            local_model = build_model(args.h_dim, device)
            local_model.load_state_dict(global_state)
            local_state, train_loss, sample_count = train_one_client(
                model=local_model,
                loader=client_loader,
                criterion=criterion,
                lr=args.lr,
                local_epochs=args.local_epochs,
                device=device,
            )
            if sample_count != len(indices):
                raise ValueError(
                    f"Client {client_id} sample count mismatch: {sample_count} != {len(indices)}"
                )
            client_updates.append(compute_update(local_state, global_state))
            client_train_losses.append(train_loss)
            del local_model

        next_state = aggregate_updates(client_updates, client_sizes, global_state)
        model.load_state_dict(next_state)
        ensure_fp32_state(model.state_dict())

        weighted_train_loss = sum(
            loss * size for loss, size in zip(client_train_losses, client_sizes)
        ) / sum(client_sizes)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        round_metric = metric_row(
            round_idx=round_idx,
            train_loss=weighted_train_loss,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            round_time=time.time() - round_start,
            lr=args.lr,
        )
        metrics.append(round_metric)
        write_metrics(metrics, output_dir)
        print_round(round_metric)

        if round_idx in save_rounds:
            save_checkpoint(
                model=model,
                round_idx=round_idx,
                config=config,
                metrics_for_round=round_metric,
                split_payload=split_payload,
                split_path=split_path,
                output_dir=output_dir,
                latest=False,
            )

    last_metric = metrics[-1] if metrics else None
    save_checkpoint(
        model=model,
        round_idx=args.rounds,
        config=config,
        metrics_for_round=last_metric,
        split_payload=split_payload,
        split_path=split_path,
        output_dir=output_dir,
        latest=True,
    )
    print(f"Done. Metrics and checkpoints saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
